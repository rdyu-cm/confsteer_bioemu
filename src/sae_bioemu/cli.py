"""Command-line entry points: run the shipped models.

confsteer sae-encode    encode real BioEmu activations with the base SAE (GPU)
confsteer steer-apply   generate a matched steered/unsteered pair with the adapter (GPU)
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

# BioEmu computes a missing sequence embedding IN THIS PROCESS, with its inlined AlphaFold2, which
# runs on JAX -- and JAX claims 75% of the GPU the moment it starts. That leaves PyTorch, which runs
# BioEmu itself, the remaining quarter: 20 GB on an 80 GB card, about 4 GB on a 16 GB one, which a
# long protein does not fit in. Allocating on demand instead changes where memory comes from, not
# any result. `setdefault`, so a user who has chosen otherwise keeps their choice. Must precede any
# import that could initialise JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch  # noqa: E402
import typer

from sae_bioemu.bioemu.extractor import BioEmuRuntimeActivationSampler, read_fasta
from sae_bioemu.core.config import ProjectConfig

app = typer.Typer(help="Run the shipped sparse autoencoder and steering adapters on BioEmu.")


#: Reference adapter shipped with the repository, relative to the checkout root.
DEFAULT_STEERING_ADAPTER = "models/rg_adapter_cold_v2.pt"


DEFAULT_BASE_SAE = "models/sae_k32_time_layer_m1.pt"


# 190 real UniRef50 records, tracked, so the BioEmu-backed demos have an input in a fresh clone.
# They take the FIRST sequence only: one generation pass is a demonstration, not a campaign.
DEMO_FASTA = "configs/steering/sweep/sweep_lr0p002_lam0.fasta"


#: Input for the steering demonstration. Deliberately NOT ``DEMO_FASTA``, for two reasons.
#:
#: * 100 of that file's 190 records are among the adapter's own 1000 training sequences, and its
#:   first record -- the one a ``[:1]`` demo would pick -- is a 15-residue peptide, whose radius of
#:   gyration reports chain extension rather than any tertiary structure worth steering.
#: * This split has ZERO overlap with the adapter's sequence set (the adapter was fitted on the
#:   *eval* 1000; this is the *train* 1000), so the demonstration is on a sequence the adapter
#:   provably never saw. That is checkable from the shipped files alone, whereas the adapter's own
#:   held-out 200 cannot be identified: its sidecar records the 700/100/200 split as counts only.
STEERING_DEMO_FASTA = "data/splits/uniref50_interpro_train_1000.fasta"

#: The atom set every distance in this package is measured over, in this order. Pinned here so the
#: CLI cannot drift from the benchmark scorers, which read the same four names in the same order.
BACKBONE_ATOMS: tuple[str, ...] = ("N", "CA", "C", "O")


def _append_reference_scores(
    path: Path, sequence_id: str, seed: int, reference: Path, scores: dict[str, float]
) -> None:
    """Append one draw's reference RMSDs, one row per arm, to a CSV beside the structures.

    These are the numbers ``--reference`` exists to produce. Printed only, they were gone once the
    terminal scrolled -- and a run with many proteins and draws is exactly the one to read later.
    """
    import csv

    new = not path.exists()
    with path.open("a", newline="") as handle:
        out = csv.writer(handle)
        if new:
            out.writerow(["sequence_id", "seed", "arm", "rmsd_angstrom", "reference"])
        for arm, value in scores.items():
            out.writerow([sequence_id, seed, arm, f"{value:.4f}", reference.name])


def _resolve_seed(requested: int | None) -> int:
    """The base seed: the one asked for, else a fresh one -- which is PRINTED, so the run can be
    reproduced afterwards.

    BioEmu's own sampler does exactly this. A fixed default looks reproducible and quietly breaks the
    obvious use: a loop over the same command returns the same structure every time.
    """
    import time

    if requested is not None:
        return int(requested)
    fresh = time.time_ns() % (2**31 - 1)
    typer.echo(
        f"seed             {fresh}   (none given; pass --seed {fresh} to reproduce this run)"
    )
    return fresh


def _resolve_device(requested: str) -> str:
    """The device to generate on: the one asked for, else CUDA when present, else the CPU.

    Says so when it falls back, because a CPU run is correct but very slow, and nothing else would
    tell the user why a command that should take a minute is taking an hour.
    """
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    typer.echo(
        "NOTE: no CUDA device is visible, so BioEmu will run on the CPU -- correct, but slow."
    )
    return "cpu"


@app.command()
def sae_encode(
    config: Path = Path("configs/examples/train_sae_paper_split.yaml"),
    checkpoint: Path = Path(DEFAULT_BASE_SAE),
    fasta: Path = Path(DEMO_FASTA),
    sequences: int = 1,
    out_dir: Path | None = None,
    seed: int | None = None,
    top_features: int = 10,
    device: str = "",
) -> None:
    """Encode REAL BioEmu activations with the base SAE. Needs a GPU and network.

    Generates BioEmu activations for real sequences at the trained layer, encodes them, and
    reports reconstruction error on the distribution the SAE was trained on -- real activations,
    not synthetic input, which has no low-dimensional structure to recover and would put the
    relative L2 near 1 whatever the model had learned.

    **THE NUMBERS DESCRIBE ONE DRAW.** They are computed on the activations of ``--sequences``
    proteins generated at ``--seed`` (default: one sequence, a fresh seed that is printed), so they
    are reproducible for a given pair and move with either. One protein at one seed is an illustration of the
    model working, not a measurement of its reconstruction quality.

    Interpreting the numbers:

    ``relative L2``     reconstruction error on real activations. On Gaussian noise this is
                        necessarily near or above 1 -- random input has no low-dimensional
                        structure. A value well below that is the whole claim of the model.
    ``active latents``  must equal ``top_k`` exactly; TopK sparsity is architectural.
    ``latents used``    distinct latents firing anywhere in the sample. Far below ``n_latents``
                        means most of the dictionary is unused **on this small sample**, which is
                        expected and is not a dead-feature measurement.

    This describes reconstruction, not interpretability. It makes no claim that any individual
    latent corresponds to a structural concept; that requires the InterPro evaluation with
    sequence-grouped folds.

    Costs a ColabFold MMseqs2 query per sequence on first use, then a BioEmu generation pass.
    """
    import torch

    from sae_bioemu.core.model import TimeConditionedTopKSAE
    from sae_bioemu.core.paths import repository_root

    root = repository_root()
    resolved = checkpoint if checkpoint.is_absolute() else root / checkpoint
    fasta_path = fasta if fasta.is_absolute() else root / fasta
    if not resolved.is_file():
        raise typer.BadParameter(f"no SAE checkpoint at {resolved}; the shipped one is in models/")
    if not fasta_path.is_file():
        raise typer.BadParameter(f"no FASTA at {fasta_path}")

    project = ProjectConfig.load(config if config.is_absolute() else root / config)
    # ``--sequences 0`` means ALL, matching steer-apply. Slicing ``[:0]`` would take NONE and then
    # report "contained no sequences", blaming the FASTA for the flag.
    parsed = read_fasta(str(fasta_path))
    records = list(parsed) if sequences <= 0 else list(parsed[:sequences])
    if not records:
        raise typer.BadParameter(f"{fasta_path} contained no sequences")

    model = TimeConditionedTopKSAE.load_checkpoint(resolved)
    model.eval()
    typer.echo(f"checkpoint       {resolved}")
    typer.echo(f"sequences        {len(records)} from {fasta_path.name}")
    typer.echo(f"layer            {project.trace.target_module}")
    typer.echo("generating BioEmu activations (first run downloads weights and queries MMseqs2)...")

    sampler = BioEmuRuntimeActivationSampler(
        # save_final_structures=False: the shipped config sets it True for a TRAINING run, which
        # writes structures through a store. Neither of these commands has one -- this one reports,
        # and steer-apply persists through SteeringStructureWriter -- and the sampler refuses
        # save_final_structures=True with structures_dir=None rather than generating and dropping
        # every structure. Turning the flag off here says the draws are consumed in-process.
        replace(project.trace, save_final_structures=False),
        cache_embeds_dir=project.training.cache_embeds_dir,
        structures_dir=None,
        batch_size=1,
        # EXPLICIT. BioEmu's model loads on the host, and the sampler moves it only if it is told
        # where to: left unset it generates on the CPU, which is how both commands ran BioEmu
        # without ever touching the GPU the README says they need.
        device=_resolve_device(device),
    )
    record = sampler.sample(records, seed=_resolve_seed(seed))
    activations, timesteps = record.activations, record.timesteps
    device = next(model.parameters()).device
    with torch.no_grad():
        output = model(activations.to(device).float(), timesteps.to(device).float())
    latents, reconstruction = output.latents, output.reconstruction

    active = latents.count_nonzero(dim=-1)
    error = float(
        (reconstruction - activations.to(device)).norm(dim=-1).mean()
        / activations.to(device).norm(dim=-1).mean()
    )
    used = int((latents.abs().sum(dim=0) > 0).sum())
    typer.echo("")
    typer.echo(
        f"tokens           {activations.shape[0]} residue-timestep rows, dim {activations.shape[1]}"
    )
    typer.echo(f"timesteps        {float(timesteps.min()):.4f} .. {float(timesteps.max()):.4f}")
    typer.echo(
        f"active latents   {int(active.min())}..{int(active.max())} (top_k={model.config.top_k})"
    )
    typer.echo(f"latents used     {used} of {model.config.n_latents} distinct, on this sample")
    typer.echo(f"relative L2      {error:.4f}   on REAL activations")

    totals = latents.abs().sum(dim=0)
    order = torch.argsort(totals, descending=True)[:top_features]
    typer.echo("")
    typer.echo(f"{'latent':>8} {'total activation':>18} {'tokens firing':>15}")
    for index in order.tolist():
        firing = int((latents[:, index] != 0).sum())
        typer.echo(f"{index:>8} {float(totals[index]):>18.3f} {firing:>15}")

    if out_dir is not None:
        _write_latents(out_dir if out_dir.is_absolute() else root / out_dir, record, latents, model)


def _write_latents(destination: Path, record: object, latents: object, model: object) -> None:
    """Write the SAE latents SPARSELY, keyed to the residue and timestep each row came from.

    These activations are the feature values every association in the paper was computed on, so
    the interesting output of this command is the latents themselves rather than the reconstruction
    error. They are written sparse because they ARE sparse -- a TopK autoencoder leaves exactly
    ``top_k`` nonzeros per row, so a dense array would be mostly zeros and hundreds of times larger.

    ``sequence_id``, ``residue_index`` and ``timestep`` travel with every row. A residue-level
    feature value that has lost which protein and which residue it came from cannot be grouped by
    sequence afterwards, and sequence is the statistical unit for everything downstream.
    """
    import numpy
    import torch

    destination.mkdir(parents=True, exist_ok=True)
    values, indices = torch.topk(latents, k=int(model.config.top_k), dim=-1)
    keep = values != 0
    # A str array, not object: an object array cannot be read back without allow_pickle=True.
    sequence_ids = numpy.asarray([str(name) for name in record.sequence_ids], dtype=str)
    payload = {
        "latent_indices": indices.detach().cpu().numpy().astype(numpy.int32),
        "latent_values": values.detach().cpu().numpy().astype(numpy.float32),
        "latent_nonzero": keep.detach().cpu().numpy(),
        "residue_index": record.residue_indices.detach().cpu().numpy().astype(numpy.int32),
        "timestep": record.timesteps.detach().cpu().numpy().astype(numpy.float32),
        "sequence_id": sequence_ids,
        "n_latents": numpy.int32(model.config.n_latents),
        "top_k": numpy.int32(model.config.top_k),
    }
    path = destination / "sae_latents.npz"
    numpy.savez_compressed(path, **payload)
    rows, k = payload["latent_values"].shape
    typer.echo("")
    typer.echo(f"latents          {path}")
    typer.echo(
        f"                 {rows} rows x {k} nonzeros of {model.config.n_latents}; each row "
        "carries its sequence_id, residue_index and timestep"
    )


def _reference_backbone(path: Path, sequence: str) -> tuple[list[int], object]:
    """Read a reference PDB's N/CA/C/O and the SEQUENCE POSITIONS it covers.

    Residues are paired by NUMBER, not by order: a reference need not cover the whole chain -- the
    kinase metric is an activation loop, the cryptic one a pocket -- and ``export_system.py``
    numbers what it writes by sequence position for exactly this. Refuses anything that would pair
    the wrong residues and still print a number: more than one chain, a residue number outside the
    sequence or not positive, a repeat, a residue whose amino acid disagrees with the FASTA at that
    position (which is what catches an author-numbered file), or a residue missing a backbone atom.
    """
    import mdtraj
    import numpy

    frame = mdtraj.load(str(path))
    wanted: list[int] = []
    rows: list[list[list[float]]] = []
    chains = set()
    for residue in frame.topology.residues:
        atoms = {atom.name: atom.index for atom in residue.atoms}
        if "CA" not in atoms:
            continue
        chains.add(residue.chain.index)
        missing = [name for name in BACKBONE_ATOMS if name not in atoms]
        if missing:
            raise typer.BadParameter(
                f"{path.name} residue {residue.resSeq} has no {missing}; this metric is N/CA/C/O "
                "over the residues the reference covers, so a partial residue cannot be scored"
            )
        index = residue.resSeq - 1
        if index < 0 or index >= len(sequence):
            raise typer.BadParameter(
                f"{path.name} numbers a residue {residue.resSeq}, outside 1..{len(sequence)} of "
                "the FASTA. Renumber it by sequence position, or write it with export_system.py"
            )
        if residue.code and residue.code != sequence[index]:
            raise typer.BadParameter(
                f"{path.name} has {residue.code} at position {residue.resSeq} where the FASTA has "
                f"{sequence[index]}. The file is numbered differently from the sequence, so the "
                "comparison would pair unrelated residues"
            )
        wanted.append(index)
        rows.append(
            [[float(v) for v in frame.xyz[0][atoms[name]] * 10.0] for name in BACKBONE_ATOMS]
        )
    if len(chains) != 1:
        raise typer.BadParameter(
            f"{path.name} has protein in {len(chains)} chains; give a single-chain reference, "
            "since residue numbers cannot say which chain the FASTA is"
        )
    if len(set(wanted)) != len(wanted):
        raise typer.BadParameter(f"{path.name} repeats a residue number; each must appear once")
    return wanted, numpy.asarray(rows, dtype=float)


def _resolve_references(reference: Path | None, root: Path, chosen: list) -> dict[str, Path | None]:
    """Map each selected protein to its reference structure, or ``None`` when it has none."""
    # ONE REFERENCE PER PROTEIN. A corpus cannot share one structure, so --reference is either a
    # single file -- only meaningful for a single protein -- or a DIRECTORY holding one file per
    # sequence id, which is what export_system.py writes. A protein with no reference is steered
    # and reported without a distance rather than skipped or silently matched to another's file.
    references: dict[str, Path | None] = dict.fromkeys((item.sequence_id for item in chosen), None)
    if reference is not None:
        located = reference if reference.is_absolute() else root / reference
        if located.is_dir():
            for item in chosen:
                # The id must end at a separator or at the extension. A bare prefix glob lets one
                # protein claim another's file whenever one id is a prefix of the other -- `P0205`
                # would match `P02051_apo.pdb` -- and the refusal below would then blame the user
                # for an ambiguity the matching invented.
                found = sorted(
                    path
                    for pattern in (f"{item.sequence_id}*.pdb", f"{item.sequence_id}*.cif")
                    for path in located.glob(pattern)
                    if path.stem == item.sequence_id
                    or path.stem[len(item.sequence_id) :].startswith(("_", "-", "."))
                )
                if len(found) > 1:
                    # export_system.py writes BOTH conformers of a pair, and which one an arm was
                    # steered toward is not recoverable from the file name: for domain motion the
                    # deposited order disagrees with the trained one in most systems. Picking the
                    # first would score most proteins against the conformer the arm was not aimed
                    # at and still print a plausible distance.
                    raise typer.BadParameter(
                        f"{located.name} has {len(found)} structures for {item.sequence_id} "
                        f"({', '.join(path.name for path in found[:4])}). Which one this arm was "
                        "steered toward cannot be read from the name, so keep ONE per protein in "
                        "the directory -- the conformer this adapter targets"
                    )
                references[item.sequence_id] = found[0] if found else None
            missing = sorted(name for name, path in references.items() if path is None)
            if len(missing) == len(chosen):
                raise typer.BadParameter(
                    f"{located} holds no structure named for any of {sorted(references)[:4]}. "
                    "A reference directory is matched by sequence id, as export_system.py writes it"
                )
            if missing:
                typer.echo(f"NOTE: no reference in {located.name} for {missing}; those are steered")
        elif len(chosen) > 1:
            raise typer.BadParameter(
                f"--reference {located.name} is one structure but {len(chosen)} proteins were "
                "selected. Give a DIRECTORY with one file per sequence id, or select one protein "
                "with --sequence-id"
            )
        else:
            references[chosen[0].sequence_id] = located
    return references


def _score_against_reference(
    reference: Path | None,
    root: Path,
    sequence: str,
    unsteered: object | None,
    steered: object,
) -> dict[str, float]:
    """Report each arm's N/CA/C/O RMSD to a target conformer, over the residues it covers.

    THE SAME ATOM SET AND THE SAME SUPERPOSITION THE BENCHMARKS USE, so a number here and a number
    from ``run_benchmark.py`` mean the same thing. Both arms are backbone-complete: the steered
    structure carries its frames on its batch and the control is generated through
    ``generate_unsteered_frames``, whose positions are byte-identical to the plain call's.

    ``unsteered`` may be ``None`` (``--no-control``), in which case the steered distance is
    reported alone and says so -- it cannot show what the intervention did.
    """
    if reference is None:
        return {}
    import numpy
    import torch

    from sae_bioemu.steering.aligned_rmsd import aligned_rmsd

    path = reference if reference.is_absolute() else root / reference
    if not path.is_file():
        raise typer.BadParameter(f"no reference structure at {path}")
    wanted, target_array = _reference_backbone(path, sequence)
    target = torch.as_tensor(target_array, dtype=torch.float64).reshape(-1, 3)

    typer.echo("")
    span = (
        "the whole chain"
        if len(wanted) == len(sequence)
        else f"{len(wanted)} of {len(sequence)} residues"
    )
    typer.echo(f"{'arm':>12} {f'N/CA/C/O RMSD to {path.name} (A)':>40}")
    typer.echo(f"{'':>12} {f'over {span}':>40}")
    values: dict[str, float] = {}
    pairs = [("steered", steered)]
    if unsteered is not None:
        pairs.insert(0, ("unsteered", unsteered))
    for label, arm in pairs:
        array = numpy.asarray(arm, dtype=float)
        if array.shape != (len(sequence), 4, 3):
            raise typer.BadParameter(
                f"the {label} arm has backbone of shape {array.shape}; expected "
                f"({len(sequence)}, 4, 3) N/CA/C/O atoms for this sequence"
            )
        selected = torch.as_tensor(array[wanted], dtype=torch.float64).reshape(-1, 3)
        values[label] = float(aligned_rmsd(selected, target).reshape(()))
        typer.echo(f"{label:>12} {values[label]:>40.3f}")
    if "unsteered" in values:
        delta = values["steered"] - values["unsteered"]
        typer.echo(f"{'change':>12} {delta:>+40.3f}")
        typer.echo(
            "negative = the intervention moved this draw CLOSER to the reference. One draw: an "
            "illustration, not an effect size."
        )
    else:
        typer.echo(
            "no unsteered arm was generated (--no-control), so this distance stands alone: "
            "whether the adapter moved the draw toward the reference cannot be read from it."
        )
    return values


@app.command()
def steer_apply(
    adapter: str = "rg_adapter_cold_v2",
    fasta: Path = Path(STEERING_DEMO_FASTA),
    sequence_id: str = "",
    sequences: int = 1,
    draws: int = 1,
    level: float | None = None,
    seed: int | None = None,
    reference: Path | None = None,
    control: bool = True,
    out_dir: Path = Path("outputs/steering/steer_apply"),
    config: Path = Path("configs/examples/train_sae_paper_split.yaml"),
    device: str = "",
) -> None:
    """Steer proteins with any shipped adapter, each draw beside its matched unsteered control.

    Works for all seven adapters -- the radius-of-gyration one and the six conformational arms --
    because the mechanism is the same: the adapter adds a residual at its own injection sites, and
    ``--level`` is the dose. The site comes from the adapter's sidecar, so a single-site and an
    eight-site arm are driven identically and neither is named here.

    ``--fasta`` is any FASTA, so this is the path for steering YOUR protein rather than a benchmark
    system. Within each draw both arms share one seed, which is what makes the pair matched and
    the difference attributable to the intervention rather than to sampling.

    ``--level`` defaults to the level the adapter was TRAINED at. The six conformational arms are
    single-direction -- each learned one level, and the opposite is a dose it never saw -- so the
    default is that level and asking for another says so.

    ``--reference`` scores each arm's N/CA/C/O RMSD to a target conformer and writes it to
    ``reference_rmsd.csv`` beside the structures. Radius of gyration is reported for both arms
    because it needs no reference.

    One sequence at one level and one seed is an anecdote, not an effect size: the sign may not
    match on a single draw, and nothing here is corrected for sampling variation.
    """
    import torch

    from sae_bioemu.bioemu.steering import steering_site_from_record
    from sae_bioemu.core.paths import repository_root
    from sae_bioemu.steering.adapter_io import load_adapter
    from sae_bioemu.steering.evaluation_structures import (
        ADAPTER_OFF_ARM,
        ADAPTER_ON_ARM,
        SteeringStructureWriter,
    )
    from sae_bioemu.steering.rg_calibration import radius_of_gyration_angstrom

    root = repository_root()
    # ``--adapter`` takes a shipped NAME (``kinase_uncond_p1``) or a path. A name is what a user
    # has; a path is what the repository layout happens to be, and they should not need to know it.
    candidate = Path(adapter)
    resolved = candidate if candidate.suffix == ".pt" else root / "models" / f"{adapter}.pt"
    if not resolved.is_absolute():
        resolved = root / resolved
    fasta_path = fasta if fasta.is_absolute() else root / fasta
    if not resolved.is_file():
        raise typer.BadParameter(f"no adapter at {resolved}")
    sidecar = resolved.with_suffix(".json")
    if not sidecar.is_file():
        raise typer.BadParameter(f"no provenance sidecar at {sidecar}; the site is read from it")

    adapter, payload = load_adapter(resolved)
    site = steering_site_from_record(
        json.loads(sidecar.read_text())["training"]["intervention_site"]
    )

    project = ProjectConfig.load(config if config.is_absolute() else root / config)
    records = read_fasta(str(fasta_path))
    if not records:
        raise typer.BadParameter(f"{fasta_path} contained no sequences")
    if sequence_id:
        chosen = [item for item in records if item.sequence_id == sequence_id]
        if not chosen:
            raise typer.BadParameter(f"{fasta_path} has no sequence named {sequence_id!r}")
    elif sequences <= 0:
        chosen = list(records)
    else:
        chosen = list(records[:sequences])

    references = _resolve_references(reference, root, chosen)

    typer.echo(f"adapter          {resolved}")
    typer.echo(f"proteins         {len(chosen)} of {len(records)} in {fasta_path.name}")
    typer.echo(f"site             {site.module_path} score call {site.score_calls}")
    # DEFAULT THE LEVEL TO WHAT THIS ADAPTER WAS TRAINED AT. The six benchmark adapters are
    # single-direction: each learned one level, and asking for the other is a dose it never saw.
    # The level-conditioned Rg adapter has a range, so it keeps the +1.0 the documentation uses.
    trained = payload["adapter_config"].get("target_levels")
    if level is None:
        level = float(trained[0]) if trained and len(trained) == 1 else 1.0
    # THE SIX CONFORMATIONAL ARMS CANNOT SEE THE LEVEL AT ALL. They are level_conditioning='none',
    # where the level "reaches nothing" -- it is recorded so the run says which direction the arm
    # was trained for, and the network never receives it. Passing another value would return a
    # structure identical to the trained level's, so a dose sweep would look like a dead adapter
    # rather than a misuse. Refused rather than noted, because the output gives no hint.
    conditioning = payload["adapter_config"]["adapter"]["attributes"].get("level_conditioning")
    if trained and len(trained) == 1 and float(level) != float(trained[0]):
        if conditioning == "none":
            raise typer.BadParameter(
                f"{resolved.stem} is a single-direction adapter: it was trained at level "
                f"{float(trained[0]):+.1f} and the level is not an input to it, so --level "
                f"{float(level):+.1f} would produce exactly the same structure. Drop --level, or "
                "use rg_adapter_cold_v2, whose level does vary the output"
            )
        typer.echo(
            f"NOTE: {resolved.stem} was trained only at level {float(trained[0]):+.1f}; "
            f"{float(level):+.1f} is a dose it never saw."
        )
    typer.echo(f"adapter          {resolved.stem}")
    typer.echo(
        f"level            {level:+.2f}   {draws} draw(s) per protein"
        + ("   shared by both arms" if control else "   steered arm only")
    )
    typer.echo(
        "generating both arms through BioEmu..."
        if control
        else "generating the steered arm through BioEmu (--no-control)..."
    )

    sampler = BioEmuRuntimeActivationSampler(
        # save_final_structures=False: the shipped config sets it True for a TRAINING run, which
        # writes structures through a store. Neither of these commands has one -- this one reports,
        # and steer-apply persists through SteeringStructureWriter -- and the sampler refuses
        # save_final_structures=True with structures_dir=None rather than generating and dropping
        # every structure. Turning the flag off here says the draws are consumed in-process.
        replace(project.trace, save_final_structures=False),
        cache_embeds_dir=project.training.cache_embeds_dir,
        structures_dir=None,
        batch_size=1,
        # EXPLICIT. BioEmu's model loads on the host, and the sampler moves it only if it is told
        # where to: left unset it generates on the CPU, which is how both commands ran BioEmu
        # without ever touching the GPU the README says they need.
        device=_resolve_device(device),
    )
    if draws < 1:
        raise typer.BadParameter("--draws must be at least 1")
    base_seed = _resolve_seed(seed)
    draw_seeds = [base_seed + index for index in range(draws)]
    for position, record in enumerate(chosen, start=1):
        reference_for_record = references[record.sequence_id]
        typer.echo("")
        typer.echo(
            f"[{position}/{len(chosen)}] {record.sequence_id} "
            f"({len(record.sequence)} residues)"
            + (f"  reference {reference_for_record.name}" if reference_for_record else "")
        )
        # DRAWS. Each draw has its own seed, base + index, and within a draw the steered and the
        # control structure SHARE it -- that is what keeps every pair matched while the draws differ.
        for draw_number, seed in enumerate(draw_seeds, start=1):
            if draws > 1:
                typer.echo(f"  draw {draw_number}/{draws}  seed {seed}")
            reference_scores: dict[str, float] = {}
            residue_mask = torch.ones(1, len(record.sequence), dtype=torch.bool)
            target = torch.tensor([float(level)])

            # BOTH ARMS BACKBONE-COMPLETE. generate_unsteered returns a bare C-alpha tensor; the _frames
            # variant returns byte-identical positions PLUS the per-residue frames, which is what makes
            # N/CA/C/O -- the atom set the benchmarks score on -- available for the control as well.
            unsteered_frames = (
                sampler.generate_unsteered_frames(record, seed=seed) if control else None
            )
            unsteered = unsteered_frames[0] if unsteered_frames is not None else None
            steered = sampler.steer(
                record,
                seed=seed,
                site=site,
                adapter=adapter,
                target=target,
                residue_mask=residue_mask,
            )
            # BioEmuSteeredSample names the structure `coordinates`, not `positions`. Read the attribute
            # directly rather than through a getattr default: a default silently hands the whole dataclass
            # to the geometry code, which then fails somewhere unrelated with no mention of the real cause.
            positions = steered.coordinates

            def rg(label: str, tensor: object) -> float:
                """Radius of gyration in angstrom, from whatever container the arm returned.

                ``radius_of_gyration_angstrom`` accepts ``(residues, 3)`` or ``(structures, residues, 3)``.
                The two arms do not necessarily hand back the same rank -- the steered arm returns a
                differentiable batch -- so normalise here, and name the shape in the error rather than
                letting a bare "positions must be ..." surface with no clue which arm produced it.
                """
                import numpy as np

                array = (
                    tensor.detach().cpu().numpy()
                    if hasattr(tensor, "detach")
                    else np.asarray(tensor)
                )
                # Drop leading singleton axes only. Never collapse a real structure or residue axis: a
                # wrong reshape would silently return the radius of a different point set.
                while array.ndim > 3 and array.shape[0] == 1:
                    array = array[0]
                if array.ndim not in (2, 3) or array.shape[-1] != 3:
                    raise typer.BadParameter(
                        f"{label} positions have shape {array.shape}, which is neither (residues, 3) nor "
                        "(structures, residues, 3); cannot compute a radius of gyration from it"
                    )
                return float(radius_of_gyration_angstrom(array)[0])

            # Name the shapes before consuming them: the two arms are produced by different code paths and
            # a rank mismatch here is a shape bug, not a scientific result.
            arms: list[tuple[str, object]] = [("steered", positions)]
            if unsteered is not None:
                arms.insert(0, ("unsteered", unsteered))
            for label, value in arms:
                shape = getattr(value, "shape", None)
                typer.echo(
                    f"  {label} positions shape: {tuple(shape) if shape is not None else type(value)}"
                )
            rg_on = rg("steered", positions)
            rg_off = rg("unsteered", unsteered) if unsteered is not None else None
            if reference_for_record is not None:
                from sae_bioemu.bioemu.steering_properties import padded_node_orientations
                from sae_bioemu.cryptic_rmsd.frames import batched_backbone_atoms

                members = [record]
                steered_backbone = (
                    batched_backbone_atoms(
                        steered.coordinates.reshape(1, len(record.sequence), 3),
                        padded_node_orientations(steered, members),
                        [record.sequence],
                    )[0]
                    .detach()
                    .cpu()
                )
                control_backbone = None
                if unsteered_frames is not None:
                    control_backbone = (
                        batched_backbone_atoms(
                            unsteered_frames[0].reshape(1, len(record.sequence), 3),
                            unsteered_frames[1].reshape(1, len(record.sequence), 3, 3),
                            [record.sequence],
                        )[0]
                        .detach()
                        .cpu()
                    )
                reference_scores = _score_against_reference(
                    reference_for_record, root, record.sequence, control_backbone, steered_backbone
                )
            typer.echo("")
            typer.echo(f"{'arm':>12} {'radius of gyration (A)':>24}")
            if rg_off is not None:
                typer.echo(f"{'unsteered':>12} {rg_off:>24.3f}")
            typer.echo(f"{'steered':>12} {rg_on:>24.3f}")
            if rg_off is not None:
                typer.echo(f"{'difference':>12} {rg_on - rg_off:>+24.3f}")
            else:
                typer.echo(
                    "no unsteered arm was generated (--no-control), so there is no difference to report: "
                    "this radius of gyration is a number about one draw, not an effect of the adapter"
                )
            fraction = float(steered.residual_fraction.detach().mean())
            typer.echo(f"{'budget used':>12} {fraction:>24.4f}   (fraction of rho_max)")

            def residue_positions(label: str, tensor: object) -> torch.Tensor:
                """Return one arm's coordinates as a ``[residues, 3]`` CPU tensor.

                The arms are produced by different code paths and need not share a rank, so drop leading
                singleton axes only. Collapsing a real structure or residue axis would silently persist
                the coordinates of a different point set under this arm's name.
                """
                array = (
                    tensor.detach().cpu() if hasattr(tensor, "detach") else torch.as_tensor(tensor)
                )
                while array.ndim > 2 and array.shape[0] == 1:
                    array = array[0]
                if array.ndim != 2 or array.shape[-1] != 3:
                    raise typer.BadParameter(
                        f"{label} positions have shape {tuple(array.shape)}, which is not (residues, 3)"
                    )
                return array

            # Persist both arms through the same store the gated evaluation writes, so a demo structure and
            # an evaluation structure are the same kind of object. Deliberately no PDB: only coordinates are
            # produced, not node orientations, so these are not backbone-complete, and rendering the steered
            # arm alone would present two arms that are not comparable artifacts. That decision, and the
            # nanometre/angstrom bookkeeping, live in sae_bioemu.steering.evaluation_structures.
            destination = out_dir if out_dir.is_absolute() else root / out_dir
            writer = SteeringStructureWriter(destination)
            # The writer is keyed on (sequence_id, seed, arm), so reusing a seed REPLACES that draw
            # rather than adding one. Without --seed every run draws fresh seeds, so this only fires
            # when a seed is given twice -- and then it is said, not discovered from a flat index.
            written_arms = [(ADAPTER_ON_ARM, positions, rg_on)]
            if unsteered is not None:
                written_arms.insert(0, (ADAPTER_OFF_ARM, unsteered, rg_off))
            already = {(record.sequence_id, seed, arm) for arm, _, _ in written_arms} & set(
                writer._keys  # noqa: SLF001 - the writer exposes no public membership test
            )
            typer.echo("")
            for arm, tensor, value in written_arms:
                entry = writer.write(
                    sequence_id=record.sequence_id,
                    sequence=record.sequence,
                    seed=seed,
                    arm=arm,
                    unit_index=0,
                    visit=0,
                    level=float(level),
                    target=float(level),
                    # The value IS the radius of gyration -- labelling it with the adapter's own
                    # property (e.g. kinase_conformer_rmsd) put an Rg under an RMSD's name on disk.
                    property_name="radius_of_gyration",
                    property_value=value,
                    positions_nanometers=residue_positions(arm, tensor),
                )
                typer.echo(f"{arm:>12}     {destination / entry.npz_path}")
            typer.echo(f"{'index':>12}     {writer.index_path}")
            if reference_scores:
                _append_reference_scores(
                    destination / "reference_rmsd.csv",
                    record.sequence_id,
                    seed,
                    reference_for_record,
                    reference_scores,
                )
            if already:
                typer.echo("")
                typer.echo(
                    f"NOTE: seed {seed} for this sequence was already in the index, so this draw "
                    "replaced an identical one. Runs without --seed draw fresh seeds and accumulate."
                )
            typer.echo("")

    typer.echo("")
    typer.echo("Each draw is one sample. This demonstrates the mechanism; it is NOT the published")
    typer.echo("result -- scripts/run_benchmark.py reproduces the benchmarks.")


if __name__ == "__main__":  # pragma: no cover - exercised as `python -m sae_bioemu.cli`
    # WITHOUT THIS, `python -m sae_bioemu.cli steer-apply ...` imports the module, defines the
    # commands, runs nothing and EXITS 0. Not a usability wart: a wrapper that checks the exit code
    # reads that silence as success, which is how a GPU job reported four generations it never ran.
    app()
