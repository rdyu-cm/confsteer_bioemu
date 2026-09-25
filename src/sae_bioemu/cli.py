"""Command-line entry points: run the shipped models.

confsteer sae-encode    encode real BioEmu activations with the base SAE (GPU)
confsteer steer-apply   generate a matched steered/unsteered pair with the adapter (GPU)
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import replace
from pathlib import Path

# BioEmu computes a missing sequence embedding IN THIS PROCESS, with its inlined AlphaFold2, which
# runs on JAX -- and JAX claims 75% of the GPU the moment it starts. That leaves PyTorch, which runs
# BioEmu itself, the remaining quarter: 20 GB on an 80 GB card, about 4 GB on a 16 GB one, which a
# long protein does not fit in. Allocating on demand instead changes where memory comes from, not
# any result. `setdefault`, so a user who has chosen otherwise keeps their choice. Must precede any
# import that could initialise JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# TensorFlow, imported by that AlphaFold2, prints INFO banners (oneDNN, CPU features) on every run.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from sae_bioemu.runtime_env import use_importable_matplotlib_backend  # noqa: E402

use_importable_matplotlib_backend()

# BioEmu asks for deterministic algorithms once per generation, and PyTorch answers with the same
# CuBLAS warning every time -- hundreds of identical paragraphs in a run with several draws, none of
# which the user can act on. Silenced here rather than by setting CUBLAS_WORKSPACE_CONFIG, which
# would change how the kernels are chosen and so could change the numbers.
warnings.filterwarnings("ignore", message="Deterministic behavior was enabled")

import torch  # noqa: E402
import typer  # noqa: E402

from sae_bioemu.bioemu.extractor import BioEmuRuntimeActivationSampler, read_fasta  # noqa: E402
from sae_bioemu.core.config import ProjectConfig  # noqa: E402

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
    typer.echo(f"seed {fresh} (pass --seed {fresh} to reproduce)")
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
    device: str = "",
) -> None:
    """Encode REAL BioEmu activations with the base SAE. Needs a GPU and network.

    Generates BioEmu activations for ``--sequences`` proteins at the trained layer, encodes them,
    writes the sparse latents to ``--out-dir``, and reports reconstruction error on the
    distribution the SAE was trained on -- real activations, not synthetic input, which has no
    low-dimensional structure to recover and would put the relative L2 near 1 whatever the model
    had learned.

    The error describes ONE draw, at ``--seed`` (default: a fresh seed, which is printed), so it
    moves with either. It describes reconstruction, not interpretability: it makes no claim that
    any individual latent corresponds to a structural concept.

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
    base_seed = _resolve_seed(seed)
    plural = "" if len(records) == 1 else "s"
    typer.echo(f"encoding {len(records)} protein{plural} with BioEmu...")

    sampler = BioEmuRuntimeActivationSampler(
        # save_final_structures=False: the shipped config sets it True for a TRAINING run, which
        # writes structures through a store. Neither of these commands has one -- this one reports
        # and writes latents -- and the sampler refuses save_final_structures=True with
        # structures_dir=None rather than generating and dropping every structure. Turning the flag
        # off here says the draws are consumed in-process.
        replace(project.trace, save_final_structures=False),
        cache_embeds_dir=project.training.cache_embeds_dir,
        structures_dir=None,
        batch_size=1,
        # EXPLICIT. BioEmu's model loads on the host, and the sampler moves it only if it is told
        # where to: left unset it generates on the CPU, which is how both commands ran BioEmu
        # without ever touching the GPU the README says they need.
        device=_resolve_device(device),
    )
    record = sampler.sample(records, seed=base_seed)
    activations, timesteps = record.activations, record.timesteps
    device = next(model.parameters()).device
    with torch.no_grad():
        output = model(activations.to(device).float(), timesteps.to(device).float())
    latents, reconstruction = output.latents, output.reconstruction

    error = float(
        (reconstruction - activations.to(device)).norm(dim=-1).mean()
        / activations.to(device).norm(dim=-1).mean()
    )
    counts = (
        f"{len(records)} protein{plural}, {activations.shape[0]} residue-timestep rows, "
        f"top {model.config.top_k} of {model.config.n_latents} latents per row"
    )
    if out_dir is None:
        typer.echo(counts)
    else:
        path = _write_latents(
            out_dir if out_dir.is_absolute() else root / out_dir, record, latents, model
        )
        typer.echo(f"wrote {out_dir / path.name}: {counts}")
    typer.echo(f"reconstruction error (relative L2): {error:.3f}")


def _write_latents(destination: Path, record: object, latents: object, model: object) -> Path:
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
    return path


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
                shown = ", ".join(missing[:3]) + (", ..." if len(missing) > 3 else "")
                typer.echo(
                    f"NOTE: {len(missing)} of {len(chosen)} proteins have no reference in "
                    f"{located.name} ({shown}); they are steered without one"
                )
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
    """Return each arm's N/CA/C/O RMSD to a target conformer, over the residues it covers.

    THE SAME ATOM SET AND THE SAME SUPERPOSITION THE BENCHMARKS USE, so a number here and a number
    from ``run_benchmark.py`` mean the same thing. Both arms are backbone-complete: the steered
    structure carries its frames on its batch and the control is generated through
    ``generate_unsteered_frames``, whose positions are byte-identical to the plain call's.

    ``unsteered`` may be ``None`` (``--no-control``), in which case only the steered distance is
    returned -- alone it cannot show what the intervention did.
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
    return values


def _value_columns(reference: bool, control: bool) -> str:
    """The one line naming what the per-draw ``off``/``on`` numbers are."""
    what = "N/CA/C/O RMSD (A) to the reference" if reference else "radius of gyration (A)"
    arms = "off = unsteered, on = steered" if control else "on = steered"
    return f"{what}: {arms}"


def _structure_filename(sequence_id: str, seed: int, arm: str) -> str:
    """The file name one arm of one draw is written under.

    The sequence id comes from a FASTA header, which may hold a separator or whitespace; either
    would put the file somewhere other than the output directory, or split one draw's name across
    two words. Replaced rather than rejected, so an ordinary FASTA is not refused for its headers.
    """
    safe = "".join(
        "_" if character.isspace() or character in (os.sep, "/") else character
        for character in sequence_id
    )
    return f"{safe}_seed{seed}_{arm}.pdb"


def _write_backbone_pdb(
    path: Path, positions_nanometers: object, node_orientations: object, sequence: str
) -> None:
    """Write one arm as a backbone PDB, the way the runtime structure store writes its own.

    BioEmu's coordinates are nanometres and its PDB writer takes Angstrom, so the positions are
    scaled and centred exactly as ``extractor.py`` does before handing them over -- a factor of ten
    here would produce a file that opens, renders, and measures nothing. Written to a temporary
    name and moved into place, so an interrupted run leaves no half-written structure behind.
    """
    import torch
    from bioemu.convert_chemgraph import _write_pdb

    residues = len(sequence)
    pos = torch.as_tensor(positions_nanometers).detach().cpu().reshape(residues, 3).float() * 10.0
    pos = pos - pos.mean(dim=0, keepdim=True)
    orientations = torch.as_tensor(node_orientations).detach().cpu().reshape(residues, 3, 3).float()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _write_pdb(pos=pos, node_orientations=orientations, sequence=sequence, filename=temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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

    Both arms are written to ``--out-dir`` as backbone PDBs, ``<id>_seed<seed>_on.pdb`` and
    ``_off.pdb``. ``--reference`` additionally scores each arm's N/CA/C/O RMSD to a target
    conformer, printed per draw and appended to ``reference_rmsd.csv`` beside the structures.

    One sequence at one level and one seed is an anecdote, not an effect size: the sign may not
    match on a single draw, and nothing here is corrected for sampling variation.
    """
    import torch

    from sae_bioemu.bioemu.steering import steering_site_from_record
    from sae_bioemu.bioemu.steering_properties import padded_node_orientations
    from sae_bioemu.core.paths import repository_root
    from sae_bioemu.cryptic_rmsd.frames import batched_backbone_atoms
    from sae_bioemu.steering.adapter_io import load_adapter
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

    if draws < 1:
        raise typer.BadParameter("--draws must be at least 1")

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

    destination = out_dir if out_dir.is_absolute() else root / out_dir
    destination.mkdir(parents=True, exist_ok=True)
    typer.echo(
        f"{resolved.stem}: {len(chosen)} protein{'' if len(chosen) == 1 else 's'} x "
        f"{draws} draw{'' if draws == 1 else 's'} -> {out_dir}"
    )
    references = _resolve_references(reference, root, chosen)
    # WHICH PROPERTY THIS ADAPTER STEERS, from the checkpoint rather than from its file name: only
    # the radius-of-gyration adapter has a value worth printing without a reference, and a renamed
    # copy of any other one would otherwise be reported as though it steered the radius.
    steers_rg = payload["adapter_config"].get("property_name") == "radius_of_gyration"
    base_seed = _resolve_seed(seed)
    if any(path is not None for path in references.values()):
        typer.echo(_value_columns(reference=True, control=control))
    elif steers_rg:
        typer.echo(_value_columns(reference=False, control=control))

    sampler = BioEmuRuntimeActivationSampler(
        # save_final_structures=False: the shipped config sets it True for a TRAINING run, which
        # writes structures through a store. Neither of these commands has one -- this one writes
        # its own PDBs -- and the sampler refuses save_final_structures=True with
        # structures_dir=None rather than generating and dropping every structure. Turning the flag
        # off here says the draws are consumed in-process.
        replace(project.trace, save_final_structures=False),
        cache_embeds_dir=project.training.cache_embeds_dir,
        structures_dir=None,
        batch_size=1,
        # EXPLICIT. BioEmu's model loads on the host, and the sampler moves it only if it is told
        # where to: left unset it generates on the CPU, which is how both commands ran BioEmu
        # without ever touching the GPU the README says they need.
        device=_resolve_device(device),
    )
    draw_seeds = [base_seed + index for index in range(draws)]
    written = 0
    wrote_csv = False
    for record in chosen:
        reference_for_record = references[record.sequence_id]
        # DRAWS. Each draw has its own seed, base + index, and within a draw the steered and the
        # control structure SHARE it -- that is what keeps every pair matched while the draws differ.
        for seed in draw_seeds:
            residue_mask = torch.ones(1, len(record.sequence), dtype=torch.bool)
            target = torch.tensor([float(level)])

            # BOTH ARMS BACKBONE-COMPLETE. generate_unsteered returns a bare C-alpha tensor; the _frames
            # variant returns byte-identical positions PLUS the per-residue frames, which is what lets the
            # control be written as a PDB and scored over N/CA/C/O, the atom set the benchmarks use.
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

            # The steered arm's frames sit on its batch, padded the same way its positions are.
            orientations = padded_node_orientations(steered, [record])
            residues = len(record.sequence)

            values: dict[str, float] = {}
            if reference_for_record is not None:
                steered_backbone = (
                    batched_backbone_atoms(
                        positions.reshape(1, residues, 3), orientations, [record.sequence]
                    )[0]
                    .detach()
                    .cpu()
                )
                control_backbone = None
                if unsteered_frames is not None:
                    control_backbone = (
                        batched_backbone_atoms(
                            unsteered_frames[0].reshape(1, residues, 3),
                            unsteered_frames[1].reshape(1, residues, 3, 3),
                            [record.sequence],
                        )[0]
                        .detach()
                        .cpu()
                    )
                values = _score_against_reference(
                    reference_for_record, root, record.sequence, control_backbone, steered_backbone
                )
                _append_reference_scores(
                    destination / "reference_rmsd.csv",
                    record.sequence_id,
                    seed,
                    reference_for_record,
                    values,
                )
                wrote_csv = True
            elif steers_rg:
                values = {"steered": rg("steered", positions)}
                if unsteered is not None:
                    values["unsteered"] = rg("unsteered", unsteered)

            arms: list[tuple[str, object, object]] = [("on", positions, orientations)]
            if unsteered_frames is not None:
                arms.append(("off", unsteered_frames[0], unsteered_frames[1]))
            for arm, arm_positions, arm_orientations in arms:
                _write_backbone_pdb(
                    destination / _structure_filename(record.sequence_id, seed, arm),
                    arm_positions,
                    arm_orientations,
                    record.sequence,
                )
                written += 1

            line = f"{record.sequence_id}  seed {seed}"
            if "unsteered" in values:
                line += f"  off {values['unsteered']:.3f}"
            if "steered" in values:
                line += f"  on {values['steered']:.3f}"
            typer.echo(line)

    typer.echo(
        f"wrote {written} structure{'' if written == 1 else 's'} to {out_dir}"
        + (" and reference_rmsd.csv" if wrote_csv else "")
    )


if __name__ == "__main__":  # pragma: no cover - exercised as `python -m sae_bioemu.cli`
    # WITHOUT THIS, `python -m sae_bioemu.cli steer-apply ...` imports the module, defines the
    # commands, runs nothing and EXITS 0. Not a usability wart: a wrapper that checks the exit code
    # reads that silence as success, which is how a GPU job reported four generations it never ran.
    app()
