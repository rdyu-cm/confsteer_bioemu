"""Unsteered BioEmu draws in the cryptic-pocket coordinate's own atom set.

The cryptic-pocket progress coordinate is built on the frame-placed virtual C-beta (PLAN.md D-12),
so a generated structure has to be read the same way the crystal structures were. That is not what
:meth:`BioEmuRuntimeActivationSampler.generate_unsteered` returns -- it returns a bare C-alpha
tensor, which is all the radius-of-gyration work needed -- and D-12 recorded the ``extractor.py``
edit as the cost of choosing ``cb``. :meth:`BioEmuRuntimeActivationSampler.generate_unsteered_frames`
is that edit; this module is what turns its output into Angstrom C-beta.

The units trap, which looks plausible when it is wrong: BioEmu's ``ChemGraph.pos`` is in
NANOMETRES. Every distance in this project is in Angstrom, so a missed factor of ten reads as a
protein a tenth the size and a ``the progress coordinate`` a tenth the size with it. :func:`generated_coordinates`
multiplies exactly once and then checks the resulting geometry against real protein bond lengths,
where a factor of ten cannot hide.

**The backbone comes from BioEmu's own frame-to-atom conversion, and that needs a module load.**
``bioemu.convert_chemgraph`` imports ``openfold``, which imports ``dm-tree``, whose C extension
needs a newer ``libstdc++`` than these compute nodes carry in ``/lib64``. Every path through
``bioemu.sample`` hits it, so this is not specific to the C-beta construction --
``BioEmuRuntimeActivationSampler`` itself fails to construct without it. ``ml devel gcc/12.4.0``
supplies the library; :data:`BACKBONE_IMPORT_HINT` carries that instruction into the error message,
and the Slurm launcher loads it.

The alternative -- transcribing AlphaFold's ideal backbone geometry into this repository -- was
implemented first, while the import looked unavailable, and then **rejected on measurement**:
``rigid_group_atom_positions`` is residue-type dependent for N and C as well as for C-beta, so a
single generic triple reproduced the reference placement only to a median of 0.007 A but a maximum
of 0.18 A. That is inside ``COORDINATE_UNCERTAINTY_ANGSTROM`` and it is a SYSTEMATIC per-residue-type
offset applied to generated structures and not to the crystal structures they are scored against, so
it does not cancel the way draw noise does. One construction, and it is BioEmu's own.

Nothing here is imported by :mod:`sae_bioemu.cryptic_steering`, which stays free of BioEmu.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import torch

from sae_bioemu.cryptic_steering.structures import _VIRTUAL_CB_WEIGHTS, ATOM_SETS

__all__ = [
    "BACKBONE_IMPORT_HINT",
    "EMBEDDING_CACHE_SUFFIXES",
    "MissingEmbeddingCacheError",
    "backbone_from_frames",
    "batched_measurement_coordinates",
    "generated_coordinates",
    "peptide_bond_lengths",
    "require_cached_embeddings",
    "virtual_cb_from_backbone",
]

#: What ``bioemu.get_embeds.get_colabfold_embeds`` writes per sequence, keyed by the sequence's own
#: SHA-256. Restated here rather than imported from ``steering_workflow`` so this module does not
#: pull in the radius-of-gyration workflow; a test pins the two lists equal.
EMBEDDING_CACHE_SUFFIXES = ("_single.npy", "_pair.npy")

#: What to do when ``bioemu.convert_chemgraph`` will not import. Carried in the error message
#: because the traceback names ``tree/_tree.cpython-311-x86_64-linux-gnu.so`` and ``GLIBCXX_3.4.20``,
#: neither of which says "load a compiler module", and because the same failure takes down
#: ``BioEmuRuntimeActivationSampler`` construction with a message about optional dependencies that
#: are in fact installed.
BACKBONE_IMPORT_HINT = (
    "run 'ml devel gcc/12.4.0' before python: /lib64/libstdc++.so.6 on these nodes predates the "
    "GLIBCXX_3.4.20 that dm-tree's extension needs, and openfold imports dm-tree"
)

#: Index of N, CA and C in OpenFold's atom37 ordering, which
#: ``bioemu.convert_chemgraph.compute_backbone`` fills for the backbone group.
_ATOM37_N, _ATOM37_CA, _ATOM37_C = 0, 1, 2

#: Median ``C(i) - N(i+1)`` distance a correctly reconstructed backbone must fall inside, in
#: Angstrom. The peptide bond is 1.329 A; the window is deliberately wide because BioEmu places each
#: residue's frame independently and a minority of consecutive pairs are genuinely stretched in any
#: draw (the radius-of-gyration arms measure ~92% of C-alpha separations inside their own window).
#: This is a check on the CONSTRUCTION, not a structure gate: a wrong convention misses it by a
#: factor, not by a tail.
PEPTIDE_BOND_MEDIAN_WINDOW_ANGSTROM = (1.0, 1.7)


class MissingEmbeddingCacheError(ValueError):
    """Raised when the Evoformer embedding cache does not cover a sequence about to be folded."""


def require_cached_embeddings(
    cache_embeds_dir: Path | str | None, sequences: Mapping[str, str]
) -> None:
    """Fail closed unless every sequence already has cached Evoformer embeddings.

    ``bioemu.get_embeds.get_colabfold_embeds`` FAILS OPEN: on a miss it queries the ColabFold
    MMseqs2 server and runs a local AlphaFold2 Evoformer pass, which is the only step in this
    programme that reaches the internet from a compute node. On a partially populated cache -- and
    the cryptic-pocket cache is being filled by a running array right now -- that either hangs the
    job or silently mixes embeddings from two pipelines into one comparison.

    The message names the SEQUENCE IDs and their digests, not just a count: the recovery action is
    to run the embedding shard for those sequences, and a count does not say which.
    """
    if cache_embeds_dir is None:
        raise MissingEmbeddingCacheError(
            "cache_embeds_dir must name the reviewed Evoformer embedding cache: a null value "
            "silently redirects to ~/.bioemu_embeds_cache and lets BioEmu recompute embeddings "
            "from a ColabFold MMseqs2 query"
        )
    root = Path(os.path.expanduser(str(cache_embeds_dir)))
    missing: list[tuple[str, str]] = []
    for sequence_id, sequence in sorted(sequences.items()):
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        if not all((root / f"{digest}{suffix}").is_file() for suffix in EMBEDDING_CACHE_SUFFIXES):
            missing.append((sequence_id, digest))
    if missing:
        listed = ", ".join(f"{name} ({digest[:12]})" for name, digest in missing[:10])
        raise MissingEmbeddingCacheError(
            f"BioEmu Evoformer embedding cache {root} is missing {len(missing)} of "
            f"{len(sequences)} sequences; generation would fall through to a ColabFold MMseqs2 "
            f"query. Missing: {listed}"
        )


def virtual_cb_from_backbone(
    nitrogen: torch.Tensor, alpha_carbon: torch.Tensor, carbon: torch.Tensor
) -> torch.Tensor:
    """Place C-beta from the backbone frame, in torch, for any residue including glycine.

    The torch twin of :func:`sae_bioemu.cryptic_steering.structures.virtual_cb`, sharing its
    weights by import rather than by copy. Both must place the same atom or the generated structure
    and the crystal structure are being compared in two different atom sets, which is a systematic
    offset dressed as a measurement.
    """
    first = alpha_carbon - nitrogen
    second = carbon - alpha_carbon
    normal = torch.linalg.cross(first, second, dim=-1)
    wa, wb, wc = _VIRTUAL_CB_WEIGHTS
    return wa * normal + wb * first + wc * second + alpha_carbon


def backbone_from_frames(
    positions_angstrom: torch.Tensor, node_orientations: torch.Tensor, sequence: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(N, CA, C)`` in Angstrom from each residue's rigid frame, via BioEmu's own code.

    ``bioemu.convert_chemgraph.get_atom37_from_frames`` composes each residue's rigid transform with
    that residue type's backbone-group ideal geometry. It is what BioEmu itself writes to a PDB, so
    using it is what makes "the generated C-beta" the same atom BioEmu would report rather than a
    near-copy of it. It needs ``sequence`` because the ideal geometry is residue-type dependent --
    measured, not assumed: a single generic triple reproduces this to a median of 0.007 A and a
    maximum of 0.18 A.

    Its docstring says ``pos`` is in nanometres. It is not: the ideal geometry it composes with is in
    Angstrom, and BioEmu's own ``save_pdb_and_xtc`` passes Angstrom. Angstrom is what this function
    takes and returns, and the caller's peptide-bond check is what would catch the other reading.

    Casts through float32 internally (its own note), which is ~1e-5 A on these magnitudes and four
    orders of magnitude inside ``COORDINATE_UNCERTAINTY_ANGSTROM``. The output is returned in the
    input dtype so the caller's arithmetic stays in float64.
    """
    if positions_angstrom.ndim != 2 or positions_angstrom.shape[-1] != 3:
        raise ValueError("positions must be [residues, 3] for one structure")
    if tuple(node_orientations.shape) != (int(positions_angstrom.shape[0]), 3, 3):
        raise ValueError("node_orientations must be [residues, 3, 3] for one structure")
    if len(sequence) != int(positions_angstrom.shape[0]):
        raise ValueError(
            f"sequence has {len(sequence)} residues but the structure has "
            f"{int(positions_angstrom.shape[0])}"
        )
    try:
        from bioemu.convert_chemgraph import get_atom37_from_frames
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            f"bioemu.convert_chemgraph is required to place the backbone from frames; "
            f"{BACKBONE_IMPORT_HINT}"
        ) from error

    atom37, _, _ = get_atom37_from_frames(
        pos=positions_angstrom, node_orientations=node_orientations, sequence=sequence
    )
    placed = atom37.to(positions_angstrom)
    return placed[:, _ATOM37_N], placed[:, _ATOM37_CA], placed[:, _ATOM37_C]


def peptide_bond_lengths(carbon: torch.Tensor, nitrogen: torch.Tensor) -> torch.Tensor:
    """Return ``|C(i) - N(i+1)|`` in Angstrom: the check that the frame convention is right.

    A wrong axis convention, a transposed rotation or a missed nanometre-to-Angstrom factor all
    leave the individual residues internally consistent and destroy this one, because it is the only
    quantity that couples two neighbouring frames.
    """
    return torch.linalg.vector_norm(carbon[:-1] - nitrogen[1:], dim=-1)


def generated_coordinates(
    positions_nm: torch.Tensor,
    node_orientations: torch.Tensor,
    sequence: str,
    *,
    atom_set: str,
    check_peptide_bonds: bool = True,
) -> torch.Tensor:
    """Return ``[residues, 3]`` generated coordinates in ANGSTROM for one structure.

    ``positions_nm`` and ``node_orientations`` are one member of
    :meth:`BioEmuRuntimeActivationSampler.generate_unsteered_frames`'s output, already trimmed to
    that member's own length -- padding rows are frames of a residue that does not exist and would
    come back as plausible coordinates.

    ``ca`` is the scaled positions themselves. ``cb`` is placed from the N, CA and C that the
    residue's own frame defines, by the same construction the pool used on deposited structures.

    ``check_peptide_bonds`` verifies the reconstruction on the structure being converted, and is on
    by default: the cost is ``L - 1`` distances against a denoiser pass. Turn it off only for a
    synthetic frame field that is not meant to be a protein.
    """
    if atom_set not in ATOM_SETS:
        raise ValueError(f"atom_set must be one of {ATOM_SETS}")
    if positions_nm.ndim != 2 or positions_nm.shape[-1] != 3:
        raise ValueError("positions must be [residues, 3] for one structure")
    if tuple(node_orientations.shape) != (int(positions_nm.shape[0]), 3, 3):
        raise ValueError("node_orientations must be [residues, 3, 3] for one structure")
    if len(sequence) != int(positions_nm.shape[0]):
        raise ValueError(
            f"sequence has {len(sequence)} residues but the generated structure has "
            f"{int(positions_nm.shape[0])}"
        )
    angstrom = positions_nm * 10.0
    if atom_set == "ca":
        return angstrom

    nitrogen, alpha_carbon, carbon = backbone_from_frames(angstrom, node_orientations, sequence)
    if check_peptide_bonds and int(angstrom.shape[0]) > 1:
        median = float(peptide_bond_lengths(carbon, nitrogen).median())
        low, high = PEPTIDE_BOND_MEDIAN_WINDOW_ANGSTROM
        if not low <= median <= high:
            raise ValueError(
                f"reconstructed backbone has a median C(i)-N(i+1) distance of {median:.3f} "
                f"Angstrom, outside {PEPTIDE_BOND_MEDIAN_WINDOW_ANGSTROM}; the frames were not "
                "read in the convention or the units this construction assumes, and every C-beta "
                "would be placed off a backbone that is not the generated one"
            )
    return virtual_cb_from_backbone(nitrogen, alpha_carbon, carbon)


def batched_measurement_coordinates(
    positions_nm: torch.Tensor,
    node_orientations: torch.Tensor,
    sequences: Sequence[str],
    *,
    atom_set: str,
    check_peptide_bonds: bool = True,
) -> torch.Tensor:
    """Return ``[members, width, 3]`` Angstrom coordinates in ``atom_set``, GRADIENT INTACT.

    **This is the training path's version of** :func:`generated_coordinates` **and the fix for
    D-69's atom-set defect.** The cryptic pocket artifact declares ``atom_set = 'cb'``, so
    ``v = d(holo) - d(apo)`` is a difference of virtual-C-beta distances; until this existed the
    steering path fed ``sampled_batch.pos`` -- bare C-alpha -- into ``cryptic_off_path_inputs``, so
    every journalled ``the progress coordinate`` and every ``control_reporting_value`` was a C-alpha realised change
    projected onto a C-beta transition. Measured: a C-alpha reading reproduced the journal on
    12634/12636 draws at median 2.7e-07, a C-beta reading on 35/12636 at median error 0.240 and
    maximum 9.199, and one protein's target origin came out at -6.55, making its holo ask 7.55
    transitions.

    ``positions_nm`` is ``[members, width, 3]`` in BioEmu's own NANOMETRES and ``node_orientations``
    is ``[members, width, 3, 3]``, both padded to the batch width; ``sequences`` gives each member's
    own length, and rows past it are never read -- they are frames of a residue that does not exist
    and would come back as entirely plausible coordinates.

    **Nothing is detached.** The scored structure carries the adapter's gradient and this is on the
    path from that structure to the loss, so a ``.detach()`` here would silently train nothing. The
    gradient reaches C-beta through the translations always, and through the rotations whenever the
    denoiser's SO(3) branch is differentiable; neither is asserted here, because a rotation branch
    that carried no gradient would still leave a correctly PLACED C-beta and a correct ``the progress coordinate``, and
    :func:`~sae_bioemu.steering.ordinary_workflow.assert_adapter_only_gradients` is what checks that
    the adapter is being trained at all.

    ``atom_set='ca'`` returns ``positions_nm * 10`` and touches no frame, so it is exactly the
    tensor the pre-D-69 path used and costs nothing beyond the multiply.
    """
    if atom_set not in ATOM_SETS:
        raise ValueError(f"atom_set must be one of {ATOM_SETS}")
    if positions_nm.ndim != 3 or positions_nm.shape[-1] != 3:
        raise ValueError("positions must be [members, residues, 3]")
    members = int(positions_nm.shape[0])
    if len(sequences) != members:
        raise ValueError("one sequence per batch member is required to read its own frames")
    if atom_set == "ca":
        return positions_nm * 10.0
    if tuple(node_orientations.shape) != (members, int(positions_nm.shape[1]), 3, 3):
        raise ValueError(
            "node_orientations must be [members, residues, 3, 3] padded to the batch width; the "
            f"{atom_set!r} atom set is placed from each residue's own frame and cannot be "
            "recovered from C-alpha alone"
        )
    width = int(positions_nm.shape[1])
    placed = positions_nm.new_zeros((members, width, 3))
    rows: list[torch.Tensor] = []
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        if length > width:
            raise ValueError(
                f"batch member {index} has {length} residues but the batch is padded to {width}"
            )
        rows.append(
            generated_coordinates(
                positions_nm[index, :length],
                node_orientations[index, :length],
                sequence,
                atom_set=atom_set,
                check_peptide_bonds=check_peptide_bonds,
            )
        )
    # Assembled out of place: an in-place index_put on ``placed`` would work, but building the
    # padded tensor by concatenation keeps the autograd graph a plain function of the per-member
    # results and makes the zero padding a constant rather than a written-over leaf.
    padded = [
        torch.cat([row, row.new_zeros((width - int(row.shape[0]), 3))], dim=0)
        if int(row.shape[0]) < width
        else row
        for row in rows
    ]
    return torch.stack(padded).to(placed.dtype)


def unsteered_draws(
    sampler: object,
    record: object,
    seeds: Sequence[int],
    *,
    atom_set: str,
    batch_size: int = 1,
) -> torch.Tensor:
    """Return ``[draws, residues, 3]`` Angstrom coordinates for one sequence's unsteered ensemble.

    One seed per draw, and the seeds are the recorded provenance of the ensemble: BioEmu's sampler
    is seeded per member, so ``(model version, config, sequence, seed, backend settings)`` is what
    reproduces a draw. The integer alone is not enough and the caller records the rest.

    ``batch_size`` groups seeds into one generation call. Members of a batch carry distinct seeds,
    which ``_steering_members`` requires and which is exactly right here: two draws sharing a seed
    would be one structure computed twice. Batching is only sound because ``require_seeded_batch_
    denoiser`` holds: under DPM-Solver-2 with ``noise: 0.0`` the trajectory is a deterministic
    function of each member's own seeded prior, so a batched run reproduces batch-size-1 runs (D8).

    Returned on the CPU in float64, deliberately. Everything the measurement then touches -- the
    pair set, the experimental apo and holo -- is CPU float64, and mixing devices raised only after
    a full protein's generation had already been paid for. The tensors are ``draws x L x 3``, which
    is kilobytes.
    """
    sequence: str = record.sequence  # type: ignore[attr-defined]
    drawn: list[torch.Tensor] = []
    ordered = [int(value) for value in seeds]
    if len(set(ordered)) != len(ordered):
        raise ValueError("unsteered draw seeds must be distinct, or two draws are one structure")
    for start in range(0, len(ordered), max(1, batch_size)):
        chunk = ordered[start : start + max(1, batch_size)]
        members = [record] * len(chunk)
        positions, orientations = sampler.generate_unsteered_frames(  # type: ignore[attr-defined]
            members if len(chunk) > 1 else record,
            seed=chunk if len(chunk) > 1 else chunk[0],
        )
        for index in range(len(chunk)):
            drawn.append(
                generated_coordinates(
                    positions[index, : len(sequence)].to(torch.float64),
                    orientations[index, : len(sequence)].to(torch.float64),
                    sequence,
                    atom_set=atom_set,
                )
                .detach()
                .cpu()
            )
    return torch.stack(drawn)


def sequence_digests(sequences: Iterable[str]) -> tuple[str, ...]:
    """Return the embedding-cache key of each sequence, for provenance in a run report."""
    return tuple(hashlib.sha256(value.encode()).hexdigest() for value in sequences)
