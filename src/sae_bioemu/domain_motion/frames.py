"""N, CA, C and O from BioEmu's per-residue frames, with the gradient intact. SCOPE: DOMAIN MOTIONS.

The domain-motion objective is a DIRECT METRIC: it measures the same superposed backbone RMSD over
N/CA/C/O that ``bioemu-benchmarks`` grades ``multiconf_domainmotion`` on (``calc_metrics_global``,
tau = 3.0 Angstrom; see ``docs/domain_motion/dm_success_metric.txt``). So the four atoms the metric
reads have to come off the denoiser with a gradient on them, not out of a written PDB.

WHY THIS IS A SEPARATE FILE AND NOT A CALL INTO ``bioemu/cryptic_generation.py``
--------------------------------------------------------------------------------

``cryptic_generation.backbone_from_frames`` places the same three of these four atoms, and it is
tempting to extend it to return O as well. It must not be extended, for a fingerprint reason rather
than a taste one: it is the sole member of the ``two_state_generation_shared`` code group, which the
cryptic-pocket AND kinase registrations both compose. An edit there re-keys every cryptic and kinase
training checkpoint for a change made on behalf of a third property that neither of them executes --
the exact defect ``steering/property_registry.py`` exists to end. This module is hashed in
``domain_motion_training_only`` and reached by domain motion alone.

The duplication is therefore deliberate and it is small: both files resolve to ONE construction,
``bioemu.convert_chemgraph.get_atom37_from_frames``, which is what BioEmu itself writes to a PDB.
Nothing here re-implements an ideal-geometry table, and ``tests/test_domain_motion_direct_metric.py``
pins the N/CA/C this module returns to the ones ``cryptic_generation`` returns, so the two cannot
drift into placing different atoms.

WHERE THE OXYGEN COMES FROM, AND WHY IT IS DIFFERENTIABLE
----------------------------------------------------------

``get_atom37_from_frames`` fills N, CA and C from ``frames_to_atom14_pos`` -- each residue's rigid
transform composed with that residue type's backbone-group ideal geometry -- and then calls
``_adjust_oxygen_pos``, which places O in the plane of CA(i), C(i) and N(i+1) at a fixed C=O bond
length, falling back to the CA->C / CA->N bisector at the C-terminus. Every step is a smooth
function of ``(pos, node_orientations)``: normalisations guarded by ``+1e-7``, a sum, a scale. The
``torch.no_grad()`` blocks inside ``convert_chemgraph`` wrap the CONSTANT lookup tables
(``restype_atom14_mask``, ``restype_atom14_rigid_group_positions``), not the transform.

That is an argument, and this project does not ship arguments. ``tests/test_domain_motion_direct_
metric.py::test_the_gradient_of_the_direct_metric_reaches_every_backbone_atom_through_the_frames``
measures it: it differentiates the aligned RMSD back to ``pos`` and ``node_orientations`` and
asserts a nonzero gradient on BOTH, including on the O-only path, where N/CA/C are detached.

Nothing here reads a reference structure or a threshold; this file turns frames into atoms and
stops.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = [
    "BACKBONE_ATOMS",
    "PEPTIDE_BOND_MEDIAN_WINDOW_ANGSTROM",
    "backbone_atoms_from_frames",
    "batched_backbone_atoms",
    "peptide_bond_lengths",
]

#: The metric's atom set, in the metric's own per-residue order. ``bioemu_benchmarks``'
#: ``calc_metrics_global`` selects exactly these four and its own comment pins the ordering;
#: :data:`sae_bioemu.cryptic_steering.success_at_b.BACKBONE_ATOMS` is the same tuple and a test
#: pins the two equal, because a reference read in one order and a draw built in another is a
#: silent per-residue permutation that leaves every RMSD finite and wrong.
BACKBONE_ATOMS: tuple[str, ...] = ("N", "CA", "C", "O")

#: Where each of :data:`BACKBONE_ATOMS` sits in OpenFold's atom37 ordering, which is
#: ``[N, CA, C, CB, O, ...]``. O is index 4 and NOT index 3: index 3 is the C-beta, and reading it
#: as the carbonyl would put a real atom in the O column and go unnoticed by every finiteness and
#: bond-length check.
_ATOM37_INDEX: dict[str, int] = {"N": 0, "CA": 1, "C": 2, "O": 4}

#: Median ``C(i) - N(i+1)`` distance a correctly reconstructed backbone falls inside, in Angstrom.
#: Copied by value from :data:`sae_bioemu.bioemu.cryptic_generation.
#: PEPTIDE_BOND_MEDIAN_WINDOW_ANGSTROM` and pinned equal by a test. It checks the CONSTRUCTION --
#: a wrong frame convention, a transposed rotation or a missed nanometre factor misses it by a
#: factor, not by a tail -- and is not a structure gate.
PEPTIDE_BOND_MEDIAN_WINDOW_ANGSTROM = (1.0, 1.7)


def _require_frames(positions: Tensor, node_orientations: Tensor, sequence: str) -> None:
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions must be [residues, 3] for one structure")
    if tuple(node_orientations.shape) != (int(positions.shape[0]), 3, 3):
        raise ValueError("node_orientations must be [residues, 3, 3] for one structure")
    if len(sequence) != int(positions.shape[0]):
        raise ValueError(
            f"sequence has {len(sequence)} residues but the structure has "
            f"{int(positions.shape[0])}"
        )


def peptide_bond_lengths(carbon: Tensor, nitrogen: Tensor) -> Tensor:
    """Return ``|C(i) - N(i+1)|`` in Angstrom: the only quantity that couples two frames."""
    return torch.linalg.vector_norm(carbon[:-1] - nitrogen[1:], dim=-1)


def backbone_atoms_from_frames(
    positions_angstrom: Tensor,
    node_orientations: Tensor,
    sequence: str,
    *,
    check_peptide_bonds: bool = True,
) -> Tensor:
    """Return ``[residues, 4, 3]`` N/CA/C/O in ANGSTROM for one structure, gradient intact.

    ``positions_angstrom`` is one member of ``generate_unsteered_frames`` / the steered batch,
    ALREADY MULTIPLIED BY TEN and already trimmed to that member's own length. BioEmu's
    ``ChemGraph.pos`` is in NANOMETRES; the conversion is the caller's because
    :func:`batched_backbone_atoms` does it once for a whole batch, and a second multiply here
    would be a factor of ten that leaves every structure internally consistent.

    **Nothing is detached.** The steered structure carries the adapter's gradient and this is on
    the path from that structure to the loss, so a ``.detach()`` here would train nothing at all
    while producing an entirely plausible loss curve.
    """
    _require_frames(positions_angstrom, node_orientations, sequence)
    try:
        from bioemu.convert_chemgraph import get_atom37_from_frames
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "bioemu.convert_chemgraph is required to place N/CA/C/O from the residue frames; run "
            "'ml devel gcc/12.4.0' before python: /lib64/libstdc++.so.6 on these nodes predates "
            "the GLIBCXX_3.4.20 that dm-tree's extension needs, and openfold imports dm-tree"
        ) from error

    atom37, _, _ = get_atom37_from_frames(
        pos=positions_angstrom, node_orientations=node_orientations, sequence=sequence
    )
    placed = atom37.to(positions_angstrom)
    # Stacked rather than sliced with a fancy index so the per-residue atom order is the literal
    # BACKBONE_ATOMS order and is readable at the call site.
    atoms = torch.stack([placed[:, _ATOM37_INDEX[name]] for name in BACKBONE_ATOMS], dim=1)
    if check_peptide_bonds and int(positions_angstrom.shape[0]) > 1:
        median = float(peptide_bond_lengths(atoms[:, 2], atoms[:, 0]).detach().median())
        low, high = PEPTIDE_BOND_MEDIAN_WINDOW_ANGSTROM
        if not low <= median <= high:
            raise ValueError(
                f"reconstructed backbone has a median C(i)-N(i+1) distance of {median:.3f} "
                f"Angstrom, outside {PEPTIDE_BOND_MEDIAN_WINDOW_ANGSTROM}; the frames were not "
                "read in the convention or the units this construction assumes, so every atom the "
                "direct metric scores would be placed off a backbone that is not the generated one"
            )
    return atoms


def batched_backbone_atoms(
    positions_nm: Tensor,
    node_orientations: Tensor,
    sequences: Sequence[str],
    *,
    check_peptide_bonds: bool = True,
) -> Tensor:
    """Return ``[members, width, 4, 3]`` Angstrom N/CA/C/O, padded to the batch width.

    ``positions_nm`` is ``[members, width, 3]`` in BioEmu's own NANOMETRES and
    ``node_orientations`` is ``[members, width, 3, 3]``, both padded; ``sequences`` gives each
    member's own length and rows past it are never read -- they are frames of a residue that does
    not exist and come back as entirely plausible coordinates.

    Assembled by concatenation rather than by an in-place write into a zero tensor, so the padded
    result is a plain function of the per-member results and the padding is a constant rather than
    a written-over leaf.
    """
    if positions_nm.ndim != 3 or positions_nm.shape[-1] != 3:
        raise ValueError("positions must be [members, residues, 3]")
    members = int(positions_nm.shape[0])
    width = int(positions_nm.shape[1])
    if len(sequences) != members:
        raise ValueError("one sequence per batch member is required to read its own frames")
    if tuple(node_orientations.shape) != (members, width, 3, 3):
        raise ValueError(
            "node_orientations must be [members, residues, 3, 3] padded to the batch width; "
            "N/CA/C/O are placed from each residue's own frame and cannot be recovered from "
            "C-alpha alone"
        )
    angstrom = positions_nm * 10.0
    rows: list[Tensor] = []
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        if length > width:
            raise ValueError(
                f"batch member {index} has {length} residues but the batch is padded to {width}"
            )
        row = backbone_atoms_from_frames(
            angstrom[index, :length],
            node_orientations[index, :length],
            sequence,
            check_peptide_bonds=check_peptide_bonds,
        )
        if length < width:
            row = torch.cat([row, row.new_zeros((width - length, len(BACKBONE_ATOMS), 3))], dim=0)
        rows.append(row)
    return torch.stack(rows)
