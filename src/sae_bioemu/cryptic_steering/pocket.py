"""The pocket pair set and the SE(3)-invariant apo->holo progress coordinate.

One pair set ``P`` and one weighting ``w`` carry the whole objective. For pocket-local residue
pairs with experimental transition ``v = delta_d_star`` and realised change ``u = delta_d_hat``:

    the progress coordinate      = <u, v>_w / ||v||_w^2          # 0 at apo, 1 at holo, signed, alignment-free
    u_perp  = u - the progress coordinate * v                    # everything the transition does not license

and the decomposition is exact:

    (1/|P|) ||u - v||_w^2 = (1/|P|) [ (the progress coordinate - 1)^2 ||v||_w^2 + ||u_perp||_w^2 ]

That identity is what lets ``the progress coordinate`` ride the existing scalar relative-target machinery while
``z = ||u_perp||_w`` becomes a separate weighted penalty. Dividing through by ``||v||_w^2`` writes
it in dimensionless form, with ``zeta = z / ||v||_w`` the off-path excursion measured in units of
the apo->holo transition:

    (1/|P|) ||u - v||_w^2 = (||v||_w^2 / |P|) [ (the progress coordinate - 1)^2 + zeta^2 ]

so ``zeta`` is directly comparable to ``the progress coordinate`` and across proteins, which raw ``z`` is not.

The identity holds ONLY when ``the progress coordinate``, ``u_perp`` and the loss share one pair set and one weighting,
which is why context-context pairs live inside ``P`` rather than in a fourth loss term
(PLAN.md D-1): their ``delta_d_star`` is set to zero, so they contribute nothing to ``v`` and any
motion in them lands entirely in ``u_perp``. The plan originally expected that to be true of the
GEOMETRY -- a context shell that barely moves -- and it is not; the zero is imposed, and the raw
change is kept in ``raw_delta_d_star`` so that what the geometry does stays measurable.

One consequence to keep in view: ``z`` therefore mixes two things, "the pocket moved off the
transition axis" on target-carrying pairs and "the fold moved" on context-context pairs, with the
second holding a third of the weight mass under the default class weights. See
:func:`off_path_class_decomposition` and :func:`transition_class_off_path`, which report the split
without changing it.

Distances are computed from C-alpha coordinates in Angstrom. Nothing here aligns or superposes:
a coordinate-space objective would put a rigid-body component into the bounded residual's gradient,
spending the ``rho_max`` budget translating and rotating the molecule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

#: Sequence separation below which a pair is treated as trivially constrained by the backbone.
#: Matches the repository's existing conventions: contact density excludes pairs closer than
#: three positions, Local-Q's reference contacts use ``|i-j| >= 3``, and the clash term uses
#: ``|i-j| > 3``.
MIN_SEQUENCE_SEPARATION = 3

#: Pair classes. ``pocket_pocket`` carries the transition; ``pocket_context`` anchors the pocket to
#: the fold, without which "open the pocket" is satisfied by locally unfolding it; ``context_context``
#: should not change.
PAIR_CLASSES = ("pocket_pocket", "pocket_context", "context_context")

#: Classes whose experimental change defines the transition direction ``v``.
#:
#: Context-context pairs are deliberately excluded. The plan originally assumed their transition
#: was near zero, so their penalty-only behaviour would emerge from the decomposition; measured on
#: the pool that is false, with a median share of 0.143 and context-context DOMINANT for 13.8% of
#: proteins. Left in ``v``, a fold rearranging around a pocket would count as progress toward the
#: target, which is the opposite of the intent.
#:
#: Zeroing their target is exactly "define the progress coordinate over pocket-pocket and pocket-context only": a pair
#: with ``v_ij = 0`` contributes to neither ``<u, v>`` nor ``||v||^2``, so it cannot move the progress coordinate, while
#: ``u_perp = u - the progress coordinate*v`` still equals ``u_ij`` there, so all of its motion lands in the penalty.
#: The decomposition is an algebraic identity in ``v`` and holds unchanged.
TRANSITION_CLASSES = ("pocket_pocket", "pocket_context")


@dataclass(frozen=True)
class PocketSpec:
    """One protein's pocket and its immediate structural neighbourhood.

    Indices are positions in the sequence handed to BioEmu, not PDB ``auth_seq_id``. Converting
    between the two is the project's highest-risk step (PLAN.md R0) and belongs upstream of this
    module, which assumes the conversion has already been verified.
    """

    sequence_id: str
    residue_count: int
    pocket_indices: tuple[int, ...]
    context_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.residue_count < 1:
            raise ValueError("residue_count must be positive")
        if not self.pocket_indices:
            raise ValueError("pocket_indices must be non-empty")
        for name, indices in (
            ("pocket_indices", self.pocket_indices),
            ("context_indices", self.context_indices),
        ):
            if len(set(indices)) != len(indices):
                raise ValueError(f"{name} must not repeat a residue")
            if any(
                not isinstance(index, int) or isinstance(index, bool) or index < 0
                for index in indices
            ):
                raise ValueError(f"{name} must contain nonnegative integers")
            if any(index >= self.residue_count for index in indices):
                raise ValueError(f"{name} must lie inside the sequence")
        if set(self.pocket_indices) & set(self.context_indices):
            raise ValueError("pocket and context residue sets must be disjoint")


@dataclass(frozen=True)
class PocketPairSet:
    """The canonical pair set for one protein, with its weights and experimental transition.

    ``pair_index`` is ``[pairs, 2]`` in canonical order (``i < j``, lexicographically sorted), so
    two builds of the same inputs produce byte-identical tensors.
    """

    sequence_id: str
    pair_index: torch.Tensor
    pair_class: tuple[str, ...]
    weights: torch.Tensor
    #: The EFFECTIVE transition: ``holo - apo`` with every non-transition-class pair force-zeroed.
    #: This is the vector ``the progress coordinate``, the loss and the decomposition are defined against.
    delta_d_star: torch.Tensor
    #: The RAW transition: ``holo - apo`` on every pair, including the context-context pairs whose
    #: effective target is zero. Nothing computes ``the progress coordinate``, ``z`` or the loss from this -- it is
    #: retained so the measurement that justified the zeroing stays reproducible from the pair set.
    #: Zeroing context-context targets was decided from a measured class decomposition (median
    #: context-context share 0.143, dominant for 13.8% of proteins); computed from
    #: ``delta_d_star`` that share is now identically zero, so without this field the evidence for a
    #: central design decision could not be recovered from any artifact the build writes.
    #:
    #: ``None`` where the raw transition was never carried -- a pair set reloaded from a pocket
    #: artifact, which stores only what the runtime needs. Consumers must fail closed on it rather
    #: than substituting ``delta_d_star``, which is a different vector.
    raw_delta_d_star: torch.Tensor | None = None

    def __post_init__(self) -> None:
        pairs = self.pair_index.shape[0]
        if self.pair_index.ndim != 2 or self.pair_index.shape[1] != 2:
            raise ValueError("pair_index must be [pairs, 2]")
        if len(self.pair_class) != pairs:
            raise ValueError("pair_class must carry one label per pair")
        if any(label not in PAIR_CLASSES for label in self.pair_class):
            raise ValueError(f"pair_class entries must be one of {PAIR_CLASSES}")
        tensors = [("weights", self.weights), ("delta_d_star", self.delta_d_star)]
        if self.raw_delta_d_star is not None:
            tensors.append(("raw_delta_d_star", self.raw_delta_d_star))
        for name, tensor in tensors:
            if tensor.shape != (pairs,) or not tensor.is_floating_point():
                raise ValueError(f"{name} must be floating with one value per pair")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite")
        if pairs and float(self.weights.min()) < 0:
            raise ValueError("weights must be nonnegative")

    def __len__(self) -> int:
        return int(self.pair_index.shape[0])


def pair_distances(coordinates: torch.Tensor, pair_index: torch.Tensor) -> torch.Tensor:
    """Return ``[..., pairs]`` distances for ``coordinates`` of shape ``[..., residues, 3]``."""
    if coordinates.shape[-1] != 3 or coordinates.ndim < 2:
        raise ValueError("coordinates must be [..., residues, 3]")
    first = coordinates.index_select(-2, pair_index[:, 0])
    second = coordinates.index_select(-2, pair_index[:, 1])
    return torch.linalg.vector_norm(first - second, dim=-1)


def build_pair_set(
    spec: PocketSpec,
    apo_coordinates: torch.Tensor,
    holo_coordinates: torch.Tensor,
    *,
    cutoff_angstrom: float,
    class_weights: dict[str, float] | None = None,
    valid_residues: Sequence[bool] | None = None,
    transition_classes: Sequence[str] = TRANSITION_CLASSES,
) -> PocketPairSet:
    """Build one protein's canonical pair set from its experimental apo and holo structures.

    A pair is kept when it is inside ``cutoff_angstrom`` in EITHER state. An apo-only cutoff would
    discard contacts that come into range on opening, and an intersection would discard them twice
    over -- those are precisely the pairs carrying the transition.

    ``class_weights`` scales each class after per-class size normalisation, so a large context shell
    cannot swamp a small pocket on count alone. Weights are deliberately NOT proportional to
    ``|delta_d_star|``: that collapses ``the progress coordinate`` onto whichever pair moves most, usually a single loop
    tip, giving a one-pair coordinate wearing the costume of a many-pair one. Use
    :func:`participation_ratio` to detect that case rather than encouraging it.

    ``valid_residues`` marks residues resolved in both structures; pairs touching an unresolved
    residue are dropped. Dropping is the only honest option -- imputing a coordinate invents the
    transition -- but it biases the pool toward pockets ordered in both states, because disordered
    loops are preferentially the ones missing from crystal structures. Report the coverage.
    """
    if apo_coordinates.shape != holo_coordinates.shape:
        raise ValueError("apo and holo coordinates must have the same shape")
    if apo_coordinates.shape[0] != spec.residue_count:
        raise ValueError("coordinates must carry one row per residue in the BioEmu sequence")

    weights_by_class = {name: 1.0 for name in PAIR_CLASSES}
    if class_weights:
        unknown = set(class_weights) - set(PAIR_CLASSES)
        if unknown:
            raise ValueError(f"unknown pair classes: {sorted(unknown)}")
        weights_by_class.update(class_weights)

    resolved = (
        [True] * spec.residue_count if valid_residues is None else [bool(v) for v in valid_residues]
    )
    if len(resolved) != spec.residue_count:
        raise ValueError("valid_residues must carry one flag per residue")

    pocket = set(spec.pocket_indices)
    context = set(spec.context_indices)
    considered = sorted(pocket | context)

    rows: list[tuple[int, int]] = []
    labels: list[str] = []
    for position, first in enumerate(considered):
        for second in considered[position + 1 :]:
            if abs(first - second) < MIN_SEQUENCE_SEPARATION:
                continue
            if not (resolved[first] and resolved[second]):
                continue
            apo_distance = torch.linalg.vector_norm(
                apo_coordinates[first] - apo_coordinates[second]
            )
            holo_distance = torch.linalg.vector_norm(
                holo_coordinates[first] - holo_coordinates[second]
            )
            if float(torch.minimum(apo_distance, holo_distance)) > cutoff_angstrom:
                continue
            in_pocket = (first in pocket, second in pocket)
            if all(in_pocket):
                labels.append("pocket_pocket")
            elif any(in_pocket):
                labels.append("pocket_context")
            else:
                labels.append("context_context")
            rows.append((first, second))

    device = apo_coordinates.device
    dtype = apo_coordinates.dtype
    pair_index = torch.tensor(rows, dtype=torch.long, device=device).reshape(-1, 2)
    if not rows:
        empty = torch.zeros((0,), dtype=dtype, device=device)
        return PocketPairSet(
            spec.sequence_id, pair_index, (), empty, empty, raw_delta_d_star=empty.clone()
        )

    counts = {name: labels.count(name) for name in PAIR_CLASSES}
    weights = torch.tensor(
        [weights_by_class[label] / counts[label] for label in labels], dtype=dtype, device=device
    )
    unknown_transition = set(transition_classes) - set(PAIR_CLASSES)
    if unknown_transition:
        raise ValueError(f"unknown transition classes: {sorted(unknown_transition)}")

    apo_distances = pair_distances(apo_coordinates, pair_index)
    holo_distances = pair_distances(holo_coordinates, pair_index)
    raw_transition = (holo_distances - apo_distances).to(dtype)
    # Pairs outside the transition classes keep their place in P, their weight and their label,
    # but carry a zero target: they can only ever contribute to the off-path penalty. The
    # pre-zeroing vector is kept beside it so the class decomposition that justified the zeroing
    # stays measurable; it feeds no coordinate, no loss and no artifact.
    contributes = torch.tensor(
        [label in set(transition_classes) for label in labels], dtype=torch.bool, device=device
    )
    transition = torch.where(contributes, raw_transition, torch.zeros_like(raw_transition))
    return PocketPairSet(
        sequence_id=spec.sequence_id,
        pair_index=pair_index,
        pair_class=tuple(labels),
        weights=weights,
        delta_d_star=transition,
        raw_delta_d_star=raw_transition,
    )


def delta_d_star(
    coordinates: torch.Tensor, reference: torch.Tensor, pair_index: torch.Tensor
) -> torch.Tensor:
    """Return ``u``, the realised pairwise distance change against a reference structure."""
    return pair_distances(coordinates, pair_index) - pair_distances(reference, pair_index)


def progress_coordinate(
    realised: torch.Tensor, transition: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(the progress coordinate, z)``: position along the apo->holo transition, and the off-path residual.

    ``the progress coordinate`` is 0 when nothing moved and 1 when the realised change equals the experimental
    transition. ``z = ||u_perp||_w`` is everything that moved which the transition does not
    license, and it is the gate against opening a pocket by deforming the protein.

    ``z`` must never be reported or selected on without ``the progress coordinate`` beside it: an adapter that does
    nothing has ``u = 0`` and therefore a perfect ``z``, which is the same trap that let the two
    weakest steerers post the best lDDT in the radius-of-gyration work.
    """
    if realised.shape[-1] != transition.shape[-1] or transition.shape != weights.shape:
        raise ValueError("realised, transition and weights must agree on the pair dimension")
    denominator = torch.sum(weights * transition * transition, dim=-1)
    if not bool(torch.all(denominator > 0)):
        raise ValueError("the experimental transition must be non-degenerate")
    progress = torch.sum(weights * realised * transition, dim=-1) / denominator
    residual = realised - progress.unsqueeze(-1) * transition
    z = torch.sqrt(torch.sum(weights * residual * residual, dim=-1))
    return progress, z


def transition_scale(transition: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return ``||v||_w``, the weighted length of one protein's apo->holo transition."""
    if transition.shape != weights.shape:
        raise ValueError("transition and weights must agree on the pair dimension")
    return torch.sqrt(torch.sum(weights * transition * transition, dim=-1))


def normalised_off_path(
    off_path: torch.Tensor, transition: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Return ``zeta = z / ||v||_w``: the off-path excursion in units of the transition itself.

    ``z`` is in Angstrom and its scale is a property of the protein, not of the intervention: on the
    pool it varies about fifteenfold across proteins, and ``z^2`` -- the form the loss actually uses
    -- varies by two orders of magnitude. A single ``lambda_perp`` applied to ``z^2`` would
    therefore impose a per-protein weight spanning that range, heaviest on exactly the proteins
    whose folds move most, which are the ones the method is meant to work on. That is the argument
    DESIGN_DECISIONS D4b makes for the PROPERTY loss, which is normalised by each sequence's own
    sigma; it is not a blanket rule, and it does not apply to a term whose scale is universal rather
    than per protein. ``lambda_g`` and ``lambda_l`` are deliberately dimensional -- PLAN.md D-15
    argues at length that ``lambda_g`` should stay that way, because it measures deviation from
    fixed constants (a 3.8 A bond, a 4.1 A clash floor) that are the same in every protein.

    ``zeta`` is dimensionless and reads directly against ``the progress coordinate``: ``zeta = 0.5`` is an off-path
    excursion half the length of the whole apo-to-holo transition, whatever protein it belongs to.
    Rescaling a protein's transition and its realised change together leaves ``zeta`` unchanged,
    which is what makes it comparable across the pool.

    Like ``z`` it must never be reported or selected on without ``the progress coordinate`` beside it: an adapter that
    does nothing has ``u = 0`` and therefore ``zeta = 0``.
    """
    scale = transition_scale(transition, weights)
    if not bool(torch.all(scale > 0)):
        raise ValueError("the experimental transition must be non-degenerate")
    return off_path / scale


def batched_progress_coordinate(
    coordinates: torch.Tensor,
    references: torch.Tensor,
    pair_sets: Sequence[PocketPairSet],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(the progress coordinate, z)`` of shape ``[members]`` for one generated batch.

    Batching across sequences is straightforward here, unlike residue Local-Q, which refuses it.
    The distinction is the size of the per-sequence reference: Local-Q carries an ``[L, L]``
    contact matrix that cannot be stacked across members of different lengths, whereas each member
    here contributes a short pair list and returns a scalar. Members are therefore looped and their
    scalars stacked, which keeps gradients intact and costs nothing measurable -- the loop runs over
    a handful of members doing small tensor operations, against a denoiser forward and backward pass
    that already dominates by orders of magnitude.

    That matters because the batch is where the paired dose term lives: sharing a batch across a
    sequence's levels is what lets the contrastive term difference two levels at one noise draw.

    ``coordinates`` and ``references`` are ``[members, residues, 3]`` in Angstrom, padded to the
    longest member; each pair set indexes only its own member's residues, so padding is never read.
    """
    if coordinates.shape != references.shape:
        raise ValueError("coordinates and references must have the same shape")
    if coordinates.ndim != 3 or coordinates.shape[0] != len(pair_sets):
        raise ValueError("coordinates must be [members, residues, 3] with one pair set per member")

    progress: list[torch.Tensor] = []
    off_path: list[torch.Tensor] = []
    for member, pairs in enumerate(pair_sets):
        if len(pairs) == 0:
            raise ValueError(f"member {member} ({pairs.sequence_id}) has an empty pair set")
        realised = delta_d_star(coordinates[member], references[member], pairs.pair_index)
        progress, z = progress_coordinate(
            realised,
            pairs.delta_d_star.to(realised),
            pairs.weights.to(realised),
        )
        progress.append(progress)
        off_path.append(z)
    return torch.stack(progress), torch.stack(off_path)


def participation_ratio(transition: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return the effective number of pairs carrying ``||delta_d_star||``.

    A value near 1 means one pair holds the entire norm, so ``the progress coordinate`` is a one-pair coordinate and the
    protein should leave the pool. Measured on CryptoBench at a 12 Angstrom cutoff, roughly a fifth
    of proteins fall below 3.
    """
    contributions = weights * transition * transition
    total = torch.sum(contributions, dim=-1)
    squared = torch.sum(contributions * contributions, dim=-1)
    return torch.where(squared > 0, total * total / squared, torch.zeros_like(total))


#: Distance differences below this are within crystallographic coordinate uncertainty and carry no
#: conformational information. Distance-difference-matrix practice reports differences in bands at
#: 0.5 and 1.0 Angstrom for exactly this reason.
COORDINATE_UNCERTAINTY_ANGSTROM = 0.5

#: Above this a distance difference is conventionally read as real conformational change.
SIGNIFICANT_CHANGE_ANGSTROM = 1.0


def transition_class_mask(
    pair_class: Sequence[str], transition_classes: Sequence[str] = TRANSITION_CLASSES
) -> torch.Tensor:
    """Return a boolean mask over ``P`` selecting the pairs that carry a target.

    Pass the SAME ``transition_classes`` given to :func:`build_pair_set`. The two must agree: the
    pairs this selects are exactly the ones whose ``delta_d_star`` was left alone, and everything it
    excludes was force-zeroed there.
    """
    unknown = set(transition_classes) - set(PAIR_CLASSES)
    if unknown:
        raise ValueError(f"unknown transition classes: {sorted(unknown)}")
    wanted = set(transition_classes)
    return torch.tensor([label in wanted for label in pair_class], dtype=torch.bool)


def transition_summary(
    transition: torch.Tensor, *, mask: torch.Tensor | None = None
) -> dict[str, float]:
    """Summarise one protein's experimental transition the way DDM practice reports it.

    ``rmsdd`` is the root-mean-square distance difference, and the two fractions are the share of
    pairs whose change exceeds coordinate uncertainty and the share that clears the conventional
    significance band. Together they answer the question the pair count cannot: whether this pocket
    moves at the BACKBONE level at all, or only through side chains that a C-alpha coordinate -- and
    BioEmu's backbone-level ensemble -- cannot represent.

    A pocket whose pairs sit almost entirely below 0.5 Angstrom is not backbone-reachable, and no
    amount of steering can express it.

    ``mask`` selects the pairs the three numbers are computed over, and callers holding a pair set
    MUST pass :func:`transition_class_mask`. All three are means over the pairs they see, and
    :func:`build_pair_set` force-zeroes ``delta_d_star`` outside ``TRANSITION_CLASSES``: summarising
    the whole vector therefore divides a numerator that only pocket pairs can contribute to by a
    denominator that also counts every context-context pair. Two proteins with the same pocket and
    different context radii then score differently, so the number partly measures ``R`` and the
    chain size rather than the pocket. ``participation_ratio`` and ``||v||`` are immune because the
    zeros drop out of their sums, which is how the defect survived a pool build.

    The all-pairs figures are recoverable exactly from the masked ones and the context-context
    share ``c`` (see :func:`context_context_fraction`): the fractions scale by ``1 - c`` and
    ``rmsdd`` by ``sqrt(1 - c)``, because the excluded pairs contribute nothing to any numerator.
    They are not reported separately, since neither the pocket nor its neighbourhood is what they
    measure.
    """
    selected = transition if mask is None else transition[mask.to(transition.device)]
    if selected.numel() == 0:
        return {"rmsdd": 0.0, "fraction_above_uncertainty": 0.0, "fraction_significant": 0.0}
    magnitude = selected.abs()
    return {
        "rmsdd": float(torch.sqrt(torch.mean(selected * selected))),
        "fraction_above_uncertainty": float(
            (magnitude > COORDINATE_UNCERTAINTY_ANGSTROM).to(selected.dtype).mean()
        ),
        "fraction_significant": float(
            (magnitude > SIGNIFICANT_CHANGE_ANGSTROM).to(selected.dtype).mean()
        ),
    }


def context_context_fraction(pairs: PocketPairSet) -> float:
    """Return ``n_context_context / |P|``: the share of the pair set that carries no target.

    It is the dilution factor of the defect :func:`transition_summary` documents, and it is a
    covariate in its own right -- it grows with the context radius ``R`` and with chain size, so any
    quantity averaged over the whole pair set inherits that dependence.
    """
    if len(pairs) == 0:
        return 0.0
    return pairs.pair_class.count("context_context") / len(pairs)


def pair_class_counts(pairs: PocketPairSet) -> dict[str, int]:
    """Return how many pairs each class holds, with every class present even at zero.

    ``class_decomposition`` gives each class's share of the squared transition; this gives the
    denominator that share was taken over. Without it a pool entry cannot answer "how many pairs
    carry a target", and a reader wanting an exact figure is left bracketing it from
    ``pair_count`` and ``context_context_fraction`` -- which recovers the context-context count
    but not the split of the remainder between ``pocket_pocket`` and ``pocket_context``.

    All three keys are always present, so a consumer never has to decide whether an absent key
    means zero pairs or an older pool.
    """
    return {name: pairs.pair_class.count(name) for name in PAIR_CLASSES}


#: Fewest pairs a class must hold before ITS share of a squared quantity is read as a per-protein
#: measurement (PLAN.md item 23).
#:
#: **Justification, measured.** On the 592-protein stage-1 train pool the ``pocket_pocket`` class
#: holds p10 = 4, p25 = 5, p50 = 10, p75 = 23, p90 = 46, p99 = 118 pairs, with 5 proteins at
#: exactly zero. A share taken over 4 pairs moves in visible jumps and has no useful sampling
#: distribution, so the drop from the pool null of about 0.43 towards 0 that would signal the
#: semi-rigid-displacement hack is a few pairs' worth of noise on such a protein. 23 is that
#: distribution's p75: the quarter of the pool on which the per-protein reading is sound. It is
#: deliberately a REPORTING floor and nothing else -- it selects no protein, filters no aggregate,
#: and enters no loss, checkpoint identity or objective.
#:
#: **Re-anchoring condition.** Re-measure this number whenever the pair set's own denominator
#: distribution changes: a new pool or split role, a change to ``cutoff_angstrom``,
#: ``context_radius_angstrom``, ``atom_set`` or the maximum residue count, or any change to how
#: :func:`classify_pairs` labels a pair. It is a quantile of a measured distribution and carries no
#: meaning away from the distribution it was measured on.
CLASS_SHARE_MINIMUM_PAIRS = 23

#: How a class share may be described given its denominator. ``empty`` is a structural zero -- the
#: class holds no pair, so the share is 0.0 by construction and is not a measurement at all;
#: ``thin`` is measured but below :data:`CLASS_SHARE_MINIMUM_PAIRS`; ``measurable`` is the reading
#: that may be quoted per protein.
CLASS_SHARE_DENOMINATOR_BANDS = ("empty", "thin", "measurable")


def class_share_denominator_band(
    pair_count: int, *, minimum_pairs: int = CLASS_SHARE_MINIMUM_PAIRS
) -> str:
    """Return which of :data:`CLASS_SHARE_DENOMINATOR_BANDS` a class share's denominator falls in.

    Three bands and not two, because "this protein has no pocket_pocket pair" and "this protein has
    four" fail differently: the first produces a share that is not a measurement, the second one
    that is imprecise. Collapsing them would let a structural zero be reported as a thin
    measurement, or a thin measurement be dropped as degenerate.
    """
    if minimum_pairs < 1:
        raise ValueError(f"minimum_pairs must be at least 1, got {minimum_pairs}")
    if pair_count <= 0:
        return "empty"
    return "thin" if pair_count < minimum_pairs else "measurable"


#: Per-coordinate error assumed when pricing how well the progress coordinate is determined. Deliberately pessimistic
#: for a resolution-filtered crystal structure, so the resulting figure is a lower bound on
#: separability rather than a flattering one.
ASSUMED_COORDINATE_ERROR_ANGSTROM = 0.5


def progress_noise(
    pairs: PocketPairSet,
    coordinates: torch.Tensor,
    *,
    coordinate_error: float | None = None,
) -> float:
    """Return ``sd(the progress coordinate)`` induced by independent Gaussian error on the scored structure.

    ``the progress coordinate`` is a weighted projection over ``P`` pair distances, but those distances are not
    independent measurements: they are functions of the same ``3N`` coordinates, with ``P`` in the
    hundreds and ``N`` between roughly 60 and 300. Perturbing one residue moves every pair it
    participates in, and because the transition is coherent -- neighbouring residues move together,
    with the same sign -- those per-pair sensitivities add constructively rather than cancelling.
    Pricing the noise pair-by-pair therefore understates it by a median factor of 1.6 on the
    CryptoBench pool and 1.3 on CryptoBank; see :func:`progress_noise_independent_bound`, which is
    that optimistic calculation.

    The exact linearised answer needs no independence assumption and no weighting algebra. ``the progress coordinate``
    is a scalar function of the coordinates, so a single backward pass gives its gradient, and for
    per-coordinate error ``sigma`` drawn independently on each of the ``3N`` components::

        sd(the progress coordinate) = sigma * ||d the progress coordinate / d x||_F

    ``coordinates`` is the ``[residues, 3]`` structure at which the linearisation is taken, in
    Angstrom, normally the experimental apo structure. The transition ``v`` and the reference
    distances are held at their deposited values, so this prices error in the ONE structure being
    scored; error in the reference structure adds its own variance on top, which the measurement
    script reports separately.

    This is the quantity that decides whether apo and holo are separable -- their span is 1 by
    construction, so ``1 / sd(the progress coordinate)`` is the separation in standard deviations. It remains a better
    eligibility criterion than the fraction of pairs clearing an absolute threshold, because a
    protein can have few individually significant pairs and still a well-determined coordinate.
    """
    error = ASSUMED_COORDINATE_ERROR_ANGSTROM if coordinate_error is None else coordinate_error
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must be [residues, 3] for one structure")
    if len(pairs) == 0:
        return float("inf")
    weights = pairs.weights.to(torch.float64)
    transition = pairs.delta_d_star.to(torch.float64)
    denominator = float(torch.sum(weights * transition * transition))
    if denominator <= 0:
        return float("inf")

    positions = coordinates.detach().to(torch.float64).clone().requires_grad_(True)
    distances = pair_distances(positions, pairs.pair_index)
    if float(distances.min()) <= 0:
        raise ValueError(
            "two paired residues share a coordinate; the distance gradient is undefined"
        )
    # The reference distances are a constant offset in the progress coordinate and drop out of the gradient, so they
    # are simply omitted here rather than passed in and subtracted.
    progress = torch.sum(weights * distances * transition) / denominator
    (gradient,) = torch.autograd.grad(progress, positions)
    return float(error * torch.linalg.vector_norm(gradient))


def progress_noise_independent_bound(
    pairs: PocketPairSet, coordinate_error: float | None = None
) -> float:
    """Return the OPTIMISTIC closed-form ``sd(the progress coordinate)``: a lower bound on the noise.

    Treating the ``P`` pair distances as independent measurements with error
    ``sigma_d = sqrt(2) * coordinate_error`` (two independently perturbed atoms per distance) gives

        sd(the progress coordinate) = sigma_d * sqrt(sum_p w_p^2 v_p^2) / sum_p w_p v_p^2

    which reduces to ``sigma_d / ||v||`` only when ``w`` is uniform -- and it never is, because
    per-class normalisation up-weights the smaller class by the count ratio.

    The independence assumption is false: the pairs are functions of ``3N`` shared coordinates and
    their sensitivities add constructively. So this is an UPPER bound on how well ``the progress coordinate`` is
    determined and a LOWER bound on its noise. Measured against :func:`progress_noise` at 0.5 A
    per-coordinate error it is optimistic on ``sd(the progress coordinate)`` by p10 1.17, p50 1.58, p90 1.99 over 491
    CryptoBench proteins and p50 1.28 over 600 CryptoBank proteins -- less than the ``sqrt(P/N)``
    of about 2.5 that fully coherent per-residue addition would give, because the transition is
    only partly coherent, but never in the safe direction.

    It is kept because it is cheap -- it needs only the pair set, not the coordinates -- and
    because a bound that is known to point one way is usable for triage. Never report it as the
    separation.
    """
    error = ASSUMED_COORDINATE_ERROR_ANGSTROM if coordinate_error is None else coordinate_error
    weights = pairs.weights.to(torch.float64)
    transition = pairs.delta_d_star.to(torch.float64)
    denominator = float(torch.sum(weights * transition * transition))
    if denominator <= 0:
        return float("inf")
    numerator = float(torch.sqrt(torch.sum(weights * weights * transition * transition)))
    return float(error * (2.0**0.5) * numerator / denominator)


#: Which transition vector a class decomposition is taken over. ``effective`` is the vector the
#: objective uses, with non-transition-class targets force-zeroed; ``raw`` is ``holo - apo`` as the
#: two crystal structures actually differ.
TRANSITION_VECTORS = ("effective", "raw")


def _class_shares(pair_class: Sequence[str], squared: torch.Tensor) -> dict[str, float]:
    """Return each class's share of a nonnegative per-pair quantity, or zeros if it vanishes."""
    total = float(squared.sum())
    if total <= 0:
        return dict.fromkeys(PAIR_CLASSES, 0.0)
    shares: dict[str, float] = {}
    for name in PAIR_CLASSES:
        mask = [index for index, label in enumerate(pair_class) if label == name]
        shares[name] = float(squared[mask].sum()) / total if mask else 0.0
    return shares


def class_decomposition(pairs: PocketPairSet, *, transition: str = "effective") -> dict[str, float]:
    """Return each pair class's share of the squared transition norm, summing to 1.

    Which class carries the transition is protein-specific and decides the class weights.
    Binding-site residues line a cavity from both sides, so pocket-pocket distances genuinely
    span the opening; but where a site is lined mostly by one peeling element, that element moves
    as a semi-rigid unit whose internal distances barely change while its distance to the body
    changes a great deal, and the signal lives in pocket-context instead.

    **The two vectors answer different questions and must not be quoted as one another.**

    * ``transition="effective"`` decomposes ``delta_d_star``, the vector ``the progress coordinate`` and the loss are
      defined against. ``build_pair_set`` force-zeroes every non-transition-class target, so
      ``context_context`` is **identically 0 by construction** here, on every protein. That is not
      a measurement of the geometry and it can never be one: it is a restatement of the design.
    * ``transition="raw"`` decomposes ``raw_delta_d_star``, which is ``holo - apo`` on every pair.
      Its ``context_context`` share is the real quantity -- how much of the apo-to-holo change sits
      in a shell the objective refuses to steer toward -- and it is what the decision to zero was
      taken on (median 0.143, dominant for 13.8% of proteins). A protein with a large raw context
      share has a fold rearranging around the pocket rather than a pocket opening within a fold.

    The raw vector is not stored in the pocket artifact, so a pair set reloaded from one carries
    ``raw_delta_d_star = None`` and ``transition="raw"`` raises rather than silently falling back
    to the zeroed vector.
    """
    if transition not in TRANSITION_VECTORS:
        raise ValueError(f"transition must be one of {TRANSITION_VECTORS}")
    if transition == "raw":
        if pairs.raw_delta_d_star is None:
            raise ValueError(
                "this pair set did not retain the raw transition, so its raw class shares cannot "
                "be recovered; rebuild it with build_pair_set rather than reading delta_d_star, "
                "which is the zeroed vector"
            )
        vector = pairs.raw_delta_d_star
    else:
        vector = pairs.delta_d_star
    return _class_shares(pairs.pair_class, (pairs.weights * vector * vector).detach())


def off_path_class_decomposition(pairs: PocketPairSet, realised: torch.Tensor) -> dict[str, float]:
    """Return each pair class's share of ``z^2``, summing to 1. REPORTING ONLY.

    ``z^2 = sum_p w_p (u_p - the progress coordinate v_p)^2`` is one number over three classes doing two different
    jobs. On a transition-class pair the residual is the part of the realised change that the
    apo-to-holo direction does not explain -- "the pocket moved, but not along the transition". On
    a context-context pair ``v_p = 0``, so the residual is the pair's whole realised change and
    ``z`` is measuring fold rigidity instead. Both belong in the penalty (that is exactly why
    context pairs sit inside ``P``: without them "open the pocket" is satisfied by locally
    unfolding it), but a single ``z`` cannot say which one a given protein's penalty is made of --
    and with the default class weights, context-context carries a third of the total weight mass
    however few or many pairs it holds.

    This changes nothing about ``z``, the loss or the weights. It reports the blend so that
    ``lambda_perp`` can be anchored knowing what it is buying, and so that a ``zeta`` quoted as
    "off-axis pocket motion" can be checked against how much of it is pocket motion at all.
    """
    if realised.shape != (len(pairs.pair_class),):
        raise ValueError("realised must carry one value per pair, for one structure")
    weights = pairs.weights.to(realised)
    transition = pairs.delta_d_star.to(realised)
    progress, _ = progress_coordinate(realised, transition, weights)
    residual = realised - progress * transition
    return _class_shares(pairs.pair_class, (weights * residual * residual).detach())


def transition_class_off_path(
    pairs: PocketPairSet,
    realised: torch.Tensor,
    *,
    transition_classes: Sequence[str] = TRANSITION_CLASSES,
) -> float:
    """Return ``zeta`` restricted to the pairs that carry a target. REPORTING ONLY.

    The pocket-only counterpart of :func:`normalised_off_path`: the off-path residual over
    pocket-pocket and pocket-context pairs alone, with their weights renormalised to sum to one,
    divided by ``||v||_w`` over those same pairs. It answers "how far off the apo-to-holo direction
    did the POCKET move", with the fold-rigidity contribution that dominates the weight mass of the
    full ``z`` removed.

    The renormalisation is a no-op on the ratio and is written out only because it makes the
    restricted quantity well defined on its own terms: scaling the restricted weights by any
    positive constant scales numerator and denominator alike. ``the progress coordinate`` is likewise unchanged by the
    restriction, because a context-context pair has ``v_p = 0`` and so enters neither ``<u, v>_w``
    nor ``||v||_w^2``. The consequence is an exact identity against
    :func:`off_path_class_decomposition`::

        zeta_pocket^2 = zeta^2 * (1 - share_context_context)

    so the two numbers recorded beside ``holo_off_path`` are one measurement expressed two ways,
    not two independent ones.

    Raises when the restricted transition is degenerate, matching
    :func:`progress_coordinate`'s refusal to divide by a vanishing transition.
    """
    mask = transition_class_mask(pairs.pair_class, transition_classes).to(realised.device)
    if realised.shape != (len(pairs.pair_class),):
        raise ValueError("realised must carry one value per pair, for one structure")
    weights = pairs.weights.to(realised)[mask]
    if weights.numel() == 0 or float(weights.sum()) <= 0:
        raise ValueError("no target-carrying pair survives the class restriction")
    weights = weights / weights.sum()
    transition = pairs.delta_d_star.to(realised)[mask]
    _, z = progress_coordinate(realised[mask], transition, weights)
    return float(normalised_off_path(z, transition, weights))


@dataclass(frozen=True)
class PocketArtifact:
    """The fingerprinted per-protein inputs the steering runtime consumes.

    The reference table schema used for ``mean``/``sigma`` is a closed set and cannot carry any of
    this, so it lives in its own artifact whose SHA-256 and content fingerprint bind into the
    training checkpoint identity -- and NOT into the schedule seed, or the conditioning arms would
    stop sharing noise draws and every cross-arm difference would be confounded with a complete
    re-randomisation.
    """

    schema_version: int
    cutoff_angstrom: float
    context_radius_angstrom: float
    #: The radius the pocket-local gate neighbourhood was grown at (schema 3). It sits beside the
    #: two radii above rather than among the gate's thresholds because it is part of what the gate
    #: MEANS and not a knob: it fixes WHICH residues are scored, so two runs that grew the
    #: neighbourhood at different radii are applying different gates. It is required, with no
    #: default, for the reason the artifact loader refuses an older file -- a silently defaulted
    #: radius would let two runs score different residue sets under one artifact identity.
    gate_radius_angstrom: float
    atom_set: str
    class_weights: dict[str, float]
    reference_kind: str
    specs: dict[str, PocketSpec]

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        if not self.gate_radius_angstrom > 0:
            raise ValueError("gate_radius_angstrom must be positive")
        if self.atom_set not in {"ca", "cb"}:
            raise ValueError("atom_set must be 'ca' or 'cb'")
        if self.reference_kind not in {"matched_unsteered_draw", "experimental_apo"}:
            raise ValueError(
                "reference_kind must be 'matched_unsteered_draw' (the training objective) or "
                "'experimental_apo' (the reported coordinate); they mean different things and a "
                "run may not silently use one while claiming the other"
            )
        if not self.specs:
            raise ValueError("a pocket artifact must describe at least one sequence")
