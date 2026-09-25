"""Differentiable coordinate-only property surrogates for steering."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

MIN_SEQUENCE_SEPARATION = 3


def _validate_coordinates(coordinates: torch.Tensor, mask: torch.Tensor) -> None:
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3 or not coordinates.is_floating_point():
        raise ValueError("coordinates must be a floating [batch, residues, 3] tensor")
    if mask.ndim != 2 or mask.dtype is not torch.bool:
        raise ValueError("mask must be a boolean [batch, residues] tensor")
    if coordinates.shape[:2] != mask.shape:
        raise ValueError("coordinates and mask shapes must agree")
    if coordinates.device != mask.device:
        raise ValueError("coordinates and mask must be on the same device")
    if not torch.isfinite(coordinates).all():
        raise ValueError("coordinate elements must be finite")
    if torch.any(mask.sum(dim=1) < 2):
        raise ValueError("each sequence must contain at least two valid residues")


def _masked_coordinates(
    coordinates: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_coordinates(coordinates, mask)
    weights = mask.unsqueeze(-1).to(dtype=coordinates.dtype)
    values = torch.where(mask.unsqueeze(-1), coordinates, torch.zeros_like(coordinates))
    return values, weights


def _geometry_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
    """Promote low-precision coordinates for stable geometric invariants."""
    return coordinates if coordinates.dtype == torch.float64 else coordinates.to(torch.float64)


def differentiable_radius_of_gyration(
    coordinates: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return masked radii of gyration in the input coordinate unit (angstrom)."""
    values, weights = _masked_coordinates(_geometry_coordinates(coordinates), mask)
    count = weights.sum(dim=1)
    centroid = values.sum(dim=1) / count
    centered = torch.where(mask.unsqueeze(-1), values - centroid.unsqueeze(1), 0.0)
    mean_square_distance = centered.square().sum(dim=-1).sum(dim=1) / count.squeeze(-1)
    positive = mean_square_distance > 0
    safe_mean_square_distance = torch.where(
        positive, mean_square_distance, torch.ones_like(mean_square_distance)
    )
    radius = torch.where(
        positive, torch.sqrt(safe_mean_square_distance), torch.zeros_like(mean_square_distance)
    )
    return radius.to(dtype=coordinates.dtype)


def differentiable_end_to_end_distance(
    coordinates: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return the distance between the first and last valid residue in angstrom."""
    _validate_coordinates(coordinates, mask)
    first_index = mask.to(dtype=torch.int64).argmax(dim=1)
    last_index = mask.shape[1] - 1 - mask.flip(dims=(1,)).to(dtype=torch.int64).argmax(dim=1)
    batch_index = torch.arange(mask.shape[0], device=coordinates.device)
    return torch.linalg.vector_norm(
        coordinates[batch_index, last_index] - coordinates[batch_index, first_index], dim=-1
    )


def differentiable_relative_shape_anisotropy(
    coordinates: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return the masked relative shape anisotropy of each gyration tensor.

    The trace invariant is algebraically equivalent to the eigenvalue definition,
    while avoiding undefined eigendecomposition gradients for repeated eigenvalues.
    """
    values, weights = _masked_coordinates(_geometry_coordinates(coordinates), mask)
    count = weights.sum(dim=1)
    centroid = values.sum(dim=1) / count
    centered = torch.where(mask.unsqueeze(-1), values - centroid.unsqueeze(1), 0.0)
    gyration = torch.matmul(centered.transpose(1, 2), centered) / count.unsqueeze(-1)
    trace = torch.diagonal(gyration, dim1=-2, dim2=-1).sum(dim=-1)
    trace_square = torch.diagonal(torch.matmul(gyration, gyration), dim1=-2, dim2=-1).sum(dim=-1)
    positive = trace > 0
    safe_trace_squared = torch.where(positive, trace.square(), torch.ones_like(trace))
    anisotropy = (3.0 * trace_square - trace.square()) / (2.0 * safe_trace_squared)
    anisotropy = torch.where(positive, anisotropy.clamp(0.0, 1.0), torch.zeros_like(anisotropy))
    return anisotropy.to(dtype=coordinates.dtype)


class SmoothDistanceFeatures(nn.Module):
    """Fixed-width smooth eligible-contact densities from residue coordinates."""

    def __init__(
        self,
        cutoffs: tuple[float, ...] = (6.0, 8.0, 10.0, 12.0),
        temperature: float = 1.0,
        min_sequence_separation: int = MIN_SEQUENCE_SEPARATION,
    ) -> None:
        super().__init__()
        if not cutoffs or any(not math.isfinite(cutoff) or cutoff <= 0 for cutoff in cutoffs):
            raise ValueError("cutoffs must be finite positive distances")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not isinstance(min_sequence_separation, int) or min_sequence_separation < 1:
            raise ValueError("min_sequence_separation must be a positive integer")
        self.cutoffs = tuple(float(cutoff) for cutoff in cutoffs)
        self.temperature = float(temperature)
        self.min_sequence_separation = min_sequence_separation

    def forward(self, coordinates: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _validate_coordinates(coordinates, mask)
        offsets = torch.arange(mask.shape[1], device=coordinates.device)
        eligible = (
            mask.unsqueeze(1)
            & mask.unsqueeze(2)
            & (offsets[:, None] - offsets[None, :]).abs().ge(self.min_sequence_separation)
            & torch.triu(torch.ones_like(mask.unsqueeze(2) & mask.unsqueeze(1)), diagonal=1)
        )
        distances = torch.linalg.vector_norm(
            coordinates.unsqueeze(2) - coordinates.unsqueeze(1), dim=-1
        )
        denominator = eligible.sum(dim=(1, 2)).to(dtype=coordinates.dtype)
        safe_denominator = denominator.clamp_min(1.0)
        contact_features = tuple(
            (torch.sigmoid((cutoff - distances) / self.temperature) * eligible).sum(dim=(1, 2))
            / safe_denominator
            for cutoff in self.cutoffs
        )
        return torch.stack(contact_features, dim=-1)


class CoordinatePropertySurrogate(nn.Module):
    """Two-hidden-layer MLP predicting a property from sequence and coordinates only."""

    def __init__(
        self, hidden_width: int, amino_acid_vocab_size: int = 20, contact_temperature: float = 1.0
    ) -> None:
        super().__init__()
        if not isinstance(hidden_width, int) or hidden_width <= 0:
            raise ValueError("hidden_width must be a positive integer")
        if not isinstance(amino_acid_vocab_size, int) or amino_acid_vocab_size <= 0:
            raise ValueError("amino_acid_vocab_size must be a positive integer")
        self.amino_acid_vocab_size = amino_acid_vocab_size
        self.distance_features = SmoothDistanceFeatures(temperature=contact_temperature)
        input_width = 3 + len(self.distance_features.cutoffs) + amino_acid_vocab_size + 1
        self.network = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(
        self, coordinates: torch.Tensor, sequence_tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        _validate_coordinates(coordinates, mask)
        if sequence_tokens.shape != mask.shape or sequence_tokens.ndim != 2:
            raise ValueError("sequence_tokens and mask shapes must agree")
        if sequence_tokens.device != coordinates.device or sequence_tokens.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("sequence_tokens must be an integer tensor on the coordinate device")
        valid_tokens = sequence_tokens[mask]
        if torch.any(valid_tokens < 0) or torch.any(valid_tokens >= self.amino_acid_vocab_size):
            raise ValueError("valid sequence_tokens must be amino-acid vocabulary indices")
        composition = torch.nn.functional.one_hot(
            valid_tokens, num_classes=self.amino_acid_vocab_size
        ).to(dtype=coordinates.dtype)
        batch_index = torch.arange(mask.shape[0], device=coordinates.device).repeat_interleave(
            mask.sum(dim=1)
        )
        composition_summary = torch.zeros(
            mask.shape[0],
            self.amino_acid_vocab_size,
            device=coordinates.device,
            dtype=coordinates.dtype,
        ).index_add_(0, batch_index, composition)
        sequence_length = mask.sum(dim=1).to(dtype=coordinates.dtype)
        composition_summary = composition_summary / sequence_length.unsqueeze(-1)
        features = torch.cat(
            (
                differentiable_radius_of_gyration(coordinates, mask).unsqueeze(-1),
                differentiable_end_to_end_distance(coordinates, mask).unsqueeze(-1),
                differentiable_relative_shape_anisotropy(coordinates, mask).unsqueeze(-1),
                self.distance_features(coordinates, mask),
                composition_summary,
                sequence_length.unsqueeze(-1),
            ),
            dim=-1,
        )
        return self.network(features).squeeze(-1)


@dataclass(frozen=True)
class SurrogateValidation:
    """Sequence-grouped held-out validation metrics for a surrogate checkpoint."""

    sequence_spearman: float
    mae: float
    calibration_slope: float
    sequence_count: int
    grouping_key: str = "sequence_id"
    aggregation: str = "per_sequence"

    def __post_init__(self) -> None:
        for name in ("sequence_spearman", "mae", "calibration_slope"):
            value = getattr(self, name)
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if not -1.0 <= self.sequence_spearman <= 1.0:
            raise ValueError("sequence_spearman must be in [-1, 1]")
        if self.mae < 0:
            raise ValueError("mae must be non-negative")
        if (
            not isinstance(self.sequence_count, int)
            or isinstance(self.sequence_count, bool)
            or self.sequence_count <= 0
        ):
            raise ValueError("sequence_count must be positive")
        if self.grouping_key != "sequence_id":
            raise ValueError("sequence grouping metadata must use sequence_id")
        if self.aggregation != "per_sequence":
            raise ValueError("sequence aggregation metadata must use per_sequence")

    def checkpoint_metadata(
        self, min_spearman: float, max_mae: float, calibration_slope_range: tuple[float, float]
    ) -> dict[str, object]:
        """Return metrics and acceptance thresholds for checkpoint serialization."""
        _validate_acceptance_thresholds(min_spearman, max_mae, calibration_slope_range)
        return {
            "sequence_spearman": self.sequence_spearman,
            "mae": self.mae,
            "calibration_slope": self.calibration_slope,
            "sequence_count": self.sequence_count,
            "grouping_key": self.grouping_key,
            "aggregation": self.aggregation,
            "min_spearman": min_spearman,
            "max_mae": max_mae,
            "calibration_slope_range": list(calibration_slope_range),
        }


def _validate_acceptance_thresholds(
    min_spearman: float, max_mae: float, calibration_slope_range: tuple[float, float]
) -> None:
    if any(
        not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value)
        for value in (min_spearman, max_mae)
    ):
        raise ValueError("acceptance thresholds must be finite")
    if not -1.0 <= min_spearman <= 1.0:
        raise ValueError("min_spearman must be in [-1, 1]")
    if max_mae < 0:
        raise ValueError("max_mae must be non-negative")
    if not isinstance(calibration_slope_range, tuple) or len(calibration_slope_range) != 2:
        raise ValueError("calibration_slope_range must contain two finite values")
    lower, upper = calibration_slope_range
    if (
        any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in (lower, upper)
        )
        or lower > upper
    ):
        raise ValueError("calibration_slope_range must contain ordered finite values")


def require_valid_surrogate(
    validation: SurrogateValidation,
    min_spearman: float,
    max_mae: float,
    calibration_slope_range: tuple[float, float],
) -> None:
    """Fail closed unless held-out sequence-grouped metrics meet every threshold."""
    if not isinstance(validation, SurrogateValidation):
        raise ValueError("validation must be a SurrogateValidation")
    validation.__post_init__()
    _validate_acceptance_thresholds(min_spearman, max_mae, calibration_slope_range)
    lower, upper = calibration_slope_range
    if (
        validation.sequence_spearman < min_spearman
        or validation.mae > max_mae
        or not lower <= validation.calibration_slope <= upper
    ):
        raise ValueError("surrogate validation does not meet acceptance thresholds")


#: Flat-bottom windows for C-alpha separations at sequence offsets 1, 2 and 3, in angstrom.
#:
#: Offset 1 is the **textbook** trans-peptide window, not a fitted one. An earlier version used
#: the 1st/99th percentiles of unsteered BioEmu output, ``[3.58, 4.09]``; that encodes the
#: model's own looseness as permitted -- 1% of its bonds exceed 4.09 A, which is the model
#: being sloppy rather than physics being permissive -- and invites the objection that the
#: reference is the thing being corrected. Measured, the two are equivalent: on real
#: structures the textbook window separates steered from unsteered by 634x and the percentile
#: window by 672x, while charging unsteered output 0.0043 against 0.0038. So the round number
#: costs nothing and is defensible from stereochemistry.
#:
#: Offsets 2 and 3 have no single textbook value because they are genuinely bimodal -- offset 2
#: separates helix (~5.4) from strand (~6.7), offset 3 helix (~5.0) from strand (~10) -- so
#: these stay empirical, rounded, from 33,000 separations over 200 unsteered structures
#: spanning 40 held-out sequences. Offset 2 is the C-alpha stand-in for a bond ANGLE: three
#: points at fixed 3.8 A spacing admit any offset-2 distance from ~0 to 7.6 A depending on the
#: chain angle, which is why AlphaFold2 carries an explicit angle term alongside its bond term.
#: Measured on steered structures, 97.5% of positions with both flanking bonds valid also have
#: a valid offset 2, so it is close to redundant against the *observed* failure -- it is kept
#: as insurance against an optimiser that satisfies bond lengths while collapsing angles.
CA_SEPARATION_WINDOWS: dict[int, tuple[float, float]] = {
    1: (3.6, 4.0),
}

#: Offsets 2 and 3 were measured and briefly used, and are now deleted rather than kept as a
#: legacy constant. They bought about 2.5% over offset 1 -- on steered structures, 97.5% of
#: positions with both flanking bonds valid already had a valid offset 2 -- and they never did
#: the job they appeared to do: a window admitting both helix (~5.4, ~5.0 A) and strand (~6.7,
#: ~10.0 A) separations lets a helix slide into a strand at zero cost. Preserving the fold is
#: :func:`smooth_lddt_loss`'s job, against a reference, which is the only thing that can tell
#: those two apart. Passing explicit ``windows=`` still works for any offset.

#: Steric floor for C-alpha pairs separated by more than :data:`CLASH_OFFSET` in sequence.
#: Taken from BioEmu's own ``ChainClashPotential`` default rather than invented here, so the
#: penalty asks for exactly what the frozen model's shipped physicality steering asks for.
#: Its ``log_physicality`` diagnostic flags clashes below 3.4 A at separation >= 4.
CLASH_FLOOR = 4.1
CLASH_OFFSET = 3


def ca_geometry_penalty(
    coordinates: torch.Tensor,
    mask: torch.Tensor,
    windows: Mapping[int, tuple[float, float]] | None = None,
    clash_floor: float | None = CLASH_FLOOR,
) -> torch.Tensor:
    """Return a flat-bottom C-alpha geometry violation, one scalar per batch member.

    This is the AlphaFold2 ``structural violation`` pattern restricted to a C-alpha trace: a
    squared hinge that is **exactly zero inside the allowed window** and grows quadratically
    outside it. Being flat-bottomed matters -- a harmonic restraint toward a single ideal
    length would also penalise the helix/strand spread that real structures need, and offsets
    2 and 3 are genuinely bimodal.

    The measured failure it exists to prevent: at ``o = +1`` the steered arm reached a mean
    consecutive separation of 6.32 A against 3.80 A, with only 13.8% of bonds inside the
    physical window, and the deformation tracked the requested dose.

    Differentiable in ``coordinates``, which must be angstrom.
    """
    selected = windows or CA_SEPARATION_WINDOWS
    values = _geometry_coordinates(coordinates)
    if mask.shape != values.shape[:2]:
        raise ValueError("mask must be a [batch, residues] boolean tensor")
    total = values.new_zeros(values.shape[0])
    weight = values.new_zeros(values.shape[0])
    for offset, (low, high) in sorted(selected.items()):
        if offset < 1 or offset >= values.shape[1]:
            continue
        if not 0 < low < high:
            raise ValueError(f"window for offset {offset} must satisfy 0 < low < high")
        separation = torch.linalg.vector_norm(
            values[:, offset:, :] - values[:, :-offset, :], dim=-1
        )
        # Both residues of the pair must be real, or a padded tail invents violations.
        pair = (mask[:, offset:] & mask[:, :-offset]).to(values.dtype)
        excess = torch.clamp(separation - high, min=0.0) + torch.clamp(low - separation, min=0.0)
        total = total + (excess.square() * pair).sum(dim=1)
        weight = weight + pair.sum(dim=1)

    violation = total / torch.clamp(weight, min=1.0)

    if clash_floor is not None and clash_floor > 0:
        # Non-adjacent pairs only: sequence-local pairs are governed by the windows above, and
        # including them here would double-penalise the same geometry.
        #
        # Normalised on its own denominator and *added*, not pooled with the windows. There are
        # O(L^2) non-bonded pairs against O(L) window pairs, so a shared denominator lets the
        # clash count swamp the windows -- measured, that diluted the steered-versus-unsteered
        # ratio from 135x to 79x and shrank the absolute penalty ~26-fold.
        pairwise = torch.cdist(values, values)
        indices = torch.arange(values.shape[1], device=values.device)
        far = (indices[:, None] - indices[None, :]).abs() > CLASH_OFFSET
        valid = ((mask[:, :, None] & mask[:, None, :]) & far).to(values.dtype)
        overlap = torch.clamp(clash_floor - pairwise, min=0.0)
        clash = (overlap.square() * valid).sum(dim=(1, 2)) / torch.clamp(
            valid.sum(dim=(1, 2)), min=1.0
        )
        violation = violation + clash

    return violation


#: Sequence separation window for the local-distance term. Secondary structure lives at
#: ``|i - j| <= 4`` (a helix turn is i,i+3 and i,i+4; a strand's register is i,i+2), so 8 covers
#: it with margin while staying far short of tertiary contacts.
#:
#: This is a deliberate departure from AlphaFold3, which restricts smooth LDDT to pairs within a
#: **spatial** cutoff of the reference. A spatial cutoff would fight this objective directly:
#: expanding the radius of gyration is exactly what the adapter is asked to do, and it
#: necessarily stretches pairs that were close in the reference. Restricting by *sequence*
#: separation instead preserves the local fold while leaving global rearrangement free.
LDDT_SEQUENCE_WINDOW = 8

#: AlphaFold3's smooth-LDDT thresholds (Algorithm 27), in angstrom.
LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def smooth_lddt_loss(
    coordinates: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
    *,
    window: int = LDDT_SEQUENCE_WINDOW,
) -> torch.Tensor:
    """Return ``1 - smooth_lddt`` between a steered trace and its unsteered reference.

    AlphaFold3's auxiliary structure loss, restricted to sequence-local pairs. For each pair
    the deviation ``delta = |d_pred - d_ref|`` is scored by a smooth step,

        eps = 1/4 * sum_t sigmoid(t - delta)   for t in (0.5, 1, 2, 4) angstrom

    which falls toward 0 as the distance drifts. Note ``sigmoid(t - 0) != 1``: at perfect
    preservation the raw AF3 score is ``mean_t sigmoid(t) = 0.804``, not 1, so a naive
    ``1 - mean(eps)`` carries a **0.196 floor that every structure pays**. That would destroy
    the property these terms are built on -- costing nothing for geometry that is already
    correct -- so the score is normalised by its own value at zero deviation. A structure
    compared with itself then scores exactly 0.

    The reference must be the **unsteered structure from the same noise draw**, or the term
    measures sampling variation rather than the intervention. Both inputs are angstrom.

    Why this exists: the flat-bottom windows enforce that a chain is physically *possible*,
    not that it is the *same fold*. An offset-3 separation of 5.0 A (helix) can slide to 10 A
    (strand) entirely inside the allowed window. This term is what preserves secondary
    structure, and it is why the window terms beyond offset 1 become redundant.
    """
    values = _geometry_coordinates(coordinates)
    target = _geometry_coordinates(reference)
    if values.shape != target.shape:
        raise ValueError("coordinates and reference must share a shape")
    if mask.shape != values.shape[:2]:
        raise ValueError("mask must be a [batch, residues] boolean tensor")
    if window < 1:
        raise ValueError("window must be a positive sequence separation")

    total = values.new_zeros(values.shape[0])
    weight = values.new_zeros(values.shape[0])
    for offset in range(1, min(window, values.shape[1] - 1) + 1):
        predicted = torch.linalg.vector_norm(values[:, offset:, :] - values[:, :-offset, :], dim=-1)
        expected = torch.linalg.vector_norm(target[:, offset:, :] - target[:, :-offset, :], dim=-1)
        pair = (mask[:, offset:] & mask[:, :-offset]).to(values.dtype)
        deviation = (predicted - expected).abs()
        score = sum(torch.sigmoid(t - deviation) for t in LDDT_THRESHOLDS) / len(LDDT_THRESHOLDS)
        total = total + (score * pair).sum(dim=1)
        weight = weight + pair.sum(dim=1)
    # mean_t sigmoid(t): the raw score at zero deviation, 0.804 for AF3's thresholds.
    perfect = sum(1 / (1 + math.exp(-t)) for t in LDDT_THRESHOLDS) / len(LDDT_THRESHOLDS)
    return 1.0 - (total / torch.clamp(weight, min=1.0)) / perfect
