"""Contracts for constrained representation-steering training and resume state."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn import functional as functional

_CHECKPOINT_SCHEMA_VERSION = 2
_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "complete",
        "manifest_fingerprint",
        "adapter_config",
        "property_callback_identity",
        "source_checkpoints",
        "checkpoint_identity_fingerprint",
        "adapter_state",
        "optimizer_state",
        "epoch",
        "unit_cursor",
        "validation_record",
    }
)


@dataclass(frozen=True)
class TargetRange:
    """Inclusive training range for a scalar property target."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        for name, value in (("lower", self.lower), ("upper", self.upper)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.lower >= self.upper:
            raise ValueError("target range lower bound must be less than upper bound")

    def validate(self, targets: torch.Tensor, *, allow_extrapolation: bool = False) -> None:
        if not isinstance(allow_extrapolation, bool):
            raise ValueError("allow_extrapolation must be a boolean")
        if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
            raise ValueError("targets must be a floating tensor")
        if not torch.isfinite(targets).all():
            raise ValueError("targets must be finite")
        if not allow_extrapolation and torch.any((targets < self.lower) | (targets > self.upper)):
            raise ValueError("targets must remain within the training range")


def sample_training_targets(
    baseline: torch.Tensor, bounds: TargetRange, offsets: Sequence[float]
) -> torch.Tensor:
    """Return ``[batch, offset]`` targets with one symmetry-preserving scale per baseline."""
    if not isinstance(bounds, TargetRange):
        raise ValueError("bounds must be a TargetRange")
    if baseline.ndim != 1 or not baseline.is_floating_point():
        raise ValueError("baseline must be a floating [batch] tensor")
    bounds.validate(baseline)
    normalized_offsets = _symmetric_offsets(offsets)
    offset_tensor = baseline.new_tensor(normalized_offsets)
    maximum_offset = offset_tensor.abs().max()
    scale = torch.minimum(
        (baseline - bounds.lower) / maximum_offset,
        (bounds.upper - baseline) / maximum_offset,
    )
    scale = torch.minimum(scale, torch.ones_like(scale)).clamp_min(0)
    targets = baseline[:, None] + scale[:, None] * offset_tensor[None, :]
    bounds.validate(targets)
    return targets


def _symmetric_offsets(offsets: Sequence[float]) -> tuple[float, ...]:
    if isinstance(offsets, str | bytes) or not offsets:
        raise ValueError("offsets must be a non-empty symmetric sequence")
    values = tuple(offsets)
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("offsets must be finite real values")
    if not any(value < 0 for value in values) or not any(value > 0 for value in values):
        raise ValueError("offsets must include both negative and positive signs")
    if Counter(values) != Counter(-value for value in values):
        raise ValueError("offsets must be symmetric around zero")
    if 0.0 not in values:
        raise ValueError("offsets must include the baseline offset zero")
    return tuple(float(value) for value in values)


#: What the property term divides the error by. **Declared by the caller, never inferred from
#: whether a tensor happens to have arrived.** These live here, next to the objective that
#: enforces them, and ``ordinary_workflow`` imports them so the config, the checkpoint identity
#: and the loss cannot spell one normalisation four ways.
#:
#: ``sequence_sigma`` is D4b: the error is divided by that sequence's own conformational spread,
#: so a floppy protein cannot outweigh a rigid one at equal relative error. Radius of gyration
#: uses it -- on BOTH channels -- and its numbers must not move. ``apo_holo_transition`` is the
#: two-state form as D-64 left it: ``the progress coordinate`` is unit-length apo->holo for every protein by
#: construction, so the transition itself is the per-protein normaliser and NEITHER channel
#: carries a further divisor. ``property_units`` is the absolute path -- AF3 and Local-Q --
#: where the error is in the property's own raw units and there is no per-observation divisor
#: at all.
#:
#: ``per_channel_sigma_transition`` is D-65, and it is the two-state path's normalisation as of
#: **The two channels do not carry the same noise, so they do not get the same
#: divisor**:
#:
#: * ``S_abs = sigma_the progress coordinate``. Under the state target the matched control CANCELS --
#:   ``residual = (the progress coordinate_s - the progress coordinate_c) - (the progress coordinate_state(o) - the progress coordinate_c) = the progress coordinate_s - the progress coordinate_state(o)`` -- so the absolute
#:   residual is reporting-frame ABSOLUTE and carries the draw's full baseline noise. Measured at
#:   ``sigma_the progress coordinate = 0.37``: ``sd(residual)`` is 0.0000 under D-46's displacement target and
#:   1.00 sigma under D-64's state target. Inverse-variance weighting applies to it.
#: * ``S_pair = 1``. The paired gap is formed from two levels of ONE noise draw, so the baseline
#:   cancels identically and the gap carries no sigma noise to divide out. Dividing it would
#:   downweight by ``1/sigma`` exactly the high-sigma proteins the contrastive term exists to
#:   rescue.
#:
#: The four are not interchangeable and the differences are not cosmetic: ``sequence_sigma``,
#: ``apo_holo_transition`` and ``per_channel_sigma_transition`` differ by factors of ``sigma^2``
#: in the loss VALUE, term by term, and ``property_units`` is a different quantity again. Before
#: this was declared, the objective read the normalisation off ``property_scale is None``, so a
#: sigma run that lost its scale was indistinguishable from a transition run that never had one,
#: and an absolute run silently acquired an unnormalised paired gap in raw property units.
#:
#: **Per-channel is a FOURTH declaration and not a flag on the other three.** The alternative --
#: ``apo_holo_transition`` plus a boolean -- would let one declared normalisation name two
#: objectives, so a checkpoint identity and a report could describe a hybrid that never ran. One
#: string names one construction, and ``_require_declared_scale`` enforces exactly that string's
#: inputs.
SIGMA_LOSS_NORMALIZATION = "sequence_sigma"
TRANSITION_LOSS_NORMALIZATION = "apo_holo_transition"
ABSOLUTE_LOSS_NORMALIZATION = "property_units"
PER_CHANNEL_LOSS_NORMALIZATION = "per_channel_sigma_transition"
#: The FIFTH declaration, for a direct metric normalised by each system's own REFERENCE
#: SEPARATION -- the Angstrom distance between that system's two deposited conformers.
#:
#: Its contract is identical to :data:`SIGMA_LOSS_NORMALIZATION`'s -- a required, strictly
#: positive, per-observation divisor applied to the ABSOLUTE channel and to no other -- and it is
#: a separate declaration anyway, because the name is part of a fingerprinted identity and
#: ``sequence_sigma`` would be a false one. ``sigma_seq`` is the spread of a property over a
#: sequence's own unsteered ensemble; ``s_p`` is a distance between two crystals. A run recorded
#: under the wrong name is a run whose report says its errors were measured in multiples of a
#: conformational spread when they were measured in multiples of a transition length, and that is
#: the kind of thing that costs a re-measurement rather than a rename.
#:
#: It is deliberately NOT in :data:`_PAIRED_DOSE_NORMALIZATIONS`. The two levels of a
#: direct-metric draw measure distances to DIFFERENT reference conformers, so their difference is
#: not a dose gap and there is no paired channel for a divisor to apply to.
SEPARATION_LOSS_NORMALIZATION = "reference_separation"
PROPERTY_LOSS_NORMALIZATIONS = frozenset(
    {
        SIGMA_LOSS_NORMALIZATION,
        TRANSITION_LOSS_NORMALIZATION,
        ABSOLUTE_LOSS_NORMALIZATION,
        PER_CHANNEL_LOSS_NORMALIZATION,
        SEPARATION_LOSS_NORMALIZATION,
    }
)

#: The normalisations that require a per-observation ``property_scale``, because their ABSOLUTE
#: channel divides by it. ``per_channel_sigma_transition`` is here and ``apo_holo_transition`` is
#: not, which is the whole of D-65 in the input contract.
_SIGMA_SCALED_NORMALIZATIONS = frozenset(
    {
        SIGMA_LOSS_NORMALIZATION,
        PER_CHANNEL_LOSS_NORMALIZATION,
        SEPARATION_LOSS_NORMALIZATION,
    }
)

#: The normalisations under which the paired dose term is defined. The gap it forms is a
#: DIFFERENCE of two property values, so it is only comparable across proteins once each of
#: them is measured in a per-protein unit: sigma supplies one, the apo->holo transition supplies
#: the other, and raw property units supply none. ``paired_dose_loss`` used to enforce this by
#: refusing ``property_scale=None`` outright, which also refused the transition path.
_PAIRED_DOSE_NORMALIZATIONS = frozenset(
    {
        SIGMA_LOSS_NORMALIZATION,
        TRANSITION_LOSS_NORMALIZATION,
        PER_CHANNEL_LOSS_NORMALIZATION,
    }
)

#: The normalisations whose PAIRED channel divides by ``property_scale``. Only the fully
#: sigma-normalised one does: radius of gyration, whose paired term has always been
#: ``((Rg_i - Rg_j) - (t_i - t_j))/sigma_seq`` and whose published numbers must not move.
_SIGMA_SCALED_PAIRED_NORMALIZATIONS = frozenset({SIGMA_LOSS_NORMALIZATION})


@dataclass(frozen=True)
class SteeringLossComponents:
    total: torch.Tensor
    target: torch.Tensor
    residual: torch.Tensor
    geometry: torch.Tensor
    #: The paired dose term. Exactly zero when ``lambda_c`` is zero or no pair was present,
    #: so a run without pairing reports the same components it always did.
    contrastive: torch.Tensor | None = None
    #: Smooth LDDT against the unsteered structure from the same draw. ``None`` when no
    #: reference was generated, which is every run before the term existed.
    local_structure: torch.Tensor | None = None

    def journal_columns(self) -> dict[str, float | None]:
        """Return this objective's extra persisted journal columns, by column name.

        Empty here, and that emptiness is the contract: the shared components carry exactly
        the radius-of-gyration/AF3/Local-Q terms, so a shared journal row is byte-identical
        to one written before any other project existed. A project that adds a term subclasses
        this and returns its own columns -- see
        :class:`sae_bioemu.cryptic_steering.loss.CrypticSteeringLossComponents`.

        The SHARED objective's own four weighted terms are reported by
        :meth:`loss_term_columns` and deliberately not here. This method is the SUBCLASS
        extension point and its emptiness on the base class is load-bearing -- a subclass
        overrides it without calling ``super()``, which is exactly what
        ``CrypticSteeringLossComponents`` does, so anything added here would be silently
        dropped by every existing subclass. Two methods, two justifications: a column here is
        gated by a weight the subclass declares in ``objective_config_extra()``, and a column
        there is gated by one of the shared weights ``_objective_config`` always records.
        """
        return {}

    def loss_term_columns(self) -> dict[str, float | None]:
        """Return the SHARED objective's own weighted terms, by journal column name.

        ``total = lambda_y*target + lambda_c*contrastive + lambda_r*residual +
        lambda_g*geometry + lambda_l*local_structure``, and three of those five terms were
        computed, weighted into ``total`` and then dropped by the journal. ``target`` and
        ``residual`` already have persisted columns; the three below did not, so a reader with
        the journal in hand could not decompose the number the adapter was actually descending.
        At the cryptic arms' ``lambda_c = 7.0`` against ``lambda_y = 0.01`` the invisible term
        is the DOMINANT one.

        ``None`` means the term did not run, and it is not "zero": the caller gates each column
        on the weight that prices it and omits a column whose weight is zero, which is what
        keeps a row's key set identical to every row written before these columns existed. See
        ``ordinary_workflow._LOSS_TERM_COLUMN_WEIGHTS``.

        ``contrastive`` is ``None`` under two different conditions -- ``lambda_c == 0``, and
        ``lambda_c > 0`` with no pair in this batch -- and the second is a value the objective
        prices at exactly zero rather than a missing measurement (see :func:`paired_dose_loss`).
        Separating the two is the gating caller's job, because only it knows the weight.

        The unit of ``contrastive_loss`` depends on the declared normalisation: divided by
        sigma under ``sequence_sigma`` and undivided under the two transition declarations
        (D-65). It is NOT split into two differently-named columns the way ``target_loss`` was,
        because ``property_loss_normalization`` is already an identity-bound field of the same
        checkpoint and, unlike ``target_loss``, this column has no reader written against an
        older meaning to mislead.
        """
        return {
            "geometry_loss": float(self.geometry.detach().reshape(())),
            "contrastive_loss": (
                None if self.contrastive is None else float(self.contrastive.detach().reshape(()))
            ),
            "local_structure_loss": (
                None
                if self.local_structure is None
                else float(self.local_structure.detach().reshape(()))
            ),
        }


class PropertyLossKind(str, Enum):
    MSE = "mse"
    HUBER = "huber"


class ResidualPenaltyKind(str, Enum):
    """Which residual quantity the parsimony term squares.

    ``FRACTION`` squares the raw masked residual fraction ``F = RMS(dh)/RMS(h)``, which is
    bounded by ``rho_max`` and therefore by ``rho_max**2`` after squaring: at
    ``rho_max = 0.10`` the whole term lives in ``[0, 0.01]``.

    ``FRACTION_OF_RHO_MAX`` squares ``F / rho_max`` instead, so the term spans ``[0, 1]``
    and is on the same footing as a sigma-normalized property term. It also makes
    ``lambda_r`` comparable across ``rho_max`` settings: under ``FRACTION`` alone,
    halving ``rho_max`` silently quadruples the regularizer's effective strength.

    ``RMS`` squares ``RMS(dh)`` directly. That has the activation's own units and no
    natural ``rho_max`` bound, so the normalization does not apply to it and it is
    deliberately left alone.

    ``RAW_FRACTION_OF_RHO_MAX`` squares the same ratio built from the *pre-cap* fraction the
    adapter asked for. The three kinds above all measure the residual after the tanh, which
    is bounded by ``rho_max`` -- so once the adapter saturates, the penalty is nearly constant
    and stops pulling back: at the 0.09999966 the paired arms reached, its gradient is
    attenuated about 47,000x, and a ``lambda_r`` large enough to overcome that would have
    suppressed the intervention entirely long beforehand. Measured over 3000 units,
    ``lambda_r`` of 0.01 and 0.03 only moved first saturation to units 1090 and 130; both
    finished pinned at the cap. Penalising the unsquashed request keeps a gradient that does
    not vanish, so this is the kind to use when the operating point must stay off the bound.
    """

    FRACTION = "fraction"
    FRACTION_OF_RHO_MAX = "fraction_of_rho_max"
    RMS = "rms"
    RAW_FRACTION_OF_RHO_MAX = "raw_fraction_of_rho_max"


#: The five weights EVERY objective on this surface prices, in the order they enter ``total``.
#: Floating-point addition is not associative, so the order is part of the objective.
SHARED_LOSS_TERMS: tuple[str, ...] = (
    "lambda_y",
    "lambda_r",
    "lambda_g",
    "lambda_c",
    "lambda_l",
)

#: The EXTENDED vocabulary: every weight any steerable property's objective prices, declared on
#: the shared surface so that turning one on is a CONFIG change rather than a signature change.
#:
#: **Declaring a weight here does not price it.** A subclass states which of these it actually
#: multiplies into ``total`` in :attr:`SteeringLoss.PRICED_EXTENDED_TERMS`, and a nonzero weight
#: outside that set is REFUSED at construction, by name, with the class that refused it. The
#: refusal is the whole reason the vocabulary can be declared generously: without it, a weight
#: declared and not priced would be accepted by the constructor, dropped from ``total``, dropped
#: from the objective identity and dropped from its journal column -- the exact shape that
#: voided six ``lambda_l`` arms, reintroduced wholesale.
#:
#: What this buys is that the plumbing is FIXED. The runtime schema, the constructor, the
#: identity emission and the journal gating already know every name below, so a property that
#: comes to price ``lambda_rigid`` or ``lambda_dihedral`` adds it to its own
#: ``PRICED_EXTENDED_TERMS`` and its own ``runtime_objective_arguments`` -- both in its own
#: package, inside its own fingerprint group -- and no shared file moves.
EXTENDED_LOSS_TERMS: tuple[str, ...] = (
    "lambda_perp",
    "lambda_perp_fraction",
    "lambda_rigid",
    "lambda_dihedral",
)

#: The complete vocabulary, shared plus extended.
STEERING_LOSS_TERMS: tuple[str, ...] = SHARED_LOSS_TERMS + EXTENDED_LOSS_TERMS


#: Units marker recorded next to the residual penalty, in the same spirit as
#: ``success_tolerance_units`` in the evaluation metrics.
RESIDUAL_PENALTY_UNITS = {
    ResidualPenaltyKind.FRACTION: "squared_residual_fraction",
    ResidualPenaltyKind.FRACTION_OF_RHO_MAX: "squared_fraction_of_rho_max",
    ResidualPenaltyKind.RMS: "squared_residual_rms",
    ResidualPenaltyKind.RAW_FRACTION_OF_RHO_MAX: "squared_raw_fraction_of_rho_max",
}


def _require_a_driving_weight(
    objective: type[SteeringLoss],
    lambda_y: float,
    additional: Mapping[str, float] | None,
) -> None:
    """Fail closed unless at least one of this objective's declared drivers is positive.

    This replaces ``if lambda_y <= 0: raise``, and the replacement is a WIDENING rather than a
    relaxation. The old line refused an objective with nothing pulling the adapter anywhere, which
    is still refused here for every property; what it ALSO refused, incidentally, was an objective
    driven by something other than the property target term. Radius of gyration and every shipped
    kinase arm have no other driver, so they keep refusing ``lambda_y = 0`` exactly as before --
    the answer is unchanged wherever the declaration is the default.

    Both directions are checked, because a one-directional check is how a driverless loss gets
    through: a declared driver whose value never arrives would be read as absent and could not
    excuse anything, and an UNDECLARED weight arriving here would let any number at all stand in
    for a driver. The set must match the declaration exactly.
    """
    declared = tuple(name for name in objective.DRIVING_WEIGHTS if name != "lambda_y")
    if "lambda_y" not in objective.DRIVING_WEIGHTS:
        raise ValueError(
            f"{objective.__name__} must declare lambda_y among its DRIVING_WEIGHTS: it is the "
            "shared property target term, which every objective carries and prices"
        )
    supplied = dict(additional or {})
    if set(supplied) != set(declared):
        raise ValueError(
            f"{objective.__name__} declares the additional driving weights {sorted(declared)} and "
            f"was constructed with {sorted(supplied)}. The two must agree exactly: a declared "
            "driver whose value never arrives cannot excuse lambda_y = 0, and an undeclared one "
            "would let any weight stand in for a driver"
        )
    values = {"lambda_y": lambda_y, **supplied}
    if not any(float(value) > 0.0 for value in values.values()):
        raise ValueError(
            f"{objective.__name__} prices no driving term: "
            + ", ".join(f"{name} = {float(value)}" for name, value in sorted(values.items()))
            + ". At least one must be positive. lambda_r, lambda_g and lambda_l are PENALTIES, "
            "so an objective made of them alone has its global minimum at the adapter doing "
            "nothing, and the run would train to exactly that while every other gate stayed green"
        )


class SteeringLoss(nn.Module):
    """Explicitly weighted property-target steering objective.

    ``property_loss_normalization`` DECLARES which quantity the property error is measured in,
    and ``forward`` requires the supplied ``property_scale`` to match that declaration:

    * :data:`ABSOLUTE_LOSS_NORMALIZATION` -- ``mean_n (prediction_n - target_n)^2`` in the
      property's own raw units, which AF3 and Local-Q keep. No ``property_scale`` may be passed,
      and ``lambda_c`` must be zero: a paired gap in raw property units is not comparable across
      proteins and is not a defined objective.
    * :data:`SIGMA_LOSS_NORMALIZATION` -- ``mean_n ((prediction_n - target_n) / sigma_n)^2``, and
      the paired gap divided by the same sigma. ``property_scale`` is REQUIRED; the
      relative-level BioEmu Rg run supplies each sequence's own sigma there (D4b).
    * :data:`TRANSITION_LOSS_NORMALIZATION` -- the two-state form D-64 introduced. ``the progress coordinate`` is
      unit-length apo->holo for every protein by construction, so BOTH channels are undivided
      and the target term is ``mean_n (the progress coordinate_n - t_n)^2``. No ``property_scale`` may be passed.
    * :data:`PER_CHANNEL_LOSS_NORMALIZATION` -- D-65, and what the two-state properties (the
      cryptic pocket's ``the progress coordinate``, the kinase DFG coordinate) run under today. The ABSOLUTE channel
      is ``mean_n ((the progress coordinate_n - t_n) / sigma_n)^2`` and the PAIRED channel is the undivided gap,
      because under the state target the matched control cancels out of the absolute residual
      (which therefore carries the full draw baseline) and cancels identically inside the gap
      (which therefore carries none). ``property_scale`` is REQUIRED, and it reaches the
      absolute channel only.

    The declaration is the whole point of the parameter. Inferring it from
    ``property_scale is None`` -- which is what this class did -- cannot separate
    "transition-normalised by design" from "a sigma path whose scale went missing" or from "an
    absolute path that was never normalised at all", and those three are three objectives. Nor
    can the presence of a scale separate ``sequence_sigma`` from
    ``per_channel_sigma_transition``, which are a fourth and fifth: both are handed the same
    tensor and differ only in whether the paired channel divides by it.

    ``rho_max`` is required by, and only by, ``ResidualPenaltyKind.FRACTION_OF_RHO_MAX``.
    It is the adapter's own intervention bound, so the residual term becomes the squared
    fraction of the budget actually spent. ``lambda_r`` remains a parsimony weight and is
    not a safety mechanism: the adapter's tanh cap already guarantees ``F <= rho_max``
    unconditionally, so this term only shapes behaviour *inside* that budget.

    **This class is the shared objective and holds no project's added terms.** It carries the
    radius-of-gyration, AF3 and Local-Q terms and nothing else, which is what lets a shared
    edit be read as a shared edit. A project that needs another term subclasses it and
    composes there -- see :class:`sae_bioemu.cryptic_steering.loss.CrypticSteeringLoss`, which
    adds the cryptic-pocket off-path split (PLAN.md D-28) without this file learning about it.
    The three extension points below are the whole interface, and they are deliberately
    content-free: :attr:`EXTRA_SAMPLE_INPUTS`, :meth:`objective_config_extra` and
    :meth:`validate_schedule` all return nothing on the shared loss, so a shared run
    constructs, records and journals exactly what it did before any subclass existed.
    """

    #: Names of the per-observation inputs this objective's ``forward`` accepts beyond the
    #: shared set, read off :class:`~sae_bioemu.steering.ordinary_workflow.GeneratedTrainingSample`
    #: by the runner and passed through verbatim. Empty here, so the shared call site passes
    #: exactly the arguments it passed before this attribute existed.
    EXTRA_SAMPLE_INPUTS: ClassVar[tuple[str, ...]] = ()

    #: Which of :data:`EXTENDED_LOSS_TERMS` this class actually multiplies into ``total``.
    #:
    #: EMPTY here, and the emptiness is the contract: the shared objective carries the radius of
    #: gyration, AF3 and Local-Q terms and nothing else, so a shared run constructs, records and
    #: journals exactly what it did before any subclass existed. A subclass that appends a term
    #: names it here; a nonzero weight outside the set is refused at construction rather than
    #: accepted and dropped.
    PRICED_EXTENDED_TERMS: ClassVar[frozenset[str]] = frozenset()

    #: The weights that DRIVE this objective: at least one of them must be strictly positive, or
    #: the loss has nothing pulling the adapter anywhere and the run measures nothing.
    #:
    #: ``("lambda_y",)`` here, which is the whole of the historical rule ``lambda_y > 0``. That
    #: rule was correct while every objective's only driver was the property target term, and it
    #: stopped being correct when a subclass added a term that grades the reported quantity
    #: DIRECTLY: the cryptic direct metric and the kinase classifier margin are each a complete
    #: objective on their own, and forcing a positive ``lambda_y`` beside them keeps a
    #: partially-correlated proxy pulling alongside the graded quantity, contributing its own
    #: errors as bias rather than information (D-82: within-protein Spearman -0.574 between ``the progress coordinate``
    #: and the experimental holo deviation, so the proxy disagrees with the metric on roughly a
    #: third of the rank variance).
    #:
    #: A subclass therefore names its own drivers and passes their values through
    #: ``additional_driving_weights``. What is NOT relaxed is the invariant the old line was
    #: really protecting: a loss with every driver at zero is still refused, by name, for every
    #: property. ``lambda_r`` stays nonnegative and is not a driver -- it is a parsimony penalty,
    #: and an objective made of penalties alone has a global minimum at "do nothing".
    DRIVING_WEIGHTS: ClassVar[tuple[str, ...]] = ("lambda_y",)

    @property
    def property_term_weight(self) -> float:
        """The multiplier :meth:`forward` applies to the shared property term (D-87).

        ``self.lambda_y`` here and for every objective but one, so the expression in
        :meth:`forward` is the expression it has always been -- the same float, the same operand
        order, the same rounding -- and no checkpoint moves for this hook existing.

        It exists because ``lambda_y`` MEANS "the weight on a progress-coordinate target", and one
        property routes a graded Angstrom METRIC through the same channel. There ``lambda_y`` would
        be a second, incompatible meaning for one name -- on a cryptic arm ``lambda_y = 0`` removes
        the ``the progress coordinate`` proxy, while on domain motion it would delete the objective -- so that property
        prices ``lambda_m`` like every other graded metric in this repository and converts it here.
        The conversion is the CONVENTION FACTOR between the shared channel and that property's own
        ``L``, and it belongs in the arithmetic rather than in the weight, so that ``lambda_m = 1``
        means the same descended loss wherever it is written.
        """
        return self.lambda_y

    def __init__(
        self,
        *,
        lambda_y: float,
        lambda_r: float,
        lambda_g: float,
        lambda_c: float = 0.0,
        lambda_l: float = 0.0,
        # The EXTENDED vocabulary, declared on the shared surface and defaulting to 0.0, so that
        # every objective takes the same constructor and a property enabling a term it already
        # computes does not change a signature any other property's fingerprint hashes. Each is
        # refused unless this class names it in ``PRICED_EXTENDED_TERMS``, so declaring them
        # here widens what can be CONFIGURED and not what can be silently ignored.
        lambda_perp: float = 0.0,
        lambda_perp_fraction: float = 0.0,
        lambda_rigid: float = 0.0,
        lambda_dihedral: float = 0.0,
        property_loss: PropertyLossKind | str = PropertyLossKind.MSE,
        property_loss_normalization: str = ABSOLUTE_LOSS_NORMALIZATION,
        residual_penalty: ResidualPenaltyKind | str = ResidualPenaltyKind.FRACTION,
        rho_max: float | None = None,
        # The values of this class's own drivers beyond ``lambda_y``, by name. Keyword-only with
        # a ``None`` default, so every existing caller is unchanged and the shared objective --
        # whose only driver IS ``lambda_y`` -- never mentions it. A subclass declaring a driver
        # must supply it here and may supply nothing else, which is checked in both directions
        # below: a declared driver that never arrives would let a driverless loss construct, and
        # an undeclared one would let any weight excuse ``lambda_y = 0``.
        additional_driving_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        extended = {
            "lambda_perp": lambda_perp,
            "lambda_perp_fraction": lambda_perp_fraction,
            "lambda_rigid": lambda_rigid,
            "lambda_dihedral": lambda_dihedral,
        }
        for name, value in (
            ("lambda_y", lambda_y),
            ("lambda_r", lambda_r),
            ("lambda_g", lambda_g),
            ("lambda_l", lambda_l),
            ("lambda_c", lambda_c),
            *extended.items(),
        ):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        negative = sorted(name for name, value in extended.items() if value < 0)
        if negative:
            raise ValueError(f"{', '.join(negative)} must be nonnegative")
        # A weight this class does not append to ``total``. Refused rather than ignored: an
        # accepted-and-dropped weight is computed nowhere, absent from ``objective_config_extra``
        # and therefore absent from its journal column, so the run would look complete and
        # measure nothing. Zero is always accepted -- a config that writes an inapplicable
        # weight out at zero to document that it is off is correct, and was correct before this
        # vocabulary existed.
        unpriced = sorted(
            name
            for name, value in extended.items()
            if value and name not in self.PRICED_EXTENDED_TERMS
        )
        if unpriced:
            raise ValueError(
                f"{type(self).__name__} does not price {', '.join(unpriced)}: the weight would "
                "be accepted, multiplied into no term, and absent from the objective identity "
                "and from the journal column gated on it. It prices "
                f"{sorted(self.PRICED_EXTENDED_TERMS) or 'no extended term'}"
            )
        for name, value in extended.items():
            setattr(self, name, float(value))
        if lambda_r < 0:
            raise ValueError("lambda_r must be nonnegative")
        _require_a_driving_weight(type(self), lambda_y, additional_driving_weights)
        if lambda_c < 0:
            raise ValueError("lambda_c must be nonnegative")
        if lambda_g < 0:
            raise ValueError("lambda_g must be nonnegative")
        self.lambda_y = float(lambda_y)
        self.lambda_r = float(lambda_r)
        self.lambda_g = float(lambda_g)
        if lambda_l < 0:
            raise ValueError("lambda_l must be nonnegative")
        self.lambda_l = float(lambda_l)
        self.lambda_c = float(lambda_c)
        if property_loss_normalization not in PROPERTY_LOSS_NORMALIZATIONS:
            raise ValueError(
                "property_loss_normalization must be one of "
                f"{sorted(PROPERTY_LOSS_NORMALIZATIONS)}; received "
                f"{property_loss_normalization!r}"
            )
        self.property_loss_normalization = str(property_loss_normalization)
        if (
            self.lambda_c > 0.0
            and self.property_loss_normalization not in _PAIRED_DOSE_NORMALIZATIONS
        ):
            # Refused at construction, not at the first batch, and refused rather than run: the
            # paired gap under this declaration is a difference in raw property units, which is
            # neither comparable across proteins nor on the same footing as the target term that
            # lambda_c is weighed against. It DID raise, from inside ``paired_dose_loss``, until
            # the transition path made ``property_scale=None`` legitimate -- at which point an
            # absolute config with lambda_c set began training on the unnormalised gap instead.
            raise ValueError(
                "the paired dose term is defined on a normalised path only: under "
                f"{ABSOLUTE_LOSS_NORMALIZATION!r} the gap is a difference in the property's own "
                "raw units, with no per-protein divisor to make two proteins comparable. Declare "
                f"one of {sorted(_PAIRED_DOSE_NORMALIZATIONS)}, or set lambda_c to zero"
            )
        try:
            self.property_loss = PropertyLossKind(property_loss)
        except ValueError as error:
            raise ValueError("unsupported property loss") from error
        try:
            self.residual_penalty = ResidualPenaltyKind(residual_penalty)
        except ValueError as error:
            raise ValueError("unsupported residual penalty") from error
        if self.residual_penalty in (
            ResidualPenaltyKind.FRACTION_OF_RHO_MAX,
            ResidualPenaltyKind.RAW_FRACTION_OF_RHO_MAX,
        ):
            if (
                isinstance(rho_max, bool)
                or not isinstance(rho_max, Real)
                or not math.isfinite(rho_max)
                or not 0 < rho_max <= 1
            ):
                raise ValueError(
                    "residual_penalty fraction_of_rho_max normalizes by the adapter's own "
                    "intervention bound, so rho_max must be supplied in (0, 1]"
                )
            self.rho_max = float(rho_max)
        elif rho_max is not None:
            raise ValueError(
                "rho_max is used only by the fraction_of_rho_max residual penalty; the "
                "fraction and rms penalties are not normalized by the intervention bound"
            )
        else:
            self.rho_max = None
        self.residual_penalty_units = RESIDUAL_PENALTY_UNITS[self.residual_penalty]

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        residual_rms: torch.Tensor,
        residual_fraction: torch.Tensor,
        geometry_loss: torch.Tensor | None = None,
        *,
        property_scale: torch.Tensor | None = None,
        pair_group: torch.Tensor | None = None,
        paired_loss: torch.Tensor | None = None,
        raw_residual_fraction: torch.Tensor | None = None,
        local_structure: torch.Tensor | None = None,
    ) -> SteeringLossComponents:
        """Return the weighted total and every component that fed it.

        This is the shared set of terms and takes no project's extra inputs. A subclass that
        adds one accepts it in its own ``forward``, declares it in :attr:`EXTRA_SAMPLE_INPUTS`
        so the runner supplies it, and adds its weighted contribution to the total this
        returns -- which is why the additions below are ordered and why a subclass appends
        rather than rebuilds. Floating-point addition is not associative, so the order in
        which terms enter ``total`` is part of the objective and not a formatting choice.

        ``paired_loss`` SUPPLIES the paired dose term instead of computing it from
        ``pair_group``, and it exists for exactly one caller: the per-member journal
        re-evaluation in ``ordinary_workflow._batch_training_records``, which slices a batch to
        one member and therefore cannot form a pair at all. It is the same shape of input
        ``geometry_loss`` and ``local_structure`` already are -- a whole-batch scalar the caller
        computed and the objective prices -- and routing it through here rather than adding
        ``lambda_c * contrastive`` outside is what keeps the term in the objective's own
        summation order. Supplying BOTH is refused: two answers for one term is how a journalled
        total comes to describe an objective nobody optimised.
        """
        for name, value in (
            ("prediction", prediction),
            ("target", target),
            ("residual_rms", residual_rms),
            ("residual_fraction", residual_fraction),
        ):
            _validate_loss_tensor(value, name)
        tensors = (prediction, target, residual_rms, residual_fraction)
        if any(value.shape != prediction.shape for value in tensors[1:]):
            raise ValueError("prediction, target, and residual diagnostics must share a shape")
        if any(value.device != prediction.device for value in tensors[1:]):
            raise ValueError("loss tensors must share a device")
        self._require_declared_scale(property_scale)
        if property_scale is None:
            if self.property_loss is PropertyLossKind.MSE:
                target_loss = functional.mse_loss(prediction, target, reduction="mean")
            else:
                target_loss = functional.huber_loss(prediction, target, reduction="mean", delta=1.0)
        else:
            # D4b: dividing by each unit's own sigma keeps the objective scale-free,
            # so a floppy protein cannot outweigh a rigid one at equal relative error.
            _validate_loss_tensor(property_scale, "property_scale")
            if property_scale.shape != prediction.shape:
                raise ValueError("property_scale must share the prediction shape")
            if property_scale.device != prediction.device:
                raise ValueError("loss tensors must share a device")
            if torch.any(property_scale <= 0):
                raise ValueError("property_scale must be positive")
            normalized_error = (prediction - target) / property_scale
            if self.property_loss is PropertyLossKind.MSE:
                target_loss = normalized_error.square().mean()
            else:
                target_loss = functional.huber_loss(
                    normalized_error,
                    torch.zeros_like(normalized_error),
                    reduction="mean",
                    delta=1.0,
                )
        if self.residual_penalty is ResidualPenaltyKind.RMS:
            residual_values = residual_rms
        elif self.residual_penalty is ResidualPenaltyKind.FRACTION:
            residual_values = residual_fraction
        elif self.residual_penalty is ResidualPenaltyKind.RAW_FRACTION_OF_RHO_MAX:
            # Fails closed: silently falling back to the capped fraction would reproduce the
            # exact saturation this kind exists to prevent, and look like it had been fixed.
            if raw_residual_fraction is None:
                raise ValueError(
                    "raw_fraction_of_rho_max needs the adapter's pre-cap residual fraction; "
                    "pass raw_residual_fraction from the AdapterOutput"
                )
            _validate_loss_tensor(raw_residual_fraction, "raw_residual_fraction")
            if raw_residual_fraction.shape != prediction.shape:
                raise ValueError("prediction, target, and residual diagnostics must share a shape")
            if raw_residual_fraction.device != prediction.device:
                raise ValueError("loss tensors must share a device")
            residual_values = raw_residual_fraction / self.rho_max
        else:
            # Exactly the fraction penalty divided by rho_max, so the term spans [0, 1]
            # and lambda_r means the same thing at every intervention bound.
            residual_values = residual_fraction / self.rho_max
        residual_loss = residual_values.square().mean()
        contrastive = None
        if paired_loss is not None:
            if pair_group is not None:
                raise ValueError(
                    "pass either pair_group or paired_loss, never both: pair_group asks the "
                    "objective to FORM the paired term and paired_loss states it, so supplying "
                    "both would silently discard one of two different numbers for one term"
                )
            if self.lambda_c <= 0.0:
                raise ValueError(
                    f"lambda_c={self.lambda_c} prices no paired dose term, so a supplied "
                    "paired_loss would be computed, weighted by zero and journalled as though "
                    "the objective had priced it. That is the shape lambda_l had when it voided "
                    "six radius-of-gyration arms"
                )
            _validate_loss_tensor(paired_loss, "paired_loss")
            contrastive = paired_loss
            if contrastive.numel() != 1:
                contrastive = contrastive.mean()
        elif self.lambda_c > 0.0 and pair_group is not None:
            # The paired channel's divisor, chosen from the DECLARED normalisation -- checked
            # above, not assumed from whether a scale arrived. Under ``sequence_sigma`` (radius
            # of gyration) it is that sequence's sigma, which is the historical normalisation and
            # is bit-identical. Under ``apo_holo_transition`` and ``per_channel_sigma_transition``
            # it is ``None``: the gap is formed inside one noise draw, so the baseline cancels
            # identically and there is no sigma noise in it to divide out, and dividing a clean
            # differential signal by sigma downweights by ``1/sigma`` exactly the high-sigma
            # proteins the term exists to rescue -- see :func:`paired_dose_loss`.
            #
            # ``per_channel_sigma_transition`` is why this is a lookup rather than
            # ``property_scale``: there the scale is REQUIRED and is applied to the absolute
            # channel above, and passing it on to here as well would collapse D-65 back into
            # ``sequence_sigma`` silently, with every other check green. ``property_units`` never
            # reaches this branch: the constructor refuses ``lambda_c > 0`` on that declaration.
            paired_scale = (
                property_scale
                if self.property_loss_normalization in _SIGMA_SCALED_PAIRED_NORMALIZATIONS
                else None
            )
            contrastive = paired_dose_loss(prediction, target, paired_scale, pair_group)
        geometry = prediction.new_zeros(()) if geometry_loss is None else geometry_loss
        _validate_loss_tensor(geometry, "geometry")
        if geometry.numel() != 1:
            geometry = geometry.mean()

        # The local-structure term. This was accepted, stored and then dropped on the floor
        # for the whole lambda_l sweep: the weight existed, the value was computed, an extra
        # unsteered generation was paid for to produce the reference, and none of it entered
        # `total`. Six arms were run as a lambda_l ablation and measured nothing.
        #
        # The weight without its input is that same shape one step earlier, and it is refused
        # here rather than silently priced at nothing -- exactly as a setpoint at a zero weight
        # is refused. Only that direction: a supplied value at ``lambda_l == 0`` is NOT a
        # half-configuration, because for the two-state properties the unsteered draw is the progress coordinate's
        # own anchor and is generated whatever ``lambda_l`` is, so the smooth LDDT against it
        # comes for free and is journalled as a diagnostic.
        if self.lambda_l > 0.0 and local_structure is None:
            raise ValueError(
                f"lambda_l={self.lambda_l} prices smooth LDDT against the unsteered draw, so "
                "local_structure is required: without it the weight is configured, recorded in "
                "the objective identity and multiplied into nothing. That is the exact shape "
                "that voided six radius-of-gyration arms"
            )
        local = None
        if local_structure is not None:
            local = local_structure
            _validate_loss_tensor(local, "local_structure")
            if local.numel() != 1:
                local = local.mean()

        total = (
            self.property_term_weight * target_loss
            + self.lambda_r * residual_loss
            + self.lambda_g * geometry
        )
        if contrastive is not None:
            total = total + self.lambda_c * contrastive
        if local is not None:
            total = total + self.lambda_l * local
        for name, component in (
            ("target", target_loss),
            ("residual", residual_loss),
            ("geometry", geometry),
            ("total", total),
        ):
            _validate_loss_tensor(component, name)
        return SteeringLossComponents(
            total, target_loss, residual_loss, geometry, contrastive, local
        )

    def _require_declared_scale(self, property_scale: torch.Tensor | None) -> None:
        """Fail closed unless the supplied scale is the one the declared normalisation needs.

        Both directions, and both for the same reason. A missing scale under
        ``sequence_sigma`` or ``per_channel_sigma_transition`` silently divides by nothing and
        turns a sigma-weighted run into an unnormalised one whose absolute channel is
        ``sigma^2`` times larger; a supplied scale under ``apo_holo_transition`` or
        ``property_units`` silently divides a quantity that is already in its final units.
        Neither shows up as an error anywhere downstream -- both show up as a number.
        """
        if self.property_loss_normalization in _SIGMA_SCALED_NORMALIZATIONS:
            if property_scale is None:
                raise ValueError(
                    f"this objective declares {self.property_loss_normalization!r} "
                    "normalisation, so property_scale is required: the ABSOLUTE channel divides "
                    "by each unit's own sigma, which is what makes equal relative errors weigh "
                    "equally (D4b) and, under the two-state state target, what applies "
                    "inverse-variance weighting to a residual that carries the whole draw "
                    "baseline (D-65). Running without it is an unnormalised objective, not a "
                    "normalised one with a missing input"
                )
        elif property_scale is not None:
            raise ValueError(
                f"this objective declares {self.property_loss_normalization!r} normalisation, "
                "which carries no per-observation divisor, so property_scale must not be "
                f"supplied. Under {TRANSITION_LOSS_NORMALIZATION!r} the target is already "
                "unit-length apo->holo by construction and dividing by sigma renormalises an "
                "already-normalised quantity"
            )

    def extended_weight_identity(self) -> dict[str, Any]:
        """Return the EXTENDED weights this objective prices, each present only when nonzero.

        ONE emission rule for the whole vocabulary, shared by every subclass, so that a term
        turned on in a config lands in the objective identity -- and therefore switches on its
        journal column -- without the subclass restating the discipline. Conditional presence is
        not tidiness: ``adapter_config`` is compared with canonical-JSON equality on resume and
        at the evaluation gate, so an unconditional key would refuse every checkpoint written
        before it.

        ``lambda_perp`` is written on BOTH forms of the off-path weight, and that is
        load-bearing: ``ordinary_workflow._OFF_PATH_COLUMN_WEIGHTS`` gates the ``perp_loss``
        column on ``objective_config_extra()["lambda_perp"]`` being truthy, so a fraction-form
        arm that recorded only its own key would compute the off-axis term, price it, and then
        be refused by the journal validator for carrying a column its weight says is off -- or,
        worse, write no column at all and lose the one measured off-axis quantity the term
        exists to produce.

        The number is honest: the per-protein weight is ``(c*lambda_y)/sigma_the progress coordinate^2``, so
        ``c*lambda_y`` is its NUMERATOR, the part that is the same for every protein. It is not a
        weight any single protein sees, and ``lambda_perp_fraction`` sits beside it in the same
        dict saying exactly that, so no reader can take one without the other.
        """
        config: dict[str, Any] = {}
        if self.lambda_perp:
            config["lambda_perp"] = self.lambda_perp
        elif self.lambda_perp_fraction:
            config["lambda_perp"] = self.lambda_perp_fraction * self.lambda_y
            config["lambda_perp_fraction"] = self.lambda_perp_fraction
        if self.lambda_rigid:
            config["lambda_rigid"] = self.lambda_rigid
        if self.lambda_dihedral:
            config["lambda_dihedral"] = self.lambda_dihedral
        return config

    def objective_config_extra(self) -> dict[str, Any]:
        """Return this objective's extra identity keys, merged into ``_objective_config``.

        Empty on the SHARED objective, and the emptiness is the contract: it prices no extended
        term, so :meth:`extended_weight_identity` returns nothing and the shared objective
        identity keeps its exact historical key set -- no radius-of-gyration, AF3 or Local-Q
        checkpoint is invalidated by a weight it never had.

        A subclass adds only the keys its own state switches on, and reaches the shared
        vocabulary through ``super()`` rather than restating it.

        The keys are also what gates the optional journal columns: a weight recorded here is
        a weight whose column must be present, and one absent here must have no column.
        """
        return self.extended_weight_identity()

    def validate_schedule(self, schedule: object, config: object) -> None:
        """Check this objective against the schedule it is about to train over.

        A no-op here. It exists so that a subclass with a configuration-dependent bound --
        the cryptic-pocket curvature bound of PLAN.md D-31 is the one that motivated it --
        can enforce it from the shared runner without the shared runner naming it. The
        runner calls this unconditionally; a shared objective therefore runs exactly the
        checks it always ran.
        """
        return None


def paired_dose_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    property_scale: torch.Tensor | None,
    pair_group: torch.Tensor,
) -> torch.Tensor:
    """Return the mean squared shortfall of the *difference* between paired levels.

    For two units of one sequence generated from the **same** noise draw,

        Rg(o) = R0(the progress coordinate) + e(o, the progress coordinate)

    so the difference ``Rg(o_i) - Rg(o_j)`` cancels ``R0(the progress coordinate)`` identically. That baseline is
    the entire training noise: measured across six draws it varies by 4.59 A while the
    intervention's effect on it varies by 0.006 A. The target difference is already carried by
    ``target``, so nothing beyond the pairing needs to be supplied.

    ``property_scale`` is the divisor the paired gap is measured in, and ``None`` means the gap is
    already dimensionless and is used as it stands. **Which of the two applies is decided by the
    objective's declared** ``property_loss_normalization``, not by this function, and NOT by
    whether the objective was handed a scale for its other channel: ``SteeringLoss.forward``
    resolves the paired divisor from the declaration and hands it here. Calling it directly
    bypasses that resolution, so a caller outside the objective owns the choice it is making. The
    two cases are not a style choice:

    * **A tensor -- the fully sigma-normalized path, which radius of gyration keeps.** Its target
      term is ``((Rg - t)/sigma_seq)^2`` in multiples of that sequence's own spread (D4b), and
      normalising the paired gap the same way is what makes ``lambda_c`` and ``lambda_y`` weights
      on comparable quantities. This is the historical behaviour and is bit-identical.
    * **``None`` -- the two-state paths, under ``apo_holo_transition`` and under
      ``per_channel_sigma_transition`` alike.** Dividing the gap by ``sigma`` would be worse than
      merely inconsistent: the paragraph above establishes that the paired difference cancels the
      common-mode noise IDENTICALLY, so the gap carries no sigma noise at all, and dividing a
      CLEAN differential signal by ``sigma`` downweights by ``1/sigma`` exactly the high-sigma
      proteins whose absolute term is drowned and which the contrastive term exists to rescue.
      Under ``per_channel_sigma_transition`` the objective's OTHER channel does divide by sigma,
      and that asymmetry is the point: only the absolute residual carries the baseline noise
      inverse-variance weighting is for.

    **What licenses ``None`` for ``the progress coordinate`` is not the radius-of-gyration measurement above.** The
    4.59 A against 0.006 A decomposition is Rg's; on the kinase panel the aligned paired gap's
    own draw-dependent residual is 116x (f = 0.02) to 756x (f = rho_max) that reference in units
    of the property's own sigma, so the Rg number cannot be cited for ``the progress coordinate``. The licence is the
    aligned paired-gap run (a measured run, ``docs/kinase_dfg/aligned_paired_gap_findings.txt``),
    which found ``sd(gap)`` PROPORTIONAL to ``sigma_the progress coordinate`` -- log-log slope 1.02, 95% interval
    [0.42, 1.62], excluding 0 and containing 1 -- so ``sd(gap)/sigma_the progress coordinate`` is flat across proteins
    and dividing by sigma would tilt it. That rests on ONE protein clearing a t = 3.0 gate at
    t = 4.55, at f = 0.02 and not at the pre-declared headline f = rho_max, with no multiplicity
    correction over ten proteins. It is the best evidence this programme has and it is not strong
    evidence.

    ``pair_group`` marks which entries share a draw: equal nonnegative values are paired, and
    negative values are excluded. Every unordered pair inside a group contributes once. A
    batch with no pair returns exactly zero, so an unpairable batch contributes nothing rather
    than silently reweighting the objective.
    """
    if pair_group.shape != prediction.shape:
        raise ValueError("pair_group must share the prediction shape")
    if pair_group.dtype not in (torch.int32, torch.int64):
        raise ValueError("pair_group must be an integer tensor")
    terms = []
    for group in torch.unique(pair_group):
        if int(group) < 0:
            continue
        index = torch.nonzero(pair_group == group, as_tuple=False).reshape(-1)
        if index.numel() < 2:
            continue
        for position, left in enumerate(index[:-1]):
            for right in index[position + 1 :]:
                gap = (prediction[left] - prediction[right]) - (target[left] - target[right])
                if property_scale is not None:
                    # Normalised by sigma exactly as the absolute term is (D4b), so lambda_c and
                    # lambda_y are weights on comparable quantities.
                    gap = gap / property_scale[left]
                terms.append(gap.square())
    if not terms:
        return prediction.new_zeros(())
    return torch.stack(terms).mean()


def _validate_loss_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def freeze_for_steering(*modules: nn.Module) -> None:
    """Freeze every module except the final adapter while preserving differentiable operations."""
    if len(modules) < 2:
        raise TypeError("freeze_for_steering requires frozen module(s) followed by one adapter")
    *frozen, adapter = modules
    _require_modules(modules)
    adapter_parameters = tuple(adapter.parameters())
    if not adapter_parameters:
        raise ValueError("adapter must have parameters")
    adapter_ids = {id(parameter) for parameter in adapter_parameters}
    if any(id(parameter) in adapter_ids for module in frozen for parameter in module.parameters()):
        raise ValueError("adapter and frozen modules have a parameter identity overlap")
    for module in frozen:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    adapter.train()
    for parameter in adapter_parameters:
        parameter.requires_grad_(True)


def assert_adapter_only_gradients(*modules: nn.Module) -> None:
    """Fail if frozen gradients exist or final-adapter gradients are invalid."""
    if len(modules) < 2:
        raise TypeError("gradient check requires frozen module(s) followed by one adapter")
    *frozen, adapter = modules
    _require_modules(modules)
    adapter_parameters = tuple(adapter.parameters())
    adapter_ids = {id(parameter) for parameter in adapter_parameters}
    if not adapter_ids:
        raise RuntimeError("adapter has no trainable parameters")
    if any(id(parameter) in adapter_ids for module in frozen for parameter in module.parameters()):
        raise RuntimeError("adapter and frozen modules have a parameter identity overlap")
    for module in frozen:
        for parameter in module.parameters():
            if parameter.requires_grad:
                raise RuntimeError("frozen parameters must have requires_grad=False")
            if parameter.grad is not None:
                raise RuntimeError("frozen parameter received a gradient")
    gradients = tuple(parameter.grad for parameter in adapter_parameters)
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("adapter parameter gradient is None")
    if any(not torch.isfinite(gradient).all() for gradient in gradients if gradient is not None):
        raise RuntimeError("adapter parameter gradient is nonfinite")
    if not any(
        torch.count_nonzero(gradient).item() for gradient in gradients if gradient is not None
    ):
        raise RuntimeError("adapter parameter gradients are all zero")


def _require_modules(modules: Sequence[nn.Module]) -> None:
    if any(not isinstance(module, nn.Module) for module in modules):
        raise TypeError("frozen modules and adapter must be torch modules")


@dataclass(frozen=True)
class AdapterCheckpoint:
    manifest_fingerprint: str
    adapter_config: Mapping[str, Any]
    property_callback_identity: str
    source_checkpoints: Mapping[str, str]
    adapter_state: Mapping[str, torch.Tensor]
    optimizer_state: Mapping[str, Any]
    epoch: int
    unit_cursor: int
    validation_record: Mapping[str, Any]


def save_adapter_checkpoint(path: str | Path, checkpoint: AdapterCheckpoint) -> None:
    """Atomically write a complete, safe-loadable primitive checkpoint on CPU."""
    if not isinstance(checkpoint, AdapterCheckpoint):
        raise TypeError("checkpoint must be an AdapterCheckpoint")
    payload = _checkpoint_payload(checkpoint)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_adapter_checkpoint(path: str | Path) -> AdapterCheckpoint:
    """Safely load a checkpoint and verify it against ITSELF, binding no caller provenance.

    Every structural gate :func:`load_adapter_checkpoint` applies still applies here -- safe
    ``weights_only`` load, exact key set, schema version, the completeness flag, and the
    self-referential ``checkpoint_identity_fingerprint``. What it does NOT do is compare the
    checkpoint against an expectation, because there are callers that must read the checkpoint's
    own ``unit_cursor`` and ``adapter_config`` in order to *derive* the expectation.

    That derivation is :mod:`sae_bioemu.steering.schedule_extension`, which reconciles a
    checkpoint with a LONGER epoch budget. It is not a relaxation of this module's contract: the
    caller still passes the derived expectation to :func:`validate_adapter_checkpoint`, so the
    same fail-closed comparison runs over everything.
    """
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("unable to safely load adapter checkpoint") from error
    return _checkpoint_from_payload(payload)


def load_adapter_checkpoint(
    path: str | Path,
    *,
    expected_manifest_fingerprint: str,
    expected_adapter_config: Mapping[str, Any],
    expected_property_callback_identity: str,
    expected_source_checkpoints: Mapping[str, str],
) -> AdapterCheckpoint:
    """Safely load a complete checkpoint bound to caller-provided provenance."""
    _validate_fingerprint(expected_manifest_fingerprint, "expected manifest fingerprint")
    _validate_json_mapping(expected_adapter_config, "expected adapter config")
    _validate_property_callback_identity(
        expected_property_callback_identity, "expected property callback identity"
    )
    _validate_string_mapping(expected_source_checkpoints, "expected source checkpoints")
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("unable to safely load adapter checkpoint") from error
    return validate_adapter_checkpoint(
        _checkpoint_from_payload(payload),
        expected_manifest_fingerprint=expected_manifest_fingerprint,
        expected_adapter_config=expected_adapter_config,
        expected_property_callback_identity=expected_property_callback_identity,
        expected_source_checkpoints=expected_source_checkpoints,
    )


def _checkpoint_from_payload(payload: object) -> AdapterCheckpoint:
    """Validate a loaded payload's structure and its self-referential identity fingerprint."""
    if not isinstance(payload, dict):
        raise ValueError("adapter checkpoint payload must be a dictionary")
    if payload.get("complete") is not True:
        raise ValueError("adapter checkpoint is not complete")
    if frozenset(payload) != _CHECKPOINT_KEYS:
        raise ValueError("adapter checkpoint keys are invalid")
    if (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != _CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported adapter checkpoint schema version")
    checkpoint = AdapterCheckpoint(
        manifest_fingerprint=payload["manifest_fingerprint"],
        adapter_config=payload["adapter_config"],
        property_callback_identity=payload["property_callback_identity"],
        source_checkpoints=payload["source_checkpoints"],
        adapter_state=payload["adapter_state"],
        optimizer_state=payload["optimizer_state"],
        epoch=payload["epoch"],
        unit_cursor=payload["unit_cursor"],
        validation_record=payload["validation_record"],
    )
    canonical_payload = _checkpoint_payload(checkpoint)
    _validate_fingerprint(
        payload["checkpoint_identity_fingerprint"], "checkpoint identity fingerprint"
    )
    if (
        payload["checkpoint_identity_fingerprint"]
        != canonical_payload["checkpoint_identity_fingerprint"]
    ):
        raise ValueError("adapter checkpoint identity fingerprint is invalid")
    return checkpoint


def validate_adapter_checkpoint(
    checkpoint: AdapterCheckpoint,
    *,
    expected_manifest_fingerprint: str,
    expected_adapter_config: Mapping[str, Any],
    expected_property_callback_identity: str,
    expected_source_checkpoints: Mapping[str, str],
) -> AdapterCheckpoint:
    """Validate an in-memory checkpoint with the same fail-closed contract as loading."""
    _validate_fingerprint(expected_manifest_fingerprint, "expected manifest fingerprint")
    _validate_json_mapping(expected_adapter_config, "expected adapter config")
    _validate_property_callback_identity(
        expected_property_callback_identity, "expected property callback identity"
    )
    _validate_string_mapping(expected_source_checkpoints, "expected source checkpoints")
    if not isinstance(checkpoint, AdapterCheckpoint):
        raise TypeError("checkpoint must be an AdapterCheckpoint")
    _checkpoint_payload(checkpoint)
    if checkpoint.manifest_fingerprint != expected_manifest_fingerprint:
        raise ValueError(
            "adapter checkpoint manifest fingerprint does not match expected manifest fingerprint"
        )
    if _canonical_json_bytes(checkpoint.adapter_config) != _canonical_json_bytes(
        expected_adapter_config
    ):
        raise ValueError("adapter checkpoint adapter config does not match expected adapter config")
    if checkpoint.property_callback_identity != expected_property_callback_identity:
        raise ValueError(
            "adapter checkpoint property callback identity does not match expected property "
            "callback identity"
        )
    if _canonical_json_bytes(checkpoint.source_checkpoints) != _canonical_json_bytes(
        expected_source_checkpoints
    ):
        raise ValueError(
            "adapter checkpoint source checkpoints do not match expected source checkpoints"
        )
    return _cpu_checkpoint(checkpoint)


def _checkpoint_payload(checkpoint: AdapterCheckpoint) -> dict[str, Any]:
    _validate_fingerprint(checkpoint.manifest_fingerprint, "manifest fingerprint")
    _validate_property_callback_identity(
        checkpoint.property_callback_identity, "property callback identity"
    )
    _validate_string_mapping(checkpoint.source_checkpoints, "source checkpoints")
    _validate_json_mapping(checkpoint.adapter_config, "adapter config")
    _validate_json_mapping(checkpoint.validation_record, "validation record")
    _validate_state_dict(checkpoint.adapter_state, "adapter state")
    _validate_optimizer_mapping(checkpoint.optimizer_state, "optimizer state")
    for name, value in (("epoch", checkpoint.epoch), ("unit_cursor", checkpoint.unit_cursor)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    payload = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "complete": True,
        "manifest_fingerprint": checkpoint.manifest_fingerprint,
        "adapter_config": _clone_json_value(checkpoint.adapter_config),
        "property_callback_identity": checkpoint.property_callback_identity,
        "source_checkpoints": dict(checkpoint.source_checkpoints),
        "adapter_state": _clone_tensor_mapping(checkpoint.adapter_state),
        "optimizer_state": _clone_optimizer_value(checkpoint.optimizer_state),
        "epoch": checkpoint.epoch,
        "unit_cursor": checkpoint.unit_cursor,
        "validation_record": _clone_json_value(checkpoint.validation_record),
    }
    payload["checkpoint_identity_fingerprint"] = _checkpoint_identity_fingerprint(payload)
    return payload


def _checkpoint_identity_fingerprint(payload: Mapping[str, Any]) -> str:
    identity = {
        "manifest_fingerprint": payload["manifest_fingerprint"],
        "adapter_config": payload["adapter_config"],
        "property_callback_identity": payload["property_callback_identity"],
        "source_checkpoints": payload["source_checkpoints"],
    }
    return hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validate_fingerprint(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 fingerprint")


def _validate_string_mapping(value: object, name: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        _validate_fingerprint(item, f"{name} values")


def _validate_json_mapping(value: object, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    _validate_json_value(value, name)


def _validate_property_callback_identity(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty")


def _validate_json_value(value: object, name: str) -> None:
    if value is None or isinstance(value, str | bool):
        return
    if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, name)
        return
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json_value(item, name)
        return
    raise ValueError(f"{name} must contain JSON-safe finite primitives")


def _clone_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _clone_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_json_value(item) for item in value]
    return value


def _validate_state_dict(value: object, name: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty tensor mapping")
    for key, tensor in value.items():
        if not isinstance(key, str) or not key or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must map non-empty names to tensors")
        _validate_tensor(tensor, name)


def _clone_tensor_mapping(value: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().to(device="cpu").clone() for key, tensor in value.items()}


def _validate_optimizer_value(value: object, name: str) -> None:
    if isinstance(value, torch.Tensor):
        _validate_tensor(value, name)
        return
    if value is None or isinstance(value, str | bool):
        return
    if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
        return
    if isinstance(value, list | tuple):
        for item in value:
            _validate_optimizer_value(item, name)
        return
    if isinstance(value, Mapping) and all(type(key) in (str, int) for key in value):
        for item in value.values():
            _validate_optimizer_value(item, name)
        return
    raise ValueError(f"{name} must contain only primitive values and tensors")


def _validate_optimizer_mapping(value: object, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    _validate_optimizer_value(value, name)


def _clone_optimizer_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        return {key: _clone_optimizer_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_optimizer_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_optimizer_value(item) for item in value)
    return value


def _validate_tensor(tensor: torch.Tensor, name: str) -> None:
    if (tensor.is_complex() or tensor.is_floating_point()) and not torch.isfinite(tensor).all():
        raise ValueError(f"{name} tensors must be finite")


def _cpu_checkpoint(checkpoint: AdapterCheckpoint) -> AdapterCheckpoint:
    return AdapterCheckpoint(
        manifest_fingerprint=checkpoint.manifest_fingerprint,
        adapter_config=_clone_json_value(checkpoint.adapter_config),
        property_callback_identity=checkpoint.property_callback_identity,
        source_checkpoints=dict(checkpoint.source_checkpoints),
        adapter_state=_clone_tensor_mapping(checkpoint.adapter_state),
        optimizer_state=_clone_optimizer_value(checkpoint.optimizer_state),
        epoch=checkpoint.epoch,
        unit_cursor=checkpoint.unit_cursor,
        validation_record=_clone_json_value(checkpoint.validation_record),
    )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
