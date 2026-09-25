"""Deterministic continuous target schedules for property steering."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real

from .config import SequenceSplit
from .training import TargetRange

TARGET_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)
EPOCH_VALIDATION_LEVELS = (-1.0, 0.0, 1.0)

#: The steered dose levels for a **two-state** property (D-44). A cryptic pocket is open or
#: closed and a kinase is DFG-in or DFG-out, so an intermediate dose names no structural ask:
#: "half open" is not a state the protein has. ``0.0`` is absent because D-10 already removed
#: it for the cryptic arm -- under the matched-draw reference the unsteered draw IS ``the progress coordinate = 0``,
#: so ``o = 0`` requests an exact no-op, which D2 forbids as a target. This implements D-10's
#: conclusion rather than revisiting it.
#:
#: **Both directions are kept, deliberately.** The obvious objection is that ``-1`` asks an
#: already-closed pocket to close further. We do not know where BioEmu's unsteered default sits
#: on the apo/holo axis -- that is what D-26's spontaneous-opening measurement is for, and it has
#: not run -- so a level is "bias toward", not "reach". If a protein is already in the target
#: state, a null response in that direction is information rather than failure. Keeping both also
#: preserves the design's guard of ranking on the WORSE direction, which exists to catch
#: one-sided artifacts, and a single-signed set would silently remove it.
TWO_STATE_TARGET_LEVELS = (-1.0, 1.0)

#: The two-state epoch-validation set, and it drops ``0.0`` rather than mirroring
#: ``EPOCH_VALIDATION_LEVELS`` (D-44). D-10's argument does not transfer automatically --
#: validation selects on a generated MAE and never calls the loss -- so the decision is made on
#: its own three grounds:
#:
#: 1. D2 forbids the exact no-op as a target. Selecting a checkpoint partly on the MAE at a
#:    target the objective is not permitted to pursue selects on something training cannot
#:    improve, which is a selection signal with no gradient behind it.
#: 2. The question ``o = 0`` would answer -- is the adapter quiet when asked for nothing -- is
#:    already answered by the matched unsteered control, which is generated regardless because
#:    it is the progress coordinate's own anchor. The information is not lost, it is just not paid for twice.
#: 3. Symmetry with the steered set makes the early-stopping metric and the reported test
#:    metric the same quantity over the same levels, rather than two MAEs over two level sets.
TWO_STATE_EPOCH_VALIDATION_LEVELS = (-1.0, 1.0)


#: The reference table's ``mean`` is an ABSOLUTE position of this sequence's own adapter-off
#: ensemble, and level 0 targets it (D3/D4). Correct whenever the prediction is measured in the
#: same absolute frame -- radius of gyration, where an unsteered structure's Rg is its own number.
TARGET_ORIGIN_SEQUENCE_MEAN = "sequence_mean"

#: The prediction is a DISPLACEMENT from a matched unsteered draw, so its zero point is zero by
#: construction and the target must carry no offset: ``t = o * k * sigma`` (D-46). Adding the
#: reference table's ``mean`` here is a frame error -- the mean is an absolute position measured
#: against a different reference, and adding it asks every level to move by that position on top
#: of the requested displacement.
TARGET_ORIGIN_MATCHED_DRAW_ZERO = "matched_draw_zero"

#: The prediction is a DISPLACEMENT from a matched unsteered draw and the ask is an ABSOLUTE STATE
#: in the REPORTING frame: ``t = the progress coordinate_state(o) - the progress coordinate_reporting(control)``. This is what a TWO-STATE
#: property means -- a cryptic pocket is open or closed and a kinase is DFG-in or DFG-out -- and it
#: carries no ``k`` and no ``sigma``, because ``the progress coordinate`` is already unit-length apo->holo for every
#: protein by construction (``the progress coordinate = <u,v>_w/||v||^2_w`` with ``v = d(holo) - d(apo)``, so ``the progress coordinate = 1``
#: at the experimental holo and ``the progress coordinate = 0`` at the experimental apo, EXACTLY).
#:
#: **This is not a return of the D-46 defect and must not be "fixed" back to
#: :data:`TARGET_ORIGIN_MATCHED_DRAW_ZERO`.** D-46 removed a PER-PROTEIN CONSTANT -- the reference
#: table's reporting-frame ``mean_the progress coordinate`` -- added to a displacement target, which biased every draw of
#: a protein by the same wrong amount and showed up as an intercept no dose slope could see. The
#: origin here is THE SPECIFIC MATCHED CONTROL'S OWN reporting-frame value, which is the exactly
#: correct one: ``the progress coordinate`` is linear in the pairwise distance vector, so
#: ``the progress coordinate_training(s) = the progress coordinate_reporting(s) - the progress coordinate_reporting(control)`` EXACTLY, and subtracting the
#: control's own value is precisely the change of frame rather than an approximation of it.
TARGET_ORIGIN_MATCHED_CONTROL_STATE = "matched_control_state"

#: The target does NOT vary with the level: it is a CONSTANT of the system, and the level selects
#: which REFERENCE the prediction is measured to. A fourth construction rather than a
#: re-parameterisation of the other three, and the fourth is the one that dissolves a problem the
#: other three carry.
#:
#: The three above all express an ask as a number the prediction must reach, so each of them needs
#: that number to have a SIGN: ``+1`` must mean "more" of something. For a property whose quantity
#: is an unsigned DISTANCE TO A DEPOSITED STRUCTURE there is no such number. Which of two crystal
#: conformers is "more open" is not determined by the data -- D-77/D-78 measured exactly that on
#: the cryptic pool and found 32.1% of systems carrying the wrong sign -- so a signed target would
#: be a per-protein coin flip baked into the objective.
#:
#: Under this origin there is nothing to flip. The target is the system's own threshold ``tau_p``,
#: identical at both levels; ``+1`` asks for the distance to conformer j to fall below it and
#: ``-1`` for the distance to conformer i, at the SHARED draw seed the two levels already carry
#: (``SteeringUnit.draw_seed``). The incoherent ask goes with the sign: under a signed coordinate
#: level ``-1`` asks an already-i-like draw to become more i-like, which names no structural
#: change, whereas here it asks for a distance to be small, which it either already is or is not.
#:
#: **Named for the CONSTRUCTION, not for the property.** Any property whose level names a reference
#: rather than a magnitude uses this; the cryptic direct-metric term is the second one queued for
#: it. ``TWO_STATE_TARGET_LEVELS`` is unchanged and is read exactly as it stands -- this is an
#: interpretation of the two existing levels, not a new dose set.
TARGET_ORIGIN_SYSTEM_CONSTANT = "system_constant"

RELATIVE_TARGET_ORIGINS = frozenset(
    {
        TARGET_ORIGIN_SEQUENCE_MEAN,
        TARGET_ORIGIN_MATCHED_DRAW_ZERO,
        TARGET_ORIGIN_MATCHED_CONTROL_STATE,
        TARGET_ORIGIN_SYSTEM_CONSTANT,
    }
)

#: The relative origins whose target is a multiple of a per-sequence SIGMA, i.e. the ones that
#: carry a dose multiplier ``k``. Membership, not equality, because three separate rules key off
#: this one question -- whether ``dose_multiplier`` is required, whether it is recorded in the
#: checkpoint identity, and whether an intermediate dose level means anything -- and until this
#: existed all three were spelled as "is the origin ``matched_control_state``", which was true of
#: the two-state properties by coincidence rather than because being two-state is what removes the
#: multiplier.
#:
#: The complement is the set of origins whose level names a DISCRETE thing: a state, or a
#: reference. For those, ``0.0`` is refused as a target level rather than mapped to a midpoint --
#: "half open" is not a conformation and there is no third crystal -- and there is no multiple of
#: a standard deviation left to set.
SIGMA_DOSE_TARGET_ORIGINS = frozenset(
    {TARGET_ORIGIN_SEQUENCE_MEAN, TARGET_ORIGIN_MATCHED_DRAW_ZERO}
)

#: The reporting-frame value each two-state dose level names. ``+1`` is the holo / DFG-out state and
#: ``-1`` is the apo / DFG-in state, and these are the literal numbers because ``the progress coordinate`` is built so
#: that the experimental holo scores exactly 1 and the experimental apo exactly 0 for every protein.
TWO_STATE_REPORTING_TARGETS = {1.0: 1.0, -1.0: 0.0}


@dataclass(frozen=True)
class DoseLevels:
    """One property's steered dose-level sets: what training walks and what validation walks.

    A bundle rather than two loose tuples because the two are chosen together and must be
    recorded together: a run that changed only the validation set would otherwise write an
    identical checkpoint identity while selecting on a different quantity.
    """

    name: str
    target: tuple[float, ...]
    epoch_validation: tuple[float, ...]
    #: Declare that this schedule walks ONE direction on purpose, which lifts the both-signs
    #: requirement below. It must be asked for BY NAME, never inferred from a one-signed tuple,
    #: because the failure this guards against -- a schedule that quietly lost a direction -- and
    #: the design below are indistinguishable from the tuple alone.
    #:
    #: WHY A ONE-DIRECTION SCHEDULE EXISTS (D-101). The level-conditioned adapter's response is
    #: EVEN in the level: measured on the trained arms, ``||odd|| / ||even||`` is 0.0133 (cryptic)
    #: and 0.0099 (domain motion), with ``cos(r(+1), -r(-1))`` at -0.9998 and -0.9999. Roughly 1%
    #: of what the adapter does is directional, which is why level -1 and +1 draws correlate at
    #: 1.000 and why two disjoint kinase pools score identically. Making the residual odd by
    #: construction (``raw = raw * level``) is what ``TargetKind.DIRECTIONAL_RESIDUE`` does and
    #: what D2 forbids for ``RELATIVE_SEQUENCE``, so the remaining route to a directional arm is
    #: to REMOVE the level: one adapter per direction, conditioned on the hidden activations
    #: alone. With no level input there is nothing to symmetrise, so evenness cannot arise.
    #:
    #: The both-signs guard is still right for every schedule that HAS two signs, and stays on by
    #: default: it is what catches a one-sided artifact by forcing the ranking onto the worse
    #: direction. A single-direction arm gives that guard up knowingly, and buys it back across
    #: the PAIR of arms rather than within one.
    single_direction: bool = False

    def __post_init__(self) -> None:
        for field, values in (("target", self.target), ("epoch_validation", self.epoch_validation)):
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{field} levels must be a non-empty set of distinct values")
            if any(
                isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                for value in values
            ):
                raise ValueError(f"{field} levels must be finite real values")
        if self.single_direction:
            # Declared one-directional: require that it REALLY is, so the flag cannot be set on a
            # two-signed set and silently mean nothing.
            for field, values in (
                ("target", self.target),
                ("epoch_validation", self.epoch_validation),
            ):
                signs = {value > 0 for value in values if value != 0}
                if len(signs) > 1:
                    raise ValueError(
                        f"{field} levels declare single_direction but span both signs: {values!r}"
                    )
        else:
            # Both signs, always. A single-signed set removes the design's guard of ranking on the
            # worse direction, which is what catches a one-sided artifact.
            for field, values in (
                ("target", self.target),
                ("epoch_validation", self.epoch_validation),
            ):
                if not (any(value < 0 for value in values) and any(value > 0 for value in values)):
                    raise ValueError(
                        f"{field} levels must request both directions; a deliberately "
                        "one-directional schedule must set single_direction=True"
                    )
        if not set(self.epoch_validation) <= set(self.target):
            raise ValueError("epoch validation levels must be a subset of the target levels")


#: The dose set for a CONTINUOUS property, where an intermediate level names a real structural
#: ask. Radius of gyration is the only one today, and this is exactly the set it has always
#: used -- an Rg schedule, checkpoint and journal are bit-identical across D-44.
CONTINUOUS_DOSE_LEVELS = DoseLevels(
    name="continuous_five_level",
    target=TARGET_LEVELS,
    epoch_validation=EPOCH_VALIDATION_LEVELS,
)

#: The dose set for a TWO-STATE property: cryptic pockets and kinase DFG. Three points per
#: sequence, one of which is free because the matched unsteered draw is generated anyway and
#: supplies the zero point by construction.
#:
#: The saving is a consequence of the science rather than the motivation for it: 5 steered
#: levels to 2 takes a cryptic arm from roughly 30 GPU-h to 12-15, and held-out evaluation from
#: 6,090 structures to about 1,824.
TWO_STATE_DOSE_LEVELS = DoseLevels(
    name="two_state_bidirectional",
    target=TWO_STATE_TARGET_LEVELS,
    epoch_validation=TWO_STATE_EPOCH_VALIDATION_LEVELS,
)

#: The two ONE-DIRECTIONAL narrowings of the two-state set (D-101): each arm of a per-direction
#: pair walks exactly one of the property's two levels, at training, epoch validation and the
#: held-out test alike. Declared by NAME rather than built from a config's list, for two reasons.
#: A persisted schedule has to be matched against declared sets before its config is known
#: (:func:`_inventory_dose_levels`), and a set constructed on the fly would be exactly the freely
#: inferred grid that matching exists to refuse. And ``name`` is what the checkpoint identity
#: records under ``dose_levels``, so ``two_state_positive`` and ``two_state_bidirectional`` are
#: two checkpoints that cannot be mistaken for each other -- which mattered on the first
#: unconditioned domain-motion run, where the config declared ``[1.0]``, the gate accepted it, and
#: the schedule that was actually walked was the parent's: 23280 train units instead of 11640,
#: levels ``{-1.0: 49, 1.0: 48}`` after 97 units, and a level-blind adapter trained toward BOTH
#: conformers at once, which is the pathology the arm exists to escape.
TWO_STATE_POSITIVE_DOSE_LEVELS = DoseLevels(
    name="two_state_positive",
    target=(1.0,),
    epoch_validation=(1.0,),
    single_direction=True,
)
TWO_STATE_NEGATIVE_DOSE_LEVELS = DoseLevels(
    name="two_state_negative",
    target=(-1.0,),
    epoch_validation=(-1.0,),
    single_direction=True,
)

#: Which one-directional sets a property's reviewed set may be narrowed to, keyed by the parent's
#: name. Only the two-state set has any: a continuous property's one-signed subset would still
#: contain intermediate doses whose spacing carries the slope, and no arm has asked for one.
SINGLE_DIRECTION_DOSE_LEVELS: Mapping[str, tuple[DoseLevels, ...]] = {
    TWO_STATE_DOSE_LEVELS.name: (TWO_STATE_POSITIVE_DOSE_LEVELS, TWO_STATE_NEGATIVE_DOSE_LEVELS),
}

#: Every dose set this code produces schedules under. A persisted inventory whose levels match
#: none of these is a schedule this code did not write.
DECLARED_DOSE_LEVELS: tuple[DoseLevels, ...] = (
    CONTINUOUS_DOSE_LEVELS,
    TWO_STATE_DOSE_LEVELS,
    TWO_STATE_POSITIVE_DOSE_LEVELS,
    TWO_STATE_NEGATIVE_DOSE_LEVELS,
)


def single_direction_dose_levels(
    levels: DoseLevels, requested: Sequence[float]
) -> DoseLevels | None:
    """Return the declared one-directional narrowing of ``levels`` that walks exactly ``requested``.

    ``None`` when there is none: the parent has no declared narrowings, or ``requested`` is not one
    of them level for level and in order. The caller decides what ``None`` means -- the schedule
    gate falls through to the parent's own expectation and refuses the block on the mismatch.
    """
    for candidate in SINGLE_DIRECTION_DOSE_LEVELS.get(levels.name, ()):
        if list(candidate.target) == list(requested):
            return candidate
    return None


#: Domain separator for the per-unit property-ensemble generation seeds (D16). It is
#: distinct from ``steering-unit-seed``, so an ensemble member seed can never be derived
#: by the same input that derives a scheduled unit seed.
PROPERTY_ENSEMBLE_SEED_DOMAIN = "steering-property-ensemble-seed"

#: How the ``M`` generation seeds of one scheduled unit are built: member 0 is the unit's
#: own scheduled seed, so ``M = 1`` is exactly the pre-ensemble draw, and members
#: ``1..M-1`` are derived from that seed and then made unique against every other seed the
#: schedule uses.
PROPERTY_ENSEMBLE_SEED_RULE = "unit_seed_then_derived_globally_unique"

#: Domain separator for the shared noise draw. Distinct from ``steering-unit-seed`` so a
#: draw seed can never coincide with some other unit's identity seed by construction.
DRAW_SEED_DOMAIN = "steering-draw-seed"

_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SteeringUnit:
    """One deterministic sequence visit with a normalized continuous target level.

    ``seed`` is the unit's **identity**: globally unique, used for journalling, resume, and
    the ensemble-seed table. ``draw_seed`` is the **noise draw** the generator actually uses,
    and it is deliberately *shared* between the levels of one (sequence, cycle).

    They are separate because the two roles conflict. The unsteered radius of gyration varies
    by ~4.6 A between draws while the intervention's effect on it varies by ~0.006 A, so a
    difference taken across two levels at the *same* draw cancels the baseline exactly and
    leaves the dose signal clean -- measured SNR 0.62 unpaired against 57 paired. Deriving one
    shared seed directly would not work: ``_allocate_unique_seeds`` resolves collisions by
    incrementing, so deliberately equal seeds are pulled apart again and the pairing silently
    does nothing.

    ``None`` means "use ``seed``", which is exactly the pre-pairing behaviour, so every
    schedule built before this field existed is unchanged.
    """

    split: str
    sequence_id: str
    cycle: int
    visit: int
    level: float
    seed: int
    draw_seed: int | None = None

    @property
    def generation_seed(self) -> int:
        """The seed the generator draws from: the shared draw when one is set."""
        return self.seed if self.draw_seed is None else self.draw_seed


@dataclass(frozen=True)
class SteeringSchedule:
    """Immutable train, validation, and held-out steering visits."""

    train: tuple[SteeringUnit, ...]
    validation: tuple[SteeringUnit, ...]
    test: tuple[SteeringUnit, ...]


@dataclass(frozen=True)
class LocalQControlUnit:
    """One inventory-bound directional residue control for Local Q."""

    kind: str
    level: float
    control: tuple[float, ...]
    start: int | None = None
    stop: int | None = None


def continuous_target(
    baseline: float, bounds: TargetRange, maximum_change: float, level: float
) -> float:
    """Return a target that preserves equal headroom in both directions."""
    _require_finite_real("baseline", baseline)
    if not isinstance(bounds, TargetRange):
        raise ValueError("bounds must be a TargetRange")
    _require_finite_real("maximum_change", maximum_change)
    _require_finite_real("level", level)
    if not bounds.lower <= baseline <= bounds.upper:
        raise ValueError("baseline must remain within bounds")
    if maximum_change < 0:
        raise ValueError("maximum_change must be nonnegative")
    if not -1.0 <= level <= 1.0:
        raise ValueError("level must be between -1 and 1")

    scale = min(maximum_change, baseline - bounds.lower, bounds.upper - baseline)
    return baseline + scale * level


def relative_sequence_target(
    sequence_mean: float,
    sequence_sigma: float,
    level: float,
    *,
    dose_multiplier: float = 1.0,
) -> float:
    """Return ``t = mean_seq + level * k * sigma_seq`` for one sequence.

    Implements D3/D4: level 0 targets the sequence's own adapter-off ensemble
    mean and level +/-1 requests ``k`` standard deviations of that sequence's own
    conformational spread, so the dose is expressed in each protein's own units
    rather than a fixed absolute shift. ``sequence_mean`` and ``sequence_sigma``
    are caller-supplied per-sequence statistics; this function reads no
    structure store or annotation file.
    """
    _require_finite_real("sequence_mean", sequence_mean)
    _require_finite_real("sequence_sigma", sequence_sigma)
    _require_finite_real("level", level)
    _require_finite_real("dose_multiplier", dose_multiplier)
    if sequence_sigma <= 0:
        raise ValueError("sequence_sigma must be positive")
    if dose_multiplier <= 0:
        raise ValueError("dose_multiplier must be positive")
    if not -1.0 <= level <= 1.0:
        raise ValueError("level must be between -1 and 1")

    return float(sequence_mean) + float(level) * float(dose_multiplier) * float(sequence_sigma)


def relative_displacement_target(
    sequence_sigma: float,
    level: float,
    *,
    dose_multiplier: float = 1.0,
) -> float:
    """Return ``t = level * k * sigma_seq``: a pure displacement, with no origin (D-46).

    For a property whose prediction is measured RELATIVE to a matched unsteered draw, the zero
    point is zero by construction -- an adapter-off draw scores exactly 0 against itself. The
    target must therefore be a displacement and nothing else.

    :func:`relative_sequence_target` adds the reference table's ``mean``, which is right when the
    prediction is an absolute quantity in the same frame as that mean, and wrong here. The
    cryptic-pocket arm used it for a day: ``mean_the progress coordinate`` is measured in the REPORTING frame (against
    the experimental apo structure) because in the training frame it would be identically 0 for
    every protein, so adding it asked every level to move by an absolute position measured against
    a different reference, on top of the requested displacement.

    The scale is unchanged and is deliberately still the reporting-frame ``sigma``. ``the progress coordinate`` is a
    weighted inner product with the transition vector and is therefore LINEAR in the pairwise
    distance vector, so ``the progress coordinate_training(s) = the progress coordinate_reporting(s) - the progress coordinate_reporting(reference)`` exactly:
    the two frames differ by an origin, not by a scale, and a displacement expressed in units of
    the protein's own reporting-frame spread is expressed in the right units.

    **No two-state property uses this any more.** The cryptic-pocket and kinase programmes moved to
    :func:`two_state_displacement_target` / :data:`TARGET_ORIGIN_MATCHED_CONTROL_STATE`: a sigma
    multiple names no state, and the sigma denominator it implies is not the normaliser an absolute
    state target needs. This function is kept because it is the correct construction for a
    displacement-valued CONTINUOUS property, which is a shape the repository may want again; it is
    not a fallback for a two-state one.
    """
    _require_finite_real("sequence_sigma", sequence_sigma)
    _require_finite_real("level", level)
    _require_finite_real("dose_multiplier", dose_multiplier)
    if sequence_sigma <= 0:
        raise ValueError("sequence_sigma must be positive")
    if dose_multiplier <= 0:
        raise ValueError("dose_multiplier must be positive")
    if not -1.0 <= level <= 1.0:
        raise ValueError("level must be between -1 and 1")
    return float(level) * float(dose_multiplier) * float(sequence_sigma)


def two_state_reporting_target(level: float) -> float:
    """Return the REPORTING-frame state one two-state dose level names: 1.0 or 0.0.

    ``+1`` is the holo / DFG-out state and ``-1`` is the apo / DFG-in state. There is no
    multiplier and no per-protein scale, because ``the progress coordinate`` already carries the only normalisation an
    absolute state target needs: ``the progress coordinate = <u,v>_w / ||v||^2_w`` with ``v = d(holo) - d(apo)``, so at
    the experimental holo ``u = v`` and ``the progress coordinate = 1`` EXACTLY, and at the experimental apo ``u = 0``
    and ``the progress coordinate = 0`` EXACTLY. The apo->holo transition is unit length in ``the progress coordinate`` for every protein by
    construction, which is precisely the per-protein normalisation ``sigma`` used to stand in for.

    ``0.0`` is refused rather than mapped to a midpoint. D-44 already removed it from
    :data:`TWO_STATE_TARGET_LEVELS`, and the reason it is refused HERE too is the reason this
    function exists: "half open" names no state the protein has, so there is no reporting-frame
    value to return for it.
    """
    _require_finite_real("level", level)
    if not -1.0 <= level <= 1.0:
        raise ValueError("level must be between -1 and 1")
    state = TWO_STATE_REPORTING_TARGETS.get(float(level))
    if state is None:
        raise ValueError(
            "a two-state level names a state, so it must be exactly -1 (apo / DFG-in) or +1 "
            f"(holo / DFG-out); received {float(level)}. An intermediate dose has no "
            "reporting-frame state to name -- 'half open' is not a conformation -- which is why "
            "TWO_STATE_TARGET_LEVELS is (-1, 1) and why there is no midpoint to fall back to"
        )
    return float(state)


def two_state_displacement_target(control_reporting_value: float, level: float) -> float:
    """Return ``t = the progress coordinate_state(o) - the progress coordinate_reporting(control)``: the state ask, in the TRAINING frame.

    "Reach holo" is a REPORTING-frame statement -- ``the progress coordinate = 1`` there, for every protein. The
    prediction is a TRAINING-frame displacement against a matched unsteered draw, so the ask has to
    be converted, and the conversion is exact rather than approximate: ``the progress coordinate`` is a weighted inner
    product with the transition vector and is therefore LINEAR in the pairwise distance vector, so

        the progress coordinate_training(s) = the progress coordinate_reporting(s) - the progress coordinate_reporting(control)

    holds identically. Subtracting the control's own reporting-frame value is that identity, read
    backwards. ``control_reporting_value`` must therefore be THIS draw's own matched control, in
    the reporting frame (anchored on the experimental apo / deposited DFG-in structure).

    **Not the D-46 defect, and the difference is the whole point.** D-46 removed the reference
    table's ``mean_the progress coordinate`` from a displacement target: a per-protein CONSTANT, the same wrong offset
    on every draw of a protein, invisible to a dose slope because it is an intercept. Here the
    origin varies per draw and is the correct one for that draw. Substituting ``mean_the progress coordinate`` for it
    would be an approximation AND would recreate D-46; there is deliberately no such fallback.

    No ``k`` and no ``sigma`` appear, and neither is missing by omission. ``k`` expressed the ask as
    a number of standard deviations, which is the right parameterisation for a CONTINUOUS property
    (radius of gyration keeps :func:`relative_sequence_target` for exactly that reason) and the
    wrong one for a two-state property. ``sigma`` was the denominator that made a sigma-scaled ask
    comparable across proteins; an absolute state target on a coordinate that is unit-length
    apo->holo by construction is already comparable, so the denominator has nothing left to do.
    """
    _require_finite_real("control_reporting_value", control_reporting_value)
    return two_state_reporting_target(level) - float(control_reporting_value)


@dataclass(frozen=True)
class SystemTargetConstant:
    """One system's own target and its own scale, under :data:`TARGET_ORIGIN_SYSTEM_CONSTANT`.

    Two numbers, both per protein and both strictly positive, both measured offline from that
    system's own reference structures:

    * :attr:`target` is the value the prediction is asked to reach at EVERY level -- the system's
      own success threshold.
    * :attr:`scale` is what the error is divided by, i.e. the ``property_scale`` the objective's
      absolute channel receives.

    **A per-protein SCALE in the objective is a real cost and is stated rather than hidden.** D-46
    removed a per-protein CONSTANT from a displacement target because it was an intercept no dose
    slope could see; this is not that -- it is a scale on a quantity whose zero is meaningful --
    but two systems' losses are still not in one unit, and the reason to accept that is the
    metric's own aggregation. ``success@B`` is an UNWEIGHTED MEAN OVER SYSTEMS of a bounded
    indicator, while a per-draw loss is a mean over draws of an unbounded quantity, so without a
    per-system scale the objective's implicit weighting is "largest transitions first" and the
    metric's is "every system once". The table reaches the checkpoint identity by digest, so a
    report can always say which weight each protein's draws carried.
    """

    target: float
    scale: float

    def __post_init__(self) -> None:
        for name, value in (("target", self.target), ("scale", self.scale)):
            _require_finite_real(f"system target {name}", value)
            if float(value) <= 0.0:
                raise ValueError(f"a system target {name} must be positive, received {value!r}")


@dataclass(frozen=True)
class SystemTargetConstants:
    """The per-system ``{target, scale}`` table one ``system_constant`` run was built from.

    A wrapper rather than a bare mapping so that the table can carry its own IDENTITY. The pair of
    numbers per protein is a per-protein weight on the objective, so a checkpoint has to be bound
    to the table it descended; a digest is recorded rather than the whole table, because a pool of
    a thousand systems would otherwise put two thousand floats into every ``adapter_config`` and
    into every fingerprint comparison.

    ``entries`` is ordered by sequence id for the digest and only for the digest; lookups are by
    name and a missing system fails closed, because the direct metric has no fallback reference
    and therefore no fallback threshold either.
    """

    entries: Mapping[str, SystemTargetConstant]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, Mapping) or not self.entries:
            raise ValueError("a system target table must carry at least one system")
        for key, value in self.entries.items():
            if not isinstance(key, str) or not key:
                raise ValueError("every system target entry must be keyed by a sequence id")
            if not isinstance(value, SystemTargetConstant):
                raise ValueError(f"{key!r} must map to a SystemTargetConstant")

    def require(self, sequence_id: str) -> SystemTargetConstant:
        entry = self.entries.get(sequence_id)
        if entry is None:
            raise KeyError(
                f"{sequence_id!r} has no system target constant. Under "
                f"{TARGET_ORIGIN_SYSTEM_CONSTANT!r} the target IS this system's own threshold and "
                "the scale IS its own reference separation, so there is nothing to fall back to "
                "-- a pooled default would apply one protein's weight to another's draws"
            )
        return entry

    def identity(self) -> dict[str, object]:
        """The block a checkpoint binds this table by: how many systems, and their digest."""
        digest = hashlib.sha256()
        digest.update(b"steering-system-target-constants:v1")
        for sequence_id in sorted(self.entries):
            entry = self.entries[sequence_id]
            digest.update(sequence_id.encode("utf-8"))
            # Rounded before hashing for the reason the domain-motion artifact rounds its
            # coordinates: both numbers are measurements whose uncertainty is far above 1e-6, and
            # a digest sensitive to the last float bit would make the table irreproducible across
            # BLAS versions while binding nothing scientific.
            digest.update(f"{float(entry.target):.6f}|{float(entry.scale):.6f}".encode())
        return {"systems": len(self.entries), "content_fingerprint": digest.hexdigest()}


def system_constant_target(constant: SystemTargetConstant, level: float) -> float:
    """Return ``t = target_p``: the system's own constant, whatever the level.

    The level is validated and then DELIBERATELY UNUSED, and that is the whole content of this
    construction. Under :data:`TARGET_ORIGIN_SYSTEM_CONSTANT` a level names which reference the
    prediction is measured to, not how far it must move, so a target that varied with it would be
    asking for two different thresholds on two different structures and calling the pair a dose
    response.

    ``0.0`` is refused for the reason :func:`two_state_reporting_target` refuses it: an
    intermediate level names no reference. There is no third deposited structure, and "half way
    between the two crystals" is not a conformer the protein has and not a thing a metric can
    score. The refusal lives here as well as in the property's own reference lookup because this
    is where a SCHEDULED UNIT becomes a target, and a schedule carrying a level the property
    cannot honour should fail before anything generates.
    """
    if not isinstance(constant, SystemTargetConstant):
        raise ValueError("system_constant_target needs a SystemTargetConstant")
    _require_finite_real("level", level)
    if float(level) not in TWO_STATE_REPORTING_TARGETS:
        raise ValueError(
            "a system-constant level NAMES A REFERENCE, so it must be exactly -1 or +1; received "
            f"{float(level)}. The prediction is an unsigned distance to a deposited structure and "
            "there is no third structure for an intermediate level to be measured against"
        )
    return float(constant.target)


def build_property_ensemble_seeds(
    schedule: SteeringSchedule, samples: int
) -> dict[int, tuple[int, ...]]:
    """Return each scheduled unit's ``samples`` generation seeds, keyed by its own seed.

    D16: with ``samples > 1`` a training or validation unit generates ``M`` structures and
    the property loss is applied once to their mean. Those ``M`` draws must be genuinely
    different draws of the *same* request, so each needs its own seed, and those seeds must
    be deterministic, reproducible from the schedule alone, and distinct from every other
    unit's -- otherwise two units would share a draw and the ensemble mean would not be an
    average over independent structures.

    Member 0 is the unit's **own** scheduled seed, so ``samples = 1`` returns exactly the
    pre-ensemble draw and an ``M``-member ensemble still contains the structure a one-member
    run would have produced. Members ``1..M-1`` are derived with the same
    :func:`_derived_integer` construction the schedule's own seeds use, under a distinct
    domain separator, and collisions are then resolved by increment against every seed
    already in use -- the schedule's own seeds included -- exactly as
    :func:`_allocate_unique_seeds` does. Units are visited in canonical identity order, not
    in list order, so a schedule reordered for length bucketing (D8) derives the same table.

    Keyed by unit seed because a validated schedule's seeds are globally unique
    (``_validate_schedule``); that is checked here rather than assumed.
    """
    if not isinstance(schedule, SteeringSchedule):
        raise ValueError("schedule must be a SteeringSchedule")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError("property ensemble samples must be a positive integer")
    units = (*schedule.train, *schedule.validation, *schedule.test)
    scheduled_seeds = [unit.seed for unit in units]
    if len(set(scheduled_seeds)) != len(scheduled_seeds):
        raise ValueError(
            "schedule seeds must be globally unique before property-ensemble seeds are "
            "derived; two units sharing a seed would share their whole ensemble"
        )
    used = set(scheduled_seeds)
    plan: dict[int, tuple[int, ...]] = {}
    # Keyed by generation seed first, so units sharing a draw are assigned one member list
    # rather than colliding through the uniqueness loop and being pulled apart.
    by_draw: dict[int, tuple[int, ...]] = {}
    for unit in sorted(units, key=_unit_identity):
        if unit.generation_seed in by_draw:
            plan[unit.seed] = by_draw[unit.generation_seed]
            continue
        # Derived from the *generation* seed, not the identity seed, so two units sharing a
        # draw share their entire member list. Otherwise pairing would cancel the baseline
        # only for member 0 and leave the other M-1 members unpaired.
        members = [unit.generation_seed]
        for index in range(1, samples):
            candidate = _derived_integer(
                PROPERTY_ENSEMBLE_SEED_DOMAIN, str(unit.generation_seed), str(index)
            ) & ((1 << 32) - 1)
            start = candidate
            while candidate in used:
                candidate = (candidate + 1) % (1 << 32)
                if candidate == start:  # pragma: no cover - 2**32 seeds are never in use.
                    raise RuntimeError("32-bit steering seed space is exhausted")
            used.add(candidate)
            members.append(candidate)
        by_draw[unit.generation_seed] = tuple(members)
        plan[unit.seed] = tuple(members)
    return plan


def build_local_q_control_units(
    sequence_id: str,
    eligible_residues: tuple[bool, ...],
    *,
    manifest_fingerprint: str,
    epoch: int,
    window_size: int = 8,
) -> tuple[LocalQControlUnit, ...]:
    """Build deterministic null, localized, and moderate global Local-Q controls."""
    if not isinstance(sequence_id, str) or not sequence_id:
        raise ValueError("sequence_id must be non-empty")
    if (
        not isinstance(eligible_residues, tuple)
        or not eligible_residues
        or any(not isinstance(value, bool) for value in eligible_residues)
        or not any(eligible_residues)
    ):
        raise ValueError("at least one contact-eligible residue is required")
    if not isinstance(manifest_fingerprint, str) or not _FINGERPRINT_RE.fullmatch(
        manifest_fingerprint
    ):
        raise ValueError("manifest_fingerprint must be a lowercase SHA-256 fingerprint")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
        raise ValueError("window_size must be a positive integer")

    length = len(eligible_residues)
    windows: list[tuple[int, int]] = []
    run_start = 0
    while run_start < length:
        if not eligible_residues[run_start]:
            run_start += 1
            continue
        run_stop = run_start
        while run_stop < length and eligible_residues[run_stop]:
            run_stop += 1
        width = min(window_size, run_stop - run_start)
        windows.extend((start, start + width) for start in range(run_start, run_stop - width + 1))
        run_start = run_stop

    zero = (0.0,) * length
    units = [LocalQControlUnit("null", 0.0, zero)]
    for index, level in enumerate((-1.0, -0.5, 0.5, 1.0)):
        offset = _derived_integer(
            "local-q-window",
            manifest_fingerprint,
            sequence_id,
            str(epoch),
            str(index),
        ) % len(windows)
        start, stop = windows[(offset + epoch) % len(windows)]
        control = tuple(level if start <= residue < stop else 0.0 for residue in range(length))
        units.append(LocalQControlUnit("localized", level, control, start, stop))
    for level in (-0.5, -0.25, 0.5, 1.0):
        control = tuple(level if eligible else 0.0 for eligible in eligible_residues)
        units.append(LocalQControlUnit("global", level, control))
    return tuple(units)


def build_steering_schedule(
    split: SequenceSplit,
    *,
    manifest_fingerprint: str,
    cycles: int,
    pair_levels: bool = False,
    levels: DoseLevels = CONTINUOUS_DOSE_LEVELS,
) -> SteeringSchedule:
    """Derive order-invariant train and fixed held-out steering schedules.

    ``pair_levels`` emits a cycle **sequence-major** -- every level of one protein together --
    instead of the default visit-major. The paired dose term differences two levels of one
    sequence, so both must land in the same micro-batch; under visit-major they sit
    ``len(sequences)`` units apart and no pair would ever form, leaving the term silently
    zero. It is off by default because visit-major decorrelates consecutive optimizer steps
    (D18), which is the better trade when nothing needs the pairing.

    ``levels`` is the property's dose set (D-44). The default is the continuous five-level set,
    so every existing caller -- radius of gyration and AF3 -- builds the schedule it always
    built, seed for seed. A two-state property passes :data:`TWO_STATE_DOSE_LEVELS`.
    """
    if not isinstance(split, SequenceSplit):
        raise ValueError("split must be a SequenceSplit")
    if not isinstance(manifest_fingerprint, str) or not _FINGERPRINT_RE.fullmatch(
        manifest_fingerprint
    ):
        raise ValueError("manifest_fingerprint must be a lowercase SHA-256 fingerprint")
    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 1:
        raise ValueError("cycles must be a positive integer")

    return _allocate_unique_seeds(
        SteeringSchedule(
            train=_build_training_units(
                split.train,
                manifest_fingerprint,
                cycles,
                pair_levels=pair_levels,
                levels=levels.target,
            ),
            validation=_build_evaluation_units(
                "validation", split.validation, manifest_fingerprint, levels.epoch_validation
            ),
            test=_build_evaluation_units("test", split.test, manifest_fingerprint, levels.target),
        )
    )


def _allocate_unique_seeds(schedule: SteeringSchedule) -> SteeringSchedule:
    """Resolve candidate collisions in one deterministic global uint32 namespace."""
    all_units = (*schedule.train, *schedule.validation, *schedule.test)
    assigned: dict[tuple[str, str, int, int, float], int] = {}
    used: set[int] = set()
    for unit in sorted(all_units, key=_unit_identity):
        candidate = unit.seed
        while candidate in used:
            candidate = (candidate + 1) % (1 << 32)
            if candidate == unit.seed:
                raise RuntimeError("32-bit steering seed space is exhausted")
        used.add(candidate)
        assigned[_unit_identity(unit)] = candidate

    def resolve(units: tuple[SteeringUnit, ...]) -> tuple[SteeringUnit, ...]:
        return tuple(replace(unit, seed=assigned[_unit_identity(unit)]) for unit in units)

    return SteeringSchedule(
        train=resolve(schedule.train),
        validation=resolve(schedule.validation),
        test=resolve(schedule.test),
    )


def _unit_identity(unit: SteeringUnit) -> tuple[str, str, int, int, float]:
    return (unit.split, unit.sequence_id, unit.cycle, unit.visit, unit.level)


def _build_training_units(
    sequence_ids: Iterable[str],
    manifest_fingerprint: str,
    cycles: int,
    *,
    pair_levels: bool = False,
    levels: tuple[float, ...] = TARGET_LEVELS,
) -> tuple[SteeringUnit, ...]:
    ranked_ids = sorted(
        sequence_ids,
        key=lambda sequence_id: (
            _derived_integer(
                "steering-training-sequence-rank",
                manifest_fingerprint,
                sequence_id,
            ),
            sequence_id,
        ),
    )
    # Every sequence is visited at *every* target level in every cycle, so one epoch carries
    # ``len(sequences) x len(levels)`` units. The earlier protocol gave each sequence
    # one rotating level per cycle, `TARGET_LEVELS[(rank + cycle) % 5]`, which meant a
    # sequence was never seen at two doses inside an epoch: dose response had to be inferred
    # across *different* proteins, and per-sequence contrast accumulated only across cycles,
    # so at five cycles each sequence contributed exactly one observation per level for the
    # whole run.
    #
    # Emission is **visit-major**: every sequence at visit 0, then every sequence at visit 1.
    # Sequence-major was tried first because it pads to nothing and puts several doses of one
    # protein in a batch. Measured against the real batching config that argument fails: 209
    # of 222 batches per cycle are already single-sequence, so the doses rarely share a batch,
    # and where they do the symmetric levels partially cancel -- mean |sum of levels| 0.71
    # against a per-unit scale of 1 -- so the batch gradient encodes the protein's mean rather
    # than its slope. It also handed Adam five consecutive steps on one protein and shrank the
    # length-bucketing window from 32 distinct sequences to 6.4, invalidating the measured
    # 1.90x. Visit-major costs padding and buys decorrelation, which is the right trade when
    # the quantity being estimated is a dose slope.
    #
    # `rank` still sets where a sequence's level cycle *starts*, so the order in
    # which a sequence meets the levels still varies between sequences and cycles.
    return tuple(
        SteeringUnit(
            split="train",
            sequence_id=sequence_id,
            cycle=cycle,
            visit=visit,
            level=levels[(rank + cycle + visit) % len(levels)],
            seed=_derived_integer(
                "steering-unit-seed",
                manifest_fingerprint,
                "train",
                sequence_id,
                str(cycle),
                str(visit),
            )
            & ((1 << 32) - 1),
            # No ``visit``: every level of one (sequence, cycle) shares this draw, which is
            # what makes a paired difference across levels cancel the baseline.
            draw_seed=_derived_integer(
                DRAW_SEED_DOMAIN,
                manifest_fingerprint,
                "train",
                sequence_id,
                str(cycle),
            )
            & ((1 << 32) - 1),
        )
        for cycle in range(cycles)
        for visit, (rank, sequence_id) in
        (
            # Sequence-major puts a protein's levels adjacent so a batch can pair them;
            # visit-major interleaves proteins so consecutive steps are decorrelated.
            ((v, r) for r in enumerate(ranked_ids) for v in range(len(levels)))
            if pair_levels
            else ((v, r) for v in range(len(levels)) for r in enumerate(ranked_ids))
        )
    )


def _build_evaluation_units(
    split_name: str,
    sequence_ids: Iterable[str],
    manifest_fingerprint: str,
    levels: tuple[float, ...],
) -> tuple[SteeringUnit, ...]:
    units = []
    for sequence_id in sorted(sequence_ids):
        offset = _derived_integer(
            "steering-target-level-offset",
            manifest_fingerprint,
            split_name,
            sequence_id,
            "0",
        ) % len(levels)
        for visit in range(len(levels)):
            units.append(
                SteeringUnit(
                    split=split_name,
                    sequence_id=sequence_id,
                    cycle=0,
                    visit=visit,
                    level=levels[(offset + visit) % len(levels)],
                    seed=_derived_integer(
                        "steering-unit-seed",
                        manifest_fingerprint,
                        split_name,
                        sequence_id,
                        "0",
                        str(visit),
                    )
                    & ((1 << 32) - 1),
                )
            )
    return tuple(units)


def _derived_integer(domain: str, *parts: str) -> int:
    digest = hashlib.sha256()
    for part in (domain, *parts):
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "big")


def _require_finite_real(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
