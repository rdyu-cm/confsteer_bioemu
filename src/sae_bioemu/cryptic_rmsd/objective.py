"""The DIRECT-METRIC cryptic-pocket objective: an Angstrom RMS deviation to a deposited conformer.

SCOPE: CRYPTIC POCKETS, AND THERE IS NO ``the progress coordinate`` IN THIS FILE. Everything here is a function of
coordinates, a deposited reference conformer and a dose level. Nothing reads a file, a config or a
checkpoint, and nothing here knows what a progress coordinate is.

THE OBJECTIVE
-------------

Per (sequence ``p``, dose level ``o``, draw)::

    X = N/CA/C/O of the steered draw, over the benchmark's metric region
    Y = the same atoms of the deposited conformer the LEVEL names (apo or holo)
    m = aligned_rmsd(X, Y)                        Angstrom; alignment set == metric set
    L = 0.5 * (m / s_p)^2                         PLAIN MSE on the per-system scale

and the property term the shared loss prices is ``mean over observations of L``.

WHY THIS IS A SECOND PROPERTY AND NOT A REPOINTED ``pocket_progress_coordinate``
---------------------------------------------------------------------------------

``pocket_progress_coordinate`` builds its target under
:data:`~sae_bioemu.steering.schedule.TARGET_ORIGIN_MATCHED_CONTROL_STATE`: the ask is a
reporting-frame ``the progress coordinate`` state converted against THIS draw's own matched unsteered control, and every
cryptic checkpoint on disk records ``relative_target_origin: matched_control_state`` in its
``adapter_config``, which the evaluation gate compares by canonical-JSON equality. Repointing that
property at :data:`~sae_bioemu.steering.schedule.TARGET_ORIGIN_SYSTEM_CONSTANT` would therefore
refuse every one of them -- and the system-constant origin is not "the same construction with
``the progress coordinate`` removed" but a complete alternative one with its own hard requirements (a per-system
``{target, scale}`` table, no ``property_reference``, no ``sigma_eligibility``, no
``dose_multiplier``, ``sequence_sigma`` normalisation and ``lambda_y = 0``).

So this is a NEW property, which is the route ``domain_motion_backbone_rmsd`` and
``kinase_conformer_rmsd`` already took, for exactly this reason in the kinase case. Registering it
moves no existing checkpoint: see :mod:`sae_bioemu.cryptic_rmsd.property` and
``tests/test_cryptic_rmsd_registration.py``, which measures the claim rather than asserting it.

WHY A DIRECT METRIC AND NOT ``the progress coordinate``
-----------------------------------

D-82 measured the within-protein rank correlation between ``the progress coordinate`` and the deviation ``success@B``
grades: median Spearman **-0.574** against the experimental holo reference, **-0.601** restricted
to proteins BioEmu folds, with the expected sign in 85% and 90% of proteins respectively. So
``the progress coordinate`` is a VALIDATED but PARTIAL proxy -- ``|rho| ~ 0.57`` leaves roughly two thirds of the
within-protein rank variance unexplained. ``success_at_b.py`` states the hazard from the other
side: ``the progress coordinate = 1`` is a hyperplane containing holo and every structure at any off-axis distance, so
training on ``the progress coordinate`` and scoring on ``the progress coordinate`` is circular. Here the quantity descended and the quantity
graded are one quantity, in Angstrom, against a structure somebody deposited.

FIVE DESIGN POINTS
-------------------

1. **PLAIN MSE, AND THERE IS NO HINGE ON THIS PATH.** ``pocket_progress_coordinate``'s
   ``lambda_m`` term ships two forms, ``mse`` and ``huberised_hinge``, and this property ships
   only the first. That is not an oversight and it is not a simplification for its own sake:
   :data:`BENCHMARK_TAU_ANGSTROM` has TWO values of record and no default anywhere in this
   programme -- D-63 measured 30.5/60.6 at ``tau = 1.0, B = 100`` against 49/85 at
   ``tau = 1.5, B = 4000`` on the same model and the same 33 systems -- so a threshold inside the
   objective would make the 1.0 A and the 1.5 A readings of ONE run two readings of two
   differently-trained adapters. Under the system-constant origin the threshold would additionally
   enter the per-system target table's DIGEST, so the two readings would not even share a
   checkpoint identity. :func:`validated_metric_objective` therefore refuses the hinge spelling
   BY NAME rather than silently accepting a form this package does not implement.

2. **THE LEVEL SELECTS WHICH REFERENCE, NOT A SIGN.** A deviation is unsigned and ``success@B``
   scores apo and holo SEPARATELY, so ``+1`` means "minimise the deviation to the deposited holo"
   and ``-1`` "to the deposited apo", at the shared ``draw_seed`` the two levels of one
   (sequence, cycle) already carry. This removes the per-protein sign-identifiability problem of
   D-77/D-78 -- there is no sign to invert -- and the incoherent ask, since "be more open when you
   are already open" names no structural change while "be closer to this crystal" either already
   holds or does not.

3. **PER-SYSTEM NORMALISATION.** ``s_p`` is that system's own deposited apo<->holo separation over
   its own metric region. ``success@B`` is an UNWEIGHTED MEAN OVER SYSTEMS of a bounded indicator
   while a per-draw Angstrom loss is a mean over draws of an unbounded quantity, so without the
   scale the objective's implicit weighting is "largest transitions first" and the metric's is
   "every system once". Stated rather than buried: this is a per-protein number inside the
   objective, of the kind D-46 removed from the ``the progress coordinate`` target. It is not that defect -- D-46 added
   a per-protein CONSTANT to a displacement target, an intercept no dose slope could see, and this
   is a SCALE on a quantity whose zero is meaningful -- but two systems' losses are still not in
   one unit, and the table's digest is in the checkpoint identity so a report can say which weight
   each protein's draws carried.

4. **NO SOFT-MIN OVER DRAWS.** ``success@B`` takes a minimum over ``B`` draws. With two levels at
   one draw per scheduled unit there is no min to take inside a unit, and a soft-min over two
   values is nearly a hard min: it would route the whole gradient to one level and silently halve
   the schedule. ``B`` is a property of the REPORTED metric; the training objective descends every
   draw it is handed.

5. **THE TRAINED FORM IS THE RMS AND THE GRADED FORM IS THE ARITHMETIC MEAN.** Copied by value
   from the ``pocket_progress_coordinate`` term, whose module docstring carries the measurement:
   ``bioemu-benchmarks``' ``calc_metrics_local`` sets ``metrics[RMSD] = np.mean(pairwise_distances)``
   -- the arithmetic mean of per-atom deviations, misnamed ``rmsd`` upstream. **Kabsch minimises
   the RMS, not the mean**, so the envelope (Danskin) argument that makes the DETACHED rotation the
   exact gradient applies to the RMS form and to nothing else. By Jensen ``rmsd >= mean_deviation``
   pointwise, so the ask is SUFFICIENT for the graded one and merely TIGHTER; measured on this
   project's own geometry (``d81_native_success.csv``, 168 comparisons) the ratio runs 1.045 to
   1.339, about 13% at the median. That gap is the price of an exact gradient and it is paid
   knowingly: :data:`TRAINED_DEVIATION_FORM` and :data:`GRADED_DEVIATION_FORM` are separate named
   constants so no reader can take one for the other.

WHAT IS COPIED BY VALUE FROM ``cryptic_steering``, AND WHY IT IS NOT IMPORTED
------------------------------------------------------------------------------

:data:`CRYPTIC_METRIC_ATOMS`, :data:`MINIMUM_SEPARATION_ANGSTROM`, :data:`TRAINED_DEVIATION_FORM`,
:data:`GRADED_DEVIATION_FORM` and the two benchmark thresholds are copied BY VALUE from
:mod:`sae_bioemu.cryptic_steering.direct_metric`, and the arithmetic of :func:`metric_mse` is a
second implementation of that module's function rather than a call into it.

``direct_metric.py`` is a member of ``cryptic_training_only``, and ``scientific_code_paths``
refuses one source in two groups. So importing it would put ``cryptic_training_only`` -- and, for
the artifact reader, ``cryptic_generation_shared`` -- into THIS property's training composition,
binding every ``pocket_conformer_rmsd`` checkpoint to the ``the progress coordinate`` pair geometry, the D-31 curvature
bound, the site channel, the stage-1 setpoint reader and the cryptic loss, none of which a run of
this property executes. That is the over-binding half of the defect
:mod:`sae_bioemu.steering.property_registry` exists to end, and ``kinase_rmsd/frames.py`` sets the
precedent for paying a duplication to avoid it. ``tests/test_cryptic_rmsd_direct_metric.py`` pins
every copied constant and the arithmetic itself against the originals, so the copies cannot drift
into two objectives.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor

from sae_bioemu.steering.aligned_rmsd import aligned_rmsd

__all__ = [
    "APO",
    "BENCHMARK_TAU_ANGSTROM",
    "BIOEMU_TAU_ANGSTROM",
    "CONFORNETS_TAU_ANGSTROM",
    "CRYPTIC_METRIC_ATOMS",
    "DEFAULT_METRIC_OBJECTIVE",
    "GRADED_DEVIATION_FORM",
    "HOLO",
    "METRIC_OBJECTIVE_FORMS",
    "METRIC_OBJECTIVE_MSE",
    "MINIMUM_SEPARATION_ANGSTROM",
    "POCKET_RMSD_PROPERTY_NAME",
    "POOL_GATE_ANGSTROM",
    "REFERENCE_STATES",
    "TRAINED_DEVIATION_FORM",
    "PocketRmsdNormalisation",
    "aligned_backbone_rmsd",
    "metric_mse",
    "reference_state_for_level",
    "validated_metric_objective",
]

#: ``target.property_name`` for this property. Spelled once, here, and imported by the descriptor
#: and by every test, so the catalog cannot route a name no descriptor answers for.
#:
#: DISTINCT from ``pocket_progress_coordinate`` and deliberately not a variant spelling of it. The
#: two are different quantities -- an Angstrom deviation against a dimensionless coordinate --
#: measured over different atom sets against different references and built on different target
#: origins, and one name would pool them in every journal, every checkpoint identity and every
#: report that reads either.
POCKET_RMSD_PROPERTY_NAME = "pocket_conformer_rmsd"

#: The atoms the benchmark measures over. ``bioemu-benchmarks`` filters both reference and sample
#: to mdtraj's ``backbone`` selection, which is exactly these four, in this per-residue order.
#: COPIED BY VALUE from :data:`sae_bioemu.cryptic_steering.direct_metric.CRYPTIC_METRIC_ATOMS`;
#: see the module docstring for why it is not imported, and
#: ``tests/test_cryptic_rmsd_direct_metric.py`` pins the two equal.
CRYPTIC_METRIC_ATOMS: tuple[str, ...] = ("N", "CA", "C", "O")

#: The arithmetic mean of per-atom deviations: what ``calc_metrics_local`` labels ``rmsd`` and what
#: the reported ``success@B`` actually grades. NOT what this objective descends -- design point 5.
GRADED_DEVIATION_FORM = "mean_deviation"
#: The true root-mean-square of the same per-atom deviations, and what this objective DESCENDS.
#: Always ``>=`` :data:`GRADED_DEVIATION_FORM` by Jensen, and the only one of the two for which the
#: Kabsch rotation is stationary.
TRAINED_DEVIATION_FORM = "rmsd"

#: ConforNets' cryptic-pocket cutoff, in Angstrom.
CONFORNETS_TAU_ANGSTROM = 1.0
#: BioEmu's own cutoff, in Angstrom.
BIOEMU_TAU_ANGSTROM = 1.5
#: The two thresholds of record, for REPORTING only. Neither reaches this objective: not its value,
#: not its gradient, and not the digest of the per-system target table a checkpoint is bound to.
#: They are here because the arm resolver reports which systems sit inside them as a STRATUM and
#: because a success rate quoted without its tau is meaningless (design point 1).
BENCHMARK_TAU_ANGSTROM: tuple[float, ...] = (CONFORNETS_TAU_ANGSTROM, BIOEMU_TAU_ANGSTROM)

#: Plain MSE on the per-system scale, ``L = 0.5*(m/s_p)^2``. The ONLY form on this path.
METRIC_OBJECTIVE_MSE = "mse"
#: The forms this property implements, and there is exactly one. Design point 1.
METRIC_OBJECTIVE_FORMS: tuple[str, ...] = (METRIC_OBJECTIVE_MSE,)
#: What a reference that names no form descends.
DEFAULT_METRIC_OBJECTIVE = METRIC_OBJECTIVE_MSE

#: The smallest ``s_p`` a system may carry into the arm, in Angstrom. Not "greater than zero", and
#: that is a MEASURED distinction rather than caution: :func:`~sae_bioemu.steering.aligned_rmsd.
#: aligned_rmsd` adds ``1e-12`` inside its square root so the derivative stays finite at coincident
#: inputs, and therefore reports about **1e-6 Angstrom** for two identical structures rather than 0.
#: A ``separation > 0`` gate is satisfied by that floor, and ``s_p`` DIVIDES this objective: a
#: protein whose deposited conformers coincide would enter the batch at ``m / 1e-6`` and own every
#: gradient in it.
#:
#: COPIED BY VALUE from :data:`sae_bioemu.cryptic_steering.direct_metric.
#: MINIMUM_SEPARATION_ANGSTROM`, which is itself
#: :data:`sae_bioemu.cryptic_steering.pocket.COORDINATE_UNCERTAINTY_ANGSTROM`: below it the two
#: crystals are one structure to within this project's own stated coordinate uncertainty.
MINIMUM_SEPARATION_ANGSTROM = 0.5

#: THE POOL GATE IS THE DIVISOR FLOOR AND IT IS NOT A BENCHMARK ``tau``.
#:
#: ``domain_motion/separation.py`` gates its pool at ``s_p > tau = 3.0`` because a FIXED threshold
#: makes ``success@tau`` trivially satisfiable for a system whose conformers are closer than tau --
#: the draw sitting on conformer i already scores against conformer j. That premise does not hold
#: here, for the reason design point 1 gives: this objective has NO threshold at all, so there is
#: nothing a sub-tau system could satisfy without moving. What remains is the DIVISOR hazard, which
#: is a different failure, and it is gated strictly.
#:
#: Systems inside :data:`BENCHMARK_TAU_ANGSTROM` are a REPORTED STRATUM and never silently dropped:
#: :func:`~sae_bioemu.cryptic_rmsd.property.resolve_pocket_rmsd_property_arm` counts and names them
#: at every resolution. What they cannot support is a claim read against the fixed thresholds.
POOL_GATE_ANGSTROM = MINIMUM_SEPARATION_ANGSTROM

#: The two reference states, spelled as ``cryptic_steering.success_at_b`` spells them.
APO = "apo"
HOLO = "holo"

#: Which deposited conformer each two-state dose level NAMES. ``+1`` asks for holo (the
#: ligand-bound open-pocket crystal) and ``-1`` for apo. A REFERENCE selection rather than a sign
#: -- design point 2 -- and a reading of the existing
#: :data:`sae_bioemu.steering.schedule.TWO_STATE_DOSE_LEVELS` rather than a new dose set.
REFERENCE_STATES: Mapping[float, str] = {1.0: HOLO, -1.0: APO}


def reference_state_for_level(level: float) -> str:
    """Return ``"apo"`` or ``"holo"``: which deposited conformer this level asks for.

    ``0.0`` is refused rather than mapped to a midpoint. There is no third deposited structure,
    and "half way between the two crystals" is neither a conformer the protein has nor a thing
    ``success@B`` can score. The refusal is the same one
    :func:`~sae_bioemu.steering.schedule.system_constant_target` makes one layer out, where a
    SCHEDULED UNIT becomes a target; both exist because a deviation measured to the wrong
    structure is a NUMBER and not an error.
    """
    if isinstance(level, bool) or not isinstance(level, Real):
        raise ValueError("level must be a real number")
    state = REFERENCE_STATES.get(float(level))
    if state is None:
        raise ValueError(
            "a direct-metric level NAMES A DEPOSITED CONFORMER, so it must be exactly -1 (apo) or "
            f"+1 (holo); received {float(level)}. The metric is an unsigned deviation from a "
            "deposited structure and there is no third structure an intermediate level could be "
            "measured against"
        )
    return state


def validated_metric_objective(value: object) -> str:
    """Return :data:`METRIC_OBJECTIVE_MSE`, or fail closed naming what this path does not have.

    One validator, imported by the artifact loader, by the loss and by the property descriptor, so
    the form cannot be accepted under one spelling on disk and dispatched under another. ``None``
    resolves to :data:`DEFAULT_METRIC_OBJECTIVE`, which is what a reference written before the key
    existed means.

    The ``huberised_hinge`` spelling is refused BY NAME rather than falling through to the generic
    message, because it is the one wrong value a reader coming from
    ``cryptic_steering/direct_metric.py`` would expect to work: that module ships both forms and
    this one ships one, for the ``tau`` reason in design point 1.
    """
    if value is None:
        return DEFAULT_METRIC_OBJECTIVE
    if value == "huberised_hinge":
        raise ValueError(
            f"{POCKET_RMSD_PROPERTY_NAME} implements only {METRIC_OBJECTIVE_MSE!r} and the "
            "thresholded hinge is deliberately absent, not missing: tau is an EVALUATION "
            f"parameter with two values of record ({list(BENCHMARK_TAU_ANGSTROM)}) and no default, "
            "so a threshold inside this objective would make the 1.0 A and the 1.5 A readings of "
            "one run two readings of two differently-trained adapters -- and under the "
            "system_constant origin it would enter the per-system target table's digest as well, "
            "so the two would not even share a checkpoint identity. The hinge remains available on "
            "pocket_progress_coordinate's own lambda_m term"
        )
    if not isinstance(value, str) or value not in METRIC_OBJECTIVE_FORMS:
        raise ValueError(
            f"metric_objective must be {list(METRIC_OBJECTIVE_FORMS)}, received {value!r}. There "
            "is no fallback: a misspelled form would be recorded in the objective identity as a "
            "protocol nobody chose"
        )
    return value


def metric_mse(deviation_angstrom: Tensor, separation_angstrom: Tensor | float) -> Tensor:
    """Return ``L = 0.5 * (m / s_p)^2``, elementwise over a batch.

    ``deviation_angstrom`` is ``m``, one value per observation, in Angstrom;
    ``separation_angstrom`` is ``s_p``, that observation's own protein's deposited apo<->holo
    separation, in Angstrom and strictly positive. There is no ``tau`` argument here to be
    defaulted, which is the strongest available statement that this form has no threshold.

    No ``sqrt`` is differentiated: the square undoes exactly the one
    :func:`~sae_bioemu.steering.aligned_rmsd.aligned_rmsd` took, but it is written as a square of
    the already-measured deviation rather than folded into the kernel, so the value the journal
    carries in Angstrom and the value the loss descends are the same measurement.

    A SECOND IMPLEMENTATION of :func:`sae_bioemu.cryptic_steering.direct_metric.metric_mse` rather
    than a call into it -- see the module docstring -- and
    ``tests/test_cryptic_rmsd_direct_metric.py`` pins the two numerically so the duplication
    cannot drift into two objectives.
    """
    if not isinstance(deviation_angstrom, Tensor):
        raise TypeError("deviation_angstrom must be a torch.Tensor")
    separation = (
        separation_angstrom
        if isinstance(separation_angstrom, Tensor)
        else torch.full_like(deviation_angstrom, float(separation_angstrom))
    )
    separation = separation.to(deviation_angstrom)
    if separation.shape != deviation_angstrom.shape:
        raise ValueError(
            "separation_angstrom is PER OBSERVATION -- each protein's own apo<->holo separation "
            f"-- so it must share the deviation's shape; got {tuple(separation.shape)} and "
            f"{tuple(deviation_angstrom.shape)}"
        )
    if bool(torch.any(separation <= 0)):
        raise ValueError(
            "separation_angstrom must be strictly positive for every observation: it is the "
            "DIVISOR the objective is measured in, and a protein whose deposited conformers "
            "coincide over the metric region has no transition to steer along (see "
            f"MINIMUM_SEPARATION_ANGSTROM={MINIMUM_SEPARATION_ANGSTROM})"
        )
    scaled = deviation_angstrom / separation
    return 0.5 * scaled * scaled


def aligned_backbone_rmsd(draw: Tensor, reference: Tensor) -> Tensor:
    """Return the superposed N/CA/C/O RMS deviation of one draw against one deposited conformer.

    ``draw`` and ``reference`` are ``[K, 4, 3]`` in Angstrom, in :data:`CRYPTIC_METRIC_ATOMS`
    order, and both are ALREADY RESTRICTED to this system's metric region.

    **There is no ``metric_residues`` argument, and its absence is the point.** The
    ``pocket_progress_coordinate`` twin (``direct_metric.aligned_metric_deviation``) takes a
    selector and indexes both operands with it, which makes "the alignment set" and "the metric
    set" two uses of one variable a caller could pass differently. Here the reference artifact
    stores both conformers PRE-RESTRICTED against one ``metric_indices`` list, so the two sets are
    the same set structurally and there is no argument through which they could differ. The draw is
    restricted by the same list at the one call site that reads it
    (:func:`~sae_bioemu.cryptic_rmsd.artifact.pocket_rmsd_metric_batch`).

    **The region is chosen by the BENCHMARK and by the crystals, never by the draw.** A draw has
    every residue of its construct; a deposited structure does not, and apo and holo do not miss
    the same ones. The artifact's builder admits a residue only where all four atoms are modelled
    in BOTH crystals, which is what makes the two levels of one sequence and the ``s_p`` they are
    normalised by three numbers over ONE atom set.

    The superposition is solved on exactly the atoms the metric is read over --
    :func:`~sae_bioemu.steering.aligned_rmsd.aligned_rmsd` refuses any other arrangement by name --
    which is what makes the detached rotation exact by the envelope theorem and the gradient pure
    shape change.

    :data:`TRAINED_DEVIATION_FORM` is what comes back: the ROOT-MEAN-SQUARE, not the arithmetic
    mean the benchmark grades. See design point 5.
    """
    atoms = len(CRYPTIC_METRIC_ATOMS)
    if draw.ndim != 3 or draw.shape[-1] != 3 or draw.shape[-2] != atoms:
        raise ValueError(
            f"draw must be [metric_residues, {atoms}, 3] holding "
            f"{'/'.join(CRYPTIC_METRIC_ATOMS)} in Angstrom; got {tuple(draw.shape)}"
        )
    if reference.shape != draw.shape:
        raise ValueError(
            f"reference must share the draw's shape; got {tuple(reference.shape)} and "
            f"{tuple(draw.shape)}. The deviation is defined between corresponding atoms, and both "
            "operands are already restricted to this system's metric region"
        )
    if int(draw.shape[0]) < 1:
        raise ValueError(
            "no residue of the metric region is resolved in both deposited conformers, so this "
            "system has no metric set at all; it belongs out of the arm rather than in it with an "
            "empty measurement"
        )
    # Flattened to (4K, 3): the deviation is over ATOMS, so all four of a residue's atoms enter the
    # mean and the superposition individually, exactly as ``calc_metrics_local`` does. One residue
    # is therefore already four points, which determines a superposition -- so the floor here is a
    # RESIDUE and not the three ``kinase_rmsd`` requires, because on this pool the region is chosen
    # by the deposited pair rather than by a fixed motif window.
    return aligned_rmsd(draw.reshape(-1, 3), reference.reshape(-1, 3).to(draw))


@dataclass(frozen=True)
class PocketRmsdNormalisation:
    """One protein's own scale: ``s_p``, and the origin the shared target channel is built on.

    Built once per system from the two deposited conformers -- the RMS form and never the mean one,
    so that the numerator and the denominator of the objective are the same quantity; mixing the
    mean form in here would scale the ask by the very 4.5-34% ratio design point 5 measures -- and
    carried in the arm identity, so a report can say what weight each protein's draws carried
    rather than inferring it.

    There is deliberately no ``tau_angstrom`` field and no ``effective_tau_angstrom`` property. The
    ``kinase_rmsd`` and ``domain_motion`` twins carry both because their hinge ablations read them;
    this path has no hinge (design point 1), and a threshold stored where nothing reads it is a
    parameter that would reach a report without reaching the objective.
    """

    sequence_id: str
    #: ``s_p``: the superposed N/CA/C/O RMS deviation between the deposited apo and holo over this
    #: protein's metric region, in Angstrom.
    separation_angstrom: float

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id:
            raise ValueError("a pocket-RMSD normalisation must name its sequence")
        value = self.separation_angstrom
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("separation_angstrom must be a real number")
        object.__setattr__(self, "separation_angstrom", float(value))
        if not self.separation_angstrom > 0.0:
            raise ValueError(f"{self.sequence_id}: separation_angstrom must be positive")
        if self.separation_angstrom < MINIMUM_SEPARATION_ANGSTROM:
            raise ValueError(
                f"{self.sequence_id}: s_p = {self.separation_angstrom:.9f} Angstrom is below "
                f"MINIMUM_SEPARATION_ANGSTROM={MINIMUM_SEPARATION_ANGSTROM}. A '> 0' gate would "
                "pass it: aligned_rmsd carries eps=1e-12 inside its square root and reports about "
                "1e-6 Angstrom for two IDENTICAL structures. s_p DIVIDES this objective, so such a "
                "system enters the batch at m/1e-6 and owns every gradient in it"
            )

    def mse(self, deviation_angstrom: Tensor) -> Tensor:
        """Return ``L = 0.5*(m/s_p)^2`` for one or more measured deviations of THIS protein."""
        return metric_mse(deviation_angstrom, float(self.separation_angstrom))

    def priced(self, deviation_angstrom: Tensor) -> Tensor:
        """Return the priced value. One dispatcher, so the journal row and the quantity descended
        cannot come from different arithmetic -- the same role the ``kinase_rmsd`` twin's
        form-dispatching ``priced`` plays, with one form to dispatch to."""
        return self.mse(deviation_angstrom)

    def target_constant_angstrom(self) -> float:
        """Return the per-system TARGET the shared ``system_constant`` origin is built from: ``s_p``.

        The shared origin requires a STRICTLY POSITIVE per-system target
        (:class:`~sae_bioemu.steering.schedule.SystemTargetConstant`), and this objective's zero is
        at ``m = 0``. So the target is ``s_p``, the prediction is ``m + s_p``, and
        ``prediction - target`` is EXACTLY ``m``: a pure origin the prediction carries too and which
        therefore cancels.

        ``s_p`` rather than a benchmark ``tau`` is what keeps the evaluation threshold out of this
        checkpoint entirely -- out of the value, out of the gradient, and out of the digest of the
        per-system table the checkpoint is bound to.

        A shifted prediction is the price of not editing ``steering/schedule.py``, which is in
        ``generation_shared`` and whose blast radius is every steering checkpoint there has ever
        been, to admit a zero target. It is not hidden: ``PocketRmsdSteeringLoss.forward``
        re-derives ``prediction - target`` against the measured ``m`` on the wire and refuses a
        disagreement, elementwise, on every batch.
        """
        return float(self.separation_angstrom)

    def prediction(self, deviation_angstrom: Tensor) -> Tensor:
        """Return the shared term's ``prediction`` for THIS protein: ``m + s_p``."""
        if not isinstance(deviation_angstrom, Tensor):
            raise TypeError("deviation_angstrom must be a torch.Tensor")
        return deviation_angstrom + self.target_constant_angstrom()

    def identity(self) -> dict[str, float | str]:
        """The per-protein numbers that enter the arm identity."""
        return {
            "sequence_id": self.sequence_id,
            "separation_angstrom": round(float(self.separation_angstrom), 6),
        }


# The pool gate IS the divisor floor on this property, and it is strictly below the smaller of the
# two reported thresholds. Asserted at import rather than left as a comment, because the failure
# this ordering prevents is a pool gated AT OR ABOVE its own reported threshold -- which would mean
# every admitted system was already inside tau before the adapter ran, and the reported rate would
# be a statement about the pool rather than about the arm.
assert POOL_GATE_ANGSTROM == MINIMUM_SEPARATION_ANGSTROM < min(BENCHMARK_TAU_ANGSTROM)
