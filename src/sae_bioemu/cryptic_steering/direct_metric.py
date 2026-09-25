"""The DIRECT-METRIC cryptic objective: train on the deviation the benchmark grades.

SCOPE: CRYPTIC POCKETS. The arithmetic here is a function of coordinates, a deposited reference
conformer and a dose level, and nothing here constructs a loss --
:class:`~sae_bioemu.cryptic_steering.loss.CrypticSteeringLoss` composes the term, exactly as it
composes the D-28 off-path split.

Beside the arithmetic sits the one thing it cannot be computed without: the DEPOSITED CONFORMERS.
:class:`CrypticMetricReference` is the signed sidecar carrying each pool protein's apo and holo
backbone over its own metric region, and :func:`metric_deviations_for_batch` is the per-batch call
site that turns one generated batch into the two tensors ``forward`` consumes. Both live HERE and
not in ``arm.py`` for a fingerprint reason rather than a taste one: ``arm.py`` is in
``cryptic_generation_shared``, which the KINASE registration also composes, so an edit there re-keys
every kinase checkpoint for a term no kinase arm can configure. This file is
``cryptic_training_only``. The direct metric enters no evaluated quantity -- ``success@B`` is scored
by :mod:`~sae_bioemu.cryptic_steering.success_at_b` from written structures and never from this --
so training-only is both the cheapest and the truthful placement. ``arm.py`` still RESOLVES the
sidecar, because resolution is what it does and because the resulting digest belongs in the arm
identity; it reaches in here for the reader.

WHY A SECOND TERM RATHER THAN A REPLACEMENT
--------------------------------------------

``the progress coordinate`` is a VALIDATED but PARTIAL proxy for the reported metric. D-82 measured the within-protein
rank correlation between ``the progress coordinate`` and the deviation ``success@B`` grades: median Spearman **-0.574**
against the experimental holo reference, **-0.601** restricted to proteins BioEmu folds, with the
expected (negative) sign in 85% and 90% of proteins respectively. So raising ``the progress coordinate`` really does
move a draw towards holo on most proteins -- and ``|rho| ~ 0.57`` leaves roughly two thirds of the
within-protein rank variance unexplained. That residual is what a direct metric can reach and a
progress coordinate cannot, and it is why :data:`CRYPTIC_METRIC_WEIGHT_NAME` is ADDITIVE alongside
the ``the progress coordinate`` target term rather than a replacement for it.

``success_at_b.py`` states the hazard from the other side: ``the progress coordinate = 1`` is a hyperplane containing
holo and every structure at any off-axis distance, so training on ``the progress coordinate`` and scoring on ``the progress coordinate`` is
circular. This term closes that circle by descending the scored quantity itself.

**A MEASURED PREDICTION THAT THIS TERM MAKES THE ARM WORSE.** D-80 section 4 predicts exactly that.
The direct metric is the holo-ANCHORED objective: it redirects the majority of the gradient towards
cancelling a per-draw off-axis quantity of typical size 1.17 transition-lengths, against a measured
adapter reach of 0.073 -- a 16:1 mismatch -- and D-77/D-78 show that quantity's per-protein sign is
not recoverable. Nothing here is tuned to avoid that prediction. The term is built clean, shipped
at zero weight, and an arm decides.

THE OBJECTIVE
-------------

Per (sequence ``p``, dose level ``o``, draw)::

    X = the N/CA/C/O of the steered draw, over the benchmark's ``metric_resid_ranges``
    Y = the same atoms of the deposited conformer the LEVEL names (apo or holo)
    m = aligned_rmsd(X, Y)                        Angstrom; alignment set == metric set
    L = 0.5 * (m / s_p)^2                         PLAIN MSE on the per-system scale -- the default
      = huberised_hinge((m - tau_p) / s_p)        the ABLATION: zero below tau_p, then quadratic,
                                                  then linear

and the term the loss prices is ``lambda_m * mean over observations of L``.

FOUR DESIGN POINTS, EACH WITH ITS REASON
-----------------------------------------

1. **PLAIN MSE BY DEFAULT; THE HINGE IS A TWO-VALUE ABLATION.** :data:`DEFAULT_METRIC_OBJECTIVE` is
   :data:`METRIC_OBJECTIVE_MSE`, ``L = 0.5*(m/s_p)^2``, and :data:`METRIC_OBJECTIVE_HINGE` is the
   other member of :data:`METRIC_OBJECTIVE_FORMS`. Three reasons, and none of them is that the hinge
   is wrong:

   * Nothing in this programme has yet moved the metric at all, so a threshold inside the objective
     is sophistication bought before the thing it refines exists.
   * A threshold makes a NULL result harder to read. Under the hinge every draw already inside
     ``tau_p`` contributes exactly zero, so "the term did nothing" and "the term had nothing to do"
     produce the same loss curve; under the MSE every draw contributes and the curve separates them.
   * ``tau`` is an EVALUATION parameter. ``success@B`` is reported at BOTH values of
     :data:`BENCHMARK_TAU_ANGSTROM`, so a tau baked into the training objective would make the
     1.0 A and the 1.5 A readings of one run two readings of two differently-trained adapters.

   **This is a DEFAULT change, not a second implementation.** Plain MSE is exactly the special case
   ``tau = 0, transition -> infinity`` of :func:`huberised_hinge`: below the knee the hinge IS
   ``0.5*z^2``, and at ``tau = 0`` with ``m >= 0`` the clamp at zero never binds. Measured rather
   than asserted -- max absolute difference **0.0** against
   ``huberised_hinge(m/s_p, transition=large)`` on a six-point vector, and
   ``tests/test_cryptic_steering_direct_metric.py`` pins :func:`metric_mse` to the hinge on that
   identity so the two cannot drift into two objectives.

   What the hinge buys, and why it stays reachable: ``success@B`` has no value below ``tau`` and
   unbounded value above it, so an unbounded loss keeps paying gradient to polish draws that already
   succeed and lets one hopeless system own the batch. A HARD hinge has the mirror defect -- exactly
   zero gradient for the far draws, which are the work. The huberised hinge is flat where there is
   nothing to win, quadratic where a small push converts a draw, and LINEAR far out, which bounds
   the per-draw gradient magnitude at :data:`HINGE_TRANSITION` / ``s_p`` however hopeless the system
   is. That is an argument for an ablation, and it is shipped as one: ``objective.metric_objective``
   takes exactly the two spellings, the form rides the objective identity beside ``lambda_m``, and
   ``tau`` keeps having NO default on the hinge path.

2. **THE LEVEL SELECTS WHICH REFERENCE, NOT A SIGN.** A distance is unsigned and ``success@B``
   scores apo and holo SEPARATELY (``success_at_b.REFERENCE_STATES``), so ``+1`` means "minimise
   the deviation to the deposited holo" and ``-1`` "to the deposited apo", at the SHARED
   ``draw_seed`` the two levels of one (sequence, cycle) already carry. Nothing in the shared
   schedule changes: :data:`REFERENCE_STATES` is an INTERPRETATION of the two existing levels, not
   a new dose set.

3. **PER-SYSTEM NORMALISATION.** ``s_p`` is that system's own apo<->holo reference separation
   (:class:`~sae_bioemu.cryptic_steering.success_at_b.ReferenceSeparation`) and
   ``tau_p = min(tau, 0.5 * s_p)``. ``success@B`` is an unweighted mean over SYSTEMS of a bounded
   indicator; a per-draw Angstrom loss is a mean over draws of an unbounded quantity, so without a
   per-system scale the objective's implicit weighting is "largest transitions first" and the
   metric's is "every system once". ``0.5 * s_p`` is ConforMix's ``adaptive_tau``, already
   implemented and already reported beside the fixed thresholds; the ``min`` takes whichever is
   HARDER, so the objective is never satisfied by a draw the reported metric would fail.

   **``s_p`` survives the move to plain MSE and ``tau_p`` does not**, which is the whole of the
   difference between the two forms. The SCALE is what makes the batch mean an average over systems
   rather than one dominated by the largest transitions, and that argument nowhere mentions a
   threshold. ``tau_p`` IS the threshold, and it is exactly the evaluation parameter design point 1
   takes back out of the training objective. So :func:`metric_mse` divides by ``s_p`` and subtracts
   nothing, and dropping the scale instead would not be a simpler objective -- it would be one whose
   implicit weighting is not the metric's.

   **Stated rather than buried: this is a per-protein number inside the objective**, of the kind
   D-46 removed from the ``the progress coordinate`` target. It is not the same defect -- D-46 added a per-protein
   CONSTANT to a displacement target, an intercept no dose slope could see, whereas this is a
   per-protein SCALE on a quantity whose zero is meaningful -- but two systems' hinge values are
   not in one unit, and a report must say which ``s_p`` each protein carried.

4. **NO SOFT-MIN OVER DRAWS.** ``success@B`` takes a minimum over ``B`` draws. With two levels at
   one draw per scheduled unit there is no min to take inside a unit, and a soft-min over two
   values is nearly a hard min: it would route the whole gradient to one level and silently halve
   the schedule. Under the hinge ablation the threshold encodes most of what the min buys anyway,
   because a draw inside ``tau_p`` contributes nothing and stops competing with its siblings; under
   the default MSE nothing encodes it, and nothing is meant to -- ``B`` is a property of the
   REPORTED metric, and the training objective descends every draw it is handed.

THE ONE FORM CHANGE, AND IT IS LOAD-BEARING
--------------------------------------------

**The reported metric is the arithmetic MEAN of per-atom deviations and this trains on the RMS.**

``bioemu-benchmarks``' ``eval/multiconf/evaluate.py::calc_metrics_local`` sets
``metrics[MetricType.RMSD] = np.mean(pairwise_distances, axis=-1)``: the arithmetic mean of the
per-atom Euclidean deviations after superposition, misnamed ``rmsd`` upstream.
:mod:`~sae_bioemu.cryptic_steering.success_at_b` already refuses to call it an RMSD and emits both
forms side by side (:data:`MEAN_DEVIATION` and :data:`RMSD`). This module trains on the RMS one,
and the reason is not preference:

**Kabsch minimises the RMS, not the mean.** ``R*`` is the stationary point of
``sum_i w_i ||R x_i + t - y_i||^2``, so the envelope (Danskin) argument in
:mod:`sae_bioemu.steering.aligned_rmsd` -- which is what makes the DETACHED rotation the exact
gradient rather than a truncation -- applies to the RMS form and to nothing else. ``R*`` is NOT
stationary for ``mean_i ||R x_i + t - y_i||``, so a mean-form objective differentiated with the
rotation detached would be differentiating a quantity whose minimiser it did not solve for, and the
true derivative would carry a term through ``R*(X)``.

**What the substitution costs, and it is a real cost.** By Jensen ``rmsd >= mean_deviation``
pointwise, so any draw passing ``tau`` under the RMS passes under the mean: the ask is SUFFICIENT
for the graded one, just TIGHTER. Measured on this project's own real geometry rather than
asserted -- ``d81_native_success.csv``, the native BioEmu arm over 28 benchmark systems x 2
reference states, 168 (system, state, statistic) comparisons -- the inequality holds in 168 of 168
and the ratio ``rmsd / mean_deviation`` runs:

    crystal vs crystal (reference separation)   min 1.045  p50 1.127  p90 1.230  max 1.255
    draw vs deposited  (median of 100 draws)    min 1.061  p50 1.137  p90 1.264  max 1.329
    draw vs deposited  (best of 100 draws)      min 1.067  p50 1.150  p90 1.304  max 1.339

The full measured span of that table is therefore **1.045 to 1.339**: training on the RMS asks for a
**4.5-34% tighter** deviation than the benchmark grades, and **about 13% at the median** (1.127,
1.137 and 1.150 across the three comparisons). Quote those, and not a wider band: D-63's own reading
of the same relationship put the draw-vs-deposited ratio at 1.20-1.49 and a brief written from it
said "13-49%", but **1.49 appears nowhere in the table above** and 1.339 is its maximum. The
direction is the same either way; the size is the one measured here. That gap is the price of an exact gradient and
it is paid knowingly. It is NOT hidden inside a tau: :data:`TRAINED_DEVIATION_FORM` and
:data:`GRADED_DEVIATION_FORM` are separate named constants, :func:`mean_deviation` computes the
graded form so the two can be compared on any real batch, and
``tests/test_cryptic_steering_direct_metric.py`` checks the inequality on geometry rather than
assuming it.

The alternative -- training on the mean with the rotation taken through autograd -- is not
obviously wrong and is not implemented here, because it costs the envelope theorem, puts gradient
on the rotation's own conditioning, and blows up on the coincident-singular-value case
:func:`~sae_bioemu.steering.aligned_rmsd.rotation_degeneracy` reports. A run that wants it is
asking for a different estimator and must argue for it.

WHY THE ALIGNMENT SET AND THE METRIC SET ARE ONE SET HERE
----------------------------------------------------------

:func:`~sae_bioemu.steering.aligned_rmsd.aligned_rmsd` REFUSES a distinct alignment set, and that
refusal is what the exactness argument rests on. It is legal to obey here because of a fact about
the ASSETS rather than about the schema: the upstream cryptic scorer dispatches on
``localresidinfo`` and uses ``calc_metrics_local``, whose ``alignment_resid_ranges`` and
``metric_resid_ranges`` are separate lists that the schema PERMITS to differ -- and in all **33 of
33** shipped ``local_residinfo`` files they are byte-identical. Verified against
``d19/raw/local_residinfo/*.json`` rather than assumed; :func:`require_one_region` re-checks it per
system at call time, because a future asset that exercised the difference would silently turn the
detached rotation from exact into wrong. See
:func:`~sae_bioemu.steering.aligned_rmsd._require_one_set` for the measured size of that leak.

THE GEOMETRY TERMS BECOME LOAD-BEARING HERE, AND MUST NOT BE DROPPED
---------------------------------------------------------------------

Under a direct-metric objective the residues OUTSIDE ``metric_resid_ranges`` receive **exactly
zero** gradient from this term. Measured on a real system rather than estimated: on ``O74933``,
whose shipped ranges are ``[[105, 120], [280, 330]]`` and which resolves 67 of those positions on a
486-residue BioEmu draw, the maximum absolute gradient outside the metric region is **exactly 0.0**
while inside it is 7.65e-3, so **419 of 486 residues are entirely unconstrained by this term**.
(A "119 of 214" figure quoted elsewhere in this project is a different system and was not
re-measured here; what carries the argument is the exact zero, which is a property of the metric
set and not of one protein.) "Satisfy the metric by deforming the rest of
the protein" is therefore wholly unpriced by ``lambda_m``, and the only things standing against it
are the shared ``lambda_g`` C-alpha geometry penalty and ``lambda_l`` smooth LDDT against the
unsteered draw. D-79 section 7.3 measured ``lambda_g`` at **0.04% of the loss budget** and
``lambda_l`` at **0** on the shipped arms -- i.e. both are decorative TODAY, under an objective
that touches every residue through ``the progress coordinate``. They stop being decorative the moment ``lambda_m`` is
nonzero. An arm that prices this term and zeroes those is not a cheaper arm; it is an arm with no
statement at all about the majority of its residues.

The same measurement reports the other half of the claim: ``direct_metric_leak`` on that draw gives
``energy_fraction = 4.9e-29``, ``net_force = 1.1e-14``, ``net_torque = 1.4e-14`` -- the arithmetic
floor, against ~0.99 for an unaligned coordinate MSE between two differently-framed structures.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import torch
from torch import Tensor

from sae_bioemu.steering.aligned_rmsd import RigidBodyLeak, aligned_rmsd, rigid_body_leak

__all__ = [
    "APO",
    "BENCHMARK_TAU_ANGSTROM",
    "BIOEMU_TAU_ANGSTROM",
    "CONFORNETS_TAU_ANGSTROM",
    "CRYPTIC_METRIC_ATOMS",
    "CRYPTIC_METRIC_WEIGHT_NAME",
    "DEFAULT_METRIC_OBJECTIVE",
    "DEVIATION_FORMS",
    "GRADED_DEVIATION_FORM",
    "HINGE_TRANSITION",
    "HOLO",
    "MEAN_DEVIATION",
    "METRIC_OBJECTIVE_FORMS",
    "METRIC_OBJECTIVE_HINGE",
    "METRIC_OBJECTIVE_MSE",
    "METRIC_REFERENCE_FILENAME",
    "METRIC_REFERENCE_SCHEMA_VERSION",
    "METRIC_SEPARATION_TOLERANCE_ANGSTROM",
    "REFERENCE_STATES",
    "RMSD",
    "TRAINED_DEVIATION_FORM",
    "CrypticMetricReference",
    "MetricNormalisation",
    "MetricReferenceEntry",
    "aligned_metric_deviation",
    "checked_metric_entry",
    "direct_metric_hinge_loss",
    "direct_metric_leak",
    "huberised_hinge",
    "load_cryptic_metric_reference",
    "mean_deviation",
    "metric_backbone_atoms",
    "metric_deviations_for_batch",
    "metric_hinge",
    "metric_mse",
    "reference_state_for_level",
    "require_one_region",
    "validated_metric_objective",
    "write_cryptic_metric_reference",
]

#: The runtime/arm-config spelling of this term's weight. One constant, imported by the loss, by
#: the arm gate and by every test, so the weight cannot be recorded under one name and applied
#: under another -- the ``lambda_l`` shape that voided six arms.
CRYPTIC_METRIC_WEIGHT_NAME = "lambda_m"

#: The atoms the benchmark measures over. ``bioemu-benchmarks`` filters both reference and sample
#: to mdtraj's ``backbone`` selection, which is exactly these four, in this per-residue order.
#: Duplicated from :data:`sae_bioemu.cryptic_steering.success_at_b.BACKBONE_ATOMS` rather than
#: imported, and the duplication is deliberate: ``success_at_b`` is not a member of any fingerprint
#: group, so importing a constant from it would let the trained objective change at an unmoved
#: training digest. ``tests/test_cryptic_steering_direct_metric.py`` pins the two equal.
CRYPTIC_METRIC_ATOMS: tuple[str, ...] = ("N", "CA", "C", "O")

#: Where each of :data:`CRYPTIC_METRIC_ATOMS` sits in OpenFold's atom37 ordering, which is
#: ``[N, CA, C, CB, O, ...]``. **O is index 4 and NOT index 3**: index 3 is the C-beta, and reading
#: it as the carbonyl would put a real atom in the O column and pass every finiteness and
#: bond-length check there is.
_ATOM37_METRIC_INDEX: Mapping[str, int] = {"N": 0, "CA": 1, "C": 2, "O": 4}

#: The arithmetic mean of per-atom deviations: what ``calc_metrics_local`` labels ``rmsd`` and what
#: the reported ``success@B`` actually grades. NOT what this objective descends -- see the module
#: docstring's form-change section.
MEAN_DEVIATION = "mean_deviation"
#: The true root-mean-square of the same per-atom deviations. Always ``>=`` :data:`MEAN_DEVIATION`
#: by Jensen, and the only one of the two for which the Kabsch rotation is stationary.
RMSD = "rmsd"
#: Both forms, in the same order ``success_at_b`` declares them.
DEVIATION_FORMS: tuple[str, ...] = (MEAN_DEVIATION, RMSD)

#: What this objective DESCENDS. An explicit named choice, not an implementation detail.
TRAINED_DEVIATION_FORM = RMSD
#: What the benchmark GRADES. Named beside the trained form so no reader can take one for the
#: other, and so a report can quote the ratio between them.
GRADED_DEVIATION_FORM = MEAN_DEVIATION

#: ConforNets' cryptic-pocket cutoff, in Angstrom.
CONFORNETS_TAU_ANGSTROM = 1.0
#: BioEmu's own cutoff, in Angstrom.
BIOEMU_TAU_ANGSTROM = 1.5
#: The two thresholds of record. There is deliberately NO default tau anywhere in this module:
#: D-63 measured 30.5/60.6 at ``tau = 1.0, B = 100`` against 49/85 at ``tau = 1.5, B = 4000`` on
#: the same model and the same 33 systems, so a success rate quoted without tau is meaningless and
#: a tau that could be picked up by omission is worse than an error.
BENCHMARK_TAU_ANGSTROM: tuple[float, ...] = (CONFORNETS_TAU_ANGSTROM, BIOEMU_TAU_ANGSTROM)

#: Plain MSE on the per-system scale, ``L = 0.5*(m/s_p)^2``. The DEFAULT, and the form every
#: direct-metric arm gets unless it names the other one.
METRIC_OBJECTIVE_MSE = "mse"
#: The one-sided huberised hinge on ``(m - tau_p)/s_p``. The ABLATION, and the only form that takes
#: a ``tau``.
METRIC_OBJECTIVE_HINGE = "huberised_hinge"
#: The two forms, and there are exactly two. A third would be a third objective and would need its
#: own argument; an unrecognised spelling is refused by name rather than falling back to either, so
#: a typo cannot silently retrain an arm under the other form.
METRIC_OBJECTIVE_FORMS: tuple[str, ...] = (METRIC_OBJECTIVE_MSE, METRIC_OBJECTIVE_HINGE)
#: What an arm that does not name a form descends. See design point 1: a threshold in the objective
#: is premature here, it makes a null result unreadable, and ``tau`` is an EVALUATION parameter that
#: must not differ between the 1.0 A and the 1.5 A readings of one run.
DEFAULT_METRIC_OBJECTIVE = METRIC_OBJECTIVE_MSE

#: Where the hinge turns from quadratic to linear, in units of the system's own separation ``s_p``.
#: At 1.0 the quadratic region is exactly one reference separation wide, so a draw that has to move
#: further than the whole apo->holo transition is in the linear regime and contributes a BOUNDED
#: gradient. This is the parameter that decides how much of a batch one hopeless system can own, so
#: it is part of the objective's identity rather than a knob hidden in a config.
HINGE_TRANSITION = 1.0

#: The two reference states, spelled as ``success_at_b`` spells them.
APO = "apo"
HOLO = "holo"

#: Which deposited conformer each two-state dose level NAMES. ``+1`` asks for holo (the ligand-bound
#: open-pocket crystal) and ``-1`` for apo. This is a reading of the existing
#: :data:`sae_bioemu.steering.schedule.TWO_STATE_TARGET_LEVELS`, not a new dose set, and it is a
#: REFERENCE selection rather than a sign: a deviation is unsigned, ``success@B`` scores the two
#: states separately and never pools them, and the per-protein sign-identifiability problem D-77
#: and D-78 record therefore does not arise for this term.
REFERENCE_STATES: Mapping[float, str] = {1.0: HOLO, -1.0: APO}


def reference_state_for_level(level: float) -> str:
    """Return ``"apo"`` or ``"holo"``: which deposited conformer this level asks for.

    ``0.0`` is refused rather than mapped to a midpoint, for the reason the two-state dose set
    removed it: there is no third deposited structure, and "half way between the two crystals" is
    neither a conformer the protein has nor a thing ``success@B`` can score.
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


def require_one_region(
    system: str,
    alignment_resid_ranges: Sequence[Sequence[int | None]],
    metric_resid_ranges: Sequence[Sequence[int | None]],
) -> None:
    """Fail closed unless this system superposes on exactly the region it measures over.

    The schema PERMITS the two range lists to differ and the shipped assets never do: all 33 of 33
    ``local_residinfo`` files carry byte-identical lists. This objective's exactness rests on that
    identity -- the envelope theorem needs ``R*`` to minimise the same functional that is being
    differentiated -- so it is re-checked per system rather than assumed from an inventory taken
    once. A future asset that exercised the difference would not raise anywhere else: it would
    silently turn the detached rotation from exact into wrong, and put gradient on atoms carrying
    no metric term at all.

    :func:`~sae_bioemu.steering.aligned_rmsd.aligned_rmsd` refuses the same thing one layer down,
    on the weights. This refuses it on the RANGES, which is where a caller can still read the
    message and choose a different estimator.
    """
    left = tuple(tuple(None if value is None else int(value) for value in entry) for entry in alignment_resid_ranges)  # fmt: skip
    right = tuple(tuple(None if value is None else int(value) for value in entry) for entry in metric_resid_ranges)  # fmt: skip
    if left != right:
        raise ValueError(
            f"{system}: alignment_resid_ranges {list(left)} and metric_resid_ranges {list(right)} "
            "differ, so a direct-metric objective over this system would superpose on one region "
            "and measure another. The detached Kabsch rotation is exact only when the two sets are "
            "one set (envelope theorem); with two sets the gradient acquires a term through R*(X) "
            "and lands on atoms the objective never scores. All 33 shipped local_residinfo files "
            "carry identical lists -- this system is the first that does not, and it needs a "
            "different estimator rather than a looser check"
        )


def huberised_hinge(value: Tensor, *, transition: float = HINGE_TRANSITION) -> Tensor:
    """Return the one-sided Huber of ``value``: 0 at or below zero, then quadratic, then linear.

        L(z) = 0                                    z <= 0
             = 0.5 * z^2                            0 < z <= transition
             = transition * (z - 0.5 * transition)  z > transition

    Continuous in value and in first derivative at both breakpoints, which is what makes it usable
    as a loss: the gradient rises linearly from 0 at ``z = 0`` to ``transition`` and is CONSTANT
    past it. The flat half stops the objective polishing draws that already succeed; the linear
    half stops one far draw owning the batch.

    Written with ``clamp`` rather than ``where`` so the zero region's gradient is exactly zero
    rather than a masked NaN, and so no ``sqrt`` is differentiated at the origin.

    Deliberately the SAME shape, the same default transition and the same spelling as
    ``domain_motion.objective.huberised_hinge``. The two are separate implementations because they
    sit in different fingerprint groups -- one file shared between them would make a domain-motion
    edit retire every cryptic checkpoint, which is the defect the property registry exists to end
    -- and ``tests/test_cryptic_steering_direct_metric.py`` pins them to each other numerically so
    the duplication cannot drift into two objectives.
    """
    if not isinstance(value, Tensor):
        raise TypeError("value must be a torch.Tensor")
    if isinstance(transition, bool) or not isinstance(transition, Real) or not transition > 0.0:
        raise ValueError("transition must be a positive real number")
    limit = float(transition)
    positive = torch.clamp(value, min=0.0)
    quadratic = torch.clamp(positive, max=limit)
    # ``positive - quadratic`` is the part beyond the knee and is exactly zero inside it, so the
    # two branches are one expression and no branch is selected by a comparison on a tensor.
    return 0.5 * quadratic * quadratic + limit * (positive - quadratic)


def aligned_metric_deviation(draw: Tensor, reference: Tensor, metric_residues: Tensor) -> Tensor:
    """Return the superposed N/CA/C/O RMS deviation of one draw against one deposited conformer.

    ``draw`` and ``reference`` are ``[residues, 4, 3]`` in Angstrom, in
    :data:`CRYPTIC_METRIC_ATOMS` order; ``metric_residues`` is a 1-D index or boolean selector
    naming the benchmark's ``metric_resid_ranges`` positions, already resolved into this draw's own
    residue frame by the caller.

    **The metric set is chosen by the BENCHMARK and by the crystals, never by the draw.** A draw has
    every residue of its construct; a deposited structure does not, and apo and holo do not miss the
    same ones. The caller must hand the residues resolved in BOTH conformers, which is what makes
    the two levels of one sequence and the reference separation ``s_p`` they are normalised by three
    numbers over ONE atom set. Measuring each level over its own reference's resolved set would make
    ``-1`` and ``+1`` incomparable and the normalisation meaningless.

    :data:`TRAINED_DEVIATION_FORM` is what comes back: the ROOT-MEAN-SQUARE, not the arithmetic mean
    the benchmark grades. See the module docstring -- the substitution is what makes the detached
    rotation exact, it is sufficient for the graded threshold by Jensen, and it is 4.5-34% tighter.

    The superposition is solved on exactly the atoms the metric is read over: ``aligned_rmsd``
    refuses any other arrangement by name, which is what makes the detached rotation exact by the
    envelope theorem and the gradient pure shape change. Report :func:`direct_metric_leak` beside
    the loss.
    """
    atoms = len(CRYPTIC_METRIC_ATOMS)
    if draw.ndim != 3 or draw.shape[-1] != 3 or draw.shape[-2] != atoms:
        raise ValueError(
            f"draw must be [residues, {atoms}, 3] holding {'/'.join(CRYPTIC_METRIC_ATOMS)} in "
            f"Angstrom; got {tuple(draw.shape)}"
        )
    if reference.shape != draw.shape:
        raise ValueError(
            f"reference must share the draw's shape; got {tuple(reference.shape)} and "
            f"{tuple(draw.shape)}. A deviation is defined between corresponding atoms"
        )
    selected_draw = draw[metric_residues]
    selected_reference = reference[metric_residues]
    if int(selected_draw.shape[0]) < 1:
        raise ValueError(
            "no residue of the metric region is resolved in both deposited conformers, so this "
            "system has no metric set at all; it belongs out of the arm rather than in it with an "
            "empty measurement"
        )
    # Flattened to (4K, 3): the metric is a deviation over ATOMS, so all four of a residue's atoms
    # enter the mean and the superposition individually, exactly as calc_metrics_local does.
    return aligned_rmsd(
        selected_draw.reshape(-1, 3), selected_reference.reshape(-1, 3).to(selected_draw)
    )


def mean_deviation(draw: Tensor, reference: Tensor, metric_residues: Tensor) -> Tensor:
    """Return :data:`GRADED_DEVIATION_FORM`: the ARITHMETIC MEAN the benchmark actually reports.

    **This is not the training objective and must never be wired into one.** It exists so that the
    cost of :data:`TRAINED_DEVIATION_FORM` can be MEASURED on whatever geometry a run is looking at
    rather than quoted from a docstring, and so a journal can carry the graded number beside the
    descended one.

    The superposition is the SAME one -- the RMS-optimal Kabsch rotation, detached -- because that
    is what the upstream scorer uses too: ``calc_metrics_local`` superposes least-squares and then
    takes ``np.mean`` of the per-atom distances. So this reproduces the graded quantity exactly.
    What it does NOT come with is an exactness guarantee for its gradient: ``R*`` is stationary for
    the RMS and not for the mean, so differentiating this with the rotation detached would be
    differentiating a functional whose minimiser was solved for something else. Use it to report,
    never to descend.
    """
    from sae_bioemu.steering.aligned_rmsd import kabsch_rotation

    atoms = len(CRYPTIC_METRIC_ATOMS)
    if draw.ndim != 3 or draw.shape[-1] != 3 or draw.shape[-2] != atoms:
        raise ValueError(f"draw must be [residues, {atoms}, 3]; got {tuple(draw.shape)}")
    if reference.shape != draw.shape:
        raise ValueError("reference must share the draw's shape")
    left = draw[metric_residues].reshape(-1, 3)
    right = reference[metric_residues].reshape(-1, 3).to(left)
    if int(left.shape[0]) < 1:
        raise ValueError("the metric region selected no atom")
    rotation, translation = kabsch_rotation(left, right)
    placed = left @ rotation.transpose(-2, -1) + translation
    return torch.linalg.vector_norm(placed - right, dim=-1).mean()


def validated_metric_objective(value: object) -> str:
    """Return one of :data:`METRIC_OBJECTIVE_FORMS`, or fail closed naming both.

    One validator, imported by the arm loader and by the loss, so the form cannot be accepted under
    one spelling in a config and dispatched under another in the objective -- the ``lambda_l`` shape
    applied to a string. ``None`` resolves to :data:`DEFAULT_METRIC_OBJECTIVE`, which is what an arm
    that names no form means and what every arm written before the key existed means.
    """
    if value is None:
        return DEFAULT_METRIC_OBJECTIVE
    if not isinstance(value, str) or value not in METRIC_OBJECTIVE_FORMS:
        raise ValueError(
            f"metric_objective must be one of {list(METRIC_OBJECTIVE_FORMS)}, received {value!r}. "
            f"{METRIC_OBJECTIVE_MSE!r} is 0.5*(m/s_p)^2 and is the default; "
            f"{METRIC_OBJECTIVE_HINGE!r} is the thresholded ablation and is the only form that "
            "takes a tau. There is no fallback: a misspelled form would silently train the other "
            "objective under a recorded identity that named neither"
        )
    return value


def _validated_deviation_pair(
    deviation_angstrom: Tensor, separation_angstrom: Tensor
) -> tuple[Tensor, Tensor]:
    """Return ``(m, s_p)`` as one dtype/device pair, refusing every shape a form could not mean.

    Shared by :func:`metric_hinge` and :func:`metric_mse` so the two forms cannot come to disagree
    about what a valid batch is: the difference between them is the arithmetic and nothing else.
    """
    if not isinstance(deviation_angstrom, Tensor) or not isinstance(separation_angstrom, Tensor):
        raise TypeError("deviation_angstrom and separation_angstrom must be torch.Tensors")
    if deviation_angstrom.shape != separation_angstrom.shape:
        raise ValueError(
            "separation_angstrom is PER OBSERVATION -- each protein's own apo<->holo separation -- "
            f"so it must share the deviation's shape; got {tuple(separation_angstrom.shape)} and "
            f"{tuple(deviation_angstrom.shape)}"
        )
    separation = separation_angstrom.to(deviation_angstrom)
    if bool(torch.any(separation <= 0)):
        raise ValueError(
            "separation_angstrom must be strictly positive for every observation: it is the "
            "DIVISOR the objective is measured in, and a protein whose deposited conformers "
            "coincide has no transition to steer along and belongs out of the arm"
        )
    return deviation_angstrom, separation


def metric_mse(deviation_angstrom: Tensor, separation_angstrom: Tensor) -> Tensor:
    """Return :data:`METRIC_OBJECTIVE_MSE`: ``L = 0.5 * (m / s_p)^2``, elementwise over a batch.

    **The DEFAULT form, and deliberately the special case of the hinge rather than a second
    implementation of anything.** ``huberised_hinge(z)`` is ``0.5*z^2`` for ``0 < z <= transition``
    and ``0`` at ``z <= 0``; with ``tau = 0`` the argument is ``m/s_p``, which is nonnegative because
    a deviation is, and with the knee taken to infinity the quadratic branch is the whole function.
    So this IS ``huberised_hinge((m - 0)/s_p, transition -> inf)``, and
    ``tests/test_cryptic_steering_direct_metric.py`` pins it numerically to that limit.

    ``separation_angstrom`` is kept and ``tau`` is gone, which is the entire difference from
    :func:`metric_hinge`. The scale is what makes the batch mean an average over SYSTEMS -- the
    weighting ``success@B`` uses -- and the threshold is the evaluation parameter design point 1
    takes out of the training objective. There is no ``tau`` argument here to be defaulted, which is
    the strongest available statement that this form has no threshold.

    No ``sqrt`` is differentiated: the square undoes exactly the one
    :func:`~sae_bioemu.steering.aligned_rmsd.aligned_rmsd` took, but it is written as a square of
    the already-measured deviation rather than folded into the kernel, so the value the journal
    carries in Angstrom and the value the loss descends are the same measurement.
    """
    deviation, separation = _validated_deviation_pair(deviation_angstrom, separation_angstrom)
    scaled = deviation / separation
    return 0.5 * scaled * scaled


def metric_hinge(
    deviation_angstrom: Tensor,
    separation_angstrom: Tensor,
    *,
    tau_angstrom: float,
    transition: float = HINGE_TRANSITION,
) -> Tensor:
    """Return ``L = huberised_hinge((m - tau_p) / s_p)``, elementwise over a batch.

    ``deviation_angstrom`` is ``m``, one value per observation, in Angstrom.
    ``separation_angstrom`` is ``s_p``, that observation's own protein's apo<->holo reference
    separation, in Angstrom and strictly positive. ``tau_angstrom`` is the GLOBAL benchmark
    threshold and has no default anywhere in this module (see :data:`BENCHMARK_TAU_ANGSTROM`).

    ``tau_p = min(tau, 0.5 * s_p)`` is formed HERE rather than handed in pre-resolved, for the
    reason :func:`~sae_bioemu.cryptic_steering.objective.context_rigidity_penalty` takes the RMS
    and not the finished hinge: the threshold is part of what the term MEANS, and an objective that
    accepted a pre-hinged number could not be said to have a threshold at all. It is what makes
    ``tau`` recordable in the objective identity and ``s_p`` a per-observation measurement.

    The clamp is one-sided on the harder side: ``0.5 * s_p`` is ConforMix's adaptive threshold and
    the fixed ``tau`` is the benchmark's, and taking the minimum means the term is never satisfied
    by a draw the reported metric would fail, while still not asking a 15 Angstrom transition to be
    reproduced to 7.5 Angstrom and called done.
    """
    if isinstance(tau_angstrom, bool) or not isinstance(tau_angstrom, Real) or tau_angstrom <= 0.0:
        raise ValueError(
            "tau_angstrom is the benchmark success threshold in Angstrom and must be positive. "
            f"There is no default: {BENCHMARK_TAU_ANGSTROM} are the two values of record and they "
            "give materially different success rates on the same model"
        )
    deviation_angstrom, separation = _validated_deviation_pair(
        deviation_angstrom, separation_angstrom
    )
    effective_tau = torch.clamp(0.5 * separation, max=float(tau_angstrom))
    return huberised_hinge((deviation_angstrom - effective_tau) / separation, transition=transition)


@dataclass(frozen=True)
class MetricNormalisation:
    """One protein's own scale and threshold: ``s_p`` and ``tau_p``. Design point 3.

    Built once per system from the two deposited conformers -- exactly
    :attr:`~sae_bioemu.cryptic_steering.success_at_b.ReferenceSeparation.rmsd`, the RMS form and
    never the mean one -- and carried in the arm identity, so a report can say what weight each
    protein's draws carried rather than inferring it.
    """

    sequence_id: str
    #: ``s_p``: the superposed N/CA/C/O RMS deviation between the deposited apo and holo over the
    #: benchmark's metric region, in Angstrom. :data:`TRAINED_DEVIATION_FORM`, so that the numerator
    #: and the denominator of the hinge are the same quantity; mixing the mean form in here would
    #: scale the ask by the very 4.5-34% ratio the module docstring measures.
    separation_angstrom: float
    #: The fixed benchmark threshold this protein was gated and normalised against. Required.
    tau_angstrom: float

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id:
            raise ValueError("a metric normalisation must name its sequence")
        for name in ("separation_angstrom", "tau_angstrom"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a real number")
            object.__setattr__(self, name, float(value))
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{self.sequence_id}: {name} must be positive")

    @property
    def adaptive_tau_angstrom(self) -> float:
        """``0.5 * s_p``: ConforMix's per-system threshold, in Angstrom."""
        return 0.5 * self.separation_angstrom

    @property
    def effective_tau_angstrom(self) -> float:
        """``tau_p = min(tau, 0.5 * s_p)``: whichever of the two thresholds is HARDER."""
        return min(float(self.tau_angstrom), self.adaptive_tau_angstrom)

    def hinge(self, deviation_angstrom: Tensor) -> Tensor:
        """Return ``L`` for one or more measured deviations of THIS protein.

        Routed through :func:`metric_hinge` rather than restating the arithmetic, so the per-system
        record and the batched loss path cannot come to mean two different things.
        """
        if not isinstance(deviation_angstrom, Tensor):
            raise TypeError("deviation_angstrom must be a torch.Tensor")
        separation = torch.full_like(
            deviation_angstrom, self.separation_angstrom, dtype=deviation_angstrom.dtype
        )
        return metric_hinge(deviation_angstrom, separation, tau_angstrom=float(self.tau_angstrom))

    def identity(self) -> dict[str, float | str]:
        """The per-protein numbers that enter the arm identity."""
        return {
            "sequence_id": self.sequence_id,
            "separation_angstrom": round(float(self.separation_angstrom), 6),
            "effective_tau_angstrom": round(float(self.effective_tau_angstrom), 6),
        }


def direct_metric_hinge_loss(
    draws: Sequence[Tensor],
    references: Sequence[Tensor],
    metric_residues: Sequence[Tensor],
    normalisations: Sequence[MetricNormalisation],
) -> tuple[Tensor, Tensor]:
    """Return ``(deviation_angstrom, hinge)``, one entry per observation, in batch order.

    Every argument is a per-observation sequence rather than a padded tensor, because the metric
    set differs per protein: two members of one batch do not have the same residues resolved in
    both of their own crystals, and a padded tensor would need a per-member mask saying exactly
    this and nothing more.

    Both tensors come back. ``deviation_angstrom`` is in the benchmark's own unit and belongs in
    the journal; ``hinge`` is what the loss prices. Journalling only the hinge would make an arm's
    progress unreadable, because a hinge of zero means "inside tau_p" and says nothing about
    whether the draw is at 2.9 Angstrom or at 0.4.
    """
    if not (len(draws) == len(references) == len(metric_residues) == len(normalisations)):
        raise ValueError(
            "draws, references, metric_residues and normalisations must be the same length: one "
            "entry per observation"
        )
    if not draws:
        raise ValueError("the direct metric needs at least one observation")
    deviations = [
        aligned_metric_deviation(draw, reference, residues)
        for draw, reference, residues in zip(draws, references, metric_residues, strict=True)
    ]
    stacked = torch.stack([value.reshape(()) for value in deviations])
    hinged = torch.stack(
        [
            normalisation.hinge(value.reshape(()))
            for normalisation, value in zip(normalisations, deviations, strict=True)
        ]
    )
    return stacked, hinged


def direct_metric_leak(deviation_angstrom: Tensor, draw: Tensor) -> RigidBodyLeak:
    """Return the rigid-body content of this term's own gradient at ``draw``.

    The number a direct-metric arm must report beside its loss, and the instrument that says the
    aligned form is doing what the module docstring claims: an UNALIGNED coordinate MSE against a
    reference puts ~99% of its gradient energy in the six rigid-body directions, and this should
    read at the arithmetic floor instead.

    ``draw`` must be the tensor ``deviation_angstrom`` was differentiated from and must require
    grad; the gradient is taken with ``retain_graph`` so a caller can still backward the loss
    afterwards.
    """
    if not isinstance(deviation_angstrom, Tensor) or deviation_angstrom.numel() != 1:
        raise ValueError("deviation_angstrom must be a single measured deviation")
    if not isinstance(draw, Tensor) or not draw.requires_grad:
        raise ValueError(
            "draw must be the coordinate tensor the deviation was differentiated from, and it must "
            "require grad; there is no gradient to decompose otherwise"
        )
    (gradient,) = torch.autograd.grad(
        deviation_angstrom.reshape(()), draw, retain_graph=True, create_graph=False
    )
    return rigid_body_leak(gradient.reshape(-1, 3), draw.detach().reshape(-1, 3))


# ---------------------------------------------------------------------------------------------
# THE DEPOSITED CONFORMERS: the sidecar, its signature, and the per-batch call site.
# ---------------------------------------------------------------------------------------------

#: Bumped whenever the stored layout changes in a way an older reader cannot interpret.
METRIC_REFERENCE_SCHEMA_VERSION = 1

#: The conventional file name beside ``pocket_artifact.json``. A CONVENTION and not a default: the
#: arm config names the path explicitly, because a sidecar found by convention is a sidecar nobody
#: chose, and this one determines what the objective descends.
METRIC_REFERENCE_FILENAME = "metric_reference.json"

#: How far the stored ``s_p`` may sit from the value recomputed from the stored backbones, in
#: Angstrom. Loose enough for the JSON round trip and float64 summation order over a few hundred
#: atoms; far tighter than any real disagreement, because a one-residue index slip moves an aligned
#: RMS by tenths of an Angstrom at least.
METRIC_SEPARATION_TOLERANCE_ANGSTROM = 1e-6

#: The smallest ``s_p`` a protein may carry into the arm, in Angstrom. Not "greater than zero", and
#: that is a MEASURED distinction rather than caution: ``aligned_rmsd`` adds ``1e-12`` inside its
#: square root so the derivative stays finite at coincident inputs, and therefore reports **1e-6
#: Angstrom** for two identical structures rather than 0 (its own docstring says so). A "> 0" gate
#: is satisfied by that floor, and ``s_p`` is the DIVISOR of the objective: a protein whose
#: deposited conformers coincide would enter the batch at ``m / 1e-6`` and own every gradient in it.
#:
#: The value is :data:`sae_bioemu.cryptic_steering.pocket.COORDINATE_UNCERTAINTY_ANGSTROM`, copied
#: by value rather than imported for the reason :data:`CRYPTIC_METRIC_ATOMS` is: ``pocket.py`` is in
#: ``cryptic_generation_shared`` and this file is ``cryptic_training_only``, so an import would let
#: the trained objective change at an unmoved training digest. Below it the two crystals are one
#: structure to within this project's own stated coordinate uncertainty, and there is no transition
#: to steer along.
MINIMUM_SEPARATION_ANGSTROM = 0.5


def _canonical_json_bytes(value: object) -> bytes:
    """Canonical encoding, identical to the artifact's, so two builds are byte-identical."""
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


@dataclass(frozen=True)
class MetricReferenceEntry:
    """One protein's two deposited conformers, over the one region the metric is read on.

    ``metric_indices`` are positions in the POCKET ARTIFACT's own sequence -- the string handed to
    the generator -- ascending and unique, and the two backbones are stored ALREADY RESTRICTED to
    them. Storing the restriction rather than a full-length array with a mask is what keeps the
    sidecar the size of a pocket instead of the size of a pool, and it is also what makes the one-set
    rule structural: there is exactly one index list here, so the alignment set and the metric set
    cannot differ (see :func:`require_one_region`, which states the same rule for the benchmark's
    two-list schema).

    The region is chosen by the DEPOSITED PAIR and never by a draw. A residue enters it only when
    all four of :data:`CRYPTIC_METRIC_ATOMS` are modelled in BOTH crystals, so the two levels of one
    sequence and the ``s_p`` they are normalised by are three numbers over ONE atom set.
    """

    sequence_id: str
    #: SHA-256 of the sequence the indices are positions in. Checked against the artifact's.
    sequence_sha256: str
    metric_indices: tuple[int, ...]
    #: ``[metric residues, 4, 3]`` float64 ANGSTROM, in :data:`CRYPTIC_METRIC_ATOMS` order.
    apo_backbone: Tensor
    holo_backbone: Tensor
    #: ``s_p``: the superposed :data:`TRAINED_DEVIATION_FORM` deviation between the two backbones
    #: above. Stored AND recomputed on every read, so it is a checkable redundancy rather than a
    #: declaration -- a sidecar whose separation came from a different measurement is refused.
    separation_angstrom: float

    def backbone_for_state(self, state: str) -> Tensor:
        """Return the conformer a reference state NAMES: :data:`APO` or :data:`HOLO`."""
        if state == APO:
            return self.apo_backbone
        if state == HOLO:
            return self.holo_backbone
        raise ValueError(f"state must be {APO!r} or {HOLO!r}; received {state!r}")


@dataclass(frozen=True)
class CrypticMetricReference:
    """The deposited-conformer sidecar for one pocket artifact, and the signature binding them.

    A SIDECAR for the reason :mod:`~sae_bioemu.cryptic_steering.apo_reference` is one, and signed
    for the reason that module spells out at length: a second file can be paired with the wrong
    artifact, and one protein's backbone read under another protein's indices produces a complete,
    finite, plausible deviation for every draw and names the wrong geometry. The signature is on
    both halves of the artifact's identity -- the whole-payload ``content_fingerprint`` and each
    entry's own sequence digest -- and :meth:`entry_for` re-checks the per-sequence half on every
    read, so a caller cannot obtain a backbone without the binding having been checked.
    """

    path: Path
    schema_version: int
    units: str
    atoms: tuple[str, ...]
    deviation_form: str
    artifact_content_fingerprint: str
    artifact_sequence_digest: str
    file_sha256: str
    content_fingerprint: str
    entries: dict[str, MetricReferenceEntry]

    @property
    def sequence_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.entries))

    def identity(self) -> dict[str, object]:
        """The block the arm identity carries so a checkpoint names the geometry it descended.

        The per-protein ``s_p`` values are deliberately NOT here, for the reason the objective
        identity omits them: they are measurements of a deposited pair, 592 of them in an identity
        would be a measurement table wearing an identity's clothes, and ``content_fingerprint``
        covers every one of them anyway.
        """
        return {
            "schema_version": self.schema_version,
            "units": self.units,
            "atoms": list(self.atoms),
            "deviation_form": self.deviation_form,
            "sequence_count": len(self.entries),
            "artifact_content_fingerprint": self.artifact_content_fingerprint,
            "artifact_sequence_digest": self.artifact_sequence_digest,
            "file_sha256": self.file_sha256,
            "content_fingerprint": self.content_fingerprint,
        }

    def require_artifact(self, artifact: object) -> None:
        """Fail closed unless this sidecar was written for ``artifact``, naming both sides."""
        identity = artifact.identity()  # type: ignore[attr-defined]
        recorded = str(identity["content_fingerprint"])
        if self.artifact_content_fingerprint != recorded:
            raise ValueError(
                f"metric reference {self.path} was signed against pocket artifact "
                f"content_fingerprint {self.artifact_content_fingerprint!r}, but the arm's "
                f"artifact carries {recorded!r}. The sidecar holds the deposited conformers the "
                "direct metric is measured to and the artifact holds the sequence those residue "
                "indices are positions in, so pairing two builds would measure one protein's "
                "deviation over another's residues"
            )
        digest = str(identity["sequence_digest"])
        if self.artifact_sequence_digest != digest:
            raise ValueError(
                f"metric reference {self.path} records artifact sequence_digest "
                f"{self.artifact_sequence_digest!r} and the arm's artifact hashes to {digest!r}"
            )

    def require_sequences(self, sequence_ids: Sequence[str], artifact: object) -> None:
        """Refuse unless every named sequence has a checked entry. The COVERAGE gate.

        Run over the WHOLE schedule at configuration time -- train, validation and test -- because
        a validation row journals the deviation too, so a protein missing here is a run that dies
        at its first validation round rather than at its first training batch. The uncovered set is
        reported as a population, like every other coverage gate in this project: these come in
        populations (a rebuilt structure bank, a widened split) and a message naming one of 2950
        would be read as an accident.
        """
        missing = sorted({str(name) for name in sequence_ids} - set(self.entries))
        if missing:
            raise ValueError(
                f"metric reference {self.path} covers {len(self.entries)} sequences and the "
                f"schedule names {len(missing)} it does not: {missing[:10]}"
                f"{' ...' if len(missing) > 10 else ''}. lambda_m > 0 measures every scheduled "
                "draw against a deposited conformer, including every validation draw, so an "
                "uncovered protein is a run that stops at its first round rather than a protein "
                "that is quietly skipped"
            )
        for sequence_id in sorted({str(name) for name in sequence_ids}):
            self.entry_for(sequence_id, artifact)

    def entry_for(self, sequence_id: str, artifact: object) -> MetricReferenceEntry:
        """Return one protein's checked entry, or fail closed.

        ``artifact`` is required rather than convenient, exactly as it is for
        :meth:`~sae_bioemu.cryptic_steering.apo_reference.ApoReferenceTable.coordinates_for`: the
        indices are positions in THAT artifact's sequence, so a reader able to obtain a backbone
        without presenting it could obtain it without the binding having been checked.
        """
        entry = self.entries.get(sequence_id)
        if entry is None:
            raise KeyError(
                f"metric reference {self.path} does not describe {sequence_id!r}; it carries "
                f"{len(self.entries)} sequences. There is no fallback: without the deposited "
                "conformers this protein has no direct metric at all, and a draw scored against a "
                "substitute would be a finite number measured to the wrong structure"
            )
        sequence = artifact.sequences.get(sequence_id)  # type: ignore[attr-defined]
        if sequence is None:
            raise KeyError(
                f"the pocket artifact does not describe {sequence_id!r}, which the metric "
                f"reference {self.path} carries"
            )
        from sae_bioemu.cryptic_steering.artifact import sequence_digest

        expected = sequence_digest(sequence)
        if entry.sequence_sha256 != expected:
            raise ValueError(
                f"{sequence_id}: the metric reference carries sequence sha256 "
                f"{entry.sequence_sha256!r} and the pocket artifact's sequence hashes to "
                f"{expected!r}. The metric indices are positions in the sidecar's sequence, so "
                "they address different residues than the artifact's do"
            )
        checked_metric_entry(
            sequence_id,
            entry.metric_indices,
            entry.apo_backbone,
            entry.holo_backbone,
            separation_angstrom=entry.separation_angstrom,
            residue_count=len(sequence),
        )
        return entry

    def normalisation(self, sequence_id: str, *, tau_angstrom: float) -> MetricNormalisation:
        """Return this protein's ``{s_p, tau_p}`` record, for a REPORT and never for the loss.

        The loss path reads ``s_p`` per observation through :func:`metric_deviations_for_batch` and
        applies the form the objective identity records; this exists so a run can state what weight
        each protein's draws carried. ``tau_angstrom`` is required and has no default here for the
        reason it has none anywhere in this module.
        """
        return MetricNormalisation(
            sequence_id=sequence_id,
            separation_angstrom=self.entries[sequence_id].separation_angstrom,
            tau_angstrom=float(tau_angstrom),
        )


def checked_metric_entry(
    sequence_id: str,
    metric_indices: Sequence[int],
    apo_backbone: Tensor,
    holo_backbone: Tensor,
    *,
    separation_angstrom: float,
    residue_count: int,
) -> float:
    """Return the recomputed ``s_p``, or fail closed. Used on BOTH write and read.

    One definition called from both sides, in ``apo_reference._checked_entry``'s pattern, because
    geometry validated only where it is written is geometry a hand-edited file carries past every
    reader. The recomputation is the load-bearing check: ``s_p`` divides the objective, and a stored
    value taken from a different measurement -- the mean form rather than the RMS, say, which the
    module docstring measures at 4.5-34% apart -- would rescale the ask on that protein with every
    other number staying finite.
    """
    indices = tuple(int(value) for value in metric_indices)
    if not indices:
        raise ValueError(
            f"{sequence_id}: no residue of the metric region is resolved in both deposited "
            "conformers, so this system has no metric set at all; it belongs out of the arm rather "
            "than in it with an empty measurement"
        )
    if any(later <= earlier for earlier, later in zip(indices, indices[1:], strict=False)):
        raise ValueError(
            f"{sequence_id}: metric_indices must be strictly ascending and unique; a repeated "
            "residue would enter the superposition twice and weight it"
        )
    if indices[0] < 0 or indices[-1] >= int(residue_count):
        raise ValueError(
            f"{sequence_id}: metric_indices run {indices[0]}..{indices[-1]} but the artifact's "
            f"sequence has {int(residue_count)} residues, so at least one index addresses a "
            "residue the generator never produces"
        )
    atoms = len(CRYPTIC_METRIC_ATOMS)
    for name, tensor in (("apo_backbone", apo_backbone), ("holo_backbone", holo_backbone)):
        if tuple(tensor.shape) != (len(indices), atoms, 3):
            raise ValueError(
                f"{sequence_id}: {name} must be [{len(indices)}, {atoms}, 3] holding "
                f"{'/'.join(CRYPTIC_METRIC_ATOMS)} in Angstrom; got {tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{sequence_id}: {name} carries a non-finite coordinate")
    recomputed = float(
        aligned_metric_deviation(
            apo_backbone.to(torch.float64),
            holo_backbone.to(torch.float64),
            torch.arange(len(indices)),
        )
    )
    if not recomputed >= MINIMUM_SEPARATION_ANGSTROM:
        raise ValueError(
            f"{sequence_id}: the deposited apo and holo sit {recomputed} Angstrom apart over the "
            f"metric region, below MINIMUM_SEPARATION_ANGSTROM={MINIMUM_SEPARATION_ANGSTROM}. They "
            "are one structure to within this project's stated coordinate uncertainty, so there is "
            "no transition to steer along -- and s_p is the DIVISOR of the objective, so this "
            "protein would own every gradient in its batch. It belongs out of the arm. (A '> 0' "
            "gate would not catch it: aligned_rmsd reports 1e-6 for coincident inputs, not 0.)"
        )
    if abs(recomputed - float(separation_angstrom)) > METRIC_SEPARATION_TOLERANCE_ANGSTROM:
        raise ValueError(
            f"{sequence_id}: the metric reference stores separation_angstrom "
            f"{float(separation_angstrom)!r} and the stored backbones give {recomputed!r}. s_p is "
            f"the DIVISOR of the objective and must be {TRAINED_DEVIATION_FORM!r} over exactly "
            "these atoms; a value from another measurement rescales this protein's ask while every "
            "other number stays finite"
        )
    return recomputed


def load_cryptic_metric_reference(path: str | Path) -> CrypticMetricReference:
    """Read and fully check one metric-reference sidecar. Opens nothing else."""
    resolved = Path(path).resolve()
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise ValueError(f"metric reference is unreadable: {resolved}") from error
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"metric reference {resolved} must contain a mapping")
    version = payload.get("schema_version")
    if version != METRIC_REFERENCE_SCHEMA_VERSION:
        raise ValueError(
            f"metric reference {resolved} declares schema_version {version!r}, not the supported "
            f"{METRIC_REFERENCE_SCHEMA_VERSION}"
        )
    atoms = tuple(str(name) for name in payload.get("atoms", ()))
    if atoms != CRYPTIC_METRIC_ATOMS:
        raise ValueError(
            f"metric reference {resolved} stores atoms {list(atoms)} and this objective measures "
            f"{list(CRYPTIC_METRIC_ATOMS)} in that order. A reference read in one order against a "
            "draw built in another is a per-residue permutation that leaves every deviation finite"
        )
    if payload.get("deviation_form") != TRAINED_DEVIATION_FORM:
        raise ValueError(
            f"metric reference {resolved} stores a {payload.get('deviation_form')!r} separation "
            f"and this objective normalises by {TRAINED_DEVIATION_FORM!r}; the two differ by "
            "4.5-34% on this project's own geometry"
        )
    entries: dict[str, MetricReferenceEntry] = {}
    for item in payload.get("entries", ()):
        sequence_id = str(item["sequence_id"])
        indices = tuple(int(value) for value in item["metric_indices"])
        apo_backbone = torch.tensor(item["apo_backbone"], dtype=torch.float64)
        holo_backbone = torch.tensor(item["holo_backbone"], dtype=torch.float64)
        separation = float(item["separation_angstrom"])
        if sequence_id in entries:
            raise ValueError(f"metric reference {resolved} names {sequence_id!r} twice")
        entries[sequence_id] = MetricReferenceEntry(
            sequence_id=sequence_id,
            sequence_sha256=str(item["sequence_sha256"]),
            metric_indices=indices,
            apo_backbone=apo_backbone,
            holo_backbone=holo_backbone,
            separation_angstrom=separation,
        )
    if not entries:
        raise ValueError(f"metric reference {resolved} describes no sequence")
    return CrypticMetricReference(
        path=resolved,
        schema_version=int(version),
        units=str(payload["units"]),
        atoms=atoms,
        deviation_form=str(payload["deviation_form"]),
        artifact_content_fingerprint=str(payload["artifact_content_fingerprint"]),
        artifact_sequence_digest=str(payload["artifact_sequence_digest"]),
        file_sha256=hashlib.sha256(raw).hexdigest(),
        content_fingerprint=hashlib.sha256(
            _canonical_json_bytes(payload.get("entries", []))
        ).hexdigest(),
        entries=entries,
    )


def write_cryptic_metric_reference(
    path: str | Path,
    artifact: object,
    entries: Mapping[str, tuple[Sequence[int], Tensor, Tensor]],
) -> CrypticMetricReference:
    """Write one sidecar and read it straight back, so nothing is written that cannot be loaded.

    ``entries`` maps a sequence id to ``(metric_indices, apo_backbone, holo_backbone)``; ``s_p`` is
    NOT taken from the caller, it is COMPUTED here by :func:`checked_metric_entry` from the
    backbones being stored. That is deliberate: the divisor of the objective must be a function of
    the bytes on disk and of nothing else, so a builder cannot hand in a separation measured on a
    different atom set, a different region or the mean form.

    Every entry goes through the same check the reader runs, and the file is re-read before this
    returns, so a sidecar that exists is a sidecar that loads.
    """
    from sae_bioemu.cryptic_steering.artifact import sequence_digest

    identity = artifact.identity()  # type: ignore[attr-defined]
    rows: list[dict[str, object]] = []
    for sequence_id in sorted(entries):
        indices, apo_backbone, holo_backbone = entries[sequence_id]
        sequence = artifact.sequences.get(sequence_id)  # type: ignore[attr-defined]
        if sequence is None:
            raise KeyError(
                f"the pocket artifact does not describe {sequence_id!r}, so a metric reference "
                "entry for it would index a sequence the generator never sees"
            )
        separation = checked_metric_entry(
            sequence_id,
            indices,
            apo_backbone.to(torch.float64),
            holo_backbone.to(torch.float64),
            # Checked against itself: the recomputation IS the value, so this cannot disagree and
            # the reader's identical check is what makes the stored number meaningful.
            separation_angstrom=float(
                aligned_metric_deviation(
                    apo_backbone.to(torch.float64),
                    holo_backbone.to(torch.float64),
                    torch.arange(len(tuple(indices))),
                )
            ),
            residue_count=len(sequence),
        )
        rows.append(
            {
                "sequence_id": sequence_id,
                "sequence_sha256": sequence_digest(sequence),
                "metric_indices": [int(value) for value in indices],
                "apo_backbone": apo_backbone.to(torch.float64).tolist(),
                "holo_backbone": holo_backbone.to(torch.float64).tolist(),
                "separation_angstrom": separation,
            }
        )
    payload = {
        "schema_version": METRIC_REFERENCE_SCHEMA_VERSION,
        "units": "angstrom",
        "atoms": list(CRYPTIC_METRIC_ATOMS),
        "deviation_form": TRAINED_DEVIATION_FORM,
        "artifact_content_fingerprint": str(identity["content_fingerprint"]),
        "artifact_sequence_digest": str(identity["sequence_digest"]),
        "entries": rows,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(payload))
    temporary.replace(destination)
    table = load_cryptic_metric_reference(destination)
    table.require_artifact(artifact)
    table.require_sequences(sorted(entries), artifact)
    return table


def metric_backbone_atoms(
    positions_nm: Tensor, node_orientations: Tensor, sequences: Sequence[str]
) -> Tensor:
    """Return ``[members, width, 4, 3]`` N/CA/C/O in ANGSTROM, GRADIENT INTACT.

    ``positions_nm`` is ``[members, width, 3]`` in BioEmu's own NANOMETRES, exactly as
    ``BioEmuSteeredSample.coordinates`` is, and ``node_orientations`` is ``[members, width, 3, 3]``,
    both padded to the batch width. ``sequences`` gives each member's own length, and rows past it
    are never read: they are frames of a residue that does not exist and would come back as
    entirely plausible coordinates.

    **Nothing is detached.** The scored structure carries the adapter's gradient and this is on the
    path from that structure to the loss, so a ``.detach()`` here would silently train nothing.

    WHY THIS IS NOT A CALL INTO ``bioemu/cryptic_generation.py``. That file's
    ``backbone_from_frames`` places three of these four atoms and must not be extended to place the
    fourth: it is the sole member of the ``two_state_generation_shared`` code group, which the
    cryptic AND kinase registrations both compose, so returning O from it would re-key every kinase
    checkpoint for an atom no kinase term reads. The duplication is one line of indexing: both
    resolve to ONE construction, ``bioemu.convert_chemgraph.get_atom37_from_frames``, which is what
    BioEmu itself writes to a PDB, and ``tests/test_cryptic_steering_direct_metric.py`` pins the
    N/CA/C this returns to the ones ``cryptic_generation`` returns so the two cannot drift into
    placing different atoms. ``domain_motion/frames.py`` is a third copy for the same reason and
    with the same pin; it is not imported here, because that file is hashed in
    ``domain_motion_training_only`` and a cryptic checkpoint must not depend on code its own digest
    does not cover.

    The O index is **4** and not 3. OpenFold's atom37 ordering is ``[N, CA, C, CB, O, ...]``, so
    index 3 is the C-beta, and reading it as the carbonyl would put a real atom in the O column and
    pass every finiteness and bond-length check there is.
    """
    if positions_nm.ndim != 3 or positions_nm.shape[-1] != 3:
        raise ValueError("positions must be [members, residues, 3] in nanometres")
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
    try:
        from bioemu.convert_chemgraph import get_atom37_from_frames
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "bioemu.convert_chemgraph is required to place the metric's backbone from frames"
        ) from error

    rows: list[Tensor] = []
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        if length > width:
            raise ValueError(
                f"batch member {index} has {length} residues but the batch is padded to {width}"
            )
        # Angstrom in, Angstrom out: ``get_atom37_from_frames``' own docstring says nanometres and
        # is wrong -- the ideal geometry it composes with is in Angstrom and BioEmu's own
        # ``save_pdb_and_xtc`` passes Angstrom. The peptide-bond window would catch the other
        # reading by a factor of ten.
        positions_angstrom = positions_nm[index, :length] * 10.0
        atom37, _, _ = get_atom37_from_frames(
            pos=positions_angstrom,
            node_orientations=node_orientations[index, :length],
            sequence=sequence,
        )
        placed = atom37.to(positions_angstrom)
        row = torch.stack([placed[:, _ATOM37_METRIC_INDEX[name]] for name in CRYPTIC_METRIC_ATOMS], dim=1)  # fmt: skip
        rows.append(row)
    padded = [
        row
        if int(row.shape[0]) == width
        else torch.cat(
            [row, row.new_zeros((width - int(row.shape[0]), len(CRYPTIC_METRIC_ATOMS), 3))], dim=0
        )
        for row in rows
    ]
    return torch.stack(padded)


def metric_deviations_for_batch(
    reference: CrypticMetricReference,
    artifact: object,
    sequence_ids: Sequence[str],
    levels: Sequence[float],
    backbone_angstrom: Tensor,
) -> tuple[Tensor, Tensor]:
    """**The per-batch call site.** Return ``(m, s_p)``, one entry per member, in schedule order.

    ``backbone_angstrom`` is ``[members, width, 4, 3]`` from :func:`metric_backbone_atoms`, padded
    to the batch width; each member's own ``metric_indices`` address only its own residues, so the
    padding is never read. ``levels`` is that member's dose level, and it SELECTS THE DEPOSITED
    CONFORMER through :func:`reference_state_for_level` -- ``+1`` holo, ``-1`` apo -- rather than
    a sign, because a deviation is unsigned and ``success@B`` scores the two states separately.

    ``m`` carries the gradient and ``s_p`` does not: the separation is a measurement of two crystals
    and is not something an adapter may move. Both come back as ``[members]`` tensors on the draw's
    own dtype and device, which is the shape
    :meth:`~sae_bioemu.cryptic_steering.loss.CrypticSteeringLoss.forward` requires of every
    per-observation input.

    The level is read off the batch rather than inferred, because a deviation to the wrong conformer
    is a NUMBER and not an error.
    """
    members = tuple(str(name) for name in sequence_ids)
    if not members:
        raise ValueError("a generated batch must hold at least one member")
    if len(levels) != len(members):
        raise ValueError("one dose level per batch member is required to select its conformer")
    if backbone_angstrom.ndim != 4 or tuple(backbone_angstrom.shape[-2:]) != (
        len(CRYPTIC_METRIC_ATOMS),
        3,
    ):
        raise ValueError(
            f"backbone_angstrom must be [members, residues, {len(CRYPTIC_METRIC_ATOMS)}, 3] in "
            f"Angstrom; got {tuple(backbone_angstrom.shape)}"
        )
    if int(backbone_angstrom.shape[0]) != len(members):
        raise ValueError("backbone_angstrom must carry one structure per batch member")
    deviations: list[Tensor] = []
    separations: list[float] = []
    for index, sequence_id in enumerate(members):
        entry = reference.entry_for(sequence_id, artifact)
        state = reference_state_for_level(float(levels[index]))
        indices = torch.as_tensor(entry.metric_indices, device=backbone_angstrom.device)
        draw = backbone_angstrom[index].index_select(0, indices)
        deposited = entry.backbone_for_state(state).to(draw)
        deviations.append(
            aligned_metric_deviation(
                draw, deposited, torch.arange(len(entry.metric_indices), device=draw.device)
            ).reshape(())
        )
        separations.append(entry.separation_angstrom)
    stacked = torch.stack(deviations)
    return stacked, stacked.detach().new_tensor(separations)
