"""The DIRECT-METRIC domain-motion objective: train on the distance the benchmark grades.

SCOPE: DOMAIN MOTIONS. Everything here is a function of coordinates, a reference conformer and a
dose level. Nothing reads a file, a config or a checkpoint.

THE OBJECTIVE
-------------

Per (sequence ``p``, reference ``r``, draw):

    X = N/CA/C/O of the steered draw, off the differentiable frames (:mod:`.frames`)
    Y = the same atoms of reference ``r``, over the residues resolved in BOTH conformers
    m = aligned_rmsd(X, Y)                        Angstrom; alignment set == metric set
    L = 0.5 * (m / s_p)^2                         PLAIN MSE on the per-system scale -- the DEFAULT
      = huberised_hinge((m - tau_p) / s_p)        the ABLATION: zero below tau_p, then quadratic,
                                                  then linear

and the property term the shared loss prices is ``mean over observations of L``.

This replaces a progress-coordinate proxy, and the reason is D-63's hazard read forwards rather
than backwards: the cryptic programme trained four months of arms on ``the progress coordinate`` and only afterwards
asked whether moving ``the progress coordinate`` moved the reported metric. On this pool the answer is measured and it
is weak -- median within-protein Spearman(the progress coordinate, RMSD to target) of -0.243 over 1476 systems, with
32.1% of systems carrying the WRONG SIGN, i.e. raising the coordinate moves the draw AWAY from the
conformer the metric scores it against (``docs/domain_motion/dm1_coordinate_predicts_metric.json``).
A direct metric has no such gap to close: the quantity descended and the quantity graded are one
quantity.

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
   * ``tau`` is an EVALUATION parameter. :data:`DIRECT_METRIC_TAU_ANGSTROM` is the threshold
     ``success@B`` is REPORTED at -- and ConforMix's ``adaptive_tau`` is reported beside it -- so a
     tau baked into the training objective makes two readings of one run two readings of two
     differently-trained adapters.

   **This is a DEFAULT change, not a second implementation.** Plain MSE is exactly the special case
   ``tau = 0, transition -> infinity`` of :func:`huberised_hinge`: below the knee the hinge IS
   ``0.5*z^2``, and at ``tau = 0`` with ``m >= 0`` the clamp at zero never binds. Measured rather
   than asserted -- max absolute difference **0.0** against ``huberised_hinge(m/s_p,
   transition=1e6)`` -- and ``tests/test_domain_motion_direct_metric.py`` pins :func:`metric_mse` to
   that limit so the two cannot drift into two objectives.

   What the hinge buys, and why it stays reachable: ``success@B`` has no value below ``tau`` and
   unbounded value above it, so an unbounded loss keeps paying gradient to polish draws that are
   already inside ``tau`` and lets one hopeless 16 Angstrom system own the batch gradient. A HARD
   hinge has the mirror defect: exactly zero gradient for the far draws, which are the actual work.
   The huberised hinge is flat where there is nothing to win, quadratic where the draw is close
   enough for a small push to convert it, and LINEAR far out, which bounds the per-draw gradient
   magnitude at :data:`HINGE_TRANSITION` / ``s_p`` however hopeless the system is. That is an
   argument for an ablation, and it is shipped as one: the artifact declares
   ``metric_objective``, the form rides the arm identity and the objective identity, and ``tau``
   reaches the objective on the hinge path alone.

2. **THE LEVEL SELECTS WHICH REFERENCE, NOT A SIGN.** A distance is unsigned and ``success@B``
   scores each conformer SEPARATELY, so ``+1`` means "minimise the RMSD to conformer j" and ``-1``
   means "minimise the RMSD to conformer i", at the SHARED draw seed
   (``schedule.SteeringUnit.draw_seed``, already shared across the levels of one (sequence, cycle)).
   Two things follow. The per-protein sign-identifiability problem recorded in D-77/D-78 does not
   arise, because there is no sign to invert -- which conformer is "open" never has to be decided.
   And the incoherent ask disappears: under a signed coordinate, level ``-1`` asks an
   already-i-like draw to become more i-like, which names no structural change; here it asks for a
   distance to be made small, which it either already is (and the hinge is silent) or is not.
   :data:`sae_bioemu.steering.schedule.TWO_STATE_TARGET_LEVELS` is unchanged and is read exactly as
   it stands: this is an interpretation of the two levels, not a new dose set.

3. **PER-SYSTEM NORMALISATION, WHICH SURVIVES THE MOVE TO PLAIN MSE.** ``s_p`` is the system's own
   reference separation and ``tau_p = min(tau, 0.5 * s_p)``; the MSE form keeps ``s_p`` and drops
   ``tau_p``, and that is the whole of the difference between the two forms. The SCALE is what makes
   the batch mean an average over systems, and that argument nowhere mentions a threshold; ``tau_p``
   IS the threshold, and it is exactly the evaluation parameter design point 1 takes out of the
   training objective. Dropping the scale as well would not be a simpler objective -- it would be
   one whose implicit weighting is not the metric's.

   This is not a fudge factor. ``success@B`` is an UNWEIGHTED MEAN
   OVER SYSTEMS of a bounded indicator, while a per-draw Angstrom loss is a mean over draws of an
   unbounded quantity, so without a per-system scale the objective's implicit weighting is "largest
   transitions first" and the metric's is "every system once". Matching the metric's own aggregation
   is part of optimising the metric. ``0.5 * s_p`` is ConforMix's ``adaptive_tau``, already
   implemented as :meth:`sae_bioemu.cryptic_steering.success_at_b.ReferenceSeparation.adaptive_tau`
   and already reported beside the fixed 3.0 (``docs/domain_motion/dm_success_metric.txt``); the
   ``min`` takes whichever of the two thresholds is HARDER, so the objective is never satisfied by a
   draw the reported metric would fail.

   **Stated honestly: this reintroduces a per-protein weight**, of the kind D-46 removed from the
   cryptic target. It is not the same defect -- D-46 added a per-protein CONSTANT to a displacement
   target, an intercept no dose slope could see, whereas this is a per-protein SCALE on a quantity
   whose zero is meaningful -- but it is a per-protein number in the objective and it does mean two
   systems' losses are not in one unit. It is recorded per system in the arm identity so a report
   can say which weight each protein carried.

4. **NO SOFT-MIN OVER DRAWS.** ``success@B`` takes a minimum over B draws, and it is tempting to
   put a soft-min inside the objective to match. With 2 levels x 1 draw per scheduled unit there is
   no min to take inside a unit, and a soft-min over 2 values is nearly a hard min: it would route
   the whole gradient to one level and silently halve the schedule. The evaluation takes the min;
   the hinge already encodes most of what the min buys, because a draw inside ``tau_p`` contributes
   nothing and stops competing with its siblings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from sae_bioemu.steering.aligned_rmsd import aligned_rmsd

__all__ = [
    "DEFAULT_METRIC_OBJECTIVE",
    "DIRECT_METRIC_TAU_ANGSTROM",
    "DOMAIN_MOTION_PROPERTY_NAME",
    "HINGE_TRANSITION",
    "METRIC_OBJECTIVE_FORMS",
    "METRIC_OBJECTIVE_HINGE",
    "METRIC_OBJECTIVE_MSE",
    "MINIMUM_SEPARATION_ANGSTROM",
    "REFERENCE_STATES",
    "SystemNormalisation",
    "aligned_backbone_rmsd",
    "huberised_hinge",
    "metric_mse",
    "reference_state_for_level",
    "validated_metric_objective",
]

#: ``target.property_name`` for this property. Spelled once, here, and imported by the descriptor
#: and by every test, so the catalog cannot route a name no descriptor answers for.
DOMAIN_MOTION_PROPERTY_NAME = "domain_motion_backbone_rmsd"

#: The benchmark's own success threshold, in Angstrom: true global RMSD over N/CA/C/O at 3.0, the
#: ``rmsd`` form and never the ``mean_deviation`` form. Settled from the upstream source, not from a
#: threshold table -- see ``docs/domain_motion/dm_success_metric.txt``.
DIRECT_METRIC_TAU_ANGSTROM = 3.0

#: Plain MSE on the per-system scale, ``L = 0.5*(m/s_p)^2``. The DEFAULT, and the form every
#: domain-motion arm gets unless its artifact names the other one.
METRIC_OBJECTIVE_MSE = "mse"
#: The one-sided huberised hinge on ``(m - tau_p)/s_p``. The ABLATION, and the only form that reads
#: a ``tau`` at all.
METRIC_OBJECTIVE_HINGE = "huberised_hinge"
#: The two forms, and there are exactly two. A third would be a third objective and would need its
#: own argument; an unrecognised spelling is refused by name rather than falling back to either, so
#: a typo cannot silently train an arm under the other form.
METRIC_OBJECTIVE_FORMS: tuple[str, ...] = (METRIC_OBJECTIVE_MSE, METRIC_OBJECTIVE_HINGE)
#: What an arm that names no form descends. See design point 1: a threshold in the objective is
#: premature here, it makes a null result unreadable, and ``tau`` is an EVALUATION parameter.
DEFAULT_METRIC_OBJECTIVE = METRIC_OBJECTIVE_MSE

#: The smallest ``s_p`` a system may carry into the arm, in Angstrom. Not "greater than zero", and
#: the distinction is MEASURED rather than cautious: :func:`~sae_bioemu.steering.aligned_rmsd.
#: aligned_rmsd` adds ``1e-12`` inside its square root so the derivative stays finite at coincident
#: inputs, and therefore reports about **1e-6 Angstrom** for two identical structures rather than 0.
#: A ``separation > 0`` gate is satisfied by that floor, and ``s_p`` DIVIDES this objective under
#: both forms: a system whose deposited conformers coincide would enter the batch at ``m / 1e-6``
#: and own every gradient in it.
#:
#: It is the cryptic programme's :data:`sae_bioemu.cryptic_steering.direct_metric.
#: MINIMUM_SEPARATION_ANGSTROM`, copied by value, and on THIS pool it is dominated:
#: :data:`~sae_bioemu.domain_motion.separation.SEPARATION_GATE_ANGSTROM` is 3.0 and is applied
#: strictly, so nothing that reaches the objective can be within 0.5 of coincident. It is declared
#: anyway, and asserted against the gate, because the two thresholds are set for different reasons
#: and a future pool gated more loosely would otherwise reintroduce the divisor hazard silently.
MINIMUM_SEPARATION_ANGSTROM = 0.5

#: Where the hinge turns from quadratic to linear, in units of the system's own separation ``s_p``.
#: At 1.0 the quadratic region is exactly one reference separation wide, so a draw that has to move
#: further than the whole transition is in the linear regime and contributes a BOUNDED gradient.
#: This is the parameter that decides how much of the batch one hopeless system can own, so it is
#: part of the objective's identity rather than a tuning knob hidden in a config.
HINGE_TRANSITION = 1.0

#: Which conformer of the deposited pair each two-state dose level names. The benchmark's own
#: ``references.csv`` columns are ``pdbidchain_i`` and ``pdbidchain_j``; ``first`` is ``i`` and
#: ``second`` is ``j``. Nothing here says one of them is open or active -- the benchmark does not
#: label them and there is no ligand to say so, which is precisely why a level selects a reference
#: rather than a sign (design point 2).
REFERENCE_STATES: Mapping[float, str] = {1.0: "second", -1.0: "first"}


def reference_state_for_level(level: float) -> str:
    """Return ``"first"`` or ``"second"``: which deposited conformer this level asks for.

    ``0.0`` is refused rather than mapped to a midpoint, for the reason
    :func:`sae_bioemu.steering.schedule.two_state_reporting_target` refuses it: there is no third
    deposited structure, and "half way between the two crystals" is not a conformer the protein
    has and not a thing the metric can score.
    """
    if isinstance(level, bool) or not isinstance(level, (int | float)):
        raise ValueError("level must be a real number")
    state = REFERENCE_STATES.get(float(level))
    if state is None:
        raise ValueError(
            "a domain-motion level NAMES A REFERENCE CONFORMER, so it must be exactly -1 "
            f"(conformer i) or +1 (conformer j); received {float(level)}. The metric is an "
            "unsigned distance to a deposited structure and there is no third structure for an "
            "intermediate level to be measured against"
        )
    return state


def validated_metric_objective(value: object) -> str:
    """Return one of :data:`METRIC_OBJECTIVE_FORMS`, or fail closed naming both.

    One validator, imported by the artifact loader, by the loss and by the property descriptor, so
    the form cannot be accepted under one spelling on disk and dispatched under another in the
    objective. ``None`` resolves to :data:`DEFAULT_METRIC_OBJECTIVE`, which is what an artifact that
    names no form means and what every artifact written before the key existed means.
    """
    if value is None:
        return DEFAULT_METRIC_OBJECTIVE
    if not isinstance(value, str) or value not in METRIC_OBJECTIVE_FORMS:
        raise ValueError(
            f"metric_objective must be one of {list(METRIC_OBJECTIVE_FORMS)}, received {value!r}. "
            f"{METRIC_OBJECTIVE_MSE!r} is 0.5*(m/s_p)^2 and is the default; "
            f"{METRIC_OBJECTIVE_HINGE!r} is the thresholded ablation and is the only form that "
            "reads a tau. There is no fallback: a misspelled form would silently train the other "
            "objective under a recorded identity that named neither"
        )
    return value


def metric_mse(metric_angstrom: Tensor, separation_angstrom: Tensor | float) -> Tensor:
    """Return :data:`METRIC_OBJECTIVE_MSE`: ``L = 0.5 * (m / s_p)^2``, elementwise over a batch.

    **The DEFAULT form, and deliberately the special case of the hinge rather than a second
    implementation of anything.** :func:`huberised_hinge` is ``0.5*z^2`` for ``0 < z <= transition``
    and ``0`` at ``z <= 0``; with ``tau = 0`` the argument is ``m/s_p``, which is nonnegative
    because an RMSD is, and with the knee taken to infinity the quadratic branch is the whole
    function. ``tests/test_domain_motion_direct_metric.py`` pins this numerically to that limit.

    ``separation_angstrom`` is kept and ``tau`` is gone, which is the entire difference from
    :meth:`SystemNormalisation.hinge`. There is no ``tau`` argument here to be defaulted, which is
    the strongest available statement that this form has no threshold.
    """
    if not isinstance(metric_angstrom, Tensor):
        raise TypeError("metric_angstrom must be a torch.Tensor")
    separation = (
        separation_angstrom
        if isinstance(separation_angstrom, Tensor)
        else torch.full_like(metric_angstrom, float(separation_angstrom))
    )
    separation = separation.to(metric_angstrom)
    if separation.shape != metric_angstrom.shape:
        raise ValueError(
            "separation_angstrom is PER OBSERVATION -- each system's own reference separation -- "
            f"so it must share the metric's shape; got {tuple(separation.shape)} and "
            f"{tuple(metric_angstrom.shape)}"
        )
    if bool(torch.any(separation <= 0)):
        raise ValueError(
            "separation_angstrom must be strictly positive for every observation: it is the "
            "DIVISOR the objective is measured in, and a system whose deposited conformers "
            f"coincide has no motion to steer along (see MINIMUM_SEPARATION_ANGSTROM="
            f"{MINIMUM_SEPARATION_ANGSTROM})"
        )
    scaled = metric_angstrom / separation
    return 0.5 * scaled * scaled


def huberised_hinge(value: Tensor, *, transition: float = HINGE_TRANSITION) -> Tensor:
    """Return the one-sided Huber of ``value``: 0 at or below zero, then quadratic, then linear.

        L(z) = 0                           z <= 0
             = 0.5 * z^2                   0 < z <= transition
             = transition * (z - 0.5*transition)   z > transition

    Continuous in value and in first derivative at both breakpoints, which is what makes it usable
    as a loss: the gradient rises linearly from 0 at ``z = 0`` to ``transition`` and is CONSTANT
    past it. The flat half is what stops the objective polishing draws that already succeed; the
    linear half is what stops one far draw owning the batch.

    Written with ``clamp`` rather than ``where`` so the zero region's gradient is exactly zero
    rather than a masked NaN, and so no ``sqrt`` is differentiated at the origin.
    """
    if not isinstance(value, Tensor):
        raise TypeError("value must be a torch.Tensor")
    if not isinstance(transition, float) or not transition > 0.0:
        raise ValueError("transition must be a positive float")
    positive = torch.clamp(value, min=0.0)
    quadratic = torch.clamp(positive, max=transition)
    # ``positive - quadratic`` is the part beyond the knee and is exactly zero inside it, so the
    # two branches are one expression and no branch is ever selected by a comparison on a tensor.
    return 0.5 * quadratic * quadratic + transition * (positive - quadratic)


def aligned_backbone_rmsd(draw: Tensor, reference: Tensor, metric_residues: Tensor) -> Tensor:
    """Return the superposed N/CA/C/O RMSD of one draw against one reference conformer.

    ``draw`` and ``reference`` are ``[residues, 4, 3]`` in Angstrom, in :data:`.frames.
    BACKBONE_ATOMS` order; ``metric_residues`` is a 1-D index or boolean selector of the residues
    the metric is read over.

    **The metric set is chosen by the two CRYSTAL conformers, not by the draw.** A draw has every
    residue of the folded sequence; a deposited structure does not, and the two deposited structures
    do not miss the same ones. Taking the residues resolved in BOTH is what makes the two levels of
    one sequence, and the reference separation ``s_p`` they are normalised by, three numbers over
    ONE atom set. Measuring each level over its own reference's resolved set would make ``-1`` and
    ``+1`` incomparable and the normalisation meaningless.

    The superposition is solved on exactly the atoms the metric is read over --
    :func:`~sae_bioemu.steering.aligned_rmsd.aligned_rmsd` refuses any other arrangement by name --
    which is what makes the detached rotation exact by the envelope theorem and the gradient pure
    shape change. Report :func:`~sae_bioemu.steering.aligned_rmsd.rigid_body_leak` beside the loss.
    """
    if draw.ndim != 3 or draw.shape[-1] != 3 or draw.shape[-2] != 4:
        raise ValueError("draw must be [residues, 4, 3] N/CA/C/O in Angstrom")
    if reference.shape != draw.shape:
        raise ValueError(
            f"reference must share the draw's shape; got {tuple(reference.shape)} and "
            f"{tuple(draw.shape)}. The metric is defined between corresponding atoms"
        )
    selected_draw = draw[metric_residues]
    selected_reference = reference[metric_residues]
    if int(selected_draw.shape[0]) < 1:
        raise ValueError(
            "no residue is resolved in both deposited conformers, so this system has no metric "
            "set at all; it belongs out of the pool rather than in it with an empty measurement"
        )
    # Flattened to (4K, 3): the metric is an RMSD over ATOMS, so all four of a residue's atoms
    # enter the mean and the superposition individually, exactly as ``calc_metrics_global`` does.
    return aligned_rmsd(
        selected_draw.reshape(-1, 3), selected_reference.reshape(-1, 3).to(selected_draw)
    )


@dataclass(frozen=True)
class SystemNormalisation:
    """One system's own scale and threshold: ``s_p`` and ``tau_p``. Design point 3.

    Built once per system from the two deposited conformers and carried in the arm identity, so a
    report can say what weight each protein's draws carried rather than inferring it.
    """

    sequence_id: str
    #: ``s_p``: the superposed N/CA/C/O RMSD between the two deposited conformers over the shared
    #: metric set, in Angstrom. The TRUE metric, not the virtual-C-beta proxy the DM-1 pool survey
    #: used: one 3.0 Angstrom threshold applied across the two is not an apples-to-apples gate.
    separation_angstrom: float
    #: The fixed benchmark threshold this system was gated and normalised against.
    tau_angstrom: float = DIRECT_METRIC_TAU_ANGSTROM

    def __post_init__(self) -> None:
        if not self.sequence_id:
            raise ValueError("a system normalisation must name its sequence")
        for name, value in (
            ("separation_angstrom", self.separation_angstrom),
            ("tau_angstrom", self.tau_angstrom),
        ):
            if isinstance(value, bool) or not isinstance(value, float) or not value > 0.0:
                raise ValueError(f"{name} must be a positive float")
        if self.separation_angstrom < MINIMUM_SEPARATION_ANGSTROM:
            raise ValueError(
                f"{self.sequence_id}: s_p = {self.separation_angstrom:.9f} Angstrom is below "
                f"MINIMUM_SEPARATION_ANGSTROM={MINIMUM_SEPARATION_ANGSTROM}. A '> 0' gate would "
                "pass it: aligned_rmsd carries eps=1e-12 inside its square root and reports about "
                "1e-6 Angstrom for two IDENTICAL structures. s_p DIVIDES this objective under both "
                "forms, so such a system enters the batch at m/1e-6 and owns every gradient in it"
            )

    @property
    def adaptive_tau_angstrom(self) -> float:
        """``0.5 * s_p``: ConforMix's per-system threshold, in Angstrom."""
        return 0.5 * self.separation_angstrom

    @property
    def effective_tau_angstrom(self) -> float:
        """``tau_p = min(tau, 0.5 * s_p)``: whichever of the two thresholds is HARDER.

        Taking the minimum rather than the fixed 3.0 means the objective is never satisfied by a
        draw the reported metric would fail, and taking it rather than the adaptive one alone means
        a 15 Angstrom hinge is not asked to reproduce the crystal to 7.5 Angstrom and called done.
        """
        return min(float(self.tau_angstrom), self.adaptive_tau_angstrom)

    def hinge(self, metric_angstrom: Tensor) -> Tensor:
        """Return ``L`` for one or more measured RMSDs of THIS system, per design point 3."""
        if not isinstance(metric_angstrom, Tensor):
            raise TypeError("metric_angstrom must be a torch.Tensor")
        offset = metric_angstrom - self.effective_tau_angstrom
        return huberised_hinge(offset / self.separation_angstrom)

    def mse(self, metric_angstrom: Tensor) -> Tensor:
        """Return ``L = 0.5*(m/s_p)^2`` for one or more measured RMSDs of THIS system."""
        return metric_mse(metric_angstrom, float(self.separation_angstrom))

    def priced(self, metric_angstrom: Tensor, metric_objective: str) -> Tensor:
        """Return the priced value under the named form. One dispatcher, so the journal row and
        the quantity descended cannot come from different arithmetic."""
        form = validated_metric_objective(metric_objective)
        return (
            self.mse(metric_angstrom)
            if form == METRIC_OBJECTIVE_MSE
            else self.hinge(metric_angstrom)
        )

    def target_constant_angstrom(self, metric_objective: str) -> float:
        """Return the per-system TARGET the shared ``system_constant`` origin is built from.

        The shared origin requires a STRICTLY POSITIVE per-system target
        (:class:`~sae_bioemu.steering.schedule.SystemTargetConstant`), and the two forms place
        their zero differently:

        * :data:`METRIC_OBJECTIVE_HINGE` -- ``tau_p``. The target IS the threshold, the prediction
          is ``max(m, tau_p)``, and ``prediction - target`` is ``max(m - tau_p, 0)``.
        * :data:`METRIC_OBJECTIVE_MSE` -- ``s_p``. The MSE has NO threshold, so its zero is ``m = 0``
          and the target is a pure ORIGIN that cancels: the prediction is ``m + s_p`` and
          ``prediction - target`` is exactly ``m``. ``s_p`` is used rather than ``tau_p`` precisely
          so that ``tau`` reaches no part of the MSE objective -- neither its value, nor its
          gradient, nor the digest of the system-target table the checkpoint is bound to.

        A shifted prediction is the price of not editing ``steering/schedule.py``, which is in
        ``generation_shared`` and whose blast radius is every steering checkpoint there has ever
        been, to admit a zero target. It is not hidden: ``DomainMotionSteeringLoss.forward``
        re-derives ``prediction - target`` against the measured ``m`` on the wire and refuses a
        disagreement, elementwise, on every batch.
        """
        form = validated_metric_objective(metric_objective)
        if form == METRIC_OBJECTIVE_MSE:
            return float(self.separation_angstrom)
        return self.effective_tau_angstrom

    def prediction(self, metric_angstrom: Tensor, metric_objective: str) -> Tensor:
        """Return the shared term's ``prediction`` for THIS system under the named form.

        ``max(m, tau_p)`` under the hinge -- the clamp is where the one-sidedness lives -- and
        ``m + s_p`` under the MSE, which is the measured metric placed on the origin
        :meth:`target_constant_angstrom` returns. Under both forms ``prediction - target`` is the
        quantity the form's own arithmetic takes, and under both ``prediction >= target`` holds.
        """
        if not isinstance(metric_angstrom, Tensor):
            raise TypeError("metric_angstrom must be a torch.Tensor")
        form = validated_metric_objective(metric_objective)
        constant = self.target_constant_angstrom(form)
        if form == METRIC_OBJECTIVE_MSE:
            return metric_angstrom + constant
        return torch.clamp(metric_angstrom, min=constant)

    def identity(self) -> dict[str, float | str]:
        """The per-system numbers that enter the arm identity."""
        return {
            "sequence_id": self.sequence_id,
            "separation_angstrom": round(float(self.separation_angstrom), 6),
            "effective_tau_angstrom": round(float(self.effective_tau_angstrom), 6),
        }


def direct_metric_hinge_loss(
    draws: Sequence[Tensor],
    references: Sequence[Tensor],
    metric_residues: Sequence[Tensor],
    normalisations: Sequence[SystemNormalisation],
) -> tuple[Tensor, Tensor]:
    """Return ``(metric_angstrom, hinge)``, one entry per observation, in batch order.

    Every argument is a per-observation sequence rather than a padded tensor, because the metric
    set differs per system: two members of one batch do not have the same residues resolved in both
    of their own conformers, and a padded tensor would need a per-member mask that says exactly
    this and nothing more.

    Both tensors come back: ``metric_angstrom`` is the quantity the benchmark grades and belongs in
    the journal in its own unit, and ``hinge`` is what the loss prices. Journalling only the hinge
    would make an arm's progress unreadable, because a hinge of zero means "inside tau" and says
    nothing about whether the draw is at 2.9 or at 0.4 Angstrom.
    """
    if not (len(draws) == len(references) == len(metric_residues) == len(normalisations)):
        raise ValueError(
            "draws, references, metric_residues and normalisations must be the same length: one "
            "entry per observation"
        )
    if not draws:
        raise ValueError("the direct metric needs at least one observation")
    metrics = [
        aligned_backbone_rmsd(draw, reference, residues)
        for draw, reference, residues in zip(draws, references, metric_residues, strict=True)
    ]
    stacked = torch.stack([value.reshape(()) for value in metrics])
    hinged = torch.stack(
        [
            normalisation.hinge(value.reshape(()))
            for normalisation, value in zip(normalisations, metrics, strict=True)
        ]
    )
    return stacked, hinged
