"""The DIRECT-METRIC kinase objective: the Angstrom distance to a deposited DFG conformer.

SCOPE: KINASES. Everything here is a function of coordinates, a reference conformer and a dose
level. Nothing reads a file, a config or a checkpoint, and nothing here knows what ``the progress coordinate`` is.

THE OBJECTIVE
-------------

Per (sequence ``p``, reference ``r``, draw):

    X = N/CA/C/O of the steered draw, off the differentiable frames (:mod:`.frames`)
    Y = the same atoms of deposited chain ``r``, over this system's METRIC SET
    m = aligned_rmsd(X, Y)                        Angstrom; alignment set == metric set
    L = 0.5 * (m / s_p)^2                         PLAIN MSE on the per-system scale -- the DEFAULT
      = huberised_hinge((m - tau_p) / s_p)        the ABLATION

and the property term the shared loss prices is ``mean over observations of L``.

WHY A DIRECT METRIC AND NOT ``the progress coordinate``
-----------------------------------

This is the kinase twin of D-63's hazard, and the kinase programme has already paid it. K-22
measured arm 8 descending ``d_K`` -- the Kincore cosine dihedral distance -- for 40 epochs while
the graded quantity moved the other way, and ``kinase_arm9_margin_workflow.yaml`` records the
consequence: within each (protein, level) cell, Spearman(|achieved the progress coordinate|, d_K) came out POSITIVE at
+0.118 / +0.163 / +0.297, i.e. raising the coordinate moved the draw AWAY from the label it was
meant to reach. A direct metric has no such gap to close: the quantity descended and the quantity
graded are one quantity, in Angstrom, against a chain somebody deposited.

THE METRIC SET IS THE ACTIVATION LOOP, AND THAT IS A MEASURED CHOICE
---------------------------------------------------------------------

The DFG-in/DFG-out transition is an ACTIVATION-LOOP rearrangement. The metric set is therefore
``activation_loop_start .. activation_loop_end`` together with ``{dfg_asp_index - 2}``, restricted
to residues carrying all four of N/CA/C/O in BOTH deposited chains. The extra residue is not
decorative: across all 43 pool genes ``activation_loop_start == dfg_asp_index - 1`` exactly, so
Kincore's five-residue motif window (X-1 through DFG-Gly) reaches one residue N-terminal of the
A-loop, and a metric set that stopped at the loop would exclude an atom the label itself is read
from.

A CONTEXT SHELL WAS CONSIDERED AND IS REFUTED BY MEASUREMENT, NOT BY ARGUMENT. All three candidate
sets were measured per gene over all 43 pool kinases in one pass, from one read of each pair of
chains, BEFORE this file was written (``scripts/kinase_rmsd/build_kinase_rmsd_reference.py
--separation-table``). The dilution is real and it is monotone::

                            residues (median)    s_p median    s_p mean    s_p > 3.0 A
    activation_loop                       16         2.669       4.185          18/43
    activation_loop_shell8                65         2.280       3.978          17/43
    activation_loop_shell12              136         2.071       3.355          17/43

The shell LOWERS the separation on the median and on the mean, and strictly lowers it for 29 of 43
genes at 8 Angstrom and 31 of 43 at 12. That is the attenuation predicted from the geometry: the
loop is about 12% of the 12 Angstrom union, so its contribution to a pooled RMSD is scaled by
``sqrt(15/125) = 0.346`` while ~110 residues that barely move between DFG-in and DFG-out carry the
rest. ``s_p`` is the DIVISOR of this whole objective, so a set that lowers it lowers the resolution
of every per-protein weight in the arm. The activation loop alone wins every column of that table,
and it is what is shipped.

The 12-to-14 genes whose ``s_p`` RISES under a shell are not noise and are worth naming: superposing
on the loop alone removes the loop's rigid-body displacement relative to the rest of the kinase,
while a shell anchors the superposition in the domain frame and lets that displacement count. For
CDK6 (7.54 -> 10.41) and PRKAA2 (6.60 -> 9.81) the displacement is the larger part of the motion.
A set that superposed on the CONTEXT and measured on the LOOP would capture both, and it is
deliberately NOT available here: this objective's alignment set IS its metric set, structurally
(see :func:`aligned_backbone_rmsd`), and separating the two is a different objective rather than a
tuning of this one.

FOUR DESIGN POINTS
-------------------

1. **PLAIN MSE BY DEFAULT; THE HINGE IS A TWO-VALUE ABLATION.** The reasons are the domain-motion
   objective's and they transfer without change: nothing in this programme has moved an Angstrom
   metric yet, so a threshold inside the objective refines something that does not exist; under the
   hinge "the term did nothing" and "the term had nothing to do" give the same loss curve; and
   ``tau`` is an EVALUATION parameter. Plain MSE is exactly the ``tau = 0, transition -> infinity``
   special case of :func:`huberised_hinge`, which is pinned numerically by
   ``tests/test_kinase_rmsd_direct_metric.py`` rather than asserted.

2. **THE LEVEL SELECTS WHICH DEPOSITED CHAIN, NOT A SIGN.** ``+1`` means "minimise the RMSD to the
   DFG-out chain" and ``-1`` means "minimise the RMSD to the DFG-in chain", at the SHARED draw seed
   the two levels of one (sequence, cycle) already carry. This is the same reading domain motion
   takes of the same two-state set, and here it has a stronger warrant than there: the two chains
   are LABELLED by Kincore (BLAminus and BBAminus), so unlike the DynDom pair there is no question
   of which conformer a level names. What it removes is the per-protein sign-identifiability
   problem of D-77/D-78 -- there is no sign to invert -- and the incoherent ask, since "be more
   DFG-in when you are already DFG-in" names no structural change while "be closer to this chain"
   either already holds or does not.

3. **PER-SYSTEM NORMALISATION.** ``s_p`` is this kinase's own deposited-chain separation over its
   own metric set and ``tau_p = min(tau, 0.5 * s_p)``. The MSE form keeps ``s_p`` and drops
   ``tau_p``. The scale is what makes the batch mean an average over PROTEINS rather than over
   Angstroms, and on this pool it matters more than it does on DynDom: a kinase A-loop transition
   is a local rearrangement of ~15 residues and the measured separations span a wide range, so
   without the scale the objective's implicit weighting would be "largest A-loop swings first".

4. **NO SOFT-MIN OVER DRAWS.** With 2 levels x 1 draw per scheduled unit there is no min to take
   inside a unit, and a soft-min over two values is nearly a hard min: it would route the whole
   gradient to one level and silently halve the schedule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from sae_bioemu.steering.aligned_rmsd import aligned_rmsd

__all__ = [
    "DEFAULT_METRIC_OBJECTIVE",
    "KINASE_RMSD_PROPERTY_NAME",
    "KINASE_RMSD_TAU_ANGSTROM",
    "METRIC_OBJECTIVE_FORMS",
    "METRIC_OBJECTIVE_HINGE",
    "METRIC_OBJECTIVE_MSE",
    "METRIC_SETS",
    "METRIC_SET_ACTIVATION_LOOP",
    "METRIC_SET_ACTIVATION_LOOP_SHELL8",
    "METRIC_SET_ACTIVATION_LOOP_SHELL12",
    "MINIMUM_SEPARATION_ANGSTROM",
    "REFERENCE_STATES",
    "POOL_GATE_ANGSTROM",
    "HINGE_TRANSITION",
    "KinaseRmsdNormalisation",
    "aligned_backbone_rmsd",
    "huberised_hinge",
    "metric_mse",
    "reference_state_for_level",
    "validated_metric_objective",
    "validated_metric_set",
]

#: ``target.property_name`` for this property. Spelled once, here, and imported by the descriptor
#: and by every test, so the catalog cannot route a name no descriptor answers for.
#:
#: DISTINCT from ``dfg_progress_coordinate`` and deliberately not a variant spelling of it. The two
#: are different quantities (an Angstrom distance against a dimensionless coordinate) measured over
#: different atom sets against different references, and one name would pool them in every journal,
#: every checkpoint identity and every report that reads either.
KINASE_RMSD_PROPERTY_NAME = "kinase_conformer_rmsd"

#: The success threshold this property is REPORTED at, in Angstrom. It is the domain-motion
#: benchmark's 3.0 over N/CA/C/O, copied BY VALUE, and it reaches no part of the default MSE
#: objective -- not its value, not its gradient, and not the digest of the per-system target table
#: a checkpoint is bound to. It is here because the hinge ablation reads it and because a report
#: has to be able to say what "success" was measured against.
#:
#: **Stated honestly: 3.0 Angstrom over a ~15-residue A-loop is a different ask than 3.0 Angstrom
#: over a whole domain.** The metric set here is small and local, so this threshold is harder in
#: shape terms than the same number is on the DynDom pool. It is not re-tuned, because retuning a
#: reported threshold per pool is how two programmes come to report one number meaning two things;
#: ``tau_p`` below takes the harder of this and ConforMix's adaptive ``0.5 * s_p`` per system.
KINASE_RMSD_TAU_ANGSTROM = 3.0

#: Plain MSE on the per-system scale, ``L = 0.5*(m/s_p)^2``. The DEFAULT.
METRIC_OBJECTIVE_MSE = "mse"
#: The one-sided huberised hinge on ``(m - tau_p)/s_p``. The ABLATION, and the only form that reads
#: a ``tau`` at all.
METRIC_OBJECTIVE_HINGE = "huberised_hinge"
#: The two forms, and there are exactly two. An unrecognised spelling is refused by name rather
#: than falling back to either, so a typo cannot silently train the other objective.
METRIC_OBJECTIVE_FORMS: tuple[str, ...] = (METRIC_OBJECTIVE_MSE, METRIC_OBJECTIVE_HINGE)
#: What an arm that names no form descends. See design point 1.
DEFAULT_METRIC_OBJECTIVE = METRIC_OBJECTIVE_MSE

#: The activation loop plus the X-DFG residue, and nothing else. THE SHIPPED SET.
METRIC_SET_ACTIVATION_LOOP = "activation_loop"
#: The same, unioned with every residue within 8 Angstrom of it. A candidate, measured and not
#: shipped.
METRIC_SET_ACTIVATION_LOOP_SHELL8 = "activation_loop_shell8"
#: The same, unioned with the pool's own 12 Angstrom context shell. A candidate, measured and not
#: shipped.
METRIC_SET_ACTIVATION_LOOP_SHELL12 = "activation_loop_shell12"
#: The three candidate index sets the reference builder can measure ``s_p`` under. They are named
#: here, in the fingerprinted objective, rather than left as builder flags: which residues the
#: metric is read over is the single largest decision in this objective, the artifact records which
#: one built it, and an artifact naming an unrecognised set is refused at load.
METRIC_SETS: tuple[str, ...] = (
    METRIC_SET_ACTIVATION_LOOP,
    METRIC_SET_ACTIVATION_LOOP_SHELL8,
    METRIC_SET_ACTIVATION_LOOP_SHELL12,
)

#: The smallest ``s_p`` a system may carry into the arm, in Angstrom. Not "greater than zero", and
#: the distinction is MEASURED rather than cautious: :func:`~sae_bioemu.steering.aligned_rmsd.
#: aligned_rmsd` adds ``1e-12`` inside its square root so the derivative stays finite at coincident
#: inputs, and therefore reports about **1e-6 Angstrom** for two identical structures rather than 0.
#: A ``separation > 0`` gate is satisfied by that floor, and ``s_p`` DIVIDES this objective under
#: both forms: a system whose deposited chains coincide would enter the batch at ``m / 1e-6`` and
#: own every gradient in it.
#:
#: **COPIED BY VALUE from** :data:`sae_bioemu.domain_motion.objective.
#: MINIMUM_SEPARATION_ANGSTROM`, which copied it by value from the cryptic programme, and NOT
#: imported. An import would put ``domain_motion_training_only`` into this property's training
#: composition and bind every kinase-RMSD checkpoint to the DynDom pool's gate, scorer and paired
#: report -- none of which this property executes.
#: ``tests/test_kinase_rmsd_direct_metric.py`` pins the three equal so the copies cannot drift.
MINIMUM_SEPARATION_ANGSTROM = 0.5

#: THE POOL GATE IS :data:`MINIMUM_SEPARATION_ANGSTROM`, AND IT IS NOT
#: :data:`KINASE_RMSD_TAU_ANGSTROM`. THIS IS A MEASURED DEPARTURE FROM DOMAIN MOTION.
#:
#: ``domain_motion/separation.py`` gates its pool at ``s_p > tau = 3.0``, and carrying that number
#: here was the first thing this package did. It is wrong on this pool, for a reason the per-gene
#: table settles rather than argues: measured over the shipped metric set, only **18 of the 43**
#: kinases in ``kinase_pool_L500.json`` have ``s_p > 3.0`` Angstrom (median 2.669, minimum 0.854 at
#: CSNK1E, maximum 10.940 at SLK). A 3.0 gate would discard 25 genes and with them the
#: pre-registered K-12 split every kinase arm in this project is compared on -- and re-deriving a
#: split from a measured separation is a selection on the outcome (D-26), which
#: ``kinase_arm9_margin_workflow.yaml`` already refuses for this pool for the same reason.
#:
#: The domain-motion gate's PREMISE also does not hold here. It exists because a FIXED ``tau`` of
#: 3.0 makes ``success@tau`` trivially satisfiable for a system whose conformers are closer than
#: tau: the draw sitting on conformer i already scores against conformer j. This objective's
#: DEFAULT form has no threshold at all, and its hinge ablation uses
#: :attr:`KinaseRmsdNormalisation.effective_tau_angstrom` ``= min(tau, 0.5 * s_p)``, which for
#: CSNK1E is 0.427 rather than 3.0 -- the adaptive half of that minimum is precisely the mechanism
#: that makes a per-system threshold meaningful on a pool this heterogeneous.
#:
#: So ``s_p <= tau`` is REPORTED AS A STRATUM and never silently dropped: the arm resolver counts
#: it, the reference builder writes it into the artifact's provenance, and the config header states
#: it. What IS gated, strictly, is :data:`MINIMUM_SEPARATION_ANGSTROM`, which is the divisor hazard
#: and a different failure; all 43 genes clear it under all three candidate index sets, the closest
#: at 1.7x.
POOL_GATE_ANGSTROM = MINIMUM_SEPARATION_ANGSTROM

#: Where the hinge turns from quadratic to linear, in units of the system's own separation ``s_p``.
#: This is the parameter that decides how much of a batch's gradient one hopeless system can own,
#: so it is part of the objective's identity rather than a tuning knob hidden in a config.
HINGE_TRANSITION = 1.0

#: Which deposited chain each two-state dose level names. Unlike the DynDom pair, both chains here
#: carry a Kincore LABEL -- the ``-1`` chain is ``DFGin``/``BLAminus`` and the ``+1`` chain is
#: ``DFGout``/``BBAminus`` -- so a level names a named state rather than an arbitrary member of a
#: pair. The spelling matches ``kinase_arm9_margin_workflow.yaml``'s reading of the same two levels
#: ("level +1 IS DFG-out ... -1 IS DFG-in"), so the two kinase properties agree about what a level
#: asks for even though they measure different things.
REFERENCE_STATES: Mapping[float, str] = {1.0: "dfg_out", -1.0: "dfg_in"}


def reference_state_for_level(level: float) -> str:
    """Return ``"dfg_in"`` or ``"dfg_out"``: which deposited chain this level asks for.

    ``0.0`` is refused rather than mapped to a midpoint. A kinase is DFG-in or DFG-out; there is no
    third deposited chain, and "half way between BLAminus and BBAminus" is not a state the protein
    has and not a thing this metric can be measured against.
    """
    if isinstance(level, bool) or not isinstance(level, (int | float)):
        raise ValueError("level must be a real number")
    state = REFERENCE_STATES.get(float(level))
    if state is None:
        raise ValueError(
            "a kinase-RMSD level NAMES A DEPOSITED CHAIN, so it must be exactly -1 (the "
            f"DFGin/BLAminus chain) or +1 (the DFGout/BBAminus chain); received {float(level)}. "
            "The metric is an unsigned distance to a deposited structure and there is no third "
            "structure for an intermediate level to be measured against"
        )
    return state


def validated_metric_objective(value: object) -> str:
    """Return one of :data:`METRIC_OBJECTIVE_FORMS`, or fail closed naming both.

    One validator, imported by the artifact loader, by the loss and by the property descriptor, so
    the form cannot be accepted under one spelling on disk and dispatched under another. ``None``
    resolves to :data:`DEFAULT_METRIC_OBJECTIVE`.
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


def validated_metric_set(value: object) -> str:
    """Return one of :data:`METRIC_SETS`, or fail closed naming all three.

    There is deliberately NO DEFAULT. Which residues the metric is read over is this objective's
    largest single decision and it changes every ``s_p`` and every measured distance, so an
    artifact that does not say which set built it is refused rather than assumed to be the shipped
    one -- an artifact written under the shell and read as the A-loop would produce finite,
    plausible numbers measured over ~110 residues that barely move.
    """
    if not isinstance(value, str) or value not in METRIC_SETS:
        raise ValueError(
            f"metric_set must be one of {list(METRIC_SETS)}, received {value!r}. It names WHICH "
            "RESIDUES the metric and its per-system normalisation s_p are read over, so there is "
            "no default and no fallback: the shell sets dilute the activation loop to ~12% of the "
            "union and would be read as the shipped set at every other check green"
        )
    return value


def metric_mse(metric_angstrom: Tensor, separation_angstrom: Tensor | float) -> Tensor:
    """Return :data:`METRIC_OBJECTIVE_MSE`: ``L = 0.5 * (m / s_p)^2``, elementwise over a batch.

    **The DEFAULT form, and deliberately the special case of the hinge rather than a second
    implementation of anything.** :func:`huberised_hinge` is ``0.5*z^2`` for ``0 < z <= transition``
    and ``0`` at ``z <= 0``; with ``tau = 0`` the argument is ``m/s_p``, which is nonnegative
    because an RMSD is, and with the knee taken to infinity the quadratic branch is the whole
    function.
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
            "separation_angstrom is PER OBSERVATION -- each kinase's own chain separation -- so it "
            f"must share the metric's shape; got {tuple(separation.shape)} and "
            f"{tuple(metric_angstrom.shape)}"
        )
    if bool(torch.any(separation <= 0)):
        raise ValueError(
            "separation_angstrom must be strictly positive for every observation: it is the "
            "DIVISOR the objective is measured in, and a kinase whose two deposited chains "
            "coincide over the metric set has no transition to steer along (see "
            f"MINIMUM_SEPARATION_ANGSTROM={MINIMUM_SEPARATION_ANGSTROM})"
        )
    scaled = metric_angstrom / separation
    return 0.5 * scaled * scaled


def huberised_hinge(value: Tensor, *, transition: float = HINGE_TRANSITION) -> Tensor:
    """Return the one-sided Huber of ``value``: 0 at or below zero, then quadratic, then linear.

        L(z) = 0                                   z <= 0
             = 0.5 * z^2                           0 < z <= transition
             = transition * (z - 0.5*transition)   z > transition

    Continuous in value and in first derivative at both breakpoints. The flat half is what stops
    the objective polishing draws that already succeed; the linear half is what stops one far draw
    owning the batch.

    Written with ``clamp`` rather than ``where`` so the zero region's gradient is exactly zero
    rather than a masked NaN, and so no ``sqrt`` is differentiated at the origin.
    """
    if not isinstance(value, Tensor):
        raise TypeError("value must be a torch.Tensor")
    if not isinstance(transition, float) or not transition > 0.0:
        raise ValueError("transition must be a positive float")
    positive = torch.clamp(value, min=0.0)
    quadratic = torch.clamp(positive, max=transition)
    return 0.5 * quadratic * quadratic + transition * (positive - quadratic)


def aligned_backbone_rmsd(draw: Tensor, reference: Tensor) -> Tensor:
    """Return the superposed N/CA/C/O RMSD of one draw against one deposited chain.

    ``draw`` and ``reference`` are ``[K, 4, 3]`` in Angstrom, in :data:`.frames.BACKBONE_ATOMS`
    order, and both are ALREADY RESTRICTED to this system's metric set.

    **There is no ``metric_residues`` argument, and its absence is the point.** The domain-motion
    twin takes a selector and indexes both operands with it, which makes "the alignment set" and
    "the metric set" two uses of one variable that a caller could pass differently. Here the
    artifact stores the two chains PRE-RESTRICTED against one ``metric_indices`` list, so the two
    sets are the same set structurally and there is no argument through which they could differ.
    The draw is restricted by the same list at the one call site that reads it
    (:func:`~sae_bioemu.kinase_rmsd.artifact.kinase_rmsd_metric_batch`).

    The superposition is solved on exactly the atoms the metric is read over --
    :func:`~sae_bioemu.steering.aligned_rmsd.aligned_rmsd` refuses any other arrangement by name --
    which is what makes the detached rotation exact by the envelope theorem and the gradient pure
    shape change. Report :func:`~sae_bioemu.steering.aligned_rmsd.rigid_body_leak` beside the loss.
    """
    if draw.ndim != 3 or draw.shape[-1] != 3 or draw.shape[-2] != 4:
        raise ValueError("draw must be [metric_residues, 4, 3] N/CA/C/O in Angstrom")
    if reference.shape != draw.shape:
        raise ValueError(
            f"reference must share the draw's shape; got {tuple(reference.shape)} and "
            f"{tuple(draw.shape)}. The metric is defined between corresponding atoms, and both "
            "operands are already restricted to this system's metric set"
        )
    if int(draw.shape[0]) < 3:
        raise ValueError(
            f"{int(draw.shape[0])} metric residues is too few to determine a superposition; a "
            "kinase whose activation loop is resolved in neither deposited chain belongs out of "
            "the pool rather than in it with a degenerate measurement"
        )
    # Flattened to (4K, 3): the metric is an RMSD over ATOMS, so all four of a residue's atoms
    # enter the mean and the superposition individually.
    return aligned_rmsd(draw.reshape(-1, 3), reference.reshape(-1, 3).to(draw))


@dataclass(frozen=True)
class KinaseRmsdNormalisation:
    """One kinase's own scale and threshold: ``s_p`` and ``tau_p``. Design point 3.

    Built once per system from the two deposited chains and carried in the arm identity, so a
    report can say what weight each protein's draws carried rather than inferring it.
    """

    sequence_id: str
    #: ``s_p``: the superposed N/CA/C/O RMSD between the DFG-in and DFG-out chains over this
    #: system's metric set, in Angstrom.
    separation_angstrom: float
    #: The fixed threshold this system was gated and normalised against.
    tau_angstrom: float = KINASE_RMSD_TAU_ANGSTROM

    def __post_init__(self) -> None:
        if not self.sequence_id:
            raise ValueError("a kinase normalisation must name its sequence")
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
        """``tau_p = min(tau, 0.5 * s_p)``: whichever of the two thresholds is HARDER."""
        return min(float(self.tau_angstrom), self.adaptive_tau_angstrom)

    def hinge(self, metric_angstrom: Tensor) -> Tensor:
        """Return ``L`` for one or more measured RMSDs of THIS system, under the hinge form."""
        if not isinstance(metric_angstrom, Tensor):
            raise TypeError("metric_angstrom must be a torch.Tensor")
        offset = metric_angstrom - self.effective_tau_angstrom
        return huberised_hinge(offset / self.separation_angstrom)

    def mse(self, metric_angstrom: Tensor) -> Tensor:
        """Return ``L = 0.5*(m/s_p)^2`` for one or more measured RMSDs of THIS system."""
        return metric_mse(metric_angstrom, float(self.separation_angstrom))

    def priced(self, metric_angstrom: Tensor, metric_objective: str) -> Tensor:
        """Return the priced value under the named form. One dispatcher, so the journal row and the
        quantity descended cannot come from different arithmetic."""
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

        * :data:`METRIC_OBJECTIVE_HINGE` -- ``tau_p``. The prediction is ``max(m, tau_p)`` and
          ``prediction - target`` is ``max(m - tau_p, 0)``.
        * :data:`METRIC_OBJECTIVE_MSE` -- ``s_p``. The MSE has no threshold, so its zero is
          ``m = 0``, the prediction is ``m + s_p`` and ``prediction - target`` is exactly ``m``.
          ``s_p`` is used rather than ``tau_p`` precisely so ``tau`` reaches no part of the MSE
          objective -- neither its value, nor its gradient, nor the digest of the system-target
          table the checkpoint is bound to.

        A shifted prediction is the price of not editing ``steering/schedule.py``, which is in
        ``generation_shared`` and whose blast radius is every steering checkpoint there has ever
        been, to admit a zero target. It is not hidden: ``KinaseRmsdSteeringLoss.forward``
        re-derives ``prediction - target`` against the measured ``m`` on the wire and refuses a
        disagreement, elementwise, on every batch.
        """
        form = validated_metric_objective(metric_objective)
        if form == METRIC_OBJECTIVE_MSE:
            return float(self.separation_angstrom)
        return self.effective_tau_angstrom

    def prediction(self, metric_angstrom: Tensor, metric_objective: str) -> Tensor:
        """Return the shared term's ``prediction`` for THIS system under the named form."""
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


# The pool gate IS the divisor floor on this property, and the reported threshold is strictly
# larger than it. Asserted at import rather than left as a comment, because the failure this
# ordering prevents is a pool gated ABOVE its own reported threshold -- which would mean every
# admitted kinase was already inside tau before the adapter ran, the defect the domain-motion gate
# exists to prevent and which on this pool is avoided by the tau_p minimum instead.
assert POOL_GATE_ANGSTROM == MINIMUM_SEPARATION_ANGSTROM < KINASE_RMSD_TAU_ANGSTROM
