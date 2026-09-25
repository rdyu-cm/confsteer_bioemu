"""The domain-motion steerable property: what ``domain_motion_backbone_rmsd`` declares.

SCOPE: DOMAIN MOTIONS. Everything here is a fact about this property -- the dose set, how a level
is read, the objective class, the weights it prices nowhere, and what its arm resolution requires.
Not one of them is in a shared file, which is the whole point of
:mod:`sae_bioemu.steering.property_registry`: adding this property edits this package and the
unfingerprinted :mod:`sae_bioemu.steering.property_catalog`, and moves no existing checkpoint.

THE THREE SHARED SEAMS THIS PROPERTY USED TO BE BLOCKED ON, AND WHAT REPLACED THEM
-----------------------------------------------------------------------------------

Until the shared seam pass, this module raised :class:`NotImplementedError` from
:func:`_sample_definition_arguments` and named three things it could not do for itself, each in a
file whose fingerprint group is shared with properties that price none of this. They are worth
keeping in view because each was closed by a HOOK rather than by a branch, which is what makes a
fifth property cost a descriptor:

1. ``bioemu/steering_properties.py`` (``generation_shared``, hashed by EVERY property) dispatched
   on ``adapter.property_name`` through a chain of branches and took one keyword per property. It
   now resolves the steered property's ``measure_sample`` hook from the registry and passes ONE
   ``definition`` keyword, so :func:`_measure_sample` below is this property's whole generation
   path and lives in this package.
2. ``bioemu/steering_workflow.py`` (``training_only``) read ``arm.artifact.atom_set`` -- two levels
   into an object shape only the progress coordinates have, so the first domain-motion run would
   have raised ``AttributeError`` inside the reference cache after loading the sampler. The
   control is now described by :class:`~sae_bioemu.steering.property_registry.ControlMeasurement`,
   and this property declares ``required=True``: a C-alpha control for EVERY draw, whatever
   ``lambda_l`` is. The metric is a distance to a DEPOSITED conformer, so the control is not the
   METRIC's anchor -- it is the PRIMARY's, because ``gap_closed.py`` grades the intervention as the
   fraction of each system's own gap closed against a matched unsteered draw at the same seed. See
   :func:`_control_measurement`: generated, never priced, and roughly doubling generation.
3. ``steering/ordinary_workflow.py`` (``training_only``) knew three target constructions and all
   three VARY WITH THE LEVEL. :data:`~sae_bioemu.steering.schedule.TARGET_ORIGIN_SYSTEM_CONSTANT`
   is the fourth: ``t = tau_p`` at every level, with the level selecting which deposited conformer
   the distance is measured to (objective design point 2). It is a GENERIC origin, not a
   domain-motion one -- the cryptic direct-metric term is queued for the same construction.

What remains genuinely absent is a HELD-OUT SCORER. ``bioemu-benchmarks``' success@B over N/CA/C/O
is the metric, and the registry refuses ``evaluate`` by name for a property with none rather than
routing it to another property's, which is the correct failure until one exists. Training,
validation and journalling run.

The arm resolution loads the artifact, re-measures every ``s_p``, applies the separation gate to
the scheduled population and cross-checks coverage. Those gates are the part a wrong configuration
fails on, and they are exercised by tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sae_bioemu.domain_motion.artifact import load_domain_motion_artifact
from sae_bioemu.domain_motion.objective import (
    DIRECT_METRIC_TAU_ANGSTROM,
    DOMAIN_MOTION_PROPERTY_NAME,
    HINGE_TRANSITION,
)
from sae_bioemu.domain_motion.separation import (
    SEPARATION_GATE_ANGSTROM,
    apply_separation_gate,
)
from sae_bioemu.steering.property_registry import (
    ControlMeasurement,
    JournalledSampleColumn,
    SteerableProperty,
    ValidationSelectionMetric,
    resolve_config_path,
)
from sae_bioemu.steering.schedule import (
    TARGET_ORIGIN_SYSTEM_CONSTANT,
    TWO_STATE_DOSE_LEVELS,
    SystemTargetConstant,
    SystemTargetConstants,
)

__all__ = [
    "DIRECT_METRIC_TARGET_ORIGIN",
    "DOMAIN_MOTION_ARM_CONFIG_KEY",
    "DOMAIN_MOTION_PROPERTY",
    "DOMAIN_MOTION_PROPERTY_NAME",
    "resolve_domain_motion_property_arm",
]

#: The ``target`` block key naming this property's arm configuration. Its own key, not a shared
#: ``arm``: the registry refuses any property carrying another's key, and one shared spelling would
#: let a cryptic arm be handed to a domain-motion run with both configs green.
DOMAIN_MOTION_ARM_CONFIG_KEY = "domain_motion_arm"

#: How this property's target is built: ``t = tau_p`` for every level of a system, with the LEVEL
#: selecting which deposited conformer the distance is measured to.
#:
#: It is the shared :data:`~sae_bioemu.steering.schedule.TARGET_ORIGIN_SYSTEM_CONSTANT`, aliased
#: here so that this package names the construction it uses and every test that pinned this
#: spelling keeps pinning a real thing. It was a domain-motion-only string, deliberately absent
#: from ``RELATIVE_TARGET_ORIGINS`` so that a configuration trying to run this property failed
#: closed at run start; the origin now exists, and the alias is the record of where it came from.
#:
#: The shared name is GENERIC and not ``system_direct_metric_threshold`` for a reason that is not
#: cosmetic: the cryptic direct-metric term is the second property queued for this construction,
#: and an origin named after the first one to need it is how a shared mechanism comes to look like
#: one programme's private arrangement.
DIRECT_METRIC_TARGET_ORIGIN = TARGET_ORIGIN_SYSTEM_CONSTANT

#: The HELD-OUT reading of the metric, in the Angstrom the benchmark grades. Gated on nothing but
#: this property running, because unlike the cryptic direct-metric term it is not an optional
#: addition: the metric IS this objective, so a domain-motion validation row without it would be a
#: row that says nothing about what the arm was for.
_DIRECT_METRIC_COLUMN = "validation_direct_metric_angstrom"

#: What a domain-motion arm SELECTS its shipped epoch on: the mean held-out deviation, in Angstrom.
#:
#: Not ``target_mae_sigma``, and the difference is the whole argument for a direct metric. Under
#: this objective ``target`` is ``tau_p`` and ``achieved`` is ``max(m, tau_p)``, so the target error
#: is exactly ``max(m - tau_p, 0) / s_p`` -- the HINGE. It is identically zero for every draw that
#: already succeeds, so selecting on it cannot tell an epoch where every system sits at 2.9
#: Angstrom from one where every system sits at 0.4, and it cannot tell an epoch that fixed one
#: hopeless system from one that broke two easy ones back to exactly tau. The unclamped mean
#: deviation can.
_DIRECT_METRIC_SELECTION_NAME = "direct_metric_angstrom_mean"

#: Weights this objective prices NOWHERE, by key and with the reason. A nonzero value for one of
#: these is refused by name before a GPU is allocated, rather than read from the config, dropped by
#: the loss constructor and absent from the objective identity -- the shape that voided six
#: radius-of-gyration arms.
_REFUSED: Mapping[str, str] = {
    "lambda_perp": (
        "the off-path term prices the component of a realised motion perpendicular to a "
        "progress coordinate's transition vector. A direct metric has no coordinate and no "
        "transition vector: every direction that raises the RMSD is already priced by the RMSD"
    ),
    "lambda_perp_fraction": (
        "the per-protein form of lambda_perp, and refused for the same reason: there is no the progress coordinate "
        "whose curvature D-31's bound is about"
    ),
    "lambda_rigid": (
        "the context-rigidity hinge is measured against a pocket's context shell and each "
        "protein's ambient rms_C from stage 1. A domain motion has no pocket, no context shell "
        "and no stage-1 setpoint table"
    ),
    "lambda_dihedral": (
        "the Kincore cosine dihedral distance is measured to a DFG cluster centroid over the "
        "activation-loop motif; this objective has no motif and no centroid table"
    ),
}

# ``lambda_c`` is deliberately NOT in the mapping above, even though this objective prices no
# paired dose term. ``refused_runtime_keys`` exists for the EXTENDED vocabulary -- weights that a
# loss constructor would accept and silently drop -- and the five shared weights are priced by
# every objective, which ``test_every_property_either_prices_or_refuses_every_declared_weight``
# holds as an invariant. ``lambda_c`` is refused instead where it cannot be dropped silently: by
# ``DomainMotionSteeringLoss.__init__``, which raises with the reason (the two levels measure
# distances to DIFFERENT reference conformers, so their gap is not a dose gap). That refusal also
# runs before anything generates.


def _loss_class() -> type:
    from sae_bioemu.domain_motion.loss import DomainMotionSteeringLoss

    return DomainMotionSteeringLoss


def _objective_arguments(*, runtime: Mapping[str, Any], arm: Any) -> Mapping[str, Any]:
    """Return the constructor arguments beyond the shared ones: the objective's FORM.

    The direct metric IS the shared target term (see :mod:`sae_bioemu.domain_motion.loss`), so this
    objective adds no WEIGHT to the shared vocabulary. What it does add is which of
    :data:`~sae_bioemu.domain_motion.objective.METRIC_OBJECTIVE_FORMS` the term is, and that comes
    off the resolved ARM rather than out of ``runtime``: the same declaration also decides the
    per-system target constant :func:`system_target_constants` builds and the prediction
    :func:`_measure_sample` measures, and those two are built from the arm in places that never see
    the workflow's objective block. One source for all three is what stops them disagreeing.

    ``property_loss`` is deliberately NOT forwarded. The workflow's own ``property_loss`` key still
    reaches the loss through the shared arguments, and ``DomainMotionSteeringLoss`` refuses the one
    that does not match the form -- so a config naming ``huber`` against an MSE artifact fails by
    name instead of being overridden into silence.

    ``lambda_m`` IS forwarded, and it is the weight on the graded metric (D-87). It comes out of
    ``runtime`` rather than off the arm because it is a weight like every other weight and belongs
    in the ``bioemu_pytorch`` block beside them; what comes off the arm is the FORM, which is a
    statement about the artifact. Absent means ``0.0``, which the loss then refuses by name as a
    driverless objective rather than accepting and training nothing.
    """
    arguments: dict[str, Any] = {"lambda_m": float(runtime.get("lambda_m", 0.0) or 0.0)}
    if arm is not None:
        arguments["metric_objective"] = arm.metric_objective
    return arguments


def resolve_domain_motion_property_arm(
    *, payload: Mapping[str, Any], base: Path, split: Any, runtime: Mapping[str, Any]
) -> Any:
    """Resolve the arm named by ``target.domain_motion_arm``: the reference artifact, gated.

    The gates run in order, cheapest failure first, and all of them run BEFORE the sampler loads:

    1. **Presence.** ``domain_motion_backbone_rmsd`` requires the block: the metric is a distance
       to THIS system's two deposited conformers and nothing in ``bioemu_pytorch`` supplies them.
    2. **The artifact loads and re-measures.** :func:`~sae_bioemu.domain_motion.artifact.
       load_domain_motion_artifact` re-derives every stored ``s_p`` from that entry's own
       coordinates and refuses a disagreement, so a stale per-protein weight cannot enter a run.
    3. **The separation gate.** Every scheduled sequence must be a system the gate KEPT. A system
       whose conformers sit closer than tau is one the metric cannot resolve -- a draw on
       conformer i scores against conformer j -- so training on it teaches the adapter to satisfy
       a threshold that was never in question. Refused here rather than filtered silently: a run
       that quietly trained on a different population than its split record names has passed every
       other gate about the wrong set of proteins.
    4. **The population is the same.** The artifact must cover every sequence the workflow's split
       walks, train, validation and test alike.

    The resolved arm IS the artifact: it carries the references, the metric sets, the per-system
    normalisations and its own identity, and there is no second object for it to hide behind.
    """
    target = payload.get("target")
    configured = target.get(DOMAIN_MOTION_ARM_CONFIG_KEY) if isinstance(target, dict) else None
    if configured is None:
        raise ValueError(
            f"{DOMAIN_MOTION_PROPERTY_NAME} requires target.{DOMAIN_MOTION_ARM_CONFIG_KEY} naming "
            "the reference artifact: the metric is a superposed N/CA/C/O RMSD to this system's "
            "own two deposited conformers, and nothing in bioemu_pytorch supplies either of them"
        )
    artifact = load_domain_motion_artifact(
        resolve_config_path(base, configured, f"target.{DOMAIN_MOTION_ARM_CONFIG_KEY}")
    )
    scheduled = sorted({*split.train, *split.validation, *split.test})
    missing = [name for name in scheduled if name not in artifact.entries]
    if missing:
        raise ValueError(
            f"{len(missing)} scheduled sequences are absent from the reference artifact "
            f"{artifact.path}; first: {missing[:5]}. The direct metric has no fallback reference"
        )
    gate = apply_separation_gate(
        {name: artifact.entries[name].separation_angstrom for name in scheduled},
        threshold=SEPARATION_GATE_ANGSTROM,
    )
    if gate.dropped:
        raise ValueError(
            f"{len(gate.dropped)} scheduled sequences have their two deposited conformers closer "
            f"than {SEPARATION_GATE_ANGSTROM} Angstrom under the true N/CA/C/O metric; first: "
            f"{list(gate.dropped)[:5]}. The success metric cannot resolve those motions -- a draw "
            "sitting on conformer i already scores against conformer j -- so the split record "
            "must be built from the gated pool rather than gated at runtime"
        )
    del runtime
    return artifact


def _adapter_arguments(arm: Any) -> Mapping[str, Any]:
    """No per-residue conditioning channel.

    There is deliberately no domain-motion equivalent of the cryptic site field. A conditioning
    channel built from the deposited conformers would tell the adapter WHERE the hinge is, and the
    direct metric would then be satisfied partly by a hand-supplied answer; whether the adapter can
    find the motion is the question. If one is ever wanted it is a new arm, declared here, and not
    a default.
    """
    del arm
    return {}


def _sample_definition_arguments(arm: Any) -> Mapping[str, Any]:
    """Hand the resolved artifact over under the SHARED ``definition`` keyword.

    It used to raise :class:`NotImplementedError` here, naming the three shared seams this
    property needed and could not make for itself. All three have landed:

    1. ``generate_exact_property_sample`` dispatches through the registry's ``measure_sample``
       hook instead of a branch chain, and takes ONE ``definition`` keyword instead of one per
       property, so :func:`_measure_sample` below is this property's whole generation path.
    2. The matched control is described by
       :class:`~sae_bioemu.steering.property_registry.ControlMeasurement` rather than by
       ``arm.artifact.atom_set``, which this artifact does not have. This property declares
       ``ControlMeasurement()`` -- C-alpha, and only where ``lambda_l`` prices it -- because the
       metric is a distance to a DEPOSITED conformer, not to the control; the control's role here
       is the local-structure term's, which is a C-alpha quantity.
    3. :data:`~sae_bioemu.steering.schedule.TARGET_ORIGIN_SYSTEM_CONSTANT` is a legal relative
       origin with a target builder, a per-system table on ``OrdinarySteeringConfig`` and its own
       checkpoint-identity block.
    """
    from sae_bioemu.bioemu.steering_properties import BioEmuDomainMotionDefinition

    return {} if arm is None else {"definition": BioEmuDomainMotionDefinition(artifact=arm)}


def _measure_sample(*, batch: Any) -> Any:
    """Measure the superposed N/CA/C/O RMSD of each steered draw, on its system's own origin.

    The whole objective, in the shape the shared ``SteeringLoss`` prices (see
    :mod:`sae_bioemu.domain_motion.loss`). Under the DEFAULT
    :data:`~sae_bioemu.domain_motion.objective.METRIC_OBJECTIVE_MSE`:

        ``prediction = m + s_p``         the measured RMSD, placed on this system's own origin
        ``target     = s_p``             supplied by ``_unit_target`` from the system table
        ``scale      = s_p``             supplied by ``_unit_sigma`` from the same table

    so ``mean_n ((prediction - target)/scale)^2`` IS ``mean_n (m_n/s_p)^2``, with ``tau`` nowhere in
    it. Under the hinge ablation the prediction is ``max(m, tau_p)`` against ``target = tau_p`` and
    the term is ``mean_n huberised_hinge((m_n - tau_p)/s_p)``. **The clamp is the hinge's
    one-sidedness and the shift is the MSE's zero**; both are re-checked elementwise by
    ``DomainMotionSteeringLoss.forward`` against the unclamped ``m`` on the wire, rather than
    trusted from this line, because both failures produce a plausible loss curve and no other
    symptom.

    The form is read off the ARTIFACT, which is this property's arm: one declaration drives the
    prediction here, the target table in :func:`system_target_constants` and the arithmetic in the
    loss.

    The UNCLAMPED ``m`` travels beside it on ``metric_deviation_angstrom``, journalled as
    ``direct_metric_angstrom``: the hinge is identically zero for every successful draw, so an arm
    whose only record is the hinge is unreadable exactly where it is working.

    The level SELECTS THE REFERENCE and is read off the batch rather than inferred, because a
    distance to the wrong conformer is a NUMBER and not an error.
    """
    from sae_bioemu.bioemu.steering_properties import BioEmuDomainMotionDefinition
    from sae_bioemu.domain_motion.artifact import direct_metric_batch
    from sae_bioemu.domain_motion.frames import batched_backbone_atoms
    from sae_bioemu.steering.property_registry import PropertyMeasurement

    batch.require_relative_levels(DOMAIN_MOTION_PROPERTY_NAME)
    definition = batch.definition
    if not isinstance(definition, BioEmuDomainMotionDefinition):
        raise ValueError(
            f"{DOMAIN_MOTION_PROPERTY_NAME} needs its resolved reference artifact: the metric is "
            "a superposed N/CA/C/O RMSD to this system's own two deposited conformers, and "
            "neither an adapter nor a structure supplies them"
        )
    artifact = definition.artifact
    sequence_ids = batch.sequence_ids
    levels = batch.levels
    # N, CA, C and O off the denoiser's per-residue frames, with the gradient intact. NOT through
    # ``bioemu/cryptic_generation.py``: that file is the sole member of the code group both
    # progress-coordinate properties compose, so extending it to return O would re-key every
    # cryptic and kinase checkpoint for a change made on behalf of a property neither executes.
    backbone = batched_backbone_atoms(
        batch.steered.coordinates,
        batch.node_orientations,
        [item.sequence for item in batch.records],
    )
    metric, _priced, _separation = direct_metric_batch(artifact, sequence_ids, levels, backbone)
    form = artifact.metric_objective
    # The per-system origin, elementwise. Built as a TENSOR of per-member constants rather than one
    # scalar because one protein's origin applied to another's draw is a finite, plausible and
    # wrong loss -- and ``SystemNormalisation.prediction`` is the one place the shift is defined,
    # so this path and the table in ``system_target_constants`` cannot disagree about it.
    constants = metric.new_tensor(
        [
            artifact.require(name).normalisation.target_constant_angstrom(form)
            for name in sequence_ids
        ]
    )
    return PropertyMeasurement(
        prediction=_prediction_on_origin(metric, constants, form),
        target=batch.target,
        sample_fields={"metric_deviation_angstrom": metric},
    )


def _prediction_on_origin(metric: Any, constants: Any, metric_objective: str) -> Any:
    """Return the batched ``prediction``: ``m + origin`` under the MSE, ``max(m, tau_p)`` under
    the hinge. The batched twin of
    :meth:`~sae_bioemu.domain_motion.objective.SystemNormalisation.prediction`, and it dispatches on
    the same validated form so the two cannot come to mean different things."""
    import torch

    from sae_bioemu.domain_motion.objective import (
        METRIC_OBJECTIVE_MSE,
        validated_metric_objective,
    )

    form = validated_metric_objective(metric_objective)
    if form == METRIC_OBJECTIVE_MSE:
        return metric + constants
    return torch.maximum(metric, constants)


def _checkpoint_arm_arguments(arm: Any) -> Mapping[str, Any]:
    """Bind the resolved artifact into the CHECKPOINT and nothing else (D-23).

    Under the GENERIC ``property_arm_identity`` field rather than a domain-motion one, so that
    binding an arm costs a descriptor line and no edit to the shared config. Deliberately NOT the
    cryptic field: that one is also what ``steers_cryptic_pocket`` is derived from, and a run
    binding its arm there would acquire the four D-35b pocket monitor columns on every validation
    row, with no pocket to measure them in.

    Deliberately not in ``schedule_seed_fingerprint``: routing an artifact digest through the seed
    would redraw every schedule and confound each cross-arm difference with a complete
    re-randomisation. The identity carries the per-system ``s_p`` and ``tau_p``, because those are
    per-protein weights on the objective and a checkpoint must be bound to the weights it descended.
    """
    if arm is None:
        return {}
    return {
        "property_arm_identity": arm.identity(),
        # The per-system ``{tau_p, s_p}`` table the shared target construction reads. It rides
        # this hook -- documented as "extra ``OrdinarySteeringConfig`` keyword arguments the
        # resolved arm supplies" -- rather than a hook of its own, because that is exactly what it
        # is: a config field whose value only the resolved arm can produce. Building it here also
        # means the numbers the objective descends and the numbers the checkpoint identity records
        # come from ONE call on ONE artifact.
        "system_target_constants": system_target_constants(arm),
    }


def _validation_row_columns(*, objective: Any, config: Any) -> Mapping[str, JournalledSampleColumn]:
    """The held-out deviation, in Angstrom, on every domain-motion validation row.

    Ungated, because for this property the metric is not an added term whose weight might be zero:
    it IS the objective, by construction of ``prediction``, ``target`` and ``property_scale``, and
    ``DomainMotionSteeringLoss`` refuses ``lambda_m = 0`` as a driverless loss before any draw.

    It was formerly argued as "the only weight that could gate it is ``lambda_y``, which is SHARED
    and positive on every objective". That is no longer true and no longer needed (D-87): the
    weight is ``lambda_m``, which only this objective and the cryptic one write, so the TRAINING
    column gates on the weight itself. This held-out column stays ungated because a validation draw
    computes no loss at all.
    """
    del objective, config
    return {_DIRECT_METRIC_COLUMN: JournalledSampleColumn(field="metric_deviation_angstrom")}


def _validation_selection_metric(*, objective: Any, arm: Any) -> ValidationSelectionMetric | None:
    """Always the mean held-out deviation. See :data:`_DIRECT_METRIC_SELECTION_NAME`.

    Unconditional rather than gated on a weight, for the reason :func:`_validation_row_columns` is:
    there is no configuration of this property under which the target error is the better
    selection signal, because the target error IS the hinge and the hinge is flat over the whole
    region an arm is trying to reach.
    """
    del objective, arm
    return ValidationSelectionMetric(
        name=_DIRECT_METRIC_SELECTION_NAME, row_column=_DIRECT_METRIC_COLUMN
    )


def system_target_constants(artifact: Any) -> SystemTargetConstants:
    """Build the per-system ``{tau_p, s_p}`` table this property's targets are made of.

    The bridge between the resolved artifact and
    :attr:`~sae_bioemu.steering.ordinary_workflow.OrdinarySteeringConfig.system_target_constants`,
    and it lives in this package because both numbers are this property's own definitions:

    * ``target`` is the FORM's own origin, from
      :meth:`~sae_bioemu.domain_motion.objective.SystemNormalisation.target_constant_angstrom`.
      Under the hinge it is ``tau_p = min(tau, 0.5 * s_p)`` -- whichever of the fixed benchmark
      threshold and ConforMix's adaptive one is HARDER, so the objective is never satisfied by a
      draw the reported metric would fail. Under the default MSE it is ``s_p``, a pure origin the
      prediction carries too and which therefore cancels exactly, chosen so that ``tau`` does not
      enter the MSE objective even through this table's digest; and
    * ``scale`` is ``s_p``, the system's own superposed N/CA/C/O separation between its two
      deposited conformers, measured over the shared metric set.

    Both are re-derived from the artifact's own coordinates at load, so this table cannot drift
    away from the structures the metric is measured against. The table's digest reaches the
    checkpoint identity, so the two forms give two tables and a run cannot resume across them.
    """
    form = artifact.metric_objective
    return SystemTargetConstants(
        entries={
            name: SystemTargetConstant(
                target=entry.normalisation.target_constant_angstrom(form),
                scale=float(entry.separation_angstrom),
            )
            for name, entry in artifact.entries.items()
        }
    )


def _control_measurement(arm: Any) -> ControlMeasurement:
    """The matched unsteered control, on request: generate it; price nothing with it.

    ``required=True`` (when the arm asks for it) and no ``atom_set``, so a C-alpha control is
    generated for every draw whatever ``lambda_l`` is. It replaces the registry's shared default
    (``ControlMeasurement()``), under which ``bioemu/steering_workflow.py``'s
    ``generates_reference = wants_local_structure or control.required`` produced a control ONLY at
    ``lambda_l > 0`` -- which is why every row of ``dm_lg20``'s ``steered_draws.jsonl`` carries
    ``"control_present": false``.

    WHY IT IS REQUIRED NOW, AND WHY NOT THROUGH ``lambda_l``
    --------------------------------------------------------

    The domain-motion PRIMARY is :mod:`sae_bioemu.domain_motion.gap_closed`: per draw,
    ``(m_control - m_steered) / max(m_control, floor)`` against an unsteered control generated from
    THE SAME ``draw_seed``. That metric refuses rather than falling back when a control is absent,
    so without this declaration an arm produces a checkpoint its own primary cannot read.

    Raising ``lambda_l`` would also generate the control and is the WRONG way to get it:
    ``lambda_l`` prices the local-structure term, so it changes the OBJECTIVE and the resulting
    arm is no longer comparable to ``dm_lg20`` on the axis the programme is about. What is wanted
    is generation, not pricing. This is D-86's shape -- an UNPRICED instrument, declared where the
    property is declared, with the loss untouched -- and measurement and pricing are separable
    exactly here.

    **NOTHING IS PRICED.** No term is added, no weight changes, and ``lambda_l`` stays 0. At
    ``lambda_l = 0`` the shared ``SteeringLoss`` adds ``0.0 * local_structure`` to a total it
    otherwise computes identically, which is bit-identical for finite values;
    ``tests/test_domain_motion_baseline_gate.py`` pins that rather than asserting it. The
    smooth-LDDT value is journalled as a free diagnostic, which is a column gained and not a term.

    THE PAIRING IS REAL, NOT NOMINAL. The workflow caches one control per
    ``(sequence_id, unit.generation_seed)`` and :class:`
    ~sae_bioemu.steering.training_structures.TrainingStructureStore` keys the persisted control on
    that same ``generation_seed`` while keying the steered draw on ``unit.seed``. That is exactly
    the correspondence ``gap_closed.paired_draws_from_store`` reads back through the sidecar's
    ``draw_seed``, so this declaration matches the existing contract instead of introducing a
    second one.

    **IT ROUGHLY DOUBLES TRAINING GENERATION**, and that is the accepted cost rather than a
    regression to diagnose later. It is somewhat better than 2x in practice: the control is cached
    per ``(sequence, draw_seed)`` and this property's two levels SHARE a draw seed, so one control
    serves both levels of a (sequence, cycle).

    ``frames`` stays ``False``. Frames would be read off the control for a reconstruction nothing
    on this path performs: the metric is a distance to a DEPOSITED conformer measured off the
    STEERED draw's own frames, and ``gap_closed`` measures the control through placed backbone
    atoms read from its persisted PDB.

    OPTED IN PER ARTIFACT, and only per artifact (D-101). An artifact carrying
    ``control_required = True`` turns the generation on; anything without the attribute gets
    ``False``. THE DEFAULT IS OFF, matching ``kinase_rmsd`` and ``cryptic_rmsd``, which was a
    behaviour change for every arm of this property: the "somewhat better than 2x" saving above
    exists only because a level-conditioned arm visits one ``draw_seed`` at both levels. A
    per-direction unconditioned arm walks ONE level, so each unit visits its draw seed once, the
    ratio goes from one control per two steered draws to one per one, and the control is a full
    extra generation per unit -- the one-GPU unconditioned arm measured 8.9 s/unit against the
    parent's 3.8, with the control a substantial share. It can be declined here because under
    ``TARGET_ORIGIN_SYSTEM_CONSTANT`` the control is not the origin the target is measured from;
    the the progress coordinate properties, where the control IS the origin, keep theirs.

    WHAT THAT COSTS, exactly. ``gap_closed`` becomes uncomputable ON THE TRAINING STORE, and
    nothing else. It is not a validation metric and never was -- selection is on
    ``direct_metric_angstrom_mean`` and the journal's columns are ``direct_metric_angstrom``,
    ``target_loss_sigma_squared``, ``geometry_loss`` and friends, with no ``gap_closed`` among
    them -- so no training decision reads it. It appears in no ``expected_*`` cell and in no row
    of the community comparison table. And the BENCHMARK generator writes its own matched control
    at the same ``draw_seed`` regardless of this flag, so ``measure_gap_closed.py`` still runs on
    the benchmark store, which is the store any reported number would come from.

    WHY AN ARM WOULD WANT THIS. The control roughly DOUBLES training generation, and for the
    per-direction unconditioned arms of D-101 it more than doubles it: the saving documented
    above -- one control serving both levels of a (sequence, cycle) -- exists only because a
    level-conditioned arm visits the same ``draw_seed`` twice. Split the arms and each visits it
    once, so the sharing disappears and the ratio goes from 1 control per 2 steered draws to 1
    per 1. Turning the control off is what keeps a pair of per-direction arms cheaper than the
    single conditioned arm they replace instead of 1.14x its cost.
    """
    return ControlMeasurement(required=bool(getattr(arm, "control_required", False)))


DOMAIN_MOTION_PROPERTY = SteerableProperty(
    name=DOMAIN_MOTION_PROPERTY_NAME,
    # TWO-STATE, and read as design point 2 reads it: the two levels are not a signed dose but a
    # CHOICE OF REFERENCE CONFORMER, at the shared draw seed. ``TWO_STATE_DOSE_LEVELS`` is taken
    # exactly as it stands -- no schedule change, no new dose set -- because (-1, +1) is already
    # the right cardinality and the right pairing; only the reading changes.
    dose_levels=TWO_STATE_DOSE_LEVELS,
    target_origin=DIRECT_METRIC_TARGET_ORIGIN,
    two_state=True,
    property_callback_identity="domain_motion_backbone_rmsd_v1",
    # Its own marker, not a bumped cryptic one: a shared version would say that a domain-motion
    # checkpoint and a cryptic one came from the same generation callback when they did not.
    generation_callback_version="bioemu_direct_backbone_rmsd:v1",
    loss_class=_loss_class,
    runtime_objective_arguments=_objective_arguments,
    resolve_arm=resolve_domain_motion_property_arm,
    adapter_arguments=_adapter_arguments,
    sample_definition_arguments=_sample_definition_arguments,
    checkpoint_arm_arguments=_checkpoint_arm_arguments,
    refused_runtime_keys=_REFUSED,
    # No scorer. ``bioemu-benchmarks``' success@B over N/CA/C/O at tau = 3.0 is the metric, and it
    # is deliberately NOT wired to another property's scorer: a shared scorer is how one property
    # silently answers another's question. The registry refuses ``evaluate`` by name for a
    # property with none, which is the correct failure until the domain-motion scorer exists.
    evaluator=None,
    measure_sample=_measure_sample,
    validation_row_columns=_validation_row_columns,
    validation_row_vocabulary=(_DIRECT_METRIC_COLUMN,),
    validation_selection_metric=_validation_selection_metric,
    control_measurement=_control_measurement,
)

#: Restated for the report and for tests: the two numbers that decide what this objective
#: considers a success and how hard it pushes past one.
DIRECT_METRIC_CONSTANTS = {
    "tau_angstrom": DIRECT_METRIC_TAU_ANGSTROM,
    "hinge_transition": HINGE_TRANSITION,
    "separation_gate_angstrom": SEPARATION_GATE_ANGSTROM,
}
