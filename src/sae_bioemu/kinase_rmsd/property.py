"""The kinase-RMSD steerable property: what ``kinase_conformer_rmsd`` declares.

SCOPE: KINASES. Everything here is a fact about this property -- the dose set, how a level is read,
the objective class, the weights it prices nowhere, and what its arm resolution requires. Not one of
them is in a shared file, which is the point of :mod:`sae_bioemu.steering.property_registry`:
adding this property edits this package and the UNFINGERPRINTED
:mod:`sae_bioemu.steering.property_catalog`, and moves no existing checkpoint.

THE NO-REKEY CLAIM, AND WHY IT HOLDS HERE
------------------------------------------

Four facts together, each checkable:

1. ``property_catalog.py`` is a member of no fingerprint group, so the registration itself hashes
   nowhere.
2. A property's fingerprint hashes only its OWN composition's groups, so a brand-new group name --
   ``kinase_rmsd_training_only`` -- appears in no existing property's payload.
3. No existing group gains or loses a member. ``aligned_rmsd_shared`` is declared BY REFERENCE,
   byte-identical to the cryptic and domain-motion declarations, and
   ``scientific_code_paths`` MERGES identical declarations and refuses two declarations of one
   member as different files -- so naming it adds nothing to it.
4. ``_TRAINING_CODE_SCHEMA`` is NOT bumped. It is the version of what a training fingerprint
   MEANS, and neither the composition nor the membership of any existing property changes, so a
   bump would retire every steering checkpoint in the project for a property none of them runs.

``tests/test_kinase_rmsd_registration.py`` proves 2 and 3 by digesting every group through
``scientific_code_paths`` with and without this registration and asserting every pre-existing group
is byte-identical.

NO ``the progress coordinate``, ANYWHERE
--------------------

There is no ``property_reference`` table, no ``sigma_eligibility`` band and no ``the progress coordinate`` column on any
row this property writes. That is not an omission to be filled in later: under
:data:`~sae_bioemu.steering.schedule.TARGET_ORIGIN_SYSTEM_CONSTANT` the workflow reads the origin
off the PROPERTY's declared ``target_origin`` and then OMITS both keys from the target
configuration, and ``OrdinarySteeringConfig`` REFUSES either if supplied. Both quantities would be
ensemble statistics of a dimensionless coordinate, and this objective has neither an ensemble
statistic nor a coordinate: its target is this kinase's own ``s_p`` and its scale is the same
number. ``domain_motion_backbone_rmsd`` already runs exactly this way.

WHAT THIS PROPERTY CANNOT DO
-----------------------------

``evaluate``. There is no held-out scorer, and the registry refuses the command BY NAME rather than
routing it to the DFG-coordinate property's -- which scores ``the progress coordinate`` and a Kincore label, in
different units, against a different reference. That refusal is the correct failure until a scorer
exists, and it does not block training: no evaluation computes a loss, so a training-only change
cannot move a generated structure or a scored number.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sae_bioemu.kinase_rmsd.artifact import load_kinase_rmsd_reference
from sae_bioemu.kinase_rmsd.loss import KINASE_RMSD_JOURNAL_COLUMN
from sae_bioemu.kinase_rmsd.objective import (
    HINGE_TRANSITION,
    KINASE_RMSD_PROPERTY_NAME,
    KINASE_RMSD_TAU_ANGSTROM,
    MINIMUM_SEPARATION_ANGSTROM,
    POOL_GATE_ANGSTROM,
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
    "KINASE_RMSD_ARM_CONFIG_KEY",
    "KINASE_RMSD_CONSTANTS",
    "KINASE_RMSD_PROPERTY",
    "KINASE_RMSD_PROPERTY_NAME",
    "resolve_kinase_rmsd_property_arm",
    "system_target_constants",
]

#: The ``target`` block key naming this property's arm configuration.
#:
#: **ITS OWN KEY, AND THE DISTINCTNESS IS LOAD-BEARING.**
#: ``property_registry.arm_config_keys`` builds a plain ``{key: owning property}`` dict, and a
#: duplicate key would SILENTLY OVERWRITE rather than raise -- one of the two properties would then
#: own a key it does not answer for, and the foreign-arm refusal in
#: ``bioemu/steering_workflow.py`` would let a config through for the loser. So this must differ
#: from ``kinase_arm``, ``cryptic_arm`` and ``domain_motion_arm``, and
#: ``tests/test_kinase_rmsd_registration.py`` asserts it by comparing the catalog's key count with
#: the number of properties that declare one rather than by reading the four strings.
KINASE_RMSD_ARM_CONFIG_KEY = "kinase_rmsd_arm"

#: The HELD-OUT reading of the metric, in Angstrom. Ungated, because for this property the metric
#: is not an added term whose weight might be zero: it IS the objective, and
#: ``KinaseRmsdSteeringLoss`` refuses ``lambda_m = 0`` as a driverless loss before any draw.
_VALIDATION_COLUMN = "validation_kinase_conformer_rmsd_angstrom"

#: What a kinase-RMSD arm SELECTS its shipped epoch on: the mean held-out deviation, in Angstrom.
#:
#: Not ``target_mae_sigma``, and the reason is a property of the objective rather than of a weight.
#: Under the hinge form the target error IS the hinge, identically zero over the whole region a
#: successful arm lives in; under the default MSE it is ``m/s_p``, which is the metric divided by a
#: per-protein constant and therefore ranks epochs by a normalised quantity rather than by the
#: Angstrom one a report quotes. ``domain_motion/property.py`` states the first half of that
#: argument at length and it transfers unchanged.
#:
#: There is a second reason here that domain motion does not have. The DFG-coordinate property's
#: default selection was ``target_mae_sigma`` on ``the progress coordinate``, and K-22 measured an arm improving that
#: proxy for 40 epochs while the graded quantity moved the OTHER WAY. Selecting this property on
#: anything but its own Angstrom metric would reintroduce exactly that.
_SELECTION_METRIC_NAME = "direct_metric_angstrom_mean"

#: Weights this objective prices NOWHERE, by key and with the reason. A nonzero value for one of
#: these is refused BY NAME before a GPU is allocated, rather than read from the config, dropped by
#: the loss constructor and absent from the objective identity -- the shape that voided six
#: radius-of-gyration arms.
#:
#: The first four are the shared ``EXTENDED_LOSS_TERMS`` vocabulary, every member of which a
#: property must either price or refuse. The last two are NOT admissible runtime keys at all --
#: ``lambda_margin`` lives in ``kinase_arm9_margin.yaml`` and ``lambda_the progress coordinate`` is the DFG-coordinate
#: programme's name for the shared ``lambda_y`` -- and they are declared anyway, because they are
#: the two weights a reader coming from the other kinase property would expect to find and a
#: refusal that names them is worth more than a schema error that does not.
_REFUSED: Mapping[str, str] = {
    "lambda_perp": (
        "the off-path term prices the component of a realised motion perpendicular to a progress "
        "coordinate's transition vector. A direct metric has no coordinate and no transition "
        "vector: every direction that raises the RMSD is already priced by the RMSD"
    ),
    "lambda_perp_fraction": (
        "the per-protein form of lambda_perp, and refused for the same reason: there is no the progress coordinate "
        "whose curvature D-31's bound is about"
    ),
    "lambda_rigid": (
        "the context-rigidity hinge is measured against a pocket's context shell and each "
        "protein's ambient rms_C from the cryptic stage 1. This objective has no pocket, no "
        "context shell and no stage-1 setpoint table -- the 12 Angstrom shell was measured as a "
        "candidate METRIC SET here and refuted, and it exists nowhere in the shipped arm"
    ),
    "lambda_dihedral": (
        "the Kincore cosine dihedral distance d_K is measured to a DFG cluster centroid over the "
        "five-residue motif window. This objective has no centroid table and does not import "
        "kinase_dfg/kincore.py; K-22 additionally measured an arm descending d_K for 40 epochs "
        "while the graded quantity moved the other way, which is the whole reason this property "
        "measures an Angstrom distance to a deposited chain instead"
    ),
    "lambda_margin": (
        "the classifier-region hinge H is the squared violation of the eight affine margins whose "
        "simultaneous nonnegativity IS the Kincore backbone label. It is read off the motif "
        "window's six dihedrals and prices membership of a LABEL REGION; this objective prices "
        "proximity to a deposited CHAIN, and there is no margin table on this path"
    ),
    "lambda_the progress coordinate": (
        "the DFG-coordinate programme's name for the shared lambda_y, which prices a "
        "progress-coordinate target. This property has no the progress coordinate, no property_reference sigma table "
        "and no sigma_eligibility band; lambda_y is spelled 0.0 in the config and "
        "KinaseRmsdSteeringLoss raises by name on any nonzero value"
    ),
}

# ``lambda_c`` is deliberately NOT in the mapping above, even though this objective prices no
# paired dose term. ``refused_runtime_keys`` exists for the EXTENDED vocabulary -- weights a loss
# constructor would accept and silently drop -- and the five SHARED weights are priced by every
# objective, which ``test_every_property_either_prices_or_refuses_every_declared_weight`` holds as
# an invariant. ``lambda_c`` is refused where it cannot be dropped silently: by
# ``KinaseRmsdSteeringLoss.__init__``, which raises with the reason. That refusal also runs before
# anything generates.


def _loss_class() -> type:
    from sae_bioemu.kinase_rmsd.loss import KinaseRmsdSteeringLoss

    return KinaseRmsdSteeringLoss


def _objective_arguments(*, runtime: Mapping[str, Any], arm: Any) -> Mapping[str, Any]:
    """Return the constructor arguments beyond the shared ones: the objective's FORM and weight.

    ``lambda_m`` comes out of ``runtime`` because it is a weight like every other weight and
    belongs in the ``bioemu_pytorch`` block beside them. ``metric_objective`` comes off the ARM,
    because it is a statement about the ARTIFACT: the same declaration also decides the per-system
    target constant :func:`system_target_constants` builds and the prediction :func:`_measure_sample`
    measures, and those two are built in places that never see the workflow's objective block.

    ``property_loss`` is deliberately NOT forwarded. The workflow's own key still reaches the loss
    through the shared arguments, and ``KinaseRmsdSteeringLoss`` refuses the one that does not
    match the form -- so a config naming ``huber`` against an MSE artifact fails by name instead of
    being overridden into silence.
    """
    arguments: dict[str, Any] = {"lambda_m": float(runtime.get("lambda_m", 0.0) or 0.0)}
    if arm is not None:
        arguments["metric_objective"] = arm.metric_objective
    return arguments


def resolve_kinase_rmsd_property_arm(
    *, payload: Mapping[str, Any], base: Path, split: Any, runtime: Mapping[str, Any]
) -> Any:
    """Resolve the arm named by ``target.kinase_rmsd_arm``: the reference, gated.

    The gates run in order, cheapest failure first, and ALL of them run BEFORE the sampler loads:

    1. **Presence.** The metric is a distance to THIS kinase's two deposited chains and nothing in
       ``bioemu_pytorch`` supplies them, so the block is required.
    2. **The reference loads and re-measures.**
       :func:`~sae_bioemu.kinase_rmsd.artifact.load_kinase_rmsd_reference` re-derives every stored
       ``s_p`` from that entry's own coordinates and refuses a disagreement, so a stale per-protein
       weight cannot enter a run.
    3. **The population is the same.** The reference must cover every sequence the workflow's
       split walks, train, validation and test alike.
    4. **The divisor floor.** Every scheduled kinase must clear
       :data:`~sae_bioemu.kinase_rmsd.objective.POOL_GATE_ANGSTROM` strictly. ``s_p`` DIVIDES this
       objective under both forms, so a kinase whose two chains nearly coincide over its
       activation loop would enter the batch at ``m / ~0`` and own every gradient in it. It is
       REFUSED here rather than filtered silently: a run that quietly trained on a different
       population than its split record names has passed every other gate about the wrong set of
       proteins.

       **THIS IS NOT ``s_p > tau``, AND THE DIFFERENCE IS MEASURED.** See
       :data:`~sae_bioemu.kinase_rmsd.objective.POOL_GATE_ANGSTROM`: a 3.0 Angstrom gate would drop
       25 of this pool's 43 genes and with them the pre-registered K-12 split. Kinases below tau
       are a REPORTED STRATUM -- counted and named in the message below at every resolution -- and
       are never dropped.

    The resolved arm IS the reference: it carries the chains, the metric indices, the per-system
    normalisations, the index set it was built under and its own identity, and there is no second
    object for it to hide behind.
    """
    target = payload.get("target")
    configured = target.get(KINASE_RMSD_ARM_CONFIG_KEY) if isinstance(target, dict) else None
    if configured is None:
        raise ValueError(
            f"{KINASE_RMSD_PROPERTY_NAME} requires target.{KINASE_RMSD_ARM_CONFIG_KEY} naming the "
            "reference artifact: the metric is a superposed N/CA/C/O RMSD over this kinase's "
            "activation loop to its own two deposited chains, and nothing in bioemu_pytorch "
            "supplies either of them"
        )
    reference = load_kinase_rmsd_reference(
        resolve_config_path(base, configured, f"target.{KINASE_RMSD_ARM_CONFIG_KEY}")
    )
    scheduled = sorted({*split.train, *split.validation, *split.test})
    missing = [name for name in scheduled if name not in reference.entries]
    if missing:
        raise ValueError(
            f"{len(missing)} scheduled sequences are absent from the kinase-RMSD reference "
            f"{reference.path}; first: {missing[:5]}. The direct metric has no fallback reference"
        )
    degenerate = [
        name
        for name in scheduled
        if float(reference.entries[name].separation_angstrom) < POOL_GATE_ANGSTROM
    ]
    if degenerate:
        raise ValueError(
            f"{len(degenerate)} scheduled kinases have their two deposited chains closer than "
            f"{POOL_GATE_ANGSTROM} Angstrom under the N/CA/C/O metric over "
            f"{reference.metric_set!r}; first: {degenerate[:5]}. s_p DIVIDES this objective under "
            "both forms, so such a kinase enters the batch at m/~0 and owns every gradient in it; "
            "the split record must be built from the gated pool rather than gated at runtime"
        )
    # THE SUB-TAU STRATUM, REPORTED AND NEVER DROPPED. These kinases train and are scored; what
    # they cannot support is a claim read against the FIXED 3.0 Angstrom threshold, because their
    # two deposited chains are closer than it to begin with. Under the default MSE form no
    # threshold enters the objective at all, and under the hinge ``tau_p = min(tau, 0.5 * s_p)``
    # adapts per kinase -- so this is a caveat on the REPORT and not a defect in the training. It
    # is printed at every resolution rather than left to a sidecar because a reader of this arm's
    # output has to know that a majority of its population is in it.
    below_tau = sorted(
        name
        for name in scheduled
        if float(reference.entries[name].separation_angstrom) <= KINASE_RMSD_TAU_ANGSTROM
    )
    print(
        f"{KINASE_RMSD_PROPERTY_NAME}: {len(below_tau)} of {len(scheduled)} scheduled kinases have "
        f"s_p <= {KINASE_RMSD_TAU_ANGSTROM} Angstrom over {reference.metric_set!r} and are carried "
        f"as a REPORTED STRATUM, not dropped; first: {below_tau[:5]}. No success@tau claim at the "
        "fixed threshold is available for them -- see kinase_rmsd/objective.POOL_GATE_ANGSTROM",
        flush=True,
    )
    del runtime
    return reference


def _adapter_arguments(arm: Any) -> Mapping[str, Any]:
    """No per-residue conditioning channel.

    There is deliberately no kinase-RMSD equivalent of the cryptic site field or of the
    DFG-coordinate motif channel. A channel built from the deposited chains would tell the adapter
    WHERE the activation loop is, and the direct metric would then be satisfied partly by a
    hand-supplied answer; whether the adapter can find the loop is the question. If one is ever
    wanted it is a new arm, declared here, and not a default.
    """
    del arm
    return {}


def _sample_definition_arguments(arm: Any) -> Mapping[str, Any]:
    """Hand the resolved reference over under the SHARED ``definition`` keyword.

    The type is this package's own :class:`~sae_bioemu.kinase_rmsd.artifact.KinaseRmsdDefinition`
    rather than a class added to ``bioemu/steering_properties.py``: that file is in
    ``generation_shared`` and a fourth definition class there would re-key every existing steering
    checkpoint for a type none of them constructs.
    """
    from sae_bioemu.kinase_rmsd.artifact import KinaseRmsdDefinition

    return {} if arm is None else {"definition": KinaseRmsdDefinition(reference=arm)}


def _measure_sample(*, batch: Any) -> Any:
    """Measure the superposed N/CA/C/O RMSD of each steered draw over its kinase's own metric set.

    The whole objective, in the shape the shared ``SteeringLoss`` prices. Under the DEFAULT
    :data:`~sae_bioemu.kinase_rmsd.objective.METRIC_OBJECTIVE_MSE`:

        ``prediction = m + s_p``         the measured RMSD, placed on this kinase's own origin
        ``target     = s_p``             supplied by the system-constant table
        ``scale      = s_p``             supplied by the same table

    so ``mean_n ((prediction - target)/scale)^2`` IS ``mean_n (m_n/s_p)^2``, with ``tau`` nowhere in
    it. Under the hinge ablation the prediction is ``max(m, tau_p)`` against ``target = tau_p``.
    **The clamp is the hinge's one-sidedness and the shift is the MSE's zero**; both are re-checked
    elementwise by ``KinaseRmsdSteeringLoss.forward`` against the unclamped ``m`` on the wire,
    rather than trusted from this line, because both failures produce a plausible loss curve and no
    other symptom.

    The UNCLAMPED ``m`` travels beside it on ``metric_deviation_angstrom`` -- the shared TRANSPORT
    field -- and is journalled under this property's own column name.

    The level SELECTS THE DEPOSITED CHAIN and is read off the batch rather than inferred, because a
    distance to the wrong chain is a NUMBER and not an error.
    """
    from sae_bioemu.kinase_rmsd.artifact import KinaseRmsdDefinition, kinase_rmsd_metric_batch
    from sae_bioemu.kinase_rmsd.frames import batched_backbone_atoms
    from sae_bioemu.steering.property_registry import PropertyMeasurement

    batch.require_relative_levels(KINASE_RMSD_PROPERTY_NAME)
    definition = batch.definition
    if not isinstance(definition, KinaseRmsdDefinition):
        raise ValueError(
            f"{KINASE_RMSD_PROPERTY_NAME} needs its resolved reference: the metric is a superposed "
            "N/CA/C/O RMSD over this kinase's activation loop to its own two deposited chains, and "
            "neither an adapter nor a structure supplies them"
        )
    reference = definition.reference
    # N, CA, C and O off the denoiser's per-residue frames, with the gradient intact. NOT through
    # ``bioemu/cryptic_generation.py``: that file is the sole member of the code group both
    # PROGRESS-COORDINATE properties compose, so extending it to return O would re-key every
    # cryptic and DFG-coordinate checkpoint for a change made on behalf of a property neither runs.
    backbone = batched_backbone_atoms(
        batch.steered.coordinates,
        batch.node_orientations,
        [item.sequence for item in batch.records],
    )
    sequence_ids = batch.sequence_ids
    metric, _priced, _separation = kinase_rmsd_metric_batch(
        reference, sequence_ids, batch.levels, backbone
    )
    form = reference.metric_objective
    # The per-kinase origin, elementwise. Built as a TENSOR of per-member constants rather than one
    # scalar because one protein's origin applied to another's draw is a finite, plausible and
    # wrong loss -- and ``KinaseRmsdNormalisation.prediction`` is the one place the shift is
    # defined, so this path and :func:`system_target_constants` cannot disagree about it.
    constants = metric.new_tensor(
        [
            reference.require(name).normalisation.target_constant_angstrom(form)
            for name in sequence_ids
        ]
    )
    return PropertyMeasurement(
        prediction=_prediction_on_origin(metric, constants, form),
        target=batch.target,
        sample_fields={"metric_deviation_angstrom": metric},
    )


def _prediction_on_origin(metric: Any, constants: Any, metric_objective: str) -> Any:
    """Return the batched ``prediction``: ``m + origin`` under the MSE, ``max(m, tau_p)`` under the
    hinge. The batched twin of
    :meth:`~sae_bioemu.kinase_rmsd.objective.KinaseRmsdNormalisation.prediction`, dispatching on
    the same validated form so the two cannot come to mean different things."""
    import torch

    from sae_bioemu.kinase_rmsd.objective import (
        METRIC_OBJECTIVE_MSE,
        validated_metric_objective,
    )

    form = validated_metric_objective(metric_objective)
    if form == METRIC_OBJECTIVE_MSE:
        return metric + constants
    return torch.maximum(metric, constants)


def _checkpoint_arm_arguments(arm: Any) -> Mapping[str, Any]:
    """Bind the resolved reference into the CHECKPOINT and nothing else.

    Under the GENERIC ``property_arm_identity`` field rather than a kinase one, and deliberately
    NOT the cryptic field: that one is also what ``steers_cryptic_pocket`` is derived from, and a
    run binding its arm there would acquire the pocket monitor columns on every validation row with
    no pocket to measure them in.

    Deliberately not in ``schedule_seed_fingerprint``: routing an artifact digest through the seed
    would redraw every schedule and confound each cross-arm difference with a complete
    re-randomisation. The identity carries the per-system ``s_p`` and ``tau_p``, because those are
    per-protein weights on the objective and a checkpoint must be bound to the weights it descended
    -- and it carries ``metric_set``, because two references over two index sets are two objectives.
    """
    if arm is None:
        return {}
    return {
        "property_arm_identity": arm.identity(),
        # The per-system ``{tau_p, s_p}`` table the shared target construction reads. It rides this
        # hook -- documented as "extra ``OrdinarySteeringConfig`` keyword arguments the resolved
        # arm supplies" -- rather than a hook of its own, because that is exactly what it is: a
        # config field whose value only the resolved arm can produce.
        "system_target_constants": system_target_constants(arm),
    }


def _validation_row_columns(*, objective: Any, config: Any) -> Mapping[str, JournalledSampleColumn]:
    """The held-out deviation, in Angstrom, on every kinase-RMSD validation row.

    Ungated, because for this property the metric is not an added term whose weight might be zero:
    it IS the objective, and ``KinaseRmsdSteeringLoss`` refuses ``lambda_m = 0`` as a driverless
    loss before any draw. The TRAINING column gates on ``lambda_m``; this one does not, because a
    validation draw computes no loss at all.
    """
    del objective, config
    return {_VALIDATION_COLUMN: JournalledSampleColumn(field="metric_deviation_angstrom")}


def _validation_selection_metric(*, objective: Any, arm: Any) -> ValidationSelectionMetric | None:
    """None: this property's arms run NO VALIDATION LEG, so there is nothing to select on.

    The K-26 pool is 41 train / 0 validation / 20 holdout -- the training set is small enough
    that spending genes on model selection costs more than selection buys, so every arm here
    sets ``model_selection: none``. Under that mode ``OrdinarySteeringConfig`` REFUSES a
    declared metric by name: it "would name a comparison this run never makes and would be
    recorded as though it had". Returning a metric unconditionally, as the domain-motion
    property does, made this property's only configuration unlaunchable -- measured: it died at 31s.

    TO RESTORE SELECTION: give the split a validation role, set ``model_selection:
    validation_split``, and return ``ValidationSelectionMetric(name=_SELECTION_METRIC_NAME,
    row_column=_VALIDATION_COLUMN)`` here. Both names stay defined and
    ``PropertyRegistration.selection_metric_names`` still admits the metric, so the completion
    gate can attest such a run without further change.
    """
    del objective, arm
    return None


def system_target_constants(reference: Any) -> SystemTargetConstants:
    """Build the per-kinase ``{target, scale}`` table this property's targets are made of.

    * ``target`` is the FORM's own origin, from
      :meth:`~sae_bioemu.kinase_rmsd.objective.KinaseRmsdNormalisation.target_constant_angstrom`:
      ``tau_p = min(tau, 0.5 * s_p)`` under the hinge, and ``s_p`` under the default MSE, where it
      is a pure origin the prediction carries too and which therefore cancels exactly.
    * ``scale`` is ``s_p``, that kinase's own superposed N/CA/C/O separation over its metric set.

    Both are re-derived from the reference's own coordinates at load, so this table cannot drift
    away from the chains the metric is measured against. The table's digest reaches the checkpoint
    identity, so the two forms give two tables and a run cannot resume across them.
    """
    form = reference.metric_objective
    return SystemTargetConstants(
        entries={
            name: SystemTargetConstant(
                target=entry.normalisation.target_constant_angstrom(form),
                scale=float(entry.separation_angstrom),
            )
            for name, entry in reference.entries.items()
        }
    )


def _control_measurement(arm: Any) -> ControlMeasurement:
    """REQUIRE the matched unsteered control. Generate it; price nothing with it.

    ``required=True`` and no ``atom_set``, so a C-alpha control is generated for every draw
    whatever ``lambda_l`` is. Without this declaration the shared
    ``generates_reference = wants_local_structure or control.required`` produces a control only at
    ``lambda_l > 0``, and this arm prices ``lambda_l`` at zero.

    WHY IT IS REQUIRED, AND WHY NOT THROUGH ``lambda_l``
    -----------------------------------------------------

    The only readable primary this programme has found for an Angstrom metric is the PAIRED one:
    per draw, the fraction of that system's own gap the intervention closed against an unsteered
    control generated from THE SAME ``draw_seed``. A fixed ``success@tau`` is not usable here --
    gating so that the baseline fails leaves a band of about (1.50, 1.66] on the cryptic pool --
    and a paired reading refuses rather than falling back when a control is absent. So without this
    declaration the arm produces a checkpoint its own primary cannot read.

    Raising ``lambda_l`` would also generate the control and is the WRONG way to get it: it prices
    the local-structure term, which changes the OBJECTIVE and makes the arm incomparable on the
    axis the programme is about. What is wanted is measurement, not pricing, and the two are
    separable exactly here.

    **NOTHING IS PRICED.** No term is added, no weight changes, and ``lambda_l`` stays 0.

    **THE COST, STATED.** It is an extra UNSTEERED draw per (sequence, draw_seed), and the two
    levels of one (sequence, cycle) share a seed, so one control serves both levels. An unsteered
    draw carries no adapter, no retained DPM iteration and no autograd graph, so on an EIGHT-SITE
    arm it is much cheaper than the steered draw it is paired with rather than a doubling of cost;
    it is stated here because it is a real addition to generation and not a free column.

    ``frames`` stays ``False``. Frames would be read off the control for a reconstruction nothing
    on this path performs: the metric is a distance to a DEPOSITED chain measured off the STEERED
    draw's own frames.

    OPTED OUT PER ARM (D-101), defaulting to the unconditional behaviour above so every arm built
    before this is bit-identical. This is the domain-motion shape of control, not the cryptic one:
    under ``TARGET_ORIGIN_SYSTEM_CONSTANT`` the control is NOT the origin the target is measured
    from, it is an unpriced instrument for the paired primary, so an arm can decline it and still
    have a target. (``dfg_progress_coordinate``, the the progress coordinate property this one replaced, cannot: there
    the control IS the origin.)

    WHAT DECLINING IT COSTS: the paired primary becomes uncomputable on the TRAINING store, and
    nothing else. Selection does not read it, no weighted term consumes it, and the BENCHMARK
    generator writes its own matched control at the same ``draw_seed`` regardless -- so the paired
    number is still available from the store any reported figure would come from.

    WHY AN ARM WOULD DECLINE IT: the control roughly doubles training generation, and for the
    per-direction unconditioned arms of D-101 it more than doubles it. The saving that makes it
    "somewhat better than 2x" -- one control serving both levels of a (sequence, cycle) -- exists
    only because a level-conditioned arm visits the same ``draw_seed`` twice. Split the arms and
    each visits it once, so the ratio goes from 1 control per 2 steered draws to 1 per 1.
    THE DEFAULT IS NOW OFF, which is a behaviour change for every arm of this property and not
    only the new ones. The arm is a ``.pt`` artifact, so a per-arm flag would mean rebuilding and
    re-keying `kinase_rmsd_reference.pt`; the control here is explicitly unpriced ("Generate it;
    price nothing with it"), the cost of generating it is roughly a doubling of training
    generation, and the only thing lost is the paired primary ON THE TRAINING STORE -- the
    benchmark generator writes its own matched control at the same ``draw_seed`` regardless. An
    arm that wants it back sets ``control_required = True`` on its artifact.
    """
    return ControlMeasurement(required=bool(getattr(arm, "control_required", False)))


KINASE_RMSD_PROPERTY = SteerableProperty(
    name=KINASE_RMSD_PROPERTY_NAME,
    # TWO-STATE. The two levels are not a signed dose but a CHOICE OF DEPOSITED CHAIN, at the
    # shared draw seed. ``TWO_STATE_DOSE_LEVELS`` is taken exactly as it stands -- no schedule
    # change, no new dose set -- because (-1, +1) is already the right cardinality and the right
    # pairing; only the reading changes. Here both chains additionally carry a Kincore LABEL, so
    # "which conformer does +1 name" has a published answer rather than a convention.
    dose_levels=TWO_STATE_DOSE_LEVELS,
    # ``t = s_p`` (MSE) or ``tau_p`` (hinge) at every level, with the level selecting which
    # deposited chain the distance is measured to. The SHARED generic origin, not a kinase one:
    # ``domain_motion_backbone_rmsd`` already runs on it and the cryptic direct-metric term is
    # queued for the same construction.
    target_origin=TARGET_ORIGIN_SYSTEM_CONSTANT,
    two_state=True,
    property_callback_identity="kinase_conformer_rmsd_v1",
    # Its own marker, not a bumped domain-motion one: a shared version would say that a kinase-RMSD
    # checkpoint and a domain-motion one came from the same generation callback when they did not.
    generation_callback_version="bioemu_kinase_conformer_rmsd:v1",
    loss_class=_loss_class,
    runtime_objective_arguments=_objective_arguments,
    resolve_arm=resolve_kinase_rmsd_property_arm,
    adapter_arguments=_adapter_arguments,
    sample_definition_arguments=_sample_definition_arguments,
    checkpoint_arm_arguments=_checkpoint_arm_arguments,
    refused_runtime_keys=_REFUSED,
    # No scorer, and deliberately NOT wired to the DFG-coordinate property's: that one scores the progress coordinate
    # and a Kincore backbone label, in different units, against a different reference. A shared
    # scorer is how one property silently answers another's question. The registry refuses
    # ``evaluate`` by name for a property with none, which is the correct failure until one exists.
    evaluator=None,
    measure_sample=_measure_sample,
    validation_row_columns=_validation_row_columns,
    validation_row_vocabulary=(_VALIDATION_COLUMN,),
    validation_selection_metric=_validation_selection_metric,
    control_measurement=_control_measurement,
)

#: Restated for the report and for tests: the numbers that decide what this objective considers a
#: success, how hard it pushes past one, and which kinases are admitted at all.
KINASE_RMSD_CONSTANTS = {
    "tau_angstrom": KINASE_RMSD_TAU_ANGSTROM,
    "hinge_transition": HINGE_TRANSITION,
    "pool_gate_angstrom": POOL_GATE_ANGSTROM,
    "minimum_separation_angstrom": MINIMUM_SEPARATION_ANGSTROM,
    "journal_column": KINASE_RMSD_JOURNAL_COLUMN,
}
