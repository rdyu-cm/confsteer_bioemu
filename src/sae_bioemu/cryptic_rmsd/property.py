"""The pocket-RMSD steerable property: what ``pocket_conformer_rmsd`` declares.

SCOPE: CRYPTIC POCKETS, WITHOUT ``the progress coordinate``. Everything here is a fact about this property -- the dose
set, how a level is read, the target origin, the objective class, the weights it prices nowhere,
and what its arm resolution requires. Not one of them is in a shared file, which is the point of
:mod:`sae_bioemu.steering.property_registry`: adding this property edits this package and the
UNFINGERPRINTED :mod:`sae_bioemu.steering.property_catalog`, and moves no existing checkpoint.

WHY A NEW PROPERTY AND NOT A REPOINTED ``pocket_progress_coordinate``
-----------------------------------------------------------------------

Because ``target_origin`` is part of what a checkpoint IS. Every cryptic checkpoint on disk records
``relative_target_origin: matched_control_state`` in its ``adapter_config``, which the resume path
and the evaluation gate compare by canonical-JSON equality, so repointing that property at
:data:`~sae_bioemu.steering.schedule.TARGET_ORIGIN_SYSTEM_CONSTANT` would refuse all of them. And
the two origins are not the same construction with ``the progress coordinate`` taken out:

    matched_control_state   ``t = the progress coordinate_state(o) - the progress coordinate_reporting(control)``; REQUIRES the matched
                            unsteered draw per row, requires ``property_reference`` for
                            ``sigma_the progress coordinate`` and an eligibility band, and normalises the loss by
                            ``per_channel_sigma_transition``.
    system_constant         ``t = target_p`` at every level, from a per-system
                            ``SystemTargetConstants`` table with ``target > 0`` and ``scale > 0``;
                            REFUSES ``property_reference``, ``sigma_eligibility`` and
                            ``dose_multiplier``; normalises by ``sequence_sigma``; and the control
                            is not the origin, so it can be declined entirely.

A new property is the route this codebase has already taken twice --
``domain_motion_backbone_rmsd`` and then ``kinase_conformer_rmsd``, the second built precisely to
avoid re-keying the ``the progress coordinate`` property beside it -- and it is what this follows.

THE NO-REKEY CLAIM, AND WHY IT HOLDS HERE
------------------------------------------

Four facts together, each checkable:

1. ``property_catalog.py`` is a member of no fingerprint group, so the registration itself hashes
   nowhere.
2. A property's fingerprint hashes only its OWN composition's groups, so a brand-new group name --
   ``cryptic_rmsd_training_only`` -- appears in no existing property's payload.
3. No existing group gains or loses a member. ``aligned_rmsd_shared`` is declared BY REFERENCE,
   byte-identical to the cryptic, domain-motion and kinase-RMSD declarations, and
   ``scientific_code_paths`` MERGES identical declarations and refuses two declarations of one
   member as different files -- so naming it adds nothing to it.
4. ``_TRAINING_CODE_SCHEMA`` is NOT bumped. It is the version of what a training fingerprint
   MEANS, and neither the composition nor the membership of any existing property changes, so a
   bump would retire every steering checkpoint in the project for a property none of them runs.

``tests/test_cryptic_rmsd_registration.py`` proves 2 and 3 by digesting every group through
``scientific_code_paths`` with and without this registration and asserting every pre-existing group
is byte-identical.

NO ``the progress coordinate``, ANYWHERE
--------------------

There is no ``property_reference`` table, no ``sigma_eligibility`` band, no ``dose_multiplier`` and
no ``the progress coordinate`` column on any row this property writes. That is not an omission to be filled in later:
under :data:`~sae_bioemu.steering.schedule.TARGET_ORIGIN_SYSTEM_CONSTANT` the workflow reads the
origin off the PROPERTY's declared ``target_origin`` and then OMITS all three from the target
configuration, ``OrdinarySteeringConfig`` REFUSES any of them if supplied, and
``_validate_system_constant_target_identity`` refuses a CHECKPOINT that records one. Both of the
first two would be ensemble statistics of a dimensionless coordinate, and this objective has
neither an ensemble statistic nor a coordinate: its target is this protein's own ``s_p`` and its
scale is the same number.

WHAT THIS PROPERTY CANNOT DO
-----------------------------

``evaluate``. There is no held-out scorer yet and the registry refuses the command BY NAME rather
than routing it to ``pocket_progress_coordinate``'s -- which scores ``the progress coordinate`` against a pair set and a
transition vector, in different units, and would additionally demand a resolved cryptic arm this
property never builds. That refusal is the correct failure until a scorer exists, and it does not
block training: no evaluation computes a loss, so a training-only change cannot move a generated
structure or a scored number. ``cryptic_steering/success_at_b.py`` is the metric a scorer would be
built from, and it reads written structures rather than this objective.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sae_bioemu.cryptic_rmsd.artifact import load_pocket_rmsd_reference
from sae_bioemu.cryptic_rmsd.loss import POCKET_RMSD_JOURNAL_COLUMN
from sae_bioemu.cryptic_rmsd.objective import (
    BENCHMARK_TAU_ANGSTROM,
    MINIMUM_SEPARATION_ANGSTROM,
    POCKET_RMSD_PROPERTY_NAME,
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
    "POCKET_RMSD_ARM_CONFIG_KEY",
    "POCKET_RMSD_CONSTANTS",
    "POCKET_RMSD_PROPERTY",
    "POCKET_RMSD_PROPERTY_NAME",
    "resolve_pocket_rmsd_property_arm",
    "system_target_constants",
]

#: The ``target`` block key naming this property's reference artifact.
#:
#: **ITS OWN KEY, AND THE DISTINCTNESS IS LOAD-BEARING.**
#: ``property_registry.arm_config_keys`` builds a plain ``{key: owning property}`` dict, and a
#: duplicate key would SILENTLY OVERWRITE rather than raise -- one of the two properties would then
#: own a key it does not answer for, and the foreign-arm refusal in ``bioemu/steering_workflow.py``
#: would let a config through for the loser. So this must differ from ``cryptic_arm``,
#: ``kinase_arm``, ``kinase_rmsd_arm`` and ``domain_motion_arm``; the collision that matters most
#: is ``cryptic_arm``, because both properties are about cryptic pockets and the two loaders open
#: entirely different files.
POCKET_RMSD_ARM_CONFIG_KEY = "pocket_rmsd_arm"

#: The HELD-OUT reading of the metric, in the Angstrom the benchmark grades. Gated on nothing but
#: this property running, because unlike ``pocket_progress_coordinate``'s ``lambda_m`` term it is
#: not an optional addition: the deviation IS this objective, and ``PocketRmsdSteeringLoss``
#: refuses ``lambda_m = 0`` as a driverless loss before any draw. A validation row without it would
#: be a row that says nothing about what the arm was for.
#:
#: A DISTINCT column name from the cryptic ``validation_metric_deviation_angstrom``, on the
#: precedent the catalog states about ``direct_metric_angstrom``: one column name carries one gate
#: and one meaning, and a deviation journalled by an arm whose TARGETS came from ``s_p`` is not the
#: same measurement as one journalled by an arm whose targets came from a ``the progress coordinate`` state.
_VALIDATION_COLUMN = "validation_pocket_conformer_rmsd_angstrom"

#: What a pocket-RMSD arm SELECTS its shipped epoch on: the mean held-out deviation, in Angstrom.
#:
#: Not ``target_mae_sigma``, and the reason is a property of the objective rather than of a weight.
#: Here ``target = s_p`` and ``prediction = m + s_p``, so the target error is exactly ``m / s_p`` --
#: the deviation divided by a per-protein constant. Selecting on it ranks epochs by a normalised
#: quantity rather than by the Angstrom one a report quotes, so an epoch that improved three large
#: transitions can lose to one that improved a single small one. The unclamped mean deviation
#: cannot do that.
#:
#: Its own spelling, distinct from the cryptic ``metric_deviation_angstrom_mean`` and from the
#: ``direct_metric_angstrom_mean`` domain motion and kinase RMSD share, for the reason the column
#: name is distinct: this is a persisted identity key, and two runs recording one name for two
#: measurements is how a comparison comes to pool them.
_SELECTION_METRIC_NAME = "pocket_conformer_rmsd_angstrom_mean"

#: Weights this objective prices NOWHERE, by key and with the reason. A nonzero value for one of
#: these is refused BY NAME before a GPU is allocated, rather than read from the config, dropped by
#: the loss constructor and absent from the objective identity -- the shape that voided six
#: radius-of-gyration arms.
#:
#: The first four are the shared ``EXTENDED_LOSS_TERMS`` vocabulary, every member of which a
#: property must either price or refuse. ``context_setpoint_percentile`` is NOT a weight and is
#: declared anyway, because it is the one further key a config copied from a cryptic arm would
#: carry: it is the rigidity term's identity-bound parameter, so a value here would be recorded
#: against a term this objective does not have. (``null`` is falsy and is therefore still
#: accepted, exactly as a zero weight is -- the shipped cryptic templates spell their inapplicable
#: settings out rather than dropping them, and that has always been correct.)
_REFUSED: Mapping[str, str] = {
    "lambda_perp": (
        "the off-path term prices the component of a realised motion perpendicular to the pocket "
        "transition vector v = d_holo - d_apo, and that vector is a property of the progress coordinate. A direct "
        "metric has no coordinate and no transition vector: every direction that raises the "
        "deviation is already priced by the deviation"
    ),
    "lambda_perp_fraction": (
        "the per-protein form of lambda_perp, c*lambda_y/sigma_the progress coordinate^2, and refused for the same "
        "reason plus one more: there is no the progress coordinate whose curvature D-31's bound is about, no sigma_the progress coordinate "
        "to divide by -- this origin loads no property_reference table at all -- and lambda_y is "
        "required to be zero here, so the weight it is a fraction OF is zero"
    ),
    "lambda_rigid": (
        "the context-rigidity hinge is measured against a pocket's CONTEXT SHELL and each "
        "protein's ambient rms_C from the cryptic stage 1, both of which come off the "
        "fingerprinted pocket artifact and its setpoint table. This property reads neither file -- "
        "that is the whole point of its own reference artifact -- so there is no c0 to hinge "
        "against and a weight here would be read from the config and applied to nothing"
    ),
    "lambda_dihedral": (
        "the Kincore cosine dihedral distance d_K is measured to a DFG cluster centroid over a "
        "five-residue motif window. This objective has no motif, no centroid table and does not "
        "import kinase_dfg/kincore.py"
    ),
    "context_setpoint_percentile": (
        "the percentile that DEFINES the context-rigidity setpoint c0, and therefore part of what "
        "lambda_rigid means (D-30). With that term refused above, a percentile here would reach "
        "the objective identity as a parameter of a term the objective does not contain"
    ),
}

# ``lambda_c`` is deliberately NOT in the mapping above, even though this objective prices no
# paired dose term. ``refused_runtime_keys`` exists for the EXTENDED vocabulary -- weights a loss
# constructor would accept and silently drop -- and the five SHARED weights are priced by every
# objective, which ``test_every_property_either_prices_or_refuses_every_declared_weight`` holds as
# an invariant. ``lambda_c`` is refused where it cannot be dropped silently: by
# ``PocketRmsdSteeringLoss.__init__``, which raises with the reason. That refusal also runs before
# anything generates.


def _loss_class() -> type:
    from sae_bioemu.cryptic_rmsd.loss import PocketRmsdSteeringLoss

    return PocketRmsdSteeringLoss


def _objective_arguments(*, runtime: Mapping[str, Any], arm: Any) -> Mapping[str, Any]:
    """Return the constructor arguments beyond the shared ones: the objective's FORM and weight.

    ``lambda_m`` comes out of ``runtime`` because it is a weight like every other weight and
    belongs in the ``bioemu_pytorch`` block beside them. ``metric_objective`` comes off the ARM,
    because it is a statement about the ARTIFACT: the same declaration also decides the per-system
    target constant :func:`system_target_constants` builds and the prediction
    :func:`_measure_sample` measures, and those two are built in places that never see the
    workflow's objective block.

    ``property_loss`` is deliberately NOT forwarded. The workflow's own key still reaches the loss
    through the shared arguments, and ``PocketRmsdSteeringLoss`` refuses the one that does not
    match the form -- so a config naming ``huber`` against this artifact fails by name instead of
    being overridden into silence.
    """
    arguments: dict[str, Any] = {"lambda_m": float(runtime.get("lambda_m", 0.0) or 0.0)}
    if arm is not None:
        arguments["metric_objective"] = arm.metric_objective
    return arguments


def resolve_pocket_rmsd_property_arm(
    *, payload: Mapping[str, Any], base: Path, split: Any, runtime: Mapping[str, Any]
) -> Any:
    """Resolve the reference named by ``target.pocket_rmsd_arm``, gated.

    The gates run in order, cheapest failure first, and ALL of them run BEFORE the sampler loads:

    1. **Presence.** The metric is a deviation from THIS protein's two deposited conformers and
       nothing in ``bioemu_pytorch`` supplies them, so the block is required.
    2. **The reference loads and re-measures.**
       :func:`~sae_bioemu.cryptic_rmsd.artifact.load_pocket_rmsd_reference` re-derives every stored
       ``s_p`` from that entry's own coordinates and refuses a disagreement, so a stale per-protein
       weight -- which on this origin is also a stale TARGET -- cannot enter a run.
    3. **The population is the same.** The reference must cover every sequence the workflow's split
       walks, train, validation and test alike, because a validation draw journals the deviation
       too. Reported as a population rather than one name at a time.
    4. **The divisor floor.** Every scheduled protein must clear
       :data:`~sae_bioemu.cryptic_rmsd.objective.POOL_GATE_ANGSTROM` strictly. ``s_p`` DIVIDES this
       objective AND is its target, so a protein whose two crystals nearly coincide over the metric
       region would enter the batch at ``m / ~0`` and own every gradient in it. It is REFUSED here
       rather than filtered silently: a run that quietly trained on a different population than its
       split record names has passed every other gate about the wrong set of proteins.

       **THIS IS NOT ``s_p > tau``, AND THE DIFFERENCE IS DELIBERATE.** See
       :data:`~sae_bioemu.cryptic_rmsd.objective.POOL_GATE_ANGSTROM`: the domain-motion pool gate
       exists because a FIXED ``tau`` makes ``success@tau`` trivially satisfiable for a system
       whose conformers are closer than tau, and this objective has no threshold at all. Proteins
       inside the benchmark thresholds are a REPORTED STRATUM -- counted and named in the message
       below at every resolution -- and are never dropped.

    The resolved arm IS the reference: it carries the conformers, the metric indices, the
    per-system normalisations and its own identity, and there is no second object for it to hide
    behind.
    """
    target = payload.get("target")
    configured = target.get(POCKET_RMSD_ARM_CONFIG_KEY) if isinstance(target, dict) else None
    if configured is None:
        raise ValueError(
            f"{POCKET_RMSD_PROPERTY_NAME} requires target.{POCKET_RMSD_ARM_CONFIG_KEY} naming the "
            "reference artifact: the metric is a superposed N/CA/C/O deviation from this protein's "
            "own deposited apo and holo conformers over the benchmark's metric region, and nothing "
            "in bioemu_pytorch supplies either of them"
        )
    reference = load_pocket_rmsd_reference(
        resolve_config_path(base, configured, f"target.{POCKET_RMSD_ARM_CONFIG_KEY}")
    )
    scheduled = sorted({*split.train, *split.validation, *split.test})
    reference.require_sequences(scheduled)
    degenerate = [
        name
        for name in scheduled
        if float(reference.entries[name].separation_angstrom) < POOL_GATE_ANGSTROM
    ]
    if degenerate:
        raise ValueError(
            f"{len(degenerate)} scheduled proteins have their deposited apo and holo closer than "
            f"{POOL_GATE_ANGSTROM} Angstrom over the metric region; first: {degenerate[:5]}. Under "
            "the system_constant origin s_p is BOTH the target and the per-observation divisor, so "
            "such a protein enters the batch at m/~0 and owns every gradient in it; the split "
            "record must be built from the gated pool rather than gated at runtime"
        )
    # THE SUB-TAU STRATUM, REPORTED AND NEVER DROPPED. These proteins train and are scored; what
    # they cannot support is a claim read against a FIXED benchmark threshold, because their two
    # deposited conformers are closer than it to begin with -- an unsteered draw sitting on apo
    # already scores against holo. No threshold enters this objective at all, so this is a caveat
    # on the REPORT and not a defect in the training. It is printed at every resolution rather than
    # left to a sidecar because a reader of this arm's output has to know how much of its
    # population is in it, and at WHICH of the two thresholds of record.
    for threshold in BENCHMARK_TAU_ANGSTROM:
        inside = sorted(
            name
            for name in scheduled
            if float(reference.entries[name].separation_angstrom) <= threshold
        )
        print(
            f"{POCKET_RMSD_PROPERTY_NAME}: {len(inside)} of {len(scheduled)} scheduled proteins "
            f"have s_p <= {threshold} Angstrom and are carried as a REPORTED STRATUM, not dropped; "
            f"first: {inside[:5]}. No success@tau claim at that threshold is available for them -- "
            "see cryptic_rmsd/objective.POOL_GATE_ANGSTROM",
            flush=True,
        )
    del runtime
    return reference


def _adapter_arguments(arm: Any) -> Mapping[str, Any]:
    """No per-residue conditioning channel.

    There is deliberately no pocket-RMSD equivalent of the cryptic ``site`` field, and that is a
    departure from ``pocket_progress_coordinate``'s arms A and B rather than an oversight. A
    channel built from the pocket residues would tell the adapter WHERE the pocket is, and the
    direct metric would then be satisfied partly by a hand-supplied answer; whether the adapter can
    find the pocket is the question. Arm C -- ``site.source: null`` -- is the cryptic control that
    already runs unconditioned, and this property's first arms are its counterpart.

    It is also what keeps this property's composition free of ``cryptic_generation_shared``: the
    site cache is built by ``cryptic_steering/site.py`` off the fingerprinted pocket artifact. If a
    channel is ever wanted it is a new arm, declared here, with the group named -- not a default.
    """
    del arm
    return {}


def _sample_definition_arguments(arm: Any) -> Mapping[str, Any]:
    """Hand the resolved reference over under the SHARED ``definition`` keyword.

    The type is this package's own
    :class:`~sae_bioemu.cryptic_rmsd.artifact.PocketRmsdDefinition` rather than a class added to
    ``bioemu/steering_properties.py``: that file is in ``generation_shared`` and a fifth definition
    class there would re-key every existing steering checkpoint for a type none of them constructs.
    """
    from sae_bioemu.cryptic_rmsd.artifact import PocketRmsdDefinition

    return {} if arm is None else {"definition": PocketRmsdDefinition(reference=arm)}


def _measure_sample(*, batch: Any) -> Any:
    """Measure each steered draw's superposed N/CA/C/O deviation over its own metric region.

    The whole objective, in the shape the shared ``SteeringLoss`` prices::

        prediction = m + s_p         the measured deviation, placed on this protein's own origin
        target     = s_p             supplied by the system-constant table
        scale      = s_p             supplied by the same table

    so ``mean_n ((prediction - target)/scale)^2`` IS ``mean_n (m_n/s_p)^2``, with no threshold in
    it. **The shift is the objective's zero**, and it is re-checked elementwise by
    ``PocketRmsdSteeringLoss.forward`` against the unclamped ``m`` on the wire rather than trusted
    from this line, because a missing shift and a swapped origin both produce a plausible loss
    curve and no other symptom.

    The UNCLAMPED ``m`` travels beside it on ``metric_deviation_angstrom`` -- the shared TRANSPORT
    field -- and is journalled under this property's own column name.

    **The matched unsteered control is NOT read here, and that is the difference from
    ``pocket_progress_coordinate``.** That property's ``_measure_sample`` raises "needs the matched
    unsteered draw as its reference", because ``the progress coordinate = <d(scored) - d(control), v>_w / ||v||_w^2`` is
    a displacement whose origin IS the control. This deviation is measured from the STEERED draw's
    own frames to a DEPOSITED structure, so it is defined from one structure and the control is an
    instrument rather than an anchor -- which is what lets :func:`_control_measurement` decline it.

    ``bioemu.convert_chemgraph`` is reached through this package's own ``frames`` module rather
    than through ``bioemu/cryptic_generation.py``: that file is the sole member of the code group
    both PROGRESS-COORDINATE properties compose, so extending it to return O would re-key every
    cryptic and DFG-coordinate checkpoint for an atom no progress-coordinate term reads.
    """
    from sae_bioemu.cryptic_rmsd.artifact import PocketRmsdDefinition, pocket_rmsd_metric_batch
    from sae_bioemu.cryptic_rmsd.frames import batched_backbone_atoms
    from sae_bioemu.steering.property_registry import PropertyMeasurement

    batch.require_relative_levels(POCKET_RMSD_PROPERTY_NAME)
    definition = batch.definition
    if not isinstance(definition, PocketRmsdDefinition):
        raise ValueError(
            f"{POCKET_RMSD_PROPERTY_NAME} needs its resolved reference: the metric is a superposed "
            "N/CA/C/O deviation from this protein's own deposited apo and holo conformers, and "
            "neither an adapter nor a structure supplies them"
        )
    reference = definition.reference
    sequences = [item.sequence for item in batch.records]
    backbone = batched_backbone_atoms(batch.steered.coordinates, batch.node_orientations, sequences)
    sequence_ids = batch.sequence_ids
    metric, _priced, _separation = pocket_rmsd_metric_batch(
        reference, sequence_ids, sequences, batch.levels, backbone
    )
    # The per-protein origin, elementwise. Built as a TENSOR of per-member constants rather than
    # one scalar because one protein's origin applied to another's draw is a finite, plausible and
    # wrong loss -- and ``PocketRmsdNormalisation.target_constant_angstrom`` is the one place the
    # shift is defined, so this path and :func:`system_target_constants` cannot disagree about it.
    constants = metric.new_tensor(
        [reference.require(name).normalisation.target_constant_angstrom() for name in sequence_ids]
    )
    return PropertyMeasurement(
        prediction=metric + constants,
        target=batch.target,
        sample_fields={"metric_deviation_angstrom": metric},
    )


def _checkpoint_arm_arguments(arm: Any) -> Mapping[str, Any]:
    """Bind the resolved reference into the CHECKPOINT and nothing else.

    Under the GENERIC ``property_arm_identity`` field rather than a cryptic one, and deliberately
    NOT ``cryptic_arm_identity``: that field is also what ``steers_cryptic_pocket`` is derived
    from, so a run binding its reference there would acquire the four D-35b pocket monitor columns
    and the realised-alignment monitor on every validation row -- all of which are computed from a
    matched control and a pair set this property has neither of.

    Deliberately not in ``schedule_seed_fingerprint``: routing an artifact digest through the seed
    would redraw every schedule and confound each cross-arm difference with a complete
    re-randomisation, whose measured noise floor (sd 0.0057 on the held-out dose slope) exceeds
    several differences worth claiming.
    """
    if arm is None:
        return {}
    return {
        "property_arm_identity": arm.identity(),
        # The per-system ``{target, scale}`` table the shared target construction reads. It rides
        # this hook -- documented as "extra ``OrdinarySteeringConfig`` keyword arguments the
        # resolved arm supplies" -- rather than a hook of its own, because that is exactly what it
        # is: a config field whose value only the resolved arm can produce. Building it here also
        # means the numbers the objective descends and the numbers the checkpoint identity records
        # come from ONE call on ONE reference.
        "system_target_constants": system_target_constants(arm),
    }


def _validation_row_columns(*, objective: Any, config: Any) -> Mapping[str, JournalledSampleColumn]:
    """The held-out deviation, in Angstrom, on every pocket-RMSD validation row.

    Ungated, because for this property the deviation is not an added term whose weight might be
    zero: it IS the objective, by construction of ``prediction``, ``target`` and ``property_scale``,
    and ``PocketRmsdSteeringLoss`` refuses ``lambda_m = 0`` as a driverless loss before any draw.
    The TRAINING column gates on ``lambda_m``; this one does not, because a validation draw
    computes no loss at all.
    """
    del objective, config
    return {_VALIDATION_COLUMN: JournalledSampleColumn(field="metric_deviation_angstrom")}


def _validation_selection_metric(*, objective: Any, arm: Any) -> ValidationSelectionMetric | None:
    """Always the mean held-out deviation. See :data:`_SELECTION_METRIC_NAME`.

    Unconditional rather than gated on a weight, for the reason :func:`_validation_row_columns` is:
    there is no configuration of this property under which the target error is the better
    selection signal, because the target error IS the deviation divided by a per-protein constant.

    This is the DOMAIN-MOTION answer and not the kinase-RMSD one. ``kinase_conformer_rmsd`` returns
    ``None`` because its arms run ``model_selection: none`` over a 41-gene training pool and
    ``OrdinarySteeringConfig`` refuses a declared metric there -- "it would name a comparison this
    run never makes". This pool is large enough to hold a validation role out of training, the
    arms below run ``validation_split``, and the column is written on every held-out row.
    """
    del objective, arm
    return ValidationSelectionMetric(name=_SELECTION_METRIC_NAME, row_column=_VALIDATION_COLUMN)


def system_target_constants(reference: Any) -> SystemTargetConstants:
    """Build the per-protein ``{target, scale}`` table this property's targets are made of.

    * ``target`` is this protein's own ``s_p``, from
      :meth:`~sae_bioemu.cryptic_rmsd.objective.PocketRmsdNormalisation.target_constant_angstrom`.
      The objective's zero is at ``m = 0``, so the target is a pure origin the prediction carries
      too and which therefore cancels exactly. A benchmark ``tau`` is deliberately NOT used here:
      it would enter this table's digest and make the 1.0 A and the 1.5 A readings of one run two
      readings of two differently-bound checkpoints.
    * ``scale`` is the same ``s_p``: the superposed N/CA/C/O separation between the two deposited
      conformers over this protein's metric region.

    Both are re-derived from the reference's own coordinates at load, so this table cannot drift
    away from the crystals the metric is measured against. The table's digest reaches the
    checkpoint identity, so a run cannot resume across two references over one split.
    """
    return SystemTargetConstants(
        entries={
            name: SystemTargetConstant(
                target=entry.normalisation.target_constant_angstrom(),
                scale=float(entry.separation_angstrom),
            )
            for name, entry in reference.entries.items()
        }
    )


def _control_measurement(arm: Any) -> ControlMeasurement:
    """DECLINE the matched unsteered control by default, and price nothing with it either way.

    **This is the domain-motion shape of control and deliberately not the cryptic one.** On
    ``pocket_progress_coordinate`` the control IS the origin -- ``the progress coordinate`` is measured as a
    displacement from it, so ``_control_measurement`` there is unconditional and
    ``_measure_sample`` raises "needs the matched unsteered draw as its reference" without one.
    Under :data:`~sae_bioemu.steering.schedule.TARGET_ORIGIN_SYSTEM_CONSTANT` the origin is this
    protein's own ``s_p``, which comes off the reference artifact, so the control is an UNPRICED
    INSTRUMENT and an arm can decline it and still have a target.

    ``atom_set`` is ``None`` rather than the cryptic ``'cb'``, and that is not a relaxation of
    D-69: the C-beta reading exists because ``the progress coordinate`` projects a realised distance change onto a
    C-beta transition vector, and there is no such projection here. This deviation is read off the
    STEERED draw's N/CA/C/O only.

    WHAT DECLINING IT COSTS: the PAIRED primary -- per draw, the fraction of that protein's own gap
    the intervention closed against an unsteered control generated from the same ``draw_seed`` --
    becomes uncomputable ON THE TRAINING STORE, and nothing else. Selection does not read it
    (:func:`_validation_selection_metric` selects on the deviation itself), no weighted term
    consumes it, and the benchmark generator writes its own matched control at the same
    ``draw_seed`` regardless -- so the paired number is still available from the store any reported
    figure would come from.

    WHY THE DEFAULT IS OFF: the control roughly doubles training generation, and for the
    per-direction unconditioned arms of D-101 it more than doubles it. The saving that makes it
    "somewhat better than 2x" -- one control serving both levels of a (sequence, cycle) -- exists
    only because a level-conditioned arm visits the same ``draw_seed`` twice. Split the arms and
    each visits it once, so the ratio goes from 1 control per 2 steered draws to 1 per 1. Turning
    it off is what keeps a pair of per-direction arms cheaper than the single conditioned arm they
    replace instead of 1.14x its cost.

    An artifact that wants it back sets ``control_required = True``; ``frames`` stays ``False``
    either way, because frames would be read off the control for a reconstruction nothing on this
    path performs.
    """
    return ControlMeasurement(required=bool(getattr(arm, "control_required", False)))


POCKET_RMSD_PROPERTY = SteerableProperty(
    name=POCKET_RMSD_PROPERTY_NAME,
    # TWO-STATE. The two levels are not a signed dose but a CHOICE OF DEPOSITED CONFORMER, at the
    # shared draw seed. ``TWO_STATE_DOSE_LEVELS`` is taken exactly as it stands -- no schedule
    # change, no new dose set -- because (-1, +1) is already the right cardinality and the right
    # pairing; only the reading changes, and ``success@B`` already scores apo and holo separately.
    dose_levels=TWO_STATE_DOSE_LEVELS,
    # ``t = s_p`` at every level, with the level selecting which deposited conformer the deviation
    # is measured to. The SHARED generic origin, which ``domain_motion_backbone_rmsd`` and
    # ``kinase_conformer_rmsd`` already run on -- and which the domain-motion descriptor's own
    # comment predicted "the cryptic direct-metric term is queued for".
    target_origin=TARGET_ORIGIN_SYSTEM_CONSTANT,
    two_state=True,
    property_callback_identity="pocket_conformer_rmsd_v1",
    # Its own marker, not a bumped cryptic one: a shared version would say that a pocket-RMSD
    # checkpoint and a ``pocket_progress_coordinate`` one came from the same generation callback
    # when they did not -- the two measure different quantities off different atom sets and build
    # their targets on different origins.
    generation_callback_version="bioemu_pocket_conformer_rmsd:v1",
    loss_class=_loss_class,
    runtime_objective_arguments=_objective_arguments,
    resolve_arm=resolve_pocket_rmsd_property_arm,
    adapter_arguments=_adapter_arguments,
    sample_definition_arguments=_sample_definition_arguments,
    checkpoint_arm_arguments=_checkpoint_arm_arguments,
    refused_runtime_keys=_REFUSED,
    # No scorer, and deliberately NOT wired to ``pocket_progress_coordinate``'s: that one scores
    # ``the progress coordinate`` against a pair set and a transition vector, in different units, and takes a resolved
    # cryptic arm this property never builds. A shared scorer is how one property silently answers
    # another's question. The registry refuses ``evaluate`` by name for a property with none, which
    # is the correct failure until one exists.
    evaluator=None,
    measure_sample=_measure_sample,
    validation_row_columns=_validation_row_columns,
    validation_row_vocabulary=(_VALIDATION_COLUMN,),
    validation_selection_metric=_validation_selection_metric,
    control_measurement=_control_measurement,
)

#: Restated for the report and for tests: the numbers that decide which proteins are admitted at
#: all, and the thresholds this property's results are REPORTED at but never trained against.
POCKET_RMSD_CONSTANTS = {
    "benchmark_tau_angstrom": BENCHMARK_TAU_ANGSTROM,
    "pool_gate_angstrom": POOL_GATE_ANGSTROM,
    "minimum_separation_angstrom": MINIMUM_SEPARATION_ANGSTROM,
    "journal_column": POCKET_RMSD_JOURNAL_COLUMN,
}
