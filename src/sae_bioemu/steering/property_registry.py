"""The steerable-property registry: the dispatcher, with no property's data inside it.

``bioemu/steering_workflow.py`` used to hold thirteen tables and branches keyed by property
name -- the dose set, the target origin, the two callback versions, the evaluator, the arm
resolver, the loss class, the refused runtime weights, the fingerprinted file list and the
fingerprint composition. Every one of them lived in a file that is itself a member of the
SHARED ``training_only`` code group, so a kinase-only edit moved the radius-of-gyration
training digest and retired every Rg checkpoint. That is exactly the defect D-43 was meant to
end, and it survived D-43 because the code performing the per-property SPLIT sat inside the
shared half.

This module is the fix. It holds the MECHANISM -- what a steerable property must declare, how
a declaration is found, and how the code groups compose -- and not one property's answers.
Adding a fourth property therefore edits:

* that property's own package, which carries its descriptor; and
* :mod:`sae_bioemu.steering.property_catalog`, which is deliberately NOT fingerprinted.

and nothing that any existing checkpoint is bound to.

WHY AN UNFINGERPRINTED CATALOG IS NOT FAIL-OPEN
-----------------------------------------------

The catalog is the one file a new property edits, so it cannot be hashed into the shared groups
without reintroducing the very re-key this exists to prevent. It is nevertheless unable to
change a run's meaning without moving that run's digest, for two independent reasons:

1. **Group membership enters every digest BY VALUE.** :func:`scientific_code_paths` returns a
   ``{group: {member name: content digest}}`` map and ``_code_fingerprints`` hashes the map
   itself. Adding, removing or renaming a member changes the hashed payload, so a catalog edit
   that dropped a file from a group would move that property's digest rather than silently stop
   covering the file. The catalog cannot hash less without saying so.
2. **A descriptor module must hash itself.** :func:`resolve` refuses a descriptor whose own
   source file is not among its property's own training-group members. Re-pointing a name at a
   different descriptor module therefore changes which file is hashed, and so changes the
   digest. A descriptor cannot be swapped underneath a name.

What the catalog holds is, by construction, only what fails loudly: names, the spelling of a
config block, and file lists. Every scientific quantity -- the dose set, the target origin, the
callback versions, the two-state declaration, the loss class, the refused weights -- lives in
the descriptor, in the property's own package, inside the property's own fingerprint group.

THIS MODULE IS FINGERPRINTED, IN ``training_only``
---------------------------------------------------

It is the dispatcher every steered run executes, so an edit to it moves every property's
training digest -- the same blast radius ``steering_workflow`` has, and the correct one for a
shared mechanism. The point of the split is that adding a property or a loss term no longer
requires such an edit.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, and deliberately not imported at runtime
    from sae_bioemu.steering.schedule import DoseLevels
    from sae_bioemu.steering.training import SteeringLoss

__all__ = [
    "ControlMeasurement",
    "JournalledSampleColumn",
    "PropertyMeasurement",
    "PropertyRegistration",
    "SteerableProperty",
    "ValidationSelectionMetric",
    "arm_config_keys",
    "contested_journal_weights",
    "journal_column_names",
    "journal_column_owners",
    "journal_column_weights",
    "monitor_column_names",
    "optional_runtime_key_owners",
    "property_code_groups",
    "registered_property_names",
    "resolve",
    "resolve_config_path",
    "scientific_code_paths",
    "selection_metric_names",
    "shared_code_paths",
]


def resolve_config_path(base: Path, value: object, name: str) -> Path:
    """Resolve a configured path against the config file's own directory.

    Lives here rather than in the workflow because the per-property arm resolvers -- which are
    now in their own packages -- need it, and duplicating it is how two files come to disagree
    about whether a relative path is relative to the config or to the process.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


@dataclass(frozen=True)
class JournalledSampleColumn:
    """One per-observation column a property journals straight off its generated sample.

    The VALIDATION-row half of the per-property journal vocabulary. A training-record column is
    gated on a weight the objective prices and comes out of the loss components
    (:func:`~sae_bioemu.steering.ordinary_workflow._off_path_scalars`); a column declared here is
    gated on nothing but this property's own hook and is read directly off
    :class:`~sae_bioemu.steering.ordinary_workflow.GeneratedTrainingSample`, because a validation
    draw computes no loss at all.

    It exists so that the extra-column decision stops being a hardcoded
    ``config.steers_cryptic_pocket``. That predicate gave the cryptic pocket its four D-35b
    geometry counts on every validation row and gave the kinase programme nothing -- so no kinase
    arm has ever been observed on held-out ``d_K`` (K-21 4.1, K-22 caveat 2), and the last arm was
    undiagnosable for exactly that reason.
    """

    #: The :class:`GeneratedTrainingSample` field this column is read from. Several columns may
    #: name one field, which is how a WIDENED channel (kinase ``dihedral_distance`` carrying
    #: ``(d_K, H)`` rather than ``d_K``) yields two columns off one wire.
    field: str
    #: Which component of a widened ``[members, k]`` channel this column is, or ``None`` for a
    #: plain ``[members]`` one. Declared rather than inferred from the tensor's shape: a channel
    #: that silently changed width would otherwise start writing a different quantity under the
    #: same column name.
    component: int | None = None
    #: True when the value is a COUNT and a non-integral value means something upstream reduced
    #: or interpolated it. The D-35b counts are the case this exists for -- ``1`` clash read as
    #: ``0.5`` defeats the entire reason D-35b counts rather than averages.
    integral: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field:
            raise ValueError("a journalled sample column must name the sample field it reads")
        if self.component is not None and (
            isinstance(self.component, bool) or not isinstance(self.component, int)
        ):
            raise ValueError("a journalled sample column component must be an integer or None")
        if self.component is not None and self.component < 0:
            raise ValueError("a journalled sample column component must be nonnegative")


@dataclass(frozen=True)
class ControlMeasurement:
    """What the matched UNSTEERED control must be generated as, for one property.

    The shared workflow used to answer this with two hardcoded expressions: ``arm is not None``
    for "is a control required", and ``arm.artifact.atom_set`` for "in which atoms". Both are
    statements about the cryptic and kinase arms wearing a shared file's clothes, and the second
    is a latent ``AttributeError`` for any property whose resolved arm is not an object with an
    ``.artifact.atom_set`` -- which the domain-motion artifact is not, so the first domain-motion
    run would have crashed inside the reference cache after loading the sampler.
    """

    #: True when the control is the property's OWN ANCHOR rather than an optional term's input.
    #: A progress coordinate is ``u = d(scored) - d(reference)``, so without a control there is no
    #: coordinate at all -- as opposed to ``lambda_l``, where there is merely no penalty.
    required: bool = False
    #: The ``bioemu.cryptic_generation`` atom set the control must ALSO be read in, or ``None``
    #: when C-alpha is the only reading this property takes. ``'ca'`` short-circuits to the
    #: C-alpha tensor inside the helper, so declaring it is the same as declaring ``None`` for the
    #: generated bytes and different for what the property is saying.
    atom_set: str | None = None
    #: True when the control's own per-residue FRAMES -- ``(positions, node_orientations)`` -- must
    #: reach the measurement, not just a placed atom set.
    #:
    #: The two existing atom readings are both derived quantities: ``ca`` is ``pos`` and ``cb`` is
    #: one virtual atom per residue. A property whose quantity is read off N/CA/C/O, or off the
    #: backbone TORSIONS, cannot be served by either, because those atoms are placed by composing
    #: each residue's rigid transform with ideal geometry and the transform is exactly what a
    #: placed-atom tensor has thrown away.
    #:
    #: **It is free wherever it is asked for today.** ``_reference_pair`` already calls
    #: ``generate_unsteered_frames`` for every property declaring a non-C-alpha ``atom_set`` and
    #: then DISCARDS the orientations; declaring this keeps them. The kinase programme is that
    #: case exactly: its control is generated in ``cb`` already, so the matched-control ``d_K``
    #: -- which K-22 names as the quantity that would separate "the motif moved" from "the
    #: absolute distance was priced" -- costs no extra denoiser pass, only this declaration.
    frames: bool = False

    def __post_init__(self) -> None:
        if self.atom_set is not None and (not isinstance(self.atom_set, str) or not self.atom_set):
            raise ValueError("a control measurement atom set must be a non-empty string or None")
        if self.frames and not (self.required or self.atom_set is not None):
            raise ValueError(
                "a control measurement that asks for frames must also say the control exists: "
                "frames are read off the SAME generated draw, so asking for them without "
                "requiring a control or naming an atom set describes a structure that is "
                "generated only when an unrelated weight happens to be priced"
            )


@dataclass(frozen=True)
class ValidationSelectionMetric:
    """Which quantity a run picks its shipped epoch on, when it is not the shared default.

    **A selection metric is a scientific decision, and until this existed it was a property of the
    TARGET KIND rather than of the property.** ``bioemu/steering_workflow.py`` resolved it as
    ``RELATIVE_SELECTION_METRIC if relative else ABSOLUTE_SELECTION_METRIC``, i.e. every
    relative-path run selected on ``target_mae_sigma`` -- the sigma-normalised error on the
    steered PROPERTY. That is exactly right when the property is the thing being claimed, which it
    is for radius of gyration and for both progress coordinates.

    It is exactly WRONG for a DIRECT-METRIC arm. Such an arm exists to remove a proxy: it trains on
    the deviation the benchmark grades and may price the progress coordinate at ``lambda_y = 0``.
    Under the old expression it would then train on the metric and choose its shipped epoch on the
    proxy -- reintroducing at model selection the very contamination the arm was built to remove,
    invisibly, in a field of the run report that reads as boilerplate.

    Two names and nothing else, so it can live where the workflow can read it without importing a
    property package:

    * :attr:`name` is the aggregate key in the persisted validation record AND the string the run
      report and ``adapter_config`` record, so the checkpoint says what it was selected on.
    * :attr:`row_column` is the per-row validation column :attr:`name` is the MEAN of. It must be a
      column the property's own ``validation_row_columns`` hook actually declares for THIS
      objective; the workflow checks that before a GPU is allocated and refuses by name rather than
      falling back to ``target_mae_sigma``, which would be indistinguishable from working.

    Lower is better, always. Every selection metric in this workflow is an error or a deviation and
    ``_validation_key`` minimises it; a metric that wanted maximising would be a different
    comparison and would have to say so here rather than be inferred from its sign.
    """

    #: The aggregate/report/identity spelling.
    name: str
    #: The per-observation validation-row column it averages.
    row_column: str

    def __post_init__(self) -> None:
        for field_name, value in (("name", self.name), ("row_column", self.row_column)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"a validation selection metric must declare a {field_name}")


def _no_validation_selection_metric(**_: Any) -> ValidationSelectionMetric | None:
    """The shared default: select on the target error, as every run made before this hook did.

    ``None`` means "the historical answer for this target kind" -- ``target_mae_sigma`` on the
    relative path and ``target_mae`` on the absolute one -- and it is what radius of gyration, both
    progress coordinates and every arm on disk resolve to. Returning a name here instead would
    write that name into ``adapter_config``, which is a persisted identity key, so the default
    stays absent rather than becoming explicit.
    """
    return None


def _no_control(arm: Any) -> ControlMeasurement:
    """The shared default: no control is generated unless an optional term asks for one.

    Radius of gyration's answer. Its property is computed from one structure, so the matched
    unsteered draw is ``lambda_l``'s input and nothing else's.
    """
    del arm
    return ControlMeasurement()


@dataclass(frozen=True)
class PropertyMeasurement:
    """What ONE property reads off ONE generated batch.

    The return type of :attr:`SteerableProperty.measure_sample`.

    Three fields, because three things vary per property and nothing else does. The residual
    statistics, the geometry penalty and the local-structure term are read off the STEERED SAMPLE
    by the shared caller and are identical for every property that returns one value per batch
    member, so they are deliberately absent here: a property that could restate them could also
    contradict them.

    ``local_q`` is the one property that reshapes the residual statistics -- it predicts one value
    per RESIDUE, so ``residual_rms`` is broadcast and masked -- and it is also the one property
    with no descriptor. That is why it stays a literal branch in
    :func:`~sae_bioemu.bioemu.steering_properties.generate_exact_property_sample` rather than being
    forced through this contract; see that function.
    """

    #: The differentiable property value, ``[members]``. Becomes ``surrogate_prediction``.
    prediction: Any
    #: The objective target ECHOED BACK to the caller. On the relative path this is the
    #: conditioning level and the real target is built by the workflow from the schedule; on the
    #: absolute path it is the standardized target itself. Returned rather than assumed so a
    #: property whose target is a function of the batch -- a per-system constant, say -- says so.
    target: Any
    #: Extra per-observation ``GeneratedTrainingSample`` fields this property's objective and
    #: journal read, by field name.
    #: EMPTY for a property that adds none, which is what keeps such a run's sample field-for-field
    #: what it was before this hook existed.
    sample_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_fields, Mapping):
            raise TypeError("a property measurement's sample_fields must be a mapping")


def _no_journalled_sample_columns(**_: Any) -> Mapping[str, JournalledSampleColumn]:
    """The shared default: a property journals no extra validation column.

    Radius of gyration's answer, and the answer for any property that has not declared one, so a
    run made before this hook existed writes the byte-identical validation row.
    """
    return {}


@dataclass(frozen=True)
class SteerableProperty:
    """Everything the shared workflow needs to know about ONE steerable property.

    Every field is a fact about the property rather than about the workflow, and every one of
    them used to be a row in a table in ``bioemu/steering_workflow.py``. A descriptor is built
    in the property's own package and hashed in that property's own code group, so changing any
    of these re-keys that property and no other.

    The callables are stored as plain functions, not methods: they are called as
    ``descriptor.loss_class()`` and receive no ``self``. That is deliberate -- a property's
    behaviour is a set of functions the registry calls, not a subclass hierarchy the registry
    has to know the shape of.
    """

    #: The ``target.property_name`` this descriptor answers for. Checked against the name the
    #: catalog routed, so a mis-routed catalog entry fails closed rather than steering the
    #: wrong quantity.
    name: str
    #: The steered dose set (D-44). Continuous for radius of gyration; two-state for a property
    #: whose level names a STATE.
    dose_levels: DoseLevels
    #: Where a level's target sits (D-46): the sequence mean, or the matched control's state.
    target_origin: str
    #: Whether the level names a STATE rather than a number of standard deviations. Gates the
    #: ``dose_multiplier`` key, the absolute-target refusal, and the reference-table report.
    two_state: bool
    #: ``property_callback_identity``, verbatim in the checkpoint identity.
    property_callback_identity: str
    #: The generation-callback marker concatenated with the training code fingerprint.
    generation_callback_version: str
    #: Returns the objective CLASS this property constructs. A callable so the import stays
    #: deferred: a radius-of-gyration run must not import the cryptic or kinase loss.
    loss_class: Callable[[], type[SteeringLoss]]
    #: ``(*, runtime, arm) -> mapping`` of the constructor arguments beyond the shared ones.
    #: This is where a property turns its runtime weights into its objective's arguments, so
    #: enabling a term that the shared loss vocabulary already carries is a change HERE and in
    #: a YAML, never in a shared file.
    runtime_objective_arguments: Callable[..., Mapping[str, Any]]
    #: ``(*, payload, base, split, runtime) -> arm or None``. Runs this property's config-time
    #: gates before anything generates.
    resolve_arm: Callable[..., Any]
    #: Extra ``BoundedResidualAdapter`` keyword arguments the resolved arm supplies.
    adapter_arguments: Callable[[Any], Mapping[str, Any]]
    #: Extra ``generate_exact_property_sample`` keyword arguments the resolved arm supplies.
    sample_definition_arguments: Callable[[Any], Mapping[str, Any]]
    #: Extra ``OrdinarySteeringConfig`` keyword arguments the resolved arm supplies.
    checkpoint_arm_arguments: Callable[[Any], Mapping[str, Any]]
    #: Runtime weights this property prices NOWHERE, by key, with the reason. A nonzero value
    #: for one of these is refused by name before a GPU is allocated, rather than read from the
    #: config, dropped by the loss constructor and absent from the objective identity -- the
    #: shape that voided six radius-of-gyration arms.
    refused_runtime_keys: Mapping[str, str]
    #: Returns this property's held-out scorer, or ``None`` when it has none. A property with
    #: no scorer is REFUSED by name at ``evaluate`` rather than routed to another property's:
    #: a scorer measures a quantity against a tolerance in that quantity's own units, and a
    #: shared scorer is how one property silently answers another's question.
    evaluator: Callable[[], Any] | None = None
    #: The keyword the resolved arm reaches the scorer under, or ``None`` when the scorer takes
    #: no arm. The shared scorer arguments are fixed and identical for every property; this is
    #: what lets one scorer require something another cannot accept, because the checkpoint
    #: carries the arm's IDENTITY and not the artifact, the sigma table or the split record.
    evaluator_arm_keyword: str | None = None
    #: ``(*, batch) -> PropertyMeasurement``: turn ONE generated batch into this property's
    #: prediction, its objective target and whatever extra per-observation fields its objective
    #: reads off the sample. ``None`` for a property whose measurement is still a literal branch
    #: in ``bioemu/steering_properties.py``.
    #:
    #: This is the hook that ended the per-property branch chain in
    #: :func:`~sae_bioemu.bioemu.steering_properties.generate_exact_property_sample`. That chain
    #: dispatched on ``adapter.property_name`` and took one keyword argument per property, in a
    #: file every property's ``generation_shared`` group hashes -- so a fourth coordinate could
    #: not be measured at all without re-keying the three that price none of its geometry.
    #:
    #: ``batch`` is a :class:`~sae_bioemu.bioemu.steering_properties.GeneratedBatch`: the steered
    #: sample, the records, the mask, the conditioning target, this property's own resolved
    #: definition and the matched control in each reading the property asked
    #: :attr:`control_measurement` for. It is ONE argument rather than eight keywords precisely
    #: because widening a keyword list is an edit to the shared file this hook exists to stop
    #: editing.
    measure_sample: Callable[..., Any] | None = None
    #: ``(*, objective, config) -> {column name: JournalledSampleColumn}``: which extra columns
    #: this property's VALIDATION rows carry, for this objective and this configuration.
    #:
    #: Per property rather than the hardcoded ``config.steers_cryptic_pocket`` predicate it
    #: replaces, and the difference is not tidiness: under that predicate the kinase programme
    #: journalled no ``d_K`` on any held-out draw, so an arm that minimised its objective while
    #: its signed error ROSE (K-22) had nothing recorded that could say which of the two moved.
    validation_row_columns: Callable[..., Mapping[str, JournalledSampleColumn]] = (
        _no_journalled_sample_columns
    )
    #: Every column name :attr:`validation_row_columns` could EVER return, whatever the arm or the
    #: weights. The static half of that hook, and it exists for the readers that have no live
    #: objective in hand: the completion gate validates a persisted checkpoint's rows against an
    #: ``adapter_config``, so it can name the property but cannot reconstruct the objective. It
    #: admits exactly this vocabulary as surplus, which keeps the gate fail-closed against an
    #: arbitrary extra key while not falsely refusing a column a weight switched on.
    #:
    #: ``validation_row_columns`` is required to return a subset of it, so the two cannot drift.
    validation_row_vocabulary: tuple[str, ...] = ()
    #: ``(*, objective, config) -> {column name: JournalledSampleColumn}``: which UNPRICED MONITOR
    #: columns this property's TRAINING rows carry. The training-row twin of
    #: :attr:`validation_row_columns`, and it exists for a defect the validation hook cannot reach.
    #:
    #: A column declared in :attr:`PropertyRegistration.journal_columns` is gated on the WEIGHT
    #: that prices it, which is right for a loss term and wrong for a measurement: the cryptic
    #: direct-metric arm runs at ``lambda_perp = 0``, so ``perp_loss`` is never computed and the
    #: arm that could falsify D-77's ``|cos| = 0.165`` ceiling records nothing that would falsify
    #: it. A monitor has no weight, because it is not in the objective at all.
    #:
    #: Every column here is read off the generated SAMPLE, never off the loss components, and is
    #: priced nowhere: it does not enter ``total``, it does not appear in the objective identity,
    #: and no selection metric, gate or aggregate reads it.
    training_row_columns: Callable[..., Mapping[str, JournalledSampleColumn]] = (
        _no_journalled_sample_columns
    )
    #: Every column name :attr:`training_row_columns` could EVER return. The static half of that
    #: hook, for the same readers :attr:`validation_row_vocabulary` exists for.
    training_row_vocabulary: tuple[str, ...] = ()
    #: ``(arm) -> ControlMeasurement``: whether the matched unsteered control is this property's
    #: own anchor, and which atom set it has to be read in. Replaces the shared workflow's
    #: ``arm is not None`` and ``arm.artifact.atom_set``, neither of which was a fact about the
    #: workflow.
    control_measurement: Callable[[Any], ControlMeasurement] = _no_control
    #: ``(*, objective, arm) -> ValidationSelectionMetric | None``: which validation column this
    #: run picks its shipped epoch on, or ``None`` for the shared default.
    #:
    #: Takes the OBJECTIVE and the resolved ARM rather than the config, unlike
    #: :attr:`validation_row_columns`, and the difference is an ordering fact rather than a taste:
    #: the answer IS a config field, so it has to be resolved before the config exists. The
    #: objective and the arm are both built first and between them carry every weight and every
    #: gate a selection choice could turn on.
    #:
    #: On the descriptor rather than on :class:`PropertyRegistration` because the answer needs
    #: BEHAVIOUR and not a name: whether a direct-metric arm selects on its metric depends on
    #: whether that arm PRICES the metric, which is a weight on the objective. The two spellings a
    #: run could ever produce are declared import-lightly beside the registration, as
    #: :attr:`PropertyRegistration.selection_metric_names`, for the readers that hold a persisted
    #: checkpoint and no objective.
    validation_selection_metric: Callable[..., ValidationSelectionMetric | None] = (
        _no_validation_selection_metric
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a steerable property must declare its name")
        if self.evaluator is None and self.evaluator_arm_keyword is not None:
            raise ValueError(
                f"{self.name!r} names a scorer arm keyword but has no scorer to pass it to"
            )
        if self.validation_row_columns is not _no_journalled_sample_columns and not (
            self.validation_row_vocabulary
        ):
            raise ValueError(
                f"{self.name!r} declares validation-row columns but no vocabulary; the readers "
                "that validate a persisted checkpoint have no objective in hand and admit "
                "exactly the declared vocabulary as surplus, so a hook without one would make "
                "its own rows unverifiable"
            )
        if self.training_row_columns is not _no_journalled_sample_columns and not (
            self.training_row_vocabulary
        ):
            raise ValueError(
                f"{self.name!r} declares training-row monitor columns but no vocabulary; a hook "
                "without one would make its own rows unverifiable, for the reason the "
                "validation-row vocabulary exists"
            )


@dataclass(frozen=True)
class PropertyRegistration:
    """The IMPORT-LIGHT half of a registration: where the descriptor is, and what it hashes.

    Split from :class:`SteerableProperty` because :func:`scientific_code_paths` runs on EVERY
    steered run and needs every property's file list, while a descriptor may only be imported
    for the property actually being steered -- importing ``cryptic_steering`` from a
    radius-of-gyration run would pull the pocket geometry, the stage-1 reader and (through
    ``bioemu.cryptic_generation``) OpenFold into a run that prices none of them.

    Nothing here is a scientific quantity. ``arm_config_key`` is the spelling of a config
    block: change it and every config naming the old spelling fails to load. The group members
    and the composition enter every digest by value, so changing them moves the digest they
    describe. See the module docstring.
    """

    #: ``target.property_name``.
    name: str
    #: The module holding the :class:`SteerableProperty`, imported only when this property runs.
    descriptor_module: str
    #: The attribute name inside that module.
    descriptor_attribute: str
    #: The ``target`` block key naming this property's arm configuration, or ``None`` for a
    #: property with no arm. REQUIRED for its own property and refused for every other one.
    arm_config_key: str | None
    #: This property's own fingerprinted sources, ``{group: {member name: path}}``.
    code_group_members: Mapping[str, Mapping[str, Path]]
    #: Which groups compose this property's two fingerprints, ``{"training": (...),
    #: "evaluation": (...)}``. Shared groups are named here too: the composition is the whole
    #: statement of what a run executes.
    code_group_composition: Mapping[str, tuple[str, ...]]
    #: ``{persisted journal column: the objective weight that gates it}`` for the columns this
    #: property's objective reports from
    #: :meth:`~sae_bioemu.steering.training.SteeringLossComponents.journal_columns`.
    #:
    #: **Here rather than in the descriptor, and that is a considered placement.** It has to be
    #: import-light: ``ordinary_workflow`` needs the UNION of every property's columns to parse a
    #: journal whose property it does not yet know, and importing ``cryptic_steering.property``
    #: to find that out would pull the pocket geometry and OpenFold into a radius-of-gyration
    #: run. It is also exactly the kind of thing this file is allowed to hold: a column NAME and
    #: the spelling of a weight, both of which fail loudly. ``_off_path_scalars`` refuses a
    #: column the union does not name; ``_validate_training_record_fields`` refuses a row whose
    #: column presence disagrees with the weight named here. A declaration naming the wrong
    #: weight therefore refuses rows rather than mislabelling them, and a spurious extra
    #: declaration is inert because no objective ever reports a value for it.
    journal_columns: Mapping[str, str] = field(default_factory=dict)
    #: Persisted TRAINING-row columns this property journals as UNPRICED MONITORS, by name alone.
    #:
    #: Here rather than in the descriptor for exactly the reason :attr:`journal_columns` is:
    #: ``ordinary_workflow`` needs the UNION of every property's optional columns in order to parse
    #: a journal whose property it does not yet know, and importing a descriptor to find that out
    #: would pull the pocket geometry and OpenFold into a radius-of-gyration run.
    #:
    #: A name and nothing else, because a monitor has NO GATING WEIGHT -- that is what makes it a
    #: monitor. Whether a given run carries the column is decided by the property's own
    #: :attr:`SteerableProperty.training_row_columns` hook, which has the objective and the config;
    #: this list is only what the persisted schema will ADMIT as surplus. A spurious extra
    #: declaration is inert, because no hook ever returns a value for it.
    monitor_columns: tuple[str, ...] = ()
    #: Runtime (``bioemu_pytorch``) keys this property OWNS: absent-means-off configuration that
    #: only its objective reads. The workflow admits the union as optional keys and REFUSES a key
    #: owned by another property, exactly as it refuses another property's ``target`` arm block --
    #: so a weight cannot be set on a run whose loss constructor would drop it, and a fifth
    #: property's private weight costs a line here rather than an edit to the shared key set.
    optional_runtime_keys: tuple[str, ...] = ()
    #: Every :attr:`ValidationSelectionMetric.name` this property's
    #: :attr:`SteerableProperty.validation_selection_metric` hook could EVER return. Empty for a
    #: property that always selects on the shared default, which is every property on disk.
    #:
    #: The import-light half of that hook, and it exists for the same reader
    #: :attr:`SteerableProperty.validation_row_vocabulary` exists for: a persisted checkpoint
    #: records the name it was selected on, and the completion gate that validates it holds an
    #: ``adapter_config`` and cannot reconstruct the objective that chose it. It admits exactly
    #: this vocabulary beside the two shared spellings, so an arbitrary string is still refused
    #: while a legitimately declared metric is not.
    #:
    #: ``resolve`` refuses a descriptor whose hook could name something absent from here only
    #: where the hook is actually exercised; the static half is checked by
    #: ``tests/test_steering_registry_hooks.py``, which drives the hook over the objectives each
    #: property can build.
    selection_metric_names: tuple[str, ...] = ()


#: Cache keyed by property name. Resolution imports a module and validates it, and the workflow
#: asks for the same descriptor at a dozen call sites in one run.
_RESOLVED: dict[str, SteerableProperty] = {}


def _catalog() -> Mapping[str, PropertyRegistration]:
    from sae_bioemu.steering.property_catalog import STEERABLE_PROPERTY_REGISTRATIONS

    return STEERABLE_PROPERTY_REGISTRATIONS


def registered_property_names() -> tuple[str, ...]:
    """Return every registered property name, in catalog order."""
    return tuple(_catalog())


def arm_config_keys() -> Mapping[str, str]:
    """Return ``{target block key: owning property name}`` for every property with an arm.

    Import-light by construction: it reads the catalog and no descriptor, because the workflow
    needs it to validate the ``target`` key set of a config whose property may be any of them.
    """
    return {
        registration.arm_config_key: name
        for name, registration in _catalog().items()
        if registration.arm_config_key is not None
    }


def journal_column_weights() -> tuple[tuple[str, str], ...]:
    """Return ``((column, gating weight), ...)`` for every registered property, merged.

    **This is the persisted off-path journal schema, and it is now a function of the catalog
    rather than a literal set in the shared workflow.** It used to be
    ``_OFF_PATH_RECORD_KEYS = frozenset({"perp_loss", "rigidity_loss", "dihedral_loss"})`` plus a
    parallel tuple of weights, both in ``steering/ordinary_workflow.py`` -- a member of the shared
    ``training_only`` group. So a project adding one journal column re-keyed every other
    property's checkpoints for a column none of them can write, which is the D-43 defect the
    registry exists to end, surviving in the one table nobody had moved yet.

    The order is catalog order then declaration order, and it is FIXED rather than sorted because
    the workflow's validator reports failures in it and a reordering would change which of two
    simultaneously broken columns is named first.

    Two properties may declare the same column, and the cryptic and kinase families do: both price
    ``lambda_perp`` and both journal ``perp_loss``. Declaring it against two DIFFERENT weights is
    refused -- one column gated by two weights is a column whose presence means two things.
    """
    merged: dict[str, str] = {}
    for name, registration in _catalog().items():
        for column, weight in registration.journal_columns.items():
            if not isinstance(column, str) or not column:
                raise ValueError(f"{name!r} declares a journal column with no name")
            if not isinstance(weight, str) or not weight:
                raise ValueError(f"{name!r} declares journal column {column!r} with no weight")
            previous = merged.get(column)
            if previous is not None and previous != weight:
                raise ValueError(
                    f"the journal column {column!r} is declared against {previous!r} and against "
                    f"{weight!r}; a column gated by two weights is a column whose presence means "
                    "two things, and the reader cannot pair it with the objective that wrote it"
                )
            merged[column] = weight
    return tuple(merged.items())


def journal_column_names() -> frozenset[str]:
    """Return every registered off-path journal column name."""
    return frozenset(column for column, _ in journal_column_weights())


def journal_column_owners() -> Mapping[str, frozenset[str]]:
    """Return ``{off-path journal column: the property names that journal it}``.

    The shared validator holds the UNION of every property's columns, because it parses a journal
    before it knows which property wrote it. Pairing a column's presence with its weight is
    therefore a cross-property statement, and it was only ever CORRECT because no two properties
    named the same weight: ``perp_loss`` is declared by cryptic and kinase against ``lambda_perp``
    and both compute it, so the union happened to coincide with each property's own set.

    ``lambda_m`` broke that coincidence (D-87). The cryptic pocket journals ``metric_loss`` and
    ``metric_deviation_angstrom`` against it, domain motion journals ``direct_metric_angstrom``
    against it, and the three are different measurements over different atom sets -- so a cryptic
    row at ``lambda_m > 0`` was suddenly required to carry a column a cryptic run cannot compute.
    This is what lets the validator ask each property only for the columns that property declares.
    """
    owners: dict[str, set[str]] = {}
    for name, registration in _catalog().items():
        for column in registration.journal_columns:
            owners.setdefault(column, set()).add(name)
    return {column: frozenset(names) for column, names in owners.items()}


def monitor_column_names() -> tuple[str, ...]:
    """Return every registered UNPRICED training-row monitor column, in catalog order.

    The persisted schema the shared journal reader admits as surplus, and -- like
    :func:`journal_column_weights` -- a function of the catalog rather than a literal set in
    ``steering/ordinary_workflow.py``. A property adding a monitor therefore costs a catalog line
    and its own hook, not an edit to a member of the shared ``training_only`` group.

    Two properties may declare the same column name and the merge is a union, because a monitor
    carries no gate that two declarations could disagree about. What a column MEANS is still one
    thing, which is why the name states its quantity and its unit.
    """
    merged: list[str] = []
    for name, registration in _catalog().items():
        for column in registration.monitor_columns:
            if not isinstance(column, str) or not column:
                raise ValueError(f"{name!r} declares a monitor column with no name")
            if column not in merged:
                merged.append(column)
    overlap = sorted(set(merged) & journal_column_names())
    if overlap:
        raise ValueError(
            f"the columns {overlap} are declared BOTH as priced journal columns and as unpriced "
            "monitors; one column name carries one gate and one meaning, and a reader could not "
            "tell which of the two wrote a given row"
        )
    return tuple(merged)


def contested_journal_weights() -> frozenset[str]:
    """Weight names that gate journal columns belonging to DIFFERENT properties.

    The shared journal validator holds the UNION of every property's off-path columns, because it
    parses a row before it knows which property wrote it, and it pairs each column's presence with
    its gating weight. That pairing is a cross-property statement and it was only ever correct
    because no two properties named one weight for different columns:

    * ``lambda_perp`` gates ``perp_loss``, and the cryptic and kinase families BOTH declare that
      column and both compute it. One weight, one column, two owners -- consistent, so the union
      check is exactly right and is left alone.
    * ``lambda_m`` (D-87) gates the cryptic ``metric_loss``/``metric_deviation_angstrom`` AND the
      domain-motion ``direct_metric_angstrom``. Those are different measurements over different
      atom sets, so a cryptic row at ``lambda_m > 0`` would be required to carry a column a cryptic
      run cannot compute.

    This returns the second kind and only the second kind, so the narrowing the validator applies
    is confined to the ambiguity that made it necessary and every other column's check is
    byte-identical to the one it has always had.
    """
    owners = journal_column_owners()
    by_weight: dict[str, set[frozenset[str]]] = {}
    for column, weight in journal_column_weights():
        by_weight.setdefault(weight, set()).add(owners[column])
    return frozenset(weight for weight, sets in by_weight.items() if len(sets) > 1)


def optional_runtime_key_owners() -> Mapping[str, tuple[str, ...]]:
    """Return ``{optional runtime key: the property names that own it}`` across the catalog.

    The runtime twin of :func:`arm_config_keys`, and used the same way: the union widens what a
    config may SPELL, and the per-key owner set is what refuses it on every property outside that
    set. Without the second half, admitting ``lambda_m`` to the shared optional set would let a
    kinase config carry it, pass the schema check, and have it dropped by a loss constructor that
    does not take it -- a weight configured, applied to nothing and absent from the objective
    identity, which is the exact shape that voided six radius-of-gyration arms.

    **A key may have MORE THAN ONE owner, and that is D-87's whole point.** ``lambda_m`` is owned
    by the cryptic pocket and by domain motion, and it means the SAME thing on both: the weight on
    the graded Angstrom metric, multiplying that property's own ``metric_mse``/``huberised_hinge``
    per observation. Refusing a co-owned key would have forced domain motion to keep pricing its
    metric through the shared ``lambda_y`` -- which on cryptic means the ``the progress coordinate`` progress-coordinate
    proxy that arm sets to zero to REMOVE, so one name meant two opposite things depending on which
    config a reader had open.

    Two owners is a claim that the two prices the SAME quantity under the same name. It is not a
    licence to share a key for convenience: a reader of a config now knows what ``lambda_m`` means
    without knowing which property is running, and that is the property the multiple ownership
    asserts.
    """
    owners: dict[str, list[str]] = {}
    for name, registration in _catalog().items():
        for key in registration.optional_runtime_keys:
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name!r} declares an unnamed optional runtime key")
            owners.setdefault(key, []).append(name)
    return {key: tuple(names) for key, names in owners.items()}


def validation_row_vocabulary(name: str) -> frozenset[str]:
    """Return every validation-row column the named property could declare, as a set."""
    return frozenset(resolve(name).validation_row_vocabulary)


def selection_metric_names() -> frozenset[str]:
    """Return every non-default validation selection metric any registered property may declare.

    Import-light: it reads the catalog and no descriptor, because the completion gate that uses it
    holds a persisted ``adapter_config`` and has no objective to ask. Empty today for every
    property but the direct-metric ones, so the admitted set is the two shared spellings alone --
    which is exactly the set every checkpoint on disk records.
    """
    return frozenset(
        name for registration in _catalog().values() for name in registration.selection_metric_names
    )


def resolve_selection_metric(
    name: str, *, objective: Any, arm: Any
) -> ValidationSelectionMetric | None:
    """Return the selection metric one property declares for THIS objective, or ``None``.

    Fails closed on a declaration the catalog does not admit: the two halves of the hook -- the
    behaviour on the descriptor and the vocabulary on the registration -- must agree, because the
    reader that validates a finished checkpoint only ever sees the second one. A metric returned
    here but absent from :attr:`PropertyRegistration.selection_metric_names` would train a run that
    its own completion gate then refuses to attest.
    """
    descriptor = resolve(name)
    metric = descriptor.validation_selection_metric(objective=objective, arm=arm)
    if metric is None:
        return None
    if not isinstance(metric, ValidationSelectionMetric):
        raise TypeError(
            f"{name!r} declared a validation selection metric that is not a "
            "ValidationSelectionMetric"
        )
    admitted = tuple(_catalog()[name].selection_metric_names)
    if metric.name not in admitted:
        raise ValueError(
            f"{name!r} selects on {metric.name!r}, which is not in its registration's "
            f"selection_metric_names {list(admitted)}. The completion gate validates a persisted "
            "checkpoint against that vocabulary and holds no objective, so a run selecting on an "
            "unadmitted name would write a checkpoint it could never attest"
        )
    return metric


def resolve(name: str) -> SteerableProperty:
    """Return the descriptor for one steerable property, importing its package on demand.

    Two gates, and both close the only hole an unfingerprinted catalog could open:

    1. the descriptor must declare the name it was routed for, so a mis-routed entry cannot
       steer one quantity under another's name; and
    2. the descriptor's own source file must be a member of one of this property's own TRAINING
       groups, so the file that decides the dose set, the target origin and the objective class
       is itself hashed into the checkpoint that decision produced. Without this a catalog edit
       could point a name at different behaviour at an unchanged digest.
    """
    cached = _RESOLVED.get(name)
    if cached is not None:
        return cached
    registration = _catalog().get(name)
    if registration is None:
        raise ValueError(
            f"{name!r} is not a registered steerable property; register it in "
            "sae_bioemu.steering.property_catalog and give it a descriptor in its own package"
        )
    module = importlib.import_module(registration.descriptor_module)
    descriptor = getattr(module, registration.descriptor_attribute, None)
    if not isinstance(descriptor, SteerableProperty):
        raise ValueError(
            f"{registration.descriptor_module}.{registration.descriptor_attribute} is not a "
            "SteerableProperty"
        )
    if descriptor.name != name:
        raise ValueError(
            f"the catalog routes {name!r} to a descriptor that declares "
            f"{descriptor.name!r}; a descriptor answers for its own property or for none"
        )
    _require_self_hashing(name, registration, module)
    _RESOLVED[name] = descriptor
    return descriptor


def _require_self_hashing(name: str, registration: PropertyRegistration, module: Any) -> None:
    """Fail closed unless the descriptor module is hashed by its own property's fingerprint."""
    source = Path(inspect.getfile(module)).resolve()
    training_groups = tuple(registration.code_group_composition.get("training", ()))
    members = {
        path.resolve()
        for group in training_groups
        for path in registration.code_group_members.get(group, {}).values()
    }
    if source not in members:
        raise ValueError(
            f"the descriptor for {name!r} lives in {source} and is not a member of any of its "
            f"own training code groups {list(training_groups)}. It decides the dose set, the "
            "target origin, the callback versions and the objective class, so a checkpoint "
            "that is not bound to it could be produced by one descriptor and evaluated under "
            "another at an unchanged digest"
        )


def shared_code_paths() -> dict[str, dict[str, Path]]:
    """Return the three groups EVERY steered property executes (D14, D-43).

    ``training_only`` -- code that shaped the trained adapter and nothing else: the training
    loop and its checkpoint/journal schema, the BioEmu configuration entry point, the batch
    planner, and this registry. ``batching.py`` is here because it decides which scheduled
    units share an optimizer step (D15): the resolved sizes are already in ``adapter_config``,
    so a change that alters sizes is caught, but a change that reshuffles batch *membership* at
    identical sizes is not, and that is a different optimization. It is training-only rather
    than generation-shared because the held-out plan it also builds is a pure compute grouping.
    ``property_registry`` is here because it is the dispatcher every steered run executes: it is
    shared, so its blast radius is shared, which is precisely why no property's data is in it.

    ``generation_shared`` -- code BOTH paths execute. Evaluation regenerates structures with the
    same sampler, hook, property surrogate, adapter architecture and target formula that
    training used, so a change here changes what a structure *is* in both.
    ``steering_artifacts`` is here because both paths sign their artifacts with it, and
    ``pyproject``/``uv.lock`` because they pin the numerics the generator runs on.

    ``property_reference`` JOINED this group with the registry work, and it closes a
    fail-OPEN: ``load_property_reference_table`` parses, validates and serves ``sigma_seq``,
    which is the DENOMINATOR of the target term on every relative-level run and the band the
    eligibility gate is applied in. The table's own digest was in the identity; the code that
    decides what the table means was in no group at all, so a change to the parser or to the
    eligibility rule moved no digest while changing every target.

    ``evaluation_only`` -- code that only reads a frozen adapter and reports numbers. Changing
    it must not invalidate a training checkpoint.
    """
    steering_package = Path(__file__).resolve().parent
    bioemu_package = steering_package.parent / "bioemu"
    repository = steering_package.parents[2]
    return {
        "training_only": {
            "batching": steering_package / "batching.py",
            "bioemu_steering_workflow": bioemu_package / "steering_workflow.py",
            "ordinary_workflow": steering_package / "ordinary_workflow.py",
            "property_registry": Path(__file__).resolve(),
            "training": steering_package / "training.py",
        },
        "generation_shared": {
            "adapter": steering_package / "adapter.py",
            "bioemu_extractor": bioemu_package / "extractor.py",
            "bioemu_steering": bioemu_package / "steering.py",
            "bioemu_steering_artifacts": bioemu_package / "steering_artifacts.py",
            "bioemu_steering_properties": bioemu_package / "steering_properties.py",
            # Generation-shared without qualification: it decides what arithmetic the
            # denoiser runs in, and both paths generate through it.
            "precision": bioemu_package / "precision.py",
            "property_reference": steering_package / "property_reference.py",
            "pyproject": repository / "pyproject.toml",
            "schedule": steering_package / "schedule.py",
            "surrogate": steering_package / "surrogate.py",
            "uv_lock": repository / "uv.lock",
        },
        "evaluation_only": {
            "bioemu_steering_evaluation": bioemu_package / "steering_evaluation.py",
            "evaluation": steering_package / "evaluation.py",
            "evaluation_structures": steering_package / "evaluation_structures.py",
        },
    }


def scientific_code_paths() -> dict[str, dict[str, Path]]:
    """Return every fingerprinted source, grouped by what each group determines.

    The three shared groups come from :func:`shared_code_paths`; every other group is declared
    by the property that executes it. A property's groups are merged, not overwritten, and two
    properties may not claim the same group name with different members -- the cryptic and
    kinase families share ``cryptic_generation_shared`` deliberately (a kinase run executes the
    cryptic pair geometry and the D-31 curvature bound), and the merge below is what lets both
    name it without either owning it.

    No source may appear in two groups. A file counted twice would be gated by the more
    permissive of them and the segmentation would be a fiction.
    """
    groups: dict[str, dict[str, Path]] = {
        name: dict(members) for name, members in shared_code_paths().items()
    }
    for registration in _catalog().values():
        for group, members in registration.code_group_members.items():
            existing = groups.setdefault(group, {})
            for member, path in members.items():
                previous = existing.get(member)
                if previous is not None and previous.resolve() != path.resolve():
                    raise ValueError(
                        f"two registrations declare {group}.{member} as different files: "
                        f"{previous} and {path}"
                    )
                existing[member] = path
    return groups


def property_code_groups() -> dict[str, dict[str, tuple[str, ...]]]:
    """Return ``{property: {"training": (...), "evaluation": (...)}}`` from the catalog.

    Which groups compose a run's fingerprints is the PROPERTY's question (D-43): a
    radius-of-gyration run executes no cryptic source, so hashing one into its checkpoint would
    retire that checkpoint for a change it could never have seen. The composition narrows blast
    radius and never narrows coverage -- every group is still hashed file by file and still
    recorded in ``code_group_fingerprints``.
    """
    declared = scientific_code_paths()
    composition: dict[str, dict[str, tuple[str, ...]]] = {}
    for name, registration in _catalog().items():
        table = {
            role: tuple(entries) for role, entries in registration.code_group_composition.items()
        }
        if set(table) != {"training", "evaluation"}:
            raise ValueError(
                f"{name!r} must compose exactly a training and an evaluation fingerprint; "
                f"found {sorted(table)}"
            )
        unknown = sorted({group for entries in table.values() for group in entries} - set(declared))
        if unknown:
            raise ValueError(
                f"{name!r} composes its fingerprints from undeclared code groups {unknown}"
            )
        composition[name] = table
    return composition
