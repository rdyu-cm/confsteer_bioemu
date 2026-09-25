"""Resumable, fail-closed training for one ordinary steering property."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from sae_bioemu.bioemu.precision import (
    DEFAULT_GENERATION_PRECISION,
    resolve_generation_precision,
)
from sae_bioemu.core.config import seed_everything

from .adapter_config import DEFAULT_RESIDUAL_CAP, AdapterArchitecture
from .batching import (
    ACCUMULATION_WEIGHTING,
    LENGTH_BUCKETED_RULE,
    UNBATCHED_RULE,
    UNIFORM_OPTIMIZER_STEP_UNITS,
    UNIFORM_STEP_RULE,
    GradientAccumulationPlan,
    SteeringBatchPlan,
    batch_offsets,
    cycle_segments,
    plan_optimizer_step_spans,
    plan_uniform_optimizer_steps,
)
from .config import ModelFamily
from .distributed import (
    DEFAULT_WORLD_SIZE,
    DistributedContext,
    agreed_first_failing_rank,
    all_reduce_adapter_gradients,
    barrier,
    broadcast_from,
    concatenated_records,
    gather_indexed,
    rank_micro_batch_indices,
    single_process_context,
)
from .evaluation import SteeringFailure
from .gradient_diagnostics import GradientProbe
from .progress import (
    ProgressReporter,
    TimingSidecar,
    load_checkpoint_timings,
    timing_summary,
)
from .property_reference import (
    DEFAULT_SIGMA_ELIGIBILITY,
    PropertyReferenceTable,
    SigmaEligibility,
)
from .property_registry import (
    JournalledSampleColumn,
    ValidationSelectionMetric,
    contested_journal_weights,
    journal_column_owners,
    journal_column_weights,
    monitor_column_names,
    registered_property_names,
    selection_metric_names,
)
from .property_registry import (
    resolve as _resolve_property,
)
from .schedule import (
    CONTINUOUS_DOSE_LEVELS,
    PROPERTY_ENSEMBLE_SEED_RULE,
    RELATIVE_TARGET_ORIGINS,
    SIGMA_DOSE_TARGET_ORIGINS,
    TARGET_ORIGIN_MATCHED_CONTROL_STATE,
    TARGET_ORIGIN_MATCHED_DRAW_ZERO,
    TARGET_ORIGIN_SEQUENCE_MEAN,
    TARGET_ORIGIN_SYSTEM_CONSTANT,
    DoseLevels,
    SteeringSchedule,
    SteeringUnit,
    SystemTargetConstants,
    build_property_ensemble_seeds,
    continuous_target,
    relative_displacement_target,
    relative_sequence_target,
    system_constant_target,
    two_state_reporting_target,
)
from .schedule_extension import (
    SCHEDULE_EXTENSION_RECORD_KEYS,
    SCHEDULE_EXTENSION_SCHEMA,
    ScheduleExtension,
    ScheduleExtensionDeclaration,
    ScheduleExtensionError,
    inventory_epochs,
    reconcile_extended_adapter_config,
)
from .training import (
    ABSOLUTE_LOSS_NORMALIZATION,
    PER_CHANNEL_LOSS_NORMALIZATION,
    SIGMA_LOSS_NORMALIZATION,
    TRANSITION_LOSS_NORMALIZATION,
    AdapterCheckpoint,
    ResidualPenaltyKind,
    SteeringLoss,
    TargetRange,
    assert_adapter_only_gradients,
    freeze_for_steering,
    load_adapter_checkpoint,
    read_adapter_checkpoint,
    save_adapter_checkpoint,
    validate_adapter_checkpoint,
)

#: Target constructions supported by this workflow.
ABSOLUTE_SEQUENCE_TARGET = "absolute_sequence"
RELATIVE_SEQUENCE_TARGET = "relative_sequence"
_TARGET_KINDS = frozenset({ABSOLUTE_SEQUENCE_TARGET, RELATIVE_SEQUENCE_TARGET})

#: Configuration fields that only mean something for an absolute-Angstrom target.
_ABSOLUTE_ONLY_FIELDS = ("target_range", "maximum_change", "target_mean", "target_scale")

#: Best-checkpoint selection metric per target kind. The absolute path keeps the raw
#: property-unit MAE; the relative path selects on the sigma-normalized MAE so that
#: selection and the sigma-normalized objective (D4b) cannot disagree.
ABSOLUTE_SELECTION_METRIC = "target_mae"
RELATIVE_SELECTION_METRIC = "target_mae_sigma"

#: How a run picks the adapter it ships. **DECLARED, never inferred.**
#:
#: ``validation_split`` is the historical regime and the default: a validation role is walked at
#: every epoch boundary, the round with the best selection metric is remembered, early stopping
#: counts rounds that did not improve, and the checkpoint restored at the end is that best round's.
#:
#: ``none`` is a run with no validation role at all. Nothing is selected, nothing is early-stopped,
#: and the adapter that ships is the one the LAST optimizer step produced. It exists because a
#: validation set can be too small to resolve the effect it is being asked about: the kinase pool
#: carves 6 of 34 training genes out for validation, which at the gene level with ties dropped
#: gives n = 4 (p = 0.125), so only a unanimous 6/6 could ever clear p < 0.05 -- and it nonetheless
#: selected round 0 on two arms and ended a 2240-unit run at 7.5% on ``target_mae`` moving 0.60384
#: to 0.60416. Removing the role returns those 6 genes to training.
#:
#: It is a DECLARATION and not an empty list, in both directions: a config that says ``none`` must
#: carry an empty validation role and a config that says nothing must carry a populated one. An
#: empty list alone could mean either "this arm deliberately selects on nothing" or "the validation
#: role was dropped by accident", and the difference has to survive into the run report and the
#: checkpoint identity for as long as the checkpoint does.
MODEL_SELECTION_VALIDATION_SPLIT = "validation_split"
MODEL_SELECTION_NONE = "none"
MODEL_SELECTION_MODES = frozenset({MODEL_SELECTION_VALIDATION_SPLIT, MODEL_SELECTION_NONE})

# The four normalisation strings are imported from ``training`` above and re-exported here, which
# is where the config, the checkpoint identity and every caller read them from. They live beside
# the objective that ENFORCES them -- ``SteeringLoss`` requires the supplied ``property_scale`` to
# match the one it was told to use -- so the string that selects the objective and the string that
# records it cannot be two strings.
#
# ``sequence_sigma`` is D4b: the target is a multiple of the sequence's own conformational spread,
# so the error is measured in the same units, on both channels. Radius of gyration uses it and its
# numbers must not move. ``per_channel_sigma_transition`` is the two-state form (D-65): the
# ABSOLUTE channel divides by sigma_the progress coordinate, because under the state target the matched control cancels
# and the residual carries the whole draw baseline, and the PAIRED channel divides by nothing,
# because the shared draw cancels inside the gap. ``apo_holo_transition`` is D-64's two-state form,
# with neither channel divided; it remains constructible and nothing resolves to it any more.
# ``property_units`` is the absolute path's raw property units.

#: The normalisations that describe a TWO-STATE target construction, i.e. the
#: ``matched_control_state`` origin. Membership rather than equality at every checkpoint-reading
#: site, because those sites ask "was this a two-state run" -- which governs whether a
#: ``dose_multiplier`` exists and which journal schema was written -- and that answer is the same
#: under D-64's normalisation and under D-65's. A checkpoint written under either must still be
#: readable as what it is.
_TWO_STATE_LOSS_NORMALIZATIONS = frozenset(
    {TRANSITION_LOSS_NORMALIZATION, PER_CHANNEL_LOSS_NORMALIZATION}
)


def resolve_property_loss_normalization(target_kind: str, relative_target_origin: str) -> str:
    """Return the loss normalisation a target construction requires.

    The single mapping from "how the target is built" to "what the error is divided by". A
    caller that constructs the objective before it constructs the config -- the BioEmu
    workflow does -- resolves the declaration here rather than restating the rule, so the
    declaration and :attr:`OrdinarySteeringConfig.property_loss_normalization` cannot drift.

    The two-state origin resolves to :data:`PER_CHANNEL_LOSS_NORMALIZATION` since D-65.
    :data:`TRANSITION_LOSS_NORMALIZATION` is deliberately still a legal declaration on
    ``SteeringLoss`` and is deliberately unreachable from here: keeping it constructible is what
    makes per-channel a genuine fourth objective rather than a rename, and keeping it out of this
    mapping is what stops a config from selecting the retired one by accident.
    """
    if target_kind != RELATIVE_SEQUENCE_TARGET:
        return ABSOLUTE_LOSS_NORMALIZATION
    if relative_target_origin == TARGET_ORIGIN_MATCHED_CONTROL_STATE:
        return PER_CHANNEL_LOSS_NORMALIZATION
    return SIGMA_LOSS_NORMALIZATION


#: What the fixed-seed diagnostic pins, and how. Recorded in the checkpoint identity so a
#: diagnostic checkpoint can never be mistaken for a run with matched per-unit draws.
FIXED_GENERATION_SEED_SCOPE = "train"
FIXED_GENERATION_SEED_RULE = "first_scheduled_train_seed_per_sequence"

#: D16: which estimator the property term uses when a unit carries more than one structure.
#: ``mean_m(Rg_m)`` enters the loss once, so the residual multiplying the gradient is the
#: ensemble's, not each draw's. This is BioEmu's own property-prediction fine-tuning
#: estimator; the alternative -- one loss per structure, averaged -- is what
#: ``gradient_accumulation`` already does and is what this replaces.
ENSEMBLE_MEAN_PROPERTY_ESTIMATOR = "ensemble_mean_property"

#: D16: and how the residual penalty reduces the same ``M`` structures, deliberately the
#: other way. Every structure received its own intervention, so every structure is an
#: observation in ``mean(F^2)``; see :func:`_ensemble_residual`.
ENSEMBLE_RESIDUAL_REDUCTION = "square_then_average_over_members"

_BASELINE_SCHEMA_VERSION = 2
#: Bumped from 4 to 5 when the relative-path journal rows gained the achieved property
#: value, the signed error, and the per-sequence sigma, and renamed the sigma-normalized
#: property loss so its unit travels with the value. Absolute-path rows are byte-identical
#: to schema 4; the version moves because one journal cannot carry two meanings.
#:
#: **It did NOT move to 6 for the two-state row schema, and that is a decision.** Those rows are
#: a third key set (``_TWO_STATE_RECORD_KEYS``): ``target_state`` and
#: ``control_reporting_value`` join the relative row, whose other columns they keep.
#: A version bump would buy nothing, because ``_records`` compares
#: a persisted row's key set with EQUALITY against the allowed sets, so a schema-5 reader already
#: refuses a two-state row by name rather than misreading it. It would cost the resume of every
#: radius-of-gyration checkpoint on disk, including the arms currently running, whose rows this
#: change leaves byte-identical. That is the D-45 distinction: this version is a check that
#: catches, not a key that binds, and there is nothing here for it to catch.
#:
#: **It did NOT move to 6 for the gated loss-term columns either**, and the same argument was
#: re-checked against the code rather than inherited. ``_journal`` rejects on exact equality with
#: this integer BEFORE any row is parsed, so a bump is strictly more destructive than the key-set
#: check it would duplicate. In the old-reader/new-row direction the key-set check already
#: refuses: a pre-change ``_records`` sees ``contrastive_loss`` as a key belonging to no allowed
#: set and raises. In the new-reader/old-row direction the row PARSES, which is the point -- a
#: journal written with all three weights zero is byte-identical and must keep resuming -- and a
#: row written with a nonzero weight and no column is refused by name, with the weight named, by
#: ``_validate_training_record_fields``. A version mismatch would say strictly less.
_LOGGER = logging.getLogger(__name__)

_JOURNAL_SCHEMA_VERSION = 5
_SOURCE_CHECKPOINT_ROLES = frozenset({"generator", "sae"})
_BASELINE_KEYS = frozenset(
    {
        "schema_version",
        "complete",
        "identity",
        "records",
        "failures",
        "attempts",
        "in_flight",
        "content_fingerprint",
    }
)


@dataclass(frozen=True)
class GeneratedTrainingSample:
    """Differentiable output for one scheduled steering visit."""

    surrogate_prediction: Tensor
    residual_rms: Tensor
    residual_fraction: Tensor
    geometry_loss: Tensor
    #: Pre-cap request, reduced over M exactly like the realised fraction. ``None`` on paths
    #: that do not supply it, which the raw penalty rejects rather than working around.
    raw_residual_fraction: Tensor | None = None
    #: Smooth LDDT against the unsteered structure from the same draw. ``None`` when no
    #: reference was available, which is every run before the term existed.
    local_structure: Tensor | None = None
    #: ``zeta_pocket^2`` per batch member, dimensionless (cryptic-steering D-28 term 1).
    #: ``None`` on every path that is not steering a pocket progress coordinate, which is
    #: every run of every other property.
    pocket_off_path_squared: Tensor | None = None
    #: ``rms_C(u)`` per batch member, in Angstrom (D-28 term 2). The RMS itself, not the
    #: hinge: the setpoint is part of the objective and is applied by ``SteeringLoss``.
    context_rms_angstrom: Tensor | None = None
    #: ``c0`` per batch member, in Angstrom (D-30). Per member because the setpoint is per
    #: protein -- that protein's own ambient ``rms_C`` percentile -- and a batch mixes
    #: proteins. Resolved from ``ContextSetpointTable`` by whoever built the batch; there is
    #: no pooled fallback, and a protein stage 1 has not measured cannot be in the batch.
    context_setpoint_angstrom: Tensor | None = None
    #: The four D-35b pocket-local geometry MONITOR counts, one value per batch member, on the
    #: steered draw and on its matched unsteered control. ``None`` on every path that is not
    #: steering a cryptic pocket, which is every run of every other property.
    #:
    #: A MONITOR, not a gate. Nothing here enters a loss, a gradient, a selection metric, a
    #: checkpoint identity or a verdict; the columns exist so that a fold coming apart under the
    #: adapter is visible in the first epoch's journal instead of in the final report, roughly
    #: 12-15 GPU-h later. Both members of each pair are carried rather than their difference:
    #: the D-35b excess is a difference of two integers and stays derivable, while a lone excess
    #: column would hide the control's own base rate and read as the gate verdict it is not.
    gate_scored_out_of_window_bonds: Tensor | None = None
    gate_control_out_of_window_bonds: Tensor | None = None
    gate_scored_clashing_pairs: Tensor | None = None
    gate_control_clashing_pairs: Tensor | None = None
    #: The matched unsteered control's own position along the transition, in the REPORTING frame,
    #: one value per batch member. ``None`` on every path but a two-state one.
    #:
    #: **Not a loss input and not a monitor: it is half of the TARGET.** The two-state ask is an
    #: absolute reporting-frame state and the prediction is a training-frame displacement, so
    #: ``t = the progress coordinate_state(o) - the progress coordinate_reporting(control)`` and the origin varies PER DRAW. That is why it
    #: travels on the generated sample rather than being derived by ``_unit_target`` from a
    #: scheduled unit: no per-unit function and no per-sequence table can produce it, and the
    #: per-protein constant that could (the reference table's ``mean_the progress coordinate``) is exactly the D-46
    #: defect this origin replaces.
    #:
    #: Gradient-free by construction. The producer detaches it, because a target the adapter
    #: could move is a loss reducible by moving the goalposts.
    control_reporting_value: Tensor | None = None
    #: ``d_K``, the Kincore cosine dihedral distance from a draw's six X-D-F-G backbone dihedrals
    #: to the centroid ITS OWN level names, one value per batch member (kinase K-15). ``None`` on
    #: every path that is not steering the DFG progress coordinate, and ``None`` on a kinase arm
    #: that prices ``lambda_dihedral`` at zero -- the same both-directions gating every other
    #: optional input here carries, because a term computed at a zero weight reaches neither the
    #: total nor its journal column.
    #:
    #: It is a loss INPUT and not a monitor: it is bounded in ``[0, 4]`` by construction and
    #: carries the gradient back to the generated N/CA/C, which is what lets an arm descend the
    #: motif rather than only the long-range coordinate.
    dihedral_distance: Tensor | None = None
    #: ``d_K`` of the MATCHED UNSTEERED CONTROL -- the same draw seed with the adapter off --
    #: measured to the same centroid its level names, one value per batch member. ``None`` on
    #: every path but a kinase arm that asked for the control's frames.
    #:
    #: **A MONITOR, and a separate field from** :attr:`dihedral_distance` **rather than a third
    #: column on it.** It enters no term: what it buys is that a held-out ``d_K`` can be read
    #: against its own control's, which is the reading K-22 could not make. ``d_K`` is AFFINE in
    #: the six-dihedral torus embedding -- ``||U||^2 = 6`` identically, so ``d_K = 2 - <U,C>/3``
    #: and ``d_K(steered) - d_K(control) = -<U_steered - U_control, C>/3`` exactly -- so the paired
    #: difference cancels the draw baseline the way ``the progress coordinate``'s does by construction, and arm 8's
    #: "objective fell while |signed_error| rose" becomes a question the journal can answer.
    #:
    #: It does NOT ride the widened ``dihedral_distance`` channel. That channel's splitter in
    #: ``kinase_dfg/loss.py`` carries one or two columns and refuses any other width by name,
    #: because both of its readings are PRICED and a column computed at a zero weight is the
    #: ``lambda_l`` shape. This reading is priced by nothing, so putting it there would mean
    #: relaxing a refusal that exists to catch exactly the thing this is not.
    control_dihedral_distance: Tensor | None = None
    #: ``m``, the superposed BACKBONE deviation of a draw from the deposited conformer ITS OWN
    #: LEVEL names, in ANGSTROM, one value per batch member. ``None`` on every path that does not
    #: measure a direct metric.
    #:
    #: **The WIRE, shared by two properties, and the wire is not the column.** The cryptic pocket
    #: sends the RMS deviation over the benchmark's ``metric_resid_ranges`` and journals it as
    #: ``metric_deviation_angstrom``; domain motion sends the N/CA/C/O RMSD over its own metric
    #: set and journals it as ``direct_metric_angstrom``. The two are different measurements over
    #: different atom sets against different references and each carries its own column name and
    #: its own gate, but they are the same SHAPE of per-observation input and a second field would
    #: be a second wire of identical shape for one call site to fill wrongly. That is the
    #: ``pocket_off_path_squared`` precedent, which the kinase programme reuses for exactly this
    #: reason, and both ends document which quantity travels here.
    metric_deviation_angstrom: Tensor | None = None
    #: ``s_p``, that observation's own protein's reference separation between the two deposited
    #: conformers, in ANGSTROM and strictly positive, one value per batch member. ``None``
    #: everywhere the direct-metric hinge is not priced.
    #:
    #: A tensor and not a scalar for the reason ``context_setpoint_angstrom`` is: a batch mixes
    #: proteins, and one number here would apply one protein's transition scale to another's.
    #: Domain motion does NOT use this field -- it passes ``s_p`` through the shared
    #: ``property_scale``, because there the metric IS the shared target term -- so this carries
    #: the cryptic hinge's divisor alone.
    metric_separation_angstrom: Tensor | None = None
    #: ``zeta_pocket^2`` per batch member, dimensionless, carried as an UNPRICED MONITOR: the
    #: same quantity ``pocket_off_path_squared`` above transports, on a separate wire and with
    #: no weight anywhere near it.
    #:
    #: **Two wires for one number, and the duplication is the content of D-86.** The priced wire
    #: is refused at a zero weight by name (``_off_path_input``), because an input that reaches
    #: neither ``total`` nor a weight-gated journal column is the ``lambda_l`` shape that voided
    #: six arms. That refusal is correct and stays. Its consequence was that the cryptic
    #: DIRECT-METRIC arm -- which runs at ``lambda_perp = 0`` -- computed ``zeta_pocket^2`` on
    #: every batch and journalled it nowhere, so the one arm D-80 section 4 identifies as able to
    #: move the realised alignment off D-76's ``|cos| = 0.165`` recorded nothing that could say
    #: whether it had.
    #:
    #: This wire is read by NO loss. It reaches the journal through the property's
    #: ``training_row_columns``/``validation_row_columns`` hooks, so it cannot enter ``total``,
    #: the objective identity, a selection metric or a gate.
    #:
    #: It is ``zeta_pocket^2 = ||u_T||_w^2/||v||_w^2 - the progress coordinate^2``, the DIMENSIONLESS ratio, which is
    #: what makes ``|cos| = |achieved| / sqrt(achieved^2 + zeta_pocket^2)`` the same number D-76
    #: section 3 computed from ``perp_loss``. Filling it from ``OffPathTerms.
    #: pocket_off_path_squared`` -- the undivided Angstrom^2 residual -- would put a per-protein
    #: scale into a quantity whose whole purpose is cross-protein comparability.
    pocket_off_axis_ratio_squared: Tensor | None = None


@dataclass(frozen=True)
class StandardizedAbsoluteTarget:
    """A callback input whose only scalar is the normalized absolute property target."""

    value: float

    def __post_init__(self) -> None:
        _finite(self.value, "standardized_absolute_target")


@dataclass(frozen=True)
class RequestedLevel:
    """A callback input carrying the bounded dimensionless dose request ``o``.

    On the relative-sequence path the adapter receives the requested level directly
    (D1); no absolute property value and no normalization constants are involved.
    """

    value: float

    def __post_init__(self) -> None:
        if not -1.0 <= _finite(self.value, "requested_level") <= 1.0:
            raise ValueError("requested_level must lie in [-1, 1]")


#: The exact top-level key set a training-subset exclusion artifact carries.
_TRAINING_SUBSET_KEYS = frozenset(
    {
        "schema_version",
        "parent_split_fingerprint",
        "criterion",
        "excluded",
        "excluded_count",
    }
)
_TRAINING_SUBSET_SCHEMA_VERSION = 1
#: Domain-separates the content fingerprint from every other canonical digest in this project.
_TRAINING_SUBSET_CONTENT_SCHEMA = "steering-training-subset-exclusions:v1"


@dataclass(frozen=True)
class TrainingSubset:
    """Which of the split's TRAIN proteins an arm actually trains on, as an EXCLUSION list.

    **This is D-23's deliberate, documented exception and it exists for one reason: two arms
    that differ only in their training population must share their noise draws.**
    ``schedule_seed_fingerprint`` covers the whole split dict, so giving one arm a *derived
    split* with a filtered train role re-randomises its VALIDATION and TEST draws as well --
    roles that are bit-identical between the arms by construction. That re-randomisation has a
    measured noise floor on this project's own headline quantity (sd 0.0057 on slope) which
    ``bioemu/steering_workflow.py`` records as exceeding several differences worth claiming. So
    both arms name the SAME split entry, their seed fingerprints are identical, every draw is
    shared, and the distinguishing fact lives here instead.

    **Exclusions rather than a subset, and the choice is load-bearing three times over.**

    * ABSENCE is the unfiltered arm. A config with no ``training_subset`` block trains on the
      whole train role and writes an ``adapter_config`` byte-identical to every checkpoint made
      before this field existed -- the same conditional-key discipline ``cryptic_arm``,
      ``dose_levels`` and ``gradient_clip_norm`` follow. Spelled as a SUBSET, the unfiltered arm
      would have to either enumerate all of the training proteins or also use absence, and one
      experiment with two spellings writes two identities.
    * It states the ABLATION rather than the population. D-66 requires the unfiltered arm's
      training set to be a strict SUPERSET of the filtered one; an exclusion list makes that
      true by construction instead of by comparing two enumerations.
    * A stale list fails LOUD. Every excluded id must be in the schedule's train role
      (:func:`apply_training_subset`), so a list carried across a split rebuild raises by name
      rather than quietly excluding nothing.

    **It must never reach** ``schedule_seed_fingerprint``. That is the whole point, and
    ``tests/test_bioemu_steering_schedule_seed.py`` asserts the seed block's literal key set.
    It IS bound into ``adapter_config`` by BOTH digests D-23 names, because they answer
    different questions: :attr:`sha256` is over the file's bytes, so any edit at all moves it,
    and :attr:`content_fingerprint` is over the parsed population, so it says whether the ARM
    changed as opposed to the file's formatting. Without them the field's contents would be
    unbound and editable in place under a finished checkpoint.
    """

    #: Absolute path to the exclusion artifact, RECORDED. What binds is the pair of digests
    #: below; the path is provenance, exactly as ``baseline_path`` is.
    path: Path
    #: SHA-256 of the file's bytes.
    sha256: str
    #: Canonical digest of the PARSED population, so reformatting the file is distinguishable
    #: from changing which proteins the arm trains on.
    content_fingerprint: str
    #: The excluded sequence ids, in ascending order.
    excluded: tuple[str, ...]
    #: The split record the exclusions were derived against, ``None`` when the artifact does not
    #: name one. Recorded rather than enforced here: this module has no split in hand, and
    #: :func:`apply_training_subset` already refuses an id the schedule does not carry.
    parent_split_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("training subset path must be an absolute Path")
        _fingerprint(self.sha256, "training subset sha256")
        _fingerprint(self.content_fingerprint, "training subset content_fingerprint")
        if not isinstance(self.excluded, tuple) or not self.excluded:
            raise ValueError(
                "a training subset excludes at least one training sequence; an EMPTY exclusion "
                "list is the unfiltered arm, which is spelled by omitting the field entirely so "
                "that one experiment cannot write two identities"
            )
        if any(not isinstance(name, str) or not name for name in self.excluded):
            raise ValueError("training subset excluded ids must be non-empty strings")
        if len(set(self.excluded)) != len(self.excluded):
            raise ValueError("training subset excluded ids must be unique")
        if tuple(sorted(self.excluded)) != self.excluded:
            raise ValueError(
                "training subset excluded ids must be sorted; the order is canonical so that "
                "two artifacts naming one population cannot carry two content fingerprints"
            )
        if self.parent_split_fingerprint is not None:
            _fingerprint(self.parent_split_fingerprint, "training subset parent_split_fingerprint")
        if self.content_fingerprint != training_subset_content_fingerprint(self.excluded):
            raise ValueError(
                "training subset content_fingerprint does not reproduce from its own excluded "
                "ids; a digest that does not re-derive binds nothing"
            )

    def identity(self) -> dict[str, Any]:
        """Return the block ``adapter_config`` records, which is what binds this field."""
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "content_fingerprint": self.content_fingerprint,
            "excluded_count": len(self.excluded),
            "parent_split_fingerprint": self.parent_split_fingerprint,
        }


def training_subset_content_fingerprint(excluded: Sequence[str]) -> str:
    """Return the canonical digest of one exclusion population.

    Domain-separated and schema-tagged, so this digest cannot collide with any other canonical
    fingerprint this project writes, and computed from the SORTED ids so that a reordered file
    is the same population.
    """
    payload = {
        "schema": _TRAINING_SUBSET_CONTENT_SCHEMA,
        "excluded": sorted(str(name) for name in excluded),
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def load_training_subset(path: str | Path) -> TrainingSubset:
    """Parse an exclusion artifact and bind it by both digests.

    The artifact is produced by ``scripts/cryptic_steering/derive_filtered_split.py
    --write-exclusion-list``, which is also what computes the exclusions and proves the derived
    split's validation and test roles are bit-identical to the parent's. Nothing here trusts
    that: the key set, the schema version, the id shape, the ordering and the self-consistency
    of ``excluded_count`` are all re-checked, because this file is the only thing standing
    between an edited list and a checkpoint that claims to have trained on it.
    """
    resolved = Path(path).resolve()
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise ValueError(f"training subset artifact {resolved} is unreadable: {error}") from error
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"training subset artifact {resolved} is not valid JSON") from error
    if not isinstance(payload, Mapping) or frozenset(payload) != _TRAINING_SUBSET_KEYS:
        raise ValueError(
            "training subset artifact keys must be exactly "
            f"{sorted(_TRAINING_SUBSET_KEYS)}; found "
            f"{sorted(payload) if isinstance(payload, Mapping) else type(payload).__name__}"
        )
    if payload["schema_version"] != _TRAINING_SUBSET_SCHEMA_VERSION:
        raise ValueError(
            "training subset artifact schema_version must be "
            f"{_TRAINING_SUBSET_SCHEMA_VERSION}, found {payload['schema_version']!r}"
        )
    excluded = payload["excluded"]
    if not isinstance(excluded, list) or any(
        not isinstance(name, str) or not name for name in excluded
    ):
        raise ValueError("training subset artifact excluded must be a list of sequence ids")
    if payload["excluded_count"] != len(excluded):
        raise ValueError(
            "training subset artifact excluded_count disagrees with its own list, so the file "
            "has been edited without its own summary being rewritten"
        )
    parent = payload["parent_split_fingerprint"]
    if parent is not None and not isinstance(parent, str):
        raise ValueError("training subset artifact parent_split_fingerprint must be a digest")
    return TrainingSubset(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        content_fingerprint=training_subset_content_fingerprint(excluded),
        excluded=tuple(excluded),
        parent_split_fingerprint=parent,
    )


def apply_training_subset(
    schedule: SteeringSchedule, subset: TrainingSubset | None
) -> SteeringSchedule:
    """Drop the excluded sequences' TRAIN units, AFTER the schedule has been built.

    **After, never by shrinking the split, and the reason is arithmetic rather than tidiness.**
    ``build_steering_schedule`` derives a sequence's ``rank`` from its position in the SORTED
    training set and sets ``level = levels[(rank + cycle + visit) % len(levels)]``, and
    ``_allocate_unique_seeds`` resolves seed collisions over one global namespace spanning
    train, validation and test. Removing a protein from ``split.train`` would therefore
    reassign the levels of every protein after it AND could move a validation or test unit's
    seed -- so a "subset" arm would differ from its sibling in the doses it saw and the noise it
    saw them at, not in its population. Filtering here changes neither: ``draw_seed`` is a pure
    function of ``(fingerprint, split, sequence_id, cycle)``, ``level`` was already fixed by the
    full-set rank, and the collision resolution has already run over the full set.

    **The excluded units do not survive as skipped work.** They are removed outright, so the
    cursor, the journal, the attempt inventory, the batch plan and the resume contract all see a
    schedule that contains exactly the units the arm will walk. Keeping them and skipping them
    would need a second notion of "done" that resume, early stopping and every completion check
    would have to agree on, and a run that crashed mid-epoch would be indistinguishable from one
    that had skipped the same units.

    An excluded id absent from the train role RAISES: that is a list carried across a split
    rebuild, and excluding nothing while claiming to have filtered is the failure this exists to
    prevent.
    """
    if subset is None:
        return schedule
    if not isinstance(schedule, SteeringSchedule):
        raise ValueError("schedule must be a SteeringSchedule")
    present = {unit.sequence_id for unit in schedule.train}
    missing = sorted(set(subset.excluded) - present)
    if missing:
        raise ValueError(
            f"{len(missing)} training-subset exclusions are not in the schedule's train role, "
            f"e.g. {missing[:5]}. That is an exclusion list built against a different split: "
            "applying it would silently exclude fewer proteins than the arm claims"
        )
    dropped = set(subset.excluded)
    retained = tuple(unit for unit in schedule.train if unit.sequence_id not in dropped)
    if not retained:
        raise ValueError(
            "the training subset excludes every training sequence, leaving no train units at all"
        )
    return SteeringSchedule(train=retained, validation=schedule.validation, test=schedule.test)


@dataclass(frozen=True)
class OrdinarySteeringConfig:
    property_name: str
    model_family: ModelFamily
    intervention_site: str
    generation_callback_identity: str
    manifest_fingerprint: str
    property_callback_identity: str
    source_checkpoints: Mapping[str, str]
    target_range: TargetRange | None = None
    maximum_change: float | None = None
    target_mean: float | None = None
    target_scale: float | None = None
    target_kind: str = ABSOLUTE_SEQUENCE_TARGET
    property_reference: PropertyReferenceTable | None = None
    dose_multiplier: float | None = None
    sigma_eligibility: SigmaEligibility | None = None
    training_seed: int = 0
    baseline_path: Path | None = None
    maximum_unit_attempts: int = 2
    validation_target_mae_maximum: float | None = None
    validation_residual_fraction_maximum: float | None = None
    early_stopping_patience: int | None = None
    #: Clip the accumulated gradient to this norm before the step. ``None`` keeps the
    #: unclipped behaviour every earlier run used. ``ensemble_workflow`` already clips; this
    #: path did not, despite gradient norms spanning 3.5 to 86 across configurations.
    gradient_clip_norm: float | None = None
    gradient_accumulation_units: int = 1
    #: Data-parallel processes sharing one optimizer step. ``W`` multiplies the accumulation
    #: group rather than dividing it, so one step covers
    #: ``micro batch x W x gradient_accumulation_units`` scheduled units. See
    #: :mod:`sae_bioemu.steering.distributed`.
    distributed_world_size: int = 1
    #: Accept a batch plan that cannot co-locate every pair the paired dose term is weighted
    #: for. ``False`` refuses such a plan outright: at ``lambda_c`` 100x ``lambda_y`` a plan
    #: that pairs nothing deletes about 99% of the objective and nothing downstream says so.
    #: Set it and the concession is recorded in the checkpoint identity, which is the whole
    #: difference between an accepted trade-off and a silent one.
    allow_partial_pair_coverage: bool = False
    property_ensemble_samples: int = 1
    generation_precision: str = DEFAULT_GENERATION_PRECISION
    diagnostic_fixed_generation_seed: bool = False
    #: ``ResolvedCrypticArm.identity()`` for a cryptic-pocket arm, ``None`` for every other
    #: property. It reaches ``adapter_config`` -- conditionally, so a radius-of-gyration
    #: checkpoint keeps its exact historical key set -- because the pocket artifact defines
    #: what ``the progress coordinate`` IS: two arms over the same split with different pair sets are steering two
    #: different coordinates and their checkpoints must not be interchangeable.
    #:
    #: It must NOT reach ``schedule_seed_fingerprint`` (PLAN.md D-23). That value covers only
    #: what determines which units exist, which is what lets arms share noise draws; routing
    #: an artifact digest through it would redraw every schedule and confound each cross-arm
    #: difference with a complete re-randomisation whose measured noise floor (sd 0.0057 on
    #: slope) exceeds several differences worth claiming.
    cryptic_arm_identity: Mapping[str, Any] | None = None
    #: ANY OTHER property's resolved-arm identity, under the generic ``property_arm`` key. Set by
    #: that property's own ``checkpoint_arm_arguments`` hook; ``None`` for a run with no arm and
    #: for every cryptic run, which keeps its own spelling above.
    #:
    #: Separate from :attr:`cryptic_arm_identity` rather than replacing it, and that is not
    #: deference to history. That field carries a SECOND meaning -- ``steers_cryptic_pocket`` is
    #: derived from it, and that predicate is what puts the four D-35b monitor columns on a run's
    #: rows -- so a fourth property binding its arm there would silently acquire four journal
    #: columns it can compute no values for. Its persisted key is also in every cryptic checkpoint
    #: on disk.
    #:
    #: Like the cryptic one it must NOT reach ``schedule_seed_fingerprint`` (D-23): that value
    #: covers only what determines which units exist, which is what lets arms share noise draws.
    property_arm_identity: Mapping[str, Any] | None = None
    #: This property's steered dose set (D-44). The default is the continuous five-level set,
    #: so a radius-of-gyration or AF3 run builds, validates and records exactly the schedule it
    #: always did. A two-state property -- a cryptic pocket is open or closed, a kinase is
    #: DFG-in or DFG-out, so an intermediate dose names no structural ask -- passes
    #: ``TWO_STATE_DOSE_LEVELS``.
    dose_levels: DoseLevels = CONTINUOUS_DOSE_LEVELS
    #: Where level 0 sits on the relative path (D-46). ``sequence_mean`` is the historical
    #: behaviour and is what radius of gyration uses: its prediction is an absolute Rg, in the
    #: same frame as the reference table's mean. ``matched_draw_zero`` is for a property whose
    #: prediction is a DISPLACEMENT from a matched unsteered draw, where the zero point is zero
    #: by construction and an added mean is a frame error. ``matched_control_state`` is the
    #: TWO-STATE form: the ask is an absolute reporting-frame state (1.0 holo / 0.0 apo) converted
    #: into a training-frame displacement against the matched control's OWN reporting-frame value,
    #: so it carries neither ``k`` nor ``sigma``.
    relative_target_origin: str = TARGET_ORIGIN_SEQUENCE_MEAN
    #: D-23's training-subset exception. ``None`` is every run made before this field existed and
    #: every arm that trains on its whole split, and it writes no identity key at all. Set, it
    #: reaches ``adapter_config`` by both digests -- so the artifact cannot be edited in place
    #: under a finished checkpoint -- and it must NEVER reach ``schedule_seed_fingerprint``,
    #: which is the entire reason it is a config field rather than a different ``split.train``.
    #: See :class:`TrainingSubset`.
    training_subset: TrainingSubset | None = None
    #: How this run picks the adapter it ships, from :data:`MODEL_SELECTION_MODES`. The default
    #: is the historical ``validation_split``, so every run made before this field existed keeps
    #: its exact behaviour AND its exact identity key set -- ``_model_selection_identity`` writes
    #: nothing for it. ``none`` is the declared no-validation regime; see the constants above.
    model_selection: str = MODEL_SELECTION_VALIDATION_SPLIT
    #: The per-system ``{target, scale}`` table, REQUIRED on the
    #: :data:`~sae_bioemu.steering.schedule.TARGET_ORIGIN_SYSTEM_CONSTANT` origin and refused on
    #: every other. ``None`` for every run made before this field existed, which writes no identity
    #: key at all.
    #:
    #: It is what stands in for :attr:`property_reference` on that origin, and the substitution is
    #: exact rather than convenient. The reference table answers "where does this sequence's own
    #: adapter-off ensemble sit and how wide is it", which is the right question when the target is
    #: a displacement in units of that spread. Under a system constant the target is a THRESHOLD
    #: measured between two deposited structures and the scale is the distance between them;
    #: neither is a property of any ensemble, so a reference table would have to be built by
    #: generating ensembles to supply two numbers it does not contain.
    system_target_constants: SystemTargetConstants | None = None
    #: How many GENERATION FAILURES this run may step over, after ``maximum_unit_attempts``
    #: retries of each. ``0`` is the historical behaviour -- one exhausted unit ends the run --
    #: and writes no identity key at all.
    #:
    #: **Failures, not units.** What a skip COSTS is the whole accumulation group the failing draw
    #: sat in, because that group is the atomic resume unit; that cost is reported in full and is
    #: deliberately not what is capped. Capping it would make a run's tolerance depend on its batch
    #: plan, and it made the mechanism unreachable on every batched arm (a measured run: a group of
    #: 27 units refused against a budget of 8, with one failing draw and seven spare).
    #:
    #: The default is ``8``. It was chosen against a measured rate: a measured run walked 27,636
    #: committed training units and lost exactly ONE draw (``2qbv_A`` at seed 1511873054, a
    #: reconstructed backbone whose median C(i)-N(i+1) distance was 0.994 Angstrom), i.e. 3.6e-5.
    #: A 20-epoch schedule of that shape is ~80,000 units, so the expected number of such draws
    #: over a whole run is about three and eight is roughly 2.5x that -- while still being 1e-4 of
    #: the schedule, far too few to move any aggregate. Anything systematically broken produces
    #: far more than eight within a few groups and is stopped by the rate cap below long before
    #: this one is reached.
    #:
    #: **``None`` removes the ceiling and leaves the RATE below as the whole policy**, which is the
    #: spelling a new arm should prefer. The paragraph above is the calibration argument for 8, and
    #: every clause of it is about a schedule of "~80,000 units": on a 23,040-unit schedule the same
    #: 8 is 3.5e-4 rather than 1e-4, and because :func:`generation_failure_budget` takes the tighter
    #: of the two caps, 8 -- not the 0.005 the config declares -- is what binds everywhere below
    #: cursor 1,600. An absolute number is a different policy on every schedule length; a fraction
    #: is the same policy on all of them. The default stays ``8`` so no config already on disk
    #: changes meaning, and no checkpoint already written changes identity.
    maximum_generation_failures: int | None = 8
    #: The RATE half of the same budget: tolerated FAILURES as a fraction of the scheduled units
    #: walked so far. The two caps bind whichever is tighter (see
    #: :func:`generation_failure_budget`), so a run that loses several draws in its first few
    #: hundred units stops immediately instead of spending the absolute cap on a generator that is
    #: simply broken.
    #:
    #: The default is ``0.005``: 140x the measured 3.6e-5, so it never binds on a healthy run,
    #: and tight enough that a 1-in-100 failure rate ends the run inside a few hundred units.
    maximum_generation_failure_fraction: float = 0.005
    #: What this run picks its shipped epoch on, when its property declares something other than
    #: the shared default. ``None`` -- every run on disk -- keeps ``target_mae_sigma`` on the
    #: relative path and ``target_mae`` on the absolute one, and writes the same
    #: ``validation_selection_metric`` string those runs write.
    #:
    #: Resolved by the caller from the property's own hook and the OBJECTIVE, because whether a
    #: direct-metric arm selects on its metric depends on whether it prices it. It is carried here
    #: rather than re-derived at each use so that the aggregate key, the identity key, the resume
    #: check and the run report cannot end up naming three different quantities.
    validation_selection_metric: ValidationSelectionMetric | None = None
    #: An OPTIONAL narrowing of which checkpoint this run may adopt. Adoption itself is automatic
    #: -- a checkpoint in ``output_dir`` that passes every structural check is continued -- so this
    #: no longer permits anything. Declared, it additionally requires that the checkpoint on disk
    #: carries exactly the named fingerprint and spans exactly the named epoch budget.
    #:
    #: It is deliberately NOT an identity key. It reaches neither ``adapter_config`` nor
    #: ``schedule_seed_fingerprint``: it is an instruction about which checkpoint this run may
    #: adopt, not a statement about what the weights are, and putting it in ``adapter_config``
    #: would make a continued run's checkpoint unable to be resumed by an ordinary later run.
    #: What the artifact records instead is the REALISED
    #: :class:`~sae_bioemu.steering.schedule_extension.ScheduleExtension`, journalled under
    #: ``schedule_extensions``; see :mod:`sae_bioemu.steering.schedule_extension`.
    schedule_extension: ScheduleExtensionDeclaration | None = None

    #: Continue from the ``.best-N`` sidecar rather than from the main checkpoint. ``None`` --
    #: every run made before this field existed -- continues the main checkpoint, unchanged.
    #:
    #: This is what makes a rolled-back run continuable at all. A selecting run that terminates
    #: restores the adapter AND optimizer from ``best_validation_index``, so unless that was the
    #: final epoch the main checkpoint's weights are not the ones its cursor produced and the
    #: resume refuses. The sidecar for epoch N does not have that problem: it was written at the
    #: moment epoch N became the best, so its ``epoch``, ``unit_cursor``, optimizer state and
    #: journal all describe the same instant. Naming it resumes there.
    #:
    #: Not an identity key either, and for the same reason: it says where the run STARTED, which
    #: the journalled resume record is what preserves.
    resume_from_validation_index: int | None = None

    @property
    def selects_on_validation(self) -> bool:
        """True when an epoch is kept or discarded on a validation role.

        False means there is no validation role, no selection, no early stopping and no
        validation gate: the run walks its whole schedule and ships the final adapter.
        """
        return self.model_selection != MODEL_SELECTION_NONE

    @property
    def property_loss_normalization(self) -> str:
        """Return the normalisation this target construction requires the objective to declare.

        One function of the two things that decide it, so the config, the checkpoint identity,
        ``_loss_inputs`` and the objective cannot disagree about which of three objectives is
        running. :func:`_validate_inputs` compares this against the objective's own declaration
        and refuses a run where they differ.
        """
        return resolve_property_loss_normalization(self.target_kind, self.relative_target_origin)

    @property
    def targets_two_state(self) -> bool:
        """True when the target is an absolute two-state ask against the matched control.

        This is a statement about the TARGET CONSTRUCTION -- ``t = the progress coordinate_state(o) -
        the progress coordinate_reporting(control)``, no ``k`` and no ``mean`` -- and it selects the journal schema,
        the identity keys and the per-observation target conversion. It says nothing about what
        the loss divides by: that is :attr:`property_loss_normalization`, which under D-65 is
        ``per_channel_sigma_transition`` here and divides the absolute channel by ``sigma_the progress coordinate``
        while leaving the paired gap alone.

        It was called ``normalizes_by_transition`` while those two questions had one answer.
        Under per-channel they do not, and the old name would have described the PAIRED channel
        only, on a property whose every other consumer reads it as "which target construction".
        """
        return self.relative_target_origin == TARGET_ORIGIN_MATCHED_CONTROL_STATE

    @property
    def targets_system_constant(self) -> bool:
        """True when the target is this SYSTEM's own constant and the level picks a reference.

        The fourth target construction (see
        :data:`~sae_bioemu.steering.schedule.TARGET_ORIGIN_SYSTEM_CONSTANT`). It is deliberately
        NOT a sub-case of :attr:`targets_two_state`, even though both are walked at ``(-1, +1)``:
        that property names a specific construction, ``t = the progress coordinate_state(o) - the progress coordinate_reporting(control)``,
        and it selects the journal schema and the per-observation origin subtraction that go with
        it. A system-constant run records no ``target_state`` and no ``control_reporting_value``
        -- there is no reporting frame and no origin to subtract -- so it writes the ORDINARY
        relative row, and reading it as two-state would demand two columns it has no values for.
        """
        return self.relative_target_origin == TARGET_ORIGIN_SYSTEM_CONSTANT

    @property
    def scales_dose_by_sigma(self) -> bool:
        """True when the target is a multiple of a per-sequence sigma, i.e. when ``k`` exists.

        The question three separate rules key off: whether ``dose_multiplier`` is required, whether
        it is recorded in the checkpoint identity, and whether an intermediate dose level names
        anything. All three used to be spelled "is this run two-state", which was true of the
        progress coordinates by coincidence rather than because being two-state is what removes
        the multiplier -- and which would have demanded a ``dose_multiplier`` from a direct-metric
        run whose target carries no sigma at all.
        """
        return self.is_relative and self.relative_target_origin in SIGMA_DOSE_TARGET_ORIGINS

    @property
    def reads_property_reference(self) -> bool:
        """True when this run's targets, selection metric and journal read the reference table.

        False on the system-constant origin alone, where :attr:`system_target_constants` supplies
        both numbers the table would have supplied and the table would be an ensemble measurement
        of a quantity the objective never looks at.
        """
        return self.is_relative and not self.targets_system_constant

    @property
    def steers_cryptic_pocket(self) -> bool:
        """True when this run is a cryptic-pocket arm and carries its resolved arm identity."""
        return self.cryptic_arm_identity is not None

    @property
    def effective_accumulation_units(self) -> int:
        """The DECLARED product of accumulation units and ranks; no longer the step.

        Data parallelism and accumulation are the same operation done in parallel and in
        sequence, so their product used to define the optimizer step. It does not any more:
        an optimizer step is :data:`UNIFORM_OPTIMIZER_STEP_UNITS` scheduled units whatever
        this says, and ``resolve_gradient_accumulation`` warns when a config declares
        anything but 1. Kept because the declaration is still validated and reported.
        """
        return self.gradient_accumulation_units * self.distributed_world_size

    @property
    def distributes_gradients(self) -> bool:
        """True when more than one process contributes to each optimizer step."""
        return self.distributed_world_size > 1

    @property
    def accumulates_gradients(self) -> bool:
        """True when the config DECLARED more than one generation batch per step.

        Under the uniform optimizer step nearly every run accumulates whatever this says, so
        this is a statement about the configuration and not about the plan. The plan's own
        :attr:`GradientAccumulationPlan.accumulates` is the realised answer.
        """
        return self.effective_accumulation_units > 1

    @property
    def reduces_generation_precision(self) -> bool:
        """True when generation runs at anything other than full FP32."""
        return self.generation_precision != DEFAULT_GENERATION_PRECISION

    @property
    def averages_property_ensemble(self) -> bool:
        """True when one scheduled unit's property is a mean over several structures (D16)."""
        return self.property_ensemble_samples > 1

    @property
    def tolerates_generation_failures(self) -> bool:
        """True when this run may step over a generation failure at all, in either phase.

        ``maximum_generation_failures: 0`` is the one spelling that disables the tolerance, and it
        reproduces the pre-tolerance behaviour exactly -- one exhausted unit ends the run. ``None``
        is the opposite declaration: no absolute ceiling, the rate alone.
        """
        return self.maximum_generation_failures is None or self.maximum_generation_failures > 0

    @property
    def is_relative(self) -> bool:
        """True when targets are built from per-sequence statistics (D3/D4)."""
        return self.target_kind == RELATIVE_SEQUENCE_TARGET

    def __post_init__(self) -> None:
        for name in (
            "property_name",
            "intervention_site",
            "generation_callback_identity",
            "property_callback_identity",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.model_family, ModelFamily):
            raise ValueError("model_family must be a ModelFamily")
        _fingerprint(self.manifest_fingerprint, "manifest_fingerprint")
        _fingerprints(self.source_checkpoints)
        if set(self.source_checkpoints) != _SOURCE_CHECKPOINT_ROLES:
            raise ValueError("source checkpoint roles must be exactly generator and sae")
        # D-46: where level 0 sits. Validated here rather than at the first target, because a
        # target built on the wrong origin is a number, not an error -- it is off by a
        # per-protein constant and every downstream check passes.
        if self.relative_target_origin not in RELATIVE_TARGET_ORIGINS:
            raise ValueError(
                f"relative_target_origin must be one of {sorted(RELATIVE_TARGET_ORIGINS)}; "
                f"received {self.relative_target_origin!r}"
            )
        if self.validation_selection_metric is not None:
            if not isinstance(self.validation_selection_metric, ValidationSelectionMetric):
                raise ValueError(
                    "validation_selection_metric must be a ValidationSelectionMetric or None"
                )
            # The spelling has to be one the readers with no objective in hand admit. A metric a
            # run selected on but its own completion gate refuses is a checkpoint that can never
            # be attested, which is worse than refusing the run here.
            if self.validation_selection_metric.name not in selection_metric_names():
                raise ValueError(
                    f"{self.validation_selection_metric.name!r} is not a registered validation "
                    "selection metric; declare it in the steered property's "
                    "PropertyRegistration.selection_metric_names so the readers that validate a "
                    "persisted checkpoint can admit it"
                )
            if not self.selects_on_validation:
                raise ValueError(
                    f"{self.model_selection!r} runs no validation round, so a declared selection "
                    f"metric ({self.validation_selection_metric.name!r}) would name a comparison "
                    "this run never makes and would be recorded as though it had"
                )
        if (
            self.relative_target_origin
            in (
                TARGET_ORIGIN_MATCHED_DRAW_ZERO,
                TARGET_ORIGIN_MATCHED_CONTROL_STATE,
                TARGET_ORIGIN_SYSTEM_CONSTANT,
            )
            and self.target_kind != RELATIVE_SEQUENCE_TARGET
        ):
            raise ValueError(
                f"{self.relative_target_origin} is a relative-path origin: it says the target is "
                "built per sequence from something other than an absolute property range -- a "
                "matched unsteered draw, or this system's own threshold -- which the absolute "
                "path has no notion of. Setting it there would silently do nothing"
            )
        if self.target_kind not in _TARGET_KINDS:
            raise ValueError(
                "target_kind must be absolute_sequence or relative_sequence, found "
                f"{self.target_kind!r}"
            )
        if self.is_relative:
            supplied = tuple(
                name for name in _ABSOLUTE_ONLY_FIELDS if getattr(self, name) is not None
            )
            if supplied:
                raise ValueError(
                    f"{', '.join(supplied)} are unused on the relative_sequence path: the "
                    "target is mean_seq + level * dose_multiplier * sigma_seq and references "
                    "no baseline, no absolute range, and no normalization constants (D3/D4/D5)"
                )
            # Still REQUIRED on every relative path but the system-constant one, the
            # transition-normalized path included, and that is not an oversight now that sigma has
            # left the target. This runner reads the table for the eligibility band, for the
            # journal's ``sequence_sigma`` column and for the sigma-normalized validation selection
            # metric -- all reporting and selection, none of it the loss. What
            # ``matched_control_state`` removes is sigma's role as a TRAINING input;
            # ``cryptic_steering.arm`` is where the path itself became optional, because the
            # arm-level gates no longer need it.
            #
            # ``system_constant`` removes the table entirely, and that is a different statement
            # rather than a looser one: the target is a THRESHOLD between two deposited structures
            # and the scale is the DISTANCE between them, so neither number is a property of any
            # generated ensemble. A reference table here would have to be built by generating
            # ensembles to supply two numbers it does not contain, and the eligibility band it
            # carries would gate the pool on a spread the objective never reads --
            # ``domain_motion.separation`` is the gate that pool actually has.
            if self.reads_property_reference:
                if not isinstance(self.property_reference, PropertyReferenceTable):
                    raise ValueError(
                        "relative_sequence targets require a loaded PropertyReferenceTable"
                    )
                if self.property_reference.property_name != self.property_name:
                    raise ValueError(
                        "property reference table property_name does not match the steered property"
                    )
            elif self.property_reference is not None or self.sigma_eligibility is not None:
                raise ValueError(
                    f"property_reference and sigma_eligibility mean nothing on the "
                    f"{TARGET_ORIGIN_SYSTEM_CONSTANT!r} origin: the target is this system's own "
                    "threshold and the scale is its own reference separation, both from "
                    "system_target_constants, so a table supplied here would be loaded, "
                    "fingerprinted and read by nothing"
                )
            if self.targets_system_constant:
                if not isinstance(self.system_target_constants, SystemTargetConstants):
                    raise ValueError(
                        f"the {TARGET_ORIGIN_SYSTEM_CONSTANT!r} origin requires "
                        "system_target_constants: the target IS this system's own threshold and "
                        "the property loss scale IS its own reference separation, and neither is "
                        "derivable from a schedule, an adapter or a structure"
                    )
            elif self.system_target_constants is not None:
                raise ValueError(
                    "system_target_constants is the per-system target table of the "
                    f"{TARGET_ORIGIN_SYSTEM_CONSTANT!r} origin and is read nowhere else; on "
                    f"{self.relative_target_origin!r} the target is built from the reference "
                    "table, so a table set here would be fingerprinted and applied to nothing"
                )
            if not self.scales_dose_by_sigma:
                # ``k`` and ``sigma`` are both gone from these targets, so a value for either is a
                # setting that would not apply. Refused rather than ignored, for the reason every
                # other half-configuration in this file is refused: a knob that is accepted and
                # then dropped is how ``lambda_l`` voided six arms.
                if self.dose_multiplier is not None:
                    raise ValueError(
                        "dose_multiplier is a sigma-relative dose and means nothing on the "
                        f"{self.relative_target_origin!r} origin: a level there names a STATE or a "
                        "REFERENCE CONFORMER rather than a distance to travel, so there is no "
                        "multiple of a standard deviation left to set"
                    )
                zero_levels = [level for level in self.dose_levels.target if float(level) == 0.0]
                if zero_levels:
                    # Worded per ORIGIN rather than generically, because what ``0.0`` fails to name
                    # is different on each: there is no half-open conformation for a state target,
                    # and no third deposited structure for a reference-selecting one. A shared
                    # message would say "names neither", which tells a reader nothing about which
                    # of the two they configured.
                    raise ValueError(
                        "a two-state dose level names a state, so every target level must be "
                        "exactly -1 or +1; this dose set contains 0.0, which names no state"
                        if self.targets_two_state
                        else (
                            "a system-constant dose level names a REFERENCE CONFORMER, so every "
                            "target level must be exactly -1 or +1; this dose set contains 0.0, "
                            "and there is no third deposited structure for it to name"
                        )
                    )
            elif _finite(self.dose_multiplier, "dose_multiplier") <= 0:
                raise ValueError("dose_multiplier must be positive")
            if not self.reads_property_reference:
                pass
            elif self.sigma_eligibility is None:
                # Never left absent: the applied band is part of the checkpoint identity,
                # so two runs with different eligibility can never collide.
                object.__setattr__(self, "sigma_eligibility", DEFAULT_SIGMA_ELIGIBILITY)
            elif not isinstance(self.sigma_eligibility, SigmaEligibility):
                raise ValueError("sigma_eligibility must be a SigmaEligibility")
            if self.validation_target_mae_maximum is not None:
                raise ValueError(
                    "validation_target_mae_maximum is an absolute-unit gate; the "
                    "relative_sequence path selects and gates on sigma-normalized error "
                    "and has no configured absolute gate (D4b)"
                )
        else:
            if (
                self.property_reference is not None
                or self.dose_multiplier is not None
                or self.sigma_eligibility is not None
                or self.system_target_constants is not None
            ):
                raise ValueError(
                    "property_reference, dose_multiplier, sigma_eligibility, and "
                    "system_target_constants are valid only for relative_sequence targets"
                )
            if not isinstance(self.target_range, TargetRange):
                raise ValueError("target_range must be a TargetRange")
            _finite_nonnegative(self.maximum_change, "maximum_change")
            _finite(self.target_mean, "target_mean")
            if _finite(self.target_scale, "target_scale") <= 0:
                raise ValueError("target_scale must be positive")
        if (
            not isinstance(self.training_seed, int)
            or isinstance(self.training_seed, bool)
            or not 0 <= self.training_seed < 2**32
        ):
            raise ValueError("training_seed must be a uint32 integer")
        if (
            not isinstance(self.maximum_unit_attempts, int)
            or isinstance(self.maximum_unit_attempts, bool)
            or self.maximum_unit_attempts < 1
        ):
            raise ValueError("maximum_unit_attempts must be a positive integer")
        if self.maximum_generation_failures is not None and (
            not isinstance(self.maximum_generation_failures, int)
            or isinstance(self.maximum_generation_failures, bool)
            or self.maximum_generation_failures < 0
        ):
            raise ValueError(
                "maximum_generation_failures must be a nonnegative integer, or null to leave the "
                "rate as the whole policy"
            )
        if (
            isinstance(self.maximum_generation_failure_fraction, bool)
            or not isinstance(self.maximum_generation_failure_fraction, int | float)
            or not math.isfinite(self.maximum_generation_failure_fraction)
            or not 0.0 <= self.maximum_generation_failure_fraction <= 1.0
        ):
            raise ValueError("maximum_generation_failure_fraction must be a fraction in [0, 1]")
        if self.maximum_generation_failures is None and not (
            self.maximum_generation_failure_fraction > 0.0
        ):
            # Removing the ceiling makes the rate the whole policy, so a zero rate beside it would
            # be a run declaring a tolerance and getting none. That is a config error rather than a
            # disabled tolerance: ``maximum_generation_failures: 0`` is how a run says "no
            # tolerance", and it says so unambiguously.
            raise ValueError(
                "maximum_generation_failures: null leaves maximum_generation_failure_fraction as "
                "the whole tolerance policy, so the fraction must be positive; set "
                "maximum_generation_failures: 0 to disable the tolerance instead"
            )
        if (
            not isinstance(self.gradient_accumulation_units, int)
            or isinstance(self.gradient_accumulation_units, bool)
            or self.gradient_accumulation_units < 1
        ):
            raise ValueError("gradient_accumulation_units must be a positive integer")
        if (
            not isinstance(self.distributed_world_size, int)
            or isinstance(self.distributed_world_size, bool)
            or self.distributed_world_size < 1
        ):
            raise ValueError("distributed_world_size must be a positive integer")
        if (
            not isinstance(self.property_ensemble_samples, int)
            or isinstance(self.property_ensemble_samples, bool)
            or self.property_ensemble_samples < 1
        ):
            raise ValueError("property_ensemble_samples must be a positive integer")
        # Validated by resolution rather than by an allowlist here, so the accepted set is
        # defined in exactly one place.
        resolve_generation_precision(self.generation_precision)
        if self.averages_property_ensemble and not self.is_relative:
            # D16's estimator is defined against D3's target, which IS the sequence's
            # ensemble-mean property; comparing mean_m(Rg_m) to it compares like with like.
            # The absolute path's target is a per-unit baseline shift with different
            # semantics, so an ensemble mean there would be a different, unreviewed
            # objective rather than a lower-variance estimate of this one.
            raise ValueError(
                "property_ensemble_samples > 1 is defined only for relative_sequence "
                "targets: the ensemble-mean property estimator compares mean_m(property_m) "
                "against a target that is itself the sequence's ensemble mean (D3/D16)"
            )
        if not isinstance(self.diagnostic_fixed_generation_seed, bool):
            raise ValueError("diagnostic_fixed_generation_seed must be a boolean")
        if self.early_stopping_patience is not None and (
            not isinstance(self.early_stopping_patience, int)
            or isinstance(self.early_stopping_patience, bool)
            or self.early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be a positive integer or null")
        if self.resume_from_validation_index is not None and (
            not isinstance(self.resume_from_validation_index, int)
            or isinstance(self.resume_from_validation_index, bool)
            or self.resume_from_validation_index < 0
        ):
            raise ValueError(
                "resume_from_validation_index must be a nonnegative integer or null: it names a "
                "``.best-N`` sidecar, and N is a validation round index"
            )
        for name in ("validation_target_mae_maximum", "validation_residual_fraction_maximum"):
            value = getattr(self, name)
            if value is not None:
                _finite_nonnegative(value, name)
        if self.model_selection not in MODEL_SELECTION_MODES:
            raise ValueError(
                f"model_selection must be one of {sorted(MODEL_SELECTION_MODES)}; "
                f"received {self.model_selection!r}"
            )
        if not self.selects_on_validation:
            # REFUSED, not ignored, and for the reason every other half-configuration in this
            # file is refused. Each of these three reads a completed validation round: early
            # stopping counts rounds that did not improve, and both gates are applied to the
            # SELECTED round's aggregate. With no validation role no round is ever run, so each
            # would be a knob that is parsed, recorded in the checkpoint identity, and can never
            # fire -- which is how a saturated-residual gate sat unreachable through two pilots
            # that pinned the residual to rho_max on 188 of 200 units and still reported
            # ``complete: true``.
            unusable = tuple(
                name
                for name in (
                    "early_stopping_patience",
                    "validation_target_mae_maximum",
                    "validation_residual_fraction_maximum",
                )
                if getattr(self, name) is not None
            )
            if unusable:
                raise ValueError(
                    f"{', '.join(unusable)} select, stop or gate on a completed validation "
                    f"round, and model_selection={MODEL_SELECTION_NONE!r} runs none: this run "
                    "has no validation role, so each would be configured, recorded and never "
                    "applied. Set them to null, or declare a validation split"
                )
            if self.resume_from_validation_index is not None:
                # Same refusal, same reason: a ``.best-N`` sidecar is written when validation
                # round N becomes the best, and a run that never validates writes none. Naming one
                # here points at a file that cannot exist.
                raise ValueError(
                    "resume_from_validation_index names a ``.best-N`` sidecar written by a "
                    f"validation round, and model_selection={MODEL_SELECTION_NONE!r} runs none"
                )


def generation_failure_budget(config: OrdinarySteeringConfig, units_seen: int) -> int:
    """How many GENERATION FAILURES a run may have stepped over by unit ``units_seen``.

    **The unit of this budget is the FAILURE, not the scheduled unit, and getting that wrong is
    what made the mechanism unreachable on every batched arm.** Job that run died at train unit 798
    of 2910 with a budget of 8 and exactly ONE failing draw, because the check compared the budget
    against the number of scheduled units the skip would cost -- and on that arm the accumulation
    group at cursor 798 held 27 units, so ``27 > 8`` refused a skip the run had ample budget for.
    Batch 0 there holds 48. The tolerance could never fire.

    The two rates this budget was calibrated from are PER DRAW -- 3.6e-5 measured on a measured run
    and 6.7e-5 from D-88b's uncensored population -- so the quantity they bound is the number of
    failing draws, which is the number of skip EVENTS. What a skip COSTS is a whole accumulation
    group, because that is the atomic resume unit: the group is generated, back-propagated and
    committed as one, so there is no state in which half of it was trained on. That cost is
    reported in full (``generation_units_skipped``, and each skip row's own ``size``) and is
    deliberately not what is capped, because capping it would make a run's tolerance depend on its
    batch plan -- two runs over one schedule at two batch sizes would tolerate different numbers of
    bad draws, which is not a property anyone chose.

    Two caps, and the tighter one binds:

    * ``maximum_generation_failures`` -- an OPTIONAL absolute ceiling of failures on the whole run,
      so a very long schedule cannot accumulate hundreds of skips at an individually harmless rate;
      and
    * ``maximum_generation_failure_fraction`` x ``units_seen`` -- a RATE of failures per unit
      walked, so a generator that is simply broken is stopped in its first few hundred units rather
      than after hours of retrying.

    **THE RATE IS THE POLICY; THE ABSOLUTE CEILING IS OPTIONAL.** ``None`` there leaves the rate
    alone binding, and that spelling exists because an absolute number calibrated against one
    schedule length silently becomes a different policy on another. The default ``8`` was chosen in
    :attr:`OrdinarySteeringConfig.maximum_generation_failures` against an "~80,000 unit" schedule,
    where it is 1e-4 of the plan; on the 23,040-unit cryptic arm the same 8 is 3.5e-4, and because
    ``min`` takes the tighter cap it -- not the declared rate -- is what the run actually ran under
    at every cursor below 1,600. A run that means "stop when more than half a percent of my draws
    fail" states the fraction and sets the ceiling to ``null``, and then the tolerance it gets is
    the tolerance it wrote down whatever the schedule's length.

    The rate term carries a floor of one. Without it the mechanism would be vacuous at the start of
    every run (``floor(0.005 x 0)`` is zero), which is precisely where a genuinely isolated bad draw
    is as likely as anywhere else; with it, the smallest tolerance any enabled budget ever grants is
    a single failure. A budget of ``0`` -- ``maximum_generation_failures = 0`` -- disables the whole
    mechanism and reproduces the pre-tolerance behaviour exactly, floor included.

    ``units_seen`` is the unit CURSOR at the moment the skip is taken, which is replayable from the
    journal: every skip records the cursor it sat at, so a later reader can re-derive the budget
    each skip was granted under and refuse a journal that exceeded it.
    """
    if not config.tolerates_generation_failures:
        return 0
    if units_seen < 0:
        raise ValueError("units_seen must be nonnegative")
    rate = max(1, math.floor(float(config.maximum_generation_failure_fraction) * float(units_seen)))
    if config.maximum_generation_failures is None:
        return int(rate)
    return min(int(config.maximum_generation_failures), int(rate))


#: Schema tag inside the resample seed derivation, so a later change to what the substitute seed is
#: derived FROM is visible as a different seed rather than as a silent re-randomisation.
_RESAMPLE_SEED_SCHEMA = "steering_generation_resample/1"


def resampled_generation_seed(
    attempt_key: str, sequence_id: str, generation_seed: int, resample_index: int
) -> int:
    """Return the noise draw attempt ``resample_index`` at ``attempt_key`` uses.

    **``resample_index == 0`` returns ``generation_seed`` unchanged**, so a run's first attempt at
    every group draws exactly the seed its schedule assigned and a healthy run is bit-identical to
    one made before resampling existed.

    Every later attempt draws a DERIVED substitute rather than the same seed again. Retrying a
    deterministic generation failure with the same input cannot rescue it -- a measured run died at
    ``validation:18:0`` with two recorded attempts reading ``2o4n_B seed=588648756 -> median
    C(i)-N(i+1) 1.868 A`` twice, bit for bit, having spent both of ``maximum_unit_attempts`` to
    reproduce one number. A different draw can and usually does: that same sequence generated
    successfully in all 18 preceding rounds.

    DERIVED, never random. The substitute is a pure function of the attempt key, the sequence, the
    seed that failed and the attempt ordinal, so a replay of this run lands on the identical
    substitute and the run stays exactly reproducible from its recorded inputs. Seeding from the
    clock or from ``os.urandom`` would rescue the draw and destroy that, which is not a trade this
    project's reproducibility rules permit.

    **It is derived from the GENERATION seed, not the unit's identity seed, and that is what keeps
    the paired dose term intact.** The two levels of one ``(sequence, cycle)`` draw group share a
    ``draw_seed`` deliberately -- the difference taken across levels at the same draw cancels the
    baseline, measured SNR 57 paired against 0.62 unpaired -- and they share ``sequence_id`` too,
    so both members derive the SAME substitute and stay paired. Deriving from ``unit.seed``, which
    is unique per unit, would silently unpair every resampled group.
    """
    if (
        not isinstance(resample_index, int)
        or isinstance(resample_index, bool)
        or resample_index < 0
    ):
        raise ValueError("resample_index must be a nonnegative integer")
    if (
        not isinstance(generation_seed, int)
        or isinstance(generation_seed, bool)
        or not 0 <= generation_seed < 2**32
    ):
        raise ValueError("generation_seed must be a uint32 integer")
    seed = int(generation_seed)
    if resample_index == 0:
        return seed
    digest = hashlib.sha256(
        json.dumps(
            {
                "schema": _RESAMPLE_SEED_SCHEMA,
                "attempt_key": str(attempt_key),
                "sequence_id": str(sequence_id),
                "seed": seed,
                "resample_index": int(resample_index),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


#: Schema tag on the resample-rate block, so a reader can tell which definition of the denominator
#: a report was written under.
GENERATION_RESAMPLE_RATE_SCHEMA = "steering_generation_resample_rate/1"


def generation_resample_rate(journal: Mapping[str, Any], unit_cursor: int) -> dict[str, Any]:
    """How often this run had to REDRAW a generation, per scheduled unit walked.

    **THIS IS A SCIENTIFIC GUARD, NOT BOOKKEEPING, and it is the whole price of resampling.**
    Redrawing a failed generation until it passes is curation: the metrics that come out are
    computed over the draws that survived a geometry check, and if the adapter starts breaking 30%
    of its draws the run will quietly report a clean number over a quietly selected sample. D-92
    (``docs/domain_motion/d92_the_dose_is_breaking_backbones.txt``) establishes that the dose grows
    and breaks backbones LATE in training, which is exactly when a rate like this stops being
    hypothetical -- a measured run's own journal shows its four generation failures at epochs 15, 17,
    17 and 18 of 24 and none before. So the rate is surfaced in the run report and in the journal,
    where it REVEALS the damage instead of absorbing it, and a rising value is the signal that the
    dose, not the generator, is the problem.

    The denominator is ``unit_cursor``: scheduled TRAIN units walked. That is deliberately the same
    denominator ``maximum_generation_failure_fraction`` is defined against, so ``resample_rate`` and
    the configured cap can be read against each other directly and a reader never has to ask which
    of two rates a number is.

    The numerator counts FAILED DRAWS -- one per recorded generation failure, in either phase --
    because one failure is one redraw. It is not the number of scheduled units lost: a failure takes
    its whole batch down with it, and reporting the cost as though it were the count is precisely
    the confusion :func:`generation_failure_budget` documents. Both are reported, separately.
    """
    failures = journal.get("failures") or ()
    train = sum(1 for row in failures if row.get("phase") == "generation")
    validation = sum(1 for row in failures if row.get("phase") == _VALIDATION_GENERATION_PHASE)
    walked = max(int(unit_cursor), 0)
    return {
        "schema": GENERATION_RESAMPLE_RATE_SCHEMA,
        # Every generation failure, each of which cost one redraw at a fresh derived seed.
        "resampled_draws": train + validation,
        "train_resampled_draws": train,
        "validation_resampled_draws": validation,
        "units_walked": walked,
        "resample_rate": (train + validation) / walked if walked else 0.0,
        # What the redraws could NOT rescue, on one shared budget: train groups stepped over plus
        # validation batches granted a fresh set of draws.
        "tolerated_failures": _tolerated_failure_count(journal),
        "train_groups_skipped": len(_skipped_units(journal)),
        "train_units_skipped": _skipped_unit_count(journal),
        "validation_batches_retried": len(_validation_retries(journal)),
    }


def _resampled_units(
    members: Sequence[SteeringUnit], attempt_key: str, resample_index: int
) -> tuple[SteeringUnit, ...]:
    """Return the units handed to the generator, with substitute noise draws on a retry.

    The scheduled units themselves are never rewritten -- records, failures and the journal keep
    the schedule's own seeds, so the committed prefix still equals the schedule and a resampled run
    stays auditable against it. This is exactly the contract :func:`_generation_units` and the
    property-ensemble seed table already work under.
    """
    members = tuple(members)
    if resample_index <= 0:
        return members
    return tuple(
        replace(
            unit,
            draw_seed=resampled_generation_seed(
                attempt_key, unit.sequence_id, unit.generation_seed, resample_index
            ),
        )
        for unit in members
    )


@dataclass(frozen=True)
class BaselineRecord:
    sequence_id: str
    value: float


@dataclass(frozen=True)
class SteeringTrainingRecord:
    """One committed training unit, in the fields its target construction defines.

    The **absolute** path keeps its exact historical fields, so its journal rows are
    byte-identical to before.

    The **relative** path (D1/D3/D4b) differs in three deliberate ways:

    * ``target_loss`` is renamed ``target_loss_sigma_squared``. The property term is
      ``((Rg - t) / sigma_seq)^2``, which is in multiples of that sequence's own sigma
      and no longer in the property's squared units. A units column would have to be
      re-read by every consumer that copies the value into a table or a plot sidecar;
      putting the unit in the name makes it travel with the number and turns a stale
      reader into a loud ``KeyError`` instead of a silently mislabelled axis. This is
      the same move as renaming ``success_tolerance`` to ``success_tolerance_sigma``,
      applied where a per-row value rather than one report-level scalar changed units.
    * ``achieved_value`` and ``signed_error`` are recorded. The loss alone cannot answer
      what the adapter actually produced: it recovers ``|Rg - t| / sigma_seq`` and
      neither the sign nor the scale, and the sign is what says whether steering moved
      the structure the requested way.
    * ``geometry_loss`` used not to be persisted, and neither was the local-structure
      component or the paired dose term. The omission was written when ``SteeringLoss``
      hard-enforced ``lambda_g == 0`` on this path, making it a constant zero column; the guard
      was lifted, arms began training with ``lambda_g > 0`` and ``lambda_c`` up to 700x
      ``lambda_y``, and the journal went on dropping all three. ``total_loss`` included them
      and nothing could decompose it. They are now persisted as
      :data:`_LOSS_TERM_RECORD_KEYS`, each gated on the weight that prices it, so a run with
      ``lambda_g == lambda_c == lambda_l == 0`` writes rows byte-identical to today's.

    The **two-state** path (``matched_control_state``) is the relative schema plus two columns,
    and both additions are forced by the target construction rather than by the objective:

    * ``target`` stays the quantity the prediction was compared against, which is per
      OBSERVATION rather than per unit, so it can no longer be re-derived from the schedule.
      :attr:`target_state` and :attr:`control_reporting_value` are persisted beside it and it is
      re-derived from THEM: the state is a pure function of the level and is checked against the
      schedule, the origin is a per-draw measurement that only the journal can preserve.
    * ``signed_error`` keeps its meaning exactly -- ``achieved - target`` in the training frame --
      and keeps it only because ``target`` is the training-frame target rather than the state.

    **It carried a third difference under D-64 and no longer does.** While the two-state target
    term was ``(the progress coordinate - t)^2`` with no denominator, ``target_loss_sigma_squared`` would have been a
    name contradicting its contents, so the column was renamed
    ``target_loss_transition_squared``. Under D-65 the absolute channel divides by ``sigma_the progress coordinate``
    again, so the recorded value IS ``((the progress coordinate - t)/sigma)^2`` and the sigma-named column is the
    correct one -- the same construction Rg records, differing only in how ``t`` was built, which
    the two extra columns state. The renamed column is REMOVED rather than kept as an alias:
    nothing has ever trained on this path, so no journal on disk moves, and an accepted-but-never-
    written column is a second name for one quantity waiting to be filled by the wrong one.
    """

    unit: SteeringUnit
    target: float
    residual_loss: float
    total_loss: float
    residual_fraction: float
    target_loss: float | None = None
    geometry_loss: float | None = None
    achieved_value: float | None = None
    signed_error: float | None = None
    sequence_sigma: float | None = None
    target_loss_sigma_squared: float | None = None
    #: The absolute REPORTING-frame state this level asked for: 1.0 holo / DFG-out, 0.0 apo /
    #: DFG-in. A pure function of ``unit.level``, so it is what the schedule is re-checked
    #: against. Two-state path only.
    target_state: float | None = None
    #: ``the progress coordinate_reporting(control)`` for THIS draw: the origin the state was converted through, and
    #: the one number in the row that cannot be recomputed without regenerating the control.
    #: Two-state path only.
    control_reporting_value: float | None = None
    #: UNWEIGHTED ``lambda_c`` term for the micro-batch this unit was optimised in, present
    #: only when ``lambda_c`` is nonzero. Per BATCH and not per unit, and that is forced by the
    #: quantity: the paired dose term is a mean over unordered PAIRS of units sharing a draw, so
    #: a single unit has no share of it to record. ``0.0`` therefore means "this batch formed no
    #: pair", which is what the objective priced it at, not "this unit contributed nothing".
    #: Divided by sigma under ``sequence_sigma`` and undivided under the transition
    #: normalisations -- the checkpoint's own ``property_loss_normalization`` says which.
    #:
    #: ``total_loss`` on the same row carries ``lambda_c`` times this value AT FULL WEIGHT, as
    #: it does for the other two per-batch terms, so ``mean(total_loss)`` over a micro-batch is
    #: the total that was optimised and ``sum(total_loss)`` over-counts all three. See
    #: :func:`_batch_training_records` for why full attribution rather than an equal split.
    contrastive_loss: float | None = None
    #: UNWEIGHTED smooth LDDT against the unsteered draw, present only when ``lambda_l`` is
    #: nonzero. Also per batch: the generator produces one whole-batch scalar, which
    #: ``_batch_training_records`` already reuses verbatim for every member's re-evaluation.
    #: A value supplied at ``lambda_l == 0`` is a free diagnostic the objective does not price
    #: and is deliberately NOT journalled, so the column means "this term was in the total".
    local_structure_loss: float | None = None
    #: The PROJECT-declared off-path columns this row carries, keyed by column name: the value a
    #: property's own objective reported for a term its own weight priced. Absent keys and
    #: ``None`` mean the same thing -- the term did not run -- and the serializer omits them, so a
    #: run pricing none of them writes a row byte-identical to one written before any project
    #: existed.
    #:
    #: **A mapping rather than one field per column, and that is what makes this the last shared
    #: edit.** ``perp_loss``, ``rigidity_loss`` and ``dihedral_loss`` were three declared fields
    #: here, in a file in the shared ``training_only`` group, so a fourth column could not be
    #: persisted without re-keying every property that cannot write it. The column VOCABULARY now
    #: comes from the registry (:data:`_OFF_PATH_COLUMN_WEIGHTS`) and the values land here, so a
    #: fifth property's column costs a catalog line.
    #:
    #: :meth:`__getattr__` keeps ``record.perp_loss`` working and returning ``None`` for a term
    #: that did not run, so every reader, plot script and validator that spells a column as an
    #: attribute is unchanged -- including this module's own ``getattr(record, name)`` loops.
    project_columns: Mapping[str, float] = field(default_factory=dict)
    #: The four D-35b pocket-local geometry MONITOR counts, present only on a cryptic-pocket arm
    #: and absent on every other run, exactly as ``perp_loss`` and ``rigidity_loss`` are absent
    #: when their weight is zero. Integers, because the whole point of D-35b is that a localised
    #: breach is ONE clash rather than ``1/L^2`` of a mean.
    #:
    #: They select nothing. ``_validation_aggregate`` does not read them, ``_validation_key``
    #: does not read them, and no gate consults them; making them select would be a separate
    #: decision with its own record.
    gate_scored_out_of_window_bonds: int | None = None
    gate_control_out_of_window_bonds: int | None = None
    gate_scored_clashing_pairs: int | None = None
    gate_control_clashing_pairs: int | None = None
    #: The registry-declared UNPRICED MONITOR columns (D-86), by name. The monitor twin of
    #: :attr:`project_columns`, and separate from it because the two are gated on different
    #: things: a project column is present exactly when its WEIGHT is nonzero, and a monitor has
    #: no weight at all -- its presence is the property's own hook's statement.
    #:
    #: A mapping rather than declared fields so that a property adding a monitor costs a catalog
    #: line and its own hook, and not an edit to this shared module. :meth:`__getattr__` answers
    #: for every registered monitor name, so ``record.pocket_off_axis_ratio_squared`` works and
    #: returns ``None`` on a row that does not carry it.
    monitor_columns: Mapping[str, float] = field(default_factory=dict)

    def __getattr__(self, name: str) -> float | None:
        """Resolve a REGISTERED project column spelled as an attribute.

        Called only when normal attribute lookup fails, so it costs nothing on any declared field
        and cannot shadow one. It answers for exactly the columns
        :data:`_OFF_PATH_RECORD_KEYS` names and raises :class:`AttributeError` for anything else:
        returning ``None`` for an arbitrary name would turn every typo in a reader into a column
        that silently reads as "this term did not run".

        ``None`` for a declared column this row does not carry, which is what the three fields
        this replaced returned and what every existing reader is written against.
        """
        if name in _OFF_PATH_RECORD_KEYS:
            return self.project_columns.get(name)
        if name in _DECLARED_MONITOR_RECORD_KEYS:
            return self.monitor_columns.get(name)
        raise AttributeError(name)

    @property
    def is_relative(self) -> bool:
        """True when this row uses one of the two relative-target schemas."""
        return self.target_loss_sigma_squared is not None

    @property
    def is_two_state(self) -> bool:
        """True when this row uses the two-state schema, on the column only it can carry.

        Keyed on ``target_state`` rather than on the target-loss column, because since D-65
        both relative schemas record ``target_loss_sigma_squared`` and the two-state one is the
        one that ALSO records the state and the per-draw origin. That is the honest
        discriminator: it is the pair of columns the target construction adds.
        """
        return self.target_state is not None


@dataclass(frozen=True)
class OrdinaryTrainingResult:
    records: tuple[SteeringTrainingRecord, ...]
    failures: tuple[SteeringFailure, ...]
    baselines: tuple[BaselineRecord, ...]
    checkpoint: AdapterCheckpoint
    validation_record: Mapping[str, Any]
    complete: bool
    checkpoint_path: Path
    #: Every TRAIN group whose generation was tolerated away, in schedule order. Empty on a run
    #: that skipped nothing, which is what the journal's absent ``skipped_units`` key means.
    #: Surfaced here rather than left inside ``validation_record`` so the run report, the log and
    #: any later reader take the count from one place.
    skipped_units: tuple[Mapping[str, Any], ...] = ()

    @property
    def skipped_unit_count(self) -> int:
        """How many scheduled train UNITS -- not groups -- this run never trained on.

        The COST of the skips. What the budget counts is :attr:`skipped_failure_count`.
        """
        return sum(int(row["size"]) for row in self.skipped_units)

    @property
    def skipped_failure_count(self) -> int:
        """How many GENERATION FAILURES this run stepped over: the quantity the budget caps."""
        return len(self.skipped_units)

    @property
    def validation_retries(self) -> tuple[Mapping[str, Any], ...]:
        """Every VALIDATION batch this run redrew after exhausting its attempts, in order."""
        return tuple(dict(row) for row in _validation_retries(self.validation_record))

    @property
    def generation_resample_rate(self) -> dict[str, Any]:
        """How often this run had to redraw a generation, as a RATE. See the module function."""
        return generation_resample_rate(self.validation_record, self.checkpoint.unit_cursor)


@dataclass(frozen=True)
class _BaselineState:
    records: Mapping[str, float]
    failures: tuple[SteeringFailure, ...]
    path: Path | None
    content_fingerprint: str | None
    complete: bool


#: The relative path never generates a baseline, so its state is empty and complete.
_NO_BASELINES = _BaselineState({}, (), None, None, True)

BaselineCallback = Callable[[SteeringUnit], float]
GenerateCallback = Callable[
    [SteeringUnit, StandardizedAbsoluteTarget | RequestedLevel], GeneratedTrainingSample
]
#: Batched generator: every returned tensor is ordered by batch member, and the property
#: prediction must be exactly one value per member.
GenerateBatchCallback = Callable[
    [
        tuple[SteeringUnit, ...],
        tuple[StandardizedAbsoluteTarget | RequestedLevel, ...],
    ],
    GeneratedTrainingSample,
]

#: Persisted training-record key sets, one per target construction.
_ABSOLUTE_RECORD_KEYS = frozenset(
    {
        "unit",
        "target",
        "target_loss",
        "residual_loss",
        "geometry_loss",
        "total_loss",
        "residual_fraction",
    }
)
_RELATIVE_RECORD_KEYS = frozenset(
    {
        "unit",
        "target",
        "achieved_value",
        "signed_error",
        "sequence_sigma",
        "target_loss_sigma_squared",
        "residual_loss",
        "total_loss",
        "residual_fraction",
    }
)
#: The two-state relative schema (``matched_control_state``): the relative set plus the two
#: columns the state target adds. It is a third set and not a variant, because ``_records``
#: compares a persisted row's key set with EQUALITY -- so a relative row that acquired a
#: ``target_state``, or a two-state row that lost its origin, fails closed by shape rather than
#: being parsed as the other schema.
#:
#: Under D-64 it also RENAMED the target-loss column, to ``target_loss_transition_squared``,
#: because the term was then ``(the progress coordinate - t)^2`` with no denominator. D-65 divides that channel by
#: ``sigma_the progress coordinate`` again, so the value is a sigma-squared quantity and the sigma-named column is the
#: one that describes it. Nothing has ever trained on this path and no cryptic runtime config
#: exists, so no journal on disk moves.
_TWO_STATE_RECORD_KEYS = frozenset(
    {
        "unit",
        # The reporting-frame ask, re-derivable from the level, and what the schedule is checked
        # against.
        "target_state",
        # This draw's own origin. NOT re-derivable: it is a measurement of a generated control.
        "control_reporting_value",
        # The training-frame target the prediction was actually compared against, which is
        # ``target_state - control_reporting_value`` and is re-checked as that identity.
        "target",
        "achieved_value",
        "signed_error",
        # A loss input again since D-65, and it was always the eligibility band, the reachability
        # strata and every reported 1/sd(the progress coordinate). The column and its meaning did not move when its
        # role did.
        "sequence_sigma",
        "target_loss_sigma_squared",
        "residual_loss",
        "total_loss",
        "residual_fraction",
    }
)

#: The cryptic-pocket off-path columns (PLAN.md D-28), one gated on each of ``lambda_perp``
#: and ``lambda_rigid`` exactly as ``_objective_config`` gates the weights themselves.
#:
#: They are OPTIONAL for a reason that is not stylistic. ``_records`` compares a persisted
#: row's key set with frozenset equality, so an unconditional column would make every
#: journal written before it existed -- including the one every currently running
#: radius-of-gyration arm is appending to right now -- fail to parse, and the failure would
#: surface as an unloadable checkpoint rather than as a schema error. A run with
#: ``lambda_perp == lambda_rigid == 0`` therefore writes byte-identical rows to today's.
#: ``dihedral_loss`` joined them for the KINASE arm (K-15) on exactly the same terms: it is the
#: unweighted ``mean_n d_K`` of :class:`sae_bioemu.kinase_dfg.loss.KinaseSteeringLoss`, gated on
#: that objective's own ``lambda_dihedral``. The set is named for the shape rather than for the
#: cryptic project -- a column a PROJECT's objective declares and a weight in that objective's
#: own identity gates.
#:
#: **IT IS NO LONGER A LITERAL, AND THAT IS THE POINT.** These two names used to be a frozenset
#: and a tuple written out in this file, which is a member of the SHARED ``training_only`` group.
#: So a project adding ONE journal column re-keyed every other property's checkpoints for a
#: column none of them can write -- the D-43 defect the property registry exists to end,
#: surviving in the one table nobody had moved. Each property now declares its own columns and
#: their gates in its :class:`~sae_bioemu.steering.property_registry.PropertyRegistration`, and
#: this module composes them. A fifth property's column costs a catalog line and no shared edit.
#:
#: The declaration is import-light by construction -- it is in the registration rather than the
#: descriptor -- so reading it here does not pull the cryptic pocket geometry or OpenFold into a
#: radius-of-gyration run's import graph. See ``PropertyRegistration.journal_columns`` for why
#: this cannot be fail-open.
#:
#: Each optional column and the objective-identity weight that gates it, in the fixed order the
#: journal validator reports failures in. The weights themselves are still read from the running
#: objective (``objective_config_extra()`` first, then the shared attribute), so nothing here
#: depends on which class defines them (D-43).
_OFF_PATH_COLUMN_WEIGHTS = journal_column_weights()
_OFF_PATH_RECORD_KEYS = frozenset(name for name, _ in _OFF_PATH_COLUMN_WEIGHTS)

#: The weight names that gate columns belonging to DIFFERENT properties, and the ONLY ones for
#: which :func:`_validate_training_record_fields` narrows the union check to the running property.
#: Empty before D-87 made ``lambda_m`` co-owned, which is why every other column's pairing is
#: byte-identical to the one it has always had.
_CONTESTED_JOURNAL_WEIGHTS = contested_journal_weights()

#: The pocket-local geometry MONITOR columns (PLAN.md D-35b), gated on the run being a
#: cryptic-pocket arm rather than on a loss weight -- which is the one way they differ from
#: :data:`_OFF_PATH_RECORD_KEYS`, and it is deliberate: a monitor has no weight, because it is
#: not in the objective at all. Everything else about the gating is the same, for the same
#: reason: ``_records`` compares a persisted row's key set with frozenset equality, so an
#: unconditional column would make every radius-of-gyration journal on disk fail to parse and
#: the failure would surface as an unloadable checkpoint rather than as a schema error.
#:
#: **They select nothing, exclude nothing and change no verdict.** ``_validation_aggregate``
#: does not read them, ``_validation_key`` does not read them, and ``_validate_gates`` does not
#: read them. What they buy is latency: the D-35 structure gates otherwise run only at held-out
#: evaluation, so a fold coming apart under the adapter is invisible for the whole training run
#: plus evaluation generation -- roughly 12-15 GPU-h -- and first appears in the final report.
#: Turning any of them into a selection or exclusion rule needs its own decision record.
#:
#: Both members of each pair are journalled rather than their difference. The D-35b verdict is
#: the paired excess, which stays derivable as a difference of two integers, while a lone excess
#: column would drop the control's own base rate -- the number ``structure_gates`` insists a
#: count is read against -- and would read as the gate verdict these columns are not.
_MONITOR_RECORD_KEYS = frozenset(
    {
        "gate_scored_out_of_window_bonds",
        "gate_control_out_of_window_bonds",
        "gate_scored_clashing_pairs",
        "gate_control_clashing_pairs",
    }
)

#: The SHARED objective's remaining weighted terms and the weight that gates each, in the fixed
#: order the journal writer and validator report them in.
#:
#: ``total = lambda_y*target + lambda_c*contrastive + lambda_r*residual + lambda_g*geometry +
#: lambda_l*local_structure``. Two of the five had persisted columns (``target_loss`` /
#: ``target_loss_sigma_squared`` and ``residual_loss``) and three did not, so ``total_loss``
#: could not be decomposed by anything reading the journal. That was tolerable while
#: ``SteeringLoss`` hard-enforced ``lambda_g == 0`` on the relative path and ``lambda_c`` was 1;
#: it stopped being tolerable when the guard was lifted and ``lambda_c`` reached 700x
#: ``lambda_y``, which makes the dominant term the invisible one.
#:
#: Gated exactly as :data:`_OFF_PATH_RECORD_KEYS` is, and for exactly the same reason rather
#: than for symmetry: ``_records`` matches a persisted row's key set against a fixed base set
#: plus a fixed set of admitted extras, so an unconditional column would make every journal
#: written before these columns existed fail to parse and the failure would surface as an
#: unloadable checkpoint rather than as a schema error. A run with all three weights zero
#: writes byte-identical rows to today's.
#:
#: ``geometry_loss`` is here AND in :data:`_ABSOLUTE_RECORD_KEYS`, which is not a duplication.
#: On the absolute path it is a MANDATORY historical column present whatever ``lambda_g`` is;
#: on the relative and two-state paths it is a new gated one. :func:`_loss_term_columns`
#: subtracts the base set, so no path is offered the same column under two gates, and
#: :func:`_validate_training_record_fields` applies the weight gate only where the base does
#: not already require the column.
_LOSS_TERM_COLUMN_WEIGHTS = (
    ("geometry_loss", "lambda_g"),
    ("contrastive_loss", "lambda_c"),
    ("local_structure_loss", "lambda_l"),
)
_LOSS_TERM_RECORD_KEYS = frozenset(name for name, _ in _LOSS_TERM_COLUMN_WEIGHTS)

#: The registry-declared UNPRICED MONITOR columns (D-86), a function of the catalog exactly as
#: :data:`_OFF_PATH_RECORD_KEYS` is. They differ from that set in having NO gating weight, and
#: from :data:`_MONITOR_RECORD_KEYS` in not being written out here: a property adding one costs a
#: ``PropertyRegistration.monitor_columns`` line and its own
#: ``SteerableProperty.training_row_columns`` hook, rather than an edit to a member of the shared
#: ``training_only`` group.
#:
#: **They are priced nowhere.** No value here reaches ``total``, the objective identity, a
#: selection metric, ``_validation_aggregate``, ``_validation_key`` or ``_validate_gates``. That is
#: the whole distinction from an off-path TERM: D-76 measured the trained arm's realised alignment
#: as ``|cos| = |achieved| / sqrt(achieved^2 + perp_loss)`` and could only do so because that arm
#: priced ``lambda_perp``; the direct-metric arm does not, so the quantity had to become a
#: measurement rather than acquire a weight it has no reason to carry.
_DECLARED_MONITOR_RECORD_KEYS = frozenset(monitor_column_names())

#: Every column that may be absent from a persisted row. The union is what :func:`_records`
#: allows as surplus over a base key set; membership of a base set still makes a column
#: mandatory on that path.
_OPTIONAL_RECORD_KEYS = (
    _OFF_PATH_RECORD_KEYS
    | _MONITOR_RECORD_KEYS
    | _LOSS_TERM_RECORD_KEYS
    | _DECLARED_MONITOR_RECORD_KEYS
)

#: The three base schemas, in the order :func:`_records` and the plot reader try them. The
#: relative set is a SUBSET of the two-state one, so the narrower candidate is tried first and
#: rejected on its surplus -- ``target_state`` is not an optional column -- before the two-state
#: candidate matches.
_BASE_RECORD_KEY_SETS = (_ABSOLUTE_RECORD_KEYS, _RELATIVE_RECORD_KEYS, _TWO_STATE_RECORD_KEYS)


def _loss_term_columns(objective: SteeringLoss, base: frozenset[str]) -> tuple[str, ...]:
    """Return the gated loss-term columns a run under ``objective`` journals on ``base``.

    A column is journalled when its weight is nonzero and the base schema does not already
    require it -- which is the one asymmetry in :data:`_LOSS_TERM_COLUMN_WEIGHTS`, and it is
    the absolute path's historical ``geometry_loss``.
    """
    return tuple(
        name
        for name, weight in _LOSS_TERM_COLUMN_WEIGHTS
        if name not in base and float(getattr(objective, weight)) > 0.0
    )


def training_record_keys(
    relative: bool,
    *,
    off_path: Sequence[str] = (),
    monitor: Sequence[str] = (),
    declared_monitor: Sequence[str] = (),
    loss_terms: Sequence[str] = (),
    two_state: bool = False,
) -> frozenset[str]:
    """Return the persisted training-record key set for one target construction.

    ``off_path`` names the off-path columns this run's objective switches on, as a subset of
    :data:`_OFF_PATH_RECORD_KEYS`; ``monitor`` names the pocket-local geometry columns a
    cryptic-pocket arm journals, as a subset of :data:`_MONITOR_RECORD_KEYS`;
    ``declared_monitor`` names the registry-declared UNPRICED monitors (D-86), as a subset of
    :data:`_DECLARED_MONITOR_RECORD_KEYS`; ``loss_terms`` names the shared objective's remaining
    weighted terms, as a subset of :data:`_LOSS_TERM_RECORD_KEYS`. All four default to empty, so
    every existing caller keeps reading exactly the schema it was written against.

    ``two_state`` selects the two-state schema (:data:`_TWO_STATE_RECORD_KEYS`) and requires
    ``relative``. It defaults to off for the same reason: every radius-of-gyration and Local-Q
    caller keeps its exact historical answer. It is named for the TARGET CONSTRUCTION rather
    than for a normalisation, which is what it always selected -- the two-state row's extra
    columns are the state and the per-draw origin, and those exist under D-64's normalisation
    and D-65's alike.

    They are separate arguments rather than one bag because they are gated by different things
    -- an off-path column by a weight the project's own objective declares, a monitor column by
    the run being a cryptic arm, a declared monitor by the property's own hook and NO weight at
    all, a loss-term column by one of the shared weights every objective identity records -- and
    folding them together would let a row present one gate's columns under the other's
    justification.
    """
    if two_state and not relative:
        raise ValueError(
            "the two-state schema is a relative-path construction: the prediction is a "
            "displacement from a matched unsteered draw, which the absolute path has no notion of"
        )
    if not relative:
        keys = _ABSOLUTE_RECORD_KEYS
    else:
        keys = _TWO_STATE_RECORD_KEYS if two_state else _RELATIVE_RECORD_KEYS
    extra = frozenset(off_path)
    if not extra <= _OFF_PATH_RECORD_KEYS:
        raise ValueError(f"off_path must be a subset of {sorted(_OFF_PATH_RECORD_KEYS)}")
    watched = frozenset(monitor)
    if not watched <= _MONITOR_RECORD_KEYS:
        raise ValueError(f"monitor must be a subset of {sorted(_MONITOR_RECORD_KEYS)}")
    declared = frozenset(declared_monitor)
    if not declared <= _DECLARED_MONITOR_RECORD_KEYS:
        raise ValueError(
            f"declared_monitor must be a subset of {sorted(_DECLARED_MONITOR_RECORD_KEYS)}"
        )
    terms = frozenset(loss_terms)
    if not terms <= _LOSS_TERM_RECORD_KEYS:
        raise ValueError(f"loss_terms must be a subset of {sorted(_LOSS_TERM_RECORD_KEYS)}")
    return keys | extra | watched | declared | terms


def validate_ordinary_evaluation_identity(
    checkpoint: AdapterCheckpoint,
    *,
    schedule: SteeringSchedule,
    adapter: nn.Module,
    objective: SteeringLoss,
    config: OrdinarySteeringConfig,
    batch_plan: SteeringBatchPlan | None = None,
) -> None:
    """Bind a loaded training checkpoint to the current evaluation runtime.

    ``batch_plan`` must be the plan the evaluation run resolved, so a checkpoint trained
    with batching cannot be evaluated by a configuration that resolves none, or a
    different one. Passing ``None`` asserts that the checkpoint was trained unbatched.

    **Code identity (D14).** Everything compared here is something that genuinely determines
    comparability: the adapter and objective configuration, the property callback identity,
    the source checkpoints, the manifest/schedule binding, the target construction, the
    training seed, and ``generation_callback_identity`` -- which on the BioEmu path carries
    the caller's *training* code fingerprint, covering the generation-shared sources
    (sampler, hook, adapter, property surrogate, target formula, runtime assets,
    environment). So evaluating a checkpoint with a changed sampler still fails closed:
    evaluation regenerates structures with that machinery, and structures from two
    generators are not comparable.

    What is deliberately **not** compared is the evaluation code fingerprint. A checkpoint is
    a product of training; scoring code is a consumer of it, and requiring it to match would
    make a metric fix or a plot label invalidate a trained adapter. The evaluation fingerprint
    is recorded by the caller instead -- in the run report, the evaluation identity, and the
    report manifest -- so a reader can always tell which scoring code produced a number.
    """
    if not isinstance(checkpoint, AdapterCheckpoint):
        raise TypeError("checkpoint must be an AdapterCheckpoint")
    # Same check the training path makes, for the same reason: the identity below records the
    # config's construction, so an evaluation objective declaring a different normalisation
    # would be compared against a description of itself that it does not satisfy.
    _require_matched_loss_normalization(objective, config)
    accumulation = resolve_gradient_accumulation(
        schedule,
        batch_plan,
        config.gradient_accumulation_units,
        # The *configured* world size, not the evaluating process's. Evaluation runs on one
        # GPU; the checkpoint it is binding to was produced by an optimizer step spanning
        # ``W x K`` micro-batches, and that is what the identity records.
        world_size=config.distributed_world_size,
    )
    expected = {
        "generation_batching": None if batch_plan is None else batch_plan.identity(),
        # Rebuilt rather than read back: an evaluation configured with different
        # accumulation describes a different optimization from the one that produced this
        # checkpoint, and the conditional keys make the two key sets differ as well.
        **_optimizer_step_identity(config, accumulation, schedule),
        # Rebuilt for the same reason: evaluation regenerates both arms, so evaluating an
        # fp32-trained checkpoint under bf16 would score it against structures its own
        # generator never produced.
        **_generation_precision_identity(config),
        "workflow": "ordinary",
        "property_name": config.property_name,
        "model_family": config.model_family.value,
        "intervention_site": config.intervention_site,
        "generation_callback_identity": config.generation_callback_identity,
        "property_callback_identity": config.property_callback_identity,
        "manifest_fingerprint": config.manifest_fingerprint,
        "source_checkpoints": dict(config.source_checkpoints),
        "schedule_fingerprint": _schedule_fingerprint(schedule),
        "schedule_inventory": _schedule_inventory(schedule),
        **_target_identity(config),
        "training_seed": config.training_seed,
        "adapter": _module_config(adapter),
        "objective": _objective_config(objective),
        # D-23's training subset. Compared here as well as written at training time, because
        # the schedule inventory the evaluation trusts is the FILTERED one: without this key a
        # checkpoint trained on one exclusion list could be scored under a configuration
        # naming another, and the only visible difference would be a path in a report.
        **_training_subset_identity(config),
        **_cryptic_arm_identity(config),
        **_property_arm_identity(config),
        "early_stopping_patience": config.early_stopping_patience,
        **(
            {}
            if config.gradient_clip_norm is None
            else {"gradient_clip_norm": config.gradient_clip_norm}
        ),
        **_partial_pair_coverage_identity(config),
        # Absence means "selected on a validation role", so a checkpoint that selected nothing
        # must not be scored by a configuration that believes it did -- and the presence check
        # below is what makes that true in both directions.
        **_model_selection_identity(config),
    }
    actual = checkpoint.adapter_config
    differing = [name for name, value in expected.items() if actual.get(name) != value]
    if differing:
        # Name the keys. A bare "does not match" forces the reader to rebuild the identity by
        # hand to learn which of twenty fields moved, and the answer is usually one of them.
        detail = ", ".join(differing)
        raise ValueError(
            f"ordinary checkpoint does not match the current evaluation identity: {detail}"
        )
    # Presence, not just value: the optimizer-step keys are conditional, so a checkpoint
    # that accumulated or pinned a diagnostic seed would otherwise slip past a
    # configuration that does neither, which is the fail-open direction.
    if any(
        (name in actual) is not (name in expected)
        for name in (
            "gradient_accumulation",
            "property_ensemble",
            "generation_precision",
            "diagnostic_fixed_generation_seed",
            # Same fail-open direction: ``cryptic_arm`` is conditional so a radius-of-gyration
            # checkpoint keeps its historical key set, which means a value comparison alone
            # would let a cryptic checkpoint be scored by a configuration that resolved no arm.
            "cryptic_arm",
            # And the same again: a checkpoint trained on an admittedly partial paired term must
            # not be scored as though it had been trained on the full one.
            "allow_partial_pair_coverage",
            # ...and once more: absence means "trained on the whole train role", so a value
            # comparison alone would let a subset-trained checkpoint be scored as an
            # unfiltered arm -- which is precisely the claim D-66 exists to test.
            "training_subset",
            # And once more again: absence means the run selected its checkpoint on a validation
            # role. Without the presence check, a no-validation checkpoint would bind happily to
            # a configuration declaring a validation split, and the report would describe a
            # selected adapter that was never selected.
            "model_selection",
        )
    ):
        raise ValueError("ordinary checkpoint does not match the current evaluation identity")


def run_ordinary_training(
    *,
    schedule: SteeringSchedule,
    generate: GenerateCallback,
    baseline: BaselineCallback | None = None,
    baseline_identity: str | None = None,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    frozen_modules: Sequence[nn.Module],
    objective: SteeringLoss,
    checkpoint_path: str | Path,
    config: OrdinarySteeringConfig,
    resume: AdapterCheckpoint | str | Path | None = None,
    stop_after_units: int | None = None,
    progress: ProgressReporter | None = None,
    generate_batch: GenerateBatchCallback | None = None,
    batch_plan: SteeringBatchPlan | None = None,
    distributed: DistributedContext | None = None,
) -> OrdinaryTrainingResult:
    """Train one adapter with exact transactional resume and validation selection.

    ``progress`` is optional operator instrumentation only. It never influences the
    schedule, the checkpoint contents that are fingerprinted, or any scientific value.

    ``baseline``/``baseline_identity`` belong to the absolute-target path alone. A
    relative-sequence run (D3) builds every target from its per-sequence reference
    table, so it generates **no** adapter-off structures during training or
    validation and must not be given a baseline callback; the checkpoint identity is
    bound to the reference table's content fingerprint instead.

    ``batch_plan``/``generate_batch`` enable batched generation (D8). The **batch** is
    then the atomic unit: one optimizer step covers every member, and the cursor
    advance, every member's training record, and the in-flight clear are one committed
    checkpoint write, so a resume can neither replay a partially committed batch nor
    skip a member. The plan is bound into the checkpoint identity, because an optimizer
    step over ``B`` units is a different optimization from ``B`` steps over one unit.
    Without a plan every batch holds exactly one unit and the path is unchanged.

    ``config.gradient_accumulation_units`` composes on top of that: ``K`` consecutive
    generation batches are generated, each backward-ed into the *same* gradient buffer,
    and one optimizer step is taken for the group, so one step covers
    ``batch size x K`` scheduled units. Each micro-step's loss is scaled by its share of
    the group's units (``n_i / N``) before ``backward()``, because the property loss is a
    mean over its own micro-batch; the accumulated gradient is then exactly the gradient
    one batch of all ``N`` units would have produced, and no sequence is reweighted by
    which micro-batch it landed in.

    **The accumulation group, not the generation batch, becomes the atomic resume unit.**
    ``zero_grad`` runs once per group and ``optimizer.step`` once per group, so a cursor
    inside a group describes a state that never existed: half a gradient with no step
    taken. A persisted cursor that is not a group boundary therefore fails closed, and a
    crash mid-group replays the whole group. Checkpointing mid-group was rejected: it
    would mean persisting the gradient buffer and the accumulated micro-batch losses
    inside the checkpoint, which turns a hardware-dependent float into part of
    ``checkpoint_file_sha256``, for no scientific gain -- the group is minutes of
    generation, not hours.

    ``config.property_ensemble_samples`` (``M``, D16) composes on top of both, and is a
    different lever again: each scheduled unit generates ``M`` structures from its own
    derived seeds, and the property loss is applied **once** to their differentiable mean
    rather than once per structure. One optimizer step therefore covers
    ``M x accumulation units x batch size`` generated structures, of which
    ``accumulation units x batch size`` are scheduled units. The two averaging mechanisms
    do not double count: the ensemble collapses ``M`` structures into one observation per
    unit *before* the loss, and accumulation's ``n_i / N`` weighting counts units, so with
    ``M`` constant across the run every unit still carries weight ``1 / N`` in its group's
    mean regardless of ``M``. Validation uses the same estimator, because selection must
    not disagree with the objective (D4b).

    ``config.diagnostic_fixed_generation_seed`` is a DIAGNOSTIC that makes every *train*
    unit of a sequence generate from that sequence's first scheduled train seed, removing
    the structure-draw noise. Validation and evaluation keep their own per-unit seeds. It
    is recorded in the checkpoint identity, so its checkpoints can never be confused with
    a real run's; the caller is responsible for refusing it outside an exploratory run and
    for stamping its artifact names.

    ``distributed`` (with ``config.distributed_world_size``) runs the same loop across
    ``W`` single-node processes. It is the third lever on the same quantity: one optimizer
    step covers ``batch size x W x K`` scheduled units, with ``W`` multiplying rather than
    dividing the accumulation group, so the effective batch grows without spending
    optimizer steps. Five properties are preserved exactly, and are what the distributed
    path is allowed to cost:

    * **Scheduling and seeding are untouched.** The schedule, the batch plan, and every
      unit's ``seed``/``draw_seed`` are built before any process group exists and are
      pure functions of the split and the schedule seed. ``W`` decides only which rank
      generates which micro-batch, never which units exist or what they draw.
    * **The accumulation group stays the atomic resume unit.** It is the boundary
      ``offsets()`` already defines, and it now maps onto exactly one cross-rank
      ``all_reduce``: every rank enters a group together and leaves it together, so no
      rank can advance past a group the others have not finished.
    * **Only rank 0 writes.** Checkpoints, the best-checkpoint copy, the timing sidecar
      and the gradient journal are written by the primary rank alone, atomically as
      before, with a barrier around every write so no rank runs ahead of a checkpoint.
    * **``maximum_unit_attempts`` is charged once per group.** A failure on any rank
      fails the group on all of them, the failing rank's record is broadcast so every
      journal is identical, and the attempt counter advances by one however many ranks
      raised.
    * **Length bucketing survives.** Whole micro-batches move between ranks, never parts
      of one, so each rank's generation batch is exactly as length-coherent as the
      planner composed it -- which matters more here than elsewhere, because
      ``keep_draw_groups_whole`` plans a micro-batch so the paired dose term's two
      members share a live graph, and splitting one across ranks would silently delete
      those pairs.

    **Per-epoch validation is sharded across the same ranks**, one wave at a time: a wave is
    the next ``min(W, remaining)`` validation batches in cursor order, rank ``r`` draws slot
    ``r``, and the journal is committed once per wave (:func:`_validation_wave`). It takes no
    optimizer step, so it costs none of the five properties above -- the batch stays whole,
    the committed cursor stays a planned batch boundary, only rank 0 writes, and one attempt
    is charged per wave attempt however many ranks raised. What it adds is a replica guard:
    the round's rows now come from every rank, so every rank's adapter state is fingerprinted
    and compared before the first draw.

    ``None`` is the single-process path, byte-identical to a run made before this existed.
    """
    if progress is not None and not isinstance(progress, ProgressReporter):
        raise TypeError("progress must be a ProgressReporter")
    context = distributed if distributed is not None else single_process_context()
    if not isinstance(context, DistributedContext):
        raise TypeError("distributed must be a DistributedContext")
    if context.world_size != config.distributed_world_size:
        raise ValueError(
            f"the process group has world size {context.world_size} but the configuration "
            f"declares distributed_world_size {config.distributed_world_size}; the declared "
            "value multiplies the accumulation group and is bound into the checkpoint "
            "identity, so a mismatch would write a checkpoint claiming an effective batch "
            "this run never took"
        )
    primary = context.is_primary
    frozen = tuple(frozen_modules)
    _validate_inputs(
        schedule,
        baseline,
        baseline_identity,
        generate,
        adapter,
        optimizer,
        frozen,
        objective,
        config,
        generate_batch,
        batch_plan,
    )
    validation_sizes = (
        (1,) * len(schedule.validation) if batch_plan is None else batch_plan.validation_batch_sizes
    )
    validation_offsets = batch_offsets(validation_sizes)
    # The declaration and the role are bound here, before anything generates: a run that selects
    # on nothing must carry no validation units, and one that selects must carry them.
    _validate_schedule(schedule, config.dose_levels, expect_validation=config.selects_on_validation)
    # One optimizer step spans one accumulation group, so the group -- not the generation
    # batch -- is what the cursor may sit on and what the identity has to record.
    accumulation = resolve_gradient_accumulation(
        schedule,
        batch_plan,
        config.gradient_accumulation_units,
        world_size=config.distributed_world_size,
    )
    train_offsets = accumulation.offsets()
    train_groups = accumulation.groups_by_offset()
    require_pair_coverage(
        objective, config, paired_dose_coverage(schedule, train_offsets, train_groups)
    )
    _validate_optimizer_parameters(optimizer, adapter)
    freeze_for_steering(*frozen, adapter)
    if stop_after_units is not None and (
        not isinstance(stop_after_units, int)
        or isinstance(stop_after_units, bool)
        or stop_after_units < 0
    ):
        raise ValueError("stop_after_units must be a nonnegative integer")

    destination = Path(checkpoint_path).resolve()
    timings = TimingSidecar.for_checkpoint(destination)
    # Operator diagnostics, in their own sidecar for the same reason wall clock is: a
    # gradient norm is a hardware- and batch-composition-dependent float, so one inside the
    # checkpoint would make two scientifically identical runs disagree on
    # ``checkpoint_file_sha256``. Nothing below may fail because of it.
    gradients = GradientProbe.for_checkpoint(destination)
    if config.is_relative:
        baseline_state = _NO_BASELINES
    elif primary:
        # Generated and published by the primary rank alone, then broadcast. Baselines are
        # adapter-off structures, so every rank would otherwise generate the identical
        # values and race one another for the same atomic file.
        baseline_state = _load_or_create_baselines(
            schedule, baseline, baseline_identity, config, destination, timings
        )
        baseline_state = broadcast_from(baseline_state, 0, context)
    else:
        baseline_state = broadcast_from(None, 0, context)
    identity = _adapter_config(
        schedule,
        baseline_identity,
        baseline_state,
        adapter,
        optimizer,
        objective,
        config,
        batch_plan,
        accumulation,
    )
    generation_seeds = _fixed_generation_seeds(schedule, config)
    ensemble_seeds = _property_ensemble_seeds(schedule, config)
    # The primary rank resolves or creates the checkpoint; the others read the file it
    # leaves behind. Ordered this way rather than letting every rank try
    # ``_resume_checkpoint`` first, because a non-primary rank that happened to find a
    # freshly created file would skip the barrier the primary is already waiting at.
    checkpoint = None
    extension: ScheduleExtension | None = None
    if primary:
        checkpoint, extension = _resume_checkpoint(resume, destination, config, identity)
        if checkpoint is None:
            journal = _empty_journal()
            journal["failures"] = [asdict(failure) for failure in baseline_state.failures]
            journal["failure_keys"] = [
                f"baseline:{failure.sequence_id}" for failure in baseline_state.failures
            ]
            checkpoint = _make_checkpoint(
                config,
                identity,
                adapter,
                optimizer,
                epoch=0,
                cursor=0,
                journal=journal,
            )
            save_adapter_checkpoint(destination, checkpoint)
    barrier(context)
    if not primary:
        # Loaded through the same identity gate the primary used, so a rank joining a
        # checkpoint it does not match fails closed instead of training against it.
        checkpoint, extension = _resume_checkpoint(resume, destination, config, identity)
        if checkpoint is None:  # pragma: no cover - the primary always publishes one.
            raise RuntimeError("the primary rank did not publish a training checkpoint")
    if checkpoint.unit_cursor > len(schedule.train):
        raise ValueError("checkpoint unit cursor exceeds schedule")

    adapter.load_state_dict(checkpoint.adapter_state)
    optimizer.load_state_dict(checkpoint.optimizer_state)
    _validate_optimizer_parameters(optimizer, adapter)
    _validate_optimizer_identity(optimizer, adapter, identity["optimizer"])
    freeze_for_steering(*frozen, adapter)
    # One reference for the whole run. The frozen stack is not allowed to change at all,
    # so a per-unit reference proved nothing a run-scoped one does not, and it cost a full
    # host copy of the generator per unit plus another inside every check.
    guard = _frozen_guard(frozen)
    journal = _journal(checkpoint.validation_record)
    if extension is not None:
        # REPUBLISH under the longer plan, before a single further unit is walked. The adopted
        # checkpoint still carries the SHORT plan's identity -- that is what it was validated
        # against -- and its journal still says the run terminated. Both are rewritten here, in
        # one atomic write, so that the artifact on disk records the continuation and every
        # later gate (progress validation, completion attestation, the evaluation identity) sees
        # one self-consistent checkpoint rather than a short plan wearing a long one's cursor.
        #
        # Every rank computes the same reopened journal, because it is a pure function of the
        # journal it loaded and the extension; only the primary writes, as everywhere else.
        #
        # The best-checkpoint SIDECAR is republished too, and must be, whenever the identity
        # moved. The journal's ``best_checkpoint_path`` keeps pointing at a file written under the
        # ORIGIN's identity, and both ``_validate_checkpoint_progress`` and
        # ``_restore_best_checkpoint`` load it with THIS run's expectation -- so without the
        # rewrite an extension of any selecting run refuses on the line after this one, and one
        # that got past it would die at completion hours later. Only the identity is rewritten:
        # the adapter and optimizer bytes are untouched, so
        # ``best_model_optimizer_fingerprint`` -- which digests exactly those two -- still
        # matches, and the file is disclosed in the resume record.
        best_path = _resumed_best_checkpoint_path(journal, extension, destination)
        extension = replace(
            extension,
            republished_best_checkpoint=(
                None
                if best_path is None
                or extension.from_manifest_fingerprint == extension.to_manifest_fingerprint
                else best_path.name
            ),
        )
        journal = _reopen_journal_for_extension(
            journal,
            extension,
            checkpoint_path=destination,
            best_model_optimizer_fingerprint=_model_optimizer_fingerprint(checkpoint),
        )
        checkpoint = _make_checkpoint(
            config,
            identity,
            adapter,
            optimizer,
            epoch=checkpoint.epoch,
            cursor=checkpoint.unit_cursor,
            journal=journal,
        )
        if primary:
            if best_path is not None and extension.republished_best_checkpoint is not None:
                _republish_best_checkpoint(best_path, config, identity)
            save_adapter_checkpoint(destination, checkpoint)
            detail = (
                f"RESUMED the training schedule from {extension.resumed_from}: "
                f"{extension.from_maximum_epochs} to {extension.to_maximum_epochs} epochs "
                f"({extension.from_train_units} to {extension.to_train_units} train units), "
                f"continuing at epoch {extension.resumed_at_epoch}, unit "
                f"{extension.resumed_at_unit_cursor}. Origin manifest "
                f"{extension.from_manifest_fingerprint} (terminal state "
                f"{extension.from_terminal_reason!r}), this run "
                f"{extension.to_manifest_fingerprint}"
            )
            _LOGGER.warning("%s", detail)
            for crossing in extension.warnings():
                _LOGGER.warning("%s", crossing)
            if extension.republished_best_checkpoint is not None:
                _LOGGER.warning(
                    "rewrote the identity of %s so this run's best-checkpoint pointer stays "
                    "loadable; its adapter and optimizer bytes are unchanged",
                    extension.republished_best_checkpoint,
                )
            if progress is not None:
                progress.announce(detail)
                for crossing in extension.warnings():
                    progress.announce(crossing)
        barrier(context)
    _validate_checkpoint_progress(
        checkpoint,
        destination,
        schedule,
        journal,
        config,
        baseline_state.records,
        train_offsets,
        validation_offsets,
        objective,
    )
    failures = _failures(journal["failures"])
    if not baseline_state.complete:
        return _result(
            checkpoint,
            destination,
            baseline_state.records,
            failures,
            journal,
            complete=False,
        )

    cursor = checkpoint.unit_cursor
    committed = 0
    boundaries = _cycle_boundaries(schedule.train)

    def commit_skip(
        current: dict[str, Any],
        attempt_key: str,
        at: int,
        size: int,
        members: Sequence[SteeringUnit],
    ) -> tuple[dict[str, Any], AdapterCheckpoint] | None:
        """Step over one exhausted GENERATION group, or return ``None`` to end the run.

        The advance and the record are ONE atomic checkpoint write, exactly as a committed group
        is: a resume can therefore neither replay a skipped group nor read a cursor that moved
        without saying why. No adapter or optimizer state moves -- the caller has already restored
        both -- so the write is a journal write with a new cursor on it.

        Every rank computes the identical verdict, because ``_generation_skip`` is a pure function
        of the journal (which the failing-rank broadcast has already made identical everywhere) and
        the configuration. Only the primary writes, as everywhere else in this loop.
        """
        skip = _generation_skip(current, attempt_key, at, size, members, config)
        if skip is None:
            return None
        candidate = copy.deepcopy(current)
        candidate.setdefault(_SKIPPED_UNITS_KEY, []).append(skip)
        candidate["in_flight"] = None
        skipped_checkpoint = _make_checkpoint(
            config,
            identity,
            adapter,
            optimizer,
            epoch=len(current["history"]),
            cursor=at + size,
            journal=candidate,
        )
        if primary:
            save_adapter_checkpoint(destination, skipped_checkpoint)
            taken, lost = len(_skipped_units(candidate)), _skipped_unit_count(candidate)
            detail = (
                f"SKIPPED train unit(s) {at}..{at + size - 1} after {skip['attempts']} generation "
                f"attempt(s); the group's first member is {skip['sequence_id']} seed "
                f"{skip['seed']} and the batch raised "
                f"{skip['exception_class']}: {skip['message']}. "
                f"{taken} of at most {skip['budget']} tolerated generation failures so far, "
                f"costing {lost} scheduled train unit(s) in total -- this one cost {size}, "
                "because the accumulation group is the atomic resume unit"
            )
            _LOGGER.warning("%s", detail)
            if progress is not None:
                progress.record_skipped_units(size)
                progress.announce(detail)
        return candidate, skipped_checkpoint

    while True:
        # Zero on a no-validation run, so the block below is not entered AT ALL -- on any rank.
        # That is deliberately stronger than entering it with an empty unit list: the body is
        # now full of collectives -- the replica guard, then one verdict and one gather per
        # wave -- so a rank that skipped it while another entered would deadlock, and one that
        # entered with nothing to do would still append a synthetic round to the history.
        due = _validation_rounds_due(boundaries, cursor, config)
        while len(journal["history"]) < due and not journal["early_stopped"]:
            # Validation is SHARDED across the ranks, one wave at a time: a wave is the next
            # min(W, remaining) validation batches in cursor order, rank r draws slot r, and
            # the journal is committed once per wave.
            #
            # This comment used to ARGUE FOR running the round on the primary alone, on the
            # grounds that "sharding it would mean a second, differently shaped resume
            # contract for a phase that takes no optimizer step". That objection is exactly
            # what the wave design answers, and it is recorded here as ANSWERED rather than
            # deleted, because a future reader who rediscovers it would otherwise restore the
            # gate. There is no second contract: a wave commits WHOLE BATCHES, so
            # ``active_validation`` keeps its ``{round_index, cursor, rows}`` schema, every
            # committed cursor is still a planned boundary of the same ``validation_offsets``
            # map the per-batch loop used, and the partition is recomputed from the RESUMED
            # cursor. The existing resume validators read what they always read.
            #
            # What the gate cost: the round is 79 batches of 2 units, 158 observations,
            # measured 16.9-17.2 min once per epoch -- ~11 h over 40 epochs, about 32% of
            # wall clock -- with the non-primary card idle holding ~42 GB throughout.
            #
            # The primary's journal is still the authoritative one. Every rank restores the
            # adapter and optimizer it borrowed, so the only state that moves is the journal
            # -- the validation history, the best-checkpoint selection, and the early-stopping
            # verdict, all of which every rank has to agree on -- and the broadcast below is
            # what keeps a non-primary rank's float state out of the committed record.
            checkpoint, validation_complete = _run_validation_checkpoint(
                schedule.validation,
                baseline_state.records,
                generate,
                generate_batch,
                validation_offsets,
                ensemble_seeds,
                adapter,
                optimizer,
                guard,
                config,
                identity,
                destination,
                cursor,
                journal,
                timings,
                context,
                progress,
                # A validation draw computes no loss, so the objective reaches here for ONE
                # reason: some of the diagnostic columns a property journals per held-out draw
                # are gated on a weight that objective prices.
                objective,
            )
            if context.world_size > 1:
                barrier(context)
                # The journal, not the checkpoint. Validation borrows the adapter and the
                # optimizer and restores both on every path, so no rank's parameters moved
                # and there is nothing to synchronise but the journal -- which also keeps
                # a pickled CUDA optimizer state off the wire. Broadcasting rather than
                # re-reading the file is deliberate: the exhausted-attempt path returns a
                # checkpoint it never published, so the file and the returned object differ.
                published, validation_complete = broadcast_from(
                    (_journal(checkpoint.validation_record), validation_complete)
                    if primary
                    else None,
                    0,
                    context,
                )
                if not primary:
                    checkpoint = _make_checkpoint(
                        config,
                        identity,
                        adapter,
                        optimizer,
                        # ``_run_validation_checkpoint`` stamps ``round_index + 1`` once a
                        # round is appended and ``round_index`` when it returns short, and
                        # both equal the history length it leaves behind.
                        epoch=len(published["history"]),
                        cursor=cursor,
                        journal=published,
                    )
            journal = _journal(checkpoint.validation_record)
            failures = _failures(journal["failures"])
            if not validation_complete:
                return _result(
                    checkpoint,
                    destination,
                    baseline_state.records,
                    failures,
                    journal,
                    complete=False,
                )
        if journal["early_stopped"] or cursor == len(schedule.train):
            break
        if stop_after_units is not None and committed >= stop_after_units:
            return _result(
                checkpoint,
                destination,
                baseline_state.records,
                failures,
                journal,
                complete=False,
            )

        size = _batch_size_at(train_offsets, cursor, "training")
        micro_sizes = train_groups[cursor]
        members = schedule.train[cursor : cursor + size]
        targets = tuple(_unit_target(unit, config, baseline_state.records) for unit in members)
        conditioned_targets = tuple(
            _conditioning(unit, target, config)
            for unit, target in zip(members, targets, strict=True)
        )
        attempt_key = f"train:{cursor}"
        if _attempts(journal, attempt_key) >= config.maximum_unit_attempts:
            # Reached on RESUME into a run whose previous process exhausted this group's attempts.
            # Every attempt this group is entitled to has already been spent, and each one after the
            # first drew a DIFFERENT derived seed, so the failure survived resampling and is not one
            # bad draw. Either the budget lets the run step over it here or nothing ever will.
            stepped = commit_skip(journal, attempt_key, cursor, size, members)
            if stepped is None:
                return _result(
                    checkpoint,
                    destination,
                    baseline_state.records,
                    failures,
                    journal,
                    complete=False,
                )
            journal, checkpoint = stepped
            cursor += size
            continue

        before_adapter = _cpu_state(adapter)
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        journal["in_flight"] = {"phase": "train", "key": attempt_key, "cursor": cursor}
        checkpoint = _make_checkpoint(
            config,
            identity,
            adapter,
            optimizer,
            epoch=len(journal["history"]),
            cursor=cursor,
            journal=journal,
        )
        if primary:
            save_adapter_checkpoint(destination, checkpoint)

        # Whole micro-batches, round robin. Each one keeps the length-coherent composition
        # the bucketing planner gave it -- and, under ``keep_draw_groups_whole``, the draw
        # group whose two members the paired dose term differences -- while every rank draws
        # from across the window rather than taking a contiguous run of its shortest or
        # longest batches.
        local_indices = rank_micro_batch_indices(len(micro_sizes), context.rank, context.world_size)
        micro_offsets = tuple(sum(micro_sizes[:index]) for index in range(len(micro_sizes)))
        phase = "generation"
        started = time.monotonic()
        local_error: Exception | None = None
        indexed_records: list[tuple[int, tuple[SteeringTrainingRecord, ...]]] = []
        # RESAMPLE, not retry. Attempt 0 draws this group's scheduled seeds, so a healthy run is
        # bit-identical to one made before this existed; every later attempt draws a derived
        # substitute. Retrying with the SAME seed cannot rescue a deterministic generation failure
        # -- it reproduces the exception and burns the attempt -- and every rank derives the same
        # index from the same journal, so the group stays one collective attempt.
        resample_index = _generation_attempts_spent(journal, attempt_key)
        try:
            # Once per optimizer step, never per micro-step: the whole point of
            # accumulation is that the K micro-batches share one gradient buffer. With W
            # ranks the buffer is per rank and the cross-rank sum below completes it.
            optimizer.zero_grad(set_to_none=True)
            for index in local_indices:
                micro_size = micro_sizes[index]
                offset = micro_offsets[index]
                micro = slice(offset, offset + micro_size)
                micro_members = members[micro]
                micro_targets = targets[micro]
                phase = "generation"
                generated_units = _generation_units(micro_members, generation_seeds)
                # D16: with an ensemble configured this generates M structures per unit and
                # returns their differentiable mean property, so the loss below is applied
                # once to the ensemble mean rather than once per structure.
                sample = _generate_property_ensemble(
                    generate,
                    generate_batch,
                    generated_units,
                    conditioned_targets[micro],
                    ensemble_seeds,
                    resample=(attempt_key, resample_index),
                )
                phase = "training"
                _validate_optimizer_identity(optimizer, adapter, identity["optimizer"])
                _assert_frozen_state(guard)
                # D4b: on the relative path the property error is measured in units of the
                # sequence's own sigma, so equal relative error contributes equally whether
                # the protein is rigid or floppy.
                target_tensor, property_scale = _loss_inputs(
                    sample, micro_members, micro_targets, config
                )
                parts = objective(
                    sample.surrogate_prediction,
                    target_tensor,
                    sample.residual_rms,
                    sample.residual_fraction,
                    geometry_loss=sample.geometry_loss,
                    property_scale=property_scale,
                    pair_group=_pair_groups(micro_members, sample.surrogate_prediction),
                    raw_residual_fraction=sample.raw_residual_fraction,
                    local_structure=sample.local_structure,
                    # Passed through unconditionally, not gated on the weight here: the
                    # objective refuses a weight whose input is missing, so a generator that
                    # skips an input and an objective that prices it fail loudly instead of
                    # quietly disagreeing. (The reverse -- an input at a zero weight -- is
                    # refused for the cryptic off-path tensors, which exist only to be priced,
                    # and permitted for ``local_structure``, which is generated regardless on
                    # a two-state arm because the unsteered draw is the progress coordinate's own anchor.) Which
                    # inputs those are is the objective's own declaration, so a shared
                    # objective is called with exactly the arguments it was called with before
                    # any project added one.
                    **_extra_sample_inputs(objective, sample),
                )
                # One step per accumulation group. Each micro-batch's loss is a mean over
                # its own members, so it is scaled by that micro-batch's share of the
                # group's units before it enters the shared gradient buffer: the
                # accumulated gradient is then exactly the gradient one batch of all N
                # units would have produced, whatever the micro-batch sizes were.
                #
                # ``size`` is the WHOLE group's unit count, not this rank's share. That is
                # what makes the cross-rank sum exact: the W ranks hold disjoint parts of
                # one ``1 / N`` mean, so adding them reconstructs the gradient a single
                # process accumulating all W x K micro-batches would have held.
                _accumulate(parts.total, micro_size, size)
                indexed_records.append(
                    (
                        index,
                        _batch_training_records(
                            micro_members,
                            micro_targets,
                            sample,
                            parts,
                            target_tensor,
                            property_scale,
                            objective,
                            config,
                        ),
                    )
                )
        except Exception as error:
            local_error = error
        # The group is the atomic unit, so its verdict is collective: a rank must not
        # commit a step the others abandoned, nor abandon one they committed.
        failing_rank = agreed_first_failing_rank(local_error is not None, context)
        batch_records: tuple[SteeringTrainingRecord, ...] = ()
        if failing_rank == context.world_size:
            batch_records = concatenated_records(gather_indexed(indexed_records, context))
            try:
                # One accumulation group, one all-reduce. After it every rank holds the
                # identical gradient, so the clip, the assertions, and the optimizer step
                # below all act on the same numbers and the ranks cannot drift apart.
                all_reduce_adapter_gradients(adapter, context)
                # On the ACCUMULATED gradient, not per micro-step: one micro-step's
                # gradient is legitimately allowed to be all zero, and the all-zero check
                # is about the step the optimizer is about to take.
                assert_adapter_only_gradients(*frozen, adapter)
                _assert_frozen_state(guard)
                _validate_optimizer_identity(optimizer, adapter, identity["optimizer"])
                # Diagnostic snapshot between the last backward() and step(), so the
                # recorded norm and the step-to-step cosine describe the ACCUMULATED
                # gradient -- the quantity the optimizer acts on. Measuring per micro-step
                # would report the very draw-to-draw noise accumulation exists to average
                # away. ``before_step`` swallows its own failures and returns None, so it
                # can never burn a unit attempt. When clipping is configured it happens
                # *before* the diagnostic reads the gradient, so the reported norm is the
                # one the optimizer acted on -- and after the all-reduce, so the clip is
                # applied once to the whole step rather than once per rank's share.
                if config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        adapter.parameters(), float(config.gradient_clip_norm)
                    )
                pending_gradient = gradients.before_step(adapter.parameters()) if primary else None
                optimizer.step()
                _validate_optimizer_identity(optimizer, adapter, identity["optimizer"])
                _assert_frozen_state(guard)
            except Exception as error:
                local_error = error
                phase = "training"
            failing_rank = agreed_first_failing_rank(local_error is not None, context)
        if failing_rank != context.world_size:
            adapter.load_state_dict(before_adapter)
            optimizer.load_state_dict(before_optimizer)
            _restore_frozen_state(guard)
            freeze_for_steering(*frozen, adapter)
            # One record, from the lowest failing rank, on every rank's journal. The
            # attempt counter therefore advances by exactly one per collective attempt
            # however many ranks raised, so ``maximum_unit_attempts`` keeps meaning the
            # number of times this group was tried rather than the number of ranks.
            failure_payload = broadcast_from(
                asdict(
                    _failure(
                        members[0],
                        targets[0],
                        phase,
                        local_error,
                        members,
                        label="accumulation group" if len(micro_sizes) > 1 else "batch",
                    )
                )
                if context.rank == failing_rank
                else None,
                failing_rank,
                context,
            )
            candidate_journal = copy.deepcopy(journal)
            candidate_journal["attempts"][attempt_key] = _attempts(journal, attempt_key) + 1
            candidate_journal["failures"].append(failure_payload)
            candidate_journal["failure_keys"].append(attempt_key)
            candidate_journal["in_flight"] = None
            checkpoint = _make_checkpoint(
                config,
                identity,
                adapter,
                optimizer,
                epoch=len(journal["history"]),
                cursor=cursor,
                journal=candidate_journal,
            )
            if primary:
                save_adapter_checkpoint(destination, checkpoint)
                _record_timing(timings, "train", attempt_key, members[0], started, "failed")
            journal = candidate_journal
            failures = _failures(journal["failures"])
            if _attempts(journal, attempt_key) >= config.maximum_unit_attempts:
                stepped = commit_skip(journal, attempt_key, cursor, size, members)
                if stepped is None:
                    return _result(
                        checkpoint,
                        destination,
                        baseline_state.records,
                        failures,
                        journal,
                        complete=False,
                    )
                journal, checkpoint = stepped
                cursor += size
            continue
        candidate_journal = copy.deepcopy(journal)
        candidate_journal["in_flight"] = None
        candidate_journal["training_records"].extend(_record(item) for item in batch_records)
        candidate_checkpoint = _make_checkpoint(
            config,
            identity,
            adapter,
            optimizer,
            epoch=len(journal["history"]),
            cursor=cursor + size,
            journal=candidate_journal,
        )
        unit_seconds = None
        if primary:
            save_adapter_checkpoint(destination, candidate_checkpoint)
            unit_seconds = _record_timing(
                timings, "train", attempt_key, members[0], started, "committed"
            )
        # No rank may read past a checkpoint that is not yet on disk: the next group
        # overwrites the adapter state this one committed.
        barrier(context)
        journal = candidate_journal
        checkpoint = candidate_checkpoint
        # Published after the scientific write it describes, exactly as the timing row is:
        # a crash between the two loses a diagnostic, never a committed optimizer step.
        if primary:
            gradients.record(
                pending_gradient,
                adapter.parameters(),
                key=attempt_key,
                epoch=len(journal["history"]),
                cursor=cursor,
                batch_size=size,
            )
        cursor += size
        committed += size
        if progress is not None and primary:
            progress.observe(
                phase="train",
                elapsed_seconds=unit_seconds,
                units=size,
                completed_units=cursor,
                epoch=len(journal["history"]),
                force=cursor == len(schedule.train),
            )

    if journal["early_stopped"]:
        journal["terminal_reason"] = "early_stopped"
        journal["not_run_after_early_stopping"] = [asdict(unit) for unit in schedule.train[cursor:]]
    elif cursor == len(schedule.train):
        journal["terminal_reason"] = "max_cycles_reached"
        journal["not_run_after_early_stopping"] = []
    complete = journal["terminal_reason"] is not None
    # A run that SELECTS must have something to have selected: no history, or a history with no
    # best round, is a run that walked its schedule and cannot say which adapter it is shipping,
    # and shipping one anyway would be an attestation about a choice never made.
    #
    # A run that selects on NOTHING has neither, by construction, and that is not a defect. It
    # reaches ``max_cycles_reached`` with an empty history and ships the adapter its last
    # optimizer step produced. Before this branch existed such a run trained in full and then
    # marked itself incomplete here, so it published no completion attestation and the whole
    # evaluation chain downstream of it never fired.
    if config.selects_on_validation and (
        not journal["history"] or journal["best_validation_index"] is None
    ):
        complete = False
    if complete:
        if config.selects_on_validation:
            checkpoint = _restore_best_checkpoint(
                checkpoint,
                destination,
                adapter,
                optimizer,
                config,
                identity,
                cursor,
                journal,
            )
            journal = _journal(checkpoint.validation_record)
            best = journal["history"][journal["best_validation_index"]]
            try:
                _validate_gates(best, config, adapter)
            except Exception as error:
                journal["validation_gate_failure"] = {
                    "phase": "validation_gate",
                    "exception_class": type(error).__name__,
                    "message": str(error) or repr(error),
                }
                complete = False
        # With no validation there is nothing to restore TO: ``_restore_best_checkpoint`` loads
        # the best-by-validation copy over the trained adapter, which is meaningless without a
        # validation round, and there is no gate to apply either. The adapter in hand is already
        # the one that ships, so ``best_validation_index``,
        # ``selected_best_model_optimizer_fingerprint`` and the whole best-checkpoint block stay
        # ``None``. Nothing synthetic is written in their place: a fabricated index or a
        # single-round history would make this run indistinguishable from a selected one at
        # every later read.
        journal["passed"] = complete
        journal["workflow_complete"] = complete
        checkpoint = _make_checkpoint(
            config,
            identity,
            adapter,
            optimizer,
            epoch=len(journal["history"]),
            cursor=cursor,
            journal=journal,
        )
        if primary:
            save_adapter_checkpoint(destination, checkpoint)
    # The run's last write, so every rank leaves with the published checkpoint on disk.
    barrier(context)

    return _result(
        checkpoint,
        destination,
        baseline_state.records,
        _failures(journal["failures"]),
        journal,
        complete=complete,
    )


def write_ordinary_completion(
    path: str | Path,
    result: OrdinaryTrainingResult,
    *,
    expected_schedule: SteeringSchedule,
) -> str:
    """Publish an attestation for the exact complete checkpoint file and state."""
    if not isinstance(result, OrdinaryTrainingResult) or not result.complete:
        raise ValueError("ordinary training is incomplete")
    if not result.validation_record.get("workflow_complete", False):
        raise ValueError("ordinary completion requires validation gates")
    checkpoint_path = result.checkpoint_path.resolve()
    loaded = load_adapter_checkpoint(
        checkpoint_path,
        expected_manifest_fingerprint=result.checkpoint.manifest_fingerprint,
        expected_adapter_config=result.checkpoint.adapter_config,
        expected_property_callback_identity=result.checkpoint.property_callback_identity,
        expected_source_checkpoints=result.checkpoint.source_checkpoints,
    )
    loaded_journal = validate_ordinary_completion_checkpoint(
        loaded,
        checkpoint_path,
        expected_schedule=expected_schedule,
    )
    expected_state = ordinary_checkpoint_state_fingerprint(result.checkpoint)
    actual_state = ordinary_checkpoint_state_fingerprint(loaded)
    if actual_state != expected_state:
        raise ValueError("checkpoint file does not match the completed training result fingerprint")
    payload = {
        "schema_version": 2,
        "workflow_kind": "ordinary_steering",
        "status": "complete",
        "manifest_fingerprint": loaded.manifest_fingerprint,
        "adapter_config": copy.deepcopy(dict(loaded.adapter_config)),
        "property_callback_identity": loaded.property_callback_identity,
        "source_checkpoints": dict(loaded.source_checkpoints),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_file_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_state_fingerprint": actual_state,
        "terminal_reason": loaded_journal["terminal_reason"],
        "not_run_after_early_stopping": copy.deepcopy(
            loaded_journal["not_run_after_early_stopping"]
        ),
    }
    payload["fingerprint"] = hashlib.sha256(_json_bytes(payload)).hexdigest()
    # Diagnostic only, and attached after signing: the checkpoint file holds no wall clock,
    # so ``checkpoint_file_sha256`` and this fingerprint are identical for two byte-identical
    # runs. ``require_ordinary_completion`` ignores this key entirely.
    recorded = load_checkpoint_timings(checkpoint_path)
    if recorded is not None:
        payload["timings"] = timing_summary(recorded)
    _atomic_json(Path(path), payload)
    return payload["fingerprint"]


def _run_validation_checkpoint(
    units: Sequence[SteeringUnit],
    baselines: Mapping[str, float],
    generate: GenerateCallback,
    generate_batch: GenerateBatchCallback | None,
    validation_offsets: Mapping[int, int],
    ensemble_seeds: Mapping[int, tuple[int, ...]] | None,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    guard: _FrozenGuard,
    config: OrdinarySteeringConfig,
    identity: Mapping[str, Any],
    checkpoint_path: Path,
    train_cursor: int,
    journal: dict[str, Any],
    timings: TimingSidecar,
    context: DistributedContext,
    progress: ProgressReporter | None = None,
    objective: SteeringLoss | None = None,
) -> tuple[AdapterCheckpoint, bool]:
    """Run one validation round, sharding its batches across the ranks wave by wave.

    A WAVE is the next ``min(world_size, remaining)`` validation batches in cursor order;
    rank ``r`` draws slot ``r``, and the journal is committed ONCE per wave. At
    ``world_size == 1`` a wave is one batch, so this degenerates to the per-batch loop it
    replaced -- same two checkpoint writes per batch, same journal bytes, same attempt keys.

    **Why the resume contract does not move.** A wave commits whole batches, so
    ``active_validation`` keeps its ``{round_index, cursor, rows}`` schema, every committed
    cursor is still a planned boundary of the same ``validation_offsets`` map, and the
    partition is recomputed from the RESUMED cursor rather than from the round's start.
    :func:`_validate_batch_boundary`, :func:`_validate_validation_rows` and the attempt-state
    check therefore read exactly what they read before sharding existed.

    **What it buys.** The round is 79 batches of 2 units, 158 observations, measured
    16.9-17.2 min once per epoch: ~11 h over 40 epochs, about 32% of wall clock, spent with
    the non-primary card idle holding ~42 GB. Generation still runs with grad ENABLED on
    every rank, for the reasons the draw below states at length.
    """
    primary = context.is_primary
    round_index = len(journal["history"])
    active = journal["active_validation"]
    if not active:
        active.update({"round_index": round_index, "cursor": 0, "rows": []})
    if active.get("round_index") != round_index:
        raise ValueError("checkpoint active validation round is inconsistent")
    validation_cursor = active["cursor"]
    checkpoint = _make_checkpoint(
        config,
        identity,
        adapter,
        optimizer,
        epoch=round_index,
        cursor=train_cursor,
        journal=journal,
    )
    # Before any rank draws: this round's rows now come from every rank, so every rank has to
    # be holding the same adapter to have drawn them from.
    _assert_validation_replica_agreement(adapter, context, round_index)

    def grant_retry(
        current: dict[str, Any],
        attempt_key: str,
        at: int,
        size: int,
        members: Sequence[SteeringUnit],
    ) -> tuple[dict[str, Any], AdapterCheckpoint] | None:
        """Grant one exhausted VALIDATION batch a fresh set of resampled draws, or ``None``.

        The validation twin of ``commit_skip`` in the training loop, and atomic on the same terms:
        clearing the attempt counter and recording why are ONE checkpoint write, so a resume can
        neither replay a grant nor find a cleared counter with nothing saying who cleared it. No
        adapter or optimizer state moves and no cursor advances -- the batch is about to be drawn
        again, at a seed it has not drawn before.

        **The verdict is COLLECTIVE BY DERIVATION, and it adds no collective of its own.**
        :func:`_validation_retry`, :func:`_generation_attempts_spent` and
        :func:`_tolerated_failure_count` are pure functions of ``current``, and the wave loop below
        leaves ``current`` byte-identical on every rank: committed rows arrive through
        ``gather_indexed`` in slot order, the single failure record arrives through
        ``broadcast_from``, and no rank's own wall-clock measurement ever enters the journal. So
        every rank reaches the same grant-or-stop answer from the same bytes and then executes the
        same collective sequence, while only the WRITE is the primary's -- exactly the division
        ``commit_skip`` makes. Re-deriving this verdict from a fresh broadcast would be a second
        copy of one decision, which is the disagreement the failure path below warns about.
        """
        retry = _validation_retry(
            current, attempt_key, train_cursor, round_index, at, size, members, config
        )
        if retry is None:
            return None
        candidate = copy.deepcopy(current)
        candidate.setdefault(_VALIDATION_RETRIES_KEY, []).append(retry)
        # CLEARED, not incremented. ``maximum_unit_attempts`` is compared against this counter and
        # the whole grant is another set of attempts; the monotone record of how many draws this
        # batch has really spent is ``_generation_attempts_spent``, which reads the failure journal
        # and is never cleared, so the next attempt's resample index still advances.
        candidate["attempts"].pop(attempt_key, None)
        candidate["in_flight"] = None
        granted_checkpoint = _make_checkpoint(
            config,
            identity,
            adapter,
            optimizer,
            epoch=round_index,
            cursor=train_cursor,
            journal=candidate,
        )
        if primary:
            save_adapter_checkpoint(checkpoint_path, granted_checkpoint)
            taken = _tolerated_failure_count(candidate)
            detail = (
                f"RESAMPLING validation draw(s) {at}..{at + size - 1} of round {round_index} "
                f"after {retry['attempts']} generation attempt(s); the batch's first member is "
                f"{retry['sequence_id']} seed {retry['seed']} and it raised "
                f"{retry['exception_class']}: {retry['message']}. "
                f"{taken} of at most {retry['budget']} tolerated generation failures so far. "
                "The batch is redrawn rather than dropped, so this round's observation_count is "
                f"still {len(units)} and stays comparable with every round before it"
            )
            _LOGGER.warning("%s", detail)
            if progress is not None:
                progress.announce(detail)
        return candidate, granted_checkpoint

    def commit_observations(
        batches: Sequence[tuple[int, int]],
        payloads: Sequence[tuple[list[dict[str, Any]], float]],
        cursor_after: int,
    ) -> None:
        """Journal the DIAGNOSTICS of the batches a wave just committed, on the primary only.

        One timing row per committed BATCH rather than one per wave, so a round's sidecar
        holds the same rows at every world size and two arms' per-batch generation costs stay
        comparable. The seconds are the generating rank's own measurement, which arrived in
        the gather; :class:`~sae_bioemu.steering.progress.TimingSidecar` records why summing
        them across a wave overstates wall clock.
        """
        if not primary:
            return
        for index, ((offset, size), (_rows, elapsed)) in enumerate(
            zip(batches, payloads, strict=True)
        ):
            _record_elapsed_timing(
                timings,
                "validation",
                f"validation:{round_index}:{offset}",
                units[offset],
                elapsed,
                "committed",
            )
            if progress is not None:
                progress.observe(
                    phase="validation",
                    elapsed_seconds=elapsed,
                    units=size,
                    completed_units=train_cursor,
                    epoch=round_index,
                    force=cursor_after == len(units) and index == len(batches) - 1,
                )

    while validation_cursor < len(units):
        # THE WAVE: min(W, remaining) consecutive batches from the cursor, slot r on rank r,
        # committed as one journal write. Recomputed from ``validation_cursor`` every
        # iteration, so a resumed round partitions what is LEFT instead of re-deriving a
        # partition of the whole round and replaying its committed prefix.
        wave = _validation_wave(
            validation_offsets, validation_cursor, len(units), context.world_size
        )
        wave_offset, wave_size = wave[0]
        attempt_key = f"validation:{round_index}:{wave_offset}"
        # On the wave-START batch, exactly as the per-batch loop checked its one batch -- and
        # that is the only batch it CAN be. A wave that fails leaves the cursor AT its earliest
        # failing batch and charges that batch alone, and a validation batch is never stepped
        # over, so the only offset at or past the cursor that can carry a charge is the cursor
        # itself. A pure function of the journal, which the collectives below keep identical
        # everywhere, so every rank retires the round together or none of them does.
        if _attempts(journal, attempt_key) >= config.maximum_unit_attempts:
            # Reached both on RESUME into a run whose previous process exhausted this batch and
            # immediately after the loop below exhausts it. Under resample semantics an exhausted
            # batch is no longer terminal by construction: each attempt drew a different derived
            # seed, and the shared tolerance may grant another set of them. Nothing is dropped, so
            # the round still reaches a full ``observation_count`` or the run ends.
            #
            # THE ONLY COPY OF THAT DECISION, and it runs on every rank. The verdict is derived
            # from the shared journal rather than voted on, so either all ranks return here or
            # all of them redraw; no rank can leave for the caller while another waits in the
            # wave's collectives below.
            granted = grant_retry(
                journal,
                attempt_key,
                validation_cursor,
                wave_size,
                tuple(units[wave_offset : wave_offset + wave_size]),
            )
            if granted is None:
                return checkpoint, False
            journal, checkpoint = granted
            continue
        journal["in_flight"] = {
            "phase": "validation",
            "key": attempt_key,
            "round_index": round_index,
            "cursor": wave_offset,
        }
        checkpoint = _make_checkpoint(
            config,
            identity,
            adapter,
            optimizer,
            epoch=round_index,
            cursor=train_cursor,
            journal=journal,
        )
        if primary:
            save_adapter_checkpoint(checkpoint_path, checkpoint)
        local_error: Exception | None = None
        local_failure: dict[str, Any] | None = None
        local_rows: list[dict[str, Any]] = []
        elapsed_seconds = 0.0
        # A short final wave leaves the high ranks with no slot. They draw nothing, snapshot
        # nothing and restore nothing, and still enter every collective below: the verdict and
        # the gather are what make the wave ONE decision.
        if context.rank < len(wave):
            offset, size = wave[context.rank]
            members = tuple(units[offset : offset + size])
            targets = tuple(_unit_target(unit, config, baselines) for unit in members)
            conditioned_targets = tuple(
                _conditioning(unit, target, config)
                for unit, target in zip(members, targets, strict=True)
            )
            adapter_state = _cpu_state(adapter)
            optimizer_state = copy.deepcopy(optimizer.state_dict())
            started = time.monotonic()
            # Tracked for exactly the reason the training loop tracks it: the tolerance may step
            # over a bad STRUCTURE and must never step over a defect in the run, and after the
            # generator has returned every remaining statement is the latter.
            validation_phase = _VALIDATION_GENERATION_PHASE
            # THIS SLOT'S OWN KEY, never the wave's, and this is the condition that makes a
            # sharded round's numbers equal a single process's. ``attempt_key`` above names the
            # batch the wave STARTS at; the batch this rank draws is a different one, and what a
            # batch redraws has to be a function of the BATCH. Both parts already are:
            # ``slot_key`` is cursor-keyed, and ``_generation_attempts_spent`` counts that key in
            # the shared ``failure_keys`` journal, which every rank holds identically and which a
            # granted retry never clears. So the substitute
            # :func:`resampled_generation_seed` derives is a pure function of the batch's offset,
            # its members and how many draws it has really spent -- identical on every rank, and
            # identical whichever rank ends up holding the batch after a retry moved it to
            # another slot. Deriving from the wave key instead would make one batch's structure a
            # function of the PARTITION: the same batch would redraw differently after a resume
            # at another world size, or after a failing neighbour moved the wave boundary, and
            # two runs over one schedule would report different validation numbers while claiming
            # the same seeds.
            slot_key = f"validation:{round_index}:{offset}"
            resample_index = _generation_attempts_spent(journal, slot_key)
            try:
                # D16: validation uses the same estimator as training. D4b's rule is that
                # selection must not disagree with the objective, and a single-draw validation
                # MAE beside an ensemble-mean objective would select on exactly the draw noise
                # the objective averages away.
                #
                # AND WITH GRAD ENABLED, deliberately. Wrapping this call in ``no_grad`` looks
                # attractive -- validation never backpropagates, and the graph it builds is
                # discarded a few lines below, which costs 78 GB against training's 41 GB at
                # the same batch size on the 8-site cryptic arms. It does not work. The
                # generation path REQUIRES the draw to be differentiable, in three places that
                # exist to catch silent breakage:
                #
                #   * ``steering.py`` refuses a denoiser output whose final positions do not
                #     require grad ("BioEmu denoiser final positions are not differentiable"),
                #   * ``steering.py`` refuses ensemble coordinates that do not "preserve
                #     gradients on the target device",
                #   * ``direct_metric.py`` refuses a draw that does not require grad, because
                #     the rigid-body leak this arm must report beside its loss is a gradient
                #     decomposition and "there is no gradient to decompose otherwise".
                #
                # Jobs 30137 and 30138 died at ``validation:0:0`` on the first of these, both
                # attempts, immediately after the first validation round was reached. Buying
                # the memory back means relaxing invariants in the property-measurement core,
                # which is not a trade a speed optimisation gets to make. If validation memory
                # has to come down, the lever is the batch cap, not the grad mode.
                #
                # Sharding changes none of it. It runs W of these draws CONCURRENTLY, one per
                # rank, each still differentiable and each still paying its own ~78 GB on its
                # own card: what the wave removes is the idle card, not the memory.
                sample = _generate_property_ensemble(
                    generate,
                    generate_batch,
                    members,
                    conditioned_targets,
                    ensemble_seeds,
                    resample=(slot_key, resample_index),
                )
                validation_phase = "validation"
                _validate_optimizer_identity(optimizer, adapter, identity["optimizer"])
                _assert_frozen_state(guard)
                rows = _validation_rows(members, targets, sample, config, objective)
                # After the phase has moved off generation, deliberately: a non-scalar row
                # column is a defect in the property callback rather than one bad structure, so
                # it is charged an attempt and refused a redraw like every other
                # post-generation failure.
                _assert_scalar_validation_rows(rows, offset)
            except Exception as error:
                local_error = error
                # Built while the batch that raised is still in scope, and broadcast below
                # only if this rank turns out to be the lowest one that failed. The phase
                # travels with it, because the phase is what decides whether the tolerance may
                # redraw this batch or must end the run.
                local_failure = asdict(
                    _failure(members[0], targets[0], validation_phase, error, members)
                )
            elapsed_seconds = max(time.monotonic() - started, 0.0)
            if local_error is None:
                adapter.load_state_dict(adapter_state)
                optimizer.load_state_dict(optimizer_state)
                # Assert rather than restore. ``_validation_rows`` reads the generated sample
                # and touches no module, so the check above already covers this window: a
                # restore here could only ever copy identical values back, at the cost of a
                # full host-to-device write of the generator per validation unit.
                _assert_frozen_state(guard)
                freeze_for_steering(*guard.modules, adapter)
                local_rows = rows
            else:
                adapter.load_state_dict(adapter_state)
                optimizer.load_state_dict(optimizer_state)
                _restore_frozen_state(guard)
                freeze_for_steering(*guard.modules, adapter)
        # The wave is the atomic commit, so its verdict is collective: a rank must not journal
        # rows the others abandoned, nor abandon rows they journalled. Slot == rank in
        # ascending cursor order, so the LOWEST failing rank is the EARLIEST failing batch and
        # everything gathered below it is a committable PREFIX of the round.
        failing_rank = agreed_first_failing_rank(local_error is not None, context)
        payloads = gather_indexed(
            [(context.rank, (local_rows, elapsed_seconds))]
            if local_error is None and context.rank < len(wave)
            else [],
            context,
        )
        if failing_rank == context.world_size:
            candidate_journal = copy.deepcopy(journal)
            candidate_journal["in_flight"] = None
            candidate_active = candidate_journal["active_validation"]
            # IN SLOT ORDER, which ``gather_indexed`` has already sorted by index: the rows of
            # a sharded round are then the rows the per-batch loop would have appended, in the
            # order it would have appended them.
            for wave_rows, _elapsed in payloads:
                candidate_active["rows"].extend(wave_rows)
            wave_units = sum(batch_size for _offset, batch_size in wave)
            candidate_active["cursor"] = validation_cursor + wave_units
            candidate_checkpoint = _make_checkpoint(
                config,
                identity,
                adapter,
                optimizer,
                epoch=round_index,
                cursor=train_cursor,
                journal=candidate_journal,
            )
            if primary:
                save_adapter_checkpoint(checkpoint_path, candidate_checkpoint)
            journal = candidate_journal
            active = journal["active_validation"]
            checkpoint = candidate_checkpoint
            validation_cursor += wave_units
            commit_observations(wave, payloads, validation_cursor)
            continue
        # ONE failure record, from the lowest failing rank, on every rank's journal, exactly as
        # a training group does it: the attempt counter advances by one per wave attempt
        # however many ranks raised, so ``maximum_unit_attempts`` keeps meaning the number of
        # times a batch was TRIED rather than the number of ranks that tried it.
        failure_payload, failed_seconds = broadcast_from(
            (local_failure, elapsed_seconds) if context.rank == failing_rank else None,
            failing_rank,
            context,
        )
        committed = payloads[:failing_rank]
        # The failing batch's own offset, which is where the cursor must land. The wave's
        # batches are consecutive, so this is exactly ``validation_cursor`` plus the units of
        # the slots below it, and it is a planned batch boundary because the wave walked to it
        # through ``_batch_size_at``.
        failed_offset = wave[failing_rank][0]
        failed_key = f"validation:{round_index}:{failed_offset}"
        candidate_journal = copy.deepcopy(journal)
        candidate_active = candidate_journal["active_validation"]
        for wave_rows, _elapsed in committed:
            candidate_active["rows"].extend(wave_rows)
        candidate_active["cursor"] = failed_offset
        # EXACTLY ONE attempt, charged to the EARLIEST failing batch. The slots above it in this
        # wave are left uncharged on purpose: their draws are abandoned rather than failed, and
        # the next wave redraws them from the same journal at the same derived seeds, so
        # charging them would spend a budget on work that is about to be repeated identically.
        candidate_journal["attempts"][failed_key] = _attempts(journal, failed_key) + 1
        candidate_journal["failures"].append(failure_payload)
        # The monotone spend, which is what the next attempt's resample index reads. It is keyed
        # by the failing BATCH, so the redraw advances to a seed that batch has not drawn --
        # whichever rank draws it next.
        candidate_journal["failure_keys"].append(failed_key)
        candidate_journal["in_flight"] = None
        # ONE atomic write for all of it: the prefix rows, the cursor naming the failing batch,
        # the attempt charge and the failure record. A resume can therefore neither replay a
        # batch whose rows are already journalled nor lose the charge that will eventually
        # retire the round.
        checkpoint = _make_checkpoint(
            config,
            identity,
            adapter,
            optimizer,
            epoch=round_index,
            cursor=train_cursor,
            journal=candidate_journal,
        )
        if primary:
            save_adapter_checkpoint(checkpoint_path, checkpoint)
        journal = candidate_journal
        active = journal["active_validation"]
        commit_observations(wave[:failing_rank], committed, failed_offset)
        if primary:
            _record_elapsed_timing(
                timings,
                "validation",
                failed_key,
                units[failed_offset],
                failed_seconds,
                "failed",
            )
        # The failed batch becomes the next wave's start, so the retry re-partitions from it.
        validation_cursor = failed_offset
        # Unconditionally back to the top, where the exhausted-attempt check now decides
        # between a granted retry and the end of the run. It used to return here, and keeping
        # a second copy of that decision beside the resume copy is how the two come to
        # disagree about when a validation batch is terminal. Sharding makes that stricter
        # rather than looser: the top of the loop is reached by every rank at once, so one copy
        # of the decision is also the only way the ranks can be sure of reaching the same one.
        continue

    # The round-end block runs on EVERY rank, unchanged. Every statement in it is a pure
    # function of the journal, and the wave loop above has just made that journal identical
    # everywhere -- committed rows in slot order, the same cursor, the same attempt charges --
    # so the aggregate, the best-round comparison and the early-stopping verdict are computed
    # rather than broadcast. Only the file writes are gated on the primary.
    rows = active["rows"]
    aggregate = _validation_aggregate(
        rows, round_index, train_cursor, config.is_relative, config.validation_selection_metric
    )
    journal["history"].append(aggregate)
    journal["active_validation"] = {}
    best_index = journal["best_validation_index"]
    best_key = (
        (math.inf, math.inf)
        if best_index is None
        else _validation_key(
            journal["history"][best_index], config.is_relative, _selection_metric_name(config)
        )
    )
    if _validation_key(aggregate, config.is_relative, _selection_metric_name(config)) < best_key:
        journal["best_validation_index"] = round_index
        journal["bad_validation_count"] = 0
        best_path = _best_checkpoint_path(checkpoint_path, round_index)
        best = _make_checkpoint(
            config,
            identity,
            adapter,
            optimizer,
            epoch=round_index + 1,
            cursor=train_cursor,
            journal=journal,
        )
        if primary:
            save_adapter_checkpoint(best_path, best)
        # Set on every rank, written by one. The path and the fingerprint are journal state
        # that the early-stopping and best-restore paths read on every rank, and both are pure
        # functions of state the ranks already agree on.
        journal["best_checkpoint_path"] = str(best_path)
        journal["best_model_optimizer_fingerprint"] = _model_optimizer_fingerprint(best)
    else:
        journal["bad_validation_count"] += 1
    if (
        config.early_stopping_patience is not None
        and journal["bad_validation_count"] >= config.early_stopping_patience
    ):
        journal["early_stopped"] = True
    checkpoint = _make_checkpoint(
        config,
        identity,
        adapter,
        optimizer,
        epoch=round_index + 1,
        cursor=train_cursor,
        journal=journal,
    )
    if primary:
        save_adapter_checkpoint(checkpoint_path, checkpoint)
    return checkpoint, True


#: What a persisted validation row may hold: JSON scalars, or nothing.
_VALIDATION_ROW_SCALARS = (bool, int, float, str)


def _assert_scalar_validation_rows(rows: Sequence[Mapping[str, Any]], offset: int) -> None:
    """Fail closed unless every column of every validation row is a plain scalar.

    Checked BEFORE the rows enter the cross-rank gather. A row is journalled, and the journal
    is JSON, so a tensor column was always wrong -- but on the primary-only path it was wrong
    cheaply and locally, at the atomic write. Sharded, every row is pickled through
    ``all_gather_object`` first: a property that began emitting a per-residue tensor would ship
    one tensor per member per rank over the wire, and a 0-dim tensor would arrive looking
    enough like a number to be averaged into ``target_mae`` instead of raising. So it fails
    here, naming the column and the batch.

    Raised inside the generating rank's own ``try`` block deliberately. That charges the batch
    one attempt and lets the wave's collective verdict retire the round on every rank, where a
    bare raise would unwind one rank out of collectives the others are still waiting in --
    turning a detectable fault into a hang.
    """
    for row in rows:
        for name, value in row.items():
            if value is None or isinstance(value, _VALIDATION_ROW_SCALARS):
                continue
            raise TypeError(
                f"validation batch at offset {offset} produced the non-scalar row column "
                f"{name!r} of type {type(value).__name__}; validation rows are journalled as "
                "JSON and gathered across ranks, so every column must be a plain float, int, "
                "str or None"
            )


def _assert_validation_replica_agreement(
    adapter: nn.Module, context: DistributedContext, round_index: int
) -> None:
    """Fail closed unless every rank enters the round holding the SAME adapter replica.

    The single most important guard sharding adds. While validation ran on the primary alone,
    a non-primary replica could have drifted arbitrarily without touching one reported number:
    the other ranks contributed nothing to the round. Now rank ``r`` draws wave slot ``r`` and
    its rows are extended into the same ``active_validation`` block whose mean IS
    ``target_mae`` -- the best-checkpoint selection metric. Undetected replica drift would
    therefore BLEND TWO DIFFERENT ADAPTERS into one round's selection metric and into the
    epoch that ships, and nothing in the journal, the checkpoint or the report would say so.

    The verdict is collective rather than local. Rank 0 always matches its own broadcast, so a
    local raise would leave the primary walking alone into the wave loop's collectives while
    the other ranks unwound -- converting a fault this function exists to detect into a hang.
    """
    if context.world_size == DEFAULT_WORLD_SIZE:
        return
    digest = _value_fingerprint(_cpu_state(adapter))
    expected = broadcast_from(digest, 0, context)
    mismatched = agreed_first_failing_rank(digest != expected, context)
    if mismatched != context.world_size:
        raise RuntimeError(
            f"validation round {round_index}: rank {mismatched} holds an adapter replica that "
            f"does not match the primary's (primary state fingerprint {expected}). Sharded "
            "validation draws wave slot r on rank r and extends every rank's rows into ONE "
            "active_validation block, so continuing would BLEND TWO DIFFERENT ADAPTERS into "
            "this round's target_mae and into the best-checkpoint selection that decides which "
            "epoch ships -- a corruption no journal, checkpoint or report would record"
        )


def _restore_best_checkpoint(
    current: AdapterCheckpoint,
    checkpoint_path: Path,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: OrdinarySteeringConfig,
    identity: Mapping[str, Any],
    cursor: int,
    journal: dict[str, Any],
) -> AdapterCheckpoint:
    best_path_value = journal.get("best_checkpoint_path")
    expected_fingerprint = journal.get("best_model_optimizer_fingerprint")
    if not isinstance(best_path_value, str) or not isinstance(expected_fingerprint, str):
        raise ValueError("best validation checkpoint identity is missing")
    best_path = Path(best_path_value)
    if best_path != _best_checkpoint_path(checkpoint_path, journal["best_validation_index"]):
        raise ValueError("best validation checkpoint path does not match this run")
    best = load_adapter_checkpoint(
        best_path,
        expected_manifest_fingerprint=config.manifest_fingerprint,
        expected_adapter_config=identity,
        expected_property_callback_identity=config.property_callback_identity,
        expected_source_checkpoints=config.source_checkpoints,
    )
    if _model_optimizer_fingerprint(best) != expected_fingerprint:
        raise ValueError("best validation checkpoint state fingerprint does not match")
    adapter.load_state_dict(best.adapter_state)
    optimizer.load_state_dict(best.optimizer_state)
    journal["selected_best_model_optimizer_fingerprint"] = expected_fingerprint
    return _make_checkpoint(
        config,
        identity,
        adapter,
        optimizer,
        epoch=current.epoch,
        cursor=cursor,
        journal=journal,
    )


def _load_or_create_baselines(
    schedule: SteeringSchedule,
    callback: BaselineCallback,
    callback_identity: str,
    config: OrdinarySteeringConfig,
    checkpoint_path: Path,
    timings: TimingSidecar,
) -> _BaselineState:
    if config.is_relative:  # pragma: no cover - guarded by the caller and _validate_inputs.
        raise RuntimeError("relative_sequence targets must never generate baselines")
    path = (
        config.baseline_path.resolve()
        if config.baseline_path is not None
        else checkpoint_path.with_suffix(checkpoint_path.suffix + ".baselines.json")
    )
    identity = {
        "path": str(path),
        "property_name": config.property_name,
        "model_family": config.model_family.value,
        "baseline_callback_identity": callback_identity,
        "property_callback_identity": config.property_callback_identity,
        "manifest_fingerprint": config.manifest_fingerprint,
        "source_checkpoints": dict(config.source_checkpoints),
    }
    if path.exists():
        payload = _load_baseline_payload(path, identity, schedule)
    else:
        payload = {
            "schema_version": _BASELINE_SCHEMA_VERSION,
            "complete": False,
            "identity": identity,
            "records": [],
            "failures": [],
            "attempts": {},
            "in_flight": None,
        }
        _publish_baseline(path, payload)

    records = {row["sequence_id"]: row["value"] for row in payload["records"]}
    representatives = _sequence_representatives(schedule)
    for sequence_id, unit in representatives.items():
        if sequence_id in records:
            continue
        while payload["attempts"].get(sequence_id, 0) < config.maximum_unit_attempts:
            payload["in_flight"] = sequence_id
            _publish_baseline(path, payload)
            started = time.monotonic()
            try:
                seed_everything(unit.seed)
                value = _finite(callback(unit), "baseline")
            except Exception as error:
                failure = _failure(unit, unit.level, "baseline", error)
                candidate = copy.deepcopy(payload)
                candidate["attempts"][sequence_id] = candidate["attempts"].get(sequence_id, 0) + 1
                candidate["failures"].append(asdict(failure))
                candidate["in_flight"] = None
                _publish_baseline(path, candidate)
                _record_timing(
                    timings, "baseline", f"baseline:{sequence_id}", unit, started, "failed"
                )
                payload = candidate
                continue
            candidate = copy.deepcopy(payload)
            candidate_records = {**records, sequence_id: value}
            candidate["records"] = [
                {"sequence_id": key, "value": candidate_records[key]}
                for key in sorted(candidate_records)
            ]
            candidate["in_flight"] = None
            _publish_baseline(path, candidate)
            _record_timing(
                timings, "baseline", f"baseline:{sequence_id}", unit, started, "committed"
            )
            payload = candidate
            records = candidate_records
            break
        if sequence_id not in records:
            break
    payload["complete"] = set(records) == set(representatives)
    _publish_baseline(path, payload)
    return _BaselineState(
        records,
        _failures(payload["failures"]),
        path,
        payload["content_fingerprint"],
        payload["complete"],
    )


def _record_timing(
    timings: TimingSidecar,
    phase: str,
    key: str,
    unit: SteeringUnit,
    started: float,
    outcome: str,
) -> float:
    """Record one wall-clock measurement in the diagnostic sidecar and return it.

    Deliberately published after the scientific write it describes: the sidecar is never
    part of the checkpoint, so a crash between the two loses only a diagnostic row.
    """
    return _record_elapsed_timing(
        timings, phase, key, unit, max(time.monotonic() - started, 0.0), outcome
    )


def _record_elapsed_timing(
    timings: TimingSidecar,
    phase: str,
    key: str,
    unit: SteeringUnit,
    elapsed_seconds: float,
    outcome: str,
) -> float:
    """Record one ALREADY MEASURED span, for work this process did not itself time.

    Sharded validation needs exactly this. A wave slot is generated and timed on the rank that
    drew it, the measurement travels to the primary inside the wave's gather, and the primary
    -- the only rank that writes -- journals a number it did not measure. Kept separate rather
    than widening :func:`_record_timing` so that every existing call site keeps its ``started``
    form and no existing measurement changes shape.
    """
    timings.record(
        phase=phase,
        key=key,
        sequence_id=unit.sequence_id,
        seed=unit.seed,
        elapsed_seconds=elapsed_seconds,
        outcome=outcome,
    )
    return elapsed_seconds


def _load_baseline_payload(
    path: Path, expected_identity: Mapping[str, Any], schedule: SteeringSchedule
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception as error:
        raise ValueError("unable to load baseline sidecar") from error
    if not isinstance(payload, dict) or frozenset(payload) != _BASELINE_KEYS:
        raise ValueError("baseline sidecar schema is invalid")
    if payload.get("schema_version") != _BASELINE_SCHEMA_VERSION or not isinstance(
        payload.get("complete"), bool
    ):
        raise ValueError("baseline sidecar schema is invalid")
    if _json_bytes(payload.get("identity")) != _json_bytes(expected_identity):
        stored = payload.get("identity")
        moved = (
            sorted(
                key
                for key in set(expected_identity) | set(stored)
                if stored.get(key) != expected_identity.get(key)
            )
            if isinstance(stored, Mapping)
            else []
        )
        # Named rather than left opaque, because ONE of the possible answers has a remedy the
        # message can state. A schedule extension moves ``manifest_fingerprint`` -- the binding
        # fingerprint digests the config FILE, and lengthening the budget edits it -- while the
        # baselines themselves are per-sequence unsteered measurements that the epoch budget does
        # not touch. This sidecar is NOT part of the checkpoint continuation mechanism and is not
        # widened by it: an absolute-target run being extended has to regenerate or repoint its
        # baselines. Every relative-target run (every progress-coordinate and cryptic arm) writes
        # no sidecar at all and never reaches this.
        raise ValueError(
            f"baseline sidecar identity does not match; {moved} differ. A schedule extension "
            "moves manifest_fingerprint and does not widen this sidecar: delete "
            f"{path} to regenerate the baselines, or set baseline_path at the shared file"
        )
    _validate_baseline_collections(payload, schedule)
    expected_fingerprint = _baseline_fingerprint(payload)
    if payload.get("content_fingerprint") != expected_fingerprint:
        raise ValueError("baseline content fingerprint is invalid")
    if payload["complete"] and {row["sequence_id"] for row in payload["records"]} != set(
        _sequence_representatives(schedule)
    ):
        raise ValueError("complete baseline sidecar does not cover the schedule")
    return payload


def _validate_baseline_collections(payload: Mapping[str, Any], schedule: SteeringSchedule) -> None:
    records = payload.get("records")
    failures = payload.get("failures")
    attempts = payload.get("attempts")
    in_flight = payload.get("in_flight")
    if (
        not isinstance(records, list)
        or not isinstance(failures, list)
        or not isinstance(attempts, dict)
    ):
        raise ValueError("baseline sidecar collections are invalid")
    allowed = set(_sequence_representatives(schedule))
    seen = set()
    for row in records:
        if (
            not isinstance(row, dict)
            or frozenset(row) != {"sequence_id", "value"}
            or not isinstance(row["sequence_id"], str)
            or row["sequence_id"] not in allowed
            or row["sequence_id"] in seen
        ):
            raise ValueError("baseline sidecar records are invalid")
        _finite(row["value"], "baseline")
        seen.add(row["sequence_id"])
    _failures(failures)
    for key, value in attempts.items():
        if key not in allowed or not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("baseline sidecar attempts are invalid")
    if in_flight is not None and (not isinstance(in_flight, str) or in_flight not in allowed):
        raise ValueError("baseline sidecar in-flight state is invalid")
    if payload["complete"] and in_flight is not None:
        raise ValueError("complete baseline sidecar cannot contain in-flight work")


def _publish_baseline(path: Path, payload: dict[str, Any]) -> None:
    payload["content_fingerprint"] = _baseline_fingerprint(payload)
    _atomic_json(path, payload)


def _baseline_fingerprint(payload: Mapping[str, Any]) -> str:
    content = {key: payload[key] for key in sorted(_BASELINE_KEYS - {"content_fingerprint"})}
    return hashlib.sha256(_json_bytes(content)).hexdigest()


def _resume_source(path: Path, config: OrdinarySteeringConfig) -> Path:
    """Return the file a run adopts its state from: the main checkpoint, or a ``.best-N`` sidecar.

    The sidecar is taken only while the destination does not already record having adopted it.
    Otherwise a run preempted after the rollback would roll back a second time on every restart,
    discarding everything it had done since -- and unlike the extension case there is nothing in
    the identity to tell the two apart, because a rollback under an unchanged config leaves the
    destination carrying exactly this run's own fingerprint. The journal is what distinguishes
    them, so the journal is what is consulted.
    """
    index = config.resume_from_validation_index
    if index is None:
        return path
    if path.exists() and _records_a_resume(read_adapter_checkpoint(path).validation_record):
        return path
    return _best_checkpoint_path(path, index)


def _records_a_resume(journal: Mapping[str, Any]) -> bool:
    """True when this journal already records a resume, i.e. the rollback has been taken."""
    if not isinstance(journal, Mapping):
        return False
    return bool(journal.get(_SCHEDULE_EXTENSIONS_KEY))


def _origin_terminal_reason(source: Path, path: Path, candidate: AdapterCheckpoint) -> str | None:
    """Return the terminal state of the RUN being continued, not of the file it is read from.

    Those are different files whenever a ``.best-N`` sidecar is named, and the run's is the one
    that answers the question. A sidecar is written mid-run, so its journal always says
    ``terminal_reason: null`` -- which would report an early-stopped run as though it had simply
    been preempted, silently losing the one fact most worth recording about it.
    """
    if source != path and path.exists():
        journal = read_adapter_checkpoint(path).validation_record
        if isinstance(journal, Mapping):
            return journal.get("terminal_reason")
    journal = candidate.validation_record
    return journal.get("terminal_reason") if isinstance(journal, Mapping) else None


def _resume_expectation(
    source: Path,
    path: Path,
    config: OrdinarySteeringConfig,
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], ScheduleExtension | None]:
    """Return what a resume must match, and the record -- if any -- of what it adopted.

    When ``source`` is the destination and already carries this run's own fingerprint -- the
    ordinary preemption restart, which is the overwhelming majority of resumes -- this is exactly
    the expectation the caller used before any of this existed: the run's own manifest fingerprint
    and its own ``adapter_config``, compared for equality, and no record written.

    Otherwise the checkpoint is read FIRST -- structurally validated against itself, but bound to
    nothing -- so that its ``unit_cursor`` and its own ``adapter_config`` can be used to derive the
    expectation. Only the five length-derived keys are ever taken from the checkpoint, and only
    after :func:`reconcile_extended_adapter_config` has compared every other key for exact equality
    and confirmed the completed prefix element for element.
    """
    expected = {
        "expected_manifest_fingerprint": config.manifest_fingerprint,
        "expected_adapter_config": dict(identity),
        "expected_property_callback_identity": config.property_callback_identity,
        "expected_source_checkpoints": config.source_checkpoints,
    }
    if not source.exists():
        return expected, None
    candidate = read_adapter_checkpoint(source)
    matches = candidate.manifest_fingerprint == config.manifest_fingerprint
    # ALREADY THIS RUN'S OWN CHECKPOINT AT THE DESTINATION: the ordinary path, no record.
    #
    # This is also what makes a continued run resumable a second time. The first resume
    # republishes the destination under this run's identity (see the training loop), so a
    # continued run that is then preempted leaves a destination carrying this run's own manifest
    # fingerprint. Reconciling that again would refuse the run's own artifact for not being the
    # checkpoint it already consumed, and on a multi-rank run the non-primary ranks would refuse
    # the freshly published checkpoint the primary just wrote.
    if matches and source == path:
        return expected, None

    _require_resumable_state(candidate, source, config)
    terminal_reason = _origin_terminal_reason(source, path, candidate)
    if matches:
        # A ROLLBACK with no identity move: the plan is this run's own plan, so there is nothing
        # to reconcile and nothing is relaxed. What is recorded is that the run starts at an
        # earlier cursor than the artifact in ``output_dir`` had reached.
        stored_train = identity["schedule_inventory"]["train"]
        extension = ScheduleExtension(
            from_manifest_fingerprint=candidate.manifest_fingerprint,
            to_manifest_fingerprint=config.manifest_fingerprint,
            from_schedule_fingerprint=str(identity["schedule_fingerprint"]),
            to_schedule_fingerprint=str(identity["schedule_fingerprint"]),
            from_maximum_epochs=inventory_epochs(stored_train),
            to_maximum_epochs=inventory_epochs(stored_train),
            from_train_units=len(stored_train),
            to_train_units=len(stored_train),
            resumed_at_epoch=int(candidate.epoch),
            resumed_at_unit_cursor=int(candidate.unit_cursor),
            resumed_from=source.name,
            from_terminal_reason=terminal_reason,
        )
        return expected, extension

    reconciliation = reconcile_extended_adapter_config(
        checkpoint_config=candidate.adapter_config,
        checkpoint_manifest_fingerprint=candidate.manifest_fingerprint,
        checkpoint_unit_cursor=candidate.unit_cursor,
        expected_config=identity,
        expected_manifest_fingerprint=config.manifest_fingerprint,
        declaration=config.schedule_extension,
    )
    expected["expected_manifest_fingerprint"] = reconciliation.expected_manifest_fingerprint
    expected["expected_adapter_config"] = dict(reconciliation.expected_adapter_config)
    return expected, replace(
        reconciliation.extension,
        resumed_at_epoch=int(candidate.epoch),
        resumed_from=source.name,
        from_terminal_reason=terminal_reason,
    )


def _require_resumable_state(
    checkpoint: AdapterCheckpoint, source: Path, config: OrdinarySteeringConfig
) -> None:
    """Refuse a checkpoint whose weights are not the weights its own cursor produced.

    This is the ONE rule. It used to be two, and the second -- "an early-stopped run may not be
    extended, because stopping was a decision" -- is gone. A longer budget does not overrule that
    decision so much as ask a different question, and the state an early-stopped run leaves behind
    is continuable on exactly the same terms as any other: its ``.best-N`` sidecar holds epoch N's
    adapter, optimizer, cursor and journal together, and resuming there restarts the patience
    counter from the epoch that earned it rather than from the tail that spent it. Refusing that
    was refusing to continue a checkpoint for a reason unrelated to whether it CAN be continued.
    What the run records instead is ``from_terminal_reason``, so an extension of an early-stopped
    run is readable as one.

    What remains is mechanical. A selecting run that terminates restores the adapter AND optimizer
    from ``best_validation_index``; when that is not the last epoch, the state in the main
    checkpoint is not the state its cursor claims, so continuing from it would be a fresh run from
    a selected adapter wearing a journal claiming epochs it did not produce. The refusal now names
    the remedy, because there is one: the ``.best-N`` sidecar for that very epoch.
    """
    journal = checkpoint.validation_record
    if not isinstance(journal, Mapping):
        raise ScheduleExtensionError("checkpoint training journal schema is invalid")
    history = journal.get("history") or []
    best = journal.get("best_validation_index")
    if (
        config.selects_on_validation
        and journal.get("terminal_reason") is not None
        and history
        and best != len(history) - 1
    ):
        raise ScheduleExtensionError(
            f"cannot continue {source.name}: model selection rolled its weights back to epoch "
            f"{best} of {len(history)}, so the adapter it holds is not the one its cursor "
            f"produced. Resume from the sidecar that epoch wrote instead, by setting "
            f"schedule.resume_from_validation_index: {best}"
        )


def _resume_checkpoint(
    resume: AdapterCheckpoint | str | Path | None,
    path: Path,
    config: OrdinarySteeringConfig,
    identity: Mapping[str, Any],
) -> tuple[AdapterCheckpoint | None, ScheduleExtension | None]:
    source = _resume_source(path, config)
    # A NAMED continuation with nothing to continue is a refusal, not a fresh run. Training from
    # zero here would walk the whole plan while the report claimed it continued a checkpoint --
    # the one reading of the artifacts that must never be available. The usual cause is a mistyped
    # or moved ``output_dir``. Checked BEFORE ``resume``, because a missing sidecar beside a
    # present destination would otherwise fall through to a silent fresh run.
    if not source.exists():
        if config.resume_from_validation_index is not None:
            raise ScheduleExtensionError(
                "this run declares schedule.resume_from_validation_index "
                f"{config.resume_from_validation_index} but there is no checkpoint at {source} "
                "to continue; a named continuation does not silently start a fresh run"
            )
        if config.schedule_extension is not None:
            raise ScheduleExtensionError(
                "this run declares schedule.extends "
                f"{config.schedule_extension.from_manifest_fingerprint} but there is no "
                f"checkpoint at {source} to continue; a declared continuation does not silently "
                "start a fresh run"
            )
        if resume is None:
            return None, None
    expected, extension = _resume_expectation(source, path, config, identity)
    authoritative = load_adapter_checkpoint(source, **expected) if source.exists() else None
    # The destination is deliberately NOT the state being adopted when a sidecar was named, so
    # the staleness comparison below would compare two checkpoints that are SUPPOSED to differ.
    if resume is None or source != path:
        return authoritative, extension
    candidate = (
        validate_adapter_checkpoint(resume, **expected)
        if isinstance(resume, AdapterCheckpoint)
        else load_adapter_checkpoint(Path(resume).resolve(), **expected)
    )
    if authoritative is not None and ordinary_checkpoint_state_fingerprint(
        candidate
    ) != ordinary_checkpoint_state_fingerprint(authoritative):
        raise ValueError("resume checkpoint is stale relative to the authoritative destination")
    return (authoritative if authoritative is not None else candidate), extension


def _adapter_config(
    schedule: SteeringSchedule,
    baseline_identity: str | None,
    baseline_state: _BaselineState,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    objective: SteeringLoss,
    config: OrdinarySteeringConfig,
    batch_plan: SteeringBatchPlan | None = None,
    accumulation: GradientAccumulationPlan | None = None,
) -> dict[str, Any]:
    # RESOLVED here when the caller did not, rather than omitted. Every run now takes optimizer
    # steps of a uniform number of scheduled units, so "no accumulation plan" is not a state a
    # checkpoint can be in: an identity written without the block could never be evaluated,
    # because the evaluation path rebuilds the plan from the schedule and would find the key
    # missing. The plan is a pure function of the schedule and the batch plan, so deriving it
    # here gives the same answer the caller would have passed.
    if accumulation is None:
        accumulation = resolve_gradient_accumulation(
            schedule,
            batch_plan,
            config.gradient_accumulation_units,
            world_size=config.distributed_world_size,
        )
    provenance: dict[str, Any] = (
        {}
        if config.is_relative
        else {
            "baseline_callback_identity": baseline_identity,
            "baseline_path": str(baseline_state.path),
            "baseline_content_fingerprint": baseline_state.content_fingerprint,
        }
    )
    return {
        "workflow": "ordinary",
        "property_name": config.property_name,
        "model_family": config.model_family.value,
        "intervention_site": config.intervention_site,
        "generation_callback_identity": config.generation_callback_identity,
        "property_callback_identity": config.property_callback_identity,
        "manifest_fingerprint": config.manifest_fingerprint,
        "source_checkpoints": dict(config.source_checkpoints),
        "schedule_fingerprint": _schedule_fingerprint(schedule),
        "schedule_inventory": _schedule_inventory(schedule),
        "target_levels": list(config.dose_levels.target),
        # Conditional so a continuous run keeps its exact historical key set. Recorded when it
        # is not the historical default, because validation SELECTS the checkpoint: two runs
        # differing only in the validation dose set are two optimizations (D-44).
        **(
            {}
            if config.dose_levels == CONTINUOUS_DOSE_LEVELS
            else {
                "dose_levels": config.dose_levels.name,
                "epoch_validation_levels": list(config.dose_levels.epoch_validation),
            }
        ),
        # D-46. Conditional for the same reason: absent means the historical origin, which is
        # what every radius-of-gyration and AF3 checkpoint was trained under. It MUST be
        # recorded when it differs, because two runs at one dose set and two origins are asking
        # for two different displacements and their checkpoints must not be interchangeable.
        **(
            {}
            if config.relative_target_origin == TARGET_ORIGIN_SEQUENCE_MEAN
            else {"relative_target_origin": config.relative_target_origin}
        ),
        # One optimizer step now covers a whole batch, so the batching is part of what
        # the checkpoint *is*: two runs with different batches cannot be confused.
        "generation_batching": None if batch_plan is None else batch_plan.identity(),
        **_optimizer_step_identity(config, accumulation, schedule),
        **_generation_precision_identity(config),
        **provenance,
        **_target_identity(config),
        "training_seed": config.training_seed,
        "adapter": _module_config(adapter),
        "objective": _objective_config(objective),
        **_training_subset_identity(config),
        **_cryptic_arm_identity(config),
        **_property_arm_identity(config),
        "optimizer": _optimizer_config(optimizer, adapter),
        "maximum_unit_attempts": config.maximum_unit_attempts,
        "validation_target_mae_maximum": config.validation_target_mae_maximum,
        "validation_residual_fraction_maximum": config.validation_residual_fraction_maximum,
        "early_stopping_patience": config.early_stopping_patience,
        **(
            {}
            if config.gradient_clip_norm is None
            else {"gradient_clip_norm": config.gradient_clip_norm}
        ),
        **_partial_pair_coverage_identity(config),
        **_model_selection_identity(config),
        **_generation_failure_tolerance_identity(config),
    }


def _generation_failure_tolerance_identity(config: OrdinarySteeringConfig) -> dict[str, Any]:
    """Return the declared generation-skip budget, and only when a run has one.

    Conditional for exactly the reason ``cryptic_arm`` and ``model_selection`` are: this block is
    compared with canonical-JSON equality on resume and at the evaluation gate, so an unconditional
    key would refuse every checkpoint ever written. Absence therefore MEANS
    ``maximum_generation_failures = 0`` -- one exhausted unit ends the run -- which is what every
    checkpoint on disk was trained under.

    Present, it is identity-bearing rather than descriptive, and the distinction matters because
    the project's own rule is to hash what DETERMINES rather than what describes. The budget
    determines which units the remaining schedule is walked over: a resume under a larger budget
    could skip a group the original run would have died on, so the two would have walked different
    schedules while sharing a checkpoint file. The journal's ``skipped_units`` block records what
    actually happened; this key records what was permitted to.
    """
    if not config.tolerates_generation_failures:
        return {}
    return {
        "generation_failure_tolerance": {
            "maximum_generation_failures": (
                None
                if config.maximum_generation_failures is None
                else int(config.maximum_generation_failures)
            ),
            "maximum_generation_failure_fraction": float(
                config.maximum_generation_failure_fraction
            ),
        }
    }


def _model_selection_identity(config: OrdinarySteeringConfig) -> dict[str, Any]:
    """Return the no-validation declaration, and only when it was made.

    Conditional for exactly the reason ``cryptic_arm`` and ``training_subset`` are: this block is
    compared with canonical-JSON equality on resume and at the evaluation gate, so an
    unconditional key would refuse every checkpoint ever written. Absence therefore MEANS
    ``validation_split``, which is what every run before this existed did.

    Present, it says in the checkpoint itself that no round of model selection was ever run. The
    schedule inventory beside it already carries an empty validation role, but an empty list is
    evidence and not a declaration -- a reader cannot tell a deliberately unselected arm from one
    whose validation role went missing, and only one of those is a result. Recording the
    declaration is what keeps the two distinguishable for as long as the checkpoint exists.
    """
    if config.selects_on_validation:
        return {}
    return {"model_selection": MODEL_SELECTION_NONE}


def _partial_pair_coverage_identity(config: OrdinarySteeringConfig) -> dict[str, Any]:
    """Return the opt-out key, and only when it was taken.

    Conditional for the same reason as every other key here: absent is what every checkpoint
    ever written carries, and the refusal is the default, so absence means "this plan formed
    every pair its lambda_c was weighted for". Present means the operator accepted a weaker
    paired term than the weights state, which is a different optimization and must not be
    interchangeable with the full one.
    """
    if not config.allow_partial_pair_coverage:
        return {}
    return {"allow_partial_pair_coverage": True}


def _training_subset_identity(config: OrdinarySteeringConfig) -> dict[str, Any]:
    """Return D-23's training-subset block, or nothing at all for an unfiltered arm.

    Conditional for exactly the reason ``cryptic_arm`` is: ``adapter_config`` is compared with
    canonical-JSON equality on resume and at the evaluation gate, so an unconditional key would
    refuse every checkpoint written before this field existed. Absence therefore MEANS "this arm
    trained on its whole train role", which is what makes the unfiltered D-66 arm byte-identical
    to a run made before the field existed.

    What it carries is the artifact's path, its byte digest, its content digest and how many
    sequences it removed -- everything needed to say that this checkpoint's training population
    was that file's, and to refuse it once that file changes. The excluded ids themselves are
    NOT here: the schedule inventory in the same ``adapter_config`` already pins which units the
    run walked, so the list would be a second copy that can disagree with the first, and the run
    report records it in full for the reader.
    """
    subset = config.training_subset
    if subset is None:
        return {}
    return {"training_subset": subset.identity()}


def _cryptic_arm_identity(config: OrdinarySteeringConfig) -> dict[str, Any]:
    """Return the cryptic arm's identity block, or nothing at all for every other property.

    Conditional for exactly the reason ``lambda_perp`` is conditional in
    :func:`_objective_config`: ``adapter_config`` is compared with canonical-JSON equality on
    resume and at the evaluation gate, so an unconditional key would refuse every
    radius-of-gyration checkpoint ever written.

    What it carries is the pocket artifact's identity, the reference table's, the setpoint
    table's digest, the split fingerprint and the config hash -- everything that decides what
    ``the progress coordinate`` MEANS. Two arms over one split with different pair sets steer two different
    coordinates, and without this their checkpoints would be interchangeable.
    """
    identity = config.cryptic_arm_identity
    if identity is None:
        return {}
    return {"cryptic_arm": dict(identity)}


def _property_arm_identity(config: OrdinarySteeringConfig) -> dict[str, Any]:
    """Return ANY property's resolved-arm identity block, under one generic key.

    The generic twin of :func:`_cryptic_arm_identity`, and the reason it exists is that the
    cryptic one is NOT generic: ``cryptic_arm_identity`` is also what
    :attr:`OrdinarySteeringConfig.steers_cryptic_pocket` is derived from, so it carries a second
    meaning -- "this run journals the D-35b monitor columns" -- that no other property's arm should
    inherit. A fourth property binding its arm through that field would silently acquire four
    journal columns it has no values for.

    So a property with an arm now returns ``property_arm_identity`` from its own
    ``checkpoint_arm_arguments`` hook and costs no field here. ``cryptic_arm_identity`` is kept
    exactly as it is because its key is persisted in every cryptic checkpoint on disk; renaming it
    would re-key them for a rename that changes no number.

    Conditional for the reason every identity key here is: ``adapter_config`` is compared with
    canonical equality, so an unconditional key would refuse every checkpoint written before it
    existed.
    """
    identity = config.property_arm_identity
    if identity is None:
        return {}
    return {"property_arm": dict(identity)}


def _optimizer_step_identity(
    config: OrdinarySteeringConfig,
    accumulation: GradientAccumulationPlan | None,
    schedule: SteeringSchedule,
) -> dict[str, Any]:
    """Return the identity keys that describe what one optimizer step covers.

    ``property_ensemble`` and ``diagnostic_fixed_generation_seed`` are **conditional**,
    present only when their feature is on, so a run that uses neither keeps the identity keys
    it had before either existed.

    ``gradient_accumulation`` is **unconditional**, and that is what re-keys every checkpoint
    trained before the uniform optimizer step. It used to be written only when the declared
    ``gradient_accumulation_units x distributed_world_size`` exceeded one, because below that
    an optimizer step was one generation batch and the plan said nothing the batch plan did
    not. It now always says something: the step holds
    :data:`UNIFORM_OPTIMIZER_STEP_UNITS` scheduled units and the realised micro-batch
    composition inside it is a property of the pool's lengths. A checkpoint trained under the
    old grouping is a different optimization from one trained under this, and the differing
    key sets are what refuse to resume or evaluate the one as the other.

    The **declared** ``gradient_accumulation_units`` is deliberately absent, for the reason
    ``distributed_world_size`` is: under the uniform rule two runs that declare ``K = 1`` and
    ``K = 6`` take the identical optimizer step, and recording the knob would retire a
    checkpoint for a number that no longer determines anything.

    ``property_ensemble`` belongs here rather than beside the target construction because
    it changes what an optimizer step optimizes: with ``M > 1`` the property term compares
    the *mean* of ``M`` structures against the target, so the residual multiplying the
    gradient is the ensemble's rather than each draw's (D16). Two runs at different ``M``
    are different optimizations in exactly the sense two runs at different accumulation are.

    **``distributed_world_size`` is deliberately NOT a key of its own**, and now it enters
    the optimization nowhere at all: the ``W`` ranks split ONE step's micro-batches between
    them, so a four-GPU run and a single-process run take the identical optimizer step over
    the identical units. Adding a ``world_size`` key would re-key a mathematically identical
    optimization by the hardware that ran it, which is the same defect as hashing the node
    name into a training fingerprint -- it would make a four-GPU checkpoint unresumable and
    unevaluable on one GPU while binding nothing. What *is* bound is the realized group
    composition, which is a function of the schedule and the length heuristic and of neither
    ``W`` nor ``K``.
    """
    identity: dict[str, Any] = {}
    if accumulation is not None:
        identity["gradient_accumulation"] = accumulation.identity()
    if config.averages_property_ensemble:
        identity["property_ensemble"] = _property_ensemble_identity(schedule, config)
    if config.diagnostic_fixed_generation_seed:
        identity["diagnostic_fixed_generation_seed"] = {
            "scope": FIXED_GENERATION_SEED_SCOPE,
            "rule": FIXED_GENERATION_SEED_RULE,
            "sequence_seeds": _fixed_generation_seeds(schedule, config),
        }
    return identity


def _generation_precision_identity(config: OrdinarySteeringConfig) -> dict[str, Any]:
    """Return the conditional record of reduced-precision generation, or nothing.

    Conditional for the same reason ``gradient_accumulation`` and ``property_ensemble`` are
    (:func:`_optimizer_step_identity`): ``fp32`` touches no backend flag and produces
    byte-identical structures to a run made before the knob existed, so it must produce a
    byte-identical identity key set as well. Anything else is recorded, so a bf16 checkpoint
    and an fp32 checkpoint can neither be resumed into nor evaluated as one another.

    That gate matters more here than it looks. TF32 and bf16 are deterministic, so nothing
    about *reproducibility* forces the separation -- what forces it is that they generate
    **different structures**. Evaluating an fp32-trained checkpoint under bf16 would score a
    frozen adapter against structures the training generator never produced, and the matched
    adapter-off arm would stop being the counterfactual the paired design rests on: exactly
    the failure D14 keeps ``generation_shared`` inside the binding fingerprint to prevent.
    """
    if not config.reduces_generation_precision:
        return {}
    return {
        "generation_precision": resolve_generation_precision(config.generation_precision).identity()
    }


def _validate_generation_precision_identity(adapter_config: Mapping[str, Any]) -> None:
    """Fail closed unless a recorded precision block is a real reduced-precision mode.

    Presence is meaning, as everywhere else in the conditional identity: a block declaring
    ``fp32`` would claim a reduced-precision run for one that took none, and a block whose
    implied policies disagree with this code would describe arithmetic this code cannot
    perform.
    """
    recorded = adapter_config.get("generation_precision")
    if recorded is None:
        return
    if not isinstance(recorded, Mapping) or "mode" not in recorded:
        raise ValueError("ordinary checkpoint generation precision identity is invalid")
    mode = recorded["mode"]
    if not isinstance(mode, str) or mode == DEFAULT_GENERATION_PRECISION:
        raise ValueError(
            "a recorded generation_precision block must name a reduced-precision mode; a "
            f"run generating in {DEFAULT_GENERATION_PRECISION} records no block at all"
        )
    if dict(recorded) != resolve_generation_precision(mode).identity():
        raise ValueError("ordinary checkpoint generation precision convention is invalid")


def _property_ensemble_identity(
    schedule: SteeringSchedule, config: OrdinarySteeringConfig
) -> dict[str, Any]:
    """Return the content-bound record of the ensemble-mean property estimator (D16).

    The ``M`` seeds of every scheduled unit are a pure function of the schedule and ``M``,
    so binding their fingerprint rather than the whole table keeps the identity compact
    while still making every seed content-bound: a checkpoint can never be ambiguous about
    which structures its optimizer steps averaged, and
    :func:`_validate_optimizer_step_identity` recomputes the fingerprint from the schedule
    rather than trusting the recorded one.
    """
    return {
        "samples": config.property_ensemble_samples,
        "estimator": ENSEMBLE_MEAN_PROPERTY_ESTIMATOR,
        "residual_reduction": ENSEMBLE_RESIDUAL_REDUCTION,
        "seed_rule": PROPERTY_ENSEMBLE_SEED_RULE,
        "seed_fingerprint": _property_ensemble_seed_fingerprint(
            build_property_ensemble_seeds(schedule, config.property_ensemble_samples)
        ),
    }


def _property_ensemble_seed_fingerprint(seeds: Mapping[int, Sequence[int]]) -> str:
    """Fingerprint the per-unit ensemble seed table in a canonical, key-sorted order."""
    return hashlib.sha256(
        _json_bytes([[key, list(seeds[key])] for key in sorted(seeds)])
    ).hexdigest()


def _property_ensemble_seeds(
    schedule: SteeringSchedule, config: OrdinarySteeringConfig
) -> dict[int, tuple[int, ...]] | None:
    """Return the per-unit ensemble generation seeds, or ``None`` at one structure per unit.

    ``None`` rather than a table of one-member tuples: the generation path then takes the
    exact pre-ensemble branch -- one ``seed_everything`` and one generator call on the
    scheduled units themselves -- so ``property_ensemble_samples: 1`` cannot differ from a
    run made before the knob existed.
    """
    if not config.averages_property_ensemble:
        return None
    return build_property_ensemble_seeds(schedule, config.property_ensemble_samples)


def _target_identity(config: OrdinarySteeringConfig) -> dict[str, Any]:
    """Return the content-bound description of how targets are constructed.

    The absolute path keeps its exact historical keys. The relative path has no
    baseline and no normalization constants, so the reference table's content
    fingerprint takes their place: a checkpoint is never ambiguous about what
    determined its targets.
    """
    if not config.is_relative:
        return {
            "target_range": asdict(config.target_range),
            "maximum_change": float(config.maximum_change),
            "target_mean": float(config.target_mean),
            "target_scale": float(config.target_scale),
        }
    identity: dict[str, Any] = {
        "target_kind": RELATIVE_SEQUENCE_TARGET,
        "property_loss_normalization": config.property_loss_normalization,
        # A property of the TARGET CONSTRUCTION -- "error on this path is measured in sigma, so
        # selection must be too (D4b)" -- and not of this run's cadence. It is therefore written
        # unconditionally, including on a ``model_selection: none`` run that selects nothing, so
        # that every relative checkpoint keeps one key set. What says whether any round ran is
        # the ``model_selection`` key beside it, and the run report's own
        # ``validation_selection_metric`` is null there.
        #
        # Since the selection metric became a PROPERTY's declaration it is this run's resolved
        # answer rather than the constant. It is ``target_mae_sigma`` for every arm on disk and
        # every arm that does not declare otherwise, so every existing checkpoint identity is
        # byte-identical; a direct-metric arm records what it actually selected on, which is the
        # entire point -- an arm that trained on a metric and shipped the epoch its the progress coordinate proxy liked
        # would otherwise say ``target_mae_sigma`` and look ordinary.
        "validation_selection_metric": _selection_metric_name(config),
        # CONDITIONAL, so absent for every run made before a property could declare a metric --
        # which is every checkpoint on disk. Present it names the validation COLUMN the metric
        # above is the mean of, and it is here rather than folded into that key because the key's
        # shape is persisted: turning a string into a mapping would change every existing identity
        # for a field none of them vary.
        #
        # It is what lets ``validate_ordinary_completion_checkpoint`` RE-DERIVE the selected
        # aggregate from the rows rather than take the recorded number on trust. Without it the
        # gate would admit the column by name and check none of its values, which is the shape of
        # a gate that passes whatever it is given.
        **(
            {}
            if config.validation_selection_metric is None
            else {"validation_selection_column": config.validation_selection_metric.row_column}
        ),
    }
    if config.reads_property_reference:
        # Absent exactly on the system-constant origin, which loads no table and is refused if it
        # is given one. The two keys are replaced by the table that DID determine its targets,
        # rather than dropped: a checkpoint is never ambiguous about what its targets came from.
        identity["property_reference"] = config.property_reference.identity()
        identity["sigma_eligibility"] = config.sigma_eligibility.identity()
    else:
        identity["system_target_constants"] = config.system_target_constants.identity()
    if config.scales_dose_by_sigma:
        # Present exactly when there IS a ``k``, so a two-state or system-constant checkpoint can
        # never be mistaken for a sigma-relative one that happened to run at ``k = 1``, and a
        # radius-of-gyration identity keeps its exact historical key set.
        identity["dose_multiplier"] = float(config.dose_multiplier)
    return identity


def _unit_target(
    unit: SteeringUnit, config: OrdinarySteeringConfig, baselines: Mapping[str, float]
) -> float:
    """Return this unit's target, in the frame a SCHEDULED UNIT can determine.

    On the absolute and sigma-relative paths that is the whole target and its frame is the
    prediction's own.

    On the TWO-STATE path it deliberately is not. What comes back there is the absolute
    REPORTING-frame STATE the level names -- 1.0 for holo / DFG-out, 0.0 for apo / DFG-in --
    and the training-frame target is that state minus THIS DRAW's matched control, which
    :func:`_loss_inputs` subtracts because only a generated sample carries it. Splitting the
    target across two functions is not a convenience: the state is a pure function of the level,
    so a persisted journal row can be re-derived and re-checked here, while the origin is a
    per-draw measurement that must be persisted rather than recomputed.
    """
    if config.is_relative:
        if config.targets_two_state:
            # The ask, and only the ask. No ``k`` and no ``sigma``: ``the progress coordinate`` is unit-length
            # apo->holo for every protein by construction, so an absolute state target is
            # already comparable across proteins and there is no denominator left to apply.
            return two_state_reporting_target(unit.level)
        if config.targets_system_constant:
            # The system's own threshold, identical at both of its levels. This IS the whole
            # target: the level selects which reference the prediction is measured to, which
            # happens where the structure is measured and not here, so a scheduled unit
            # determines the target completely and no per-draw origin is subtracted later.
            return system_constant_target(
                config.system_target_constants.require(unit.sequence_id), unit.level
            )
        statistics = config.property_reference.statistics(unit.sequence_id)
        if config.relative_target_origin == TARGET_ORIGIN_MATCHED_DRAW_ZERO:
            # A pure displacement. The prediction is measured against a matched unsteered draw,
            # so its zero point IS zero and the reference table's mean -- an absolute position
            # measured in a different frame -- must not be added to it (D-46).
            return relative_displacement_target(
                statistics.sigma,
                unit.level,
                dose_multiplier=config.dose_multiplier,
            )
        return relative_sequence_target(
            statistics.mean,
            statistics.sigma,
            unit.level,
            dose_multiplier=config.dose_multiplier,
        )
    return continuous_target(
        baselines[unit.sequence_id],
        config.target_range,
        config.maximum_change,
        unit.level,
    )


def _unit_sigma(unit: SteeringUnit, config: OrdinarySteeringConfig) -> float:
    """Return the per-sequence property loss scale for one relative-path unit.

    One function of the run's own origin, so the loss, the journal's ``sequence_sigma`` column and
    every checkpoint-row re-check read ONE number. On every origin but the system-constant one it
    is the reference table's ``sigma_seq``: the sequence's own conformational spread, which is what
    a sigma-relative or a state target is measured in.

    On :data:`~sae_bioemu.steering.schedule.TARGET_ORIGIN_SYSTEM_CONSTANT` it is the SYSTEM's own
    reference separation, and the column keeps its name while changing its meaning. That is stated
    here and in ``domain_motion/loss.py`` rather than papered over with a rename: renaming
    ``sequence_sigma`` would move the persisted journal schema and re-key every checkpoint on disk
    for a column whose values are unchanged everywhere they already existed. What a reader needs is
    that the number is a per-protein scale in the property's own units, which it is on both
    origins, plus the run's own ``relative_target_origin`` to say which quantity it measures -- and
    that is in the checkpoint identity.
    """
    if config.targets_system_constant:
        return config.system_target_constants.require(unit.sequence_id).scale
    return config.property_reference.statistics(unit.sequence_id).sigma


def _batch_size_at(offsets: Mapping[int, int], cursor: int, label: str) -> int:
    """Return the units of the atomic step starting at ``cursor``, or fail closed.

    A cursor that is not a planned start means the atomic unit was committed in pieces,
    which is exactly the resume hazard batching introduces: resuming there would either
    replay members that already took an optimizer step or skip members that never did.
    Under gradient accumulation the atomic unit is the accumulation group -- ``zero_grad``
    and ``step`` bracket the whole group -- so a cursor at an inner generation-batch
    boundary describes half an accumulated gradient with no step taken, and is refused.
    """
    size = offsets.get(cursor)
    if size is None:
        raise ValueError(
            f"{label} cursor {cursor} is not a planned batch boundary; resuming here would "
            "replay or skip members of a partially committed batch or accumulation group"
        )
    return size


def _validation_wave(
    validation_offsets: Mapping[int, int],
    cursor: int,
    total: int,
    world_size: int,
) -> tuple[tuple[int, int], ...]:
    """Return the ``(offset, size)`` batches of the next validation WAVE, starting at ``cursor``.

    A wave is the next ``min(world_size, remaining)`` consecutive validation batches in cursor
    order, and rank ``r`` draws slot ``r``. It is the unit sharded validation commits: the
    per-batch loop this replaced generated 79 batches of 2 units one at a time, measured
    16.9-17.2 min per round once per epoch -- ~11 h over 40 epochs, about 32% of wall clock --
    with every non-primary card idle holding ~42 GB for the duration.

    **Walked from ``cursor``, never from the round's start.** The validation cursor is
    resumable, so a round that resumed part way through has to partition what is LEFT.
    Computing the partition from offset zero and slicing it would hand slot 0 a batch whose
    rows are already journalled, and re-journal them.

    **Every offset it returns is a planned batch boundary**, because the walk steps by
    :func:`_batch_size_at` rather than by a fixed stride. That is what keeps the resume
    contract where it was: a wave commits ``sum(size)`` units and therefore lands on a
    boundary the batch plan declared, so ``active_validation`` keeps its
    ``{round_index, cursor, rows}`` schema and no second, differently shaped contract exists
    for the sharded phase.

    At ``world_size == 1`` a wave is exactly one batch, so the caller's loop degenerates to
    the per-batch loop -- one committed checkpoint per batch, same cursor sequence.
    """
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    wave: list[tuple[int, int]] = []
    offset = cursor
    while len(wave) < world_size and offset < total:
        size = _batch_size_at(validation_offsets, offset, "validation")
        wave.append((offset, size))
        offset += size
    return tuple(wave)


def _validate_batch_boundary(
    cursor: int, offsets: Mapping[int, int], total: int, label: str
) -> None:
    """Fail closed unless a persisted cursor sits on a planned batch boundary.

    ``offsets`` is the atomic-step map: generation batches without accumulation, and
    accumulation groups with it.
    """
    if cursor != total and cursor not in offsets:
        raise ValueError(
            f"checkpoint {label} cursor {cursor} does not sit on a planned batch boundary"
        )


def resolve_gradient_accumulation(
    schedule: SteeringSchedule,
    batch_plan: SteeringBatchPlan | None,
    accumulation_units: int,
    *,
    world_size: int = DEFAULT_WORLD_SIZE,
    step_units: int = UNIFORM_OPTIMIZER_STEP_UNITS,
) -> GradientAccumulationPlan:
    """Return the optimizer-step grouping this batch plan implies, at uniform step units.

    Deterministic and side-effect free, so the training loop, the run report, and the
    evaluation identity check all derive the same plan instead of passing one around.
    Groups never cross a training cycle boundary: a validation round is due there, and the
    cursor has to be able to stop on it.

    **``accumulation_units`` and ``world_size`` no longer set the group.** An optimizer step
    is :data:`UNIFORM_OPTIMIZER_STEP_UNITS` scheduled units for every steering property, and
    how many generation batches that takes is a memory question answered inside the step. The
    two arguments are still accepted and validated -- every caller resolves them from a config
    that may state them -- and a declared value other than 1 is warned about rather than
    silently honoured, because a run that asked for ``K = 6`` gets a step of 32 units either
    way and must not be re-keyed for a knob that no longer determines anything.

    That also removes ``world_size`` from the optimization entirely, where it used to multiply
    the effective batch: the ``W`` ranks now split ONE step's micro-batches between them, so a
    four-GPU run and a one-GPU run take the identical optimizer step and differ only in how
    long it takes.
    """
    if not isinstance(schedule, SteeringSchedule):
        raise TypeError("schedule must be a SteeringSchedule")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if (
        isinstance(accumulation_units, bool)
        or not isinstance(accumulation_units, int)
        or accumulation_units < 1
    ):
        raise ValueError("gradient_accumulation_units must be a positive integer")
    if accumulation_units * world_size > 1:
        _LOGGER.warning(
            "gradient_accumulation_units=%d and distributed_world_size=%d are superseded: one "
            "optimizer step holds %d scheduled units whatever they say, and the micro-batch "
            "count follows from the length heuristic inside that step",
            accumulation_units,
            world_size,
            step_units,
        )
    train_sizes = (1,) * len(schedule.train) if batch_plan is None else batch_plan.train_batch_sizes
    spans = plan_optimizer_step_spans(
        schedule.train,
        step_units=step_units,
        segments=cycle_segments(schedule.train) if schedule.train else None,
    )
    return plan_uniform_optimizer_steps(
        train_sizes,
        tuple(stop - start for start, stop in spans),
        step_units=step_units,
    )


def _accumulate(total: Tensor, micro_units: int, group_units: int) -> None:
    """Add one micro-step's share of the group's gradient to the shared buffer.

    ``total`` is a mean over the micro-batch's own ``micro_units`` units, so it enters the
    accumulated gradient weighted by ``micro_units / group_units``. That makes the
    accumulated gradient equal to the gradient of the mean over all ``group_units``
    members, for any mixture of micro-batch sizes. Weighting every micro-step equally
    instead would inflate whichever sequences landed in the smaller micro-batches.

    The weight is applied only when it is not exactly 1, so a group holding a single
    micro-batch -- which is every group at ``gradient_accumulation_units: 1`` -- calls
    ``backward()`` on the untouched loss tensor and stays bit-identical to the
    pre-accumulation path.

    The counts are **scheduled units**, not generated structures, and that stays correct
    under the ensemble estimator (D16): ``M`` structures are collapsed into one observation
    per unit before the loss, so a micro-batch of ``n`` units is still a mean over ``n``
    observations whatever ``M`` is. Counting structures here would be wrong twice over --
    it would divide by ``M`` a loss that was never summed over ``M``, and with ``M``
    constant it would cancel anyway.
    """
    weight = _micro_step_weight(micro_units, group_units)
    (total if weight == 1.0 else total * weight).backward()


def _micro_step_weight(micro_units: int, group_units: int) -> float:
    """Return the weight a micro-batch's mean loss carries in its accumulation group."""
    if (
        isinstance(micro_units, bool)
        or isinstance(group_units, bool)
        or not isinstance(micro_units, int)
        or not isinstance(group_units, int)
        or micro_units < 1
        or group_units < micro_units
    ):
        raise ValueError("micro-step unit counts must be positive and fit their group")
    return micro_units / group_units


def _fixed_generation_seeds(
    schedule: SteeringSchedule, config: OrdinarySteeringConfig
) -> dict[str, int] | None:
    """Return the per-sequence diagnostic generation seed map, or ``None`` when off.

    Each training sequence is pinned to the seed of its *first scheduled train unit*, so
    the map is a deterministic function of the schedule and appears in the checkpoint
    identity. Only the train split is pinned: validation is a selection signal and
    evaluation is the scientific product, and both keep their own matched per-unit seeds.
    """
    if not config.diagnostic_fixed_generation_seed:
        return None
    seeds: dict[str, int] = {}
    for unit in schedule.train:
        seeds.setdefault(unit.sequence_id, unit.seed)
    return dict(sorted(seeds.items()))


def _generation_units(
    members: Sequence[SteeringUnit], generation_seeds: Mapping[str, int] | None
) -> tuple[SteeringUnit, ...]:
    """Return the units handed to the generator, with the diagnostic seed substituted.

    The scheduled units themselves are never rewritten: records, failures, and the journal
    keep the schedule's own seeds, so the committed prefix still equals the schedule and a
    fixed-seed run remains auditable against it.
    """
    members = tuple(members)
    if generation_seeds is None:
        return members
    return tuple(
        replace(unit, seed=generation_seeds[unit.sequence_id])
        if unit.split == "train" and unit.sequence_id in generation_seeds
        else unit
        for unit in members
    )


def _generated(
    generate: GenerateCallback,
    generate_batch: GenerateBatchCallback | None,
    members: Sequence[SteeringUnit],
    conditioned_targets: Sequence[StandardizedAbsoluteTarget | RequestedLevel],
) -> object:
    """Invoke the configured generator for one batch of scheduled units."""
    if generate_batch is None:
        if len(members) != 1:  # pragma: no cover - guarded by _validate_inputs.
            raise ValueError("a multi-unit batch requires a batched generator")
        return generate(members[0], conditioned_targets[0])
    return generate_batch(tuple(members), tuple(conditioned_targets))


def _generate_property_ensemble(
    generate: GenerateCallback,
    generate_batch: GenerateBatchCallback | None,
    members: Sequence[SteeringUnit],
    conditioned_targets: Sequence[StandardizedAbsoluteTarget | RequestedLevel],
    ensemble_seeds: Mapping[int, tuple[int, ...]] | None,
    resample: tuple[str, int] | None = None,
) -> GeneratedTrainingSample:
    """Generate this micro-batch's structures and reduce them to one value per unit.

    ``ensemble_seeds is None`` is the pre-ensemble path exactly: seed once from the first
    member's seed and call the generator once, on the scheduled units themselves.

    Otherwise each unit generates ``M`` structures, one per derived member seed, and the
    ``M`` batches are reduced by :func:`_ensemble_sample`. The generator is called once per
    ensemble member with the whole micro-batch, not once per (unit, member) pair, so
    generation batching (D8) still applies to every member and D8's per-member seeded
    priors keep working: each batch member carries its own seed for that draw, exactly as
    it carries its scheduled seed when ``M = 1``.

    The scheduled units are never rewritten -- only the copies handed to the generator
    carry the derived seeds -- so the journal, the failures, and the committed prefix keep
    the schedule's own seeds and stay auditable against it, exactly as the fixed-seed
    diagnostic does.

    ``resample`` is ``(attempt_key, resample_index)`` and is applied LAST, after the ensemble
    member seed has been substituted, so the substitute is derived from the draw this attempt would
    otherwise have made rather than from the scheduled one. At index 0 -- every first attempt, and
    therefore every draw of a healthy run -- :func:`_resampled_units` returns its input unchanged
    and this function is byte-identical to the version before resampling existed.
    """
    members = tuple(members)
    attempt_key, resample_index = resample if resample is not None else ("", 0)
    if ensemble_seeds is None:
        members = _resampled_units(members, attempt_key, resample_index)
        seed_everything(members[0].generation_seed)
        return _sample(_generated(generate, generate_batch, members, conditioned_targets))
    samples = []
    for index in range(_ensemble_size(members, ensemble_seeds)):
        # The member seed replaces the *draw*, never the identity: the journal, the failures
        # and the committed prefix keep the schedule's own seeds and stay auditable.
        drawn = tuple(replace(unit, draw_seed=ensemble_seeds[unit.seed][index]) for unit in members)
        # After the table, deliberately. The M ensemble members already carry M distinct draws, so
        # deriving each substitute from the member seed keeps them distinct under resampling too;
        # deriving from the scheduled seed instead would collapse all M onto one draw and silently
        # turn an M-structure ensemble estimator into a single-structure one on every retry.
        drawn = _resampled_units(drawn, attempt_key, resample_index)
        seed_everything(drawn[0].generation_seed)
        samples.append(_sample(_generated(generate, generate_batch, drawn, conditioned_targets)))
    return _ensemble_sample(tuple(samples))


def _ensemble_size(
    members: Sequence[SteeringUnit], ensemble_seeds: Mapping[int, tuple[int, ...]]
) -> int:
    """Return the shared ensemble size of a micro-batch, failing closed on any mismatch."""
    sizes = set()
    for unit in members:
        seeds = ensemble_seeds.get(unit.seed)
        if seeds is None:
            raise ValueError(
                f"unit {unit.sequence_id}#{unit.seed} has no property-ensemble seeds; the "
                "seed table is derived from the schedule and must cover every unit it walks"
            )
        sizes.add(len(seeds))
    if len(sizes) != 1:
        raise ValueError("every unit of a batch must carry the same property-ensemble size")
    return sizes.pop()


def _ensemble_sample(samples: Sequence[GeneratedTrainingSample]) -> GeneratedTrainingSample:
    """Reduce ``M`` generated structures to one differentiable observation per unit (D16).

    **The property is averaged before the loss sees it, and that is the entire point.** The
    returned prediction is a differentiable mean of ``M`` differentiable property values,
    so one loss is applied to ``mean_m(Rg_m)`` and the gradient is

        2 * (mean_m(Rg_m) - t) / sigma^2 * mean_m(dRg_m/dtheta),

    one averaged residual multiplying one averaged Jacobian. Building ``M`` separate losses
    and averaging them instead gives ``mean_m[2 * (Rg_m - t) * dRg_m/dtheta] / sigma^2``,
    where each draw's Jacobian is scaled by *its own* residual -- the quantity dominated by
    the stochastic structure draw, and exactly the estimator that gradient accumulation
    already implements and that this replaces. The two agree only when every ``Rg_m`` is
    equal, so they cannot be swapped as an optimization.
    """
    samples = tuple(samples)
    if not samples:  # pragma: no cover - the caller always generates at least one member.
        raise ValueError("an ensemble must contain at least one generated structure")
    if len(samples) == 1:
        return samples[0]
    predictions = tuple(item.surrogate_prediction for item in samples)
    if any(value.shape != predictions[0].shape for value in predictions[1:]):
        raise ValueError("every ensemble member must return the same property shape")
    raw_fractions = tuple(item.raw_residual_fraction for item in samples)
    # The rigidity term is a HINGE on a dimensional RMS, and no reduction over members
    # commutes with it: mean_m hinge(rms_m) and hinge(mean_m rms_m) differ whenever any
    # member sits on the other side of the setpoint, which is precisely the interesting case.
    # D16 decided the property and residual reductions explicitly; it has not decided this
    # one, so refuse rather than pick. Silently choosing would change what is optimised and
    # leave no trace in the identity.
    if any(
        item.context_rms_angstrom is not None or item.context_setpoint_angstrom is not None
        for item in samples
    ):
        raise ValueError(
            "property_ensemble_samples > 1 with the context-rigidity term is not defined: the "
            "hinge does not commute with the ensemble reduction, and D16 has no decision for "
            "it. Run the off-path arms at M = 1 until one is recorded"
        )
    # Same refusal, same reason, for the D-35b monitor counts. A count over ``M`` draws is not
    # one count: summing, averaging and taking the maximum are three different monitors, and
    # the column is an INTEGER count of a localised breach precisely so that no denominator
    # dilutes it. Picking a reduction here would silently redefine what the column reports, so
    # this refuses instead. Unreachable for radius of gyration and Local-Q, which carry none of
    # these fields, so the ``M > 1`` runs that exist are untouched.
    if any(
        any(getattr(item, name) is not None for name in sorted(_MONITOR_RECORD_KEYS))
        for item in samples
    ):
        raise ValueError(
            "property_ensemble_samples > 1 with the pocket-local geometry monitor is not "
            "defined: an integer breach count has no decided reduction over the M draws of an "
            "ensemble, and choosing one would change what the journalled column means. Run a "
            "cryptic-pocket arm at M = 1 until a reduction is recorded"
        )
    # And the two-state target's origin. Reachable independently of the monitor guard above --
    # the kinase programme is a two-state arm with no cryptic monitor columns -- and refused for
    # a sharper reason than "undecided": each of the M draws has its OWN matched control and
    # therefore its own origin, so ``t`` is not one number for the unit at all, and averaging M
    # origins would compare an averaged prediction against a target no draw was given. D16
    # decided the property and residual reductions and has no decision for a per-draw target.
    if any(item.control_reporting_value is not None for item in samples):
        raise ValueError(
            "property_ensemble_samples > 1 with a two-state target is not defined: the target is "
            "t = the progress coordinate_state(o) - the progress coordinate_reporting(control) and each of the M draws carries its own "
            "control, so the unit has M targets rather than one. Run a two-state arm at M = 1 "
            "until a reduction is recorded"
        )
    pocket = tuple(item.pocket_off_path_squared for item in samples)
    monitored = tuple(item.pocket_off_axis_ratio_squared for item in samples)
    return GeneratedTrainingSample(
        torch.stack(predictions, dim=0).mean(dim=0),
        _ensemble_residual(tuple(item.residual_rms for item in samples)),
        _ensemble_residual(tuple(item.residual_fraction for item in samples)),
        torch.stack([item.geometry_loss for item in samples], dim=0).mean(dim=0),
        # Same D16 reduction as the realised fraction: the penalty is a spend per structure,
        # so members are squared then averaged rather than averaged then squared.
        None
        if any(value is None for value in raw_fractions)
        else _ensemble_residual(raw_fractions),
        # ``local_structure`` is not carried across the ensemble and never was: this
        # reduction has dropped it since it was written, so passing None here is the
        # behaviour every ensemble run already had. Restoring it is a separate change with
        # its own D16 argument to make, and is deliberately not smuggled in here.
        None,
        # ``zeta_pocket^2`` is already squared and the loss means over observations, so the
        # mean over members composes to mean_{n,m} exactly as ``_ensemble_residual``'s
        # identity does for the residual. This one IS decided by D16's own argument.
        None if any(value is None for value in pocket) else torch.stack(pocket, dim=0).mean(dim=0),
        # Unreachable with a live rigidity term, which the guard above refuses outright. Spelled
        # out rather than left to the field default so the ensemble reduction says what it does
        # with every off-path field it is handed.
        None,
        # The four D-35b monitor counts, likewise unreachable: the guard above refuses an
        # ensemble that carries any of them rather than reducing them by an undecided rule.
        None,
        None,
        None,
        None,
        # The two-state target's origin, unreachable for the same reason and spelled out rather
        # than left to the field default, so this reduction states what it does with every field.
        None,
        # The UNPRICED zeta_pocket^2 monitor, reduced by exactly the rule its priced twin is: it
        # is already squared and the journal means over observations, so the mean over members
        # composes to mean_{n,m}. Named rather than positional so that adding a field to the
        # sample cannot silently shift what this reduction writes, and REDUCED rather than
        # dropped: a monitor that vanished at M > 1 would read as "this arm's alignment was not
        # measured" for a reason that is an implementation detail of the reducer.
        pocket_off_axis_ratio_squared=(
            None
            if any(value is None for value in monitored)
            else torch.stack(monitored, dim=0).mean(dim=0)
        ),
    )


def _ensemble_residual(values: Sequence[Tensor]) -> Tensor:
    """Reduce ``M`` members' residual magnitudes to the per-unit RMS over the ensemble.

    D16 decides this deliberately, and the **opposite** way from the property term. The
    property error is a signed error against a target whose estimand is the sequence's own
    ensemble mean (D3), so averaging before squaring compares like with like. A residual
    magnitude has no target: every generated structure really did receive its own
    intervention, so every structure is an observation of what was spent, and the penalty
    stays ``mean(F^2)`` over all ``M x n`` of them exactly as D13 defines it. Passing
    ``RMS_m(F)`` per unit into :class:`~sae_bioemu.steering.training.SteeringLoss`, which
    squares and means over units, is that identity written so the loss class needs no
    ensemble-aware shape:

        mean_n ( RMS_m(F_n,m) )^2 = mean_n mean_m F_n,m^2 = mean_{n,m} F_n,m^2

    Averaging ``F`` first would instead penalize ``(E F)^2``, leaving the across-draw
    variance of the intervention free (``(E F)^2 <= E F^2``) and reporting a spend that no
    structure actually received.
    """
    values = tuple(values)
    if any(value.shape != values[0].shape for value in values[1:]):
        raise ValueError("every ensemble member must return the same residual shape")
    mean_square = torch.stack(values, dim=0).square().mean(dim=0)
    # ``sqrt`` has an infinite derivative at zero and a residual may legitimately be
    # exactly zero, so zero entries are routed around it rather than clamped: the value is
    # unchanged and the gradient there is zero, which is what the composed penalty
    # ``mean_m F^2`` has at ``F = 0`` anyway.
    positive = mean_square > 0
    guarded = torch.where(positive, mean_square, torch.ones_like(mean_square))
    return torch.where(positive, guarded.sqrt(), torch.zeros_like(mean_square))


def paired_dose_coverage(
    schedule: SteeringSchedule,
    offsets: Mapping[int, int],
    groups_by_offset: Mapping[int, Sequence[int]],
) -> dict[str, int | float]:
    """Report how many dose pairs the batch plan actually lets the paired term form.

    ``_pair_groups`` is opportunistic: it pairs whatever co-occurs in a micro-batch and
    contributes exactly zero for anything left alone. That is correct but **silent**, and the
    amount of pairing is a property of the batch plan, not of the objective -- a change to
    ``maximum_batch_size`` or ``gradient_graph_divisor`` can zero the term out without any
    error. A near-identical silent failure has already cost this project an arm: an earlier
    run believed it was pairing and reached only 21% of its batches.

    So the coverage is computed up front from the schedule and the plan, and reported. The
    denominator is the pairs a perfect plan would form -- every unit sharing a draw co-located
    -- and the numerator is what this plan actually co-locates.
    """
    by_draw: dict[int, set[float]] = {}
    for unit in schedule.train:
        if unit.draw_seed is not None:
            by_draw.setdefault(unit.draw_seed, set()).add(float(unit.level))
    possible = sum(len(levels) * (len(levels) - 1) // 2 for levels in by_draw.values())

    realised = 0
    for cursor in sorted(offsets):
        members = schedule.train[cursor : cursor + offsets[cursor]]
        start = 0
        for micro in groups_by_offset[cursor]:
            chunk = members[start : start + micro]
            start += micro
            levels: dict[int, set[float]] = {}
            for unit in chunk:
                if unit.draw_seed is not None:
                    levels.setdefault(unit.draw_seed, set()).add(float(unit.level))
            realised += sum(len(v) * (len(v) - 1) // 2 for v in levels.values())
    return {
        "draws": len(by_draw),
        "pairs_possible": possible,
        "pairs_realised": realised,
        "fraction": (realised / possible) if possible else 0.0,
    }


def require_pair_coverage(
    objective: SteeringLoss,
    config: OrdinarySteeringConfig,
    coverage: Mapping[str, int | float],
) -> float:
    """Refuse a batch plan that cannot form the pairs the paired term is weighted for.

    This used to log and continue. It was written when ``lambda_c`` was 1 and the paired term
    was an amplification of a channel that already worked; ``lambda_c`` is now 100 times
    ``lambda_y`` on the two-state arms, so zero pairing silently deletes about 99% of the
    intended objective and leaves a run that trains, reports and looks complete. The project
    has already discarded one arm that believed it was pairing and reached 21%.

    ``config.allow_partial_pair_coverage`` is the opt-out and is recorded in the checkpoint
    identity, so a run that took it cannot be read back as one that did not. The achieved
    fraction is reported in every case, including the one that raises: it is the number that
    says how much of the objective the plan actually delivers.

    Returns the achieved fraction so a caller can record it.
    """
    fraction = float(coverage["fraction"])
    if objective.lambda_c <= 0.0:
        return fraction
    realised = int(coverage["pairs_realised"])
    possible = int(coverage["pairs_possible"])
    summary = (
        f"the paired dose term will see {100 * fraction:.1f}% of its pairs "
        f"({realised} of {possible}) under this batch plan"
    )
    if fraction >= 1.0:
        _LOGGER.info("%s at lambda_c=%s", summary, objective.lambda_c)
        return fraction
    if config.allow_partial_pair_coverage:
        _LOGGER.warning(
            "%s at lambda_c=%s; allow_partial_pair_coverage is set, so this run is training a "
            "weaker paired term than its weights state and records that in its identity",
            summary,
            objective.lambda_c,
        )
        return fraction
    raise ValueError(
        f"{summary}, but lambda_c={objective.lambda_c} weights it as though every pair were "
        "formed. Raise maximum_batch_size or lower gradient_graph_divisor so a sequence's "
        "levels are co-located, or set allow_partial_pair_coverage to accept the weaker "
        "objective and record it in the checkpoint identity"
    )


def _pair_groups(members: Sequence[SteeringUnit], prediction: torch.Tensor) -> torch.Tensor | None:
    """Mark which members of a micro-batch share a noise draw, for the paired dose term.

    Two units are paired when they carry the same ``draw_seed`` -- which the schedule assigns
    per ``(sequence_id, cycle)`` -- and different levels, so their generations differ only in
    the requested dose and the shared baseline cancels in their difference.

    Pairing is opportunistic: whatever happens to co-occur in this micro-batch is paired, and
    a batch with nothing to pair returns a grouping that contributes exactly zero. That keeps
    the term correct under any batching or bucketing rather than depending on the planner to
    co-locate levels, which it makes no promise to do.

    ``None`` when no unit carries a shared draw at all, so a schedule built before ``draw_seed``
    existed takes the original code path untouched.
    """
    draws = [unit.draw_seed for unit in members]
    if all(draw is None for draw in draws):
        return None
    assignment: dict[int, int] = {}
    groups = []
    for draw in draws:
        if draw is None:
            groups.append(-1)
            continue
        groups.append(assignment.setdefault(draw, len(assignment)))
    # A group of one contributes nothing, and so does a group whose members share a level:
    # differencing two identical requests measures noise, not dose.
    levels_by_group: dict[int, set[float]] = {}
    for group, unit in zip(groups, members, strict=True):
        if group >= 0:
            levels_by_group.setdefault(group, set()).add(float(unit.level))
    resolved = [
        group if group >= 0 and len(levels_by_group.get(group, ())) > 1 else -1 for group in groups
    ]
    if all(group < 0 for group in resolved):
        return None
    return torch.tensor(resolved, dtype=torch.int64, device=prediction.device).reshape(
        prediction.shape
    )


def _loss_inputs(
    sample: GeneratedTrainingSample,
    members: Sequence[SteeringUnit],
    targets: Sequence[float],
    config: OrdinarySteeringConfig,
) -> tuple[Tensor, Tensor | None]:
    """Return the per-observation target tensor and optional sigma normalization.

    ``targets`` is what :func:`_unit_target` produced for each member, which on the two-state
    path is the reporting-frame STATE rather than the finished target. **This is where the target
    becomes PER OBSERVATION.** The prediction is a training-frame displacement from a matched
    unsteered draw, so the state ask is converted by subtracting THAT DRAW's own reporting-frame
    ``the progress coordinate``, which arrives on the generated sample. ``the progress coordinate`` is linear in the pairwise distance
    vector, so ``the progress coordinate_training(s) = the progress coordinate_reporting(s) - the progress coordinate_reporting(control)`` holds exactly and the
    subtraction is a change of frame rather than an approximation of one. Two members of one
    batch that share a level do NOT share a target, and that is the point: the reference table's
    per-protein ``mean_the progress coordinate`` would make them share one, which is the D-46 defect.

    The scale is ``None`` on the absolute path only. **The two-state path carries a sigma again
    since D-65, and it is not the sigma normalisation radius of gyration uses.** The scale
    returned here reaches the objective's ABSOLUTE channel and nothing else: under the state
    target the matched control cancels out of the absolute residual, which therefore carries the
    draw's whole baseline and gets inverse-variance weighting, while the paired gap cancels it
    identically and is left undivided by ``SteeringLoss`` itself. One tensor, two channels, and
    the split is made from the declared normalisation rather than here.
    """
    prediction = sample.surrogate_prediction
    # Read off the one declaration the objective is also checked against, so "a scale arrives"
    # and "the objective says it is sigma-normalised" cannot be two different answers. The same
    # declaration -- and not a property name, an arm, or the presence of a tensor -- is what
    # selects the two-state conversion below.
    normalization = config.property_loss_normalization
    normalizes_by_sigma = normalization in (
        SIGMA_LOSS_NORMALIZATION,
        PER_CHANNEL_LOSS_NORMALIZATION,
    )
    if normalization in _TWO_STATE_LOSS_NORMALIZATIONS:
        if prediction.shape != (len(members),):
            raise ValueError(
                "a two-state target is per observation, so the property callback must return "
                f"exactly one value per batch member; expected shape {(len(members),)}, got "
                f"{tuple(prediction.shape)}"
            )
        origin = _control_reporting_tensor(sample, len(members))
        scale = (
            prediction.new_tensor([_unit_sigma(unit, config) for unit in members])
            if normalizes_by_sigma
            else None
        )
        # ``t = the progress coordinate_state(o) - the progress coordinate_reporting(control)``, one target per observation.
        return prediction.new_tensor(list(targets)) - origin.to(prediction), scale
    if len(members) == 1:
        # Unchanged from the unbatched path: one scalar target broadcast over whatever
        # shape the property callback returns, which may be per residue.
        target_tensor = prediction.new_full(prediction.shape, targets[0])
        scale = (
            prediction.new_full(prediction.shape, _unit_sigma(members[0], config))
            if normalizes_by_sigma
            else None
        )
        return target_tensor, scale
    if prediction.shape != (len(members),):
        raise ValueError(
            "a batched generation must return exactly one property value per batch "
            f"member; expected shape {(len(members),)}, got {tuple(prediction.shape)}"
        )
    target_tensor = prediction.new_tensor(list(targets))
    scale = (
        prediction.new_tensor([_unit_sigma(unit, config) for unit in members])
        if normalizes_by_sigma
        else None
    )
    return target_tensor, scale


def _control_reporting_tensor(sample: GeneratedTrainingSample, count: int) -> Tensor:
    """Return the matched control's reporting-frame ``the progress coordinate`` per batch member, or fail closed.

    Detached here as well as by the producer. It is the target's ORIGIN: a gradient reaching it
    would let the loss fall by moving the goalposts rather than the structure, and the two-state
    target is the one place in this module where a tensor that is not the prediction enters the
    quantity the objective is asked to match.
    """
    values = sample.control_reporting_value
    if values is None:
        raise ValueError(
            "the two-state target is t = the progress coordinate_state(o) - the progress coordinate_reporting(control), so a run on the "
            f"{TRANSITION_LOSS_NORMALIZATION!r} normalisation must carry this draw's matched "
            "control in the REPORTING frame on control_reporting_value. There is deliberately no "
            "fallback to the reference table's mean_the progress coordinate: that is a per-protein constant and is the "
            "D-46 defect this origin replaces"
        )
    if values.shape != (count,):
        raise ValueError(
            "control_reporting_value must carry one value per batch member; expected shape "
            f"{(count,)}, got {tuple(values.shape)}"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("control_reporting_value must be finite")
    return values.detach()


def _member_scalars(values: Tensor, count: int) -> tuple[float, ...]:
    """Return one scalar per batch member, reducing the single-member case as before."""
    if count == 1:
        return (_scalar(values.mean()),)
    if values.shape != (count,):
        raise ValueError(
            f"a batched generation must return one value per member; expected shape "
            f"{(count,)}, got {tuple(values.shape)}"
        )
    return tuple(_scalar(values[index]) for index in range(count))


def _monitor_count(value: float, name: str) -> int:
    """Return one D-35b monitor column as the integer count it is, or fail closed.

    The counts travel from the property callback as a float tensor per batch member, because
    that is the only shape ``GeneratedTrainingSample`` carries. They are integers, and a value
    that is not a nonnegative whole number means something upstream reduced or interpolated
    them -- which would make ``1`` clash read as ``0.5`` and defeat the entire reason D-35b
    counts rather than averages.
    """
    resolved = _finite_nonnegative(value, f"pocket geometry monitor {name}")
    if resolved != int(resolved):
        raise ValueError(f"pocket geometry monitor {name} must be a whole count, got {resolved}")
    return int(resolved)


def _control_reporting_rows(
    sample: GeneratedTrainingSample, count: int, config: OrdinarySteeringConfig
) -> tuple[float | None, ...]:
    """Return each member's target origin, or ``None`` per member off the two-state path.

    Gated on ``config.targets_two_state`` -- the run's own identity-bearing declaration --
    and not on whether the sample happens to carry the field, for the reason
    :func:`_monitor_rows` is gated the same way: the persisted key set has to be a function of
    the configuration, so a resumed journal's schema is decided by the same thing that decided
    the schema it is appended to.
    """
    if not config.targets_two_state:
        return tuple(None for _ in range(count))
    return _member_scalars(_control_reporting_tensor(sample, count), count)


def _monitor_rows(
    sample: GeneratedTrainingSample, count: int, config: OrdinarySteeringConfig
) -> tuple[dict[str, int], ...]:
    """Return the D-35b monitor columns for each batch member, or an empty dict per member.

    Gated on ``config.steers_cryptic_pocket`` -- the run's own identity-bearing configuration --
    and not on whether the sample happens to carry the fields. That direction matters: the
    persisted key set has to be a function of the configuration, so that a resumed journal's
    schema is decided by the same thing that decided the schema it is being appended to. A
    cryptic arm whose sample is missing a column is therefore a hard failure here rather than a
    row that quietly narrows.

    Empty for every other property, which is what keeps a radius-of-gyration row byte-identical:
    the caller spreads this mapping into the record, and spreading ``{}`` adds nothing.
    """
    if not config.steers_cryptic_pocket:
        return tuple({} for _ in range(count))
    resolved: dict[str, tuple[float, ...]] = {}
    for name in sorted(_MONITOR_RECORD_KEYS):
        values = getattr(sample, name)
        if values is None:
            raise ValueError(
                f"a cryptic-pocket run must carry {name}: the pocket-local geometry monitor "
                "columns are gated on the arm rather than on a weight, so a missing one is a "
                "journal row whose key set no longer matches its configuration"
            )
        resolved[name] = _member_scalars(values, count)
    return tuple(
        {name: _monitor_count(values[index], name) for name, values in resolved.items()}
        for index in range(count)
    )


def _declared_training_columns(
    config: OrdinarySteeringConfig, objective: SteeringLoss | None
) -> dict[str, JournalledSampleColumn]:
    """Return the UNPRICED MONITOR columns this run's property declares for its TRAINING rows.

    The training-row twin of :func:`_declared_validation_columns`, and it exists for a hole that
    hook could not reach. A column in ``PropertyRegistration.journal_columns`` is gated on the
    WEIGHT that prices it, which is right for a loss term and wrong for a measurement: the cryptic
    direct-metric arm runs at ``lambda_perp = 0``, so ``perp_loss`` is never computed and the one
    arm D-80 section 4 names as able to move the realised alignment off D-76's ``|cos| = 0.165``
    journalled nothing that could say whether it had.

    UNREGISTERED IS EMPTY, for the reason :func:`_declared_validation_columns` states: this is the
    shared ordinary loop and the AF3, Local-Q and surrogate-validator paths drive it with property
    names the catalog does not carry.
    """
    if config.property_name not in registered_property_names():
        return {}
    descriptor = _resolve_property(config.property_name)
    declared = dict(descriptor.training_row_columns(objective=objective, config=config))
    unknown = sorted(set(declared) - set(descriptor.training_row_vocabulary))
    if unknown:
        raise ValueError(
            f"{config.property_name!r} declared the training monitor columns {unknown}, which are "
            "not in its own training_row_vocabulary"
        )
    outside = sorted(set(declared) - _DECLARED_MONITOR_RECORD_KEYS)
    if outside:
        raise ValueError(
            f"{config.property_name!r} declared the training monitor columns {outside}, which its "
            "PropertyRegistration.monitor_columns does not name. The persisted journal schema "
            "admits exactly that list as surplus, so these columns would be written and then "
            "refused by the gate the run itself has to pass"
        )
    return declared


def _declared_monitor_rows(
    sample: GeneratedTrainingSample,
    count: int,
    config: OrdinarySteeringConfig,
    objective: SteeringLoss | None,
) -> tuple[dict[str, float], ...]:
    """Return each batch member's declared monitor values, or an empty dict per member.

    A declared column whose sample field is missing is a HARD FAILURE rather than a narrowed row,
    for the reason :func:`_monitor_rows` gives: the persisted key set has to be a function of the
    configuration, so that a resumed journal's schema is decided by the same thing that decided
    the schema it is appended to.

    Empty for every property that declares none, which is what keeps a radius-of-gyration,
    kinase or domain-motion row byte-identical: the caller spreads this mapping into the record,
    and spreading ``{}`` adds nothing.
    """
    declared = _declared_training_columns(config, objective)
    if not declared:
        return tuple({} for _ in range(count))
    resolved: dict[str, tuple[float, ...]] = {}
    for name, column in sorted(declared.items()):
        values = getattr(sample, column.field, None)
        if values is None:
            raise ValueError(
                f"{config.property_name!r} declares the training monitor column {name!r}, which "
                f"reads {column.field!r} off the generated sample, and this sample carries none. "
                "A monitor is gated on the property's own declaration rather than on a weight, so "
                "a missing field is a journal row whose key set no longer matches its "
                "configuration"
            )
        if column.component is not None:
            values = values[..., column.component]
        resolved[name] = _member_scalars(values, count)
    return tuple(
        {name: values[index] for name, values in resolved.items()} for index in range(count)
    )


def _declared_validation_columns(
    config: OrdinarySteeringConfig, objective: SteeringLoss | None
) -> dict[str, JournalledSampleColumn]:
    """Return the extra VALIDATION-row columns this run's property declares, by column name.

    **This replaced a hardcoded ``config.steers_cryptic_pocket``, and the replacement is the
    single highest-value diagnostic change in the registry pass.** Under that predicate the
    cryptic pocket got its four D-35b geometry counts on every held-out draw and every other
    property got nothing -- so no kinase arm has ever been observed on held-out ``d_K`` (K-21
    4.1, K-22 caveat 2), and the arm that minimised its objective (t = -5.38) while its
    ``|signed_error|`` ROSE (t = +3.49) had nothing recorded that could say which of the two had
    moved. A property now answers for its own diagnostic columns in its own package.

    The objective is passed because a column may be gated on a weight -- ``d_K`` exists on the
    sample only where ``lambda_dihedral`` prices it -- and the CONFIG is passed because a column
    may instead be gated on the arm, which is how the geometry monitor is gated: it is not in the
    objective at all, so there is no weight for it to ride.
    """
    # UNREGISTERED IS EMPTY, and that is not fail-open: it is what keeps this runner usable by
    # the properties that have no descriptor at all. ``run_ordinary_training`` is the shared
    # ordinary loop, and the AF3, Local-Q and surrogate-validator paths drive it with property
    # names the steerable-property catalog does not (and should not) carry. A property with no
    # registration declares no diagnostic columns, which is exactly the row it wrote before this
    # hook existed. Every other failure inside ``resolve`` -- a descriptor that is not a
    # SteerableProperty, one that answers for the wrong name, one that is not hashed by its own
    # property -- still propagates, so a REGISTERED property cannot silently lose its columns.
    if config.property_name not in registered_property_names():
        return {}
    descriptor = _resolve_property(config.property_name)
    declared = dict(descriptor.validation_row_columns(objective=objective, config=config))
    unknown = sorted(set(declared) - set(descriptor.validation_row_vocabulary))
    if unknown:
        raise ValueError(
            f"{config.property_name!r} declared the validation columns {unknown}, which are not "
            "in its own validation_row_vocabulary. The readers that validate a persisted "
            "checkpoint admit exactly that vocabulary, so a column outside it would be written "
            "and then refused by the gate the run itself has to pass"
        )
    return declared


def _declared_validation_rows(
    sample: GeneratedTrainingSample,
    count: int,
    config: OrdinarySteeringConfig,
    objective: SteeringLoss | None,
) -> tuple[dict[str, float | int], ...]:
    """Return each batch member's declared validation columns, read off the generated sample.

    Gated on the run's own identity-bearing configuration and objective, never on whether the
    sample happens to carry a field. That direction matters for the reason
    :func:`_monitor_rows` states: the persisted key set has to be a function of the
    configuration, so a resumed journal's schema is decided by the same thing that decided the
    schema it is appended to. A run whose property declares a column and whose sample is missing
    it is therefore a hard failure here rather than a row that quietly narrows.

    Empty for every property that declares nothing, which is what keeps a radius-of-gyration row
    byte-identical: the caller spreads this mapping into the row, and spreading ``{}`` adds
    nothing.
    """
    declared = _declared_validation_columns(config, objective)
    if not declared:
        return tuple({} for _ in range(count))
    resolved: dict[str, tuple[float, ...]] = {}
    # Sorted, so the column a half-configured sample is reported against is a function of
    # the names rather than of a dict's insertion order.
    for name, column in sorted(declared.items()):
        values = getattr(sample, column.field, None)
        if values is None:
            raise ValueError(
                f"this run's property declares the validation column {name!r}, read from the "
                f"generated sample's {column.field!r}, and the sample does not carry it. The "
                "column is gated on the configuration rather than on the sample, so a missing "
                "one is a journal row whose key set no longer matches its configuration"
            )
        if column.component is not None:
            # A channel of width one is spelled ``[members]`` rather than ``[members, 1]``, which
            # is the shape every arm emitted before a second reading was stacked onto the wire and
            # the shape the objective still splits on. Component 0 of it IS the channel; any other
            # component of it is a column whose quantity this batch does not carry, and is refused
            # rather than read off the wrong axis.
            if values.ndim == 1:
                if column.component != 0:
                    raise ValueError(
                        f"the validation column {name!r} is component {column.component} of the "
                        f"{column.field!r} channel, and this batch's channel is a plain "
                        f"{tuple(values.shape)} carrying only component 0. The extra reading was "
                        "not computed for this batch, so the column has no value to record"
                    )
            elif int(values.shape[-1]) <= column.component:
                raise ValueError(
                    f"the validation column {name!r} is component {column.component} of the "
                    f"{column.field!r} channel, and this batch's channel is "
                    f"{tuple(values.shape)}. A channel that silently changed width would start "
                    "writing a different quantity under an unchanged column name"
                )
            else:
                values = values[..., column.component]
        resolved[name] = _member_scalars(values, count)
    return tuple(
        {
            name: (
                _monitor_count(values[index], name)
                if declared[name].integral
                else _finite(values[index], f"validation row {name}")
            )
            for name, values in resolved.items()
        }
        for index in range(count)
    )


def _training_record(
    unit: SteeringUnit,
    target: float,
    achieved: float,
    target_loss: float,
    residual_loss: float,
    geometry_loss: float,
    total_loss: float,
    residual_fraction: float,
    config: OrdinarySteeringConfig,
    off_path: Mapping[str, float | None] | None = None,
    *,
    monitor: Mapping[str, int] | None = None,
    declared_monitor: Mapping[str, float] | None = None,
    loss_terms: Mapping[str, float] | None = None,
    control_reporting_value: float | None = None,
) -> SteeringTrainingRecord:
    """Build one journal record in the schema its target construction defines.

    ``target`` is what :func:`_unit_target` produced, which on the two-state path is the
    reporting-frame STATE. ``control_reporting_value`` is then required, and the row's ``target``
    column is the training-frame ``state - origin`` the objective was actually given -- so the
    column named ``target`` holds the quantity ``signed_error`` is a difference against, and the
    state is recorded under its own name.

    ``off_path`` carries the project-declared columns -- ``perp_loss``, ``rigidity_loss``,
    ``dihedral_loss`` -- each ``None`` unless its weight is nonzero, which is what keeps every
    historical schema byte-identical. It is a mapping rather than a positional pair so that a
    project adding a term edits ``_OFF_PATH_RECORD_KEYS`` and nothing else; see
    :func:`_off_path_scalars`, which refuses a column that schema does not name.
    ``monitor`` is empty unless this run steers a cryptic pocket, for the same reason -- see
    ``_MONITOR_RECORD_KEYS``. ``loss_terms`` is empty unless the shared weights that price them
    are nonzero, for the same reason again -- see ``_LOSS_TERM_COLUMN_WEIGHTS``. It may not
    carry ``geometry_loss`` on the absolute path, where that column is already a mandatory
    positional argument; :func:`_loss_term_columns` is what guarantees it does not.
    """
    # ``None`` is the schema signal for "this term did not run", and it is spelled as ABSENCE
    # here: the record keeps only the columns that have a value, ``__getattr__`` answers ``None``
    # for the rest, and the serializer therefore cannot emit a null placeholder that would break
    # every historical journal's key set.
    off_path = {name: value for name, value in dict(off_path or {}).items() if value is not None}
    watched = dict(monitor or {})
    # D-86's registry-declared monitors travel in their OWN mapping rather than being spread into
    # the dataclass like ``watched``, which is what keeps a new monitor a catalog line instead of a
    # new field on a shared dataclass. ``SteeringTrainingRecord.__getattr__`` makes the two
    # indistinguishable to every reader.
    monitored = {
        name: value for name, value in dict(declared_monitor or {}).items() if value is not None
    }
    terms = dict(loss_terms or {})
    if not config.is_relative:
        if "geometry_loss" in terms:
            raise ValueError(
                "the absolute schema already records geometry_loss as a mandatory column, so "
                "offering it again as a gated loss term would write one quantity under two "
                "gates"
            )
        return SteeringTrainingRecord(
            unit=unit,
            target=target,
            target_loss=target_loss,
            residual_loss=residual_loss,
            geometry_loss=geometry_loss,
            total_loss=total_loss,
            residual_fraction=residual_fraction,
            project_columns=off_path,
            monitor_columns=monitored,
            **watched,
            **terms,
        )
    if config.targets_two_state:
        if control_reporting_value is None:
            raise ValueError(
                "a two-state journal row records the origin its target was converted through, so "
                "control_reporting_value is required; without it the row's target is a number no "
                "reader can re-derive and no resume can re-check"
            )
        observed = target - control_reporting_value
        return SteeringTrainingRecord(
            unit=unit,
            target_state=target,
            control_reporting_value=control_reporting_value,
            target=observed,
            residual_loss=residual_loss,
            total_loss=total_loss,
            residual_fraction=residual_fraction,
            achieved_value=achieved,
            signed_error=achieved - observed,
            sequence_sigma=_unit_sigma(unit, config),
            # The same column the sigma-relative row records, because since D-65 it is the same
            # construction: ``((achieved - target)/sigma)^2``. What makes this row two-state is
            # the pair of columns above, which say how ``target`` was built.
            target_loss_sigma_squared=target_loss,
            project_columns=off_path,
            monitor_columns=monitored,
            **watched,
            **terms,
        )
    return SteeringTrainingRecord(
        unit=unit,
        target=target,
        residual_loss=residual_loss,
        total_loss=total_loss,
        residual_fraction=residual_fraction,
        achieved_value=achieved,
        signed_error=achieved - target,
        sequence_sigma=_unit_sigma(unit, config),
        target_loss_sigma_squared=target_loss,
        project_columns=off_path,
        monitor_columns=monitored,
        **watched,
        **terms,
    )


def _batch_training_records(
    members: Sequence[SteeringUnit],
    targets: Sequence[float],
    sample: GeneratedTrainingSample,
    parts: Any,
    target_tensor: Tensor,
    property_scale: Tensor | None,
    objective: SteeringLoss,
    config: OrdinarySteeringConfig,
) -> tuple[SteeringTrainingRecord, ...]:
    """Return one committed record per batch member.

    A single-member batch records the batch loss components verbatim, so it is
    bit-identical to the unbatched path. A multi-member batch re-evaluates the same
    ``SteeringLoss`` on each member's own detached slice rather than reimplementing the
    decomposition, so the recorded per-member components always average back to the
    batch components that produced the gradient.

    Under gradient accumulation this is called once per micro-batch, on that micro-batch's
    own loss components. The accumulation weight is deliberately **not** applied: a record
    states what the objective was for that unit, not what share of an optimizer step it
    contributed, and scaling it would make the persisted loss depend on how the group
    happened to be cut.

    The gated loss-term columns are read off the BATCH components ``parts`` and repeated on
    every member's row, not off ``member_parts``. All three are whole-batch scalars: geometry
    and local structure are single values the generator produces per batch and the member
    re-evaluation is handed verbatim, and the paired dose term is a mean over unordered PAIRS
    of members, which a one-member slice cannot form at all -- re-deriving it per member would
    record 0.0 for a term that dominated the batch's total.

    **The batch's paired contribution is attributed to every member's ``total_loss`` AT FULL
    VALUE, exactly as geometry and local structure already are**, by handing the batch's
    ``contrastive`` back to the objective through ``paired_loss``. Three reasons, none of them
    a preference:

    * a one-member batch already records ``parts.total`` verbatim, paired term included, so any
      other attribution would make ``total_loss`` mean a different quantity at batch size 1 and
      at batch size 2;
    * it is what preserves this function's own contract, stated above -- the per-member
      components AVERAGE back to the batch components. The mean of a constant is that constant,
      so ``mean_i(total_loss_i) == parts.total`` exactly; an equal split would leave the mean
      short by ``(1 - 1/n) * lambda_c * contrastive``;
    * it makes the row self-consistent against its OWN columns:
      ``total_loss = lambda_y*target + lambda_c*contrastive_loss + lambda_r*residual +
      lambda_g*geometry_loss + lambda_l*local_structure_loss``. Under an equal split the row
      would carry a ``contrastive_loss`` that could not be reconciled with its own total.

    The consequence, stated because it is the price: **the correct micro-batch aggregate of
    ``total_loss`` is the MEAN, never the sum**, which is already true of ``geometry_loss``,
    ``contrastive_loss`` and ``local_structure_loss`` and is why those columns are documented
    as per-batch.

    This corrects a defect that predated the loss-term columns: the member re-evaluation was
    passed no ``pair_group`` -- one member cannot form a pair -- so ``member_parts.total`` was
    the batch total MINUS ``lambda_c * contrastive``. Measured on a two-member paired batch at
    ``lambda_c = 2``: the optimised total was 9.0004 and the mean journalled total 1.0004, 89%
    of the objective absent, and at the cryptic arms' ``lambda_c = 7.0`` against
    ``lambda_y = 0.01`` it would have been about 99%. The fix follows the ``lambda_l`` precedent
    the branch below records -- pass the whole-batch scalar through the objective -- rather than
    adding the weighted term outside it, so nothing here does arithmetic on a term it does not
    own and the summation order stays the objective's.
    """
    prediction = sample.surrogate_prediction
    achieved = _member_scalars(prediction, len(members))
    fractions = _member_scalars(sample.residual_fraction, len(members))
    terms = _loss_term_scalars(parts, objective, config)
    # Read off the generated sample rather than off the loss: the monitor is not in the
    # objective, so there is nothing in ``parts`` for it to come from.
    monitor = _monitor_rows(sample, len(members), config)
    # D-86's registry-declared monitors, on the same footing and for the same reason: the
    # property's own hook says which columns this run carries, and every value comes off the
    # SAMPLE. Repeated per member below rather than re-derived from ``member_parts``, because no
    # objective computes any of them.
    declared_monitor = _declared_monitor_rows(sample, len(members), config, objective)
    # Likewise from the sample rather than from ``target_tensor``: the row records the ORIGIN, and
    # deriving it back out of ``state - target`` would make the recorded origin a function of the
    # recorded target instead of an independent measurement the identity can be re-checked from.
    origins = _control_reporting_rows(sample, len(members), config)
    if len(members) == 1:
        return (
            _training_record(
                members[0],
                targets[0],
                achieved[0],
                _scalar(parts.target),
                _scalar(parts.residual),
                _scalar(parts.geometry),
                _scalar(parts.total),
                fractions[0],
                config,
                _off_path_scalars(parts),
                monitor=monitor[0],
                declared_monitor=declared_monitor[0],
                loss_terms=terms,
                control_reporting_value=origins[0],
            ),
        )
    records = []
    with torch.no_grad():
        for index, (unit, target) in enumerate(zip(members, targets, strict=True)):
            member = slice(index, index + 1)
            member_parts = objective(
                prediction[member].detach(),
                target_tensor[member].detach(),
                sample.residual_rms[member].detach(),
                sample.residual_fraction[member].detach(),
                geometry_loss=sample.geometry_loss.detach(),
                # The batch's local-structure value, like its geometry value: both are
                # whole-batch scalars, so a member re-evaluation reuses them rather than
                # recomputing. Omitting it made the journalled total disagree with the total
                # that was actually optimised, by exactly the lambda_l term.
                local_structure=(
                    None if sample.local_structure is None else sample.local_structure.detach()
                ),
                # The batch's PAIRED value, supplied rather than re-derived: a one-member slice
                # forms no pair, so ``pair_group`` cannot be passed and the term would silently
                # leave the total. ``None`` when the batch itself formed no pair, which is the
                # zero the objective priced.
                paired_loss=(None if parts.contrastive is None else parts.contrastive.detach()),
                property_scale=(
                    None if property_scale is None else property_scale[member].detach()
                ),
                raw_residual_fraction=(
                    None
                    if sample.raw_residual_fraction is None
                    else sample.raw_residual_fraction[member].detach()
                ),
                # Per member, unlike geometry and local structure: the declared extra inputs
                # are per-protein scalars carried one per batch member, so a member's own
                # slice is what its row must report. c0 is sliced with the RMS it divides,
                # because it is per protein (D-30) and a member's row must be hinged against
                # that member's own setpoint and not the batch's first.
                **_extra_sample_inputs(objective, sample, member=member),
            )
            records.append(
                _training_record(
                    unit,
                    target,
                    achieved[index],
                    _scalar(member_parts.target),
                    _scalar(member_parts.residual),
                    _scalar(member_parts.geometry),
                    _scalar(member_parts.total),
                    fractions[index],
                    config,
                    _off_path_scalars(member_parts),
                    monitor=monitor[index],
                    declared_monitor=declared_monitor[index],
                    loss_terms=terms,
                    control_reporting_value=origins[index],
                )
            )
    return tuple(records)


def _extra_sample_inputs(
    objective: SteeringLoss,
    sample: GeneratedTrainingSample,
    *,
    member: slice | None = None,
) -> dict[str, Tensor | None]:
    """Return the per-observation inputs this objective declares beyond the shared set.

    Empty for the shared :class:`~sae_bioemu.steering.training.SteeringLoss`, whose
    ``EXTRA_SAMPLE_INPUTS`` is ``()`` -- so the runner's call is byte-for-byte the call it
    made before any project added a term, and adding one never edits this module again.

    ``member`` slices each input to one batch member for the per-member journal
    re-evaluation. Every declared input is per-observation by contract, which is what makes
    one slicing rule correct for all of them.
    """
    resolved: dict[str, Tensor | None] = {}
    for name in type(objective).EXTRA_SAMPLE_INPUTS:
        value = getattr(sample, name)
        if value is not None and member is not None:
            value = value[member].detach()
        resolved[name] = value
    return resolved


def _off_path_scalars(parts: Any) -> dict[str, float | None]:
    """Return every project-declared journal column for one row, ``None`` where the term is off.

    ``None`` is not "zero": it is the term's absence, and it is what keeps the row's key set
    identical to every row written before these columns existed. The values come from the
    components' own :meth:`~sae_bioemu.steering.training.SteeringLossComponents.journal_columns`,
    which is empty on the shared components -- so a shared objective yields ``None`` for every
    column here without this module computing anything about a term it does not have.

    Keyed over :data:`_OFF_PATH_RECORD_KEYS` rather than returned as a fixed tuple: a project that
    adds a term adds it to that schema, and the positional form silently dropped any column past
    the second. A column a subclass reports that this schema does not name is REFUSED rather than
    ignored, because ``SteeringTrainingRecord`` would take it as an unknown field anyway and the
    failure should name the schema.
    """
    columns = parts.journal_columns()
    unknown = sorted(set(columns) - _OFF_PATH_RECORD_KEYS)
    if unknown:
        raise ValueError(
            f"the objective reports journal columns {unknown} that the persisted schema does not "
            "name; declare them in this property's PropertyRegistration.journal_columns, against "
            "the weight that prices each, which is what gates the column on that weight. That is "
            "a line in the property catalog and not an edit to this shared module"
        )
    return {name: columns.get(name) for name in sorted(_OFF_PATH_RECORD_KEYS)}


def _loss_term_scalars(
    parts: Any, objective: SteeringLoss, config: OrdinarySteeringConfig
) -> dict[str, float]:
    """Return the gated shared loss-term columns for a journal row, keyed by column name.

    Absence is the schema signal for "this weight is zero", so the dict holds only the columns
    :func:`_loss_term_columns` says this objective switches on, and holds a value for every one
    of them. The values come from the components' own
    :meth:`~sae_bioemu.steering.training.SteeringLossComponents.loss_term_columns`, so this
    module does no arithmetic on a term it does not own.

    A gated-on column whose component value is ``None`` is a real case for exactly one term and
    is not a missing measurement: ``lambda_c > 0`` with no pair in this micro-batch is priced at
    exactly zero by :func:`~sae_bioemu.steering.training.paired_dose_loss`, so the row records
    ``0.0``. Any other ``None`` at a nonzero weight is a value the objective was supposed to
    have and did not, and raises rather than being recorded as a zero that reads like a
    measurement.
    """
    columns = parts.loss_term_columns()
    base = _base_record_keys(config)
    resolved: dict[str, float] = {}
    for name in _loss_term_columns(objective, base):
        value = columns.get(name)
        if value is None:
            if name != "contrastive_loss":
                raise ValueError(
                    f"the objective prices {name} at a nonzero weight but reported no value; "
                    "recording that as zero would put a number in the journal the loss never "
                    "computed"
                )
            value = 0.0
        resolved[name] = _finite(value, f"training record {name}")
    return resolved


def _base_record_keys(config: OrdinarySteeringConfig) -> frozenset[str]:
    """Return the base (ungated) persisted key set this run's target construction defines."""
    if not config.is_relative:
        return _ABSOLUTE_RECORD_KEYS
    return _TWO_STATE_RECORD_KEYS if config.targets_two_state else _RELATIVE_RECORD_KEYS


def _validation_rows(
    members: Sequence[SteeringUnit],
    targets: Sequence[float],
    sample: GeneratedTrainingSample,
    config: OrdinarySteeringConfig,
    objective: SteeringLoss | None = None,
) -> list[dict[str, Any]]:
    """Return one validation row per batch member.

    ``target_absolute_error`` and ``residual_magnitude`` keep their exact meaning and
    units on both paths. The relative path additionally records the achieved property
    value, the signed error, and the sigma used to normalize, because the absolute error
    alone says neither which way the structure moved nor against what spread.

    A property additionally records whatever diagnostic columns it DECLARES
    (:func:`_declared_validation_columns`) -- the cryptic pocket's four D-35b geometry counts,
    the kinase motif channel, a direct metric in Angstrom. This is the LATENCY fix and nothing
    more: the D-35 gates
    otherwise run once, at held-out evaluation, so a fold coming apart under the adapter is
    invisible for the whole training run plus evaluation generation. Journalling them per
    validation draw makes the first epoch say it. They are not consulted by
    :func:`_validation_aggregate`, by :func:`_validation_key`, or by :func:`_validate_gates`:
    the epoch's selection metric and both configured gates are exactly what they were.
    """
    achieved_values = _member_scalars(sample.surrogate_prediction, len(members))
    fractions = _member_scalars(sample.residual_fraction, len(members))
    declared = _declared_validation_rows(sample, len(members), config, objective)
    origins = _control_reporting_rows(sample, len(members), config)
    rows = []
    for index, (unit, target) in enumerate(zip(members, targets, strict=True)):
        origin = origins[index]
        # On the two-state path ``targets`` carries the reporting-frame STATE, so the row's
        # ``target`` -- the quantity the achieved value is an error against -- is that state
        # converted through THIS draw's own origin. Both halves are recorded, because the state
        # is re-derivable from the schedule and the origin is not.
        resolved_target = target if origin is None else target - origin
        signed_error = achieved_values[index] - resolved_target
        row: dict[str, Any] = {
            "sequence_id": unit.sequence_id,
            "seed": unit.seed,
            "target": resolved_target,
            "target_absolute_error": abs(signed_error),
            "residual_magnitude": abs(fractions[index]),
        }
        if config.is_relative:
            # Siblings, not reinterpretations: the raw error keeps its units and meaning,
            # and the sigma-normalized sibling is what selection uses.
            sigma = _unit_sigma(unit, config)
            row["achieved_value"] = achieved_values[index]
            row["signed_error"] = signed_error
            row["sequence_sigma"] = sigma
            row["target_sigma_error"] = row["target_absolute_error"] / sigma
            if origin is not None:
                row["target_state"] = target
                row["control_reporting_value"] = origin
        # Empty for every property that declares no diagnostic column, so a radius-of-gyration
        # row is byte-identical.
        row.update(declared[index])
        rows.append(row)
    return rows


def _conditioning(
    unit: SteeringUnit, target: float, config: OrdinarySteeringConfig
) -> StandardizedAbsoluteTarget | RequestedLevel:
    """Return the adapter conditioning input for one scheduled unit."""
    if config.is_relative:
        return RequestedLevel(float(unit.level))
    return _standardized_absolute_target(target, config)


def _module_config(module: nn.Module) -> dict[str, Any]:
    attributes = {}
    for name in (
        "property_name",
        "model_family",
        "width",
        "input_width",
        "hidden_width",
        "hidden_layers",
        # Bound into the checkpoint identity: two adapters differing only in how the level
        # reaches the hidden state are different functions and must not be interchangeable.
        "level_conditioning",
        "variant",
        "selected_feature_index",
        "sae_checkpoint_fingerprint",
        "sae_state_fingerprint",
        "decoder_direction_fingerprint",
        # A warm-started adapter and a cold-started one reach different functions from the
        # same config, so the initialisation belongs to the identity like any other.
        "warm_start_direction_fingerprint",
        "rho_max",
        "eps",
    ):
        value = getattr(module, name, None)
        if isinstance(value, str | bool | int | float) or value is None:
            attributes[name] = value
    # Conditional, so every existing checkpoint keeps its exact historical key set: absent and
    # 1.0 both mean "the level branch is driven at unit gain". It has to appear when it is on,
    # though -- two branch arms differing only in the gain train to different functions, and
    # without this they would write byte-identical adapter identities and be interchangeable.
    # That is precisely the hole lambda_l sat in: accepted, applied, and invisible.
    level_gain = getattr(module, "level_gain", 1.0)
    if isinstance(level_gain, int | float) and not isinstance(level_gain, bool):
        if float(level_gain) != 1.0:
            attributes["level_gain"] = float(level_gain)
    # Conditional for the same reason, and load-bearing for a stronger one. Absent means "no
    # site channel", which is every radius-of-gyration checkpoint and cryptic arm C, so their
    # historical key set is untouched. Present, it names WHICH source filled the channel:
    # A_binary, A_smooth and B share every shape and every weight name and differ only here, so
    # without it three different experiments would write byte-identical adapter identities and
    # a checkpoint from one could be evaluated under another's config with the gate passing.
    site_conditioning = getattr(module, "site_conditioning", None)
    if isinstance(site_conditioning, str):
        attributes["site_conditioning"] = site_conditioning
    # Conditional for the same reason once more, and load-bearing for the strongest one yet.
    # Absent means ``tanh`` -- the capped adapter every checkpoint on disk is -- so no historical
    # key set moves. Present, it says the residual was NOT bounded by ``rho_max``. Nothing else
    # would say so: the mode changes no tensor shape and no parameter name, so an uncapped
    # checkpoint and a capped one at the same ``rho_max`` would otherwise write byte-identical
    # adapter identities and each could be evaluated under the other's config with every gate
    # green, reporting a dose the run never applied.
    residual_cap = getattr(module, "residual_cap", None)
    if isinstance(residual_cap, str) and residual_cap != DEFAULT_RESIDUAL_CAP:
        attributes["residual_cap"] = residual_cap
    config = {
        "class": f"{type(module).__module__}.{type(module).__qualname__}",
        "attributes": attributes,
        "state": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in module.state_dict().items()
        },
    }
    architecture_payload = getattr(module, "architecture_payload", None)
    if callable(architecture_payload):
        payload = architecture_payload()
        if payload is not None:
            config["portable_architecture"] = AdapterArchitecture.from_payload(payload).payload()
    return config


def _objective_config(objective: SteeringLoss) -> dict[str, Any]:
    config = {
        "class": f"{type(objective).__module__}.{type(objective).__qualname__}",
        "lambda_y": objective.lambda_y,
        "lambda_r": objective.lambda_r,
        "lambda_g": objective.lambda_g,
        # Bound into the identity: two runs differing only in the paired dose weight are
        # different optimizations, and without this a lambda_c = 1 checkpoint would be
        # interchangeable with a lambda_c = 0 one.
        "lambda_c": objective.lambda_c,
        "property_loss": objective.property_loss.value,
        "residual_penalty": objective.residual_penalty.value,
    }
    # Conditional for the same reason residual_penalty_units is: a run that does not use the
    # local-structure term must keep the historical key set, or every absolute and AF3
    # checkpoint is invalidated by a weight they never had. Present-and-zero and absent both
    # mean "no local term", so collapsing them loses nothing.
    #
    # It has to appear when it is on, though, and that was missing: the lambda_g = 0.3 pair
    # differing only in lambda_l wrote byte-identical adapter_config, so a checkpoint could
    # not say which arm produced it and the gate could not tell the two apart.
    if objective.lambda_l:
        config["lambda_l"] = objective.lambda_l
    # The normalisation the objective DECLARED, conditional for the same reason as everything
    # else here. Absent means one of the two historical constructions -- ``sequence_sigma`` on
    # the relative path, ``property_units`` on the absolute one -- which the adapter_config this
    # dict sits inside already names unambiguously at top level (``property_loss_normalization``
    # for a relative target, ``target_range`` and friends for an absolute one). Recording it
    # unconditionally would refuse every radius-of-gyration, AF3 and Local-Q checkpoint ever
    # written, and ``_require_matched_loss_normalization`` already guarantees this declaration is
    # the construction the top-level keys describe.
    if objective.property_loss_normalization in _TWO_STATE_LOSS_NORMALIZATIONS:
        config["property_loss_normalization"] = objective.property_loss_normalization
    # Whatever weights this objective's own project adds, gated on exactly the same rule and
    # for exactly the same reason: ``adapter_config`` is compared with canonical-JSON equality
    # on resume and at the evaluation gate, so an unconditional key would refuse every
    # radius-of-gyration checkpoint ever written, including the ones being appended to now.
    # Empty on the shared objective, so this line adds nothing to a shared identity -- see
    # :meth:`~sae_bioemu.steering.training.SteeringLoss.objective_config_extra`.
    config.update(objective.objective_config_extra())
    # Only the rho_max-normalized penalty carries a bound, and only it adds keys, so the
    # absolute and AF3 objective identities keep their exact historical key set.
    if objective.residual_penalty is ResidualPenaltyKind.FRACTION_OF_RHO_MAX:
        config["residual_penalty_units"] = objective.residual_penalty_units
        config["residual_penalty_rho_max"] = objective.rho_max
    return config


def _optimizer_config(optimizer: torch.optim.Optimizer, adapter: nn.Module) -> dict[str, Any]:
    names_by_id = {id(parameter): name for name, parameter in adapter.named_parameters()}
    state = optimizer.state_dict()
    return {
        "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "param_groups": [
            {
                **{key: _primitive(value) for key, value in state_group.items() if key != "params"},
                "parameter_names": [
                    names_by_id[id(parameter)] for parameter in live_group["params"]
                ],
            }
            for state_group, live_group in zip(
                state["param_groups"], optimizer.param_groups, strict=True
            )
        ],
    }


def _validate_schedule(
    schedule: SteeringSchedule,
    levels: DoseLevels = CONTINUOUS_DOSE_LEVELS,
    *,
    expect_validation: bool = True,
) -> None:
    """Fail closed on a schedule this workflow could not have produced.

    ``expect_validation`` is the caller's DECLARATION of which model-selection regime this run
    is in, never a property of the schedule it is given, and it is enforced as a BICONDITIONAL.
    ``True`` -- the default, and every run made before this existed -- requires a populated
    validation role, so a role that went missing still fails here. ``False`` requires an EMPTY
    one, because units that exist in the schedule, are fingerprinted into the checkpoint
    identity, and are then never generated are worse than no units at all: they would withhold
    those proteins from training and score nothing with them.
    """
    if not isinstance(schedule, SteeringSchedule) or not schedule.train:
        raise ValueError("schedule must contain train and validation units")
    if bool(schedule.validation) is not expect_validation:
        raise ValueError(
            "schedule must contain train and validation units"
            if expect_validation
            else (
                f"model_selection {MODEL_SELECTION_NONE!r} declares no validation role, but the "
                f"schedule carries {len(schedule.validation)} validation units: they would be "
                "bound into the checkpoint identity and never generated, withholding their "
                "proteins from training while scoring nothing"
            )
        )
    split_units = {
        "train": schedule.train,
        "validation": schedule.validation,
        "test": schedule.test,
    }
    sequence_sets = {}
    for split, units in split_units.items():
        if any(not isinstance(unit, SteeringUnit) or unit.split != split for unit in units):
            raise ValueError("schedule unit split does not match its role")
        sequence_sets[split] = {unit.sequence_id for unit in units}
        if any(
            not isinstance(unit.sequence_id, str)
            or not unit.sequence_id
            or not isinstance(unit.cycle, int)
            or isinstance(unit.cycle, bool)
            or unit.cycle < 0
            or not isinstance(unit.visit, int)
            or isinstance(unit.visit, bool)
            or unit.visit < 0
            or not isinstance(unit.seed, int)
            or isinstance(unit.seed, bool)
            or unit.seed < 0
            or isinstance(unit.level, bool)
            or not isinstance(unit.level, Real)
            or not math.isfinite(unit.level)
            for unit in units
        ):
            raise ValueError("schedule unit fields are invalid")
        if len(set(units)) != len(units):
            raise ValueError("schedule contains duplicate units")
        keys = [(unit.sequence_id, unit.cycle, unit.visit) for unit in units]
        if len(set(keys)) != len(keys):
            raise ValueError("schedule contains duplicate sequence/cycle/visit units")
        if any(unit.level not in levels.target for unit in units):
            raise ValueError("schedule levels must be canonical")
        # ``test`` keeps strict canonical order. ``validation`` is allowed to be a
        # permutation so it can be length-bucketed for batching: BioEmu pads a batch to its
        # longest member and cost scales as L^2, so chunking validation in canonical order
        # mixed an L=61 protein with an L=297 one and measured 1.73x worse than not batching
        # at all. The set is still pinned exactly -- only the order is free -- and the order
        # a run walks is deterministic and recorded in the batch plan.
        if split == "test" and tuple(units) != tuple(
            sorted(units, key=lambda unit: (unit.sequence_id, unit.cycle, unit.visit))
        ):
            raise ValueError("schedule units must use canonical contiguous ordering")
        if split == "validation" and units:
            # ``and units`` because an empty validation role has no membership to pin: with no
            # units the counter below is empty and its ``{1}`` comparison would refuse the very
            # schedule the guard above just declared legal. The emptiness itself is checked
            # there, against the caller's declaration, which is the only place it means
            # anything.
            #
            # Order is free, membership is not. Duplicate units and duplicate
            # (sequence, cycle, visit) keys are already rejected above, so what remains to
            # pin is that every sequence appears at every validation level exactly once --
            # a reorder must not be able to smuggle in a dropped or repeated dose.
            pairs = Counter((unit.sequence_id, unit.level) for unit in units)
            # Renamed from ``levels``, which shadowed this function's dose-set parameter and
            # made every downstream use of it read a set of floats instead (D-44).
            present = {unit.level for unit in units}
            if set(pairs.values()) != {1} or len(pairs) != len(sequence_sets[split]) * len(present):
                raise ValueError("validation schedule must visit each sequence/level pair once")
    all_units = (*schedule.train, *schedule.validation, *schedule.test)
    if len({unit.seed for unit in all_units}) != len(all_units):
        raise ValueError("schedule seeds must be globally unique")
    if any(
        sequence_sets[left] & sequence_sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("schedule sequence IDs must be disjoint across splits")

    train_sequence_ids = sequence_sets["train"]
    train_groups: dict[int, list[SteeringUnit]] = defaultdict(list)
    for unit in schedule.train:
        train_groups[unit.cycle].append(unit)
    if set(train_groups) != set(range(len(train_groups))):
        raise ValueError("train schedule epochs must be contiguous")
    flattened = []
    for epoch, units in sorted(train_groups.items()):
        # One epoch visits every training sequence at every target level, so it holds
        # ``len(sequences) x len(TARGET_LEVELS)`` units and each (sequence, level) pair
        # exactly once. That per-pair check is what makes the dose contrast within an epoch
        # auditable rather than merely counted.
        expected_units = len(train_sequence_ids) * len(levels.target)
        if {unit.sequence_id for unit in units} != train_sequence_ids or (
            len(units) != expected_units
        ):
            raise ValueError("train schedule epoch must contain every sequence at every level")
        pairs = Counter((unit.sequence_id, unit.level) for unit in units)
        if set(pairs.values()) != {1} or len(pairs) != expected_units:
            raise ValueError("train schedule epoch must visit each sequence/level pair once")
        counts = Counter(unit.level for unit in units)
        level_counts = tuple(counts[level] for level in levels.target)
        if max(level_counts) - min(level_counts) > 1:
            raise ValueError("train schedule epoch target levels must be balanced")
        if any(unit.cycle != epoch for unit in units):
            raise ValueError("train schedule epoch is inconsistent")
        flattened.extend(units)
    if tuple(flattened) != schedule.train:
        raise ValueError("train schedule epochs must use canonical contiguous ordering")
    # Each sequence meets every target level once per cycle, and the order in which it does
    # so is a canonical rotation of TARGET_LEVELS whose starting offset varies by sequence
    # and cycle. The rotation used to run *across* epochs, one level per sequence per epoch;
    # it now runs *within* one, so a sequence carries its own dose contrast inside a single
    # epoch instead of accumulating it over the whole run.
    by_sequence_cycle: dict[tuple[str, int], list[SteeringUnit]] = defaultdict(list)
    for unit in schedule.train:
        by_sequence_cycle[(unit.sequence_id, unit.cycle)].append(unit)
    rotations = {
        tuple(
            levels.target[(offset + index) % len(levels.target)]
            for index in range(len(levels.target))
        )
        for offset in range(len(levels.target))
    }
    if any(
        tuple(unit.level for unit in sorted(units, key=lambda item: item.visit)) not in rotations
        for units in by_sequence_cycle.values()
    ):
        raise ValueError("train schedule levels must rotate canonically within each epoch")
    for split in ("validation", "test"):
        groups: dict[str, list[SteeringUnit]] = defaultdict(list)
        for unit in split_units[split]:
            groups[unit.sequence_id].append(unit)
        for units in groups.values():
            if any(unit.cycle != 0 for unit in units):
                raise ValueError(f"{split} schedule must use one fixed cycle")
            expected = levels.epoch_validation if split == "validation" else levels.target
            _validate_level_group(units, f"{split} schedule", expected)


def _validate_level_group(
    units: Sequence[SteeringUnit], name: str, levels: tuple[float, ...]
) -> None:
    ordered_levels = tuple(unit.level for unit in sorted(units, key=lambda unit: unit.visit))
    rotations = {levels[offset:] + levels[:offset] for offset in range(len(levels))}
    if (
        len(units) != len(levels)
        or Counter(unit.level for unit in units) != Counter(levels)
        or {unit.visit for unit in units} != set(range(len(levels)))
        or ordered_levels not in rotations
    ):
        raise ValueError(f"{name} must contain one complete canonical target-level cycle")


def _schedule_fingerprint(schedule: SteeringSchedule) -> str:
    return hashlib.sha256(_json_bytes(_schedule_inventory(schedule))).hexdigest()


def _schedule_inventory(schedule: SteeringSchedule) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": [asdict(unit) for unit in schedule.train],
        "validation": [asdict(unit) for unit in schedule.validation],
        "test": [asdict(unit) for unit in schedule.test],
    }


def _schedule_from_inventory(value: object, *, expect_validation: bool = True) -> SteeringSchedule:
    if not isinstance(value, Mapping) or set(value) != {"train", "validation", "test"}:
        raise ValueError("ordinary checkpoint schedule inventory schema is invalid")
    unit_keys = {"split", "sequence_id", "cycle", "visit", "level", "seed"}
    # ``draw_seed`` is optional so a checkpoint written before the shared draw existed still
    # loads: absent means None, which is "generate from the identity seed" -- exactly the
    # pre-pairing behaviour. It is accepted but never required.
    optional_keys = {"draw_seed"}
    units = {}
    for split in ("train", "validation", "test"):
        rows = value[split]
        if not isinstance(rows, list) or any(
            not isinstance(row, dict)
            or not unit_keys <= set(row)
            or not set(row) <= unit_keys | optional_keys
            for row in rows
        ):
            raise ValueError("ordinary checkpoint schedule inventory schema is invalid")
        try:
            units[split] = tuple(SteeringUnit(**row) for row in rows)
        except (TypeError, ValueError) as error:
            raise ValueError("ordinary checkpoint schedule inventory schema is invalid") from error
    schedule = SteeringSchedule(units["train"], units["validation"], units["test"])
    # No config here: this parses a persisted inventory, and the levels it must satisfy are the
    # ones the run that wrote it used. The caller re-validates against its own dose set below,
    # and the schedule equality check that follows is what actually binds the two.
    #
    # ``expect_validation`` comes from the caller's reading of the checkpoint's own
    # ``model_selection`` declaration rather than from this inventory: an empty role here is the
    # consequence of that declaration, so deciding it from the inventory would make the check
    # satisfy itself.
    _validate_schedule(
        schedule,
        _inventory_dose_levels(schedule),
        expect_validation=expect_validation,
    )
    return schedule


def _inventory_dose_levels(schedule: SteeringSchedule) -> DoseLevels:
    """Return the dose set a persisted schedule was written under, or refuse it.

    A persisted inventory has to be parseable before its config is known, so the level set is
    recovered from the units rather than assumed. It is matched against the DECLARED sets and
    never inferred freely: an inventory whose levels match no declared set is a schedule this
    code cannot have produced, and accepting it would let an arbitrary dose grid resume as
    though it were canonical.
    """
    from .schedule import DECLARED_DOSE_LEVELS

    observed = {unit.level for unit in (*schedule.train, *schedule.validation, *schedule.test)}
    # The declared list includes the two one-directional two-state sets (D-101), so a
    # per-direction arm's checkpoint can be resumed, completed and evaluated: before they were
    # declared, a schedule whose every unit sat at ``1.0`` matched nothing here and the arm would
    # have trained to a checkpoint this function refused to read back.
    for candidate in DECLARED_DOSE_LEVELS:
        if observed == set(candidate.target):
            return candidate
    raise ValueError(
        f"ordinary checkpoint schedule inventory uses levels {sorted(observed)}, which match no "
        "declared dose set; a schedule this code did not produce must not resume as canonical"
    )


def _validation_rounds_due(
    boundaries: Sequence[int], cursor: int, config: OrdinarySteeringConfig
) -> int:
    """Return how many validation rounds the schedule owes by ``cursor``.

    **Zero, always, on a run that declared no validation.** A round comes due at an epoch
    boundary only because there is a validation role to run it on; with none, no boundary owes
    anything. Saying that once, here, rather than branching at each reader is what keeps the
    training loop's round trigger, the resume validator's history-boundary checks and the
    in-flight phase check all true without three separate special cases -- and a training loop
    whose ``due`` grew while its history could not would simply spin forever.
    """
    if not config.selects_on_validation:
        return 0
    return sum(boundary <= cursor for boundary in boundaries)


def _cycle_boundaries(units: Sequence[SteeringUnit]) -> tuple[int, ...]:
    boundaries = []
    for index, unit in enumerate(units, start=1):
        if index == len(units) or units[index].cycle != unit.cycle:
            boundaries.append(index)
    return tuple(boundaries)


def _sequence_representatives(schedule: SteeringSchedule) -> dict[str, SteeringUnit]:
    values = {}
    for unit in (*schedule.train, *schedule.validation):
        values.setdefault(unit.sequence_id, unit)
    return dict(sorted(values.items()))


def _empty_journal() -> dict[str, Any]:
    return {
        "schema_version": _JOURNAL_SCHEMA_VERSION,
        "training_records": [],
        "failures": [],
        "failure_keys": [],
        "attempts": {},
        "in_flight": None,
        "history": [],
        "active_validation": {},
        "best_validation_index": None,
        "best_checkpoint_path": None,
        "best_model_optimizer_fingerprint": None,
        "selected_best_model_optimizer_fingerprint": None,
        "bad_validation_count": 0,
        "early_stopped": False,
        "terminal_reason": None,
        "not_run_after_early_stopping": [],
        "passed": False,
        "workflow_complete": False,
        "validation_gate_failure": None,
    }


#: The exact key set one skipped training group records. Every field is either what the group WAS
#: (``key``/``cursor``/``size``/``units``) or what ended it (``phase``/``exception_class``/
#: ``message``/``attempts``), plus the ``budget`` it was granted so a reader can re-derive the
#: decision without the config.
_SKIPPED_UNIT_KEYS = frozenset(
    {
        "key",
        "cursor",
        "size",
        "units",
        "phase",
        "exception_class",
        "message",
        "attempts",
        "budget",
        "sequence_id",
        "seed",
    }
)

#: The journal key a run writes exactly when it SKIPPED at least one training group.
#:
#: Optional rather than a member of :func:`_empty_journal`, and rather than a
#: ``_JOURNAL_SCHEMA_VERSION`` bump, for the reason ``tests/test_steering_loss_term_journal.py``
#: records for the gated loss-term columns: the version is compared with exact equality BEFORE any
#: row is parsed, so moving it is strictly more destructive than the key-set check it would
#: duplicate. Here the key-set check says MORE than a version could, in both directions. A
#: pre-change reader handed a journal carrying this key refuses it by name (its ``set(value) !=
#: set(expected)`` fails), and a post-change reader handed a journal without it reads a run that
#: skipped nothing -- which is exactly what such a run is.
#:
#: The absence is therefore load-bearing: a run that skipped nothing journals BYTE-IDENTICALLY to
#: one made before the mechanism existed, and a run that skipped something cannot be mistaken for
#: it, because the key is present and non-empty. An always-present ``[]`` would have made the two
#: read identically at every later gate, which is what this key exists to prevent.
_SKIPPED_UNITS_KEY = "skipped_units"

#: The journal key a run writes exactly when it CONTINUED a shorter-budget checkpoint.
#:
#: Optional for exactly the reason :data:`_SKIPPED_UNITS_KEY` is, and the absence is equally
#: load-bearing: a run that was never extended journals byte-identically to one made before the
#: mechanism existed, and a run that WAS extended cannot be mistaken for one that always had the
#: longer budget, because the key is present and names the origin plan, the new plan, and the
#: epoch and unit it resumed at. An always-present ``[]`` would have made the two read identically
#: at every later gate, which is the whole thing this key exists to prevent.
#:
#: It is a LIST because extensions compose: a 40 -> 80 -> 120 run carries two records, in order.
_SCHEDULE_EXTENSIONS_KEY = "schedule_extensions"

#: The journal key a run writes exactly when a VALIDATION batch exhausted its attempts and the
#: shared tolerance granted it another set of resampled draws.
#:
#: Optional for exactly the reason :data:`_SKIPPED_UNITS_KEY` is, and the absence is equally
#: load-bearing: a run that never exhausted a validation batch journals byte-identically to one made
#: before this mechanism existed. It is a LIST because the grant can be made more than once, and it
#: is the validation analogue of ``skipped_units`` -- both are charged to ONE budget and
#: :func:`_tolerated_failure_count` adds them.
_VALIDATION_RETRIES_KEY = "validation_retries"

#: The exact key set one granted validation retry records. ``cursor`` is the TRAIN cursor, so it is
#: on the same axis the budget is measured on and the merged replay in
#: :func:`_validate_skip_budget` can sort the two inventories together; ``round_index`` and
#: ``validation_cursor`` say where inside validation it happened.
_VALIDATION_RETRY_KEYS = frozenset(
    {
        "key",
        "cursor",
        "round_index",
        "validation_cursor",
        "size",
        "units",
        "phase",
        "exception_class",
        "message",
        "sequence_id",
        "seed",
        "attempts",
        "budget",
    }
)

#: The phase a VALIDATION draw that died inside the generator is recorded under.
#:
#: Distinct from ``"validation"``, which now means every other way a validation batch can fail --
#: ``_validation_rows``, the optimizer-identity check, the frozen-stack assertion. The tolerance
#: consults this string, so a defect in the run still ends it while a bad structure does not, and
#: the distinction is recorded in the journal rather than inferred from the message. Distinct from
#: ``"generation"`` too, because ``_validate_failure_units`` resolves a ``"generation"`` failure
#: against the TRAIN split and a validation draw is not in it.
_VALIDATION_GENERATION_PHASE = "validation_generation"

#: Every combination of the optional journal keys, as exact key sets.
_OPTIONAL_JOURNAL_KEYS = (
    _SKIPPED_UNITS_KEY,
    _SCHEDULE_EXTENSIONS_KEY,
    _VALIDATION_RETRIES_KEY,
)


def _reopen_journal_for_extension(
    journal: Mapping[str, Any],
    extension: ScheduleExtension,
    *,
    checkpoint_path: Path | None = None,
    best_model_optimizer_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Return the journal of the adopted checkpoint, reopened at its own cursor and saying so.

    The run being continued may have reached ``max_cycles_reached`` or ``early_stopped`` and
    attested itself complete. Those five fields describe a decision made under the ORIGIN plan and
    are false under this one, so they are cleared rather than carried:

    ``terminal_reason`` / ``passed`` / ``workflow_complete``
        The run has not reached the end of the new plan, and
        :func:`_validate_checkpoint_progress` would refuse a terminal journal whose cursor is not
        the new plan's length -- correctly, because it is not. What the terminal state WAS is not
        lost: the resume record carries it as ``from_terminal_reason``.

    ``early_stopped`` / ``not_run_after_early_stopping``
        An early-stopped origin is continued from the ``.best-N`` sidecar, whose journal already
        carries ``early_stopped: false`` and an empty inventory because round N was a new best.
        Both are cleared anyway so the invariant holds however the checkpoint was reached: this
        run has not stopped, and it has not declined to walk anything.

    ``selected_best_model_optimizer_fingerprint``
        Set only at completion, and refused before it. The SELECTION survives --
        ``best_validation_index`` is untouched -- because the best epoch so far is still the best
        epoch so far; what is retracted is the claim that it is final.

    ``best_checkpoint_path`` and ``best_model_optimizer_fingerprint`` are REPAIRED rather than
    carried, and only when the state came from a ``.best-N`` sidecar. Those two fields are written
    immediately AFTER the sidecar is saved, so inside ``.best-30`` they still name ``.best-23``
    while ``best_validation_index`` already says 30. Every other field of that sidecar is
    self-consistent -- the real kinase arm's ``.best-30`` carries ``epoch=31``, ``cursor=1736``,
    31 history rows and 1736 training records -- but those two lag by one selection, and
    :func:`_validate_checkpoint_progress` compares them against
    ``_best_checkpoint_path(path, best_validation_index)``. They are recomputed from the adopted
    checkpoint itself, which IS the artifact ``best_validation_index`` names.

    ``history``, ``training_records`` and ``attempts`` are untouched. They need no truncation:
    a ``.best-N`` sidecar was written at the instant epoch N ended, so its journal already IS the
    journal truncated to that cursor. Truncating a journal that was never long is the whole reason
    resuming from a non-final epoch needs no reconstruction.
    """
    reopened = copy.deepcopy(dict(journal))
    reopened["terminal_reason"] = None
    reopened["passed"] = False
    reopened["workflow_complete"] = False
    reopened["selected_best_model_optimizer_fingerprint"] = None
    reopened["early_stopped"] = False
    reopened["not_run_after_early_stopping"] = []
    if checkpoint_path is not None and best_model_optimizer_fingerprint is not None:
        _repair_best_pointer(reopened, extension, checkpoint_path, best_model_optimizer_fingerprint)
    reopened[_SCHEDULE_EXTENSIONS_KEY] = [
        *list(reopened.get(_SCHEDULE_EXTENSIONS_KEY) or []),
        extension.record(),
    ]
    return _journal(reopened)


def _resumed_best_checkpoint_path(
    journal: Mapping[str, Any], extension: ScheduleExtension, checkpoint_path: Path
) -> Path | None:
    """Return the sidecar this run's selection state will point at once the journal is reopened.

    ``None`` when the adopted journal selected nothing -- a run with no validation role, or one
    preempted before its first round -- which is exactly when there is no sidecar to republish.
    """
    index = journal.get("best_validation_index")
    if index is None:
        return None
    return _best_checkpoint_path(checkpoint_path, int(index))


def _republish_best_checkpoint(
    path: Path, config: OrdinarySteeringConfig, identity: Mapping[str, Any]
) -> None:
    """Rewrite one sidecar's IDENTITY under this run's, leaving its weights exactly as they are.

    Read back through :func:`read_adapter_checkpoint`, so a sidecar that is truncated, incomplete
    or fails its own identity fingerprint raises here rather than being re-signed. What changes is
    the four provenance fields and nothing else; ``_model_optimizer_fingerprint`` digests only the
    adapter and optimizer state, so the journal's recorded value still describes this file.
    """
    stored = read_adapter_checkpoint(path)
    save_adapter_checkpoint(
        path,
        replace(
            stored,
            manifest_fingerprint=config.manifest_fingerprint,
            adapter_config=copy.deepcopy(dict(identity)),
            property_callback_identity=config.property_callback_identity,
            source_checkpoints=dict(config.source_checkpoints),
        ),
    )


def _repair_best_pointer(
    journal: dict[str, Any],
    extension: ScheduleExtension,
    checkpoint_path: Path,
    fingerprint: str,
) -> None:
    """Point the journal's best-checkpoint fields at the sidecar the state actually came from."""
    index = journal.get("best_validation_index")
    if index is None:
        return
    if extension.resumed_from != _best_checkpoint_path(checkpoint_path, index).name:
        # Not a sidecar resume, or a sidecar whose own index disagrees with the journal it
        # carries. The first needs no repair; the second is a corrupt artifact and the unchanged
        # progress validator refuses it by name a moment later.
        return
    journal["best_checkpoint_path"] = str(_best_checkpoint_path(checkpoint_path, index))
    journal["best_model_optimizer_fingerprint"] = fingerprint


def _validate_schedule_extensions(journal: Mapping[str, Any]) -> None:
    """Refuse a fabricated or malformed extension chain."""
    records = journal.get(_SCHEDULE_EXTENSIONS_KEY)
    if records is None:
        return
    if not isinstance(records, list) or not records:
        raise ValueError(
            "checkpoint schedule extension journal is invalid; the key is written only by a run "
            "that continued a shorter plan and is never empty"
        )
    previous: dict[str, Any] | None = None
    for record in records:
        if (
            not isinstance(record, Mapping)
            or set(record) != SCHEDULE_EXTENSION_RECORD_KEYS
            or record["schema"] != SCHEDULE_EXTENSION_SCHEMA
        ):
            raise ValueError("checkpoint schedule extension record schema is invalid")
        if not isinstance(record["crossed"], list) or any(
            not isinstance(row, Mapping) or set(row) != {"key", "from", "to"}
            for row in record["crossed"]
        ):
            raise ValueError("checkpoint schedule extension crossing disclosure is invalid")
        # A record exists because the run adopted something that was NOT a plain continuation of
        # its own destination, and there are exactly two ways for that to be true: the plan got
        # longer, or the run restarted at an earlier cursor than the artifact had reached. A
        # record that is neither describes nothing and is refused, and so is one that claims a
        # SHORTER plan.
        lengthened = (
            record["to_train_units"] > record["from_train_units"]
            and record["to_maximum_epochs"] > record["from_maximum_epochs"]
        )
        rolled_back = record["resumed_at_unit_cursor"] < record["from_train_units"]
        if (
            record["to_train_units"] < record["from_train_units"]
            or record["to_maximum_epochs"] < record["from_maximum_epochs"]
            or not 0 <= record["resumed_at_unit_cursor"] <= record["from_train_units"]
            or not (lengthened or rolled_back)
        ):
            raise ValueError(
                "checkpoint schedule extension record does not describe a lengthened plan or an "
                "earlier resume point"
            )
        # A chain must be a chain: each extension starts where the previous one ended.
        if previous is not None and (
            record["from_manifest_fingerprint"] != previous["to_manifest_fingerprint"]
            or record["from_train_units"] != previous["to_train_units"]
        ):
            raise ValueError("checkpoint schedule extension chain is discontinuous")
        previous = dict(record)


def _journal(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = _empty_journal()
    # EVERY subset of the optional keys, derived rather than enumerated. Enumerating them was fine
    # at two keys and is 8 combinations at three; deriving it means a fourth optional key cannot
    # ship with a missing combination, which would refuse a valid journal for a key set nobody
    # thought to write down.
    allowed = [
        set(expected) | set(optional)
        for count in range(len(_OPTIONAL_JOURNAL_KEYS) + 1)
        for optional in itertools.combinations(_OPTIONAL_JOURNAL_KEYS, count)
    ]
    if not isinstance(value, Mapping) or set(value) not in allowed:
        raise ValueError("checkpoint training journal schema is invalid")
    journal = copy.deepcopy(dict(value))
    _validate_skipped_units(journal)
    _validate_validation_retries(journal)
    _validate_schedule_extensions(journal)
    if (
        type(journal["schema_version"]) is not int
        or journal["schema_version"] != _JOURNAL_SCHEMA_VERSION
    ):
        raise ValueError("checkpoint training journal schema version is invalid")
    if any(
        not isinstance(journal[name], bool)
        for name in ("early_stopped", "passed", "workflow_complete")
    ):
        raise ValueError("checkpoint training journal completion flags must be boolean")
    if not isinstance(journal["training_records"], list):
        raise ValueError("checkpoint training records are invalid")
    _records(journal["training_records"])
    _failures(journal["failures"])
    if (
        not isinstance(journal["failure_keys"], list)
        or len(journal["failure_keys"]) != len(journal["failures"])
        or any(not isinstance(key, str) or not key for key in journal["failure_keys"])
    ):
        raise ValueError("checkpoint failure attempt keys are invalid")
    if not isinstance(journal["attempts"], dict) or any(
        not isinstance(key, str)
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        for key, count in journal["attempts"].items()
    ):
        raise ValueError("checkpoint attempt journal is invalid")
    for name in ("history",):
        if not isinstance(journal[name], list):
            raise ValueError("checkpoint validation history is invalid")
    if not isinstance(journal["active_validation"], dict):
        raise ValueError("checkpoint active validation state is invalid")
    if journal["in_flight"] is not None and not isinstance(journal["in_flight"], dict):
        raise ValueError("checkpoint in-flight state is invalid")
    if journal["terminal_reason"] not in (None, "early_stopped", "max_cycles_reached"):
        raise ValueError("checkpoint terminal reason is invalid")
    if not isinstance(journal["not_run_after_early_stopping"], list):
        raise ValueError("checkpoint early-stop inventory is invalid")
    try:
        tuple(SteeringUnit(**row) for row in journal["not_run_after_early_stopping"])
    except Exception as error:
        raise ValueError("checkpoint early-stop inventory is invalid") from error
    return journal


def _generation_skip(
    journal: Mapping[str, Any],
    attempt_key: str,
    cursor: int,
    size: int,
    members: Sequence[SteeringUnit],
    config: OrdinarySteeringConfig,
) -> dict[str, Any] | None:
    """Return the row an exhausted TRAIN group earns by being skipped, or ``None`` to end the run.

    Three conditions, all necessary:

    * the run declares a budget at all (:attr:`OrdinarySteeringConfig.tolerates_generation_failures`);
    * every recorded attempt at this group died in the GENERATION phase -- a unit lost while
      computing the loss, taking the optimizer step, or checking the frozen stack is a defect in
      the run rather than one bad structure, and stepping over it would hide the class of failure
      that must end a run; and
    * the skip fits inside :func:`generation_failure_budget` at this cursor, counted against
      :func:`_tolerated_failure_count`, which is ONE counter shared with the validation phase.

    **A SKIP IS THE TRAIN PHASE'S TOLERANCE AND IT IS NOT THE VALIDATION PHASE'S.** A validation
    failure is no longer fatal -- see :func:`_validation_retry` -- but what it is granted is another
    set of RESAMPLED draws, never a dropped observation. That is not a preference. A round's
    ``observation_count`` is checked against ``len(schedule.validation)`` on every resume
    (:func:`_validate_validation_record`), so a partial round cannot be persisted at all; and it
    should not be, because a round with a different denominator is not comparable with the rounds
    before it, and that comparability is what makes the loss trajectory readable. A training draw is
    one of tens of thousands and its loss is averaged into a gradient, so dropping it is a rounding
    error; a validation draw is one of 12 to 180 and its value enters the selection metric directly.
    Both phases are tolerated, on one rate, and each is tolerated in the only way its phase permits.
    """
    if not config.tolerates_generation_failures:
        return None
    recorded = [
        failure
        for failure, key in zip(
            _failures(journal["failures"]), journal["failure_keys"], strict=True
        )
        if key == attempt_key
    ]
    if not recorded or any(failure.phase != "generation" for failure in recorded):
        return None
    budget = generation_failure_budget(config, cursor)
    # ONE failure, whatever the group costs. ``size`` is what the skip LOSES and is recorded below;
    # it is not what the budget counts. Comparing the two is the defect a measured run exposed: at
    # cursor 798 the group held 27 units against a budget of 8, so a run with seven failures of
    # headroom refused to step over its first one. See :func:`generation_failure_budget`.
    if _tolerated_failure_count(journal) + 1 > budget:
        return None
    last = recorded[-1]
    # ``sequence_id``/``seed`` are the GROUP'S FIRST MEMBER, not necessarily the draw that failed,
    # and that is a limit of what is knowable rather than a shortcut. ``_failure`` builds its
    # record from ``members[0]`` because the generator raises for the whole batch: the exception
    # escapes ``generate_batch`` with no member attached, so the runner cannot say which of the 27
    # draws produced the bad backbone. ``units`` below carries EVERY member, so a reader has the
    # full candidate set rather than a single name that may be the wrong one -- which matters,
    # because a measured run's record reads ``CHLOR1R1 seed 2809694211`` and that is the first
    # member of the group at cursor 798, not a demonstrated culprit.
    return {
        "key": attempt_key,
        "cursor": int(cursor),
        "size": int(size),
        "units": [asdict(unit) for unit in members],
        "phase": last.phase,
        "exception_class": last.exception_class,
        "message": last.message,
        "sequence_id": last.sequence_id,
        "seed": last.seed,
        "attempts": _attempts(journal, attempt_key),
        "budget": budget,
    }


def _validate_skipped_units(journal: Mapping[str, Any]) -> None:
    """Fail closed unless a present ``skipped_units`` block is a well-formed, ordered inventory."""
    if _SKIPPED_UNITS_KEY not in journal:
        return
    rows = journal[_SKIPPED_UNITS_KEY]
    # A present-but-empty block would be a run claiming it skipped something and naming nothing,
    # and it would also make a no-skip run's journal differ from a pre-mechanism one for no
    # content at all. Absence is the only spelling of "nothing was skipped".
    if not isinstance(rows, list) or not rows:
        raise ValueError("checkpoint skipped-unit inventory is invalid")
    previous = -1
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _SKIPPED_UNIT_KEYS:
            raise ValueError("checkpoint skipped-unit schema is invalid")
        cursor, size = row["cursor"], row["size"]
        if (
            not isinstance(cursor, int)
            or isinstance(cursor, bool)
            or cursor < 0
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
            or cursor <= previous
            or row["key"] != f"train:{cursor}"
            or not isinstance(row["attempts"], int)
            or isinstance(row["attempts"], bool)
            or row["attempts"] < 1
            or not isinstance(row["budget"], int)
            or isinstance(row["budget"], bool)
            or row["budget"] < 1
            or not isinstance(row["units"], list)
            or len(row["units"]) != size
        ):
            raise ValueError("checkpoint skipped-unit inventory is inconsistent")
        # Generation and generation only. A unit lost while COMPUTING the loss is a defect in the
        # objective, the optimizer or the frozen stack, not one bad structure, and stepping over it
        # would hide exactly the class of failure that must end a run.
        if row["phase"] != "generation":
            raise ValueError(
                "checkpoint skipped-unit phase is invalid: only a generation failure may be "
                f"tolerated, and this row records {row['phase']!r}"
            )
        for name in ("exception_class", "message", "sequence_id"):
            if not isinstance(row[name], str) or not row[name]:
                raise ValueError("checkpoint skipped-unit failure detail is invalid")
        if not isinstance(row["seed"], int) or isinstance(row["seed"], bool):
            raise ValueError("checkpoint skipped-unit failure detail is invalid")
        try:
            tuple(SteeringUnit(**unit) for unit in row["units"])
        except Exception as error:
            raise ValueError("checkpoint skipped-unit inventory is invalid") from error
        previous = cursor


def _validation_retry(
    journal: Mapping[str, Any],
    attempt_key: str,
    train_cursor: int,
    round_index: int,
    validation_cursor: int,
    size: int,
    members: Sequence[SteeringUnit],
    config: OrdinarySteeringConfig,
) -> dict[str, Any] | None:
    """Return the row an exhausted VALIDATION batch earns by being REDRAWN, or ``None`` to stop.

    The validation half of the one shared tolerance, and it grants a different thing from the
    training half because a validation draw can only be tolerated one way. A tolerated training
    failure SKIPS its accumulation group; a tolerated validation failure clears the batch's attempt
    counter so it gets another ``maximum_unit_attempts`` draws, each at a fresh derived seed
    (:func:`resampled_generation_seed`). Nothing is ever dropped, so a round either reaches
    ``len(schedule.validation)`` observations or the run ends -- which is what keeps every round's
    ``observation_count`` equal to every other round's and the loss trajectory comparable.

    The same three conditions as :func:`_generation_skip`, deliberately, so the two halves cannot
    drift into two policies:

    * the run declares a budget at all;
    * every recorded attempt at this batch died in the GENERATION phase -- ``validation_generation``
      here, because a failure inside ``_validation_rows`` or the frozen-stack assertions is a defect
      in the run and must still end it; and
    * the grant fits inside :func:`generation_failure_budget` at the TRAIN cursor, counted against
      the same :func:`_tolerated_failure_count` a skip is counted against.

    The budget is evaluated at the train cursor rather than at the validation cursor so both halves
    are measured on ONE axis. A rate per validation draw and a rate per training unit are different
    quantities, and a budget that switched axis by phase would mean two policies wearing one name.

    **A PRE-CHANGE JOURNAL'S ``"validation"`` FAILURE IS NOT RECONCILABLE HERE, DELIBERATELY.** Job
    a measured run's journal carries ``validation:18:0 -> 2`` with both failures recorded under the flat
    ``"validation"`` phase, because that is the only phase the loop wrote before this split existed.
    Accepting it would be fail-open on exactly the class of failure that must end a run: under the
    old spelling a bad structure and a broken frozen stack are the same string, so a run resumed
    that way could redraw its way past a defect in itself. The refusal costs nothing anyway --
    reaching this function at all requires this code to be present, which moves the ``training_only``
    group digest, which moves ``training_code_fingerprint``, which moves the non-exempt
    ``generation_callback_identity`` key of ``adapter_config``, which
    ``validate_adapter_checkpoint`` compares for canonical-JSON equality. No checkpoint written
    before this change can be resumed under it by any route, so there is no journal in existence
    that this clause makes worse.
    """
    if not config.tolerates_generation_failures:
        return None
    recorded = [
        failure
        for failure, key in zip(
            _failures(journal["failures"]), journal["failure_keys"], strict=True
        )
        if key == attempt_key
    ]
    if not recorded or any(failure.phase != _VALIDATION_GENERATION_PHASE for failure in recorded):
        return None
    budget = generation_failure_budget(config, train_cursor)
    if _tolerated_failure_count(journal) + 1 > budget:
        return None
    last = recorded[-1]
    # ``sequence_id``/``seed`` name the batch's FIRST MEMBER and not a demonstrated culprit, for
    # exactly the reason ``_generation_skip`` states: the generator raises for the whole batch with
    # no member attached. ``units`` carries every member so a reader has the full candidate set.
    return {
        "key": attempt_key,
        "cursor": int(train_cursor),
        "round_index": int(round_index),
        "validation_cursor": int(validation_cursor),
        "size": int(size),
        "units": [asdict(unit) for unit in members],
        "phase": last.phase,
        "exception_class": last.exception_class,
        "message": last.message,
        "sequence_id": last.sequence_id,
        "seed": last.seed,
        "attempts": _generation_attempts_spent(journal, attempt_key),
        "budget": budget,
    }


def _validate_validation_retries(journal: Mapping[str, Any]) -> None:
    """Fail closed unless a present ``validation_retries`` block is a well-formed inventory."""
    if _VALIDATION_RETRIES_KEY not in journal:
        return
    rows = journal[_VALIDATION_RETRIES_KEY]
    # Present-but-empty is refused for the reason ``skipped_units`` refuses it: absence is the only
    # spelling of "nothing was redrawn", so a run that redrew nothing journals byte-identically to
    # one made before this mechanism existed.
    if not isinstance(rows, list) or not rows:
        raise ValueError("checkpoint validation-retry inventory is invalid")
    previous = (-1, -1)
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _VALIDATION_RETRY_KEYS:
            raise ValueError("checkpoint validation-retry schema is invalid")
        cursor, size = row["cursor"], row["size"]
        round_index, validation_cursor = row["round_index"], row["validation_cursor"]
        if (
            any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in (cursor, round_index, validation_cursor)
            )
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
            or not isinstance(row["units"], list)
            or len(row["units"]) != size
            or row["key"] != f"validation:{round_index}:{validation_cursor}"
            or not isinstance(row["attempts"], int)
            or isinstance(row["attempts"], bool)
            or row["attempts"] < 1
            or not isinstance(row["budget"], int)
            or isinstance(row["budget"], bool)
            or row["budget"] < 1
            # Non-decreasing in (round, validation cursor). A round is walked front to back and
            # rounds are walked in order, so a row out of order is a hand-edited journal.
            or (round_index, validation_cursor) < previous
        ):
            raise ValueError("checkpoint validation-retry inventory is inconsistent")
        if row["phase"] != _VALIDATION_GENERATION_PHASE:
            raise ValueError(
                "checkpoint validation-retry phase is invalid: only a generation failure may be "
                f"tolerated, and this row records {row['phase']!r}"
            )
        for name in ("exception_class", "message", "sequence_id"):
            if not isinstance(row[name], str) or not row[name]:
                raise ValueError("checkpoint validation-retry failure detail is invalid")
        if not isinstance(row["seed"], int) or isinstance(row["seed"], bool):
            raise ValueError("checkpoint validation-retry failure detail is invalid")
        try:
            tuple(SteeringUnit(**unit) for unit in row["units"])
        except Exception as error:
            raise ValueError("checkpoint validation-retry inventory is invalid") from error
        previous = (round_index, validation_cursor)


def _skipped_units(journal: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Every skipped training group this journal records, in schedule order."""
    return tuple(journal.get(_SKIPPED_UNITS_KEY) or ())


def _validation_retries(journal: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Every validation batch this journal granted a fresh set of resampled draws, in order."""
    return tuple(journal.get(_VALIDATION_RETRIES_KEY) or ())


def _tolerated_failure_count(journal: Mapping[str, Any]) -> int:
    """How many generation failures this run has stepped over, in EITHER phase.

    ONE counter against ONE budget. Keeping two would make a run's tolerance depend on which phase
    happened to lose the draws, which is not a property anyone chose -- and a generator that is
    breaking backbones breaks them in both phases, so counting them apart would let a broken run
    spend two budgets before either cap noticed.
    """
    return len(_skipped_units(journal)) + len(_validation_retries(journal))


def _generation_attempts_spent(journal: Mapping[str, Any], attempt_key: str) -> int:
    """How many attempts at ``attempt_key`` have already failed, over the whole run.

    Distinct from :func:`_attempts`, and the difference is the resample index. ``attempts`` is the
    counter ``maximum_unit_attempts`` is compared against, and a granted validation retry CLEARS it
    so the batch gets a fresh set of draws; this count is monotone and never cleared, so the
    substitute seed a retry draws is always one it has not drawn before. The two are equal on every
    run that was never granted a retry, which is every run on disk.
    """
    return sum(1 for key in journal["failure_keys"] if key == attempt_key)


def _skipped_unit_count(journal: Mapping[str, Any]) -> int:
    """How many scheduled TRAIN UNITS -- not groups -- this journal skipped."""
    return sum(int(row["size"]) for row in _skipped_units(journal))


def _trained_prefix(
    schedule_train: Sequence[SteeringUnit], cursor: int, journal: Mapping[str, Any]
) -> tuple[SteeringUnit, ...]:
    """The schedule prefix a run at ``cursor`` actually holds training records for.

    The committed prefix minus the skipped groups. With no skips this is
    ``schedule_train[:cursor]`` exactly, which is what every check that used that expression
    before the tolerance existed compared against.
    """
    blocked: set[int] = set()
    for row in _skipped_units(journal):
        blocked.update(range(int(row["cursor"]), int(row["cursor"]) + int(row["size"])))
    return tuple(unit for index, unit in enumerate(schedule_train[:cursor]) if index not in blocked)


def _validate_skip_budget(journal: Mapping[str, Any], config: OrdinarySteeringConfig) -> None:
    """Fail closed unless every tolerated failure this journal records fit the budget it had then.

    Replayed rather than trusted, over BOTH halves of the tolerance merged into one sequence. The
    budget is a pure function of the configured caps and the cursor the tolerance was taken at,
    both of which the journal carries, so a checkpoint whose tolerances exceed what its own
    configuration permits is refused on load instead of being resumed into a run that then reports
    ``complete: true``.

    Merging by cursor is exact rather than approximate: :func:`generation_failure_budget` is a
    function of the cursor alone and is non-decreasing in it, so two rows at one cursor grant the
    same budget in either order and the running count is the same either way.

    **The recorded budget is compared with ``<=``, not with ``==``, and that is a fix rather than a
    loosening.** The harm this check exists to catch is ``taken > budget`` -- a run that stepped
    over more failures than it was allowed -- and a row recorded under a TIGHTER budget is
    trivially inside a looser one. Demanding equality refused a LOOSENED config: a checkpoint whose
    rows say ``budget: 8`` could not be resumed by a run that had since raised the cap to 16, even
    though every one of its skips is permitted many times over under 16. What must still be refused
    is the other direction -- a row recorded under a LOOSER budget than this config grants, which is
    a run being resumed into a policy it already violated -- and ``row["budget"] > budget`` refuses
    exactly that.
    """
    rows = sorted(
        (*_skipped_units(journal), *_validation_retries(journal)),
        key=lambda row: int(row["cursor"]),
    )
    taken = 0
    for row in rows:
        taken += 1
        budget = generation_failure_budget(config, int(row["cursor"]))
        if int(row["budget"]) > budget or taken > budget:
            raise ValueError(
                "checkpoint tolerated-failure budget is inconsistent: the run recorded "
                f"{taken} tolerated generation failures at cursor {row['cursor']} under a "
                f"recorded budget of {row['budget']}, where "
                f"maximum_generation_failures={config.maximum_generation_failures} and "
                f"maximum_generation_failure_fraction="
                f"{config.maximum_generation_failure_fraction} permit {budget}"
            )


def _validate_checkpoint_progress(
    checkpoint: AdapterCheckpoint,
    checkpoint_path: Path,
    schedule: SteeringSchedule,
    journal: Mapping[str, Any],
    config: OrdinarySteeringConfig,
    baselines: Mapping[str, float],
    train_offsets: Mapping[int, int] | None = None,
    validation_offsets: Mapping[int, int] | None = None,
    objective: SteeringLoss | None = None,
) -> None:
    cursor = checkpoint.unit_cursor
    records = _records(journal["training_records"])
    # The committed prefix MINUS the groups this run skipped. With no skips -- every checkpoint
    # written before the tolerance existed, and every healthy run made since -- ``_trained_prefix``
    # returns ``schedule.train[:cursor]`` and ``_skipped_unit_count`` returns zero, so this is the
    # identical comparison it was before the tolerance existed.
    _validate_skip_budget(journal, config)
    skipped = _skipped_unit_count(journal)
    expected_prefix = _trained_prefix(schedule.train, cursor, journal)
    if (
        len(records) != cursor - skipped
        or tuple(record.unit for record in records) != expected_prefix
    ):
        raise ValueError("checkpoint training records must equal the committed schedule prefix")
    if any(int(row["cursor"]) + int(row["size"]) > cursor for row in _skipped_units(journal)):
        raise ValueError("checkpoint skipped-unit inventory runs past the committed cursor")
    if train_offsets is not None:
        _validate_batch_boundary(cursor, train_offsets, len(schedule.train), "training")
    for record in records:
        expected_target = _unit_target(record.unit, config, baselines)
        # On the two-state path the schedule determines the reporting-frame STATE and nothing
        # else: the training-frame target also depends on that draw's own control, which is a
        # measurement rather than a schedule property. So the re-derivable half is re-derived
        # here and the identity binding it to the persisted target is checked in
        # ``_validate_training_record_fields``. Comparing ``record.target`` against the state
        # would be comparing two frames and would refuse every correct row.
        observed = record.target_state if config.targets_two_state else record.target
        if observed != expected_target:
            raise ValueError("checkpoint training record target does not match the schedule")
        _validate_training_record_fields(record, config, objective)

    boundaries = _cycle_boundaries(schedule.train)
    # Zero on a no-validation run, which makes the two checks below say the right thing without
    # a branch apiece: a persisted history is refused outright (nothing could have written one),
    # and no epoch boundary is ever reported as skipped.
    due = _validation_rounds_due(boundaries, cursor, config)
    history = journal["history"]
    if len(history) > due:
        raise ValueError("checkpoint validation history exceeds completed epoch boundaries")
    for index, record in enumerate(history):
        _validate_validation_record(
            record,
            index,
            boundaries[index],
            schedule.validation,
            baselines,
            config,
            objective,
        )

    active = journal["active_validation"]
    if active:
        if set(active) != {"round_index", "cursor", "rows"}:
            raise ValueError("checkpoint active validation schema is invalid")
        validation_cursor = active["cursor"]
        if (
            active["round_index"] != len(history)
            or not isinstance(validation_cursor, int)
            or isinstance(validation_cursor, bool)
            or not 0 <= validation_cursor <= len(schedule.validation)
            or not isinstance(active["rows"], list)
            or len(active["rows"]) != validation_cursor
            or len(history) >= due
            or cursor != boundaries[len(history)]
        ):
            raise ValueError("checkpoint active validation boundary is inconsistent")
        if validation_offsets is not None:
            _validate_batch_boundary(
                validation_cursor, validation_offsets, len(schedule.validation), "validation"
            )
        _validate_validation_rows(
            active["rows"],
            schedule.validation[:validation_cursor],
            baselines,
            config,
            objective,
        )
    elif len(history) < due and cursor != boundaries[len(history)]:
        raise ValueError("checkpoint validation history skipped an epoch boundary")

    best_index = journal["best_validation_index"]
    if history:
        expected_best = min(
            range(len(history)),
            key=lambda index: _validation_key(
                history[index], config.is_relative, _selection_metric_name(config)
            ),
        )
        if best_index != expected_best:
            raise ValueError("checkpoint best validation index is inconsistent")
        if journal["bad_validation_count"] != len(history) - expected_best - 1:
            raise ValueError("checkpoint bad validation count is inconsistent")
        if not isinstance(journal["best_checkpoint_path"], str) or not isinstance(
            journal["best_model_optimizer_fingerprint"], str
        ):
            raise ValueError("checkpoint best validation identity is missing")
        _fingerprint(
            journal["best_model_optimizer_fingerprint"],
            "best model optimizer fingerprint",
        )
        expected_best_path = _best_checkpoint_path(checkpoint_path, best_index)
        if Path(journal["best_checkpoint_path"]) != expected_best_path:
            raise ValueError("checkpoint best validation path is inconsistent")
        best_checkpoint = load_adapter_checkpoint(
            expected_best_path,
            expected_manifest_fingerprint=config.manifest_fingerprint,
            expected_adapter_config=checkpoint.adapter_config,
            expected_property_callback_identity=config.property_callback_identity,
            expected_source_checkpoints=config.source_checkpoints,
        )
        if (
            _model_optimizer_fingerprint(best_checkpoint)
            != journal["best_model_optimizer_fingerprint"]
        ):
            raise ValueError("checkpoint best validation state fingerprint is inconsistent")
    elif (
        best_index is not None
        or journal["bad_validation_count"] != 0
        or journal["best_checkpoint_path"] is not None
        or journal["best_model_optimizer_fingerprint"] is not None
    ):
        raise ValueError("checkpoint best validation state exists without history")

    if checkpoint.epoch != len(history):
        raise ValueError("checkpoint epoch must equal completed validation history count")
    _validate_attempt_keys(journal["attempts"], schedule, journal, cursor, config)
    _validate_failure_schedule(journal["failures"], schedule, baselines, config)
    _validate_attempt_failure_consistency(journal, schedule)
    _validate_in_flight(journal["in_flight"], schedule, journal, due, config)

    terminal_reason = journal["terminal_reason"]
    skipped = tuple(SteeringUnit(**row) for row in journal["not_run_after_early_stopping"])
    if terminal_reason == "early_stopped":
        if (
            journal["early_stopped"] is not True
            or config.early_stopping_patience is None
            or journal["bad_validation_count"] < config.early_stopping_patience
            or cursor not in boundaries
            or skipped != schedule.train[cursor:]
            or active
            or len(history) != due
        ):
            raise ValueError("checkpoint early-stop terminal inventory is inconsistent")
    elif terminal_reason == "max_cycles_reached":
        if journal["early_stopped"] or cursor != len(schedule.train) or skipped:
            raise ValueError("checkpoint max-cycle terminal state is inconsistent")
    elif skipped:
        raise ValueError("checkpoint terminal state is incomplete")
    elif journal["early_stopped"] and (
        config.early_stopping_patience is None
        or journal["bad_validation_count"] < config.early_stopping_patience
        or cursor not in boundaries
        or active
        or len(history) != due
        or journal["workflow_complete"]
        or journal["passed"]
    ):
        raise ValueError("checkpoint persisted early-stop decision is inconsistent")
    if journal["passed"] is not journal["workflow_complete"]:
        raise ValueError("checkpoint completion flags are inconsistent")
    selected = journal["selected_best_model_optimizer_fingerprint"]
    if journal["workflow_complete"]:
        if selected != journal["best_model_optimizer_fingerprint"]:
            raise ValueError("checkpoint selected best state is inconsistent")
    elif selected is not None:
        raise ValueError("checkpoint selected best state exists before completion")
    if journal["workflow_complete"] or journal["passed"]:
        if terminal_reason is None or journal["in_flight"] is not None:
            raise ValueError("checkpoint completion cannot attest unfinished work")


def _validate_validation_record(
    record: object,
    index: int,
    train_cursor: int,
    units: Sequence[SteeringUnit],
    baselines: Mapping[str, float],
    config: OrdinarySteeringConfig,
    objective: SteeringLoss | None = None,
) -> None:
    if not _validation_record_schema_is_valid(
        record, config.is_relative, _selection_metric_name(config)
    ):
        raise ValueError("checkpoint validation history schema is invalid")
    if (
        record["validation_index"] != index
        or record["train_cursor"] != train_cursor
        or record["observation_count"] != len(units)
        or not isinstance(record["rows"], list)
        or len(record["rows"]) != len(units)
    ):
        raise ValueError("checkpoint validation history boundary is inconsistent")
    _validate_validation_rows(record["rows"], units, baselines, config, objective)
    expected = _validation_aggregate(
        record["rows"], index, train_cursor, config.is_relative, config.validation_selection_metric
    )
    if any(record[name] != expected[name] for name in expected if name != "rows"):
        raise ValueError("checkpoint validation history aggregate is inconsistent")


#: Columns `_learning_diagnostics` appends to every validation aggregate. ACCEPTED, NEVER REQUIRED,
#: and the asymmetry is the whole point: a journal written before these existed is still valid, and
#: a journal written WITH them must not be refused by the very reader that produced it.
#:
#: Added after exactly that happened. `_validation_aggregate` has called
#: `aggregate.update(_learning_diagnostics(rows))` since the diagnostics landed, but this key set
#: never listed the three, and both readers below compared with `set(record) != keys`. So a run
#: REFUSED ITS OWN OUTPUT: resume rejected any checkpoint carrying a completed validation round, and
#: `validate_ordinary_completion_checkpoint` rejected the completion it had just written. Training
#: succeeded, the adapter and journals were correct on disk, and a read-back validator threw them
#: away at the last gate -- the same failure shape as the `keep_draw_groups_whole` incident that
#: cost a 17-hour arm, reintroduced three weeks later in a different validator.
_LEARNING_DIAGNOSTIC_KEYS = frozenset(
    {
        "achieved_abs_median",
        "achieved_over_ask_median",
        "achieved_target_sign_agreement",
    }
)


def _validation_record_schema_is_valid(
    record: object, relative: bool, selection_metric: str | None = None
) -> bool:
    """Is ``record`` a validation aggregate: every required key, and no unexpected extra?

    Spelled once because it is checked in two places -- on resume and on completion attestation --
    and a disagreement between them is a run that can attest what it cannot reload.
    """
    if not isinstance(record, dict):
        return False
    required = _validation_record_keys(relative, selection_metric)
    return required <= set(record) and set(record) <= required | _LEARNING_DIAGNOSTIC_KEYS


def _validation_record_keys(relative: bool, selection_metric: str | None = None) -> set[str]:
    """The persisted validation-aggregate key set for one run's configuration.

    ``selection_metric`` is the DECLARED metric's aggregate key, and it is added rather than
    substituted: the sibling metrics are still computed, so a declared metric widens the record.
    ``None`` -- every run on disk -- reproduces the exact historical key set on both paths.
    """
    keys = {
        "validation_index",
        "train_cursor",
        "target_mae",
        "residual_magnitude",
        "observation_count",
        "rows",
    }
    if relative:
        keys = keys | {RELATIVE_SELECTION_METRIC}
    if selection_metric is not None and selection_metric not in (
        ABSOLUTE_SELECTION_METRIC,
        RELATIVE_SELECTION_METRIC,
    ):
        keys = keys | {selection_metric}
    return keys


def _validation_row_keys(
    relative: bool, declared: Sequence[str] = (), two_state: bool = False
) -> set[str]:
    """Return the persisted validation-row key set for one run's configuration.

    ``declared`` adds the diagnostic columns this run's PROPERTY declares -- the cryptic
    pocket's four D-35b monitor counts, the kinase motif channel, a direct metric in Angstrom --
    and defaults to empty, so every existing caller and every journal on disk keeps exactly the
    schema it was written against. It used to be a ``cryptic: bool``, and that predicate is why
    the kinase programme has no held-out ``d_K`` on any arm it has ever run.

    ``two_state`` adds the two columns that make the two-state row's ``target`` readable: the
    reporting-frame ``target_state`` the level named, and the ``control_reporting_value`` it was
    converted through. Without them the row's ``target`` is a number no reader could re-derive or
    interpret -- the same objection that made them persisted training-record columns. The two
    flags are independent arguments even though a cryptic arm is the only two-state run today,
    because one is gated on the ARM and the other on the TARGET CONSTRUCTION, and the kinase
    programme is a two-state arm that is not a cryptic one.
    """
    keys = {
        "sequence_id",
        "seed",
        "target",
        "target_absolute_error",
        "residual_magnitude",
    }
    if relative:
        keys |= {"achieved_value", "signed_error", "sequence_sigma", "target_sigma_error"}
        if two_state:
            keys |= {"target_state", "control_reporting_value"}
    return keys | set(declared)


def _validate_training_record_fields(
    record: SteeringTrainingRecord,
    config: OrdinarySteeringConfig,
    objective: SteeringLoss | None = None,
) -> None:
    """Fail closed unless a persisted record matches its target construction exactly.

    ``objective`` binds the optional off-path columns AND the shared loss-term columns to the
    weights that produce them: a journal written with ``lambda_perp > 0`` or ``lambda_c > 0``
    cannot be resumed under a configuration that prices either at zero, and vice versa. Passing
    ``None`` skips only that pairing, for callers that validate a record's shape without an
    objective in hand.
    """
    if record.is_relative is not config.is_relative:
        raise ValueError(
            "checkpoint training record schema does not match the configured target kind"
        )
    if record.is_two_state is not config.targets_two_state:
        raise ValueError(
            "checkpoint training record schema does not match the configured target origin: a "
            "two-state row records target_state and control_reporting_value beside its target, "
            "because that target is a per-draw state conversion the schedule alone cannot "
            "re-derive, and a sigma-relative row records neither"
        )
    names = ["residual_loss", "total_loss", "residual_fraction"]
    if not config.is_relative:
        names += ["target_loss", "geometry_loss"]
    elif config.targets_two_state:
        names += [
            "target_state",
            "control_reporting_value",
            "achieved_value",
            "signed_error",
            "sequence_sigma",
            "target_loss_sigma_squared",
        ]
    else:
        names += ["achieved_value", "signed_error", "sequence_sigma", "target_loss_sigma_squared"]
    for name in names:
        _finite(getattr(record, name), f"training record {name}")
    # The optional columns and the weight each is gated by. The pairing is part of the
    # PERSISTED schema, so it stays here with the reader that parses it -- but the weight is
    # looked up in the objective's own extra identity keys rather than as an attribute, so
    # this module never has to know which project defines them. A weight is in that mapping
    # exactly when it is nonzero, which is the same condition that writes the column.
    weights = {} if objective is None else objective.objective_config_extra()
    # Which off-path columns THIS property journals, or ``None`` for a property the catalog does
    # not carry. See the narrowing a few lines down for why the union alone stopped being enough,
    # and why it applies to the CONTESTED weights only.
    declaring_columns: frozenset[str] | None = None
    if config.property_name in registered_property_names():
        declaring_columns = frozenset(
            column
            for column, owners in journal_column_owners().items()
            if config.property_name in owners
        )
    for name, weight_name in _OFF_PATH_COLUMN_WEIGHTS:
        value = getattr(record, name)
        if value is not None:
            _finite(value, f"training record {name}")
        if objective is None:
            continue
        # A column's gate is looked up in the objective's own EXTENDED identity keys, never as an
        # attribute, and that is load-bearing rather than stylistic. Those keys are conditional --
        # present exactly when the thing they name is nonzero -- and, crucially, they are written
        # only by the objective that has the term. One of the five SHARED weights would be a
        # useless gate: every objective there has ever been carries a positive ``lambda_y``, so a
        # column gated on it would be demanded of every property's rows, including the rows of
        # properties that cannot compute it. A property whose driving weight IS a shared one
        # therefore restates it under its own name in ``objective_config_extra`` and gates on
        # that -- see ``DomainMotionSteeringLoss.direct_metric_weight``.
        expected = bool(weights.get(weight_name))
        # ...AND, for a CONTESTED weight only, this property has to be one that journals the
        # column. The table above is the UNION over every property, which was correct only while no
        # two of them named one weight for DIFFERENT columns. ``lambda_m`` (D-87) does: it gates the
        # cryptic ``metric_loss``/``metric_deviation_angstrom`` and the domain-motion
        # ``direct_metric_angstrom``, which are different measurements over different atom sets --
        # so without this a cryptic row at ``lambda_m > 0`` would be required to carry a column a
        # cryptic run cannot compute.
        #
        # ``contested_journal_weights()`` is exactly that set and nothing else, so every other
        # column keeps the check it has always had -- including ``perp_loss``, which the cryptic and
        # kinase families both declare against ``lambda_perp`` and both compute. An UNREGISTERED
        # property (the AF3, Local-Q and surrogate-validator paths) is left on the union too.
        if (
            expected
            and declaring_columns is not None
            and weight_name in _CONTESTED_JOURNAL_WEIGHTS
            and name not in declaring_columns
        ):
            expected = False
        if (value is not None) is not expected:
            raise ValueError(
                f"checkpoint training record {'carries' if value is not None else 'omits'} "
                f"{name} while {weight_name} is "
                f"{'zero' if not expected else 'nonzero'}; the column is gated on the weight, "
                "so this journal was written by a different objective"
            )
    # D-86's registry-declared monitors, paired with the PROPERTY'S OWN DECLARATION rather than
    # with a weight -- which is what a monitor is. A row that carries one the hook did not ask
    # for, or omits one it did, was written by a differently-configured run and is refused rather
    # than read as a narrower schema. Skipped when no objective is in hand, exactly as the weight
    # pairings above are, because the hook takes one.
    if objective is not None:
        declared_monitors = set(_declared_training_columns(config, objective))
        for name in sorted(_DECLARED_MONITOR_RECORD_KEYS):
            value = getattr(record, name)
            if value is not None:
                _finite(value, f"training record {name}")
            if (value is not None) is not (name in declared_monitors):
                raise ValueError(
                    f"checkpoint training record {'carries' if value is not None else 'omits'} "
                    f"the unpriced monitor {name} while this run's property declares "
                    f"{'no such column' if value is not None else 'it'}; a monitor is gated on "
                    "the property's own declaration, so this journal was written by a "
                    "differently-configured run"
                )
    # The shared objective's own weighted terms, on exactly the same footing. The weights are
    # read off the objective directly rather than out of ``objective_config_extra()``, because
    # unlike the off-path pair these are SHARED attributes that ``_objective_config`` records
    # for every objective there has ever been -- there is no project to ask.
    #
    # The gate is skipped where the base schema already makes the column mandatory, which is
    # the absolute path's ``geometry_loss``: it has been written unconditionally since before
    # ``lambda_g`` could be zero, so pairing it with the weight would refuse every AF3 and
    # Local-Q journal on disk.
    base = _base_record_keys(config)
    for name, weight_name in _LOSS_TERM_COLUMN_WEIGHTS:
        if name in base:
            continue
        value = getattr(record, name)
        if value is not None:
            _finite(value, f"training record {name}")
        if objective is None:
            continue
        expected = float(getattr(objective, weight_name)) > 0.0
        if (value is not None) is not expected:
            raise ValueError(
                f"checkpoint training record {'carries' if value is not None else 'omits'} "
                f"{name} while {weight_name} is "
                f"{'zero' if not expected else 'nonzero'}; the column is gated on the weight, "
                "so this journal was written by a different objective"
            )
    # The D-35b monitor columns, paired with the configuration that writes them rather than with
    # a weight, because a monitor is not in the objective. Same fail-closed shape in both
    # directions: a cryptic journal that omits one, or a radius-of-gyration journal that carries
    # one, was written by a different configuration and must not be resumed into this one.
    for name in sorted(_MONITOR_RECORD_KEYS):
        value = getattr(record, name)
        if (value is not None) is not config.steers_cryptic_pocket:
            raise ValueError(
                f"checkpoint training record {'carries' if value is not None else 'omits'} "
                f"{name} while this run "
                f"{'does not steer' if not config.steers_cryptic_pocket else 'steers'} a cryptic "
                "pocket; the column is gated on the arm, so this journal was written by a "
                "different configuration"
            )
        if value is not None:
            _monitor_count(value, name)
    if not config.is_relative:
        return
    if record.sequence_sigma != _unit_sigma(record.unit, config):
        raise ValueError("checkpoint training record sigma does not match the reference table")
    if record.is_two_state and not math.isclose(
        record.target,
        record.target_state - record.control_reporting_value,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "checkpoint training record target is not its state minus its recorded origin; the "
            "two-state target is t = the progress coordinate_state(o) - the progress coordinate_reporting(control) and the row records "
            "both halves so that identity can be checked rather than trusted"
        )
    if not math.isclose(
        record.signed_error,
        record.achieved_value - record.target,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("checkpoint training record signed error is inconsistent")


def _validate_validation_rows(
    rows: Sequence[object],
    units: Sequence[SteeringUnit],
    baselines: Mapping[str, float],
    config: OrdinarySteeringConfig,
    objective: SteeringLoss | None = None,
) -> None:
    """Fail closed unless every persisted validation row matches its configuration exactly.

    ``objective`` is what resolves the declared diagnostic columns that a WEIGHT gates rather
    than the arm -- the kinase motif channel exists only where ``lambda_dihedral`` prices it. The
    key set is compared with EQUALITY and the objective is not optional in practice: the one
    production caller always has it, because it has just constructed the loss it is validating a
    journal against. A row wider or narrower than its configuration says was written by a
    different configuration and must not be resumed into this one.
    """
    declared = set(_declared_validation_columns(config, objective))
    keys = _validation_row_keys(config.is_relative, sorted(declared), config.targets_two_state)
    for row, unit in zip(rows, units, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != keys
            or row["sequence_id"] != unit.sequence_id
            or row["seed"] != unit.seed
        ):
            raise ValueError("checkpoint validation rows do not match the schedule prefix")
        for name in sorted(set(row) - {"sequence_id", "seed"}):
            _finite(row[name], f"validation row {name}")
        expected_target = _unit_target(unit, config, baselines)
        # The re-derivable half only, for the reason ``_validate_checkpoint_progress`` states:
        # the two-state row's ``target`` also carries a per-draw measurement, so what the
        # schedule fixes is the state and the identity between the two is checked below.
        observed = row["target_state"] if config.targets_two_state else row["target"]
        if observed != expected_target:
            raise ValueError("checkpoint validation row target does not match the schedule")
        if not config.is_relative:
            continue
        if config.targets_two_state and not math.isclose(
            row["target"],
            row["target_state"] - row["control_reporting_value"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "checkpoint validation row target is not its state minus its recorded origin"
            )
        sigma = _unit_sigma(unit, config)
        if row["sequence_sigma"] != sigma:
            raise ValueError("checkpoint validation row sigma does not match the reference table")
        if not math.isclose(
            row["target_sigma_error"],
            row["target_absolute_error"] / sigma,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("checkpoint validation row sigma-normalized error is inconsistent")
        if row["target_absolute_error"] != abs(row["signed_error"]) or not math.isclose(
            row["signed_error"],
            row["achieved_value"] - row["target"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("checkpoint validation row signed error is inconsistent")


def _validate_attempt_keys(
    attempts: Mapping[str, int],
    schedule: SteeringSchedule,
    journal: Mapping[str, Any],
    cursor: int,
    config: OrdinarySteeringConfig,
) -> None:
    history_count = len(journal["history"])
    allowed = {f"train:{index}" for index in range(min(cursor + 1, len(schedule.train)))}
    allowed.update(
        f"validation:{round_index}:{index}"
        for round_index in range(history_count)
        for index in range(len(schedule.validation))
    )
    active = journal["active_validation"]
    if active:
        allowed.update(
            f"validation:{history_count}:{index}"
            for index in range(min(active["cursor"] + 1, len(schedule.validation)))
        )
    if any(
        key not in allowed or count > config.maximum_unit_attempts
        for key, count in attempts.items()
    ):
        raise ValueError("checkpoint scientific attempt state is inconsistent")


#: The top-level ``adapter_config`` keys EVERY ordinary checkpoint carries, whatever it steered.
#:
#: Extracted from :func:`validate_ordinary_completion_checkpoint` so the assembled key set is
#: reachable from a test. The gate below compares with SET EQUALITY, so this is a schema and not a
#: minimum: a key that is written and not listed here fails a run's own checkpoint at completion,
#: which is how ``cryptic_arm`` -- conditional since it was introduced, and never admitted here --
#: made a cryptic run unable to validate its own artifact.
_COMPLETION_CONFIG_KEYS = frozenset(
    {
        "workflow",
        "property_name",
        "model_family",
        "intervention_site",
        "generation_callback_identity",
        "property_callback_identity",
        "manifest_fingerprint",
        "source_checkpoints",
        "schedule_fingerprint",
        "schedule_inventory",
        "target_levels",
        "generation_batching",
        "training_seed",
        "adapter",
        "objective",
        "optimizer",
        "maximum_unit_attempts",
        "validation_target_mae_maximum",
        "validation_residual_fraction_maximum",
        "early_stopping_patience",
    }
)

#: Keys ``_adapter_config`` writes ONLY when the feature that owns them is on.
#:
#: Absent means the feature was off, which is what keeps a plain radius-of-gyration run's key set
#: byte-identical to every run made before any of these existed. Membership here is therefore not a
#: judgement about importance -- every one of these is identity-bearing when present -- it is the
#: statement that its absence is itself a valid, historical configuration.
_CONDITIONAL_COMPLETION_CONFIG_KEYS = (
    "gradient_accumulation",
    "property_ensemble",
    "diagnostic_fixed_generation_seed",
    "gradient_clip_norm",
    # D-44's two-state dose set. Absent means the continuous five-level set, which is what
    # every run before D-44 used and what radius of gyration still uses.
    "dose_levels",
    "epoch_validation_levels",
    # D-46's target origin. Absent means ``sequence_mean``, the historical behaviour.
    "relative_target_origin",
    # The cryptic arm's identity block, written by ``_cryptic_arm_identity`` exactly when a run
    # steered a cryptic pocket. It is conditional for the reason every key above is -- an
    # unconditional key would refuse every radius-of-gyration checkpoint ever written -- and it
    # was missing from this list, so a cryptic checkpoint failed its own completion gate on a
    # key its own writer had put there. ``cryptic`` is derived from the same key a few lines
    # down, so admitting it here is what makes the reader and the writer agree on ONE schema
    # rather than on two that happen to coincide for radius of gyration.
    "cryptic_arm",
    # The paired-coverage opt-out, written by ``_partial_pair_coverage_identity`` exactly when
    # the operator took it. Absent means the plan formed every pair ``lambda_c`` was weighted
    # for, which is now the only way a run without it reaches this gate at all.
    "allow_partial_pair_coverage",
    # D-23's training subset, written by ``_training_subset_identity`` exactly when an arm
    # trains on less than its whole train role. Absent means it trained on all of it, which is
    # what every run before the field existed did and what D-66's DEFAULT arm does.
    "training_subset",
    # The no-validation declaration, written by ``_model_selection_identity`` exactly when a run
    # selected on nothing. Absent means it selected on a validation role, which is what every
    # run before the field existed did. Listed here for the reason ``cryptic_arm`` had to be:
    # this gate compares with SET EQUALITY, so a key its own writer emits and this list omits
    # makes a run unable to attest the checkpoint it just finished training.
    "model_selection",
    # The declared generation-skip budget, written by ``_generation_failure_tolerance_identity``
    # exactly when a run has one. Absent means zero -- one exhausted train unit ends the run --
    # which is what every checkpoint written before the budget existed carries.
    "generation_failure_tolerance",
    # The validation column a declared selection metric averages, written by ``_target_identity``
    # exactly when the steered property declared one. Absent means the shared default, which is
    # every checkpoint on disk; see ``_completion_selection_metric`` for why the name is
    # unconditional and the column is not.
    "validation_selection_column",
    # ANY non-cryptic property's resolved-arm identity, written by ``_property_arm_identity``
    # exactly when that property's ``checkpoint_arm_arguments`` hook supplied one. Conditional for
    # the reason ``cryptic_arm`` is, and generic so that a fifth property with an arm costs a
    # descriptor line rather than an entry here.
    "property_arm",
)


def _required_completion_config_keys(
    adapter_config: Mapping[str, Any], *, relative: bool
) -> set[str]:
    """Return the exact top-level ``adapter_config`` key set a completed checkpoint must carry."""
    keys = set(_COMPLETION_CONFIG_KEYS)
    keys |= {name for name in _CONDITIONAL_COMPLETION_CONFIG_KEYS if name in adapter_config}
    # The system-constant origin replaces the reference table and its eligibility band with its own
    # per-system table, and carries no ``k``. Keyed on the presence of the table rather than on the
    # recorded origin because the table is what the substitution IS: a checkpoint carrying it and a
    # reference table too would be claiming both target constructions, which
    # ``_validate_system_constant_target_identity`` refuses by name.
    system_constant = "system_target_constants" in adapter_config
    keys |= (
        (
            {
                "target_kind",
                "property_loss_normalization",
                "validation_selection_metric",
            }
            | (
                {"system_target_constants"}
                if system_constant
                else {"property_reference", "sigma_eligibility"}
            )
        )
        # ``dose_multiplier`` exists exactly on the sigma-relative path. The two-state target has
        # no ``k`` at all, so requiring the key would refuse a two-state checkpoint for not
        # recording a number it deliberately does not have.
        #
        # Keyed on the NORMALIZATION and not on whether the key happens to be present: presence
        # would make the requirement self-satisfying, so a sigma-normalized checkpoint that lost
        # its ``k`` would pass this gate. An absent ``property_loss_normalization`` reads as the
        # sigma path, which is what every checkpoint written before that key existed is. Both
        # two-state normalisations count, because what removes ``k`` is the target construction
        # they share and not the divisor they differ in.
        | (
            set()
            if system_constant
            or adapter_config.get("property_loss_normalization") in _TWO_STATE_LOSS_NORMALIZATIONS
            else {"dose_multiplier"}
        )
        if relative
        else {
            "baseline_callback_identity",
            "baseline_path",
            "baseline_content_fingerprint",
            "target_range",
            "maximum_change",
            "target_mean",
            "target_scale",
        }
    )
    return keys


def validate_ordinary_completion_checkpoint(
    checkpoint: AdapterCheckpoint,
    checkpoint_path: str | Path,
    *,
    expected_schedule: SteeringSchedule,
) -> dict[str, Any]:
    """Validate completion against a caller-trusted full schedule."""
    if not isinstance(checkpoint, AdapterCheckpoint):
        raise TypeError("checkpoint must be an AdapterCheckpoint")
    if set(checkpoint.source_checkpoints) != _SOURCE_CHECKPOINT_ROLES:
        raise ValueError("ordinary checkpoint source roles must be exactly generator and sae")
    adapter_config = checkpoint.adapter_config
    if not isinstance(adapter_config, Mapping):
        raise ValueError("ordinary checkpoint adapter config schema is invalid")
    relative = adapter_config.get("target_kind") == RELATIVE_SEQUENCE_TARGET
    # Which journal schema the validation rows were written in, recovered from the checkpoint's
    # own identity: ``cryptic_arm`` is present exactly when this run steered a cryptic pocket
    # (see ``_cryptic_arm_identity``), which is the same condition ``_monitor_rows`` gates the
    # D-35b columns on. Deriving it here rather than assuming it keeps the reader and the writer
    # agreeing about one schema.
    cryptic = "cryptic_arm" in adapter_config
    # Which of the two RELATIVE journal schemas, recovered from the same declaration the runner
    # and the objective are both checked against. A cryptic arm is the only two-state run today,
    # but the two flags are derived separately because they are different properties: ``cryptic``
    # says which MONITOR columns exist and ``two_state`` says how the TARGET was built, and the
    # kinase programme is a two-state arm that is not a cryptic one. Membership rather than
    # equality: the two-state row schema is the same under D-64's normalisation and D-65's,
    # because what adds its columns is the state target they share.
    two_state = adapter_config.get("property_loss_normalization") in _TWO_STATE_LOSS_NORMALIZATIONS
    # Which quantity this run's epochs were compared on, recovered from the two keys the identity
    # records rather than re-derived from the property, because this gate has no objective and the
    # declaration is objective-dependent. ``None`` -- every checkpoint on disk -- is the shared
    # default and reproduces the previous behaviour exactly.
    selection_metric = _completion_selection_metric(adapter_config) if relative else None
    # Whether this run selected a checkpoint at all, recovered from its own declaration rather
    # than from the emptiness of its validation role -- reading it off the role would make every
    # check below satisfy itself. The two are then required to AGREE, a few lines down, which is
    # what makes an empty validation role without the declaration -- or the declaration beside a
    # populated role -- a refusal rather than a silently accepted third state.
    #
    # The key is written only for ``none`` -- absence IS ``validation_split`` -- so any other
    # value is a payload this code never produced and is refused rather than read as selecting.
    if "model_selection" in adapter_config and (
        adapter_config["model_selection"] != MODEL_SELECTION_NONE
    ):
        raise ValueError("ordinary checkpoint model selection declaration is invalid")
    selects_on_validation = "model_selection" not in adapter_config
    # No config on this entry point, so the dose set is recovered from the schedule the caller
    # trusts and matched against the declared sets (D-44). The `target_levels` comparison below
    # then binds the checkpoint to it, so a checkpoint written under one dose set cannot be
    # completed against a schedule built under another.
    dose_levels = _inventory_dose_levels(expected_schedule)
    required_config_keys = _required_completion_config_keys(adapter_config, relative=relative)
    if (
        set(adapter_config) != required_config_keys
        or adapter_config.get("workflow") != "ordinary"
        or adapter_config.get("manifest_fingerprint") != checkpoint.manifest_fingerprint
        or adapter_config.get("property_callback_identity") != checkpoint.property_callback_identity
        or adapter_config.get("source_checkpoints") != dict(checkpoint.source_checkpoints)
        or adapter_config.get("target_levels") != list(dose_levels.target)
    ):
        raise ValueError("ordinary checkpoint adapter config schema is invalid")
    fingerprint_names = ["manifest_fingerprint", "schedule_fingerprint"]
    if not relative:
        fingerprint_names.append("baseline_content_fingerprint")
    for name in fingerprint_names:
        _fingerprint(adapter_config[name], f"ordinary adapter config {name}")
    # The declaration and the role have to agree, in BOTH directions: a run that selected on
    # nothing must carry no validation units, and a run that carries none must have said so.
    # Checked before the schedule is parsed, because the parse itself is what the declaration
    # authorises.
    if bool(expected_schedule.validation) is not selects_on_validation:
        raise ValueError(
            "ordinary checkpoint model selection declaration does not match its validation "
            "role: a checkpoint declaring model_selection "
            f"{MODEL_SELECTION_NONE!r} must carry no validation units, and one declaring none "
            "must carry them"
        )
    schedule = _schedule_from_inventory(
        adapter_config["schedule_inventory"], expect_validation=selects_on_validation
    )
    if _schedule_fingerprint(schedule) != adapter_config["schedule_fingerprint"]:
        raise ValueError("ordinary checkpoint schedule inventory fingerprint is invalid")
    _validate_schedule(expected_schedule, dose_levels, expect_validation=selects_on_validation)
    if schedule != expected_schedule or adapter_config[
        "schedule_fingerprint"
    ] != _schedule_fingerprint(expected_schedule):
        raise ValueError("ordinary checkpoint does not match the trusted full schedule")
    _validate_generation_batching_identity(adapter_config["generation_batching"], schedule)
    _validate_optimizer_step_identity(adapter_config, schedule)
    _validate_generation_precision_identity(adapter_config)
    if relative:
        _validate_relative_target_identity(adapter_config)
    else:
        baseline_path = adapter_config["baseline_path"]
        if not isinstance(baseline_path, str) or not Path(baseline_path).is_absolute():
            raise ValueError("ordinary checkpoint baseline path is invalid")
    journal = _journal(checkpoint.validation_record)
    records = _records(journal["training_records"])
    # How many scheduled train units this run walked but never trained on, because their generation
    # was tolerated away. Zero on every checkpoint written before the tolerance existed, so every
    # comparison below reduces to the one it replaced.
    skipped_units = _skipped_unit_count(journal)
    if skipped_units and "generation_failure_tolerance" not in adapter_config:
        raise ValueError(
            "ordinary checkpoint skipped training units while declaring no generation-failure "
            "budget; the skips and the permission to take them are one statement"
        )
    history = journal["history"]
    best_index = journal["best_validation_index"]
    if (
        checkpoint.unit_cursor != len(records) + skipped_units
        or not records
        or checkpoint.epoch != len(history)
        or journal["active_validation"]
        or journal["in_flight"] is not None
        or journal["terminal_reason"] not in {"early_stopped", "max_cycles_reached"}
        or journal["selected_best_model_optimizer_fingerprint"]
        != journal["best_model_optimizer_fingerprint"]
    ):
        raise ValueError("ordinary checkpoint validation completion journal is invalid")
    # Split out of the block above, because the two regimes require OPPOSITE things of the same
    # four fields and an ``or`` chain that satisfied both would require nothing of either.
    #
    # A selecting run must show a history and an in-range best index; the equality of the
    # selected and best fingerprints is already checked above and is what pins the restored
    # state. A non-selecting run must show the whole block EMPTY -- no history, no index, no
    # best-checkpoint path, no fingerprint and a bad-round counter of zero -- so an attestation
    # can never be produced from a fabricated round. (The fingerprint equality above holds
    # trivially there, both sides being ``None``, which is exactly right: no state was restored.)
    if selects_on_validation:
        if (
            not history
            or not isinstance(best_index, int)
            or isinstance(best_index, bool)
            or not 0 <= best_index < len(history)
        ):
            raise ValueError("ordinary checkpoint validation completion journal is invalid")
    elif (
        history
        or best_index is not None
        or journal["best_checkpoint_path"] is not None
        or journal["best_model_optimizer_fingerprint"] is not None
        or journal["bad_validation_count"] != 0
        or journal["validation_gate_failure"] is not None
    ):
        raise ValueError(
            "ordinary checkpoint declares no model selection but carries validation selection "
            "state; a run with no validation role runs no round, selects no checkpoint and "
            "applies no gate"
        )
    # ``passed``/``workflow_complete`` are False for two very different reasons, and only one
    # of them means the checkpoint is unusable. Training that stopped short is genuinely
    # incomplete. Training that ran to a terminal reason and then failed the residual gate is
    # a *scientific verdict about the adapter*, not a damaged artifact: the schedule was
    # walked in full, the best round was selected, and the state was restored. Refusing to
    # evaluate it discards the held-out measurement that would say what such an adapter
    # actually does -- which is the question the gate exists to raise, not to suppress.
    #
    # So a gate failure is admitted, and only a gate failure: everything above still had to
    # hold, and the caller is handed the recorded failure so it can never be reported as a
    # clean pass (the project's guidelines forbids presenting a partial result as complete).
    gate_failure = journal.get("validation_gate_failure")
    if journal["passed"] is not True or journal["workflow_complete"] is not True:
        if not isinstance(gate_failure, Mapping) or not gate_failure:
            raise ValueError("ordinary checkpoint validation completion journal is invalid")
    if any(record.unit.split != "train" for record in records) or len(
        {record.unit.seed for record in records}
    ) != len(records):
        raise ValueError("ordinary checkpoint completion training records are invalid")
    if any(record.is_relative is not relative for record in records):
        raise ValueError(
            "ordinary checkpoint completion training records do not match the target kind"
        )
    if any(record.is_two_state is not two_state for record in records):
        raise ValueError(
            "ordinary checkpoint completion training records do not match the target origin"
        )
    if tuple(record.unit for record in records) != _trained_prefix(
        schedule.train, checkpoint.unit_cursor, journal
    ):
        raise ValueError("ordinary checkpoint completion training schedule prefix is invalid")
    # Every check from here to the terminal-reason block describes the SELECTION: which rounds
    # ran, which one won, and that the state on disk is that round's. A run with no validation
    # role made none of those statements, and the block above has already required it to carry
    # none of that state, so there is nothing here to check rather than something to relax.
    #
    # In particular the shipped checkpoint is the FINAL one, which is why no best-checkpoint file
    # is loaded and no state fingerprint is compared against a selected round: there is no
    # separate artifact for it to be compared to, and inventing one would be the fabrication this
    # regime exists to avoid.
    if selects_on_validation:
        previous_cursor = 0
        validation_signature = None
        boundaries = _cycle_boundaries(schedule.train)
        for index, record in enumerate(history):
            signature = _validate_completion_history_record(
                record,
                index,
                relative,
                cryptic,
                two_state,
                _completion_row_vocabulary(adapter_config),
                selection_metric,
            )
            if (
                index >= len(boundaries)
                or record["train_cursor"] != boundaries[index]
                or record["train_cursor"] <= previous_cursor
                # The SCHEDULE cursor a round was taken at, which is what ``_cycle_boundaries``
                # enumerates and what a skipped group advances past. Comparing it against the
                # record count instead would refuse every round taken after a skip.
                or record["train_cursor"] > checkpoint.unit_cursor
            ):
                raise ValueError("ordinary checkpoint completion validation boundaries are invalid")
            previous_cursor = record["train_cursor"]
            if tuple((sequence_id, seed) for sequence_id, seed, _target in signature) != tuple(
                (unit.sequence_id, unit.seed) for unit in schedule.validation
            ):
                raise ValueError("ordinary checkpoint completion validation schedule is invalid")
            if validation_signature is None:
                validation_signature = signature
            elif signature != validation_signature:
                raise ValueError("ordinary checkpoint completion validation inventory changed")
        if previous_cursor != checkpoint.unit_cursor:
            raise ValueError("ordinary checkpoint completion omitted final validation")
        expected_best = min(
            range(len(history)),
            key=lambda index: _validation_key(
                history[index],
                relative,
                None if selection_metric is None else selection_metric.name,
            ),
        )
        if (
            best_index != expected_best
            or journal["bad_validation_count"] != len(history) - expected_best - 1
        ):
            raise ValueError("ordinary checkpoint completion best validation state is invalid")
        best_fingerprint = journal["best_model_optimizer_fingerprint"]
        _fingerprint(best_fingerprint, "ordinary checkpoint best model optimizer fingerprint")
        destination = Path(checkpoint_path).resolve()
        best_path = _best_checkpoint_path(destination, best_index)
        if Path(journal["best_checkpoint_path"]) != best_path:
            raise ValueError("ordinary checkpoint completion best path is invalid")
        best = load_adapter_checkpoint(
            best_path,
            expected_manifest_fingerprint=checkpoint.manifest_fingerprint,
            expected_adapter_config=checkpoint.adapter_config,
            expected_property_callback_identity=checkpoint.property_callback_identity,
            expected_source_checkpoints=checkpoint.source_checkpoints,
        )
        if _model_optimizer_fingerprint(best) != best_fingerprint:
            raise ValueError("ordinary checkpoint completion best fingerprint is invalid")
        if (
            _model_optimizer_fingerprint(checkpoint)
            != journal["selected_best_model_optimizer_fingerprint"]
        ):
            raise ValueError(
                "ordinary checkpoint final state does not match the selected best state"
            )
    if journal["terminal_reason"] == "max_cycles_reached" and (
        journal["early_stopped"]
        or journal["not_run_after_early_stopping"]
        or checkpoint.unit_cursor != len(schedule.train)
    ):
        raise ValueError("ordinary checkpoint max-cycle terminal journal is invalid")
    if journal["terminal_reason"] == "early_stopped" and journal["early_stopped"] is not True:
        raise ValueError("ordinary checkpoint early-stop terminal journal is invalid")
    patience = adapter_config["early_stopping_patience"]
    if journal["terminal_reason"] == "early_stopped" and (
        not isinstance(patience, int)
        or isinstance(patience, bool)
        or patience < 1
        or journal["bad_validation_count"] < patience
    ):
        raise ValueError("ordinary checkpoint early-stop patience is invalid")
    try:
        skipped = tuple(SteeringUnit(**row) for row in journal["not_run_after_early_stopping"])
    except (TypeError, ValueError) as error:
        raise ValueError("ordinary checkpoint early-stop schedule inventory is invalid") from error
    if (
        journal["terminal_reason"] == "early_stopped"
        and skipped != schedule.train[checkpoint.unit_cursor :]
    ):
        raise ValueError("ordinary checkpoint early-stop schedule suffix is invalid")
    return journal


def _validate_generation_batching_identity(value: object, schedule: SteeringSchedule) -> None:
    """Fail closed unless recorded batching covers the schedule with bounded batches."""
    if value is None:
        return
    expected = {
        "rule",
        "config",
        "train_batch_sizes",
        "validation_batch_sizes",
        "train_batch_count",
        "validation_batch_count",
        "maximum_train_batch_size",
        "maximum_validation_batch_size",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("ordinary checkpoint generation batching identity is invalid")
    if value["rule"] not in (LENGTH_BUCKETED_RULE, UNBATCHED_RULE):
        raise ValueError("ordinary checkpoint generation batching rule is invalid")
    config = value["config"]
    config_keys = {
        "batch_size_100",
        "maximum_batch_size",
        "bucket_window_batches",
        "gradient_graph_divisor",
    }
    # `keep_draw_groups_whole` is OPTIONAL and must be accepted here, because
    # `GenerationBatchConfig.identity` emits it CONDITIONALLY -- present only when the flag is on,
    # deliberately absent otherwise so a plan made before the flag existed keeps a byte-identical
    # identity. This validator was written against the four unconditional keys and demanded set
    # equality, so every checkpoint written with the flag ON was refused ON READ-BACK, after
    # training had already succeeded. That cost a 17-hour arm (a measured run) at its last gate: the
    # adapter, both validation-selected checkpoints and the journals were all on disk and correct,
    # and only this check stood between them and an evaluation.
    #
    # Accepted rather than required: absent is the pre-flag planner and is equally valid. The value
    # is checked as a real bool, since `identity` only ever emits True and a False would mean a
    # payload this code did not write.
    optional_keys = {"keep_draw_groups_whole"}
    if (
        not isinstance(config, Mapping)
        or not config_keys <= set(config)
        or not set(config) <= config_keys | optional_keys
        or any(
            isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1
            for name in config_keys
        )
        or not all(isinstance(config[name], bool) for name in optional_keys & set(config))
    ):
        raise ValueError("ordinary checkpoint generation batching configuration is invalid")
    for name, units, count_name, maximum_name in (
        ("train_batch_sizes", schedule.train, "train_batch_count", "maximum_train_batch_size"),
        (
            "validation_batch_sizes",
            schedule.validation,
            "validation_batch_count",
            "maximum_validation_batch_size",
        ),
    ):
        sizes = value[name]
        if (
            not isinstance(sizes, list)
            or any(
                isinstance(size, bool)
                or not isinstance(size, int)
                or not 1 <= size <= config["maximum_batch_size"]
                for size in sizes
            )
            or sum(sizes) != len(units)
            or value[count_name] != len(sizes)
            or value[maximum_name] != max(sizes, default=0)
        ):
            raise ValueError(f"ordinary checkpoint {name} do not cover the schedule")


def _validate_optimizer_step_identity(
    adapter_config: Mapping[str, Any], schedule: SteeringSchedule
) -> None:
    """Fail closed unless the conditional optimizer-step keys describe this schedule.

    Presence is meaning here, so a fabricated key is refused rather than tolerated: a
    ``gradient_accumulation`` block declaring one micro-step per step would claim
    accumulation for a run that took none, and a fixed-seed block naming seeds the
    schedule does not contain would claim a diagnostic that never ran.
    """
    accumulation = adapter_config.get("gradient_accumulation")
    if accumulation is not None:
        expected = {
            "units",
            "rule",
            "weighting",
            "optimizer_step_count",
            "maximum_optimizer_step_units",
            "minimum_optimizer_step_units",
            "target_optimizer_step_units",
            "optimizer_step_unit_counts",
            "train_group_batch_sizes",
            "train_group_unit_sizes",
        }
        if not isinstance(accumulation, Mapping) or set(accumulation) != expected:
            raise ValueError("ordinary checkpoint gradient accumulation identity is invalid")
        units = accumulation["units"]
        # ONE micro-batch per step is now legitimate and says nothing about the step's size: a
        # 32-unit step over a short-protein window is one batch of 32, and over a long-protein
        # window two of 16. What the step covers is ``train_group_unit_sizes``, checked below
        # against the schedule, so the old "at least two micro-steps" refusal -- which existed
        # to stop a block being recorded for a run that accumulated nothing -- would now refuse
        # correct plans.
        if not isinstance(units, int) or isinstance(units, bool) or units < 1:
            raise ValueError("ordinary checkpoint gradient accumulation identity is invalid")
        if (
            accumulation["rule"] != UNIFORM_STEP_RULE
            or accumulation["weighting"] != ACCUMULATION_WEIGHTING
            or accumulation["target_optimizer_step_units"] != UNIFORM_OPTIMIZER_STEP_UNITS
        ):
            raise ValueError("ordinary checkpoint gradient accumulation convention is invalid")
        groups = accumulation["train_group_batch_sizes"]
        sizes = accumulation["train_group_unit_sizes"]
        if (
            not isinstance(groups, list)
            or not isinstance(sizes, list)
            or len(groups) != len(sizes)
            or accumulation["optimizer_step_count"] != len(groups)
            or any(
                not isinstance(group, list)
                or not group
                or len(group) > units
                or any(
                    isinstance(size, bool) or not isinstance(size, int) or size < 1
                    for size in group
                )
                for group in groups
            )
            or [sum(group) for group in groups] != sizes
            or sum(sizes) != len(schedule.train)
            or accumulation["maximum_optimizer_step_units"] != max(sizes, default=0)
            or accumulation["minimum_optimizer_step_units"] != min(sizes, default=0)
            # The realised histogram, recomputed rather than trusted: it is the record a
            # reader checks the uniform rule against, so a hand-edited one claiming 32 units
            # everywhere over groups that are not 32 units must be refused.
            or accumulation["optimizer_step_unit_counts"]
            != [[size, sizes.count(size)] for size in sorted(set(sizes))]
        ):
            raise ValueError(
                "ordinary checkpoint gradient accumulation groups do not cover the schedule"
            )
    ensemble = adapter_config.get("property_ensemble")
    if ensemble is not None:
        expected_keys = {
            "samples",
            "estimator",
            "residual_reduction",
            "seed_rule",
            "seed_fingerprint",
        }
        if not isinstance(ensemble, Mapping) or set(ensemble) != expected_keys:
            raise ValueError("ordinary checkpoint property ensemble identity is invalid")
        samples = ensemble["samples"]
        if not isinstance(samples, int) or isinstance(samples, bool) or samples < 2:
            raise ValueError(
                "a recorded property_ensemble block must declare at least two structures "
                "per scheduled unit; a run whose property is one structure records no "
                "block at all"
            )
        if (
            ensemble["estimator"] != ENSEMBLE_MEAN_PROPERTY_ESTIMATOR
            or ensemble["residual_reduction"] != ENSEMBLE_RESIDUAL_REDUCTION
            or ensemble["seed_rule"] != PROPERTY_ENSEMBLE_SEED_RULE
        ):
            raise ValueError("ordinary checkpoint property ensemble convention is invalid")
        # Recomputed, not trusted: the fingerprint is what binds every member seed, so a
        # recorded digest that the schedule does not reproduce means the checkpoint's
        # optimizer steps averaged structures this schedule never asks for.
        if ensemble["seed_fingerprint"] != _property_ensemble_seed_fingerprint(
            build_property_ensemble_seeds(schedule, samples)
        ):
            raise ValueError(
                "ordinary checkpoint property ensemble seeds do not match the schedule"
            )
    diagnostic = adapter_config.get("diagnostic_fixed_generation_seed")
    if diagnostic is None:
        return
    if not isinstance(diagnostic, Mapping) or set(diagnostic) != {
        "scope",
        "rule",
        "sequence_seeds",
    }:
        raise ValueError("ordinary checkpoint fixed-seed diagnostic identity is invalid")
    if (
        diagnostic["scope"] != FIXED_GENERATION_SEED_SCOPE
        or diagnostic["rule"] != FIXED_GENERATION_SEED_RULE
    ):
        raise ValueError("ordinary checkpoint fixed-seed diagnostic convention is invalid")
    expected_seeds: dict[str, int] = {}
    for unit in schedule.train:
        expected_seeds.setdefault(unit.sequence_id, unit.seed)
    if diagnostic["sequence_seeds"] != dict(sorted(expected_seeds.items())):
        raise ValueError(
            "ordinary checkpoint fixed-seed diagnostic seeds do not match the schedule"
        )


def _validate_relative_target_identity(adapter_config: Mapping[str, Any]) -> None:
    """Fail closed unless a relative checkpoint names exactly what produced its targets."""
    if "system_target_constants" in adapter_config:
        _validate_system_constant_target_identity(adapter_config)
        return
    reference = adapter_config["property_reference"]
    expected_keys = {
        "property_name",
        "units",
        "source_format",
        "sequence_count",
        "file_sha256",
        "content_fingerprint",
    }
    if not isinstance(reference, Mapping) or set(reference) != expected_keys:
        raise ValueError("ordinary checkpoint property reference identity is invalid")
    for name in ("file_sha256", "content_fingerprint"):
        _fingerprint(reference[name], f"ordinary checkpoint property reference {name}")
    if reference["property_name"] != adapter_config["property_name"]:
        raise ValueError("ordinary checkpoint property reference property does not match")
    for name in ("units", "source_format"):
        if not isinstance(reference[name], str) or not reference[name]:
            raise ValueError("ordinary checkpoint property reference identity is invalid")
    if (
        not isinstance(reference["sequence_count"], int)
        or isinstance(reference["sequence_count"], bool)
        or reference["sequence_count"] < 1
    ):
        raise ValueError("ordinary checkpoint property reference sequence count is invalid")
    normalization = adapter_config["property_loss_normalization"]
    if normalization in _TWO_STATE_LOSS_NORMALIZATIONS:
        # No ``k`` on this path, so its absence is checked rather than its value. A checkpoint
        # carrying both would be claiming a sigma-relative dose under a two-state objective,
        # which is two target constructions in one identity.
        if "dose_multiplier" in adapter_config:
            raise ValueError(
                "a two-state checkpoint records no dose_multiplier: level +1 IS the "
                "holo / DFG-out state, so there is no multiple of a standard deviation to record"
            )
    elif normalization == SIGMA_LOSS_NORMALIZATION:
        if "dose_multiplier" not in adapter_config:
            raise ValueError(
                "a sigma-normalized checkpoint must record its dose_multiplier: the target is "
                "level * k * sigma_seq and k is half of what produced it"
            )
        if _finite(adapter_config["dose_multiplier"], "ordinary dose_multiplier") <= 0:
            raise ValueError("ordinary checkpoint dose_multiplier must be positive")
    else:
        raise ValueError(
            "relative_sequence checkpoints must record either the sigma-normalized property "
            f"loss or one of {sorted(_TWO_STATE_LOSS_NORMALIZATIONS)}"
        )
    eligibility = adapter_config["sigma_eligibility"]
    if not isinstance(eligibility, Mapping) or set(eligibility) != {"lower", "upper"}:
        raise ValueError("ordinary checkpoint sigma eligibility identity is invalid")
    SigmaEligibility(
        _finite(eligibility["lower"], "ordinary sigma eligibility lower"),
        _finite(eligibility["upper"], "ordinary sigma eligibility upper"),
    )
    _validate_recorded_selection_metric(adapter_config)


def _validate_system_constant_target_identity(adapter_config: Mapping[str, Any]) -> None:
    """Fail closed unless a SYSTEM-CONSTANT checkpoint names exactly what produced its targets.

    The twin of the reference-table check above, over the table that replaced it. Three statements,
    and each of them is the fail-closed direction of a way this origin could be confused with a
    sigma-relative run:

    1. **No reference table and no eligibility band.** Their presence would mean a checkpoint whose
       targets came from a per-system threshold while its identity advertises an ensemble table it
       never read.
    2. **No ``dose_multiplier``.** There is no sigma in this target, so a ``k`` recorded beside it
       would be a dose that applied to nothing -- the same reason a two-state checkpoint records
       none.
    3. **The table's own digest and system count.** They are per-protein weights on the objective,
       so a checkpoint has to be bound to the table it descended; two runs over one split with two
       thresholds are two objectives.
    """
    for absent in ("property_reference", "sigma_eligibility", "dose_multiplier"):
        if absent in adapter_config:
            raise ValueError(
                f"a {TARGET_ORIGIN_SYSTEM_CONSTANT!r} checkpoint records no {absent}: its target "
                "is this system's own threshold and its property loss scale is that system's own "
                f"reference separation, so a recorded {absent} would describe an input the run "
                "never read"
            )
    table = adapter_config["system_target_constants"]
    if not isinstance(table, Mapping) or set(table) != {"systems", "content_fingerprint"}:
        raise ValueError("ordinary checkpoint system target constants identity is invalid")
    if (
        not isinstance(table["systems"], int)
        or isinstance(table["systems"], bool)
        or table["systems"] < 1
    ):
        raise ValueError("ordinary checkpoint system target constants system count is invalid")
    _fingerprint(
        table["content_fingerprint"], "ordinary checkpoint system target constants fingerprint"
    )
    if adapter_config["property_loss_normalization"] != SIGMA_LOSS_NORMALIZATION:
        raise ValueError(
            f"a {TARGET_ORIGIN_SYSTEM_CONSTANT!r} checkpoint divides the per-observation error by "
            f"its own system's scale, which is the {SIGMA_LOSS_NORMALIZATION!r} contract: a "
            "required, strictly positive, per-observation divisor on the absolute channel"
        )
    _validate_recorded_selection_metric(adapter_config)


def _validate_recorded_selection_metric(adapter_config: Mapping[str, Any]) -> None:
    """Fail closed unless a relative checkpoint names a selection metric someone admits.

    It used to be an equality against :data:`RELATIVE_SELECTION_METRIC`, which was exactly right
    while the metric was a function of the target kind. It is a PROPERTY's declaration now, so the
    check is membership in the shared default plus the vocabulary the catalog admits -- which is
    still a closed set, and still refuses an arbitrary string, but no longer refuses a
    direct-metric arm for recording the quantity it actually selected on.

    The vocabulary is read from the catalog and not from a descriptor, because this gate holds a
    persisted ``adapter_config`` and cannot reconstruct the objective that chose the metric. That
    is the whole reason
    :attr:`~sae_bioemu.steering.property_registry.PropertyRegistration.selection_metric_names`
    exists beside the behavioural hook.
    """
    recorded = adapter_config["validation_selection_metric"]
    if recorded == RELATIVE_SELECTION_METRIC:
        return
    if recorded in selection_metric_names():
        return
    raise ValueError(
        f"a relative_sequence checkpoint records validation_selection_metric={recorded!r}, which "
        f"is neither the shared {RELATIVE_SELECTION_METRIC!r} nor any metric a registered property "
        f"declares ({sorted(selection_metric_names())}). A checkpoint selected on a quantity no "
        "reader can name cannot be attested"
    )


def _completion_selection_metric(
    adapter_config: Mapping[str, Any],
) -> ValidationSelectionMetric | None:
    """Rebuild the declared selection metric from a persisted ``adapter_config``, or ``None``.

    The two halves are recorded separately -- the NAME unconditionally, because every relative
    checkpoint has always carried that key, and the COLUMN conditionally, because no checkpoint
    before this one had a column to carry. ``None`` here therefore means "the shared default",
    which is every checkpoint on disk and every run that declares nothing.

    Refuses the two half-configurations rather than guessing at either. A column without a
    non-default name would name the mean of a column nothing selects on; a non-default name
    without a column would leave the gate unable to re-derive the aggregate it is validating, which
    is a gate that checks the number it was handed against itself.
    """
    name = adapter_config["validation_selection_metric"]
    column = adapter_config.get("validation_selection_column")
    default = name in (ABSOLUTE_SELECTION_METRIC, RELATIVE_SELECTION_METRIC)
    if default and column is None:
        return None
    if default or column is None:
        raise ValueError(
            "an ordinary checkpoint records validation_selection_metric="
            f"{name!r} and validation_selection_column={column!r}; a declared selection metric "
            "carries both -- the aggregate key and the validation column it is the mean of -- so "
            "one without the other is a selection nothing can re-derive"
        )
    if column not in _completion_row_vocabulary(adapter_config):
        raise ValueError(
            f"an ordinary checkpoint selects on the validation column {column!r}, which is not in "
            "the steered property's declared validation_row_vocabulary. The rows it selected on "
            "are therefore not rows this property is able to write"
        )
    return ValidationSelectionMetric(name=str(name), row_column=str(column))


def _completion_row_vocabulary(adapter_config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the declared validation-column vocabulary of the property this checkpoint steered.

    Empty for an absent or unregistered property name rather than raising, for the reason
    :func:`_declared_validation_columns` gives: this runner is shared with the AF3, Local-Q and
    surrogate-validator paths, whose property names are deliberately not in the steerable-property
    catalog, and a property with no registration declares no diagnostic column to admit.
    """
    name = adapter_config.get("property_name")
    if not isinstance(name, str) or name not in registered_property_names():
        return ()
    return tuple(_resolve_property(name).validation_row_vocabulary)


def _validate_completion_history_record(
    record: object,
    index: int,
    relative: bool = False,
    cryptic: bool = False,
    two_state: bool = False,
    vocabulary: Sequence[str] = (),
    selection_metric: ValidationSelectionMetric | None = None,
) -> tuple[tuple[str, int, float], ...]:
    if (
        not _validation_record_schema_is_valid(
            record, relative, None if selection_metric is None else selection_metric.name
        )
        or record["validation_index"] != index
        or not isinstance(record["train_cursor"], int)
        or isinstance(record["train_cursor"], bool)
        or not isinstance(record["rows"], list)
        or not record["rows"]
        or record["observation_count"] != len(record["rows"])
    ):
        raise ValueError("ordinary checkpoint completion validation history schema is invalid")
    # CONTAINMENT plus a bounded surplus, rather than set equality. This gate reads a persisted
    # ``adapter_config`` and cannot reconstruct the objective, so it cannot know which of a
    # property's weight-gated diagnostic columns a given arm switched on. ``vocabulary`` is that
    # property's own static declaration, so an arbitrary extra key is still refused; the exact
    # per-configuration pairing is checked by ``_validate_validation_rows``, which has the
    # configuration and the objective in hand.
    row_keys = _validation_row_keys(relative, (), two_state)
    admissible = row_keys | set(vocabulary) | (set(_MONITOR_RECORD_KEYS) if cryptic else set())
    signature = []
    for row in record["rows"]:
        if not isinstance(row, dict) or not row_keys <= set(row) or not set(row) <= admissible:
            raise ValueError("ordinary checkpoint completion validation row schema is invalid")
        for name in sorted(set(row) - {"sequence_id", "seed"}):
            _finite(row[name], f"ordinary completion validation {name}")
        if (
            not isinstance(row["sequence_id"], str)
            or not row["sequence_id"]
            or not isinstance(row["seed"], int)
        ):
            raise ValueError("ordinary checkpoint completion validation row identity is invalid")
        signature.append((row["sequence_id"], row["seed"], row["target"]))
    expected = _validation_aggregate(
        record["rows"], index, record["train_cursor"], relative, selection_metric
    )
    if any(record[name] != expected[name] for name in expected if name != "rows"):
        raise ValueError("ordinary checkpoint completion validation aggregate is invalid")
    return tuple(signature)


def _validate_attempt_failure_consistency(
    journal: Mapping[str, Any], schedule: SteeringSchedule
) -> None:
    failures = _failures(journal["failures"])
    scientific_keys = []
    for failure, key in zip(failures, journal["failure_keys"], strict=True):
        if failure.phase == "baseline":
            if key != f"baseline:{failure.sequence_id}":
                raise ValueError("checkpoint baseline failure attempt key is inconsistent")
            continue
        expected_prefix = "validation:" if failure.phase == "validation" else "train:"
        if not key.startswith(expected_prefix):
            raise ValueError("checkpoint scientific failure attempt key is inconsistent")
        scientific_keys.append(key)
    if Counter(scientific_keys) != Counter(journal["attempts"]):
        raise ValueError("checkpoint scientific attempt counts do not match failure keys")
    if any(journal["attempts"][key] != count for key, count in Counter(scientific_keys).items()):
        raise ValueError("checkpoint training scientific attempts do not match recorded failures")


def _validate_failure_schedule(
    values: object,
    schedule: SteeringSchedule,
    baselines: Mapping[str, float],
    config: OrdinarySteeringConfig,
) -> None:
    """Fail closed unless every recorded failure matches the unit the schedule puts at its seed.

    ``SteeringFailure.target`` is whatever :func:`_unit_target` produced, which on the two-state
    path is the reporting-frame STATE and not the training-frame target. That is deliberate and
    is the only honest value: a failure is a draw that does not exist, so it has no matched
    control and therefore no origin, and the training-frame target is undefined rather than
    unrecorded. The re-derivation below is exact for the same reason -- the state is a pure
    function of the level.
    """
    train_by_seed = {unit.seed: unit for unit in schedule.train}
    validation_by_seed = {unit.seed: unit for unit in schedule.validation}
    baseline_by_seed = {unit.seed: unit for unit in _sequence_representatives(schedule).values()}
    for failure in _failures(values):
        if failure.phase == "baseline":
            if config.is_relative:
                raise ValueError("relative_sequence checkpoints cannot contain baseline failures")
            unit = baseline_by_seed.get(failure.seed)
            expected_target = None if unit is None else unit.level
        elif failure.phase in {"generation", "training"}:
            unit = train_by_seed.get(failure.seed)
            expected_target = None if unit is None else _unit_target(unit, config, baselines)
        elif failure.phase in {"validation", _VALIDATION_GENERATION_PHASE}:
            # Both resolve against the VALIDATION split. The generation half is spelled separately
            # only so the tolerance can tell a bad structure from a defect in the run; which split
            # the draw came from is the same question for both.
            unit = validation_by_seed.get(failure.seed)
            expected_target = None if unit is None else _unit_target(unit, config, baselines)
        else:
            raise ValueError("checkpoint failure phase is invalid")
        if (
            unit is None
            or failure.sequence_id != unit.sequence_id
            or failure.target != expected_target
        ):
            raise ValueError("checkpoint failure does not match its scheduled unit")


def _validate_in_flight(
    in_flight: object,
    schedule: SteeringSchedule,
    journal: Mapping[str, Any],
    due: int,
    config: OrdinarySteeringConfig,
) -> None:
    if in_flight is None:
        return
    if not isinstance(in_flight, dict) or in_flight.get("phase") not in {"train", "validation"}:
        raise ValueError("checkpoint in-flight state is invalid")
    # The SCHEDULE cursor, which is the record count plus whatever this run skipped: the in-flight
    # key names a position in the schedule and a skipped group advances that position without
    # leaving a record behind. Identical to the record count on every run that skipped nothing.
    cursor = len(journal["training_records"]) + _skipped_unit_count(journal)
    if in_flight["phase"] == "train":
        expected = {"phase": "train", "key": f"train:{cursor}", "cursor": cursor}
        if in_flight != expected or due != len(journal["history"]):
            raise ValueError("checkpoint in-flight training unit is inconsistent")
    else:
        active = journal["active_validation"]
        validation_cursor = active.get("cursor")
        expected = {
            "phase": "validation",
            "key": f"validation:{len(journal['history'])}:{validation_cursor}",
            "round_index": len(journal["history"]),
            "cursor": validation_cursor,
        }
        if in_flight != expected:
            raise ValueError("checkpoint in-flight validation unit is inconsistent")
    if _attempts(journal, in_flight["key"]) >= config.maximum_unit_attempts:
        raise ValueError("checkpoint in-flight unit exceeds the scientific retry budget")


def _make_checkpoint(
    config: OrdinarySteeringConfig,
    identity: Mapping[str, Any],
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    cursor: int,
    journal: Mapping[str, Any],
) -> AdapterCheckpoint:
    _validate_optimizer_identity(optimizer, adapter, identity["optimizer"])
    return AdapterCheckpoint(
        config.manifest_fingerprint,
        copy.deepcopy(dict(identity)),
        config.property_callback_identity,
        dict(config.source_checkpoints),
        _cpu_state(adapter),
        copy.deepcopy(optimizer.state_dict()),
        epoch,
        cursor,
        copy.deepcopy(dict(journal)),
    )


def _result(
    checkpoint: AdapterCheckpoint,
    checkpoint_path: Path,
    baselines: Mapping[str, float],
    failures: tuple[SteeringFailure, ...],
    journal: Mapping[str, Any],
    *,
    complete: bool,
) -> OrdinaryTrainingResult:
    return OrdinaryTrainingResult(
        _records(journal["training_records"]),
        failures,
        tuple(BaselineRecord(key, value) for key, value in sorted(baselines.items())),
        checkpoint,
        copy.deepcopy(dict(journal)),
        complete,
        checkpoint_path,
        tuple(copy.deepcopy(dict(row)) for row in _skipped_units(journal)),
    )


def _record(record: SteeringTrainingRecord) -> dict[str, Any]:
    """Serialize one record in exactly the key set its target construction defines."""
    row: dict[str, Any] = {
        "unit": asdict(record.unit),
        "target": record.target,
        "residual_loss": record.residual_loss,
        "total_loss": record.total_loss,
        "residual_fraction": record.residual_fraction,
    }
    if record.is_two_state:
        row.update(
            target_state=record.target_state,
            control_reporting_value=record.control_reporting_value,
            achieved_value=record.achieved_value,
            signed_error=record.signed_error,
            sequence_sigma=record.sequence_sigma,
            target_loss_sigma_squared=record.target_loss_sigma_squared,
        )
    elif record.is_relative:
        row.update(
            achieved_value=record.achieved_value,
            signed_error=record.signed_error,
            sequence_sigma=record.sequence_sigma,
            target_loss_sigma_squared=record.target_loss_sigma_squared,
        )
    else:
        row.update(target_loss=record.target_loss, geometry_loss=record.geometry_loss)
    # Emitted only when the term ran. A column of ``null`` would be an unconditional key by
    # another name and would break every historical journal's key set just as loudly.
    for name, _ in _OFF_PATH_COLUMN_WEIGHTS:
        value = getattr(record, name)
        if value is not None:
            row[name] = value
    # Emitted only on a cryptic-pocket arm, for the same reason: a column of ``null`` is an
    # unconditional key by another name and would break every historical journal's key set.
    for name in sorted(_MONITOR_RECORD_KEYS):
        value = getattr(record, name)
        if value is not None:
            row[name] = value
    # The registry-declared UNPRICED monitors (D-86), emitted only where the property's own hook
    # asked for them, for the same reason again.
    for name in sorted(_DECLARED_MONITOR_RECORD_KEYS):
        value = getattr(record, name)
        if value is not None:
            row[name] = value
    # The shared objective's remaining weighted terms, emitted only where the weight priced
    # them, for the same reason. ``geometry_loss`` is skipped when the absolute branch above
    # already wrote it, so one quantity never appears under two gates.
    for name, _ in _LOSS_TERM_COLUMN_WEIGHTS:
        value = getattr(record, name)
        if name not in row and value is not None:
            row[name] = value
    return row


def _records(values: object) -> tuple[SteeringTrainingRecord, ...]:
    if not isinstance(values, list):
        raise ValueError("training records must be a list")
    try:
        parsed = []
        for row in values:
            if not isinstance(row, dict):
                raise ValueError("training record must be an object")
            keys = frozenset(row)
            # Each off-path column is independently optional, because each is gated on its
            # own weight: ``lambda_perp > 0`` with ``lambda_rigid == 0`` is a configuration
            # D-14's sweep actually runs. The D-35b monitor columns are optional on the same
            # terms, gated on the arm instead of on a weight, and the shared loss-term columns
            # on the same terms again, gated on the shared weight that prices each. The pairing
            # with the configuration that wrote them is checked by
            # ``_validate_training_record_fields``, which has the config and objective in hand.
            #
            # A base set must be CONTAINED in the row and the surplus must be optional, rather
            # than the surplus being subtracted before an equality test. The two are the same
            # test while every optional column is disjoint from every base set, and they stop
            # being the same the moment one is not: ``geometry_loss`` is a mandatory column of
            # ``_ABSOLUTE_RECORD_KEYS`` and an optional one on the relative paths, and
            # subtracting it would have made an absolute row that LOST its geometry column
            # parse as a valid schema. Containment keeps every base column mandatory on its own
            # path, so a row missing or renaming a required key still fails closed.
            if not any(
                base <= keys and keys - base <= _OPTIONAL_RECORD_KEYS
                for base in _BASE_RECORD_KEY_SETS
            ):
                raise ValueError("training record key set is invalid")
            # The registry-declared project columns land in one mapping and every other column
            # is a declared field, which is the only place this split is visible: ``__getattr__``
            # makes the two indistinguishable to every reader, including the validators below.
            project = keys & _OFF_PATH_RECORD_KEYS
            monitored = keys & _DECLARED_MONITOR_RECORD_KEYS
            parsed.append(
                SteeringTrainingRecord(
                    unit=SteeringUnit(**row["unit"]),
                    project_columns={name: row[name] for name in sorted(project)},
                    monitor_columns={name: row[name] for name in sorted(monitored)},
                    **{name: row[name] for name in keys - project - monitored - {"unit"}},
                )
            )
        return tuple(parsed)
    except Exception as error:
        raise ValueError("training records are invalid") from error


def _failures(values: object) -> tuple[SteeringFailure, ...]:
    if not isinstance(values, list):
        raise ValueError("failures must be a list")
    try:
        keys = {
            "sequence_id",
            "seed",
            "target",
            "phase",
            "exception_class",
            "message",
        }
        if any(not isinstance(row, dict) or set(row) != keys for row in values):
            raise ValueError
        return tuple(SteeringFailure(**row) for row in values)
    except Exception as error:
        raise ValueError("failure records are invalid") from error


def _attempts(journal: Mapping[str, Any], key: str) -> int:
    return int(journal["attempts"].get(key, 0))


def _failure(
    unit: SteeringUnit,
    target: float,
    phase: str,
    error: Exception,
    members: Sequence[SteeringUnit] | None = None,
    *,
    label: str = "batch",
) -> SteeringFailure:
    """Record one failure, naming every member when a whole batch failed together.

    A batch is one atomic attempt, so it produces one failure row and one attempt
    increment -- the journal's attempt/failure consistency check counts them one to one.
    The row is keyed on the batch's first member so it still matches a scheduled unit,
    and the message names every member so no failed unit is hidden.

    ``label`` names the atomic unit that failed. Under gradient accumulation that is the
    accumulation group, which spans several generation batches and takes one optimizer
    step, so calling it a batch would understate what was rolled back.
    """
    message = str(error) or repr(error)
    if members is not None and len(members) > 1:
        inventory = ", ".join(f"{item.sequence_id}#{item.seed}" for item in members)
        message = (
            f"{message} [{label} of {len(members)} failed as one atomic unit; "
            f"members: {inventory}]"
        )
    return SteeringFailure(
        unit.sequence_id,
        unit.seed,
        float(target),
        phase,
        type(error).__name__,
        message,
    )


def _sample(value: object) -> GeneratedTrainingSample:
    if not isinstance(value, GeneratedTrainingSample):
        raise TypeError("generate must return GeneratedTrainingSample")
    for tensor in (
        value.surrogate_prediction,
        value.residual_rms,
        value.residual_fraction,
        value.geometry_loss,
    ):
        if (
            not isinstance(tensor, Tensor)
            or not tensor.is_floating_point()
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("generated sample tensors must be finite floating tensors")
    return value


def _standardized_absolute_target(
    absolute_target: float, config: OrdinarySteeringConfig
) -> StandardizedAbsoluteTarget:
    standardized = (absolute_target - config.target_mean) / config.target_scale
    inverse = config.target_mean + standardized * config.target_scale
    if not math.isclose(inverse, absolute_target, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("ordinary steering target normalization is inconsistent")
    return StandardizedAbsoluteTarget(standardized)


def _validate_optimizer_parameters(optimizer: torch.optim.Optimizer, adapter: nn.Module) -> None:
    adapter_parameters = tuple(adapter.parameters())
    optimizer_parameters = tuple(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )
    adapter_ids = tuple(id(parameter) for parameter in adapter_parameters)
    optimizer_ids = tuple(id(parameter) for parameter in optimizer_parameters)
    if (
        len(adapter_ids) != len(set(adapter_ids))
        or len(optimizer_ids) != len(set(optimizer_ids))
        or set(adapter_ids) != set(optimizer_ids)
    ):
        raise ValueError("optimizer parameters must equal adapter parameters exactly")


def _validate_optimizer_identity(
    optimizer: torch.optim.Optimizer,
    adapter: nn.Module,
    expected: object,
) -> None:
    if _optimizer_config(optimizer, adapter) != expected:
        raise ValueError("optimizer parameter-group identity does not match adapter config")


#: Partition count for the frozen-integrity digest. See ``_tensor_digest``.
_DIGEST_LANES = 16

#: Signed integer view of a raw bit pattern, keyed by element size in bytes.
_BIT_VIEW_DTYPES = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


@dataclass(frozen=True)
class _FrozenGuard:
    """Run-scoped proof that the frozen generator/SAE stack never changes.

    Built once per run, not once per unit. The old per-unit reference cloned every frozen
    tensor to host memory before each unit and again inside every check, so a run spent
    the great majority of its wall clock moving an unchanging model across PCIe rather
    than generating structures. The frozen stack is invariant by construction, so one
    reference is the right scope, and the per-step check compares device-side digests.

    ``rollback`` is the one host copy the run keeps: it is what a failed unit is rolled
    back to, and it is the exact comparison used to name a mutated tensor in an error.

    **Placement is not part of being frozen.** Neither the inventory nor the digest
    records which device a tensor sits on; see ``_frozen_inventory``.
    """

    modules: tuple[nn.Module, ...]
    inventories: tuple[tuple[tuple[str, str, str, tuple[int, ...]], ...], ...]
    digests: tuple[Tensor, ...]
    rollback: tuple[dict[str, dict[str, Tensor]], ...]


def _assert_frozen_state(guard: _FrozenGuard) -> None:
    """Fail closed if any frozen parameter or buffer changed since the guard was built.

    The comparison is a digest reduced on each tensor's own device, so the hot path moves
    32 int64 per tensor rather than the whole model. Only when a digest disagrees is the
    full host copy read, and then only to name the offending tensor in the error.

    Every raise names what changed. A bare message cost a GPU pilot a whole diagnosis
    cycle, because the failure was a lazily deferred device move that the message could
    not distinguish from a corrupted weight.
    """
    for module, inventory, digest, rollback in zip(
        guard.modules, guard.inventories, guard.digests, guard.rollback, strict=True
    ):
        if module.training:
            raise RuntimeError("frozen modules must remain in eval mode")
        current_inventory = _frozen_inventory(module)
        if current_inventory != inventory:
            difference = _frozen_inventory_difference(current_inventory, inventory)
            raise RuntimeError(
                f"frozen module stateful buffers or parameters mutated: {difference}"
            )
        if not torch.equal(_frozen_digest(module), digest):
            detail = _frozen_mutation_detail(module, rollback)
            raise RuntimeError(f"frozen module stateful buffers or parameters mutated: {detail}")


def _restore_frozen_state(guard: _FrozenGuard) -> None:
    """Roll the frozen modules back to the state captured at the start of the run.

    Rollback is defined against the run-scoped copy rather than a per-unit one. That is
    the same state a per-unit copy would hold: every committed unit ends with a passing
    integrity check, so the frozen stack is bit-identical at every unit boundary, and a
    unit that failed *because* the stack moved must be undone to the reference anyway.
    """
    for module, state in zip(guard.modules, guard.rollback, strict=True):
        current = _frozen_named_tensors(module)
        if any(set(current[kind]) != set(state[kind]) for kind in ("parameters", "buffers")):
            raise RuntimeError("frozen module tensor inventory mutated and cannot be restored")
        with torch.no_grad():
            for kind in ("parameters", "buffers"):
                for name, value in state[kind].items():
                    current[kind][name].copy_(
                        value.to(
                            device=current[kind][name].device,
                            dtype=current[kind][name].dtype,
                        )
                    )


def _frozen_mutation_detail(module: nn.Module, state: Mapping[str, Mapping[str, Tensor]]) -> str:
    """Name the first frozen tensor that differs from the reference copy.

    Failure path only: this is the expensive exact comparison the per-step check exists
    to avoid, and it runs at most once, immediately before the unit is failed.
    """
    current = _frozen_named_tensors(module)
    for kind in ("parameters", "buffers"):
        for name, value in state[kind].items():
            live = current[kind].get(name)
            if live is None or not torch.equal(live.detach().cpu(), value):
                return f"{kind}.{name}"
    # The digest is bit-exact and ``torch.equal`` is value-exact, so they can only
    # disagree on a value with more than one encoding: a flipped signed zero or a
    # different NaN payload. Still a write into a frozen tensor, so still a failure, but
    # say which kind it is rather than leaving the next reader to rediscover it.
    return (
        "a frozen tensor's bit pattern changed while every value still compares equal "
        "(flipped signed zero or NaN payload)"
    )


def _frozen_named_tensors(module: nn.Module) -> dict[str, dict[str, Tensor]]:
    return {
        "parameters": dict(module.named_parameters(remove_duplicate=False)),
        "buffers": dict(module.named_buffers(remove_duplicate=False)),
    }


def _frozen_state(module: nn.Module) -> dict[str, dict[str, Tensor]]:
    """Clone every frozen tensor to host memory. Once per run, never per unit."""
    return {
        kind: {name: value.detach().cpu().clone() for name, value in tensors.items()}
        for kind, tensors in _frozen_named_tensors(module).items()
    }


def _frozen_inventory(module: nn.Module) -> tuple[tuple[str, str, str, tuple[int, ...]], ...]:
    """Name, dtype and shape of every frozen tensor, in deterministic order.

    **Device is deliberately absent, and must not be added back.** Being frozen is a
    statement about a tensor's values, not about which memory holds them. BioEmu's loader
    returns the score model on the host and the *first* generation in a process moves it
    to the device -- that is what D7's discarded warm-up is for ("one discarded generation
    moves the tables to the device"), and ``capture_or_steer_from_denoiser`` and
    ``generate_unsteered`` both call ``score_model.to(device)`` themselves. A run-scoped
    reference is therefore built while the model is still on the host, and recording the
    device made that lawful, value-preserving move indistinguishable from a corrupted
    weight. It failed both GPU pilots on their first accumulation group.

    Dtype and shape stay, and dtype is a real strengthening: ``torch.equal`` ignores
    dtype, so the whole-model comparison this replaced would not have noticed a frozen
    parameter silently recast.
    """
    return tuple(
        (kind, name, str(value.dtype), tuple(value.shape))
        for kind, tensors in _frozen_named_tensors(module).items()
        for name, value in tensors.items()
    )


def _frozen_inventory_difference(
    current: Sequence[tuple[str, str, str, tuple[int, ...]]],
    expected: Sequence[tuple[str, str, str, tuple[int, ...]]],
) -> str:
    """Name the first inventory entry that differs, for the error message."""
    current_names = [entry[:2] for entry in current]
    expected_names = [entry[:2] for entry in expected]
    for kind, name in expected_names:
        if (kind, name) not in current_names:
            return f"{kind}.{name} disappeared"
    for kind, name in current_names:
        if (kind, name) not in expected_names:
            return f"{kind}.{name} appeared"
    for live, reference in zip(current, expected, strict=False):
        if live != reference:
            return (
                f"{live[0]}.{live[1]} changed from {reference[2]}{list(reference[3])} "
                f"to {live[2]}{list(live[3])}"
            )
    return "frozen tensor order changed"


def _tensor_digest(tensor: Tensor) -> Tensor:
    """Return an integer digest of one tensor's raw bits, computed on its own device.

    The reduction runs over the bit pattern reinterpreted as a signed integer of the same
    width, so it is exact integer arithmetic: no rounding, and no NaN that would compare
    unequal to itself and raise a false alarm on an untouched model. Two independent
    partitions are summed -- ``_DIGEST_LANES`` strided lanes (element ``i`` joins lane
    ``i % _DIGEST_LANES``) and ``_DIGEST_LANES`` contiguous blocks. A mutation escapes
    only if its deltas cancel exactly within a residue class *and* within a contiguous
    block simultaneously, which no single-element write can do. Element count, dtype,
    shape and device are not digested because ``_frozen_inventory`` already compares them
    exactly on the host. Nothing of the tensor's size is allocated: the bit view is a
    view, and the two reductions are memory-bandwidth bound.
    """
    flat = tensor.detach().reshape(-1)
    if flat.is_complex():
        flat = torch.view_as_real(flat).reshape(-1)
    bits = flat.contiguous().view(_BIT_VIEW_DTYPES[flat.element_size()])
    split = bits.numel() - bits.numel() % _DIGEST_LANES
    head, tail = bits[:split], bits[split:]
    lanes = head.view(-1, _DIGEST_LANES).sum(dim=0, dtype=torch.int64)
    blocks = head.view(_DIGEST_LANES, -1).sum(dim=1, dtype=torch.int64)
    if tail.numel():
        remainder = tail.to(torch.int64)
        lanes[: remainder.numel()] += remainder
        blocks[-1] += remainder.sum()
    return torch.cat((lanes, blocks))


def _frozen_digest(module: nn.Module) -> Tensor:
    """Digest every frozen tensor into one host-side vector in canonical order.

    Returned on the host and **not** keyed by device, for the reason ``_frozen_inventory``
    gives: an earlier version returned one digest per device, which made a lawful device
    move fail the comparison even once the inventory stopped recording placement. Keying
    by position in the (host-compared) inventory order instead makes that whole class of
    false positive structurally impossible.

    The transfer is 32 int64 per tensor -- of order 100 KB for a BioEmu-scale model, in
    one transfer and one synchronization per module, against the whole-model host copy
    this check replaced.
    """
    digests = [
        _tensor_digest(value)
        for tensors in _frozen_named_tensors(module).values()
        for value in tensors.values()
    ]
    if not digests:
        return torch.zeros(0, dtype=torch.int64)
    if len({digest.device for digest in digests}) == 1:
        return torch.cat(digests).cpu()
    return torch.cat([digest.cpu() for digest in digests])


def _frozen_guard(modules: Sequence[nn.Module]) -> _FrozenGuard:
    """Build the run-scoped frozen-integrity reference. Called once per run."""
    frozen = tuple(modules)
    return _FrozenGuard(
        modules=frozen,
        inventories=tuple(_frozen_inventory(module) for module in frozen),
        digests=tuple(_frozen_digest(module) for module in frozen),
        rollback=tuple(_frozen_state(module) for module in frozen),
    )


def _validate_gates(
    record: Mapping[str, Any],
    config: OrdinarySteeringConfig,
    adapter: nn.Module | None = None,
) -> None:
    """Apply this run's declared gates to the SELECTED validation round's aggregate.

    ``adapter`` is here only so the residual message can name the cap mode. Under
    ``residual_cap='tanh'`` this gate can essentially never fire -- the cap bounds every round's
    ``residual_magnitude`` by ``rho_max``, and the shipped thresholds sit just under it -- so it
    reads as a formality. Under ``residual_cap='none'`` it is the ONLY bound on the dose, and a
    run that blows past it must be legible as a refused run rather than a quiet one; hence the
    mode in the text. ``bioemu/steering_workflow.py`` requires the threshold to be set at all
    under that mode, so "uncapped and ungated" is not a reachable configuration.
    """
    # ``validation_target_mae_maximum`` is an absolute-unit gate and is rejected outright
    # on the relative path, so this comparison is never reached with mismatched units.
    if (
        config.validation_target_mae_maximum is not None
        and record["target_mae"] > config.validation_target_mae_maximum
    ):
        raise ValueError("validation target MAE gate failed")
    if (
        config.validation_residual_fraction_maximum is not None
        and record["residual_magnitude"] > config.validation_residual_fraction_maximum
    ):
        cap = getattr(adapter, "residual_cap", DEFAULT_RESIDUAL_CAP)
        if cap != DEFAULT_RESIDUAL_CAP:
            raise ValueError(
                "validation residual magnitude gate failed: the selected round's mean "
                f"residual fraction {record['residual_magnitude']} exceeds "
                f"{config.validation_residual_fraction_maximum}, and this run is "
                f"residual_cap={cap!r}, so nothing bounded it. The residual is NOT truncated "
                "to the threshold and the run is not completed; retrain with a priced residual "
                "penalty, a lower threshold, or residual_cap='tanh'"
            )
        raise ValueError("validation residual magnitude gate failed")


def _validation_aggregate(
    rows: Sequence[Mapping[str, Any]],
    round_index: int,
    train_cursor: int,
    relative: bool,
    selection_metric: ValidationSelectionMetric | None = None,
) -> dict[str, Any]:
    """Summarize one completed validation round.

    ``target_mae`` keeps its raw property units on both paths. The relative path adds the
    sigma-normalized sibling ``target_mae_sigma``, and selects on it unless the property declared
    otherwise, so best-checkpoint selection cannot disagree with the sigma-normalized training
    objective (D4b).

    ``selection_metric`` adds ONE aggregate: the mean over this round's rows of the column the
    property named. The sibling metrics are still computed and still recorded, so a declared metric
    WIDENS the record rather than replacing it and a report can always read both -- which is how a
    direct-metric arm's write-up shows that the epoch it shipped was chosen on the graded
    deviation and what its ``the progress coordinate`` error was doing at the same time.

    It FAILS CLOSED. A declared column absent from a row means the property's own
    ``validation_row_columns`` hook and its selection declaration disagree, and the cost of
    guessing is a published result selected on the proxy the arm exists to avoid, so the round
    raises here rather than falling back to ``target_mae_sigma``.
    """
    aggregate = {
        "validation_index": round_index,
        "train_cursor": train_cursor,
        "target_mae": sum(row["target_absolute_error"] for row in rows) / len(rows),
        "residual_magnitude": sum(row["residual_magnitude"] for row in rows) / len(rows),
        "observation_count": len(rows),
        "rows": copy.deepcopy(list(rows)),
    }
    if relative:
        aggregate["target_mae_sigma"] = sum(row["target_sigma_error"] for row in rows) / len(rows)
    if selection_metric is not None:
        missing = sum(1 for row in rows if selection_metric.row_column not in row)
        if missing:
            raise ValueError(
                f"this run selects on {selection_metric.name!r}, the mean of the validation column "
                f"{selection_metric.row_column!r}, and {missing} of {len(rows)} rows do not carry "
                "it. The property's validation_row_columns hook and its selection declaration "
                "disagree; there is deliberately no fallback to the target error, because an arm "
                "that trained on a direct metric and silently shipped the epoch its proxy "
                "preferred would be indistinguishable from one that worked"
            )
        aggregate[selection_metric.name] = sum(
            float(row[selection_metric.row_column]) for row in rows
        ) / len(rows)
    aggregate.update(_learning_diagnostics(rows))
    return aggregate


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if not ordered:
        return float("nan")
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2)


def _learning_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Per-round answers to 'is the adapter learning', derived from the rows already recorded.

    `target_mae` alone cannot separate "not learning" from "learning the right direction far too
    weakly", and that ambiguity made two 17-hour arms undiagnosable: MAE sat at 0.70 across every
    round while the adapter was in fact moving in the correct direction on 55.7% of observations
    and delivering 1.1% of the ask. These three numbers say that directly.

    Nothing here is a new measurement. Every value is computed from the same `rows` the aggregate
    already deep-copies, so this costs one pass over 264 records and changes no training behaviour.

      achieved_over_ask_median  the fraction of the ask actually delivered -- the headline. 1.0 is
                                the target reached; 0.011 is what the rho_max = 1.0 arm did.
      achieved_target_sign_agreement
                                fraction of observations whose achieved value has the SAME SIGN as
                                its target. 0.5 is chance, so this separates "weak but correct" from
                                "no directional signal at all" -- which MAE cannot do.
      achieved_abs_median       the raw scale, so the ratio above can be read against the ask.
    """
    # `achieved_value` is recorded on the two-state/relative path and NOT on the absolute one, so
    # these diagnostics are ABSENT rather than zero where they are undefined. Assuming the key
    # existed broke 29 tests on the absolute path: a diagnostic must never make a run fail, and an
    # absent key is how a reader tells "this path does not define it" from "it was zero".
    if not rows or any("achieved_value" not in row or "target" not in row for row in rows):
        return {}
    achieved = [float(row["achieved_value"]) for row in rows]
    targets = [float(row["target"]) for row in rows]
    ratios = [
        abs(value) / abs(target) for value, target in zip(achieved, targets, strict=True) if target
    ]
    signed = [
        value * target for value, target in zip(achieved, targets, strict=True) if value and target
    ]
    return {
        "achieved_abs_median": _median([abs(value) for value in achieved]),
        "achieved_over_ask_median": _median(ratios) if ratios else float("nan"),
        "achieved_target_sign_agreement": (
            sum(1 for product in signed if product > 0) / len(signed) if signed else float("nan")
        ),
    }


def _selection_metric(relative: bool, selection_metric: str | None = None) -> str:
    """Return the aggregate key best-checkpoint selection minimises.

    ``selection_metric`` is the run's own resolved declaration, and ``None`` is the shared default
    -- the historical expression, unchanged, and what every run on disk resolves to. A declared
    name overrides both paths rather than only the relative one: what it says is "this run's claim
    is not its target error", which is as true of an absolute-path property as of a relative one.
    """
    if selection_metric is not None:
        return selection_metric
    return RELATIVE_SELECTION_METRIC if relative else ABSOLUTE_SELECTION_METRIC


def _selection_metric_name(config: OrdinarySteeringConfig) -> str:
    """Return the selection-metric spelling this CONFIG records and selects on."""
    declared = config.validation_selection_metric
    return _selection_metric(config.is_relative, None if declared is None else declared.name)


def _validation_key(
    record: Mapping[str, Any], relative: bool = False, selection_metric: str | None = None
) -> tuple[float, float]:
    return (
        float(record[_selection_metric(relative, selection_metric)]),
        float(record["residual_magnitude"]),
    )


def _best_checkpoint_path(checkpoint_path: Path, validation_index: int) -> Path:
    return checkpoint_path.with_suffix(checkpoint_path.suffix + f".best-{validation_index}")


def _model_optimizer_fingerprint(checkpoint: AdapterCheckpoint) -> str:
    return _value_fingerprint(
        {"adapter_state": checkpoint.adapter_state, "optimizer_state": checkpoint.optimizer_state}
    )


def ordinary_checkpoint_state_fingerprint(checkpoint: AdapterCheckpoint) -> str:
    """Fingerprint the complete ordinary adapter and resumable workflow state.

    Wall-clock timings never reach this value: they live in a separate diagnostic sidecar
    so that two checkpoints holding the same scientific state fingerprint identically no
    matter how long the run took.
    """
    if not isinstance(checkpoint, AdapterCheckpoint):
        raise TypeError("checkpoint must be an AdapterCheckpoint")
    return _value_fingerprint(
        {
            "manifest_fingerprint": checkpoint.manifest_fingerprint,
            "adapter_config": checkpoint.adapter_config,
            "property_callback_identity": checkpoint.property_callback_identity,
            "source_checkpoints": checkpoint.source_checkpoints,
            "adapter_state": checkpoint.adapter_state,
            "optimizer_state": checkpoint.optimizer_state,
            "epoch": checkpoint.epoch,
            "unit_cursor": checkpoint.unit_cursor,
            "validation_record": checkpoint.validation_record,
        }
    )


def _value_fingerprint(value: object) -> str:
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            update(str(tensor.dtype))
            update(list(tensor.shape))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(
                item, key=lambda candidate: (type(candidate).__name__, repr(candidate))
            ):
                update(key)
                update(item[key])
        elif isinstance(item, list | tuple):
            digest.update(b"sequence")
            for child in item:
                update(child)
        elif item is None:
            digest.update(b"none")
        elif isinstance(item, bool):
            digest.update(b"bool:1" if item else b"bool:0")
        elif isinstance(item, int):
            digest.update(f"int:{item}".encode())
        elif isinstance(item, float):
            digest.update(f"float:{item.hex()}".encode())
        elif isinstance(item, str):
            encoded = item.encode()
            digest.update(b"str:")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        else:
            raise ValueError(f"unsupported checkpoint fingerprint value: {type(item).__name__}")

    update(value)
    return digest.hexdigest()


def _cpu_state(module: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _scalar(value: Tensor) -> float:
    return float(value.detach().cpu())


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _finite_nonnegative(value: object, name: str) -> float:
    value = _finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _fingerprint(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 fingerprint")


def _fingerprints(value: object) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("source_checkpoints must be a non-empty mapping")
    for key, fingerprint in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("source_checkpoints keys must be non-empty strings")
        _fingerprint(fingerprint, "source_checkpoints values")


def _validate_inputs(
    schedule: object,
    baseline: object,
    baseline_identity: object,
    generate: object,
    adapter: object,
    optimizer: object,
    frozen: Sequence[nn.Module],
    objective: object,
    config: object,
    generate_batch: object = None,
    batch_plan: object = None,
) -> None:
    # Report which input is wrong. The previous single compound condition raised one opaque
    # message for eight distinct causes, which cost a GPU job to diagnose.
    invalid = [
        name
        for name, ok in (
            ("schedule", isinstance(schedule, SteeringSchedule)),
            ("generate", callable(generate)),
            ("adapter", isinstance(adapter, nn.Module)),
            ("optimizer", isinstance(optimizer, torch.optim.Optimizer)),
            ("objective", isinstance(objective, SteeringLoss)),
            ("config", isinstance(config, OrdinarySteeringConfig)),
            # At least one frozen module is required so freeze_for_steering has something to
            # freeze. An exact count is NOT correct here: the ordinary path freezes only the
            # generator, while the four-module (generator, SAE, predictor, surrogate) shape
            # belongs to the ensemble path, which enforces it in ensemble_workflow.
            ("frozen_modules", bool(frozen) and all(isinstance(m, nn.Module) for m in frozen)),
        )
        if not ok
    ]
    if invalid:
        raise ValueError(f"invalid ordinary steering inputs: {', '.join(invalid)}")
    if batch_plan is not None:
        if not isinstance(batch_plan, SteeringBatchPlan):
            raise ValueError("batch_plan must be a SteeringBatchPlan")
        if batch_plan.schedule != schedule:
            raise ValueError(
                "batch_plan was built for a different schedule; the plan orders the train "
                "split for length bucketing, so the runner must be given plan.schedule"
            )
        if not callable(generate_batch):
            raise ValueError("batched ordinary steering requires a callable generate_batch")
    elif generate_batch is not None and not callable(generate_batch):
        raise ValueError("generate_batch must be callable when supplied")
    if config.is_relative:
        # D3/D6: the relative target references no baseline, so training and validation
        # generate no adapter-off structures at all. Accepting an unused callback here
        # would leave the operator believing baselines still constrain the targets.
        if baseline is not None or baseline_identity is not None:
            raise ValueError(
                "relative_sequence steering must not be given a baseline callback or "
                "identity: its targets are built from the per-sequence reference table "
                "and no adapter-off generation happens during training or validation"
            )
        scheduled = tuple(
            unit.sequence_id for unit in (*schedule.train, *schedule.validation, *schedule.test)
        )
        if config.reads_property_reference:
            config.property_reference.require_coverage(scheduled, context="scheduled")
            config.property_reference.require_eligibility(
                scheduled, config.sigma_eligibility, context="scheduled"
            )
        else:
            # The system-constant twin of those two gates, over the table that replaced them, and
            # it is COVERAGE ONLY. There is no eligibility band here and its absence is a
            # statement: the band filters a pool on the width of each sequence's own ensemble,
            # which is a quantity this objective never reads, and this pool's gate is the
            # SEPARATION gate applied where the split record was built.
            #
            # Refused unit by unit rather than by set difference so the message names the first
            # missing system: the target IS this system's own threshold, so an uncovered sequence
            # has no target at all and a pooled default would put one protein's weight on
            # another's draws.
            for sequence_id in scheduled:
                config.system_target_constants.require(sequence_id)
    elif not callable(baseline) or not isinstance(baseline_identity, str) or not baseline_identity:
        raise ValueError("invalid ordinary steering inputs")
    if getattr(adapter, "property_name", None) != config.property_name:
        raise ValueError("adapter config property_name does not match ordinary steering property")
    if getattr(adapter, "model_family", None) != config.model_family.value:
        raise ValueError("adapter config model_family does not match ordinary steering model")
    if config.is_relative and getattr(adapter, "target_kind", None) != RELATIVE_SEQUENCE_TARGET:
        raise ValueError(
            "relative_sequence steering requires a relative_sequence adapter; the adapter "
            "input is the requested level, not a standardized absolute target"
        )
    _require_matched_loss_normalization(objective, config)
    # Whatever bound this objective's own project enforces against the population it is about
    # to train over. A no-op on the shared objective, so a radius-of-gyration run runs exactly
    # the checks it always ran; the cryptic-pocket curvature bound of D-31 is what it is for.
    objective.validate_schedule(schedule, config)


def _require_matched_loss_normalization(
    objective: SteeringLoss, config: OrdinarySteeringConfig
) -> None:
    """Fail closed unless the objective's declared normalisation is the config's own.

    The objective is built by the caller, before and separately from the config, so the two
    are two statements about one thing. They differ by ``sigma^2`` in the loss value, and
    nothing downstream distinguishes them: the identity records the CONFIG's construction, so
    an objective that quietly normalised differently would be recorded as the one it was not.
    """
    declared = objective.property_loss_normalization
    required = config.property_loss_normalization
    if declared != required:
        raise ValueError(
            f"the objective declares {declared!r} property-loss normalisation but this target "
            f"construction is {required!r}: target_kind={config.target_kind!r} with "
            f"relative_target_origin={config.relative_target_origin!r}. The two differ by a "
            "factor of sigma^2 in the loss, and the checkpoint identity records the "
            "construction, so the run would be recorded as an objective it did not optimise"
        )


def _primitive(value: object) -> Any:
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list | tuple):
        return [_primitive(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _primitive(item) for key, item in value.items()}
    raise ValueError("optimizer config must contain only primitive values")


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, allow_nan=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
