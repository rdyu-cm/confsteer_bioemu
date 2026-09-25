"""Exact differentiable BioEmu steering property callbacks.

THE DISPATCH IS A LOOKUP, NOT A CHAIN
--------------------------------------

:func:`generate_exact_property_sample` used to branch on ``adapter.property_name`` through four
literal ``elif``s and take ONE KEYWORD PER PROPERTY (``local_q=``, ``cryptic=``, ``kinase=``).
Every one of those branches held a property's own geometry -- which atoms its coordinate is read
between, which artifact supplies its reference, which of its weights switch which sample field on
-- inside a file that is a member of the SHARED ``generation_shared`` code group, which every
property's training fingerprint hashes. So measuring a FOURTH quantity was impossible without
re-keying the three that price none of its geometry, which is precisely the defect
:mod:`sae_bioemu.steering.property_registry` exists to end, surviving in the one place nobody had
moved yet.

Each registered property now carries its own ``measure_sample`` hook, in its own package, hashed in
its own code group. This file builds the batch, runs the sampler, hands the property a
:class:`GeneratedBatch`, and computes the two quantities that are the SAME for every property --
the C-alpha geometry penalty and the local-structure term against the matched control. A fifth
property costs a descriptor and a catalog entry.

THE ONE REMAINING LITERAL BRANCH, AND WHY IT STAYS ONE
-------------------------------------------------------

``local_q`` HAS NO DESCRIPTOR. It is not in
:mod:`sae_bioemu.steering.property_catalog` and ``property_registry.resolve`` refuses it by name.
Giving it one would mean inventing a dose set, a target origin, a loss class and a fingerprint
group for a property that no active experiment steers (the active experiment set), and it is also
the one property whose measurement does not fit the shared contract: it predicts one value per
RESIDUE, so its residual statistics are broadcast and masked rather than read straight off the
steered sample, and :class:`~sae_bioemu.steering.property_registry.PropertyMeasurement` carries no
residual fields because a property that could restate them could also contradict them.

So it stays a literal branch, deliberately and with its reason stated here rather than left to
look like an omission. It is the LAST one: the ``else`` below refuses by name, and a registered
property reaches its own hook without this file knowing it exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import torch

from sae_bioemu.bioemu.extractor import BioEmuRuntimeActivationSampler, SequenceRecord
from sae_bioemu.bioemu.steering import SteeringSite
from sae_bioemu.steering.adapter import BoundedResidualAdapter, TargetKind
from sae_bioemu.steering.local_q import (
    differentiable_local_q,
    directional_local_q_targets,
    reference_local_q_contacts,
)
from sae_bioemu.steering.ordinary_workflow import GeneratedTrainingSample
from sae_bioemu.steering.property_registry import PropertyMeasurement
from sae_bioemu.steering.property_registry import resolve as resolve_property
from sae_bioemu.steering.surrogate import ca_geometry_penalty, smooth_lddt_loss

__all__ = [
    "CRYPTIC_PROPERTY_NAME",
    "KINASE_PROPERTY_NAME",
    "LOCAL_Q_PROPERTY_NAME",
    "BioEmuCrypticDefinition",
    "BioEmuDomainMotionDefinition",
    "BioEmuKinaseDefinition",
    "BioEmuLocalQDefinition",
    "ControlFrames",
    "GeneratedBatch",
    "batch_residue_mask",
    "generate_exact_property_sample",
    "padded_node_orientations",
]

#: The cryptic-pocket progress coordinate. Spelled out rather than imported from
#: ``cryptic_steering.arm``: this module is on the import path of every radius-of-gyration and
#: Local-Q run, and a top-level import would pull the whole cryptic package -- and its pocket
#: geometry, stage-1 and artifact readers -- into their import graph for one string. The same
#: duplication the calibration module makes, and a test pins the spellings
#: equal.
#:
#: Kept after the dispatch became a lookup because ``BioEmuCrypticDefinition`` still lives here
#: (see its own docstring) and because tests pin the spelling; nothing in this file branches on it.
CRYPTIC_PROPERTY_NAME = "pocket_progress_coordinate"

#: The kinase DFG-in -> DFG-out progress coordinate, spelled out for exactly the same reason and
#: kept DISTINCT from the cryptic one on purpose. The coordinate is literally the same function of
#: the same pair machinery; the dispatch key is not shared, because a kinase run resolves a kinase
#: arm and a pocket artifact that a cryptic run has neither of. A test pins this spelling to
#: ``kinase_dfg.objective.KINASE_PROPERTY_NAME``.
KINASE_PROPERTY_NAME = "dfg_progress_coordinate"

#: The one property with no descriptor, and hence the one literal branch left below. See the
#: module docstring for why it has none and why that is a decision rather than a gap.
LOCAL_Q_PROPERTY_NAME = "local_q"


@dataclass(frozen=True)
class BioEmuLocalQDefinition:
    reference: torch.Tensor
    baseline: torch.Tensor


@dataclass(frozen=True)
class BioEmuCrypticDefinition:
    """The resolved arm one cryptic-pocket batch is scored against.

    It holds the arm and nothing else. Every other input the objective needs -- the pair set,
    the weights, the transition vector, each protein's ``c0``, and arms A/B's per-residue site
    channel -- is reached through
    :class:`~sae_bioemu.cryptic_steering.arm.ResolvedCrypticArm`, whose existence is the
    guarantee that every configuration gate already passed. The sequences are taken from the
    batch's own ``SequenceRecord``s rather than carried here, so the residue string validated
    against the artifact is by construction the one handed to the model.
    """

    arm: Any


@dataclass(frozen=True)
class BioEmuKinaseDefinition:
    """The resolved arm one kinase DFG batch is scored against.

    It holds the arm and nothing else, for the reason :class:`BioEmuCrypticDefinition` does: the
    pair set, the transition vector, the DFG-in origin, the motif window and the weights are all
    reached through :class:`~sae_bioemu.kinase_dfg.arm.ResolvedKinaseArm`, whose existence is the
    guarantee that every configuration gate already passed. The sequences are taken from the batch's
    own ``SequenceRecord``s rather than carried here, so the residue string validated against the
    artifact is by construction the one handed to the model.

    A SEPARATE class from the cryptic one rather than a shared "arm" field, so that the two hooks
    cannot be reached with each other's arm: the two resolved arms share no method the other
    would raise on, and a mistyped dispatch would otherwise produce a finite, plausible coordinate
    measured against the wrong transition. Each property's hook type-checks the definition it was
    handed, which is what keeps that guarantee now that the dispatch is a lookup.
    """

    arm: Any


@dataclass(frozen=True)
class BioEmuDomainMotionDefinition:
    """The resolved reference artifact one domain-motion batch is measured against.

    A THIRD class for the reason the second exists: the domain-motion arm IS its artifact, and it
    carries no ``.artifact.atom_set``, no pair set and no transition vector, so handing it to
    either progress-coordinate hook would raise rather than mis-measure -- but handing a cryptic
    arm to THIS hook would find ``.require``/``.entries`` missing only after the sampler had run.
    Naming the type is what makes that a construction-time statement.
    """

    artifact: Any


@dataclass(frozen=True)
class ControlFrames:
    """The matched unsteered control's own per-residue FRAMES, padded to the batch width.

    Requested by :attr:`~sae_bioemu.steering.property_registry.ControlMeasurement.frames`, and
    produced by exactly the generation that already produces the control's atoms -- see
    ``bioemu/steering_workflow.py:_reference_pair``. It is a pair rather than a placed atom set
    because the atoms are placed by composing each residue's rigid transform with ideal geometry,
    and a placed-atom tensor has thrown that transform away: a property reading N/CA/C/O, or
    reading backbone TORSIONS, cannot recover them from ``ca`` or from a virtual ``cb``.

    ``positions_nm`` is in BioEmu's own NANOMETRES, as ``ChemGraph.pos`` is and as every frame
    reconstruction in this repository expects; the C-alpha control beside it is in ANGSTROM,
    because that is what ``smooth_lddt_loss`` and the D-35b monitor read. The two units are kept
    apart at the type rather than reconciled here: ``positions_angstrom / 10`` is not exactly
    ``positions_nm`` in float32, and a factor of ten leaves every structure internally consistent.
    """

    positions_nm: torch.Tensor
    node_orientations: torch.Tensor

    def __post_init__(self) -> None:
        if self.positions_nm.ndim != 3 or self.positions_nm.shape[-1] != 3:
            raise ValueError("control frame positions must be [members, width, 3] in nanometres")
        members, width = int(self.positions_nm.shape[0]), int(self.positions_nm.shape[1])
        if tuple(self.node_orientations.shape) != (members, width, 3, 3):
            raise ValueError(
                "control frame orientations must be [members, width, 3, 3] and padded to the "
                "same width as their positions; N/CA/C/O are placed from each residue's own frame"
            )


def padded_node_orientations(steered: object, records: Sequence[SequenceRecord]) -> torch.Tensor:
    """Return the steered batch's frames as ``[members, width, 3, 3]``, padded like its positions.

    ``BioEmuSteeredSample.coordinates`` is already padded to the batch width; the frames sit on the
    raw ``sampled_batch`` as one flat ``[sum(lengths), 3, 3]`` block, exactly as
    :meth:`~sae_bioemu.bioemu.extractor.BioEmuRuntimeActivationSampler.generate_unsteered_frames`
    finds them, and the same split-and-pad is applied here so the two tensors index the same
    residues.

    Fails closed rather than falling back to C-alpha. Without the frames the virtual C-beta cannot
    be placed, and the fallback is precisely the D-69 defect: a coordinate that is finite,
    plausible, and measured in the wrong atom set.
    """
    orientations = getattr(getattr(steered, "sampled_batch", None), "node_orientations", None)
    lengths = [len(item.sequence) for item in records]
    total = sum(lengths)
    if (
        not isinstance(orientations, torch.Tensor)
        or not orientations.is_floating_point()
        or orientations.numel() != total * 9
        or not torch.isfinite(orientations).all()
    ):
        raise ValueError(
            "the steered batch carries no finite [residues, 3, 3] node_orientations, so the "
            "frame-placed atoms this property's coordinate is read between cannot be constructed; "
            "reading C-alpha instead is the D-69 defect and is refused rather than defaulted"
        )
    flat = orientations.reshape(total, 3, 3)
    width = max(lengths)
    if len(lengths) == 1:
        return flat.reshape(1, lengths[0], 3, 3)
    rows = torch.split(flat, lengths)
    return torch.stack(
        [
            row
            if int(row.shape[0]) == width
            else torch.cat([row, row.new_zeros((width - int(row.shape[0]), 3, 3))], dim=0)
            for row in rows
        ]
    )


@dataclass(frozen=True)
class GeneratedBatch:
    """ONE generated batch, as a property's ``measure_sample`` hook receives it.

    One argument rather than eight keywords, and that is the whole point of the seam: widening a
    keyword list on :func:`generate_exact_property_sample` is an edit to a file in the SHARED
    ``generation_shared`` group, so the previous shape made a fourth property's inputs cost a
    re-key of the three that price none of them. A hook reads what it needs off this object and
    ignores the rest.

    Nothing here is detached. The steered structure carries the adapter's gradient and a hook sits
    on the path from that structure to the loss; the CONTROL halves are already detached by their
    producer, because a gradient reaching the reference would let the loss fall by moving the
    reference rather than the structure.
    """

    #: The batch members, in schedule order. ``sequence_id`` and ``sequence`` are read off these
    #: rather than carried separately, so the residue string a hook validates against an artifact
    #: is by construction the one that was handed to the model.
    records: tuple[SequenceRecord, ...]
    adapter: BoundedResidualAdapter
    #: The raw steered sample: ``coordinates`` (nanometres, padded), ``residual_rms``,
    #: ``residual_fraction``, ``raw_residual_fraction`` and ``sampled_batch``.
    steered: Any
    #: ``steered.coordinates * 10``: the padded C-alpha positions in ANGSTROM, computed once so
    #: every hook and the shared geometry terms read the same tensor.
    coordinates_angstrom: torch.Tensor
    #: ``[members, width]`` validity mask; padded columns past a member's own length are false.
    mask: torch.Tensor
    #: The ADAPTER's conditioning input, one value per member: a standardized absolute value on
    #: the absolute path and the bounded requested level ``o`` on the relative one (D1).
    target: torch.Tensor
    #: This property's own resolved definition -- ``BioEmuCrypticDefinition``,
    #: ``BioEmuKinaseDefinition``, ``BioEmuDomainMotionDefinition`` -- or ``None`` for a property
    #: with no arm. Each hook type-checks its own.
    definition: Any = None
    #: The matched unsteered control in C-ALPHA and ANGSTROM, ``[members, width, 3]``, or ``None``.
    reference_coordinates: torch.Tensor | None = None
    #: The SAME draw read in the atom set this property's ``control_measurement`` named, or
    #: ``None`` where it named ``ca`` or none at all.
    reference_measurement_coordinates: torch.Tensor | None = None
    #: The SAME draw's per-residue frames, where this property's ``control_measurement`` declared
    #: ``frames``. ``None`` otherwise, and never a second generation.
    reference_frames: ControlFrames | None = None

    @property
    def sequence_ids(self) -> list[str]:
        return [item.sequence_id for item in self.records]

    @property
    def sequences(self) -> dict[str, str]:
        return {item.sequence_id: item.sequence for item in self.records}

    @property
    def levels(self) -> list[float]:
        """The conditioning value per member, as plain floats.

        On a two-state property this IS the dose level ``o``, which selects the reference or the
        centroid a member's quantity is measured against -- so it is read off the batch rather
        than inferred, because a distance to the wrong reference is a NUMBER and not an error.
        """
        return [float(value) for value in self.target.detach().reshape(-1)]

    @cached_property
    def node_orientations(self) -> torch.Tensor:
        """The STEERED batch's frames, ``[members, width, 3, 3]``, padded and gradient-bearing.

        Cached because two hooks read it twice in one call (the kinase coordinate and its motif)
        and because it is a reshape of a tensor the sampler already returned, not a generation.
        Computed lazily: a radius-of-gyration sampler need not carry frames at all, and this
        raises rather than falling back where a hook asks for them and they are absent.
        """
        return padded_node_orientations(self.steered, self.records)

    def require_relative_levels(self, property_name: str) -> None:
        """Fail closed unless the adapter is on the relative path with one level per member.

        The shared half of the check both two-state hooks opened with, spelled once here so the
        two cannot drift into accepting different shapes.
        """
        if self.adapter.target_kind != TargetKind.RELATIVE_SEQUENCE.value or self.target.shape != (
            len(self.records),
        ):
            raise ValueError(
                f"{property_name} requires one relative sequence level per member: the level "
                "names a STATE or a REFERENCE and the adapter is conditioned on it"
            )


def batch_residue_mask(
    records: Sequence[SequenceRecord], device: torch.device | str | None = None
) -> torch.Tensor:
    """Return the ``[B, max_residues]`` validity mask for one generation batch.

    Members may differ in length; the padded columns past a member's own sequence
    length are false, so every masked reduction downstream excludes them. With one
    member this is exactly ``torch.ones((1, len(sequence)))``, the pre-batching mask.
    """
    lengths = tuple(len(record.sequence) for record in records)
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("every batch member must have a non-empty sequence")
    mask = torch.zeros((len(lengths), max(lengths)), dtype=torch.bool, device=device)
    for index, length in enumerate(lengths):
        mask[index, :length] = True
    return mask


def _local_q_measurement(
    batch: GeneratedBatch, local_q: BioEmuLocalQDefinition | None
) -> tuple[PropertyMeasurement, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The one literal branch: Local Q, which has no descriptor. See the module docstring.

    Returns its own residual statistics beside the measurement because it is the one property that
    reshapes them: the prediction is per RESIDUE, so the per-MEMBER residual rms and fraction are
    broadcast across residues and then masked to the contact-eligible ones. That reshaping is why
    ``PropertyMeasurement`` carries no residual fields and why forcing this property through the
    shared contract would mean widening it for its sole exception.
    """
    if batch.adapter.target_kind != TargetKind.DIRECTIONAL_RESIDUE.value or local_q is None:
        raise ValueError("local_q requires a directional residue adapter and definition")
    if len(batch.records) != 1:
        raise ValueError(
            "local_q steering is defined against one sequence's reference contacts, so it "
            "is not batched across sequences here"
        )
    steered = batch.steered
    coordinates_angstrom = batch.coordinates_angstrom
    reference = local_q.reference.to(coordinates_angstrom)
    baseline = local_q.baseline.to(coordinates_angstrom)
    contacts = reference_local_q_contacts(reference)
    prediction, valid = differentiable_local_q(
        coordinates_angstrom, reference, contacts, batch.mask
    )
    objective_target = directional_local_q_targets(baseline, batch.target, valid)
    return (
        PropertyMeasurement(prediction=prediction[valid], target=objective_target[valid]),
        steered.residual_rms[:, None].expand_as(valid)[valid],
        steered.residual_fraction[:, None].expand_as(valid)[valid],
        None
        if steered.raw_residual_fraction is None
        else steered.raw_residual_fraction[:, None].expand_as(valid)[valid],
    )


def generate_exact_property_sample(
    *,
    sampler: BioEmuRuntimeActivationSampler,
    record: SequenceRecord | Sequence[SequenceRecord],
    seed: int | Sequence[int],
    site: SteeringSite,
    adapter: BoundedResidualAdapter,
    target: torch.Tensor,
    local_q: BioEmuLocalQDefinition | None = None,
    definition: Any = None,
    reference_coordinates: torch.Tensor | None = None,
    reference_measurement_coordinates: torch.Tensor | None = None,
    reference_frames: ControlFrames | None = None,
) -> tuple[GeneratedTrainingSample, torch.Tensor]:
    """Generate one batch of structures and return their exact property and target.

    ``record``/``seed`` accept a single value or ``B >= 1`` members; ``target`` then
    carries one conditioning value per member and every returned tensor is ordered by
    batch member. ``target`` is the adapter's conditioning input, which is a
    standardized absolute value for ``TargetKind.ABSOLUTE_SEQUENCE`` and the bounded
    requested level ``o`` for ``TargetKind.RELATIVE_SEQUENCE`` (D1). The second return
    value echoes that conditioning; on the relative path the property target itself is
    built by the workflow from each sequence's own reference statistics and is not
    derivable here.

    ``definition`` is ONE keyword carrying whichever resolved definition the steered property's
    descriptor supplied, and it replaced the per-property ``cryptic=``/``kinase=`` pair. The
    change is not cosmetic: this file is in the SHARED ``generation_shared`` code group, so a
    fourth property's keyword re-keyed the three that never pass it. A property that invents its
    own keyword now gets a ``TypeError`` at the call site instead of a new parameter here.

    ``local_q`` keeps its own keyword because it is the one property with no descriptor and hence
    no hook to route a definition to; see the module docstring.

    ``reference_coordinates``, ``reference_measurement_coordinates`` and ``reference_frames`` are
    the matched unsteered control in the three readings a property may declare through
    :class:`~sae_bioemu.steering.property_registry.ControlMeasurement` -- C-alpha Angstrom, a named
    atom set, and the raw per-residue frames. All three come out of ONE generation. Whether any of
    them is REQUIRED is the property's statement, not this file's: a progress coordinate is
    ``u = d(scored) - d(reference)`` and is undefined without its control, while ``lambda_l`` is
    merely unpenalised, and each hook refuses by absence for its own reason. The C-alpha reading is
    additionally read HERE, by ``smooth_lddt_loss``, which is a C-alpha quantity by definition.
    """
    records = (record,) if isinstance(record, SequenceRecord) else tuple(record)
    target = target.to(next(adapter.parameters()).device)
    mask = batch_residue_mask(records, device=target.device)
    property_name = adapter.property_name
    descriptor = None if property_name == LOCAL_Q_PROPERTY_NAME else resolve_property(property_name)
    if descriptor is not None and descriptor.measure_sample is None:
        raise ValueError(
            f"{property_name!r} is registered but declares no measure_sample hook, so there is no "
            "code that says what its property IS. Routing it through another property's hook "
            "would produce a finite, plausible number measured in the wrong atom set (D-69)"
        )
    # The adapter's per-residue conditioning channel, where this property's arm has one. Built
    # before the steer call rather than inside a hook because it is a per-protein CONSTANT
    # measured on the deposited structure: a per-call rebuild would spend a distance matrix per
    # denoiser iteration for a value that cannot change, and rebuilding it from the tensor being
    # steered would let a structure that pushed residues away from the pocket shrink its own
    # conditioning field. ``None`` on every arm without one, so the unconditioned path is
    # byte-identical to the pre-channel one.
    site_field = _site_field(definition, records, mask, target)
    steered = sampler.steer(
        records if len(records) > 1 else records[0],
        seed=seed,
        site=site,
        adapter=adapter,
        target=target,
        residue_mask=mask,
        site_field=site_field,
    )
    batch = GeneratedBatch(
        records=records,
        adapter=adapter,
        steered=steered,
        coordinates_angstrom=steered.coordinates * 10.0,
        mask=mask,
        target=target,
        definition=definition,
        reference_coordinates=reference_coordinates,
        reference_measurement_coordinates=reference_measurement_coordinates,
        reference_frames=reference_frames,
    )
    if descriptor is None:
        measurement, residual_rms, residual_fraction, raw_residual_fraction = _local_q_measurement(
            batch, local_q
        )
    else:
        measurement = descriptor.measure_sample(batch=batch)
        if not isinstance(measurement, PropertyMeasurement):
            raise TypeError(
                f"{property_name!r}'s measure_sample must return a PropertyMeasurement; it decides "
                "what this run's property IS and the shared caller cannot infer the contract from "
                "a tuple"
            )
        # Read off the steered sample rather than restated by the hook, so a property cannot
        # contradict the residual statistics of the structure it just measured.
        residual_rms = steered.residual_rms
        residual_fraction = steered.residual_fraction
        raw_residual_fraction = steered.raw_residual_fraction
    _require_declared_sample_fields(property_name, measurement.sample_fields)
    # Geometry is measured on every run, weighted only when lambda_g > 0. Unsteered BioEmu
    # output scores ~0.0005 here; the steered arm at o = +1 scored 4.08, so this is a live
    # diagnostic even for runs that do not optimise it.
    geometry = ca_geometry_penalty(batch.coordinates_angstrom, mask).mean()
    # Local structure against the unsteered draw. The flat-bottom window says a chain is
    # physically possible; only this says it is the same fold. C-alpha by definition, which is
    # why the C-alpha reading of the control never leaves this file's hands.
    local_structure = (
        None
        if reference_coordinates is None
        else smooth_lddt_loss(batch.coordinates_angstrom, reference_coordinates, mask).mean()
    )
    return (
        GeneratedTrainingSample(
            measurement.prediction,
            residual_rms,
            residual_fraction,
            geometry,
            raw_residual_fraction,
            local_structure,
            # Empty for every property that declares none, so a radius-of-gyration and a Local-Q
            # sample are field-for-field what they were before any property had extras.
            **dict(measurement.sample_fields),
        ),
        measurement.target,
    )


#: Every ``GeneratedTrainingSample`` field a property's hook is allowed to fill. The shared
#: contract's own fields -- the prediction, the residual statistics, the geometry penalty and the
#: local-structure term -- are read off the STEERED SAMPLE by the caller above and are deliberately
#: absent: a hook that could restate them could also contradict them, and the contradiction would
#: be a plausible number rather than an error.
_HOOK_FILLABLE_SAMPLE_FIELDS = frozenset(
    {
        "pocket_off_path_squared",
        "context_rms_angstrom",
        "context_setpoint_angstrom",
        "gate_scored_out_of_window_bonds",
        "gate_control_out_of_window_bonds",
        "gate_scored_clashing_pairs",
        "gate_control_clashing_pairs",
        "control_reporting_value",
        "dihedral_distance",
        "control_dihedral_distance",
        "metric_deviation_angstrom",
        "metric_separation_angstrom",
        # D-86's UNPRICED monitor wire. Offered on the same terms as every field above -- a hook
        # may fill it, the caller's own readings are still unrestatable -- and read by no loss.
        "pocket_off_axis_ratio_squared",
    }
)


def _require_declared_sample_fields(property_name: str, fields: Mapping[str, Any]) -> None:
    """Fail closed on a hook filling a field the shared sample does not offer it.

    Without this a misspelled field would reach ``GeneratedTrainingSample(**fields)`` as a
    ``TypeError`` naming a dataclass rather than a property, and -- worse -- a hook filling
    ``geometry_loss`` or ``residual_fraction`` would silently override the value the caller read
    off the steered structure, which is the one place a property could describe a structure other
    than the one it generated.
    """
    surplus = sorted(set(fields) - _HOOK_FILLABLE_SAMPLE_FIELDS)
    if surplus:
        raise ValueError(
            f"{property_name!r} filled sample fields {surplus}, which the shared generated sample "
            f"does not offer a property hook. The offered set is "
            f"{sorted(_HOOK_FILLABLE_SAMPLE_FIELDS)}; the prediction, the residual statistics, the "
            "geometry penalty and the local-structure term are read off the steered structure by "
            "the caller and must not be restatable by the property that measured it"
        )


def _site_field(
    definition: Any,
    records: Sequence[SequenceRecord],
    mask: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor | None:
    """Return the per-residue conditioning channel this batch's definition supplies, or ``None``.

    Duck-typed on ``definition.arm.site`` rather than on a property name, for the reason the whole
    dispatch stopped keying on property names: only the cryptic arms A and B carry a site channel
    today, and a fifth property that wants one should not need this file to learn its name.
    Arm C's cache is empty and its ``site`` is absent, so the unconditioned path -- which is every
    radius-of-gyration, Local-Q, kinase and domain-motion call -- is byte-identical to the
    pre-channel one.
    """
    site = getattr(getattr(definition, "arm", None), "site", None)
    if site is None:
        return None
    return site.batch(
        [item.sequence_id for item in records],
        {item.sequence_id: item.sequence for item in records},
        mask,
        dtype=target.dtype,
        device=target.device,
    )
