"""Continue an adapter checkpoint that is not this run's own: longer plan, or earlier epoch.

A checkpoint written under ``maximum_epochs: 40`` could not be pointed at a config that differed
only in ``maximum_epochs: 80``. It refused with

    adapter checkpoint manifest fingerprint does not match expected manifest fingerprint

which is the first of *four* gates the continuation trips, and none of them is about the trained
weights:

``manifest_fingerprint``
    ``steering_workflow.build`` derives the binding fingerprint from, among other things, the
    ``workflow_config_fingerprint`` -- a digest of the YAML FILE. Editing any byte of that file
    moves it, and ``maximum_epochs`` is a byte of that file. The epoch budget is therefore
    *entangled* with the run identity and cannot be separated without re-deriving the identity of
    every checkpoint ever written. It is left exactly as it is; see below for what replaces it.

``schedule_fingerprint`` / ``schedule_inventory``
    The inventory is the FULL per-unit plan. A 40-epoch plan has 2240 train units and an 80-epoch
    plan 4480, so these necessarily differ -- and they differ in the *tail*, which no completed
    epoch ever touched.

``generation_batching`` / ``gradient_accumulation``
    Both record realised per-batch sizes over the whole plan, so both grow with it.

What actually determines the trained weights is the sequence of units the run has ALREADY walked.
The first 40 epochs of an 80-epoch plan are the same 40 epochs: every unit's sequence, cycle,
visit, level, seed and draw seed is a pure function of ``(sequence_id, cycle, visit)`` and the
schedule SEED, and the seed deliberately excludes ``maximum_epochs``
(``steering_workflow._schedule_seed_fingerprint``). This module therefore replaces "the whole
intended inventory must be equal" with "the completed PREFIX must be equal", and leaves every
other gate untouched.

**It is on by default.** It used to require an explicit ``schedule.extends`` block naming the
origin fingerprint, on the reasoning that an opaque digest cannot say WHICH config byte moved. It
still cannot -- but the declaration never answered that question either. It authorised a resume it
did not inspect, and the answer was always in the key-by-key comparison below, which names the
first thing that actually moved. What the declaration bought was a second chance to refuse a stale
checkpoint left in ``output_dir``; what it cost was that every continuation had to be hand-written
into a config, and a checkpoint nobody declared for was simply unusable. The declaration is still
accepted and still enforced when it is present -- see :class:`ScheduleExtensionDeclaration` -- but
it is no longer what permits the resume.

**What replaces it is disclosure.** Adoption is automatic, so it must be impossible to read an
adopted run as one that always had these settings. Every exempt key that actually DIFFERED is
recorded, with before and after, in :attr:`ScheduleExtension.crossed`; that record is journalled
into the checkpoint under ``schedule_extensions`` and echoed into the run report, and the workflow
logs a warning naming each crossed key. A run that adopted nothing writes no record at all, so its
artifacts stay byte-identical to one made before this mechanism existed.

**Only five keys are exempt, each gets its own structural check, and the exemption is
closed-world.** Everything else -- objective weights, adapter shape, ``rho_max``, the optimizer
block (``lr`` and ``weight_decay`` live inside it), split, source checkpoints, property callback
identity, intervention site, the training code fingerprint -- is still compared for exact equality,
key by key, and a mismatch names the key. Inside the two batching blocks the exemption is narrower
still: only the growing per-unit lists may differ, every other sub-key must be EQUAL, and a sub-key
this module does not know about must be equal too. So a run that changed ``batch_size_100`` or
``gradient_accumulation.units`` -- a real change to what one optimizer step is -- refuses by name
rather than being adopted. Notably also NOT exempt:

* ``property_ensemble``, whose ``seed_fingerprint`` digests the per-unit ensemble seed table over
  the whole plan. An ensemble run therefore refuses to extend, by name. That is deliberate and
  fail-closed: prefix-restricting that fingerprint is a separate change with its own evidence.
* ``diagnostic_fixed_generation_seed``, for the same reason.

**A checkpoint may only be continued from a state its own cursor produced.** A selecting run that
terminates rolls its weights BACK to ``best_validation_index``. If that is not the final epoch, the
adapter in the main checkpoint is not the adapter its last epoch produced, so continuing from it is
not a continuation of anything. That still refuses -- but it is no longer a dead end, because every
``.best-N`` sidecar carries the state of epoch N together with the cursor, epoch and optimizer that
produced it. Naming that sidecar resumes from a self-consistent point; see
``ordinary_workflow._resume_checkpoint``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Schema of the record journalled and reported for each adopted resume.
#:
#: ``/2`` because the record grew the three fields that make an ALWAYS-ON adoption readable: which
#: exempt keys were crossed, which file the state came from, and what the origin run's terminal
#: state was. ``/1`` was written by nothing that survives -- the mechanism shipped and was never
#: run under a declaration -- so the schema is bumped rather than widened, and a ``/1`` record is
#: refused rather than half-understood.
SCHEDULE_EXTENSION_SCHEMA = "steering_schedule_extension/2"

#: The ONLY ``adapter_config`` keys permitted to differ between a checkpoint and the plan it is
#: being continued into. Every one is a pure function of the plan's LENGTH and every one gets its
#: own structural prefix check below; no key reaches this set merely by being inconvenient.
_EXTENSION_EXEMPT_KEYS = frozenset(
    {
        "manifest_fingerprint",
        "schedule_fingerprint",
        "schedule_inventory",
        "generation_batching",
        "gradient_accumulation",
    }
)

#: The ``generation_batching`` sub-key holding the realised train plan; prefix-checked.
_BATCHING_PREFIX_KEY = "train_batch_sizes"

#: The ``gradient_accumulation`` sub-key holding the realised train plan; prefix-checked.
_ACCUMULATION_PREFIX_KEY = "train_group_batch_sizes"

#: ``generation_batching`` sub-keys that are pure functions of :data:`_BATCHING_PREFIX_KEY` and may
#: therefore differ once that list has been prefix-checked. Each is RE-DERIVED below rather than
#: trusted, so a hand-edited identity claiming a batch count its own list does not have is refused.
#: Every other sub-key, named or not, must be equal.
_BATCHING_DERIVED = {
    "train_batch_count": lambda sizes: len(sizes),
    "maximum_train_batch_size": lambda sizes: max(sizes, default=0),
}

#: ``gradient_accumulation`` sub-keys that are pure functions of
#: :data:`_ACCUMULATION_PREFIX_KEY`, on the same terms.
_ACCUMULATION_DERIVED = {
    "train_group_unit_sizes": lambda groups: [sum(group) for group in groups],
    "optimizer_step_count": lambda groups: len(groups),
    "maximum_optimizer_step_units": lambda groups: max((sum(group) for group in groups), default=0),
    "minimum_optimizer_step_units": lambda groups: min((sum(group) for group in groups), default=0),
    # The realised per-step unit histogram: derived, because extending a schedule by whole
    # cycles adds steps and therefore adds to the counts. ``target_optimizer_step_units`` is
    # deliberately NOT here -- it is the uniform rule itself, and a run that changed it is not
    # continuing this checkpoint's optimization.
    "optimizer_step_unit_counts": lambda groups: [
        [size, [sum(group) for group in groups].count(size)]
        for size in sorted({sum(group) for group in groups})
    ],
}


class ScheduleExtensionError(ValueError):
    """A checkpoint in ``output_dir`` is not a continuation of the plan this run will walk."""


@dataclass(frozen=True)
class ScheduleExtensionDeclaration:
    """An OPTIONAL statement that this run continues one specific checkpoint.

    It no longer permits anything -- adoption is automatic and is gated on the structural checks
    below. What it still does is NARROW: declared, the checkpoint on disk must carry exactly this
    fingerprint and its inventory must span exactly this many epochs, so a run that means to
    continue one artifact refuses a different one rather than adopting whatever it finds. Use it
    where that guarantee is worth stating; omit it everywhere else.
    """

    from_manifest_fingerprint: str
    from_maximum_epochs: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.from_manifest_fingerprint, str)
            or len(self.from_manifest_fingerprint) != 64
            or any(
                character not in "0123456789abcdef" for character in self.from_manifest_fingerprint
            )
        ):
            raise ValueError(
                "schedule.extends.from_manifest_fingerprint must be a lowercase SHA-256 "
                "fingerprint"
            )
        if (
            not isinstance(self.from_maximum_epochs, int)
            or isinstance(self.from_maximum_epochs, bool)
            or self.from_maximum_epochs < 1
        ):
            raise ValueError("schedule.extends.from_maximum_epochs must be a positive integer")

    def identity(self) -> dict[str, Any]:
        """Return the declaration as it is recorded in the run report."""
        return {
            "from_manifest_fingerprint": self.from_manifest_fingerprint,
            "from_maximum_epochs": int(self.from_maximum_epochs),
        }


@dataclass(frozen=True)
class ScheduleExtension:
    """A REALISED continuation: what was adopted, from which file, and what that crossed.

    This is what makes an adopted run distinguishable from one that always had these settings. It
    is journalled into the checkpoint and echoed in the run report; a run that adopted nothing
    writes neither, so its artifacts stay byte-identical to one made before this mechanism existed.

    ``crossed`` is the disclosure that always-on adoption requires. Every exempt ``adapter_config``
    key that genuinely differed appears in it with a before and an after, so a reader of the report
    sees what the resume stepped over without opening the checkpoint. It is a list rather than a
    flag because the answer "which ones" is the whole content.
    """

    from_manifest_fingerprint: str
    to_manifest_fingerprint: str
    from_schedule_fingerprint: str
    to_schedule_fingerprint: str
    from_maximum_epochs: int
    to_maximum_epochs: int
    from_train_units: int
    to_train_units: int
    resumed_at_epoch: int
    resumed_at_unit_cursor: int
    #: File name -- not path -- the adopted state was read from: the main checkpoint or a
    #: ``.best-N`` sidecar. Names WHICH epoch's weights this run continues.
    resumed_from: str = ""
    #: The terminal state of the RUN this one continues -- read off the checkpoint in
    #: ``output_dir``, not off ``resumed_from``, because a ``.best-N`` sidecar is written mid-run
    #: and always says ``None``. ``early_stopped`` or ``max_cycles_reached`` for a run that ended,
    #: ``None`` for one that was preempted. Recorded rather than gated on: an early-stopped run is
    #: continuable on the same terms as any other, and this is what keeps it readable as one.
    from_terminal_reason: str | None = None
    #: Rewritten under this run's identity so the journal's best-checkpoint pointer stays loadable
    #: after the identity moved; ``None`` when nothing was rewritten.
    republished_best_checkpoint: str | None = None
    crossed: tuple[Mapping[str, Any], ...] = field(default=())

    def record(self) -> dict[str, Any]:
        """Return the JSON record journalled into the checkpoint and the run report."""
        return {
            "schema": SCHEDULE_EXTENSION_SCHEMA,
            "from_manifest_fingerprint": self.from_manifest_fingerprint,
            "to_manifest_fingerprint": self.to_manifest_fingerprint,
            "from_schedule_fingerprint": self.from_schedule_fingerprint,
            "to_schedule_fingerprint": self.to_schedule_fingerprint,
            "from_maximum_epochs": int(self.from_maximum_epochs),
            "to_maximum_epochs": int(self.to_maximum_epochs),
            "from_train_units": int(self.from_train_units),
            "to_train_units": int(self.to_train_units),
            "resumed_at_epoch": int(self.resumed_at_epoch),
            "resumed_at_unit_cursor": int(self.resumed_at_unit_cursor),
            "resumed_from": str(self.resumed_from),
            "from_terminal_reason": self.from_terminal_reason,
            "republished_best_checkpoint": self.republished_best_checkpoint,
            "crossed": [dict(row) for row in self.crossed],
        }

    def warnings(self) -> tuple[str, ...]:
        """Return one WARNING line per exempt key this resume crossed, naming both values."""
        return tuple(
            f"schedule resume crossed adapter config key {row['key']!r}: the checkpoint carries "
            f"{row['from']!r} and this run is configured for {row['to']!r}"
            for row in self.crossed
        )


#: The exact key set of one journalled resume record.
SCHEDULE_EXTENSION_RECORD_KEYS = frozenset(
    ScheduleExtension(
        from_manifest_fingerprint="0" * 64,
        to_manifest_fingerprint="0" * 64,
        from_schedule_fingerprint="0" * 64,
        to_schedule_fingerprint="0" * 64,
        from_maximum_epochs=1,
        to_maximum_epochs=2,
        from_train_units=1,
        to_train_units=2,
        resumed_at_epoch=1,
        resumed_at_unit_cursor=1,
    ).record()
)


@dataclass(frozen=True)
class ExtensionReconciliation:
    """What to hand the unchanged checkpoint validator, plus the resume it authorises.

    ``expected_adapter_config`` and ``expected_manifest_fingerprint`` are the NEW plan's values
    with the five exempt keys replaced by the checkpoint's own, so
    :func:`~sae_bioemu.steering.training.validate_adapter_checkpoint` still runs its whole-blob
    equality check afterwards -- unrelaxed -- over everything this module did not itself compare.
    """

    expected_adapter_config: Mapping[str, Any]
    expected_manifest_fingerprint: str
    extension: ScheduleExtension


def inventory_epochs(units: Sequence[Mapping[str, Any]]) -> int:
    """Return the epoch count a train inventory describes, from its own cycle numbers."""
    if not units:
        raise ScheduleExtensionError("a train schedule inventory cannot be empty")
    cycles = {unit["cycle"] for unit in units}
    if cycles != set(range(max(cycles) + 1)):
        raise ScheduleExtensionError(
            "train schedule inventory cycles are not a contiguous range from zero"
        )
    return max(cycles) + 1


def _inventory_summary(inventory: Mapping[str, Any]) -> dict[str, int]:
    """Summarise an inventory as unit counts per role; the full plan is far too large to report."""
    return {role: len(units) for role, units in sorted(inventory.items())}


def _batching_summary(batching: Mapping[str, Any] | None) -> Any:
    if not isinstance(batching, Mapping):
        return batching
    sizes = list(batching.get(_BATCHING_PREFIX_KEY) or [])
    return {
        "config": batching.get("config"),
        "train_batch_count": len(sizes),
        "train_units": sum(int(size) for size in sizes),
    }


def _accumulation_summary(accumulation: Mapping[str, Any] | None) -> Any:
    if not isinstance(accumulation, Mapping):
        return accumulation
    groups = list(accumulation.get(_ACCUMULATION_PREFIX_KEY) or [])
    return {
        "units": accumulation.get("units"),
        "train_group_count": len(groups),
        # A group is the LIST of micro-batch sizes that share one optimizer step, so the units
        # it covers are that list's sum. This used to read ``int(size) for size in groups`` and
        # was unreachable: the block was written only when the declared accumulation exceeded
        # one, and no such run ever reached a crossing disclosure. Under the uniform optimizer
        # step every run records the block, so the summary is reached and must be right.
        "train_units": sum(sum(group) for group in groups),
    }


#: How each exempt key is rendered for the disclosure. The two fingerprints go in whole; the three
#: structural blocks are summarised, because a 4480-unit inventory cannot be put in a run report
#: and a reader needs the SHAPE of what moved rather than its contents.
_CROSSING_SUMMARIES = {
    "manifest_fingerprint": lambda value: value,
    "schedule_fingerprint": lambda value: value,
    "schedule_inventory": _inventory_summary,
    "generation_batching": _batching_summary,
    "gradient_accumulation": _accumulation_summary,
}


def exempt_crossings(
    checkpoint_config: Mapping[str, Any], expected_config: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Return every EXEMPT key that actually differs, with a before and an after.

    The exemption is what makes always-on adoption possible; this is what makes it readable. A key
    that is exempt and happens to be equal is not reported, so the list answers "what did this
    resume step over" and nothing else.
    """
    crossings = []
    for key in sorted(_EXTENSION_EXEMPT_KEYS):
        if key not in checkpoint_config and key not in expected_config:
            continue
        stored = checkpoint_config.get(key)
        planned = expected_config.get(key)
        if stored == planned:
            continue
        summarise = _CROSSING_SUMMARIES[key]
        crossings.append({"key": key, "from": summarise(stored), "to": summarise(planned)})
    return tuple(crossings)


def reconcile_extended_adapter_config(
    *,
    checkpoint_config: Mapping[str, Any],
    checkpoint_manifest_fingerprint: str,
    checkpoint_unit_cursor: int,
    expected_config: Mapping[str, Any],
    expected_manifest_fingerprint: str,
    declaration: ScheduleExtensionDeclaration | None = None,
) -> ExtensionReconciliation:
    """Permit a resume when the COMPLETED PREFIX matches, and refuse loudly otherwise.

    ``declaration`` is optional. Present, it additionally requires that the checkpoint on disk is
    the one named; absent -- the ordinary case -- the structural checks below stand on their own.

    Raises :class:`ScheduleExtensionError` naming the first thing that moved: the differing
    ``adapter_config`` key, or the index and both values of the first differing scheduled unit.
    """
    if (
        declaration is not None
        and checkpoint_manifest_fingerprint != declaration.from_manifest_fingerprint
    ):
        raise ScheduleExtensionError(
            "the checkpoint being extended is not the one declared: "
            f"schedule.extends.from_manifest_fingerprint is "
            f"{declaration.from_manifest_fingerprint} and the checkpoint on disk carries "
            f"{checkpoint_manifest_fingerprint}"
        )
    if not isinstance(checkpoint_config, Mapping) or not isinstance(expected_config, Mapping):
        raise ScheduleExtensionError("adapter config must be a mapping")
    if set(checkpoint_config) != set(expected_config):
        added = sorted(set(expected_config) - set(checkpoint_config))
        removed = sorted(set(checkpoint_config) - set(expected_config))
        raise ScheduleExtensionError(
            "a schedule extension cannot change the adapter config schema; keys added "
            f"{added}, keys removed {removed}"
        )

    # EVERY key that is not one of the five length-derived ones is compared for exact equality,
    # and the failure names the key. This is what keeps the objective weights, the adapter shape
    # and ``rho_max``, the optimizer block, the split, the source checkpoints, the property
    # callback identity, the intervention site and the training code fingerprint refusing on
    # mismatch: none of them is exempt, and an unrecognised future key is not exempt either.
    for key in sorted(set(expected_config) - _EXTENSION_EXEMPT_KEYS):
        if checkpoint_config[key] != expected_config[key]:
            raise ScheduleExtensionError(
                f"a schedule extension may only lengthen the plan, but adapter config key "
                f"{key!r} differs: the checkpoint carries {checkpoint_config[key]!r} and the "
                f"new plan expects {expected_config[key]!r}"
            )

    extension = _reconcile_schedule(
        checkpoint_config=checkpoint_config,
        expected_config=expected_config,
        checkpoint_unit_cursor=checkpoint_unit_cursor,
        checkpoint_manifest_fingerprint=checkpoint_manifest_fingerprint,
        expected_manifest_fingerprint=expected_manifest_fingerprint,
        declaration=declaration,
    )
    _reconcile_batching(checkpoint_config, expected_config, checkpoint_unit_cursor)
    _reconcile_accumulation(checkpoint_config, expected_config)

    reconciled = dict(expected_config)
    for key in _EXTENSION_EXEMPT_KEYS & set(expected_config):
        reconciled[key] = checkpoint_config[key]
    return ExtensionReconciliation(
        expected_adapter_config=reconciled,
        expected_manifest_fingerprint=checkpoint_manifest_fingerprint,
        extension=extension,
    )


def _reconcile_schedule(
    *,
    checkpoint_config: Mapping[str, Any],
    expected_config: Mapping[str, Any],
    checkpoint_unit_cursor: int,
    checkpoint_manifest_fingerprint: str,
    expected_manifest_fingerprint: str,
    declaration: ScheduleExtensionDeclaration | None,
) -> ScheduleExtension:
    stored = checkpoint_config["schedule_inventory"]
    planned = expected_config["schedule_inventory"]
    if not isinstance(stored, Mapping) or not isinstance(planned, Mapping):
        raise ScheduleExtensionError("schedule inventory must be a mapping of role to units")
    if set(stored) != set(planned):
        raise ScheduleExtensionError("schedule inventory roles differ")

    # The held-out roles are NOT extended: they are a fixed evaluation set whose size is a
    # function of the split and the dose set, never of the epoch budget. A difference there is a
    # different experiment.
    for role in sorted(set(planned) - {"train"}):
        if stored[role] != planned[role]:
            raise ScheduleExtensionError(
                f"a schedule extension must not change the {role!r} role; it has "
                f"{len(stored[role])} units in the checkpoint and {len(planned[role])} in the "
                "new plan"
            )

    stored_train = stored["train"]
    planned_train = planned["train"]
    if not isinstance(checkpoint_unit_cursor, int) or isinstance(checkpoint_unit_cursor, bool):
        raise ScheduleExtensionError("checkpoint unit cursor must be an integer")
    if not 0 <= checkpoint_unit_cursor <= len(stored_train):
        raise ScheduleExtensionError("checkpoint unit cursor is outside its own train schedule")

    # A LONGER plan resumes; a SHORTER one is a different experiment, not a continuation. The
    # comparison is against the checkpoint's own DECLARED plan rather than against its cursor:
    # a plan that merely reaches the same length by a different tail is not an extension either.
    if len(planned_train) <= len(stored_train):
        raise ScheduleExtensionError(
            "a schedule extension must lengthen the training plan: the checkpoint was written "
            f"under {len(stored_train)} train units and the new plan has "
            f"{len(planned_train)}. Fewer epochs than have already been run is a different "
            "experiment, not a continuation"
        )

    # THE PREFIX CHECK. Element for element over the units the checkpoint actually walked;
    # anything past the cursor never determined a weight, so the new plan owns it.
    for index in range(checkpoint_unit_cursor):
        if stored_train[index] != planned_train[index]:
            raise ScheduleExtensionError(
                "the completed prefix of this checkpoint is not a prefix of the new plan; "
                f"train unit {index} of {checkpoint_unit_cursor} differs: the checkpoint "
                f"trained on {stored_train[index]!r} and the new plan schedules "
                f"{planned_train[index]!r}"
            )

    from_epochs = inventory_epochs(stored_train)
    if declaration is not None and from_epochs != declaration.from_maximum_epochs:
        raise ScheduleExtensionError(
            "schedule.extends.from_maximum_epochs does not describe the checkpoint: it declares "
            f"{declaration.from_maximum_epochs} and the checkpoint's own inventory spans "
            f"{from_epochs}"
        )
    return ScheduleExtension(
        from_manifest_fingerprint=checkpoint_manifest_fingerprint,
        to_manifest_fingerprint=expected_manifest_fingerprint,
        from_schedule_fingerprint=str(checkpoint_config["schedule_fingerprint"]),
        to_schedule_fingerprint=str(expected_config["schedule_fingerprint"]),
        from_maximum_epochs=from_epochs,
        to_maximum_epochs=inventory_epochs(planned_train),
        from_train_units=len(stored_train),
        to_train_units=len(planned_train),
        resumed_at_epoch=-1,
        resumed_at_unit_cursor=checkpoint_unit_cursor,
        crossed=exempt_crossings(checkpoint_config, expected_config),
    )


def _reconcile_batching(
    checkpoint_config: Mapping[str, Any],
    expected_config: Mapping[str, Any],
    checkpoint_unit_cursor: int,
) -> None:
    """Refuse any batching difference except the per-unit list growing at its tail.

    ``generation_batching`` is top-level exempt because its realised size list grows with the
    plan. Nothing else about it may move: ``config`` carries ``batch_size_100`` and
    ``maximum_batch_size``, which decide how many units share one optimizer step, so a checkpoint
    trained at one gradient batch is NOT continuable by a run configured for another. The
    comparison is closed-world -- a sub-key this module has never heard of must be equal too --
    because an always-on adoption cannot afford a future field to be exempt by omission.
    """
    stored = checkpoint_config.get("generation_batching")
    planned = expected_config.get("generation_batching")
    if stored is None and planned is None:
        return
    if not isinstance(stored, Mapping) or not isinstance(planned, Mapping):
        raise ScheduleExtensionError(
            "a schedule extension requires both plans to batch or neither to"
        )
    _require_equal_outside(
        stored, planned, _BATCHING_PREFIX_KEY, _BATCHING_DERIVED, "generation batching"
    )
    stored_sizes = list(stored.get(_BATCHING_PREFIX_KEY) or [])
    planned_sizes = list(planned.get(_BATCHING_PREFIX_KEY) or [])
    if planned_sizes[: len(stored_sizes)] != stored_sizes:
        raise ScheduleExtensionError(
            "the checkpoint's train batch sizes are not a prefix of the new plan's; the batch "
            "boundaries the completed units were committed on have moved"
        )
    # The cursor must still land on a batch boundary of the NEW plan, or the resume would
    # re-enter a batch it already committed half of.
    offset = 0
    for size in planned_sizes:
        if offset == checkpoint_unit_cursor:
            return
        offset += size
    if offset != checkpoint_unit_cursor:
        raise ScheduleExtensionError(
            f"unit cursor {checkpoint_unit_cursor} is not a batch boundary of the extended plan"
        )


def _reconcile_accumulation(
    checkpoint_config: Mapping[str, Any], expected_config: Mapping[str, Any]
) -> None:
    """Refuse any accumulation difference except the per-group list growing at its tail.

    Closed-world for the same reason as batching, and the stake is the same: ``units`` is how many
    generation batches one optimizer step covers, so it multiplies the gradient batch.
    """
    stored = checkpoint_config.get("gradient_accumulation")
    planned = expected_config.get("gradient_accumulation")
    if stored is None and planned is None:
        return
    if not isinstance(stored, Mapping) or not isinstance(planned, Mapping):
        raise ScheduleExtensionError(
            "a schedule extension requires both plans to accumulate or neither to"
        )
    _require_equal_outside(
        stored, planned, _ACCUMULATION_PREFIX_KEY, _ACCUMULATION_DERIVED, "gradient accumulation"
    )
    stored_groups = list(stored.get(_ACCUMULATION_PREFIX_KEY) or [])
    planned_groups = list(planned.get(_ACCUMULATION_PREFIX_KEY) or [])
    if planned_groups[: len(stored_groups)] != stored_groups:
        raise ScheduleExtensionError(
            "the checkpoint's accumulation groups are not a prefix of the new plan's; the "
            "optimizer-step boundaries the completed units were committed on have moved"
        )


def _require_equal_outside(
    stored: Mapping[str, Any],
    planned: Mapping[str, Any],
    growing: str,
    derived: Mapping[str, Any],
    label: str,
) -> None:
    """Every sub-key must be EQUAL except the growing list and the values derived from it.

    Closed-world in both directions. A sub-key this module has never heard of is not exempt by
    omission -- it must be equal -- and a sub-key that IS exempt is re-derived from the list it
    claims to summarise on both sides, so an identity asserting a batch count or an optimizer-step
    count its own plan does not have is refused rather than carried through the resume.
    """
    if set(stored) != set(planned):
        raise ScheduleExtensionError(
            f"a schedule extension must not change {label}; the recorded fields differ "
            f"(added {sorted(set(planned) - set(stored))}, "
            f"removed {sorted(set(stored) - set(planned))})"
        )
    for key in sorted(set(stored) - {growing} - set(derived)):
        if stored[key] != planned[key]:
            raise ScheduleExtensionError(
                f"a schedule extension must not change {label}; {key!r} differs: the checkpoint "
                f"carries {stored[key]!r} and the new plan expects {planned[key]!r}"
            )
    for block in (stored, planned):
        realised = list(block.get(growing) or [])
        for key, derive in sorted(derived.items()):
            if key in block and block[key] != derive(realised):
                raise ScheduleExtensionError(
                    f"{label} is internally inconsistent: {key!r} is {block[key]!r} but its own "
                    f"{growing!r} gives {derive(realised)!r}"
                )
