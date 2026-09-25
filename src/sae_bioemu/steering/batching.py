"""Deterministic, length-bucketed generation batching for steering schedules.

BioEmu sizes its own sampling batches as ``int(batch_size_100 * (100 / L) ** 2)``
(``bioemu/sample.py`` line 239) and natively batches *replicates* of one sequence
(``Batch.from_data_list([chemgraph] * batch_size)``, line 378). Batching across
*different* sequences works because the structure module densifies its input with
``to_dense_batch``, but the cost of a batch is set by its **padded maximum** length,
so a batch mixing a 40-residue and a 300-residue protein wastes most of its compute.
``docs/steering/DESIGN_DECISIONS.md`` D8 records both facts, and adds the
reproducibility condition: per-member initial seeding is sufficient only under the
noise-free DPM sampler, which
:func:`~sae_bioemu.bioemu.extractor.require_seeded_batch_denoiser` enforces.

This module turns a :class:`~sae_bioemu.steering.schedule.SteeringSchedule` into a
deterministic plan with two parts:

* an **order** for the train split that groups similar residue counts together, and
* the **batch sizes** the runner commits atomically.

Both are pure functions of the schedule, the per-sequence residue counts, and the
configuration, so two runs with the same inputs produce the same plan, and the plan
is recorded in the checkpoint identity and the run report.

Batch size 1 is a deliberate special case: with one member there is no padding, so
length bucketing has nothing to do. The plan then leaves the schedule order exactly
as built, which is what keeps a batch-size-1 run bit-identical to the pre-batching
path.

The held-out split gets its own plan (:func:`build_evaluation_batch_plan`). It is the
easy case: every unit of one test sequence is the *same* sequence at a different
requested level, so a batch is BioEmu's native replicate batching with zero padding
waste and nothing to bucket. The held-out order is never permuted, because the
per-unit cache filenames and the report inventory are indexed by position in it.

Because the two cases are that different, they take **separate configurations**: the
training plan and the evaluation plan are built from their own
:class:`GenerationBatchConfig`. D15 measured train/validation batching to be a net loss
while evaluation batching cuts denoiser invocations 3.27x, and one shared knob meant
turning off the loss also turned off the win.

**Gradient accumulation sits on top of generation batching, and is a different lever.**
A generation batch is how many structures one denoiser invocation produces; an
accumulation group is how many *consecutive* generation batches contribute to one
optimizer step. Batching trades padding for launch efficiency and is a compute choice;
accumulation averages the property gradient over more independently drawn structures and
is an optimization choice. :class:`GradientAccumulationPlan` composes them, so both numbers
are recorded. An accumulation group never crosses a training cycle boundary, because a
validation round is due there and the cursor has to be able to stop on it.

**The optimizer step is a fixed number of scheduled units, and the batches are planned
inside it.** ``int(batch_size_100 * (100 / L) ** 2)`` is a MEMORY statement, so one optimizer
step per generation batch made the step a function of how long its proteins happened to be:
a completed domain-motion run took 140 steps holding between 3 and 48 proteins each, mean
20.8. Adam's update is approximately scale-invariant in the gradient, so per-protein
influence varied by 16x for a reason that is not scientific, and loss re-weighting cannot
repair it because Adam divides by ``sqrt(v)``. :func:`plan_optimizer_step_spans` therefore
cuts each cycle into spans of :data:`UNIFORM_OPTIMIZER_STEP_UNITS` scheduled units first,
from the schedule alone, and :func:`plan_batch_sizes` fills each span with whatever
micro-batches memory allows -- 32 x 1 where the pool is short, 16 x 2 where it is long. The
realised per-step unit counts are reported, so the claim is checked rather than asserted.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .schedule import SteeringSchedule, SteeringUnit

_LOGGER = logging.getLogger(__name__)

#: BioEmu's own anchor: the batch size it would use for a 100-residue sequence
#: (``bioemu/sample.py`` line 76, ``batch_size_100: int = 10``).
DEFAULT_BATCH_SIZE_100 = 10

#: Hard ceiling on members per batch, so peak GPU memory stays bounded no matter how
#: short the proteins in a bucket are.
DEFAULT_MAXIMUM_BATCH_SIZE = 8

#: How many batches one length-bucketing window spans. The window is sorted by residue
#: count and then split, so the coarse schedule order survives while each batch stays
#: length-homogeneous.
DEFAULT_BUCKET_WINDOW_BATCHES = 8

#: D9: the steered path retains the score-model Jacobian at the single denoiser
#: iteration containing the intervened call, and that iteration makes two score calls,
#: so a gradient-retaining batch holds two score-model graphs. The inference-derived
#: heuristic is divided by this before the cap is applied.
DEFAULT_GRADIENT_GRAPH_DIVISOR = 2

#: Name of the composition rule recorded in the plan identity.
LENGTH_BUCKETED_RULE = "length_bucketed_within_cycle"

#: Name recorded when batching is configured but degenerates to one member per batch.
UNBATCHED_RULE = "schedule_order_single_member"

#: Name of the held-out composition rule: one sequence's own units, in schedule order.
SEQUENCE_GROUPED_RULE = "sequence_grouped_evaluation_units"

#: Generation batches whose gradients are summed into one optimizer step. ``1`` is the
#: pre-accumulation path exactly: one optimizer step per generation batch.
DEFAULT_GRADIENT_ACCUMULATION_UNITS = 1

#: How consecutive generation batches are grouped into one optimizer step.
ACCUMULATION_RULE = "consecutive_batches_within_cycle"

#: **Scheduled units in one optimizer step, for every steering property.** Not a knob: the
#: generation batch is sized for MEMORY (``int(batch_size_100 * (100 / L) ** 2)``), so with one
#: optimizer step per generation batch a step held anywhere from 3 to 48 proteins depending on
#: how long they are. Adam's update is approximately scale-invariant in the gradient, so a step
#: built from 3 proteins moves the weights about as much as one built from 48 and per-protein
#: influence varied by up to 16x for a reason that is not scientific. A sequence is this
#: project's independent biological unit and ``success@B`` is an unweighted mean over systems,
#: so the evaluation already treats proteins equally; this makes the training do the same.
#:
#: Re-weighting the loss cannot fix it -- Adam divides by ``sqrt(v)``, so scaling a group's loss
#: changes nothing -- and equal ``N`` per step is therefore the only lever.
UNIFORM_OPTIMIZER_STEP_UNITS = 32

#: How generation batches are grouped into one optimizer step under the uniform rule.
UNIFORM_STEP_RULE = "uniform_scheduled_units_within_cycle"

#: How each micro-step is weighted before its gradient is added to the accumulated one.
#: The property loss is a *mean* over its micro-batch, so a micro-batch of ``n`` units
#: carries weight ``n / N`` against the ``N`` units of its group; equal weights would
#: silently over-count the sequences that happened to land in a short micro-batch.
ACCUMULATION_WEIGHTING = "unit_count"


@dataclass(frozen=True)
class GenerationBatchConfig:
    """Knobs controlling how scheduled units are grouped into generation batches."""

    batch_size_100: int = DEFAULT_BATCH_SIZE_100
    maximum_batch_size: int = DEFAULT_MAXIMUM_BATCH_SIZE
    bucket_window_batches: int = DEFAULT_BUCKET_WINDOW_BATCHES
    gradient_graph_divisor: int = DEFAULT_GRADIENT_GRAPH_DIVISOR
    #: Forbid a training batch from ending inside a run of units sharing a ``draw_seed``.
    #: The paired dose term needs both members' graphs live in the same micro-batch, and a
    #: split group loses those pairs silently. Default false so every existing plan is
    #: byte-identical; the identity records it either way.
    keep_draw_groups_whole: bool = False

    def __post_init__(self) -> None:
        for name in (
            "batch_size_100",
            "maximum_batch_size",
            "bucket_window_batches",
            "gradient_graph_divisor",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.keep_draw_groups_whole, bool):
            raise ValueError("keep_draw_groups_whole must be a boolean")

    @property
    def buckets_lengths(self) -> bool:
        """True when the plan reorders at all; false at one member per batch."""
        return self.maximum_batch_size > 1

    def identity(self) -> dict[str, int]:
        """Return the primitive record bound into checkpoint identities and reports."""
        return {
            "batch_size_100": self.batch_size_100,
            "maximum_batch_size": self.maximum_batch_size,
            "bucket_window_batches": self.bucket_window_batches,
            "gradient_graph_divisor": self.gradient_graph_divisor,
            # Conditional: absent means the original planner, so a plan made before this
            # existed keeps a byte-identical identity.
            **({"keep_draw_groups_whole": True} if self.keep_draw_groups_whole else {}),
        }


#: The plan used when no batching is configured at all: exactly the pre-batching path.
UNBATCHED_CONFIG = GenerationBatchConfig(maximum_batch_size=1)


def generation_batch_config_from_payload(
    value: object, *, name: str = "generation_batch"
) -> GenerationBatchConfig | None:
    """Parse one batching configuration block, failing closed on extras.

    ``None`` selects the unbatched path, which is what the pre-batching runs used.
    ``name`` is the configuration key being parsed, so a run with separate training and
    evaluation blocks reports which one is malformed.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{name} must be an object with batch_size_100, maximum_batch_size, "
            "bucket_window_batches, and gradient_graph_divisor, or null for one unit per batch"
        )
    expected = {
        "batch_size_100",
        "maximum_batch_size",
        "bucket_window_batches",
        "gradient_graph_divisor",
    }
    # Optional and absent-means-false, so every shipped block keeps parsing unchanged.
    optional = {"keep_draw_groups_whole"}
    if set(value) - optional != expected:
        raise ValueError(f"{name} requires exactly {', '.join(sorted(expected))}")
    parsed = {item: value[item] for item in sorted(expected)}
    if "keep_draw_groups_whole" in value:
        parsed["keep_draw_groups_whole"] = value["keep_draw_groups_whole"]
    return GenerationBatchConfig(**parsed)


def heuristic_batch_size(
    max_residues: int, config: GenerationBatchConfig, *, retains_gradient: bool
) -> int:
    """Return BioEmu's own length heuristic, gradient-adjusted and capped.

    ``int(batch_size_100 * (100 / L) ** 2)`` is reproduced verbatim from
    ``bioemu/sample.py`` line 239 rather than replaced by a constant; ``L`` is the
    **padded** maximum residue count of the batch, which is what actually costs.
    """
    if not isinstance(max_residues, int) or isinstance(max_residues, bool) or max_residues < 1:
        raise ValueError("max_residues must be a positive integer")
    if not isinstance(config, GenerationBatchConfig):
        raise TypeError("config must be a GenerationBatchConfig")
    if not isinstance(retains_gradient, bool):
        raise TypeError("retains_gradient must be a boolean")
    raw = int(config.batch_size_100 * (100.0 / max_residues) ** 2)
    if retains_gradient:
        raw //= config.gradient_graph_divisor
    return max(1, min(raw, config.maximum_batch_size))


def _residue_count(counts: Mapping[str, int], unit: SteeringUnit) -> int:
    value = counts.get(unit.sequence_id)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(
            "every scheduled sequence needs a positive residue count for batching; "
            f"{unit.sequence_id!r} has {value!r}"
        )
    return value


def cycle_segments(units: Sequence[SteeringUnit]) -> tuple[tuple[int, int], ...]:
    """Return the ``[start, stop)`` span of each contiguous training cycle."""
    segments: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(units) + 1):
        if index == len(units) or units[index].cycle != units[start].cycle:
            segments.append((start, index))
            start = index
    cycles = [units[begin].cycle for begin, _stop in segments]
    if cycles != sorted(set(cycles)) or cycles != list(range(len(cycles))):
        raise ValueError("train schedule cycles must be contiguous and zero-based to batch")
    return tuple(segments)


def order_train_units_for_batching(
    units: Sequence[SteeringUnit],
    residue_counts: Mapping[str, int],
    config: GenerationBatchConfig,
) -> tuple[SteeringUnit, ...]:
    """Reorder each training cycle so neighbouring units have similar residue counts.

    Only positions *within* a cycle move, so every cycle still contains every training
    sequence exactly once and cycles stay contiguous -- the two properties the schedule
    validator and the epoch-boundary validation cadence rely on.

    The reordering is windowed rather than a global sort: a window of
    ``bucket_window_batches * maximum_batch_size`` consecutive scheduled units is sorted
    by residue count and split into batches, so the schedule's own hash-derived sequence
    order still governs the coarse order of an epoch and length correlates with update
    position only inside a window.
    """
    units = tuple(units)
    if not isinstance(config, GenerationBatchConfig):
        raise TypeError("config must be a GenerationBatchConfig")
    if not config.buckets_lengths:
        # One member per batch means no padding, so bucketing would only perturb the
        # update order for nothing. Leaving the order alone is what makes batch size 1
        # bit-identical to the pre-batching path.
        return units
    window = config.bucket_window_batches * config.maximum_batch_size
    ordered: list[SteeringUnit] = []
    for start, stop in cycle_segments(units):
        cycle = units[start:stop]
        for offset in range(0, len(cycle), window):
            chunk = cycle[offset : offset + window]
            bucketed = sorted(
                enumerate(chunk),
                key=lambda item: (_residue_count(residue_counts, item[1]), item[0]),
            )
            ordered.extend(unit for _position, unit in bucketed)
    return tuple(ordered)


def order_validation_units_for_batching(
    units: Sequence[SteeringUnit],
    residue_counts: Mapping[str, int],
    config: GenerationBatchConfig,
) -> tuple[SteeringUnit, ...]:
    """Reorder validation so neighbouring units have similar residue counts.

    Validation is one flat block -- it has no cycles to keep contiguous and no epoch
    boundary a batch could straddle -- so it is windowed exactly like a single training
    cycle. Ordering is deterministic: residue count, then the unit's own canonical
    ``(sequence_id, cycle, visit)`` position as a tiebreak, so two runs of the same schedule
    walk validation in the same order and the recorded batch sizes reproduce.

    The set is unchanged; only the order moves. ``_validate_schedule`` pins validation
    membership rather than order for exactly this reason.
    """
    units = tuple(units)
    if not isinstance(config, GenerationBatchConfig):
        raise TypeError("config must be a GenerationBatchConfig")
    if not config.buckets_lengths:
        return units
    canonical = sorted(units, key=lambda unit: (unit.sequence_id, unit.cycle, unit.visit))
    window = config.bucket_window_batches * config.maximum_batch_size
    ordered: list[SteeringUnit] = []
    for offset in range(0, len(canonical), window):
        chunk = canonical[offset : offset + window]
        bucketed = sorted(
            enumerate(chunk),
            key=lambda item: (_residue_count(residue_counts, item[1]), item[0]),
        )
        ordered.extend(unit for _position, unit in bucketed)
    return tuple(ordered)


def draw_group_runs(units: Sequence[SteeringUnit]) -> tuple[tuple[int, int], ...]:
    """Return contiguous ``[start, stop)`` runs of units sharing one ``draw_seed``.

    A unit without a draw is its own run of one, so a schedule built before ``draw_seed``
    existed decomposes into singletons and every group-aware path reduces to the original.
    """
    runs: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(units):
        draw = units[cursor].draw_seed
        stop = cursor + 1
        if draw is not None:
            while stop < len(units) and units[stop].draw_seed == draw:
                stop += 1
        runs.append((cursor, stop))
        cursor = stop
    return tuple(runs)


def plan_optimizer_step_spans(
    units: Sequence[SteeringUnit],
    *,
    step_units: int = UNIFORM_OPTIMIZER_STEP_UNITS,
    segments: Sequence[tuple[int, int]] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Return the ``[start, stop)`` span of each optimizer step, at ``step_units`` units each.

    This is the primary statement of the uniform rule and it is made BEFORE any memory
    consideration: the spans are a function of the schedule alone, and the generation batches
    are sized inside them afterwards. That order is what makes per-protein influence
    independent of protein length -- the reverse order is the defect, because
    ``int(batch_size_100 * (100 / L) ** 2)`` is a memory statement and made the optimizer step
    a function of how long the proteins in it happened to be.

    **A span never crosses a segment boundary.** ``segments`` are the training cycles: a
    validation round is due at each cycle end, the persisted cursor has to be able to stop
    there, and ``_validate_completion_history_record`` requires every recorded round's
    ``train_cursor`` to equal a cycle boundary exactly. Carrying a cycle's remainder into the
    next cycle would therefore not merely delay a validation round, it would make the run
    unresumable and its completion record invalid.

    **A span never splits a run of units sharing a ``draw_seed``.** The paired dose term
    differences two levels generated from one noise draw, so the members of a draw group must
    be able to share a micro-batch; a step boundary inside a group would make that impossible
    for the pair it cut, silently. Steps are therefore tiled over whole draw groups. A schedule
    emitted visit-major has no contiguous groups at all (a group's members sit
    ``len(sequences)`` apart), so its groups are singletons and the constraint is vacuous.

    **The remainder.** A cycle's unit count is rarely a multiple of ``step_units`` -- the
    domain-motion pool is ``291 systems x 2 levels = 582`` units per cycle, which is
    ``18 x 32 + 6``. The six are NOT left as a short final step, because a step of 6 would give
    those proteins 5.3x the per-protein influence of every other step, once per cycle, which is
    the defect this function exists to remove reappearing at the end of every epoch; and they
    are not dropped, because the schedule order repeats across cycles, so the same proteins
    would be excluded from training in every epoch. They are instead spread UPWARDS across the
    cycle's steps: the cycle is cut into ``k = n // step_units`` spans at evenly spaced
    cumulative targets, so no step is ever short and every step holds ``step_units`` units up
    to the one draw group that integer division forces. Measured per cycle:

    ========================================= ===================================
    pool                                      realised units per optimizer step
    ========================================= ===================================
    domain motion, 291 x 2 = 582              15 x 32, 3 x 34
    cryptic, 1974 x 2 = 3948                  117 x 32, 6 x 34
    radius of gyration, 800 x 5 = 4000        125 x 32
    ========================================= ===================================

    A residual spread of 1.06x, against the 16x it replaces, and exactly ``step_units``
    whenever the cycle divides. A cycle shorter than twice the target -- a smoke run truncated
    to one unit, or a deliberately tiny overfitting pool -- is one step holding the whole
    cycle, which is uniform by construction; there are no units to invent.
    """
    if not isinstance(step_units, int) or isinstance(step_units, bool) or step_units < 1:
        raise ValueError("step_units must be a positive integer")
    units = tuple(units)
    if not units:
        return ()
    spans = tuple(segments) if segments is not None else ((0, len(units)),)
    if spans[0][0] != 0 or spans[-1][1] != len(units):
        raise ValueError("optimizer step segments must cover the units exactly")
    steps: list[tuple[int, int]] = []
    for start, stop in spans:
        atoms = [
            (run_start + start, run_stop + start)
            for run_start, run_stop in draw_group_runs(units[start:stop])
        ]
        for span_start, span_stop in _even_step_spans(atoms, step_units):
            steps.append((span_start, span_stop))
    cursor = 0
    for span_start, span_stop in steps:
        if span_start != cursor or span_stop <= span_start:
            raise RuntimeError("planned optimizer steps do not cover every scheduled unit")
        cursor = span_stop
    if cursor != len(units):
        raise RuntimeError("planned optimizer steps do not cover every scheduled unit")
    return tuple(steps)


def _even_step_spans(
    atoms: Sequence[tuple[int, int]], step_units: int
) -> tuple[tuple[int, int], ...]:
    """Cut one cycle's draw groups into spans of ``step_units`` units, as evenly as possible."""
    if not atoms:
        return ()
    start = atoms[0][0]
    stop = atoms[-1][1]
    total = stop - start
    # FLOOR, not nearest. Both spread the remainder, but the floor spreads it upwards: the
    # mean span is ``total / (total // step_units) >= step_units``, so no step is ever SHORT.
    # That asymmetry is the point -- a step holding fewer units than the others gives its
    # proteins more influence each, which is the defect, while a step holding one draw group
    # more gives them slightly less and costs nothing but a 1.06x spread. Measured on the
    # 1836-protein cryptic retention, nearest emits eight steps of 30 against 222 of 32 where
    # the floor emits 32 and 34; on a cycle shorter than twice the target the floor emits one
    # step holding the whole cycle, which is perfectly uniform by construction.
    sizes = {span_stop - span_start for span_start, span_stop in atoms}
    if len(sizes) == 1:
        # Every draw group is the same size, which is what a schedule emits: a group is one
        # sequence's level set. Then the spread is minimised by partitioning the GROUPS
        # evenly rather than the units -- a five-level paired cycle of 4000 units cut on unit
        # targets lands 150 steps on 30 units and 100 on 35, where the same cycle cut on group
        # targets lands 131 on 30 and 2 on 35, because 32 is not a multiple of 5 and every cut
        # against a unit target inherits that mismatch.
        group = sizes.pop()
        per_step = max(1, step_units // group)
        count = max(1, len(atoms) // per_step)
        return _partition(atoms, count, len(atoms), by_groups=True)
    # Draw groups of different sizes: no group count describes the step, so the targets are
    # unit counts and the groups stay whole around them.
    return _partition(atoms, max(1, total // step_units), total, by_groups=False)


def _partition(
    atoms: Sequence[tuple[int, int]], count: int, measure: int, *, by_groups: bool
) -> tuple[tuple[int, int], ...]:
    """Cut ``atoms`` into ``count`` spans at evenly spaced cumulative ``measure`` targets."""
    start = atoms[0][0]
    stop = atoms[-1][1]
    if count == 1:
        return ((start, stop),)
    spans: list[tuple[int, int]] = []
    cursor = start
    index = 0
    for boundary in range(1, count):
        # Evenly spaced cumulative targets rather than a fixed stride, so the rounding error
        # cannot accumulate: every cut is placed against the cycle's own total.
        target = (boundary * measure + count // 2) // count
        if by_groups:
            index = target
        else:
            while index < len(atoms) and atoms[index][1] - start <= target:
                index += 1
        # The cut lands on the first whole draw group that reaches the target. ``index`` may
        # already sit past it when a group straddles the target, which is what keeps groups
        # whole; an empty span is impossible because a group is at least one unit.
        cut = atoms[index][0] if index < len(atoms) else stop
        if cut <= cursor:
            continue
        spans.append((cursor, cut))
        cursor = cut
    if cursor < stop:
        spans.append((cursor, stop))
    return tuple(spans)


def plan_batch_sizes(
    units: Sequence[SteeringUnit],
    residue_counts: Mapping[str, int],
    config: GenerationBatchConfig,
    *,
    retains_gradient: bool,
    segments: Sequence[tuple[int, int]] | None = None,
    keep_draw_groups_whole: bool = False,
) -> tuple[int, ...]:
    """Return the deterministic batch sizes covering ``units`` in order.

    A batch grows greedily while the padded-maximum heuristic still admits another
    member, and never crosses a segment boundary, so a training batch cannot straddle
    an epoch boundary and hide a due validation round.

    ``keep_draw_groups_whole`` additionally forbids a batch from ending inside a run of units
    that share a ``draw_seed``. The paired dose term differentiates the *difference* between
    two levels, so both members must have live graphs in the same micro-batch; a group split
    across batches loses those pairs **silently**. That is length-dependent and therefore
    arbitrary: ``int(batch_size_100 * (100 / L) ** 2)`` gives exactly 5 at L = 299 and 4 at
    L = 301, so one residue decides whether a sequence contributes its pairs.

    **This costs memory, and the amount is length-dependent.** The batch still respects
    ``maximum_batch_size``, but it may exceed the *length heuristic* that normally shrinks a
    batch for long proteins. Measured effect on a 5-level group:

    ====== ================= ============ ==================
    length heuristic members kept whole   pairs recovered
    ====== ================= ============ ==================
    299    5                 5            40/40 (no change)
    301    4                 5            20 -> 40
    400    2                 5            8 -> 40
    600    1                 5            0 -> 40
    ====== ================= ============ ==================

    At L = 600 that is five times the members the heuristic wanted. Extrapolating the batching
    benchmark (9.02 GiB at 16 members of 94 residues on a 44.5 GiB A40, cost ~ ``L^2``), five
    members of 600 residues would not fit. So a warning is emitted whenever the group exceeds
    the heuristic, naming the multiple, and ``maximum_batch_size`` remains the operator's
    memory statement. A group larger than that ceiling raises rather than silently splitting.
    """
    units = tuple(units)
    if not units:
        return ()
    spans = tuple(segments) if segments is not None else ((0, len(units)),)
    if spans[0][0] != 0 or spans[-1][1] != len(units):
        raise ValueError("batch segments must cover the units exactly")
    sizes: list[int] = []
    for start, stop in spans:
        cursor = start
        if keep_draw_groups_whole:
            runs = [
                (run_start + start, run_stop + start)
                for run_start, run_stop in draw_group_runs(units[start:stop])
            ]
            index = 0
            while index < len(runs):
                run_start, run_stop = runs[index]
                span = run_stop - run_start
                if span > config.maximum_batch_size:
                    raise ValueError(
                        f"a draw group of {span} units cannot be kept whole under "
                        f"maximum_batch_size={config.maximum_batch_size}; raise the ceiling or "
                        "disable keep_draw_groups_whole and accept that the paired dose term "
                        "will see only part of its pairs"
                    )
                padded_max = max(
                    _residue_count(residue_counts, unit) for unit in units[run_start:run_stop]
                )
                allowed_alone = heuristic_batch_size(
                    padded_max, config, retains_gradient=retains_gradient
                )
                if span > allowed_alone:
                    _LOGGER.warning(
                        "keeping a %d-unit draw group whole at %d residues exceeds the length "
                        "heuristic of %d by %.1fx; peak generation memory scales roughly as "
                        "members x residues^2, so lower maximum_batch_size if this runs out",
                        span,
                        padded_max,
                        allowed_alone,
                        span / allowed_alone,
                    )
                size = span
                index += 1
                # Pack further whole groups in while the heuristic and the ceiling both allow.
                while index < len(runs):
                    next_start, next_stop = runs[index]
                    candidate_span = next_stop - next_start
                    candidate = max(
                        padded_max,
                        *(
                            _residue_count(residue_counts, unit)
                            for unit in units[next_start:next_stop]
                        ),
                    )
                    allowed = heuristic_batch_size(
                        candidate, config, retains_gradient=retains_gradient
                    )
                    if size + candidate_span > allowed:
                        break
                    padded_max = candidate
                    size += candidate_span
                    index += 1
                sizes.append(size)
            continue
        while cursor < stop:
            padded_max = _residue_count(residue_counts, units[cursor])
            size = 1
            while cursor + size < stop:
                candidate = max(padded_max, _residue_count(residue_counts, units[cursor + size]))
                if size + 1 > heuristic_batch_size(
                    candidate, config, retains_gradient=retains_gradient
                ):
                    break
                padded_max = candidate
                size += 1
            sizes.append(size)
            cursor += size
    if sum(sizes) != len(units):
        raise RuntimeError("planned batch sizes do not cover every scheduled unit")
    return tuple(sizes)


def batch_offsets(sizes: Sequence[int]) -> dict[int, int]:
    """Return ``{start offset: batch size}`` for a batch-size sequence."""
    offsets: dict[int, int] = {}
    cursor = 0
    for size in sizes:
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("batch sizes must be positive integers")
        offsets[cursor] = size
        cursor += size
    return offsets


@dataclass(frozen=True)
class SteeringBatchPlan:
    """A schedule ordered for batching, with the atomic batch sizes that cover it."""

    schedule: SteeringSchedule
    train_batch_sizes: tuple[int, ...]
    validation_batch_sizes: tuple[int, ...]
    config: GenerationBatchConfig

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, SteeringSchedule):
            raise TypeError("schedule must be a SteeringSchedule")
        if not isinstance(self.config, GenerationBatchConfig):
            raise TypeError("config must be a GenerationBatchConfig")
        if sum(self.train_batch_sizes) != len(self.schedule.train):
            raise ValueError("train batch sizes must cover the train schedule exactly")
        if sum(self.validation_batch_sizes) != len(self.schedule.validation):
            raise ValueError("validation batch sizes must cover the validation schedule exactly")
        batch_offsets(self.train_batch_sizes)
        batch_offsets(self.validation_batch_sizes)

    @property
    def rule(self) -> str:
        return LENGTH_BUCKETED_RULE if self.config.buckets_lengths else UNBATCHED_RULE

    def identity(self) -> dict[str, object]:
        """Return the auditable batching record for the checkpoint identity and report.

        The batch sizes themselves are included, not just the configuration: the
        optimizer step now covers a whole batch, so two runs whose batches differ are
        different optimizations and must never share a checkpoint identity.
        """
        return {
            "rule": self.rule,
            "config": self.config.identity(),
            "train_batch_sizes": list(self.train_batch_sizes),
            "validation_batch_sizes": list(self.validation_batch_sizes),
            "train_batch_count": len(self.train_batch_sizes),
            "validation_batch_count": len(self.validation_batch_sizes),
            "maximum_train_batch_size": max(self.train_batch_sizes, default=0),
            "maximum_validation_batch_size": max(self.validation_batch_sizes, default=0),
        }


@dataclass(frozen=True)
class GradientAccumulationPlan:
    """Which consecutive generation batches share one optimizer step.

    ``groups`` holds, per optimizer step, the sizes of the generation batches whose
    gradients are accumulated before the step is taken. At
    ``units == 1`` every group holds exactly one batch, which is the pre-accumulation path.

    The plan is a pure function of the train batch sizes, the accumulation knob, and the
    cycle boundaries, so it is deterministic and can be rebuilt for an identity check
    rather than carried around.
    """

    units: int
    groups: tuple[tuple[int, ...], ...]
    #: The uniform scheduled-unit target every optimizer step was planned to hold, or ``None``
    #: for a plan built from a fixed number of generation batches instead. Present means
    #: ``units`` is the realised MAXIMUM micro-batch count of a step rather than a knob: under
    #: the uniform rule the step is a fixed number of scheduled units and the micro-batch count
    #: is whatever memory allowed inside it.
    step_units: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.units, int) or isinstance(self.units, bool) or self.units < 1:
            raise ValueError("gradient_accumulation_units must be a positive integer")
        if self.step_units is not None and (
            not isinstance(self.step_units, int)
            or isinstance(self.step_units, bool)
            or self.step_units < 1
        ):
            raise ValueError("step_units must be a positive integer or None")
        if not isinstance(self.groups, tuple) or any(
            not isinstance(group, tuple) or not group for group in self.groups
        ):
            raise ValueError("each accumulation group must be a non-empty tuple of batch sizes")
        for group in self.groups:
            if len(group) > self.units:
                raise ValueError("an accumulation group cannot hold more than units batches")
            if any(
                not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in group
            ):
                raise ValueError("batch sizes must be positive integers")

    @property
    def accumulates(self) -> bool:
        """True when one optimizer step spans more than one generation batch."""
        return self.units > 1

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        """The underlying generation batch sizes, in order."""
        return tuple(size for group in self.groups for size in group)

    @property
    def group_unit_sizes(self) -> tuple[int, ...]:
        """Scheduled units covered by each optimizer step: the effective batch size."""
        return tuple(sum(group) for group in self.groups)

    @property
    def total_units(self) -> int:
        return sum(self.group_unit_sizes)

    def offsets(self) -> dict[int, int]:
        """Return ``{unit offset: units in that optimizer step}``.

        This is the resume boundary map: with accumulation the atomic unit is the
        accumulation group, not the generation batch, so a cursor inside a group is not a
        resumable position.
        """
        return batch_offsets(self.group_unit_sizes)

    def groups_by_offset(self) -> dict[int, tuple[int, ...]]:
        """Return ``{unit offset: micro-batch sizes of that optimizer step}``."""
        mapping: dict[int, tuple[int, ...]] = {}
        cursor = 0
        for group in self.groups:
            mapping[cursor] = group
            cursor += sum(group)
        return mapping

    def summary(self) -> dict[str, object]:
        """Return the compact composition record for the run report.

        ``minimum_optimizer_step_units`` and ``maximum_optimizer_step_units`` are computed from
        the REALISED group sizes, not restated from the knob, so they are the one-line proof
        that the uniform rule held: both read :data:`UNIFORM_OPTIMIZER_STEP_UNITS` on a run
        whose cycles divide by it, and the histogram beside them says exactly how the remainder
        landed when they do not.
        """
        sizes = self.group_unit_sizes
        uniform = self.step_units is not None
        return {
            "units": self.units,
            "rule": UNIFORM_STEP_RULE if uniform else ACCUMULATION_RULE,
            "weighting": ACCUMULATION_WEIGHTING,
            "optimizer_step_count": len(self.groups),
            "maximum_optimizer_step_units": max(sizes, default=0),
            "minimum_optimizer_step_units": min(sizes, default=0),
            # Conditional, so a plan built from a batch count keeps the exact record it had
            # before the uniform rule existed.
            **(
                {
                    "target_optimizer_step_units": self.step_units,
                    "optimizer_step_unit_counts": [
                        [size, sizes.count(size)] for size in sorted(set(sizes))
                    ],
                }
                if uniform
                else {}
            ),
        }

    def identity(self) -> dict[str, object]:
        """Return the auditable record bound into the checkpoint identity.

        The realized group composition is included, not just the knob: one optimizer step
        now spans ``batch size x accumulation units`` scheduled units, so two runs whose
        groups differ are different optimizations exactly as two runs with different batch
        sizes are (D8), and must never resume into one another.
        """
        return {
            **self.summary(),
            "train_group_batch_sizes": [list(group) for group in self.groups],
            "train_group_unit_sizes": list(self.group_unit_sizes),
        }


def plan_gradient_accumulation(
    batch_sizes: Sequence[int],
    accumulation_units: int,
    *,
    boundaries: Sequence[int] = (),
) -> GradientAccumulationPlan:
    """Group consecutive generation batches into optimizer steps, respecting boundaries.

    ``boundaries`` are unit offsets a group must not cross -- the training cycle ends,
    because a validation round is due there and the persisted cursor has to be able to
    stop on it. Every boundary must already be a generation batch boundary, or the plan
    fails closed rather than silently emitting a group that straddles it.

    A group at the end of a cycle may hold fewer than ``accumulation_units`` batches. That
    is recorded rather than padded, and the unit-count weighting below keeps such a group
    an honest mean over its own members.
    """
    if (
        not isinstance(accumulation_units, int)
        or isinstance(accumulation_units, bool)
        or accumulation_units < 1
    ):
        raise ValueError("gradient_accumulation_units must be a positive integer")
    sizes = tuple(batch_sizes)
    batch_offsets(sizes)
    cuts = set()
    cursor = 0
    for size in sizes:
        cursor += size
        cuts.add(cursor)
    for boundary in boundaries:
        if boundary not in cuts and boundary != 0:
            raise ValueError(
                f"accumulation boundary {boundary} is not a generation batch boundary; an "
                "optimizer step would straddle a cycle boundary and hide a due validation "
                "round"
            )
    groups: list[tuple[int, ...]] = []
    current: list[int] = []
    cursor = 0
    stops = set(boundaries)
    for size in sizes:
        current.append(size)
        cursor += size
        if len(current) == accumulation_units or cursor in stops:
            groups.append(tuple(current))
            current = []
    if current:
        groups.append(tuple(current))
    return GradientAccumulationPlan(accumulation_units, tuple(groups))


def plan_uniform_optimizer_steps(
    batch_sizes: Sequence[int],
    step_unit_sizes: Sequence[int],
    *,
    step_units: int = UNIFORM_OPTIMIZER_STEP_UNITS,
) -> GradientAccumulationPlan:
    """Group generation batches into the optimizer steps ``step_unit_sizes`` describes.

    The inverse composition of :func:`plan_gradient_accumulation`: there the step is a fixed
    number of generation BATCHES and the units per step follow from how long the proteins were,
    here the step is a fixed number of scheduled UNITS and the batch count per step follows from
    how much memory they needed. A step therefore holds two micro-batches of 16 where the pool
    is long and one of 32 where it is short, and moves the adapter by the same amount either
    way.

    Fails closed if a step boundary is not also a generation batch boundary: a micro-batch
    straddling two optimizer steps would have its gradient counted in the first and its units
    counted in the second, and the ``n / N`` weighting would silently stop describing the step.
    """
    if not isinstance(step_units, int) or isinstance(step_units, bool) or step_units < 1:
        raise ValueError("step_units must be a positive integer")
    sizes = tuple(batch_sizes)
    steps = tuple(step_unit_sizes)
    batch_offsets(sizes)
    batch_offsets(steps)
    if sum(sizes) != sum(steps):
        raise ValueError("optimizer steps and generation batches must cover the same units")
    groups: list[tuple[int, ...]] = []
    current: list[int] = []
    covered = 0
    index = 0
    for span in steps:
        target = covered + span
        while covered < target:
            if index >= len(sizes):
                raise RuntimeError("generation batches do not cover the planned optimizer steps")
            size = sizes[index]
            if covered + size > target:
                raise ValueError(
                    f"a generation batch of {size} units straddles the optimizer step boundary "
                    f"at unit {target}; the step spans must bound the batch plan"
                )
            current.append(size)
            covered += size
            index += 1
        groups.append(tuple(current))
        current = []
    if index != len(sizes):
        raise RuntimeError("generation batches do not cover the planned optimizer steps")
    return GradientAccumulationPlan(
        max((len(group) for group in groups), default=1),
        tuple(groups),
        step_units=step_units,
    )


def evaluation_sequence_segments(units: Sequence[SteeringUnit]) -> tuple[tuple[int, int], ...]:
    """Return the ``[start, stop)`` span of each held-out sequence's own units.

    ``schedule.py`` emits the held-out split as one contiguous run of visits per
    sequence, and the report inventory, the per-unit cache filenames, and the paired
    off/on keys are all indexed by position in that order. This therefore only *reads*
    the order and fails closed if one sequence's units are not contiguous, rather than
    reordering anything.
    """
    segments: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(units) + 1):
        if index == len(units) or units[index].sequence_id != units[start].sequence_id:
            segments.append((start, index))
            start = index
    identifiers = [units[begin].sequence_id for begin, _stop in segments]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "held-out evaluation units of one sequence must be contiguous to batch; "
            "a batch holds one sequence's own visits so that padding is a no-op"
        )
    return tuple(segments)


@dataclass(frozen=True)
class SteeringEvaluationBatchPlan:
    """The atomic generation batches covering a held-out split, in schedule order.

    Every member of a batch is the *same* sequence at a different requested level and
    with its own seed, which is BioEmu's native replicate batching
    (``Batch.from_data_list([chemgraph] * B)``, ``bioemu/sample.py`` line 378) and
    therefore carries no padding at all. Nothing is reordered: the held-out order is
    load bearing for the per-unit cache and the report inventory, and there is nothing
    to bucket when every member has the same residue count.
    """

    units: tuple[SteeringUnit, ...]
    batch_sizes: tuple[int, ...]
    config: GenerationBatchConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, GenerationBatchConfig):
            raise TypeError("config must be a GenerationBatchConfig")
        if sum(self.batch_sizes) != len(self.units):
            raise ValueError("evaluation batch sizes must cover the held-out split exactly")
        batch_offsets(self.batch_sizes)
        cursor = 0
        for size in self.batch_sizes:
            members = self.units[cursor : cursor + size]
            if len({unit.sequence_id for unit in members}) != 1:
                raise ValueError("an evaluation batch must hold exactly one sequence's units")
            cursor += size

    @property
    def rule(self) -> str:
        return SEQUENCE_GROUPED_RULE if self.config.buckets_lengths else UNBATCHED_RULE

    def identity(self) -> dict[str, object]:
        """Return the auditable record bound into the evaluation identity.

        The sizes themselves are included, not just the configuration: batching is not
        bit-identical (D8), so two runs with different batches are different runs and
        must not resume into one another's per-unit cache.
        """
        return {
            "rule": self.rule,
            "config": self.config.identity(),
            "batch_sizes": list(self.batch_sizes),
            "batch_count": len(self.batch_sizes),
            "maximum_batch_size": max(self.batch_sizes, default=0),
        }


def build_evaluation_batch_plan(
    units: Sequence[SteeringUnit],
    residue_counts: Mapping[str, int],
    config: GenerationBatchConfig,
) -> SteeringEvaluationBatchPlan:
    """Group each held-out sequence's units into the batches the runner commits.

    A batch never crosses a sequence boundary, so ``member_nodes`` is uniform and
    :func:`~sae_bioemu.bioemu.extractor.pad_member_coordinates` is a no-op. The size
    within a sequence still follows BioEmu's own length heuristic, gradient-adjusted:
    the held-out arms build the same two score-model graphs the training path does,
    because ``capture_or_steer_from_denoiser`` requires differentiable final positions.
    """
    units = tuple(units)
    if not isinstance(residue_counts, Mapping):
        raise TypeError("residue_counts must be a mapping of sequence_id to residue count")
    if not isinstance(config, GenerationBatchConfig):
        raise TypeError("config must be a GenerationBatchConfig")
    if not units:
        return SteeringEvaluationBatchPlan((), (), config)
    if not config.buckets_lengths:
        return SteeringEvaluationBatchPlan(units, (1,) * len(units), config)
    sizes = plan_batch_sizes(
        units,
        residue_counts,
        config,
        retains_gradient=True,
        segments=evaluation_sequence_segments(units),
    )
    return SteeringEvaluationBatchPlan(units, sizes, config)


def build_generation_batch_plan(
    schedule: SteeringSchedule,
    residue_counts: Mapping[str, int],
    config: GenerationBatchConfig,
) -> SteeringBatchPlan:
    """Order the train split for bucketing and size every train batch.

    The train split is treated as gradient-retaining: it builds the two score-model graphs
    D9 describes.

    **Validation is bucketed and batched too.** It used to get one member per batch
    regardless of ``config``, because ``_validate_schedule`` demanded canonical
    ``(sequence_id, cycle, visit)`` order for every non-train split, and chunking validation
    contiguously in that order mixes an ``L=61`` protein with an ``L=297`` one and pads to
    the maximum -- ``1.73x`` worse than not batching, since cost scales as ``L^2``. The
    validator now pins validation *membership* (every sequence at every validation level
    exactly once) while leaving its order free, so validation buckets exactly as training
    does and reaches the same measured ``1.90x`` at batch 4. That matters because validation
    had become the per-epoch bottleneck: roughly 10 minutes against 8 of training.

    Validation has no cycles to keep contiguous and retains no gradient, so it is bucketed as
    a single segment with ``retains_gradient=False``.
    """
    if not isinstance(schedule, SteeringSchedule):
        raise TypeError("schedule must be a SteeringSchedule")
    if not isinstance(residue_counts, Mapping):
        raise TypeError("residue_counts must be a mapping of sequence_id to residue count")
    if not isinstance(config, GenerationBatchConfig):
        raise TypeError("config must be a GenerationBatchConfig")
    ordered_train = order_train_units_for_batching(schedule.train, residue_counts, config)
    ordered_validation = order_validation_units_for_batching(
        schedule.validation, residue_counts, config
    )
    ordered = SteeringSchedule(ordered_train, ordered_validation, schedule.test)
    # The segments a training batch may not cross are the OPTIMIZER STEPS, not the cycles.
    # Cycles are still respected, because a step never crosses one, and bounding the batches by
    # the steps is what makes the step a whole number of micro-batches: a micro-batch that
    # straddled two steps would be counted in one and weighted against the other. The micro-batch
    # inside a step is sized exactly as it always was, by the length heuristic and the ceiling,
    # so memory is unchanged and only the grouping moved.
    train_sizes = plan_batch_sizes(
        ordered_train,
        residue_counts,
        config,
        retains_gradient=True,
        segments=(
            plan_optimizer_step_spans(ordered_train, segments=cycle_segments(ordered_train))
            if ordered_train
            else None
        ),
        keep_draw_groups_whole=config.keep_draw_groups_whole,
    )
    validation_sizes = plan_batch_sizes(
        ordered_validation,
        residue_counts,
        config,
        retains_gradient=False,
    )
    return SteeringBatchPlan(ordered, train_sizes, validation_sizes, config)
