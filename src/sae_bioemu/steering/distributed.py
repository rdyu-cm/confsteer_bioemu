"""Single-node data-parallel execution for the ordinary steering training loop.

**What this buys, and why it is the same lever as gradient accumulation.**
Accumulation and data parallelism are the same operation: sum the gradients of ``N``
micro-batches, then take one optimizer step. Accumulation does it sequentially on one
device and pays for the larger effective batch with optimizer *steps*; data parallelism
does it concurrently on ``W`` devices and pays with *devices*. Measured on the two
completed arms, the constraint is gradient direction rather than step size --
``huber_b8`` took 2582 steps at mean batch 3.06 for a consecutive-gradient cosine of
+0.0341, ``huber_b16`` took 924 steps at mean batch 8.55 for +0.0474 -- so buying
effective batch without spending steps is exactly the trade this path wants.

**The composition.** One optimizer step covers

    micro_batch x world_size x gradient_accumulation_units

scheduled units. ``world_size`` multiplies the accumulation group rather than dividing
it: a group holds ``W x K`` generation batches, of which each rank generates ``K``.

**Why there is no** :class:`~torch.nn.parallel.DistributedDataParallel` **wrapper here.**
:mod:`sae_bioemu.bioemu.shared_training` wraps its SAE in DDP because that model's
``forward`` is called once per iteration, on every rank, with the same shapes. None of
that holds for the steering adapter:

* the adapter is not called by this loop at all. It is invoked from inside BioEmu's
  denoiser through the intervention hook, several times per generated structure, so a
  DDP wrapper would have to be threaded through the sampler to be the thing that is
  called -- and DDP's reducer keys on the module's own ``forward``;
* the number of adapter forwards differs between ranks whenever a group's micro-batch
  count is not a multiple of ``W`` (the last group of a training cycle is routinely
  short). DDP's reducer expects every rank to fire the same number of buckets per
  iteration and hangs otherwise;
* DDP *averages* gradients. This loop weights each micro-batch by ``n_i / N``, its
  share of the whole group's scheduled units, so that no sequence is reweighted by
  which micro-batch it landed in. The correct cross-rank reduction is therefore a
  **sum**, and reproducing it under DDP would mean multiplying every loss by ``W`` to
  cancel DDP's own division.

So the reduction is done explicitly, with one ``all_reduce(SUM)`` over the adapter
gradients at the accumulation-group boundary. That boundary is already the atomic
resume unit (:meth:`GradientAccumulationPlan.offsets`), which makes one accumulation
group map onto exactly one collective. The process-group bootstrap, the context
dataclass, and the barrier are reused from :mod:`sae_bioemu.bioemu.shared_training`
rather than duplicated.

**Exactness.** With the ``n_i / N`` weighting kept against the *whole* group's unit
count on every rank, the summed gradient is bit-for-bit the gradient one device would
have accumulated, up to floating-point summation order. A run at ``W = 4, K = 1`` is
therefore the same optimization as a run at ``W = 1, K = 4``, and the checkpoint
identity says so: it records the realized accumulation grouping, which already folds in
``W x K``, and does *not* record ``world_size`` separately.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from sae_bioemu.bioemu.shared_training import (
    DistributedContext,
    distributed_barrier,
    initialize_distributed,
)

__all__ = [
    "DEFAULT_WORLD_SIZE",
    "WORLD_SIZE_ENVIRONMENT_VARIABLE",
    "DistributedContext",
    "agreed_first_failing_rank",
    "all_reduce_adapter_gradients",
    "barrier",
    "broadcast_from",
    "effective_batch_record",
    "gather_indexed",
    "initialize_steering_distributed",
    "rank_device",
    "rank_micro_batch_indices",
    "resolve_world_size",
    "single_process_context",
]

#: One process, one device: the pre-distributed path exactly.
DEFAULT_WORLD_SIZE = 1

#: Read only as a fallback when the workflow config does not declare a world size. The
#: configured value is authoritative because it is part of what the optimizer step *is*
#: (it multiplies the accumulation group), and an ``evaluate`` command rebuilding the
#: training identity on one GPU has to resolve the same number the training run did.
WORLD_SIZE_ENVIRONMENT_VARIABLE = "SAE_STEERING_WORLD_SIZE"


def resolve_world_size(value: object, *, name: str = "distributed_world_size") -> int:
    """Return the declared data-parallel world size, defaulting to one process.

    ``None`` and an absent key both mean one process, which is byte-identical to a run
    made before this existed. A present-but-invalid value is an error rather than a
    silent fallback: a misspelled world size that quietly trains at ``W = 1`` would
    produce a checkpoint whose accumulation grouping -- and therefore whose effective
    batch -- is a quarter of what the operator asked for, with nothing to read it back
    from.
    """
    if value is None:
        environment = os.environ.get(WORLD_SIZE_ENVIRONMENT_VARIABLE)
        if environment is None:
            return DEFAULT_WORLD_SIZE
        try:
            value = int(environment)
        except ValueError as error:
            raise ValueError(
                f"{WORLD_SIZE_ENVIRONMENT_VARIABLE} must be a positive integer"
            ) from error
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"{name} must be a positive integer; 1 runs one process on one device, which "
            "is the pre-distributed path"
        )
    return int(value)


def single_process_context(device: str | torch.device | None = None) -> DistributedContext:
    """Return the one-process context, without touching :mod:`torch.distributed`."""
    if device is None:
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    return DistributedContext(0, DEFAULT_WORLD_SIZE, 0, resolved)


def initialize_steering_distributed(
    *, declared_world_size: int, device: str | torch.device | None = None
) -> DistributedContext:
    """Join the launcher's process group and check it against the declared world size.

    ``declared_world_size`` comes from the workflow configuration, which is what the
    checkpoint identity is derived from. The launcher's ``WORLD_SIZE`` is what actually
    ran. A disagreement is fatal: training under a process group of two while the
    identity says four would write a checkpoint claiming an effective batch it never saw.
    """
    if declared_world_size == DEFAULT_WORLD_SIZE and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        return single_process_context(device)
    context = initialize_distributed()
    if context.world_size != declared_world_size:
        raise ValueError(
            f"launched world size {context.world_size} does not match the configured "
            f"distributed_world_size {declared_world_size}; the configured value multiplies "
            "the accumulation group and is bound into the checkpoint identity, so the two "
            "must agree"
        )
    return context


def rank_device(context: DistributedContext, configured: str | torch.device) -> torch.device:
    """Return the device this rank generates on.

    A configuration naming a bare ``cuda`` means "this job's GPU" at one process and
    "this rank's GPU" at several, so the local rank is pinned onto it. An explicitly
    indexed device is left alone, because an operator who wrote ``cuda:1`` meant it.
    """
    device = torch.device(configured)
    if device.type != "cuda" or device.index is not None or context.world_size == 1:
        return device
    return torch.device("cuda", context.local_rank)


def rank_micro_batch_indices(micro_batch_count: int, rank: int, world_size: int) -> tuple[int, ...]:
    """Return which of a group's micro-batches this rank generates.

    **Whole micro-batches, never split.** ``batching.py`` buckets scheduled units by
    residue count because a generation batch costs what its *padded maximum* length
    costs; splitting a micro-batch across ranks would re-pad both halves and throw that
    away. The partition therefore moves whole micro-batches, leaving each one exactly as
    the length-bucketing planner composed it.

    **Round robin, not contiguous chunks.** Within one bucketing window the planner
    emits micro-batches in ascending length order, so a contiguous split would hand rank
    0 every short batch and rank ``W - 1`` every long one. An optimizer step costs the
    slowest rank, so that split would spend the whole win on padding imbalance. Striding
    gives every rank one batch from each part of the window.

    A group whose micro-batch count is not a multiple of ``world_size`` -- routinely the
    last group of a training cycle -- leaves the high ranks with fewer batches, and
    possibly none. That is correct rather than padded: the ``n_i / N`` weighting counts
    units against the whole group, so an idle rank contributes a zero gradient and the
    step is still the mean over exactly the group's scheduled units.
    """
    if (
        isinstance(micro_batch_count, bool)
        or not isinstance(micro_batch_count, int)
        or micro_batch_count < 0
    ):
        raise ValueError("micro_batch_count must be a nonnegative integer")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
        raise ValueError("rank must lie in [0, world_size)")
    return tuple(index for index in range(micro_batch_count) if index % world_size == rank)


def effective_batch_record(
    *,
    maximum_micro_batch_units: int,
    world_size: int,
    gradient_accumulation_units: int,
    optimizer_step_units: int | None = None,
) -> dict[str, int]:
    """Return the three factors of the effective batch and their product.

    Recorded as factors rather than as one number because none of them describes the
    optimizer step on its own, and the run report is the only place a reader can recover
    which of the three an arm actually varied.

    ``optimizer_step_units`` is the MEASURED units of one optimizer step, from the resolved
    plan. Under the uniform optimizer step the product of the three factors is no longer the
    effective batch -- the step is a fixed number of scheduled units and the micro-batch count
    follows from the length heuristic inside it -- so when it is given it replaces the product
    rather than being checked against it. Omitting it keeps the record every run written
    before the uniform step had.
    """
    for name, value in (
        ("maximum_micro_batch_units", maximum_micro_batch_units),
        ("world_size", world_size),
        ("gradient_accumulation_units", gradient_accumulation_units),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if optimizer_step_units is not None and (
        isinstance(optimizer_step_units, bool)
        or not isinstance(optimizer_step_units, int)
        or optimizer_step_units < 1
    ):
        raise ValueError("optimizer_step_units must be a positive integer")
    return {
        "maximum_micro_batch_units": maximum_micro_batch_units,
        "world_size": world_size,
        "gradient_accumulation_units": gradient_accumulation_units,
        "maximum_effective_batch_units": (
            maximum_micro_batch_units * world_size * gradient_accumulation_units
            if optimizer_step_units is None
            else optimizer_step_units
        ),
        **({} if optimizer_step_units is None else {"optimizer_step_units": optimizer_step_units}),
    }


def barrier(context: DistributedContext) -> None:
    """Hold every rank until all have arrived; a no-op at one process."""
    if context.world_size > 1:
        distributed_barrier()


def all_reduce_adapter_gradients(adapter: nn.Module, context: DistributedContext) -> None:
    """Sum the adapter's accumulated gradients across ranks, in place.

    **Sum, not mean.** Each rank has already scaled its micro-batches by ``n_i / N``
    against the *whole* group's unit count, so the ranks hold disjoint parts of one mean
    and the reduction that reassembles it is addition. Dividing here would shrink every
    step by ``W`` and make a four-GPU run a different optimization from the single-GPU
    one it is supposed to reproduce.

    A rank that drew no micro-batches has ``grad is None`` on every parameter. Those are
    materialized as zeros first, so the collective has a tensor to reduce and so
    ``assert_adapter_only_gradients`` -- which runs after this and refuses a ``None``
    gradient -- sees the reduced gradient the optimizer is about to act on.
    """
    if context.world_size == 1:
        return
    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("adapter has no trainable parameters to reduce")
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    # One collective per parameter keeps the reduction independent of how the adapter is
    # laid out in memory, so every rank observes bit-identical results and the optimizer
    # states cannot drift apart across ranks.
    for parameter in parameters:
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def agreed_first_failing_rank(failed: bool, context: DistributedContext) -> int:
    """Return the lowest rank that failed this group, or ``world_size`` if none did.

    Every rank must reach the same verdict about an accumulation group, because the
    group is the atomic resume unit and the retry budget is durable: one rank retiring a
    group that the others committed would leave the persisted cursor describing a state
    no rank is in. Taking the minimum makes the answer deterministic and independent of
    which ranks failed, so ``maximum_unit_attempts`` is charged exactly once per group
    per collective attempt however many ranks raised.
    """
    local = context.rank if failed else context.world_size
    if context.world_size == 1:
        return local
    tensor = torch.tensor(local, dtype=torch.long, device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def broadcast_from(value: Any, source: int, context: DistributedContext) -> Any:
    """Return ``value`` as held by ``source`` on every rank."""
    if context.world_size == 1:
        return value
    payload = [value if context.rank == source else None]
    dist.broadcast_object_list(payload, src=source)
    return payload[0]


def gather_indexed(
    items: Sequence[tuple[int, Any]], context: DistributedContext
) -> tuple[Any, ...]:
    """Return every rank's ``(index, payload)`` pairs, ordered by index.

    The index is the micro-batch's position in its accumulation group, so reassembling
    by it reproduces the order a single process would have produced. That matters: the
    journal's training records are compared position by position against the schedule's
    committed prefix.
    """
    if context.world_size == 1:
        gathered: list[tuple[int, Any]] = list(items)
    else:
        buckets: list[Any] = [None] * context.world_size
        dist.all_gather_object(buckets, list(items))
        gathered = [pair for bucket in buckets for pair in bucket]
    indices = [index for index, _payload in gathered]
    if len(set(indices)) != len(indices):
        raise RuntimeError("micro-batch indices collided across ranks")
    return tuple(payload for _index, payload in sorted(gathered, key=lambda pair: pair[0]))


def concatenated_records(items: Iterable[Sequence[Any]]) -> tuple[Any, ...]:
    """Flatten per-micro-batch record tuples into one ordered tuple."""
    return tuple(record for group in items for record in group)
