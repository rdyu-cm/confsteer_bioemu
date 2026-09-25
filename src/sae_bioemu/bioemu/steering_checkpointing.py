"""Activation checkpointing for the BioEmu structure-module encoder, during steered generation.

WHY THIS EXISTS
---------------

Multi-site injection does not cost one extra retained forward; it costs one extra retained
forward *per remaining DPM iteration*. ``bioemu/denoiser.py`` wraps each ``get_score`` call in
``torch.set_grad_enabled(grad_is_enabled and (i in record_grad_steps))``, but the DPM state
updates that CONSUME the score (``u = ... + ... * score["pos"]``, ``pos_next = ... + ... *
score_u["pos"]``) run OUTSIDE that context. So the moment the first intervened score call
requires grad, ``pos`` requires grad for the rest of the trajectory, and every later retained
iteration builds a full autograd graph through TWO complete forwards of the eight-layer IPA
stack -- predictor and corrector -- rather than only through the injection point.

That graph is quadratic in sequence length. Measured on CPU with
``torch.autograd.graph.saved_tensors_hooks``, deduplicated by storage, against the pinned v1.1
architecture: a full-grad forward saves 34,794 bytes per residue PAIR plus a ~155 MiB
intercept, per structure. The O(L^2) items are all in ``bioemu/structure_module.py``: the
``torch.norm`` point-distance difference, the ``softmax`` attention output, and the
``einsum("bhij,bijhc->bihc", attn, v_pair)`` pair-value contraction, which saves both ``attn``
and the permuted ``v_pair``. Recomputing the layers instead of storing them is what makes
multi-site injection fit on a 93 GiB card.

THE INVARIANT: THE LAST LAYER IS NEVER CHECKPOINTED
---------------------------------------------------

**This is a correctness requirement, not a tuning choice.** The steering intervention is a
forward hook on ``layers[-1]`` (module path ``model_nn.st_module.encoder.layers.-1``; see
:class:`~sae_bioemu.bioemu.steering.BioEmuSteeringHook`). ``torch.utils.checkpoint`` calls the
wrapped function a second time during backward, and it calls the function it was handed --
here the layer's own ``forward``, not its ``__call__`` -- so a recomputed segment containing
the hooked layer would re-run the layer WITHOUT its hook. The recomputed activation would then
differ from the stored one: the adapter's residual would be absent from the recomputation, the
gradient that reaches the adapter would be the gradient of a different function, and nothing
would raise. The adapter would train against a silently wrong derivative.

Two independent guards therefore apply, both in :func:`checkpointed_encoder_layers`:

1. the final layer is excluded BY INDEX, so a refactor that reorders the stack cannot pull it
   in; and
2. any layer carrying a forward, forward-pre or backward hook is excluded by inspection, which
   covers a site configured at some other layer (``layer_m3`` and friends are configurable) and
   any capture hook a diagnostic left attached.

WHAT THIS DOES NOT CHANGE
-------------------------

Nothing scientific. ``use_reentrant=False`` records the autograd graph in the forward pass and
recomputes only what backward needs, and the recomputation is a rerun of the identical frozen
eval-mode layer on identical inputs, so forward values are bit-identical and gradients are
bit-exact -- verified in ``tests/test_bioemu_steering_encoder_checkpointing.py``, which asserts
``torch.equal`` on the adapter's parameter gradients rather than a tolerance.

Determinism is untouched. The IPA layers consume no RNG under steering: the score model is in
``eval()`` with frozen parameters (``freeze_for_steering``), which makes every ``nn.Dropout``
in the stack an identity that draws nothing. ``preserve_rng_state`` is left at its default
``True`` regardless, so even a stack that did consume RNG would recompute from the stashed
state rather than advancing the stream -- stashing and restoring consumes no draws, so
:class:`~sae_bioemu.bioemu.extractor.SeededPriorSampling`'s ambient-RNG assertion still holds.

WHEN IT IS A WIN, AND WHEN IT IS NOT
------------------------------------

Only when a retained score call runs with its INPUT already requiring grad -- which is the
multi-site case described above. With a single intervened site the layers before the hook see
constant inputs and build no graph at all, so there is nothing to recompute and checkpointing
merely adds the wrapper's retained inputs: measured at 0.03x, i.e. slightly worse. It is
enabled by default because the arms that need it are the multi-site ones, and
``SAE_STEERING_CHECKPOINT_ENCODER=0`` turns it off for an A/B or a debugging run. Which path
produced a checkpoint is recorded in the run report under
``generation_activation_checkpointing``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

#: The environment variable that turns encoder checkpointing off. Read ONCE, at import, so a
#: single generation cannot change policy halfway through a trajectory and leave a report
#: describing neither half.
ENCODER_CHECKPOINT_ENVIRONMENT_VARIABLE = "SAE_STEERING_CHECKPOINT_ENCODER"

#: On by default: the arms this was written for inject at several sites, which is exactly the
#: case that does not fit without it.
DEFAULT_ENCODER_CHECKPOINTING = True

#: Non-reentrant checkpointing is the only variant used here. It records the autograd graph in
#: the forward pass, supports arbitrary keyword arguments (the IPA layers take five), and stops
#: recomputation once backward has what it needs.
CHECKPOINT_IMPLEMENTATION = "torch.utils.checkpoint(use_reentrant=False)"

_FALSE_SPELLINGS = frozenset({"0", "false", "no", "off"})
_TRUE_SPELLINGS = frozenset({"1", "true", "yes", "on"})


def _read_environment_setting() -> bool | None:
    """Parse the off switch once, or return ``None`` when it is unset.

    An unparseable value fails closed rather than being treated as "on": a run launched with
    ``SAE_STEERING_CHECKPOINT_ENCODER=oof`` meant to say something, and silently ignoring it
    would put the opposite path in the report.
    """
    raw = os.environ.get(ENCODER_CHECKPOINT_ENVIRONMENT_VARIABLE)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in _FALSE_SPELLINGS:
        return False
    if value in _TRUE_SPELLINGS:
        return True
    raise ValueError(
        f"{ENCODER_CHECKPOINT_ENVIRONMENT_VARIABLE} must be one of "
        f"{sorted(_TRUE_SPELLINGS | _FALSE_SPELLINGS)}; got {raw!r}"
    )


#: The parsed environment setting, or ``None``. Module level and read at import: tests set this
#: directly, which is also how an A/B is driven in-process without re-reading the environment.
_ENVIRONMENT_SETTING: bool | None = _read_environment_setting()


@dataclass(frozen=True)
class EncoderCheckpointPolicy:
    """Whether steered generation recomputes the encoder layers, and where that was decided."""

    enabled: bool
    source: str

    def identity(self) -> dict[str, object]:
        """Return the auditable record of what this policy changes.

        Shaped like :meth:`~sae_bioemu.bioemu.precision.GenerationPrecision.identity`, and for
        the same reason: a reader of a run report has to be able to tell which code path
        produced the checkpoint beside it. Unlike precision, this one states that it changes
        no number -- because it does not, and because a memory knob that silently changed a
        gradient is exactly what the gradient-equality test exists to refuse.
        """
        return {
            "enabled": self.enabled,
            "source": self.source,
            "environment_variable": ENCODER_CHECKPOINT_ENVIRONMENT_VARIABLE,
            "implementation": CHECKPOINT_IMPLEMENTATION if self.enabled else None,
            "checkpoints_final_encoder_layer": False,
            "preserve_rng_state": True,
            # Forward values bit-identical, gradients bit-exact: this trades compute for
            # memory only. Stated in the report so the claim is auditable rather than assumed.
            "gradient_exact": True,
            "deterministic": True,
        }


def resolve_encoder_checkpoint_policy() -> EncoderCheckpointPolicy:
    """Return the active policy, defaulting to on when the environment says nothing."""
    setting = _ENVIRONMENT_SETTING
    if setting is None:
        return EncoderCheckpointPolicy(enabled=DEFAULT_ENCODER_CHECKPOINTING, source="default")
    return EncoderCheckpointPolicy(enabled=setting, source="environment")


@dataclass
class EncoderCheckpointRecord:
    """What a scope actually did, so a test can assert on it instead of on memory alone."""

    #: The encoder-layer indices whose ``forward`` the scope wrapped. The final index is never
    #: present; see the module docstring.
    wrapped_indices: tuple[int, ...] = ()
    #: Why nothing was wrapped, when nothing was. ``None`` when the scope was applied.
    skipped_reason: str | None = None
    #: Indices whose ``forward`` actually entered ``torch.utils.checkpoint``, in call order.
    #: A wrapper invoked under ``torch.no_grad()`` appends nothing, which is what proves the
    #: unsteered path is untouched.
    checkpointed_calls: list[int] = field(default_factory=list)


def _encoder_layers(score_model: nn.Module, module_path: str) -> nn.ModuleList | None:
    """Resolve the ``nn.ModuleList`` holding the encoder stack from the steered layer's path.

    Derived from the site's own module path (``...encoder.layers.-1``) rather than from a
    hard-coded ``st_module.encoder``: the path is what the hook attached to, so deriving the
    stack from it cannot disagree with the hook about which stack is being steered.
    """
    marker = ".layers."
    if marker not in module_path:
        return None
    prefix = module_path[: module_path.rindex(marker)]
    obj: object = score_model
    for part in prefix.split("."):
        if not part:
            continue
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    layers = getattr(obj, "layers", None)
    return layers if isinstance(layers, nn.ModuleList) else None


def _has_hooks(module: nn.Module) -> bool:
    """True when anything is attached that a recomputation would re-run or skip."""
    return bool(
        module._forward_hooks
        or module._forward_pre_hooks
        or module._backward_hooks
        or getattr(module, "_backward_pre_hooks", None)
    )


@contextmanager
def checkpointed_encoder_layers(
    score_model: nn.Module,
    module_path: str,
    policy: EncoderCheckpointPolicy | None = None,
) -> Iterator[EncoderCheckpointRecord]:
    """Recompute every encoder layer except the last for the duration of the scope.

    Scoped and restored in ``finally``, including on exception, because the wrapping is a
    mutation of a shared loaded model: a leaked wrapper would silently apply to the next
    generation, the unsteered draws, and any diagnostic in the same process -- none of which
    recorded that they ran under it. The restore is unconditional for the same reason
    :func:`~sae_bioemu.bioemu.precision.generation_precision_scope` restores its backend
    flags in ``finally``: D7's ``maximum_unit_attempts`` retries a failed generation in the
    same process.

    The wrapping shadows each layer's ``forward`` with an instance attribute rather than
    replacing the ``ModuleList`` entries. Replacing entries would re-prefix every parameter
    name under ``layers.<i>``, which is what a checkpoint's own state dict is keyed by;
    shadowing ``forward`` leaves the module graph, the parameter names and the hook registry
    exactly as they were.

    **The final layer is excluded by index, and any hooked layer by inspection.** That is the
    correctness invariant, not an optimization: see the module docstring.
    """
    resolved = resolve_encoder_checkpoint_policy() if policy is None else policy
    record = EncoderCheckpointRecord()
    if not resolved.enabled:
        record.skipped_reason = "policy_disabled"
        yield record
        return
    layers = _encoder_layers(score_model, module_path)
    if layers is None or len(layers) < 2:
        # Nothing to recompute, or a score model that is not the IPA stack (the CPU test
        # doubles, for instance). Reported rather than silently indistinguishable from an
        # applied scope, so a test can tell "did not apply" from "applied and saved nothing".
        record.skipped_reason = "no_multi_layer_encoder_stack"
        yield record
        return

    final_index = len(layers) - 1
    wrapped: list[tuple[int, nn.Module]] = []
    try:
        for index, layer in enumerate(layers):
            # THE INVARIANT. The hooked layer must run exactly once, with its hook, and its
            # activations must be the stored ones; see the module docstring.
            if index == final_index or _has_hooks(layer):
                continue
            original = layer.forward

            def wrapper(*args: object, _original=original, _index=index, **kwargs: object):
                # Checked at CALL time, not at scope entry: one trajectory interleaves
                # grad-enabled retained score calls with grad-disabled ones, and
                # ``torch.utils.checkpoint`` must not be entered for the latter.
                if not torch.is_grad_enabled():
                    return _original(*args, **kwargs)
                record.checkpointed_calls.append(_index)
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)

            layer.forward = wrapper  # type: ignore[method-assign]
            wrapped.append((index, layer))
        record.wrapped_indices = tuple(index for index, _ in wrapped)
        yield record
    finally:
        for _, layer in wrapped:
            # ``pop`` rather than ``del``: restoration must not itself raise and leave the
            # remaining layers wrapped.
            layer.__dict__.pop("forward", None)
