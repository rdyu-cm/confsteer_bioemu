"""Rebuild a trained adapter from its checkpoint, without the config that trained it.

The training workflow constructs an adapter from a YAML config. That is the wrong dependency for
anyone *using* a released adapter: the config points at a specific machine's FASTA pools and
output directories, none of which exist for them, and none of which the adapter needs in order to
run.

Every checkpoint already records the architecture it was built with, under
``adapter_config.adapter.attributes``, so it is self-describing. This module reads that and
returns a working module. It is what makes a shipped ``.pt`` usable from a fresh clone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sae_bioemu.steering.adapter import BoundedResidualAdapter, TargetKind
from sae_bioemu.steering.adapter_config import DEFAULT_RESIDUAL_CAP


def read_adapter_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load the raw checkpoint payload."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "adapter_state" not in payload:
        raise ValueError(f"{path} is not a steering adapter checkpoint")
    return payload


def adapter_attributes(payload: dict[str, Any]) -> dict[str, Any]:
    """The recorded architecture block, or a clear failure."""
    try:
        return dict(payload["adapter_config"]["adapter"]["attributes"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            "checkpoint does not record adapter_config.adapter.attributes, so its architecture "
            "cannot be reconstructed; it predates self-describing checkpoints"
        ) from error


def load_adapter(path: str | Path) -> tuple[BoundedResidualAdapter, dict[str, Any]]:
    """Return ``(adapter, checkpoint_payload)`` with trained weights loaded, on CPU in eval mode.

    ``load_state_dict`` is strict: a shape or key mismatch between the recorded attributes and the
    stored tensors raises rather than silently producing a partly-initialised adapter that would
    generate plausible-looking nonsense.
    """
    payload = read_adapter_checkpoint(path)
    attributes = adapter_attributes(payload)
    config = payload.get("adapter_config", {})
    target_kind = config.get("target_kind", TargetKind.RELATIVE_SEQUENCE.value)
    adapter = BoundedResidualAdapter(
        int(attributes["width"]),
        str(attributes["property_name"]),
        str(attributes["model_family"]),
        hidden_width=int(attributes["hidden_width"]),
        hidden_layers=int(attributes["hidden_layers"]),
        rho_max=float(attributes["rho_max"]),
        eps=float(attributes["eps"]),
        level_conditioning=str(attributes.get("level_conditioning", "concat")),
        # Absent means "no site channel", which is every radius-of-gyration checkpoint and arm
        # C. Present, it changes the second hidden layer's fan-in, so getting it wrong is not a
        # silent degradation: the strict ``load_state_dict`` below fails on the shape.
        site_conditioning=attributes.get("site_conditioning"),
        # Absent means ``tanh``, the capped adapter every checkpoint on disk is. This one is
        # NOT caught by the strict ``load_state_dict`` below and cannot be: the cap changes no
        # tensor shape and no parameter name, so an uncapped checkpoint rebuilt without it
        # would load cleanly and then emit ``rho_max * tanh(r)`` where it was trained to emit
        # ``r`` -- a silently different intervention from a checkpoint that looked fine.
        residual_cap=str(attributes.get("residual_cap", DEFAULT_RESIDUAL_CAP)),
        target_kind=TargetKind(target_kind),
    )
    adapter.load_state_dict(payload["adapter_state"])
    adapter.eval()
    return adapter, payload
