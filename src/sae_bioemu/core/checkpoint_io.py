from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, BinaryIO

import torch

from sae_bioemu.core.config import SAEConfig

SAE_CHECKPOINT_SCHEMA_VERSION = 1
"""Version of the primitive SAE configuration stored in new checkpoints."""

_SCHEMA_VERSION_KEY = "checkpoint_schema_version"
_LEGACY_SAE_CONFIG_GLOBAL = ("sae_bioemu.config", "SAEConfig")
_SAE_CONFIG_FIELD_NAMES = frozenset(field.name for field in fields(SAEConfig))


class _LegacySAEConfigUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == _LEGACY_SAE_CONFIG_GLOBAL:
            return SAEConfig
        return super().find_class(module, name)


class _LegacySAEConfigPickleModule:
    __name__ = pickle.__name__
    Unpickler = _LegacySAEConfigUnpickler
    load = staticmethod(pickle.load)


def prepare_sae_checkpoint_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Encode one SAE checkpoint with a versioned primitive config dictionary."""

    config = payload.get("config")
    if not isinstance(config, SAEConfig):
        raise TypeError("SAE checkpoint config must be an SAEConfig")
    return {
        **payload,
        _SCHEMA_VERSION_KEY: SAE_CHECKPOINT_SCHEMA_VERSION,
        "config": asdict(config),
    }


def load_sae_checkpoint(
    source: str | Path | BinaryIO,
    *,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """Load a trusted SAE checkpoint and normalize its validated configuration."""

    payload = torch.load(
        source,
        map_location=map_location,
        pickle_module=_LegacySAEConfigPickleModule,
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("SAE checkpoint payload must be a mapping")

    config = payload.get("config")
    schema_version = payload.get(_SCHEMA_VERSION_KEY)
    if isinstance(config, SAEConfig):
        if schema_version is not None:
            raise ValueError("versioned SAE checkpoint config must be a primitive dictionary")
        normalized_config = SAEConfig(**asdict(config))
    else:
        if type(schema_version) is not int or schema_version != SAE_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "SAE checkpoint schema version must be the integer "
                f"{SAE_CHECKPOINT_SCHEMA_VERSION}"
            )
        if type(config) is not dict:
            raise TypeError("versioned SAE checkpoint config must be a dictionary")
        config_fields = frozenset(config)
        if config_fields != _SAE_CONFIG_FIELD_NAMES:
            missing = sorted(_SAE_CONFIG_FIELD_NAMES - config_fields)
            extra = sorted(config_fields - _SAE_CONFIG_FIELD_NAMES, key=repr)
            raise ValueError(
                "versioned SAE checkpoint config fields must exactly match SAEConfig "
                f"(missing={missing!r}, extra={extra!r})"
            )
        normalized_config = SAEConfig(**config)

    return {**payload, "config": normalized_config}
