"""Per-optimizer-step gradient diagnostics for the ordinary steering training loop.

A pilot training run reached 591/1000 units with a flat loss, ``|error|/sigma`` sitting at
the analytic no-op expectation, and ``residual_fraction`` under 1% of the ``rho_max``
budget, while the adapter weights moved a long way. Nothing recorded the gradients, so
"the gradients are tiny" and "the gradients are large but inconsistent across steps and
cancel" were indistinguishable after the fact. This module records enough per step to
separate the two.

Three quantities per optimizer step:

* ``update_gradient_norm`` -- the global L2 norm of the adapter gradient *before* any
  clipping, named as on the AF3 path.
* ``update_clipped_gradient_norm`` -- the norm after clipping, or ``null`` when no
  clipping was applied. :func:`~sae_bioemu.steering.ordinary_workflow.run_ordinary_training`
  does not clip, so it records ``null`` rather than repeating the unclipped value, which
  would read as "clipping ran and changed nothing".
* ``applied_parameter_update_norm`` -- ``||theta_after - theta_before||``, the step the
  optimizer actually took. Under Adam this is close to ``lr * sqrt(P)`` almost regardless
  of the gradient magnitude, which is precisely why it cannot stand in for the gradient
  norm and why both are recorded.

and one consistency quantity:

* ``gradient_cosine_to_previous_step`` -- cosine similarity between this step's gradient
  and the previous step's, over the flattened adapter gradient. Consistently positive
  means the steps agree on a direction and the objective is simply insensitive; a running
  mean near zero means successive batches disagree and their updates cancel. It is
  measured on the *gradient*, not on the applied update, because Adam's per-coordinate
  normalization destroys gradient magnitude and would make a tiny consistent gradient and
  a large cancelling one look alike.

Cost is one clone of the flattened adapter gradient (order 10^5 floats for the production
adapter) plus two dot products per optimizer step, against a generation step measured in
tens of seconds.

**Everything here is per optimizer step, and under gradient accumulation that is the whole
accumulation group.** ``batch_size`` is the number of scheduled units the step covered --
generation batch size times accumulation units -- and the gradient measured is the
accumulated one, captured after the last micro-step's ``backward()`` and before ``step()``.
That is deliberate and load bearing: the cosine is the statistic accumulation exists to
move, so measuring it per micro-step would report exactly the draw-to-draw noise being
averaged away and would show no improvement even if accumulation worked perfectly. The
micro-batch composition of each step is recorded in the checkpoint identity, not here.

**These are diagnostics, exactly like the wall-clock timings next door.** They live in
their own ``<checkpoint>.gradients.json`` sidecar and never enter the adapter checkpoint,
the journal, the baseline sidecar, or any fingerprint. Two reasons, both load bearing.
Gradient norms are floating-point results that depend on the GPU, the cuBLAS reduction
blocking, and the batch composition -- D8 already measures 1-3 ULP coordinate differences
from batching alone -- so a single one inside the checkpoint would make
``checkpoint_file_sha256`` and the completion fingerprint differ between two runs that are
scientifically identical, which is the hazard ``progress.py`` documents for wall clock.
And they are per *optimizer step*, which spans ``B`` scheduled units, so they do not fit
the per-unit timing journal's closed record schema without either loosening that validator
or attributing a batch-level quantity to a single unit. A sibling sidecar keeps both
schemas closed and lets either one be lost independently.

Nothing here can fail a run: every probe method swallows its own exceptions, a malformed
sidecar is discarded and started over, and an unwritable one silently stops recording.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import torch

GRADIENT_SCHEMA_VERSION = 1
GRADIENT_SIDECAR_SUFFIX = ".gradients.json"

_STEP_KEYS = frozenset(
    {
        "key",
        "epoch",
        "cursor",
        "batch_size",
        "update_gradient_norm",
        "update_clipped_gradient_norm",
        "applied_parameter_update_norm",
        "gradient_cosine_to_previous_step",
    }
)
_TOTAL_KEYS = frozenset(
    {
        "step_count",
        "gradient_norm_sum",
        "applied_parameter_update_norm_sum",
        "cosine_count",
        "cosine_sum",
    }
)
_JOURNAL_KEYS = frozenset({"schema_version", "steps", "totals"})
_SUMMARY_KEYS = frozenset({"schema_version", "totals", "mean_gradient_norm", "mean_cosine"})
_SIDECAR_KEYS = frozenset({"schema_version", "checkpoint_path", "gradients"})


@dataclass(frozen=True)
class GradientStep:
    """One committed optimizer step's gradient diagnostics.

    ``batch_size`` counts the scheduled units the step covered, which is the generation
    batch size without accumulation and the whole accumulation group with it.
    """

    key: str
    epoch: int
    cursor: int
    batch_size: int
    update_gradient_norm: float
    update_clipped_gradient_norm: float | None
    applied_parameter_update_norm: float
    gradient_cosine_to_previous_step: float | None

    def as_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "batch_size": self.batch_size,
            "update_gradient_norm": self.update_gradient_norm,
            "update_clipped_gradient_norm": self.update_clipped_gradient_norm,
            "applied_parameter_update_norm": self.applied_parameter_update_norm,
            "gradient_cosine_to_previous_step": self.gradient_cosine_to_previous_step,
        }


def empty_gradient_diagnostics() -> dict[str, Any]:
    """Return the empty journal a fresh gradient sidecar starts from."""
    return {
        "schema_version": GRADIENT_SCHEMA_VERSION,
        "steps": [],
        "totals": {
            "step_count": 0,
            "gradient_norm_sum": 0.0,
            "applied_parameter_update_norm_sum": 0.0,
            "cosine_count": 0,
            "cosine_sum": 0.0,
        },
    }


def append_gradient_step(diagnostics: Mapping[str, Any], step: GradientStep) -> dict[str, Any]:
    """Return a new journal with one optimizer step appended."""
    if not isinstance(step, GradientStep):
        raise TypeError("step must be a GradientStep")
    validated = validate_gradient_diagnostics(diagnostics)
    record = _validated_step(step.as_record())
    validated["steps"].append(record)
    totals = validated["totals"]
    totals["step_count"] = int(totals["step_count"]) + 1
    totals["gradient_norm_sum"] = (
        float(totals["gradient_norm_sum"]) + record["update_gradient_norm"]
    )
    totals["applied_parameter_update_norm_sum"] = (
        float(totals["applied_parameter_update_norm_sum"]) + record["applied_parameter_update_norm"]
    )
    cosine = record["gradient_cosine_to_previous_step"]
    if cosine is not None:
        totals["cosine_count"] = int(totals["cosine_count"]) + 1
        totals["cosine_sum"] = float(totals["cosine_sum"]) + cosine
    return validated


def validate_gradient_diagnostics(value: object) -> dict[str, Any]:
    """Fail closed on a malformed gradient journal and return a deep copy."""
    if not isinstance(value, Mapping) or frozenset(value) != _JOURNAL_KEYS:
        raise ValueError("gradient diagnostics schema is invalid")
    diagnostics = copy.deepcopy(dict(value))
    if (
        type(diagnostics["schema_version"]) is not int
        or diagnostics["schema_version"] != GRADIENT_SCHEMA_VERSION
    ):
        raise ValueError("gradient diagnostics schema version is invalid")
    if not isinstance(diagnostics["steps"], list):
        raise ValueError("gradient diagnostics steps must be a list")
    steps = [_validated_step(row) for row in diagnostics["steps"]]
    totals = diagnostics["totals"]
    if not isinstance(totals, Mapping) or frozenset(totals) != _TOTAL_KEYS:
        raise ValueError("gradient diagnostics totals schema is invalid")
    for name in ("step_count", "cosine_count"):
        if not isinstance(totals[name], int) or isinstance(totals[name], bool) or totals[name] < 0:
            raise ValueError("gradient diagnostics totals counts are invalid")
    for name in ("gradient_norm_sum", "applied_parameter_update_norm_sum", "cosine_sum"):
        _finite(totals[name], f"gradient diagnostics {name}")
    if totals["step_count"] != len(steps):
        raise ValueError("gradient diagnostics step count does not match its steps")
    if totals["cosine_count"] != sum(
        1 for row in steps if row["gradient_cosine_to_previous_step"] is not None
    ):
        raise ValueError("gradient diagnostics cosine count does not match its steps")
    diagnostics["steps"] = steps
    diagnostics["totals"] = dict(totals)
    return diagnostics


def gradient_diagnostics_summary(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact aggregate an operator reads without opening every step."""
    validated = validate_gradient_diagnostics(diagnostics)
    totals = validated["totals"]
    steps = int(totals["step_count"])
    cosines = int(totals["cosine_count"])
    return {
        "schema_version": GRADIENT_SCHEMA_VERSION,
        "totals": dict(totals),
        "mean_gradient_norm": (float(totals["gradient_norm_sum"]) / steps if steps else None),
        # The discriminating number: near +1 the steps agree and the objective is merely
        # insensitive; near 0 successive batches disagree and their updates cancel.
        "mean_cosine": (float(totals["cosine_sum"]) / cosines if cosines else None),
    }


def validate_gradient_summary(value: object) -> dict[str, Any]:
    """Fail closed on a malformed gradient summary."""
    if not isinstance(value, Mapping) or frozenset(value) != _SUMMARY_KEYS:
        raise ValueError("gradient summary schema is invalid")
    if type(value["schema_version"]) is not int or value["schema_version"] != (
        GRADIENT_SCHEMA_VERSION
    ):
        raise ValueError("gradient summary schema version is invalid")
    totals = value["totals"]
    if not isinstance(totals, Mapping) or frozenset(totals) != _TOTAL_KEYS:
        raise ValueError("gradient summary totals schema is invalid")
    for name in ("mean_gradient_norm", "mean_cosine"):
        if value[name] is not None:
            _finite(value[name], f"gradient summary {name}")
    return {
        "schema_version": GRADIENT_SCHEMA_VERSION,
        "totals": dict(totals),
        "mean_gradient_norm": value["mean_gradient_norm"],
        "mean_cosine": value["mean_cosine"],
    }


def gradient_sidecar_path(checkpoint_path: str | Path) -> Path:
    """Return the diagnostic gradient sidecar that sits beside an adapter checkpoint."""
    path = Path(checkpoint_path)
    return path.with_suffix(path.suffix + GRADIENT_SIDECAR_SUFFIX)


def load_gradient_diagnostics(path: str | Path) -> dict[str, Any] | None:
    """Return the recorded diagnostics, or ``None`` when the sidecar is unusable."""
    sidecar = Path(path)
    try:
        payload = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return None
    if (
        not isinstance(payload, Mapping)
        or frozenset(payload) != _SIDECAR_KEYS
        or payload.get("schema_version") != GRADIENT_SCHEMA_VERSION
    ):
        return None
    try:
        return validate_gradient_diagnostics(payload["gradients"])
    except (TypeError, ValueError):
        return None


def load_checkpoint_gradient_diagnostics(checkpoint_path: str | Path) -> dict[str, Any] | None:
    """Return the diagnostics recorded beside ``checkpoint_path``, if any are usable."""
    return load_gradient_diagnostics(gradient_sidecar_path(checkpoint_path))


@dataclass(frozen=True)
class _PendingStep:
    """State captured between ``backward()`` and ``optimizer.step()``."""

    gradient: torch.Tensor
    parameters: torch.Tensor
    gradient_norm: float
    clipped_gradient_norm: float | None


class GradientProbe:
    """Append-only per-optimizer-step gradient journal beside an adapter checkpoint.

    Deliberately not part of the checkpoint: gradient norms are hardware- and
    batch-composition-dependent floats, so one of them inside the checkpoint would make
    two scientifically identical runs disagree on ``checkpoint_file_sha256`` and on the
    completion fingerprint derived from it.

    Every method is exception-safe by construction. The training loop's ``except`` clause
    rolls a unit back and burns an attempt, so a probe that raised would not merely lose a
    diagnostic -- it would fail scheduled units and could end a multi-hour run.
    """

    def __init__(self, path: str | Path, diagnostics: Mapping[str, Any] | None = None) -> None:
        self.path = Path(path)
        self._diagnostics = (
            empty_gradient_diagnostics()
            if diagnostics is None
            else validate_gradient_diagnostics(diagnostics)
        )
        self._available = True
        self._previous_gradient: torch.Tensor | None = None

    @classmethod
    def for_checkpoint(cls, checkpoint_path: str | Path) -> GradientProbe:
        """Load the sidecar for ``checkpoint_path``, starting fresh if it is unusable."""
        path = gradient_sidecar_path(checkpoint_path)
        return cls(path, load_gradient_diagnostics(path))

    @property
    def diagnostics(self) -> dict[str, Any]:
        return copy.deepcopy(self._diagnostics)

    @property
    def available(self) -> bool:
        """False once a measurement or write has failed; the run continues regardless."""
        return self._available

    def summary(self) -> dict[str, Any] | None:
        try:
            return gradient_diagnostics_summary(self._diagnostics)
        except (TypeError, ValueError):  # pragma: no cover - the journal is validated
            return None

    def before_step(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        clipped_gradient_norm: float | None = None,
    ) -> _PendingStep | None:
        """Capture the gradient and parameters between ``backward()`` and ``step()``.

        ``clipped_gradient_norm`` is for a caller that clips; the ordinary BioEmu path does
        not, and records ``null`` so a reader cannot mistake an unclipped norm for a
        clipped one.
        """
        if not self._available:
            return None
        try:
            values = tuple(parameters)
            gradient = _flattened(parameter.grad for parameter in values)
            if gradient is None:
                return None
            flat_parameters = _flattened(parameter.detach() for parameter in values)
            if flat_parameters is None:
                return None
            clipped = (
                None if clipped_gradient_norm is None else _finite(clipped_gradient_norm, "clipped")
            )
            return _PendingStep(
                gradient=gradient,
                parameters=flat_parameters,
                gradient_norm=float(gradient.norm().item()),
                clipped_gradient_norm=clipped,
            )
        except Exception:
            # Instrumentation must never abort a multi-hour scientific run.
            self._available = False
            return None

    def record(
        self,
        pending: _PendingStep | None,
        parameters: Iterable[torch.nn.Parameter],
        *,
        key: str,
        epoch: int,
        cursor: int,
        batch_size: int,
    ) -> GradientStep | None:
        """Complete one step's measurement and atomically republish the sidecar.

        Called *after* the scientific checkpoint write it describes, exactly as the timing
        sidecar is, so a crash between the two loses only a diagnostic row.
        """
        if pending is None or not self._available:
            return None
        try:
            updated = _flattened(parameter.detach() for parameter in parameters)
            if updated is None or updated.shape != pending.parameters.shape:
                return None
            step = GradientStep(
                key=key,
                epoch=epoch,
                cursor=cursor,
                batch_size=batch_size,
                update_gradient_norm=pending.gradient_norm,
                update_clipped_gradient_norm=pending.clipped_gradient_norm,
                applied_parameter_update_norm=float((updated - pending.parameters).norm().item()),
                gradient_cosine_to_previous_step=_cosine(self._previous_gradient, pending.gradient),
            )
            self._diagnostics = append_gradient_step(self._diagnostics, step)
            self._previous_gradient = pending.gradient
            _atomic_json(
                self.path,
                {
                    "schema_version": GRADIENT_SCHEMA_VERSION,
                    "checkpoint_path": str(self.path),
                    "gradients": self._diagnostics,
                },
            )
            return step
        except Exception:
            self._available = False
            return None


def _flattened(tensors: Iterable[torch.Tensor | None]) -> torch.Tensor | None:
    """Return one detached flat float vector, or ``None`` when it is unusable.

    ``None`` rather than an exception on a missing or nonfinite tensor: a diagnostic that
    cannot be measured is skipped, never escalated into a failed training unit.
    """
    parts: list[torch.Tensor] = []
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            return None
        parts.append(tensor.detach().reshape(-1).to(dtype=torch.float32))
    if not parts:
        return None
    flat = torch.cat(parts)
    if not bool(torch.isfinite(flat).all()):
        return None
    return flat


def _cosine(previous: torch.Tensor | None, current: torch.Tensor) -> float | None:
    """Return the cosine between consecutive gradients, or ``None`` when undefined."""
    if previous is None or previous.shape != current.shape:
        return None
    previous_norm = float(previous.norm().item())
    current_norm = float(current.norm().item())
    if not math.isfinite(previous_norm) or not math.isfinite(current_norm):
        return None
    if previous_norm <= 0.0 or current_norm <= 0.0:
        return None
    value = float(torch.dot(previous, current).item()) / (previous_norm * current_norm)
    if not math.isfinite(value):
        return None
    return max(-1.0, min(1.0, value))


def _validated_step(row: object) -> dict[str, Any]:
    if not isinstance(row, Mapping) or frozenset(row) != _STEP_KEYS:
        raise ValueError("gradient diagnostics step schema is invalid")
    if not isinstance(row["key"], str) or not row["key"]:
        raise ValueError("gradient diagnostics step key must be non-empty")
    for name in ("epoch", "cursor"):
        if not isinstance(row[name], int) or isinstance(row[name], bool) or row[name] < 0:
            raise ValueError(f"gradient diagnostics step {name} must be a nonnegative integer")
    if (
        not isinstance(row["batch_size"], int)
        or isinstance(row["batch_size"], bool)
        or row["batch_size"] < 1
    ):
        raise ValueError("gradient diagnostics step batch_size must be a positive integer")
    norms = {
        name: _nonnegative(row[name], f"gradient diagnostics step {name}")
        for name in ("update_gradient_norm", "applied_parameter_update_norm")
    }
    clipped = row["update_clipped_gradient_norm"]
    if clipped is not None:
        clipped = _nonnegative(clipped, "gradient diagnostics step update_clipped_gradient_norm")
    cosine = row["gradient_cosine_to_previous_step"]
    if cosine is not None:
        cosine = _finite(cosine, "gradient diagnostics step gradient_cosine_to_previous_step")
        if not -1.0 <= cosine <= 1.0:
            raise ValueError("gradient diagnostics step cosine must lie in [-1, 1]")
    return {
        "key": row["key"],
        "epoch": int(row["epoch"]),
        "cursor": int(row["cursor"]),
        "batch_size": int(row["batch_size"]),
        "update_gradient_norm": norms["update_gradient_norm"],
        "update_clipped_gradient_norm": clipped,
        "applied_parameter_update_norm": norms["applied_parameter_update_norm"],
        "gradient_cosine_to_previous_step": cosine,
    }


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _nonnegative(value: object, name: str) -> float:
    number = _finite(value, name)
    if number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


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


__all__ = [
    "GRADIENT_SCHEMA_VERSION",
    "GRADIENT_SIDECAR_SUFFIX",
    "GradientProbe",
    "GradientStep",
    "append_gradient_step",
    "empty_gradient_diagnostics",
    "gradient_diagnostics_summary",
    "gradient_sidecar_path",
    "load_checkpoint_gradient_diagnostics",
    "load_gradient_diagnostics",
    "validate_gradient_diagnostics",
    "validate_gradient_summary",
]
