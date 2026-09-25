"""Wall-clock timing sidecar and operator progress reporting for steering runs.

Timings are operator diagnostics, never scientific state, and they live in their own
``<checkpoint>.timings.json`` sidecar rather than inside the adapter checkpoint. This
repository's verification discipline rests on "identical inputs produce identical
fingerprints": path dependence is at least deterministic for a fixed path, but wall clock
is not deterministic at all, so a single timing byte inside the checkpoint would make
``checkpoint_file_sha256`` -- and the completion fingerprint that certifies a finished
training run -- differ between two byte-identical runs.

Every operation here degrades to "timings unavailable" instead of failing. Instrumentation
must never be load bearing: a missing or malformed sidecar cannot block resume, block
completion verification, or end a run.

``gradient_diagnostics.py`` holds a sibling sidecar under the same discipline, for
per-optimizer-step gradient norms. It is separate because those are per optimizer step,
which spans a whole batch, while a timing row is per scheduled unit.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from numbers import Real
from pathlib import Path
from typing import Any, TextIO

TIMING_SCHEMA_VERSION = 1
TIMING_PHASES = ("baseline", "train", "validation", "evaluation")
TIMING_OUTCOMES = ("committed", "failed")
TIMING_SIDECAR_SUFFIX = ".timings.json"

_RECORD_KEYS = frozenset({"phase", "key", "sequence_id", "seed", "elapsed_seconds", "outcome"})
_TIMING_KEYS = frozenset({"schema_version", "records", "totals"})
_SUMMARY_KEYS = frozenset({"schema_version", "totals"})
_TOTAL_KEYS = frozenset({"count", "seconds"})
_SIDECAR_KEYS = frozenset({"schema_version", "checkpoint_path", "timings"})


def empty_timings() -> dict[str, Any]:
    """Return the empty timing journal a fresh sidecar starts from."""
    return {
        "schema_version": TIMING_SCHEMA_VERSION,
        "records": [],
        "totals": {phase: {"count": 0, "seconds": 0.0} for phase in TIMING_PHASES},
    }


def append_timing(
    timings: Mapping[str, Any],
    *,
    phase: str,
    key: str,
    sequence_id: str,
    seed: int,
    elapsed_seconds: float,
    outcome: str = "committed",
) -> dict[str, Any]:
    """Return a new timing journal with one finished unit appended.

    The caller commits the returned value inside its own atomic checkpoint write, so the
    timing journal is exactly as durable and as transactional as the scientific journal.
    """
    validated = validate_timings(timings)
    if phase not in TIMING_PHASES:
        raise ValueError(f"timing phase must be one of {TIMING_PHASES}")
    if outcome not in TIMING_OUTCOMES:
        raise ValueError(f"timing outcome must be one of {TIMING_OUTCOMES}")
    if not isinstance(key, str) or not key:
        raise ValueError("timing key must be non-empty")
    if not isinstance(sequence_id, str) or not sequence_id:
        raise ValueError("timing sequence_id must be non-empty")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("timing seed must be a nonnegative integer")
    seconds = _nonnegative_seconds(elapsed_seconds)
    validated["records"].append(
        {
            "phase": phase,
            "key": key,
            "sequence_id": sequence_id,
            "seed": seed,
            "elapsed_seconds": seconds,
            "outcome": outcome,
        }
    )
    total = validated["totals"][phase]
    total["count"] = int(total["count"]) + 1
    total["seconds"] = float(total["seconds"]) + seconds
    return validated


def validate_timings(value: object) -> dict[str, Any]:
    """Fail closed on a malformed timing journal and return a deep copy."""
    if not isinstance(value, Mapping) or frozenset(value) != _TIMING_KEYS:
        raise ValueError("timing journal schema is invalid")
    timings = copy.deepcopy(dict(value))
    if (
        type(timings["schema_version"]) is not int
        or timings["schema_version"] != TIMING_SCHEMA_VERSION
    ):
        raise ValueError("timing journal schema version is invalid")
    if not isinstance(timings["records"], list):
        raise ValueError("timing journal records must be a list")
    for row in timings["records"]:
        if not isinstance(row, Mapping) or frozenset(row) != _RECORD_KEYS:
            raise ValueError("timing journal record schema is invalid")
        if (
            row["phase"] not in TIMING_PHASES
            or row["outcome"] not in TIMING_OUTCOMES
            or not isinstance(row["key"], str)
            or not row["key"]
            or not isinstance(row["sequence_id"], str)
            or not row["sequence_id"]
            or not isinstance(row["seed"], int)
            or isinstance(row["seed"], bool)
            or row["seed"] < 0
        ):
            raise ValueError("timing journal record identity is invalid")
        _nonnegative_seconds(row["elapsed_seconds"])
    totals = timings["totals"]
    if not isinstance(totals, Mapping) or set(totals) != set(TIMING_PHASES):
        raise ValueError("timing journal totals must cover every phase")
    for phase, total in totals.items():
        _validated_total(total)
        rows = [row for row in timings["records"] if row["phase"] == phase]
        if total["count"] != len(rows):
            raise ValueError("timing journal total count does not match its records")
    timings["records"] = [dict(row) for row in timings["records"]]
    timings["totals"] = {phase: dict(total) for phase, total in totals.items()}
    return timings


def exclude_timings(value: Mapping[str, Any], key: str = "timings") -> dict[str, Any]:
    """Return ``value`` without its timing key, for deterministic content fingerprints."""
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    return {name: item for name, item in value.items() if name != key}


def timing_totals(timings: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Return only the per-phase aggregate, for compact completion reports."""
    return validate_timings(timings)["totals"]


def timing_summary(timings: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact per-phase summary embedded in a completion report."""
    return {"schema_version": TIMING_SCHEMA_VERSION, "totals": timing_totals(timings)}


def validate_timing_summary(value: object) -> dict[str, Any]:
    """Fail closed on a malformed completion-report timing summary."""
    if not isinstance(value, Mapping) or frozenset(value) != _SUMMARY_KEYS:
        raise ValueError("timing summary schema is invalid")
    if type(value["schema_version"]) is not int or value["schema_version"] != TIMING_SCHEMA_VERSION:
        raise ValueError("timing summary schema version is invalid")
    totals = value["totals"]
    if not isinstance(totals, Mapping) or set(totals) != set(TIMING_PHASES):
        raise ValueError("timing summary totals must cover every phase")
    summary = {phase: dict(_validated_total(totals[phase])) for phase in TIMING_PHASES}
    return {"schema_version": TIMING_SCHEMA_VERSION, "totals": summary}


def _validated_total(total: object) -> Mapping[str, Any]:
    if not isinstance(total, Mapping) or frozenset(total) != _TOTAL_KEYS:
        raise ValueError("timing total schema is invalid")
    if (
        not isinstance(total["count"], int)
        or isinstance(total["count"], bool)
        or total["count"] < 0
    ):
        raise ValueError("timing total count is invalid")
    _nonnegative_seconds(total["seconds"])
    return total


def timing_sidecar_path(checkpoint_path: str | Path) -> Path:
    """Return the diagnostic timing sidecar that sits beside an adapter checkpoint."""
    path = Path(checkpoint_path)
    return path.with_suffix(path.suffix + TIMING_SIDECAR_SUFFIX)


class TimingSidecar:
    """Append-only wall-clock journal beside an adapter checkpoint.

    Deliberately not part of the checkpoint: ``checkpoint_file_sha256`` and the completion
    fingerprint derived from it certify a finished training run, and two byte-identical
    runs must produce the same digest. Nothing here can fail a run -- a malformed sidecar
    is discarded and started over, and an unwritable one silently stops recording.

    **Per-batch validation ``elapsed_seconds`` OVERLAP in wall clock at ``W > 1``.** A
    validation wave draws ``W`` batches CONCURRENTLY, one per rank; each rank times its own
    slot, and the primary journals one row per committed batch from the numbers the wave's
    gather brought back. So the validation rows of a sharded round describe concurrent spans,
    and SUMMING them overestimates the wall clock that round took, by up to a factor of ``W``.
    Each row is still the true cost of generating its own batch, which is what keeps per-batch
    costs comparable across rounds, arms and world sizes. This is a per-batch cost journal
    rather than a wall-clock ledger, and it is diagnostic either way.
    """

    def __init__(self, path: str | Path, timings: Mapping[str, Any] | None = None) -> None:
        self.path = Path(path)
        self._timings = empty_timings() if timings is None else validate_timings(timings)
        self._available = True

    @classmethod
    def for_checkpoint(cls, checkpoint_path: str | Path) -> TimingSidecar:
        """Load the sidecar for ``checkpoint_path``, starting fresh if it is unusable."""
        path = timing_sidecar_path(checkpoint_path)
        return cls(path, load_timings(path))

    @property
    def timings(self) -> dict[str, Any]:
        return copy.deepcopy(self._timings)

    @property
    def available(self) -> bool:
        """False once a write has failed; recording has stopped but the run continues."""
        return self._available

    def summary(self) -> dict[str, Any]:
        return timing_summary(self._timings)

    def record(
        self,
        *,
        phase: str,
        key: str,
        sequence_id: str,
        seed: int,
        elapsed_seconds: float,
        outcome: str = "committed",
    ) -> None:
        """Append one measurement and atomically republish, degrading on any failure."""
        if not self._available:
            return
        try:
            self._timings = append_timing(
                self._timings,
                phase=phase,
                key=key,
                sequence_id=sequence_id,
                seed=seed,
                elapsed_seconds=elapsed_seconds,
                outcome=outcome,
            )
            _atomic_json(
                self.path,
                {
                    "schema_version": TIMING_SCHEMA_VERSION,
                    "checkpoint_path": str(self.path),
                    "timings": self._timings,
                },
            )
        except (OSError, TypeError, ValueError):
            self._available = False


def load_timings(path: str | Path) -> dict[str, Any] | None:
    """Return the recorded timings, or ``None`` when the sidecar is absent or unusable."""
    sidecar = Path(path)
    try:
        payload = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return None
    if (
        not isinstance(payload, Mapping)
        or frozenset(payload) != _SIDECAR_KEYS
        or payload.get("schema_version") != TIMING_SCHEMA_VERSION
    ):
        return None
    try:
        return validate_timings(payload["timings"])
    except ValueError:
        return None


def load_checkpoint_timings(checkpoint_path: str | Path) -> dict[str, Any] | None:
    """Return the timings recorded beside ``checkpoint_path``, if any are usable."""
    return load_timings(timing_sidecar_path(checkpoint_path))


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


def format_duration(seconds: float) -> str:
    """Render a nonnegative duration as ``H:MM:SS`` for the operator log."""
    total = int(round(_nonnegative_seconds(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}"


class ProgressReporter:
    """Emit a periodic human-readable progress line for a long steering job.

    The reporter owns no scientific state: it only formats counts, means, and an ETA that
    an operator can tail in a Slurm log.
    """

    def __init__(
        self,
        total_units: int,
        *,
        label: str = "steering",
        stream: TextIO | None = None,
        interval_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(total_units, int) or isinstance(total_units, bool) or total_units < 0:
            raise ValueError("total_units must be a nonnegative integer")
        if not isinstance(label, str) or not label:
            raise ValueError("label must be non-empty")
        if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, Real):
            raise ValueError("interval_seconds must be a finite nonnegative number")
        _nonnegative_seconds(interval_seconds)
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.total_units = total_units
        self.label = label
        self.interval_seconds = float(interval_seconds)
        self._stream = sys.stdout if stream is None else stream
        self._clock = clock
        self._started = float(clock())
        self._last_emitted: float | None = None
        self._counts = dict.fromkeys(TIMING_PHASES, 0)
        self._seconds = dict.fromkeys(TIMING_PHASES, 0.0)
        self._skipped_units = 0

    def record_skipped_units(self, units: int) -> int:
        """Count scheduled units the run stepped over, and return the running total.

        Deliberately NOT routed through :meth:`observe`. A skipped unit consumed generation time
        and produced no training record, so feeding it to the accumulators would count it in the
        seconds-per-unit mean and in the ETA as though it had trained -- which is the one thing a
        reader of this line must not conclude. It is a separate counter, shown only when nonzero,
        so a healthy run's line is byte-identical to the line it printed before skips existed.
        """
        if not isinstance(units, int) or isinstance(units, bool) or units < 1:
            raise ValueError("units must be a positive integer")
        self._skipped_units += units
        return self._skipped_units

    @property
    def skipped_units(self) -> int:
        return self._skipped_units

    def announce(self, message: str) -> str:
        """Write one unconditional labelled line."""
        line = f"[{self.label}] {message}"
        self._write(line)
        return line

    def observe(
        self,
        *,
        phase: str,
        elapsed_seconds: float,
        completed_units: int,
        epoch: int,
        units: int = 1,
        force: bool = False,
    ) -> str | None:
        """Accumulate one finished observation and emit a progress line when it is due.

        ``units`` is how many scheduled units ``elapsed_seconds`` covers. One batched
        generation commits ``B`` units in one measurement, so counting the observation
        rather than its units made ``mean`` a seconds-per-*batch* figure labelled
        ``s/unit`` and multiplied by the remaining *units*: a real 4.8 h run displayed an
        ETA of 38 h (D15). The mean is now total train seconds over total train units, so
        ``mean * remaining_units`` is dimensionally correct at every batch size.
        """
        if phase not in TIMING_PHASES:
            raise ValueError(f"timing phase must be one of {TIMING_PHASES}")
        if (
            not isinstance(completed_units, int)
            or isinstance(completed_units, bool)
            or completed_units < 0
            or not isinstance(epoch, int)
            or isinstance(epoch, bool)
            or epoch < 0
        ):
            raise ValueError("completed_units and epoch must be nonnegative integers")
        if not isinstance(units, int) or isinstance(units, bool) or units < 1:
            raise ValueError("units must be a positive integer")
        seconds = _nonnegative_seconds(elapsed_seconds)
        self._counts[phase] += units
        self._seconds[phase] += seconds
        now = float(self._clock())
        due = (
            force or self._last_emitted is None or now - self._last_emitted >= self.interval_seconds
        )
        if not due:
            return None
        self._last_emitted = now
        line = self._line(phase, seconds, units, completed_units, epoch, now)
        self._write(line)
        return line

    def _line(
        self, phase: str, seconds: float, units: int, completed_units: int, epoch: int, now: float
    ) -> str:
        train_units = self._counts["train"]
        mean = self._seconds["train"] / train_units if train_units else None
        remaining = max(self.total_units - completed_units, 0)
        percent = 100.0 * completed_units / self.total_units if self.total_units else 0.0
        # A batched observation names its own unit count so ``last`` is never mistaken
        # for a per-unit duration.
        last = f"{seconds:.1f}s" if units == 1 else f"{seconds:.1f}s/{units}u"
        parts = [
            f"{completed_units}/{self.total_units} train units ({percent:.1f}%)",
            f"epoch {epoch}",
            f"last={phase} {last}",
            f"mean={'n/a' if mean is None else f'{mean:.1f}s/unit'}",
            f"elapsed={format_duration(now - self._started)}",
            f"eta={'n/a' if mean is None else format_duration(mean * remaining)}",
        ]
        # Shown only when a unit was actually stepped over. A run that skipped nothing prints the
        # line it always printed, and a run that skipped something says so on EVERY subsequent
        # line rather than only in the one warning that scrolled past hours ago.
        if self._skipped_units:
            parts.append(f"skipped={self._skipped_units}")
        return f"[{self.label}] {' | '.join(parts)}"

    def _write(self, line: str) -> None:
        # Instrumentation must never abort a multi-hour scientific run: a closed or full
        # log stream is reported by Slurm, not by killing training mid-unit.
        if self._stream is None:
            return
        try:
            self._stream.write(f"{line}\n")
            flush = getattr(self._stream, "flush", None)
            if callable(flush):
                flush()
        except (OSError, ValueError):
            self._stream = None


def _nonnegative_seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("elapsed seconds must be finite")
    seconds = float(value)
    if seconds < 0:
        raise ValueError("elapsed seconds must be nonnegative")
    return seconds
