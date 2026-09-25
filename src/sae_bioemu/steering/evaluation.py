"""Fail-closed paired held-out evaluation for property steering."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from .schedule import SteeringUnit

PairKey = tuple[str, int, float]
PairedGenerationCallback = Callable[[SteeringUnit, bool], Mapping[str, object]]
TargetResolver = Callable[[SteeringUnit], float]

#: ``success_tolerance`` measured in the property's own units (Angstrom for Rg).
ABSOLUTE_TOLERANCE_UNITS = "absolute_property_units"
#: ``success_tolerance`` measured in multiples of each sequence's own sigma (D4b).
SIGMA_TOLERANCE_UNITS = "sequence_sigma"
_TOLERANCE_UNITS = frozenset({ABSOLUTE_TOLERANCE_UNITS, SIGMA_TOLERANCE_UNITS})


@dataclass(frozen=True)
class SteeringEvaluationRow:
    """One successful adapter-off/on property comparison.

    ``success_tolerance`` is expressed in multiples of ``property_scale``. With the
    default ``property_scale = 1.0`` it keeps its historical meaning: a threshold in
    the property's own absolute units. The relative-level BioEmu Rg run supplies each
    sequence's own sigma there, so success becomes ``|on - t| / sigma_seq <=
    tolerance`` and the threshold is comparable across proteins (D4b).

    ``identity_cluster`` is the 30%-identity sequence cluster D-24 makes the independent
    statistical unit. It is the label the bootstrap resamples, carried on the row so
    ``predictions.csv`` states the grouping a downstream script would otherwise have to be
    told out of band -- which is how every published Rg dose-slope interval came to be
    sequence-grouped while the cryptic side was cluster-grouped. Rows do not have to carry
    it: ``None`` means no cluster label was available for this run, and
    :func:`write_steering_report` fills it in from ``family_clusters`` when the report is
    published, so the column always agrees with the unit the metrics were resampled on.
    """

    sequence_id: str
    seed: int
    target: float
    adapter_off_value: float
    adapter_on_value: float
    off_absolute_error: float
    on_absolute_error: float
    signed_movement_toward_target: float
    success: bool
    residual_fraction: float
    direction_success: bool
    success_tolerance: float
    property_scale: float = 1.0
    identity_cluster: str | None = None

    def __post_init__(self) -> None:
        _validate_key((self.sequence_id, self.seed, self.target))
        if self.identity_cluster is not None and (
            not isinstance(self.identity_cluster, str) or not self.identity_cluster
        ):
            raise ValueError("identity_cluster must be a non-empty string or None")
        for name in (
            "adapter_off_value",
            "adapter_on_value",
            "off_absolute_error",
            "on_absolute_error",
            "signed_movement_toward_target",
        ):
            _finite(getattr(self, name), name)
        if not isinstance(self.success, bool):
            raise ValueError("success must be a boolean")
        tolerance = _nonnegative_finite(self.success_tolerance, "success_tolerance")
        scale = _positive_finite(self.property_scale, "property_scale")
        _residual_fraction(self.residual_fraction)
        if not isinstance(self.direction_success, bool):
            raise ValueError("direction_success must be a boolean")
        off_error = abs(self.adapter_off_value - self.target)
        on_error = abs(self.adapter_on_value - self.target)
        if self.off_absolute_error < 0 or not _approximately_equal(
            self.off_absolute_error, off_error
        ):
            raise ValueError("off_absolute_error must equal abs(adapter_off_value - target)")
        if self.on_absolute_error < 0 or not _approximately_equal(self.on_absolute_error, on_error):
            raise ValueError("on_absolute_error must equal abs(adapter_on_value - target)")
        if not _approximately_equal(
            self.signed_movement_toward_target, self.off_absolute_error - self.on_absolute_error
        ):
            raise ValueError(
                "signed_movement_toward_target must equal off_absolute_error - on_absolute_error"
            )
        if self.direction_success != _direction_success(
            self.adapter_off_value, self.adapter_on_value, self.target
        ):
            raise ValueError("direction_success must equal the requested-movement sign agreement")
        if self.success != (self.on_absolute_error / scale <= tolerance):
            raise ValueError(
                "success must equal (on_absolute_error / property_scale <= success_tolerance)"
            )

    @property
    def key(self) -> PairKey:
        return (self.sequence_id, self.seed, float(self.target))


@dataclass(frozen=True)
class LocalQEvaluationRow:
    """One residue-level paired Local-Q steering comparison."""

    sequence_id: str
    seed: int
    control_dose: float
    signed_response: float
    target_window_change: float
    nontarget_spillover: float
    requested_sign_success: float | None
    target_mae: float
    residual_fraction: float
    targeted_residue_count: int
    valid_residue_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id:
            raise ValueError("Local-Q sequence_id must be non-empty")
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("Local-Q seed must be a uint32 integer")
        for name in (
            "control_dose",
            "signed_response",
            "target_window_change",
            "nontarget_spillover",
            "target_mae",
        ):
            _finite(getattr(self, name), name)
        if self.requested_sign_success is not None:
            value = _finite(self.requested_sign_success, "requested_sign_success")
            if not 0 <= value <= 1:
                raise ValueError("requested_sign_success must be in [0,1]")
        _residual_fraction(self.residual_fraction)
        if (
            not isinstance(self.targeted_residue_count, int)
            or isinstance(self.targeted_residue_count, bool)
            or not isinstance(self.valid_residue_count, int)
            or isinstance(self.valid_residue_count, bool)
            or not 0 <= self.targeted_residue_count <= self.valid_residue_count
        ):
            raise ValueError("Local-Q residue counts are invalid")


@dataclass(frozen=True)
class SteeringFailure:
    """A property-evaluation failure retained in the reported denominator."""

    sequence_id: str
    seed: int
    target: float
    phase: str
    exception_class: str
    message: str

    def __post_init__(self) -> None:
        _validate_key((self.sequence_id, self.seed, self.target))
        for name in ("phase", "exception_class", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

    @property
    def key(self) -> PairKey:
        return (self.sequence_id, self.seed, float(self.target))


@dataclass(frozen=True)
class SteeringMetrics:
    """Sequence-grouped summaries, with failures retained as attempted units."""

    independent_group: str
    sequence_count: int
    success_count: int
    failure_count: int
    success_denominator: int
    sequence_mean_success_rate: float
    bootstrap_success_rate: float | None
    monotonicity_by_sequence: dict[str, float | None]
    monotonicity_status_by_sequence: dict[str, str]
    mean_monotonicity: float | None
    sequence_mean_off_absolute_error: float
    sequence_mean_on_absolute_error: float
    sequence_mean_improvement: float
    sequence_mean_direction_success_rate: float
    sequence_mean_tolerance_success_rate: float
    sequence_mean_residual_fraction: float
    validity_numerator: int
    validity_denominator: int
    coverage_numerator: int
    coverage_denominator: int
    target_achieved_spearman: float | None
    success_tolerance: float | None
    sequence_mean_paired_improvement_rate: float
    bootstrap_unit: str
    bootstrap_group_count: int
    bootstrap_intervals: dict[str, tuple[float | None, float | None]]
    success_tolerance_units: str = ABSOLUTE_TOLERANCE_UNITS

    @property
    def metrics(self) -> dict[str, float | None]:
        """Concise scientific metric names for reports and interval matching."""

        return {
            "mae": self.sequence_mean_on_absolute_error,
            "paired_improvement": self.sequence_mean_paired_improvement_rate,
            "movement": self.sequence_mean_improvement,
            "signed_movement": self.sequence_mean_improvement,
            "direction_success": self.sequence_mean_direction_success_rate,
            "tolerance_success": self.sequence_mean_tolerance_success_rate,
            "target_achieved_spearman": self.target_achieved_spearman,
            "monotonicity": self.mean_monotonicity,
            "residual_fraction": self.sequence_mean_residual_fraction,
            "validity": (
                self.validity_numerator / self.validity_denominator
                if self.validity_denominator
                else 0.0
            ),
            "coverage": (
                self.coverage_numerator / self.coverage_denominator
                if self.coverage_denominator
                else 0.0
            ),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**asdict(self), **self.metrics}


@dataclass(frozen=True)
class PairedEvaluationResult:
    """Complete outcome inventory for fixed held-out steering units."""

    attempted_units: tuple[SteeringUnit, ...]
    rows: tuple[SteeringEvaluationRow, ...]
    failures: tuple[SteeringFailure, ...]


def run_paired_evaluation(
    units: Iterable[SteeringUnit],
    generate: PairedGenerationCallback,
    property_fn: Callable[[object], float],
    *,
    target_resolver: TargetResolver,
    success_tolerance: float = 0.0,
) -> PairedEvaluationResult:
    """Generate adapter-off/on pairs with identical scheduled stochastic inputs.

    ``generate`` receives the exact same :class:`SteeringUnit` for both calls; its
    boolean argument selects adapter-on. Each attempted unit yields exactly one
    successful row or one retained failure.
    """
    tolerance = _nonnegative_finite(success_tolerance, "success_tolerance")
    if not callable(generate) or not callable(property_fn) or not callable(target_resolver):
        raise ValueError("generate, property_fn, and target_resolver must be callable")
    attempted = _evaluation_units(units)
    resolved = tuple(
        (unit, _finite(target_resolver(unit), "resolved target")) for unit in attempted
    )
    resolved_keys = tuple((unit.sequence_id, unit.seed, target) for unit, target in resolved)
    if len(set(resolved_keys)) != len(resolved_keys):
        raise ValueError("resolved absolute targets must not create duplicate paired keys")
    rows: list[SteeringEvaluationRow] = []
    failures: list[SteeringFailure] = []
    for unit, target in resolved:
        key = (unit.sequence_id, unit.seed, target)
        try:
            off = _generated_record(unit, target, generate(unit, False))
        except Exception as error:
            failures.append(_failure(key, "adapter_off_generation", error))
            continue
        try:
            on = _generated_record(unit, target, generate(unit, True))
        except Exception as error:
            failures.append(_failure(key, "adapter_on_generation", error))
            continue
        pair_rows, pair_failures = evaluate_paired_steering(
            (off,), (on,), property_fn, success_tolerance=tolerance
        )
        rows.extend(pair_rows)
        failures.extend(pair_failures)
    valid_rows, valid_failures = _validate_outcomes(tuple(rows), tuple(failures))
    observed = {row.key for row in valid_rows} | {failure.key for failure in valid_failures}
    expected = set(resolved_keys)
    if observed != expected:
        raise AssertionError("paired evaluation did not retain every attempted unit")
    return PairedEvaluationResult(attempted, valid_rows, valid_failures)


def evaluate_paired_steering(
    adapter_off: Iterable[Mapping[str, object]],
    adapter_on: Iterable[Mapping[str, object]],
    property_fn: Callable[[object], float],
    *,
    success_tolerance: float = 0.0,
    property_scale: float = 1.0,
) -> tuple[tuple[SteeringEvaluationRow, ...], tuple[SteeringFailure, ...]]:
    """Evaluate exact adapter pairs using the authoritative structure callback.

    Positive signed movement means the adapter reduced absolute error to the requested
    target.  Thus crossing the target is penalized by its final absolute error.

    ``success_tolerance`` is measured in multiples of ``property_scale``. The default
    scale of one keeps the threshold in the property's own absolute units; passing the
    sequence's own sigma makes it a sigma threshold (D4b). ``property_scale`` never
    changes the reported errors or movements, only the success decision.
    """
    tolerance = _nonnegative_finite(success_tolerance, "success_tolerance")
    scale = _positive_finite(property_scale, "property_scale")
    if not callable(property_fn):
        raise ValueError("property_fn must be callable")
    off = _inventory(adapter_off, "adapter_off")
    on = _inventory(adapter_on, "adapter_on")
    if set(off) != set(on):
        raise ValueError("adapter-off/on paired inventories do not match")

    rows: list[SteeringEvaluationRow] = []
    failures: list[SteeringFailure] = []
    for key in sorted(off):
        off_record, on_record = off[key], on[key]
        try:
            off_value = _property_value(property_fn, off_record)
        except Exception as error:  # callback failures are output data, not discarded rows.
            failures.append(_failure(key, "adapter_off_property", error))
            continue
        try:
            on_value = _property_value(property_fn, on_record)
        except Exception as error:
            failures.append(_failure(key, "adapter_on_property", error))
            continue
        residual = _residual_fraction(_field(on_record, "residual_fraction"))
        off_error, on_error = abs(off_value - key[2]), abs(on_value - key[2])
        rows.append(
            SteeringEvaluationRow(
                *key,
                off_value,
                on_value,
                off_error,
                on_error,
                off_error - on_error,
                on_error / scale <= tolerance,
                residual,
                _direction_success(off_value, on_value, key[2]),
                tolerance,
                scale,
            )
        )
    _validate_outcomes(rows, failures)
    return tuple(rows), tuple(failures)


def evaluate_local_q_pair(
    *,
    sequence_id: str,
    seed: int,
    control: np.ndarray,
    baseline: np.ndarray,
    steered: np.ndarray,
    valid_mask: np.ndarray,
    residual_fraction: float,
) -> LocalQEvaluationRow:
    """Calculate target-window and spillover metrics for one paired Local-Q unit."""
    direction = np.asarray(control)
    off = np.asarray(baseline)
    on = np.asarray(steered)
    valid = np.asarray(valid_mask)
    if (
        direction.dtype != np.float32
        or off.dtype != np.float32
        or on.dtype != np.float32
        or valid.dtype != np.bool_
        or direction.ndim != 1
        or direction.shape != off.shape
        or direction.shape != on.shape
        or direction.shape != valid.shape
        or not valid.any()
        or not np.isfinite(direction).all()
        or not np.isfinite(off).all()
        or not np.isfinite(on).all()
        or np.any((direction < -1) | (direction > 1))
        or np.any((off[valid] < 0) | (off[valid] > 1))
        or np.any((on[valid] < 0) | (on[valid] > 1))
        or np.any(direction[~valid] != 0)
    ):
        raise ValueError("Local-Q paired arrays are invalid")
    targeted = valid & (direction != 0)
    nontarget = valid & ~targeted
    delta = on - off
    target = np.where(direction >= 0, off + direction * (1 - off), off + direction * off)
    targeted_count = int(targeted.sum())
    signed_response = float(np.mean(direction[valid] * delta[valid]))
    target_change = float(np.mean(delta[targeted])) if targeted_count else 0.0
    spillover = float(np.mean(np.abs(delta[nontarget]))) if nontarget.any() else 0.0
    sign_success = (
        float(np.mean(direction[targeted] * delta[targeted] > 0)) if targeted_count else None
    )
    return LocalQEvaluationRow(
        sequence_id=sequence_id,
        seed=seed,
        control_dose=float(np.mean(np.abs(direction[targeted]))) if targeted_count else 0.0,
        signed_response=signed_response,
        target_window_change=target_change,
        nontarget_spillover=spillover,
        requested_sign_success=sign_success,
        target_mae=float(np.mean(np.abs(on[valid] - target[valid]))),
        residual_fraction=float(residual_fraction),
        targeted_residue_count=targeted_count,
        valid_residue_count=int(valid.sum()),
    )


def grouped_local_q_metrics(
    rows: Iterable[LocalQEvaluationRow],
    *,
    bootstrap_replicates: int = 0,
    seed: int = 0,
) -> dict[str, object]:
    """Aggregate residue-level Local-Q rows without treating residues as proteins."""
    values = tuple(rows)
    if not values:
        raise ValueError("Local-Q metrics require at least one successful row")
    if any(not isinstance(row, LocalQEvaluationRow) for row in values):
        raise TypeError("Local-Q metrics require LocalQEvaluationRow values")
    if (
        not isinstance(bootstrap_replicates, int)
        or isinstance(bootstrap_replicates, bool)
        or bootstrap_replicates < 0
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
    ):
        raise ValueError("Local-Q bootstrap configuration is invalid")
    by_sequence: dict[str, tuple[LocalQEvaluationRow, ...]] = {
        sequence_id: tuple(row for row in values if row.sequence_id == sequence_id)
        for sequence_id in sorted({row.sequence_id for row in values})
    }

    def sequence_means(attribute: str) -> tuple[float, ...]:
        return tuple(
            float(np.mean([getattr(row, attribute) for row in rows]))
            for rows in by_sequence.values()
        )

    sign_values = tuple(
        float(
            np.mean(
                [
                    row.requested_sign_success
                    for row in rows
                    if row.requested_sign_success is not None
                ]
            )
        )
        for rows in by_sequence.values()
        if any(row.requested_sign_success is not None for row in rows)
    )
    dose_rows = tuple(row for row in values if row.targeted_residue_count > 0)
    result = {
        "independent_group": "sequence_id",
        "row_count": len(values),
        "sequence_count": len(by_sequence),
        "mean_signed_response": float(np.mean(sequence_means("signed_response"))),
        "mean_target_window_change": float(np.mean(sequence_means("target_window_change"))),
        "mean_nontarget_spillover": float(np.mean(sequence_means("nontarget_spillover"))),
        "mean_requested_sign_success": (None if not sign_values else float(np.mean(sign_values))),
        "mean_target_mae": float(np.mean(sequence_means("target_mae"))),
        "mean_residual_fraction": float(np.mean(sequence_means("residual_fraction"))),
        "dose_response_spearman": _spearman(
            tuple(row.control_dose for row in dose_rows),
            tuple(row.signed_response for row in dose_rows),
        ),
        "valid_residue_count": sum(row.valid_residue_count for row in values),
        "targeted_residue_count": sum(row.targeted_residue_count for row in values),
    }
    metric_attributes = {
        "mean_signed_response": "signed_response",
        "mean_target_window_change": "target_window_change",
        "mean_nontarget_spillover": "nontarget_spillover",
        "mean_target_mae": "target_mae",
        "mean_residual_fraction": "residual_fraction",
    }
    bootstrap_values: dict[str, list[float]] = {name: [] for name in metric_attributes}
    if bootstrap_replicates:
        generator = np.random.default_rng(seed)
        sequence_ids = tuple(by_sequence)
        for _ in range(bootstrap_replicates):
            selected = generator.choice(sequence_ids, size=len(sequence_ids), replace=True)
            for metric, attribute in metric_attributes.items():
                bootstrap_values[metric].append(
                    float(
                        np.mean(
                            [
                                np.mean([getattr(row, attribute) for row in by_sequence[str(item)]])
                                for item in selected
                            ]
                        )
                    )
                )
    result["bootstrap_unit"] = "sequence"
    result["bootstrap_group_count"] = len(by_sequence)
    result["bootstrap_intervals"] = {
        metric: (
            (None, None)
            if not samples
            else (
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            )
        )
        for metric, samples in bootstrap_values.items()
    }
    return result


def write_local_q_report(
    destination: str | Path,
    rows: Iterable[LocalQEvaluationRow],
    failures: Iterable[Mapping[str, object]] = (),
    *,
    manifest: Mapping[str, object],
    attempted_units: Iterable[Mapping[str, object]],
    feature_fingerprints: Mapping[str, str],
    canonical_fingerprints: Mapping[str, str],
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 0,
) -> Path:
    """Atomically publish residue-level paired Local-Q metrics and sidecars."""
    values = tuple(rows)
    failure_values = tuple(failures)
    if any(not isinstance(item, Mapping) for item in failure_values):
        raise ValueError("Local-Q failures must be mappings")
    sequence_ids = {row.sequence_id for row in values} | {
        str(item.get("sequence_id")) for item in failure_values
    }
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    if not isinstance(feature_fingerprints, Mapping) or set(feature_fingerprints) != sequence_ids:
        raise ValueError("feature_fingerprints must exactly cover Local-Q outcomes")
    required = {"manifest", "config", "checkpoint", "runtime"}
    if not isinstance(canonical_fingerprints, Mapping) or set(canonical_fingerprints) != required:
        raise ValueError("canonical_fingerprints must contain exact bundle identities")
    normalized_features = {
        sequence_id: _fingerprint(feature_fingerprints[sequence_id], "feature_fingerprint")
        for sequence_id in sorted(sequence_ids)
    }
    normalized_canonical = {
        name: _fingerprint(canonical_fingerprints[name], name) for name in sorted(required)
    }
    attempted = [dict(unit) for unit in attempted_units]
    metrics = (
        grouped_local_q_metrics(
            values,
            bootstrap_replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        if values
        else {"independent_group": "sequence_id", "row_count": 0, "sequence_count": 0}
    )
    metrics.update(
        attempted_unit_count=len(attempted),
        failure_count=len(failure_values),
        validity=(len(values) / len(attempted) if attempted else 0.0),
        coverage_numerator=len({row.sequence_id for row in values}),
        coverage_denominator=len({str(unit["sequence_id"]) for unit in attempted}),
    )
    payload = [
        {**_json_row(row), "feature_fingerprint": normalized_features[row.sequence_id]}
        for row in values
    ]
    failure_payload = [
        {
            **_json_safe(dict(item)),
            "feature_fingerprint": normalized_features[str(item["sequence_id"])],
        }
        for item in failure_values
    ]
    destination = Path(destination)
    if destination.exists():
        raise ValueError("report destination already exists; refusing partial replacement")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        _write_json(temporary / "predictions.json", payload)
        _write_csv(
            temporary / "predictions.csv",
            payload,
            (
                "sequence_id",
                "seed",
                "control_dose",
                "signed_response",
                "target_window_change",
                "nontarget_spillover",
                "requested_sign_success",
                "target_mae",
                "residual_fraction",
                "targeted_residue_count",
                "valid_residue_count",
                "feature_fingerprint",
            ),
        )
        _write_json(temporary / "failures.json", failure_payload)
        _write_csv(
            temporary / "failures.csv",
            failure_payload,
            (
                "sequence_id",
                "seed",
                "control_kind",
                "level",
                "error_type",
                "message",
                "feature_fingerprint",
            ),
        )
        _write_json(temporary / "metrics.json", metrics)
        _write_json(
            temporary / "report.json",
            {
                "attempted_units": attempted,
                "paired_rows": payload,
                "failures": failure_payload,
                "metrics": metrics,
            },
        )
        artifact_hashes = {
            name: _sha256(temporary / name)
            for name in (
                "predictions.csv",
                "predictions.json",
                "failures.csv",
                "failures.json",
                "metrics.json",
                "report.json",
            )
        }
        identity = {
            "source_manifest_fingerprint": _fingerprint(
                manifest.get("fingerprint"), "manifest fingerprint"
            ),
            "source_fingerprints": list(_source_fingerprints(manifest.get("source_fingerprints"))),
            "canonical_fingerprints": normalized_canonical,
            "feature_fingerprints": normalized_features,
            "attempted_units_fingerprint": _canonical_fingerprint(attempted),
            "artifact_hashes": artifact_hashes,
        }
        report_manifest = {"fingerprint": _canonical_fingerprint(identity), **identity}
        _write_json(temporary / "manifest.json", report_manifest)
        _write_json(
            temporary / "complete.json",
            {
                "schema_version": 1,
                "complete": True,
                "manifest_fingerprint": report_manifest["fingerprint"],
                "manifest_sha256": _sha256(temporary / "manifest.json"),
            },
        )
        _fsync_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def grouped_steering_metrics(
    rows: Iterable[SteeringEvaluationRow],
    failures: Iterable[SteeringFailure],
    bootstrap_replicates: int,
    seed: int,
    *,
    success_tolerance: float | None = None,
    success_tolerance_units: str = ABSOLUTE_TOLERANCE_UNITS,
    family_clusters: Mapping[str, str] | None = None,
) -> SteeringMetrics:
    """Aggregate by sequence and bootstrap whole sequence/family inventories.

    Every bootstrap draw expands to all successful and failed levels/seeds for the
    selected sequence or manifest-declared family cluster. Structures and samples
    are therefore never resampled as independent biological observations.

    ``success_tolerance_units`` is declared by the caller and cross-checked against
    the rows' ``property_scale`` so a sigma threshold can never be reported as an
    absolute one.
    """
    if not isinstance(bootstrap_replicates, int) or isinstance(bootstrap_replicates, bool):
        raise ValueError("bootstrap_replicates must be a nonnegative integer")
    if bootstrap_replicates < 0 or not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("bootstrap_replicates must be nonnegative and seed must be an integer")
    validated_rows, validated_failures = _validate_outcomes(tuple(rows), tuple(failures))
    observed_tolerance = _uniform_success_tolerance(validated_rows)
    configured_tolerance = (
        observed_tolerance
        if success_tolerance is None
        else _nonnegative_finite(success_tolerance, "success_tolerance")
    )
    if observed_tolerance is not None and observed_tolerance != configured_tolerance:
        raise ValueError("row success_tolerance does not match the configured success_tolerance")
    _validate_tolerance_units(success_tolerance_units, validated_rows)
    by_sequence: dict[str, list[SteeringEvaluationRow | SteeringFailure]] = defaultdict(list)
    for row in validated_rows:
        by_sequence[row.sequence_id].append(row)
    for failure in validated_failures:
        by_sequence[failure.sequence_id].append(failure)
    sequence_ids = tuple(sorted(by_sequence))
    group_by_sequence, bootstrap_groups, independent_group = _bootstrap_groups(
        sequence_ids, family_clusters
    )
    tolerance_rates = tuple(
        sum(isinstance(item, SteeringEvaluationRow) and item.success for item in by_sequence[key])
        / len(by_sequence[key])
        for key in sequence_ids
    )
    bootstrap_summaries = _bootstrap_metric_summaries(
        by_sequence,
        group_by_sequence,
        bootstrap_groups,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    bootstrap = _bootstrap_mean(bootstrap_summaries, "tolerance_success")
    intervals = _bootstrap_intervals(bootstrap_summaries)
    monotonicity, monotonicity_status = _monotonicity_by_sequence(validated_rows, sequence_ids)
    available = tuple(value for value in monotonicity.values() if value is not None)
    successful = {
        sequence_id: tuple(item for item in items if isinstance(item, SteeringEvaluationRow))
        for sequence_id, items in by_sequence.items()
    }
    valid_sequence_ids = tuple(
        sequence_id for sequence_id in sequence_ids if successful[sequence_id]
    )

    def sequence_mean(attribute: str) -> tuple[float, ...]:
        return tuple(
            float(np.mean([getattr(row, attribute) for row in successful[sequence_id]]))
            for sequence_id in valid_sequence_ids
        )

    sequence_off_mae = sequence_mean("off_absolute_error")
    sequence_on_mae = sequence_mean("on_absolute_error")
    sequence_improvement = sequence_mean("signed_movement_toward_target")
    sequence_residuals = sequence_mean("residual_fraction")
    sequence_direction_rates = tuple(
        sum(
            isinstance(item, SteeringEvaluationRow) and item.direction_success
            for item in by_sequence[key]
        )
        / len(by_sequence[key])
        for key in sequence_ids
    )
    sequence_tolerance_rates = tolerance_rates
    sequence_improvement_rates = tuple(
        float(np.mean([row.signed_movement_toward_target > 0 for row in successful[key]]))
        for key in valid_sequence_ids
    )
    return SteeringMetrics(
        independent_group=independent_group,
        sequence_count=len(sequence_ids),
        success_count=sum(row.success for row in validated_rows),
        failure_count=len(validated_failures),
        success_denominator=len(validated_rows) + len(validated_failures),
        sequence_mean_success_rate=float(np.mean(tolerance_rates)) if tolerance_rates else 0.0,
        bootstrap_success_rate=bootstrap,
        monotonicity_by_sequence=monotonicity,
        monotonicity_status_by_sequence=monotonicity_status,
        mean_monotonicity=float(np.mean(available)) if available else None,
        sequence_mean_off_absolute_error=_mean_or_zero(sequence_off_mae),
        sequence_mean_on_absolute_error=_mean_or_zero(sequence_on_mae),
        sequence_mean_improvement=_mean_or_zero(sequence_improvement),
        sequence_mean_direction_success_rate=_mean_or_zero(sequence_direction_rates),
        sequence_mean_tolerance_success_rate=_mean_or_zero(sequence_tolerance_rates),
        sequence_mean_residual_fraction=_mean_or_zero(sequence_residuals),
        validity_numerator=len(validated_rows),
        validity_denominator=len(validated_rows) + len(validated_failures),
        coverage_numerator=len(valid_sequence_ids),
        coverage_denominator=len(sequence_ids),
        target_achieved_spearman=_target_achieved_spearman(validated_rows),
        success_tolerance=configured_tolerance,
        sequence_mean_paired_improvement_rate=_mean_or_zero(sequence_improvement_rates),
        bootstrap_unit=("sequence" if independent_group == "sequence_id" else independent_group),
        bootstrap_group_count=len(bootstrap_groups),
        bootstrap_intervals=intervals,
        success_tolerance_units=success_tolerance_units,
    )


def _validate_tolerance_units(units: object, rows: tuple[SteeringEvaluationRow, ...]) -> str:
    if units not in _TOLERANCE_UNITS:
        raise ValueError(
            "success_tolerance_units must be "
            f"{' or '.join(sorted(_TOLERANCE_UNITS))}, found {units!r}"
        )
    if units == ABSOLUTE_TOLERANCE_UNITS and any(row.property_scale != 1.0 for row in rows):
        raise ValueError(
            "rows carry a non-unit property_scale, so their success_tolerance is measured "
            "in multiples of that scale and must not be reported as absolute property units"
        )
    return str(units)


_BOOTSTRAP_METRICS = (
    "target_mae",
    "paired_improvement",
    "signed_movement",
    "direction_success",
    "tolerance_success",
    "target_achieved_spearman",
    "monotonicity",
    "residual_fraction",
    "validity",
    "coverage",
)


def _bootstrap_groups(
    sequence_ids: tuple[str, ...], family_clusters: Mapping[str, str] | None
) -> tuple[dict[str, str], tuple[str, ...], str]:
    if family_clusters is None:
        return (
            {sequence_id: sequence_id for sequence_id in sequence_ids},
            sequence_ids,
            "sequence_id",
        )
    if not isinstance(family_clusters, Mapping) or set(family_clusters) != set(sequence_ids):
        raise ValueError("family_clusters must map every and only observed sequence ID")
    normalized: dict[str, str] = {}
    for sequence_id in sequence_ids:
        cluster = family_clusters[sequence_id]
        if not isinstance(cluster, str) or not cluster:
            raise ValueError("family cluster IDs must be non-empty strings")
        normalized[sequence_id] = cluster
    return normalized, tuple(sorted(set(normalized.values()))), "family_cluster"


def _bootstrap_metric_summaries(
    by_sequence: Mapping[str, list[SteeringEvaluationRow | SteeringFailure]],
    group_by_sequence: Mapping[str, str],
    groups: tuple[str, ...],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[dict[str, float | None], ...]:
    if not bootstrap_replicates or not groups:
        return ()
    members = {
        group: tuple(
            sequence_id
            for sequence_id in sorted(by_sequence)
            if group_by_sequence[sequence_id] == group
        )
        for group in groups
    }
    generator = np.random.default_rng(seed)
    summaries = []
    for _ in range(bootstrap_replicates):
        selected = generator.choice(groups, size=len(groups), replace=True)
        sequence_occurrences = tuple(
            sequence_id for group in selected for sequence_id in members[str(group)]
        )
        summaries.append(_resampled_metric_summary(by_sequence, sequence_occurrences))
    return tuple(summaries)


def _resampled_metric_summary(
    by_sequence: Mapping[str, list[SteeringEvaluationRow | SteeringFailure]],
    sequence_occurrences: tuple[str, ...],
) -> dict[str, float | None]:
    successful_occurrences: list[tuple[SteeringEvaluationRow, ...]] = []
    all_rows: list[SteeringEvaluationRow] = []
    attempted = 0
    covered = 0
    monotonicity: list[float] = []
    for occurrence_index, sequence_id in enumerate(sequence_occurrences):
        items = by_sequence[sequence_id]
        rows = tuple(item for item in items if isinstance(item, SteeringEvaluationRow))
        attempted += len(items)
        all_rows.extend(rows)
        if rows:
            covered += 1
            successful_occurrences.append(rows)
            targets = tuple(row.target for row in rows)
            achieved = tuple(row.adapter_on_value for row in rows)
            if len(set(targets)) >= 3 and len(set(achieved)) >= 2:
                value = _spearman(targets, achieved)
                if value is not None:
                    monotonicity.append(value)
        del occurrence_index  # The occurrence itself, not its label, carries bootstrap weight.

    def sequence_mean(attribute: str) -> float:
        values = [
            float(np.mean([getattr(row, attribute) for row in rows]))
            for rows in successful_occurrences
        ]
        return _mean_or_zero(tuple(values))

    paired = [
        float(np.mean([row.signed_movement_toward_target > 0 for row in rows]))
        for rows in successful_occurrences
    ]
    direction = []
    tolerance = []
    for sequence_id in sequence_occurrences:
        items = by_sequence[sequence_id]
        direction.append(
            sum(
                isinstance(item, SteeringEvaluationRow) and item.direction_success for item in items
            )
            / len(items)
        )
        tolerance.append(
            sum(isinstance(item, SteeringEvaluationRow) and item.success for item in items)
            / len(items)
        )
    return {
        "target_mae": sequence_mean("on_absolute_error"),
        "paired_improvement": _mean_or_zero(tuple(paired)),
        "signed_movement": sequence_mean("signed_movement_toward_target"),
        "direction_success": _mean_or_zero(tuple(direction)),
        "tolerance_success": _mean_or_zero(tuple(tolerance)),
        "target_achieved_spearman": _target_achieved_spearman(tuple(all_rows)),
        "monotonicity": float(np.mean(monotonicity)) if monotonicity else None,
        "residual_fraction": sequence_mean("residual_fraction"),
        "validity": len(all_rows) / attempted if attempted else 0.0,
        "coverage": covered / len(sequence_occurrences) if sequence_occurrences else 0.0,
    }


def _bootstrap_intervals(
    summaries: tuple[dict[str, float | None], ...],
) -> dict[str, tuple[float | None, float | None]]:
    intervals: dict[str, tuple[float | None, float | None]] = {}
    for metric in _BOOTSTRAP_METRICS:
        values = [summary[metric] for summary in summaries if summary[metric] is not None]
        intervals[metric] = (
            (None, None)
            if not values
            else (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        )
    intervals["mae"] = intervals["target_mae"]
    intervals["movement"] = intervals["signed_movement"]
    return intervals


def _bootstrap_mean(summaries: tuple[dict[str, float | None], ...], metric: str) -> float | None:
    values = [summary[metric] for summary in summaries if summary[metric] is not None]
    return None if not values else float(np.mean(values))


def _evaluation_units(units: Iterable[SteeringUnit]) -> tuple[SteeringUnit, ...]:
    attempted = tuple(units)
    if any(not isinstance(unit, SteeringUnit) for unit in attempted):
        raise ValueError("units must contain SteeringUnit values")
    if any(unit.split != "test" for unit in attempted):
        raise ValueError("paired evaluation only accepts test units")
    keys = tuple((unit.sequence_id, unit.seed, float(unit.level)) for unit in attempted)
    if len(set(keys)) != len(keys):
        raise ValueError("paired evaluation units must not contain duplicate keys")
    return tuple(sorted(attempted, key=lambda unit: (unit.sequence_id, unit.seed, unit.level)))


def _generated_record(
    unit: SteeringUnit, target: float, generated: Mapping[str, object]
) -> dict[str, object]:
    if not isinstance(generated, Mapping):
        raise ValueError("generation callback must return a mapping")
    if "structure" not in generated or "residual_fraction" not in generated:
        raise ValueError("generated record requires structure and residual_fraction")
    _residual_fraction(generated["residual_fraction"])
    if "target" in generated and not _approximately_equal(
        _finite(generated["target"], "generated target"), target
    ):
        raise ValueError("generated target does not agree with the resolved absolute target")
    return {
        "sequence_id": unit.sequence_id,
        "seed": unit.seed,
        "target": target,
        "structure": generated["structure"],
        "residual_fraction": generated["residual_fraction"],
    }


def write_steering_report(
    destination: str | Path,
    rows: Iterable[SteeringEvaluationRow],
    failures: Iterable[SteeringFailure],
    manifest: Mapping[str, object],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    success_tolerance: float,
    success_tolerance_units: str = ABSOLUTE_TOLERANCE_UNITS,
    family_clusters: Mapping[str, str] | None = None,
    family_cluster_assignments: Mapping[str, str] | None = None,
    feature_fingerprints: Mapping[str, str] | None = None,
    canonical_fingerprints: Mapping[str, str] | None = None,
    attempted_units: Iterable[Mapping[str, object]] | None = None,
) -> Path:
    """Atomically publish complete CSV/JSON sidecars and their inventory manifest."""
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    metric_config = _metric_config(
        bootstrap_replicates, bootstrap_seed, success_tolerance, success_tolerance_units
    )
    source_manifest_fingerprint = _fingerprint(manifest.get("fingerprint"), "manifest fingerprint")
    source_fingerprints = _source_fingerprints(manifest.get("source_fingerprints"))
    valid_rows, valid_failures = _validate_outcomes(tuple(rows), tuple(failures))
    observed_sequence_ids = tuple(
        sorted({row.sequence_id for row in (*valid_rows, *valid_failures)})
    )
    observed_families = (
        None
        if family_clusters is None
        else _bootstrap_groups(observed_sequence_ids, family_clusters)[0]
    )
    # A row that already names its own cluster must agree with the map the bootstrap groups by,
    # or the published CSV would describe a different grouping from the published metrics.
    for row in valid_rows:
        if row.identity_cluster is None:
            continue
        if observed_families is None:
            raise ValueError(
                "rows carry an identity_cluster but no family_clusters were supplied, so the "
                "metrics would be sequence-grouped while predictions.csv named clusters (D-24)"
            )
        if observed_families[row.sequence_id] != row.identity_cluster:
            raise ValueError(
                "row identity_cluster disagrees with family_clusters for "
                f"{row.sequence_id!r}: {row.identity_cluster!r} against "
                f"{observed_families[row.sequence_id]!r}"
            )
    assignments = (
        family_clusters if family_cluster_assignments is None else family_cluster_assignments
    )
    if assignments is None:
        normalized_families = None
    else:
        if not isinstance(assignments, Mapping) or not set(observed_sequence_ids).issubset(
            assignments
        ):
            raise ValueError("family_cluster_assignments must cover every observed sequence ID")
        normalized_families = {}
        for sequence_id in sorted(assignments):
            cluster = assignments[sequence_id]
            if (
                not isinstance(sequence_id, str)
                or not sequence_id
                or not isinstance(cluster, str)
                or not cluster
            ):
                raise ValueError("family cluster assignments require non-empty string IDs")
            normalized_families[sequence_id] = cluster
        if observed_families is not None and any(
            normalized_families[sequence_id] != cluster
            for sequence_id, cluster in observed_families.items()
        ):
            raise ValueError("family cluster assignments disagree with bootstrap groups")
    if feature_fingerprints is None:
        normalized_features = None
    else:
        if not isinstance(feature_fingerprints, Mapping) or set(feature_fingerprints) != set(
            observed_sequence_ids
        ):
            raise ValueError("feature_fingerprints must map every and only observed sequence ID")
        normalized_features = {
            sequence_id: _fingerprint(feature_fingerprints[sequence_id], "feature_fingerprint")
            for sequence_id in observed_sequence_ids
        }
    expected = _manifest_inventory(manifest.get("expected_inventory"))
    observed_success = tuple(row.key for row in valid_rows)
    observed_failure = tuple(failure.key for failure in valid_failures)
    observed = set(observed_success) | set(observed_failure)
    if expected is not None and observed != set(expected):
        raise ValueError("expected inventory does not match observed successes and failures")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("report destination already exists; refusing partial replacement")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        prediction_payload = [_json_row(row) for row in valid_rows]
        failure_payload = [_json_row(row) for row in valid_failures]
        # D-24: the published rows name the unit the metrics were resampled on. Filled from
        # ``family_clusters`` -- the map the bootstrap actually grouped by -- and NOT from
        # ``family_cluster_assignments``, which can cover sequences whose clusters were not the
        # grouping. A cluster column beside a sequence-grouped interval is exactly the confusion
        # this closes.
        if observed_families is not None:
            for payload in prediction_payload:
                payload["identity_cluster"] = observed_families[str(payload["sequence_id"])]
        if normalized_features is not None:
            for payload in (*prediction_payload, *failure_payload):
                payload["feature_fingerprint"] = normalized_features[str(payload["sequence_id"])]
        metrics = grouped_steering_metrics(
            valid_rows,
            valid_failures,
            bootstrap_replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            success_tolerance=success_tolerance,
            success_tolerance_units=success_tolerance_units,
            family_clusters=family_clusters,
        ).to_payload()
        attempted_payload = (
            [dict(item) for item in attempted_units]
            if attempted_units is not None
            else [
                {"sequence_id": sequence_id, "seed": seed, "target": target}
                for sequence_id, seed, target in (expected or tuple(sorted(observed)))
            ]
        )
        _write_json(temporary / "predictions.json", prediction_payload)
        feature_fields = ("feature_fingerprint",) if normalized_features is not None else ()
        _write_csv(
            temporary / "predictions.csv",
            prediction_payload,
            (*_prediction_fields(), *feature_fields),
        )
        _write_json(temporary / "failures.json", failure_payload)
        _write_csv(
            temporary / "failures.csv",
            failure_payload,
            (*_failure_fields(), *feature_fields),
        )
        _write_json(temporary / "metrics.json", metrics)
        _write_json(temporary / "intervals.json", metrics["bootstrap_intervals"])
        _write_json(
            temporary / "report.json",
            {
                "attempted_units": attempted_payload,
                "paired_rows": prediction_payload,
                "failures": failure_payload,
                "metrics": metrics,
                "family_clusters": normalized_families,
                "feature_fingerprints": normalized_features,
            },
        )
        artifact_names = (
            "predictions.csv",
            "predictions.json",
            "failures.csv",
            "failures.json",
            "metrics.json",
            "intervals.json",
            "report.json",
        )
        artifact_hashes = {name: _sha256(temporary / name) for name in artifact_names}
        report_identity = {
            "source_manifest_fingerprint": source_manifest_fingerprint,
            "source_fingerprints": list(source_fingerprints),
            "expected_inventory": _inventory_payload(expected) if expected is not None else None,
            "observed_success_inventory": _inventory_payload(observed_success),
            "observed_failure_inventory": _inventory_payload(observed_failure),
            "artifact_hashes": artifact_hashes,
            "metric_config": metric_config,
            "family_clusters": normalized_families,
            "family_clusters_fingerprint": (
                None if normalized_families is None else _canonical_fingerprint(normalized_families)
            ),
            "feature_fingerprints": normalized_features,
        }
        if canonical_fingerprints is not None:
            # Exactly two accepted shapes, both closed sets. ``evaluation_code`` is the
            # optional D14 record of which evaluation code produced these numbers; it is
            # provenance, and callers that do not split their code identity keep the
            # historical four-key bundle unchanged.
            required = {"manifest", "config", "checkpoint", "runtime"}
            supplied = (
                set(canonical_fingerprints) if isinstance(canonical_fingerprints, Mapping) else None
            )
            if supplied not in (required, required | {"evaluation_code"}):
                raise ValueError("canonical_fingerprints must contain exact bundle identities")
            report_identity["canonical_fingerprints"] = {
                name: _fingerprint(canonical_fingerprints[name], name) for name in sorted(supplied)
            }
        report_manifest = {
            "fingerprint": _canonical_fingerprint(report_identity),
            **report_identity,
        }
        _write_json(temporary / "manifest.json", report_manifest)
        _write_json(
            temporary / "complete.json",
            {
                "schema_version": 1,
                "complete": True,
                "manifest_fingerprint": report_manifest["fingerprint"],
                "manifest_sha256": _sha256(temporary / "manifest.json"),
            },
        )
        _fsync_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _inventory(
    records: Iterable[Mapping[str, object]], name: str
) -> dict[PairKey, Mapping[str, object]]:
    result: dict[PairKey, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"{name} records must be mappings")
        key = _key(record)
        if key in result:
            raise ValueError(f"{name} contains duplicate paired key {key!r}")
        _residual_fraction(_field(record, "residual_fraction"))
        result[key] = record
    return result


def _key(record: Mapping[str, object]) -> PairKey:
    return _validate_key(
        (_field(record, "sequence_id"), _field(record, "seed"), _field(record, "target"))
    )


def _validate_key(values: tuple[object, object, object]) -> PairKey:
    sequence_id, seed, target = values
    if not isinstance(sequence_id, str) or not sequence_id:
        raise ValueError("sequence_id must be a non-empty string")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    return (sequence_id, seed, _finite(target, "target"))


def _field(record: Mapping[str, object], name: str) -> object:
    if name not in record:
        raise ValueError(f"record missing {name}")
    return record[name]


def _property_value(property_fn: Callable[[object], float], record: Mapping[str, object]) -> float:
    return _finite(property_fn(_field(record, "structure")), "authoritative property value")


def _failure(key: PairKey, phase: str, error: Exception) -> SteeringFailure:
    return SteeringFailure(*key, phase, type(error).__name__, str(error) or repr(error))


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _nonnegative_finite(value: object, name: str) -> float:
    value = _finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive_finite(value: object, name: str) -> float:
    value = _finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _approximately_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def _direction_success(off_value: float, on_value: float, target: float) -> bool:
    return _sign(on_value - off_value) == _sign(target - off_value)


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def _residual_fraction(value: object) -> float:
    value = _nonnegative_finite(value, "residual_fraction")
    if value > 1:
        raise ValueError("residual_fraction must be bounded by one")
    return value


def _validate_outcomes(
    rows: tuple[SteeringEvaluationRow, ...], failures: tuple[SteeringFailure, ...]
) -> tuple[tuple[SteeringEvaluationRow, ...], tuple[SteeringFailure, ...]]:
    if any(not isinstance(row, SteeringEvaluationRow) for row in rows):
        raise ValueError("rows must contain SteeringEvaluationRow values")
    if any(not isinstance(failure, SteeringFailure) for failure in failures):
        raise ValueError("failures must contain SteeringFailure values")
    row_keys, failure_keys = {row.key for row in rows}, {failure.key for failure in failures}
    if len(row_keys) != len(rows) or len(failure_keys) != len(failures):
        raise ValueError("success and failure keys must not be duplicated")
    if row_keys & failure_keys:
        raise ValueError("failure keys must not collide with successful keys")
    return tuple(sorted(rows, key=lambda row: row.key)), tuple(
        sorted(failures, key=lambda failure: failure.key)
    )


def _monotonicity_by_sequence(
    rows: tuple[SteeringEvaluationRow, ...], sequence_ids: tuple[str, ...]
) -> tuple[dict[str, float | None], dict[str, str]]:
    grouped: dict[str, list[SteeringEvaluationRow]] = defaultdict(list)
    for row in rows:
        grouped[row.sequence_id].append(row)
    result: dict[str, float | None] = {}
    status: dict[str, str] = {}
    for sequence_id in sequence_ids:
        records = grouped.get(sequence_id, [])
        if not records:
            result[sequence_id] = None
            status[sequence_id] = "no_successful_rows"
            continue
        targets = tuple(record.target for record in records)
        achieved = tuple(record.adapter_on_value for record in records)
        if len(set(targets)) < 3:
            result[sequence_id] = None
            status[sequence_id] = "insufficient_distinct_targets"
        elif len(set(achieved)) < 2:
            result[sequence_id] = None
            status[sequence_id] = "constant_achieved"
        else:
            result[sequence_id] = _spearman(targets, achieved)
            status[sequence_id] = "ok"
    return result, status


def _target_achieved_spearman(rows: tuple[SteeringEvaluationRow, ...]) -> float | None:
    targets = tuple(row.target for row in rows)
    achieved = tuple(row.adapter_on_value for row in rows)
    if len(set(targets)) < 3 or len(set(achieved)) < 2:
        return None
    return _spearman(targets, achieved)


def _mean_or_zero(values: tuple[float, ...]) -> float:
    return float(np.mean(values)) if values else 0.0


def _uniform_success_tolerance(rows: tuple[SteeringEvaluationRow, ...]) -> float | None:
    if not rows:
        return None
    tolerance = rows[0].success_tolerance
    if any(row.success_tolerance != tolerance for row in rows[1:]):
        raise ValueError("rows must have one uniform success_tolerance")
    return tolerance


def _spearman(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    left_ranks, right_ranks = _ranks(left), _ranks(right)
    left_centered = np.asarray(left_ranks) - np.mean(left_ranks)
    right_centered = np.asarray(right_ranks) - np.mean(right_ranks)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    return None if denominator == 0 else float(np.dot(left_centered, right_centered) / denominator)


def _ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    ranks = [0.0] * len(values)
    for index, value in enumerate(values):
        tied = [other for other, candidate in enumerate(values) if candidate == value]
        ranks[index] = 1 + sum(other < value for other in values) + (len(tied) - 1) / 2
    return tuple(ranks)


def _manifest_inventory(value: object) -> tuple[PairKey, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("expected_inventory must be a list")
    inventory = tuple(
        _validate_key(tuple(item))
        for item in value
        if isinstance(item, list | tuple) and len(item) == 3
    )
    if len(inventory) != len(value) or len(set(inventory)) != len(inventory):
        raise ValueError("expected_inventory must contain unique [sequence_id, seed, target] keys")
    return tuple(sorted(inventory))


def _metric_config(
    bootstrap_replicates: object,
    bootstrap_seed: object,
    success_tolerance: object,
    success_tolerance_units: object = ABSOLUTE_TOLERANCE_UNITS,
) -> dict[str, int | float | str]:
    if not isinstance(bootstrap_replicates, int) or isinstance(bootstrap_replicates, bool):
        raise ValueError("bootstrap_replicates must be a positive integer for published reports")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be a positive integer for published reports")
    if not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool):
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if success_tolerance_units not in _TOLERANCE_UNITS:
        raise ValueError(
            "success_tolerance_units must be "
            f"{' or '.join(sorted(_TOLERANCE_UNITS))}, found {success_tolerance_units!r}"
        )
    return {
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "success_tolerance": _nonnegative_finite(success_tolerance, "success_tolerance"),
        "success_tolerance_units": str(success_tolerance_units),
    }


def _source_fingerprints(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("source_fingerprints must be a non-empty list")
    fingerprints = tuple(_fingerprint(item, "source_fingerprints") for item in value)
    return tuple(sorted(fingerprints))


def _fingerprint(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 fingerprint")
    return value


def _canonical_fingerprint(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _prediction_fields() -> tuple[str, ...]:
    return tuple(SteeringEvaluationRow.__dataclass_fields__)


def _failure_fields() -> tuple[str, ...]:
    return tuple(SteeringFailure.__dataclass_fields__)


def _json_row(row: object) -> dict[str, object]:
    return _json_safe(asdict(row))


def _inventory_payload(keys: Iterable[PairKey]) -> list[list[object]]:
    return [[sequence_id, seed, target] for sequence_id, seed, target in sorted(keys)]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("report payload contains non-finite value")
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise ValueError(f"report payload is not JSON-safe: {type(value).__name__}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(_json_safe(payload), sort_keys=True, indent=2) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
