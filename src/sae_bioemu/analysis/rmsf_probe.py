from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm as mpl_cm
from matplotlib import colors as mpl_colors
from scipy import stats as scipy_stats

from sae_bioemu.analysis.prior_probe_comparison import (
    PROBE_COMPARISON_ARTIFACTS,
    ProbeComparisonSeries,
    build_normalized_comparison_summary,
    prepare_comparison_directory,
    write_probe_comparison,
)

DEFAULT_RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
DEFAULT_RMSF_FEATURE_COUNTS = (1, 2, 4, 8, 16, 32, 64)
HISTORICAL_RMSF_OUTER_FOLDS = 5
HISTORICAL_RMSF_INNER_FOLDS = 3
HISTORICAL_RMSF_FOLD_SEED = 0
HISTORICAL_RMSF_MAX_ROWS = 4096
HISTORICAL_RMSF_RESERVOIR_SEED = 0


@dataclass(frozen=True)
class PredictionRowKey:
    """Durable identity for one residue-aligned prediction row."""

    sequence_id: str
    residue_index: int
    sample_index: int | None
    source: str


class _PredictionRowKeyLike(Protocol):
    sequence_id: str
    residue_index: int
    sample_index: int | None
    source: str


@dataclass(frozen=True)
class RMSFSourceIdentity:
    """Minimal report-backed source data required for an RMSF comparison."""

    row_keys: tuple[PredictionRowKey, ...]
    targets: np.ndarray
    groups: np.ndarray
    fold_ids: np.ndarray
    prediction_rows: tuple[Mapping[str, object], ...]
    feature_counts: tuple[int, ...]
    probe_config: Mapping[str, object]
    sampling_identity: Mapping[str, object]
    input_provenance: Mapping[str, object]
    source_latent_width: int
    row_identity_sha256: str
    fold_identity_sha256: str


@dataclass(frozen=True)
class RidgeProbeModel:
    fold: int
    feature_indices: tuple[int, ...]
    feature_means: np.ndarray
    feature_scales: np.ndarray
    weights: np.ndarray
    intercept: float
    alpha: float
    training_groups: tuple[str, ...]
    heldout_groups: tuple[str, ...]


@dataclass(frozen=True)
class RMSFProbeResult:
    models: tuple[RidgeProbeModel, ...]
    oof_predictions: np.ndarray
    fold_ids: np.ndarray
    valid_mask: np.ndarray
    fold_metrics: tuple[dict[str, Any], ...]
    sequence_metrics: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def deterministic_group_folds(
    group_ids: np.ndarray,
    max_folds: int,
    *,
    fold_seed: int = 0,
) -> np.ndarray:
    if max_folds < 2:
        raise ValueError("max_folds must be at least 2")
    groups = sorted(
        {str(group) for group in group_ids},
        key=lambda value: hashlib.sha256(
            (value if fold_seed == 0 else f"{fold_seed}\0{value}").encode("utf-8")
        ).digest(),
    )
    n_folds = min(max_folds, len(groups))
    if n_folds < 2:
        return np.full(len(group_ids), -1, dtype=np.int16)
    fold_by_group = {group: index % n_folds for index, group in enumerate(groups)}
    return np.asarray([fold_by_group[str(group)] for group in group_ids], dtype=np.int16)


def predict_ridge_model(model: RidgeProbeModel, latents: np.ndarray) -> np.ndarray:
    values = np.asarray(latents)
    selected = values[:, model.feature_indices].astype(np.float64, copy=False)
    standardized = (selected - model.feature_means) / model.feature_scales
    return standardized @ model.weights + model.intercept


def run_grouped_rmsf_probe(
    latents: np.ndarray,
    rmsf: np.ndarray,
    group_ids: np.ndarray,
    *,
    outer_folds: int = 5,
    inner_folds: int = 3,
    top_features: int = 64,
    alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    fold_seed: int = 0,
) -> RMSFProbeResult:
    values = np.asarray(latents)
    targets = np.asarray(rmsf, dtype=np.float64)
    groups = np.asarray(group_ids, dtype=object)
    if values.ndim != 2:
        raise ValueError("latents must have shape [rows, features]")
    if targets.shape != (values.shape[0],) or groups.shape != (values.shape[0],):
        raise ValueError("latents, rmsf, and group_ids must have matching rows")
    if top_features <= 0:
        raise ValueError("top_features must be positive")
    if not alphas or any(alpha < 0 for alpha in alphas):
        raise ValueError("alphas must contain non-negative values")

    valid_mask = np.isfinite(targets) & np.isfinite(values).all(axis=1)
    predictions = np.full(values.shape[0], np.nan, dtype=np.float64)
    fold_ids = np.full(values.shape[0], -1, dtype=np.int16)
    valid_targets = targets[valid_mask]
    valid_groups = groups[valid_mask]
    if valid_targets.size == 0:
        return _skipped_result(predictions, fold_ids, valid_mask, "no finite RMSF rows")
    unique_groups = np.unique(valid_groups.astype(str))
    if unique_groups.size < 2:
        return _skipped_result(predictions, fold_ids, valid_mask, "fewer than two proteins")
    if float(np.std(valid_targets)) <= 1e-12:
        return _skipped_result(
            predictions,
            fold_ids,
            valid_mask,
            "RMSF target has no variance",
        )

    valid_folds = deterministic_group_folds(
        valid_groups,
        outer_folds,
        fold_seed=fold_seed,
    )
    fold_ids[valid_mask] = valid_folds
    valid_indices = np.flatnonzero(valid_mask)
    models: list[RidgeProbeModel] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(int(value) for value in np.unique(valid_folds) if value >= 0):
        train_local = valid_folds != fold
        test_local = valid_folds == fold
        if int(train_local.sum()) < 3 or int(test_local.sum()) < 2:
            fold_rows.append({"fold": fold, "status": "skipped", "reason": "insufficient rows"})
            continue
        train_indices = valid_indices[train_local]
        test_indices = valid_indices[test_local]
        feature_indices = _select_features(
            values[train_indices], targets[train_indices], top_features
        )
        if feature_indices.size == 0:
            fold_rows.append({"fold": fold, "status": "skipped", "reason": "no variable features"})
            continue
        alpha = _select_alpha(
            values[train_indices],
            targets[train_indices],
            groups[train_indices],
            feature_indices,
            alphas,
            inner_folds,
            fold_seed,
        )
        training_groups = tuple(sorted({str(value) for value in groups[train_indices]}))
        heldout_groups = tuple(sorted({str(value) for value in groups[test_indices]}))
        model = _fit_ridge(
            values[train_indices],
            targets[train_indices],
            feature_indices,
            alpha,
            fold,
            training_groups,
            heldout_groups,
        )
        test_predictions = predict_ridge_model(model, values[test_indices])
        predictions[test_indices] = test_predictions
        fold_metrics = _metrics(targets[test_indices], test_predictions)
        fold_rows.append(
            {
                "fold": fold,
                "status": "ok",
                "n_train_rows": int(train_indices.size),
                "n_test_rows": int(test_indices.size),
                "n_train_groups": len(training_groups),
                "n_test_groups": len(heldout_groups),
                "alpha": alpha,
                "num_features": len(model.feature_indices),
                **fold_metrics,
            }
        )
        models.append(model)

    scored_mask = valid_mask & np.isfinite(predictions)
    sequence_rows = _sequence_metrics(targets, predictions, groups, scored_mask)
    if not bool(scored_mask.any()):
        summary = {
            "status": "skipped",
            "reason": "no outer folds could be scored",
            "n_rows": int(valid_mask.sum()),
            "n_groups": int(unique_groups.size),
            "n_folds": len(models),
        }
    else:
        summary = {
            "status": "ok",
            "n_rows": int(valid_mask.sum()),
            "n_scored_rows": int(scored_mask.sum()),
            "n_groups": int(unique_groups.size),
            "n_folds": len(models),
            **{
                f"oof_{key}": value
                for key, value in _metrics(targets[scored_mask], predictions[scored_mask]).items()
            },
            "macro_spearman": _finite_mean(row["spearman"] for row in sequence_rows),
            "macro_r2": _finite_mean(row["r2"] for row in sequence_rows),
            "macro_mae": _finite_mean(row["mae"] for row in sequence_rows),
        }
    return RMSFProbeResult(
        models=tuple(models),
        oof_predictions=predictions,
        fold_ids=fold_ids,
        valid_mask=valid_mask,
        fold_metrics=tuple(fold_rows),
        sequence_metrics=tuple(sequence_rows),
        summary=summary,
    )


def run_grouped_rmsf_probe_sweep(
    latents: np.ndarray,
    rmsf: np.ndarray,
    group_ids: np.ndarray,
    *,
    feature_counts: tuple[int, ...] = DEFAULT_RMSF_FEATURE_COUNTS,
    outer_folds: int = 5,
    inner_folds: int = 3,
    alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    fold_seed: int = 0,
) -> dict[int, RMSFProbeResult]:
    if not feature_counts:
        raise ValueError("feature_counts must not be empty")
    if any(count <= 0 for count in feature_counts):
        raise ValueError("feature_counts must contain positive values")
    if len(set(feature_counts)) != len(feature_counts):
        raise ValueError("feature_counts must be unique")
    return {
        count: run_grouped_rmsf_probe(
            latents,
            rmsf,
            group_ids,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            top_features=count,
            alphas=alphas,
            fold_seed=fold_seed,
        )
        for count in feature_counts
    }


def prediction_row_key_payload(key: _PredictionRowKeyLike) -> dict[str, object]:
    return {
        "sequence_id": key.sequence_id,
        "residue_index": key.residue_index,
        "sample_index": key.sample_index,
        "source": key.source,
    }


def rmsf_row_identity_sha256(
    row_keys: Sequence[_PredictionRowKeyLike],
    targets: np.ndarray,
    group_ids: np.ndarray,
) -> str:
    values = np.asarray(targets, dtype=np.float64)
    groups = np.asarray(group_ids, dtype=object)
    if len(row_keys) != values.size or groups.shape != values.shape:
        raise ValueError("RMSF row keys, targets, and groups must have matching rows")
    payload = [
        {
            "row": row,
            "row_key": prediction_row_key_payload(key),
            "group_id": str(groups[row]),
            "target_hex": float(values[row]).hex(),
        }
        for row, key in enumerate(row_keys)
    ]
    return _stable_sha256(payload)


def rmsf_fold_identity_sha256(
    fold_ids: np.ndarray,
    row_keys: Sequence[_PredictionRowKeyLike],
) -> str:
    """Fingerprint exact row keys and their row-level fold assignments."""

    folds = np.asarray(fold_ids, dtype=np.int64)
    if folds.shape != (len(row_keys),):
        raise ValueError("fold_ids and row_keys must have matching rows")
    payload = [
        {
            "row": row,
            "row_key": prediction_row_key_payload(key),
            "fold": int(fold),
        }
        for row, (key, fold) in enumerate(zip(row_keys, folds, strict=True))
    ]
    return _stable_sha256(payload)


def load_rmsf_source_identity(
    source_report: Mapping[str, object] | Path,
    *,
    legacy_rmsf_by_sequence: Mapping[str, np.ndarray] | None = None,
    legacy_probe_config: Mapping[str, object] | None = None,
    legacy_sampling_identity: Mapping[str, object] | None = None,
    legacy_source_latent_width: int | None = None,
    legacy_input_provenance: Mapping[str, object] | None = None,
    legacy_row_source: str = "legacy_rmsf_value_match",
    legacy_sample_index: int | None = None,
) -> RMSFSourceIdentity:
    """Load and strictly validate the minimal RMSF source identity from a report."""

    report, report_dir = _load_source_report(source_report)
    artifacts = report.get("artifacts")
    raw_path = artifacts.get("oof_predictions_csv") if isinstance(artifacts, Mapping) else None
    prediction_path = _resolve_source_artifact(raw_path, report_dir)
    if prediction_path is None or not prediction_path.is_file():
        raise ValueError("source RMSF report is missing OOF predictions")
    with prediction_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("source RMSF OOF rows must be complete and unique")
    try:
        row_indices = [int(row["row"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source RMSF OOF rows must be complete and unique") from exc
    if sorted(row_indices) != list(range(len(rows))) or len(set(row_indices)) != len(rows):
        raise ValueError("source RMSF OOF rows must be complete and unique")
    reported_rows = report.get("num_prediction_rows")
    if reported_rows is not None and int(reported_rows) != len(rows):
        raise ValueError("source RMSF OOF rows must be complete and unique")
    ordered = [row for _, row in sorted(zip(row_indices, rows, strict=True))]

    is_keyed = all(
        {"sequence_id", "residue_index", "sample_index", "source"} <= row.keys()
        and row["residue_index"] != ""
        and row["source"] != ""
        for row in ordered
    )
    targets = np.asarray([float(row["true_rmsf"]) for row in ordered], dtype=np.float64)
    groups = np.asarray([str(row["sequence_id"]) for row in ordered], dtype=object)
    folds = np.asarray([int(row["fold"]) for row in ordered], dtype=np.int16)
    if is_keyed:
        keys = tuple(
            PredictionRowKey(
                sequence_id=str(row["sequence_id"]),
                residue_index=int(row["residue_index"]),
                sample_index=_optional_int(row["sample_index"]),
                source=str(row["source"]),
            )
            for row in ordered
        )
        probe_config = _normalize_probe_config(report.get("probe_config"))
        sampling_identity = _normalize_sampling_identity(
            report.get("sampling_identity"),
            expected_rows=len(rows),
        )
        source_width = _positive_int(
            report.get("source_latent_width"),
            "source_latent_width",
        )
    else:
        if legacy_rmsf_by_sequence is None:
            raise ValueError(
                "legacy RMSF report lacks row keys; provide exact RMSF references "
                "or regenerate the report"
            )
        keys = _legacy_row_keys(
            groups,
            targets,
            legacy_rmsf_by_sequence,
            source=legacy_row_source,
            sample_index=legacy_sample_index,
        )
        report_probe_config = report.get("probe_config")
        if report_probe_config is None and legacy_probe_config is None:
            raise ValueError(
                "legacy RMSF report cannot prove its historical probe configuration; "
                "regenerate or migrate the source report"
            )
        report_sampling = report.get("sampling_identity")
        report_width = report.get("source_latent_width")
        if (
            report_sampling is None
            and legacy_sampling_identity is None
            or report_width is None
            and legacy_source_latent_width is None
        ):
            raise ValueError(
                "legacy RMSF report cannot prove its historical sampling identity; "
                "regenerate or migrate the source report"
            )
        probe_config = _normalize_probe_config(
            report_probe_config if report_probe_config is not None else legacy_probe_config
        )
        _require_historical_legacy_probe_config(probe_config)
        sampling_identity = _normalize_sampling_identity(
            report_sampling if report_sampling is not None else legacy_sampling_identity,
            expected_rows=len(rows),
        )
        _require_historical_legacy_sampling(sampling_identity)
        _require_historical_legacy_folds(folds, groups, targets)
        source_width = _positive_int(
            report_width if report_width is not None else legacy_source_latent_width,
            "source_latent_width",
        )
        if (
            report_probe_config is not None
            and legacy_probe_config is not None
            and (_normalize_probe_config(legacy_probe_config) != probe_config)
        ):
            raise ValueError("legacy asserted probe configuration does not match source report")
        if (
            report_sampling is not None
            and legacy_sampling_identity is not None
            and (
                _normalize_sampling_identity(legacy_sampling_identity, expected_rows=len(rows))
                != sampling_identity
            )
        ):
            raise ValueError("legacy asserted sampling identity does not match source report")
        if (
            report_width is not None
            and legacy_source_latent_width is not None
            and _positive_int(legacy_source_latent_width, "legacy_source_latent_width")
            != source_width
        ):
            raise ValueError("legacy asserted latent width does not match source report")

    input_provenance = _normalize_input_provenance(
        (
            report.get("input_provenance")
            if is_keyed or report.get("input_provenance") is not None
            else legacy_input_provenance
        ),
        expected_sequence_ids={str(group) for group in groups},
    )
    if (
        not is_keyed
        and report.get("input_provenance") is not None
        and legacy_input_provenance is not None
        and input_provenance
        != _normalize_input_provenance(
            legacy_input_provenance,
            expected_sequence_ids={str(group) for group in groups},
        )
    ):
        raise ValueError("legacy asserted input provenance does not match source report")

    if len(set(keys)) != len(keys):
        raise ValueError("duplicate RMSF OOF row key")
    if any(key.sequence_id != str(group) for key, group in zip(keys, groups, strict=True)):
        raise ValueError("RMSF OOF row key and protein group do not match")
    row_identity = rmsf_row_identity_sha256(keys, targets, groups)
    reported_row_identity = report.get("row_identity_sha256")
    if is_keyed and reported_row_identity != row_identity:
        raise ValueError("source RMSF row identity fingerprint mismatch")
    fold_identity = rmsf_fold_identity_sha256(folds, keys)
    reported_fold_identity = report.get("fold_identity_sha256")
    if is_keyed and reported_fold_identity != fold_identity:
        raise ValueError("source RMSF fold identity fingerprint mismatch")
    raw_folds = report.get("fold_ids")
    if raw_folds is not None and not np.array_equal(np.asarray(raw_folds, dtype=np.int16), folds):
        raise ValueError("source RMSF summary fold IDs do not match OOF rows")
    feature_counts = tuple(int(value) for value in probe_config["feature_counts"])
    reported_feature_counts = report.get("feature_counts")
    if (
        reported_feature_counts is not None
        and tuple(int(value) for value in cast(Sequence[object], reported_feature_counts))
        != feature_counts
    ):
        raise ValueError("source RMSF feature counts do not match probe configuration")
    if feature_counts != DEFAULT_RMSF_FEATURE_COUNTS:
        raise ValueError(
            "RMSF prior comparison requires feature counts " f"{list(DEFAULT_RMSF_FEATURE_COUNTS)}"
        )
    prediction_rows = tuple(
        {
            "row": row,
            **prediction_row_key_payload(keys[row]),
            "fold": int(folds[row]),
            "observed": float(targets[row]),
            "predicted": _optional_float(ordered[row].get("predicted_rmsf")),
        }
        for row in range(len(rows))
    )
    return RMSFSourceIdentity(
        row_keys=keys,
        targets=targets,
        groups=groups,
        fold_ids=folds,
        prediction_rows=prediction_rows,
        feature_counts=feature_counts,
        probe_config=probe_config,
        sampling_identity=sampling_identity,
        input_provenance=input_provenance,
        source_latent_width=source_width,
        row_identity_sha256=row_identity,
        fold_identity_sha256=fold_identity,
    )


def recover_legacy_rmsf_source_identity(
    source_report: Path,
    *,
    rmsf_by_sequence: Mapping[str, np.ndarray],
    source: str,
    sample_index: int,
    source_latent_width: int,
    eligible_rows_seen: int,
    input_provenance: Mapping[str, object],
) -> RMSFSourceIdentity:
    """Recover one exact modern identity without modifying a legacy RMSF report."""

    if not source:
        raise ValueError("legacy RMSF source must be nonempty")
    if sample_index < 0:
        raise ValueError("legacy RMSF sample_index must be non-negative")
    if eligible_rows_seen <= 0:
        raise ValueError("eligible_rows_seen must be a positive integer")
    if input_provenance.get("representation_route") != "esm3":
        raise ValueError("legacy RMSF recovery requires ESM3 input provenance")
    if input_provenance.get("prompt_source") != source:
        raise ValueError("legacy RMSF prompt source does not match recovered rows")
    report, report_dir = _load_source_report(source_report)
    artifacts = report.get("artifacts")
    raw_path = artifacts.get("oof_predictions_csv") if isinstance(artifacts, Mapping) else None
    prediction_path = _resolve_source_artifact(raw_path, report_dir)
    if prediction_path is None or not prediction_path.is_file():
        raise ValueError("source RMSF report is missing OOF predictions")
    with prediction_path.open(newline="") as handle:
        retained_rows = len(list(csv.DictReader(handle)))
    if retained_rows <= 0 or eligible_rows_seen < retained_rows:
        raise ValueError("eligible_rows_seen must cover every retained RMSF row")
    probe_config = {
        "feature_counts": list(DEFAULT_RMSF_FEATURE_COUNTS),
        "outer_folds": HISTORICAL_RMSF_OUTER_FOLDS,
        "inner_folds": HISTORICAL_RMSF_INNER_FOLDS,
        "alphas": list(DEFAULT_RIDGE_ALPHAS),
        "fold_seed": HISTORICAL_RMSF_FOLD_SEED,
    }
    sampling_identity = {
        "strategy": "reservoir_v1",
        "max_rows": HISTORICAL_RMSF_MAX_ROWS,
        "rng_seed": HISTORICAL_RMSF_RESERVOIR_SEED,
        "seen": eligible_rows_seen,
        "retained_rows": retained_rows,
    }
    return load_rmsf_source_identity(
        source_report,
        legacy_rmsf_by_sequence=rmsf_by_sequence,
        legacy_probe_config=probe_config,
        legacy_sampling_identity=sampling_identity,
        legacy_source_latent_width=source_latent_width,
        legacy_input_provenance=input_provenance,
        legacy_row_source=source,
        legacy_sample_index=sample_index,
    )


def write_recovered_rmsf_source(
    source_report: Path,
    identity: RMSFSourceIdentity,
    destination: Path,
) -> Path:
    """Write a relocation-safe modern copy while leaving legacy source bytes unchanged."""

    source_payload, _ = _load_source_report(source_report)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = destination.parent / "oof_predictions.csv"
    prediction_rows = [
        {
            "row": int(row["row"]),
            "sequence_id": str(row["sequence_id"]),
            "residue_index": int(row["residue_index"]),
            "sample_index": row["sample_index"],
            "source": str(row["source"]),
            "fold": int(row["fold"]),
            "true_rmsf": float(row["observed"]),
            "predicted_rmsf": row["predicted"],
        }
        for row in identity.prediction_rows
    ]
    _write_rows(prediction_path, prediction_rows)
    recovered = {
        **source_payload,
        "artifacts": {"oof_predictions_csv": prediction_path.name},
        "num_prediction_rows": len(identity.row_keys),
        "row_identity_sha256": identity.row_identity_sha256,
        "fold_identity_sha256": identity.fold_identity_sha256,
        "fold_ids": identity.fold_ids.astype(int).tolist(),
        "feature_counts": list(identity.feature_counts),
        "probe_config": dict(identity.probe_config),
        "sampling_identity": dict(identity.sampling_identity),
        "input_provenance": dict(identity.input_provenance),
        "source_latent_width": identity.source_latent_width,
    }
    _write_json(destination, recovered)
    return destination


def run_rmsf_prior_comparison(
    *,
    prior_by_row_key: Mapping[object, Mapping[str, np.ndarray]],
    source_report: Mapping[str, object] | Path,
    output_dir: Path,
    source_sample: Any | None = None,
    dry_run: bool = False,
    provenance: Mapping[str, object] | None = None,
    legacy_rmsf_by_sequence: Mapping[str, np.ndarray] | None = None,
    legacy_probe_config: Mapping[str, object] | None = None,
    legacy_sampling_identity: Mapping[str, object] | None = None,
    legacy_source_latent_width: int | None = None,
    legacy_input_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Fit RMSF probes on report-aligned priors without modifying source artifacts."""

    _validate_prior_comparison_output(output_dir, source_report)
    report, report_dir = _load_source_report(source_report)
    identity = load_rmsf_source_identity(
        source_report,
        legacy_rmsf_by_sequence=legacy_rmsf_by_sequence,
        legacy_probe_config=legacy_probe_config,
        legacy_sampling_identity=legacy_sampling_identity,
        legacy_source_latent_width=legacy_source_latent_width,
        legacy_input_provenance=legacy_input_provenance,
    )
    if source_sample is not None:
        _validate_source_sample(source_sample, identity)
    keys = identity.row_keys
    if set(prior_by_row_key) != set(keys):
        raise ValueError("prior representations do not exactly cover source RMSF row keys")
    representation_names = tuple(prior_by_row_key[keys[0]]) if keys else ()
    if not representation_names:
        raise ValueError("no RMSF prior representations were provided")
    if any(tuple(prior_by_row_key[key]) != representation_names for key in keys):
        raise ValueError("prior representation names differ across RMSF rows")
    matrices: dict[str, np.ndarray] = {}
    for name in representation_names:
        matrix = np.stack(
            [np.asarray(prior_by_row_key[key][name], dtype=np.float32) for key in keys]
        )
        if matrix.ndim != 2 or matrix.shape[0] != identity.targets.size:
            raise ValueError(f"{name} does not have one vector per RMSF row")
        expected_width = {
            "evoformer_single": 384,
            "evoformer_single_pair_mean": 512,
            "esm3_input": 1536,
        }.get(name)
        if expected_width is not None and matrix.shape[1] != expected_width:
            raise ValueError(f"{name} must have exactly {expected_width} features")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} contains non-finite RMSF prior values")
        matrices[name] = matrix

    validation_report: dict[str, object] = {
        "status": "validated",
        "dry_run": dry_run,
        "feature_counts": list(identity.feature_counts),
        "row_identity_sha256": identity.row_identity_sha256,
        "fold_identity_sha256": identity.fold_identity_sha256,
        "num_rows": int(identity.targets.size),
        "representations": {name: int(values.shape[1]) for name, values in matrices.items()},
        "probe_config": dict(identity.probe_config),
        "sampling_identity": dict(identity.sampling_identity),
        "input_provenance": dict(identity.input_provenance),
    }
    if dry_run:
        return validation_report

    source_report_path = _comparison_source_report_path(
        source_report,
        report,
        Path(output_dir),
    )
    representations = {
        name: {
            "width": int(values.shape[1]),
            "sha256": _array_sha256(values),
        }
        for name, values in sorted(matrices.items())
    }
    coverage = {
        "rows": int(identity.targets.size),
        "groups": len(set(str(value) for value in identity.groups)),
        "row_identity_sha256": identity.row_identity_sha256,
    }
    probe_configuration = {
        **dict(identity.probe_config),
        "feature_counts": list(identity.feature_counts),
        "fold_identity_sha256": identity.fold_identity_sha256,
        "sampling_identity": dict(identity.sampling_identity),
    }
    fingerprint_identity = {
        "representations": {name: dict(values) for name, values in representations.items()},
        "row_identity_sha256": identity.row_identity_sha256,
        "fold_identity_sha256": identity.fold_identity_sha256,
        "probe_configuration": probe_configuration,
        "input_provenance": dict(identity.input_provenance),
        "esm3_model_cache_identity": (
            dict(provenance["esm3_model_cache_identity"])
            if provenance is not None
            and isinstance(provenance.get("esm3_model_cache_identity"), Mapping)
            else None
        ),
    }
    preview = build_normalized_comparison_summary(
        status="ok",
        source_report=source_report_path,
        summary_path=Path(output_dir) / "summary.json",
        representations=representations,
        targets={"rmsf": {"status": "pending"}},
        coverage=coverage,
        probe_configuration=probe_configuration,
        fingerprint_identity=fingerprint_identity,
        artifacts=[],
    )
    if prepare_comparison_directory(
        Path(output_dir),
        expected_fingerprint=str(preview["fingerprint"]),
        expected_source_report=source_report_path,
        expected_representations=representations,
        expected_targets=("rmsf",),
        expected_artifacts=PROBE_COMPARISON_ARTIFACTS,
        representation_payload_fields={
            name: ("width", "sha256", "series") for name in representations
        },
        target_payload_fields={
            "rmsf": (
                "status",
                "series",
                "row_identity_sha256",
                "fold_identity_sha256",
            )
        },
        expected_target_values={
            "rmsf": {
                "row_identity_sha256": identity.row_identity_sha256,
                "fold_identity_sha256": identity.fold_identity_sha256,
            }
        },
        dry_run=False,
    ):
        return cast(dict[str, object], json.loads((Path(output_dir) / "summary.json").read_text()))

    series = _source_sae_series(
        identity=identity,
        source_report=report,
        report_dir=report_dir,
    )
    config = identity.probe_config
    for name, matrix in matrices.items():
        results = run_grouped_rmsf_probe_sweep(
            matrix,
            identity.targets,
            identity.groups,
            feature_counts=identity.feature_counts,
            outer_folds=int(config["outer_folds"]),
            inner_folds=int(config["inner_folds"]),
            alphas=tuple(float(value) for value in config["alphas"]),
            fold_seed=int(config["fold_seed"]),
        )
        for count, result in results.items():
            if not np.array_equal(result.fold_ids, identity.fold_ids):
                raise ValueError(f"{name} RMSF folds do not match source fold IDs")
            series.append(
                _comparison_series(
                    representation=name,
                    display_name=_prior_display_name(name),
                    width=matrix.shape[1],
                    feature_count=count,
                    result=result,
                    targets=identity.targets,
                    groups=identity.groups,
                    keys=keys,
                )
            )

    comparison_provenance = {
        "row_identity_sha256": identity.row_identity_sha256,
        "fold_identity_sha256": identity.fold_identity_sha256,
        "probe_config": dict(identity.probe_config),
        "sampling_identity": dict(identity.sampling_identity),
        "input_provenance": dict(identity.input_provenance),
        "source_report": _source_report_provenance(source_report),
        **dict(provenance or {}),
    }
    comparison_details = write_probe_comparison(
        output_dir=Path(output_dir),
        target="rmsf",
        series=series,
        feature_counts=identity.feature_counts,
        provenance=comparison_provenance,
    )
    prior_statuses = [item.status for item in series if item.representation in representation_names]
    status = (
        "ok"
        if "ok" in prior_statuses
        else "degenerate"
        if prior_statuses and all(item in {"degenerate", "skipped"} for item in prior_statuses)
        else "incomplete"
    )
    for name in representations:
        representations[name]["series"] = [
            item for item in comparison_details["series"] if item["representation"] == name
        ]
    summary = build_normalized_comparison_summary(
        status=status,
        source_report=source_report_path,
        summary_path=Path(output_dir) / "summary.json",
        representations=representations,
        targets={
            "rmsf": {
                "status": status,
                "series": comparison_details["series"],
                "row_identity_sha256": identity.row_identity_sha256,
                "fold_identity_sha256": identity.fold_identity_sha256,
            }
        },
        coverage=coverage,
        probe_configuration=probe_configuration,
        fingerprint_identity=fingerprint_identity,
        artifacts=comparison_details["artifacts"],
    )
    _atomic_json(Path(output_dir) / "summary.json", summary)
    return summary


def _comparison_source_report_path(
    source_report: Mapping[str, object] | Path,
    loaded_report: Mapping[str, object],
    output_dir: Path,
) -> Path:
    if not isinstance(source_report, Mapping):
        return Path(source_report)
    candidate = output_dir.parent / "summary.json"
    if not candidate.is_file():
        raise ValueError("writing an RMSF comparison requires a file-backed source report")
    candidate_report = json.loads(candidate.read_text())
    if candidate_report != dict(loaded_report):
        raise ValueError("file-backed RMSF source report disagrees with supplied report")
    return candidate


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_source_report(
    source_report: Mapping[str, object] | Path,
) -> tuple[dict[str, object], Path | None]:
    if isinstance(source_report, Mapping):
        return dict(source_report), None
    path = Path(source_report)
    return json.loads(path.read_text()), path.parent


def _source_sae_series(
    *,
    identity: RMSFSourceIdentity,
    source_report: Mapping[str, object],
    report_dir: Path | None,
) -> list[ProbeComparisonSeries]:
    raw_sweep = source_report.get("sparsity_sweep")
    if not isinstance(raw_sweep, list):
        path = _resolve_source_artifact("sparsity_sweep.json", report_dir)
        if path is None or not path.is_file():
            raise ValueError("source RMSF report is missing sparsity sweep rows")
        raw_sweep = json.loads(path.read_text())
    sweep_by_count = {int(row["requested_features"]): row for row in raw_sweep}
    if set(sweep_by_count) != set(identity.feature_counts):
        raise ValueError("source RMSF sweep does not match its feature counts")
    series: list[ProbeComparisonSeries] = []
    for count in identity.feature_counts:
        summary = dict(sweep_by_count[count])
        status = str(summary.pop("status", "skipped"))
        reason_value = summary.pop("reason", None)
        summary.pop("requested_features", None)
        series.append(
            ProbeComparisonSeries(
                representation="sae",
                display_name="SAE latents",
                width=identity.source_latent_width,
                feature_count=count,
                status=status,
                reason=str(reason_value) if reason_value is not None else None,
                metrics=summary,
                prediction_rows=(
                    identity.prediction_rows if count == max(identity.feature_counts) else ()
                ),
            )
        )
    return series


def _resolve_source_artifact(raw_path: object, report_dir: Path | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    candidates: list[Path] = [path]
    if report_dir is not None:
        if not path.is_absolute():
            candidates.append(report_dir / path)
        candidates.append(report_dir / path.name)
    existing: list[Path] = []
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve() not in {item.resolve() for item in existing}:
            existing.append(candidate)
    if not existing:
        return candidates[-1]
    hashes = {hashlib.sha256(candidate.read_bytes()).hexdigest() for candidate in existing}
    if len(hashes) > 1:
        raise ValueError(f"ambiguous source artifact resolution for {path.name}")
    if report_dir is not None:
        local = report_dir / path.name
        if local.is_file():
            return local
    return existing[0]


def _validate_source_sample(source_sample: Any, identity: RMSFSourceIdentity) -> None:
    targets = np.asarray(source_sample.property_values["rmsf"], dtype=np.float64)
    groups = source_sample.groups_for("rmsf")
    if groups is None:
        raise ValueError("source RMSF sample has no protein group IDs")
    raw_keys = tuple(source_sample.keys_for("rmsf"))
    if any(key is None for key in raw_keys):
        raise ValueError("source RMSF sample has incomplete row identity")
    keys = cast(tuple[_PredictionRowKeyLike, ...], raw_keys)
    if keys != identity.row_keys:
        raise ValueError("source RMSF sample row keys do not match source report")
    if not np.array_equal(targets, identity.targets) or not np.array_equal(groups, identity.groups):
        raise ValueError("source RMSF sample targets/groups do not match source report")
    latent_width = int(np.asarray(source_sample.latents_for("rmsf")).shape[1])
    if latent_width != identity.source_latent_width:
        raise ValueError("source RMSF sample latent width does not match source report")
    sampling = _normalize_sampling_identity(
        source_sample.sampling_for("rmsf"),
        expected_rows=len(keys),
    )
    if sampling != identity.sampling_identity:
        raise ValueError("source RMSF sampling identity does not match source report")


def _validate_prior_comparison_output(
    output_dir: Path,
    source_report: Mapping[str, object] | Path,
) -> None:
    destination = Path(output_dir)
    if destination.name != "prior_comparison" or destination.parent.name != "rmsf_probe":
        raise ValueError("RMSF comparison output must end in rmsf_probe/prior_comparison")
    if not isinstance(source_report, Mapping):
        source_dir = Path(source_report).parent.resolve()
        if destination.resolve() == source_dir:
            raise ValueError("RMSF prior_comparison output cannot replace source artifacts")


def _normalize_probe_config(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "legacy RMSF report lacks exact probe configuration; "
            "provide validated settings or regenerate the report"
        )
    counts = tuple(int(item) for item in value.get("feature_counts", ()))
    outer = _positive_int(value.get("outer_folds"), "outer_folds")
    inner = _positive_int(value.get("inner_folds"), "inner_folds")
    if outer < 2 or inner < 2:
        raise ValueError("RMSF outer_folds and inner_folds must be at least 2")
    raw_alphas = value.get("alphas")
    if not isinstance(raw_alphas, list | tuple) or not raw_alphas:
        raise ValueError("RMSF probe configuration must include alpha candidates")
    alphas = tuple(float(item) for item in raw_alphas)
    if any(not math.isfinite(item) or item < 0 for item in alphas):
        raise ValueError("RMSF alpha candidates must be finite and non-negative")
    fold_seed = int(value.get("fold_seed"))
    return {
        "feature_counts": list(counts),
        "outer_folds": outer,
        "inner_folds": inner,
        "alphas": list(alphas),
        "fold_seed": fold_seed,
    }


def _require_historical_legacy_probe_config(config: Mapping[str, object]) -> None:
    expected = {
        "feature_counts": list(DEFAULT_RMSF_FEATURE_COUNTS),
        "outer_folds": HISTORICAL_RMSF_OUTER_FOLDS,
        "inner_folds": HISTORICAL_RMSF_INNER_FOLDS,
        "alphas": list(DEFAULT_RIDGE_ALPHAS),
        "fold_seed": HISTORICAL_RMSF_FOLD_SEED,
    }
    if dict(config) != expected:
        raise ValueError(
            "legacy report does not contain the exact historical RMSF probe configuration; "
            "regenerate or migrate the source report"
        )


def _require_historical_legacy_sampling(sampling: Mapping[str, object]) -> None:
    if (
        int(sampling["max_rows"]) != HISTORICAL_RMSF_MAX_ROWS
        or int(sampling["rng_seed"]) != HISTORICAL_RMSF_RESERVOIR_SEED
    ):
        raise ValueError(
            "legacy report does not contain the exact historical RMSF sampling configuration; "
            "regenerate or migrate the source report"
        )


def _require_historical_legacy_folds(
    folds: np.ndarray,
    groups: np.ndarray,
    targets: np.ndarray,
) -> None:
    expected = np.full(folds.shape, -1, dtype=np.int16)
    finite = np.isfinite(targets)
    finite_groups = groups[finite]
    finite_targets = targets[finite]
    if (
        finite_targets.size > 0
        and np.unique(finite_groups.astype(str)).size >= 2
        and float(np.std(finite_targets)) > 1e-12
    ):
        expected[finite] = deterministic_group_folds(
            finite_groups,
            HISTORICAL_RMSF_OUTER_FOLDS,
            fold_seed=HISTORICAL_RMSF_FOLD_SEED,
        )
    if not np.array_equal(folds, expected):
        raise ValueError(
            "legacy report fold IDs do not match the exact historical RMSF configuration; "
            "regenerate or migrate the source report"
        )


def _normalize_input_provenance(
    value: object,
    *,
    expected_sequence_ids: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "source RMSF report lacks bound input provenance; regenerate or migrate "
            "the source RMSF report with input provenance"
        )
    route = str(value.get("representation_route", ""))
    if route not in {"bioemu", "esm3"}:
        raise ValueError("source RMSF input provenance has an invalid representation route")
    schema_version = int(value.get("schema_version", -1))
    if schema_version != 1:
        raise ValueError("unsupported source RMSF input provenance schema")
    model_family = _nonempty_provenance_string(value.get("model_family"), "model_family")
    prompt_mode = _nonempty_provenance_string(value.get("prompt_mode"), "prompt_mode")
    prompt_source = _nonempty_provenance_string(value.get("prompt_source"), "prompt_source")
    config_sha256 = _sha256_value(
        value.get("representation_config_sha256"),
        "representation_config_sha256",
    )
    raw_sequences = value.get("sequence_sha256")
    if not isinstance(raw_sequences, Mapping):
        raise ValueError("source RMSF input provenance lacks sequence-content hashes")
    sequences = {
        str(key): _sha256_value(item, f"sequence_sha256[{key}]")
        for key, item in raw_sequences.items()
    }
    if not expected_sequence_ids <= set(sequences):
        raise ValueError("source RMSF sequence-content provenance has incorrect coverage")
    evaluation_fasta = _normalize_file_identity(
        value.get("evaluation_fasta"),
        "evaluation_fasta",
    )
    cache_manifest = _normalize_optional_file_identity(
        value.get("cache_manifest"),
        "cache_manifest",
    )
    structure_index = _normalize_optional_file_identity(
        value.get("structure_index"),
        "structure_index",
    )
    if route == "bioemu" and cache_manifest is None:
        raise ValueError("BioEmu RMSF input provenance lacks a cache manifest fingerprint")
    if route == "esm3" and prompt_mode != "sequence_to_structure_function":
        if structure_index is None:
            raise ValueError(
                "structure-conditioned ESM3 RMSF input provenance lacks "
                "a structure-index fingerprint"
            )
    return {
        "schema_version": schema_version,
        "representation_route": route,
        "model_family": model_family,
        "prompt_mode": prompt_mode,
        "prompt_source": prompt_source,
        "representation_config_sha256": config_sha256,
        "sequence_sha256": dict(sorted(sequences.items())),
        "evaluation_fasta": evaluation_fasta,
        "cache_manifest": cache_manifest,
        "structure_index": structure_index,
    }


def _nonempty_provenance_string(value: object, field: str) -> str:
    normalized = str(value or "")
    if not normalized:
        raise ValueError(f"source RMSF input provenance lacks {field}")
    return normalized


def _sha256_value(value: object, field: str) -> str:
    normalized = str(value or "")
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"source RMSF input provenance has invalid {field}")
    return normalized


def _normalize_file_identity(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"source RMSF input provenance lacks {field} fingerprint")
    size_bytes = int(value.get("size_bytes", -1))
    if size_bytes < 0:
        raise ValueError(f"source RMSF input provenance has invalid {field} size")
    return {
        "sha256": _sha256_value(value.get("sha256"), f"{field}.sha256"),
        "size_bytes": size_bytes,
    }


def _normalize_optional_file_identity(
    value: object,
    field: str,
) -> dict[str, object] | None:
    if value is None:
        return None
    return _normalize_file_identity(value, field)


def _normalize_sampling_identity(value: object, *, expected_rows: int) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "legacy RMSF report lacks exact sampling identity; "
            "provide validated settings or regenerate the report"
        )
    strategy = str(value.get("strategy", ""))
    if not strategy:
        raise ValueError("RMSF sampling identity must include strategy")
    max_rows = _positive_int(value.get("max_rows"), "max_rows")
    seen = _positive_int(value.get("seen"), "seen")
    retained = _positive_int(value.get("retained_rows"), "retained_rows")
    if retained != expected_rows or max_rows < retained or seen < retained:
        raise ValueError("RMSF sampling identity does not match retained rows")
    seed_value = value.get("rng_seed")
    rng_seed = None if seed_value is None else int(seed_value)
    return {
        "strategy": strategy,
        "max_rows": max_rows,
        "rng_seed": rng_seed,
        "seen": seen,
        "retained_rows": retained,
    }


def _legacy_row_keys(
    groups: np.ndarray,
    targets: np.ndarray,
    rmsf_by_sequence: Mapping[str, np.ndarray],
    *,
    source: str = "legacy_rmsf_value_match",
    sample_index: int | None = None,
) -> tuple[PredictionRowKey, ...]:
    keys: list[PredictionRowKey] = []
    for group, target in zip(groups, targets, strict=True):
        reference = rmsf_by_sequence.get(str(group))
        matches = (
            np.flatnonzero(np.asarray(reference, dtype=np.float64) == float(target))
            if reference is not None
            else np.empty(0, dtype=int)
        )
        if matches.size != 1:
            raise ValueError("ambiguous legacy RMSF row identity")
        keys.append(
            PredictionRowKey(
                sequence_id=str(group),
                residue_index=int(matches[0]),
                sample_index=sample_index,
                source=source,
            )
        )
    if len(set(keys)) != len(keys):
        raise ValueError("ambiguous legacy RMSF row identity")
    return tuple(keys)


def _optional_int(value: object) -> int | None:
    return None if value in (None, "", "None") else int(str(value))


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    number = float(str(value))
    return number if math.isfinite(number) else None


def _positive_int(value: object, name: str) -> int:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _stable_sha256(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _comparison_series(
    *,
    representation: str,
    display_name: str,
    width: int,
    feature_count: int,
    result: RMSFProbeResult,
    targets: np.ndarray,
    groups: np.ndarray,
    keys: tuple[_PredictionRowKeyLike, ...],
) -> ProbeComparisonSeries:
    summary = dict(result.summary)
    status = str(summary.pop("status", "skipped"))
    reason_value = summary.pop("reason", None)
    scored = result.valid_mask & np.isfinite(result.oof_predictions)
    prediction_rows = tuple(
        {
            "row": int(row),
            "sequence_id": str(groups[row]),
            "residue_index": keys[row].residue_index,
            "sample_index": keys[row].sample_index,
            "source": keys[row].source,
            "fold": int(result.fold_ids[row]),
            "observed": float(targets[row]),
            "predicted": float(result.oof_predictions[row]),
        }
        for row in np.flatnonzero(scored)
    )
    return ProbeComparisonSeries(
        representation=representation,
        display_name=display_name,
        width=int(width),
        feature_count=feature_count,
        status=status,
        reason=str(reason_value) if reason_value is not None else None,
        metrics=summary,
        prediction_rows=prediction_rows,
    )


def _prior_display_name(name: str) -> str:
    return {
        "evoformer_single": "Evoformer single (diagnostic)",
        "evoformer_single_pair_mean": "Evoformer single + pair mean (primary)",
        "esm3_input": "ESM3 pre-transformer input",
    }.get(name, name)


def _source_report_provenance(source_report: Mapping[str, object] | Path) -> object:
    if isinstance(source_report, Mapping):
        canonical = json.dumps(source_report, sort_keys=True, separators=(",", ":"), default=str)
        return {"sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}
    path = Path(source_report)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_rmsf_json_atomic(path: Path, payload: Any) -> None:
    """Atomically replace one RMSF JSON metadata artifact."""

    _atomic_json(path, payload)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _skipped_result(
    predictions: np.ndarray,
    fold_ids: np.ndarray,
    valid_mask: np.ndarray,
    reason: str,
) -> RMSFProbeResult:
    return RMSFProbeResult(
        models=(),
        oof_predictions=predictions,
        fold_ids=fold_ids,
        valid_mask=valid_mask,
        fold_metrics=(),
        sequence_metrics=(),
        summary={"status": "skipped", "reason": reason},
    )


def _select_features(x: np.ndarray, y: np.ndarray, limit: int) -> np.ndarray:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    x_centered = x_values - x_values.mean(axis=0, keepdims=True)
    y_centered = y_values - y_values.mean()
    denominator = np.linalg.norm(x_centered, axis=0) * np.linalg.norm(y_centered)
    correlations = np.divide(
        x_centered.T @ y_centered,
        denominator,
        out=np.zeros(x_values.shape[1], dtype=np.float64),
        where=denominator > 0,
    )
    variable = np.flatnonzero(np.std(x_values, axis=0) > 1e-12)
    ranked = variable[np.argsort(np.abs(correlations[variable]))[::-1]]
    return ranked[: min(limit, ranked.size)]


def _fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray,
    alpha: float,
    fold: int,
    training_groups: tuple[str, ...],
    heldout_groups: tuple[str, ...],
) -> RidgeProbeModel:
    selected = np.asarray(x)[:, feature_indices].astype(np.float64, copy=False)
    means = selected.mean(axis=0)
    scales = selected.std(axis=0)
    scales[scales <= 1e-12] = 1.0
    standardized = (selected - means) / scales
    intercept = float(np.mean(y))
    centered_y = np.asarray(y, dtype=np.float64) - intercept
    gram = standardized.T @ standardized
    gram.flat[:: gram.shape[0] + 1] += alpha
    weights = np.linalg.solve(gram, standardized.T @ centered_y)
    return RidgeProbeModel(
        fold=fold,
        feature_indices=tuple(int(value) for value in feature_indices),
        feature_means=means,
        feature_scales=scales,
        weights=weights,
        intercept=intercept,
        alpha=float(alpha),
        training_groups=training_groups,
        heldout_groups=heldout_groups,
    )


def _select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_indices: np.ndarray,
    alphas: tuple[float, ...],
    inner_folds: int,
    fold_seed: int,
) -> float:
    assignments = deterministic_group_folds(
        groups,
        inner_folds,
        fold_seed=fold_seed,
    )
    losses: list[tuple[float, float]] = []
    for alpha in alphas:
        fold_losses = []
        for fold in sorted(int(value) for value in np.unique(assignments) if value >= 0):
            train = assignments != fold
            test = assignments == fold
            if int(train.sum()) < 2 or not bool(test.any()):
                continue
            model = _fit_ridge(
                x[train],
                y[train],
                feature_indices,
                alpha,
                fold,
                tuple(sorted({str(value) for value in groups[train]})),
                tuple(sorted({str(value) for value in groups[test]})),
            )
            residual = y[test] - predict_ridge_model(model, x[test])
            fold_losses.append(float(np.mean(residual * residual)))
        if fold_losses:
            losses.append((float(np.mean(fold_losses)), float(alpha)))
    return min(losses)[1] if losses else float(alphas[0])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true_values = np.asarray(y_true, dtype=np.float64)
    predicted_values = np.asarray(y_pred, dtype=np.float64)
    residual = true_values - predicted_values
    denominator = float(((true_values - true_values.mean()) ** 2).sum())
    if true_values.size >= 2 and np.std(true_values) > 0 and np.std(predicted_values) > 0:
        spearman = float(scipy_stats.spearmanr(true_values, predicted_values).statistic)
    else:
        spearman = float("nan")
    return {
        "spearman": spearman,
        "r2": float(1.0 - (residual @ residual) / denominator) if denominator > 0 else float("nan"),
        "mae": float(np.mean(np.abs(residual))),
    }


def _sequence_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    scored_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for group in sorted({str(value) for value in groups[scored_mask]}):
        mask = scored_mask & (groups.astype(str) == group)
        rows.append(
            {
                "sequence_id": group,
                "n_rows": int(mask.sum()),
                **_metrics(targets[mask], predictions[mask]),
            }
        )
    return rows


def _finite_mean(values: Any) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def write_rmsf_probe_artifacts(
    result: RMSFProbeResult,
    rmsf: np.ndarray,
    group_ids: np.ndarray,
    output_dir: Path,
    *,
    row_keys: Sequence[PredictionRowKey],
    probe_config: Mapping[str, object],
    sampling_identity: Mapping[str, object],
    source_latent_width: int,
    input_provenance: Mapping[str, object],
    feature_counts: tuple[int, ...] | None = None,
    sparsity_sweep: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = np.asarray(rmsf, dtype=np.float64)
    groups = np.asarray(group_ids, dtype=object)
    keys = tuple(row_keys)
    if len(keys) != targets.size or groups.shape != targets.shape:
        raise ValueError("RMSF source rows require one durable key per target")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate RMSF source row key")
    if any(key.sequence_id != str(group) for key, group in zip(keys, groups, strict=True)):
        raise ValueError("RMSF source row key and protein group do not match")
    normalized_probe_config = _normalize_probe_config(probe_config)
    normalized_sampling = _normalize_sampling_identity(
        sampling_identity,
        expected_rows=len(keys),
    )
    normalized_input_provenance = _normalize_input_provenance(
        input_provenance,
        expected_sequence_ids={str(group) for group in groups},
    )
    report = dict(result.summary)
    report["output_dir"] = str(output_dir)
    report["fold_ids"] = result.fold_ids.astype(int).tolist()
    report["row_identity_sha256"] = rmsf_row_identity_sha256(keys, targets, groups)
    report["fold_identity_sha256"] = rmsf_fold_identity_sha256(result.fold_ids, keys)
    report["num_prediction_rows"] = len(keys)
    report["probe_config"] = normalized_probe_config
    report["sampling_identity"] = normalized_sampling
    report["input_provenance"] = normalized_input_provenance
    report["source_latent_width"] = _positive_int(
        source_latent_width,
        "source_latent_width",
    )
    if feature_counts is not None:
        report["feature_counts"] = list(feature_counts)
    if sparsity_sweep is not None:
        report["sparsity_sweep"] = sparsity_sweep
    prediction_rows = [
        {
            "row": row,
            **prediction_row_key_payload(key),
            "fold": int(result.fold_ids[row]),
            "true_rmsf": float(targets[row]),
            "predicted_rmsf": float(result.oof_predictions[row]),
            "scored": bool(result.valid_mask[row] and np.isfinite(result.oof_predictions[row])),
        }
        for row, key in enumerate(keys)
    ]
    prediction_path = output_dir / "oof_predictions.csv"
    _write_rows(prediction_path, prediction_rows)
    report["artifacts"] = {"oof_predictions_csv": str(prediction_path)}
    if result.summary.get("status") != "ok":
        _write_json(output_dir / "summary.json", report)
        return report

    fold_rows = [dict(row) for row in result.fold_metrics]
    sequence_rows = [dict(row) for row in result.sequence_metrics]
    _write_rows(output_dir / "fold_metrics.csv", fold_rows)
    _write_json(output_dir / "fold_metrics.json", fold_rows)
    _write_rows(output_dir / "sequence_metrics.csv", sequence_rows)
    _write_json(output_dir / "sequence_metrics.json", sequence_rows)

    scored = result.valid_mask & np.isfinite(result.oof_predictions)
    model_rows = [_model_payload(model) for model in result.models]
    _write_json(output_dir / "fold_models.json", model_rows)

    observed_path = output_dir / "observed_vs_predicted.png"
    _write_observed_plot(targets[scored], result.oof_predictions[scored], observed_path)
    fold_plot_path = output_dir / "metrics_by_fold.png"
    _write_fold_plot(fold_rows, fold_plot_path)
    stability_path = output_dir / "selected_feature_stability.png"
    _write_stability_plot(result.models, stability_path)
    report["artifacts"] = {
        "fold_metrics_csv": str(output_dir / "fold_metrics.csv"),
        "sequence_metrics_csv": str(output_dir / "sequence_metrics.csv"),
        "oof_predictions_csv": str(prediction_path),
        "fold_models_json": str(output_dir / "fold_models.json"),
        "observed_vs_predicted_plot": str(observed_path),
        "metrics_by_fold_plot": str(fold_plot_path),
        "selected_feature_stability_plot": str(stability_path),
    }
    _write_json(output_dir / "summary.json", report)
    return report


def write_rmsf_probe_sweep_artifacts(
    results: dict[int, RMSFProbeResult],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [
        {"requested_features": requested_features, **result.summary}
        for requested_features, result in results.items()
    ]
    fold_rows = []
    for requested_features, result in results.items():
        models_by_fold = {model.fold: model for model in result.models}
        for row in result.fold_metrics:
            fold = int(row["fold"])
            model = models_by_fold.get(fold)
            fold_rows.append(
                {
                    "requested_features": requested_features,
                    **row,
                    "feature_indices": list(model.feature_indices) if model is not None else [],
                }
            )
    _write_rows(output_dir / "sparsity_sweep.csv", summary_rows)
    _write_json(output_dir / "sparsity_sweep.json", summary_rows)
    _write_rows(output_dir / "sparsity_fold_metrics.csv", fold_rows)
    _write_json(output_dir / "sparsity_fold_metrics.json", fold_rows)
    plot_path = output_dir / "sparsity_sweep.png"
    _write_sparsity_sweep_plot(summary_rows, plot_path)
    return {
        "summary": summary_rows,
        "fold_metrics": fold_rows,
        "artifacts": {
            "summary_csv": str(output_dir / "sparsity_sweep.csv"),
            "summary_json": str(output_dir / "sparsity_sweep.json"),
            "fold_metrics_csv": str(output_dir / "sparsity_fold_metrics.csv"),
            "fold_metrics_json": str(output_dir / "sparsity_fold_metrics.json"),
            "plot": str(plot_path),
        },
    }


def write_residue_value_pdb(
    input_pdb_path: Path,
    output_pdb_path: Path,
    normalized_values: np.ndarray,
) -> None:
    values = np.asarray(normalized_values, dtype=np.float64)
    output_pdb_path.parent.mkdir(parents=True, exist_ok=True)
    with input_pdb_path.open() as source, output_pdb_path.open("w") as target:
        for line in source:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 66:
                try:
                    residue_index = int(line[22:26].strip()) - 1
                except ValueError:
                    target.write(line)
                    continue
                value = values[residue_index] if 0 <= residue_index < len(values) else 0.0
                bfactor = 100.0 * float(np.clip(value, 0.0, 1.0))
                target.write(f"{line[:54]}{1.00:6.2f}{bfactor:6.2f}{line[66:]}")
            else:
                target.write(line)


def write_rmsf_structure_artifacts(
    source_pdb_path: Path,
    sequence_id: str,
    true_rmsf: np.ndarray,
    predicted_rmsf: np.ndarray,
    output_dir: Path,
    render: Callable[[Path, Path], dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    true_values = np.asarray(true_rmsf, dtype=np.float64)
    predicted_values = np.asarray(predicted_rmsf, dtype=np.float64)
    length = min(true_values.size, predicted_values.size)
    true_values = true_values[:length]
    predicted_values = predicted_values[:length]
    finite = np.isfinite(true_values) & np.isfinite(predicted_values)
    if not bool(finite.any()):
        return {
            "sequence_id": sequence_id,
            "status": "skipped",
            "reason": "no finite residue predictions",
        }

    combined = np.concatenate([true_values[finite], predicted_values[finite]])
    scale_low, scale_high = (float(value) for value in np.quantile(combined, [0.02, 0.98]))
    if scale_high <= scale_low:
        scale_high = scale_low + 1.0
    true_normalized = _normalize_range(true_values, scale_low, scale_high)
    predicted_normalized = _normalize_range(predicted_values, scale_low, scale_high)
    errors = predicted_values - true_values
    error_limit = scale_high
    if error_limit <= 0:
        error_limit = 1.0
    error_normalized = np.clip(0.5 + 0.5 * errors / error_limit, 0.0, 1.0)
    error_normalized[~finite] = np.nan

    rows = [
        {
            "sequence_id": sequence_id,
            "residue_index": residue_idx,
            "true_rmsf": float(true_values[residue_idx]),
            "predicted_rmsf": float(predicted_values[residue_idx]),
            "prediction_error": float(errors[residue_idx]),
        }
        for residue_idx in np.flatnonzero(finite)
    ]
    _write_rows(output_dir / "residue_predictions.csv", rows)

    render_reports: dict[str, dict[str, Any]] = {}
    image_paths = []
    for name, normalized in (
        ("true_rmsf", true_normalized),
        ("predicted_rmsf", predicted_normalized),
        ("prediction_error", error_normalized),
    ):
        pdb_path = output_dir / f"{name}.pdb"
        png_path = output_dir / f"{name}.png"
        write_residue_value_pdb(source_pdb_path, pdb_path, normalized)
        render_reports[name] = render(pdb_path, png_path)
        image_paths.append(png_path)

    profile_path = output_dir / "rmsf_profile.png"
    _write_rmsf_profile(true_values, predicted_values, profile_path, sequence_id)
    grid_path = output_dir / "true_predicted_error_grid.png"
    _write_structure_grid(
        image_paths,
        grid_path,
        sequence_id,
        scale_low,
        scale_high,
        error_limit,
    )
    return {
        "sequence_id": sequence_id,
        "status": "ok",
        "num_finite_residues": int(finite.sum()),
        "rmsf_scale_min": scale_low,
        "rmsf_scale_max": scale_high,
        "error_scale_limit": error_limit,
        "residue_predictions_csv": str(output_dir / "residue_predictions.csv"),
        "profile_plot": str(profile_path),
        "structure_grid_plot": str(grid_path),
        "renderers": render_reports,
    }


def _model_payload(model: RidgeProbeModel) -> dict[str, Any]:
    return {
        "fold": model.fold,
        "feature_indices": list(model.feature_indices),
        "feature_means": model.feature_means.tolist(),
        "feature_scales": model.feature_scales.tolist(),
        "weights": model.weights.tolist(),
        "intercept": model.intercept,
        "alpha": model.alpha,
        "training_groups": list(model.training_groups),
        "heldout_groups": list(model.heldout_groups),
    }


def _write_json(path: Path, payload: Any) -> None:
    _atomic_json(path, payload)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            newline="",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            if rows:
                fields = list(dict.fromkeys(key for row in rows for key in row))
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_observed_plot(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(y_true, y_pred, s=10, alpha=0.45, edgecolors="none")
    bounds = [float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))]
    ax.plot(bounds, bounds, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlabel("true RMSF (angstrom)")
    ax.set_ylabel("out-of-fold predicted RMSF (angstrom)")
    ax.set_title("RMSF ridge probe: held-out predictions")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_fold_plot(rows: list[dict[str, Any]], path: Path) -> None:
    scored = [row for row in rows if row.get("status") == "ok"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    positions = np.arange(len(scored))
    ax.bar(positions - 0.18, [float(row["spearman"]) for row in scored], 0.36, label="Spearman")
    ax.bar(positions + 0.18, [float(row["r2"]) for row in scored], 0.36, label="R2")
    ax.set_xticks(positions, [str(row["fold"]) for row in scored])
    ax.set_xlabel("held-out protein fold")
    ax.set_ylabel("score")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_sparsity_sweep_plot(rows: list[dict[str, Any]], path: Path) -> None:
    scored = [row for row in rows if row.get("status") == "ok"]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    counts = [int(row["requested_features"]) for row in scored]
    ax.plot(
        counts,
        [float(row["oof_spearman"]) for row in scored],
        marker="o",
        label="OOF Spearman",
    )
    ax.plot(
        counts,
        [float(row["oof_r2"]) for row in scored],
        marker="o",
        label="OOF R2",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts, [str(count) for count in counts])
    ax.set_xlabel("selected SAE features per fold")
    ax.set_ylabel("held-out score")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.legend()
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_stability_plot(models: tuple[RidgeProbeModel, ...], path: Path) -> None:
    counts = Counter(feature for model in models for feature in model.feature_indices)
    selected = counts.most_common(20)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar([str(feature) for feature, _ in selected], [count for _, count in selected])
    ax.set_xlabel("SAE feature")
    ax.set_ylabel("outer folds selected")
    ax.set_title("RMSF probe feature stability")
    ax.tick_params(axis="x", labelrotation=60)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _normalize_range(values: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = (values - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~np.isfinite(values)] = np.nan
    return normalized


def _write_rmsf_profile(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    path: Path,
    sequence_id: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    ax.plot(true_values, color="#333333", linewidth=1.4, label="true RMSF")
    ax.plot(predicted_values, color="#b23b3b", linewidth=1.2, label="OOF predicted RMSF")
    ax.set_xlabel("residue index")
    ax.set_ylabel("RMSF (angstrom)")
    ax.set_title(sequence_id)
    ax.legend()
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_structure_grid(
    image_paths: list[Path],
    path: Path,
    sequence_id: str,
    scale_low: float,
    scale_high: float,
    error_limit: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.3), squeeze=False)
    titles = ("true RMSF", "OOF predicted RMSF", "prediction error")
    for index, (ax, image_path, title) in enumerate(
        zip(axes.flat, image_paths, titles, strict=True)
    ):
        if image_path.is_file():
            ax.imshow(plt.imread(image_path))
        ax.axis("off")
        ax.set_title(title)
        if index < 2:
            normalizer = mpl_colors.Normalize(vmin=scale_low, vmax=scale_high)
        else:
            normalizer = mpl_colors.Normalize(vmin=-error_limit, vmax=error_limit)
        scalar = mpl_cm.ScalarMappable(norm=normalizer, cmap="coolwarm")
        fig.colorbar(scalar, ax=ax, orientation="horizontal", fraction=0.045, pad=0.03)
    fig.suptitle(sequence_id)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
