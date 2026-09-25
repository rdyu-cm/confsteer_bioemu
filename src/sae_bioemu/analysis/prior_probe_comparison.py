"""Normalized, regenerable artifacts for comparisons against probe priors."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt

PLOT_DPI = 300
PROBE_COMPARISON_ARTIFACTS = (
    "observed_vs_predicted_with_prior.png",
    "observed_vs_predicted_with_prior.csv",
    "feature_count_sweep_with_prior.png",
    "feature_count_sweep_with_prior.csv",
)
_VALID_STATUSES = frozenset({"ok", "incomplete", "degenerate", "skipped"})
_NORMALIZED_SUMMARY_STATUSES = frozenset({"ok", "incomplete", "degenerate"})
_NORMALIZED_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "source_report",
        "representations",
        "targets",
        "coverage",
        "probe_configuration",
        "fingerprint",
        "artifacts",
    }
)
_RESERVED_SIDECAR_COLUMNS = frozenset(
    {
        "target",
        "representation",
        "display_name",
        "width",
        "feature_count",
        "status",
        "reason",
        "row_kind",
        "metrics_json",
    }
)


@dataclass(frozen=True)
class ProbeComparisonSeries:
    """One representation's probe result at an optional SAE feature count."""

    representation: str
    display_name: str
    width: int
    feature_count: int | None
    status: str
    reason: str | None
    metrics: Mapping[str, float | int | str | None]
    prediction_rows: tuple[Mapping[str, object], ...] = ()


def comparison_fingerprint(payload: Mapping[str, object]) -> str:
    """Return a stable SHA-256 digest for a normalized comparison payload."""
    canonical = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_normalized_comparison_summary(
    *,
    status: str,
    source_report: Path,
    summary_path: Path,
    representations: Mapping[str, object],
    targets: Mapping[str, object],
    coverage: Mapping[str, object],
    probe_configuration: Mapping[str, object],
    fingerprint_identity: Mapping[str, object],
    artifacts: Sequence[str],
) -> dict[str, object]:
    """Build the common relocation-safe schema for a post-hoc comparison."""

    if status not in _NORMALIZED_SUMMARY_STATUSES:
        raise ValueError(
            f"normalized comparison status must be one of "
            f"{sorted(_NORMALIZED_SUMMARY_STATUSES)}"
        )
    source = Path(source_report).resolve()
    destination = Path(summary_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing comparison source report: {source}")
    if _contains_absolute_path(fingerprint_identity):
        raise ValueError("comparison fingerprint identity must not contain absolute paths")
    semantic_source_path = Path(os.path.relpath(source, destination.parent)).as_posix()
    source_identity = {
        "path": semantic_source_path,
        "sha256": _sha256_file(source),
    }
    fingerprint = comparison_fingerprint(
        {
            "schema_version": 1,
            "source_report": source_identity,
            "identity": fingerprint_identity,
        }
    )
    return {
        "schema_version": 1,
        "status": status,
        "source_report": source_identity,
        "representations": _json_safe(representations),
        "targets": _json_safe(targets),
        "coverage": _json_safe(coverage),
        "probe_configuration": _json_safe(probe_configuration),
        "fingerprint": fingerprint,
        "artifacts": [
            _artifact_identity(
                summary_path=destination,
                comparison_root=_comparison_root(destination),
                artifact_path=str(value),
            )
            for value in artifacts
        ],
    }


def comparison_summary_is_current(
    path: Path,
    expected_fingerprint: object,
    *,
    expected_source_report: Path,
    comparison_root: Path,
    expected_representations: Mapping[str, object],
    expected_targets: Sequence[str],
    expected_artifacts: Sequence[str],
    representation_payload_fields: Mapping[str, Sequence[str]] | None = None,
    target_payload_fields: Mapping[str, Sequence[str]] | None = None,
    expected_target_values: Mapping[str, Mapping[str, object]] | None = None,
) -> bool:
    """Return whether an existing summary exactly matches the normalized contract."""

    if not isinstance(expected_fingerprint, str) or len(expected_fingerprint) != 64:
        return False
    summary_path = Path(path).resolve()
    root = Path(comparison_root).resolve()
    source = Path(expected_source_report).resolve()
    try:
        payload = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return _normalized_summary_is_current(
        payload,
        summary_path=summary_path,
        expected_fingerprint=expected_fingerprint,
        expected_source_report=source,
        comparison_root=root,
        expected_representations=expected_representations,
        expected_targets=expected_targets,
        expected_artifacts=expected_artifacts,
        representation_payload_fields=representation_payload_fields,
        target_payload_fields=target_payload_fields,
        expected_target_values=expected_target_values,
        visited=set(),
    )


def prepare_comparison_directory(
    output_dir: Path,
    *,
    expected_fingerprint: str,
    expected_source_report: Path,
    expected_representations: Mapping[str, object],
    expected_targets: Sequence[str],
    expected_artifacts: Sequence[str],
    representation_payload_fields: Mapping[str, Sequence[str]],
    target_payload_fields: Mapping[str, Sequence[str]],
    expected_target_values: Mapping[str, Mapping[str, object]],
    dry_run: bool,
) -> bool:
    """Skip current output or remove only one stale ``prior_comparison`` tree."""

    destination = Path(output_dir)
    if destination.name != "prior_comparison":
        raise ValueError("comparison replacement is restricted to a prior_comparison directory")
    if comparison_summary_is_current(
        destination / "summary.json",
        expected_fingerprint,
        expected_source_report=expected_source_report,
        comparison_root=destination,
        expected_representations=expected_representations,
        expected_targets=expected_targets,
        expected_artifacts=expected_artifacts,
        representation_payload_fields=representation_payload_fields,
        target_payload_fields=target_payload_fields,
        expected_target_values=expected_target_values,
    ):
        return True
    if destination.exists() and not dry_run:
        shutil.rmtree(destination)
    return False


def _comparison_root(summary_path: Path) -> Path:
    for parent in (summary_path.parent, *summary_path.parents):
        if parent.name == "prior_comparison":
            return parent.resolve()
    raise ValueError("normalized comparison summary must be inside prior_comparison")


def _safe_artifact_path(
    *,
    summary_path: Path,
    comparison_root: Path,
    artifact_path: str,
) -> Path:
    relative = Path(artifact_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe comparison artifact path: {artifact_path!r}")
    candidate = summary_path.parent / relative
    resolved = candidate.resolve()
    try:
        resolved.relative_to(comparison_root)
    except ValueError as exc:
        raise ValueError(
            f"comparison artifact escapes prior_comparison: {artifact_path!r}"
        ) from exc
    return candidate


def _artifact_identity(
    *,
    summary_path: Path,
    comparison_root: Path,
    artifact_path: str,
) -> dict[str, object]:
    resolved = _safe_artifact_path(
        summary_path=summary_path,
        comparison_root=comparison_root,
        artifact_path=artifact_path,
    )
    try:
        mode = resolved.lstat().st_mode
    except OSError as exc:
        raise FileNotFoundError(f"missing comparison artifact: {resolved}") from exc
    if resolved.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"comparison artifact must be a regular file: {resolved}")
    return {
        "path": artifact_path,
        "sha256": _sha256_file(resolved),
        "size": resolved.stat().st_size,
    }


def _normalized_summary_is_current(
    payload: object,
    *,
    summary_path: Path,
    expected_fingerprint: str,
    expected_source_report: Path,
    comparison_root: Path,
    expected_representations: Mapping[str, object] | None,
    expected_targets: Sequence[str] | None,
    expected_artifacts: Sequence[str] | None,
    representation_payload_fields: Mapping[str, Sequence[str]] | None,
    target_payload_fields: Mapping[str, Sequence[str]] | None,
    expected_target_values: Mapping[str, Mapping[str, object]] | None,
    visited: set[Path],
) -> bool:
    if (
        not isinstance(payload, dict)
        or set(payload) != _NORMALIZED_SUMMARY_KEYS
        or payload.get("schema_version") != 1
        or payload.get("status") not in _NORMALIZED_SUMMARY_STATUSES
        or payload.get("fingerprint") != expected_fingerprint
        or summary_path in visited
    ):
        return False
    visited.add(summary_path)
    source_identity = payload.get("source_report")
    expected_source_path = Path(
        os.path.relpath(expected_source_report, summary_path.parent)
    ).as_posix()
    if (
        not expected_source_report.is_file()
        or not isinstance(source_identity, dict)
        or set(source_identity) != {"path", "sha256"}
        or source_identity.get("path") != expected_source_path
        or source_identity.get("sha256") != _sha256_file(expected_source_report)
    ):
        return False
    representations = payload.get("representations")
    targets = payload.get("targets")
    if (
        not isinstance(representations, dict)
        or not isinstance(targets, dict)
        or not isinstance(payload.get("coverage"), dict)
        or not isinstance(payload.get("probe_configuration"), dict)
        or not targets
    ):
        return False
    if expected_representations is not None:
        if set(representations) != set(expected_representations):
            return False
        for name, expected in expected_representations.items():
            actual = representations[name]
            if (
                not isinstance(actual, dict)
                or not isinstance(expected, Mapping)
                or any(actual.get(key) != _json_safe(value) for key, value in expected.items())
            ):
                return False
    if expected_targets is not None and set(targets) != set(expected_targets):
        return False
    if representation_payload_fields is not None:
        if set(representations) != set(representation_payload_fields):
            return False
        for name, fields in representation_payload_fields.items():
            actual = representations[name]
            if not isinstance(actual, dict) or set(actual) != set(fields):
                return False
    if target_payload_fields is not None:
        if set(targets) != set(target_payload_fields):
            return False
        for name, fields in target_payload_fields.items():
            actual = targets[name]
            if not isinstance(actual, dict) or set(actual) != set(fields):
                return False
    if expected_target_values is not None:
        if set(expected_target_values) - set(targets):
            return False
        for name, expected in expected_target_values.items():
            actual = targets[name]
            if not isinstance(actual, dict) or any(
                actual.get(key) != _json_safe(value) for key, value in expected.items()
            ):
                return False
    if any(
        not isinstance(representation, dict)
        or not representation
        or not (
            (isinstance(representation.get("width"), int) and representation["width"] > 0)
            or (
                isinstance(representation.get("predictions"), list)
                and bool(representation["predictions"])
            )
        )
        for representation in representations.values()
    ):
        return False
    target_statuses: list[str] = []
    for target in targets.values():
        if not isinstance(target, dict) or target.get("status") not in {
            "ok",
            "degenerate",
            "incomplete",
        }:
            return False
        target_statuses.append(str(target["status"]))
    status_value = payload["status"]
    if status_value == "ok" and (
        not representations
        or "ok" not in target_statuses
        or any(value not in {"ok", "degenerate"} for value in target_statuses)
    ):
        return False
    if status_value == "degenerate" and (
        not representations
        or "degenerate" not in target_statuses
        or any(value not in {"ok", "degenerate"} for value in target_statuses)
    ):
        return False
    if status_value == "incomplete" and "incomplete" not in target_statuses:
        return False
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or status_value in {"ok", "degenerate"} and not artifacts:
        return False
    artifact_paths = [artifact.get("path") for artifact in artifacts if isinstance(artifact, dict)]
    if len(artifact_paths) != len(set(artifact_paths)):
        return False
    if expected_artifacts is not None and set(artifact_paths) != set(expected_artifacts):
        return False
    for artifact in artifacts:
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"path", "sha256", "size"}
            or not isinstance(artifact.get("path"), str)
            or not isinstance(artifact.get("sha256"), str)
            or len(artifact["sha256"]) != 64
            or not isinstance(artifact.get("size"), int)
            or artifact["size"] < 0
        ):
            return False
        try:
            artifact_file = _safe_artifact_path(
                summary_path=summary_path,
                comparison_root=comparison_root,
                artifact_path=artifact["path"],
            )
            mode = artifact_file.lstat().st_mode
        except (OSError, ValueError):
            return False
        if (
            artifact_file.is_symlink()
            or not stat.S_ISREG(mode)
            or artifact_file.stat().st_size != artifact["size"]
            or _sha256_file(artifact_file) != artifact["sha256"]
        ):
            return False
        if artifact_file.name == "summary.json":
            try:
                child = json.loads(artifact_file.read_text())
            except (OSError, json.JSONDecodeError):
                return False
            if isinstance(child, dict) and set(child) == _NORMALIZED_SUMMARY_KEYS:
                child_fingerprint = child.get("fingerprint")
                if not isinstance(child_fingerprint, str) or not _normalized_summary_is_current(
                    child,
                    summary_path=artifact_file,
                    expected_fingerprint=child_fingerprint,
                    expected_source_report=expected_source_report,
                    comparison_root=comparison_root,
                    expected_representations=None,
                    expected_targets=None,
                    expected_artifacts=None,
                    representation_payload_fields=None,
                    target_payload_fields=None,
                    expected_target_values=None,
                    visited=visited,
                ):
                    return False
    return True


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_absolute_path(item) for item in value)
    return isinstance(value, str) and Path(value).is_absolute()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_probe_comparison(
    output_dir: Path,
    target: str,
    series: Sequence[ProbeComparisonSeries],
    feature_counts: Sequence[int] | None,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Write atomic comparison plots and complete sidecars for a probe target."""
    _validate_series(series)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    normalized_series = [_json_safe(asdict(item)) for item in series]
    prediction_rows = _prediction_sidecar_rows(target, series)
    sweep_rows = _sweep_sidecar_rows(target, series)

    observed_csv = destination / "observed_vs_predicted_with_prior.csv"
    sweep_csv = destination / "feature_count_sweep_with_prior.csv"
    observed_png = destination / "observed_vs_predicted_with_prior.png"
    sweep_png = destination / "feature_count_sweep_with_prior.png"
    _atomic_csv(observed_csv, prediction_rows)
    _atomic_csv(sweep_csv, sweep_rows)
    _write_observed_vs_predicted(series, observed_png, target)
    _write_feature_count_sweep(series, sweep_png, target)

    fingerprint_payload: dict[str, object] = {
        "target": target,
        "series": normalized_series,
        "feature_counts": list(feature_counts) if feature_counts is not None else None,
        "provenance": provenance,
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "target": target,
        "plot_dpi": PLOT_DPI,
        "feature_counts": list(feature_counts) if feature_counts is not None else None,
        "provenance": _json_safe(provenance),
        "series": normalized_series,
        "fingerprint": comparison_fingerprint(fingerprint_payload),
        "artifacts": [
            observed_png.name,
            observed_csv.name,
            sweep_png.name,
            sweep_csv.name,
        ],
    }
    _atomic_json(destination / "summary.json", report)
    return report


def _validate_series(series: Sequence[ProbeComparisonSeries]) -> None:
    for item in series:
        if item.status not in _VALID_STATUSES:
            raise ValueError(f"Unknown status {item.status!r}; expected one of {_VALID_STATUSES!r}")
        _validate_reserved_sidecar_keys(item.metrics, "metrics")
        for prediction in item.prediction_rows:
            _validate_reserved_sidecar_keys(prediction, "prediction row")

    identities = [(item.representation, item.feature_count) for item in series]
    duplicates: list[tuple[str, int | None]] = []
    seen: set[tuple[str, int | None]] = set()
    for identity in identities:
        if identity in seen and identity not in duplicates:
            duplicates.append(identity)
        seen.add(identity)
    if duplicates:
        raise ValueError(f"Duplicate representation/feature_count identities: {duplicates!r}")


def _validate_reserved_sidecar_keys(values: Mapping[str, object], source: str) -> None:
    collisions = _RESERVED_SIDECAR_COLUMNS.intersection(values)
    if collisions:
        raise ValueError(f"{source} contains reserved sidecar keys: {sorted(collisions)!r}")


def _prediction_sidecar_rows(
    target: str, series: Sequence[ProbeComparisonSeries]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in series:
        common: dict[str, object] = {
            "target": target,
            "representation": item.representation,
            "display_name": item.display_name,
            "width": item.width,
            "feature_count": item.feature_count,
            "status": item.status,
            "reason": item.reason,
            "metrics_json": json.dumps(_json_safe(item.metrics), sort_keys=True),
        }
        if not item.prediction_rows:
            rows.append({**common, "row_kind": "series"})
            continue
        for prediction in item.prediction_rows:
            rows.append({**common, "row_kind": "prediction", **_json_safe(prediction)})
    return rows


def _sweep_sidecar_rows(
    target: str, series: Sequence[ProbeComparisonSeries]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in series:
        rows.append(
            {
                "target": target,
                "representation": item.representation,
                "display_name": item.display_name,
                "width": item.width,
                "feature_count": item.feature_count,
                "status": item.status,
                "reason": item.reason,
                **_json_safe(item.metrics),
            }
        )
    return rows


def _write_observed_vs_predicted(
    series: Sequence[ProbeComparisonSeries], path: Path, target: str
) -> None:
    representations = list(dict.fromkeys(item.representation for item in series)) or ["none"]
    figure, axes = plt.subplots(1, len(representations), figsize=(4.0 * len(representations), 4.0))
    axes_list = [axes] if len(representations) == 1 else list(axes)
    finite_values = [
        value
        for item in series
        if item.status == "ok"
        for row in item.prediction_rows
        for value in (_finite_row_value(row, "observed"), _finite_row_value(row, "predicted"))
        if value is not None
    ]
    limits = (min(finite_values), max(finite_values)) if finite_values else None
    for representation, axis in zip(representations, axes_list, strict=True):
        subset = [item for item in series if item.representation == representation]
        for item in subset:
            pairs = _finite_prediction_pairs(item) if item.status == "ok" else []
            if pairs:
                observed, predicted = zip(*pairs, strict=True)
                label = item.display_name
                if item.feature_count is not None:
                    label = f"{label} ({item.feature_count} features)"
                axis.scatter(observed, predicted, s=18, alpha=0.8, label=label)
            else:
                annotation = item.status
                if item.reason:
                    annotation = f"{annotation}: {item.reason}"
                axis.text(
                    0.5,
                    0.5,
                    annotation,
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    wrap=True,
                )
        if limits is not None:
            low, high = limits
            if low == high:
                padding = 1.0 if low == 0 else abs(low) * 0.05
                low, high = low - padding, high + padding
            axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
            axis.set(xlim=(low, high), ylim=(low, high))
        axis.set(
            title=_display_name(subset, representation),
            xlabel="Observed",
            ylabel="Out-of-fold prediction",
        )
        if any(_finite_prediction_pairs(item) for item in subset if item.status == "ok"):
            axis.legend(fontsize="small")
    figure.suptitle(f"{target}: observed versus predicted")
    _atomic_figure(figure, path)


def _write_feature_count_sweep(
    series: Sequence[ProbeComparisonSeries], path: Path, target: str
) -> None:
    metric_names = sorted(
        {
            key
            for item in series
            if item.feature_count is not None
            for key, value in item.metrics.items()
            if _finite_number(value) is not None
        }
    )
    if not metric_names:
        metric_names = ["no finite sweep metrics"]
    figure, axes = plt.subplots(1, len(metric_names), figsize=(4.5 * len(metric_names), 4.0))
    axes_list = [axes] if len(metric_names) == 1 else list(axes)
    for metric_name, axis in zip(metric_names, axes_list, strict=True):
        if metric_name == "no finite sweep metrics":
            axis.text(0.5, 0.5, "No finite feature-count sweep metrics", ha="center", va="center")
            axis.set_axis_off()
            continue
        for representation in dict.fromkeys(item.representation for item in series):
            points = [
                (item.feature_count, _finite_number(item.metrics.get(metric_name)))
                for item in series
                if item.representation == representation
                and item.feature_count is not None
                and item.status == "ok"
            ]
            finite_points = [(count, value) for count, value in points if value is not None]
            if finite_points:
                counts, values = zip(*sorted(finite_points), strict=True)
                axis.plot(
                    counts,
                    values,
                    marker="o",
                    label=_display_name_for_representation(series, representation),
                )
        axis.set(title=metric_name, xlabel="Feature count", ylabel=metric_name)
        if axis.lines:
            axis.legend(fontsize="small")
        else:
            axis.text(0.5, 0.5, "No non-degenerate values", transform=axis.transAxes, ha="center")
    figure.suptitle(f"{target}: feature-count sweep")
    _atomic_figure(figure, path)


def _finite_prediction_pairs(item: ProbeComparisonSeries) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for row in item.prediction_rows:
        observed = _finite_row_value(row, "observed")
        predicted = _finite_row_value(row, "predicted")
        if observed is not None and predicted is not None:
            pairs.append((observed, predicted))
    return pairs


def _finite_row_value(row: Mapping[str, object], name: str) -> float | None:
    return _finite_number(row.get(name))


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _display_name(series: Sequence[ProbeComparisonSeries], fallback: str) -> str:
    return series[0].display_name if series else fallback


def _display_name_for_representation(
    series: Sequence[ProbeComparisonSeries], representation: str
) -> str:
    return _display_name(
        [item for item in series if item.representation == representation], representation
    )


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if not columns:
        columns = ["row_kind"]
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(_json_safe(dict(row)) for row in rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_figure(figure: object, path: Path) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        figure.savefig(temporary, dpi=PLOT_DPI, format="png", bbox_inches="tight")  # type: ignore[attr-defined]
        temporary.replace(path)
    finally:
        plt.close(figure)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
