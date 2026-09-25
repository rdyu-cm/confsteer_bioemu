"""Sequence-aware diagnostics for sample-level BioEmu properties."""

from __future__ import annotations

import colorsys
import csv
import hashlib
import json
import math
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

StructureKey = tuple[str, int]
SCATTER_SCHEMA_VERSION = 2
SCATTER_COLUMNS = (
    "sequence_id",
    "sample_index",
    "raw_property",
    "relative_property",
    "raw_feature",
    "raw_feature_selection_source",
    "relative_feature",
    "relative_feature_selection_source",
    "raw_feature_activation",
    "raw_feature_activation_zscore",
    "relative_feature_activation",
    "relative_feature_activation_zscore",
    "raw_property_value",
    "relative_property_value",
    "sequence_color",
    "feature_aggregation",
    "sampling_scope",
)
PANEL_NAMES = (
    "raw_feature_raw_property",
    "raw_feature_relative_property_normalized",
    "relative_feature_raw_property",
    "relative_feature_relative_property_normalized",
)


def stable_sequence_color(sequence_id: str) -> str:
    """Return a deterministic, visible color for a sequence identifier."""
    digest = hashlib.sha256(sequence_id.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:4], byteorder="big") / float(2**32)
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.78)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def _add_within_sequence_feature_zscores(rows: list[dict[str, object]]) -> None:
    activations: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"raw": [], "relative": []}
    )
    for row in rows:
        sequence_id = str(row["sequence_id"])
        activations[sequence_id]["raw"].append(float(row["raw_feature_activation"]))
        activations[sequence_id]["relative"].append(float(row["relative_feature_activation"]))
    statistics = {
        sequence_id: {
            name: (float(np.mean(values)), float(np.std(values)))
            for name, values in by_feature.items()
        }
        for sequence_id, by_feature in activations.items()
    }
    for row in rows:
        sequence_id = str(row["sequence_id"])
        raw_mean, raw_std = statistics[sequence_id]["raw"]
        relative_mean, relative_std = statistics[sequence_id]["relative"]
        row["raw_feature_activation_zscore"] = (
            (float(row["raw_feature_activation"]) - raw_mean) / raw_std if raw_std > 1e-12 else 0.0
        )
        row["relative_feature_activation_zscore"] = (
            (float(row["relative_feature_activation"]) - relative_mean) / relative_std
            if relative_std > 1e-12
            else 0.0
        )


def build_property_scatter_rows(
    structure_feature_summaries: Mapping[StructureKey, np.ndarray],
    structure_properties: Mapping[StructureKey, Mapping[str, float]],
    raw_property_name: str,
    relative_property_name: str,
    raw_feature: int,
    relative_feature: int,
    *,
    expected_samples_per_sequence: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Align pooled feature vectors and properties into one row per structure."""
    if raw_feature < 0 or relative_feature < 0:
        raise ValueError("selected feature indices must be non-negative")

    summary_keys = set(structure_feature_summaries)
    property_keys = set(structure_properties)
    aligned_keys = sorted(summary_keys & property_keys)
    rows: list[dict[str, Any]] = []
    nonfinite_keys: list[list[str | int]] = []

    for sequence_id, sample_index in aligned_keys:
        vector = np.asarray(structure_feature_summaries[(sequence_id, sample_index)])
        if vector.ndim != 1 or max(raw_feature, relative_feature) >= vector.size:
            raise ValueError("structure feature summary does not match selected feature indices")
        properties = structure_properties[(sequence_id, sample_index)]
        raw_property_value = float(properties.get(raw_property_name, float("nan")))
        relative_property_value = float(properties.get(relative_property_name, float("nan")))
        raw_activation = float(vector[raw_feature])
        relative_activation = float(vector[relative_feature])
        if not all(
            math.isfinite(value)
            for value in (
                raw_property_value,
                relative_property_value,
                raw_activation,
                relative_activation,
            )
        ):
            nonfinite_keys.append([sequence_id, sample_index])
            continue
        rows.append(
            {
                "sequence_id": sequence_id,
                "sample_index": sample_index,
                "raw_property": raw_property_name,
                "relative_property": relative_property_name,
                "raw_feature": raw_feature,
                "raw_feature_selection_source": "raw",
                "relative_feature": relative_feature,
                "relative_feature_selection_source": "relative",
                "raw_feature_activation": raw_activation,
                "relative_feature_activation": relative_activation,
                "raw_property_value": raw_property_value,
                "relative_property_value": relative_property_value,
                "sequence_color": stable_sequence_color(sequence_id),
                "feature_aggregation": ("structure_mean_all_residues_all_recorded_timesteps"),
                "sampling_scope": "one_row_per_sequence_sample_structure",
            }
        )

    _add_within_sequence_feature_zscores(rows)

    counts = Counter(str(row["sequence_id"]) for row in rows)
    partial_sequences: dict[str, int] = {}
    if expected_samples_per_sequence is not None:
        if expected_samples_per_sequence <= 0:
            raise ValueError("expected_samples_per_sequence must be positive")
        sequence_ids = sorted({sequence_id for sequence_id, _ in summary_keys | property_keys})
        partial_sequences = {
            sequence_id: counts.get(sequence_id, 0)
            for sequence_id in sequence_ids
            if counts.get(sequence_id, 0) != expected_samples_per_sequence
        }

    coverage: dict[str, object] = {
        "num_structures": len(rows),
        "num_sequences": len(counts),
        "expected_samples_per_sequence": expected_samples_per_sequence,
        "missing_summary_keys": [list(key) for key in sorted(property_keys - summary_keys)],
        "missing_property_keys": [list(key) for key in sorted(summary_keys - property_keys)],
        "nonfinite_keys": nonfinite_keys,
        "partial_sequences": partial_sequences,
    }
    return rows, coverage


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        return Path(handle.name)


def _write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    temporary_path = _temporary_sibling(path)
    try:
        with temporary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCATTER_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    temporary_path = _temporary_sibling(path)
    try:
        temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _render_property_scatter(
    rows: list[Mapping[str, object]],
    metadata: Mapping[str, object],
    plot_path: Path,
) -> None:
    if not rows:
        raise ValueError("cannot plot property scatter without finite aligned rows")
    raw_property_name = str(metadata["raw_property_name"])
    relative_property_name = str(metadata["relative_property_name"])
    raw_feature = int(metadata["raw_feature"])
    relative_feature = int(metadata["relative_feature"])
    raw_spearman = float(metadata["raw_held_out_spearman"])
    relative_spearman = float(metadata["relative_held_out_spearman"])
    colors = [str(row["sequence_color"]) for row in rows]

    panels = (
        (
            "raw_feature_activation",
            "raw_property_value",
            f"Raw-selected feature {raw_feature}\nheld-out ρ={raw_spearman:.3f}",
            f"Feature {raw_feature} structure mean",
            raw_property_name,
        ),
        (
            "raw_feature_activation_zscore",
            "relative_property_value",
            f"Raw-selected feature {raw_feature}, normalized axes",
            f"Feature {raw_feature} within-sequence z-score",
            f"{relative_property_name} (within-sequence z-score)",
        ),
        (
            "relative_feature_activation",
            "raw_property_value",
            f"Relative-selected feature {relative_feature}, raw axes",
            f"Feature {relative_feature} structure mean",
            raw_property_name,
        ),
        (
            "relative_feature_activation_zscore",
            "relative_property_value",
            (
                f"Relative-selected feature {relative_feature}, normalized axes\n"
                f"held-out ρ={relative_spearman:.3f}"
            ),
            f"Feature {relative_feature} within-sequence z-score",
            f"{relative_property_name} (within-sequence z-score)",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    for axis, (x_name, y_name, title, x_label, y_label) in zip(axes.flat, panels, strict=True):
        axis.scatter(
            [float(row[x_name]) for row in rows],
            [float(row[y_name]) for row in rows],
            c=colors,
            s=7,
            alpha=0.22,
            linewidths=0,
        )
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.18)
    coverage = metadata["coverage"]
    if not isinstance(coverage, Mapping):
        raise ValueError("scatter metadata coverage must be a mapping")
    figure.suptitle(
        f"{raw_property_name}: {coverage['num_structures']} structures, "
        f"{coverage['num_sequences']} sequences\n"
        "Each point is one structure; sequence is the independent biological group"
    )
    temporary_path = _temporary_sibling(plot_path)
    try:
        figure.savefig(temporary_path, format="png", dpi=300)
        temporary_path.replace(plot_path)
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)


def write_property_scatter_artifacts(
    structure_feature_summaries: Mapping[StructureKey, np.ndarray],
    structure_properties: Mapping[StructureKey, Mapping[str, float]],
    raw_property_name: str,
    relative_property_name: str,
    raw_feature: int,
    relative_feature: int,
    raw_spearman: float,
    relative_spearman: float,
    output_dir: Path,
    *,
    expected_samples_per_sequence: int | None = 32,
) -> dict[str, object]:
    """Write exact selected-feature sidecars and their four-panel plot."""
    rows, coverage = build_property_scatter_rows(
        structure_feature_summaries,
        structure_properties,
        raw_property_name,
        relative_property_name,
        raw_feature,
        relative_feature,
        expected_samples_per_sequence=expected_samples_per_sequence,
    )
    if not rows:
        raise ValueError("no finite aligned observations for property scatter")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{raw_property_name}_sequence_colored"
    csv_path = output_dir / f"{stem}.csv"
    metadata_path = output_dir / f"{stem}.json"
    plot_path = output_dir / f"{stem}.png"
    metadata: dict[str, object] = {
        "schema_version": SCATTER_SCHEMA_VERSION,
        "raw_property_name": raw_property_name,
        "relative_property_name": relative_property_name,
        "raw_feature": raw_feature,
        "raw_feature_selection_source": "raw",
        "relative_feature": relative_feature,
        "relative_feature_selection_source": "relative",
        "raw_held_out_spearman": raw_spearman,
        "relative_held_out_spearman": relative_spearman,
        "panels": list(PANEL_NAMES),
        "panel_coordinates": {
            PANEL_NAMES[0]: ["raw_feature_activation", "raw_property_value"],
            PANEL_NAMES[1]: [
                "raw_feature_activation_zscore",
                "relative_property_value",
            ],
            PANEL_NAMES[2]: [
                "relative_feature_activation",
                "raw_property_value",
            ],
            PANEL_NAMES[3]: [
                "relative_feature_activation_zscore",
                "relative_property_value",
            ],
        },
        "relative_axis_normalization": "within_sequence_zscore",
        "coverage": coverage,
        "feature_aggregation": "structure_mean_all_residues_all_recorded_timesteps",
        "sampling_scope": "one_row_per_sequence_sample_structure",
        "csv_columns": list(SCATTER_COLUMNS),
    }
    _write_csv_atomic(csv_path, rows)
    _write_json_atomic(metadata_path, metadata)
    _render_property_scatter(rows, metadata, plot_path)
    return {
        "status": "written",
        "plot_path": str(plot_path),
        "csv_path": str(csv_path),
        "metadata_path": str(metadata_path),
        **metadata,
    }


def regenerate_property_scatter(
    csv_path: Path,
    metadata_path: Path,
    plot_path: Path,
) -> None:
    """Regenerate a four-panel plot exclusively from its durable sidecars."""
    with csv_path.open(newline="") as handle:
        rows: list[dict[str, object]] = list(csv.DictReader(handle))
    metadata = json.loads(metadata_path.read_text())
    schema_version = int(metadata.get("schema_version", -1))
    if schema_version == 1:
        _add_within_sequence_feature_zscores(rows)
        metadata = {**metadata, "schema_version": SCATTER_SCHEMA_VERSION}
    elif schema_version != SCATTER_SCHEMA_VERSION:
        raise ValueError("unsupported property scatter sidecar schema")
    _render_property_scatter(rows, metadata, plot_path)


def selected_raw_relative_features(
    property_importance: Mapping[str, list[Mapping[str, object]]],
    raw_property_name: str,
    relative_property_name: str,
) -> tuple[int, int, float, float] | None:
    """Return raw/relative selected features and their held-out Spearman values."""

    def first_selected(
        property_name: str,
    ) -> tuple[int, float] | None:
        for row in property_importance.get(property_name, []):
            importance = float(row.get("importance_score", 0.0))
            spearman = float(row.get("spearman_r_sample", float("nan")))
            feature = int(row.get("feature", -1))
            if importance > 0.0 and math.isfinite(spearman) and feature >= 0:
                return feature, spearman
        return None

    raw_selected = first_selected(raw_property_name)
    relative_selected = first_selected(relative_property_name)
    if raw_selected is None or relative_selected is None:
        return None
    return (
        raw_selected[0],
        relative_selected[0],
        raw_selected[1],
        relative_selected[1],
    )


def write_raw_relative_property_scatter_pairs(
    property_importance: Mapping[str, list[Mapping[str, object]]],
    structure_feature_summaries: Mapping[StructureKey, np.ndarray],
    structure_properties: Mapping[StructureKey, Mapping[str, float]],
    output_dir: Path,
    *,
    expected_samples_per_sequence: int | None = 32,
) -> list[dict[str, object]]:
    """Write fixed radius/contact raw-relative comparison artifacts."""
    reports: list[dict[str, object]] = []
    for raw_property_name, relative_property_name in (
        ("radius_of_gyration", "relative_radius_of_gyration"),
        ("contact_density", "relative_contact_density"),
    ):
        selected = selected_raw_relative_features(
            property_importance,
            raw_property_name,
            relative_property_name,
        )
        if selected is None:
            reports.append(
                {
                    "status": "missing",
                    "reason": "raw or relative property has no selected feature",
                    "raw_property_name": raw_property_name,
                    "relative_property_name": relative_property_name,
                }
            )
            continue
        raw_feature, relative_feature, raw_spearman, relative_spearman = selected
        rows, coverage = build_property_scatter_rows(
            structure_feature_summaries,
            structure_properties,
            raw_property_name,
            relative_property_name,
            raw_feature,
            relative_feature,
            expected_samples_per_sequence=expected_samples_per_sequence,
        )
        if not rows:
            reports.append(
                {
                    "status": "missing",
                    "reason": "no finite aligned structure summaries and properties",
                    "raw_property_name": raw_property_name,
                    "relative_property_name": relative_property_name,
                    "coverage": coverage,
                }
            )
            continue
        reports.append(
            write_property_scatter_artifacts(
                structure_feature_summaries,
                structure_properties,
                raw_property_name,
                relative_property_name,
                raw_feature,
                relative_feature,
                raw_spearman,
                relative_spearman,
                output_dir,
                expected_samples_per_sequence=expected_samples_per_sequence,
            )
        )
    return reports
