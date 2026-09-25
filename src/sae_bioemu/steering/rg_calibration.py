"""Radius-of-gyration calibration from cached BioEmu structure stores.

The BioEmu radius-of-gyration steering workflow needs three configuration values that
cannot be chosen without measuring the baseline property distribution first:

``target.range``
    ``continuous_target`` raises when a baseline falls outside this range, and it is
    called outside the ``try``/``except`` in :mod:`sae_bioemu.steering.ordinary_workflow`,
    so an under-sized range kills the training job after every baseline generation has
    already been paid for.
``bioemu_pytorch.target_mean`` / ``bioemu_pytorch.target_scale``
    Adapter input standardization constants, therefore training-time constants that must
    be derived from training-split data only.

The baseline the workflow actually uses is
``differentiable_radius_of_gyration(generate_unsteered(record, seed=unit.seed) * 10.0, mask)``
(see ``sae_bioemu.bioemu.steering_workflow``). Cached final structures are produced by the
same unsteered sampler, so their radii are the same quantity. This module therefore reuses
:func:`sae_bioemu.steering.surrogate.differentiable_radius_of_gyration` directly rather than
reimplementing the definition, and converts the stored nanometre positions to angstrom with
the same factor the property callback uses.

A protein sequence is the independent statistical unit: every summary is grouped by
``sequence_id`` first, and the repeated seeds of one sequence are never treated as
independent proteins.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sae_bioemu.steering.surrogate import differentiable_radius_of_gyration

__all__ = [
    "NANOMETERS_TO_ANGSTROM",
    "SequenceRadiusStatistics",
    "StoreRadiusCalibration",
    "StructureFailure",
    "StructureRadius",
    "StructureStoreSpec",
    "build_rg_calibration_report",
    "calibrate_structure_store",
    "parse_store_spec",
    "radius_of_gyration_angstrom",
    "write_rg_calibration",
]

#: Stored BioEmu positions are nanometres; the steering property callback multiplies by ten.
NANOMETERS_TO_ANGSTROM = 10.0

#: Unit label every accepted structure index row and npz archive must declare.
EXPECTED_POSITION_UNITS = "nanometers"

#: Reported quantiles of the per-sequence and within-sequence distributions.
PERCENTILES: tuple[float, ...] = (0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 100.0)

#: Default number of within-sequence standard deviations of head-room added to ``target.range``.
DEFAULT_UNSEEN_SEED_SIGMAS = 4.0

#: Default candidate ``maximum_change`` values whose implied safe range is reported.
DEFAULT_MAXIMUM_CHANGES: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)

_SCHEMA_VERSION = 1
_MAXIMUM_JSON_FAILURE_EXAMPLES = 200
_SPLITS = ("train", "eval")


@dataclass(frozen=True)
class StructureStoreSpec:
    """One labelled structure source: ordered roots merged over an explicit seed range."""

    label: str
    split: str
    roots: tuple[Path, ...]
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be non-empty")
        if self.split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}")
        if not isinstance(self.roots, tuple) or not self.roots:
            raise ValueError("roots must be a non-empty tuple")
        if any(not isinstance(root, Path) for root in self.roots):
            raise ValueError("roots must be Path objects")
        if len({root for root in self.roots}) != len(self.roots):
            raise ValueError("roots must be distinct")
        if not isinstance(self.seeds, tuple) or not self.seeds:
            raise ValueError("seeds must be a non-empty tuple")
        if any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in self.seeds
        ):
            raise ValueError("seeds must be non-negative integers")
        if len(set(self.seeds)) != len(self.seeds) or list(self.seeds) != sorted(self.seeds):
            raise ValueError("seeds must be distinct and ascending")


@dataclass(frozen=True)
class StructureRadius:
    """One cached structure's radius of gyration, keyed by sequence and sample seed."""

    sequence_id: str
    seed: int
    residue_count: int
    radius_of_gyration_angstrom: float
    store_root: str


@dataclass(frozen=True)
class StructureFailure:
    """An explicitly reported structure that could not contribute to the calibration."""

    label: str
    sequence_id: str | None
    seed: int | None
    reason: str
    detail: str


@dataclass(frozen=True)
class SequenceRadiusStatistics:
    """Per-sequence radius statistics; the sequence is the independent statistical unit."""

    sequence_id: str
    residue_count: int
    sample_count: int
    mean: float
    sd: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class StoreRadiusCalibration:
    """Per-structure and per-sequence radii for one labelled structure source."""

    label: str
    split: str
    roots: tuple[str, ...]
    index_paths: tuple[str, ...]
    seeds: tuple[int, ...]
    sequences: tuple[SequenceRadiusStatistics, ...]
    structures: tuple[StructureRadius, ...]
    failures: tuple[StructureFailure, ...]
    expected_structure_count: int


@dataclass(frozen=True)
class _IndexRow:
    sequence_id: str
    seed: int
    sequence: str
    npz_path: Path
    store_root: Path
    index_path: Path


def parse_store_spec(text: str) -> StructureStoreSpec:
    """Parse ``LABEL:SPLIT:SEED_START-SEED_END:ROOT[,ROOT...]`` into a store spec.

    Multiple roots are merged into a single ensemble keyed by ``(sequence_id, seed)``.
    """
    if not isinstance(text, str):
        raise ValueError("store spec must be a string")
    fields = text.split(":")
    if len(fields) != 4:
        raise ValueError(
            "store spec must be LABEL:SPLIT:SEED_START-SEED_END:ROOT[,ROOT...]: " f"{text!r}"
        )
    label, split, seed_range, roots = fields
    start_text, separator, stop_text = seed_range.partition("-")
    if not separator:
        raise ValueError(f"store spec seed range must be START-END: {text!r}")
    try:
        start, stop = int(start_text), int(stop_text)
    except ValueError as error:
        raise ValueError(f"store spec seed range must be integers: {text!r}") from error
    if start > stop:
        raise ValueError(f"store spec seed range must be ascending: {text!r}")
    root_paths = tuple(Path(part) for part in roots.split(",") if part)
    return StructureStoreSpec(
        label=label,
        split=split,
        roots=root_paths,
        seeds=tuple(range(start, stop + 1)),
    )


def radius_of_gyration_angstrom(positions_nanometers: np.ndarray) -> np.ndarray:
    """Return ``sqrt(mean_i ||x_i - centroid||^2)`` in angstrom for nanometre positions.

    Accepts ``(residues, 3)`` or ``(structures, residues, 3)``. The definition, including
    the float64 geometry promotion, comes from
    :func:`sae_bioemu.steering.surrogate.differentiable_radius_of_gyration`, so calibration
    cannot drift from the training objective.
    """
    array = np.asarray(positions_nanometers)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("positions must be a (residues, 3) or (structures, residues, 3) array")
    coordinates = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float64))
    coordinates = coordinates * NANOMETERS_TO_ANGSTROM
    mask = torch.ones(coordinates.shape[:2], dtype=torch.bool)
    with torch.no_grad():
        radii = differentiable_radius_of_gyration(coordinates, mask)
    return radii.numpy()


def calibrate_structure_store(spec: StructureStoreSpec) -> StoreRadiusCalibration:
    """Compute per-structure and per-sequence radii for one labelled structure source.

    Duplicate ``(sequence_id, seed)`` pairs are a fatal ambiguity and raise. Missing or
    unusable structures are reported explicitly as :class:`StructureFailure` records and
    are never silently dropped.
    """
    if not isinstance(spec, StructureStoreSpec):
        raise ValueError("spec must be a StructureStoreSpec")
    rows, index_paths, sequences_text = _read_merged_index(spec)
    failures: list[StructureFailure] = []
    structures: list[StructureRadius] = []
    statistics: list[SequenceRadiusStatistics] = []

    for sequence_id in sorted(sequences_text):
        expected_residues = len(sequences_text[sequence_id])
        positions: list[np.ndarray] = []
        seeds: list[int] = []
        roots: list[str] = []
        for seed in spec.seeds:
            row = rows.get((sequence_id, seed))
            if row is None:
                failures.append(
                    StructureFailure(
                        label=spec.label,
                        sequence_id=sequence_id,
                        seed=seed,
                        reason="missing_structure_index_entry",
                        detail=f"no index row for seed {seed}",
                    )
                )
                continue
            loaded = _load_positions(row, spec.label, expected_residues)
            if isinstance(loaded, StructureFailure):
                failures.append(loaded)
                continue
            positions.append(loaded)
            seeds.append(seed)
            roots.append(str(row.store_root))
        if not positions:
            failures.append(
                StructureFailure(
                    label=spec.label,
                    sequence_id=sequence_id,
                    seed=None,
                    reason="sequence_without_usable_structures",
                    detail="every requested seed failed; sequence excluded from statistics",
                )
            )
            continue
        radii = radius_of_gyration_angstrom(np.stack(positions, axis=0))
        for seed, radius, root in zip(seeds, radii.tolist(), roots, strict=True):
            structures.append(
                StructureRadius(
                    sequence_id=sequence_id,
                    seed=seed,
                    residue_count=expected_residues,
                    radius_of_gyration_angstrom=float(radius),
                    store_root=root,
                )
            )
        statistics.append(
            SequenceRadiusStatistics(
                sequence_id=sequence_id,
                residue_count=expected_residues,
                sample_count=int(radii.size),
                mean=float(radii.mean()),
                sd=float(radii.std(ddof=1)) if radii.size > 1 else 0.0,
                minimum=float(radii.min()),
                maximum=float(radii.max()),
            )
        )

    return StoreRadiusCalibration(
        label=spec.label,
        split=spec.split,
        roots=tuple(str(root) for root in spec.roots),
        index_paths=index_paths,
        seeds=spec.seeds,
        sequences=tuple(statistics),
        structures=tuple(structures),
        failures=tuple(failures),
        expected_structure_count=len(sequences_text) * len(spec.seeds),
    )


def build_rg_calibration_report(
    calibrations: Sequence[StoreRadiusCalibration],
    *,
    training_label: str,
    unseen_seed_sigmas: float = DEFAULT_UNSEEN_SEED_SIGMAS,
    maximum_changes: Sequence[float] = DEFAULT_MAXIMUM_CHANGES,
) -> dict[str, object]:
    """Assemble the JSON payload: per-store statistics, safe ranges, and standardization.

    ``target_mean``/``target_scale`` come from ``training_label`` alone so that no
    evaluation structure can leak into a training-time normalization constant. Stores are
    summarized separately and never pooled; only the ``target.range`` envelope spans
    stores, because that range must admit every train and validation baseline.
    """
    if not calibrations:
        raise ValueError("at least one store calibration is required")
    labels = [calibration.label for calibration in calibrations]
    if len(set(labels)) != len(labels):
        raise ValueError("store labels must be distinct")
    if not math.isfinite(unseen_seed_sigmas) or unseen_seed_sigmas < 0:
        raise ValueError("unseen_seed_sigmas must be finite and non-negative")
    if any(not math.isfinite(value) or value < 0 for value in maximum_changes):
        raise ValueError("maximum_changes must be finite and non-negative")
    training = next(
        (calibration for calibration in calibrations if calibration.label == training_label), None
    )
    if training is None:
        raise ValueError(f"training_label {training_label!r} is not among the calibrated stores")
    if training.split != "train":
        raise ValueError("target_mean and target_scale require a training-split store")

    stores = [_summarize_store(calibration) for calibration in calibrations]
    global_minimum = min(
        float(summary["structures"]["minimum"])
        for summary in stores
        if summary["structures"]["count"]
    )
    global_maximum = max(
        float(summary["structures"]["maximum"])
        for summary in stores
        if summary["structures"]["count"]
    )
    maximum_within_sequence_sd = max(
        float(summary["within_sequence_sd_angstrom"]["maximum"])
        for summary in stores
        if summary["within_sequence_sd_angstrom"]["count"]
    )
    unseen_seed_margin = unseen_seed_sigmas * maximum_within_sequence_sd
    minimum_safe_range = [
        global_minimum - unseen_seed_margin,
        global_maximum + unseen_seed_margin,
    ]

    training_means = np.asarray(
        [statistic.mean for statistic in training.sequences], dtype=np.float64
    )
    if training_means.size < 2:
        raise ValueError("target_scale requires at least two training sequences")
    training_radii = np.asarray(
        [structure.radius_of_gyration_angstrom for structure in training.structures],
        dtype=np.float64,
    )

    return {
        "schema_version": _SCHEMA_VERSION,
        "property_name": "radius_of_gyration",
        "units": "angstrom",
        "position_units_in_store": EXPECTED_POSITION_UNITS,
        "nanometers_to_angstrom": NANOMETERS_TO_ANGSTROM,
        "radius_definition": (
            "sqrt(mean_i ||x_i - centroid||^2) over C-alpha positions, computed by "
            "sae_bioemu.steering.surrogate.differentiable_radius_of_gyration"
        ),
        "statistical_unit": "sequence_id",
        "seed_field": "sample_index (BioEmu writes sample_index = sampling seed)",
        "stores": stores,
        "headline": {
            "within_sequence_sd_angstrom": {
                summary["label"]: summary["within_sequence_sd_angstrom"] for summary in stores
            },
            "per_sequence_mean_angstrom": {
                summary["label"]: summary["per_sequence_mean_angstrom"] for summary in stores
            },
            "global_structure_minimum_angstrom": global_minimum,
            "global_structure_maximum_angstrom": global_maximum,
        },
        "target_range": {
            "envelope_source_labels": labels,
            "global_structure_minimum_angstrom": global_minimum,
            "global_structure_maximum_angstrom": global_maximum,
            "maximum_within_sequence_sd_angstrom": maximum_within_sequence_sd,
            "unseen_seed_sigmas": unseen_seed_sigmas,
            "unseen_seed_margin_angstrom": unseen_seed_margin,
            "minimum_safe_range": minimum_safe_range,
            "range_by_maximum_change": {
                f"{value:g}": [
                    minimum_safe_range[0] - value,
                    minimum_safe_range[1] + value,
                ]
                for value in maximum_changes
            },
            "note": (
                "target.range must envelope every individual train and validation baseline, "
                "not just the per-sequence means: continuous_target raises outside the "
                "try/except in ordinary_workflow and kills the job. The steering schedule "
                "draws uint32 seeds derived from the manifest fingerprint, not the cached "
                "sample_index seeds, so the observed structure extrema are a sample of the "
                "baseline support and the unseen-seed margin above is required head-room. "
                "range_by_maximum_change additionally keeps min(maximum_change, "
                "baseline - lower, upper - baseline) equal to maximum_change for every "
                "observed baseline, so no sequence silently loses steering head-room."
            ),
        },
        "adapter_standardization": {
            "training_label": training.label,
            "training_split": training.split,
            "leakage_note": (
                "computed from the training store only; evaluation structures never enter "
                "these training-time normalization constants"
            ),
            "aggregation_unit": "sequence_id",
            "target_mean": float(training_means.mean()),
            "target_scale": float(training_means.std(ddof=1)),
            "sequence_count": int(training_means.size),
            "structure_pooled_reference": {
                "note": "reported for transparency only; not the recommended constants",
                "mean": float(training_radii.mean()),
                "sd": float(training_radii.std(ddof=1)) if training_radii.size > 1 else 0.0,
                "structure_count": int(training_radii.size),
            },
        },
    }


def write_rg_calibration(
    output_dir: str | Path,
    report: Mapping[str, object],
    calibrations: Sequence[StoreRadiusCalibration],
    *,
    stem: str = "rg_calibration",
) -> dict[str, Path]:
    """Write the JSON report plus per-structure, per-sequence, and failure CSV sidecars.

    The sidecars are sufficient to regenerate any distribution plot without re-reading the
    structure store.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": destination / f"{stem}.json",
        "structures_csv": destination / f"{stem}_structures.csv",
        "sequences_csv": destination / f"{stem}_sequences.csv",
        "failures_csv": destination / f"{stem}_failures.csv",
    }
    payload = dict(report)
    payload["sidecars"] = {name: path.name for name, path in paths.items() if name != "json"}
    payload["failures"] = _failure_summary(calibrations, paths["failures_csv"])
    payload["coverage"] = _coverage_summary(calibrations)

    _atomic_write_csv(
        paths["structures_csv"],
        (
            "store_label",
            "split",
            "sequence_id",
            "seed",
            "residue_count",
            "radius_of_gyration_angstrom",
        ),
        (
            (
                calibration.label,
                calibration.split,
                structure.sequence_id,
                structure.seed,
                structure.residue_count,
                structure.radius_of_gyration_angstrom,
            )
            for calibration in calibrations
            for structure in calibration.structures
        ),
    )
    _atomic_write_csv(
        paths["sequences_csv"],
        (
            "store_label",
            "split",
            "sequence_id",
            "residue_count",
            "sample_count",
            "mean_radius_of_gyration_angstrom",
            "sd_radius_of_gyration_angstrom",
            "min_radius_of_gyration_angstrom",
            "max_radius_of_gyration_angstrom",
        ),
        (
            (
                calibration.label,
                calibration.split,
                statistic.sequence_id,
                statistic.residue_count,
                statistic.sample_count,
                statistic.mean,
                statistic.sd,
                statistic.minimum,
                statistic.maximum,
            )
            for calibration in calibrations
            for statistic in calibration.sequences
        ),
    )
    _atomic_write_csv(
        paths["failures_csv"],
        ("store_label", "sequence_id", "seed", "reason", "detail"),
        (
            (
                failure.label,
                failure.sequence_id if failure.sequence_id is not None else "",
                failure.seed if failure.seed is not None else "",
                failure.reason,
                failure.detail,
            )
            for calibration in calibrations
            for failure in calibration.failures
        ),
    )
    _atomic_write_text(paths["json"], json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return paths


def _read_merged_index(
    spec: StructureStoreSpec,
) -> tuple[dict[tuple[str, int], _IndexRow], tuple[str, ...], dict[str, str]]:
    requested = set(spec.seeds)
    rows: dict[tuple[str, int], _IndexRow] = {}
    sequences_text: dict[str, str] = {}
    index_paths: list[str] = []
    for root in spec.roots:
        store_root, index_path = _resolve_index(root)
        index_paths.append(str(index_path))
        with index_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                payload = json.loads(raw_line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{index_path}:{line_number}: row must be a JSON object")
                seed = payload["sample_index"]
                if not isinstance(seed, int) or isinstance(seed, bool):
                    raise ValueError(f"{index_path}:{line_number}: sample_index must be an integer")
                if seed not in requested:
                    continue
                sequence_id = payload["sequence_id"]
                sequence = payload["sequence"]
                if not isinstance(sequence_id, str) or not isinstance(sequence, str):
                    raise ValueError(f"{index_path}:{line_number}: invalid sequence identity")
                if payload.get("pos_units") != EXPECTED_POSITION_UNITS:
                    raise ValueError(
                        f"{index_path}:{line_number}: pos_units must be "
                        f"{EXPECTED_POSITION_UNITS!r}, found {payload.get('pos_units')!r}"
                    )
                npz_path = payload.get("npz_path")
                if not isinstance(npz_path, str) or not npz_path:
                    raise ValueError(f"{index_path}:{line_number}: npz_path is required")
                known = sequences_text.setdefault(sequence_id, sequence)
                if known != sequence:
                    raise ValueError(
                        f"{index_path}:{line_number}: conflicting sequence for {sequence_id}"
                    )
                key = (sequence_id, seed)
                if key in rows:
                    raise ValueError(
                        "duplicate (sequence_id, seed) across merged structure stores: "
                        f"{sequence_id} seed {seed} in {rows[key].index_path} and {index_path}"
                    )
                rows[key] = _IndexRow(
                    sequence_id=sequence_id,
                    seed=seed,
                    sequence=sequence,
                    npz_path=_member_path(store_root, npz_path),
                    store_root=store_root,
                    index_path=index_path,
                )
    if not rows:
        raise ValueError(f"store {spec.label!r} has no index rows for the requested seed range")
    return rows, tuple(index_paths), sequences_text


def _resolve_index(root: Path) -> tuple[Path, Path]:
    """Return ``(store_root, index_path)`` for a store root or its final-structures directory."""
    if (root / "final_structures" / "index.jsonl").is_file():
        return root, root / "final_structures" / "index.jsonl"
    if (root / "index.jsonl").is_file():
        return root.parent, root / "index.jsonl"
    raise ValueError(f"structure store has no index.jsonl: {root}")


def _member_path(store_root: Path, member: str) -> Path:
    relative = Path(member)
    if relative.is_absolute():
        raise ValueError(f"npz_path must be relative to the structure store: {member}")
    canonical_root = store_root.resolve(strict=False)
    canonical_member = (canonical_root / relative).resolve(strict=False)
    if canonical_root not in canonical_member.parents:
        raise ValueError(f"npz_path escapes the structure store: {member}")
    return canonical_member


def _load_positions(
    row: _IndexRow, label: str, expected_residues: int
) -> np.ndarray | StructureFailure:
    def failure(reason: str, detail: str) -> StructureFailure:
        return StructureFailure(
            label=label,
            sequence_id=row.sequence_id,
            seed=row.seed,
            reason=reason,
            detail=detail,
        )

    try:
        with np.load(row.npz_path, allow_pickle=False) as archive:
            positions = np.asarray(archive["pos"])
            units = archive["pos_units"]
    except FileNotFoundError:
        return failure("missing_structure_file", str(row.npz_path))
    except (OSError, ValueError, KeyError) as error:
        return failure("unreadable_structure_file", f"{type(error).__name__}: {error}")
    if units.ndim != 0 or str(units.item()) != EXPECTED_POSITION_UNITS:
        raise ValueError(
            f"{row.npz_path}: pos_units must be {EXPECTED_POSITION_UNITS!r}, found {units!r}"
        )
    if positions.ndim != 2 or positions.shape[1] != 3:
        return failure("invalid_position_shape", f"pos has shape {positions.shape}")
    if positions.shape[0] != expected_residues:
        return failure(
            "structure_length_mismatch",
            f"pos has {positions.shape[0]} residues, sequence has {expected_residues}",
        )
    if expected_residues < 2:
        return failure("insufficient_residues", f"{expected_residues} residues")
    if not np.isfinite(positions).all():
        return failure("non_finite_coordinates", "pos contains non-finite values")
    return positions.astype(np.float64, copy=False)


def _distribution(values: np.ndarray) -> dict[str, object]:
    if values.size == 0:
        return {"count": 0, "mean": None, "sd": None, "minimum": None, "maximum": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "percentiles": {
            f"p{percentile:g}": float(np.percentile(values, percentile))
            for percentile in PERCENTILES
        },
    }


def _summarize_store(calibration: StoreRadiusCalibration) -> dict[str, object]:
    means = np.asarray([statistic.mean for statistic in calibration.sequences], dtype=np.float64)
    spreads = np.asarray(
        [statistic.sd for statistic in calibration.sequences if statistic.sample_count > 1],
        dtype=np.float64,
    )
    radii = np.asarray(
        [structure.radius_of_gyration_angstrom for structure in calibration.structures],
        dtype=np.float64,
    )
    ranges = np.asarray(
        [
            statistic.maximum - statistic.minimum
            for statistic in calibration.sequences
            if statistic.sample_count > 1
        ],
        dtype=np.float64,
    )
    return {
        "label": calibration.label,
        "split": calibration.split,
        "roots": list(calibration.roots),
        "index_paths": list(calibration.index_paths),
        "seed_minimum": calibration.seeds[0],
        "seed_maximum": calibration.seeds[-1],
        "seed_count": len(calibration.seeds),
        "sequence_count": len(calibration.sequences),
        "structure_count": len(calibration.structures),
        "expected_structure_count": calibration.expected_structure_count,
        "single_sample_sequence_count": sum(
            1 for statistic in calibration.sequences if statistic.sample_count == 1
        ),
        "per_sequence_mean_angstrom": _distribution(means),
        "within_sequence_sd_angstrom": _distribution(spreads),
        "within_sequence_range_angstrom": _distribution(ranges),
        "structures": _distribution(radii),
    }


def _failure_summary(
    calibrations: Sequence[StoreRadiusCalibration], failures_csv: Path
) -> dict[str, object]:
    failures = [failure for calibration in calibrations for failure in calibration.failures]
    by_reason: dict[str, int] = {}
    for failure in failures:
        by_reason[failure.reason] = by_reason.get(failure.reason, 0) + 1
    return {
        "total": len(failures),
        "by_reason": dict(sorted(by_reason.items())),
        "complete_listing_csv": failures_csv.name,
        "examples": [
            {
                "store_label": failure.label,
                "sequence_id": failure.sequence_id,
                "seed": failure.seed,
                "reason": failure.reason,
                "detail": failure.detail,
            }
            for failure in failures[:_MAXIMUM_JSON_FAILURE_EXAMPLES]
        ],
        "examples_truncated": len(failures) > _MAXIMUM_JSON_FAILURE_EXAMPLES,
    }


def _coverage_summary(calibrations: Sequence[StoreRadiusCalibration]) -> dict[str, object]:
    return {
        "complete": all(
            len(calibration.structures) == calibration.expected_structure_count
            for calibration in calibrations
        ),
        "by_store": {
            calibration.label: {
                "observed_structures": len(calibration.structures),
                "expected_structures": calibration.expected_structure_count,
                "missing_structures": calibration.expected_structure_count
                - len(calibration.structures),
                "sequences_with_statistics": len(calibration.sequences),
            }
            for calibration in calibrations
        },
    }


def _atomic_write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
