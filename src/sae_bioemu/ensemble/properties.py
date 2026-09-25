from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit

from sae_bioemu.analysis.general import (
    StructureIndexEntry,
    add_within_sequence_normalized_sample_properties,
    align_trajectory_to_reference,
    contact_matrix,
    load_structure_positions,
)
from sae_bioemu.ensemble.config import property_scientific_dependency_fingerprint

CONTACT_CUTOFF_ANGSTROM = 8.0
MIN_SEQUENCE_SEPARATION = 3
LOCAL_Q_BETA_PER_ANGSTROM = 5.0
LOCAL_Q_DISTANCE_MULTIPLIER = 1.2
LOCAL_Q_ORDERED_SCORE_SLOPE = 20.0
LOCAL_Q_BOUNDARY_CANDIDATES = tuple(float(value) / 100.0 for value in range(30, 71))
LOCAL_Q_SENSITIVITY_PARAMETERS = tuple(
    (cutoff, beta, distance_multiplier)
    for cutoff in (7.5, 8.0, 8.5)
    for beta in (4.0, 5.0, 6.0)
    for distance_multiplier in (1.1, 1.2, 1.3)
)
PRIMARY_LOCAL_Q_PARAMETERS = (
    CONTACT_CUTOFF_ANGSTROM,
    LOCAL_Q_BETA_PER_ANGSTROM,
    LOCAL_Q_DISTANCE_MULTIPLIER,
)
KCAL_PER_MOL_K = 0.00198720425864083
LOCAL_DELTA_G_TEMPERATURE_K = 300.0
PRACTICAL_LOCAL_DELTA_G_LIMIT_KCAL_MOL = 1.5
SAMPLE_PROPERTIES = (
    "radius_of_gyration",
    "contact_density",
    "end_to_end_distance",
    "relative_shape_anisotropy",
)
CONVERGENCE_PROPERTY_UNITS = {
    "radius_of_gyration_mean": "angstrom",
    "contact_density_mean": "fraction",
    "end_to_end_distance_mean": "angstrom",
    "relative_shape_anisotropy_mean": "dimensionless",
}

_SAMPLE_FIELDS = (
    "sequence_id",
    "sample_index",
    "radius_of_gyration",
    "relative_radius_of_gyration",
    "contact_density",
    "contact_density_status",
    "relative_contact_density",
    "end_to_end_distance",
    "relative_shape_anisotropy",
)
_SEQUENCE_FIELDS = (
    "sequence_id",
    "n_target_samples",
    "expected_target_samples",
    "complete_target_coverage",
    "contact_pair_status",
    "radius_of_gyration_mean",
    "radius_of_gyration_variance",
    "contact_density_mean",
    "contact_density_variance",
    "end_to_end_distance_mean",
    "end_to_end_distance_variance",
    "relative_shape_anisotropy_mean",
    "relative_shape_anisotropy_variance",
    "mean_reference_aligned_pairwise_rmsd",
    "mean_eligible_pair_binary_contact_entropy",
)
_RESIDUE_FIELDS = (
    "sequence_id",
    "residue_index",
    "n_target_samples",
    "rmsf",
    "contact_number_mean",
    "contact_number_variance",
    "eligible_contact_count",
    "eligible_contact_status",
    "mean_eligible_contact_probability",
    "mean_eligible_pair_contact_entropy",
    "reference_local_contact_count",
    "local_q_status",
    "local_q_mean",
    "local_q_variance",
)
_LOCAL_DELTA_G_FIELDS = (
    "property_name",
    "sequence_id",
    "residue_index",
    "sampling_scope",
    "sample_indices",
    "sample_count",
    "units",
    "status",
    "reason",
    "value",
    "censored",
    "censoring_bound_kcal_mol",
    "censoring_direction",
    "p_ordered",
    "effective_ordered_observations",
    "effective_disordered_observations",
    "bootstrap_ci_low_kcal_mol",
    "bootstrap_ci_high_kcal_mol",
    "bootstrap_ci_width_kcal_mol",
    "bootstrap_replicates",
    "reference_contact_count",
    "sensitivity_evaluated_count",
    "sensitivity_estimable_count",
    "estimator_kind",
    "estimator_definition_id",
    "sensitivity_grid_id",
    "calibration_fingerprint",
    "calibration_artifact_sha256",
    "calibration_input_manifest_sha256",
)
_CONVERGENCE_FIELDS = (
    "sequence_id",
    "property",
    "target_count",
    "sample_indices",
    "observed_sample_count",
    "coverage_fraction",
    "estimate",
    "successive_absolute_change",
    "units",
    "status",
    "reason",
    "bootstrap_95_lower",
    "bootstrap_95_upper",
    "bootstrap_95_width",
    "bootstrap_replicates",
    "bootstrap_seed",
    "bootstrap_schedule_fingerprint",
)
_FAILURE_FIELDS = (
    "sequence_id",
    "sample_index",
    "role",
    "reason",
    "detail",
)
CANONICAL_PROPERTY_OUTPUT_KEYS = frozenset(
    f"{table}_{suffix}"
    for table in ("sample", "sequence", "residue", "convergence", "failures")
    for suffix in ("csv", "json")
) | {"tables_npz", "coverage_json"}
CANONICAL_PROPERTY_ARTIFACT_KEYS = CANONICAL_PROPERTY_OUTPUT_KEYS | {"metadata_json"}
CANONICAL_PROPERTY_TABLE_PATH_KEYS = CANONICAL_PROPERTY_ARTIFACT_KEYS | {
    "publication_marker_json",
    "publication_manifest_json",
}
_FORBIDDEN_PROPERTY_DECLARATIONS = (
    "calibrated_effective_local_delta_g",
    "local_delta_g",
)
_DECLARATION_BEARING_PROPERTY_SECTIONS = frozenset(
    ("outputs", "artifacts", "units", "definitions", "coverage")
)


@dataclass(frozen=True)
class FailureRecord:
    sequence_id: str | None
    sample_index: int | None
    role: str
    reason: str
    detail: str


@dataclass(frozen=True)
class SequencePropertyComputation:
    sample: tuple[dict[str, object], ...]
    sequence: dict[str, object]
    residue: dict[str, np.ndarray]


@dataclass(frozen=True)
class EnsemblePropertyTables:
    output_dir: Path
    sample: tuple[dict[str, object], ...]
    sequence: tuple[dict[str, object], ...]
    residue: tuple[dict[str, object], ...]
    convergence: tuple[dict[str, object], ...]
    failures: tuple[FailureRecord, ...]
    coverage: dict[str, object]
    paths: dict[str, Path]


def validate_canonical_property_publication(
    *,
    metadata: Mapping[str, object],
    coverage: Mapping[str, object],
    marker_artifact_sha256: Mapping[str, object],
) -> None:
    """Reject stale delta-G declarations and noncanonical Task 5 inventories."""

    for label, payload in (
        ("metadata", metadata),
        ("coverage", coverage),
        ("marker artifact hashes", marker_artifact_sha256),
    ):
        _reject_local_delta_g_declarations(
            payload,
            location=label,
            declaration_context=label == "coverage",
        )
    outputs = metadata.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != CANONICAL_PROPERTY_OUTPUT_KEYS:
        raise ValueError("property metadata output artifact inventory is noncanonical")
    if set(marker_artifact_sha256) != CANONICAL_PROPERTY_ARTIFACT_KEYS:
        raise ValueError("property marker artifact inventory is noncanonical")


def _reject_local_delta_g_declarations(
    value: object,
    *,
    location: str,
    declaration_context: bool = False,
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_local_delta_g_declarations(
                key,
                location=location,
                declaration_context=True,
            )
            _reject_local_delta_g_declarations(
                item,
                location=location,
                declaration_context=(
                    declaration_context
                    or isinstance(key, str)
                    and key.lower() in _DECLARATION_BEARING_PROPERTY_SECTIONS
                ),
            )
        return
    if isinstance(value, list | tuple):
        for item in value:
            _reject_local_delta_g_declarations(
                item,
                location=location,
                declaration_context=declaration_context,
            )
        return
    if (
        declaration_context
        and isinstance(value, str)
        and any(declaration in value.lower() for declaration in _FORBIDDEN_PROPERTY_DECLARATIONS)
    ):
        raise ValueError(
            f"unexpected property artifact in {location}: local delta G is outside scope"
        )


def validate_published_property_tables(
    tables: EnsemblePropertyTables,
    *,
    target_sample_indices: tuple[int, ...],
    bootstrap_replicates: int,
    random_seed: int,
    property_sampling_scope: str = "independent_supplemental_128",
    convergence_target_counts: tuple[int, ...] = (32, 64, 96, 128),
) -> str:
    """Validate the immutable Task 5 generation and return its content fingerprint."""

    marker_path = tables.paths.get("publication_marker_json")
    manifest_path = tables.paths.get("publication_manifest_json")
    if marker_path is None or manifest_path is None:
        raise ValueError("legacy Task 5 outputs are unpublished and must be regenerated")
    try:
        marker = json.loads(marker_path.read_text())
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("property publication metadata is invalid") from error
    if set(tables.paths) != CANONICAL_PROPERTY_TABLE_PATH_KEYS:
        raise ValueError("property table artifact inventory is noncanonical")
    artifact_paths = {
        name: path
        for name, path in tables.paths.items()
        if name not in {"publication_marker_json", "publication_manifest_json"}
    }
    metadata_path = artifact_paths.get("metadata_json")
    if metadata_path is None:
        raise ValueError("property generation lacks metadata")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("property generation metadata is invalid") from error
    source_identity = metadata.get("source_identity") if isinstance(metadata, dict) else None
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 2:
        raise ValueError("legacy Task 5 property schema requires regeneration")
    try:
        coverage = json.loads(artifact_paths["coverage_json"].read_text())
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise ValueError("property coverage metadata is invalid") from error
    artifact_sha256 = marker.get("artifact_sha256") if isinstance(marker, dict) else None
    if not isinstance(coverage, dict) or not isinstance(artifact_sha256, dict):
        raise ValueError("property publication inventories are invalid")
    validate_canonical_property_publication(
        metadata=metadata,
        coverage=coverage,
        marker_artifact_sha256=artifact_sha256,
    )
    scope_contract = _property_scope_contract(
        property_sampling_scope=property_sampling_scope,
        target_sample_indices=target_sample_indices,
        convergence_target_counts=convergence_target_counts,
    )
    if scope_contract and any(
        any(payload.get(name) != value for name, value in scope_contract.items())
        for payload in (metadata, coverage, marker, manifest)
    ):
        raise ValueError("property publication scope or analysis design is stale or invalid")
    if scope_contract:
        expected_seed_scope = {
            "alignment_reference_sample_index": 0,
            "alignment_reference_source": "original_structure_root",
            "target_sample_indices": list(target_sample_indices),
            "target_source": "original_structure_root",
            "reference_included_in_target_moments": True,
        }
        if metadata.get("seed_scope") != expected_seed_scope:
            raise ValueError("property publication seed scope is stale or invalid")
    _validate_convergence_publication(
        metadata,
        artifact_paths,
        target_sample_indices=target_sample_indices,
        bootstrap_replicates=bootstrap_replicates,
        random_seed=random_seed,
        convergence_target_counts=convergence_target_counts,
        include_successive_absolute_change=(property_sampling_scope != "shared_original_32"),
    )
    marker_without_fingerprint = {
        "schema_version": 2,
        "artifact": "expanded_ensemble_properties",
        "target_sample_indices": list(target_sample_indices),
        "bootstrap_replicates": bootstrap_replicates,
        "random_seed": random_seed,
        "source_identity": source_identity,
        "artifact_sha256": {
            name: _sha256_file(path) for name, path in sorted(artifact_paths.items())
        },
        **scope_contract,
    }
    if metadata.get("analysis_mode") == "same_trajectory_exploratory_32":
        marker_without_fingerprint["analysis_mode"] = "same_trajectory_exploratory_32"
        marker_without_fingerprint["predictor_target_overlap"] = list(range(32))
    fingerprint = hashlib.sha256(
        json.dumps(
            marker_without_fingerprint,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if marker != {**marker_without_fingerprint, "fingerprint": fingerprint}:
        raise ValueError("property generation marker or artifact hash is invalid")
    generation_root = marker_path.parent
    publication_root = manifest_path.parent
    if manifest != {
        "schema_version": 2,
        "artifact": "expanded_ensemble_properties",
        "fingerprint": fingerprint,
        "generation": generation_root.relative_to(publication_root).as_posix(),
        "marker_sha256": _sha256_file(marker_path),
        **scope_contract,
    }:
        raise ValueError("property publication manifest is stale or invalid")
    return fingerprint


def _validate_local_delta_g_publication(
    metadata: Mapping[str, object],
    artifact_paths: Mapping[str, Path],
    *,
    target_sample_indices: tuple[int, ...],
) -> None:
    for name in (
        "calibrated_effective_local_delta_g_csv",
        "calibrated_effective_local_delta_g_json",
        "coverage_json",
        "failures_json",
    ):
        if name not in artifact_paths:
            raise ValueError(f"Task 5 generation lacks required artifact: {name}")
    try:
        raw_rows = json.loads(artifact_paths["calibrated_effective_local_delta_g_json"].read_text())
        coverage = json.loads(artifact_paths["coverage_json"].read_text())
        failure_rows = json.loads(artifact_paths["failures_json"].read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("calibrated effective local delta G sidecars are invalid") from error
    if (
        not isinstance(raw_rows, list)
        or not isinstance(coverage, dict)
        or not isinstance(failure_rows, list)
    ):
        raise ValueError("calibrated effective local delta G sidecars have invalid roots")
    delta_coverage = coverage.get("calibrated_effective_local_delta_g")
    if not isinstance(delta_coverage, dict):
        raise ValueError("local delta G coverage summary is missing")
    production_scope = tuple(target_sample_indices) == tuple(range(32, 160))
    if not production_scope:
        expected_nonproduction_coverage = {
            "applicable": False,
            "reason": "requires_exact_supplemental_samples_32_159",
            "sampling_scope": "not_applicable",
            "sequence_rows": [],
        }
        if raw_rows or delta_coverage != expected_nonproduction_coverage:
            raise ValueError("non-expanded Task 5 scope has invalid local delta G rows")
        return
    calibration_identity = metadata.get("calibration_identity")
    if not isinstance(calibration_identity, dict):
        raise ValueError("Task 5 calibration identity is missing")
    if set(calibration_identity) != {
        "path",
        "artifact_sha256",
        "calibration_fingerprint",
        "input_manifest_sha256",
        "boundary",
    } or not isinstance(calibration_identity["path"], str):
        raise ValueError("Task 5 calibration identity schema is invalid")
    loaded = LocalQCalibration.load(Path(calibration_identity["path"]))
    expected_identity = {
        "path": str(loaded.source_path.resolve(strict=False)),
        "artifact_sha256": loaded.artifact_sha256,
        "calibration_fingerprint": loaded.calibration_fingerprint,
        "input_manifest_sha256": loaded.input_manifest_sha256,
        "boundary": loaded.boundary,
    }
    if calibration_identity != expected_identity:
        raise ValueError("Task 5 calibration artifact or fingerprint is stale")
    source_identity = metadata.get("source_identity")
    fasta_identity = source_identity.get("fasta") if isinstance(source_identity, dict) else None
    fasta_records = fasta_identity.get("records") if isinstance(fasta_identity, dict) else None
    if not isinstance(fasta_records, list) or any(
        not isinstance(record, list)
        or len(record) != 2
        or not isinstance(record[0], str)
        or not isinstance(record[1], str)
        or not record[0]
        or not record[1]
        for record in fasta_records
    ):
        raise ValueError("Task 5 lacks canonical FASTA residue identities")
    expected_identities = tuple(
        (record[0], residue_index)
        for record in fasta_records
        for residue_index in range(len(record[1]))
    )
    seen: set[tuple[str, int]] = set()
    observed_identities: list[tuple[str, int]] = []
    for row in raw_rows:
        if not isinstance(row, dict) or set(row) != set(_LOCAL_DELTA_G_FIELDS):
            raise ValueError("local delta G row schema is invalid")
        if (
            not isinstance(row["sequence_id"], str)
            or not isinstance(row["residue_index"], int)
            or isinstance(row["residue_index"], bool)
        ):
            raise ValueError("local delta G residue identity types are invalid")
        identity = (row["sequence_id"], row["residue_index"])
        if identity in seen:
            raise ValueError("duplicate local delta G residue identity")
        seen.add(identity)
        observed_identities.append(identity)
        if (
            row["property_name"] != "calibrated_effective_local_delta_g"
            or row["sampling_scope"] != "supplemental_samples_32_159"
            or row["units"] != "kcal/mol"
            or row["estimator_kind"] != "calibrated_effective_estimator"
            or row["estimator_definition_id"] != "soft_local_q_expit20_boundary_300K_log_odds_v1"
            or row["sensitivity_grid_id"] != _local_delta_g_sensitivity_grid_id()
            or row["calibration_fingerprint"] != loaded.calibration_fingerprint
            or row["calibration_artifact_sha256"] != loaded.artifact_sha256
            or row["calibration_input_manifest_sha256"] != loaded.input_manifest_sha256
            or row["status"] not in {"ok", "censored", "not_estimable"}
        ):
            raise ValueError("local delta G row identity or provenance is invalid")
        observed_indices = tuple(int(value) for value in row["sample_indices"])
        if observed_indices != tuple(range(32, 160)):
            raise ValueError("local delta G row sample scope is invalid")
        if (
            not isinstance(row["sample_count"], int)
            or isinstance(row["sample_count"], bool)
            or row["sample_count"] < 0
            or row["sample_count"] > len(observed_indices)
        ):
            raise ValueError("local delta G row sample count is invalid")
        status = row["status"]
        if (
            (
                status == "ok"
                and (
                    not isinstance(row["value"], int | float)
                    or isinstance(row["value"], bool)
                    or not math.isfinite(float(row["value"]))
                    or row["reason"] is not None
                    or row["censored"] is not False
                    or row["censoring_bound_kcal_mol"] is not None
                    or row["censoring_direction"] is not None
                )
            )
            or (
                status == "censored"
                and (
                    row["value"] is not None
                    or row["reason"] is not None
                    or row["censored"] is not True
                    or not isinstance(
                        row["censoring_bound_kcal_mol"],
                        int | float,
                    )
                    or isinstance(row["censoring_bound_kcal_mol"], bool)
                    or not math.isfinite(float(row["censoring_bound_kcal_mol"]))
                    or row["censoring_direction"]
                    not in {"less_than_or_equal", "greater_than_or_equal"}
                )
            )
            or (
                status == "not_estimable"
                and (
                    row["value"] is not None
                    or not isinstance(row["reason"], str)
                    or not row["reason"]
                    or row["censored"] is not False
                    or row["censoring_bound_kcal_mol"] is not None
                    or row["censoring_direction"] is not None
                )
            )
        ):
            raise ValueError("local delta G row status fields are inconsistent")
    if tuple(observed_identities) != expected_identities:
        raise ValueError("local delta G rows do not match the exact canonical residue ID set/order")
    structural_failure_requirements = {
        "missing_seed0_reference": ("alignment_reference", None),
        "reference_sequence_length_mismatch": (
            "alignment_reference",
            "reference_sequence_length_mismatch",
        ),
        "no_usable_target_structures": ("target", None),
        "no_residue_aligned_target_structures": ("target", None),
        "incomplete_structure_coverage": ("target", None),
        "alignment_failed": ("alignment", "alignment_failed"),
        "calibration_estimation_failed": (
            "calibrated_effective_local_delta_g",
            "calibration_estimation_failed",
        ),
    }
    for row in raw_rows:
        requirement = structural_failure_requirements.get(row["reason"])
        if requirement is None:
            continue
        required_role, required_reason = requirement
        if not any(
            isinstance(failure, dict)
            and failure.get("sequence_id") == row["sequence_id"]
            and failure.get("role") == required_role
            and (required_reason is None or failure.get("reason") == required_reason)
            for failure in failure_rows
        ):
            raise ValueError("local delta G structural status lacks a bound failure record")
    expected_sequence_coverage: list[dict[str, object]] = []
    row_offset = 0
    for sequence_id, sequence in fasta_records:
        sequence_rows = raw_rows[row_offset : row_offset + len(sequence)]
        row_offset += len(sequence)
        expected_sequence_coverage.append(
            {
                "sequence_id": sequence_id,
                **_local_delta_g_coverage_from_rows(
                    sequence_rows,
                    requested_residues=len(sequence),
                    calibration=loaded,
                ),
            }
        )
    expected_coverage = _json_safe(
        _aggregate_local_delta_g_coverage(
            raw_rows,
            expected_sequence_coverage,
        )
    )
    observed_serialized = json.dumps(
        delta_coverage,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    expected_serialized = json.dumps(
        expected_coverage,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if observed_serialized != expected_serialized:
        raise ValueError("local delta G coverage does not exactly match canonical published rows")


def _validate_convergence_publication(
    metadata: Mapping[str, object],
    artifact_paths: Mapping[str, Path],
    *,
    target_sample_indices: tuple[int, ...],
    bootstrap_replicates: int,
    random_seed: int,
    convergence_target_counts: tuple[int, ...] = (32, 64, 96, 128),
    include_successive_absolute_change: bool = True,
) -> None:
    for name in ("sample_json", "convergence_json"):
        if name not in artifact_paths:
            raise ValueError(f"Task 5 generation lacks required convergence artifact: {name}")
    try:
        sample_rows = json.loads(artifact_paths["sample_json"].read_text())
        convergence_rows = json.loads(artifact_paths["convergence_json"].read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Task 5 convergence sidecars are invalid") from error
    if not isinstance(sample_rows, list) or not isinstance(convergence_rows, list):
        raise ValueError("Task 5 convergence sidecars have invalid roots")
    if metadata.get("analysis_mode") == "same_trajectory_exploratory_32":
        coverage = metadata.get("coverage")
        expected = {
            "applicable": False,
            "reason": "same_trajectory_exploratory_32_has_exactly_32_samples",
        }
        if (
            convergence_rows
            or not isinstance(coverage, Mapping)
            or coverage.get("sample_count_convergence") != expected
        ):
            raise ValueError("exploratory property convergence exclusion is invalid")
        return

    source_identity = metadata.get("source_identity")
    fasta_identity = source_identity.get("fasta") if isinstance(source_identity, dict) else None
    fasta_records = fasta_identity.get("records") if isinstance(fasta_identity, dict) else None
    if isinstance(fasta_records, list) and all(
        isinstance(record, list) and len(record) == 2 and isinstance(record[0], str) and record[0]
        for record in fasta_records
    ):
        sequence_ids = tuple(record[0] for record in fasta_records)
    else:
        sequence_ids = tuple(
            dict.fromkeys(
                str(row.get("sequence_id"))
                for rows in (sample_rows, convergence_rows)
                for row in rows
                if isinstance(row, dict) and isinstance(row.get("sequence_id"), str)
            )
        )

    sample_rows_by_sequence: dict[str, list[dict[str, object]]] = {
        sequence_id: [] for sequence_id in sequence_ids
    }
    for row in sample_rows:
        if not isinstance(row, dict) or row.get("sequence_id") not in sample_rows_by_sequence:
            raise ValueError("Task 5 convergence sample identities are invalid")
        sample_rows_by_sequence[str(row["sequence_id"])].append(row)
    expected_rows = [
        row
        for sequence_id in sequence_ids
        for row in compute_convergence_rows(
            sequence_id=sequence_id,
            sample_rows=sample_rows_by_sequence[sequence_id],
            target_sample_indices=target_sample_indices,
            bootstrap_replicates=bootstrap_replicates,
            random_seed=random_seed,
            convergence_target_counts=convergence_target_counts,
            include_successive_absolute_change=include_successive_absolute_change,
        )
    ]
    if json.dumps(
        convergence_rows,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) != json.dumps(
        _json_safe(expected_rows),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ):
        raise ValueError(
            "Task 5 convergence rows, intervals, status, order, or fingerprints are inconsistent"
        )


@dataclass(frozen=True)
class LocalQObservationSeries:
    """One parameterization's exactly paired complete-structure observations."""

    sample_indices: tuple[int, ...]
    q_values: np.ndarray
    reference_contact_count: int


@dataclass(frozen=True)
class LocalQSensitivityGrid:
    """Complete predeclared local-Q robustness grid for one residue."""

    entries: tuple[
        tuple[tuple[float, float, float], LocalQObservationSeries],
        ...,
    ]

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[tuple[float, float, float], LocalQObservationSeries],
        *,
        primary_sample_indices: tuple[int, ...],
    ) -> LocalQSensitivityGrid:
        expected_indices = _validate_sample_indices(
            primary_sample_indices,
            "primary",
        )
        normalized: dict[tuple[float, float, float], LocalQObservationSeries] = {}
        for raw_parameters, raw_series in values.items():
            if not isinstance(raw_parameters, tuple) or len(raw_parameters) != 3:
                raise ValueError("extra or malformed sensitivity parameter key")
            parameters = tuple(float(value) for value in raw_parameters)
            if parameters in normalized:
                raise ValueError(f"duplicate sensitivity parameter key: {parameters}")
            if not isinstance(raw_series, LocalQObservationSeries):
                raise ValueError(
                    f"sensitivity grid entry {parameters} must be a " "LocalQObservationSeries"
                )
            normalized[parameters] = raw_series
        expected = set(LOCAL_Q_SENSITIVITY_PARAMETERS)
        observed = set(normalized)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing:
            raise ValueError(f"missing sensitivity grid entries: {missing}")
        if extra:
            raise ValueError(f"extra sensitivity grid entries: {extra}")

        entries = []
        for parameters in LOCAL_Q_SENSITIVITY_PARAMETERS:
            raw_series = normalized[parameters]
            observed_indices = _validate_sample_indices(
                raw_series.sample_indices,
                f"sensitivity grid entry {parameters}",
            )
            if len(observed_indices) != len(expected_indices):
                raise ValueError(
                    f"omitted sensitivity structure IDs for {parameters}: "
                    f"expected {len(expected_indices)}, found {len(observed_indices)}"
                )
            if observed_indices != expected_indices:
                if set(observed_indices) == set(expected_indices):
                    raise ValueError(f"permuted sensitivity structure IDs for {parameters}")
                raise ValueError(f"mismatched sensitivity structure IDs for {parameters}")
            vector = np.asarray(raw_series.q_values, dtype=np.float64)
            if vector.ndim != 1 or len(vector) != len(expected_indices):
                raise ValueError(
                    "misaligned sensitivity grid entry "
                    f"{parameters}: expected {len(expected_indices)} complete structures"
                )
            if not np.isfinite(vector).all():
                raise ValueError(f"nonfinite sensitivity grid entry: {parameters}")
            if np.any((vector < 0.0) | (vector > 1.0)):
                raise ValueError(f"sensitivity Q values outside [0, 1]: {parameters}")
            contact_count = _validate_reference_contact_count(
                raw_series.reference_contact_count,
                f"sensitivity grid entry {parameters}",
            )
            frozen = vector.copy()
            frozen.setflags(write=False)
            entries.append(
                (
                    parameters,
                    LocalQObservationSeries(
                        sample_indices=observed_indices,
                        q_values=frozen,
                        reference_contact_count=contact_count,
                    ),
                )
            )
        return cls(entries=tuple(entries))

    def series_for(
        self,
        parameters: tuple[float, float, float],
    ) -> LocalQObservationSeries:
        for candidate, series in self.entries:
            if candidate == parameters:
                return series
        raise KeyError(parameters)


@dataclass(frozen=True)
class _TestLocalQCalibration:
    boundary: float


@dataclass(frozen=True, init=False)
class LocalQCalibration:
    """Load-only frozen mapping from local-Q to an effective ordered-state score."""

    boundary: float
    candidate_count: int
    candidate_hashes: tuple[tuple[str, dict[str, object]], ...]
    code_hashes: tuple[tuple[str, str], ...]
    loco_folds: tuple[dict[str, object], ...]
    loco_mean_balanced_accuracy: float
    all_data_refit_metrics: tuple[dict[str, object], ...]
    input_manifest_sha256: str
    calibration_fingerprint: str
    source_path: Path
    artifact_sha256: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("LocalQCalibration is load-only; use LocalQCalibration.load(path)")

    @classmethod
    def for_test(cls, *, boundary: float) -> _TestLocalQCalibration:
        """Build an explicitly non-production calibration for analytic unit tests."""

        _validate_local_q_boundary(boundary)
        return _TestLocalQCalibration(boundary=float(boundary))

    @classmethod
    def load(cls, path: str | Path) -> LocalQCalibration:
        if cls is not LocalQCalibration:
            raise TypeError("LocalQCalibration.load requires the exact LocalQCalibration type")
        calibration_path = Path(path).resolve()
        try:
            artifact_bytes = calibration_path.read_bytes()
        except FileNotFoundError:
            raise
        except OSError as error:
            raise ValueError(f"invalid local-Q calibration artifact: {calibration_path}") from error
        payload = _load_json_object_without_duplicates(
            artifact_bytes,
            calibration_path,
        )
        return cls._from_frozen_payload(
            payload,
            source_path=calibration_path,
            artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        )

    @classmethod
    def _from_frozen_payload(
        cls,
        payload: Mapping[str, object],
        *,
        source_path: Path,
        artifact_sha256: str,
    ) -> LocalQCalibration:
        required = {
            "schema_version",
            "artifact_type",
            "status",
            "boundary",
            "boundary_candidates",
            "candidate_count",
            "candidate_hashes",
            "code_hashes",
            "loco_metrics",
            "all_data_refit",
            "input_manifest_sha256",
            "calibration_fingerprint",
        }
        if set(payload) != required:
            missing = sorted(required - set(payload))
            extra = sorted(set(payload) - required)
            raise ValueError(
                f"incomplete local-Q calibration artifact; missing={missing}, extra={extra}"
            )
        if payload["schema_version"] != 1:
            raise ValueError("unsupported local-Q calibration schema")
        if payload["artifact_type"] != "effective_local_delta_g_boundary":
            raise ValueError("invalid local-Q calibration artifact type")
        if payload["status"] != "frozen":
            raise ValueError("local-Q calibration artifact is not frozen")
        fingerprint = str(payload["calibration_fingerprint"])
        if fingerprint != local_q_calibration_fingerprint(payload):
            raise ValueError("local-Q calibration fingerprint mismatch")

        boundary = _strict_finite_float(payload["boundary"], "boundary")
        _validate_local_q_boundary(boundary)
        raw_candidates = payload["boundary_candidates"]
        if not isinstance(raw_candidates, list) or tuple(raw_candidates) != (
            LOCAL_Q_BOUNDARY_CANDIDATES
        ):
            raise ValueError("local-Q calibration boundary candidates mismatch")
        candidate_count = _strict_positive_int(payload["candidate_count"], "candidate_count")
        if candidate_count < 10:
            raise ValueError("local-Q calibration requires at least ten candidates")
        candidate_hashes = _validate_candidate_hashes(
            payload["candidate_hashes"],
            candidate_count=candidate_count,
        )
        code_hashes = _validate_code_hashes(payload["code_hashes"])
        if dict(code_hashes)["local_q_algorithm_sha256"] != local_q_definition_fingerprint():
            raise ValueError("local-Q calibration code fingerprint mismatch")
        calibration_script_path = (
            Path(__file__).resolve().parents[3] / "scripts" / "calibrate_local_q_boundary.py"
        )
        if not calibration_script_path.is_file() or dict(code_hashes)[
            "calibration_script_sha256"
        ] != _sha256_path(calibration_script_path):
            raise ValueError("local-Q calibration script fingerprint mismatch")
        all_data_metrics = _validate_all_data_refit(
            payload["all_data_refit"],
            expected_candidate_ids={candidate_id for candidate_id, _ in candidate_hashes},
        )
        selected_boundary = _select_boundary_from_metric_rows(all_data_metrics)
        if not math.isclose(selected_boundary, boundary, abs_tol=1e-12):
            raise ValueError("local-Q calibration all-data refit boundary mismatch")
        loco_folds, loco_mean = _validate_loco_metrics(
            payload["loco_metrics"],
            all_data_metrics=all_data_metrics,
            expected_candidate_ids={candidate_id for candidate_id, _ in candidate_hashes},
        )
        input_manifest_sha256 = _validate_sha256(
            payload["input_manifest_sha256"],
            "input_manifest_sha256",
        )
        calibration = object.__new__(cls)
        values = {
            "boundary": boundary,
            "candidate_count": candidate_count,
            "candidate_hashes": candidate_hashes,
            "code_hashes": code_hashes,
            "loco_folds": loco_folds,
            "loco_mean_balanced_accuracy": loco_mean,
            "all_data_refit_metrics": all_data_metrics,
            "input_manifest_sha256": input_manifest_sha256,
            "calibration_fingerprint": fingerprint,
            "source_path": source_path,
            "artifact_sha256": _validate_sha256(
                artifact_sha256,
                "artifact_sha256",
            ),
        }
        for name, value in values.items():
            object.__setattr__(calibration, name, value)
        return calibration


def _revalidate_local_q_calibration(
    calibration: LocalQCalibration,
) -> LocalQCalibration:
    if type(calibration) is not LocalQCalibration:
        raise TypeError("production calibration requires the exact LocalQCalibration type")
    try:
        artifact_bytes = calibration.source_path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"local-Q calibration source is unavailable: {calibration.source_path}"
        ) from error
    current_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if current_sha256 != calibration.artifact_sha256:
        raise ValueError("local-Q calibration artifact bytes changed after loading")
    payload = _load_json_object_without_duplicates(
        artifact_bytes,
        calibration.source_path,
    )
    current = LocalQCalibration._from_frozen_payload(
        payload,
        source_path=calibration.source_path,
        artifact_sha256=current_sha256,
    )
    if current.calibration_fingerprint != calibration.calibration_fingerprint:
        raise ValueError("local-Q calibration object fingerprint mismatch")
    return current


@dataclass(frozen=True)
class LocalDeltaGResult:
    """Reliability-filtered calibrated effective local conformational contrast."""

    status: str
    delta_g_kcal_mol: float | None
    failure_reason: str | None
    p_ordered: float | None
    effective_ordered_observations: float
    effective_disordered_observations: float
    bootstrap_ci_low_kcal_mol: float | None
    bootstrap_ci_high_kcal_mol: float | None
    bootstrap_ci_width_kcal_mol: float | None
    bootstrap_replicates: int
    reference_contact_count: int
    sensitivity_evaluated_count: int
    sensitivity_estimable_count: int
    censoring: str | None
    censoring_bound_kcal_mol: float | None

    def to_json_dict(self) -> dict[str, object]:
        return _json_safe(asdict(self))


def local_q_definition_fingerprint() -> str:
    """Hash the complete implementation module for deliberately broad invalidation."""

    return _sha256_path(Path(__file__).resolve())


def local_q_calibration_fingerprint(payload: Mapping[str, object]) -> str:
    fingerprint_payload = {
        key: value for key, value in payload.items() if key != "calibration_fingerprint"
    }
    return hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object_without_duplicates(
    data: bytes,
    path: Path,
) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        payload = json.loads(data, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return payload


def estimate_effective_local_delta_g(
    q_values: np.ndarray,
    reference_contact_count: int,
    calibration: LocalQCalibration | _TestLocalQCalibration,
    bootstrap_replicates: int,
    random_seed: int,
    *,
    sample_indices: tuple[int, ...] | None = None,
    sensitivity_q_values: (
        LocalQSensitivityGrid | Mapping[tuple[float, float, float], LocalQObservationSeries] | None
    ) = None,
) -> LocalDeltaGResult:
    """Estimate a calibrated effective local delta G with reliability censoring."""

    q = _validate_q_vector(q_values, "primary")
    primary_contact_count = _validate_reference_contact_count(
        reference_contact_count,
        "primary",
    )
    if isinstance(bootstrap_replicates, bool) or not isinstance(
        bootstrap_replicates, int | np.integer
    ):
        raise ValueError("bootstrap_replicates must be an integer")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int | np.integer):
        raise ValueError("random_seed must be an integer")
    if type(calibration) is LocalQCalibration:
        calibration = _revalidate_local_q_calibration(calibration)
        test_only = False
    elif type(calibration) is _TestLocalQCalibration:
        test_only = True
    elif isinstance(calibration, LocalQCalibration):
        raise TypeError("production calibration requires the exact LocalQCalibration type")
    else:
        raise TypeError("calibration must be loaded or created by for_test")
    if not test_only and bootstrap_replicates != 2_000:
        raise ValueError("production estimate requires exactly 2000 bootstrap replicates")
    if sample_indices is None:
        if not test_only:
            raise ValueError("production estimate requires explicit complete-structure IDs")
        primary_sample_indices = tuple(range(len(q)))
    else:
        primary_sample_indices = _validate_sample_indices(sample_indices, "primary")
        if len(primary_sample_indices) != len(q):
            raise ValueError("primary sample IDs are misaligned with q_values")
    if not test_only and sensitivity_q_values is None:
        raise ValueError("production estimate requires the complete sensitivity Q grid")

    grid = None
    if sensitivity_q_values is not None:
        raw_sensitivity_values = (
            sensitivity_q_values.entries
            if isinstance(sensitivity_q_values, LocalQSensitivityGrid)
            else sensitivity_q_values.items()
        )
        sensitivity_mapping: dict[
            tuple[float, float, float],
            LocalQObservationSeries,
        ] = {}
        for parameters, values in raw_sensitivity_values:
            if parameters in sensitivity_mapping:
                raise ValueError(f"duplicate sensitivity parameter key: {parameters}")
            sensitivity_mapping[parameters] = values
        grid = LocalQSensitivityGrid.from_mapping(
            sensitivity_mapping,
            primary_sample_indices=primary_sample_indices,
        )
        primary_series = grid.series_for(PRIMARY_LOCAL_Q_PARAMETERS)
        if not np.array_equal(primary_series.q_values, q):
            raise ValueError("primary sensitivity grid vector does not match q_values")
        if primary_series.reference_contact_count != primary_contact_count:
            raise ValueError("primary reference contact count does not match sensitivity grid")

    primary = _local_delta_g_diagnostics(
        q,
        boundary=calibration.boundary,
        bootstrap_replicates=int(bootstrap_replicates),
        random_seed=int(random_seed),
    )
    sensitivity_evaluated_count = 0
    sensitivity_estimable_count = 0
    if grid is not None:
        primary_sign = _stable_sign(float(primary["delta_g"]))
        primary_estimable = _local_delta_g_is_estimable(
            primary,
            reference_contact_count=primary_contact_count,
        )
        stable = True
        for parameter_index, (parameters, series) in enumerate(grid.entries):
            diagnostics = (
                primary
                if parameters == PRIMARY_LOCAL_Q_PARAMETERS
                else _local_delta_g_diagnostics(
                    series.q_values,
                    boundary=calibration.boundary,
                    bootstrap_replicates=int(bootstrap_replicates),
                    random_seed=_stable_bootstrap_seed(
                        int(random_seed),
                        "effective_local_delta_g_sensitivity",
                        parameter_index,
                        "grid",
                    ),
                )
            )
            sensitivity_evaluated_count += 1
            estimable = _local_delta_g_is_estimable(
                diagnostics,
                reference_contact_count=series.reference_contact_count,
            )
            sensitivity_estimable_count += int(estimable)
            if (
                estimable != primary_estimable
                or _stable_sign(float(diagnostics["delta_g"])) != primary_sign
            ):
                stable = False
        if not stable:
            return _local_delta_g_failure(
                primary,
                reason="sensitivity_unstable",
                bootstrap_replicates=int(bootstrap_replicates),
                reference_contact_count=primary_contact_count,
                sensitivity_evaluated_count=sensitivity_evaluated_count,
                sensitivity_estimable_count=sensitivity_estimable_count,
            )

    primary_failure_reason = _local_delta_g_primary_failure_reason(
        primary,
        reference_contact_count=primary_contact_count,
    )
    if primary_failure_reason is not None:
        return _local_delta_g_failure(
            primary,
            reason=primary_failure_reason,
            bootstrap_replicates=int(bootstrap_replicates),
            reference_contact_count=primary_contact_count,
            sensitivity_evaluated_count=sensitivity_evaluated_count,
            sensitivity_estimable_count=sensitivity_estimable_count,
        )

    delta_g = float(primary["delta_g"])
    if bool(primary["censored"]):
        censoring = "below_practical_range" if delta_g < 0 else "above_practical_range"
        return LocalDeltaGResult(
            status="censored",
            delta_g_kcal_mol=None,
            failure_reason=None,
            p_ordered=float(primary["p_ordered"]),
            effective_ordered_observations=float(primary["effective_ordered"]),
            effective_disordered_observations=float(primary["effective_disordered"]),
            bootstrap_ci_low_kcal_mol=float(primary["ci_low"]),
            bootstrap_ci_high_kcal_mol=float(primary["ci_high"]),
            bootstrap_ci_width_kcal_mol=float(primary["ci_width"]),
            bootstrap_replicates=int(bootstrap_replicates),
            reference_contact_count=primary_contact_count,
            sensitivity_evaluated_count=sensitivity_evaluated_count,
            sensitivity_estimable_count=sensitivity_estimable_count,
            censoring=censoring,
            censoring_bound_kcal_mol=(-1.5 if delta_g < 0 else 1.5),
        )
    return LocalDeltaGResult(
        status="ok",
        delta_g_kcal_mol=delta_g,
        failure_reason=None,
        p_ordered=float(primary["p_ordered"]),
        effective_ordered_observations=float(primary["effective_ordered"]),
        effective_disordered_observations=float(primary["effective_disordered"]),
        bootstrap_ci_low_kcal_mol=float(primary["ci_low"]),
        bootstrap_ci_high_kcal_mol=float(primary["ci_high"]),
        bootstrap_ci_width_kcal_mol=float(primary["ci_width"]),
        bootstrap_replicates=int(bootstrap_replicates),
        reference_contact_count=primary_contact_count,
        sensitivity_evaluated_count=sensitivity_evaluated_count,
        sensitivity_estimable_count=sensitivity_estimable_count,
        censoring=None,
        censoring_bound_kcal_mol=None,
    )


def build_calibrated_effective_local_delta_g_rows(
    *,
    sequence_id: str,
    sample_indices: tuple[int, ...],
    q_by_parameter: Mapping[tuple[float, float, float], np.ndarray],
    reference_contact_counts_by_parameter: Mapping[tuple[float, float, float], np.ndarray],
    calibration: object,
    random_seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build compact residue rows from the frozen calibration and full sensitivity grid."""

    expected_samples = tuple(range(32, 160))
    observed_samples = _validate_sample_indices(sample_indices, "local delta G")
    if observed_samples != expected_samples:
        raise ValueError(
            "calibrated effective local delta G requires exact supplemental samples 32..159"
        )
    if not sequence_id:
        raise ValueError("local delta G sequence_id must be nonempty")
    expected_parameters = set(LOCAL_Q_SENSITIVITY_PARAMETERS)
    if set(q_by_parameter) != expected_parameters:
        raise ValueError("local delta G Q grid must contain the exact 27 sensitivity entries")
    if set(reference_contact_counts_by_parameter) != expected_parameters:
        raise ValueError(
            "local delta G contact-count grid must contain the exact 27 sensitivity entries"
        )
    q_arrays: dict[tuple[float, float, float], np.ndarray] = {}
    contact_arrays: dict[tuple[float, float, float], np.ndarray] = {}
    residue_count: int | None = None
    for parameters in LOCAL_Q_SENSITIVITY_PARAMETERS:
        q_values = np.asarray(q_by_parameter[parameters], dtype=np.float64)
        contacts = np.asarray(
            reference_contact_counts_by_parameter[parameters],
            dtype=np.int64,
        )
        if (
            q_values.ndim != 2
            or q_values.shape[0] != len(expected_samples)
            or not np.isfinite(q_values).all()
            or np.any((q_values < 0.0) | (q_values > 1.0))
        ):
            raise ValueError(f"local delta G Q grid entry is invalid: {parameters}")
        if contacts.ndim != 1 or len(contacts) != q_values.shape[1] or np.any(contacts < 0):
            raise ValueError(f"local delta G contact-count entry is invalid: {parameters}")
        if residue_count is None:
            residue_count = q_values.shape[1]
        elif q_values.shape[1] != residue_count:
            raise ValueError("local delta G sensitivity entries are residue-misaligned")
        q_arrays[parameters] = q_values
        contact_arrays[parameters] = contacts
    if residue_count is None:
        raise ValueError("local delta G grid is empty")

    calibration_fingerprint = _validate_sha256(
        getattr(calibration, "calibration_fingerprint", None),
        "calibration_fingerprint",
    )
    calibration_artifact_sha256 = _validate_sha256(
        getattr(calibration, "artifact_sha256", None),
        "calibration_artifact_sha256",
    )
    calibration_input_manifest_sha256 = _validate_sha256(
        getattr(calibration, "input_manifest_sha256", None),
        "calibration_input_manifest_sha256",
    )
    sensitivity_grid_id = _local_delta_g_sensitivity_grid_id()
    rows: list[dict[str, object]] = []
    for residue_index in range(residue_count):
        primary_q = q_arrays[PRIMARY_LOCAL_Q_PARAMETERS][:, residue_index]
        primary_contacts = int(contact_arrays[PRIMARY_LOCAL_Q_PARAMETERS][residue_index])
        sensitivity = LocalQSensitivityGrid.from_mapping(
            {
                parameters: LocalQObservationSeries(
                    sample_indices=observed_samples,
                    q_values=q_arrays[parameters][:, residue_index],
                    reference_contact_count=int(contact_arrays[parameters][residue_index]),
                )
                for parameters in LOCAL_Q_SENSITIVITY_PARAMETERS
            },
            primary_sample_indices=observed_samples,
        )
        result = estimate_effective_local_delta_g(
            q_values=primary_q,
            reference_contact_count=primary_contacts,
            calibration=calibration,  # type: ignore[arg-type]
            bootstrap_replicates=2_000,
            random_seed=_stable_bootstrap_seed(
                random_seed,
                sequence_id,
                residue_index,
                "calibrated_effective_local_delta_g",
            ),
            sample_indices=observed_samples,
            sensitivity_q_values=sensitivity,
        )
        censoring_direction = {
            "below_practical_range": "less_than_or_equal",
            "above_practical_range": "greater_than_or_equal",
            None: None,
        }[result.censoring]
        rows.append(
            {
                "property_name": "calibrated_effective_local_delta_g",
                "sequence_id": sequence_id,
                "residue_index": residue_index,
                "sampling_scope": "supplemental_samples_32_159",
                "sample_indices": observed_samples,
                "sample_count": len(observed_samples),
                "units": "kcal/mol",
                "status": result.status,
                "reason": result.failure_reason,
                "value": result.delta_g_kcal_mol,
                "censored": result.status == "censored",
                "censoring_bound_kcal_mol": result.censoring_bound_kcal_mol,
                "censoring_direction": censoring_direction,
                "p_ordered": result.p_ordered,
                "effective_ordered_observations": (result.effective_ordered_observations),
                "effective_disordered_observations": (result.effective_disordered_observations),
                "bootstrap_ci_low_kcal_mol": result.bootstrap_ci_low_kcal_mol,
                "bootstrap_ci_high_kcal_mol": result.bootstrap_ci_high_kcal_mol,
                "bootstrap_ci_width_kcal_mol": result.bootstrap_ci_width_kcal_mol,
                "bootstrap_replicates": result.bootstrap_replicates,
                "reference_contact_count": result.reference_contact_count,
                "sensitivity_evaluated_count": result.sensitivity_evaluated_count,
                "sensitivity_estimable_count": result.sensitivity_estimable_count,
                "estimator_kind": "calibrated_effective_estimator",
                "estimator_definition_id": ("soft_local_q_expit20_boundary_300K_log_odds_v1"),
                "sensitivity_grid_id": sensitivity_grid_id,
                "calibration_fingerprint": calibration_fingerprint,
                "calibration_artifact_sha256": calibration_artifact_sha256,
                "calibration_input_manifest_sha256": (calibration_input_manifest_sha256),
            }
        )
    coverage = _local_delta_g_coverage_from_rows(
        rows,
        requested_residues=residue_count,
        calibration=calibration,
    )
    return rows, coverage


def _local_delta_g_sensitivity_grid_id() -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "parameters": LOCAL_Q_SENSITIVITY_PARAMETERS,
                "ordered_score_slope": LOCAL_Q_ORDERED_SCORE_SLOPE,
                "temperature_k": LOCAL_DELTA_G_TEMPERATURE_K,
                "practical_limit_kcal_mol": PRACTICAL_LOCAL_DELTA_G_LIMIT_KCAL_MOL,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _local_delta_g_coverage_from_rows(
    rows: list[dict[str, object]],
    *,
    requested_residues: int,
    calibration: object,
) -> dict[str, object]:
    reasons: dict[str, int] = {}
    for row in rows:
        reason = row["reason"]
        if isinstance(reason, str):
            reasons[reason] = reasons.get(reason, 0) + 1
    coverage = {
        "sampling_scope": "supplemental_samples_32_159",
        "sample_indices": tuple(range(32, 160)),
        "requested_residues": requested_residues,
        "published_residues": len(rows),
        "estimable_residues": sum(row["status"] == "ok" for row in rows),
        "censored_residues": sum(row["status"] == "censored" for row in rows),
        "not_estimable_residues": sum(row["status"] == "not_estimable" for row in rows),
        "not_estimable_reason_counts": reasons,
        "partial": (
            len(rows) != requested_residues or any(row["status"] == "not_estimable" for row in rows)
        ),
        "calibration_fingerprint": calibration.calibration_fingerprint,
        "sensitivity_grid_id": _local_delta_g_sensitivity_grid_id(),
    }
    return coverage


def _unavailable_local_delta_g_rows(
    *,
    sequence_id: str,
    residue_count: int,
    observed_sample_indices: tuple[int, ...],
    calibration: LocalQCalibration,
    reason: str,
) -> list[dict[str, object]]:
    return [
        {
            "property_name": "calibrated_effective_local_delta_g",
            "sequence_id": sequence_id,
            "residue_index": residue_index,
            "sampling_scope": "supplemental_samples_32_159",
            "sample_indices": tuple(range(32, 160)),
            "sample_count": len(observed_sample_indices),
            "units": "kcal/mol",
            "status": "not_estimable",
            "reason": reason,
            "value": None,
            "censored": False,
            "censoring_bound_kcal_mol": None,
            "censoring_direction": None,
            "p_ordered": None,
            "effective_ordered_observations": 0.0,
            "effective_disordered_observations": 0.0,
            "bootstrap_ci_low_kcal_mol": None,
            "bootstrap_ci_high_kcal_mol": None,
            "bootstrap_ci_width_kcal_mol": None,
            "bootstrap_replicates": 2_000,
            "reference_contact_count": 0,
            "sensitivity_evaluated_count": 0,
            "sensitivity_estimable_count": 0,
            "estimator_kind": "calibrated_effective_estimator",
            "estimator_definition_id": ("soft_local_q_expit20_boundary_300K_log_odds_v1"),
            "sensitivity_grid_id": _local_delta_g_sensitivity_grid_id(),
            "calibration_fingerprint": calibration.calibration_fingerprint,
            "calibration_artifact_sha256": calibration.artifact_sha256,
            "calibration_input_manifest_sha256": calibration.input_manifest_sha256,
        }
        for residue_index in range(residue_count)
    ]


def _aggregate_local_delta_g_coverage(
    rows: list[dict[str, object]],
    sequence_rows: list[dict[str, object]],
) -> dict[str, object]:
    reason_counts: dict[str, int] = {}
    for row in rows:
        reason = row["reason"]
        if isinstance(reason, str):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "applicable": True,
        "sampling_scope": "supplemental_samples_32_159",
        "sample_indices": tuple(range(32, 160)),
        "requested_sequences": len(sequence_rows),
        "requested_sequence_ids": tuple(str(row["sequence_id"]) for row in sequence_rows),
        "requested_residues": sum(int(row["requested_residues"]) for row in sequence_rows),
        "published_residues": len(rows),
        "estimable_residues": sum(row["status"] == "ok" for row in rows),
        "censored_residues": sum(row["status"] == "censored" for row in rows),
        "not_estimable_residues": sum(row["status"] == "not_estimable" for row in rows),
        "not_estimable_reason_counts": reason_counts,
        "sequence_rows": sequence_rows,
        "partial": (
            len(rows) != sum(int(row["requested_residues"]) for row in sequence_rows)
            or any(row["status"] == "not_estimable" for row in rows)
        ),
    }


def eligible_contact_matrix(
    positions: np.ndarray,
    *,
    cutoff: float = CONTACT_CUTOFF_ANGSTROM,
    min_sequence_separation: int = MIN_SEQUENCE_SEPARATION,
) -> np.ndarray:
    """Return symmetric eligible contacts using the repository distance kernel."""

    xyz = _validate_positions(positions, "structure")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if min_sequence_separation < 1:
        raise ValueError("min_sequence_separation must be positive")
    strict_cutoff = np.nextafter(float(cutoff), -np.inf)
    contacts = contact_matrix(xyz, contact_cutoff=strict_cutoff)
    indices = np.arange(len(xyz))
    eligible = np.abs(indices[:, None] - indices[None, :]) >= min_sequence_separation
    return np.logical_and(contacts, eligible)


def eligible_pair_indices(
    residue_count: int,
    *,
    min_sequence_separation: int = MIN_SEQUENCE_SEPARATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eligible upper-triangle residue-pair indexes."""

    if residue_count < 1:
        raise ValueError("residue_count must be positive")
    if min_sequence_separation < 1:
        raise ValueError("min_sequence_separation must be positive")
    return np.nonzero(
        np.triu(
            _eligible_pair_mask(
                residue_count,
                min_sequence_separation=min_sequence_separation,
            ),
            k=1,
        )
    )


def align_target_coordinates(
    target_positions: np.ndarray,
    reference_positions: np.ndarray,
) -> np.ndarray:
    """Kabsch-align target structures to one external geometry-only reference."""

    targets = np.asarray(target_positions, dtype=np.float64)
    reference = _validate_positions(reference_positions, "reference")
    if targets.ndim != 3 or targets.shape[1:] != reference.shape or len(targets) == 0:
        raise ValueError("target positions must have shape [samples, residues, 3]")
    if not np.isfinite(targets).all():
        raise ValueError("target positions contain non-finite values")
    return align_trajectory_to_reference(np.concatenate((reference[None, ...], targets), axis=0))[
        1:
    ]


def contact_and_local_q_arrays(
    target_positions: np.ndarray,
    reference_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-sample contacts, local-Q, and reference contact counts."""

    targets = np.asarray(target_positions, dtype=np.float64)
    reference = _validate_positions(reference_positions, "reference")
    if targets.ndim != 3 or targets.shape[1:] != reference.shape or len(targets) == 0:
        raise ValueError("target positions must have shape [samples, residues, 3]")
    if not np.isfinite(targets).all():
        raise ValueError("target positions contain non-finite values")
    contacts = np.stack([eligible_contact_matrix(frame) for frame in targets], axis=0)
    local_q, reference_contact_counts = _local_q_by_sample(targets, reference)
    return contacts, local_q, reference_contact_counts


def radius_of_gyration(positions: np.ndarray) -> float:
    xyz = _validate_positions(positions, "structure")
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.square(centered).sum(axis=-1).mean()))


def relative_shape_anisotropy(positions: np.ndarray) -> float:
    xyz = _validate_positions(positions, "structure")
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    eigenvalues = np.clip(
        np.linalg.eigvalsh(centered.T @ centered / len(centered)),
        0.0,
        None,
    )
    total_variance = float(eigenvalues.sum())
    if total_variance == 0.0:
        return 0.0
    eigenvalues /= total_variance
    pair_products = (
        eigenvalues[0] * eigenvalues[1]
        + eigenvalues[1] * eigenvalues[2]
        + eigenvalues[2] * eigenvalues[0]
    )
    return float(np.clip(1.0 - 3.0 * pair_products, 0.0, 1.0))


def compute_sequence_properties(
    target_positions: np.ndarray,
    *,
    reference_positions: np.ndarray,
    sample_indices: tuple[int, ...],
    include_relative_properties: bool = True,
) -> SequencePropertyComputation:
    """Compute target-only moments after alignment to a fixed external reference."""

    targets = np.asarray(target_positions, dtype=np.float64)
    if targets.ndim != 3 or targets.shape[-1] != 3 or targets.shape[0] == 0:
        raise ValueError("target positions must have shape [samples, residues, 3]")
    if not np.isfinite(targets).all():
        raise ValueError("target positions contain non-finite values")
    reference = _validate_positions(reference_positions, "reference")
    if targets.shape[1:] != reference.shape:
        raise ValueError("target positions must match the reference residue shape")
    if len(sample_indices) != len(targets) or len(set(sample_indices)) != len(sample_indices):
        raise ValueError("sample indices must uniquely identify every target structure")

    aligned = align_target_coordinates(targets, reference)
    contacts, local_q, reference_contact_counts = contact_and_local_q_arrays(targets, reference)
    eligible = _eligible_pair_mask(
        len(reference),
        min_sequence_separation=MIN_SEQUENCE_SEPARATION,
    )
    possible_contacts = int(np.triu(eligible, k=1).sum())

    sample_rows: list[dict[str, object]] = []
    for sample_index, frame, frame_contacts in zip(
        sample_indices,
        targets,
        contacts,
        strict=True,
    ):
        contact_count = int(np.triu(frame_contacts, k=1).sum())
        sample_rows.append(
            {
                "sample_index": int(sample_index),
                "radius_of_gyration": radius_of_gyration(frame),
                "contact_density": (
                    float(contact_count / possible_contacts) if possible_contacts > 0 else None
                ),
                "contact_density_status": (
                    "measured" if possible_contacts > 0 else "undefined_no_eligible_pairs"
                ),
                "end_to_end_distance": float(np.linalg.norm(frame[-1] - frame[0])),
                "relative_shape_anisotropy": relative_shape_anisotropy(frame),
            }
        )
    normalized = {
        ("sequence", int(row["sample_index"])): {
            name: value
            for name in SAMPLE_PROPERTIES
            if (value := _finite_number(row.get(name))) is not None
        }
        for row in sample_rows
    }
    if include_relative_properties:
        add_within_sequence_normalized_sample_properties(
            normalized,
            property_names=("radius_of_gyration", "contact_density"),
        )
    for row in sample_rows:
        values = normalized[("sequence", int(row["sample_index"]))]
        row["relative_radius_of_gyration"] = values.get("relative_radius_of_gyration")
        row["relative_contact_density"] = values.get("relative_contact_density")

    sequence: dict[str, object] = {
        "n_target_samples": len(targets),
        "contact_pair_status": (
            "measured" if possible_contacts > 0 else "undefined_no_eligible_pairs"
        ),
    }
    for property_name in SAMPLE_PROPERTIES:
        values = np.asarray(
            [
                value
                for row in sample_rows
                if (value := _finite_number(row.get(property_name))) is not None
            ],
            dtype=np.float64,
        )
        sequence[f"{property_name}_mean"] = float(values.mean()) if len(values) else None
        sequence[f"{property_name}_variance"] = _sample_variance(values) if len(values) else None
    sequence["mean_reference_aligned_pairwise_rmsd"] = _mean_pairwise_rmsd(aligned)

    contact_probability = contacts.mean(axis=0, dtype=np.float64)
    contact_entropy = _binary_entropy(contact_probability)
    sequence["mean_eligible_pair_binary_contact_entropy"] = _eligible_pair_mean(
        contact_entropy,
        eligible,
    )

    mean_aligned = aligned.mean(axis=0)
    rmsf = np.sqrt(np.square(aligned - mean_aligned).sum(axis=-1).mean(axis=0))
    contact_numbers = contacts.sum(axis=2, dtype=np.int64)
    eligible_counts = eligible.sum(axis=1)
    contact_number_mean = contact_numbers.mean(axis=0)
    contact_number_variance = _sample_variance_axis_zero(contact_numbers)
    contact_number_mean[eligible_counts == 0] = np.nan
    contact_number_variance[eligible_counts == 0] = np.nan
    contact_probability_by_residue = np.divide(
        (contact_probability * eligible).sum(axis=1),
        eligible_counts,
        out=np.full(len(reference), np.nan, dtype=np.float64),
        where=eligible_counts > 0,
    )
    contact_entropy_by_residue = np.divide(
        (contact_entropy * eligible).sum(axis=1),
        eligible_counts,
        out=np.full(len(reference), np.nan, dtype=np.float64),
        where=eligible_counts > 0,
    )
    residue = {
        "rmsf": rmsf,
        "contact_number_mean": contact_number_mean,
        "contact_number_variance": contact_number_variance,
        "eligible_contact_count": eligible_counts,
        "mean_eligible_contact_probability": contact_probability_by_residue,
        "mean_eligible_pair_contact_entropy": contact_entropy_by_residue,
        "reference_local_contact_count": reference_contact_counts,
        "local_q_mean": _nanmean_axis_zero(local_q),
        "local_q_variance": _sample_nanvariance_axis_zero(local_q),
    }
    return SequencePropertyComputation(
        sample=tuple(sample_rows),
        sequence=sequence,
        residue=residue,
    )


def deterministic_convergence_subsets(
    target_sample_indices: tuple[int, ...],
    *,
    convergence_target_counts: tuple[int, ...] = (32, 64, 96, 128),
) -> dict[int, tuple[int, ...]]:
    if len(set(target_sample_indices)) != len(target_sample_indices):
        raise ValueError("target sample indices must be unique")
    ordered = tuple(
        sorted(
            target_sample_indices,
            key=lambda value: (hashlib.sha256(str(value).encode()).hexdigest(), value),
        )
    )
    _validate_convergence_target_counts(convergence_target_counts)
    return {
        target_count: ordered[:target_count]
        for target_count in convergence_target_counts
        if target_count <= len(ordered)
    }


def compute_convergence_rows(
    *,
    sequence_id: str,
    sample_rows: list[dict[str, object]] | tuple[dict[str, object], ...],
    target_sample_indices: tuple[int, ...],
    bootstrap_replicates: int,
    random_seed: int,
    convergence_target_counts: tuple[int, ...] = (32, 64, 96, 128),
    include_successive_absolute_change: bool = True,
) -> list[dict[str, object]]:
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    rows_by_sample = {int(row["sample_index"]): row for row in sample_rows}
    if len(rows_by_sample) != len(sample_rows):
        raise ValueError("sample rows must have unique sample indices")
    subsets = deterministic_convergence_subsets(
        target_sample_indices,
        convergence_target_counts=convergence_target_counts,
    )
    results: list[dict[str, object]] = []
    previous: dict[str, float] = {}
    for target_count, subset in subsets.items():
        available_indices = tuple(index for index in subset if index in rows_by_sample)
        for property_name in SAMPLE_PROPERTIES:
            output_name = f"{property_name}_mean"
            observations = tuple(
                (index, value)
                for index in available_indices
                if (value := _finite_number(rows_by_sample[index].get(property_name))) is not None
            )
            observed_indices = tuple(index for index, _value in observations)
            values = np.asarray([value for _index, value in observations], dtype=np.float64)
            coverage_fraction = len(values) / target_count
            bootstrap_seed = _stable_bootstrap_seed(
                random_seed,
                sequence_id,
                target_count,
                output_name,
            )
            status = "ok" if len(values) == target_count else "not_estimable"
            reason = (
                None
                if status == "ok"
                else ("no_finite_observations" if not len(values) else "incomplete_target_coverage")
            )
            estimate = float(values.mean()) if status == "ok" else None
            interval = (
                _bootstrap_mean_interval(
                    values,
                    bootstrap_replicates=bootstrap_replicates,
                    random_seed=bootstrap_seed,
                )
                if status == "ok"
                else None
            )
            lower = interval[0] if interval is not None else None
            upper = interval[1] if interval is not None else None
            width = upper - lower if lower is not None and upper is not None else None
            prior = previous.get(output_name)
            change = abs(estimate - prior) if estimate is not None and prior is not None else None
            if estimate is not None:
                previous[output_name] = estimate
            bootstrap_schedule_fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "schema_version": 1,
                        "sequence_id": sequence_id,
                        "property": output_name,
                        "target_count": target_count,
                        "sample_indices": subset,
                        "observed_sample_indices": observed_indices,
                        "bootstrap_replicates": bootstrap_replicates,
                        "bootstrap_seed": bootstrap_seed,
                        "quantiles": (0.025, 0.975),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            row = {
                "sequence_id": sequence_id,
                "property": output_name,
                "target_count": target_count,
                "sample_indices": subset,
                "observed_sample_count": len(values),
                "coverage_fraction": float(coverage_fraction),
                "estimate": estimate,
                "successive_absolute_change": change,
                "units": CONVERGENCE_PROPERTY_UNITS[output_name],
                "status": status,
                "reason": reason,
                "bootstrap_95_lower": lower,
                "bootstrap_95_upper": upper,
                "bootstrap_95_width": width,
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_schedule_fingerprint": bootstrap_schedule_fingerprint,
            }
            if not include_successive_absolute_change:
                row.pop("successive_absolute_change")
            results.append(row)
    return results


def compute_ensemble_properties(
    original_structure_root: Path,
    supplemental_structure_root: Path,
    sequence_ids: tuple[str, ...],
    target_sample_indices: tuple[int, ...],
    output_dir: Path,
    bootstrap_replicates: int,
    random_seed: int,
    input_fasta_path: Path | None = None,
    analysis_config_fingerprint: str | None = None,
    analysis_dependency_fingerprint: str | None = None,
    property_sampling_scope: str = "independent_supplemental_128",
    convergence_target_counts: tuple[int, ...] = (32, 64, 96, 128),
    analysis_mode: str = "independent_supplemental_128",
    include_convergence: bool = True,
) -> EnsemblePropertyTables:
    """Compute expanded-target geometry without reading activation archives."""

    original_structure_root = Path(original_structure_root)
    supplemental_structure_root = Path(supplemental_structure_root)
    output_dir = Path(output_dir)
    _validate_requested_scope(
        sequence_ids,
        target_sample_indices,
        bootstrap_replicates,
        property_sampling_scope=property_sampling_scope,
        analysis_mode=analysis_mode,
    )
    scope_contract = _property_scope_contract(
        property_sampling_scope=property_sampling_scope,
        target_sample_indices=target_sample_indices,
        convergence_target_counts=convergence_target_counts,
    )

    failures: list[FailureRecord] = []
    original_entries = _load_index(
        original_structure_root,
        role="alignment_reference",
        sequence_ids=sequence_ids,
        sample_indices=(0,),
        failures=failures,
    )
    target_entries = _load_index(
        supplemental_structure_root,
        role="target",
        sequence_ids=sequence_ids,
        sample_indices=target_sample_indices,
        failures=failures,
    )
    canonical_sequences = _canonical_sequence_scope(
        sequence_ids=sequence_ids,
        input_fasta_path=input_fasta_path,
        original_entries=original_entries,
        supplemental_entries=target_entries,
        required=False,
    )
    source_identity = (
        build_property_source_identity(
            input_fasta_path=input_fasta_path,
            original_structure_root=original_structure_root,
            supplemental_structure_root=supplemental_structure_root,
            sequence_ids=sequence_ids,
            target_sample_indices=target_sample_indices,
            analysis_config_fingerprint=analysis_config_fingerprint,
            analysis_dependency_fingerprint=analysis_dependency_fingerprint,
            original_entries=original_entries,
            supplemental_entries=target_entries,
        )
        if input_fasta_path is not None
        and analysis_config_fingerprint is not None
        and analysis_dependency_fingerprint is not None
        else None
    )

    sample_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    residue_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    loaded_target_structures = 0
    analyzed_target_structures = 0
    sequences_with_reference = 0
    complete_sequences = 0
    sequences_without_eligible_contact_pairs = 0

    def record_convergence_rows(
        sequence_id: str,
        sequence_sample_rows: (list[dict[str, object]] | tuple[dict[str, object], ...]) = (),
    ) -> None:
        if not include_convergence:
            return
        convergence_rows.extend(
            compute_convergence_rows(
                sequence_id=sequence_id,
                sample_rows=sequence_sample_rows,
                target_sample_indices=target_sample_indices,
                bootstrap_replicates=bootstrap_replicates,
                random_seed=random_seed,
                convergence_target_counts=convergence_target_counts,
                include_successive_absolute_change=(
                    property_sampling_scope != "shared_original_32"
                ),
            )
        )

    for sequence_id in sequence_ids:
        canonical_residue_count = len(canonical_sequences.get(sequence_id, ""))
        reference_entry = original_entries.get((sequence_id, 0))
        if reference_entry is None:
            _append_missing_failure(
                failures,
                sequence_id=sequence_id,
                sample_index=0,
                role="alignment_reference",
                detail="seed-0 entry is absent from the original structure index",
            )
            reference = None
        else:
            reference = _load_coordinates(
                original_structure_root,
                reference_entry,
                role="alignment_reference",
                failures=failures,
            )
            if reference is not None:
                sequences_with_reference += 1

        target_coordinates: list[tuple[int, np.ndarray]] = []
        for sample_index in target_sample_indices:
            entry = target_entries.get((sequence_id, sample_index))
            if entry is None:
                _append_missing_failure(
                    failures,
                    sequence_id=sequence_id,
                    sample_index=sample_index,
                    role="target",
                    detail="entry is absent from the supplemental structure index",
                )
                continue
            positions = _load_coordinates(
                supplemental_structure_root,
                entry,
                role="target",
                failures=failures,
            )
            if positions is None:
                continue
            target_coordinates.append((sample_index, positions))
            loaded_target_structures += 1
        if reference is None:
            record_convergence_rows(sequence_id)
            continue
        if len(reference) != canonical_residue_count:
            failures.append(
                FailureRecord(
                    sequence_id=sequence_id,
                    sample_index=0,
                    role="alignment_reference",
                    reason="reference_sequence_length_mismatch",
                    detail=(
                        f"seed-0 coordinates contain {len(reference)} residues but "
                        f"the canonical sequence contains {canonical_residue_count}"
                    ),
                )
            )
            record_convergence_rows(sequence_id)
            continue

        loaded_positions: list[np.ndarray] = []
        loaded_indices: list[int] = []
        for sample_index, positions in target_coordinates:
            if positions.shape != reference.shape:
                failures.append(
                    FailureRecord(
                        sequence_id=sequence_id,
                        sample_index=sample_index,
                        role="target",
                        reason="coordinate_length_mismatch",
                        detail=(
                            f"target shape {positions.shape} does not match "
                            f"seed-0 shape {reference.shape}"
                        ),
                    )
                )
                continue
            loaded_positions.append(positions)
            loaded_indices.append(sample_index)
            analyzed_target_structures += 1
        if not loaded_positions:
            record_convergence_rows(sequence_id)
            continue

        complete = len(loaded_indices) == len(target_sample_indices)
        if complete:
            complete_sequences += 1
        try:
            computed = compute_sequence_properties(
                np.stack(loaded_positions, axis=0),
                reference_positions=reference,
                sample_indices=tuple(loaded_indices),
                include_relative_properties=complete,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            failures.append(
                FailureRecord(
                    sequence_id=sequence_id,
                    sample_index=None,
                    role="alignment",
                    reason="alignment_failed",
                    detail=f"{type(error).__name__}: {error}",
                )
            )
            record_convergence_rows(sequence_id)
            continue
        if computed.sequence["contact_pair_status"] != "measured":
            sequences_without_eligible_contact_pairs += 1
        sequence_sample_rows: list[dict[str, object]] = []
        for row in computed.sample:
            complete_row = {"sequence_id": sequence_id, **row}
            if not complete:
                complete_row["relative_radius_of_gyration"] = None
                complete_row["relative_contact_density"] = None
            sequence_sample_rows.append(complete_row)
            sample_rows.append(complete_row)
        sequence_rows.append(
            {
                "sequence_id": sequence_id,
                **computed.sequence,
                "expected_target_samples": len(target_sample_indices),
                "complete_target_coverage": complete,
            }
        )
        for residue_index in range(len(reference)):
            eligible_contact_count = int(computed.residue["eligible_contact_count"][residue_index])
            reference_local_contact_count = int(
                computed.residue["reference_local_contact_count"][residue_index]
            )
            residue_rows.append(
                {
                    "sequence_id": sequence_id,
                    "residue_index": residue_index,
                    "n_target_samples": len(loaded_indices),
                    **{
                        name: float(values[residue_index])
                        for name, values in computed.residue.items()
                        if name
                        not in {
                            "eligible_contact_count",
                            "reference_local_contact_count",
                        }
                    },
                    "eligible_contact_count": eligible_contact_count,
                    "eligible_contact_status": (
                        "measured" if eligible_contact_count > 0 else "undefined_no_eligible_pairs"
                    ),
                    "reference_local_contact_count": reference_local_contact_count,
                    "local_q_status": (
                        "measured"
                        if reference_local_contact_count > 0
                        else "undefined_no_reference_contacts"
                    ),
                }
            )
        record_convergence_rows(sequence_id, sequence_sample_rows)

    requested_target_structures = len(sequence_ids) * len(target_sample_indices)
    coverage: dict[str, object] = {
        "requested_sequences": len(sequence_ids),
        "requested_target_samples_per_sequence": len(target_sample_indices),
        "requested_target_structures": requested_target_structures,
        "loaded_target_structures": loaded_target_structures,
        "analyzed_target_structures": analyzed_target_structures,
        "missing_target_coordinates": (requested_target_structures - loaded_target_structures),
        "missing_or_invalid_target_structures": (
            requested_target_structures - analyzed_target_structures
        ),
        "sequences_with_seed_zero_reference": sequences_with_reference,
        "complete_sequences": complete_sequences,
        "sequences_without_eligible_contact_pairs": (sequences_without_eligible_contact_pairs),
        "invalid_structure_index_records": sum(
            failure.reason == "invalid_structure_index" for failure in failures
        ),
        "unreadable_structure_indexes": sum(
            failure.reason == "unreadable_structure_index" for failure in failures
        ),
        "partial": (
            analyzed_target_structures != requested_target_structures
            or sequences_with_reference != len(sequence_ids)
        ),
        **scope_contract,
    }
    if scope_contract and (coverage["partial"] or failures):
        diagnostic_dir = _write_unpublished_property_diagnostics(
            output_dir=output_dir,
            failures=failures,
            coverage=coverage,
            scope_contract=scope_contract,
        )
        failure_summary = "; ".join(
            f"{failure.role}:{failure.reason}"
            f"[sequence_id={failure.sequence_id},sample_index={failure.sample_index}]: "
            f"{failure.detail}"
            for failure in failures
        )
        raise ValueError(
            "shared property publication requires complete exact 32-seed coverage"
            f"; unpublished diagnostics: {diagnostic_dir}"
            + (f"; {failure_summary}" if failure_summary else "")
        )
    if not include_convergence:
        coverage["sample_count_convergence"] = {
            "applicable": False,
            "reason": "same_trajectory_exploratory_32_has_exactly_32_samples",
        }
    paths = _write_property_tables(
        output_dir=output_dir,
        sample_rows=sample_rows,
        sequence_rows=sequence_rows,
        residue_rows=residue_rows,
        convergence_rows=convergence_rows,
        failures=failures,
        coverage=coverage,
        target_sample_indices=target_sample_indices,
        bootstrap_replicates=bootstrap_replicates,
        random_seed=random_seed,
        source_identity=source_identity,
        property_sampling_scope=property_sampling_scope,
        convergence_target_counts=convergence_target_counts,
        analysis_mode=analysis_mode,
        include_convergence=include_convergence,
    )
    return EnsemblePropertyTables(
        output_dir=paths["metadata_json"].parent,
        sample=tuple(sample_rows),
        sequence=tuple(sequence_rows),
        residue=tuple(residue_rows),
        convergence=tuple(convergence_rows),
        failures=tuple(failures),
        coverage=coverage,
        paths=paths,
    )


def _validate_positions(values: np.ndarray, name: str) -> np.ndarray:
    positions = np.asarray(values, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ValueError(f"{name} positions must have shape [residues, 3]")
    if not np.isfinite(positions).all():
        raise ValueError(f"{name} positions contain non-finite values")
    return positions


def _eligible_pair_mask(
    residue_count: int,
    *,
    min_sequence_separation: int,
) -> np.ndarray:
    indices = np.arange(residue_count)
    return np.abs(indices[:, None] - indices[None, :]) >= min_sequence_separation


def _sample_variance(values: np.ndarray) -> float:
    return float(values.var(ddof=1)) if len(values) >= 2 else math.nan


def _sample_variance_axis_zero(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.full(values.shape[1], np.nan, dtype=np.float64)
    return np.asarray(values, dtype=np.float64).var(axis=0, ddof=1)


def _finite_number(value: object) -> float | None:
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, int | float | np.integer | np.floating)
    ):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _mean_pairwise_rmsd(aligned: np.ndarray) -> float:
    values = [
        float(np.sqrt(np.square(aligned[left] - aligned[right]).sum(axis=-1).mean()))
        for left in range(len(aligned))
        for right in range(left + 1, len(aligned))
    ]
    return float(np.mean(values)) if values else math.nan


def _binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    result = np.zeros_like(probabilities, dtype=np.float64)
    interior = np.logical_and(probabilities > 0.0, probabilities < 1.0)
    p = probabilities[interior]
    result[interior] = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return result


def _eligible_pair_mean(
    values: np.ndarray,
    eligible: np.ndarray,
) -> float | None:
    upper = np.triu(eligible, k=1)
    return float(values[upper].mean()) if upper.any() else None


def _validate_local_q_boundary(boundary: float) -> None:
    if (
        isinstance(boundary, bool)
        or not isinstance(boundary, int | float | np.integer | np.floating)
        or not math.isfinite(float(boundary))
        or not 0.0 < float(boundary) < 1.0
    ):
        raise ValueError("local-Q boundary must be finite and in (0, 1)")


def _strict_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _strict_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_reference_contact_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} reference_contact_count must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} reference_contact_count must be non-negative")
    return result


def _validate_sample_indices(
    value: object,
    name: str,
) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{name} sample indices must be a non-empty tuple")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in value):
        raise ValueError(f"{name} sample indices must be integers")
    if len(set(value)) != len(value):
        raise ValueError(f"duplicate {name} sample indices")
    return value


def _validate_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_candidate_hashes(
    value: object,
    *,
    candidate_count: int,
) -> tuple[tuple[str, dict[str, object]], ...]:
    if not isinstance(value, dict) or len(value) != candidate_count:
        raise ValueError("local-Q calibration candidate hashes are incomplete")
    required = {
        "candidate_sha256",
        "screen_manifest_sha256",
        "screen_summary_sha256",
        "sample_metrics_sha256",
        "reference_sha256",
        "structure_index_sha256",
        "calibration_time_coordinate_sha256",
        "calibration_time_coordinate_corpus_sha256",
    }
    validated = []
    coordinate_path_owners: dict[str, str] = {}
    coordinate_hash_owners: dict[str, str] = {}
    coordinate_corpus_owners: dict[str, str] = {}
    for candidate_id, raw_hashes in sorted(value.items()):
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("local-Q calibration candidate ID must be non-empty")
        if not isinstance(raw_hashes, dict) or set(raw_hashes) != required:
            raise ValueError(f"candidate {candidate_id} hashes are incomplete")
        hashes = dict(raw_hashes)
        for name in required - {"calibration_time_coordinate_sha256"}:
            _validate_sha256(hashes[name], f"{candidate_id}.{name}")
        coordinate_hashes = hashes["calibration_time_coordinate_sha256"]
        if not isinstance(coordinate_hashes, dict) or not coordinate_hashes:
            raise ValueError(f"candidate {candidate_id} coordinate hashes are incomplete")
        for path, digest in coordinate_hashes.items():
            if not isinstance(path, str) or not path:
                raise ValueError(f"candidate {candidate_id} coordinate path is invalid")
            validated_digest = _validate_sha256(
                digest,
                f"{candidate_id}.coordinate_sha256",
            )
            path_owner = coordinate_path_owners.setdefault(path, candidate_id)
            if path_owner != candidate_id:
                raise ValueError("coordinate path reused across candidates")
            hash_owner = coordinate_hash_owners.setdefault(validated_digest, candidate_id)
            if hash_owner != candidate_id:
                raise ValueError("coordinate hash reused across candidates")
        corpus_digest = str(hashes["calibration_time_coordinate_corpus_sha256"])
        corpus_owner = coordinate_corpus_owners.setdefault(corpus_digest, candidate_id)
        if corpus_owner != candidate_id:
            raise ValueError("coordinate corpus reused across candidates")
        validated.append((candidate_id, hashes))
    return tuple(validated)


def _validate_code_hashes(value: object) -> tuple[tuple[str, str], ...]:
    required = {"local_q_algorithm_sha256", "calibration_script_sha256"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("local-Q calibration code hashes are incomplete")
    return tuple((name, _validate_sha256(value[name], name)) for name in sorted(required))


def _validate_all_data_refit(
    value: object,
    *,
    expected_candidate_ids: set[str],
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, dict) or set(value) != {"boundary", "boundary_metrics"}:
        raise ValueError("local-Q calibration all-data refit is incomplete")
    declared_boundary = _strict_finite_float(
        value["boundary"],
        "all-data refit boundary",
    )
    raw_metrics = value["boundary_metrics"]
    if not isinstance(raw_metrics, list) or len(raw_metrics) != len(LOCAL_Q_BOUNDARY_CANDIDATES):
        raise ValueError("local-Q calibration all-data refit metrics are incomplete")
    metrics = []
    for expected_boundary, raw_row in zip(
        LOCAL_Q_BOUNDARY_CANDIDATES,
        raw_metrics,
        strict=True,
    ):
        if not isinstance(raw_row, dict) or set(raw_row) != {
            "boundary",
            "mean_balanced_accuracy",
            "candidate_scores",
        }:
            raise ValueError("invalid local-Q calibration all-data refit metric")
        boundary = _strict_finite_float(raw_row["boundary"], "all-data boundary")
        if not math.isclose(boundary, expected_boundary, abs_tol=1e-12):
            raise ValueError("local-Q calibration all-data boundary grid mismatch")
        mean_accuracy = _strict_finite_float(
            raw_row["mean_balanced_accuracy"],
            "mean_balanced_accuracy",
        )
        if not 0.0 <= mean_accuracy <= 1.0:
            raise ValueError("mean_balanced_accuracy must be in [0, 1]")
        candidate_scores = raw_row["candidate_scores"]
        if not isinstance(candidate_scores, list) or len(candidate_scores) != len(
            expected_candidate_ids
        ):
            raise ValueError("local-Q calibration candidate scores are incomplete")
        seen_candidates: set[str] = set()
        accuracies = []
        for score in candidate_scores:
            if not isinstance(score, dict) or set(score) != {
                "candidate_id",
                "balanced_accuracy",
                "folded_recall",
                "unfolded_recall",
                "folded_samples",
                "unfolded_samples",
            }:
                raise ValueError("invalid local-Q calibration candidate score")
            candidate_id = score["candidate_id"]
            if not isinstance(candidate_id, str) or candidate_id in seen_candidates:
                raise ValueError("duplicate or invalid calibration candidate score")
            seen_candidates.add(candidate_id)
            for name in ("balanced_accuracy", "folded_recall", "unfolded_recall"):
                metric = _strict_finite_float(score[name], name)
                if not 0.0 <= metric <= 1.0:
                    raise ValueError(f"{name} must be in [0, 1]")
                if name == "balanced_accuracy":
                    accuracies.append(metric)
            expected_balanced_accuracy = 0.5 * (
                float(score["folded_recall"]) + float(score["unfolded_recall"])
            )
            if not math.isclose(
                float(score["balanced_accuracy"]),
                expected_balanced_accuracy,
                abs_tol=1e-12,
            ):
                raise ValueError("candidate balanced accuracy mismatch")
            for name in ("folded_samples", "unfolded_samples"):
                if _strict_positive_int(score[name], name) < 10:
                    raise ValueError(f"{name} must be at least ten")
        if seen_candidates != expected_candidate_ids:
            raise ValueError("local-Q calibration candidate score IDs mismatch")
        if not math.isclose(
            mean_accuracy,
            float(np.mean(accuracies)),
            abs_tol=1e-12,
        ):
            raise ValueError("local-Q calibration all-data mean accuracy mismatch")
        metrics.append(dict(raw_row))
    validated = tuple(metrics)
    selected = _select_boundary_from_metric_rows(validated)
    if not math.isclose(selected, declared_boundary, abs_tol=1e-12):
        raise ValueError("local-Q calibration declared all-data boundary mismatch")
    return validated


def _select_boundary_from_metric_rows(
    metrics: tuple[dict[str, object], ...],
) -> float:
    selected = min(
        metrics,
        key=lambda row: (
            -float(row["mean_balanced_accuracy"]),
            abs(float(row["boundary"]) - 0.5),
            float(row["boundary"]),
        ),
    )
    return float(selected["boundary"])


def _validate_loco_metrics(
    value: object,
    *,
    all_data_metrics: tuple[dict[str, object], ...],
    expected_candidate_ids: set[str],
) -> tuple[tuple[dict[str, object], ...], float]:
    if not isinstance(value, dict) or set(value) != {
        "mean_held_out_balanced_accuracy",
        "folds",
    }:
        raise ValueError("local-Q calibration LOCO metrics are incomplete")
    raw_folds = value["folds"]
    if not isinstance(raw_folds, list) or len(raw_folds) != len(expected_candidate_ids):
        raise ValueError("local-Q calibration LOCO folds are incomplete")
    scores_by_boundary = {
        float(row["boundary"]): {
            str(score["candidate_id"]): score
            for score in row["candidate_scores"]  # type: ignore[union-attr]
        }
        for row in all_data_metrics
    }
    validated = []
    seen_candidates: set[str] = set()
    held_out_accuracies = []
    for raw_fold in raw_folds:
        required = {
            "candidate_id",
            "selected_boundary",
            "training_mean_balanced_accuracy",
            "held_out_balanced_accuracy",
            "held_out_folded_recall",
            "held_out_unfolded_recall",
            "folded_samples",
            "unfolded_samples",
        }
        if not isinstance(raw_fold, dict) or set(raw_fold) != required:
            raise ValueError("invalid local-Q calibration LOCO fold")
        candidate_id = raw_fold["candidate_id"]
        if (
            not isinstance(candidate_id, str)
            or candidate_id in seen_candidates
            or candidate_id not in expected_candidate_ids
        ):
            raise ValueError("duplicate or invalid local-Q calibration LOCO candidate")
        seen_candidates.add(candidate_id)
        training_rows = tuple(
            {
                "boundary": boundary,
                "mean_balanced_accuracy": float(
                    np.mean(
                        [
                            float(score["balanced_accuracy"])
                            for other_id, score in scores.items()
                            if other_id != candidate_id
                        ]
                    )
                ),
            }
            for boundary, scores in scores_by_boundary.items()
        )
        expected_boundary = _select_boundary_from_metric_rows(training_rows)
        selected_boundary = _strict_finite_float(
            raw_fold["selected_boundary"],
            "LOCO selected boundary",
        )
        if not math.isclose(selected_boundary, expected_boundary, abs_tol=1e-12):
            raise ValueError("local-Q calibration LOCO boundary leakage or mismatch")
        held_out_score = scores_by_boundary[selected_boundary][candidate_id]
        expected_training_mean = next(
            float(row["mean_balanced_accuracy"])
            for row in training_rows
            if math.isclose(float(row["boundary"]), selected_boundary, abs_tol=1e-12)
        )
        comparisons = {
            "training_mean_balanced_accuracy": expected_training_mean,
            "held_out_balanced_accuracy": float(held_out_score["balanced_accuracy"]),
            "held_out_folded_recall": float(held_out_score["folded_recall"]),
            "held_out_unfolded_recall": float(held_out_score["unfolded_recall"]),
        }
        for name, expected_value in comparisons.items():
            observed = _strict_finite_float(raw_fold[name], name)
            if not math.isclose(observed, expected_value, abs_tol=1e-12):
                raise ValueError(f"local-Q calibration LOCO {name} mismatch")
        for name in ("folded_samples", "unfolded_samples"):
            if raw_fold[name] != held_out_score[name]:
                raise ValueError(f"local-Q calibration LOCO {name} mismatch")
        held_out_accuracies.append(float(raw_fold["held_out_balanced_accuracy"]))
        validated.append(dict(raw_fold))
    if seen_candidates != expected_candidate_ids:
        raise ValueError("local-Q calibration LOCO candidate IDs mismatch")
    mean_accuracy = _strict_finite_float(
        value["mean_held_out_balanced_accuracy"],
        "mean_held_out_balanced_accuracy",
    )
    if not math.isclose(
        mean_accuracy,
        float(np.mean(held_out_accuracies)),
        abs_tol=1e-12,
    ):
        raise ValueError("local-Q calibration mean LOCO accuracy mismatch")
    return tuple(validated), mean_accuracy


def _validate_q_vector(q_values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(q_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} Q values must be a non-empty one-dimensional vector")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} Q values contain nonfinite values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} Q values must be in [0, 1]")
    return values


def _local_delta_g_diagnostics(
    q_values: np.ndarray,
    *,
    boundary: float,
    bootstrap_replicates: int,
    random_seed: int,
) -> dict[str, float | bool]:
    ordered_scores = expit(LOCAL_Q_ORDERED_SCORE_SLOPE * (q_values - boundary))
    p_ordered = float(ordered_scores.mean())
    effective_ordered = float(ordered_scores.sum())
    effective_disordered = float((1.0 - ordered_scores).sum())
    scale = KCAL_PER_MOL_K * LOCAL_DELTA_G_TEMPERATURE_K
    delta_g = float(-scale * np.log(p_ordered / (1.0 - p_ordered)))
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(
        0,
        len(ordered_scores),
        size=(bootstrap_replicates, len(ordered_scores)),
    )
    bootstrap_probabilities = ordered_scores[indices].mean(axis=1)
    bootstrap_delta_g = -scale * np.log(bootstrap_probabilities / (1.0 - bootstrap_probabilities))
    ci_low, ci_high = np.percentile(bootstrap_delta_g, [2.5, 97.5])
    ci_width = float(ci_high - ci_low)
    return {
        "p_ordered": p_ordered,
        "effective_ordered": effective_ordered,
        "effective_disordered": effective_disordered,
        "delta_g": delta_g,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci_width": ci_width,
        "enough_effective_samples": (effective_ordered >= 10.0 and effective_disordered >= 10.0),
        "interval_narrow": ci_width <= 1.0,
        "censored": abs(delta_g) > PRACTICAL_LOCAL_DELTA_G_LIMIT_KCAL_MOL,
    }


def _local_delta_g_failure(
    diagnostics: Mapping[str, float | bool],
    *,
    reason: str,
    bootstrap_replicates: int,
    reference_contact_count: int,
    sensitivity_evaluated_count: int = 0,
    sensitivity_estimable_count: int = 0,
) -> LocalDeltaGResult:
    return LocalDeltaGResult(
        status="not_estimable",
        delta_g_kcal_mol=None,
        failure_reason=reason,
        p_ordered=float(diagnostics["p_ordered"]),
        effective_ordered_observations=float(diagnostics["effective_ordered"]),
        effective_disordered_observations=float(diagnostics["effective_disordered"]),
        bootstrap_ci_low_kcal_mol=float(diagnostics["ci_low"]),
        bootstrap_ci_high_kcal_mol=float(diagnostics["ci_high"]),
        bootstrap_ci_width_kcal_mol=float(diagnostics["ci_width"]),
        bootstrap_replicates=bootstrap_replicates,
        reference_contact_count=reference_contact_count,
        sensitivity_evaluated_count=sensitivity_evaluated_count,
        sensitivity_estimable_count=sensitivity_estimable_count,
        censoring=None,
        censoring_bound_kcal_mol=None,
    )


def _local_delta_g_is_estimable(
    diagnostics: Mapping[str, float | bool],
    *,
    reference_contact_count: int,
) -> bool:
    return bool(
        reference_contact_count >= 5
        and diagnostics["enough_effective_samples"]
        and diagnostics["interval_narrow"]
        and not diagnostics["censored"]
    )


def _local_delta_g_primary_failure_reason(
    diagnostics: Mapping[str, float | bool],
    *,
    reference_contact_count: int,
) -> str | None:
    if reference_contact_count < 5:
        return "too_few_contacts"
    if not bool(diagnostics["enough_effective_samples"]):
        return "too_few_effective_state_samples"
    if not bool(diagnostics["interval_narrow"]):
        return "wide_interval"
    return None


def _stable_sign(value: float) -> int:
    if math.isclose(value, 0.0, abs_tol=1e-12):
        return 0
    return -1 if value < 0.0 else 1


def _local_q_by_sample(
    targets: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return compute_local_q_by_sample(
        targets,
        reference,
        cutoff=CONTACT_CUTOFF_ANGSTROM,
        beta=LOCAL_Q_BETA_PER_ANGSTROM,
        distance_multiplier=LOCAL_Q_DISTANCE_MULTIPLIER,
    )


def compute_local_q_by_sample(
    targets: np.ndarray,
    reference: np.ndarray,
    *,
    cutoff: float,
    beta: float,
    distance_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute residue-local Q using the shared production contact semantics."""

    target_values = np.asarray(targets, dtype=np.float64)
    reference_values = _validate_positions(reference, "reference")
    if (
        target_values.ndim != 3
        or target_values.shape[1:] != reference_values.shape
        or target_values.shape[0] == 0
    ):
        raise ValueError("target positions must have shape [samples, residues, 3]")
    if not np.isfinite(target_values).all():
        raise ValueError("target positions contain non-finite values")
    if cutoff <= 0.0 or beta <= 0.0 or distance_multiplier <= 0.0:
        raise ValueError("local-Q parameters must be positive")
    reference_contacts = eligible_contact_matrix(
        reference_values,
        cutoff=cutoff,
        min_sequence_separation=MIN_SEQUENCE_SEPARATION,
    )
    reference_distances = np.linalg.norm(
        reference_values[:, None, :] - reference_values[None, :, :],
        axis=-1,
    )
    target_distances = np.linalg.norm(
        target_values[:, :, None, :] - target_values[:, None, :, :],
        axis=-1,
    )
    exponent = np.clip(
        beta * (target_distances - distance_multiplier * reference_distances[None, ...]),
        -60.0,
        60.0,
    )
    weights = 1.0 / (1.0 + np.exp(exponent))
    local_q = np.full(
        (len(target_values), len(reference_values)),
        np.nan,
        dtype=np.float64,
    )
    for residue_index in range(len(reference_values)):
        local_contacts = reference_contacts[residue_index]
        if local_contacts.any():
            local_q[:, residue_index] = weights[:, residue_index, local_contacts].mean(axis=1)
    return local_q, reference_contacts.sum(axis=1, dtype=np.int64)


def _nanmean_axis_zero(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    for index in range(values.shape[1]):
        finite = values[:, index][np.isfinite(values[:, index])]
        if len(finite):
            result[index] = float(finite.mean())
    return result


def _sample_nanvariance_axis_zero(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    for index in range(values.shape[1]):
        finite = values[:, index][np.isfinite(values[:, index])]
        if len(finite) >= 2:
            result[index] = float(finite.var(ddof=1))
    return result


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    bootstrap_replicates: int,
    random_seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_seed)
    draws = rng.integers(0, len(values), size=(bootstrap_replicates, len(values)))
    estimates = values[draws].mean(axis=1)
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return float(lower), float(upper)


def _stable_bootstrap_seed(
    random_seed: int,
    sequence_id: str,
    target_count: int,
    property_name: str,
) -> int:
    payload = f"{random_seed}|{sequence_id}|{target_count}|{property_name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _validate_requested_scope(
    sequence_ids: tuple[str, ...],
    target_sample_indices: tuple[int, ...],
    bootstrap_replicates: int,
    *,
    property_sampling_scope: str,
    analysis_mode: str = "independent_supplemental_128",
) -> None:
    if not sequence_ids or len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("sequence IDs must be non-empty and unique")
    if not target_sample_indices or len(set(target_sample_indices)) != len(target_sample_indices):
        raise ValueError("target sample indices must be non-empty and unique")
    if analysis_mode == "same_trajectory_exploratory_32":
        if target_sample_indices != tuple(range(32)):
            raise ValueError("exploratory target sample indices must be exactly 0..31")
    elif property_sampling_scope == "shared_original_32":
        if target_sample_indices != tuple(range(32)):
            raise ValueError("shared property sample indices must be exactly 0..31")
    elif property_sampling_scope == "independent_supplemental_128":
        if any(index < 32 or index > 159 for index in target_sample_indices):
            raise ValueError("target sample indices must be within 32..159")
    else:
        raise ValueError("property sampling scope is invalid")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")


def _validate_convergence_target_counts(target_counts: tuple[int, ...]) -> None:
    if (
        not target_counts
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in target_counts
        )
        or tuple(sorted(set(target_counts))) != target_counts
    ):
        raise ValueError("convergence target counts must be unique increasing positive integers")


def _property_scope_contract(
    *,
    property_sampling_scope: str,
    target_sample_indices: tuple[int, ...],
    convergence_target_counts: tuple[int, ...],
) -> dict[str, object]:
    _validate_convergence_target_counts(convergence_target_counts)
    if property_sampling_scope == "independent_supplemental_128":
        if convergence_target_counts != (32, 64, 96, 128):
            raise ValueError("independent convergence target counts must be 32, 64, 96, and 128")
        return {}
    if property_sampling_scope != "shared_original_32":
        raise ValueError("property sampling scope is invalid")
    if target_sample_indices != tuple(range(32)):
        raise ValueError("shared property sample indices must be exactly 0..31")
    if convergence_target_counts != (32,):
        raise ValueError("shared convergence target counts must contain only 32")
    return {
        "analysis_design": "shared_original_32",
        "property_sampling_scope": "shared_original_32",
        "convergence_target_counts": [32],
    }


def _load_index(
    root: Path,
    *,
    role: str,
    sequence_ids: tuple[str, ...],
    sample_indices: tuple[int, ...],
    failures: list[FailureRecord],
) -> dict[tuple[str, int], StructureIndexEntry]:
    index_path = root / "final_structures" / "index.jsonl"
    try:
        raw_lines = index_path.read_text().splitlines()
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError) as error:
        _append_index_failure(
            failures,
            sequence_id=None,
            role=role,
            sample_index=None,
            reason="unreadable_structure_index",
            detail=f"{index_path}: {type(error).__name__}: {error}",
        )
        return {}
    requested_sequences = set(sequence_ids)
    requested_samples = set(sample_indices)
    by_key: dict[tuple[str, int], StructureIndexEntry] = {}
    duplicate_keys: set[tuple[str, int]] = set()
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            continue
        payload: object = None
        try:
            payload = json.loads(raw_line)
            entry = _parse_structure_index_entry(payload, root=root)
        except (json.JSONDecodeError, ValueError) as error:
            payload_sequence_id, payload_sample_index = _index_failure_identity(
                payload,
                requested_sequences=requested_sequences,
                requested_samples=requested_samples,
            )
            _append_index_failure(
                failures,
                sequence_id=payload_sequence_id,
                role=role,
                sample_index=payload_sample_index,
                reason="invalid_structure_index",
                detail=f"{index_path}:{line_number}: {error}",
            )
            continue
        key = (entry.sequence_id, entry.sample_index)
        if (
            entry.sequence_id not in requested_sequences
            or entry.sample_index not in requested_samples
        ):
            continue
        if key in duplicate_keys:
            continue
        if key in by_key:
            failures.append(
                FailureRecord(
                    sequence_id=entry.sequence_id,
                    sample_index=entry.sample_index,
                    role=role,
                    reason="duplicate_structure_index_entry",
                    detail=f"duplicate entry in {index_path}",
                )
            )
            duplicate_keys.add(key)
            by_key.pop(key, None)
            continue
        by_key[key] = entry
    return by_key


def _parse_structure_index_entry(
    payload: object,
    *,
    root: Path,
) -> StructureIndexEntry:
    if not isinstance(payload, dict):
        raise ValueError("row must be a JSON object")
    for field in ("sequence_id", "sample_index", "sequence"):
        if field not in payload:
            raise ValueError(f"missing required field {field!r}")
    sequence_id = payload["sequence_id"]
    if not isinstance(sequence_id, str) or not sequence_id:
        raise ValueError("sequence_id must be a non-empty string")
    sample_index = payload["sample_index"]
    if not isinstance(sample_index, int) or isinstance(sample_index, bool):
        raise ValueError("sample_index must be an integer")
    sequence = payload["sequence"]
    if not isinstance(sequence, str):
        raise ValueError("sequence must be a string")

    members: dict[str, str | None] = {}
    for field in ("npz_path", "pdb_path"):
        value = payload.get(field)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{field} must be a non-empty string or null")
        members[field] = (
            str(_canonical_structure_member(root, value, field=field))
            if value is not None
            else None
        )
    if members["npz_path"] is None and members["pdb_path"] is None:
        raise ValueError("row must declare npz_path or pdb_path")

    units: dict[str, str | None] = {}
    for field in ("pos_units", "pdb_units"):
        value = payload.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} must be a string or null")
        units[field] = value
    return StructureIndexEntry(
        sequence_id=sequence_id,
        sample_index=sample_index,
        sequence=sequence,
        npz_path=members["npz_path"],
        pdb_path=members["pdb_path"],
        pos_units=units["pos_units"],
        pdb_units=units["pdb_units"],
    )


def _canonical_structure_member(root: Path, member: str, *, field: str) -> Path:
    relative = Path(member)
    if relative.is_absolute():
        raise ValueError(
            f"{field} escapes configured structure root: absolute paths are not allowed"
        )
    try:
        canonical_root = root.resolve(strict=False)
        canonical_member = (canonical_root / relative).resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"{field} cannot be canonicalized within configured structure root: {member}"
        ) from error
    if canonical_member == canonical_root or canonical_root not in canonical_member.parents:
        raise ValueError(f"{field} escapes configured structure root: {member}")
    return canonical_member


def _index_failure_identity(
    payload: object,
    *,
    requested_sequences: set[str],
    requested_samples: set[int],
) -> tuple[str | None, int | None]:
    if not isinstance(payload, dict):
        return None, None
    sequence_id = payload.get("sequence_id")
    sample_index = payload.get("sample_index")
    return (
        (
            sequence_id
            if isinstance(sequence_id, str) and sequence_id in requested_sequences
            else None
        ),
        (
            sample_index
            if isinstance(sample_index, int)
            and not isinstance(sample_index, bool)
            and sample_index in requested_samples
            else None
        ),
    )


def _append_index_failure(
    failures: list[FailureRecord],
    *,
    sequence_id: str | None,
    role: str,
    sample_index: int | None,
    reason: str,
    detail: str,
) -> None:
    failures.append(
        FailureRecord(
            sequence_id=sequence_id,
            sample_index=sample_index,
            role=role,
            reason=reason,
            detail=detail,
        )
    )


def _append_missing_failure(
    failures: list[FailureRecord],
    *,
    sequence_id: str,
    sample_index: int,
    role: str,
    detail: str,
) -> None:
    if any(
        failure.sequence_id == sequence_id
        and failure.role == role
        and failure.reason == "invalid_structure_index"
        and failure.sample_index == sample_index
        for failure in failures
    ):
        return
    failures.append(
        FailureRecord(
            sequence_id=sequence_id,
            sample_index=sample_index,
            role=role,
            reason="missing_structure_coordinates",
            detail=detail,
        )
    )


def _load_coordinates(
    root: Path,
    entry: StructureIndexEntry,
    *,
    role: str,
    failures: list[FailureRecord],
) -> np.ndarray | None:
    try:
        return _validate_positions(
            load_structure_positions(root, entry),
            f"{entry.sequence_id} sample {entry.sample_index}",
        )
    except (FileNotFoundError, OSError) as error:
        reason = "missing_structure_coordinates"
        error_detail = f"{type(error).__name__}: {error}"
    except (KeyError, ValueError) as error:
        reason = "invalid_structure_coordinates"
        error_detail = f"{type(error).__name__}: {error}"
    failures.append(
        FailureRecord(
            sequence_id=entry.sequence_id,
            sample_index=entry.sample_index,
            role=role,
            reason=reason,
            detail=error_detail,
        )
    )
    return None


def build_property_source_identity(
    *,
    input_fasta_path: Path,
    original_structure_root: Path,
    supplemental_structure_root: Path,
    sequence_ids: tuple[str, ...],
    target_sample_indices: tuple[int, ...],
    analysis_config_fingerprint: str,
    analysis_dependency_fingerprint: str,
    original_entries: Mapping[tuple[str, int], StructureIndexEntry] | None = None,
    supplemental_entries: Mapping[tuple[str, int], StructureIndexEntry] | None = None,
) -> dict[str, object]:
    """Hash every mutable source that defines one Task 5 generation."""

    fasta_path = Path(input_fasta_path)
    fasta_bytes = fasta_path.read_bytes()
    parsed_records = _parse_fasta_identity(fasta_bytes)
    if tuple(record[0] for record in parsed_records) != sequence_ids:
        raise ValueError("Task 5 FASTA record order disagrees with requested sequence IDs")
    original_root = Path(original_structure_root)
    supplemental_root = Path(supplemental_structure_root)
    if original_entries is None or supplemental_entries is None:
        identity_failures: list[FailureRecord] = []
        original_entries = _load_index(
            original_root,
            role="alignment_reference",
            sequence_ids=sequence_ids,
            sample_indices=(0,),
            failures=identity_failures,
        )
        supplemental_entries = _load_index(
            supplemental_root,
            role="target",
            sequence_ids=sequence_ids,
            sample_indices=target_sample_indices,
            failures=identity_failures,
        )
    structure_rows = []
    for role, root, entries in (
        ("alignment_reference", original_root, original_entries),
        ("target", supplemental_root, supplemental_entries),
    ):
        for (sequence_id, sample_index), entry in sorted(entries.items()):
            member = entry.npz_path if entry.npz_path is not None else entry.pdb_path
            if member is None:
                continue
            member_path = Path(member)
            member_path = member_path if member_path.is_absolute() else root / member_path
            structure_rows.append(
                {
                    "role": role,
                    "sequence_id": sequence_id,
                    "sample_index": sample_index,
                    "member": str(member_path.resolve(strict=False)),
                    "byte_count": member_path.stat().st_size,
                    "sha256": _sha256_file(member_path),
                }
            )
    indexes = {}
    for name, root in (
        ("original", original_root),
        ("supplemental", supplemental_root),
    ):
        path = root / "final_structures" / "index.jsonl"
        raw = path.read_bytes()
        indexes[name] = {
            "path": str(path.resolve(strict=False)),
            "byte_count": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return {
        "schema_version": 2,
        "fasta": {
            "path": str(fasta_path.resolve(strict=False)),
            "byte_count": len(fasta_bytes),
            "sha256": hashlib.sha256(fasta_bytes).hexdigest(),
            "records": [list(record) for record in parsed_records],
        },
        "structure_indexes": indexes,
        "referenced_structures": structure_rows,
        "analysis_config_fingerprint": analysis_config_fingerprint,
        "analysis_dependency_fingerprint": analysis_dependency_fingerprint,
        "property_scientific_dependency_fingerprint": (
            property_scientific_dependency_fingerprint()
        ),
    }


def _parse_fasta_identity(content: bytes) -> tuple[tuple[str, str], ...]:
    try:
        text = content.decode()
    except UnicodeDecodeError as error:
        raise ValueError("Task 5 FASTA is not UTF-8") from error
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    sequence_parts: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if current_id is not None:
                records.append((current_id, "".join(sequence_parts)))
            current_id = line[1:].split()[0]
            sequence_parts = []
        elif line.strip():
            if current_id is None:
                raise ValueError("Task 5 FASTA has sequence bytes before its first header")
            sequence_parts.append(line.strip())
    if current_id is not None:
        records.append((current_id, "".join(sequence_parts)))
    if not records:
        raise ValueError("Task 5 FASTA contains no records")
    return tuple(records)


def _canonical_sequence_scope(
    *,
    sequence_ids: tuple[str, ...],
    input_fasta_path: Path | None,
    original_entries: Mapping[tuple[str, int], StructureIndexEntry],
    supplemental_entries: Mapping[tuple[str, int], StructureIndexEntry],
    required: bool,
) -> dict[str, str]:
    if input_fasta_path is not None:
        records = _parse_fasta_identity(Path(input_fasta_path).read_bytes())
        if tuple(sequence_id for sequence_id, _sequence in records) != sequence_ids:
            raise ValueError("canonical FASTA order disagrees with requested sequence IDs")
        sequences = dict(records)
    else:
        sequences = {}
        for sequence_id in sequence_ids:
            candidates = {
                entry.sequence
                for entries in (original_entries, supplemental_entries)
                for (entry_sequence_id, _sample_index), entry in entries.items()
                if entry_sequence_id == sequence_id
            }
            if len(candidates) > 1:
                raise ValueError(
                    f"structure indexes disagree on canonical sequence for {sequence_id}"
                )
            if candidates:
                sequences[sequence_id] = candidates.pop()
    for entries in (original_entries, supplemental_entries):
        for (sequence_id, _sample_index), entry in entries.items():
            canonical = sequences.get(sequence_id)
            if canonical is not None and entry.sequence != canonical:
                raise ValueError(
                    f"structure index sequence disagrees with canonical FASTA for {sequence_id}"
                )
    if required:
        missing = tuple(
            sequence_id for sequence_id in sequence_ids if not sequences.get(sequence_id)
        )
        if missing:
            raise ValueError(
                "calibrated effective local delta G requires canonical residue "
                f"identities for every sequence; missing {missing}"
            )
    return sequences


def _write_property_tables(
    *,
    output_dir: Path,
    sample_rows: list[dict[str, object]],
    sequence_rows: list[dict[str, object]],
    residue_rows: list[dict[str, object]],
    convergence_rows: list[dict[str, object]],
    failures: list[FailureRecord],
    coverage: dict[str, object],
    target_sample_indices: tuple[int, ...],
    bootstrap_replicates: int,
    random_seed: int,
    source_identity: Mapping[str, object] | None = None,
    property_sampling_scope: str = "independent_supplemental_128",
    convergence_target_counts: tuple[int, ...] = (32, 64, 96, 128),
    analysis_mode: str = "independent_supplemental_128",
    include_convergence: bool = True,
) -> dict[str, Path]:
    scope_contract = _property_scope_contract(
        property_sampling_scope=property_sampling_scope,
        target_sample_indices=target_sample_indices,
        convergence_target_counts=convergence_target_counts,
    )
    if scope_contract and any(
        coverage.get(name) != value for name, value in scope_contract.items()
    ):
        raise ValueError("shared property coverage scope contract is missing or invalid")
    destination = output_dir
    destination.mkdir(parents=True, exist_ok=True)
    output_dir = destination / "runs" / f".building-{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=False)
    convergence_fields = (
        tuple(field for field in _CONVERGENCE_FIELDS if field != "successive_absolute_change")
        if property_sampling_scope == "shared_original_32"
        else _CONVERGENCE_FIELDS
    )
    table_specs = {
        "sample": (sample_rows, _SAMPLE_FIELDS),
        "sequence": (sequence_rows, _SEQUENCE_FIELDS),
        "residue": (residue_rows, _RESIDUE_FIELDS),
        "convergence": (convergence_rows, convergence_fields),
        "failures": ([asdict(failure) for failure in failures], _FAILURE_FIELDS),
    }
    paths: dict[str, Path] = {}
    npz_tables: dict[str, np.ndarray] = {}
    for table_name, (rows, fields) in table_specs.items():
        csv_path = output_dir / f"{table_name}.csv"
        json_path = output_dir / f"{table_name}.json"
        _atomic_write_bytes(csv_path, _csv_bytes(rows, fields))
        _atomic_write_bytes(
            json_path,
            (json.dumps(_json_safe(rows), indent=2, sort_keys=True) + "\n").encode(),
        )
        paths[f"{table_name}_csv"] = csv_path
        paths[f"{table_name}_json"] = json_path
        for field in fields:
            npz_tables[f"{table_name}__{field}"] = _npz_column(rows, field)

    tables_path = output_dir / "tables.npz"
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **npz_tables)
    _atomic_write_bytes(tables_path, buffer.getvalue())
    paths["tables_npz"] = tables_path

    coverage_path = output_dir / "coverage.json"
    _atomic_write_bytes(
        coverage_path,
        (json.dumps(_json_safe(coverage), indent=2, sort_keys=True) + "\n").encode(),
    )
    paths["coverage_json"] = coverage_path

    metadata = {
        "schema_version": 2,
        "scientific_scope": (
            "Descriptive expanded-ensemble geometry and local-contact outcomes; "
            "not a thermodynamic or free-energy estimate."
        ),
        "seed_scope": {
            "alignment_reference_sample_index": 0,
            "alignment_reference_source": "original_structure_root",
            "target_sample_indices": list(target_sample_indices),
            "target_source": (
                "original_structure_root"
                if property_sampling_scope == "shared_original_32"
                else "supplemental_structure_root"
            ),
            "reference_included_in_target_moments": (
                property_sampling_scope == "shared_original_32"
            ),
        },
        "units": {
            "radius_of_gyration": "angstrom",
            "contact_density": "fraction",
            "end_to_end_distance": "angstrom",
            "relative_shape_anisotropy": "dimensionless",
            "mean_reference_aligned_pairwise_rmsd": "angstrom",
            "binary_contact_entropy": "nats",
            "rmsf": "angstrom",
            "contact_number": "contacts",
            "contact_probability": "fraction",
            "local_q": "dimensionless",
            "within_sequence_z_scores": "standard deviations within sequence",
        },
        "definitions": {
            "contact": {
                "comparison": "distance < 8.0 angstrom",
                "minimum_sequence_separation": 3,
            },
            "missing_contact_statuses": {
                "measured": "at least one eligible pair contributes",
                "undefined_no_eligible_pairs": "no |i-j| >= 3 pair exists",
                "undefined_no_reference_contacts": (
                    "no eligible seed-0 contact exists for residue local-Q"
                ),
            },
            "structure_index_members": (
                "relative NPZ/PDB paths canonicalized within their configured "
                "structure root before coordinate reads"
            ),
            "variance": "sample variance across target structures (ddof=1)",
            "alignment": (
                "each target is Kabsch-aligned to original sample 0; sample 0 is "
                + (
                    "included in shared target moments"
                    if property_sampling_scope == "shared_original_32"
                    else "excluded from target moments"
                )
            ),
            "local_q": (
                "seed-0 contacts use |i-j| >= 3 and distance < 8 angstrom; "
                "weight=1/(1+exp(clip(5*(distance-1.2*reference_distance),-60,60)))"
            ),
            "convergence_order": (
                "ascending SHA-256 digest of the base-10 sample-index string; "
                "target counts are nested prefixes"
            ),
            "convergence_bootstrap": (
                "exact 2.5% and 97.5% percentile endpoints and their width from "
                "deterministic target-sample bootstrap means; incomplete target "
                "coverage is not estimable"
            ),
            "relative_sample_properties": (
                "population-standardized within sequence only when all requested "
                "target structures are present"
            ),
        },
        "bootstrap_replicates": bootstrap_replicates,
        "random_seed": random_seed,
        "source_identity": source_identity,
        "coverage": coverage,
        "outputs": {
            name: path.relative_to(output_dir).as_posix() for name, path in sorted(paths.items())
        },
        **scope_contract,
    }
    if analysis_mode == "same_trajectory_exploratory_32":
        metadata.update(
            {
                "analysis_mode": analysis_mode,
                "predictor_target_overlap": list(range(32)),
                "claim_limitation": (
                    "Predictors and targets use the same 32 trajectories; shared sampling noise "
                    "may make effect estimates optimistic."
                ),
                "disabled_analyses": [
                    "sample_count_convergence",
                    "trajectory_split_64_by_64",
                    "supplemental_activation_convergence",
                    "supplemental_state_conditional_analysis",
                    "calibrated_local_conformational_state_contrasts",
                ],
            }
        )
        metadata["seed_scope"] = {
            "alignment_reference_sample_index": 0,
            "alignment_reference_source": "original_structure_root",
            "target_sample_indices": list(range(32)),
            "target_source": "original_structure_root",
            "reference_included_in_target_moments": True,
        }
        metadata["definitions"]["convergence_order"] = "not applicable in exploratory_32"
        metadata["definitions"]["convergence_bootstrap"] = "not applicable in exploratory_32"
        if include_convergence or convergence_rows:
            raise ValueError("exploratory_32 must not publish convergence rows")
    metadata_path = output_dir / "metadata.json"
    _atomic_write_bytes(
        metadata_path,
        (json.dumps(_json_safe(metadata), indent=2, sort_keys=True) + "\n").encode(),
    )
    paths["metadata_json"] = metadata_path
    marker_payload = {
        "schema_version": 2,
        "artifact": "expanded_ensemble_properties",
        "target_sample_indices": list(target_sample_indices),
        "bootstrap_replicates": bootstrap_replicates,
        "random_seed": random_seed,
        "source_identity": source_identity,
        "artifact_sha256": {name: _sha256_file(path) for name, path in sorted(paths.items())},
        **scope_contract,
    }
    if analysis_mode == "same_trajectory_exploratory_32":
        marker_payload["analysis_mode"] = analysis_mode
        marker_payload["predictor_target_overlap"] = list(range(32))
    validate_canonical_property_publication(
        metadata=metadata,
        coverage=coverage,
        marker_artifact_sha256=marker_payload["artifact_sha256"],
    )
    fingerprint = hashlib.sha256(
        json.dumps(marker_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    marker_payload["fingerprint"] = fingerprint
    marker_path = output_dir / ".published.json"
    _atomic_write_bytes(
        marker_path,
        (json.dumps(marker_payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    paths["publication_marker_json"] = marker_path
    generation = destination / "runs" / fingerprint
    publication_path = destination / "manifest.json"
    if publication_path.is_file():
        try:
            current_publication = json.loads(publication_path.read_text())
            current_relative = current_publication.get("generation")
            current_generation = (
                destination / current_relative if isinstance(current_relative, str) else None
            )
            if (
                current_publication.get("fingerprint") == fingerprint
                and current_generation is not None
                and current_generation.resolve(strict=False) != destination.resolve(strict=False)
                and destination.resolve(strict=False)
                in current_generation.resolve(strict=False).parents
            ):
                generation = current_generation
        except (OSError, json.JSONDecodeError):
            pass
    if generation.exists():
        relative_members = tuple(
            path.relative_to(output_dir) for path in output_dir.iterdir() if path.is_file()
        )
        reusable = all(
            (generation / relative).is_file()
            and _sha256_file(generation / relative) == _sha256_file(output_dir / relative)
            for relative in relative_members
        )
        if reusable:
            shutil.rmtree(output_dir)
        else:
            generation = destination / "runs" / fingerprint / "repairs" / uuid.uuid4().hex
            generation.parent.mkdir(parents=True, exist_ok=True)
            os.replace(output_dir, generation)
    else:
        os.replace(output_dir, generation)
    final_paths = {name: generation / path.relative_to(output_dir) for name, path in paths.items()}
    publication = {
        "schema_version": 2,
        "artifact": "expanded_ensemble_properties",
        "fingerprint": fingerprint,
        "generation": generation.relative_to(destination).as_posix(),
        "marker_sha256": _sha256_file(final_paths["publication_marker_json"]),
        **scope_contract,
    }
    _atomic_write_bytes(
        publication_path,
        (json.dumps(publication, indent=2, sort_keys=True) + "\n").encode(),
    )
    final_paths["publication_manifest_json"] = publication_path
    return final_paths


def _write_unpublished_property_diagnostics(
    *,
    output_dir: Path,
    failures: list[FailureRecord],
    coverage: Mapping[str, object],
    scope_contract: Mapping[str, object],
) -> Path:
    """Atomically retain failed shared-run diagnostics without publishing tables."""

    destination = Path(output_dir)
    runs_root = destination / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    building = runs_root / f".building-{uuid.uuid4().hex}"
    building.mkdir(parents=False, exist_ok=False)
    try:
        failure_rows = [asdict(failure) for failure in failures]
        artifact_paths = {
            "coverage_json": building / "coverage.json",
            "failures_csv": building / "failures.csv",
            "failures_json": building / "failures.json",
        }
        _atomic_write_bytes(
            artifact_paths["coverage_json"],
            (json.dumps(_json_safe(coverage), indent=2, sort_keys=True) + "\n").encode(),
        )
        _atomic_write_bytes(
            artifact_paths["failures_csv"],
            _csv_bytes(failure_rows, _FAILURE_FIELDS),
        )
        _atomic_write_bytes(
            artifact_paths["failures_json"],
            (json.dumps(_json_safe(failure_rows), indent=2, sort_keys=True) + "\n").encode(),
        )
        marker = {
            "schema_version": 1,
            "artifact": "expanded_ensemble_property_diagnostics",
            "status": "unpublished",
            "publishable": False,
            "artifact_sha256": {
                name: _sha256_file(path) for name, path in sorted(artifact_paths.items())
            },
            **scope_contract,
        }
        fingerprint = hashlib.sha256(
            json.dumps(marker, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        marker["fingerprint"] = fingerprint
        marker_path = building / ".unpublished.json"
        _atomic_write_bytes(
            marker_path,
            (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode(),
        )

        unpublished_root = runs_root / "unpublished"
        unpublished_root.mkdir(parents=True, exist_ok=True)
        generation = unpublished_root / fingerprint
        if generation.exists():
            expected_members = {*artifact_paths.values(), marker_path}
            reusable = {path.name for path in generation.iterdir()} == {
                path.name for path in expected_members
            } and all(
                (generation / path.name).is_file()
                and _sha256_file(generation / path.name) == _sha256_file(path)
                for path in expected_members
            )
            if reusable:
                shutil.rmtree(building)
            else:
                generation = unpublished_root / f"{fingerprint}-repair-{uuid.uuid4().hex}"
                os.replace(building, generation)
        else:
            os.replace(building, generation)
        return generation
    finally:
        if building.exists():
            shutil.rmtree(building)


def _csv_bytes(
    rows: list[dict[str, object]],
    fields: tuple[str, ...],
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    return buffer.getvalue().encode()


def _csv_value(value: object) -> object:
    if isinstance(value, tuple | list):
        return json.dumps(_json_safe(value), separators=(",", ":"))
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if value is None:
        return ""
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _npz_column(rows: list[dict[str, object]], field: str) -> np.ndarray:
    values = [row.get(field) for row in rows]
    if not values:
        return np.asarray([], dtype=np.float64)
    if field == "bootstrap_seed":
        return np.asarray(
            ["" if value is None else str(value) for value in values],
            dtype=np.str_,
        )
    if all(
        value is None or isinstance(value, bool | int | float | np.bool_ | np.integer | np.floating)
        for value in values
    ):
        return np.asarray(
            [np.nan if value is None else float(value) for value in values],
            dtype=np.float64,
        )
    return np.asarray(
        [
            json.dumps(_json_safe(value), separators=(",", ":"))
            if isinstance(value, tuple | list)
            else ""
            if value is None
            else str(value)
            for value in values
        ],
        dtype=np.str_,
    )


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
