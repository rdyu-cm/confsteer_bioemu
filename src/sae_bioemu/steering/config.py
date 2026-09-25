"""Leakage-safe property evidence selection and sequence split configuration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 compatibility.

    class StrEnum(str, Enum):
        pass


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PropertyScope(StrEnum):
    """How a property is defined over a protein's sampled structures."""

    SINGLE_STRUCTURE = "single_structure"
    ENSEMBLE = "ensemble"
    WITHIN_SEQUENCE_RELATIVE = "within_sequence_relative"
    EXTERNAL_LABEL = "external_label"


class ModelFamily(StrEnum):
    """Generator family supplying the representation evidence."""

    BIOEMU = "bioemu"
    ALPHAFOLD3 = "alphafold3"


@dataclass(frozen=True)
class PropertyEvidence:
    """A held-out property association from one analyzed representation site."""

    property_name: str
    scope: PropertyScope
    model_family: ModelFamily
    activation_source: str
    layer: str
    captured_call: int
    analysis_fingerprint: str
    univariate_heldout_spearman: float | None = None
    pls1_heldout_spearman: float | None = None
    ridge_heldout_spearman: float | None = None
    feature_index: int | None = None

    def __post_init__(self) -> None:
        for name in ("property_name", "activation_source", "layer"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.scope, PropertyScope):
            raise ValueError("scope must be a PropertyScope")
        if not isinstance(self.model_family, ModelFamily):
            raise ValueError("model_family must be a ModelFamily")
        if not isinstance(self.captured_call, int) or isinstance(self.captured_call, bool):
            raise ValueError("captured_call must be a nonnegative integer")
        if self.captured_call < 0:
            raise ValueError("captured_call must be nonnegative")
        if self.feature_index is not None and (
            not isinstance(self.feature_index, int)
            or isinstance(self.feature_index, bool)
            or self.feature_index < 0
        ):
            raise ValueError("feature_index must be a nonnegative integer or null")
        _require_fingerprint("analysis_fingerprint", self.analysis_fingerprint)
        if not self.heldout_metrics:
            raise ValueError("at least one held-out metric is required")
        for name, value in self.heldout_metrics:
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")

    @property
    def heldout_metrics(self) -> tuple[tuple[str, float], ...]:
        return tuple(
            (name, value)
            for name, value in (
                ("univariate_heldout_spearman", self.univariate_heldout_spearman),
                ("pls1_heldout_spearman", self.pls1_heldout_spearman),
                ("ridge_heldout_spearman", self.ridge_heldout_spearman),
            )
            if value is not None
        )


@dataclass(frozen=True)
class PropertySelection:
    """Evidence selected for a direct-prediction or steering manifest."""

    evidence: PropertyEvidence
    ranking_score: float
    selected_metric_name: str
    selected_metric_value: float
    steering_eligible: bool
    steering_eligibility_reason: str
    property_definition: str = ""
    aggregation_unit: str = ""
    units: str = ""
    direction: str = ""
    training_range: tuple[float, float] | None = None
    source_fingerprints: tuple[str, ...] = ()
    training_sequence_ids: tuple[str, ...] = ()

    @property
    def property_name(self) -> str:
        return self.evidence.property_name


@dataclass(frozen=True)
class _RankedProperties:
    direct_prediction: tuple[PropertySelection, ...]
    steering_eligible: tuple[PropertySelection, ...]


@dataclass(frozen=True)
class SequenceSplit:
    """Disjoint sequence groups for training, validation, and final testing."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        roles = (self.train, self.validation, self.test)
        sequence_ids = tuple(sequence_id for role in roles for sequence_id in role)
        if any(not isinstance(sequence_id, str) or not sequence_id for sequence_id in sequence_ids):
            raise ValueError("sequence IDs must be non-empty strings")
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError("a sequence may not appear in multiple split roles")


@dataclass(frozen=True)
class SteeringManifest:
    """Canonical, fingerprinted property-selection manifest."""

    schema_version: int
    analysis_fingerprint: str
    direct_prediction: tuple[PropertySelection, ...]
    steering_eligible: tuple[PropertySelection, ...]
    split: SequenceSplit
    fingerprint: str
    family_clusters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported steering manifest schema_version")
        _require_fingerprint("analysis_fingerprint", self.analysis_fingerprint)
        _require_fingerprint("fingerprint", self.fingerprint)
        if self.direct_prediction:
            raise ValueError("active steering manifests must not contain direct_prediction rows")
        for selection in (*self.direct_prediction, *self.steering_eligible):
            _require_selection_metadata(selection)
            if selection.evidence.analysis_fingerprint != self.analysis_fingerprint:
                raise ValueError("selection evidence analysis fingerprint does not match manifest")
            _require_training_sequence_ids(selection, self.split)
        for selection in self.steering_eligible:
            if (
                not selection.steering_eligible
                or selection.evidence.scope is not PropertyScope.SINGLE_STRUCTURE
            ):
                raise ValueError("steering manifest contains an ineligible steering selection")
        clusters = dict(self.family_clusters)
        sequence_ids = (*self.split.train, *self.split.validation, *self.split.test)
        if self.family_clusters:
            if (
                len(clusters) != len(self.family_clusters)
                or set(clusters) != set(sequence_ids)
                or any(
                    not isinstance(sequence_id, str)
                    or not sequence_id
                    or not isinstance(cluster, str)
                    or not cluster
                    for sequence_id, cluster in self.family_clusters
                )
            ):
                raise ValueError("manifest family_clusters must map every sequence exactly once")
            role_clusters = [
                {clusters[sequence_id] for sequence_id in role}
                for role in (self.split.train, self.split.validation, self.split.test)
            ]
            if any(
                role_clusters[left] & role_clusters[right]
                for left in range(len(role_clusters))
                for right in range(left + 1, len(role_clusters))
            ):
                raise ValueError("manifest family clusters may not cross split roles")
            object.__setattr__(self, "family_clusters", tuple(sorted(self.family_clusters)))
        if self.fingerprint != _manifest_fingerprint(self.to_payload()):
            raise ValueError("manifest fingerprint does not match canonical content")

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "analysis_fingerprint": self.analysis_fingerprint,
            "direct_prediction": [_selection_payload(row) for row in self.direct_prediction],
            "steering_eligible": [_selection_payload(row) for row in self.steering_eligible],
            "split": {
                "train": list(self.split.train),
                "validation": list(self.split.validation),
                "test": list(self.split.test),
            },
        }
        if self.family_clusters:
            payload["family_clusters"] = dict(self.family_clusters)
        return payload


def rank_property_evidence(
    evidence: Iterable[PropertyEvidence], limit: int = 5
) -> _RankedProperties:
    """Rank held-out evidence and keep only single-structure steering targets."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    selections = tuple(_selection(row) for row in evidence)
    ranked = tuple(sorted(selections, key=lambda row: (-row.ranking_score, row.property_name)))
    steering_eligible = tuple(row for row in ranked if row.steering_eligible)[:limit]
    return _RankedProperties((), steering_eligible)


def _selection(evidence: PropertyEvidence) -> PropertySelection:
    metric_name, metric_value = max(
        evidence.heldout_metrics, key=lambda item: (abs(item[1]), item[0])
    )
    eligible = evidence.scope is PropertyScope.SINGLE_STRUCTURE
    reason = (
        "defined on one completed structure"
        if eligible
        else f"{evidence.scope.value} properties are not eligible for steering"
    )
    return PropertySelection(
        evidence=evidence,
        ranking_score=abs(metric_value),
        selected_metric_name=metric_name,
        selected_metric_value=metric_value,
        steering_eligible=eligible,
        steering_eligibility_reason=reason,
    )


def build_sequence_split(
    rows: Iterable[Mapping[str, object]],
    train_count: int,
    validation_count: int,
    test_count: int,
    seed: int,
) -> SequenceSplit:
    """Create a deterministic, length/property-stratified sequence split."""
    counts = (train_count, validation_count, test_count)
    if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts):
        raise ValueError("split counts must be nonnegative integers")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    normalized = tuple(sorted((_normalized_split_row(row) for row in rows), key=lambda row: row[0]))
    if len({row[0] for row in normalized}) != len(normalized):
        raise ValueError("duplicate sequence IDs are not allowed")
    if sum(counts) != len(normalized):
        raise ValueError("split counts must equal the number of sequences")
    if not normalized:
        return SequenceSplit((), (), ())

    length_values = np.asarray([row[1] for row in normalized], dtype=float)
    property_values = np.asarray([row[2] for row in normalized], dtype=float)
    cells: dict[tuple[int, int], list[str]] = {}
    for (sequence_id, _, _), length_bin, property_bin in zip(
        normalized,
        _quantile_bins(length_values),
        _quantile_bins(property_values),
        strict=True,
    ):
        cells.setdefault((int(length_bin), int(property_bin)), []).append(sequence_id)

    generator = np.random.default_rng(seed)
    ordered_cells = tuple(cells[key] for key in sorted(cells))
    for cell in ordered_cells:
        generator.shuffle(cell)
    allocations = _proportional_cell_allocations(tuple(len(cell) for cell in ordered_cells), counts)
    roles: tuple[list[str], list[str], list[str]] = ([], [], [])
    for cell, allocation in zip(ordered_cells, allocations, strict=True):
        offset = 0
        for role, count in zip(roles, allocation, strict=True):
            role.extend(cell[offset : offset + count])
            offset += count
    return SequenceSplit(
        train=tuple(roles[0]),
        validation=tuple(roles[1]),
        test=tuple(roles[2]),
    )


def load_steering_manifest(
    path: str | Path, expected_analysis_fingerprint: str
) -> SteeringManifest:
    """Load a canonical manifest, rejecting stale analysis or split leakage."""
    _require_fingerprint("expected_analysis_fingerprint", expected_analysis_fingerprint)
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load steering manifest: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("steering manifest must contain a JSON object")
    fingerprint = payload.pop("fingerprint", None)
    if _manifest_fingerprint(payload) != fingerprint:
        raise ValueError("manifest fingerprint does not match canonical content")
    try:
        manifest = SteeringManifest(
            schema_version=payload["schema_version"],
            analysis_fingerprint=payload["analysis_fingerprint"],
            direct_prediction=tuple(
                _selection_from_payload(item) for item in payload["direct_prediction"]
            ),
            steering_eligible=tuple(
                _selection_from_payload(item) for item in payload["steering_eligible"]
            ),
            split=_split_from_payload(payload["split"]),
            fingerprint=fingerprint,
            family_clusters=_family_clusters_from_payload(payload.get("family_clusters")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
    if manifest.analysis_fingerprint != expected_analysis_fingerprint:
        raise ValueError("manifest analysis fingerprint is stale")
    return manifest


def write_steering_manifest(
    path: str | Path,
    *,
    analysis_fingerprint: str,
    steering_eligible: Iterable[PropertySelection],
    split: SequenceSplit,
    family_clusters: Mapping[str, str] | Iterable[tuple[str, str]] = (),
    direct_prediction: Iterable[PropertySelection] = (),
    schema_version: int = 1,
) -> str:
    """Atomically author a manifest that ``load_steering_manifest`` accepts unchanged.

    Every read-time invariant is enforced here so an unusable manifest fails at authoring
    rather than inside a generation job: the payload is built by the same canonical helpers
    the loader verifies against, and the written file is loaded back from its temporary path
    before it is published.

    The ranking fields of each selection (``ranking_score``, ``selected_metric_name``,
    ``selected_metric_value``, ``steering_eligible``, ``steering_eligibility_reason``) are a
    pure function of its evidence and are re-derived here; only the evidence and the curated
    metadata fields are authoritative inputs.

    Returns the manifest fingerprint.
    """
    _require_fingerprint("analysis_fingerprint", analysis_fingerprint)
    if tuple(direct_prediction):
        raise ValueError("active steering manifests must not contain direct_prediction rows")
    if not isinstance(split, SequenceSplit):
        raise ValueError("split must be a SequenceSplit")
    selections = tuple(_canonical_selection(row) for row in steering_eligible)
    if not selections:
        raise ValueError("a steering manifest requires at least one steering_eligible selection")
    clusters = tuple(
        sorted(
            family_clusters.items()
            if isinstance(family_clusters, Mapping)
            else (tuple(item) for item in family_clusters)
        )
    )
    canonical_payload: dict[str, Any] = {
        "schema_version": schema_version,
        "analysis_fingerprint": analysis_fingerprint,
        "direct_prediction": [],
        "steering_eligible": [_selection_payload(row) for row in selections],
        "split": {
            "train": list(split.train),
            "validation": list(split.validation),
            "test": list(split.test),
        },
    }
    if clusters:
        canonical_payload["family_clusters"] = dict(clusters)
    manifest = SteeringManifest(
        schema_version=schema_version,
        analysis_fingerprint=analysis_fingerprint,
        direct_prediction=(),
        steering_eligible=selections,
        split=split,
        fingerprint=_manifest_fingerprint(canonical_payload),
        family_clusters=clusters,
    )
    payload = manifest.to_payload()
    payload["fingerprint"] = manifest.fingerprint
    _atomically_write_verified_manifest(Path(path), payload, manifest)
    return manifest.fingerprint


def _atomically_write_verified_manifest(
    path: Path, payload: Mapping[str, Any], manifest: SteeringManifest
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if load_steering_manifest(temporary, manifest.analysis_fingerprint) != manifest:
            raise ValueError("written steering manifest does not reload to the same manifest")
        temporary.replace(path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _canonical_selection(selection: PropertySelection) -> PropertySelection:
    if not isinstance(selection, PropertySelection) or not isinstance(
        selection.evidence, PropertyEvidence
    ):
        raise ValueError("steering selections must be PropertySelection instances")
    training_range = selection.training_range
    if training_range is not None:
        if not isinstance(training_range, tuple | list):
            raise ValueError("selection training_range must contain lower and upper bounds")
        training_range = tuple(training_range)
    derived = _selection(selection.evidence)
    return PropertySelection(
        evidence=derived.evidence,
        ranking_score=derived.ranking_score,
        selected_metric_name=derived.selected_metric_name,
        selected_metric_value=derived.selected_metric_value,
        steering_eligible=derived.steering_eligible,
        steering_eligibility_reason=derived.steering_eligibility_reason,
        property_definition=selection.property_definition,
        aggregation_unit=selection.aggregation_unit,
        units=selection.units,
        direction=selection.direction,
        training_range=training_range,
        source_fingerprints=tuple(selection.source_fingerprints),
        training_sequence_ids=tuple(selection.training_sequence_ids),
    )


def _family_clusters_from_payload(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError("manifest family_clusters must be an object")
    return tuple(sorted(value.items()))


def _normalized_split_row(row: Mapping[str, object]) -> tuple[str, float, float]:
    try:
        sequence_id = row["sequence_id"]
        length = row["sequence_length"]
        property_value = row["property_value"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "split rows require sequence_id, sequence_length, and property_value"
        ) from error
    if not isinstance(sequence_id, str) or not sequence_id:
        raise ValueError("sequence_id must be non-empty")
    if any(
        not isinstance(value, int | float) or isinstance(value, bool)
        for value in (length, property_value)
    ):
        raise ValueError("sequence_length and property_value must be numeric")
    if not math.isfinite(length) or not math.isfinite(property_value) or length <= 0:
        raise ValueError("split row values must be finite and sequence_length positive")
    return sequence_id, float(length), float(property_value)


def _quantile_bins(values: np.ndarray, bins: int = 5) -> np.ndarray:
    if len(values) == 1 or np.all(values == values[0]):
        return np.zeros(len(values), dtype=int)
    boundaries = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
    return np.searchsorted(boundaries, values, side="right")


def _proportional_cell_allocations(
    cell_sizes: tuple[int, ...], role_counts: tuple[int, int, int]
) -> tuple[tuple[int, int, int], ...]:
    total = sum(cell_sizes)
    if total != sum(role_counts) or total <= 0:
        raise ValueError("cell and role totals must be equal and positive")
    return _minimum_cost_allocation(cell_sizes, role_counts)


def _split_from_payload(payload: object) -> SequenceSplit:
    if not isinstance(payload, dict):
        raise ValueError("manifest split must be an object")
    return SequenceSplit(
        train=tuple(payload["train"]),
        validation=tuple(payload["validation"]),
        test=tuple(payload["test"]),
    )


def _minimum_cost_allocation(
    cell_sizes: tuple[int, ...], role_counts: tuple[int, int, int]
) -> tuple[tuple[int, int, int], ...]:
    total = sum(cell_sizes)
    source = 0
    cell_offset = 1
    role_offset = cell_offset + len(cell_sizes)
    sink = role_offset + len(role_counts)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    tracked: list[_FlowEdge] = []
    lex_base = total + 1
    primary_scale = lex_base ** (len(cell_sizes) * len(role_counts))
    for cell, cell_size in enumerate(cell_sizes):
        _add_flow_edge(graph, source, cell_offset + cell, cell_size, 0)
        for role, role_count in enumerate(role_counts):
            numerator = cell_size * role_count
            for unit in range(1, cell_size + 1):
                primary_cost = (unit * total - numerator) ** 2 - (
                    (unit - 1) * total - numerator
                ) ** 2
                lex_cost = lex_base ** (
                    len(cell_sizes) * len(role_counts) - 1 - (cell * len(role_counts) + role)
                )
                cost = primary_cost * primary_scale + lex_cost
                tracked.append(
                    _add_flow_edge(
                        graph, cell_offset + cell, role_offset + role, 1, cost, cell, role
                    )
                )
    for role, role_count in enumerate(role_counts):
        _add_flow_edge(graph, role_offset + role, sink, role_count, 0)

    for _ in range(total):
        previous = _shortest_flow_path(graph, source, sink)
        if previous is None:
            raise RuntimeError("could not allocate stratified sequence split")
        node = sink
        while node != source:
            edge = previous[node]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            edge.flow += edge.direction
            node = graph[node][edge.reverse].to
    allocation = [[0, 0, 0] for _ in cell_sizes]
    for edge in tracked:
        allocation[edge.cell][edge.role] += 1 - edge.capacity
    return tuple(tuple(row) for row in allocation)


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: int
    cell: int = -1
    role: int = -1
    direction: int = 0
    flow: int = 0


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    source: int,
    target: int,
    capacity: int,
    cost: int,
    cell: int = -1,
    role: int = -1,
) -> _FlowEdge:
    forward = _FlowEdge(target, len(graph[target]), capacity, cost, cell, role, 1)
    reverse = _FlowEdge(source, len(graph[source]), 0, -cost, cell, role, -1)
    graph[source].append(forward)
    graph[target].append(reverse)
    return forward


def _shortest_flow_path(
    graph: list[list[_FlowEdge]], source: int, sink: int
) -> list[_FlowEdge | None] | None:
    distance: list[int | None] = [None] * len(graph)
    previous: list[_FlowEdge | None] = [None] * len(graph)
    distance[source] = 0
    queue = deque([source])
    queued = {source}
    while queue:
        node = queue.popleft()
        queued.remove(node)
        for edge in graph[node]:
            if not edge.capacity:
                continue
            candidate = distance[node] + edge.cost
            if distance[edge.to] is None or candidate < distance[edge.to]:
                distance[edge.to] = candidate
                previous[edge.to] = edge
                if edge.to not in queued:
                    queue.append(edge.to)
                    queued.add(edge.to)
    return previous if previous[sink] is not None else None


def _selection_payload(selection: PropertySelection) -> dict[str, object]:
    evidence = selection.evidence
    evidence_payload = {
        "property_name": evidence.property_name,
        "scope": evidence.scope.value,
        "model_family": evidence.model_family.value,
        "activation_source": evidence.activation_source,
        "layer": evidence.layer,
        "captured_call": evidence.captured_call,
        "analysis_fingerprint": evidence.analysis_fingerprint,
        "univariate_heldout_spearman": evidence.univariate_heldout_spearman,
        "pls1_heldout_spearman": evidence.pls1_heldout_spearman,
        "ridge_heldout_spearman": evidence.ridge_heldout_spearman,
    }
    if evidence.feature_index is not None:
        evidence_payload["feature_index"] = evidence.feature_index
    return {
        "evidence": evidence_payload,
        "ranking_score": selection.ranking_score,
        "selected_metric_name": selection.selected_metric_name,
        "selected_metric_value": selection.selected_metric_value,
        "steering_eligible": selection.steering_eligible,
        "steering_eligibility_reason": selection.steering_eligibility_reason,
        "property_definition": selection.property_definition,
        "aggregation_unit": selection.aggregation_unit,
        "units": selection.units,
        "direction": selection.direction,
        "training_range": list(selection.training_range) if selection.training_range else None,
        "source_fingerprints": list(selection.source_fingerprints),
        "training_sequence_ids": list(selection.training_sequence_ids),
    }


def _selection_from_payload(payload: object) -> PropertySelection:
    if not isinstance(payload, dict) or not isinstance(payload.get("evidence"), dict):
        raise ValueError("manifest selection must contain evidence")
    evidence_payload = payload["evidence"]
    evidence = PropertyEvidence(
        property_name=evidence_payload["property_name"],
        scope=PropertyScope(evidence_payload["scope"]),
        model_family=ModelFamily(evidence_payload["model_family"]),
        activation_source=evidence_payload["activation_source"],
        layer=evidence_payload["layer"],
        captured_call=evidence_payload["captured_call"],
        analysis_fingerprint=evidence_payload["analysis_fingerprint"],
        univariate_heldout_spearman=evidence_payload.get("univariate_heldout_spearman"),
        pls1_heldout_spearman=evidence_payload.get("pls1_heldout_spearman"),
        ridge_heldout_spearman=evidence_payload.get("ridge_heldout_spearman"),
        feature_index=evidence_payload.get("feature_index"),
    )
    selection = _selection(evidence)
    training_range = payload.get("training_range")
    if training_range is not None:
        training_range = tuple(training_range)
    selection = PropertySelection(
        evidence=selection.evidence,
        ranking_score=selection.ranking_score,
        selected_metric_name=selection.selected_metric_name,
        selected_metric_value=selection.selected_metric_value,
        steering_eligible=selection.steering_eligible,
        steering_eligibility_reason=selection.steering_eligibility_reason,
        property_definition=payload.get("property_definition", ""),
        aggregation_unit=payload.get("aggregation_unit", ""),
        units=payload.get("units", ""),
        direction=payload.get("direction", ""),
        training_range=training_range,
        source_fingerprints=tuple(payload.get("source_fingerprints", ())),
        training_sequence_ids=tuple(payload.get("training_sequence_ids", ())),
    )
    if _selection_payload(selection) != payload:
        raise ValueError("manifest selection is not canonical")
    return selection


def _manifest_fingerprint(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_selection_metadata(selection: PropertySelection) -> None:
    for name in ("property_definition", "aggregation_unit", "units", "direction"):
        value = getattr(selection, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"selection {name} must be a non-empty string")
    if selection.training_range is None or len(selection.training_range) != 2:
        raise ValueError("selection training_range must contain lower and upper bounds")
    lower, upper = selection.training_range
    if not all(
        isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
        for value in (lower, upper)
    ):
        raise ValueError("selection training_range must be finite")
    if lower >= upper:
        raise ValueError("selection training_range must be increasing")
    if not selection.source_fingerprints:
        raise ValueError("selection source_fingerprints must be non-empty")
    for fingerprint in selection.source_fingerprints:
        _require_fingerprint("selection source_fingerprint", fingerprint)
    if not selection.training_sequence_ids or any(
        not sequence_id for sequence_id in selection.training_sequence_ids
    ):
        raise ValueError("selection training_sequence_ids must be non-empty")


def _require_training_sequence_ids(selection: PropertySelection, split: SequenceSplit) -> None:
    training_ids = selection.training_sequence_ids
    if len(set(training_ids)) != len(training_ids):
        raise ValueError("selection training_sequence_ids must not contain duplicates")
    if not set(training_ids).issubset(split.train):
        raise ValueError("selection training_sequence_ids must be contained in split training IDs")
    if set(training_ids) & (set(split.validation) | set(split.test)):
        raise ValueError(
            "selection training_sequence_ids must be disjoint from validation and test IDs"
        )


def _require_fingerprint(name: str, value: object) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 fingerprint")
