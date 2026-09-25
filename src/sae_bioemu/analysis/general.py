from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, colormaps, colors
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy import stats as scipy_stats

from sae_bioemu.analysis.concepts import (
    ResidueFeatureActivationAccumulator,
    analyze_interpro_concepts,
)
from sae_bioemu.analysis.prior_probe_comparison import (
    ProbeComparisonSeries,
    build_normalized_comparison_summary,
    write_probe_comparison,
)
from sae_bioemu.analysis.property_scatter import (
    stable_sequence_color,
    write_raw_relative_property_scatter_pairs,
)
from sae_bioemu.analysis.rmsf_probe import (
    DEFAULT_RIDGE_ALPHAS,
    DEFAULT_RMSF_FEATURE_COUNTS,
    PredictionRowKey,
    RidgeProbeModel,
    RMSFProbeResult,
    deterministic_group_folds,
    predict_ridge_model,
    run_grouped_rmsf_probe,
    run_grouped_rmsf_probe_sweep,
    write_rmsf_json_atomic,
    write_rmsf_probe_artifacts,
    write_rmsf_probe_sweep_artifacts,
    write_rmsf_structure_artifacts,
)
from sae_bioemu.analysis.sampling_scope import StructureSamplingScope, require_equilibrium_source
from sae_bioemu.bioemu.extractor import BioEmuRuntimeActivationSampler, SequenceRecord
from sae_bioemu.core.config import ProjectConfig
from sae_bioemu.core.store import ActivationStore
from sae_bioemu.core.trainer import SAETrainer, sequence_metadata
from sae_bioemu.core.types import ActivationRecord

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
PLOT_DPI = 300
VOLCANO_NEG_LOG10_CAP = 50.0
EARLY_TIMESTEP_MAX = 0.25
LATE_TIMESTEP_MIN = 0.75
RMSF_COORDINATE_OBSERVATIONS_PER_RESIDUE = 10
BIOEMU_TIMESTEP_AXIS_LABEL = "normalized timestep"
ESM3_LAYER_AXIS_LABEL = "layer position"
CROSS_ABLATION_ANALYSIS_PLOTS = (
    ("rmsf_sparsity_sweep", "rmsf_probe/sparsity_sweep.png", "RMSF Probe Sparsity Sweep"),
    (
        "selected_feature_timestep_profiles_overlay",
        "selected_feature_timestep_profiles/selected_feature_timestep_profiles_overlay.png",
        "Selected Feature Timestep Profiles",
    ),
    ("interpro_top_domain_f1", "interpro_concepts/top_domain_f1.png", "InterPro Domain F1"),
    (
        "rmsf_observed_vs_predicted",
        "rmsf_probe/observed_vs_predicted.png",
        "RMSF Observed vs Predicted",
    ),
    (
        "rmsf_selected_feature_stability",
        "rmsf_probe/selected_feature_stability.png",
        "RMSF Probe Feature Stability",
    ),
    (
        "property_predictive_features",
        "property_importance/all_property_predictive_features.png",
        "Structural Property Predictive Features",
    ),
    (
        "feature_activation_sparsity",
        "feature_summaries/active_fraction_vs_max_activation.png",
        "Feature Activation and Sparsity",
    ),
)


@dataclass(frozen=True)
class StructurePropertyResult:
    """Paths and counts produced by post-hoc SAE structure analysis."""

    output_dir: Path
    property_report_path: Path
    volcano_dir: Path
    logo_dir: Path
    num_sequences_with_rmsf: int
    num_properties_tested: int


@dataclass(frozen=True)
class AblationComparisonResult:
    """Artifacts produced by comparing completed SAE ablation runs."""

    output_dir: Path
    summary_json_path: Path
    summary_csv_path: Path
    plots_dir: Path
    num_runs: int


@dataclass(frozen=True)
class StructureIndexEntry:
    """One final BioEmu sample linked to a sequence and sample index."""

    sequence_id: str
    sample_index: int
    sequence: str
    npz_path: str | None = None
    pdb_path: str | None = None
    pos_units: str | None = None
    pdb_units: str | None = None


@dataclass(frozen=True)
class RMSFReference:
    """BioEmu ensemble used as the source of per-residue RMSF labels."""

    root: Path
    entries: tuple[StructureIndexEntry, ...]
    values_by_sequence: dict[str, np.ndarray]


@dataclass
class FeatureGroupStats:
    """Running per-feature sums for one sequence/property group."""

    high_sums: torch.Tensor
    high_counts: torch.Tensor
    low_sums: torch.Tensor
    low_counts: torch.Tensor


@dataclass(frozen=True)
class TokenProperty:
    """Binary token property used for feature-association volcano plots."""

    name: str
    description: str
    labels_by_sequence: dict[str, np.ndarray] | None = None
    label_fn: Any | None = None
    high_label: str = "high"
    low_label: str = "low"


@dataclass(frozen=True)
class ContinuousTokenProperty:
    """Continuous token property used for predictive feature ranking."""

    name: str
    description: str
    values_by_sequence: dict[str, np.ndarray] | None = None
    value_fn: Any | None = None
    units: str = ""
    aggregation_level: str = "token"
    timestep_bin: str | None = None
    normalize_features_within_sequence: bool = False


@dataclass(order=True)
class FeatureExampleCandidate:
    """Top activating residue-token example retained for logo construction."""

    activation: float
    sequence_id: str
    residue_index: int
    timestep: float
    sample_index: int | None


@dataclass
class RuntimeFeatureAnalysisPass:
    """Outputs collected from one deterministic runtime activation replay."""

    stats_by_property: dict[str, dict[str, FeatureGroupStats]]
    examples_by_feature: dict[int, list[FeatureExampleCandidate]]
    feature_summary: list[dict[str, float | int]]
    timestep_profiles: dict[int, list[dict[str, float | int]]]
    concept_metrics: dict[str, list[dict[str, float | int]]]
    property_importance: dict[str, list[dict[str, float | int | str]]]
    prediction_sample: PredictionSample | None
    reconstruction_metrics: dict[str, float | int] | None = None
    structure_feature_summaries: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)
    structure_input_prior_summaries: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)


@dataclass
class FeaturePredictionAccumulator:
    """Streaming feature-property sums for continuous predictive metrics."""

    count: int
    sum_y: float
    sum_y2: float
    sum_x: torch.Tensor
    sum_x2: torch.Tensor
    sum_xy: torch.Tensor

    @classmethod
    def create(cls, n_latents: int) -> FeaturePredictionAccumulator:
        return cls(
            count=0,
            sum_y=0.0,
            sum_y2=0.0,
            sum_x=torch.zeros(n_latents),
            sum_x2=torch.zeros(n_latents),
            sum_xy=torch.zeros(n_latents),
        )


@dataclass
class SampleFeatureAggregate:
    """Per-structure feature aggregates for scalar sample properties."""

    count: int
    sums: torch.Tensor
    property_values: dict[str, float]


@dataclass
class SampleFeaturePredictionAccumulator:
    """Feature aggregates keyed by one sequence/sample structure."""

    n_latents: int
    aggregates: dict[tuple[str, int, str], SampleFeatureAggregate]

    @classmethod
    def create(cls, n_latents: int) -> SampleFeaturePredictionAccumulator:
        return cls(n_latents=n_latents, aggregates={})


@dataclass
class PredictionSample:
    """Small deterministic in-memory sample for rank/AUC plots."""

    latents: np.ndarray
    property_values: dict[str, np.ndarray]
    group_ids: np.ndarray | None = None
    row_keys: tuple[PredictionRowKey | None, ...] = ()
    latents_by_property: dict[str, np.ndarray] = field(default_factory=dict)
    group_ids_by_property: dict[str, np.ndarray] = field(default_factory=dict)
    row_keys_by_property: dict[str, tuple[PredictionRowKey | None, ...]] = field(
        default_factory=dict
    )
    sampling_identity: dict[str, Any] = field(default_factory=dict)
    sampling_identity_by_property: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sampling_identity:
            rows = int(np.asarray(self.latents).shape[0])
            self.sampling_identity = {
                "strategy": "provided_rows",
                "max_rows": rows,
                "rng_seed": None,
                "seen": rows,
                "retained_rows": rows,
            }

    def latents_for(self, property_name: str) -> np.ndarray:
        return self.latents_by_property.get(property_name, self.latents)

    def groups_for(self, property_name: str) -> np.ndarray | None:
        return self.group_ids_by_property.get(property_name, self.group_ids)

    def keys_for(self, property_name: str) -> tuple[PredictionRowKey | None, ...]:
        property_keys = getattr(self, "row_keys_by_property", {})
        return property_keys.get(property_name, getattr(self, "row_keys", ()))

    def sampling_for(self, property_name: str) -> dict[str, Any]:
        property_sampling = getattr(self, "sampling_identity_by_property", {})
        return property_sampling.get(
            property_name,
            getattr(self, "sampling_identity", {}),
        )


@dataclass
class PredictionReservoir:
    """Deterministic bounded sample of latent rows, discarded after analysis."""

    max_rows: int
    rng_seed: int = 0
    seen: int = 0
    rows: list[np.ndarray] | None = None
    values: list[dict[str, float]] | None = None
    groups: list[str] | None = None
    rng: np.random.Generator | None = None
    priorities: list[float] | None = None
    row_keys: list[PredictionRowKey | None] | None = None
    sampling_strategy: str | None = None

    def __post_init__(self) -> None:
        self.rows = []
        self.values = []
        self.groups = []
        self.rng = np.random.default_rng(self.rng_seed)
        self.priorities = []
        self.row_keys = []

    def add(
        self,
        latent_row: torch.Tensor,
        property_values: dict[str, float],
        group_id: str,
        row_key: PredictionRowKey | None = None,
    ) -> None:
        if not property_values:
            return
        assert self.rows is not None
        assert self.values is not None
        assert self.groups is not None
        assert self.row_keys is not None
        assert self.rng is not None
        self._record_sampling_strategy("reservoir_v1")
        self.seen += 1
        if len(self.rows) < self.max_rows:
            self.rows.append(latent_row.detach().cpu().to(torch.float16).numpy())
            self.values.append(property_values)
            self.groups.append(group_id)
            self.row_keys.append(row_key)
            return
        replace_idx = int(self.rng.integers(0, self.seen))
        if replace_idx < self.max_rows:
            self.rows[replace_idx] = latent_row.detach().cpu().to(torch.float16).numpy()
            self.values[replace_idx] = property_values
            self.groups[replace_idx] = group_id
            self.row_keys[replace_idx] = row_key

    def add_batch(
        self,
        latents: torch.Tensor,
        property_values: dict[str, np.ndarray],
        group_ids: list[str],
        row_keys: list[PredictionRowKey | None] | None = None,
    ) -> None:
        if latents.shape[0] == 0 or not property_values:
            return
        assert self.rows is not None
        assert self.values is not None
        assert self.groups is not None
        assert self.rng is not None
        assert self.priorities is not None
        self._record_sampling_strategy("priority_v1")
        n_rows = latents.shape[0]
        if row_keys is None:
            row_keys = [None] * n_rows
        if len(row_keys) != n_rows:
            raise ValueError("row_keys must have one entry per latent row")
        priorities = self.rng.random(n_rows)
        keep = min(self.max_rows, n_rows)
        local = np.argpartition(priorities, keep - 1)[:keep] if keep else np.empty(0, int)
        candidates = [
            (
                float(priorities[index]),
                latents[index].detach().cpu().to(torch.float16).numpy(),
                {name: float(values[index]) for name, values in property_values.items()},
                group_ids[index],
                row_keys[index],
            )
            for index in local.tolist()
        ]
        assert self.row_keys is not None
        existing = list(
            zip(
                self.priorities,
                self.rows,
                self.values,
                self.groups,
                self.row_keys,
                strict=True,
            )
        )
        retained = sorted([*existing, *candidates], key=lambda item: item[0])[: self.max_rows]
        self.priorities = [item[0] for item in retained]
        self.rows = [item[1] for item in retained]
        self.values = [item[2] for item in retained]
        self.groups = [item[3] for item in retained]
        self.row_keys = [item[4] for item in retained]
        self.seen += n_rows

    def finalize(self, property_names: list[str]) -> PredictionSample | None:
        assert self.rows is not None
        assert self.values is not None
        assert self.groups is not None
        row_keys = getattr(self, "row_keys", None)
        if not self.rows:
            return None
        latents = np.asarray(self.rows, dtype=np.float32)
        property_values = {
            name: np.asarray(
                [row.get(name, np.nan) for row in self.values],
                dtype=np.float64,
            )
            for name in property_names
        }
        return PredictionSample(
            latents=latents,
            property_values=property_values,
            group_ids=np.asarray(self.groups, dtype=object),
            row_keys=tuple(row_keys) if row_keys is not None else (None,) * len(self.rows),
            sampling_identity={
                "strategy": getattr(self, "sampling_strategy", None) or "unknown",
                "max_rows": self.max_rows,
                "rng_seed": self.rng_seed,
                "seen": self.seen,
                "retained_rows": len(self.rows),
            },
        )

    def _record_sampling_strategy(self, strategy: str) -> None:
        current = getattr(self, "sampling_strategy", None)
        if current is None:
            self.sampling_strategy = strategy
        elif current != strategy:
            self.sampling_strategy = "mixed"


class RMSFFeaturePoolingAccumulator:
    """Pool RMSF-prediction rows to residue and optional structure units."""

    def __init__(
        self,
        n_latents: int,
        rmsf_by_sequence: Mapping[str, np.ndarray],
        prediction_sample_size: int,
        retain_structure_summaries: bool = True,
        selected_sample_index: int | None = None,
    ) -> None:
        self.n_latents = n_latents
        self.rmsf_by_sequence = rmsf_by_sequence
        self.reservoir = PredictionReservoir(max_rows=prediction_sample_size)
        self.metric_accumulator = FeaturePredictionAccumulator.create(n_latents)
        self.retain_structure_summaries = retain_structure_summaries
        self.selected_sample_index = selected_sample_index
        self.structure_feature_summaries: dict[tuple[str, int], np.ndarray] = {}
        self.current_sequence: str | None = None
        self.completed_sequences: set[str] = set()
        self.residue_sums: torch.Tensor | None = None
        self.residue_counts: torch.Tensor | None = None
        self.structure_key: tuple[str, int] | None = None
        self.structure_sum = torch.zeros(n_latents)
        self.structure_count = 0

    def accumulate(
        self,
        latents: torch.Tensor,
        record: ActivationRecord,
        start: int,
        end: int,
    ) -> None:
        sequence_ids = record.sequence_ids[start:end]
        if not sequence_ids:
            return
        sequence_id = sequence_ids[0]
        if any(candidate != sequence_id for candidate in sequence_ids):
            raise ValueError("RMSF pooling batches must contain one sequence")
        sample_indices = record.sample_indices
        if sample_indices is None:
            raise ValueError("RMSF pooling requires sample indices")
        batch_samples = sample_indices[start:end].detach().cpu()
        if self.selected_sample_index is not None and not bool(
            torch.any(batch_samples == self.selected_sample_index)
        ):
            return
        sample_index = int(batch_samples[0])
        if self.retain_structure_summaries and not bool(torch.all(batch_samples == sample_index)):
            raise ValueError("RMSF structure-summary batches must contain one structure")
        self._ensure_sequence(sequence_id)
        if self.retain_structure_summaries:
            key = (sequence_id, sample_index)
            if self.structure_key is None:
                self.structure_key = key
            elif self.structure_key != key:
                raise ValueError("finish_record must be called between RMSF structures")

        values = latents.detach().cpu().float()
        residues = record.residue_indices[start:end].detach().cpu().long()
        assert self.residue_sums is not None
        assert self.residue_counts is not None
        sample_mask = torch.ones_like(batch_samples, dtype=torch.bool)
        if self.selected_sample_index is not None:
            sample_mask = batch_samples == self.selected_sample_index
        valid = sample_mask & (residues >= 0) & (residues < self.residue_sums.shape[0])
        if bool(valid.any()):
            valid_residues = residues[valid]
            valid_values = values[valid]
            self.residue_sums.index_add_(0, valid_residues, valid_values)
            self.residue_counts.index_add_(
                0,
                valid_residues,
                torch.ones(valid_residues.shape[0]),
            )
            if self.retain_structure_summaries:
                self.structure_sum += valid_values.sum(dim=0)
                self.structure_count += int(valid_values.shape[0])

    def finish_record(self) -> None:
        if self.structure_key is None:
            return
        if self.structure_count <= 0:
            if self.selected_sample_index is not None:
                self.structure_key = None
                return
            raise ValueError(f"RMSF structure has no biological residue rows: {self.structure_key}")
        if self.retain_structure_summaries:
            self.structure_feature_summaries[self.structure_key] = (
                self.structure_sum / self.structure_count
            ).numpy()
        self.structure_key = None
        self.structure_sum = torch.zeros(self.n_latents)
        self.structure_count = 0

    def finalize(
        self,
    ) -> tuple[PredictionSample | None, dict[tuple[str, int], np.ndarray]]:
        self.finish_record()
        self._finish_sequence()
        return self.reservoir.finalize(["rmsf"]), dict(self.structure_feature_summaries)

    def _ensure_sequence(self, sequence_id: str) -> None:
        if self.current_sequence == sequence_id:
            return
        self._finish_sequence()
        if sequence_id in self.completed_sequences:
            raise ValueError("RMSF structure entries must be contiguous by sequence")
        rmsf = self.rmsf_by_sequence.get(sequence_id)
        if rmsf is None:
            raise ValueError(f"missing RMSF labels for sequence: {sequence_id}")
        self.current_sequence = sequence_id
        self.residue_sums = torch.zeros((len(rmsf), self.n_latents))
        self.residue_counts = torch.zeros(len(rmsf))

    def _finish_sequence(self) -> None:
        if self.current_sequence is None:
            return
        assert self.residue_sums is not None
        assert self.residue_counts is not None
        require_exact_rmsf_coordinate_counts(self.residue_counts, self.current_sequence)
        rmsf = self.rmsf_by_sequence[self.current_sequence]
        for residue_index in torch.nonzero(self.residue_counts > 0).flatten().tolist():
            feature_values = self.residue_sums[residue_index] / self.residue_counts[residue_index]
            target = float(rmsf[residue_index])
            self.metric_accumulator.count += 1
            self.metric_accumulator.sum_y += target
            self.metric_accumulator.sum_y2 += target * target
            self.metric_accumulator.sum_x += feature_values
            self.metric_accumulator.sum_x2 += feature_values * feature_values
            self.metric_accumulator.sum_xy += feature_values * target
            self.reservoir.add(
                feature_values,
                {"rmsf": target},
                group_id=self.current_sequence,
                row_key=PredictionRowKey(
                    sequence_id=self.current_sequence,
                    residue_index=residue_index,
                    sample_index=self.selected_sample_index,
                    source="rmsf_residue_mean",
                ),
            )
        self.completed_sequences.add(self.current_sequence)
        self.current_sequence = None
        self.residue_sums = None
        self.residue_counts = None


def require_exact_rmsf_coordinate_counts(counts: torch.Tensor, sequence_id: str) -> None:
    invalid = torch.nonzero(counts != RMSF_COORDINATE_OBSERVATIONS_PER_RESIDUE).flatten()
    if invalid.numel() == 0:
        return
    residue_index = int(invalid[0])
    observed = int(counts[residue_index])
    raise ValueError(
        "RMSF pooling requires exactly "
        f"{RMSF_COORDINATE_OBSERVATIONS_PER_RESIDUE} coordinate observations for "
        f"{sequence_id} residue {residue_index}; got {observed}"
    )


def bioemu_rmsf_probe_provenance() -> dict[str, Any]:
    return {
        "label_source": "aligned 32-structure BioEmu ensemble",
        "activation_input_source": "BioEmu",
        "activation_sample_index": 0,
        "pooling": "mean over ten BioEmu score calls per residue",
    }


def build_rmsf_source_input_provenance(
    config: ProjectConfig,
    sequences: Mapping[str, str],
    *,
    representation_route: str,
    prompt_source: str,
    structure_index: str | Path | None = None,
    bioemu_embedding_cache: str | Path | None = None,
) -> dict[str, object]:
    """Bind an RMSF report to the exact representation inputs used by replay."""

    evaluation_fasta = config.analysis.evaluation_input_fasta
    if evaluation_fasta is None:
        raise ValueError("RMSF input provenance requires analysis.evaluation_input_fasta")
    if representation_route == "bioemu":
        model_family = config.trace.model_name
        prompt_mode = "bioemu_denoising"
        representation_config = {
            "model_family": model_family,
            "target_module": config.trace.target_module,
        }
        cache_dir = Path(
            bioemu_embedding_cache
            or config.analysis.evaluation_cache_embeds_dir
            or config.training.cache_embeds_dir
        )
        cache_manifest = _rmsf_file_identity(cache_dir / "embedding_manifest.json")
        structure_identity = None
    else:
        raise ValueError(
            f"representation_route must be 'bioemu' in this release, not "
            f"{representation_route!r}: the ESM3 route is not shipped"
        )
    canonical_config = json.dumps(
        representation_config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "schema_version": 1,
        "representation_route": representation_route,
        "model_family": model_family,
        "prompt_mode": prompt_mode,
        "prompt_source": prompt_source,
        "representation_config_sha256": hashlib.sha256(canonical_config).hexdigest(),
        "sequence_sha256": {
            str(sequence_id): hashlib.sha256(str(sequence).encode()).hexdigest()
            for sequence_id, sequence in sorted(sequences.items())
        },
        "evaluation_fasta": _rmsf_file_identity(Path(evaluation_fasta)),
        "cache_manifest": cache_manifest,
        "structure_index": structure_identity,
    }


def _rmsf_file_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"RMSF input provenance dependency is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def write_not_applicable_rmsf_probe(
    output_dir: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    report = {
        "status": "not_applicable",
        "reason": "RMSF probes require one activation structure per sequence",
        "provenance": dict(provenance),
    }
    write_rmsf_json_atomic(output_dir / "summary.json", report)
    return report


@dataclass
class FeatureSummaryAccumulator:
    """Streaming feature statistics inspired by InterPLM per-feature summaries."""

    sums: torch.Tensor
    counts: torch.Tensor
    active_counts: torch.Tensor
    max_values: torch.Tensor
    active_sequences: list[set[str]]
    active_samples: list[set[tuple[str, int]]]
    active_residue_counts_by_sequence: list[dict[str, int]]
    token_counts_by_sequence: dict[str, int]
    total_tokens: int = 0
    dense_tracking: bool = False
    dense_active_counts_by_sequence: dict[str, torch.Tensor] | None = None
    dense_active_samples: dict[tuple[str, int], torch.Tensor] | None = None

    @classmethod
    def create(
        cls,
        n_latents: int,
        dense_tracking: bool = False,
    ) -> FeatureSummaryAccumulator:
        return cls(
            sums=torch.zeros(n_latents),
            counts=torch.zeros(n_latents),
            active_counts=torch.zeros(n_latents),
            max_values=torch.full((n_latents,), -float("inf")),
            active_sequences=[set() for _ in range(n_latents)],
            active_samples=[set() for _ in range(n_latents)],
            active_residue_counts_by_sequence=[defaultdict(int) for _ in range(n_latents)],
            token_counts_by_sequence=defaultdict(int),
            dense_tracking=dense_tracking,
            dense_active_counts_by_sequence={} if dense_tracking else None,
            dense_active_samples={} if dense_tracking else None,
        )


def analyze_sae_structure_features(
    config: ProjectConfig,
    checkpoint_path: str | Path,
    activation_store: str | Path | None = None,
    output_dir: str | Path | None = None,
    top_k_features: int = 25,
    top_k_examples: int = 200,
    logo_window: int = 5,
    high_quantile: float = 0.75,
    low_quantile: float = 0.25,
) -> StructurePropertyResult:
    """Analyze trained SAE features against final-structure properties.

    The analysis computes per-sequence residue RMSF from saved final BioEmu
    structures, compares feature activations on high-RMSF versus low-RMSF
    residue tokens, and writes a volcano plot plus sequence-logo style plots
    for the strongest RMSF-associated features.
    """

    if top_k_features <= 0:
        raise ValueError("top_k_features must be positive")
    if top_k_examples <= 0:
        raise ValueError("top_k_examples must be positive")
    if logo_window < 0:
        raise ValueError("logo_window must be non-negative")
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError("require 0 <= low_quantile < high_quantile <= 1")

    if activation_store is not None:
        config = replace(
            config,
            training=replace(config.training, activation_store=str(activation_store)),
        )
    store = ActivationStore(config.training.activation_store)
    output_dir = Path(output_dir or Path(config.training.output_dir) / "structure_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    structure_entries = load_structure_index(store)
    rmsf_by_sequence, structure_properties = compute_structure_analysis_properties(
        store.root,
        structure_entries,
        sampling_scope=StructureSamplingScope.BIOEMU_32,
    )
    radius_by_sample = {
        key: values["radius_of_gyration"] for key, values in structure_properties.items()
    }
    property_report_path = output_dir / "structure_property_report.json"

    trainer = SAETrainer(config)
    trainer.store = store
    trainer.load_model_checkpoint(checkpoint_path)
    sequences = sequence_metadata(store.manifest)
    properties = build_token_properties(
        sequences=sequences,
        rmsf_by_sequence=rmsf_by_sequence,
        structure_properties=structure_properties,
        high_quantile=high_quantile,
        low_quantile=low_quantile,
    )
    continuous_properties = build_continuous_token_properties(
        rmsf_by_sequence=rmsf_by_sequence,
        structure_properties=structure_properties,
    )
    volcano_dir = output_dir / "volcano"
    volcano_dir.mkdir(parents=True, exist_ok=True)
    volcano_reports = {}
    volcano_plot_specs = []
    selected_feature_set: set[int] = set()
    interpro_annotations_path = config.analysis.interpro_annotations_path
    has_concept_annotations = (
        interpro_annotations_path is not None and Path(interpro_annotations_path).exists()
    )
    concept_accumulator = (
        ResidueFeatureActivationAccumulator(sequences) if has_concept_annotations else None
    )
    store_pass = collect_store_feature_analysis_pass(
        trainer=trainer,
        store=store,
        token_properties=properties,
        continuous_properties=continuous_properties,
        top_k_examples=top_k_examples,
        residue_feature_accumulator=concept_accumulator,
    )
    property_importance_dir = output_dir / "property_importance"
    reported_property_importance = store_pass.property_importance
    property_importance_reports = write_property_importance_artifacts(
        property_importance=reported_property_importance,
        prediction_sample=store_pass.prediction_sample,
        timestep_profiles=store_pass.timestep_profiles,
        output_dir=property_importance_dir,
        top_k_features=top_k_features,
    )
    property_highlight_features = property_highlights_from_importance(reported_property_importance)
    source_input_provenance = (
        build_rmsf_source_input_provenance(
            config,
            sequences,
            representation_route="bioemu",
            prompt_source="bioemu_sample0",
        )
        if config.analysis.evaluation_input_fasta is not None
        else None
    )
    rmsf_probe_report, rmsf_probe_result = run_analysis_rmsf_probe(
        store_pass.prediction_sample,
        output_dir / "rmsf_probe",
        feature_counts=DEFAULT_RMSF_FEATURE_COUNTS,
        source_input_provenance=source_input_provenance,
    )
    rmsf_probe_report["structure_views"] = write_analysis_rmsf_structure_views(
        rmsf_probe_result,
        trainer,
        config,
        structure_entries,
        RMSFReference(store.root, tuple(structure_entries), rmsf_by_sequence),
        output_dir / "rmsf_probe" / "structure_views",
        store=store,
    )
    write_rmsf_json_atomic(
        output_dir / "rmsf_probe" / "summary.json",
        rmsf_probe_report,
    )
    selected_features_by_property: dict[str, list[int]] = {}
    logo_feature_set: set[int] = set()
    for token_property in properties:
        feature_stats = store_pass.stats_by_property[token_property.name]
        volcano_rows = compute_volcano_rows(feature_stats)
        volcano_plot_path = volcano_dir / f"{token_property.name}.png"
        volcano_csv_path = volcano_plot_path.with_suffix(".csv")
        volcano_json_path = volcano_plot_path.with_suffix(".json")
        if volcano_rows:
            write_dict_rows_csv(volcano_csv_path, volcano_rows)
            with volcano_json_path.open("w") as handle:
                json.dump(volcano_rows, handle, indent=2)
        volcano_title = f"SAE Feature Association: {token_property.description}"
        volcano_x_label = (
            f"Mean SAE feature activation: {token_property.high_label} - {token_property.low_label}"
        )
        write_volcano_plot(
            rows=volcano_rows,
            path=volcano_plot_path,
            title=volcano_title,
            x_label=volcano_x_label,
            highlighted_features=highlighted_features_for_token_property(
                token_property.name,
                property_highlight_features,
                selected_feature_set,
            ),
        )
        volcano_plot_specs.append(
            (
                token_property,
                volcano_rows,
                volcano_title,
                volcano_x_label,
                highlighted_features_for_token_property(
                    token_property.name,
                    property_highlight_features,
                    selected_feature_set,
                ),
            )
        )
        volcano_reports[token_property.name] = {
            "description": token_property.description,
            "high_label": token_property.high_label,
            "low_label": token_property.low_label,
            "plot_path": str(volcano_plot_path),
            "csv_path": str(volcano_csv_path),
            "json_path": str(volcano_json_path),
            "features": volcano_rows,
        }
    combined_volcano_path = volcano_dir / "all_volcano_plots.png"
    write_volcano_grid_plot(volcano_plot_specs, combined_volcano_path)

    feature_summary_dir = output_dir / "feature_summaries"
    feature_summary_report = write_feature_summary_artifacts(
        feature_summary=store_pass.feature_summary,
        timestep_profiles=store_pass.timestep_profiles,
        output_dir=feature_summary_dir,
    )
    concept_metrics_dir = output_dir / "concept_metrics"
    concept_metric_reports = write_concept_metric_tables(
        concept_metrics=store_pass.concept_metrics,
        output_dir=concept_metrics_dir,
    )
    interpro_concept_report = analyze_interpro_concepts(
        accumulator=concept_accumulator,
        annotations_path=interpro_annotations_path,
        sequences=sequences,
        output_dir=output_dir / "interpro_concepts",
        min_positive_residues=config.analysis.concept_min_positive_residues,
        min_positive_domains=config.analysis.concept_min_positive_domains,
        min_positive_sequences=config.analysis.concept_min_positive_sequences,
        top_k=top_k_features,
    )
    selected_interpro_concepts: list[dict[str, Any]] = []
    interpro_selected_features_by_property = select_concept_features_from_report(
        interpro_concept_report,
        concept_prefix="interpro",
        max_concepts=10,
    )
    selected_features_by_property = merge_property_features_for_localization(
        property_highlight_features,
        interpro_selected_features_by_property,
    )
    if interpro_selected_features_by_property:
        selected_interpro_concepts = selected_concept_rows_from_report(
            interpro_concept_report,
            max_concepts=10,
        )
        for feature_indices in interpro_selected_features_by_property.values():
            logo_feature_set.update(feature_indices)
    feature_property_labels = feature_property_labels_from_selection(selected_features_by_property)
    selected_feature_set.update(
        feature_idx
        for feature_indices in interpro_selected_features_by_property.values()
        for feature_idx in feature_indices
    )

    interpro_property_timestep_report = write_selected_feature_property_timestep_reports(
        timestep_profiles=store_pass.timestep_profiles,
        selected_features_by_property=interpro_selected_features_by_property,
        f1_per_domain_by_property_feature=concept_feature_f1_scores(
            selected_interpro_concepts,
            concept_prefix="interpro",
        ),
        output_dir=property_importance_dir / "interpro_timestep_profiles",
    )
    property_importance_reports.extend(interpro_property_timestep_report)
    selected_features = ordered_selected_features(selected_features_by_property)
    timestep_selected_features = selected_timestep_features_by_property(
        property_highlight_features,
        selected_features_by_property,
    )
    selected_timestep_reports = write_selected_feature_timestep_profile_plots(
        timestep_profiles=store_pass.timestep_profiles,
        selected_features_by_property=timestep_selected_features,
        output_dir=output_dir / "selected_feature_timestep_profiles",
    )
    logo_dir = output_dir / "logos"
    logo_dir.mkdir(parents=True, exist_ok=True)
    examples = {
        feature_idx: store_pass.examples_by_feature.get(feature_idx, [])
        for feature_idx in selected_features
    }
    logo_examples = {
        feature_idx: store_pass.examples_by_feature.get(feature_idx, [])
        for feature_idx in sorted(logo_feature_set)
    }
    logo_reports = write_logo_plots(
        examples=logo_examples,
        sequences=sequences,
        output_dir=logo_dir,
        window=logo_window,
    )
    logo_reports.extend(
        write_property_logo_grids(
            examples=examples,
            selected_features_by_property=selected_features_by_property,
            property_importance=reported_property_importance,
            sequences=sequences,
            output_dir=logo_dir / "property_grids",
            window=logo_window,
        )
    )
    residue_plot_dir = output_dir / "residue_activation_plots"
    residue_plot_reports = write_residue_activation_plots(
        trainer=trainer,
        config=config,
        structure_entries=None,
        store=store,
        examples=examples,
        sequences=sequences,
        output_dir=residue_plot_dir,
        max_examples_per_feature=3,
    )
    localization_reports = write_feature_localization_artifacts(
        trainer=trainer,
        config=config,
        store=store,
        examples=examples,
        feature_summary=store_pass.feature_summary,
        structure_entries=structure_entries,
        structure_root=store.root,
        sequences=sequences,
        rmsf_by_sequence=rmsf_by_sequence,
        feature_property_labels=feature_property_labels,
        selected_features_by_property=selected_features_by_property,
        property_importance=reported_property_importance,
        output_dir=output_dir / "feature_localization",
        max_examples_per_feature=4,
        max_total_artifacts=max(50, 4 * len(selected_features)),
        categorize_bioemu_features=True,
    )

    report = {
        "summary": {
            "activation_store": str(store.root),
            "checkpoint": str(checkpoint_path),
            **summarize_structure_sample_counts(structure_entries),
            "num_sequences_with_rmsf": len(rmsf_by_sequence),
            "num_properties_tested": len(volcano_reports),
            "high_rmsf_quantile": high_quantile,
            "low_rmsf_quantile": low_quantile,
            "top_k_features": top_k_features,
            "localization_source": "continuous_properties_and_interpro_concepts",
            "max_interpro_concepts_for_visualization": 10,
            "top_k_examples": top_k_examples,
            "logo_window": logo_window,
            "combined_volcano_plot_path": str(combined_volcano_path),
            "full_activation_passes": 1,
            "targeted_profile_replays": (
                "store-backed profile lookups for selected examples only; no activation tensors stored"
            ),
            "activation_storage": "no activation tensors written by analysis",
        },
        "rmsf": {sequence_id: values.tolist() for sequence_id, values in rmsf_by_sequence.items()},
        "radius_of_gyration": {
            f"{sequence_id}|{sample_index}": value
            for (sequence_id, sample_index), value in radius_by_sample.items()
        },
        "structure_sample_properties": {
            f"{sequence_id}|{sample_index}": values
            for (sequence_id, sample_index), values in structure_properties.items()
        },
        "volcano": volcano_reports,
        "logos": logo_reports,
        "feature_summary": feature_summary_report,
        "concept_metrics": concept_metric_reports,
        "interpro_concepts": interpro_concept_report,
        "selected_interpro_concepts": selected_interpro_concepts,
        "property_importance": property_importance_reports,
        "rmsf_probe": rmsf_probe_report,
        "selected_feature_timestep_profiles": selected_timestep_reports,
        "residue_activation_plots": residue_plot_reports,
        "feature_localization": localization_reports,
    }
    with property_report_path.open("w") as handle:
        json.dump(report, handle, indent=2)

    return StructurePropertyResult(
        output_dir=output_dir,
        property_report_path=property_report_path,
        volcano_dir=volcano_dir,
        logo_dir=logo_dir,
        num_sequences_with_rmsf=len(rmsf_by_sequence),
        num_properties_tested=len(volcano_reports),
    )


def analyze_sae_runtime_structure_features(
    config: ProjectConfig,
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    top_k_features: int = 25,
    top_k_examples: int = 200,
    logo_window: int = 5,
    high_quantile: float = 0.75,
    low_quantile: float = 0.25,
    *,
    precomputed_pass: RuntimeFeatureAnalysisPass | None = None,
    precomputed_concept_accumulator: ResidueFeatureActivationAccumulator | None = None,
    precomputed_structure_entries: list[StructureIndexEntry] | None = None,
    precomputed_rmsf_by_sequence: dict[str, np.ndarray] | None = None,
    precomputed_structure_properties: dict[tuple[str, int], dict[str, float]] | None = None,
    runtime_sampler_factory: Any | None = None,
) -> StructurePropertyResult:
    """Analyze runtime-sampled BioEmu SAE features without stored activations."""

    if config.training.activation_source != "bioemu":
        raise ValueError("runtime structure analysis currently supports BioEmu configs")
    runtime_root = Path(config.training.runtime_structure_dir or config.training.activation_store)
    structure_entries = precomputed_structure_entries
    if structure_entries is None:
        structure_entries = load_structure_index_path(
            runtime_root / "final_structures" / "index.jsonl"
        )
    rmsf_by_sequence = precomputed_rmsf_by_sequence
    structure_properties = precomputed_structure_properties
    if rmsf_by_sequence is None or structure_properties is None:
        rmsf_by_sequence, structure_properties = compute_structure_analysis_properties(
            runtime_root,
            structure_entries,
            sampling_scope=StructureSamplingScope.BIOEMU_32,
        )
    radius_by_sample = {
        key: values["radius_of_gyration"] for key, values in structure_properties.items()
    }
    sequences = {entry.sequence_id: entry.sequence for entry in structure_entries}
    output_dir = Path(output_dir or Path(config.training.output_dir) / "structure_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = SAETrainer(config)
    trainer.load_model_checkpoint(checkpoint_path)
    properties = build_token_properties(
        sequences=sequences,
        rmsf_by_sequence=rmsf_by_sequence,
        structure_properties=structure_properties,
        high_quantile=high_quantile,
        low_quantile=low_quantile,
    )
    continuous_properties = build_continuous_token_properties(
        rmsf_by_sequence=rmsf_by_sequence,
        structure_properties=structure_properties,
    )
    volcano_dir = output_dir / "volcano"
    volcano_dir.mkdir(parents=True, exist_ok=True)
    volcano_reports = {}
    volcano_plot_specs = []
    selected_feature_set: set[int] = set()
    interpro_annotations_path = config.analysis.interpro_annotations_path
    has_concept_annotations = (
        interpro_annotations_path is not None and Path(interpro_annotations_path).exists()
    )
    concept_accumulator = precomputed_concept_accumulator
    if concept_accumulator is None and has_concept_annotations:
        concept_accumulator = ResidueFeatureActivationAccumulator(sequences)
    runtime_pass = precomputed_pass
    if runtime_pass is None:
        runtime_pass = collect_runtime_feature_analysis_pass(
            trainer=trainer,
            config=config,
            structure_entries=structure_entries,
            token_properties=properties,
            continuous_properties=continuous_properties,
            top_k_examples=top_k_examples,
            residue_feature_accumulator=concept_accumulator,
        )
    property_importance_dir = output_dir / "property_importance"
    reported_property_importance = runtime_pass.property_importance
    property_importance_reports = write_property_importance_artifacts(
        property_importance=reported_property_importance,
        prediction_sample=runtime_pass.prediction_sample,
        timestep_profiles=runtime_pass.timestep_profiles,
        output_dir=property_importance_dir,
        top_k_features=top_k_features,
        structure_feature_summaries=runtime_pass.structure_feature_summaries,
        structure_properties=structure_properties,
    )
    property_highlight_features = property_highlights_from_importance(reported_property_importance)
    rmsf_probe_report, rmsf_probe_result = run_analysis_rmsf_probe(
        runtime_pass.prediction_sample,
        output_dir / "rmsf_probe",
        feature_counts=DEFAULT_RMSF_FEATURE_COUNTS,
        source_input_provenance=build_rmsf_source_input_provenance(
            config,
            sequences,
            representation_route="bioemu",
            prompt_source="bioemu_sample0",
        ),
    )
    rmsf_probe_report["provenance"] = bioemu_rmsf_probe_provenance()
    rmsf_probe_report["structure_views"] = write_analysis_rmsf_structure_views(
        rmsf_probe_result,
        trainer,
        config,
        structure_entries,
        RMSFReference(runtime_root, tuple(structure_entries), rmsf_by_sequence),
        output_dir / "rmsf_probe" / "structure_views",
        runtime_sampler_factory=runtime_sampler_factory,
    )
    write_rmsf_json_atomic(
        output_dir / "rmsf_probe" / "summary.json",
        rmsf_probe_report,
    )
    selected_features_by_property: dict[str, list[int]] = {}
    logo_feature_set: set[int] = set()
    for token_property in properties:
        feature_stats = runtime_pass.stats_by_property[token_property.name]
        volcano_rows = compute_volcano_rows(feature_stats)
        volcano_plot_path = volcano_dir / f"{token_property.name}.png"
        volcano_csv_path = volcano_plot_path.with_suffix(".csv")
        volcano_json_path = volcano_plot_path.with_suffix(".json")
        if volcano_rows:
            write_dict_rows_csv(volcano_csv_path, volcano_rows)
            with volcano_json_path.open("w") as handle:
                json.dump(volcano_rows, handle, indent=2)
        volcano_title = f"SAE Feature Association: {token_property.description}"
        volcano_x_label = (
            f"Mean SAE feature activation: {token_property.high_label} - {token_property.low_label}"
        )
        write_volcano_plot(
            rows=volcano_rows,
            path=volcano_plot_path,
            title=volcano_title,
            x_label=volcano_x_label,
            highlighted_features=highlighted_features_for_token_property(
                token_property.name,
                property_highlight_features,
                selected_feature_set,
            ),
        )
        volcano_plot_specs.append(
            (
                token_property,
                volcano_rows,
                volcano_title,
                volcano_x_label,
                highlighted_features_for_token_property(
                    token_property.name,
                    property_highlight_features,
                    selected_feature_set,
                ),
            )
        )
        volcano_reports[token_property.name] = {
            "description": token_property.description,
            "high_label": token_property.high_label,
            "low_label": token_property.low_label,
            "plot_path": str(volcano_plot_path),
            "csv_path": str(volcano_csv_path),
            "json_path": str(volcano_json_path),
            "features": volcano_rows,
        }
    combined_volcano_path = volcano_dir / "all_volcano_plots.png"
    write_volcano_grid_plot(volcano_plot_specs, combined_volcano_path)
    feature_summary_dir = output_dir / "feature_summaries"
    feature_summary_report = write_feature_summary_artifacts(
        feature_summary=runtime_pass.feature_summary,
        timestep_profiles=runtime_pass.timestep_profiles,
        output_dir=feature_summary_dir,
    )
    concept_metrics_dir = output_dir / "concept_metrics"
    concept_metric_reports = write_concept_metric_tables(
        concept_metrics=runtime_pass.concept_metrics,
        output_dir=concept_metrics_dir,
    )
    interpro_concept_report = analyze_interpro_concepts(
        accumulator=concept_accumulator,
        annotations_path=interpro_annotations_path,
        sequences=sequences,
        output_dir=output_dir / "interpro_concepts",
        min_positive_residues=config.analysis.concept_min_positive_residues,
        min_positive_domains=config.analysis.concept_min_positive_domains,
        min_positive_sequences=config.analysis.concept_min_positive_sequences,
        top_k=top_k_features,
    )
    selected_interpro_concepts: list[dict[str, Any]] = []
    interpro_selected_features_by_property = select_concept_features_from_report(
        interpro_concept_report,
        concept_prefix="interpro",
        max_concepts=10,
    )
    selected_features_by_property = merge_property_features_for_localization(
        property_highlight_features,
        interpro_selected_features_by_property,
    )
    if interpro_selected_features_by_property:
        selected_interpro_concepts = selected_concept_rows_from_report(
            interpro_concept_report,
            max_concepts=10,
        )
        for feature_indices in interpro_selected_features_by_property.values():
            logo_feature_set.update(feature_indices)
    feature_property_labels = feature_property_labels_from_selection(selected_features_by_property)
    selected_feature_set.update(
        feature_idx
        for feature_indices in interpro_selected_features_by_property.values()
        for feature_idx in feature_indices
    )

    interpro_property_timestep_report = write_selected_feature_property_timestep_reports(
        timestep_profiles=runtime_pass.timestep_profiles,
        selected_features_by_property=interpro_selected_features_by_property,
        f1_per_domain_by_property_feature=concept_feature_f1_scores(
            selected_interpro_concepts,
            concept_prefix="interpro",
        ),
        output_dir=property_importance_dir / "interpro_timestep_profiles",
    )
    property_importance_reports.extend(interpro_property_timestep_report)
    selected_features = ordered_selected_features(selected_features_by_property)
    timestep_selected_features = selected_timestep_features_by_property(
        property_highlight_features,
        selected_features_by_property,
    )
    selected_timestep_reports = write_selected_feature_timestep_profile_plots(
        timestep_profiles=runtime_pass.timestep_profiles,
        selected_features_by_property=timestep_selected_features,
        output_dir=output_dir / "selected_feature_timestep_profiles",
    )
    examples = {}
    for feature_idx in selected_features:
        candidates = runtime_pass.examples_by_feature.get(feature_idx, [])
        if not candidates and concept_accumulator is not None:
            candidates = feature_examples_from_residue_accumulator(
                concept_accumulator,
                feature_idx,
                top_k_examples,
            )
        examples[feature_idx] = candidates
    logo_examples = {
        feature_idx: examples.get(
            feature_idx,
            runtime_pass.examples_by_feature.get(feature_idx, []),
        )
        for feature_idx in sorted(logo_feature_set)
    }
    logo_dir = output_dir / "logos"
    logo_dir.mkdir(parents=True, exist_ok=True)
    logo_reports = write_logo_plots(
        examples=logo_examples,
        sequences=sequences,
        output_dir=logo_dir,
        window=logo_window,
    )
    logo_reports.extend(
        write_property_logo_grids(
            examples=examples,
            selected_features_by_property=selected_features_by_property,
            property_importance=reported_property_importance,
            sequences=sequences,
            output_dir=logo_dir / "property_grids",
            window=logo_window,
        )
    )
    localization_reports = write_feature_localization_artifacts(
        trainer=trainer,
        config=config,
        store=None,
        examples=examples,
        feature_summary=runtime_pass.feature_summary,
        structure_entries=structure_entries,
        structure_root=runtime_root,
        sequences=sequences,
        rmsf_by_sequence=rmsf_by_sequence,
        feature_property_labels=feature_property_labels,
        selected_features_by_property=selected_features_by_property,
        property_importance=reported_property_importance,
        output_dir=output_dir / "feature_localization",
        max_examples_per_feature=4,
        max_total_artifacts=max(50, 4 * len(selected_features)),
        categorize_bioemu_features=True,
        runtime_sampler_factory=runtime_sampler_factory,
    )
    residue_plot_reports = [
        {
            "skipped": True,
            "reason": (
                "runtime residue activation profiles require a second BioEmu "
                "activation replay; skipped so runtime analysis uses exactly "
                "one activation pass and writes no activation tensors"
            ),
        }
    ]
    property_report_path = output_dir / "structure_property_report.json"
    with property_report_path.open("w") as handle:
        json.dump(
            {
                "summary": {
                    "runtime_structure_dir": str(runtime_root),
                    "checkpoint": str(checkpoint_path),
                    **summarize_structure_sample_counts(structure_entries),
                    "num_sequences_with_rmsf": len(rmsf_by_sequence),
                    "num_properties_tested": len(volcano_reports),
                    "high_rmsf_quantile": high_quantile,
                    "low_rmsf_quantile": low_quantile,
                    "top_k_features": top_k_features,
                    "localization_source": "continuous_properties_and_interpro_concepts",
                    "max_interpro_concepts_for_visualization": 10,
                    "top_k_examples": top_k_examples,
                    "logo_window": logo_window,
                    "combined_volcano_plot_path": str(combined_volcano_path),
                    "full_activation_passes": 1,
                    "activation_pass_mode": (
                        "shared_sequence_major_multi_sae"
                        if precomputed_pass is not None
                        else "single_sae_deterministic_replay"
                    ),
                    "property_activation_aggregation": {
                        "rmsf": "mean SAE activation per sequence residue",
                        "sample_structure_properties": (
                            "mean SAE activation per generated structure"
                        ),
                        "interpro": (
                            "max SAE activation per sequence residue, normalized per feature"
                        ),
                    },
                    "targeted_profile_replays": (
                        "up to four deterministic BioEmu replays per selected feature for "
                        "full-residue example profiles"
                    ),
                    "activation_storage": (
                        "temporary per-protein summary shards removed after successful analysis; "
                        "no BioEmu activation tensors retained"
                        if precomputed_pass is not None
                        else "no activation tensors written by analysis"
                    ),
                },
                "rmsf": {
                    sequence_id: values.tolist() for sequence_id, values in rmsf_by_sequence.items()
                },
                "radius_of_gyration": {
                    f"{sequence_id}|{sample_index}": value
                    for (sequence_id, sample_index), value in radius_by_sample.items()
                },
                "structure_sample_properties": {
                    f"{sequence_id}|{sample_index}": values
                    for (sequence_id, sample_index), values in structure_properties.items()
                },
                "volcano": volcano_reports,
                "logos": logo_reports,
                "feature_summary": feature_summary_report,
                "concept_metrics": concept_metric_reports,
                "interpro_concepts": interpro_concept_report,
                "selected_interpro_concepts": selected_interpro_concepts,
                "property_importance": property_importance_reports,
                "rmsf_probe": rmsf_probe_report,
                "selected_feature_timestep_profiles": selected_timestep_reports,
                "residue_activation_plots": residue_plot_reports,
                "feature_localization": localization_reports,
            },
            handle,
            indent=2,
        )
    return StructurePropertyResult(
        output_dir=output_dir,
        property_report_path=property_report_path,
        volcano_dir=volcano_dir,
        logo_dir=logo_dir,
        num_sequences_with_rmsf=len(rmsf_by_sequence),
        num_properties_tested=len(volcano_reports),
    )


def feature_examples_from_residue_accumulator(
    accumulator: ResidueFeatureActivationAccumulator,
    feature_idx: int,
    top_k: int,
) -> list[FeatureExampleCandidate]:
    residue_indices, values = accumulator.feature_residue_values(feature_idx)
    if residue_indices.size == 0 or top_k <= 0:
        return []
    retained = min(top_k, residue_indices.size)
    selected = np.argpartition(values, -retained)[-retained:]
    selected = selected[np.argsort(values[selected])[::-1]]
    sequence_ranges = [
        (
            offset,
            offset + len(accumulator.sequences[sequence_id]),
            sequence_id,
        )
        for sequence_id, offset in accumulator.offset_by_sequence.items()
    ]
    examples = []
    for index in selected:
        global_residue = int(residue_indices[index])
        for start, end, sequence_id in sequence_ranges:
            if start <= global_residue < end:
                examples.append(
                    FeatureExampleCandidate(
                        activation=float(values[index]),
                        sequence_id=sequence_id,
                        residue_index=global_residue - start,
                        timestep=(
                            float(
                                accumulator.dense_argmax_timestep_by_feature[
                                    feature_idx, global_residue
                                ]
                            )
                            / 255.0
                            if accumulator.dense_argmax_timestep_by_feature is not None
                            else 0.5
                        ),
                        sample_index=(
                            int(
                                accumulator.dense_argmax_sample_by_feature[
                                    feature_idx, global_residue
                                ]
                            )
                            if accumulator.dense_argmax_sample_by_feature is not None
                            else 0
                        ),
                    )
                )
                break
    return examples


def compare_ablation_runs(
    config_paths: list[str | Path],
    output_dir: str | Path,
) -> AblationComparisonResult:
    """Compare completed ablation runs using their training/evaluation metrics.

    The comparison writes overlay loss curves, final metric bar plots, and a
    reconstruction/sparsity plot. These are standard SAE diagnostics: fidelity
    is tracked through training reconstruction loss and explained variance,
    while sparsity is tracked through L0 for the active latent count.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    runs = []
    for config_path in config_paths:
        config_path = Path(config_path)
        config = ProjectConfig.load(config_path)
        run_output = Path(config.training.output_dir)
        metrics_path = run_output / "metrics.json"
        checkpoint_path = run_output / "sae.pt"
        if not checkpoint_path.exists() or not metrics_path.exists():
            continue
        with metrics_path.open() as handle:
            metrics = json.load(handle)
        if not isinstance(metrics, list):
            raise ValueError(f"Expected a list of metrics in {metrics_path}")
        runs.append(
            {
                "name": config_path.stem,
                "config_path": str(config_path),
                "output_dir": str(run_output),
                "runtime_structure_dir": config.training.runtime_structure_dir,
                "checkpoint_path": str(checkpoint_path),
                "metrics_path": str(metrics_path),
                "feature_report_path": str(run_output / "feature_report.json"),
                "structure_analysis_dir": str(run_output / "structure_analysis"),
                "sae": config.sae,
                "trace": config.trace,
                "training": config.training,
                "metrics": metrics,
            }
        )

    if not runs:
        raise ValueError("No completed ablation runs found")

    summary_rows = [_summarize_ablation_run(run) for run in runs]
    summary_json_path = output_dir / "ablation_summary.json"
    with summary_json_path.open("w") as handle:
        json.dump(
            {
                "runs": summary_rows,
                "notes": {
                    "loss": "Lower is better for training reconstruction loss.",
                    "explained_variance": "Higher is better.",
                    "l0": "Average number of active SAE latents per token.",
                    "reconstruction_sparsity": (
                        "A standard SAE fidelity/sparsity view: compare "
                        "reconstruction loss against L0."
                    ),
                },
            },
            handle,
            indent=2,
        )

    summary_csv_path = output_dir / "ablation_summary.csv"
    with summary_csv_path.open("w", newline="") as handle:
        fieldnames = list(summary_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    _write_ablation_loss_overlay(runs, plots_dir / "loss_curves_by_run.png")
    _write_ablation_learning_rate_overlay(runs, plots_dir / "learning_rate_by_run.png")
    _write_ablation_final_bars(summary_rows, plots_dir / "final_metrics_by_run.png")
    _write_ablation_reconstruction_sparsity(
        summary_rows,
        plots_dir / "reconstruction_sparsity_frontier.png",
    )
    write_cross_ablation_analysis_grids(runs, plots_dir / "across_ablations")

    return AblationComparisonResult(
        output_dir=output_dir,
        summary_json_path=summary_json_path,
        summary_csv_path=summary_csv_path,
        plots_dir=plots_dir,
        num_runs=len(runs),
    )


def _summarize_ablation_run(run: dict[str, Any]) -> dict[str, str | float | int | None]:
    metrics = run["metrics"]
    train_entries = [entry for entry in metrics if "loss" in entry]
    final_train = train_entries[-1] if train_entries else {}
    sae = run["sae"]
    trace = run["trace"]
    training = run["training"]
    return {
        "name": run["name"],
        "config_path": run["config_path"],
        "output_dir": run["output_dir"],
        "runtime_structure_dir": run["runtime_structure_dir"],
        "top_k": int(sae.top_k),
        "n_latents": int(sae.n_latents),
        "use_time_conditioning": int(bool(sae.use_time_conditioning)),
        "target_module": trace.target_module,
        "batch_size": int(training.batch_size),
        "epochs": int(training.epochs),
        "samples_per_sequence": int(trace.samples_per_sequence),
        "lr_scheduler": training.lr_scheduler,
        "lr_warmup_steps": int(training.lr_warmup_steps),
        "final_step": _metric_int(final_train, "step"),
        "final_loss": _metric_float(final_train, "loss"),
        "final_reconstruction_loss": _metric_float(final_train, "reconstruction_loss"),
        "final_auxk_loss": _metric_float(final_train, "auxk_loss"),
        "final_l0": _metric_float(final_train, "l0"),
        "final_explained_variance": _metric_float(final_train, "explained_variance"),
        "final_learning_rate": _metric_float(final_train, "learning_rate"),
    }


def _metric_float(entry: dict[str, Any], key: str) -> float | None:
    if key not in entry:
        return None
    return float(entry[key])


def _metric_int(entry: dict[str, Any], key: str) -> int | None:
    if key not in entry:
        return None
    return int(entry[key])


def _entries_with_metric(
    metrics: list[dict[str, Any]],
    key: str,
) -> tuple[list[int], list[float]]:
    steps = [int(entry["step"]) for entry in metrics if key in entry]
    values = [float(entry[key]) for entry in metrics if key in entry]
    return steps, values


def _write_ablation_loss_overlay(runs: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for run in runs:
        train_steps, train_losses = _entries_with_metric(run["metrics"], "loss")
        if train_losses:
            ax.plot(train_steps, train_losses, linewidth=1.4, label=run["name"])
    ax.set_ylabel("train loss")
    ax.set_xlabel("optimizer step")
    ax.set_title("SAE Training Loss by Ablation")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def _write_ablation_learning_rate_overlay(runs: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for run in runs:
        steps, values = _entries_with_metric(run["metrics"], "learning_rate")
        if values:
            ax.plot(steps, values, linewidth=1.5, label=run["name"])
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning Rate Schedule by Ablation")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def _write_ablation_final_bars(
    summary_rows: list[dict[str, str | float | int | None]],
    path: Path,
) -> None:
    names = [str(row["name"]) for row in summary_rows]
    metrics = [
        ("final_loss", "final training loss"),
        ("final_reconstruction_loss", "final reconstruction loss"),
        ("final_explained_variance", "final explained variance"),
        ("final_l0", "final L0"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))
    for ax, (key, label) in zip(axes, metrics, strict=True):
        values = [row[key] for row in summary_rows]
        numeric_values = [float(value) if value is not None else np.nan for value in values]
        ax.bar(names, numeric_values)
        ax.set_title(label)
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def _write_ablation_reconstruction_sparsity(
    summary_rows: list[dict[str, str | float | int | None]],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in summary_rows:
        x_value = row["final_l0"]
        y_value = row["final_reconstruction_loss"]
        if x_value is None or y_value is None:
            continue
        ax.scatter(float(x_value), float(y_value), s=70)
        ax.annotate(str(row["name"]), (float(x_value), float(y_value)), fontsize=8)
    ax.set_xlabel("L0, active latents per token")
    ax.set_ylabel("reconstruction loss")
    ax.set_title("SAE Reconstruction/Sparsity Comparison")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_cross_ablation_png_grid(
    panels: list[dict[str, Any]],
    path: Path,
    title: str,
) -> None:
    n_cols = 2
    n_rows = max(1, math.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 4.8 * n_rows),
        squeeze=False,
    )
    for ax, panel in zip(axes.flat, panels, strict=False):
        source_path = Path(str(panel["source_path"]))
        ax.set_title(str(panel["run_name"]), fontsize=10)
        ax.axis("off")
        if panel["status"] != "missing" and source_path.is_file():
            ax.imshow(plt.imread(source_path))
        else:
            ax.text(
                0.5,
                0.5,
                f"missing\n{panel['relative_source_path']}",
                ha="center",
                va="center",
                wrap=True,
                transform=ax.transAxes,
            )
    for ax in list(axes.flat)[len(panels) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_cross_ablation_analysis_grids(
    runs: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key, relative_source, title in CROSS_ABLATION_ANALYSIS_PLOTS:
        panels = []
        for run in runs:
            analysis_dir = Path(str(run["structure_analysis_dir"]))
            status = "missing"
            if key == "selected_feature_timestep_profiles_overlay":
                status = regenerate_selected_timestep_overlay(analysis_dir)
            source_path = analysis_dir / relative_source
            if key != "selected_feature_timestep_profiles_overlay" and source_path.is_file():
                status = "existing"
            panels.append(
                {
                    "run_name": str(run["name"]),
                    "source_path": str(source_path),
                    "relative_source_path": relative_source,
                    "status": status,
                }
            )
        plot_path = output_dir / f"{key}.png"
        write_cross_ablation_png_grid(panels, plot_path, title)
        manifest[key] = {
            "key": key,
            "title": title,
            "plot_path": str(plot_path),
            "panels": panels,
        }
    manifest_path = output_dir / "manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_path.replace(manifest_path)
    return manifest


def load_structure_index(store: ActivationStore) -> list[StructureIndexEntry]:
    metadata = store.manifest.get("metadata") or {}
    index = metadata.get("structure_index")
    if index is None:
        raise ValueError(
            f"Activation store {store.root} does not declare a structure_index in metadata"
        )
    index_path = store.root / index
    if not index_path.exists():
        raise ValueError(f"Structure index does not exist: {index_path}")
    return load_structure_index_path(index_path)


def load_structure_index_path(index_path: str | Path) -> list[StructureIndexEntry]:
    index_path = Path(index_path)
    entries = []
    with index_path.open() as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            entries.append(
                StructureIndexEntry(
                    sequence_id=str(payload["sequence_id"]),
                    sample_index=int(payload["sample_index"]),
                    sequence=str(payload["sequence"]),
                    npz_path=str(payload["npz_path"]) if payload.get("npz_path") else None,
                    pdb_path=payload.get("pdb_path"),
                    pos_units=payload.get("pos_units"),
                    pdb_units=payload.get("pdb_units"),
                )
            )
    return entries


def summarize_structure_sample_counts(
    entries: list[StructureIndexEntry],
) -> dict[str, float | int]:
    counts = Counter(entry.sequence_id for entry in entries)
    values = np.asarray(list(counts.values()), dtype=np.float64)
    if values.size == 0:
        return {
            "num_sequences": 0,
            "num_structures": 0,
            "min_samples_per_sequence": 0,
            "max_samples_per_sequence": 0,
            "mean_samples_per_sequence": 0.0,
        }
    return {
        "num_sequences": len(counts),
        "num_structures": len(entries),
        "min_samples_per_sequence": int(values.min()),
        "max_samples_per_sequence": int(values.max()),
        "mean_samples_per_sequence": float(values.mean()),
    }


def compute_rmsf_by_sequence(
    store_root: Path,
    entries: list[StructureIndexEntry],
    *,
    sampling_scope: StructureSamplingScope,
) -> dict[str, np.ndarray]:
    require_equilibrium_source(sampling_scope)
    by_sequence: dict[str, list[StructureIndexEntry]] = defaultdict(list)
    for entry in entries:
        by_sequence[entry.sequence_id].append(entry)

    rmsf_by_sequence: dict[str, np.ndarray] = {}
    for sequence_id, sequence_entries in by_sequence.items():
        positions = []
        for entry in sorted(sequence_entries, key=lambda item: item.sample_index):
            positions.append(load_structure_positions(store_root, entry))
        if len(positions) < 2:
            continue
        stack = np.stack(positions, axis=0)
        aligned = align_trajectory_to_reference(stack)
        mean_pos = aligned.mean(axis=0, keepdims=True)
        rmsf_by_sequence[sequence_id] = np.sqrt(
            ((aligned - mean_pos) ** 2).sum(axis=-1).mean(axis=0)
        )
    return rmsf_by_sequence


def load_configured_rmsf_reference(config: ProjectConfig) -> RMSFReference | None:
    path_text = config.analysis.rmsf_reference_structure_dir
    if path_text is None:
        return None
    root = Path(path_text)
    index_path = root / "final_structures" / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing RMSF reference index: {index_path}")
    entries = tuple(load_structure_index_path(index_path))
    return RMSFReference(
        root=root,
        entries=entries,
        values_by_sequence=compute_rmsf_by_sequence(
            root, list(entries), sampling_scope=StructureSamplingScope.BIOEMU_32
        ),
    )


def write_rmsf_reference_cache(
    reference_root: Path,
    values_by_sequence: Mapping[str, np.ndarray],
    destination: Path,
) -> Path:
    """Atomically persist a compact RMSF mapping bound to one exact structure index."""

    root = Path(reference_root)
    index = root / "final_structures/index.jsonl"
    if not index.is_file():
        raise FileNotFoundError(f"missing RMSF reference index: {index}")
    values = {
        str(sequence_id): np.asarray(sequence_values, dtype=np.float64).tolist()
        for sequence_id, sequence_values in sorted(values_by_sequence.items())
    }
    if not values or any(not row or not np.isfinite(row).all() for row in values.values()):
        raise ValueError("RMSF reference cache values must be nonempty and finite")
    values_bytes = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    payload = {
        "schema_version": 1,
        "status": "complete",
        "structure_index": {
            "path": "final_structures/index.jsonl",
            "size": index.stat().st_size,
            "sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        },
        "sequence_count": len(values),
        "values_sha256": hashlib.sha256(values_bytes).hexdigest(),
        "values_by_sequence": values,
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_rmsf_reference_cache(
    cache_path: Path,
    reference_root: Path,
) -> dict[str, np.ndarray]:
    """Load a compact RMSF cache only when its exact structure index is unchanged."""

    path = Path(cache_path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("RMSF reference cache must be a JSON object")
    root = Path(reference_root)
    index = root / "final_structures/index.jsonl"
    identity = payload.get("structure_index")
    values = payload.get("values_by_sequence")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "complete"
        or not isinstance(identity, dict)
        or identity.get("path") != "final_structures/index.jsonl"
        or not index.is_file()
        or identity.get("size") != index.stat().st_size
        or identity.get("sha256") != hashlib.sha256(index.read_bytes()).hexdigest()
        or not isinstance(values, dict)
        or payload.get("sequence_count") != len(values)
    ):
        raise ValueError("RMSF reference cache identity is incomplete or stale")
    values_bytes = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    if payload.get("values_sha256") != hashlib.sha256(values_bytes).hexdigest():
        raise ValueError("RMSF reference cache values are stale")
    parsed = {
        str(sequence_id): np.asarray(sequence_values, dtype=np.float64)
        for sequence_id, sequence_values in values.items()
    }
    if not parsed or any(
        row.ndim != 1 or row.size == 0 or not np.isfinite(row).all() for row in parsed.values()
    ):
        raise ValueError("RMSF reference cache values must be nonempty and finite")
    return parsed


def compute_structure_analysis_properties(
    store_root: Path,
    entries: list[StructureIndexEntry],
    contact_cutoff: float = 8.0,
    *,
    sampling_scope: StructureSamplingScope,
) -> tuple[dict[str, np.ndarray], dict[tuple[str, int], dict[str, float]]]:
    """Load each structure once while computing ensemble and sample properties."""

    require_equilibrium_source(sampling_scope)

    by_sequence: dict[str, list[StructureIndexEntry]] = defaultdict(list)
    for entry in entries:
        by_sequence[entry.sequence_id].append(entry)
    rmsf_by_sequence: dict[str, np.ndarray] = {}
    properties: dict[tuple[str, int], dict[str, float]] = {}
    for sequence_id, sequence_entries in by_sequence.items():
        ordered_entries = sorted(sequence_entries, key=lambda item: item.sample_index)
        positions = [load_structure_positions(store_root, entry) for entry in ordered_entries]
        for entry, sample_positions in zip(ordered_entries, positions, strict=True):
            center = sample_positions.mean(axis=0, keepdims=True)
            properties[(sequence_id, entry.sample_index)] = {
                "radius_of_gyration": float(
                    np.sqrt(((sample_positions - center) ** 2).sum(axis=-1).mean())
                ),
                "contact_density": contact_density(
                    sample_positions,
                    contact_cutoff=contact_cutoff,
                ),
            }
        if len(positions) >= 2:
            aligned = align_trajectory_to_reference(np.stack(positions, axis=0))
            mean_pos = aligned.mean(axis=0, keepdims=True)
            rmsf_by_sequence[sequence_id] = np.sqrt(
                ((aligned - mean_pos) ** 2).sum(axis=-1).mean(axis=0)
            )
    add_within_sequence_normalized_sample_properties(
        properties,
        property_names=("radius_of_gyration", "contact_density"),
    )
    return rmsf_by_sequence, properties


def compute_radius_of_gyration_by_sample(
    store_root: Path,
    entries: list[StructureIndexEntry],
    *,
    sampling_scope: StructureSamplingScope,
) -> dict[tuple[str, int], float]:
    return {
        key: values["radius_of_gyration"]
        for key, values in compute_structure_sample_properties(
            store_root, entries, sampling_scope=sampling_scope
        ).items()
    }


def compute_structure_sample_properties(
    store_root: Path,
    entries: list[StructureIndexEntry],
    contact_cutoff: float = 8.0,
    *,
    sampling_scope: StructureSamplingScope,
) -> dict[tuple[str, int], dict[str, float]]:
    require_equilibrium_source(sampling_scope)
    by_sequence: dict[str, list[StructureIndexEntry]] = defaultdict(list)
    loaded_positions: dict[tuple[str, int], np.ndarray] = {}
    for entry in entries:
        by_sequence[entry.sequence_id].append(entry)
        loaded_positions[(entry.sequence_id, entry.sample_index)] = load_structure_positions(
            store_root,
            entry,
        )

    properties: dict[tuple[str, int], dict[str, float]] = {}
    for sequence_id, sequence_entries in by_sequence.items():
        ordered_sample_indices = sorted(entry.sample_index for entry in sequence_entries)
        for sample_index in ordered_sample_indices:
            positions = loaded_positions[(sequence_id, sample_index)]
            center = positions.mean(axis=0, keepdims=True)
            radius = float(np.sqrt(((positions - center) ** 2).sum(axis=-1).mean()))
            properties[(sequence_id, sample_index)] = {
                "radius_of_gyration": radius,
                "contact_density": contact_density(positions, contact_cutoff=contact_cutoff),
            }
    add_within_sequence_normalized_sample_properties(
        properties,
        property_names=("radius_of_gyration", "contact_density"),
    )
    return properties


#: The single owner of the stored-coordinate unit rule.
#
# BioEmu writes ``pos`` in NANOMETRES; every downstream constant in this repository
# (``CONTACT_CUTOFF_ANGSTROM = 8.0``, ``LOCAL_Q_BETA_PER_ANGSTROM = 5.0``, radius of gyration,
# RMSD, RMSF) is ANGSTROM. Any loader that opens a structure archive must route the stored array
# through :func:`positions_in_angstrom`; a second loader that re-implemented the ``pos_units``
# VALIDATION without the CONVERSION is exactly how nanometre coordinates once reached the 8 A
# contact cutoff and saturated ``contact_density`` at 1.0.
NANOMETER_TO_ANGSTROM = 10.0
ANGSTROM_UNIT_NAMES = frozenset({"angstrom", "angstroms", "angstrom_centered"})

#: The trans CA-CA virtual bond is 3.8 A in every folded or unfolded protein backbone, so the
#: median consecutive CA-CA distance is a protein-independent unit assay: nanometres read 0.38,
#: a double conversion reads 38, and garbled coordinates read anything else.
CA_CA_VIRTUAL_BOND_ANGSTROM = 3.8
MIN_MEDIAN_CA_CA_DISTANCE_ANGSTROM = 3.5
MAX_MEDIAN_CA_CA_DISTANCE_ANGSTROM = 4.1


class CoordinateUnitError(ValueError):
    """Ingested coordinates are not credible Angstrom CA positions."""


def positions_in_angstrom(
    positions: np.ndarray, units: str | None, *, source: object = None
) -> np.ndarray:
    """Convert stored structure coordinates to Angstrom, the unit every constant assumes."""

    if units == "nanometers":
        return positions * NANOMETER_TO_ANGSTROM
    if units in ANGSTROM_UNIT_NAMES:
        return positions
    raise ValueError(f"Unsupported coordinate units {units!r} in {source}")


def median_consecutive_ca_distance(positions: np.ndarray) -> float:
    """Return the median distance between sequence-consecutive CA atoms."""

    coordinates = np.asarray(positions, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or coordinates.shape[0] < 2:
        raise CoordinateUnitError(
            "CA-CA bond-length invariant needs at least two residues of shape [residues, 3]; "
            f"received array of shape {tuple(coordinates.shape)}"
        )
    steps = np.linalg.norm(np.diff(coordinates, axis=0), axis=-1)
    return float(np.median(steps))


def require_angstrom_ca_geometry(positions: np.ndarray, *, source: object = None) -> float:
    """Fail closed unless the CA backbone is on the Angstrom scale, returning the measurement.

    Placed at coordinate ingestion so a unit slip raises at the line that made the Angstrom
    assumption, rather than surfacing thousands of shards later as saturated contact density.
    """

    measured = median_consecutive_ca_distance(positions)
    if not (MIN_MEDIAN_CA_CA_DISTANCE_ANGSTROM <= measured <= MAX_MEDIAN_CA_CA_DISTANCE_ANGSTROM):
        raise CoordinateUnitError(
            "coordinates are not Angstrom CA positions: median consecutive CA-CA distance is "
            f"{measured:.6g}, expected the trans virtual bond "
            f"{CA_CA_VIRTUAL_BOND_ANGSTROM} A within "
            f"[{MIN_MEDIAN_CA_CA_DISTANCE_ANGSTROM}, {MAX_MEDIAN_CA_CA_DISTANCE_ANGSTROM}] A "
            f"(0.38 indicates unconverted nanometres, 38 a double conversion); source: {source}"
        )
    return measured


def load_structure_positions(store_root: Path, entry: StructureIndexEntry) -> np.ndarray:
    if entry.npz_path is not None:
        npz_path = resolve_structure_path(store_root, entry.npz_path)
        with np.load(npz_path) as data:
            positions = np.asarray(data["pos"], dtype=np.float64)
            stored_units = str(data["pos_units"]) if "pos_units" in data else None
        units = entry.pos_units or stored_units or "nanometers"
        return positions_in_angstrom(positions, units, source=npz_path)
    if entry.pdb_path is not None:
        return load_pdb_ca_positions(resolve_structure_path(store_root, entry.pdb_path))
    raise ValueError(
        f"Structure entry {entry.sequence_id} sample {entry.sample_index} has no npz_path or pdb_path"
    )


def resolve_structure_path(store_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else store_root / path


def load_pdb_ca_positions(pdb_path: Path) -> np.ndarray:
    ca_positions: list[tuple[float, float, float]] = []
    first_atom_by_residue: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            residue_key = (line[21].strip(), line[22:26].strip(), line[26].strip())
            try:
                coords = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except ValueError:
                continue
            first_atom_by_residue.setdefault(residue_key, coords)
            if atom_name == "CA":
                ca_positions.append(coords)
    positions = ca_positions or list(first_atom_by_residue.values())
    if not positions:
        raise ValueError(f"No residue coordinates found in PDB: {pdb_path}")
    return np.asarray(positions, dtype=np.float64)


def add_within_sequence_normalized_sample_properties(
    structure_properties: dict[tuple[str, int], dict[str, float]],
    property_names: tuple[str, ...],
) -> None:
    for property_name in property_names:
        values_by_sequence: dict[str, list[tuple[tuple[str, int], float]]] = defaultdict(list)
        for key, values in structure_properties.items():
            value = values.get(property_name)
            if value is None or not math.isfinite(float(value)):
                continue
            values_by_sequence[key[0]].append((key, float(value)))
        for entries in values_by_sequence.values():
            if not entries:
                continue
            raw_values = np.asarray([value for _, value in entries], dtype=np.float64)
            mean = float(raw_values.mean())
            std = float(raw_values.std())
            normalized_name = f"relative_{property_name}"
            for key, value in entries:
                structure_properties[key][normalized_name] = (
                    (value - mean) / std if std > 1e-12 else 0.0
                )


def within_sequence_zscore_features(
    sequence_ids: list[str],
    feature_values: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    if feature_values.ndim != 2:
        raise ValueError("feature_values must have shape [structures, features]")
    if len(sequence_ids) != feature_values.shape[0]:
        raise ValueError("sequence_ids must match feature rows")
    normalized = torch.zeros_like(feature_values, dtype=torch.float32)
    rows_by_sequence: dict[str, list[int]] = defaultdict(list)
    for row_idx, sequence_id in enumerate(sequence_ids):
        rows_by_sequence[sequence_id].append(row_idx)
    values = feature_values.float()
    for row_indices in rows_by_sequence.values():
        group = values[row_indices]
        centered = group - group.mean(dim=0, keepdim=True)
        std = group.std(dim=0, correction=0, keepdim=True)
        varying = std > epsilon
        normalized[row_indices] = torch.where(
            varying,
            centered / std.clamp_min(epsilon),
            0.0,
        )
    return normalized


def contact_density(
    positions: np.ndarray,
    contact_cutoff: float,
    min_sequence_separation: int = 3,
) -> float:
    contacts = contact_matrix(positions, contact_cutoff)
    n = contacts.shape[0]
    residue_indices = np.arange(n)
    eligible = np.triu(
        np.abs(residue_indices[:, None] - residue_indices[None, :]) >= min_sequence_separation,
        k=1,
    )
    possible = int(eligible.sum())
    if possible <= 0:
        return 0.0
    return float(np.logical_and(contacts, eligible).sum() / possible)


def contact_matrix(positions: np.ndarray, contact_cutoff: float) -> np.ndarray:
    if positions.size == 0:
        return np.zeros((0, 0), dtype=bool)
    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt((deltas**2).sum(axis=-1))
    contacts = distances <= contact_cutoff
    np.fill_diagonal(contacts, False)
    return contacts


def align_trajectory_to_reference(positions: np.ndarray) -> np.ndarray:
    """Rigidly align all frames to the first frame using Kabsch superposition."""

    reference = positions[0] - positions[0].mean(axis=0, keepdims=True)
    aligned = np.empty_like(positions)
    for frame_idx, frame in enumerate(positions):
        centered = frame - frame.mean(axis=0, keepdims=True)
        covariance = centered.T @ reference
        u, _, vt = np.linalg.svd(covariance)
        correction = np.eye(3)
        correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
        rotation = u @ correction @ vt
        aligned[frame_idx] = centered @ rotation
    return aligned


def rmsf_property_labels(
    rmsf_by_sequence: dict[str, np.ndarray],
    high_quantile: float,
    low_quantile: float,
) -> dict[str, np.ndarray]:
    labels = {}
    for sequence_id, rmsf in rmsf_by_sequence.items():
        high = np.quantile(rmsf, high_quantile)
        low = np.quantile(rmsf, low_quantile)
        values = np.zeros_like(rmsf, dtype=np.int8)
        values[rmsf >= high] = 1
        values[rmsf <= low] = -1
        labels[sequence_id] = values
    return labels


def rmsf_token_property(
    rmsf_by_sequence: dict[str, np.ndarray],
    high_quantile: float,
    low_quantile: float,
) -> TokenProperty:
    return TokenProperty(
        name="high_rmsf",
        description="high-RMSF residues",
        labels_by_sequence=rmsf_property_labels(
            rmsf_by_sequence,
            high_quantile=high_quantile,
            low_quantile=low_quantile,
        ),
        high_label="high RMSF residues",
        low_label="low RMSF residues",
    )


def build_token_properties(
    sequences: dict[str, str],
    rmsf_by_sequence: dict[str, np.ndarray],
    structure_properties: dict[tuple[str, int], dict[str, float]],
    high_quantile: float,
    low_quantile: float,
) -> list[TokenProperty]:
    properties = [
        rmsf_token_property(rmsf_by_sequence, high_quantile, low_quantile),
        TokenProperty(
            name="early_vs_late_timestep",
            description="early/noisy versus late/denoised diffusion timesteps",
            label_fn=lambda token: _quantile_label(
                token["timestep"],
                low=low_quantile,
                high=high_quantile,
            ),
            high_label="early/noisy high-t timesteps",
            low_label="late/denoised low-t timesteps",
        ),
        TokenProperty(
            name="glycine_proline_context",
            description="glycine/proline-containing local sequence context",
            labels_by_sequence=sequence_context_labels(
                sequences,
                residues={"G", "P"},
                window=2,
            ),
            high_label="G/P local context",
            low_label="non-G/P local context",
        ),
        TokenProperty(
            name="disorder_promoting_residues",
            description="disorder-promoting residue classes",
            labels_by_sequence=residue_class_labels(
                sequences,
                positive=set("ARGQSPEK"),
                negative=set("WCFIYVLN"),
            ),
            high_label="disorder-promoting residues",
            low_label="order-promoting hydrophobics",
        ),
        TokenProperty(
            name="hydrophobic_residues",
            description="hydrophobic residue identity",
            labels_by_sequence=residue_class_labels(
                sequences,
                positive=set("AVLIMFWY"),
                negative=set("DEKRHNQSTGP"),
            ),
            high_label="hydrophobic residues",
            low_label="polar/charged/special residues",
        ),
        TokenProperty(
            name="charged_residues",
            description="charged residue identity",
            labels_by_sequence=residue_class_labels(
                sequences,
                positive=set("DEKRH"),
                negative=set("AVLIMFWYSTNQCGP"),
            ),
            high_label="charged residues",
            low_label="uncharged residues",
        ),
    ]
    sample_property_specs = [
        (
            "radius_of_gyration",
            "expanded_final_structure",
            "expanded versus compact final structures by radius of gyration",
            "expanded/high-Rg samples",
            "compact/low-Rg samples",
        ),
        (
            "contact_density",
            "dense_contact_structure",
            "dense versus sparse contact maps",
            "high contact-density samples",
            "low contact-density samples",
        ),
    ]
    for property_name, name, description, high_label, low_label in sample_property_specs:
        thresholds = sample_property_thresholds_by_sequence(
            structure_properties,
            property_name,
            high_quantile=high_quantile,
            low_quantile=low_quantile,
        )
        if not thresholds:
            continue
        properties.append(
            TokenProperty(
                name=name,
                description=description,
                label_fn=sample_property_label_fn(
                    structure_properties=structure_properties,
                    property_name=property_name,
                    thresholds=thresholds,
                ),
                high_label=high_label,
                low_label=low_label,
            )
        )
    return properties


def build_sequence_token_properties(
    sequences: dict[str, str],
    high_quantile: float,
    low_quantile: float,
    timestep_name: str = "late_vs_early_timestep",
    timestep_description: str = "late versus early model-depth positions",
    timestep_high_label: str = "late/high model-depth positions",
    timestep_low_label: str = "early/low model-depth positions",
) -> list[TokenProperty]:
    """Token properties that do not depend on sampled structure ensembles."""

    return [
        TokenProperty(
            name=timestep_name,
            description=timestep_description,
            label_fn=lambda token: _quantile_label(
                token["timestep"],
                low=low_quantile,
                high=high_quantile,
            ),
            high_label=timestep_high_label,
            low_label=timestep_low_label,
        ),
        TokenProperty(
            name="glycine_proline_context",
            description="glycine/proline-containing local sequence context",
            labels_by_sequence=sequence_context_labels(
                sequences,
                residues={"G", "P"},
                window=2,
            ),
            high_label="G/P local context",
            low_label="non-G/P local context",
        ),
        TokenProperty(
            name="disorder_promoting_residues",
            description="disorder-promoting residue classes",
            labels_by_sequence=residue_class_labels(
                sequences,
                positive=set("ARGQSPEK"),
                negative=set("WCFIYVLN"),
            ),
            high_label="disorder-promoting residues",
            low_label="order-promoting hydrophobics",
        ),
        TokenProperty(
            name="hydrophobic_residues",
            description="hydrophobic residue identity",
            labels_by_sequence=residue_class_labels(
                sequences,
                positive=set("AVLIMFWY"),
                negative=set("DEKRHNQSTGP"),
            ),
            high_label="hydrophobic residues",
            low_label="polar/charged/special residues",
        ),
        TokenProperty(
            name="charged_residues",
            description="charged residue identity",
            labels_by_sequence=residue_class_labels(
                sequences,
                positive=set("DEKRH"),
                negative=set("AVLIMFWYSTNQCGP"),
            ),
            high_label="charged residues",
            low_label="uncharged residues",
        ),
    ]


def build_continuous_token_properties(
    rmsf_by_sequence: dict[str, np.ndarray],
    structure_properties: dict[tuple[str, int], dict[str, float]],
) -> list[ContinuousTokenProperty]:
    properties = [
        ContinuousTokenProperty(
            name="rmsf",
            description="per-residue RMSF",
            values_by_sequence=rmsf_by_sequence,
            units="angstrom",
        ),
        ContinuousTokenProperty(
            name="diffusion_timestep",
            description="BioEmu diffusion timestep",
            value_fn=lambda token: float(token["timestep"]),
        ),
    ]
    sample_property_specs = [
        ("radius_of_gyration", "radius of gyration", "angstrom"),
        ("contact_density", "contact density", ""),
        (
            "relative_radius_of_gyration",
            "within-sequence relative radius of gyration",
            "z-score",
        ),
        (
            "relative_contact_density",
            "within-sequence relative contact density",
            "z-score",
        ),
    ]
    for property_name, description, units in sample_property_specs:
        if any(property_name in values for values in structure_properties.values()):
            properties.append(
                ContinuousTokenProperty(
                    name=property_name,
                    description=description,
                    value_fn=sample_property_value_fn(structure_properties, property_name),
                    units=units,
                    aggregation_level="sample",
                    normalize_features_within_sequence=property_name.startswith("relative_"),
                )
            )
    timestep_property_specs = [
        (
            "relative_radius_of_gyration",
            "within-sequence relative radius of gyration",
            "z-score",
        ),
        (
            "relative_contact_density",
            "within-sequence relative contact density",
            "z-score",
        ),
    ]
    for base_property_name, base_description, units in timestep_property_specs:
        if not any(base_property_name in values for values in structure_properties.values()):
            continue
        for timestep_bin in ("early", "late"):
            properties.append(
                ContinuousTokenProperty(
                    name=f"{timestep_bin}_{base_property_name}",
                    description=f"{base_description} from {timestep_bin} timestep activations",
                    value_fn=sample_property_value_fn(
                        structure_properties,
                        base_property_name,
                    ),
                    units=units,
                    aggregation_level="sample",
                    timestep_bin=timestep_bin,
                    normalize_features_within_sequence=True,
                )
            )
    return properties


def sample_property_label_fn(
    structure_properties: dict[tuple[str, int], dict[str, float]],
    property_name: str,
    thresholds: dict[str, tuple[float, float]],
) -> Any:
    values_by_sample = {
        key: values[property_name]
        for key, values in structure_properties.items()
        if property_name in values
    }

    def label(token: dict[str, Any]) -> int:
        sample_index = token["sample_index"]
        sequence_id = token["sequence_id"]
        if sample_index is None or sequence_id not in thresholds:
            return 0
        value = values_by_sample.get((sequence_id, int(sample_index)))
        if value is None:
            return 0
        low, high = thresholds[sequence_id]
        if value >= high:
            return 1
        if value <= low:
            return -1
        return 0

    return label


def sample_property_value_fn(
    structure_properties: dict[tuple[str, int], dict[str, float]],
    property_name: str,
) -> Any:
    values_by_sample = {
        key: float(values[property_name])
        for key, values in structure_properties.items()
        if property_name in values and math.isfinite(float(values[property_name]))
    }

    def value(token: dict[str, Any]) -> float:
        sample_index = token["sample_index"]
        if sample_index is None:
            return float("nan")
        return values_by_sample.get(
            (token["sequence_id"], int(sample_index)),
            float("nan"),
        )

    return value


def sample_property_thresholds_by_sequence(
    structure_properties: dict[tuple[str, int], dict[str, float]],
    property_name: str,
    high_quantile: float,
    low_quantile: float,
) -> dict[str, tuple[float, float]]:
    values_by_sequence: dict[str, list[float]] = defaultdict(list)
    for (sequence_id, _sample_index), values in structure_properties.items():
        value = values.get(property_name)
        if value is not None and math.isfinite(float(value)):
            values_by_sequence[sequence_id].append(float(value))
    thresholds = {}
    for sequence_id, values in values_by_sequence.items():
        if len(values) < 2:
            continue
        arr = np.asarray(values, dtype=np.float64)
        thresholds[sequence_id] = (
            float(np.quantile(arr, low_quantile)),
            float(np.quantile(arr, high_quantile)),
        )
    return thresholds


def residue_class_labels(
    sequences: dict[str, str],
    positive: set[str],
    negative: set[str],
) -> dict[str, np.ndarray]:
    labels = {}
    for sequence_id, sequence in sequences.items():
        values = np.zeros(len(sequence), dtype=np.int8)
        for idx, aa in enumerate(sequence):
            if aa in positive:
                values[idx] = 1
            elif aa in negative:
                values[idx] = -1
        labels[sequence_id] = values
    return labels


def sequence_context_labels(
    sequences: dict[str, str],
    residues: set[str],
    window: int,
) -> dict[str, np.ndarray]:
    labels = {}
    for sequence_id, sequence in sequences.items():
        values = np.full(len(sequence), -1, dtype=np.int8)
        for idx in range(len(sequence)):
            start = max(0, idx - window)
            end = min(len(sequence), idx + window + 1)
            if any(aa in residues for aa in sequence[start:end]):
                values[idx] = 1
        labels[sequence_id] = values
    return labels


def collect_runtime_feature_analysis_pass(
    trainer: SAETrainer,
    config: ProjectConfig,
    structure_entries: list[StructureIndexEntry],
    token_properties: list[TokenProperty],
    continuous_properties: list[ContinuousTokenProperty],
    top_k_examples: int,
    prediction_sample_size: int = 4096,
    residue_feature_accumulator: ResidueFeatureActivationAccumulator | None = None,
) -> RuntimeFeatureAnalysisPass:
    stats_by_property: dict[str, dict[str, FeatureGroupStats]] = {
        token_property.name: {} for token_property in token_properties
    }
    example_candidates: dict[int, dict[tuple[str, int], FeatureExampleCandidate]] = defaultdict(
        dict
    )
    summary_accumulator: FeatureSummaryAccumulator | None = None
    timestep_accumulators: dict[int, dict[str, torch.Tensor]] | None = None
    concept_accumulators: dict[str, dict[str, Any]] | None = None
    prediction_accumulators: dict[str, FeaturePredictionAccumulator] | None = None
    sample_prediction_accumulator: SampleFeaturePredictionAccumulator | None = None
    prediction_reservoir = PredictionReservoir(max_rows=prediction_sample_size)
    token_continuous_properties = [
        prop for prop in continuous_properties if prop.aggregation_level == "token"
    ]
    rmsf_property = next(
        (prop for prop in token_continuous_properties if prop.name == "rmsf"),
        None,
    )
    streamed_token_continuous_properties = [
        prop for prop in token_continuous_properties if prop.name != "rmsf"
    ]
    sample_continuous_properties = [
        prop for prop in continuous_properties if prop.aggregation_level == "sample"
    ]
    thresholds = (0.0,)
    trainer.model.eval()
    rmsf_pooling_accumulator: RMSFFeaturePoolingAccumulator | None = None
    with torch.no_grad():
        for record in iter_bioemu_runtime_activation_records(
            config, structure_entries, trainer.device
        ):
            for start in range(0, record.n_tokens, trainer.config.training.batch_size):
                end = start + trainer.config.training.batch_size
                activations = record.activations[start:end].to(trainer.device)
                timesteps = record.timesteps[start:end].to(trainer.device)
                with trainer.autocast_context():
                    latents = trainer.model(activations, timesteps).latents.detach().cpu()
                if summary_accumulator is None:
                    n_latents = latents.shape[1]
                    summary_accumulator = FeatureSummaryAccumulator.create(n_latents)
                    timestep_accumulators = create_timestep_accumulators(n_latents)
                    concept_accumulators = create_concept_accumulators(
                        token_properties,
                        n_latents,
                        thresholds,
                    )
                    prediction_accumulators = create_prediction_accumulators(
                        streamed_token_continuous_properties,
                        n_latents,
                    )
                    if rmsf_property is not None:
                        rmsf_pooling_accumulator = RMSFFeaturePoolingAccumulator(
                            n_latents=n_latents,
                            rmsf_by_sequence=rmsf_property.values_by_sequence or {},
                            prediction_sample_size=prediction_sample_size,
                            retain_structure_summaries=False,
                            selected_sample_index=0,
                        )
                    sample_prediction_accumulator = SampleFeaturePredictionAccumulator.create(
                        n_latents
                    )
                _accumulate_feature_summary(summary_accumulator, latents, record, start, end)
                if residue_feature_accumulator is not None:
                    residue_feature_accumulator.accumulate(latents, record, start, end)
                assert timestep_accumulators is not None
                _accumulate_timestep_profiles(timestep_accumulators, latents, record, start, end)
                assert prediction_accumulators is not None
                _accumulate_prediction_metrics(
                    accumulators=prediction_accumulators,
                    reservoir=prediction_reservoir,
                    latents=latents,
                    record=record,
                    start=start,
                    end=end,
                    continuous_properties=streamed_token_continuous_properties,
                )
                if rmsf_pooling_accumulator is not None:
                    rmsf_pooling_accumulator.accumulate(latents, record, start, end)
                if sample_continuous_properties:
                    assert sample_prediction_accumulator is not None
                    _accumulate_sample_prediction_metrics(
                        accumulator=sample_prediction_accumulator,
                        latents=latents,
                        record=record,
                        start=start,
                        end=end,
                        continuous_properties=sample_continuous_properties,
                        descending_timestep=True,
                    )
                for token_property in token_properties:
                    _accumulate_feature_group_stats(
                        stats=stats_by_property[token_property.name],
                        latents=latents,
                        record=record,
                        start=start,
                        end=end,
                        token_property=token_property,
                    )
                    assert concept_accumulators is not None
                    _accumulate_concept_metrics(
                        accumulator=concept_accumulators[token_property.name],
                        latents=latents,
                        record=record,
                        start=start,
                        end=end,
                        token_property=token_property,
                    )
                if top_k_examples > 0:
                    _accumulate_top_feature_examples(
                        candidates_by_feature=example_candidates,
                        latents=latents,
                        record=record,
                        start=start,
                        end=end,
                        top_k_examples=top_k_examples,
                    )
    prediction_sample = prediction_reservoir.finalize(
        [prop.name for prop in streamed_token_continuous_properties]
    )
    property_importance = finalize_property_importance(
        prediction_accumulators,
        prediction_sample,
        streamed_token_continuous_properties,
    )
    if rmsf_pooling_accumulator is not None and rmsf_property is not None:
        rmsf_sample, _ = rmsf_pooling_accumulator.finalize()
        property_importance.update(
            finalize_property_importance(
                {"rmsf": rmsf_pooling_accumulator.metric_accumulator},
                rmsf_sample,
                [rmsf_property],
            )
        )
        prediction_sample = merge_prediction_samples(
            prediction_sample,
            {"rmsf": rmsf_sample} if rmsf_sample is not None else {},
        )
    sample_importance, sample_prediction_samples = finalize_sample_property_importance(
        sample_prediction_accumulator,
        sample_continuous_properties,
        prediction_sample_size=prediction_sample_size,
    )
    property_importance.update(sample_importance)
    prediction_sample = merge_prediction_samples(prediction_sample, sample_prediction_samples)
    return RuntimeFeatureAnalysisPass(
        stats_by_property=stats_by_property,
        examples_by_feature={
            feature_idx: sorted(
                candidates.values(), key=lambda item: item.activation, reverse=True
            )[:top_k_examples]
            for feature_idx, candidates in example_candidates.items()
        },
        feature_summary=finalize_feature_summary(summary_accumulator),
        timestep_profiles=finalize_timestep_profiles(timestep_accumulators),
        concept_metrics=finalize_concept_metrics(concept_accumulators),
        property_importance=property_importance,
        prediction_sample=prediction_sample,
    )


def _accumulate_top_feature_examples(
    candidates_by_feature: dict[int, dict[tuple[str, int], FeatureExampleCandidate]],
    latents: torch.Tensor,
    record: ActivationRecord,
    start: int,
    end: int,
    top_k_examples: int,
) -> None:
    sequence_ids = record.sequence_ids[start:end]
    residues = record.residue_indices[start:end].detach().cpu().tolist()
    sample_indices = (
        record.sample_indices[start:end].detach().cpu().tolist()
        if record.sample_indices is not None
        else [None] * len(sequence_ids)
    )
    timestep_values = record.timesteps[start:end].detach().cpu().tolist()
    active_features = torch.nonzero(latents > 0, as_tuple=False)
    for token_idx_tensor, feature_idx_tensor in active_features:
        token_idx = int(token_idx_tensor.item())
        feature_idx = int(feature_idx_tensor.item())
        candidate = FeatureExampleCandidate(
            activation=float(latents[token_idx, feature_idx].item()),
            sequence_id=sequence_ids[token_idx],
            residue_index=int(residues[token_idx]),
            timestep=float(timestep_values[token_idx]),
            sample_index=(
                int(sample_indices[token_idx]) if sample_indices[token_idx] is not None else None
            ),
        )
        candidates = candidates_by_feature[feature_idx]
        key = (candidate.sequence_id, candidate.residue_index)
        previous = candidates.get(key)
        if previous is None or candidate.activation > previous.activation:
            candidates[key] = candidate
        if len(candidates) > 2 * top_k_examples:
            retained = sorted(
                candidates.values(),
                key=lambda item: item.activation,
                reverse=True,
            )[:top_k_examples]
            candidates.clear()
            candidates.update(((item.sequence_id, item.residue_index), item) for item in retained)


def collect_store_feature_analysis_pass(
    trainer: SAETrainer,
    store: ActivationStore,
    token_properties: list[TokenProperty],
    continuous_properties: list[ContinuousTokenProperty],
    top_k_examples: int,
    prediction_sample_size: int = 4096,
    residue_feature_accumulator: ResidueFeatureActivationAccumulator | None = None,
) -> RuntimeFeatureAnalysisPass:
    """Collect all store-backed activation analyses in one read pass."""

    stats_by_property: dict[str, dict[str, FeatureGroupStats]] = {
        token_property.name: {} for token_property in token_properties
    }
    example_candidates: dict[int, dict[tuple[str, int], FeatureExampleCandidate]] = defaultdict(
        dict
    )
    summary_accumulator: FeatureSummaryAccumulator | None = None
    timestep_accumulators: dict[int, dict[str, torch.Tensor]] | None = None
    concept_accumulators: dict[str, dict[str, Any]] | None = None
    prediction_accumulators: dict[str, FeaturePredictionAccumulator] | None = None
    sample_prediction_accumulator: SampleFeaturePredictionAccumulator | None = None
    prediction_reservoir = PredictionReservoir(max_rows=prediction_sample_size)
    token_continuous_properties = [
        prop for prop in continuous_properties if prop.aggregation_level == "token"
    ]
    sample_continuous_properties = [
        prop for prop in continuous_properties if prop.aggregation_level == "sample"
    ]
    thresholds = (0.0,)
    trainer.model.eval()
    with torch.no_grad():
        for record in store.iter_records():
            for start in range(0, record.n_tokens, trainer.config.training.batch_size):
                end = start + trainer.config.training.batch_size
                activations = record.activations[start:end].to(trainer.device)
                timesteps = record.timesteps[start:end].to(trainer.device)
                with trainer.autocast_context():
                    latents = trainer.model(activations, timesteps).latents.detach().cpu()
                if summary_accumulator is None:
                    n_latents = latents.shape[1]
                    summary_accumulator = FeatureSummaryAccumulator.create(n_latents)
                    timestep_accumulators = create_timestep_accumulators(n_latents)
                    concept_accumulators = create_concept_accumulators(
                        token_properties,
                        n_latents,
                        thresholds,
                    )
                    prediction_accumulators = create_prediction_accumulators(
                        token_continuous_properties,
                        n_latents,
                    )
                    sample_prediction_accumulator = SampleFeaturePredictionAccumulator.create(
                        n_latents
                    )
                _accumulate_feature_summary(summary_accumulator, latents, record, start, end)
                if residue_feature_accumulator is not None:
                    residue_feature_accumulator.accumulate(latents, record, start, end)
                assert timestep_accumulators is not None
                _accumulate_timestep_profiles(timestep_accumulators, latents, record, start, end)
                assert prediction_accumulators is not None
                _accumulate_prediction_metrics(
                    accumulators=prediction_accumulators,
                    reservoir=prediction_reservoir,
                    latents=latents,
                    record=record,
                    start=start,
                    end=end,
                    continuous_properties=token_continuous_properties,
                )
                if sample_continuous_properties:
                    assert sample_prediction_accumulator is not None
                    _accumulate_sample_prediction_metrics(
                        accumulator=sample_prediction_accumulator,
                        latents=latents,
                        record=record,
                        start=start,
                        end=end,
                        continuous_properties=sample_continuous_properties,
                        descending_timestep=True,
                    )
                for token_property in token_properties:
                    _accumulate_feature_group_stats(
                        stats=stats_by_property[token_property.name],
                        latents=latents,
                        record=record,
                        start=start,
                        end=end,
                        token_property=token_property,
                    )
                    assert concept_accumulators is not None
                    _accumulate_concept_metrics(
                        accumulator=concept_accumulators[token_property.name],
                        latents=latents,
                        record=record,
                        start=start,
                        end=end,
                        token_property=token_property,
                    )
                if top_k_examples > 0:
                    _accumulate_top_feature_examples(
                        candidates_by_feature=example_candidates,
                        latents=latents,
                        record=record,
                        start=start,
                        end=end,
                        top_k_examples=top_k_examples,
                    )
    prediction_sample = prediction_reservoir.finalize(
        [prop.name for prop in token_continuous_properties]
    )
    property_importance = finalize_property_importance(
        prediction_accumulators,
        prediction_sample,
        token_continuous_properties,
    )
    sample_importance, sample_prediction_samples = finalize_sample_property_importance(
        sample_prediction_accumulator,
        sample_continuous_properties,
        prediction_sample_size=prediction_sample_size,
    )
    property_importance.update(sample_importance)
    prediction_sample = merge_prediction_samples(prediction_sample, sample_prediction_samples)
    return RuntimeFeatureAnalysisPass(
        stats_by_property=stats_by_property,
        examples_by_feature={
            feature_idx: sorted(
                candidates.values(), key=lambda item: item.activation, reverse=True
            )[:top_k_examples]
            for feature_idx, candidates in example_candidates.items()
        },
        feature_summary=finalize_feature_summary(summary_accumulator),
        timestep_profiles=finalize_timestep_profiles(timestep_accumulators),
        concept_metrics=finalize_concept_metrics(concept_accumulators),
        property_importance=property_importance,
        prediction_sample=prediction_sample,
    )


def _accumulate_feature_summary(
    accumulator: FeatureSummaryAccumulator,
    latents: torch.Tensor,
    record: ActivationRecord,
    start: int,
    end: int,
) -> None:
    values = latents.float()
    n_tokens = values.shape[0]
    accumulator.total_tokens += n_tokens
    accumulator.sums += values.sum(dim=0)
    accumulator.counts += n_tokens
    accumulator.active_counts += (values > 0).sum(dim=0)
    accumulator.max_values = torch.maximum(accumulator.max_values, values.max(dim=0).values)

    sequence_ids = record.sequence_ids[start:end]
    sample_indices = (
        record.sample_indices[start:end].detach().cpu().tolist()
        if record.sample_indices is not None
        else [None] * len(sequence_ids)
    )
    if accumulator.dense_tracking:
        assert accumulator.dense_active_counts_by_sequence is not None
        assert accumulator.dense_active_samples is not None
        active = values > 0
        sample_values = (
            record.sample_indices[start:end].detach().cpu().long()
            if record.sample_indices is not None
            else None
        )
        for sequence_id in dict.fromkeys(sequence_ids):
            sequence_mask = torch.tensor(
                [candidate == sequence_id for candidate in sequence_ids],
                dtype=torch.bool,
            )
            accumulator.token_counts_by_sequence[sequence_id] += int(sequence_mask.sum())
            counts = active[sequence_mask].sum(dim=0, dtype=torch.int32)
            previous = accumulator.dense_active_counts_by_sequence.get(sequence_id)
            accumulator.dense_active_counts_by_sequence[sequence_id] = (
                counts if previous is None else previous + counts
            )
            if sample_values is not None:
                for sample_index in torch.unique(sample_values[sequence_mask]).tolist():
                    sample_mask = torch.logical_and(sequence_mask, sample_values == sample_index)
                    key = (sequence_id, int(sample_index))
                    sample_active = active[sample_mask].any(dim=0)
                    previous_sample = accumulator.dense_active_samples.get(key)
                    accumulator.dense_active_samples[key] = (
                        sample_active
                        if previous_sample is None
                        else torch.logical_or(previous_sample, sample_active)
                    )
        return

    for sequence_id in sequence_ids:
        accumulator.token_counts_by_sequence[sequence_id] += 1
    active_features = torch.nonzero(values > 0, as_tuple=False)
    for token_idx_tensor, feature_idx_tensor in active_features:
        token_idx = int(token_idx_tensor.item())
        feature_idx = int(feature_idx_tensor.item())
        sequence_id = sequence_ids[token_idx]
        accumulator.active_sequences[feature_idx].add(sequence_id)
        accumulator.active_residue_counts_by_sequence[feature_idx][sequence_id] += 1
        sample_index = sample_indices[token_idx]
        if sample_index is not None:
            accumulator.active_samples[feature_idx].add((sequence_id, int(sample_index)))


def finalize_feature_summary(
    accumulator: FeatureSummaryAccumulator | None,
) -> list[dict[str, float | int]]:
    if accumulator is None:
        return []
    n_latents = accumulator.sums.numel()
    total_sequences = max(len(accumulator.token_counts_by_sequence), 1)
    total_tokens = max(accumulator.total_tokens, 1)
    total_samples = max(
        (
            len(accumulator.dense_active_samples or {})
            if accumulator.dense_tracking
            else len(
                {
                    sample
                    for feature_samples in accumulator.active_samples
                    for sample in feature_samples
                }
            )
        ),
        1,
    )
    if accumulator.dense_tracking:
        assert accumulator.dense_active_counts_by_sequence is not None
        assert accumulator.dense_active_samples is not None
        active_sequence_counts = torch.zeros(n_latents)
        active_sample_counts = torch.zeros(n_latents)
        fraction_sums = torch.zeros(n_latents)
        for sequence_id, counts in accumulator.dense_active_counts_by_sequence.items():
            present = counts > 0
            active_sequence_counts += present
            token_count = max(accumulator.token_counts_by_sequence[sequence_id], 1)
            fraction_sums += counts.float() / token_count
        for present in accumulator.dense_active_samples.values():
            active_sample_counts += present
        fraction_denominator = active_sequence_counts.clamp_min(1)
        avg_fraction_when_present = fraction_sums / fraction_denominator
    else:
        active_sequence_counts = None
        active_sample_counts = None
        avg_fraction_when_present = None

    rows = []
    for feature_idx in range(n_latents):
        if accumulator.dense_tracking:
            assert active_sequence_counts is not None
            assert active_sample_counts is not None
            assert avg_fraction_when_present is not None
            active_sequence_count = float(active_sequence_counts[feature_idx])
            active_sample_count = float(active_sample_counts[feature_idx])
            mean_active_fraction = float(avg_fraction_when_present[feature_idx])
        else:
            active_sequences = accumulator.active_sequences[feature_idx]
            active_residue_counts = accumulator.active_residue_counts_by_sequence[feature_idx]
            pct_when_present = []
            for sequence_id, active_count in active_residue_counts.items():
                sequence_token_count = accumulator.token_counts_by_sequence.get(sequence_id, 0)
                if sequence_token_count > 0:
                    pct_when_present.append(active_count / sequence_token_count)
            active_sequence_count = float(len(active_sequences))
            active_sample_count = float(len(accumulator.active_samples[feature_idx]))
            mean_active_fraction = float(np.mean(pct_when_present) if pct_when_present else 0.0)
        max_value = float(accumulator.max_values[feature_idx].item())
        if not math.isfinite(max_value):
            max_value = 0.0
        rows.append(
            {
                "feature": feature_idx,
                "mean_activation": float(accumulator.sums[feature_idx].item() / total_tokens),
                "max_activation": max_value,
                "active_token_fraction": float(
                    accumulator.active_counts[feature_idx].item() / total_tokens
                ),
                "active_sequence_fraction": float(active_sequence_count / total_sequences),
                "active_sample_fraction": float(active_sample_count / total_samples),
                "avg_active_token_fraction_when_present": mean_active_fraction,
            }
        )
    return rows


def create_timestep_accumulators(
    n_latents: int,
    n_bins: int = 10,
) -> dict[int, dict[str, torch.Tensor]]:
    return {
        bin_idx: {
            "sums": torch.zeros(n_latents),
            "active_counts": torch.zeros(n_latents),
            "counts": torch.zeros(n_latents),
        }
        for bin_idx in range(n_bins)
    }


def accumulate_timestep_profiles(
    accumulators: dict[int, dict[str, torch.Tensor]],
    latents: torch.Tensor,
    record: ActivationRecord,
    start: int,
    end: int,
) -> None:
    values = latents.float()
    timesteps = record.timesteps[start:end].detach().cpu()
    n_bins = len(accumulators)
    bin_indices = torch.clamp((timesteps * n_bins).long(), min=0, max=n_bins - 1)
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if not bool(mask.any()):
            continue
        subset = values[mask]
        accumulators[bin_idx]["sums"] += subset.sum(dim=0)
        accumulators[bin_idx]["active_counts"] += (subset > 0).sum(dim=0)
        accumulators[bin_idx]["counts"] += subset.shape[0]


_accumulate_timestep_profiles = accumulate_timestep_profiles


def finalize_timestep_profiles(
    accumulators: dict[int, dict[str, torch.Tensor]] | None,
) -> dict[int, list[dict[str, float | int]]]:
    if accumulators is None:
        return {}
    profiles: dict[int, list[dict[str, float | int]]] = {}
    n_bins = len(accumulators)
    for bin_idx, values in accumulators.items():
        counts = values["counts"].clamp(min=1)
        mean_activation = values["sums"] / counts
        active_fraction = values["active_counts"] / counts
        lower = bin_idx / n_bins
        upper = (bin_idx + 1) / n_bins
        for feature_idx in range(mean_activation.numel()):
            profiles.setdefault(feature_idx, []).append(
                {
                    "timestep_bin": bin_idx,
                    "timestep_low": lower,
                    "timestep_high": upper,
                    "mean_activation": float(mean_activation[feature_idx].item()),
                    "active_fraction": float(active_fraction[feature_idx].item()),
                }
            )
    return profiles


def create_concept_accumulators(
    token_properties: list[TokenProperty],
    n_latents: int,
    thresholds: tuple[float, ...],
) -> dict[str, dict[str, Any]]:
    return {
        token_property.name: {
            "thresholds": thresholds,
            "tp": torch.zeros((n_latents, len(thresholds))),
            "fp": torch.zeros((n_latents, len(thresholds))),
            "positive_count": 0,
            "negative_count": 0,
        }
        for token_property in token_properties
    }


def _accumulate_concept_metrics(
    accumulator: dict[str, Any],
    latents: torch.Tensor,
    record: ActivationRecord,
    start: int,
    end: int,
    token_property: TokenProperty,
    active_by_threshold: tuple[torch.Tensor, ...] | None = None,
) -> None:
    labels = token_property_labels_for_record(token_property, record, start, end)
    if not labels:
        return
    values = latents.float()
    label_tensor = torch.tensor(labels, dtype=torch.int8)
    positive_mask = label_tensor > 0
    negative_mask = label_tensor < 0
    accumulator["positive_count"] += int(positive_mask.sum().item())
    accumulator["negative_count"] += int(negative_mask.sum().item())
    if not bool(positive_mask.any()) and not bool(negative_mask.any()):
        return
    for threshold_idx, threshold in enumerate(accumulator["thresholds"]):
        active = (
            active_by_threshold[threshold_idx]
            if active_by_threshold is not None
            else values > float(threshold)
        )
        if bool(positive_mask.any()):
            accumulator["tp"][:, threshold_idx] += active[positive_mask].sum(dim=0)
        if bool(negative_mask.any()):
            accumulator["fp"][:, threshold_idx] += active[negative_mask].sum(dim=0)


def token_property_labels_for_record(
    token_property: TokenProperty,
    record: ActivationRecord,
    start: int,
    end: int,
) -> list[int]:
    sequence_ids = record.sequence_ids[start:end]
    residues = record.residue_indices[start:end].detach().cpu().tolist()
    sample_indices = (
        record.sample_indices[start:end].detach().cpu().tolist()
        if record.sample_indices is not None
        else [None] * len(sequence_ids)
    )
    timestep_values = record.timesteps[start:end].detach().cpu().tolist()
    return [
        token_property_label(
            token_property,
            sequence_id=sequence_id,
            residue_idx=int(residues[row_idx]),
            sample_index=(
                int(sample_indices[row_idx]) if sample_indices[row_idx] is not None else None
            ),
            timestep=float(timestep_values[row_idx]),
        )
        for row_idx, sequence_id in enumerate(sequence_ids)
    ]


def finalize_concept_metrics(
    accumulators: dict[str, dict[str, Any]] | None,
) -> dict[str, list[dict[str, float | int]]]:
    if accumulators is None:
        return {}
    metrics_by_property = {}
    for property_name, accumulator in accumulators.items():
        thresholds = accumulator["thresholds"]
        tp = accumulator["tp"]
        fp = accumulator["fp"]
        positive_count = max(int(accumulator["positive_count"]), 1)
        rows: list[dict[str, float | int]] = []
        for feature_idx in range(tp.shape[0]):
            for threshold_idx, threshold in enumerate(thresholds):
                true_positive = float(tp[feature_idx, threshold_idx].item())
                false_positive = float(fp[feature_idx, threshold_idx].item())
                if true_positive == 0 and false_positive == 0:
                    continue
                precision = true_positive / max(true_positive + false_positive, 1.0)
                recall = true_positive / positive_count
                f1 = (
                    2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
                )
                rows.append(
                    {
                        "feature": feature_idx,
                        "threshold": float(threshold),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        "tp": int(true_positive),
                        "fp": int(false_positive),
                        "positive_tokens": int(accumulator["positive_count"]),
                        "negative_tokens": int(accumulator["negative_count"]),
                    }
                )
        metrics_by_property[property_name] = sorted(
            rows,
            key=lambda row: float(row["f1"]),
            reverse=True,
        )
    return metrics_by_property


def _accumulate_feature_group_stats(
    stats: dict[str, FeatureGroupStats],
    latents: torch.Tensor,
    record: ActivationRecord,
    start: int,
    end: int,
    token_property: TokenProperty,
) -> None:
    sequence_ids = record.sequence_ids[start:end]
    labels = torch.tensor(
        token_property_labels_for_record(token_property, record, start, end),
        dtype=torch.int8,
    )
    values = latents.float()
    for sequence_id in dict.fromkeys(sequence_ids):
        sequence_mask = torch.tensor(
            [candidate == sequence_id for candidate in sequence_ids],
            dtype=torch.bool,
        )
        sequence_stats = stats.get(sequence_id)
        if sequence_stats is None:
            n_latents = latents.shape[1]
            sequence_stats = FeatureGroupStats(
                high_sums=torch.zeros(n_latents),
                high_counts=torch.zeros(n_latents),
                low_sums=torch.zeros(n_latents),
                low_counts=torch.zeros(n_latents),
            )
            stats[sequence_id] = sequence_stats
        high_mask = torch.logical_and(sequence_mask, labels > 0)
        low_mask = torch.logical_and(sequence_mask, labels < 0)
        high_count = int(high_mask.sum())
        low_count = int(low_mask.sum())
        if high_count:
            sequence_stats.high_sums += values[high_mask].sum(dim=0)
            sequence_stats.high_counts += high_count
        if low_count:
            sequence_stats.low_sums += values[low_mask].sum(dim=0)
            sequence_stats.low_counts += low_count


def token_property_label(
    token_property: TokenProperty,
    sequence_id: str,
    residue_idx: int,
    sample_index: int | None,
    timestep: float,
) -> int:
    if token_property.labels_by_sequence is not None:
        labels = token_property.labels_by_sequence.get(sequence_id)
        if labels is None or residue_idx >= len(labels):
            return 0
        return int(labels[residue_idx])
    if token_property.label_fn is not None:
        return int(
            token_property.label_fn(
                {
                    "sequence_id": sequence_id,
                    "residue_index": residue_idx,
                    "sample_index": sample_index,
                    "timestep": timestep,
                }
            )
        )
    return 0


def continuous_property_value(
    continuous_property: ContinuousTokenProperty,
    sequence_id: str,
    residue_idx: int,
    sample_index: int | None,
    timestep: float,
) -> float:
    if continuous_property.values_by_sequence is not None:
        values = continuous_property.values_by_sequence.get(sequence_id)
        if values is None or residue_idx >= len(values):
            return float("nan")
        return float(values[residue_idx])
    if continuous_property.value_fn is not None:
        return float(
            continuous_property.value_fn(
                {
                    "sequence_id": sequence_id,
                    "residue_index": residue_idx,
                    "sample_index": sample_index,
                    "timestep": timestep,
                }
            )
        )
    return float("nan")


def create_prediction_accumulators(
    continuous_properties: list[ContinuousTokenProperty],
    n_latents: int,
) -> dict[str, FeaturePredictionAccumulator]:
    return {
        continuous_property.name: FeaturePredictionAccumulator.create(n_latents)
        for continuous_property in continuous_properties
    }


def _accumulate_prediction_metrics(
    accumulators: dict[str, FeaturePredictionAccumulator],
    reservoir: PredictionReservoir,
    latents: torch.Tensor,
    record: ActivationRecord,
    start: int,
    end: int,
    continuous_properties: list[ContinuousTokenProperty],
    row_key_source: str | None = None,
) -> None:
    sequence_ids = record.sequence_ids[start:end]
    residues = record.residue_indices[start:end].detach().cpu().tolist()
    sample_indices = (
        record.sample_indices[start:end].detach().cpu().tolist()
        if record.sample_indices is not None
        else [None] * len(sequence_ids)
    )
    timestep_values = record.timesteps[start:end].detach().cpu().tolist()
    values = latents.float()
    property_arrays = {}
    any_valid = np.zeros(len(sequence_ids), dtype=bool)
    for continuous_property in continuous_properties:
        y_values = np.asarray(
            [
                continuous_property_value(
                    continuous_property,
                    sequence_id=sequence_id,
                    residue_idx=int(residues[row_idx]),
                    sample_index=(
                        int(sample_indices[row_idx])
                        if sample_indices[row_idx] is not None
                        else None
                    ),
                    timestep=float(timestep_values[row_idx]),
                )
                for row_idx, sequence_id in enumerate(sequence_ids)
            ],
            dtype=np.float64,
        )
        valid = np.isfinite(y_values)
        if not bool(valid.any()):
            continue
        any_valid |= valid
        property_arrays[continuous_property.name] = y_values
        valid_tensor = torch.from_numpy(valid)
        y_tensor = torch.from_numpy(y_values[valid]).to(torch.float32)
        latent_values = values[valid_tensor]
        accumulator = accumulators[continuous_property.name]
        accumulator.count += int(valid.sum())
        accumulator.sum_y += float(y_tensor.sum())
        accumulator.sum_y2 += float((y_tensor * y_tensor).sum())
        accumulator.sum_x += latent_values.sum(dim=0)
        accumulator.sum_x2 += (latent_values * latent_values).sum(dim=0)
        accumulator.sum_xy += (latent_values * y_tensor[:, None]).sum(dim=0)
    if bool(any_valid.any()):
        retained_indices = np.flatnonzero(any_valid)
        reservoir.add_batch(
            values[torch.from_numpy(any_valid)],
            {name: array[any_valid] for name, array in property_arrays.items()},
            [sequence_ids[index] for index in retained_indices],
            row_keys=(
                [
                    PredictionRowKey(
                        sequence_id=sequence_ids[index],
                        residue_index=int(residues[index]),
                        sample_index=(
                            int(sample_indices[index])
                            if sample_indices[index] is not None
                            else None
                        ),
                        source=row_key_source,
                    )
                    for index in retained_indices
                ]
                if row_key_source is not None
                else None
            ),
        )


def _accumulate_sample_prediction_metrics(
    accumulator: SampleFeaturePredictionAccumulator,
    latents: torch.Tensor,
    record: ActivationRecord,
    start: int,
    end: int,
    continuous_properties: list[ContinuousTokenProperty],
    descending_timestep: bool = False,
) -> None:
    sequence_ids = record.sequence_ids[start:end]
    sample_indices = (
        record.sample_indices[start:end].detach().cpu().tolist()
        if record.sample_indices is not None
        else [None] * len(sequence_ids)
    )
    residues = record.residue_indices[start:end].detach().cpu().tolist()
    timestep_values = record.timesteps[start:end].detach().cpu().tolist()
    groups: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    property_values_by_key: dict[tuple[str, int, str], dict[str, float]] = {}
    for row_idx, sequence_id in enumerate(sequence_ids):
        sample_index = sample_indices[row_idx]
        if sample_index is None:
            continue
        row_keys = set()
        timestep = float(timestep_values[row_idx])
        for continuous_property in continuous_properties:
            if not timestep_matches_bin(
                timestep,
                continuous_property.timestep_bin,
                descending=descending_timestep,
            ):
                continue
            y_value = continuous_property_value(
                continuous_property,
                sequence_id=sequence_id,
                residue_idx=int(residues[row_idx]),
                sample_index=int(sample_index),
                timestep=timestep,
            )
            if not math.isfinite(y_value):
                continue
            bin_name = continuous_property.timestep_bin or "all"
            key = (sequence_id, int(sample_index), bin_name)
            property_values_by_key.setdefault(key, {})[continuous_property.name] = y_value
            row_keys.add(key)
        for key in row_keys:
            groups[key].append(row_idx)
    values = latents.float()
    for key, row_indices in groups.items():
        if not row_indices:
            continue
        aggregate = accumulator.aggregates.get(key)
        if aggregate is None:
            aggregate = SampleFeatureAggregate(
                count=0,
                sums=torch.zeros(accumulator.n_latents),
                property_values=property_values_by_key[key],
            )
            accumulator.aggregates[key] = aggregate
        group = values[row_indices]
        aggregate.count += int(group.shape[0])
        aggregate.sums += group.sum(dim=0)


def timestep_matches_bin(
    timestep: float,
    timestep_bin: str | None,
    descending: bool = False,
) -> bool:
    if timestep_bin is None:
        return True
    if timestep_bin == "early":
        return timestep >= LATE_TIMESTEP_MIN if descending else timestep <= EARLY_TIMESTEP_MAX
    if timestep_bin == "late":
        return timestep <= EARLY_TIMESTEP_MAX if descending else timestep >= LATE_TIMESTEP_MIN
    raise ValueError(f"Unknown timestep bin: {timestep_bin}")


def finalize_property_importance(
    accumulators: dict[str, FeaturePredictionAccumulator] | None,
    prediction_sample: PredictionSample | None,
    continuous_properties: list[ContinuousTokenProperty],
) -> dict[str, list[dict[str, float | int | str]]]:
    if accumulators is None:
        return {}
    property_info = {prop.name: prop for prop in continuous_properties}
    importance: dict[str, list[dict[str, float | int | str]]] = {}
    for property_name, accumulator in accumulators.items():
        rows = finalize_single_property_importance(
            property_name=property_name,
            accumulator=accumulator,
            prediction_sample=prediction_sample,
            units=property_info.get(
                property_name, ContinuousTokenProperty(property_name, property_name)
            ).units,
        )
        importance[property_name] = rows
    return importance


def finalize_sample_property_importance(
    accumulator: SampleFeaturePredictionAccumulator | None,
    continuous_properties: list[ContinuousTokenProperty],
    prediction_sample_size: int,
) -> tuple[
    dict[str, list[dict[str, float | int | str]]],
    dict[str, PredictionSample],
]:
    if accumulator is None or not continuous_properties:
        return {}, {}
    importance: dict[str, list[dict[str, float | int | str]]] = {}
    samples: dict[str, PredictionSample] = {}
    for continuous_property in continuous_properties:
        metric_accumulator = FeaturePredictionAccumulator.create(accumulator.n_latents)
        reservoir = PredictionReservoir(max_rows=prediction_sample_size)
        property_entries: list[tuple[str, float, torch.Tensor]] = []
        for (
            sequence_id,
            _sample_index,
            _bin_name,
        ), sample_aggregate in accumulator.aggregates.items():
            y_value = sample_aggregate.property_values.get(continuous_property.name)
            if y_value is None or not math.isfinite(y_value) or sample_aggregate.count == 0:
                continue
            property_entries.append(
                (sequence_id, y_value, sample_aggregate.sums / sample_aggregate.count)
            )
        if property_entries:
            feature_matrix = torch.stack([entry[2] for entry in property_entries])
            if continuous_property.normalize_features_within_sequence:
                feature_matrix = within_sequence_zscore_features(
                    [entry[0] for entry in property_entries],
                    feature_matrix,
                )
        else:
            feature_matrix = torch.empty((0, accumulator.n_latents))
        for (sequence_id, y_value, _raw_feature_values), feature_values in zip(
            property_entries,
            feature_matrix,
            strict=True,
        ):
            metric_accumulator.count += 1
            metric_accumulator.sum_y += y_value
            metric_accumulator.sum_y2 += y_value * y_value
            metric_accumulator.sum_x += feature_values
            metric_accumulator.sum_x2 += feature_values * feature_values
            metric_accumulator.sum_xy += feature_values * y_value
            reservoir.add(
                feature_values,
                {continuous_property.name: y_value},
                group_id=sequence_id,
            )
        sample = reservoir.finalize([continuous_property.name])
        if sample is not None:
            samples[continuous_property.name] = sample
        rows = finalize_single_property_importance(
            property_name=continuous_property.name,
            accumulator=metric_accumulator,
            prediction_sample=sample,
            units=continuous_property.units,
        )
        for row in rows:
            row["aggregation_level"] = "sample"
            row["feature_aggregation"] = "structure_mean"
            row["feature_normalization"] = (
                "within_sequence_zscore"
                if continuous_property.normalize_features_within_sequence
                else "none"
            )
            row["n_structures"] = row["n_tokens"]
            row["n_tokens"] = 0
        importance[continuous_property.name] = sorted(
            rows,
            key=lambda row: (
                float(row["importance_score"]),
                abs(float(row["pearson_r"])),
            ),
            reverse=True,
        )
    return importance, samples


def merge_prediction_samples(
    base: PredictionSample | None,
    samples_by_property: dict[str, PredictionSample],
) -> PredictionSample | None:
    if base is None and not samples_by_property:
        return None
    if base is None:
        first = next(iter(samples_by_property.values()))
        base = PredictionSample(
            latents=first.latents,
            property_values={},
            group_ids=first.group_ids,
            row_keys=first.row_keys,
            sampling_identity=first.sampling_identity,
        )
    for property_name, sample in samples_by_property.items():
        base.property_values[property_name] = sample.property_values[property_name]
        base.latents_by_property[property_name] = sample.latents
        if sample.group_ids is not None:
            base.group_ids_by_property[property_name] = sample.group_ids
        base.row_keys_by_property[property_name] = sample.keys_for(property_name)
        base.sampling_identity_by_property[property_name] = sample.sampling_for(property_name)
    return base


def finalize_single_property_importance(
    property_name: str,
    accumulator: FeaturePredictionAccumulator,
    prediction_sample: PredictionSample | None,
    units: str = "",
) -> list[dict[str, float | int | str]]:
    n = accumulator.count
    if n < 3:
        return []
    n_float = float(n)
    mean_y = accumulator.sum_y / n_float
    var_y = max(accumulator.sum_y2 - n_float * mean_y * mean_y, 0.0)
    mean_x = accumulator.sum_x / n_float
    var_x = torch.clamp(accumulator.sum_x2 - n_float * mean_x * mean_x, min=0.0)
    cov_xy = accumulator.sum_xy - accumulator.sum_x * mean_y
    denominator = torch.sqrt(var_x * var_y).clamp(min=1e-12)
    pearson = torch.clamp(cov_xy / denominator, min=-1.0, max=1.0)
    r2 = pearson * pearson

    spearman = np.full(pearson.numel(), np.nan, dtype=np.float64)
    auc_high = np.full(pearson.numel(), np.nan, dtype=np.float64)
    auc_low = np.full(pearson.numel(), np.nan, dtype=np.float64)
    cv_selection_count = np.zeros(pearson.numel(), dtype=np.int64)
    cv_train_spearman = np.full(pearson.numel(), np.nan, dtype=np.float64)
    cv_heldout_spearman = np.full(pearson.numel(), np.nan, dtype=np.float64)
    cv_folds = 0
    if prediction_sample is not None and property_name in prediction_sample.property_values:
        sample_y = prediction_sample.property_values[property_name]
        valid_mask = np.isfinite(sample_y)
        if int(valid_mask.sum()) >= 8:
            sample_x = prediction_sample.latents_for(property_name)[valid_mask]
            sample_y = sample_y[valid_mask]
            y_rank = scipy_stats.rankdata(sample_y).astype(np.float64)
            y_rank -= y_rank.mean()
            y_norm = float(np.linalg.norm(y_rank))
            high_cut = float(np.quantile(sample_y, 0.8))
            low_cut = float(np.quantile(sample_y, 0.2))
            high_labels = sample_y >= high_cut
            low_labels = sample_y <= low_cut
            for feature_idx in range(sample_x.shape[1]):
                x_values = sample_x[:, feature_idx].astype(np.float64, copy=False)
                if np.allclose(x_values, x_values[0]):
                    continue
                x_rank = scipy_stats.rankdata(x_values).astype(np.float64)
                x_rank -= x_rank.mean()
                x_norm = float(np.linalg.norm(x_rank))
                if x_norm > 0 and y_norm > 0:
                    spearman[feature_idx] = float(np.dot(x_rank, y_rank) / (x_norm * y_norm))
                auc_high[feature_idx] = rank_auc_from_scores(x_values, high_labels)
                auc_low[feature_idx] = rank_auc_from_scores(x_values, low_labels)
            groups = prediction_sample.groups_for(property_name)
            if groups is not None:
                (
                    cv_selection_count,
                    cv_train_spearman,
                    cv_heldout_spearman,
                    cv_folds,
                ) = nested_grouped_spearman_selection(
                    sample_x,
                    sample_y,
                    groups[valid_mask],
                    top_k_per_direction=3,
                    max_folds=5,
                )

    rows: list[dict[str, float | int | str]] = []
    for feature_idx in range(pearson.numel()):
        pearson_value = float(pearson[feature_idx].item())
        full_spearman_value = (
            float(spearman[feature_idx]) if math.isfinite(spearman[feature_idx]) else 0.0
        )
        has_cv_score = cv_selection_count[feature_idx] > 0 and math.isfinite(
            cv_heldout_spearman[feature_idx]
        )
        spearman_value = (
            float(cv_heldout_spearman[feature_idx]) if has_cv_score else full_spearman_value
        )
        r2_value = float(r2[feature_idx].item())
        importance_score = (
            abs(spearman_value) * cv_selection_count[feature_idx] / max(cv_folds, 1)
            if cv_folds > 0
            else abs(spearman_value)
        )
        rows.append(
            {
                "property": property_name,
                "feature": feature_idx,
                "n_tokens": n,
                "aggregation_level": "token",
                "feature_aggregation": "token",
                "units": units,
                "pearson_r": pearson_value,
                "spearman_r_sample": spearman_value,
                "full_data_spearman_r_sample": full_spearman_value,
                "cv_train_spearman_r": float(cv_train_spearman[feature_idx])
                if math.isfinite(cv_train_spearman[feature_idx])
                else 0.0,
                "cv_selection_count": int(cv_selection_count[feature_idx]),
                "cv_selection_fraction": float(cv_selection_count[feature_idx] / max(cv_folds, 1)),
                "evaluation_scope": (
                    "nested_five_fold_held_out_sequences"
                    if cv_folds > 0
                    else "exploratory_full_data_association"
                ),
                "r2_univariate": r2_value,
                "auc_high_property_sample": float(auc_high[feature_idx])
                if math.isfinite(auc_high[feature_idx])
                else 0.5,
                "auc_low_property_sample": float(auc_low[feature_idx])
                if math.isfinite(auc_low[feature_idx])
                else 0.5,
                "importance_score": float(importance_score),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            float(row["importance_score"]),
            abs(float(row["pearson_r"])),
        ),
        reverse=True,
    )


def rank_auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = scipy_stats.rankdata(scores).astype(np.float64)
    pos_rank_sum = float(ranks[labels].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def nested_grouped_spearman_selection(
    x_values: np.ndarray,
    y_values: np.ndarray,
    group_ids: np.ndarray,
    top_k_per_direction: int,
    max_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Select features on training groups and score only held-out groups."""

    unique_groups = sorted(
        {str(group) for group in group_ids},
        key=lambda value: hashlib.sha256(value.encode("utf-8")).digest(),
    )
    n_folds = min(max_folds, len(unique_groups))
    n_features = x_values.shape[1]
    selected_count = np.zeros(n_features, dtype=np.int64)
    train_sums = np.zeros(n_features, dtype=np.float64)
    test_sums = np.zeros(n_features, dtype=np.float64)
    if n_folds < 2:
        return selected_count, train_sums, test_sums, 0
    fold_by_group = {group: index % n_folds for index, group in enumerate(unique_groups)}
    row_folds = np.asarray([fold_by_group[str(group)] for group in group_ids], dtype=np.int16)
    for fold_idx in range(n_folds):
        train_mask = row_folds != fold_idx
        test_mask = row_folds == fold_idx
        if int(train_mask.sum()) < 8 or int(test_mask.sum()) < 3:
            continue
        train_correlations = vectorized_spearman(x_values[train_mask], y_values[train_mask])
        test_correlations = vectorized_spearman(x_values[test_mask], y_values[test_mask])
        positive = np.flatnonzero(train_correlations > 0)
        negative = np.flatnonzero(train_correlations < 0)
        positive = positive[np.argsort(train_correlations[positive])[-top_k_per_direction:]]
        negative = negative[np.argsort(train_correlations[negative])[:top_k_per_direction]]
        for feature_idx in np.concatenate([positive, negative]):
            test_value = test_correlations[feature_idx]
            if not math.isfinite(float(test_value)):
                continue
            selected_count[feature_idx] += 1
            train_sums[feature_idx] += train_correlations[feature_idx]
            test_sums[feature_idx] += test_value
    selected = selected_count > 0
    train_sums[selected] /= selected_count[selected]
    test_sums[selected] /= selected_count[selected]
    train_sums[~selected] = np.nan
    test_sums[~selected] = np.nan
    return selected_count, train_sums, test_sums, n_folds


def vectorized_spearman(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    if x_values.shape[0] != y_values.shape[0]:
        raise ValueError("x_values and y_values must contain the same rows")
    x_ranks = scipy_stats.rankdata(x_values, axis=0).astype(np.float64)
    y_ranks = scipy_stats.rankdata(y_values).astype(np.float64)
    x_ranks -= x_ranks.mean(axis=0, keepdims=True)
    y_ranks -= y_ranks.mean()
    denominator = np.linalg.norm(x_ranks, axis=0) * np.linalg.norm(y_ranks)
    correlations = np.divide(
        x_ranks.T @ y_ranks,
        denominator,
        out=np.full(x_values.shape[1], np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    return np.clip(correlations, -1.0, 1.0)


def _quantile_label(value: float, low: float, high: float) -> int:
    if value >= high:
        return 1
    if value <= low:
        return -1
    return 0


def _radius_label(
    token: dict[str, Any],
    radius_by_sample: dict[tuple[str, int], float],
    low: float,
    high: float,
) -> int:
    sample_index = token["sample_index"]
    if sample_index is None:
        return 0
    radius = radius_by_sample.get((token["sequence_id"], int(sample_index)))
    if radius is None:
        return 0
    if radius >= high:
        return 1
    if radius <= low:
        return -1
    return 0


def compute_volcano_rows(stats: dict[str, FeatureGroupStats]) -> list[dict[str, float | int]]:
    if not stats:
        return []
    n_latents = next(iter(stats.values())).high_sums.numel()
    rows: list[dict[str, float | int]] = []
    for feature_idx in range(n_latents):
        paired_effects = []
        high_total = 0
        low_total = 0
        for sequence_stats in stats.values():
            high_count = float(sequence_stats.high_counts[feature_idx].item())
            low_count = float(sequence_stats.low_counts[feature_idx].item())
            if high_count == 0 or low_count == 0:
                continue
            high_mean = float(sequence_stats.high_sums[feature_idx].item()) / high_count
            low_mean = float(sequence_stats.low_sums[feature_idx].item()) / low_count
            paired_effects.append(high_mean - low_mean)
            high_total += int(high_count)
            low_total += int(low_count)
        if not paired_effects:
            continue
        effects = np.asarray(paired_effects, dtype=np.float64)
        effect = float(effects.mean())
        effect_std = float(effects.std(ddof=1)) if effects.size > 1 else 0.0
        effect_sem = effect_std / math.sqrt(effects.size) if effects.size > 1 else 0.0
        t_statistic, p_value = paired_t_test_against_zero(effects)
        rows.append(
            {
                "feature": feature_idx,
                "effect_high_minus_low": effect,
                "effect_std": effect_std,
                "effect_sem": effect_sem,
                "t_statistic": t_statistic,
                "p_value": p_value,
                "neg_log10_p": float(-math.log10(max(p_value, 1e-300))),
                "num_sequence_replicates": int(effects.size),
                "high_token_count": high_total,
                "low_token_count": low_total,
            }
        )
    add_benjamini_hochberg_q_values(rows)
    return rows


def paired_t_test_against_zero(effects: np.ndarray) -> tuple[float, float]:
    """Two-sided one-sample t-test on paired per-sequence effects.

    Each value in ``effects`` is one sequence-level high-minus-low contrast for
    a feature. Testing this paired contrast avoids treating many residue tokens
    from the same sequence as independent biological replicates.
    """

    effects = np.asarray(effects, dtype=np.float64)
    effects = effects[np.isfinite(effects)]
    if effects.size < 2:
        return 0.0, 1.0
    if np.allclose(effects, effects[0]):
        if np.isclose(float(effects[0]), 0.0):
            return 0.0, 1.0
        return math.copysign(float("inf"), float(effects[0])), 0.0
    result = scipy_stats.ttest_1samp(effects, popmean=0.0, nan_policy="omit")
    t_statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not math.isfinite(t_statistic):
        t_statistic = 0.0
    if not math.isfinite(p_value):
        p_value = 1.0
    return t_statistic, min(max(p_value, 0.0), 1.0)


def add_benjamini_hochberg_q_values(rows: list[dict[str, float | int]]) -> None:
    """Annotate volcano rows with Benjamini-Hochberg FDR q-values in place."""

    if not rows:
        return
    p_values = np.asarray([float(row["p_value"]) for row in rows], dtype=np.float64)
    p_values = np.clip(p_values, 0.0, 1.0)
    q_values = benjamini_hochberg(p_values)
    for row, q_value in zip(rows, q_values, strict=True):
        q_value = float(q_value)
        row["q_value"] = q_value
        row["neg_log10_q"] = float(-math.log10(max(q_value, 1e-300)))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted q-values for a family of tests."""

    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.size == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(p_values)
    ranked = p_values[order]
    n_tests = ranked.size
    adjusted = ranked * n_tests / np.arange(1, n_tests + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_values = np.empty_like(adjusted)
    q_values[order] = adjusted
    return q_values


def write_volcano_plot(
    rows: list[dict[str, float | int]],
    path: Path,
    title: str,
    x_label: str,
    highlighted_features: set[int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_volcano_plot(
        ax,
        rows,
        title=title,
        x_label=x_label,
        highlighted_features=highlighted_features,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_volcano_grid_plot(
    plot_specs: list[tuple[TokenProperty, list[dict[str, float | int]], str, str, set[int] | None]],
    path: Path,
) -> None:
    if not plot_specs:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    n_plots = len(plot_specs)
    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.2 * n_cols, 4.6 * n_rows),
        squeeze=False,
    )
    for ax, (_token_property, rows, title, x_label, highlighted_features) in zip(
        axes.flat,
        plot_specs,
        strict=False,
    ):
        _draw_volcano_plot(
            ax,
            rows,
            title=title,
            x_label=x_label,
            label_top_k=3,
            highlighted_features=highlighted_features,
        )
    for ax in list(axes.flat)[n_plots:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def _draw_volcano_plot(
    ax: Any,
    rows: list[dict[str, float | int]],
    title: str,
    x_label: str,
    label_top_k: int = 5,
    highlighted_features: set[int] | None = None,
) -> None:
    if rows:
        highlighted_features = highlighted_features or set()
        x = [float(row["effect_high_minus_low"]) for row in rows]
        y_values = [
            min(float(row.get("neg_log10_q", row["neg_log10_p"])), VOLCANO_NEG_LOG10_CAP)
            for row in rows
        ]
        significant = [float(row.get("q_value", 1.0)) <= 0.05 for row in rows]
        colors = ["#b23b3b" if is_significant else "#3b6fb6" for is_significant in significant]
        ax.scatter(x, y_values, s=14, alpha=0.72, c=colors, edgecolors="none")
        highlighted_rows = [row for row in rows if int(row["feature"]) in highlighted_features]
        if highlighted_rows:
            ax.scatter(
                [float(row["effect_high_minus_low"]) for row in highlighted_rows],
                [
                    min(
                        float(row.get("neg_log10_q", row["neg_log10_p"])),
                        VOLCANO_NEG_LOG10_CAP,
                    )
                    for row in highlighted_rows
                ],
                s=42,
                facecolors="none",
                edgecolors="#111111",
                linewidths=0.9,
                zorder=3,
            )
        rows_to_label = highlighted_rows or select_top_volcano_label_rows(
            rows,
            top_k=label_top_k,
        )
        for row in rows_to_label:
            feature = int(row["feature"])
            effect = float(row["effect_high_minus_low"])
            neg_log_q = min(
                float(row.get("neg_log10_q", row["neg_log10_p"])),
                VOLCANO_NEG_LOG10_CAP,
            )
            ax.annotate(
                str(feature),
                xy=(effect, neg_log_q),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="#202020",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "#777777",
                    "linewidth": 0.4,
                    "alpha": 0.82,
                },
            )
        ax.axvline(0.0, color="#555555", linewidth=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"-log10(FDR q-value), capped at {VOLCANO_NEG_LOG10_CAP:g}")
    ax.set_title(title)


def select_top_volcano_label_rows(
    rows: list[dict[str, float | int]],
    top_k: int = 5,
) -> list[dict[str, float | int]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("neg_log10_q", row["neg_log10_p"])),
            abs(float(row["effect_high_minus_low"])),
        ),
        reverse=True,
    )[:top_k]


def select_top_features(rows: list[dict[str, float | int]], top_k: int) -> list[int]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row.get("neg_log10_q", row["neg_log10_p"])),
            abs(float(row["effect_high_minus_low"])),
        ),
        reverse=True,
    )
    return [int(row["feature"]) for row in ordered[:top_k]]


def ordered_selected_features(selected_features_by_property: dict[str, list[int]]) -> list[int]:
    """Return selected features in property-importance order without index sorting."""

    ordered = []
    seen = set()
    for feature_indices in selected_features_by_property.values():
        for feature_idx in feature_indices:
            if feature_idx not in seen:
                ordered.append(feature_idx)
                seen.add(feature_idx)
    return ordered


def merge_property_features_for_localization(
    property_features: dict[str, list[int]],
    concept_features: dict[str, list[int]],
) -> dict[str, list[int]]:
    """Include continuous-property and concept features in localization outputs."""

    return {
        **{name: list(features) for name, features in property_features.items()},
        **{name: list(features) for name, features in concept_features.items()},
    }


def run_analysis_rmsf_probe(
    prediction_sample: PredictionSample | None,
    output_dir: Path,
    *,
    top_features: int = 64,
    feature_counts: tuple[int, ...] | None = None,
    outer_folds: int = 5,
    inner_folds: int = 3,
    alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    fold_seed: int = 0,
    source_input_provenance: Mapping[str, object] | None,
) -> tuple[dict[str, Any], RMSFProbeResult | None]:
    """Fit the grouped RMSF probe from latents already sampled by analysis."""

    if source_input_provenance is None:
        report = {
            "status": "skipped",
            "reason": "RMSF prior augmentation unavailable: missing exact input provenance",
            "output_dir": str(output_dir),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_rmsf_json_atomic(output_dir / "summary.json", report)
        return report, None
    if prediction_sample is None or "rmsf" not in prediction_sample.property_values:
        report = {
            "status": "skipped",
            "reason": "no sampled RMSF labels",
            "output_dir": str(output_dir),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_rmsf_json_atomic(output_dir / "summary.json", report)
        return report, None
    groups = prediction_sample.groups_for("rmsf")
    if groups is None:
        report = {
            "status": "skipped",
            "reason": "no protein group IDs for RMSF rows",
            "output_dir": str(output_dir),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_rmsf_json_atomic(output_dir / "summary.json", report)
        return report, None
    latents = prediction_sample.latents_for("rmsf")
    rmsf = prediction_sample.property_values["rmsf"]
    row_keys = prediction_sample.keys_for("rmsf")
    if len(row_keys) != len(rmsf) or any(key is None for key in row_keys):
        raise ValueError("fresh RMSF source reports require complete durable row keys")
    durable_row_keys = tuple(key for key in row_keys if key is not None)
    configured_feature_counts = feature_counts or (top_features,)
    probe_config = {
        "feature_counts": list(configured_feature_counts),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "alphas": list(alphas),
        "fold_seed": fold_seed,
    }
    sweep_rows: list[dict[str, Any]] | None = None
    if feature_counts is None:
        result = run_grouped_rmsf_probe(
            latents,
            rmsf,
            groups,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            top_features=top_features,
            alphas=alphas,
            fold_seed=fold_seed,
        )
    else:
        if top_features not in feature_counts:
            raise ValueError("feature_counts must include top_features")
        results = run_grouped_rmsf_probe_sweep(
            latents,
            rmsf,
            groups,
            feature_counts=feature_counts,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            alphas=alphas,
            fold_seed=fold_seed,
        )
        sweep_report = write_rmsf_probe_sweep_artifacts(results, output_dir)
        sweep_rows = sweep_report["summary"]
        result = results[top_features]
    report = write_rmsf_probe_artifacts(
        result,
        rmsf,
        groups,
        output_dir,
        row_keys=durable_row_keys,
        probe_config=probe_config,
        sampling_identity=prediction_sample.sampling_for("rmsf"),
        source_latent_width=int(np.asarray(latents).shape[1]),
        input_provenance=source_input_provenance,
        feature_counts=feature_counts,
        sparsity_sweep=sweep_rows,
    )
    return report, result


def residue_probe_predictions_from_record(
    model: RidgeProbeModel,
    latents: np.ndarray,
    residue_indices: np.ndarray,
    length: int,
) -> np.ndarray:
    """Average a probe's token predictions into one value per residue."""

    token_predictions = predict_ridge_model(model, latents)
    residues = np.asarray(residue_indices, dtype=np.int64)
    valid = (residues >= 0) & (residues < length) & np.isfinite(token_predictions)
    sums = np.bincount(residues[valid], weights=token_predictions[valid], minlength=length).astype(
        np.float64
    )
    counts = np.bincount(residues[valid], minlength=length)
    return np.divide(
        sums,
        counts,
        out=np.full(length, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def probe_predictions_for_record(
    trainer: SAETrainer,
    record: ActivationRecord,
    model: RidgeProbeModel,
    sequence_id: str,
    length: int,
) -> np.ndarray:
    """Encode one targeted record and return residue-averaged probe predictions."""

    latent_batches = []
    residue_batches = []
    trainer.model.eval()
    with torch.no_grad():
        for start in range(0, record.n_tokens, trainer.config.training.batch_size):
            end = min(start + trainer.config.training.batch_size, record.n_tokens)
            row_mask = np.asarray(record.sequence_ids[start:end], dtype=object) == sequence_id
            if not bool(row_mask.any()):
                continue
            activations = record.activations[start:end].to(trainer.device)
            timesteps = record.timesteps[start:end].to(trainer.device)
            with trainer.autocast_context():
                latents = trainer.model(activations, timesteps).latents.detach().cpu().numpy()
            latent_batches.append(latents[row_mask])
            residue_batches.append(
                record.residue_indices[start:end].detach().cpu().numpy()[row_mask]
            )
    if not latent_batches:
        return np.full(length, np.nan, dtype=np.float64)
    return residue_probe_predictions_from_record(
        model,
        np.concatenate(latent_batches, axis=0),
        np.concatenate(residue_batches, axis=0),
        length,
    )


def write_analysis_rmsf_structure_views(
    result: RMSFProbeResult | None,
    trainer: SAETrainer,
    config: ProjectConfig,
    activation_entries: list[StructureIndexEntry],
    rmsf_reference: RMSFReference,
    output_dir: Path,
    *,
    store: ActivationStore | None = None,
    runtime_sampler: BioEmuRuntimeActivationSampler | None = None,
    record_provider: Any | None = None,
    runtime_sampler_factory: Any | None = None,
) -> list[dict[str, Any]]:
    """Render one held-out protein's true/predicted/error RMSF view per fold."""

    if result is None or result.summary.get("status") != "ok":
        return []
    activation_by_sequence: dict[str, list[StructureIndexEntry]] = defaultdict(list)
    reference_by_sequence: dict[str, list[StructureIndexEntry]] = defaultdict(list)
    for entry in activation_entries:
        activation_by_sequence[entry.sequence_id].append(entry)
    for entry in rmsf_reference.entries:
        if entry.pdb_path is not None:
            reference_by_sequence[entry.sequence_id].append(entry)
    if store is None and record_provider is None and runtime_sampler is None:
        sampler_factory = runtime_sampler_factory or BioEmuRuntimeActivationSampler
        runtime_sampler = sampler_factory(
            config=config.trace,
            cache_embeds_dir=config.training.cache_embeds_dir,
            structures_dir=None,
            batch_size=config.training.runtime_sequence_batch_size,
            device=trainer.device,
        )

    reports = []
    record_cache: dict[tuple[str, int | None], ActivationRecord | None] = {}
    for model in result.models:
        candidates = [
            sequence_id
            for sequence_id in model.heldout_groups
            if sequence_id in rmsf_reference.values_by_sequence
            and activation_by_sequence.get(sequence_id)
            and reference_by_sequence.get(sequence_id)
        ]
        candidates.sort(key=lambda value: hashlib.sha256(value.encode()).digest())
        if not candidates:
            reports.append(
                {
                    "fold": model.fold,
                    "status": "skipped",
                    "reason": "no held-out protein has both activations and a BioEmu PDB",
                }
            )
            continue
        sequence_id = candidates[0]
        activation_entry = min(
            activation_by_sequence[sequence_id], key=lambda entry: entry.sample_index
        )
        reference_entry = min(
            reference_by_sequence[sequence_id], key=lambda entry: entry.sample_index
        )
        candidate = FeatureExampleCandidate(
            activation=0.0,
            sequence_id=sequence_id,
            residue_index=0,
            timestep=0.0,
            sample_index=activation_entry.sample_index,
        )
        record = find_record_for_candidate(
            config=config,
            structure_entries=activation_entries,
            store=store,
            candidate=candidate,
            device=trainer.device,
            runtime_sampler=runtime_sampler,
            record_provider=record_provider,
            record_cache=record_cache,
        )
        if record is None:
            reports.append(
                {
                    "fold": model.fold,
                    "sequence_id": sequence_id,
                    "status": "skipped",
                    "reason": "targeted activation record is unavailable",
                }
            )
            continue
        true_rmsf = rmsf_reference.values_by_sequence[sequence_id]
        predicted_rmsf = probe_predictions_for_record(
            trainer,
            record,
            model,
            sequence_id,
            len(true_rmsf),
        )
        source_pdb = resolve_structure_path(rmsf_reference.root, reference_entry.pdb_path)
        if not source_pdb.is_file():
            reports.append(
                {
                    "fold": model.fold,
                    "sequence_id": sequence_id,
                    "status": "skipped",
                    "reason": f"BioEmu reference PDB does not exist: {source_pdb}",
                }
            )
            continue
        view_report = write_rmsf_structure_artifacts(
            source_pdb,
            sequence_id,
            true_rmsf,
            predicted_rmsf,
            output_dir / f"fold_{model.fold:02d}_{_safe_name(sequence_id)}",
            render_rmsf_probe_pdb,
        )
        reports.append({"fold": model.fold, **view_report})
    return reports


def select_concept_features_from_report(
    concept_report: dict[str, Any] | None,
    concept_prefix: str,
    max_concepts: int,
) -> dict[str, list[int]]:
    selected = {}
    for row in selected_concept_rows_from_report(concept_report, max_concepts=max_concepts):
        concept = str(row.get("concept", "unknown"))
        feature = row.get("refit_feature", row.get("feature"))
        if feature is None:
            continue
        selected[f"{concept_prefix}:{concept}"] = [int(feature)]
    return selected


def concept_feature_f1_scores(
    selected_concept_rows: list[dict[str, Any]],
    concept_prefix: str,
) -> dict[tuple[str, int], float]:
    scores = {}
    for row in selected_concept_rows:
        concept = str(row.get("concept", "unknown"))
        feature = row.get("refit_feature", row.get("feature"))
        if feature is None:
            continue
        scores[(f"{concept_prefix}:{concept}", int(feature))] = float(row["f1_per_domain"])
    return scores


def feature_examples_by_feature_from_residue_accumulator(
    accumulator: ResidueFeatureActivationAccumulator | None,
    feature_indices: list[int],
    top_k_examples: int,
) -> dict[int, list[FeatureExampleCandidate]]:
    if accumulator is None or accumulator.dense_max_by_feature is None or top_k_examples <= 0:
        return {}
    sequence_ids = list(accumulator.sequences)
    starts = np.asarray(
        [accumulator.offset_by_sequence[sequence_id] for sequence_id in sequence_ids],
        dtype=np.int64,
    )
    sample_argmax = accumulator.dense_argmax_sample_by_feature
    timestep_argmax = accumulator.dense_argmax_timestep_by_feature
    examples: dict[int, list[FeatureExampleCandidate]] = {}
    for feature_idx in feature_indices:
        values = accumulator.dense_max_by_feature[feature_idx].astype(np.float32, copy=False)
        active = np.flatnonzero(values > 0)
        if active.size == 0:
            examples[feature_idx] = []
            continue
        keep = min(top_k_examples, active.size)
        if active.size > keep:
            selected = active[np.argpartition(values[active], -keep)[-keep:]]
        else:
            selected = active
        selected = selected[np.argsort(values[selected])[::-1]]
        feature_examples = []
        for global_residue in selected.tolist():
            sequence_position = int(np.searchsorted(starts, global_residue, side="right") - 1)
            sequence_id = sequence_ids[sequence_position]
            residue_index = global_residue - int(starts[sequence_position])
            sample_index = (
                int(sample_argmax[feature_idx, global_residue])
                if sample_argmax is not None
                else None
            )
            timestep = (
                float(timestep_argmax[feature_idx, global_residue]) / 255.0
                if timestep_argmax is not None
                else 0.5
            )
            feature_examples.append(
                FeatureExampleCandidate(
                    activation=float(values[global_residue]),
                    sequence_id=sequence_id,
                    residue_index=residue_index,
                    timestep=timestep,
                    sample_index=sample_index,
                )
            )
        examples[feature_idx] = feature_examples
    return examples


def selected_concept_rows_from_report(
    concept_report: dict[str, Any] | None,
    max_concepts: int,
) -> list[dict[str, Any]]:
    if not concept_report:
        return []
    path_text = concept_report.get("top_domain_f1_json_path")
    if not path_text:
        return []
    path = Path(str(path_text))
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    selected = []
    seen_concepts = set()
    for row in rows:
        concept = str(row.get("concept", ""))
        if not concept or concept in seen_concepts:
            continue
        if float(row.get("f1_per_domain", 0.0)) <= 0:
            continue
        selected.append(row)
        seen_concepts.add(concept)
        if len(selected) >= max_concepts:
            break
    return selected


def feature_property_labels_from_selection(
    selected_features_by_property: dict[str, list[int]],
) -> dict[int, str]:
    properties_by_feature: dict[int, list[str]] = defaultdict(list)
    for property_name, feature_indices in selected_features_by_property.items():
        for feature_idx in feature_indices:
            properties_by_feature[int(feature_idx)].append(property_name)
    return {
        feature_idx: ", ".join(property_names)
        for feature_idx, property_names in properties_by_feature.items()
    }


def selected_timestep_features_by_property(
    property_features: dict[str, list[int]],
    concept_features: dict[str, list[int]],
) -> dict[str, list[int]]:
    return merge_property_features_for_localization(property_features, concept_features)


def human_readable_property_label(property_label: str) -> str:
    labels = []
    for label in property_label.split(", "):
        if label.startswith("interpro:") and label.count(":") >= 3:
            label = label.split(":", 3)[-1]
        labels.append(label)
    return ", ".join(labels)


def highlighted_features_for_token_property(
    token_property_name: str,
    selected_features_by_property: dict[str, list[int]],
    fallback_features: set[int],
    max_labels: int = 12,
) -> set[int]:
    token_to_continuous = {
        "high_rmsf": "rmsf",
        "early_vs_late_timestep": "diffusion_timestep",
        "expanded_final_structure": "radius_of_gyration",
        "dense_contact_structure": "contact_density",
    }
    property_name = token_to_continuous.get(token_property_name)
    if property_name is not None and property_name in selected_features_by_property:
        return set(selected_features_by_property[property_name][:max_labels])
    return set(sorted(fallback_features)[:max_labels])


def property_highlights_from_importance(
    property_importance: dict[str, list[dict[str, float | int | str]]],
) -> dict[str, list[int]]:
    highlights = {}
    for property_name, rows in property_importance.items():
        selected = [
            int(row["feature"])
            for row in top_positive_negative_importance_rows(rows, 3)
            if float(row.get("importance_score", 0.0)) > 0
        ]
        if selected:
            highlights[property_name] = selected
    return highlights


def iter_bioemu_runtime_activation_records(
    config: ProjectConfig,
    structure_entries: list[StructureIndexEntry],
    device: torch.device,
) -> Any:
    sampler = BioEmuRuntimeActivationSampler(
        config=config.trace,
        cache_embeds_dir=config.training.cache_embeds_dir,
        structures_dir=None,
        batch_size=config.training.runtime_sequence_batch_size,
        device=device,
    )
    for entry in structure_entries:
        yield sampler.sample(
            [SequenceRecord(sequence_id=entry.sequence_id, sequence=entry.sequence)],
            seed=entry.sample_index,
        )


def write_feature_summary_artifacts(
    feature_summary: list[dict[str, float | int]],
    timestep_profiles: dict[int, list[dict[str, float | int]]],
    output_dir: Path,
    *,
    timestep_axis_label: str = BIOEMU_TIMESTEP_AXIS_LABEL,
) -> dict[str, str | int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "feature_summary.json"
    summary_csv = output_dir / "feature_summary.csv"
    with summary_json.open("w") as handle:
        json.dump(feature_summary, handle, indent=2)
    if feature_summary:
        with summary_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(feature_summary[0]))
            writer.writeheader()
            writer.writerows(feature_summary)
        write_feature_summary_plots(feature_summary, output_dir)

    timestep_csv = output_dir / "feature_timestep_profiles.csv"
    timestep_rows = []
    for feature_idx, rows in timestep_profiles.items():
        for row in rows:
            timestep_rows.append({"feature": feature_idx, **row})
    if timestep_rows:
        with timestep_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(timestep_rows[0]))
            writer.writeheader()
            writer.writerows(timestep_rows)
        write_timestep_profile_heatmap(
            timestep_rows,
            output_dir / "feature_timestep_profiles.png",
            timestep_axis_label=timestep_axis_label,
        )

    return {
        "num_features": len(feature_summary),
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "timestep_profile_csv": str(timestep_csv),
        "plots_dir": str(output_dir),
    }


def write_feature_summary_plots(
    feature_summary: list[dict[str, float | int]],
    output_dir: Path,
) -> None:
    active_token = np.asarray(
        [row["active_token_fraction"] for row in feature_summary], dtype=float
    )
    active_sequence = np.asarray(
        [row["active_sequence_fraction"] for row in feature_summary],
        dtype=float,
    )
    active_sample = np.asarray(
        [row["active_sample_fraction"] for row in feature_summary],
        dtype=float,
    )
    max_activation = np.asarray([row["max_activation"] for row in feature_summary], dtype=float)
    mean_activation = np.asarray([row["mean_activation"] for row in feature_summary], dtype=float)

    write_histogram(
        active_token,
        output_dir / "active_token_fraction_hist.png",
        "Feature Active Token Fraction",
        "active token fraction",
    )
    write_histogram(
        max_activation,
        output_dir / "max_activation_hist.png",
        "Feature Max Activation",
        "max activation",
    )
    write_scatter_plot(
        active_token,
        max_activation,
        output_dir / "active_fraction_vs_max_activation.png",
        "Feature Frequency vs Max Activation",
        "active token fraction",
        "max activation",
    )
    write_scatter_plot(
        active_sequence,
        active_sample,
        output_dir / "active_sequence_vs_sample_fraction.png",
        "Feature Sequence/Sample Coverage",
        "active sequence fraction",
        "active sample fraction",
    )
    write_scatter_plot(
        active_token,
        mean_activation,
        output_dir / "active_fraction_vs_mean_activation.png",
        "Feature Frequency vs Mean Activation",
        "active token fraction",
        "mean activation",
    )


def write_histogram(values: np.ndarray, path: Path, title: str, x_label: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values[np.isfinite(values)], bins=60, color="#3b6fb6", alpha=0.82)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("features")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_scatter_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    ax.scatter(x_values[mask], y_values[mask], s=10, alpha=0.55, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_dict_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write plot source rows with a stable union of observed columns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_timestep_profile_heatmap(
    rows: list[dict[str, float | int]],
    path: Path,
    top_k: int = 100,
    *,
    timestep_axis_label: str = BIOEMU_TIMESTEP_AXIS_LABEL,
) -> None:
    if not rows:
        return
    by_feature: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for row in rows:
        by_feature[int(row["feature"])].append(row)
    ranked_features = sorted(
        by_feature,
        key=lambda feature: max(float(row["mean_activation"]) for row in by_feature[feature]),
        reverse=True,
    )[:top_k]
    n_bins = max(int(row["timestep_bin"]) for row in rows) + 1
    matrix = np.zeros((len(ranked_features), n_bins), dtype=float)
    for feature_row, feature_idx in enumerate(ranked_features):
        for row in by_feature[feature_idx]:
            matrix[feature_row, int(row["timestep_bin"])] = float(row["mean_activation"])
    row_max = np.max(matrix, axis=1, keepdims=True)
    normalized_matrix = np.divide(
        matrix,
        row_max,
        out=np.zeros_like(matrix),
        where=row_max > 0,
    )
    fig, ax = plt.subplots(figsize=(8, max(4, len(ranked_features) * 0.08)))
    im = ax.imshow(normalized_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel(timestep_axis_label)
    ax.set_ylabel("feature rank")
    ax.set_title("Top Feature Mean Activation by Timestep (row-normalized)")
    fig.colorbar(im, ax=ax, label="mean activation / feature max")
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_selected_feature_timestep_profile_plots(
    timestep_profiles: dict[int, list[dict[str, float | int]]],
    selected_features_by_property: dict[str, list[int]],
    output_dir: Path,
    *,
    timestep_axis_label: str = BIOEMU_TIMESTEP_AXIS_LABEL,
) -> list[dict[str, Any]]:
    if not selected_features_by_property:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_labels = feature_property_labels_from_selection(selected_features_by_property)
    selected_features = ordered_selected_features(selected_features_by_property)
    rows = []
    reports = []
    plot_specs = []
    for feature_idx in selected_features:
        profile_rows = sorted(
            timestep_profiles.get(feature_idx, []),
            key=lambda row: int(row["timestep_bin"]),
        )
        if not profile_rows:
            continue
        label = feature_labels.get(feature_idx, "")
        for row in profile_rows:
            rows.append(
                {
                    "feature": feature_idx,
                    "selected_for_property": label,
                    **row,
                }
            )
        plot_path = output_dir / f"feature_{feature_idx:04d}_timestep_profile.png"
        write_single_feature_timestep_profile_plot(
            feature_idx=feature_idx,
            property_label=label,
            rows=profile_rows,
            path=plot_path,
            timestep_axis_label=timestep_axis_label,
        )
        reports.append(
            {
                "feature": feature_idx,
                "selected_for_property": label,
                "plot_path": str(plot_path),
            }
        )
        plot_specs.append((feature_idx, label, profile_rows))
    if not rows:
        return []
    csv_path = output_dir / "selected_feature_timestep_profiles.csv"
    json_path = output_dir / "selected_feature_timestep_profiles.json"
    grid_path = output_dir / "selected_feature_timestep_profiles.png"
    write_dict_rows_csv(csv_path, rows)
    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2)
    if plot_specs:
        write_selected_feature_timestep_profile_grid(
            plot_specs,
            grid_path,
            timestep_axis_label=timestep_axis_label,
        )
    reports.append(
        {
            "plot_type": "selected_feature_timestep_profile_grid",
            "plot_path": str(grid_path) if plot_specs else None,
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "num_features": len(plot_specs),
        }
    )
    if plot_specs:
        overlay_path = output_dir / "selected_feature_timestep_profiles_overlay.png"
        write_selected_feature_timestep_profile_overlay(
            specs=plot_specs,
            path=overlay_path,
            title="Selected Feature Activation Across Timesteps",
            timestep_axis_label=timestep_axis_label,
        )
        reports.append(
            {
                "plot_type": "selected_feature_timestep_profile_overlay",
                "plot_path": str(overlay_path),
                "num_features": len(plot_specs),
                "normalization": "each feature divided by its maximum mean activation",
                "color_by": f"{timestep_axis_label} of maximum activation",
            }
        )
        specs_by_feature = {spec[0]: spec for spec in plot_specs}
        property_overlay_dir = output_dir / "property_overlays"
        property_overlay_dir.mkdir(parents=True, exist_ok=True)
        for property_name, feature_indices in selected_features_by_property.items():
            property_specs = [
                specs_by_feature[feature_idx]
                for feature_idx in dict.fromkeys(feature_indices)
                if feature_idx in specs_by_feature
            ]
            if not property_specs:
                continue
            property_path = property_overlay_dir / (
                f"{_safe_name(property_name)}_timestep_profiles_overlay.png"
            )
            write_selected_feature_timestep_profile_overlay(
                specs=property_specs,
                path=property_path,
                title=(
                    f"{human_readable_property_label(property_name)} Selected Feature Activation"
                ),
                timestep_axis_label=timestep_axis_label,
            )
            reports.append(
                {
                    "plot_type": "selected_feature_timestep_profile_property_overlay",
                    "property": property_name,
                    "plot_path": str(property_path),
                    "num_features": len(property_specs),
                    "normalization": "each feature divided by its maximum mean activation",
                    "color_by": f"{timestep_axis_label} of maximum activation",
                }
            )
    return reports


def write_single_feature_timestep_profile_plot(
    feature_idx: int,
    property_label: str,
    rows: list[dict[str, float | int]],
    path: Path,
    *,
    timestep_axis_label: str = BIOEMU_TIMESTEP_AXIS_LABEL,
) -> None:
    x_values = np.asarray(
        [0.5 * (float(row["timestep_low"]) + float(row["timestep_high"])) for row in rows],
        dtype=np.float64,
    )
    mean_activation = np.asarray([float(row["mean_activation"]) for row in rows])
    active_fraction = np.asarray([float(row["active_fraction"]) for row in rows])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(x_values, mean_activation, marker="o", color="#3b6fb6", label="mean activation")
    ax.set_xlabel(timestep_axis_label)
    ax.set_ylabel("mean activation")
    ax.grid(True, alpha=0.25)
    twin = ax.twinx()
    twin.plot(
        x_values,
        active_fraction,
        marker="s",
        linestyle="--",
        color="#b26b3b",
        label="active fraction",
    )
    twin.set_ylabel("active fraction")
    title = f"Feature {feature_idx} Activation Across Timesteps"
    if property_label:
        title = f"{title}\n{human_readable_property_label(property_label)}"
    ax.set_title(title, fontsize=10)
    lines, labels = ax.get_legend_handles_labels()
    twin_lines, twin_labels = twin.get_legend_handles_labels()
    ax.legend(lines + twin_lines, labels + twin_labels, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def normalized_selected_timestep_profile_arrays(
    rows: list[dict[str, float | int]],
) -> tuple[np.ndarray, np.ndarray, float]:
    if not rows:
        return np.asarray([]), np.asarray([]), 0.0
    x_values = np.asarray(
        [0.5 * (float(row["timestep_low"]) + float(row["timestep_high"])) for row in rows],
        dtype=np.float64,
    )
    mean_activation = np.asarray(
        [float(row["mean_activation"]) for row in rows],
        dtype=np.float64,
    )
    feature_max = float(mean_activation.max())
    normalized = np.divide(
        mean_activation,
        feature_max,
        out=np.zeros_like(mean_activation),
        where=feature_max > 0,
    )
    return x_values, normalized, float(x_values[int(np.argmax(normalized))])


def write_selected_feature_timestep_profile_overlay(
    specs: list[tuple[int, str, list[dict[str, float | int]]]],
    path: Path,
    title: str,
    *,
    timestep_axis_label: str = BIOEMU_TIMESTEP_AXIS_LABEL,
) -> None:
    normalizer = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = colormaps["rainbow_r"]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for _feature_idx, _property_label, rows in specs:
        x_values, normalized, peak = normalized_selected_timestep_profile_arrays(rows)
        if x_values.size == 0:
            continue
        ax.plot(
            x_values,
            normalized,
            color=cmap(normalizer(peak)),
            alpha=0.55,
            linewidth=1.2,
        )
    ax.set_xlabel(timestep_axis_label)
    ax.set_ylabel("mean activation / feature maximum")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    fig.colorbar(
        cm.ScalarMappable(norm=normalizer, cmap=cmap),
        ax=ax,
        label=f"{timestep_axis_label} of maximum activation",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def regenerate_selected_timestep_overlay(analysis_dir: Path) -> str:
    profile_dir = analysis_dir / "selected_feature_timestep_profiles"
    overlay_path = profile_dir / "selected_feature_timestep_profiles_overlay.png"
    if overlay_path.is_file():
        return "existing"
    json_path = profile_dir / "selected_feature_timestep_profiles.json"
    if not json_path.is_file():
        return "missing"
    rows = json.loads(json_path.read_text())
    if not isinstance(rows, list) or not rows:
        return "missing"
    rows_by_feature: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    labels_by_feature: dict[int, str] = {}
    feature_order = []
    for row in rows:
        feature_idx = int(row["feature"])
        if feature_idx not in rows_by_feature:
            feature_order.append(feature_idx)
        rows_by_feature[feature_idx].append(row)
        labels_by_feature[feature_idx] = str(row.get("selected_for_property", ""))
    specs = [
        (
            feature_idx,
            labels_by_feature[feature_idx],
            sorted(rows_by_feature[feature_idx], key=lambda row: int(row["timestep_bin"])),
        )
        for feature_idx in feature_order
    ]
    write_selected_feature_timestep_profile_overlay(
        specs,
        overlay_path,
        "Selected Feature Activation Across Timesteps",
    )
    return "generated"


def write_selected_feature_timestep_profile_grid(
    specs: list[tuple[int, str, list[dict[str, float | int]]]],
    path: Path,
    *,
    timestep_axis_label: str = BIOEMU_TIMESTEP_AXIS_LABEL,
) -> None:
    n_plots = len(specs)
    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 3.4 * n_rows),
        squeeze=False,
    )
    for ax, (feature_idx, property_label, rows) in zip(axes.flat, specs, strict=False):
        x_values = np.asarray(
            [0.5 * (float(row["timestep_low"]) + float(row["timestep_high"])) for row in rows],
            dtype=np.float64,
        )
        mean_activation = np.asarray([float(row["mean_activation"]) for row in rows])
        ax.plot(x_values, mean_activation, marker="o", color="#3b6fb6")
        ax.set_title(
            f"feature {feature_idx}\n{human_readable_property_label(property_label)}",
            fontsize=8,
        )
        ax.set_xlabel(timestep_axis_label)
        ax.set_ylabel("mean activation")
        ax.grid(True, alpha=0.22)
    for ax in list(axes.flat)[n_plots:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_concept_metric_tables(
    concept_metrics: dict[str, list[dict[str, float | int]]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for property_name, rows in concept_metrics.items():
        csv_path = output_dir / f"{property_name}_threshold_metrics.csv"
        json_path = output_dir / f"{property_name}_threshold_metrics.json"
        with json_path.open("w") as handle:
            json.dump(rows, handle, indent=2)
        if rows:
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            write_concept_f1_plot(rows, output_dir / f"{property_name}_top_f1.png", property_name)
        reports.append(
            {
                "property": property_name,
                "rows": len(rows),
                "csv_path": str(csv_path),
                "json_path": str(json_path),
            }
        )
    return reports


def write_property_importance_artifacts(
    property_importance: dict[str, list[dict[str, float | int | str]]],
    prediction_sample: PredictionSample | None,
    timestep_profiles: dict[int, list[dict[str, float | int]]],
    output_dir: Path,
    top_k_features: int,
    *,
    timestep_axis_label: str = BIOEMU_TIMESTEP_AXIS_LABEL,
    structure_feature_summaries: dict[tuple[str, int], np.ndarray] | None = None,
    structure_properties: dict[tuple[str, int], dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    combined_specs = []
    timestep_combined_specs = []
    all_rows = [row for rows in property_importance.values() for row in rows]
    if all_rows:
        combined_csv_path = output_dir / "property_feature_importance.csv"
        combined_json_path = output_dir / "property_feature_importance.json"
        with combined_json_path.open("w") as handle:
            json.dump(all_rows, handle, indent=2)
        write_dict_rows_csv(combined_csv_path, all_rows)
    for property_name, rows in property_importance.items():
        property_dir = property_specific_output_dir(output_dir, property_name)
        property_dir.mkdir(parents=True, exist_ok=True)
        csv_path = property_dir / "feature_importance.csv"
        json_path = property_dir / "feature_importance.json"
        with json_path.open("w") as handle:
            json.dump(rows, handle, indent=2)
        if rows:
            write_dict_rows_csv(csv_path, rows)
            ranking_path = property_dir / "top_predictive_features.png"
            write_property_importance_bar_plot(
                rows,
                ranking_path,
                property_name,
                top_k_features=min(3, top_k_features),
            )
            if is_timestep_conditioned_property(property_name):
                timestep_combined_specs.append((property_name, rows))
            else:
                combined_specs.append((property_name, rows))
            scatter_reports = write_property_prediction_scatter_plots(
                property_name=property_name,
                rows=rows,
                prediction_sample=prediction_sample,
                output_dir=property_dir,
                top_k_features=min(3, top_k_features),
            )
            timestep_profile_reports = write_property_timestep_profile_artifacts(
                property_name=property_name,
                rows=rows,
                timestep_profiles=timestep_profiles,
                output_dir=property_dir / "timestep_profiles",
                top_k_features=min(3, top_k_features),
                timestep_axis_label=timestep_axis_label,
            )
        else:
            ranking_path = None
            scatter_reports = []
            timestep_profile_reports = []
        reports.append(
            {
                "property": property_name,
                "rows": len(rows),
                "csv_path": str(csv_path),
                "json_path": str(json_path),
                "ranking_plot_path": str(ranking_path) if ranking_path is not None else None,
                "prediction_plots": scatter_reports,
                "timestep_profile_plots": timestep_profile_reports,
            }
        )
    if combined_specs:
        write_property_importance_grid_plot(
            combined_specs,
            output_dir / "all_property_predictive_features.png",
            top_k_features=min(3, top_k_features),
        )
    if timestep_combined_specs:
        timestep_dir = output_dir / "timestep_conditioned"
        timestep_dir.mkdir(parents=True, exist_ok=True)
        timestep_rows = [row for _property_name, rows in timestep_combined_specs for row in rows]
        write_dict_rows_csv(timestep_dir / "property_feature_importance.csv", timestep_rows)
        with (timestep_dir / "property_feature_importance.json").open("w") as handle:
            json.dump(timestep_rows, handle, indent=2)
        write_property_importance_grid_plot(
            timestep_combined_specs,
            timestep_dir / "all_timestep_conditioned_property_predictive_features.png",
            top_k_features=min(3, top_k_features),
        )
    all_timestep_report = write_all_property_timestep_profile_artifacts(
        property_importance=property_importance,
        timestep_profiles=timestep_profiles,
        output_dir=output_dir / "all_property_timestep_profiles",
        top_k_features=min(3, top_k_features),
        timestep_axis_label=timestep_axis_label,
    )
    if all_timestep_report:
        reports.append(all_timestep_report)
    if structure_feature_summaries is not None and structure_properties is not None:
        reports.extend(
            write_raw_relative_property_scatter_pairs(
                property_importance,
                structure_feature_summaries,
                structure_properties,
                output_dir / "structure_property_scatters",
            )
        )
    return reports


def write_normalized_structure_property_prediction_artifacts(
    structure_feature_summaries: Mapping[tuple[str, int], np.ndarray],
    structure_properties: Mapping[tuple[str, int], Mapping[str, float]],
    property_name: str,
    output_dir: Path,
    *,
    expected_samples_per_sequence: int = 32,
    top_features: int = 64,
) -> dict[str, Any]:
    """Write sequence-grouped held-out predictions for one relative structure property."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{property_name}_observed_vs_predicted.json"
    prepared = _prepare_normalized_structure_property_probe_inputs(
        structure_feature_summaries,
        structure_properties,
        property_name,
        expected_samples_per_sequence=expected_samples_per_sequence,
    )
    finite_keys = prepared["keys"]
    metadata = prepared["metadata"]
    partial_sequences = prepared["partial_sequences"]
    if partial_sequences or not finite_keys:
        report = {
            "status": "skipped",
            "reason": "incomplete ensemble coverage",
            **metadata,
            "rows": [],
        }
        json_path.write_text(json.dumps(report, indent=2) + "\n")
        return {**report, "json_path": str(json_path)}

    groups = prepared["groups"]
    normalized_latents = prepared["normalized_features"]
    targets = prepared["targets"]
    result = run_grouped_rmsf_probe(
        normalized_latents,
        targets,
        groups,
        top_features=top_features,
    )
    scored = result.valid_mask & np.isfinite(result.oof_predictions)
    rows = [
        {
            "sequence_id": finite_keys[row_idx][0],
            "sample_index": finite_keys[row_idx][1],
            "fold": int(result.fold_ids[row_idx]),
            "normalized_ground_truth": float(targets[row_idx]),
            "normalized_heldout_prediction": float(result.oof_predictions[row_idx]),
            "sequence_color": stable_sequence_color(finite_keys[row_idx][0]),
            "sampling_scope": metadata["sampling_scope"],
            "feature_aggregation": metadata["feature_aggregation"],
        }
        for row_idx in np.flatnonzero(scored)
    ]
    report = {
        **metadata,
        **result.summary,
        "fold_metrics": [dict(row) for row in result.fold_metrics],
        "rows": rows,
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    if result.summary.get("status") != "ok":
        return {**report, "json_path": str(json_path)}

    csv_path = json_path.with_suffix(".csv")
    plot_path = json_path.with_suffix(".png")
    write_dict_rows_csv(csv_path, rows)
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    axis.scatter(
        [row["normalized_ground_truth"] for row in rows],
        [row["normalized_heldout_prediction"] for row in rows],
        c=[row["sequence_color"] for row in rows],
        s=8,
        alpha=0.25,
        linewidths=0,
    )
    values = np.concatenate([targets[scored], result.oof_predictions[scored]])
    bounds = [float(values.min()), float(values.max())]
    axis.plot(bounds, bounds, color="#333333", linestyle="--", linewidth=1.0)
    axis.set_xlabel("normalized ground truth")
    axis.set_ylabel("normalized held-out prediction")
    axis.set_title(
        f"{human_readable_property_label(property_name)}\n"
        f"held-out Spearman ρ={float(result.summary['oof_spearman']):.3f}"
    )
    axis.grid(True, alpha=0.22)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=PLOT_DPI)
    plt.close(figure)
    return {
        **report,
        "plot_type": "normalized_observed_vs_predicted",
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "plot_path": str(plot_path),
    }


def _prepare_normalized_structure_property_probe_inputs(
    structure_feature_summaries: Mapping[tuple[str, int], np.ndarray],
    structure_properties: Mapping[tuple[str, int], Mapping[str, float]],
    property_name: str,
    *,
    expected_samples_per_sequence: int,
) -> dict[str, Any]:
    """Return complete, sequence-normalized rows for one ensemble property probe."""

    aligned_keys = sorted(set(structure_feature_summaries) & set(structure_properties))
    finite_keys = [
        key
        for key in aligned_keys
        if math.isfinite(float(structure_properties[key].get(property_name, float("nan"))))
        and np.isfinite(np.asarray(structure_feature_summaries[key])).all()
    ]
    counts = Counter(sequence_id for sequence_id, _sample_index in finite_keys)
    all_sequences = sorted({sequence_id for sequence_id, _ in structure_properties})
    partial_sequences = {
        sequence_id: counts.get(sequence_id, 0)
        for sequence_id in all_sequences
        if counts.get(sequence_id, 0) != expected_samples_per_sequence
    }
    coverage = {
        "num_structures": len(finite_keys),
        "num_sequences": len(counts),
        "expected_samples_per_sequence": expected_samples_per_sequence,
        "partial_sequences": partial_sequences,
        "missing_summary_keys": [
            list(key)
            for key in sorted(set(structure_properties) - set(structure_feature_summaries))
        ],
        "missing_property_keys": [
            list(key)
            for key in sorted(set(structure_feature_summaries) - set(structure_properties))
        ],
    }
    metadata: dict[str, Any] = {
        "property": property_name,
        "target_normalization": "within_sequence_zscore",
        "feature_normalization": "within_sequence_zscore",
        "feature_aggregation": "structure_mean_all_residues_all_10_layers",
        "sampling_scope": "bioemu_32_structure_ensemble",
        "coverage": coverage,
    }
    if partial_sequences or not finite_keys:
        return {
            "keys": finite_keys,
            "groups": np.asarray([], dtype=object),
            "normalized_features": np.empty((0, 0), dtype=np.float32),
            "targets": np.asarray([], dtype=np.float64),
            "metadata": metadata,
            "partial_sequences": partial_sequences,
        }
    latents = torch.from_numpy(
        np.stack([np.asarray(structure_feature_summaries[key]) for key in finite_keys])
    ).float()
    groups = np.asarray([key[0] for key in finite_keys], dtype=object)
    return {
        "keys": finite_keys,
        "groups": groups,
        "normalized_features": within_sequence_zscore_features(groups.tolist(), latents).numpy(),
        "targets": np.asarray(
            [float(structure_properties[key][property_name]) for key in finite_keys],
            dtype=np.float64,
        ),
        "metadata": metadata,
        "partial_sequences": partial_sequences,
    }


def write_normalized_structure_property_prior_comparison(
    *,
    source_feature_summaries: Mapping[tuple[str, int], np.ndarray],
    prior_feature_summaries: Mapping[str, Mapping[tuple[str, int], np.ndarray]],
    structure_properties: Mapping[tuple[str, int], Mapping[str, float]],
    property_name: str,
    output_dir: Path,
    source_report: Path,
    expected_samples_per_sequence: int = 32,
    top_features: int = 64,
    dry_run: bool = False,
) -> dict[str, object]:
    """Compare ensemble-property ridge predictions to matched input priors."""

    prepared = _prepare_normalized_structure_property_probe_inputs(
        source_feature_summaries,
        structure_properties,
        property_name,
        expected_samples_per_sequence=expected_samples_per_sequence,
    )
    keys = prepared["keys"]
    metadata = prepared["metadata"]
    partial_sequences = prepared["partial_sequences"]
    if partial_sequences or not keys:
        status = "incomplete"
        summary = build_normalized_comparison_summary(
            status=status,
            source_report=source_report,
            summary_path=output_dir / "prior_comparison" / property_name / "summary.json",
            representations={},
            targets={
                property_name: {
                    "status": status,
                    "reason": "incomplete ensemble coverage",
                    "series": {},
                }
            },
            coverage=metadata["coverage"],
            probe_configuration={
                "top_features": top_features,
                "target_normalization": metadata["target_normalization"],
                "feature_normalization": metadata["feature_normalization"],
                "sampling_scope": metadata["sampling_scope"],
            },
            fingerprint_identity={
                "property": property_name,
                "coverage": metadata["coverage"],
            },
            artifacts=[],
        )
        if not dry_run:
            comparison_dir = output_dir / "prior_comparison" / property_name
            comparison_dir.mkdir(parents=True, exist_ok=True)
            write_rmsf_json_atomic(comparison_dir / "summary.json", summary)
        return summary
    groups = prepared["groups"]
    targets = prepared["targets"]
    expected_key_set = set(keys)
    prior_matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, summaries in prior_feature_summaries.items():
        if set(summaries) != expected_key_set:
            raise ValueError(f"{name} must contain exactly the source structure keys")
        values = np.stack([np.asarray(summaries[key]) for key in keys]).astype(np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"{name} prior summaries must be finite feature vectors")
        normalized = within_sequence_zscore_features(
            groups.tolist(), torch.from_numpy(values)
        ).numpy()
        prior_matrices[name] = values, normalized
    representations = {
        "sae": {"width": int(prepared["normalized_features"].shape[1])},
        **{
            name: {
                "width": int(values.shape[1]),
                "sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
            }
            for name, (values, _normalized) in sorted(prior_matrices.items())
        },
    }
    probe_configuration = {
        "top_features": top_features,
        "target_normalization": metadata["target_normalization"],
        "feature_normalization": metadata["feature_normalization"],
        "feature_aggregation": metadata["feature_aggregation"],
        "sampling_scope": metadata["sampling_scope"],
    }
    if dry_run:
        return {
            "status": "validated",
            "dry_run": True,
            "representations": representations,
            "coverage": metadata["coverage"],
            "probe_configuration": probe_configuration,
        }
    source_result = run_grouped_rmsf_probe(
        prepared["normalized_features"], targets, groups, top_features=top_features
    )
    source_series = _structure_property_probe_series(
        "sae", source_result, keys, targets, width=prepared["normalized_features"].shape[1]
    )
    series = [source_series]
    series_by_name: dict[str, dict[str, object]] = {
        "sae": _structure_property_series_report(source_series, keys, source_result.fold_ids)
    }
    for name, (values, normalized) in prior_matrices.items():
        if not bool(np.any(np.abs(normalized) > 1e-12)):
            prior_series = ProbeComparisonSeries(
                representation=name,
                display_name=name,
                width=values.shape[1],
                feature_count=top_features,
                status="degenerate",
                reason="prior has no within-sequence variation",
                metrics={},
            )
            fold_ids = source_result.fold_ids
        else:
            result = run_grouped_rmsf_probe(normalized, targets, groups, top_features=top_features)
            if not np.array_equal(result.fold_ids, source_result.fold_ids):
                raise ValueError("prior probe folds do not match source SAE folds")
            prior_series = _structure_property_probe_series(
                name, result, keys, targets, width=values.shape[1]
            )
            fold_ids = result.fold_ids
        series.append(prior_series)
        series_by_name[name] = _structure_property_series_report(prior_series, keys, fold_ids)
    comparison_dir = output_dir / "prior_comparison" / property_name
    details = write_probe_comparison(
        comparison_dir,
        property_name,
        series,
        feature_counts=None,
        provenance={
            **metadata,
            "top_features": top_features,
            "row_ids": [list(key) for key in keys],
            "source_fold_ids": source_result.fold_ids.tolist(),
        },
    )
    prior_statuses = [value["status"] for name, value in series_by_name.items() if name != "sae"]
    status = (
        "ok"
        if "ok" in prior_statuses
        else "degenerate"
        if prior_statuses and all(value in {"degenerate", "skipped"} for value in prior_statuses)
        else "incomplete"
    )
    summary = build_normalized_comparison_summary(
        status=status,
        source_report=source_report,
        summary_path=comparison_dir / "summary.json",
        representations=representations,
        targets={
            property_name: {
                "status": status,
                "series": series_by_name,
                "comparison_series": details["series"],
            }
        },
        coverage=metadata["coverage"],
        probe_configuration=probe_configuration,
        fingerprint_identity={
            "property": property_name,
            "representations": representations,
            "row_ids": [list(key) for key in keys],
            "source_fold_ids": source_result.fold_ids.tolist(),
            "probe_configuration": probe_configuration,
        },
        artifacts=details["artifacts"],
    )
    write_rmsf_json_atomic(comparison_dir / "summary.json", summary)
    return summary


def write_persisted_structure_property_prior_comparison(
    *,
    source_prediction_report: Mapping[str, object],
    prior_feature_summaries: Mapping[str, Mapping[tuple[str, int], np.ndarray]],
    property_name: str,
    output_dir: Path,
    source_report: Path,
    source_width: int,
    top_features: int = 64,
    comparison_root: Path | None = None,
) -> dict[str, object]:
    """Fit priors on immutable BASE OOF rows without recomputing the SAE source."""

    raw_rows = source_prediction_report.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"persisted {property_name} source predictions are missing")
    keys: list[tuple[str, int]] = []
    seen_keys: set[tuple[str, int]] = set()
    targets: list[float] = []
    predictions: list[float] = []
    folds: list[int] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError(f"persisted {property_name} source row is invalid")
        key = (str(raw["sequence_id"]), int(raw["sample_index"]))
        if key in seen_keys:
            raise ValueError(f"persisted {property_name} source rows contain duplicates")
        seen_keys.add(key)
        keys.append(key)
        targets.append(float(raw["normalized_ground_truth"]))
        predictions.append(float(raw["normalized_heldout_prediction"]))
        folds.append(int(raw["fold"]))
        if (
            raw.get("sampling_scope") != "bioemu_32_structure_ensemble"
            or raw.get("feature_aggregation") != "structure_mean_all_residues_all_10_layers"
        ):
            raise ValueError(f"persisted {property_name} source row scope mismatch")
    target_values = np.asarray(targets, dtype=np.float64)
    source_predictions = np.asarray(predictions, dtype=np.float64)
    source_folds = np.asarray(folds, dtype=np.int16)
    if not np.isfinite(target_values).all() or not np.isfinite(source_predictions).all():
        raise ValueError(f"persisted {property_name} source rows must be finite")
    groups = np.asarray([key[0] for key in keys], dtype=object)
    counts = Counter(groups.tolist())
    if not counts or set(counts.values()) != {32}:
        raise ValueError(f"persisted {property_name} source coverage is incomplete")
    folds_by_group = {
        group: {int(source_folds[index]) for index in np.flatnonzero(groups == group)}
        for group in counts
    }
    if any(len(group_folds) != 1 for group_folds in folds_by_group.values()):
        raise ValueError(f"persisted {property_name} source folds split a sequence")
    expected_source_folds = deterministic_group_folds(groups, 5)
    if not np.array_equal(source_folds, expected_source_folds):
        raise ValueError(f"persisted {property_name} source fold identity is invalid")
    expected_keys = set(keys)
    prior_matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, summaries in prior_feature_summaries.items():
        if set(summaries) != expected_keys:
            raise ValueError(f"{name} must contain exactly the persisted source structure keys")
        values = np.stack([np.asarray(summaries[key]) for key in keys]).astype(np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"{name} prior summaries must be finite feature vectors")
        normalized = within_sequence_zscore_features(groups.tolist(), torch.from_numpy(values))
        prior_matrices[name] = values, normalized.numpy()
    if not prior_matrices:
        raise ValueError("structure-property prior replay requires an input representation")
    source_series = ProbeComparisonSeries(
        representation="sae",
        display_name="SAE",
        width=source_width,
        feature_count=top_features,
        status=str(source_prediction_report.get("status", "skipped")),
        reason=(
            str(source_prediction_report["reason"])
            if source_prediction_report.get("reason") is not None
            else None
        ),
        metrics={
            key: value
            for key, value in source_prediction_report.items()
            if key
            not in {
                "rows",
                "fold_metrics",
                "coverage",
                "property",
                "status",
                "reason",
                "target_normalization",
                "feature_normalization",
                "feature_aggregation",
                "sampling_scope",
                "_source_artifacts",
            }
            and isinstance(value, str | int | float | bool)
        },
        prediction_rows=tuple(
            {
                "sequence_id": key[0],
                "sample_index": key[1],
                "fold": int(source_folds[index]),
                "observed": float(target_values[index]),
                "predicted": float(source_predictions[index]),
            }
            for index, key in enumerate(keys)
        ),
    )
    series = [source_series]
    series_by_name: dict[str, dict[str, object]] = {
        "sae": _structure_property_series_report(source_series, keys, source_folds)
    }
    for name, (values, normalized) in prior_matrices.items():
        if not bool(np.any(np.abs(normalized) > 1e-12)):
            prior_series = ProbeComparisonSeries(
                representation=name,
                display_name=name,
                width=values.shape[1],
                feature_count=top_features,
                status="degenerate",
                reason="prior has no within-sequence variation",
                metrics={},
            )
            prior_folds = source_folds
        else:
            result = run_grouped_rmsf_probe(
                normalized,
                target_values,
                groups,
                top_features=top_features,
            )
            if not np.array_equal(result.fold_ids, source_folds):
                raise ValueError("prior probe folds do not match persisted source fold IDs")
            prior_series = _structure_property_probe_series(
                name,
                result,
                keys,
                target_values,
                width=values.shape[1],
            )
            prior_folds = result.fold_ids
        series.append(prior_series)
        series_by_name[name] = _structure_property_series_report(
            prior_series,
            keys,
            prior_folds,
        )
    comparison_dir = (comparison_root or output_dir / "prior_comparison") / property_name
    source_artifacts = source_prediction_report.get("_source_artifacts")
    provenance = {
        "property": property_name,
        "row_ids": [list(key) for key in keys],
        "source_fold_ids": source_folds.tolist(),
        "source_feature_aggregation": "all_10_layers",
        "prior_feature_aggregation": "single_encoder_output",
        "source_artifacts": source_artifacts,
    }
    details = write_probe_comparison(
        comparison_dir,
        property_name,
        series,
        feature_counts=None,
        provenance=provenance,
    )
    prior_statuses = [value["status"] for name, value in series_by_name.items() if name != "sae"]
    status = (
        "ok"
        if "ok" in prior_statuses
        else "degenerate"
        if prior_statuses and all(value in {"degenerate", "skipped"} for value in prior_statuses)
        else "incomplete"
    )
    representations = {
        "sae": {
            "width": source_width,
            "aggregation": "all_10_layers",
        },
        **{
            name: {
                "width": int(values.shape[1]),
                "aggregation": "single_encoder_output",
                "sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
            }
            for name, (values, _normalized) in sorted(prior_matrices.items())
        },
    }
    probe_configuration = {
        "top_features": top_features,
        "target_normalization": "within_sequence_zscore",
        "feature_normalization": "within_sequence_zscore",
        "source_feature_aggregation": "all_10_layers",
        "prior_feature_aggregation": "single_encoder_output",
        "sampling_scope": "bioemu_32_structure_ensemble",
    }
    summary = build_normalized_comparison_summary(
        status=status,
        source_report=source_report,
        summary_path=comparison_dir / "summary.json",
        representations=representations,
        targets={
            property_name: {
                "status": status,
                "series": series_by_name,
                "comparison_series": details["series"],
            }
        },
        coverage=dict(source_prediction_report["coverage"]),  # type: ignore[arg-type]
        probe_configuration=probe_configuration,
        fingerprint_identity={
            **provenance,
            "representations": representations,
            "probe_configuration": probe_configuration,
        },
        artifacts=details["artifacts"],
    )
    write_rmsf_json_atomic(comparison_dir / "summary.json", summary)
    return summary


def _validate_persisted_structure_property_source(
    report: Mapping[str, object],
    *,
    property_name: str,
    expected_keys: set[tuple[str, int]],
) -> None:
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_keys):
        raise ValueError(f"persisted {property_name} source coverage is incomplete")
    seen: set[tuple[str, int]] = set()
    groups: list[str] = []
    folds: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"persisted {property_name} source row is invalid")
        key = (str(row["sequence_id"]), int(row["sample_index"]))
        if key in seen or key not in expected_keys:
            raise ValueError(f"persisted {property_name} source row identity is invalid")
        seen.add(key)
        groups.append(key[0])
        folds.append(int(row["fold"]))
        numeric = np.asarray(
            [
                float(row["normalized_ground_truth"]),
                float(row["normalized_heldout_prediction"]),
            ],
            dtype=np.float64,
        )
        if not np.isfinite(numeric).all():
            raise ValueError(f"persisted {property_name} source rows must be finite")
    if seen != expected_keys or set(Counter(groups).values()) != {32}:
        raise ValueError(f"persisted {property_name} source coverage is incomplete")
    group_values = np.asarray(groups, dtype=object)
    fold_values = np.asarray(folds, dtype=np.int16)
    if not np.array_equal(fold_values, deterministic_group_folds(group_values, 5)):
        raise ValueError(f"persisted {property_name} source fold identity is invalid")


def _load_persisted_structure_property_predictions(
    source_report: Path,
    properties: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Resolve copied BASE normalized OOF JSON/CSV sidecars by local content."""

    payload = json.loads(source_report.read_text())
    reports = payload.get("property_importance")
    if not isinstance(reports, list):
        raise ValueError("source structure-property report lacks property importance artifacts")
    by_property = {
        str(report.get("property")): report
        for report in reports
        if isinstance(report, Mapping) and report.get("property") in properties
    }
    if set(by_property) != set(properties):
        raise ValueError("source structure-property normalized predictions are incomplete")
    loaded: dict[str, dict[str, object]] = {}
    local_root = source_report.parent / "property_importance" / "normalized_predictions"
    for property_name in properties:
        declaration = by_property[property_name]
        json_path = _resolve_copied_property_artifact(
            source_report.parent,
            local_root,
            declaration.get("json_path"),
        )
        csv_path = _resolve_copied_property_artifact(
            source_report.parent,
            local_root,
            declaration.get("csv_path"),
        )
        report = json.loads(json_path.read_text())
        if (
            not isinstance(report, dict)
            or report.get("property") != property_name
            or report.get("status") != "ok"
            or report.get("sampling_scope") != "bioemu_32_structure_ensemble"
            or report.get("feature_aggregation") != "structure_mean_all_residues_all_10_layers"
        ):
            raise ValueError(f"persisted {property_name} source report contract mismatch")
        with csv_path.open(newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        json_rows = report.get("rows")
        if not isinstance(json_rows, list) or len(csv_rows) != len(json_rows):
            raise ValueError(f"persisted {property_name} source CSV/JSON rows disagree")
        identity_fields = (
            "sequence_id",
            "sample_index",
            "fold",
            "normalized_ground_truth",
            "normalized_heldout_prediction",
            "sampling_scope",
            "feature_aggregation",
        )
        for csv_row, json_row in zip(csv_rows, json_rows, strict=True):
            if not isinstance(json_row, Mapping) or any(
                str(csv_row[field]) != str(json_row[field]) for field in identity_fields
            ):
                raise ValueError(f"persisted {property_name} source CSV/JSON rows disagree")
        report["_source_artifacts"] = {
            "json": {
                "sha256": _analysis_file_sha256(json_path),
                "size_bytes": json_path.stat().st_size,
            },
            "csv": {
                "sha256": _analysis_file_sha256(csv_path),
                "size_bytes": csv_path.stat().st_size,
            },
        }
        loaded[property_name] = report
    return loaded


def _resolve_copied_property_artifact(
    report_root: Path,
    local_root: Path,
    raw_path: object,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("source structure-property artifact path is missing")
    raw = Path(raw_path)
    candidates = [local_root / raw.name]
    if not raw.is_absolute() and ".." not in raw.parts:
        candidates.append(report_root / raw)
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise ValueError(f"copied structure-property artifact is missing: {raw.name}")
    hashes = {_analysis_file_sha256(path) for path in existing}
    if len(hashes) != 1:
        raise ValueError(f"copied structure-property artifact is ambiguous: {raw.name}")
    return existing[0]


def _resolve_relocated_structure_pdb(structure_root: Path, raw_path: str | Path) -> Path:
    raw = Path(raw_path)
    local_candidates: list[Path] = []
    if not raw.is_absolute() and ".." not in raw.parts:
        local_candidates.append(structure_root / raw)
    if "final_structures" in raw.parts:
        marker = raw.parts.index("final_structures")
        local_candidates.append(structure_root.joinpath(*raw.parts[marker:]))
    local_candidates.extend(
        (
            structure_root / "final_structures" / raw.name,
            structure_root / raw.name,
        )
    )
    existing_local = list(
        dict.fromkeys(path.resolve() for path in local_candidates if path.is_file())
    )
    if existing_local:
        hashes = {_analysis_file_sha256(path) for path in existing_local}
        if len(hashes) != 1:
            raise ValueError(f"relocated structure PDB is ambiguous: {raw.name}")
        return existing_local[0]
    if raw.is_absolute() and raw.is_file():
        return raw.resolve()
    raise FileNotFoundError(f"missing relocated structure PDB: {raw}")


def _analysis_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _structure_property_probe_series(
    name: str,
    result: RMSFProbeResult,
    keys: list[tuple[str, int]],
    targets: np.ndarray,
    *,
    width: int,
) -> ProbeComparisonSeries:
    scored = result.valid_mask & np.isfinite(result.oof_predictions)
    return ProbeComparisonSeries(
        representation=name,
        display_name="SAE" if name == "sae" else name,
        width=width,
        feature_count=None,
        status=str(result.summary.get("status", "skipped")),
        reason=result.summary.get("reason"),
        metrics={
            key: value for key, value in result.summary.items() if key not in {"status", "reason"}
        },
        prediction_rows=tuple(
            {
                "sequence_id": keys[row][0],
                "sample_index": keys[row][1],
                "fold": int(result.fold_ids[row]),
                "observed": float(targets[row]),
                "predicted": float(result.oof_predictions[row]),
            }
            for row in np.flatnonzero(scored)
        ),
    )


def _structure_property_series_report(
    series: ProbeComparisonSeries,
    keys: list[tuple[str, int]],
    fold_ids: np.ndarray,
) -> dict[str, object]:
    return {
        "status": series.status,
        "reason": series.reason,
        "row_ids": [list(key) for key in keys],
        "fold_ids": fold_ids.tolist(),
        "metrics": dict(series.metrics),
    }


def write_selected_feature_property_timestep_reports(
    timestep_profiles: dict[int, list[dict[str, float | int]]],
    selected_features_by_property: dict[str, list[int]],
    f1_per_domain_by_property_feature: Mapping[tuple[str, int], float],
    output_dir: Path,
    *,
    timestep_axis_label: str = BIOEMU_TIMESTEP_AXIS_LABEL,
) -> list[dict[str, Any]]:
    if not selected_features_by_property:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    all_specs = []
    for property_name, feature_indices in selected_features_by_property.items():
        rows = [
            {
                "feature": feature_idx,
                "metric_name": "f1_per_domain",
                "metric_value": f1_per_domain_by_property_feature[(property_name, feature_idx)],
                "importance_score": float("nan"),
            }
            for feature_idx in feature_indices
        ]
        specs = property_timestep_profile_specs(property_name, rows, timestep_profiles)
        if not specs:
            continue
        all_specs.extend(specs)
        property_dir = property_specific_output_dir(output_dir, property_name)
        data_rows = timestep_profile_data_rows(specs)
        safe_property = _safe_name(property_name)
        csv_path = property_dir / f"{safe_property}_timestep_profiles.csv"
        json_path = property_dir / f"{safe_property}_timestep_profiles.json"
        plot_path = property_dir / f"{safe_property}_timestep_profiles.png"
        property_dir.mkdir(parents=True, exist_ok=True)
        write_dict_rows_csv(csv_path, data_rows)
        with json_path.open("w") as handle:
            json.dump(data_rows, handle, indent=2)
        write_property_timestep_profile_plot(
            property_name,
            specs,
            plot_path,
            timestep_axis_label=timestep_axis_label,
        )
        reports.append(
            {
                "property": property_name,
                "plot_type": "interpro_property_timestep_profiles",
                "features": [spec["feature"] for spec in specs],
                "plot_path": str(plot_path),
                "csv_path": str(csv_path),
                "json_path": str(json_path),
                "num_features": len(specs),
                "normalization": "each feature line divided by its max mean activation across timestep bins",
            }
        )
    if all_specs:
        csv_path = output_dir / "all_interpro_timestep_profiles.csv"
        json_path = output_dir / "all_interpro_timestep_profiles.json"
        plot_path = output_dir / "all_interpro_timestep_profiles.png"
        data_rows = timestep_profile_data_rows(all_specs)
        write_dict_rows_csv(csv_path, data_rows)
        with json_path.open("w") as handle:
            json.dump(data_rows, handle, indent=2)
        write_all_property_timestep_profile_heatmap(
            all_specs,
            plot_path,
            timestep_axis_label=timestep_axis_label,
        )
        reports.append(
            {
                "property": "interpro_concepts",
                "plot_type": "all_interpro_timestep_profile_heatmap",
                "plot_path": str(plot_path),
                "csv_path": str(csv_path),
                "json_path": str(json_path),
                "num_feature_property_pairs": len(all_specs),
                "normalization": "each feature-property row divided by its max mean activation across timestep bins",
            }
        )
        overlay_path = output_dir / "all_interpro_timestep_profiles_overlay.png"
        write_selected_feature_timestep_profile_overlay(
            specs=[
                (int(spec["feature"]), str(spec["property"]), spec["rows"]) for spec in all_specs
            ],
            path=overlay_path,
            title="InterPro-Selected Feature Activation Across Timesteps",
            timestep_axis_label=timestep_axis_label,
        )
        reports.append(
            {
                "property": "interpro_concepts",
                "plot_type": "all_interpro_timestep_profile_overlay",
                "plot_path": str(overlay_path),
                "csv_path": str(csv_path),
                "json_path": str(json_path),
                "num_feature_property_pairs": len(all_specs),
                "normalization": "each feature-property line divided by its maximum mean activation",
                "color_by": f"{timestep_axis_label} of maximum activation",
            }
        )
    return reports


def write_property_timestep_profile_artifacts(
    property_name: str,
    rows: list[dict[str, float | int | str]],
    timestep_profiles: dict[int, list[dict[str, float | int]]],
    output_dir: Path,
    top_k_features: int,
    *,
    timestep_axis_label: str = BIOEMU_TIMESTEP_AXIS_LABEL,
) -> list[dict[str, Any]]:
    selected_rows = top_positive_negative_importance_rows(rows, top_k_features)
    specs = property_timestep_profile_specs(property_name, selected_rows, timestep_profiles)
    if not specs:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    data_rows = timestep_profile_data_rows(specs)
    safe_property = _safe_name(property_name)
    csv_path = output_dir / f"{safe_property}_timestep_profiles.csv"
    json_path = output_dir / f"{safe_property}_timestep_profiles.json"
    plot_path = output_dir / f"{safe_property}_timestep_profiles.png"
    write_dict_rows_csv(csv_path, data_rows)
    with json_path.open("w") as handle:
        json.dump(data_rows, handle, indent=2)
    write_property_timestep_profile_plot(
        property_name,
        specs,
        plot_path,
        timestep_axis_label=timestep_axis_label,
    )
    return [
        {
            "plot_type": "property_timestep_profiles",
            "property": property_name,
            "features": [spec["feature"] for spec in specs],
            "plot_path": str(plot_path),
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "num_features": len(specs),
            "normalization": "each feature line divided by its max mean activation across timestep bins",
        }
    ]


def write_all_property_timestep_profile_artifacts(
    property_importance: dict[str, list[dict[str, float | int | str]]],
    timestep_profiles: dict[int, list[dict[str, float | int]]],
    output_dir: Path,
    top_k_features: int,
    *,
    timestep_axis_label: str = BIOEMU_TIMESTEP_AXIS_LABEL,
) -> dict[str, Any] | None:
    specs = []
    for property_name, rows in property_importance.items():
        selected_rows = top_positive_negative_importance_rows(rows, top_k_features)
        specs.extend(
            property_timestep_profile_specs(property_name, selected_rows, timestep_profiles)
        )
    if not specs:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "all_property_timestep_profiles.csv"
    json_path = output_dir / "all_property_timestep_profiles.json"
    plot_path = output_dir / "all_property_timestep_profiles.png"
    data_rows = timestep_profile_data_rows(specs)
    write_dict_rows_csv(csv_path, data_rows)
    with json_path.open("w") as handle:
        json.dump(data_rows, handle, indent=2)
    write_all_property_timestep_profile_heatmap(
        specs,
        plot_path,
        timestep_axis_label=timestep_axis_label,
    )
    return {
        "property": "all_properties",
        "plot_type": "all_property_timestep_profile_heatmap",
        "plot_path": str(plot_path),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "num_feature_property_pairs": len(specs),
        "normalization": "each feature-property row divided by its max mean activation across timestep bins",
    }


def property_timestep_profile_specs(
    property_name: str,
    selected_rows: list[dict[str, float | int | str]],
    timestep_profiles: dict[int, list[dict[str, float | int]]],
) -> list[dict[str, Any]]:
    specs = []
    seen: set[tuple[str, int]] = set()
    for row in selected_rows:
        feature_idx = int(row["feature"])
        key = (property_name, feature_idx)
        if key in seen:
            continue
        seen.add(key)
        profile_rows = sorted(
            timestep_profiles.get(feature_idx, []),
            key=lambda profile_row: int(profile_row["timestep_bin"]),
        )
        if not profile_rows:
            continue
        metric_name = str(row.get("metric_name", "spearman_r_sample"))
        if metric_name == "f1_per_domain":
            metric_value = float(row["metric_value"])
            direction = "neutral"
        else:
            metric_value = float(row.get("spearman_r_sample", row.get("spearman_r", 0.0)))
            direction = "positive" if metric_value >= 0.0 else "negative"
        specs.append(
            {
                "property": property_name,
                "feature": feature_idx,
                "direction": direction,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "importance_score": float(row.get("importance_score", float("nan"))),
                "rows": profile_rows,
            }
        )
    return specs


def timestep_profile_data_rows(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    data_rows = []
    for spec in specs:
        for row in spec["rows"]:
            data_row = {
                "property": spec["property"],
                "feature": spec["feature"],
                "direction": spec["direction"],
                "importance_score": spec["importance_score"],
                **row,
            }
            data_row[str(spec["metric_name"])] = spec["metric_value"]
            data_rows.append(data_row)
    return data_rows


def write_property_timestep_profile_plot(
    property_name: str,
    specs: list[dict[str, Any]],
    path: Path,
    *,
    timestep_axis_label: str = BIOEMU_TIMESTEP_AXIS_LABEL,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    label_specs = []
    for spec in specs:
        x_values, y_values = normalized_timestep_profile_arrays(spec["rows"])
        if x_values.size == 0:
            continue
        color = {
            "positive": "#b23b3b",
            "negative": "#3b6fb6",
            "neutral": "#5b5b5b",
        }[spec["direction"]]
        ax.plot(x_values, y_values, marker="o", linewidth=1.6, color=color, alpha=0.86)
        label_specs.append(
            {
                "x": float(x_values[-1]),
                "y": float(y_values[-1]),
                "text": feature_timestep_label(spec),
                "color": color,
            }
        )
    add_nonoverlapping_line_labels(ax, label_specs)
    ax.set_xlim(0.0, 1.16)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel(timestep_axis_label)
    ax.set_ylabel("mean activation / feature max")
    display_name = human_readable_property_label(property_name)
    ax.set_title(f"{display_name}: held-out selected features across timesteps")
    ax.grid(True, alpha=0.24)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_all_property_timestep_profile_heatmap(
    specs: list[dict[str, Any]],
    path: Path,
    *,
    timestep_axis_label: str = BIOEMU_TIMESTEP_AXIS_LABEL,
) -> None:
    n_bins = max(int(row["timestep_bin"]) for spec in specs for row in spec["rows"]) + 1
    matrix = np.zeros((len(specs), n_bins), dtype=np.float64)
    labels = []
    for row_idx, spec in enumerate(specs):
        labels.append(feature_property_timestep_label(spec))
        x_values, y_values = normalized_timestep_profile_arrays(spec["rows"])
        for x_value, y_value in zip(x_values, y_values, strict=False):
            bin_idx = min(n_bins - 1, max(0, int(round(float(x_value) * n_bins - 0.5))))
            matrix[row_idx, bin_idx] = float(y_value)
    fig_height = max(5.0, min(28.0, 0.38 * len(specs)))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    label_size = max(5, min(8, int(320 / max(len(labels), 1))))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=label_size)
    ax.set_xticks(np.arange(n_bins))
    ax.set_xlabel(timestep_axis_label)
    ax.set_ylabel("property-selected feature")
    ax.set_title("Property-selected feature activation across timesteps")
    fig.colorbar(im, ax=ax, label="mean activation / feature max")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def normalized_timestep_profile_arrays(
    rows: list[dict[str, float | int]],
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(
        [0.5 * (float(row["timestep_low"]) + float(row["timestep_high"])) for row in rows],
        dtype=np.float64,
    )
    y_values = np.asarray([float(row["mean_activation"]) for row in rows], dtype=np.float64)
    max_value = float(np.nanmax(y_values)) if y_values.size else 0.0
    if max_value > 0.0:
        y_values = y_values / max_value
    else:
        y_values = np.zeros_like(y_values)
    return x_values, y_values


def feature_timestep_label(spec: dict[str, Any]) -> str:
    if spec.get("metric_name") == "f1_per_domain":
        return f"f{int(spec['feature'])} F1={float(spec['metric_value']):.2f}"
    value = float(spec["metric_value"] if "metric_value" in spec else spec["spearman_r_sample"])
    sign = "+" if value >= 0.0 else "-"
    return f"f{int(spec['feature'])} {sign}{abs(value):.2f}"


def feature_property_timestep_label(spec: dict[str, Any]) -> str:
    property_label = compact_property_label(str(spec["property"]))
    return f"{property_label} | {feature_timestep_label(spec)}"


def compact_property_label(property_name: str) -> str:
    label = human_readable_property_label(property_name).replace("relative_", "rel_")
    label = label.replace("radius_of_gyration", "Rg")
    label = label.replace("contact_density", "contact")
    label = label.replace("diffusion_timestep", "timestep")
    label = label.replace("early_", "early ").replace("late_", "late ")
    label = label.replace("_", " ")
    return label[:46] + "..." if len(label) > 49 else label


def add_nonoverlapping_line_labels(ax: plt.Axes, label_specs: list[dict[str, Any]]) -> None:
    if not label_specs:
        return
    ordered = sorted(label_specs, key=lambda item: float(item["y"]))
    min_gap = 0.055
    positions = []
    previous = -float("inf")
    for item in ordered:
        y_pos = max(float(item["y"]), previous + min_gap, 0.02)
        positions.append(y_pos)
        previous = y_pos
    overflow = positions[-1] - 1.02
    if overflow > 0.0:
        positions = [max(0.02, y_pos - overflow) for y_pos in positions]
    for item, y_pos in zip(ordered, positions, strict=False):
        ax.text(
            min(float(item["x"]) + 0.025, 1.06),
            y_pos,
            str(item["text"]),
            color=str(item["color"]),
            fontsize=8,
            va="center",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
        )


def write_property_importance_bar_plot(
    rows: list[dict[str, float | int | str]],
    path: Path,
    property_name: str,
    top_k_features: int,
) -> None:
    selected_rows = top_positive_negative_importance_rows(rows, top_k_features)
    if not selected_rows:
        return
    labels = [str(int(row["feature"])) for row in selected_rows]
    values = [float(row["spearman_r_sample"]) for row in selected_rows]
    colors = ["#b23b3b" if value >= 0 else "#3b6fb6" for value in values]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.34), 4.5))
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("SAE feature")
    ax.set_ylabel("held-out sequence Spearman r")
    ax.set_title(f"Held-Out Selected Features: {human_readable_property_label(property_name)}")
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_property_importance_grid_plot(
    specs: list[tuple[str, list[dict[str, float | int | str]]]],
    path: Path,
    top_k_features: int,
) -> None:
    n_plots = len(specs)
    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.6 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )
    for ax, (property_name, rows) in zip(axes.flat, specs, strict=False):
        selected_rows = top_positive_negative_importance_rows(rows, top_k_features)
        labels = [str(int(row["feature"])) for row in selected_rows]
        values = [float(row["spearman_r_sample"]) for row in selected_rows]
        colors = ["#b23b3b" if value >= 0 else "#3b6fb6" for value in values]
        ax.bar(labels, values, color=colors)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(human_readable_property_label(property_name))
        ax.set_xlabel("feature")
        ax.set_ylabel("held-out sequence Spearman r")
        ax.tick_params(axis="x", labelrotation=60)
    for ax in list(axes.flat)[n_plots:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_property_prediction_scatter_plots(
    property_name: str,
    rows: list[dict[str, float | int | str]],
    prediction_sample: PredictionSample | None,
    output_dir: Path,
    top_k_features: int,
) -> list[dict[str, Any]]:
    if prediction_sample is None or property_name not in prediction_sample.property_values:
        return []
    selected_rows = top_positive_negative_importance_rows(rows, top_k_features)
    if not selected_rows:
        return []
    y_values = prediction_sample.property_values[property_name]
    reports = []
    subplot_specs = []
    for row in selected_rows:
        feature_idx = int(row["feature"])
        sample_latents = prediction_sample.latents_for(property_name)
        if feature_idx >= sample_latents.shape[1]:
            continue
        x_values = sample_latents[:, feature_idx].astype(np.float64)
        path = output_dir / f"feature_{feature_idx:04d}_activation_vs_property.png"
        data_path = path.with_suffix(".csv")
        write_property_prediction_scatter_data(
            feature_idx=feature_idx,
            property_name=property_name,
            x_values=x_values,
            y_values=y_values,
            path=data_path,
        )
        display_name = human_readable_property_label(property_name)
        write_scatter_plot(
            x_values=x_values,
            y_values=y_values,
            path=path,
            title=f"{display_name}: feature {feature_idx}",
            x_label="SAE feature activation",
            y_label=display_name,
        )
        reports.append(
            {
                "feature": feature_idx,
                "plot_path": str(path),
                "spearman_r_sample": float(row["spearman_r_sample"]),
                "pearson_r": float(row["pearson_r"]),
                "r2_univariate": float(row["r2_univariate"]),
                "data_path": str(data_path),
            }
        )
        subplot_specs.append((feature_idx, x_values, y_values, row))
    if subplot_specs:
        grid_path = output_dir / "top_feature_activation_vs_property_grid.png"
        write_property_prediction_scatter_grid(property_name, subplot_specs, grid_path)
        reports.append({"combined_plot_path": str(grid_path), "num_features": len(subplot_specs)})
    return reports


def write_property_prediction_scatter_data(
    feature_idx: int,
    property_name: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    path: Path,
) -> None:
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    rows = [
        {
            "sample_row": int(row_idx),
            "feature": feature_idx,
            "property": property_name,
            "activation": float(x_values[row_idx]),
            "property_value": float(y_values[row_idx]),
        }
        for row_idx in np.nonzero(mask)[0].tolist()
    ]
    write_dict_rows_csv(path, rows)


def write_property_prediction_scatter_grid(
    property_name: str,
    specs: list[tuple[int, np.ndarray, np.ndarray, dict[str, float | int | str]]],
    path: Path,
) -> None:
    n_plots = len(specs)
    n_cols = min(2, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 4.2 * n_rows),
        squeeze=False,
    )
    for ax, (feature_idx, x_values, y_values, row) in zip(axes.flat, specs, strict=False):
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        ax.scatter(x_values[mask], y_values[mask], s=9, alpha=0.48, edgecolors="none")
        ax.set_title(
            f"Feature {feature_idx} "
            f"(rho={float(row['spearman_r_sample']):.2f}, R2={float(row['r2_univariate']):.2f})"
        )
        ax.set_xlabel("activation")
        ax.set_ylabel(human_readable_property_label(property_name))
        ax.grid(True, alpha=0.22)
    for ax in list(axes.flat)[n_plots:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def top_positive_negative_importance_rows(
    rows: list[dict[str, float | int | str]],
    top_k_features: int,
) -> list[dict[str, float | int | str]]:
    positive = sorted(
        [
            row
            for row in rows
            if float(row.get("spearman_r_sample", 0.0)) >= 0
            and float(row.get("importance_score", 0.0)) > 0
        ],
        key=lambda row: float(row["importance_score"]),
        reverse=True,
    )[:top_k_features]
    negative = sorted(
        [
            row
            for row in rows
            if float(row.get("spearman_r_sample", 0.0)) < 0
            and float(row.get("importance_score", 0.0)) > 0
        ],
        key=lambda row: float(row["importance_score"]),
        reverse=True,
    )[:top_k_features]
    return positive + list(reversed(negative))


def write_concept_f1_plot(
    rows: list[dict[str, float | int]],
    path: Path,
    property_name: str,
    top_k: int = 30,
) -> None:
    top_rows = sorted(rows, key=lambda row: float(row["f1"]), reverse=True)[:top_k]
    if not top_rows:
        return
    labels = [f"{int(row['feature'])}@{float(row['threshold']):.2g}" for row in top_rows]
    values = [float(row["f1"]) for row in top_rows]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.28), 4.5))
    ax.bar(labels, values, color="#3b6fb6")
    ax.set_ylabel("F1")
    ax.set_title(
        f"Top Feature/Threshold Concept Matches: {human_readable_property_label(property_name)}"
    )
    ax.tick_params(axis="x", labelrotation=65)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_residue_activation_plots(
    trainer: SAETrainer,
    config: ProjectConfig,
    structure_entries: list[StructureIndexEntry] | None,
    store: ActivationStore | None,
    examples: dict[int, list[FeatureExampleCandidate]],
    sequences: dict[str, str],
    output_dir: Path,
    max_examples_per_feature: int,
    max_total_plots: int = 50,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if structure_entries is not None:
        return [
            {
                "skipped": True,
                "reason": (
                    "runtime residue activation profiles require an additional "
                    "activation replay; skipped so analysis does not recompute "
                    "activations"
                ),
            }
        ]
    reports = []
    for feature_idx, candidates in examples.items():
        for example_idx, candidate in enumerate(candidates[:max_examples_per_feature]):
            if len(reports) >= max_total_plots:
                return reports
            plot_values = feature_activation_profile_for_candidate(
                trainer=trainer,
                config=config,
                structure_entries=structure_entries,
                store=store,
                feature_idx=feature_idx,
                candidate=candidate,
            )
            if plot_values is None:
                continue
            sequence = sequences.get(candidate.sequence_id, "")
            path = output_dir / (
                f"feature_{feature_idx:04d}_example_{example_idx:02d}_"
                f"{_safe_name(candidate.sequence_id)}_sample_{candidate.sample_index}.png"
            )
            data_path = path.with_suffix(".csv")
            write_residue_activation_profile_data(
                values=plot_values,
                sequence=sequence,
                candidate=candidate,
                feature_idx=feature_idx,
                path=data_path,
            )
            write_residue_activation_plot(
                values=plot_values,
                sequence=sequence,
                candidate=candidate,
                feature_idx=feature_idx,
                path=path,
            )
            reports.append(
                {
                    "feature": feature_idx,
                    "sequence_id": candidate.sequence_id,
                    "sample_index": candidate.sample_index,
                    "residue_index": candidate.residue_index,
                    "timestep": candidate.timestep,
                    "plot_path": str(path),
                    "data_path": str(data_path),
                }
            )
    return reports


def feature_activation_profile_for_candidate(
    trainer: SAETrainer,
    config: ProjectConfig,
    structure_entries: list[StructureIndexEntry] | None,
    store: ActivationStore | None,
    feature_idx: int,
    candidate: FeatureExampleCandidate,
    runtime_sampler: BioEmuRuntimeActivationSampler | None = None,
    record_provider: Any | None = None,
    record_cache: dict[tuple[str, int | None], ActivationRecord | None] | None = None,
) -> np.ndarray | None:
    record = find_record_for_candidate(
        config=config,
        structure_entries=structure_entries,
        store=store,
        candidate=candidate,
        device=trainer.device,
        runtime_sampler=runtime_sampler,
        record_provider=record_provider,
        record_cache=record_cache,
    )
    if record is None:
        return None
    trainer.model.eval()
    values_by_timestep: dict[float, dict[int, float]] = defaultdict(dict)
    with torch.no_grad():
        for start in range(0, record.n_tokens, trainer.config.training.batch_size):
            end = start + trainer.config.training.batch_size
            activations = record.activations[start:end].to(trainer.device)
            timesteps = record.timesteps[start:end].to(trainer.device)
            with trainer.autocast_context():
                latents = trainer.model(activations, timesteps).latents.detach().cpu()
            sequence_ids = record.sequence_ids[start:end]
            residues = record.residue_indices[start:end].detach().cpu().tolist()
            timestep_values = record.timesteps[start:end].detach().cpu().tolist()
            sample_indices = (
                record.sample_indices[start:end].detach().cpu().tolist()
                if record.sample_indices is not None
                else [candidate.sample_index] * len(sequence_ids)
            )
            for row_idx, sequence_id in enumerate(sequence_ids):
                if sequence_id != candidate.sequence_id:
                    continue
                if (
                    candidate.sample_index is not None
                    and sample_indices[row_idx] != candidate.sample_index
                ):
                    continue
                timestep = float(timestep_values[row_idx])
                residue_idx = int(residues[row_idx])
                values_by_timestep[timestep][residue_idx] = float(
                    latents[row_idx, feature_idx].item()
                )
    if not values_by_timestep:
        return None
    matching_timesteps = [
        timestep
        for timestep in values_by_timestep
        if np.isclose(timestep, candidate.timestep, atol=(1.0 / 255.0))
    ]
    if matching_timesteps:
        selected_timestep = min(
            matching_timesteps,
            key=lambda timestep: abs(timestep - candidate.timestep),
        )
    else:
        candidate_timesteps = [
            timestep
            for timestep, values in values_by_timestep.items()
            if candidate.residue_index in values
        ]
        if not candidate_timesteps:
            return None
        selected_timestep = max(
            candidate_timesteps,
            key=lambda timestep: values_by_timestep[timestep][candidate.residue_index],
        )
    values_by_residue = values_by_timestep[selected_timestep]
    length = max(values_by_residue) + 1
    values = np.zeros(length, dtype=float)
    for residue_idx, value in values_by_residue.items():
        values[residue_idx] = value
    return values


def find_record_for_candidate(
    config: ProjectConfig,
    structure_entries: list[StructureIndexEntry] | None,
    store: ActivationStore | None,
    candidate: FeatureExampleCandidate,
    device: torch.device,
    runtime_sampler: BioEmuRuntimeActivationSampler | None = None,
    record_provider: Any | None = None,
    record_cache: dict[tuple[str, int | None], ActivationRecord | None] | None = None,
) -> ActivationRecord | None:
    cache_key = (candidate.sequence_id, candidate.sample_index)
    if record_cache is not None and cache_key in record_cache:
        return record_cache[cache_key]
    if record_provider is not None:
        record = record_provider(candidate)
        if record_cache is not None:
            record_cache[cache_key] = record
        return record
    if store is not None:
        for record in store.iter_records():
            if candidate.sequence_id not in record.sequence_ids:
                continue
            if record_cache is not None:
                record_cache[cache_key] = record
            return record
        if record_cache is not None:
            record_cache[cache_key] = None
        return None
    if structure_entries is None:
        return None
    entry = next(
        (
            item
            for item in structure_entries
            if item.sequence_id == candidate.sequence_id
            and item.sample_index == candidate.sample_index
        ),
        None,
    )
    if entry is None:
        if record_cache is not None:
            record_cache[cache_key] = None
        return None
    sampler = runtime_sampler or BioEmuRuntimeActivationSampler(
        config=config.trace,
        cache_embeds_dir=config.training.cache_embeds_dir,
        structures_dir=None,
        batch_size=config.training.runtime_sequence_batch_size,
        device=device,
    )
    record = sampler.sample(
        [SequenceRecord(sequence_id=entry.sequence_id, sequence=entry.sequence)],
        seed=entry.sample_index,
    )
    if record_cache is not None:
        record_cache[cache_key] = record
    return record


def write_residue_activation_plot(
    values: np.ndarray,
    sequence: str,
    candidate: FeatureExampleCandidate,
    feature_idx: int,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(values) * 0.08), 3.2))
    x = np.arange(len(values))
    ax.plot(x, values, color="#3b6fb6", linewidth=1.4)
    label_nonzero_residue_activations(ax, values, sequence, min_activation=0.0)
    ax.set_xlabel("residue index")
    ax.set_ylabel("SAE activation")
    ax.set_title(
        f"Feature {feature_idx}: {candidate.sequence_id}, sample {candidate.sample_index}, "
        f"t={candidate.timestep:.3f}"
    )
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_residue_activation_profile_data(
    values: np.ndarray,
    sequence: str,
    candidate: FeatureExampleCandidate,
    feature_idx: int,
    path: Path,
) -> None:
    rows = [
        {
            "feature": feature_idx,
            "sequence_id": candidate.sequence_id,
            "sample_index": candidate.sample_index,
            "timestep": float(candidate.timestep),
            "highlight_residue": int(residue_idx == candidate.residue_index),
            "residue_index": residue_idx,
            "amino_acid": sequence[residue_idx] if residue_idx < len(sequence) else "",
            "activation": float(activation),
        }
        for residue_idx, activation in enumerate(np.asarray(values, dtype=np.float64))
    ]
    write_dict_rows_csv(path, rows)


def write_feature_localization_artifacts(
    trainer: SAETrainer,
    config: ProjectConfig,
    store: ActivationStore | None,
    examples: dict[int, list[FeatureExampleCandidate]],
    feature_summary: list[dict[str, float | int]],
    structure_entries: list[StructureIndexEntry],
    structure_root: Path,
    sequences: dict[str, str],
    rmsf_by_sequence: dict[str, np.ndarray],
    feature_property_labels: dict[int, str],
    selected_features_by_property: dict[str, list[int]],
    property_importance: dict[str, list[dict[str, float | int | str]]],
    output_dir: Path,
    max_examples_per_feature: int,
    max_total_artifacts: int = 50,
    max_top_residues: int = 12,
    record_provider: Any | None = None,
    categorize_bioemu_features: bool = False,
    runtime_sampler_factory: Any | None = None,
) -> list[dict[str, Any]]:
    """Write compact residue and structure views for selected high-activation examples."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_by_feature = {int(row["feature"]): row for row in feature_summary if "feature" in row}
    entries_by_sample = {
        (entry.sequence_id, entry.sample_index): entry for entry in structure_entries
    }
    record_cache: dict[tuple[str, int | None], ActivationRecord | None] = {}
    sampler_factory = runtime_sampler_factory or BioEmuRuntimeActivationSampler
    runtime_sampler = (
        sampler_factory(
            config=config.trace,
            cache_embeds_dir=config.training.cache_embeds_dir,
            structures_dir=None,
            batch_size=config.training.runtime_sequence_batch_size,
            device=trainer.device,
        )
        if store is None and record_provider is None
        else None
    )
    reports = []
    example_report_count = 0
    for feature_idx, candidates in examples.items():
        if example_report_count >= max_total_artifacts:
            break
        feature_max = float(summary_by_feature.get(feature_idx, {}).get("max_activation", 0.0))
        property_label = feature_property_labels.get(feature_idx, "unknown property")
        used_sequences: set[str] = set()
        full_profiles = []
        structure_png_paths = []
        feature_example_count = 0
        for candidate in candidates:
            if candidate.sequence_id in used_sequences:
                continue
            used_sequences.add(candidate.sequence_id)
            if feature_example_count >= max_examples_per_feature:
                break
            if example_report_count >= max_total_artifacts:
                break
            profile = feature_activation_profile_for_candidate(
                trainer=trainer,
                config=config,
                structure_entries=structure_entries,
                store=store,
                feature_idx=feature_idx,
                candidate=candidate,
                runtime_sampler=runtime_sampler,
                record_provider=record_provider,
                record_cache=record_cache,
            )
            hits = top_residue_hits_for_sample(
                candidates=candidates,
                sequence_id=candidate.sequence_id,
                sample_index=candidate.sample_index,
                feature_max=feature_max,
                max_residues=max_top_residues,
            )
            if not hits:
                continue
            feature_dir = feature_localization_directory(
                output_dir,
                feature_idx,
                property_label,
                categorized=categorize_bioemu_features,
            )
            feature_dir.mkdir(parents=True, exist_ok=True)
            sample_label = (
                f"{_safe_name(candidate.sequence_id)}_sample_"
                f"{candidate.sample_index if candidate.sample_index is not None else 'none'}"
            )
            has_rmsf = candidate.sequence_id in rmsf_by_sequence
            residue_plot_stem = "rmsf_residue_hits" if has_rmsf else "activation_profile"
            residue_plot_path = feature_dir / f"{sample_label}_{residue_plot_stem}.png"
            residue_data_path = residue_plot_path.with_suffix(".csv")
            report = {
                "feature": feature_idx,
                "selected_for_property": property_label,
                "sequence_id": candidate.sequence_id,
                "sample_index": candidate.sample_index,
                "feature_max_activation": feature_max,
                "residue_hits": [
                    {
                        "residue_index": residue_idx,
                        "activation": values["activation"],
                        "normalized_activation": values["normalized_activation"],
                    }
                    for residue_idx, values in sorted(
                        hits.items(),
                        key=lambda item: item[1]["activation"],
                        reverse=True,
                    )
                ],
                "residue_plot_path": str(residue_plot_path),
                "residue_data_path": str(residue_data_path),
                "residue_plot_type": residue_plot_stem,
            }
            if profile is not None:
                normalized_profile = normalize_feature_profile(profile, feature_max)
                full_profiles.append(
                    {
                        "candidate": candidate,
                        "profile": profile,
                        "normalized_profile": normalized_profile,
                        "sequence": sequences.get(candidate.sequence_id, ""),
                        "rmsf": rmsf_by_sequence.get(candidate.sequence_id),
                    }
                )
                full_hits = residue_hits_from_profile(
                    normalized_profile,
                    top_k=max_top_residues,
                )
                structure_values = residue_values_from_profile(normalized_profile)
                report["full_residue_profile"] = {
                    "num_residues": int(len(normalized_profile)),
                    "max_normalized_activation": float(np.max(normalized_profile))
                    if len(normalized_profile)
                    else 0.0,
                    "top_residues": [
                        {
                            "residue_index": residue_idx,
                            "normalized_activation": values["normalized_activation"],
                            "activation": float(profile[residue_idx])
                            if residue_idx < len(profile)
                            else 0.0,
                        }
                        for residue_idx, values in full_hits.items()
                    ],
                }
            else:
                full_hits = hits
                structure_values = hits
            plot_values = structure_values if profile is not None else hits
            write_feature_rmsf_localization_data(
                sequence=sequences.get(candidate.sequence_id, ""),
                rmsf=rmsf_by_sequence.get(candidate.sequence_id),
                residue_hits=plot_values,
                feature_idx=feature_idx,
                sequence_id=candidate.sequence_id,
                sample_index=candidate.sample_index,
                path=residue_data_path,
            )
            write_feature_rmsf_localization_plot(
                sequence=sequences.get(candidate.sequence_id, ""),
                rmsf=rmsf_by_sequence.get(candidate.sequence_id),
                residue_hits=plot_values,
                feature_idx=feature_idx,
                sequence_id=candidate.sequence_id,
                sample_index=candidate.sample_index,
                property_label=property_label,
                path=residue_plot_path,
            )
            entry = entries_by_sample.get(
                (
                    candidate.sequence_id,
                    int(candidate.sample_index) if candidate.sample_index is not None else -1,
                )
            )
            if entry is None or entry.pdb_path is None:
                report["structure_skipped"] = "cached PDB path is not available for this sample"
                reports.append(report)
                feature_example_count += 1
                example_report_count += 1
                continue
            pdb_path = structure_root / entry.pdb_path
            if not pdb_path.exists():
                report["structure_skipped"] = f"cached PDB does not exist: {pdb_path}"
                reports.append(report)
                feature_example_count += 1
                example_report_count += 1
                continue
            colored_pdb_path = feature_dir / f"{sample_label}_activation_bfactor.pdb"
            pymol_script_path = feature_dir / f"{sample_label}_view.pml"
            render_raw_png_path = feature_dir / f"{sample_label}_render_raw.png"
            pymol_scale_path = feature_dir / f"{sample_label}_activation_scale.png"
            pymol_png_path = feature_dir / f"{sample_label}_pymol.png"
            fallback_png_path = feature_dir / f"{sample_label}_matplotlib_fallback.png"
            pymol_png_path.unlink(missing_ok=True)
            fallback_png_path.unlink(missing_ok=True)
            html_path = feature_dir / f"{sample_label}_viewer.html"
            write_activation_bfactor_pdb(pdb_path, colored_pdb_path, structure_values)
            write_pymol_activation_script(
                pdb_path=colored_pdb_path,
                script_path=pymol_script_path,
                png_path=render_raw_png_path,
                feature_idx=feature_idx,
                sequence_id=candidate.sequence_id,
                property_label=property_label,
            )
            pymol_render = render_pymol_png(pymol_script_path, render_raw_png_path)
            write_activation_scale_png(pymol_scale_path)
            renderer = str(pymol_render.get("renderer", "none"))
            rendered_png_path = pymol_png_path if renderer == "pymol" else fallback_png_path
            composed = False
            if render_raw_png_path.exists() and bool(pymol_render.get("rendered")):
                composed = compose_pymol_png_with_scale(
                    image_path=render_raw_png_path,
                    scale_path=pymol_scale_path,
                    output_path=rendered_png_path,
                )
            if renderer == "pymol" and bool(pymol_render.get("rendered")) and composed:
                structure_png_paths.append(pymol_png_path)
            write_3dmol_activation_html(
                pdb_path=colored_pdb_path,
                html_path=html_path,
                residue_hits=structure_values,
                feature_idx=feature_idx,
                sequence_id=candidate.sequence_id,
                property_label=property_label,
            )
            report.update(
                {
                    "colored_pdb_path": str(colored_pdb_path),
                    "pymol_script_path": str(pymol_script_path),
                    "render_raw_png_path": str(render_raw_png_path),
                    "pymol_scale_path": str(pymol_scale_path),
                    "pymol_render": pymol_render,
                    "renderer": renderer,
                    "html_viewer_path": str(html_path),
                }
            )
            if renderer == "pymol" and bool(pymol_render.get("rendered")) and composed:
                report["pymol_png_path"] = str(pymol_png_path)
            elif (
                renderer == "matplotlib_fallback"
                and bool(pymol_render.get("rendered"))
                and composed
            ):
                report["fallback_png_path"] = str(fallback_png_path)
            reports.append(report)
            feature_example_count += 1
            example_report_count += 1
        if full_profiles:
            feature_dir = feature_localization_directory(
                output_dir,
                feature_idx,
                property_label,
                categorized=categorize_bioemu_features,
            )
            grid_path = feature_dir / f"feature_{feature_idx:04d}_full_residue_profiles.png"
            grid_data_path = grid_path.with_suffix(".csv")
            structure_grid_path = feature_dir / f"feature_{feature_idx:04d}_structure_views.png"
            write_full_residue_profile_grid_data(
                profiles=full_profiles,
                feature_idx=feature_idx,
                path=grid_data_path,
            )
            write_full_residue_profile_grid(
                profiles=full_profiles,
                feature_idx=feature_idx,
                property_label=property_label,
                path=grid_path,
            )
            structure_grid_written = write_structure_view_grid(
                png_paths=structure_png_paths,
                path=structure_grid_path,
                feature_idx=feature_idx,
                property_label=property_label,
            )
            reports.append(
                {
                    "feature": feature_idx,
                    "full_residue_profile_grid_path": str(grid_path),
                    "full_residue_profile_grid_data_path": str(grid_data_path),
                    "structure_view_grid_path": str(structure_grid_path)
                    if structure_grid_written
                    else None,
                    "num_profiles": len(full_profiles),
                    "normalization": "activation divided by feature max_activation from the analysis pass",
                }
            )
        ensure_feature_localization_category_link(
            output_dir,
            feature_idx,
            property_label,
            categorized=categorize_bioemu_features,
        )
    property_grid_reports = write_property_best_feature_localization_grids(
        localization_reports=reports,
        selected_features_by_property=selected_features_by_property,
        property_importance=property_importance,
        output_dir=output_dir / "property_best_examples",
        top_k_features=3,
    )
    reports.extend(property_grid_reports)
    return reports


def feature_localization_categories(property_label: str) -> tuple[bool, bool]:
    labels = [label for label in property_label.split(", ") if label]
    is_interpro = any(label.startswith("interpro:") for label in labels)
    is_bioemu = any(not label.startswith("interpro:") for label in labels)
    return is_bioemu or not labels, is_interpro


def feature_localization_directory(
    output_dir: Path,
    feature_idx: int,
    property_label: str,
    *,
    categorized: bool,
) -> Path:
    feature_name = f"feature_{feature_idx:04d}"
    if not categorized:
        return output_dir / feature_name
    is_bioemu, _is_interpro = feature_localization_categories(property_label)
    category = "bioemu_features" if is_bioemu else "interpro_concepts"
    return output_dir / category / feature_name


def ensure_feature_localization_category_link(
    output_dir: Path,
    feature_idx: int,
    property_label: str,
    *,
    categorized: bool,
) -> None:
    if not categorized or feature_localization_categories(property_label) != (True, True):
        return
    feature_name = f"feature_{feature_idx:04d}"
    canonical = output_dir / "bioemu_features" / feature_name
    link = output_dir / "interpro_concepts" / feature_name
    link.parent.mkdir(parents=True, exist_ok=True)
    target = Path("..") / "bioemu_features" / feature_name
    if link.is_symlink():
        if link.readlink() != target:
            raise FileExistsError(f"conflicting feature localization symlink: {link}")
        return
    if link.exists():
        raise FileExistsError(f"feature localization category path already exists: {link}")
    if canonical.is_dir():
        link.symlink_to(target, target_is_directory=True)


def write_property_best_feature_localization_grids(
    localization_reports: list[dict[str, Any]],
    selected_features_by_property: dict[str, list[int]],
    property_importance: dict[str, list[dict[str, float | int | str]]],
    output_dir: Path,
    top_k_features: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    first_render_by_feature: dict[int, dict[str, Any]] = {}
    for report in localization_reports:
        if (
            report.get("renderer") != "pymol"
            or "pymol_png_path" not in report
            or "feature" not in report
        ):
            continue
        png_path = Path(str(report["pymol_png_path"]))
        if not png_path.exists():
            continue
        feature_idx = int(report["feature"])
        first_render_by_feature.setdefault(feature_idx, report)

    reports = []
    for property_name, rows in property_importance.items():
        allowed_features = set(selected_features_by_property.get(property_name, []))
        selected_rows = [
            row
            for row in top_positive_negative_importance_rows(rows, top_k_features)
            if int(row["feature"]) in allowed_features
        ]
        grid_rows = []
        for rank, row in enumerate(selected_rows, start=1):
            feature_idx = int(row["feature"])
            render_report = first_render_by_feature.get(feature_idx)
            if render_report is None:
                continue
            direction = (
                "positive" if float(row.get("spearman_r_sample", 0.0)) >= 0.0 else "negative"
            )
            grid_rows.append(
                {
                    "property": property_name,
                    "rank": rank,
                    "direction": direction,
                    "feature": feature_idx,
                    "spearman_r_sample": float(row.get("spearman_r_sample", float("nan"))),
                    "importance_score": float(row.get("importance_score", float("nan"))),
                    "sequence_id": render_report.get("sequence_id"),
                    "sample_index": render_report.get("sample_index"),
                    "pymol_png_path": render_report.get("pymol_png_path"),
                    "residue_plot_path": render_report.get("residue_plot_path"),
                    "residue_data_path": render_report.get("residue_data_path"),
                }
            )
        if not grid_rows:
            continue

        property_dir = property_specific_output_dir(output_dir, property_name)
        property_dir.mkdir(parents=True, exist_ok=True)
        safe_property = _safe_name(property_name)
        csv_path = property_dir / f"{safe_property}_best_feature_examples.csv"
        json_path = property_dir / f"{safe_property}_best_feature_examples.json"
        plot_path = property_dir / f"{safe_property}_best_feature_examples.png"
        write_dict_rows_csv(csv_path, grid_rows)
        with json_path.open("w") as handle:
            json.dump(grid_rows, handle, indent=2)
        write_property_best_feature_localization_grid_plot(
            rows=grid_rows,
            property_name=property_name,
            path=plot_path,
        )
        reports.append(
            {
                "property": property_name,
                "plot_path": str(plot_path),
                "csv_path": str(csv_path),
                "json_path": str(json_path),
                "num_features": len(grid_rows),
            }
        )
    return reports


def write_property_best_feature_localization_grid_plot(
    rows: list[dict[str, Any]],
    property_name: str,
    path: Path,
) -> None:
    n_plots = len(rows)
    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.8 * n_rows),
        squeeze=False,
    )
    for ax, row in zip(axes.flat, rows, strict=False):
        png_path = Path(str(row["pymol_png_path"]))
        image = plt.imread(png_path)
        ax.imshow(image)
        ax.axis("off")
        direction = str(row["direction"])
        feature_idx = int(row["feature"])
        rho = float(row["spearman_r_sample"])
        ax.set_title(
            f"{direction} | feature {feature_idx} | rho={rho:.2f}\n"
            f"{row['sequence_id']}, sample {row['sample_index']}",
            fontsize=9,
        )
    for ax in list(axes.flat)[n_plots:]:
        ax.axis("off")
    fig.suptitle(
        f"{human_readable_property_label(property_name)}: "
        "best localized examples for held-out selected features",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def top_residue_hits_for_sample(
    candidates: list[FeatureExampleCandidate],
    sequence_id: str,
    sample_index: int | None,
    feature_max: float,
    max_residues: int,
) -> dict[int, dict[str, float]]:
    hits: dict[int, dict[str, float]] = {}
    denominator = feature_max if feature_max > 0 else 1.0
    for candidate in candidates:
        if candidate.sequence_id != sequence_id or candidate.sample_index != sample_index:
            continue
        residue_idx = int(candidate.residue_index)
        activation = float(candidate.activation)
        previous = hits.get(residue_idx)
        if previous is not None and previous["activation"] >= activation:
            continue
        hits[residue_idx] = {
            "activation": activation,
            "normalized_activation": float(np.clip(activation / denominator, 0.0, 1.0)),
        }
    return dict(
        sorted(
            hits.items(),
            key=lambda item: item[1]["activation"],
            reverse=True,
        )[:max_residues]
    )


def normalize_feature_profile(values: np.ndarray, feature_max: float) -> np.ndarray:
    denominator = feature_max if feature_max > 0 else 1.0
    normalized = np.asarray(values, dtype=np.float64) / denominator
    return np.clip(normalized, 0.0, 1.0)


def residue_hits_from_profile(
    normalized_profile: np.ndarray,
    top_k: int,
) -> dict[int, dict[str, float]]:
    if normalized_profile.size == 0 or top_k <= 0:
        return {}
    ranked = np.argsort(normalized_profile)[::-1][:top_k]
    return {
        int(residue_idx): {
            "activation": float(normalized_profile[residue_idx]),
            "normalized_activation": float(normalized_profile[residue_idx]),
        }
        for residue_idx in ranked
        if float(normalized_profile[residue_idx]) > 0.0
    }


def residue_values_from_profile(normalized_profile: np.ndarray) -> dict[int, dict[str, float]]:
    return {
        int(residue_idx): {
            "activation": float(value),
            "normalized_activation": float(value),
        }
        for residue_idx, value in enumerate(normalized_profile)
    }


def write_full_residue_profile_grid(
    profiles: list[dict[str, Any]],
    feature_idx: int,
    property_label: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_profiles = min(len(profiles), 4)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), squeeze=False)
    for ax, profile_info in zip(axes.flat, profiles[:n_profiles], strict=False):
        candidate = profile_info["candidate"]
        normalized_profile = np.asarray(profile_info["normalized_profile"], dtype=np.float64)
        sequence = str(profile_info.get("sequence") or "")
        rmsf = profile_info.get("rmsf")
        show_rmsf = should_show_rmsf_overlay(property_label)
        x = np.arange(len(normalized_profile))
        ax.plot(x, normalized_profile, color="#b23b3b", linewidth=1.5, label="feature activation")
        ax.fill_between(x, normalized_profile, color="#b23b3b", alpha=0.18)
        label_nonzero_residue_activations(ax, normalized_profile, sequence)
        if show_rmsf and rmsf is not None and len(rmsf) > 0:
            rmsf_values = np.asarray(rmsf, dtype=np.float64)
            rmsf_max = float(np.nanmax(rmsf_values))
            rmsf_norm = rmsf_values / rmsf_max if rmsf_max > 0 else rmsf_values
            ax.plot(
                np.arange(len(rmsf_norm)),
                rmsf_norm,
                color="#444444",
                linewidth=1.1,
                alpha=0.8,
                label="RMSF",
            )
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("residue index")
        ax.set_ylabel("normalized value")
        display_label = human_readable_property_label(property_label)
        ax.set_title(
            f"{candidate.sequence_id}, sample {candidate.sample_index}, "
            f"t={candidate.timestep:.3f}\nselected for: {display_label}",
            fontsize=10,
        )
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=7, loc="upper right")
    for ax in list(axes.flat)[n_profiles:]:
        ax.axis("off")
    fig.suptitle(
        f"Feature {feature_idx}: full per-residue activation profiles",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_full_residue_profile_grid_data(
    profiles: list[dict[str, Any]],
    feature_idx: int,
    path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for profile_idx, profile_info in enumerate(profiles[:4]):
        candidate = profile_info["candidate"]
        normalized_profile = np.asarray(profile_info["normalized_profile"], dtype=np.float64)
        raw_profile = np.asarray(profile_info["profile"], dtype=np.float64)
        sequence = str(profile_info.get("sequence") or "")
        rmsf = profile_info.get("rmsf")
        rmsf_values = np.asarray(rmsf, dtype=np.float64) if rmsf is not None else None
        rmsf_max = (
            float(np.nanmax(rmsf_values))
            if rmsf_values is not None and rmsf_values.size > 0
            else 0.0
        )
        for residue_idx, normalized_activation in enumerate(normalized_profile):
            rmsf_value = (
                float(rmsf_values[residue_idx])
                if rmsf_values is not None and residue_idx < len(rmsf_values)
                else float("nan")
            )
            rows.append(
                {
                    "feature": feature_idx,
                    "profile_index": profile_idx,
                    "sequence_id": candidate.sequence_id,
                    "sample_index": candidate.sample_index,
                    "timestep": float(candidate.timestep),
                    "residue_index": residue_idx,
                    "amino_acid": sequence[residue_idx] if residue_idx < len(sequence) else "",
                    "activation": float(raw_profile[residue_idx])
                    if residue_idx < len(raw_profile)
                    else float("nan"),
                    "normalized_activation": float(normalized_activation),
                    "rmsf": rmsf_value,
                    "normalized_rmsf": (rmsf_value / rmsf_max)
                    if rmsf_max > 0 and math.isfinite(rmsf_value)
                    else float("nan"),
                }
            )
    write_dict_rows_csv(path, rows)


def write_feature_rmsf_localization_plot(
    sequence: str,
    rmsf: np.ndarray | None,
    residue_hits: dict[int, dict[str, float]],
    feature_idx: int,
    sequence_id: str,
    sample_index: int | None,
    property_label: str,
    path: Path,
) -> None:
    length = max(
        [len(sequence), *(residue_idx + 1 for residue_idx in residue_hits)]
        + ([len(rmsf)] if rmsf is not None else [])
    )
    x = np.arange(length)
    fig, ax = plt.subplots(figsize=(max(7, length * 0.08), 3.6))
    if should_show_rmsf_overlay(property_label) and rmsf is not None and len(rmsf) > 0:
        rmsf_values = np.asarray(rmsf, dtype=np.float64)
        rmsf_max = float(np.nanmax(rmsf_values))
        rmsf_norm = rmsf_values / rmsf_max if rmsf_max > 0 else rmsf_values
        ax.plot(
            np.arange(len(rmsf_norm)),
            rmsf_norm,
            color="#555555",
            linewidth=1.3,
            label="normalized RMSF",
        )
    profile_values = np.zeros(length, dtype=np.float64)
    for residue_idx, values in residue_hits.items():
        if residue_idx < length:
            profile_values[residue_idx] = float(values["normalized_activation"])
    residue_indices = np.nonzero(profile_values > 0)[0]
    ax.vlines(
        residue_indices,
        0.0,
        profile_values[residue_indices],
        color="#b23b3b",
        linewidth=1.1,
        alpha=0.65,
    )
    label_nonzero_residue_activations(ax, profile_values, sequence)
    ax.set_xlim(-0.5, max(len(x) - 0.5, 0.5))
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("residue index")
    ax.set_ylabel("normalized value")
    display_label = human_readable_property_label(property_label)
    ax.set_title(
        f"Feature {feature_idx}: {sequence_id}, sample {sample_index}\n"
        f"selected for: {display_label}"
    )
    ax.grid(True, alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def label_nonzero_residue_activations(
    ax: plt.Axes,
    normalized_profile: np.ndarray,
    sequence: str,
    min_activation: float = 0.0,
) -> None:
    for residue_idx, value in enumerate(normalized_profile):
        if not math.isfinite(float(value)) or float(value) <= min_activation:
            continue
        aa = sequence[residue_idx] if residue_idx < len(sequence) else ""
        label_residue_activation(
            ax,
            residue_idx=residue_idx,
            value=float(value),
            amino_acid=aa,
            y_offset=0.025,
        )


def should_show_rmsf_overlay(property_label: str) -> bool:
    return "rmsf" in {part.strip() for part in property_label.split(",")}


def label_residue_activation(
    ax: plt.Axes,
    residue_idx: int,
    value: float,
    amino_acid: str,
    y_offset: float,
) -> None:
    label = amino_acid or str(residue_idx)
    y_value = min(float(value) + y_offset, 1.02)
    ax.text(
        residue_idx,
        y_value,
        label,
        ha="center",
        va="bottom",
        fontsize=7,
        rotation=0,
        color=amino_acid_color(amino_acid),
        fontweight="bold",
        zorder=4,
    )


def write_feature_rmsf_localization_data(
    sequence: str,
    rmsf: np.ndarray | None,
    residue_hits: dict[int, dict[str, float]],
    feature_idx: int,
    sequence_id: str,
    sample_index: int | None,
    path: Path,
) -> None:
    length = max(
        [len(sequence), *(residue_idx + 1 for residue_idx in residue_hits)]
        + ([len(rmsf)] if rmsf is not None else [])
    )
    rmsf_values = np.asarray(rmsf, dtype=np.float64) if rmsf is not None else None
    rmsf_max = (
        float(np.nanmax(rmsf_values)) if rmsf_values is not None and rmsf_values.size > 0 else 0.0
    )
    rows = []
    for residue_idx in range(length):
        hit = residue_hits.get(residue_idx)
        rmsf_value = (
            float(rmsf_values[residue_idx])
            if rmsf_values is not None and residue_idx < len(rmsf_values)
            else float("nan")
        )
        rows.append(
            {
                "feature": feature_idx,
                "sequence_id": sequence_id,
                "sample_index": sample_index,
                "residue_index": residue_idx,
                "amino_acid": sequence[residue_idx] if residue_idx < len(sequence) else "",
                "activation": float(hit["activation"]) if hit is not None else 0.0,
                "normalized_activation": float(hit["normalized_activation"])
                if hit is not None
                else 0.0,
                "is_top_hit": int(hit is not None),
                "rmsf": rmsf_value,
                "normalized_rmsf": (rmsf_value / rmsf_max)
                if rmsf_max > 0 and math.isfinite(rmsf_value)
                else float("nan"),
            }
        )
    write_dict_rows_csv(path, rows)


def write_activation_bfactor_pdb(
    input_pdb_path: Path,
    output_pdb_path: Path,
    residue_hits: dict[int, dict[str, float]],
) -> None:
    residue_bfactors = {
        residue_idx + 1: 100.0 * float(values["normalized_activation"])
        for residue_idx, values in residue_hits.items()
    }
    output_pdb_path.parent.mkdir(parents=True, exist_ok=True)
    with input_pdb_path.open() as source, output_pdb_path.open("w") as target:
        for line in source:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 66:
                try:
                    residue_number = int(line[22:26].strip())
                except ValueError:
                    target.write(line)
                    continue
                bfactor = residue_bfactors.get(residue_number, 0.0)
                target.write(f"{line[:54]}{1.00:6.2f}{bfactor:6.2f}{line[66:]}")
            else:
                target.write(line)


def write_pymol_activation_script(
    pdb_path: Path,
    script_path: Path,
    png_path: Path,
    feature_idx: int,
    sequence_id: str,
    property_label: str,
) -> None:
    script_path.write_text(
        "\n".join(
            [
                "reinitialize",
                f"load {pdb_path.resolve()}",
                "hide everything",
                "show cartoon",
                "color gray80",
                "spectrum b, blue_white_red, minimum=0, maximum=100",
                f"set title, Feature {feature_idx}: {sequence_id} | selected for "
                f"{human_readable_property_label(property_label)}",
                "orient",
                f"png {png_path.resolve()}, width=1400, height=1000, dpi={PLOT_DPI}, ray=1",
            ]
        )
        + "\n"
    )


def render_rmsf_probe_pdb(pdb_path: Path, png_path: Path) -> dict[str, Any]:
    """Render an RMSF probe PDB whose B factors encode a 0--100 color scale."""

    script_path = pdb_path.with_suffix(".pml")
    script_path.write_text(
        "\n".join(
            [
                "reinitialize",
                f"load {pdb_path.resolve()}",
                "hide everything",
                "show cartoon",
                "color gray80",
                "spectrum b, blue_white_red, minimum=0, maximum=100",
                "orient",
                f"png {png_path.resolve()}, width=1400, height=1000, dpi={PLOT_DPI}, ray=1",
            ]
        )
        + "\n"
    )
    return render_pymol_png(script_path, png_path)


def render_pymol_png(script_path: Path, png_path: Path) -> dict[str, Any]:
    png_path.unlink(missing_ok=True)
    pdb_path = pdb_path_from_pymol_script(script_path)
    api_rendered = render_pymol_api_png(pdb_path, png_path)
    if api_rendered["rendered"]:
        return _normalize_successful_pymol_render(api_rendered)
    png_path.unlink(missing_ok=True)
    external_api_rendered = render_pymol_external_python_png(pdb_path, png_path)
    if external_api_rendered["rendered"]:
        return _normalize_successful_pymol_render(external_api_rendered)
    png_path.unlink(missing_ok=True)
    pymol = pymol_executable_path()
    if pymol is None:
        fallback = (
            render_activation_backbone_png(pdb_path, png_path) if pdb_path is not None else False
        )
        return {
            "rendered": fallback,
            "renderer": "matplotlib_fallback" if fallback else "none",
            "reason": "pymol executable not found",
            "pymol_api": api_rendered,
            "pymol_external_api": external_api_rendered,
        }
    executable_rendered = render_pymol_executable(script_path, png_path, pdb_path, pymol)
    if (
        executable_rendered.get("rendered")
        and executable_rendered.get("renderer") != "matplotlib_fallback"
    ):
        executable_rendered = _normalize_successful_pymol_render(executable_rendered)
    if executable_rendered.get("renderer") != "matplotlib_fallback":
        executable_rendered["pymol_api"] = api_rendered
        executable_rendered["pymol_external_api"] = external_api_rendered
    return executable_rendered


def _normalize_successful_pymol_render(render: dict[str, Any]) -> dict[str, Any]:
    """Expose successful PyMOL implementations under one artifact provenance label."""

    normalized = dict(render)
    normalized["renderer_backend"] = str(normalized.get("renderer", "pymol"))
    normalized["renderer"] = "pymol"
    return normalized


def pymol_executable_path() -> str | None:
    repo_pymol = Path(__file__).resolve().parents[3] / "envs" / "pymol" / "bin" / "pymol"
    if repo_pymol.is_file() and os.access(repo_pymol, os.X_OK):
        return str(repo_pymol)
    pymol = shutil.which("pymol")
    return str(pymol) if pymol is not None else None


def render_pymol_executable(
    script_path: Path,
    png_path: Path,
    pdb_path: Path | None,
    pymol: str,
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [pymol, "-cq", str(script_path)],
            cwd=script_path.parent,
            text=True,
            capture_output=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        fallback = (
            render_activation_backbone_png(pdb_path, png_path) if pdb_path is not None else False
        )
        return {
            "rendered": fallback,
            "renderer": "matplotlib_fallback" if fallback else "pymol",
            "reason": "pymol render timed out",
            "stdout": (exc.stdout or "")[-2000:],
            "stderr": (exc.stderr or "")[-2000:],
        }
    except OSError as exc:
        fallback = (
            render_activation_backbone_png(pdb_path, png_path) if pdb_path is not None else False
        )
        return {
            "rendered": fallback,
            "renderer": "matplotlib_fallback" if fallback else "pymol",
            "reason": f"pymol executable failed: {exc}",
        }
    if result.returncode != 0 or not png_path.exists():
        fallback = (
            render_activation_backbone_png(pdb_path, png_path) if pdb_path is not None else False
        )
        if fallback:
            return {
                "rendered": True,
                "renderer": "matplotlib_fallback",
                "pymol_returncode": result.returncode,
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:],
            }
    return {
        "rendered": png_path.exists() and result.returncode == 0,
        "renderer": "pymol",
        "executable": pymol,
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-2000:],
    }


def render_pymol_api_png(pdb_path: Path | None, png_path: Path) -> dict[str, Any]:
    if pdb_path is None or not pdb_path.exists():
        return {"rendered": False, "renderer": "pymol_api", "reason": "missing PDB path"}
    try:
        import pymol  # type: ignore[import-not-found]
        from pymol import cmd  # type: ignore[import-not-found]
    except ImportError as exc:
        return {
            "rendered": False,
            "renderer": "pymol_api",
            "reason": f"pymol Python API unavailable: {exc}",
        }
    try:
        try:
            pymol.finish_launching(["pymol", "-cq"])
        except Exception:
            pass
        cmd.reinitialize()
        cmd.load(str(pdb_path.resolve()))
        cmd.hide("everything")
        cmd.show("cartoon")
        cmd.color("gray80")
        cmd.bg_color("white")
        cmd.spectrum("b", "blue_white_red", "all", minimum=0, maximum=100)
        cmd.orient()
        cmd.set("antialias", 2)
        cmd.set("ray_shadows", 0)
        cmd.ray(2500, 2500)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.png(str(png_path.resolve()), dpi=PLOT_DPI)
        return {
            "rendered": png_path.exists(),
            "renderer": "pymol_api",
        }
    except Exception as exc:
        return {
            "rendered": False,
            "renderer": "pymol_api",
            "reason": str(exc)[-2000:],
        }


def render_pymol_external_python_png(pdb_path: Path | None, png_path: Path) -> dict[str, Any]:
    pymol_python = Path(__file__).resolve().parents[3] / "envs" / "pymol" / "bin" / "python"
    if pdb_path is None or not pdb_path.exists():
        return {"rendered": False, "renderer": "pymol_external_api", "reason": "missing PDB path"}
    if not pymol_python.exists():
        return {
            "rendered": False,
            "renderer": "pymol_external_api",
            "reason": f"missing PyMOL Python environment: {pymol_python}",
        }
    code = """
import sys
import pymol
from pymol import cmd

pdb_path, png_path = sys.argv[1], sys.argv[2]
pymol.finish_launching(["pymol", "-cq"])
cmd.reinitialize()
cmd.load(pdb_path)
cmd.hide("everything")
cmd.show("cartoon")
cmd.color("gray80")
cmd.bg_color("white")
cmd.spectrum("b", "blue_white_red", "all", minimum=0, maximum=100)
cmd.orient()
cmd.set("antialias", 2)
cmd.set("ray_shadows", 0)
cmd.ray(2500, 2500)
cmd.png(png_path, dpi=300)
cmd.quit()
"""
    try:
        result = subprocess.run(
            [str(pymol_python), "-c", code, str(pdb_path.resolve()), str(png_path.resolve())],
            cwd=png_path.parent,
            text=True,
            capture_output=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "rendered": False,
            "renderer": "pymol_external_api",
            "reason": "pymol external API render timed out",
            "stdout": (exc.stdout or "")[-2000:],
            "stderr": (exc.stderr or "")[-2000:],
        }
    return {
        "rendered": png_path.exists() and result.returncode == 0,
        "renderer": "pymol_external_api",
        "python": str(pymol_python),
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-2000:],
    }


def pdb_path_from_pymol_script(script_path: Path) -> Path | None:
    if not script_path.exists():
        return None
    for line in script_path.read_text().splitlines():
        if line.startswith("load "):
            value = line.removeprefix("load ").strip()
            return Path(value)
    return None


def render_activation_backbone_png(pdb_path: Path | None, png_path: Path) -> bool:
    if pdb_path is None or not pdb_path.exists():
        return False
    coords, activations = ca_coordinates_and_bfactors(pdb_path)
    if coords.size == 0:
        return False
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    draw_activation_backbone_tube(ax, coords, activations)
    ax.set_axis_off()
    set_equal_3d_axes(ax, coords)
    ax.view_init(elev=18, azim=-68)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=PLOT_DPI)
    plt.close(fig)
    return png_path.exists()


def write_activation_scale_png(path: Path) -> None:
    gradient = np.linspace(0.0, 1.0, 256, dtype=np.float64)[None, :]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 0.7))
    ax.imshow(gradient, aspect="auto", cmap="bwr", extent=(0.0, 1.0, 0.0, 1.0))
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("normalized SAE activation", fontsize=8)
    ax.tick_params(axis="x", labelsize=7, length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def draw_activation_scale_axis(ax: Any) -> None:
    gradient = np.linspace(0.0, 1.0, 256, dtype=np.float64)[None, :]
    ax.imshow(gradient, aspect="auto", cmap="bwr", extent=(0.0, 1.0, 0.0, 1.0))
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("normalized SAE activation", fontsize=8)
    ax.tick_params(axis="x", labelsize=7, length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def compose_pymol_png_with_scale(
    image_path: Path,
    scale_path: Path,
    output_path: Path,
) -> bool:
    if not image_path.exists() or not scale_path.exists():
        return False
    image = plt.imread(image_path)
    scale = plt.imread(scale_path)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.4),
        gridspec_kw={"height_ratios": [12, 1]},
    )
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[1].imshow(scale)
    axes[1].axis("off")
    fig.tight_layout(pad=0.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)
    return output_path.exists()


def draw_activation_backbone_tube(
    ax: Any,
    coords: np.ndarray,
    activations: np.ndarray,
) -> Any:
    if len(coords) == 1:
        point = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=activations,
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            s=90,
            depthshade=False,
        )
        return point
    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    segment_values = (activations[:-1] + activations[1:]) / 2.0
    outline = Line3DCollection(
        segments,
        colors="#555555",
        linewidths=10.5,
        alpha=0.18,
    )
    ax.add_collection3d(outline)
    tube = Line3DCollection(
        segments,
        cmap="coolwarm",
        norm=plt.Normalize(0.0, 1.0),
        linewidths=8.0,
        alpha=0.96,
    )
    tube.set_array(segment_values)
    ax.add_collection3d(tube)
    return tube


def write_structure_view_grid(
    png_paths: list[Path],
    path: Path,
    feature_idx: int,
    property_label: str,
) -> bool:
    existing_paths = [png_path for png_path in png_paths if png_path.exists()]
    if not existing_paths:
        return False
    n_cols = min(2, len(existing_paths))
    n_rows = math.ceil(len(existing_paths) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.2 * n_cols, 5.0 * n_rows),
        squeeze=False,
    )
    for ax, png_path in zip(axes.flat, existing_paths, strict=False):
        image = plt.imread(png_path)
        ax.imshow(image)
        ax.set_title(png_path.name.removesuffix("_pymol.png"), fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[len(existing_paths) :]:
        ax.axis("off")
    display_label = human_readable_property_label(property_label)
    fig.suptitle(f"Feature {feature_idx} structure views | selected for: {display_label}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    return path.exists()


def ca_coordinates_and_bfactors(pdb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    coords = []
    activations = []
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or line[12:16].strip() != "CA" or len(line) < 66:
                continue
            try:
                coords.append(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
                activations.append(float(line[60:66]) / 100.0)
            except ValueError:
                continue
    return np.asarray(coords, dtype=np.float64), np.asarray(activations, dtype=np.float64)


def set_equal_3d_axes(ax: Any, coords: np.ndarray) -> None:
    mins = coords.min(axis=0)
    maxes = coords.max(axis=0)
    centers = (mins + maxes) / 2.0
    radius = float(np.max(maxes - mins) / 2.0)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def write_3dmol_activation_html(
    pdb_path: Path,
    html_path: Path,
    residue_hits: dict[int, dict[str, float]],
    feature_idx: int,
    sequence_id: str,
    property_label: str,
) -> None:
    pdb_text = pdb_path.read_text()
    residue_payload = {
        str(residue_idx + 1): float(values["normalized_activation"])
        for residue_idx, values in residue_hits.items()
    }
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                '<script src="https://3Dmol.org/build/3Dmol-min.js"></script>',
                "</head>",
                "<body>",
                '<div id="viewer" style="width: 900px; height: 650px;"></div>',
                "<script>",
                f"const pdbData = {json.dumps(pdb_text)};",
                f"const activations = {json.dumps(residue_payload)};",
                "const viewer = $3Dmol.createViewer('viewer', {backgroundColor: 'white'});",
                "viewer.addModel(pdbData, 'pdb');",
                "viewer.setStyle({}, {cartoon: {color: 'lightgray'}});",
                "function activationColor(value) {",
                "  const red = Math.round(255 * value);",
                "  const blue = Math.round(190 * (1 - value));",
                "  return `rgb(${red},50,${blue})`;",
                "}",
                "for (const [resi, value] of Object.entries(activations)) {",
                "  viewer.setStyle({resi: resi}, {cartoon: {color: activationColor(value)}});",
                "}",
                "viewer.zoomTo();",
                "viewer.render();",
                "</script>",
                (
                    f"<p>Feature {feature_idx}: {sequence_id}. "
                    f"Selected for: {property_label}. "
                    "Residues are colored by normalized activation.</p>"
                ),
                "</body>",
                "</html>",
            ]
        )
        + "\n"
    )


def _safe_name(value: str, max_length: int = 48) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value)[:max_length].strip("-") or "seq"


def property_specific_output_dir(output_dir: Path, property_name: str) -> Path:
    if is_timestep_conditioned_property(property_name):
        return output_dir / "timestep_conditioned" / property_name
    return output_dir / property_name


def is_timestep_conditioned_property(property_name: str) -> bool:
    return property_name.startswith(("early_", "late_"))


def write_logo_plots(
    examples: dict[int, list[FeatureExampleCandidate]],
    sequences: dict[str, str],
    output_dir: Path,
    window: int,
) -> list[dict[str, Any]]:
    reports = []
    for feature_idx, candidates in examples.items():
        matrix, used = logo_frequency_matrix(candidates, sequences, window)
        matrix_path = output_dir / f"feature_{feature_idx:04d}_logo_matrix.csv"
        np.savetxt(
            matrix_path,
            matrix,
            delimiter=",",
            header=",".join(AMINO_ACIDS),
            comments="",
        )
        plot_path = output_dir / f"feature_{feature_idx:04d}_logo.png"
        write_text_logo(matrix, plot_path, window)
        top_context_path = output_dir / f"feature_{feature_idx:04d}_top4_contexts.png"
        top_context_csv = output_dir / f"feature_{feature_idx:04d}_top4_contexts.csv"
        top_contexts = write_top_logo_context_grid(
            candidates=candidates[:4],
            sequences=sequences,
            feature_idx=feature_idx,
            window=window,
            plot_path=top_context_path,
            csv_path=top_context_csv,
        )
        reports.append(
            {
                "feature": feature_idx,
                "examples_used": used,
                "matrix_path": str(matrix_path),
                "plot_path": str(plot_path),
                "top_context_plot_path": str(top_context_path) if top_contexts else None,
                "top_context_csv_path": str(top_context_csv) if top_contexts else None,
                "top_contexts": top_contexts,
            }
        )
    return reports


def write_property_logo_grids(
    examples: dict[int, list[FeatureExampleCandidate]],
    selected_features_by_property: dict[str, list[int]],
    property_importance: dict[str, list[dict[str, float | int | str]]],
    sequences: dict[str, str],
    output_dir: Path,
    window: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for property_name, feature_indices in selected_features_by_property.items():
        ordered_features = ordered_property_features_for_logo_grid(
            property_name=property_name,
            feature_indices=feature_indices,
            property_importance=property_importance,
        )
        panels = []
        data_rows = []
        for rank, (feature_idx, direction, score) in enumerate(ordered_features, start=1):
            candidates, matrix = logo_candidates_for_feature(
                examples.get(feature_idx, []),
                sequences=sequences,
                window=window,
            )
            if not candidates or matrix is None:
                continue
            panels.append((feature_idx, direction, score, candidates, matrix))
            data_rows.extend(
                logo_grid_data_rows(
                    property_name=property_name,
                    feature_idx=feature_idx,
                    direction=direction,
                    rank=rank,
                    candidates=candidates,
                    matrix=matrix,
                    window=window,
                )
            )
        if not panels:
            continue
        property_dir = property_specific_output_dir(output_dir, property_name)
        property_dir.mkdir(parents=True, exist_ok=True)
        safe_property = _safe_name(property_name)
        plot_path = property_dir / f"{safe_property}_highest_activation_logos.png"
        csv_path = property_dir / f"{safe_property}_highest_activation_logos.csv"
        json_path = property_dir / f"{safe_property}_highest_activation_logos.json"
        write_property_logo_grid_plot(
            panels=panels,
            property_name=property_name,
            window=window,
            path=plot_path,
        )
        write_dict_rows_csv(csv_path, data_rows)
        with json_path.open("w") as handle:
            json.dump(data_rows, handle, indent=2)
        reports.append(
            {
                "plot_type": "property_highest_activation_logo_grid",
                "property": property_name,
                "features": [feature_idx for feature_idx, *_ in panels],
                "plot_path": str(plot_path),
                "csv_path": str(csv_path),
                "json_path": str(json_path),
                "num_features": len(panels),
            }
        )
    return reports


def ordered_property_features_for_logo_grid(
    property_name: str,
    feature_indices: list[int],
    property_importance: dict[str, list[dict[str, float | int | str]]],
) -> list[tuple[int, str, float]]:
    rows = property_importance.get(property_name)
    if not rows:
        return [(feature_idx, "selected", float("nan")) for feature_idx in feature_indices]
    selected = set(feature_indices)
    ordered = []
    for row in top_positive_negative_importance_rows(rows, top_k_features=3):
        feature_idx = int(row["feature"])
        if feature_idx not in selected:
            continue
        direction = "positive" if float(row.get("spearman_r_sample", 0.0)) >= 0.0 else "negative"
        ordered.append((feature_idx, direction, float(row.get("spearman_r_sample", float("nan")))))
    seen = {feature_idx for feature_idx, _, _ in ordered}
    ordered.extend(
        (feature_idx, "selected", float("nan"))
        for feature_idx in feature_indices
        if feature_idx not in seen
    )
    return ordered


def best_logo_candidate_for_feature(
    candidates: list[FeatureExampleCandidate],
    sequences: dict[str, str],
    window: int,
) -> tuple[FeatureExampleCandidate | None, np.ndarray | None]:
    for candidate in candidates:
        matrix, used = logo_frequency_matrix([candidate], sequences, window)
        if used > 0:
            return candidate, matrix
    return None, None


def logo_candidates_for_feature(
    candidates: list[FeatureExampleCandidate],
    sequences: dict[str, str],
    window: int,
    max_examples: int = 50,
) -> tuple[list[FeatureExampleCandidate], np.ndarray | None]:
    usable_candidates = []
    for candidate in candidates:
        sequence = sequences.get(candidate.sequence_id, "")
        if 0 <= candidate.residue_index < len(sequence):
            usable_candidates.append(candidate)
        if len(usable_candidates) >= max_examples:
            break
    if not usable_candidates:
        return [], None
    matrix, used = logo_frequency_matrix(usable_candidates, sequences, window)
    if used == 0:
        return [], None
    return usable_candidates, matrix


def logo_grid_data_rows(
    property_name: str,
    feature_idx: int,
    direction: str,
    rank: int,
    candidates: list[FeatureExampleCandidate],
    matrix: np.ndarray,
    window: int,
) -> list[dict[str, Any]]:
    rows = []
    top_candidate = candidates[0]
    for pos_idx, row in enumerate(matrix):
        relative_position = pos_idx - window
        for aa_idx, frequency in enumerate(row):
            if float(frequency) <= 0.0:
                continue
            rows.append(
                {
                    "property": property_name,
                    "rank": rank,
                    "direction": direction,
                    "feature": feature_idx,
                    "examples_used": len(candidates),
                    "top_sequence_id": top_candidate.sequence_id,
                    "top_sample_index": top_candidate.sample_index,
                    "top_center_residue_index": top_candidate.residue_index,
                    "top_timestep": float(top_candidate.timestep),
                    "top_activation": float(top_candidate.activation),
                    "relative_position": relative_position,
                    "amino_acid": AMINO_ACIDS[aa_idx],
                    "frequency": float(frequency),
                }
            )
    return rows


def write_property_logo_grid_plot(
    panels: list[tuple[int, str, float, list[FeatureExampleCandidate], np.ndarray]],
    property_name: str,
    window: int,
    path: Path,
) -> None:
    n_plots = len(panels)
    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    for ax, (feature_idx, direction, score, candidates, matrix) in zip(
        axes.flat,
        panels,
        strict=False,
    ):
        draw_text_logo(ax, matrix, window)
        score_label = f", rho={score:.2f}" if math.isfinite(score) else ""
        top_candidate = candidates[0]
        ax.set_title(
            f"{direction} | feature {feature_idx}{score_label}\n"
            f"n={len(candidates)} examples | top: {top_candidate.sequence_id}, residue "
            f"{top_candidate.residue_index}",
            fontsize=8,
        )
    for ax in list(axes.flat)[n_plots:]:
        ax.axis("off")
    display_name = property_logo_display_name(property_name)
    fig.suptitle(display_name, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def property_logo_display_name(property_name: str) -> str:
    return human_readable_property_label(property_name)


def logo_frequency_matrix(
    candidates: list[FeatureExampleCandidate],
    sequences: dict[str, str],
    window: int,
) -> tuple[np.ndarray, int]:
    matrix = np.zeros((2 * window + 1, len(AMINO_ACIDS)), dtype=np.float64)
    aa_to_idx = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
    used = 0
    seen_residues: set[tuple[str, int]] = set()
    for candidate in candidates:
        residue_key = (candidate.sequence_id, candidate.residue_index)
        if residue_key in seen_residues:
            continue
        seen_residues.add(residue_key)
        sequence = sequences.get(candidate.sequence_id)
        if sequence is None:
            continue
        center = candidate.residue_index
        if center - window < 0 or center + window >= len(sequence):
            continue
        used += 1
        for offset in range(-window, window + 1):
            aa = sequence[center + offset]
            aa_idx = aa_to_idx.get(aa)
            if aa_idx is not None:
                matrix[offset + window, aa_idx] += 1
    if used > 0:
        matrix /= matrix.sum(axis=1, keepdims=True).clip(min=1.0)
    return matrix, used


def write_text_logo(matrix: np.ndarray, path: Path, window: int) -> None:
    fig, ax = plt.subplots(figsize=(max(6, matrix.shape[0] * 0.45), 2.8))
    draw_text_logo(ax, matrix, window)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def draw_text_logo(ax: plt.Axes, matrix: np.ndarray, window: int) -> None:
    for pos_idx, row in enumerate(matrix):
        for aa_idx in np.argsort(row):
            frequency = float(row[aa_idx])
            if frequency <= 0:
                continue
            glyph_height = min(0.18, 0.035 + 0.45 * frequency)
            add_scaled_logo_letter(
                ax,
                AMINO_ACIDS[aa_idx],
                x_center=float(pos_idx),
                y_bottom=frequency - glyph_height,
                height=glyph_height,
            )
    ax.set_xlim(-0.75, matrix.shape[0] - 0.25)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(matrix.shape[0]))
    ax.set_xticklabels([str(idx) for idx in range(-window, window + 1)])
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Position relative to activating residue")
    ax.set_title("Top Activating Residue Context")


def write_top_logo_context_grid(
    candidates: list[FeatureExampleCandidate],
    sequences: dict[str, str],
    feature_idx: int,
    window: int,
    plot_path: Path,
    csv_path: Path,
) -> list[dict[str, Any]]:
    context_rows = []
    matrices = []
    for context_idx, candidate in enumerate(candidates):
        matrix, used = logo_frequency_matrix([candidate], sequences, window)
        if used == 0:
            continue
        matrices.append((candidate, matrix))
        sequence = sequences.get(candidate.sequence_id, "")
        for offset in range(-window, window + 1):
            residue_index = candidate.residue_index + offset
            aa = sequence[residue_index] if 0 <= residue_index < len(sequence) else ""
            context_rows.append(
                {
                    "feature": feature_idx,
                    "context_rank": context_idx + 1,
                    "sequence_id": candidate.sequence_id,
                    "sample_index": candidate.sample_index,
                    "center_residue_index": candidate.residue_index,
                    "residue_index": residue_index,
                    "relative_position": offset,
                    "amino_acid": aa,
                    "activation": candidate.activation,
                }
            )
    if not matrices:
        return []
    write_dict_rows_csv(csv_path, context_rows)
    n_cols = min(2, len(matrices))
    n_rows = math.ceil(len(matrices) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.2 * n_cols, 2.9 * n_rows),
        squeeze=False,
    )
    for ax, (candidate, matrix) in zip(axes.flat, matrices, strict=False):
        draw_text_logo(ax, matrix, window)
        ax.set_title(
            f"Feature {feature_idx} | residue {candidate.residue_index} | "
            f"activation {candidate.activation:.3g}"
        )
    for ax in list(axes.flat)[len(matrices) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=PLOT_DPI)
    plt.close(fig)
    return context_rows


def add_scaled_logo_letter(
    ax: plt.Axes,
    letter: str,
    x_center: float,
    y_bottom: float,
    height: float,
    max_width: float = 0.8,
) -> None:
    glyph = TextPath((0.0, 0.0), letter, size=1.0, prop=FontProperties(weight="bold"))
    bbox = glyph.get_extents()
    if bbox.height <= 0 or bbox.width <= 0:
        return
    y_scale = height / bbox.height
    x_scale = min(max_width / bbox.width, y_scale * 1.56)
    x_left = x_center - (bbox.width * x_scale) / 2.0 - bbox.x0 * x_scale
    y_offset = y_bottom - bbox.y0 * y_scale
    transform = Affine2D().scale(x_scale, y_scale).translate(x_left, y_offset) + ax.transData
    patch = PathPatch(
        glyph,
        transform=transform,
        facecolor=amino_acid_color(letter),
        edgecolor="none",
        alpha=0.95,
    )
    ax.add_patch(patch)


def amino_acid_color(aa: str) -> str:
    if aa in "AVLIMFWY":
        return "#2f6f4e"
    if aa in "DE":
        return "#b23b3b"
    if aa in "KRH":
        return "#315f9c"
    if aa in "STNQ":
        return "#8a6f2a"
    if aa in "GP":
        return "#7a4f9a"
    return "#555555"
