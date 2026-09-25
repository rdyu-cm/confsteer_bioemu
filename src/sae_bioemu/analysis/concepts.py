from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from sae_bioemu.core.types import ActivationRecord

PLOT_DPI = 300
DEFAULT_THRESHOLDS = (0.0, 0.15, 0.5, 0.8)

CATEGORICAL_FEATURES = (
    ("Active site", "ACT_SITE", "note", True),
    ("Binding site", "BINDING", "ligand", True),
    ("Cofactor", "COFACTOR", "Name", True),
    ("Glycosylation", "CARBOHYD", "note", True),
    ("Modified residue", "MOD_RES", "note", True),
    ("Transit peptide", "TRANSIT", "note", False),
    ("Compositional bias", "COMPBIAS", "note", False),
    ("Domain [FT]", "DOMAIN", "note", False),
    ("Region", "REGION", "note", False),
    ("Zinc finger", "ZN_FING", "note", False),
    ("Motif", "MOTIF", "note", False),
    ("Signal peptide", "SIGNAL", "note", False),
)
BINARY_FEATURES = (
    ("Turn", "TURN", False),
    ("Helix", "HELIX", False),
    ("Beta strand", "STRAND", False),
    ("Coiled coil", "COILED", False),
)
INTERACTION_FEATURES = (("Disulfide bond", "DISULFID", True),)
AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class ConceptDefinition:
    """One Swiss-Prot concept mapped onto residue/domain labels."""

    name: str
    labels_by_sequence: dict[str, np.ndarray]
    is_aa_level: bool
    positive_residues: int
    positive_domains: int
    sequence_count: int


@dataclass(frozen=True)
class SwissProtConceptSet:
    """Parsed local Swiss-Prot concepts and coverage metadata."""

    concepts: list[ConceptDefinition]
    coverage_rows: list[dict[str, Any]]
    summary: dict[str, Any]


class ResidueFeatureActivationAccumulator:
    """Max activation per sequence residue and feature.

    The analysis stream may contain repeated samples and timesteps for the same
    residue. Concept analysis collapses those to one max activation per residue
    so repeated BioEmu/ESM3 observations do not inflate counts. Runtime streams
    use sparse dictionaries; shared evaluation can use bounded dense float16
    matrices.
    """

    def __init__(self, sequences: dict[str, str], dense_storage: bool = False) -> None:
        self.sequences = dict(sequences)
        self.dense_storage = dense_storage
        self.offset_by_sequence: dict[str, int] = {}
        offset = 0
        for sequence_id, sequence in self.sequences.items():
            self.offset_by_sequence[sequence_id] = offset
            offset += len(sequence)
        self.total_residues = offset
        self.max_by_feature: list[dict[int, float]] | None = None
        self.dense_max_by_feature: np.ndarray | None = None
        self.dense_argmax_sample_by_feature: np.ndarray | None = None
        self.dense_argmax_timestep_by_feature: np.ndarray | None = None
        self.feature_max: np.ndarray | None = None

    @property
    def has_data(self) -> bool:
        return self.feature_max is not None and (
            self.max_by_feature is not None or self.dense_max_by_feature is not None
        )

    @classmethod
    def from_dense_feature_maxima(
        cls,
        sequences: dict[str, str],
        dense_max_by_feature: np.ndarray,
        dense_argmax_sample_by_feature: np.ndarray | None = None,
    ) -> ResidueFeatureActivationAccumulator:
        accumulator = cls(sequences)
        values = np.asarray(dense_max_by_feature)
        if values.ndim != 2 or values.shape[1] != accumulator.total_residues:
            raise ValueError("dense feature maxima must have shape [features, total_residues]")
        accumulator.dense_max_by_feature = values
        if dense_argmax_sample_by_feature is not None:
            argmax = np.asarray(dense_argmax_sample_by_feature)
            if argmax.shape != values.shape:
                raise ValueError("dense argmax samples must match dense feature maxima")
            accumulator.dense_argmax_sample_by_feature = argmax
        accumulator.feature_max = values.max(axis=1).astype(np.float32, copy=False)
        return accumulator

    def feature_residue_values(self, feature_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if self.dense_max_by_feature is not None:
            values = self.dense_max_by_feature[feature_idx].astype(np.float32, copy=False)
            indices = np.flatnonzero(values > 0)
            return indices.astype(np.int64, copy=False), values[indices]
        if self.max_by_feature is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        residue_values = self.max_by_feature[feature_idx]
        return (
            np.fromiter(residue_values.keys(), dtype=np.int64),
            np.fromiter(residue_values.values(), dtype=np.float32),
        )

    def accumulate(
        self,
        latents: torch.Tensor,
        record: ActivationRecord,
        start: int,
        end: int,
    ) -> None:
        values = latents.float()
        if self.dense_storage:
            self._accumulate_dense(values, record, start, end)
            return
        if self.max_by_feature is None:
            n_features = values.shape[1]
            self.max_by_feature = [dict() for _ in range(n_features)]
            self.feature_max = np.zeros(n_features, dtype=np.float32)

        assert self.max_by_feature is not None
        assert self.feature_max is not None
        sequence_ids = record.sequence_ids[start:end]
        residues = record.residue_indices[start:end].detach().cpu().tolist()
        active = torch.nonzero(values > 0, as_tuple=False)
        for token_idx_tensor, feature_idx_tensor in active:
            token_idx = int(token_idx_tensor.item())
            sequence_id = sequence_ids[token_idx]
            offset = self.offset_by_sequence.get(sequence_id)
            if offset is None:
                continue
            residue_idx = int(residues[token_idx])
            if residue_idx < 0 or residue_idx >= len(self.sequences[sequence_id]):
                continue
            feature_idx = int(feature_idx_tensor.item())
            activation = float(values[token_idx, feature_idx].item())
            global_idx = offset + residue_idx
            previous = self.max_by_feature[feature_idx].get(global_idx)
            if previous is None or activation > previous:
                self.max_by_feature[feature_idx][global_idx] = activation
            if activation > self.feature_max[feature_idx]:
                self.feature_max[feature_idx] = activation

    def _accumulate_dense(
        self,
        values: torch.Tensor,
        record: ActivationRecord,
        start: int,
        end: int,
    ) -> None:
        n_features = values.shape[1]
        if self.dense_max_by_feature is None:
            self.dense_max_by_feature = np.zeros(
                (n_features, self.total_residues),
                dtype=np.float16,
            )
            self.dense_argmax_sample_by_feature = np.zeros(
                (n_features, self.total_residues),
                dtype=np.uint8,
            )
            self.dense_argmax_timestep_by_feature = np.zeros(
                (n_features, self.total_residues),
                dtype=np.uint8,
            )
            self.feature_max = np.zeros(n_features, dtype=np.float32)

        assert self.dense_max_by_feature is not None
        assert self.dense_argmax_sample_by_feature is not None
        assert self.dense_argmax_timestep_by_feature is not None
        assert self.feature_max is not None
        active = torch.nonzero(values > 0, as_tuple=False)
        if active.numel() == 0:
            return
        token_indices = active[:, 0].numpy()
        feature_indices = active[:, 1].numpy()
        sequence_ids = record.sequence_ids[start:end]
        residues = record.residue_indices[start:end].detach().cpu().numpy()
        if len(set(sequence_ids)) == 1:
            global_residues = residues + self.offset_by_sequence[sequence_ids[0]]
        else:
            offsets = np.fromiter(
                (self.offset_by_sequence[sequence_id] for sequence_id in sequence_ids),
                dtype=np.int64,
                count=len(sequence_ids),
            )
            global_residues = residues + offsets
        flat_indices = feature_indices * self.total_residues + global_residues[token_indices]
        activations = values[active[:, 0], active[:, 1]].numpy().astype(np.float16, copy=False)
        flat_values = self.dense_max_by_feature.reshape(-1)
        previous_values = flat_values[flat_indices].copy()
        np.maximum.at(flat_values, flat_indices, activations)
        improved = np.logical_and(
            activations > previous_values,
            activations == flat_values[flat_indices],
        )

        sample_indices = record.sample_indices
        if sample_indices is None:
            samples = np.zeros(len(sequence_ids), dtype=np.uint8)
        else:
            samples = sample_indices[start:end].detach().cpu().numpy().astype(np.uint8)
        timesteps = record.timesteps[start:end].detach().cpu().numpy()
        encoded_timesteps = np.clip(np.rint(timesteps * 255.0), 0, 255).astype(np.uint8)
        improved_tokens = token_indices[improved]
        improved_flat = flat_indices[improved]
        self.dense_argmax_sample_by_feature.reshape(-1)[improved_flat] = samples[improved_tokens]
        self.dense_argmax_timestep_by_feature.reshape(-1)[improved_flat] = encoded_timesteps[
            improved_tokens
        ]
        np.maximum.at(self.feature_max, feature_indices, activations.astype(np.float32))


def analyze_swissprot_concepts(
    accumulator: ResidueFeatureActivationAccumulator | None,
    annotations_path: str | Path | None,
    sequences: dict[str, str],
    output_dir: str | Path,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    min_positive_residues: int = 10,
    min_positive_domains: int = 2,
    top_k: int = 30,
) -> dict[str, Any] | None:
    """Write Swiss-Prot concept/domain F1 artifacts from one activation pass."""

    if annotations_path is None:
        return None
    if accumulator is None or not accumulator.has_data:
        return None
    annotations_path = Path(annotations_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not annotations_path.exists():
        report = {
            "status": "skipped_missing_annotations",
            "annotations_path": str(annotations_path),
        }
        report_path = output_dir / "swissprot_concept_report.json"
        write_json(report_path, report)
        report["report_path"] = str(report_path)
        return report

    concept_set = load_swissprot_concepts(
        annotations_path=annotations_path,
        sequences=sequences,
        min_positive_residues=min_positive_residues,
        min_positive_domains=min_positive_domains,
    )
    coverage_csv = output_dir / "coverage_report.csv"
    coverage_json = output_dir / "coverage_report.json"
    concept_summary_csv = output_dir / "concept_summary.csv"
    concept_summary_json = output_dir / "concept_summary.json"
    write_dict_rows_csv(coverage_csv, concept_set.coverage_rows)
    write_json(coverage_json, concept_set.coverage_rows)
    concept_summary_rows = [
        {
            "concept": concept.name,
            "is_aa_level": concept.is_aa_level,
            "positive_residues": concept.positive_residues,
            "positive_domains": concept.positive_domains,
            "sequence_count": concept.sequence_count,
        }
        for concept in concept_set.concepts
    ]
    write_dict_rows_csv(concept_summary_csv, concept_summary_rows)
    write_json(concept_summary_json, concept_summary_rows)

    rows = calculate_concept_f1_rows(
        accumulator=accumulator,
        concepts=concept_set.concepts,
        thresholds=thresholds,
    )
    f1_csv = output_dir / "concept_f1_scores.csv"
    f1_json = output_dir / "concept_f1_scores.json"
    write_dict_rows_csv(f1_csv, rows)
    write_json(f1_json, rows)

    best_domain_rows = best_rows_by_concept(rows, metric="f1_per_domain")
    best_residue_rows = best_rows_by_concept(rows, metric="f1")
    domain_plot = output_dir / "top_domain_f1.png"
    residue_plot = output_dir / "top_residue_f1.png"
    write_metric_bar_plot(
        best_domain_rows[:top_k],
        path=domain_plot,
        metric="f1_per_domain",
        title="Top Swiss-Prot Domain-Level Concept F1",
    )
    write_plot_sidecars(domain_plot, best_domain_rows[:top_k])
    write_metric_bar_plot(
        best_residue_rows[:top_k],
        path=residue_plot,
        metric="f1",
        title="Top Swiss-Prot Residue-Level Concept F1",
    )
    write_plot_sidecars(residue_plot, best_residue_rows[:top_k])

    report = {
        "annotations_path": str(annotations_path),
        "coverage": concept_set.summary,
        "coverage_csv_path": str(coverage_csv),
        "coverage_json_path": str(coverage_json),
        "concept_summary_csv_path": str(concept_summary_csv),
        "concept_summary_json_path": str(concept_summary_json),
        "f1_csv_path": str(f1_csv),
        "f1_json_path": str(f1_json),
        "top_domain_f1_plot_path": str(domain_plot),
        "top_domain_f1_csv_path": str(domain_plot.with_suffix(".csv")),
        "top_domain_f1_json_path": str(domain_plot.with_suffix(".json")),
        "top_residue_f1_plot_path": str(residue_plot),
        "top_residue_f1_csv_path": str(residue_plot.with_suffix(".csv")),
        "top_residue_f1_json_path": str(residue_plot.with_suffix(".json")),
        "thresholds": list(thresholds),
        "activation_aggregation": "max per sequence residue and feature",
        "activation_normalization": "per-feature max over aggregated residue activations",
        "num_concepts": len(concept_set.concepts),
        "num_rows": len(rows),
    }
    report_path = output_dir / "swissprot_concept_report.json"
    write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def analyze_interpro_concepts(
    accumulator: ResidueFeatureActivationAccumulator | None,
    annotations_path: str | Path | None,
    sequences: dict[str, str],
    output_dir: str | Path,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    min_positive_residues: int = 10,
    min_positive_domains: int = 10,
    min_positive_sequences: int = 10,
    top_k: int = 30,
) -> dict[str, Any] | None:
    """Write InterPro/Pfam concept/domain F1 artifacts from one activation pass."""

    if annotations_path is None:
        return None
    if accumulator is None or not accumulator.has_data:
        return None
    annotations_path = Path(annotations_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not annotations_path.exists():
        report = {
            "status": "skipped_missing_annotations",
            "annotations_path": str(annotations_path),
        }
        report_path = output_dir / "interpro_concept_report.json"
        write_json(report_path, report)
        report["report_path"] = str(report_path)
        return report

    concept_set = load_interpro_concepts(
        annotations_path=annotations_path,
        sequences=sequences,
        min_positive_residues=min_positive_residues,
        min_positive_domains=min_positive_domains,
        min_positive_sequences=min_positive_sequences,
    )
    coverage_csv = output_dir / "coverage_report.csv"
    coverage_json = output_dir / "coverage_report.json"
    concept_summary_csv = output_dir / "concept_summary.csv"
    concept_summary_json = output_dir / "concept_summary.json"
    write_dict_rows_csv(coverage_csv, concept_set.coverage_rows)
    write_json(coverage_json, concept_set.coverage_rows)
    concept_summary_rows = [
        {
            "concept": concept.name,
            "is_aa_level": concept.is_aa_level,
            "positive_residues": concept.positive_residues,
            "positive_domains": concept.positive_domains,
            "sequence_count": concept.sequence_count,
        }
        for concept in concept_set.concepts
    ]
    write_dict_rows_csv(concept_summary_csv, concept_summary_rows)
    write_json(concept_summary_json, concept_summary_rows)

    rows = calculate_concept_f1_rows(
        accumulator=accumulator,
        concepts=concept_set.concepts,
        thresholds=thresholds,
    )
    f1_csv = output_dir / "concept_f1_scores.csv"
    f1_json = output_dir / "concept_f1_scores.json"
    write_dict_rows_csv(f1_csv, rows)
    write_json(f1_json, rows)

    fold_rows, best_domain_rows = calculate_cross_validated_concept_rows(
        accumulator=accumulator,
        concepts=concept_set.concepts,
        thresholds=thresholds,
        full_data_rows=rows,
        n_splits=5,
    )
    cv_csv = output_dir / "concept_cross_validation.csv"
    cv_json = output_dir / "concept_cross_validation.json"
    cv_summary_csv = output_dir / "concept_cross_validation_summary.csv"
    cv_summary_json = output_dir / "concept_cross_validation_summary.json"
    write_dict_rows_csv(cv_csv, fold_rows)
    write_json(cv_json, fold_rows)
    write_dict_rows_csv(cv_summary_csv, best_domain_rows)
    write_json(cv_summary_json, best_domain_rows)
    domain_plot = output_dir / "top_domain_f1.png"
    write_metric_bar_plot(
        best_domain_rows[:top_k],
        path=domain_plot,
        metric="f1_per_domain",
        title="Top InterPro/Pfam Domain-Level Concept F1",
        empty_label="No InterPro/Pfam concept hits",
    )
    write_plot_sidecars(domain_plot, best_domain_rows[:top_k])

    report = {
        "annotations_path": str(annotations_path),
        "coverage": concept_set.summary,
        "coverage_csv_path": str(coverage_csv),
        "coverage_json_path": str(coverage_json),
        "concept_summary_csv_path": str(concept_summary_csv),
        "concept_summary_json_path": str(concept_summary_json),
        "f1_csv_path": str(f1_csv),
        "f1_json_path": str(f1_json),
        "cross_validation_csv_path": str(cv_csv),
        "cross_validation_json_path": str(cv_json),
        "cross_validation_summary_csv_path": str(cv_summary_csv),
        "cross_validation_summary_json_path": str(cv_summary_json),
        "top_domain_f1_plot_path": str(domain_plot),
        "top_domain_f1_csv_path": str(domain_plot.with_suffix(".csv")),
        "top_domain_f1_json_path": str(domain_plot.with_suffix(".json")),
        "thresholds": list(thresholds),
        "activation_aggregation": "max per sequence residue and feature",
        "activation_normalization": "per-feature max over aggregated residue activations",
        "reported_metric_scope": "five-fold held-out sequences",
        "localization_feature_scope": "refit on all sequences after cross-validation",
        "cross_validation_folds": 5,
        "num_concepts": len(concept_set.concepts),
        "num_rows": len(rows),
    }
    report_path = output_dir / "interpro_concept_report.json"
    write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def calculate_cross_validated_concept_rows(
    accumulator: ResidueFeatureActivationAccumulator,
    concepts: list[ConceptDefinition],
    thresholds: tuple[float, ...],
    full_data_rows: list[dict[str, Any]],
    n_splits: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select feature/threshold on training sequences and score held-out sequences."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not accumulator.has_data:
        return [], []
    full_best = {
        str(row["concept"]): row
        for row in best_rows_by_concept(full_data_rows, metric="f1_per_domain")
    }
    fold_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for concept in concepts:
        labels = global_label_array(accumulator, concept)
        folds = concept_sequence_folds(accumulator, concept, n_splits)
        concept_fold_rows = []
        for fold_idx in range(n_splits):
            test_mask = folds == fold_idx
            train_mask = np.logical_not(test_mask)
            selected = select_concept_feature_on_mask(
                accumulator=accumulator,
                labels=labels,
                eligible_mask=train_mask,
                thresholds=thresholds,
                concept=concept,
            )
            if selected is None:
                continue
            evaluated = evaluate_concept_feature_on_mask(
                accumulator=accumulator,
                labels=labels,
                eligible_mask=test_mask,
                feature_idx=int(selected["feature"]),
                threshold=float(selected["threshold"]),
                normalization_max=float(selected["normalization_max"]),
                concept=concept,
            )
            row = {
                "concept": concept.name,
                "fold": fold_idx,
                "selected_feature": int(selected["feature"]),
                "selected_threshold": float(selected["threshold"]),
                "training_f1_per_domain": float(selected["f1_per_domain"]),
                **evaluated,
            }
            fold_rows.append(row)
            concept_fold_rows.append(row)
        if not concept_fold_rows:
            continue
        tp = sum(int(row["tp"]) for row in concept_fold_rows)
        fp = sum(int(row["fp"]) for row in concept_fold_rows)
        positive_residues = sum(int(row["positive_residues"]) for row in concept_fold_rows)
        tp_domains = sum(int(row["tp_per_domain"]) for row in concept_fold_rows)
        positive_domains = sum(int(row["positive_domains"]) for row in concept_fold_rows)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(positive_residues, 1)
        recall_per_domain = tp_domains / max(positive_domains, 1)
        refit = full_best.get(concept.name)
        if refit is None:
            continue
        aggregate_rows.append(
            {
                "concept": concept.name,
                "refit_feature": int(refit["feature"]),
                "refit_threshold": float(refit["threshold"]),
                "precision": float(precision),
                "recall": float(recall),
                "recall_per_domain": float(recall_per_domain),
                "f1": float(f1_score(precision, recall)),
                "f1_per_domain": float(f1_score(precision, recall_per_domain)),
                "tp": tp,
                "fp": fp,
                "fn": positive_residues - tp,
                "tp_per_domain": tp_domains,
                "fn_per_domain": positive_domains - tp_domains,
                "positive_residues": positive_residues,
                "positive_domains": positive_domains,
                "positive_sequences": concept.sequence_count,
                "cv_selected_features": sorted(
                    {int(row["selected_feature"]) for row in concept_fold_rows}
                ),
                "cv_folds": len(concept_fold_rows),
                "evaluation_scope": "held_out_sequences",
                "metric_scope": "cross_validated_feature_selection_procedure",
                "localization_feature_scope": "all_data_refit_not_scored_by_held_out_metric",
            }
        )
    aggregate_rows.sort(key=lambda row: float(row["f1_per_domain"]), reverse=True)
    return fold_rows, aggregate_rows


def concept_sequence_folds(
    accumulator: ResidueFeatureActivationAccumulator,
    concept: ConceptDefinition,
    n_splits: int,
) -> np.ndarray:
    positive_sequences = [
        sequence_id
        for sequence_id, labels in concept.labels_by_sequence.items()
        if bool(np.any(labels > 0))
    ]
    positive_set = set(positive_sequences)
    negative_sequences = [
        sequence_id for sequence_id in accumulator.sequences if sequence_id not in positive_set
    ]
    fold_by_sequence: dict[str, int] = {}
    for sequence_ids in (positive_sequences, negative_sequences):
        ordered = sorted(sequence_ids, key=stable_sequence_key)
        for index, sequence_id in enumerate(ordered):
            fold_by_sequence[sequence_id] = index % n_splits
    folds = np.empty(accumulator.total_residues, dtype=np.int16)
    for sequence_id, sequence in accumulator.sequences.items():
        offset = accumulator.offset_by_sequence[sequence_id]
        folds[offset : offset + len(sequence)] = fold_by_sequence[sequence_id]
    return folds


def stable_sequence_key(sequence_id: str) -> bytes:
    return hashlib.sha256(sequence_id.encode("utf-8")).digest()


def select_concept_feature_on_mask(
    accumulator: ResidueFeatureActivationAccumulator,
    labels: np.ndarray,
    eligible_mask: np.ndarray,
    thresholds: tuple[float, ...],
    concept: ConceptDefinition,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    assert accumulator.feature_max is not None
    for feature_idx in range(accumulator.feature_max.size):
        residue_indices, values = accumulator.feature_residue_values(feature_idx)
        if residue_indices.size == 0:
            continue
        train_values = values[eligible_mask[residue_indices]]
        if train_values.size == 0:
            continue
        normalization_max = float(train_values.max())
        if normalization_max <= 0:
            continue
        for threshold in thresholds:
            metrics = evaluate_sparse_concept_values(
                residue_indices=residue_indices,
                values=values,
                labels=labels,
                eligible_mask=eligible_mask,
                threshold=float(threshold),
                normalization_max=normalization_max,
                concept=concept,
            )
            candidate = {
                "feature": feature_idx,
                "threshold": float(threshold),
                "normalization_max": normalization_max,
                **metrics,
            }
            score = (float(candidate["f1_per_domain"]), float(candidate["f1"]))
            if best is None or score > (float(best["f1_per_domain"]), float(best["f1"])):
                best = candidate
    return best


def evaluate_concept_feature_on_mask(
    accumulator: ResidueFeatureActivationAccumulator,
    labels: np.ndarray,
    eligible_mask: np.ndarray,
    feature_idx: int,
    threshold: float,
    normalization_max: float,
    concept: ConceptDefinition,
) -> dict[str, Any]:
    residue_indices, values = accumulator.feature_residue_values(feature_idx)
    return evaluate_sparse_concept_values(
        residue_indices=residue_indices,
        values=values,
        labels=labels,
        eligible_mask=eligible_mask,
        threshold=threshold,
        normalization_max=normalization_max,
        concept=concept,
    )


def evaluate_sparse_concept_values(
    residue_indices: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    eligible_mask: np.ndarray,
    threshold: float,
    normalization_max: float,
    concept: ConceptDefinition,
) -> dict[str, Any]:
    active = np.logical_and(
        eligible_mask[residue_indices],
        values / max(normalization_max, 1e-12) > threshold,
    )
    active_indices = residue_indices[active]
    active_labels = labels[active_indices]
    positive_mask = active_labels > 0
    tp = int(positive_mask.sum())
    fp = int(active_indices.size - tp)
    eligible_labels = labels[eligible_mask]
    positive_residues = int((eligible_labels > 0).sum())
    positive_domains = int(np.unique(eligible_labels[eligible_labels > 0]).size)
    tp_domains = tp if concept.is_aa_level else int(np.unique(active_labels[positive_mask]).size)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(positive_residues, 1)
    recall_per_domain = tp_domains / max(positive_domains, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "recall_per_domain": float(recall_per_domain),
        "f1": float(f1_score(precision, recall)),
        "f1_per_domain": float(f1_score(precision, recall_per_domain)),
        "tp": tp,
        "fp": fp,
        "fn": positive_residues - tp,
        "tp_per_domain": tp_domains,
        "fn_per_domain": positive_domains - tp_domains,
        "positive_residues": positive_residues,
        "positive_domains": positive_domains,
    }


def calculate_concept_f1_rows(
    accumulator: ResidueFeatureActivationAccumulator,
    concepts: list[ConceptDefinition],
    thresholds: tuple[float, ...],
) -> list[dict[str, Any]]:
    assert accumulator.feature_max is not None
    label_arrays = {concept.name: global_label_array(accumulator, concept) for concept in concepts}
    rows: list[dict[str, Any]] = []
    for feature_idx in range(accumulator.feature_max.size):
        residue_indices, feature_values = accumulator.feature_residue_values(feature_idx)
        if residue_indices.size == 0:
            continue
        feature_max = float(accumulator.feature_max[feature_idx])
        if feature_max <= 0:
            continue
        normalized = feature_values / feature_max
        for threshold in thresholds:
            active_indices = residue_indices[normalized > float(threshold)]
            if active_indices.size == 0:
                continue
            for concept in concepts:
                labels = label_arrays[concept.name][active_indices]
                positive_mask = labels > 0
                true_positive = int(positive_mask.sum())
                if true_positive == 0:
                    continue
                false_positive = int(active_indices.size - true_positive)
                precision = true_positive / max(true_positive + false_positive, 1)
                recall = true_positive / max(concept.positive_residues, 1)
                if concept.is_aa_level:
                    true_positive_domains = true_positive
                    recall_per_domain = recall
                else:
                    true_positive_domains = int(np.unique(labels[positive_mask]).size)
                    recall_per_domain = true_positive_domains / max(concept.positive_domains, 1)
                rows.append(
                    {
                        "concept": concept.name,
                        "feature": feature_idx,
                        "threshold": float(threshold),
                        "precision": float(precision),
                        "recall": float(recall),
                        "recall_per_domain": float(recall_per_domain),
                        "f1": float(f1_score(precision, recall)),
                        "f1_per_domain": float(f1_score(precision, recall_per_domain)),
                        "tp": true_positive,
                        "fp": false_positive,
                        "tp_per_domain": true_positive_domains,
                        "positive_residues": concept.positive_residues,
                        "positive_domains": concept.positive_domains,
                        "positive_sequences": concept.sequence_count,
                        "is_aa_level_concept": concept.is_aa_level,
                    }
                )
    return sorted(rows, key=lambda row: float(row["f1_per_domain"]), reverse=True)


def load_interpro_concepts(
    annotations_path: str | Path,
    sequences: dict[str, str],
    min_positive_residues: int = 10,
    min_positive_domains: int = 2,
    min_positive_sequences: int = 2,
) -> SwissProtConceptSet:
    rows = read_jsonl_rows(annotations_path)
    rows_by_sequence: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sequence_id = str(row.get("sequence_id", ""))
        if sequence_id:
            rows_by_sequence[sequence_id].append(row)
    rows_by_accession: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        accession = str(row.get("accession", ""))
        if accession:
            rows_by_accession[accession].append(row)

    concept_labels: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    concept_is_aa_level: dict[str, bool] = {}
    domain_counters: defaultdict[str, int] = defaultdict(int)
    coverage_rows: list[dict[str, Any]] = []

    for sequence_id, sequence in sequences.items():
        accession = accession_from_sequence_id(sequence_id)
        sequence_rows = rows_by_sequence.get(sequence_id) or rows_by_accession.get(accession, [])
        if not sequence_rows:
            coverage_rows.append(
                {
                    "sequence_id": sequence_id,
                    "accession": accession,
                    "status": "missing_annotation",
                    "sequence_length": len(sequence),
                    "num_intervals": 0,
                }
            )
            continue
        matched = 0
        for row in sequence_rows:
            start = safe_int(row.get("start"))
            end = safe_int(row.get("end"))
            if start is None or end is None:
                continue
            source = clean_concept_name(str(row.get("source_database", "interpro")).lower())
            entry_accession = clean_concept_name(str(row.get("entry_accession", "unknown")))
            entry_name = clean_concept_name(str(row.get("name", "unknown")))
            concept_name = f"{source}:{entry_accession}:{entry_name}"
            add_interval_concept(
                concept_labels=concept_labels,
                concept_is_aa_level=concept_is_aa_level,
                domain_counters=domain_counters,
                sequence_id=sequence_id,
                sequence_length=len(sequence),
                concept_name=concept_name,
                start=start - 1,
                end=end - 1,
                is_aa_level=False,
            )
            matched += 1
        coverage_rows.append(
            {
                "sequence_id": sequence_id,
                "accession": accession,
                "status": "matched" if matched else "missing_valid_intervals",
                "sequence_length": len(sequence),
                "num_intervals": matched,
            }
        )

    concepts = []
    for concept_name, labels_by_sequence in concept_labels.items():
        positive_residues = int(sum((labels > 0).sum() for labels in labels_by_sequence.values()))
        domain_ids = {
            int(value)
            for labels in labels_by_sequence.values()
            for value in np.unique(labels)
            if int(value) > 0
        }
        positive_domains = len(domain_ids)
        sequence_count = sum(bool((labels > 0).any()) for labels in labels_by_sequence.values())
        if positive_residues < min_positive_residues:
            continue
        if positive_domains < min_positive_domains:
            continue
        if sequence_count < min_positive_sequences:
            continue
        concepts.append(
            ConceptDefinition(
                name=concept_name,
                labels_by_sequence=labels_by_sequence,
                is_aa_level=concept_is_aa_level.get(concept_name, False),
                positive_residues=positive_residues,
                positive_domains=positive_domains,
                sequence_count=sequence_count,
            )
        )
    coverage_counts = defaultdict(int)
    for row in coverage_rows:
        coverage_counts[str(row["status"])] += 1
    summary = {
        "num_sequences": len(sequences),
        "num_annotation_rows": len(rows),
        "num_concepts": len(concepts),
        "coverage_counts": dict(coverage_counts),
        "matched_fraction": (coverage_counts["matched"] / len(sequences) if sequences else 0.0),
    }
    return SwissProtConceptSet(
        concepts=sorted(concepts, key=lambda concept: concept.name),
        coverage_rows=coverage_rows,
        summary=summary,
    )


def load_swissprot_concepts(
    annotations_path: str | Path,
    sequences: dict[str, str],
    min_positive_residues: int = 10,
    min_positive_domains: int = 2,
) -> SwissProtConceptSet:
    rows = read_tsv_rows(annotations_path)
    annotation_by_accession = {str(row.get("Entry", "")): row for row in rows}
    concept_labels: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    concept_is_aa_level: dict[str, bool] = {}
    coverage_rows: list[dict[str, Any]] = []
    domain_counters: defaultdict[str, int] = defaultdict(int)

    for sequence_id, sequence in sequences.items():
        accession = accession_from_sequence_id(sequence_id)
        row = annotation_by_accession.get(accession)
        if row is None:
            coverage_rows.append(
                {
                    "sequence_id": sequence_id,
                    "accession": accession,
                    "status": "missing_annotation",
                    "sequence_length": len(sequence),
                }
            )
            continue
        swiss_sequence = str(row.get("Sequence", ""))
        if swiss_sequence and swiss_sequence != sequence:
            status = "sequence_mismatch"
        else:
            status = "matched"
        coverage_rows.append(
            {
                "sequence_id": sequence_id,
                "accession": accession,
                "status": status,
                "sequence_length": len(sequence),
                "annotation_length": len(swiss_sequence) if swiss_sequence else "",
            }
        )
        if status != "matched":
            continue
        add_amino_acid_concepts(concept_labels, concept_is_aa_level, sequence_id, sequence)
        for column, code, category_key, is_aa_level in CATEGORICAL_FEATURES:
            column_value = get_column(row, column)
            for start, end, category in parse_feature_entries(column_value, code, category_key):
                add_interval_concept(
                    concept_labels=concept_labels,
                    concept_is_aa_level=concept_is_aa_level,
                    domain_counters=domain_counters,
                    sequence_id=sequence_id,
                    sequence_length=len(sequence),
                    concept_name=f"{clean_concept_name(column)}:any",
                    start=start,
                    end=end,
                    is_aa_level=is_aa_level,
                )
                if category:
                    add_interval_concept(
                        concept_labels=concept_labels,
                        concept_is_aa_level=concept_is_aa_level,
                        domain_counters=domain_counters,
                        sequence_id=sequence_id,
                        sequence_length=len(sequence),
                        concept_name=f"{clean_concept_name(column)}:{category}",
                        start=start,
                        end=end,
                        is_aa_level=is_aa_level,
                    )
        for column, code, is_aa_level in BINARY_FEATURES:
            for start, end, _ in parse_feature_entries(get_column(row, column), code, "note"):
                add_interval_concept(
                    concept_labels=concept_labels,
                    concept_is_aa_level=concept_is_aa_level,
                    domain_counters=domain_counters,
                    sequence_id=sequence_id,
                    sequence_length=len(sequence),
                    concept_name=clean_concept_name(column),
                    start=start,
                    end=end,
                    is_aa_level=is_aa_level,
                )
        for column, code, is_aa_level in INTERACTION_FEATURES:
            for start, end, _ in parse_feature_entries(get_column(row, column), code, "note"):
                add_interval_concept(
                    concept_labels=concept_labels,
                    concept_is_aa_level=concept_is_aa_level,
                    domain_counters=domain_counters,
                    sequence_id=sequence_id,
                    sequence_length=len(sequence),
                    concept_name=clean_concept_name(column),
                    start=start,
                    end=end,
                    is_aa_level=is_aa_level,
                )

    concepts = []
    for concept_name, labels_by_sequence in concept_labels.items():
        positive_residues = int(sum((labels > 0).sum() for labels in labels_by_sequence.values()))
        domain_ids = {
            int(value)
            for labels in labels_by_sequence.values()
            for value in np.unique(labels)
            if int(value) > 0
        }
        positive_domains = len(domain_ids)
        sequence_count = sum(bool((labels > 0).any()) for labels in labels_by_sequence.values())
        is_aa_level = concept_is_aa_level.get(concept_name, False)
        if positive_residues < min_positive_residues:
            continue
        if not is_aa_level and positive_domains < min_positive_domains:
            continue
        concepts.append(
            ConceptDefinition(
                name=concept_name,
                labels_by_sequence=labels_by_sequence,
                is_aa_level=is_aa_level,
                positive_residues=positive_residues,
                positive_domains=positive_domains,
                sequence_count=sequence_count,
            )
        )
    coverage_counts = defaultdict(int)
    for row in coverage_rows:
        coverage_counts[str(row["status"])] += 1
    summary = {
        "num_sequences": len(sequences),
        "num_annotation_rows": len(rows),
        "num_concepts": len(concepts),
        "coverage_counts": dict(coverage_counts),
        "matched_fraction": (coverage_counts["matched"] / len(sequences) if sequences else 0.0),
    }
    return SwissProtConceptSet(
        concepts=sorted(concepts, key=lambda concept: concept.name),
        coverage_rows=coverage_rows,
        summary=summary,
    )


def add_amino_acid_concepts(
    concept_labels: dict[str, dict[str, np.ndarray]],
    concept_is_aa_level: dict[str, bool],
    sequence_id: str,
    sequence: str,
) -> None:
    for amino_acid in AMINO_ACIDS:
        labels = np.zeros(len(sequence), dtype=np.int32)
        labels[[idx for idx, aa in enumerate(sequence) if aa == amino_acid]] = 1
        concept_name = f"amino_acid:{amino_acid}"
        concept_labels[concept_name][sequence_id] = labels
        concept_is_aa_level[concept_name] = True


def add_interval_concept(
    concept_labels: dict[str, dict[str, np.ndarray]],
    concept_is_aa_level: dict[str, bool],
    domain_counters: defaultdict[str, int],
    sequence_id: str,
    sequence_length: int,
    concept_name: str,
    start: int,
    end: int,
    is_aa_level: bool,
) -> None:
    start = max(start, 0)
    end = min(end, sequence_length - 1)
    if start > end:
        return
    labels = concept_labels[concept_name].setdefault(
        sequence_id,
        np.zeros(sequence_length, dtype=np.int32),
    )
    if is_aa_level:
        labels[start : end + 1] = 1
    else:
        domain_counters[concept_name] += 1
        labels[start : end + 1] = domain_counters[concept_name]
    concept_is_aa_level[concept_name] = is_aa_level


def parse_feature_entries(
    value: str | None,
    code: str,
    category_key: str,
) -> list[tuple[int, int, str]]:
    if not value or value == "nan":
        return []
    entries = str(value).split(f"{code} ")[1:]
    parsed = []
    for entry in entries:
        location = entry.split(";")[0].strip()
        if ":" in location or "?" in location:
            continue
        interval = parse_location(location)
        if interval is None:
            continue
        category = extract_category(entry, category_key)
        parsed.append((interval[0], interval[1], category))
    return parsed


def parse_location(location: str) -> tuple[int, int] | None:
    location = location.strip().replace("<", "").replace(">", "")
    if ".." in location:
        start_text, end_text = location.split("..", maxsplit=1)
    else:
        start_text = end_text = location
    if not start_text.isdigit() or not end_text.isdigit():
        return None
    return int(start_text) - 1, int(end_text) - 1


def extract_category(entry: str, category_key: str) -> str:
    patterns = [
        rf'/{re.escape(category_key)}="([^"]+)"',
        rf"/{re.escape(category_key)}=([^;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, entry)
        if match:
            return clean_concept_name(match.group(1).split(";")[0])
    return ""


def global_label_array(
    accumulator: ResidueFeatureActivationAccumulator,
    concept: ConceptDefinition,
) -> np.ndarray:
    labels = np.zeros(accumulator.total_residues, dtype=np.int32)
    for sequence_id, sequence_labels in concept.labels_by_sequence.items():
        offset = accumulator.offset_by_sequence.get(sequence_id)
        if offset is None:
            continue
        end = min(offset + len(sequence_labels), labels.size)
        labels[offset:end] = sequence_labels[: end - offset]
    return labels


def best_rows_by_concept(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        concept = str(row["concept"])
        if concept not in best or float(row[metric]) > float(best[concept][metric]):
            best[concept] = row
    return sorted(best.values(), key=lambda row: float(row[metric]), reverse=True)


def write_metric_bar_plot(
    rows: list[dict[str, Any]],
    path: Path,
    metric: str,
    title: str,
    empty_label: str = "No Swiss-Prot concept hits",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.28 * len(rows))))
    if not rows:
        ax.text(0.5, 0.5, empty_label, ha="center", va="center")
        ax.axis("off")
    else:
        labels = []
        for row in rows:
            feature = row.get("refit_feature", row.get("feature"))
            threshold = row.get("refit_threshold", row.get("threshold"))
            suffix = (
                f" | refit f{int(feature)} @ {float(threshold):.2g}"
                if feature is not None and threshold is not None
                else ""
            )
            labels.append(f"{row['concept']}{suffix}")
        values = [float(row[metric]) for row in rows]
        y = np.arange(len(rows))
        ax.barh(y, values, color="#4c78a8")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(metric)
        ax.set_xlim(0, max(0.05, min(1.0, max(values) * 1.15)))
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_plot_sidecars(path: Path, rows: list[dict[str, Any]]) -> None:
    write_dict_rows_csv(path.with_suffix(".csv"), rows)
    write_json(path.with_suffix(".json"), rows)


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def read_tsv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_dict_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def get_column(row: dict[str, str], name: str) -> str:
    if name in row:
        return row[name]
    normalized = clean_header(name)
    for key, value in row.items():
        if clean_header(key) == normalized:
            return value
    return ""


def accession_from_sequence_id(sequence_id: str) -> str:
    if "|" in sequence_id:
        parts = sequence_id.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    value = sequence_id
    for prefix in ("UniRef50_", "UniRef50-", "UniRef90_", "UniRef90-", "UniRef100_", "UniRef100-"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    return value.split()[0].split("/")[0]


def clean_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def clean_concept_name(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    value = value.replace(" [FT]", "")
    value = re.sub(r"[^A-Za-z0-9_.:+ -]+", "", value)
    return value.replace(" ", "_")[:120] or "unknown"


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
