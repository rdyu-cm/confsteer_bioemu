from __future__ import annotations

import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from sae_bioemu.bioemu.extractor import BioEmuRuntimeActivationSampler, SequenceRecord, read_fasta
from sae_bioemu.core.checkpoint_io import (
    load_sae_checkpoint,
    prepare_sae_checkpoint_payload,
)
from sae_bioemu.core.config import ProjectConfig, TrainingConfig, seed_everything
from sae_bioemu.core.loss import SAELoss
from sae_bioemu.core.model import TimeConditionedTopKSAE
from sae_bioemu.core.store import ActivationStore, StoreBatch
from sae_bioemu.core.types import ActivationRecord


@dataclass(frozen=True)
class TrainResult:
    """Paths and terminal loss produced by a completed training run."""

    checkpoint_path: Path
    metrics_path: Path
    loss_plot_path: Path
    final_loss: float


@dataclass(frozen=True)
class EvalResult:
    """Aggregate evaluation metrics and optional interpretability artifact paths."""

    loss: float
    reconstruction_loss: float
    auxk_loss: float
    l0: float
    explained_variance: float
    dead_latents: int
    active_latents: int
    num_latents: int
    num_tokens: int
    num_batches: int
    feature_report_path: str | None = None
    feature_plots_dir: str | None = None


@dataclass(frozen=True)
class EvalBatch:
    """Evaluation batch with optional provenance for feature-report examples."""

    activations: torch.Tensor
    timesteps: torch.Tensor
    examples: FeatureExamples | None


@dataclass(frozen=True)
class FeatureExamples:
    """Per-token metadata needed to turn active latents into interpretable examples."""

    sequence_ids: tuple[str, ...]
    residue_indices: torch.Tensor
    sample_indices: torch.Tensor | None

    @classmethod
    def from_record(cls, record: ActivationRecord, start: int, end: int) -> FeatureExamples:
        return cls(
            sequence_ids=record.sequence_ids[start:end],
            residue_indices=record.residue_indices[start:end],
            sample_indices=(
                record.sample_indices[start:end] if record.sample_indices is not None else None
            ),
        )


@dataclass(frozen=True)
class FeatureExample:
    """A single high-activating token example for a reported SAE feature."""

    activation: float
    sequence_id: str
    residue_index: int
    sample_index: int | None
    timestep: float
    amino_acid: str | None = None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "activation": self.activation,
            "sequence_id": self.sequence_id,
            "residue_index": self.residue_index,
            "sample_index": self.sample_index,
            "timestep": self.timestep,
            "amino_acid": self.amino_acid,
        }


class FeatureCollector:
    """Accumulates feature-level interpretability statistics during eval.

    The collector receives sparse SAE latents batch by batch and summarizes
    which features are active, when in diffusion time they activate, where they
    activate along the sequence, and which residue tokens are the strongest
    examples. It does not assign structural labels; it reports unsupervised
    activation patterns that can later be correlated with saved structures.
    """

    def __init__(
        self,
        n_latents: int,
        timestep_bins: int,
        top_k_examples: int,
        device: torch.device,
        sequences: dict[str, str] | None = None,
    ) -> None:
        if timestep_bins <= 0:
            raise ValueError("timestep_bins must be positive")
        if top_k_examples <= 0:
            raise ValueError("top_k_examples must be positive")
        self.n_latents = n_latents
        self.timestep_bins = timestep_bins
        self.top_k_examples = top_k_examples
        self.active_counts = torch.zeros(n_latents, device=device, dtype=torch.long)
        self.activation_sums = torch.zeros(n_latents, device=device)
        self.activation_maxes = torch.full((n_latents,), float("-inf"), device=device)
        self.timestep_sums = torch.zeros(n_latents, device=device)
        self.timestep_sq_sums = torch.zeros(n_latents, device=device)
        self.timestep_histograms = torch.zeros(
            n_latents,
            timestep_bins,
            device=device,
            dtype=torch.long,
        )
        self.top_examples: list[list[FeatureExample]] = [[] for _ in range(n_latents)]
        self.sequences = sequences or {}
        self.residue_activation_sums: dict[str, torch.Tensor] = {
            sequence_id: torch.zeros(n_latents, len(sequence), device=device)
            for sequence_id, sequence in self.sequences.items()
        }
        self.residue_active_counts: dict[str, torch.Tensor] = {
            sequence_id: torch.zeros(n_latents, len(sequence), device=device, dtype=torch.long)
            for sequence_id, sequence in self.sequences.items()
        }

    @property
    def active_latent_count(self) -> int:
        return int((self.active_counts > 0).sum().detach().cpu())

    def update(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        examples: FeatureExamples | None,
    ) -> None:
        active = latents > 0
        if not active.any():
            return
        values = latents.clamp_min(0)
        self.active_counts += active.sum(dim=0)
        self.activation_sums += values.sum(dim=0)
        self.activation_maxes = torch.maximum(self.activation_maxes, values.max(dim=0).values)

        timestep_weights = active.to(timesteps.dtype)
        self.timestep_sums += (timestep_weights * timesteps[:, None]).sum(dim=0)
        self.timestep_sq_sums += (timestep_weights * timesteps[:, None].pow(2)).sum(dim=0)
        bin_indices = (
            torch.clamp(timesteps, 0.0, 1.0)
            .mul(self.timestep_bins)
            .floor()
            .to(torch.long)
            .clamp(max=self.timestep_bins - 1)
        )
        feature_indices = torch.nonzero(active.any(dim=0), as_tuple=False).flatten()
        for feature_idx_tensor in feature_indices:
            feature_idx = int(feature_idx_tensor.item())
            self.timestep_histograms[feature_idx].scatter_add_(
                0,
                bin_indices[active[:, feature_idx]],
                torch.ones(
                    int(active[:, feature_idx].sum().item()),
                    device=latents.device,
                    dtype=torch.long,
                ),
            )
        if examples is not None:
            self._update_residue_profiles(values, active, examples)
            self._update_top_examples(values, active, timesteps, examples, feature_indices)

    def _update_residue_profiles(
        self,
        values: torch.Tensor,
        active: torch.Tensor,
        examples: FeatureExamples,
    ) -> None:
        residue_indices = examples.residue_indices.detach().cpu().tolist()
        for local_token_idx, sequence_id in enumerate(examples.sequence_ids):
            if sequence_id not in self.residue_activation_sums:
                continue
            residue_index = int(residue_indices[local_token_idx])
            if (
                residue_index < 0
                or residue_index >= self.residue_activation_sums[sequence_id].shape[1]
            ):
                continue
            self.residue_activation_sums[sequence_id][:, residue_index] += values[local_token_idx]
            self.residue_active_counts[sequence_id][:, residue_index] += active[local_token_idx].to(
                torch.long
            )

    def _update_top_examples(
        self,
        values: torch.Tensor,
        active: torch.Tensor,
        timesteps: torch.Tensor,
        examples: FeatureExamples,
        feature_indices: torch.Tensor,
    ) -> None:
        residue_indices = examples.residue_indices.detach().cpu().tolist()
        sample_indices = (
            examples.sample_indices.detach().cpu().tolist()
            if examples.sample_indices is not None
            else [None] * len(examples.sequence_ids)
        )
        timesteps_cpu = timesteps.detach().cpu().tolist()
        for feature_idx_tensor in feature_indices:
            feature_idx = int(feature_idx_tensor.item())
            token_indices = torch.nonzero(active[:, feature_idx], as_tuple=False).flatten()
            if token_indices.numel() == 0:
                continue
            k = min(self.top_k_examples, token_indices.numel())
            top = torch.topk(values[token_indices, feature_idx], k=k)
            current = self.top_examples[feature_idx]
            for value, local_token_idx in zip(
                top.values.detach().cpu().tolist(),
                token_indices[top.indices].detach().cpu().tolist(),
                strict=True,
            ):
                current.append(
                    FeatureExample(
                        activation=float(value),
                        sequence_id=examples.sequence_ids[local_token_idx],
                        residue_index=int(residue_indices[local_token_idx]),
                        sample_index=(
                            int(sample_indices[local_token_idx])
                            if sample_indices[local_token_idx] is not None
                            else None
                        ),
                        timestep=float(timesteps_cpu[local_token_idx]),
                        amino_acid=self._amino_acid(
                            examples.sequence_ids[local_token_idx],
                            int(residue_indices[local_token_idx]),
                        ),
                    )
                )
            current.sort(key=lambda example: example.activation, reverse=True)
            del current[self.top_k_examples :]

    def report(self, top_k_features: int, num_tokens: int) -> dict:
        counts = self.active_counts.detach().cpu()
        activation_sums = self.activation_sums.detach().cpu()
        activation_maxes = self.activation_maxes.detach().cpu()
        timestep_sums = self.timestep_sums.detach().cpu()
        timestep_sq_sums = self.timestep_sq_sums.detach().cpu()
        histograms = self.timestep_histograms.detach().cpu()
        active_feature_indices = torch.nonzero(counts > 0, as_tuple=False).flatten()
        ranked = sorted(
            active_feature_indices.tolist(),
            key=lambda idx: float(activation_maxes[idx]),
            reverse=True,
        )[:top_k_features]
        features = []
        for feature_idx in ranked:
            count = int(counts[feature_idx].item())
            mean_timestep = float(timestep_sums[feature_idx].item() / count)
            timestep_variance = max(
                float(timestep_sq_sums[feature_idx].item() / count) - mean_timestep**2,
                0.0,
            )
            preferred_bin = int(histograms[feature_idx].argmax().item())
            profile_sequence_id, residue_profile = self._residue_profile(feature_idx)
            features.append(
                {
                    "feature_id": feature_idx,
                    "activation_frequency": count / num_tokens,
                    "active_tokens": count,
                    "mean_activation_when_active": float(
                        activation_sums[feature_idx].item() / count
                    ),
                    "max_activation": float(activation_maxes[feature_idx].item()),
                    "mean_timestep_when_active": mean_timestep,
                    "timestep_std_when_active": math.sqrt(timestep_variance),
                    "preferred_timestep_bin": preferred_bin,
                    "timestep_histogram": histograms[feature_idx].tolist(),
                    "profile_sequence_id": profile_sequence_id,
                    "residue_activation_profile": residue_profile,
                    "top_examples": [
                        example.to_dict() for example in self.top_examples[feature_idx]
                    ],
                }
            )
        return {
            "summary": {
                "num_tokens": num_tokens,
                "num_latents": self.n_latents,
                "active_latents": self.active_latent_count,
                "dead_latents": self.n_latents - self.active_latent_count,
                "timestep_bins": self.timestep_bins,
                "top_k_features": top_k_features,
                "top_k_examples": self.top_k_examples,
            },
            "features": features,
        }

    def _residue_profile(self, feature_idx: int) -> tuple[str | None, list[float]]:
        if not self.residue_activation_sums:
            return None, []
        sequence_id = max(
            self.residue_activation_sums,
            key=lambda key: int(self.residue_active_counts[key][feature_idx].sum().item()),
        )
        sums = self.residue_activation_sums[sequence_id][feature_idx].detach().cpu()
        counts = self.residue_active_counts[sequence_id][feature_idx].detach().cpu().clamp_min(1)
        means = sums / counts
        means[self.residue_active_counts[sequence_id][feature_idx].detach().cpu() == 0] = 0.0
        return sequence_id, means.tolist()

    def _amino_acid(self, sequence_id: str, residue_index: int) -> str | None:
        sequence = self.sequences.get(sequence_id)
        if sequence is None or residue_index < 0 or residue_index >= len(sequence):
            return None
        return sequence[residue_index]


class RuntimeActivationSource:
    """Small adapter that turns train-time model calls into StoreBatch objects."""

    def __init__(self, config: ProjectConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.generator_seed = config.training.seed
        self.records = self._load_sequence_records()
        self.next_sample_seed: int | None = None
        self.buffer_activations: torch.Tensor | None = None
        self.buffer_timesteps: torch.Tensor | None = None
        self.cached_activations: InMemoryActivationEpochs | None = None
        if config.training.activation_source == "bioemu":
            structures_dir = (
                config.training.runtime_structure_dir or config.training.activation_store
            )
            self.source = BioEmuRuntimeActivationSampler(
                config=config.trace,
                cache_embeds_dir=config.training.cache_embeds_dir,
                structures_dir=structures_dir,
                batch_size=config.training.runtime_sequence_batch_size,
                device=device,
            )
            self.prompt_count = len(self.records) * config.trace.samples_per_sequence
        else:
            raise ValueError(
                "RuntimeActivationSource requires activation_source 'bioemu' in this release"
            )
        if self.prompt_count <= 0:
            raise ValueError("runtime activation source has no prompts")

    def batch(self, seed: int, batch_size: int) -> StoreBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        cached_activations = getattr(self, "cached_activations", None)
        if cached_activations is not None:
            return cached_activations.batch(step=seed, batch_size=batch_size)
        if self.next_sample_seed is None:
            self.next_sample_seed = seed
        target_tokens = max(batch_size, self.config.training.runtime_activation_buffer_size)
        self._fill_buffer(min_tokens=target_tokens, batch_size=batch_size)
        if self.buffer_activations is None or self.buffer_timesteps is None:
            raise ValueError("runtime activation buffer is empty")
        generator = torch.Generator().manual_seed(self.generator_seed + seed)
        order = torch.randperm(self.buffer_activations.shape[0], generator=generator)
        activations = self.buffer_activations[order]
        timesteps = self.buffer_timesteps[order]
        batch = StoreBatch(
            activations=activations[:batch_size],
            timesteps=timesteps[:batch_size],
        )
        self.buffer_activations = activations[batch_size:]
        self.buffer_timesteps = timesteps[batch_size:]
        return batch

    def initialization_batch(self, seed: int, batch_size: int) -> StoreBatch:
        """Sample SAE initialization data without advancing the training buffer."""
        cached_activations = getattr(self, "cached_activations", None)
        if cached_activations is not None:
            return cached_activations.batch(step=0, batch_size=batch_size)
        return self._single_batch(seed=seed, batch_size=batch_size)

    def training_steps(self, batch_size: int, epochs: int) -> int | None:
        cached_activations = getattr(self, "cached_activations", None)
        if cached_activations is None:
            return None
        return cached_activations.steps_per_epoch(batch_size) * epochs

    def _single_batch(self, seed: int, batch_size: int) -> StoreBatch:
        record = self._sample_record(seed, min_tokens=batch_size)
        if record.n_tokens == 0:
            raise ValueError("runtime activation source returned no tokens")
        generator = torch.Generator().manual_seed(self.generator_seed + seed)
        if record.n_tokens >= batch_size:
            indices = torch.randperm(record.n_tokens, generator=generator)[:batch_size]
        else:
            indices = torch.randint(record.n_tokens, (batch_size,), generator=generator)
        return StoreBatch(
            activations=record.activations[indices],
            timesteps=record.timesteps[indices],
        )

    def _fill_buffer(self, min_tokens: int, batch_size: int) -> None:
        while self._buffer_tokens < min_tokens:
            if self.next_sample_seed is None:
                raise ValueError("runtime sample seed is not initialized")
            record = self._sample_record(self.next_sample_seed, min_tokens=batch_size)
            self.next_sample_seed += 1
            if record.n_tokens == 0:
                continue
            if self.buffer_activations is None or self.buffer_timesteps is None:
                self.buffer_activations = record.activations
                self.buffer_timesteps = record.timesteps
            else:
                self.buffer_activations = torch.cat(
                    [self.buffer_activations, record.activations],
                    dim=0,
                )
                self.buffer_timesteps = torch.cat(
                    [self.buffer_timesteps, record.timesteps],
                    dim=0,
                )

    @property
    def _buffer_tokens(self) -> int:
        if self.buffer_activations is None:
            return 0
        return int(self.buffer_activations.shape[0])

    def _sample_record(self, seed: int, min_tokens: int) -> ActivationRecord:
        records: list[ActivationRecord] = []
        token_count = 0
        for prompt_indices in self._prompt_index_chunks(seed):
            if self.config.training.activation_source == "bioemu":
                record = self._sample_bioemu_prompt_indices(prompt_indices)
            else:
                if self.source is None:
                    raise ValueError("ESM3 runtime source was released after activation caching")
                record = self.source.sample(prompt_indices, seed=seed)
            records.append(record)
            token_count += record.n_tokens
            if token_count >= min_tokens:
                break
        if not records:
            raise ValueError("runtime activation source has no sampled records")
        return ActivationRecord.concat(records)

    def _sample_bioemu_prompt_indices(self, prompt_indices: list[int]) -> ActivationRecord:
        records: list[ActivationRecord] = []
        samples_per_sequence = self.config.trace.samples_per_sequence
        by_sample_index: dict[int, list[SequenceRecord]] = {}
        for prompt_idx in prompt_indices:
            sequence_idx = prompt_idx // samples_per_sequence
            sample_idx = prompt_idx % samples_per_sequence
            by_sample_index.setdefault(sample_idx, []).append(self.records[sequence_idx])
        for sample_idx in sorted(by_sample_index):
            records.append(self.source.sample(by_sample_index[sample_idx], seed=sample_idx))
        return ActivationRecord.concat(records)

    def _prompt_index_chunks(self, seed: int) -> list[list[int]]:
        chunk_size = min(self.config.training.runtime_sequence_batch_size, self.prompt_count)
        generator = torch.Generator().manual_seed(self.generator_seed + seed)
        order = torch.randperm(self.prompt_count, generator=generator).tolist()
        return [order[start : start + chunk_size] for start in range(0, len(order), chunk_size)]

    def _load_sequence_records(self) -> list[SequenceRecord]:
        input_fasta = self.config.training.input_fasta
        if input_fasta is None:
            return []
        return read_fasta(input_fasta)


@dataclass
class InMemoryActivationEpochs:
    """CPU-resident deterministic activations traversed exactly once per epoch."""

    activations: torch.Tensor
    timesteps: torch.Tensor
    seed: int
    _epoch: int | None = None
    _order: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.activations.ndim != 2:
            raise ValueError("cached activations must be a 2D tensor")
        if self.timesteps.shape != (self.activations.shape[0],):
            raise ValueError("cached timesteps must match cached activation rows")
        if self.activations.shape[0] == 0:
            raise ValueError("cached activations must not be empty")

    def steps_per_epoch(self, batch_size: int) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return math.ceil(self.activations.shape[0] / batch_size)

    def batch(self, step: int, batch_size: int) -> StoreBatch:
        steps_per_epoch = self.steps_per_epoch(batch_size)
        epoch = step // steps_per_epoch
        batch_idx = step % steps_per_epoch
        if self._epoch != epoch or self._order is None:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self._order = torch.randperm(self.activations.shape[0], generator=generator)
            self._epoch = epoch
        start = batch_idx * batch_size
        end = min(start + batch_size, self.activations.shape[0])
        indices = self._order[start:end]
        return StoreBatch(
            activations=self.activations[indices].to(torch.float32),
            timesteps=self.timesteps[indices].to(torch.float32),
        )


class SAETrainer:
    """Training and evaluation harness for ``TimeConditionedTopKSAE``.

    The trainer owns the activation store, model, loss, optimizer loop, periodic
    validation, checkpoint writing, and post-hoc eval reports. Training is
    step-based over shuffled activation tokens rather than epoch-based over
    protein sequences.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.store = (
            ActivationStore(config.training.activation_store)
            if config.training.activation_source == "store"
            else None
        )
        self.runtime_source: RuntimeActivationSource | None = None
        self.model = TimeConditionedTopKSAE(config.sae)
        self.loss_fn = SAELoss(auxk_coef=config.sae.aux_k_coef)
        self.device = resolve_device(config.training.device)
        self.model.to(self.device)

    def train(self, resume_from: str | Path | None = None) -> TrainResult:
        seed_everything(self.config.training.seed)
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        max_steps = self.config.training.max_steps
        if (
            self.store is None
            and self.config.training.activation_source == "esm3"
            and self.config.training.cache_runtime_activations_in_memory
        ):
            self.runtime_source = RuntimeActivationSource(self.config, self.device)
            cached_steps = self.runtime_source.training_steps(
                self.config.training.batch_size,
                self.config.training.epochs,
            )
            if cached_steps is None:
                raise ValueError("cached ESM3 runtime source did not report training steps")
            max_steps = cached_steps
            print(
                f"ESM3 cached training schedule: epochs={self.config.training.epochs} "
                f"steps={max_steps}",
                flush=True,
            )
        checkpoint: dict | None = None
        if resume_from is not None:
            checkpoint = load_training_checkpoint(resume_from, self.device)
            if checkpoint["config"] != self.config.sae:
                raise ValueError("checkpoint SAE config does not match current training config")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.to(self.device)

        learning_rate = resolve_learning_rate(self.config)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        scheduler = build_lr_scheduler(
            optimizer,
            self.config.training,
            learning_rate,
            max_steps=max_steps,
        )
        metrics: list[dict[str, float | int]] = []
        step = 0
        if checkpoint is not None:
            if checkpoint.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            step = int(checkpoint.get("step", 0))
            metrics = list(checkpoint.get("metrics", []))
        curve_metrics = list(metrics)

        train_indices, val_indices = self._train_val_indices()
        last_validation_step: int | None = None
        if resume_from is None:
            init_batch = self._initialization_batch(train_indices)
            init_batch = move_batch_to_device(init_batch, self.device)
            self.model.initialize_from_data(init_batch.activations, init_batch.timesteps)

        final_loss = float("nan")
        scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)
        while step < max_steps:
            for batch in self._iter_train_batches(step, train_indices):
                batch = move_batch_to_device(batch, self.device)
                optimizer.zero_grad(set_to_none=True)
                autocast_context = self.autocast_context()
                with autocast_context:
                    output = self.model(batch.activations, batch.timesteps)
                    loss_output = self.loss_fn(
                        output.reconstruction,
                        batch.activations,
                        output.latents,
                        aux_reconstruction=output.aux_reconstruction,
                        aux_target_baseline=output.baseline,
                    )
                scaler.scale(loss_output.loss).backward()
                self.model.remove_decoder_gradient_parallel_component_()
                learning_rate = optimizer.param_groups[0]["lr"]
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                self.model.normalize_decoder_()

                final_loss = float(loss_output.loss.detach().cpu())
                step += 1
                train_entry = {
                    "step": step,
                    "loss": final_loss,
                    "reconstruction_loss": float(loss_output.reconstruction_loss.detach().cpu()),
                    "auxk_loss": float(loss_output.auxk_loss.detach().cpu()),
                    "l0": float(loss_output.l0.detach().cpu()),
                    "explained_variance": float(loss_output.explained_variance.detach().cpu()),
                    "learning_rate": learning_rate,
                }
                metrics.append(train_entry)
                curve_metrics.append(train_entry)
                validation_entry = None
                if should_validate(step, val_indices, self.config.training.validation_interval):
                    validation_entry = self.validation_metrics(step, val_indices)
                    last_validation_step = step
                should_record_train = should_record_training_metrics(
                    step,
                    self.config.training.metrics_interval,
                    max_steps,
                    validation_entry,
                )
                if should_record_train:
                    log_training_progress(step, train_entry, validation_entry)
                if validation_entry is not None:
                    metrics.append(validation_entry)
                    curve_metrics.append(validation_entry)
                if step >= max_steps:
                    break

        if len(val_indices) > 0 and last_validation_step != step:
            validation_entry = self.validation_metrics(step, val_indices)
            metrics.append(validation_entry)
            curve_metrics.append(validation_entry)
            log_validation_progress(step, validation_entry)

        checkpoint_path = output_dir / "sae.pt"
        metrics_path = output_dir / "metrics.json"
        loss_plot_path = output_dir / "loss_curves.png"
        save_training_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            metrics=metrics,
        )
        with metrics_path.open("w") as handle:
            json.dump(metrics, handle, indent=2)
        write_training_curves(curve_metrics, loss_plot_path)
        self.config.save(output_dir / "config.yaml")
        return TrainResult(checkpoint_path, metrics_path, loss_plot_path, final_loss)

    def evaluate(self, indices: torch.Tensor | None = None) -> float:
        return self.evaluate_metrics(indices=indices).loss

    def validation_metrics(self, step: int, indices: torch.Tensor) -> dict[str, float | int]:
        result = self.evaluate_metrics(indices=indices)
        return {
            "step": step,
            "val_loss": result.loss,
            "val_reconstruction_loss": result.reconstruction_loss,
            "val_auxk_loss": result.auxk_loss,
            "val_l0": result.l0,
            "val_explained_variance": result.explained_variance,
        }

    def evaluate_metrics(
        self,
        checkpoint_path: str | Path | None = None,
        indices: torch.Tensor | None = None,
        feature_report_path: str | Path | None = None,
        top_k_features: int = 50,
        top_k_examples: int = 10,
        timestep_bins: int = 10,
    ) -> EvalResult:
        if checkpoint_path is not None:
            self.load_model_checkpoint(checkpoint_path)

        totals = {
            "loss": 0.0,
            "reconstruction_loss": 0.0,
            "auxk_loss": 0.0,
            "l0": 0.0,
            "explained_variance": 0.0,
        }
        num_tokens = 0
        num_batches = 0
        collector = FeatureCollector(
            n_latents=self.config.sae.n_latents,
            timestep_bins=timestep_bins,
            top_k_examples=top_k_examples,
            device=self.device,
            sequences=sequence_metadata(self.store.manifest) if self.store is not None else {},
        )
        saved_dead_stats = self.model.stats_last_nonzero.detach().clone()
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for batch in self._iter_eval_batches(indices):
                    activations = batch.activations.to(self.device, non_blocking=True)
                    timesteps = batch.timesteps.to(self.device, non_blocking=True)
                    with self.autocast_context():
                        output = self.model(activations, timesteps)
                        loss_output = self.loss_fn(
                            output.reconstruction,
                            activations,
                            output.latents,
                            aux_reconstruction=output.aux_reconstruction,
                            aux_target_baseline=output.baseline,
                        )
                    batch_tokens = int(activations.shape[0])
                    totals["loss"] += float(loss_output.loss.detach().cpu()) * batch_tokens
                    totals["reconstruction_loss"] += (
                        float(loss_output.reconstruction_loss.detach().cpu()) * batch_tokens
                    )
                    totals["auxk_loss"] += (
                        float(loss_output.auxk_loss.detach().cpu()) * batch_tokens
                    )
                    totals["l0"] += float(loss_output.l0.detach().cpu()) * batch_tokens
                    totals["explained_variance"] += (
                        float(loss_output.explained_variance.detach().cpu()) * batch_tokens
                    )
                    collector.update(
                        latents=output.latents,
                        timesteps=timesteps,
                        examples=batch.examples,
                    )
                    num_tokens += batch_tokens
                    num_batches += 1
        finally:
            self.model.stats_last_nonzero.copy_(saved_dead_stats)
            self.model.train(was_training)
        if num_tokens == 0:
            return EvalResult(
                loss=float("nan"),
                reconstruction_loss=float("nan"),
                auxk_loss=float("nan"),
                l0=float("nan"),
                explained_variance=float("nan"),
                dead_latents=self.config.sae.n_latents,
                active_latents=0,
                num_latents=self.config.sae.n_latents,
                num_tokens=0,
                num_batches=0,
                feature_report_path=None,
                feature_plots_dir=None,
            )
        feature_report_path_str = None
        feature_plots_dir_str = None
        if feature_report_path is not None:
            feature_report_path = Path(feature_report_path)
            feature_report_path.parent.mkdir(parents=True, exist_ok=True)
            report = collector.report(
                top_k_features=top_k_features,
                num_tokens=num_tokens,
            )
            with feature_report_path.open("w") as handle:
                json.dump(report, handle, indent=2)
            plots_dir = feature_report_path.with_name(f"{feature_report_path.stem}_plots")
            write_feature_plots(report, plots_dir)
            feature_report_path_str = str(feature_report_path)
            feature_plots_dir_str = str(plots_dir)
        active_count = collector.active_latent_count
        return EvalResult(
            loss=totals["loss"] / num_tokens,
            reconstruction_loss=totals["reconstruction_loss"] / num_tokens,
            auxk_loss=totals["auxk_loss"] / num_tokens,
            l0=totals["l0"] / num_tokens,
            explained_variance=totals["explained_variance"] / num_tokens,
            dead_latents=self.config.sae.n_latents - active_count,
            active_latents=active_count,
            num_latents=self.config.sae.n_latents,
            num_tokens=num_tokens,
            num_batches=num_batches,
            feature_report_path=feature_report_path_str,
            feature_plots_dir=feature_plots_dir_str,
        )

    def load_model_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = load_training_checkpoint(checkpoint_path, self.device)
        if checkpoint["config"] != self.config.sae:
            raise ValueError("checkpoint SAE config does not match current eval config")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)

    def _iter_eval_batches(self, indices: torch.Tensor | None) -> Any:
        if self.store is None:
            raise ValueError("evaluation requires a stored activation store")
        if indices is not None:
            for batch in self.store.iter_shard_batches(
                self.config.training.batch_size,
                indices=indices,
                shuffle_shards=False,
                shuffle_tokens=False,
            ):
                yield EvalBatch(
                    activations=batch.activations,
                    timesteps=batch.timesteps,
                    examples=None,
                )
            return
        for record in self.store.iter_records():
            for start in range(0, record.n_tokens, self.config.training.batch_size):
                end = start + self.config.training.batch_size
                yield EvalBatch(
                    activations=record.activations[start:end],
                    timesteps=record.timesteps[start:end],
                    examples=FeatureExamples.from_record(record, start, end),
                )

    def _train_val_indices(self) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.store is None:
            return None, torch.empty(0, dtype=torch.long)
        return self.store.train_val_indices(
            self.config.training.validation_fraction,
            self.config.training.seed,
        )

    def _initialization_batch(self, train_indices: torch.Tensor | None) -> StoreBatch:
        if self.store is None:
            if self.runtime_source is None:
                self.runtime_source = RuntimeActivationSource(self.config, self.device)
            if hasattr(self.runtime_source, "initialization_batch"):
                return self.runtime_source.initialization_batch(
                    seed=self.config.training.seed,
                    batch_size=self.config.training.batch_size,
                )
            return self._runtime_batch(seed=self.config.training.seed)
        return next(
            self.store.iter_shard_batches(
                self.config.training.batch_size,
                indices=train_indices,
                shuffle_shards=True,
                shuffle_tokens=True,
                seed=self.config.training.seed,
            )
        )

    def _iter_train_batches(
        self,
        step: int,
        train_indices: torch.Tensor | None,
    ) -> Any:
        if self.store is None:
            yield self._runtime_batch(seed=step)
            return
        yield from self.store.iter_shard_batches(
            self.config.training.batch_size,
            indices=train_indices,
            shuffle_shards=True,
            shuffle_tokens=True,
            seed=self.config.training.seed + step,
        )

    def _runtime_batch(self, seed: int) -> StoreBatch:
        if self.runtime_source is None:
            self.runtime_source = RuntimeActivationSource(self.config, self.device)
        return self.runtime_source.batch(seed=seed, batch_size=self.config.training.batch_size)

    @property
    def _use_amp(self) -> bool:
        return self.config.training.mixed_precision and self.device.type == "cuda"

    def autocast_context(self) -> Any:
        if not self._use_amp:
            return nullcontext()
        return torch.amp.autocast(device_type="cuda")


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    learning_rate: float,
    max_steps: int | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    max_steps = max_steps or config.max_steps
    min_ratio = config.min_learning_rate / learning_rate
    decay_steps = max(max_steps - config.lr_warmup_steps, 1)

    def lr_lambda(step: int) -> float:
        if config.lr_warmup_steps > 0 and step < config.lr_warmup_steps:
            return max((step + 1) / config.lr_warmup_steps, min_ratio)
        if config.lr_scheduler == "constant":
            return 1.0
        decay_progress = min(max((step - config.lr_warmup_steps) / decay_steps, 0.0), 1.0)
        if config.lr_scheduler == "linear":
            return min_ratio + (1.0 - min_ratio) * (1.0 - decay_progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def resolve_learning_rate(config: ProjectConfig) -> float:
    """Resolve explicit or sparsify-style auto-scaled learning rate."""

    if config.training.learning_rate is not None:
        return config.training.learning_rate
    return 2e-4 / math.sqrt(config.sae.n_latents / (2**14))


def should_validate(
    step: int,
    val_indices: torch.Tensor,
    validation_interval: int | None,
) -> bool:
    return (
        len(val_indices) > 0
        and validation_interval is not None
        and step > 0
        and step % validation_interval == 0
    )


def should_record_training_metrics(
    step: int,
    metrics_interval: int,
    max_steps: int,
    validation_entry: dict[str, float | int] | None = None,
) -> bool:
    return step % metrics_interval == 0 or step >= max_steps or validation_entry is not None


def log_training_progress(
    step: int,
    train_entry: dict[str, float | int],
    validation_entry: dict[str, float | int] | None = None,
) -> None:
    parts = [
        f"[{_timestamp()}]",
        "training",
        f"step={step}",
        f"loss={float(train_entry['loss']):.6f}",
        f"reconstruction_loss={float(train_entry['reconstruction_loss']):.6f}",
        f"auxk_loss={float(train_entry['auxk_loss']):.6f}",
        f"l0={float(train_entry['l0']):.6f}",
        f"explained_variance={float(train_entry['explained_variance']):.6f}",
        f"learning_rate={float(train_entry['learning_rate']):.6g}",
    ]
    if validation_entry is not None:
        parts.extend(_format_validation_metrics(validation_entry))
    print(" ".join(parts), flush=True)


def log_validation_progress(step: int, validation_entry: dict[str, float | int]) -> None:
    print(
        " ".join(
            [
                f"[{_timestamp()}]",
                "validation",
                f"step={step}",
                *_format_validation_metrics(validation_entry),
            ]
        ),
        flush=True,
    )


def _format_validation_metrics(validation_entry: dict[str, float | int]) -> list[str]:
    return [
        f"val_loss={float(validation_entry['val_loss']):.6f}",
        f"val_reconstruction_loss={float(validation_entry['val_reconstruction_loss']):.6f}",
        f"val_auxk_loss={float(validation_entry['val_auxk_loss']):.6f}",
        f"val_l0={float(validation_entry['val_l0']):.6f}",
        f"val_explained_variance={float(validation_entry['val_explained_variance']):.6f}",
    ]


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_training_curves(metrics: list[dict[str, float | int]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    train_steps = [int(entry["step"]) for entry in metrics if "loss" in entry]
    train_losses = [float(entry["loss"]) for entry in metrics if "loss" in entry]
    reconstruction_steps = [
        int(entry["step"]) for entry in metrics if "reconstruction_loss" in entry
    ]
    reconstruction_losses = [
        float(entry["reconstruction_loss"]) for entry in metrics if "reconstruction_loss" in entry
    ]
    val_steps = [int(entry["step"]) for entry in metrics if "val_loss" in entry]
    val_losses = [float(entry["val_loss"]) for entry in metrics if "val_loss" in entry]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(8, 5))
    if train_losses:
        ax.plot(train_steps, train_losses, label="train loss", linewidth=1.6)
    if reconstruction_losses:
        ax.plot(
            reconstruction_steps,
            reconstruction_losses,
            label="reconstruction loss",
            linewidth=1.2,
            alpha=0.8,
        )
    if val_losses:
        ax.plot(val_steps, val_losses, label="validation loss", marker="o", linewidth=1.6)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def write_feature_plots(report: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = report.get("features", [])
    if not features:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feature_labels = [str(feature["feature_id"]) for feature in features]
    timestep_matrix = torch.tensor(
        [feature.get("timestep_histogram", []) for feature in features],
        dtype=torch.float32,
    )
    if timestep_matrix.numel() > 0:
        row_sums = timestep_matrix.sum(dim=1, keepdim=True).clamp_min(1)
        timestep_matrix = timestep_matrix / row_sums
        fig, ax = plt.subplots(figsize=(8, max(4, 0.24 * len(features))))
        im = ax.imshow(timestep_matrix.numpy(), aspect="auto", interpolation="nearest")
        tick_count = timestep_matrix.shape[1]
        ax.set_xticks(range(tick_count))
        ax.set_xticklabels([f"{idx / tick_count:.1f}" for idx in range(tick_count)])
        ax.set_xlabel("timestep t, low to high")
        ax.set_ylabel("feature id")
        ax.set_yticks(range(len(feature_labels)))
        ax.set_yticklabels(feature_labels)
        ax.set_title("Feature activation distribution over diffusion time")
        fig.colorbar(im, ax=ax, label="fraction of active tokens")
        fig.tight_layout()
        fig.savefig(output_dir / "feature_timestep_heatmap.png", dpi=300)
        plt.close(fig)

    profiles = [
        feature.get("residue_activation_profile", [])
        for feature in features
        if feature.get("residue_activation_profile")
    ]
    if profiles:
        max_len = max(len(profile) for profile in profiles)
        residue_matrix = torch.zeros(len(profiles), max_len, dtype=torch.float32)
        for row, profile in enumerate(profiles):
            residue_matrix[row, : len(profile)] = torch.tensor(profile, dtype=torch.float32)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.24 * len(profiles))))
        im = ax.imshow(residue_matrix.numpy(), aspect="auto", interpolation="nearest")
        ax.set_xlabel("residue index")
        ax.set_ylabel("feature id")
        ax.set_yticks(range(len(feature_labels[: len(profiles)])))
        ax.set_yticklabels(feature_labels[: len(profiles)])
        ax.set_title("Mean feature activation by residue")
        fig.colorbar(im, ax=ax, label="mean activation when evaluated")
        fig.tight_layout()
        fig.savefig(output_dir / "feature_residue_heatmap.png", dpi=300)
        plt.close(fig)

        profiles_dir = output_dir / "top_feature_residue_profiles"
        profiles_dir.mkdir(exist_ok=True)
        for feature in features:
            profile = feature.get("residue_activation_profile", [])
            if not profile:
                continue
            feature_id = int(feature["feature_id"])
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(range(len(profile)), profile, linewidth=1.4)
            ax.set_xlabel("residue index")
            ax.set_ylabel("mean activation")
            ax.set_title(f"Feature {feature_id} residue profile")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(profiles_dir / f"feature_{feature_id}_residue_profile.png", dpi=300)
            plt.close(fig)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    return type(batch)(
        activations=batch.activations.to(device, non_blocking=True),
        timesteps=batch.timesteps.to(device, non_blocking=True),
    )


def sequence_metadata(manifest: dict) -> dict[str, str] | None:
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return None
    sequences = metadata.get("sequences")
    if not isinstance(sequences, dict):
        return None
    return {str(sequence_id): str(sequence) for sequence_id, sequence in sequences.items()}


def save_training_checkpoint(
    path: str | Path,
    model: TimeConditionedTopKSAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    metrics: list[dict[str, float | int]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        prepare_sae_checkpoint_payload(
            {
                "config": model.config,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step,
                "metrics": metrics,
            }
        ),
        path,
    )


def load_training_checkpoint(path: str | Path, device: torch.device) -> dict:
    return load_sae_checkpoint(path, map_location=device)
