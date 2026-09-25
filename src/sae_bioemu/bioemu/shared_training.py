from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from sae_bioemu.bioemu.extractor import (
    BioEmuMultiLayerRuntimeSampler,
    SequenceRecord,
    read_fasta,
)
from sae_bioemu.core.checkpoint_io import prepare_sae_checkpoint_payload
from sae_bioemu.core.config import ProjectConfig, seed_everything
from sae_bioemu.core.loss import SAELoss
from sae_bioemu.core.model import TimeConditionedTopKSAE
from sae_bioemu.core.trainer import (
    build_lr_scheduler,
    load_training_checkpoint,
    log_training_progress,
    resolve_learning_rate,
    write_training_curves,
)
from sae_bioemu.core.types import ActivationRecord


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


@dataclass
class SharedSAEState:
    name: str
    config: ProjectConfig
    target_module: str
    model: TimeConditionedTopKSAE
    loss_fn: SAELoss
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
    distributed_model: DistributedDataParallel | None = None
    scaler: torch.amp.GradScaler | None = None
    metrics: list[dict[str, float | int]] | None = None
    checkpoint: dict[str, Any] | None = None
    step: int = 0

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = []


class MultiLayerActivationBuffer:
    """Bounded CPU activation buffer shared by all SAE variants."""

    def __init__(self, target_modules: tuple[str, ...]) -> None:
        self.target_modules = target_modules
        self.activations: dict[str, list[torch.Tensor]] = {target: [] for target in target_modules}
        self.timesteps: list[torch.Tensor] = []
        self.n_tokens = 0

    def append(self, records: dict[str, ActivationRecord]) -> None:
        if set(records) != set(self.target_modules):
            raise ValueError("captured layer set does not match activation buffer")
        first = records[self.target_modules[0]]
        for target in self.target_modules:
            record = records[target]
            if record.n_tokens != first.n_tokens:
                raise ValueError("captured BioEmu layers have different token counts")
            if not torch.equal(record.timesteps, first.timesteps):
                raise ValueError("captured BioEmu layers have different timesteps")
            self.activations[target].append(record.activations)
        self.timesteps.append(first.timesteps)
        self.n_tokens += first.n_tokens

    def pop(self, n_rows: int, seed: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if n_rows <= 0 or self.n_tokens == 0:
            return {}, torch.empty(0)
        n_rows = min(n_rows, self.n_tokens)
        layer_values = {
            target: torch.cat(parts, dim=0) for target, parts in self.activations.items()
        }
        timesteps = torch.cat(self.timesteps, dim=0)
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(self.n_tokens, generator=generator)
        selected = order[:n_rows]
        retained = order[n_rows:]
        batch = {target: values[selected] for target, values in layer_values.items()}
        batch_timesteps = timesteps[selected]
        self.activations = {
            target: [values[retained]] if retained.numel() else []
            for target, values in layer_values.items()
        }
        self.timesteps = [timesteps[retained]] if retained.numel() else []
        self.n_tokens -= n_rows
        return batch, batch_timesteps


class SharedAblationOptimizationCore:
    """Shared optimizer/DDP machinery for runtime and stored trainers."""

    def __init__(
        self, configs: dict[str, ProjectConfig], context: DistributedContext, total_steps: int
    ) -> None:
        self.context = context
        self.total_steps = total_steps
        self.states: dict[str, SharedSAEState] = {}
        for name, config in configs.items():
            seed_everything(config.training.seed)
            self.states[name] = SharedSAEState(
                name,
                config,
                config.trace.target_module,
                TimeConditionedTopKSAE(config.sae).to(context.device),
                SAELoss(config.sae.aux_k_coef),
            )

    @property
    def initialized(self) -> bool:
        return all(state.optimizer is not None for state in self.states.values())

    def initialize_models(
        self, layer_batches: dict[str, torch.Tensor], timesteps: torch.Tensor
    ) -> None:
        for state in self.states.values():
            state.model.initialize_from_data(
                layer_batches[state.target_module].to(self.context.device),
                timesteps.to(self.context.device),
            )
        self._initialize_optimizers()

    def _initialize_optimizers(self) -> None:
        for state in self.states.values():
            learning_rate = resolve_learning_rate(state.config)
            state.optimizer = torch.optim.AdamW(
                state.model.parameters(),
                lr=learning_rate,
                weight_decay=state.config.training.weight_decay,
            )
            state.scheduler = build_lr_scheduler(
                state.optimizer,
                replace(state.config.training, max_steps=self.total_steps),
                learning_rate,
            )
            state.scaler = torch.amp.GradScaler(
                "cuda",
                enabled=state.config.training.mixed_precision
                and self.context.device.type == "cuda",
            )
            if state.checkpoint is not None:
                state.optimizer.load_state_dict(state.checkpoint["optimizer_state_dict"])
                state.scheduler.load_state_dict(state.checkpoint["scheduler_state_dict"])
                if state.checkpoint.get("scaler_state_dict") is not None:
                    state.scaler.load_state_dict(state.checkpoint["scaler_state_dict"])
            if self.context.world_size > 1:
                kwargs: dict[str, Any] = {"find_unused_parameters": True}
                if self.context.device.type == "cuda":
                    kwargs.update(
                        device_ids=[self.context.local_rank], output_device=self.context.local_rank
                    )
                state.distributed_model = DistributedDataParallel(state.model, **kwargs)

    def train_step(
        self, layer_batches: dict[str, torch.Tensor], timesteps: torch.Tensor, n_rows: int
    ) -> None:
        SharedBioEmuAblationTrainer._train_step(self, layer_batches, timesteps, n_rows)

    def checkpoint_payloads(self, progress_metadata: dict[str, Any]) -> dict[str, dict]:
        result = {}
        for name, state in self.states.items():
            assert (
                state.optimizer is not None
                and state.scheduler is not None
                and state.scaler is not None
            )
            result[name] = prepare_sae_checkpoint_payload(
                {
                    "config": state.model.config,
                    "state_dict": state.model.state_dict(),
                    "optimizer_state_dict": state.optimizer.state_dict(),
                    "scheduler_state_dict": state.scheduler.state_dict(),
                    "scaler_state_dict": state.scaler.state_dict(),
                    "step": state.step,
                    "metrics": state.metrics,
                    "shared_training": progress_metadata,
                }
            )
        return result

    def restore(self, checkpoints: dict[str, dict]) -> None:
        for name, checkpoint in checkpoints.items():
            state = self.states[name]
            if checkpoint["config"] != state.config.sae:
                raise ValueError(f"SAE config mismatch for {name}")
            state.model.load_state_dict(checkpoint["state_dict"])
            state.step = int(checkpoint.get("step", 0))
            state.metrics = list(checkpoint.get("metrics", []))
            state.checkpoint = checkpoint
        self._initialize_optimizers()


class SharedBioEmuAblationTrainer:
    """Train all BioEmu SAE ablations from one balanced trajectory stream."""

    def __init__(self, configs: dict[str, ProjectConfig], resume: bool = False) -> None:
        if not configs:
            raise ValueError("at least one ablation config is required")
        self.configs = configs
        self.resume = resume
        self.context = initialize_distributed()
        self.base = validate_shared_configs(configs)
        config_key = hashlib.sha256("\n".join(sorted(configs)).encode("utf-8")).hexdigest()[:12]
        self.manifest_path = Path(
            f"outputs/ablations_interpro/.shared_training_state_{config_key}.json"
        )
        if self.base.training.epochs != 1:
            raise ValueError("shared BioEmu training currently requires epochs=1")
        seed_everything(self.base.training.seed + self.context.rank)
        self.records = read_fasta(self.base.training.input_fasta)
        if not self.records:
            raise ValueError("input FASTA contains no sequences")
        self.target_modules = tuple(
            dict.fromkeys(config.trace.target_module for config in configs.values())
        )
        self.local_batch_size = distributed_batch_size(
            self.base.training.batch_size,
            self.context.world_size,
        )
        self.total_steps = total_synchronized_steps(
            records=self.records,
            samples_per_sequence=self.base.trace.samples_per_sequence,
            recorded_score_calls=self.base.trace.recorded_score_calls,
            global_batch_size=self.base.training.batch_size,
            world_size=self.context.world_size,
            seed=self.base.training.seed,
        )
        self.optimization_core = SharedAblationOptimizationCore(
            configs, self.context, self.total_steps
        )
        self.states = self.optimization_core.states
        self.completed_sample_index = -1
        if resume:
            self._load_checkpoints()
        elif self.context.is_primary:
            self._reject_existing_outputs()
        distributed_barrier()
        self.sampler = BioEmuMultiLayerRuntimeSampler(
            config=self.base.trace,
            target_modules=self.target_modules,
            cache_embeds_dir=self.base.training.cache_embeds_dir,
            structures_dir=self.base.training.runtime_structure_dir,
            batch_size=self.base.training.runtime_sequence_batch_size,
            device=self.context.device,
        )
        self.buffer = MultiLayerActivationBuffer(self.target_modules)
        self.flush_index = 0
        if self.completed_sample_index >= 0:
            self._initialize_optimizers_from_loaded_models()

    def train(self) -> None:
        for sample_index in range(
            self.completed_sample_index + 1,
            self.base.trace.samples_per_sequence,
        ):
            local_records = records_for_rank(
                self.records,
                sample_index=sample_index,
                seed=self.base.training.seed,
                rank=self.context.rank,
                world_size=self.context.world_size,
            )
            self._train_sample_index(sample_index, local_records)
            self.completed_sample_index = sample_index
            self._save_outputs()
            distributed_barrier()
            if self.context.is_primary:
                print(
                    f"completed BioEmu sample seed {sample_index + 1}/"
                    f"{self.base.trace.samples_per_sequence}",
                    flush=True,
                )
        if self.context.is_primary:
            print("shared BioEmu ablation training complete", flush=True)

    def _train_sample_index(
        self,
        sample_index: int,
        local_records: list[SequenceRecord],
    ) -> None:
        sequence_batch_size = self.base.training.runtime_sequence_batch_size
        local_chunks = math.ceil(len(local_records) / sequence_batch_size)
        max_chunks = distributed_max_int(local_chunks, self.context.device)
        local_buffer_limit = max(
            self.local_batch_size,
            math.ceil(self.base.training.runtime_activation_buffer_size / self.context.world_size),
        )
        for chunk_index in range(max_chunks):
            start = chunk_index * sequence_batch_size
            record_batch = local_records[start : start + sequence_batch_size]
            if record_batch:
                self.buffer.append(self.sampler.sample_layers(record_batch, seed=sample_index))
            local_full_batches = (
                self.buffer.n_tokens // self.local_batch_size
                if self.buffer.n_tokens >= local_buffer_limit
                else 0
            )
            shared_full_batches = distributed_min_int(
                local_full_batches,
                self.context.device,
            )
            for _ in range(shared_full_batches):
                self._train_buffer_rows(self.local_batch_size, sample_index)

        while distributed_max_int(self.buffer.n_tokens, self.context.device) > 0:
            self._train_buffer_rows(
                min(self.local_batch_size, self.buffer.n_tokens),
                sample_index,
            )

    def _train_buffer_rows(self, n_rows: int, sample_index: int) -> None:
        layer_batches, timesteps = self.buffer.pop(
            n_rows,
            seed=(
                self.base.training.seed
                + sample_index * 1_000_000
                + self.flush_index
                + self.context.rank * 10_000
            ),
        )
        self.flush_index += 1
        if not self._models_initialized():
            if not layer_batches:
                raise RuntimeError("cannot initialize shared SAEs from an empty rank")
            self._initialize_models(layer_batches, timesteps)
        self._train_step(layer_batches, timesteps, n_rows)

    def _initialize_models(
        self,
        layer_batches: dict[str, torch.Tensor],
        timesteps: torch.Tensor,
    ) -> None:
        for state in self.states.values():
            state.model.initialize_from_data(
                layer_batches[state.target_module].to(self.context.device),
                timesteps.to(self.context.device),
            )
        self._initialize_optimizers_from_loaded_models()

    def _initialize_optimizers_from_loaded_models(self) -> None:
        for state in self.states.values():
            learning_rate = resolve_learning_rate(state.config)
            state.optimizer = torch.optim.AdamW(
                state.model.parameters(),
                lr=learning_rate,
                weight_decay=state.config.training.weight_decay,
            )
            scheduler_config = replace(
                state.config.training,
                max_steps=self.total_steps,
            )
            state.scheduler = build_lr_scheduler(
                state.optimizer,
                scheduler_config,
                learning_rate,
            )
            state.scaler = torch.amp.GradScaler(
                "cuda",
                enabled=(
                    state.config.training.mixed_precision and self.context.device.type == "cuda"
                ),
            )
            checkpoint = state.checkpoint
            if checkpoint is not None:
                state.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                state.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                if checkpoint.get("scaler_state_dict") is not None:
                    state.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if self.context.world_size > 1:
                state.distributed_model = DistributedDataParallel(
                    state.model,
                    device_ids=[self.context.local_rank],
                    output_device=self.context.local_rank,
                    find_unused_parameters=True,
                )

    def _train_step(
        self,
        layer_batches: dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        n_rows: int,
    ) -> None:
        global_rows = distributed_sum_int(n_rows, self.context.device)
        if global_rows <= 0:
            return
        timestep_batch = timesteps.to(self.context.device)
        for state in self.states.values():
            assert state.optimizer is not None
            assert state.scheduler is not None
            assert state.scaler is not None
            state.optimizer.zero_grad(set_to_none=True)
            previous_dead_stats = state.model.stats_last_nonzero.detach().clone()
            if n_rows > 0:
                activations = layer_batches[state.target_module].to(self.context.device)
            else:
                activations = torch.zeros(
                    (2, state.config.sae.input_dim),
                    device=self.context.device,
                )
                timestep_batch = torch.zeros(2, device=self.context.device)
            model = state.distributed_model or state.model
            use_amp = state.config.training.mixed_precision and self.context.device.type == "cuda"
            with torch.amp.autocast(self.context.device.type, enabled=use_amp):
                output = model(activations, timestep_batch)
                loss_output = state.loss_fn(
                    output.reconstruction,
                    activations,
                    output.latents,
                    aux_reconstruction=output.aux_reconstruction,
                    aux_target_baseline=output.baseline,
                )
                gradient_scale = n_rows * self.context.world_size / global_rows
                loss = loss_output.loss * gradient_scale
            active_features = (
                (output.latents > 1e-3).any(dim=0)
                if n_rows > 0
                else torch.zeros(
                    state.config.sae.n_latents,
                    dtype=torch.bool,
                    device=self.context.device,
                )
            )
            if dist.is_initialized():
                dist.all_reduce(active_features, op=dist.ReduceOp.MAX)
            state.model.stats_last_nonzero.copy_(previous_dead_stats)
            state.model.stats_last_nonzero *= (~active_features).to(
                state.model.stats_last_nonzero.dtype
            )
            state.model.stats_last_nonzero += global_rows
            state.scaler.scale(loss).backward()
            state.scaler.unscale_(state.optimizer)
            state.model.remove_decoder_gradient_parallel_component_()
            state.scaler.step(state.optimizer)
            state.scaler.update()
            state.scheduler.step()
            state.model.normalize_decoder_()
            state.step += 1
            metric_values = distributed_weighted_metrics(
                loss_output,
                n_rows=n_rows,
                device=self.context.device,
            )
            entry = {
                "step": state.step,
                **metric_values,
                "learning_rate": state.optimizer.param_groups[0]["lr"],
            }
            assert state.metrics is not None
            state.metrics.append(entry)
            if self.context.is_primary and state.step % state.config.training.metrics_interval == 0:
                print(f"ablation={state.name}", end=" ")
                log_training_progress(state.step, entry)

    def _models_initialized(self) -> bool:
        return all(state.optimizer is not None for state in self.states.values())

    def _save_outputs(self) -> None:
        if not self.context.is_primary:
            return
        pending: list[tuple[SharedSAEState, Path, Path]] = []
        for state in self.states.values():
            assert state.optimizer is not None
            assert state.scheduler is not None
            assert state.scaler is not None
            assert state.metrics is not None
            output_dir = Path(state.config.training.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = prepare_sae_checkpoint_payload(
                {
                    "config": state.model.config,
                    "state_dict": state.model.state_dict(),
                    "optimizer_state_dict": state.optimizer.state_dict(),
                    "scheduler_state_dict": state.scheduler.state_dict(),
                    "scaler_state_dict": state.scaler.state_dict(),
                    "step": state.step,
                    "metrics": state.metrics,
                    "shared_training": {
                        "completed_sample_index": self.completed_sample_index,
                        "samples_per_sequence": self.base.trace.samples_per_sequence,
                        "num_sequences": len(self.records),
                        "world_size": self.context.world_size,
                        "effective_global_batch_size": (
                            self.local_batch_size * self.context.world_size
                        ),
                        "total_steps": self.total_steps,
                    },
                }
            )
            checkpoint_path = output_dir / "sae.pt"
            next_checkpoint = output_dir / "sae.pt.next"
            torch.save(checkpoint, next_checkpoint)
            pending.append((state, checkpoint_path, next_checkpoint))

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_temporary = self.manifest_path.with_suffix(".tmp")
        manifest_temporary.write_text(
            json.dumps(
                {
                    "completed_sample_index": self.completed_sample_index,
                    "runs": sorted(self.states),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        manifest_temporary.replace(self.manifest_path)

        for state, checkpoint_path, next_checkpoint in pending:
            next_checkpoint.replace(checkpoint_path)
            output_dir = checkpoint_path.parent
            (output_dir / "metrics.json").write_text(json.dumps(state.metrics, indent=2) + "\n")
            write_training_curves(state.metrics, output_dir / "loss_curves.png")
            replace(
                state.config,
                training=replace(state.config.training, max_steps=self.total_steps),
            ).save(output_dir / "config.yaml")

    def _load_checkpoints(self) -> None:
        expected_index = None
        if self.manifest_path.is_file():
            expected_index = int(
                json.loads(self.manifest_path.read_text())["completed_sample_index"]
            )
        completed_indices = set()
        for state in self.states.values():
            checkpoint_path = Path(state.config.training.output_dir) / "sae.pt"
            candidate_paths = [checkpoint_path, checkpoint_path.with_name("sae.pt.next")]
            available = [path for path in candidate_paths if path.exists()]
            if not available:
                raise FileNotFoundError(f"Missing shared-training checkpoint: {checkpoint_path}")
            checkpoints = []
            for path in available:
                checkpoint = load_training_checkpoint(path, self.context.device)
                checkpoint_index = int(
                    checkpoint.get("shared_training", {}).get("completed_sample_index", -2)
                )
                if expected_index is None or checkpoint_index == expected_index:
                    checkpoints.append(checkpoint)
                    break
            if not checkpoints:
                raise ValueError(
                    f"No committed shared-training checkpoint is available for {state.name}"
                )
            checkpoint = checkpoints[0]
            if checkpoint["config"] != state.config.sae:
                raise ValueError(f"SAE config mismatch for {state.name}")
            shared = checkpoint.get("shared_training")
            if not isinstance(shared, dict):
                raise ValueError(f"Checkpoint for {state.name} is not a shared-training checkpoint")
            completed_indices.add(int(shared["completed_sample_index"]))
            state.model.load_state_dict(checkpoint["state_dict"])
            state.step = int(checkpoint.get("step", 0))
            state.metrics = list(checkpoint.get("metrics", []))
            state.checkpoint = checkpoint
        if len(completed_indices) != 1:
            raise ValueError("Shared ablation checkpoints disagree on completed sample seed")
        self.completed_sample_index = completed_indices.pop()
        if expected_index is not None and self.completed_sample_index != expected_index:
            raise ValueError("Shared training manifest disagrees with committed checkpoints")

    def _reject_existing_outputs(self) -> None:
        existing = [
            Path(state.config.training.output_dir) / "sae.pt"
            for state in self.states.values()
            if (Path(state.config.training.output_dir) / "sae.pt").exists()
        ]
        if existing:
            raise FileExistsError(
                "Shared training outputs already exist; use --resume or choose new output paths: "
                + ", ".join(str(path) for path in existing)
            )


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            torch.cuda.set_device(local_rank)
        init_kwargs: dict[str, Any] = {"backend": "nccl" if has_cuda else "gloo"}
        timeout_seconds = os.environ.get("SAE_DISTRIBUTED_TIMEOUT_SECONDS")
        if timeout_seconds is not None:
            init_kwargs["timeout"] = timedelta(seconds=int(timeout_seconds))
        if not dist.is_initialized():
            dist.init_process_group(**init_kwargs)
        device = torch.device("cuda", local_rank) if has_cuda else torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistributedContext(rank, world_size, local_rank, device)


def distributed_barrier() -> None:
    if dist.is_initialized():
        device_ids = [torch.cuda.current_device()] if dist.get_backend() == "nccl" else None
        dist.barrier(device_ids=device_ids)


def distributed_batch_size(global_batch_size: int, world_size: int) -> int:
    return math.ceil(global_batch_size / world_size)


def records_for_rank(
    records: list[SequenceRecord],
    sample_index: int,
    seed: int,
    rank: int,
    world_size: int,
) -> list[SequenceRecord]:
    generator = torch.Generator().manual_seed(seed + sample_index)
    order = torch.randperm(len(records), generator=generator).tolist()
    return [records[index] for position, index in enumerate(order) if position % world_size == rank]


def total_synchronized_steps(
    records: list[SequenceRecord],
    samples_per_sequence: int,
    recorded_score_calls: int | None,
    global_batch_size: int,
    world_size: int,
    seed: int,
) -> int:
    if recorded_score_calls is None:
        raise ValueError("shared training requires trace.recorded_score_calls")
    local_batch_size = distributed_batch_size(global_batch_size, world_size)
    total = 0
    for sample_index in range(samples_per_sequence):
        tokens_by_rank = []
        for rank in range(world_size):
            rank_records = records_for_rank(
                records,
                sample_index=sample_index,
                seed=seed,
                rank=rank,
                world_size=world_size,
            )
            tokens_by_rank.append(
                sum(len(record.sequence) for record in rank_records) * recorded_score_calls
            )
        total += max(math.ceil(tokens / local_batch_size) for tokens in tokens_by_rank)
    return total


def validate_shared_configs(configs: dict[str, ProjectConfig]) -> ProjectConfig:
    base = next(iter(configs.values()))
    common_fields = (
        "input_fasta",
        "cache_embeds_dir",
        "runtime_structure_dir",
        "runtime_sequence_batch_size",
        "runtime_activation_buffer_size",
        "batch_size",
        "epochs",
        "seed",
        "mixed_precision",
        "learning_rate",
        "lr_scheduler",
        "lr_warmup_steps",
        "min_learning_rate",
        "weight_decay",
        "metrics_interval",
    )
    for name, config in configs.items():
        if config.training.activation_source != "bioemu":
            raise ValueError(f"{name} is not a BioEmu runtime config")
        for field_name in common_fields:
            if getattr(config.training, field_name) != getattr(base.training, field_name):
                raise ValueError(f"{name} differs in shared training field {field_name}")
        if config.trace.samples_per_sequence != base.trace.samples_per_sequence:
            raise ValueError(f"{name} differs in samples_per_sequence")
        if config.trace.recorded_score_calls != base.trace.recorded_score_calls:
            raise ValueError(f"{name} differs in recorded_score_calls")
        for field_name in (
            "model_name",
            "denoiser_type",
            "denoiser_config_path",
            "cache_so3_dir",
            "save_final_structures",
            "final_structures_subdir",
            "ckpt_path",
            "model_config_path",
        ):
            if getattr(config.trace, field_name) != getattr(base.trace, field_name):
                raise ValueError(f"{name} differs in shared trace field {field_name}")
    if base.training.input_fasta is None:
        raise ValueError("shared training requires training.input_fasta")
    if base.training.runtime_structure_dir is None:
        raise ValueError("shared training requires training.runtime_structure_dir")
    return base


def distributed_int(value: int, device: torch.device, op: dist.ReduceOp) -> int:
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.long, device=device)
    dist.all_reduce(tensor, op=op)
    return int(tensor.item())


def distributed_min_int(value: int, device: torch.device) -> int:
    return distributed_int(value, device, dist.ReduceOp.MIN)


def distributed_max_int(value: int, device: torch.device) -> int:
    return distributed_int(value, device, dist.ReduceOp.MAX)


def distributed_sum_int(value: int, device: torch.device) -> int:
    return distributed_int(value, device, dist.ReduceOp.SUM)


def distributed_weighted_metrics(
    loss_output: Any,
    n_rows: int,
    device: torch.device,
) -> dict[str, float]:
    values = torch.tensor(
        [
            float(loss_output.loss.detach()) * n_rows,
            float(loss_output.reconstruction_loss.detach()) * n_rows,
            float(loss_output.auxk_loss.detach()) * n_rows,
            float(loss_output.l0.detach()) * n_rows,
            float(loss_output.explained_variance.detach()) * n_rows,
            float(n_rows),
        ],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    count = max(float(values[-1].item()), 1.0)
    return {
        "loss": float(values[0].item() / count),
        "reconstruction_loss": float(values[1].item() / count),
        "auxk_loss": float(values[2].item() / count),
        "l0": float(values[3].item() / count),
        "explained_variance": float(values[4].item() / count),
    }
