from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml

from sae_bioemu.core.paths import expand_placeholders


@dataclass(frozen=True)
class SAEConfig:
    """Architecture settings for the time-conditioned TopK sparse autoencoder.

    The SAE is trained on per-token BioEmu activation vectors. ``input_dim``
    must match the hooked BioEmu layer width, ``n_latents`` is the dictionary
    size, and ``top_k`` fixes the number of active SAE features per token.
    The auxiliary-k settings follow the common SAE trick of periodically
    training dead latents against reconstruction residuals.
    """

    input_dim: int = 512
    n_latents: int = 4096
    top_k: int = 32
    aux_k: int | None = 256
    aux_k_coef: float = 1 / 32
    dead_tokens_threshold: int = 1_000_000
    timestep_mlp_hidden_dim: int = 128
    use_time_conditioning: bool = True
    normalize_inputs: bool = True

    def __post_init__(self) -> None:
        _require_positive("input_dim", self.input_dim)
        _require_positive("n_latents", self.n_latents)
        _require_positive("top_k", self.top_k)
        if self.aux_k is not None:
            _require_positive("aux_k", self.aux_k)
        _require_positive("timestep_mlp_hidden_dim", self.timestep_mlp_hidden_dim)
        _require_positive("dead_tokens_threshold", self.dead_tokens_threshold)
        if self.aux_k_coef < 0:
            raise ValueError("aux_k_coef must be >= 0")
        if self.top_k > self.n_latents:
            raise ValueError("top_k must be <= n_latents")


@dataclass(frozen=True)
class TraceConfig:
    """Controls which BioEmu activations are extracted into an activation store.

    Extraction wraps BioEmu's denoiser loop and records a target PyTorch module
    every time the score model is called, or a uniform subset when
    ``recorded_score_calls`` is set. ``samples_per_sequence`` repeats the
    stochastic denoising trajectory for the same sequence so the store can
    contain multiple Boltzmann-emulator samples per input sequence.
    """

    model_name: str = "bioemu-v1.1"
    target_module: str = "model_nn.st_module.encoder.layers.-1"
    denoiser_type: Literal["dpm", "heun"] = "dpm"
    denoiser_config_path: str | None = None
    cache_so3_dir: str | None = None
    max_tokens_per_shard: int = 1_000_000
    recorded_score_calls: int | None = None
    samples_per_sequence: int = 1
    save_final_structures: bool = True
    final_structures_subdir: str = "final_structures"
    ckpt_path: str | None = None
    model_config_path: str | None = None

    def __post_init__(self) -> None:
        if not self.target_module:
            raise ValueError("target_module must be non-empty")
        if self.denoiser_type not in {"dpm", "heun"}:
            raise ValueError("denoiser_type must be 'dpm' or 'heun'")
        _require_positive("max_tokens_per_shard", self.max_tokens_per_shard)
        if self.recorded_score_calls is not None:
            _require_positive("recorded_score_calls", self.recorded_score_calls)
        _require_positive("samples_per_sequence", self.samples_per_sequence)
        if self.save_final_structures and not self.final_structures_subdir:
            raise ValueError("final_structures_subdir must be non-empty")


@dataclass(frozen=True)
class TrainingConfig:
    """Training and evaluation settings for fitting an SAE from runtime activations.

    ``max_steps`` counts optimizer updates rather than epochs. The default path
    samples BioEmu activations during training while reusing cached embeddings;
    ``activation_source='store'`` remains available only for explicit legacy
    stored activation datasets.
    """

    activation_store: str = "activation_stores/default_runtime_structures"
    activation_source: Literal["store", "bioemu"] = "bioemu"
    input_fasta: str | None = "data/processed/uniref50_sampled_1000.fasta"
    cache_embeds_dir: str | None = "data/processed/embeds_cache"
    runtime_structure_dir: str | None = "activation_stores/default_runtime_structures"
    runtime_sequence_batch_size: int = 1
    runtime_activation_buffer_size: int = 32768
    cache_runtime_activations_in_memory: bool = False
    output_dir: str = "outputs/default"
    batch_size: int = 256
    max_steps: int = 1000
    epochs: int = 1
    learning_rate: float | None = None
    lr_scheduler: Literal["constant", "cosine", "linear"] = "constant"
    lr_warmup_steps: int = 0
    min_learning_rate: float = 0.0
    weight_decay: float = 0.0
    validation_fraction: float = 0.0
    validation_interval: int | None = None
    metrics_interval: int = 50
    seed: int = 0
    device: str = "auto"
    mixed_precision: bool = False

    def __post_init__(self) -> None:
        _require_positive("batch_size", self.batch_size)
        if self.activation_source not in {"store", "bioemu"}:
            raise ValueError("activation_source must be 'store' or 'bioemu'")
        _require_positive("runtime_sequence_batch_size", self.runtime_sequence_batch_size)
        _require_positive("runtime_activation_buffer_size", self.runtime_activation_buffer_size)
        if self.activation_source == "bioemu" and self.input_fasta is None:
            raise ValueError("input_fasta is required for BioEmu runtime activation sampling")
        _require_positive("max_steps", self.max_steps)
        _require_positive("epochs", self.epochs)
        if self.learning_rate is not None:
            _require_positive("learning_rate", self.learning_rate)
        if self.lr_scheduler not in {"constant", "cosine", "linear"}:
            raise ValueError("lr_scheduler must be 'constant', 'cosine', or 'linear'")
        if self.lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be >= 0")
        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate must be >= 0")
        if self.learning_rate is not None and self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must be <= learning_rate")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.validation_interval is not None:
            _require_positive("validation_interval", self.validation_interval)
        _require_positive("metrics_interval", self.metrics_interval)
        if self.device not in {"auto", "cpu", "cuda", "mps"} and not self.device.startswith(
            "cuda:"
        ):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', 'cuda:<index>', or 'mps'")


@dataclass(frozen=True)
class UniRef50Config:
    """Sampling settings for building FASTA inputs from UniRef50 clusters.

    UniRef50 FASTA entries are cluster representatives. The sampler streams
    through a UniRef50 FASTA file, keeps only representatives whose sequence
    length is at most ``max_sequence_length`` when set, and reservoir-samples
    ``num_clusters`` records into one output FASTA.
    """

    num_clusters: int = 1000
    max_sequence_length: int | None = 1024
    seed: int = 0

    def __post_init__(self) -> None:
        _require_positive("num_clusters", self.num_clusters)
        if self.max_sequence_length is not None:
            _require_positive("max_sequence_length", self.max_sequence_length)


@dataclass(frozen=True)
class AnalysisConfig:
    """Optional post-hoc interpretation inputs."""

    interpro_annotations_path: str | None = None
    evaluation_input_fasta: str | None = None
    evaluation_cache_embeds_dir: str | None = None
    evaluation_structure_dir: str | None = None
    evaluation_activation_archive: str | None = None
    rmsf_reference_structure_dir: str | None = None
    evaluation_samples_per_sequence: int = 32
    concept_min_positive_residues: int = 100
    concept_min_positive_domains: int = 10
    concept_min_positive_sequences: int = 10

    def __post_init__(self) -> None:
        if self.evaluation_samples_per_sequence <= 0:
            raise ValueError("evaluation_samples_per_sequence must be positive")
        if self.concept_min_positive_residues < 0:
            raise ValueError("concept_min_positive_residues must be >= 0")
        if self.concept_min_positive_domains < 0:
            raise ValueError("concept_min_positive_domains must be >= 0")
        if self.concept_min_positive_sequences < 0:
            raise ValueError("concept_min_positive_sequences must be >= 0")


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level configuration object persisted to YAML/JSON for each run.

    The project config intentionally groups the workflow into three phases:
    ``trace`` for BioEmu activation extraction, ``sae`` for model structure,
    and ``training`` for optimizer/evaluation settings. Loading also performs
    light backwards-compatible key coercion for older config names.
    """

    sae: SAEConfig = field(default_factory=SAEConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    uniref50: UniRef50Config = field(default_factory=UniRef50Config)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectConfig:
        # ESM3 and AlphaFold3 are not part of this release. A config naming either is REFUSED rather
        # than loaded with the section dropped: dropping it would run something other than what the
        # file asks for, and say nothing.
        unsupported = sorted({"esm3", "alphafold3"} & set(data))
        if unsupported:
            raise ValueError(
                f"config sections {unsupported} are not supported: this release runs BioEmu only"
            )
        return cls(
            sae=SAEConfig(**_coerce_sae(data.get("sae", {}))),
            trace=TraceConfig(**_coerce_trace(data.get("trace", {}))),
            training=TrainingConfig(**data.get("training", {})),
            uniref50=UniRef50Config(**data.get("uniref50", {})),
            analysis=AnalysisConfig(**data.get("analysis", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> ProjectConfig:
        path = Path(path)
        with path.open() as handle:
            if path.suffix == ".json":
                data = json.load(handle)
            else:
                data = yaml.safe_load(handle) or {}
        # ``${SAE_DATA_ROOT}`` / ``${SAE_OUTPUT_ROOT}`` so a config written on one machine runs on
        # another. Configs holding absolute paths are unaffected: they contain no placeholders.
        return cls.from_dict(expand_placeholders(data))

    def to_dict(self) -> dict[str, Any]:
        return _primitive_config_value(asdict(self))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            if path.suffix == ".json":
                json.dump(self.to_dict(), handle, indent=2)
            else:
                yaml.safe_dump(self.to_dict(), handle, sort_keys=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _coerce_sae(data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)
    if "timestep_embedding_dim" in data and "timestep_mlp_hidden_dim" not in data:
        data["timestep_mlp_hidden_dim"] = data.pop("timestep_embedding_dim")
    if "dead_steps_threshold" in data and "dead_tokens_threshold" not in data:
        data["dead_tokens_threshold"] = data.pop("dead_steps_threshold")
    return data


def _coerce_trace(data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)
    data.pop("timesteps", None)
    return data


def _primitive_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _primitive_config_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_primitive_config_value(item) for item in value]
    return value
