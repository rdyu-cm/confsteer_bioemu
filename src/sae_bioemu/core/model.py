from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from sae_bioemu.core.checkpoint_io import (
    load_sae_checkpoint,
    prepare_sae_checkpoint_payload,
)
from sae_bioemu.core.config import SAEConfig


@dataclass(frozen=True)
class SAEOutput:
    """Structured outputs from a forward pass of the time-conditioned SAE.

    ``latents_pre_act`` are dense encoder activations before TopK sparsification;
    ``latents`` are the sparse features used for the main reconstruction.
    ``baseline`` is the decoder output for an all-zero latent vector and is
    used as the auxiliary residual baseline when aux-k training is active.
    """

    latents_pre_act: torch.Tensor
    latents: torch.Tensor
    reconstruction: torch.Tensor
    baseline: torch.Tensor
    aux_latents: torch.Tensor | None = None
    aux_reconstruction: torch.Tensor | None = None


class TopK(nn.Module):
    """Keep only the largest ``k`` activations along the final tensor dimension.

    This module implements the hard sparsity constraint used by the SAE. Values
    outside the top-k set are zeroed, while selected values pass through a
    post-activation, ReLU by default.
    """

    def __init__(self, k: int, postact: nn.Module | None = None) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.postact = postact or nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.k > x.shape[-1]:
            raise ValueError("k cannot exceed the last tensor dimension")
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact(topk.values)
        out = torch.zeros_like(x)
        out.scatter_(-1, topk.indices, values)
        return out


class TimestepModulator(nn.Module):
    """Small FiLM-style network that conditions activations on diffusion time.

    BioEmu activations are collected at different denoising timesteps. This
    module maps each scalar timestep to a per-feature scale and shift, allowing
    the SAE encoder to represent features whose baseline behavior changes over
    the reverse-diffusion trajectory.
    """

    def __init__(self, hidden_dim: int, input_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * input_dim),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        params = self.net(timesteps.to(dtype=x.dtype, device=x.device)[:, None])
        scale, shift = params.chunk(2, dim=-1)
        return x * (1.0 + scale) + shift


class TimeConditionedTopKSAE(nn.Module):
    """Sparse autoencoder for BioEmu token activations with timestep conditioning.

    The model preprocesses each activation vector, optionally applies
    timestep-dependent modulation, encodes to ``n_latents`` features, keeps exactly ``top_k``
    nonzero features per token, and decodes back to the original activation
    space. Decoder columns are kept unit-normalized, and an optional aux-k path
    trains latents that have been inactive for many activation tokens.
    """

    def __init__(self, config: SAEConfig) -> None:
        super().__init__()
        self.config = config
        self.pre_bias = nn.Parameter(torch.zeros(config.input_dim))
        self.encoder = nn.Linear(config.input_dim, config.n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(config.n_latents))
        self.timestep_modulator = TimestepModulator(
            config.timestep_mlp_hidden_dim,
            config.input_dim,
        )
        self.activation = TopK(config.top_k)
        self.decoder = nn.Linear(config.n_latents, config.input_dim, bias=False)
        self.register_buffer("stats_last_nonzero", torch.zeros(config.n_latents, dtype=torch.long))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        self.normalize_decoder_()

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.config.normalize_inputs:
            return x, {}
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-5)
        return (x - mean) / std, {"mean": mean, "std": std}

    def modulate_input(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if not self.config.use_time_conditioning:
            return x
        return self.timestep_modulator(x, timesteps)

    def encode_pre_act(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        x = self.modulate_input(x, timesteps)
        return self.encoder(x - self.pre_bias) + self.latent_bias

    def encode(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        x, _ = self.preprocess(x)
        return self.activation(self.encode_pre_act(x, timesteps))

    def decode(
        self, latents: torch.Tensor, info: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        out = self.decoder(latents) + self.pre_bias
        if self.config.normalize_inputs:
            if info is None:
                raise ValueError("normalization info is required for decode")
            out = out * info["std"] + info["mean"]
        return out

    def auxiliary_latents(self, latents_pre_act: torch.Tensor) -> torch.Tensor | None:
        if self.config.aux_k is None or self.config.aux_k_coef == 0:
            return None
        dead_mask = self.stats_last_nonzero > self.config.dead_tokens_threshold
        if not dead_mask.any():
            return None
        masked = latents_pre_act.masked_fill(~dead_mask[None, :], float("-inf"))
        k = min(self.config.aux_k, int(dead_mask.sum().item()))
        topk = torch.topk(masked, k=k, dim=-1)
        values = torch.relu(topk.values)
        aux_latents = torch.zeros_like(latents_pre_act)
        aux_latents.scatter_(-1, topk.indices, values)
        return aux_latents

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> SAEOutput:
        if x.shape[-1] != self.config.input_dim:
            raise ValueError(f"expected input_dim={self.config.input_dim}, got {x.shape[-1]}")
        if timesteps.shape != (x.shape[0],):
            raise ValueError("timesteps must have shape [batch]")
        x_preprocessed, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x_preprocessed, timesteps)
        latents = self.activation(latents_pre_act)
        aux_latents = self.auxiliary_latents(latents_pre_act)
        reconstruction = self.decode(latents, info)
        baseline = self.decode(torch.zeros_like(latents), info)
        aux_reconstruction = self.decode(aux_latents, info) if aux_latents is not None else None
        self.update_dead_latent_stats(latents)
        return SAEOutput(
            latents_pre_act=latents_pre_act,
            latents=latents,
            reconstruction=reconstruction,
            baseline=baseline,
            aux_latents=aux_latents,
            aux_reconstruction=aux_reconstruction,
        )

    @torch.no_grad()
    def update_dead_latent_stats(self, latents: torch.Tensor) -> None:
        active = (latents > 1e-3).any(dim=0)
        self.stats_last_nonzero *= (~active).to(self.stats_last_nonzero.dtype)
        self.stats_last_nonzero += latents.shape[0]

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    @torch.no_grad()
    def remove_decoder_gradient_parallel_component_(self) -> None:
        if self.decoder.weight.grad is None:
            return
        parallel_component = (self.decoder.weight.grad * self.decoder.weight.data).sum(
            dim=0, keepdim=True
        )
        self.decoder.weight.grad -= parallel_component * self.decoder.weight.data

    @torch.no_grad()
    def initialize_from_data(self, activations: torch.Tensor, timesteps: torch.Tensor) -> None:
        if activations.ndim != 2 or activations.shape[1] != self.config.input_dim:
            raise ValueError("activations must have shape [tokens, input_dim]")
        if timesteps.shape != (activations.shape[0],):
            raise ValueError("timesteps must have shape [tokens]")
        activations = activations.to(self.pre_bias.device, dtype=self.pre_bias.dtype)
        timesteps = timesteps.to(self.pre_bias.device)
        encoder_inputs, _ = self.preprocess(activations)
        self.pre_bias.data.copy_(encoder_inputs.mean(dim=0))
        output = self(activations, timesteps)
        reconstruction_norm = (output.reconstruction - output.baseline).norm(dim=-1).mean()
        if torch.isfinite(reconstruction_norm) and reconstruction_norm > 1e-8:
            self.encoder.weight.data /= reconstruction_norm
        self.stats_last_nonzero.zero_()

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            prepare_sae_checkpoint_payload(
                {"config": self.config, "state_dict": self.state_dict()}
            ),
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> TimeConditionedTopKSAE:
        payload = load_sae_checkpoint(path)
        model = cls(payload["config"])
        model.load_state_dict(payload["state_dict"])
        return model

    def decoder_column_norms(self) -> torch.Tensor:
        return self.decoder.weight.norm(dim=0)
