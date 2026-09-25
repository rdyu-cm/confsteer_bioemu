from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LossOutput:
    """Loss scalars and diagnostic metrics emitted for one SAE batch."""

    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    auxk_loss: torch.Tensor
    l0: torch.Tensor
    explained_variance: torch.Tensor


class SAELoss(nn.Module):
    """Normalized reconstruction objective plus optional auxiliary-k loss.

    The main term is MSE normalized by the original activation energy per token,
    which makes losses more comparable across BioEmu timesteps and residues.
    When aux-k reconstruction is provided, it is trained against the detached
    residual so dead latents can learn useful directions without changing the
    main reconstruction target.
    """

    def __init__(self, auxk_coef: float = 0.0) -> None:
        super().__init__()
        if auxk_coef < 0:
            raise ValueError("auxk_coef must be >= 0")
        self.auxk_coef = auxk_coef

    def forward(
        self,
        reconstruction: torch.Tensor,
        original: torch.Tensor,
        latents: torch.Tensor,
        aux_reconstruction: torch.Tensor | None = None,
        aux_target_baseline: torch.Tensor | None = None,
    ) -> LossOutput:
        reconstruction_loss = normalized_mse(reconstruction, original)
        auxk_loss = torch.zeros((), device=original.device, dtype=original.dtype)
        if aux_reconstruction is not None and self.auxk_coef > 0:
            residual_target = original - reconstruction.detach()
            if aux_target_baseline is not None:
                residual_target = residual_target + aux_target_baseline.detach()
            auxk_loss = normalized_mse(aux_reconstruction, residual_target).nan_to_num(0)
        loss = reconstruction_loss + self.auxk_coef * auxk_loss
        l0 = (latents != 0).sum(dim=-1).float().mean()
        explained_variance = _explained_variance(reconstruction, original)
        return LossOutput(loss, reconstruction_loss, auxk_loss, l0, explained_variance)


def normalized_mse(reconstruction: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    denom = original.pow(2).mean(dim=-1).clamp_min(1e-8)
    return (reconstruction - original).pow(2).mean(dim=-1).div(denom).mean()


def _explained_variance(reconstruction: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    residual_var = (original - reconstruction).var()
    original_var = original.var().clamp_min(1e-8)
    return 1.0 - residual_var / original_var
