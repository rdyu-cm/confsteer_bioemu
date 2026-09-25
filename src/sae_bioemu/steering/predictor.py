"""Masked direct-property prediction primitives."""

from __future__ import annotations

import torch
from torch import nn


def _validate_masked_values(values: torch.Tensor, mask: torch.Tensor, name: str) -> None:
    if values.ndim != 3 or not values.is_floating_point():
        raise ValueError(f"{name} must be a floating [batch, residues, width] tensor")
    if mask.ndim != 2 or mask.dtype is not torch.bool:
        raise ValueError("mask must be a boolean [batch, residues] tensor")
    if values.shape[:2] != mask.shape:
        raise ValueError(f"{name} and mask shapes must agree")
    if values.device != mask.device:
        raise ValueError(f"{name} and mask must be on the same device")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("each sequence must contain at least one valid residue")
    if not torch.isfinite(values[mask]).all():
        raise ValueError(f"valid {name} elements must be finite")


def _validate_scalar(
    values: torch.Tensor, name: str, batch_size: int, reference: torch.Tensor
) -> None:
    if values.ndim != 1 or values.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape [batch]")
    if (
        not values.is_floating_point()
        or values.device != reference.device
        or values.dtype != reference.dtype
    ):
        raise ValueError(f"{name} must match activation device and dtype")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite")


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a per-sequence mean over valid residues only."""
    _validate_masked_values(values, mask, "values")
    weights = mask.unsqueeze(-1).to(dtype=values.dtype)
    return (torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values)) * weights).sum(
        dim=1
    ) / weights.sum(dim=1)


def masked_rms(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a per-sequence RMS over every valid residue-coordinate element."""
    _validate_masked_values(values, mask, "values")
    valid_values = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values))
    scale = valid_values.abs().amax(dim=(1, 2))
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    normalized_values = valid_values / safe_scale[:, None, None]
    denominator = mask.sum(dim=1).to(dtype=values.dtype) * values.shape[-1]
    mean_square = normalized_values.square().sum(dim=(1, 2)) / denominator
    safe_mean_square = torch.where(scale == 0, torch.ones_like(mean_square), mean_square)
    rms = safe_scale * torch.sqrt(safe_mean_square)
    return torch.where(scale == 0, torch.zeros_like(rms), rms)


class ScalarStandardizer(nn.Module):
    """Standardization fitted on scalar training coordinates."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        if mean.ndim != 0 or std.ndim != 0:
            raise ValueError("mean and std must be scalar tensors")
        if not mean.is_floating_point() or mean.device != std.device or mean.dtype != std.dtype:
            raise ValueError("mean and std must share a floating dtype and device")
        if not torch.isfinite(mean) or not torch.isfinite(std) or std <= 0:
            raise ValueError("mean must be finite and std must be finite and positive")
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())

    @classmethod
    def fit(cls, values: torch.Tensor) -> ScalarStandardizer:
        """Fit moments from a nonempty vector of training coordinates."""
        if values.ndim != 1 or not values.is_floating_point() or values.numel() == 0:
            raise ValueError("values must be a nonempty floating [rows] tensor")
        if not torch.isfinite(values).all():
            raise ValueError("values must be finite")
        return cls(values.mean(), values.std(unbiased=False).clamp_min(1e-6))

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the fitted scalar transform without changing shape."""
        if values.ndim != 1 or not values.is_floating_point():
            raise ValueError("values must be a floating [rows] tensor")
        if values.device != self.mean.device or values.dtype != self.mean.dtype:
            raise ValueError("values must match fitted standardizer device and dtype")
        if not torch.isfinite(values).all():
            raise ValueError("values must be finite")
        return (values - self.mean) / self.std

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.transform(values)


class MaskedStandardizer(nn.Module):
    """Feature-wise standardization fitted only on valid training residues."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
            raise ValueError("mean and std must be matching one-dimensional tensors")
        if not mean.is_floating_point() or mean.device != std.device or mean.dtype != std.dtype:
            raise ValueError("mean and std must share a floating dtype and device")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("mean must be finite and std must be finite and positive")
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())

    @classmethod
    def fit(cls, values: torch.Tensor, mask: torch.Tensor) -> MaskedStandardizer:
        """Fit feature moments from valid training residues alone."""
        _validate_masked_values(values, mask, "values")
        training_rows = values[mask]
        mean = training_rows.mean(dim=0)
        std = training_rows.std(dim=0, unbiased=False).clamp_min(1e-6)
        return cls(mean, std)

    def transform(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Standardize valid rows and retain zero-valued padding."""
        _validate_masked_values(values, mask, "values")
        if values.shape[-1] != self.mean.shape[0]:
            raise ValueError("values width must match fitted standardizer")
        if values.device != self.mean.device or values.dtype != self.mean.dtype:
            raise ValueError("values must match fitted standardizer device and dtype")
        standardized = (values - self.mean) / self.std
        return torch.where(mask.unsqueeze(-1), standardized, torch.zeros_like(standardized))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.transform(values, mask)


class DirectPropertyPredictor(nn.Module):
    """Small MLP over declared masked representation summaries."""

    def __init__(self, width: int, hidden_width: int, sae_width: int | None = None) -> None:
        super().__init__()
        if not isinstance(width, int) or width <= 0:
            raise ValueError("width must be a positive integer")
        if not isinstance(hidden_width, int) or hidden_width <= 0:
            raise ValueError("hidden_width must be a positive integer")
        if sae_width is not None and (not isinstance(sae_width, int) or sae_width <= 0):
            raise ValueError("sae_width must be a positive integer when provided")
        self.width = width
        self.sae_width = sae_width
        input_width = width + 2 + (sae_width or 0)
        self.hidden = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden_width, 1)

    def forward(
        self,
        activations: torch.Tensor,
        mask: torch.Tensor,
        standardized_call: torch.Tensor,
        sae_coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_masked_values(activations, mask, "activations")
        if activations.shape[-1] != self.width:
            raise ValueError("activation width does not match predictor")
        _validate_scalar(standardized_call, "standardized_call", activations.shape[0], activations)
        if self.sae_width is None:
            if sae_coordinates is not None:
                raise ValueError("predictor was not configured for SAE coordinates")
            sae_summary: tuple[torch.Tensor, ...] = ()
        else:
            if sae_coordinates is None:
                raise ValueError("SAE coordinates are required by this predictor")
            _validate_masked_values(sae_coordinates, mask, "sae_coordinates")
            if sae_coordinates.shape[-1] != self.sae_width:
                raise ValueError("SAE coordinate width does not match predictor")
            if (
                sae_coordinates.device != activations.device
                or sae_coordinates.dtype != activations.dtype
            ):
                raise ValueError("SAE coordinates must match activation device and dtype")
            sae_summary = (masked_mean(sae_coordinates, mask),)
        sequence_length = mask.sum(dim=1).to(dtype=activations.dtype).unsqueeze(-1)
        inputs = torch.cat(
            (
                masked_mean(activations, mask),
                *sae_summary,
                standardized_call.unsqueeze(-1),
                sequence_length,
            ),
            dim=-1,
        )
        return self.output(self.hidden(inputs)).squeeze(-1)
