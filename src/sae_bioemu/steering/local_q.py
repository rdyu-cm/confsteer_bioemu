"""Exact differentiable residue-level Local-Q steering utilities."""

from __future__ import annotations

import torch

from sae_bioemu.ensemble.properties import (
    CONTACT_CUTOFF_ANGSTROM,
    LOCAL_Q_BETA_PER_ANGSTROM,
    LOCAL_Q_DISTANCE_MULTIPLIER,
    MIN_SEQUENCE_SEPARATION,
)


def directional_local_q_targets(
    baseline: torch.Tensor, control: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Map directional controls to baseline-relative Local-Q targets."""
    if baseline.shape != control.shape or baseline.shape != mask.shape or baseline.ndim != 2:
        raise ValueError("baseline, control, and mask must share shape [batch, residues]")
    if (
        not baseline.is_floating_point()
        or not control.is_floating_point()
        or mask.dtype is not torch.bool
    ):
        raise ValueError("baseline/control must be floating and mask must be boolean")
    if baseline.device != control.device or baseline.device != mask.device:
        raise ValueError("baseline, control, and mask must share a device")
    if not torch.isfinite(baseline[mask]).all() or not torch.isfinite(control[mask]).all():
        raise ValueError("valid baseline and control values must be finite")
    if torch.any((baseline[mask] < 0) | (baseline[mask] > 1)):
        raise ValueError("valid Local-Q baselines must lie in [0, 1]")
    if torch.any((control[mask] < -1) | (control[mask] > 1)):
        raise ValueError("valid Local-Q controls must lie in [-1, 1]")
    positive = baseline + control * (1.0 - baseline)
    negative = baseline + control * baseline
    target = torch.where(control >= 0, positive, negative)
    return torch.where(mask, target, torch.zeros_like(target))


def reference_local_q_contacts(
    reference: torch.Tensor,
    *,
    cutoff: float = CONTACT_CUTOFF_ANGSTROM,
    min_sequence_separation: int = MIN_SEQUENCE_SEPARATION,
) -> torch.Tensor:
    """Derive the frozen reference-contact mask used by Local Q."""
    if reference.ndim != 2 or reference.shape[-1] != 3 or not reference.is_floating_point():
        raise ValueError("reference must be floating [residues, 3]")
    if not torch.isfinite(reference).all():
        raise ValueError("reference coordinates must be finite")
    if cutoff <= 0 or min_sequence_separation < 1:
        raise ValueError("Local-Q contact parameters must be positive")
    distances = torch.linalg.vector_norm(reference[:, None] - reference[None, :], dim=-1)
    indices = torch.arange(reference.shape[0], device=reference.device)
    eligible = (indices[:, None] - indices[None, :]).abs() >= min_sequence_separation
    return eligible & (distances < cutoff)


def differentiable_local_q(
    coordinates: torch.Tensor,
    reference: torch.Tensor,
    reference_contacts: torch.Tensor,
    residue_mask: torch.Tensor,
    *,
    beta: float = LOCAL_Q_BETA_PER_ANGSTROM,
    distance_multiplier: float = LOCAL_Q_DISTANCE_MULTIPLIER,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return residue Local Q and its valid/contact-defined supervision mask."""
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3 or not coordinates.is_floating_point():
        raise ValueError("coordinates must be floating [batch, residues, 3]")
    if reference.shape != coordinates.shape[1:] or reference.dtype != coordinates.dtype:
        raise ValueError("reference must match coordinate residue shape and dtype")
    if (
        reference_contacts.shape != coordinates.shape[1:2] * 2
        or reference_contacts.dtype is not torch.bool
    ):
        raise ValueError("reference_contacts must be boolean [residues, residues]")
    if residue_mask.shape != coordinates.shape[:2] or residue_mask.dtype is not torch.bool:
        raise ValueError("residue_mask must be boolean [batch, residues]")
    if any(
        value.device != coordinates.device
        for value in (reference, reference_contacts, residue_mask)
    ):
        raise ValueError("Local-Q tensors must share a device")
    if beta <= 0 or distance_multiplier <= 0:
        raise ValueError("Local-Q smooth-distance parameters must be positive")
    if not torch.isfinite(coordinates[residue_mask]).all() or not torch.isfinite(reference).all():
        raise ValueError("Local-Q coordinates must be finite")
    reference_distances = torch.linalg.vector_norm(reference[:, None] - reference[None, :], dim=-1)
    target_distances = torch.linalg.vector_norm(
        coordinates[:, :, None] - coordinates[:, None, :], dim=-1
    )
    weights = torch.sigmoid(
        -beta * (target_distances - distance_multiplier * reference_distances.unsqueeze(0))
    )
    pair_mask = (
        reference_contacts.unsqueeze(0) & residue_mask.unsqueeze(1) & residue_mask.unsqueeze(2)
    )
    counts = pair_mask.sum(dim=-1)
    safe_counts = counts.clamp_min(1).to(coordinates.dtype)
    values = (weights * pair_mask).sum(dim=-1) / safe_counts
    valid = residue_mask & (counts > 0)
    return torch.where(valid, values, torch.zeros_like(values)), valid
