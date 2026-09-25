from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActivationRecord:
    """Flat token table of BioEmu activations and their biological provenance.

    Each row corresponds to one residue token at one denoiser score-model call.
    ``activations`` has shape ``[tokens, dim]``; the parallel metadata tensors
    identify the denoising timestep, residue index, sequence id, and optional
    stochastic sample/repetition index for every token.
    """

    activations: torch.Tensor
    timesteps: torch.Tensor
    sequence_ids: tuple[str, ...]
    residue_indices: torch.Tensor
    sample_indices: torch.Tensor | None = None
    mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.activations.ndim != 2:
            raise ValueError("activations must have shape [tokens, dim]")
        tokens = self.activations.shape[0]
        if self.timesteps.shape != (tokens,):
            raise ValueError("timesteps must have shape [tokens]")
        if self.residue_indices.shape != (tokens,):
            raise ValueError("residue_indices must have shape [tokens]")
        if len(self.sequence_ids) != tokens:
            raise ValueError("sequence_ids must have one id per token")
        if self.sample_indices is not None and self.sample_indices.shape != (tokens,):
            raise ValueError("sample_indices must have shape [tokens]")
        if self.mask is not None and self.mask.shape != (tokens,):
            raise ValueError("mask must have shape [tokens]")

    @property
    def n_tokens(self) -> int:
        return int(self.activations.shape[0])

    @property
    def activation_dim(self) -> int:
        return int(self.activations.shape[1])

    @classmethod
    def concat(cls, records: list[ActivationRecord]) -> ActivationRecord:
        if not records:
            raise ValueError("records must be non-empty")
        activation_dim = records[0].activation_dim
        if any(record.activation_dim != activation_dim for record in records):
            raise ValueError("all records must have the same activation_dim")
        sample_indices = None
        if all(record.sample_indices is not None for record in records):
            sample_indices = torch.cat(
                [record.sample_indices for record in records if record.sample_indices is not None]
            )
        mask = None
        if all(record.mask is not None for record in records):
            mask = torch.cat([record.mask for record in records if record.mask is not None])
        return cls(
            activations=torch.cat([record.activations for record in records], dim=0),
            timesteps=torch.cat([record.timesteps for record in records], dim=0),
            sequence_ids=tuple(
                sequence_id for record in records for sequence_id in record.sequence_ids
            ),
            residue_indices=torch.cat([record.residue_indices for record in records], dim=0),
            sample_indices=sample_indices,
            mask=mask,
        )

    @classmethod
    def from_dense(
        cls,
        activations: torch.Tensor,
        timestep: float | torch.Tensor,
        sequence_ids: str | list[str] | tuple[str, ...],
        mask: torch.Tensor | None = None,
        sample_indices: int | list[int] | tuple[int, ...] | torch.Tensor | None = None,
    ) -> ActivationRecord:
        if activations.ndim != 3:
            raise ValueError("dense activations must have shape [batch, length, dim]")
        batch, length, dim = activations.shape
        flat_mask = torch.ones(batch, length, dtype=torch.bool, device=activations.device)
        if mask is not None:
            if mask.shape != (batch, length):
                raise ValueError("mask must have shape [batch, length]")
            flat_mask = mask.bool()

        flat = activations.reshape(batch * length, dim)
        keep = flat_mask.reshape(batch * length)
        flat = flat[keep]

        if isinstance(timestep, torch.Tensor):
            if timestep.ndim == 0:
                timestep_values = torch.full((batch, length), float(timestep.item()))
            elif timestep.shape == (batch,):
                timestep_values = timestep[:, None].expand(batch, length)
            elif timestep.shape == (batch, length):
                timestep_values = timestep
            else:
                raise ValueError("timestep tensor must be scalar, [batch], or [batch, length]")
        else:
            timestep_values = torch.full((batch, length), float(timestep))
        timesteps = timestep_values.reshape(batch * length).to(flat.device)[keep].to(torch.float32)

        residues = torch.arange(length, device=activations.device).expand(batch, length)
        residue_indices = residues.reshape(batch * length)[keep].to(torch.long)
        if sample_indices is None:
            sample_index_values = torch.arange(batch, device=activations.device)
        elif isinstance(sample_indices, int):
            sample_index_values = torch.full((batch,), sample_indices, device=activations.device)
        elif isinstance(sample_indices, torch.Tensor):
            if sample_indices.shape != (batch,):
                raise ValueError("sample_indices tensor must have shape [batch]")
            sample_index_values = sample_indices.to(device=activations.device, dtype=torch.long)
        else:
            if len(sample_indices) != batch:
                raise ValueError("sample_indices must have one index per dense batch row")
            sample_index_values = torch.tensor(
                list(sample_indices),
                device=activations.device,
                dtype=torch.long,
            )
        sample_grid = sample_index_values[:, None].expand(batch, length)
        flat_sample_indices = sample_grid.reshape(batch * length)[keep].to(torch.long)
        batch_rows = torch.arange(batch, device=activations.device)[:, None].expand(batch, length)

        if isinstance(sequence_ids, str):
            seq_by_batch = [sequence_ids] * batch
        else:
            seq_by_batch = list(sequence_ids)
            if len(seq_by_batch) != batch:
                raise ValueError("sequence_ids must have one id per dense batch row")
        flat_sequence_ids = [
            seq_by_batch[batch_row]
            for batch_row, keep_token in zip(
                batch_rows.reshape(batch * length).detach().cpu().tolist(),
                keep.detach().cpu().tolist(),
                strict=True,
            )
            if keep_token
        ]

        return cls(
            activations=flat.detach().to(torch.float32).cpu(),
            timesteps=timesteps.detach().cpu(),
            sequence_ids=tuple(flat_sequence_ids),
            residue_indices=residue_indices.detach().cpu(),
            sample_indices=flat_sample_indices.detach().cpu(),
            mask=None,
        )
