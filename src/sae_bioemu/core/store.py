from __future__ import annotations

import json
import os
import shutil
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sae_bioemu.core.types import ActivationRecord


@dataclass(frozen=True)
class StoreBatch:
    """Mini-batch read from an activation store for SAE training/evaluation."""

    activations: torch.Tensor
    timesteps: torch.Tensor


@dataclass(frozen=True)
class StoreMetadata:
    """Manifest metadata describing how an activation store was produced.

    Metadata is used as a compatibility guard: re-opening an existing store
    with different extraction settings raises instead of silently appending
    incompatible activations. Sequence text is stored so eval reports can map
    residue indices back to amino-acid identities.
    """

    source_name: str
    model_name: str
    target_module: str
    sequence_count: int | None = None
    sequences: dict[str, str] | None = None
    denoiser_type: str | None = None
    denoiser_config_path: str | None = None
    recorded_score_calls: int | None = None
    samples_per_sequence: int | None = None
    structure_index: str | None = None

    def to_manifest(self) -> dict:
        payload = asdict(self)
        if payload["sequences"] is not None:
            payload["sequences"] = dict(sorted(payload["sequences"].items()))
        return payload


@dataclass(frozen=True)
class ShardIndexGroup:
    """Global token indices grouped by the shard file that contains them."""

    shard_idx: int
    file_name: str
    local_indices: torch.Tensor


class ActivationStoreWriter:
    """Buffered writer that packs many activation records into larger shards.

    The extractor naturally produces one ``ActivationRecord`` per recorded
    denoiser call. Writing each call as its own shard would create many small
    files, so this writer concatenates pending records until
    ``max_tokens_per_shard`` is reached and then flushes one shard.
    """

    def __init__(
        self,
        store: ActivationStore,
        max_tokens_per_shard: int = 1_000_000,
        *,
        auto_flush: bool = True,
    ) -> None:
        if max_tokens_per_shard <= 0:
            raise ValueError("max_tokens_per_shard must be positive")
        self.store = store
        self.max_tokens_per_shard = max_tokens_per_shard
        self.auto_flush = auto_flush
        self.pending_records: list[ActivationRecord] = []
        self.pending_tokens = 0

    @property
    def should_flush(self) -> bool:
        return self.pending_tokens >= self.max_tokens_per_shard

    def append(self, record: ActivationRecord) -> None:
        self.store._ensure_writable()
        self.pending_records.append(record)
        self.pending_tokens += record.n_tokens
        if self.auto_flush and self.should_flush:
            self.flush()

    def flush(self) -> None:
        self.store._ensure_writable()
        if not self.pending_records:
            return
        self.store.append(ActivationRecord.concat(self.pending_records))
        self.pending_records = []
        self.pending_tokens = 0


class ActivationStore:
    """On-disk sharded dataset of BioEmu activation tokens.

    A store directory contains a ``manifest.json`` plus ``shards/*.pt`` files.
    The manifest records shard sizes and extraction metadata; each shard holds
    an ``ActivationRecord`` payload. Training can stream whole shards for
    locality or address shuffled global token indices for validation splits.
    """

    def __init__(
        self,
        root: str | Path,
        metadata: StoreMetadata | None = None,
        *,
        read_only: bool = False,
    ) -> None:
        self.root = Path(root)
        self.read_only = read_only
        self.manifest_path = self.root / "manifest.json"
        self.shard_dir = self.root / "shards"
        if read_only:
            if (
                not self.root.is_dir()
                or not self.shard_dir.is_dir()
                or not self.manifest_path.is_file()
            ):
                raise FileNotFoundError(f"read-only activation store is incomplete: {self.root}")
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_dir.mkdir(exist_ok=True)
        if not self.manifest_path.exists():
            self._write_manifest(
                {
                    "version": 1,
                    "activation_dim": None,
                    "metadata": metadata.to_manifest() if metadata else None,
                    "shards": [],
                }
            )
        elif metadata is not None:
            self._validate_metadata(metadata)

    @property
    def manifest(self) -> dict:
        with self.manifest_path.open() as handle:
            return json.load(handle)

    def append(self, record: ActivationRecord) -> Path:
        self._ensure_writable()
        manifest = self.manifest
        activation_dim = manifest.get("activation_dim")
        if activation_dim is None:
            manifest["activation_dim"] = record.activation_dim
        elif activation_dim != record.activation_dim:
            raise ValueError(
                f"record activation_dim={record.activation_dim} does not match store={activation_dim}"
            )

        shard_idx = len(manifest["shards"])
        shard_name = f"shard_{shard_idx:06d}.pt"
        shard_path = self.shard_dir / shard_name
        torch.save(_record_to_payload(record), shard_path)
        sample_indices = record.sample_indices
        unique_sample_indices = (
            torch.unique(sample_indices).tolist() if sample_indices is not None else []
        )
        manifest["shards"].append(
            {
                "file": str(Path("shards") / shard_name),
                "num_tokens": record.n_tokens,
                "activation_dim": record.activation_dim,
                "dtype": str(record.activations.dtype),
                "sample_index_min": min(unique_sample_indices) if unique_sample_indices else None,
                "sample_index_max": max(unique_sample_indices) if unique_sample_indices else None,
                "sample_index_count": len(unique_sample_indices),
                "sequence_ids": sorted(set(record.sequence_ids)),
            }
        )
        self._write_manifest(manifest)
        return shard_path

    def __len__(self) -> int:
        return sum(int(shard["num_tokens"]) for shard in self.manifest["shards"])

    def iter_records(self) -> Iterator[ActivationRecord]:
        for shard in self.manifest["shards"]:
            yield self.read_shard(shard["file"])

    def read_shard(self, file_name: str | Path) -> ActivationRecord:
        payload = torch.load(self.root / file_name, map_location="cpu", weights_only=False)
        return _payload_to_record(payload)

    def load_all(self) -> StoreBatch:
        records = list(self.iter_records())
        if not records:
            raise ValueError("activation store is empty")
        return StoreBatch(
            activations=torch.cat([record.activations for record in records], dim=0),
            timesteps=torch.cat([record.timesteps for record in records], dim=0),
        )

    def train_val_indices(
        self, validation_fraction: float, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_tokens = len(self)
        if n_tokens == 0:
            raise ValueError("activation store is empty")
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_tokens, generator=generator)
        n_val = int(round(n_tokens * validation_fraction))
        val = perm[:n_val]
        train = perm[n_val:]
        if len(train) == 0:
            raise ValueError("validation_fraction leaves no training tokens")
        return train, val

    def iter_batches(
        self,
        batch_size: int,
        indices: torch.Tensor | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[StoreBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        n_tokens = len(self)
        if n_tokens == 0:
            raise ValueError("activation store is empty")
        if indices is None:
            indices = torch.arange(n_tokens)
        if shuffle:
            generator = torch.Generator().manual_seed(seed)
            indices = indices[torch.randperm(indices.numel(), generator=generator)]
        for start in range(0, indices.numel(), batch_size):
            batch_idx = indices[start : start + batch_size]
            yield self._read_global_indices(batch_idx)

    def iter_shard_batches(
        self,
        batch_size: int,
        indices: torch.Tensor | None = None,
        shuffle_shards: bool = True,
        shuffle_tokens: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> Iterator[StoreBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        groups = self.group_indices_by_shard(indices)
        generator = torch.Generator().manual_seed(seed)
        if shuffle_shards and len(groups) > 1:
            order = torch.randperm(len(groups), generator=generator).tolist()
            groups = [groups[idx] for idx in order]
        for group in groups:
            record = self.read_shard(group.file_name)
            local_indices = group.local_indices
            if shuffle_tokens and local_indices.numel() > 1:
                local_indices = local_indices[
                    torch.randperm(local_indices.numel(), generator=generator)
                ]
            for start in range(0, local_indices.numel(), batch_size):
                batch_idx = local_indices[start : start + batch_size]
                if drop_last and batch_idx.numel() < batch_size:
                    continue
                yield StoreBatch(
                    activations=record.activations[batch_idx],
                    timesteps=record.timesteps[batch_idx],
                )

    def group_indices_by_shard(self, indices: torch.Tensor | None = None) -> list[ShardIndexGroup]:
        manifest = self.manifest
        n_tokens = len(self)
        if n_tokens == 0:
            raise ValueError("activation store is empty")
        if indices is None:
            indices = torch.arange(n_tokens)
        shard_offsets = _shard_offsets(manifest)
        groups: dict[int, list[int]] = {}
        for global_idx in indices.detach().cpu().tolist():
            shard_idx, local_idx = _locate_global_index(int(global_idx), shard_offsets)
            groups.setdefault(shard_idx, []).append(local_idx)
        return [
            ShardIndexGroup(
                shard_idx=shard_idx,
                file_name=manifest["shards"][shard_idx]["file"],
                local_indices=torch.tensor(local_indices, dtype=torch.long),
            )
            for shard_idx, local_indices in groups.items()
        ]

    def _write_manifest(self, manifest: dict) -> None:
        self._ensure_writable()
        temporary_path = self.root / f".manifest.json.{os.getpid()}.tmp"
        with temporary_path.open("w") as handle:
            json.dump(manifest, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(self.manifest_path)

    def _validate_metadata(self, metadata: StoreMetadata) -> None:
        expected = metadata.to_manifest()
        existing = self.manifest.get("metadata")
        if existing is None:
            manifest = self.manifest
            manifest["metadata"] = expected
            self._write_manifest(manifest)
            return
        if existing != expected:
            raise ValueError(
                f"ActivationStore at {self.root} has incompatible metadata. "
                f"Existing={existing}; requested={expected}"
            )

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise PermissionError(f"read-only activation store cannot be modified: {self.root}")

    def _read_global_indices(self, indices: torch.Tensor) -> StoreBatch:
        manifest = self.manifest
        shard_offsets = _shard_offsets(manifest)
        activations: list[torch.Tensor] = []
        timesteps: list[torch.Tensor] = []
        shard_cache: dict[int, ActivationRecord] = {}
        for global_idx in indices.detach().cpu().tolist():
            shard_idx, local_idx = _locate_global_index(int(global_idx), shard_offsets)
            if shard_idx not in shard_cache:
                shard_cache[shard_idx] = self.read_shard(manifest["shards"][shard_idx]["file"])
            record = shard_cache[shard_idx]
            activations.append(record.activations[local_idx])
            timesteps.append(record.timesteps[local_idx])
        return StoreBatch(
            activations=torch.stack(activations, dim=0),
            timesteps=torch.stack(timesteps, dim=0),
        )


def merge_activation_stores(
    source_roots: list[str | Path],
    output_root: str | Path,
    overwrite: bool = False,
) -> ActivationStore:
    if not source_roots:
        raise ValueError("source_roots must be non-empty")
    output_root = Path(output_root)
    source_roots = [Path(root) for root in source_roots]
    output_resolved = output_root.resolve()
    if any(root.resolve() == output_resolved for root in source_roots):
        raise ValueError("output_root cannot also be a source store")
    if output_root.exists():
        if not overwrite:
            existing_manifest = output_root / "manifest.json"
            if existing_manifest.exists():
                with existing_manifest.open() as handle:
                    manifest = json.load(handle)
                if manifest.get("shards"):
                    raise ValueError(
                        f"output store {output_root} already contains shards; "
                        "choose a new path or pass overwrite=True"
                    )
            elif any(output_root.iterdir()):
                raise ValueError(
                    f"output path {output_root} already exists and is not an activation store"
                )
        else:
            shutil.rmtree(output_root)

    stores = [ActivationStore(root) for root in source_roots]
    metadata = _merge_store_metadata([store.manifest.get("metadata") for store in stores])
    merged = ActivationStore(output_root, metadata=metadata)
    expected_dim: int | None = None
    for store in stores:
        manifest = store.manifest
        activation_dim = manifest.get("activation_dim")
        if activation_dim is not None:
            if expected_dim is None:
                expected_dim = int(activation_dim)
            elif int(activation_dim) != expected_dim:
                raise ValueError("source stores have incompatible activation dimensions")
        for record in store.iter_records():
            merged.append(record)
    _merge_structure_indices(stores, merged)
    return merged


def repack_activation_store(
    source_root: str | Path,
    output_root: str | Path,
    max_tokens_per_shard: int = 1_000_000,
    buffer_shards: int = 4,
    seed: int = 0,
    overwrite: bool = False,
) -> ActivationStore:
    """Rewrite an activation store into shuffled, mixed shards.

    The repack reads a bounded window of source shards, concatenates their
    tokens, shuffles the window, and writes the shuffled tokens into fresh
    output shards. Larger ``buffer_shards`` gives stronger cross-sequence
    mixing; setting it to the number of source shards gives a full global
    shuffle at the cost of loading the whole store.
    """

    if max_tokens_per_shard <= 0:
        raise ValueError("max_tokens_per_shard must be positive")
    if buffer_shards <= 0:
        raise ValueError("buffer_shards must be positive")
    source_root = Path(source_root)
    output_root = Path(output_root)
    if source_root.resolve() == output_root.resolve():
        raise ValueError("output_root cannot be the source store")
    if output_root.exists():
        if not overwrite:
            existing_manifest = output_root / "manifest.json"
            if existing_manifest.exists():
                with existing_manifest.open() as handle:
                    manifest = json.load(handle)
                if manifest.get("shards"):
                    raise ValueError(
                        f"output store {output_root} already contains shards; "
                        "choose a new path or pass overwrite=True"
                    )
            elif any(output_root.iterdir()):
                raise ValueError(
                    f"output path {output_root} already exists and is not an activation store"
                )
        else:
            shutil.rmtree(output_root)

    source = ActivationStore(source_root)
    source_manifest = source.manifest
    metadata = _store_metadata_from_manifest(source_manifest.get("metadata"))
    output = ActivationStore(output_root, metadata=metadata)
    writer = ActivationStoreWriter(output, max_tokens_per_shard=max_tokens_per_shard)
    generator = torch.Generator().manual_seed(seed)
    shard_files = [shard["file"] for shard in source_manifest["shards"]]
    if len(shard_files) > 1:
        shard_order = torch.randperm(len(shard_files), generator=generator).tolist()
        shard_files = [shard_files[idx] for idx in shard_order]

    try:
        for start in range(0, len(shard_files), buffer_shards):
            records = [
                source.read_shard(file_name)
                for file_name in shard_files[start : start + buffer_shards]
            ]
            if not records:
                continue
            record = ActivationRecord.concat(records)
            token_order = torch.randperm(record.n_tokens, generator=generator)
            for chunk_start in range(0, record.n_tokens, max_tokens_per_shard):
                chunk_indices = token_order[chunk_start : chunk_start + max_tokens_per_shard]
                writer.append(_slice_record(record, chunk_indices))
    finally:
        writer.flush()

    _copy_structure_index_if_present(source, output)
    return output


def _record_to_payload(record: ActivationRecord) -> dict:
    return {
        "activations": record.activations,
        "timesteps": record.timesteps,
        "sequence_ids": list(record.sequence_ids),
        "residue_indices": record.residue_indices,
        "sample_indices": record.sample_indices,
        "mask": record.mask,
    }


def _payload_to_record(payload: dict) -> ActivationRecord:
    return ActivationRecord(
        activations=payload["activations"],
        timesteps=payload["timesteps"],
        sequence_ids=tuple(payload["sequence_ids"]),
        residue_indices=payload["residue_indices"],
        sample_indices=payload.get("sample_indices"),
        mask=payload.get("mask"),
    )


def _slice_record(record: ActivationRecord, indices: torch.Tensor) -> ActivationRecord:
    indices = indices.detach().cpu().to(torch.long)
    sequence_ids = tuple(record.sequence_ids[int(idx)] for idx in indices.tolist())
    sample_indices = record.sample_indices[indices] if record.sample_indices is not None else None
    mask = record.mask[indices] if record.mask is not None else None
    return ActivationRecord(
        activations=record.activations[indices],
        timesteps=record.timesteps[indices],
        sequence_ids=sequence_ids,
        residue_indices=record.residue_indices[indices],
        sample_indices=sample_indices,
        mask=mask,
    )


def _store_metadata_from_manifest(metadata: dict | None) -> StoreMetadata | None:
    if metadata is None:
        return None
    fields = StoreMetadata.__dataclass_fields__
    return StoreMetadata(**{key: metadata.get(key) for key in fields})


def _copy_structure_index_if_present(source: ActivationStore, output: ActivationStore) -> None:
    metadata = source.manifest.get("metadata") or {}
    structure_index = metadata.get("structure_index")
    if structure_index is None:
        return
    source_index = source.root / structure_index
    if not source_index.exists():
        raise ValueError(f"missing structure index {source_index}")
    source_dir = source_index.parent
    target_dir = output.root / Path(structure_index).parent
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)


def _merge_store_metadata(metadata_items: list[dict | None]) -> StoreMetadata | None:
    present = [metadata for metadata in metadata_items if metadata is not None]
    if not present:
        return None
    first = present[0]
    compatibility_keys = (
        "model_name",
        "target_module",
        "denoiser_type",
        "denoiser_config_path",
        "recorded_score_calls",
        "samples_per_sequence",
    )
    for metadata in present[1:]:
        for key in compatibility_keys:
            if metadata.get(key) != first.get(key):
                raise ValueError(f"source stores have incompatible metadata field {key!r}")

    sequences: dict[str, str] = {}
    sequence_count = 0
    for metadata in present:
        metadata_sequences = metadata.get("sequences")
        if metadata_sequences:
            for sequence_id, sequence in metadata_sequences.items():
                existing = sequences.get(sequence_id)
                if existing is not None and existing != sequence:
                    raise ValueError(f"source stores disagree on sequence {sequence_id!r}")
                sequences[sequence_id] = sequence
        elif metadata.get("sequence_count") is not None:
            sequence_count += int(metadata["sequence_count"])

    if sequences:
        sequence_count = len(sequences)
    structure_index = (
        str(Path("final_structures") / "index.jsonl")
        if any(metadata.get("structure_index") for metadata in present)
        else None
    )
    return StoreMetadata(
        source_name="merged_activation_stores",
        model_name=first["model_name"],
        target_module=first["target_module"],
        sequence_count=sequence_count or None,
        sequences=sequences or None,
        denoiser_type=first.get("denoiser_type"),
        denoiser_config_path=first.get("denoiser_config_path"),
        recorded_score_calls=first.get("recorded_score_calls"),
        samples_per_sequence=first.get("samples_per_sequence"),
        structure_index=structure_index,
    )


def _merge_structure_indices(
    source_stores: list[ActivationStore], merged_store: ActivationStore
) -> None:
    merged_metadata = merged_store.manifest.get("metadata") or {}
    merged_index = merged_metadata.get("structure_index")
    if merged_index is None:
        return
    merged_index_path = merged_store.root / merged_index
    merged_index_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_index_path.open("w") as out_handle:
        for source_store in source_stores:
            metadata = source_store.manifest.get("metadata") or {}
            source_index = metadata.get("structure_index")
            if source_index is None:
                continue
            source_index_path = source_store.root / source_index
            if not source_index_path.exists():
                raise ValueError(f"missing structure index {source_index_path}")
            source_prefix = Path("final_structures") / source_store.root.name
            with source_index_path.open() as in_handle:
                for raw_line in in_handle:
                    if not raw_line.strip():
                        continue
                    entry = json.loads(raw_line)
                    for key in ("npz_path", "pdb_path"):
                        source_relative = Path(entry[key])
                        target_relative = source_prefix / source_relative.name
                        target_path = merged_store.root / target_relative
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_store.root / source_relative, target_path)
                        entry[key] = str(target_relative)
                    out_handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _shard_offsets(manifest: dict) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    start = 0
    for shard in manifest["shards"]:
        end = start + int(shard["num_tokens"])
        offsets.append((start, end))
        start = end
    return offsets


def _locate_global_index(global_idx: int, shard_offsets: list[tuple[int, int]]) -> tuple[int, int]:
    shard_ends = [end for _, end in shard_offsets]
    shard_idx = bisect_right(shard_ends, global_idx)
    if shard_idx < len(shard_offsets):
        start, end = shard_offsets[shard_idx]
        if start <= global_idx < end:
            return shard_idx, global_idx - start
    raise IndexError(f"global activation index {global_idx} is out of range")
