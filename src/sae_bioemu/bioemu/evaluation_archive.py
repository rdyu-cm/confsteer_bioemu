from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import uuid
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from sae_bioemu.bioemu import extractor
from sae_bioemu.bioemu.extractor import BioEmuMultiLayerRuntimeSampler, read_fasta
from sae_bioemu.core.config import ProjectConfig, TraceConfig
from sae_bioemu.core.store import ActivationStore, StoreMetadata
from sae_bioemu.core.types import ActivationRecord

EVALUATION_LAYER_TARGETS = {
    "layer_m1": "model_nn.st_module.encoder.layers.-1",
    "layer_m3": "model_nn.st_module.encoder.layers.-3",
    "layer_m5": "model_nn.st_module.encoder.layers.-5",
    "layer_m7": "model_nn.st_module.encoder.layers.-7",
}


@dataclass(frozen=True)
class EvaluationArchiveConfig:
    root: Path
    input_fasta: Path
    cache_embeds_dir: Path
    structures_dir: Path
    trace: TraceConfig
    layer_targets: dict[str, str]
    samples_per_sequence: int
    recorded_score_calls: int
    sample_start: int = 0
    activation_dim: int = 512
    activation_dtype: str = "float32"
    max_tokens_per_shard: int = 1_000_000
    world_size: int = 4
    extraction_dependencies: tuple[Path, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sample_start, int)
            or isinstance(self.sample_start, bool)
            or self.sample_start < 0
        ):
            raise ValueError("sample_start must be a non-negative integer")

    @property
    def sample_indices(self) -> tuple[int, ...]:
        return tuple(range(self.sample_start, self.sample_start + self.samples_per_sequence))

    @classmethod
    def from_project_configs(
        cls,
        configs: Mapping[str, ProjectConfig],
        world_size: int = 4,
    ) -> EvaluationArchiveConfig:
        if not configs:
            raise ValueError("evaluation archive requires ablation configs")
        values = list(configs.values())
        base = values[0]
        targets = {config.trace.target_module for config in values}
        if targets != set(EVALUATION_LAYER_TARGETS.values()):
            raise ValueError("evaluation configs do not cover the four archive layers")
        roots = {config.analysis.evaluation_activation_archive for config in values}
        if len(roots) != 1 or None in roots:
            raise ValueError("evaluation configs must share one activation archive")
        shared = {
            (
                config.analysis.evaluation_input_fasta,
                config.analysis.evaluation_cache_embeds_dir,
                config.analysis.evaluation_structure_dir,
                config.analysis.evaluation_samples_per_sequence,
                config.trace.recorded_score_calls,
                config.trace.model_name,
                config.trace.denoiser_type,
                config.trace.denoiser_config_path,
                config.trace.ckpt_path,
                config.trace.model_config_path,
            )
            for config in values
        }
        if len(shared) != 1:
            raise ValueError("evaluation extraction settings do not agree")
        root = roots.pop()
        if root is None:
            raise ValueError("evaluation activation archive is required")
        required_paths = {
            "evaluation_input_fasta": base.analysis.evaluation_input_fasta,
            "evaluation_cache_embeds_dir": base.analysis.evaluation_cache_embeds_dir,
            "evaluation_structure_dir": base.analysis.evaluation_structure_dir,
        }
        missing = [name for name, value in required_paths.items() if value is None]
        if missing:
            raise ValueError("missing evaluation archive paths: " + ", ".join(missing))
        if base.trace.recorded_score_calls != 10:
            raise ValueError("evaluation archive requires ten recorded score calls")
        return cls(
            root=Path(root),
            input_fasta=Path(str(base.analysis.evaluation_input_fasta)),
            cache_embeds_dir=Path(str(base.analysis.evaluation_cache_embeds_dir)),
            structures_dir=Path(str(base.analysis.evaluation_structure_dir)),
            trace=base.trace,
            layer_targets=dict(EVALUATION_LAYER_TARGETS),
            samples_per_sequence=base.analysis.evaluation_samples_per_sequence,
            recorded_score_calls=10,
            max_tokens_per_shard=base.trace.max_tokens_per_shard,
            world_size=world_size,
            extraction_dependencies=(Path(extractor.__file__),),
        )

    @classmethod
    def for_test(
        cls,
        *,
        root: Path,
        input_fasta: Path,
        extraction_dependencies: tuple[Path, ...],
    ) -> EvaluationArchiveConfig:
        return cls(
            root=root,
            input_fasta=input_fasta,
            cache_embeds_dir=root / "embeddings",
            structures_dir=root / "structures",
            trace=TraceConfig(recorded_score_calls=10, samples_per_sequence=2),
            layer_targets=dict(EVALUATION_LAYER_TARGETS),
            samples_per_sequence=2,
            recorded_score_calls=10,
            world_size=2,
            extraction_dependencies=extraction_dependencies,
        )


@dataclass(frozen=True)
class EvaluationRankAudit:
    rank: int
    completed_sequence_indices: tuple[int, ...]
    layer_token_counts: dict[str, int]
    store_manifest_sha256: dict[str, str]


@dataclass(frozen=True)
class EvaluationExtractionResult:
    rank: int
    sequence_indices: tuple[int, ...]
    preexisting_sequence_indices: tuple[int, ...]


@dataclass(frozen=True)
class ArchivedEvaluationSequence:
    sequence_index: int
    sequence_id: str
    sequence: str
    layers: dict[str, ActivationRecord]
    sample_indices: tuple[int, ...]


def owned_sequence_indices(count: int, rank: int, world_size: int) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("sequence count must be positive")
    if world_size < 1:
        raise ValueError("world size must be positive")
    if rank not in range(world_size):
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    return tuple(range(rank, count, world_size))


def prepare_evaluation_embedding_manifest(config: EvaluationArchiveConfig) -> Path:
    cache = config.cache_embeds_dir
    if not cache.is_dir():
        raise FileNotFoundError(f"evaluation embedding cache does not exist: {cache}")
    manifest_path = cache / "embedding_manifest.json"
    files = {
        path.relative_to(cache).as_posix(): _sha256_file(path)
        for path in sorted(cache.rglob("*"))
        if path.is_file()
        and path != manifest_path
        and not (path.name.startswith(".embedding_manifest.json.") and path.suffix == ".tmp")
    }
    if not files:
        raise ValueError("evaluation embedding cache contains no files")
    payload = {
        "schema_version": 1,
        "input_fasta_sha256": _sha256_file(config.input_fasta),
        "files": files,
    }
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError("evaluation embedding cache manifest is invalid") from error
        if existing != payload:
            raise ValueError("evaluation embedding cache no longer matches its manifest")
        return manifest_path
    _atomic_write_text(manifest_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest_path


def extract_evaluation_archive_rank(
    config: EvaluationArchiveConfig,
    sampler_factory: Callable = BioEmuMultiLayerRuntimeSampler,
) -> EvaluationExtractionResult:
    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if world_size != config.world_size or rank not in range(world_size):
        raise ValueError("RANK/WORLD_SIZE must match the evaluation archive config")
    records = read_fasta(config.input_fasta)
    audit = audit_evaluation_archive_rank(config, rank, allow_empty=True)
    completed = set(audit.completed_sequence_indices)
    owned = owned_sequence_indices(len(records), rank, world_size)
    pending = tuple(index for index in owned if index not in completed)
    if not pending:
        return EvaluationExtractionResult(rank, (), tuple(sorted(completed)))
    device = None
    if sampler_factory is BioEmuMultiLayerRuntimeSampler:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank not in range(world_size) or not torch.cuda.is_available():
            raise RuntimeError("evaluation archive extraction requires rank-local CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    sampler = sampler_factory(
        config=config.trace,
        target_modules=tuple(config.layer_targets.values()),
        cache_embeds_dir=config.cache_embeds_dir,
        structures_dir=config.structures_dir,
        batch_size=1,
        device=device,
    )
    sampled: list[int] = []
    for sequence_index in pending:
        sequence = records[sequence_index]
        per_layer: dict[str, list[ActivationRecord]] = {name: [] for name in config.layer_targets}
        for sample_index in config.sample_indices:
            by_target = sampler.sample_layers([sequence], sample_index)
            if set(by_target) != set(config.layer_targets.values()):
                raise ValueError("sampler did not return all evaluation target layers")
            for name, target in config.layer_targets.items():
                per_layer[name].append(by_target[target])
        layers = {name: ActivationRecord.concat(parts) for name, parts in per_layer.items()}
        write_evaluation_sequence_transaction(config, rank, sequence_index, layers)
        sampled.append(sequence_index)
    return EvaluationExtractionResult(rank, tuple(sampled), tuple(sorted(completed)))


def iter_evaluation_archive_sequences(
    manifest_path: str | Path,
    sequence_indices: Collection[int] | None = None,
) -> Iterator[ArchivedEvaluationSequence]:
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError("evaluation archive manifest is invalid") from error
    _validate_replay_manifest(path, manifest)
    count = int(manifest["sequence_count"])
    selected = tuple(range(count)) if sequence_indices is None else tuple(sorted(sequence_indices))
    if any(index not in range(count) for index in selected):
        raise ValueError("requested evaluation sequence index is out of range")
    root = path.parent
    for sequence_index in selected:
        rank = int(manifest["sequence_owners"][str(sequence_index)])
        shard_index = int(manifest["sequence_shards"][str(rank)][str(sequence_index)])
        layers: dict[str, ActivationRecord] = {}
        for name in manifest["layers"]:
            store = ActivationStore(root / manifest["rank_stores"][str(rank)][name], read_only=True)
            shard = store.manifest["shards"][shard_index]
            layers[name] = store.read_shard(shard["file"])
        sequence_id = str(manifest["sequence_ids"][sequence_index])
        sequence = str(manifest["sequences"][sequence_index])
        _validate_replayed_layers(manifest, layers, sequence_id, len(sequence))
        sample_indices = layers["layer_m1"].sample_indices
        if sample_indices is None:
            raise ValueError("archived evaluation sequence lacks sample indices")
        yield ArchivedEvaluationSequence(
            sequence_index=sequence_index,
            sequence_id=sequence_id,
            sequence=sequence,
            layers=layers,
            sample_indices=tuple(int(value) for value in sample_indices.unique(sorted=True)),
        )


def write_evaluation_sequence_transaction(
    config: EvaluationArchiveConfig,
    rank: int,
    sequence_index: int,
    layers: Mapping[str, ActivationRecord],
) -> None:
    records = read_fasta(config.input_fasta)
    owned = owned_sequence_indices(len(records), rank, config.world_size)
    if sequence_index not in owned:
        raise ValueError(f"rank {rank} does not own sequence index {sequence_index}")
    stores = _rank_stores(config, rank, records)
    completed_before = _read_committed_sequence_boundary(config, rank, stores, owned)
    if sequence_index in completed_before:
        raise ValueError(f"sequence index {sequence_index} is already committed")
    expected_prefix = owned[: len(completed_before) + 1]
    if expected_prefix[-1] != sequence_index:
        raise ValueError("sequence transactions must commit in owned FASTA order")
    sequence = records[sequence_index]
    _validate_layer_transaction(config, layers, sequence.sequence_id, len(sequence.sequence))
    for name in sorted(stores):
        stores[name].append(layers[name])
    completed = (*completed_before, sequence_index)
    payload = {
        "schema_version": 1,
        "rank": rank,
        "world_size": config.world_size,
        "archive_fingerprint": evaluation_archive_fingerprint(config),
        "completed_sequence_indices": list(completed),
        "store_manifest_sha256": {
            name: _sha256_file(store.manifest_path) for name, store in stores.items()
        },
    }
    _atomic_write_text(_rank_state_path(config, rank), json.dumps(payload, indent=2) + "\n")


def audit_evaluation_archive_rank(
    config: EvaluationArchiveConfig,
    rank: int,
    *,
    allow_empty: bool = False,
) -> EvaluationRankAudit:
    if rank not in range(config.world_size):
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    roots = {
        name: config.root / "activations" / f"rank_{rank}" / name for name in config.layer_targets
    }
    existing = {name for name, root in roots.items() if (root / "manifest.json").is_file()}
    state_path = _rank_state_path(config, rank)
    if not existing and not state_path.exists() and allow_empty:
        return EvaluationRankAudit(
            rank,
            (),
            {name: 0 for name in config.layer_targets},
            {},
        )
    if existing != set(config.layer_targets):
        raise ValueError("partial cross-layer transaction: layer store set is incomplete")
    stores = {name: ActivationStore(root) for name, root in roots.items()}
    shard_counts = {name: len(store.manifest["shards"]) for name, store in stores.items()}
    if len(set(shard_counts.values())) != 1:
        raise ValueError(f"partial cross-layer transaction: shard counts {shard_counts}")
    if not state_path.is_file():
        if allow_empty and not any(shard_counts.values()):
            return EvaluationRankAudit(
                rank,
                (),
                {name: len(store) for name, store in stores.items()},
                {name: _sha256_file(store.manifest_path) for name, store in stores.items()},
            )
        raise ValueError("partial cross-layer transaction: committed state is missing")
    try:
        state = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError("invalid evaluation rank state") from error
    manifest_hashes = {name: _sha256_file(store.manifest_path) for name, store in stores.items()}
    expected_state = {
        "rank": rank,
        "world_size": config.world_size,
        "archive_fingerprint": evaluation_archive_fingerprint(config),
        "store_manifest_sha256": manifest_hashes,
    }
    for key, expected in expected_state.items():
        if state.get(key) != expected:
            raise ValueError(f"committed evaluation rank state mismatch for {key}")
    completed = tuple(int(value) for value in state.get("completed_sequence_indices", ()))
    records = read_fasta(config.input_fasta)
    owned = owned_sequence_indices(len(records), rank, config.world_size)
    if completed != owned[: len(completed)]:
        raise ValueError("committed sequence indices are not an owned FASTA-order prefix")
    count = next(iter(shard_counts.values()))
    if len(completed) != count:
        raise ValueError("partial cross-layer transaction: state and shard counts differ")
    for shard_index, sequence_index in enumerate(completed):
        aligned = {
            name: store.read_shard(store.manifest["shards"][shard_index]["file"])
            for name, store in stores.items()
        }
        sequence = records[sequence_index]
        _validate_layer_transaction(
            config,
            aligned,
            sequence.sequence_id,
            len(sequence.sequence),
        )
    return EvaluationRankAudit(
        rank=rank,
        completed_sequence_indices=completed,
        layer_token_counts={name: len(store) for name, store in stores.items()},
        store_manifest_sha256=manifest_hashes,
    )


def finalize_evaluation_archive(config: EvaluationArchiveConfig) -> Path:
    records = read_fasta(config.input_fasta)
    audits = [audit_evaluation_archive_rank(config, rank) for rank in range(config.world_size)]
    owners: dict[int, int] = {}
    for audit in audits:
        for sequence_index in audit.completed_sequence_indices:
            if sequence_index in owners:
                raise ValueError("duplicate evaluation sequence ownership")
            owners[sequence_index] = audit.rank
    if sorted(owners) != list(range(len(records))):
        raise ValueError("archive does not have exact evaluation sequence coverage")
    payload = {
        "schema_version": 1,
        "archive_fingerprint": evaluation_archive_fingerprint(config),
        "sequence_count": len(records),
        "sequence_ids": [record.sequence_id for record in records],
        "sequences": [record.sequence for record in records],
        "sequence_lengths": [len(record.sequence) for record in records],
        "sample_start": config.sample_start,
        "sample_indices": list(config.sample_indices),
        "samples_per_sequence": config.samples_per_sequence,
        "recorded_score_calls": config.recorded_score_calls,
        "activation_dim": config.activation_dim,
        "activation_dtype": config.activation_dtype,
        "layers": config.layer_targets,
        "world_size": config.world_size,
        "sequence_owners": {str(index): rank for index, rank in sorted(owners.items())},
        "rank_stores": {
            str(rank): {
                name: str(Path("activations") / f"rank_{rank}" / name)
                for name in config.layer_targets
            }
            for rank in range(config.world_size)
        },
        "sequence_shards": {
            str(audit.rank): {
                str(sequence_index): shard_index
                for shard_index, sequence_index in enumerate(audit.completed_sequence_indices)
            }
            for audit in audits
        },
        "store_manifest_sha256": {str(audit.rank): audit.store_manifest_sha256 for audit in audits},
        "structure_index": str(
            config.structures_dir / config.trace.final_structures_subdir / "index.jsonl"
        ),
    }
    manifest_path = config.root / "dataset_manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError("existing evaluation archive manifest is invalid") from error
        if existing != payload:
            raise ValueError("existing evaluation archive manifest conflicts with finalization")
        return manifest_path
    _atomic_write_text(manifest_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest_path


def _rank_stores(
    config: EvaluationArchiveConfig,
    rank: int,
    records: list,
) -> dict[str, ActivationStore]:
    return {
        name: ActivationStore(
            config.root / "activations" / f"rank_{rank}" / name,
            metadata=StoreMetadata(
                source_name="bioemu_interpro_eval_1000",
                model_name=config.trace.model_name,
                target_module=target,
                sequence_count=len(records),
                denoiser_type=config.trace.denoiser_type,
                denoiser_config_path=config.trace.denoiser_config_path,
                recorded_score_calls=config.recorded_score_calls,
                samples_per_sequence=config.samples_per_sequence,
                structure_index=str(
                    config.structures_dir / config.trace.final_structures_subdir / "index.jsonl"
                ),
            ),
        )
        for name, target in config.layer_targets.items()
    }


def _rank_state_path(config: EvaluationArchiveConfig, rank: int) -> Path:
    return config.root / "state" / f"rank_{rank}.json"


def _read_committed_sequence_boundary(
    config: EvaluationArchiveConfig,
    rank: int,
    stores: Mapping[str, ActivationStore],
    owned: tuple[int, ...],
) -> tuple[int, ...]:
    shard_counts = {name: len(store.manifest["shards"]) for name, store in stores.items()}
    if len(set(shard_counts.values())) != 1:
        raise ValueError(f"partial cross-layer transaction: shard counts {shard_counts}")
    count = next(iter(shard_counts.values()))
    state_path = _rank_state_path(config, rank)
    if not state_path.is_file():
        if count == 0:
            return ()
        raise ValueError("partial cross-layer transaction: committed state is missing")
    try:
        state = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError("invalid evaluation rank state") from error
    hashes = {name: _sha256_file(store.manifest_path) for name, store in stores.items()}
    if (
        state.get("rank") != rank
        or state.get("world_size") != config.world_size
        or state.get("archive_fingerprint") != evaluation_archive_fingerprint(config)
        or state.get("store_manifest_sha256") != hashes
    ):
        raise ValueError("committed evaluation rank state does not match layer stores")
    completed = tuple(int(value) for value in state.get("completed_sequence_indices", ()))
    if completed != owned[: len(completed)] or len(completed) != count:
        raise ValueError("partial cross-layer transaction: invalid committed boundary")
    return completed


def _validate_layer_transaction(
    config: EvaluationArchiveConfig,
    layers: Mapping[str, ActivationRecord],
    sequence_id: str,
    sequence_length: int,
) -> None:
    if set(layers) != set(config.layer_targets):
        raise ValueError("evaluation transaction does not contain all four layers")
    reference = layers["layer_m1"]
    for name, record in layers.items():
        if record.activation_dim != config.activation_dim:
            raise ValueError(f"{name} activation width does not match archive")
        if str(record.activations.dtype) != f"torch.{config.activation_dtype}":
            raise ValueError(f"{name} activation dtype does not match archive")
        if record.sample_indices is None:
            raise ValueError(f"{name} lacks sample indices")
        if set(record.sequence_ids) != {sequence_id}:
            raise ValueError(f"{name} sequence provenance does not match FASTA")
        if name != "layer_m1" and not _same_provenance(reference, record):
            raise ValueError(f"{name} is not aligned with layer_m1")
        expected_tokens = (
            config.samples_per_sequence * config.recorded_score_calls * sequence_length
        )
        if record.n_tokens != expected_tokens:
            raise ValueError(f"{name} does not have exact evaluation token coverage")
        sample_values = record.sample_indices.unique(sorted=True).tolist()
        if sample_values != list(config.sample_indices):
            raise ValueError(f"{name} does not have exact evaluation sample coverage")
        for sample_index in sample_values:
            selected = record.sample_indices == int(sample_index)
            residues = record.residue_indices[selected]
            expected_residues = torch.arange(sequence_length).repeat(config.recorded_score_calls)
            if not torch.equal(residues.cpu(), expected_residues):
                raise ValueError(f"{name} residue/call alignment is invalid")
            times = record.timesteps[selected].reshape(
                config.recorded_score_calls,
                sequence_length,
            )
            if not torch.all(times == times[:, :1]):
                raise ValueError(f"{name} timestep rows are not residue-aligned")
            if len(torch.unique(times[:, 0])) != config.recorded_score_calls:
                raise ValueError(f"{name} does not preserve all score calls")


def _validate_replay_manifest(path: Path, manifest: object) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("evaluation archive manifest schema is invalid")
    required = {
        "sequence_count",
        "sequence_ids",
        "sequences",
        "sequence_lengths",
        "samples_per_sequence",
        "recorded_score_calls",
        "activation_dim",
        "activation_dtype",
        "layers",
        "sequence_owners",
        "rank_stores",
        "sequence_shards",
        "store_manifest_sha256",
    }
    if not required.issubset(manifest):
        raise ValueError("evaluation archive manifest fields are incomplete")
    _manifest_sample_indices(manifest)
    root = path.parent
    for rank, layer_hashes in manifest["store_manifest_sha256"].items():
        for name, expected_hash in layer_hashes.items():
            store_manifest = root / manifest["rank_stores"][rank][name] / "manifest.json"
            if not store_manifest.is_file() or _sha256_file(store_manifest) != expected_hash:
                raise ValueError(f"evaluation archive store manifest hash mismatch: {rank}/{name}")


def _validate_replayed_layers(
    manifest: dict,
    layers: Mapping[str, ActivationRecord],
    sequence_id: str,
    sequence_length: int,
) -> None:
    if set(layers) != set(manifest["layers"]):
        raise ValueError("archived evaluation sequence has incomplete layers")
    reference = layers["layer_m1"]
    sample_indices = _manifest_sample_indices(manifest)
    samples = len(sample_indices)
    calls = int(manifest["recorded_score_calls"])
    expected_tokens = samples * calls * sequence_length
    for name, record in layers.items():
        if record.activation_dim != int(manifest["activation_dim"]):
            raise ValueError(f"{name} archived activation width is invalid")
        if str(record.activations.dtype) != f"torch.{manifest['activation_dtype']}":
            raise ValueError(f"{name} archived activation dtype is invalid")
        if record.n_tokens != expected_tokens or record.sample_indices is None:
            raise ValueError(f"{name} archived token coverage is invalid")
        if set(record.sequence_ids) != {sequence_id}:
            raise ValueError(f"{name} archived sequence provenance is invalid")
        if name != "layer_m1" and not _same_provenance(reference, record):
            raise ValueError(f"{name} archived provenance is not layer-aligned")
        sample_values = record.sample_indices.unique(sorted=True).tolist()
        if sample_values != list(sample_indices):
            raise ValueError(f"{name} archived sample coverage is invalid")
        expected_sample_rows = torch.repeat_interleave(
            torch.tensor(sample_indices, dtype=record.sample_indices.dtype),
            calls * sequence_length,
        )
        if not torch.equal(record.sample_indices.cpu(), expected_sample_rows):
            raise ValueError(f"{name} archived sample ordering is invalid")
        for sample_index in sample_values:
            selected = record.sample_indices == int(sample_index)
            residues = record.residue_indices[selected]
            if not torch.equal(residues.cpu(), torch.arange(sequence_length).repeat(calls)):
                raise ValueError(f"{name} archived residue/call alignment is invalid")
            times = record.timesteps[selected].reshape(calls, sequence_length)
            if not torch.all(times == times[:, :1]) or len(torch.unique(times[:, 0])) != calls:
                raise ValueError(f"{name} archived timestep/call coverage is invalid")


def _same_provenance(left: ActivationRecord, right: ActivationRecord) -> bool:
    return (
        left.sequence_ids == right.sequence_ids
        and torch.equal(left.timesteps, right.timesteps)
        and torch.equal(left.residue_indices, right.residue_indices)
        and torch.equal(left.sample_indices, right.sample_indices)
    )


def _manifest_sample_indices(manifest: Mapping[str, object]) -> tuple[int, ...]:
    samples_per_sequence = manifest.get("samples_per_sequence")
    if not isinstance(samples_per_sequence, int) or isinstance(samples_per_sequence, bool):
        raise ValueError("evaluation archive manifest sample range is invalid")
    has_sample_start = "sample_start" in manifest
    has_sample_indices = "sample_indices" in manifest
    if has_sample_start != has_sample_indices:
        raise ValueError("evaluation archive manifest sample range is inconsistent")
    if not has_sample_start:
        return tuple(range(samples_per_sequence))
    sample_start = manifest["sample_start"]
    declared = manifest["sample_indices"]
    if (
        not isinstance(sample_start, int)
        or isinstance(sample_start, bool)
        or not isinstance(declared, list)
        or any(not isinstance(value, int) or isinstance(value, bool) for value in declared)
    ):
        raise ValueError("evaluation archive manifest sample range is invalid")
    expected = tuple(range(sample_start, sample_start + samples_per_sequence))
    if declared != list(expected):
        raise ValueError("evaluation archive manifest sample range is inconsistent")
    return expected


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluation_archive_fingerprint(config: EvaluationArchiveConfig) -> str:
    dependencies = {
        str(path.resolve()): _sha256_file(path) for path in config.extraction_dependencies
    }
    payload = {
        "schema_version": 1,
        "scientific_config": {
            "trace": asdict(config.trace),
            "layer_targets": config.layer_targets,
            "sample_start": config.sample_start,
            "sample_indices": config.sample_indices,
            "samples_per_sequence": config.samples_per_sequence,
            "recorded_score_calls": config.recorded_score_calls,
            "activation_dim": config.activation_dim,
            "activation_dtype": config.activation_dtype,
        },
        "fasta_sha256": _sha256_file(config.input_fasta),
        "embedding_manifest_sha256": _optional_sha256_file(
            config.cache_embeds_dir / "embedding_manifest.json"
        ),
        "extraction_dependencies": dependencies,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "bioemu": _package_version("bioemu"),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256_file(path: Path) -> str | None:
    return _sha256_file(path) if path.is_file() else None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
