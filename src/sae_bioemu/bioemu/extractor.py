from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml

from sae_bioemu.bioemu.precision import (
    GenerationPrecision,
    generation_precision_scope,
    measurement_coordinates,
    resolve_generation_precision,
)
from sae_bioemu.core.config import TraceConfig, seed_everything
from sae_bioemu.core.store import ActivationStore, ActivationStoreWriter, StoreMetadata
from sae_bioemu.core.types import ActivationRecord

if TYPE_CHECKING:
    from sae_bioemu.bioemu.steering import SteeringSite
    from sae_bioemu.steering.adapter import BoundedResidualAdapter


@dataclass(frozen=True)
class SequenceRecord:
    """One named protein sequence resolved from FASTA, CSV, or CLI input."""

    sequence_id: str
    sequence: str


@dataclass(frozen=True)
class DenoiserCapture:
    """Activations captured during one denoising trajectory plus final sample."""

    n_records: int
    sampled_batch: object


@contextmanager
def _so3_cache_lock(
    cache_dir: str | Path | None,
    model_config_path: str | Path,
):
    root = _effective_so3_cache_dir(cache_dir, model_config_path)
    if root is None:
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".sae_bioemu_so3.lock").open("a") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        yield


def _effective_so3_cache_dir(
    cache_dir: str | Path | None,
    model_config_path: str | Path,
) -> Path | None:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    payload = yaml.safe_load(Path(model_config_path).read_text())
    if not isinstance(payload, Mapping):
        return None
    sdes = payload.get("sdes")
    orientations = sdes.get("node_orientations") if isinstance(sdes, Mapping) else None
    configured = orientations.get("cache_dir") if isinstance(orientations, Mapping) else None
    return Path(configured).expanduser() if isinstance(configured, str) and configured else None


@dataclass(frozen=True)
class FinalStructureEntry:
    """Metadata linking one sampled trajectory to its saved final structure."""

    sequence_id: str
    sample_index: int
    sequence: str
    npz_path: str
    pdb_path: str
    pos_units: str = "nanometers"
    pdb_units: str = "angstrom_centered"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class FinalStructureWriter:
    """Writes final denoised structures and an index keyed by sequence/sample."""

    def __init__(self, store_root: str | Path, subdir: str = "final_structures") -> None:
        self.store_root = Path(store_root)
        self.subdir = Path(subdir)
        self.root = self.store_root / self.subdir
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self.index_lock_path = self.root / "index.jsonl.lock"
        self._indexed_keys = self._load_indexed_keys()

    @property
    def index_relative_path(self) -> str:
        return str(self.subdir / "index.jsonl")

    def append_batch(
        self,
        sampled_batch: object,
        records: list[SequenceRecord],
        sample_index: int | Sequence[int],
        skip_existing: bool = False,
    ) -> list[FinalStructureEntry]:
        """Publish one batch. ``sample_index`` may be PER MEMBER, and for one caller it must be.

        The entry stem is ``(sequence_id, sample_index)``. A batch of DIFFERENT sequences at one
        index is fine -- the ids separate them -- and that is the only shape ``sample()`` produces,
        which is why a scalar was enough for years.

        The unsteered path is the other shape: ``cryptic_generation.unsteered_draws`` batches ONE
        record repeated across seeds (``members = [record] * len(chunk)``), so every member of a
        batch shares a sequence_id AND, under a scalar, an index. All but the first then collide on
        one stem and are silently skipped. Measured: stage 1 wrote 3904 structures where 976 x 32
        were expected -- exactly 976 x 4, one per BATCH at batch_size 8, not one per DRAW.
        """
        # ARGUMENTS FIRST, before the optional-dependency import below. A caller passing a
        # colliding index should get that error, not a confusing message about missing bioemu
        # extras -- and this way the guard is exercisable without the GPU stack, which the project's guidelines
        # requires of the test suite.
        if isinstance(sample_index, int):
            indices: list[int] = [int(sample_index)] * len(records)
        else:
            indices = [int(value) for value in sample_index]
            if len(indices) != len(records):
                raise ValueError(
                    f"sample_index has {len(indices)} entries for {len(records)} records; a "
                    f"per-member index must name every member or the extras collide on one stem"
                )
        stems = [
            _structure_entry_stem(record.sequence_id, index)
            for record, index in zip(records, indices, strict=True)
        ]
        if len(set(stems)) != len(stems):
            # Fails closed rather than silently writing one structure for several draws, which is
            # exactly how 7 of every 8 stage-1 draws were lost.
            raise ValueError(
                f"{len(stems) - len(set(stems))} of {len(stems)} batch members share an entry "
                f"stem, so they would overwrite one another. Pass a per-member sample_index when "
                f"a batch repeats one sequence across seeds"
            )

        try:
            from bioemu.convert_chemgraph import _write_pdb
        except ImportError as exc:
            raise RuntimeError(
                "Saving final structures requires the optional bioemu dependencies"
            ) from exc

        if isinstance(sampled_batch, tuple):
            sampled_batch = sampled_batch[0]
        if not hasattr(sampled_batch, "to_data_list"):
            raise TypeError("denoiser output must be a torch_geometric Batch")
        sampled_chemgraphs = sampled_batch.to_data_list()
        if len(sampled_chemgraphs) != len(records):
            raise ValueError(
                f"denoiser returned {len(sampled_chemgraphs)} structures for "
                f"{len(records)} input records"
            )

        entries: list[FinalStructureEntry] = []
        with self.index_path.open("a") as index_handle:
            for record, chemgraph, sample_index in zip(
                records, sampled_chemgraphs, indices, strict=True
            ):
                pos = chemgraph.pos.detach().cpu()
                node_orientations = chemgraph.node_orientations.detach().cpu()
                entry_stem = _structure_entry_stem(record.sequence_id, sample_index)
                npz_path = self.root / f"{entry_stem}.npz"
                pdb_path = self.root / f"{entry_stem}.pdb"
                lock_path = self.root / f"{entry_stem}.lock"
                entry = FinalStructureEntry(
                    sequence_id=record.sequence_id,
                    sample_index=sample_index,
                    sequence=record.sequence,
                    npz_path=str(npz_path.relative_to(self.store_root)),
                    pdb_path=str(pdb_path.relative_to(self.store_root)),
                )
                with lock_path.open("a") as lock_handle:
                    fcntl.flock(lock_handle, fcntl.LOCK_EX)
                    if npz_path.exists() or pdb_path.exists():
                        if skip_existing and npz_path.exists() and pdb_path.exists():
                            self._write_index_entry_once(index_handle, entry)
                            entries.append(entry)
                            continue
                        if not skip_existing:
                            raise FileExistsError(
                                f"final structure already exists for "
                                f"{record.sequence_id} sample {sample_index}"
                            )

                    tmp_npz_path = npz_path.with_name(f".{npz_path.name}.{os.getpid()}.tmp")
                    tmp_pdb_path = pdb_path.with_name(f".{pdb_path.name}.{os.getpid()}.tmp")
                    try:
                        with tmp_npz_path.open("wb") as npz_handle:
                            np.savez(
                                npz_handle,
                                pos=pos.numpy(),
                                pos_units="nanometers",
                                node_orientations=node_orientations.numpy(),
                                sequence=record.sequence,
                                sequence_id=record.sequence_id,
                                sample_index=sample_index,
                            )
                        pos_angstrom = pos * 10.0
                        pos_angstrom = pos_angstrom - pos_angstrom.mean(dim=0, keepdim=True)
                        _write_pdb(
                            pos=pos_angstrom,
                            node_orientations=node_orientations,
                            sequence=record.sequence,
                            filename=tmp_pdb_path,
                        )
                        tmp_npz_path.replace(npz_path)
                        tmp_pdb_path.replace(pdb_path)
                    finally:
                        tmp_npz_path.unlink(missing_ok=True)
                        tmp_pdb_path.unlink(missing_ok=True)
                    self._write_index_entry_once(index_handle, entry)
                    entries.append(entry)
        return entries

    def _write_index_entry_once(self, index_handle, entry: FinalStructureEntry) -> None:
        key = (entry.sequence_id, entry.sample_index)
        with self.index_lock_path.open("a") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            self._indexed_keys = self._load_indexed_keys()
            if key in self._indexed_keys:
                return
            index_handle.write(entry.to_json() + "\n")
            index_handle.flush()
            os.fsync(index_handle.fileno())
            self._indexed_keys.add(key)

    def _load_indexed_keys(self) -> set[tuple[str, int]]:
        if not self.index_path.exists():
            return set()
        keys: set[tuple[str, int]] = set()
        with self.index_path.open() as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                entry = json.loads(raw_line)
                keys.add((str(entry["sequence_id"]), int(entry["sample_index"])))
        return keys


class ActivationRecordList:
    """In-memory sink with the same append interface as ActivationStoreWriter."""

    def __init__(self) -> None:
        self.records: list[ActivationRecord] = []

    def append(self, record: ActivationRecord) -> None:
        self.records.append(record)

    def concat(self) -> ActivationRecord:
        return ActivationRecord.concat(self.records)


@dataclass
class BioEmuActivationExtractor:
    """Coordinates BioEmu loading, denoising, and activation-store writing.

    Extraction builds BioEmu context chemgraphs from input sequences, runs the
    configured denoiser for one or more stochastic repetitions per sequence,
    and records intermediate score-model activations through
    ``RecordingScoreModel``. The final output is a sharded ``ActivationStore``
    suitable for SAE training.
    """

    config: TraceConfig = field(default_factory=TraceConfig)

    def capture_from_denoiser(
        self,
        score_model: torch.nn.Module,
        denoiser: object,
        sdes: dict,
        batch: object,
        store: ActivationStore | ActivationStoreWriter,
        sequence_ids: str | list[str] | tuple[str, ...] = "unknown",
        sample_indices: int | list[int] | tuple[int, ...] | torch.Tensor | None = None,
        device: str | torch.device | None = None,
        expected_score_calls: int | None = None,
        seed: int | None = None,
    ) -> DenoiserCapture:
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if seed is not None:
            seed_everything(seed)
        recorder = RecordingScoreModel(
            score_model=score_model,
            target_module=self.config.target_module,
            store=store,
            sequence_ids=sequence_ids,
            sample_indices=sample_indices,
            record_score_call_indices=record_score_call_indices(
                expected_score_calls,
                self.config.recorded_score_calls,
            ),
        ).to(device)
        try:
            if hasattr(batch, "to"):
                batch = batch.to(device)
            with torch.no_grad():
                sampled_batch = denoiser(
                    sdes=sdes,
                    device=device,
                    batch=batch,
                    score_model=recorder,
                )
        finally:
            recorder.remove()
        return DenoiserCapture(n_records=recorder.n_records, sampled_batch=sampled_batch)

    def extract_sequence_records(
        self,
        records: Iterable[SequenceRecord],
        output_store: str | Path,
        cache_embeds_dir: str | Path | None = None,
        batch_size: int = 1,
        source_name: str = "sequences",
        device: str | torch.device | None = None,
    ) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        records = list(records)
        try:
            from bioemu.model_utils import load_model, load_sdes, maybe_download_checkpoint
            from bioemu.sample import get_context_chemgraph
            from torch_geometric.data import Batch
        except ImportError as exc:
            raise RuntimeError(
                "BioEmu denoiser-loop extraction requires installing the optional bioemu dependencies"
            ) from exc

        ckpt_path, model_config_path = maybe_download_checkpoint(
            model_name=self.config.model_name,
            ckpt_path=self.config.ckpt_path,
            model_config_path=self.config.model_config_path,
        )
        score_model = load_model(ckpt_path, model_config_path)
        score_model.eval()
        with _so3_cache_lock(self.config.cache_so3_dir, model_config_path):
            sdes = load_sdes(
                model_config_path=model_config_path,
                cache_so3_dir=self.config.cache_so3_dir,
            )
        denoiser_config = load_bioemu_denoiser_config(self.config)
        expected_score_calls = expected_bioemu_score_calls(self.config, denoiser_config)
        denoiser = instantiate_bioemu_denoiser(denoiser_config)
        structure_index = (
            str(Path(self.config.final_structures_subdir) / "index.jsonl")
            if self.config.save_final_structures
            else None
        )
        store = ActivationStore(
            output_store,
            metadata=StoreMetadata(
                source_name=source_name,
                model_name=self.config.model_name,
                target_module=self.config.target_module,
                sequence_count=len(records),
                sequences={record.sequence_id: record.sequence for record in records},
                denoiser_type=self.config.denoiser_type,
                denoiser_config_path=self.config.denoiser_config_path,
                recorded_score_calls=self.config.recorded_score_calls,
                samples_per_sequence=self.config.samples_per_sequence,
                structure_index=structure_index,
            ),
        )
        structure_writer = (
            FinalStructureWriter(store.root, self.config.final_structures_subdir)
            if self.config.save_final_structures
            else None
        )
        writer = ActivationStoreWriter(
            store,
            max_tokens_per_shard=self.config.max_tokens_per_shard,
        )
        n_records = 0
        processed_sequences = 0
        next_sequence_log = 100
        try:
            for record_batch in batched(records, batch_size):
                chemgraphs = [
                    get_context_chemgraph(
                        sequence=record.sequence,
                        cache_embeds_dir=cache_embeds_dir,
                    )
                    for record in record_batch
                ]
                batch = Batch.from_data_list(chemgraphs)
                sequence_ids = [record.sequence_id for record in record_batch]
                for sample_idx in range(self.config.samples_per_sequence):
                    capture = self.capture_from_denoiser(
                        score_model=score_model,
                        denoiser=denoiser,
                        sdes=sdes,
                        batch=batch,
                        store=writer,
                        sequence_ids=sequence_ids,
                        sample_indices=[sample_idx] * len(sequence_ids),
                        device=device,
                        expected_score_calls=expected_score_calls,
                    )
                    n_records += capture.n_records
                    if structure_writer is not None:
                        structure_writer.append_batch(
                            sampled_batch=capture.sampled_batch,
                            records=record_batch,
                            sample_index=sample_idx,
                        )
                processed_sequences += len(record_batch)
                while processed_sequences >= next_sequence_log:
                    print(
                        f"[{_timestamp()}] extraction "
                        f"processed_sequences={processed_sequences}/{len(records)} "
                        f"activation_records={n_records}",
                        flush=True,
                    )
                    next_sequence_log += 100
        finally:
            writer.flush()
        return n_records


DEFAULT_CONTEXT_CHEMGRAPH_CACHE_SIZE = 4


class ContextChemGraphCache:
    """Bounded per-process LRU cache of prepared BioEmu context chemgraphs.

    ``bioemu.sample.get_context_chemgraph`` is called fresh on every generation
    and is memoized nowhere upstream: with cached embeddings it re-reads
    ``{sha}_single.npy`` and ``{sha}_pair.npy`` from shared storage each time,
    and the pair array alone is ``L**2 * 128 * 4`` bytes -- about 46 MB at the
    300-residue maximum used here. Matched adapter-off/on pairs, 32 seeds, and
    several target levels repeat those reads for one sequence thousands of times
    per run.

    Entries are shared, never copied, so a returned chemgraph must be treated as
    immutable. That is safe for every consumer in this repository because each
    one hands the chemgraph straight to ``Batch.from_data_list``, whose collate
    step concatenates every tensor attribute with ``torch.cat`` and therefore
    allocates fresh storage, and because the BioEmu denoisers only rebind fields
    through the shallow ``ChemGraph.replace``. Writing into a returned tensor in
    place would silently correlate every later sample of that sequence, which is
    far worse than the input/output the cache saves.

    A miss additionally verifies that loading consumed no torch RNG. Cached
    Evoformer embeddings make ``get_context_chemgraph`` a pure file read, so the
    cache cannot move any sampled structure; if a loader ever drew random
    numbers, memoizing it would change every later trajectory for that sequence
    and this fails closed instead.
    """

    def __init__(self, maxsize: int = DEFAULT_CONTEXT_CHEMGRAPH_CACHE_SIZE) -> None:
        if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize <= 0:
            raise ValueError("maxsize must be a positive integer")
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._entries: OrderedDict[str, object] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def sequences(self) -> tuple[str, ...]:
        """Cached sequences ordered least-recently-used first."""
        return tuple(self._entries)

    def get(self, sequence: str, load: Callable[[str], object]) -> object:
        """Return the prepared chemgraph for ``sequence``, loading it once."""
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("sequence must be a non-empty string")
        if not callable(load):
            raise TypeError("load must be callable")
        if sequence in self._entries:
            self._entries.move_to_end(sequence)
            self.hits += 1
            return self._entries[sequence]
        rng_state = torch.get_rng_state()
        chemgraph = load(sequence)
        if not torch.equal(torch.get_rng_state(), rng_state):
            raise RuntimeError(
                "preparing a BioEmu context chemgraph consumed torch RNG; caching it would "
                "change every later trajectory for this sequence"
            )
        self.misses += 1
        self._entries[sequence] = chemgraph
        self._entries.move_to_end(sequence)
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)
            self.evictions += 1
        return chemgraph


_MISSING = object()


@dataclass(frozen=True)
class _PriorCall:
    """One recorded ``prior_sampling`` call, replayed to rebuild RNG position."""

    field: str
    member_shapes: tuple[tuple[int, ...], ...]
    device: object


class SeededPriorSampling:
    """Draw each batch member's initial noised state from its own seeded stream.

    The initial state is created by the BioEmu denoisers, which replace the
    context batch fields with ``sdes[field].prior_sampling(shape, device=...)``
    (``bioemu/denoiser.py`` lines 303-308 for ``dpm_solver`` and lines 171-176
    for ``heun_denoiser``). Today one ``seed_everything(seed)`` precedes a
    batch-size-1 generation, so a single RNG draw covers the whole batch; if
    ``B`` structures were batched, per-structure seeds would stop existing and a
    batched run could no longer reproduce a cached batch-size-1 structure.

    While active, each field's ``prior_sampling`` is served per member: for
    member ``b`` the global torch RNG is reseeded with ``seeds[b]``, every
    earlier field of that member is replayed at its own per-member shape and
    device, the requested field is drawn, and the members are concatenated along
    the node dimension in seed order. Replaying instead of snapshotting keeps
    the draw bit-identical for CPU and CUDA streams alike and needs no
    assumption about which fields exist or what they are shaped like.

    With one seed this is exactly the unpatched sequence of calls, so both the
    drawn state and the RNG position handed to the rest of the trajectory are
    bit-identical to the pre-change path -- the compatibility constraint against
    already-cached structures. The caller must have seeded the global RNG with
    ``seeds[0]`` immediately beforehand; the first draw verifies that and fails
    closed if anything consumed torch RNG in between.

    BioEmu's ``prior_sampling`` implementations take no ``generator`` argument
    (``bioemu/sde_lib.py`` lines 137-142 and ``bioemu/so3_sde.py`` lines 1238
    and 1260, whose SO(3) angle draw also needs BioEmu's cached CDF tables), so
    reusing them exactly requires reseeding the global RNG they read rather than
    passing an explicit ``torch.Generator``.

    ``member_nodes`` gives each member's node count when a batch mixes sequences
    of different lengths, as length-bucketed batching does. Without it the node
    dimension is split into equal parts, which is the only possibility for one
    member and for a batch of replicates.
    """

    def __init__(
        self,
        sdes: Mapping[str, object],
        seeds: Sequence[int],
        *,
        member_nodes: Sequence[int] | None = None,
        allow_duplicate_seeds: bool = False,
    ) -> None:
        seeds = tuple(seeds)
        if not seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
        ):
            raise ValueError("seeds must be a non-empty sequence of nonnegative integers")
        if not allow_duplicate_seeds and len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique so every batch member is a distinct structure")
        if not isinstance(sdes, Mapping) or not sdes:
            raise TypeError("sdes must be a non-empty mapping of BioEmu SDEs")
        for name, sde in sdes.items():
            if not isinstance(name, str) or not callable(getattr(sde, "prior_sampling", None)):
                raise TypeError(f"sdes entry {name!r} does not expose a callable prior_sampling")
        if member_nodes is not None:
            member_nodes = tuple(member_nodes)
            if len(member_nodes) != len(seeds) or any(
                isinstance(count, bool) or not isinstance(count, int) or count < 1
                for count in member_nodes
            ):
                raise ValueError("member_nodes must give one positive node count per seed")
        self.sdes = dict(sdes)
        self.seeds = seeds
        self.member_nodes = member_nodes
        self.drawn: dict[str, torch.Tensor] = {}
        self._history: list[_PriorCall] = []
        self._originals: dict[str, Callable[..., torch.Tensor]] = {}
        self._restore: dict[str, object] = {}
        self._active = False

    def __enter__(self) -> SeededPriorSampling:
        if self._active:
            raise RuntimeError("SeededPriorSampling is already active")
        self._active = True
        try:
            for name, sde in self.sdes.items():
                self._originals[name] = sde.prior_sampling
                self._restore[name] = vars(sde).get("prior_sampling", _MISSING)
                sde.prior_sampling = self._make_prior_sampling(name)  # type: ignore[attr-defined]
        except Exception:
            self._restore_prior_sampling()
            raise
        return self

    def __exit__(self, exception_type: object, _exception: object, _traceback: object) -> bool:
        self._restore_prior_sampling()
        if exception_type is None:
            missing = tuple(name for name in self.sdes if name not in self.drawn)
            if missing:
                raise RuntimeError(
                    "BioEmu denoiser did not draw the seeded prior state for "
                    f"{missing}; per-member seeding did not take effect"
                )
        return False

    def _restore_prior_sampling(self) -> None:
        for name, sde in self.sdes.items():
            previous = self._restore.pop(name, _MISSING)
            if previous is _MISSING:
                if "prior_sampling" in vars(sde):
                    delattr(sde, "prior_sampling")
            else:
                sde.prior_sampling = previous  # type: ignore[attr-defined]
        self._active = False

    def _make_prior_sampling(self, name: str) -> Callable[..., torch.Tensor]:
        def prior_sampling(shape: torch.Size | tuple, device: object = None) -> torch.Tensor:
            return self._draw(name, shape, device)

        return prior_sampling

    def _member_shapes(self, name: str, shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        members = len(self.seeds)
        if not shape:
            raise ValueError(f"BioEmu prior shape for field {name!r} must not be scalar")
        if self.member_nodes is None:
            if shape[0] % members != 0:
                raise ValueError(
                    f"BioEmu prior shape {shape} for field {name!r} is not divisible into "
                    f"{members} batch members"
                )
            return tuple((shape[0] // members, *shape[1:]) for _ in range(members))
        if shape[0] != sum(self.member_nodes):
            raise ValueError(
                f"BioEmu prior shape {shape} for field {name!r} does not match the "
                f"{members} declared member node counts {self.member_nodes}"
            )
        return tuple((count, *shape[1:]) for count in self.member_nodes)

    def _draw(self, name: str, shape: torch.Size | tuple, device: object) -> torch.Tensor:
        if name in self.drawn:
            raise RuntimeError(f"BioEmu prior_sampling was called twice for field {name!r}")
        shape = tuple(int(size) for size in shape)
        members = len(self.seeds)
        member_shapes = self._member_shapes(name, shape)
        values: list[torch.Tensor] = []
        for index, seed in enumerate(self.seeds):
            verify_ambient_seed = index == 0 and not self._history
            rng_state = torch.get_rng_state() if verify_ambient_seed else None
            seed_everything(seed)
            if rng_state is not None and not torch.equal(torch.get_rng_state(), rng_state):
                raise RuntimeError(
                    "the global torch RNG was consumed between seed_everything(seeds[0]) and "
                    "BioEmu prior sampling; cached batch-size-1 structures would not be reproduced"
                )
            for previous in self._history:
                self._originals[previous.field](
                    previous.member_shapes[index], device=previous.device
                )
            values.append(self._originals[name](member_shapes[index], device=device))
        self._history.append(_PriorCall(field=name, member_shapes=member_shapes, device=device))
        drawn = values[0] if members == 1 else torch.cat(values, dim=0)
        if tuple(drawn.shape) != shape:
            raise RuntimeError(
                f"seeded BioEmu prior state for field {name!r} has shape "
                f"{tuple(drawn.shape)}, expected {shape}"
            )
        self.drawn[name] = drawn
        return drawn


def pad_member_coordinates(positions: torch.Tensor, member_residues: Sequence[int]) -> torch.Tensor:
    """Scatter node-concatenated coordinates into padded ``[B, max_residues, 3]``.

    ``torch_geometric.data.Batch.from_data_list`` concatenates node attributes in
    data-list order, so a batch of unequal-length sequences leaves the denoiser as a
    flat ``[sum(L_b), 3]`` tensor. This is the exact inverse of that packing. It is an
    out-of-place scatter so gradients flow back to every valid coordinate, and padded
    rows are exact zeros that the caller's residue mask must exclude.
    """
    counts = tuple(int(count) for count in member_residues)
    if not counts or any(count < 1 for count in counts):
        raise ValueError("member_residues must be positive residue counts")
    if not isinstance(positions, torch.Tensor) or positions.shape[-1] != 3:
        raise ValueError("positions must be a [..., 3] tensor")
    flat = positions.reshape(-1, 3)
    if flat.shape[0] != sum(counts):
        raise ValueError(
            f"BioEmu positions hold {flat.shape[0]} residues but the batch declares "
            f"{sum(counts)}"
        )
    members = len(counts)
    width = max(counts)
    index = torch.cat(
        [
            torch.arange(count, device=flat.device, dtype=torch.long) + member * width
            for member, count in enumerate(counts)
        ]
    )
    return (
        flat.new_zeros((members * width, 3)).index_copy(0, index, flat).reshape(members, width, 3)
    )


def require_seeded_batch_denoiser(
    config: TraceConfig | None, denoiser_config: Mapping[str, object]
) -> None:
    """Fail closed unless per-member initial seeding determines the whole trajectory.

    ``docs/steering/DESIGN_DECISIONS.md`` D8: seeding only each member's *initial*
    prior state reproduces independent batch-size-1 runs **because** the active
    ``dpm.yaml`` sets ``noise: 0.0``, which makes the trajectory a deterministic
    function of that initial state. Under ``heun`` (``noise: 0.5``) or
    ``stochastic_dpm`` every intermediate ``randn_like`` in
    ``EulerMaruyamaPredictor.update_given_drift_and_diffusion`` (``bioemu/denoiser.py``
    line 61) would additionally need per-member seeding, and a batched run would
    silently stop being reproducible. An absent ``noise`` key is the DPM solver's own
    default of ``0.0`` (``bioemu/denoiser.py`` line 275) and is accepted as such,
    matching :func:`~sae_bioemu.bioemu.steering.bioemu_score_call_timesteps`.
    """
    if not isinstance(denoiser_config, Mapping):
        raise TypeError("denoiser_config must be a mapping")
    denoiser_type = getattr(config, "denoiser_type", None)
    target = str(denoiser_config.get("_target_", ""))
    if denoiser_type != "dpm" or (target and "dpm_solver" not in target):
        raise ValueError(
            "batched BioEmu generation requires the DPM-Solver-2 denoiser: per-member "
            "seeding of the initial prior state is sufficient only for a deterministic "
            f"sampler (denoiser_type={denoiser_type!r} target={target!r})"
        )
    noise = denoiser_config.get("noise", 0.0)
    if isinstance(noise, bool) or not isinstance(noise, int | float) or float(noise) != 0.0:
        raise ValueError(
            "batched BioEmu generation requires denoiser noise 0.0: with noise every "
            "intermediate randn_like would also need per-member seeding, so a batched run "
            f"would not reproduce independent batch-size-1 runs (noise={noise!r})"
        )


#: Why one generation is thrown away before any measured one, quoted into every record of it.
WARM_UP_REASON = (
    "DESIGN_DECISIONS.md D7: BioEmu's dpm_solver draws the SO(3) orientation prior before "
    "so3_sde.to(device), so the first generation in a process draws it on CPU and every later "
    "one on GPU. Same seed, different stream, different structure -- and on real hardware the "
    "first generation either raises 'coordinates and mask must be on the same device' or, "
    "worse, silently returns a structure no later generation reproduces. One discarded "
    "generation moves the tables to the device, after which every measured generation shares "
    "one stream."
)

#: Lowercased substrings identifying D7's first-generation device mismatch. Only an error
#: matching one of these is absorbed by the warm-up; everything else is a real failure and is
#: re-raised, because a warm-up that swallowed an arbitrary exception would hide exactly the
#: class of problem D7 shows the cost of hiding.
WARM_UP_DEVICE_ERROR_MARKERS = (
    "must be on the same device",
    "expected all tensors to be on the same device",
    "found at least two devices",
)


def is_warm_up_device_error(error: BaseException) -> bool:
    """Return whether ``error`` is the device mismatch D7 predicts for a cold process."""
    message = str(error).lower()
    return any(marker in message for marker in WARM_UP_DEVICE_ERROR_MARKERS)


#: Attribute marking that a discarded warm-up generation is running on this sampler. Read by
#: :meth:`BioEmuRuntimeActivationSampler._persist_final_structures`, which refuses to publish
#: while it is set, and written only by :func:`warm_up_persistence_suppressed`.
WARM_UP_IN_PROGRESS_ATTRIBUTE = "_warm_up_in_progress"


@contextmanager
def warm_up_persistence_suppressed(sampler: object) -> Iterator[bool]:
    """Detach ``sampler``'s structure writer while the discarded D7 generation runs.

    **The warm-up used to be persisted, and it evicted the draw it was warming up for.**
    ``discarded_warm_up_generation`` calls ``generate_unsteered`` on the record and seeds of the
    generation about to be measured -- deliberately, so it exercises the same path -- and that
    method publishes through ``self.structure_writer`` under ``(sequence_id, seed)``. The
    discarded structure therefore landed on the MEASURED stem. The real draws then arrived,
    found the files already there, and ``skip_existing=True`` dropped them without a word.

    Measured over the 63,168 persisted stage-1 structures by recomputing each draw's ``the progress coordinate``
    from its stored coordinates against the ``the progress coordinate`` the run recorded: 8 of 1974 proteins held
    the cold warm-up instead of their measured draws -- the first protein of each of the 8
    shards, one batch of 8 draws each, deviating by 0.26 to 21 in ``the progress coordinate`` where the other 1965
    proteins agree at exactly 0.0. The published gate was unaffected because it measured the
    real draws in-process; only the coordinates on disk were wrong, and nothing said so.

    Suppression rather than a different stem, and rather than skipping the writer inside
    ``_unsteered_sampled_batch``: the discarded generation must remain byte-for-byte the same
    call the measured one makes (that is what makes it a warm-up at all), and a stem that
    encoded "warm-up" would still write a structure nothing should ever read. Detaching the
    writer removes the only durable effect while leaving the generation identical.

    RNG-neutral: this touches no random stream, and the warm-up still runs where it ran, before
    the caller's own ``seed_everything(seeds[0])``.

    Duck-typed for the same reason ``discarded_warm_up_generation`` is a free function: the
    sensitivity probe and the tests apply it to stub samplers that have no writer at all. A
    sampler with no ``structure_writer`` attribute is left alone and yields ``False``.

    Yields whether a writer was actually detached.
    """
    missing = object()
    writer = getattr(sampler, "structure_writer", missing)
    suppressed = writer is not missing and writer is not None
    previous = getattr(sampler, WARM_UP_IN_PROGRESS_ATTRIBUTE, False)
    setattr(sampler, WARM_UP_IN_PROGRESS_ATTRIBUTE, True)
    if suppressed:
        sampler.structure_writer = None  # type: ignore[attr-defined]
    try:
        yield suppressed
    finally:
        setattr(sampler, WARM_UP_IN_PROGRESS_ATTRIBUTE, previous)
        if writer is not missing:
            sampler.structure_writer = writer  # type: ignore[attr-defined]


def discarded_warm_up_generation(
    sampler: object,
    record: SequenceRecord | Sequence[SequenceRecord],
    seed: int | Sequence[int],
) -> dict[str, object]:
    """Burn one discarded generation and return the auditable record of what happened.

    This is the **single** implementation of the warm-up. It is a free function so the probe
    can apply it to a stub sampler and so
    :meth:`BioEmuRuntimeActivationSampler.ensure_warm_up_generation` can wrap it in a
    once-per-instance guard rather than restating it.

    Measured on real hardware: the first generation of a process returned Rg = 20.613 while
    every later generation of the same sequence and seed returned 11.269 to the bit -- 2.4
    sigma of pure artifact against an intervention effect of 0.005 angstrom.

    The device mismatch is expected here and is the *point*: it happens to the discarded
    generation instead of to a measured one. It is recorded rather than swallowed silently,
    because the day it stops happening is the day D7 was fixed upstream and every report
    should say so. Any other failure is re-raised untouched.

    ``record``/``seed`` take the *same* one-or-many forms as the generation about to be
    measured, and are passed through unchanged: the discarded generation is deliberately the
    same shape as the real one, so it exercises the same path and cannot fail for a reason
    the real call would not have had.

    **Discarded means discarded on disk too.** Because it shares the measured call's record and
    seeds it also shares its ``(sequence_id, seed)`` entry stems, so persisting it evicts the
    draws it exists to protect -- see :func:`warm_up_persistence_suppressed`, which detaches the
    structure writer around the generation and is the reason ``persisted`` is ``False`` in the
    returned record.
    """
    records = (record,) if isinstance(record, SequenceRecord) else tuple(record)
    seeds = (seed,) if isinstance(seed, int) and not isinstance(seed, bool) else tuple(seed)
    if not records or any(not isinstance(item, SequenceRecord) for item in records):
        raise TypeError("the warm-up generation needs one or more SequenceRecords")
    if len(seeds) != len(records) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in seeds
    ):
        raise ValueError("the warm-up needs one nonnegative integer seed per record")
    payload: dict[str, object] = {
        "performed": True,
        "reason": WARM_UP_REASON,
        "sequence_id": records[0].sequence_id,
        "seed": int(seeds[0]),
        "sequence_ids": [item.sequence_id for item in records],
        "seeds": [int(value) for value in seeds],
        # Recorded, not implied. The warm-up shares the measured generation's stems, so "this
        # structure was discarded rather than published" is the one property of it a later
        # reader of the store most needs, and an audit that has to infer it from code cannot.
        "persisted": False,
    }
    try:
        with warm_up_persistence_suppressed(sampler) as suppressed:
            payload["structure_writer_suppressed"] = suppressed
            coordinates = sampler.generate_unsteered(
                record,
                seed=seed,
                # A discarded generation cannot waste a measured one, and the caller reseeds
                # with seed_everything(seeds[0]) immediately afterwards, so what this consumed
                # is irrelevant. The paired dose objective legitimately sends members sharing
                # one draw, and the warm-up must mirror that batch's shape rather than reject
                # it.
                allow_duplicate_seeds=True,
            )  # type: ignore[attr-defined]
    except (RuntimeError, ValueError) as error:
        if not is_warm_up_device_error(error):
            payload.update(
                errored=True,
                error_class=type(error).__name__,
                error=f"{type(error).__name__}: {error}",
                absorbed=False,
            )
            raise
        payload.update(
            errored=True,
            error_class=type(error).__name__,
            error=f"{type(error).__name__}: {error}",
            absorbed=True,
        )
        return payload
    payload.update(
        errored=False,
        error_class=None,
        error=None,
        absorbed=False,
        discarded_residue_count=int(coordinates.shape[-2]),
    )
    return payload


class BioEmuRuntimeActivationSampler:
    """Keeps BioEmu loaded while sampling activation batches during SAE training."""

    #: What the once-per-instance warm-up did, or ``None`` before it has been attempted.
    #: A class default so the attribute exists on a bare instance built by ``__new__``.
    warm_up: dict[str, object] | None = None

    #: Whether a discarded D7 warm-up is running right now, set by
    #: :func:`warm_up_persistence_suppressed`. A class default for the same reason as
    #: :attr:`warm_up`, and read only by :meth:`_persist_final_structures`.
    _warm_up_in_progress: bool = False

    #: Generation precision, as a class default so the attribute always exists. FP32 is what
    #: every path did before the knob existed and is a strict no-op
    #: (:mod:`sae_bioemu.bioemu.precision`), so defaulting here cannot silently reduce
    #: precision anywhere.
    precision: GenerationPrecision = GenerationPrecision("fp32")

    def __init__(
        self,
        config: TraceConfig,
        cache_embeds_dir: str | Path | None = None,
        structures_dir: str | Path | None = None,
        batch_size: int = 1,
        device: str | torch.device | None = None,
        chemgraph_cache_size: int = DEFAULT_CONTEXT_CHEMGRAPH_CACHE_SIZE,
        precision: GenerationPrecision | str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        try:
            from bioemu.model_utils import load_model, load_sdes, maybe_download_checkpoint
            from bioemu.sample import get_context_chemgraph
            from torch_geometric.data import Batch
        except ImportError as exc:
            raise RuntimeError(
                "BioEmu runtime activation sampling requires installing the optional bioemu dependencies"
            ) from exc
        self.config = config
        self.cache_embeds_dir = cache_embeds_dir
        self.batch_size = batch_size
        self.device = device
        # Steering generation precision (fp32 by default, and then a strict no-op). It is
        # deliberately NOT applied to :meth:`sample`, which serves activation capture for
        # SAE training: those activations are a durable product of a different protocol and
        # must not change dtype because a steering run asked for speed.
        self.precision = resolve_generation_precision(precision)
        self.get_context_chemgraph = get_context_chemgraph
        self.chemgraph_cache = ContextChemGraphCache(chemgraph_cache_size)
        self.Batch = Batch
        ckpt_path, model_config_path = maybe_download_checkpoint(
            model_name=config.model_name,
            ckpt_path=config.ckpt_path,
            model_config_path=config.model_config_path,
        )
        self.resolved_checkpoint_path = Path(ckpt_path).expanduser().resolve()
        self.resolved_model_config_path = Path(model_config_path).expanduser().resolve()
        self.resolved_denoiser_config_path = _resolve_bioemu_denoiser_config_path(config)
        self.score_model = load_model(ckpt_path, model_config_path)
        self.score_model.eval()
        with _so3_cache_lock(config.cache_so3_dir, model_config_path):
            self.sdes = load_sdes(
                model_config_path=model_config_path,
                cache_so3_dir=config.cache_so3_dir,
            )
        self.denoiser_config = load_bioemu_denoiser_config(config)
        self.expected_score_calls = expected_bioemu_score_calls(config, self.denoiser_config)
        self.denoiser = instantiate_bioemu_denoiser(self.denoiser_config)
        self.runtime_asset_provenance = self._runtime_asset_provenance()
        self.extractor = BioEmuActivationExtractor(config)
        # FAIL CLOSED on the combination that reads as "saving" and saves nothing. Asking for
        # `save_final_structures=True` without a destination used to fall silently into the `else`
        # and disable the writer, so a caller could set the flag, run for GPU-hours, exit 0, and
        # discard every structure -- which is exactly what `run_stage1_gate.py` did. A request to
        # persist that cannot be honoured is a configuration error, not a default.
        if config.save_final_structures and structures_dir is None:
            raise ValueError(
                "TraceConfig(save_final_structures=True) needs a structures_dir: without one "
                "there is nowhere to write, and silently continuing would generate every "
                "structure and drop it while the flag reads as satisfied. Pass structures_dir, "
                "or set save_final_structures=False to say the draws are genuinely disposable"
            )
        self.structure_writer = (
            FinalStructureWriter(structures_dir, config.final_structures_subdir)
            if structures_dir is not None and config.save_final_structures
            else None
        )

    def _context_chemgraph(self, sequence: str) -> object:
        """Return one prepared context chemgraph, reusing the bounded LRU cache.

        The returned chemgraph is shared with the cache and must be treated as
        immutable; see :class:`ContextChemGraphCache`.
        """

        def load(value: str) -> object:
            return self.get_context_chemgraph(
                sequence=value,
                cache_embeds_dir=self.cache_embeds_dir,
            )

        cache = getattr(self, "chemgraph_cache", None)
        if cache is None:
            return load(sequence)
        return cache.get(sequence, load)

    def sample(self, records: list[SequenceRecord], seed: int) -> ActivationRecord:
        sink = ActivationRecordList()
        for record_batch in batched(records, self.batch_size):
            chemgraphs = [self._context_chemgraph(record.sequence) for record in record_batch]
            batch = self.Batch.from_data_list(chemgraphs)
            sequence_ids = [record.sequence_id for record in record_batch]
            capture = self.extractor.capture_from_denoiser(
                score_model=self.score_model,
                denoiser=self.denoiser,
                sdes=self.sdes,
                batch=batch,
                store=sink,
                sequence_ids=sequence_ids,
                sample_indices=[seed] * len(sequence_ids),
                device=self.device,
                expected_score_calls=self.expected_score_calls,
                seed=seed,
            )
            self._persist_final_structures(
                sampled_batch=capture.sampled_batch,
                records=list(record_batch),
                sample_index=seed,
            )
        return sink.concat()

    def _persist_final_structures(
        self,
        *,
        sampled_batch: object,
        records: list[SequenceRecord],
        sample_index: int | Sequence[int],
    ) -> None:
        """Publish one generated batch, and refuse outright to publish a discarded one.

        ``skip_existing=True`` at every call site: a resumed shard regenerates deterministically
        from the same seed, so re-publishing an identical structure is wasted I/O rather than a
        correction.

        **The guard is what that convenience cost once already.** The discarded D7 warm-up used
        to reach this path under the measured generation's own stems, so the real draws arrived
        to find their files present and were skipped in silence -- 8 of 1974 stage-1 proteins
        held the cold warm-up on disk after a 30 GPU-h run, and nothing in any report said so.
        :func:`warm_up_persistence_suppressed` now detaches the writer for the duration, and
        this raises if one is somehow reachable anyway, because a second silent skip is the one
        outcome that must not be possible.
        """
        writer = self.structure_writer
        if writer is None:
            return
        if self._warm_up_in_progress:
            raise RuntimeError(
                "the discarded D7 warm-up generation reached the final-structure writer. It "
                "shares the measured generation's (sequence_id, seed) stems, so publishing it "
                "would evict the very draws it exists to protect and skip_existing would hide "
                "that. Route every warm-up through warm_up_persistence_suppressed"
            )
        writer.append_batch(
            sampled_batch=sampled_batch,
            records=records,
            sample_index=sample_index,
            skip_existing=True,
        )

    def ensure_warm_up_generation(
        self,
        record: SequenceRecord | Sequence[SequenceRecord],
        seed: int | Sequence[int],
    ) -> dict[str, object]:
        """Burn one discarded generation, once per sampler, before any measured one (D7).

        **Lives here rather than in each tool because it has now been rediscovered four
        times**: steering training raised and survived on a retry, the sensitivity probe v1
        silently produced a 9.34 angstrom artifact that read as a 2.36 sigma signal, the probe
        v2 added its own warm-up, and the precision benchmark then died 32 seconds in with the
        same device mismatch because it had none. A consumer cannot forget what the sampler
        does for it.

        **Lazy, not in ``__init__``.** A caller that only ever calls :meth:`sample`, or that
        constructs a sampler to read ``denoiser_config``, must not pay for a generation it
        never asked for.

        **RNG-neutral by construction.** Every entry point calls this *before* its own
        ``seed_everything(seeds[0])``, which resets the Python, NumPy, torch, and CUDA
        streams outright, so whatever the discarded generation consumed cannot reach the first
        measured one and ``SeededPriorSampling``'s ambient-RNG assertion still holds. Calling
        it after that seeding would silently change every generated structure.

        Idempotent: the record is written *before* the inner generation runs, so the recursive
        :meth:`generate_unsteered` call inside it returns immediately instead of recursing.

        **Scope.** Applied by :meth:`steer` and :meth:`generate_unsteered`, the two steering
        entry points D7 has bitten, and deliberately *not* by :meth:`sample`: that path
        produces the durable activation caches of a different protocol, and silently inserting
        a generation ahead of the first captured batch would change what those caches contain.
        A process that calls :meth:`sample` first pays nothing extra -- it has already moved
        the SO(3) tables to the device, so the warm-up here simply does not raise.
        """
        recorded = self.warm_up
        if recorded is not None:
            return recorded
        pending: dict[str, object] = {
            "performed": True,
            "reason": WARM_UP_REASON,
            "in_progress": True,
        }
        self.warm_up = pending
        try:
            performed = discarded_warm_up_generation(self, record, seed)
        except BaseException:
            # An unrelated failure is not absorbed, and must not leave a sampler claiming a
            # warm-up it never completed.
            self.warm_up = None
            raise
        self.warm_up = performed
        return performed

    def assert_runtime_assets_unchanged(self) -> None:
        """Fail if any loaded BioEmu runtime asset changed after initialization."""
        if self._runtime_asset_provenance() != self.runtime_asset_provenance:
            raise RuntimeError("loaded BioEmu runtime assets changed during steering")

    def _runtime_asset_provenance(self) -> dict[str, dict[str, str]]:
        return {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in (
                ("checkpoint", self.resolved_checkpoint_path),
                ("model_config", self.resolved_model_config_path),
                ("denoiser_config", self.resolved_denoiser_config_path),
            )
        }

    def _steering_members(
        self,
        record: SequenceRecord | Sequence[SequenceRecord],
        seed: int | Sequence[int],
        conditioning: torch.Tensor | None = None,
        allow_duplicate_seeds: bool = False,
    ) -> tuple[tuple[SequenceRecord, ...], tuple[int, ...]]:
        """Normalize one-or-many records and seeds and fail closed on a bad batch.

        ``self.batch_size`` deliberately does **not** bound this: it sizes the
        activation-extraction batches of :meth:`sample`, which is a different
        concern from how many steered structures one differentiable generation
        produces. What must hold here is that every member carries its own seed
        so a batched run reproduces independent batch-size-1 runs (D8).

        The invariant that actually matters is that no two members produce the *same*
        structure, and a structure is determined by ``(sequence, seed, conditioning)``, not by
        the seed alone. The paired dose objective deliberately generates one sequence twice
        from one draw at two different levels -- that shared draw is what cancels the baseline
        noise -- so seed-only uniqueness would reject a correct batch. When ``conditioning`` is
        supplied the triple is checked; without it the original seed-only rule stands, so
        every other caller is unchanged.
        """
        records = (record,) if isinstance(record, SequenceRecord) else tuple(record)
        seeds = (seed,) if isinstance(seed, int) and not isinstance(seed, bool) else tuple(seed)
        if not records or any(not isinstance(item, SequenceRecord) for item in records):
            raise TypeError("record must be a SequenceRecord or a non-empty sequence of them")
        if len(seeds) != len(records):
            raise ValueError("BioEmu steering requires exactly one seed per batch member")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in seeds
        ):
            raise ValueError("seed must be a nonnegative integer")
        if conditioning is None:
            if not allow_duplicate_seeds and len(set(seeds)) != len(seeds):
                raise ValueError("BioEmu batch member seeds must be unique")
        else:
            values = conditioning.detach().reshape(-1).tolist()
            if len(values) != len(records):
                # Deliberately not raised here. A mismatched conditioning shape already has a
                # specific downstream error naming the tensor, and preempting it with a vaguer
                # one would make a shape bug harder to read. Fall back to the seed-only rule.
                if len(set(seeds)) != len(seeds):
                    raise ValueError("BioEmu batch member seeds must be unique")
                return records, seeds
            identity = [
                (item.sequence_id, member_seed, float(value))
                for item, member_seed, value in zip(records, seeds, values, strict=True)
            ]
            if len(set(identity)) != len(identity):
                raise ValueError(
                    "BioEmu batch members must differ in sequence, seed, or requested level: "
                    "two members sharing all three would generate one structure twice"
                )
        if len(records) > 1:
            require_seeded_batch_denoiser(self.config, self.denoiser_config)
        return records, seeds

    @staticmethod
    def _assert_member_residue_mask(
        residue_mask: torch.Tensor, records: Sequence[SequenceRecord]
    ) -> None:
        """Require a ``[B, max_residues]`` mask that is exactly each member's length."""
        lengths = tuple(len(item.sequence) for item in records)
        expected = (len(records), max(lengths))
        if residue_mask.ndim != 2 or residue_mask.dtype is not torch.bool:
            raise ValueError("residue_mask must be a boolean [batch, residues] tensor")
        if tuple(residue_mask.shape) != expected:
            raise ValueError(
                f"residue_mask must be {expected} for this batch, got {tuple(residue_mask.shape)}"
            )
        for index, length in enumerate(lengths):
            row = residue_mask[index]
            if not bool(row[:length].all()) or bool(row[length:].any()):
                raise ValueError(
                    "residue_mask must mark exactly the first sequence_length residues of "
                    f"batch member {index} as valid and pad the rest"
                )

    def steer(
        self,
        record: SequenceRecord | Sequence[SequenceRecord],
        *,
        seed: int | Sequence[int],
        site: SteeringSite,
        adapter: BoundedResidualAdapter,
        target: torch.Tensor,
        residue_mask: torch.Tensor,
        site_field: torch.Tensor | None = None,
    ) -> object:
        """Generate one differentiable adapter-steered BioEmu structure per member.

        ``record``/``seed`` accept a single value or one sequence of ``B >= 1``
        members. Members may differ in length: the structure module densifies with
        ``to_dense_batch``, so ``residue_mask`` marks the valid residues of each
        padded row and padding is masked everywhere downstream.

        ``site_field`` is the cryptic arms' per-residue channel, required by and only by an
        adapter declaring ``site_conditioning``. It is a per-protein constant supplied by the
        caller; nothing here derives it, and it reaches neither the warm-up generation nor
        ``seed_everything`` nor ``prior_seeds``, so two arms differing only in this channel walk
        identical noise draws (D-23).
        """
        from sae_bioemu.bioemu.steering import capture_or_steer_from_denoiser

        records, seeds = self._steering_members(record, seed, conditioning=target)
        self._assert_member_residue_mask(residue_mask, records)
        # Before the seeding, never after it: seed_everything below resets every stream, so
        # the discarded warm-up cannot perturb this generation (D7). It is given this call's
        # own members, so the discarded generation has the shape the measured one will have.
        self.ensure_warm_up_generation(records, seeds)
        seed_everything(seeds[0])
        chemgraphs = [self._context_chemgraph(item.sequence) for item in records]
        batch = self.Batch.from_data_list(chemgraphs)
        return capture_or_steer_from_denoiser(
            score_model=self.score_model,
            denoiser=self.denoiser,
            sdes=self.sdes,
            batch=batch,
            site=site,
            adapter=adapter,
            target=target,
            residue_mask=residue_mask,
            expected_property_name=adapter.property_name,
            expected_model_family="bioemu",
            prior_seeds=seeds,
            precision=self.precision,
            site_field=site_field,
        )

    def _unsteered_sampled_batch(
        self,
        record: SequenceRecord | Sequence[SequenceRecord],
        seed: int | Sequence[int],
        allow_duplicate_seeds: bool,
    ) -> tuple[Any, tuple[int, ...]]:
        """Run one adapter-off generation and return the raw ChemGraph plus member lengths.

        Extracted from :meth:`generate_unsteered` verbatim so that a second consumer can read
        an attribute other than ``.pos`` off the same object. The split changes no operation
        and no ordering: the warm-up, the seeding, the prior sampling and the precision scope
        are all still done here, once, in the order D7 requires.
        """
        records, seeds = self._steering_members(
            record, seed, allow_duplicate_seeds=allow_duplicate_seeds
        )
        lengths = tuple(len(item.sequence) for item in records)
        # Before the seeding, for the reason given in ``ensure_warm_up_generation``. This is
        # also the call the warm-up itself makes, and it returns immediately there.
        self.ensure_warm_up_generation(records, seeds)
        seed_everything(seeds[0])
        chemgraphs = [self._context_chemgraph(item.sequence) for item in records]
        batch = self.Batch.from_data_list(chemgraphs)
        device = next(self.score_model.parameters()).device
        if self.device is not None:
            device = torch.device(self.device)
        self.score_model.to(device)
        if hasattr(batch, "to"):
            batch = batch.to(device)
        member_nodes = lengths if len(seeds) > 1 else None
        with (
            SeededPriorSampling(
                self.sdes,
                seeds,
                member_nodes=member_nodes,
                # Carried through for the discarded D7 warm-up alone, which mirrors the shape
                # of a paired batch. Every measured unsteered call leaves it False, where two
                # members sharing a seed really would be one structure computed twice.
                allow_duplicate_seeds=allow_duplicate_seeds,
            ),
            torch.no_grad(),
        ):
            # The matched adapter-off arm must run at the SAME precision as the steered
            # arm: it is the counterfactual the paired metrics rest on (D6), so generating
            # it under different numerics would compare two generators, not two adapters.
            with generation_precision_scope(self.precision, device.type):
                sampled_batch = self.denoiser(
                    sdes=self.sdes,
                    device=device,
                    batch=batch,
                    score_model=self.score_model,
                )
        if isinstance(sampled_batch, tuple) or not hasattr(sampled_batch, "pos"):
            raise TypeError("BioEmu denoiser output must be one ChemGraph-like object with .pos")
        # PERSIST THE DRAW. Without this the adapter-off path generated structures and dropped
        # them: `self.structure_writer` was only ever consulted in `sample()` and
        # `sample_layers()`, and stage 1 reaches neither -- it comes in through
        # `generate_unsteered_frames` -> here. So `TraceConfig(save_final_structures=True)` was a
        # NO-OP for the one caller that most needs it, and the flag read as satisfied while every
        # draw was still discarded. That is PLAN.md item 26 a second time: a stage that costs
        # GPU-hours, whose raw observations cannot be recovered afterwards by any arithmetic.
        #
        # `skip_existing=True` matches `sample()`: a resumed shard regenerates deterministically
        # from the same seed, so re-publishing an identical structure is wasted I/O rather than a
        # correction. `seeds[0]` is the sample index for the same reason it is the value passed to
        # `seed_everything` above -- for a multi-member batch the members are one seeded
        # generation, indexed by their position within it.
        #
        # The discarded D7 warm-up comes through here too, with THIS call's records and seeds, so
        # `_persist_final_structures` is what keeps it off the disk: see
        # `warm_up_persistence_suppressed`.
        self._persist_final_structures(
            sampled_batch=sampled_batch,
            records=list(records),
            sample_index=list(seeds),
        )
        return sampled_batch, lengths

    def generate_unsteered(
        self,
        record: SequenceRecord | Sequence[SequenceRecord],
        *,
        seed: int | Sequence[int],
        allow_duplicate_seeds: bool = False,
    ) -> torch.Tensor:
        """Generate one adapter-off structure per member as an exact matched baseline.

        ``allow_duplicate_seeds`` is for the discarded D7 warm-up alone. Without conditioning
        two members sharing a seed really do generate one structure twice, so the guard is
        correct for every measured call and stays on by default.

        Returns padded ``[B, max_residues, 3]`` coordinates; rows past a member's own
        length are zero and must be excluded by the caller's residue mask.
        """
        sampled_batch, lengths = self._unsteered_sampled_batch(record, seed, allow_duplicate_seeds)
        positions = sampled_batch.pos
        expected = sum(lengths) * 3
        if (
            not isinstance(positions, torch.Tensor)
            or not positions.is_floating_point()
            or positions.numel() != expected
            or not torch.isfinite(positions).all()
        ):
            raise ValueError("BioEmu baseline positions must be finite and residue-aligned")
        positions = measurement_coordinates(positions, self.precision)
        if len(lengths) == 1:
            return positions.reshape(1, lengths[0], 3)
        return pad_member_coordinates(positions, lengths)

    def generate_unsteered_frames(
        self,
        record: SequenceRecord | Sequence[SequenceRecord],
        *,
        seed: int | Sequence[int],
        allow_duplicate_seeds: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate adapter-off structures and return their POSITIONS AND FRAMES.

        :meth:`generate_unsteered` returns a bare C-alpha tensor, which is all the
        radius-of-gyration work ever needed. The cryptic-pocket coordinate is built on the
        frame-placed virtual C-beta (PLAN.md D-12), which cannot be recovered from C-alpha
        alone: it needs the per-residue rigid frame, and reading it is the ``extractor.py``
        edit D-12 recorded as the cost of choosing ``cb``.

        Returns ``(positions, node_orientations)`` as ``[B, max_residues, 3]`` and
        ``[B, max_residues, 3, 3]``. ``positions`` is byte-identical to what
        :meth:`generate_unsteered` returns for the same arguments -- the same generation, the
        same validation, the same padding -- and is in BioEmu's own units, nanometres. Rows and
        rotations past a member's own length are zeros and must be excluded by the caller.
        """
        sampled_batch, lengths = self._unsteered_sampled_batch(record, seed, allow_duplicate_seeds)
        positions = sampled_batch.pos
        expected = sum(lengths) * 3
        if (
            not isinstance(positions, torch.Tensor)
            or not positions.is_floating_point()
            or positions.numel() != expected
            or not torch.isfinite(positions).all()
        ):
            raise ValueError("BioEmu baseline positions must be finite and residue-aligned")
        orientations = getattr(sampled_batch, "node_orientations", None)
        if (
            not isinstance(orientations, torch.Tensor)
            or not orientations.is_floating_point()
            or tuple(orientations.shape) != (sum(lengths), 3, 3)
            or not torch.isfinite(orientations).all()
        ):
            raise ValueError(
                "BioEmu baseline node_orientations must be a finite [residues, 3, 3] tensor; "
                "without it the frame-placed virtual C-beta cannot be constructed"
            )
        positions = measurement_coordinates(positions, self.precision)
        orientations = measurement_coordinates(orientations.reshape(-1, 3), self.precision).reshape(
            -1, 3, 3
        )
        if len(lengths) == 1:
            return (
                positions.reshape(1, lengths[0], 3),
                orientations.reshape(1, lengths[0], 3, 3),
            )
        width = max(lengths)
        padded = orientations.new_zeros((len(lengths), width, 3, 3))
        for member, (member_orientations, length) in enumerate(
            zip(torch.split(orientations, list(lengths)), lengths, strict=True)
        ):
            padded[member, :length] = member_orientations
        return pad_member_coordinates(positions, lengths), padded


class BioEmuMultiLayerRuntimeSampler(BioEmuRuntimeActivationSampler):
    """Capture several BioEmu layers from each trajectory in one model pass."""

    def __init__(
        self,
        config: TraceConfig,
        target_modules: tuple[str, ...],
        cache_embeds_dir: str | Path | None = None,
        structures_dir: str | Path | None = None,
        batch_size: int = 1,
        device: str | torch.device | None = None,
        chemgraph_cache_size: int = DEFAULT_CONTEXT_CHEMGRAPH_CACHE_SIZE,
    ) -> None:
        if not target_modules:
            raise ValueError("target_modules must be non-empty")
        super().__init__(
            config=config,
            cache_embeds_dir=cache_embeds_dir,
            structures_dir=structures_dir,
            batch_size=batch_size,
            device=device,
            chemgraph_cache_size=chemgraph_cache_size,
        )
        self.target_modules = tuple(dict.fromkeys(target_modules))

    def sample_layers(
        self,
        records: list[SequenceRecord],
        seed: int,
    ) -> dict[str, ActivationRecord]:
        sinks = {target: ActivationRecordList() for target in self.target_modules}
        for record_batch in batched(records, self.batch_size):
            chemgraphs = [self._context_chemgraph(record.sequence) for record in record_batch]
            batch = self.Batch.from_data_list(chemgraphs)
            sequence_ids = [record.sequence_id for record in record_batch]
            seed_everything(seed)
            recorder = MultiLayerRecordingScoreModel(
                score_model=self.score_model,
                target_modules=self.target_modules,
                stores=sinks,
                sequence_ids=sequence_ids,
                sample_indices=[seed] * len(sequence_ids),
                record_score_call_indices=record_score_call_indices(
                    self.expected_score_calls,
                    self.config.recorded_score_calls,
                ),
            ).to(self.device)
            try:
                if hasattr(batch, "to"):
                    batch = batch.to(self.device)
                with torch.no_grad():
                    sampled_batch = self.denoiser(
                        sdes=self.sdes,
                        device=self.device,
                        batch=batch,
                        score_model=recorder,
                    )
            finally:
                recorder.remove()
            self._persist_final_structures(
                sampled_batch=sampled_batch,
                records=list(record_batch),
                sample_index=seed,
            )
        return {target: sink.concat() for target, sink in sinks.items()}


class RecordingScoreModel(torch.nn.Module):
    """Proxy score model that records a target module during denoising.

    BioEmu denoisers call the score model repeatedly with the current noisy
    structure and timestep. This wrapper forwards those calls to the original
    score model, uses a forward hook on ``target_module`` to capture dense
    ``[batch, length, dim]`` activations, masks padded residues when possible,
    and appends flat ``ActivationRecord`` rows to the configured store.
    """

    def __init__(
        self,
        score_model: torch.nn.Module,
        target_module: str,
        store: ActivationStore | ActivationStoreWriter,
        sequence_ids: str | list[str] | tuple[str, ...],
        sample_indices: int | list[int] | tuple[int, ...] | torch.Tensor | None = None,
        record_score_call_indices: set[int] | None = None,
    ) -> None:
        super().__init__()
        if record_score_call_indices is not None and not record_score_call_indices:
            raise ValueError("record_score_call_indices must be non-empty")
        if record_score_call_indices is not None and min(record_score_call_indices) <= 0:
            raise ValueError("record_score_call_indices must contain positive indices")
        self.score_model = score_model
        self.store = store
        self.sequence_ids = sequence_ids
        self.sample_indices = sample_indices
        self.captured: list[torch.Tensor] = []
        self.record_score_call_indices = record_score_call_indices
        self.n_score_calls = 0
        self.n_records = 0
        self._record_current_call = False
        target = resolve_module_path(self.score_model, target_module)
        self.handle = target.register_forward_hook(self._hook)

    def _hook(self, _module: torch.nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        if not self._record_current_call:
            return
        if not isinstance(output, torch.Tensor):
            raise TypeError("target BioEmu hook output must be a tensor")
        self.captured.append(output.detach().cpu())

    def forward(self, batch: object, t: torch.Tensor) -> object:
        self.n_score_calls += 1
        should_record = (
            self.record_score_call_indices is None
            or self.n_score_calls in self.record_score_call_indices
        )
        self._record_current_call = should_record
        self.captured.clear()
        try:
            out = self.score_model(batch, t)
            if should_record:
                if not self.captured:
                    raise RuntimeError("BioEmu hook did not capture any activations")
                activations = self.captured[-1]
                mask = _get_dense_residue_mask(batch, activations)
                timestep = t.detach().cpu()
                record = ActivationRecord.from_dense(
                    activations=activations,
                    timestep=timestep,
                    sequence_ids=self.sequence_ids,
                    sample_indices=self.sample_indices,
                    mask=mask,
                )
                self.store.append(record)
                self.n_records += 1
            return out
        finally:
            self._record_current_call = False

    def remove(self) -> None:
        self.handle.remove()


class MultiLayerRecordingScoreModel(torch.nn.Module):
    """Record aligned activation rows from multiple modules in one score call."""

    def __init__(
        self,
        score_model: torch.nn.Module,
        target_modules: tuple[str, ...],
        stores: dict[str, ActivationRecordList],
        sequence_ids: str | list[str] | tuple[str, ...],
        sample_indices: int | list[int] | tuple[int, ...] | torch.Tensor | None = None,
        record_score_call_indices: set[int] | None = None,
    ) -> None:
        super().__init__()
        if set(target_modules) != set(stores):
            raise ValueError("target_modules and stores must have identical keys")
        self.score_model = score_model
        self.target_modules = target_modules
        self.stores = stores
        self.sequence_ids = sequence_ids
        self.sample_indices = sample_indices
        self.record_score_call_indices = record_score_call_indices
        self.n_score_calls = 0
        self._record_current_call = False
        self.captured: dict[str, list[torch.Tensor]] = {target: [] for target in target_modules}
        self.handles = [
            resolve_module_path(score_model, target).register_forward_hook(self._make_hook(target))
            for target in target_modules
        ]

    def _make_hook(self, target: str):
        def hook(_module: torch.nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            if not self._record_current_call:
                return
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"BioEmu hook output for {target} must be a tensor")
            self.captured[target].append(output.detach().cpu())

        return hook

    def forward(self, batch: object, t: torch.Tensor) -> object:
        self.n_score_calls += 1
        should_record = (
            self.record_score_call_indices is None
            or self.n_score_calls in self.record_score_call_indices
        )
        self._record_current_call = should_record
        for captured in self.captured.values():
            captured.clear()
        try:
            output = self.score_model(batch, t)
            if should_record:
                for target in self.target_modules:
                    if not self.captured[target]:
                        raise RuntimeError(f"BioEmu hook did not capture {target}")
                    activations = self.captured[target][-1]
                    mask = _get_dense_residue_mask(batch, activations)
                    self.stores[target].append(
                        ActivationRecord.from_dense(
                            activations=activations,
                            timestep=t.detach().cpu(),
                            sequence_ids=self.sequence_ids,
                            sample_indices=self.sample_indices,
                            mask=mask,
                        )
                    )
            return output
        finally:
            self._record_current_call = False

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def default_store_path(
    source_name: str,
    trace_config: TraceConfig,
    base_dir: str | Path = "activation_stores",
) -> Path:
    readable_source = _slugify(source_name)
    readable_model = _slugify(trace_config.model_name)
    readable_target = _slugify(trace_config.target_module)
    digest = hashlib.sha1(  # noqa: S324 - non-security hash for short deterministic names
        "|".join(
            [
                source_name,
                trace_config.model_name,
                trace_config.target_module,
                trace_config.denoiser_type,
                str(trace_config.denoiser_config_path),
                str(trace_config.recorded_score_calls),
                str(trace_config.samples_per_sequence),
            ]
        ).encode()
    ).hexdigest()[:10]
    return Path(base_dir) / f"{readable_source}__{readable_model}__{readable_target}__{digest}"


def load_bioemu_denoiser(config: TraceConfig) -> object:
    return instantiate_bioemu_denoiser(load_bioemu_denoiser_config(config))


def load_bioemu_denoiser_config(config: TraceConfig) -> dict:
    denoiser_config_path = _resolve_bioemu_denoiser_config_path(config)
    with denoiser_config_path.open() as handle:
        return yaml.safe_load(handle)


def _resolve_bioemu_denoiser_config_path(config: TraceConfig) -> Path:
    try:
        from bioemu.sample import DEFAULT_DENOISER_CONFIG_DIR
    except ImportError as exc:
        raise RuntimeError(
            "BioEmu denoiser loading requires installing the optional bioemu dependencies"
        ) from exc
    value = (
        Path(DEFAULT_DENOISER_CONFIG_DIR) / f"{config.denoiser_type}.yaml"
        if config.denoiser_config_path is None
        else Path(config.denoiser_config_path)
    )
    return value.expanduser().resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def instantiate_bioemu_denoiser(denoiser_config: dict) -> object:
    try:
        import hydra
    except ImportError as exc:
        raise RuntimeError(
            "BioEmu denoiser loading requires installing the optional bioemu dependencies"
        ) from exc
    return hydra.utils.instantiate(denoiser_config)


def expected_bioemu_score_calls(config: TraceConfig, denoiser_config: dict) -> int:
    n_steps = int(denoiser_config["N"])
    target = str(denoiser_config.get("_target_", ""))
    if config.denoiser_type == "dpm" or "dpm_solver" in target:
        return 2 * (n_steps - 1)
    if config.denoiser_type == "heun" or "heun_denoiser" in target:
        return 2 * n_steps - 1
    raise ValueError(f"Cannot infer score-call count for denoiser target {target!r}")


def dpm_gradient_steps_for_score_calls(score_calls: Iterable[int]) -> set[int]:
    """Map zero-based DPM score calls to iterations gated by ``record_grad_steps``."""
    calls = tuple(score_calls)
    if not calls or any(
        isinstance(call, bool) or not isinstance(call, int) or call < 0 for call in calls
    ):
        raise ValueError("score_calls must contain nonnegative zero-based integers")
    return {call // 2 for call in calls}


def record_score_call_indices(
    expected_score_calls: int | None,
    recorded_score_calls: int | None,
) -> set[int] | None:
    if recorded_score_calls is None:
        return None
    if expected_score_calls is None:
        raise ValueError("expected_score_calls is required when recorded_score_calls is set")
    if expected_score_calls <= 0:
        raise ValueError("expected_score_calls must be positive")
    if recorded_score_calls <= 0:
        raise ValueError("recorded_score_calls must be positive")
    if recorded_score_calls >= expected_score_calls:
        return None
    if recorded_score_calls == 1:
        return {round((expected_score_calls + 1) / 2)}
    return {
        round(1 + idx * (expected_score_calls - 1) / (recorded_score_calls - 1))
        for idx in range(recorded_score_calls)
    }


def source_name_from_path(path: str | Path) -> str:
    return Path(path).expanduser().name


def read_fasta(path: str | Path) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    current_id: str | None = None
    current_chunks: list[str] = []
    with Path(path).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append(SequenceRecord(current_id, _normalize_sequence(current_chunks)))
                current_id = line[1:].strip().split()[0]
                if not current_id:
                    raise ValueError("FASTA record has an empty id")
                current_chunks = []
            else:
                if current_id is None:
                    raise ValueError("FASTA sequence data appeared before the first header")
                current_chunks.append(line)
    if current_id is not None:
        records.append(SequenceRecord(current_id, _normalize_sequence(current_chunks)))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def read_sequence_csv(
    path: str | Path,
    sequence_column: str = "sequence",
    id_column: str = "id",
) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {path} has no header")
        missing = {sequence_column, id_column}.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV file {path} is missing columns: {sorted(missing)}")
        for row_idx, row in enumerate(reader, start=2):
            sequence_id = (row[id_column] or "").strip()
            sequence = (row[sequence_column] or "").strip().upper()
            if not sequence_id:
                raise ValueError(f"CSV row {row_idx} has an empty id")
            if not sequence:
                raise ValueError(f"CSV row {row_idx} has an empty sequence")
            records.append(SequenceRecord(sequence_id, sequence))
    if not records:
        raise ValueError(f"No sequence rows found in {path}")
    return records


def prepare_bioemu_embeddings(
    records: Iterable[SequenceRecord],
    cache_embeds_dir: str | Path | None,
    batch_size: int = 1,
    max_attempts: int = 4,
    retry_seconds: float = 30.0,
    msa_read_timeout: float = 120.0,
) -> int:
    """Run BioEmu context preparation so Evoformer/MSA embeddings are cached."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if retry_seconds < 0:
        raise ValueError("retry_seconds must be non-negative")
    if msa_read_timeout <= 0:
        raise ValueError("msa_read_timeout must be positive")
    try:
        import requests
        from bioemu.sample import get_context_chemgraph
    except ImportError as exc:
        raise RuntimeError(
            "Preparing BioEmu embeddings requires installing the optional bioemu dependencies"
        ) from exc

    processed = 0
    with _colabfold_read_timeout(msa_read_timeout):
        for record_batch in batched(records, batch_size):
            for record in record_batch:
                for attempt in range(1, max_attempts + 1):
                    try:
                        get_context_chemgraph(
                            sequence=record.sequence,
                            cache_embeds_dir=cache_embeds_dir,
                        )
                        break
                    except (requests.exceptions.RequestException, TimeoutError) as exc:
                        if attempt == max_attempts:
                            raise
                        delay = min(retry_seconds * (2 ** (attempt - 1)), 600.0)
                        print(
                            f"[{_timestamp()}] transient ColabFold failure for "
                            f"{record.sequence_id}; retrying in {delay:.0f}s "
                            f"({attempt}/{max_attempts}): {exc}",
                            flush=True,
                        )
                        time.sleep(delay)
                processed += 1
            if processed % 100 == 0:
                print(f"[{_timestamp()}] prepared BioEmu embeddings for {processed} sequences")
    return processed


@contextmanager
def _colabfold_read_timeout(minimum_read_timeout: float):
    """Raise BioEmu's short MMseqs result-download timeout for this process."""

    from bioemu.colabfold_inline import msa_client

    original_get = msa_client.requests.get
    original_post = msa_client.requests.post

    def with_read_timeout(request):
        def wrapped(*args, **kwargs):
            timeout = kwargs.get("timeout")
            if isinstance(timeout, int | float):
                kwargs["timeout"] = (timeout, max(float(timeout), minimum_read_timeout))
            return request(*args, **kwargs)

        return wrapped

    msa_client.requests.get = with_read_timeout(original_get)
    msa_client.requests.post = with_read_timeout(original_post)
    try:
        yield
    finally:
        msa_client.requests.get = original_get
        msa_client.requests.post = original_post


def batched(records: Iterable[SequenceRecord], batch_size: int) -> Iterable[list[SequenceRecord]]:
    batch: list[SequenceRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _normalize_sequence(chunks: list[str]) -> str:
    sequence = "".join(chunks).replace(" ", "").upper()
    if not sequence:
        raise ValueError("sequence is empty")
    return sequence


def _slugify(value: str, max_length: int = 48) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return (value or "unnamed")[:max_length]


def resolve_module_path(root: torch.nn.Module, path: str) -> torch.nn.Module:
    obj: object = root
    for part in path.split("."):
        if isinstance(obj, torch.nn.ModuleList):
            obj = obj[int(part)]
        elif isinstance(obj, list | tuple):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    if not isinstance(obj, torch.nn.Module):
        raise TypeError(f"path {path!r} did not resolve to a torch module")
    return obj


def _structure_entry_stem(sequence_id: str, sample_index: int) -> str:
    readable_id = _slugify(sequence_id, max_length=64)
    digest = hashlib.sha1(sequence_id.encode()).hexdigest()[:10]  # noqa: S324
    return f"{readable_id}-{digest}__sample_{sample_index:06d}"


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _get_dense_residue_mask(batch: object, activations: torch.Tensor) -> torch.Tensor | None:
    existing_mask = getattr(batch, "mask", None)
    if existing_mask is not None:
        return existing_mask.detach().cpu()
    batch_index = getattr(batch, "batch", None)
    if batch_index is None or activations.ndim != 3:
        return None
    num_graphs, max_length = activations.shape[:2]
    counts = torch.bincount(batch_index.detach().cpu(), minlength=num_graphs)
    positions = torch.arange(max_length)
    return positions[None, :] < counts[:, None]
