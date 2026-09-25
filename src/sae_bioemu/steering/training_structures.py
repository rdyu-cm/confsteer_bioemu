"""Durable store for the structures a steering TRAINING run generates.

**Why this exists.** ``bioemu/steering_workflow.py`` built its sampler with no
``structures_dir``, and nothing else in that module wrote a PDB or an NPZ, so every structure a
training arm generated -- the steered draws and their matched unsteered controls alike -- was
computed at four seconds of H100 time apiece and then dropped. ``evaluation_structures.py``
records the reasoning that was applied at the time ("training structures come from a moving
adapter, so they are comparable neither to each other nor to anything downstream"), and the
standing instruction now overrides it: inference is the expensive part, every question this
project has asked of an old ensemble arrived after the run that produced it, and a coordinate
that was never written cannot be re-examined by any arithmetic.

**What is stored, and how a steered draw finds its control.**

    <root>/control/final_structures/<slug>-<digest>__sample_<draw_seed>.{pdb,npz}
    <root>/steered/pass_<k>/final_structures/<slug>-<digest>__sample_<unit_seed>.{pdb,npz}
    <root>/steered_draws.jsonl

The two structure directories are ordinary
:class:`~sae_bioemu.bioemu.extractor.FinalStructureWriter` stores -- the same layout, the same
``index.jsonl``, the same ``<slug>-<digest>__sample_<index>`` stem and the same nanometre/
Angstrom-centred unit conventions that the stage-1 training inputs already
holds -- so a reader that can open stage 1 can open this without learning a second format.

The control store is written by the sampler itself: ``generate_unsteered`` publishes through
``BioEmuRuntimeActivationSampler.structure_writer``, so pointing that at ``<root>/control``
is the whole of the control-side change.

**The two stores are keyed by DIFFERENT seeds, and that is the mapping.** A steered draw is
indexed by ``unit.seed``, which ``schedule._allocate_unique_seeds`` makes globally unique over
(split, sequence, cycle, visit, level). Its control is indexed by ``unit.generation_seed`` --
the shared ``draw_seed`` -- because one unsteered draw is the matched counterfactual for BOTH
levels of a pair and is generated once. So the two levels of one protein in one cycle are two
steered files pointing at one control file, which is exactly the paired design and not an
accident of naming. ``steered_draws.jsonl`` states the correspondence explicitly rather than
leaving a reader to know it: one row per persisted steered draw, carrying the split role, the
sequence id, the cycle, the visit, the requested level, the adapter's conditioning value, both
seeds, and the relative paths of all four files.

**``pass_<k>`` exists because validation units are walked again every epoch.**
``build_steering_schedule`` builds the validation units ONCE and the trainer replays them after
each cycle, so a validation unit's ``(sequence_id, unit.seed)`` stem repeats while the structure
under it does not -- the adapter has moved. A single store would hand those repeats to
``skip_existing=True`` and silently keep epoch 0's structure for all three epochs, which is the
failure mode that cost this project 7 of every 8 stage-1 draws once already. Each replay
therefore lands in its own pass directory. Train units are visited once, so they are always
``pass_00``; the directory is still named uniformly rather than special-cased, because a layout
that changes shape depending on the split is a layout a reader has to guess at.

The pass counter is seeded from what is already on disk, so a resumed run continues the
sequence instead of overwriting it. A run interrupted midway through a validation pass can
leave some units at pass ``k`` and some at ``k+1``; every row still names its own pass and its
own unit, so the store stays readable -- it is duplicated work, never a mislabelled structure.

**This module is in no code-fingerprint group, deliberately.** It writes files and reads none;
it takes no part in choosing a noise draw, building a target, computing a property, or scoring
anything. Hashing a pure writer into a checkpoint's identity would retire trained adapters for
a change that could not have altered them -- the "hash what determines, not what describes"
rule that ``_scientific_code_paths`` applies to stage 1 and the calibration writers for the
same reason.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sae_bioemu.bioemu.extractor import FinalStructureWriter, SequenceRecord

#: The sidecar's schema name. Bump it if a column's MEANING changes; adding a column does not
#: need a bump, because every reader here selects by key.
SIDECAR_SCHEMA = "steering_training_structures/1"

#: Directory holding the matched unsteered controls, written by the sampler's own writer.
CONTROL_DIRECTORY = "control"

#: Directory holding the steered draws, one ``pass_<k>`` store per replay of a unit.
STEERED_DIRECTORY = "steered"

#: One JSON object per persisted steered draw.
SIDECAR_NAME = "steered_draws.jsonl"


@dataclass(frozen=True)
class SteeredDrawRecord:
    """One persisted steered draw and the control it is matched to.

    Every field a reader needs to say what this structure IS: which protein, which split role,
    which cycle and visit of that role, which direction and how much of it, which noise draw,
    and where the counterfactual with the same noise draw and no intervention lives.
    """

    schema: str
    property_name: str
    split: str
    sequence_id: str
    cycle: int
    visit: int
    #: The scheduled dose level. For a two-state property this NAMES A STATE (-1 apo, +1 holo);
    #: for radius of gyration it is a multiple of the sequence's own sigma.
    level: float
    #: What the adapter was actually conditioned on for this draw. Equal to ``level`` on the
    #: relative path and a standardized absolute value on the absolute one, so the two are
    #: recorded separately rather than assumed identical.
    conditioning: float
    #: Globally unique over (split, sequence, cycle, visit, level): the steered stem.
    unit_seed: int
    #: The noise draw, shared by the levels of one (sequence, cycle): the control stem.
    draw_seed: int
    #: Which replay of this unit produced the structure; 0 for every train unit.
    pass_index: int
    steered_pdb: str
    steered_npz: str
    control_pdb: str
    control_npz: str
    #: Whether the control files were on disk when this row was written. False is possible and
    #: is not hidden: a relative radius-of-gyration run with ``lambda_l = 0`` generates no
    #: matched unsteered draw at all, so its steered structures have no counterfactual and a
    #: reader must not infer one from a path that was merely predicted.
    control_present: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class TrainingStructureStore:
    """Own the steered structure stores and the sidecar that links them to their controls."""

    def __init__(self, root: str | Path, *, property_name: str, subdir: str = "final_structures"):
        self.root = Path(root)
        self.property_name = str(property_name)
        self.subdir = str(subdir)
        self.control_root = self.root / CONTROL_DIRECTORY
        self.steered_root = self.root / STEERED_DIRECTORY
        self.sidecar_path = self.root / SIDECAR_NAME
        self.root.mkdir(parents=True, exist_ok=True)
        self._writers: dict[int, FinalStructureWriter] = {}
        self._next_pass: dict[tuple[str, int], int] = self._resume_pass_counts()

    def _pass_root(self, index: int) -> Path:
        return self.steered_root / f"pass_{index:02d}"

    def _resume_pass_counts(self) -> dict[tuple[str, int], int]:
        """Seed the per-unit replay counter from the sidecar a previous attempt left behind.

        Read from the SIDECAR rather than from the pass directories' own indices, because the
        sidecar is the only file that knows a structure's split role -- and a validation unit
        and a train unit could in principle carry the same seed only if the schedule allocator
        were changed, so keying on (split, seed) rather than on the seed alone keeps this
        correct without depending on that.
        """
        counts: dict[tuple[str, int], int] = {}
        if not self.sidecar_path.exists():
            return counts
        with self.sidecar_path.open() as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line)
                    key = (str(payload["split"]), int(payload["unit_seed"]))
                    seen = int(payload["pass_index"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"malformed steered-draw sidecar line in {self.sidecar_path}: a run "
                        "cannot resume against a store whose own record it cannot read"
                    ) from error
                counts[key] = max(counts.get(key, 0), seen + 1)
        return counts

    def _writer(self, index: int) -> FinalStructureWriter:
        writer = self._writers.get(index)
        if writer is None:
            writer = FinalStructureWriter(self._pass_root(index), self.subdir)
            self._writers[index] = writer
        return writer

    def control_entry_paths(self, sequence_id: str, draw_seed: int) -> tuple[str, str]:
        """Return the control's ``(pdb, npz)`` paths relative to this store's root."""
        from sae_bioemu.bioemu.extractor import _structure_entry_stem

        stem = _structure_entry_stem(sequence_id, int(draw_seed))
        base = Path(CONTROL_DIRECTORY) / self.subdir
        return str(base / f"{stem}.pdb"), str(base / f"{stem}.npz")

    def record_batch(
        self,
        *,
        sampled_batch: object,
        records: Sequence[SequenceRecord],
        units: Sequence[Any],
        conditioning: Sequence[float],
    ) -> tuple[SteeredDrawRecord, ...]:
        """Persist one generated batch of steered structures and append its sidecar rows.

        The whole batch goes to ONE pass directory, and the members must agree on which. They
        always do: a generation batch is a contiguous run of one role's scheduled units, so a
        batch is either all-train (always ``pass_00``) or all-validation at one epoch. The
        agreement is asserted rather than assumed, because a batch that spanned two replays
        would otherwise write two epochs' structures into one directory and the second would be
        skipped as an existing stem.

        Writing the batch in one ``append_batch`` call also keeps that writer's own guard live:
        it refuses a batch whose members share an entry stem, which is what caught stage 1
        writing one structure per batch where one per draw was expected.
        """
        if not (len(records) == len(units) == len(conditioning)):
            raise ValueError(
                f"steered-draw recording needs one record, unit and conditioning value per "
                f"batch member; got {len(records)}, {len(units)} and {len(conditioning)}"
            )
        if not units:
            return ()
        for record, unit in zip(records, units, strict=True):
            if record.sequence_id != unit.sequence_id:
                raise ValueError(
                    f"batch member names sequence {record.sequence_id!r} while its scheduled "
                    f"unit names {unit.sequence_id!r}; recording it would file one protein's "
                    "structure under another's"
                )
        indices = {self._next_pass.get((str(unit.split), int(unit.seed)), 0) for unit in units}
        if len(indices) != 1:
            raise ValueError(
                f"this generation batch mixes replay passes {sorted(indices)}; a batch spans "
                "one role at one epoch by construction, and writing it to one pass directory "
                "would silently drop the later replay"
            )
        index = indices.pop()
        entries = self._writer(index).append_batch(
            sampled_batch=sampled_batch,
            records=list(records),
            sample_index=[int(unit.seed) for unit in units],
            # A replay lands in its own pass directory, so a stem that already exists here
            # really is the same draw regenerated after an interruption -- deterministic from
            # the same seed and the same checkpointed adapter state.
            skip_existing=True,
        )
        pass_base = Path(STEERED_DIRECTORY) / f"pass_{index:02d}"
        rows: list[SteeredDrawRecord] = []
        for unit, entry, value in zip(units, entries, conditioning, strict=True):
            self._next_pass[(str(unit.split), int(unit.seed))] = index + 1
            control_pdb, control_npz = self.control_entry_paths(
                unit.sequence_id, int(unit.generation_seed)
            )
            rows.append(
                SteeredDrawRecord(
                    schema=SIDECAR_SCHEMA,
                    property_name=self.property_name,
                    split=str(unit.split),
                    sequence_id=str(unit.sequence_id),
                    cycle=int(unit.cycle),
                    visit=int(unit.visit),
                    level=float(unit.level),
                    conditioning=float(value),
                    unit_seed=int(unit.seed),
                    draw_seed=int(unit.generation_seed),
                    pass_index=index,
                    steered_pdb=str(pass_base / entry.pdb_path),
                    steered_npz=str(pass_base / entry.npz_path),
                    control_pdb=control_pdb,
                    control_npz=control_npz,
                    control_present=(self.root / control_pdb).exists()
                    and (self.root / control_npz).exists(),
                )
            )
        self._append_rows(rows)
        return tuple(rows)

    def _append_rows(self, rows: Sequence[SteeredDrawRecord]) -> None:
        if not rows:
            return
        with self.sidecar_path.open("a") as handle:
            for row in rows:
                handle.write(row.to_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def identity(self) -> dict[str, object]:
        """Return what the run report says about this store. Provenance, never a gate."""
        return {
            "schema": SIDECAR_SCHEMA,
            "root": str(self.root),
            "control_root": str(self.control_root),
            "steered_root": str(self.steered_root),
            "sidecar": str(self.sidecar_path),
        }


def read_steered_draws(root: str | Path) -> tuple[SteeredDrawRecord, ...]:
    """Load every steered-draw row, failing closed on a line this schema cannot describe."""
    path = Path(root) / SIDECAR_NAME
    if not path.exists():
        return ()
    rows: list[SteeredDrawRecord] = []
    with path.open() as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            try:
                rows.append(SteeredDrawRecord(**json.loads(raw_line)))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"malformed steered-draw sidecar line in {path}") from error
    return tuple(rows)


class SteeredDrawRecorder:
    """A sampler that also keeps the steered structures it generates.

    ``generate_exact_property_sample`` reaches the generator through exactly one method,
    ``steer``, and returns a ``GeneratedTrainingSample`` that carries the property and the
    adapter diagnostics but not the ChemGraph. Wrapping the sampler is therefore what lets the
    caller keep the structures without any file in the ``generation_shared`` fingerprint group
    changing: the raw batch is already on ``BioEmuSteeredSample.sampled_batch``, and this reads
    it on the way past. Nothing here touches a seed, a target, a residue mask or the site
    channel, so the draw is bit-identical to the unwrapped one.

    Every other attribute is forwarded, so the wrapper is a drop-in for the sampler.
    """

    def __init__(self, sampler: object, store: TrainingStructureStore) -> None:
        object.__setattr__(self, "_sampler", sampler)
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_pending", None)

    @property
    def sampler(self) -> object:
        return self._sampler

    @property
    def store(self) -> TrainingStructureStore:
        return self._store

    @contextmanager
    def expecting(self, units: Sequence[Any], conditioning: Sequence[float]) -> Iterator[None]:
        """Name the scheduled units the next ``steer`` call will generate, in batch order."""
        if self._pending is not None:
            raise RuntimeError(
                "a steered-draw recording is already open; nesting them would attribute one "
                "batch's structures to another batch's units"
            )
        object.__setattr__(self, "_pending", (tuple(units), tuple(float(v) for v in conditioning)))
        try:
            yield
        finally:
            object.__setattr__(self, "_pending", None)

    def steer(self, record: object, **kwargs: object) -> object:
        sample = self._sampler.steer(record, **kwargs)
        pending = self._pending
        if pending is None:
            # Nothing is silently dropped: every call site that generates a SCHEDULED draw
            # opens a recording, so an unrecorded call is a caller this store does not know
            # how to label and must be made to say so.
            raise RuntimeError(
                "a steered generation ran outside a steered-draw recording, so its structure "
                "could not be attributed to a scheduled unit. Wrap the call in "
                "SteeredDrawRecorder.expecting(...) or pass the bare sampler"
            )
        units, conditioning = pending
        records = (record,) if isinstance(record, SequenceRecord) else tuple(record)
        self._store.record_batch(
            sampled_batch=sample.sampled_batch,
            records=list(records),
            units=units,
            conditioning=conditioning,
        )
        return sample

    def __getattr__(self, name: str) -> object:
        return getattr(object.__getattribute__(self, "_sampler"), name)


@contextmanager
def record_steered_draws(
    sampler: object, units: Sequence[Any], conditioning: Sequence[float]
) -> Iterator[None]:
    """Open a recording when ``sampler`` records, and do nothing when it does not."""
    if not isinstance(sampler, SteeredDrawRecorder):
        yield
        return
    with sampler.expecting(units, conditioning):
        yield
