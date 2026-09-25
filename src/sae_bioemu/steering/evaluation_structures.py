"""Durable structure store for held-out steering evaluation units.

the project's guidelines puts final PDB/NPZ outputs in the durable category, and regenerating one
evaluation structure costs a full BioEmu denoiser run, so both arms of every held-out
unit are persisted next to the run's other artifacts rather than discarded once the
radius of gyration has been read off them.

The layout follows the runtime structure store BioEmu extraction already writes
(``activation_stores/*/final_structures``): one ``index.jsonl`` of one JSON record per
line beside per-structure ``.npz`` files, with the same
``<slug>-<digest>__sample_<seed>`` stem
(:func:`~sae_bioemu.bioemu.extractor._structure_entry_stem`, whose ``sample_index`` *is*
the seed -- ``extractor.py`` lines 468 and 477). Two things are made explicit here that
the generic store leaves implicit:

* **The seed.** It is recorded as ``seed`` as well as ``sample_index``, so a reader never
  has to know that the generic store's ``sample_index`` happens to be a seed.
* **The units.** BioEmu coordinates are nanometres internally while the radius of
  gyration is computed in Angstrom after multiplying by ten, so every entry and every
  ``.npz`` carries ``pos_units``, ``property_units``, and the explicit
  ``pos_to_property_units_scale``. ``property_units`` is derived from the property's NAME
  (:func:`property_units`) rather than fixed at ``"angstrom"``: the cryptic-pocket
  progress coordinate ``the progress coordinate`` is a ratio of projections onto the experimental apo->holo
  transition vector and carries no length unit at all, so a store that stamped every
  property ``"angstrom"`` would put a length on a dimensionless quantity and nothing
  downstream would catch it.

**Scope.** Only coordinates are persisted, not node orientations, so these files are not
backbone-complete and no PDB is written. The adapter-off arm reaches this module as
coordinates alone (``BioEmuRuntimeActivationSampler.generate_unsteered`` returns
positions), and writing a renderable PDB for the steered arm only would produce two arms
that are not comparable artifacts. The schema name records this rather than hiding it.

Training structures are deliberately **not** persisted anywhere: they come from a moving
adapter, so they are comparable neither to each other nor to anything downstream.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from sae_bioemu.bioemu.extractor import _structure_entry_stem

#: The null control arm: no adapter intervention, the unit's own seed.
ADAPTER_OFF_ARM = "adapter_off"
#: The steered arm: the same seed, the same sequence, the unit's requested level.
ADAPTER_ON_ARM = "adapter_on"
_ARMS = (ADAPTER_OFF_ARM, ADAPTER_ON_ARM)

#: Positions-only schema; see the module docstring for why no PDB is written.
#:
#: ``/2`` records that ``property_units`` is now resolved from the property name instead of being
#: the constant ``"angstrom"``. ``/1`` stays READABLE, and readable only as Angstrom: every entry
#: ever written under it came from the radius-of-gyration path, so rejecting it would strand the
#: held-out structure stores of every evaluated arm to buy nothing.
STRUCTURE_SCHEMA = "steering_evaluation_structure/2"
_LEGACY_STRUCTURE_SCHEMA = "steering_evaluation_structure/1"
_READABLE_SCHEMAS = (_LEGACY_STRUCTURE_SCHEMA, STRUCTURE_SCHEMA)

#: BioEmu coordinates are nanometres; a LENGTH-valued property is computed in Angstrom.
POSITION_UNITS = "nanometers"
PROPERTY_UNITS = "angstrom"
#: ``the progress coordinate`` is a ratio of projections onto the apo->holo transition vector: 0 at the experimental
#: apo, 1 at the experimental holo, and no Angstrom anywhere. It is the one steered property in
#: this project that is not a length, and the shared writer used to stamp it as one.
DIMENSIONLESS_PROPERTY_UNITS = "dimensionless"
#: Every property this store may persist, and the units its value carries. An unknown name is
#: refused rather than defaulted: defaulting is exactly how ``the progress coordinate`` acquired an Angstrom label.
_PROPERTY_UNITS = {
    "radius_of_gyration": PROPERTY_UNITS,
    "pocket_progress_coordinate": DIMENSIONLESS_PROPERTY_UNITS,
    # The three conformational objectives. Each is a superposed N/CA/C/O deviation from a deposited
    # conformer over that benchmark's metric region, so each is a LENGTH in angstrom -- the same
    # units as the radius of gyration. Absent, the writer refused every structure these six adapters
    # produced, which is right: it fails closed on an undeclared name rather than stamping a guess.
    "pocket_conformer_rmsd": PROPERTY_UNITS,
    "domain_motion_backbone_rmsd": PROPERTY_UNITS,
    "kinase_conformer_rmsd": PROPERTY_UNITS,
}
#: The nanometre -> Angstrom conversion applied to the COORDINATES before the property function
#: runs. It is a coordinate conversion, not a property conversion: for a dimensionless property
#: the scaled coordinates cancel and the recorded value is unitless regardless.
POSITION_TO_PROPERTY_SCALE = 10.0

_INDEX_NAME = "index.jsonl"


def property_units(property_name: str) -> str:
    """Return the units a steered property's value carries, failing closed on an unknown name."""
    try:
        return _PROPERTY_UNITS[property_name]
    except KeyError:
        raise ValueError(
            f"no units are declared for steered property {property_name!r}; add it to "
            "evaluation_structures._PROPERTY_UNITS rather than letting the store stamp a "
            "default, because a wrong unit on a persisted property is silent downstream"
        ) from None


@dataclass(frozen=True)
class SteeringStructureEntry:
    """One persisted held-out structure, keyed by sequence, seed, and arm."""

    schema: str
    sequence_id: str
    sequence: str
    seed: int
    sample_index: int
    arm: str
    unit_index: int
    visit: int
    level: float
    target: float
    residue_count: int
    npz_path: str
    pos_units: str
    property_name: str
    property_value: float
    property_units: str
    pos_to_property_units_scale: float

    def __post_init__(self) -> None:
        if self.schema not in _READABLE_SCHEMAS:
            raise ValueError(f"structure schema must be one of {_READABLE_SCHEMAS}")
        if self.arm not in _ARMS:
            raise ValueError(f"structure arm must be one of {_ARMS}")
        if self.seed != self.sample_index:
            raise ValueError(
                "sample_index must repeat the seed: the generic BioEmu structure store "
                "keys on sample_index and that index is the seed"
            )
        if self.pos_units != POSITION_UNITS:
            raise ValueError(f"structure positions must be recorded as {POSITION_UNITS}")
        # The units follow the PROPERTY, not the store.
        expected = property_units(self.property_name)
        if self.property_units != expected:
            raise ValueError(
                f"structure property units for {self.property_name!r} must be {expected!r}, "
                f"found {self.property_units!r}"
            )
        # ``/1`` hard-coded ``"angstrom"``, so a legacy entry can only ever have described a
        # length-valued property. Accepting one for a dimensionless property would accept a
        # file whose units were never checked as though they had been.
        if self.schema == _LEGACY_STRUCTURE_SCHEMA and expected != PROPERTY_UNITS:
            raise ValueError(
                f"{_LEGACY_STRUCTURE_SCHEMA!r} stamped every property {PROPERTY_UNITS!r}, so it "
                f"cannot describe {self.property_name!r}, which is {expected}"
            )

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.sequence_id, self.seed, self.arm)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class SteeringStructureWriter:
    """Append-only store of held-out evaluation structures, safe to re-enter.

    A structure is written **before** its unit's cache JSON is committed, so an
    interrupted batch can only ever leave structures whose unit will be regenerated. Both
    the payload write and the index append are therefore idempotent: the ``.npz`` is
    replaced atomically and the index gains at most one line per
    ``(sequence_id, seed, arm)`` key.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / _INDEX_NAME
        self._keys = {entry.key for entry in read_structure_index(self.root)}

    @property
    def entry_count(self) -> int:
        """Return how many structures the index currently declares."""
        return len(self._keys)

    def write(
        self,
        *,
        sequence_id: str,
        sequence: str,
        seed: int,
        arm: str,
        unit_index: int,
        visit: int,
        level: float,
        target: float,
        property_name: str,
        property_value: float,
        positions_nanometers: torch.Tensor,
    ) -> SteeringStructureEntry:
        """Persist one member's coordinates and index them exactly once."""
        positions = positions_nanometers.detach().to("cpu", dtype=torch.float32)
        if positions.ndim != 2 or positions.shape[-1] != 3 or positions.shape[0] < 1:
            raise ValueError("positions_nanometers must be a [residues, 3] tensor")
        if positions.shape[0] != len(sequence):
            raise ValueError(
                "persisted coordinates must be residue-aligned to the sequence: "
                f"{positions.shape[0]} rows for {len(sequence)} residues"
            )
        if not torch.isfinite(positions).all():
            raise ValueError("persisted coordinates must be finite")
        stem = f"{_structure_entry_stem(sequence_id, seed)}__{arm}"
        npz_path = self.root / f"{stem}.npz"
        units = property_units(str(property_name))
        entry = SteeringStructureEntry(
            schema=STRUCTURE_SCHEMA,
            sequence_id=sequence_id,
            sequence=sequence,
            seed=int(seed),
            sample_index=int(seed),
            arm=str(arm),
            unit_index=int(unit_index),
            visit=int(visit),
            level=float(level),
            target=float(target),
            residue_count=int(positions.shape[0]),
            npz_path=str(npz_path.relative_to(self.root)),
            pos_units=POSITION_UNITS,
            property_name=str(property_name),
            property_value=float(property_value),
            property_units=units,
            pos_to_property_units_scale=POSITION_TO_PROPERTY_SCALE,
        )
        temporary = npz_path.with_name(f".{npz_path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez(
                    handle,
                    pos=positions.numpy(),
                    pos_units=POSITION_UNITS,
                    sequence=sequence,
                    sequence_id=sequence_id,
                    seed=int(seed),
                    sample_index=int(seed),
                    arm=str(arm),
                    level=float(level),
                    target=float(target),
                    property_name=str(property_name),
                    property_value=float(property_value),
                    property_units=units,
                    pos_to_property_units_scale=POSITION_TO_PROPERTY_SCALE,
                )
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(npz_path)
        finally:
            temporary.unlink(missing_ok=True)
        if entry.key not in self._keys:
            with self.index_path.open("a") as index_handle:
                index_handle.write(entry.to_json() + "\n")
                index_handle.flush()
                os.fsync(index_handle.fileno())
            self._keys.add(entry.key)
        return entry


def read_structure_index(root: str | Path) -> tuple[SteeringStructureEntry, ...]:
    """Load every indexed structure record, failing closed on a malformed line."""
    index_path = Path(root) / _INDEX_NAME
    if not index_path.is_file():
        return ()
    entries: list[SteeringStructureEntry] = []
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"malformed steering structure index line in {index_path}")
        entries.append(SteeringStructureEntry(**payload))
    return tuple(entries)


def read_structure_positions(root: str | Path, entry: SteeringStructureEntry) -> np.ndarray:
    """Return one indexed structure's coordinates, in the units the entry declares."""
    with np.load(Path(root) / entry.npz_path, allow_pickle=False) as payload:
        positions = np.asarray(payload["pos"])
        if str(payload["pos_units"]) != entry.pos_units:
            raise ValueError("persisted coordinate units disagree with the index entry")
    if positions.shape != (entry.residue_count, 3):
        raise ValueError("persisted coordinates disagree with the indexed residue count")
    return positions


def iter_matched_arms(
    entries: tuple[SteeringStructureEntry, ...],
) -> Iterator[tuple[SteeringStructureEntry, SteeringStructureEntry]]:
    """Yield each unit's ``(adapter_off, adapter_on)`` pair, failing closed on a gap.

    The matched pair *is* the evaluation design: both arms share one seed, so a missing
    arm is a broken comparison and must be reported rather than skipped.
    """
    by_unit: dict[tuple[str, int], dict[str, SteeringStructureEntry]] = {}
    for entry in entries:
        by_unit.setdefault((entry.sequence_id, entry.seed), {})[entry.arm] = entry
    for key in sorted(by_unit):
        arms = by_unit[key]
        missing = tuple(arm for arm in _ARMS if arm not in arms)
        if missing:
            raise ValueError(
                f"held-out unit {key} is missing its {', '.join(missing)} structure; "
                "the paired off/on comparison is incomplete"
            )
        yield arms[ADAPTER_OFF_ARM], arms[ADAPTER_ON_ARM]
