"""Serialize and load the fingerprinted pocket artifact.

The per-sequence reference table used for ``mean``/``sigma`` has a closed schema and can carry
nothing but ``{sequence_id, mean, sigma, sample_count, store_label}``, so everything the progress
coordinate needs -- pocket residues, pair indices, weights, the experimental transition, and the
parameters that produced them -- lives here instead.

Identity follows the reference table's pattern deliberately: a file SHA-256 plus a
path-independent content fingerprint, both bound into the training checkpoint. The filesystem
path is excluded so the same reviewed artifact copied to another checkout produces the same
identity, while any change to its contents does not.

The fingerprint must NOT reach ``schedule_seed_fingerprint``. That value covers only
``{schema, provenance, source_checkpoints, split}``, which is what lets conditioning arms share
noise draws; routing the artifact through it would re-randomise every arm and confound each
cross-arm difference with a complete redraw whose noise floor exceeds the effects being claimed.

Every entry carries its **sequence and that sequence's SHA-256**, and both are bound into the
identity. Without them the artifact stores residue indices "in BioEmu sequence coordinates" while
having no idea what that sequence is: rebuild the FASTA from a pool with one more or one fewer
observed residue and every index shifts by one, with every identity check still green. That is not
a hypothetical tolerance -- a one-residue shift was measured to leave the transition vector at a
median cosine of 0.68-0.78 with the truth, and above 0.9 for roughly a fifth of proteins, so it
degrades part of the pool while looking exactly like the pre-registered null.
:meth:`PocketArtifactTable.require_sequence` is the gate, and generation must call it with the
sequence it is actually about to hand the model.

Schema 3 adds the pocket-local gate's residue set for the same reason
------------------------------------------------------------------------

``gate_radius_angstrom`` is a top-level parameter beside ``cutoff_angstrom`` and
``context_radius_angstrom``, and ``gate_indices`` is a per-entry sorted index list. Both live here
rather than being recomputed where the gates are scored, because the neighbour set is an INPUT to
the gate and not an output of it: recomputing it from the SCORED structure would make the gate
self-selecting, since a steered structure that pushes its neighbours away shrinks its own
neighbourhood, and recomputing it offline lets the radius or the residue map drift between the run
that produced a structure and the run that scores it. The radius is in :meth:`identity` because it
is part of what the gate MEANS -- two runs that grew the neighbourhood at different radii are
scoring different residue sets and must not share one identity. See
``structure_gates.ARTIFACT_GATE_FIELDS``, whose :func:`~sae_bioemu.cryptic_steering.structure_gates.
neighbourhood_payload` is the canonical encoding this module reads and writes; the dependency runs
that way (``structure_gates`` imports this module) so the encoding is defined once.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from sae_bioemu.cryptic_steering.pocket import PAIR_CLASSES, PocketPairSet, PocketSpec

#: Bumped whenever the stored layout changes in a way older readers cannot interpret. Version 2
#: added the per-sequence string and digest; a version 1 artifact cannot be read as version 2
#: because it does not say which sequence its residue indices are coordinates in. Version 3 added
#: the top-level ``gate_radius_angstrom`` and the per-sequence ``gate_indices``: a version 2
#: artifact cannot be read as version 3 because it does not carry them, and loading one with the
#: radius defaulted would let two runs compute different pocket-local gates under one identity.
SCHEMA_VERSION = 3

_ATOM_SETS = frozenset({"ca", "cb"})
_REFERENCE_KINDS = frozenset({"matched_unsteered_draw", "experimental_apo"})

#: The exact fields ``structure_gates.neighbourhood_payload`` emits, restated here rather than
#: imported because ``structure_gates`` imports THIS module and the dependency may not reverse.
#: ``structure_gates.ARTIFACT_GATE_FIELDS`` is the definition and a test pins the two together.
_GATE_PAYLOAD_FIELDS = frozenset({"gate_radius_angstrom", "gate_indices"})


def _coordinate_payload(payload: dict) -> dict:
    """Return the payload MINUS the gate fields: everything that decides what ``the progress coordinate`` is (D-45).

    The gate fields are stripped at both levels -- the top-level ``gate_radius_angstrom`` and each
    sequence record's ``gate_indices`` -- because a partial strip would leave the gate in the
    digest through the per-sequence half and the relaxation would be a fiction.
    """
    stripped = {name: value for name, value in payload.items() if name not in _GATE_PAYLOAD_FIELDS}
    records = stripped.get("sequences")
    if isinstance(records, dict):
        stripped["sequences"] = {
            sequence_id: (
                {name: value for name, value in record.items() if name not in _GATE_PAYLOAD_FIELDS}
                if isinstance(record, dict)
                else record
            )
            for sequence_id, record in records.items()
        }
    return stripped


def _gate_payload(payload: dict) -> dict:
    """Return exactly what :func:`_coordinate_payload` removed, keyed by sequence.

    The complement is built explicitly rather than as "whatever is left", so a field added to
    the artifact lands in the coordinate digest by default -- the fail-CLOSED direction for a
    digest that binds a training checkpoint.
    """
    records = payload.get("sequences")
    per_sequence: dict[str, object] = {}
    if isinstance(records, dict):
        for sequence_id, record in records.items():
            if not isinstance(record, dict):
                continue
            gate = {name: value for name, value in record.items() if name in _GATE_PAYLOAD_FIELDS}
            if gate:
                per_sequence[str(sequence_id)] = gate
    return {
        "gate_radius_angstrom": payload.get("gate_radius_angstrom"),
        "sequences": dict(sorted(per_sequence.items())),
    }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sequence_digest(sequence: str) -> str:
    """Return the SHA-256 of one sequence, the quantity that pins the index frame."""
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PocketArtifactTable:
    """One fingerprinted artifact: every protein's pocket, pair set and transition."""

    path: Path
    schema_version: int
    cutoff_angstrom: float
    context_radius_angstrom: float
    #: The radius the pocket-local gate neighbourhood was grown at, one value for the whole
    #: artifact. RECORDED, and deliberately not in :meth:`identity` -- see D-45. A reader must
    #: know what the gate was scored over; a training checkpoint must not be invalidated by it.
    gate_radius_angstrom: float
    atom_set: str
    reference_kind: str
    class_weights: dict[str, float]
    #: SHA-256 of the file's raw bytes. RECORDED, not in :meth:`identity` -- it re-binds byte-level
    #: detail that determines nothing, key order and whitespace included (D-45).
    file_sha256: str
    #: Hashes everything that decides what ``the progress coordinate`` IS, and nothing that decides how a structure is
    #: scored afterwards: the payload MINUS the gate fields (D-45). It binds the training
    #: checkpoint, so what it covers is what a checkpoint may not be resumed or evaluated across.
    content_fingerprint: str
    #: Hashes exactly the gate fields ``content_fingerprint`` excludes: the radius and every
    #: sequence's derived ``gate_indices``. It keys the EVALUATION, which is the only thing the
    #: gate determines. Split from ``content_fingerprint`` rather than dropped, because the gate
    #: verdict every ``the progress coordinate`` is reported beside must still be replayable only against the gate that
    #: produced it -- what changed is which unit of work it rekeys, not whether it is checked.
    gate_fingerprint: str
    specs: dict[str, PocketSpec]
    pair_sets: dict[str, PocketPairSet]
    #: The exact residue string each entry's indices are positions in.
    sequences: dict[str, str]
    #: Per sequence, the sorted residue indices the pocket-local structure gates are scored over:
    #: the union over the apo and holo experimental states of the pocket residues and every residue
    #: within ``gate_radius_angstrom`` of one. Read it through
    #: ``structure_gates.neighbourhood_from_artifact``, which rebuilds the validated
    #: ``GateNeighbourhood``; a bare index list has no sequence bound to it.
    gate_indices: dict[str, tuple[int, ...]]

    @property
    def sequence_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.specs))

    @property
    def sequence_digests(self) -> dict[str, str]:
        """SHA-256 per sequence, in the form the identity and the generation gate compare."""
        return {name: sequence_digest(value) for name, value in self.sequences.items()}

    def identity(self) -> dict[str, object]:
        """Return the content-bound identity block for the training checkpoint.

        Every field that changes what the coordinate measures is present, so a run cannot alter
        the cutoff, the atom set or the reference and still resume from the same checkpoint.

        **The gate is deliberately absent, and that reversed an earlier decision (D-45.)** Neither
        ``gate_radius_angstrom`` nor the ``gate_indices`` it produced is here, and
        ``content_fingerprint`` no longer covers them either -- removing the key while leaving the
        payload hash over them would have been a cosmetic fix presenting itself as a real one. The
        structure gates are EVALUATION-only: they cannot change a generated structure or a loss, so
        binding them into a block that keys a training checkpoint meant re-growing the
        neighbourhood invalidated checkpoints whose objective never moved. The earlier version of
        this docstring stated that cost and accepted it; the cost was not worth paying.

        The gate is not unbound, it is bound at the right granularity: ``gate_fingerprint`` covers
        exactly what left, and the cryptic evaluation's resume identity carries it, so a scored
        draw still cannot be replayed against a gate that did not produce it. The radius stays
        RECORDED on this table, and ``gate_indices`` are still DERIVED from the artifact rather
        than recomputed at scoring time -- that derivation is the self-selection guard and it is
        untouched.

        ``file_sha256`` left this block with the gate, and for the same reason (D-45). It is the
        raw BYTES of the file, so it re-binds everything ``content_fingerprint`` deliberately
        excludes -- the gate among them -- and additionally re-binds things that determine nothing
        at all, such as key order and whitespace. Leaving it in would have made the gate relaxation
        cosmetic: the identity would have moved anyway, through a different key. It stays RECORDED
        on the table, where a reader can still say which file this was read from.

        ``sequence_digest`` is a single hash over ``{sequence_id: sha256(sequence)}`` rather than
        one entry per protein, so the block stays a fixed size while still failing on a
        one-residue change anywhere in the pool. It is redundant with ``content_fingerprint`` by
        construction; it is named separately because a reviewer comparing two identity blocks
        should be able to see *that the sequences differ* rather than only that something did.
        """
        return {
            "schema_version": self.schema_version,
            "cutoff_angstrom": self.cutoff_angstrom,
            "context_radius_angstrom": self.context_radius_angstrom,
            "atom_set": self.atom_set,
            "reference_kind": self.reference_kind,
            "class_weights": dict(sorted(self.class_weights.items())),
            "sequence_count": len(self.specs),
            "sequence_digest": hashlib.sha256(
                _canonical_json_bytes(dict(sorted(self.sequence_digests.items())))
            ).hexdigest(),
            "content_fingerprint": self.content_fingerprint,
        }

    def require_sequence(self, sequence_id: str, sequence: str) -> None:
        """Fail closed unless ``sequence`` is byte-identical to the one the indices assume.

        Call this at generation time with the string actually being handed to the model. The
        failure it exists for is silent: a pool rebuilt with one more or one fewer observed
        residue shifts every index by one, and nothing else in the pipeline notices -- not the
        residue count, which shifts with it, and not the file digest, which describes the artifact
        rather than the FASTA. The error message reports the length difference and the first
        differing position, because "the sequences differ" is not enough to tell an off-by-one
        from a different protein.
        """
        expected = self.sequences.get(sequence_id)
        if expected is None:
            raise ValueError(
                f"pocket artifact does not describe sequence {sequence_id!r}; "
                f"it carries {len(self.sequences)} sequences"
            )
        if sequence == expected:
            return
        detail = f"expected length {len(expected)}, received {len(sequence)}"
        if len(sequence) == len(expected):
            first = next(
                index for index, (a, b) in enumerate(zip(expected, sequence, strict=True)) if a != b
            )
            detail = f"same length {len(expected)}, first difference at position {first}"
        raise ValueError(
            f"sequence for {sequence_id!r} does not match the pocket artifact ({detail}); "
            "the stored residue indices are positions in the artifact's sequence, so they are "
            "meaningless against this one"
        )

    def require_sequences(self, sequence_ids: list[str]) -> None:
        """Fail closed when a split names a sequence the artifact does not describe.

        Silently skipping would train on a smaller pool than the split declares, which is the
        kind of discrepancy that survives every downstream check.
        """
        missing = sorted(set(sequence_ids) - set(self.specs))
        if missing:
            raise ValueError(
                f"pocket artifact is missing {len(missing)} split sequences; "
                f"first ten: {missing[:10]}"
            )


def _spec_payload(
    spec: PocketSpec, pairs: PocketPairSet, sequence: str, gate_indices: Sequence[int]
) -> dict[str, object]:
    return {
        "residue_count": spec.residue_count,
        # The sequence and its digest are what make "pocket_indices are in BioEmu sequence
        # coordinates" a checkable statement rather than a comment. The digest is stored beside
        # the sequence, redundantly, so a reader that never loads the string still fingerprints it.
        "sequence": sequence,
        "sequence_sha256": sequence_digest(sequence),
        "pocket_indices": list(spec.pocket_indices),
        "context_indices": list(spec.context_indices),
        "gate_indices": [int(value) for value in gate_indices],
        "pair_index": pairs.pair_index.tolist(),
        "pair_class": list(pairs.pair_class),
        "weights": [float(value) for value in pairs.weights],
        "delta_d_star": [float(value) for value in pairs.delta_d_star],
    }


def _checked_gate_indices(
    sequence_id: str, indices: Sequence[int], spec: PocketSpec
) -> tuple[int, ...]:
    """Return the gate index list, or fail closed. Used on BOTH write and load.

    One definition, called from both sides, because a set validated only where it is written is a
    set a hand-edited file can carry past every reader. These are exactly the invariants
    ``GateNeighbourhood.__post_init__`` enforces on the object this list rebuilds; the object
    re-checks them, and this catches a broken file before anyone asks for the object.
    """
    values = tuple(int(value) for value in indices)
    if not values:
        raise ValueError(f"gate_indices for {sequence_id!r} is empty; the gate would score nothing")
    if len(set(values)) != len(values) or tuple(sorted(values)) != values:
        raise ValueError(
            f"gate_indices for {sequence_id!r} must be sorted and unique, so two builds of the "
            "same inputs are byte-identical"
        )
    if any(value < 0 or value >= spec.residue_count for value in values):
        raise ValueError(
            f"gate_indices for {sequence_id!r} must lie inside the sequence "
            f"({spec.residue_count} residues)"
        )
    if not set(spec.pocket_indices) <= set(values):
        raise ValueError(
            f"gate_indices for {sequence_id!r} must contain the pocket it was grown from; a gate "
            "that can exclude a pocket residue is not measuring the pocket"
        )
    return values


def write_pocket_artifact(
    path: str | Path,
    specs: dict[str, PocketSpec],
    pair_sets: dict[str, PocketPairSet],
    sequences: dict[str, str],
    gate_neighbourhoods: Mapping[str, Mapping[str, object]],
    *,
    cutoff_angstrom: float,
    context_radius_angstrom: float,
    atom_set: str,
    reference_kind: str,
    class_weights: dict[str, float],
) -> Path:
    """Write the artifact in canonical form, so two identical builds are byte-identical.

    ``sequences`` is required, not optional: an artifact of residue indices that does not say
    which sequence they index cannot be validated against anything at generation time.

    ``gate_neighbourhoods`` maps each sequence id to the canonical encoding
    ``structure_gates.neighbourhood_payload`` produces. It is required for the same reason and one
    more: a defaulted gate radius would let two runs score the pocket-local gates over different
    residue sets while presenting one artifact identity. The radius is hoisted to a single
    top-level field and every entry must agree on it, because it is one value for the gate rather
    than a per-protein choice.
    """
    if atom_set not in _ATOM_SETS:
        raise ValueError(f"atom_set must be one of {sorted(_ATOM_SETS)}")
    if reference_kind not in _REFERENCE_KINDS:
        raise ValueError(f"reference_kind must be one of {sorted(_REFERENCE_KINDS)}")
    if set(specs) != set(pair_sets):
        raise ValueError("specs and pair_sets must describe the same sequences")
    if set(specs) != set(sequences):
        raise ValueError("specs and sequences must describe the same sequences")
    if set(specs) != set(gate_neighbourhoods):
        raise ValueError("specs and gate_neighbourhoods must describe the same sequences")
    unknown = set(class_weights) - set(PAIR_CLASSES)
    if unknown:
        raise ValueError(f"unknown pair classes in class_weights: {sorted(unknown)}")

    radii: set[float] = set()
    gate_indices: dict[str, tuple[int, ...]] = {}
    for sequence_id, spec in specs.items():
        if len(sequences[sequence_id]) != spec.residue_count:
            raise ValueError(
                f"sequence for {sequence_id!r} has {len(sequences[sequence_id])} residues but its "
                f"spec declares {spec.residue_count}; the pocket indices would not address it"
            )
        neighbourhood = gate_neighbourhoods[sequence_id]
        if set(neighbourhood) != _GATE_PAYLOAD_FIELDS:
            raise ValueError(
                f"the gate neighbourhood for {sequence_id!r} must be the canonical encoding "
                "structure_gates.neighbourhood_payload produces, whose fields are exactly "
                f"{sorted(_GATE_PAYLOAD_FIELDS)}; received {sorted(neighbourhood)}"
            )
        radius = float(neighbourhood["gate_radius_angstrom"])  # type: ignore[arg-type]
        if not radius > 0:
            raise ValueError(f"gate_radius_angstrom for {sequence_id!r} must be positive")
        radii.add(radius)
        gate_indices[sequence_id] = _checked_gate_indices(
            sequence_id,
            neighbourhood["gate_indices"],  # type: ignore[arg-type]
            spec,
        )
    if len(radii) != 1:
        raise ValueError(
            "every gate neighbourhood in one artifact must be grown at the same radius, because "
            "gate_radius_angstrom is a single top-level field bound into the identity; received "
            f"{sorted(radii)}"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "cutoff_angstrom": float(cutoff_angstrom),
        "context_radius_angstrom": float(context_radius_angstrom),
        "gate_radius_angstrom": next(iter(radii)),
        "atom_set": atom_set,
        "reference_kind": reference_kind,
        "class_weights": {name: float(class_weights.get(name, 1.0)) for name in PAIR_CLASSES},
        "sequences": {
            sequence_id: _spec_payload(
                specs[sequence_id],
                pair_sets[sequence_id],
                sequences[sequence_id],
                gate_indices[sequence_id],
            )
            for sequence_id in sorted(specs)
        },
    }
    target = Path(path)
    target.write_bytes(_canonical_json_bytes(payload))
    return target


def load_pocket_artifact(path: str | Path) -> PocketArtifactTable:
    """Load and validate an artifact, recomputing both digests from what is on disk."""
    resolved = Path(path).resolve()
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise ValueError(f"pocket artifact is unreadable: {resolved}") from error

    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("pocket artifact must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"pocket artifact {resolved} declares schema version "
            f"{payload.get('schema_version')!r}, which is not the supported version "
            f"{SCHEMA_VERSION}. Version {SCHEMA_VERSION} added the top-level "
            "'gate_radius_angstrom' and the per-sequence 'gate_indices' that the pocket-local "
            "structure gates are scored over, and an older file does not carry them; loading it "
            "with the radius defaulted would let two runs score different residue sets under one "
            "artifact identity. Regenerate it with scripts/cryptic_steering/build_stage1_inputs.py "
            "into a NEW --output-dir, which rewrites pocket_artifact.json and stage1_inputs.json "
            "together."
        )
    atom_set = payload["atom_set"]
    reference_kind = payload["reference_kind"]
    if atom_set not in _ATOM_SETS or reference_kind not in _REFERENCE_KINDS:
        raise ValueError("pocket artifact declares an unapproved atom set or reference kind")

    if "gate_radius_angstrom" not in payload:
        raise ValueError(
            f"pocket artifact declares schema {SCHEMA_VERSION} but carries no top-level "
            "gate_radius_angstrom; the radius the pocket-local gate neighbourhood was grown at is "
            "part of what the gate means and may not be defaulted"
        )
    gate_radius_angstrom = float(payload["gate_radius_angstrom"])
    if not gate_radius_angstrom > 0:
        raise ValueError("pocket artifact declares a nonpositive gate_radius_angstrom")

    specs: dict[str, PocketSpec] = {}
    pair_sets: dict[str, PocketPairSet] = {}
    sequences: dict[str, str] = {}
    gate_indices: dict[str, tuple[int, ...]] = {}
    for sequence_id, record in payload["sequences"].items():
        sequence = str(record["sequence"])
        # The stored digest is recomputed rather than trusted: a hand-edited sequence with its
        # digest left behind would otherwise pass every check that reads the digest alone.
        if record["sequence_sha256"] != sequence_digest(sequence):
            raise ValueError(
                f"pocket artifact entry {sequence_id!r} carries a sequence digest that does not "
                "match its sequence"
            )
        if len(sequence) != int(record["residue_count"]):
            raise ValueError(
                f"pocket artifact entry {sequence_id!r} has a sequence of {len(sequence)} residues "
                f"but declares {record['residue_count']}"
            )
        sequences[sequence_id] = sequence
        spec = PocketSpec(
            sequence_id=sequence_id,
            residue_count=int(record["residue_count"]),
            pocket_indices=tuple(int(value) for value in record["pocket_indices"]),
            context_indices=tuple(int(value) for value in record["context_indices"]),
        )
        pair_index = torch.tensor(record["pair_index"], dtype=torch.long).reshape(-1, 2)
        pair_sets[sequence_id] = PocketPairSet(
            sequence_id=sequence_id,
            pair_index=pair_index,
            pair_class=tuple(record["pair_class"]),
            weights=torch.tensor(record["weights"], dtype=torch.float64),
            delta_d_star=torch.tensor(record["delta_d_star"], dtype=torch.float64),
        )
        if "gate_indices" not in record:
            raise ValueError(
                f"pocket artifact entry {sequence_id!r} declares schema {SCHEMA_VERSION} but "
                "carries no gate_indices; the pocket-local gates have no residue set to score over"
            )
        gate_indices[sequence_id] = _checked_gate_indices(sequence_id, record["gate_indices"], spec)
        specs[sequence_id] = spec

    if not specs:
        raise ValueError(f"pocket artifact {resolved} describes no sequences")

    return PocketArtifactTable(
        path=resolved,
        schema_version=SCHEMA_VERSION,
        cutoff_angstrom=float(payload["cutoff_angstrom"]),
        context_radius_angstrom=float(payload["context_radius_angstrom"]),
        gate_radius_angstrom=gate_radius_angstrom,
        atom_set=atom_set,
        reference_kind=reference_kind,
        class_weights={name: float(value) for name, value in payload["class_weights"].items()},
        file_sha256=hashlib.sha256(raw).hexdigest(),
        content_fingerprint=hashlib.sha256(
            _canonical_json_bytes(_coordinate_payload(payload))
        ).hexdigest(),
        gate_fingerprint=hashlib.sha256(_canonical_json_bytes(_gate_payload(payload))).hexdigest(),
        specs=specs,
        pair_sets=pair_sets,
        sequences=sequences,
        gate_indices=gate_indices,
    )
