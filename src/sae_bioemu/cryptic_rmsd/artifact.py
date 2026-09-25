"""The pocket-RMSD reference: each system's deposited apo and holo, the metric region, and ``s_p``.

SCOPE: CRYPTIC POCKETS. This module decides WHICH ATOMS the direct metric is read over and WHAT
each system's normalisation is, so it is a member of this property's training fingerprint group: a
change here changes every deviation the objective descends.

WHY A STANDALONE ARTIFACT AND NOT THE EXISTING ``metric_reference.json``
-------------------------------------------------------------------------

``pocket_progress_coordinate``'s direct-metric term reads a SIDECAR --
:class:`~sae_bioemu.cryptic_steering.direct_metric.CrypticMetricReference` -- which is signed
against the fingerprinted pocket artifact and whose ``entry_for`` REQUIRES that artifact on every
read, because its residue indices are positions in the artifact's own sequence. That binding is
correct for a property that also reads the artifact for its pair set, its transition vector and its
site channel. This property reads none of those.

Consuming the sidecar here would therefore mean loading ``cryptic_steering/artifact.py`` too, and
``scientific_code_paths`` refuses one source in two groups -- so both files would have to be named
through ``cryptic_training_only`` and ``cryptic_generation_shared``, binding every
``pocket_conformer_rmsd`` checkpoint to the ``the progress coordinate`` pair geometry, the D-31 curvature bound, the
apo sidecar, the stage-1 setpoint reader, the site-field cache and the cryptic loss. A run of this
property executes not one line of any of them.

So the geometry is REPACKAGED once, offline, by
``scripts/cryptic_rmsd/build_pocket_rmsd_reference.py``, which reads the signed sidecar and its
artifact, carries their digests into :attr:`PocketRmsdReference.provenance`, and writes a file this
module can validate alone. That is the ``kinase_rmsd`` shape and it was taken for the same reason:
the training path must be able to check everything it depends on without importing another
property's package.

TWO THINGS THIS ARTIFACT CARRIES THAT THE SIDECAR DOES NOT
-----------------------------------------------------------

1. **THE SEQUENCE ITSELF, not a digest of one held elsewhere.** The sidecar stores
   ``sequence_sha256`` and defers to the pocket artifact for the string. Here the string is stored,
   and :func:`pocket_rmsd_metric_batch` re-checks it against the sequence the generator was
   actually handed, character for character, on every batch. That is the index-frame gate
   ``PocketArtifactTable.require_sequence`` is for the progress coordinate: the metric indices are
   positions in THIS string, and a structure bank rebuilt with one more observed residue shifts
   every one of them while every other check stays green. Measured elsewhere in this project, a
   one-residue shift leaves the transition vector at a median cosine of 0.68-0.78 with the truth --
   it degrades part of the pool while looking exactly like the pre-registered null.

2. **THE BACKBONES ARE PRE-RESTRICTED** to ``metric_indices``: ``[K, 4, 3]`` rather than
   ``[L, 4, 3]`` plus a mask. There is then no selector to pass differently, and
   :func:`~sae_bioemu.cryptic_rmsd.objective.aligned_backbone_rmsd` takes no ``metric_residues``
   argument at all, so "the alignment set is the metric set" is structural rather than a convention
   two call sites agree on. The sidecar already stores its backbones this way and the property is
   preserved here deliberately.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from sae_bioemu.cryptic_rmsd.frames import BACKBONE_ATOMS
from sae_bioemu.cryptic_rmsd.objective import (
    DEFAULT_METRIC_OBJECTIVE,
    GRADED_DEVIATION_FORM,
    TRAINED_DEVIATION_FORM,
    PocketRmsdNormalisation,
    aligned_backbone_rmsd,
    reference_state_for_level,
    validated_metric_objective,
)

__all__ = [
    "SCHEMA_VERSION",
    "SEPARATION_TOLERANCE_ANGSTROM",
    "PocketRmsdDefinition",
    "PocketRmsdEntry",
    "PocketRmsdReference",
    "load_pocket_rmsd_reference",
    "pocket_rmsd_metric_batch",
]

#: Bumped whenever the meaning of a stored field changes. A mismatched version is refused at load:
#: the alternative is reading an old file's coordinates under a new field's meaning, which produces
#: finite, plausible deviations measured against the wrong thing.
SCHEMA_VERSION = 1

#: How far a re-measured ``s_p`` may sit from the stored one, in Angstrom, before the load is
#: refused. Loose enough for a float64 Kabsch across BLAS versions; far tighter than any real
#: disagreement, because a one-residue index slip moves an aligned RMS by tenths of an Angstrom at
#: least.
SEPARATION_TOLERANCE_ANGSTROM = 1e-3


@dataclass(frozen=True)
class PocketRmsdEntry:
    """One system's two deposited conformers, restricted to the residues the metric is read over."""

    sequence_id: str
    #: The sequence BioEmu folds. ``metric_indices`` are positions in THIS string, and
    #: :func:`pocket_rmsd_metric_batch` re-checks each member's string at generation time.
    sequence: str
    #: Positions in :attr:`sequence` the metric is read over, ascending and distinct. The metric
    #: region, the alignment set and the set ``s_p`` is measured over: one set, by construction.
    metric_indices: tuple[int, ...]
    #: ``[K, 4, 3]`` in Angstrom, in :data:`~sae_bioemu.cryptic_rmsd.frames.BACKBONE_ATOMS` order,
    #: ALREADY RESTRICTED to :attr:`metric_indices`. The deposited APO conformer.
    apo: Tensor
    #: The same, for the deposited HOLO conformer.
    holo: Tensor
    #: ``s_p``, in Angstrom, measured by the objective's own kernel over :attr:`metric_indices`.
    separation_angstrom: float
    #: The two deposited entries, ``(apo, holo)``, as the pool spells them. Provenance, so a
    #: rendering can be checked against the crystal it was measured from without reopening the
    #: pocket artifact this reference was repackaged from.
    entry_ids: tuple[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id:
            raise ValueError("a pocket-RMSD reference entry must name its sequence")
        if not self.sequence:
            raise ValueError(f"{self.sequence_id}: the entry carries no sequence")
        indices = self.metric_indices
        if sorted(set(indices)) != list(indices):
            raise ValueError(
                f"{self.sequence_id}: metric_indices must be ascending and distinct; a repeated "
                "residue would enter the superposition twice and weight it"
            )
        if not indices:
            raise ValueError(
                f"{self.sequence_id}: no residue of the metric region is resolved in both "
                "deposited conformers, so this system has no metric set at all; it belongs out of "
                "the arm rather than in it with an empty measurement"
            )
        if indices[0] < 0 or indices[-1] >= len(self.sequence):
            raise ValueError(
                f"{self.sequence_id}: metric_indices run {indices[0]}..{indices[-1]} but the "
                f"sequence has {len(self.sequence)} residues, so at least one index addresses a "
                "residue the generator never produces"
            )
        for name, tensor in (("apo", self.apo), ("holo", self.holo)):
            expected = (len(indices), len(BACKBONE_ATOMS), 3)
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{self.sequence_id}: {name} is {tuple(tensor.shape)} and must be {expected} "
                    f"in {BACKBONE_ATOMS} order. The conformers are stored PRE-RESTRICTED to "
                    "metric_indices so the alignment set and the metric set cannot differ"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{self.sequence_id}: {name} carries a non-finite metric atom")

    @property
    def index(self) -> Tensor:
        """``metric_indices`` as a long tensor, for restricting a ``[residues, 4, 3]`` draw."""
        return torch.as_tensor(list(self.metric_indices), dtype=torch.long)

    @property
    def normalisation(self) -> PocketRmsdNormalisation:
        return PocketRmsdNormalisation(
            sequence_id=self.sequence_id, separation_angstrom=float(self.separation_angstrom)
        )

    def conformer(self, state: str) -> Tensor:
        """Return the deposited conformer one level names: ``"apo"`` or ``"holo"``."""
        if state == "apo":
            return self.apo
        if state == "holo":
            return self.holo
        raise ValueError(f"state must be 'apo' or 'holo', received {state!r}")

    def recomputed_separation(self) -> float:
        """Re-measure ``s_p`` from the stored coordinates, for the load-time cross-check."""
        with torch.no_grad():
            return float(aligned_backbone_rmsd(self.apo, self.holo).reshape(()))

    def identity(self) -> dict[str, object]:
        """The per-system block that enters the arm identity."""
        block: dict[str, object] = dict(self.normalisation.identity())
        block["metric_residues"] = len(self.metric_indices)
        block["entry_ids"] = list(self.entry_ids)
        return block


@dataclass(frozen=True)
class PocketRmsdReference:
    """Every system's conformers, plus what the file is and where it came from."""

    schema_version: int
    entries: Mapping[str, PocketRmsdEntry]
    #: Where the geometry came from: the pocket artifact's ``content_fingerprint`` and
    #: ``sequence_digest`` and the metric sidecar's ``file_sha256``, written by the builder. It is
    #: RECORDED and not CHECKED here, deliberately -- checking it would mean importing
    #: ``cryptic_steering.artifact``, which is what this file exists not to do. What is checked is
    #: the geometry itself: every ``s_p`` is re-measured from the stored coordinates on every read,
    #: and every sequence is re-checked against the generator's own string on every batch.
    provenance: Mapping[str, object]
    path: Path
    #: Which of :data:`~sae_bioemu.cryptic_rmsd.objective.METRIC_OBJECTIVE_FORMS` the arm reading
    #: this file descends. There is exactly one, and it is recorded anyway: a checkpoint that did
    #: not name the arithmetic it descended could be resumed under a second form if one is ever
    #: added.
    metric_objective: str = DEFAULT_METRIC_OBJECTIVE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metric_objective", validated_metric_objective(self.metric_objective)
        )
        if not self.entries:
            raise ValueError(f"pocket-RMSD reference {self.path} carries no entries")

    def __len__(self) -> int:
        return len(self.entries)

    def require(self, sequence_id: str) -> PocketRmsdEntry:
        entry = self.entries.get(sequence_id)
        if entry is None:
            raise KeyError(
                f"{sequence_id!r} is not in the pocket-RMSD reference {self.path}; the direct "
                "metric is a deviation from THIS protein's own deposited conformers and there is "
                "nothing to fall back to -- a draw scored against a substitute would be a finite "
                "number measured to the wrong structure"
            )
        return entry

    def require_sequences(self, sequence_ids: Sequence[str]) -> None:
        """Refuse unless every named sequence has an entry. The COVERAGE gate.

        Run over the WHOLE schedule at configuration time -- train, validation and test -- because
        a validation row journals the deviation too, so a protein missing here is a run that dies
        at its first validation round rather than at its first training batch. The uncovered set is
        reported as a POPULATION, like every other coverage gate in this project: these come in
        populations (a rebuilt structure bank, a widened split) and a message naming one of
        hundreds would be read as an accident.
        """
        missing = sorted({str(name) for name in sequence_ids} - set(self.entries))
        if missing:
            raise ValueError(
                f"pocket-RMSD reference {self.path} covers {len(self.entries)} sequences and the "
                f"schedule names {len(missing)} it does not: {missing[:10]}"
                f"{' ...' if len(missing) > 10 else ''}. Every scheduled draw, validation draws "
                "included, is measured against a deposited conformer, so an uncovered protein is a "
                "run that stops at its first round rather than a protein that is quietly skipped"
            )

    def content_fingerprint(self) -> str:
        """Digest of everything that changes what the metric measures.

        Coordinates are rounded to 1e-4 Angstrom before hashing: they are measurements with a far
        larger uncertainty than that, and a digest sensitive to the last float bit would make the
        artifact irreproducible across BLAS versions while binding nothing scientific.

        The SEQUENCES are hashed as well as the indices, because the indices are positions in them:
        two files over one pool built from structure banks that resolve one residue differently
        carry the same index lists addressing different residues.
        """
        digest = hashlib.sha256()
        digest.update(f"cryptic_rmsd.artifact.v{self.schema_version}".encode())
        digest.update(TRAINED_DEVIATION_FORM.encode())
        for sequence_id in sorted(self.entries):
            entry = self.entries[sequence_id]
            digest.update(sequence_id.encode())
            digest.update(entry.sequence.encode())
            digest.update(json.dumps(list(entry.metric_indices)).encode())
            for tensor in (entry.apo, entry.holo):
                rounded = torch.round(tensor.to(torch.float64) * 1e4) / 1e4
                digest.update(rounded.numpy().tobytes())
        return digest.hexdigest()

    def identity(self) -> dict[str, object]:
        """The block a training checkpoint binds this artifact by.

        The per-system ``s_p`` values ARE here, unlike in the sidecar's identity, and the reason is
        the target origin. Under
        :data:`~sae_bioemu.steering.schedule.TARGET_ORIGIN_SYSTEM_CONSTANT` each ``s_p`` is both
        the target and the per-observation divisor of the objective, so it is a per-protein WEIGHT
        the adapter descended rather than a measurement sitting beside one, and a report must be
        able to say which weight each protein's draws carried without reopening the file.
        """
        return {
            "schema_version": self.schema_version,
            "atom_set": list(BACKBONE_ATOMS),
            "deviation_form": TRAINED_DEVIATION_FORM,
            "graded_deviation_form": GRADED_DEVIATION_FORM,
            "metric_objective": self.metric_objective,
            "systems": len(self.entries),
            "content_fingerprint": self.content_fingerprint(),
            "normalisation": [self.entries[name].identity() for name in sorted(self.entries)],
        }


@dataclass(frozen=True)
class PocketRmsdDefinition:
    """The resolved reference one pocket-RMSD batch is measured against.

    **It lives HERE and not in** ``bioemu/steering_properties.py``, **and that is the whole point
    of the shared ``definition`` keyword.** That file is a member of ``generation_shared``, hashed
    by every steerable property there is, so a fifth definition class declared there would re-key
    every radius-of-gyration, cryptic, DFG-coordinate, domain-motion and kinase-RMSD checkpoint for
    a type none of them constructs. ``generate_exact_property_sample`` takes ONE ``definition``
    keyword typed ``Any`` and each property's own hook type-checks what it was handed.

    A DISTINCT type from ``BioEmuCrypticDefinition`` even though both are about cryptic pockets,
    and the distinction is load-bearing rather than tidy: that one wraps a resolved arm carrying a
    pair set, a transition vector and a site channel, and a mistyped dispatch between the two would
    look up this property's sequence ids in the other's tables. Naming the type is what makes the
    hook's ``isinstance`` check a real gate.

    ``_site_field`` in the shared generation path duck-types on ``definition.arm.site``. This class
    deliberately has no ``arm`` attribute, so it supplies no per-residue conditioning channel and
    the unconditioned path is taken byte-identically -- which is what
    :func:`~sae_bioemu.cryptic_rmsd.property._adapter_arguments` says in words.
    """

    reference: PocketRmsdReference


def load_pocket_rmsd_reference(
    path: str | Path, *, separation_tolerance_angstrom: float = SEPARATION_TOLERANCE_ANGSTROM
) -> PocketRmsdReference:
    """Read and validate a pocket-RMSD reference artifact.

    The stored ``s_p`` is RE-MEASURED from the stored coordinates on every read and the two must
    agree to ``separation_tolerance_angstrom``. That check is the whole reason the separation is
    stored at all rather than recomputed: storing it lets the gate, the report and the objective
    quote one number, and re-measuring it here is what stops that number drifting away from the
    coordinates it describes -- a stale ``s_p`` is a silently wrong per-protein weight, a silently
    wrong target and a silently wrong pool gate at once, because under the system-constant origin
    it is all three.
    """
    resolved = Path(path).resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        declared = payload.get("schema_version") if isinstance(payload, dict) else None
        raise ValueError(
            f"pocket-RMSD reference {resolved} declares schema version {declared!r}, not the "
            f"supported {SCHEMA_VERSION}"
        )
    if payload.get("atom_set") != list(BACKBONE_ATOMS):
        raise ValueError(
            f"pocket-RMSD reference {resolved} is built on atom set {payload.get('atom_set')!r}; "
            f"this metric is over {list(BACKBONE_ATOMS)} in that order, and a reference read in "
            "one order against a draw built in another is a per-residue permutation that leaves "
            "every deviation finite"
        )
    if payload.get("deviation_form") != TRAINED_DEVIATION_FORM:
        raise ValueError(
            f"pocket-RMSD reference {resolved} stores a {payload.get('deviation_form')!r} "
            f"separation and this objective normalises by {TRAINED_DEVIATION_FORM!r}; the two "
            "differ by 4.5-34% on this project's own geometry, so the ask on every protein would "
            "be rescaled by a ratio nothing downstream records"
        )
    raw = payload.get("entries")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"pocket-RMSD reference {resolved} carries no entries")

    entries: dict[str, PocketRmsdEntry] = {}
    for sequence_id in sorted(raw):
        record = raw[sequence_id]
        entry = PocketRmsdEntry(
            sequence_id=str(sequence_id),
            sequence=str(record["sequence"]),
            metric_indices=tuple(int(value) for value in record["metric_indices"]),
            apo=torch.as_tensor(record["apo"], dtype=torch.float64),
            holo=torch.as_tensor(record["holo"], dtype=torch.float64),
            separation_angstrom=float(record["separation_angstrom"]),
            entry_ids=(str(record["entry_ids"][0]), str(record["entry_ids"][1])),
        )
        measured = entry.recomputed_separation()
        if abs(measured - entry.separation_angstrom) > separation_tolerance_angstrom:
            raise ValueError(
                f"{sequence_id}: the artifact records s_p = {entry.separation_angstrom:.6f} "
                f"Angstrom but its own coordinates give {measured:.6f}. Under the system-constant "
                "origin s_p is this protein's TARGET, its property-loss divisor and the number its "
                "pool gate was applied to, so a stale value is wrong three times over"
            )
        entries[entry.sequence_id] = entry
    return PocketRmsdReference(
        schema_version=SCHEMA_VERSION,
        entries=entries,
        provenance=dict(payload.get("provenance") or {}),
        path=resolved,
        metric_objective=validated_metric_objective(payload.get("metric_objective")),
    )


def pocket_rmsd_metric_batch(
    reference: PocketRmsdReference,
    sequence_ids: Sequence[str],
    sequences: Sequence[str],
    levels: Sequence[float],
    backbone_angstrom: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(deviation_angstrom, priced, separation)`` for one generation batch.

    ``backbone_angstrom`` is ``[members, width, 4, 3]`` from
    :func:`~sae_bioemu.cryptic_rmsd.frames.batched_backbone_atoms`, padded to the batch width; each
    member is trimmed to its OWN sequence length and then restricted to its OWN ``metric_indices``,
    because a padding row is a frame of a residue that does not exist and comes back as an entirely
    plausible atom.

    **This is the only place the draw is restricted, and it uses the same list the stored
    conformers were restricted by at build time.** That is what makes "alignment set == metric set"
    a structural property here rather than a convention two call sites agree on.

    ``sequences`` is the string the GENERATOR was handed for each member, and it must equal the
    entry's character for character. That is the index-frame gate: ``metric_indices`` are positions
    in the entry's own string, and a pool rebuilt with one more observed residue shifts every one
    of them while every other check in this file stays green. It is checked per batch rather than
    once at resolution because the workflow resolves the arm before it reads a FASTA.

    ``levels`` SELECT A CONFORMER rather than a sign (objective design point 2), so the
    per-member reference comes from that member's own entry and its own level, and nothing about
    the batch's composition enters the measurement.

    ``separation`` is returned beside the two because it is the per-protein weight
    (``property_scale``-shaped, one value per observation) and the journal has to be able to divide
    by it to recover an Angstrom quantity from a dimensionless one.
    """
    members = tuple(str(name) for name in sequence_ids)
    if not members:
        raise ValueError("a generated batch must hold at least one member")
    if backbone_angstrom.ndim != 4 or tuple(backbone_angstrom.shape[-2:]) != (
        len(BACKBONE_ATOMS),
        3,
    ):
        raise ValueError(
            f"backbone_angstrom must be [members, width, {len(BACKBONE_ATOMS)}, 3] in "
            f"{BACKBONE_ATOMS} order; got {tuple(backbone_angstrom.shape)}"
        )
    if int(backbone_angstrom.shape[0]) != len(members):
        raise ValueError("backbone_angstrom must carry one structure per batch member")
    if len(sequences) != len(members) or len(levels) != len(members):
        raise ValueError(
            "one sequence id, one sequence and one dose level per batch member is required: the "
            "level selects the conformer and the sequence is what the metric indices are positions "
            "in"
        )

    deviations: list[Tensor] = []
    priced: list[Tensor] = []
    separations: list[float] = []
    for index, sequence_id in enumerate(members):
        entry = reference.require(sequence_id)
        sequence = str(sequences[index])
        if sequence != entry.sequence:
            raise ValueError(
                f"{sequence_id}: the generator was handed a {len(sequence)}-residue sequence and "
                f"the pocket-RMSD reference stores a {len(entry.sequence)}-residue one. The metric "
                "indices are positions in the REFERENCE's string, so a different string addresses "
                "different residues; a one-residue shift leaves every deviation finite and "
                "measured over the wrong atoms"
            )
        length = len(entry.sequence)
        if length > int(backbone_angstrom.shape[1]):
            raise ValueError(
                f"{sequence_id} has {length} residues but the batch is padded to "
                f"{int(backbone_angstrom.shape[1])}"
            )
        drawn = backbone_angstrom[index, :length]
        selected = drawn[entry.index.to(drawn.device)]
        deposited = entry.conformer(reference_state_for_level(float(levels[index]))).to(selected)
        value = aligned_backbone_rmsd(selected, deposited).reshape(())
        deviations.append(value)
        priced.append(entry.normalisation.priced(value))
        separations.append(float(entry.separation_angstrom))
    return (
        torch.stack(deviations),
        torch.stack(priced),
        torch.tensor(separations, dtype=backbone_angstrom.dtype, device=backbone_angstrom.device),
    )
