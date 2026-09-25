"""The kinase-RMSD reference artifact: two deposited chains, the metric set and ``s_p``, per gene.

SCOPE: KINASES. This module decides WHICH ATOMS the direct metric is read over and WHAT each
system's normalisation is, so it is a member of this property's training fingerprint group: a
change here changes every distance the objective descends.

WHY THE TRAINING PATH READS AN ARTIFACT AND NOT A STRUCTURE FILE
-----------------------------------------------------------------

The metric needs the two deposited chains' N/CA/C/O on the folded sequence's own residue frame.
Producing that from Kincore's alignment-numbered mmCIF release means a zip reader, an mmCIF parser,
a global sequence alignment and a resolution mask -- this project's highest-risk step
and a per-epoch cost for a value that cannot change. The cryptic, DFG-coordinate and domain-motion
programmes all solved this the same way and this follows them: build the geometry once, offline,
write it with a digest, and have the arm read a ``torch.load`` plus this file's validation.

It also keeps the training fingerprint honest. The offline builder imports
``kinase_dfg/panel.py``'s mmCIF reader and Biopython's aligner; the TRAINING path imports neither.
If the arm parsed structures itself it would execute ``kinase_dfg`` code, which would either be
fail-OPEN (that module is in ``kinase_generation_shared``, which this property's composition does
NOT name, so a change to the reader would move no digest while changing every reference coordinate)
or would force this property to bind the whole DFG-coordinate group. Neither is acceptable; the
artifact is what makes the third option available.

TWO THINGS THIS ARTIFACT DOES THAT THE DOMAIN-MOTION ONE DOES NOT
-------------------------------------------------------------------

1. **THE BACKBONES ARE PRE-RESTRICTED.** ``dfg_in`` and ``dfg_out`` are ``[K, 4, 3]`` where ``K``
   is ``len(metric_indices)``, not ``[L, 4, 3]`` with zero rows outside the mask. The
   domain-motion artifact stores full-length coordinates and a separate index, so "the alignment
   set" and "the metric set" are two uses of one variable and a caller that indexed one and not
   the other would produce a finite, plausible, wrong RMSD. Here the two are the same set
   structurally: there is no selector to pass differently, and
   :func:`~sae_bioemu.kinase_rmsd.objective.aligned_backbone_rmsd` takes no ``metric_residues``
   argument at all. The DRAW is restricted by the same ``metric_indices`` at the one call site
   below.

   The kinase metric set makes that worth paying for. It is a ~15-residue activation loop inside a
   ~260-residue domain, so a mask applied to one operand and not the other would be measuring 15
   residues against 260 -- a shape mismatch here, but a silent 17x dilution under any scheme that
   padded instead.

2. **``metric_set`` IS RECORDED AND HAS NO DEFAULT.** Which residues the metric is read over was
   the decision this property was gated on, and the shell candidates dilute the activation loop to
   about 12% of the union. An artifact that did not say which set built it could be read as the
   shipped one at every other check green.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from sae_bioemu.kinase_rmsd.frames import BACKBONE_ATOMS
from sae_bioemu.kinase_rmsd.objective import (
    DEFAULT_METRIC_OBJECTIVE,
    KINASE_RMSD_TAU_ANGSTROM,
    KinaseRmsdNormalisation,
    aligned_backbone_rmsd,
    reference_state_for_level,
    validated_metric_objective,
    validated_metric_set,
)

__all__ = [
    "SCHEMA_VERSION",
    "KinaseRmsdDefinition",
    "KinaseRmsdEntry",
    "KinaseRmsdReference",
    "kinase_rmsd_metric_batch",
    "load_kinase_rmsd_reference",
]

#: Bumped whenever the meaning of a stored field changes. A mismatched version is refused at load:
#: the alternative is reading an old file's coordinates under a new field's meaning, which produces
#: finite, plausible distances measured against the wrong thing.
SCHEMA_VERSION = 1

#: How far a re-measured ``s_p`` may sit from the stored one, in Angstrom, before the load is
#: refused. Loose enough for a float64 Kabsch across BLAS versions; far tighter than any real
#: disagreement, because a stale value differs by the whole change that made it stale.
SEPARATION_TOLERANCE_ANGSTROM = 1e-3


@dataclass(frozen=True)
class KinaseRmsdEntry:
    """One kinase's two deposited chains, restricted to the residues the metric is read over."""

    sequence_id: str
    #: The HGNC gene symbol, carried for the report. It is not the key: the split, the FASTA and
    #: the schedule all speak ``<gene>_kd``, and a second spelling of the identity is how a report
    #: comes to aggregate two halves of one protein.
    gene: str
    #: The kinase-domain sequence BioEmu folds. ``metric_indices`` are positions in THIS string,
    #: and the workflow re-checks each member's string at generation time.
    sequence: str
    #: Positions in :attr:`sequence` the metric is read over, ascending and distinct. The metric
    #: set, the alignment set and the set ``s_p`` is measured over: one set, by construction.
    metric_indices: tuple[int, ...]
    #: ``[K, 4, 3]`` in Angstrom, in :data:`~sae_bioemu.kinase_rmsd.frames.BACKBONE_ATOMS` order,
    #: ALREADY RESTRICTED to :attr:`metric_indices`. The Kincore ``DFGin``/``BLAminus`` chain.
    dfg_in: Tensor
    #: The same, for the Kincore ``DFGout``/``BBAminus`` chain.
    dfg_out: Tensor
    #: ``s_p``, in Angstrom, measured by the objective's own kernel over :attr:`metric_indices`.
    separation_angstrom: float
    #: Which of :data:`~sae_bioemu.kinase_rmsd.objective.METRIC_SETS` produced
    #: :attr:`metric_indices`. No default: see the module docstring.
    metric_set: str
    #: The two deposited entries, ``(dfg_in, dfg_out)`` as ``"<PDB id><chain>"``. Provenance, so a
    #: rendering can be checked against the chain it was measured from without reopening the pool.
    entry_ids: tuple[str, str]
    #: Per-chain fraction of the A-loop span that survived the all-four-atoms-in-both-chains
    #: restriction. A coverage number, carried so a kinase whose loop is half disordered is a
    #: reported fact rather than a short index list nobody looked at.
    resolved_fraction: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_set", validated_metric_set(self.metric_set))
        if not self.sequence_id or not self.gene:
            raise ValueError("a kinase reference entry must name its sequence and its gene")
        if not self.sequence:
            raise ValueError(f"{self.sequence_id}: the entry carries no sequence")
        indices = self.metric_indices
        if sorted(set(indices)) != list(indices):
            raise ValueError(f"{self.sequence_id}: metric_indices must be ascending and distinct")
        if len(indices) < 3:
            raise ValueError(
                f"{self.sequence_id}: {len(indices)} residues are resolved in both deposited "
                "chains over the metric set, which is too few to determine a superposition; this "
                "kinase has no metric set and belongs out of the pool"
            )
        if indices[0] < 0 or indices[-1] >= len(self.sequence):
            raise ValueError(
                f"{self.sequence_id}: metric_indices fall outside the {len(self.sequence)}-residue "
                "sequence, so the draw would be restricted to residues the model never folds"
            )
        for name, tensor in (("dfg_in", self.dfg_in), ("dfg_out", self.dfg_out)):
            expected = (len(indices), len(BACKBONE_ATOMS), 3)
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{self.sequence_id}: {name} is {tuple(tensor.shape)} and must be {expected} "
                    f"in {BACKBONE_ATOMS} order. The chains are stored PRE-RESTRICTED to "
                    "metric_indices so the alignment set and the metric set cannot differ"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{self.sequence_id}: {name} carries non-finite metric atoms")

    @property
    def index(self) -> Tensor:
        """``metric_indices`` as a long tensor, for restricting a ``[residues, 4, 3]`` draw."""
        return torch.as_tensor(list(self.metric_indices), dtype=torch.long)

    @property
    def normalisation(self) -> KinaseRmsdNormalisation:
        return KinaseRmsdNormalisation(
            sequence_id=self.sequence_id,
            separation_angstrom=float(self.separation_angstrom),
            tau_angstrom=KINASE_RMSD_TAU_ANGSTROM,
        )

    def chain(self, state: str) -> Tensor:
        """Return the deposited chain one level names: ``"dfg_in"`` or ``"dfg_out"``."""
        if state == "dfg_in":
            return self.dfg_in
        if state == "dfg_out":
            return self.dfg_out
        raise ValueError(f"state must be 'dfg_in' or 'dfg_out', received {state!r}")

    def recomputed_separation(self) -> float:
        """Re-measure ``s_p`` from the stored coordinates, for the load-time cross-check."""
        with torch.no_grad():
            return float(aligned_backbone_rmsd(self.dfg_in, self.dfg_out).reshape(()))

    def identity(self) -> dict[str, object]:
        """The per-system block that enters the arm identity."""
        block: dict[str, object] = dict(self.normalisation.identity())
        block["gene"] = self.gene
        block["metric_residues"] = len(self.metric_indices)
        block["entry_ids"] = list(self.entry_ids)
        return block


@dataclass(frozen=True)
class KinaseRmsdReference:
    """Every kinase's chains, plus what the file is and where it came from."""

    schema_version: int
    entries: Mapping[str, KinaseRmsdEntry]
    #: Which of :data:`~sae_bioemu.kinase_rmsd.objective.METRIC_SETS` every entry was built under.
    #: One per FILE as well as one per entry, and the loader refuses a file whose entries disagree:
    #: a table half measured over the A-loop and half over a 12 Angstrom shell would normalise two
    #: populations against two quantities under one objective.
    metric_set: str
    provenance: Mapping[str, object]
    path: Path
    #: Which of :data:`~sae_bioemu.kinase_rmsd.objective.METRIC_OBJECTIVE_FORMS` the arm reading
    #: this file descends. It lives HERE rather than in the workflow's ``objective`` block because
    #: the artifact IS this property's arm, and because the form determines three things built in
    #: three different places from this one object: the loss's arithmetic, the per-system target
    #: constant the schedule builds, and the prediction the generation path measures.
    metric_objective: str = DEFAULT_METRIC_OBJECTIVE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metric_objective", validated_metric_objective(self.metric_objective)
        )
        object.__setattr__(self, "metric_set", validated_metric_set(self.metric_set))
        disagreeing = sorted(
            name for name, entry in self.entries.items() if entry.metric_set != self.metric_set
        )
        if disagreeing:
            raise ValueError(
                f"{len(disagreeing)} entries were built under a different metric_set than the "
                f"file declares ({self.metric_set!r}); first: {disagreeing[:5]}. One table cannot "
                "normalise half its population against an activation loop and half against a "
                "context shell under one objective"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def require(self, sequence_id: str) -> KinaseRmsdEntry:
        entry = self.entries.get(sequence_id)
        if entry is None:
            raise KeyError(
                f"{sequence_id!r} is not in the kinase-RMSD reference {self.path}; the direct "
                "metric is a distance to THIS kinase's own deposited chains and there is nothing "
                "to fall back to"
            )
        return entry

    def content_fingerprint(self) -> str:
        """Digest of everything that changes what the metric measures.

        Coordinates are rounded to 1e-4 Angstrom before hashing: they are measurements with a far
        larger uncertainty than that, and a digest sensitive to the last float bit would make the
        artifact irreproducible across BLAS versions while binding nothing scientific.

        ``metric_set`` is hashed FIRST. Two files built over the same genes under two index sets
        carry different coordinates and would digest differently anyway; hashing the name as well
        means the difference is legible rather than only detectable.
        """
        digest = hashlib.sha256()
        digest.update(f"kinase_rmsd.artifact.v{self.schema_version}".encode())
        digest.update(self.metric_set.encode())
        for sequence_id in sorted(self.entries):
            entry = self.entries[sequence_id]
            digest.update(sequence_id.encode())
            digest.update(entry.sequence.encode())
            digest.update(json.dumps(list(entry.metric_indices)).encode())
            for tensor in (entry.dfg_in, entry.dfg_out):
                rounded = torch.round(tensor.to(torch.float64) * 1e4) / 1e4
                digest.update(rounded.numpy().tobytes())
        return digest.hexdigest()

    def identity(self) -> dict[str, object]:
        """The block a training checkpoint binds this artifact by."""
        return {
            "schema_version": self.schema_version,
            "systems": len(self.entries),
            "atom_set": list(BACKBONE_ATOMS),
            # THE INDEX SET, unconditionally. It has no default anywhere in this package, and a
            # checkpoint that did not record it could be resumed against a table measured over a
            # different ~110 residues at an unchanged weight.
            "metric_set": self.metric_set,
            "metric_objective": self.metric_objective,
            "content_fingerprint": self.content_fingerprint(),
            # Per system, because ``s_p`` and ``tau_p`` are per-protein weights on the objective
            # and a report must be able to say which weight each kinase's draws carried without
            # reopening the artifact.
            "normalisation": [self.entries[name].identity() for name in sorted(self.entries)],
        }


@dataclass(frozen=True)
class KinaseRmsdDefinition:
    """The resolved reference one kinase-RMSD batch is measured against.

    **It lives HERE and not in** ``bioemu/steering_properties.py``, **and that is the whole point
    of the shared ``definition`` keyword.** That file is a member of ``generation_shared``, hashed
    by every steerable property there is, so a fourth definition class declared there would re-key
    every radius-of-gyration, cryptic, DFG-coordinate and domain-motion checkpoint for a type none
    of them constructs. ``generate_exact_property_sample`` takes ONE ``definition`` keyword typed
    ``Any`` and each property's own hook type-checks what it was handed; declaring the type in the
    property's own package is what that seam was opened for.

    A DISTINCT type from ``BioEmuDomainMotionDefinition`` even though both wrap a reference table,
    and the distinction is load-bearing rather than tidy. The two tables have overlapping surfaces
    -- both answer ``require`` and ``entries`` -- so a mistyped dispatch between them would NOT
    raise: it would look up a DynDom system id in a kinase table, miss, and raise only for the
    systems that happen to collide, or worse succeed against the wrong coordinates. Naming the type
    is what makes the hook's ``isinstance`` check a real gate.

    ``_site_field`` in the shared generation path duck-types on ``definition.arm.site``. This class
    deliberately has no ``arm`` attribute, so it supplies no per-residue conditioning channel and
    the unconditioned path is taken byte-identically -- which is what
    :func:`~sae_bioemu.kinase_rmsd.property._adapter_arguments` says in words.
    """

    reference: KinaseRmsdReference


def load_kinase_rmsd_reference(
    path: str | Path, *, separation_tolerance_angstrom: float = SEPARATION_TOLERANCE_ANGSTROM
) -> KinaseRmsdReference:
    """Read and validate a kinase-RMSD reference artifact.

    The stored ``s_p`` is RE-MEASURED from the stored coordinates on every read and the two must
    agree to ``separation_tolerance_angstrom``. That check is the whole reason the separation is
    stored at all rather than recomputed: storing it lets the gate, the report and the objective
    quote one number, and re-measuring it here is what stops that number drifting away from the
    coordinates it describes -- a stale ``s_p`` is a silently wrong per-protein weight and a
    silently wrong pool gate at once.
    """
    resolved = Path(path).resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        declared = payload.get("schema_version") if isinstance(payload, dict) else None
        raise ValueError(
            f"kinase-RMSD reference {resolved} declares schema version {declared!r}, not the "
            f"supported {SCHEMA_VERSION}"
        )
    if payload.get("atom_set") != list(BACKBONE_ATOMS):
        raise ValueError(
            f"kinase-RMSD reference {resolved} is built on atom set {payload.get('atom_set')!r}; "
            f"this metric is over {list(BACKBONE_ATOMS)} in that order, and a reference in another "
            "atom set is a different measurement rather than a looser one"
        )
    # No ``.get(..., default)`` and no fallback: the index set is this objective's largest single
    # decision, so a file that does not name it is refused rather than assumed.
    metric_set = validated_metric_set(payload.get("metric_set"))
    raw = payload.get("entries")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"kinase-RMSD reference {resolved} carries no entries")

    entries: dict[str, KinaseRmsdEntry] = {}
    for sequence_id in sorted(raw):
        record = raw[sequence_id]
        entry = KinaseRmsdEntry(
            sequence_id=str(sequence_id),
            gene=str(record["gene"]),
            sequence=str(record["sequence"]),
            metric_indices=tuple(int(value) for value in record["metric_indices"]),
            dfg_in=torch.as_tensor(record["dfg_in"], dtype=torch.float64),
            dfg_out=torch.as_tensor(record["dfg_out"], dtype=torch.float64),
            separation_angstrom=float(record["separation_angstrom"]),
            metric_set=str(record.get("metric_set", metric_set)),
            entry_ids=(str(record["entry_ids"][0]), str(record["entry_ids"][1])),
            resolved_fraction=(
                float(record["resolved_fraction"][0]),
                float(record["resolved_fraction"][1]),
            ),
        )
        measured = entry.recomputed_separation()
        if abs(measured - entry.separation_angstrom) > separation_tolerance_angstrom:
            raise ValueError(
                f"{sequence_id}: the artifact records s_p = {entry.separation_angstrom:.6f} "
                f"Angstrom but its own coordinates give {measured:.6f}. s_p is the per-protein "
                "weight on this kinase's whole objective and the number its pool gate was applied "
                "to, so a stale value is wrong twice over"
            )
        entries[entry.sequence_id] = entry
    return KinaseRmsdReference(
        schema_version=SCHEMA_VERSION,
        entries=entries,
        metric_set=metric_set,
        provenance=dict(payload.get("provenance") or {}),
        path=resolved,
        metric_objective=validated_metric_objective(payload.get("metric_objective")),
    )


def kinase_rmsd_metric_batch(
    reference: KinaseRmsdReference,
    sequence_ids: Sequence[str],
    levels: Sequence[float],
    backbone_angstrom: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(metric_angstrom, priced, separation)`` for one generation batch.

    ``backbone_angstrom`` is ``[members, width, 4, 3]`` from
    :func:`~sae_bioemu.kinase_rmsd.frames.batched_backbone_atoms`, padded to the batch width; each
    member is trimmed to its OWN sequence length and then restricted to its OWN
    ``metric_indices``, because a padding row is a frame of a residue that does not exist and comes
    back as an entirely plausible atom.

    **This is the only place the draw is restricted, and it uses the same list the stored chains
    were restricted by at build time.** That is what makes "alignment set == metric set" a
    structural property here rather than a convention two call sites agree on.

    ``levels`` SELECT A CHAIN rather than a sign (objective design point 2), so the per-member
    reference comes from that member's own entry and its own level, and nothing about the batch's
    composition enters the measurement.

    ``separation`` is returned beside the two because it is the per-protein weight
    (``property_scale``-shaped, one value per observation) and the journal has to be able to divide
    by it to recover an Angstrom quantity from a dimensionless one.
    """
    if backbone_angstrom.ndim != 4 or tuple(backbone_angstrom.shape[-2:]) != (
        len(BACKBONE_ATOMS),
        3,
    ):
        raise ValueError(
            f"backbone_angstrom must be [members, width, {len(BACKBONE_ATOMS)}, 3] in "
            f"{BACKBONE_ATOMS} order"
        )
    members = int(backbone_angstrom.shape[0])
    if len(sequence_ids) != members or len(levels) != members:
        raise ValueError("one sequence id and one level per batch member is required")

    metrics: list[Tensor] = []
    priced: list[Tensor] = []
    separations: list[float] = []
    for index, (sequence_id, level) in enumerate(zip(sequence_ids, levels, strict=True)):
        entry = reference.require(sequence_id)
        length = len(entry.sequence)
        if length > int(backbone_angstrom.shape[1]):
            raise ValueError(
                f"{sequence_id} has {length} residues but the batch is padded to "
                f"{int(backbone_angstrom.shape[1])}"
            )
        drawn = backbone_angstrom[index, :length]
        selected = drawn[entry.index.to(drawn.device)]
        chain = entry.chain(reference_state_for_level(float(level))).to(selected)
        value = aligned_backbone_rmsd(selected, chain).reshape(())
        metrics.append(value)
        priced.append(entry.normalisation.priced(value, reference.metric_objective))
        separations.append(float(entry.separation_angstrom))
    return (
        torch.stack(metrics),
        torch.stack(priced),
        torch.tensor(separations, dtype=backbone_angstrom.dtype, device=backbone_angstrom.device),
    )
