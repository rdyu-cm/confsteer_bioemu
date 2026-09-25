"""The domain-motion reference artifact: the two crystals, the metric set and ``s_p``, per system.

SCOPE: DOMAIN MOTIONS. This module decides WHICH ATOMS the direct metric is read over and WHAT
each system's normalisation is, so it is a member of the training fingerprint group: a change here
changes every distance the objective descends.

WHY THE TRAINING PATH READS AN ARTIFACT AND NOT A STRUCTURE FILE
-----------------------------------------------------------------

The metric needs the two deposited conformers' N/CA/C/O on the folded sequence's own residue frame.
Producing that from a PDB or mmCIF means a parser, a global sequence alignment and a resolution
mask -- the project's highest-risk step and a per-epoch cost for a value that cannot
change. The cryptic and kinase programmes both solved this the same way: build the geometry once,
offline, write it with a digest, and have the arm read it. This follows them.

It also keeps the training fingerprint honest. The offline builder may import whatever it likes,
including ``cryptic_steering.success_at_b``'s mmCIF reader; the TRAINING path imports a torch load
and this file's validation. If the arm parsed structures itself it would execute a cryptic module
that is in no fingerprint group at all, so a change to that reader would move no domain-motion
digest while changing every reference coordinate -- fail-open, and the exact shape D-43 fixed.

THE RESOLUTION MASK IS NOT A DETAIL
------------------------------------

``metric_residues`` is the intersection: residues the alignment placed on the sequence in BOTH
conformers AND which carry all four backbone atoms in both. It is stored rather than recomputed
because all three quantities that have to agree -- the level ``-1`` metric, the level ``+1`` metric
and the separation ``s_p`` that normalises them -- are read over it. An entry whose coordinates and
mask disagree is refused at load rather than measured over a zero row, which would be a fabricated
atom at the origin rather than a missing one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from sae_bioemu.domain_motion.frames import BACKBONE_ATOMS
from sae_bioemu.domain_motion.objective import (
    DEFAULT_METRIC_OBJECTIVE,
    DIRECT_METRIC_TAU_ANGSTROM,
    SystemNormalisation,
    aligned_backbone_rmsd,
    reference_state_for_level,
    validated_metric_objective,
)

__all__ = [
    "SCHEMA_VERSION",
    "DomainMotionArtifact",
    "DomainMotionReference",
    "direct_metric_batch",
    "load_domain_motion_artifact",
]

#: Bumped whenever the meaning of a stored field changes. A mismatched version is refused at load:
#: the alternative is reading an old file's coordinates under a new field's meaning, which produces
#: finite, plausible distances measured against the wrong thing.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DomainMotionReference:
    """One system's two deposited conformers, resolved onto the sequence BioEmu folds."""

    sequence_id: str
    sequence: str
    #: ``[residues, 4, 3]`` in Angstrom, in :data:`~sae_bioemu.domain_motion.frames.BACKBONE_ATOMS`
    #: order. Rows outside :attr:`metric_residues` are zeros and are never read.
    first: Tensor
    second: Tensor
    #: Residue indices resolved in BOTH conformers, ascending. The metric set, the alignment set
    #: and the set ``s_p`` is measured over: one set, by construction.
    metric_residues: tuple[int, ...]
    #: ``s_p``, in Angstrom, measured by the objective's own kernel over ``metric_residues``.
    separation_angstrom: float
    #: Per-conformer fraction of observed residues the alignment placed. A coverage number, carried
    #: so a system that maps at 0.72 is a reported fact rather than a rounding error.
    matched_fraction: tuple[float, float]

    def __post_init__(self) -> None:
        if len(self.sequence) != int(self.first.shape[0]):
            raise ValueError(
                f"{self.sequence_id}: the sequence has {len(self.sequence)} residues and the "
                f"reference carries {int(self.first.shape[0])}"
            )
        if self.first.shape != self.second.shape:
            raise ValueError(f"{self.sequence_id}: the two conformers have different shapes")
        if self.first.ndim != 3 or tuple(self.first.shape[1:]) != (len(BACKBONE_ATOMS), 3):
            raise ValueError(
                f"{self.sequence_id}: references must be [residues, {len(BACKBONE_ATOMS)}, 3] "
                f"in {BACKBONE_ATOMS} order"
            )
        if len(self.metric_residues) < 3:
            raise ValueError(
                f"{self.sequence_id}: {len(self.metric_residues)} residues are resolved in both "
                "conformers, which is too few to determine a superposition; this system has no "
                "metric set and belongs out of the pool"
            )
        if sorted(set(self.metric_residues)) != list(self.metric_residues):
            raise ValueError(f"{self.sequence_id}: metric_residues must be ascending and distinct")
        if not 0 <= self.metric_residues[0] and self.metric_residues[-1] < len(self.sequence):
            raise ValueError(f"{self.sequence_id}: metric_residues fall outside the sequence")
        index = self.index
        for name, tensor in (("first", self.first), ("second", self.second)):
            if not bool(torch.isfinite(tensor[index]).all()):
                raise ValueError(f"{self.sequence_id}: {name} carries non-finite metric atoms")

    @property
    def index(self) -> Tensor:
        """``metric_residues`` as a long tensor, for indexing a ``[residues, 4, 3]`` structure."""
        return torch.as_tensor(list(self.metric_residues), dtype=torch.long)

    @property
    def normalisation(self) -> SystemNormalisation:
        return SystemNormalisation(
            sequence_id=self.sequence_id,
            separation_angstrom=float(self.separation_angstrom),
            tau_angstrom=DIRECT_METRIC_TAU_ANGSTROM,
        )

    def conformer(self, state: str) -> Tensor:
        """Return the deposited conformer one level names: ``"first"`` (i) or ``"second"`` (j)."""
        if state == "first":
            return self.first
        if state == "second":
            return self.second
        raise ValueError(f"state must be 'first' or 'second', received {state!r}")

    def recomputed_separation(self) -> float:
        """Re-measure ``s_p`` from the stored coordinates, for the load-time cross-check."""
        with torch.no_grad():
            return float(aligned_backbone_rmsd(self.first, self.second, self.index).reshape(()))


@dataclass(frozen=True)
class DomainMotionArtifact:
    """Every system's references, plus what the file is and where it came from."""

    schema_version: int
    entries: Mapping[str, DomainMotionReference]
    provenance: Mapping[str, object]
    path: Path
    #: Which of :data:`~sae_bioemu.domain_motion.objective.METRIC_OBJECTIVE_FORMS` the arm that
    #: reads this artifact descends. It lives HERE rather than in the workflow's ``objective``
    #: block because the artifact IS this property's arm -- ``target.domain_motion_arm`` names this
    #: file directly -- and because the form determines three things that are built in three
    #: different places from this one object: the loss's arithmetic, the per-system target constant
    #: the schedule builds, and the prediction the generation path measures. One declaration on the
    #: object all three already hold is what stops them disagreeing; a workflow key would reach the
    #: loss and neither of the other two.
    metric_objective: str = DEFAULT_METRIC_OBJECTIVE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metric_objective", validated_metric_objective(self.metric_objective)
        )

    def __len__(self) -> int:
        return len(self.entries)

    def require(self, sequence_id: str) -> DomainMotionReference:
        entry = self.entries.get(sequence_id)
        if entry is None:
            raise KeyError(
                f"{sequence_id!r} is not in the domain-motion reference artifact {self.path}; the "
                "direct metric is a distance to THIS system's own deposited conformers and there "
                "is nothing to fall back to"
            )
        return entry

    def content_fingerprint(self) -> str:
        """Digest of everything that changes what the metric measures.

        Coordinates are rounded to 1e-4 Angstrom before hashing: they are measurements with a far
        larger uncertainty than that, and a digest sensitive to the last float bit would make the
        artifact irreproducible across BLAS versions while binding nothing scientific.
        """
        digest = hashlib.sha256()
        digest.update(f"domain_motion.artifact.v{self.schema_version}".encode())
        for sequence_id in sorted(self.entries):
            entry = self.entries[sequence_id]
            digest.update(sequence_id.encode())
            digest.update(entry.sequence.encode())
            digest.update(json.dumps(list(entry.metric_residues)).encode())
            for tensor in (entry.first, entry.second):
                rounded = torch.round(tensor.to(torch.float64) * 1e4) / 1e4
                digest.update(rounded.numpy().tobytes())
        return digest.hexdigest()

    def identity(self) -> dict[str, object]:
        """The block a training checkpoint binds this artifact by."""
        return {
            "schema_version": self.schema_version,
            "systems": len(self.entries),
            "atom_set": list(BACKBONE_ATOMS),
            # The FORM, unconditionally and including where it is the default. Two artifacts with
            # identical coordinates under the two forms are two objectives, and an identity that
            # named neither would let a checkpoint trained under one be resumed under the other.
            "metric_objective": self.metric_objective,
            "content_fingerprint": self.content_fingerprint(),
            # Per system, because ``s_p`` and ``tau_p`` are per-protein weights on the objective
            # (objective design point 3) and a report must be able to say which weight each
            # protein's draws carried without reopening the artifact.
            # Both per-system numbers, under both forms. ``separation_angstrom`` is the divisor the
            # objective is measured in under either one; ``effective_tau_angstrom`` is the
            # threshold, which the MSE form does not read -- ``metric_objective`` above is what
            # says which of them this artifact's objective actually applied.
            "normalisation": [
                self.entries[name].normalisation.identity() for name in sorted(self.entries)
            ],
        }


def load_domain_motion_artifact(
    path: str | Path, *, separation_tolerance_angstrom: float = 1e-3
) -> DomainMotionArtifact:
    """Read and validate a domain-motion reference artifact.

    The stored ``s_p`` is RE-MEASURED from the stored coordinates and the two must agree to
    ``separation_tolerance_angstrom``. That check is the whole reason the separation is stored at
    all rather than recomputed on every load: storing it lets the gate, the report and the
    objective quote one number, and re-measuring it here is what stops that number drifting away
    from the coordinates it describes -- a stale ``s_p`` is a silently wrong per-protein weight and
    a silently wrong pool gate at once.
    """
    resolved = Path(path).resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"domain-motion artifact {resolved} declares schema version "
            f"{payload.get('schema_version') if isinstance(payload, dict) else None!r}, not the "
            f"supported {SCHEMA_VERSION}"
        )
    if payload.get("atom_set") != list(BACKBONE_ATOMS):
        raise ValueError(
            f"domain-motion artifact {resolved} is built on atom set {payload.get('atom_set')!r}; "
            f"the benchmark's global RMSD is over {list(BACKBONE_ATOMS)} in that order, and a "
            "reference in another atom set is a different measurement rather than a looser one"
        )
    raw = payload.get("entries")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"domain-motion artifact {resolved} carries no entries")

    entries: dict[str, DomainMotionReference] = {}
    for sequence_id in sorted(raw):
        record = raw[sequence_id]
        entry = DomainMotionReference(
            sequence_id=str(sequence_id),
            sequence=str(record["sequence"]),
            first=torch.as_tensor(record["first"], dtype=torch.float64),
            second=torch.as_tensor(record["second"], dtype=torch.float64),
            metric_residues=tuple(int(value) for value in record["metric_residues"]),
            separation_angstrom=float(record["separation_angstrom"]),
            matched_fraction=(
                float(record["matched_fraction"][0]),
                float(record["matched_fraction"][1]),
            ),
        )
        measured = entry.recomputed_separation()
        if abs(measured - entry.separation_angstrom) > separation_tolerance_angstrom:
            raise ValueError(
                f"{sequence_id}: the artifact records s_p = {entry.separation_angstrom:.6f} "
                f"Angstrom but its own coordinates give {measured:.6f}. s_p is the per-protein "
                "weight on this system's whole objective and the number its pool gate was applied "
                "to, so a stale value is wrong twice over"
            )
        entries[entry.sequence_id] = entry
    return DomainMotionArtifact(
        schema_version=SCHEMA_VERSION,
        entries=entries,
        provenance=dict(payload.get("provenance") or {}),
        path=resolved,
        # Absent means the DEFAULT, which is plain MSE: that is what an artifact written before the
        # key existed means and what an arm naming no form means. A present-but-unrecognised
        # spelling is refused by name rather than defaulted, because a typo that fell back would
        # train the other objective under an identity naming neither.
        metric_objective=validated_metric_objective(payload.get("metric_objective")),
    )


def direct_metric_batch(
    artifact: DomainMotionArtifact,
    sequence_ids: list[str],
    levels: list[float],
    backbone_angstrom: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(metric_angstrom, priced, separation)`` for one generation batch.

    ``priced`` is the value of the artifact's OWN
    :attr:`~DomainMotionArtifact.metric_objective` -- ``0.5*(m/s_p)^2`` under the default
    :data:`~sae_bioemu.domain_motion.objective.METRIC_OBJECTIVE_MSE` and
    ``huberised_hinge((m - tau_p)/s_p)`` under the ablation -- dispatched through
    :meth:`~sae_bioemu.domain_motion.objective.SystemNormalisation.priced` so that this path and
    the loss cannot come to price two different things.

    ``backbone_angstrom`` is ``[members, width, 4, 3]`` from
    :func:`~sae_bioemu.domain_motion.frames.batched_backbone_atoms`, padded to the batch width;
    each member is trimmed to its OWN sequence length before anything is measured, because a
    padding row is a frame of a residue that does not exist and comes back as a plausible atom.

    ``levels`` are the scheduled dose levels, and each one SELECTS A REFERENCE rather than a sign
    (objective design point 2). The per-member reference and metric set therefore come from that
    member's own artifact entry and its own level, and nothing about the batch's composition
    enters the measurement.

    ``separation`` is returned beside the two because it is the per-protein weight
    (``property_scale``-shaped, one value per observation) and the journal has to be able to divide
    by it to recover an Angstrom quantity from a hinge.
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
        entry = artifact.require(sequence_id)
        length = len(entry.sequence)
        if length > int(backbone_angstrom.shape[1]):
            raise ValueError(
                f"{sequence_id} has {length} residues but the batch is padded to "
                f"{int(backbone_angstrom.shape[1])}"
            )
        state = reference_state_for_level(float(level))
        drawn = backbone_angstrom[index, :length]
        reference = entry.conformer(state).to(drawn)
        value = aligned_backbone_rmsd(drawn, reference, entry.index.to(drawn.device)).reshape(())
        metrics.append(value)
        priced.append(entry.normalisation.priced(value, artifact.metric_objective))
        separations.append(float(entry.separation_angstrom))
    return (
        torch.stack(metrics),
        torch.stack(priced),
        torch.tensor(separations, dtype=backbone_angstrom.dtype, device=backbone_angstrom.device),
    )
