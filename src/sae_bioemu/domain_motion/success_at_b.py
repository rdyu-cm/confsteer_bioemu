"""success@B on BioEmu's ``multiconf_domainmotion`` benchmark: the held-out scorer (D-89).

SCOPE: DOMAIN MOTIONS. **EVALUATION-ONLY.** Nothing here is read by the training loop, no loss is
computed, and no value produced by this module reaches a checkpoint identity. That placement is the
whole reason it is a separate file from :mod:`sae_bioemu.domain_motion.objective`: a change to a
scorer must be able to re-key an evaluation and must never retire a trained adapter.

THE METRIC, settled from upstream source in ``docs/domain_motion/dm_success_metric.txt`` and
reproduced here exactly:

    TRUE RMSD over N, CA, C, O, on the GLOBAL path, at tau = 3.0 Angstrom.

``bioemu_benchmarks/eval/multiconf/evaluate.py`` computes ``MetricType.RMSD`` two different ways
and which one runs is decided by whether a benchmark slice ships a ``local_residinfo`` directory:

    calc_metrics_global:  mdtraj.rmsd(...) * NM2ANGS       <- a TRUE QCP RMSD over N, CA, C, O
    calc_metrics_local:   np.mean(pairwise_distances, -1)  <- the arithmetic MEAN of per-atom
                                                              deviations, which is NOT an RMS

``domainmotion/`` ships ``reference``, ``references.csv`` and ``testcases.csv`` and NO
``local_residinfo``, so it takes the GLOBAL path. That is a MEASURED absence, listed from the
upstream tree and independently recorded by :mod:`sae_bioemu.domain_motion.benchmark`. A mean
deviation is never larger than its RMS, so reporting a domain-motion number in the cryptic slice's
convention would silently loosen the 3.0. **Only the true RMSD is emitted here**; the
``mean_deviation`` form that :mod:`sae_bioemu.cryptic_steering.success_at_b` must also emit is
deliberately absent, because on this benchmark it corresponds to no upstream computation.

THE SUPERPOSITION IS THE METRIC'S OWN. ``mdtraj.rmsd`` SELF-SUPERPOSES over exactly the atoms it
measures -- there is no separate alignment region, because there is no ``alignment_resid_ranges``
file to define one. :func:`superposed_rmsd` therefore does a Kabsch superposition over the metric
atoms themselves and then the root-mean-square, which is what QCP computes;
``tests/test_domain_motion_success_at_b.py`` pins it numerically against ``mdtraj.rmsd`` rather
than asserting the equivalence in prose.

success@B IS A MEAN OVER SYSTEMS. A system succeeds when the best of its first ``B`` draws is
within ``tau``; the rate is the unweighted fraction of systems that succeed. Draws are NEVER
pooled: 32 draws of one sequence are 32 draws of one protein, and pooling them would weight a
system by how many draws it happens to have and would treat repeated draws as independent
proteins. The bootstrap resamples SYSTEMS for the same reason.

THE RESIDUE FRAME IS THE SILENT FAILURE, here as everywhere in this project. The benchmark's
reference PDBs are renumbered into the test case's own sequence frame, mostly cleanly and not
always: ``P0A8W0`` carries a constant offset and two systems carry an indel no single offset
absorbs. So the frame is not assumed. Both the deposited conformers and the generated draws are
placed on the test-case sequence by the SAME letter-matched global alignment the rest of this
project uses (:data:`ALIGNER_SCORES`, copied by value from
:data:`sae_bioemu.domain_motion.benchmark.ALIGNER_SCORES` and pinned by a test), and a position
whose letters disagree is DROPPED rather than guessed. Every count is reported:
:class:`SelectionCoverage` carries what was requested, what each side resolved and what was used,
and a system below :data:`DEFAULT_COVERAGE_FLOOR` is refused by name rather than scored on a
silently narrowed selection.

WHY THIS FILE DOES ITS OWN PDB PARSE. :mod:`sae_bioemu.domain_motion.benchmark` reads the same
files and keeps C-alpha and C-beta only -- the atoms the pool gate and the DynDom decomposition
need -- so it cannot supply N/C/O. Rather than widen a module that every domain-motion TRAINING
run imports, the four-atom read lives here, in the evaluation group, beside the metric it decides
the atom set for. That is the same argument :func:`sae_bioemu.cryptic_steering.success_at_b.
read_reference_chain` makes for repeating ``structures.read_chain``.
"""

from __future__ import annotations

import csv
import json
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "ALIGNER_SCORES",
    "BACKBONE_ATOMS",
    "DEFAULT_COVERAGE_FLOOR",
    "DEFAULT_TAU_ANGSTROM",
    "BackboneChain",
    "BackboneResidue",
    "DomainMotionSuccess",
    "ReferenceSeparation",
    "SelectionCoverage",
    "SystemReference",
    "SystemScore",
    "bootstrap_interval",
    "load_system_references",
    "read_generated_backbone",
    "read_reference_backbone",
    "reference_separation",
    "require_held_out",
    "score_draw",
    "sequence_frame",
    "success_at_b",
    "superposed_rmsd",
    "take_b",
]

#: The four atoms the GLOBAL path measures over. ``bioemu-benchmarks`` filters both reference and
#: sample to mdtraj's ``backbone`` selection, which is exactly these, and its own comment pins the
#: per-residue order as ``[N, CA, C, O]``. BioEmu's draws carry a fifth atom, the C-beta placed from
#: the residue frame; it is excluded by that same filter upstream and is excluded here.
BACKBONE_ATOMS: tuple[str, ...] = ("N", "CA", "C", "O")

#: The threshold of record for this benchmark slice, from ``docs/domain_motion/dm_success_metric.txt``
#: and from ConforNets' ``assets/config.yaml`` (``domainmotion: rmsd_threshold: 3.0``). It is a
#: DEFAULT and is written into every result beside the rate, because tau dominates the number.
DEFAULT_TAU_ANGSTROM = 3.0

#: The fraction of a conformer's own observed residues that must resolve onto BOTH sides before a
#: system is scored. Crystals do not observe every residue of the construct, so demanding 100%
#: would refuse nearly every system; below this the surviving selection is a different region from
#: the one the benchmark defines and the number would not be comparable.
DEFAULT_COVERAGE_FLOOR = 0.9

#: Superposition needs at least this many atoms to be anything other than arbitrary.
MINIMUM_SUPERPOSITION_ATOMS = 9

#: The global-alignment scores, copied BY VALUE from
#: :data:`sae_bioemu.domain_motion.benchmark.ALIGNER_SCORES` and pinned by a test, so the frame this
#: scorer reads the metric in is derived by the same rule the training pool's frame is. Copied
#: rather than imported for the reason this module does its own PDB parse: ``benchmark.py`` is
#: imported by every domain-motion training run and this file is evaluation-only.
ALIGNER_SCORES = {
    "open_gap_score": -11.0,
    "extend_gap_score": -1.0,
    "match_score": 2.0,
    "mismatch_score": -1.0,
}

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip


class DomainMotionScoringError(ValueError):
    """A refusal that names the system it concerns."""

    def __init__(self, system: str, message: str) -> None:
        self.system = system
        super().__init__(f"{system}: {message}")


# --------------------------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BackboneResidue:
    """One residue's N/CA/C/O, in whatever numbering its own file uses."""

    #: ``auth_seq_id`` plus stripped insertion code for a deposited chain; the 0-based residue
    #: index for a generated draw.
    key: str
    letter: str
    atoms: dict[str, np.ndarray]

    def backbone(self, atom_names: Sequence[str] = BACKBONE_ATOMS) -> np.ndarray | None:
        """Return ``(len(atom_names), 3)``, or ``None`` when any requested atom is absent."""
        if any(name not in self.atoms for name in atom_names):
            return None
        return np.stack([self.atoms[name] for name in atom_names])


@dataclass(frozen=True)
class BackboneChain:
    """An ordered residue list plus the one-letter sequence it spells."""

    residues: tuple[BackboneResidue, ...]
    source: str

    @property
    def letters(self) -> str:
        return "".join(residue.letter for residue in self.residues)

    def __len__(self) -> int:
        return len(self.residues)


_ATOM_RECORD = re.compile(r"^ATOM  ")


def _parse_pdb_atoms(text: str, *, source: str, first_model_only: bool = True) -> BackboneChain:
    """Read ``ATOM`` records into residues, keeping :data:`BACKBONE_ATOMS`.

    Only the FIRST model is read. The benchmark's reference files are single-model but carry an
    explicit ``MODEL``/``ENDMDL`` wrapper from MDTraj, and a multi-model file silently concatenated
    would give two residues one key.
    """
    residues: list[BackboneResidue] = []
    atoms: dict[str, np.ndarray] = {}
    current: tuple[str, str, str] | None = None
    wanted = set(BACKBONE_ATOMS)

    def flush() -> None:
        if current is None:
            return
        letter = _THREE_TO_ONE.get(current[1], "")
        if len(letter) == 1:
            residues.append(BackboneResidue(key=current[0], letter=letter, atoms=dict(atoms)))

    for line in text.splitlines():
        if first_model_only and line.startswith("ENDMDL"):
            break
        if not _ATOM_RECORD.match(line):
            continue
        name = line[12:16].strip()
        resname = line[17:20].strip().upper()
        key = f"{line[22:26].strip()}{line[26].strip()}"
        chain = line[21]
        marker = (key, resname, chain)
        if current is None or marker != current:
            flush()
            atoms = {}
            current = marker
        if name in wanted:
            atoms[name] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float
            )
    flush()
    return BackboneChain(tuple(residues), source)


def read_reference_backbone(path: Path | str) -> BackboneChain:
    """Read one benchmark reference PDB, keeping N/CA/C/O.

    The benchmark's 44 reference files each hold exactly ONE chain, rewritten to ``A`` by MDTraj
    while the filename still names the deposited chain (``6KML_B``). That was verified on all 44 by
    :func:`sae_bioemu.domain_motion.benchmark.read_pdb_chain`, whose docstring records it, so this
    reader takes the single chain and refuses a second rather than guessing which one is meant.
    """
    path = Path(path)
    chain = _parse_pdb_atoms(path.read_text(), source=path.name)
    chains = {line[21] for line in path.read_text().splitlines() if _ATOM_RECORD.match(line)}
    if len(chains) > 1:
        raise ValueError(
            f"{path} holds {len(chains)} chains; the benchmark's reference files hold exactly one, "
            "and a multi-chain file would give two residues one key"
        )
    return chain


def read_generated_backbone(path: Path | str, sequence: str | None = None) -> BackboneChain:
    """Read a BioEmu final-structure PDB: one chain, residues numbered ``0..L-1``.

    BioEmu writes the backbone plus the C-beta it places from the residue frame, so N/CA/C/O are
    all present and no C-alpha-only fallback is ever needed or taken. When ``sequence`` is given,
    the residue letters must spell it and the numbering must be the plain ``0..L-1`` index, which
    ties the file on disk to the sequence recorded for it in the sample index rather than trusting
    the file name.
    """
    path = Path(path)
    chain = _parse_pdb_atoms(path.read_text(), source=path.name)
    if sequence is not None:
        if chain.letters != sequence:
            raise ValueError(
                f"{path.name}: residue letters do not spell the indexed sequence "
                f"({len(chain)} residues against {len(sequence)})"
            )
        if [residue.key for residue in chain.residues] != [
            str(index) for index in range(len(sequence))
        ]:
            raise ValueError(f"{path.name}: residues are not numbered 0..L-1 as BioEmu writes them")
    return chain


# --------------------------------------------------------------------------------------------
# The residue frame
# --------------------------------------------------------------------------------------------


def _letter_matched_alignment(left: str, right: str) -> list[tuple[int, int]]:
    """Global pairwise alignment at :data:`ALIGNER_SCORES`, keeping only letter-matching pairs."""
    from Bio.Align import PairwiseAligner

    if not left or not right:
        return []
    aligner = PairwiseAligner()
    aligner.mode = "global"
    for name, value in ALIGNER_SCORES.items():
        setattr(aligner, name, value)
    alignment = aligner.align(left, right)[0]
    pairs: list[tuple[int, int]] = []
    for (left_start, left_end), (right_start, _) in zip(*alignment.aligned, strict=False):
        for offset in range(left_end - left_start):
            if left[left_start + offset] == right[right_start + offset]:
                pairs.append((left_start + offset, right_start + offset))
    return pairs


def sequence_frame(chain: BackboneChain, sequence: str) -> dict[int, int]:
    """Return ``{0-based sequence index: index into this chain's residue list}``.

    A position the alignment declined to place, or placed on a differing letter, is ABSENT rather
    than guessed. That is what makes the coverage a coverage number instead of a confidence.
    """
    return {
        sequence_index: chain_index
        for chain_index, sequence_index in _letter_matched_alignment(chain.letters, sequence)
    }


@dataclass(frozen=True)
class SelectionCoverage:
    """How much of the sequence survived onto both structures, and why not more."""

    sequence_length: int
    on_reference: int
    on_sample: int
    used: int
    #: Positions lost because one side did not place the residue, or placed it without a complete
    #: N/CA/C/O. Counted, never hidden.
    missing_on_reference: int
    missing_on_sample: int
    incomplete_backbone: int

    @property
    def fraction(self) -> float:
        """``used`` as a fraction of what the REFERENCE conformer itself observes.

        The denominator is deliberately not the whole construct. A crystal that resolves 300 of a
        640-residue chain is not 47% covered in any sense the metric cares about -- every residue
        it does resolve is in the selection -- so the floor below would refuse nearly every system
        if it were read that way.
        """
        return self.used / self.on_reference if self.on_reference else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "sequence_length": self.sequence_length,
            "on_reference": self.on_reference,
            "on_sample": self.on_sample,
            "used": self.used,
            "missing_on_reference": self.missing_on_reference,
            "missing_on_sample": self.missing_on_sample,
            "incomplete_backbone": self.incomplete_backbone,
            "fraction": self.fraction,
        }


@dataclass(frozen=True)
class Selection:
    """Matched N/CA/C/O rows from two structures, in the test-case sequence frame."""

    reference_rows: np.ndarray
    sample_rows: np.ndarray
    residues: int
    coverage: SelectionCoverage


def build_selection(
    sequence: str,
    reference: BackboneChain,
    reference_frame: Mapping[int, int],
    sample: BackboneChain,
    sample_frame: Mapping[int, int],
) -> Selection:
    """Gather every sequence position BOTH structures place with a complete backbone.

    This is the GLOBAL path: there is no metric region and no alignment region, because the
    benchmark slice ships neither. Correspondence runs through the test-case sequence index rather
    than through either structure's own numbering, so a deposited chain renumbered away from the
    frame and a draw numbered from zero meet at the same place. Both frames are letter-matched, so
    a used position carries the same residue letter on both sides by construction; that is
    asserted rather than assumed.
    """
    reference_rows: list[np.ndarray] = []
    sample_rows: list[np.ndarray] = []
    missing_on_reference = 0
    missing_on_sample = 0
    incomplete = 0
    for position in range(len(sequence)):
        reference_index = reference_frame.get(position)
        sample_index = sample_frame.get(position)
        if reference_index is None:
            missing_on_reference += 1
        if sample_index is None:
            missing_on_sample += 1
        if reference_index is None or sample_index is None:
            continue
        reference_residue = reference.residues[reference_index]
        sample_residue = sample.residues[sample_index]
        if reference_residue.letter != sample_residue.letter:
            raise AssertionError(
                f"sequence position {position} carries {reference_residue.letter!r} on the "
                f"reference and {sample_residue.letter!r} on the sample"
            )
        left = reference_residue.backbone()
        right = sample_residue.backbone()
        if left is None or right is None:
            incomplete += 1
            continue
        reference_rows.append(left)
        sample_rows.append(right)

    used = len(reference_rows)
    empty = np.zeros((0, 3), dtype=float)
    return Selection(
        reference_rows=np.concatenate(reference_rows) if used else empty,
        sample_rows=np.concatenate(sample_rows) if used else empty,
        residues=used,
        coverage=SelectionCoverage(
            sequence_length=len(sequence),
            on_reference=len(reference_frame),
            on_sample=len(sample_frame),
            used=used,
            missing_on_reference=missing_on_reference,
            missing_on_sample=missing_on_sample,
            incomplete_backbone=incomplete,
        ),
    )


# --------------------------------------------------------------------------------------------
# The metric
# --------------------------------------------------------------------------------------------


def superposed_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """Return the TRUE RMSD after optimal superposition over the SAME atoms it measures.

    This is what ``mdtraj.rmsd`` computes on the global path: QCP finds the rotation minimising the
    root-mean-square deviation over the selected atoms and returns that minimum. Kabsch finds the
    same rotation -- both solve one orthogonal Procrustes problem -- and
    ``tests/test_domain_motion_success_at_b.py`` pins this function numerically against
    ``mdtraj.rmsd`` rather than leaving the equivalence as a claim.

    NOT a mean of per-atom distances. That is ``calc_metrics_local``, which the benchmark labels
    ``rmsd`` on the slices that ship a ``local_residinfo`` directory and which this slice does not
    take. A mean is never larger than its RMS, so the two are not interchangeable at a fixed tau.
    """
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("mobile and target must be matching (n, 3) arrays")
    if mobile.shape[0] < MINIMUM_SUPERPOSITION_ATOMS:
        raise ValueError(f"superposition needs at least {MINIMUM_SUPERPOSITION_ATOMS} atoms")
    centred_mobile = mobile - mobile.mean(axis=0)
    centred_target = target - target.mean(axis=0)
    covariance = centred_mobile.T @ centred_target
    left, _, right = np.linalg.svd(covariance)
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(right.T @ left.T)))])
    rotation = right.T @ correction @ left.T
    moved = centred_mobile @ rotation.T
    return float(np.sqrt(np.mean(np.sum((moved - centred_target) ** 2, axis=1))))


# --------------------------------------------------------------------------------------------
# Benchmark systems
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemReference:
    """One ``multiconf_domainmotion`` test case: its sequence and its two deposited conformers."""

    test_case: str
    sequence: str
    conformer_ids: tuple[str, str]
    chains: tuple[BackboneChain, BackboneChain]
    frames: tuple[dict[int, int], dict[int, int]]
    #: How many ``testcases.csv`` rows carried this accession. Greater than one means the benchmark
    #: ships two constructs and the one both conformers map onto best was chosen; retained so that
    #: choice is visible.
    candidate_sequences: int

    @property
    def matched_fraction(self) -> tuple[float, float]:
        return tuple(  # type: ignore[return-value]
            len(frame) / len(chain) if len(chain) else 0.0
            for chain, frame in zip(self.chains, self.frames, strict=True)
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _reference_path(root: Path, test_case: str, pdb_id_chain: str) -> Path:
    directory = root / "reference" / test_case
    exact = directory / f"{pdb_id_chain}.pdb"
    if exact.is_file():
        return exact
    wanted = f"{pdb_id_chain}.pdb".lower()
    for candidate in sorted(directory.glob("*.pdb")):
        if candidate.name.lower() == wanted:
            return candidate
    raise FileNotFoundError(f"{test_case}: no reference PDB for {pdb_id_chain} under {directory}")


def load_system_references(asset_dir: Path | str) -> list[SystemReference]:
    """Load every benchmark system with its N/CA/C/O conformers, placed on its own sequence.

    Where an accession ships two constructs, the one both conformers align to best is chosen by the
    MINIMUM of the two matched fractions, so a sequence that suits one conformer and not the other
    cannot win -- byte-for-byte the rule
    :func:`sae_bioemu.domain_motion.benchmark.load_benchmark_systems` uses, so the sequence this
    scorer measures against is the sequence the pool was indexed on.
    """
    root = Path(asset_dir)
    sequences: dict[str, list[str]] = {}
    for row in _read_csv(root / "testcases.csv"):
        sequences.setdefault(row["test_case"], []).append(row["sequence"])

    systems: list[SystemReference] = []
    for row in _read_csv(root / "references.csv"):
        name = row["test_case"]
        ids = (row["pdbidchain_i"], row["pdbidchain_j"])
        chains = tuple(read_reference_backbone(_reference_path(root, name, item)) for item in ids)
        if not all(len(chain) for chain in chains):
            raise DomainMotionScoringError(name, "a reference chain is absent or unreadable")
        candidates = sequences.get(name)
        if not candidates:
            raise DomainMotionScoringError(name, "named in references.csv, absent from testcases")

        best: tuple[float, int] | None = None
        chosen: tuple[str, tuple[dict[int, int], dict[int, int]]] | None = None
        for index, sequence in enumerate(candidates):
            frames = tuple(sequence_frame(chain, sequence) for chain in chains)
            score = (
                min(len(frame) / len(chain) for chain, frame in zip(chains, frames, strict=True)),
                -index,
            )
            if best is None or score > best:
                best = score
                chosen = (sequence, frames)  # type: ignore[assignment]
        assert chosen is not None
        sequence, frames = chosen
        systems.append(
            SystemReference(
                test_case=name,
                sequence=sequence,
                conformer_ids=ids,
                chains=chains,  # type: ignore[arg-type]
                frames=frames,
                candidate_sequences=len(candidates),
            )
        )
    return sorted(systems, key=lambda system: system.test_case)


@dataclass(frozen=True)
class ReferenceSeparation:
    """The experimental conformer-to-conformer RMSD: the scale the metric is read against.

    Measured under EXACTLY the sample protocol -- same atoms, same self-superposition, same frame
    -- so that a draw's RMSD and this number are the same measurement. It is also the source of
    ConforMix's adaptive threshold ``tau_i = 0.5 * s_p``, which
    ``docs/domain_motion/dm_success_metric.txt`` requires beside the fixed tau: a fixed 3.0 on a
    15 Angstrom hinge means "get a fifth of the way" and on a 3.5 Angstrom one means "reproduce the
    other crystal", and only the adaptive threshold is comparable across the two.
    """

    system: str
    rmsd_angstrom: float
    residues: int
    coverage: SelectionCoverage

    @property
    def adaptive_tau(self) -> float:
        return 0.5 * self.rmsd_angstrom


def reference_separation(
    system: SystemReference, coverage_floor: float = DEFAULT_COVERAGE_FLOOR
) -> ReferenceSeparation:
    """Measure conformer i against conformer j, under the sample protocol."""
    selection = build_selection(
        system.sequence, system.chains[0], system.frames[0], system.chains[1], system.frames[1]
    )
    if selection.coverage.fraction < coverage_floor:
        raise DomainMotionScoringError(
            system.test_case,
            f"the two conformers share {selection.coverage.used} of "
            f"{selection.coverage.on_reference} placed residues "
            f"({selection.coverage.fraction:.3f} < {coverage_floor:.3f})",
        )
    return ReferenceSeparation(
        system=system.test_case,
        rmsd_angstrom=superposed_rmsd(selection.sample_rows, selection.reference_rows),
        residues=selection.residues,
        coverage=selection.coverage,
    )


# --------------------------------------------------------------------------------------------
# Scoring draws
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DrawDeviation:
    """One generated draw measured against ONE deposited conformer."""

    system: str
    conformer: str
    conformer_index: int
    sample_id: str
    sample_index: int
    rmsd_angstrom: float
    residues: int
    atoms: int


def score_draw(
    system: SystemReference,
    conformer_index: int,
    sample: BackboneChain,
    sample_id: str,
    sample_index: int,
    coverage_floor: float = DEFAULT_COVERAGE_FLOOR,
) -> tuple[DrawDeviation, SelectionCoverage]:
    """Measure one draw against one conformer, refusing rather than truncating the selection."""
    if conformer_index not in (0, 1):
        raise ValueError("conformer_index must be 0 or 1")
    reference = system.chains[conformer_index]
    selection = build_selection(
        system.sequence,
        reference,
        system.frames[conformer_index],
        sample,
        sequence_frame(sample, system.sequence),
    )
    if selection.coverage.fraction < coverage_floor:
        raise DomainMotionScoringError(
            system.test_case,
            f"conformer {system.conformer_ids[conformer_index]} resolved "
            f"{selection.coverage.used} of {selection.coverage.on_reference} placed residues "
            f"({selection.coverage.fraction:.3f} < {coverage_floor:.3f}); missing on sample "
            f"{selection.coverage.missing_on_sample}, incomplete backbone "
            f"{selection.coverage.incomplete_backbone}",
        )
    return (
        DrawDeviation(
            system=system.test_case,
            conformer=system.conformer_ids[conformer_index],
            conformer_index=conformer_index,
            sample_id=sample_id,
            sample_index=sample_index,
            rmsd_angstrom=superposed_rmsd(selection.sample_rows, selection.reference_rows),
            residues=selection.residues,
            atoms=int(selection.reference_rows.shape[0]),
        ),
        selection.coverage,
    )


@dataclass
class SystemScore:
    """Everything measured for one system, before any tau or B is applied."""

    system: str
    deviations: dict[int, list[DrawDeviation]] = field(default_factory=dict)
    separation: ReferenceSeparation | None = None

    def draws(self, conformer_index: int) -> list[DrawDeviation]:
        return self.deviations.get(conformer_index, [])


def take_b(draws: Sequence[DrawDeviation], b: int | None) -> list[DrawDeviation]:
    """Return the first ``B`` draws in ascending sample-index order.

    Deterministic and independent of directory listing order, because ``B`` is part of the metric:
    "the best of 32" and "the best of 100" are different questions, and WHICH 32 must not depend on
    which file the filesystem returned first.
    """
    ordered = sorted(draws, key=lambda item: (item.sample_index, item.sample_id))
    return ordered if b is None else ordered[:b]


def bootstrap_interval(
    successes: Sequence[bool],
    *,
    resamples: int = 10000,
    seed: int = 20260914,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap over SYSTEMS, which are the independent units here.

    One system is one protein. Resampling draws within a system would treat 32 draws of one
    sequence as 32 proteins and shrink the interval by a factor that is pure leakage.
    """
    flags = np.asarray(list(successes), dtype=float)
    if flags.size == 0:
        return (float("nan"), float("nan"))
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, flags.size, size=(resamples, flags.size))
    rates = flags[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return (float(np.quantile(rates, tail)), float(np.quantile(rates, 1.0 - tail)))


@dataclass(frozen=True)
class DomainMotionSuccess:
    """One ``(conformer, tau, B)`` cell of the reported table."""

    #: The deposition-order conformer this cell scored, or ``None`` when the conformer was chosen
    #: PER SYSTEM (see ``conformer_choice``), which no single index can describe.
    conformer_index: int | None
    tau_angstrom: float
    b_requested: int | None
    systems: tuple[str, ...]
    successes: tuple[bool, ...]
    best_rmsd: tuple[float, ...]
    draws_per_system: tuple[int, ...]
    confidence_low: float
    confidence_high: float
    bootstrap_resamples: int
    bootstrap_seed: int
    #: ``True`` when ``tau`` was ConforMix's per-system ``0.5 * s_p`` rather than a fixed number,
    #: in which case ``tau_angstrom`` is NaN because there is no single threshold.
    adaptive: bool = False
    #: What ``conformer_index`` meant, named in the output so a reader cannot mistake a canonical
    #: cell for a deposition-order one. ``"deposition"`` is conformer 0 or 1 of the deposited pair
    #: for every system; ``"canonical"`` is each system's own choice.
    conformer_choice: str = "deposition"
    #: ``system -> deposition index`` actually scored, present only for a canonical cell.
    conformer_per_system: tuple[tuple[str, int], ...] = ()
    #: What this cell scored against, as one name a caller can select on: ``deposition_0``,
    #: ``deposition_1``, ``canonical_nearer`` or ``canonical_further``. A single-direction arm has
    #: exactly one right answer here, and picking it by position in the cell list is how the
    #: deposited order gets read as the canonical one.
    reference_state: str = ""

    @property
    def n_systems(self) -> int:
        return len(self.systems)

    @property
    def rate(self) -> float:
        return float(np.mean(self.successes)) if self.successes else float("nan")

    def as_dict(self) -> dict[str, object]:
        return {
            "conformer_index": self.conformer_index,
            "conformer_choice": self.conformer_choice,
            "reference_state": self.reference_state
            or (
                f"deposition_{self.conformer_index}"
                if self.conformer_index is not None
                else "canonical"
            ),
            "conformer_per_system": dict(self.conformer_per_system),
            "tau_angstrom": self.tau_angstrom,
            "adaptive_tau": self.adaptive,
            "b_requested": self.b_requested,
            "b_effective_min": min(self.draws_per_system) if self.draws_per_system else 0,
            "b_effective_max": max(self.draws_per_system) if self.draws_per_system else 0,
            "n_systems": self.n_systems,
            "n_successes": int(sum(self.successes)),
            "success_rate": self.rate,
            "ci_low": self.confidence_low,
            "ci_high": self.confidence_high,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "systems": list(self.systems),
            "successes": [bool(value) for value in self.successes],
            "best_rmsd_angstrom": list(self.best_rmsd),
        }


def success_at_b(
    scores: Iterable[SystemScore],
    *,
    conformer_index: int | Mapping[str, int],
    reference_state: str = "",
    b: int | None,
    tau: float | None = DEFAULT_TAU_ANGSTROM,
    resamples: int = 10000,
    seed: int = 20260914,
    confidence: float = 0.95,
) -> DomainMotionSuccess:
    """A system succeeds when the best of its first ``B`` draws is within ``tau``.

    ``tau=None`` selects ConforMix's ADAPTIVE threshold, ``0.5 * s_p`` per system, which requires
    each score to carry its :attr:`SystemScore.separation`.

    ``conformer_index`` is either a DEPOSITION index used for every system, or a mapping naming one
    per system. The mapping exists because the deposited order is arbitrary: which member of a pair
    is "conformer 0" says nothing about which one an arm was steered toward. The single-direction
    arms were trained on the CANONICAL order -- level -1 toward whichever conformer the unsteered
    ensemble already sits nearer -- and that is conformer 1 in 17 of the 22 benchmark systems. A
    fixed index therefore scores most systems against the conformer the arm was NOT aimed at, and
    still reports a plausible rate.

    The rate is the UNWEIGHTED mean over systems. Draws are never pooled: a system with 64 draws
    and one with 8 count once each, which is the weighting ``success@B`` is defined with and the
    weighting the training objective's per-system ``s_p`` normalisation is chosen to match.
    """
    if b is not None and b < 1:
        raise ValueError("B must be at least 1")
    systems: list[str] = []
    successes: list[bool] = []
    best: list[float] = []
    counts: list[int] = []
    chosen: dict[str, int] = {}
    for score in sorted(scores, key=lambda item: item.system):
        if isinstance(conformer_index, int):
            index = conformer_index
        elif score.system in conformer_index:
            index = int(conformer_index[score.system])
        else:
            raise DomainMotionScoringError(
                score.system,
                "no conformer is named for this system in the per-system mapping; refusing rather "
                "than falling back to the deposited order, which is arbitrary",
            )
        taken = take_b(score.draws(index), b)
        if not taken:
            continue
        if tau is None:
            if score.separation is None:
                raise DomainMotionScoringError(
                    score.system,
                    "the adaptive threshold is 0.5 * the reference separation and this system "
                    "carries none; measure it with reference_separation() or pass a fixed tau",
                )
            threshold = score.separation.adaptive_tau
        else:
            threshold = float(tau)
        closest = min(draw.rmsd_angstrom for draw in taken)
        systems.append(score.system)
        chosen[score.system] = index
        counts.append(len(taken))
        best.append(closest)
        successes.append(closest <= threshold)
    low, high = bootstrap_interval(successes, resamples=resamples, seed=seed, confidence=confidence)
    return DomainMotionSuccess(
        conformer_index=conformer_index if isinstance(conformer_index, int) else None,
        conformer_choice="deposition" if isinstance(conformer_index, int) else "canonical",
        reference_state=reference_state,
        conformer_per_system=()
        if isinstance(conformer_index, int)
        else tuple(sorted(chosen.items())),
        tau_angstrom=float("nan") if tau is None else float(tau),
        b_requested=b,
        systems=tuple(systems),
        successes=tuple(successes),
        best_rmsd=tuple(best),
        draws_per_system=tuple(counts),
        confidence_low=low,
        confidence_high=high,
        bootstrap_resamples=resamples,
        bootstrap_seed=seed,
        adaptive=tau is None,
    )


# --------------------------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------------------------


def require_held_out(system_ids: Sequence[str], split_record: Path | str) -> dict[str, object]:
    """Fail closed unless NO scored system appears in any role of the training split record.

    The two namespaces are different by construction -- the benchmark is indexed by UniProt
    accession and the DynDom pool by movement identifier -- so this check cannot pass by accident
    and it is not the only protection: ``docs/domain_motion/dm1_split_record.json`` removed every
    merged homology COMPONENT that touches the evaluation instrument from the pool entirely, which
    is the statement that actually makes these 22 systems held out. This asserts the weaker,
    checkable half and returns the stronger half's own record so a report can carry both.

    Checked rather than assumed because "the ids look different" is exactly the kind of reasoning
    that stops being true when a pool is rebuilt.
    """
    payload = json.loads(Path(split_record).read_text())
    split = payload["split"]
    trained: set[str] = set()
    for role in ("train", "validation", "test"):
        trained.update(split.get(role, ()))
    overlap = sorted(set(system_ids) & trained)
    if overlap:
        raise ValueError(
            f"the benchmark systems {overlap} appear in the training split record; a held-out "
            "score computed over them would be a training score"
        )
    return {
        "split_fingerprint": payload["split_fingerprint"],
        "holdout_instrument": payload["holdout"]["instrument"],
        "holdout_removed_components": payload["holdout"]["removed_components"],
        "holdout_removed_systems": payload["holdout"]["removed_systems"],
        "split_systems": {
            role: len(split.get(role, ())) for role in ("train", "validation", "test")
        },
        "scored_systems": len(system_ids),
    }


def read_sample_index(index_path: Path | str) -> list[tuple[str, str, int, Path]]:
    """Read a BioEmu ``final_structures/index.jsonl`` as ``(sequence_id, sequence, index, path)``."""
    index_path = Path(index_path)
    root = index_path.parent.parent
    entries: list[tuple[str, str, int, Path]] = []
    for line in index_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        entries.append(
            (
                str(record["sequence_id"]),
                str(record["sequence"]),
                int(record["sample_index"]),
                root / str(record["pdb_path"]),
            )
        )
    return entries


def score_ensemble(
    systems: Sequence[SystemReference],
    entries: Sequence[tuple[str, str, int, Path]],
    *,
    coverage_floor: float = DEFAULT_COVERAGE_FLOOR,
) -> tuple[dict[str, SystemScore], dict[str, str], int]:
    """Score every draw of every system, returning ``(scores, refusals, draws outside the pool)``.

    A draw whose ``sequence_id`` names no benchmark system is COUNTED and skipped -- a native
    ensemble carries a warm-up sequence -- and a system whose draws cannot be placed on its own
    sequence is refused BY NAME rather than dropped, because a silently missing system moves the
    denominator of a mean over systems.
    """
    by_name = {system.test_case: system for system in systems}
    scores: dict[str, SystemScore] = {}
    refusals: dict[str, str] = {}
    outside = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sequence_id, sequence, sample_index, path in entries:
            system = by_name.get(sequence_id)
            if system is None:
                outside += 1
                continue
            if sequence != system.sequence:
                refusals[sequence_id] = (
                    "the indexed sequence is not the benchmark test-case sequence "
                    f"({len(sequence)} residues against {len(system.sequence)})"
                )
                continue
            chain = read_generated_backbone(path, sequence)
            score = scores.setdefault(sequence_id, SystemScore(system=sequence_id))
            for conformer_index in (0, 1):
                try:
                    deviation, _coverage = score_draw(
                        system,
                        conformer_index,
                        chain,
                        sample_id=path.name,
                        sample_index=sample_index,
                        coverage_floor=coverage_floor,
                    )
                except DomainMotionScoringError as error:
                    refusals[f"{sequence_id}:{conformer_index}"] = str(error)
                    continue
                score.deviations.setdefault(conformer_index, []).append(deviation)
    return scores, refusals, outside
