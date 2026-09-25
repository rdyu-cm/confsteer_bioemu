"""success@B on the BioEmu cryptic-pocket benchmark: the reported metric (D-63).

WHY THIS MODULE EXISTS. Both previous cryptic metrics were functions of the coordinate the adapter
is trained on (``delta_the progress coordinate_dose_slope_sigma_the progress coordinate`` and ``the progress coordinate_crossing_fraction``). ``the progress coordinate = 1`` is
necessary for holo but not sufficient -- it is a hyperplane containing holo and every structure at
any off-axis distance -- so training on ``the progress coordinate`` and scoring on ``the progress coordinate`` is circular. success@B replaces
it with the field's own instrument: a distance in Angstrom to the experimental reference, computed
the way ``bioemu-benchmarks`` computes it, which ConforNets and ConforMix both re-use. ``the progress coordinate`` stays
the training coordinate and is demoted to mechanism.

THE PROTOCOL, per system, per sample, per reference state (apo and holo scored separately):

1. superpose the sample onto the reference over ``alignment_resid_ranges``;
2. take per-atom deviations over ``metric_resid_ranges`` on backbone N/CA/C/O;
3. the system is a success at ``(tau, B)`` when at least one of ``B`` samples is within ``tau``.

Five things make a number that looks comparable and is not. Each is handled here explicitly.

**1. BioEmu's "local RMSD" is not an RMSD.** ``eval/multiconf/evaluate.py::calc_metrics_local``
does ``metrics[MetricType.RMSD] = np.mean(pairwise_distances, axis=-1)`` -- the arithmetic MEAN of
per-atom Euclidean deviations after superposition. Only the GLOBAL path calls ``mdtraj.rmsd``. Mean
<= RMS, so a 1 A "local RMSD" is LOOSER than a 1 A true RMSD, and ConforNets' claim that it
computes RMSD "using mdtraj" while following BioEmu's local pipeline cannot be exactly reconciled
with that source. This module therefore emits BOTH forms for every (sample, reference) pair and
reports success at both: :data:`MEAN_DEVIATION` reproduces BioEmu's ``calc_metrics_local``, and
:data:`RMSD` is the true root-mean-square. Nothing here is ever called a bare ``rmsd`` unless it
is one.

**2. The ranges are regions, not pocket-lining residues.** They run 15-324 residues; ``P0DP23`` is
``[[1, 147]]``, essentially the whole of calmodulin. ``alignment_resid_ranges`` and
``metric_resid_ranges`` are separate lists and are allowed to differ, so each is used for its own
job. Substituting this project's own pocket residues would be a different metric and is not done.

**3. tau and B dominate the number.** They are arguments with no defaults that could be mistaken
for a protocol, and both are written into the artifact next to every rate.

**4. Published baselines are not our control.** BioEmu's own numbers come from a different sampler
(the benchmark repo warns v0.0.2 is not comparable to v1.0.0+, and the public release uses DPM/30
steps against the paper's Heun/100). This project samples with DPM-Solver-2 at 49 steps, so the
native arm must be regenerated under our settings; this module scores whatever ensemble it is
given and records the sampler settings it was told about rather than assuming them.

**5. Dataset-construction criteria are not generation metrics.** CryptoBench's 2 A pocket RMSD,
CryptoBank's crypticity score and PocketMiner's LIGSITE volume label whether a SITE is cryptic.
None of them scores a conformer and none appears here.

A SIXTH THING, ADDED RATHER THAN SUBSTITUTED: THE SPECIFICITY CRITERION
-----------------------------------------------------------------------

Step 3 above presumes apo and holo are far enough apart for "within ``tau`` of holo" to be a
statement about holo, and on this project's training pool they frequently are not: 62% of the 1920
systems have their two crystal conformers closer together than ``tau = 1.5`` over the graded
region, so sitting at apo already scores as a holo success.
:mod:`~sae_bioemu.cryptic_steering.conformer_separation` removes those systems from the pool; what
remains for the SCORER is the band ``tau < s <= 2*tau``, where the two ``tau``-balls still overlap.

:func:`specific_success_at_b` is the answer, and it is an ADDITIONAL cell, never a replacement:
a sample counts only if it is within ``tau`` of the target state AND NOT within ``tau`` of the
other. :func:`success_at_b` is unchanged, keeps :data:`SUCCESS_CRITERION_PLAIN`, and remains the
number that is comparable with published figures. Every serialised cell names its
:data:`SUCCESS_CRITERIA` member and carries the definition in words, because the whole point of the
five paragraphs above is that a variant presented under the incumbent's name is not comparable, and
that applies to this project's own variant first of all. Both criteria are carried through BOTH
:data:`DEVIATION_FORMS`; neither form is picked.

Every result also carries a per-system separation STRATUM -- ``clean`` (``s > 2*tau``),
``ambiguous`` (``tau < s <= 2*tau``), ``degenerate`` (``s <= tau``) or ``unknown`` -- as a tag that
enters no success test. It is reported, never used to filter: the rate is the rate over every
system scored, and :meth:`SuccessResult.rate_within` exists so a band can be read beside that rate
instead of removed from it.

THE NUMBERING FRAME IS THE SILENT FAILURE, and it is real rather than hypothetical. The benchmark's
ranges index its OWN processed reference PDBs, whose README warns they "may have a different
residue numbering as the ones deposited in the PDB". Measured against the cached deposited mmCIFs,
the modal ``auth_seq_id`` offset from the benchmark frame is non-zero for 15 of the 33 scorable
systems (+1048 for ``P08581``, +1000 for ``P12758``, +797 for ``Q02763``, +116 for ``P08037``), and
for ``P08709``, ``P12758``, ``P45568`` and ``P56817`` the apo and holo offsets DIFFER from each
other -- so no reading of the ranges in deposited numbering can be right for both references of one
system.
The frame used here is therefore the benchmark's own: **1-based positions in the test-case sequence
from ``testcases.csv``**, whose length is exactly the ``n_residues`` the upstream evaluator uses to
expand an open-ended range. Deposited references and generated samples are each mapped into that
frame by letter-matched global alignment, and positions that fail to resolve on either side are
counted and reported, never dropped silently: below :data:`DEFAULT_COVERAGE_FLOOR` the system is
refused by name with :class:`ResidueCorrespondenceError`.

``Q16539`` has no ``local_residinfo`` file upstream (33 files for 34 test cases) and this project's
independent D-19 inventory found the same gap. It cannot be scored on this metric and is reported
as EXCLUDED with that reason -- never as a failure, which would silently move the denominator.
"""

from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The strata vocabulary is defined ONCE, in the module that owns the admission rule, and imported
# here rather than restated: the pool and the benchmark must be sorted into the same three bands by
# the same arithmetic, or "ambiguous" means two things in one report. That module imports nothing
# heavy at module scope, so this costs nothing.
from sae_bioemu.cryptic_steering.conformer_separation import (
    SEPARATION_STRATA,
    STRATUM_UNKNOWN,
    separation_stratum,
)

#: BioEmu's benchmark filters both reference and sample to mdtraj's ``backbone`` selection, which
#: is exactly these four atoms, and its own comment pins the per-residue order as ``[N, CA, C, O]``.
#: BioEmu samples carry a fifth atom, the C-beta placed from the residue frame; it is excluded by
#: that same filter upstream and is excluded here.
BACKBONE_ATOMS: tuple[str, ...] = ("N", "CA", "C", "O")

#: The arithmetic mean of per-atom deviations. This is what ``calc_metrics_local`` labels ``rmsd``.
MEAN_DEVIATION = "mean_deviation"
#: The true root-mean-square of the same per-atom deviations. Always >= :data:`MEAN_DEVIATION`.
RMSD = "rmsd"
#: Both forms are always emitted; a table carrying only one of them is not interpretable.
DEVIATION_FORMS: tuple[str, ...] = (MEAN_DEVIATION, RMSD)

#: The two reference states. They are different numbers -- apo is much harder -- and are never
#: pooled.
APO = "apo"
HOLO = "holo"
REFERENCE_STATES: tuple[str, ...] = (APO, HOLO)

#: THE FIELD'S OWN CRITERION: at least one of ``B`` samples within ``tau`` of the reference state.
#: This is what ``bioemu-benchmarks``, ConforNets and ConforMix all report, and it is what makes a
#: number from this module comparable to theirs. It is never redefined, never blended with the
#: criterion below, and never silently substituted -- see the module docstring.
SUCCESS_CRITERION_PLAIN = "plain"

#: THE SPECIFICITY CRITERION, reported BESIDE the plain one and never in place of it: at least one
#: of ``B`` samples within ``tau`` of the target state AND NOT within ``tau`` of the other state.
#:
#: It exists because on this project's pool the two deposited conformers are frequently closer to
#: each other than ``tau``, so "within ``tau`` of holo" can be satisfied by sitting at apo. The
#: admission rule in :mod:`~sae_bioemu.cryptic_steering.conformer_separation` removes the systems
#: where that is certain (``s <= tau``); this criterion removes it per SAMPLE on the systems that
#: survive, including the whole ambiguous band ``tau < s <= 2*tau`` where a sample inside both
#: ``tau``-balls still exists geometrically.
#:
#: It SUBSUMES the rigorous ``s > 2*tau`` admission cut rather than competing with it: above
#: ``2*tau`` the two balls are disjoint, so every plain success is already specific and the two
#: criteria coincide. That is why the pool is cut at ``tau`` and the rigour is taken here.
SUCCESS_CRITERION_SPECIFIC = "specific"

#: Both are always available and a report carrying one of them unlabelled is not interpretable.
SUCCESS_CRITERIA: tuple[str, ...] = (SUCCESS_CRITERION_PLAIN, SUCCESS_CRITERION_SPECIFIC)

#: Written into every serialised cell, so an artifact states which rule produced its number rather
#: than relying on a reader knowing what the label means.
SUCCESS_CRITERION_DEFINITIONS: dict[str, str] = {
    SUCCESS_CRITERION_PLAIN: (
        "at least one of B samples is within tau of the named reference state. The field's "
        "instrument, comparable with bioemu-benchmarks / ConforNets / ConforMix."
    ),
    SUCCESS_CRITERION_SPECIFIC: (
        "at least one of B samples is within tau of the named reference state AND NOT within tau "
        "of the other state. A specificity criterion reported BESIDE the plain rate, never in "
        "place of it; it is NOT comparable with published success@B numbers."
    ),
}

#: The fraction of requested benchmark positions that must resolve on both sides before a system
#: is scored. Crystals do not observe every residue of the construct, so demanding 100% would
#: refuse nearly every system; below this, the surviving selection is a different region from the
#: one the benchmark defines and the number would not be comparable.
DEFAULT_COVERAGE_FLOOR = 0.9

#: Superposition needs at least this many residues to be anything other than arbitrary.
MINIMUM_ALIGNMENT_RESIDUES = 3

#: Sample sequences must match the benchmark test-case sequence this closely over aligned
#: positions. A sample generated from a different construct is a different protein.
DEFAULT_SEQUENCE_IDENTITY_FLOOR = 0.95

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip


class SuccessAtBError(ValueError):
    """Base class for refusals that name the system they concern."""

    def __init__(self, system: str, message: str) -> None:
        self.system = system
        super().__init__(f"{system}: {message}")


class ResidueCorrespondenceError(SuccessAtBError):
    """The benchmark ranges could not be resolved onto a structure well enough to score it.

    Raised rather than returning a truncated selection. A selection that has quietly lost a third
    of its residues still produces a plausible Angstrom number, and that number is not the
    benchmark's.
    """


class MissingRangesError(SuccessAtBError):
    """The system has no upstream ``local_residinfo`` file, so it cannot be scored at all."""


# --------------------------------------------------------------------------------------------
# Benchmark ranges
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkRanges:
    """One system's alignment and metric regions, in the benchmark's own numbering frame."""

    system: str
    alignment_resid_ranges: tuple[tuple[int, int], ...]
    metric_resid_ranges: tuple[tuple[int, int], ...]
    #: The test-case sequence length, which is what the upstream evaluator uses to expand an
    #: open-ended range.
    n_residues: int
    sha256: str | None = None
    is_whole_chain: bool = False

    @property
    def alignment_positions(self) -> tuple[int, ...]:
        return expand_ranges(self.alignment_resid_ranges, self.n_residues)

    @property
    def metric_positions(self) -> tuple[int, ...]:
        return expand_ranges(self.metric_resid_ranges, self.n_residues)


def expand_ranges(resid_ranges: Sequence[Sequence[int | None]], n_residues: int) -> tuple[int, ...]:
    """Expand ``[[begin, end], ...]`` into explicit 1-based positions, as upstream does.

    ``LocalResidInfo.parse_range`` treats ``None`` as "from the first residue" and "up to
    ``n_residues``" in the Python-slice sense, requires ranges in strictly ascending order, and
    stops at ``n_residues`` when a range runs past the end. All three behaviours are reproduced:
    the shipped files happen to use no ``None`` and stay in bounds, but a silently different
    expansion is exactly the kind of divergence that makes a number look comparable and not be.
    """
    positions: list[int] = []
    last_end = -1
    for entry in resid_ranges:
        if len(entry) != 2:
            raise ValueError(f"a resid range must be [begin, end]; got {list(entry)!r}")
        begin = 1 if entry[0] is None else int(entry[0])
        end = int(n_residues) if entry[1] is None else int(entry[1])
        if begin > end:
            raise ValueError(f"resid range {list(entry)!r} is not ascending")
        if begin <= last_end:
            raise ValueError(f"resid ranges overlap or are unsorted at {list(entry)!r}")
        if end > n_residues:
            positions.extend(range(begin, n_residues + 1))
            return tuple(positions)
        positions.extend(range(begin, end + 1))
        last_end = end
    return tuple(positions)


def load_benchmark_ranges(path: Path | str) -> tuple[dict[str, BenchmarkRanges], dict[str, str]]:
    """Load the D-19 ranges file, returning usable systems and the reasons the rest are excluded.

    The excluded map is part of the result rather than a warning: a system with no upstream range
    file must reach the report as an exclusion with its reason, and never as a failed system.
    """
    payload = json.loads(Path(path).read_text())
    systems: dict[str, BenchmarkRanges] = {}
    excluded: dict[str, str] = {}
    for name, record in sorted(payload["systems"].items()):
        if not record.get("has_upstream_file", False) or "metric_resid_ranges" not in record:
            excluded[name] = str(
                record.get(
                    "note",
                    "no upstream local_residinfo file; ranges are not recoverable",
                )
            )
            continue
        systems[name] = BenchmarkRanges(
            system=name,
            alignment_resid_ranges=tuple(
                (int(begin), int(end)) for begin, end in record["alignment_resid_ranges"]
            ),
            metric_resid_ranges=tuple(
                (int(begin), int(end)) for begin, end in record["metric_resid_ranges"]
            ),
            n_residues=int(record["n_residues"]),
            sha256=record.get("sha256"),
            is_whole_chain=bool(record.get("is_whole_chain", False)),
        )
    return systems, excluded


# --------------------------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BackboneResidue:
    """One residue's backbone atoms, in whatever numbering its own file uses."""

    #: ``auth_seq_id`` plus stripped insertion code for a deposited chain; the 0-based residue
    #: index for a generated sample.
    key: str
    letter: str
    atoms: dict[str, np.ndarray]

    def coordinates(self, atom_names: Sequence[str]) -> np.ndarray | None:
        """Return the requested atoms as ``(len(atom_names), 3)``, or ``None`` if any is absent."""
        if any(name not in self.atoms for name in atom_names):
            return None
        return np.stack([self.atoms[name] for name in atom_names])


@dataclass(frozen=True)
class BackboneChain:
    """An ordered list of residues plus the one-letter sequence they spell."""

    residues: tuple[BackboneResidue, ...]
    source: str

    @property
    def letters(self) -> str:
        return "".join(residue.letter for residue in self.residues)

    def __len__(self) -> int:
        return len(self.residues)


def read_reference_chain(
    path: Path | str, chain_id: str, atom_names: Sequence[str] = BACKBONE_ATOMS
) -> BackboneChain:
    """Read one auth chain of a deposited mmCIF, keeping the requested backbone atoms.

    ``structures.read_chain`` deliberately keeps only C-alpha and C-beta, because the steering
    coordinate is built on those; this metric needs N/CA/C/O, so the read is repeated here with
    the same conventions -- Biopython's ``MMCIFParser`` keys chains by ``auth_asym_id`` and
    residues by ``auth_seq_id``, hetero residues and residues without a C-alpha are skipped, and a
    duplicate key is dropped rather than silently overwriting the residue that came first.
    """
    from Bio.PDB import MMCIFParser

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = MMCIFParser(QUIET=True).get_structure(Path(path).stem, str(path))
    try:
        chain = structure[0][chain_id]
    except KeyError:
        return BackboneChain((), f"{Path(path).name}:{chain_id}")

    seen: set[str] = set()
    residues: list[BackboneResidue] = []
    wanted = set(atom_names)
    for residue in chain:
        hetflag, seq_id, icode = residue.id
        if hetflag.strip() or "CA" not in residue:
            continue
        letter = _THREE_TO_ONE.get(residue.get_resname().strip().upper(), "")
        if len(letter) != 1:
            continue
        key = f"{int(seq_id)}{str(icode).strip()}"
        if key in seen:
            continue
        seen.add(key)
        atoms = {
            atom.get_name().strip(): np.asarray(atom.get_coord(), dtype=float)
            for atom in residue
            if atom.get_name().strip() in wanted
        }
        residues.append(BackboneResidue(key=key, letter=letter, atoms=atoms))
    return BackboneChain(tuple(residues), f"{Path(path).name}:{chain_id}")


_ATOM_RECORD = re.compile(r"^ATOM  ")


def read_generated_backbone(
    path: Path | str,
    sequence: str | None = None,
    atom_names: Sequence[str] = BACKBONE_ATOMS,
) -> BackboneChain:
    """Read a BioEmu final-structure PDB: one chain, residues numbered ``0..L-1``.

    BioEmu writes backbone plus the C-beta it places from the residue frame, so N/CA/C/O are all
    present and no C-alpha-only fallback is ever needed or taken. When ``sequence`` is given, every
    residue's three-letter code must agree with it and the residue numbering must be the plain
    ``0..L-1`` index, which ties the file on disk to the sequence recorded for it in the sample
    index rather than trusting the file name.
    """
    residues: list[BackboneResidue] = []
    atoms: dict[str, np.ndarray] = {}
    current: tuple[int, str] | None = None
    wanted = set(atom_names)

    def flush() -> None:
        if current is None:
            return
        letter = _THREE_TO_ONE.get(current[1], "")
        if len(letter) == 1:
            residues.append(BackboneResidue(key=str(current[0]), letter=letter, atoms=dict(atoms)))

    for line in Path(path).read_text().splitlines():
        if not _ATOM_RECORD.match(line):
            continue
        name = line[12:16].strip()
        resname = line[17:20].strip().upper()
        resseq = int(line[22:26])
        if current is None or (resseq, resname) != current:
            flush()
            atoms = {}
            current = (resseq, resname)
        if name in wanted:
            atoms[name] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float
            )
    flush()

    chain = BackboneChain(tuple(residues), str(Path(path).name))
    if sequence is not None:
        if chain.letters != sequence:
            raise ValueError(
                f"{Path(path).name}: residue letters do not spell the indexed sequence "
                f"({len(chain)} residues against {len(sequence)})"
            )
        expected = [str(index) for index in range(len(sequence))]
        if [residue.key for residue in chain.residues] != expected:
            raise ValueError(
                f"{Path(path).name}: residues are not numbered 0..L-1 as BioEmu writes them"
            )
    return chain


# --------------------------------------------------------------------------------------------
# The benchmark numbering frame
# --------------------------------------------------------------------------------------------


def _letter_matched_alignment(left: str, right: str) -> list[tuple[int, int]]:
    """Global pairwise alignment, keeping only aligned pairs whose letters agree.

    Same aligner settings as :func:`structures.alignment_map`, so the two maps this project builds
    between residue frames cannot disagree because of scoring.
    """
    from Bio.Align import PairwiseAligner

    if not left or not right:
        return []
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    alignment = aligner.align(left, right)[0]
    pairs: list[tuple[int, int]] = []
    for (left_start, left_end), (right_start, _) in zip(*alignment.aligned, strict=False):
        for offset in range(left_end - left_start):
            if left[left_start + offset] == right[right_start + offset]:
                pairs.append((left_start + offset, right_start + offset))
    return pairs


@dataclass(frozen=True)
class FrameMap:
    """A structure's residues placed in the benchmark's 1-based test-case sequence frame."""

    #: benchmark position (1-based) -> index into the chain's residue list.
    position_to_index: dict[int, int]
    #: Residues of the chain that the alignment placed.
    matched: int
    #: Residues of the chain in total.
    chain_residues: int
    #: The modal ``auth_seq_id - benchmark_position`` offset and how many distinct offsets occur.
    #: A non-zero modal offset is the reason this frame exists rather than reading the ranges as
    #: deposited numbering.
    auth_offset_modal: int | None
    auth_offset_distinct: int

    @property
    def matched_fraction(self) -> float:
        return self.matched / self.chain_residues if self.chain_residues else 0.0


def benchmark_frame(chain: BackboneChain, testcase_sequence: str) -> FrameMap:
    """Place a chain's residues on the benchmark test-case sequence by letter-matched alignment."""
    pairs = _letter_matched_alignment(chain.letters, testcase_sequence)
    position_to_index = {sequence_index + 1: chain_index for chain_index, sequence_index in pairs}
    offsets: Counter[int] = Counter()
    for chain_index, sequence_index in pairs:
        key = chain.residues[chain_index].key
        digits = re.match(r"^-?\d+", key)
        if digits is not None:
            offsets[int(digits.group()) - (sequence_index + 1)] += 1
    modal = offsets.most_common(1)[0][0] if offsets else None
    return FrameMap(
        position_to_index=position_to_index,
        matched=len(pairs),
        chain_residues=len(chain),
        auth_offset_modal=modal,
        auth_offset_distinct=len(offsets),
    )


@dataclass(frozen=True)
class SelectionCoverage:
    """How much of a requested benchmark region survived onto both structures, and why not more."""

    requested: int
    on_reference: int
    on_sample: int
    used: int
    #: Positions lost because the reference crystal does not observe them, the sample construct
    #: does not contain them, or one side is missing a backbone atom. Counted, never hidden.
    missing_on_reference: int
    missing_on_sample: int
    incomplete_atoms: int

    @property
    def fraction(self) -> float:
        return self.used / self.requested if self.requested else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "requested": self.requested,
            "on_reference": self.on_reference,
            "on_sample": self.on_sample,
            "used": self.used,
            "missing_on_reference": self.missing_on_reference,
            "missing_on_sample": self.missing_on_sample,
            "incomplete_atoms": self.incomplete_atoms,
            "fraction": self.fraction,
        }


@dataclass(frozen=True)
class RegionCoverage:
    """How much of one benchmark region a SINGLE structure resolves with a complete backbone.

    Computed without any sample, so it says in advance which systems an ensemble could be scored
    on: the reference crystal is the only side that can fail to observe a residue once a sample
    exists, since a generated structure has every residue of its construct.
    """

    requested: int
    resolved: int
    complete_backbone: int

    @property
    def fraction(self) -> float:
        return self.complete_backbone / self.requested if self.requested else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "complete_backbone": self.complete_backbone,
            "fraction": self.fraction,
        }


def region_coverage(
    positions: Sequence[int],
    chain: BackboneChain,
    frame: FrameMap,
    atom_names: Sequence[str] = BACKBONE_ATOMS,
) -> RegionCoverage:
    """Count how many of ``positions`` one structure observes with all of ``atom_names``."""
    resolved = 0
    complete = 0
    for position in positions:
        index = frame.position_to_index.get(position)
        if index is None:
            continue
        resolved += 1
        if chain.residues[index].coordinates(atom_names) is not None:
            complete += 1
    return RegionCoverage(requested=len(positions), resolved=resolved, complete_backbone=complete)


@dataclass(frozen=True)
class Selection:
    """Matched atom rows for one region, plus the coverage that produced them."""

    reference_rows: np.ndarray
    sample_rows: np.ndarray
    residues: int
    coverage: SelectionCoverage


def build_selection(
    positions: Sequence[int],
    reference: BackboneChain,
    reference_frame: FrameMap,
    sample: BackboneChain,
    sample_frame: FrameMap,
    atom_names: Sequence[str] = BACKBONE_ATOMS,
) -> Selection:
    """Gather the atoms of ``positions`` from both structures, in the benchmark frame.

    Correspondence runs through the benchmark position rather than either structure's own
    numbering, so a deposited chain numbered ``1048`` residues away from the frame and a sample
    numbered from zero meet at the same place. Both alignments are letter-matched, so a used
    position carries the same residue letter on both sides by construction; that is asserted
    rather than assumed.
    """
    reference_rows: list[np.ndarray] = []
    sample_rows: list[np.ndarray] = []
    on_reference = 0
    on_sample = 0
    missing_on_reference = 0
    missing_on_sample = 0
    incomplete = 0
    for position in positions:
        reference_index = reference_frame.position_to_index.get(position)
        sample_index = sample_frame.position_to_index.get(position)
        if reference_index is not None:
            on_reference += 1
        else:
            missing_on_reference += 1
        if sample_index is not None:
            on_sample += 1
        else:
            missing_on_sample += 1
        if reference_index is None or sample_index is None:
            continue
        reference_residue = reference.residues[reference_index]
        sample_residue = sample.residues[sample_index]
        if reference_residue.letter != sample_residue.letter:
            raise AssertionError(
                f"benchmark position {position} carries {reference_residue.letter!r} on the "
                f"reference and {sample_residue.letter!r} on the sample"
            )
        left = reference_residue.coordinates(atom_names)
        right = sample_residue.coordinates(atom_names)
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
            requested=len(positions),
            on_reference=on_reference,
            on_sample=on_sample,
            used=used,
            missing_on_reference=missing_on_reference,
            missing_on_sample=missing_on_sample,
            incomplete_atoms=incomplete,
        ),
    )


# --------------------------------------------------------------------------------------------
# Superposition and deviations
# --------------------------------------------------------------------------------------------


def kabsch_superpose(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rotation, translation)`` taking ``mobile`` onto ``target`` in least squares.

    The same construction as :func:`sae_bioemu.steering.ensemble_rmsf.align_ensemble_kabsch` --
    centre both sets, take the SVD of the covariance and force a proper rotation by correcting the
    determinant -- expressed for one pair of point sets in NumPy instead of a torch ensemble
    aligned to its own first member. ``tests/test_cryptic_steering_success_at_b.py`` checks the two
    agree on a shared case, because a superposition that quietly differs from the project's other
    one would move every number here.
    """
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("mobile and target must be matching (n, 3) arrays")
    if mobile.shape[0] < 3:
        raise ValueError("superposition needs at least three points")
    mobile_centre = mobile.mean(axis=0)
    target_centre = target.mean(axis=0)
    covariance = (mobile - mobile_centre).T @ (target - target_centre)
    left, _, right = np.linalg.svd(covariance)
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(right.T @ left.T)))])
    rotation = right.T @ correction @ left.T
    return rotation, target_centre - rotation @ mobile_centre


def deviation_forms(distances: np.ndarray) -> dict[str, float]:
    """Return both deviation forms from per-atom Euclidean distances.

    ``mean_deviation`` is what ``bioemu-benchmarks`` calls ``rmsd`` on its local path; ``rmsd`` is
    the root-mean-square. By Jensen the mean is never larger, so a tau applied to the mean is
    always the looser test -- which is why both are reported side by side rather than one being
    chosen.
    """
    if distances.size == 0:
        raise ValueError("no atoms contributed to the deviation")
    return {
        MEAN_DEVIATION: float(np.mean(distances)),
        RMSD: float(np.sqrt(np.mean(distances**2))),
    }


def superposed_deviations(
    alignment: Selection, metric: Selection
) -> tuple[dict[str, float], np.ndarray]:
    """Superpose on the alignment region, then measure over the metric region.

    The two regions are separate lists in the benchmark and are used separately here: the rotation
    comes from ``alignment_resid_ranges`` only, and no part of the metric region influences it.
    """
    if alignment.residues < MINIMUM_ALIGNMENT_RESIDUES:
        raise ValueError(
            f"superposition needs at least {MINIMUM_ALIGNMENT_RESIDUES} matched residues; "
            f"got {alignment.residues}"
        )
    rotation, translation = kabsch_superpose(alignment.sample_rows, alignment.reference_rows)
    moved = metric.sample_rows @ rotation.T + translation
    distances = np.linalg.norm(moved - metric.reference_rows, axis=1)
    return deviation_forms(distances), distances


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleDeviation:
    """One sample measured against one reference state."""

    system: str
    reference_state: str
    sample_id: str
    sample_index: int
    mean_deviation: float
    rmsd: float
    metric_residues: int
    metric_atoms: int

    def value(self, form: str) -> float:
        if form == MEAN_DEVIATION:
            return self.mean_deviation
        if form == RMSD:
            return self.rmsd
        raise ValueError(f"form must be one of {DEVIATION_FORMS}")


@dataclass(frozen=True)
class ReferenceSeparation:
    """The experimental apo<->holo deviation of one system, under the benchmark's own regions.

    This is the scale the metric is measured against and the source of ConforMix's adaptive
    threshold ``tau_i = 0.5 x RMSD(apo_i, holo_i)``. It is symmetric: superposing apo onto holo
    over the alignment region and superposing holo onto apo give the same optimal rotation up to
    inversion and the same per-atom distances.
    """

    system: str
    mean_deviation: float
    rmsd: float
    metric_residues: int
    alignment_residues: int
    metric_coverage: SelectionCoverage
    alignment_coverage: SelectionCoverage

    def value(self, form: str) -> float:
        """Return ``s`` in one deviation form: the scale the admission rule cuts on.

        Same spelling as :meth:`SampleDeviation.value` on purpose. ``s`` and a sample's deviation
        must be read in the SAME form for a stratum to describe the threshold the samples are
        graded at, and two different accessor names is how one ends up comparing a mean-form
        separation against an RMS-form success.
        """
        if form == MEAN_DEVIATION:
            return self.mean_deviation
        if form == RMSD:
            return self.rmsd
        raise ValueError(f"form must be one of {DEVIATION_FORMS}")

    def adaptive_tau(self, form: str) -> float:
        return 0.5 * (self.mean_deviation if form == MEAN_DEVIATION else self.rmsd)


@dataclass
class SystemScore:
    """Everything measured for one benchmark system, before any tau or B is applied."""

    system: str
    deviations: dict[str, list[SampleDeviation]] = field(default_factory=dict)
    coverage: dict[str, dict[str, SelectionCoverage]] = field(default_factory=dict)
    frames: dict[str, dict[str, int | None | float]] = field(default_factory=dict)
    reference_separation: ReferenceSeparation | None = None

    def samples(self, reference_state: str) -> list[SampleDeviation]:
        return self.deviations.get(reference_state, [])

    def separation_stratum(self, *, form: str, tau: float) -> str:
        """Which separation band this system sits in, at one ``tau`` and in one deviation form.

        ``s`` and the samples must be read in the SAME form, so this takes ``form`` rather than
        choosing one. A system whose apo<->holo separation could not be measured -- five of the
        benchmark's 33 -- returns
        :data:`~sae_bioemu.cryptic_steering.conformer_separation.STRATUM_UNKNOWN` rather than
        being dropped from the table or counted as clean.
        """
        if form not in DEVIATION_FORMS:
            raise ValueError(f"form must be one of {DEVIATION_FORMS}")
        if self.reference_separation is None:
            return STRATUM_UNKNOWN
        return separation_stratum(self.reference_separation.value(form), tau_angstrom=tau)

    def paired_samples(
        self, *, target_state: str, b: int | None
    ) -> list[tuple[SampleDeviation, SampleDeviation]]:
        """Return the first ``B`` samples as ``(target, other)`` pairs, or refuse.

        The specificity criterion is a statement about ONE SAMPLE measured against BOTH references,
        so the two per-state lists must be joined on the sample itself -- ``(sample_index,
        sample_id)`` -- and not merely counted. A sample present against the target and absent
        against the other state raises: treating the missing measurement as "not within tau" would
        manufacture specific successes out of a gap in the table, which is exactly the shape of
        failure this module refuses everywhere else.
        """
        if target_state not in REFERENCE_STATES:
            raise ValueError(f"target_state must be one of {REFERENCE_STATES}")
        other_state = HOLO if target_state == APO else APO
        other = {
            (sample.sample_index, sample.sample_id): sample for sample in self.samples(other_state)
        }
        pairs: list[tuple[SampleDeviation, SampleDeviation]] = []
        for sample in take_b(self.samples(target_state), b):
            key = (sample.sample_index, sample.sample_id)
            counterpart = other.get(key)
            if counterpart is None:
                raise SuccessAtBError(
                    self.system,
                    f"sample {sample.sample_id!r} (index {sample.sample_index}) is measured "
                    f"against {target_state} but not against {other_state}. The specificity "
                    "criterion asks whether ONE structure is near one reference and far from the "
                    "other, so it cannot be evaluated on a sample with only one measurement",
                )
            pairs.append((sample, counterpart))
        return pairs


def score_sample(
    system: str,
    reference_state: str,
    sample_id: str,
    sample_index: int,
    ranges: BenchmarkRanges,
    reference: BackboneChain,
    reference_frame: FrameMap,
    sample: BackboneChain,
    sample_frame: FrameMap,
    atom_names: Sequence[str] = BACKBONE_ATOMS,
    coverage_floor: float = DEFAULT_COVERAGE_FLOOR,
) -> tuple[SampleDeviation, dict[str, SelectionCoverage]]:
    """Measure one sample against one reference, refusing rather than truncating the selection."""
    alignment = build_selection(
        ranges.alignment_positions, reference, reference_frame, sample, sample_frame, atom_names
    )
    metric = build_selection(
        ranges.metric_positions, reference, reference_frame, sample, sample_frame, atom_names
    )
    for name, selection in (("alignment", alignment), ("metric", metric)):
        if selection.coverage.fraction < coverage_floor:
            raise ResidueCorrespondenceError(
                system,
                f"{reference_state} {name} region resolved "
                f"{selection.coverage.used}/{selection.coverage.requested} positions "
                f"({selection.coverage.fraction:.3f} < {coverage_floor:.3f}); "
                f"missing on reference {selection.coverage.missing_on_reference}, "
                f"missing on sample {selection.coverage.missing_on_sample}, "
                f"incomplete backbone {selection.coverage.incomplete_atoms}",
            )
    values, _ = superposed_deviations(alignment, metric)
    return (
        SampleDeviation(
            system=system,
            reference_state=reference_state,
            sample_id=sample_id,
            sample_index=sample_index,
            mean_deviation=values[MEAN_DEVIATION],
            rmsd=values[RMSD],
            metric_residues=metric.residues,
            metric_atoms=int(metric.reference_rows.shape[0]),
        ),
        {"alignment": alignment.coverage, "metric": metric.coverage},
    )


def score_reference_separation(
    system: str,
    ranges: BenchmarkRanges,
    apo: BackboneChain,
    apo_frame: FrameMap,
    holo: BackboneChain,
    holo_frame: FrameMap,
    atom_names: Sequence[str] = BACKBONE_ATOMS,
    coverage_floor: float = DEFAULT_COVERAGE_FLOOR,
) -> ReferenceSeparation:
    """Measure the experimental apo->holo deviation under exactly the sample protocol."""
    alignment = build_selection(
        ranges.alignment_positions, apo, apo_frame, holo, holo_frame, atom_names
    )
    metric = build_selection(ranges.metric_positions, apo, apo_frame, holo, holo_frame, atom_names)
    for name, selection in (("alignment", alignment), ("metric", metric)):
        if selection.coverage.fraction < coverage_floor:
            raise ResidueCorrespondenceError(
                system,
                f"apo/holo {name} region resolved "
                f"{selection.coverage.used}/{selection.coverage.requested} positions "
                f"({selection.coverage.fraction:.3f} < {coverage_floor:.3f})",
            )
    values, _ = superposed_deviations(alignment, metric)
    return ReferenceSeparation(
        system=system,
        mean_deviation=values[MEAN_DEVIATION],
        rmsd=values[RMSD],
        metric_residues=metric.residues,
        alignment_residues=alignment.residues,
        metric_coverage=metric.coverage,
        alignment_coverage=alignment.coverage,
    )


# --------------------------------------------------------------------------------------------
# success@B
# --------------------------------------------------------------------------------------------


def take_b(samples: Sequence[SampleDeviation], b: int | None) -> list[SampleDeviation]:
    """Return the first ``B`` samples in ascending sample-index order.

    Deterministic and independent of directory listing order, because ``B`` is part of the metric:
    "the best of 100 draws" and "the best of 32" are different questions and the choice of WHICH 32
    must not depend on which file the filesystem returned first.
    """
    ordered = sorted(samples, key=lambda item: (item.sample_index, item.sample_id))
    return ordered if b is None else ordered[:b]


@dataclass(frozen=True)
class SuccessResult:
    """One (criterion, reference state, deviation form, tau, B) cell of the reported table.

    ``criterion`` names WHICH success rule produced ``successes`` and is part of the cell's
    identity, not a label. :data:`SUCCESS_CRITERION_PLAIN` and :data:`SUCCESS_CRITERION_SPECIFIC`
    are different numbers over the same samples and a table carrying one of them unlabelled is not
    interpretable -- which is the substitution this module's docstring exists to prevent.
    """

    reference_state: str
    form: str
    tau: float
    b_requested: int | None
    systems: tuple[str, ...]
    successes: tuple[bool, ...]
    #: Samples actually available per system, which is what ``B`` was truncated against.
    samples_per_system: tuple[int, ...]
    confidence_low: float
    confidence_high: float
    bootstrap_resamples: int
    bootstrap_seed: int
    #: WHICH success rule this cell reports. Defaults to the plain criterion so every cell written
    #: before the specific one existed keeps its meaning and its number.
    criterion: str = SUCCESS_CRITERION_PLAIN
    #: One separation stratum per entry of :attr:`systems`, from
    #: :func:`~sae_bioemu.cryptic_steering.conformer_separation.separation_stratum` at this cell's
    #: own ``tau``. Empty only when no ``tau`` was available to stratify at, which
    #: :meth:`as_dict` reports as ``null`` rather than as an absence of ambiguous systems.
    strata: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.criterion not in SUCCESS_CRITERIA:
            raise ValueError(f"criterion must be one of {SUCCESS_CRITERIA}; got {self.criterion!r}")
        if self.strata and len(self.strata) != len(self.systems):
            raise ValueError(
                f"strata carries {len(self.strata)} entries for {len(self.systems)} systems; a "
                "stratum is per system and a table that cannot be joined back to one is not a "
                "stratification"
            )

    @property
    def n_systems(self) -> int:
        return len(self.systems)

    @property
    def stratum_counts(self) -> dict[str, int] | None:
        """Systems per separation stratum, or ``None`` when this cell was not stratified."""
        if not self.strata:
            return None
        counts = dict.fromkeys(SEPARATION_STRATA, 0)
        for stratum in self.strata:
            counts[stratum] += 1
        return counts

    def rate_within(self, stratum: str) -> float:
        """The success rate restricted to one stratum. NaN when the stratum is empty.

        A READING of the result, never a filter applied to it: :attr:`rate` stays the rate over
        every admitted system, and this exists so the ambiguous band can be reported beside it
        rather than removed from it.
        """
        if stratum not in SEPARATION_STRATA:
            raise ValueError(f"stratum must be one of {SEPARATION_STRATA}; got {stratum!r}")
        if not self.strata:
            raise ValueError("this result carries no strata, so it cannot be read within one")
        selected = [
            success
            for success, value in zip(self.successes, self.strata, strict=True)
            if value == stratum
        ]
        return float(np.mean(selected)) if selected else float("nan")

    @property
    def rate(self) -> float:
        return float(np.mean(self.successes)) if self.successes else float("nan")

    @property
    def b_effective_min(self) -> int:
        return min(self.samples_per_system) if self.samples_per_system else 0

    @property
    def b_effective_max(self) -> int:
        return max(self.samples_per_system) if self.samples_per_system else 0

    def as_dict(self) -> dict[str, object]:
        return {
            "criterion": self.criterion,
            "criterion_definition": SUCCESS_CRITERION_DEFINITIONS[self.criterion],
            "reference_state": self.reference_state,
            "deviation_form": self.form,
            "tau_angstrom": self.tau,
            "b_requested": self.b_requested,
            "b_effective_min": self.b_effective_min,
            "b_effective_max": self.b_effective_max,
            "n_systems": self.n_systems,
            "n_successes": int(sum(self.successes)),
            "success_rate": self.rate,
            "ci_low": self.confidence_low,
            "ci_high": self.confidence_high,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "stratum_counts": self.stratum_counts,
            "rate_within_stratum": (
                None
                if not self.strata
                else {stratum: self.rate_within(stratum) for stratum in SEPARATION_STRATA}
            ),
            "systems": list(self.systems),
            "successes": [bool(value) for value in self.successes],
            "strata": list(self.strata),
        }


def bootstrap_interval(
    successes: Sequence[bool],
    *,
    resamples: int = 10000,
    seed: int = 20260905,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap over SYSTEMS, which are the independent units here.

    One system is one protein. Resampling samples within a system instead would treat 32 draws of
    one sequence as 32 proteins and shrink the interval by a factor that is pure leakage.
    """
    flags = np.asarray(list(successes), dtype=float)
    if flags.size == 0:
        return (float("nan"), float("nan"))
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, flags.size, size=(resamples, flags.size))
    rates = flags[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(rates, tail)),
        float(np.quantile(rates, 1.0 - tail)),
    )


def success_at_b(
    scores: Iterable[SystemScore],
    *,
    reference_state: str,
    form: str,
    tau: float,
    b: int | None,
    resamples: int = 10000,
    seed: int = 20260905,
    confidence: float = 0.95,
) -> SuccessResult:
    """A system succeeds when at least one of its first ``B`` samples is within ``tau``.

    THE FIELD'S CRITERION, UNCHANGED. This function's number is what ``bioemu-benchmarks``,
    ConforNets and ConforMix report and it is not touched by the specificity work: the only thing
    added here is the ``strata`` tag, which is descriptive and enters no success test. See
    :func:`specific_success_at_b` for the criterion that does differ, reported beside this one.
    """
    if form not in DEVIATION_FORMS:
        raise ValueError(f"form must be one of {DEVIATION_FORMS}")
    if reference_state not in REFERENCE_STATES:
        raise ValueError(f"reference_state must be one of {REFERENCE_STATES}")
    if b is not None and b < 1:
        raise ValueError("B must be at least 1")
    systems: list[str] = []
    successes: list[bool] = []
    counts: list[int] = []
    strata: list[str] = []
    for score in sorted(scores, key=lambda item: item.system):
        taken = take_b(score.samples(reference_state), b)
        if not taken:
            continue
        systems.append(score.system)
        counts.append(len(taken))
        successes.append(any(sample.value(form) <= tau for sample in taken))
        strata.append(score.separation_stratum(form=form, tau=tau))
    low, high = bootstrap_interval(successes, resamples=resamples, seed=seed, confidence=confidence)
    return SuccessResult(
        reference_state=reference_state,
        form=form,
        tau=float(tau),
        b_requested=b,
        systems=tuple(systems),
        successes=tuple(successes),
        samples_per_system=tuple(counts),
        strata=tuple(strata),
        criterion=SUCCESS_CRITERION_PLAIN,
        confidence_low=low,
        confidence_high=high,
        bootstrap_resamples=resamples,
        bootstrap_seed=seed,
    )


def specific_success_at_b(
    scores: Iterable[SystemScore],
    *,
    reference_state: str,
    form: str,
    tau: float,
    b: int | None,
    resamples: int = 10000,
    seed: int = 20260905,
    confidence: float = 0.95,
) -> SuccessResult:
    """THE SPECIFICITY CRITERION: within ``tau`` of the target AND NOT within ``tau`` of the other.

    A system succeeds when at least one of its first ``B`` samples satisfies BOTH conditions AT
    ONCE. The conjunction is per SAMPLE, not per system: "some sample reached holo and some other
    sample was far from apo" is a much weaker statement and is not this metric.

    **This does not redefine ``success@B``.** :func:`success_at_b` is untouched and stays the
    field-comparable number; this is an ADDITIONAL cell, tagged
    :data:`SUCCESS_CRITERION_SPECIFIC`, and a report must carry both under their own names. The
    module docstring is emphatic about silent substitution for a reason, and a specificity rate
    presented as ``success@B`` would not be comparable with any published figure.

    Carried through BOTH deviation forms, like everything else here: ``form`` selects which, and
    the threshold is applied to the same form on both sides of the conjunction so that the
    arithmetic mean's looser test is looser on both.

    The relationship to the admission rule, restated because the two are easy to confuse: above
    ``s > 2*tau`` the two ``tau``-balls are disjoint, so this criterion and the plain one give the
    SAME successes and the ``strata`` tag says so by reporting those systems as ``clean``. The two
    can only differ on ``ambiguous`` systems, and the gap between the rates is that band's
    contribution.
    """
    if form not in DEVIATION_FORMS:
        raise ValueError(f"form must be one of {DEVIATION_FORMS}")
    if reference_state not in REFERENCE_STATES:
        raise ValueError(f"reference_state must be one of {REFERENCE_STATES}")
    if b is not None and b < 1:
        raise ValueError("B must be at least 1")
    systems: list[str] = []
    successes: list[bool] = []
    counts: list[int] = []
    strata: list[str] = []
    for score in sorted(scores, key=lambda item: item.system):
        pairs = score.paired_samples(target_state=reference_state, b=b)
        if not pairs:
            continue
        systems.append(score.system)
        counts.append(len(pairs))
        successes.append(
            any(target.value(form) <= tau and other.value(form) > tau for target, other in pairs)
        )
        strata.append(score.separation_stratum(form=form, tau=tau))
    low, high = bootstrap_interval(successes, resamples=resamples, seed=seed, confidence=confidence)
    return SuccessResult(
        reference_state=reference_state,
        form=form,
        tau=float(tau),
        b_requested=b,
        systems=tuple(systems),
        successes=tuple(successes),
        samples_per_system=tuple(counts),
        strata=tuple(strata),
        criterion=SUCCESS_CRITERION_SPECIFIC,
        confidence_low=low,
        confidence_high=high,
        bootstrap_resamples=resamples,
        bootstrap_seed=seed,
    )


def coverage_at_adaptive_tau(
    scores: Iterable[SystemScore],
    *,
    reference_state: str,
    form: str,
    b: int | None,
    resamples: int = 10000,
    seed: int = 20260905,
    stratum_tau: float | None = None,
) -> SuccessResult:
    """ConforMix's secondary: per-system ``tau_i = 0.5 x`` the apo<->holo reference separation.

    Reported as a :class:`SuccessResult` with ``tau`` set to NaN, because there is no single tau:
    the threshold is adaptive and lives per system in
    :attr:`ReferenceSeparation.adaptive_tau`.

    ``stratum_tau`` is the POPULATION's tau -- the evaluation parameter of record -- and is used
    only to tag each system's separation band. It is not a threshold applied to anything here and
    it does not touch a single success. It is separate from ``tau`` because the adaptive threshold
    is per system: stratifying at ``tau_i = s_i/2`` would classify every system as ``clean`` by
    construction (``s_i > s_i/2`` always), which is a tautology dressed as a stratification. Left
    unset, the result carries no strata and says so.
    """
    systems: list[str] = []
    successes: list[bool] = []
    counts: list[int] = []
    strata: list[str] = []
    for score in sorted(scores, key=lambda item: item.system):
        taken = take_b(score.samples(reference_state), b)
        if not taken or score.reference_separation is None:
            continue
        threshold = score.reference_separation.adaptive_tau(form)
        systems.append(score.system)
        counts.append(len(taken))
        successes.append(any(sample.value(form) <= threshold for sample in taken))
        if stratum_tau is not None:
            strata.append(score.separation_stratum(form=form, tau=float(stratum_tau)))
    low, high = bootstrap_interval(successes, resamples=resamples, seed=seed)
    return SuccessResult(
        reference_state=reference_state,
        form=form,
        tau=float("nan"),
        b_requested=b,
        systems=tuple(systems),
        successes=tuple(successes),
        samples_per_system=tuple(counts),
        strata=tuple(strata),
        confidence_low=low,
        confidence_high=high,
        bootstrap_resamples=resamples,
        bootstrap_seed=seed,
    )


# --------------------------------------------------------------------------------------------
# Mapping generated samples onto benchmark systems
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleEntry:
    """One row of a BioEmu ``final_structures/index.jsonl``."""

    sequence_id: str
    sequence: str
    sample_index: int
    pdb_path: Path

    @property
    def pdb_key(self) -> tuple[str, str] | None:
        """``(PDB entry, chain)`` parsed from ``1abc_A``-style sequence ids, else ``None``."""
        parts = self.sequence_id.split("_")
        if len(parts) != 2 or len(parts[0]) != 4:
            return None
        return (parts[0].upper(), parts[1])


def read_sample_index(index_path: Path | str) -> list[SampleEntry]:
    """Read a final-structure index, resolving the PDB paths relative to the store root."""
    index_path = Path(index_path)
    root = index_path.parent.parent
    entries: list[SampleEntry] = []
    for line in index_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        entries.append(
            SampleEntry(
                sequence_id=str(record["sequence_id"]),
                sequence=str(record["sequence"]),
                sample_index=int(record["sample_index"]),
                pdb_path=root / str(record["pdb_path"]),
            )
        )
    return entries


def sequence_identity(left: str, right: str) -> float:
    """Fraction of the shorter sequence that letter-matched alignment places identically."""
    if not left or not right:
        return 0.0
    pairs = _letter_matched_alignment(left, right)
    return len(pairs) / min(len(left), len(right))


def map_samples_to_systems(
    entries: Sequence[SampleEntry],
    references: dict[str, dict[str, tuple[str, str]]],
    testcase_sequences: dict[str, str],
    identity_floor: float = DEFAULT_SEQUENCE_IDENTITY_FLOOR,
) -> tuple[dict[str, list[SampleEntry]], dict[str, str], int]:
    """Assign generated samples to benchmark systems by reference entry, then verify by sequence.

    ``references`` maps system -> ``{"apo": (PDB, chain), "holo": (PDB, chain)}``. A sample whose
    sequence id names one of those entries is a candidate; it is only accepted when its sequence
    also matches the benchmark test-case sequence to ``identity_floor``, because a pool entry can
    carry the same PDB entry and a different construct. A candidate that fails that check is
    refused by name rather than scored -- silently scoring a different construct against these
    references is precisely the failure this metric exists to avoid.
    """
    lookup: dict[tuple[str, str], str] = {}
    for system, states in references.items():
        for _state, key in states.items():
            lookup[(key[0].upper(), key[1])] = system

    matched: dict[str, list[SampleEntry]] = {}
    refused: dict[str, str] = {}
    outside = 0
    for entry in entries:
        key = entry.pdb_key
        system = lookup.get(key) if key is not None else None
        if system is None and entry.sequence_id in references:
            # A NATIVE BENCHMARK ensemble is keyed by the system itself (the test-case name, a
            # UniProt accession) rather than by a PDB entry and chain, because it is generated
            # from `testcases.csv` and never passes through the pool. Pool samples keep the
            # `<pdb>_<chain>` shape and reach this function through `lookup` above.
            #
            # This widens only how a CANDIDATE is proposed. The verification below is unchanged
            # and is what actually protects the metric: the sample's own sequence must still match
            # the benchmark test case, so naming a system cannot smuggle a different construct in.
            system = entry.sequence_id
        if system is None:
            outside += 1
            continue
        identity = sequence_identity(entry.sequence, testcase_sequences.get(system, ""))
        if identity < identity_floor:
            refused[entry.sequence_id] = (
                f"{system}: generated sequence matches the benchmark test case at "
                f"{identity:.3f} < {identity_floor:.3f} identity"
            )
            continue
        matched.setdefault(system, []).append(entry)
    return matched, refused, outside
