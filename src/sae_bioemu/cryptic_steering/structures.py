"""Experimental structure reading and the apo/holo/BioEmu residue correspondence.

This module owns the project's highest-risk step. Three indexing frames must be reconciled -- the
apo PDB's ``auth_seq_id``, the holo PDB's, and position in the sequence string handed to BioEmu --
and a mistake here fails silently: a shifted pocket mask still yields a plausible progress
coordinate and a plausible localisation score. Measured on 25 CryptoBench pairs, a one-residue slip
leaves the transition vector at a median cosine of about 0.7 with the truth, and above 0.9 for
roughly a fifth of proteins, so an off-by-one degrades part of the pool without ever failing loudly.

The defence is redundancy: build the map three independent ways and require them to agree, then
assert the round trip closes. :func:`residue_correspondence` fails closed rather than returning a
best guess. :func:`round_trip_closes` states precisely which part of R0 it does and does not
cover -- the part it cannot cover, that the FASTA reaching the generator is the string these
positions index, is closed by the pocket artifact's per-sequence digest.

The map is only authoritative where CryptoBench annotates, which is the pocket. Everything else,
including the whole context shell, is mapped by :func:`holo_partner_positions` through the
alignment. That matters because ``pocket_context`` carries the transition: a shell residue mapped
by ``auth_seq_id`` alone can write a fabricated displacement into the direction being steered
toward.

Both directions of that map are guarded, and the second guard was added after its absence
published. :func:`round_trip_closes` requires the apo side to be bijective on observed residues;
:func:`holo_partner_positions` requires the apo->holo map itself to be **injective** and raises
:class:`NonInjectiveCorrespondenceError` otherwise. Without it two apo residues could be gathered
onto one holo residue -- one by the dataset pairing, one by the alignment, letters agreeing so the
letter guard passed -- and the pool of record carried the resulting fabricated transitions on 138 of
its 2123 split proteins.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Sentinel returned instead of a letter when a residue is not one of the twenty standard types.
_UNKNOWN = ""

#: Weights of the standard virtual C-beta construction, placing C-beta from the backbone frame.
#:
#: Using this rather than the deposited C-beta is deliberate on three counts. It is defined for
#: glycine, which has no C-beta and is 7.8% of pocket residues. It is defined where a crystal
#: simply did not model the atom (0.12% of the rest). And it is exactly what BioEmu emits, since
#: the model produces a per-residue rigid frame and places C-beta from it at a fixed offset -- so
#: target and generated structures carry the same quantity rather than two similar ones.
#:
#: This costs almost nothing in fidelity: measured against 22,099 deposited C-beta atoms, the
#: construction agrees to a median of 0.069 A, with 0.10% above the 0.5 A coordinate-uncertainty
#: floor. C-beta is fixed by the backbone -- chi1 rotates about the CA-CB bond and does not move
#: it -- which is why the agreement is so tight.
_VIRTUAL_CB_WEIGHTS = (-0.58273431, 0.56802827, -0.54067466)

#: The atom sets the coordinate may be built on. ``cb`` carries the residue's frame orientation and
#: measures roughly 1.48x the transition magnitude of ``ca`` (above 1 for 96.5% of proteins).
ATOM_SETS = ("ca", "cb")

#: The four atoms the BENCHMARK's deviation is measured over, in the benchmark's own per-residue
#: order. Deliberately NOT a member of :data:`ATOM_SETS`: those name a one-atom-per-residue
#: progress coordinate and this names a four-atom deviation set, so a caller cannot ask for
#: "the atom set" and be handed a tensor of a different rank. Spelled identically to
#: :data:`sae_bioemu.cryptic_steering.direct_metric.CRYPTIC_METRIC_ATOMS` and to
#: :data:`sae_bioemu.cryptic_steering.success_at_b.BACKBONE_ATOMS`, and a test pins all three
#: equal: a reference read in one order against a draw built in another is a silent per-residue
#: permutation that leaves every deviation finite and wrong.
BACKBONE_ATOMS: tuple[str, ...] = ("N", "CA", "C", "O")

#: How an apo residue's holo partner was determined, most trustworthy first. ``dataset`` is
#: CryptoBench's own index-wise pairing of its two pocket selections; ``alignment`` is the pairwise
#: sequence alignment; ``auth`` is bare ``auth_seq_id`` identity, which is only ever a last resort.
PARTNER_SOURCES = ("dataset", "alignment", "auth")


@dataclass(frozen=True)
class PartnerCollision:
    """Two or more apo residues claiming ONE holo residue as their partner."""

    #: Position of the contested holo residue in the holo residue list, and its PDB key.
    holo_index: int
    holo_key: str
    holo_letter: str
    #: Positions of the apo residues that claimed it, ascending, and their PDB keys and letters.
    apo_indices: tuple[int, ...]
    apo_keys: tuple[str, ...]
    apo_letters: tuple[str, ...]
    #: Which of :data:`PARTNER_SOURCES` produced each claim, in the same order.
    sources: tuple[str, ...]

    def describe(self) -> str:
        claims = ", ".join(
            f"apo {key} (index {index}, {letter}, by {source})"
            for key, index, letter, source in zip(
                self.apo_keys, self.apo_indices, self.apo_letters, self.sources, strict=True
            )
        )
        return f"holo {self.holo_key} ({self.holo_letter}) claimed by {claims}"


class NonInjectiveCorrespondenceError(ValueError):
    """Raised when the apo->holo partner map would gather two apo residues onto one holo residue.

    This is a ``ValueError`` so that callers already written to reject an unusable correspondence
    keep rejecting it, and it carries :attr:`collisions` so a funnel can name the residues rather
    than only counting the protein.
    """

    def __init__(self, collisions: Sequence[PartnerCollision]) -> None:
        self.collisions = tuple(collisions)
        detail = "; ".join(collision.describe() for collision in self.collisions)
        super().__init__(
            f"apo->holo partner map is not injective in {len(self.collisions)} place(s): {detail}"
        )


def virtual_cb(nitrogen: np.ndarray, alpha_carbon: np.ndarray, carbon: np.ndarray) -> np.ndarray:
    """Place C-beta from the backbone frame, for any residue including glycine."""
    first = alpha_carbon - nitrogen
    second = carbon - alpha_carbon
    normal = np.cross(first, second)
    wa, wb, wc = _VIRTUAL_CB_WEIGHTS
    return wa * normal + wb * first + wc * second + alpha_carbon


@dataclass(frozen=True)
class ObservedResidue:
    """One residue with coordinates, in its own structure's auth numbering."""

    auth_seq_id: int
    insertion_code: str
    resname: str
    letter: str
    ca: np.ndarray
    #: The deposited C-beta, absent for glycine and for the rare unmodelled case.
    cb: np.ndarray | None
    #: C-beta placed from the backbone frame. Defined for every residue with a complete
    #: backbone, and the quantity BioEmu itself produces.
    cb_virtual: np.ndarray | None
    #: ``[4, 3]`` N/CA/C/O in deposition order, or ``None`` when any of the four is unmodelled.
    #: NOT part of :meth:`coordinate` and not a member of :data:`ATOM_SETS`: the progress
    #: coordinate is a one-atom-per-residue quantity and this is the four-atom set the BENCHMARK's
    #: deviation is measured over, so mixing them would let a caller ask for "the atom set" and get
    #: a different rank of tensor. ``O`` is included and is why this cannot be reconstructed from
    #: :attr:`cb_virtual`'s inputs: the carbonyl is a real deposited atom here, not a placed one.
    #: Defaulted so that every existing construction of this dataclass is unchanged.
    backbone: np.ndarray | None = None

    @property
    def key(self) -> str:
        """The ``auth_seq_id`` plus insertion code, matching CryptoBench's selection format."""
        return f"{self.auth_seq_id}{self.insertion_code.strip()}"

    def coordinate(self, atom_set: str) -> np.ndarray | None:
        """Return this residue's coordinate for the requested atom set."""
        if atom_set == "ca":
            return self.ca
        if atom_set == "cb":
            return self.cb_virtual
        raise ValueError(f"atom_set must be one of {ATOM_SETS}")


@dataclass(frozen=True)
class Correspondence:
    """A verified apo->holo residue map plus the evidence that it is trustworthy.

    ``sequence`` is the one-letter string that would be handed to BioEmu, and ``apo_positions``
    maps an apo residue key to its index in that string. Those two together are what convert a
    CryptoBench pocket selection into the residue indices the objective uses.
    """

    #: The dataset's index-wise pairing of the two pocket selections. It covers the POCKET ONLY,
    #: because that is all CryptoBench annotates; ``.get`` on any other residue returns ``None``.
    apo_to_holo: dict[str, str]
    #: The pairwise-alignment map over the whole chain, letter-matched. This is what the context
    #: shell must be mapped by: ``auth_seq_id`` identity held on only 20 of 24 sampled pairs.
    alignment: dict[str, str]
    apo_positions: dict[str, int]
    sequence: str
    agreement_auth: int
    agreement_alignment: int
    #: Pocket residues the alignment declined to map at all, as opposed to mapping differently.
    #: Almost always a point mutation between the two entries or an indel nearby: the alignment
    #: only keeps positions whose residue letters match, so it abstains rather than guessing.
    abstain_alignment: int
    #: Pocket residues where the alignment named a DIFFERENT holo partner than the dataset. This
    #: is the only genuine contradiction, and the only one that should cost a protein its place.
    contradict_alignment: int
    checked: int
    round_trip_ok: bool

    @property
    def fully_agreed(self) -> bool:
        """True when nothing contradicts the dataset pairing and the round trip closes.

        Abstention is deliberately not disqualifying. Treating it as disagreement drops roughly
        one protein in ten, and does so non-randomly: the abstaining cases are point mutants and
        indel-bearing constructs, which is exactly the population cryptic-pocket work cares about
        -- the published opening-probability benchmarks are built on mutant series.
        """
        return self.checked > 0 and self.contradict_alignment == 0 and self.round_trip_ok


def read_chain(path: Path | str, chain_id: str) -> list[ObservedResidue]:
    """Return one auth chain's observed standard protein residues, in deposition order.

    Biopython's ``MMCIFParser`` keys chains by ``auth_asym_id`` and residues by ``auth_seq_id``,
    which is the convention CryptoBench's selections use, so no translation happens here. Residues
    without a C-alpha are skipped: they carry no usable geometry.
    """
    from Bio.PDB import MMCIFParser
    from Bio.PDB.Polypeptide import protein_letters_3to1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = MMCIFParser(QUIET=True).get_structure(Path(path).stem, str(path))

    try:
        chain = structure[0][chain_id]
    except KeyError:
        return []

    residues: list[ObservedResidue] = []
    for residue in chain:
        hetflag, seq_id, icode = residue.id
        if hetflag.strip() or "CA" not in residue:
            continue
        backbone_complete = all(atom in residue for atom in ("N", "CA", "C"))
        # Bio.PDB.Polypeptide keys this dict by uppercase three-letter codes, unlike
        # Bio.Data.IUPACData, which keys the capitalized form. Getting this wrong silently
        # discards every residue.
        letter = protein_letters_3to1.get(residue.get_resname().strip().upper(), _UNKNOWN)
        if len(letter) != 1:
            continue
        residues.append(
            ObservedResidue(
                auth_seq_id=int(seq_id),
                insertion_code=str(icode),
                resname=residue.get_resname(),
                letter=letter,
                ca=np.asarray(residue["CA"].get_coord(), dtype=float),
                cb=(
                    np.asarray(residue["CB"].get_coord(), dtype=float) if "CB" in residue else None
                ),
                cb_virtual=(
                    virtual_cb(
                        np.asarray(residue["N"].get_coord(), dtype=float),
                        np.asarray(residue["CA"].get_coord(), dtype=float),
                        np.asarray(residue["C"].get_coord(), dtype=float),
                    )
                    if backbone_complete
                    else None
                ),
                # ``None`` unless ALL FOUR are modelled, including the carbonyl oxygen that
                # ``backbone_complete`` above does not require. A residue missing one atom of the
                # metric's set has no deviation defined over it, and substituting a placed atom
                # would compare a reconstruction against a crystal on that row alone.
                backbone=(
                    np.stack(
                        [
                            np.asarray(residue[name].get_coord(), dtype=float)
                            for name in BACKBONE_ATOMS
                        ]
                    )
                    if all(name in residue for name in BACKBONE_ATOMS)
                    else None
                ),
            )
        )
    return residues


def alignment_map(apo: list[ObservedResidue], holo: list[ObservedResidue]) -> dict[str, str]:
    """Map apo residue keys to holo keys by global pairwise sequence alignment.

    Only aligned positions whose residue letters match are kept, so an alignment that slips through
    a divergent region cannot contribute a false correspondence.
    """
    from Bio.Align import PairwiseAligner

    if not apo or not holo:
        return {}

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0

    alignment = aligner.align(
        "".join(residue.letter for residue in apo),
        "".join(residue.letter for residue in holo),
    )[0]

    mapping: dict[str, str] = {}
    for (apo_start, apo_end), (holo_start, _) in zip(*alignment.aligned, strict=False):
        for offset in range(apo_end - apo_start):
            left = apo[apo_start + offset]
            right = holo[holo_start + offset]
            if left.letter == right.letter:
                mapping[left.key] = right.key
    return mapping


def selection_keys(selection: list[str]) -> list[str]:
    """Strip the ``{auth_asym_id}_`` prefix from CryptoBench selection entries."""
    return [entry.split("_", 1)[1] for entry in selection]


def round_trip_closes(
    apo: list[ObservedResidue],
    sequence: str,
    apo_positions: dict[str, int],
    keys: Sequence[str],
) -> bool:
    """Check pocket key -> index in the BioEmu sequence -> back to PDB numbering -> identity.

    An earlier version of this check read ``apo[apo_positions[key]].key == key`` against a
    dictionary built two lines above from the same list. That is ``{k: i}[k] == k``: it could not
    fail, and it was documented as the mitigation for the project's top risk. Three properties are
    checked here instead, and each of them can actually fail:

    * the map is **bijective on observed residues** (``len(apo_positions) == len(apo)``). Two
      residues sharing an ``auth_seq_id`` and insertion code collapse into a single dictionary
      entry, which makes the earlier one unreachable and silently rehomes any pocket key naming it
      onto the later one's coordinates.
    * the sequence is **the same length** as the residue list whose positions index it, so a mask
      built from those positions addresses the string handed to BioEmu rather than some truncation
      or filtering of it.
    * for every pocket key, the residue **named** by that key and the residue **at** that index are
      the same residue, and the letter at that index of the sequence string is that residue's
      letter. This is the link that ties a PDB-numbered annotation to a sequence offset.

    Honest scope, because this function is cited as evidence. When
    :func:`residue_correspondence` calls it, the map and the sequence are both built from ``apo``
    inside that function, so the clause that can fire there is bijectivity; the other two fire for
    any caller that builds the map or the sequence separately, which is what the regression tests
    do and what a future SEQRES-based sequence would hit immediately. What is **not** covered at
    this layer at all is that the FASTA actually reaching the generator is this string -- that
    crosses a file boundary and is closed by the per-sequence digest in
    :mod:`sae_bioemu.cryptic_steering.artifact`, not here.
    """
    if len(apo_positions) != len(apo) or len(sequence) != len(apo):
        return False
    keys_in_order = [residue.key for residue in apo]
    for key in keys:
        position = apo_positions.get(key)
        if position is None or not 0 <= position < len(apo):
            return False
        if keys_in_order[position] != key or sequence[position] != apo[position].letter:
            return False
    return True


def holo_partner_positions(
    apo: list[ObservedResidue],
    holo: list[ObservedResidue],
    correspondence: Correspondence,
) -> tuple[dict[int, int], dict[int, str]]:
    """Map every apo residue's position to its holo partner's position, and record how.

    Pocket residues come from the dataset's own pairing, which is authoritative and is allowed to
    pair residues whose letters differ -- point mutants are a wanted population, not an error.
    Every other residue, which is to say the whole context shell, is mapped by the pairwise
    sequence alignment, falling back to ``auth_seq_id`` identity only when the alignment abstains
    **and** the two residue letters agree.

    Mapping the shell by ``auth_seq_id`` alone is not safe, and this function exists because it was
    being done. Identity held on only 20 of 24 sampled pairs, and 530 of CryptoBench's 5493 pairs
    carry differing apo/holo numbering. ``pocket_context`` is a TRANSITION class, so a mis-mapped
    shell residue does not merely add noise to a penalty: it writes a fabricated displacement into
    ``v``, the direction the adapter is trained to move along.

    **The map must be injective, and this function raises rather than returning one that is not.**
    Two apo residues gathered onto one holo residue produce two byte-identical holo coordinate rows,
    so every pair between them gets ``d_holo = 0`` and therefore ``v = -d_apo`` -- a fabricated
    transition, on a class that carries a target -- and every other pair touching either residue
    carries a transition measured to the wrong partner. All of it is finite and plausible, and every
    downstream verification reproduces it, because the pool it is verified against was built through
    this same function. A silent many-to-one map is how a fabricated target reached a fingerprinted
    artifact of record on 138 of 2123 split proteins; see FINDINGS.md.

    Raises :class:`NonInjectiveCorrespondenceError`, naming the apo residues that collide, the holo
    residue they collide on, and which source proposed each claim.
    """
    holo_positions = {residue.key: index for index, residue in enumerate(holo)}
    partners: dict[int, int] = {}
    sources: dict[int, str] = {}
    for apo_index, residue in enumerate(apo):
        key = residue.key
        proposed = correspondence.apo_to_holo.get(key)
        source = "dataset"
        if proposed is None:
            proposed = correspondence.alignment.get(key)
            source = "alignment"
        if proposed is None and key in holo_positions:
            proposed = key
            source = "auth"
        holo_index = holo_positions.get(proposed) if proposed is not None else None
        if holo_index is None:
            continue
        if source != "dataset" and holo[holo_index].letter != residue.letter:
            continue
        partners[apo_index] = holo_index
        sources[apo_index] = source

    claimants: dict[int, list[int]] = {}
    for apo_index, holo_index in partners.items():
        claimants.setdefault(holo_index, []).append(apo_index)
    collisions = [
        PartnerCollision(
            holo_index=holo_index,
            holo_key=holo[holo_index].key,
            holo_letter=holo[holo_index].letter,
            apo_indices=tuple(sorted(apo_indices)),
            apo_keys=tuple(apo[index].key for index in sorted(apo_indices)),
            apo_letters=tuple(apo[index].letter for index in sorted(apo_indices)),
            sources=tuple(sources[index] for index in sorted(apo_indices)),
        )
        for holo_index, apo_indices in sorted(claimants.items())
        if len(apo_indices) > 1
    ]
    if collisions:
        raise NonInjectiveCorrespondenceError(collisions)
    return partners, sources


def residue_correspondence(
    apo: list[ObservedResidue],
    holo: list[ObservedResidue],
    apo_selection: list[str],
    holo_selection: list[str],
) -> Correspondence:
    """Build the apo->holo map three ways and record how far they agree.

    The three sources are independent: the dataset's own index-wise pairing of its two selections,
    identity of ``auth_seq_id`` between the structures, and a pairwise sequence alignment. The
    dataset pairing is authoritative -- CryptoBench emits the same residue set in two numbering
    frames, verified as identical in length for 5491 of its 5493 pairs -- but ``auth_seq_id``
    identity is *not* a safe shortcut: it disagreed on 4 of 24 sampled pairs.

    The caller decides what to do with partial agreement; this function reports rather than
    silently repairing, because a repaired map is exactly the failure mode that publishes.
    """
    if len(apo_selection) != len(holo_selection):
        raise ValueError(
            "apo and holo pocket selections must have equal length to be paired index-wise; "
            f"got {len(apo_selection)} and {len(holo_selection)}"
        )

    apo_positions = {residue.key: index for index, residue in enumerate(apo)}
    holo_positions = {residue.key: index for index, residue in enumerate(holo)}

    apo_keys = selection_keys(apo_selection)
    holo_keys = selection_keys(holo_selection)
    dataset_map = dict(zip(apo_keys, holo_keys, strict=True))
    aligned = alignment_map(apo, holo)

    resolved = [
        key for key in apo_keys if key in apo_positions and dataset_map[key] in holo_positions
    ]
    agreement_auth = sum(1 for key in resolved if key in holo_positions and dataset_map[key] == key)

    # Three outcomes, not two. The alignment can agree, abstain (no mapping for this residue,
    # because a point mutation or a neighbouring indel means its letters do not match), or
    # contradict (name a different holo partner). Only contradiction is evidence that the
    # dataset pairing is wrong.
    agreement_alignment = 0
    abstain_alignment = 0
    contradict_alignment = 0
    for key in resolved:
        proposed = aligned.get(key)
        if proposed is None:
            abstain_alignment += 1
        elif proposed == dataset_map[key]:
            agreement_alignment += 1
        else:
            contradict_alignment += 1

    sequence = "".join(residue.letter for residue in apo)
    round_trip_ok = round_trip_closes(apo, sequence, apo_positions, resolved)

    return Correspondence(
        apo_to_holo=dataset_map,
        alignment=aligned,
        apo_positions=apo_positions,
        sequence=sequence,
        agreement_auth=agreement_auth,
        agreement_alignment=agreement_alignment,
        abstain_alignment=abstain_alignment,
        contradict_alignment=contradict_alignment,
        checked=len(resolved),
        round_trip_ok=round_trip_ok,
    )


def context_shell(
    coordinates: np.ndarray,
    pocket_indices: list[int],
    radius_angstrom: float,
    *,
    eligible: Sequence[bool] | None = None,
) -> tuple[int, ...]:
    """Return residues within ``radius_angstrom`` of the pocket, excluding the pocket itself.

    The shell is deliberately local. Using the whole protein as context would dilute the pocket
    signal by chain length -- the same order-L dilution that makes the global geometry penalty
    roughly 250x weaker for a local coordinate than for radius of gyration at L around 250.

    ``eligible`` marks residues whose row of ``coordinates`` is real. Rows for residues without a
    usable coordinate are placeholders, and a placeholder at the origin would join or leave the
    shell on a fabricated distance, so ineligible residues are excluded outright rather than
    measured.
    """
    if radius_angstrom <= 0:
        raise ValueError("radius_angstrom must be positive")
    pocket = set(pocket_indices)
    if not pocket:
        raise ValueError("pocket_indices must be non-empty")
    if eligible is None:
        usable = [True] * int(coordinates.shape[0])
    else:
        usable = [bool(value) for value in eligible]
        if len(usable) != int(coordinates.shape[0]):
            raise ValueError("eligible must carry one flag per residue")
        if any(not usable[index] for index in pocket):
            raise ValueError("every pocket residue must have a usable coordinate")

    pocket_coordinates = coordinates[sorted(pocket)]
    differences = coordinates[:, None, :] - pocket_coordinates[None, :, :]
    nearest = np.sqrt((differences**2).sum(-1)).min(axis=1)
    return tuple(
        index
        for index in range(coordinates.shape[0])
        if index not in pocket and usable[index] and nearest[index] <= radius_angstrom
    )
