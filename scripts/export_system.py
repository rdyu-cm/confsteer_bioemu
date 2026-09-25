#!/usr/bin/env python3
"""Write a benchmark system's FASTA and its two reference conformers as PDB.

The benchmark sequences ship as JSON pools and the reference geometry as coordinate tensors, which
is what the generators and scorers read. Neither is what ``confsteer steer-apply`` takes, so this
turns one system into the files that command does take:

    python scripts/export_system.py --benchmark kinase --system abl1_kd --out-dir examples/abl1_kd
    confsteer steer-apply --adapter kinase_uncond_p1 --fasta examples/abl1_kd/abl1_kd.fasta \\
        --reference examples/abl1_kd/abl1_kd_dfg_out.pdb

**EACH REFERENCE COVERS THE RESIDUES ITS BENCHMARK'S METRIC IS DEFINED OVER, not the whole chain:**

    kinase          the activation loop (about 14 residues)
    cryptic pocket  the pocket region (about 90 residues)
    domain motion   every residue resolved in both deposited structures

Residues are numbered by SEQUENCE POSITION (1 = the first residue of the FASTA), so
``steer-apply --reference`` pairs each with the right residue of a draw and scores exactly that
region. Backbone atoms only (N, CA, C, O). Residues unresolved in a crystal are omitted rather
than written at the origin, where they would corrupt any RMSD measured against the file.

One draw scored this way is an illustration. The paper's numbers come from ``run_benchmark.py``.

CPU only, no network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
DATA = REPOSITORY / "data/benchmarks"

#: benchmark -> (reference artifact, the two state keys, the pool that carries every sequence)
BENCHMARKS = {
    "kinase": ("references/kinase.pt", ("dfg_in", "dfg_out"), "pools/kinase.json"),
    "domain_motion": (
        "references/domain_motion_benchmark.pt",
        ("first", "second"),
        "pools/domain_motion.json",
    ),
    "domain_motion_test": (
        "references/domain_motion_testsplit.pt",
        ("first", "second"),
        None,
    ),
    "cryptic_test": ("references/cryptic_testsplit.pt", ("apo", "holo"), None),
    "cryptic": ("references/cryptic_benchmark.pt", ("apo", "holo"), "pools/cryptic_benchmark.json"),
}

#: What a level means for each benchmark, so the exported file names say which arm targets which.
ARM_FOR_STATE = {
    "dfg_in": "kinase_uncond_m1",
    "dfg_out": "kinase_uncond_p1",
    "first": "domain_motion_uncond_m1",
    "second": "domain_motion_uncond_p1",
    "apo": "cryptic_uncond_m1",
    "holo": "cryptic_uncond_p1",
}

BACKBONE = ("N", "CA", "C", "O")


def cryptic_benchmark_frames(system: str) -> tuple[str, list[int], dict]:
    """Build the cryptic BENCHMARK reference from the deposited entries, on sequence positions.

    This is how ``references/cryptic_benchmark.pt`` was built, and is kept so it can be rebuilt from
    public data: the export itself reads that artifact. The references are the deposited apo and holo
    chains, fetched by ``scripts/release/fetch_structures.py`` and paired by ``cryptic/pairing.csv``. They
    arrive in the depositors' own numbering, which the benchmark's own README warns may differ from
    the residue numbers its ranges use, so the chain is placed on the test-case sequence by the
    SAME letter-matched alignment the scorer uses (``benchmark_frame``) and written out on sequence
    positions. Reusing the scorer's alignment is the point: a reference aligned any other way would
    not be the thing the reported number is measured against.

    Returns ``(sequence, positions, {state: [len(positions), 4, 3] angstrom})`` over the metric
    region, keeping only positions BOTH deposited chains resolve -- an unresolved residue has no
    coordinates, and filling one in would put an atom where the crystal says nothing.
    """
    import csv

    import numpy

    from sae_bioemu.cryptic_steering.success_at_b import (
        BACKBONE_ATOMS as SCORER_ATOMS,
    )
    from sae_bioemu.cryptic_steering.success_at_b import (
        benchmark_frame,
        load_benchmark_ranges,
        read_reference_chain,
    )

    pairing = {
        row["accession"]: row
        for row in csv.DictReader((DATA / "cryptic/pairing.csv").read_text().splitlines())
    }
    row = pairing.get(system)
    if row is None:
        raise SystemExit(
            f"REFUSING: {system!r} is not in cryptic/pairing.csv. It has {len(pairing)} systems, "
            f"e.g. {sorted(pairing)[:4]}"
        )
    ranges, _derived = load_benchmark_ranges(DATA / "cryptic/metric_ranges.json")
    if system not in ranges:
        raise SystemExit(
            f"REFUSING: {system} has no metric ranges, so its metric region is unknown"
        )
    # The test-case sequences live in their own table, which is what the scorer reads too. The
    # second value from load_benchmark_ranges is a derived-pocket note, not a sequence map.
    sequences = {
        row["test_case"]: row["sequence"]
        for row in csv.DictReader((DATA / "cryptic/sequences.csv").read_text().splitlines())
    }
    if system not in sequences:
        raise SystemExit(f"REFUSING: {system} has no sequence in cryptic/sequences.csv")
    sequence = sequences[system]

    frames: dict[str, object] = {}
    resolved: dict[str, dict[int, object]] = {}
    for state in ("apo", "holo"):
        entry = row[f"{state}_pdb_id"]
        path = DATA / "structures" / f"{entry.lower()}.cif"
        if not path.is_file():
            raise SystemExit(
                f"REFUSING: {path} is absent. The cryptic benchmark references are deposited "
                "entries; fetch them once with scripts/release/fetch_structures.py"
            )
        chain = read_reference_chain(path, row[f"{state}_auth_chain"])
        frame = benchmark_frame(chain, sequence)
        resolved[state] = {
            position: chain.residues[index]
            for position, index in frame.position_to_index.items()
            if all(name in chain.residues[index].atoms for name in SCORER_ATOMS)
        }
    positions = [
        position
        for position in ranges[system].metric_positions
        if position in resolved["apo"] and position in resolved["holo"]
    ]
    if not positions:
        raise SystemExit(
            f"REFUSING: neither deposited chain of {system} resolves any of its metric residues"
        )
    for state in ("apo", "holo"):
        frames[state] = numpy.asarray(
            [
                [resolved[state][position].atoms[name] for name in SCORER_ATOMS]
                for position in positions
            ],
            dtype=float,
        )
    requested = len(ranges[system].metric_positions)
    print(
        f"metric region: {len(positions)} of {requested} positions resolved in both "
        "deposited chains"
    )
    return sequence, [position - 1 for position in positions], frames


def metric_frame(entry, states) -> tuple[list[int], dict]:
    """Return the metric residue INDICES and each state's coordinates over exactly those residues.

    Two conventions are in the artifacts and both are handled, because silently reading the wrong
    one gives a full-length array of plausible coordinates:

    * kinase and cryptic store ONLY the metric residues, so ``coordinates[i]`` is sequence
      position ``indices[i]``;
    * domain motion stores the WHOLE chain, so the metric residues are ``coordinates[indices]``.

    The distinction matters beyond indexing: a domain-motion chain carries residues that were
    unresolved in the crystal and are stored as zeros, and ``metric_residues`` is exactly the set
    that excludes them. Writing all of them would put atoms at the origin and corrupt any RMSD
    measured against the file.
    """
    indices = [int(value) for value in (entry.get("metric_indices") or entry["metric_residues"])]
    frames = {}
    for state in states:
        coordinates = entry[state]
        if coordinates.shape[0] == len(indices):
            frames[state] = coordinates
        elif coordinates.shape[0] == len(entry["sequence"]):
            frames[state] = coordinates[indices]
        else:
            raise SystemExit(
                f"REFUSING: {state} has {coordinates.shape[0]} residues, which matches neither "
                f"the {len(indices)} metric residues nor the {len(entry['sequence'])}-residue "
                "sequence; the indexing convention is unknown and guessing it is how a confident "
                "wrong structure gets written"
            )
    return indices, frames


def write_pdb(path: Path, coordinates, indices: list[int], sequence: str) -> int:
    """Write N/CA/C/O backbone atoms as a PDB, numbering residues by SEQUENCE POSITION.

    ``resSeq`` is the 1-based index into the sequence, not a counter over the written residues, so
    a partial reference states which residues it covers and a reader can map it back. That is what
    lets ``steer-apply --reference`` score a metric region rather than requiring a whole chain.

    Columns follow the PDB spec exactly -- atom name 13-16, altLoc 17, residue 18-20, chain 22,
    resSeq 23-26, coordinates from 31 -- because mdtraj rejects a misaligned residue name outright.
    """
    lines = []
    for serial, (position, index) in enumerate(enumerate(indices), start=1):
        residue = _three_letter(sequence[index])
        for offset, atom in enumerate(BACKBONE):
            x, y, z = (float(value) for value in coordinates[position][offset])
            lines.append(
                # 1-6 record, 7-11 serial, 13-16 atom, 17 altLoc, 18-20 residue, 22 chain,
                # 23-26 resSeq, 31-38/39-46/47-54 xyz. One missing space here and mdtraj refuses
                # the file with "Misaligned residue name".
                f"ATOM  {serial * 4 - 4 + offset + 1:>5d} {atom:<4s} {residue:>3s} "
                f"A{index + 1:>4d}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00"
                f"          {atom[0]:>2s}"
            )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return len(indices)


_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}  # fmt: skip


def _three_letter(letter: str) -> str:
    return _THREE.get(letter.upper(), "UNK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument("--system", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    arguments = parser.parse_args()

    relative, states, pool = BENCHMARKS[arguments.benchmark]
    entry: dict | None = None
    if relative is None:
        sequence, indices, frames = cryptic_benchmark_frames(arguments.system)
    else:
        import torch

        payload = torch.load(DATA / relative, map_location="cpu", weights_only=False)
        entries = payload["entries"]
        entry = entries.get(arguments.system)
        if entry is None:
            raise SystemExit(
                f"REFUSING: {arguments.system!r} is not in {arguments.benchmark}. It has "
                f"{len(entries)} systems, e.g. {sorted(entries)[:4]}"
            )
        sequence = str(entry["sequence"])
    out = arguments.out_dir
    out.mkdir(parents=True, exist_ok=True)
    fasta = out / f"{arguments.system}.fasta"
    fasta.write_text(
        f">{arguments.system}\n"
        + "\n".join(sequence[i : i + 60] for i in range(0, len(sequence), 60))
        + "\n"
    )
    print(f"wrote {fasta}")

    if entry is not None:
        indices, frames = metric_frame(entry, states)
    for state in states:
        target = out / f"{arguments.system}_{state}.pdb"
        write_pdb(target, frames[state], indices, sequence)
        print(f"wrote {target}  (target of {ARM_FOR_STATE[state]})")

    if pool is not None:
        names = [e["sequence_id"] for e in json.loads((DATA / pool).read_text())["entries"]]
        if arguments.system not in names:
            print(f"  note: {arguments.system} is in the reference artifact but not in {pool}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
