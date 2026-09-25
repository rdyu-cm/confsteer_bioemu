"""Emit one native-format ``index.jsonl`` per LEVEL from a cryptic steered store.

WHY AN ADAPTER IS NEEDED
------------------------
``score_success_at_b.py`` consumes a BioEmu ``final_structures/index.jsonl``: a flat list of
``(sequence_id, sequence, sample_index, npz_path, pdb_path)``. The cryptic steered generator does
not write one. It writes ``steered/pass_<nn>/final_structures/index.jsonl`` -- one directory per
(system, level) pass -- plus a master sidecar ``steered_draws.jsonl`` that is the only place the
LEVEL of each draw is recorded.

So the structures are already in the right format; they are merely split across passes and carry no
level. This regroups them. The per-pass row schema and the native row schema are IDENTICAL
(verified: same seven keys), so nothing is translated, only gathered.

TWO CHOICES THAT MATTER
-----------------------
**Absolute paths.** ``read_sample_index`` resolves ``root / pdb_path`` with
``root = index.parent.parent``, and a level's draws come from many pass directories, so no single
relative root exists. ``pathlib`` returns the right-hand side unchanged when it is absolute, so
absolute paths are the honest way to express "these live in several places".

**sample_index = draw_seed.** Not a running counter. The steered arm was generated at the SAME
SEEDS as the native control (0..99), and ``take_b`` selects the FIRST B draws in sample-index
order. Keying on the seed means steered sample_index N and native sample_index N are the same
draw, so a B-subset of one is the seed-matched B-subset of the other. A running counter would
silently break that pairing while producing a perfectly plausible number.

REFUSES on a level whose systems do not all have the same depth, because a ragged pool makes
``pool_size_min`` govern every cell, and on a draw whose structure file is missing.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

LEVEL_NAMES = {-1.0: "level_m1", 1.0: "level_p1"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path, help="the steered store root")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--require-systems",
        type=int,
        default=None,
        help="refuse unless each level covers exactly this many systems",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help=(
            "trim every (level, system) to the N LOWEST-NUMBERED SEEDS, making a ragged store "
            "uniform without discarding systems. Refuses if any (level, system) holds fewer "
            "than N."
        ),
    )
    args = parser.parse_args()

    sidecar = args.store / "steered_draws.jsonl"
    if not sidecar.exists():
        raise SystemExit(f"REFUSING: {sidecar} absent; the level of each draw is unknown")

    # sequence per (pass, system), read from the pass indexes -- the sidecar does not carry it.
    seq_of: dict[tuple[int, str], str] = {}
    for pass_index_path in sorted(args.store.glob("steered/pass_*/final_structures/index.jsonl")):
        pass_no = int(pass_index_path.parent.parent.name.split("_")[1])
        for line in pass_index_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seq_of[(pass_no, str(row["sequence_id"]))] = str(row["sequence"])

    by_level: dict[float, list[dict]] = collections.defaultdict(list)
    for line in sidecar.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row["sequence_id"])
        if sid.startswith("__"):
            continue
        level = float(row["level"])
        pass_no = int(row["pass_index"])
        npz = (args.store / str(row["steered_npz"])).resolve()
        pdb = (args.store / str(row["steered_pdb"])).resolve()
        if not pdb.exists():
            raise SystemExit(f"REFUSING: {pdb} is named by the sidecar but absent from disk")
        sequence = seq_of.get((pass_no, sid))
        if sequence is None:
            raise SystemExit(f"REFUSING: no pass index carries a sequence for {sid} pass {pass_no}")
        by_level[level].append(
            {
                "sequence_id": sid,
                "sequence": sequence,
                # THE SEED, so a B-subset is seed-matched to the native arm's.
                "sample_index": int(row["draw_seed"]),
                "npz_path": str(npz),
                "pdb_path": str(pdb),
                "pdb_units": "angstrom_centered",
                "pos_units": "nanometers",
            }
        )

    # TRIM TO A COMMON SEED BLOCK RATHER THAN DROPPING SYSTEMS.
    #
    # This store is ragged for a legitimate reason: three systems were DEEPENED past 100 by the
    # pool-extension runs (O74933 to 300, P02787 to 262, P00766 to 208, with seeds running past
    # 1100), while the other thirty sit at exactly 100. The ragged-store refusal below is right --
    # pool_size_min would otherwise govern every cell -- but the response to it had been to build
    # the views over only the 28 systems that happened to share a depth, which silently shrank the
    # denominator of every steered cryptic cell and is what put a 28-system rate next to a
    # 32-system one in the comparison table.
    #
    # Trimming by SEED, not by arrival order, is what makes this safe. The steered arm is defined
    # as the same draws with the adapter on, so seeds 0..N-1 are the block that is paired with the
    # native arm; taking the N lowest seeds keeps that pairing exact and is deterministic
    # regardless of the order passes were written or discovered.
    if args.depth is not None:
        for level, rows in by_level.items():
            per: dict[str, list[dict]] = collections.defaultdict(list)
            for r in rows:
                per[r["sequence_id"]].append(r)
            short = {s: len(v) for s, v in per.items() if len(v) < args.depth}
            if short:
                raise SystemExit(
                    f"REFUSING: level {level} cannot be trimmed to depth {args.depth}; "
                    f"these hold fewer draws: {short}"
                )
            kept: list[dict] = []
            for v in per.values():
                kept.extend(sorted(v, key=lambda r: r["sample_index"])[: args.depth])
            by_level[level] = kept

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for level, rows in sorted(by_level.items()):
        name = LEVEL_NAMES.get(level, f"level_{level}")
        depth = collections.Counter(r["sequence_id"] for r in rows)
        depths = sorted(set(depth.values()))
        out = args.out_dir / name / "final_structures"
        out.mkdir(parents=True, exist_ok=True)
        rows.sort(key=lambda r: (r["sequence_id"], r["sample_index"]))
        (out / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"  {name}: {len(depth)} systems, {len(rows)} draws, depths {depths}")
        if len(depths) != 1:
            raise SystemExit(
                f"REFUSING: {name} is ragged ({depths}); pool_size_min would govern every cell"
            )
        if args.require_systems is not None and len(depth) != args.require_systems:
            raise SystemExit(
                f"REFUSING: {name} covers {len(depth)} systems, expected {args.require_systems}"
            )
        print(f"     wrote {out / 'index.jsonl'}")


if __name__ == "__main__":
    main()
