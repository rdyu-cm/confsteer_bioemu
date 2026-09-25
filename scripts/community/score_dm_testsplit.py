#!/usr/bin/env python3
"""Panel e for DOMAIN MOTION: score the held-out test split on the training metric.

The domain-motion twin of ``score_cryptic_testsplit.py --reference-artifact``. For every
(system, draw) of ONE unconditioned arm, compare the steered draw with its seed-matched control on
the metric the arm was trained on -- N/CA/C/O over the artifact's ``metric_residues``, superposed on
those same atoms -- against the conformer the arm's level names (``domain_motion/objective.py``
REFERENCE_STATES: level -1 -> ``first``, +1 -> ``second``; in the canonical artifact ``first`` is the
conformer NEARER the unsteered BioEmu ensemble).

``measure_success_at_b.py`` cannot do this: it reads the multiconf benchmark copy and matches none of
the DynDom test systems, which is how an earlier run reported 0/0 for every cell and exited 0.

Only the arm's OWN level is scored: the generator writes both levels, and the other one is a dose a
single-direction adapter never trained on.

SELF-CHECK before scoring: the artifact records each system's ``separation_angstrom`` (first vs
second over the metric residues). It is recomputed here with the same RMSD routine; disagreement
means the routine or the residue indexing is wrong, and the run refuses.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_cryptic_testsplit import metric_region_rmsd  # noqa: E402


def kabsch_rmsd(x, y) -> float:
    import numpy as np

    x = np.asarray(x, dtype=float).reshape(-1, 3)
    y = np.asarray(y, dtype=float).reshape(-1, 3)
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    u, _, vt = np.linalg.svd(xc.T @ yc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt(((xc @ rotation.T - yc) ** 2).sum(axis=1).mean()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True, help="<testsplit>/store")
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--level", type=float, required=True, help="the arm's trained level")
    parser.add_argument("--tau", type=float, default=3.0, help="domain-motion success cutoff")
    parser.add_argument(
        "--min-draws",
        type=int,
        default=5,
        help="fewest scorable draws a system needs to be scored; 5 at the reported depth, where it leaves all 69 systems",
    )
    parser.add_argument("--max-draws", type=int, default=0, help="cap by lowest draw_seed; 0 = all")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    import torch

    entries = torch.load(args.reference_artifact, map_location="cpu", weights_only=False)["entries"]
    state = "first" if args.level < 0 else "second"

    worst = 0.0
    for name in sorted(entries)[:5]:
        e = entries[name]
        idx = list(e["metric_residues"])
        got = kabsch_rmsd(e["first"][idx].numpy(), e["second"][idx].numpy())
        worst = max(worst, abs(got - float(e["separation_angstrom"])))
    if worst > 1e-3:
        raise SystemExit(f"REFUSING: separation self-check off by {worst:.4f} A")
    print(f"self-check: recorded separation reproduced to {worst:.1e} A on 5 systems")

    draws = collections.defaultdict(list)
    for line in (args.store / "steered_draws.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if float(r["level"]) != args.level or not r.get("control_present"):
            continue
        draws[str(r["sequence_id"])].append(r)

    rows, skipped = [], collections.Counter()
    for sid in sorted(draws):
        e = entries.get(sid)
        if e is None:
            skipped["absent_from_reference_artifact"] += 1
            continue
        idx = list(e["metric_residues"])
        frame = e[state][idx].numpy()
        rs = sorted(draws[sid], key=lambda r: int(r["draw_seed"]))
        if args.max_draws:
            rs = rs[: args.max_draws]
        steered, control = [], []
        for r in rs:
            s = metric_region_rmsd(args.store / r["steered_pdb"], frame, idx)
            c = metric_region_rmsd(args.store / r["control_pdb"], frame, idx)
            if s is None or c is None:
                continue
            steered.append(s)
            control.append(c)
        if len(steered) < args.min_draws:
            skipped["too_few_scorable_draws"] += 1
            continue
        rows.append(
            {
                "sequence_id": sid,
                "level": args.level,
                "reference": state,
                "n_draws": len(steered),
                "mean_delta_rmsd": round(
                    statistics.mean(s - c for s, c in zip(steered, control, strict=True)), 4
                ),
                "steered_min_rmsd": round(min(steered), 4),
                "control_min_rmsd": round(min(control), 4),
                "steered_success": int(min(steered) < args.tau),
                "control_success": int(min(control) < args.tau),
                "separation_angstrom": round(float(e["separation_angstrom"]), 4),
            }
        )

    if not rows:
        raise SystemExit(
            f"REFUSING: no system had the {args.min_draws} scorable draws --min-draws asks for; "
            f"skipped {dict(skipped)}"
        )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    d = [r["mean_delta_rmsd"] for r in rows]
    n = len(d)
    m = statistics.mean(d)
    se = statistics.stdev(d) / math.sqrt(n) if n > 1 else float("nan")
    n_draws = sorted({r["n_draws"] for r in rows})
    summary = {
        "level": args.level,
        "reference": state,
        "n_systems": n,
        "draws_per_system": n_draws,
        "mean_delta_rmsd": round(m, 4),
        "standard_error": round(se, 4),
        "t": round(abs(m / se), 3),
        "systems_improved": sum(1 for x in d if x < 0),
        "tau_angstrom": args.tau,
        "steered_success_rate": round(sum(r["steered_success"] for r in rows) / n, 4),
        "control_success_rate": round(sum(r["control_success"] for r in rows) / n, 4),
        "skipped": dict(skipped),
        "min_draws": args.min_draws,
        "delta_meaning": (
            f"negative delta = steered draws sit CLOSER to the level's own conformer ({state}; "
            "N/CA/C/O over metric_residues, superposed on the same atoms) than their seed-matched "
            "control"
        ),
        "success_meaning": f"success = at least one of the system's draws within {args.tau} A (success@{n_draws})",
    }
    args.out_json.write_text(json.dumps(summary, indent=1))
    print(
        f"level {args.level:+.0f} ({state}): n={n} draws={n_draws} mean delta {m:+.4f} A "
        f"SE {se:.4f} |t| {abs(m / se):.2f} improved {summary['systems_improved']}/{n}; "
        f"success@{n_draws} steered {summary['steered_success_rate']:.3f} "
        f"control {summary['control_success_rate']:.3f}; skipped {dict(skipped)}"
    )
    print(summary["delta_meaning"])


if __name__ == "__main__":
    main()
