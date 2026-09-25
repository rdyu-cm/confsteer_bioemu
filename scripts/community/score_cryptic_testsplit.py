"""Paired apo->holo RMSD for the cryptic arm on its OWN held-out test split.

WHY THIS EXISTS, AND WHY IT IS NOT success@B
--------------------------------------------
Every cryptic number measured so far is on the 34-system BENCHMARK panel. That panel is disjoint
from the arm's own split (verified: zero overlap with train, validation or test), so it is a clean
held-out claim -- but it is held out on a DIFFERENT distribution, and 34 systems is thin.

The arm's own test split is 134 systems and has never been scored. This scores it.

METRIC: PAIRED MEAN RMSD TO THE DEPOSITED HOLO, NOT success@B. Measured on the benchmark panel's
own draws, the two statistics disagree in SIGN on the same arm: mean RMSD says the steered draws
sit 0.35 A closer to the reference (|t| = 3.3) while best-of-100 says they sit 0.26 A further
away. Both are true. The arm shifts the whole distribution toward the target while thinning the
lucky left tail a best-of-B pick feeds on.

For a held-out claim the mean is the right statistic, for three reasons:
  * it is what the arm optimises, so it measures the thing that was trained
  * it is stable in the draw count, whereas min-over-n is a TAIL statistic whose expectation
    depends on n (+0.031 at n=5 drifting to +0.261 at n=100 on the same draws) -- the number would
    be a property of the budget, not of the arm
  * MEASURED precision: the paired delta is flat from 5 draws up and its standard error barely
    moves past 25 (0.110 -> 0.103 going 25 -> 100), because BETWEEN-system variance dominates.
    That is why the generation is 25 draws and not 100.

PAIRED BY SEED. Each steered draw has a matched unsteered control at the same seed, written by the
generator into control/. Differencing within a seed removes the draw-to-draw variance that
otherwise swamps a 0.1-0.3 A effect, which is why |t| here is large while the effect is modest.

THE HOLO STRUCTURE IS THE POINT, AND IT TOOK THREE TRIES TO FIND. the progress coordinate is defined as a projection
onto the apo->holo transition vector, so every training system HAS both endpoints; but the holo is
not stored as coordinates in stage1_inputs, and cryptobank_dataset.json is a trimmed export that
resolves only 46 of the 134. The pairing comes from cryptobank_dataset.json AND
cryptobench_dataset.json AND the full pickle -- see testsplit134_apo_holo_full.json. Systems whose
holo cannot be resolved are REPORTED, never dropped silently.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import statistics
from pathlib import Path

import mdtraj


def _ca(structure, chain_id: str | None):
    """Cα of ONE chain when the pairing names it, else of everything.

    SELECTING THE CHAIN IS NOT OPTIONAL, and omitting it silently discarded two thirds of the
    panel. A deposited holo entry is routinely a multimer -- 5tle has 14 chains and 1397 Cα,
    1tkk has 40 and 2886 -- while a steered draw is one modelled chain. Comparing a model
    against every chain at once makes the lengths disagree by far more than the 10% guard below
    allows, so `backbone_rmsd` returned None, every draw of that system was dropped, and the
    system fell out under `too_few_scorable_draws`. MEASURED: 89 of 134 systems skipped that way,
    and the 45 that survived were the ones whose holo happened to be near-monomeric -- a subset
    selected by crystal packing, which is not the population the split names.
    """
    topology = structure.topology
    if chain_id:
        wanted = [c.index for c in topology.chains if str(getattr(c, "chain_id", "")) == chain_id]
        if wanted:
            clause = " or ".join(f"chainid {index}" for index in wanted)
            selection = topology.select(f"name CA and protein and ({clause})")
            if selection.size:
                return structure.atom_slice(selection)
    return structure.atom_slice(topology.select("name CA and protein"))


#: The backbone the arms are trained on. `cryptic_rmsd/objective.py` names exactly these four, and
#: every training artifact declares `atom_set=['N','CA','C','O']`. CB is written by BioEmu and is
#: deliberately NOT here: it is the first SIDE-CHAIN atom, not backbone. CA is the alpha carbon; C
#: is the carbonyl carbon, a different atom.
METRIC_ATOMS = ("N", "CA", "C", "O")


def metric_region_rmsd(path: Path, reference_frame, metric_indices) -> float | None:
    """Superposed N/CA/C/O RMSD of one draw against a reference frame, over the metric region.

    THE SAME QUANTITY THE ARM WAS TRAINED ON, unlike the whole-chain C-alpha path below. The
    reference comes from the artifact, where the chain was resolved once at build time, so there is
    no `.cif` to open, no chain letter to match, and no system dropped for a lookup failure --
    which is what cost 13 of 134 systems, 6 of them to chain resolution alone.

    Residues are addressed by `metric_indices` into the draw's own 0..L-1 numbering, which is the
    frame the artifact's indices were recorded against.
    """
    import numpy as np

    try:
        draw = mdtraj.load(str(path))
    except Exception:
        return None
    topology = draw.topology
    wanted = []
    for residue_index in metric_indices:
        by_name = {a.name: a.index for a in topology.residue(int(residue_index)).atoms}
        if not all(name in by_name for name in METRIC_ATOMS):
            return None
        wanted.extend(by_name[name] for name in METRIC_ATOMS)
    x = draw.xyz[0][wanted] * 10.0  # nm -> Angstrom
    y = np.asarray(reference_frame, dtype=float).reshape(-1, 3)
    if x.shape != y.shape:
        return None
    # Kabsch: centre both, rotate x onto y, then RMS over all atoms.
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    u, _, vt = np.linalg.svd(xc.T @ yc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt(((xc @ rotation.T - yc) ** 2).sum(axis=1).mean()))


def load_reference_ca(path: Path, chain_id: str | None):
    """Load a reference structure ONCE and keep only the Cα of the named chain.

    The reference is the same file for every draw of a system, and it is the EXPENSIVE one: a
    deposited entry is a multimer of thousands of atoms while a draw is a single modelled chain.
    Reloading it per call cost two full parses per draw -- MEASURED at ~100x more reference loads
    than systems, and it is why a 10-draw benchmark run took ~9 minutes rather than seconds.
    Hoisting it out of the loop is the whole optimisation; the arithmetic below is unchanged.
    """
    try:
        return _ca(mdtraj.load(str(path)), chain_id)
    except Exception:
        return None


def backbone_rmsd(path_a: Path, reference_ca, chain_a: str | None = None) -> float | None:
    """Cα RMSD after superposition, over the residues both structures resolve.

    ``reference_ca`` is a pre-sliced trajectory from :func:`load_reference_ca`, not a path.
    """
    if reference_ca is None:
        return None
    try:
        a = mdtraj.load(str(path_a))
    except Exception:
        return None
    ca_a = _ca(a, chain_a)
    ca_b = reference_ca
    n = min(ca_a.n_atoms, ca_b.n_atoms)
    if n < 20:
        return None
    # REFUSE rather than guess when the two differ in length by more than a few residues: a
    # truncated superposition over a mismatched correspondence reports a small, confident, wrong
    # number.
    if abs(ca_a.n_atoms - ca_b.n_atoms) > 0.1 * n:
        return None
    # atom_slice returns a new trajectory, so the CACHED reference is never mutated -- but
    # superpose() writes in place, so ca_a must be the copy that moves and ca_b a fresh slice.
    ca_a = ca_a.atom_slice(list(range(n)))
    ca_b = ca_b.atom_slice(list(range(n)))
    ca_a.superpose(ca_b)
    d = ca_a.xyz[0] - ca_b.xyz[0]
    return float((d**2).sum(axis=1).mean() ** 0.5 * 10.0)  # nm -> Angstrom


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True, type=Path, help="the steered store root")
    p.add_argument("--pairing", required=True, type=Path, help="testsplit134_apo_holo_full.json")
    p.add_argument("--structures", required=True, type=Path, help="cryptobank/structures")
    p.add_argument(
        "--min-draws",
        type=int,
        default=5,
        help="fewest scorable draws a system needs to be scored; 5 at the reported depth, where it leaves 121 of the cryptic test systems",
    )
    p.add_argument(
        "--max-draws",
        type=int,
        default=0,
        help="cap each (system, level) at this many draws, taken as the N LOWEST draw_seeds; "
        "0 keeps everything. Without a cap the depth is whatever generation happened to leave "
        "behind -- in bench_ms8_best36 that is 100 to 300 draws and level +1 has 8900 against "
        "level -1's 3770 -- so the two levels would be measured at different depths and the "
        "per-system means would carry different amounts of noise",
    )
    p.add_argument(
        "--reference-artifact",
        type=Path,
        default=None,
        help="pocket_rmsd_reference.pt. When given, the reference frames come from the ARTIFACT "
        "and the metric becomes N/CA/C/O over the metric region -- the same atoms and region the "
        "arms were TRAINED on and the benchmark scores them on. That also removes .cif loading "
        "and chain selection entirely, so every system in the split scores instead of 121 of 134. "
        "Without it the metric is whole-chain C-alpha against the deposited chain, which is what "
        "the published -0.424 A used and is NOT comparable",
    )
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--out-csv", required=True, type=Path)
    args = p.parse_args()

    reference_entries = None
    if args.reference_artifact is not None:
        import torch

        reference_entries = torch.load(
            args.reference_artifact, map_location="cpu", weights_only=False
        )["entries"]
        print(
            f"reference artifact: {len(reference_entries)} systems, N/CA/C/O over the metric "
            "region -- the training metric; no .cif is read and no chain is matched"
        )

    pairing = json.loads(args.pairing.read_text())
    pairs = pairing["pairs"] if "pairs" in pairing else pairing

    sidecar = args.store / "steered_draws.jsonl"
    if not sidecar.exists():
        raise SystemExit(f"REFUSING: {sidecar} absent")

    draws = collections.defaultdict(list)
    for line in sidecar.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sid = str(r["sequence_id"])
        if sid.startswith("__") or not r.get("control_present"):
            continue
        draws[(sid, float(r["level"]))].append(r)

    # Deterministic cap BEFORE anything is scored. Sorting on draw_seed rather than file order
    # makes the selection reproducible and makes the levels share seeds where both reach that
    # depth, so the two levels are read on the same noise draws rather than on whichever subset
    # each happened to generate first.
    capped_from = {}
    if args.max_draws:
        for key, rs in draws.items():
            if len(rs) > args.max_draws:
                capped_from[f"{key[0]}|{key[1]}"] = len(rs)
            draws[key] = sorted(rs, key=lambda r: (int(r["draw_seed"]), int(r["pass_index"])))[
                : args.max_draws
            ]

    rows, skipped = [], collections.Counter()
    for (sid, level), rs in sorted(draws.items()):
        spec = pairs.get(sid)
        if not spec:
            skipped["no_holo_pairing"] += 1
            continue
        if reference_entries is not None:
            # ARTIFACT MODE. The frames were resolved once when the artifact was built, so there is
            # no chain to match and nothing to drop. The level names WHICH deposited conformer this
            # arm was trained toward, so it is the one scored against -- measuring an apo-trained
            # arm against holo would score a target it never had.
            entry = reference_entries.get(sid)
            if entry is None:
                skipped["absent_from_reference_artifact"] += 1
                continue
            state = "apo" if float(level) < 0 else "holo"
            reference_frame = entry[state]
            metric_indices = list(entry["metric_indices"])
        else:
            holo = args.structures / f"{spec['holo_pdb_id']}.cif"
            if not holo.exists():
                skipped["holo_cif_absent"] += 1
                continue
        # The steered/control draws are single-chain models, so only the HOLO side needs a chain;
        # the pairing recorded it. Loaded ONCE per system rather than twice per draw.
        if reference_entries is None:
            holo_ca = load_reference_ca(holo, spec.get("holo_chain"))
            if holo_ca is None:
                skipped["holo_cif_unreadable"] += 1
                continue
        deltas = []
        for r in rs:
            if reference_entries is not None:
                s = metric_region_rmsd(
                    args.store / str(r["steered_pdb"]), reference_frame, metric_indices
                )
                c = metric_region_rmsd(
                    args.store / str(r["control_pdb"]), reference_frame, metric_indices
                )
            else:
                s = backbone_rmsd(args.store / str(r["steered_pdb"]), holo_ca)
                c = backbone_rmsd(args.store / str(r["control_pdb"]), holo_ca)
            if s is None or c is None:
                continue
            deltas.append(s - c)
        if len(deltas) < args.min_draws:
            skipped["too_few_scorable_draws"] += 1
            continue
        rows.append(
            {
                "sequence_id": sid,
                "level": level,
                "holo": f"{spec['holo_pdb_id']}_{spec.get('holo_chain')}",
                # what the delta was actually measured against: artifact mode scores each level
                # against ITS OWN trained state, so a level -1 row is an APO distance, not holo
                "reference": (
                    f"artifact_{'apo' if float(level) < 0 else 'holo'}"
                    if reference_entries is not None
                    else "holo_cif"
                ),
                "n_draws": len(deltas),
                "mean_delta_rmsd": round(statistics.mean(deltas), 4),
            }
        )

    if not rows:
        raise SystemExit(
            f"REFUSING: no system had the {args.min_draws} scorable draws --min-draws asks for; "
            f"skipped {dict(skipped)}"
        )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {"skipped": dict(skipped), "min_draws": args.min_draws, "levels": {}}
    print(
        f"{'level':>6s} {'n sys':>6s} {'mean delta':>11s} {'SE':>7s} {'|t|':>6s} {'improved':>10s}"
    )
    for level in sorted({r["level"] for r in rows}):
        d = [r["mean_delta_rmsd"] for r in rows if r["level"] == level]
        m = statistics.mean(d)
        se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        better = sum(1 for x in d if x < 0)
        print(
            f"{level:+6.0f} {len(d):6d} {m:+11.4f} {se:7.4f} {abs(m/se):6.2f} {better:>6d}/{len(d):<4d}"
        )
        summary["levels"][str(level)] = {
            "n_systems": len(d),
            "mean_delta_rmsd": round(m, 4),
            "standard_error": round(se, 4),
            "t": round(abs(m / se), 3),
            "systems_improved": better,
        }
    summary["draw_cap"] = {
        "max_draws": args.max_draws or None,
        "selection": "the N lowest draw_seeds" if args.max_draws else "every draw in the sidecar",
        "units_capped": len(capped_from),
        "depth_before_cap": dict(sorted(capped_from.items())),
    }
    if args.max_draws:
        print(
            f"\ncap: {args.max_draws} draws per (system, level) by lowest draw_seed; "
            f"{len(capped_from)} of {len(draws)} units were deeper and were trimmed"
        )
    print(f"\nskipped: {dict(skipped)}  (reported, not dropped silently)")
    if reference_entries is not None:
        meaning = (
            "negative delta = steered draws sit CLOSER to the level's own trained reference "
            "(level -1: apo, level +1: holo; N/CA/C/O over the metric region) than their "
            "seed-matched control"
        )
    else:
        meaning = (
            "negative delta = steered draws sit CLOSER to the deposited holo (whole-chain CA) "
            "than their seed-matched control"
        )
    summary["delta_meaning"] = meaning
    print(meaning)
    args.out_json.write_text(json.dumps(summary, indent=1))
    print(f"wrote {args.out_json} and {args.out_csv}")


if __name__ == "__main__":
    main()
