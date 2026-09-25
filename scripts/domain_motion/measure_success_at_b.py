#!/usr/bin/env python3
"""Score one BioEmu ensemble on ``multiconf_domainmotion`` success@B and write its sidecar.

A THIN LAUNCHER. Every scientific decision -- the atom set, the residue frame, the
self-superposition, the aggregation over systems, the leakage check -- lives in
:mod:`sae_bioemu.domain_motion.success_at_b`, which is in the ``domain_motion_evaluation_only``
fingerprint group. This file chooses paths and prints.

The NATIVE (adapter-off) ensemble scored this way is the baseline any steered claim must beat, and
it has to be regenerated under this project's own sampler settings rather than taken from BioEmu's
published numbers: the benchmark repository warns that v0.0.2 is not comparable to v1.0.0+, and the
public release samples with DPM/30 steps against the paper's Heun/100 while this project uses
DPM-Solver-2 at 49 steps.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sae_bioemu.domain_motion.success_at_b import (
    DEFAULT_TAU_ANGSTROM,
    load_system_references,
    read_sample_index,
    reference_separation,
    require_held_out,
    score_ensemble,
    success_at_b,
)


def canonical_choices(
    nearer: dict[str, int],
) -> list[tuple[str, dict[str, int]]]:
    """Name each canonical cell and the per-system conformer it scores.

    ``nearer`` maps a system to the DEPOSITION index of the conformer the unsteered ensemble sits
    nearer, so ``canonical_nearer`` IS that mapping and ``canonical_further`` is its complement.

    A function, not two lines inside ``main``, because swapping the two names is the one mutation
    that produces a complete-looking table in which every single-direction arm is reported against
    the conformer it was steered AWAY from -- and it is invisible to any check that greps for the
    labels, since both labels are still there. As a function it can be, and is, tested.
    """
    return [
        ("canonical_nearer", dict(nearer)),
        ("canonical_further", {name: 1 - index for name, index in nearer.items()}),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path, help="multiconf_domainmotion copy")
    parser.add_argument("--index", required=True, type=Path, help="final_structures/index.jsonl")
    parser.add_argument("--split-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="JSON sidecar")
    parser.add_argument("--draws-csv", type=Path, help="optional per-draw CSV")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU_ANGSTROM)
    parser.add_argument("--b", type=int, default=32)
    parser.add_argument("--label", default="native")
    parser.add_argument(
        "--canonical-labels",
        type=Path,
        default=None,
        help="domain_motion_canonical_labels.json. With it, each system is scored against the "
        "conformer the CANONICAL order names -- the one the single-direction arms were trained "
        "toward -- instead of a fixed deposition index. The deposited order is arbitrary and "
        "disagrees with the canonical one in 17 of the 22 benchmark systems, so without this a "
        "steered arm is mostly scored against the conformer it was not aimed at",
    )
    arguments = parser.parse_args()

    systems = load_system_references(arguments.assets)
    holdout = require_held_out([system.test_case for system in systems], arguments.split_record)
    separations = {system.test_case: reference_separation(system) for system in systems}

    entries = read_sample_index(arguments.index)
    scores, refusals, outside = score_ensemble(systems, entries)
    for name, score in scores.items():
        score.separation = separations[name]

    # Both deposition cells are always emitted, because they are what bioemu-benchmarks reports
    # and dropping them would make this incomparable. The canonical cells are ADDED when the map is
    # given, and are the ones a single-direction arm should be read on.
    choices: list[tuple[str, int | dict[str, int]]] = [("deposition_0", 0), ("deposition_1", 1)]
    if arguments.canonical_labels is not None:
        labels = json.loads(arguments.canonical_labels.read_text())
        nearer = {str(row["system"]): int(str(row["nearer"]).rsplit("_", 1)[1]) for row in labels}
        missing = sorted({system.test_case for system in systems} - set(nearer))
        if missing:
            raise SystemExit(
                f"REFUSING: {arguments.canonical_labels} names no conformer for {missing}. "
                "Scoring them on the deposited order would silently mix two conventions"
            )
        choices += canonical_choices(nearer)
    cells = []
    for state, conformer_index in choices:
        for tau in (arguments.tau, None):
            cells.append(
                success_at_b(
                    scores.values(),
                    conformer_index=conformer_index,
                    reference_state=state,
                    b=arguments.b,
                    tau=tau,
                ).as_dict()
            )

    payload = {
        "schema": "domain_motion.success_at_b.v1",
        "label": arguments.label,
        "what": (
            "bioemu-benchmarks success@B on multiconf_domainmotion: TRUE RMSD over N/CA/C/O, the "
            "GLOBAL path (mdtraj.rmsd self-superposes), reported per conformer at a fixed tau and "
            "at ConforMix's adaptive 0.5*s_p. An unweighted mean over SYSTEMS; draws are not pooled"
        ),
        "tau_angstrom": arguments.tau,
        "b_requested": arguments.b,
        "index": str(arguments.index),
        "assets": str(arguments.assets),
        "systems_scored": len(scores),
        "draws_outside_the_benchmark": outside,
        "refusals": refusals,
        "holdout": holdout,
        "reference_separations": {
            name: {
                "rmsd_angstrom": item.rmsd_angstrom,
                "adaptive_tau_angstrom": item.adaptive_tau,
                "residues": item.residues,
                "coverage": item.coverage.as_dict(),
            }
            for name, item in sorted(separations.items())
        },
        "cells": cells,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=1, sort_keys=True))

    if arguments.draws_csv is not None:
        arguments.draws_csv.parent.mkdir(parents=True, exist_ok=True)
        with arguments.draws_csv.open("w", newline="\n") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                [
                    "sequence_id",
                    "conformer",
                    "conformer_index",
                    "sample_index",
                    "rmsd_angstrom",
                    "residues",
                    "atoms",
                    "reference_separation_angstrom",
                ]
            )
            for name in sorted(scores):
                for conformer_index in (0, 1):
                    for draw in sorted(
                        scores[name].draws(conformer_index), key=lambda item: item.sample_index
                    ):
                        writer.writerow(
                            [
                                draw.system,
                                draw.conformer,
                                draw.conformer_index,
                                draw.sample_index,
                                f"{draw.rmsd_angstrom:.6f}",
                                draw.residues,
                                draw.atoms,
                                f"{separations[name].rmsd_angstrom:.6f}",
                            ]
                        )

    for cell in cells:
        label = "adaptive" if cell["adaptive_tau"] else f"tau={cell['tau_angstrom']}"
        print(
            f"conformer {cell['conformer_index']}  {label:>12s}  "
            f"{cell['n_successes']}/{cell['n_systems']}  rate={cell['success_rate']:.3f}  "
            f"ci=[{cell['ci_low']:.3f}, {cell['ci_high']:.3f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
