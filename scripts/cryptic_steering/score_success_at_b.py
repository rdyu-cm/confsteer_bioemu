"""Score success@B on the 34 BioEmu cryptic-pocket systems (D-63).

CPU-only and post-hoc: this reads structures that already exist, generates nothing, and touches no
loss. For each system it superposes each sample onto the apo and the holo reference over
``alignment_resid_ranges``, measures the deviation over ``metric_resid_ranges`` on backbone
N/CA/C/O, and reports the fraction of systems where at least one of B samples lands within tau --
apo and holo separately, at tau = 1.0 A (ConforNets) and tau = 1.5 A (BioEmu), in BOTH deviation
forms, with a system-level bootstrap interval.

Both deviation forms are always reported because BioEmu's local metric is the arithmetic MEAN of
per-atom deviations while it is named ``rmsd`` in that code; see
:mod:`sae_bioemu.cryptic_steering.success_at_b` for the source lines and the rest of the protocol.

Every run also reports the experimental apo<->holo separation of each system under the same
regions. That is the scale tau is being compared against, and halving it gives ConforMix's adaptive
secondary threshold. It needs no generated samples, so it is produced even when an ensemble covers
none of the benchmark systems -- which is the expected state, since the benchmark is held out of
this project's training pool by construction.

TWO CRITERIA, NEVER BLENDED. Every (state, form, tau) cell is reported TWICE and each row names
which it is in its own ``criterion`` field:

* ``plain`` -- at least one of B samples within tau. The field's instrument, unchanged, and the
  only rows comparable with ``bioemu-benchmarks``, ConforNets or ConforMix.
* ``specific`` -- at least one sample within tau of the named state AND NOT within tau of the
  other. It exists because apo and holo are frequently closer together than tau, so "within tau of
  holo" can be satisfied by sitting at apo; see
  :mod:`sae_bioemu.cryptic_steering.conformer_separation`. It is NOT comparable with any published
  figure and must never be quoted as success@B.

Every row also carries each system's separation STRATUM -- ``clean`` (s > 2*tau), ``ambiguous``
(tau < s <= 2*tau), ``degenerate`` (s <= tau) or ``unknown`` -- as a tag that enters no success
test. Reported, never used to filter.

Outputs: a JSON artifact with every per-sample deviation, a CSV sufficient to redraw the figure
without re-scoring, and a 300 DPI figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sae_bioemu.cryptic_steering.conformer_separation import (  # noqa: E402
    separation_stratum,
)
from sae_bioemu.cryptic_steering.success_at_b import (  # noqa: E402
    APO,
    BACKBONE_ATOMS,
    DEFAULT_COVERAGE_FLOOR,
    DEVIATION_FORMS,
    HOLO,
    MEAN_DEVIATION,
    REFERENCE_STATES,
    RMSD,
    BackboneChain,
    ResidueCorrespondenceError,
    SuccessResult,
    SystemScore,
    benchmark_frame,
    coverage_at_adaptive_tau,
    load_benchmark_ranges,
    map_samples_to_systems,
    read_generated_backbone,
    read_reference_chain,
    read_sample_index,
    region_coverage,
    score_reference_separation,
    score_sample,
    specific_success_at_b,
    success_at_b,
    take_b,
)

REPOSITORY = Path(__file__).resolve().parents[2]
#: Every default is REPOSITORY-RELATIVE, so a fresh clone scores without being told where
#: anything lives. ``structures`` is the one directory a clone does not start with: the mmCIF
#: entries are public and are fetched by ``scripts/release/fetch_structures.py``.
DEFAULT_RANGES = REPOSITORY / "data/benchmarks/cryptic/metric_ranges.json"
DEFAULT_INVENTORY = REPOSITORY / "data/benchmarks/cryptic/pairing.csv"
DEFAULT_TESTCASES = REPOSITORY / "data/benchmarks/cryptic/sequences.csv"
DEFAULT_STRUCTURES = REPOSITORY / "data/benchmarks/structures"

#: ConforNets' cryptic-pocket cutoff and BioEmu's own. Neither is a default that can be omitted
#: from a report: at B = 100 BioEmu scores 30.5/60.6 at 1.0 A and 49/85 at 1.5 A on the same
#: systems, so a rate without its tau and its B says nothing.
DEFAULT_TAUS = (1.0, 1.5)

#: The population every number on this artifact describes.
POPULATION = "bioemu-benchmarks multiconf_crypticpocket, 34 test cases"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_inventory(path: Path) -> dict[str, dict[str, tuple[str, str]]]:
    """system -> {"apo": (PDB, chain), "holo": (PDB, chain)} from the D-19 inventory."""
    references: dict[str, dict[str, tuple[str, str]]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            references[row["accession"]] = {
                APO: (row["apo_pdb_id"].upper(), row["apo_auth_chain"]),
                HOLO: (row["holo_pdb_id"].upper(), row["holo_auth_chain"]),
            }
    return references


def read_testcase_sequences(path: Path) -> dict[str, str]:
    """system -> the benchmark's test-case sequence, which defines its numbering frame."""
    with path.open() as handle:
        return {row["test_case"]: row["sequence"] for row in csv.DictReader(handle)}


def load_reference(
    structures: Path, pdb_id: str, chain: str, atom_names: tuple[str, ...]
) -> BackboneChain:
    path = structures / f"{pdb_id.lower()}.cif"
    if not path.exists():
        raise FileNotFoundError(f"cached mmCIF not found: {path}")
    return read_reference_chain(path, chain, atom_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranges", type=Path, default=DEFAULT_RANGES)
    parser.add_argument(
        "--pairing", "--inventory", dest="inventory", type=Path, default=DEFAULT_INVENTORY
    )
    parser.add_argument("--testcases-csv", type=Path, default=DEFAULT_TESTCASES)
    parser.add_argument("--structures", type=Path, default=DEFAULT_STRUCTURES)
    parser.add_argument(
        "--samples",
        type=Path,
        action="append",
        default=None,
        metavar="INDEX_JSONL",
        help="a BioEmu final_structures/index.jsonl; repeat for more than one arm",
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=None,
        help="label for each --samples index, in the same order (default: native)",
    )
    parser.add_argument(
        "--b",
        type=int,
        default=None,
        help="samples per system; default uses every sample present and reports the count",
    )
    parser.add_argument("--tau", type=float, action="append", default=None)
    parser.add_argument("--coverage-floor", type=float, default=DEFAULT_COVERAGE_FLOOR)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    parser.add_argument(
        "--sampler-denoiser",
        default="dpm-solver-2",
        help="declared by the caller and recorded, not read from the samples",
    )
    parser.add_argument("--sampler-steps", type=int, default=49)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    arguments = parser.parse_args()

    taus = tuple(arguments.tau) if arguments.tau else DEFAULT_TAUS
    atom_names = BACKBONE_ATOMS
    sample_indices = list(arguments.samples or [])
    arms = list(arguments.arm or [])
    if not arms:
        arms = [f"arm{index}" if index else "native" for index in range(len(sample_indices))]
    if len(arms) != len(sample_indices):
        parser.error("--arm must be given once per --samples, in the same order")

    systems, excluded = load_benchmark_ranges(arguments.ranges)
    references = read_inventory(arguments.inventory)
    testcases = read_testcase_sequences(arguments.testcases_csv)
    print(f"benchmark systems with ranges: {len(systems)}; excluded upstream: {sorted(excluded)}")
    for name, reason in sorted(excluded.items()):
        print(f"  EXCLUDED {name}: {reason}")

    # ---------------------------------------------------------------- references and their frames
    chains: dict[str, dict[str, BackboneChain]] = {}
    frames: dict[str, dict[str, object]] = {}
    separations = {}
    coverage_by_reference: dict[str, dict[str, dict[str, object]]] = {}
    #: Systems that cannot be scored at all. Kept apart from `separation_unavailable`, which only
    #: costs the ConforMix adaptive threshold and leaves success@B at a fixed tau intact.
    refused: dict[str, str] = dict(excluded)
    separation_unavailable: dict[str, str] = {}
    for system, ranges in systems.items():
        sequence = testcases.get(system)
        if sequence is None:
            refused[system] = "no test-case sequence: the benchmark numbering frame is undefined"
            continue
        if len(sequence) != ranges.n_residues:
            refused[system] = (
                f"test-case sequence is {len(sequence)} long but the ranges file records "
                f"n_residues={ranges.n_residues}"
            )
            continue
        chains[system] = {}
        frames[system] = {}
        for state in REFERENCE_STATES:
            pdb_id, chain_id = references[system][state]
            chain = load_reference(arguments.structures, pdb_id, chain_id, atom_names)
            if not len(chain):
                refused[system] = f"{state} chain {pdb_id}_{chain_id} has no standard residues"
                break
            chains[system][state] = chain
            frame = benchmark_frame(chain, sequence)
            frames[system][state] = frame
            coverage_by_reference.setdefault(system, {})[state] = {
                "alignment": region_coverage(
                    ranges.alignment_positions, chain, frame, atom_names
                ).as_dict(),
                "metric": region_coverage(
                    ranges.metric_positions, chain, frame, atom_names
                ).as_dict(),
            }
        else:
            try:
                separations[system] = score_reference_separation(
                    system,
                    ranges,
                    chains[system][APO],
                    frames[system][APO],
                    chains[system][HOLO],
                    frames[system][HOLO],
                    atom_names,
                    arguments.coverage_floor,
                )
            except (ResidueCorrespondenceError, ValueError) as error:
                # Only the adaptive secondary threshold is lost: both crystals must observe a
                # residue for the apo<->holo separation, while a generated sample observes every
                # residue of its construct, so success@B at a fixed tau is unaffected.
                separation_unavailable[system] = str(error)

    print(
        f"reference apo/holo separations measured: {len(separations)}/{len(systems)}; "
        f"unavailable for {sorted(separation_unavailable)}"
    )
    for name, reason in sorted(separation_unavailable.items()):
        print(f"  NO ADAPTIVE TAU {name}: {reason}")

    # Which systems an ensemble COULD be scored on, decided by the reference crystals alone. A
    # generated sample observes every residue of its construct, so this is the whole of the
    # constraint and it can be reported before a single draw exists.
    scorable = {
        state: sorted(
            system
            for system, states in coverage_by_reference.items()
            if state in states
            and min(states[state]["alignment"]["fraction"], states[state]["metric"]["fraction"])
            >= arguments.coverage_floor
        )
        for state in REFERENCE_STATES
    }
    for state in REFERENCE_STATES:
        print(
            f"scorable against the {state} reference once samples exist: "
            f"{len(scorable[state])}/{len(systems)}"
        )

    # ---------------------------------------------------------------------------- samples per arm
    scores: dict[str, dict[str, SystemScore]] = {}
    arm_coverage: dict[str, dict[str, object]] = {}
    for arm, index_path in zip(arms, sample_indices, strict=True):
        entries = read_sample_index(index_path)
        matched, sequence_refusals, outside = map_samples_to_systems(entries, references, testcases)
        arm_coverage[arm] = {
            "index_path": str(index_path),
            "index_sha256": sha256_file(Path(index_path)),
            "samples_indexed": len(entries),
            "sequences_indexed": len({entry.sequence_id for entry in entries}),
            "samples_on_benchmark_systems": sum(len(value) for value in matched.values()),
            "samples_outside_benchmark": outside,
            "systems_covered": sorted(matched),
            "systems_not_covered": sorted(set(systems) - set(matched)),
            "sequence_refusals": sequence_refusals,
        }
        print(
            f"[{arm}] {len(entries)} indexed samples; "
            # `matched` can exceed `systems`: a native benchmark ensemble is generated from all 34
            # test cases while `systems` holds only the 33 that carry ranges, so Q16539 is matched
            # and then unscorable. Reporting "34/33" read as an error in a line whose whole job is
            # to say whether the samples reached the benchmark.
            f"{len(set(matched) & set(systems))}/{len(systems)} benchmark systems covered "
            f"({len(set(matched) - set(systems))} matched but unscorable); "
            f"{outside} samples belong to no benchmark system"
        )
        for sequence_id, reason in sorted(sequence_refusals.items()):
            print(f"  REFUSED {sequence_id}: {reason}")

        scores[arm] = {}
        for system, sample_entries in sorted(matched.items()):
            if system not in chains:
                refused.setdefault(system, "no usable reference frame")
                continue
            score = SystemScore(system=system)
            sequence = testcases[system]
            try:
                for entry in sorted(sample_entries, key=lambda item: item.sample_index):
                    sample = read_generated_backbone(entry.pdb_path, entry.sequence, atom_names)
                    sample_frame = benchmark_frame(sample, sequence)
                    for state in REFERENCE_STATES:
                        deviation, coverage = score_sample(
                            system,
                            state,
                            entry.sequence_id,
                            entry.sample_index,
                            systems[system],
                            chains[system][state],
                            frames[system][state],
                            sample,
                            sample_frame,
                            atom_names,
                            arguments.coverage_floor,
                        )
                        score.deviations.setdefault(state, []).append(deviation)
                        score.coverage[state] = coverage
            except (ResidueCorrespondenceError, ValueError) as error:
                refused[system] = f"[{arm}] {error}"
                continue
            score.reference_separation = separations.get(system)
            scores[arm][system] = score

    # ------------------------------------------------------------------------------ success table
    table: list[dict[str, object]] = []
    for arm in arms:
        arm_scores = list(scores.get(arm, {}).values())
        for state in REFERENCE_STATES:
            for form in DEVIATION_FORMS:
                for tau in taus:
                    # BOTH criteria, in their own rows, each naming itself. The plain one is the
                    # field-comparable number and is unchanged; the specific one is reported
                    # beside it and is NOT comparable with any published figure. They are never
                    # blended and a consumer must read `criterion` to know which it has.
                    for scorer in (success_at_b, specific_success_at_b):
                        result = scorer(
                            arm_scores,
                            reference_state=state,
                            form=form,
                            tau=tau,
                            b=arguments.b,
                            resamples=arguments.bootstrap_resamples,
                            seed=arguments.bootstrap_seed,
                        )
                        if result.n_systems == 0:
                            continue
                        row = result.as_dict()
                        row["arm"] = arm
                        row["threshold"] = "fixed"
                        table.append(row)
                adaptive: SuccessResult = coverage_at_adaptive_tau(
                    arm_scores,
                    reference_state=state,
                    form=form,
                    b=arguments.b,
                    resamples=arguments.bootstrap_resamples,
                    seed=arguments.bootstrap_seed,
                    # The population's tau of record, used ONLY to tag each system's separation
                    # band. It thresholds nothing here; the adaptive tau_i does that.
                    stratum_tau=max(taus),
                )
                if adaptive.n_systems == 0:
                    continue
                row = adaptive.as_dict()
                row["arm"] = arm
                row["threshold"] = "adaptive_half_reference_separation"
                table.append(row)

    # ------------------------------------------------------------------------------------ outputs
    artifact = {
        "decision": "D-63: success@B is the reported cryptic-pocket metric; the progress coordinate is mechanism",
        "population": POPULATION,
        "protocol": {
            "superposition_region": "alignment_resid_ranges",
            "metric_region": "metric_resid_ranges",
            "atom_set": list(atom_names),
            "atom_set_note": (
                "BioEmu final structures carry N, CA, C, CB, O; the C-beta is excluded, matching "
                "mdtraj's `backbone` selection used upstream. No C-alpha-only fallback exists."
            ),
            "deviation_forms": {
                MEAN_DEVIATION: (
                    "arithmetic mean of per-atom deviations -- what bioemu-benchmarks' "
                    "calc_metrics_local returns under the name `rmsd`"
                ),
                RMSD: "true root-mean-square of the same per-atom deviations",
            },
            "taus_angstrom": list(taus),
            "b_requested": arguments.b,
            "coverage_floor": arguments.coverage_floor,
            "numbering_frame": (
                "1-based positions in the benchmark test-case sequence; deposited chains and "
                "generated samples are mapped into it by letter-matched global alignment"
            ),
            "bootstrap": {
                "unit": "system (one protein)",
                "resamples": arguments.bootstrap_resamples,
                "seed": arguments.bootstrap_seed,
                "confidence": 0.95,
            },
        },
        "sampler_declared": {
            "denoiser": arguments.sampler_denoiser,
            "steps": arguments.sampler_steps,
            "note": (
                "declared by the caller and recorded so the number is quotable; it is not read "
                "back from the samples. Published BioEmu baselines used other settings and are "
                "not a control for this arm."
            ),
        },
        "inputs": {
            "ranges": {"path": str(arguments.ranges), "sha256": sha256_file(arguments.ranges)},
            "inventory": {
                "path": str(arguments.inventory),
                "sha256": sha256_file(arguments.inventory),
            },
            "testcases_csv": {
                "path": str(arguments.testcases_csv),
                "sha256": sha256_file(arguments.testcases_csv),
            },
            "structures": str(arguments.structures),
        },
        "systems_with_ranges": sorted(systems),
        "excluded_systems": refused,
        "separation_unavailable": separation_unavailable,
        "reference_range_coverage": coverage_by_reference,
        "scorable_once_samples_exist": scorable,
        "arms": arm_coverage,
        "reference_separation": {
            system: {
                "mean_deviation": value.mean_deviation,
                "rmsd": value.rmsd,
                "metric_residues": value.metric_residues,
                "alignment_residues": value.alignment_residues,
                "metric_coverage": value.metric_coverage.as_dict(),
                "alignment_coverage": value.alignment_coverage.as_dict(),
                "adaptive_tau_mean_deviation": value.adaptive_tau(MEAN_DEVIATION),
                "adaptive_tau_rmsd": value.adaptive_tau(RMSD),
            }
            for system, value in sorted(separations.items())
        },
        "reference_frames": {
            system: {
                state: {
                    "pdb": f"{references[system][state][0]}_{references[system][state][1]}",
                    "chain_residues": chains[system][state].__len__(),
                    "matched_to_frame": frame.matched,
                    "auth_offset_modal": frame.auth_offset_modal,
                    "auth_offset_distinct": frame.auth_offset_distinct,
                }
                for state, frame in states.items()
            }
            for system, states in sorted(frames.items())
        },
        "success_table": table,
        "sample_deviations": {
            arm: [
                asdict(deviation)
                for score in sorted(arm_scores.values(), key=lambda item: item.system)
                for state in REFERENCE_STATES
                for deviation in score.samples(state)
            ]
            for arm, arm_scores in scores.items()
        },
    }
    arguments.out_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.out_json.write_text(json.dumps(artifact, indent=1, sort_keys=True, default=float))

    write_csv(arguments.out_csv, systems, separations, scores, taus, arguments.b)
    write_figure(
        arguments.out_png,
        separations,
        scores,
        table,
        taus,
        arguments.b,
        refused,
        arms,
        systems,
        separation_unavailable,
        scorable,
    )

    print()
    print(
        f"{'arm':10s} {'criterion':9s} {'state':5s} {'form':15s} {'tau':>5s} {'B':>5s} {'n':>3s} "
        f"{'rate':>6s} {'95% CI':>15s}  strata"
    )
    for row in table:
        tau = row["tau_angstrom"]
        label = "adaptive" if row["threshold"] != "fixed" else f"{tau:.1f}"
        counts = row.get("stratum_counts")
        strata = (
            "not stratified"
            if not counts
            else " ".join(f"{name[:4]}={value}" for name, value in counts.items() if value)
        )
        print(
            f"{row['arm']:10s} {str(row['criterion']):9s} {row['reference_state']:5s} "
            f"{row['deviation_form']:15s} "
            f"{label:>5s} {str(row['b_requested']):>5s} {row['n_systems']:3d} "
            f"{row['success_rate']:6.3f} "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]  {strata}"
        )
    print(
        "\n`criterion` is load-bearing: 'plain' is the field-comparable success@B and 'specific' "
        "additionally requires the SAME sample to sit outside tau of the other reference state. "
        "Only the plain rows are comparable with bioemu-benchmarks, ConforNets or ConforMix."
    )
    if not table:
        print("no arm carried samples on any benchmark system: nothing to score")


def write_csv(path: Path, systems, separations, scores, taus, b) -> None:
    """One row per (arm, system, reference state, deviation form): enough to redraw the figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        [
            "arm",
            "system",
            "reference_state",
            "deviation_form",
            "n_samples",
            "best_deviation",
            "median_deviation",
            "reference_separation",
            "adaptive_tau",
            "metric_residues",
            "metric_coverage_fraction",
        ]
        + [f"success_tau_{tau:g}" for tau in taus]
        # BOTH criteria per system, in their own named columns. `success_tau_*` is the
        # field-comparable plain rule and is unchanged; `specific_success_tau_*` additionally
        # requires the SAME sample to sit outside tau of the other reference state.
        + [f"specific_success_tau_{tau:g}" for tau in taus]
        # ...and the separation band the system sits in at each tau, reported and never filtered.
        + [f"separation_stratum_tau_{tau:g}" for tau in taus]
        + ["success_adaptive"]
    )
    rows: list[list[object]] = []
    for system in sorted(systems):
        separation = separations.get(system)
        for state in REFERENCE_STATES:
            for form in DEVIATION_FORMS:
                base_separation = (
                    getattr(separation, form) if separation is not None else float("nan")
                )
                adaptive = separation.adaptive_tau(form) if separation is not None else float("nan")
                found = False
                for arm, arm_scores in scores.items():
                    score = arm_scores.get(system)
                    if score is None:
                        continue
                    taken = take_b(score.samples(state), b)
                    if not taken:
                        continue
                    found = True
                    values = [sample.value(form) for sample in taken]
                    # The specific criterion is per SAMPLE: the same draw must be inside tau of
                    # `state` and outside tau of the other state. An unpaired sample raises rather
                    # than being read as "far from the other state".
                    pairs = score.paired_samples(target_state=state, b=b)
                    specific = [
                        int(
                            any(
                                target.value(form) <= tau and other.value(form) > tau
                                for target, other in pairs
                            )
                        )
                        for tau in taus
                    ]
                    rows.append(
                        [
                            arm,
                            system,
                            state,
                            form,
                            len(taken),
                            min(values),
                            float(np.median(values)),
                            base_separation,
                            adaptive,
                            taken[0].metric_residues,
                            score.coverage.get(state, {}).get("metric").fraction
                            if score.coverage.get(state, {}).get("metric") is not None
                            else "",
                            *[int(min(values) <= tau) for tau in taus],
                            *specific,
                            *[score.separation_stratum(form=form, tau=tau) for tau in taus],
                            int(min(values) <= adaptive),
                        ]
                    )
                if not found:
                    rows.append(
                        [
                            "",
                            system,
                            state,
                            form,
                            0,
                            "",
                            "",
                            base_separation,
                            adaptive,
                            separation.metric_residues if separation is not None else "",
                            separation.metric_coverage.fraction if separation is not None else "",
                            *["" for _ in taus],
                            *["" for _ in taus],
                            *[
                                separation_stratum(
                                    getattr(separation, form) if separation is not None else None,
                                    tau_angstrom=tau,
                                )
                                for tau in taus
                            ],
                            "",
                        ]
                    )
    with path.open("w", newline="") as handle:
        # lineterminator="\n": csv defaults to CRLF, and a stray \r reads as trailing whitespace
        # to `git diff --check`, one of the project's required checks, so the sidecar cannot be
        # committed cleanly. Third writer in this repo to need it -- measure_length_cap.py and
        # check_pocketminer_overlap.py were the first two.
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_figure(
    path: Path,
    separations,
    scores,
    table,
    taus,
    b,
    refused,
    arms,
    systems,
    separation_unavailable,
    scorable,
) -> None:
    """Panel A: the scale tau is measured against. Panel B: the rates, or why there are none."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13.0, 8.6))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.32)

    axis = figure.add_subplot(grid[0, 0])
    ordered = sorted(separations, key=lambda name: separations[name].rmsd)
    positions = np.arange(len(ordered))
    axis.scatter(
        [separations[name].mean_deviation for name in ordered],
        positions,
        s=26,
        color="#3b6ea5",
        label="mean deviation (BioEmu's `local rmsd`)",
    )
    axis.scatter(
        [separations[name].rmsd for name in ordered],
        positions,
        s=26,
        marker="D",
        color="#c0392b",
        label="true RMSD",
    )
    for position, name in zip(positions, ordered, strict=True):
        axis.plot(
            [separations[name].mean_deviation, separations[name].rmsd],
            [position, position],
            color="#999999",
            linewidth=0.8,
            zorder=0,
        )
    for tau, style in zip(taus, ("-", "--", ":"), strict=False):
        axis.axvline(tau, color="#111111", linestyle=style, linewidth=1.0)
        axis.text(
            tau,
            len(ordered) - 0.6,
            f" tau {tau:g} A",
            fontsize=7.5,
            ha="left",
            va="top",
            rotation=90,
        )
    axis.set_yticks(positions)
    axis.set_yticklabels(ordered, fontsize=7)
    axis.set_xscale("log")
    axis.set_xlabel("experimental apo<->holo deviation over the benchmark metric region (A)")
    axis.set_title(
        f"A. The scale tau is measured against ({len(ordered)} of {len(systems)} systems)\n"
        "superposed on `alignment_resid_ranges`, measured on `metric_resid_ranges`, N/CA/C/O",
        fontsize=9.5,
    )
    axis.legend(fontsize=7.5, loc="lower right")
    axis.grid(axis="x", alpha=0.25)
    if separation_unavailable:
        # Named on the figure rather than dropped from it: these are systems whose two crystals do
        # not jointly observe enough of the region, not systems that came out well or badly.
        axis.text(
            0.02,
            -0.13,
            "no apo<->holo separation (both crystals must observe a residue): "
            + ", ".join(sorted(separation_unavailable)),
            transform=axis.transAxes,
            fontsize=7.5,
            va="top",
        )

    axis = figure.add_subplot(grid[0, 1])
    fixed = [row for row in table if row["threshold"] == "fixed"]
    if fixed:
        labels = [
            f"{row['arm']} {row['reference_state']} {row['deviation_form']} "
            f"tau={row['tau_angstrom']:g}"
            for row in fixed
        ]
        rates = [row["success_rate"] for row in fixed]
        lows = [row["success_rate"] - row["ci_low"] for row in fixed]
        highs = [row["ci_high"] - row["success_rate"] for row in fixed]
        y = np.arange(len(fixed))
        axis.barh(y, rates, color="#3b6ea5")
        axis.errorbar(rates, y, xerr=[lows, highs], fmt="none", ecolor="#111111", capsize=2)
        axis.set_yticks(y)
        axis.set_yticklabels(labels, fontsize=7)
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("success@B (fraction of scored systems)")
        axis.set_title(
            f"B. success@B, B = {fixed[0]['b_requested'] or fixed[0]['b_effective_min']}, "
            f"n = {fixed[0]['n_systems']} systems",
            fontsize=9.5,
        )
    else:
        axis.axis("off")
        covered = sorted({name for arm in scores.values() for name in arm})
        lines = [
            "B. NO SUCCESS RATE IS REPORTABLE",
            "",
            f"benchmark systems with upstream ranges:   {len(systems)} of 34",
            f"systems covered by the supplied ensemble: {len(covered)}",
            "",
            "The benchmark systems are held out of this project's training pool",
            "by construction (holdout.py excludes them on UniProt accession AND",
            "PDB entry), so no ensemble generated from that pool can cover them.",
            "Reporting 0 here is the coverage, not a score of 0.",
            "",
            "Scorable once samples exist, by reference state:",
            f"  apo  {len(scorable.get(APO, ())):2d} of {len(systems)}",
            f"  holo {len(scorable.get(HOLO, ())):2d} of {len(systems)}",
            "  (a system needs >= the coverage floor of BOTH ranges on that",
            "   reference; a generated sample observes every residue itself)",
            "",
            "What would need generating:",
            f"  {len(systems)} systems x B draws, ~2 s/draw",
            "  at B = 100 that is about 2 GPU-h per arm",
            "  sequences: the benchmark test-case sequences (testcases.csv)",
            "  sampler: the settings recorded in this artifact, unchanged",
            "",
            "Excluded from scoring regardless of generation:",
        ]
        for name, reason in sorted(refused.items()):
            lines.append(f"  {name}:")
            lines += textwrap.wrap(
                reason, width=62, initial_indent="    ", subsequent_indent="    "
            )
        axis.text(
            0.0,
            1.0,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=8.5,
            family="monospace",
            transform=axis.transAxes,
        )

    figure.suptitle(
        "D-63 success@B on the BioEmu cryptic-pocket benchmark "
        f"({POPULATION}; arms: {', '.join(arms) or 'none'})",
        fontsize=11.5,
    )
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
