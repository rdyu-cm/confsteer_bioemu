#!/usr/bin/env python3
"""Steer one benchmark with a shipped adapter and score it. One command per arm.

    python scripts/run_benchmark.py --adapter kinase_uncond_p1
    python scripts/run_benchmark.py --adapter cryptic_uncond_m1 --split test --systems 2 --draws 10

Each shipped adapter was trained toward ONE state, and this scores it against that state only:
the other reference is a dose the adapter never saw, and reporting it would measure nothing the
arm was asked to do. The level therefore comes from the checkpoint, not from a flag.

WHAT IT RUNS, per benchmark:

* **cryptic** needs a matched UNSTEERED control store and refuses to generate without one, so this
  generates the control first, then the steered draws against it. Both use the same draw seeds,
  which is what makes the comparison paired rather than two independent samples.
* **domain motion** and **kinase** generate their own controls inline, so they are one pass.

**GPU AND NETWORK.** Generation runs BioEmu, and BioEmu needs per-sequence embeddings it computes
through the ColabFold MSA server on first use -- about 21 MB and a few minutes for each system,
cached under ``--embeds`` and reused afterwards. ``--systems`` exists because of that: the default
is a handful, so the documented command finishes rather than appearing to hang, and ``--systems 0``
asks for the full panel the paper reports.

**FOR YOUR OWN PROTEIN, USE ``confsteer steer-apply --adapter <arm> --fasta yours.fasta``.**
This script is for the BENCHMARK systems: it scores, and scoring needs the two reference
conformers, which exist for those systems and not for an arbitrary sequence. ``steer-apply``
drives the same adapters at the same sites on any FASTA and generates structures without scoring.

**THIS REPRODUCES THE PIPELINE, NOT THE PUBLISHED TABLE**, unless it is run at the paper's depth
(``--systems 0 --draws 100``) on every arm. A two-system run is a smoke test of the machinery.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from sae_bioemu.runtime_env import use_importable_matplotlib_backend

# Every stage below runs as a child process and inherits this environment, so repair it once here.
use_importable_matplotlib_backend()

# ... and the same for BioEmu's CuBLAS determinism warning, which PyTorch repeats once per
# generation. The children are separate interpreters, so the filter has to travel as an environment
# variable rather than through warnings.filterwarnings. NOT CUBLAS_WORKSPACE_CONFIG: that changes
# how the kernels are chosen and so could change the numbers.
os.environ["PYTHONWARNINGS"] = ",".join(
    value
    for value in (
        os.environ.get("PYTHONWARNINGS", ""),
        "ignore:Deterministic behavior was enabled",
    )
    if value
)

REPOSITORY = Path(__file__).resolve().parents[1]
DATA = REPOSITORY / "data/benchmarks"
MODELS = REPOSITORY / "models"

#: arm -> (benchmark, the reference the arm trained toward). The trained LEVEL is read from the
#: checkpoint rather than restated here, so the two can never disagree.
ARMS = {
    "cryptic_uncond_m1": ("cryptic", "apo"),
    "cryptic_uncond_p1": ("cryptic", "holo"),
    # Nearer and further are relative to the UNSTEERED ensemble, which is the order the arms were
    # trained on and not the deposited one.
    "domain_motion_uncond_m1": ("domain_motion", "nearer conformer"),
    "domain_motion_uncond_p1": ("domain_motion", "further conformer"),
    "kinase_uncond_m1": ("kinase", "DFG-in"),
    "kinase_uncond_p1": ("kinase", "DFG-out"),
}

TAU = {"cryptic": 1.0, "domain_motion": 3.0, "kinase": 2.0}

#: The cryptic pool holds 34 and the reported panel is 32: one system has no recoverable metric
#: ranges upstream, and one resolves too little of its apo/holo alignment to measure a separation.
#: Named here because a rate is an unweighted mean over systems, so the denominator is part of it.
CRYPTIC_SCORED_PANEL = 32

#: The fewest scorable draws a test-split system needs to be scored. At the reported depth this
#: is part of the statistic -- it leaves 121 of the cryptic test systems -- so it holds there;
#: a shallower run lowers it to its own depth rather than scoring nothing.
MIN_PAIRED_DRAWS = 5


def here(path: Path) -> str:
    """A path as the reader would type it: relative to the working directory where it can be."""
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def run(
    command: list[str],
    label: str,
    log: Path,
    partial_is_expected: bool = False,
    show_log: bool = False,
) -> None:
    """Run one stage into its own log file, printing one line and refusing on failure.

    A stage's own output is a scorer table, a progress stream or a refusal paragraph, and hours of
    it scrolls past whatever the run actually concluded. It goes to ``log`` instead, which stays for
    reading afterwards, and the tail is printed when the stage fails -- which is when it is wanted.
    ``show_log`` names the file up front for the stages long enough to want watching.

    ``partial_is_expected`` covers the ONE case where a nonzero exit is the right answer: the
    scorers exit 1 when the population is restricted, because a rate over a subset is a different
    denominator wearing the panel's name. A run limited by ``--systems`` is exactly that, so the
    exit is not a failure -- otherwise the command this repository documents always ends in
    REFUSING and looks broken.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"{label}{f' (log: {here(log)})' if show_log else ''}...", end="", flush=True)
    with log.open("w") as handle:
        result = subprocess.run(
            command,
            cwd=REPOSITORY,
            stdout=handle,
            stderr=subprocess.STDOUT,
            # Line by line, so the log a user was told to follow fills as the stage runs.
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    if result.returncode == 0 or (partial_is_expected and result.returncode == 1):
        print(" done", flush=True)
        return
    print(" FAILED", flush=True)
    tail = log.read_text(errors="replace").splitlines()[-30:]
    print("\n".join(tail), flush=True)
    raise SystemExit(f"REFUSING: {label} exited {result.returncode}")


def confirm_coverage(
    sidecar: Path,
    expected: int,
    state: str,
    depth: int,
    arm: str = "",
    reference: str = "",
    panel: int = 0,
) -> None:
    """Read back what was scored and refuse unless it covers the population that was asked for.

    The scorers exit 1 for two unrelated reasons: a restricted POPULATION, which is a real
    incompleteness, and -- for kinase -- scoring one STATE of two, which is what a single-direction
    arm is for and is never going to change. Treating the exit code alone as the verdict therefore
    either fails a correct full-panel run or hides a run that silently lost systems. So the exit
    code is not the gate: this is. It reads the sidecar the scorer wrote and checks the system
    count against the roster, which is the thing that would actually invalidate a rate.
    """
    if not sidecar.is_file():
        raise SystemExit(
            f"REFUSING: the scorer wrote no {sidecar.name}, so nothing was scored. Its log in "
            f"{here(sidecar.parent.parent / 'logs')} says why -- most often no draws were "
            "generated for the selected systems"
        )
    payload = json.loads(sidecar.read_text())
    # THE SIDECAR MUST BE THIS RUN'S. ``run`` treats exit 1 as an expected partial result, so a
    # scorer that refused for an unrelated reason leaves the PREVIOUS run's file on disk and this
    # gate would read it as the answer -- at whatever depth that run used.
    written = payload.get("b_requested") or payload.get("b")
    if written is None:
        for cell in payload.get("cells") or payload.get("success_table") or []:
            written = cell.get("b_requested") or cell.get("b")
            if written is not None:
                break
    if written is not None and int(written) != int(depth):
        raise SystemExit(
            f"REFUSING: {sidecar.name} reports B={written} but this run asked for {depth}. It is "
            "left over from an earlier run; the scorer above refused and wrote nothing. Delete the "
            "scored/ directory and rerun"
        )
    # ``cells`` for kinase and domain motion, ``success_table`` for cryptic pockets. One reader, so
    # the cryptic path gets the same denominator check as the other two instead of none.
    rows = payload.get("cells") or payload.get("success_table") or []
    cells = [
        cell
        for cell in rows
        if cell.get("reference_state", state) == state
        and "success_rate" in cell
        # cryptic emits four forms per state; the field-comparable one is the PLAIN criterion on
        # rmsd. 'specific' additionally requires the draw to sit outside tau of the other state,
        # which is a different question and not what the paper reports.
        and cell.get("criterion", "plain") == "plain"
        and cell.get("deviation_form", "rmsd") == "rmsd"
    ]
    if not cells:
        raise SystemExit(f"REFUSING: {sidecar} records no scored cell for {state}")
    scored = max(int(cell.get("n_systems", 0)) for cell in cells)
    if scored != expected:
        raise SystemExit(
            f"REFUSING: {sidecar.name} scored {scored} systems, not the {expected} this run "
            "asked for. A rate over a different population is a different statistic"
        )
    for cell in sorted(cells, key=lambda c: c.get("b_requested") or c.get("b") or 0):
        tau = cell.get("tau_angstrom")
        if tau is None or (isinstance(tau, float) and math.isnan(tau)):
            continue  # the adaptive cell has no single threshold; the fixed-tau one is the report
        cell_depth = cell.get("b_requested") or cell.get("b")
        print(
            f"{arm} -> {reference}: success@{cell_depth} = {cell['success_rate']:.3f} "
            f"over {scored} of {panel or scored} systems (tau {tau} A)"
        )
    print(f"results: {here(sidecar)}")
    # WHAT THIS NUMBER IS NOT. The published rate is the whole panel at 100 draws; anything less is
    # a different denominator or a shallower estimate, and says so rather than looking like the
    # paper's value with a different number in it.
    if scored < (panel or scored) or int(depth) < 100:
        print("subset run; --systems 0 --draws 100 gives the reported rate")


def report_paired_shift(sidecar: Path, arm: str, reference: str, level: float) -> None:
    """The test-split result in one line: the mean paired RMSD change and the systems behind it.

    The two paired-shift scorers write the same quantity under different shapes -- the cryptic one
    keys its cells by level, the domain-motion one has a single top-level cell -- so both are read
    here rather than restating either schema in prose.
    """
    payload = json.loads(sidecar.read_text())
    cells = payload.get("levels") or {}
    cell = cells.get(str(float(level)))
    if cell is None:
        cell = next(iter(cells.values())) if len(cells) == 1 else payload
    mean, systems = cell.get("mean_delta_rmsd"), cell.get("n_systems")
    if mean is None or systems is None:
        raise SystemExit(
            f"REFUSING: {sidecar.name} records no mean shift for level {level:+.0f}, so this run "
            "scored nothing for the arm's own reference"
        )
    print(
        f"{arm} -> {reference}: mean RMSD change (on - off) {float(mean):+.3f} A "
        f"over {systems} systems"
    )
    print(f"results: {here(sidecar)}")


def trained_level(adapter: Path) -> float:
    import torch

    payload = torch.load(adapter, map_location="cpu", weights_only=False)
    levels = payload["adapter_config"].get("target_levels")
    if not levels:
        raise SystemExit(f"REFUSING: {adapter.name} records no target level")
    if len(levels) != 1:
        raise SystemExit(f"REFUSING: {adapter.name} is not single-direction; levels {levels}")
    return float(levels[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # ``--adapter`` names the SAME shipped model steer-apply takes, so one name means one
    # thing across the two commands.
    parser.add_argument("--adapter", required=True, choices=sorted(ARMS), dest="arm")
    parser.add_argument("--split", default="benchmark", choices=("benchmark", "test"))
    parser.add_argument(
        "--systems", type=int, default=2, help="how many systems; 0 means the whole panel"
    )
    parser.add_argument("--draws", type=int, default=10, help="draws per system (paper: 100)")
    parser.add_argument(
        "--out-dir", type=Path, default=REPOSITORY / "outputs/benchmarks", dest="out"
    )
    parser.add_argument(
        "--embeds",
        type=Path,
        default=REPOSITORY / "outputs/embeds",
        help="BioEmu embedding cache; computed on first use and reused",
    )
    parser.add_argument(
        "--system",
        action="append",
        default=None,
        dest="named_systems",
        metavar="ID",
        help="run these systems by name, repeatable. Overrides --systems, which only takes the "
        "first N alphabetically and so cannot ask for a particular protein",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-only", action="store_true", help="skip generation")
    arguments = parser.parse_args()

    # Relative to where the user is, and made absolute before any child runs: children run from
    # the repository root, so a relative path would name a different directory there.
    arguments.out, arguments.embeds = arguments.out.resolve(), arguments.embeds.resolve()
    benchmark, reference = ARMS[arguments.arm]
    if benchmark == "kinase" and arguments.split == "test":
        raise SystemExit(
            "the kinase benchmark IS its held-out split -- the 20 genes are disjoint from the "
            "41 the arms trained on -- so --split test would name the same systems"
        )
    adapter = MODELS / f"{arguments.arm}.pt"
    level = trained_level(adapter)
    destination = arguments.out / f"{arguments.arm}_{arguments.split}"
    so3 = REPOSITORY / "data/processed/so3_cache"
    python = sys.executable

    logs = destination / "logs"

    # The pool the run actually uses. ``generate_benchmark_ensemble.py`` takes a pool and no system
    # filter, so without this a two-system smoke run would still generate the unsteered control for
    # all 34 cryptic systems -- the expensive half, and the reason a "quick" run was not quick.
    pool = (
        DATA
        / {
            ("cryptic", "benchmark"): "pools/cryptic_benchmark.json",
            ("cryptic", "test"): "pools/cryptic_testsplit.json",
            ("domain_motion", "benchmark"): "pools/domain_motion.json",
            ("domain_motion", "test"): "pools/domain_motion_testsplit.json",
            ("kinase", "benchmark"): "pools/kinase.json",
        }[(benchmark, arguments.split)]
    )

    # The full roster is read whatever the run asks for, because the coverage gate needs to know
    # the population a complete run WOULD have scored, not just the one this run selected.
    payload = json.loads(pool.read_text())
    all_systems = (
        [str(e["sequence_id"]) for e in payload["entries"]]
        if isinstance(payload, dict)
        else [str(name) for name in payload]
    )
    systems: list[str] = []
    if arguments.systems or arguments.named_systems:
        names = all_systems
        if arguments.named_systems:
            unknown = sorted(set(arguments.named_systems) - set(names))
            if unknown:
                raise SystemExit(
                    f"REFUSING: {unknown} are not in this benchmark's {arguments.split} roster. "
                    f"It has {len(names)} systems, e.g. {sorted(names)[:3]}"
                )
            systems = sorted(set(arguments.named_systems))
        else:
            systems = sorted(names)[: arguments.systems]

    if systems:
        entries = [
            entry
            for entry in json.loads(pool.read_text())["entries"]
            if str(entry["sequence_id"]) in set(systems)
        ]
        destination.mkdir(parents=True, exist_ok=True)
        pool = destination / "pool.json"
        pool.write_text(json.dumps({"entries": entries}, indent=1) + "\n")

    # The cryptic BENCHMARK scorer measures against the DEPOSITED apo and holo entries, which ship as
    # a fetch rather than as files. Fetched HERE, before anything is generated: left to the scorer,
    # a missing entry surfaced only after every draw had been generated, so hours of GPU work ended
    # in FileNotFoundError. Skips what is already present, so a rerun costs nothing. The test split
    # reads its references from a shipped artifact and needs none of this.
    if benchmark == "cryptic" and arguments.split == "benchmark":
        run(
            [python, "scripts/release/fetch_structures.py"],
            "fetching the deposited apo/holo entries",
            logs / "fetch.log",
        )

    # BioEmu FAILS CLOSED on a missing embedding, rather than computing one mid-generation: a
    # ColabFold query from inside a GPU job is how a run silently becomes a network-bound one. So
    # the cache is built first, as its own restartable step.
    if not arguments.score_only:
        run(
            [
                python,
                "scripts/cryptic_steering/build_embedding_caches.py",
                "--pool",
                str(pool),
                "--cache-dir",
                str(arguments.embeds),
                "--report",
                str(destination / "embeddings.json"),
            ],  # fmt: skip
            f"embedding {len(systems) or len(all_systems)} sequences",
            logs / "embeddings.log",
            show_log=True,
        )

    count = len(systems) or len(all_systems)
    generating = f"generating {count} systems x {arguments.draws} draws"
    if not arguments.score_only:
        if benchmark == "cryptic":
            control = destination / "control"
            command = [
                python, "scripts/cryptic_steering/generate_benchmark_ensemble.py",
                "--pool", str(pool), "--output-dir", str(control),
                "--cache-embeds-dir", str(arguments.embeds), "--cache-so3-dir", str(so3),
                "--draws", str(arguments.draws),
            ]  # fmt: skip
            run(
                command,
                f"{generating}: unsteered controls",
                logs / "control.log",
                show_log=True,
            )
            command = [
                python, "scripts/cryptic_steering/generate_benchmark_steered.py",
                "--pool", str(pool), "--adapter", str(adapter),
                "--train-run-report", str(adapter.with_suffix(".json")),
                "--output-dir", str(destination / "steered"),
                "--native-index", str(control / "final_structures/index.jsonl"),
                "--cache-embeds-dir", str(arguments.embeds), "--cache-so3-dir", str(so3),
                "--level", str(level), "--draws", str(arguments.draws),
                "--device", arguments.device,
            ]  # fmt: skip
            for name in systems:
                command += ["--system", name]
            run(
                command,
                f"{generating}: steered",
                logs / "generate.log",
                show_log=True,
            )
        else:
            script = {
                "domain_motion": "scripts/domain_motion/generate_benchmark_draws.py",
                "kinase": "scripts/kinase_dfg/generate_kinase_benchmark_draws.py",
            }[benchmark]
            artifact = {
                ("domain_motion", "benchmark"): DATA / "references/domain_motion_benchmark.pt",
                ("domain_motion", "test"): DATA / "references/domain_motion_testsplit.pt",
                ("kinase", "benchmark"): DATA / "references/kinase.pt",
            }[(benchmark, arguments.split)]
            command = [
                python, script, "--artifact", str(artifact), "--adapter", str(adapter),
                "--train-run-report", str(adapter.with_suffix(".json")), "--output-dir", str(destination),
                "--cache-embeds-dir", str(arguments.embeds), "--cache-so3-dir", str(so3),
                "--draws", str(arguments.draws), "--level", str(level),
                "--batch-size", "1", "--device", arguments.device,
            ]  # fmt: skip
            # The role is RECORDED on every draw. Left at its default, a test-split run writes
            # held-out draws under the benchmark's name in every downstream sidecar, and nothing
            # further on can tell them apart.
            if benchmark == "domain_motion":
                command += ["--split-role", arguments.split]
            for name in systems:
                command += ["--sequence", name]
            run(command, generating, logs / "generate.log", show_log=True)

    # ------------------------------------------------------------------------------- scoring
    # Each scorer emits BOTH references whatever it is given; the arm's own is selected here, and
    # the report says which, because scoring a single-direction arm on the other reference is the
    # one mistake that produces a plausible number for a dose the adapter never saw.
    tau = TAU[benchmark]
    scored = destination / "scored"
    scored.mkdir(parents=True, exist_ok=True)
    # CLEARED FIRST. `run` treats a scorer's exit 1 as an expected partial result, and every refusal
    # in the scorers is also exit 1 -- so a scorer that refused would leave the PREVIOUS run's file
    # here, and the gate below would print it as this run's answer. With nothing left over, a
    # refusal means no file, and the gate says so.
    for leftover in (
        "success_at_b.json", "success_at_b_draws.csv", "success_at_b.csv", "success_at_b.png",
        "paired_shift.json", "paired_shift.csv",
    ):  # fmt: skip
        (scored / leftover).unlink(missing_ok=True)
    label = f"{arguments.arm}_{arguments.split}"
    if benchmark == "cryptic":
        views = destination / "views"
        run(
            [
                python,
                "scripts/community/adapt_cryptic_steered_index.py",
                "--store",
                str(destination / "steered"),
                "--out-dir",
                str(views),
                "--depth",
                str(arguments.draws),
            ],  # fmt: skip
            "building a scoreable view",
            logs / "view.log",
        )
        index = next(views.rglob("index.jsonl"), None)
        if index is None:
            raise SystemExit(f"REFUSING: no index.jsonl under {views}")
        if arguments.split == "test":
            # The test split is scored as a PAIRED SHIFT against the artifact's own frames, the
            # metric the arms were trained on, not as success@B: success@B is defined on the
            # 34-system benchmark panel and its rate is not comparable across populations.
            run(
                [
                    python,
                    "scripts/community/score_cryptic_testsplit.py",
                    "--store",
                    str(destination / "steered"),
                    "--pairing",
                    str(DATA / "rosters/cryptic_testsplit_apo_holo.json"),
                    "--structures",
                    str(DATA / "structures"),
                    "--reference-artifact",
                    str(DATA / "references/cryptic_testsplit.pt"),
                    "--out-json",
                    str(scored / "paired_shift.json"),
                    "--out-csv",
                    str(scored / "paired_shift.csv"),
                    "--min-draws",
                    str(min(MIN_PAIRED_DRAWS, arguments.draws)),
                ],  # fmt: skip
                "scoring",
                logs / "score.log",
            )
            report_paired_shift(scored / "paired_shift.json", arguments.arm, reference, level)
            return 0
        run(
            [
                python,
                "scripts/cryptic_steering/score_success_at_b.py",
                "--samples",
                str(index),
                "--arm",
                label,
                "--b",
                str(arguments.draws),
                "--tau",
                str(tau),
                "--coverage-floor",
                "0.8",
                "--out-json",
                str(scored / "success_at_b.json"),
                "--out-csv",
                str(scored / "success_at_b.csv"),
                "--out-png",
                str(scored / "success_at_b.png"),
            ],  # fmt: skip
            "scoring",
            logs / "score.log",
            # The scorer reports BOTH states in one table, by design: that is the field-comparable
            # form. So it always looks 'complete' and its exit code says nothing about which state
            # this arm was trained toward. The gate below selects the arm's own state by NAME and
            # checks the denominator, exactly as the kinase and domain-motion paths do.
            partial_is_expected=True,
        )
        confirm_coverage(
            scored / "success_at_b.json",
            len(systems) or CRYPTIC_SCORED_PANEL,
            "apo" if level < 0 else "holo",
            arguments.draws,
            arm=arguments.arm,
            reference=reference,
            panel=CRYPTIC_SCORED_PANEL,
        )
    elif benchmark == "domain_motion":
        level_tag = "m1" if level < 0 else "p1"
        index = destination / "views" / f"level_{level_tag}" / "final_structures/index.jsonl"
        if arguments.split == "benchmark":
            # THE PUBLISHED BENCHMARK NUMBER IS success@B ON ``multiconf_domainmotion``: true RMSD
            # over N/CA/C/O with mdtraj's own superposition. The test-split scorer below measures
            # the METRIC REGION the arms were trained on, which is a different quantity -- it gave
            # 20 of 22 systems where this gives 19 -- so the two are not interchangeable and the
            # split decides which one runs.
            run(
                [
                    python,
                    "scripts/domain_motion/measure_success_at_b.py",
                    "--assets",
                    str(DATA / "domain_motion_assets"),
                    "--index",
                    str(index),
                    "--split-record",
                    str(DATA / "domain_motion_split_record.json"),
                    "--tau",
                    str(tau),
                    "--b",
                    str(arguments.draws),
                    "--label",
                    label,
                    # WITHOUT THIS the cells are the DEPOSITED conformer order, which is arbitrary
                    # and disagrees with the order these arms were trained on in 17 of 22 systems.
                    "--canonical-labels",
                    str(DATA / "rosters/domain_motion_canonical_labels.json"),
                    "--output",
                    str(scored / "success_at_b.json"),
                    "--draws-csv",
                    str(scored / "success_at_b_draws.csv"),
                ],  # fmt: skip
                "scoring",
                logs / "score.log",
                partial_is_expected=bool(systems),
            )
            confirm_coverage(
                scored / "success_at_b.json",
                len(systems) or len(all_systems),
                "canonical_nearer" if level < 0 else "canonical_further",
                arguments.draws,
                arm=arguments.arm,
                reference=reference,
                panel=len(all_systems),
            )
            return 0
        artifact = DATA / "references/domain_motion_testsplit.pt"
        run(
            [
                python,
                "scripts/community/score_dm_testsplit.py",
                "--store",
                str(destination / "store"),
                "--reference-artifact",
                str(artifact),
                "--level",
                str(level),
                "--tau",
                str(tau),
                "--out-json",
                str(scored / "paired_shift.json"),
                "--out-csv",
                str(scored / "paired_shift.csv"),
                "--min-draws",
                str(min(MIN_PAIRED_DRAWS, arguments.draws)),
            ],  # fmt: skip
            "scoring",
            logs / "score.log",
        )
        report_paired_shift(scored / "paired_shift.json", arguments.arm, reference, level)
        return 0
    else:
        state = "dfg_in" if level < 0 else "dfg_out"
        run(
            [
                python,
                "scripts/kinase_dfg/measure_success_at_b.py",
                "--artifact",
                str(DATA / "references/kinase.pt"),
                "--index",
                str(destination / "views" / f"level_{state}/final_structures/index.jsonl"),
                "--index-state",
                state,
                "--tau",
                str(tau),
                "--b",
                str(arguments.draws),
                "--label",
                label,
                "--output",
                str(scored / "success_at_b.json"),
                "--draws-csv",
                str(scored / "success_at_b_draws.csv"),
                # Hand the scorer its own smoke flag, so the restriction is RECORDED in the
                # sidecar (complete=false) rather than inferred from a short systems list.
                *[flag for name in systems for flag in ("--smoke-system", name)],
            ],  # fmt: skip
            "scoring",
            logs / "score.log",
            # Always: this scorer reports a single-direction arm as incomplete because it scored
            # one of the artifact's two states. confirm_coverage is the gate instead.
            partial_is_expected=True,
        )
        confirm_coverage(
            scored / "success_at_b.json",
            len(systems) or len(all_systems),
            state,
            arguments.draws,
            arm=arguments.arm,
            reference=reference,
            panel=len(all_systems),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
