"""Generate a STEERED BioEmu ensemble for the cryptic-pocket benchmark systems (D-68).

This is the arm that ``benchmark_native`` is the control for. It answers the one question the
cryptic programme exists to answer and that no trained adapter has ever been asked: does steering
change ``success@B`` on the 34-system ``multiconf_crypticpocket`` benchmark?

WHAT THIS DOES AND DOES NOT NEED. A steered draw needs the trained adapter, the intervention site
it was trained at, a residue mask, and the LEVEL ``o`` -- and nothing else. The adapter is
conditioned on ``o`` alone (``steering_properties.generate_exact_property_sample``: "the target is
the two-state displacement t = the progress coordinate_state(o) - the progress coordinate_reporting(control) and the adapter is conditioned
on o"), so generating on a held-out system needs no pocket artifact, no apo sidecar and no
``sigma_the progress coordinate`` table. Those define ``the progress coordinate``, which is the MECHANISM coordinate; this script computes no
``the progress coordinate`` at all. The benchmark metric is RMSD to a deposited structure and is scored post hoc by
``scripts/cryptic_steering/score_success_at_b.py`` off the structures written here.

The adapter must therefore be an ARM C checkpoint -- ``site_conditioning`` absent. An arm A/B
checkpoint conditions on a per-residue channel measured on the deposited apo structure, which does
not exist for a held-out benchmark system, and this script refuses one by name rather than passing
``site_field=None`` and silently generating under a channel of zeros the adapter never saw.

PAIRING. Every draw here uses the SAME seed as the native draw of the same system, so the two arms
differ only by the adapter. ``SeededPriorSampling`` reseeds the global RNG per batch MEMBER from
that member's own seed and replays every earlier field at that member's own shape, so the prior
draw is a function of ``(seed, residue count)`` and not of batch composition -- which is what lets
this run at a different batch size from the native one and stay matched. That is not taken on
faith: ``--replay-control-draws`` regenerates the first N unsteered draws of each system through
the same sampler and compares them to the native store on disk.

RETENTION, WHICH IS THE FAILURE THIS SCRIPT IS MOST LIKELY TO HAVE HAD.
``BioEmuRuntimeActivationSampler.steer`` NEVER persists a structure -- only the adapter-off path
calls ``_persist_final_structures`` -- so a run can set ``save_final_structures=True``, burn
GPU-hours and write nothing but controls. That has already destroyed one measurement on this
project. Every steered draw here goes through ``steering/training_structures.SteeredDrawRecorder``,
which reads the raw ChemGraph off ``BioEmuSteeredSample.sampled_batch`` on the way past, and the
first recorded batch of every run is VERIFIED against two references before the run continues:

  * the persisted ``.npz`` must reproduce the coordinates ``steer()`` returned, exactly; and
  * it must DIFFER from the native unsteered draw at the same seed.

A file count is not evidence, and neither is either check alone: the first without the second
passes for a store full of controls.

RESUME. One row per persisted steered draw lands in ``steered_draws.jsonl``, flushed and fsynced by
the store, and a restarted run skips every ``(sequence_id, seed)`` already recorded. Resume is at
the DRAW, so a preemption costs at most the batch in flight -- seconds to a minute -- which is what
lets this launcher take a preemptible queue. Each system's timing is appended to a progress log as soon as
that system finishes, so an interrupted run still says how far it got.

One GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sae_bioemu.bioemu.cryptic_generation import require_cached_embeddings  # noqa: E402
from sae_bioemu.bioemu.extractor import (  # noqa: E402
    BioEmuRuntimeActivationSampler,
    SequenceRecord,
)
from sae_bioemu.bioemu.steering import SteeringSite  # noqa: E402
from sae_bioemu.bioemu.steering_properties import batch_residue_mask  # noqa: E402
from sae_bioemu.core.config import TraceConfig, seed_everything  # noqa: E402
from sae_bioemu.steering.adapter_io import adapter_attributes, load_adapter  # noqa: E402
from sae_bioemu.steering.training_structures import (  # noqa: E402
    SteeredDrawRecorder,
    TrainingStructureStore,
    read_steered_draws,
    record_steered_draws,
)

#: The property this script is for. A radius-of-gyration checkpoint conditions on a standardized
#: absolute value rather than a two-state level, so ``--level +1`` would mean something else
#: entirely and the run would look complete.
CRYPTIC_PROPERTY = "pocket_progress_coordinate"
#: The cryptic-pocket properties this generator will draw for. Both steer the same benchmark on the
#: same apo/holo references; they differ in what a LEVEL means, which matters for interpreting the
#: result and not for producing the draws. `pocket_progress_coordinate` is the the progress coordinate-based
#: dimensionless coordinate; `pocket_conformer_rmsd` is the Angstrom-scaled RMSD target the
#: unconditioned arms train. The store is labelled with the adapter's OWN property so a sidecar can
#: never claim a property the checkpoint does not have.
CRYPTIC_PROPERTIES = (CRYPTIC_PROPERTY, "pocket_conformer_rmsd")

#: The two levels the cryptic arms were trained on (``schedule.train_target_levels``). ``+1`` asks
#: for the holo state and ``-1`` for the apo state. Nothing else was ever presented to the adapter,
#: so nothing else is accepted here.
TRAINED_LEVELS = (-1.0, 1.0)

#: The split label written into every sidecar row. These systems are in no training split; the
#: field exists because the store's resume counter keys on it.
SPLIT_LABEL = "benchmark"

#: Below this the persisted structure is the one ``steer()`` returned. It is expected to be 0.0 --
#: the writer stores ``chemgraph.pos`` and the returned coordinates are a reshape of the same
#: tensor -- and the tolerance exists only so a future dtype view cannot fail the run spuriously.
RETENTION_MATCH_TOLERANCE_NM = 1.0e-9

#: Above this the persisted structure is NOT the native control. One Angstrom is 0.1 here, so this
#: is four orders of magnitude below any structural difference and is a "these are not the same
#: file" test, not a "the adapter did something useful" test.
RETENTION_DIFFERENCE_FLOOR_NM = 1.0e-6


@dataclass(frozen=True)
class BenchmarkUnit:
    """One scheduled steered draw, in the shape ``TrainingStructureStore`` reads.

    ``seed`` keys the steered structure and ``generation_seed`` keys its control. Here they are
    the SAME integer, deliberately: the control is the native draw of the same system at the same
    seed, which already exists on disk, so the sidecar's ``control_pdb``/``control_npz`` resolve
    into the native store through the ``control`` symlink this script plants.
    """

    split: str
    sequence_id: str
    cycle: int
    visit: int
    level: float
    seed: int
    generation_seed: int


def read_native_index(index_path: Path) -> dict[tuple[str, int], Path]:
    """``(sequence_id, sample_index) -> npz path`` for the native control store."""
    root = index_path.parent.parent
    native: dict[tuple[str, int], Path] = {}
    for line in index_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        key = (str(record["sequence_id"]), int(record["sample_index"]))
        native[key] = root / str(record["npz_path"])
    return native


def load_intervention_site(report_path: Path) -> tuple[SteeringSite, dict[str, object]]:
    """Rebuild the TRAINING site from the training run report beside the checkpoint.

    Read rather than re-resolved from ``(timestep, layer)``. Re-resolving would recompute the
    denoiser's timestep grid and could land on a neighbouring score call if any sampler setting
    differed, and an adapter evaluated at a site it was not trained at is a different experiment
    that reports a complete-looking number. The recorded block names the absolute score call.
    """
    payload = json.loads(report_path.read_text())
    site_spec = payload.get("intervention_site")
    if not isinstance(site_spec, dict):
        raise SystemExit(
            f"{report_path} carries no intervention_site block, so the site the checkpoint was "
            "trained at is unknown and cannot be reproduced"
        )
    # BOTH RECORD SHAPES. A MULTI-SITE arm's identity carries the plural ``resolved_timesteps`` and
    # deliberately NONE of the scalar single-site keys, so that a reader cannot mistake it for a
    # single-site record by finding a call index describing only one of the calls actually
    # intervened on. This function used to read ``site_spec["resolved_timestep"]`` unconditionally,
    # which is correct for the single-site arms it was written for and raises
    # ``KeyError: 'resolved_timestep'`` for every multisite8 arm -- MEASURED, a measured run and
    # a measured run, dead in 13 s and 25 s against the cryptic multisite8 checkpoint. ``score_calls`` is
    # already plural in both shapes. scripts/kinase_dfg/generate_kinase_benchmark_draws.py carries
    # the same reconstruction for the same reason.
    if "resolved_timesteps" in site_spec:
        expected_timesteps = tuple(float(value) for value in site_spec["resolved_timesteps"])
    elif "resolved_timestep" in site_spec:
        expected_timesteps = (float(site_spec["resolved_timestep"]),)
    else:
        raise SystemExit(
            f"{report_path}'s intervention_site names neither resolved_timesteps nor "
            "resolved_timestep, so the timesteps the selected score calls must be observed at are "
            "unknown. Generating without them would let a wrong call index pass silently"
        )
    site = SteeringSite(
        module_path=str(site_spec["module_path"]),
        score_calls=tuple(int(value) for value in site_spec["score_calls"]),
        expected_timesteps=expected_timesteps,
        expected_score_call_total=int(site_spec["expected_score_call_total"]),
        timestep_tolerance=float(site_spec["timestep_tolerance"]),
    )
    return site, site_spec


def select_systems(
    pool_ids: list[str],
    population_json: Path | None,
    explicit: list[str] | None,
    expect_systems: int | None,
) -> tuple[list[str], dict[str, object]]:
    """Choose the systems to generate, READING a committed scorable set rather than deriving one.

    ``--population-from`` takes a published ``success_at_b`` artifact and keeps
    ``systems_with_ranges`` minus ``excluded_systems``. That is the 28 D-63 and D-67 report on, and
    it is read off the artifact of record so this script cannot quietly disagree with the scorer
    about which systems the denominator holds. ``--expect-systems`` turns a changed artifact into a
    refusal instead of a silently different population.
    """
    provenance: dict[str, object] = {"pool_systems": len(pool_ids)}
    if explicit:
        chosen = [name for name in pool_ids if name in set(explicit)]
        missing = sorted(set(explicit) - set(chosen))
        if missing:
            raise SystemExit(f"--system named systems that are not in the pool: {missing}")
        provenance["source"] = "explicit --system"
    elif population_json is not None:
        payload = json.loads(population_json.read_text())
        with_ranges = {str(name) for name in payload["systems_with_ranges"]}
        excluded = {str(name) for name in payload["excluded_systems"]}
        scorable = with_ranges - excluded
        chosen = [name for name in pool_ids if name in scorable]
        provenance["source"] = str(population_json)
        provenance["systems_with_ranges"] = len(with_ranges)
        provenance["excluded_systems"] = sorted(excluded)
    else:
        chosen = list(pool_ids)
        provenance["source"] = "the whole pool"
    provenance["selected"] = list(chosen)
    if expect_systems is not None and len(chosen) != expect_systems:
        raise SystemExit(
            f"selected {len(chosen)} systems but --expect-systems asked for {expect_systems}: "
            f"{sorted(chosen)}. The population changed under the run; refusing rather than "
            "measuring a different denominator"
        )
    return chosen, provenance


def verify_retention(
    *,
    store_root: Path,
    units: tuple[BenchmarkUnit, ...],
    coordinates: torch.Tensor,
    lengths: tuple[int, ...],
    native: dict[tuple[str, int], Path],
) -> list[dict[str, object]]:
    """Prove the persisted steered structures are STEERED, before spending the rest of the run.

    Two comparisons per draw, and both are required. Against the coordinates ``steer()`` returned,
    which establishes that what landed on disk is the structure that was generated; and against the
    native unsteered draw at the same seed, which establishes it is not a control. A store full of
    controls passes the first check on its own.
    """
    rows = {
        (row.sequence_id, row.unit_seed): row
        for row in read_steered_draws(store_root)
        if row.split == SPLIT_LABEL
    }
    verified: list[dict[str, object]] = []
    for index, unit in enumerate(units):
        row = rows.get((unit.sequence_id, unit.seed))
        if row is None:
            raise SystemExit(
                f"no steered-draw row was written for {unit.sequence_id} seed {unit.seed}: the "
                "recorder did not persist this batch, which is the exact failure this check exists "
                "for"
            )
        persisted_path = store_root / row.steered_npz
        with np.load(persisted_path, allow_pickle=False) as payload:
            persisted = torch.from_numpy(np.asarray(payload["pos"], dtype=np.float64))
        returned = coordinates[index, : lengths[index]].detach().to("cpu", torch.float64)
        if persisted.shape != returned.shape:
            raise SystemExit(
                f"{persisted_path} holds {tuple(persisted.shape)} coordinates while steer() "
                f"returned {tuple(returned.shape)} for {unit.sequence_id} seed {unit.seed}"
            )
        against_returned = float((persisted - returned).abs().max())
        native_path = native.get((unit.sequence_id, unit.seed))
        if native_path is None or not native_path.is_file():
            raise SystemExit(
                f"no native draw for {unit.sequence_id} seed {unit.seed}, so the retained steered "
                "structure cannot be checked against the control it is paired with"
            )
        with np.load(native_path, allow_pickle=False) as payload:
            control = torch.from_numpy(np.asarray(payload["pos"], dtype=np.float64))
        against_control = float((persisted - control).abs().max())
        if against_returned > RETENTION_MATCH_TOLERANCE_NM:
            raise SystemExit(
                f"{persisted_path} does not reproduce the coordinates steer() returned for "
                f"{unit.sequence_id} seed {unit.seed}: max |delta| = {against_returned:.3e} nm. "
                "The structure on disk is not the structure that was generated"
            )
        if against_control <= RETENTION_DIFFERENCE_FLOOR_NM:
            raise SystemExit(
                f"{persisted_path} is indistinguishable from the NATIVE unsteered draw for "
                f"{unit.sequence_id} seed {unit.seed}: max |delta| = {against_control:.3e} nm. "
                "Either the adapter did nothing at all or a control was retained under a steered "
                "stem; both make every number downstream a measurement of the native ensemble"
            )
        verified.append(
            {
                "sequence_id": unit.sequence_id,
                "seed": unit.seed,
                "level": unit.level,
                "steered_npz": str(persisted_path),
                "native_npz": str(native_path),
                "max_abs_delta_against_returned_nm": against_returned,
                "max_abs_delta_against_native_control_nm": against_control,
            }
        )
    return verified


def replay_controls(
    *,
    sampler: BioEmuRuntimeActivationSampler,
    record: SequenceRecord,
    seeds: list[int],
    replay_root: Path,
    native: dict[tuple[str, int], Path],
    subdir: str,
) -> list[dict[str, object]]:
    """Regenerate unsteered draws at this run's batching and compare them to the native store.

    The pairing claim rests on the prior draw being a function of ``(seed, residue count)`` alone.
    That is what ``SeededPriorSampling`` documents, and this measures it: a nonzero difference here
    means the two arms are NOT matched and the comparison is unpaired, whatever the seeds say.
    """
    sampler.generate_unsteered_frames([record] * len(seeds), seed=list(seeds))
    replayed: list[dict[str, object]] = []
    index = read_native_index(replay_root / subdir / "index.jsonl")
    for seed in seeds:
        key = (record.sequence_id, seed)
        mine = index.get(key)
        theirs = native.get(key)
        if mine is None or theirs is None or not mine.is_file() or not theirs.is_file():
            raise SystemExit(
                f"control replay for {record.sequence_id} seed {seed} is missing a side "
                f"(replayed={mine}, native={theirs})"
            )
        with np.load(mine, allow_pickle=False) as payload:
            left = np.asarray(payload["pos"], dtype=np.float64)
        with np.load(theirs, allow_pickle=False) as payload:
            right = np.asarray(payload["pos"], dtype=np.float64)
        if left.shape != right.shape:
            raise SystemExit(
                f"control replay for {record.sequence_id} seed {seed} has shape {left.shape} "
                f"against the native store's {right.shape}"
            )
        replayed.append(
            {
                "sequence_id": record.sequence_id,
                "seed": seed,
                "max_abs_delta_nm": float(np.abs(left - right).max()),
                "replayed_npz": str(mine),
                "native_npz": str(theirs),
            }
        )
    return replayed


def append_progress(path: Path, payload: dict[str, object]) -> None:
    """Publish one system's result the moment it finishes, not at the end of the run."""
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="benchmark_pool.json")
    parser.add_argument("--adapter", type=Path, required=True, help="the trained .pt checkpoint")
    parser.add_argument(
        "--train-run-report",
        type=Path,
        required=True,
        help="the training run report the intervention site is read from",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--native-index",
        type=Path,
        required=True,
        help="benchmark_native/final_structures/index.jsonl -- the matched control arm",
    )
    parser.add_argument("--cache-embeds-dir", type=Path, required=True)
    parser.add_argument("--cache-so3-dir", type=Path, required=True)
    parser.add_argument(
        "--level",
        type=float,
        required=True,
        help="the requested state: +1 holo, -1 apo. Nothing else was ever trained on",
    )
    parser.add_argument("--draws", type=int, default=100, help="B; must match the native arm")
    parser.add_argument("--draw-seed-base", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument(
        "--system",
        action="append",
        default=None,
        help="restrict to named systems; repeat. Overrides --population-from",
    )
    parser.add_argument(
        "--population-from",
        type=Path,
        default=None,
        help="a committed success_at_b JSON; keeps systems_with_ranges minus excluded_systems",
    )
    parser.add_argument("--expect-systems", type=int, default=None)
    parser.add_argument(
        "--replay-control-draws",
        type=int,
        default=0,
        help="regenerate this many unsteered draws per system and compare them to the native "
        "store, which is what makes the pairing claim a measurement",
    )
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=8,
        help="batches between torch.cuda.empty_cache(); the steered path retains a graph from the "
        "intervened score call onward",
    )
    args = parser.parse_args()

    if float(args.level) not in TRAINED_LEVELS:
        raise SystemExit(
            f"--level must be one of {list(TRAINED_LEVELS)}; received {args.level}. The cryptic "
            "adapter is conditioned on a two-state level, and no other value was presented to it "
            "during training"
        )
    if args.draws < 1 or args.batch_size < 1:
        raise SystemExit("--draws and --batch-size must be positive")

    # ------------------------------------------------------------------ population and sequences
    entries = json.loads(args.pool.read_text())["entries"]
    entries = sorted(entries, key=lambda item: str(item["sequence_id"]))
    pool_ids = [str(item["sequence_id"]) for item in entries]
    chosen, population = select_systems(
        pool_ids, args.population_from, args.system, args.expect_systems
    )
    sequences = {str(item["sequence_id"]): str(item["sequence"]) for item in entries}
    if args.shards > 1:
        if not 0 <= args.shard < args.shards:
            raise SystemExit("--shard must be in [0, --shards)")
        chosen = chosen[args.shard :: args.shards]
        print(f"shard {args.shard} of {args.shards}: {len(chosen)} systems")
    if not chosen:
        raise SystemExit("no system to generate; refusing to report an arm over an empty run")
    records = {name: SequenceRecord(sequence_id=name, sequence=sequences[name]) for name in chosen}
    require_cached_embeddings(
        args.cache_embeds_dir, {name: item.sequence for name, item in records.items()}
    )
    print(f"embedding cache:   all {len(records)} sequences covered")

    # ---------------------------------------------------------------- the control arm, and pairing
    native = read_native_index(args.native_index)
    seeds_wanted = [args.draw_seed_base + index for index in range(args.draws)]
    unpaired = sorted(
        {name for name in chosen for seed in seeds_wanted if (name, seed) not in native}
    )
    if unpaired:
        raise SystemExit(
            f"the native store does not carry every requested seed for {unpaired}. This arm is "
            "defined as the same draws with the adapter on; generating other seeds would make the "
            "comparison unpaired and put it on this project's re-randomisation noise floor"
        )
    print(f"native control arm: {len(native)} indexed draws; every requested seed is present")

    # ------------------------------------------------------------------------- adapter and site
    adapter, payload = load_adapter(args.adapter)
    attributes = adapter_attributes(payload)
    adapter_property = str(attributes["property_name"])
    if adapter_property not in CRYPTIC_PROPERTIES:
        raise SystemExit(
            f"{args.adapter} is a {adapter_property!r} adapter, not one of {CRYPTIC_PROPERTIES!r}. "
            "A level means a different thing on every property, so this refuses rather than "
            "drawing at a dose the property does not define."
        )
    if attributes.get("site_conditioning") is not None:
        raise SystemExit(
            f"{args.adapter} declares site_conditioning="
            f"{attributes['site_conditioning']!r}, so it is an arm A/B checkpoint. Its per-residue "
            "channel is measured on the deposited apo structure of a TRAINING protein and does not "
            "exist for a held-out benchmark system; generating with no channel would score it as a "
            "different experiment"
        )
    site, site_spec = load_intervention_site(args.train_run_report)
    device = torch.device(args.device)
    adapter.to(device)
    adapter.eval()
    # SHAPE-AGNOSTIC REPORTING. The single-site record names `resolved_timestep`,
    # `denoiser_iteration` and `denoiser_role`; the MULTI-SITE record names the plural
    # `resolved_timesteps`, `denoiser_iterations` and `denoiser_roles` and deliberately none of the
    # singular keys. Reading the singular ones here raised `KeyError: 'resolved_timestep'` for the
    # multisite8 arm AFTER load_intervention_site had already been taught both shapes -- MEASURED,
    # a measured run, dead in 15 s with the pool, embedding cache and native seed pairing all
    # verified immediately above. Timesteps come off `site`, which is already normalised to a
    # tuple, so this line cannot drift from the loader again.
    iterations = site_spec.get("denoiser_iterations", site_spec.get("denoiser_iteration"))
    roles = site_spec.get("denoiser_roles", site_spec.get("denoiser_role"))
    timesteps = ", ".join(f"{value:.6f}" for value in site.expected_timesteps)
    print(
        f"adapter            {args.adapter}\n"
        f"  rho_max          {attributes['rho_max']}\n"
        f"  level cond.      {attributes['level_conditioning']}\n"
        f"  site             {site.module_path} score call {site.score_calls}\n"
        f"  timesteps        {timesteps}\n"
        f"  denoiser         iteration {iterations} {roles}\n"
        f"  level            {args.level:+.1f}"
    )

    # ------------------------------------------------------------------------- stores and sampler
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = TrainingStructureStore(args.output_dir, property_name=adapter_property)
    # The sidecar names each steered draw's control by relative path. Pointing `control` at the
    # native store makes those paths RESOLVE, so `control_present` is a true statement about the
    # matched draw rather than a false one about a directory this run never wrote.
    control_link = args.output_dir / "control"
    native_store_root = args.native_index.parent.parent
    if not control_link.exists():
        control_link.symlink_to(native_store_root)
    replay_root = args.output_dir / "control_replay"
    seed_everything(args.draw_seed_base)
    trace = TraceConfig(cache_so3_dir=str(args.cache_so3_dir), save_final_structures=True)
    sampler = BioEmuRuntimeActivationSampler(
        trace,
        cache_embeds_dir=str(args.cache_embeds_dir),
        # ONLY the control replay writes through the sampler's own writer. `steer` never touches
        # it -- that is the defect this script's recorder exists for -- so nothing else lands here.
        structures_dir=str(replay_root),
        batch_size=1,
        device=args.device,
    )
    sampler.assert_runtime_assets_unchanged()
    recorder = SteeredDrawRecorder(sampler, store)

    done = {
        (row.sequence_id, row.unit_seed)
        for row in read_steered_draws(args.output_dir)
        if row.split == SPLIT_LABEL and float(row.level) == float(args.level)
    }
    if done:
        print(f"resume: {len(done)} steered draws already recorded and will be skipped")

    progress_path = args.output_dir / f"steered_progress.shard{args.shard}of{args.shards}.jsonl"
    retention: list[dict[str, object]] = []
    replays: list[dict[str, object]] = []
    per_sequence: list[dict[str, object]] = []
    started_all = time.monotonic()
    batches = 0
    for position, name in enumerate(chosen, start=1):
        record = records[name]
        started = time.monotonic()
        if args.replay_control_draws > 0:
            replays += replay_controls(
                sampler=sampler,
                record=record,
                seeds=seeds_wanted[: args.replay_control_draws],
                replay_root=replay_root,
                native=native,
                subdir=trace.final_structures_subdir,
            )
        pending = [seed for seed in seeds_wanted if (name, seed) not in done]
        generated = 0
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start : start + args.batch_size]
            members = [record] * len(chunk)
            units = tuple(
                BenchmarkUnit(
                    split=SPLIT_LABEL,
                    sequence_id=name,
                    cycle=0,
                    visit=0,
                    level=float(args.level),
                    seed=int(seed),
                    generation_seed=int(seed),
                )
                for seed in chunk
            )
            conditioning = [float(args.level)] * len(chunk)
            mask = batch_residue_mask(members, device=device)
            target = torch.tensor(conditioning, dtype=torch.float32, device=device)
            with record_steered_draws(recorder, units, conditioning):
                sample = recorder.steer(
                    members,
                    seed=list(chunk),
                    site=site,
                    adapter=adapter,
                    target=target,
                    residue_mask=mask,
                    site_field=None,
                )
            generated += len(chunk)
            batches += 1
            if not retention:
                retention = verify_retention(
                    store_root=args.output_dir,
                    units=units,
                    coordinates=sample.coordinates,
                    lengths=tuple(len(item.sequence) for item in members),
                    native=native,
                )
                for row in retention:
                    print(
                        f"  RETENTION {row['sequence_id']} seed {row['seed']}: persisted vs "
                        f"returned {row['max_abs_delta_against_returned_nm']:.3e} nm, "
                        f"vs native control "
                        f"{row['max_abs_delta_against_native_control_nm']:.3e} nm",
                        flush=True,
                    )
            del sample
            if (
                args.empty_cache_every > 0
                and batches % args.empty_cache_every == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
        sampler.assert_runtime_assets_unchanged()
        elapsed = time.monotonic() - started
        entry = {
            "sequence_id": name,
            "residue_count": len(record.sequence),
            "level": float(args.level),
            "draws_requested": args.draws,
            "draws_generated": generated,
            "draws_replayed_from_store": args.draws - generated,
            "seeds": [seeds_wanted[0], seeds_wanted[-1]],
            "seconds": elapsed,
            "seconds_per_generated_draw": elapsed / max(1, generated),
        }
        per_sequence.append(entry)
        append_progress(progress_path, entry)
        print(
            f"  [{position}/{len(chosen)}] {name:10s} L={len(record.sequence):4d} "
            f"{generated} draws  {elapsed / max(1, generated):.2f} s/draw",
            flush=True,
        )

    report = args.output_dir / f"benchmark_steered.shard{args.shard}of{args.shards}.json"
    report.write_text(
        json.dumps(
            {
                "what": (
                    "Steered BioEmu ensemble for the cryptic-pocket benchmark systems, the arm "
                    "benchmark_native is the matched control for. Structures only; success@B is "
                    "scored post hoc by scripts/cryptic_steering/score_success_at_b.py."
                ),
                "schema": "cryptic_steering.benchmark_steered.v1",
                "decision": "D-68",
                "pool": str(args.pool),
                "population": population,
                "adapter": str(args.adapter),
                "adapter_attributes": attributes,
                "train_run_report": str(args.train_run_report),
                "intervention_site": site_spec,
                "level": float(args.level),
                "draws_per_sequence": args.draws,
                "draw_seed_base": args.draw_seed_base,
                "batch_size": args.batch_size,
                "shard": args.shard,
                "shards": args.shards,
                "native_index": str(args.native_index),
                "native_store_root": str(native_store_root),
                "pairing": (
                    "every steered draw uses the seed of the native draw of the same system, so "
                    "the arms differ only by the adapter"
                ),
                "retention_check": retention,
                "control_replay": replays,
                "structure_store": store.identity(),
                "steered_index": str(
                    args.output_dir
                    / "steered/pass_00"
                    / trace.final_structures_subdir
                    / "index.jsonl"
                ),
                "sequences": per_sequence,
                "total_seconds": time.monotonic() - started_all,
                "environment": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "device_name": (
                        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
                    ),
                    "python": platform.python_version(),
                    "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                    "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                    "precision": sampler.precision.mode,
                    "denoiser_type": trace.denoiser_type,
                    "model_name": trace.model_name,
                    "runtime_assets": sampler.runtime_asset_provenance,
                },
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    print(f"\nwrote {report}")
    print(f"steered structures under {args.output_dir / 'steered/pass_00'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
