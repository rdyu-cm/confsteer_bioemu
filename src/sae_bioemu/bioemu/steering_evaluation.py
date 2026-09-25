"""Held-out BioEmu radius-of-gyration evaluation: the consumer of a trained adapter.

This module is deliberately separate from ``steering_workflow.py``. A training checkpoint
is a *product* of training, so it is bound to the code that produced it; evaluation code is
a *consumer* that reads a frozen adapter and reports numbers from it. Hashing both into one
fingerprint made a metric fix or a label change invalidate a ~12 GPU-h training run
(``docs/steering/DESIGN_DECISIONS.md`` D14).

Keeping the two in one file made that impossible to express honestly: a whole-file hash
would have covered both concerns, and a hand-maintained list of "evaluation regions" inside
one file rots the first time a function moves. The module boundary makes the segmentation
structural -- ``steering_workflow.py`` never imports this module's internals, this module
never imports ``steering_workflow``, and their only shared code is
``steering_artifacts.py``, which is classified as generation-shared.

**What this module is NOT free to change.** Evaluation regenerates structures with the same
sampler, hook, adapter, property surrogate, and target formula that training used. Those
sources are generation-shared: they live in the *training* fingerprint and are still checked
here, through ``validate_ordinary_evaluation_identity``. Only scoring, reporting, caching,
and structure persistence are evaluation-only.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from importlib import resources
from pathlib import Path
from typing import Any

import torch

from sae_bioemu.bioemu.extractor import BioEmuRuntimeActivationSampler
from sae_bioemu.bioemu.steering import SteeringSite
from sae_bioemu.bioemu.steering_artifacts import (
    atomic_json,
    canonical_fingerprint,
    positive_integer,
    sha256_file,
    uint32,
)
from sae_bioemu.bioemu.steering_properties import batch_residue_mask
from sae_bioemu.steering.adapter import BoundedResidualAdapter
from sae_bioemu.steering.batching import (
    UNBATCHED_CONFIG,
    GenerationBatchConfig,
    SteeringBatchPlan,
    build_evaluation_batch_plan,
)
from sae_bioemu.steering.evaluation import (
    ABSOLUTE_TOLERANCE_UNITS,
    SIGMA_TOLERANCE_UNITS,
    SteeringEvaluationRow,
    SteeringFailure,
    evaluate_paired_steering,
    write_steering_report,
)
from sae_bioemu.steering.evaluation_structures import (
    ADAPTER_OFF_ARM,
    ADAPTER_ON_ARM,
    SteeringStructureWriter,
)
from sae_bioemu.steering.ordinary_workflow import (
    OrdinarySteeringConfig,
    validate_ordinary_completion_checkpoint,
    validate_ordinary_evaluation_identity,
)
from sae_bioemu.steering.progress import ProgressReporter
from sae_bioemu.steering.schedule import continuous_target, relative_sequence_target
from sae_bioemu.steering.surrogate import differentiable_radius_of_gyration
from sae_bioemu.steering.training import SteeringLoss, load_adapter_checkpoint

#: The shipped 30%-identity cluster map for the radius-of-gyration pool, written by
#: ``scripts/steering/build_rg_cluster_map.py``. It is an asset rather than a configuration key
#: because the pool is a fixed set of UniRef50 identifiers and the map is a property of those
#: sequences, not of a run: two arms over the same proteins must not be able to declare different
#: independent units.
RG_FAMILY_CLUSTER_ASSET = "rg_family_clusters.json"


def load_rg_family_clusters() -> dict[str, str]:
    """Return the shipped ``sequence_id -> 30%-identity cluster`` map for the Rg pool.

    D-24 fixed the independent statistical unit as the merged 30%-identity cluster, not the
    sequence. ``docs/cryptic_steering/stage0_split_record.json`` cannot supply it here -- its keys
    are PDB chains and this pool is UniRef50, with zero overlap -- so the Rg pool has its own map,
    built by the same MMseqs2 procedure and the same
    :func:`sae_bioemu.cryptic_steering.split.merge_clusters_by_identity`.

    Raises rather than returning ``{}`` when the asset is malformed: an empty map read as "no
    clusters" would silently restore the sequence-grouped interval this exists to end.
    """
    source = resources.files("sae_bioemu.steering.assets").joinpath(RG_FAMILY_CLUSTER_ASSET)
    payload = json.loads(source.read_text())
    clusters = payload.get("clusters") if isinstance(payload, dict) else None
    if not isinstance(clusters, dict) or not clusters:
        raise ValueError(f"{RG_FAMILY_CLUSTER_ASSET} carries no 'clusters' mapping")
    for sequence_id, cluster in clusters.items():
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError(f"{RG_FAMILY_CLUSTER_ASSET}: sequence IDs must be non-empty strings")
        if not isinstance(cluster, str) or not cluster:
            raise ValueError(f"{RG_FAMILY_CLUSTER_ASSET}: cluster labels must be non-empty strings")
    return dict(clusters)


def resolve_rg_family_clusters(
    sequence_ids: Iterable[str],
) -> tuple[dict[str, str] | None, dict[str, object]]:
    """Return the cluster map restricted to ``sequence_ids``, with a coverage record.

    All-or-nothing on purpose. ``grouped_steering_metrics`` requires a map over exactly the
    observed sequences, and filling the gaps by treating an unlabelled sequence as its own
    cluster is precisely the too-narrow interval D-24 forbids -- for exactly the sequences the
    map does not cover, and invisibly. Partial coverage therefore yields ``None``, the report
    publishes ``family_clusters: null``, and ``scripts/steering/score_dose_slope.py`` refuses to
    quote a cluster-grouped interval until someone supplies a map or opts out in writing.

    The second element is the record: it names the covered and missing counts and the first few
    missing identifiers, and it travels into the run report so a sequence-grouped run says so.
    """
    observed = sorted(set(sequence_ids))
    try:
        clusters = load_rg_family_clusters()
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as error:
        return None, {
            "available": False,
            "reason": f"{type(error).__name__}: {error}",
            "covered": 0,
            "missing": len(observed),
        }
    missing = [sequence_id for sequence_id in observed if sequence_id not in clusters]
    record: dict[str, object] = {
        "available": True,
        "asset": RG_FAMILY_CLUSTER_ASSET,
        "covered": len(observed) - len(missing),
        "missing": len(missing),
        "missing_examples": missing[:5],
    }
    if missing:
        record["reason"] = (
            f"{len(missing)} of {len(observed)} evaluated sequences have no cluster label; "
            "rebuild the map with scripts/steering/build_rg_cluster_map.py over this pool's FASTA"
        )
        return None, record
    restricted = {sequence_id: clusters[sequence_id] for sequence_id in observed}
    record["clusters"] = len(set(restricted.values()))
    return restricted, record


def evaluate_radius_of_gyration(
    *,
    progress: ProgressReporter | None = None,
    checkpoint_path: Path,
    output_dir: Path,
    adapter: BoundedResidualAdapter,
    sampler: BioEmuRuntimeActivationSampler,
    records: dict[str, Any],
    site: SteeringSite,
    schedule: Any,
    config: OrdinarySteeringConfig,
    objective: SteeringLoss,
    runtime: dict[str, object],
    manifest_fingerprint: str,
    training_code_fingerprint: str,
    evaluation_code_fingerprint: str,
    workflow_config_fingerprint: str,
    artifact_prefix: str,
    provenance: str,
    intervention_site: dict[str, object],
    batch_plan: SteeringBatchPlan | None = None,
    evaluation_batch_config: GenerationBatchConfig | None = None,
) -> dict[str, object]:
    """Score a frozen training checkpoint on the held-out split.

    ``training_code_fingerprint`` is the gate: it covers the sampler, the hook, the adapter,
    the property surrogate, the target formula, the runtime assets, and the environment, and
    it is what ``config.generation_callback_identity`` carries into the checkpoint identity.
    A checkpoint trained by different generation code is rejected below rather than scored
    against structures it never could have produced.

    ``evaluation_code_fingerprint`` is *recorded*, never matched against the checkpoint. It
    keys this run's per-unit cache and structure store and is published in the report
    manifest, so a reader can always tell which evaluation code produced a number and two
    differently evaluated runs stay distinguishable -- and so a cached row computed by an
    older metric definition is never silently mixed with a fresh one.
    """
    if not checkpoint_path.is_file():
        raise ValueError("BioEmu evaluation requires a complete training checkpoint")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or not isinstance(raw.get("adapter_config"), dict):
        raise ValueError("BioEmu adapter checkpoint has an invalid schema")
    checkpoint = load_adapter_checkpoint(
        checkpoint_path,
        expected_manifest_fingerprint=manifest_fingerprint,
        expected_adapter_config=raw["adapter_config"],
        expected_property_callback_identity=config.property_callback_identity,
        expected_source_checkpoints=config.source_checkpoints,
    )
    training_journal = validate_ordinary_completion_checkpoint(
        checkpoint, checkpoint_path, expected_schedule=schedule
    )
    # A checkpoint that ran to a terminal reason but failed its residual gate is admitted for
    # evaluation, because the held-out measurement is exactly what says whether such an
    # adapter does anything. It must never read as a clean pass, so the failure travels into
    # the report rather than staying buried in the training journal.
    training_gate_failure = training_journal.get("validation_gate_failure")
    validate_ordinary_evaluation_identity(
        checkpoint,
        schedule=schedule,
        adapter=adapter,
        objective=objective,
        config=config,
        batch_plan=batch_plan,
    )
    adapter.load_state_dict(checkpoint.adapter_state)
    adapter.eval()
    checkpoint_fingerprint = sha256_file(checkpoint_path)
    # D8: a held-out batch holds one sequence's own units, which differ only in requested
    # level and seed, so padding is a no-op and nothing is reordered. The plan is bound
    # into the evaluation identity below because batching is not bit-identical, so a run
    # with different batches must not resume into another run's per-unit cache.
    #
    # D15: this config is the caller's own ``evaluation_generation_batch``, NOT the
    # training plan's. Held-out batching is sequence-grouped and pays 3.27x; train batching
    # pays padding and measured a net loss. Deriving one from the other meant disabling the
    # loss also disabled the win. ``None`` means one unit per generation.
    evaluation_batching = build_evaluation_batch_plan(
        schedule.test,
        {sequence_id: len(record.sequence) for sequence_id, record in records.items()},
        UNBATCHED_CONFIG if evaluation_batch_config is None else evaluation_batch_config,
    )
    evaluation_identity = canonical_fingerprint(
        {
            "manifest": manifest_fingerprint,
            "config": workflow_config_fingerprint,
            "checkpoint": checkpoint_fingerprint,
            # D14: both code fingerprints key the cache. The training one because it
            # determines the structures; the evaluation one because a cached unit stores a
            # scored row, and a row computed by an older metric definition must never be
            # replayed beside a freshly computed one.
            "training_code": training_code_fingerprint,
            "evaluation_code": evaluation_code_fingerprint,
            "units": [asdict(unit) for unit in schedule.test],
            "batching": evaluation_batching.identity(),
        }
    )
    rows: list[SteeringEvaluationRow] = []
    failures: list[SteeringFailure] = []
    attempted: list[dict[str, object]] = []
    tolerance = float(
        runtime["success_tolerance_sigma"] if config.is_relative else runtime["success_tolerance"]
    )
    tolerance_units = SIGMA_TOLERANCE_UNITS if config.is_relative else ABSOLUTE_TOLERANCE_UNITS
    device = next(adapter.parameters()).device
    unit_root = output_dir / f"{artifact_prefix}_evaluation_units" / evaluation_identity
    unit_root.mkdir(parents=True, exist_ok=True)
    structures = SteeringStructureWriter(
        output_dir / f"{artifact_prefix}_evaluation_structures" / evaluation_identity
    )
    outcomes: list[dict[str, object]] = []
    cursor = 0
    for size in evaluation_batching.batch_sizes:
        members = tuple(schedule.test[cursor : cursor + size])
        identities = tuple(
            {
                "index": cursor + offset,
                "unit": asdict(unit),
                "evaluation": evaluation_identity,
            }
            for offset, unit in enumerate(members)
        )
        paths = tuple(unit_root / f"{cursor + offset:08d}.json" for offset in range(size))
        # A cached unit is complete on its own: its JSON is signed over its own identity
        # and carries both arms of its own seed, so loading it beside a freshly generated
        # sibling cannot break any matched pairing. Only the uncached members are rerun,
        # and they are rerun together as one batch.
        resolved: dict[int, dict[str, object]] = {
            offset: _load_evaluation_unit(paths[offset], identities[offset])
            for offset in range(size)
            if paths[offset].is_file()
        }
        pending = tuple(offset for offset in range(size) if offset not in resolved)
        if pending:
            fresh = _run_evaluation_batch(
                sampler=sampler,
                record=records[members[0].sequence_id],
                units=tuple(members[offset] for offset in pending),
                identities=tuple(identities[offset] for offset in pending),
                site=site,
                adapter=adapter,
                config=config,
                tolerance=tolerance,
                device=device,
                structures=structures,
            )
            # The batch commits per member, not as a group. A unit is the atomic
            # scientific quantity -- its JSON is self-validating and describes only its
            # own seed's off/on pair -- while a batch is purely a compute grouping, and
            # group atomicity across N separate files would need a batch marker file that
            # the per-unit cache and a future --shard flag would then have to reconcile.
            # The whole batch is generated before anything is written, so a crash can
            # only fall between two complete units and never inside one.
            for offset, outcome in zip(pending, fresh, strict=True):
                atomic_json(paths[offset], outcome)
                resolved[offset] = outcome
        for offset in range(size):
            outcome = resolved[offset]
            outcomes.append(outcome)
            if progress is not None:
                progress.observe(
                    phase="evaluation",
                    elapsed_seconds=_unit_elapsed_seconds(outcome),
                    completed_units=cursor + offset + 1,
                    epoch=0,
                    force=cursor + offset + 1 == len(schedule.test),
                )
        cursor += size
    for outcome in outcomes:
        attempted.append(dict(outcome["attempted"]))
        if outcome["status"] == "success":
            rows.append(SteeringEvaluationRow(**outcome["row"]))
        else:
            failures.append(SteeringFailure(**outcome["failure"]))
    report_dir = output_dir / f"{artifact_prefix}_evaluation"
    expected_inventory = [(item["sequence_id"], item["seed"], item["target"]) for item in attempted]
    # D-24: the independent statistical unit is the 30%-identity cluster, not the sequence. This
    # evaluator published ``family_clusters: null`` on every journal it has ever written, so its
    # intervals -- and every dose slope computed from them -- were sequence-grouped, which is too
    # narrow wherever the pool contains homologues. ``None`` is still allowed and still honest:
    # the report then says it has no cluster labels and the downstream scorer refuses to quote a
    # cluster-grouped interval, rather than quietly quoting a sequence-grouped one as if it were.
    family_clusters, cluster_coverage = resolve_rg_family_clusters(
        item["sequence_id"] for item in attempted
    )
    if family_clusters is None and progress is not None:
        progress.announce(
            "evaluation: no 30%-identity cluster map covers this pool "
            f"({cluster_coverage.get('reason', 'unavailable')}); metrics are sequence-grouped "
            "and the report records family_clusters: null (D-24)"
        )
    # ``runtime`` keeps its historical key name and now carries the training code
    # fingerprint, which is exactly the runtime identity: assets, environment, and the code
    # that generates structures. ``evaluation_code`` is the added, non-gating record.
    canonical = {
        "manifest": manifest_fingerprint,
        "config": workflow_config_fingerprint,
        "checkpoint": checkpoint_fingerprint,
        "runtime": training_code_fingerprint,
        "evaluation_code": evaluation_code_fingerprint,
    }
    if report_dir.exists():
        _validate_complete_report(report_dir, canonical)
    else:
        write_steering_report(
            report_dir,
            rows,
            tuple(failures),
            {
                "fingerprint": manifest_fingerprint,
                "source_fingerprints": list(config.source_checkpoints.values()),
                "expected_inventory": expected_inventory,
            },
            bootstrap_replicates=positive_integer(
                runtime["bootstrap_replicates"], "bootstrap_replicates"
            ),
            bootstrap_seed=uint32(runtime["bootstrap_seed"], "bootstrap_seed"),
            success_tolerance=tolerance,
            success_tolerance_units=tolerance_units,
            family_clusters=family_clusters,
            attempted_units=attempted,
            canonical_fingerprints=canonical,
        )
    return {
        "backend": "bioemu",
        "property_name": "radius_of_gyration",
        "complete": True,
        # Non-null means the trained adapter violated its own residual bound. Held-out
        # numbers below are still real, but they describe an intervention that was not
        # bounded as designed, and must be reported that way.
        "training_validation_gate_failure": training_gate_failure,
        "evaluation_path": str(report_dir),
        "evaluated_units": len(rows),
        "failures": len(failures),
        # The unit the intervals were resampled on, stated in the run report rather than left to
        # be inferred from report.json. A sequence-grouped run must never look like a
        # cluster-grouped one (D-24).
        "bootstrap_unit": "sequence" if family_clusters is None else "family_cluster",
        "family_cluster_coverage": cluster_coverage,
        "generation_batching": evaluation_batching.identity(),
        "structures_path": str(structures.root),
        "structures_indexed": structures.entry_count,
        "property_loss": objective.property_loss.value,
        "residual_penalty": objective.residual_penalty.value,
        "residual_penalty_units": objective.residual_penalty_units,
        "residual_penalty_rho_max": objective.rho_max,
        "provenance": provenance,
        "intervention_site": intervention_site,
        "training_code_fingerprint": training_code_fingerprint,
        "evaluation_code_fingerprint": evaluation_code_fingerprint,
    }


def _run_evaluation_batch(
    *,
    sampler: BioEmuRuntimeActivationSampler,
    record: Any,
    units: tuple[Any, ...],
    site: SteeringSite,
    adapter: BoundedResidualAdapter,
    config: OrdinarySteeringConfig,
    tolerance: float,
    device: torch.device,
    identities: tuple[dict[str, object], ...],
    structures: SteeringStructureWriter | None = None,
) -> tuple[dict[str, object], ...]:
    """Generate one sequence's held-out units as two batched arms and score each unit.

    Every member is the same sequence at its own requested level and its own seed, so
    the two batches are BioEmu's native replicate batching and the residue mask is all
    valid. The arms cannot share a batch -- one is steered and one is not -- but member
    ``i`` of both arms is seeded with ``units[i].seed``, in the same order, so ``on -
    off`` stays the matched comparison the whole paired design rests on.
    """
    started = time.monotonic()
    members = tuple(units)
    size = len(members)
    if size < 1 or len(identities) != size:
        raise ValueError("an evaluation batch needs at least one unit and one identity per unit")
    if len({unit.sequence_id for unit in members}) != 1:
        raise ValueError("an evaluation batch must hold exactly one sequence's units")
    if any(unit.sequence_id != record.sequence_id for unit in members):
        raise ValueError("evaluation batch members must match the supplied sequence record")
    seeds = [unit.seed for unit in members]
    batch_records = [record] * size
    mask = batch_residue_mask(batch_records, device=device)
    if config.is_relative:
        # D3: the target depends only on the sequence's reference statistics, so it is
        # known before any structure exists. The matched adapter-off arm below is still
        # generated: it is the null control and the basis of every paired metric.
        statistics = config.property_reference.statistics(members[0].sequence_id)
        targets = [
            relative_sequence_target(
                statistics.mean,
                statistics.sigma,
                unit.level,
                dose_multiplier=config.dose_multiplier,
            )
            for unit in members
        ]
        conditioning = [float(unit.level) for unit in members]
        property_scale = statistics.sigma
    else:
        targets = [
            min(
                config.target_range.upper,
                max(
                    config.target_range.lower,
                    config.target_mean + unit.level * config.maximum_change,
                ),
            )
            for unit in members
        ]
        conditioning = [0.0] * size
        property_scale = 1.0
    phase = "adapter_off_generation"
    try:
        sampler.assert_runtime_assets_unchanged()
        off_nanometers = sampler.generate_unsteered(batch_records, seed=seeds).to(device)
        off_values = differentiable_radius_of_gyration(off_nanometers * 10.0, mask)
        baselines = [float(off_values[index].item()) for index in range(size)]
        if not config.is_relative:
            targets = [
                continuous_target(
                    baselines[index], config.target_range, config.maximum_change, unit.level
                )
                for index, unit in enumerate(members)
            ]
            conditioning = [
                (target - config.target_mean) / config.target_scale for target in targets
            ]
        phase = "adapter_on_generation"
        steered = sampler.steer(
            batch_records,
            seed=seeds,
            site=site,
            adapter=adapter,
            target=torch.tensor(conditioning, dtype=torch.float32, device=device),
            residue_mask=mask,
        )
        sampler.assert_runtime_assets_unchanged()
        on_tensor = differentiable_radius_of_gyration(steered.coordinates * 10.0, mask)
        on_values = [float(on_tensor[index].item()) for index in range(size)]
        residual_fractions = [
            float(steered.residual_fraction[index].item()) for index in range(size)
        ]
        if structures is not None:
            # Written before any unit JSON is committed, so an interrupted batch can only
            # leave structures whose unit will be regenerated, never a committed unit
            # whose structures are missing.
            _persist_evaluation_structures(
                structures,
                record=record,
                units=members,
                identities=identities,
                targets=targets,
                property_name=adapter.property_name,
                off_nanometers=off_nanometers,
                on_nanometers=steered.coordinates,
                off_values=baselines,
                on_values=on_values,
            )
        outcomes: list[dict[str, object]] = []
        for index, unit in enumerate(members):
            pair_rows, pair_failures = evaluate_paired_steering(
                (
                    {
                        "sequence_id": unit.sequence_id,
                        "seed": unit.seed,
                        "target": targets[index],
                        "structure": baselines[index],
                        "residual_fraction": 0.0,
                    },
                ),
                (
                    {
                        "sequence_id": unit.sequence_id,
                        "seed": unit.seed,
                        "target": targets[index],
                        "structure": on_values[index],
                        "residual_fraction": residual_fractions[index],
                    },
                ),
                float,
                success_tolerance=tolerance,
                property_scale=property_scale,
            )
            if pair_failures or len(pair_rows) != 1:
                raise RuntimeError("paired evaluation did not produce exactly one row")
            outcomes.append(
                {
                    "schema_version": 1,
                    "identity": identities[index],
                    "status": "success",
                    "attempted": {
                        "sequence_id": unit.sequence_id,
                        "seed": unit.seed,
                        "target": targets[index],
                    },
                    "row": asdict(pair_rows[0]),
                }
            )
    except Exception as error:
        # A batched arm succeeds or fails as a whole, so every member of a failed batch
        # is retained as its own attempted unit with its own key. Nothing is dropped.
        outcomes = [
            {
                "schema_version": 1,
                "identity": identities[index],
                "status": "failure",
                "attempted": {
                    "sequence_id": unit.sequence_id,
                    "seed": unit.seed,
                    "target": targets[index],
                },
                "failure": asdict(
                    SteeringFailure(
                        unit.sequence_id,
                        unit.seed,
                        targets[index],
                        phase,
                        type(error).__name__,
                        str(error) or repr(error),
                    )
                ),
            }
            for index, unit in enumerate(members)
        ]
    elapsed = max(time.monotonic() - started, 0.0)
    validated: list[dict[str, object]] = []
    for index, outcome in enumerate(outcomes):
        outcome["content_fingerprint"] = canonical_fingerprint(outcome)
        # Attached after signing: wall clock must not enter the cached-response
        # fingerprint, and neither may the batch composition, which would otherwise stop
        # a completed unit from replaying under a different resume grouping.
        outcome["timing"] = {
            "elapsed_seconds": elapsed / size,
            "batch_size": size,
            "batch_seconds": elapsed,
        }
        validated.append(_validate_evaluation_unit(outcome, identities[index]))
    return tuple(validated)


def _persist_evaluation_structures(
    structures: SteeringStructureWriter,
    *,
    record: Any,
    units: tuple[Any, ...],
    identities: tuple[dict[str, object], ...],
    targets: list[float],
    property_name: str,
    off_nanometers: torch.Tensor,
    on_nanometers: torch.Tensor,
    off_values: list[float],
    on_values: list[float],
) -> None:
    """Persist both arms of every member, in nanometres, keyed by the unit's own seed."""
    residues = len(record.sequence)
    off_cpu = off_nanometers.detach().to("cpu")
    on_cpu = on_nanometers.detach().to("cpu")
    for index, unit in enumerate(units):
        for arm, coordinates, value in (
            (ADAPTER_OFF_ARM, off_cpu, off_values[index]),
            (ADAPTER_ON_ARM, on_cpu, on_values[index]),
        ):
            structures.write(
                sequence_id=unit.sequence_id,
                sequence=record.sequence,
                seed=unit.seed,
                arm=arm,
                unit_index=int(identities[index]["index"]),
                visit=int(unit.visit),
                level=float(unit.level),
                target=float(targets[index]),
                property_name=property_name,
                property_value=value,
                positions_nanometers=coordinates[index, :residues],
            )


def _unit_elapsed_seconds(outcome: Mapping[str, object]) -> float:
    """Return a cached unit's recorded duration, or zero when it was not recorded."""
    timing = outcome.get("timing")
    if not isinstance(timing, Mapping):
        return 0.0
    value = timing.get("elapsed_seconds")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return max(float(value), 0.0) if math.isfinite(float(value)) else 0.0


def _load_evaluation_unit(path: Path, identity: dict[str, object]) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load BioEmu evaluation unit: {path}") from error
    return _validate_evaluation_unit(value, identity)


def _validate_evaluation_unit(value: object, identity: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict) or value.get("status") not in {"success", "failure"}:
        raise ValueError("BioEmu evaluation unit identity is invalid")
    outcome_key = "row" if value["status"] == "success" else "failure"
    expected_keys = {
        "schema_version",
        "identity",
        "status",
        "attempted",
        outcome_key,
        "content_fingerprint",
    }
    # ``timing`` is optional diagnostic wall clock: it is excluded from the signed content
    # and never required, so a cached unit written without it still replays.
    unsigned_keys = {"content_fingerprint", "timing"}
    if (
        set(value) - {"timing"} != expected_keys
        or value.get("schema_version") != 1
        or value.get("identity") != identity
    ):
        raise ValueError("BioEmu evaluation unit identity is invalid")
    fingerprint = value["content_fingerprint"]
    unsigned = {name: item for name, item in value.items() if name not in unsigned_keys}
    if not isinstance(fingerprint, str) or fingerprint != canonical_fingerprint(unsigned):
        raise ValueError("BioEmu evaluation unit content fingerprint is invalid")
    attempted = value.get("attempted")
    scheduled = identity.get("unit")
    if (
        not isinstance(attempted, dict)
        or set(attempted) != {"sequence_id", "seed", "target"}
        or not isinstance(scheduled, dict)
        or attempted["sequence_id"] != scheduled.get("sequence_id")
        or attempted["seed"] != scheduled.get("seed")
    ):
        raise ValueError("BioEmu evaluation unit does not match its scheduled key")
    outcome = value[outcome_key]
    if not isinstance(outcome, dict):
        raise ValueError("BioEmu evaluation unit outcome must be an object")
    validated = (
        SteeringEvaluationRow(**outcome) if outcome_key == "row" else SteeringFailure(**outcome)
    )
    if validated.key != (attempted["sequence_id"], attempted["seed"], float(attempted["target"])):
        raise ValueError("BioEmu evaluation outcome does not match its attempted key")
    return value


def _validate_complete_report(path: Path, canonical: dict[str, str]) -> None:
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        complete = json.loads((path / "complete.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("existing BioEmu evaluation report is incomplete") from error
    unsigned_manifest = (
        {name: value for name, value in manifest.items() if name != "fingerprint"}
        if isinstance(manifest, dict)
        else {}
    )
    expected_artifacts = {
        "predictions.csv",
        "predictions.json",
        "failures.csv",
        "failures.json",
        "metrics.json",
        "intervals.json",
        "report.json",
    }
    artifact_hashes = manifest.get("artifact_hashes") if isinstance(manifest, dict) else None
    valid_artifacts = (
        isinstance(artifact_hashes, dict) and set(artifact_hashes) == expected_artifacts
    )
    if valid_artifacts:
        valid_artifacts = all(
            isinstance(digest, str)
            and (path / name).is_file()
            and digest == sha256_file(path / name)
            for name, digest in artifact_hashes.items()
        )
    if (
        not isinstance(manifest, dict)
        or manifest.get("fingerprint") != canonical_fingerprint(unsigned_manifest)
        or manifest.get("canonical_fingerprints") != canonical
        or not isinstance(complete, dict)
        or set(complete)
        != {"schema_version", "complete", "manifest_fingerprint", "manifest_sha256"}
        or complete.get("schema_version") != 1
        or complete.get("complete") is not True
        or complete.get("manifest_fingerprint") != manifest.get("fingerprint")
        or complete.get("manifest_sha256") != sha256_file(path / "manifest.json")
        or not valid_artifacts
    ):
        raise ValueError("existing BioEmu evaluation report identity is invalid")
