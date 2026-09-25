"""Generate a NATIVE, unsteered BioEmu ensemble for the cryptic-pocket benchmark systems.

This is the control arm of D-63's ``success@B``, and it exists because the published BioEmu number
cannot serve as one. The benchmark repository warns that v0.0.2 results are not comparable to
v1.0.0+, the public release samples with DPM/30 steps against the paper's Heun/100, and this
project samples with DPM-Solver-2 at 49 steps. A control has to be generated under the settings the
steered arm will use, or the comparison measures samplers rather than steering.

WHY THESE SEQUENCES ARE NOT ALREADY IN A CACHE. The 34 benchmark systems are held out of the
training pool by ``cryptic_steering.holdout`` on UniProt accession AND PDB entry, which is what
makes the benchmark a valid test. No ensemble drawn from the pool can contain them, so their MSAs
and Evoformer embeddings have to be built separately -- ``check_generation_inputs.py`` over
``benchmark_pool.json`` is the gate for that and should pass before this runs.

WHAT THIS WRITES, AND WHAT IT DOES NOT COMPUTE. Structures only. ``success@B`` is scored entirely
post hoc by ``cryptic_steering.success_at_b`` against the benchmark's own residue ranges, so this
script deliberately computes no metric: a generator that also scored would make the control and the
measurement one artifact, and re-scoring at a different tau or B would mean regenerating.

Persistence is the sampler's own: constructed with ``structures_dir`` and
``save_final_structures=True``, ``generate_unsteered_frames`` publishes each draw through
``FinalStructureWriter`` -- atomically, with an ``index.jsonl`` guarded by ``fcntl.flock`` so shards
may share one directory. That path became real only recently; before it,
``save_final_structures=True`` with no ``structures_dir`` silently wrote nothing, and the sampler
now refuses that combination rather than accepting it.

Seeds are recorded, not merely used. ``draw_seed_base + index`` is deterministic and the report
carries the model version, the trace configuration, the backend determinism settings and the
runtime asset provenance beside them, because an integer alone does not reproduce a trajectory.

One GPU.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from sae_bioemu.bioemu.cryptic_generation import require_cached_embeddings
from sae_bioemu.bioemu.extractor import BioEmuRuntimeActivationSampler, SequenceRecord
from sae_bioemu.core.config import TraceConfig, seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="benchmark_pool.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-embeds-dir", type=Path, required=True)
    parser.add_argument("--cache-so3-dir", type=Path, required=True)
    parser.add_argument(
        "--draws",
        type=int,
        default=100,
        help="B. ConforNets reports success@100; BioEmu's own paper uses 4000. The number is "
        "meaningless without B, so it is recorded in the report and must be stated beside any "
        "success rate.",
    )
    parser.add_argument("--draw-seed-base", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()

    entries = json.loads(args.pool.read_text())["entries"]
    entries = sorted(entries, key=lambda item: str(item["sequence_id"]))
    if args.shards > 1:
        if not 0 <= args.shard < args.shards:
            raise SystemExit("--shard must be in [0, --shards)")
        entries = entries[args.shard :: args.shards]
        print(f"shard {args.shard} of {args.shards}: {len(entries)} sequences")
    if not entries:
        raise SystemExit("no sequences to generate; refusing to report success over an empty run")

    records = {
        str(item["sequence_id"]): SequenceRecord(
            sequence_id=str(item["sequence_id"]), sequence=str(item["sequence"])
        )
        for item in entries
    }
    # Fails closed rather than falling back to a network fetch mid-generation, which would leave a
    # GPU idle on a web request and put two provenance vintages in one ensemble.
    require_cached_embeddings(
        args.cache_embeds_dir, {name: item.sequence for name, item in records.items()}
    )
    print(f"embedding cache:   all {len(records)} sequences covered")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.draw_seed_base)
    trace = TraceConfig(cache_so3_dir=str(args.cache_so3_dir), save_final_structures=True)
    sampler = BioEmuRuntimeActivationSampler(
        trace,
        cache_embeds_dir=str(args.cache_embeds_dir),
        structures_dir=str(args.output_dir),
        batch_size=1,
        device=args.device,
    )
    sampler.assert_runtime_assets_unchanged()

    started_all = time.monotonic()
    per_sequence: list[dict] = []
    for position, (name, record) in enumerate(sorted(records.items()), start=1):
        seeds = [args.draw_seed_base + index for index in range(args.draws)]
        started = time.monotonic()
        for start in range(0, len(seeds), max(1, args.batch_size)):
            chunk = list(seeds[start : start + max(1, args.batch_size)])
            members = [record] * len(chunk)
            sampler.generate_unsteered_frames(
                members if len(chunk) > 1 else record,
                seed=chunk if len(chunk) > 1 else chunk[0],
            )
        elapsed = time.monotonic() - started
        per_sequence.append(
            {
                "sequence_id": name,
                "residue_count": len(record.sequence),
                "draws": args.draws,
                "seeds": [seeds[0], seeds[-1]],
                "seconds": elapsed,
                "seconds_per_draw": elapsed / max(1, args.draws),
            }
        )
        print(
            f"  [{position}/{len(records)}] {name:10s} L={len(record.sequence):4d} "
            f"{args.draws} draws  {elapsed / max(1, args.draws):.2f} s/draw",
            flush=True,
        )

    report = args.output_dir / f"benchmark_ensemble.shard{args.shard}of{args.shards}.json"
    report.write_text(
        json.dumps(
            {
                "what": (
                    "Native unsteered BioEmu ensemble for the cryptic-pocket benchmark systems, "
                    "the control arm of D-63's success@B. Structures only; no metric is computed "
                    "here."
                ),
                "schema": "cryptic_steering.benchmark_ensemble.v1",
                "pool": str(args.pool),
                "draws_per_sequence": args.draws,
                "draw_seed_base": args.draw_seed_base,
                "shard": args.shard,
                "shards": args.shards,
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
        )
        + "\n"
    )
    print(f"\nwrote {report}")
    print(f"structures under {args.output_dir / trace.final_structures_subdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
