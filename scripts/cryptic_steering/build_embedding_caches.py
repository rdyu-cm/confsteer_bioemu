"""Precompute BioEmu Evoformer embeddings for the cryptic-pocket pool.

Every GPU stage needs these: the runtime fails closed on a cache miss rather than computing
embeddings mid-run, so an uncached sequence stops a training or evaluation job dead. Building them
up front converts that hazard into one bounded, restartable job.

This step can reach the internet from a compute node -- the MSA comes from the ColabFold MMseqs2
server -- so it is written to tolerate per-sequence failure and resume, rather than aborting a
500-sequence batch on one timeout. It does not have to: given ``--msa-cache-dir``, MSAs already
fetched by ``prefetch_msas.py`` are passed as ``msa_file`` and the network is never touched, which
is the whole reason that script exists. Run the GPU phase with ``--require-cached-msa`` so a
missing MSA is a loud failure rather than a GPU quietly idling on a web request.

Caching is keyed by a hash of the sequence itself, so it is independent of which pool filters are
finally applied: building for the whole pre-threshold pool is a superset of any later choice and
never has to be redone.

Why this script verifies what it is about to skip
--------------------------------------------------

``get_colabfold_embeds`` decides a sequence is cached from ``os.path.exists`` on two paths, and it
writes those two paths with two non-atomic ``np.save`` calls in sequence. A kill during the SECOND
save leaves a complete ``_single.npy`` beside a truncated ``_pair.npy`` with both paths present, so
the sequence is skipped by every later run and every downstream number for it is computed from a
short array. A kill during the first save is harmless -- the ``and`` forces the recompute -- so the
hazard is exactly the second-save window, and it is invisible to a path check. This job is the one
that gets SIGKILLed at its walltime, so it is the job that both creates the hazard and is in the
best position to detect it.

So each sequence is checked with :mod:`sae_bioemu.cryptic_steering.embedding_cache` twice: before
it is skipped, and again after a fresh computation, which catches a truncation this run created
rather than publishing it. Repair is opt-in and never deletes: ``--on-corrupt quarantine`` renames
the bad files aside, which both preserves the evidence of how the writer was interrupted and makes
the existence test miss so the sequence is recomputed. The default, ``report``, changes nothing on
disk and exits non-zero, because silently recomputing over a corrupt entry would destroy the only
record of the failure mode.

Why the write itself is staged, which is the actual fix
-------------------------------------------------------

Detection is second best. ``get_colabfold_embeds`` derives both filenames from the sequence hash
INSIDE whatever directory it is handed, so handing it a private per-sequence staging directory and
then ``os.replace``-ing the finished files into the real cache moves every partial write out of the
cache entirely. ``os.replace`` is atomic within a filesystem on POSIX, so a kill at any instant
leaves the real cache either untouched or complete.

Two details that make that true rather than nearly true:

* The staging directory must be on the SAME filesystem as the cache, or ``os.replace`` degrades to
  a copy and the atomicity is gone. It therefore defaults to a sibling of the cache directory on
  shared scratch, never ``/tmp`` or node-local scratch.
* The three files are published in the order FASTA, single, pair -- the reverse of how much the
  reader depends on them. ``get_colabfold_embeds``'s hit test is ``exists(single) and
  exists(pair)``, so publishing the pair array LAST means the final rename is the commit: before
  it, the test misses and the sequence is recomputed; after it, the entry is whole. There is no
  instant at which a reader sees a complete single beside an incomplete pair.

The upstream package is left alone. ``site-packages`` is shared across checkouts and is not ours
to edit; this achieves the same guarantee from the caller's side.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from sae_bioemu.cryptic_steering.embedding_cache import (
    EMBEDDING_FASTA_SUFFIX,
    EMBEDDING_PAIR_SUFFIX,
    EMBEDDING_SINGLE_SUFFIX,
    check_sequence,
    quarantine_entry,
    sequence_digest,
)
from sae_bioemu.cryptic_steering.msa_cache import check_a3m

#: Publication order. The pair array is last because it is the second half of the caller's hit
#: test, so its rename is what commits the entry.
_PUBLISH_ORDER = (EMBEDDING_FASTA_SUFFIX, EMBEDDING_SINGLE_SUFFIX, EMBEDDING_PAIR_SUFFIX)


class WrittenEmbeddingNotIntactError(RuntimeError):
    """A freshly computed embedding did not pass the integrity check.

    Distinguished from a network or MSA failure because the two need different responses: an
    MMseqs2 timeout is expected and retried, while a short array on disk means this job was
    interrupted mid-write and the run must not exit green.
    """


def compute_embedding_atomically(
    compute: Callable[..., object],
    sequence: str,
    cache_dir: Path,
    staging_dir: Path,
    msa_file: Path | None = None,
) -> None:
    """Compute one sequence's embeddings in a private directory and publish them atomically.

    ``compute`` is ``bioemu.get_embeds.get_colabfold_embeds``, taken as an argument so this can be
    exercised without the CUDA/JAX stack or the ColabFold server.

    ``msa_file`` is passed straight through. When it is given, ``_get_a3m_string`` reads it and the
    ColabFold MMseqs2 server is never contacted, which is what lets this phase run back to back on
    a GPU instead of blocking on a remote queue.

    The staging directory is emptied on both success and failure: a partial computation left behind
    would be recomputed anyway, and leaving it would grow an unbounded pile of half-written pair
    arrays on Lustre.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    digest = sequence_digest(sequence)
    workspace = Path(tempfile.mkdtemp(prefix=f"{digest[:12]}-", dir=str(staging_dir)))
    try:
        kwargs = {"msa_file": str(msa_file)} if msa_file is not None else {}
        compute(seq=sequence, cache_embeds_dir=str(workspace), **kwargs)
        staged = check_sequence(workspace, sequence)
        if not staged.arrays_intact:
            raise WrittenEmbeddingNotIntactError(
                f"embedding computed for {digest[:12]} is not intact in staging: "
                + "; ".join(staged.problems)
            )
        for suffix in _PUBLISH_ORDER:
            source = workspace / f"{digest}{suffix}"
            if source.exists():
                os.replace(source, cache_dir / f"{digest}{suffix}")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="pool JSON from build_pool.py")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="Index of this shard. The pool is split by position after sorting, so shards are "
        "disjoint and their union is the whole pool.",
    )
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--msa-cache-dir",
        type=Path,
        default=None,
        help="a3m cache from prefetch_msas.py. When a sequence has a cached MSA it is passed as "
        "msa_file, so this job runs the Evoformer pass alone and never contacts the ColabFold "
        "server -- which is the point of the split. Sequences without one fall back to fetching, "
        "unless --require-cached-msa.",
    )
    parser.add_argument(
        "--require-cached-msa",
        action="store_true",
        help="Fail closed on a sequence with no cached MSA instead of fetching it. Use this for "
        "the GPU phase, where an unnoticed fallback means a GPU idling on a web request.",
    )
    parser.add_argument(
        "--msa-cache-only",
        action="store_true",
        help="Restrict the work list to sequences that already have a usable cached MSA. This is "
        "the production shape of the GPU phase: it consumes exactly what the CPU prefetch has "
        "delivered so far, so the two phases can run concurrently and this job can be resubmitted "
        "as the a3m cache fills, without reporting every not-yet-fetched sequence as a failure.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Directory for per-sequence staging before the atomic publish. MUST be on the same "
        "filesystem as --cache-dir, or os.replace degrades to a copy and stops being atomic; "
        "defaults to a sibling of the cache directory, which guarantees that.",
    )
    parser.add_argument(
        "--on-corrupt",
        choices=("report", "recompute", "quarantine"),
        default="report",
        help="What to do with an existing entry that is present but incomplete. 'report' leaves "
        "it alone, records it and exits non-zero. 'recompute' overwrites it via the same atomic "
        "publish, which needs no deletion and cannot leave a half-repaired entry. 'quarantine' "
        "renames it aside first -- still never deletes -- when the corrupt bytes are wanted as "
        "evidence of how the writer was interrupted.",
    )
    args = parser.parse_args()

    # Imported here so --help works without the CUDA/JAX stack present.
    from bioemu.get_embeds import get_colabfold_embeds

    entries = json.loads(args.pool.read_text())["entries"]
    entries = sorted(entries, key=lambda record: record["sequence_id"])
    if args.msa_cache_only:
        if args.msa_cache_dir is None:
            raise SystemExit("--msa-cache-only needs --msa-cache-dir")
        # Filtered BEFORE --limit and --shard so both apply to the work actually available, and a
        # shard is a slice of what can be done now rather than of the whole pool.
        available = [
            entry for entry in entries if check_a3m(args.msa_cache_dir, entry["sequence"]).usable
        ]
        print(
            f"MSA cache:          {len(available)} of {len(entries)} pool sequences have a usable "
            f"cached a3m; restricting to those"
        )
        entries = available
    if args.limit is not None:
        entries = entries[: args.limit]
    if args.shards > 1:
        if not 0 <= args.shard < args.shards:
            raise SystemExit("--shard must be in [0, --shards)")
        entries = entries[args.shard :: args.shards]
        print(f"shard {args.shard} of {args.shards}: {len(entries)} sequences")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = args.staging_dir or args.cache_dir.parent / f"{args.cache_dir.name}_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    if staging_dir.stat().st_dev != args.cache_dir.stat().st_dev:
        raise SystemExit(
            f"staging directory {staging_dir} is on a different filesystem from the cache "
            f"{args.cache_dir}; os.replace would copy rather than swap and the write would not "
            f"be atomic"
        )
    print(f"staging:            {staging_dir}")

    done: list[str] = []
    failed: dict[str, str] = {}
    corrupt: dict[str, str] = {}
    quarantined: dict[str, list[str]] = {}
    integrity_failures: dict[str, str] = {}
    msa_misses: dict[str, str] = {}
    started = time.monotonic()

    for position, entry in enumerate(entries, start=1):
        sequence_id = entry["sequence_id"]
        sequence = entry["sequence"]

        # Verify what would be skipped, rather than trusting that two paths exist. This is the
        # whole point: the existence test that decides "already cached" is precisely the test the
        # second-save window defeats.
        existing = check_sequence(args.cache_dir, sequence)
        compute_this = True
        if existing.present and existing.arrays_intact:
            done.append(sequence_id)
            compute_this = False
        elif existing.present:
            reason = "; ".join(existing.problems)
            corrupt[sequence_id] = reason
            print(
                f"  [{position}/{len(entries)}] {sequence_id} CORRUPT CACHE ENTRY "
                f"{existing.digest[:12]}: {reason}",
                flush=True,
            )
            if args.on_corrupt == "quarantine":
                moved = quarantine_entry(args.cache_dir, existing.digest)
                quarantined[sequence_id] = [str(path) for path in moved]
                print(f"      quarantined to {', '.join(path.name for path in moved)}")
            elif args.on_corrupt == "recompute":
                # No deletion and no rename: the staged publish below overwrites in place with
                # os.replace, so the entry is corrupt until the instant it is whole.
                print("      recomputing over it (atomic overwrite, nothing deleted)")
            else:
                # Recomputing over it in place would destroy the only record of how the writer
                # was interrupted, and quarantining is the caller's decision, not this loop's.
                failed[sequence_id] = f"corrupt cache entry left in place: {reason}"
                compute_this = False

        msa_file: Path | None = None
        if compute_this and args.msa_cache_dir is not None:
            status = check_a3m(args.msa_cache_dir, sequence)
            if status.usable:
                msa_file = status.path
            elif args.require_cached_msa:
                failed[sequence_id] = f"no usable cached MSA: {status.problem}"
                msa_misses[sequence_id] = str(status.problem)
                print(
                    f"  [{position}/{len(entries)}] {sequence_id} NO CACHED MSA "
                    f"({status.problem}); refusing to fetch one from a GPU job",
                    flush=True,
                )
                compute_this = False
            else:
                msa_misses[sequence_id] = str(status.problem)

        if compute_this:
            for attempt in range(1, args.retries + 2):
                try:
                    compute_embedding_atomically(
                        get_colabfold_embeds,
                        sequence,
                        args.cache_dir,
                        staging_dir,
                        msa_file=msa_file,
                    )
                    # Belt and braces: re-verify at the published location, not just in staging,
                    # so a partial publish is a failure rather than a silent success.
                    published = check_sequence(args.cache_dir, sequence)
                    if not published.arrays_intact:
                        raise WrittenEmbeddingNotIntactError(
                            f"embedding published for {published.digest[:12]} is not intact: "
                            + "; ".join(published.problems)
                        )
                    done.append(sequence_id)
                    break
                except Exception as error:  # noqa: BLE001 - one bad sequence must not end the run
                    if attempt > args.retries:
                        failed[sequence_id] = f"{type(error).__name__}: {error}"
                        if isinstance(error, WrittenEmbeddingNotIntactError):
                            integrity_failures[sequence_id] = str(error)
                        print(f"  [{position}/{len(entries)}] {sequence_id} FAILED: {error}")
                    else:
                        print(
                            f"  [{position}/{len(entries)}] {sequence_id} retry {attempt}: {error}"
                        )
                        time.sleep(args.pause * 5)

        if position % 10 == 0 or position == len(entries):
            elapsed = time.monotonic() - started
            rate = elapsed / position
            print(
                f"[{position}/{len(entries)}] cached={len(done)} failed={len(failed)} "
                f"{rate:.1f}s/seq eta={(len(entries) - position) * rate / 60:.0f}min",
                flush=True,
            )
        time.sleep(args.pause)

    print(
        f"\ncached {len(done)} / {len(entries)}; {len(failed)} failed; "
        f"{len(corrupt)} pre-existing entries corrupt ({len(quarantined)} quarantined)"
    )
    if args.msa_cache_dir is not None:
        print(
            f"MSA cache: {len(entries) - len(msa_misses)} of {len(entries)} sequences had a cached "
            f"a3m, so that many Evoformer passes ran with no network round trip"
        )
    for sequence_id, reason in list(failed.items())[:10]:
        print(f"  {sequence_id}: {reason}")
    for sequence_id, reason in list(corrupt.items())[:10]:
        print(f"  corrupt {sequence_id}: {reason}")

    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "cached": sorted(done),
                    "failed": failed,
                    "corrupt": corrupt,
                    "quarantined": quarantined,
                    "integrity_failures": integrity_failures,
                    "msa_cache_dir": str(args.msa_cache_dir) if args.msa_cache_dir else None,
                    "msa_cache_misses": msa_misses,
                    "on_corrupt": args.on_corrupt,
                    "staging_dir": str(staging_dir),
                    "requested": len(entries),
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"wrote {args.report}")

    # Only the integrity conditions change the exit status. A per-sequence MMseqs2 timeout has
    # always exited zero here -- the job is explicitly written to tolerate and resume past one --
    # and flipping that would recolour every partially-complete shard as a failure. A corrupt
    # entry left in place, or one this run wrote and could not verify, is different in kind: it
    # persists after the job ends and a green status would let a downstream stage trust it.
    if integrity_failures or (corrupt and args.on_corrupt == "report"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
