"""Integrity checking for the BioEmu Evoformer embedding cache.

Why existence is not enough
---------------------------

``bioemu.get_embeds.get_colabfold_embeds`` treats a sequence as cached when two paths exist::

    if os.path.exists(single_rep_file) and os.path.exists(pair_rep_file):
        return single_rep_file, pair_rep_file
    ...
    np.save(single_rep_file, single_repr)      # (L, 384) float16
    np.save(pair_rep_file, pair_repr)          # (L, L, 128) float16
    write_fasta(...)                           # only after BOTH arrays

Neither ``np.save`` is atomic: there is no temp file and no rename. A process killed while the
SECOND save is in flight leaves a complete ``_single.npy`` next to a truncated ``_pair.npy`` with
both paths present, so every later run skips that sequence as cached and every downstream number
computed from it is computed from a short array. A kill during the FIRST save is harmless, because
``and`` forces the recompute. The window is therefore real but narrow, and it is invisible to any
check that only stats paths -- which is what
:func:`sae_bioemu.bioemu.cryptic_generation.require_cached_embeddings` does.

Eleven embedding shards have been SIGKILLed at their walltime across two rounds of this
programme's cache build, so the window has been sampled eleven times.

What "complete" means here
--------------------------

Three independent conditions, because a file truncated on a block boundary can satisfy the first
two on its own:

1. The ``.npy`` header parses. A kill inside the first 128 bytes leaves an unparsable stub.
2. The byte count implied by the declared shape and dtype matches the bytes actually on disk. This
   is the condition that catches a truncated body, and it is why the check reads the header rather
   than the array: ``np.load(..., mmap_mode="r")`` and the header parse both cost one small read,
   while a real load of this cache would move tens of gigabytes.
3. The declared shape agrees with the sequence it claims to encode -- ``(L, C)`` for the single
   representation and ``(L, L, C)`` for the pair representation, with ``L`` the residue count from
   the sibling FASTA. Condition 2 alone cannot see a file that was truncated and then re-written
   shorter, and this is also the check that catches a hash collision or a mis-keyed entry.

The channel widths ``C`` are recorded but not asserted against a constant: they are a property of
the AlphaFold2 configuration BioEmu ships, not of this repository, and pinning them here would
turn an upstream version bump into a spurious corruption report. Truncation changes the *leading*
dimensions or the byte count, both of which are checked.

The sibling FASTA is a corroborating signal, not a requirement. ``write_fasta`` runs after both
saves, so an entry with two arrays and no FASTA was interrupted somewhere in exactly the dangerous
window -- but an entry can also be legitimately array-complete and FASTA-less if the kill landed
between the second save and the FASTA write. Missing or mismatched FASTA is reported as its own
problem class so a reader can tell the two apart; only the array conditions are what
:meth:`EmbeddingCacheEntry.arrays_intact` reports.

Nothing here imports torch, touches a GPU, or reaches the network.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

__all__ = [
    "ArrayIntegrity",
    "CacheAudit",
    "CorruptEmbeddingCacheError",
    "EMBEDDING_FASTA_SUFFIX",
    "EMBEDDING_PAIR_SUFFIX",
    "EMBEDDING_SINGLE_SUFFIX",
    "EmbeddingCacheEntry",
    "audit_cache",
    "cache_embeddings_atomically",
    "check_digest",
    "check_sequence",
    "iter_cache_digests",
    "quarantine_entry",
    "read_cached_fasta",
    "require_intact_embeddings",
    "sequence_digest",
]

#: Suffixes ``get_colabfold_embeds`` writes, in the order it writes them.
EMBEDDING_SINGLE_SUFFIX = "_single.npy"
EMBEDDING_PAIR_SUFFIX = "_pair.npy"
EMBEDDING_FASTA_SUFFIX = ".fasta"


#: Prefix of the private staging directories :func:`cache_embeddings_atomically` writes into.
#: Inside the cache directory on purpose -- ``os.replace`` is only atomic WITHIN a filesystem, and
#: a staging area under ``/tmp`` or ``$TMPDIR`` would land on a different mount and silently
#: degrade the publish into a copy, which is exactly the non-atomic write this function exists to
#: remove. The leading dot keeps it out of :func:`iter_cache_digests`, which globs the three
#: suffixes.
EMBEDDING_STAGING_PREFIX = ".staging-"


class CorruptEmbeddingCacheError(RuntimeError):
    """Raised when a cache entry exists but is not a usable embedding."""


def sequence_digest(sequence: str) -> str:
    """Return the cache key BioEmu uses for ``sequence``.

    A byte-for-byte copy of ``bioemu.get_embeds.shahexencode``, restated rather than imported so
    that the audit runs without the CUDA/JAX stack that importing ``bioemu`` pulls in.
    """
    return hashlib.sha256(sequence.encode()).hexdigest()


@dataclass(frozen=True)
class ArrayIntegrity:
    """One ``.npy`` file's header, its size on disk, and what disagrees between them."""

    path: Path
    exists: bool
    file_bytes: int
    header_bytes: int | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    expected_bytes: int | None = None
    problem: str | None = None

    @property
    def intact(self) -> bool:
        return self.exists and self.problem is None

    @property
    def missing_bytes(self) -> int | None:
        """Bytes the declared shape needs that the file does not have; negative means surplus."""
        if self.header_bytes is None or self.expected_bytes is None:
            return None
        return self.header_bytes + self.expected_bytes - self.file_bytes


def _inspect_array(path: Path) -> ArrayIntegrity:
    """Read only the ``.npy`` header and compare the shape it declares against the file size."""
    try:
        file_bytes = path.stat().st_size
    except OSError:
        return ArrayIntegrity(path=path, exists=False, file_bytes=0, problem="missing")
    if not path.is_file():
        return ArrayIntegrity(path=path, exists=False, file_bytes=0, problem="missing")

    try:
        with path.open("rb") as handle:
            version = npy_format.read_magic(handle)
            # ``read_array_header_1_0``/``_2_0`` are the public readers; anything else is a
            # format this cache was not written in and is reported rather than guessed at.
            if version == (1, 0):
                shape, fortran_order, descr = npy_format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran_order, descr = npy_format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"unsupported .npy format version {version}")
            header_bytes = handle.tell()
    except Exception as error:  # noqa: BLE001 - any header failure is the same verdict
        return ArrayIntegrity(
            path=path,
            exists=True,
            file_bytes=file_bytes,
            problem=f"unreadable header ({type(error).__name__}: {error})",
        )

    try:
        # numpy >= 1.26's readers already resolve the descr to a dtype; older ones hand back the
        # raw descr. Accept both rather than assuming, because getting this wrong reports the
        # entire cache as corrupt -- which is exactly what it did before this branch existed.
        dtype = descr if isinstance(descr, np.dtype) else npy_format.descr_to_dtype(descr)
    except Exception as error:  # noqa: BLE001
        return ArrayIntegrity(
            path=path,
            exists=True,
            file_bytes=file_bytes,
            header_bytes=header_bytes,
            shape=tuple(shape),
            problem=f"undecodable dtype {descr!r} ({type(error).__name__}: {error})",
        )

    expected_bytes = int(dtype.itemsize) * int(np.prod(shape, dtype=np.int64)) if shape else 0
    common = {
        "path": path,
        "exists": True,
        "file_bytes": file_bytes,
        "header_bytes": header_bytes,
        "shape": tuple(int(item) for item in shape),
        "dtype": dtype.str,
        "expected_bytes": expected_bytes,
    }
    if dtype.hasobject:
        return ArrayIntegrity(**common, problem="pickled object array, not a raw embedding")
    if fortran_order:
        return ArrayIntegrity(**common, problem="Fortran-ordered, which np.save did not write")

    actual_body = file_bytes - header_bytes
    if actual_body < expected_bytes:
        return ArrayIntegrity(
            **common,
            problem=(
                f"truncated body: shape {tuple(shape)} {dtype.str} needs {expected_bytes} bytes, "
                f"file holds {actual_body}"
            ),
        )
    if actual_body > expected_bytes:
        return ArrayIntegrity(
            **common,
            problem=(
                f"trailing bytes: shape {tuple(shape)} {dtype.str} needs {expected_bytes} bytes, "
                f"file holds {actual_body}"
            ),
        )
    return ArrayIntegrity(**common)


def read_cached_fasta(path: Path) -> str | None:
    """Return the sequence in a cache-sibling FASTA, or ``None`` if it is absent or empty.

    ``write_fasta`` wraps at 60 columns, so the sequence is the concatenation of every non-header
    line. Parsed by hand rather than through Biopython to keep the audit dependency-free.
    """
    try:
        text = path.read_text()
    except OSError:
        return None
    letters = [line.strip() for line in text.splitlines() if line and not line.startswith(">")]
    joined = "".join(letters)
    return joined or None


@dataclass(frozen=True)
class EmbeddingCacheEntry:
    """The verdict on one ``(digest, single, pair, fasta)`` cache entry."""

    digest: str
    single: ArrayIntegrity
    pair: ArrayIntegrity
    fasta_path: Path
    fasta_sequence: str | None
    residue_count: int | None
    problems: tuple[str, ...]

    @property
    def present(self) -> bool:
        """True when both arrays exist, which is all ``get_colabfold_embeds`` checks."""
        return self.single.exists and self.pair.exists

    @property
    def arrays_intact(self) -> bool:
        """True when both arrays parse, are the declared length, and match the residue count."""
        return not self.problems

    @property
    def fasta_present(self) -> bool:
        return self.fasta_sequence is not None

    @property
    def length_verified(self) -> bool:
        """True when a residue count was available to compare the leading dimensions against.

        A False here is not corruption; it is the absence of condition 3, and it is counted
        separately so a clean audit cannot be read as stronger than the evidence behind it.
        """
        return self.residue_count is not None

    def describe(self) -> str:
        return f"{self.digest}: " + "; ".join(self.problems) if self.problems else self.digest


def _shape_problems(
    entry_kind: str, array: ArrayIntegrity, rank: int, residue_count: int | None
) -> list[str]:
    """Check declared rank, and the leading dimensions against the residue count when it is known."""
    if array.shape is None:
        return []
    problems: list[str] = []
    if len(array.shape) != rank:
        problems.append(f"{entry_kind} rank {len(array.shape)}, expected {rank}")
        return problems
    if residue_count is None:
        return problems
    leading = array.shape[: rank - 1]
    if any(int(size) != residue_count for size in leading):
        problems.append(
            f"{entry_kind} shape {array.shape} disagrees with the FASTA residue count "
            f"{residue_count}"
        )
    return problems


def check_digest(
    cache_dir: Path | str, digest: str, sequence: str | None = None
) -> EmbeddingCacheEntry:
    """Verify one cache entry by its sequence hash.

    ``sequence`` is optional: when given it fixes the expected residue count and is checked against
    the recorded FASTA, and when omitted the residue count comes from that FASTA alone. An entry
    with neither a ``sequence`` argument nor a FASTA is still checked for header and byte-count
    completeness; only the length agreement is skipped, and that is reported.
    """
    root = Path(os.path.expanduser(str(cache_dir)))
    single = _inspect_array(root / f"{digest}{EMBEDDING_SINGLE_SUFFIX}")
    pair = _inspect_array(root / f"{digest}{EMBEDDING_PAIR_SUFFIX}")
    fasta_path = root / f"{digest}{EMBEDDING_FASTA_SUFFIX}"
    fasta_sequence = read_cached_fasta(fasta_path)

    problems: list[str] = []
    if not single.exists:
        problems.append("single representation missing")
    elif single.problem is not None:
        problems.append(f"single {single.problem}")
    if not pair.exists:
        problems.append("pair representation missing")
    elif pair.problem is not None:
        problems.append(f"pair {pair.problem}")

    residue_count: int | None = None
    if sequence is not None:
        residue_count = len(sequence)
        expected = sequence_digest(sequence)
        if expected != digest:
            problems.append(f"digest {digest} is not sha256 of the supplied sequence ({expected})")
        if fasta_sequence is not None and fasta_sequence != sequence:
            problems.append("recorded FASTA does not match the supplied sequence")
    elif fasta_sequence is not None:
        residue_count = len(fasta_sequence)

    if fasta_sequence is not None and sequence_digest(fasta_sequence) != digest:
        problems.append("recorded FASTA does not hash to this entry's digest")

    problems.extend(_shape_problems("single", single, 2, residue_count))
    problems.extend(_shape_problems("pair", pair, 3, residue_count))

    return EmbeddingCacheEntry(
        digest=digest,
        single=single,
        pair=pair,
        fasta_path=fasta_path,
        fasta_sequence=fasta_sequence,
        residue_count=residue_count,
        problems=tuple(problems),
    )


def check_sequence(cache_dir: Path | str, sequence: str) -> EmbeddingCacheEntry:
    """Verify the cache entry a sequence would hash to."""
    return check_digest(cache_dir, sequence_digest(sequence), sequence=sequence)


def iter_cache_digests(cache_dir: Path | str) -> list[str]:
    """Return every digest the cache directory mentions, from any of the three suffixes.

    Taking the union rather than intersecting on ``_single.npy`` is deliberate: an entry with only
    a pair array, or only a FASTA, is itself a symptom and must appear in the audit rather than be
    filtered out of it.
    """
    root = Path(os.path.expanduser(str(cache_dir)))
    digests: set[str] = set()
    for path in root.iterdir():
        name = path.name
        for suffix in (EMBEDDING_SINGLE_SUFFIX, EMBEDDING_PAIR_SUFFIX, EMBEDDING_FASTA_SUFFIX):
            if name.endswith(suffix):
                digests.add(name[: -len(suffix)])
                break
    return sorted(digests)


@dataclass(frozen=True)
class CacheAudit:
    """The result of walking a whole cache directory."""

    cache_dir: Path
    entries: tuple[EmbeddingCacheEntry, ...]

    @property
    def total(self) -> int:
        return len(self.entries)

    @property
    def intact(self) -> tuple[EmbeddingCacheEntry, ...]:
        return tuple(entry for entry in self.entries if entry.arrays_intact)

    @property
    def corrupt(self) -> tuple[EmbeddingCacheEntry, ...]:
        return tuple(entry for entry in self.entries if not entry.arrays_intact)

    @property
    def missing_fasta(self) -> tuple[EmbeddingCacheEntry, ...]:
        """Entries whose arrays exist but whose FASTA does not.

        ``write_fasta`` runs last, so this set is the upper bound on entries interrupted anywhere
        after the first save -- a superset of, and a cheap independent cross-check on, the entries
        the byte-count test flags.
        """
        return tuple(entry for entry in self.entries if entry.present and not entry.fasta_present)

    @property
    def unverified_length(self) -> tuple[EmbeddingCacheEntry, ...]:
        """Entries checked for byte completeness but with no residue count to compare against."""
        return tuple(entry for entry in self.entries if not entry.length_verified)


def audit_cache(cache_dir: Path | str, sequences: Mapping[str, str] | None = None) -> CacheAudit:
    """Verify every entry in ``cache_dir``.

    ``sequences`` optionally maps digest to sequence for entries whose length is known from a pool
    rather than from the cache's own FASTA; entries absent from it fall back to the FASTA.
    """
    root = Path(os.path.expanduser(str(cache_dir)))
    known = dict(sequences or {})
    entries = [
        check_digest(root, digest, sequence=known.get(digest))
        for digest in iter_cache_digests(root)
    ]
    return CacheAudit(cache_dir=root, entries=tuple(entries))


def require_intact_embeddings(cache_dir: Path | str, sequences: Mapping[str, str]) -> None:
    """Fail closed unless every named sequence has a cached embedding that is actually complete.

    The strict twin of :func:`sae_bioemu.bioemu.cryptic_generation.require_cached_embeddings`,
    which checks only that two paths exist. The message names the sequence IDs, because the
    recovery action is to recompute those sequences and a count does not say which.
    """
    root = Path(os.path.expanduser(str(cache_dir)))
    bad: list[tuple[str, EmbeddingCacheEntry]] = []
    for sequence_id, sequence in sorted(sequences.items()):
        entry = check_sequence(root, sequence)
        if not entry.arrays_intact:
            bad.append((sequence_id, entry))
    if bad:
        listed = "; ".join(
            f"{name} ({entry.digest[:12]}: {', '.join(entry.problems)})" for name, entry in bad[:10]
        )
        raise CorruptEmbeddingCacheError(
            f"BioEmu Evoformer embedding cache {root} has {len(bad)} of {len(sequences)} sequences "
            f"missing or corrupt. Affected: {listed}"
        )


def cache_embeddings_atomically(
    cache_dir: Path | str,
    sequence: str,
    compute: Callable[[str], object],
) -> tuple[Path, Path]:
    """Populate one cache entry so that an interrupted write can never look complete.

    THE WINDOW THIS CLOSES is the one this module's docstring opens with: ``get_colabfold_embeds``
    writes ``_single.npy`` then ``_pair.npy`` with two bare ``np.save`` calls, so a process killed
    inside the SECOND save leaves a complete single beside a truncated pair with both paths
    present -- and every later run skips that sequence as cached. Eleven shards of this programme's
    cache have already been SIGKILLed at their walltime, and moving embedding work to the
    preemptible queue samples the window far more often than a walltime kill does.

    THE FIX IS ORDER, NOT LOCKING. ``compute`` is handed a PRIVATE STAGING DIRECTORY inside the
    cache, runs there, and is verified there; only then is the result published with ``os.replace``,
    which is atomic within a filesystem. The publish order is deliberate and is the whole guarantee:

        fasta -> single -> **pair last**

    because the reader's gate is ``exists(single) and exists(pair)``. Making the pair the last name
    to appear means an interruption at ANY point leaves the entry looking ABSENT, so the next run
    recomputes it. There is no window in which a partial entry reads as complete.

    IT ALSO REMOVES THE CONCURRENT-WRITER HAZARD, which is why the staging directory carries a pid
    and a uuid rather than just the digest. Two processes computing the same sequence stage into
    different directories and publish identical bytes; last writer wins and both outcomes are
    complete. The previous failure mode -- two jobs interleaving ``np.save`` calls on one path --
    cannot arise, so a cache build no longer has to be serialised to be safe.

    A PRE-EXISTING ENTRY IS NEVER TRUSTED ON EXISTENCE ALONE. An intact entry is returned without
    recomputing; a corrupt one is QUARANTINED rather than deleted, per :func:`quarantine_entry`,
    so the evidence of how the writer was interrupted survives and the next call recomputes.

    ``compute`` receives the staging directory as a string and is expected to write the same three
    files ``get_colabfold_embeds`` would, keyed by the same digest::

        cache_embeddings_atomically(
            cache_dir, seq, lambda staging: get_colabfold_embeds(seq=seq, cache_embeds_dir=staging)
        )

    Returns the published ``(single, pair)`` paths. Raises :class:`CorruptEmbeddingCacheError` if
    ``compute`` produced something that does not verify, WITHOUT publishing it -- a failure here
    leaves the cache exactly as it was found.
    """
    root = Path(os.path.expanduser(str(cache_dir)))
    digest = sequence_digest(sequence)
    single = root / f"{digest}{EMBEDDING_SINGLE_SUFFIX}"
    pair = root / f"{digest}{EMBEDDING_PAIR_SUFFIX}"

    existing = check_digest(root, digest, sequence=sequence)
    if existing.arrays_intact:
        return single, pair
    if existing.present:
        quarantine_entry(root, digest)

    root.mkdir(parents=True, exist_ok=True)
    staging = root / f"{EMBEDDING_STAGING_PREFIX}{digest}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        compute(str(staging))
        staged = check_digest(staging, digest, sequence=sequence)
        if not staged.arrays_intact:
            raise CorruptEmbeddingCacheError(
                f"embedding computation for {digest[:12]} produced an unusable entry and was NOT "
                f"published; the cache at {root} is unchanged. Problems: {'; '.join(staged.problems)}"
            )
        # Publish. The pair is last because the reader's gate short-circuits on it.
        staged_fasta = staging / f"{digest}{EMBEDDING_FASTA_SUFFIX}"
        if staged_fasta.is_file():
            os.replace(staged_fasta, root / f"{digest}{EMBEDDING_FASTA_SUFFIX}")
        os.replace(staging / f"{digest}{EMBEDDING_SINGLE_SUFFIX}", single)
        os.replace(staging / f"{digest}{EMBEDDING_PAIR_SUFFIX}", pair)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return single, pair


def quarantine_entry(cache_dir: Path | str, digest: str, tag: str = "corrupt") -> tuple[Path, ...]:
    """Rename a bad entry's files aside so the next run recomputes it. Never deletes.

    Deletion is the obvious repair and the wrong one: a corrupt embedding is the only evidence of
    where and how the writer was interrupted, and a rename both preserves it and makes
    ``get_colabfold_embeds``'s existence test miss. Returns the new paths.
    """
    root = Path(os.path.expanduser(str(cache_dir)))
    moved: list[Path] = []
    for suffix in (EMBEDDING_SINGLE_SUFFIX, EMBEDDING_PAIR_SUFFIX, EMBEDDING_FASTA_SUFFIX):
        source = root / f"{digest}{suffix}"
        if not source.exists():
            continue
        target = root / f"{digest}{suffix}.{tag}"
        index = 1
        while target.exists():
            target = root / f"{digest}{suffix}.{tag}.{index}"
            index += 1
        source.rename(target)
        moved.append(target)
    return tuple(moved)


def format_audit(audit: CacheAudit, limit: int | None = None) -> Iterator[str]:
    """Yield a human-readable audit summary, corrupt entries first."""
    yield f"cache:            {audit.cache_dir}"
    yield f"entries:          {audit.total}"
    yield f"complete:         {len(audit.intact)}"
    yield f"corrupt/missing:  {len(audit.corrupt)}"
    yield f"arrays w/o fasta: {len(audit.missing_fasta)}"
    yield f"length unchecked: {len(audit.unverified_length)}"
    corrupt: Iterable[EmbeddingCacheEntry] = audit.corrupt
    if limit is not None:
        corrupt = list(audit.corrupt)[:limit]
    for entry in corrupt:
        yield f"  {entry.digest}: {'; '.join(entry.problems)}"
