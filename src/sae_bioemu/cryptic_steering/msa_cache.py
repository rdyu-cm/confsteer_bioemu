"""An a3m cache that lets the MSA fetch and the Evoformer pass run on different machines.

The inefficiency this exists to remove
---------------------------------------

``bioemu.get_embeds.get_colabfold_embeds`` does two unrelated things in one call::

    a3m_string = _get_a3m_string(seq, msa_file, msa_host_url)   # ColabFold MMseqs2, over the network
    single_repr, pair_repr = get_embeddings(seq, a3m_string)     # AlphaFold2 Evoformer, on the GPU

The first blocks on a remote queue on a shared public web service; only the second needs a GPU.
Because they are one call, a GPU sits idle for the whole MSA round trip. ``run_mmseqs2``'s own
progress bar budgets ``TIME_ESTIMATE = 150 * len(seqs_unique)`` seconds -- 2.5 minutes per sequence
of purely remote work -- against a measured whole-pipeline cost of about 4.5 minutes per sequence.

``get_colabfold_embeds`` already takes the way out: passing ``msa_file`` makes ``_get_a3m_string``
read that file and never touch the network. So the fetch can be done by cheap, network-bound,
CPU-only jobs on the least contended partition, and the GPU jobs can run back to back on cached
a3m files.

Why the write is staged
------------------------

The same reason the embedding write is. An a3m half on disk is indistinguishable from a whole one
to any check that stats a path, and the MSA phase is meant to run preemptible, where preemption
is routine rather than exceptional. :func:`write_a3m` writes to a temporary file in the same
directory and then ``os.replace``s it into place, which is atomic within a filesystem on POSIX. The
cache entry is therefore an immutable, idempotent unit that is either absent or complete, which is
the restart evidence required before a stage may run on a preemptible queue.

Note what content validation can and cannot do here. Unlike a ``.npy``, an a3m carries no declared
length, so there is no byte count to check a truncated file against; a file cut short mid-alignment
is still syntactically an a3m. The guarantee comes from the atomic write, not from inspection.
:func:`check_a3m` therefore verifies identity -- that this file is the MSA for THIS sequence -- and
non-emptiness, and the docstrings say so rather than implying a completeness check it cannot make.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sae_bioemu.cryptic_steering.embedding_cache import sequence_digest

__all__ = [
    "A3M_SUFFIX",
    "A3mStatus",
    "a3m_path",
    "check_a3m",
    "cached_a3m_digests",
    "write_a3m",
]

#: Cached MSAs are keyed by the same sequence digest as the embeddings, so one sequence has one
#: name in both caches and the two can be joined without a side table.
A3M_SUFFIX = ".a3m"


def a3m_path(cache_dir: Path | str, sequence: str) -> Path:
    """Return where a sequence's MSA is cached."""
    root = Path(os.path.expanduser(str(cache_dir)))
    return root / f"{sequence_digest(sequence)}{A3M_SUFFIX}"


@dataclass(frozen=True)
class A3mStatus:
    """Whether a cached MSA exists and belongs to the sequence it is keyed under."""

    path: Path
    exists: bool
    query_matches: bool
    row_count: int
    problem: str | None = None

    @property
    def usable(self) -> bool:
        return self.exists and self.problem is None


def check_a3m(cache_dir: Path | str, sequence: str) -> A3mStatus:
    """Verify a cached a3m is present, non-empty, and is the MSA for ``sequence``.

    The first record of a ColabFold a3m is the query itself, so the sequence on the line after the
    first header identifies the file. With ``use_env=True`` the UniRef and environmental alignments
    are concatenated and the query appears again partway through; only the first occurrence is
    checked, because that is the one whose position is guaranteed.

    This is an IDENTITY check, not a completeness check: an a3m declares no length, so a file cut
    short mid-alignment cannot be distinguished from a short alignment by inspection. Completeness
    comes from :func:`write_a3m`'s atomic publish instead.
    """
    path = a3m_path(cache_dir, sequence)
    if not path.is_file():
        return A3mStatus(
            path=path, exists=False, query_matches=False, row_count=0, problem="missing"
        )
    try:
        text = path.read_text()
    except OSError as error:
        return A3mStatus(
            path=path,
            exists=True,
            query_matches=False,
            row_count=0,
            problem=f"unreadable ({type(error).__name__}: {error})",
        )
    lines = text.splitlines()
    row_count = sum(1 for line in lines if line.startswith(">"))
    if not lines or not lines[0].startswith(">"):
        return A3mStatus(
            path=path,
            exists=True,
            query_matches=False,
            row_count=row_count,
            problem="empty or not in a3m format",
        )
    query = lines[1].strip() if len(lines) > 1 else ""
    if query != sequence:
        return A3mStatus(
            path=path,
            exists=True,
            query_matches=False,
            row_count=row_count,
            problem=(
                f"query row is {len(query)} residues and does not match the {len(sequence)}-residue "
                f"sequence this entry is keyed under"
            ),
        )
    return A3mStatus(path=path, exists=True, query_matches=True, row_count=row_count)


def write_a3m(cache_dir: Path | str, sequence: str, a3m_text: str) -> Path:
    """Publish one MSA atomically, and refuse to publish one that is not for this sequence.

    Written to a temporary file in the DESTINATION DIRECTORY -- not ``/tmp`` and not
    node-local scratch -- because ``os.replace`` is atomic only within a filesystem; across one it
    degrades to a copy and the guarantee is gone.
    """
    root = Path(os.path.expanduser(str(cache_dir)))
    root.mkdir(parents=True, exist_ok=True)
    lines = a3m_text.splitlines()
    if len(lines) < 2 or not lines[0].startswith(">") or lines[1].strip() != sequence:
        raise ValueError(
            "refusing to cache an a3m whose query row is not the sequence it is keyed under; "
            "a mismatched MSA would silently condition the Evoformer pass on the wrong protein"
        )
    target = root / f"{sequence_digest(sequence)}{A3M_SUFFIX}"
    handle, staged = tempfile.mkstemp(dir=str(root), prefix=f"{target.stem}-", suffix=".partial")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(a3m_text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, target)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise
    return target


def cached_a3m_digests(cache_dir: Path | str) -> set[str]:
    """Return every digest with a published a3m, ignoring in-flight ``.partial`` staging files."""
    root = Path(os.path.expanduser(str(cache_dir)))
    if not root.is_dir():
        return set()
    return {
        path.name[: -len(A3M_SUFFIX)] for path in root.iterdir() if path.name.endswith(A3M_SUFFIX)
    }
