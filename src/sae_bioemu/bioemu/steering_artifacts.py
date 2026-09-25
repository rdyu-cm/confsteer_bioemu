"""Identity and atomic-write helpers shared by BioEmu steering training and evaluation.

These live in their own module on purpose. ``steering_workflow.py`` (the training entry
point) and ``steering_evaluation.py`` (the held-out evaluation loop) are hashed into
*different* code fingerprints (``docs/steering/DESIGN_DECISIONS.md`` D14), so neither may
reach into the other for a helper: importing across that boundary would smuggle evaluation
code into the training fingerprint, or training code into the evaluation one, and the
segmentation would stop being structural.

This module is classified with the generation-shared group, because both paths execute it
and a change to fingerprint canonicalization or to atomic publication changes what both a
checkpoint and an evaluation report *are*.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_fingerprint(value: object) -> str:
    """Return the SHA-256 of a canonically serialized JSON value."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Publish a JSON document atomically, fsyncing the file and its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def uint32(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**32:
        raise ValueError(f"{name} must be a uint32 integer")
    return value
