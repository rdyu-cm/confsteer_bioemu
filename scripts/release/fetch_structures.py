#!/usr/bin/env python3
"""Fetch the public mmCIF entries cryptic-pocket scoring reads, from RCSB.

The 66 entries are 52 MB of public crystallography that would otherwise sit in the repository
forever. ``data/benchmarks/rosters/cryptic_structures.json`` names them; this downloads each into
``data/benchmarks/structures/<entry>.cif``, skipping what is already present, and reports what it
could not get rather than leaving a short directory to be discovered by a scorer later.

Needs network access. Nothing else in the release does.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster",
        type=Path,
        default=REPOSITORY / "data/benchmarks/rosters/cryptic_structures.json",
    )
    parser.add_argument("--out", type=Path, default=REPOSITORY / "data/benchmarks/structures")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--pause", type=float, default=0.2)
    arguments = parser.parse_args()

    roster = json.loads(arguments.roster.read_text())
    entries = [str(e).lower() for e in roster["entries"]]
    template = str(roster.get("source") or "https://files.rcsb.org/download/{entry}.cif")
    arguments.out.mkdir(parents=True, exist_ok=True)

    fetched = skipped = 0
    failures: dict[str, str] = {}
    for entry in entries:
        target = arguments.out / f"{entry}.cif"
        if target.exists() and target.stat().st_size > 0:
            skipped += 1
            continue
        url = template.format(entry=entry)
        for attempt in range(1, arguments.retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                    payload = response.read()
                scratch = target.with_suffix(".cif.part")
                scratch.write_bytes(payload)
                scratch.replace(target)
                fetched += 1
                break
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if attempt == arguments.retries:
                    failures[entry] = f"{type(error).__name__}: {error}"
                else:
                    time.sleep(arguments.pause * attempt)
        time.sleep(arguments.pause)

    print(f"fetched {fetched}, already present {skipped}, of {len(entries)} entries")
    if failures:
        for entry, reason in sorted(failures.items()):
            print(f"  FAILED {entry}: {reason}")
        raise SystemExit(
            f"REFUSING: {len(failures)} entries did not download. Cryptic scoring reads every one "
            "of them, and a short directory would surface later as a smaller panel."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
