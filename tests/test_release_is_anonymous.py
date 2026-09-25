"""No tracked file may carry an author's identity, and none may carry an absolute path.

Two properties, checked structurally. The patterns below describe SHAPES -- an e-mail address, a
personal repository URL, a home or scratch directory, a cluster node name -- rather than listing
the particular names this project had to remove. That is deliberate: a test that enumerated them
would ship a directory of exactly the identifiers it exists to keep out, readable by anyone who
opened the file. The name-specific scan is a build-time step and is not part of the release.

The absolute-path rule is not only about anonymity. A default pointing at one machine's filesystem
runs nowhere else, and it fails at the end of a long job rather than at argument parsing.

Cheap, no GPU, no network. If this is ever weakened, weaken it deliberately and say why.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Shapes that identify a person, an account or a machine, whoever they happen to be. The
#: repository allowlist covers upstream projects this work cites, which are citations, not leaks.
IDENTIFYING_SHAPES = (
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "an e-mail address"),
    (r"github\.com/(?!microsoft/|aqlaboratory/|deepmind/)[A-Za-z0-9-]+/", "a personal repository"),
    (r"/(?:home|Users)/[A-Za-z0-9_.-]+/", "a home directory"),
    (r"/scratch/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", "a scratch directory"),
    # A compute-node hostname in the usual rack-slot form (letters, two digits, a dash, two digits, 'n', two
    # digits). Written generically
    # on purpose: a pattern naming one site's prefix would identify the site.
    (r"\b[a-z]{2,4}\d\d-\d\dn\d\d\b", "a cluster node name"),
    # A scheduler job id: an eight-digit number in a comment is a cross-reference into logs only
    # the authors can read, and the range itself narrows down the site. Added after 17 of them
    # shipped in provenance comments, which every other pattern here passed over.
    (r"\bjobs?\s+\d{7,9}\b", "a scheduler job id"),
    # A queue named AS a queue, not the ordinary English word: the scheduler's partition flag, or
    # a submission command. Naming the queues names the site.
    (
        "--parti"
        + r"tion[= ]|\bsba"
        + r"tch\b|\bsal"
        + r"loc\b|\bsq"
        + r"ueue\b|\bscon"
        + r"trol\b|(?i:#sba"
        + r"tch)|\bsr"
        + r"un\b|\bsac"
        + r"ct\b",
        "a scheduler command",
    ),
    # Site filesystem variables. Some are common across scheduler sites, but a shipped file has no
    # business naming any of them: a clone resolves its inputs from its own root. The literals below
    # are SPLIT so this file does not match its own patterns -- exempting the file instead is what
    # previously let it carry the very identifiers it guards.
    (
        r"\$(?:L_)?SCRAT" + r"CH\b|\$GROUP_" + r"(?:HOME|SCRATCH)\b|\$O" + r"AK\b",
        "a site filesystem variable",
    ),
)

#: Dates that tie the work to internal logs. Allowed where a date is part of PUBLIC provenance --
#: the tool version stamped into a deposited structure, and the release date itself.
INTERNAL_DATE = r"\b20(?:2[4-9])-[01]\d-[0-3]\d\b"
DATE_ALLOWED = {"CITATION.cff", "data/benchmarks/domain_motion_assets"}

#: Absolute paths anywhere in shipped code: a clone must resolve inputs from its own root.
ABSOLUTE_PATH = r'["\'](/(?:scratch|home|Users|tmp|var|mnt|data|opt)/[^"\']*)["\']'

#: ``uv.lock`` records upstream wheel URLs; none of them are ours. This file is skipped by the
#: absolute-path rule only: its regex literals are path SHAPES, not paths, and they are still
#: subject to the identity rules above -- which is the point of describing shapes rather than
#: naming anyone.
SKIP = {"uv.lock"}
SKIP_ABSOLUTE_PATH = SKIP | {"tests/test_release_is_anonymous.py"}


def _in_git_checkout() -> bool:
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=REPO, capture_output=True, check=True,
        )  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


#: These are RELEASE guards: they check what the source repository ships, and git is what says so.
#: A copy downloaded as an archive has no git metadata, and in it they would also scan whatever the
#: reader has since written into the tree -- failing on the reader's files rather than on the
#: release. So they skip there, and say why, instead of failing for a reason that is not a defect.
needs_git = pytest.mark.skipif(
    not _in_git_checkout(),
    reason="a release guard on what the source repository ships; this copy has no git metadata",
)


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return [REPO / name for name in out.stdout.split("\0") if name]


def readable_text(path: Path) -> str | None:
    """The file's text, or None when it is binary. Binaries are checked separately."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


@needs_git
def test_no_tracked_file_identifies_a_person_or_a_machine() -> None:
    offences: list[str] = []
    for path in tracked_files():
        relative = str(path.relative_to(REPO))
        if relative in SKIP:
            continue
        text = readable_text(path)
        if text is None:
            continue
        for pattern, what in IDENTIFYING_SHAPES:
            for match in re.finditer(pattern, text):
                line = text.count("\n", 0, match.start()) + 1
                offences.append(f"{relative}:{line}: {match.group(0)!r} looks like {what}")
    assert not offences, "identifying strings survive:\n  " + "\n  ".join(offences[:40])


@needs_git
def test_no_tracked_path_identifies_a_person_or_a_machine() -> None:
    offences = [
        f"{path.relative_to(REPO)} ({what})"
        for path in tracked_files()
        for pattern, what in IDENTIFYING_SHAPES
        if re.search(pattern, str(path.relative_to(REPO)))
    ]
    assert not offences, f"file names leak: {offences}"


@needs_git
def test_no_python_file_carries_an_absolute_default_path() -> None:
    offences: list[str] = []
    for path in tracked_files():
        if path.suffix != ".py" or str(path.relative_to(REPO)) in SKIP_ABSOLUTE_PATH:
            continue
        text = readable_text(path) or ""
        for match in re.finditer(ABSOLUTE_PATH, text):
            line = text.count("\n", 0, match.start()) + 1
            offences.append(f"{path.relative_to(REPO)}:{line}: {match.group(1)}")
    assert not offences, "absolute paths in shipped code:\n  " + "\n  ".join(offences[:40])


@needs_git
def test_no_tracked_file_carries_an_internal_date() -> None:
    """A date in a shipped comment cross-references a log nobody outside can read.

    Kept separate from the identity shapes because a date is only identifying in combination, and
    because two places carry one legitimately: the release date, and the tool version stamped into
    the deposited structures by whoever processed them.
    """
    offences: list[str] = []
    for path in tracked_files():
        relative = str(path.relative_to(REPO))
        if any(relative.startswith(allowed) for allowed in DATE_ALLOWED):
            continue
        text = readable_text(path)
        if text is None:
            continue
        for match in re.finditer(INTERNAL_DATE, text):
            line = text.count("\n", 0, match.start()) + 1
            offences.append(f"{relative}:{line}: {match.group(0)}")
    assert not offences, "internal dates survive:\n  " + "\n  ".join(offences[:40])


def test_no_shipped_checkpoint_hides_an_identifier_in_its_metadata() -> None:
    """A text scan cannot see inside a pickle, and that is exactly where one path survived.

    Covers the benchmark reference artifacts as well as the models: both are pickles, and only one
    of the two directories was scanned before.
    """
    torch = pytest.importorskip("torch")
    offences: list[str] = []
    checkpoints = sorted((REPO / "models").glob("*.pt")) + sorted(
        (REPO / "data/benchmarks/references").glob("*.pt")
    )
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        text = json.dumps(
            {k: v for k, v in payload.items() if not hasattr(v, "shape") and k != "state_dict"},
            default=str,
        )
        for pattern, what in IDENTIFYING_SHAPES:
            match = re.search(pattern, text)
            if match:
                offences.append(f"{checkpoint.name}: {match.group(0)!r} looks like {what}")
    assert not offences, "checkpoint metadata leaks:\n  " + "\n  ".join(offences)


@needs_git
def test_every_shipped_script_is_one_the_release_actually_runs() -> None:
    """Nothing under ``scripts/`` ships unless the README names it or the benchmark runs it.

    Reachability, not taste. Build tooling that lands here is the dangerous kind of extra file: the
    anonymisation scrub writes a report LISTING the identifiers it removed, which is a directory of
    exactly what must not ship, and no shape-based pattern can catch a bare surname. It became
    tracked once, when a `.gitignore` rewrite dropped the rule that had been excluding it.
    """
    readme = (REPO / "README.md").read_text()
    launcher = (REPO / "scripts/run_benchmark.py").read_text()
    unreachable = [
        relative
        for path in tracked_files()
        if (relative := str(path.relative_to(REPO))).startswith("scripts/")
        and relative not in readme
        and relative not in launcher
        and Path(relative).name not in launcher
    ]
    assert not unreachable, (
        "these ship but nothing runs them; if one is build tooling it must not ship at all:\n  "
        + "\n  ".join(unreachable)
    )
