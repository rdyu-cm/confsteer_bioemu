"""Filesystem roots, so a config written on one machine runs on another.

Configs in this repository were written against absolute scratch paths, which is correct for a
record of what ran and useless for anyone who clones it. This module gives configs two
placeholders to write instead:

``${SAE_DATA_ROOT}``    inputs -- FASTA pools, property references, embedding caches
``${SAE_OUTPUT_ROOT}``  outputs -- run directories, checkpoints, reports, figures

Both resolve from the environment when set and otherwise fall back to ``<repo>/data`` and
``<repo>/outputs``, so a fresh clone works with no environment at all while a cluster user keeps
pointing them at scratch.

**Placeholder expansion happens after the config is fingerprinted, never before.** The workflow
hashes the config *file bytes*, so two users with different roots produce the same fingerprint for
the same experiment. That is the intended behaviour and matches the direction already taken when
``platform.platform()`` was removed from the training fingerprint: machine identity is not part of
experiment identity. If expansion ran before hashing, every user would get a different fingerprint
for an identical experiment and no checkpoint would ever be shared.

Unset non-standard variables are an **error**, not an empty string. ``os.path.expandvars`` leaves
an unknown ``${FOO}`` in place, which turns into a path with a literal dollar sign in it and fails
somewhere much later with a confusing message.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

#: Inputs. FASTA pools, property references, embedding caches.
DATA_ROOT_VARIABLE = "SAE_DATA_ROOT"
#: Outputs. Run directories, checkpoints, reports, figures.
OUTPUT_ROOT_VARIABLE = "SAE_OUTPUT_ROOT"

#: ``${NAME}`` only. Bare ``$NAME`` is deliberately not expanded: paths legitimately contain
#: dollar signs, and the braced form is unambiguous.
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def repository_root() -> Path:
    """The checkout root, found by walking up for ``pyproject.toml``.

    Falls back to three levels above this file, which is the layout when the package is installed
    editable from the checkout. An installed-non-editable package has no repository, and the
    caller should be setting the environment variables in that case anyway.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[2]


def data_root() -> Path:
    """Input root: ``$SAE_DATA_ROOT`` when set, else ``<repo>/data``."""
    value = os.environ.get(DATA_ROOT_VARIABLE)
    return Path(value).expanduser() if value else repository_root() / "data"


def output_root() -> Path:
    """Output root: ``$SAE_OUTPUT_ROOT`` when set, else ``<repo>/outputs``."""
    value = os.environ.get(OUTPUT_ROOT_VARIABLE)
    return Path(value).expanduser() if value else repository_root() / "outputs"


def external_root() -> Path:
    """Trusted root that externally-supplied absolute paths must resolve under.

    This backs a security check, not a convenience: submission inventories accept absolute paths
    from a config and confine them to one subtree so a crafted config cannot reach arbitrary
    files. It used to be a hard-coded account directory, which confined it to one person's
    machine.

    Resolution order is ``$SAE_EXTERNAL_ROOT``, then :func:`output_root`, so the guard is
    meaningful on any host rather than only where one variable happens to point. It used to also
    read a scheduler-site scratch variable, which named a filesystem convention this code has no
    reason to know: the project's own variable says the same thing without it.
    """
    for variable in ("SAE_EXTERNAL_ROOT",):
        value = os.environ.get(variable)
        if value:
            return Path(value).expanduser()
    return output_root()


def _substitute(name: str) -> str:
    if name == DATA_ROOT_VARIABLE:
        return str(data_root())
    if name == OUTPUT_ROOT_VARIABLE:
        return str(output_root())
    value = os.environ.get(name)
    if value is None:
        raise KeyError(
            f"config refers to ${{{name}}} but that variable is not set. Set it, or use "
            f"${{{DATA_ROOT_VARIABLE}}} / ${{{OUTPUT_ROOT_VARIABLE}}}, which have defaults."
        )
    return value


def expand_text(value: str) -> str:
    """Expand every ``${NAME}`` in one string."""
    return _PLACEHOLDER.sub(lambda match: _substitute(match.group(1)), value)


def expand_placeholders(value: Any) -> Any:
    """Recursively expand ``${NAME}`` through dicts, lists, tuples and strings.

    Mapping *keys* are left alone. A config key is a schema name, not a path, and expanding one
    would let an environment variable silently rename a field.
    """
    if isinstance(value, str):
        return expand_text(value)
    if isinstance(value, dict):
        return {key: expand_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_placeholders(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_placeholders(item) for item in value)
    return value
