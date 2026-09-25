"""Process environment the entry points repair before anything imports matplotlib."""

from __future__ import annotations

import importlib.util
import os


def use_importable_matplotlib_backend() -> None:
    """Fall back to ``Agg`` when ``MPLBACKEND`` names a backend this environment cannot import.

    Nothing here draws to a screen, but matplotlib is still imported: BioEmu's inlined AlphaFold2
    imports TensorFlow, which imports Keras, which imports matplotlib. Notebook hosts such as Colab
    export ``MPLBACKEND=module://matplotlib_inline.backend_inline`` for their own kernel, and a
    project environment without that package makes matplotlib refuse at import -- killing the
    embedding step before it starts. A backend that does import is left alone.
    """
    backend = os.environ.get("MPLBACKEND", "")
    if not backend.startswith("module://"):
        return
    try:
        found = importlib.util.find_spec(backend.removeprefix("module://")) is not None
    except (ImportError, ValueError):
        found = False
    if not found:
        os.environ["MPLBACKEND"] = "Agg"
