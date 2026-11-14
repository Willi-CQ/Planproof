"""Per-Job PDF-Render-Cache for the perfect-BIM scripts.

The two scripts ``diagnose_door_placement.py`` and
``merge_doors_and_windows_bim.py`` each call
``_render_pdf_grayscale(PDF_PATH, RENDER_MATRIX)`` with the *same*
arguments — once in the door pipeline and twice inside the full-BIM
merge. With ``RENDER_MATRIX=4.0`` on a typical plan that is three
renderings of an ~8000x6000 grayscale page, identical bytes each time.

This module offers ``install_render_cache(module, cache)``: a
context-manager-style helper that monkey-patches the script module's
``_render_pdf_grayscale`` symbol with a cached wrapper for the
duration of a ``main()`` invocation, then restores the original.

The cache is **per job** (the orchestrator owns the dict) so cached
arrays cannot leak between independent worker runs.

Safety notes
------------
* The wrapper returns a fresh ``.copy()`` on every call — identical
  to the original function — so callers cannot mutate the cached
  array through aliasing.
* Cache key is ``(absolute_pdf_path, matrix)`` so unrelated PDFs or
  a (currently never used) second render scale do not collide.
* The patch is installed *and removed* by the wrapper functions
  (``run_door_placement`` / ``run_full_bim``); if ``main()`` raises,
  the original symbol is still restored via ``finally``.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, Tuple

import numpy as np


CacheKey = Tuple[str, float]
RenderCache = Dict[CacheKey, np.ndarray]


def _build_cached_render(
    original: Callable[..., np.ndarray],
    cache: RenderCache,
) -> Callable[..., np.ndarray]:
    """Wrap ``_render_pdf_grayscale`` with a job-local memoization layer."""

    def cached_render(pdf_path: Any, matrix: Any) -> np.ndarray:
        # Match the original signature: positional (pdf_path, matrix). The
        # scripts always call it positionally, but be defensive against
        # future refactors that might pass kwargs.
        key: CacheKey = (str(Path(pdf_path).resolve()), float(matrix))
        arr = cache.get(key)
        if arr is None:
            arr = original(pdf_path, matrix)
            cache[key] = arr
        # Always hand the caller a fresh copy so any in-place CV2 op on
        # the returned image cannot leak back into the cached buffer.
        return arr.copy()

    return cached_render


@contextlib.contextmanager
def install_render_cache(
    module: ModuleType,
    cache: RenderCache | None,
) -> Iterator[None]:
    """Install the cached renderer on ``module`` for the duration of a block.

    No-op when ``cache`` is ``None`` so callers can opt out without
    branching at the call site.
    """
    if cache is None:
        yield
        return

    original = getattr(module, "_render_pdf_grayscale", None)
    if original is None:
        # Defensive: if the script ever renames the function the patch
        # must not silently swallow the call.
        yield
        return

    module._render_pdf_grayscale = _build_cached_render(original, cache)
    try:
        yield
    finally:
        module._render_pdf_grayscale = original
