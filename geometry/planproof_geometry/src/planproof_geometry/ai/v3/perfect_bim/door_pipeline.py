"""Library entry point for the perfect-BIM door-placement pipeline.

Wraps ``scripts/diagnose_door_placement.py`` so production code can call
its ``main()`` flow with explicit paths instead of the script's
hardcoded constants. The script itself is not modified.

Output layout in ``out_dir`` (matches the script's CLI behaviour):

* ``bim.glb``            — door scene (swing + sliding geometry)
* ``bim_viewer.html``    — standalone door viewer
* ``bim_walls_and_bridges.png`` + verify PNGs (diagnostic)

Returns a dict with the artifact paths so the caller can chain into
:func:`run_full_bim`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ._render_cache import RenderCache, install_render_cache
from ._script_loader import load_diagnose_module


def run_door_placement(
    pdf_path: Path,
    ann_path: Path,
    out_dir: Path,
    *,
    render_matrix: float = 4.0,
    wall_height_m: float = 2.80,
    render_cache: Optional[RenderCache] = None,
) -> Dict[str, Any]:
    """Execute the diagnose door-placement pipeline on a PDF + annotations.

    Args:
        pdf_path: source plan PDF (single page).
        ann_path: ``annotations.json`` sidecar produced by the desktop
            label editor.
        out_dir: directory the script writes artifacts into. Will be
            created if missing. Existing files may be overwritten.
        render_matrix: PDF render scale (4× = production default).
        wall_height_m: floor-to-floor height for the BIM mesh.
        render_cache: optional per-job dict shared with :func:`run_full_bim`
            so identical ``(pdf_path, matrix)`` renders are reused instead
            of repeated by pdfium. The cached array is copied on every
            return so callers cannot leak in-place CV2 mutations back
            into the cache.

    Returns:
        dict with at least ``bim_glb_path`` (Path) pointing at the
        produced ``bim.glb``.

    Raises:
        RuntimeError: if the script's ``main()`` exits non-zero or the
            expected ``bim.glb`` does not appear in ``out_dir``.
    """
    pdf_path = Path(pdf_path)
    ann_path = Path(ann_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diag = load_diagnose_module()
    # Patch the script's module-level constants so its main() points at
    # *our* job paths and not the developer's hardcoded sample.
    diag.PDF_PATH = pdf_path
    diag.ANN_PATH = ann_path
    diag.OUT = out_dir
    diag.RENDER_MATRIX = float(render_matrix)
    diag.WALL_HEIGHT_M = float(wall_height_m)

    try:
        with install_render_cache(diag, render_cache):
            diag.main()
    except SystemExit as exc:
        # The script raises SystemExit on calibration failure; surface as
        # a normal exception so the cloud worker fails the job cleanly.
        raise RuntimeError(
            f"diagnose_door_placement.main() exited: {exc}"
        ) from exc

    bim_glb = out_dir / "bim.glb"
    if not bim_glb.exists():
        raise RuntimeError(
            f"diagnose_door_placement.main() finished but {bim_glb} is missing"
        )
    return {
        "bim_glb_path": bim_glb,
        "bim_viewer_html_path": out_dir / "bim_viewer.html",
    }
