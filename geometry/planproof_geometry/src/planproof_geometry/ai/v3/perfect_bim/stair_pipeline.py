"""Library entry point for the independent stairs post-step.

Wraps ``scripts/add_stairs_to_bim.py`` so the production pipeline can
call its ``main()`` flow with explicit paths instead of the script's
hardcoded constants. The script itself is not modified.

The stair post-step runs AFTER :func:`run_full_bim` has produced the
``bim_complete_*.glb``. It is independent — the perfect_bim algorithm
(doors + windows + bridges + glass post-process) remains untouched and
is the source of truth. The stair step loads the existing GLB, appends
stair meshes as additional scene nodes, and writes a layered
``bim_with_stairs_<unix_ts>.glb`` next to it.

Output layout in ``out_dir`` (in addition to whatever the merge script
already produced):

* ``bim_with_stairs_<ts>.glb`` — GLB with stair geometry layered on top.
* ``bim_with_stairs_viewer.html`` — standalone viewer referencing the
  layered GLB.
* ``stairs.json`` — per-stair records (status, footprint, layout).
* ``stairs_diagnostic.png`` — top-down overlay diagnostic.

Soft-fail by convention: callers should treat any exception here as a
non-fatal warning so the perfect_bim bundle stays usable when stair
extraction is impossible (e.g. no stair annotations, calibration
failure).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ._script_loader import load_stairs_module


def run_add_stairs(
    pdf_path: Path,
    ann_path: Path,
    out_dir: Path,
    *,
    render_matrix: float = 4.0,
) -> Dict[str, Any]:
    """Execute the add-stairs-to-BIM post-step on a PDF + annotations.

    Args:
        pdf_path: source plan PDF (single page).
        ann_path: ``annotations.json`` sidecar.
        out_dir: directory containing the existing
            ``bim_complete_*.glb`` produced by :func:`run_full_bim`.
            Layered outputs are written into the same directory.
        render_matrix: PDF render scale (4× = production default).

    Returns:
        dict with ``bim_with_stairs_glb_path`` (Path) if the script
        produced one, else an empty dict. The empty case is normal
        (e.g. no stair annotations in the plan).

    Raises:
        RuntimeError: if the script's ``main()`` returns a non-zero
            exit code. Callers may catch and continue (soft-fail).
    """
    pdf_path = Path(pdf_path)
    ann_path = Path(ann_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stairs = load_stairs_module()
    stairs.PDF_PATH = pdf_path
    stairs.ANN_PATH = ann_path
    stairs.OUT = out_dir
    stairs.RENDER_MATRIX = float(render_matrix)

    rc = stairs.main()
    if rc != 0:
        raise RuntimeError(f"add_stairs_to_bim.main() returned {rc}")

    layered = sorted(
        out_dir.glob("bim_with_stairs_*.glb"),
        key=lambda p: p.stat().st_mtime,
    )
    if not layered:
        return {}
    return {
        "bim_with_stairs_glb_path": layered[-1],
        "bim_with_stairs_viewer_html_path": out_dir / "bim_with_stairs_viewer.html",
        "stairs_json_path": out_dir / "stairs.json",
        "stairs_diagnostic_png_path": out_dir / "stairs_diagnostic.png",
    }


def find_latest_bim_with_stairs(out_dir: Path) -> Optional[Path]:
    """Helper: return the newest ``bim_with_stairs_*.glb`` in ``out_dir``."""
    layered = sorted(
        Path(out_dir).glob("bim_with_stairs_*.glb"),
        key=lambda p: p.stat().st_mtime,
    )
    return layered[-1] if layered else None
