"""Library entry point for the perfect-BIM full assembly pipeline.

Wraps ``scripts/merge_doors_and_windows_bim.py`` so the V3 production
pipeline can produce the same ``bim_complete_*.glb`` the merge CLI
currently writes. Expects ``run_door_placement`` to have run first
(the merge script reads ``bim.glb`` from ``out_dir``).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ._render_cache import RenderCache, install_render_cache
from ._script_loader import load_merge_module

logger = logging.getLogger(__name__)


def run_full_bim(
    pdf_path: Path,
    ann_path: Path,
    out_dir: Path,
    *,
    render_matrix: float = 4.0,
    render_cache: Optional[RenderCache] = None,
    enable_furniture: bool = True,
    furniture_classifier: Any = None,
    furniture_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute the merge_doors_and_windows_bim pipeline.

    Reads ``out_dir / "bim.glb"`` (produced by
    :func:`run_door_placement`), re-runs the wall extrusion, runs the
    window pipeline, appends bridges, post-processes glass materials,
    and writes ``bim_complete_<unix_ts>.glb`` plus
    ``bim_complete_viewer.html``.

    Args:
        pdf_path: source plan PDF.
        ann_path: ``annotations.json`` sidecar.
        out_dir: directory the script writes into. Must already
            contain ``bim.glb`` from :func:`run_door_placement`.
        render_matrix: PDF render scale.
        render_cache: optional per-job dict shared with
            :func:`run_door_placement` so identical
            ``(pdf_path, matrix)`` renders are reused instead of
            repeated by pdfium. This script calls
            ``_render_pdf_grayscale`` twice internally — both calls
            hit the cache, plus the one inherited from door-placement.

    Returns:
        dict with ``bim_complete_glb_path`` (Path) pointing at the
        newest ``bim_complete_*.glb`` in ``out_dir``.

    Raises:
        RuntimeError: if the script's ``main()`` exits non-zero or no
            ``bim_complete_*.glb`` is produced.
    """
    pdf_path = Path(pdf_path)
    ann_path = Path(ann_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bim_glb = out_dir / "bim.glb"
    if not bim_glb.exists():
        raise RuntimeError(
            f"run_full_bim() needs {bim_glb} to exist. "
            "Call run_door_placement() first."
        )

    merge = load_merge_module()
    merge.PDF_PATH = pdf_path
    merge.ANN_PATH = ann_path
    merge.OUT = out_dir
    merge.RENDER_MATRIX = float(render_matrix)

    try:
        with install_render_cache(merge, render_cache):
            merge.main()
    except SystemExit as exc:
        raise RuntimeError(
            f"merge_doors_and_windows_bim.main() exited: {exc}"
        ) from exc

    glbs = sorted(
        out_dir.glob("bim_complete_*.glb"),
        key=lambda p: p.stat().st_mtime,
    )
    if not glbs:
        raise RuntimeError(
            f"merge_doors_and_windows_bim.main() finished but no "
            f"bim_complete_*.glb appeared in {out_dir}"
        )

    result: Dict[str, Any] = {
        "bim_complete_glb_path": glbs[-1],
        "bim_complete_viewer_html_path": out_dir / "bim_complete_viewer.html",
    }

    # ── Moebel-Schritt: Items in die fertige BIM anhaengen (soft-fail) ──
    # Reuse der Merge-Modul-Funktionen (gleiche scale_xy + px_to_m) garantiert,
    # dass Moebel im selben Welt-Frame wie Waende/Tueren/Fenster landen.
    if enable_furniture and os.environ.get("P2B_FURNITURE_ENABLED", "1").strip().lower() not in ("0", "false", "no"):
        try:
            furn = _append_furniture(
                merge, pdf_path, ann_path, glbs[-1], render_matrix, render_cache,
                furniture_classifier, furniture_config,
            )
            if furn is not None:
                result["furniture"] = furn
        except Exception as exc:  # noqa: BLE001 — darf die BIM-Erzeugung nie brechen
            logger.warning("furniture step failed (soft): %s", exc)

    return result


def _append_furniture(
    merge,
    pdf_path: Path,
    ann_path: Path,
    glb_path: Path,
    render_matrix: float,
    render_cache: Optional[RenderCache],
    classifier: Any,
    config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Moebel klassifizieren (EU-LLM) + als Knoten an die BIM-GLB anhaengen.

    Gibt None zurueck, wenn kein LLM-Classifier konfiguriert ist (dann werden
    bewusst KEINE generischen Platzhalter in die Produktions-BIM geschrieben).
    """
    from planproof_geometry.ai.furniture import append_furniture_to_glb
    from planproof_geometry.ai.furniture.transport import resolve_default_classifier

    clf = classifier if classifier is not None else resolve_default_classifier(config)
    if clf is None:
        logger.info("furniture: kein EU-LLM-Classifier konfiguriert — Schritt uebersprungen")
        return None

    raw = json.loads(ann_path.read_text(encoding="utf-8"))
    annotations = raw.get("annotations") or []
    if not annotations:
        return None

    with install_render_cache(merge, render_cache):
        image_gray = merge._render_pdf_grayscale(pdf_path, render_matrix)
    h, w = image_gray.shape[:2]
    img_size = raw.get("image_size") or {}
    ann_w = int(img_size.get("width") or 0) or w
    ann_h = int(img_size.get("height") or 0) or h
    sx = w / ann_w
    sy = h / ann_h
    px_to_m = merge._calibrate_px_to_m(annotations, (sx, sy))
    if px_to_m is None:
        logger.info("furniture: px_to_m-Kalibrierung fehlgeschlagen — Schritt uebersprungen")
        return None

    result, added = append_furniture_to_glb(
        glb_path, annotations, image_gray,
        scale_xy=(sx, sy), px_to_m=px_to_m, classifier=clf,
    )
    logger.info(
        "furniture: %d Items in %d Boxen platziert, %d Knoten angehaengt",
        result.n_items, len(result.placements), added,
    )
    return {
        "n_items": result.n_items,
        "n_boxes": len(result.placements),
        "n_nodes_added": added,
        "semantics": result.to_semantics(),
        "metrics": result.metrics,
    }


def find_latest_bim_complete(out_dir: Path) -> Optional[Path]:
    """Helper: return the newest ``bim_complete_*.glb`` in ``out_dir``."""
    glbs = sorted(
        Path(out_dir).glob("bim_complete_*.glb"),
        key=lambda p: p.stat().st_mtime,
    )
    return glbs[-1] if glbs else None
