#!/usr/bin/env python3
"""Independent post-step that adds stair geometry to an existing BIM.

This script runs AFTER ``scripts/merge_doors_and_windows_bim.py`` has
finished. It does NOT modify the perfect_bim algorithm — instead it:

    1. Loads the existing ``bim_complete_*.glb`` (the perfect_bim
       output, doors + windows + bridges, fully assembled).
    2. Reads the same ``annotations.json`` + PDF the perfect_bim
       pipeline used.
    3. Runs the stair-extraction pipeline (``process_stairs``) and
       appends each stair mesh as a separate scene node.
    4. Writes ``bim_with_stairs_<unix_ts>.glb`` + a stair-specific
       viewer HTML next to the input.
    5. Writes ``stairs.json`` (record sidecar) and
       ``stairs_diagnostic.png`` (visual overlay) next to the GLB.

The perfect_bim algorithm is unchanged. If this script fails for any
reason, the original ``bim_complete_*.glb`` remains intact and the
user can still use the doors+windows BIM.

Constants follow the same conventions as
``scripts/merge_doors_and_windows_bim.py`` so the cloud-worker library
wrapper can monkey-patch the same way (see
``planproof_geometry.ai.v3.perfect_bim`` adapter package).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

OUT = Path("C:/tmp/debug_pipeline_out")
PDF_PATH = Path(
    "C:/Users/deepn/OneDrive/Desktop/Planproof_Projects/Test"
    "/Zeichnungen/Matterport Sample_BW.pdf"
)
ANN_PATH = Path(
    "C:/Users/deepn/OneDrive/Desktop/Planproof_Projects/Test"
    "/plan_bim_data/Matterport Sample_BW/annotations.json"
)
RENDER_MATRIX = 4.0


def _render_pdf_grayscale(pdf_path: Path, matrix: float) -> np.ndarray:
    # pdfium statt PyMuPDF (AGPL), 14.09.2026 - gleiche Groesse, siehe planproof_geometry.pdf.
    from planproof_geometry.pdf import PdfDatei

    with PdfDatei(pdf_path) as pdf:
        # Issue #47: die Seitengeometrie kommt AUS DER DATEI. Mit festem Faktor
        # 4,0 ergab eine praeparierte MediaBox von 10800 x 10800 pt ein Bild von
        # 43200 x 43200 Pixeln — und der Render-Cache haelt es fest, waehrend
        # jeder der drei Aufrufer eine .copy() bekommt.
        from ...pdf_render_budget import fit_zoom

        matrix = fit_zoom(*pdf.seitengroesse(0), matrix)
        return pdf.rendern_grau(0, matrix)


def _calibrate_px_to_m(annotations, scale_xy, typical_door_m: float = 0.85):
    """Identical contract to merge_doors_and_windows_bim._calibrate_px_to_m.

    Calibrates the metres-per-pixel scale via ``door_single``
    bbox widths (median, robust). The merge script uses the exact same
    logic — we mirror it so stair coordinates land in the same plan
    frame as the perfect_bim geometry.
    """
    sx, sy = scale_xy
    widths: List[float] = []
    for a in annotations:
        if (a.get("class") or "") != "door_single":
            continue
        bbox = a.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        widths.append(min(abs((x1 - x0) * sx), abs((y1 - y0) * sy)))
    if len(widths) < 3:
        return None
    widths.sort()
    median = widths[len(widths) // 2]
    if median <= 5:
        return None
    return typical_door_m / median


def _poly_from_annotation(ann, scale):
    """Identical contract to merge_doors_and_windows_bim._poly_from_annotation."""
    sx, sy = scale
    if ann.get("type") == "polygon":
        pts = ann.get("points") or []
        return [(float(p[0]) * sx, float(p[1]) * sy) for p in pts]
    bbox = ann.get("bbox")
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = bbox
        return [
            (x0 * sx, y0 * sy), (x1 * sx, y0 * sy),
            (x1 * sx, y1 * sy), (x0 * sx, y1 * sy),
        ]
    return []


def _latest_bim_complete_glb(out_dir: Path) -> Path | None:
    glbs = sorted(out_dir.glob("bim_complete_*.glb"), key=lambda p: p.stat().st_mtime)
    return glbs[-1] if glbs else None


def main() -> int:
    bim_complete_glb = _latest_bim_complete_glb(OUT)
    if bim_complete_glb is None:
        print(
            f"FAIL: no bim_complete_*.glb in {OUT} — "
            f"run perfect_bim.merge_doors_and_windows_bim first."
        )
        return 1
    print(f"Loading perfect_bim output: {bim_complete_glb}")

    import trimesh
    scene = trimesh.load(str(bim_complete_glb), force="scene")
    print(f"  scene: {len(scene.geometry)} geometries")

    print("\nResolving plan + annotations ...")
    image_gray = _render_pdf_grayscale(PDF_PATH, RENDER_MATRIX)
    H, W = image_gray.shape
    raw = json.loads(ANN_PATH.read_text(encoding="utf-8"))
    img_size = raw.get("image_size") or {}
    ann_w = int(img_size.get("width") or 0) or W
    ann_h = int(img_size.get("height") or 0) or H
    sx = W / ann_w
    sy = H / ann_h
    annotations = raw.get("annotations") or []

    px_to_m = _calibrate_px_to_m(annotations, (sx, sy))
    if px_to_m is None:
        print("FAIL: scale calibration failed (no door_swing annotations).")
        return 1
    print(f"  px_to_m = {px_to_m:.6f}")

    # Wall envelope is built from the user wall polygons (independent
    # path — we do NOT re-run the perfect_bim wall extrusion here, to
    # avoid coupling). For a stair to be considered "inside the
    # building" a convex hull of all wall annotations is enough.
    user_walls_px: List[List[Tuple[float, float]]] = []
    for a in annotations:
        cls = (a.get("class") or "").strip()
        if cls not in {"wall", "wall_interior", "wall_exterior"}:
            continue
        poly_px = _poly_from_annotation(a, (sx, sy))
        if len(poly_px) >= 3:
            user_walls_px.append(poly_px)
    user_walls_m = [
        [(x * px_to_m, y * px_to_m) for x, y in poly]
        for poly in user_walls_px
    ]
    print(f"  wall polygons (for envelope): {len(user_walls_m)}")

    from planproof_geometry.ai.v3.pixel_bim.stairs.diagnostic_renderer import (
        render_label_vs_bim_overlay,
        render_stairs_diagnostic,
    )
    from planproof_geometry.ai.v3.pixel_bim.stairs.pipeline import (
        process_stairs,
        to_semantics_dict_list,
    )

    stair_results = process_stairs(
        annotations=annotations,
        scale_xy=(sx, sy),
        px_to_m=px_to_m,
        image_gray=image_gray,
        bim_walls_m=user_walls_m,
    )
    # Used below — guard against import-stripping formatters.
    _label_vs_bim_render = render_label_vs_bim_overlay

    n_added = 0
    for sr in stair_results:
        if sr.mesh is None or sr.mesh.flight is None:
            continue
        node_name = f"stair/{sr.record.label_id}/flight"
        scene.add_geometry(sr.mesh.flight, node_name=node_name)
        n_added += 1
    n_skip = sum(1 for r in stair_results if r.mesh is None)
    print(
        f"  stair nodes added: {n_added} "
        f"(skipped {n_skip} with status records)"
    )

    # Sidecars: records + diagnostic PNG.
    stair_semantics: List[Dict[str, Any]] = to_semantics_dict_list(stair_results)
    try:
        (OUT / "stairs.json").write_text(
            json.dumps({"stairs": stair_semantics}, indent=2),
            encoding="utf-8",
        )
        print(f"  wrote {OUT / 'stairs.json'}")
    except OSError as exc:
        print(f"  stairs.json soft-failed: {exc}")

    if stair_results:
        # Rebuild bboxes by walking stair-class annotations in lockstep
        # with the results — same shape the diagnostic renderer expects.
        stair_ann_iter = (
            a for a in annotations
            if (a.get("class") or "") in {"stairs", "stairs_spiral", "elevator"}
        )
        bboxes_px: list = []
        for _r in stair_results:
            try:
                ann = next(stair_ann_iter)
            except StopIteration:
                bboxes_px.append(None)
                continue
            poly_px = _poly_from_annotation(ann, (sx, sy))
            if not poly_px:
                bboxes_px.append(None)
                continue
            xs = [p[0] for p in poly_px]
            ys = [p[1] for p in poly_px]
            bboxes_px.append((min(xs), min(ys), max(xs), max(ys)))

        try:
            diag_path = OUT / "stairs_diagnostic.png"
            render_stairs_diagnostic(
                image_gray=image_gray,
                px_to_m=px_to_m,
                results=stair_results,
                bboxes_px=bboxes_px,
                output_path=diag_path,
            )
            print(f"  wrote {diag_path}")
        except Exception as exc:
            print(f"  stairs_diagnostic.png soft-failed: {exc}")

        try:
            overlay_path = OUT / "stairs_label_vs_bim.png"
            _label_vs_bim_render(
                image_gray=image_gray,
                px_to_m=px_to_m,
                results=stair_results,
                output_path=overlay_path,
            )
            print(f"  wrote {overlay_path}")
        except Exception as exc:
            print(f"  stairs_label_vs_bim.png soft-failed: {exc}")

    # Write the BIM-with-stairs GLB next to the original. We do NOT
    # overwrite bim_complete_*.glb because the perfect_bim algorithm
    # must remain authoritative; the new file is a layered output.
    import time as _t
    cache_tag = str(int(_t.time()))
    out_glb = OUT / f"bim_with_stairs_{cache_tag}.glb"
    scene.export(str(out_glb))
    print(f"\nWrote {out_glb}")

    # Drop older bim_with_stairs_*.glb (keep only the newest).
    for old in OUT.glob("bim_with_stairs_*.glb"):
        if old.name != out_glb.name:
            try:
                old.unlink()
            except OSError:
                pass

    # Layered viewer pointing at the new GLB. We keep the original
    # bim_complete_viewer.html untouched.
    viewer_html = OUT / "bim_with_stairs_viewer.html"
    cache_bust = cache_tag
    glb_url = f"{out_glb.name}?v={cache_bust}"
    viewer_html.write_text(
        f"""<!doctype html>
<html><head>
<title>Plan-to-BIM — Doors + Windows + Stairs</title>
<script type="module"
  src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
<style>
  html,body {{ margin:0; padding:0; height:100%; background:#1c1c20; color:#eee;
              font-family: system-ui, -apple-system, "Segoe UI"; }}
  .info {{ position:fixed; top:8px; left:8px; padding:8px 12px;
           background:rgba(0,0,0,0.55); border-radius:6px; font-size:13px;
           line-height:1.5; pointer-events:none; }}
  model-viewer {{ width:100vw; height:100vh; --poster-color:#1c1c20; }}
</style>
</head>
<body>
<model-viewer src="{glb_url}" camera-controls touch-action="pan-y"
   alt="Planproof BIM with stairs"
   style="background:#d8dbe0"></model-viewer>
<div class="info">
  <strong>Plan-to-BIM — mit Treppen</strong><br>
  Walls + Doors + Windows: unverändert vom perfect_bim Pipeline<br>
  Stairs: {n_added} flight(s) added, {n_skip} skipped<br>
  build={cache_bust}
</div>
</body></html>""",
        encoding="utf-8",
    )
    print(f"Wrote {viewer_html}")

    print()
    print("Zum Anzeigen im Browser:")
    print(f"  cd {OUT}")
    print("  python -m http.server 8765")
    print(f"  http://127.0.0.1:8765/{viewer_html.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
