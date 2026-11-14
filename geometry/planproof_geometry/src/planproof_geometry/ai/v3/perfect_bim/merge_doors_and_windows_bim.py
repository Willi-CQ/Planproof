#!/usr/bin/env python3
"""Merge advanced door BIM (bim.glb from diagnose_door_placement.py) with the
new window pipeline output and emit a unified BIM viewer.

Pipeline:
  1. Load ``bim.glb`` from ``C:/tmp/debug_pipeline_out/`` (must already exist —
     run ``scripts/diagnose_door_placement.py`` first).
  2. Run the new window pipeline (find_window_blocks -> clip_blocks_to_walls
     -> place_windows_in_blocks -> build_window_meshes) on the same plan +
     annotations.
  3. Append every window mesh as separate scene node to the door GLB.
  4. Write ``bim_complete.glb`` and ``bim_complete_viewer.html``.
  5. Print HTTP-server commands so the user can serve and view.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

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


def main() -> int:
    bim_glb = OUT / "bim.glb"
    if not bim_glb.exists():
        print(f"FAIL: {bim_glb} missing — run perfect_bim.diagnose_door_placement first.")
        return 1

    print(f"Loading door BIM: {bim_glb}")
    import trimesh
    door_scene = trimesh.load(str(bim_glb), force="scene")
    print(f"  door scene: {len(door_scene.geometry)} geometries")

    print("\nResolving plan + annotations…")
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
        print("FAIL: scale calibration failed")
        return 1
    print(f"  px_to_m = {px_to_m:.6f}")

    # Phase 1+2: refine + extrude um die FERTIGEN BIM-Waende zu bekommen.
    # Algorithmen rechnen gegen diese Polygone, nicht gegen User-Labels.
    user_walls_px: List[List[Tuple[float, float]]] = []
    for a in annotations:
        cls = a.get("class") or ""
        if cls not in {"wall", "wall_interior", "wall_exterior"}:
            continue
        poly_px = _poly_from_annotation(a, (sx, sy))
        if len(poly_px) >= 3:
            user_walls_px.append(poly_px)

    door_polys_for_subtraction = [
        _poly_from_annotation(a, (sx, sy))
        for a in annotations
        if (a.get("class") or "") == "door_sliding"
        and len(_poly_from_annotation(a, (sx, sy))) >= 3
    ]

    print("  running refine_walls + extrude_wall_mask ...")
    import cv2
    from planproof_geometry.ai.v3.pixel_bim.refine.wall_dispatcher import refine_walls
    from planproof_geometry.ai.v3.pixel_bim.extrude import extrude_wall_mask

    image_gray = _render_pdf_grayscale(PDF_PATH, RENDER_MATRIX)
    H_img, W_img = image_gray.shape
    wall_mask, dispatch_info = refine_walls(
        image_gray, user_walls_px, px_to_m=px_to_m,
        door_polygons_px=door_polys_for_subtraction or None,
    )
    corrected = dispatch_info.get("wall_polygons_corrected_px")
    if corrected is not None:
        user_walls_px = corrected
    user_wall_union_mask = np.zeros((H_img, W_img), dtype=np.uint8)
    for poly in user_walls_px:
        if len(poly) < 3:
            continue
        pts = np.array(
            [[int(round(x)), int(round(y))] for x, y in poly], dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.fillPoly(user_wall_union_mask, [pts], 255)
    extrude_result = extrude_wall_mask(
        wall_mask, px_to_m=px_to_m, wall_height_m=2.80,
        simplify_m=0.005, min_polygon_area_m2=0.005,
        image_gray=image_gray,
        user_polygon_union_mask=user_wall_union_mask,
    )
    bim_walls_m = [list(wp.shell_xy) for wp in extrude_result.polygons]
    print(f"  BIM walls (post-extrude): {len(bim_walls_m)}")

    from planproof_geometry.ai.v3.pixel_bim.openings.window_label_blocks import (
        find_window_blocks,
    )
    from planproof_geometry.ai.v3.pixel_bim.openings.window_block_clipper import (
        clip_blocks_to_walls,
    )
    from planproof_geometry.ai.v3.pixel_bim.openings.window_placement import (
        place_windows_in_blocks,
        verify_placement_criterion_1,
    )
    from planproof_geometry.ai.v3.pixel_bim.openings.window_mesh_builder import (
        build_window_meshes,
    )

    blocks = find_window_blocks(
        annotations, scale_xy=(sx, sy), px_to_m=px_to_m,
    )
    clipped = clip_blocks_to_walls(blocks, bim_walls_m)
    placed = place_windows_in_blocks(clipped)
    crit = verify_placement_criterion_1(clipped, placed)
    n_snapped = sum(
        1 for cb in clipped
        for n in cb.notes
        if n.startswith("snapped_")
    )
    print(f"\nWindows: {len(blocks)} blocks -> {len(placed)} placed  "
          f"|  criterion={'PASS' if crit.passed else 'FAIL'}  "
          f"|  snapped notes: {n_snapped}")

    # The diagnose script writes the door BIM in plan-world-coords (no
    # centring). Our window meshes use the same plan coords (PDF-Y flipped
    # to scene-Z in the mesh builder). They align directly; no translation
    # needed. Print the scene bounds for sanity.
    scene_bmin = np.array([np.inf, np.inf, np.inf])
    scene_bmax = np.array([-np.inf, -np.inf, -np.inf])
    for name in door_scene.graph.nodes_geometry:
        transform, geom = door_scene.graph[name]
        m = door_scene.geometry[geom].copy()
        m.apply_transform(transform)
        b0, b1 = m.bounds
        scene_bmin = np.minimum(scene_bmin, b0)
        scene_bmax = np.maximum(scene_bmax, b1)
    print(
        f"  door scene bounds: "
        f"X=[{scene_bmin[0]:.2f}..{scene_bmax[0]:.2f}] "
        f"Y=[{scene_bmin[1]:.2f}..{scene_bmax[1]:.2f}] "
        f"Z=[{scene_bmin[2]:.2f}..{scene_bmax[2]:.2f}]"
    )

    n_window_nodes_added = 0
    win_bmin = np.array([np.inf, np.inf, np.inf])
    win_bmax = np.array([-np.inf, -np.inf, -np.inf])
    for w in placed:
        meshes = build_window_meshes(w, wall_height_m=2.80)
        tag = w.label_class
        for component_name, mesh in (
            ("parapet", meshes.parapet),
            ("lintel", meshes.lintel),
            ("frame", meshes.frame),
            ("glass", meshes.fill),
        ):
            if mesh is None or len(mesh.vertices) == 0:
                continue
            b0, b1 = mesh.bounds
            win_bmin = np.minimum(win_bmin, b0)
            win_bmax = np.maximum(win_bmax, b1)
            node_name = f"{tag}/{w.block_idx}/{w.sub_index}/{component_name}"
            door_scene.add_geometry(mesh, node_name=node_name)
            n_window_nodes_added += 1
    print(f"  window nodes added: {n_window_nodes_added}")
    print(
        f"  window meshes bounds: "
        f"X=[{win_bmin[0]:.2f}..{win_bmax[0]:.2f}] "
        f"Y=[{win_bmin[1]:.2f}..{win_bmax[1]:.2f}] "
        f"Z=[{win_bmin[2]:.2f}..{win_bmax[2]:.2f}]"
    )

    # ── Door bridges: lintel ueber dem Tuer-Rahmen, EXAKT Tuer-Rahmen-
    # Breite (nicht breiter), exakt lokale Wand-Dicke, mit minimalem
    # Fusion-Overlap an den Wand-Stuempfen.
    n_door_bridges = _add_door_bridges(
        door_scene=door_scene,
        annotations=annotations,
        scale_xy=(sx, sy),
        px_to_m=px_to_m,
        bim_walls_m=bim_walls_m,
    )
    print(f"  door bridge nodes added: {n_door_bridges}")

    import time as _t
    cache_tag = str(int(_t.time()))
    out_glb = OUT / f"bim_complete_{cache_tag}.glb"
    door_scene.export(str(out_glb))
    print(f"\nWrote {out_glb}")
    _post_process_glb_for_glass(out_glb)
    # Cleanup alter Builds (nur die mit timestamp-Pattern)
    for old in OUT.glob("bim_complete_*.glb"):
        if old.name != out_glb.name:
            try:
                old.unlink()
            except Exception:
                pass

    import time as _time
    cache_bust = str(int(_time.time()))
    viewer_html = OUT / "bim_complete_viewer.html"
    glb_url = f"{out_glb.name}?v={cache_bust}"
    viewer_html.write_text(
        f"""<!doctype html>
<html><head>
<title>Plan-to-BIM - Doors + Windows (complete)</title>
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
   alt="Planproof BIM"
   style="background:#d8dbe0"></model-viewer>
<div class="info">
  <strong>Plan-to-BIM - vollstaendiger Stack</strong><br>
  Walls (Phase 1+2 mit User-Polygon-Orthogonalisierung)<br>
  Doors: SSOT-Pipeline + Mount-Classifier<br>
  Windows: Hartes Kriterium #1 PASS - Frame + Mullion + Wand-Snap<br>
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


def _post_process_glb_for_glass(glb_path: Path) -> None:
    """Wandle Materials mit baseColor-Alpha < 0.95 in BLEND-Glas mit
    ``KHR_materials_transmission`` um."""
    import struct
    raw = glb_path.read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        return
    _, _, total_len = struct.unpack("<III", raw[:12])
    json_len, json_tag = struct.unpack("<II", raw[12:20])
    if json_tag != 0x4E4F534A:  # "JSON"
        return
    gltf_bytes = raw[20:20 + json_len]
    rest = raw[20 + json_len:]
    try:
        gltf = json.loads(gltf_bytes.decode("utf-8"))
    except Exception:
        return

    mats = gltf.get("materials", [])
    if not mats:
        return
    used_ext = set(gltf.get("extensionsUsed", []))
    changed = False
    for mat in mats:
        pbr = mat.get("pbrMetallicRoughness") or {}
        base = pbr.get("baseColorFactor") or [1, 1, 1, 1]
        if len(base) < 4 or base[3] >= 0.95:
            continue
        mat.setdefault("extensions", {})["KHR_materials_transmission"] = {
            "transmissionFactor": 0.30,
        }
        # baseColor alpha auf 1.0 — Transparenz kommt aus Transmission;
        # alphaMode BLEND damit der Tint sichtbar bleibt.
        pbr["baseColorFactor"] = [base[0], base[1], base[2], 1.0]
        pbr["roughnessFactor"] = 0.05
        pbr["metallicFactor"] = 0.0
        mat["pbrMetallicRoughness"] = pbr
        mat["alphaMode"] = "BLEND"
        mat["doubleSided"] = True
        used_ext.add("KHR_materials_transmission")
        changed = True
    if not changed:
        return
    gltf["extensionsUsed"] = sorted(used_ext)
    new_json = json.dumps(gltf).encode("utf-8")
    # Padding auf 4 Byte-Grenze
    pad_len = (4 - (len(new_json) % 4)) % 4
    new_json = new_json + b" " * pad_len
    new_json_len = len(new_json)
    new_total = 12 + 8 + new_json_len + len(rest)
    header = struct.pack("<III", 0x46546C67, 2, new_total)
    json_chunk_header = struct.pack("<II", new_json_len, 0x4E4F534A)
    glb_path.write_bytes(header + json_chunk_header + new_json + rest)
    print(f"  postprocessed glass materials in {glb_path.name}")


# Sliding-Doors bekommen KEINE Bridge — die Tuer-Geometrie ist von der
# Wand entkoppelt (Track + Panel beside/in_wall) und der Lintel-Bereich
# wird in der Nachbearbeitung manuell behandelt.
_DOOR_CLASSES_WITH_BRIDGE = {"door", "door_single", "door_double"}
# Standard DIN-Tuerhoehe (DIN 18101): Sturz-Unterkante 2.01 m.
_DOOR_HEAD_M = 2.01
# Tuer-Rahmen-Oberkante: das ``door_pipeline`` baut den Frame bis ca.
# 2.105 m (DUK + Stockprofil-Hoehe). Die Bridge muss ABOVE this start,
# sonst ueberlappt sie den Tuer-Rahmen optisch.
_DOOR_FRAME_TOP_M = 2.110
_WALL_HEIGHT_M = 2.80
# Tangentiales Overlap der Bridge in die linken/rechten Wand-Stuempfe.
# Der Bridge-Lintel reicht ueber das Door-Rechteck hinaus IN die Wand
# hinein, sodass die zwei Wand-Seiten links und rechts der Tuer durch
# die Bridge ueber dem Sturz miteinander verbunden sind (Willi 2026-
# 05-07: "die bridges sollen die beiden Seiten miteinander verbinden").
# 10 cm reicht weit genug ins Wand-Material rein, dass die Verbindung
# im Render eindeutig sichtbar ist, und ist klein genug, dass keine
# Phantom-Wand entsteht wenn der Wand-Stumpf kuerzer als 10 cm ist.
_BRIDGE_WALL_OVERLAP_M = 0.10


def _add_door_bridges(
    *,
    door_scene,
    annotations,
    scale_xy,
    px_to_m: float,
    bim_walls_m,
) -> int:
    """Pro Swing-/Double-Tuer: Lintel-Bridge ueber dem Tuer-Rahmen, die
    die Wand-Stuempfe LINKS und RECHTS der Tuer ueber dem Sturz
    durchgaengig miteinander verbindet.

    Sliding-Doors werden uebersprungen (Track + Panel sind von der Wand
    entkoppelt; der Lintel-Bereich wird manuell nachbearbeitet).

    Algorithmus:
      1. Sammle alle Frame-Top-Dreiecke (alle 3 Vertices in
         ``[2.05, 2.115]`` m) aus dem ``door_scene`` — das sind die
         Jambs + Head-Trim der bereits platzierten Tueren.
      2. Pro Swing-/Double-Tuer-Annotation: filtere die Frame-Punkte in
         der Annotation-Bbox + 0.10 m.
      3. Bestimme die Tangenten-/Normalen-Achse des Frame-Footprints via
         PCA der Punkte.
      4. Projiziere die Punkte auf (tangent, normal) → Frame-Extent
         ``[s_lo, s_hi] × [n_lo, n_hi]``.
      5. Extrudiere ein Rechteck ``[s_lo - overlap, s_hi + overlap]
         × [n_lo, n_hi]`` von ``_DOOR_FRAME_TOP_M`` bis ``_WALL_HEIGHT_M``.
         Das Tangent-Overlap ``_BRIDGE_WALL_OVERLAP_M`` reicht in die
         Wand-Stuempfe hinein und macht die linke/rechte Verbindung
         visuell durchgehend.
    """
    import numpy as np
    from planproof_geometry.ai.v3.pixel_bim.openings.window_mesh_builder import (
        _extrude_polygon_in_plan,
        _polygon_from_axis,
    )

    # 1. Sammle alle Frame-Top-Vertices aus dem door_scene.
    frame_y_lo = _DOOR_FRAME_TOP_M - 0.06    # ~2.05 m
    frame_y_hi = _DOOR_FRAME_TOP_M + 0.005   # ~2.115 m
    frame_pts_xz_chunks = []
    for geom in door_scene.geometry.values():
        verts = np.asarray(geom.vertices, dtype=np.float64)
        faces = np.asarray(geom.faces, dtype=np.int64)
        if len(verts) == 0 or len(faces) == 0:
            continue
        ys = verts[faces][:, :, 1]
        mask = np.all((ys >= frame_y_lo) & (ys <= frame_y_hi), axis=1)
        if not mask.any():
            continue
        frame_pts_xz_chunks.append(verts[faces[mask].flatten()][:, [0, 2]])
    if not frame_pts_xz_chunks:
        return 0
    all_frame_xz = np.concatenate(frame_pts_xz_chunks, axis=0)

    sx, sy = scale_xy
    # Wand-Vertex-Farbe (220/220/220) aus dem Door-BIM. Bridge bekommt
    # face_colors statt PBR-Material — genau die gleiche Render-Pfad-
    # Klasse wie das Wand-Mesh (kein PBR), damit das Shading exakt
    # uebereinstimmt.
    lintel_rgba = (220, 220, 220, 255)
    n_added = 0
    for d_idx, ann in enumerate(annotations):
        cls = ann.get("class") or ""
        if cls not in _DOOR_CLASSES_WITH_BRIDGE:
            continue
        bbox_m = _ann_bbox_m(ann, sx, sy, px_to_m)
        if bbox_m is None:
            continue
        x0, y0, x1, y1 = bbox_m

        # 2. Frame-Punkte in der Tuer-Bbox + Margin filtern.
        # GLB-Konvention (mesh_builder.py): scene_Z = +plan_Y.
        margin = 0.10
        in_door = (
            (all_frame_xz[:, 0] >= x0 - margin)
            & (all_frame_xz[:, 0] <= x1 + margin)
            & (all_frame_xz[:, 1] >= y0 - margin)
            & (all_frame_xz[:, 1] <= y1 + margin)
        )
        frame_pts = all_frame_xz[in_door]
        if len(frame_pts) < 4:
            continue  # kein Frame in der Bbox → Tuer nicht platziert

        # 3. Frame-Achse via PCA: Tangent = lange Achse (Wand-Richtung),
        #    Normal = kurze Achse (Wand-Dicke).
        centroid = frame_pts.mean(axis=0)
        centered = frame_pts - centroid
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # eigh sortiert aufsteigend → letzte Spalte = groesster Eigenwert.
        tangent = eigvecs[:, -1]
        normal = eigvecs[:, 0]
        tangent = tangent / max(1e-9, np.linalg.norm(tangent))
        normal = normal / max(1e-9, np.linalg.norm(normal))

        # 4. Frame-Extent in (tangent, normal)-Frame.
        s_vals = centered @ tangent
        n_vals = centered @ normal
        s_lo, s_hi = float(s_vals.min()), float(s_vals.max())
        n_lo, n_hi = float(n_vals.min()), float(n_vals.max())

        # 5. Bridge-Polygon in Scene-XZ: Tangent-Extent erweitern um
        #    _BRIDGE_WALL_OVERLAP_M auf jeder Seite, Normal-Extent =
        #    Wand-Dicke (unveraendert).
        s_lo_ext = s_lo - _BRIDGE_WALL_OVERLAP_M
        s_hi_ext = s_hi + _BRIDGE_WALL_OVERLAP_M
        bridge_xz = _polygon_from_axis(
            s_lo_ext, s_hi_ext, n_lo, n_hi,
            (float(centroid[0]), float(centroid[1])),
            (float(tangent[0]), float(tangent[1])),
            (float(normal[0]), float(normal[1])),
        )
        # Scene-XZ → plan-XY (scene_Z = +plan_Y → plan_Y = scene_Z).
        plan_coords = [(float(c[0]), float(c[1])) for c in bridge_xz]

        lintel_h = _WALL_HEIGHT_M - _DOOR_FRAME_TOP_M
        if lintel_h <= 1e-3:
            continue
        mesh = _extrude_polygon_in_plan(
            plan_coords, base_y_m=_DOOR_FRAME_TOP_M, height_y_m=lintel_h,
            flip_y_for_pdf=True,
        )
        if mesh is None or len(mesh.vertices) == 0:
            continue
        mesh.visual.face_colors = lintel_rgba
        node_name = f"{cls}/bridge/{d_idx}/lintel"
        door_scene.add_geometry(mesh, node_name=node_name)
        n_added += 1
    return n_added


def _ann_bbox_m(ann, sx, sy, px_to_m):
    if ann.get("type") == "polygon":
        pts = ann.get("points") or []
        if not pts:
            return None
        xs = [float(p[0]) * sx * px_to_m for p in pts]
        ys = [float(p[1]) * sy * px_to_m for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))
    bbox = ann.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = bbox
    xs = sorted([float(x0) * sx * px_to_m, float(x1) * sx * px_to_m])
    ys = sorted([float(y0) * sy * px_to_m, float(y1) * sy * px_to_m])
    return (xs[0], ys[0], xs[1], ys[1])


if __name__ == "__main__":
    sys.exit(main())
