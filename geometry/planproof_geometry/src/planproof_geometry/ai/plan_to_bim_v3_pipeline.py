"""V3 Pixel-BIM pipeline orchestrator (Plan-to-BIM, walls + openings).

Replaces the V2 vector-transfer pipeline as the production default.
Approach:
    1. Render PDF at high resolution (RENDER_MATRIX × native).
    2. Phase 1 — classical refine of each user wall annotation.
    3. Phase 2 — mask → polygon → extruded 3D wall mesh.
    4. Phase 3 — place opening meshes in the gaps between flanking walls.
    5. Export glTF + semantics_v3.json + viewer.html.
    6. Walls-preserve guardrail (mesh vertex hash).
    7. Bundle artefacts into ``<output_dir>/input_0_solution.zip``.

Calls into ``planproof_geometry.ai.v3.pixel_bim`` for all work.
"""
from __future__ import annotations

import json
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from planproof_geometry.pipeline.core import PipelineResult, StepResult


WALL_CLASSES = {"wall", "wall_interior", "wall_exterior"}
OPENING_CLASSES = {
    "door", "door_single", "door_double", "door_sliding",
}
WINDOW_LABEL_CLASSES = {"window", "window_large"}
_IFC_MAP = {
    "door":         ("IfcDoor", "DOOR", "SINGLE_SWING_LEFT"),
    "door_single":  ("IfcDoor", "DOOR", "SINGLE_SWING_LEFT"),
    "door_double":  ("IfcDoor", "DOOR", "DOUBLE_SWING_LEFT"),
    "door_sliding": ("IfcDoor", "DOOR", "SLIDING_TO_LEFT"),
    "window":       ("IfcWindow", "WINDOW", None),
    "window_large": ("IfcWindow", "WINDOW", None),
}


@dataclass
class V3State:
    """Mutable state passed through the V3 pipeline.

    Carries the same shape the V2 pipeline used so cloud_entry.py and the
    worker can read .metrics, .model_path, .package_path uniformly.
    """
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None
    package_path: Optional[str] = None


def _render_pdf_grayscale(pdf_path: Path, matrix: float):
    # pdfium statt PyMuPDF (AGPL), 14.09.2026 - gleiche Groesse, siehe planproof_geometry.pdf.
    from planproof_geometry.pdf import PdfDatei  # noqa: PLC0415

    with PdfDatei(pdf_path) as pdf:
        # Issue #47: die Seitengeometrie kommt AUS DER DATEI. Mit festem Faktor
        # 4,0 ergab eine praeparierte MediaBox von 10800 x 10800 pt ein Bild von
        # 43200 x 43200 Pixeln — und der Render-Cache haelt es fest, waehrend
        # jeder der drei Aufrufer eine .copy() bekommt.
        from .pdf_render_budget import fit_zoom

        matrix = fit_zoom(*pdf.seitengroesse(0), matrix)
        return pdf.rendern_grau(0, matrix)


def _calibrate_px_to_m(annotations, scale_xy, typical_door_m: float = 0.85):
    sx, sy = scale_xy
    widths = []
    for a in annotations:
        if (a.get("class") or "") != "door_single":
            continue
        bbox = a.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        w_px = abs((x1 - x0) * sx)
        h_px = abs((y1 - y0) * sy)
        widths.append(min(w_px, h_px))
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


def _poly_to_m(poly_px, px_to_m):
    return [(x * px_to_m, y * px_to_m) for x, y in poly_px]


def _export_gltf_scene(wall_mesh, opening_node_meshes, out_path: Path) -> None:
    import trimesh  # noqa: PLC0415
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    scene.add_geometry(wall_mesh, node_name="walls")
    for name, mesh in opening_node_meshes:
        if mesh is None or len(mesh.vertices) == 0:
            continue
        scene.add_geometry(mesh, node_name=name)
    scene.export(str(out_path), file_type="gltf")


def _post_process_gltf_for_glass(gltf_path: Path) -> None:
    """Promote vertex-alpha materials to KHR_materials_transmission glass."""
    try:
        raw = json.loads(gltf_path.read_text(encoding="utf-8"))
    except Exception:
        return
    materials = raw.get("materials", [])
    if not materials:
        return
    ext_used = set(raw.get("extensionsUsed", []))
    changed = False
    for mat in materials:
        pbr = mat.get("pbrMetallicRoughness") or {}
        base = pbr.get("baseColorFactor") or [1, 1, 1, 1]
        if len(base) < 4 or base[3] >= 0.95:
            continue
        mat.setdefault("extensions", {})["KHR_materials_transmission"] = {
            "transmissionFactor": 0.85
        }
        pbr["baseColorFactor"] = [base[0], base[1], base[2], 1.0]
        pbr["roughnessFactor"] = 0.05
        pbr["metallicFactor"] = 0.0
        mat["pbrMetallicRoughness"] = pbr
        mat["alphaMode"] = "BLEND"
        mat["doubleSided"] = True
        ext_used.add("KHR_materials_transmission")
        changed = True
    if not changed:
        return
    raw["extensionsUsed"] = sorted(ext_used)
    gltf_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def _write_viewer_html(out_dir: Path, gltf_rel: str) -> Path:
    """Write a viewer template that loads the V3 glTF + semantics."""
    # Template liegt seit ADR-016 als Package-Data im v3-Modul (vorher
    # repo_root/scripts/ — der Worker-Container musste dafuer das ganze
    # scripts/-Verzeichnis mitschiffen).
    template = Path(__file__).resolve().parent / "v3" / "bim_review_viewer.html"
    viewer = out_dir / "viewer.html"
    cache_bust = str(int(time.time()))
    if not template.exists():
        viewer.write_text(
            f"<!doctype html><title>V3 BIM</title>"
            f'<script type="module" '
            f'src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/'
            f'model-viewer.min.js"></script>'
            f'<model-viewer src="{gltf_rel}?v={cache_bust}" camera-controls '
            f'style="width:100vw;height:100vh"></model-viewer>',
            encoding="utf-8",
        )
        return viewer
    html = template.read_text(encoding="utf-8")
    html = html.replace("__GLTF_URL__", f"{gltf_rel}?v={cache_bust}")
    html = html.replace(
        "__SEMANTICS_URL__", f"semantics_v3.json?v={cache_bust}",
    )
    viewer.write_text(html, encoding="utf-8")
    return viewer


def _write_semantics(
    out_dir: Path,
    records: List[Any],
    px_to_m: float,
    n_walls: int,
    extrude_total_area_m2: float,
) -> Path:
    from planproof_geometry.ai.v3.pixel_bim.openings import PlacementStatus  # noqa: PLC0415
    sem = {
        "schema_version": "v3",
        "px_to_m": px_to_m,
        "segments": [],
        "elements": [r.to_semantics_dict() for r in records],
        "rooms": [],
        "stats": {
            "n_walls": n_walls,
            "wall_area_m2": round(float(extrude_total_area_m2), 3),
            "n_openings": len(records),
            "n_placed": sum(1 for r in records if r.status in (
                PlacementStatus.OK, PlacementStatus.OK_SINGLE_SIDED,
            )),
            "n_skipped": sum(1 for r in records
                             if str(r.status.value).startswith("skipped_")),
            "n_failed": sum(1 for r in records
                            if str(r.status.value).startswith("failed_")),
        },
    }
    out = out_dir / "semantics_v3.json"
    out.write_text(json.dumps(sem, indent=2), encoding="utf-8")
    return out


def _bundle_outputs(input_path: Path, output_dir: Path, work_dir: Path) -> Path:
    """Zip the per-run solution folder into ``<output_dir>/<basename>.zip``.

    Mirrors V2's CreatePackageStep behaviour: cloud worker expects exactly
    one zip artefact named ``<input_basename>_solution.zip`` next to the
    solution folder.
    """
    basename = input_path.stem
    solution_dir = output_dir / f"{basename}_solution"
    zip_path = output_dir / f"{basename}_solution.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in solution_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(solution_dir))
    return zip_path


def run_plan_to_bim_v3_pipeline(
    *,
    input_path: str,
    output_dir: str,
    work_dir: str,
    config: Dict[str, Any],
    logger_fn: Optional[Callable[[str], None]] = None,
) -> PipelineResult[V3State]:
    """Run the V3 Pixel-BIM pipeline.

    Returns a PipelineResult so cloud_entry.py / runner.py can read
    ``.ok``, ``.state.metrics``, ``.state.model_path``, ``.state.package_path``,
    ``.steps`` (StepResult list) the same way they consume V2 results.
    """
    log = logger_fn or (lambda m: None)
    state = V3State()
    steps: List[StepResult] = []

    input_p = Path(input_path)
    output_p = Path(output_dir)
    work_p = Path(work_dir)
    output_p.mkdir(parents=True, exist_ok=True)
    work_p.mkdir(parents=True, exist_ok=True)
    basename = input_p.stem
    solution_dir = output_p / f"{basename}_solution"
    if solution_dir.exists():
        shutil.rmtree(solution_dir)
    solution_dir.mkdir(parents=True, exist_ok=True)

    wall_height_m = float(config.get("wall_height_m", 2.80))
    render_matrix = float(config.get("v3_render_matrix", 4.0))

    # Annotations file resolution: pipe input or sibling lookup.
    ann_candidates = [
        input_p.with_suffix(".annotations.json"),
        input_p.parent / "input_0.annotations.json",
        Path(config.get("manual_annotations_path", "")) if config.get(
            "manual_annotations_path") else None,
    ]
    ann_path: Optional[Path] = None
    for c in ann_candidates:
        if c is not None and Path(c).exists():
            ann_path = Path(c)
            break

    def _step(name: str, started: float, ok: bool, error: Optional[str] = None) -> None:
        steps.append(StepResult(
            name=name, ok=ok, started_at=started,
            finished_at=time.time(), error=error,
        ))

    # ── Step 1 : Render PDF ──────────────────────────────────────────
    t0 = time.time()
    log(f"v3:render_pdf input={input_p.name} matrix={render_matrix}")
    try:
        image_gray = _render_pdf_grayscale(input_p, render_matrix)
        _step("v3_render_pdf", t0, True)
    except Exception as exc:
        _step("v3_render_pdf", t0, False, error=str(exc))
        return PipelineResult(ok=False, state=None, steps=steps, error=str(exc))
    H, W = image_gray.shape
    state.metrics["v3_image_w"] = W
    state.metrics["v3_image_h"] = H

    # ── Step 2 : Load + scale annotations ────────────────────────────
    t0 = time.time()
    if ann_path is None:
        msg = "v3 pipeline requires an annotations.json sidecar"
        _step("v3_load_annotations", t0, False, error=msg)
        return PipelineResult(ok=False, state=None, steps=steps, error=msg)
    raw_ann = json.loads(ann_path.read_text(encoding="utf-8"))
    img_size = raw_ann.get("image_size") or {}
    ann_w = int(img_size.get("width") or 0) or W
    ann_h = int(img_size.get("height") or 0) or H
    sx = W / ann_w if ann_w else 1.0
    sy = H / ann_h if ann_h else 1.0
    annotations = raw_ann.get("annotations") or []
    state.metrics["v3_n_annotations"] = len(annotations)
    _step("v3_load_annotations", t0, True)

    # ── Step 3 : Calibrate px_to_m from door median ──────────────────
    t0 = time.time()
    px_to_m = _calibrate_px_to_m(annotations, (sx, sy))
    if px_to_m is None:
        msg = "scale calibration failed (need ≥ 3 door_single labels)"
        _step("v3_scale_calibration", t0, False, error=msg)
        return PipelineResult(ok=False, state=None, steps=steps, error=msg)
    state.metrics["v3_px_to_m"] = px_to_m
    _step("v3_scale_calibration", t0, True)

    # ── Step 4 : Phase 1 refine wall labels via mode-aware dispatcher ─
    t0 = time.time()
    log("v3:refine_walls")
    from planproof_geometry.ai.v3.pixel_bim.refine.wall_dispatcher import (  # noqa: PLC0415
        refine_walls,
    )
    wall_polys_px: List[Any] = []
    for ann in annotations:
        cls = ann.get("class") or ""
        if cls not in WALL_CLASSES:
            continue
        poly = _poly_from_annotation(ann, (sx, sy))
        if len(poly) < 3:
            continue
        wall_polys_px.append(poly)
    state.metrics["v3_n_wall_labels_refined"] = len(wall_polys_px)
    if not wall_polys_px:
        msg = "no wall annotations to refine"
        _step("v3_refine_walls", t0, False, error=msg)
        return PipelineResult(ok=False, state=None, steps=steps, error=msg)
    # Only sliding doors are drawn in-wall — their pocket overlaps the
    # wall annotation and the symbol's ink confuses classical refine.
    # Swing doors have ink that legitimately belongs to the wall body
    # (jambs, lintel) and must not be subtracted.
    door_polys_for_subtraction: List[Any] = []
    for ann in annotations:
        cls = ann.get("class") or ""
        if cls != "door_sliding":
            continue
        poly = _poly_from_annotation(ann, (sx, sy))
        if len(poly) < 3:
            continue
        door_polys_for_subtraction.append(poly)
    wall_mask, dispatch_info = refine_walls(
        image_gray, wall_polys_px, px_to_m=px_to_m,
        door_polygons_px=door_polys_for_subtraction or None,
    )
    # Tag-9: dispatcher orthogonalizes mostly-axial user polygons; use the
    # corrected ones for the downstream authority-reclip union mask so the
    # extruded BIM polygon doesn't inherit the original tilt.
    corrected = dispatch_info.get("wall_polygons_corrected_px")
    if corrected is not None:
        wall_polys_px = corrected
    mode = dispatch_info.get("mode", "unknown")
    state.metrics["v3_wall_refine_mode"] = mode
    state.metrics["v3_wall_paired_ratio"] = float(dispatch_info.get("paired_ratio", 0.0))
    state.metrics["v3_wallseg_polygon_fallbacks"] = int(
        dispatch_info.get("n_fallback", dispatch_info.get("fallbacks", 0)),
    )
    if mode == "filled":
        state.metrics["v3_wallseg_model_loaded"] = bool(dispatch_info.get("model_loaded"))
        state.metrics["v3_wallseg_used_tiled"] = bool(dispatch_info.get("used_tiled"))
        state.metrics["v3_wallseg_tolerance_px"] = int(dispatch_info.get("tolerance_px", 0))
        state.metrics["v3_wallseg_pixels_in_field"] = int(
            dispatch_info.get("ml_pixels_in_field", 0),
        )
        state.metrics["v3_wallseg_pixels_dropped_outside_field"] = int(
            dispatch_info.get("ml_pixels_outside_field", 0),
        )
    state.metrics["v3_wall_phantom_filtered"] = int(
        dispatch_info.get("phantom_filtered", 0),
    )
    log(
        f"v3:refine_walls mode={mode} "
        f"paired_ratio={dispatch_info.get('paired_ratio', 0):.2f} "
        f"polys={len(wall_polys_px)} "
        f"fallbacks={dispatch_info.get('n_fallback', dispatch_info.get('fallbacks', 0))} "
        f"phantom_filtered={dispatch_info.get('phantom_filtered', 0)}"
    )
    try:
        import cv2  # noqa: PLC0415
        cv2.imwrite(str(solution_dir / "refined_walls.png"), wall_mask)
    except Exception:
        pass
    _step("v3_refine_walls", t0, True)

    # ── Step 5 : Phase 2 extrude wall mask ───────────────────────────
    t0 = time.time()
    log("v3:extrude_walls")
    from planproof_geometry.ai.v3.pixel_bim.extrude import extrude_wall_mask  # noqa: PLC0415
    import cv2 as _cv2  # noqa: PLC0415
    import numpy as _np  # noqa: PLC0415
    user_wall_union_mask = _np.zeros((image_gray.shape[0], image_gray.shape[1]), dtype=_np.uint8)
    for poly in wall_polys_px:
        if len(poly) < 3:
            continue
        pts = _np.array(
            [[int(round(x)), int(round(y))] for x, y in poly], dtype=_np.int32,
        ).reshape(-1, 1, 2)
        _cv2.fillPoly(user_wall_union_mask, [pts], 255)
    try:
        extrude_result = extrude_wall_mask(
            wall_mask,
            px_to_m=px_to_m,
            wall_height_m=wall_height_m,
            simplify_m=0.005,
            min_polygon_area_m2=0.005,
            image_gray=image_gray,
            user_polygon_union_mask=user_wall_union_mask,
        )
        _step("v3_extrude_walls", t0, True)
    except Exception as exc:
        _step("v3_extrude_walls", t0, False, error=str(exc))
        return PipelineResult(ok=False, state=None, steps=steps, error=str(exc))
    if extrude_result.closed_wall_mask is not None:
        try:
            import cv2  # noqa: PLC0415
            cv2.imwrite(
                str(solution_dir / "closed_wall_mask.png"),
                extrude_result.closed_wall_mask,
            )
        except Exception:
            pass
    wall_mesh = extrude_result.mesh
    state.metrics["v3_n_wall_polygons"] = len(extrude_result.polygons)
    state.metrics["v3_wall_area_m2"] = round(
        float(extrude_result.total_area_m2), 3,
    )
    state.metrics["v3_wall_mesh_vertices"] = int(len(wall_mesh.vertices))
    state.metrics["v3_wall_mesh_faces"] = int(len(wall_mesh.faces))
    state.metrics["v3_wall_mesh_watertight"] = bool(wall_mesh.is_watertight)
    wall_vertex_hash_before = hash(wall_mesh.vertices.tobytes())

    # ── Step 6 : Phase 3 place opening meshes in gaps ────────────────
    t0 = time.time()
    log("v3:place_openings")
    from planproof_geometry.ai.v3.pixel_bim.openings import (  # noqa: PLC0415
        OpeningRecord, PlacementStatus,
        build_opening_meshes, compute_gap_geometry,
        find_flanking_walls, resolve_heights,
    )
    records: List[OpeningRecord] = []
    opening_meshes_per_record: List[Tuple[Any, Any]] = []
    for ann in annotations:
        cls = ann.get("class") or ""
        if cls not in OPENING_CLASSES:
            continue
        label_px = _poly_from_annotation(ann, (sx, sy))
        if len(label_px) < 3:
            continue
        label_m = _poly_to_m(label_px, px_to_m)
        ifc_cls, ifc_predef, ifc_op = _IFC_MAP.get(
            cls, ("IfcDoor", "DOOR", None),
        )
        rec = OpeningRecord(
            label_id=ann["id"], label_class=cls,
            ifc_class=ifc_cls, ifc_predefined_type=ifc_predef,
            ifc_operation_type=ifc_op,
            wall1_index=-1, wall2_index=None,
        )
        flanks = find_flanking_walls(label_m, extrude_result.polygons)
        if flanks is None:
            rec.status = PlacementStatus.SKIPPED_ISOLATED
            rec.notes.append("no wall within 0.5 m of label centre")
            records.append(rec)
            continue
        rec.wall1_index = flanks.wall1_index
        rec.wall2_index = flanks.wall2_index
        w1 = extrude_result.polygons[flanks.wall1_index]
        w2 = (extrude_result.polygons[flanks.wall2_index]
              if flanks.wall2_index is not None else None)
        try:
            gap = compute_gap_geometry(w1, w2, label_m)
        except Exception as exc:
            rec.status = PlacementStatus.FAILED_MESH_BUILD
            rec.notes.append(f"gap_geometry failed: {exc}")
            records.append(rec)
            continue
        rec.notes.extend(gap.notes)
        # Bbox-fallback for corner-case openings: when compute_gap_geometry
        # collapses the gap (typical Klasse-A / corner doors where the
        # label-axis projection misses the inter-wall gap), recover the
        # door width from the label's own extent along W1's major axis.
        # Mirrors the fallback in scripts/diagnose_door_placement.py so V3
        # production matches the diagnose output instead of dropping the
        # opening.
        if gap.gap_length_m < 0.05:
            import numpy as _np  # noqa: PLC0415
            label_pts = _np.asarray(label_m, dtype=_np.float64)
            origin_arr = _np.asarray(gap.origin, dtype=_np.float64)
            major_arr = _np.asarray(gap.major_axis, dtype=_np.float64)
            proj = (label_pts - origin_arr) @ major_arr
            fb_lo = float(proj.min())
            fb_hi = float(proj.max())
            fb_len = fb_hi - fb_lo
            if 0.05 <= fb_len <= 10.0:
                gap.gap_start = fb_lo
                gap.gap_end = fb_hi
                gap.gap_length_m = fb_len
                rec.notes.append(
                    f"bbox-fallback: gap_length {fb_len:.3f} m from label-axis "
                    "projection (compute_gap_geometry returned 0)"
                )
        if gap.gap_length_m < 0.05 or gap.gap_length_m > 10.0:
            rec.status = PlacementStatus.SKIPPED_GAP_DEGENERATE
            rec.notes.append(
                f"gap_length {gap.gap_length_m:.3f} m out of plausible range"
            )
            records.append(rec)
            continue
        if gap.is_corner or gap.is_single_sided:
            rec.status = PlacementStatus.OK_SINGLE_SIDED
        else:
            rec.status = PlacementStatus.OK
        heights = resolve_heights(
            cls, wall_height_m=wall_height_m,
            meta=ann.get("meta") or {},
        )
        rec.sill_m = heights.sill_m
        rec.head_m = heights.head_m
        rec.height_source = heights.source
        rec.din_reference = heights.din_reference
        try:
            meshes = build_opening_meshes(
                rec, gap, wall_height_m=wall_height_m,
            )
        except Exception as exc:
            rec.status = PlacementStatus.FAILED_MESH_BUILD
            rec.notes.append(f"mesh_builder failed: {exc}")
            records.append(rec)
            continue
        # Door-Models: build the visible door geometry (frame jambs + head
        # trim, swing leaf or sliding panel + track) for door classes via
        # door_models.build_door_meshes. This mirrors what
        # scripts/diagnose_door_placement.py does so the V3 production
        # output now contains actual door panels, not just the wall
        # opening + bridge. Default swing/mount values are used —
        # follow-up sprints can wire in the diagnose-script voting for
        # correct hinge / pocket signs.
        if cls in OPENING_CLASSES:
            from planproof_geometry.ai.v3.pixel_bim.openings.door_models import (  # noqa: PLC0415
                PlacedDoor, build_door_meshes,
            )
            ox, oy = gap.origin
            mx, my = gap.major_axis
            nx, ny = gap.minor_axis
            half_d = gap.wall_depth_m * 0.5
            door_corners = [
                (ox + gap.gap_end * mx + half_d * nx,
                 oy + gap.gap_end * my + half_d * ny),
                (ox + gap.gap_start * mx + half_d * nx,
                 oy + gap.gap_start * my + half_d * ny),
                (ox + gap.gap_start * mx - half_d * nx,
                 oy + gap.gap_start * my - half_d * ny),
                (ox + gap.gap_end * mx - half_d * nx,
                 oy + gap.gap_end * my - half_d * ny),
            ]
            placed_door = PlacedDoor(
                polygon_xy=door_corners,
                label_class=cls,
                label_id=ann["id"],
                head_height_m=rec.head_m,
                wall_depth_m=gap.wall_depth_m,
            )
            try:
                door_models_mesh = build_door_meshes([placed_door])
                if door_models_mesh is not None and len(door_models_mesh.vertices) > 0:
                    meshes.materials.append("door_models")
                    # Store the mesh on the meshes object so Step 8 picks it up.
                    setattr(meshes, "door_models", door_models_mesh)
            except Exception as exc:
                rec.notes.append(f"door_models failed: {exc}")
        opening_meshes_per_record.append((rec, meshes))
        records.append(rec)
    placed = sum(1 for r in records if r.status in (
        PlacementStatus.OK, PlacementStatus.OK_SINGLE_SIDED,
    ))
    skipped = sum(1 for r in records
                  if str(r.status.value).startswith("skipped_"))
    failed = sum(1 for r in records
                 if str(r.status.value).startswith("failed_"))
    state.metrics["v3_n_openings"] = len(records)
    state.metrics["v3_openings_placed"] = placed
    state.metrics["v3_openings_skipped"] = skipped
    state.metrics["v3_openings_failed"] = failed
    _step("v3_place_openings", t0, True)

    # ── Step 6b : Phase 3b — Windows via 2D-Block-Pipeline ───────────
    # Hartes Kriterium #1: pro Wand-Block exakt count Fenster, equipartition,
    # geclippt auf Wand-Dicke, Klassen folgen der Tangent-Sortierung.
    t0 = time.time()
    log("v3:place_windows")
    from planproof_geometry.ai.v3.pixel_bim.openings import (  # noqa: PLC0415
        clip_blocks_to_walls,
        find_window_blocks,
        place_windows_in_blocks,
        verify_placement_criterion_1,
        build_window_meshes,
    )
    # Clipper bekommt die FERTIGEN BIM-Waende (post-extrude), nicht die
    # rohen User-Labels. Die extrudierten Polygone landen final im glTF —
    # Wand-Anschluesse muessen gegen sie rechnen, sonst Drift.
    bim_walls_m = [list(wp.shell_xy) for wp in extrude_result.polygons]
    window_blocks = find_window_blocks(
        annotations,
        scale_xy=(sx, sy), px_to_m=px_to_m,
    )
    window_clipped = clip_blocks_to_walls(window_blocks, bim_walls_m)
    window_placed = place_windows_in_blocks(window_clipped)
    window_crit = verify_placement_criterion_1(window_clipped, window_placed)

    state.metrics["v3_n_window_labels"] = sum(
        1 for a in annotations
        if (a.get("class") or "") in WINDOW_LABEL_CLASSES
    )
    state.metrics["v3_n_window_blocks"] = len(window_blocks)
    state.metrics["v3_n_window_clipped"] = len(window_clipped)
    state.metrics["v3_n_windows_placed"] = len(window_placed)
    state.metrics["v3_window_criterion_passed"] = bool(window_crit.passed)
    state.metrics["v3_window_criterion_issues"] = len(window_crit.issues)

    log(
        f"v3:place_windows blocks={len(window_blocks)} "
        f"placed={len(window_placed)} "
        f"criterion={'PASS' if window_crit.passed else 'FAIL'}"
    )

    for w in window_placed:
        ifc_cls, ifc_predef, ifc_op = _IFC_MAP.get(
            w.label_class, ("IfcWindow", "WINDOW", None),
        )
        host_wall = window_clipped[w.block_idx].wall_idx
        rec = OpeningRecord(
            label_id=f"window/{w.block_idx}/{w.sub_index}",
            label_class=w.label_class,
            ifc_class=ifc_cls,
            ifc_predefined_type=ifc_predef,
            ifc_operation_type=ifc_op,
            wall1_index=host_wall if host_wall is not None else -1,
            wall2_index=None,
        )
        rec.sill_m = 0.90
        rec.head_m = min(2.10, wall_height_m - 0.01)
        rec.height_source = "din_default"
        rec.din_reference = "DIN 18055"
        rec.status = PlacementStatus.OK
        rec.width_m = float(w.width_m)
        rec.depth_m = float(w.wall_thickness_m)
        rec.center_xy_m = (float(w.centroid_m[0]), float(w.centroid_m[1]))
        try:
            meshes = build_window_meshes(w, wall_height_m=wall_height_m)
        except Exception as exc:
            rec.status = PlacementStatus.FAILED_MESH_BUILD
            rec.notes.append(f"window_mesh_builder failed: {exc}")
            records.append(rec)
            continue
        opening_meshes_per_record.append((rec, meshes))
        records.append(rec)
    _step("v3_place_windows", t0, True)

    # ── Step 7 : Walls-preserve guardrail ────────────────────────────
    t0 = time.time()
    walls_preserved = wall_vertex_hash_before == hash(wall_mesh.vertices.tobytes())
    state.metrics["v3_walls_preserved"] = bool(walls_preserved)
    if not walls_preserved:
        msg = "walls-preserve guardrail FAILED — wall_mesh vertices changed"
        _step("v3_walls_preserve_guardrail", t0, False, error=msg)
        return PipelineResult(ok=False, state=state, steps=steps, error=msg)
    _step("v3_walls_preserve_guardrail", t0, True)

    # ── Step 8 : Centre scene + assemble glTF ────────────────────────
    t0 = time.time()
    log("v3:export_gltf")
    opening_nodes: List[Tuple[str, Any]] = []
    n_door_bridges = 0
    n_door_models = 0
    for rec, meshes in opening_meshes_per_record:
        tag = rec.label_class
        if meshes.parapet is not None:
            opening_nodes.append((f"{tag}/{rec.label_id}/parapet", meshes.parapet))
        if meshes.lintel is not None:
            opening_nodes.append((f"{tag}/{rec.label_id}/lintel", meshes.lintel))
        if getattr(meshes, "frame", None) is not None:
            opening_nodes.append((f"{tag}/{rec.label_id}/frame", meshes.frame))
        if meshes.fill is not None:
            opening_nodes.append((f"{tag}/{rec.label_id}/fill", meshes.fill))
        if getattr(meshes, "bridge", None) is not None:
            opening_nodes.append((f"{tag}/{rec.label_id}/bridge", meshes.bridge))
            n_door_bridges += 1
        door_models_mesh = getattr(meshes, "door_models", None)
        if door_models_mesh is not None and len(door_models_mesh.vertices) > 0:
            opening_nodes.append((f"{tag}/{rec.label_id}/door_models", door_models_mesh))
            n_door_models += 1
    state.metrics["v3_n_door_bridges"] = n_door_bridges
    state.metrics["v3_n_door_models"] = n_door_models
    b_min, b_max = wall_mesh.bounds
    tx = -(b_min[0] + b_max[0]) / 2
    tz = -(b_min[2] + b_max[2]) / 2
    ty = -b_min[1]
    wall_mesh.apply_translation([tx, ty, tz])
    for _, m in opening_nodes:
        m.apply_translation([tx, ty, tz])
    gltf_path = solution_dir / "model.gltf"
    _export_gltf_scene(wall_mesh, opening_nodes, gltf_path)
    _post_process_gltf_for_glass(gltf_path)
    state.model_path = str(gltf_path)
    _step("v3_export_gltf", t0, True)

    # ── Step 8.5 : Perfect-BIM overwrite ─────────────────────────────
    # Run the diagnose+merge perfect-BIM pipeline and replace V3's
    # model.gltf with the resulting bim_complete_*.glb. Willi's
    # confirmed-perfect output now lands in the cloud bundle directly,
    # so the ScansPage shows the same BIM the local 3-script chain
    # produced. V3's own door/window/bridge meshes from Step 8 become
    # vestigial; we keep the V3 work for metric collection (walls,
    # n_openings, etc.) and the walls-preserve guardrail signal.
    if config.get("use_perfect_bim", True):
        t0 = time.time()
        log("v3:perfect_bim_overwrite")
        try:
            from planproof_geometry.ai.v3.perfect_bim import (  # noqa: PLC0415
                run_door_placement,
                run_full_bim,
            )
            run_door_placement(
                input_p, ann_path, solution_dir,
                render_matrix=render_matrix,
                wall_height_m=wall_height_m,
            )
            full_result = run_full_bim(
                input_p, ann_path, solution_dir,
                render_matrix=render_matrix,
            )
            perfect_glb = full_result["bim_complete_glb_path"]
            target_glb = solution_dir / "model.glb"
            shutil.copy2(perfect_glb, target_glb)
            # Drop the V3-produced glTF + buffer fragments and the
            # perfect-BIM intermediate files so the bundle stays clean.
            for stale in (
                solution_dir / "model.gltf",
                solution_dir / "bim.glb",
                solution_dir / "bim_viewer.html",
                solution_dir / "bim_complete_viewer.html",
                perfect_glb,
            ):
                try:
                    if stale.exists():
                        stale.unlink()
                except OSError:
                    pass
            for buf in solution_dir.glob("gltf_buffer_*.bin"):
                try:
                    buf.unlink()
                except OSError:
                    pass
            state.model_path = str(target_glb)
            state.metrics["v3_perfect_bim_glb_size_kb"] = round(
                target_glb.stat().st_size / 1024, 1,
            )
            state.metrics["v3_perfect_bim_active"] = True
            _step("v3_perfect_bim_overwrite", t0, True)
        except Exception as exc:
            _step("v3_perfect_bim_overwrite", t0, False, error=str(exc))
            return PipelineResult(
                ok=False, state=state, steps=steps,
                error=f"perfect_bim_overwrite: {exc}",
            )
    else:
        state.metrics["v3_perfect_bim_active"] = False

    # ── Step 9 : Semantics + viewer ──────────────────────────────────
    t0 = time.time()
    sem_path = _write_semantics(
        solution_dir, records, px_to_m,
        n_walls=len(extrude_result.polygons),
        extrude_total_area_m2=extrude_result.total_area_m2,
    )
    viewer_target = "model.glb" if state.metrics.get("v3_perfect_bim_active") else "model.gltf"
    _write_viewer_html(solution_dir, viewer_target)
    state.metrics["v3_semantics_path"] = str(sem_path)
    _step("v3_write_semantics", t0, True)

    # ── Step 10 : Bundle to zip ──────────────────────────────────────
    t0 = time.time()
    try:
        zip_path = _bundle_outputs(input_p, output_p, work_p)
        state.package_path = str(zip_path)
        _step("v3_bundle_outputs", t0, True)
    except Exception as exc:
        _step("v3_bundle_outputs", t0, False, error=str(exc))
        return PipelineResult(
            ok=False, state=state, steps=steps, error=str(exc),
        )

    return PipelineResult(ok=True, state=state, steps=steps, error=None)
