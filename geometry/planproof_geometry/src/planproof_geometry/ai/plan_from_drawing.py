from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from ai.plan_prior import PlanPrior, PlannedObject, save_prior

from ai.plan_extract_pdf import extract_pdf_vector
from ai.plan_measure_parse import parse_scale, parse_measurements
from ai.plan_transform import transform_extraction_to_m
from ai.plan_anchor import attach_measurement_anchors
from ai.plan_line_merge import merge_collinear_segments
from ai.plan_gltf_walls import write_walls_gltf
from ai.plan_debug_render import write_debug_svg
from ai.plan_input_detect import detect_plan_input
from ai.plan_extract_raster import extract_raster_plan
from ai.plan_scale_infer import infer_points_to_m_from_measurements

# ============================================================
# RAW MODE: geometry-first
# - alle erkannten Linien ins glTF (ohne semantische Filterung)
# - Texte/Werte bleiben im JSON
# ============================================================
RAW_GEOMETRY_MODE = True


def _to_raw_segments(extraction_dict: dict) -> list[dict]:
    """
    Alle Linien 1:1 aus extraction.geometry -> Segmente fürs glTF.
    Wichtig: id + kind werden mitgegeben für spätere Viewer/ML-Filter.
    """
    segs: list[dict] = []
    for g in extraction_dict.get("geometry", []):
        if g.get("type") != "line":
            continue
        a, b = g.get("a"), g.get("b")
        if not a or not b:
            continue
        segs.append(
            {
                "id": g.get("id"),
                "a": [float(a[0]), float(a[1])],
                "b": [float(b[0]), float(b[1])],
                "kind": "raw_line",
            }
        )
    return segs


def _bbox_from_segments(segs: list[dict]) -> Optional[tuple[float, float, float, float]]:
    """(minx, maxx, miny, maxy) in Meter aus Segmenten."""
    if not segs:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for s in segs:
        (x1, y1) = s["a"]
        (x2, y2) = s["b"]
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    if not xs or not ys:
        return None
    return (min(xs), max(xs), min(ys), max(ys))


# -------------------------------------------------------------------
# Legacy: Door-gap Heuristik (nur im Legacy-Mode verwendet)
# -------------------------------------------------------------------
def _detect_door_like_gaps(
    wall_segments,
    bbox,
    ratio_min: float = 0.02,
    ratio_max: float = 0.25,
    angle_eps_deg: float = 8.0,
):
    """
    Sehr grobe Heuristik: finde Lücken zwischen fast-kollinearen Wandsegmenten,
    deren Abstand relativ zur Gesamtgrundrissgröße typischen Türbreiten entspricht.
    Gibt eine Liste von Dicts mit center und width zurück (im Planraum, Meter/Relativ).
    """
    minx, maxx, miny, maxy = bbox
    span = max(maxx - minx, maxy - miny, 1e-6)

    doors = []

    def seg_vec(s):
        (x1, y1) = s["a"]
        (x2, y2) = s["b"]
        return x2 - x1, y2 - y1

    def seg_len(v):
        return (v[0] ** 2 + v[1] ** 2) ** 0.5

    n = len(wall_segments)
    for i in range(n):
        s1 = wall_segments[i]
        v1 = seg_vec(s1)
        L1 = seg_len(v1)
        if L1 < 1e-6:
            continue

        for j in range(i + 1, n):
            s2 = wall_segments[j]
            v2 = seg_vec(s2)
            L2 = seg_len(v2)
            if L2 < 1e-6:
                continue

            # Orientierung prüfen (fast parallel)
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            ang = 0.0
            denom = L1 * L2
            if denom > 1e-9:
                cosang = max(-1.0, min(1.0, abs(dot) / denom))
                ang = math.degrees(math.acos(cosang))
            if ang > angle_eps_deg:
                continue

            # Kleinsten Abstand zwischen Endpunkten betrachten
            endpoints1 = [tuple(s1["a"]), tuple(s1["b"])]
            endpoints2 = [tuple(s2["a"]), tuple(s2["b"])]
            best_d = None
            best_mid = None
            for p in endpoints1:
                for q in endpoints2:
                    dx = p[0] - q[0]
                    dy = p[1] - q[1]
                    d = (dx * dx + dy * dy) ** 0.5
                    if best_d is None or d < best_d:
                        best_d = d
                        best_mid = (
                            (p[0] + q[0]) * 0.5,
                            (p[1] + q[1]) * 0.5,
                        )

            if best_d is None:
                continue

            ratio = best_d / span
            if ratio < ratio_min or ratio > ratio_max:
                continue

            cx, cy = best_mid

            # Duplikate unterdrücken: Türzentren, die sehr dicht beieinander liegen
            too_close = False
            for d in doors:
                dcx = d["center"][0] - cx
                dcy = d["center"][1] - cy
                if (dcx * dcx + dcy * dcy) ** 0.5 < 0.01 * span:
                    too_close = True
                    break
            if too_close:
                continue

            doors.append({"center": (cx, cy), "width": best_d})

    return doors


def _dist_seg_to_seg(a1, a2, b1, b2) -> float:
    # minimaler Abstand zwischen 2 Segmenten über Endpunkt->Segment (grob, aber reicht)
    def dptseg(p, s1, s2):
        px, py = p
        ax, ay = s1
        bx, by = s2
        vx, vy = (bx - ax, by - ay)
        wx, wy = (px - ax, py - ay)
        vv = vx * vx + vy * vy
        if vv <= 1e-12:
            return math.hypot(px - ax, py - ay)
        t = (wx * vx + wy * vy) / vv
        t = max(0.0, min(1.0, t))
        cx, cy = (ax + t * vx, ay + t * vy)
        return math.hypot(px - cx, py - cy)

    return min(
        dptseg(a1, b1, b2),
        dptseg(a2, b1, b2),
        dptseg(b1, a1, a2),
        dptseg(b2, a1, a2),
    )


def build_plan_bim_from_drawing(
    drawing_path: str,
    out_model_path: str,
    out_prior_path: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    drawing = Path(drawing_path)
    out_model = Path(out_model_path)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    # 1) Detect input + extract
    det = detect_plan_input(str(drawing), max_pages=1)

    if det.kind == "vector_pdf":
        extraction = extract_pdf_vector(str(drawing), max_pages=1)
    elif det.kind in ("raster_pdf", "image"):
        extraction = extract_raster_plan(
            str(drawing),
            max_pages=1,
            target_pdf_render_max_px=4500,
            min_line_len_px=50,
            assumed_image_dpi=300,
            ocr_lang="deu+eng",
        )
    else:
        raise RuntimeError(f"Unsupported/unknown input: {drawing} ({det.reason})")

    extraction_dict = extraction.to_dict()
    extraction_dict["doc"]["input_detect"] = {
        "kind": det.kind,
        "quality": det.quality,
        "reason": det.reason,
        "meta": det.meta,
    }

    # optional: OCR setup info fürs Debug/Support
    try:
        from ai.ocr_setup import configure_tesseract
        extraction_dict["doc"]["ocr_setup"] = configure_tesseract()
    except Exception as e:
        extraction_dict["doc"]["ocr_setup"] = {"ok": False, "error": repr(e)}

    # 2) Parse scale + measurements
    # -> wird NICHT mehr als Gatekeeper genutzt, aber im JSON gespeichert (für später/ML)
    scale = parse_scale(extraction_dict.get("texts", []))
    measures = parse_measurements(extraction_dict.get("texts", []))
    extraction_dict["doc"]["scale"] = {
        "found": scale.found,
        "ratio": scale.ratio,
        "confidence": scale.confidence,
        "source_text_id": scale.source_text_id,
    }
    extraction_dict["measurements"] = [m.__dict__ for m in measures]

    # Optional: scale inference (wenn OCR-Maße vorhanden sind)
    extraction_dict = infer_points_to_m_from_measurements(extraction_dict)

    # 3) Transform to meters
    extraction_dict = transform_extraction_to_m(extraction_dict)

    # ============================================================
    # RAW MODE: ALLE Linien ins 3D Modell übertragen (ohne Filter)
    # ============================================================
    if RAW_GEOMETRY_MODE:
        raw_lines = _to_raw_segments(extraction_dict)
        extraction_dict["raw_lines"] = raw_lines

        # bbox aus allen Linien (für späteres Viewer-Filtering / Kamera-Framing)
        plan_bbox = _bbox_from_segments(raw_lines)
        if plan_bbox is not None:
            (minx, maxx, miny, maxy) = plan_bbox
            extraction_dict["doc"]["plan_bbox_m"] = {
                "minx": float(minx),
                "maxx": float(maxx),
                "miny": float(miny),
                "maxy": float(maxy),
            }

        # 4) Save extraction json + debug svg
        stem = drawing.stem
        out_extraction = out_model.parent / f"{stem}__plan_extraction.json"
        out_extraction.write_text(json.dumps(extraction_dict, indent=2), encoding="utf-8")

        out_debug = out_model.parent / f"{stem}__plan_debug.svg"
        # keine Anchors/Dimension-Linien mehr als "Truth" -> show_anchor_lines=False
        write_debug_svg(
            extraction_dict,
            str(out_debug),
            canvas_w=1400,
            show_text_labels=False,
            show_anchor_lines=False,
        )

        # 5) glTF: alle Linien als 3D (Wand-/Linien-Extrusion)
        # - Höhe: Standard 2.8m (falls keine explizite Angabe vorliegt)
        # - Normalisierung: Zentriere Modell auf den Plan-Mittelpunkt
        origin_xy = None
        if plan_bbox is not None:
            (minx, maxx, miny, maxy) = plan_bbox
            cx0 = (minx + maxx) * 0.5
            cy0 = (miny + maxy) * 0.5
            origin_xy = (float(cx0), float(cy0))
            extraction_dict["doc"]["plan_center_m"] = {"x": float(cx0), "y": float(cy0)}

        write_walls_gltf(
            str(out_model),
            wall_segments=raw_lines,        # ALLE Linien rein
            wall_height_m=2.80,             # 2.6m-3.0m -> Default 2.8m
            wall_thickness_m=0.05,          # besser sichtbar als 1cm
            add_slab=False,
            slab_thickness_m=0.05,
            overlay_segments_2d=None,       # bewusst aus, damit keine doppelte Geometrie

            # Viewer-Ausrichtung an PDF-Orientierung
            plan_origin_xy=origin_xy,
            flip_plan_y_for_pdf=True,
        )

        # 6) Prior im Raw-Mode leer (Filter/ML später)
        prior_path = None
        if out_prior_path:
            prior = PlanPrior(
                version="prior_v1",
                source_drawing=str(drawing),
                source_plan_model=str(out_model),
                objects=[],
                meta={
                    "mode": "raw_geometry",
                    "extraction_json": str(out_extraction),
                    "debug_svg": str(out_debug),
                    "scale": extraction_dict["doc"].get("scale"),
                    "transform": extraction_dict["doc"].get("transform"),
                    "raw_lines_count": len(raw_lines),
                    "input_kind": extraction_dict["doc"].get("input_detect", {}).get("kind"),
                    "quality": extraction_dict["doc"].get("input_detect", {}).get("quality"),
                },
            )
            prior_path = save_prior(prior, out_prior_path)

        return str(out_model), prior_path

    # ============================================================
    # LEGACY MODE (alter Pfad bleibt erhalten)
    # ============================================================

    # 4) Anchor measurements to nearest lines (dimension lines)
    extraction_dict = attach_measurement_anchors(extraction_dict, max_dist_m=1.20)

    geom_lines = [g for g in extraction_dict.get("geometry", []) if g.get("type") == "line"]

    # Dimension-Hauptlinien
    dim_lines = [
        g for g in geom_lines
        if g.get("is_dimension_line") or g.get("is_dimension_mainline")
    ]

    def _is_near_dimension(seg: dict) -> bool:
        a = seg.get("a"); b = seg.get("b")
        if not a or not b:
            return False
        A = (float(a[0]), float(a[1]))
        B = (float(b[0]), float(b[1]))
        L = math.hypot(B[0]-A[0], B[1]-A[1])

        near_thresh = max(0.02, 0.05 * L)

        for d in dim_lines:
            da = d.get("a"); db = d.get("b")
            if not da or not db:
                continue
            D1 = (float(da[0]), float(da[1]))
            D2 = (float(db[0]), float(db[1]))
            if _dist_seg_to_seg(A, B, D1, D2) <= near_thresh:
                return True
        return False

    wall_candidates = [
        g for g in geom_lines
        if not g.get("is_dimension_line")
        and not g.get("is_dimension_tick")
        and not g.get("is_dimension_mainline")
        and not _is_near_dimension(g)
    ]

    # 6) Merge collinear segments -> walls
    doc = extraction_dict.get("doc", {})
    scale_info = doc.get("scale") or {}
    tf = doc.get("transform") or {}
    ps = doc.get("page_size") or {}

    points_to_m = float(tf.get("points_to_m") or 1.0)
    page_short_m = min(float(ps.get("w") or 0.0), float(ps.get("h") or 0.0)) * points_to_m

    collinear_tol_m = 0.02
    join_tol_m = 0.06
    min_len_m = 0.50

    if not scale_info.get("found", False) and page_short_m > 0:
        min_len_m = max(0.003, 0.03 * page_short_m)
        collinear_tol_m = max(0.001, 0.01 * page_short_m)
        join_tol_m = max(0.003, 0.03 * page_short_m)

    wall_segments = merge_collinear_segments(
        wall_candidates,
        angle_eps_deg=3.0,
        collinear_tol_m=collinear_tol_m,
        join_tol_m=join_tol_m,
        min_len_m=min_len_m,
    )

    extraction_dict["walls"] = wall_segments

    # bbox aus walls
    plan_bbox = None
    if wall_segments:
        xs: list[float] = []
        ys: list[float] = []
        for w in wall_segments:
            (x1, y1) = w["a"]
            (x2, y2) = w["b"]
            xs.extend([x1, x2])
            ys.extend([y1, y2])

        if xs and ys:
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            plan_bbox = (minx, maxx, miny, maxy)
            doc["plan_bbox_m"] = {
                "minx": float(minx),
                "maxx": float(maxx),
                "miny": float(miny),
                "maxy": float(maxy),
            }

    stem = drawing.stem
    out_extraction = out_model.parent / f"{stem}__plan_extraction.json"
    out_extraction.write_text(json.dumps(extraction_dict, indent=2), encoding="utf-8")

    out_debug = out_model.parent / f"{stem}__plan_debug.svg"
    write_debug_svg(extraction_dict, str(out_debug), canvas_w=1400, show_text_labels=False, show_anchor_lines=True)

    overlay = extraction_dict.get("geometry", [])

    write_walls_gltf(
        str(out_model),
        wall_segments=wall_segments,
        wall_height_m=3.00,
        wall_thickness_m=0.12,
        add_slab=False,
        slab_thickness_m=0.05,
        overlay_segments_2d=overlay,
        overlay_thickness_m=0.01,
        overlay_height_m=0.002,
    )

    # legacy prior
    prior_objects: list[PlannedObject] = []
    if plan_bbox is not None and wall_segments:
        door_candidates = _detect_door_like_gaps(wall_segments, plan_bbox)

        minx, maxx, miny, maxy = plan_bbox
        span_x = max(maxx - minx, 1e-6)
        span_y = max(maxy - miny, 1e-6)

        wall_height_m = 3.0
        door_height_m = 2.1

        center_z_rel = (door_height_m * 0.5) / wall_height_m
        dims_z_rel = door_height_m / wall_height_m

        span_short = min(span_x, span_y)
        depth_m = 0.1 * span_short
        depth_rel = depth_m / span_y if span_y > 0 else 0.05

        for idx, door in enumerate(door_candidates, start=1):
            cx, cy = door["center"]
            width = door["width"]

            cx_rel = (cx - minx) / span_x
            cy_rel = (cy - miny) / span_y

            dims_x_rel = width / span_x
            dims_y_rel = depth_rel

            prior_objects.append(
                PlannedObject(
                    id=f"door_{idx}",
                    coarse="door",
                    fine=None,
                    center_rel=(float(cx_rel), float(cy_rel), float(center_z_rel)),
                    dims_rel=(float(dims_x_rel), float(dims_y_rel), float(dims_z_rel)),
                    mount="wall",
                    meta={
                        "kind": "door_gap",
                        "center_m": [float(cx), float(cy)],
                        "width_m": float(width),
                    },
                )
            )

    prior_path = None
    if out_prior_path:
        prior = PlanPrior(
            version="prior_v1",
            source_drawing=str(drawing),
            source_plan_model=str(out_model),
            objects=prior_objects,
            meta={
                "extraction_json": str(out_extraction),
                "debug_svg": str(out_debug),
                "scale": extraction_dict["doc"].get("scale"),
                "transform": extraction_dict["doc"].get("transform"),
                "wall_segments_count": len(wall_segments),
                "merge_params": {
                    "collinear_tol_m": collinear_tol_m,
                    "join_tol_m": join_tol_m,
                    "min_len_m": min_len_m,
                    "scale_found": bool(scale_info.get("found", False)),
                    "page_short_m": page_short_m,
                },
                "input_kind": extraction_dict["doc"].get("input_detect", {}).get("kind"),
                "quality": extraction_dict["doc"].get("input_detect", {}).get("quality"),
            },
        )
        prior_path = save_prior(prior, out_prior_path)

    return str(out_model), prior_path
