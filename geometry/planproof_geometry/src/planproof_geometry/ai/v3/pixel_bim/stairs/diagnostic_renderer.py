"""Stair diagnostic PNG renderer (Tag 7).

Produces a single annotated overview PNG that lets a developer (or
Willi) eyeball every stair-extraction decision without firing up the
BIM viewer. The PNG is written into ``OUT`` next to ``bim_complete_*.glb``
and named ``stairs_diagnostic.png`` (separate name so it does NOT
overwrite the existing ``bim_walls_and_bridges.png`` diagnostic from
the door pipeline).

Layout: a single OpenCV-rendered image stacking, top-to-bottom:
    * the page top-down render (grayscale, downscaled to ~1600 px wide)
      with per-stair overlay:
          - filled translucent rectangle for the polygon bbox
          - red arrow from centroid along ``direction_up_xy`` of
            length 0.6 * length_m
          - long-axis dashed line in green
          - tread-centerlines as thin orange lines at evenly-spaced
            going_m positions
          - text label with id + status + n_steps + Schrittmaßflag
    * a strip of mini density-profile plots, one per stair, drawn as a
      simple line-graph next to its id

Production-pipeline integration: write only. Auto-opening with
``cmd start`` is the standalone CLI's responsibility
(``scripts/p2b/diagnose_stairs.py``), not the worker's.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .pipeline import StairPipelineResult
from .records import StairStatus

# Visual styles ------------------------------------------------------------
_BBOX_COLOR = (60, 120, 220)         # OpenCV BGR — blue-ish
_BBOX_ALPHA = 0.18                   # translucent overlay
_LONG_AXIS_COLOR = (0, 200, 0)       # green
_DIRECTION_COLOR = (0, 0, 220)       # red
_TREAD_LINE_COLOR = (0, 140, 240)    # orange
_TEXT_COLOR = (15, 15, 15)
_TEXT_OUTLINE = (240, 240, 240)
_STATUS_COLOR_OK = (0, 140, 0)
_STATUS_COLOR_SKIP = (160, 80, 80)
_PROFILE_BG = (250, 250, 250)
_PROFILE_LINE = (40, 60, 130)
_PROFILE_AXIS = (180, 180, 180)

_PROFILE_WIDTH_PX = 320
_PROFILE_HEIGHT_PX = 80
_PROFILE_PAD_PX = 8


def _scale_image_to_max_width(image: np.ndarray, max_width: int) -> tuple:
    """Downscale a grayscale image to ``max_width`` and return (resized, scale).

    Scale = render-px / output-px (so callers can invert overlay coords).
    """
    import cv2

    H, W = image.shape[:2]
    if W <= max_width:
        return image.copy(), 1.0
    scale = W / float(max_width)
    new_w = max_width
    new_h = int(round(H / scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _put_text_with_outline(
    img: np.ndarray,
    text: str,
    org: tuple,
    *,
    font_scale: float = 0.45,
    thickness: int = 1,
) -> None:
    """Draw text with a thin light outline so it reads on any background."""
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                _TEXT_OUTLINE, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                _TEXT_COLOR, thickness, cv2.LINE_AA)


def _draw_profile(profile: np.ndarray) -> np.ndarray:
    """Render one density profile as a small line graph (BGR image)."""
    import cv2

    canvas = np.full(
        (_PROFILE_HEIGHT_PX, _PROFILE_WIDTH_PX, 3), 255, dtype=np.uint8,
    )
    canvas[:] = _PROFILE_BG

    if profile is None or profile.size == 0:
        _put_text_with_outline(
            canvas, "(no profile)", (_PROFILE_PAD_PX, _PROFILE_HEIGHT_PX // 2),
        )
        return canvas

    p_min = float(profile.min())
    p_max = float(profile.max())
    span = max(p_max - p_min, 1e-6)
    inner_w = _PROFILE_WIDTH_PX - 2 * _PROFILE_PAD_PX
    inner_h = _PROFILE_HEIGHT_PX - 2 * _PROFILE_PAD_PX
    # Axis line
    cv2.line(
        canvas,
        (_PROFILE_PAD_PX, _PROFILE_HEIGHT_PX - _PROFILE_PAD_PX),
        (_PROFILE_WIDTH_PX - _PROFILE_PAD_PX,
         _PROFILE_HEIGHT_PX - _PROFILE_PAD_PX),
        _PROFILE_AXIS, 1,
    )
    # Polyline of profile
    pts = []
    n = profile.size
    for i in range(n):
        x = _PROFILE_PAD_PX + int(round((i / (n - 1)) * inner_w)) if n > 1 else _PROFILE_PAD_PX
        y = (_PROFILE_HEIGHT_PX - _PROFILE_PAD_PX
             - int(round(((profile[i] - p_min) / span) * inner_h)))
        pts.append((x, y))
    pts_np = np.array(pts, dtype=np.int32)
    cv2.polylines(canvas, [pts_np], False, _PROFILE_LINE, 1, cv2.LINE_AA)
    return canvas


def _overlay_stair(
    canvas_bgr: np.ndarray,
    *,
    result: StairPipelineResult,
    bbox_px: Optional[tuple],
    scale_for_output: float,
    px_to_m: float,
) -> None:
    """Draw all per-stair overlay glyphs onto an already-BGR canvas."""
    import cv2

    record = result.record

    if bbox_px is not None:
        # bbox_px is (x0, y0, x1, y1) in render-px; downscale to output coords.
        x0, y0, x1, y1 = (int(round(v / scale_for_output)) for v in bbox_px)
        # Translucent fill.
        overlay = canvas_bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), _BBOX_COLOR, thickness=-1)
        cv2.addWeighted(overlay, _BBOX_ALPHA, canvas_bgr, 1 - _BBOX_ALPHA, 0,
                        dst=canvas_bgr)
        cv2.rectangle(canvas_bgr, (x0, y0), (x1, y1), _BBOX_COLOR, 1)

    if record.status != StairStatus.OK:
        # No geometry overlays for skipped stairs, just a label.
        label = f"{record.label_id}: {record.status.value}"
        anchor = (
            int(round((bbox_px[0] if bbox_px else 0) / scale_for_output)) + 4,
            int(round((bbox_px[1] if bbox_px else 0) / scale_for_output)) - 6,
        )
        _put_text_with_outline(canvas_bgr, label, anchor)
        return

    cx_m, cy_m = record.centroid_xy_m
    cx_px = int(round(cx_m / px_to_m / scale_for_output))
    cy_px = int(round(cy_m / px_to_m / scale_for_output))

    # Long axis dashed line
    if record.length_m > 0.0:
        rad = math.radians(record.long_axis_deg)
        lx, ly = math.cos(rad), math.sin(rad)
        half_len_px = (record.length_m * 0.5 / px_to_m) / scale_for_output
        for sign in (-1, +1):
            for k in range(0, 16, 2):
                t0 = sign * (half_len_px * (k / 16.0))
                t1 = sign * (half_len_px * ((k + 1) / 16.0))
                p0 = (int(round(cx_px + t0 * lx)),
                      int(round(cy_px + t0 * ly)))
                p1 = (int(round(cx_px + t1 * lx)),
                      int(round(cy_px + t1 * ly)))
                cv2.line(canvas_bgr, p0, p1, _LONG_AXIS_COLOR, 1, cv2.LINE_AA)

    # Direction-up arrow
    ux, uy = record.direction_up_xy
    arrow_len_px = (record.length_m * 0.30 / px_to_m) / scale_for_output if record.length_m else 30
    p_end = (
        int(round(cx_px + ux * arrow_len_px)),
        int(round(cy_px + uy * arrow_len_px)),
    )
    cv2.arrowedLine(canvas_bgr, (cx_px, cy_px), p_end,
                    _DIRECTION_COLOR, 2, cv2.LINE_AA, tipLength=0.18)

    # Tread centerlines
    if (record.n_steps is not None and record.n_steps > 0
            and record.going_m and record.going_m > 0.0
            and record.length_m > 0.0):
        # Tread positions along the long axis, walking from the LOW
        # end (centroid - 0.5 * length * direction_up) outward.
        going_px_out = (record.going_m / px_to_m) / scale_for_output
        half_w_px = (record.width_m * 0.5 / px_to_m) / scale_for_output
        # Short-axis perpendicular to direction-up.
        sx, sy = -uy, ux
        low_cx_px = cx_px - (record.length_m * 0.5 / px_to_m) / scale_for_output * ux
        low_cy_px = cy_px - (record.length_m * 0.5 / px_to_m) / scale_for_output * uy
        for i in range(record.n_steps):
            t = (i + 0.5) * going_px_out
            mid_x = low_cx_px + t * ux
            mid_y = low_cy_px + t * uy
            a = (int(round(mid_x + sx * half_w_px)),
                 int(round(mid_y + sy * half_w_px)))
            b = (int(round(mid_x - sx * half_w_px)),
                 int(round(mid_y - sy * half_w_px)))
            cv2.line(canvas_bgr, a, b, _TREAD_LINE_COLOR, 1, cv2.LINE_AA)

    # Text annotation under the bbox.
    label_parts = [f"{record.label_id}", f"n={record.n_steps}"]
    if record.rise_m is not None and record.going_m is not None:
        label_parts.append(f"{record.rise_m * 100:.0f}/{record.going_m * 100:.0f}cm")
    if record.din_deviations:
        kinds = "+".join(d["kind"] for d in record.din_deviations)
        label_parts.append(f"DIN:{kinds}")
    if any("LOW_CONFIDENCE_TREAD_COUNT" in n for n in record.notes):
        label_parts.append("LOW_CONF")
    if any("direction_up_default_used" in n for n in record.notes):
        label_parts.append("DEFAULT_DIR")
    label = " | ".join(label_parts)
    anchor_y = (int(round((bbox_px[3] if bbox_px else cy_px) / scale_for_output))
                + 14)
    anchor_x = int(round((bbox_px[0] if bbox_px else cx_px) / scale_for_output))
    _put_text_with_outline(canvas_bgr, label, (anchor_x, anchor_y))


def _poly_m_to_canvas_int(
    polygon_m: Sequence[Tuple[float, float]],
    *,
    px_to_m: float,
    scale_for_output: float,
) -> np.ndarray:
    """Project a plan-metre polygon into output-canvas integer pixel coords."""
    if not polygon_m or px_to_m <= 0.0:
        return np.zeros((0, 2), dtype=np.int32)
    arr = np.asarray(polygon_m, dtype=np.float64) / px_to_m / scale_for_output
    return np.round(arr).astype(np.int32)


def _slice_step_polygons_m(
    *,
    trimmed_exterior_m: Sequence[Tuple[float, float]],
    trimmed_holes_m: Sequence[Sequence[Tuple[float, float]]],
    record,  # StairRecord
) -> List[List[Tuple[float, float]]]:
    """Mirror the mesh-builder's per-step slicing in PDF-Y plan coords.

    The mesh builder slices in the flipped (plan_x, -plan_y) frame; we
    repeat that slicing here in the *unflipped* plan-XY frame (PDF Y
    points down) so the output is directly drawable on the rendered
    plan image. Geometry is otherwise identical: a strip of width
    ``width_m`` perpendicular to ``direction_up_xy``, intersected with
    the stair polygon, swept from low-end up.

    Returns one polygon per step (may be empty if the slice was empty),
    each as a list of ``(x_m, y_m)`` vertices.
    """
    from shapely.geometry import MultiPolygon, Polygon  # noqa: PLC0415

    if not trimmed_exterior_m:
        return []
    n_steps = record.n_steps
    going_m = record.going_m
    width_m = record.width_m
    length_m = record.length_m
    if not n_steps or not going_m or going_m <= 0.0 or width_m <= 0.0:
        return []

    try:
        stair_poly = Polygon(
            list(trimmed_exterior_m),
            holes=[list(h) for h in trimmed_holes_m],
        ).buffer(0)
    except Exception:
        return []
    if stair_poly.is_empty or not stair_poly.is_valid:
        return []
    if isinstance(stair_poly, MultiPolygon):
        stair_poly = max(stair_poly.geoms, key=lambda g: g.area)

    ux, uy = record.direction_up_xy
    norm = math.hypot(ux, uy)
    if norm <= 1e-9:
        return []
    ux, uy = ux / norm, uy / norm
    sx, sy = -uy, ux
    half_w = max(width_m, 1.0) * 0.5
    cx, cy = record.centroid_xy_m
    low_x = cx - 0.5 * length_m * ux
    low_y = cy - 0.5 * length_m * uy

    slices: List[List[Tuple[float, float]]] = []
    for i in range(n_steps):
        t_lo = i * going_m
        t_hi = t_lo + going_m
        strip = Polygon([
            (low_x + t_lo * ux - half_w * sx, low_y + t_lo * uy - half_w * sy),
            (low_x + t_hi * ux - half_w * sx, low_y + t_hi * uy - half_w * sy),
            (low_x + t_hi * ux + half_w * sx, low_y + t_hi * uy + half_w * sy),
            (low_x + t_lo * ux + half_w * sx, low_y + t_lo * uy + half_w * sy),
        ]).buffer(0)
        if strip.is_empty or not strip.is_valid:
            slices.append([])
            continue
        try:
            inter = stair_poly.intersection(strip)
        except Exception:
            slices.append([])
            continue
        if inter.is_empty:
            slices.append([])
            continue
        if isinstance(inter, MultiPolygon):
            inter = max(inter.geoms, key=lambda g: g.area)
        if not isinstance(inter, Polygon):
            slices.append([])
            continue
        slices.append(list(inter.exterior.coords))
    return slices


# Label-vs-BIM colours --------------------------------------------------------
# User label uses pure cyan with a near-black drop-line so the dashed
# outline reads on both bright (page-white) and busy (gray-ink) plan
# backgrounds.
_LABEL_POLY_COLOR = (255, 200, 0)        # bright cyan/teal in BGR
_LABEL_OUTLINE_COLOR = (15, 15, 60)      # dark outline drawn first
_LABEL_FILL_ALPHA = 0.04
_BIM_POLY_COLOR = (200, 30, 200)         # magenta for trimmed BIM footprint
_BIM_FILL_ALPHA = 0.10                   # lighter so per-step fills read clearly
_STEP_SLICE_COLOR = (0, 140, 240)        # orange outlines per step
_STEP_SLICE_FILL_ALPHA = 0.32            # filled, alternating fill per step
_STEP_SLICE_ALT_COLOR = (60, 180, 250)   # alternating lighter orange
_STEP_NUMBER_COLOR = (15, 15, 15)
_STEP_NUMBER_OUTLINE = (245, 245, 245)
# Density-peak centrelines drawn directly on top of the plan ink so
# the user can verify the pixel tread-count visually. Green = peak,
# alpha-blended over the magenta BIM fill so plan ink remains readable.
_TREAD_CENTRELINE_COLOR = (40, 200, 40)
_STRINGER_COLOR = (160, 20, 160)         # dim magenta — the wangen
_CUTLINE_COLOR = (0, 90, 230)            # bold orange/red — Höhensprung
_HOUGH_TREAD_COLOR = (40, 160, 40)       # dim green — every Hough segment


def _draw_polygon(
    canvas_bgr: np.ndarray,
    polygon_canvas_px: np.ndarray,
    *,
    color: tuple,
    fill_alpha: float = 0.0,
    line_thickness: int = 2,
    dashed: bool = False,
    holes_canvas_px: Optional[Sequence[np.ndarray]] = None,
) -> None:
    """Outline a polygon on the canvas, optionally with a translucent fill."""
    import cv2

    if polygon_canvas_px.shape[0] < 3:
        return

    if fill_alpha > 0.0:
        overlay = canvas_bgr.copy()
        cv2.fillPoly(overlay, [polygon_canvas_px], color)
        if holes_canvas_px:
            # Paint holes back to neutral grey so they read as cut-outs.
            for h in holes_canvas_px:
                if h.shape[0] >= 3:
                    cv2.fillPoly(overlay, [h], (245, 245, 245))
        cv2.addWeighted(overlay, fill_alpha, canvas_bgr, 1.0 - fill_alpha, 0.0,
                        dst=canvas_bgr)

    if dashed:
        # Draw a thicker dark outline first so the dashes read on any
        # background, then the bright dashes on top.
        cv2.polylines(canvas_bgr, [polygon_canvas_px], True,
                      _LABEL_OUTLINE_COLOR, line_thickness + 3, cv2.LINE_AA)
        n = polygon_canvas_px.shape[0]
        for i in range(n):
            p0 = tuple(int(v) for v in polygon_canvas_px[i])
            p1 = tuple(int(v) for v in polygon_canvas_px[(i + 1) % n])
            # 8-segment dashes on each edge.
            for k in range(0, 8, 2):
                a = (
                    int(round(p0[0] + (p1[0] - p0[0]) * (k / 8.0))),
                    int(round(p0[1] + (p1[1] - p0[1]) * (k / 8.0))),
                )
                b = (
                    int(round(p0[0] + (p1[0] - p0[0]) * ((k + 1) / 8.0))),
                    int(round(p0[1] + (p1[1] - p0[1]) * ((k + 1) / 8.0))),
                )
                cv2.line(canvas_bgr, a, b, color, line_thickness + 1, cv2.LINE_AA)
    else:
        cv2.polylines(canvas_bgr, [polygon_canvas_px], True, color,
                      line_thickness, cv2.LINE_AA)
        if holes_canvas_px:
            for h in holes_canvas_px:
                if h.shape[0] >= 3:
                    cv2.polylines(canvas_bgr, [h], True, color,
                                  max(1, line_thickness - 1), cv2.LINE_AA)


def render_label_vs_bim_overlay(
    *,
    image_gray: np.ndarray,
    px_to_m: float,
    results: Sequence[StairPipelineResult],
    output_path: Path,
    max_width_px: int = 2000,
) -> Path:
    """Render a label-vs-BIM overlay PNG (Self-Check mode).

    For each stair label this draws three things on top of the rendered
    plan image:

        1. The user label polygon (blue dashed outline, very light fill).
        2. The trimmed BIM footprint polygon (magenta solid outline,
           translucent fill). This is the polygon the mesh-builder
           sliced into per-step boxes — i.e. exactly what ends up in
           the BIM.
        3. The per-step polygon slices (orange thin outlines), one per
           extruded box. With 14–17 typical Wohnungs-Treppen each box
           is visible as a thin strip across the stair.

    A compact legend + per-stair status banner is rendered into the
    top header. Skipped stairs only show the user label + their
    skip-reason banner so they are still locatable.
    """
    import cv2

    top, scale_for_output = _scale_image_to_max_width(image_gray, max_width_px)
    top_bgr = cv2.cvtColor(top, cv2.COLOR_GRAY2BGR)

    # Brighten the plan slightly so the overlays read better.
    top_bgr = cv2.addWeighted(top_bgr, 0.85,
                              np.full_like(top_bgr, 255), 0.15, 0)

    for r in results:
        rec = r.record

        # 1) User label (always drawn, even for skipped stairs).
        user_canvas = _poly_m_to_canvas_int(
            r.user_polygon_m,
            px_to_m=px_to_m,
            scale_for_output=scale_for_output,
        )
        if user_canvas.shape[0] >= 3:
            _draw_polygon(
                top_bgr, user_canvas,
                color=_LABEL_POLY_COLOR,
                fill_alpha=_LABEL_FILL_ALPHA,
                line_thickness=2,
                dashed=True,
            )

        # Anchor for the per-stair text banner.
        if user_canvas.shape[0] > 0:
            text_x = int(user_canvas[:, 0].min())
            text_y = int(user_canvas[:, 1].min()) - 8
        else:
            text_x, text_y = 8, 20

        if rec.status != StairStatus.OK:
            banner = f"{rec.label_id} [{rec.status.value}]"
            _put_text_with_outline(
                top_bgr, banner, (text_x, max(text_y, 14)),
                font_scale=0.5, thickness=2,
            )
            continue

        # 2) Trimmed BIM footprint (the actual BIM polygon).
        bim_canvas = _poly_m_to_canvas_int(
            r.trimmed_polygon_exterior_m,
            px_to_m=px_to_m,
            scale_for_output=scale_for_output,
        )
        hole_canvases = [
            _poly_m_to_canvas_int(
                h, px_to_m=px_to_m, scale_for_output=scale_for_output,
            )
            for h in r.trimmed_polygon_holes_m
        ]
        if bim_canvas.shape[0] >= 3:
            _draw_polygon(
                top_bgr, bim_canvas,
                color=_BIM_POLY_COLOR,
                fill_alpha=_BIM_FILL_ALPHA,
                line_thickness=2,
                dashed=False,
                holes_canvas_px=hole_canvases,
            )

        # 3) Per-step polygon slices. Prefer the IST-Zustand polygons
        # computed in the pipeline from actual density-peak positions;
        # only fall back to the equal-spacing slicer when those are
        # unavailable (e.g. pixel detection failed).
        step_polys_m = list(r.step_polygons_xy_m)
        if not step_polys_m:
            step_polys_m = _slice_step_polygons_m(
                trimmed_exterior_m=r.trimmed_polygon_exterior_m,
                trimmed_holes_m=r.trimmed_polygon_holes_m,
                record=rec,
            )
        n_drawn = 0
        for sp in step_polys_m:
            if not sp:
                continue
            sp_canvas = _poly_m_to_canvas_int(
                sp,
                px_to_m=px_to_m,
                scale_for_output=scale_for_output,
            )
            if sp_canvas.shape[0] >= 3:
                cv2.polylines(top_bgr, [sp_canvas], True,
                              _STEP_SLICE_COLOR, 1, cv2.LINE_AA)
                n_drawn += 1

        # 3b) Tread centrelines (green) — exactly where the pixel
        # extractor detected a tread peak in the plan ink. These let
        # the user visually verify that BIM step boundaries align with
        # plan treads.
        for line in r.tread_centerlines_xy_m:
            if len(line) != 2:
                continue
            p0 = _poly_m_to_canvas_int(
                [line[0]], px_to_m=px_to_m, scale_for_output=scale_for_output,
            )
            p1 = _poly_m_to_canvas_int(
                [line[1]], px_to_m=px_to_m, scale_for_output=scale_for_output,
            )
            if p0.shape[0] and p1.shape[0]:
                cv2.line(
                    top_bgr,
                    (int(p0[0, 0]), int(p0[0, 1])),
                    (int(p1[0, 0]), int(p1[0, 1])),
                    _TREAD_CENTRELINE_COLOR,
                    2, cv2.LINE_AA,
                )

        # 4) Direction-up arrow originating at the centroid.
        cx_m, cy_m = rec.centroid_xy_m
        cx_px = int(round(cx_m / px_to_m / scale_for_output))
        cy_px = int(round(cy_m / px_to_m / scale_for_output))
        ux, uy = rec.direction_up_xy
        norm = math.hypot(ux, uy)
        if norm > 1e-9 and rec.length_m > 0.0:
            ux, uy = ux / norm, uy / norm
            arrow_len_px = (rec.length_m * 0.35 / px_to_m) / scale_for_output
            p_end = (
                int(round(cx_px + ux * arrow_len_px)),
                int(round(cy_px + uy * arrow_len_px)),
            )
            cv2.arrowedLine(top_bgr, (cx_px, cy_px), p_end,
                            _DIRECTION_COLOR, 3, cv2.LINE_AA, tipLength=0.20)

        # 5) Per-stair text banner.
        banner_parts = [
            rec.label_id,
            f"{rec.length_m:.2f}x{rec.width_m:.2f}m",
            f"n={rec.n_steps}",
        ]
        if rec.rise_m is not None and rec.going_m is not None:
            banner_parts.append(
                f"{rec.rise_m * 100:.0f}/{rec.going_m * 100:.0f}cm"
            )
        if r.wall_subtraction_used:
            banner_parts.append("WALL_SUB")
        banner_parts.append(f"steps_drawn={n_drawn}/{rec.n_steps}")
        if any("direction_up_source=arrow" in n for n in rec.notes):
            banner_parts.append("DIR:arrow")
        elif any("direction_up_default_used" in n for n in rec.notes):
            banner_parts.append("DIR:DEFAULT")
        elif any("direction_up_source=" in n for n in rec.notes):
            banner_parts.append("DIR:density")
        banner = " | ".join(banner_parts)
        _put_text_with_outline(
            top_bgr, banner, (text_x, max(text_y, 14)),
            font_scale=0.5, thickness=2,
        )

    # Legend strip (top) — measure text widths so chips do not overlap.
    legend_h = 60
    legend = np.full((legend_h, top_bgr.shape[1], 3), 30, dtype=np.uint8)
    pad = 14
    box_w = 22
    box_h = 18
    chip_gap = 22
    items = [
        ("USER LABEL (dashed)", _LABEL_POLY_COLOR, True),
        ("BIM FOOTPRINT (= what ends up extruded)", _BIM_POLY_COLOR, False),
        ("PER-STEP SLICES", _STEP_SLICE_COLOR, False),
        ("DIRECTION-UP", _DIRECTION_COLOR, False),
    ]
    x = pad
    for txt, color, dashed in items:
        cv2.rectangle(legend, (x, 30), (x + box_w, 30 + box_h), color, -1)
        if dashed:
            cv2.rectangle(legend, (x, 30), (x + box_w, 30 + box_h),
                          (255, 255, 255), 1, cv2.LINE_AA)
        text_org = (x + box_w + 6, 30 + box_h - 4)
        cv2.putText(legend, txt, text_org,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA)
        (tw, _th), _ = cv2.getTextSize(
            txt, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1,
        )
        x += box_w + 6 + tw + chip_gap

    n_ok = sum(1 for r in results if r.record.status == StairStatus.OK)
    n_skip = len(results) - n_ok
    title = (
        f"Stairs Label-vs-BIM Self-Check — {len(results)} stair(s) "
        f"({n_ok} OK / {n_skip} skipped, px_to_m={px_to_m:.5f}, "
        f"downscale={scale_for_output:.2f}x)"
    )
    cv2.putText(legend, title, (pad, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (245, 245, 245), 1, cv2.LINE_AA)

    canvas = np.vstack([legend, top_bgr])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def render_label_vs_bim_per_stair_zooms(
    *,
    image_gray: np.ndarray,
    px_to_m: float,
    results: Sequence[StairPipelineResult],
    output_dir: Path,
    pad_m: float = 1.5,
    zoom_min_canvas_px: int = 900,
) -> List[Path]:
    """Crop the rendered plan tightly around each stair and overlay both polygons.

    Produces one PNG per stair label so a developer can actually see
    the per-step slices, the wall-subtraction notches (when active),
    and the user-label vs BIM-footprint mismatch at full pixel
    resolution rather than at the downscaled overview zoom.

    Args:
        image_gray: rendered grayscale plan, uint8.
        px_to_m: metres per pixel of ``image_gray``.
        results: pipeline outputs (lockstep with stair labels).
        output_dir: directory the per-stair PNGs are written into.
        pad_m: padding in metres around the union of user-label and
            BIM-footprint bbox.
        zoom_min_canvas_px: minimum output edge in pixels — small
            stairs are upscaled to at least this resolution.

    Returns:
        List of written PNG paths.
    """
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    H, W = image_gray.shape[:2]

    for r in results:
        rec = r.record
        all_pts_m: List[Tuple[float, float]] = []
        all_pts_m.extend(list(r.user_polygon_m))
        all_pts_m.extend(list(r.trimmed_polygon_exterior_m))
        if not all_pts_m:
            continue

        xs = [p[0] for p in all_pts_m]
        ys = [p[1] for p in all_pts_m]
        min_x_m = min(xs) - pad_m
        max_x_m = max(xs) + pad_m
        min_y_m = min(ys) - pad_m
        max_y_m = max(ys) + pad_m

        x0 = max(0, int(round(min_x_m / px_to_m)))
        y0 = max(0, int(round(min_y_m / px_to_m)))
        x1 = min(W, int(round(max_x_m / px_to_m)))
        y1 = min(H, int(round(max_y_m / px_to_m)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue

        crop_gray = image_gray[y0:y1, x0:x1]
        crop_h, crop_w = crop_gray.shape[:2]
        upscale = max(1.0, zoom_min_canvas_px / max(crop_w, crop_h))
        if upscale > 1.0:
            new_w = int(round(crop_w * upscale))
            new_h = int(round(crop_h * upscale))
            crop_gray = cv2.resize(crop_gray, (new_w, new_h),
                                   interpolation=cv2.INTER_CUBIC)
            crop_w, crop_h = new_w, new_h
        crop_bgr = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
        crop_bgr = cv2.addWeighted(crop_bgr, 0.85,
                                   np.full_like(crop_bgr, 255), 0.15, 0)

        def _to_canvas(pts_m: Sequence[Tuple[float, float]]) -> np.ndarray:
            if not pts_m:
                return np.zeros((0, 2), dtype=np.int32)
            arr = np.asarray(pts_m, dtype=np.float64) / px_to_m
            arr[:, 0] = (arr[:, 0] - x0) * upscale
            arr[:, 1] = (arr[:, 1] - y0) * upscale
            return np.round(arr).astype(np.int32)

        # User label (dashed blue).
        user_canvas = _to_canvas(r.user_polygon_m)
        if user_canvas.shape[0] >= 3:
            _draw_polygon(crop_bgr, user_canvas,
                          color=_LABEL_POLY_COLOR,
                          fill_alpha=0.05,
                          line_thickness=2,
                          dashed=True)

        if rec.status == StairStatus.OK:
            bim_canvas = _to_canvas(r.trimmed_polygon_exterior_m)
            hole_canvases = [_to_canvas(h) for h in r.trimmed_polygon_holes_m]
            if bim_canvas.shape[0] >= 3:
                _draw_polygon(crop_bgr, bim_canvas,
                              color=_BIM_POLY_COLOR,
                              fill_alpha=_BIM_FILL_ALPHA,
                              line_thickness=3,
                              dashed=False,
                              holes_canvas_px=hole_canvases)

            step_polys_m = list(r.step_polygons_xy_m)
            if not step_polys_m:
                step_polys_m = _slice_step_polygons_m(
                    trimmed_exterior_m=r.trimmed_polygon_exterior_m,
                    trimmed_holes_m=r.trimmed_polygon_holes_m,
                    record=rec,
                )
            n_drawn = 0
            for step_index, sp in enumerate(step_polys_m):
                if not sp:
                    continue
                sp_canvas = _to_canvas(sp)
                if sp_canvas.shape[0] < 3:
                    continue
                # Filled (alternating) per-step rectangle.
                fill_color = (
                    _STEP_SLICE_COLOR if (step_index % 2 == 0)
                    else _STEP_SLICE_ALT_COLOR
                )
                step_overlay = crop_bgr.copy()
                cv2.fillPoly(step_overlay, [sp_canvas], fill_color)
                cv2.addWeighted(
                    step_overlay, _STEP_SLICE_FILL_ALPHA,
                    crop_bgr, 1.0 - _STEP_SLICE_FILL_ALPHA, 0.0,
                    dst=crop_bgr,
                )
                cv2.polylines(
                    crop_bgr, [sp_canvas], True,
                    _STEP_SLICE_COLOR, 2, cv2.LINE_AA,
                )
                # Step number in the centroid of the slice.
                cx_step = int(sp_canvas[:, 0].mean())
                cy_step = int(sp_canvas[:, 1].mean())
                step_label = str(step_index + 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(step_label, font, 0.7, 2)
                org = (cx_step - tw // 2, cy_step + th // 2)
                cv2.putText(
                    crop_bgr, step_label, org, font, 0.7,
                    _STEP_NUMBER_OUTLINE, 4, cv2.LINE_AA,
                )
                cv2.putText(
                    crop_bgr, step_label, org, font, 0.7,
                    _STEP_NUMBER_COLOR, 2, cv2.LINE_AA,
                )
                n_drawn += 1

            # Per-Hough segment: every detected tread line drawn thin
            # green. Lets the user see the ink the voter actually voted
            # on, not just the cluster centroids.
            for line in r.hough_tread_lines_xy_m:
                if len(line) != 2:
                    continue
                p0 = _to_canvas([line[0]])
                p1 = _to_canvas([line[1]])
                if p0.shape[0] and p1.shape[0]:
                    cv2.line(
                        crop_bgr,
                        (int(p0[0, 0]), int(p0[0, 1])),
                        (int(p1[0, 0]), int(p1[0, 1])),
                        _HOUGH_TREAD_COLOR,
                        2, cv2.LINE_AA,
                    )

            # Stringers (wangen) — two dim-magenta longitudinal lines.
            for line in r.stringer_lines_xy_m:
                if len(line) != 2:
                    continue
                p0 = _to_canvas([line[0]])
                p1 = _to_canvas([line[1]])
                if p0.shape[0] and p1.shape[0]:
                    cv2.line(
                        crop_bgr,
                        (int(p0[0, 0]), int(p0[0, 1])),
                        (int(p1[0, 0]), int(p1[0, 1])),
                        _STRINGER_COLOR,
                        3, cv2.LINE_AA,
                    )

            # Cut-lines — bold orange-red strokes at the flight end.
            for line in r.cutlines_xy_m:
                if len(line) != 2:
                    continue
                p0 = _to_canvas([line[0]])
                p1 = _to_canvas([line[1]])
                if p0.shape[0] and p1.shape[0]:
                    cv2.line(
                        crop_bgr,
                        (int(p0[0, 0]), int(p0[0, 1])),
                        (int(p1[0, 0]), int(p1[0, 1])),
                        _CUTLINE_COLOR,
                        4, cv2.LINE_AA,
                    )

            # Tread centrelines from the actual density peaks (drawn
            # last so they overlay everything as the strongest cue).
            for line in r.tread_centerlines_xy_m:
                if len(line) != 2:
                    continue
                p0 = _to_canvas([line[0]])
                p1 = _to_canvas([line[1]])
                if p0.shape[0] and p1.shape[0]:
                    cv2.line(
                        crop_bgr,
                        (int(p0[0, 0]), int(p0[0, 1])),
                        (int(p1[0, 0]), int(p1[0, 1])),
                        _TREAD_CENTRELINE_COLOR,
                        3, cv2.LINE_AA,
                    )

            # Direction-up arrow.
            cx_canvas = (rec.centroid_xy_m[0] / px_to_m - x0) * upscale
            cy_canvas = (rec.centroid_xy_m[1] / px_to_m - y0) * upscale
            ux, uy = rec.direction_up_xy
            norm = math.hypot(ux, uy)
            if norm > 1e-9 and rec.length_m > 0.0:
                ux, uy = ux / norm, uy / norm
                arrow_len = (rec.length_m * 0.35 / px_to_m) * upscale
                p_start = (int(round(cx_canvas)), int(round(cy_canvas)))
                p_end = (
                    int(round(cx_canvas + ux * arrow_len)),
                    int(round(cy_canvas + uy * arrow_len)),
                )
                cv2.arrowedLine(crop_bgr, p_start, p_end,
                                _DIRECTION_COLOR, 4, cv2.LINE_AA, tipLength=0.22)
        else:
            n_drawn = 0

        # Header strip — extended with voter-source + confidence so the
        # user can tell at a glance whether n_steps came from Hough,
        # density, or geometric fallback.
        hdr_h = 84
        header = np.full((hdr_h, crop_w, 3), 30, dtype=np.uint8)
        title = f"{rec.label_id}  [{rec.status.value}]"
        if rec.status == StairStatus.OK:
            extras = [
                f"L={rec.length_m:.2f}m W={rec.width_m:.2f}m",
                f"n_steps={rec.n_steps}",
            ]
            if rec.rise_m and rec.going_m:
                extras.append(
                    f"rise/going={rec.rise_m * 100:.0f}/{rec.going_m * 100:.0f}cm"
                )
            extras.append(
                "WALL_SUB:YES" if r.wall_subtraction_used else "WALL_SUB:no-op"
            )
            extras.append(f"steps_drawn={n_drawn}/{rec.n_steps}")
            if any("direction_up_source=arrow" in n for n in rec.notes):
                extras.append("DIR=arrow")
            elif any("direction_up_overridden_by_cut_line" in n for n in rec.notes):
                extras.append("DIR=cut_line")
            elif any("direction_up_default_used" in n for n in rec.notes):
                extras.append("DIR=DEFAULT")
            elif any("direction_up_source=density" in n for n in rec.notes):
                extras.append("DIR=density")
            subtitle = "  |  ".join(extras)
            # Third row: voter provenance + confidence.
            voters = r.n_steps_voters or {}
            third_line = (
                f"voter: {r.n_steps_source} | "
                f"conf={r.n_steps_confidence:.2f} | "
                f"H={voters.get('hough')} D={voters.get('density')} "
                f"DIN={voters.get('din')} | "
                f"stringers={len(r.stringer_lines_xy_m)}/2 | "
                f"cutlines={len(r.cutlines_xy_m)} | "
                f"hough_treads={len(r.hough_tread_lines_xy_m)}"
            )
        else:
            subtitle = "(no BIM footprint — only user label drawn)"
            third_line = ""
        cv2.putText(header, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(header, subtitle, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1, cv2.LINE_AA)
        if third_line:
            cv2.putText(header, third_line, (12, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (190, 220, 255), 1, cv2.LINE_AA)

        canvas = np.vstack([header, crop_bgr])
        out_path = output_dir / f"stair_{rec.label_id}_label_vs_bim.png"
        cv2.imwrite(str(out_path), canvas)
        written.append(out_path)

    return written


def render_bim_only_per_stair_zooms(
    *,
    image_gray: np.ndarray,
    px_to_m: float,
    results: Sequence[StairPipelineResult],
    output_dir: Path,
    pad_m: float = 1.5,
    zoom_min_canvas_px: int = 1200,
) -> List[Path]:
    """Render one PNG per stair showing ONLY the BIM output.

    Three panels side by side:
        * left  — top-down view: per-step polygons (filled, numbered),
                  stair outline (thick magenta), direction-up arrow.
                  NO user bbox, NO Hough lines, NO tread centerlines —
                  only what actually gets extruded into the BIM.
        * mid   — top-down on plain background so the polygons read
                  without competing plan ink.
        * right — side-view of the staircase profile: stacked rectangles
                  showing rise × going per step, the way the BIM mesh
                  is built layer by layer.

    Plus a parameter banner at the top with the actual BIM dimensions:
    n_steps, rise, going, length, width, total_height.
    """
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    H, W = image_gray.shape[:2]

    for r in results:
        rec = r.record
        if rec.status != StairStatus.OK:
            continue
        # ---- common geometry frame -----------------------------------
        all_pts_m: List[Tuple[float, float]] = []
        all_pts_m.extend(list(r.user_polygon_m))
        # Add per-step polygon vertices so the crop fits whatever the
        # BIM actually contains (which may be tighter than the bbox).
        for sp in r.step_polygons_xy_m:
            all_pts_m.extend(list(sp))
        if not all_pts_m:
            continue
        xs = [p[0] for p in all_pts_m]
        ys = [p[1] for p in all_pts_m]
        min_x_m = min(xs) - pad_m
        max_x_m = max(xs) + pad_m
        min_y_m = min(ys) - pad_m
        max_y_m = max(ys) + pad_m
        x0 = max(0, int(round(min_x_m / px_to_m)))
        y0 = max(0, int(round(min_y_m / px_to_m)))
        x1 = min(W, int(round(max_x_m / px_to_m)))
        y1 = min(H, int(round(max_y_m / px_to_m)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue

        crop_gray = image_gray[y0:y1, x0:x1]
        crop_h, crop_w = crop_gray.shape[:2]
        upscale = max(1.0, zoom_min_canvas_px / max(crop_w, crop_h))
        if upscale > 1.0:
            new_w = int(round(crop_w * upscale))
            new_h = int(round(crop_h * upscale))
            crop_gray = cv2.resize(
                crop_gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC,
            )
            crop_w, crop_h = new_w, new_h

        # ---- panel 1: plan + BIM ------------------------------------
        plan_bgr = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
        plan_bgr = cv2.addWeighted(
            plan_bgr, 0.45, np.full_like(plan_bgr, 255), 0.55, 0,
        )

        def _to_canvas(pts_m: Sequence[Tuple[float, float]]) -> np.ndarray:
            if not pts_m:
                return np.zeros((0, 2), dtype=np.int32)
            arr = np.asarray(pts_m, dtype=np.float64) / px_to_m
            arr[:, 0] = (arr[:, 0] - x0) * upscale
            arr[:, 1] = (arr[:, 1] - y0) * upscale
            return np.round(arr).astype(np.int32)

        # Stair outline derived from the actual per-step polygons (the
        # convex hull of all step vertices). This is the polygon that
        # ends up extruded in the BIM.
        all_step_pts: List[Tuple[float, float]] = []
        for sp in r.step_polygons_xy_m:
            all_step_pts.extend(list(sp))
        if all_step_pts:
            arr_pts = np.asarray(all_step_pts, dtype=np.float32)
            hull = cv2.convexHull(arr_pts.reshape(-1, 1, 2))
            hull_m = [(float(p[0][0]), float(p[0][1])) for p in hull]
            outline_canvas = _to_canvas(hull_m)
        else:
            outline_canvas = _to_canvas(r.trimmed_polygon_exterior_m)

        for panel_bgr, draw_plan in ((plan_bgr, True),
                                     (None, False)):
            pass  # placeholder for clarity
        # Render two top-down panels: with-plan and clean-background.
        panels: List[np.ndarray] = []
        for include_plan in (True, False):
            if include_plan:
                tpanel = plan_bgr.copy()
            else:
                tpanel = np.full(
                    (crop_h, crop_w, 3), 245, dtype=np.uint8,
                )
            # Numbered, filled, alternating-colour step polygons.
            for step_index, sp in enumerate(r.step_polygons_xy_m):
                if not sp:
                    continue
                sp_canvas = _to_canvas(sp)
                if sp_canvas.shape[0] < 3:
                    continue
                fill_color = (
                    (60, 140, 240) if (step_index % 2 == 0)
                    else (100, 180, 250)
                )
                overlay = tpanel.copy()
                cv2.fillPoly(overlay, [sp_canvas], fill_color)
                cv2.addWeighted(
                    overlay, 0.55, tpanel, 0.45, 0.0, dst=tpanel,
                )
                cv2.polylines(
                    tpanel, [sp_canvas], True, (40, 90, 200), 2,
                    cv2.LINE_AA,
                )
                cx_step = int(sp_canvas[:, 0].mean())
                cy_step = int(sp_canvas[:, 1].mean())
                step_label = str(step_index + 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(step_label, font, 0.8, 2)
                org = (cx_step - tw // 2, cy_step + th // 2)
                cv2.putText(
                    tpanel, step_label, org, font, 0.8,
                    (245, 245, 245), 5, cv2.LINE_AA,
                )
                cv2.putText(
                    tpanel, step_label, org, font, 0.8,
                    (15, 15, 15), 2, cv2.LINE_AA,
                )

            # BIM outline (thick magenta).
            if outline_canvas.shape[0] >= 3:
                cv2.polylines(
                    tpanel, [outline_canvas], True,
                    _BIM_POLY_COLOR, 3, cv2.LINE_AA,
                )

            # Direction-up arrow (red, thick).
            cx_canvas = (rec.centroid_xy_m[0] / px_to_m - x0) * upscale
            cy_canvas = (rec.centroid_xy_m[1] / px_to_m - y0) * upscale
            ux, uy = rec.direction_up_xy
            norm = math.hypot(ux, uy)
            if norm > 1e-9 and rec.length_m > 0.0:
                ux, uy = ux / norm, uy / norm
                arrow_len = (rec.length_m * 0.40 / px_to_m) * upscale
                p_start = (int(round(cx_canvas)), int(round(cy_canvas)))
                p_end = (
                    int(round(cx_canvas + ux * arrow_len)),
                    int(round(cy_canvas + uy * arrow_len)),
                )
                cv2.arrowedLine(
                    tpanel, p_start, p_end, _DIRECTION_COLOR, 5,
                    cv2.LINE_AA, tipLength=0.22,
                )
            panels.append(tpanel)

        # ---- panel 3: side-view profile -----------------------------
        side_panel_w = max(crop_h, 600)  # square-ish
        side_panel = np.full(
            (crop_h, side_panel_w, 3), 250, dtype=np.uint8,
        )
        if rec.n_steps and rec.going_m and rec.rise_m:
            n = int(rec.n_steps)
            going_m = float(rec.going_m)
            rise_m = float(rec.rise_m)
            total_w_m = n * going_m + 0.5  # +50cm padding for ground
            total_h_m = n * rise_m + 0.5
            # Fit profile into the panel with 10 % padding on all sides.
            margin = 40
            avail_w = side_panel_w - 2 * margin
            avail_h = crop_h - 2 * margin
            scale = min(avail_w / total_w_m, avail_h / total_h_m)
            origin_x = margin
            origin_y = crop_h - margin
            # Ground line.
            cv2.line(
                side_panel,
                (margin, origin_y),
                (side_panel_w - margin, origin_y),
                (60, 60, 60), 2, cv2.LINE_AA,
            )
            # Each step rectangle: x from i*going to (i+1)*going, y from
            # i*rise down to (i+1)*rise above ground.
            for i in range(n):
                lx = origin_x + int(round(i * going_m * scale))
                rx = origin_x + int(round((i + 1) * going_m * scale))
                ty = origin_y - int(round((i + 1) * rise_m * scale))
                by = origin_y - int(round(i * rise_m * scale))
                fill_color = (
                    (60, 140, 240) if (i % 2 == 0) else (100, 180, 250)
                )
                cv2.rectangle(
                    side_panel, (lx, ty), (rx, by), fill_color, -1,
                )
                cv2.rectangle(
                    side_panel, (lx, ty), (rx, by), (40, 90, 200), 2,
                    cv2.LINE_AA,
                )
                # Step number on top tread.
                font = cv2.FONT_HERSHEY_SIMPLEX
                label = str(i + 1)
                (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
                cx = (lx + rx) // 2
                cv2.putText(
                    side_panel, label, (cx - tw // 2, ty - 6),
                    font, 0.55, (40, 40, 40), 1, cv2.LINE_AA,
                )
            # Dimension annotations (rise + going) for step 1.
            ax = origin_x + int(round(going_m * scale))
            ay = origin_y - int(round(rise_m * scale))
            cv2.putText(
                side_panel,
                f"going={going_m*100:.0f}cm",
                (margin + 10, origin_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA,
            )
            cv2.putText(
                side_panel,
                f"rise={rise_m*100:.0f}cm",
                (ax + 8, ay + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA,
            )
            # Total height annotation on the right.
            top_y = origin_y - int(round(n * rise_m * scale))
            cv2.line(
                side_panel,
                (side_panel_w - margin + 4, origin_y),
                (side_panel_w - margin + 4, top_y),
                (40, 40, 40), 2, cv2.LINE_AA,
            )
            cv2.putText(
                side_panel,
                f"H_total={rec.total_height_m:.2f}m",
                (side_panel_w - margin - 130, top_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                side_panel, "(no step geometry)",
                (margin, crop_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 60, 60), 2,
                cv2.LINE_AA,
            )

        # ---- combine panels horizontally ----------------------------
        # Separator strips between panels.
        sep = np.full((crop_h, 6, 3), 30, dtype=np.uint8)
        body = np.hstack([panels[0], sep, panels[1], sep, side_panel])

        # ---- header with BIM parameters -----------------------------
        hdr_h = 78
        header = np.full((hdr_h, body.shape[1], 3), 28, dtype=np.uint8)
        title = f"{rec.label_id}  —  BIM-only view"
        cv2.putText(
            header, title, (16, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA,
        )
        line2 = (
            f"n_steps={rec.n_steps}  |  "
            f"rise={rec.rise_m*100:.0f}cm  |  going={rec.going_m*100:.0f}cm  |  "
            f"L={rec.length_m:.2f}m  W={rec.width_m:.2f}m  |  "
            f"H_total={rec.total_height_m:.2f}m  "
            f"({rec.height_source})"
        ) if rec.rise_m and rec.going_m else f"n_steps={rec.n_steps}"
        cv2.putText(
            header, line2, (16, 52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
        )
        line3 = (
            "left: BIM-on-plan  |  mid: BIM only  |  "
            "right: side view (rise × going stack)"
        )
        cv2.putText(
            header, line3, (16, 72),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1, cv2.LINE_AA,
        )

        canvas = np.vstack([header, body])
        out_path = output_dir / f"stair_{rec.label_id}_bim_only.png"
        cv2.imwrite(str(out_path), canvas)
        written.append(out_path)

    return written


def render_stairs_diagnostic(
    *,
    image_gray: np.ndarray,
    px_to_m: float,
    results: Sequence[StairPipelineResult],
    bboxes_px: Sequence[Optional[tuple]],
    output_path: Path,
    max_width_px: int = 1600,
) -> Path:
    """Render the full diagnostic PNG and write it to ``output_path``.

    Args:
        image_gray: the rendered plan grayscale image (uint8).
        px_to_m: metre-per-pixel scale of ``image_gray``.
        results: orchestrator outputs (one per stair label).
        bboxes_px: per-result optional ``(x0, y0, x1, y1)`` bbox in
            render-px. Indexed in lock-step with ``results``. Used to
            place the overlay glyphs and text annotations.
        output_path: target ``.png`` path.
        max_width_px: top render is downscaled to at most this width.

    Returns:
        ``output_path`` for chaining.
    """
    import cv2

    if image_gray is None:
        # Caller passed no image — write a minimal placeholder.
        placeholder = np.full((200, 800, 3), 240, dtype=np.uint8)
        _put_text_with_outline(
            placeholder, "stairs_diagnostic: no image provided", (20, 100),
            font_scale=0.6, thickness=2,
        )
        cv2.imwrite(str(output_path), placeholder)
        return output_path

    top, scale_for_output = _scale_image_to_max_width(image_gray, max_width_px)
    top_bgr = cv2.cvtColor(top, cv2.COLOR_GRAY2BGR)

    for r, bbox in zip(results, bboxes_px):
        _overlay_stair(
            top_bgr,
            result=r,
            bbox_px=bbox,
            scale_for_output=scale_for_output,
            px_to_m=px_to_m,
        )

    # Profile strip below the top render.
    profile_rows = []
    for r in results:
        if r.pixel_details is None or r.pixel_details.density_profile is None:
            continue
        prof_img = _draw_profile(r.pixel_details.density_profile)
        # Prepend the stair id in a left margin band.
        band = np.full((_PROFILE_HEIGHT_PX, 90, 3), 245, dtype=np.uint8)
        _put_text_with_outline(
            band, r.record.label_id, (8, _PROFILE_HEIGHT_PX // 2 + 4),
            font_scale=0.45,
        )
        # Status colour pad on the far right.
        status_color = (
            _STATUS_COLOR_OK if r.record.status == StairStatus.OK
            else _STATUS_COLOR_SKIP
        )
        status_pad = np.full((_PROFILE_HEIGHT_PX, 14, 3), status_color, dtype=np.uint8)
        row = np.hstack([band, prof_img, status_pad])
        profile_rows.append(row)

    if profile_rows:
        # Pad profile rows to the width of the top render with a thin
        # separator band so they stack cleanly.
        target_w = top_bgr.shape[1]
        padded_rows = []
        sep = np.full((4, target_w, 3), 230, dtype=np.uint8)
        padded_rows.append(sep)
        for row in profile_rows:
            if row.shape[1] < target_w:
                pad = np.full(
                    (row.shape[0], target_w - row.shape[1], 3), 250, dtype=np.uint8,
                )
                row = np.hstack([row, pad])
            elif row.shape[1] > target_w:
                row = row[:, :target_w]
            padded_rows.append(row)
            padded_rows.append(sep)
        bottom = np.vstack(padded_rows)
        canvas = np.vstack([top_bgr, bottom])
    else:
        canvas = top_bgr

    # Title header
    header = np.full((40, canvas.shape[1], 3), 30, dtype=np.uint8)
    n_ok = sum(1 for r in results if r.record.status == StairStatus.OK)
    n_skip = sum(1 for r in results if r.record.status != StairStatus.OK)
    title = (
        f"Stairs Diagnostic — {len(results)} stair(s): {n_ok} OK, {n_skip} skipped "
        f"(px_to_m={px_to_m:.5f}, downscale={scale_for_output:.2f}x)"
    )
    import cv2
    cv2.putText(header, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (245, 245, 245), 1, cv2.LINE_AA)
    canvas = np.vstack([header, canvas])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path
