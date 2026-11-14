"""High-level orchestrator for the stair-extraction pipeline.

Glue code that consumes the foundation building blocks and produces a
list of fully-populated ``StairRecord`` plus per-record ``StairMesh``.
The function is pure (no scene-integration side effects); the caller
(``scripts/add_stairs_to_bim.py`` for production, ``scripts/
diagnose_stairs.py`` for local dev) is responsible for adding the
meshes to the BIM scene and serialising the records.

**Decoupled from the perfect_bim pipeline.** The original
``perfect_bim`` flow (door pipeline + window/door-bridge merge)
remains untouched. Stairs are extracted and added as an independent
post-step that loads the existing ``bim_complete_*.glb``, appends
stair meshes, and rewrites the GLB.

**No DIN enforcement.** Consistent with the rest of the perfect_bim
algorithms (which do not enforce DIN at extraction time), the
orchestrator does NOT call ``validate_din``. The DIN module remains
available as a library helper for downstream consumers (QualityPage
BIM-Audit, IFC-export) but is not part of the extraction loop.

Call sequence per annotation entry (only ``class`` ∈ {stairs,
stairs_spiral, elevator}):

    1. ``classify_stair`` -> STRAIGHT | UNSUPPORTED
       UNSUPPORTED -> emit record with ``SKIPPED_UNSUPPORTED_CLASS_V1``.
    2. ``extract_stair_pixels``: pixel-derived footprint +
       n_steps + density profile + Otsu-adaptive threshold.
       Falls back to ``extract_stair_footprint`` (polygon PCA) if the
       pixel pass raises ``PixelExtractionUnavailable``.
    3. ``detect_direction_up``: density-profile-based voter.
    4. ``build_wall_envelope_from_polys``: filter stairs whose
       centroid sits outside the building convex hull -> emit
       ``SKIPPED_OUTSIDE_WALLS`` record.
    5. ``resolve_total_height_m``: annotation > config > pipeline-default.
    6. ``derive_step_layout`` if the pixel pass did not return a usable
       n_steps.
    7. ``build_stair_mesh``: trimesh.Trimesh ready to add to the scene.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .footprint import extract_stair_footprint
from .mesh_builder import (
    StairMesh,
    build_stair_mesh,
    build_stair_mesh_from_polygon,
)
from .records import StairRecord, StairStatus
from .stair_footprint_polygon import (
    compute_stair_polygon,
    pca_axes_from_polygon,
)
from .stair_classifier import (
    STAIR_ALGORITHM_UNSUPPORTED,
    classify_stair,
    resolve_ifc_predefined_type,
)
from .stair_arrow_detector import detect_arrow_direction
from .stair_din import (
    derive_step_layout,
    resolve_total_height_m,
)
from .stair_direction import build_walking_line_xy, detect_direction_up
from .stair_pixel_extractor import (
    PixelExtractionUnavailable,
    StairPixelDetails,
    extract_stair_pixels,
)
from .wall_envelope import (
    build_wall_envelope_from_polys,
    is_centroid_inside,
)

DEFAULT_CEILING_BUILDUP_M = 0.15  # Wand-Hoehe + Deckenaufbau = Geschosshoehe
DEFAULT_PIPELINE_FLOOR_TO_FLOOR_M = 2.95  # 2.80 lichte + 0.15 Aufbau
MIN_FOOTPRINT_M = 0.30
MAX_FOOTPRINT_M = 50.0


# Stair label-class set the orchestrator inspects from annotations.json.
# Catches the supported V1 class plus the V2-deferred classes (so they
# end up in the record stream with SKIPPED_UNSUPPORTED_CLASS_V1 instead
# of silently disappearing).
STAIR_LABEL_CLASSES = {"stairs", "stairs_spiral", "elevator"}


@dataclass
class StairPipelineResult:
    """Per-stair orchestrator output. The caller picks what to do."""

    record: StairRecord
    mesh: Optional[StairMesh] = None
    pixel_details: Optional[StairPixelDetails] = None
    notes: List[str] = field(default_factory=list)
    # Tag 9 — the trimmed footprint polygon in plan-metre coordinates
    # (PDF Y points down). This is the polygon the mesh-builder sliced
    # the per-step boxes from; expose it so diagnostic renderers can
    # draw the actual BIM footprint side-by-side with the user label.
    trimmed_polygon_exterior_m: List[Tuple[float, float]] = field(default_factory=list)
    trimmed_polygon_holes_m: List[List[Tuple[float, float]]] = field(default_factory=list)
    user_polygon_m: List[Tuple[float, float]] = field(default_factory=list)
    wall_subtraction_used: bool = False
    # Density-peak positions: each tread's centreline expressed as two
    # endpoints in plan-XY metres (centreline = short-axis chord across
    # the stair polygon at that long-axis position). Populated when the
    # pixel extractor returned per-tread positions; empty otherwise.
    tread_centerlines_xy_m: List[List[Tuple[float, float]]] = field(default_factory=list)
    # Per-step polygon footprints (one per actual tread, sliced at the
    # half-way positions between consecutive density peaks). Same length
    # as ``tread_centerlines_xy_m``. Used by the diagnostic overlay AND
    # by the mesh builder so the BIM strips align with the plan ink
    # instead of being equally-spaced.
    step_polygons_xy_m: List[List[Tuple[float, float]]] = field(default_factory=list)

    # ---- Hough-line-extractor outputs (Sprint v2) ----
    # Per-tread Hough segment endpoints in plan-XY metres (one entry per
    # detected cluster, longest segment per cluster). Drawn green-bold
    # by the diagnostic renderer to show the actual plan ink the voter
    # locked onto.
    hough_tread_lines_xy_m: List[List[Tuple[float, float]]] = field(default_factory=list)
    # Stringer (wangen) pair: two long lines along the run, parallel to
    # the long axis. Two entries when both stringers were detected,
    # empty for open stairs.
    stringer_lines_xy_m: List[List[Tuple[float, float]]] = field(default_factory=list)
    # Cut lines (Höhensprung): the thick stroke that marks the visible
    # end of the flight. Zero or one entries typical; two on stairs
    # where both ends are cut.
    cutlines_xy_m: List[List[Tuple[float, float]]] = field(default_factory=list)
    # Aggregator output for tread count: which voter drove the answer
    # and how confident we are.
    n_steps_source: str = "none"
    n_steps_confidence: float = 0.0
    n_steps_voters: Dict[str, Optional[int]] = field(default_factory=dict)


def _poly_from_annotation(
    ann: Dict[str, Any],
    scale_xy: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """Materialise an annotation's polygon/bbox in render-pixel coordinates.

    Mirrors the helper in ``scripts/merge_doors_and_windows_bim.py``
    so the call-sites stay consistent.
    """
    sx, sy = scale_xy
    pts = ann.get("points") or []
    if pts:
        return [(float(p[0]) * sx, float(p[1]) * sy) for p in pts]
    bbox = ann.get("bbox")
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = (float(v) for v in bbox)
        return [
            (x0 * sx, y0 * sy),
            (x1 * sx, y0 * sy),
            (x1 * sx, y1 * sy),
            (x0 * sx, y1 * sy),
        ]
    return []


def _poly_to_m(
    poly_px: Sequence[Tuple[float, float]],
    px_to_m: float,
) -> List[Tuple[float, float]]:
    """Convert pixel polygon to plan-metre polygon."""
    return [(x * px_to_m, y * px_to_m) for x, y in poly_px]


def _line_pixel_ratio_m(
    line_m: Tuple[Tuple[float, float], Tuple[float, float]],
    image_gray: Optional[np.ndarray],
    dark_threshold: int,
    px_to_m: float,
    *,
    n_samples: int = 80,
    neighborhood_px: int = 2,
) -> float:
    """Fraction of sample points along ``line_m`` whose
    ``neighborhood_px``-radius image neighbourhood contains dark ink.

    Returns 0.0 when no image is available. A perfect Hough-detected
    ink stroke yields a ratio close to 1.0; a synthetic line through
    empty floor drops to near 0.0. Used as the hard "line-on-pixels"
    criterion (≥0.95 ⇒ accept; < ⇒ reject).
    """
    if image_gray is None or px_to_m <= 0.0:
        return 0.0
    H_img, W_img = image_gray.shape[:2]
    p1_px = (line_m[0][0] / px_to_m, line_m[0][1] / px_to_m)
    p2_px = (line_m[1][0] / px_to_m, line_m[1][1] / px_to_m)
    ts = np.linspace(0.0, 1.0, max(2, n_samples), dtype=np.float64)
    xs = p1_px[0] + (p2_px[0] - p1_px[0]) * ts
    ys = p1_px[1] + (p2_px[1] - p1_px[1]) * ts
    n_dark = 0
    n_total = 0
    nbr = max(0, neighborhood_px)
    for xf, yf in zip(xs, ys):
        ix = int(round(xf))
        iy = int(round(yf))
        if not (0 <= ix < W_img and 0 <= iy < H_img):
            continue
        n_total += 1
        x_lo = max(0, ix - nbr)
        x_hi = min(W_img, ix + nbr + 1)
        y_lo = max(0, iy - nbr)
        y_hi = min(H_img, iy + nbr + 1)
        block = image_gray[y_lo:y_hi, x_lo:x_hi]
        if (block < dark_threshold).any():
            n_dark += 1
    if n_total == 0:
        return 0.0
    return n_dark / n_total


def _build_pixel_step_polygons_from_ink(
    *,
    image_gray: np.ndarray,
    px_to_m: float,
    dark_threshold: int,
    user_polygon_px: Sequence[Tuple[float, float]],
    confidence_field,
    tread_lines_m: Sequence[Sequence[Tuple[float, float]]],
    origin_xy_m: Tuple[float, float],
    long_unit: Tuple[float, float],
    short_unit: Tuple[float, float],
    total_height_m: float,
    confidence_floor: float = 0.25,
    min_step_width_m: float = 0.05,
    min_step_going_m: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Pixel-first per-step polygons for a single-run stair (incl. L).

    Every step polygon is an axis-aligned rectangle in the stair\'s
    (long, short) frame whose four corners sit on real ink:
      * the long-axis edges are the detected tread Hough lines (step
        risers) -- already validated to lie on plan ink,
      * the short-axis edges are the [1st, 99th] percentile of the
        ACTUAL ink extent inside that tread band, so the L-wing\'s
        taller steps and the upper band\'s shorter steps each get their
        true height with no synthetic continuation.

    No infinite-line intersection, no user-polygon edge, no morphology
    bulge -- so the result is resolution-stable and free of the diagonal
    / skew artefacts the v5 outline-snap produced.

    Returns ``None`` when the ink region is too sparse to trust (caller
    falls back to projection slicing).
    """
    import cv2  # noqa: PLC0415

    if image_gray is None or px_to_m <= 0.0 or len(tread_lines_m) < 2:
        return None
    H_img, W_img = image_gray.shape[:2]
    low_x, low_y = origin_xy_m
    ux, uy = long_unit
    sx, sy = short_unit

    # ---- Stage 1: ink pixels attributed to THIS stair (sub-window) ----
    field_sub = getattr(confidence_field, "field_sub", None)
    if (
        confidence_field is not None
        and field_sub is not None
        and getattr(field_sub, "size", 0) > 1
    ):
        x0 = int(confidence_field.x_offset_px)
        y0 = int(confidence_field.y_offset_px)
        sh, sw = field_sub.shape
        x1 = min(W_img, x0 + sw)
        y1 = min(H_img, y0 + sh)
        gray_sub = image_gray[y0:y1, x0:x1]
        conf_sub = field_sub[: y1 - y0, : x1 - x0]
        region = (gray_sub < dark_threshold) & (conf_sub >= confidence_floor)
    else:
        arr = np.asarray(user_polygon_px, dtype=np.float64)
        margin_px = max(8, int(round(0.30 / max(px_to_m, 1e-9))))
        x0 = max(0, int(arr[:, 0].min()) - margin_px)
        x1 = min(W_img, int(arr[:, 0].max()) + margin_px)
        y0 = max(0, int(arr[:, 1].min()) - margin_px)
        y1 = min(H_img, int(arr[:, 1].max()) + margin_px)
        if x1 <= x0 or y1 <= y0:
            return None
        gray_sub = image_gray[y0:y1, x0:x1]
        poly_mask = np.zeros(gray_sub.shape[:2], dtype=np.uint8)
        pts = np.array(
            [[int(round(px - x0)), int(round(py - y0))]
             for px, py in user_polygon_px],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.fillPoly(poly_mask, [pts], 255)
        region = (gray_sub < dark_threshold) & (poly_mask > 0)

    ys_idx, xs_idx = np.nonzero(region)
    if xs_idx.size < 50:
        return None

    # Pixel -> world metres -> (u, s) projections.
    xm = (xs_idx.astype(np.float64) + x0) * px_to_m
    ym = (ys_idx.astype(np.float64) + y0) * px_to_m
    u_arr = (xm - low_x) * ux + (ym - low_y) * uy
    s_arr = (xm - low_x) * sx + (ym - low_y) * sy

    # ---- band boundaries along the long axis = tread risers ----
    def _u_of_point(p):
        return (p[0] - low_x) * ux + (p[1] - low_y) * uy

    tread_u = sorted(
        0.5 * (_u_of_point(T[0]) + _u_of_point(T[1])) for T in tread_lines_m
    )
    going_est = (tread_u[-1] - tread_u[0]) / max(len(tread_u) - 1, 1)
    if going_est <= 0.0:
        going_est = 0.29

    u_lo_region = float(np.percentile(u_arr, 0.5))
    u_hi_region = float(np.percentile(u_arr, 99.5))

    # Treads are step fronts. Bands between consecutive treads are the
    # tread surfaces. Add a ground-contact band below the down-end tread
    # when the ink region extends there (the lowest step has no riser
    # line of its own). The up-end (landing) gets no extra step.
    bounds = list(tread_u)
    if u_lo_region < tread_u[0] - 0.30 * going_est:
        bounds = [max(u_lo_region, tread_u[0] - 1.5 * going_est)] + bounds
    if u_hi_region > tread_u[-1] + 0.70 * going_est:
        bounds = bounds + [min(u_hi_region, tread_u[-1] + 1.5 * going_est)]

    def _world(u_val, s_val):
        return (low_x + u_val * ux + s_val * sx,
                low_y + u_val * uy + s_val * sy)

    step_polygons: List[List[Tuple[float, float]]] = []
    widths: List[float] = []
    cx_acc = cy_acc = 0.0
    n_acc = 0
    for ua, ub in zip(bounds[:-1], bounds[1:]):
        if (ub - ua) < min_step_going_m:
            continue
        sel = (u_arr >= ua - 1e-6) & (u_arr <= ub + 1e-6)
        if int(sel.sum()) < 15:
            continue
        s_lo = float(np.percentile(s_arr[sel], 1.0))
        s_hi = float(np.percentile(s_arr[sel], 99.0))
        if (s_hi - s_lo) < min_step_width_m:
            continue
        rect = [
            _world(ua, s_lo), _world(ub, s_lo),
            _world(ub, s_hi), _world(ua, s_hi),
        ]
        step_polygons.append(rect)
        widths.append(s_hi - s_lo)
        for v in rect:
            cx_acc += v[0]
            cy_acc += v[1]
            n_acc += 1

    if not step_polygons or n_acc == 0:
        return None

    widths_sorted = sorted(widths)
    width_median = widths_sorted[len(widths_sorted) // 2]
    length_m = float(bounds[-1] - bounds[0])
    n_phys = len(step_polygons)
    going_m = length_m / n_phys if n_phys > 0 else None
    rise_m = total_height_m / n_phys if n_phys > 0 else None
    centroid = (cx_acc / n_acc, cy_acc / n_acc)
    width_spread = (widths_sorted[-1] - widths_sorted[0])

    # Tread centrelines for the diagnostic overlay = the detected tread
    # Hough segments themselves (already on ink).
    centerlines = [[(float(T[0][0]), float(T[0][1])),
                    (float(T[1][0]), float(T[1][1]))] for T in tread_lines_m]

    return {
        "step_polygons_xy_m": step_polygons,
        "tread_centerlines_xy_m": centerlines,
        "n_steps": n_phys,
        "length_m": length_m,
        "width_m": float(width_median),
        "going_m": going_m,
        "rise_m": rise_m,
        "centroid_xy_m": centroid,
        "width_spread_m": float(width_spread),
    }


def _stair_ink_us(
    *,
    image_gray: np.ndarray,
    px_to_m: float,
    dark_threshold: int,
    user_polygon_px: Sequence[Tuple[float, float]],
    confidence_field,
    origin_xy_m: Tuple[float, float],
    long_unit: Tuple[float, float],
    short_unit: Tuple[float, float],
    confidence_floor: float = 0.25,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Attribute this stair's ink and return its ``(u, s)`` projections.

    ``u`` is the long-axis offset (metres) from ``origin_xy_m``, ``s``
    the short-axis offset. Mirrors Stage 1 of
    :func:`_build_pixel_step_polygons_from_ink` (confidence sub-window
    when available, else the dilated user-polygon mask) so the
    polygonize builder shares the exact same ink-attribution as the v6
    band builder. Returns ``None`` when the region is too sparse.
    """
    import cv2  # noqa: PLC0415

    if image_gray is None or px_to_m <= 0.0:
        return None
    H_img, W_img = image_gray.shape[:2]
    low_x, low_y = origin_xy_m
    ux, uy = long_unit
    sx, sy = short_unit

    field_sub = getattr(confidence_field, "field_sub", None)
    if (
        confidence_field is not None
        and field_sub is not None
        and getattr(field_sub, "size", 0) > 1
    ):
        x0 = int(confidence_field.x_offset_px)
        y0 = int(confidence_field.y_offset_px)
        sh, sw = field_sub.shape
        x1 = min(W_img, x0 + sw)
        y1 = min(H_img, y0 + sh)
        gray_sub = image_gray[y0:y1, x0:x1]
        conf_sub = field_sub[: y1 - y0, : x1 - x0]
        region = (gray_sub < dark_threshold) & (conf_sub >= confidence_floor)
    else:
        arr = np.asarray(user_polygon_px, dtype=np.float64)
        margin_px = max(8, int(round(0.30 / max(px_to_m, 1e-9))))
        x0 = max(0, int(arr[:, 0].min()) - margin_px)
        x1 = min(W_img, int(arr[:, 0].max()) + margin_px)
        y0 = max(0, int(arr[:, 1].min()) - margin_px)
        y1 = min(H_img, int(arr[:, 1].max()) + margin_px)
        if x1 <= x0 or y1 <= y0:
            return None
        gray_sub = image_gray[y0:y1, x0:x1]
        poly_mask = np.zeros(gray_sub.shape[:2], dtype=np.uint8)
        pts = np.array(
            [[int(round(px - x0)), int(round(py - y0))]
             for px, py in user_polygon_px],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.fillPoly(poly_mask, [pts], 255)
        region = (gray_sub < dark_threshold) & (poly_mask > 0)

    ys_idx, xs_idx = np.nonzero(region)
    if xs_idx.size < 50:
        return None
    xm = (xs_idx.astype(np.float64) + x0) * px_to_m
    ym = (ys_idx.astype(np.float64) + y0) * px_to_m
    u_arr = (xm - low_x) * ux + (ym - low_y) * uy
    s_arr = (xm - low_x) * sx + (ym - low_y) * sy
    return u_arr, s_arr


def _polygonize_steps_from_risers(
    *,
    image_gray: np.ndarray,
    px_to_m: float,
    dark_threshold: int,
    user_polygon_px: Sequence[Tuple[float, float]],
    confidence_field,
    main_riser_lines_m: Sequence[Sequence[Tuple[float, float]]],
    cross_riser_lines_m: Sequence[Sequence[Tuple[float, float]]],
    origin_xy_m: Tuple[float, float],
    long_unit: Tuple[float, float],
    short_unit: Tuple[float, float],
    total_height_m: float,
    confidence_floor: float = 0.25,
    min_step_side_m: float = 0.05,
    min_step_area_m2: float = 0.08 * 0.08,
) -> Optional[Dict[str, Any]]:
    """Planar-partition step decomposition for winder / L / straight runs.

    Generic over geometry: a straight flight has no cross-risers and
    falls out as pure rectangles; a quarter-winder's wing is carved into
    nested L-treads by its cross-risers — same code, no branching.

    Method (validated prototype ``wing_poly3``):

      1. Project the attributed ink + every riser line into the stair's
         ``(u, s)`` frame (``u`` = long axis, ``s`` = short axis) where
         all risers are axis-aligned.
      2. **Pixel-terminated outer hull:** per main-riser band measure the
         actual short-axis ink extent, cluster the band depths into
         discrete levels (so the wing's deep bands and the run's shallow
         bands each get ONE clean depth, free of per-band noise), snap
         each level onto the nearest cross-riser ink line, union the band
         boxes. This is the v6 boundary, de-jagged and pixel-terminated.
      3. **Node + partition:** round the hull exterior and ALL riser
         lines onto a common ~2 cm grid, ``unary_union`` them, and
         ``polygonize`` the result. Coincident boundary/cross-riser
         segments collapse on the shared grid instead of breaking the
         partition (the plan's boundary-vs-cross-riser dedupe).
      4. **Faces -> steps:** keep faces inside the hull with area >=
         ``min_step_area_m2`` and min side >= ``min_step_side_m``. L-treads
         (>= 6 vertices) and rectangles fall out automatically.
      5. Order along the run, transform corners back to world XY.

    Returns ``None`` when the partition is degenerate (caller falls back
    to the v6 band builder, then v2.1 projection slicing).
    """
    from shapely.geometry import (  # noqa: PLC0415
        LineString,
        MultiPolygon,
        Polygon as _ShPoly,
        box as _box,
    )
    from shapely.ops import polygonize, unary_union  # noqa: PLC0415

    if image_gray is None or px_to_m <= 0.0:
        return None
    if len(main_riser_lines_m) < 2:
        return None

    low_x, low_y = origin_xy_m

    # Self-consistent frame: the partition + the pixel-terminated hull
    # assume the main risers are axis-aligned in the (long, short) frame.
    # The caller's ``long_unit`` is the walking direction, which on an
    # L/winder footprint comes from a PCA that picks a DIAGONAL — that
    # would make the axis-aligned risers diagonal in this frame and emit
    # slanted step polygons. So derive the frame directly from the main
    # risers: short axis = their dominant direction, long axis = its
    # perpendicular. Sign is anchored to the caller's long_unit so the
    # run still reads "forward" the same way.
    sdx = sdy = 0.0
    ref: Optional[Tuple[float, float]] = None
    for ln in main_riser_lines_m:
        dx = ln[1][0] - ln[0][0]
        dy = ln[1][1] - ln[0][1]
        n = math.hypot(dx, dy)
        if n < 1e-9:
            continue
        u = (dx / n, dy / n)
        if ref is None:
            ref = u
        elif u[0] * ref[0] + u[1] * ref[1] < 0:
            u = (-u[0], -u[1])
        sdx += u[0]
        sdy += u[1]
    smag = math.hypot(sdx, sdy)
    if smag < 1e-9:
        sx, sy = short_unit
        ux, uy = long_unit
    else:
        sx, sy = sdx / smag, sdy / smag
        ux, uy = -sy, sx
        # Align long-axis sign with the caller's walking direction.
        if ux * long_unit[0] + uy * long_unit[1] < 0:
            ux, uy = -ux, -uy

    ink = _stair_ink_us(
        image_gray=image_gray, px_to_m=px_to_m, dark_threshold=dark_threshold,
        user_polygon_px=user_polygon_px, confidence_field=confidence_field,
        origin_xy_m=origin_xy_m, long_unit=(ux, uy), short_unit=(sx, sy),
        confidence_floor=confidence_floor,
    )
    if ink is None:
        return None
    u_arr, s_arr = ink

    def _to_us(p: Tuple[float, float]) -> Tuple[float, float]:
        return ((p[0] - low_x) * ux + (p[1] - low_y) * uy,
                (p[0] - low_x) * sx + (p[1] - low_y) * sy)

    def _to_world(u_val: float, s_val: float) -> Tuple[float, float]:
        return (low_x + u_val * ux + s_val * sx,
                low_y + u_val * uy + s_val * sy)

    # --- risers in the (u, s) frame ---
    main_us = []  # (u_pos, s_lo, s_hi)
    for ln in main_riser_lines_m:
        a = _to_us(ln[0])
        b = _to_us(ln[1])
        u_pos = 0.5 * (a[0] + b[0])
        main_us.append((u_pos, min(a[1], b[1]), max(a[1], b[1])))
    main_us.sort(key=lambda t: t[0])
    tread_u = [m[0] for m in main_us]

    cross_us = []  # (s_pos, u_lo, u_hi)
    for ln in cross_riser_lines_m:
        a = _to_us(ln[0])
        b = _to_us(ln[1])
        s_pos = 0.5 * (a[1] + b[1])
        cross_us.append((s_pos, min(a[0], b[0]), max(a[0], b[0])))
    cross_s_positions = sorted(c[0] for c in cross_us)

    # --- pixel-terminated outer hull from per-band ink extents ---
    going_est = (tread_u[-1] - tread_u[0]) / max(len(tread_u) - 1, 1)
    if going_est <= 0.0:
        going_est = 0.29
    u_lo_region = float(np.percentile(u_arr, 0.5))
    u_hi_region = float(np.percentile(u_arr, 99.5))
    bounds = list(tread_u)
    if u_lo_region < tread_u[0] - 0.30 * going_est:
        bounds = [max(u_lo_region, tread_u[0] - 1.5 * going_est)] + bounds
    if u_hi_region > tread_u[-1] + 0.70 * going_est:
        bounds = bounds + [min(u_hi_region, tread_u[-1] + 1.5 * going_est)]

    raw_bands = []  # (ua, ub, s_lo, s_hi)
    for ua, ub in zip(bounds[:-1], bounds[1:]):
        if (ub - ua) <= 0.0:
            continue
        sel = (u_arr >= ua - 1e-6) & (u_arr <= ub + 1e-6)
        if int(sel.sum()) < 15:
            continue
        s_lo = float(np.percentile(s_arr[sel], 1.0))
        s_hi = float(np.percentile(s_arr[sel], 99.0))
        raw_bands.append([ua, ub, s_lo, s_hi])
    if len(raw_bands) < 1:
        return None

    def _levels(vals: List[float], eps_m: float, agg) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        cur = [order[0]]
        for i in order[1:]:
            if vals[i] - vals[cur[-1]] <= eps_m:
                cur.append(i)
            else:
                rep = agg(vals[j] for j in cur)
                for j in cur:
                    out[j] = rep
                cur = [i]
        rep = agg(vals[j] for j in cur)
        for j in cur:
            out[j] = rep
        return out

    lvl_lo = _levels([b[2] for b in raw_bands], 0.10, min)
    lvl_hi = _levels([b[3] for b in raw_bands], 0.10, max)

    def _snap_to_cross(val: float, tol_m: float = 0.12) -> float:
        best = val
        bd = tol_m
        for sp in cross_s_positions:
            if abs(sp - val) < bd:
                bd = abs(sp - val)
                best = sp
        return best

    lvl_lo = [_snap_to_cross(v) for v in lvl_lo]
    lvl_hi = [_snap_to_cross(v) for v in lvl_hi]

    band_boxes = []
    for (ua, ub, _s0, _s1), slo, shi in zip(raw_bands, lvl_lo, lvl_hi):
        if shi - slo < min_step_side_m:
            continue
        band_boxes.append(_box(ua, slo, ub, shi))
    if not band_boxes:
        return None
    hull = unary_union(band_boxes).buffer(0)
    if isinstance(hull, MultiPolygon):
        hull = max(hull.geoms, key=lambda g: g.area)
    if hull.is_empty or hull.area <= 0.0:
        return None

    # --- node on a 2 cm grid for a robust planar partition ---
    # The grid only makes ``unary_union`` robust against near-coincident
    # floats; it does NOT decide final edge positions — step 6 below
    # snaps every resulting edge back onto the EXACT detected ink line,
    # so the grid drift never reaches the output (Willi: every step edge
    # must sit exactly on a pixel line).
    snap = 0.02

    def _snap_pt(p: Tuple[float, float]) -> Tuple[float, float]:
        return (round(p[0] / snap) * snap, round(p[1] / snap) * snap)

    boundary = _ShPoly([_snap_pt(p) for p in hull.exterior.coords])
    region = boundary.buffer(0)

    # --- connectivity weld: every detected riser must terminate ON a
    # perpendicular feature so it closes a cell (Willi: each line is a
    # real boundary). A riser whose ink extent dangles a few cm short of
    # its L-corner leaves the winder wing as ONE merged face. Weld each
    # riser END onto the nearest perpendicular feature it should meet:
    #   * a main-riser END (along s) -> nearest CROSS-riser whose u-span
    #     covers this u, else the hull edge at this u;
    #   * a cross-riser END (along u) -> nearest MAIN-riser whose s-span
    #     covers this s, else the hull edge at this s.
    # Bounded by MAX_WELD so it never bridges across a whole tread.
    MAX_WELD = 0.16

    def _hull_span_along(fixed: float, vertical: bool) -> Tuple[float, float]:
        """(lo, hi) of the hull's extent on the free axis at ``fixed``."""
        if vertical:
            seg = LineString([(fixed, hb[1] - 0.5), (fixed, hb[3] + 0.5)])
        else:
            seg = LineString([(hb[0] - 0.5, fixed), (hb[2] + 0.5, fixed)])
        try:
            inter = boundary.intersection(seg)
        except Exception:
            return (hb[1], hb[3]) if vertical else (hb[0], hb[2])
        vals: List[float] = []
        for part in getattr(inter, "geoms", [inter]):
            if hasattr(part, "coords"):
                for x, y in part.coords:
                    vals.append(y if vertical else x)
        if not vals:
            return (hb[1], hb[3]) if vertical else (hb[0], hb[2])
        return (min(vals), max(vals))

    hb = hull.bounds

    def _weld_end(v: float, candidates: List[float]) -> float:
        if not candidates:
            return v
        best = min(candidates, key=lambda c: abs(c - v))
        return best if abs(best - v) <= MAX_WELD else v

    cross_s_all = [(s, u0, u1) for s, u0, u1 in cross_us]
    main_u_all = [(u, s0, s1) for u, s0, s1 in main_us]

    geoms = [boundary.exterior]
    for u_pos, s_lo, s_hi in main_us:
        cov = [s for s, u0, u1 in cross_s_all if u0 - 0.03 <= u_pos <= u1 + 0.03]
        h_lo, h_hi = _hull_span_along(u_pos, vertical=True)
        s_lo2 = _weld_end(s_lo, cov + [h_lo])
        s_hi2 = _weld_end(s_hi, cov + [h_hi])
        if abs(s_hi2 - s_lo2) < min_step_side_m:
            continue
        geoms.append(LineString([_snap_pt((u_pos, s_lo2)), _snap_pt((u_pos, s_hi2))]))
    for s_pos, u_lo, u_hi in cross_us:
        cov = [u for u, s0, s1 in main_u_all if s0 - 0.03 <= s_pos <= s1 + 0.03]
        h_lo, h_hi = _hull_span_along(s_pos, vertical=False)
        u_lo2 = _weld_end(u_lo, cov + [h_lo])
        u_hi2 = _weld_end(u_hi, cov + [h_hi])
        if abs(u_hi2 - u_lo2) < min_step_side_m:
            continue
        geoms.append(LineString([_snap_pt((u_lo2, s_pos)), _snap_pt((u_hi2, s_pos))]))
    try:
        merged = unary_union(geoms)
        faces = list(polygonize(merged))
    except Exception:
        return None
    if not faces:
        return None

    # Step 6 (Willi-Idee) — snap every face-edge coordinate back onto the
    # EXACT detected ink line it came from. Build the precise coordinate
    # lattice: real main-riser u-positions, real cross-riser s-positions,
    # plus the hull's own coords (pixel-terminated outer edges). Each face
    # vertex's u snaps to the nearest main-riser-u (or hull-x) and its s
    # to the nearest cross-riser-s (or hull-y) within ``ink_tol`` — just
    # enough to undo the 2 cm node-grid drift without re-assigning a vertex
    # to the wrong line. After this, every internal step boundary lies on
    # a real riser stroke and every outer edge on the pixel hull.
    ink_tol = 0.035
    hull_xs = sorted({float(x) for x, _y in hull.exterior.coords})
    hull_ys = sorted({float(y) for _x, y in hull.exterior.coords})
    u_lattice = sorted(set(float(u) for u, _, _ in main_us) | set(hull_xs))
    s_lattice = sorted(set(float(s) for s, _, _ in cross_us) | set(hull_ys))

    def _snap_axis(v: float, lattice: List[float]) -> float:
        if not lattice:
            return v
        best = min(lattice, key=lambda t: abs(t - v))
        return best if abs(best - v) <= ink_tol else v

    def _snap_to_ink(poly: _ShPoly) -> _ShPoly:
        ring = [
            (_snap_axis(float(x), u_lattice), _snap_axis(float(y), s_lattice))
            for x, y in poly.exterior.coords
        ]
        try:
            snapped = _ShPoly(ring).buffer(0)
        except Exception:
            return poly
        if snapped.is_empty:
            return poly
        if isinstance(snapped, MultiPolygon):
            snapped = max(snapped.geoms, key=lambda g: g.area)
        return snapped
    steps_us: List[_ShPoly] = []
    for f in faces:
        if f.area <= min_step_area_m2:
            continue
        if not region.contains(f.representative_point()):
            continue
        # min side via the minimum-rotated-rectangle short edge
        mrr = f.minimum_rotated_rectangle
        mc = list(mrr.exterior.coords)
        if len(mc) >= 3:
            e1 = math.hypot(mc[1][0] - mc[0][0], mc[1][1] - mc[0][1])
            e2 = math.hypot(mc[2][0] - mc[1][0], mc[2][1] - mc[1][1])
            if min(e1, e2) < min_step_side_m:
                continue
        steps_us.append(_snap_to_ink(f))
    if not steps_us:
        return None
    steps_us.sort(key=lambda f: f.centroid.x)

    # --- materialise step polygons in world XY + aggregates ---
    def _drop_collinear(
        ring: List[Tuple[float, float]], eps: float = 1e-4,
    ) -> List[Tuple[float, float]]:
        """Remove duplicate + collinear vertices from a closed ring.

        Endpoint snapping can leave a redundant point ON an edge (e.g.
        a riser top at 0.30 snapped to the hull edge at 0.28 leaves both
        on the same vertical boundary). Dropping them makes rectangles
        exactly 4 vertices and L-treads exactly 6 — the clean ring the
        winder decomposition promises.
        """
        pts = [p for p in ring[:-1]]  # open ring
        n = len(pts)
        if n < 4:
            return ring
        keep: List[Tuple[float, float]] = []
        for i in range(n):
            a = pts[(i - 1) % n]
            b = pts[i]
            c = pts[(i + 1) % n]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            d_ab = math.hypot(b[0] - a[0], b[1] - a[1])
            if d_ab < eps:
                continue  # duplicate of previous
            if abs(cross) < eps:
                continue  # collinear with neighbours
            keep.append(b)
        if len(keep) < 3:
            return ring
        return keep + [keep[0]]

    step_polygons: List[List[Tuple[float, float]]] = []
    widths: List[float] = []
    cx_acc = cy_acc = 0.0
    n_acc = 0
    for f in steps_us:
        ring = _drop_collinear(list(f.exterior.coords))
        world_ring = [_to_world(u, s) for (u, s) in ring]
        step_polygons.append(world_ring)
        ub = f.bounds  # (minu, mins, maxu, maxs)
        widths.append(float(ub[3] - ub[1]))
        for v in world_ring[:-1]:
            cx_acc += v[0]
            cy_acc += v[1]
            n_acc += 1
    if not step_polygons or n_acc == 0:
        return None

    widths_sorted = sorted(widths)
    width_median = widths_sorted[len(widths_sorted) // 2]
    width_spread = widths_sorted[-1] - widths_sorted[0]
    hb = hull.bounds
    length_m = float(hb[2] - hb[0])
    n_phys = len(step_polygons)
    going_m = length_m / n_phys if n_phys > 0 else None
    rise_m = total_height_m / n_phys if n_phys > 0 else None
    centroid = (cx_acc / n_acc, cy_acc / n_acc)

    centerlines = [[(float(ln[0][0]), float(ln[0][1])),
                    (float(ln[1][0]), float(ln[1][1]))]
                   for ln in main_riser_lines_m]

    return {
        "step_polygons_xy_m": step_polygons,
        "tread_centerlines_xy_m": centerlines,
        "n_steps": n_phys,
        "length_m": length_m,
        "width_m": float(width_median),
        "going_m": going_m,
        "rise_m": rise_m,
        "centroid_xy_m": centroid,
        "width_spread_m": float(width_spread),
        "n_cross_risers": len(cross_us),
    }


def _line_line_intersect(
    p1: Tuple[float, float], p2: Tuple[float, float],
    p3: Tuple[float, float], p4: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """Intersection of two infinite lines, each given by two endpoints.

    Returns ``(x, y)`` of the intersection point, or ``None`` when the
    lines are parallel. The lines are treated as *infinite* so the
    point can lie outside both input segments — this is what we want
    when extending a Hough tread line to meet a Hough stringer.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _make_skipped_record(
    *,
    label_id: str,
    label_class: str,
    status: StairStatus,
    note: str,
    subtype: str = "",
) -> StairRecord:
    """Build a placeholder record with a non-OK status for the bundle."""
    ifc_pre = resolve_ifc_predefined_type(label_class, subtype or None)
    return StairRecord(
        label_id=label_id,
        label_class=label_class,
        ifc_class="IfcStair",
        ifc_predefined_type=ifc_pre,
        subtype=subtype,
        status=status,
        notes=[note],
    )


def process_stairs(
    *,
    annotations: Sequence[Dict[str, Any]],
    scale_xy: Tuple[float, float],
    px_to_m: float,
    image_gray: Optional[np.ndarray],
    bim_walls_m: Sequence[Sequence[Tuple[float, float]]],
    config: Optional[Dict[str, Any]] = None,
) -> List[StairPipelineResult]:
    """Process every stair label in ``annotations`` end-to-end.

    Args:
        annotations: raw ``annotations.json`` annotation list. Only
            entries whose ``class`` is in :data:`STAIR_LABEL_CLASSES`
            are processed; the rest are ignored silently.
        scale_xy: ``(sx, sy)`` scale factor from annotation-image
            coordinates to render-image coordinates (mirrors the
            existing door/window pipelines).
        px_to_m: metre-per-pixel scale of the rendered plan image.
        image_gray: rendered grayscale image. ``None`` forces the
            polygon-PCA fallback for every stair.
        bim_walls_m: post-extrude wall polygons in plan metres
            (typically ``[list(wp.shell_xy) for wp in result.polygons]``).
            Used to build the wall-envelope filter (Tag 4).
        config: optional pipeline config. Recognised keys:
            * ``stair_floor_to_floor_m`` — per-job height override.
            * ``ceiling_buildup_m`` — default ceiling buildup,
              default 0.15 m.

    Returns:
        One ``StairPipelineResult`` per processed annotation, in the
        order they appeared in ``annotations``. Skipped stairs still
        emit a result so the bundle's semantics_v3.json reflects every
        label decision.
    """
    config = config or {}
    floor_to_floor_config = config.get("stair_floor_to_floor_m")
    ceiling_buildup = float(
        config.get("ceiling_buildup_m", DEFAULT_CEILING_BUILDUP_M)
    )
    wall_height_default = float(
        config.get("wall_height_m", DEFAULT_PIPELINE_FLOOR_TO_FLOOR_M - ceiling_buildup)
    )
    pipeline_default_m = wall_height_default + ceiling_buildup

    envelope = build_wall_envelope_from_polys(bim_walls_m)

    # Sprint v3 — pre-compute every annotation's polygon in render-px
    # so the confidence-field builder can let each stair label compete
    # for ownership of nearby pixels. WALL labels are deliberately
    # excluded from the competition: a stair-stringer (wangen) is
    # almost always drawn on top of the room's wall ink, and the wall
    # label "owns" that ink. If walls competed, the wangen pixels
    # would always belong to the wall and the stair would never see
    # its own structural envelope. Walls are structural background;
    # they can co-exist with stairs/doors/windows on the same pixels.
    EXCLUDED_NEIGHBOR_CLASSES = {
        "wall", "wall_interior", "wall_exterior",
    }
    all_annotation_polys_px: List[Tuple[str, List[Tuple[float, float]]]] = []
    for ann_all in annotations or []:
        ann_id = str(ann_all.get("id") or "")
        ann_cls = (ann_all.get("class") or "").strip()
        if ann_cls in EXCLUDED_NEIGHBOR_CLASSES:
            continue
        ann_poly = _poly_from_annotation(ann_all, scale_xy)
        if len(ann_poly) >= 3:
            all_annotation_polys_px.append((ann_id, ann_poly))

    results: List[StairPipelineResult] = []
    for idx, ann in enumerate(annotations or []):
        cls = (ann.get("class") or "").strip()
        if cls not in STAIR_LABEL_CLASSES:
            continue

        label_id = str(ann.get("id") or f"stair-{idx}")
        # Sprint Stairs Tag 10 — read optional user-selected subtype
        # from annotation.meta. Drives routing (V2-deferred subtypes
        # produce SKIPPED_UNSUPPORTED_CLASS_V1) and IFC PredefinedType.
        ann_meta = ann.get("meta") or {}
        subtype = str(ann_meta.get("subtype") or "").strip()
        algorithm = classify_stair(cls, subtype=subtype)
        ifc_predefined_type = resolve_ifc_predefined_type(cls, subtype or None)

        # Materialise the user polygon early so EVERY skip path can
        # still expose it to the diagnostic renderer (Bug #4 fix —
        # otherwise skipped stairs are invisible in the per-stair
        # zoom output).
        poly_px = _poly_from_annotation(ann, scale_xy)
        poly_m_early = (
            _poly_to_m(poly_px, px_to_m) if len(poly_px) >= 3 else []
        )

        if algorithm == STAIR_ALGORITHM_UNSUPPORTED:
            note = (
                f"{cls}/{subtype} routed to V2 sprint"
                if subtype
                else f"{cls} routed to V2 sprint"
            )
            results.append(StairPipelineResult(
                record=_make_skipped_record(
                    label_id=label_id,
                    label_class=cls,
                    status=StairStatus.SKIPPED_UNSUPPORTED_CLASS_V1,
                    note=note,
                    subtype=subtype,
                ),
                user_polygon_m=list(poly_m_early),
            ))
            continue

        if len(poly_px) < 3:
            results.append(StairPipelineResult(
                record=_make_skipped_record(
                    label_id=label_id,
                    label_class=cls,
                    status=StairStatus.SKIPPED_DEGENERATE_FOOTPRINT,
                    note="annotation has < 3 polygon points",
                    subtype=subtype,
                ),
                user_polygon_m=list(poly_m_early),
            ))
            continue

        # Tag 9 — compute the trimmed footprint polygon
        # (user_polygon - wall_union). Walls in bim_walls_m are the
        # source of truth: they carve notches and holes out of the
        # user-drawn bbox so non-rectangular stair forms come out of
        # the pipeline with the correct geometry.
        poly_m = poly_m_early  # already materialised above
        trimmed = compute_stair_polygon(poly_m, bim_walls_m)
        clipped_rings_px: List[List[Tuple[float, float]]] = []
        exterior_m = trimmed.exterior_xy()
        if exterior_m and px_to_m > 0.0:
            clipped_rings_px.append(
                [(x / px_to_m, y / px_to_m) for x, y in exterior_m]
            )
            for hole_m in trimmed.hole_rings_xy():
                clipped_rings_px.append(
                    [(x / px_to_m, y / px_to_m) for x, y in hole_m]
                )

        # Step 1 — pixel-derived geometry, fall back to polygon-PCA.
        pixel_details: Optional[StairPixelDetails] = None
        used_pixel_path = False
        try:
            footprint, pixel_details = extract_stair_pixels(
                image_gray,
                poly_px,
                px_to_m=px_to_m,
                clipped_polygon_px=clipped_rings_px or None,
            )
            used_pixel_path = True
        except PixelExtractionUnavailable as exc:
            try:
                # Prefer the trimmed polygon over the raw user polygon
                # for the fallback footprint as well — gives length /
                # width that already account for the wall subtraction.
                if trimmed.polygon.area > 0:
                    long_axis, _short, length, width, centroid = (
                        pca_axes_from_polygon(trimmed.polygon)
                    )
                    from .footprint import StairFootprint  # noqa: PLC0415
                    footprint = StairFootprint(
                        centroid_m=centroid,
                        long_axis=long_axis,
                        short_axis=(-long_axis[1], long_axis[0]),
                        length_m=length,
                        width_m=width,
                    )
                else:
                    footprint = extract_stair_footprint(poly_m)
            except ValueError as fpc:
                results.append(StairPipelineResult(
                    record=_make_skipped_record(
                        label_id=label_id,
                        label_class=cls,
                        status=StairStatus.SKIPPED_PCA_FAILED,
                        note=f"polygon-PCA failed ({fpc})",
                        subtype=subtype,
                    ),
                    notes=[f"pixel path unavailable: {exc}"],
                    user_polygon_m=list(poly_m),
                ))
                continue

        if footprint.length_m < MIN_FOOTPRINT_M or footprint.width_m < MIN_FOOTPRINT_M:
            results.append(StairPipelineResult(
                record=_make_skipped_record(
                    label_id=label_id,
                    label_class=cls,
                    status=StairStatus.SKIPPED_TOO_SMALL,
                    note=(
                        f"footprint too small: "
                        f"{footprint.length_m:.2f} x {footprint.width_m:.2f} m"
                    ),
                    subtype=subtype,
                ),
                pixel_details=pixel_details,
                user_polygon_m=list(poly_m),
            ))
            continue

        if footprint.length_m > MAX_FOOTPRINT_M:
            results.append(StairPipelineResult(
                record=_make_skipped_record(
                    label_id=label_id,
                    label_class=cls,
                    status=StairStatus.SKIPPED_TOO_LARGE,
                    note=f"footprint length {footprint.length_m:.2f} m > {MAX_FOOTPRINT_M} m",
                    subtype=subtype,
                ),
                pixel_details=pixel_details,
                user_polygon_m=list(poly_m),
            ))
            continue

        # Tag 11 (Bug #1 fix) — snap long-axis to user-bbox cardinal
        # whenever wall-subtraction produced a non-rectangular footprint.
        # PCA on an L-shaped polygon picks a diagonal even at small
        # deviations (~10-15°), which is enough to make the per-step
        # slicer cut diagonally across the stair AND across adjacent
        # walls. The user bbox encodes the stair's intended direction
        # (the label editor only draws axis-aligned bboxes), so it's
        # always the authoritative anchor when wall-subtraction was
        # active.
        axis_snap_note: Optional[str] = None
        if trimmed.used_wall_subtraction and len(poly_m) >= 3:
            from .stair_footprint_polygon import (  # noqa: PLC0415
                axis_anchor_from_user_polygon,
                reproject_polygon_onto_axis,
            )
            anchor = axis_anchor_from_user_polygon(poly_m)
            if anchor is not None:
                anchor_long, anchor_short, _anchor_len, _anchor_w, _anchor_c = anchor
                cos_deviation = abs(
                    footprint.long_axis[0] * anchor_long[0]
                    + footprint.long_axis[1] * anchor_long[1]
                )
                new_length, new_width, new_centroid = (
                    reproject_polygon_onto_axis(trimmed.polygon, anchor_long)
                )
                if new_length > 0.0 and new_width > 0.0:
                    from .footprint import StairFootprint  # noqa: PLC0415
                    footprint = StairFootprint(
                        centroid_m=new_centroid,
                        long_axis=anchor_long,
                        short_axis=anchor_short,
                        length_m=new_length,
                        width_m=new_width,
                    )
                    axis_snap_note = (
                        f"axis_snapped_to_user_bbox "
                        f"(pca_deviation={math.degrees(math.acos(min(1.0, cos_deviation))):.1f}deg, "
                        f"new L={new_length:.2f}m W={new_width:.2f}m)"
                    )

        # Sprint v2 — Hough-based tread / stringer / cut-line voter.
        # Runs in parallel to the density extractor. It gives us:
        #   * a tread-count voter anchored to actual ink linestrokes
        #     (more robust than the 1-D density profile on plans where
        #     the labelled bbox is sloppy or includes wall ink);
        #   * a measured stringer distance — overrides ``footprint.width_m``
        #     when both wangen were detected;
        #   * cut-line endpoints — used as direction-up signal #6 and as
        #     the authoritative high-end of the flight when present.
        from .stair_line_extractor import (  # noqa: PLC0415
            extract_stair_lines as _extract_stair_lines,
        )
        from .stair_confidence_field import (  # noqa: PLC0415
            build_confidence_field_sub as _build_confidence_field_sub,
        )
        # Sprint v3 — per-pixel confidence field. The user bbox is a
        # gravity well, not a hard ROI cap. Every OTHER annotation on
        # this plan (any class) competes for ownership of nearby pixels.
        # Sub-window form avoids a 380 MB H×W float32 allocation per
        # stair label (otherwise the pipeline spends most of its time
        # zeroing and pasting buffers).
        if image_gray is not None:
            neighbour_polys_px = [
                p for (aid, p) in all_annotation_polys_px
                if aid != label_id
            ]
            confidence_field = _build_confidence_field_sub(
                poly_px, neighbour_polys_px,
                image_gray.shape[:2],
                px_to_m=px_to_m,
            )
        else:
            confidence_field = None

        hough_details = _extract_stair_lines(
            image_gray, poly_px, px_to_m=px_to_m,
            confidence_field=confidence_field,
        )
        hough_centroids_m: List[Tuple[float, float]] = list(
            hough_details.tread_centroids_xy_m,
        )
        hough_n = len(hough_centroids_m)

        # Stringer-width override: replaces the PCA-derived width when
        # the Hough detector identified BOTH wangen with a plausible
        # gap (between 60 cm minimum width per DIN 18065 and the bbox
        # short extent + a 30 cm tolerance for sloppy labels).
        stringer_width_override: Optional[float] = None
        if (
            hough_details.stringer_distance_m is not None
            and hough_details.stringer_distance_m >= 0.60
            and hough_details.stringer_distance_m
            <= max(footprint.width_m, hough_details.bbox_short_extent_m) + 0.30
        ):
            stringer_width_override = float(
                hough_details.stringer_distance_m,
            )
            from .footprint import StairFootprint  # noqa: PLC0415
            footprint = StairFootprint(
                centroid_m=footprint.centroid_m,
                long_axis=footprint.long_axis,
                short_axis=footprint.short_axis,
                length_m=footprint.length_m,
                width_m=stringer_width_override,
            )

        # Sprint v3 — Bayesian long-axis resolver. Pixel evidence (the
        # stringer pair) overrides the bbox-cardinal prior when its
        # authority weight exceeds STRINGER_AXIS_AUTHORITY_FLOOR. The
        # weight combines stringer coverage (how much of the run the
        # wangen actually span) with the mean per-line confidence
        # (how deep inside the bbox the wangen ink lies). Note: the
        # axis can be ANY angle — we never snap back to cardinal when
        # the pixel evidence is strong. A slanted bbox can be human
        # error OR genuine plan geometry; the algorithm has no business
        # deciding which.
        STRINGER_AXIS_AUTHORITY_FLOOR = 0.45
        long_axis_from_stringer_note: Optional[str] = None
        if (
            hough_details.stringer_axis_authority
            >= STRINGER_AXIS_AUTHORITY_FLOOR
            and hough_details.stringer_long_axis != (0.0, 0.0)
        ):
            new_long = hough_details.stringer_long_axis
            new_short = (-new_long[1], new_long[0])
            from .footprint import StairFootprint  # noqa: PLC0415
            footprint = StairFootprint(
                centroid_m=footprint.centroid_m,
                long_axis=(float(new_long[0]), float(new_long[1])),
                short_axis=(float(new_short[0]), float(new_short[1])),
                length_m=footprint.length_m,
                width_m=footprint.width_m,
            )
            # Cardinal prior for diagnostics.
            cardinal = hough_details.bbox_long_unit
            cos_diff = abs(
                cardinal[0] * new_long[0] + cardinal[1] * new_long[1]
            )
            cos_diff = max(-1.0, min(1.0, cos_diff))
            diff_deg = math.degrees(math.acos(cos_diff))
            long_axis_from_stringer_note = (
                f"long_axis_from_stringer_pixels "
                f"(authority={hough_details.stringer_axis_authority:.2f}, "
                f"cardinal_diff={diff_deg:.1f}deg, "
                f"override_floor={STRINGER_AXIS_AUTHORITY_FLOOR:.2f})"
            )

        # Step 2 — direction-up.
        # 2a. Try the explicit arrow detector first (Tag 8/11). When the
        #     user drew an up-arrow inside the stair we use it directly;
        #     the density heuristic is a fallback for arrow-less plans.
        #     Tag 11 (Bug #2 fix): 3-pass detection.
        #       Pass A — strict dark threshold, no ROI expansion.
        #                Catches obvious black arrows fully inside the
        #                user bbox.
        #       Pass B — narrow GREY band [primary+10, 220] with a 30 cm
        #                ROI dilation. The narrow band excludes the
        #                black tread lines (which would otherwise be
        #                the dominant elongated components) and the
        #                white page background. The dilation lets the
        #                detector see arrows that poke just past the
        #                user bbox into the adjacent flight area.
        #       Pass C — permissive dark threshold 200 with the same
        #                ROI dilation. Last-resort catch for arrows
        #                drawn in medium-grey ink that overlap the
        #                tread band; less reliable because tread lines
        #                also pass the threshold, so a real arrow has
        #                to outrun every tread line on length.
        arrow_direction_xy = None
        arrow_confidence = 0.0
        if (
            pixel_details is not None
            and pixel_details.length_px > 0.0
            and image_gray is not None
        ):
            primary_threshold = int(getattr(
                pixel_details, "dark_threshold_used", 140,
            ) or 140)
            # 60 cm dilation around the user polygon, in pixels.
            # Direction arrows in floor plans often extend past the
            # labelled bbox by a tread-or-two; 60 cm is enough to
            # encompass the full arrow shaft+head on production scans
            # without bringing adjacent stairs / walls into the
            # connected-component graph.
            roi_expand_px = max(0.0, 0.60 / max(px_to_m, 1e-9))

            # Pass A — strict ink threshold, no expansion.
            arrow_detection = detect_arrow_direction(
                image_gray=image_gray,
                user_polygon_px=poly_px,
                long_axis_xy=pixel_details.long_axis_px,
                stair_length_px=pixel_details.length_px,
                dark_threshold=primary_threshold,
            )

            # Pass B — narrow grey band + ROI dilation.
            if arrow_detection.direction_xy is None:
                grey_lo = max(0, primary_threshold + 10)
                grey_hi = 220
                if grey_hi > grey_lo:
                    arrow_detection = detect_arrow_direction(
                        image_gray=image_gray,
                        user_polygon_px=poly_px,
                        long_axis_xy=pixel_details.long_axis_px,
                        stair_length_px=pixel_details.length_px,
                        dark_band=(grey_lo, grey_hi),
                        expand_px=roi_expand_px,
                    )

            # Pass C — permissive ink threshold + ROI dilation.
            if arrow_detection.direction_xy is None:
                arrow_detection = detect_arrow_direction(
                    image_gray=image_gray,
                    user_polygon_px=poly_px,
                    long_axis_xy=pixel_details.long_axis_px,
                    stair_length_px=pixel_details.length_px,
                    dark_threshold=200,
                    expand_px=roi_expand_px,
                )
            if arrow_detection.direction_xy is not None:
                # Arrow is detected in pixel-frame (PDF Y points down).
                # Footprint long_axis is also in pixel-frame, so the
                # conversion to plan-metre frame is identity: the
                # downstream consumers treat plan-XY as identical to
                # the rendered pixel XY (px_to_m is a pure scalar
                # scale, no axis flip).
                arrow_direction_xy = arrow_detection.direction_xy
                arrow_confidence = arrow_detection.confidence

        if pixel_details is not None and pixel_details.density_profile is not None:
            direction_result = detect_direction_up(
                pixel_details.density_profile,
                long_axis=footprint.long_axis,
                arrow_direction_xy=arrow_direction_xy,
                arrow_confidence=arrow_confidence,
            )
        else:
            direction_result = detect_direction_up(
                np.zeros(0, dtype=np.float64),  # forces heuristic fallback
                long_axis=footprint.long_axis,
                arrow_direction_xy=arrow_direction_xy,
                arrow_confidence=arrow_confidence,
            )

        # Sprint v2 — Cut-line direction-up override (signal #6). The
        # Höhensprung is drawn at the visible end of a flight, which is
        # by DIN convention the UPPER end. If we have exactly one cut
        # line and no high-confidence arrow signal, point direction-up
        # AWAY from the cut line.
        cut_line_override_note: Optional[str] = None
        if (
            len(hough_details.cutlines_xy_m) == 1
            and arrow_confidence < 0.4
        ):
            cl = hough_details.cutlines_xy_m[0]
            if len(cl) == 2:
                cl_mid = (
                    0.5 * (cl[0][0] + cl[1][0]),
                    0.5 * (cl[0][1] + cl[1][1]),
                )
                # Vector from cut-line midpoint to footprint centroid.
                away_dx = footprint.centroid_m[0] - cl_mid[0]
                away_dy = footprint.centroid_m[1] - cl_mid[1]
                away_norm = math.hypot(away_dx, away_dy)
                if away_norm > 1e-6:
                    cl_dir = (away_dx / away_norm, away_dy / away_norm)
                    # Project onto the (snapped) footprint long axis
                    # and pick the sign.
                    proj = (
                        cl_dir[0] * footprint.long_axis[0]
                        + cl_dir[1] * footprint.long_axis[1]
                    )
                    if abs(proj) > 0.3:
                        sign = 1.0 if proj > 0 else -1.0
                        new_dir = (
                            sign * footprint.long_axis[0],
                            sign * footprint.long_axis[1],
                        )
                        # Override when the existing direction was a
                        # default fallback OR disagrees with the cut
                        # line by > 90°.
                        old_dir = direction_result.direction_up_xy
                        dot = (
                            old_dir[0] * new_dir[0]
                            + old_dir[1] * new_dir[1]
                        )
                        if direction_result.used_default or dot < 0.0:
                            from .stair_direction import DirectionVotingResult  # noqa: PLC0415
                            direction_result = DirectionVotingResult(
                                direction_up_xy=new_dir,
                                confidence=0.7,
                                used_default=False,
                                source="cut_line",
                                arrow_confidence=0.0,
                            )
                            cut_line_override_note = (
                                "direction_up_overridden_by_cut_line "
                                "(arrow_low_confidence, cut_line points to LOW end)"
                            )

        # Step 3 — wall-envelope filter.
        if envelope is not None and not is_centroid_inside(
            footprint.centroid_m, envelope,
        ):
            results.append(StairPipelineResult(
                record=_make_skipped_record(
                    label_id=label_id,
                    label_class=cls,
                    status=StairStatus.SKIPPED_OUTSIDE_WALLS,
                    note="centroid outside building convex hull",
                    subtype=subtype,
                ),
                pixel_details=pixel_details,
                user_polygon_m=list(poly_m),
            ))
            continue

        # Step 4 — height resolution.
        height_res = resolve_total_height_m(
            annotation_meta=ann.get("meta"),
            config_value_m=floor_to_floor_config,
            pipeline_default_m=pipeline_default_m,
        )
        total_height_m = height_res["height_m"]
        height_source = height_res["height_source"]

        # Step 5 — step layout. Multi-voter aggregator. Pixel ink is
        # IST-Zustand and remains authoritative; the algorithm trusts
        # what it sees in the plan, NOT a DIN-derived expectation.
        # Sprint v2: a third voter (Hough-line clusters) joins the
        # density-peak voter; the DIN-target voter stays only as a
        # sanity check. Aggregation rule (analog to door hinge
        # ``vote_hinge_long_sign`` — confidence = ratio × min(1, n/3)):
        #   * Voter-H (Hough cluster count)        — primary
        #   * Voter-D (density-peak count)          — equal weight
        #   * Voter-N (DIN target = L / 0.29)       — sanity
        # If H and D agree within ±1, that count wins with high
        # confidence; if they disagree, the one that aligns better
        # with Voter-N wins.
        voter_hough = hough_n if hough_n >= 2 else None
        voter_density = (
            int(pixel_details.n_steps)
            if pixel_details is not None and pixel_details.n_steps is not None
            else None
        )
        voter_din = (
            max(1, int(round(footprint.length_m / 0.29)))
            if footprint.length_m > 0.0 else None
        )

        n_steps_source = "none"
        n_steps_confidence = 0.0
        if voter_hough is not None and voter_density is not None:
            if abs(voter_hough - voter_density) <= 1:
                # Both pixel voters agree — strong consensus.
                n_steps = int(round(0.5 * (voter_hough + voter_density)))
                n_steps_source = "hough+density"
                n_steps_confidence = min(1.0, 2 / 3.0) * 1.0
            else:
                # Disagreement. The one closer to DIN target wins.
                if voter_din is None:
                    n_steps = voter_hough
                    n_steps_source = "hough"
                else:
                    if abs(voter_hough - voter_din) <= abs(voter_density - voter_din):
                        n_steps = voter_hough
                        n_steps_source = "hough_vs_density_tiebreak_din"
                    else:
                        n_steps = voter_density
                        n_steps_source = "density_vs_hough_tiebreak_din"
                n_steps_confidence = min(1.0, 2 / 3.0) * 0.5
        elif voter_hough is not None:
            n_steps = voter_hough
            n_steps_source = "hough"
            n_steps_confidence = min(1.0, 1 / 3.0) * 1.0
        elif voter_density is not None:
            n_steps = voter_density
            n_steps_source = "density"
            n_steps_confidence = min(1.0, 1 / 3.0) * 1.0
        else:
            # All pixel voters failed → geometry layout.
            layout = derive_step_layout(
                length_m=footprint.length_m, total_height_m=total_height_m,
            )
            n_steps = layout["n_steps"]
            n_steps_source = "din_geometry"
            n_steps_confidence = 0.2

        # Going & rise. Peak-extent-based going (NOT length / n) so
        # L-shaped flights where the labelled bbox extends across the
        # landing produce a DIN-conformant going. Fall back to length/n
        # when we lack peak positions.
        peak_extent_m: Optional[float] = None
        if hough_n >= 2:
            # Project the hough centroids onto the footprint long axis
            # and take min-to-max.
            ts_h = [
                (
                    p[0] * footprint.long_axis[0]
                    + p[1] * footprint.long_axis[1]
                )
                for p in hough_centroids_m
            ]
            peak_extent_m = float(max(ts_h) - min(ts_h))
        elif (
            pixel_details is not None
            and pixel_details.tread_positions_xy_m is not None
            and len(pixel_details.tread_positions_xy_m) >= 2
        ):
            ts_d = [
                (
                    p[0] * footprint.long_axis[0]
                    + p[1] * footprint.long_axis[1]
                )
                for p in pixel_details.tread_positions_xy_m
            ]
            peak_extent_m = float(max(ts_d) - min(ts_d))

        if (
            peak_extent_m is not None
            and n_steps is not None
            and n_steps > 1
            and peak_extent_m > 0.0
        ):
            going_m = peak_extent_m / (n_steps - 1)
        elif n_steps is not None and n_steps > 0 and footprint.length_m > 0.0:
            going_m = footprint.length_m / n_steps
        else:
            going_m = None

        if n_steps is not None and n_steps > 0:
            rise_m = total_height_m / max(n_steps, 1)
        else:
            rise_m = None

        n_steps_voter_log = {
            "hough": voter_hough,
            "density": voter_density,
            "din": voter_din,
        }

        # Decide which set of peak positions feeds the per-step slicer.
        # Hough centroids are the IST-Zustand authority when available;
        # density peaks are the fallback. ``peak_source`` is recorded
        # in the notes so we can audit which path produced each stair.
        if hough_n >= 2:
            peak_source = "hough"
        elif (
            pixel_details is not None
            and pixel_details.tread_positions_xy_m is not None
            and len(pixel_details.tread_positions_xy_m) > 0
        ):
            peak_source = "density"
        else:
            peak_source = "none"

        # Tag 11 (Bug #3 v2) — half-flight heuristic for total_height.
        # When the pixel-derived n_steps yields rise > 30 cm (no human
        # stair has 30 cm steps), the issue is NOT the pixel count —
        # it's that the labelled flight only spans HALF a floor (typical
        # for U-shaped or split stairs where two flights connect at a
        # landing) but ``total_height`` defaulted to floor-to-floor.
        #
        # Without an explicit annotation override we halve total_height
        # and re-derive rise. n_steps and going stay at the pixel-truth
        # values. The note tells the user where the assumption came
        # from so they can correct it via annotation_meta.total_height_m
        # if needed.
        rise_sanity_note: Optional[str] = None
        if (
            rise_m is not None
            and rise_m > 0.30
            and total_height_m > 0.30
            and height_source == "pipeline_default"
        ):
            halved = total_height_m * 0.5
            new_rise = halved / max(n_steps, 1)
            rise_sanity_note = (
                f"half_flight_heuristic: rise={rise_m * 100:.0f}cm "
                f"with total_height={total_height_m:.2f}m exceeds 30cm physical max — "
                f"assuming half-flight, total_height halved to {halved:.2f}m, "
                f"new rise={new_rise * 100:.0f}cm (n_steps={n_steps} unchanged)"
            )
            total_height_m = halved
            rise_m = new_rise

        walking_line = build_walking_line_xy(
            centroid_xy_m=footprint.centroid_m,
            direction_up_xy=direction_result.direction_up_xy,
            length_m=footprint.length_m,
        )

        record = StairRecord(
            label_id=label_id,
            label_class=cls,
            ifc_class="IfcStair",
            ifc_predefined_type=ifc_predefined_type,
            subtype=subtype,
            centroid_xy_m=footprint.centroid_m,
            long_axis_deg=footprint.long_axis_deg,
            length_m=footprint.length_m,
            width_m=footprint.width_m,
            n_steps=n_steps,
            rise_m=rise_m,
            going_m=going_m,
            total_height_m=total_height_m,
            direction_up_xy=direction_result.direction_up_xy,
            walking_line_xy=walking_line,
            height_source=height_source,
            status=StairStatus.OK,
        )

        # Provenance notes.
        record.notes.append(
            f"footprint via {'pixel' if used_pixel_path else 'polygon'}-pca"
        )
        if pixel_details is not None and pixel_details.used_component_filter:
            record.notes.append(
                f"component_filter: {pixel_details.n_stripe_components}"
                f" stripe / {pixel_details.n_block_components} block dropped"
            )
        if pixel_details is not None and pixel_details.low_confidence_tread_count:
            record.notes.append("LOW_CONFIDENCE_TREAD_COUNT")
        record.notes.append(f"direction_up_source={direction_result.source}")
        if direction_result.source == "arrow":
            record.notes.append(
                f"arrow_confidence={direction_result.arrow_confidence:.2f}"
            )
        if direction_result.used_default:
            record.notes.append("direction_up_default_used")
        if trimmed.used_wall_subtraction:
            record.notes.append(
                f"wall_subtraction: {trimmed.n_holes} hole(s), "
                f"{trimmed.n_dropped_pieces} dropped piece(s)"
            )
        if axis_snap_note is not None:
            record.notes.append(axis_snap_note)
        if long_axis_from_stringer_note is not None:
            record.notes.append(long_axis_from_stringer_note)
        if rise_sanity_note is not None:
            record.notes.append(rise_sanity_note)
        if cut_line_override_note is not None:
            record.notes.append(cut_line_override_note)
        if stringer_width_override is not None:
            record.notes.append(
                f"width_override_from_stringer={stringer_width_override:.2f}m "
                f"(was {footprint.width_m:.2f}m)"
            )
        record.notes.append(
            f"n_steps_source={n_steps_source} conf={n_steps_confidence:.2f} "
            f"voters={n_steps_voter_log}"
        )
        record.notes.append(f"peak_source={peak_source}")
        if hough_details.notes:
            record.notes.append(
                f"hough: {hough_details.status} "
                f"(lines={hough_details.n_hough_lines}, "
                f"tread_cand={hough_details.n_tread_candidates}, "
                f"stringer_cand={hough_details.n_stringer_candidates}, "
                f"clusters={hough_n}, "
                f"cutlines={len(hough_details.cutlines_xy_m)}, "
                f"stringers={'pair' if hough_details.stringer_lines_xy_m else 'none'}, "
                f"stringer_conf={hough_details.mean_stringer_line_confidence:.2f}, "
                f"stringer_auth={hough_details.stringer_axis_authority:.2f}, "
                f"tread_conf={hough_details.mean_tread_line_confidence:.2f})"
            )

        # Tag 11 (IST-Zustand) — IST-Zustand alignment of BIM strips to
        # the plan ink. We use the ABSOLUTE plan-XY peak positions the
        # pixel extractor produced. These are anchored to the actual
        # ink, not derived from the (possibly axis-snapped) footprint
        # frame — so even on L-shaped trimmed polygons the peaks stay
        # locked to the visible treads.
        tread_centerlines_xy_m: List[List[Tuple[float, float]]] = []
        step_polygons_xy_m: List[List[Tuple[float, float]]] = []
        # ``peak_source`` is already set above (next to the n_steps
        # voter). ``peak_positions_xy`` is consumed below for slicing.
        peak_positions_xy: List[Tuple[float, float]] = []
        if peak_source == "hough":
            peak_positions_xy = list(hough_centroids_m)
        elif peak_source == "density" and pixel_details is not None:
            peak_positions_xy = list(pixel_details.tread_positions_xy_m or [])

        if peak_positions_xy:
            # Project each peak onto the (snapped) long axis to get its
            # parametric position along the stair. The long_axis used
            # for slicing must match the one used for direction-up,
            # because the per-step strips run PERPENDICULAR to it.
            ux, uy = direction_result.direction_up_xy
            norm = math.hypot(ux, uy)
            if norm > 1e-9:
                ux, uy = ux / norm, uy / norm
                sx_axis, sy_axis = -uy, ux
                half_w = footprint.width_m * 0.5
                cx_centre = footprint.centroid_m[0]
                cy_centre = footprint.centroid_m[1]
                low_x = cx_centre - 0.5 * footprint.length_m * ux
                low_y = cy_centre - 0.5 * footprint.length_m * uy

                # Build the trimmed polygon once for chord-clipping +
                # per-step slicing below.
                from shapely.geometry import (  # noqa: PLC0415
                    LineString,
                    MultiLineString,
                    MultiPolygon,
                    Polygon as _ShPoly,
                )
                stair_poly_unflipped = _ShPoly(
                    list(trimmed.exterior_xy()),
                    holes=[list(h) for h in trimmed.hole_rings_xy()],
                ).buffer(0)
                if isinstance(stair_poly_unflipped, MultiPolygon):
                    stair_poly_unflipped = max(
                        stair_poly_unflipped.geoms, key=lambda g: g.area,
                    )

                # Centrelines: a chord across the footprint at each
                # peak's parametric position, clipped to the trimmed
                # polygon so they don't run outside the BIM footprint
                # (relevant on L-shapes where the chord would otherwise
                # cross the wall-subtracted notch).
                projected_t: List[float] = []
                for pxm, pym in peak_positions_xy:
                    t = (pxm - low_x) * ux + (pym - low_y) * uy
                    projected_t.append(float(t))
                    cx_t = low_x + t * ux
                    cy_t = low_y + t * uy
                    a = (cx_t + sx_axis * half_w, cy_t + sy_axis * half_w)
                    b = (cx_t - sx_axis * half_w, cy_t - sy_axis * half_w)
                    if not stair_poly_unflipped.is_empty:
                        try:
                            clipped = stair_poly_unflipped.intersection(
                                LineString([a, b]),
                            )
                        except Exception:
                            clipped = None
                        if isinstance(clipped, LineString) and not clipped.is_empty:
                            coords = list(clipped.coords)
                            if len(coords) >= 2:
                                tread_centerlines_xy_m.append([
                                    (float(coords[0][0]), float(coords[0][1])),
                                    (float(coords[-1][0]), float(coords[-1][1])),
                                ])
                                continue
                        elif isinstance(clipped, MultiLineString) and not clipped.is_empty:
                            longest = max(clipped.geoms, key=lambda g: g.length)
                            coords = list(longest.coords)
                            if len(coords) >= 2:
                                tread_centerlines_xy_m.append([
                                    (float(coords[0][0]), float(coords[0][1])),
                                    (float(coords[-1][0]), float(coords[-1][1])),
                                ])
                                continue
                    # Fallback: full chord, unclipped.
                    tread_centerlines_xy_m.append([a, b])

                # Sprint v6 — Pixel-first per-step polygons from ink
                # bands. The hard requirement is: every step boundary
                # sits on actual plan ink and nothing is synthesised.
                # Risers = the detected tread Hough lines (validated to
                # lie >= 95 % on dark ink); each step's short-axis height
                # = the [1st, 99th] percentile of the ACTUAL ink extent
                # inside that tread band. Axis-aligned by construction,
                # so the v5 outline-snap artefacts (synthetic diagonal
                # from infinite-line intersection, skewed parallelograms,
                # 1 cm sliver, resolution instability) cannot occur, and
                # the L-wing is captured because each band measures its
                # own ink extent. See _build_pixel_step_polygons_from_ink.
                pixel_pure_slicing_used = False
                tread_lines_xy = list(hough_details.tread_lines_xy_m)
                stringer_lines = list(hough_details.stringer_lines_xy_m)

                if len(stringer_lines) == 2 and len(tread_lines_xy) >= 2:
                    # Sort tread lines low->high along the long axis.
                    def _long_proj_mid(ln):
                        mx = 0.5 * (ln[0][0] + ln[1][0])
                        my = 0.5 * (ln[0][1] + ln[1][1])
                        return (mx - low_x) * ux + (my - low_y) * uy

                    tread_sorted_2d = sorted(tread_lines_xy, key=_long_proj_mid)

                    # Hard pixel-criterion: keep only tread lines whose
                    # Hough segment lies >= MIN_LINE_PIXEL_RATIO on real
                    # dark ink (drops Hough ghosts before they become
                    # step risers).
                    pixel_threshold = int(hough_details.dark_threshold_used or 140)
                    MIN_LINE_PIXEL_RATIO = 0.95
                    tread_validated = []
                    n_dropped_treads = 0
                    for T in tread_sorted_2d:
                        ratio = _line_pixel_ratio_m(
                            (T[0], T[1]), image_gray, pixel_threshold, px_to_m,
                        )
                        if ratio >= MIN_LINE_PIXEL_RATIO:
                            tread_validated.append(T)
                        else:
                            n_dropped_treads += 1
                    if n_dropped_treads > 0:
                        record.notes.append(
                            f"dropped_{n_dropped_treads}_tread_lines_below_"
                            f"pixel_ratio_{MIN_LINE_PIXEL_RATIO}"
                        )

                    # Sprint Winder — planar-partition step decomposition.
                    # ONE generic algorithm: the main risers + the
                    # winder cross-risers are noded against a pixel-
                    # terminated outer hull and ``polygonize``d. A
                    # straight run (no cross-risers) falls out as pure
                    # rectangles identical to the v6 bands; a winder wing
                    # is carved into nested L-treads — no special-casing.
                    # Cross-risers are pixel-validated in the extractor.
                    cross_riser_lines = [
                        ln for ln in hough_details.cross_riser_lines_xy_m
                        if len(ln) == 2
                    ]
                    step_builder = "none"
                    pixel_steps = None
                    if len(tread_validated) >= 2:
                        pixel_steps = _polygonize_steps_from_risers(
                            image_gray=image_gray,
                            px_to_m=px_to_m,
                            dark_threshold=pixel_threshold,
                            user_polygon_px=poly_px,
                            confidence_field=confidence_field,
                            main_riser_lines_m=tread_validated,
                            cross_riser_lines_m=cross_riser_lines,
                            origin_xy_m=(low_x, low_y),
                            long_unit=(ux, uy),
                            short_unit=(sx_axis, sy_axis),
                            total_height_m=total_height_m,
                        )
                        if pixel_steps is not None:
                            step_builder = "polygonize"

                    # Fallback — v6 ink-band rectangles when the planar
                    # partition was degenerate (too sparse / broke).
                    if pixel_steps is None and len(tread_validated) >= 2:
                        pixel_steps = _build_pixel_step_polygons_from_ink(
                            image_gray=image_gray,
                            px_to_m=px_to_m,
                            dark_threshold=pixel_threshold,
                            user_polygon_px=poly_px,
                            confidence_field=confidence_field,
                            tread_lines_m=tread_validated,
                            origin_xy_m=(low_x, low_y),
                            long_unit=(ux, uy),
                            short_unit=(sx_axis, sy_axis),
                            total_height_m=total_height_m,
                        )
                        if pixel_steps is not None:
                            step_builder = "v6_bands"

                    if pixel_steps is not None:
                        step_polygons_xy_m = pixel_steps["step_polygons_xy_m"]
                        tread_centerlines_xy_m = pixel_steps[
                            "tread_centerlines_xy_m"
                        ]
                        physical_n_steps = pixel_steps["n_steps"]
                        new_length = pixel_steps["length_m"]
                        new_width = pixel_steps["width_m"]
                        if (
                            physical_n_steps != record.n_steps
                            or abs(new_length - record.length_m) > 0.01
                        ):
                            record.notes.append(
                                "step_polygons_from_ink_bands: "
                                f"n_steps={record.n_steps}->{physical_n_steps}, "
                                f"L={record.length_m:.2f}->{new_length:.2f}m, "
                                f"W={record.width_m:.2f}->{new_width:.2f}m"
                            )
                        record.n_steps = physical_n_steps
                        record.length_m = float(new_length)
                        record.width_m = float(new_width)
                        if pixel_steps["going_m"] is not None:
                            record.going_m = float(pixel_steps["going_m"])
                        if pixel_steps["rise_m"] is not None:
                            record.rise_m = float(pixel_steps["rise_m"])
                        record.centroid_xy_m = (
                            float(pixel_steps["centroid_xy_m"][0]),
                            float(pixel_steps["centroid_xy_m"][1]),
                        )
                        record.walking_line_xy = build_walking_line_xy(
                            centroid_xy_m=record.centroid_xy_m,
                            direction_up_xy=record.direction_up_xy,
                            length_m=record.length_m,
                        )
                        if pixel_steps["width_spread_m"] > 0.20:
                            record.notes.append(
                                "VARYING_WIDTH_L (per-step width spread="
                                f"{pixel_steps['width_spread_m']:.2f}m -> "
                                "non-rectangular L footprint)"
                            )
                        n_cross = pixel_steps.get("n_cross_risers", 0)
                        record.notes.append(
                            f"step_builder={step_builder}: {physical_n_steps} "
                            f"steps on {len(tread_validated)} main risers + "
                            f"{n_cross} cross-risers"
                        )
                        pixel_pure_slicing_used = True

                if not pixel_pure_slicing_used:
                    # Sprint v2.1 fallback — projection-based slicing
                    # when either the stringer pair or the tread lines
                    # were missing. Step polygons here are synthetic
                    # rectangles in the long/short frame, so vertices
                    # do NOT necessarily sit on plan ink.
                    projected_t.sort()

                    stair_start_t: float = 0.0
                    stair_end_t: float = float(footprint.length_m)
                    if (
                        hough_details.stringer_lines_xy_m
                        and len(hough_details.stringer_lines_xy_m) == 2
                    ):
                        stringer_ts: List[float] = []
                        for sl in hough_details.stringer_lines_xy_m:
                            for spt in sl:
                                t_proj = (
                                    (spt[0] - low_x) * ux
                                    + (spt[1] - low_y) * uy
                                )
                                stringer_ts.append(float(t_proj))
                        stair_start_t = max(0.0, min(stringer_ts))
                        stair_end_t = min(
                            float(footprint.length_m), max(stringer_ts),
                        )
                    elif len(projected_t) >= 2:
                        going_est = (
                            (projected_t[-1] - projected_t[0])
                            / max(len(projected_t) - 1, 1)
                        )
                        stair_start_t = max(0.0, projected_t[0] - 0.5 * going_est)
                        stair_end_t = min(
                            float(footprint.length_m),
                            projected_t[-1] + 0.5 * going_est,
                        )

                    short_lo: float = -half_w
                    short_hi: float = +half_w
                    if (
                        hough_details.stringer_lines_xy_m
                        and len(hough_details.stringer_lines_xy_m) == 2
                    ):
                        s_short_offsets: List[float] = []
                        for sl in hough_details.stringer_lines_xy_m:
                            mid_x = 0.5 * (sl[0][0] + sl[1][0])
                            mid_y = 0.5 * (sl[0][1] + sl[1][1])
                            s_short = (
                                (mid_x - low_x) * sx_axis
                                + (mid_y - low_y) * sy_axis
                            )
                            s_short_offsets.append(float(s_short))
                        short_lo = min(s_short_offsets)
                        short_hi = max(s_short_offsets)

                    inner_ts = [
                        t for t in projected_t
                        if stair_start_t < t < stair_end_t
                    ]
                    bounds = [stair_start_t] + inner_ts + [stair_end_t]

                    physical_n_steps = max(1, len(bounds) - 1)
                    stair_span_long_m = max(0.0, stair_end_t - stair_start_t)
                    stair_span_short_m = max(0.0, short_hi - short_lo)
                    if physical_n_steps != record.n_steps:
                        record.notes.append(
                            "step_count_updated_to_between_treads: "
                            f"{record.n_steps} -> {physical_n_steps} "
                            f"(tread_edges={len(inner_ts)} + start_step + end_step)"
                        )
                    record.n_steps = physical_n_steps
                    if stair_span_long_m > 0.0:
                        record.going_m = stair_span_long_m / physical_n_steps
                        record.length_m = stair_span_long_m
                    if stair_span_short_m > 0.0:
                        record.width_m = stair_span_short_m
                    record.rise_m = total_height_m / physical_n_steps

                    outline_cx = low_x + 0.5 * (stair_start_t + stair_end_t) * ux \
                        + 0.5 * (short_lo + short_hi) * sx_axis
                    outline_cy = low_y + 0.5 * (stair_start_t + stair_end_t) * uy \
                        + 0.5 * (short_lo + short_hi) * sy_axis
                    record.centroid_xy_m = (
                        float(outline_cx), float(outline_cy),
                    )
                    record.walking_line_xy = build_walking_line_xy(
                        centroid_xy_m=record.centroid_xy_m,
                        direction_up_xy=record.direction_up_xy,
                        length_m=record.length_m,
                    )

                    if not stair_poly_unflipped.is_empty:
                        for i in range(len(bounds) - 1):
                            t_lo = bounds[i]
                            t_hi = bounds[i + 1]
                            if t_hi <= t_lo:
                                step_polygons_xy_m.append([])
                                continue
                            strip = _ShPoly([
                                (low_x + t_lo * ux + short_lo * sx_axis,
                                 low_y + t_lo * uy + short_lo * sy_axis),
                                (low_x + t_hi * ux + short_lo * sx_axis,
                                 low_y + t_hi * uy + short_lo * sy_axis),
                                (low_x + t_hi * ux + short_hi * sx_axis,
                                 low_y + t_hi * uy + short_hi * sy_axis),
                                (low_x + t_lo * ux + short_hi * sx_axis,
                                 low_y + t_lo * uy + short_hi * sy_axis),
                            ]).buffer(0)
                            try:
                                inter = stair_poly_unflipped.intersection(strip)
                            except Exception:
                                step_polygons_xy_m.append([])
                                continue
                            if inter.is_empty:
                                step_polygons_xy_m.append([])
                                continue
                            if isinstance(inter, MultiPolygon):
                                inter = max(inter.geoms, key=lambda g: g.area)
                            try:
                                step_polygons_xy_m.append(list(inter.exterior.coords))
                            except AttributeError:
                                step_polygons_xy_m.append([])

        # Step 6 — mesh. Tag 9 polygon path follows the trimmed
        # footprint shape (so wall notches / holes carry through into
        # the stair mesh). When wall subtraction was a no-op the
        # polygon equals the user bbox and the result is the same as
        # the rectangular-box mesh, so we always go through the polygon
        # builder for consistency.
        from shapely.affinity import scale as _shapely_scale  # noqa: PLC0415
        try:
            flipped_polygon = _shapely_scale(
                trimmed.polygon, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0),
            )
            mesh = build_stair_mesh_from_polygon(record, flipped_polygon)
            if mesh.flight is None:
                # Fallback to the legacy rectangular builder.
                mesh = build_stair_mesh(record)
                if mesh.flight is not None:
                    record.notes.append(
                        "mesh: polygon-slice returned empty, used rectangular fallback"
                    )
        except Exception as exc:
            record.notes.append(f"polygon-mesh failed ({exc}), using rect")
            mesh = build_stair_mesh(record)
        if mesh.flight is None:
            record.status = StairStatus.FAILED_MESH_BUILD
            record.notes.append("mesh-builder returned empty flight")

        results.append(StairPipelineResult(
            record=record,
            mesh=mesh if mesh.flight is not None else None,
            pixel_details=pixel_details,
            trimmed_polygon_exterior_m=list(trimmed.exterior_xy()),
            trimmed_polygon_holes_m=[list(h) for h in trimmed.hole_rings_xy()],
            user_polygon_m=list(poly_m),
            wall_subtraction_used=bool(trimmed.used_wall_subtraction),
            tread_centerlines_xy_m=tread_centerlines_xy_m,
            step_polygons_xy_m=step_polygons_xy_m,
            hough_tread_lines_xy_m=list(hough_details.tread_lines_xy_m),
            stringer_lines_xy_m=list(hough_details.stringer_lines_xy_m),
            cutlines_xy_m=list(hough_details.cutlines_xy_m),
            n_steps_source=n_steps_source,
            n_steps_confidence=float(n_steps_confidence),
            n_steps_voters=dict(n_steps_voter_log),
        ))

    # V2 postprocess — replaces the V1 mesh with the validated cell-based
    # stair (locked algorithm: extract_steps_v2 + per-CC arrow detector +
    # junction-step detector + R1-R3 target rules + nosed wood slabs).
    # Runs only when an image is available; falls back to V1 silently per
    # stair when V2 cannot extract cells (so legacy / synthetic inputs
    # stay on V1 and the 193 existing stair tests keep passing).
    if image_gray is not None:
        _apply_v2_postprocess(
            results=results,
            annotations=annotations,
            scale_xy=scale_xy,
            px_to_m=px_to_m,
            image_gray=image_gray,
            all_annotation_polys_px=all_annotation_polys_px,
            pipeline_default_floor_m=pipeline_default_m,
            floor_to_floor_config=floor_to_floor_config,
        )

    return results


def _apply_v2_postprocess(
    *,
    results: List[StairPipelineResult],
    annotations: Sequence[Dict[str, Any]],
    scale_xy: Tuple[float, float],
    px_to_m: float,
    image_gray: np.ndarray,
    all_annotation_polys_px: Sequence[Tuple[str, List[Tuple[float, float]]]],
    pipeline_default_floor_m: float,
    floor_to_floor_config: Optional[float],
) -> None:
    """In-place V2 algorithm postprocess.

    For every successful result, re-extract treads as cells (heatmap +
    ink division lines), detect climb direction per-CC, run a
    junction-step detector across all stairs, apply R1-R3 target rules
    to decide whether a flight reaches the next storey or stops at an
    intermediate level, and rebuild the flight mesh as nosed wood slabs.

    Algorithm itself is locked (see ``stair_v2_cells.py`` /
    ``stair_v2_arrow.py`` / ``stair_v2_target_rules.py``); only the mesh
    design in ``build_v2_flight_mesh`` is iterated.
    """
    # Deferred imports keep the formatter from stripping them as "unused".
    from .mesh_builder import build_v2_flight_mesh as _build_v2_flight  # noqa: PLC0415
    from .stair_confidence_field import (  # noqa: PLC0415
        build_confidence_field as _build_v2_cf,
    )
    from .stair_v2_arrow import (  # noqa: PLC0415
        detect_arrow_direction_v2 as _v2_arrow,
    )
    from .stair_v2_cells import extract_steps_v2 as _v2_cells  # noqa: PLC0415
    from .stair_v2_target_rules import (  # noqa: PLC0415
        DIN_RISE_M,
        classify_stair_target as _v2_target,
    )

    from shapely.geometry import Polygon as _ShPoly  # noqa: PLC0415

    if not results:
        return

    # Map each successful result to its raw annotation polygon so the V2
    # extractor sees the same gravity well the user labelled.
    stair_anns: List[Dict[str, Any]] = [
        a for a in (annotations or [])
        if (a.get("class") or "").strip() in STAIR_LABEL_CLASSES
    ]
    if len(stair_anns) != len(results):
        return  # bail safely on any iteration-order mismatch

    H_img, W_img = image_gray.shape[:2]
    extras: Dict[int, Dict[str, Any]] = {}

    for idx, (res, ann) in enumerate(zip(results, stair_anns)):
        if res.record.status != StairStatus.OK:
            continue
        poly_px_full = _poly_from_annotation(ann, scale_xy)
        if len(poly_px_full) < 3:
            continue
        poly_arr = np.asarray(poly_px_full, dtype=np.float64)
        # Crop the plan around the stair bbox so V2 runs on a small
        # window (~1 MP) instead of the full plan (~95 MP).
        pad_px = max(16, int(round(0.30 / max(px_to_m, 1e-9))))
        x_lo = max(0, int(math.floor(poly_arr[:, 0].min())) - pad_px)
        x_hi = min(W_img, int(math.ceil(poly_arr[:, 0].max())) + pad_px)
        y_lo = max(0, int(math.floor(poly_arr[:, 1].min())) - pad_px)
        y_hi = min(H_img, int(math.ceil(poly_arr[:, 1].max())) + pad_px)
        if x_hi <= x_lo or y_hi <= y_lo:
            continue
        crop = image_gray[y_lo:y_hi, x_lo:x_hi]
        if crop.size == 0:
            continue
        # V2 arrow detector + cell extractor were validated at the
        # prototype's effective scale (px_to_m~0.001 m/px, i.e. PDF
        # render at M=10). At the production scale (M=4 -> px_to_m
        # ~0.0027) the 7x7 OPEN kernel inside the arrow detector
        # erodes too much of the 3px arrow stem, and the triangle-base
        # asymmetry that anchors the direction sign collapses — both
        # legs of the U-pair would invert. Upsample the crop to the
        # prototype's effective scale before running the locked V2
        # functions. The algorithm is unchanged; only its input
        # resolution is normalised.
        import cv2  # noqa: PLC0415
        TARGET_PX_TO_M = 0.0011
        if px_to_m > TARGET_PX_TO_M * 1.3:
            scale_up = float(px_to_m / TARGET_PX_TO_M)
            new_h = max(2, int(round(crop.shape[0] * scale_up)))
            new_w = max(2, int(round(crop.shape[1] * scale_up)))
            crop_v2 = cv2.resize(
                crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC,
            )
            px_to_m_v2 = float(px_to_m / scale_up)
        else:
            scale_up = 1.0
            crop_v2 = crop
            px_to_m_v2 = px_to_m
        poly_local = np.array(
            [[int(round((x - x_lo) * scale_up)),
              int(round((y - y_lo) * scale_up))]
             for x, y in poly_px_full],
            dtype=np.int32,
        )
        neighbour_local = []
        for aid, p in all_annotation_polys_px:
            if aid == res.record.label_id:
                continue
            neighbour_local.append(
                [(float((x - x_lo) * scale_up),
                  float((y - y_lo) * scale_up))
                 for x, y in p]
            )
        try:
            cf = _build_v2_cf(
                [(float(x), float(y)) for x, y in poly_local],
                neighbour_local,
                crop_v2.shape[:2],
                px_to_m=px_to_m_v2,
            )
        except Exception:
            continue
        direction_local = _v2_arrow(crop_v2, poly_local, px_to_m_v2)
        cells_local = _v2_cells(
            crop_v2, poly_local, px_to_m_v2, cf,
            arrow_dir=direction_local,
        )
        if not cells_local:
            continue
        if direction_local is None:
            # Fall back to the V1 walking-line direction so the mesh has
            # SOME consistent ranking direction.
            wl = res.record.direction_up_xy
            direction_local = np.array(
                [float(wl[0]), float(wl[1])], dtype=np.float64,
            )
        d_local = np.asarray(direction_local, dtype=np.float64)
        nrm = float(np.linalg.norm(d_local))
        if nrm <= 1e-9:
            continue
        d_local /= nrm
        # Convert each cell from upsampled-crop-px back to global plan
        # metres (undo the scale_up, then add the crop offset and scale
        # by the ORIGINAL px_to_m).
        ordered: List[Tuple[float, Any]] = []
        for c in cells_local:
            ring_m = [
                ((float(px) / scale_up + x_lo) * px_to_m,
                 (float(py) / scale_up + y_lo) * px_to_m)
                for px, py in c
            ]
            try:
                sp = _ShPoly(ring_m).buffer(0)
            except Exception:
                continue
            if sp.is_empty or sp.area <= 0.0:
                continue
            if sp.geom_type == "MultiPolygon":
                sp = max(sp.geoms, key=lambda g: g.area)
            cx, cy = sp.centroid.coords[0]
            t = float(cx * d_local[0] + cy * d_local[1])
            ordered.append((t, sp))
        if not ordered:
            continue
        ordered.sort(key=lambda kv: kv[0])
        # ORIGINAL storey height (pre-V1-half_flight_heuristic). V1 halves
        # the record's total_height when it sees rise > 30cm on a default
        # 2.95m storey — that assumption is correct for a SINGLE half-flight
        # interpretation but contradicts V2's own paired-U model where each
        # leg is built START..END through the full storey and the junction
        # step caps it. Resolve from annotation meta -> config -> pipeline
        # default instead of trusting V1's mutated record value.
        ann_meta = ann.get("meta") or {}
        try:
            ann_height = ann_meta.get("total_height_m") or ann_meta.get("storey_height_m")
            ann_height = float(ann_height) if ann_height not in (None, "") else None
        except (TypeError, ValueError):
            ann_height = None
        original_floor_h = (
            ann_height
            if ann_height is not None and ann_height > 0.0
            else (float(floor_to_floor_config)
                  if floor_to_floor_config is not None
                  else float(pipeline_default_floor_m))
        )
        extras[idx] = {
            "ordered": ordered,
            "d": d_local,
            "floor_h": original_floor_h,
        }

    if not extras:
        return

    # ---- SAFE cross-stair junction-step detector (locked logic) ----
    junction_for: Dict[int, Any] = {}
    paired_idx: set = set()
    indices = sorted(extras.keys())
    for ii in range(len(indices)):
        for jj in range(ii + 1, len(indices)):
            i = indices[ii]
            j = indices[jj]
            a = extras[i]["ordered"][-1][1]
            b = extras[j]["ordered"][-1][1]
            gap = a.distance(b)
            if not (0.01 < gap < 0.45):
                continue
            try:
                bridge = a.union(b).convex_hull.difference(a.union(b))
            except Exception:
                continue
            parts = list(getattr(bridge, "geoms", [bridge]))
            parts = [
                p for p in parts
                if p.geom_type == "Polygon" and p.area > 0.04
            ]
            if not parts:
                continue
            land = max(parts, key=lambda p: p.area)
            if land.area > 2.5:
                continue
            paired_idx.update([i, j])
            junction_for[i] = land

    # ---- R1-R3 target classification + V2 mesh build per stair ----
    for idx, info in extras.items():
        n = len(info["ordered"])
        target_label, top_m = _v2_target(
            n,
            is_paired=idx in paired_idx,
            floor_height_m=info["floor_h"],
            din_rise_m=DIN_RISE_M,
        )
        rise_m = top_m / n if n > 0 else 0.0
        if rise_m <= 0.0:
            continue
        cells_xy_m = [list(sp.exterior.coords) for _, sp in info["ordered"]]
        junction_step_xy_m = None
        junction_top_m = None
        if idx in junction_for:
            junction_step_xy_m = list(junction_for[idx].exterior.coords)
            junction_top_m = top_m
        flight = _build_v2_flight(
            cells_xy_m,
            rise_m=rise_m,
            direction_xy=(float(info["d"][0]), float(info["d"][1])),
            junction_step_xy_m=junction_step_xy_m,
            junction_top_m=junction_top_m,
        )
        if flight is None:
            continue
        res = results[idx]
        if res.mesh is None:
            res.mesh = StairMesh()
        res.mesh.flight = flight
        # Record updates: n_steps, going/rise, V2-target note.
        if n != res.record.n_steps:
            res.record.notes.append(
                f"v2_cells: n_steps={res.record.n_steps}->{n}"
            )
            res.record.n_steps = n
        # Override V1's half-flight-halved total_height ONLY when V2's
        # junction detector confirms this is a paired U-stair. For solo
        # flights V1's height resolution (annotation > config > default,
        # with the half-flight heuristic) remains authoritative — it
        # encodes building-context info V2 doesn't have access to.
        if (
            target_label == "next_floor_pair"
            and abs(top_m - res.record.total_height_m) > 0.01
        ):
            res.record.notes.append(
                f"v2_total_height: {res.record.total_height_m:.2f}m -> {top_m:.2f}m"
                " (U-pair confirmed, V1 half-flight halving undone)"
            )
            res.record.total_height_m = float(top_m)
        res.record.rise_m = float(rise_m)
        # Going is computed along the climb axis across the run.
        try:
            xs = [
                float(c[0] * info["d"][0] + c[1] * info["d"][1])
                for sp in (kv[1] for kv in info["ordered"])
                for c in sp.exterior.coords
            ]
            if xs:
                res.record.going_m = float((max(xs) - min(xs)) / max(n, 1))
        except Exception:
            pass
        res.record.notes.append(
            f"v2_postprocess: target={target_label} top={top_m:.2f}m "
            f"rise={rise_m:.3f}m n={n}"
            + ("  + junction_step" if idx in junction_for else "")
        )


def to_semantics_dict_list(results: Sequence[StairPipelineResult]) -> List[Dict[str, Any]]:
    """Project orchestrator results into the semantics_v3.json shape.

    Convenience for the merge-script integration so it does not have
    to know about ``StairPipelineResult`` internals.
    """
    return [r.record.to_semantics_dict() for r in results]
