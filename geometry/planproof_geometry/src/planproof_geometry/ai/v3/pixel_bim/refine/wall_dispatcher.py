"""Mode-aware wall refinement dispatcher.

Routes between two existing wall refiners based on the plan's drawing
style:

* **paired_outline** (Matterport-style: two thin parallel strokes with a
  white wall body in between)  →  ``refine_label_classical`` per polygon.
  The classical refiner with style-detection handles paired outlines
  well: Otsu inside the polygon + morph-close + fill_holes + LCC + AND
  with the user polygon → pixel-accurate wall body clipped to the user
  authority region.

* **filled** (solid dark wall body)  →  ``refine_walls_ml_gravity``.
  The ML wall-seg model was trained primarily on filled walls; it finds
  the wall body directly. The gravity-well gate keeps user authority.

Mode is auto-detected once per plan from the Otsu mask intersected with
the dilated union of user polygons (= ROI). Detection is fast (~50 ms)
and only feeds the dispatch decision; it never mutates wall geometry.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from ..extrude.paired_outline import detect_paired_outline_mode
from .classical import refine_label_classical
from .ml_gravity import refine_walls_ml_gravity
from .polygon_axis_correct import axis_correct_user_polygon_px

logger = logging.getLogger(__name__)


def _build_user_polygon_mask(
    polygons_px: Sequence[Sequence[Tuple[float, float]]],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    for poly in polygons_px:
        if len(poly) < 3:
            continue
        pts = np.array(
            [[int(round(x)), int(round(y))] for x, y in poly],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def _cleanup_paired_union(
    union_mask: np.ndarray,
    user_polygon_union: np.ndarray,
    *,
    close_kernel_px: int,
    authority_dilate_px: int,
) -> Tuple[np.ndarray, dict]:
    """Stitch per-polygon refined masks into a clean global wall mask.

    The per-polygon classical refiner clips each label individually, so
    where two adjacent user polygons abut without overlap a thin gap is
    left between them; L-/T-junction corners stay unfilled; wall ends
    where the paired outlines don't quite close keep an open mouth.

    This pass runs *after* the per-polygon OR. It:

      1. closes hairline gaps between adjacent polygons (morph-close)
      2. fills paired-outline interiors that remained partially open
         (flood-fill from corner, invert holes, OR back in)
      3. clamps the result to the user-polygon union dilated by an
         authority tolerance, so the cleanup never grows past where
         the user actually drew walls
    """
    H, W = union_mask.shape[:2]
    closed = union_mask
    if close_kernel_px > 0:
        # MORPH_RECT — preserves sharp 90° corners; ELLIPSE would round them
        # off, which is the dominant artefact at L-/T-junctions in BIM.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_kernel_px, close_kernel_px),
        )
        closed = cv2.morphologyEx(union_mask, cv2.MORPH_CLOSE, kernel)

    # flood-fill from a guaranteed-background pixel, invert to get holes.
    # Pad by 1 so the seed (0,0) is always background even if a wall touches
    # the image edge.
    padded = cv2.copyMakeBorder(closed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 128)
    holes = (flood == 0).astype(np.uint8)[1:-1, 1:-1] * 255
    filled = cv2.bitwise_or(closed, holes)

    if authority_dilate_px > 0:
        k = max(1, int(authority_dilate_px))
        # MORPH_RECT keeps the dilated authority region axis-aligned;
        # ELLIPSE would create a ring of bowed-out wall body around every
        # corner of the user polygon, which is exactly the outward-drift
        # Willi spotted on the magenta overlay.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        authority = cv2.dilate(user_polygon_union, kernel, iterations=1)
    else:
        authority = user_polygon_union
    clipped = cv2.bitwise_and(filled, authority)

    info = {
        "cleanup_close_kernel_px": int(close_kernel_px),
        "cleanup_authority_dilate_px": int(authority_dilate_px),
        "cleanup_pixels_added_by_close": int(((closed > 0) & (union_mask == 0)).sum()),
        "cleanup_pixels_added_by_fill": int(((filled > 0) & (closed == 0)).sum()),
        "cleanup_pixels_clipped_by_authority": int(((filled > 0) & (clipped == 0)).sum()),
    }
    return clipped, info


def _subtract_door_polygons(
    wall_poly_px: Sequence[Tuple[float, float]],
    door_polys_px: Sequence[Sequence[Tuple[float, float]]],
    *,
    door_buffer_px: float = 1.0,
    min_piece_area_px: float = 100.0,
) -> list:
    """Subtract every door annotation from a wall annotation.

    Returns a list of result-polygons (in pixel coords). Most walls
    return one piece; when a door splits a wall in the middle, two
    pieces come back. ``door_buffer_px`` slightly inflates each door
    polygon so a hairline shared edge is fully removed; ``min_piece_
    area_px`` discards splinters from misaligned overlap.
    """
    from shapely.geometry import Polygon  # lazy import
    from shapely.ops import unary_union

    if not door_polys_px:
        return [list(wall_poly_px)]
    try:
        wall = Polygon(wall_poly_px).buffer(0)
    except Exception:
        return [list(wall_poly_px)]
    if wall.is_empty or not wall.is_valid:
        return [list(wall_poly_px)]
    door_geoms = []
    for d in door_polys_px:
        if len(d) < 3:
            continue
        try:
            dp = Polygon(d).buffer(door_buffer_px)
        except Exception:
            continue
        if not dp.is_empty and dp.is_valid:
            door_geoms.append(dp)
    if not door_geoms:
        return [list(wall_poly_px)]
    union = unary_union(door_geoms)
    try:
        result = wall.difference(union)
    except Exception:
        return [list(wall_poly_px)]
    if result.is_empty:
        return []
    pieces = (
        result.geoms if result.geom_type == "MultiPolygon" else [result]
    )
    out = []
    for piece in pieces:
        if piece.is_empty or piece.area < min_piece_area_px:
            continue
        if not piece.is_valid:
            piece = piece.buffer(0)
            if piece.is_empty:
                continue
        coords = list(piece.exterior.coords)
        out.append([(float(x), float(y)) for x, y in coords])
    return out


def refine_walls(
    image_gray: np.ndarray,
    wall_polygons_px: Sequence[Sequence[Tuple[float, float]]],
    *,
    px_to_m: float,
    tolerance_m: float = 0.15,
    roi_dilate_px: int = 30,
    cleanup_close_m: float = 0.012,
    cleanup_authority_m: float = 0.015,
    phantom_min_coverage: float = 0.20,
    door_polygons_px: Optional[
        Sequence[Sequence[Tuple[float, float]]]
    ] = None,
) -> Tuple[np.ndarray, dict]:
    """Refine wall pixels via mode-aware dispatch.

    Parameters
    ----------
    image_gray
        Full-resolution grayscale image (H, W) uint8.
    wall_polygons_px
        User wall polygons in pixel coordinates. May be empty.
    px_to_m
        Metres per pixel.
    tolerance_m
        Outward tolerance band for the gravity well (filled-mode only).
    roi_dilate_px
        Padding around the user-polygon union when sampling Otsu ink for
        mode detection. Slightly bigger than the typical wall thickness
        so paired outlines outside a tight polygon still register.

    Returns
    -------
    wall_mask
        uint8 (0/255) full-resolution mask of refined wall pixels.
    info
        Diagnostic dict with at least ``mode`` (``"paired"`` | ``"filled"``
        | ``"empty"``), plus mode-specific counters from the underlying
        refiner.
    """
    H, W = image_gray.shape[:2]
    info: dict = {
        "mode": "empty",
        "paired_ratio": 0.0,
        "n_polygons": len(wall_polygons_px),
    }
    if not wall_polygons_px:
        return np.zeros((H, W), dtype=np.uint8), info
    if px_to_m <= 0:
        raise ValueError("px_to_m must be > 0")

    # Tag-7 Fix 3: subtract door annotations from each wall annotation
    # before classical refine. Sliding doors and double doors are often
    # drawn IN-WALL (the door symbol overlaps the wall annotation), and
    # the symbol's tick marks / arc / leaf produce stray ink inside the
    # wall ROI that the classical refiner then trips over — leaving a
    # bite out of the wall body or a slanted end-cap. Subtracting the
    # door polygons up-front gives a wall-only ROI.
    info["door_subtraction_applied"] = 0
    if door_polygons_px:
        new_walls: list[Sequence[Tuple[float, float]]] = []
        for wp in wall_polygons_px:
            pieces = _subtract_door_polygons(wp, door_polygons_px)
            if pieces:
                new_walls.extend(pieces)
                if len(pieces) != 1 or len(pieces[0]) != len(wp):
                    info["door_subtraction_applied"] += 1
            else:
                new_walls.append(list(wp))
        wall_polygons_px = new_walls
        info["n_polygons"] = len(wall_polygons_px)

    # Tag-9 Fix: orthogonalize slightly-tilted user polygons before they
    # become the authority for refine + extrude. Mouse-drawn polygons are
    # often 4-7° off-axis; the post-refine axis_snap stage uses a 5°
    # threshold and misses these, so the BIM polygon inherits the tilt
    # via authority-reclip and renders crooked. The axis-majority gate
    # (>=85% of edge length within 15° of an axis) preserves authentic
    # diagonals (verandas, angled bays).
    info["axis_corrected"] = 0
    info["axis_correct_per_polygon"] = []
    corrected_walls: list[Sequence[Tuple[float, float]]] = []
    for wp in wall_polygons_px:
        snapped, ax_info = axis_correct_user_polygon_px(wp, px_to_m=px_to_m)
        if ax_info["corrected"]:
            info["axis_corrected"] += 1
        info["axis_correct_per_polygon"].append(ax_info)
        corrected_walls.append(snapped)
    wall_polygons_px = corrected_walls
    info["wall_polygons_corrected_px"] = wall_polygons_px

    # Step 1+2: build ROI = dilated user-polygon union, then Otsu the
    # grayscale image inside the ROI to get the ink mask used for mode
    # detection. This ignores furniture, text and symbols outside the
    # labeled wall regions.
    user_mask = _build_user_polygon_mask(wall_polygons_px, image_gray.shape)
    if roi_dilate_px > 0:
        k = max(1, int(roi_dilate_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        roi_mask = cv2.dilate(user_mask, kernel, iterations=1)
    else:
        roi_mask = user_mask

    _, otsu_full = cv2.threshold(
        image_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    roi_ink_mask = cv2.bitwise_and(otsu_full, roi_mask)

    # Step 3: paired-outline detection.
    px_per_m = 1.0 / px_to_m
    report = detect_paired_outline_mode(roi_ink_mask, px_per_m=px_per_m)
    info["paired_ratio"] = float(report.paired_ratio)
    info["paired_total_lines"] = int(report.total_lines)
    info["paired_pair_count"] = int(report.line_pair_count)

    if report.paired_mode:
        # Step 4a: classical route — one refiner call per polygon, OR
        # all refined masks. The classical refiner already clips to the
        # user polygon so the final mask honours user authority.
        info["mode"] = "paired"
        info["fallbacks"] = 0
        info["phantom_filtered"] = 0
        out = np.zeros((H, W), dtype=np.uint8)
        per_polygon_sources: list[str] = []
        per_polygon_coverage: list[float] = []
        phantom_indices: list[int] = []
        for idx, poly in enumerate(wall_polygons_px):
            if len(poly) < 3:
                continue
            try:
                refined = refine_label_classical(
                    image_gray,
                    label_id=f"wall_{idx}",
                    label_class="wall",
                    user_polygon_xy=poly,
                )
            except ValueError as exc:
                logger.warning("classical refine failed on poly %d: %s", idx, exc)
                continue
            # Phase E.1 phantom-wall filter: a wall annotation drawn over
            # a sliding-door lintel (or any non-wall element) only has ink
            # at the user-polygon ends, never along the long axis. Such a
            # polygon falls back to "fallback_no_ink" in the worst case,
            # or in the lintel-only case the classical refiner finds a
            # tiny strip of ink that covers <20% of the user polygon.
            # Drop these so phantom walls stop appearing in the BIM.
            cov = float(refined.area_ratio_vs_user_polygon)
            per_polygon_coverage.append(cov)
            if (
                refined.source == "fallback_no_ink"
                or refined.source == "fallback_empty_crop"
                or cov < phantom_min_coverage
            ):
                phantom_indices.append(idx)
                info["phantom_filtered"] += 1
                continue
            np.maximum(out, refined.refined_mask, out=out)
            per_polygon_sources.append(refined.source)
            if refined.source.startswith("fallback"):
                info["fallbacks"] += 1
        info["per_polygon_sources"] = per_polygon_sources
        info["per_polygon_coverage"] = per_polygon_coverage
        info["phantom_polygon_indices"] = phantom_indices

        # Step 5: stitch per-polygon masks into a clean global wall mask.
        # close hairline polygon-to-polygon gaps and L/T-junction corners,
        # fill paired-outline interiors that ended partially open, then
        # clamp to the dilated user-polygon union so user authority is
        # preserved.
        close_kernel_px = max(1, int(round(cleanup_close_m / px_to_m)))
        authority_dilate_px = max(1, int(round(cleanup_authority_m / px_to_m)))
        out, cleanup_info = _cleanup_paired_union(
            out, user_mask,
            close_kernel_px=close_kernel_px,
            authority_dilate_px=authority_dilate_px,
        )
        info.update(cleanup_info)
        return out, info

    # Step 4b: filled route — single ML+gravity call.
    info["mode"] = "filled"
    wall_mask, ml_info = refine_walls_ml_gravity(
        image_gray,
        wall_polygons_px,
        px_to_m=px_to_m,
        tolerance_m=tolerance_m,
    )
    info.update(ml_info)
    return wall_mask, info


__all__ = ["refine_walls"]
