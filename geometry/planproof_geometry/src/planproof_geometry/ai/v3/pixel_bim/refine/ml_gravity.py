"""ML wall segmentation + gravity-well refinement (Willi 2026-04-29).

Replaces the bbox-clipped classical refiner. Walls in the BIM no longer
trace the user polygons (with light pixel touch-up); they trace the
**actual wall pixels** the ML model finds, attracted toward the user
polygons by a gravity-well field.

Concept (Willi-Spec):
  * The user labels create a 2D terrain with pits (= gravity wells) at
    the labeled regions. The pit is FLAT-MAX inside each polygon and
    slopes linearly outward over a tolerance band; beyond the band the
    field is zero.
  * The wall-segmentation ML model is the BALL: it predicts per-pixel
    wall probability across the whole image. Pixels within a gravity
    well get to count as walls; pixels outside any well are discarded
    even if the model thinks they are walls (= gravity has no pull,
    the ball does not roll there).
  * User authority: every wall annotation must contribute wall pixels.
    If the ML model finds nothing inside (or near) a given polygon,
    the rasterized polygon itself becomes the fallback for that
    annotation. The user is never overruled, only refined.

This single function replaces ``refine_label_classical`` + per-label
``decide_fallback`` + per-label ``_rasterize_polygon`` for walls.
"""
from __future__ import annotations

import logging
from typing import Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _rasterize_polygon(
    polygon_xy: Sequence[Tuple[float, float]],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Rasterize one polygon into a uint8 mask (full image resolution)."""
    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    if len(polygon_xy) < 3:
        return mask
    pts = np.array(
        [[int(round(x)), int(round(y))] for x, y in polygon_xy],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def build_gravity_well(
    polygons_px: Sequence[Sequence[Tuple[float, float]]],
    image_shape: Tuple[int, int],
    *,
    tolerance_px: int,
) -> np.ndarray:
    """2D gravity-well field built from user wall polygons.

    Field semantics (per Willi 2026-04-29):
        well[y, x] = 1.0                       if (y,x) inside any polygon
                   = 1 - dist/tolerance_px     for 0 < dist < tolerance_px
                   = 0.0                       otherwise

    Returns a float32 array of shape (H, W).
    """
    H, W = image_shape[:2]
    user_mask = np.zeros((H, W), dtype=np.uint8)
    for poly in polygons_px:
        if len(poly) < 3:
            continue
        pts = np.array(
            [[int(round(x)), int(round(y))] for x, y in poly],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.fillPoly(user_mask, [pts], 255)
    inverted = (user_mask == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
    well = 1.0 - np.clip(dist / max(1.0, float(tolerance_px)), 0.0, 1.0)
    return well.astype(np.float32)


def _predict_probs_tiled(
    image_bgr: np.ndarray,
    model,
    *,
    overlap: float = 0.5,
) -> np.ndarray:
    """Tiled inference returning float32 probability map at original resolution.

    Mirrors the tiling logic in ``infer_wall_mask_tiled`` but skips the
    threshold step so we can multiply the raw probs with the gravity-well
    field. Necessary on plans larger than ~2k px on the long edge — global
    resize collapses wall strokes below the model's effective receptive field.
    """
    from planproof_geometry.ai.plan_wall_segmentation import (  # noqa: PLC0415
        _predict_probs_fullres,
    )

    tile = int(model.img_size)
    stride = max(1, int(round(tile * (1.0 - overlap))))
    h, w = image_bgr.shape[:2]
    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)
    if h + pad_h > tile:
        extra_h = (stride - ((h + pad_h - tile) % stride)) % stride
        pad_h += extra_h
    if w + pad_w > tile:
        extra_w = (stride - ((w + pad_w - tile) % stride)) % stride
        pad_w += extra_w

    padded = cv2.copyMakeBorder(
        image_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101,
    )
    ph, pw = padded.shape[:2]
    prob_acc = np.zeros((ph, pw), dtype=np.float32)
    count_acc = np.zeros((ph, pw), dtype=np.float32)

    y_positions = list(range(0, ph - tile + 1, stride))
    x_positions = list(range(0, pw - tile + 1, stride))
    if not y_positions or y_positions[-1] != ph - tile:
        y_positions.append(ph - tile)
    if not x_positions or x_positions[-1] != pw - tile:
        x_positions.append(pw - tile)

    for y in y_positions:
        for x in x_positions:
            tile_img = padded[y:y + tile, x:x + tile]
            probs, _ = _predict_probs_fullres(tile_img, model)
            prob_acc[y:y + tile, x:x + tile] += probs
            count_acc[y:y + tile, x:x + tile] += 1.0

    avg_probs = prob_acc / np.maximum(count_acc, 1e-6)
    return avg_probs[:h, :w]


def refine_walls_ml_gravity(
    image_gray: np.ndarray,
    wall_polygons_px: Sequence[Sequence[Tuple[float, float]]],
    *,
    px_to_m: float,
    tolerance_m: float = 0.15,
    ml_threshold: float = 0.30,
    min_ml_pixels_per_polygon: int = 30,
    tile_auto_threshold_px: int = 2048,
) -> Tuple[np.ndarray, dict]:
    """ML-detected wall pixels, gravity-well-biased toward user polygons.

    Parameters
    ----------
    image_gray : np.ndarray
        Full-resolution grayscale image (H, W) uint8.
    wall_polygons_px : list of list of (x, y)
        User wall polygons in pixel coordinates. May be empty (returns
        empty mask).
    px_to_m : float
        Metres per pixel — used to convert ``tolerance_m`` to pixels.
    tolerance_m : float
        Outward slope of the gravity well in metres (default 15 cm).
        Beyond this distance from any user polygon, ML pixels are dropped.
    ml_threshold : float
        Minimum ML wall-probability (after the well multiplier) to count
        as wall (default 0.30).
    min_ml_pixels_per_polygon : int
        If a user polygon receives fewer than this many ML pixels inside,
        the polygon is rasterized as fallback (Willi-rule: user is authority).
    tile_auto_threshold_px : int
        Use tiled inference when ``max(H, W) > tile_auto_threshold_px``.

    Returns
    -------
    wall_mask : np.ndarray
        uint8 (0/255) mask of refined walls at full resolution.
    info : dict
        Diagnostic counters: n_polygons, n_fallback, ml_pixels_in_field,
        ml_pixels_outside_field, well_area_px, used_tiled, threshold,
        tolerance_px.
    """
    H, W = image_gray.shape[:2]
    info: dict = {
        "n_polygons": len(wall_polygons_px),
        "n_fallback": 0,
        "ml_pixels_in_field": 0,
        "ml_pixels_outside_field": 0,
        "well_area_px": 0,
        "used_tiled": False,
        "threshold": float(ml_threshold),
        "tolerance_px": 0,
        "model_loaded": False,
    }
    if not wall_polygons_px:
        return np.zeros((H, W), dtype=np.uint8), info
    if px_to_m <= 0:
        raise ValueError("px_to_m must be > 0")

    tolerance_px = max(1, int(round(tolerance_m / px_to_m)))
    info["tolerance_px"] = tolerance_px

    well = build_gravity_well(
        wall_polygons_px, image_gray.shape, tolerance_px=tolerance_px,
    )
    info["well_area_px"] = int((well > 0.0).sum())

    # Run the ML wall-seg model.
    from planproof_geometry.ai.plan_wall_segmentation import (  # noqa: PLC0415
        _predict_probs_fullres, get_wall_seg_model,
    )

    model = None
    try:
        model = get_wall_seg_model()
    except Exception as exc:
        logger.warning("wall_seg load failed (%s) — fallback to polygons", exc)
    if model is None:
        logger.warning("wall_seg model unavailable — fallback to polygons")
        out = np.zeros((H, W), dtype=np.uint8)
        for poly in wall_polygons_px:
            np.maximum(out, _rasterize_polygon(poly, image_gray.shape), out=out)
        info["n_fallback"] = len(wall_polygons_px)
        return out, info
    info["model_loaded"] = True

    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    use_tiled = max(H, W) > tile_auto_threshold_px
    info["used_tiled"] = use_tiled
    if use_tiled:
        probs = _predict_probs_tiled(image_bgr, model, overlap=0.5)
    else:
        probs, _ = _predict_probs_fullres(image_bgr, model)

    # Combine: ML probability × gravity-well field.
    # The ball "rolls" — its score gets weighted by where the gravity is.
    biased = probs * well
    wall_pixels = (biased >= ml_threshold).astype(np.uint8) * 255

    info["ml_pixels_in_field"] = int(((probs >= ml_threshold) & (well > 0)).sum())
    info["ml_pixels_outside_field"] = int(((probs >= ml_threshold) & (well == 0)).sum())

    # Per-polygon authority fallback.
    out = wall_pixels.copy()
    for poly in wall_polygons_px:
        if len(poly) < 3:
            continue
        poly_mask = _rasterize_polygon(poly, image_gray.shape)
        ml_inside = int(((wall_pixels > 0) & (poly_mask > 0)).sum())
        if ml_inside < min_ml_pixels_per_polygon:
            np.maximum(out, poly_mask, out=out)
            info["n_fallback"] += 1

    return out, info


__all__ = [
    "build_gravity_well",
    "refine_walls_ml_gravity",
]
