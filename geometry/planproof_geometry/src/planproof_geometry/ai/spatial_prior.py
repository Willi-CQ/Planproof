"""Spatial Prior heatmaps from manual annotations ("Gravity-Well").

For each manual annotation we paint a 2D Gaussian into a per-class heatmap.
The peak height equals the annotation confidence; the radius (sigma) shrinks
with confidence so that high-confidence labels create *steep* attractors and
low-confidence labels create *gentle* hills.

The heatmap is consumed at inference time by ``apply_gravity_well`` in
``v2/detection_merge/merge.py`` to bias ML detection scores toward
human-labeled regions.

Math:

    sigma_px = sigma_scale * box_radius * (1 - confidence + sigma_floor)
    H_c[y, x] = max over annotations of class c:
        confidence * exp(-((x - cx)^2 + (y - cy)^2) / (2 * sigma_px^2))

Multiple overlapping boxes combine via element-wise ``max`` rather than sum
so the prior never exceeds 1.0.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Class index used internally by the prior heatmap. Order is fixed so that
# downstream lookups (gravity-well scoring) can map detection class -> index.
PRIOR_CLASS_ORDER: Tuple[str, ...] = (
    "wall",
    "door",
    "window",
    "stairs",
    "column",
    "room_label",
)


def _class_index(class_name: str) -> Optional[int]:
    try:
        return PRIOR_CLASS_ORDER.index(class_name)
    except ValueError:
        return None


def _bbox_center_radius_px(bbox_px: Sequence[float]) -> Tuple[float, float, float]:
    """Return (cx, cy, radius) in pixels for a bbox."""
    x1, y1, x2, y2 = (
        min(bbox_px[0], bbox_px[2]),
        min(bbox_px[1], bbox_px[3]),
        max(bbox_px[0], bbox_px[2]),
        max(bbox_px[1], bbox_px[3]),
    )
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    radius = math.hypot(x2 - x1, y2 - y1) * 0.5
    return cx, cy, radius


def _paint_gaussian(
    heatmap: np.ndarray,
    cx: float,
    cy: float,
    sigma: float,
    peak: float,
    radius_cutoff: float = 3.0,
) -> None:
    """In-place: paint a Gaussian bump into ``heatmap`` (combine via max).

    Only writes within ``radius_cutoff * sigma`` of the centre to keep
    the cost O(sigma^2) instead of O(H*W).
    """
    if sigma <= 0 or peak <= 0:
        return
    h, w = heatmap.shape
    rng = int(math.ceil(radius_cutoff * sigma))
    x0 = max(0, int(cx) - rng)
    x1 = min(w, int(cx) + rng + 1)
    y0 = max(0, int(cy) - rng)
    y1 = min(h, int(cy) + rng + 1)
    if x1 <= x0 or y1 <= y0:
        return

    ys = np.arange(y0, y1, dtype=np.float32) - cy
    xs = np.arange(x0, x1, dtype=np.float32) - cx
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)
    bump = peak * np.exp(-(xx * xx + yy * yy) * inv_2sigma2)

    # max-combine so overlapping boxes don't exceed peak
    region = heatmap[y0:y1, x0:x1]
    np.maximum(region, bump, out=region)


def build_prior_heatmap(
    manual_segments: Optional[List[Dict[str, Any]]],
    manual_detections: Optional[List[Dict[str, Any]]],
    image_size_px: Tuple[int, int],
    *,
    sigma_scale: float = 0.45,
    sigma_floor: float = 0.05,
    downsample: int = 2,
) -> Tuple[Optional[np.ndarray], Tuple[str, ...]]:
    """Build the prior heatmap stack from manual annotations.

    Args:
        manual_segments: Wall segments produced by the bridge (carry bbox_px in meta).
        manual_detections: Door/window/etc. produced by the bridge.
        image_size_px: (width, height) of the source image in pixels.
        sigma_scale: Multiplier on box radius for the Gaussian sigma.
        sigma_floor: Lower bound term so confidence==1.0 still has finite sigma.
        downsample: Resolution divisor (heatmap stored at lower resolution to
            keep memory bounded; values >1 reduce HxW by that factor).

    Returns:
        (heatmap, class_order) where heatmap has shape
        (len(class_order), H/downsample, W/downsample) and dtype float32, or
        ``(None, class_order)`` when no annotations exist.
    """
    width, height = int(image_size_px[0]), int(image_size_px[1])
    if width <= 0 or height <= 0:
        return None, PRIOR_CLASS_ORDER

    items: List[Tuple[str, List[float], float]] = []
    if manual_segments:
        for seg in manual_segments:
            cls = seg.get("class") or seg.get("meta", {}).get("manual_class_raw") or ""
            if cls.startswith("wall"):
                cls = "wall"
            bbox = seg.get("meta", {}).get("bbox_px")
            if not bbox:
                continue
            items.append((cls, list(bbox), float(seg.get("confidence", 1.0))))

    if manual_detections:
        for det in manual_detections:
            cls = det.get("class") or ""
            bbox = det.get("meta", {}).get("bbox_px")
            if not bbox:
                continue
            items.append((cls, list(bbox), float(det.get("confidence", 1.0))))

    if not items:
        return None, PRIOR_CLASS_ORDER

    ds = max(1, int(downsample))
    h_ds = max(1, height // ds)
    w_ds = max(1, width // ds)
    heatmap = np.zeros((len(PRIOR_CLASS_ORDER), h_ds, w_ds), dtype=np.float32)

    used_count = 0
    for cls, bbox, conf in items:
        idx = _class_index(cls)
        if idx is None:
            continue
        cx_px, cy_px, radius_px = _bbox_center_radius_px(bbox)
        cx_ds = cx_px / ds
        cy_ds = cy_px / ds
        radius_ds = max(1.0, radius_px / ds)
        # high conf -> small sigma (steep peak); low conf -> large sigma (soft hill)
        sigma_ds = sigma_scale * radius_ds * max(sigma_floor, 1.0 - conf + sigma_floor)
        _paint_gaussian(heatmap[idx], cx_ds, cy_ds, sigma_ds, conf)
        used_count += 1

    logger.info(
        "spatial_prior: heatmap %dx%d (ds=%d) with %d items across %d classes",
        h_ds, w_ds, ds, used_count, int(np.sum(heatmap.max(axis=(1, 2)) > 0)),
    )
    return heatmap, PRIOR_CLASS_ORDER


def sample_prior(
    heatmap: np.ndarray,
    class_order: Sequence[str],
    class_name: str,
    cx_px: float,
    cy_px: float,
    image_size_px: Tuple[int, int],
) -> float:
    """Bilinear-sample the per-class prior at a pixel position."""
    idx = _class_index(class_name) if class_name in PRIOR_CLASS_ORDER else None
    if idx is None or idx >= heatmap.shape[0]:
        return 0.0
    h_ds, w_ds = heatmap.shape[1], heatmap.shape[2]
    width, height = int(image_size_px[0]), int(image_size_px[1])
    if width <= 0 or height <= 0:
        return 0.0
    fx = cx_px * (w_ds / width)
    fy = cy_px * (h_ds / height)
    ix = int(np.clip(round(fx), 0, w_ds - 1))
    iy = int(np.clip(round(fy), 0, h_ds - 1))
    return float(heatmap[idx, iy, ix])
