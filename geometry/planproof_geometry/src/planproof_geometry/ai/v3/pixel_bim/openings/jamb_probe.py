"""Shared three-tier jamb-anchor probe used by doors and windows.

Doors and windows both need to ask the same question — does the
short-axis end of an opening rectangle touch wall material? The
answer drives placement scoring (which candidate edge wins) and
post-placement validation (does the opening actually sit between two
jambs?).

The probe runs a three-tier cascade so it is forgiving of
annotation gaps without losing precision:

  1. **Polygon test** — a small probe rectangle (long-axis end +
     ``snap_tolerance_m`` buffer) is intersected against every
     ``wall_polygon``. Cheap, exact, but punished by Phase-2 morpho-
     close that drops thin walls from the polygon set.

  2. **Wall-mask test** — when the polygon test misses, the same
     probe rectangle is rasterised into ``wall_mask`` and the fraction
     of wall pixels (≥ ``density_threshold``) decides. Catches walls
     present in the raster but absent from the polygon list.

  3. **Dark-pixel test** — when both polygon and mask miss, the same
     rectangle is sampled against ``image_gray`` and pixels darker
     than ``dark_threshold`` are counted. Catches walls visible in
     the raw PDF that Phase-1 annotation skipped entirely (so they
     are absent from both ``wall_polygons`` AND ``wall_mask``).

The signature is opening-class-agnostic: ``polygon_xy`` is a 4-corner
rectangle and the long axis is auto-detected from the edge lengths.
``wall_polygons`` is duck-typed on the ``shell_xy`` attribute so both
``WallPolygon`` instances and any compatible record work.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon


# Default thresholds — tuned by the door pipeline (Tag-7.10 .. Tag-7.16)
# and re-used unchanged by windows so the rule is the same.
DEFAULT_SNAP_TOL_M = 0.10
DEFAULT_PROBE_ALONG_M = 0.06
DEFAULT_DARK_THRESHOLD = 130
DEFAULT_DENSITY_THRESHOLD = 0.10


def _long_axis_basis(
    polygon_xy: Sequence[Tuple[float, float]],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
    """Detect the long axis of a 4-corner rectangle and return the basis.

    Returns ``(center, long_unit, short_unit, long_len, short_len)`` or
    ``None`` when the polygon is degenerate.
    """
    pts = np.asarray(polygon_xy, dtype=np.float64)
    if pts.shape[0] < 4:
        return None
    e01 = pts[1] - pts[0]
    e12 = pts[2] - pts[1]
    n01 = float(np.linalg.norm(e01))
    n12 = float(np.linalg.norm(e12))
    if n01 >= n12:
        long_vec, long_len, short_len = e01, n01, n12
    else:
        long_vec, long_len, short_len = e12, n12, n01
    if long_len < 1e-6:
        return None
    long_u = long_vec / long_len
    short_u = np.array([-long_u[1], long_u[0]], dtype=np.float64)
    return pts.mean(axis=0), long_u, short_u, long_len, short_len


def _end_probe_corners_m(
    center: np.ndarray,
    long_u: np.ndarray,
    short_u: np.ndarray,
    long_len: float,
    short_len: float,
    end_sign: float,
    snap_tolerance_m: float,
    probe_along_m: float,
) -> np.ndarray:
    """Four corners of the probe rectangle at one short-axis end.

    Centred ``probe_along_m / 2`` outside the rectangle along ``long_u``
    and inflated by ``snap_tolerance_m`` on every side so the polygon
    intersection test forgives tiny gaps.
    """
    half_W = long_len * 0.5
    half_S = max(short_len, 0.10) * 0.5
    probe_long = max(probe_along_m, 0.03)
    end_center = center + end_sign * (half_W + probe_long * 0.5) * long_u
    half_long = probe_long * 0.5 + snap_tolerance_m
    half_short = half_S + snap_tolerance_m
    return np.array([
        end_center + half_long * long_u + half_short * short_u,
        end_center + half_long * long_u - half_short * short_u,
        end_center - half_long * long_u - half_short * short_u,
        end_center - half_long * long_u + half_short * short_u,
    ], dtype=np.float64)


def _pixel_mask_density_in_corners(
    corners_m: np.ndarray,
    raster: np.ndarray,
    px_to_m: float,
    *,
    threshold_value: int,
    threshold_op: str,
) -> Optional[float]:
    """Density of pixels in ``raster`` matching ``threshold_op`` over the
    rasterised polygon defined by ``corners_m``. Returns ``None`` when
    the rectangle is fully outside the image."""
    import cv2

    H, W = raster.shape[:2]
    corners_px = np.empty_like(corners_m, dtype=np.int32)
    corners_px[:, 0] = np.round(corners_m[:, 0] / px_to_m).astype(np.int32)
    corners_px[:, 1] = np.round(corners_m[:, 1] / px_to_m).astype(np.int32)
    if (corners_px[:, 0].max() < 0 or corners_px[:, 1].max() < 0
            or corners_px[:, 0].min() >= W or corners_px[:, 1].min() >= H):
        return None

    rect_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(rect_mask, [corners_px.reshape(-1, 1, 2)], 255)
    n_total = int((rect_mask > 0).sum())
    if n_total == 0:
        return None
    if threshold_op == "gt":
        n_match = int(((rect_mask > 0) & (raster > threshold_value)).sum())
    elif threshold_op == "lt":
        n_match = int(((rect_mask > 0) & (raster < threshold_value)).sum())
    else:
        return None
    return n_match / n_total


def _pixel_mask_at_jamb_end(
    polygon_xy: Sequence[Tuple[float, float]],
    end_sign: float,
    wall_mask: np.ndarray,
    px_to_m: float,
    *,
    snap_tolerance_m: float,
    probe_along_m: float,
    density_threshold: float = DEFAULT_DENSITY_THRESHOLD,
) -> bool:
    """Tier-2: probe ``wall_mask`` for wall pixels at the requested end.

    Mirrors the Tag-7.14 door helper. Catches walls present in the
    raster mask but missing from the polygon list (Phase-2 morphological
    closing drops thin walls). Anchored when the fraction of
    ``wall_mask > 0`` pixels inside the probe rectangle exceeds
    ``density_threshold``.
    """
    if px_to_m <= 0:
        return False
    basis = _long_axis_basis(polygon_xy)
    if basis is None:
        return False
    center, long_u, short_u, long_len, short_len = basis
    corners_m = _end_probe_corners_m(
        center, long_u, short_u, long_len, short_len,
        end_sign, snap_tolerance_m, probe_along_m,
    )
    density = _pixel_mask_density_in_corners(
        corners_m, wall_mask, px_to_m,
        threshold_value=0, threshold_op="gt",
    )
    return density is not None and density >= density_threshold


def _dark_pixel_at_jamb_end(
    polygon_xy: Sequence[Tuple[float, float]],
    end_sign: float,
    image_gray: np.ndarray,
    px_to_m: float,
    *,
    snap_tolerance_m: float,
    probe_along_m: float,
    dark_threshold: int = DEFAULT_DARK_THRESHOLD,
    density_threshold: float = DEFAULT_DENSITY_THRESHOLD,
) -> bool:
    """Tier-3: probe ``image_gray`` for dark pixels at the requested end.

    Mirrors the Tag-7.16 door helper. Catches walls visible in the raw
    PDF that never made it into ``wall_polygons`` OR ``wall_mask``
    (Phase-1 annotation skipped them). Anchored when the fraction of
    pixels with ``image_gray < dark_threshold`` inside the probe
    rectangle exceeds ``density_threshold``.

    Trade-off: text and dimension lines also produce dark pixels; the
    small probe footprint and 10 % density floor keep false positives
    rare in practice.
    """
    if px_to_m <= 0:
        return False
    basis = _long_axis_basis(polygon_xy)
    if basis is None:
        return False
    center, long_u, short_u, long_len, short_len = basis
    corners_m = _end_probe_corners_m(
        center, long_u, short_u, long_len, short_len,
        end_sign, snap_tolerance_m, probe_along_m,
    )
    density = _pixel_mask_density_in_corners(
        corners_m, image_gray, px_to_m,
        threshold_value=dark_threshold, threshold_op="lt",
    )
    return density is not None and density >= density_threshold


def jamb_anchor_status(
    polygon_xy: Sequence[Tuple[float, float]],
    wall_polygons: Sequence,
    *,
    snap_tolerance_m: float = DEFAULT_SNAP_TOL_M,
    probe_along_m: float = DEFAULT_PROBE_ALONG_M,
    wall_mask: Optional[np.ndarray] = None,
    px_to_m: Optional[float] = None,
    image_gray: Optional[np.ndarray] = None,
) -> Tuple[bool, bool]:
    """Per-side jamb anchor status. Returns ``(anchored_minus,
    anchored_plus)`` — which long-axis end touches wall material.

    ``wall_polygons`` is duck-typed on ``shell_xy`` (any record with a
    list-of-(x,y) shell works). The probe runs three tiers in order;
    each tier only fires for sides the previous tier has not already
    anchored:

      1. polygon intersection (always runs)
      2. wall-mask raster density (when ``wall_mask`` and ``px_to_m``
         provided)
      3. raw-PDF dark-pixel density (when ``image_gray`` and
         ``px_to_m`` provided)

    All thresholds match the door pipeline so doors and windows share
    one rule.
    """
    basis = _long_axis_basis(polygon_xy)
    if basis is None:
        return False, False
    center, long_u, short_u, long_len, short_len = basis

    walls_shp = []
    for wp in wall_polygons:
        shell = getattr(wp, "shell_xy", None)
        if shell is None or len(shell) < 3:
            continue
        try:
            poly = Polygon(shell).buffer(0)
        except Exception:
            continue
        if poly.is_valid and not poly.is_empty:
            walls_shp.append(poly)
    flags: List[bool] = []
    for sign in (-1.0, +1.0):
        corners_m = _end_probe_corners_m(
            center, long_u, short_u, long_len, short_len,
            sign, snap_tolerance_m, probe_along_m,
        )
        try:
            probe_poly = Polygon(
                [(float(c[0]), float(c[1])) for c in corners_m]
            )
        except Exception:
            flags.append(False)
            continue
        if probe_poly.is_empty or not probe_poly.is_valid:
            flags.append(False)
            continue
        flags.append(any(probe_poly.intersects(wp) for wp in walls_shp))

    if wall_mask is not None and px_to_m is not None and px_to_m > 0:
        for sign_idx, sign in enumerate((-1.0, +1.0)):
            if flags[sign_idx]:
                continue
            try:
                if _pixel_mask_at_jamb_end(
                    polygon_xy, sign, wall_mask, px_to_m,
                    snap_tolerance_m=snap_tolerance_m,
                    probe_along_m=probe_along_m,
                ):
                    flags[sign_idx] = True
            except Exception:
                pass

    if image_gray is not None and px_to_m is not None and px_to_m > 0:
        for sign_idx, sign in enumerate((-1.0, +1.0)):
            if flags[sign_idx]:
                continue
            try:
                if _dark_pixel_at_jamb_end(
                    polygon_xy, sign, image_gray, px_to_m,
                    snap_tolerance_m=snap_tolerance_m,
                    probe_along_m=probe_along_m,
                ):
                    flags[sign_idx] = True
            except Exception:
                pass

    return bool(flags[0]), bool(flags[1])


def count_jamb_anchors(
    polygon_xy: Sequence[Tuple[float, float]],
    wall_polygons: Sequence,
    *,
    snap_tolerance_m: float = DEFAULT_SNAP_TOL_M,
    probe_along_m: float = DEFAULT_PROBE_ALONG_M,
    wall_mask: Optional[np.ndarray] = None,
    px_to_m: Optional[float] = None,
    image_gray: Optional[np.ndarray] = None,
) -> int:
    """Sum of ``jamb_anchor_status`` — total number of anchored ends."""
    minus, plus = jamb_anchor_status(
        polygon_xy, wall_polygons,
        snap_tolerance_m=snap_tolerance_m,
        probe_along_m=probe_along_m,
        wall_mask=wall_mask,
        px_to_m=px_to_m,
        image_gray=image_gray,
    )
    return int(minus) + int(plus)
