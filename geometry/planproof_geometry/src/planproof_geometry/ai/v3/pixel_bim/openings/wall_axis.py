"""Measure the wall a door pierces by ray-casting perpendicular to the
door axis.

This is the single source of truth for two derived placement attributes
that the door geometry needs to sit centrally inside the wall:

  * ``wall_depth_m`` — the actual wall thickness at the door, measured
    against the post-refine wall mesh (NOT the polygon's short edge,
    which can drift after authority-reclip in ``mask_to_mesh``).
  * ``short_offset_m`` — signed offset of the wall centerline from the
    door polygon centre along +short_axis, so the frame/void/bridge can
    be translated onto the wall midplane.

Originally lived in ``scripts/verify_bim_view.py`` as a verification
helper. Promoted to library in Sprint Door-Wall-Bridge Tag-2 so the
place-door cascade can compute both values BEFORE the void is carved
(window-pipeline pattern).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


# Plausibility band for one wall cross-section (metres). Outside this
# range the ray clearly missed (too thin = no wall) or hit a multi-wall
# T-junction / concatenated polygon (too thick).
_THICKNESS_VALID_RANGE_M = (0.04, 0.50)

# When several rays return different thicknesses, drop any whose value
# is more than this × the minimum thickness — those rays travelled ALONG
# a perpendicular wall at a T-junction instead of CROSSING a parallel one.
_THICKNESS_OUTLIER_MULT = 1.5


def measure_wall_axis_at_door(
    door_polygon_xy: Sequence[Tuple[float, float]],
    wall_polygons,
    *,
    pad_min_m: float = 0.05,
    pad_frac: float = 0.05,
    sample_offsets_m: Sequence[float] = (0.0, 0.04, 0.08),
) -> Tuple[float, float]:
    """Cast perpendicular rays just past each long-axis end of the door
    and read the wall thickness + centerline offset.

    Returns ``(thickness_m, short_offset_m)``:
      * ``thickness_m`` — minimum non-zero ray length within the
        plausibility band (4-60 cm). The minimum is robust to
        T-junction artefacts that inflate one ray with a perpendicular
        wall.
      * ``short_offset_m`` — signed offset of the wall's centerline
        from the door polygon centre, projected onto +short_axis. Use
        this to translate the door geometry onto the wall midplane.

    Returns ``(0.0, 0.0)`` if no plausible wall hit (caller should
    fall back to the placement-time wall depth and zero offset).
    """
    from shapely.geometry import LineString, Polygon

    pts = np.asarray(door_polygon_xy, dtype=np.float64)
    if pts.shape[0] < 4:
        return 0.0, 0.0
    e01 = pts[1] - pts[0]
    e12 = pts[2] - pts[1]
    if np.linalg.norm(e01) >= np.linalg.norm(e12):
        long_vec, long_len = e01, float(np.linalg.norm(e01))
    else:
        long_vec, long_len = e12, float(np.linalg.norm(e12))
    if long_len < 1e-6:
        return 0.0, 0.0
    long_u = long_vec / long_len
    short_u = np.array([-long_u[1], long_u[0]])
    center = pts.mean(axis=0)

    shapely_walls = []
    for wp in wall_polygons:
        shell = getattr(wp, "shell_xy", None)
        if shell is None or len(shell) < 3:
            continue
        try:
            poly = Polygon(shell, holes=getattr(wp, "holes_xy", None)).buffer(0)
        except Exception:
            continue
        if poly.is_empty or poly.area <= 1e-6:
            continue
        shapely_walls.append(poly)
    if not shapely_walls:
        return 0.0, 0.0

    t_min, t_max = _THICKNESS_VALID_RANGE_M
    pad = max(pad_min_m, long_len * pad_frac)
    thicknesses = []
    offsets = []
    for sign in (-1.0, +1.0):
        for extra in sample_offsets_m:
            offset_long = sign * (long_len * 0.5 + pad + extra)
            sx = float(center[0]) + offset_long * long_u[0]
            sy = float(center[1]) + offset_long * long_u[1]
            line = LineString([
                (sx - 1.5 * short_u[0], sy - 1.5 * short_u[1]),
                (sx + 1.5 * short_u[0], sy + 1.5 * short_u[1]),
            ])
            best_seg, best_len = None, 0.0
            for poly in shapely_walls:
                inter = poly.intersection(line)
                if inter.is_empty:
                    continue
                geoms = (
                    [inter] if inter.geom_type == "LineString"
                    else list(inter.geoms) if hasattr(inter, "geoms")
                    else []
                )
                for g in geoms:
                    if g.geom_type != "LineString":
                        continue
                    if g.length > best_len:
                        best_len = float(g.length)
                        best_seg = g
            if best_seg is None or not (t_min <= best_len <= t_max):
                continue
            mid = best_seg.interpolate(0.5, normalized=True)
            off = ((mid.x - float(center[0])) * short_u[0]
                   + (mid.y - float(center[1])) * short_u[1])
            thicknesses.append(best_len)
            offsets.append(float(off))

    if not thicknesses:
        return 0.0, 0.0
    t_arr = np.asarray(thicknesses)
    o_arr = np.asarray(offsets)
    near = np.abs(o_arr) <= 0.30
    if not near.any():
        return float(t_arr.min()), 0.0
    t_arr, o_arr = t_arr[near], o_arr[near]
    t_min_v = float(t_arr.min())
    keep = t_arr <= t_min_v * _THICKNESS_OUTLIER_MULT
    if not keep.any():
        return t_min_v, 0.0
    return t_min_v, float(np.median(o_arr[keep]))


def measure_actual_wall_thickness(door_polygon_xy, wall_polygons, **kw) -> float:
    """Backwards-compatible thin wrapper returning thickness only."""
    t, _ = measure_wall_axis_at_door(door_polygon_xy, wall_polygons, **kw)
    return t


__all__ = [
    "measure_wall_axis_at_door",
    "measure_actual_wall_thickness",
]
