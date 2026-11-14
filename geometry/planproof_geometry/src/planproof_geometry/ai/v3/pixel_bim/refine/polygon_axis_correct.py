"""Orthogonalize slightly-tilted user wall polygons.

User-drawn wall annotations often have edges that are 4-7 degrees off the
horizontal/vertical axes due to mouse jitter — the user *meant* to draw
an axis-aligned wall but the polygon's vertices ended up slightly skew.

The post-refine ``axis_snap`` stage in ``extrude/axis_snap.py`` uses a 5
degree threshold and misses these edges. The BIM polygon then inherits
the tilt via the authority-reclip stage in ``extrude/mask_to_mesh.py``,
which clips the BIM polygon to the user-polygon dilated by 15 mm. The
result is a "crooked" BIM wall whose long edges are 1-3 degrees off
axis even though every other pipeline stage is pixel-accurate.

This module fixes the root cause: it orthogonalizes the *user* polygon
itself, before the dispatcher uses it as authority. Authentic diagonal
walls (verandas, angled bays) are preserved by an axis-majority gate —
only polygons whose edges are >=85% axis-aligned (within a wider 15
degree threshold) are snapped.

The snap reuses ``axis_snap_polygon`` with the larger threshold, so
clustering and rail-collapse logic is identical to the post-refine
stage.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from ..extrude.axis_snap import axis_snap_polygon


def axis_correct_user_polygon_px(
    coords_px: Sequence[Tuple[float, float]],
    *,
    px_to_m: float,
    angle_threshold_deg: float = 10.0,
    axis_majority: float = 0.85,
    cluster_tol_m: float = 0.025,
) -> Tuple[List[Tuple[float, float]], dict]:
    """Snap a mostly-axial user polygon onto X/Y rails.

    Parameters
    ----------
    coords_px
        Closed polygon vertices in pixel coordinates. First vertex may
        equal last (shapely-style); both are kept on output.
    px_to_m
        Metres per pixel. Used to convert ``cluster_tol_m`` into pixel
        units so the rail-merge tolerance is DPI-independent.
    angle_threshold_deg
        An edge counts as axial if its angle to the nearest axis is
        within this many degrees. 10 degrees catches typical mouse
        jitter (4-7 degrees observed in real annotations) without
        sweeping in genuine diagonals (typically >=20 degrees).
    axis_majority
        Length-weighted fraction of axial edges required to apply the
        snap. 0.85 keeps verandas/bay walls untouched.
    cluster_tol_m
        Two same-axis edges whose mean coordinate differs by less than
        this are snapped onto the same rail. 25 mm is below typical
        wall thicknesses (>=80 mm in DIN 1356) so left/right outlines
        of the same wall stay on distinct rails.

    Returns
    -------
    coords_out, info
        coords_out is the (possibly snapped) polygon. info reports
        ``corrected`` (bool), ``axis_fraction`` (0..1) and vertex counts.
    """
    info: dict = {
        "corrected": False,
        "axis_fraction": 0.0,
        "n_vertices_in": len(coords_px),
        "n_vertices_out": len(coords_px),
    }
    n = len(coords_px)
    if n < 4:
        return [tuple(p) for p in coords_px], info
    if px_to_m <= 0:
        return [tuple(p) for p in coords_px], info

    pts = list(coords_px)
    closed = (
        pts[0][0] == pts[-1][0] and pts[0][1] == pts[-1][1]
    )
    work = pts[:-1] if closed else pts
    if len(work) < 3:
        return [tuple(p) for p in coords_px], info

    angle_thresh_rad = math.radians(angle_threshold_deg)
    total_len = 0.0
    axial_len = 0.0
    for i in range(len(work)):
        x1, y1 = work[i]
        x2, y2 = work[(i + 1) % len(work)]
        dx = x2 - x1
        dy = y2 - y1
        L = math.hypot(dx, dy)
        if L == 0.0:
            continue
        total_len += L
        # Angle to nearest axis = min(angle to H, angle to V).
        # atan2(|dy|, |dx|) is in [0, pi/2]; min with pi/2 - that.
        a = math.atan2(abs(dy), abs(dx))
        angle_to_axis = min(a, math.pi / 2.0 - a)
        if angle_to_axis <= angle_thresh_rad:
            axial_len += L

    if total_len <= 0.0:
        return [tuple(p) for p in coords_px], info

    fraction = axial_len / total_len
    info["axis_fraction"] = float(fraction)

    if fraction < axis_majority:
        return [tuple(p) for p in coords_px], info

    cluster_tol_px = cluster_tol_m / px_to_m
    snapped = axis_snap_polygon(
        coords_px,
        angle_deg=angle_threshold_deg,
        cluster_tol_m=cluster_tol_px,
    )
    info["corrected"] = True
    info["n_vertices_out"] = len(snapped)
    return snapped, info


__all__ = ["axis_correct_user_polygon_px"]
