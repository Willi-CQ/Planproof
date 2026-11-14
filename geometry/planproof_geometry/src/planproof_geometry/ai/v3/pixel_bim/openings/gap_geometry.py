"""Derive the common wall axis of W1 (and W2), project label + wall
polygons onto that axis, and compute the gap interval.

Key outputs (per opening):
  - major_axis, minor_axis, origin   — common wall reference frame
  - gap_start, gap_end (along major) — opening interval
  - wall_depth_m (along minor)       — wall thickness averaged over W1+W2
  - collinearity_deg                 — angle between W1 and W2 axes;
                                        > 30 deg means "corner", caller
                                        falls back to overlap-based pick
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..extrude.mask_to_mesh import WallPolygon


# If the angle between W1 and W2 major axes exceeds this value, the two
# walls are NOT collinear — we treat the situation as a corner opening
# and place the opening on W1 only.
CORNER_ANGLE_THRESHOLD_DEG = 30.0


@dataclass
class GapGeometry:
    origin: Tuple[float, float]           # anchor point on the common axis
    major_axis: Tuple[float, float]       # unit vector, wall direction
    minor_axis: Tuple[float, float]       # unit vector, across wall (normal)
    gap_start: float                      # t along major (m), opening start (label-bounded)
    gap_end: float                        # t along major (m), opening end   (label-bounded)
    gap_length_m: float
    wall_gap_start: float                 # raw inter-wall gap (before label clipping)
    wall_gap_end: float
    wall_gap_length_m: float
    wall_depth_m: float                   # averaged thickness of W1 (+ W2)
    collinearity_deg: float               # angle between W1 and W2 majors
    is_corner: bool
    is_single_sided: bool                 # True if only W1 is present
    notes: List[str]


# ── PCA helpers ─────────────────────────────────────────────────────────

def _pca_axes(points: np.ndarray
              ) -> Tuple[Tuple[float, float], Tuple[float, float],
                         Tuple[float, float]]:
    """(origin, major_unit, minor_unit) for a 2D point cloud."""
    if points.shape[0] < 2:
        raise ValueError("PCA needs >= 2 points")
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # np.linalg.eigh returns ascending — largest eigval = last column.
    major = eigvecs[:, -1]
    minor = eigvecs[:, 0]
    return (float(mean[0]), float(mean[1])), \
           (float(major[0]), float(major[1])), \
           (float(minor[0]), float(minor[1]))


def _unsigned_angle_deg(u: Tuple[float, float],
                        v: Tuple[float, float]) -> float:
    """Unsigned angle between two unit vectors in [0, 180] degrees."""
    dot = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))
    # Treat antiparallel axes as the same line.
    ang = math.degrees(math.acos(abs(dot)))
    return ang


def _project(points: Sequence[Tuple[float, float]],
             origin: Tuple[float, float],
             axis: Tuple[float, float]) -> List[float]:
    return [
        (p[0] - origin[0]) * axis[0] + (p[1] - origin[1]) * axis[1]
        for p in points
    ]


def _minor_extent(points: Sequence[Tuple[float, float]],
                  origin: Tuple[float, float],
                  minor: Tuple[float, float]) -> Tuple[float, float]:
    ts = _project(points, origin, minor)
    return min(ts), max(ts)


# ── Public API ──────────────────────────────────────────────────────────


def compute_gap_geometry(
    w1: WallPolygon,
    w2: Optional[WallPolygon],
    label_polygon_xy_m: Sequence[Tuple[float, float]],
    *,
    corner_angle_threshold_deg: float = CORNER_ANGLE_THRESHOLD_DEG,
) -> GapGeometry:
    """Derive the common wall frame and the gap interval for one opening.

    If ``w2`` is None, the opening is single-sided (exterior). ``gap_end``
    is then the edge of the label bbox on the open side, and
    ``is_single_sided`` is True.
    """
    notes: List[str] = []

    w1_pts = np.array(list(w1.shell_xy), dtype=np.float64)
    if w1_pts.shape[0] < 2:
        raise ValueError("W1 polygon has < 2 vertices")

    # Build the common axis frame.
    if w2 is not None:
        w2_pts = np.array(list(w2.shell_xy), dtype=np.float64)
        # Pre-check collinearity: PCA on each wall independently.
        (_, major_w1, _) = _pca_axes(w1_pts)
        (_, major_w2, _) = _pca_axes(w2_pts)
        collin_deg = _unsigned_angle_deg(major_w1, major_w2)
        is_corner = collin_deg > corner_angle_threshold_deg
    else:
        collin_deg = 0.0
        is_corner = False

    if w2 is not None and not is_corner:
        combined = np.vstack([w1_pts, w2_pts])
        origin, major, minor = _pca_axes(combined)
        notes.append(f"collinear W1/W2 ({collin_deg:.1f} deg)")
    else:
        origin, major, minor = _pca_axes(w1_pts)
        if is_corner:
            notes.append(
                f"corner case: angle {collin_deg:.1f} deg > "
                f"{corner_angle_threshold_deg:.0f} deg — placing on W1 only"
            )
        else:
            notes.append("single-sided (no W2)")

    # Project the two walls onto the major axis to find their extents.
    w1_t = _project([(x, y) for x, y in w1_pts], origin, major)
    w1_lo, w1_hi = min(w1_t), max(w1_t)

    # Project the label polygon.
    lbl_t = _project(label_polygon_xy_m, origin, major)
    lbl_lo, lbl_hi = min(lbl_t), max(lbl_t)
    lbl_mid = 0.5 * (lbl_lo + lbl_hi)

    # Wall thickness from W1 minor-axis extent (and averaged with W2
    # when available).
    w1_minor_lo, w1_minor_hi = _minor_extent(
        [(x, y) for x, y in w1_pts], origin, minor
    )
    w1_depth = w1_minor_hi - w1_minor_lo

    if w2 is not None and not is_corner:
        w2_pts_list = list(w2.shell_xy)
        w2_t = _project(w2_pts_list, origin, major)
        w2_lo, w2_hi = min(w2_t), max(w2_t)
        w2_minor_lo, w2_minor_hi = _minor_extent(w2_pts_list, origin, minor)
        w2_depth = w2_minor_hi - w2_minor_lo
        wall_depth = 0.5 * (w1_depth + w2_depth)

        # Gap = between whichever wall is on the lower-t side (end) and
        # whichever is on the higher-t side (start).
        if lbl_mid >= w1_hi and lbl_mid <= w2_lo:
            gap_start, gap_end = w1_hi, w2_lo
        elif lbl_mid >= w2_hi and lbl_mid <= w1_lo:
            gap_start, gap_end = w2_hi, w1_lo
        else:
            # Label centre is not between W1 and W2 — caller will detect
            # and skip with "label_not_in_gap".
            # Pick the closer pair anyway (for diagnostic completeness).
            if lbl_mid > w1_hi:
                gap_start = w1_hi
                gap_end = max(w2_lo, w2_hi)
            else:
                gap_start = min(w1_lo, w1_hi)
                gap_end = w2_lo if w2_lo < gap_start else w2_hi
            notes.append("label_center_outside_gap — fallback heuristic")
    else:
        # Single-sided: gap_end = whichever label-bbox edge is FURTHER from w1.
        wall_depth = w1_depth
        # Which side of W1 is the label on?
        if lbl_mid > w1_hi:
            gap_start = w1_hi
            gap_end = lbl_hi
        else:
            gap_start = lbl_lo
            gap_end = w1_lo

    # Ensure gap_start <= gap_end (sign conventions).
    if gap_start > gap_end:
        gap_start, gap_end = gap_end, gap_start
    wall_gap_length = max(0.0, gap_end - gap_start)

    # The INTER-WALL gap may be much larger than the actual opening — e.g.
    # when W1 and W2 are on opposite sides of a room and the door sits in
    # a dividing wall between them. Clip to the label's extent along the
    # major axis so the opening width matches the label.
    opening_start = max(gap_start, lbl_lo)
    opening_end = min(gap_end, lbl_hi)
    if opening_start > opening_end:
        # Label does not overlap the inter-wall gap at all -> keep raw gap
        # and let the caller flag via "label_not_in_gap".
        opening_start, opening_end = gap_start, gap_end
        notes.append("label_extent_disjoint_from_wall_gap")
    opening_length = max(0.0, opening_end - opening_start)

    return GapGeometry(
        origin=origin,
        major_axis=major,
        minor_axis=minor,
        gap_start=opening_start,
        gap_end=opening_end,
        gap_length_m=opening_length,
        wall_gap_start=gap_start,
        wall_gap_end=gap_end,
        wall_gap_length_m=wall_gap_length,
        wall_depth_m=wall_depth,
        collinearity_deg=collin_deg,
        is_corner=is_corner,
        is_single_sided=(w2 is None),
        notes=notes,
    )
