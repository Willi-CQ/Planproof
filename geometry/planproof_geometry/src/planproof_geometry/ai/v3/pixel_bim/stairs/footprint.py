"""Extract a stair footprint (centroid + axes + extents) from a label polygon.

PCA over the polygon vertices yields the dominant (long) axis and the
orthogonal (short) axis. Projecting all vertices onto each axis gives
length / width.

The same pattern is used in ``openings.window_label_blocks`` for window
groups; stairs are simpler (single label, single block) so the API is
inlined here rather than reused.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass
class StairFootprint:
    """PCA-derived geometry of a stair label.

    long_axis / short_axis are unit vectors in the plan-XY frame.
    length_m is the extent along long_axis; width_m along short_axis.
    """

    centroid_m: Tuple[float, float]
    long_axis: Tuple[float, float]
    short_axis: Tuple[float, float]
    length_m: float
    width_m: float

    @property
    def long_axis_deg(self) -> float:
        """Angle of long axis in [0, 180) degrees, ccw from +X."""
        ax, ay = self.long_axis
        deg = float(np.degrees(np.arctan2(ay, ax)))
        if deg < 0:
            deg += 180.0
        if deg >= 180.0:
            deg -= 180.0
        return deg


def extract_stair_footprint(
    polygon_m: Sequence[Tuple[float, float]],
) -> StairFootprint:
    """Compute centroid, axes and extents of a polygon via PCA.

    Args:
        polygon_m: Sequence of (x, y) vertices in metres (plan frame).

    Raises:
        ValueError: if the polygon has < 3 unique points or PCA degenerates.
    """
    pts = np.asarray(polygon_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("polygon needs >= 3 (x, y) points")
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    cov = np.cov(centred, rowvar=False)
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance matrix has NaN/Inf")
    eigvals, eigvecs = np.linalg.eigh(cov)
    if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
        raise ValueError("eigendecomposition produced NaN/Inf")
    # eigh returns ascending eigenvalues; the largest is the long axis.
    long_axis = eigvecs[:, -1]
    short_axis = eigvecs[:, 0]
    # Make long_axis point in +x preferred direction (sign-convention).
    if long_axis[0] < 0 or (long_axis[0] == 0 and long_axis[1] < 0):
        long_axis = -long_axis
    # Right-handed pair: short_axis points 90 deg ccw from long_axis.
    short_axis = np.array([-long_axis[1], long_axis[0]], dtype=np.float64)
    proj_long = centred @ long_axis
    proj_short = centred @ short_axis
    length_m = float(proj_long.max() - proj_long.min())
    width_m = float(proj_short.max() - proj_short.min())
    return StairFootprint(
        centroid_m=(float(centroid[0]), float(centroid[1])),
        long_axis=(float(long_axis[0]), float(long_axis[1])),
        short_axis=(float(short_axis[0]), float(short_axis[1])),
        length_m=length_m,
        width_m=width_m,
    )
