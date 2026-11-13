"""Find the two wall polygons that flank an opening label.

Input: the list of WallPolygon objects from Phase 2 (in plan-metre
coordinates) and one opening label (polygon in plan-metre coordinates).

Output: FlankingWalls(wall1_index, wall2_index) or a failure reason.

Rules:
  - nearest wall is the one closest to the label centre (point-to-polygon
    distance). Max 0.5 m; farther -> isolated, skip.
  - second-nearest wall must be within 2 m. Farther -> exterior opening,
    W2 = None (single-sided).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..extrude.mask_to_mesh import WallPolygon


MAX_DIST_W1_M = 1.50   # label centre must be within 1.5 m of the primary wall
                       # (label bbox usually sits over the gap centre; this
                       #  allows up to ~1.5 m opening width, DIN-plausible).
MAX_DIST_W2_M = 3.00   # second wall within 3 m counts as "flanking";
                       # else exterior / single-sided.


@dataclass
class FlankingWalls:
    wall1_index: int
    wall2_index: Optional[int]
    w1_distance_m: float
    w2_distance_m: Optional[float]
    note: str = ""


def _polygon_centroid(xy: Sequence[Tuple[float, float]]
                      ) -> Tuple[float, float]:
    if not xy:
        return (0.0, 0.0)
    xs = [p[0] for p in xy]
    ys = [p[1] for p in xy]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _point_polygon_distance_m(
    point: Tuple[float, float],
    shell: Sequence[Tuple[float, float]],
) -> float:
    """Distance from a point to a polygon (shell only). Returns 0 if
    the point is inside. Uses shapely for correctness."""
    from shapely.geometry import Point, Polygon
    if len(shell) < 3:
        return float("inf")
    try:
        return Polygon(shell).distance(Point(point))
    except Exception:
        return float("inf")


def find_flanking_walls(
    label_polygon_xy_m: Sequence[Tuple[float, float]],
    wall_polygons: List[WallPolygon],
    *,
    max_dist_w1_m: float = MAX_DIST_W1_M,
    max_dist_w2_m: float = MAX_DIST_W2_M,
) -> Optional[FlankingWalls]:
    """Return the two wall polygons that flank the given opening label.

    Returns ``None`` if no wall is within ``max_dist_w1_m`` of the label
    centre (isolated label — Phase 3 will skip it).
    """
    if not wall_polygons:
        return None
    label_center = _polygon_centroid(label_polygon_xy_m)

    # Compute distance from label centre to every wall polygon.
    distances: List[Tuple[int, float]] = []
    for i, wp in enumerate(wall_polygons):
        d = _point_polygon_distance_m(label_center, wp.shell_xy)
        distances.append((i, d))
    distances.sort(key=lambda t: t[1])

    # W1 = nearest.
    w1_idx, w1_d = distances[0]
    if w1_d > max_dist_w1_m:
        return None  # isolated

    if len(distances) == 1:
        return FlankingWalls(
            wall1_index=w1_idx, wall2_index=None,
            w1_distance_m=w1_d, w2_distance_m=None,
            note="single wall in scene — exterior opening",
        )

    # W2 = second-nearest, only if within max_dist_w2_m.
    w2_idx, w2_d = distances[1]
    if w2_d > max_dist_w2_m:
        return FlankingWalls(
            wall1_index=w1_idx, wall2_index=None,
            w1_distance_m=w1_d, w2_distance_m=None,
            note=f"only W1 within {max_dist_w2_m:.1f}m — exterior opening",
        )

    return FlankingWalls(
        wall1_index=w1_idx, wall2_index=w2_idx,
        w1_distance_m=w1_d, w2_distance_m=w2_d,
    )
