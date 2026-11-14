"""
Wall Reconstruction: merge parallel line pairs into unified wall elements.

Floor plans draw walls as two parallel lines (wall edges). This module detects
such pairs and replaces them with a single centerline segment carrying the
measured wall thickness in its metadata.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

Point = Tuple[float, float]


def _seg_vec(a: Point, b: Point) -> Tuple[float, float]:
    return (b[0] - a[0], b[1] - a[1])


def _seg_len(a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    return math.hypot(dx, dy)


def _dot(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    return u[0] * v[0] + u[1] * v[1]


def _perp_dist_midpoint(
    ai: Point, bi: Point, aj: Point, bj: Point, li: float,
) -> float:
    """Perpendicular distance from midpoint of segment j to infinite line of segment i."""
    dxi, dyi = bi[0] - ai[0], bi[1] - ai[1]
    mid_jx = (aj[0] + bj[0]) * 0.5
    mid_jy = (aj[1] + bj[1]) * 0.5
    return abs(dxi * (mid_jy - ai[1]) - dyi * (mid_jx - ai[0])) / li


def _projection_overlap(
    ai: Point, bi: Point, aj: Point, bj: Point,
) -> float:
    """Fraction of the shorter segment that overlaps when projected onto the longer one.

    Returns a value in [0, 1].
    """
    dx, dy = bi[0] - ai[0], bi[1] - ai[1]
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return 0.0
    inv_L2 = 1.0 / L2

    # Project endpoints of j onto direction of i
    t_aj = ((aj[0] - ai[0]) * dx + (aj[1] - ai[1]) * dy) * inv_L2
    t_bj = ((bj[0] - ai[0]) * dx + (bj[1] - ai[1]) * dy) * inv_L2

    proj_lo = min(t_aj, t_bj)
    proj_hi = max(t_aj, t_bj)

    # Segment i spans [0, 1] in its own parameterisation
    overlap_lo = max(0.0, proj_lo)
    overlap_hi = min(1.0, proj_hi)
    overlap = max(0.0, overlap_hi - overlap_lo)

    shorter_len = min(math.sqrt(L2), _seg_len(aj, bj))
    if shorter_len < 1e-6:
        return 0.0

    # overlap is in parameter space of i; convert to absolute length
    overlap_m = overlap * math.sqrt(L2)
    return overlap_m / shorter_len


def _centerline(
    ai: Point, bi: Point, aj: Point, bj: Point,
) -> Tuple[Point, Point]:
    """Compute the centerline between two parallel segments.

    The centerline runs between the averaged endpoints, projected onto the
    average direction so the result is geometrically coherent.
    """
    dx_i, dy_i = bi[0] - ai[0], bi[1] - ai[1]
    dx_j, dy_j = bj[0] - aj[0], bj[1] - aj[1]

    # Ensure j points in the same direction as i
    if dx_i * dx_j + dy_i * dy_j < 0:
        aj, bj = bj, aj

    # Average direction (unit vector)
    avg_dx = (dx_i + (bj[0] - aj[0])) * 0.5
    avg_dy = (dy_i + (bj[1] - aj[1])) * 0.5
    L = math.hypot(avg_dx, avg_dy)
    if L < 1e-9:
        # Degenerate — just average the midpoints
        mx = (ai[0] + bi[0] + aj[0] + bj[0]) * 0.25
        my = (ai[1] + bi[1] + aj[1] + bj[1]) * 0.25
        return ((mx, my), (mx, my))
    ux, uy = avg_dx / L, avg_dy / L

    # Project all four endpoints onto the average direction (origin = midpoint of midpoints)
    ox = (ai[0] + bi[0] + aj[0] + bj[0]) * 0.25
    oy = (ai[1] + bi[1] + aj[1] + bj[1]) * 0.25

    projections = []
    for px, py in [ai, bi, aj, bj]:
        t = (px - ox) * ux + (py - oy) * uy
        projections.append(t)

    t_min = min(projections)
    t_max = max(projections)

    # Midpoint line (offset from average direction by half perpendicular distance)
    mid_perp_x = (ai[0] + aj[0] + bi[0] + bj[0]) * 0.25
    mid_perp_y = (ai[1] + aj[1] + bi[1] + bj[1]) * 0.25

    ca = (mid_perp_x + ux * t_min, mid_perp_y + uy * t_min)
    cb = (mid_perp_x + ux * t_max, mid_perp_y + uy * t_max)
    return (ca, cb)


# ---------------------------------------------------------------------------
# Core reconstruction algorithm
# ---------------------------------------------------------------------------

def reconstruct_walls(
    segments: List[Dict[str, Any]],
    *,
    min_thickness_m: float = 0.04,
    max_thickness_m: float = 0.50,
    angle_dot_min: float = 0.985,     # ~10 degrees
    overlap_min: float = 0.35,        # 35% projection overlap
    min_wall_length_m: float = 0.20,
) -> List[Dict[str, Any]]:
    """Merge parallel wall segment pairs into single centerline segments.

    Parameters
    ----------
    segments : list of segment dicts from semantics_dict["segments"]
    min_thickness_m : minimum perpendicular distance to consider a pair (m)
    max_thickness_m : maximum perpendicular distance (m)
    angle_dot_min : minimum |dot product| of unit direction vectors (0.985 ~ 10 deg)
    overlap_min : minimum projection overlap fraction of shorter segment
    min_wall_length_m : minimum length to participate in pairing

    Returns
    -------
    list of segment dicts — merged where possible, originals otherwise.
    """
    wall_classes = {"wall", "raw_line"}

    # Separate wall segments from non-wall (doors, windows, etc.)
    wall_indices: List[int] = []
    non_wall: List[Dict] = []

    for idx, s in enumerate(segments):
        cls = s.get("class", s.get("kind", "raw_line"))
        if cls in wall_classes:
            wall_indices.append(idx)
        else:
            non_wall.append(s)

    # Pre-compute geometry for wall segments
    class _WallInfo:
        __slots__ = ("idx", "seg", "a", "b", "dx", "dy", "length")

        def __init__(self, idx: int, seg: Dict):
            self.idx = idx
            self.seg = seg
            a = seg.get("a", [0, 0])
            b = seg.get("b", [0, 0])
            self.a = (float(a[0]), float(a[1]))
            self.b = (float(b[0]), float(b[1]))
            self.dx = self.b[0] - self.a[0]
            self.dy = self.b[1] - self.a[1]
            self.length = math.hypot(self.dx, self.dy)

    walls = [_WallInfo(i, segments[i]) for i in wall_indices]
    # Sort by length descending — longest walls pair first
    walls.sort(key=lambda w: w.length, reverse=True)

    consumed = set()  # indices of segments already merged
    merged: List[Dict] = []
    _next_id = [0]

    def _gen_id() -> str:
        _next_id[0] += 1
        return f"rw_{_next_id[0]}"

    for wi in walls:
        if wi.idx in consumed:
            continue
        if wi.length < min_wall_length_m:
            continue

        best_partner: Optional[_WallInfo] = None
        best_perp: float = 1e18

        for wj in walls:
            if wj.idx == wi.idx or wj.idx in consumed:
                continue
            if wj.length < min_wall_length_m:
                continue

            # Check near-parallel
            li, lj = wi.length, wj.length
            if li < 1e-6 or lj < 1e-6:
                continue
            dot_val = abs(
                (wi.dx * wj.dx + wi.dy * wj.dy) / (li * lj)
            )
            if dot_val < angle_dot_min:
                continue

            # Perpendicular distance (midpoint of j to line of i)
            perp = _perp_dist_midpoint(wi.a, wi.b, wj.a, wj.b, li)
            if perp < min_thickness_m or perp > max_thickness_m:
                continue

            # Projection overlap check
            overlap = _projection_overlap(wi.a, wi.b, wj.a, wj.b)
            if overlap < overlap_min:
                continue

            # Prefer closest parallel pair
            if perp < best_perp:
                best_perp = perp
                best_partner = wj

        if best_partner is not None:
            # Merge the pair
            ca, cb = _centerline(wi.a, wi.b, best_partner.a, best_partner.b)
            center_len = _seg_len(ca, cb)

            # Take the higher-confidence source
            conf_i = float(wi.seg.get("confidence", 0.5))
            conf_j = float(best_partner.seg.get("confidence", 0.5))
            source_seg = wi.seg if conf_i >= conf_j else best_partner.seg

            new_seg: Dict[str, Any] = {
                "id": _gen_id(),
                "a": list(ca),
                "b": list(cb),
                "class": "wall",
                "confidence": max(conf_i, conf_j),
                "source": source_seg.get("source", "heuristic"),
                "meta": {
                    **source_seg.get("meta", {}),
                    "measured_thickness_m": round(best_perp, 4),
                    "is_reconstructed_wall": True,
                    "merged_from": [wi.seg.get("id", ""), best_partner.seg.get("id", "")],
                },
            }
            merged.append(new_seg)
            consumed.add(wi.idx)
            consumed.add(best_partner.idx)
        # else: leave as unpaired, will be collected below

    # Collect unpaired walls (keep them unchanged)
    unpaired = [
        segments[wi.idx] for wi in walls
        if wi.idx not in consumed
    ]

    result = non_wall + merged + unpaired

    logger.info(
        "Wall reconstruction: %d input walls -> %d merged + %d unpaired = %d total "
        "(+ %d non-wall = %d segments)",
        len(walls), len(merged), len(unpaired),
        len(merged) + len(unpaired), len(non_wall), len(result),
    )

    return result


# ---------------------------------------------------------------------------
# Junction snapping
# ---------------------------------------------------------------------------

def snap_wall_junctions(
    segments: List[Dict[str, Any]],
    *,
    snap_tolerance_m: float = 0.15,
) -> List[Dict[str, Any]]:
    """Snap nearby wall endpoints to clean up T/L/X junctions.

    For each wall endpoint, finds nearby endpoints from other walls and
    averages them so they meet cleanly.
    """
    wall_classes = {"wall", "raw_line"}

    # Collect all wall endpoint references
    class _EP:
        __slots__ = ("seg_idx", "which", "x", "y")

        def __init__(self, seg_idx: int, which: str, x: float, y: float):
            self.seg_idx = seg_idx
            self.which = which  # "a" or "b"
            self.x = x
            self.y = y

    endpoints: List[_EP] = []
    for idx, s in enumerate(segments):
        cls = s.get("class", s.get("kind", "raw_line"))
        if cls not in wall_classes:
            continue
        a = s.get("a", [0, 0])
        b = s.get("b", [0, 0])
        endpoints.append(_EP(idx, "a", float(a[0]), float(a[1])))
        endpoints.append(_EP(idx, "b", float(b[0]), float(b[1])))

    if not endpoints:
        return segments

    snap2 = snap_tolerance_m * snap_tolerance_m

    # Build clusters of nearby endpoints via union-find
    parent = list(range(len(endpoints)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            if endpoints[i].seg_idx == endpoints[j].seg_idx:
                continue
            dx = endpoints[i].x - endpoints[j].x
            dy = endpoints[i].y - endpoints[j].y
            if dx * dx + dy * dy < snap2:
                _union(i, j)

    # Compute cluster averages
    from collections import defaultdict
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(endpoints)):
        clusters[_find(i)].append(i)

    snapped = 0
    result = [dict(s) for s in segments]  # shallow copy

    for members in clusters.values():
        if len(members) < 2:
            continue
        # Average position
        avg_x = sum(endpoints[i].x for i in members) / len(members)
        avg_y = sum(endpoints[i].y for i in members) / len(members)

        for i in members:
            ep = endpoints[i]
            seg = result[ep.seg_idx]
            if seg.get(ep.which) is None:
                continue
            old = seg[ep.which]
            if abs(float(old[0]) - avg_x) > 1e-9 or abs(float(old[1]) - avg_y) > 1e-9:
                seg[ep.which] = [avg_x, avg_y]
                snapped += 1

    logger.info("Junction snapping: %d endpoints snapped across %d clusters",
                snapped, sum(1 for m in clusters.values() if len(m) >= 2))

    return result
