"""4-directional wall scanner for in-wall openings (sliding doors).

Sprint Sliding-Door-Rewrite Tag-1: replace the gap+jamb-finding cascade
for sliding doors with a deterministic 4-directional scan from the
door bbox centre. The scan returns a ``WallContext`` that captures:

  * which wall polygon contains the bbox centre
  * the wall's local long/short axes and measured thickness
  * how far continuous wall material extends in each of 4 directions
    (±long_u, ±short_u) from the bbox centre

Downstream the WallContext is consumed by ``build_sliding_door_polygon``
(Tag 2) to construct the door rectangle on the wall midplane, with
pocket_sign falling out of ``extent_pos vs extent_neg`` directly — no
separate ``classify_sliding_mount`` pass needed.

The probe-box pattern is lifted from
``pocket_surface_classifier._continuous_wall_length`` so the geometry
behaviour is identical to the existing pocket-side measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .mount_classifier import (
    MountClassification,
    classify_sliding_mount_voting,
)


@dataclass(frozen=True)
class WallContext:
    """Describes the wall that encloses a sliding-door bbox.

    All extents are measured in metres from ``wall_centerline_xy`` along
    the corresponding unit vector. They are the contiguous wall length
    before the first gap > ``gap_tolerance_m``, capped at
    ``max_extent_m``.

    ``confidence`` is a 0..1 signal: 1.0 means the bbox centre lies
    cleanly inside one wall polygon AND the long-axis sweeps both
    returned non-zero extents (i.e. the door is in the middle of a
    wall, not at a corner). Below 0.5 the caller should fall back to
    the legacy place_door cascade.

    ``mount_classification`` is populated when scan_wall_4dir is called
    with a bbox + classifier inputs (Tag-2). It carries the in_wall vs
    beside_wall vote that drives Tag-3 routing. ``None`` when classifier
    inputs are not provided — caller falls back to the existing
    confidence-based gate.
    """
    wall_polygon_idx: int
    wall_axis_long_u: np.ndarray
    wall_axis_short_u: np.ndarray
    wall_depth_m: float
    wall_centerline_xy: np.ndarray
    extent_long_pos_m: float
    extent_long_neg_m: float
    extent_short_pos_m: float
    extent_short_neg_m: float
    confidence: float
    notes: List[str] = field(default_factory=list)
    mount_classification: Optional[MountClassification] = None

    @property
    def found(self) -> bool:
        return self.wall_polygon_idx >= 0 and self.wall_depth_m > 0.0


def _local_wall_basis(
    shell_xy: Sequence[Tuple[float, float]],
    bbox_center: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Local wall axes at ``bbox_center``, derived from a length-weighted
    score over polygon edges.

    Three robustness behaviours rolled into one heuristic:

      * **Mid-wall door**: nearest edge IS one of the long parallel
        wall faces — pick its tangent directly.
      * **End-of-wall door**: nearest edge is the short cap, but a
        long parallel wall face is a few cm further away — score
        edges by ``length / (distance + wall_depth_floor)`` so the
        long edge wins despite being slightly more distant.
      * **Diagonal / rotated wall**: the score does not depend on
        axis alignment, so it works for any rotation.

    Returns ``(long_u, short_u)`` — long_u along the chosen edge
    direction, short_u its 90° CCW rotation.
    """
    pts = np.asarray(shell_xy, dtype=np.float64)
    if pts.shape[0] < 2:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    closed = np.vstack([pts, pts[0:1]])
    # Floor to keep the score finite when the bbox sits ON an edge.
    dist_floor = 0.05  # 5 cm — about a third of a typical wall depth
    best_score = -1.0
    best_tangent = np.array([1.0, 0.0])
    for i in range(len(closed) - 1):
        a = closed[i]
        b = closed[i + 1]
        ab = b - a
        ab_len2 = float(ab @ ab)
        if ab_len2 < 1e-12:
            continue
        ab_len = float(np.sqrt(ab_len2))
        t = float((bbox_center - a) @ ab) / ab_len2
        t_clamped = max(0.0, min(1.0, t))
        foot = a + t_clamped * ab
        d = float(np.linalg.norm(bbox_center - foot))
        # Score: longer edges close to the bbox win. The dist_floor
        # prevents the nearest-but-shortest edge from dominating just
        # because its distance is near-zero.
        score = ab_len / (d + dist_floor)
        if score > best_score:
            best_score = score
            best_tangent = ab / ab_len
    long_u = best_tangent
    short_u = np.array([-long_u[1], long_u[0]])
    return long_u, short_u


def _ray_cast_wall_depth(
    bbox_center: np.ndarray,
    long_u: np.ndarray,
    short_u: np.ndarray,
    wall_union,
    *,
    max_search_m: float = 1.0,
) -> Tuple[float, float]:
    """Cast a perpendicular line through ``bbox_center`` and measure how
    much wall material it crosses → wall thickness + signed offset to
    the wall midplane.

    Returns ``(wall_depth_m, short_offset_m)``. If the ray misses,
    returns ``(0.0, 0.0)``.

    The line spans ±max_search_m along ``short_u`` (perpendicular to
    the wall). We intersect with ``wall_union``; the longest single
    LineString component is the wall cross-section.
    """
    from shapely.geometry import LineString

    if wall_union is None or wall_union.is_empty:
        return 0.0, 0.0
    a = bbox_center - max_search_m * short_u
    b = bbox_center + max_search_m * short_u
    line = LineString([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))])
    try:
        inter = line.intersection(wall_union)
    except Exception:
        return 0.0, 0.0
    if inter.is_empty:
        return 0.0, 0.0
    geoms = (
        [inter] if inter.geom_type == "LineString"
        else list(inter.geoms) if hasattr(inter, "geoms")
        else []
    )
    best_seg = None
    best_len = 0.0
    for g in geoms:
        if g.geom_type != "LineString":
            continue
        if g.length > best_len:
            best_len = float(g.length)
            best_seg = g
    if best_seg is None or best_len < 0.04 or best_len > 0.50:
        return 0.0, 0.0
    mid = best_seg.interpolate(0.5, normalized=True)
    delta = np.array([mid.x, mid.y]) - bbox_center
    short_offset = float(delta @ short_u)
    return best_len, short_offset


def _continuous_extent(
    start_xy: np.ndarray,
    direction: np.ndarray,
    cross: np.ndarray,
    wall_union,
    wall_depth_m: float,
    *,
    max_extent_m: float,
    gap_tolerance_m: float,
    sample_step_m: float,
) -> float:
    """Probe-box sweep along ``direction`` from ``start_xy``.

    Mirrors the algorithm in ``pocket_surface_classifier._continuous_wall_length``
    but takes a precomputed ``wall_union`` (shapely geometry) so multiple
    sweeps in different directions don't rebuild the union each time.

    Returns the contiguous extent in metres before the first gap longer
    than ``gap_tolerance_m``, capped at ``max_extent_m``.
    """
    from shapely.geometry import Polygon as SPoly

    if wall_union is None or wall_union.is_empty:
        return 0.0
    half_d = max(0.05, wall_depth_m) * 0.5
    total_continuous = 0.0
    gap_running = 0.0
    n_samples = max(1, int(round(max_extent_m / sample_step_m)))
    for i in range(n_samples):
        d_center = (i + 0.5) * sample_step_m
        sample_center = start_xy + d_center * direction
        half_step = sample_step_m * 0.5
        corners = [
            sample_center + half_step * direction + half_d * cross,
            sample_center + half_step * direction - half_d * cross,
            sample_center - half_step * direction - half_d * cross,
            sample_center - half_step * direction + half_d * cross,
        ]
        try:
            box = SPoly(corners).buffer(0)
        except Exception:
            continue
        try:
            has_wall = box.intersects(wall_union) and (
                box.intersection(wall_union).area > 0.1 * box.area
            )
        except Exception:
            has_wall = False
        if has_wall:
            if gap_running > gap_tolerance_m:
                break
            total_continuous += sample_step_m + gap_running
            gap_running = 0.0
        else:
            gap_running += sample_step_m
            if gap_running > gap_tolerance_m:
                break
    return min(total_continuous, max_extent_m)


def _find_enclosing_wall(
    bbox_center_xy: Tuple[float, float],
    wall_polygons,
    *,
    nearest_max_m: float = 0.30,
):
    """Return ``(idx, wall_polygon)`` for the wall containing the bbox centre.

    If no wall contains the centre, returns the closest wall whose shell
    is within ``nearest_max_m``. Returns ``(-1, None)`` if nothing is
    close enough.
    """
    from shapely.geometry import Point, Polygon as SPoly

    pt = Point(bbox_center_xy)
    best_idx = -1
    best_dist = float("inf")
    for i, wp in enumerate(wall_polygons):
        shell = getattr(wp, "shell_xy", None)
        if shell is None or len(shell) < 3:
            continue
        try:
            poly = SPoly(shell, holes=getattr(wp, "holes_xy", None)).buffer(0)
        except Exception:
            continue
        if poly.is_empty:
            continue
        if poly.contains(pt):
            return i, wp
        d = poly.distance(pt)
        if d < best_dist:
            best_dist = d
            best_idx = i
    if best_idx >= 0 and best_dist <= nearest_max_m:
        return best_idx, wall_polygons[best_idx]
    return -1, None


def scan_wall_4dir(
    bbox_center_xy: Tuple[float, float],
    wall_polygons,
    *,
    max_extent_m: float = 5.0,
    sample_step_m: float = 0.020,
    gap_tolerance_m: float = 0.030,
    nearest_max_m: float = 0.30,
    bbox_m: Optional[Tuple[float, float, float, float]] = None,
    image_gray=None,
    px_to_m: Optional[float] = None,
    panel_pixel_endpoints=None,
) -> WallContext:
    """4-directional wall scan for an in-wall opening (sliding door).

    1. Find the wall polygon that contains ``bbox_center_xy`` (or the
       closest wall within ``nearest_max_m``).
    2. PCA on its shell gives ``long_u`` (wall direction), ``short_u``
       (across-wall), and a polygon-derived ``wall_depth_m``.
    3. Project ``bbox_center`` onto the wall midplane to get
       ``wall_centerline_xy``.
    4. Four directional sweeps (±long_u, ±short_u) using the shared
       probe-box pattern; each returns the contiguous wall extent
       before the first gap > ``gap_tolerance_m``.
    5. Confidence: 0.5 base for "wall found", +0.25 each if both
       long-axis extents are non-zero (door sits inside the wall, not
       at a corner) and if both short-axis extents are <= 1.5 ×
       wall_depth (i.e. the centre is inside, not flanking).

    Returns a ``WallContext``; ``confidence < 0.5`` means the caller
    should fall back to the legacy placement cascade.
    """
    from shapely.geometry import Polygon as SPoly
    from shapely.ops import unary_union

    notes: List[str] = []
    idx, wp = _find_enclosing_wall(
        bbox_center_xy, wall_polygons, nearest_max_m=nearest_max_m,
    )
    if idx < 0 or wp is None:
        notes.append(
            f"no wall polygon contains bbox_center={bbox_center_xy} "
            f"and none within {nearest_max_m * 1000:.0f}mm",
        )
        return WallContext(
            wall_polygon_idx=-1,
            wall_axis_long_u=np.array([1.0, 0.0]),
            wall_axis_short_u=np.array([0.0, 1.0]),
            wall_depth_m=0.0,
            wall_centerline_xy=np.asarray(bbox_center_xy, dtype=np.float64),
            extent_long_pos_m=0.0,
            extent_long_neg_m=0.0,
            extent_short_pos_m=0.0,
            extent_short_neg_m=0.0,
            confidence=0.0,
            notes=notes,
        )

    bc = np.asarray(bbox_center_xy, dtype=np.float64)

    # Local axes from the nearest polygon edge tangent — robust for
    # L-shaped or otherwise irregular walls where full-shell PCA would
    # pick a diagonal axis that has nothing to do with the local wall
    # direction at the bbox.
    long_u, short_u = _local_wall_basis(wp.shell_xy, bc)

    wall_shapely = []
    for w in wall_polygons:
        shell = getattr(w, "shell_xy", None)
        if shell is None or len(shell) < 3:
            continue
        try:
            sp = SPoly(shell, holes=getattr(w, "holes_xy", None)).buffer(0)
            if not sp.is_empty and sp.is_valid:
                wall_shapely.append(sp)
        except Exception:
            continue
    if wall_shapely:
        try:
            wall_union = unary_union(wall_shapely)
        except Exception:
            wall_union = wall_shapely[0]
    else:
        wall_union = None

    # Wall depth + centerline offset from a single perpendicular ray
    # cast through the bbox centre. This is the LOCAL wall thickness,
    # not the polygon's overall extent — works for walls of any shape.
    ray_depth, ray_offset = _ray_cast_wall_depth(
        bc, long_u, short_u, wall_union,
    )
    ray_cast_failed = ray_depth <= 0.04
    if not ray_cast_failed:
        wall_depth_m = ray_depth
        centerline = bc + ray_offset * short_u
    else:
        # Ray missed: usually means the bbox sits in a wall-annotation
        # gap (user drew walls that don't quite reach the door). The
        # 4-dir scan cannot place the door reliably here — fall back
        # to a 12 cm guess but DROP confidence so the caller falls
        # back to the legacy place_door cascade.
        wall_depth_m = 0.12
        centerline = bc
        notes.append("ray-cast wall depth missed; using 12 cm fallback")

    sweep_kwargs = dict(
        wall_union=wall_union,
        wall_depth_m=wall_depth_m,
        max_extent_m=max_extent_m,
        gap_tolerance_m=gap_tolerance_m,
        sample_step_m=sample_step_m,
    )
    extent_long_pos = _continuous_extent(centerline, long_u, short_u, **sweep_kwargs)
    extent_long_neg = _continuous_extent(centerline, -long_u, short_u, **sweep_kwargs)
    extent_short_pos = _continuous_extent(centerline, short_u, long_u, **sweep_kwargs)
    extent_short_neg = _continuous_extent(centerline, -short_u, long_u, **sweep_kwargs)

    confidence = 0.5  # wall found
    if extent_long_pos > 0.0 and extent_long_neg > 0.0:
        confidence += 0.25
    else:
        notes.append(
            f"long-axis extent asymmetric: +{extent_long_pos*1000:.0f}mm "
            f"/ -{extent_long_neg*1000:.0f}mm (door at wall end?)",
        )
    if (extent_short_pos <= 1.5 * wall_depth_m
            and extent_short_neg <= 1.5 * wall_depth_m):
        confidence += 0.25
    else:
        notes.append(
            f"short-axis extent leaks past wall ({extent_short_pos*1000:.0f}mm "
            f"/ {extent_short_neg*1000:.0f}mm vs wall_depth {wall_depth_m*1000:.0f}mm)",
        )
    # Ray-cast miss is a hard penalty: without a real wall thickness
    # we can't place the door reliably. Cap below the 0.5 caller-side
    # threshold so the legacy cascade takes over.
    if ray_cast_failed:
        confidence = min(confidence, 0.4)

    ctx = WallContext(
        wall_polygon_idx=idx,
        wall_axis_long_u=long_u,
        wall_axis_short_u=short_u,
        wall_depth_m=float(wall_depth_m),
        wall_centerline_xy=centerline,
        extent_long_pos_m=float(extent_long_pos),
        extent_long_neg_m=float(extent_long_neg),
        extent_short_pos_m=float(extent_short_pos),
        extent_short_neg_m=float(extent_short_neg),
        confidence=float(min(1.0, confidence)),
        notes=notes,
    )

    # Tag-2: classify in_wall vs beside_wall when the caller passed a
    # bbox. The classifier itself short-circuits on ``not ctx.found`` so
    # we don't need to gate it here — but we do need a bbox to classify.
    if bbox_m is not None:
        mc = classify_sliding_mount_voting(
            bbox_m, ctx,
            wall_polygons=wall_polygons,
            image_gray=image_gray,
            px_to_m=px_to_m,
            panel_pixel_endpoints=panel_pixel_endpoints,
        )
        ctx = replace(ctx, mount_classification=mc)
    return ctx


__all__ = ["WallContext", "scan_wall_4dir"]
