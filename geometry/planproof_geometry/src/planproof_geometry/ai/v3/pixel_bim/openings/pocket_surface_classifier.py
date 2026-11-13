"""Classify a sliding door as pocket vs surface (barn-door) mount and
pick the correct pocket side.

The dominant signal is **wall-sufficiency**: a pocket sliding door
needs at least ``panel_width × SLIDE_OPEN_FRACTION`` of continuous
wall material on the pocket side to hide the panel when open. If
neither side has enough wall, the door cannot be a pocket door —
it must be a surface (barn-door) mount.

This replaces the legacy ``compute_mount_type`` + ``_pick_pocket_side``
which used bbox geometry alone and got the side wrong whenever the
bbox extension didn't match the actual wall continuation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


def _door_basis(
    polygon_xy: Sequence[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (center, long_u, short_u, long_extent_m) for a door rectangle."""
    pts = np.asarray(polygon_xy, dtype=np.float64)
    if pts.shape[0] < 4:
        raise ValueError("door polygon needs >= 4 vertices")
    edges = pts[1:] - pts[:-1]
    edge_lengths = np.linalg.norm(edges, axis=1)
    longest_idx = int(np.argmax(edge_lengths))
    long_u = edges[longest_idx] / max(edge_lengths[longest_idx], 1e-9)
    short_u = np.array([-long_u[1], long_u[0]])
    center = pts[:4].mean(axis=0)
    return center, long_u, short_u, float(edge_lengths[longest_idx])


def _continuous_wall_length(
    edge_xy: np.ndarray,
    sign: int,
    long_u: np.ndarray,
    short_u: np.ndarray,
    wall_polygons_xy: Sequence[Sequence[Tuple[float, float]]],
    wall_depth_m: float,
    *,
    max_extent_m: float = 5.0,
    gap_tolerance_m: float = 0.030,
    sample_step_m: float = 0.020,
) -> float:
    """Measure the length of CONTINUOUS wall material starting at
    ``edge_xy`` and extending in the ``sign * long_u`` direction.

    Samples along the ray every ``sample_step_m``; at each sample point
    a small box (``wall_depth × sample_step``) is intersected with the
    union of wall polygons. The first sample where no wall is found
    starts a "gap"; gaps shorter than ``gap_tolerance_m`` are tolerated
    (annotation ends often have small slivers), longer gaps terminate
    the count.

    Returns the contiguous wall length found before the first
    gap > ``gap_tolerance_m``, capped at ``max_extent_m``.
    """
    try:
        from shapely.geometry import Polygon as SPoly  # noqa: PLC0415
        from shapely.ops import unary_union  # noqa: PLC0415
    except ImportError:
        return 0.0
    wall_shapely = []
    for shell in wall_polygons_xy:
        if len(shell) < 3:
            continue
        try:
            wp = SPoly(shell).buffer(0)
            if not wp.is_empty and wp.is_valid:
                wall_shapely.append(wp)
        except Exception:
            continue
    if not wall_shapely:
        return 0.0
    try:
        wall_union = unary_union(wall_shapely)
    except Exception:
        wall_union = None

    half_d = max(0.05, wall_depth_m) * 0.5
    total_continuous = 0.0
    gap_running = 0.0
    n_samples = max(1, int(round(max_extent_m / sample_step_m)))
    for i in range(n_samples):
        # Sample box centre at distance i*step + step/2 past edge
        d_center = (i + 0.5) * sample_step_m
        sample_center = edge_xy + sign * d_center * long_u
        half_step = sample_step_m * 0.5
        corners = [
            sample_center + half_step * long_u + half_d * short_u,
            sample_center + half_step * long_u - half_d * short_u,
            sample_center - half_step * long_u - half_d * short_u,
            sample_center - half_step * long_u + half_d * short_u,
        ]
        try:
            box = SPoly(corners).buffer(0)
        except Exception:
            continue
        has_wall = False
        if wall_union is not None and not wall_union.is_empty:
            try:
                has_wall = box.intersects(wall_union) and (
                    box.intersection(wall_union).area > 0.1 * box.area
                )
            except Exception:
                has_wall = False
        if has_wall:
            if gap_running > 0.0 and gap_running > gap_tolerance_m:
                # Gap was wider than tolerance — stop counting
                break
            total_continuous += sample_step_m + gap_running
            gap_running = 0.0
        else:
            gap_running += sample_step_m
            if gap_running > gap_tolerance_m:
                break
    return min(total_continuous, max_extent_m)


@dataclass(frozen=True)
class SlidingMountResult:
    mount_type: str          # "pocket" | "surface"
    pocket_sign: int         # +1 or -1 (direction the panel slides toward)
    confidence: float        # 0..1 (how strongly the wall data supports the call)
    wall_len_pos: float      # continuous wall length on +long_u side (m)
    wall_len_neg: float      # continuous wall length on -long_u side (m)
    panel_slide_m: float     # required slide distance (m)


def classify_sliding_mount(
    door_polygon_xy: Sequence[Tuple[float, float]],
    bbox_m: Tuple[float, float, float, float],
    wall_depth_m: float,
    wall_polygons_xy: Optional[Sequence[Sequence[Tuple[float, float]]]] = None,
    *,
    panel_width_m: Optional[float] = None,
    slide_open_fraction: float = 0.7,
    gap_tolerance_m: float = 0.030,
) -> SlidingMountResult:
    """Classify a sliding door's mount type and pick its pocket side.

    Algorithm:
      1. Measure continuous wall length on ±long_u sides of the door.
      2. ``panel_slide = panel_width_m × slide_open_fraction`` (default
         70 % — typical sliding panel travel).
      3. If max(wall_len_+, wall_len_-) >= panel_slide:
            → pocket; pocket_sign = side with the longer wall.
         else:
            → surface (barn-door); pocket_sign = bbox-extension side
              (where the bbox extends past the door rectangle, this is
              the visible-panel side).
      4. confidence = how decisively the longer wall exceeds panel_slide.

    Parameters
    ----------
    door_polygon_xy
        Door rectangle (4 corners) in plan metres.
    bbox_m
        Annotation bbox (xmin, ymin, xmax, ymax) in plan metres. Used
        for surface-mount side-fallback.
    wall_depth_m
        Local wall thickness at the door (metres). Used for the probe
        box width when measuring wall-sufficiency.
    wall_polygons_xy
        User-drawn wall polygons (raw annotations) in plan metres.
        Pass these (not the post-refine BIM walls) — they represent
        what the annotator drew, which is what determines the mount.
    panel_width_m
        Final placed door width. If ``None``, falls back to the bbox
        long-axis extent.
    slide_open_fraction
        How far the panel travels relative to its width. 0.7 matches
        the existing render constant ``SLIDE_OPEN_FRACTION``.
    gap_tolerance_m
        Wall annotations sometimes have ~1-3 cm gaps at junctions; gaps
        ≤ this are treated as continuous, gaps > this terminate the
        wall-length count.
    """
    center, long_u, short_u, door_long = _door_basis(door_polygon_xy)
    if panel_width_m is None:
        panel_width_m = door_long
    panel_slide_m = panel_width_m * slide_open_fraction

    edge_pos = center + (door_long * 0.5) * long_u
    edge_neg = center - (door_long * 0.5) * long_u

    if wall_polygons_xy is None:
        wall_polygons_xy = []

    wall_len_pos = _continuous_wall_length(
        edge_pos, +1, long_u, short_u,
        wall_polygons_xy, wall_depth_m,
        gap_tolerance_m=gap_tolerance_m,
    )
    wall_len_neg = _continuous_wall_length(
        edge_neg, -1, long_u, short_u,
        wall_polygons_xy, wall_depth_m,
        gap_tolerance_m=gap_tolerance_m,
    )

    max_side = max(wall_len_pos, wall_len_neg)
    if max_side >= panel_slide_m:
        mount = "pocket"
        pocket_sign = +1 if wall_len_pos >= wall_len_neg else -1
        # Confidence: how much margin past panel_slide
        margin = (max_side - panel_slide_m) / max(0.05, panel_slide_m)
        confidence = float(min(1.0, 0.5 + 0.5 * margin))
    else:
        mount = "surface"
        # For surface, the panel hangs on the side where the bbox extends
        # past the door rectangle. Same logic as legacy _pick_pocket_side.
        bbox_center = np.array([
            (bbox_m[0] + bbox_m[2]) * 0.5,
            (bbox_m[1] + bbox_m[3]) * 0.5,
        ])
        delta = bbox_center - center
        proj = float(delta @ long_u)
        pocket_sign = +1 if proj >= 0.0 else -1
        # Confidence: how far short of needed wall length we are
        deficit = (panel_slide_m - max_side) / max(0.05, panel_slide_m)
        confidence = float(min(1.0, 0.5 + 0.5 * deficit))

    return SlidingMountResult(
        mount_type=mount,
        pocket_sign=pocket_sign,
        confidence=confidence,
        wall_len_pos=wall_len_pos,
        wall_len_neg=wall_len_neg,
        panel_slide_m=panel_slide_m,
    )


__all__ = ["classify_sliding_mount", "SlidingMountResult"]
