"""Vectorise a binary wall-mask into centre-line segments with thickness.

Pipeline:

1. **Skeletonise** the mask with :func:`skimage.morphology.skeletonize` so
   walls collapse to a 1-pixel-wide centre line.
2. **Distance transform** the original mask with OpenCV — pixel value = the
   distance from that pixel to the nearest non-wall pixel (i.e. half the
   local wall thickness).
3. **Hough probabilistic line fit** on the skeleton to extract straight
   segments. Sample distance-transform values along each segment to get a
   robust per-segment thickness in pixels.
4. **Merge collinear segments** so Hough's micro-fragmentation does not
   leak duplicates into the downstream merge step.

Output: ``List[Dict]`` matching the schema consumed by
:func:`planproof_geometry.ai.v2.detection_merge.merge.merge_detections`:

    {"a": (x, y), "b": (x, y), "thickness_px": float, "confidence": float,
     "class": "wall"}

Coordinates are pixel-space; the WallSegmentationStep converts thickness to
metres downstream once ``points_to_m`` is known.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from planproof_geometry.ai.cv_compat import hough_lines_p

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


def _normalise_mask(mask: np.ndarray) -> np.ndarray:
    """Coerce arbitrary mask dtypes to a clean uint8 binary image (0/255)."""
    if mask is None:
        raise ValueError("vectorize_wall_mask: mask is None")
    if mask.ndim != 2:
        raise ValueError(f"vectorize_wall_mask: expected 2D mask, got shape {mask.shape}")
    binary = (mask > 0).astype(np.uint8) * 255
    return binary


def _segment_length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _segment_angle_deg(a: Point, b: Point) -> float:
    """Return the segment angle in degrees, normalised to [0, 180)."""
    angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    if angle < 0:
        angle += 180.0
    return angle


def _sample_thickness_along(
    distance_map: np.ndarray,
    a: Point,
    b: Point,
    *,
    n_samples: int = 8,
) -> Optional[float]:
    """Sample distance-transform along the line, return mean × 2.

    Distance-transform values measure the minimum distance from a pixel to
    the nearest *background* pixel, so on the wall centre line that value
    equals half the local wall thickness.
    """
    h, w = distance_map.shape[:2]
    samples = []
    for i in range(n_samples):
        t = (i + 0.5) / n_samples
        x = int(round(a[0] + t * (b[0] - a[0])))
        y = int(round(a[1] + t * (b[1] - a[1])))
        if 0 <= x < w and 0 <= y < h:
            samples.append(float(distance_map[y, x]))
    if not samples:
        return None
    # Median is more robust to noisy edge samples than mean
    samples.sort()
    mid = samples[len(samples) // 2]
    return 2.0 * mid


def _angular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in [0, 180)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _project_point_on_line(
    p: Point, a: Point, direction: Tuple[float, float],
) -> float:
    """Signed parameter ``t`` such that ``a + t*direction`` is the projection."""
    return (p[0] - a[0]) * direction[0] + (p[1] - a[1]) * direction[1]


def merge_collinear_segments(
    segments: List[Dict[str, Any]],
    *,
    angle_tol_deg: float = 5.0,
    perp_tol_px: float = 8.0,
    gap_tol_px: float = 30.0,
    min_segment_px: float = 0.0,
) -> List[Dict[str, Any]]:
    """Merge near-collinear segments (Hough often emits them as fragments).

    Two segments are merged when:
      * their angles differ by less than ``angle_tol_deg``
      * the perpendicular distance from the second segment's midpoint to the
        first segment's infinite line is below ``perp_tol_px``
      * along the shared direction, the colinear gap between them is below
        ``gap_tol_px`` (overlap counts as zero gap)
    """
    if not segments:
        return []

    # Pre-compute angle + endpoint geometry once
    annotated: List[Dict[str, Any]] = []
    for seg in segments:
        a = tuple(seg["a"])
        b = tuple(seg["b"])
        annotated.append({
            **seg,
            "a": a,
            "b": b,
            "_angle": _segment_angle_deg(a, b),
            "_length": _segment_length(a, b),
        })

    # Greedy merge — sort by length descending so dominant lines win
    annotated.sort(key=lambda s: s["_length"], reverse=True)
    used = [False] * len(annotated)
    merged: List[Dict[str, Any]] = []

    for i, seg in enumerate(annotated):
        if used[i]:
            continue
        used[i] = True
        a = seg["a"]
        b = seg["b"]
        angle = seg["_angle"]
        rad = math.radians(angle)
        direction = (math.cos(rad), math.sin(rad))
        thickness_samples = []
        if seg.get("thickness_px") is not None:
            thickness_samples.append((seg["_length"], float(seg["thickness_px"])))

        # Collect points to recompute the merged span
        all_points: List[Point] = [a, b]

        for j in range(i + 1, len(annotated)):
            if used[j]:
                continue
            other = annotated[j]
            if _angular_diff_deg(angle, other["_angle"]) > angle_tol_deg:
                continue

            mid = (
                (other["a"][0] + other["b"][0]) / 2.0,
                (other["a"][1] + other["b"][1]) / 2.0,
            )
            # Perpendicular distance from `mid` to infinite line through `a` along direction
            normal = (-direction[1], direction[0])
            perp = abs((mid[0] - a[0]) * normal[0] + (mid[1] - a[1]) * normal[1])
            if perp > perp_tol_px:
                continue

            # Along-line gap: project both segments' endpoints onto the line
            seg_extent = (
                min(_project_point_on_line(a, a, direction),
                    _project_point_on_line(b, a, direction)),
                max(_project_point_on_line(a, a, direction),
                    _project_point_on_line(b, a, direction)),
            )
            other_extent = (
                min(_project_point_on_line(other["a"], a, direction),
                    _project_point_on_line(other["b"], a, direction)),
                max(_project_point_on_line(other["a"], a, direction),
                    _project_point_on_line(other["b"], a, direction)),
            )
            gap = max(0.0, max(seg_extent[0], other_extent[0])
                          - min(seg_extent[1], other_extent[1]))
            if gap > gap_tol_px:
                continue

            used[j] = True
            all_points.extend([other["a"], other["b"]])
            if other.get("thickness_px") is not None:
                thickness_samples.append((other["_length"], float(other["thickness_px"])))

        # Compute merged endpoints by projecting all points and taking the extent
        projected = [(p, _project_point_on_line(p, a, direction)) for p in all_points]
        projected.sort(key=lambda x: x[1])
        new_a = projected[0][0]
        new_b = projected[-1][0]

        if thickness_samples:
            total = sum(w for w, _ in thickness_samples)
            avg_thickness = sum(w * t for w, t in thickness_samples) / total if total > 0 else None
        else:
            avg_thickness = seg.get("thickness_px")

        merged.append({
            "a": (float(new_a[0]), float(new_a[1])),
            "b": (float(new_b[0]), float(new_b[1])),
            "thickness_px": float(avg_thickness) if avg_thickness is not None else None,
            "confidence": float(seg.get("confidence", 0.85)),
            "class": seg.get("class", "wall"),
        })

    if min_segment_px > 0:
        merged = [s for s in merged if _segment_length(s["a"], s["b"]) >= min_segment_px]

    return merged


def vectorize_wall_mask(
    mask: np.ndarray,
    *,
    min_segment_px: int = 20,
    hough_threshold: int = 30,
    hough_max_gap: int = 10,
    angle_tol_deg: float = 5.0,
    gap_tol_px: float = 30.0,
    perp_tol_px: float = 8.0,
    min_merge_segment_px: float = 0.0,
    confidence: float = 0.85,
) -> List[Dict[str, Any]]:
    """Convert a binary wall-mask into centre-line segments with thickness.

    Args:
        mask: 2D numpy array, non-zero where walls are.
        min_segment_px: Minimum segment length in pixels (passed to Hough).
        hough_threshold: Hough accumulator threshold (votes for a line).
        hough_max_gap: Max gap (px) between segments to consider them one
            line in Hough.
        angle_tol_deg: Angular tolerance for the post-merge step.
        gap_tol_px: Along-line gap tolerance for the post-merge step.
        confidence: Per-segment confidence written into the output.

    Returns:
        ``List[Dict]`` ready for :func:`merge_detections`.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for vectorize_wall_mask") from exc
    try:
        from skimage.morphology import skeletonize
    except ImportError as exc:
        raise RuntimeError(
            "scikit-image is required for vectorize_wall_mask. "
            "Install with: pip install planproof-core[wallseg]"
        ) from exc

    binary = _normalise_mask(mask)
    if not binary.any():
        return []

    # Skeleton (boolean) → uint8 0/255 for OpenCV
    skeleton = skeletonize(binary > 0)
    skeleton_u8 = (skeleton.astype(np.uint8)) * 255

    # Distance transform on the original mask (not the skeleton): values in
    # skeleton pixels equal half the local wall thickness in px
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # (N, 4) in OpenCV 4 wie 5 - siehe planproof_geometry.ai.cv_compat.
    lines = hough_lines_p(
        skeleton_u8,
        rho=1,
        theta=math.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_segment_px,
        maxLineGap=hough_max_gap,
    )

    raw_segments: List[Dict[str, Any]] = []
    for x1, y1, x2, y2 in lines:
        a = (float(x1), float(y1))
        b = (float(x2), float(y2))
        if _segment_length(a, b) < min_segment_px:
            continue
        thickness_px = _sample_thickness_along(distance, a, b)
        raw_segments.append({
            "a": a,
            "b": b,
            "thickness_px": thickness_px,
            "confidence": confidence,
            "class": "wall",
        })

    if not raw_segments:
        logger.info("Wall vectorization: HoughLinesP found no segments")
        return []

    merged = merge_collinear_segments(
        raw_segments,
        angle_tol_deg=angle_tol_deg,
        gap_tol_px=gap_tol_px,
        perp_tol_px=perp_tol_px,
        min_segment_px=min_merge_segment_px,
    )

    logger.info(
        "Wall vectorization: %d Hough segments → %d after collinear merge",
        len(raw_segments), len(merged),
    )
    return merged


def attach_thickness_in_metres(
    segments: List[Dict[str, Any]],
    points_to_m: float,
) -> List[Dict[str, Any]]:
    """Convenience: enrich pixel-space segments with ``thickness_m``.

    The detection-merge step expects ``thickness_m`` (or falls back to
    ``thickness_px * points_to_m`` via meta). Doing the conversion here
    keeps :class:`WallSegmentationStep` thin.
    """
    if points_to_m <= 0:
        return segments
    out = []
    for seg in segments:
        new = dict(seg)
        if seg.get("thickness_px") is not None:
            new["thickness_m"] = float(seg["thickness_px"]) * float(points_to_m)
        out.append(new)
    return out


__all__ = [
    "vectorize_wall_mask",
    "merge_collinear_segments",
    "attach_thickness_in_metres",
]
