"""
Heuristic door and window detection for architectural floor plans.

Detects openings without YOLO by exploiting visual conventions:
  - Doors: quarter-circle arcs (Tuerbogen) drawn as door swing indicators
  - Windows: clusters of 3-4 short parallel lines within a wall segment

These heuristics are especially useful for plans where YOLO fails to detect
small opening elements (e.g. raster-scanned Altbau drawings at low resolution).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HeuristicDetection:
    """A detection produced by geometric heuristics (not ML)."""
    class_name: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float
    source: str  # "heuristic_arc", "heuristic_parallel"
    meta: Dict[str, Any] = field(default_factory=dict)


def detect_door_arcs(
    gray_image: np.ndarray,
    px_per_m: float,
    min_radius_m: float = 0.35,
    max_radius_m: float = 0.65,
    canny_low: int = 50,
    canny_high: int = 150,
) -> List[HeuristicDetection]:
    """Detect quarter-circle door arcs via Hough circle detection.

    Architectural plans show door swing as a quarter-circle arc with radius
    equal to the door leaf width (typically 0.4-0.6 m).  This function finds
    circular arcs and filters for quarter-arcs (edge density in only ~1 quadrant).

    Args:
        gray_image: Grayscale image (uint8).
        px_per_m: Pixels per metre (for radius filtering).
        min_radius_m: Minimum door leaf radius in metres.
        max_radius_m: Maximum door leaf radius in metres.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available, skipping arc detection")
        return []

    if len(gray_image.shape) == 3:
        gray = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_image

    h, w = gray.shape[:2]

    # Radius bounds in pixels
    min_r_px = max(8, int(min_radius_m * px_per_m))
    max_r_px = min(min(h, w) // 4, int(max_radius_m * px_per_m))
    if min_r_px >= max_r_px:
        logger.debug("Arc detection: radius range too narrow (%d-%d px)", min_r_px, max_r_px)
        return []

    # Light blur to reduce noise on raster scans
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)

    # Hough circle detection
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=min_r_px * 2,
        param1=canny_high,
        param2=30,
        minRadius=min_r_px,
        maxRadius=max_r_px,
    )

    if circles is None:
        logger.debug("Arc detection: no circles found")
        return []

    circles = np.round(circles[0]).astype(int)
    logger.info("Arc detection: %d candidate circles (r=%d-%d px)", len(circles), min_r_px, max_r_px)

    # Edge map for quadrant analysis
    edges = cv2.Canny(blurred, canny_low, canny_high)

    detections: List[HeuristicDetection] = []

    for cx, cy, r in circles:
        # Check each quadrant for edge density within an annular ring
        # (r-margin .. r+margin) around the circle center.
        margin = max(3, r // 6)
        quadrant_density = []

        for q, (dy_range, dx_range) in enumerate([
            ((-r - margin, 0), (0, r + margin)),        # Q0: top-right
            ((-r - margin, 0), (-r - margin, 0)),       # Q1: top-left
            ((0, r + margin), (-r - margin, 0)),        # Q2: bottom-left
            ((0, r + margin), (0, r + margin)),          # Q3: bottom-right
        ]):
            y0 = max(0, cy + dy_range[0])
            y1 = min(h, cy + dy_range[1])
            x0 = max(0, cx + dx_range[0])
            x1 = min(w, cx + dx_range[1])

            if y1 <= y0 or x1 <= x0:
                quadrant_density.append(0.0)
                continue

            patch = edges[y0:y1, x0:x1]

            # Count edge pixels within the annular ring (distance r +/- margin from center)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            ring_mask = (dist >= r - margin) & (dist <= r + margin)
            ring_pixels = ring_mask.sum()
            edge_in_ring = (patch > 0) & ring_mask
            density = edge_in_ring.sum() / max(1, ring_pixels)
            quadrant_density.append(density)

        # A door arc has significant edge density in exactly 1 quadrant
        densities = sorted(quadrant_density, reverse=True)
        top_density = densities[0]
        second_density = densities[1]

        # Criteria: top quadrant has meaningful edges, others are much weaker
        if top_density < 0.08:
            continue  # Not enough edge signal
        if second_density > top_density * 0.5:
            continue  # Too symmetric — likely a full circle or noise

        # Confidence based on how arc-like the distribution is
        arc_ratio = top_density / max(0.01, sum(densities))
        confidence = min(0.70, 0.40 + arc_ratio * 0.40)

        # Build bounding box around the arc region
        # The arc is in the dominant quadrant; bbox covers the quarter-circle
        bbox = (
            float(max(0, cx - r - margin)),
            float(max(0, cy - r - margin)),
            float(min(w, cx + r + margin)),
            float(min(h, cy + r + margin)),
        )

        detections.append(HeuristicDetection(
            class_name="door",
            bbox=bbox,
            confidence=confidence,
            source="heuristic_arc",
            meta={
                "center_px": (int(cx), int(cy)),
                "radius_px": int(r),
                "radius_m": round(r / px_per_m, 2),
                "quadrant_density": [round(d, 3) for d in quadrant_density],
                "dominant_quadrant": int(quadrant_density.index(top_density)),
            },
        ))

    # NMS within arc detections (close arcs are likely the same door)
    detections = _nms_heuristic(detections, iou_threshold=0.30)
    logger.info("Arc detection: %d door arcs after NMS", len(detections))
    return detections


def detect_window_parallel_lines(
    gray_image: np.ndarray,
    wall_segments: List[Dict[str, Any]],
    px_per_m: float,
    min_parallel_count: int = 3,
    sample_interval_m: float = 0.10,
    scan_width_m: float = 0.15,
) -> List[HeuristicDetection]:
    """Detect windows as clusters of parallel lines within wall segments.

    Windows in architectural plans are typically drawn as 3-4 thin parallel lines
    spanning a portion of the wall.  This function scans perpendicular profiles
    along each wall segment and looks for such parallel-line clusters.

    Args:
        gray_image: Grayscale image (uint8).
        wall_segments: List of wall segment dicts with 'a', 'b' in pixel coords.
        px_per_m: Pixels per metre.
        min_parallel_count: Minimum parallel lines to classify as window.
        sample_interval_m: Distance between perpendicular profile samples.
        scan_width_m: Half-width of the perpendicular scan in metres.
    """
    try:
        import cv2
    except ImportError:
        return []

    if len(gray_image.shape) == 3:
        gray = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_image

    h, w = gray.shape[:2]
    scan_width_px = int(scan_width_m * px_per_m)
    sample_interval_px = max(3, int(sample_interval_m * px_per_m))

    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    detections: List[HeuristicDetection] = []

    for seg in wall_segments:
        a = seg.get("a")
        b = seg.get("b")
        if not a or not b:
            continue

        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        seg_len_px = math.hypot(bx - ax, by - ay)

        if seg_len_px < sample_interval_px * 3:
            continue  # Too short to scan meaningfully

        # Direction and normal
        dx = (bx - ax) / seg_len_px
        dy = (by - ay) / seg_len_px
        nx, ny = -dy, dx  # perpendicular

        # Sample perpendicular profiles along the wall
        n_samples = max(2, int(seg_len_px / sample_interval_px))
        profile_line_counts = []

        for i in range(n_samples):
            t = (i + 0.5) / n_samples
            cx = ax + t * (bx - ax)
            cy = ay + t * (by - ay)

            # Sample points along perpendicular
            profile = []
            for d in range(-scan_width_px, scan_width_px + 1):
                px_x = int(cx + d * nx)
                px_y = int(cy + d * ny)
                if 0 <= px_x < w and 0 <= px_y < h:
                    profile.append(binary[px_y, px_x])
                else:
                    profile.append(0)

            # Count transitions (0->255 or 255->0) = number of line crossings
            transitions = 0
            for j in range(1, len(profile)):
                if profile[j] != profile[j - 1]:
                    transitions += 1
            line_count = transitions // 2  # Each line has 2 transitions
            profile_line_counts.append((t, line_count))

        # Find contiguous regions where line_count >= min_parallel_count
        window_regions = []
        region_start = None
        for t, lc in profile_line_counts:
            if lc >= min_parallel_count:
                if region_start is None:
                    region_start = t
            else:
                if region_start is not None:
                    window_regions.append((region_start, t))
                    region_start = None
        if region_start is not None:
            window_regions.append((region_start, 1.0))

        # Convert regions to detections
        min_window_m = 0.50
        max_window_m = 2.50
        for t_start, t_end in window_regions:
            region_len_m = (t_end - t_start) * seg_len_px / px_per_m
            if region_len_m < min_window_m or region_len_m > max_window_m:
                continue

            # Build bbox for this window region
            p1x = ax + t_start * (bx - ax)
            p1y = ay + t_start * (by - ay)
            p2x = ax + t_end * (bx - ax)
            p2y = ay + t_end * (by - ay)

            margin = scan_width_px
            x1 = min(p1x, p2x) - abs(nx * margin)
            y1 = min(p1y, p2y) - abs(ny * margin)
            x2 = max(p1x, p2x) + abs(nx * margin)
            y2 = max(p1y, p2y) + abs(ny * margin)

            bbox = (
                float(max(0, x1)),
                float(max(0, y1)),
                float(min(w, x2)),
                float(min(h, y2)),
            )

            avg_lines = sum(
                lc for t, lc in profile_line_counts
                if t_start <= t <= t_end
            ) / max(1, sum(1 for t, _ in profile_line_counts if t_start <= t <= t_end))

            confidence = min(0.65, 0.35 + (avg_lines - min_parallel_count) * 0.10)

            detections.append(HeuristicDetection(
                class_name="window",
                bbox=bbox,
                confidence=confidence,
                source="heuristic_parallel",
                meta={
                    "wall_segment": seg.get("id", "?"),
                    "t_range": (round(t_start, 3), round(t_end, 3)),
                    "avg_parallel_lines": round(avg_lines, 1),
                    "width_m": round(region_len_m, 2),
                },
            ))

    detections = _nms_heuristic(detections, iou_threshold=0.35)
    logger.info("Window parallel-line detection: %d windows", len(detections))
    return detections


def _nms_heuristic(
    dets: List[HeuristicDetection],
    iou_threshold: float = 0.35,
) -> List[HeuristicDetection]:
    """Simple NMS for heuristic detections."""
    if len(dets) <= 1:
        return dets
    dets_sorted = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept: List[HeuristicDetection] = []
    for det in dets_sorted:
        if any(_iou_boxes(det.bbox, k.bbox) > iou_threshold for k in kept):
            continue
        kept.append(det)
    return kept


def _iou_boxes(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)
