"""Reviewed Track 1 pixel routines from the teammate contribution.

Source pin and adaptation record: deterministic/TRACK1_SOURCE.json.
Typed app contracts and its duplicate orchestrator are intentionally not imported.
Upstream heuristic scores are not calibrated correctness probabilities.
"""

from __future__ import annotations

import math
from typing import Dict, List

import cv2
import numpy as np


def extract_ink_mask(image: np.ndarray) -> np.ndarray:
    """Extract ink mask from image.

    Args:
        image: RGB or grayscale image array

    Returns:
        Binary mask (255 = ink, 0 = background)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Adaptive thresholding
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return binary


def extract_components(mask: np.ndarray) -> List[Dict]:
    """Extract connected components with holes preserved.

    Args:
        mask: Binary mask

    Returns:
        List of component dicts with bbox, area, contours
    """
    # Find contours with hierarchy to preserve holes
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    components = []
    for i, contour in enumerate(contours):
        # Skip holes (hierarchy[3] != -1 means it's a hole)
        if hierarchy is not None and hierarchy[0][i][3] != -1:
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Get all contours for this component (including holes)
        component_contours = [contour]
        if hierarchy is not None:
            # Find child contours (holes)
            child_idx = hierarchy[0][i][2]  # First child
            while child_idx != -1:
                component_contours.append(contours[child_idx])
                child_idx = hierarchy[0][child_idx][0]  # Next sibling

        components.append(
            {
                "id": f"comp-{i}",
                "bbox": (x, y, x + w, y + h),
                "area": area,
                "contours": component_contours,
                "hierarchy": hierarchy[0][i] if hierarchy is not None else None,
            }
        )

    return components


def extract_contour_rings(mask: np.ndarray) -> List[Dict]:
    """Extract contour rings (outer and inner boundaries).

    Args:
        mask: Binary mask

    Returns:
        List of ring dicts with points and hierarchy
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    rings = []
    for i, contour in enumerate(contours):
        # Simplify contour
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Get points
        points = approx.reshape(-1, 2).tolist()

        # Determine if outer or inner
        is_outer = hierarchy is not None and hierarchy[0][i][3] == -1
        parent_idx = hierarchy[0][i][3] if hierarchy is not None else -1

        rings.append(
            {
                "id": f"ring-{i}",
                "points": points,
                "is_outer": is_outer,
                "parent_idx": parent_idx,
                "area": cv2.contourArea(contour),
                "bbox": cv2.boundingRect(contour),
            }
        )

    return rings


def extract_line_segments(mask: np.ndarray) -> List[Dict]:
    """Extract raw line segments using LSD/Hough.

    Args:
        mask: Binary mask

    Returns:
        List of segment dicts with endpoints and confidence
    """
    segments = []

    # Try LSD first (better for curved/slanted lines)
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = lsd.detect(mask)[0]

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.hypot(x2 - x1, y2 - y1)
                if length > 10:  # Minimum length
                    segments.append(
                        {
                            "id": f"seg-{len(segments)}",
                            "a": (x1, y1),
                            "b": (x2, y2),
                            "length": length,
                            "method": "lsd",
                            "confidence": 1.0,
                        }
                    )
    except Exception:
        pass

    # Fallback to Hough if LSD fails or finds few lines
    if len(segments) < 10:
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10
        )

        if lines is not None:
            for line in lines:
                # Normalize OpenCV line coordinates before unpacking.
                x1, y1, x2, y2 = line.reshape(4)
                length = math.hypot(x2 - x1, y2 - y1)
                if length > 10:
                    segments.append(
                        {
                            "id": f"seg-{len(segments)}",
                            "a": (x1, y1),
                            "b": (x2, y2),
                            "length": length,
                            "method": "hough",
                            "confidence": 0.8,
                        }
                    )

    return segments


def detect_opening_candidates(mask: np.ndarray) -> List[Dict]:
    """Detect potential door/window openings.

    Looks for gaps in walls, arc-like patterns, and rectangular openings.

    Args:
        mask: Binary mask

    Returns:
        List of opening candidate dicts
    """
    candidates = []

    img_h, img_w = mask.shape[:2]
    min_dim = float(min(img_h, img_w))
    # An opening is a small local feature: bound candidates to a modest
    # fraction of the page so merged ink blobs and legend bands cannot
    # masquerade as doors/windows (T2-03 scope sanity).
    max_dim = 0.30 * min_dim
    max_area_frac = 0.02
    img_area = float(img_h * img_w)

    # Method 1: Detect enclosed white holes inside ink (wall gaps,
    # doorway/window reveals). With RETR_CCOMP a hole contour has a
    # non-negative parent index in hierarchy[0][i][3].
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        if hierarchy is None or hierarchy[0][i][3] < 0:
            continue  # not a hole

        area = cv2.contourArea(contour)
        if area < 100:  # Too small
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w > max_dim or h > max_dim or w * h > max_area_frac * img_area:
            continue
        aspect_ratio = w / h if h > 0 else 0

        # Door-like: taller than wide; window-like: wider than tall
        if 0.3 < aspect_ratio < 3.0 and 15 < w < 400 and 15 < h < 400:
            candidates.append(
                {
                    "id": f"opening-{i}",
                    "bbox": (x, y, x + w, y + h),
                    "area": area,
                    "aspect_ratio": aspect_ratio,
                    "kind": "door_candidate" if aspect_ratio < 0.8 else "window_candidate",
                    "is_hole": True,
                }
            )

    # Method 2: Detect arcs (door swings) — small curved ink strokes only.
    # Skip hole contours: a hole already reported as an opening candidate
    # must not be double-counted as an arc of the same contour.
    for i, contour in enumerate(contours):
        if hierarchy is not None and hierarchy[0][i][3] >= 0:
            continue  # hole contour, handled by Method 1
        area = cv2.contourArea(contour)
        if area < 500:  # Too small for door swing
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w > max_dim or h > max_dim or w * h > max_area_frac * img_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        # Circularity = 4π * area / perimeter²
        # Rectangles have circularity ~0.785, circles have ~1.0
        circularity = 4 * math.pi * area / (perimeter * perimeter)

        # Door swings are often quarter-circles or arcs
        if 0.3 < circularity < 0.9:
            candidates.append(
                {
                    "id": f"arc-{i}",
                    "bbox": (x, y, x + w, y + h),
                    "area": area,
                    "aspect_ratio": w / h if h > 0 else 0,
                    "kind": "door_swing_candidate",
                    "is_hole": False,
                    "circularity": circularity,
                }
            )

    return candidates


def detect_stair_candidates(mask: np.ndarray) -> List[Dict]:
    """Detect potential stair candidates.

    Looks for parallel line patterns and tread profiles.

    Args:
        mask: Binary mask

    Returns:
        List of stair candidate dicts
    """
    candidates = []

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 500:  # Too small
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Stairs are typically rectangular and larger
        if w > 50 and h > 50 and w / h > 0.5 and w / h < 3.0:
            candidates.append(
                {
                    "id": f"stair-{i}",
                    "bbox": (x, y, x + w, y + h),
                    "area": area,
                    "kind": "stair_candidate",
                }
            )

    return candidates
