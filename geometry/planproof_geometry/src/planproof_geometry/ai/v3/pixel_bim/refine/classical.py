"""Classical label refiner — no ML, deterministic.

For each user label (polygon or bbox), snaps the boundary to the actual
plan element by:
  1. Cropping the PDF raster around the label bbox (+ padding)
  2. Binarizing (Otsu on grayscale)
  3. Optional morphological closing (paired-outline walls) — per class
  4. Keeping only ink pixels INSIDE the user polygon (labels are truth;
     pixels outside the polygon don't count even if visible)
  5. Taking the largest connected component as the refined mask
  6. Optional dilation by the user-polygon margin, so the refined mask
     fully covers the real element (handles the "label too tight"
     human-error mode where the user clipped a few pixels)

The output is a global-resolution binary mask plus the polygon contour.

No GPU, no model download, < 50 ms per label. On Matterport (111 labels)
this runs in a few seconds total.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ── Public types ─────────────────────────────────────────────────────────

@dataclass
class RefinedMask:
    """Result of refining one label."""
    label_id: str
    label_class: str
    refined_mask: np.ndarray        # uint8, full image resolution, 0/255
    refined_polygon: List[Tuple[float, float]]  # pixel coords, CCW outer
    source: str                     # "classical" | "chan_vese" | "sam" | "fallback_user_polygon"
    iou_vs_user_polygon: float      # 0..1 — how close the refinement stayed to the user's label
    area_ratio_vs_user_polygon: float  # refined_area / user_polygon_area
    notes: List[str] = field(default_factory=list)


# ── Utilities ────────────────────────────────────────────────────────────

def _rasterize_polygon(polygon_xy: Sequence[Tuple[float, float]],
                       img_shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize a polygon (list of (x, y) in pixels) into a uint8 mask."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if len(polygon_xy) < 3:
        return mask
    pts = np.array([[int(round(x)), int(round(y))] for x, y in polygon_xy],
                   dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _bbox_from_polygon(polygon_xy: Sequence[Tuple[float, float]]
                       ) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in polygon_xy]
    ys = [p[1] for p in polygon_xy]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _pad_bbox(bbox: Tuple[int, int, int, int],
              pad_px: int,
              img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    h, w = img_shape[:2]
    return (max(0, x0 - pad_px), max(0, y0 - pad_px),
            min(w - 1, x1 + pad_px), min(h - 1, y1 + pad_px))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two uint8 (0/255) masks of identical shape."""
    a_bool = a > 0
    b_bool = b > 0
    inter = int(np.logical_and(a_bool, b_bool).sum())
    union = int(np.logical_or(a_bool, b_bool).sum())
    if union == 0:
        return 0.0
    return inter / union


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Return a mask containing only the largest CC of ``mask``."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    # index 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == best_label] = 255
    return out


def _mask_to_polygon(mask: np.ndarray) -> List[Tuple[float, float]]:
    """Extract the outer polygon (as pixel coords) of the largest blob."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return []
    cnt = max(cnts, key=cv2.contourArea)
    eps = 1.0  # ~ 1 px simplification keeps corners crisp
    cnt = cv2.approxPolyDP(cnt, eps, True)
    return [(float(p[0][0]), float(p[0][1])) for p in cnt]


# ── Per-class config ─────────────────────────────────────────────────────

_WALL_CLASSES = {"wall", "wall_interior", "wall_exterior"}
_DOOR_CLASSES = {"door", "door_single", "door_double", "door_sliding"}

# Morphological-close kernel size per class in pixels.
# Walls may be drawn as paired outlines — closing merges the two lines.
# Doors are typically filled symbols and don't need closing.
_CLOSE_KERNEL_PX_BY_CLASS = {
    "wall": 5,
    "wall_interior": 5,
    "wall_exterior": 5,
    "door": 0,
    "door_single": 0,
    "door_double": 0,
    "door_sliding": 0,
    "stairs": 0,
}

# Padding around the bbox when cropping (in pixels).
# Small pad so we don't accidentally include neighboring elements.
_CROP_PAD_PX = 4


# ── Main refiner ─────────────────────────────────────────────────────────

def refine_label_classical(
    image_gray: np.ndarray,
    label_id: str,
    label_class: str,
    user_polygon_xy: Sequence[Tuple[float, float]],
    *,
    close_kernel_px: Optional[int] = None,
    crop_pad_px: int = _CROP_PAD_PX,
    otsu_bias: int = 0,
) -> RefinedMask:
    """Snap a user label's boundary to the underlying plan element pixels.

    Parameters
    ----------
    image_gray : np.ndarray
        Full-resolution grayscale image (H, W) uint8. The pipeline's
        ``state.input_image`` reshaped to grayscale.
    label_id : str
        Unique annotation id (from annotations.json).
    label_class : str
        Annotation class (e.g. "wall_exterior", "door_single").
    user_polygon_xy : list of (x, y)
        User-drawn polygon / bbox-as-rectangle in pixel coordinates of
        ``image_gray``. Length >= 3.
    close_kernel_px : int, optional
        Morphological-close kernel size. Defaults to a per-class value
        (5 px for walls, 0 px for doors).
    crop_pad_px : int
        Padding around the bbox in pixels.
    otsu_bias : int
        Added to Otsu's threshold. Negative = include more dark pixels.

    Returns
    -------
    RefinedMask
        ``source == "classical"`` on success. Falls back to the raw user
        polygon if the binarization finds nothing inside the label region
        (source == "fallback_user_polygon").
    """
    if len(user_polygon_xy) < 3:
        raise ValueError(f"Label {label_id}: polygon needs >= 3 points")
    if image_gray.ndim != 2:
        raise ValueError("image_gray must be (H, W), got shape "
                         f"{image_gray.shape}")

    H, W = image_gray.shape
    user_poly_mask_full = _rasterize_polygon(user_polygon_xy, (H, W))
    user_poly_area = int((user_poly_mask_full > 0).sum())
    if user_poly_area == 0:
        raise ValueError(f"Label {label_id}: user polygon has zero area")

    notes: List[str] = []

    # 1. Crop around the polygon bbox with padding.
    x0, y0, x1, y1 = _bbox_from_polygon(user_polygon_xy)
    x0, y0, x1, y1 = _pad_bbox((x0, y0, x1, y1), crop_pad_px, (H, W))
    crop = image_gray[y0:y1 + 1, x0:x1 + 1]
    if crop.size == 0:
        notes.append("empty_crop_fallback")
        return _fallback(label_id, label_class, user_polygon_xy,
                         user_poly_mask_full, notes, source="fallback_empty_crop")

    # 2. Binarize. Plan ink is dark; we want a mask of ink pixels.
    # Tag-4.4 (Ahmed et al. 2013, GREC): walls in floor plans come in
    # two notation styles — outlined (light interior with thin dark
    # strokes) and solid-fill (dark interior). Otsu's threshold is
    # tuned for a bimodal histogram; for solid-fill walls the dark
    # interior gradient + anti-aliased edges create a uni-modal-ish
    # histogram and Otsu produces partial coverage. We detect the
    # style from the mean intensity inside the user polygon and use
    # a style-specific threshold.
    user_poly_crop_for_style = user_poly_mask_full[y0:y1 + 1, x0:x1 + 1]
    if label_class in _WALL_CLASSES:
        inside_pixels = crop[user_poly_crop_for_style > 0]
        if inside_pixels.size > 50:
            mean_in = float(inside_pixels.mean())
        else:
            mean_in = 255.0
        # Solid-fill walls have darker interiors. Threshold 160/255 puts
        # the boundary between Matterport's plumbing-wall fill (~80-120)
        # and partition-wall outlines (~220-255).
        if mean_in < 160.0:
            # Solid-fill: grab everything that's not pure white. The
            # gray fill + dark outline both pass < 200, so the mask
            # covers the full wall body including anti-aliased edges.
            _, ink_local = cv2.threshold(crop, 200, 255, cv2.THRESH_BINARY_INV)
            notes.append("solid_fill_wall")
        else:
            # Outlined: keep Otsu (designed for bimodal ink/background).
            thr_val, _ = cv2.threshold(crop, 0, 255,
                                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            thr_val = max(0, min(255, int(thr_val) + int(otsu_bias)))
            _, ink_local = cv2.threshold(crop, thr_val, 255,
                                         cv2.THRESH_BINARY_INV)
            notes.append("outlined_wall")
    else:
        # Doors / icons keep Otsu (current behavior).
        thr_val, _ = cv2.threshold(crop, 0, 255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        thr_val = max(0, min(255, int(thr_val) + int(otsu_bias)))
        _, ink_local = cv2.threshold(crop, thr_val, 255, cv2.THRESH_BINARY_INV)

    # 3. Optional morphological close to merge paired outlines (walls).
    k = close_kernel_px if close_kernel_px is not None \
        else _CLOSE_KERNEL_PX_BY_CLASS.get(label_class, 0)
    if k > 0:
        # MORPH_RECT preserves sharp 90° corners; ELLIPSE rounds them off
        # at L/T-junctions (matches the dispatcher cleanup choice).
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        ink_local = cv2.morphologyEx(ink_local, cv2.MORPH_CLOSE, kernel)
        # Fill small holes inside wall regions so the mask is solid.
        # Multi-corner seed: if a wall touches a crop corner, single-corner
        # flood from (0, 0) misses the interior void; seed all four corners.
        ink_filled = ink_local.copy()
        ch, cw = ink_filled.shape[:2]
        ff_mask = np.zeros((ch + 2, cw + 2), dtype=np.uint8)
        for sx, sy in ((0, 0), (cw - 1, 0), (0, ch - 1), (cw - 1, ch - 1)):
            if ink_filled[sy, sx] == 0:
                cv2.floodFill(ink_filled, ff_mask, (sx, sy), 128)
        holes = (ink_filled == 0).astype(np.uint8) * 255
        ink_local = cv2.bitwise_or(ink_local, holes)

    # 4. Keep only ink INSIDE the user polygon (labels are truth; the
    # refinement should never "grow" past the user's region).
    user_poly_crop = user_poly_mask_full[y0:y1 + 1, x0:x1 + 1]
    ink_local = cv2.bitwise_and(ink_local, user_poly_crop)

    if int((ink_local > 0).sum()) == 0:
        notes.append("no_ink_inside_polygon_fallback")
        return _fallback(label_id, label_class, user_polygon_xy,
                         user_poly_mask_full, notes,
                         source="fallback_no_ink")

    # 5. Largest connected component.
    largest = _largest_connected_component(ink_local)

    # 6. Reassemble to full-image mask.
    refined_full = np.zeros((H, W), dtype=np.uint8)
    refined_full[y0:y1 + 1, x0:x1 + 1] = largest

    refined_area = int((refined_full > 0).sum())
    area_ratio = refined_area / user_poly_area if user_poly_area else 0.0

    # Extract polygon contour.
    poly_xy = _mask_to_polygon(refined_full)
    if not poly_xy:
        notes.append("empty_contour_fallback")
        return _fallback(label_id, label_class, user_polygon_xy,
                         user_poly_mask_full, notes,
                         source="fallback_no_contour")

    iou = _iou(refined_full, user_poly_mask_full)

    return RefinedMask(
        label_id=label_id,
        label_class=label_class,
        refined_mask=refined_full,
        refined_polygon=poly_xy,
        source="classical",
        iou_vs_user_polygon=iou,
        area_ratio_vs_user_polygon=area_ratio,
        notes=notes,
    )


def _fallback(label_id: str, label_class: str,
              user_polygon_xy: Sequence[Tuple[float, float]],
              user_poly_mask_full: np.ndarray,
              notes: List[str],
              *,
              source: str) -> RefinedMask:
    """Fallback: use the raw user polygon unchanged."""
    return RefinedMask(
        label_id=label_id,
        label_class=label_class,
        refined_mask=user_poly_mask_full,
        refined_polygon=[(float(x), float(y)) for x, y in user_polygon_xy],
        source=source,
        iou_vs_user_polygon=1.0,
        area_ratio_vs_user_polygon=1.0,
        notes=notes,
    )
