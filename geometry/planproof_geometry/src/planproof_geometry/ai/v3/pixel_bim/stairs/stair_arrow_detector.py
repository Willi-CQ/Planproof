"""Detect the up-arrow inside a stair user-polygon.

Floor plans conventionally draw a directional arrow inside the stair
footprint pointing in the walking-up direction. The arrow is typically:

    * a long thin shaft roughly parallel to the stair's long axis,
    * a small triangular arrowhead at one end (wider than the shaft).

Detecting this arrow is much more reliable than the indirect density-
based heuristics in ``stair_direction``: the arrow is an explicit
intent signal from whoever drew the plan.

Algorithm (Tag 8 — 2026-05-22):

    1. Rasterise the user polygon, dark-mask the ROI.
    2. Compute connected components with ``cv2.connectedComponentsWithStats``.
    3. For each component, run PCA to get its (long_axis_unit,
       length_px, width_px, elongation). Filter:
           * area >= MIN_COMPONENT_AREA_PX
           * elongation >= MIN_ELONGATION (long & thin)
           * angle vs stair-long-axis <= MAX_ARROW_AXIS_DEVIATION_DEG
           * length_px >= MIN_ARROW_LENGTH_RATIO * stair-length-px
       The candidate set rarely exceeds 1-2 entries on real plans —
       tread lines are perpendicular (≈90° off), wall ink is too short,
       label text is too short or oriented along reading-axis.
    4. Pick the longest remaining candidate.
    5. Resolve the arrowhead end: bucketise the candidate's pixels along
       its long axis and measure transverse spread per bucket. The
       arrowhead end has noticeably wider spread than the shaft.
    6. Return ``(direction_xy, confidence)`` where confidence reflects
       the asymmetry ratio.

When no candidate survives, returns ``(None, 0.0)``. The caller is
expected to fall back to the density-based heuristic voters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Tunables -----------------------------------------------------------------
MIN_COMPONENT_AREA_PX = 30
MIN_ELONGATION = 8.0          # max_eigval / min_eigval — arrow is very thin
MAX_ARROW_AXIS_DEVIATION_DEG = 25.0
MIN_ARROW_LENGTH_RATIO = 0.25
N_TIP_BUCKETS = 12            # along-axis buckets for tip detection
TIP_SPREAD_RATIO_MIN = 1.20   # tip spread / shaft spread must exceed this
DARK_THRESHOLD = 140


@dataclass
class ArrowDetection:
    """Per-candidate diagnostic plus the chosen direction.

    ``direction_xy`` is the unit vector pointing **from arrow-tail to
    arrow-tip** in plan-px frame (matches the convention used by
    ``stair_direction.detect_direction_up``: a +long_axis arrow with
    its tip on the +long_axis end means direction_up = +long_axis).
    """

    direction_xy: Optional[Tuple[float, float]] = None
    confidence: float = 0.0
    component_length_px: float = 0.0
    component_elongation: float = 0.0
    tip_spread_ratio: float = 0.0
    candidates_considered: int = 0
    notes: List[str] = field(default_factory=list)


def _polygon_mask_px(
    polygon_px: Sequence[Tuple[float, float]],
    image_shape_hw: Tuple[int, int],
    expand_px: float = 0.0,
) -> np.ndarray:
    """Rasterise a polygon into a uint8 mask (255 inside, 0 outside).

    When ``expand_px > 0`` the polygon is dilated outward by that many
    pixels before rasterisation — useful when an inset detector
    (arrows in floor plans) needs to scan a bit beyond the user bbox
    because the arrow's shaft / tip extends past the labelled region.
    """
    import cv2  # noqa: PLC0415

    H, W = image_shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.array(
        [[int(round(x)), int(round(y))] for x, y in polygon_px],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    if expand_px > 0.0:
        kernel_size = max(1, int(round(expand_px * 2 + 1)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size),
        )
        # Dilate only a cropped window around the polygon bbox, then paste
        # back — the result is BIT-FOR-BIT identical to dilating the full
        # image (verified), because the dilation reaches at most ``kernel//2``
        # px beyond any set pixel and the window is padded by more than that.
        #
        # A full-image cv2.dilate with this ELLIPSE kernel is O(H*W*r^2): at
        # 4x render (~95 MP) with a ~450 px kernel it cost ~220 s PER CALL and
        # was ~90 % of the entire stairs pipeline runtime. The polygon is tiny
        # (~350 px), so the window is ~1.3 MP instead of 95 MP — ~80x faster,
        # same output.
        pad = kernel_size // 2 + 1
        ys, xs = np.where(mask > 0)
        if len(xs):
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(W, int(xs.max()) + pad + 1)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(H, int(ys.max()) + pad + 1)
            mask[y0:y1, x0:x1] = cv2.dilate(
                mask[y0:y1, x0:x1], kernel, iterations=1,
            )
    return mask


def _component_pca(
    ys: np.ndarray, xs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Return (centroid, long_axis_unit, length_px, width_px, eigvals).

    eigvals are returned in ascending order (numpy convention).
    """
    pts = np.column_stack((xs, ys)).astype(np.float64)
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    cov = np.cov(centred, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis = eigvecs[:, -1]
    short_axis = np.array([-long_axis[1], long_axis[0]], dtype=np.float64)
    proj_long = centred @ long_axis
    proj_short = centred @ short_axis
    length_px = float(proj_long.max() - proj_long.min())
    width_px = float(proj_short.max() - proj_short.min())
    return centroid, long_axis, length_px, width_px, eigvals


def _angle_diff_deg(a_unit: np.ndarray, b_unit: Tuple[float, float]) -> float:
    """Smallest unsigned angle between two unit vectors mod 180°."""
    bx, by = b_unit
    cos_t = abs(a_unit[0] * bx + a_unit[1] * by)
    cos_t = max(-1.0, min(1.0, cos_t))
    return math.degrees(math.acos(cos_t))


def _tip_spread_per_bucket(
    pts: np.ndarray,
    centroid: np.ndarray,
    long_axis: np.ndarray,
    n_buckets: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bucketise pts along long_axis, return per-bucket transverse spread.

    Returns (bucket_centres_proj_long, bucket_spreads).
    """
    short_axis = np.array([-long_axis[1], long_axis[0]], dtype=np.float64)
    centred = pts - centroid
    proj_long = centred @ long_axis
    proj_short = centred @ short_axis
    edges = np.linspace(proj_long.min(), proj_long.max(), n_buckets + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    spreads = np.zeros(n_buckets, dtype=np.float64)
    for j in range(n_buckets):
        in_bucket = (proj_long >= edges[j]) & (proj_long <= edges[j + 1])
        if in_bucket.any():
            spreads[j] = float(proj_short[in_bucket].max()
                               - proj_short[in_bucket].min())
    return centres, spreads


def detect_arrow_direction(
    image_gray: Optional[np.ndarray],
    user_polygon_px: Sequence[Tuple[float, float]],
    long_axis_xy: Tuple[float, float],
    *,
    stair_length_px: float,
    dark_threshold: int = DARK_THRESHOLD,
    dark_band: Optional[Tuple[int, int]] = None,
    expand_px: float = 0.0,
) -> ArrowDetection:
    """Find the up-arrow inside the polygon, return its direction.

    Args:
        image_gray: rendered plan grayscale image.
        user_polygon_px: stair user polygon in render-pixel coords.
        long_axis_xy: stair PCA long-axis unit vector. Used to filter
            candidates by parallel-ness.
        stair_length_px: length of the stair footprint in render pixels
            (used as scale for the minimum arrow-length filter).
        dark_threshold: ink threshold. Pixels with intensity ``< dark_threshold``
            count as ink (default 140).
        dark_band: optional ``(lo, hi)`` intensity band. When provided
            it OVERRIDES ``dark_threshold`` — only pixels with
            ``lo <= intensity <= hi`` count as ink. Useful for detecting
            grey arrows that sit above the strict ink threshold but
            below the page-white background, while excluding the
            black tread lines that share the ROI.
        expand_px: dilate the user polygon by this many pixels before
            scanning. Floor-plan direction arrows sometimes extend past
            the user bbox (e.g. the arrowhead pokes out the side); a
            small dilation captures the full arrow without bringing in
            adjacent walls.

    Returns:
        ``ArrowDetection`` — ``direction_xy is None`` if no arrow was
        identified (caller falls back to other voters).
    """
    if image_gray is None or len(user_polygon_px) < 3:
        return ArrowDetection(notes=["no input image / polygon"])

    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return ArrowDetection(notes=["cv2 unavailable"])

    H, W = image_gray.shape[:2]
    mask = _polygon_mask_px(user_polygon_px, (H, W), expand_px=expand_px)
    if mask.sum() == 0:
        return ArrowDetection(notes=["empty polygon mask"])

    if dark_band is not None:
        lo, hi = dark_band
        ink = (image_gray >= lo) & (image_gray <= hi)
    else:
        ink = image_gray < dark_threshold
    binary = (ink & (mask > 0)).astype(np.uint8)
    if binary.sum() < MIN_COMPONENT_AREA_PX:
        return ArrowDetection(notes=["too few dark pixels"])

    n_components, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )

    min_len_px = MIN_ARROW_LENGTH_RATIO * max(1.0, stair_length_px)
    candidates: List[dict] = []

    for cid in range(1, n_components):
        comp_mask = (labels == cid)
        area = int(comp_mask.sum())
        if area < MIN_COMPONENT_AREA_PX:
            continue
        ys, xs = np.where(comp_mask)
        try:
            centroid, long_axis, length_px, width_px, eigvals = _component_pca(
                ys, xs,
            )
        except (np.linalg.LinAlgError, ValueError):
            continue
        if length_px < min_len_px:
            continue
        if eigvals[0] <= 1e-9:
            elongation = float("inf")
        else:
            elongation = float(eigvals[-1] / eigvals[0])
        if elongation < MIN_ELONGATION:
            continue
        if _angle_diff_deg(long_axis, long_axis_xy) > MAX_ARROW_AXIS_DEVIATION_DEG:
            continue
        candidates.append({
            "cid": cid,
            "centroid": centroid,
            "long_axis": long_axis,
            "length_px": length_px,
            "width_px": width_px,
            "elongation": elongation,
            "ys": ys,
            "xs": xs,
        })

    if not candidates:
        return ArrowDetection(
            candidates_considered=n_components - 1,
            notes=["no parallel elongated component found"],
        )

    # Pick the longest candidate. Real arrows are the longest parallel
    # stripe inside the stair (tread lines run perpendicular).
    candidates.sort(key=lambda c: -c["length_px"])
    arrow = candidates[0]

    pts = np.column_stack((arrow["xs"], arrow["ys"])).astype(np.float64)
    _centres, spreads = _tip_spread_per_bucket(
        pts, arrow["centroid"], arrow["long_axis"], N_TIP_BUCKETS,
    )

    # Compare the first-third vs last-third bucket spreads. The
    # arrowhead end has wider spread because of the triangular tip.
    third = max(1, N_TIP_BUCKETS // 3)
    head_spread = float(spreads[:third].max()) if third else 0.0
    tail_spread = float(spreads[-third:].max()) if third else 0.0
    if head_spread <= 1e-6 and tail_spread <= 1e-6:
        return ArrowDetection(
            candidates_considered=len(candidates),
            component_length_px=arrow["length_px"],
            component_elongation=arrow["elongation"],
            notes=["no spread asymmetry detected"],
        )

    head_first = head_spread > tail_spread
    big = max(head_spread, tail_spread)
    small = max(min(head_spread, tail_spread), 1e-6)
    ratio = big / small
    if ratio < TIP_SPREAD_RATIO_MIN:
        return ArrowDetection(
            candidates_considered=len(candidates),
            component_length_px=arrow["length_px"],
            component_elongation=arrow["elongation"],
            tip_spread_ratio=ratio,
            notes=["spread ratio below threshold"],
        )

    # The arrow's long_axis from PCA has an arbitrary sign. The arrow
    # tip is the end with wider transverse spread; the direction points
    # FROM tail TO tip.
    long_axis = arrow["long_axis"]
    # spreads is indexed first→last along arrow's long-axis direction.
    # If head_spread (first bucket) > tail_spread, the tip is at the
    # -long_axis end (because bucket 0 corresponds to proj_long.min()).
    tip_at_negative_end = head_first
    if tip_at_negative_end:
        direction = (-float(long_axis[0]), -float(long_axis[1]))
    else:
        direction = (float(long_axis[0]), float(long_axis[1]))

    # Confidence: how strongly the asymmetry exceeds the threshold,
    # clamped to [0, 1].
    confidence = min(1.0, (ratio - 1.0) / (TIP_SPREAD_RATIO_MIN * 1.5))

    return ArrowDetection(
        direction_xy=direction,
        confidence=confidence,
        component_length_px=arrow["length_px"],
        component_elongation=arrow["elongation"],
        tip_spread_ratio=ratio,
        candidates_considered=len(candidates),
        notes=[
            f"arrow ok: length={arrow['length_px']:.1f}px "
            f"elong={arrow['elongation']:.1f} ratio={ratio:.2f}",
        ],
    )
