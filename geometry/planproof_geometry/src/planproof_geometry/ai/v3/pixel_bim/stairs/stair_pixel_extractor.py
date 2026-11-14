"""Pixel-based stair footprint + tread-count extraction (Tag 2).

Mirrors the door/window pixel-scan pattern (``wall_scan``, ``jamb_probe``):
the user's bbox/polygon is treated as a *region of interest*, not as the
final footprint. The actual stair geometry is recovered from the dark
pixels inside that region.

Pipeline inside this module:

    1. Build a binary mask ``in_region & ink_dark``. The ink threshold
       is **adaptive** (Otsu over the ROI with a bimodality safety net
       and a percentile fallback for multimodal / flat histograms) so
       light CAD strokes, hatched plans and scanned blueprints all
       resolve to the actual tread ink.
    2. Run PCA over the dark pixel positions -> tighter long axis,
       short axis, and extents (typically 5-15 % smaller than the user
       region because the user includes margin).
    3. Sample dark-pixel density along the long axis at ``n_samples``
       ticks within a narrow short-axis strip -> a 1-D profile whose
       peaks correspond to tread edges.
    4. Aggregate the tread count via a 2-Voter pattern (Density-Peak
       primary + DIN-Target sanity-check). If the two voters disagree
       by > 30 % a ``low_confidence_tread_count`` note is recorded;
       n_steps still falls back to the Density-Peak count.

Direction-up resolution is intentionally NOT in this module. It lives
in ``stairs.stair_direction`` (Tag 3) and consumes the same 1-D
density profile this module produces.

Falls back gracefully:
    * ``image_gray is None``      -> ``PixelExtractionUnavailable``
    * mask has < 50 dark pixels   -> ``PixelExtractionUnavailable``
    * PCA degenerates             -> ``PixelExtractionUnavailable``
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .footprint import StairFootprint
from .stair_din import DEFAULT_TARGET_GOING_M

# Fixed floor / safety threshold. Adaptive Otsu may return lower
# values on very dark scans; we never go below this to keep tread
# ink and wall ink in the same darker-class.
DARK_THRESHOLD_FLOOR = 140
# Minimum dark pixels under the user mask to attempt PCA.
MIN_DARK_PIXELS = 50
# Histogram-bimodality cutoff. Below this score we fall back to the
# percentile threshold because Otsu's assumption (two distinct peaks)
# does not hold.
BIMODALITY_CUTOFF = 0.30
# Percentile of ROI intensities used when the histogram is multimodal
# / flat. 15 % captures "the darkest 15 % of pixels" — robust enough
# for hatched plans and scanned blueprints.
PERCENTILE_FALLBACK = 15
# 2-Voter aggregator: confidences and disagreement threshold.
VOTER_A_BASE_CONFIDENCE = 1.0   # Density-Peak (Pixel-Evidence)
VOTER_B_BASE_CONFIDENCE = 0.4   # DIN-Target (length / 0.29)
VOTER_DISAGREEMENT_PCT = 0.30   # > 30 % spread -> low-confidence note

# Connected-components filter (Tag 8 — 2026-05-22) ------------------------
# Stripe components (tread lines, arrow shaft) are LONG and THIN; block
# components (wall stubs, columns intruding into the user bbox) are
# COMPACT and CHUNKY. We exclude block components from the footprint
# PCA so non-rectangular stair labels containing wall ink do not skew
# the centroid / extents. Density-based classification was tried first
# but failed because a 2-px-wide pen-stroke tread line has bbox density
# = 1.0 (the bbox tightly hugs the line) — aspect ratio is the only
# reliable discriminator.
COMPONENT_MIN_AREA_PX = 30
COMPONENT_STRIPE_MIN_ASPECT = 3.0
# Maximum deviation (degrees) between a stripe component's long axis
# and the *expected* tread-line orientation (perpendicular to the
# stair's long axis) before the component is reclassified as "wall"
# and dropped from the PCA pool. Tread lines are drawn perpendicular
# to the walking direction; bbox-edge wall ink runs PARALLEL to the
# walking direction and used to fool the classifier into accepting
# wall-ink as a tread.
COMPONENT_TREAD_PERPENDICULAR_TOL_DEG = 25.0


class PixelExtractionUnavailable(RuntimeError):
    """Raised when the rendered image cannot drive the extraction.

    The pipeline catches this and falls back to polygon-PCA on the
    raw user polygon.
    """


@dataclass
class StairPixelDetails:
    """Diagnostic side-channel for the pixel-extraction pass.

    Carries data downstream (Tag 3 direction-up uses ``density_profile``
    and ``long_axis_px`` to vote without re-rendering the ROI).
    """

    # Extraction stats
    n_dark_pixels: int = 0
    dark_threshold_used: int = 0
    threshold_source: str = "otsu"
    # ^ one of: "otsu" | "percentile_fallback" | "floor"

    # Tread count voters
    voter_a_density_peak: Optional[int] = None
    voter_b_din_target: Optional[int] = None
    voter_a_confidence: float = 0.0
    voter_b_confidence: float = VOTER_B_BASE_CONFIDENCE
    n_steps: Optional[int] = None
    detected_going_m: Optional[float] = None
    low_confidence_tread_count: bool = False

    # Density profile (used by Tag 3 direction-up). Plain numpy 1-D
    # array of length ``profile_n_samples``; the long-axis-tick at
    # index 0 is the -long_axis side, index -1 is the +long_axis side.
    density_profile: Optional[np.ndarray] = None
    profile_n_samples: int = 0
    # Absolute plan-XY positions (in metres) of each detected tread
    # centroid in the rendered-plan frame. ``None`` if no peaks were
    # detected. These positions are anchored to the actual ink — they
    # do NOT shift when downstream code re-anchors the footprint to a
    # different axis (e.g. user-bbox snap), because they reference the
    # original PCA frame absolutely.
    tread_positions_xy_m: Optional[List[Tuple[float, float]]] = None
    # Long-axis unit vector in pixel-frame, used by direction module
    # to convert profile-index back into a +/- long_axis sign without
    # re-running PCA.
    long_axis_px: Tuple[float, float] = (1.0, 0.0)
    # Stair length in render pixels — passed to the arrow detector so
    # it can filter candidates by minimum length.
    length_px: float = 0.0

    # Component-classification stats (Tag 8 connected-components filter).
    n_components: int = 0
    n_stripe_components: int = 0
    n_block_components: int = 0
    used_component_filter: bool = False

    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _polygon_mask_px(
    polygon_px: Sequence[Tuple[float, float]],
    image_shape_hw: Tuple[int, int],
) -> np.ndarray:
    """Rasterise a polygon into a uint8 mask (255 inside, 0 outside)."""
    import cv2  # noqa: PLC0415

    H, W = image_shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.array(
        [[int(round(x)), int(round(y))] for x, y in polygon_px],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _polygon_with_holes_mask_px(
    rings_px: Sequence[Sequence[Tuple[float, float]]],
    image_shape_hw: Tuple[int, int],
) -> np.ndarray:
    """Rasterise a polygon-with-holes into a uint8 mask.

    ``rings_px[0]`` is the exterior ring (filled with 255). Each
    subsequent ring is a hole (filled back to 0).
    """
    import cv2  # noqa: PLC0415

    H, W = image_shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    if not rings_px:
        return mask
    exterior = rings_px[0]
    if len(exterior) >= 3:
        pts = np.array(
            [[int(round(x)), int(round(y))] for x, y in exterior],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    for ring in rings_px[1:]:
        if len(ring) < 3:
            continue
        pts = np.array(
            [[int(round(x)), int(round(y))] for x, y in ring],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 0)
    return mask


def _bimodality_score(hist: np.ndarray) -> float:
    """Cheap bimodality proxy: ratio of two-largest-peaks vs total mass.

    A clean black/white floor-plan histogram has two strong, well-
    separated peaks; this function returns a number close to 1.0 in
    that case and decays to ~0 for flat or many-peaked distributions.

    The proxy is intentionally simple — we only need to decide whether
    Otsu's assumption (clean bimodal) holds. Anything more elaborate
    is overkill for a per-stair pre-flight.

    Args:
        hist: 1-D histogram with non-negative counts (any length).

    Returns:
        Score in [0.0, 1.0].
    """
    if hist.size == 0:
        return 0.0
    total = float(hist.sum())
    if total <= 0.0:
        return 0.0
    # Find two largest values that are at least 16 bins apart (so we
    # do not double-count the same peak's shoulder).
    sorted_idx = np.argsort(-hist)
    peak1 = int(sorted_idx[0])
    peak2_idx = None
    for i in sorted_idx[1:]:
        if abs(int(i) - peak1) >= 16:
            peak2_idx = int(i)
            break
    if peak2_idx is None:
        return 0.0
    p1 = float(hist[peak1])
    p2 = float(hist[peak2_idx])
    return (p1 + p2) / total


def compute_adaptive_threshold(
    roi_intensities: np.ndarray,
) -> Tuple[int, str]:
    """Pick an ink-threshold for the stair ROI.

    Strategy (in order):
        1. Compute Otsu over the ROI intensities and the bimodality
           score of the underlying histogram.
        2. If bimodal (score >= ``BIMODALITY_CUTOFF``):
               threshold = max(otsu, DARK_THRESHOLD_FLOOR).
           Source tag ``"otsu"`` or ``"floor"`` depending on which won.
        3. Otherwise (multimodal / flat):
               threshold = max(percentile-15 of ROI, DARK_THRESHOLD_FLOOR).
           Source tag ``"percentile_fallback"``.

    Args:
        roi_intensities: 1-D array of uint8 pixel values inside the
            user polygon. Pixels outside the polygon must NOT be
            included — Otsu would otherwise be dominated by the
            page-white background.

    Returns:
        Tuple of ``(threshold_value, source_tag)``.
    """
    if roi_intensities.size == 0:
        return DARK_THRESHOLD_FLOOR, "floor"

    import cv2  # noqa: PLC0415

    # cv2.threshold needs a 2-D array.
    arr = roi_intensities.reshape(-1, 1).astype(np.uint8)
    otsu_t, _ = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_t = int(otsu_t)

    hist = np.bincount(roi_intensities.ravel(), minlength=256).astype(np.float64)
    score = _bimodality_score(hist)

    if score >= BIMODALITY_CUTOFF:
        if otsu_t >= DARK_THRESHOLD_FLOOR:
            return otsu_t, "otsu"
        return DARK_THRESHOLD_FLOOR, "floor"

    pct = int(np.percentile(roi_intensities, PERCENTILE_FALLBACK))
    if pct >= DARK_THRESHOLD_FLOOR:
        return pct, "percentile_fallback"
    return DARK_THRESHOLD_FLOOR, "floor"


def _classify_components_stripe_vs_block(
    binary: np.ndarray,
    *,
    stair_long_axis_hint: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, dict]:
    """Filter dark mask to keep only stripe-like (tread/arrow) components.

    A "stripe" has:
        * area >= ``COMPONENT_MIN_AREA_PX``
        * aspect ratio (long/short bbox edge) >= ``COMPONENT_STRIPE_MIN_ASPECT``
        * EITHER no orientation hint provided, OR the component's PCA
          long axis is PERPENDICULAR (±25°) to ``stair_long_axis_hint``.
          A stripe that runs PARALLEL to the stair's long axis is wall
          ink masquerading as a tread — drop it.

    A "block" has either:
        * area >= 3 * COMPONENT_MIN_AREA_PX and aspect ratio < 3
          (chunky, near-square — wall corner, column)
        * OR (with orientation hint) is a stripe that runs PARALLEL to
          the stair's long axis. These are wall slabs along the bbox
          edge that the L-shape includes inside the trimmed polygon
          before wall-subtraction trims them off.

    Args:
        binary: uint8 image with 0/255 ink values.
        stair_long_axis_hint: optional unit vector for the stair's
            walking direction in image-px coords. When provided, stripes
            whose own long axis is within 25° of this direction are
            treated as walls and excluded from the PCA pool.

    Returns:
        ``(filtered_mask, stats_dict)`` where ``filtered_mask`` has the
        same shape as ``binary`` and contains only stripe + other
        components (blocks zeroed out). ``stats_dict`` carries
        ``n_components / n_stripe / n_block / n_wall_orient`` for
        diagnostics.
    """
    import cv2  # noqa: PLC0415

    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)
    n_components, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )

    keep = np.zeros_like(binary, dtype=bool)
    n_stripe = 0
    n_block = 0
    n_wall_orient = 0

    # Normalise the hint to a unit vector once.
    hint = None
    if stair_long_axis_hint is not None:
        hx, hy = stair_long_axis_hint
        norm = float(np.hypot(hx, hy))
        if norm > 1e-9:
            hint = (hx / norm, hy / norm)

    for cid in range(1, n_components):
        x, y, w, h, area = stats[cid]
        if area < COMPONENT_MIN_AREA_PX:
            continue
        long_edge = max(w, h)
        short_edge = max(1, min(w, h))
        aspect = long_edge / short_edge

        is_stripe = aspect >= COMPONENT_STRIPE_MIN_ASPECT
        is_block = (
            not is_stripe and area >= 3 * COMPONENT_MIN_AREA_PX
        )

        # Orientation-aware re-classification: if the stripe is parallel
        # to the stair long axis, it's wall ink — re-label as "wall".
        if is_stripe and hint is not None:
            ys, xs = np.where(labels == cid)
            if ys.size >= 8:  # PCA needs at least a few pixels.
                pts = np.column_stack((xs, ys)).astype(np.float64)
                centred = pts - pts.mean(axis=0)
                try:
                    cov = np.cov(centred, rowvar=False)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    comp_long = eigvecs[:, -1]
                    if comp_long[0] < 0:
                        comp_long = -comp_long
                    cos_angle = abs(
                        float(comp_long[0]) * hint[0]
                        + float(comp_long[1]) * hint[1]
                    )
                    cos_angle = min(1.0, cos_angle)
                    angle_deg = float(np.degrees(np.arccos(cos_angle)))
                    parallel_to_long = (
                        angle_deg <= COMPONENT_TREAD_PERPENDICULAR_TOL_DEG
                    )
                    if parallel_to_long:
                        n_wall_orient += 1
                        continue
                except (np.linalg.LinAlgError, ValueError):
                    pass

        if is_stripe:
            keep[labels == cid] = True
            n_stripe += 1
        elif is_block:
            n_block += 1
        else:
            keep[labels == cid] = True

    return (
        keep,
        {
            "n_components": max(0, n_components - 1),
            "n_stripe": n_stripe,
            "n_block": n_block,
            "n_wall_orient": n_wall_orient,
        },
    )


def _pca_over_pixels(
    ys_px: np.ndarray,
    xs_px: np.ndarray,
    *,
    px_to_m: float,
) -> Tuple[
    Tuple[float, float],
    np.ndarray,
    np.ndarray,
    float,
    float,
    Tuple[float, float],
]:
    """Run PCA over (y, x) pixel positions and return metric geometry.

    Returns
        centroid_m, long_axis_unit, short_axis_unit, length_m, width_m,
        centroid_px.
    """
    pts_px = np.column_stack((xs_px, ys_px)).astype(np.float64)
    centroid_px = pts_px.mean(axis=0)
    centred = pts_px - centroid_px
    cov = np.cov(centred, rowvar=False)
    if not np.all(np.isfinite(cov)):
        raise PixelExtractionUnavailable("covariance has NaN/Inf")
    eigvals, eigvecs = np.linalg.eigh(cov)
    if not np.all(np.isfinite(eigvals)):
        raise PixelExtractionUnavailable("eigvals NaN/Inf")
    long_axis = eigvecs[:, -1]
    if long_axis[0] < 0 or (long_axis[0] == 0 and long_axis[1] < 0):
        long_axis = -long_axis
    short_axis = np.array([-long_axis[1], long_axis[0]], dtype=np.float64)
    proj_long = centred @ long_axis
    proj_short = centred @ short_axis
    length_px = float(proj_long.max() - proj_long.min())
    width_px = float(proj_short.max() - proj_short.min())
    centroid_m = (
        float(centroid_px[0]) * px_to_m,
        float(centroid_px[1]) * px_to_m,
    )
    return (
        centroid_m,
        long_axis.astype(np.float64),
        short_axis.astype(np.float64),
        length_px * px_to_m,
        width_px * px_to_m,
        (float(centroid_px[0]), float(centroid_px[1])),
    )


def _tread_density_profile(
    image_gray: np.ndarray,
    region_mask: np.ndarray,
    centroid_px: Tuple[float, float],
    long_axis: np.ndarray,
    short_axis: np.ndarray,
    length_px: float,
    width_px: float,
    *,
    dark_threshold: int,
    n_samples: int = 200,
) -> np.ndarray:
    """Sample dark-pixel density along the long axis."""
    H, W = image_gray.shape[:2]
    ts = np.linspace(-0.5 * length_px, 0.5 * length_px, n_samples)
    half_w = 0.35 * width_px
    short_samples = np.linspace(-half_w, half_w, 7)
    cx, cy = centroid_px
    profile = np.zeros(n_samples, dtype=np.float64)
    for i, t in enumerate(ts):
        dark = 0
        total = 0
        for s in short_samples:
            x = cx + t * long_axis[0] + s * short_axis[0]
            y = cy + t * long_axis[1] + s * short_axis[1]
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < W and 0 <= iy < H and region_mask[iy, ix] > 0:
                if int(image_gray[iy, ix]) < dark_threshold:
                    dark += 1
                total += 1
        profile[i] = (dark / total) if total > 0 else 0.0
    return profile


def _detect_treads_from_profile(
    profile: np.ndarray,
    *,
    min_separation: int = 5,
    rel_height: float = 0.5,
    edge_margin: int = 6,
) -> Tuple[Optional[int], Optional[float], Optional[List[int]]]:
    """Count peaks in the density profile -> tread count.

    A peak is a local maximum whose density exceeds
    ``profile.min() + rel_height * (profile.max() - profile.min())``.
    Peaks must be at least ``min_separation`` samples apart and at
    least ``edge_margin`` samples away from either end of the profile
    (so the polygon-edge ink streak doesn't masquerade as a tread).

    Returns ``(n_peaks, mean_spacing_in_samples, peak_indices)`` or
    ``(None, None, None)`` when the profile is too flat to extract peaks.
    ``peak_indices`` are the sample-frame positions of the detected
    treads (0 = low end, ``profile.size - 1`` = high end).
    """
    if profile.size == 0:
        return None, None, None
    p_max = float(profile.max())
    p_min = float(profile.min())
    if p_max - p_min < 0.05:
        return None, None, None
    threshold = p_min + rel_height * (p_max - p_min)
    peaks: List[int] = []
    last_peak = -10 * min_separation
    lo_bound = max(1, edge_margin)
    hi_bound = max(lo_bound + 1, len(profile) - 1 - edge_margin)
    for i in range(lo_bound, hi_bound):
        v = profile[i]
        if (
            v >= threshold
            and v >= profile[i - 1]
            and v >= profile[i + 1]
            and (i - last_peak) >= min_separation
        ):
            peaks.append(i)
            last_peak = i
    if len(peaks) < 2:
        return None, None, None
    spacing_samples = float(np.mean(np.diff(peaks)))
    return len(peaks), spacing_samples, peaks


def aggregate_n_steps(
    *,
    voter_a_density_peak: Optional[int],
    length_m: float,
    going_target_m: float = DEFAULT_TARGET_GOING_M,
) -> Tuple[Optional[int], float, bool, int]:
    """2-Voter aggregator for tread count.

    Voter A — Density-Peak (Pixel-Evidence). Strongest authority when
    available. Confidence equals ``VOTER_A_BASE_CONFIDENCE``.

    Voter B — DIN-Target (length_m / 0.29 rounded). Always available
    as a sanity-check. Confidence ``VOTER_B_BASE_CONFIDENCE`` so it
    does not overrule a clear pixel-derived count, but it pulls the
    aggregator out of degenerate cases.

    Aggregation rule:
        * If only B is available -> n_steps = B, confidence = B_conf,
          low_confidence flag = True (we lack pixel evidence).
        * If both available and within tolerance -> n_steps = A,
          confidence = max(A_conf, B_conf), low_confidence = False.
        * If both available but disagree by > ``VOTER_DISAGREEMENT_PCT``
          -> n_steps = A (pixel-evidence wins), low_confidence = True.
        * Voter B alone (A None) on a degenerate length_m -> None.

    Args:
        voter_a_density_peak: Tread count from the density profile or
            ``None`` if the profile was too flat.
        length_m: Footprint extent along the long axis (m).
        going_target_m: DIN comfort going (default 0.29 m).

    Returns:
        ``(n_steps, confidence, low_confidence_flag, voter_b_din_target)``.
        The returned ``voter_b_din_target`` is for diagnostics so the
        caller can record both voters even if only one drove n_steps.
    """
    if length_m <= 0.0 or going_target_m <= 0.0:
        return None, 0.0, True, 0

    voter_b = max(1, int(round(length_m / going_target_m)))

    if voter_a_density_peak is None:
        return voter_b, VOTER_B_BASE_CONFIDENCE, True, voter_b

    if voter_a_density_peak <= 0:
        return voter_b, VOTER_B_BASE_CONFIDENCE, True, voter_b

    spread = abs(voter_a_density_peak - voter_b) / max(voter_b, 1)
    if spread > VOTER_DISAGREEMENT_PCT:
        return (
            voter_a_density_peak,
            VOTER_A_BASE_CONFIDENCE,
            True,
            voter_b,
        )
    return (
        voter_a_density_peak,
        max(VOTER_A_BASE_CONFIDENCE, VOTER_B_BASE_CONFIDENCE),
        False,
        voter_b,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_stair_pixels(
    image_gray: Optional[np.ndarray],
    user_polygon_px: Sequence[Tuple[float, float]],
    *,
    px_to_m: float,
    going_target_m: float = DEFAULT_TARGET_GOING_M,
    profile_n_samples: int = 200,
    clipped_polygon_px: Optional[
        Sequence[Sequence[Tuple[float, float]]]
    ] = None,
) -> Tuple[StairFootprint, StairPixelDetails]:
    """Extract stair footprint + tread layout from rendered ink.

    Args:
        image_gray: rendered plan grayscale image (uint8). ``None`` ->
            raises ``PixelExtractionUnavailable``.
        user_polygon_px: user bbox/polygon vertices in render-image
            pixel coordinates (already scaled with ``sx, sy``).
        px_to_m: metre-per-pixel scale of ``image_gray``.
        going_target_m: DIN comfort going used by the n_steps Voter B.
        profile_n_samples: density-profile resolution (default 200 ticks
            along the long axis).
        clipped_polygon_px: optional refined ROI. When provided this
            is rasterised instead of ``user_polygon_px`` — the
            difference being that the clipped polygon has already had
            wall regions subtracted out (Tag 9 Walls-as-Boundary).
            The polygon may contain holes; pass it as a list of
            rings ``[exterior, hole1, hole2, ...]`` where each ring
            is a sequence of ``(x, y)`` tuples in render pixels.

    Returns:
        Tuple of ``(footprint, details)``. The footprint is a regular
        ``StairFootprint`` so the pipeline can use it identically to
        the polygon-PCA fallback result. The ``details`` carries the
        density profile + voter outputs for Tag 3 (direction-up) and
        diagnostics.

    Raises:
        PixelExtractionUnavailable: when extraction is not possible.
    """
    if image_gray is None or not isinstance(image_gray, np.ndarray):
        raise PixelExtractionUnavailable("image_gray missing")
    if px_to_m <= 0.0:
        raise PixelExtractionUnavailable("px_to_m must be positive")
    if len(user_polygon_px) < 3:
        raise PixelExtractionUnavailable("user polygon < 3 points")

    if clipped_polygon_px:
        region_mask = _polygon_with_holes_mask_px(
            clipped_polygon_px, image_gray.shape[:2],
        )
    else:
        region_mask = _polygon_mask_px(
            user_polygon_px, image_gray.shape[:2],
        )
    if region_mask.sum() == 0:
        raise PixelExtractionUnavailable("polygon rasterises to empty mask")

    # Adaptive threshold computed from ROI intensities only — the page
    # background outside the polygon must not influence Otsu.
    roi_intensities = image_gray[region_mask > 0]
    dark_threshold, threshold_source = compute_adaptive_threshold(roi_intensities)

    dark = (image_gray < dark_threshold) & (region_mask > 0)
    n_dark = int(dark.sum())
    if n_dark < MIN_DARK_PIXELS:
        raise PixelExtractionUnavailable(
            f"too few dark pixels under user region "
            f"({n_dark} < {MIN_DARK_PIXELS})"
        )

    # User-bbox cardinal long axis hint for the orientation-aware
    # component filter: the longer edge of the axis-aligned user
    # polygon bbox tells us the walking direction. We use it to drop
    # stripe components that run PARALLEL to that direction (= wall
    # ink along the bbox edge). Without this hint, the L-shape's
    # bbox-edge walls fool the classifier into accepting them as treads
    # because they pass the aspect-ratio gate.
    bbox_long_hint: Optional[Tuple[float, float]] = None
    poly_arr = np.asarray(user_polygon_px, dtype=np.float64)
    if poly_arr.ndim == 2 and poly_arr.shape[1] == 2 and len(poly_arr) >= 3:
        extent = poly_arr.max(axis=0) - poly_arr.min(axis=0)
        if extent[0] >= extent[1]:
            bbox_long_hint = (1.0, 0.0)
        else:
            bbox_long_hint = (0.0, 1.0)

    # Connected-components filter: drop "block"-shaped components (wall
    # stubs, columns intruding into the user bbox) so the footprint PCA
    # only sees tread lines + arrow shaft. Falls back to the unfiltered
    # mask if filtering drops below MIN_DARK_PIXELS (e.g. on a stair
    # drawn as a solid filled rectangle).
    filtered_mask, cc_stats = _classify_components_stripe_vs_block(
        (dark.astype(np.uint8) * 255),
        stair_long_axis_hint=bbox_long_hint,
    )
    # Convert numpy bool -> python bool for clean serialisation and
    # ``is True / is False`` checks in tests.
    used_component_filter = bool(filtered_mask.sum() >= MIN_DARK_PIXELS)
    if used_component_filter:
        dark_pca = filtered_mask
        n_dark_pca = int(filtered_mask.sum())
    else:
        dark_pca = dark
        n_dark_pca = n_dark

    ys_px, xs_px = np.where(dark_pca)
    centroid_m, long_axis, short_axis, length_m, width_m, centroid_px = (
        _pca_over_pixels(ys_px, xs_px, px_to_m=px_to_m)
    )
    if length_m < 0.20 or width_m < 0.20:
        raise PixelExtractionUnavailable(
            f"pixel-extracted footprint too small "
            f"({length_m:.2f}x{width_m:.2f} m)"
        )

    length_px = length_m / px_to_m
    width_px = width_m / px_to_m
    # Erode the region mask inward by ~2 cm so wall ink that sits
    # exactly on the user-bbox edge does not pollute the density
    # profile. The component-filter already drops parallel-to-axis
    # walls, but a wall running PERPENDICULAR to the long axis (and
    # therefore aligned with the tread direction) at the bbox edge
    # would otherwise produce a false peak right at the start/end of
    # the profile. 2 cm of erosion preserves real treads (which are
    # typically 5-15 cm from any edge) while killing the bbox-edge ink.
    import cv2  # noqa: PLC0415
    erode_px = max(1, int(round(0.02 / max(px_to_m, 1e-9))))
    erode_kernel_size = max(3, erode_px * 2 + 1)
    if erode_kernel_size % 2 == 0:
        erode_kernel_size += 1
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (erode_kernel_size, erode_kernel_size),
    )
    profile_mask = cv2.erode(region_mask, erode_kernel, iterations=1)
    if int(profile_mask.sum()) == 0:
        # Erosion ate everything — fall back to the un-eroded mask.
        profile_mask = region_mask
    profile = _tread_density_profile(
        image_gray, profile_mask,
        centroid_px=centroid_px,
        long_axis=long_axis, short_axis=short_axis,
        length_px=length_px, width_px=width_px,
        dark_threshold=dark_threshold,
        n_samples=profile_n_samples,
    )
    n_peaks, mean_spacing_samples, peak_indices = _detect_treads_from_profile(profile)
    n_steps, confidence, low_conf, voter_b = aggregate_n_steps(
        voter_a_density_peak=n_peaks,
        length_m=length_m,
        going_target_m=going_target_m,
    )

    going_m: Optional[float] = None
    if n_peaks is not None and mean_spacing_samples and length_m > 0.0:
        going_m = float(length_m * mean_spacing_samples / profile.size)
    elif n_steps is not None and n_steps > 0 and length_m > 0.0:
        going_m = length_m / n_steps

    # Convert peak sample-indices into absolute plan-metre positions.
    # The profile is sampled uniformly from -L/2 to +L/2 along the
    # pixel-PCA long_axis, anchored at centroid_m. The absolute position
    # of a peak at sample index i is therefore:
    #   pos_xy = centroid_m + ((i / (N-1)) - 0.5) * length_m * long_axis_unit
    # Storing this absolute form keeps the peaks pinned to the actual
    # ink even if downstream code re-anchors the footprint to a
    # different axis (e.g. user-bbox snap).
    tread_positions_xy_m: Optional[List[Tuple[float, float]]] = None
    if peak_indices is not None and length_m > 0.0 and profile.size > 1:
        N = float(profile.size)
        long_unit = (float(long_axis[0]), float(long_axis[1]))
        tread_positions_xy_m = []
        for i in peak_indices:
            t = ((i / (N - 1.0)) - 0.5) * length_m
            tread_positions_xy_m.append((
                centroid_m[0] + t * long_unit[0],
                centroid_m[1] + t * long_unit[1],
            ))

    footprint = StairFootprint(
        centroid_m=centroid_m,
        long_axis=(float(long_axis[0]), float(long_axis[1])),
        short_axis=(float(short_axis[0]), float(short_axis[1])),
        length_m=length_m,
        width_m=width_m,
    )
    details = StairPixelDetails(
        n_dark_pixels=n_dark,
        dark_threshold_used=int(dark_threshold),
        threshold_source=threshold_source,
        voter_a_density_peak=n_peaks,
        voter_b_din_target=voter_b,
        voter_a_confidence=(VOTER_A_BASE_CONFIDENCE if n_peaks is not None else 0.0),
        voter_b_confidence=VOTER_B_BASE_CONFIDENCE,
        n_steps=n_steps,
        detected_going_m=going_m,
        low_confidence_tread_count=low_conf,
        density_profile=profile,
        profile_n_samples=profile.size,
        tread_positions_xy_m=tread_positions_xy_m,
        long_axis_px=(float(long_axis[0]), float(long_axis[1])),
        length_px=length_px,
        n_components=int(cc_stats["n_components"]),
        n_stripe_components=int(cc_stats["n_stripe"]),
        n_block_components=int(cc_stats["n_block"]),
        used_component_filter=used_component_filter,
        notes=[
            f"pixel-extracted: {n_dark_pca} dark pixels "
            f"(of {n_dark} raw, filter={used_component_filter}), "
            f"L={length_m:.3f}m, W={width_m:.3f}m",
            f"threshold={int(dark_threshold)} ({threshold_source})",
            f"components: {cc_stats['n_components']} total, "
            f"{cc_stats['n_stripe']} stripe, {cc_stats['n_block']} block",
        ],
    )
    if n_peaks is not None:
        details.notes.append(
            f"voter_a_density_peak={n_peaks}, voter_b_din={voter_b}, "
            f"n_steps={n_steps}, conf={confidence:.2f}"
        )
    else:
        details.notes.append(
            f"voter_a unavailable, fallback to voter_b_din={voter_b}"
        )
    if low_conf:
        details.notes.append("LOW_CONFIDENCE_TREAD_COUNT")
    return footprint, details
