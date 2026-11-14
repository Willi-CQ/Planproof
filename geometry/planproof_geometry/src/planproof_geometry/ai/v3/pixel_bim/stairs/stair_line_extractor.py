"""Hough-line-based tread / stringer / cut-line detection (Sprint v2).

The legacy ``stair_pixel_extractor`` uses a 1-D density profile along a
PCA-derived long axis to count treads. That works for pristine straight
runs but falls over when:

    * the labelled bbox is sloppy and includes wall ink (the PCA picks
      a diagonal even after the component filter),
    * the architect drew the treads at an oblique angle to the bbox
      (rare in CAD plans, common in scanned drawings), or
    * the profile is too short relative to the visible flight (L-shaped
      stairs where the labelled bbox extends across the landing).

This module treats tread detection as a **line-detection** problem
rather than a density problem:

    1. Build a binary ink mask inside a dilated ROI around the user
       polygon (analog to the door / window pattern — 20 cm by default).
    2. Run ``cv2.HoughLinesP`` over the Canny edges of that mask. Hough
       parameters scale with ``px_to_m`` so the algorithm is
       resolution-invariant.
    3. Classify each segment as a TREAD candidate (perpendicular to
       the bbox long axis ±15°) or a STRINGER candidate (parallel to
       the bbox long axis ±10°). Everything else is noise.
    4. Cluster tread candidates along the bbox long axis (1-D DBSCAN
       with ``eps ≈ 4 cm``). Cluster centroids = tread positions.
    5. Detect stringers: pick the two longest parallel candidates that
       sit on opposite short-axis sides of the centroid.
    6. Detect cut lines: the bbox-end-most thick perpendicular line
       (typically a noticeably bolder ink stroke).

The output is consumed by ``pipeline.py`` as Voter #1 for n_steps and
as the geometry source for stair width (stringer distance) and
flight endpoints (cut-line replaces tread-cluster extremum).

The module does NOT raise — it always returns a ``StairLineDetails``
with the fields it could fill in. The pipeline's existing density-based
extractor stays as a fallback for plans where Hough fails (very pale
ink, hatched treads, scanned blueprints with high noise).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from planproof_geometry.ai.cv_compat import hough_lines_p


# ROI margin around the user polygon. The Hough detector uses this as
# a hard cap for ink sampling (performance), but the actual
# *attribution* of every line uses the confidence-field weight. 50 cm
# is enough headroom for sloppy labels without flooding the detector
# with noise from adjacent rooms.
STAIR_ROI_MARGIN_M = 0.50

# Hough parameters scaled by px_to_m. minLineLength is expressed as a
# fraction of the bbox short edge (treads span ≥60 % of the run width
# in any honest CAD plan).
HOUGH_MIN_LINE_LENGTH_FRAC = 0.55
HOUGH_MAX_LINE_GAP_M = 0.04
HOUGH_THRESHOLD_BASE_PX = 40

# Confidence floor for a line segment to be considered as evidence.
# Computed as the mean confidence-field value along the segment. Lines
# whose mean confidence is below this floor are discarded (they belong
# to a different label or to free space). 0.08 keeps stringer ink that
# sits right on the bbox edge (where the confidence field is still
# bumping along ~0.10–0.15 from the exponential falloff) while still
# rejecting lines that drift > ~80 cm beyond the label.
LINE_CONFIDENCE_FLOOR = 0.08

# Angle tolerances for orientation-based line classification.
# Tread lines run perpendicular to the walking direction; stringers run
# parallel to it. ±15° / ±10° cover hand-drawn imperfections without
# letting diagonal noise leak across.
TREAD_PERPENDICULAR_TOL_DEG = 15.0
STRINGER_PARALLEL_TOL_DEG = 12.0

# Tread cluster eps (1-D DBSCAN along the bbox long axis). Going values
# at the DIN floor (23 cm) are well above this, so 4 cm safely merges
# two Hough segments that describe the same physical tread without
# fusing adjacent treads.
TREAD_CLUSTER_EPS_M = 0.04

# Stringer minimum length as a fraction of the bbox long edge. A line
# below this threshold is too short to be a real wangen. 0.15 keeps us
# robust against Canny-fragmented wall strokes — a single 2 m wangen
# often produces 3–5 shorter Hough segments because edge detection cuts
# at every density discontinuity.
STRINGER_MIN_LENGTH_FRAC = 0.15

# Cross-riser detection (winder / quarter-turn stairs). Cross-risers are
# the step fronts inside the turning wing: they run PARALLEL to the
# long axis (perpendicular to the main risers) and are typically SHORT
# and nested. They are detected by a dedicated second Hough pass with a
# lower minLineLength so the short nested strokes survive, then
# clustered along the SHORT axis. The main pass (treads + stringers) is
# untouched so existing straight/L behaviour is byte-for-byte stable.
#
# minLineLength as a fraction of the SMALLER bbox dimension (NOT just
# the run width) — a winder wing's cross-risers are short relative to
# the full run, so anchoring on the run width would miss them.
CROSS_RISER_MIN_LENGTH_FRAC = 0.18
# Cluster eps along the short axis. Cross-riser spacing (~37 cm on the
# Matterport sample) is well above this, so 5 cm merges the Hough
# fragments of one physical cross-riser without fusing neighbours.
CROSS_RISER_CLUSTER_EPS_M = 0.05
# Hard pixel criterion: a clustered cross-riser must lie this fraction
# on real dark ink or it is dropped (Hough ghost / Canny artefact).
CROSS_RISER_MIN_INK_RATIO = 0.90

# Cut-line detection. A cut line is a noticeably THICKER tread-direction
# line at one of the flight ends, drawn to mark the Höhensprung.
# We approximate "thicker" via local dilation count rather than raw
# stroke width measurement.
CUTLINE_THICKNESS_PROBE_PX_M = 0.006  # 6 mm — distinguishes 2 mm tread
                                      # ink from 5+ mm cut-line ink

# Minimum number of dark pixels under the ROI mask. Below this the
# Hough-based detector bails out — the density extractor will provide
# the result.
MIN_ROI_INK_PIXELS = 80


@dataclass
class StairLineDetails:
    """Side-channel output of the Hough-based extractor.

    Geometric fields are in **plan-XY metres**, NOT pixels. Pipeline
    consumers compose these directly with the pipeline's per-step
    slicing and BIM-footprint reconstruction.
    """

    # The cardinal long-axis unit vector derived from the user bbox.
    # (1, 0) for landscape bboxes, (0, 1) for portrait. The detector
    # always quotes its output relative to this axis so the pipeline can
    # plug the result straight into the same per-step slicer that other
    # voters use.
    bbox_long_unit: Tuple[float, float] = (1.0, 0.0)
    bbox_short_unit: Tuple[float, float] = (0.0, 1.0)
    bbox_long_extent_m: float = 0.0
    bbox_short_extent_m: float = 0.0

    # Tread-cluster centroids (plan-XY in metres). One point per
    # detected tread. Empty when Hough found no perpendicular lines.
    tread_centroids_xy_m: List[Tuple[float, float]] = field(default_factory=list)
    # Representative chord per cluster (median of the cluster's Hough
    # segments). Used by the diagnostic overlay to draw the actual ink
    # line, not just a synthetic chord across the polygon.
    tread_lines_xy_m: List[List[Tuple[float, float]]] = field(default_factory=list)

    # Detected stringer pair (left & right wangen). Each entry is a
    # two-point line in plan-metres. Empty when the run is open and no
    # stringers were drawn.
    stringer_lines_xy_m: List[List[Tuple[float, float]]] = field(
        default_factory=list,
    )
    stringer_distance_m: Optional[float] = None  # measured run width

    # Detected cut lines (zero, one, or two — one at each flight end).
    cutlines_xy_m: List[List[Tuple[float, float]]] = field(default_factory=list)

    # All clustered cross-risers (winder step fronts), parallel to the
    # long axis, in plan-XY metres. Each entry is a two-point line at the
    # cluster's real ink extent (no synthetic extension), pixel-validated
    # to lie >= CROSS_RISER_MIN_INK_RATIO on dark ink. Empty for straight
    # runs. Consumed by the polygonize step-decomposition in pipeline.py
    # to carve winder wings into L-treads. Distinct from
    # ``stringer_lines_xy_m`` (the two longest outer wangen): this field
    # exposes EVERY parallel riser, including the short nested ones a
    # winder needs.
    cross_riser_lines_xy_m: List[List[Tuple[float, float]]] = field(
        default_factory=list,
    )

    # Raw stats for diagnostics.
    n_hough_lines: int = 0
    n_tread_candidates: int = 0
    n_stringer_candidates: int = 0
    n_cutline_candidates: int = 0
    dark_threshold_used: int = 0

    # Confidence-weighted aggregates. ``mean_tread_line_confidence`` is
    # the mean of every tread candidate's mean line-confidence; a value
    # ≥0.7 means most ink the voter saw lay deep inside the user
    # bbox, ≤0.3 means most of the evidence came from sloppy-extension
    # territory.
    mean_tread_line_confidence: float = 0.0
    mean_stringer_line_confidence: float = 0.0
    # Bayesian-style evidence weight for the stair Long-Axis: the
    # stringer-pair "Authority" the pipeline uses to decide whether to
    # override the Bbox-Cardinal-prior. Computed as
    #   min(total_stringer_length / bbox_long_extent, 1.0)
    #   × mean_stringer_line_confidence
    # Range [0, 1]; 0 means no stringers were detected.
    stringer_axis_authority: float = 0.0
    # Detected long-axis from the stringer pair, in plan-XY metres
    # (unit vector). (0, 0) when no stringers were detected.
    stringer_long_axis: Tuple[float, float] = (0.0, 0.0)

    # Status / failure mode. ``"ok"`` means n_steps from this detector
    # is usable. Other values give the pipeline a hint why it should
    # fall back to the density extractor.
    status: str = "ok"
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bbox_axes_from_polygon_px(
    polygon_px: Sequence[Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float], float, float]:
    """Return (long_unit_xy, short_unit_xy, long_extent_px, short_extent_px).

    Axis-aligned cardinal selection: the bbox's longer edge wins. Mirrors
    the door/window convention where the user always draws axis-aligned
    bboxes in the label editor.
    """
    arr = np.asarray(polygon_px, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
        return (1.0, 0.0), (0.0, 1.0), 0.0, 0.0
    extent = arr.max(axis=0) - arr.min(axis=0)
    ex, ey = float(extent[0]), float(extent[1])
    if ex >= ey:
        return (1.0, 0.0), (0.0, 1.0), ex, ey
    return (0.0, 1.0), (-1.0, 0.0), ey, ex


def _dilate_polygon_mask(
    polygon_px: Sequence[Tuple[float, float]],
    image_shape_hw: Tuple[int, int],
    *,
    expand_px: float,
) -> np.ndarray:
    """Rasterise polygon, dilate by ``expand_px``, return uint8 mask."""
    import cv2  # noqa: PLC0415

    H, W = image_shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.array(
        [[int(round(x)), int(round(y))] for x, y in polygon_px],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)

    if expand_px > 0:
        kernel_size = max(3, int(round(expand_px)) * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size),
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _adaptive_ink_threshold(roi_intensities: np.ndarray) -> Tuple[int, str]:
    """Reuse the same Otsu/percentile logic as the density extractor."""
    from .stair_pixel_extractor import compute_adaptive_threshold  # noqa: PLC0415

    return compute_adaptive_threshold(roi_intensities)


def _hough_lines(
    binary_mask: np.ndarray,
    *,
    min_line_length_px: int,
    max_line_gap_px: int,
    threshold: int,
) -> np.ndarray:
    """Run probabilistic Hough on a binary ink mask, return Nx4 endpoints.

    Canny edges are computed first so thick CAD strokes do not produce
    multiple parallel lines per stroke. Empty array on failure.
    """
    import cv2  # noqa: PLC0415

    # Canny on the binary mask gives us thin one-pixel edges of every
    # stroke contour. Hough then picks lines from these edges.
    edges = cv2.Canny(binary_mask, 50, 150, apertureSize=3, L2gradient=True)
    # (N, 4) in OpenCV 4 wie 5 - siehe planproof_geometry.ai.cv_compat.
    return hough_lines_p(
        edges,
        rho=1.0,
        theta=np.pi / 360.0,  # 0.5° resolution
        threshold=int(threshold),
        minLineLength=int(min_line_length_px),
        maxLineGap=int(max_line_gap_px),
    )


def _line_orientation_to_axis_deg(
    line_xyxy: np.ndarray,
    axis_unit: Tuple[float, float],
) -> float:
    """Return the absolute angle (0-90°) between a line and an axis."""
    x0, y0, x1, y1 = line_xyxy
    dx, dy = float(x1 - x0), float(y1 - y0)
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return 90.0
    line_unit = (dx / norm, dy / norm)
    cos_a = abs(line_unit[0] * axis_unit[0] + line_unit[1] * axis_unit[1])
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


def _line_length_px(line_xyxy: np.ndarray) -> float:
    x0, y0, x1, y1 = line_xyxy
    return math.hypot(float(x1 - x0), float(y1 - y0))


def _project_point_onto_axis(
    point_xy_px: Tuple[float, float],
    *,
    origin_xy_px: Tuple[float, float],
    axis_unit: Tuple[float, float],
) -> float:
    """Signed projection of a point onto an axis through origin."""
    dx = point_xy_px[0] - origin_xy_px[0]
    dy = point_xy_px[1] - origin_xy_px[1]
    return dx * axis_unit[0] + dy * axis_unit[1]


def _line_midpoint_px(line_xyxy: np.ndarray) -> Tuple[float, float]:
    x0, y0, x1, y1 = line_xyxy
    return (0.5 * (float(x0) + float(x1)), 0.5 * (float(y0) + float(y1)))


def _line_mean_confidence(
    line_xyxy: np.ndarray,
    confidence_field,
    *,
    n_samples: int = 24,
) -> float:
    """Mean confidence value sampled uniformly along a line segment.

    Accepts either a :class:`ConfidenceField` (sub-window form) or a
    raw ``np.ndarray`` (full-image form, for backward compatibility).
    Sampling at 24 points is enough to detect a segment that drifts
    from a high-confidence region into a low-confidence one.
    """
    if confidence_field is None:
        return 1.0
    # Sub-window form has a dedicated sampler that subtracts the offset.
    if hasattr(confidence_field, "sample_along_line"):
        x0, y0, x1, y1 = (float(v) for v in line_xyxy)
        return float(
            confidence_field.sample_along_line(
                x0, y0, x1, y1, n_samples=n_samples,
            )
        )
    H, W = confidence_field.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in line_xyxy)
    if n_samples < 2:
        n_samples = 2
    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    xs = x0 + (x1 - x0) * ts
    ys = y0 + (y1 - y0) * ts
    ix = np.clip(np.round(xs).astype(np.int64), 0, W - 1)
    iy = np.clip(np.round(ys).astype(np.int64), 0, H - 1)
    return float(confidence_field[iy, ix].mean())


def _line_ink_ratio_px(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    image_gray: np.ndarray,
    dark_threshold: int,
    *,
    n_samples: int = 80,
    neighborhood_px: int = 2,
) -> float:
    """Fraction of sample points along a pixel-space segment whose
    ``neighborhood_px`` neighbourhood contains dark ink.

    Mirrors ``pipeline._line_pixel_ratio_m`` but takes pixel endpoints
    directly. Used to validate clustered cross-risers against the plan
    ink so Hough ghosts never become step fronts.
    """
    H, W = image_gray.shape[:2]
    ts = np.linspace(0.0, 1.0, max(2, n_samples), dtype=np.float64)
    xs = p0[0] + (p1[0] - p0[0]) * ts
    ys = p0[1] + (p1[1] - p0[1]) * ts
    nbr = max(0, neighborhood_px)
    n_dark = n_total = 0
    for xf, yf in zip(xs, ys):
        ix = int(round(xf))
        iy = int(round(yf))
        if not (0 <= ix < W and 0 <= iy < H):
            continue
        n_total += 1
        block = image_gray[
            max(0, iy - nbr):iy + nbr + 1, max(0, ix - nbr):ix + nbr + 1
        ]
        if (block < dark_threshold).any():
            n_dark += 1
    return n_dark / n_total if n_total else 0.0


def _cluster_1d(
    samples: List[Tuple[float, int]],
    *,
    eps: float,
) -> List[List[int]]:
    """Trivial 1-D DBSCAN-by-sort: group samples whose sorted neighbour
    distance ≤ eps.

    ``samples`` is a list of ``(value, original_index)``. Returns lists
    of original indices, one per cluster, sorted by cluster centroid.
    """
    if not samples:
        return []
    samples_sorted = sorted(samples, key=lambda kv: kv[0])
    clusters: List[List[int]] = []
    current: List[int] = [samples_sorted[0][1]]
    current_values: List[float] = [samples_sorted[0][0]]
    for v, idx in samples_sorted[1:]:
        if v - current_values[-1] <= eps:
            current.append(idx)
            current_values.append(v)
        else:
            clusters.append(current)
            current = [idx]
            current_values = [v]
    clusters.append(current)
    return clusters


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_stair_lines(
    image_gray: Optional[np.ndarray],
    user_polygon_px: Sequence[Tuple[float, float]],
    *,
    px_to_m: float,
    roi_margin_m: float = STAIR_ROI_MARGIN_M,
    confidence_field: Optional[np.ndarray] = None,
) -> StairLineDetails:
    """Run the Hough-based tread/stringer/cut-line detector.

    Args:
        image_gray: rendered grayscale plan (uint8). ``None`` returns a
            details object with ``status="image_missing"``.
        user_polygon_px: vertices of the labelled bbox in render-px.
        px_to_m: metre-per-pixel scale of ``image_gray``.
        roi_margin_m: dilation of the user polygon (m) used for the
            *ink-sampling* mask. Default 1 m so the Hough detector can
            see plan ink that escaped a sloppy human label. The actual
            attribution of each line uses the per-pixel
            ``confidence_field`` — the ROI is just a performance cap.
        confidence_field: optional float32 ``(H, W)`` per-pixel
            confidence map produced by
            :mod:`stair_confidence_field`. When provided, every
            Hough segment is scored by its mean confidence along the
            segment; segments below
            :data:`LINE_CONFIDENCE_FLOOR` are discarded.

    Returns:
        Populated :class:`StairLineDetails`. The pipeline checks
        ``status`` and ``tread_centroids_xy_m`` to decide whether to
        prefer this voter or to fall back to the density extractor.
    """
    details = StairLineDetails()
    if image_gray is None or not isinstance(image_gray, np.ndarray):
        details.status = "image_missing"
        details.notes.append("image_gray is None — Hough extractor skipped")
        return details
    if px_to_m <= 0.0:
        details.status = "bad_scale"
        details.notes.append("px_to_m <= 0")
        return details
    if len(user_polygon_px) < 3:
        details.status = "bad_polygon"
        details.notes.append("user polygon < 3 points")
        return details

    # 1. Bbox axes (cardinal). All downstream metrics are quoted in this
    # frame so the pipeline can splice the result directly into the per-
    # step slicer without further re-projection.
    long_unit, short_unit, long_ext_px, short_ext_px = (
        _bbox_axes_from_polygon_px(user_polygon_px)
    )
    details.bbox_long_unit = long_unit
    details.bbox_short_unit = short_unit
    details.bbox_long_extent_m = float(long_ext_px * px_to_m)
    details.bbox_short_extent_m = float(short_ext_px * px_to_m)

    if long_ext_px <= 1.0 or short_ext_px <= 1.0:
        details.status = "bbox_degenerate"
        details.notes.append("user bbox extent too small")
        return details

    # 2. ROI mask = dilated polygon. Doors/windows use 5 mm to 1.5 m
    # depending on the detector; for stairs 20 cm is the right scale to
    # catch tread ink that escapes a sloppy label without including
    # adjacent walls of the next room.
    expand_px = max(0.0, roi_margin_m / max(px_to_m, 1e-9))
    region_mask = _dilate_polygon_mask(
        user_polygon_px, image_gray.shape[:2], expand_px=expand_px,
    )
    if int(region_mask.sum()) == 0:
        details.status = "empty_mask"
        details.notes.append("ROI rasterises to empty mask")
        return details

    # 3. Adaptive ink threshold (reuse the density extractor's logic so
    # that Hough and density paths agree on what is "ink").
    roi_intensities = image_gray[region_mask > 0]
    threshold, threshold_source = _adaptive_ink_threshold(roi_intensities)
    details.dark_threshold_used = int(threshold)
    details.notes.append(f"ink_threshold={int(threshold)} ({threshold_source})")

    dark_mask = ((image_gray < threshold) & (region_mask > 0)).astype(np.uint8) * 255
    n_dark = int((dark_mask > 0).sum())
    if n_dark < MIN_ROI_INK_PIXELS:
        details.status = "too_few_ink_pixels"
        details.notes.append(
            f"only {n_dark} ink pixels under ROI (need {MIN_ROI_INK_PIXELS})"
        )
        return details

    # 4. Hough on Canny edges.
    min_line_length_px = max(
        6, int(round(short_ext_px * HOUGH_MIN_LINE_LENGTH_FRAC)),
    )
    max_line_gap_px = max(
        2, int(round(HOUGH_MAX_LINE_GAP_M / max(px_to_m, 1e-9))),
    )
    threshold_px = HOUGH_THRESHOLD_BASE_PX
    lines_xyxy = _hough_lines(
        dark_mask,
        min_line_length_px=min_line_length_px,
        max_line_gap_px=max_line_gap_px,
        threshold=threshold_px,
    )
    details.n_hough_lines = int(lines_xyxy.shape[0])
    if details.n_hough_lines == 0:
        details.status = "no_lines"
        details.notes.append(
            f"HoughLinesP found no segments "
            f"(minLineLength={min_line_length_px}, maxGap={max_line_gap_px})"
        )
        return details

    # 4b. Per-segment confidence. When a confidence field is provided
    # we score every segment by its mean confidence along the line.
    # Segments below LINE_CONFIDENCE_FLOOR are discarded: that ink
    # belongs to a different label (a neighbour bbox is much closer)
    # or to free space far outside the labelled stair.
    line_confidences = np.ones(details.n_hough_lines, dtype=np.float64)
    if confidence_field is not None:
        for i in range(details.n_hough_lines):
            line_confidences[i] = _line_mean_confidence(
                lines_xyxy[i], confidence_field,
            )
        keep_mask = line_confidences >= LINE_CONFIDENCE_FLOOR
        n_dropped = int(details.n_hough_lines - int(keep_mask.sum()))
        if n_dropped > 0:
            details.notes.append(
                f"confidence_filter dropped {n_dropped}/"
                f"{details.n_hough_lines} segments (floor="
                f"{LINE_CONFIDENCE_FLOOR:.2f})"
            )

    def _conf_ok(i: int) -> bool:
        return bool(line_confidences[i] >= LINE_CONFIDENCE_FLOOR)

    # 5. Classify lines by orientation relative to the bbox long axis.
    tread_indices: List[int] = []
    stringer_indices: List[int] = []
    for i in range(details.n_hough_lines):
        if not _conf_ok(i):
            continue
        angle = _line_orientation_to_axis_deg(lines_xyxy[i], long_unit)
        # Perpendicular line → angle close to 90°. Parallel → close to 0°.
        if 90.0 - angle <= TREAD_PERPENDICULAR_TOL_DEG:
            tread_indices.append(i)
        elif angle <= STRINGER_PARALLEL_TOL_DEG:
            stringer_indices.append(i)
    details.n_tread_candidates = len(tread_indices)
    details.n_stringer_candidates = len(stringer_indices)

    if tread_indices:
        details.mean_tread_line_confidence = float(
            line_confidences[tread_indices].mean(),
        )
    if stringer_indices:
        details.mean_stringer_line_confidence = float(
            line_confidences[stringer_indices].mean(),
        )

    if not tread_indices:
        details.status = "no_perpendicular_lines"
        details.notes.append("no Hough segments perpendicular to bbox long axis")
        return details

    # 6. Cluster tread candidates along the bbox long axis. Origin =
    # min-corner of the user bbox so the projected coordinate is a
    # positive offset from "low end of bbox in long-axis direction".
    arr = np.asarray(user_polygon_px, dtype=np.float64)
    bbox_min = (float(arr[:, 0].min()), float(arr[:, 1].min()))
    bbox_max = (float(arr[:, 0].max()), float(arr[:, 1].max()))
    bbox_center_px = (
        0.5 * (bbox_min[0] + bbox_max[0]),
        0.5 * (bbox_min[1] + bbox_max[1]),
    )

    samples: List[Tuple[float, int]] = []
    for i in tread_indices:
        mid = _line_midpoint_px(lines_xyxy[i])
        s = _project_point_onto_axis(
            mid, origin_xy_px=bbox_min, axis_unit=long_unit,
        )
        samples.append((float(s) * px_to_m, i))

    clusters_idx = _cluster_1d(samples, eps=TREAD_CLUSTER_EPS_M)

    # Drop micro-clusters with < 30 % of the median cluster size. Single-
    # line clusters survive as long as they're not dwarfed by larger
    # ones; on faint plans the median cluster has size 1, so this only
    # kicks in when the median is solid (i.e. on noisy plans where some
    # spurious singletons leaked in).
    sizes = sorted(len(c) for c in clusters_idx)
    if sizes:
        median_size = sizes[len(sizes) // 2]
        threshold_size = max(1, int(0.3 * median_size))
        clusters_idx = [c for c in clusters_idx if len(c) >= threshold_size]

    # 7. Build per-cluster representatives: median midpoint and median
    # line endpoints in plan-metre coords.
    tread_centroids_xy_m: List[Tuple[float, float]] = []
    tread_lines_xy_m: List[List[Tuple[float, float]]] = []
    for cluster in clusters_idx:
        midpoints_px = [_line_midpoint_px(lines_xyxy[i]) for i in cluster]
        mids_arr = np.asarray(midpoints_px, dtype=np.float64)
        cluster_centroid_px = mids_arr.mean(axis=0)
        tread_centroids_xy_m.append((
            float(cluster_centroid_px[0]) * px_to_m,
            float(cluster_centroid_px[1]) * px_to_m,
        ))

        # Representative line = the longest segment in the cluster.
        longest_idx = max(cluster, key=lambda j: _line_length_px(lines_xyxy[j]))
        x0, y0, x1, y1 = lines_xyxy[longest_idx]
        tread_lines_xy_m.append([
            (float(x0) * px_to_m, float(y0) * px_to_m),
            (float(x1) * px_to_m, float(y1) * px_to_m),
        ])

    # Sort clusters by projection on the long axis so downstream
    # consumers can rely on monotonically increasing positions.
    def _proj_centroid_m(centroid: Tuple[float, float]) -> float:
        # Project the metre-frame centroid back onto bbox_min anchor.
        dx = centroid[0] / px_to_m - bbox_min[0]
        dy = centroid[1] / px_to_m - bbox_min[1]
        return (dx * long_unit[0] + dy * long_unit[1]) * px_to_m

    order = sorted(
        range(len(tread_centroids_xy_m)),
        key=lambda i: _proj_centroid_m(tread_centroids_xy_m[i]),
    )
    tread_centroids_xy_m = [tread_centroids_xy_m[i] for i in order]
    tread_lines_xy_m = [tread_lines_xy_m[i] for i in order]
    details.tread_centroids_xy_m = tread_centroids_xy_m
    details.tread_lines_xy_m = tread_lines_xy_m

    # 8. Stringer detection. Keep stringers that are at least
    # STRINGER_MIN_LENGTH_FRAC of the bbox long edge, group by short-
    # axis sign, take the longest from each side.
    stringer_min_length_px = STRINGER_MIN_LENGTH_FRAC * long_ext_px
    left_candidates: List[Tuple[float, int]] = []   # (length_px, idx)
    right_candidates: List[Tuple[float, int]] = []
    for i in stringer_indices:
        L = _line_length_px(lines_xyxy[i])
        if L < stringer_min_length_px:
            continue
        mid = _line_midpoint_px(lines_xyxy[i])
        s_short = _project_point_onto_axis(
            mid, origin_xy_px=bbox_center_px, axis_unit=short_unit,
        )
        if s_short >= 0:
            right_candidates.append((L, i))
        else:
            left_candidates.append((L, i))

    stringer_lines_xy_m: List[List[Tuple[float, float]]] = []
    if left_candidates and right_candidates:
        left_best = max(left_candidates, key=lambda kv: kv[0])[1]
        right_best = max(right_candidates, key=lambda kv: kv[0])[1]
        for li in (left_best, right_best):
            x0, y0, x1, y1 = lines_xyxy[li]
            stringer_lines_xy_m.append([
                (float(x0) * px_to_m, float(y0) * px_to_m),
                (float(x1) * px_to_m, float(y1) * px_to_m),
            ])
        # Stringer distance = perpendicular distance between the two
        # midpoints projected onto the short axis.
        left_mid = _line_midpoint_px(lines_xyxy[left_best])
        right_mid = _line_midpoint_px(lines_xyxy[right_best])
        dl = _project_point_onto_axis(
            left_mid, origin_xy_px=bbox_center_px, axis_unit=short_unit,
        )
        dr = _project_point_onto_axis(
            right_mid, origin_xy_px=bbox_center_px, axis_unit=short_unit,
        )
        details.stringer_distance_m = abs(dr - dl) * px_to_m

        # Stringer-derived long-axis: average of the two stringer line
        # directions. Each stringer line has its own unit vector; we
        # take the mean (sign-aligned so the two don't cancel each
        # other when they point in opposite endpoint orders).
        stringer_units: List[Tuple[float, float]] = []
        reference_unit = None
        for li in (left_best, right_best):
            x0, y0, x1, y1 = lines_xyxy[li]
            dx, dy = float(x1 - x0), float(y1 - y0)
            n = math.hypot(dx, dy)
            if n < 1e-9:
                continue
            u = (dx / n, dy / n)
            if reference_unit is None:
                reference_unit = u
            else:
                # Flip to align with the reference.
                if u[0] * reference_unit[0] + u[1] * reference_unit[1] < 0:
                    u = (-u[0], -u[1])
            stringer_units.append(u)
        if stringer_units:
            mean_ux = sum(u[0] for u in stringer_units) / len(stringer_units)
            mean_uy = sum(u[1] for u in stringer_units) / len(stringer_units)
            mag = math.hypot(mean_ux, mean_uy)
            if mag > 1e-6:
                details.stringer_long_axis = (
                    float(mean_ux / mag), float(mean_uy / mag),
                )

        # Authority weight for the stringer-derived long axis. Combines
        # how much of the run is actually covered by the stringer ink
        # (length sum / bbox long extent, capped at 1) with the mean
        # confidence of the stringer segments themselves.
        total_stringer_px = (
            _line_length_px(lines_xyxy[left_best])
            + _line_length_px(lines_xyxy[right_best])
        )
        coverage = min(
            1.0,
            float(total_stringer_px) / max(2.0 * long_ext_px, 1.0),
        )
        details.stringer_axis_authority = float(
            coverage * details.mean_stringer_line_confidence,
        )
    details.stringer_lines_xy_m = stringer_lines_xy_m

    # 9. Cut-line detection. A cut line is a perpendicular line at one
    # of the bbox ends drawn THICKER than the tread lines. We score
    # each tread-direction line by counting ink pixels in a thin band
    # along its segment in the original dark_mask — thicker strokes
    # produce more pixels per unit length.
    cutlines_xy_m: List[List[Tuple[float, float]]] = []
    if tread_indices:
        # Build a quick density-per-segment using a 1-px-wide line on a
        # blank canvas, dilated by the probe radius, then AND-ed with
        # the dark_mask. This counts the "ink under stroke" per line.
        import cv2  # noqa: PLC0415
        probe_radius_px = max(
            2, int(round(CUTLINE_THICKNESS_PROBE_PX_M / max(px_to_m, 1e-9))),
        )
        scores: List[Tuple[float, int]] = []
        for i in tread_indices:
            x0, y0, x1, y1 = lines_xyxy[i]
            canvas = np.zeros_like(dark_mask, dtype=np.uint8)
            cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), 255, 1)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (probe_radius_px * 2 + 1, probe_radius_px * 2 + 1),
            )
            band = cv2.dilate(canvas, kernel, iterations=1)
            ink_under = int(((band > 0) & (dark_mask > 0)).sum())
            line_length = max(_line_length_px(lines_xyxy[i]), 1.0)
            scores.append((ink_under / line_length, i))

        if scores:
            # Use median as baseline. A cut line is ≥1.7x the median
            # density AND sits in the outer 25 % of the bbox along the
            # long axis.
            score_vals = sorted(s for s, _ in scores)
            median_score = score_vals[len(score_vals) // 2]
            outer_band_lo = 0.0
            outer_band_hi = 0.25 * long_ext_px
            inner_band_lo = 0.75 * long_ext_px
            inner_band_hi = long_ext_px

            for s, i in scores:
                if s < 1.7 * max(median_score, 1e-3):
                    continue
                mid = _line_midpoint_px(lines_xyxy[i])
                proj = _project_point_onto_axis(
                    mid, origin_xy_px=bbox_min, axis_unit=long_unit,
                )
                if (outer_band_lo <= proj <= outer_band_hi
                        or inner_band_lo <= proj <= inner_band_hi):
                    x0, y0, x1, y1 = lines_xyxy[i]
                    cutlines_xy_m.append([
                        (float(x0) * px_to_m, float(y0) * px_to_m),
                        (float(x1) * px_to_m, float(y1) * px_to_m),
                    ])
            details.n_cutline_candidates = len(cutlines_xy_m)
    details.cutlines_xy_m = cutlines_xy_m

    # 9b. Cross-riser detection (winder / quarter-turn wings). The main
    # pass above is tuned to find the long treads + stringers; a winder
    # wing's step fronts are SHORT, nested, parallel to the long axis,
    # and would be filtered out by the main pass's minLineLength. Run a
    # dedicated low-minLineLength pass, keep only near-parallel segments,
    # cluster them along the short axis, and expose each cluster at its
    # real ink extent. The two outer stringers naturally fall out here
    # too; that is harmless — they snap onto the boundary downstream.
    cross_min_len_px = max(
        4,
        int(round(min(long_ext_px, short_ext_px) * CROSS_RISER_MIN_LENGTH_FRAC)),
    )
    cross_lines_xyxy = _hough_lines(
        dark_mask,
        min_line_length_px=cross_min_len_px,
        max_line_gap_px=max_line_gap_px,
        threshold=threshold_px,
    )
    cross_parallel_idx: List[int] = []
    for i in range(cross_lines_xyxy.shape[0]):
        if (
            confidence_field is not None
            and _line_mean_confidence(cross_lines_xyxy[i], confidence_field)
            < LINE_CONFIDENCE_FLOOR
        ):
            continue
        if (
            _line_orientation_to_axis_deg(cross_lines_xyxy[i], long_unit)
            <= STRINGER_PARALLEL_TOL_DEG
        ):
            cross_parallel_idx.append(i)

    cross_samples: List[Tuple[float, int]] = []
    for i in cross_parallel_idx:
        mid = _line_midpoint_px(cross_lines_xyxy[i])
        s = _project_point_onto_axis(
            mid, origin_xy_px=bbox_min, axis_unit=short_unit,
        )
        cross_samples.append((float(s) * px_to_m, i))
    cross_clusters = _cluster_1d(cross_samples, eps=CROSS_RISER_CLUSTER_EPS_M)

    cross_riser_lines_xy_m: List[List[Tuple[float, float]]] = []
    for cluster in cross_clusters:
        long_projs: List[float] = []
        short_projs: List[float] = []
        for i in cluster:
            x0, y0, x1, y1 = cross_lines_xyxy[i]
            for px, py in ((float(x0), float(y0)), (float(x1), float(y1))):
                long_projs.append(_project_point_onto_axis(
                    (px, py), origin_xy_px=bbox_min, axis_unit=long_unit,
                ))
                short_projs.append(_project_point_onto_axis(
                    (px, py), origin_xy_px=bbox_min, axis_unit=short_unit,
                ))
        s_pos = float(np.median(short_projs))
        l_lo, l_hi = min(long_projs), max(long_projs)
        a_px = (
            bbox_min[0] + l_lo * long_unit[0] + s_pos * short_unit[0],
            bbox_min[1] + l_lo * long_unit[1] + s_pos * short_unit[1],
        )
        b_px = (
            bbox_min[0] + l_hi * long_unit[0] + s_pos * short_unit[0],
            bbox_min[1] + l_hi * long_unit[1] + s_pos * short_unit[1],
        )
        if math.hypot(b_px[0] - a_px[0], b_px[1] - a_px[1]) < cross_min_len_px:
            continue
        if (
            _line_ink_ratio_px(a_px, b_px, image_gray, int(threshold))
            < CROSS_RISER_MIN_INK_RATIO
        ):
            continue
        cross_riser_lines_xy_m.append([
            (a_px[0] * px_to_m, a_px[1] * px_to_m),
            (b_px[0] * px_to_m, b_px[1] * px_to_m),
        ])
    cross_riser_lines_xy_m.sort(
        key=lambda ln: _project_point_onto_axis(
            (
                0.5 * (ln[0][0] + ln[1][0]) / px_to_m,
                0.5 * (ln[0][1] + ln[1][1]) / px_to_m,
            ),
            origin_xy_px=bbox_min,
            axis_unit=short_unit,
        )
    )
    details.cross_riser_lines_xy_m = cross_riser_lines_xy_m
    if cross_riser_lines_xy_m:
        details.notes.append(
            f"cross_risers={len(cross_riser_lines_xy_m)} "
            f"(min_len_px={cross_min_len_px})"
        )

    # 10. Final status.
    if not details.tread_centroids_xy_m:
        details.status = "no_clusters"
    else:
        details.status = "ok"
        details.notes.append(
            f"hough: {details.n_hough_lines} lines, "
            f"tread_cands={details.n_tread_candidates}, "
            f"stringer_cands={details.n_stringer_candidates}, "
            f"clusters={len(details.tread_centroids_xy_m)}, "
            f"stringers={'pair' if stringer_lines_xy_m else 'none'}, "
            f"cutlines={len(details.cutlines_xy_m)}, "
            f"stringer_min_length_px={int(stringer_min_length_px)}"
        )
    return details


__all__ = [
    "StairLineDetails",
    "extract_stair_lines",
    "STAIR_ROI_MARGIN_M",
]
