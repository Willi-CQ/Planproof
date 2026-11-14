"""Multi-signal classifier: does a sliding-door symbol sit IN a wall
(``in_wall`` — the panel slides inside wall material) or BESIDE one
(``beside_wall`` — barn-door style mounted on the wall surface)?

Tag-1 wired two geometric signals (polygon containment + bbox
short-axis straddle). Tag-2 adds two image-based signals (Otsu ink
test + panel-endpoint test) so up to 4 voters can disagree, with
ratio + count both factored into confidence (mirrors
``vote_hinge_long_sign``).

Output ``MountClassification.mount_class`` drives downstream routing
in ``diagnose_door_placement``:

  * ``in_wall``      → existing ``build_sliding_door_polygon`` path
                       (door is an aperture in a continuous wall).
  * ``beside_wall``  → new Tag-3 ``build_beside_wall_sliding_polygon``
                       path (panel hangs off the wall surface, no
                       wall-aperture cut).
  * ``abstain``      → caller falls back to the legacy ``place_door``
                       cascade (today's behaviour below
                       ``wall_ctx.confidence < 0.5``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Bbox needs at least this many metres on each side of the wall midline
# to count as "straddling". Even very thin walls (4 cm) get a sensible
# floor instead of a near-zero threshold every annotation would pass.
_STRADDLE_FLOOR_M = 0.04
# Bbox-extent fraction of wall_depth that counts as straddle on a side.
_STRADDLE_FRACTION = 0.4

# Tag-2: ink test parameters. Dilate absorbs 1-2 px gaps in symbols so
# a thin sliding-line through the bbox centre still registers as ink.
_INK_DILATE_PX = 3
# Neighbour offset for the "all 4 neighbours ink-free" tolerance — only
# vote against in-wall when the centre AND all 4 neighbours are clear.
_INK_NEIGHBOUR_OFFSET_PX = 5
# Panel endpoint: max distance from endpoint to nearest ink for the
# endpoint to count as "on the wall". 5 cm matches typical wall depths.
_PANEL_ENDPOINT_TOLERANCE_M = 0.05

# bbox_wall_overlap: thresholds on intersection-area / bbox-area. The
# fractions catch typical slider symbols where the bbox wraps a panel
# indicator that extends past the wall — substantial overlap remains
# even when the bbox centre falls outside the wall polygon.
# In_wall threshold 0.10 calibrated against Matterport b5s0_15
# (13.5% overlap, asymmetric slider bbox); barn-door cases sit at
# 0-3% so the 3pp gap to the beside threshold is comfortable.
_OVERLAP_IN_WALL_FRAC = 0.10
_OVERLAP_BESIDE_FRAC = 0.03


@dataclass(frozen=True)
class MountClassification:
    """Result of the in-wall vs beside-wall vote."""

    mount_class: str               # "in_wall" | "beside_wall" | "abstain"
    confidence: float              # 0..1, ratio * min(1, n/3)
    votes: Dict[str, int]          # signal -> +1 in_wall | -1 beside | 0 abstain
    notes: List[str] = field(default_factory=list)


def _bbox_center_xy(bbox_m: Tuple[float, float, float, float]) -> np.ndarray:
    return np.array([
        (bbox_m[0] + bbox_m[2]) * 0.5,
        (bbox_m[1] + bbox_m[3]) * 0.5,
    ])


def _bbox_corners_xy(bbox_m: Tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = bbox_m
    return np.array([
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
    ], dtype=np.float64)


def signal_polygon_containment(
    bbox_m: Tuple[float, float, float, float],
    wall_polygons: Sequence,
) -> int:
    """``+1`` if the bbox centre lies inside any wall polygon, else ``0``.

    Containment is a strong "in-wall" signal. Non-containment is NOT a
    strong "beside-wall" signal: user wall annotations frequently fall
    1-2 cm short of the door symbol. So we abstain rather than voting
    against in-wall — let other signals decide.
    """
    try:
        from shapely.geometry import Point, Polygon as SPoly  # noqa: PLC0415
    except ImportError:
        return 0
    pt = Point(*_bbox_center_xy(bbox_m).tolist())
    for wp in wall_polygons:
        shell = getattr(wp, "shell_xy", None)
        if shell is None or len(shell) < 3:
            continue
        try:
            poly = SPoly(shell, holes=getattr(wp, "holes_xy", None)).buffer(0)
        except Exception:
            continue
        if not poly.is_empty and poly.contains(pt):
            return +1
    return 0


def signal_bbox_straddle(
    bbox_m: Tuple[float, float, float, float],
    wall_ctx,
) -> int:
    """``+1`` if the bbox extends past the wall midline on BOTH short-axis
    sides; ``-1`` if it extends only on ONE side; ``0`` if it barely
    extends on either.

    Pocket-slider symbols are drawn inside the wall material — bbox
    straddles the midline. Barn-door symbols hang off ONE wall surface —
    bbox is one-sided.

    Threshold per side:
    ``max(_STRADDLE_FLOOR_M, _STRADDLE_FRACTION × wall_depth)``.
    """
    if wall_ctx is None or not getattr(wall_ctx, "found", False):
        return 0
    short_u = np.asarray(wall_ctx.wall_axis_short_u, dtype=np.float64)
    centerline = np.asarray(wall_ctx.wall_centerline_xy, dtype=np.float64)
    wall_depth = float(getattr(wall_ctx, "wall_depth_m", 0.0))
    threshold = max(_STRADDLE_FLOOR_M, _STRADDLE_FRACTION * wall_depth)
    proj = (_bbox_corners_xy(bbox_m) - centerline) @ short_u
    min_proj = float(proj.min())
    max_proj = float(proj.max())
    pos_side = max_proj >= +threshold
    neg_side = min_proj <= -threshold
    if pos_side and neg_side:
        return +1
    if pos_side or neg_side:
        return -1
    return 0


def _otsu_ink_mask(image_gray, dilate_px: int = 0):
    """Return an Otsu-thresholded ink mask: 255 where ink, 0 elsewhere.

    Mirrors the convention used everywhere else in this package
    (``cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU``). Optional dilation
    absorbs sub-pixel gaps in symbol strokes.
    """
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return None
    if image_gray is None:
        return None
    _, otsu = cv2.threshold(
        image_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (dilate_px, dilate_px),
        )
        otsu = cv2.dilate(otsu, kernel, iterations=1)
    return otsu


def _safe_sample(mask, x_px: int, y_px: int) -> bool:
    H, W = mask.shape[:2]
    if 0 <= y_px < H and 0 <= x_px < W:
        return bool(mask[y_px, x_px] > 0)
    return False


def signal_ink_test(
    bbox_m: Tuple[float, float, float, float],
    image_gray,
    px_to_m: Optional[float],
    *,
    anchor_xy: Optional[Tuple[float, float]] = None,
) -> int:
    """``+1`` if the anchor lands on Otsu-detected ink (after a 3 px
    dilate); ``-1`` only if BOTH the anchor AND all 4 ±5 px neighbours
    are ink-free; ``0`` otherwise.

    ``anchor_xy`` defaults to the bbox centre. ``classify_sliding_mount_voting``
    overrides it with the bbox∩wall intersection centroid so asymmetric
    slider bboxes (where the centre falls into the room past the wall)
    are still tested at a wall-anchored point — the panel sits in the
    wall, the test should land on the wall's ink stripe.

    Pocket-slider symbols are drawn between the paired-outline strokes,
    so the wall-anchored point lands on ink. Barn-door symbols sit on
    white background — anchor and neighbours all clear. The neighbour
    tolerance prevents 1-2 cm gaps in a sliding symbol's centre line
    from voting against in-wall.
    """
    if image_gray is None or px_to_m is None or px_to_m <= 0:
        return 0
    mask = _otsu_ink_mask(image_gray, dilate_px=_INK_DILATE_PX)
    if mask is None:
        return 0
    if anchor_xy is None:
        ax = (bbox_m[0] + bbox_m[2]) * 0.5
        ay = (bbox_m[1] + bbox_m[3]) * 0.5
    else:
        ax, ay = anchor_xy
    cx_px = int(round(ax / px_to_m))
    cy_px = int(round(ay / px_to_m))
    if _safe_sample(mask, cx_px, cy_px):
        return +1
    off = _INK_NEIGHBOUR_OFFSET_PX
    neighbours = [
        (cx_px + off, cy_px),
        (cx_px - off, cy_px),
        (cx_px, cy_px + off),
        (cx_px, cy_px - off),
    ]
    if any(_safe_sample(mask, x, y) for x, y in neighbours):
        return 0
    return -1


def _bbox_wall_overlap_fraction(
    bbox_m: Tuple[float, float, float, float],
    wall_polygons: Sequence,
) -> Optional[float]:
    """Return ``intersection_area / bbox_area`` or ``None`` if not
    computable. Public diagnostic helper for callers that want to log
    the raw fraction alongside the ±1/0 vote."""
    try:
        from shapely.geometry import Polygon as SPoly  # noqa: PLC0415
        from shapely.ops import unary_union  # noqa: PLC0415
    except ImportError:
        return None
    x0, y0, x1, y1 = bbox_m
    bbox_area = (x1 - x0) * (y1 - y0)
    if bbox_area <= 0:
        return None
    bbox_poly = SPoly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    geoms = []
    for wp in wall_polygons:
        shell = getattr(wp, "shell_xy", None)
        if shell is None or len(shell) < 3:
            continue
        try:
            sp = SPoly(shell, holes=getattr(wp, "holes_xy", None)).buffer(0)
            if not sp.is_empty:
                geoms.append(sp)
        except Exception:
            continue
    if not geoms:
        return None
    try:
        return float(bbox_poly.intersection(unary_union(geoms)).area) / bbox_area
    except Exception:
        return None


def signal_bbox_wall_overlap(
    bbox_m: Tuple[float, float, float, float],
    wall_polygons: Sequence,
) -> int:
    """``+1`` if at least 15% of the bbox area sits inside a wall polygon;
    ``-1`` if at most 3% does; ``0`` in between.

    Direct measurement of in-wall placement that survives asymmetric
    slider bboxes — the bbox centre may fall outside the wall (because
    the symbol bbox includes a slide-direction indicator extending past
    the wall) but the OVERLAP fraction still reveals an in-wall slider.
    """
    frac = _bbox_wall_overlap_fraction(bbox_m, wall_polygons)
    if frac is None:
        return 0
    if frac >= _OVERLAP_IN_WALL_FRAC:
        return +1
    if frac <= _OVERLAP_BESIDE_FRAC:
        return -1
    return 0


def _intersection_centroid(
    bbox_m: Tuple[float, float, float, float],
    wall_polygons: Sequence,
) -> Optional[Tuple[float, float]]:
    """Centroid of the bbox∩wall intersection. ``None`` if no overlap.

    Used as a wall-anchored test point for ``signal_ink_test`` so
    asymmetric slider bboxes are sampled inside the wall, not in the
    arrow/indicator overhang past the wall.
    """
    try:
        from shapely.geometry import Polygon as SPoly  # noqa: PLC0415
        from shapely.ops import unary_union  # noqa: PLC0415
    except ImportError:
        return None
    x0, y0, x1, y1 = bbox_m
    if (x1 - x0) <= 0 or (y1 - y0) <= 0:
        return None
    bbox_poly = SPoly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    geoms = []
    for wp in wall_polygons:
        shell = getattr(wp, "shell_xy", None)
        if shell is None or len(shell) < 3:
            continue
        try:
            sp = SPoly(shell, holes=getattr(wp, "holes_xy", None)).buffer(0)
            if not sp.is_empty:
                geoms.append(sp)
        except Exception:
            continue
    if not geoms:
        return None
    try:
        inter = bbox_poly.intersection(unary_union(geoms))
    except Exception:
        return None
    if inter.is_empty or inter.area < 1e-6:
        return None
    c = inter.centroid
    return float(c.x), float(c.y)


def signal_panel_endpoint(
    panel_pixel_endpoints,
    image_gray,
    px_to_m: Optional[float],
) -> int:
    """``+1`` if both panel endpoints land within ``5 cm`` of ink;
    ``-1`` if both are clearly off ink; ``0`` otherwise.

    Pocket-slider panels terminate inside wall material — both
    endpoints sit on (or within wall_depth/2 of) ink. Barn-door panels
    hang in room space — both endpoints far from ink.
    """
    if (
        panel_pixel_endpoints is None
        or len(panel_pixel_endpoints) < 2
        or image_gray is None
        or px_to_m is None
        or px_to_m <= 0
    ):
        return 0
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return 0
    mask = _otsu_ink_mask(image_gray, dilate_px=0)
    if mask is None:
        return 0
    inv = (mask == 0).astype(np.uint8)
    dist_px = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    tol_px = _PANEL_ENDPOINT_TOLERANCE_M / px_to_m
    H, W = dist_px.shape
    on_ink: List[bool] = []
    for ep in panel_pixel_endpoints[:2]:
        x, y = int(round(ep[0])), int(round(ep[1]))
        if 0 <= x < W and 0 <= y < H:
            on_ink.append(bool(dist_px[y, x] <= tol_px))
        else:
            on_ink.append(False)
    if all(on_ink):
        return +1
    if not any(on_ink):
        return -1
    return 0


def _aggregate_votes(votes: Dict[str, int]) -> Tuple[str, float]:
    """Mirror ``vote_hinge_long_sign``: majority decides class, confidence
    is ``ratio × min(1, n/3)`` where ``n`` is the non-abstain voter count.

    A tie returns ``("abstain", 0.5 × min(1, n/3))`` — caller treats
    abstain as "let the legacy cascade decide".
    """
    valid = [v for v in votes.values() if v in (-1, 1)]
    n = len(valid)
    if n == 0:
        return "abstain", 0.0
    pos = sum(1 for v in valid if v > 0)
    neg = n - pos
    n_factor = min(1.0, n / 3.0)
    if pos == neg:
        return "abstain", 0.5 * n_factor
    if pos > neg:
        return "in_wall", (pos / n) * n_factor
    return "beside_wall", (neg / n) * n_factor


def classify_sliding_mount_voting(
    bbox_m: Tuple[float, float, float, float],
    wall_ctx,
    *,
    wall_polygons: Sequence = (),
    image_gray=None,
    px_to_m: Optional[float] = None,
    panel_pixel_endpoints=None,
) -> MountClassification:
    """Classify a sliding door as ``in_wall`` / ``beside_wall`` / ``abstain``.

    Up to four signals contribute, each independently abstaining when
    its inputs are missing:

    * ``polygon_containment`` (geometric): always available.
    * ``bbox_straddle``       (geometric): always available.
    * ``ink_test``            (image): needs ``image_gray`` + ``px_to_m``.
    * ``panel_endpoint``      (image): needs endpoints + image + scale.

    Short-circuit: when no wall was found (``wall_ctx.found`` is False),
    abstain immediately — without a wall the in/beside comparison is
    meaningless and the caller should fall back to the legacy cascade.
    """
    notes: List[str] = []
    if wall_ctx is None or not getattr(wall_ctx, "found", False):
        notes.append("no wall found - short-circuit abstain")
        return MountClassification(
            mount_class="abstain",
            confidence=0.0,
            votes={},
            notes=notes,
        )

    # Wall-anchored test point for ink_test: when the bbox overlaps a
    # wall, sample at the intersection centroid (always inside a wall
    # polygon) instead of the bbox centre. Asymmetric slider bboxes
    # otherwise sample the slide-direction overhang past the wall.
    ink_anchor = _intersection_centroid(bbox_m, wall_polygons)

    votes: Dict[str, int] = {
        "polygon_containment": signal_polygon_containment(bbox_m, wall_polygons),
        "bbox_straddle": signal_bbox_straddle(bbox_m, wall_ctx),
        "bbox_wall_overlap": signal_bbox_wall_overlap(bbox_m, wall_polygons),
        "ink_test": signal_ink_test(
            bbox_m, image_gray, px_to_m, anchor_xy=ink_anchor,
        ),
        "panel_endpoint": signal_panel_endpoint(
            panel_pixel_endpoints, image_gray, px_to_m,
        ),
    }

    mount_class, confidence = _aggregate_votes(votes)
    notes.append(f"votes={votes} -> {mount_class} conf={confidence:.2f}")
    return MountClassification(
        mount_class=mount_class,
        confidence=confidence,
        votes=votes,
        notes=notes,
    )


__all__ = [
    "MountClassification",
    "classify_sliding_mount_voting",
    "signal_bbox_straddle",
    "signal_bbox_wall_overlap",
    "signal_ink_test",
    "signal_panel_endpoint",
    "signal_polygon_containment",
]
