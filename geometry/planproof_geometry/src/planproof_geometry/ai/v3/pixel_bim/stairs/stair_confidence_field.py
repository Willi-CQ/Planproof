"""Per-pixel confidence field for stair pixel attribution.

The user label is a *gravity well*, not a hard ROI cap. Pixels just
outside the bbox almost certainly belong to the stair (sloppy human
label, ink that bleeds past the click point); pixels far outside almost
certainly don't. And pixels closer to a *different* label (a neighbour
door/window/wall) probably belong to that neighbour, not to this stair.

This module builds a per-pixel confidence map encoding that intuition,
which the downstream Hough detector then uses as an evidence weight on
every linestroke it finds.

Confidence shape:
    * 1.0 inside the user polygon
    * Exponential decay outside: exp(-d / τ) with τ chosen so that
      conf(0 cm) ≈ 1.0, conf(5 cm) ≈ 0.95, conf(100 cm) ≈ 0.05
    * Multiplied by a neighbour-competition ratio
          ratio = d_neighbour / (d_neighbour + d_user)
      which is 1.0 when the user is much closer, 0.5 when distances are
      equal, and 0.0 when the pixel is inside a neighbour. Inside the
      user polygon the ratio is forced to 1.0 — we never let a
      neighbour erase pixels that already lie inside the labelled bbox.

The map is float32 in [0, 1] and the SAME shape as ``image_gray`` so it
can be element-wise multiplied with the ink mask.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class ConfidenceField:
    """Sub-window confidence map + the image-space offset.

    Avoids allocating a ``H × W`` float32 array per stair (would be
    ~380 MB on an 11 584 × 8 192 plan) — instead the caller queries the
    field via :meth:`sample` which translates image-space coords into
    the sub-window frame.
    """

    field_sub: np.ndarray             # float32 (sub_h, sub_w), in [0, 1]
    x_offset_px: int                  # column offset in original image
    y_offset_px: int                  # row offset in original image
    image_shape_hw: Tuple[int, int]   # original image shape (for clamping)

    def sample(self, x_px: float, y_px: float) -> float:
        H_sub, W_sub = self.field_sub.shape
        ix = int(round(x_px)) - self.x_offset_px
        iy = int(round(y_px)) - self.y_offset_px
        if ix < 0 or iy < 0 or ix >= W_sub or iy >= H_sub:
            return 0.0
        return float(self.field_sub[iy, ix])

    def sample_along_line(
        self,
        x0: float, y0: float, x1: float, y1: float,
        *,
        n_samples: int = 24,
    ) -> float:
        ts = np.linspace(0.0, 1.0, max(2, n_samples), dtype=np.float64)
        xs = x0 + (x1 - x0) * ts - self.x_offset_px
        ys = y0 + (y1 - y0) * ts - self.y_offset_px
        H_sub, W_sub = self.field_sub.shape
        ix = np.round(xs).astype(np.int64)
        iy = np.round(ys).astype(np.int64)
        valid = (ix >= 0) & (ix < W_sub) & (iy >= 0) & (iy < H_sub)
        if not bool(valid.any()):
            return 0.0
        ix_v = ix[valid]
        iy_v = iy[valid]
        # Pixels outside the sub-window count as 0 confidence.
        n_valid = int(valid.sum())
        s = float(self.field_sub[iy_v, ix_v].sum())
        return s / float(len(ts))


# Falloff radius: distance at which confidence drops to ~5 %. The Hough
# detector treats anything below ~0.05 as effectively zero, so 100 cm
# is the practical maximum reach of a single stair label.
DEFAULT_FALLOFF_RADIUS_M = 1.00

# Floor confidence inside the user polygon. The polygon is always
# treated as fully owned by this stair — even when a neighbour bbox
# overlaps it, the user already declared this region "stair".
INSIDE_USER_CONFIDENCE = 1.0

# Numerical floor so divisions never blow up at exact pixel boundaries.
DIST_EPS_M = 1e-4


def _polygon_inside_mask(
    polygon_px: Sequence[Tuple[float, float]],
    image_shape_hw: Tuple[int, int],
) -> np.ndarray:
    """Rasterise polygon → uint8 mask (255 inside, 0 outside)."""
    import cv2  # noqa: PLC0415

    H, W = image_shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    if len(polygon_px) < 3:
        return mask
    pts = np.array(
        [[int(round(x)), int(round(y))] for x, y in polygon_px],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _distance_outside_polygon_m(
    polygon_px: Sequence[Tuple[float, float]],
    image_shape_hw: Tuple[int, int],
    *,
    px_to_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(inside_mask_uint8, distance_outside_m_float32)``.

    Inside mask is 255 inside the polygon, 0 outside. Distance map is 0
    inside and grows outward in metres.
    """
    import cv2  # noqa: PLC0415

    inside = _polygon_inside_mask(polygon_px, image_shape_hw)
    if int(inside.sum()) == 0:
        return inside, np.full(image_shape_hw, 1e9, dtype=np.float32)
    # Distance to NEAREST inside pixel from every position; 0 inside,
    # grows outward.
    dist_px = cv2.distanceTransform(
        255 - inside, cv2.DIST_L2, maskSize=5,
    )
    return inside, (dist_px.astype(np.float32) * float(px_to_m))


def build_confidence_field_sub(
    user_polygon_px: Sequence[Tuple[float, float]],
    neighbor_polygons_px: Sequence[Sequence[Tuple[float, float]]],
    image_shape_hw: Tuple[int, int],
    *,
    px_to_m: float,
    falloff_radius_m: float = DEFAULT_FALLOFF_RADIUS_M,
    sub_window_margin_m: float = 2.0,
) -> "ConfidenceField":
    """Fast variant: returns a ConfidenceField wrapping the sub-window.

    Identical semantics to :func:`build_confidence_field` but avoids
    allocating an ``H × W`` float32 paste-back buffer (which is ~380 MB
    on an 11 584 × 8 192 plan and is the main hot-path bottleneck when
    processing several stairs in one pipeline pass).
    """
    H, W = image_shape_hw
    if px_to_m <= 0.0 or falloff_radius_m <= 0.0 or len(user_polygon_px) < 3:
        return ConfidenceField(
            field_sub=np.zeros((1, 1), dtype=np.float32),
            x_offset_px=0, y_offset_px=0,
            image_shape_hw=image_shape_hw,
        )
    margin_px = max(8, int(round(sub_window_margin_m / max(px_to_m, 1e-9))))
    poly_arr = np.asarray(user_polygon_px, dtype=np.float64)
    x_lo = max(0, int(math.floor(poly_arr[:, 0].min())) - margin_px)
    x_hi = min(W, int(math.ceil(poly_arr[:, 0].max())) + margin_px)
    y_lo = max(0, int(math.floor(poly_arr[:, 1].min())) - margin_px)
    y_hi = min(H, int(math.ceil(poly_arr[:, 1].max())) + margin_px)
    if x_hi <= x_lo or y_hi <= y_lo:
        return ConfidenceField(
            field_sub=np.zeros((1, 1), dtype=np.float32),
            x_offset_px=0, y_offset_px=0,
            image_shape_hw=image_shape_hw,
        )
    sub_h = y_hi - y_lo
    sub_w = x_hi - x_lo

    def _translate(poly):
        return [(x - x_lo, y - y_lo) for x, y in poly]

    user_sub = _translate(user_polygon_px)
    user_inside, user_dist_outside_m = _distance_outside_polygon_m(
        user_sub, (sub_h, sub_w), px_to_m=px_to_m,
    )
    if int(user_inside.sum()) == 0:
        return ConfidenceField(
            field_sub=np.zeros((sub_h, sub_w), dtype=np.float32),
            x_offset_px=x_lo, y_offset_px=y_lo,
            image_shape_hw=image_shape_hw,
        )

    tau_m = float(falloff_radius_m) / float(math.log(20.0))
    conf_sub = np.ones((sub_h, sub_w), dtype=np.float32) * INSIDE_USER_CONFIDENCE
    outside_mask = user_inside == 0
    if outside_mask.any():
        conf_sub[outside_mask] = np.exp(
            -user_dist_outside_m[outside_mask] / tau_m,
        ).astype(np.float32)

    for nbr_poly in neighbor_polygons_px:
        if len(nbr_poly) < 3:
            continue
        nbr_arr = np.asarray(nbr_poly, dtype=np.float64)
        if (nbr_arr[:, 0].max() < x_lo or nbr_arr[:, 0].min() > x_hi
                or nbr_arr[:, 1].max() < y_lo or nbr_arr[:, 1].min() > y_hi):
            continue
        nbr_sub = _translate(nbr_poly)
        nbr_inside, nbr_dist_outside_m = _distance_outside_polygon_m(
            nbr_sub, (sub_h, sub_w), px_to_m=px_to_m,
        )
        if int(nbr_inside.sum()) == 0:
            continue
        ratio = nbr_dist_outside_m / (
            nbr_dist_outside_m + user_dist_outside_m + DIST_EPS_M
        )
        ratio = np.where(user_inside > 0, 1.0, ratio).astype(np.float32)
        conf_sub *= ratio

    np.clip(conf_sub, 0.0, 1.0, out=conf_sub)
    return ConfidenceField(
        field_sub=conf_sub,
        x_offset_px=x_lo, y_offset_px=y_lo,
        image_shape_hw=image_shape_hw,
    )


def build_confidence_field(
    user_polygon_px: Sequence[Tuple[float, float]],
    neighbor_polygons_px: Sequence[Sequence[Tuple[float, float]]],
    image_shape_hw: Tuple[int, int],
    *,
    px_to_m: float,
    falloff_radius_m: float = DEFAULT_FALLOFF_RADIUS_M,
    sub_window_margin_m: float = 2.0,
) -> np.ndarray:
    """Build the per-pixel confidence map for one stair label.

    Args:
        user_polygon_px: vertices of the labelled bbox / polygon in
            render-image pixel coords.
        neighbor_polygons_px: every other label polygon on the same
            plan (regardless of class) — doors, windows, walls,
            furniture, other stairs. Each one competes for ownership of
            nearby pixels. Empty list = no competition.
        image_shape_hw: ``(H, W)`` of the plan render. The output map
            is the same shape.
        px_to_m: metres-per-pixel of the plan render.
        falloff_radius_m: distance at which confidence decays to ~5 %.
            Default 1.0 m: at 0 cm conf ≈ 1.0, at 5 cm ≈ 0.95, at
            100 cm ≈ 0.05.
        sub_window_margin_m: padding around the user polygon for the
            sub-window. Confidence outside this window is forced to 0
            (a stair never reaches > sub_window_margin_m from its
            label).

    Returns:
        float32 ``(H, W)`` map in [0, 1]. Higher = "this pixel more
        likely belongs to this stair". Pixels outside the sub-window
        are guaranteed 0 — distance-transform is only computed inside
        the window for performance.
    """
    if px_to_m <= 0.0 or falloff_radius_m <= 0.0:
        return np.zeros(image_shape_hw, dtype=np.float32)

    H, W = image_shape_hw

    if len(user_polygon_px) < 3:
        return np.full((H, W), 0.05, dtype=np.float32)

    # Sub-window: user-polygon bbox + margin. Distance-transform on the
    # full plan would be ~95 MP per neighbour — way too slow for
    # 100+ annotation neighbours per stair. Restricting the
    # confidence-field computation to a window around the user bbox is
    # safe because confidence is ~0 anyway beyond falloff_radius_m + a
    # bit of padding for neighbour-competition zones.
    margin_px = max(8, int(round(sub_window_margin_m / max(px_to_m, 1e-9))))
    poly_arr = np.asarray(user_polygon_px, dtype=np.float64)
    x_lo = max(0, int(math.floor(poly_arr[:, 0].min())) - margin_px)
    x_hi = min(W, int(math.ceil(poly_arr[:, 0].max())) + margin_px)
    y_lo = max(0, int(math.floor(poly_arr[:, 1].min())) - margin_px)
    y_hi = min(H, int(math.ceil(poly_arr[:, 1].max())) + margin_px)
    if x_hi <= x_lo or y_hi <= y_lo:
        return np.zeros((H, W), dtype=np.float32)

    sub_h = y_hi - y_lo
    sub_w = x_hi - x_lo

    # Translate every polygon into the sub-window frame.
    def _translate(poly: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [(x - x_lo, y - y_lo) for x, y in poly]

    user_sub = _translate(user_polygon_px)
    user_inside, user_dist_outside_m = _distance_outside_polygon_m(
        user_sub, (sub_h, sub_w), px_to_m=px_to_m,
    )
    if int(user_inside.sum()) == 0:
        return np.full((H, W), 0.05, dtype=np.float32)

    # 1. Base confidence: exponential decay outside the user polygon.
    # τ chosen so that exp(-falloff_radius / τ) = 0.05 → τ = R / ln(20)
    tau_m = float(falloff_radius_m) / float(math.log(20.0))
    conf_sub = np.ones((sub_h, sub_w), dtype=np.float32) * INSIDE_USER_CONFIDENCE
    outside_mask = user_inside == 0
    if outside_mask.any():
        conf_sub[outside_mask] = np.exp(
            -user_dist_outside_m[outside_mask] / tau_m,
        ).astype(np.float32)

    # 2. Neighbour competition — restricted to neighbours whose bbox
    # actually intersects the sub-window (skip everything far away).
    for nbr_poly in neighbor_polygons_px:
        if len(nbr_poly) < 3:
            continue
        nbr_arr = np.asarray(nbr_poly, dtype=np.float64)
        n_x_lo = float(nbr_arr[:, 0].min())
        n_x_hi = float(nbr_arr[:, 0].max())
        n_y_lo = float(nbr_arr[:, 1].min())
        n_y_hi = float(nbr_arr[:, 1].max())
        if n_x_hi < x_lo or n_x_lo > x_hi or n_y_hi < y_lo or n_y_lo > y_hi:
            continue
        nbr_sub = _translate(nbr_poly)
        nbr_inside, nbr_dist_outside_m = _distance_outside_polygon_m(
            nbr_sub, (sub_h, sub_w), px_to_m=px_to_m,
        )
        if int(nbr_inside.sum()) == 0:
            continue

        ratio = nbr_dist_outside_m / (
            nbr_dist_outside_m + user_dist_outside_m + DIST_EPS_M
        )
        ratio = np.where(user_inside > 0, 1.0, ratio).astype(np.float32)
        conf_sub *= ratio

    # 3. Paste sub-window back into a full-size map (outside = 0).
    confidence = np.zeros((H, W), dtype=np.float32)
    confidence[y_lo:y_hi, x_lo:x_hi] = np.clip(conf_sub, 0.0, 1.0)
    return confidence


__all__ = [
    "build_confidence_field",
    "build_confidence_field_sub",
    "ConfidenceField",
    "DEFAULT_FALLOFF_RADIUS_M",
]
