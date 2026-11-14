"""Snap polygon vertices onto the nearest parallel ink stroke.

After ``simplify`` and before ``axis_snap``, polygon edges are mostly
straight but can sit a few mm off the actual paper-ink wall stroke
because of rasterisation jitter, paired-outline asymmetry, or close-
kernel side effects. This snapper measures the distance from each
edge to the nearest ink pixel and moves the edge endpoints onto the
ink stroke when the geometry says it's safe to do so.

Algorithm (per polygon):

  1. Pre-compute ``dist_to_ink_px`` (distance transform of inverted
     ink mask) and its gradient ``(grad_x, grad_y)`` once per image.
  2. For each edge of the polygon:
       * Sample N points along the edge.
       * Read distance and gradient at each sample.
       * If the edge tangent is parallel to the local ink stroke
         (|tangent · gradient| < sin(parallel_tol_deg)) AND median
         distance ≤ snap_cap_m, take the median gradient direction as
         the "into ink" vector and the median distance as the offset.
       * Push that displacement onto both endpoints' move-bucket.
  3. For each vertex: average the displacements voted by its adjacent
     edges and apply the result.

Edges that don't satisfy the parallel + cap test contribute no
displacement to their endpoints. This keeps wall ends from being
yanked sideways onto a perpendicular stroke (door jamb shooting onto
the wall axis it sits next to).

Independent of ``axis_snap``: pixel_snap aligns to actual ink, axis_
snap aligns to globally consistent X/Y rails. Run pixel_snap first.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import cv2
import numpy as np


def precompute_ink_field(
    image_gray: np.ndarray,
    user_polygon_union_mask: np.ndarray | None = None,
    *,
    roi_dilate_px: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (dist_to_ink_px, grad_x, grad_y) on the source image.

    The grad fields are the spatial derivatives of the distance field
    (Sobel, normalised to unit length where the magnitude is non-zero).
    Direction points away from the nearest ink pixel.

    If ``user_polygon_union_mask`` is given, the Otsu ink mask is
    clipped to the dilated user-polygon ROI so furniture, dimension
    text and ornaments outside the labelled walls don't act as snap
    targets.
    """
    _, otsu = cv2.threshold(
        image_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    if user_polygon_union_mask is not None:
        if roi_dilate_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (roi_dilate_px, roi_dilate_px),
            )
            roi = cv2.dilate(user_polygon_union_mask, kernel, iterations=1)
        else:
            roi = user_polygon_union_mask
        otsu = cv2.bitwise_and(otsu, roi)
    inverted = (otsu == 0).astype(np.uint8)
    dist_px = cv2.distanceTransform(inverted, cv2.DIST_L2, 5).astype(np.float32)
    gx = cv2.Sobel(dist_px, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dist_px, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-6
    return dist_px, gx / mag, gy / mag


def pixel_snap_polygon(
    coords_m: Sequence[Tuple[float, float]],
    *,
    px_to_m: float,
    dist_to_ink_px: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    snap_cap_m: float = 0.010,
    parallel_tol_deg: float = 8.0,
    sample_step_px: float = 3.0,
    min_samples: int = 4,
) -> Tuple[list, dict]:
    """Snap one closed polygon's vertices onto nearby parallel ink.

    Returns the updated coords and a small info dict with edge-level
    diagnostics.
    """
    n = len(coords_m)
    if n < 4:
        return list(coords_m), {"edges": 0, "snapped": 0, "skipped": 0}

    H, W = dist_to_ink_px.shape[:2]
    inv_px = 1.0 / px_to_m if px_to_m > 0 else 1.0
    pts_px = np.asarray(
        [[x * inv_px, y * inv_px] for x, y in coords_m], dtype=np.float64,
    )
    closed = np.allclose(pts_px[0], pts_px[-1])
    work = pts_px[:-1].copy() if closed else pts_px.copy()
    m = len(work)
    if m < 3:
        return list(coords_m), {"edges": 0, "snapped": 0, "skipped": 0}

    snap_cap_px = snap_cap_m / max(px_to_m, 1e-9)
    sin_parallel = math.sin(math.radians(parallel_tol_deg))

    # accumulated displacement vectors per vertex, plus a count for
    # averaging (so a vertex shared between 2 edges that both vote for
    # a displacement gets the average, not the sum).
    disp = np.zeros_like(work)
    votes = np.zeros(m, dtype=np.int32)
    n_edges = 0
    n_snapped = 0
    n_skipped = 0

    for i in range(m):
        j = (i + 1) % m
        p0, p1 = work[i], work[j]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        seg_len_px = float(math.hypot(dx, dy))
        if seg_len_px < 1.0:
            continue
        n_edges += 1
        tangent = np.array([dx / seg_len_px, dy / seg_len_px])

        n_samp = max(min_samples, int(round(seg_len_px / sample_step_px)))
        # exclude exact endpoints to avoid double-counting at vertices
        ts = np.linspace(0.0, 1.0, n_samp + 2)[1:-1]
        sample_dists = []
        sample_gx = []
        sample_gy = []
        sample_dot = []
        for t in ts:
            x = p0[0] + t * dx
            y = p0[1] + t * dy
            xi = int(round(x))
            yi = int(round(y))
            if not (0 <= xi < W and 0 <= yi < H):
                continue
            d = float(dist_to_ink_px[yi, xi])
            ngx = float(grad_x[yi, xi])
            ngy = float(grad_y[yi, xi])
            sample_dists.append(d)
            sample_gx.append(ngx)
            sample_gy.append(ngy)
            # parallel check: tangent · gradient ≈ 0 → edge runs along ink
            sample_dot.append(abs(tangent[0] * ngx + tangent[1] * ngy))

        if not sample_dists:
            n_skipped += 1
            continue

        median_d = float(np.median(sample_dists))
        median_dot = float(np.median(sample_dot))
        if median_d > snap_cap_px or median_dot > sin_parallel:
            n_skipped += 1
            continue

        # Median gradient direction = the "away from ink" unit vector.
        # We move the edge by -median_d * gradient_dir to land on ink.
        med_gx = float(np.median(sample_gx))
        med_gy = float(np.median(sample_gy))
        gmag = math.hypot(med_gx, med_gy)
        if gmag < 1e-3:
            n_skipped += 1
            continue
        nrm_gx = med_gx / gmag
        nrm_gy = med_gy / gmag

        shift_x = -median_d * nrm_gx
        shift_y = -median_d * nrm_gy

        disp[i, 0] += shift_x
        disp[i, 1] += shift_y
        disp[j, 0] += shift_x
        disp[j, 1] += shift_y
        votes[i] += 1
        votes[j] += 1
        n_snapped += 1

    # Apply averaged displacements
    moved = work.copy()
    for k in range(m):
        if votes[k] == 0:
            continue
        moved[k, 0] += disp[k, 0] / votes[k]
        moved[k, 1] += disp[k, 1] / votes[k]

    out = [(float(p[0]) * px_to_m, float(p[1]) * px_to_m) for p in moved]
    if closed:
        out.append(out[0])
    return out, {
        "edges": n_edges,
        "snapped": n_snapped,
        "skipped": n_skipped,
    }


__all__ = ["pixel_snap_polygon", "precompute_ink_field"]
