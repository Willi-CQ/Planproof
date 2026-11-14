"""Pure-numpy RANSAC plane segmentation (fallback when open3d unavailable)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PlaneResult:
    """A single detected plane."""
    normal: np.ndarray          # (3,) unit normal
    d: float                    # signed distance from origin (ax+by+cz+d=0)
    inlier_mask: np.ndarray     # (N,) bool mask over original point set
    inlier_count: int
    centroid: np.ndarray        # (3,) mean of inliers
    fit_quality: float          # fraction of inliers within threshold


def fit_plane_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.03,
    max_iterations: int = 1000,
    min_inliers: int = 500,
) -> Optional[PlaneResult]:
    """Fit a single plane to points via RANSAC.

    Args:
        points: (N, 3) array.
        distance_threshold: max distance from plane to count as inlier (metres).
        max_iterations: RANSAC iterations.
        min_inliers: minimum inlier count to accept the plane.

    Returns:
        PlaneResult or None if no plane meets the threshold.
    """
    n = points.shape[0]
    if n < 3:
        return None

    best_inlier_count = 0
    best_normal = None
    best_d = 0.0

    rng = np.random.default_rng(42)

    for _ in range(max_iterations):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx[0]], points[idx[1]], points[idx[2]]

        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal = normal / norm_len
        d = -np.dot(normal, p0)

        dists = np.abs(points @ normal + d)
        inlier_count = int(np.sum(dists < distance_threshold))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_normal = normal
            best_d = d

    if best_normal is None or best_inlier_count < min_inliers:
        return None

    # Refine with all inliers (least-squares refit)
    dists = np.abs(points @ best_normal + best_d)
    inlier_mask = dists < distance_threshold
    inlier_pts = points[inlier_mask]

    # SVD refit on inlier set
    centroid = inlier_pts.mean(axis=0)
    centered = inlier_pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    refined_normal = vh[2]  # smallest singular value direction
    refined_d = -np.dot(refined_normal, centroid)

    # Recompute inliers with refined plane
    dists = np.abs(points @ refined_normal + refined_d)
    inlier_mask = dists < distance_threshold
    inlier_count = int(inlier_mask.sum())

    return PlaneResult(
        normal=refined_normal,
        d=refined_d,
        inlier_mask=inlier_mask,
        inlier_count=inlier_count,
        centroid=points[inlier_mask].mean(axis=0),
        fit_quality=inlier_count / n,
    )


def segment_planes(
    points: np.ndarray,
    distance_threshold: float = 0.03,
    max_iterations: int = 1000,
    min_plane_points: int = 500,
    max_planes: int = 50,
) -> List[PlaneResult]:
    """Iteratively detect planes via RANSAC.

    Removes inliers after each plane detection and continues
    until no more planes can be found.

    Returns:
        List of PlaneResult, ordered by inlier count (descending).
    """
    remaining_mask = np.ones(points.shape[0], dtype=bool)
    results: List[PlaneResult] = []

    for _ in range(max_planes):
        remaining = points[remaining_mask]
        if remaining.shape[0] < min_plane_points:
            break

        plane = fit_plane_ransac(
            remaining,
            distance_threshold=distance_threshold,
            max_iterations=max_iterations,
            min_inliers=min_plane_points,
        )
        if plane is None:
            break

        # Map inlier mask back to original indices
        remaining_indices = np.where(remaining_mask)[0]
        full_mask = np.zeros(points.shape[0], dtype=bool)
        full_mask[remaining_indices[plane.inlier_mask]] = True

        results.append(PlaneResult(
            normal=plane.normal,
            d=plane.d,
            inlier_mask=full_mask,
            inlier_count=int(full_mask.sum()),
            centroid=points[full_mask].mean(axis=0),
            fit_quality=plane.fit_quality,
        ))

        # Remove inliers from remaining points
        remaining_mask &= ~full_mask

    results.sort(key=lambda p: p.inlier_count, reverse=True)
    return results


def classify_plane(normal: np.ndarray) -> str:
    """Classify a plane as horizontal or vertical based on its normal.

    Returns:
        'horizontal' if |normal.z| > 0.8, 'vertical' if |normal.z| < 0.3,
        'oblique' otherwise.
    """
    abs_z = abs(normal[2])
    if abs_z > 0.8:
        return "horizontal"
    if abs_z < 0.3:
        return "vertical"
    return "oblique"
