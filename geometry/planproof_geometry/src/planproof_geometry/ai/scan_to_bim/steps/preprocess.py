"""Step 3: Preprocess point cloud (downsample, denoise, align, center)."""

from __future__ import annotations

import numpy as np


class PreprocessCloudStep:
    name = "Preprocess cloud"

    def run(self, ctx, state):
        points = state.points
        colors = state.colors
        n_original = points.shape[0]

        config = state.config
        max_points = config.get("max_input_points", 2_000_000)

        # 1. Voxel downsampling
        voxel_size = config.get("voxel_size")
        if voxel_size is None:
            voxel_size = _adaptive_voxel_size(points, max_points)

        points, colors, indices = _voxel_downsample(points, colors, voxel_size)
        ctx.log(f"Voxel downsample (size={voxel_size:.3f}m): {n_original:,} -> {points.shape[0]:,}")

        # 2. Statistical outlier removal
        outlier_k = config.get("outlier_k", 20)
        outlier_std = config.get("outlier_std_ratio", 2.0)
        n_before_sor = points.shape[0]
        points, colors = _statistical_outlier_removal(points, colors, outlier_k, outlier_std)
        n_removed = n_before_sor - points.shape[0]
        if n_removed > 0:
            ctx.log(f"Outlier removal: {n_removed:,} points removed")

        # 3. Z-up normalization (ensure Z is the up-axis)
        points = _normalize_up_axis(points)

        # 4. Center at origin
        centroid = points.mean(axis=0)
        points = points - centroid

        state.points = points
        state.colors = colors
        state.metrics["points_after_preprocess"] = points.shape[0]
        state.metrics["voxel_size_m"] = voxel_size
        state.metrics["points_removed_outlier"] = n_removed

        return state


def _adaptive_voxel_size(points: np.ndarray, target_count: int) -> float:
    """Compute voxel size to reach approximately target_count points."""
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    diagonal = np.linalg.norm(bbox_max - bbox_min)

    n = points.shape[0]
    if n <= target_count:
        return 0.01  # minimal downsampling

    # Heuristic: voxel_size proportional to diagonal, scaled by density ratio
    ratio = n / target_count
    voxel_size = diagonal / (5000 / ratio)
    return float(np.clip(voxel_size, 0.01, 0.10))


def _voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray | None,
    voxel_size: float,
):
    """Voxel grid downsampling using open3d or numpy fallback."""
    try:
        return _voxel_downsample_o3d(points, colors, voxel_size)
    except Exception:
        return _voxel_downsample_numpy(points, colors, voxel_size)


def _voxel_downsample_o3d(points, colors, voxel_size):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    down = pcd.voxel_down_sample(voxel_size)
    new_points = np.asarray(down.points, dtype=np.float64)
    new_colors = np.asarray(down.colors, dtype=np.float32) if down.has_colors() else None
    return new_points, new_colors, None


def _voxel_downsample_numpy(points, colors, voxel_size):
    """Pure numpy voxel grid: keep one point per voxel cell."""
    quantized = np.floor(points / voxel_size).astype(np.int64)
    # Use structured array for unique
    key = quantized[:, 0] * 10_000_000 + quantized[:, 1] * 10_000 + quantized[:, 2]
    _, unique_idx = np.unique(key, return_index=True)
    unique_idx.sort()

    new_points = points[unique_idx]
    new_colors = colors[unique_idx] if colors is not None else None
    return new_points, new_colors, unique_idx


def _statistical_outlier_removal(
    points: np.ndarray,
    colors: np.ndarray | None,
    k: int,
    std_ratio: float,
):
    """Remove statistical outliers using open3d or numpy fallback."""
    try:
        return _sor_o3d(points, colors, k, std_ratio)
    except Exception:
        return _sor_numpy(points, colors, k, std_ratio)


def _sor_o3d(points, colors, k, std_ratio):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    clean, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std_ratio)
    new_points = np.asarray(clean.points, dtype=np.float64)
    new_colors = np.asarray(clean.colors, dtype=np.float32) if clean.has_colors() else None
    return new_points, new_colors


def _sor_numpy(points, colors, k, std_ratio):
    """Simple distance-based outlier removal (numpy only).

    For each point, compute mean distance to k nearest neighbors.
    Remove points where mean_dist > global_mean + std_ratio * global_std.
    """
    from scipy.spatial import cKDTree

    n = points.shape[0]
    if n < k + 1:
        return points, colors

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)  # includes self
    mean_dists = dists[:, 1:].mean(axis=1)  # exclude self-distance

    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_ratio * global_std

    mask = mean_dists < threshold
    new_points = points[mask]
    new_colors = colors[mask] if colors is not None else None
    return new_points, new_colors


def _normalize_up_axis(points: np.ndarray) -> np.ndarray:
    """Ensure Z is the up-axis.

    Uses the dominant horizontal plane normal to determine up direction.
    If already Z-up, returns unchanged.
    """
    # Quick heuristic: check Z-range vs X/Y-range
    ranges = points.max(axis=0) - points.min(axis=0)
    z_range = ranges[2]
    xy_max_range = max(ranges[0], ranges[1])

    # If Z-range is much smaller than XY (typical for buildings), Z is likely up
    if z_range < xy_max_range * 0.8:
        return points  # already Z-up

    # If Y-range is smallest, might be Y-up (common in some scan formats)
    if ranges[1] < min(ranges[0], ranges[2]) * 0.8:
        # Swap Y and Z
        return points[:, [0, 2, 1]]

    return points
