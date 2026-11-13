"""Step 5: Project wall planes to 2D floor plan line segments."""

from __future__ import annotations

import numpy as np


class ExtractFloorPlanStep:
    name = "Extract floor plan"

    def run(self, ctx, state):
        wall_planes = state.wall_planes
        if not wall_planes:
            ctx.log("No wall planes detected, skipping floor plan extraction")
            state.wall_segments_2d = []
            return state

        points = state.points
        enforce_manhattan = state.config.get("enforce_manhattan", True)
        segments = []

        for i, wp in enumerate(wall_planes):
            inlier_mask = wp["inlier_mask"]
            wall_pts = points[inlier_mask]

            # Project to XY plane (ignore Z)
            xy = wall_pts[:, :2]

            if xy.shape[0] < 2:
                continue

            # PCA to find principal axis
            centroid_2d = xy.mean(axis=0)
            centered = xy - centroid_2d
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)

            # Major axis = eigenvector with largest eigenvalue
            major_axis = eigvecs[:, -1]

            # Project onto major axis to find extent
            projections = centered @ major_axis
            t_min, t_max = projections.min(), projections.max()

            a = centroid_2d + t_min * major_axis
            b = centroid_2d + t_max * major_axis

            # Manhattan snapping
            if enforce_manhattan:
                a, b = _snap_to_manhattan(a, b)

            length = np.linalg.norm(b - a)
            if length < 0.1:  # skip very short segments
                continue

            segments.append({
                "id": f"w{i + 1}",
                "a": (float(a[0]), float(a[1])),
                "b": (float(b[0]), float(b[1])),
                "kind": "wall",
                "plane_idx": i,
                "length_m": float(length),
                "inlier_count": wp["inlier_count"],
            })

        state.wall_segments_2d = segments
        state.metrics["wall_segments_count"] = len(segments)

        ctx.log(f"Extracted {len(segments)} wall segments from {len(wall_planes)} wall planes")
        return state


def _snap_to_manhattan(a: np.ndarray, b: np.ndarray) -> tuple:
    """Snap a line segment to the nearest axis-aligned direction (0/90/180/270)."""
    direction = b - a
    angle = np.arctan2(direction[1], direction[0])

    # Snap to nearest 90-degree increment
    snapped_angle = round(angle / (np.pi / 2)) * (np.pi / 2)

    length = np.linalg.norm(direction)
    snapped_dir = np.array([np.cos(snapped_angle), np.sin(snapped_angle)])

    midpoint = (a + b) / 2
    new_a = midpoint - (length / 2) * snapped_dir
    new_b = midpoint + (length / 2) * snapped_dir

    return new_a, new_b
