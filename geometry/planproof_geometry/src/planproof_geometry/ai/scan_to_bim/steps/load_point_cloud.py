"""Step 2: Load point cloud from file into numpy arrays."""

from __future__ import annotations

from ..point_cloud_io import load_point_cloud, detect_format


class LoadPointCloudStep:
    name = "Load point cloud"

    def run(self, ctx, state):
        fmt = detect_format(state.input_path)
        ctx.log(f"Loading {fmt} point cloud...")

        points, colors = load_point_cloud(state.input_path)

        state.points = points
        state.colors = colors
        state.metrics["input_points_count"] = points.shape[0]
        state.metrics["has_colors"] = colors is not None

        ctx.log(
            f"Loaded {points.shape[0]:,} points"
            f"{' (with RGB)' if colors is not None else ''}"
        )
        return state
