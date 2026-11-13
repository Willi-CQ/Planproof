"""Step 8: Validate and estimate scale from detected geometry."""

from __future__ import annotations

import numpy as np


class EstimateScaleStep:
    name = "Estimate scale"
    soft_fail = True

    def run(self, ctx, state):
        points = state.points
        floor_plane = state.floor_plane
        ceiling_plane = state.ceiling_plane

        if points is None:
            return state

        # Bounding box
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        bbox_size = bbox_max - bbox_min
        diagonal = float(np.linalg.norm(bbox_size))

        state.metrics["bbox_x_m"] = round(float(bbox_size[0]), 2)
        state.metrics["bbox_y_m"] = round(float(bbox_size[1]), 2)
        state.metrics["bbox_z_m"] = round(float(bbox_size[2]), 2)
        state.metrics["bbox_diagonal_m"] = round(diagonal, 2)

        # Room height from floor/ceiling
        room_height = None
        if floor_plane and ceiling_plane:
            room_height = ceiling_plane["centroid_z"] - floor_plane["centroid_z"]

        # Validate room height
        height_valid = True
        if room_height is not None:
            if room_height < 2.0:
                state.quality_warnings.append(
                    f"Room height {room_height:.2f}m seems too low (< 2.0m)"
                )
                height_valid = False
            elif room_height > 5.0:
                state.quality_warnings.append(
                    f"Room height {room_height:.2f}m seems too high (> 5.0m)"
                )
                height_valid = False

        # Auto-scale detection
        # If heights/dimensions seem wildly off, check if data is in mm or cm
        scale_factor = 1.0
        if room_height is not None and room_height > 100:
            # Likely millimetres
            scale_factor = 0.001
            ctx.log(f"Auto-detected mm scale, converting (height was {room_height:.0f})")
        elif room_height is not None and room_height > 10:
            # Likely centimetres
            scale_factor = 0.01
            ctx.log(f"Auto-detected cm scale, converting (height was {room_height:.1f})")

        if scale_factor != 1.0:
            state.points = points * scale_factor
            if state.colors is not None:
                pass  # colors don't change

            # Recompute metrics
            room_height *= scale_factor
            bbox_size *= scale_factor
            diagonal *= scale_factor

            # Update plane centroids
            if floor_plane:
                floor_plane["centroid_z"] *= scale_factor
            if ceiling_plane:
                ceiling_plane["centroid_z"] *= scale_factor
            for wp in (state.wall_planes or []):
                wp["centroid_z"] *= scale_factor
                wp["centroid"] = [c * scale_factor for c in wp["centroid"]]

            # Update wall segments
            for seg in (state.wall_segments_2d or []):
                seg["a"] = (seg["a"][0] * scale_factor, seg["a"][1] * scale_factor)
                seg["b"] = (seg["b"][0] * scale_factor, seg["b"][1] * scale_factor)
                seg["length_m"] *= scale_factor

            state.metrics["auto_scale_factor"] = scale_factor

        # Final room height
        if room_height is not None and height_valid:
            state.metrics["room_height_m"] = round(room_height, 2)
        else:
            default_h = state.config.get("default_room_height_m", 2.80)
            state.metrics["room_height_m"] = default_h
            if room_height is None:
                ctx.log(f"No floor/ceiling detected, using default height {default_h}m")

        # Floor area estimate
        floor_area = float(bbox_size[0] * bbox_size[1])
        state.metrics["floor_area_m2"] = round(floor_area, 1)

        # Plausibility check
        if diagonal < 1.0:
            state.quality_warnings.append(f"Very small scan (diagonal {diagonal:.2f}m)")
        elif diagonal > 200.0:
            state.quality_warnings.append(f"Very large scan (diagonal {diagonal:.1f}m)")

        ctx.log(
            f"Scale: {bbox_size[0]:.1f} x {bbox_size[1]:.1f} x {bbox_size[2]:.1f}m, "
            f"diagonal {diagonal:.1f}m, "
            f"height {state.metrics['room_height_m']}m"
        )
        return state
