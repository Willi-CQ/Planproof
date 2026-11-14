"""Step 7: Detect openings (doors/windows) from density gaps in wall planes."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class DetectOpeningsStep:
    name = "Detect openings"
    soft_fail = True

    def run(self, ctx, state):
        wall_planes = state.wall_planes
        points = state.points
        floor_plane = state.floor_plane

        if not wall_planes or points is None:
            ctx.log("No wall planes available, skipping opening detection")
            state.openings = []
            return state

        config = state.config
        min_width = config.get("min_opening_width_m", 0.5)
        max_width = config.get("max_opening_width_m", 3.0)

        floor_z = floor_plane["centroid_z"] if floor_plane else points[:, 2].min()
        room_height = state.metrics.get("detected_room_height_m", 2.80)

        openings = []
        for i, wp in enumerate(wall_planes):
            wall_openings = _detect_openings_in_wall(
                points, wp, floor_z, room_height, min_width, max_width, wall_idx=i
            )
            openings.extend(wall_openings)

        state.openings = openings
        state.metrics["openings_detected"] = len(openings)

        doors = [o for o in openings if o["type"] == "door"]
        windows = [o for o in openings if o["type"] == "window"]
        state.metrics["doors_detected"] = len(doors)
        state.metrics["windows_detected"] = len(windows)

        ctx.log(f"Detected {len(openings)} openings ({len(doors)} doors, {len(windows)} windows)")
        return state


def _detect_openings_in_wall(
    points: np.ndarray,
    wall_plane: Dict,
    floor_z: float,
    room_height: float,
    min_width: float,
    max_width: float,
    wall_idx: int,
) -> List[Dict]:
    """Detect openings by analyzing point density along wall."""
    inlier_mask = wall_plane["inlier_mask"]
    wall_pts = points[inlier_mask]

    if wall_pts.shape[0] < 50:
        return []

    # Project wall points onto wall's major axis (horizontal extent)
    normal = np.array(wall_plane["normal"])
    # Wall direction: perpendicular to normal in XY plane
    wall_dir = np.array([-normal[1], normal[0], 0.0])
    wall_dir_norm = np.linalg.norm(wall_dir[:2])
    if wall_dir_norm < 1e-6:
        return []
    wall_dir = wall_dir / wall_dir_norm

    centroid = np.array(wall_plane["centroid"])
    relative = wall_pts - centroid

    # Project along wall direction (horizontal) and Z (vertical)
    h_proj = relative @ wall_dir  # horizontal position along wall
    z_vals = wall_pts[:, 2]  # absolute Z

    # Create density histogram along wall
    h_min, h_max = h_proj.min(), h_proj.max()
    wall_length = h_max - h_min
    if wall_length < min_width:
        return []

    n_bins = max(20, int(wall_length / 0.05))  # ~5cm bins
    hist, bin_edges = np.histogram(h_proj, bins=n_bins)

    # Normalize by expected density
    median_density = np.median(hist[hist > 0]) if np.any(hist > 0) else 1
    if median_density == 0:
        return []

    # Find gaps (bins with significantly lower density)
    gap_threshold = median_density * 0.3
    is_gap = hist < gap_threshold

    # Group consecutive gap bins into opening candidates
    openings = []
    gap_start = None
    for j in range(len(is_gap)):
        if is_gap[j] and gap_start is None:
            gap_start = j
        elif not is_gap[j] and gap_start is not None:
            gap_end = j
            opening_start = bin_edges[gap_start]
            opening_end = bin_edges[gap_end]
            width = opening_end - opening_start

            if min_width <= width <= max_width:
                # Determine opening type by analyzing Z distribution in the gap
                gap_mask = (h_proj >= opening_start) & (h_proj <= opening_end)
                if gap_mask.sum() > 0:
                    gap_z = z_vals[gap_mask]
                    base_height = gap_z.min() - floor_z
                    top_height = gap_z.max() - floor_z
                else:
                    base_height = 0.0
                    top_height = room_height * 0.8

                opening_type = _classify_opening(base_height, top_height, width, room_height)

                # Compute 3D position
                center_h = (opening_start + opening_end) / 2
                center_3d = centroid + center_h * wall_dir

                openings.append({
                    "id": f"opening_w{wall_idx}_{len(openings) + 1}",
                    "type": opening_type,
                    "parent_wall_idx": wall_idx,
                    "width_m": round(float(width), 2),
                    "height_m": round(float(top_height - base_height), 2),
                    "base_height_m": round(float(base_height), 2),
                    "center": (float(center_3d[0]), float(center_3d[1])),
                    "center_z": float(floor_z + (base_height + top_height) / 2),
                })

            gap_start = None

    # Handle gap at end of wall
    if gap_start is not None:
        opening_start = bin_edges[gap_start]
        opening_end = bin_edges[-1]
        width = opening_end - opening_start
        if min_width <= width <= max_width:
            center_h = (opening_start + opening_end) / 2
            center_3d = centroid + center_h * wall_dir
            openings.append({
                "id": f"opening_w{wall_idx}_{len(openings) + 1}",
                "type": "door",  # default for end-of-wall gaps
                "parent_wall_idx": wall_idx,
                "width_m": round(float(width), 2),
                "height_m": round(float(room_height * 0.8), 2),
                "base_height_m": 0.0,
                "center": (float(center_3d[0]), float(center_3d[1])),
                "center_z": float(floor_z + room_height * 0.4),
            })

    return openings


def _classify_opening(base_height: float, top_height: float, width: float, room_height: float) -> str:
    """Classify opening as door or window based on geometry."""
    # Doors: base near floor (< 0.3m), height > 1.8m
    # Windows: base above floor (> 0.5m), usually smaller height
    if base_height < 0.3 and (top_height - base_height) > 1.5:
        return "door"
    if base_height > 0.5:
        return "window"
    if width > 1.5 and base_height < 0.2:
        return "door"  # wide opening at floor = door/archway
    return "window"  # default
