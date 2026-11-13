"""Step 6: Detect rooms from wall segments using connectivity analysis."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class DetectRoomsStep:
    name = "Detect rooms"
    soft_fail = True

    def run(self, ctx, state):
        segments = state.wall_segments_2d
        if not segments or len(segments) < 3:
            ctx.log("Too few wall segments for room detection")
            state.rooms = []
            return state

        config = state.config
        snap_dist = config.get("snap_distance_m", 0.15)
        min_area = config.get("min_room_area_m2", 3.0)
        max_area = config.get("max_room_area_m2", 500.0)

        rooms = _detect_rooms_from_segments(segments, snap_dist, min_area, max_area)

        state.rooms = rooms
        state.metrics["rooms_detected"] = len(rooms)

        if rooms:
            areas = [r["area_m2"] for r in rooms]
            state.metrics["total_floor_area_m2"] = round(sum(areas), 1)
            ctx.log(f"Detected {len(rooms)} rooms (total {sum(areas):.1f} m²)")
        else:
            ctx.log("No enclosed rooms detected")
            state.quality_warnings.append("no_rooms_detected")

        return state


def _detect_rooms_from_segments(
    segments: List[Dict],
    snap_dist: float,
    min_area: float,
    max_area: float,
) -> List[Dict]:
    """Detect rooms by finding enclosed polygons from wall segments."""
    # Collect all endpoints
    endpoints = []
    for seg in segments:
        endpoints.append(np.array(seg["a"]))
        endpoints.append(np.array(seg["b"]))

    if not endpoints:
        return []

    points = np.array(endpoints)

    # Snap nearby endpoints together
    merged_points, point_map = _snap_endpoints(points, snap_dist)

    # Build adjacency graph
    graph: Dict[int, List[int]] = {}
    for i, seg in enumerate(segments):
        a_idx = point_map[i * 2]
        b_idx = point_map[i * 2 + 1]
        if a_idx == b_idx:
            continue
        graph.setdefault(a_idx, []).append(b_idx)
        graph.setdefault(b_idx, []).append(a_idx)

    # Find minimal cycles (rooms) using planar face detection
    cycles = _find_minimal_cycles(graph, merged_points)

    rooms = []
    for i, cycle_indices in enumerate(cycles):
        polygon = merged_points[cycle_indices]
        area = _polygon_area(polygon)

        if area < min_area or area > max_area:
            continue

        centroid = polygon.mean(axis=0)

        # Find which wall segments belong to this room
        wall_ids = _find_room_walls(cycle_indices, segments, point_map)

        rooms.append({
            "id": f"room_{i + 1}",
            "polygon": polygon.tolist(),
            "area_m2": round(float(area), 2),
            "centroid": (float(centroid[0]), float(centroid[1])),
            "wall_ids": wall_ids,
            "n_walls": len(wall_ids),
        })

    return rooms


def _snap_endpoints(points: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Merge nearby points and return merged points + index mapping."""
    n = points.shape[0]
    labels = np.arange(n)

    # Union-find style merging
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) < threshold:
                # Merge j into i's group
                root_i = _find_root(labels, i)
                root_j = _find_root(labels, j)
                if root_i != root_j:
                    labels[root_j] = root_i

    # Resolve all labels
    for i in range(n):
        labels[i] = _find_root(labels, i)

    # Compute merged points (centroid of each group)
    unique_labels = np.unique(labels)
    label_to_new_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}

    merged = np.zeros((len(unique_labels), 2))
    counts = np.zeros(len(unique_labels))
    for i in range(n):
        new_idx = label_to_new_idx[labels[i]]
        merged[new_idx] += points[i]
        counts[new_idx] += 1

    merged = merged / counts[:, None]
    point_map = np.array([label_to_new_idx[labels[i]] for i in range(n)])

    return merged, point_map


def _find_root(labels, i):
    while labels[i] != i:
        labels[i] = labels[labels[i]]
        i = labels[i]
    return i


def _find_minimal_cycles(graph: Dict[int, List[int]], points: np.ndarray) -> List[List[int]]:
    """Find minimal cycles in a planar graph using angular ordering."""
    cycles = []
    visited_edges = set()

    for start_node in graph:
        neighbors = graph.get(start_node, [])
        for next_node in neighbors:
            edge = (start_node, next_node)
            if edge in visited_edges:
                continue

            cycle = _trace_cycle(graph, points, start_node, next_node, visited_edges)
            if cycle and len(cycle) >= 3:
                cycles.append(cycle)

    return cycles


def _trace_cycle(
    graph: Dict[int, List[int]],
    points: np.ndarray,
    start: int,
    second: int,
    visited_edges: set,
) -> List[int]:
    """Trace a minimal cycle by always turning left (counter-clockwise)."""
    cycle = [start]
    prev = start
    current = second
    max_steps = 20  # prevent infinite loops

    for _ in range(max_steps):
        visited_edges.add((prev, current))
        cycle.append(current)

        if current == start:
            return cycle[:-1]  # complete cycle

        neighbors = graph.get(current, [])
        if not neighbors:
            return []

        # Choose next node by smallest CCW angle from incoming direction
        incoming = points[prev] - points[current]
        incoming_angle = np.arctan2(incoming[1], incoming[0])

        best_next = None
        best_angle_diff = float("inf")

        for nb in neighbors:
            if nb == prev and len(neighbors) > 1:
                continue
            outgoing = points[nb] - points[current]
            outgoing_angle = np.arctan2(outgoing[1], outgoing[0])
            diff = (outgoing_angle - incoming_angle) % (2 * np.pi)
            if diff < best_angle_diff:
                best_angle_diff = diff
                best_next = nb

        if best_next is None:
            return []

        prev = current
        current = best_next

    return []  # exceeded max steps


def _polygon_area(polygon: np.ndarray) -> float:
    """Compute area of a 2D polygon using the shoelace formula."""
    n = len(polygon)
    if n < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return abs(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1] + x[-1] * y[0] - x[0] * y[-1]))


def _find_room_walls(cycle_indices: List[int], segments: List[Dict], point_map: np.ndarray) -> List[str]:
    """Find wall segment IDs that form the room boundary."""
    cycle_set = set(cycle_indices)
    wall_ids = []
    for i, seg in enumerate(segments):
        a_idx = point_map[i * 2]
        b_idx = point_map[i * 2 + 1]
        if a_idx in cycle_set and b_idx in cycle_set:
            wall_ids.append(seg["id"])
    return wall_ids
