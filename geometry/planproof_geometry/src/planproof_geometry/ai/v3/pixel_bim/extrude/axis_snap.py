"""Snap polygon edges to the dominant axes of the floor plan.

After Douglas-Peucker simplification a wall polygon's edges are mostly
straight, but tiny pixel jitter still produces near-horizontal edges
that drift by ~1 cm and near-vertical edges that wobble between
columns. Visually that reads as wavy walls.

This snapper finds horizontal/vertical edges in the polygon and
collapses Y-coordinates of horizontal-edge endpoints onto a shared
median, and X-coordinates of vertical-edge endpoints onto a shared
median, using single-linkage clustering with a metre-space tolerance.

Non-axial edges (diagonal walls, curved Veranda outlines) are left
untouched.

Plan-rotation: this module assumes the floor plan is drawn axis-aligned
(walls within ±5° of horizontal/vertical). Matterport, ArchiCAD and
most CAD exports satisfy this. For rotated plans, callers should rotate
the polygons into plan-aligned space before snapping and rotate back
after.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


def _cluster_values(values: Sequence[float], tol: float) -> List[List[int]]:
    """Single-linkage 1-D clustering: indices of values whose pairwise
    distance is <= ``tol`` end up in the same cluster.

    Returns a list of clusters, each a list of indices into ``values``.
    """
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    clusters: List[List[int]] = [[order[0]]]
    for idx in order[1:]:
        if values[idx] - values[clusters[-1][-1]] <= tol:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return clusters


def axis_snap_polygon(
    coords: Sequence[Tuple[float, float]],
    *,
    angle_deg: float = 5.0,
    cluster_tol_m: float = 0.05,
) -> List[Tuple[float, float]]:
    """Snap near-axial edges of a closed polygon to shared X/Y rails.

    Parameters
    ----------
    coords
        Closed polygon coordinates in metres. First and last coord may
        coincide (shapely-style); both are kept.
    angle_deg
        An edge is considered near-horizontal if its angle to the X axis
        is within ``angle_deg`` degrees, and near-vertical analogously.
    cluster_tol_m
        Two horizontal edges whose mean Y differs by less than this are
        snapped onto the same Y rail. Same logic for vertical edges and
        their X rails. 5 cm matches typical drafting jitter without
        merging genuinely distinct walls.
    """
    n = len(coords)
    if n < 4:
        return [tuple(p) for p in coords]

    pts = np.asarray(coords, dtype=np.float64).copy()
    closed = np.allclose(pts[0], pts[-1])
    work = pts[:-1] if closed else pts
    m = len(work)
    if m < 3:
        return [tuple(p) for p in coords]

    angle_rad = math.radians(angle_deg)
    tan_thresh = math.tan(angle_rad)

    # Classify each edge i -> i+1
    horiz_edges: List[Tuple[int, int, float]] = []  # (i, j, y_mean)
    vert_edges: List[Tuple[int, int, float]] = []   # (i, j, x_mean)
    for i in range(m):
        j = (i + 1) % m
        dx = work[j, 0] - work[i, 0]
        dy = work[j, 1] - work[i, 1]
        if dx == 0.0 and dy == 0.0:
            continue
        if abs(dx) >= abs(dy):
            # near-horizontal candidate: |dy/dx| <= tan(angle_deg)
            if abs(dy) <= abs(dx) * tan_thresh:
                horiz_edges.append((i, j, 0.5 * (work[i, 1] + work[j, 1])))
        else:
            # near-vertical candidate: |dx/dy| <= tan(angle_deg)
            if abs(dx) <= abs(dy) * tan_thresh:
                vert_edges.append((i, j, 0.5 * (work[i, 0] + work[j, 0])))

    # Cluster horizontal edges by their mean Y → snap endpoints onto cluster median
    if horiz_edges:
        ys = [e[2] for e in horiz_edges]
        clusters = _cluster_values(ys, cluster_tol_m)
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            cluster_y = float(np.median([ys[k] for k in cluster]))
            endpoints: set[int] = set()
            for k in cluster:
                i, j, _ = horiz_edges[k]
                endpoints.add(i)
                endpoints.add(j)
            for idx in endpoints:
                work[idx, 1] = cluster_y

    if vert_edges:
        xs = [e[2] for e in vert_edges]
        clusters = _cluster_values(xs, cluster_tol_m)
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            cluster_x = float(np.median([xs[k] for k in cluster]))
            endpoints = set()
            for k in cluster:
                i, j, _ = vert_edges[k]
                endpoints.add(i)
                endpoints.add(j)
            for idx in endpoints:
                work[idx, 0] = cluster_x

    out = [(float(p[0]), float(p[1])) for p in work]
    if closed:
        out.append(out[0])
    return out


__all__ = ["axis_snap_polygon"]
