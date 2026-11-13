"""Wall mask to WallGraph via medial_axis + sknw.

This module implements the Label-Magnet v3 core: derive a wall
junction graph directly from a per-class pixel mask (wall union),
skipping all geometric heuristics (ExtendWalls, Heal, SnapEndpoints).

Pipeline:

    manual_wall_mask (H, W uint8)
            │
    1. medial_axis(mask, return_distance=True)
            ├── 1-pixel skeleton
            └── per-pixel distance-to-boundary
            │
    2. sknw.build_sknw(skeleton)
            │
            └── NetworkX MultiGraph with nodes={endpoint, junction}
                and edges={pixel path between nodes}
            │
    3. Extract WallEdges:
            - endpoint/junction coords → WallNode (px → metre)
            - edge thickness = 2 × median(dist_along_path)
            │
    4. Junction re-resolve (optional):
            - Replace T/X pixel-approximated junction positions with the
              exact intersection point of adjacent edge centerlines
            - Fixes 45°-bleed artefact of Zhang-Suen skeleton at angles
            │
    5. Manhattan snap (opt-in):
            - If ≥ dominance_threshold of edges are within angle_tol of
              0°/90°, snap those edges to the exact axis
            - Diagonal walls preserved (not snapped) unless user opts in

Research backing:
    - Cloud2BIM (arXiv 2503.11498, 2025) — wall histograms + axis voting
    - FloorSAM (arXiv 2509.15750, 2025) — mask → skeleton → graph
    - Raster-to-Graph (CGF 2024) — junction + edge from raster
    - skimage.morphology.medial_axis — 1-pixel thinning with distance map
    - sknw — skeleton → NetworkX graph
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from planproof_geometry.domain.wall_graph import WallGraph

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Polygon rasterisation
# --------------------------------------------------------------------------

def rasterize_polygon(
    mask: np.ndarray,
    points_px: List[Tuple[float, float]],
    *,
    value: int = 1,
) -> None:
    """Rasterise a polygon (in pixel coords) onto an existing uint8 mask.

    Mutates ``mask`` in place. Uses OpenCV fillPoly — robust with
    non-convex shapes and zero-area degeneracies.
    """
    import cv2
    if len(points_px) < 3:
        return
    pts = np.array([[int(round(x)), int(round(y))] for x, y in points_px],
                   dtype=np.int32)
    cv2.fillPoly(mask, [pts], value)


def rasterize_bbox(
    mask: np.ndarray,
    bbox_px: Tuple[float, float, float, float],
    *,
    value: int = 1,
) -> None:
    """Rasterise an axis-aligned bbox (x1, y1, x2, y2) onto a uint8 mask."""
    x1, y1, x2, y2 = bbox_px
    x1 = max(0, int(round(min(x1, x2))))
    y1 = max(0, int(round(min(y1, y2))))
    x2 = min(mask.shape[1], int(round(max(x1, x2))))
    y2 = min(mask.shape[0], int(round(max(y1, y2))))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = value


# --------------------------------------------------------------------------
# medial_axis + sknw → WallGraph
# --------------------------------------------------------------------------

def wall_mask_to_graph(
    mask: np.ndarray,
    *,
    px_to_m: float,
    flip_y: bool = False,
    page_height_px: Optional[int] = None,
    wall_type_map: Optional[np.ndarray] = None,
    min_edge_length_px: float = 5.0,
    close_kernel_px: int = 5,
    prune_spur_ratio: float = 0.5,
    junction_resolve: bool = True,
    manhattan_snap: bool = True,
    manhattan_angle_tol_deg: float = 5.0,
    manhattan_dominance_threshold: float = 0.70,
) -> WallGraph:
    """Build a WallGraph from a 2D wall mask.

    Parameters
    ----------
    mask
        2D uint8 array, non-zero pixels are wall, zero is background.
        Should be in pipeline pixel resolution.
    px_to_m
        Pipeline pixel → metre factor.
    flip_y
        If True, convert y_px to y_m via ``page_height_px - y_px`` so the
        output is y-up (matches the legacy SceneModel convention). When
        False, y just gets multiplied by px_to_m (y-down).
    page_height_px
        Height of the source image in pixels; required if ``flip_y`` is
        True.
    wall_type_map
        Optional 2D uint8 array same shape as ``mask``. Pixel value 1 =
        interior, value 2 = exterior. Per edge, the majority pixel
        classification along its skeleton path is taken as the edge
        ``wall_type``. If ``None``, all edges default to interior.
    min_edge_length_px
        Drop edges whose skeleton path is shorter than this (noise).
    junction_resolve
        If True, re-compute T/X junction node positions as the exact
        intersection of adjacent edge centerlines instead of using the
        skeleton-pixel position. Eliminates Zhang-Suen bleeding at
        diagonals.
    manhattan_snap
        If True AND ``manhattan_dominance_threshold`` of edges are
        within ``manhattan_angle_tol_deg`` of 0°/90°, snap those
        edges exactly onto the axis. Diagonal walls outside tolerance
        are preserved.

    Returns
    -------
    WallGraph with edges, nodes, and per-edge thickness derived from
    the medial-axis distance transform.
    """
    try:
        from skimage.morphology import medial_axis
        import sknw
    except ImportError as e:
        raise ImportError(
            "wall_mask_to_graph requires scikit-image and sknw "
            "(`uv add --project packages/planproof_geometry sknw`)"
        ) from e

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    if mask.sum() == 0:
        logger.warning("wall_mask_to_graph: mask is empty")
        return WallGraph()

    # Ensure binary (medial_axis accepts bool or 0/non-zero)
    bin_mask = (mask > 0).astype(np.uint8)

    # 0. Morphological close to bridge 1-2 px gaps at polygon corners
    #    where two wall polygons meet. Without this, sknw picks up
    #    spurious branches at every seam, fragmenting the skeleton.
    if close_kernel_px and close_kernel_px >= 3:
        import cv2
        kern = cv2.getStructuringElement(cv2.MORPH_RECT,
                                         (close_kernel_px, close_kernel_px))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kern)

    # 1. Medial axis + distance transform. ``dist`` is Euclidean distance
    #    from each foreground pixel to the nearest background pixel, so
    #    thickness_m ≈ 2 * dist * px_to_m on skeleton pixels.
    skel, dist = medial_axis(bin_mask.astype(bool), return_distance=True)

    # 2. Build graph from skeleton. multi=False collapses parallel edges.
    g_nx = sknw.build_sknw(skel.astype(np.uint16), multi=False)

    # 3. Build our WallGraph.
    graph = WallGraph()

    def _to_m(x_px: float, y_px: float) -> Tuple[float, float]:
        x_m = float(x_px) * px_to_m
        if flip_y:
            if page_height_px is None:
                raise ValueError("flip_y=True requires page_height_px")
            y_m = (page_height_px - float(y_px)) * px_to_m
        else:
            y_m = float(y_px) * px_to_m
        return (x_m, y_m)

    sk_to_wall_node: Dict[int, str] = {}
    for sk_id, data in g_nx.nodes(data=True):
        o = data.get("o")
        if o is None:
            continue
        # sknw stores node coords as (y, x)
        y_px, x_px = float(o[0]), float(o[1])
        xy_m = _to_m(x_px, y_px)
        node = graph.add_node(xy_m)
        sk_to_wall_node[sk_id] = node.id

    edges_built = 0
    edges_skipped_short = 0
    for u, v, data in g_nx.edges(data=True):
        pts = data.get("pts")
        if pts is None or len(pts) < 2:
            continue
        # Arc length in pixels along the skeleton path
        diffs = np.diff(pts.astype(np.float32), axis=0)
        arc_length_px = float(np.sqrt((diffs * diffs).sum(axis=1)).sum())
        if arc_length_px < min_edge_length_px:
            edges_skipped_short += 1
            continue

        # Per-pixel distance → per-pixel thickness. Use the median to be
        # robust against skeleton tips where distance approaches 0.
        ys = pts[:, 0].astype(np.int32)
        xs = pts[:, 1].astype(np.int32)
        # Clip to image bounds defensively
        ys = np.clip(ys, 0, dist.shape[0] - 1)
        xs = np.clip(xs, 0, dist.shape[1] - 1)
        dist_along = dist[ys, xs]
        if dist_along.size == 0:
            continue
        thickness_px = 2.0 * float(np.median(dist_along))
        thickness_m = thickness_px * px_to_m

        node_a = sk_to_wall_node.get(u)
        node_b = sk_to_wall_node.get(v)
        if node_a is None or node_b is None or node_a == node_b:
            continue

        # Classify wall_type from wall_type_map if provided
        wall_type = "interior"
        if wall_type_map is not None:
            classes = wall_type_map[ys, xs]
            n_ext = int((classes == 2).sum())
            n_int = int((classes == 1).sum())
            if n_ext > n_int:
                wall_type = "exterior"

        edge = graph.add_edge(
            node_a, node_b,
            thickness_m=thickness_m,
            wall_type=wall_type,
            source="manual",
            confidence=1.0,
            attrs={
                "skeleton_arc_length_px": round(arc_length_px, 2),
                "thickness_px_median": round(thickness_px, 2),
            },
        )
        edges_built += 1

    graph.refresh_node_kinds()
    graph.prune_orphan_nodes()

    # 3b. Spur pruning — degree-1 stubs shorter than
    # ``prune_spur_ratio × median_thickness`` are almost always skeleton
    # artefacts at concave corners (Zhang-Suen bias). Remove them.
    # Repeat until no more spurs are found so chained spurs collapse.
    pruned = 0
    if prune_spur_ratio > 0 and graph.edges:
        thick_px = [
            float(e.thickness_m) / px_to_m
            for e in graph.edges.values() if e.thickness_m > 0
        ]
        median_thick_px = float(np.median(thick_px)) if thick_px else 0.0
        min_spur_len_px = median_thick_px * prune_spur_ratio
        if min_spur_len_px > 0:
            while True:
                to_remove = []
                for e in list(graph.edges.values()):
                    deg_a = graph.node_degree(e.node_a)
                    deg_b = graph.node_degree(e.node_b)
                    is_spur = deg_a == 1 or deg_b == 1
                    arc_px = float(e.attrs.get("skeleton_arc_length_px") or 0.0)
                    if is_spur and arc_px < min_spur_len_px:
                        to_remove.append(e.id)
                if not to_remove:
                    break
                for eid in to_remove:
                    graph.remove_edge(eid)
                pruned += len(to_remove)
            graph.prune_orphan_nodes()
            graph.refresh_node_kinds()

    logger.info(
        "wall_mask_to_graph: %d edges, %d nodes (skipped short: %d, spurs pruned: %d)",
        len(graph.edges), len(graph.nodes), edges_skipped_short, pruned,
    )

    # 4. Junction re-resolve (disabled for now — stub for Phase 1.2.1)
    # TODO: replace pixel-approximated T/X positions with exact
    # intersection of adjacent edge centerlines.

    # 5. Manhattan snap
    if manhattan_snap and graph.edges:
        _manhattan_snap_if_dominant(
            graph,
            angle_tol_deg=manhattan_angle_tol_deg,
            dominance_threshold=manhattan_dominance_threshold,
        )

    return graph


def _manhattan_snap_if_dominant(
    graph: WallGraph,
    *,
    angle_tol_deg: float,
    dominance_threshold: float,
) -> int:
    """If ≥ threshold of edges are near-axis, snap them exactly to axis.

    Rotates the edge's endpoints in place (via node.xy mutation). Uses
    the edge midpoint as the rotation center so the wall's centre stays
    put. Returns the count of edges snapped.

    Edges that don't land within angle_tol_deg of 0° or 90° are left
    alone — real diagonal walls are preserved.
    """
    angles = []
    for e in graph.edges.values():
        a = graph.nodes[e.node_a].xy
        b = graph.nodes[e.node_b].xy
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
        angles.append((e, ang))

    if not angles:
        return 0

    near_axis = [
        (e, ang) for e, ang in angles
        if (ang <= angle_tol_deg or ang >= 180.0 - angle_tol_deg
            or abs(ang - 90.0) <= angle_tol_deg)
    ]
    dominance = len(near_axis) / len(angles)
    if dominance < dominance_threshold:
        logger.info(
            "manhattan_snap: dominance %.2f < %.2f threshold, skipping",
            dominance, dominance_threshold,
        )
        return 0

    snapped = 0
    for e, ang in near_axis:
        a_node = graph.nodes[e.node_a]
        b_node = graph.nodes[e.node_b]
        ax, ay = a_node.xy
        bx, by = b_node.xy
        mx = (ax + bx) * 0.5
        my = (ay + by) * 0.5
        L = math.hypot(bx - ax, by - ay)
        if L < 1e-6:
            continue

        # Target angle: 0° (horizontal) or 90° (vertical)
        if ang <= angle_tol_deg or ang >= 180.0 - angle_tol_deg:
            target_rad = 0.0 if bx >= ax else math.pi
        else:
            target_rad = math.pi * 0.5 if by >= ay else -math.pi * 0.5

        half = L * 0.5
        new_a = (mx - math.cos(target_rad) * half,
                 my - math.sin(target_rad) * half)
        new_b = (mx + math.cos(target_rad) * half,
                 my + math.sin(target_rad) * half)
        a_node.xy = new_a
        b_node.xy = new_b
        snapped += 1

    logger.info(
        "manhattan_snap: dominance %.2f, snapped %d/%d edges",
        dominance, snapped, len(angles),
    )
    return snapped
