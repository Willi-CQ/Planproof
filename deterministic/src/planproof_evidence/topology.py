"""Topological analysis of wall networks.

Builds junction graphs, polygonizes enclosed faces with holes,
and attaches openings to walls.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon as ShapelyPolygon, Point as ShapelyPoint
from shapely.ops import polygonize, unary_union
from scipy.spatial import cKDTree

from .contracts import (
    BBox,
    JunctionNode,
    Observation,
    ObservationKind,
    OpeningAssociation,
    PageFrame,
    Point,
    Polygon,
    RoomFace,
    SourceFamily,
    Mode,
    VectorEvidence,
    WallEdge,
    bbox_to_polygon,
    polygon_area,
    polygon_centroid,
)


def build_wall_network(
    wall_edges: List[Dict],
    junction_tolerance: float = 10.0,
) -> Tuple[List[JunctionNode], List[WallEdge]]:
    """Build junction graph from wall edges.
    
    Args:
        wall_edges: List of wall edge dicts from vectors.py
        junction_tolerance: Distance tolerance for junction detection
    
    Returns:
        (junction_nodes, wall_edges_with_ids)
    """
    # Collect all endpoints
    endpoints = []
    for edge in wall_edges:
        endpoints.append((edge["a"], edge["id"], "start"))
        endpoints.append((edge["b"], edge["id"], "end"))
    
    # Cluster endpoints that are close together
    junctions = []
    used = set()
    
    for i, (pt1, edge_id1, end1) in enumerate(endpoints):
        if i in used:
            continue
        
        # Find all endpoints within tolerance
        cluster = [(pt1, edge_id1, end1)]
        used.add(i)
        
        for j, (pt2, edge_id2, end2) in enumerate(endpoints):
            if j in used or i == j:
                continue
            
            dist = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
            if dist < junction_tolerance:
                cluster.append((pt2, edge_id2, end2))
                used.add(j)
        
        # Compute junction position (centroid)
        positions = [c[0] for c in cluster]
        junction_pos = (
            sum(p[0] for p in positions) / len(positions),
            sum(p[1] for p in positions) / len(positions),
        )
        
        junctions.append({
            "id": f"junction-{len(junctions)}",
            "position": junction_pos,
            "degree": len(cluster),
            "edges": [(c[1], c[2]) for c in cluster],
        })
    
    # Create JunctionNode objects
    nodes = []
    for junc in junctions:
        nodes.append(JunctionNode(
            id=junc["id"],
            position=junc["position"],
            degree=junc["degree"],
            kind="junction" if junc["degree"] > 2 else "corner",
            source_ids=[e[0] for e in junc["edges"]],
        ))
    
    # Update wall edges with node IDs
    updated_edges = []
    for edge in wall_edges:
        # Find which junctions this edge connects
        start_junction = None
        end_junction = None
        
        for junc in junctions:
            for edge_id, end_type in junc["edges"]:
                if edge_id == edge["id"]:
                    if end_type == "start":
                        start_junction = junc["id"]
                    else:
                        end_junction = junc["id"]
        
        updated_edges.append(WallEdge(
            id=edge["id"],
            start_node=start_junction or "unknown",
            end_node=end_junction or "unknown",
            geometry={"a": edge["a"], "b": edge["b"]},
            thickness_px=edge.get("thickness"),
            length_px=edge.get("length"),
            source_ids=[edge["id"]],
        ))
    
    return nodes, updated_edges


def polygonize_faces(
    wall_edges: List[WallEdge],
    image_size: Tuple[int, int],
) -> Tuple[List[RoomFace], List[str]]:
    """Polygonize enclosed faces from wall network.
    
    Uses Shapely polygonize to find enclosed regions.
    
    Args:
        wall_edges: List of wall edges
        image_size: (width, height) of image
    
    Returns:
        (room_faces, diagnostics)
    """
    diagnostics = []
    room_faces = []
    
    # Convert wall edges to Shapely LineStrings
    lines = []
    for edge in wall_edges:
        a = edge.geometry["a"]
        b = edge.geometry["b"]
        line = LineString([a, b])
        lines.append(line)
    
    if not lines:
        diagnostics.append("No wall edges to polygonize")
        return room_faces, diagnostics
    
    # Union all lines to create a planar graph
    try:
        union = unary_union(lines)
        
        # Polygonize to find enclosed faces
        polygons = list(polygonize(union))
        
        for i, poly in enumerate(polygons):
            # Check if polygon is valid
            if not poly.is_valid:
                diagnostics.append(f"Invalid polygon {i}: {poly.is_valid}")
                continue
            
            # Get exterior and interior rings
            exterior = list(poly.exterior.coords)
            interiors = [list(ring.coords) for ring in poly.interiors]
            
            # Convert to our format
            exterior_polygon = [(x, y) for x, y in exterior[:-1]]  # Remove closing point
            hole_polygons = [[(x, y) for x, y in ring[:-1]] for ring in interiors]
            
            # Compute area
            area = poly.area
            
            # Skip very small polygons (likely noise)
            if area < 100:  # 100 px² minimum
                continue
            
            # Compute centroid
            centroid = poly.centroid
            
            room_faces.append(RoomFace(
                id=f"room-{i}",
                polygon=exterior_polygon,
                holes=hole_polygons,
                area_px=area,
                centroid=(centroid.x, centroid.y),
                source_ids=[edge.id for edge in wall_edges],
                is_valid=True,
                invalid_reason=None,
                depends_on_inferred=False,
            ))
        
        diagnostics.append(f"Polygonized {len(room_faces)} faces from {len(lines)} wall edges")
        
    except Exception as e:
        diagnostics.append(f"Polygonization failed: {e}")
    
    return room_faces, diagnostics


def attach_openings_to_walls(
    openings: List[Observation],
    wall_edges: List[WallEdge],
    distance_threshold: float = 20.0,
) -> List[OpeningAssociation]:
    """Attach opening candidates to nearby walls.
    
    Args:
        openings: List of opening observations
        wall_edges: List of wall edges
        distance_threshold: Maximum distance to consider attachment
    
    Returns:
        List of OpeningAssociation objects
    """
    associations = []
    
    # Build KD-tree of wall edge midpoints
    wall_points = []
    wall_indices = []
    for i, edge in enumerate(wall_edges):
        a = edge.geometry["a"]
        b = edge.geometry["b"]
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        wall_points.append(mid)
        wall_indices.append(i)
    
    if not wall_points:
        return associations
    
    wall_tree = cKDTree(wall_points)
    
    for opening in openings:
        if opening.kind not in [ObservationKind.OPENING_CANDIDATE, ObservationKind.DOOR, ObservationKind.WINDOW]:
            continue
        
        # Get opening center
        if opening.bbox:
            x0, y0, x1, y1 = opening.bbox
            center = ((x0 + x1) / 2, (y0 + y1) / 2)
        else:
            continue
        
        # Find nearest wall
        dist, idx = wall_tree.query(center)
        
        if dist < distance_threshold:
            wall_idx = wall_indices[idx]
            wall_edge = wall_edges[wall_idx]
            
            # Determine orientation
            a = wall_edge.geometry["a"]
            b = wall_edge.geometry["b"]
            dx = abs(b[0] - a[0])
            dy = abs(b[1] - a[1])
            
            if dx > dy:
                orientation = "horizontal"
            elif dy > dx:
                orientation = "vertical"
            else:
                orientation = "diagonal"
            
            associations.append(OpeningAssociation(
                opening_id=opening.id,
                wall_edge_id=wall_edge.id,
                position=center,
                overlap_px=dist,
                orientation=orientation,
                is_attached=True,
                source_ids=[opening.id, wall_edge.id],
            ))
        else:
            # Unattached opening
            associations.append(OpeningAssociation(
                opening_id=opening.id,
                wall_edge_id=None,
                position=center,
                overlap_px=dist,
                orientation="unknown",
                is_attached=False,
                source_ids=[opening.id],
            ))
    
    return associations


def estimate_plan_region(
    mask: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[BBox, float]:
    """Estimate the main floor-plan region of the page (T2-03).

    Architectural sheets contain the plan plus legends, title blocks and
    marginalia. The plan's walls typically form the largest connected ink
    component that does not span essentially the whole sheet (a sheet
    border/frame). We therefore take the bounding box of the largest
    connected component whose bbox covers at most ~90% of the image area;
    if every component is sheet-spanning we fall back to the largest
    component overall, then to the full image.

    Args:
        mask: Binary ink mask
        image_size: (width, height) of canonical frame

    Returns:
        (plan_bbox, plan_area_px2)
    """
    width, height = image_size
    image_area = float(width * height)

    binary = (mask > 0).astype(np.uint8)
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    best_idx = -1
    best_area = -1
    fallback_idx = -1
    fallback_area = -1

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area <= 0:
            continue
        bbox_area = float(w * h)
        if area > fallback_area:
            fallback_area = area
            fallback_idx = i
        # Reject sheet-spanning frames/borders: bbox covering ~whole page
        if w >= 0.9 * width and h >= 0.9 * height:
            continue
        if bbox_area <= 0.9 * image_area and area > best_area:
            best_area = area
            best_idx = i

    idx = best_idx if best_idx >= 0 else fallback_idx
    if idx < 0:
        return (0.0, 0.0, float(width), float(height)), image_area

    x, y, w, h, _area = stats[idx]
    bbox = (float(x), float(y), float(x + w), float(y + h))
    return bbox, float(w * h)


def classify_room_faces(
    room_faces: List[RoomFace],
    plan_bbox: BBox,
    plan_area: float,
    image_size: Tuple[int, int],
    min_area_abs: float = 1200.0,
    area_fraction: float = 0.005,
    max_aspect: float = 8.0,
    region_margin_frac: float = 0.10,
) -> List[str]:
    """Classify polygonized faces into semantic room classes (T2-15).

    Rules (versioned development parameters, not measured optima):
      - "room": enclosed face inside the plan region with
        area >= max(min_area_abs, area_fraction * plan_area) and
        bounded aspect ratio.
      - "sub_feature": enclosed loop below the area threshold — text
        glyph counters (o, a, 0), fixtures, small legend boxes.
      - "non_room_loop": fails aspect bound or lies outside the plan
        region — legend strips, title-block boxes, wall-cavity slivers.

    Faces are never dropped here; classification only retags them.
    """
    diagnostics = []
    min_area = max(min_area_abs, area_fraction * plan_area)

    # Expand plan bbox by a margin so legitimate edge rooms are not
    # excluded while detached legend bands still fall outside.
    px0, py0, px1, py1 = plan_bbox
    mx = region_margin_frac * (px1 - px0)
    my = region_margin_frac * (py1 - py0)
    ex = (px0 - mx, py0 - my, px1 + mx, py1 + my)

    counts = {"room": 0, "sub_feature": 0, "non_room_loop": 0}
    for face in room_faces:
        if not face.is_valid:
            face.semantic_class = "non_room_loop"
            face.invalid_reason = face.invalid_reason or "invalid_geometry"
            counts["non_room_loop"] += 1
            continue

        # Aspect ratio from polygon bbox
        xs = [p[0] for p in face.polygon]
        ys = [p[1] for p in face.polygon]
        bw = max(xs) - min(xs) if xs else 0.0
        bh = max(ys) - min(ys) if ys else 0.0
        aspect = max(bw, bh) / max(1.0, min(bw, bh))

        # Centroid inside expanded plan region?
        cx, cy = face.centroid
        inside_plan = ex[0] <= cx <= ex[2] and ex[1] <= cy <= ex[3]

        if face.area_px < min_area:
            face.semantic_class = "sub_feature"
            counts["sub_feature"] += 1
        elif aspect > max_aspect or not inside_plan:
            face.semantic_class = "non_room_loop"
            counts["non_room_loop"] += 1
        else:
            face.semantic_class = "room"
            counts["room"] += 1

    diagnostics.append(
        f"Face classification: plan_area={plan_area:.0f}px^2 "
        f"min_area={min_area:.0f}px^2 max_aspect={max_aspect} "
        f"-> room={counts['room']} sub_feature={counts['sub_feature']} "
        f"non_room_loop={counts['non_room_loop']}"
    )
    return diagnostics


def extract_topology(
    mask: np.ndarray,
    wall_edges: List[Dict],
    openings: List[Observation],
    frame: PageFrame,
) -> VectorEvidence:
    """Extract complete topology from mask and wall edges.
    
    Args:
        mask: Binary mask
        wall_edges: List of wall edge dicts from vectors.py
        openings: List of opening observations
        frame: PageFrame
    
    Returns:
        VectorEvidence with all topology information
    """
    diagnostics = []
    
    # Build wall network
    junction_nodes, wall_edge_objs = build_wall_network(wall_edges)
    diagnostics.append(f"Built {len(junction_nodes)} junction nodes, {len(wall_edge_objs)} wall edges")
    
    # Polygonize faces
    room_faces, poly_diagnostics = polygonize_faces(wall_edge_objs, frame.image_size)
    diagnostics.extend(poly_diagnostics)
    
    # T2-03/T2-15: estimate plan region and classify faces semantically
    plan_bbox, plan_area = estimate_plan_region(mask, frame.image_size)
    diagnostics.append(
        f"Plan region bbox={[round(v) for v in plan_bbox]} area={plan_area:.0f}px^2"
    )
    diagnostics.extend(classify_room_faces(room_faces, plan_bbox, plan_area, frame.image_size))
    
    # Attach openings
    opening_associations = attach_openings_to_walls(openings, wall_edge_objs)
    diagnostics.append(f"Attached {len([a for a in opening_associations if a.is_attached])} openings to walls")
    
    return VectorEvidence(
        frame_id=f"p{frame.page_index}-canonical",
        primitives=[],  # Will be filled from vectors.py
        wall_edges=wall_edge_objs,
        junction_nodes=junction_nodes,
        room_faces=room_faces,
        opening_associations=opening_associations,
        dimensions=[],  # Will be filled by measurements.py
        calibrations=[],  # Will be filled by measurements.py
        topology_diagnostics=diagnostics,
        coverage={"pages_processed": 1},
        plan_region_bbox=plan_bbox,
        plan_region_area_px=plan_area,
    )
