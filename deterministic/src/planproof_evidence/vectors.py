"""Vector evidence extraction and fitting.

Vectorizes wall masks using skeletonization, distance transform, and
collinear segment merging. Preserves slanted walls and distinguishes
filled wall bodies from thin paired outlines.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy import ndimage

from .contracts import (
    BBox,
    Observation,
    ObservationKind,
    PageFrame,
    Point,
    Polygon,
    SourceFamily,
    Mode,
    VectorPrimitive,
    WallEdge,
    JunctionNode,
    bbox_to_polygon,
    polygon_area,
)
from .pixels import extract_line_segments


def compute_distance_transform(mask: np.ndarray) -> np.ndarray:
    """Compute distance transform of binary mask.
    
    Args:
        mask: Binary mask (255 = ink, 0 = background)
    
    Returns:
        Distance transform array (float64)
    """
    # Invert mask for distance transform (distance from background)
    inv_mask = cv2.bitwise_not(mask)
    dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
    return dist.astype(np.float64)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Skeletonize binary mask to get centerlines.
    
    Args:
        mask: Binary mask
    
    Returns:
        Skeletonized mask
    """
    # Ensure binary
    binary = (mask > 0).astype(np.uint8) * 255
    skeleton = skeletonize(binary > 0)
    return (skeleton * 255).astype(np.uint8)


def extract_wall_thickness(
    mask: np.ndarray,
    skeleton: np.ndarray,
) -> Dict[str, float]:
    """Extract wall thickness statistics from distance transform.
    
    Args:
        mask: Binary mask
        skeleton: Skeletonized mask
    
    Returns:
        Dict with thickness statistics
    """
    dist = compute_distance_transform(mask)
    
    # Get distances at skeleton points
    skeleton_coords = np.where(skeleton > 0)
    if len(skeleton_coords[0]) == 0:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}
    
    distances = dist[skeleton_coords]
    
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "std": float(np.std(distances)),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
    }


def merge_collinear_segments(
    segments: List[Dict],
    angle_tolerance: float = 2.0,
    distance_tolerance: float = 5.0,
) -> List[Dict]:
    """Merge collinear segments that are close together.
    
    Args:
        segments: List of line segments
        angle_tolerance: Angle tolerance in degrees
        distance_tolerance: Distance tolerance in pixels
    
    Returns:
        List of merged segments
    """
    if len(segments) <= 1:
        return segments
    
    # Ensure all segments have angle computed
    for seg in segments:
        if "angle" not in seg:
            a = seg["a"]
            b = seg["b"]
            seg["angle"] = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    
    merged = []
    used = set()
    
    for i, seg1 in enumerate(segments):
        if i in used:
            continue
        
        # Start with this segment
        current = seg1.copy()
        used.add(i)
        
        # Try to merge with other segments
        for j, seg2 in enumerate(segments):
            if j in used:
                continue
            
            # Check angle similarity
            angle_diff = abs(seg1["angle"] - seg2["angle"])
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            if angle_diff > angle_tolerance:
                continue
            
            # Check if endpoints are close
            a1 = np.array(seg1["a"])
            b1 = np.array(seg1["b"])
            a2 = np.array(seg2["a"])
            b2 = np.array(seg2["b"])
            
            # Check if segments are collinear (distance from endpoints to line)
            line_dir = b1 - a1
            line_len = np.linalg.norm(line_dir)
            if line_len == 0:
                continue
            line_dir = line_dir / line_len
            
            # Distance from a2 to line through a1,b1
            # 2D cross product written out: np.cross rejects 2D vectors in NumPy >= 2.5.
            offset_a2, offset_b2 = a2 - a1, b2 - a1
            dist_a2 = abs(line_dir[0] * offset_a2[1] - line_dir[1] * offset_a2[0])
            dist_b2 = abs(line_dir[0] * offset_b2[1] - line_dir[1] * offset_b2[0])
            
            if dist_a2 < distance_tolerance and dist_b2 < distance_tolerance:
                # Merge segments
                all_points = [a1, b1, a2, b2]
                projections = [np.dot(p - a1, line_dir) for p in all_points]
                min_idx = np.argmin(projections)
                max_idx = np.argmax(projections)
                
                new_a = all_points[min_idx]
                new_b = all_points[max_idx]
                
                current["a"] = tuple(new_a)
                current["b"] = tuple(new_b)
                current["length"] = np.linalg.norm(new_b - new_a)
                current["angle"] = math.degrees(math.atan2(
                    new_b[1] - new_a[1], new_b[0] - new_a[0]
                ))
                current["n_points"] = current.get("n_points", 0) + seg2.get("n_points", 0)
                current["support"] = max(current.get("support", 0), seg2.get("support", 0))
                
                used.add(j)
        
        merged.append(current)
    
    return merged


def extract_wall_edges_from_segments(
    segments: List[Dict],
    mask: np.ndarray,
    min_wall_length: float = 30.0,
) -> List[Dict]:
    """Extract wall edges from line segments.
    
    Args:
        segments: List of line segments from pixels.py
        mask: Binary mask for thickness computation
        min_wall_length: Minimum wall length in pixels
    
    Returns:
        List of wall edge dicts
    """
    # Filter segments by length
    valid_segments = [s for s in segments if s["length"] >= min_wall_length]
    
    if not valid_segments:
        return []
    
    # Convert segments to wall edges
    wall_edges = []
    dist = compute_distance_transform(mask)
    
    for i, seg in enumerate(valid_segments):
        a = seg["a"]
        b = seg["b"]
        
        # Sample points along segment to get thickness
        length = seg["length"]
        n_samples = max(5, int(length / 10))
        thicknesses = []
        
        for t in np.linspace(0, 1, n_samples):
            p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            x, y = int(p[0]), int(p[1])
            if 0 <= x < dist.shape[1] and 0 <= y < dist.shape[0]:
                thicknesses.append(dist[y, x] * 2)  # Diameter
        
        avg_thickness = np.mean(thicknesses) if thicknesses else 0
        
        wall_edges.append({
            "id": f"wall-{i}",
            "a": a,
            "b": b,
            "length": length,
            "angle": math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])),
            "thickness": avg_thickness,
            "n_points": seg.get("n_points", 1),
            "support": seg.get("confidence", 0.5),
        })
    
    return wall_edges


def extract_wall_edges(
    mask: np.ndarray,
    skeleton: np.ndarray,
    min_wall_length: float = 30.0,
) -> List[Dict]:
    """Extract wall edges from mask using line segments.
    
    Args:
        mask: Binary mask
        skeleton: Skeletonized mask (for compatibility)
        min_wall_length: Minimum wall length in pixels
    
    Returns:
        List of wall edge dicts
    """
    # Extract line segments using LSD/Hough
    segments = extract_line_segments(mask)
    
    # Merge collinear segments
    merged = merge_collinear_segments(segments)
    
    # Extract wall edges from merged segments
    wall_edges = extract_wall_edges_from_segments(merged, mask, min_wall_length)
    
    return wall_edges


def build_junction_graph(
    wall_edges: List[Dict],
    junction_tolerance: float = 10.0,
) -> Tuple[List[JunctionNode], List[WallEdge]]:
    """Build junction graph from wall edges.
    
    Args:
        wall_edges: List of wall edge dicts
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


def extract_vector_evidence(
    mask: np.ndarray,
    frame: PageFrame,
    min_wall_length: float = 30.0,
) -> List[VectorPrimitive]:
    """Extract vector primitives from mask.
    
    Args:
        mask: Binary mask
        frame: PageFrame
        min_wall_length: Minimum wall length
    
    Returns:
        List of VectorPrimitive objects
    """
    primitives = []
    
    # Skeletonize
    skeleton = skeletonize_mask(mask)
    
    # Extract wall edges
    wall_edges = extract_wall_edges(mask, skeleton, min_wall_length)
    
    for edge in wall_edges:
        primitives.append(VectorPrimitive(
            id=edge["id"],
            kind="line",
            geometry={
                "a": edge["a"],
                "b": edge["b"],
                "length": edge["length"],
                "angle": edge["angle"],
                "thickness": edge["thickness"],
            },
            source_ids=[],  # Will be filled later
            confidence=edge["support"],
            support=edge["support"],
        ))
    
    return primitives
