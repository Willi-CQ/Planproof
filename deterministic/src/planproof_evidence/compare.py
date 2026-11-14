"""ML vs geometry comparison and aggregation.

Compares ML predictions with classical geometry evidence,
identifies agreements/conflicts, and creates explicit aggregates.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from scipy.spatial import cKDTree

from .contracts import (
    Aggregate,
    BBox,
    Comparison,
    ComparisonRelation,
    Observation,
    ObservationKind,
    PageFrame,
    Point,
    SourceFamily,
    Mode,
    bbox_iou,
    bbox_intersection,
)


# Matching thresholds
IOU_THRESHOLD = 0.5  # Minimum IoU for agreement
DISTANCE_THRESHOLD = 20.0  # Maximum distance for matching


def compute_bbox_distance(bbox1: BBox, bbox2: BBox) -> float:
    """Compute minimum distance between two bboxes.
    
    Args:
        bbox1: First bbox
        bbox2: Second bbox
    
    Returns:
        Minimum distance between bboxes
    """
    # Check if bboxes intersect
    inter = bbox_intersection(bbox1, bbox2)
    if inter is not None:
        return 0.0
    
    # Compute distance between closest points
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2
    
    # Find closest points
    dx = max(0, max(x0_2 - x1_1, x0_1 - x1_2))
    dy = max(0, max(y0_2 - y1_1, y0_1 - y1_2))
    
    return math.sqrt(dx * dx + dy * dy)


def match_observations(
    ml_obs: List[Observation],
    geom_obs: List[Observation],
    iou_threshold: float = IOU_THRESHOLD,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> List[Tuple[Observation, Observation, float]]:
    """Match ML and geometry observations by spatial proximity.
    
    Args:
        ml_obs: List of ML observations
        geom_obs: List of geometry observations
        iou_threshold: IoU threshold for matching
        distance_threshold: Distance threshold for matching
    
    Returns:
        List of (ml_obs, geom_obs, score) tuples
    """
    matches = []
    used_geom = set()
    
    for ml in ml_obs:
        if ml.bbox is None:
            continue
        
        best_match = None
        best_score = 0.0
        
        for i, geom in enumerate(geom_obs):
            if i in used_geom or geom.bbox is None:
                continue
            
            # Check class compatibility
            if ml.kind != geom.kind:
                continue
            
            # Compute IoU
            iou = bbox_iou(ml.bbox, geom.bbox)
            
            # Compute distance
            dist = compute_bbox_distance(ml.bbox, geom.bbox)
            
            # Score: combination of IoU and distance
            score = iou - (dist / distance_threshold) * 0.5
            
            if score > best_score and (iou > iou_threshold or dist < distance_threshold):
                best_score = score
                best_match = (i, geom)
        
        if best_match is not None:
            geom_idx, geom = best_match
            matches.append((ml, geom, best_score))
            used_geom.add(geom_idx)
    
    return matches


def create_comparison(
    ml_obs: Observation,
    geom_obs: Optional[Observation],
    relation: ComparisonRelation,
    metrics: Dict[str, float],
) -> Comparison:
    """Create a comparison object.
    
    Args:
        ml_obs: ML observation
        geom_obs: Geometry observation (or None)
        relation: Comparison relation
        metrics: Comparison metrics
    
    Returns:
        Comparison object
    """
    # Compute disagreement region
    disagreement_region = None
    if ml_obs.bbox and geom_obs and geom_obs.bbox:
        # Union of bboxes
        x0 = min(ml_obs.bbox[0], geom_obs.bbox[0])
        y0 = min(ml_obs.bbox[1], geom_obs.bbox[1])
        x1 = max(ml_obs.bbox[2], geom_obs.bbox[2])
        y1 = max(ml_obs.bbox[3], geom_obs.bbox[3])
        disagreement_region = (x0, y0, x1, y1)
    elif ml_obs.bbox:
        disagreement_region = ml_obs.bbox
    
    return Comparison(
        id=f"cmp-{ml_obs.id}",
        relation=relation,
        ml_observation_id=ml_obs.id,
        geometry_observation_id=geom_obs.id if geom_obs else None,
        supporting_geometry=geom_obs.geometry if geom_obs else None,
        disagreement_region=disagreement_region,
        crop_id=None,  # Will be set when crop is generated
        warnings=[],
        metrics=metrics,
    )


def compare_ml_and_geometry(
    ml_obs: List[Observation],
    geom_obs: List[Observation],
    frame: PageFrame,
) -> List[Comparison]:
    """Compare ML and geometry observations.
    
    Args:
        ml_obs: List of ML observations
        geom_obs: List of geometry observations
        frame: PageFrame
    
    Returns:
        List of Comparison objects
    """
    comparisons = []
    
    # Match observations
    matches = match_observations(ml_obs, geom_obs)
    matched_ml_ids = set()
    matched_geom_ids = set()
    
    # Process matches
    for ml, geom, score in matches:
        matched_ml_ids.add(ml.id)
        matched_geom_ids.add(geom.id)
        
        # Compute metrics
        iou = bbox_iou(ml.bbox, geom.bbox) if ml.bbox and geom.bbox else 0.0
        dist = compute_bbox_distance(ml.bbox, geom.bbox) if ml.bbox and geom.bbox else float('inf')
        
        # Determine relation
        if ml.kind != geom.kind:
            relation = ComparisonRelation.CLASS_CONFLICT
        elif iou > 0.7 and dist < 10:
            relation = ComparisonRelation.AGREEMENT
        elif iou > 0.3 or dist < 30:
            relation = ComparisonRelation.GEOMETRY_CONFLICT
        else:
            relation = ComparisonRelation.GEOMETRY_CONFLICT
        
        comparisons.append(create_comparison(
            ml, geom, relation,
            {"iou": iou, "distance": dist, "match_score": score}
        ))
    
    # Process unmatched ML observations
    for ml in ml_obs:
        if ml.id not in matched_ml_ids:
            comparisons.append(create_comparison(
                ml, None, ComparisonRelation.UNMATCHED_ML,
                {"iou": 0.0, "distance": float('inf'), "match_score": 0.0}
            ))
    
    # Process unmatched geometry observations
    for geom in geom_obs:
        if geom.id not in matched_geom_ids:
            # Create a dummy ML observation for comparison
            dummy_ml = Observation(
                id=f"dummy-{geom.id}",
                kind=geom.kind,
                source_family=SourceFamily.ML,
                mode=Mode.INDEPENDENT,
                page_index=frame.page_index,
                frame_id=f"p{frame.page_index}-canonical",
                bbox=geom.bbox,
                geometry={},
                method="dummy",
                method_version="v1",
            )
            comparisons.append(create_comparison(
                dummy_ml, geom, ComparisonRelation.UNMATCHED_GEOMETRY,
                {"iou": 0.0, "distance": float('inf'), "match_score": 0.0}
            ))
    
    return comparisons


def create_aggregates(
    observations: List[Observation],
    comparisons: List[Comparison],
    frame: PageFrame,
) -> List[Aggregate]:
    """Create aggregates from observations and comparisons.
    
    Args:
        observations: All observations
        comparisons: Comparison results
        frame: PageFrame
    
    Returns:
        List of Aggregate objects
    """
    aggregates = []
    
    # Group by class and source
    class_groups = {}
    for obs in observations:
        if obs.kind not in [ObservationKind.WALL, ObservationKind.ROOM, 
                           ObservationKind.DOOR, ObservationKind.WINDOW,
                           ObservationKind.STAIRS]:
            continue
        
        key = (obs.kind.value, obs.source_family.value)
        if key not in class_groups:
            class_groups[key] = []
        class_groups[key].append(obs)
    
    # Create aggregates for each class/source combination
    for (class_name, source_family), obs_list in class_groups.items():
        # Determine counting unit
        counting_unit = {
            "wall": "wall_edge",
            "room": "room_face",
            "door": "door_assembly",
            "window": "window_assembly",
            "stairs": "stair_flight",
        }.get(class_name, "object")
        
        # Check for duplicates (observations with high IoU)
        duplicate_groups = []
        used = set()
        
        for i, obs1 in enumerate(obs_list):
            if i in used:
                continue
            
            group = [obs1.id]
            used.add(i)
            
            for j, obs2 in enumerate(obs_list):
                if j in used or obs1.bbox is None or obs2.bbox is None:
                    continue
                
                iou = bbox_iou(obs1.bbox, obs2.bbox)
                if iou > 0.8:  # High overlap = duplicate
                    group.append(obs2.id)
                    used.add(j)
            
            if len(group) > 1:
                duplicate_groups.append(group)
        
        # Determine status
        if duplicate_groups:
            status = "partial"  # Has duplicates that need resolution
        else:
            status = "resolved"
        
        # Count unique objects (excluding duplicates)
        unique_count = len(obs_list) - sum(len(g) - 1 for g in duplicate_groups)
        
        aggregates.append(Aggregate(
            id=f"agg-{class_name}-{source_family}",
            scope="page",
            page_index=frame.page_index,
            class_name=class_name,
            counting_unit=counting_unit,
            member_ids=[obs.id for obs in obs_list],
            duplicate_groups=duplicate_groups,
            value=float(unique_count),
            unit="count",
            status=status,
            coverage={"total_observations": len(obs_list)},
            source_counts={source_family: len(obs_list)},
            warnings=[],
        ))
    
    return aggregates


def prepare_disagreement_crops(
    comparisons: List[Comparison],
    out_dir,
) -> List[Comparison]:
    """Prepare disagreement regions for crop extraction.
    
    Args:
        comparisons: List of comparisons
        out_dir: Output directory for crops
    
    Returns:
        Updated comparisons with crop_id set
    """
    from .crops import extract_crop
    
    for comp in comparisons:
        if comp.relation in [ComparisonRelation.UNMATCHED_ML, 
                           ComparisonRelation.UNMATCHED_GEOMETRY,
                           ComparisonRelation.CLASS_CONFLICT,
                           ComparisonRelation.GEOMETRY_CONFLICT]:
            if comp.disagreement_region:
                # Extract crop
                crop_result = extract_crop(
                    evidence_path=None,  # Will be set properly
                    page_index=0,
                    bbox=comp.disagreement_region,
                    out_dir=out_dir,
                    padding_px=10,
                )
                comp.crop_id = crop_result.get("crop_id")
    
    return comparisons
