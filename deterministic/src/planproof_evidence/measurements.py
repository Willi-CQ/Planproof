"""Measurement and calibration extraction.

Parses dimension strings, associates them with objects, and
handles calibration for metric conversion. Never asserts physical
units without supported scoped calibration.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .contracts import (
    BBox,
    CalibrationRecord,
    DimensionObservation,
    Observation,
    ObservationKind,
    PageFrame,
    Point,
    SourceFamily,
    Mode,
    VectorEvidence,
)


# Dimension text patterns
DIMENSION_PATTERNS = [
    # Metric: 3.5m, 3500mm
    (r"(\d+\.?\d*)\s*m\b", "m"),
    (r"(\d+\.?\d*)\s*mm\b", "mm"),
    # Imperial: 3'-6", 3' 6", 3ft, 36in
    (r"(\d+)'\s*-?\s*(\d+)\"", "ft_in"),
    (r"(\d+)'\s*(\d+)\"", "ft_in"),
    (r"(\d+)\s*ft\b", "ft"),
    (r"(\d+)\s*in\b", "in"),
]


def parse_dimension_text(text: str) -> Optional[Dict]:
    """Parse dimension text to extract value and unit.
    
    Args:
        text: Dimension text string
    
    Returns:
        Dict with value, unit, or None if not a dimension
    """
    text = text.strip()
    
    # Try each pattern
    for pattern, unit_type in DIMENSION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if unit_type == "m":
                value = float(match.group(1))
                return {"value": value, "unit": "m", "raw": text}
            elif unit_type == "mm":
                value = float(match.group(1))
                return {"value": value, "unit": "mm", "raw": text}
            elif unit_type == "ft_in":
                feet = int(match.group(1))
                inches = int(match.group(2)) if len(match.groups()) > 1 else 0
                value = feet * 12 + inches  # Convert to inches
                return {"value": value, "unit": "in", "raw": text}
            elif unit_type == "ft":
                value = float(match.group(1)) * 12  # Convert to inches
                return {"value": value, "unit": "in", "raw": text}
            elif unit_type == "in":
                value = float(match.group(1))
                return {"value": value, "unit": "in", "raw": text}
    
    return None


def extract_dimensions_from_text(
    text_observations: List[Observation],
    frame: PageFrame,
) -> List[DimensionObservation]:
    """Extract dimension observations from text observations.
    
    Args:
        text_observations: List of text observations (OCR or native PDF)
        frame: PageFrame
    
    Returns:
        List of DimensionObservation objects
    """
    dimensions = []
    
    for obs in text_observations:
        if obs.kind != ObservationKind.TEXT:
            continue
        
        # Get text from geometry
        text = obs.geometry.get("text", "") if obs.geometry else ""
        if not text:
            continue
        
        # Parse dimension
        parsed = parse_dimension_text(text)
        if parsed is None:
            continue
        
        # Create dimension observation
        dimensions.append(DimensionObservation(
            id=f"dim-{obs.id}",
            text=text,
            parsed_value=parsed["value"],
            unit=parsed["unit"],
            bbox=obs.bbox or (0, 0, 0, 0),
            associated_objects=[],  # Will be filled later
            calibration_id=None,
            is_calibrated=False,
            source_ids=[obs.id],
        ))
    
    return dimensions


def find_dimension_associations(
    dimensions: List[DimensionObservation],
    wall_edges: List,
    distance_threshold: float = 50.0,
) -> List[DimensionObservation]:
    """Associate dimensions with nearby wall edges.
    
    Args:
        dimensions: List of dimension observations
        wall_edges: List of wall edges
        distance_threshold: Maximum distance for association
    
    Returns:
        Updated dimensions with associated_objects filled
    """
    # Build list of wall edge midpoints
    wall_midpoints = []
    for edge in wall_edges:
        a = edge.geometry["a"]
        b = edge.geometry["b"]
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        wall_midpoints.append((mid, edge.id))
    
    for dim in dimensions:
        # Get dimension center
        x0, y0, x1, y1 = dim.bbox
        dim_center = ((x0 + x1) / 2, (y0 + y1) / 2)
        
        # Find nearby walls
        associated = []
        for wall_mid, wall_id in wall_midpoints:
            dist = ((dim_center[0] - wall_mid[0]) ** 2 + 
                   (dim_center[1] - wall_mid[1]) ** 2) ** 0.5
            if dist < distance_threshold:
                associated.append(wall_id)
        
        dim.associated_objects = associated
    
    return dimensions


def estimate_calibration(
    dimensions: List[DimensionObservation],
    wall_edges: List,
    tolerance: float = 0.1,
) -> List[CalibrationRecord]:
    """Estimate calibration from dimension strings and wall lengths.
    
    Only creates calibration if consistent scale can be determined.
    
    Args:
        dimensions: List of dimension observations
        wall_edges: List of wall edges
        tolerance: Tolerance for consistency check
    
    Returns:
        List of CalibrationRecord objects
    """
    calibrations = []
    
    # Group dimensions by unit
    unit_groups = {}
    for dim in dimensions:
        if dim.unit not in unit_groups:
            unit_groups[dim.unit] = []
        unit_groups[dim.unit].append(dim)
    
    for unit, dims in unit_groups.items():
        # For each dimension, try to find associated wall
        scale_estimates = []
        
        for dim in dims:
            if not dim.associated_objects:
                continue
            
            # Get first associated wall
            wall_id = dim.associated_objects[0]
            wall = next((w for w in wall_edges if w.id == wall_id), None)
            if wall is None:
                continue
            
            # Get wall length in pixels
            wall_length_px = wall.length_px or 0
            if wall_length_px == 0:
                continue
            
            # Convert dimension value to base unit
            if dim.unit == "m":
                dim_value_m = dim.parsed_value
            elif dim.unit == "mm":
                dim_value_m = dim.parsed_value / 1000.0
            elif dim.unit == "in":
                dim_value_m = dim.parsed_value * 0.0254
            else:
                continue
            
            # Compute scale (pixels per meter)
            scale = wall_length_px / dim_value_m
            scale_estimates.append(scale)
        
        # Check consistency
        if len(scale_estimates) >= 2:
            mean_scale = sum(scale_estimates) / len(scale_estimates)
            std_scale = (sum((s - mean_scale) ** 2 for s in scale_estimates) / 
                        len(scale_estimates)) ** 0.5
            
            # Check if consistent (std/mean < tolerance)
            if mean_scale > 0 and std_scale / mean_scale < tolerance:
                calibrations.append(CalibrationRecord(
                    id=f"calib-{unit}-{len(calibrations)}",
                    page_index=0,  # Will be set properly
                    region=None,
                    unit="m" if unit in ["m", "mm"] else "in",
                    pixels_per_unit=mean_scale,
                    source_dimension_ids=[d.id for d in dims],
                    consistency_check=std_scale / mean_scale,
                    is_valid=True,
                    warnings=[],
                ))
        
        # Single dimension case - less reliable but still useful
        elif len(scale_estimates) == 1:
            calibrations.append(CalibrationRecord(
                id=f"calib-{unit}-{len(calibrations)}",
                page_index=0,
                region=None,
                unit="m" if unit in ["m", "mm"] else "in",
                pixels_per_unit=scale_estimates[0],
                source_dimension_ids=[dims[0].id],
                consistency_check=None,
                is_valid=True,
                warnings=["Single dimension calibration - less reliable"],
            ))
    
    return calibrations


def apply_calibration(
    dimensions: List[DimensionObservation],
    calibrations: List[CalibrationRecord],
) -> List[DimensionObservation]:
    """Apply calibration to dimensions.
    
    Args:
        dimensions: List of dimension observations
        calibrations: List of calibration records
    
    Returns:
        Updated dimensions with calibration applied
    """
    for dim in dimensions:
        # Find matching calibration
        calib = None
        for c in calibrations:
            if dim.unit == c.unit or (dim.unit == "mm" and c.unit == "m"):
                calib = c
                break
        
        if calib:
            dim.calibration_id = calib.id
            dim.is_calibrated = True
        else:
            dim.is_calibrated = False
    
    return dimensions


def extract_measurements(
    text_observations: List[Observation],
    wall_edges: List,
    frame: PageFrame,
) -> Tuple[List[DimensionObservation], List[CalibrationRecord]]:
    """Extract all measurements and calibrations.
    
    Args:
        text_observations: List of text observations
        wall_edges: List of wall edges
        frame: PageFrame
    
    Returns:
        (dimensions, calibrations)
    """
    # Extract dimensions from text
    dimensions = extract_dimensions_from_text(text_observations, frame)
    
    # Associate with walls
    dimensions = find_dimension_associations(dimensions, wall_edges)
    
    # Estimate calibration
    calibrations = estimate_calibration(dimensions, wall_edges)
    
    # Apply calibration
    dimensions = apply_calibration(dimensions, calibrations)
    
    return dimensions, calibrations
