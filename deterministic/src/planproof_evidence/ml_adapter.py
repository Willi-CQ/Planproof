"""ML adapter for planproof-detection-v1 schema.

Consumes frozen ML predictions from Willi's baseline models
(corrected-3500-global, cad-objects-3000-global) with strict
validation and provenance tracking.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from .contracts import (
    BBox,
    Observation,
    ObservationKind,
    PageFrame,
    SourceFamily,
    Mode,
)


# ML schema version
ML_SCHEMA_VERSION = "planproof-detection-v1"


class MLInstance(BaseModel):
    """A single ML detection instance."""
    id: str
    class_name: str = Field(..., alias="class")
    bbox: BBox
    confidence: float
    polygons: Optional[List[List[Tuple[float, float]]]] = None
    score_semantics: str = "model_probability"
    metadata: Dict = Field(default_factory=dict)
    
    model_config = {"populate_by_name": True}
    
    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v):
        if len(v) != 4:
            raise ValueError("bbox must have 4 elements")
        x0, y0, x1, y1 = v
        if not all(math.isfinite(c) for c in v):
            raise ValueError("bbox coordinates must be finite")
        if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
            raise ValueError("bbox coordinates must be non-negative")
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bbox must have positive width and height")
        return v
    
    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("confidence must be in [0, 1]")
        return v


class MLPrediction(BaseModel):
    """Complete ML prediction for a document."""
    schema_version: str
    document_id: str
    image_size: Tuple[int, int]
    model_version: str
    coordinate_system: str = "top_left_pixel"
    inference_mode: str = "global1024"
    instances: List[MLInstance] = Field(default_factory=list)
    metadata: Dict = Field(default_factory=dict)
    
    @field_validator("schema_version")
    @classmethod
    def validate_schema(cls, v):
        if v != ML_SCHEMA_VERSION:
            raise ValueError(f"Expected schema_version '{ML_SCHEMA_VERSION}', got '{v}'")
        return v
    
    @field_validator("image_size")
    @classmethod
    def validate_size(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError("image_size must be (width, height) with positive values")
        return v


def load_ml_predictions(path: Path) -> MLPrediction:
    """Load ML predictions from JSON file.
    
    Args:
        path: Path to prediction JSON file
    
    Returns:
        MLPrediction object
    
    Raises:
        ValueError: If schema is invalid or file is malformed
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return MLPrediction.model_validate(data)


def validate_ml_binding(
    prediction: MLPrediction,
    frame: PageFrame,
) -> Tuple[bool, List[str]]:
    """Validate that ML prediction is bound to the correct document/page.
    
    Args:
        prediction: ML prediction
        frame: PageFrame for the document
    
    Returns:
        (is_valid, warnings)
    """
    warnings = []
    
    # Check document binding
    if prediction.document_id != frame.document_id:
        warnings.append(f"Document ID mismatch: ML={prediction.document_id}, frame={frame.document_id}")
        return False, warnings
    
    # Check image size
    if prediction.image_size != frame.image_size:
        warnings.append(f"Image size mismatch: ML={prediction.image_size}, frame={frame.image_size}")
        return False, warnings
    
    # Check coordinate system
    if prediction.coordinate_system != "top_left_pixel":
        warnings.append(f"Unexpected coordinate system: {prediction.coordinate_system}")
    
    return True, warnings


def ml_instance_to_observation(
    instance: MLInstance,
    page_index: int,
    frame_id: str,
) -> Observation:
    """Convert ML instance to Observation.
    
    Args:
        instance: ML instance
        page_index: Page index
        frame_id: Frame ID
    
    Returns:
        Observation object
    """
    # Map ML class to ObservationKind
    class_map = {
        "wall": ObservationKind.WALL,
        "room": ObservationKind.ROOM,
        "door": ObservationKind.DOOR,
        "window": ObservationKind.WINDOW,
        "stairs": ObservationKind.STAIRS,
        "text": ObservationKind.TEXT,
    }
    
    kind = class_map.get(instance.class_name, ObservationKind.WALL)
    
    return Observation(
        id=f"ml-{instance.id}",
        kind=kind,
        source_family=SourceFamily.ML,
        mode=Mode.INDEPENDENT,  # ML is independent unless refined
        page_index=page_index,
        frame_id=frame_id,
        bbox=instance.bbox,
        geometry={
            "class": instance.class_name,
            "polygons": instance.polygons,
            "confidence": instance.confidence,
        },
        method="ml_inference",
        method_version=instance.metadata.get("model_version", "unknown"),
        score=instance.confidence,
        score_semantics=instance.score_semantics,
        metadata=instance.metadata,
    )


def extract_ml_observations(
    prediction: MLPrediction,
    frame: PageFrame,
) -> Tuple[List[Observation], List[str]]:
    """Extract observations from ML prediction.
    
    Args:
        prediction: ML prediction
        frame: PageFrame
    
    Returns:
        (observations, warnings)
    """
    # Validate binding
    is_valid, warnings = validate_ml_binding(prediction, frame)
    if not is_valid:
        return [], warnings
    
    observations = []
    for instance in prediction.instances:
        obs = ml_instance_to_observation(
            instance,
            page_index=frame.page_index,
            frame_id=f"p{frame.page_index}-canonical",
        )
        observations.append(obs)
    
    return observations, warnings


def load_ml_maps(path: Path) -> Dict:
    """Load ML auxiliary maps (e.g., segmentation masks).
    
    Args:
        path: Path to maps JSON file
    
    Returns:
        Dict with map data
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_ml_maps(
    maps: Dict,
    frame: PageFrame,
) -> Tuple[bool, List[str]]:
    """Validate ML maps binding.
    
    Args:
        maps: ML maps dict
        frame: PageFrame
    
    Returns:
        (is_valid, warnings)
    """
    warnings = []
    
    # Check if maps have document binding
    if "document_id" in maps:
        if maps["document_id"] != frame.document_id:
            warnings.append(f"Maps document ID mismatch")
            return False, warnings
    
    # Check image size if present
    if "image_size" in maps:
        if tuple(maps["image_size"]) != frame.image_size:
            warnings.append(f"Maps image size mismatch")
            return False, warnings
    
    return True, warnings
