"""Moebel-Erkennung + -Platzierung fuer die Plan-to-BIM-Pipeline.

Architektur (research-belegt, siehe .brain):
  LOKALISIEREN  = User/YOLO-Label (Bbox)        -> kein LLM
  ERKENNEN      = EU-Vision-LLM (benennt, geschlossener Katalog) -> classify
  PLATZIEREN    = deterministisch (Geometrie, Run-Layout, Orientierung) -> placement

Einstieg: :func:`run_furniture_step` (Klassifikation + Platzierung) und
:func:`append_furniture_to_glb` (Integration: Knoten an die fertige BIM-GLB).
"""
from __future__ import annotations

from .catalog import (
    FURNITURE_CATALOG,
    GENERIC_ID,
    UNKNOWN_ID,
    CatalogItem,
    catalog_enum,
    catalog_prompt_table,
    get_item,
    resolve_item,
)
from .classify import (
    ClassifyInput,
    FurnitureClassification,
    FurnitureClassifier,
    FurnitureContent,
    build_classification_messages,
    classify_furniture,
    fallback_classification,
    parse_classification_response,
)
from .crop import Crop, extract_crop
from .mesh import append_furniture_to_scene, build_item_mesh, furniture_nodes
from .models import build_standard_model, has_standard_model
from .pipeline import (
    FurnitureResult,
    append_furniture_to_glb,
    run_furniture_step,
)
from .placement import PlacedFurniture, PlacedItem, place_furniture
from .rooms import RoomRegion, assign_room, classify_room_type, extract_rooms
from .transport import InternalApiFurnitureClassifier, resolve_default_classifier

__all__ = [
    # catalog
    "FURNITURE_CATALOG", "CatalogItem", "catalog_enum", "catalog_prompt_table",
    "get_item", "resolve_item", "GENERIC_ID", "UNKNOWN_ID",
    # classify
    "FurnitureClassification", "FurnitureContent", "FurnitureClassifier",
    "ClassifyInput", "classify_furniture", "parse_classification_response",
    "build_classification_messages", "fallback_classification",
    # crop
    "Crop", "extract_crop",
    # rooms
    "RoomRegion", "extract_rooms", "assign_room", "classify_room_type",
    # placement
    "PlacedFurniture", "PlacedItem", "place_furniture",
    # mesh + models
    "build_item_mesh", "furniture_nodes", "append_furniture_to_scene",
    "build_standard_model", "has_standard_model",
    # pipeline
    "FurnitureResult", "run_furniture_step", "append_furniture_to_glb",
    # transport (EU egress)
    "InternalApiFurnitureClassifier", "resolve_default_classifier",
]
