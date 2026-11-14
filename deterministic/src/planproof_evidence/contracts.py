"""Shared evidence contracts for planproof-evidence-v1.

Typed Pydantic models and JSON serializers for the deterministic 2D evidence
extraction pipeline. All coordinate systems use canonical pixel frame:
origin top-left, x right, y down, pixel edges.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "planproof-evidence-v1"


class SourceFamily(str, Enum):
    """Provenance of an observation."""
    NATIVE_PDF = "native_pdf"
    CLASSICAL_PIXELS = "classical_pixels"
    OCR = "ocr"
    ML = "ml"
    HYBRID = "hybrid"


class Mode(str, Enum):
    """Dependency mode: independent or hybrid (ML-assisted)."""
    INDEPENDENT = "independent"
    HYBRID = "hybrid"


class ComparisonRelation(str, Enum):
    """Relationship between ML and geometry evidence."""
    AGREEMENT = "agreement"
    CLASS_CONFLICT = "class_conflict"
    GEOMETRY_CONFLICT = "geometry_conflict"
    MEASUREMENT_CONFLICT = "measurement_conflict"
    UNMATCHED_ML = "unmatched_ml"
    UNMATCHED_GEOMETRY = "unmatched_geometry"


# ---------------------------------------------------------------------------
# Core geometry types
# ---------------------------------------------------------------------------

BBox = Tuple[float, float, float, float]  # [x0, y0, x1, y1]
Point = Tuple[float, float]
Polygon = List[Point]  # Outer ring


class Transform3x3(BaseModel):
    """3x3 homogeneous transformation matrix."""
    matrix: List[List[float]] = Field(..., min_length=3, max_length=3)
    
    @field_validator("matrix")
    @classmethod
    def validate_matrix(cls, v):
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("Matrix must be 3x3")
        return v
    
    def apply(self, point: Point) -> Point:
        """Apply transform to a 2D point."""
        x, y = point
        m = self.matrix
        x_new = m[0][0] * x + m[0][1] * y + m[0][2]
        y_new = m[1][0] * x + m[1][1] * y + m[1][2]
        return (x_new, y_new)
    
    def inverse(self) -> "Transform3x3":
        """Compute inverse transform."""
        # For affine transforms, compute inverse
        m = self.matrix
        # Simplified: assume affine (last row is [0, 0, 1])
        a, b, c = m[0]
        d, e, f = m[1]
        # Inverse of 2x2 part
        det = a * e - b * d
        if abs(det) < 1e-10:
            return Transform3x3(matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        inv_det = 1.0 / det
        inv_a = e * inv_det
        inv_b = -b * inv_det
        inv_d = -d * inv_det
        inv_e = a * inv_det
        
        # Inverse translation
        inv_c = -(inv_a * c + inv_b * f)
        inv_f = -(inv_d * c + inv_e * f)
        
        return Transform3x3(matrix=[
            [inv_a, inv_b, inv_c],
            [inv_d, inv_e, inv_f],
            [0, 0, 1]
        ])


class PageMetadata(BaseModel):
    """PDF page metadata."""
    width: float
    height: float
    rotation: float = 0.0
    crop_box: Optional[BBox] = None
    media_box: Optional[BBox] = None


class PageFrame(BaseModel):
    """Canonical pixel frame for a document page."""
    document_id: str  # SHA256 of source file
    page_index: int
    source_sha256: str  # Hash of original file bytes
    raster_sha256: str  # Hash of decoded RGB pixels
    image_size: Tuple[int, int]  # (width, height)
    color_convention: str = "RGB"  # or "RGBA", "L"
    source_to_canonical: Transform3x3
    canonical_to_source: Transform3x3
    page_metadata: Optional[PageMetadata] = None
    exif_orientation: Optional[int] = None
    
    @field_validator("image_size")
    @classmethod
    def validate_size(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError("image_size must be (width, height) with positive values")
        return v


class ArtifactRef(BaseModel):
    """Reference to an artifact file in the job directory."""
    path: str  # Relative path under job root
    sha256: str
    kind: str  # e.g., "mask", "overlay", "crop", "json"
    dimensions: Optional[Tuple[int, int]] = None
    frame_id: Optional[str] = None
    
    @field_validator("path")
    @classmethod
    def validate_path(cls, v):
        if ".." in v or v.startswith("/"):
            raise ValueError("Artifact path must be relative and not contain '..'")
        return v


# ---------------------------------------------------------------------------
# Observation and evidence types
# ---------------------------------------------------------------------------

class ObservationKind(str, Enum):
    """Types of observations."""
    WALL = "wall"
    ROOM = "room"
    DOOR = "door"
    WINDOW = "window"
    STAIRS = "stairs"
    TEXT = "text"
    DIMENSION = "dimension"
    OPENING_CANDIDATE = "opening_candidate"
    STAIR_CANDIDATE = "stair_candidate"
    RAW_SEGMENT = "raw_segment"
    CONTOUR_RING = "contour_ring"
    COMPONENT = "component"
    MASK_REGION = "mask_region"
    SUB_FEATURE = "sub_feature"
    NON_ROOM_LOOP = "non_room_loop"


class Observation(BaseModel):
    """A single piece of evidence."""
    id: str
    kind: ObservationKind
    source_family: SourceFamily
    mode: Mode
    page_index: int
    frame_id: str
    bbox: Optional[BBox] = None
    geometry: Optional[Dict[str, Any]] = None  # Polygon, segments, etc.
    method: str  # Algorithm/method version
    method_version: str = "v1"
    dependency_ids: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    crop_id: Optional[str] = None
    score: Optional[float] = None  # Named semantics (e.g., "model_probability")
    score_semantics: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PixelEvidence(BaseModel):
    """Pixel-level evidence for a page."""
    frame: PageFrame
    observations: List[Observation] = Field(default_factory=list)
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    availability: Dict[str, Any] = Field(default_factory=dict)  # e.g., {"ocr": "available"}
    coverage: Dict[str, Any] = Field(default_factory=dict)  # e.g., {"pages_processed": 1}
    diagnostics: List[str] = Field(default_factory=list)


class VectorPrimitive(BaseModel):
    """A fitted vector primitive."""
    id: str
    kind: str  # "line", "arc", "polygon", "point"
    geometry: Dict[str, Any]  # Depends on kind
    source_ids: List[str] = Field(default_factory=list)  # Parent observation IDs
    confidence: Optional[float] = None
    support: Optional[float] = None  # Pixel support metric
    

class WallEdge(BaseModel):
    """A wall edge in the junction graph."""
    id: str
    start_node: str
    end_node: str
    geometry: Dict[str, Any]  # Line segment or polyline
    thickness_px: Optional[float] = None
    length_px: Optional[float] = None
    source_ids: List[str] = Field(default_factory=list)


class JunctionNode(BaseModel):
    """A junction node in the wall graph."""
    id: str
    position: Point
    degree: int
    kind: str = "junction"  # "endpoint", "junction", "corner"
    source_ids: List[str] = Field(default_factory=list)


class RoomFace(BaseModel):
    """An enclosed room face."""
    id: str
    polygon: Polygon  # Outer ring
    holes: List[Polygon] = Field(default_factory=list)
    area_px: float
    centroid: Point
    source_ids: List[str] = Field(default_factory=list)
    is_valid: bool = True
    invalid_reason: Optional[str] = None
    depends_on_inferred: bool = False  # True if depends on gap repair
    # T2-15 semantic classification: "room" | "sub_feature" | "non_room_loop" | "unclassified"
    semantic_class: str = "unclassified"


class OpeningAssociation(BaseModel):
    """Association between opening and wall."""
    opening_id: str
    wall_edge_id: Optional[str] = None
    position: Point
    overlap_px: float
    orientation: str  # "horizontal", "vertical", "diagonal"
    is_attached: bool
    source_ids: List[str] = Field(default_factory=list)


class DimensionObservation(BaseModel):
    """A dimension measurement."""
    id: str
    text: str  # Raw dimension text
    parsed_value: Optional[float] = None
    unit: Optional[str] = None  # "m", "mm", "px", etc.
    bbox: BBox
    associated_objects: List[str] = Field(default_factory=list)
    calibration_id: Optional[str] = None
    is_calibrated: bool = False
    source_ids: List[str] = Field(default_factory=list)


class CalibrationRecord(BaseModel):
    """A calibration record for metric conversion."""
    id: str
    page_index: int
    region: Optional[BBox] = None  # Scope of calibration
    unit: str  # "m", "mm", etc.
    pixels_per_unit: float
    source_dimension_ids: List[str] = Field(default_factory=list)
    consistency_check: Optional[float] = None  # Std dev of scale estimates
    is_valid: bool = True
    warnings: List[str] = Field(default_factory=list)


class VectorEvidence(BaseModel):
    """Vector-level evidence for a page."""
    frame_id: str
    primitives: List[VectorPrimitive] = Field(default_factory=list)
    wall_edges: List[WallEdge] = Field(default_factory=list)
    junction_nodes: List[JunctionNode] = Field(default_factory=list)
    room_faces: List[RoomFace] = Field(default_factory=list)
    opening_associations: List[OpeningAssociation] = Field(default_factory=list)
    dimensions: List[DimensionObservation] = Field(default_factory=list)
    calibrations: List[CalibrationRecord] = Field(default_factory=list)
    topology_diagnostics: List[str] = Field(default_factory=list)
    coverage: Dict[str, Any] = Field(default_factory=dict)
    # T2-03: estimated main plan region used for scope exclusion
    plan_region_bbox: Optional[BBox] = None
    plan_region_area_px: Optional[float] = None


class Comparison(BaseModel):
    """Comparison between ML and geometry evidence."""
    id: str
    relation: ComparisonRelation
    ml_observation_id: Optional[str] = None
    geometry_observation_id: Optional[str] = None
    supporting_geometry: Optional[Dict[str, Any]] = None
    disagreement_region: Optional[BBox] = None
    crop_id: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)  # IoU, distance, etc.


class Aggregate(BaseModel):
    """Aggregated count/measurement for a class."""
    id: str
    scope: str  # "page", "document", "region"
    page_index: Optional[int] = None
    class_name: str  # "wall", "room", "door", "window", "stairs"
    counting_unit: str  # "wall_edge", "room_face", "door_assembly", "stair_flight"
    member_ids: List[str] = Field(default_factory=list)
    duplicate_groups: List[List[str]] = Field(default_factory=list)
    value: Optional[float] = None  # Count or measurement
    unit: Optional[str] = None  # "count", "px", "px^2", "m", "m^2"
    status: str = "resolved"  # "resolved", "unresolved", "partial"
    coverage: Dict[str, Any] = Field(default_factory=dict)
    source_counts: Dict[str, int] = Field(default_factory=dict)  # Per-source counts
    warnings: List[str] = Field(default_factory=list)


class EvidenceDocument(BaseModel):
    """Complete evidence document for a drawing."""
    schema_version: str = SCHEMA_VERSION
    document_id: str
    source_path: str
    source_sha256: str
    pages: List[PageFrame] = Field(default_factory=list)
    pixel_evidence: List[PixelEvidence] = Field(default_factory=list)
    vector_evidence: List[VectorEvidence] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)  # All observations
    comparisons: List[Comparison] = Field(default_factory=list)
    aggregates: List[Aggregate] = Field(default_factory=list)
    calibrations: List[CalibrationRecord] = Field(default_factory=list)
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    diagnostics: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        # Convert numpy types to Python native types
        def convert_numpy(obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(item) for item in obj]
            return obj
        
        # Use python mode to avoid Pydantic trying to serialize numpy types
        data = self.model_dump(mode="python")
        data = convert_numpy(data)
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path.write_text(self.to_json(), encoding="utf-8")
    
    @classmethod
    def load(cls, path: Path) -> "EvidenceDocument":
        """Load from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def bbox_to_polygon(bbox: BBox) -> Polygon:
    """Convert bbox to polygon."""
    x0, y0, x1, y1 = bbox
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def polygon_area(polygon: Polygon) -> float:
    """Compute polygon area using shoelace formula."""
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def polygon_centroid(polygon: Polygon) -> Point:
    """Compute polygon centroid."""
    if len(polygon) < 3:
        return (0.0, 0.0)
    area = polygon_area(polygon)
    if area == 0:
        return (sum(p[0] for p in polygon) / len(polygon),
                sum(p[1] for p in polygon) / len(polygon))
    
    cx = 0.0
    cy = 0.0
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        cross = polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1]
        cx += (polygon[i][0] + polygon[j][0]) * cross
        cy += (polygon[i][1] + polygon[j][1]) * cross
    
    factor = 1.0 / (6.0 * area)
    return (cx * factor, cy * factor)


def bbox_intersection(bbox1: BBox, bbox2: BBox) -> Optional[BBox]:
    """Compute intersection of two bounding boxes."""
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])
    if x0 >= x1 or y0 >= y1:
        return None
    return (x0, y0, x1, y1)


def bbox_union(bbox1: BBox, bbox2: BBox) -> BBox:
    """Compute union of two bounding boxes."""
    return (
        min(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
        max(bbox1[3], bbox2[3]),
    )


def bbox_area(bbox: BBox) -> float:
    """Compute bbox area."""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def bbox_iou(bbox1: BBox, bbox2: BBox) -> float:
    """Compute IoU of two bounding boxes."""
    inter = bbox_intersection(bbox1, bbox2)
    if inter is None:
        return 0.0
    inter_area = bbox_area(inter)
    union_area = bbox_area(bbox1) + bbox_area(bbox2) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def point_in_bbox(point: Point, bbox: BBox) -> bool:
    """Check if point is inside bbox."""
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Check if point is inside polygon (ray casting)."""
    x, y = point
    n = len(polygon)
    inside = False
    
    for i in range(n):
        j = (i + 1) % n
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
    
    return inside


# ---------------------------------------------------------------------------
# planproof-door-result-v1 adapter (Imad & Johan backend contract)
# ---------------------------------------------------------------------------

def _package_version() -> str:
    """Return installed package version, falling back to a literal."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("planproof-evidence")
        except PackageNotFoundError:
            return "0.1.0"
    except Exception:
        return "0.1.0"


def to_door_result_v1(
    doc: "EvidenceDocument",
    page_frame_override: Optional[Dict[str, Any]] = None,
    page_index: int = 0,
) -> Dict[str, Any]:
    """Convert an EvidenceDocument into the planproof-door-result-v1 contract.

    Matches backend/verification.py AnalyzerResult:
      schema_version, source(PageFrame), version, status, mode, complete,
      detections[{id, bbox}].

    Args:
        doc: EvidenceDocument produced by analyze()
        page_frame_override: Optional dict merged into the source PageFrame
            (e.g. teammate-supplied frame identity). Keys override ours.
        page_index: Page to report (default 0).

    Returns:
        Dict suitable for POST /v1/door-result or direct JSON use.
    """
    # Locate the page frame
    frame = None
    for pf in doc.pages:
        if pf.page_index == page_index:
            frame = pf
            break

    if frame is None:
        return {
            "schema_version": "planproof-door-result-v1",
            "source": {
                "document_id": doc.document_id or "unknown",
                "page_index": page_index,
                "source_sha256": doc.source_sha256 or "0" * 64,
                "raster_sha256": "0" * 64,
                "width": 0,
                "height": 0,
                "coordinate_system": "pixel_edges_xy_top_left",
            },
            "version": _package_version(),
            "status": "unavailable",
            "mode": "independent",
            "complete": False,
            "detections": [],
        }

    width, height = frame.image_size
    source = {
        "document_id": frame.document_id,
        "page_index": frame.page_index,
        "source_sha256": frame.source_sha256,
        "raster_sha256": frame.raster_sha256,
        "width": int(width),
        "height": int(height),
        "coordinate_system": "pixel_edges_xy_top_left",
    }
    # Backend PageFrame is extra="forbid": only contract fields may appear
    # in `source`, so whitelist override keys accordingly.
    _FRAME_FIELDS = {
        "document_id", "page_index", "source_sha256", "raster_sha256",
        "width", "height", "coordinate_system",
    }
    if page_frame_override:
        source.update({
            k: v for k, v in page_frame_override.items()
            if k in _FRAME_FIELDS and v is not None
        })

    # Door observations on this page, sorted deterministically, capped at 80
    # (backend AnalyzerResult.detections max_length=80).
    door_obs = [
        obs for obs in doc.observations
        if obs.page_index == page_index
        and obs.kind == ObservationKind.DOOR
        and obs.bbox is not None
    ]
    door_obs.sort(key=lambda o: o.id)

    detections = []
    for obs in door_obs[:80]:
        x0, y0, x1, y1 = obs.bbox
        # Clip to canonical page bounds; keep positive area
        x0 = max(0.0, min(float(x0), float(width)))
        y0 = max(0.0, min(float(y0), float(height)))
        x1 = max(0.0, min(float(x1), float(width)))
        y1 = max(0.0, min(float(y1), float(height)))
        if x1 <= x0 or y1 <= y0:
            continue
        detections.append({"id": obs.id, "bbox": [x0, y0, x1, y1]})

    # Mode: hybrid only if any contributing observation depends on ML.
    mode = "hybrid" if any(
        obs.mode == Mode.HYBRID or obs.source_family == SourceFamily.HYBRID
        for obs in door_obs
    ) else "independent"

    # Completeness: the run is "ok"/complete when the page was fully
    # analyzed (vector topology present), detections were not truncated
    # at the contract cap, and the door aggregate has no unresolved
    # duplicate groups. Zero detections on a clean run is a legitimate
    # "ok" result — backend arbitration treats it as agreement only
    # when the ML analyzer also reports none.
    door_agg = None
    for agg in doc.aggregates:
        if (agg.class_name == "door" and agg.scope == "page"
                and agg.page_index == page_index):
            door_agg = agg
            break

    page_vec = any(
        ve.frame_id == f"p{page_index}-canonical"
        for ve in doc.vector_evidence
    )
    truncated = len(door_obs) > 80
    unresolved = door_agg is not None and door_agg.status != "resolved"
    degraded = (not page_vec) or truncated or unresolved

    status = "partial" if degraded else "ok"
    complete = not degraded

    return {
        "schema_version": "planproof-door-result-v1",
        "source": source,
        "version": _package_version(),
        "status": status,
        "mode": mode,
        "complete": complete,
        "detections": detections,
    }
