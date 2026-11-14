"""Canonical pixel-edge frames, strict JSON, stable provenance and artifact binding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

CLASSES = ("wall", "room", "door", "window", "stairs")
SCHEMA = "planproof-evidence-v1"
COORDINATES = "pixel_edges_xy_top_left"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb_hash(image):
    image = np.asarray(image, dtype=np.uint8)
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def stable_id(namespace, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return namespace + ":" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def transform_points(points, matrix):
    points = np.asarray(points, dtype=float)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-12):
        raise ValueError("Invalid coordinate transform")
    return (homogeneous[:, :2] / homogeneous[:, 2:]).tolist()


def validate_bbox(bbox, size):
    if len(bbox) != 4 or not np.isfinite(bbox).all():
        raise ValueError("Box must contain four finite numbers")
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise ValueError(f"Invalid/out-of-frame box {bbox} for {size}")


def polygon_json(polygon):
    return {
        "outer": [list(p) for p in polygon.exterior.coords],
        "holes": [[list(p) for p in r.coords] for r in polygon.interiors],
    }


def polygons_of(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [p for g in getattr(geometry, "geoms", []) for p in polygons_of(g)]


def item_geometry(item):
    from shapely.geometry import box
    from shapely.ops import unary_union

    if item.get("polygons"):
        polygons = [Polygon(p["outer"], p.get("holes", [])) for p in item["polygons"]]
        if any(not p.is_valid or p.is_empty or not np.isfinite(p.bounds).all() for p in polygons):
            raise ValueError("Invalid polygon (holes and ring topology must be retained)")
        return unary_union(polygons)
    return box(*item["bbox"])


@dataclass(frozen=True)
class Config:
    version: str = "integrated-evidence-0.3"
    pixel_backend: str = "track1"
    enable_ocr: bool = True
    ocr_languages: str = "eng"
    max_side: int = 1400
    max_segments: int = 1800
    max_pairs: int = 1200
    min_line_px: float = 10.0
    angle_deg: float = 5.0
    min_room_area_fraction: float = 0.001
    max_room_area_fraction: float = 0.65
    pdf_dpi: float = 120.0
    pdf_max_pixels: int = 24_000_000
    pdf_total_pixel_budget: int = 64_000_000
    match_iou: float = 0.5
    min_arc_radius_px: int = 10
    max_arc_radius_fraction: float = 0.20
    min_arc_support: float = 0.60
    min_stair_treads: int = 5

    def __post_init__(self):
        if self.pixel_backend not in ("track1", "reference"):
            raise ValueError("Unknown pixel backend")
        if (
            not isinstance(self.enable_ocr, bool)
            or not self.ocr_languages
            or len(self.ocr_languages) > 100
        ):
            raise ValueError("Invalid OCR configuration")
        if not (128 <= self.max_side <= 4096 and 1 <= self.max_segments <= 10000):
            raise ValueError("Invalid bounded extraction budget")
        if not (0 < self.angle_deg < 45 and 0 < self.match_iou <= 1):
            raise ValueError("Invalid geometry thresholds")
        if not (0 < self.min_room_area_fraction < self.max_room_area_fraction < 1):
            raise ValueError("Invalid room area limits")
        if self.max_pairs < 1 or self.pdf_dpi <= 0 or self.pdf_max_pixels < 1:
            raise ValueError("Invalid render/pair budget")
        if self.pdf_total_pixel_budget < self.pdf_max_pixels:
            raise ValueError("PDF total pixel budget must fit at least one page")
        if not (0 < self.min_arc_support <= 1 and self.min_stair_treads >= 3):
            raise ValueError("Invalid symbol thresholds")


@dataclass(frozen=True)
class PageFrame:
    document_id: str
    page_index: int
    source_sha256: str
    raster_sha256: str
    image_size: list[int]
    source_to_canonical: list[list[float]]
    canonical_to_source: list[list[float]]
    metadata: dict = field(default_factory=dict)
    coordinate_system: str = COORDINATES

    @property
    def id(self):
        return f"p{self.page_index}-canonical"

    def to_dict(self):
        return {"id": self.id, **asdict(self)}


def validate_document(document):
    """Reject invalid frame bindings, cycles, invalid geometry and hybrid lineage lies."""
    if document["schema_version"] != SCHEMA:
        raise ValueError("Unknown evidence schema")
    frames = {p["frame"]["id"]: p["frame"] for p in document["pages"]}
    if len(frames) != len(document["pages"]):
        raise ValueError("Duplicate page frame IDs")
    for artifact in document.get("artifacts", []):
        path = artifact["path"]
        if Path(path).is_absolute() or ".." in Path(path).parts or ":" in path or "\\" in path:
            raise ValueError("Artifact path escapes job directory")
    observations = document["observations"]
    by_id = {o["id"]: o for o in observations}
    if len(by_id) != len(observations):
        raise ValueError("Duplicate observation IDs")
    for frame in frames.values():
        if (
            frame["document_id"] != document["document_id"]
            or frame["source_sha256"] != document["source_sha256"]
        ):
            raise ValueError("Document/page source identity mismatch")
        a = np.asarray(frame["source_to_canonical"])
        b = np.asarray(frame["canonical_to_source"])
        if a.shape != (3, 3) or not np.allclose(a @ b, np.eye(3), atol=1e-7):
            raise ValueError("Frame transforms are not inverses")
    visited, active = set(), set()

    def visit(oid):
        if oid in active:
            raise ValueError("Cyclic observation dependencies")
        if oid in visited:
            return
        if oid not in by_id:
            raise ValueError("Missing observation dependency: " + oid)
        active.add(oid)
        o = by_id[oid]
        if o["source_family"] not in ("classical_pixels", "native_pdf", "ocr", "ml", "hybrid"):
            raise ValueError("Unknown source family")
        if o["mode"] not in ("independent", "hybrid"):
            raise ValueError("Unknown extraction mode")
        if o["source_family"] in ("ml", "hybrid") and o["mode"] != "hybrid":
            raise ValueError("Independent evidence depends on ML source")
        frame = frames[o["frame_id"]]
        if o["page_index"] != frame["page_index"]:
            raise ValueError("Page/frame mismatch")
        validate_bbox(o["bbox"], frame["image_size"])
        if o.get("polygons"):
            g = item_geometry(o)
            w, h = frame["image_size"]
            if g.bounds[0] < 0 or g.bounds[1] < 0 or g.bounds[2] > w or g.bounds[3] > h:
                raise ValueError("Polygon outside frame")
        for parent in o.get("dependency_ids", []):
            visit(parent)
            if o["mode"] == "independent" and by_id[parent]["mode"] == "hybrid":
                raise ValueError("Independent observation depends on floorplan ML")
        active.remove(oid)
        visited.add(oid)

    for oid in by_id:
        visit(oid)
    for aggregate in document["aggregates"]:
        if any(oid not in by_id for oid in aggregate["member_ids"]):
            raise ValueError("Aggregate has unknown member")
        if aggregate.get("value") is not None and aggregate["value"] != len(
            aggregate["member_ids"]
        ):
            raise ValueError("Count does not equal explicit membership")
    json.dumps(document, allow_nan=False)
    return document


def validate_pixels(data, image_size):
    """Validate the replaceable full-page adapter before using any measurements."""
    w, h = data["working_size"]
    if not isinstance(w, int) or not isinstance(h, int) or min(w, h) < 1:
        raise ValueError("Invalid working raster size")
    expected = [[image_size[0] / w, 0, 0], [0, image_size[1] / h, 0], [0, 0, 1]]
    if not np.allclose(data["working_to_canonical"], expected, atol=1e-9):
        raise ValueError("Adapter must return the recorded full-page scaling transform")
    for name in ("ink", "wall_mask", "skeleton_mask"):
        mask = np.asarray(data[name])
        if mask.shape != (h, w) or not np.isin(mask, [0, 1]).all():
            raise ValueError("Invalid binary adapter mask: " + name)
    for candidate in data["candidates"]:
        if candidate["class"] not in CLASSES:
            raise ValueError("Unknown candidate class; abstentions belong in unclassified_patterns")
        geometry = candidate["geometry"]
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("Invalid adapter candidate geometry")
        validate_bbox(geometry.bounds, [w, h])
        if not np.isfinite(candidate["support"]) or not 0 <= candidate["support"] <= 1:
            raise ValueError("Invalid rule-support score")
    if data.get("mode", "independent") != "independent":
        raise ValueError("ML-driven pixels cannot enter the independent adapter")
