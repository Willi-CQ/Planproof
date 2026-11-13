"""
Semantic Fusion: Merge geometry extraction with YOLOv8 detections.

This module combines:
1. Line segments from PDF/raster extraction (geometry)
2. Object detections from YOLOv8 (semantics)

The result is annotated geometry with semantic labels.
"""
from __future__ import annotations

import difflib
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from planproof_geometry.ai.label_schema import PLAN_ELEMENT_CLASSES, to_plan_element_class

logger = logging.getLogger(__name__)


@dataclass
class AnnotatedSegment:
    """A line segment with semantic annotation."""
    id: str
    a: Tuple[float, float]  # Start point
    b: Tuple[float, float]  # End point
    class_name: str         # Semantic class
    confidence: float       # Classification confidence
    source: str             # "yolo", "heuristic", "manual"
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def length(self) -> float:
        """Calculate segment length."""
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])
    
    def midpoint(self) -> Tuple[float, float]:
        """Calculate segment midpoint."""
        return ((self.a[0] + self.b[0]) / 2, (self.a[1] + self.b[1]) / 2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "a": list(self.a),
            "b": list(self.b),
            "class": self.class_name,
            "confidence": self.confidence,
            "source": self.source,
            "length": self.length(),
            "meta": self.meta,
        }


@dataclass
class AnnotatedElement:
    """A higher-level element composed of multiple segments."""
    id: str
    class_name: str
    confidence: float
    segment_ids: List[str]
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    polygon: Optional[List[Tuple[float, float]]] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class": self.class_name,
            "confidence": self.confidence,
            "segment_ids": self.segment_ids,
            "bbox": list(self.bbox),
            "polygon": self.polygon,
            "properties": self.properties,
        }


@dataclass
class Room:
    """A detected room with label and properties."""
    id: str
    label: Optional[str]
    area_m2: Optional[float]
    polygon: List[Tuple[float, float]]
    center: Tuple[float, float]
    bounding_segments: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "area_m2": self.area_m2,
            "polygon": self.polygon,
            "center": list(self.center),
            "bounding_segments": self.bounding_segments,
            "meta": self.meta,
        }


@dataclass
class SemanticFusionResult:
    """Result of merging geometry with semantic detections."""
    segments: List[AnnotatedSegment]
    elements: List[AnnotatedElement]
    rooms: List[Room]
    texts: List[Dict[str, Any]]
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "segments": [s.to_dict() for s in self.segments],
            "elements": [e.to_dict() for e in self.elements],
            "rooms": [r.to_dict() for r in self.rooms],
            "texts": self.texts,
            "meta": self.meta,
            "stats": {
                "total_segments": len(self.segments),
                "total_elements": len(self.elements),
                "total_rooms": len(self.rooms),
                "segments_by_class": self._count_by_class(self.segments),
                "elements_by_class": self._count_by_class(self.elements),
            }
        }
    
    def _count_by_class(self, items: List) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in items:
            cls = item.class_name
            counts[cls] = counts.get(cls, 0) + 1
        return counts


def _point_in_bbox(
    point: Tuple[float, float],
    bbox: Tuple[float, float, float, float],
    margin: float = 0.0,
) -> bool:
    """Check if point is inside bounding box (with optional margin)."""
    x, y = point
    x1, y1, x2, y2 = bbox
    return (x1 - margin <= x <= x2 + margin and
            y1 - margin <= y <= y2 + margin)


def _segment_in_bbox(
    seg_a: Tuple[float, float],
    seg_b: Tuple[float, float],
    bbox: Tuple[float, float, float, float],
    threshold: float = 0.5,
    margin: float = 0.0,
) -> bool:
    """Check if segment is inside bounding box.

    Returns True if at least threshold fraction of segment is inside.

    Args:
        margin: Absolute margin added to each side of the bbox, in the same
                coordinate units as seg_a/seg_b/bbox. Pass a pre-computed
                value (e.g. 3 % of bbox size) instead of a pixel constant.
    """
    mid = ((seg_a[0] + seg_b[0]) / 2, (seg_a[1] + seg_b[1]) / 2)

    a_in = _point_in_bbox(seg_a, bbox, margin)
    b_in = _point_in_bbox(seg_b, bbox, margin)
    mid_in = _point_in_bbox(mid, bbox, margin)

    if threshold <= 0.5:
        return a_in or b_in or mid_in
    else:
        return (a_in and b_in) or (mid_in and (a_in or b_in))


def _bbox_overlap(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
) -> float:
    """Calculate IoU (Intersection over Union) between two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def _segment_bbox(
    seg_a: Tuple[float, float],
    seg_b: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    """Get bounding box of a segment."""
    return (
        min(seg_a[0], seg_b[0]),
        min(seg_a[1], seg_b[1]),
        max(seg_a[0], seg_b[0]),
        max(seg_a[1], seg_b[1]),
    )


def fuse_geometry_with_detections(
    extraction_dict: Dict[str, Any],
    detection_result: Dict[str, Any],
    px_to_coord_scale: float = 1.0,
    *,
    fusion_weight_confidence: float = 0.6,
    fusion_weight_iou: float = 0.4,
    class_overlap_thresholds: Optional[Dict[str, float]] = None,
) -> SemanticFusionResult:
    """
    Merge extracted geometry with YOLOv8 detections.
    
    Args:
        extraction_dict: Output from plan_extract_pdf/plan_extract_raster
        detection_result: Output from plan_yolo_detect.DetectionResult.to_dict()
        px_to_coord_scale: Scale factor to convert detection px coords to extraction coords
        fusion_weight_confidence: Weight for detection confidence in matching score (0–1)
        fusion_weight_iou: Weight for IoU overlap in matching score (0–1). Sollte 1 - fusion_weight_confidence sein.
        class_overlap_thresholds: Optional per-class overlap thresholds, z.B. {"door": 0.5, "window": 0.5}
    
    Returns:
        SemanticFusionResult with annotated segments and elements
    """
    # Parse geometry
    geometry = extraction_dict.get("geometry", [])
    texts = extraction_dict.get("texts", [])
    doc_meta = extraction_dict.get("doc", {})
    
    # Parse detections
    detections = detection_result.get("detections", [])
    det_meta = detection_result.get("meta", {})
    
    # Create annotated segments
    annotated_segments: List[AnnotatedSegment] = []
    segment_to_detection: Dict[str, str] = {}  # segment_id -> detection_id
    
    for geom in geometry:
        if geom.get("type") != "line":
            continue
        
        seg_id = geom.get("id", f"g{len(annotated_segments)}")
        a = geom.get("a", (0, 0))
        b = geom.get("b", (0, 0))
        
        if isinstance(a, dict):
            a = (a.get("x", 0), a.get("y", 0))
        if isinstance(b, dict):
            b = (b.get("x", 0), b.get("y", 0))
        
        a = (float(a[0]), float(a[1]))
        b = (float(b[0]), float(b[1]))
        
        # Find best matching detection using combined score (confidence + overlap)
        best_detection = None
        best_score = 0.0

        seg_bbox = _segment_bbox(a, b)

        for det in detections:
            det_bbox_raw = det.get("bbox", [0, 0, 0, 0])
            det_bbox_scaled = (
                det_bbox_raw[0] * px_to_coord_scale,
                det_bbox_raw[1] * px_to_coord_scale,
                det_bbox_raw[2] * px_to_coord_scale,
                det_bbox_raw[3] * px_to_coord_scale,
            )

            det_class = det.get("class", "")
            thresh = 0.5 if det_class in ("door", "window") else 0.3
            if class_overlap_thresholds and det_class in class_overlap_thresholds:
                thresh = class_overlap_thresholds[det_class]
            # Margin: 8 % for doors/windows (small elements need more tolerance),
            # 3 % for all other classes. Unit-safe regardless of coordinate scale.
            _bw = det_bbox_scaled[2] - det_bbox_scaled[0]
            _bh = det_bbox_scaled[3] - det_bbox_scaled[1]
            _margin_factor = 0.08 if det_class in ("door", "window") else 0.03
            _bbox_margin = max(_bw, _bh) * _margin_factor
            if _segment_in_bbox(a, b, det_bbox_scaled, threshold=thresh, margin=_bbox_margin):
                det_conf = det.get("confidence", 0.5)
                iou = _bbox_overlap(seg_bbox, det_bbox_scaled)
                score = fusion_weight_confidence * det_conf + fusion_weight_iou * iou
                if score > best_score:
                    best_score = score
                    best_detection = det
        
        # Assign class based on detection or use heuristics
        if best_detection:
            class_name = best_detection.get("class", "unknown")
            confidence = best_detection.get("confidence", 0.5)
            source = "yolo"
            segment_to_detection[seg_id] = best_detection.get("id", "")
            # Long segments inside door/window BBox are likely walls (ComputeOpeningsStep splits)
            seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
            if seg_len > 0.5 and class_name in ("door", "window"):
                class_name = "wall"
                confidence = 0.5
        else:
            # Apply heuristic classification
            class_name, confidence = _heuristic_classify_segment(
                a, b, geom, doc_meta, all_segments=geometry,
            )
            source = "heuristic"
        
        annotated_segments.append(AnnotatedSegment(
            id=seg_id,
            a=a,
            b=b,
            class_name=class_name,
            confidence=confidence,
            source=source,
            meta={
                "original_geom": geom,
                "detection_id": segment_to_detection.get(seg_id),
            }
        ))
    
    # --- Fusion diagnostics ---
    yolo_matched = sum(1 for s in annotated_segments if s.source == "yolo")
    heuristic_count = sum(1 for s in annotated_segments if s.source == "heuristic")
    logger.info(
        f"SemanticFusion: {yolo_matched} segments matched via YOLO, "
        f"{heuristic_count} via heuristic (total detections: {len(detections)})"
    )

    if detections and yolo_matched == 0:
        # YOLO returned detections but NONE matched any segment — likely coordinate mismatch
        # Log bbox ranges for debugging
        det_xs = [d["bbox"][0] * px_to_coord_scale for d in detections[:5]]
        det_ys = [d["bbox"][1] * px_to_coord_scale for d in detections[:5]]
        seg_xs = [s.a[0] for s in annotated_segments[:5]]
        seg_ys = [s.a[1] for s in annotated_segments[:5]]
        logger.warning(
            f"COORDINATE MISMATCH: {len(detections)} YOLO detections but 0 segment matches! "
            f"Detection X range (scaled): {min(det_xs):.2f}–{max(det_xs):.2f}, "
            f"Segment X range: {min(seg_xs):.4f}–{max(seg_xs):.4f}, "
            f"px_to_coord_scale={px_to_coord_scale:.6f}"
        )

    # Create elements from detections
    elements: List[AnnotatedElement] = []

    for det in detections:
        det_id = det.get("id", f"elem_{len(elements)}")
        det_bbox = det.get("bbox", [0, 0, 0, 0])
        det_bbox_scaled = (
            det_bbox[0] * px_to_coord_scale,
            det_bbox[1] * px_to_coord_scale,
            det_bbox[2] * px_to_coord_scale,
            det_bbox[3] * px_to_coord_scale,
        )
        
        # Find segments belonging to this detection
        seg_ids = [
            seg.id for seg in annotated_segments
            if segment_to_detection.get(seg.id) == det_id
        ]
        
        elements.append(AnnotatedElement(
            id=det_id,
            class_name=det.get("class", "unknown"),
            confidence=det.get("confidence", 0.5),
            segment_ids=seg_ids,
            bbox=det_bbox_scaled,
            polygon=det.get("polygon"),
            properties=det.get("meta", {}),
        ))
    
    # Inject synthetic door segments from wall-gap detection.
    # Always runs as supplementary — catches doors YOLO missed even when YOLO
    # found some openings.  Deduplicates against existing YOLO door detections.
    _opening_cls = {"door", "window"}
    _has_yolo_openings = any(
        d.get("class", "").lower() in _opening_cls for d in detections
    )
    door_gaps = _detect_door_gaps(geometry, doc_meta)
    # Deduplicate: skip gaps whose midpoint is within 0.50 m of an existing door
    existing_door_mids = []
    if _has_yolo_openings:
        for seg in annotated_segments:
            if seg.class_name in ("door", "door_single", "door_double"):
                mx = (seg.a[0] + seg.b[0]) / 2.0
                my = (seg.a[1] + seg.b[1]) / 2.0
                existing_door_mids.append((mx, my))
    for gap_idx, (gap_a, gap_b) in enumerate(door_gaps):
        gap_mid = ((gap_a[0] + gap_b[0]) / 2.0, (gap_a[1] + gap_b[1]) / 2.0)
        if any(
            math.hypot(gap_mid[0] - em[0], gap_mid[1] - em[1]) < 0.50
            for em in existing_door_mids
        ):
            continue  # too close to an existing YOLO door
        gap_id = f"door_gap_{gap_idx}"
        annotated_segments.append(AnnotatedSegment(
            id=gap_id,
            a=gap_a,
            b=gap_b,
            class_name="door",
            confidence=0.35,
            source="gap_heuristic",
            meta={"original_geom": {"id": gap_id, "type": "line",
                                     "a": list(gap_a), "b": list(gap_b)},
                  "detection_id": None},
        ))
    if door_gaps:
        logger.info(f"Gap-heuristic detected {len(door_gaps)} door opening(s)")

    # Prune low-value heuristic segments: short + low-confidence heuristic-only
    # segments are likely noise from raster line detection (furniture edges,
    # hatching, dimension tick marks).
    _prune_min_len = 0.50   # metres
    _prune_min_conf = 0.20
    before_prune = len(annotated_segments)
    annotated_segments = [
        s for s in annotated_segments
        if not (
            s.source == "heuristic"
            and s.confidence < _prune_min_conf
            and s.length() < _prune_min_len
        )
    ]
    if before_prune != len(annotated_segments):
        logger.info(
            "Pruned %d low-confidence heuristic segments (< %.1fm, < %.0f%% conf)",
            before_prune - len(annotated_segments),
            _prune_min_len,
            _prune_min_conf * 100,
        )

    # Extract rooms from room_label detections and texts
    rooms = _extract_rooms(annotated_segments, elements, texts)
    
    # Prepare text output
    text_output = []
    for t in texts:
        if isinstance(t, dict):
            text_output.append(t)
        else:
            # Handle TextSpan dataclass
            text_output.append({
                "id": getattr(t, "id", ""),
                "text": getattr(t, "text", ""),
                "bbox": getattr(t, "bbox", (0, 0, 0, 0)),
                "confidence": getattr(t, "conf", -1),
            })
    
    return SemanticFusionResult(
        segments=annotated_segments,
        elements=elements,
        rooms=rooms,
        texts=text_output,
        meta={
            "extraction_source": doc_meta.get("source_file", ""),
            "detection_engine": det_meta.get("engine", ""),
            "px_to_coord_scale": px_to_coord_scale,
            "total_geometry_lines": len(geometry),
            "total_detections": len(detections),
        }
    )


def _segment_angle_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Segment angle in degrees (0..180), 0 = horizontal, 90 = vertical."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    ang = math.degrees(math.atan2(abs(dy), abs(dx)))
    return ang


def _is_axis_aligned(a: Tuple[float, float], b: Tuple[float, float], tolerance_deg: float = 10.0) -> bool:
    """Check if segment is approximately horizontal or vertical."""
    ang = _segment_angle_deg(a, b)
    return ang <= tolerance_deg or ang >= (90.0 - tolerance_deg)


def _both_in_border_margin(
    a: Tuple[float, float],
    b: Tuple[float, float],
    all_segments: List[Dict[str, Any]],
    margin_frac: float = 0.08,
) -> bool:
    """Return True if both endpoints lie in the outer border strip of the plan.

    Dimension lines in German architectural drawings cluster in an ~8% margin
    around the page perimeter.  Windows never appear in this zone.
    """
    xs: List[float] = []
    ys: List[float] = []
    for s in all_segments:
        sa = s.get("a")
        sb = s.get("b")
        if sa:
            xs.append(float(sa[0])); ys.append(float(sa[1]))
        if sb:
            xs.append(float(sb[0])); ys.append(float(sb[1]))
    if not xs:
        return False
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = (max_x - min_x) * margin_frac
    dy = (max_y - min_y) * margin_frac

    def _in_margin(p: Tuple[float, float]) -> bool:
        return (p[0] < min_x + dx or p[0] > max_x - dx or
                p[1] < min_y + dy or p[1] > max_y - dy)

    return _in_margin(a) and _in_margin(b)


def _dist_point_to_segment(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """Euclidean distance from point p to segment a-b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / seg_len_sq))
    proj_x = a[0] + t * dx
    proj_y = a[1] + t * dy
    return math.hypot(p[0] - proj_x, p[1] - proj_y)


def _heuristic_classify_segment(
    a: Tuple[float, float],
    b: Tuple[float, float],
    geom: Dict[str, Any],
    doc_meta: Dict[str, Any],
    all_segments: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, float]:
    """
    Apply heuristic rules to classify a segment when no YOLO detection matches.

    Uses multiple signals:
    1. Extraction flags (dimension line markers from scale inference)
    2. Segment orientation (horizontal/vertical = more likely wall)
    3. Relative length (walls are typically 3-15% of page diagonal)
    4. Absolute length in meters (if scale is known)
    5. Parallel-segment proximity (wall pairs at 8-25cm distance)

    Returns (class_name, confidence)
    """
    length = math.hypot(b[0] - a[0], b[1] - a[1])

    # 1. Extraction flags from scale inference — dimension lines are non-structural
    if geom.get("is_dimension_line") or geom.get("is_dimension_mainline"):
        return "dimension_line", 0.8

    if geom.get("is_dimension_tick") or geom.get("is_dimension_extension"):
        return "dimension_line", 0.7

    # LLM-based zone classification: segments in dimension zones or outside
    # the building outline are non-structural.
    _llm_config = doc_meta.get("_pipeline_config", {})
    _page_size = doc_meta.get("page_size", {})
    _pw = float(_page_size.get("w", 0))
    _ph = float(_page_size.get("h", 0))

    if _pw > 0 and _ph > 0:
        _dim_zones = _llm_config.get("_llm_dimension_zones", [])
        if _dim_zones:
            from planproof_geometry.ai.plan_llm_preclassify import segment_in_dimension_zone
            if segment_in_dimension_zone(a, b, _pw, _ph, _dim_zones):
                return "dimension_line", 0.85

        _outline = _llm_config.get("_llm_building_outline")
        if _outline:
            from planproof_geometry.ai.plan_llm_preclassify import segment_outside_building_outline
            if segment_outside_building_outline(a, b, _pw, _ph, _outline):
                return "annotation", 0.80

    # 2. Page-relative metrics
    page_size = doc_meta.get("page_size", {})
    page_diag = math.hypot(
        page_size.get("w", 1000),
        page_size.get("h", 1000)
    )
    rel_length = length / page_diag if page_diag > 0 else 0

    # 3. Orientation bonus: axis-aligned segments are more likely structural
    axis_aligned = _is_axis_aligned(a, b)

    # 4. Check if scale is known for absolute-length reasoning
    transform = doc_meta.get("transform", {})
    points_to_m = transform.get("points_to_m")
    abs_length_m = length * points_to_m if points_to_m else None

    # Proximity-based dimension line classification: short segments (<0.5m)
    # near a known dimension mainline are likely tick marks or extension lines.
    if abs_length_m is not None and abs_length_m < 0.5 and all_segments and points_to_m:
        seg_mid = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        for other in all_segments:
            if not (other.get("is_dimension_mainline") or other.get("is_dimension_line")):
                continue
            oa = other.get("a")
            ob = other.get("b")
            if not oa or not ob:
                continue
            oA = (float(oa[0]), float(oa[1]))
            oB = (float(ob[0]), float(ob[1]))
            dist_pt = _dist_point_to_segment(seg_mid, oA, oB)
            if dist_pt * points_to_m < 0.30:
                return "dimension_line", 0.65

    # --- Annotation heuristics (border frame, hatching) ---

    # Border frame lines: segments in the outer 8% margin that run parallel to
    # the page edge are annotation lines, not structural walls.
    if all_segments and _both_in_border_margin(a, b, all_segments, margin_frac=0.08):
        if axis_aligned and rel_length > 0.10:
            return "annotation", 0.75

    # Very short non-structural noise: sub-3cm lines are rarely walls
    if abs_length_m is not None and abs_length_m < 0.03:
        return "annotation", 0.6

    # 5. Parallel-segment proximity: if a nearby segment runs parallel at
    #    typical wall thickness (8-25cm), both are likely a wall pair
    has_parallel_partner = False
    if all_segments and abs_length_m is not None and abs_length_m > 0.3:
        has_parallel_partner = _has_parallel_segment(
            a, b, all_segments, doc_meta,
            min_distance_m=0.06, max_distance_m=0.30,
        )

    # --- Opening heuristics (door/window without YOLO) ---
    # These produce intentionally low confidence so LLM refinement can upgrade.

    if abs_length_m is not None and all_segments:
        # Door heuristic: short segment (0.5-1.3m) that sits in a gap
        # between two wall-segment endpoints (both within snap distance).
        # Guard: must NOT lie in border margin (where dimension ticks look like door gaps).
        if 0.5 <= abs_length_m <= 1.3 and axis_aligned:
            if (_is_in_wall_gap(a, b, all_segments, doc_meta, snap_m=0.15)
                    and not _both_in_border_margin(a, b, all_segments)):
                return "door", 0.30

        # Window heuristic: segment (0.4-2.0m) with a parallel partner
        # at very close distance (frame thickness 0.04-0.10m) — typical for
        # window lines in architectural plans.
        # Guard: both endpoints must NOT lie in the border-margin zone (where
        # dimension lines live) to avoid classifying Maßketten as windows.
        if 0.4 <= abs_length_m <= 2.0 and axis_aligned:
            if (_has_parallel_segment(a, b, all_segments, doc_meta,
                                      min_distance_m=0.02, max_distance_m=0.08)
                    and not _both_in_border_margin(a, b, all_segments)):
                return "window", 0.25

    # --- Classification logic ---

    # Very short segments (< 1% page diagonal) are rarely structural
    if rel_length < 0.005:
        return "wall", 0.15

    # Wall classification based on multiple signals
    if abs_length_m is not None:
        if abs_length_m >= 0.5:
            conf = 0.45
            if axis_aligned:
                conf += 0.10
            if has_parallel_partner:
                conf += 0.15
            if abs_length_m >= 1.0:
                conf += 0.05
            return "wall", min(conf, 0.75)

        if abs_length_m < 0.15:
            return "wall", 0.20

        if axis_aligned:
            return "wall", 0.35
        return "wall", 0.25

    # Fallback: no scale known, use relative length only
    if rel_length > 0.05:
        conf = 0.40
        if axis_aligned:
            conf += 0.10
        return "wall", conf

    if rel_length > 0.02:
        if axis_aligned:
            return "wall", 0.35
        return "wall", 0.30

    if rel_length < 0.01:
        return "wall", 0.20

    return "wall", 0.25


def _has_parallel_segment(
    a: Tuple[float, float],
    b: Tuple[float, float],
    all_segments: List[Dict[str, Any]],
    doc_meta: Dict[str, Any],
    min_distance_m: float = 0.06,
    max_distance_m: float = 0.30,
) -> bool:
    """
    Check if there is a parallel segment at typical wall-thickness distance.
    This is a strong indicator that both segments form a wall pair.
    """
    transform = doc_meta.get("transform", {})
    points_to_m = transform.get("points_to_m", 1.0)
    if not points_to_m or points_to_m <= 0:
        return False

    seg_ang = _segment_angle_deg(a, b)
    seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
    seg_mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    min_dist_pt = min_distance_m / points_to_m
    max_dist_pt = max_distance_m / points_to_m
    min_parallel_len = seg_len * 0.4

    for other in all_segments:
        if other.get("type") != "line":
            continue
        oa = other.get("a")
        ob = other.get("b")
        if not oa or not ob:
            continue
        oax, oay = float(oa[0]), float(oa[1])
        obx, oby = float(ob[0]), float(ob[1])
        other_len = math.hypot(obx - oax, oby - oay)
        if other_len < min_parallel_len:
            continue

        other_ang = _segment_angle_deg((oax, oay), (obx, oby))
        ang_diff = abs(seg_ang - other_ang)
        if ang_diff > 8.0 and abs(ang_diff - 180.0) > 8.0:
            continue

        # Use perpendicular distance from other's midpoint to the line through a→b.
        # The old midpoint-to-midpoint Euclidean distance was incorrect for
        # non-overlapping parallel segments (gave false positives for shifted pairs).
        other_mid = ((oax + obx) / 2.0, (oay + oby) / 2.0)
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            continue
        # |AB × AM| / |AB|  — perpendicular distance
        perp_dist = abs(dx * (other_mid[1] - a[1]) - dy * (other_mid[0] - a[0])) / math.sqrt(seg_len_sq)
        # Projection parameter t: other_mid projected onto line a→b.
        # Allow slight overshoot (-0.5 … 1.5) so ends of slightly longer partner count.
        t = ((other_mid[0] - a[0]) * dx + (other_mid[1] - a[1]) * dy) / seg_len_sq
        if min_dist_pt <= perp_dist <= max_dist_pt and -0.5 <= t <= 1.5:
            return True

    return False


def _is_in_wall_gap(
    a: Tuple[float, float],
    b: Tuple[float, float],
    all_segments: List[Dict[str, Any]],
    doc_meta: Dict[str, Any],
    snap_m: float = 0.15,
) -> bool:
    """Check if segment sits in a gap between two wall-segment endpoints.

    A door is typically a short segment whose endpoints are each close to an
    endpoint of a longer neighbouring wall segment (the split pieces).
    """
    transform = doc_meta.get("transform", {})
    points_to_m = transform.get("points_to_m", 1.0)
    if not points_to_m or points_to_m <= 0:
        return False

    snap_pt = snap_m / points_to_m
    snap2 = snap_pt * snap_pt

    seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
    min_wall_len = seg_len * 1.5  # neighbouring walls should be longer than door

    a_near_wall = False
    b_near_wall = False

    for other in all_segments:
        if other.get("type") != "line":
            continue
        oa = other.get("a")
        ob = other.get("b")
        if not oa or not ob:
            continue
        oax, oay = float(oa[0]), float(oa[1])
        obx, oby = float(ob[0]), float(ob[1])
        other_len = math.hypot(obx - oax, oby - oay)
        if other_len < min_wall_len:
            continue

        # Check if endpoint a is near either endpoint of this wall segment
        if not a_near_wall:
            if (a[0] - oax) ** 2 + (a[1] - oay) ** 2 < snap2:
                a_near_wall = True
            elif (a[0] - obx) ** 2 + (a[1] - oby) ** 2 < snap2:
                a_near_wall = True

        if not b_near_wall:
            if (b[0] - oax) ** 2 + (b[1] - oay) ** 2 < snap2:
                b_near_wall = True
            elif (b[0] - obx) ** 2 + (b[1] - oby) ** 2 < snap2:
                b_near_wall = True

        if a_near_wall and b_near_wall:
            return True

    return False


def _detect_door_gaps(
    segments: List[Dict[str, Any]],
    doc_meta: Dict[str, Any],
    gap_min_m: float = 0.60,
    gap_max_m: float = 1.20,
    collinear_tol_m: float = 0.08,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Detect door openings as gaps in collinear wall segments.

    Architectural plans don't draw a line inside a door gap — the gap is the
    absence of wall.  This function finds pairs of collinear wall segments whose
    endpoints face each other across a gap of 0.60–1.20 m (typical door width).

    Returns a list of (point_a, point_b) pairs representing the centre line of
    each detected door opening (to be injected as synthetic door segments).
    """
    transform = doc_meta.get("transform", {})
    if not transform.get("points_to_m"):
        return []  # no transform applied yet — can't trust coordinates

    # After TransformCoordinatesStep, geometry is already in metres.
    # Use thresholds directly — no conversion needed.
    gap_min_thr = gap_min_m
    gap_max_thr = gap_max_m
    tol = collinear_tol_m

    lines = [s for s in segments if s.get("type") == "line"]

    # Separate into horizontal and vertical groups by angle
    h_segs: List[Dict] = []
    v_segs: List[Dict] = []
    for s in lines:
        a = s.get("a"); b = s.get("b")
        if not a or not b:
            continue
        ang = _segment_angle_deg((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
        if ang <= 25.0:
            h_segs.append(s)
        elif ang >= 65.0:
            v_segs.append(s)

    doors: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    def _find_gaps(group: List[Dict], axis: int) -> None:
        """axis=0 → horizontal (sort by x), axis=1 → vertical (sort by y)."""
        perp = 1 - axis  # perpendicular axis

        # Group by approximate perpendicular coordinate (shared wall line)
        buckets: Dict[int, List[Dict]] = {}
        for s in group:
            a = s.get("a"); b = s.get("b")
            if not a or not b:
                continue
            perp_val = (float(a[perp]) + float(b[perp])) / 2.0
            bucket_key = int(perp_val / tol)
            buckets.setdefault(bucket_key, []).append(s)

        for key, bucket in buckets.items():
            if len(bucket) < 2:
                continue
            # Within bucket, sort segments by their start coordinate on the axis
            def _seg_min(s: Dict) -> float:
                a = s.get("a"); b = s.get("b")
                return min(float(a[axis]), float(b[axis]))  # type: ignore[index]

            bucket_sorted = sorted(bucket, key=_seg_min)

            for i in range(len(bucket_sorted) - 1):
                s1 = bucket_sorted[i]
                s2 = bucket_sorted[i + 1]
                a1 = s1.get("a"); b1 = s1.get("b")
                a2 = s2.get("a"); b2 = s2.get("b")
                if not a1 or not b1 or not a2 or not b2:
                    continue

                # Perpendicular coords must be close (same wall line)
                p1 = (float(a1[perp]) + float(b1[perp])) / 2.0
                p2 = (float(a2[perp]) + float(b2[perp])) / 2.0
                if abs(p1 - p2) > tol * 2:
                    continue

                # Gap = distance from right end of s1 to left end of s2
                end1 = max(float(a1[axis]), float(b1[axis]))
                start2 = min(float(a2[axis]), float(b2[axis]))
                gap = start2 - end1
                if gap_min_thr <= gap <= gap_max_thr:
                    # Build synthetic door segment along the axis
                    perp_mid = (p1 + p2) / 2.0
                    if axis == 0:
                        pa: Tuple[float, float] = (end1, perp_mid)
                        pb: Tuple[float, float] = (start2, perp_mid)
                    else:
                        pa = (perp_mid, end1)
                        pb = (perp_mid, start2)
                    doors.append((pa, pb))

    _find_gaps(h_segs, axis=0)
    _find_gaps(v_segs, axis=1)
    return doors


def _polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """Shoelace formula for polygon area."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def _polygon_centroid(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Centroid of a polygon."""
    n = len(polygon)
    if n == 0:
        return (0.0, 0.0)
    cx = sum(p[0] for p in polygon) / n
    cy = sum(p[1] for p in polygon) / n
    return (cx, cy)


def _point_in_polygon(px: float, py: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


_ROOM_TYPE_MAP: Dict[str, str] = {
    # German
    "kueche": "kitchen", "küche": "kitchen", "kü": "kitchen",
    "bad": "bathroom", "wc": "bathroom", "dusche": "bathroom",
    "badezimmer": "bathroom", "gäste-wc": "bathroom", "gaeste-wc": "bathroom",
    "schlafzimmer": "bedroom", "sz": "bedroom", "schlafen": "bedroom",
    "kinderzimmer": "bedroom", "kind": "bedroom", "kizi": "bedroom",
    "wohnzimmer": "living", "wz": "living", "wohnen": "living",
    "flur": "hallway", "diele": "hallway", "gang": "hallway",
    "eingang": "hallway", "vorraum": "hallway",
    "balkon": "balcony", "terrasse": "terrace", "loggia": "balcony",
    "abstellraum": "storage", "ar": "storage", "abstell": "storage",
    "hwr": "utility", "hauswirtschaft": "utility",
    "buero": "office", "büro": "office", "arbeitszimmer": "office",
    "esszimmer": "dining", "essen": "dining", "ez": "dining",
    "ankleide": "dressing", "garderobe": "dressing",
    "keller": "basement", "technik": "utility", "heizung": "utility",
    "garage": "garage", "carport": "garage",
    "treppe": "stairs", "treppenhaus": "stairs",
    "zimmer": "bedroom",  # standalone "Zimmer" defaults to bedroom
    # English
    "kitchen": "kitchen", "bathroom": "bathroom", "bath": "bathroom",
    "bedroom": "bedroom", "living": "living", "living room": "living",
    "hallway": "hallway", "corridor": "hallway", "hall": "hallway",
    "balcony": "balcony", "terrace": "terrace",
    "storage": "storage", "utility": "utility", "laundry": "utility",
    "office": "office", "study": "office",
    "dining": "dining", "dining room": "dining",
    "garage": "garage",
}


def _ocr_normalize(text: str) -> str:
    """Apply common OCR error corrections for architectural room labels."""
    result = text.strip().lower()
    # Remove noise characters common in OCR of scanned plans
    for ch in (".", "-", "'", "`", "\u2019", "\u2018"):
        result = result.replace(ch, "")
    # Collapse whitespace
    result = "".join(result.split())
    # Common OCR substitutions
    for old, new in (("l", "i"), ("1", "i"), ("|", "i"), ("0", "o")):
        result = result.replace(old, new)
    return result


# Known OCR-garbled forms of German room names
_OCR_CORRECTIONS: Dict[str, str] = {
    "zimmer": "zimmer", "zlmmer": "zimmer", "z1mmer": "zimmer",
    "kachn": "kueche", "kuche": "kueche",
    "saion": "salon",
}


def _classify_room_type(label: Optional[str]) -> str:
    """Classify room type from label text (German/English).

    Uses exact match, substring match, OCR correction, and fuzzy match
    (via difflib) to handle OCR errors such as "Zlmmer" -> "Zimmer".
    """
    if not label:
        return ""
    normalized = label.strip().lower().replace(".", "").replace("-", "")
    # Exact match
    if normalized in _ROOM_TYPE_MAP:
        return _ROOM_TYPE_MAP[normalized]
    # Substring match
    for key, room_type in _ROOM_TYPE_MAP.items():
        if key in normalized:
            return room_type

    # OCR-corrected match: normalize then try exact + substring
    corrected = _ocr_normalize(label)
    if corrected in _OCR_CORRECTIONS:
        corrected = _OCR_CORRECTIONS[corrected]
    if corrected in _ROOM_TYPE_MAP:
        return _ROOM_TYPE_MAP[corrected]
    for key, room_type in _ROOM_TYPE_MAP.items():
        if key in corrected:
            return room_type

    # Fuzzy match — lowered cutoff from 0.80 to 0.55 for OCR-garbled text
    candidates = difflib.get_close_matches(corrected, _ROOM_TYPE_MAP.keys(), n=1, cutoff=0.55)
    if candidates:
        logger.debug("OCR fuzzy correction: %r -> %r (normalized: %r)", label, candidates[0], corrected)
        return _ROOM_TYPE_MAP[candidates[0]]
    return ""


def _find_label_for_polygon(
    polygon: List[Tuple[float, float]],
    texts: List[Dict[str, Any]],
) -> Optional[str]:
    """Find the first text whose center falls inside the polygon."""
    for text in texts:
        text_str = text.get("text", "").strip()
        if not text_str or len(text_str) > 40:
            continue
        text_bbox = text.get("bbox", (0, 0, 0, 0))
        if isinstance(text_bbox, (list, tuple)) and len(text_bbox) == 4:
            tcx = (text_bbox[0] + text_bbox[2]) / 2
            tcy = (text_bbox[1] + text_bbox[3]) / 2
            if _point_in_polygon(tcx, tcy, polygon):
                return text_str
    return None


def _detect_rooms_from_walls(
    segments: List[AnnotatedSegment],
    texts: List[Dict[str, Any]],
) -> List[Room]:
    """Detect rooms by finding minimal enclosed cycles in wall geometry.

    Builds a planar graph from wall segment endpoints (snapped within tolerance),
    then finds minimal faces via left-turn following.
    """
    wall_classes = {"wall", "raw_line"}
    # Pre-filter: keep only structural wall segments.
    # Exclude dimension lines, very short noise segments, and segments
    # already classified as openings (they break the planar graph).
    wall_segs = [
        s for s in segments
        if (s.class_name in wall_classes
            and s.length() >= 0.30  # skip sub-30cm noise
            and not s.meta.get("original_geom", {}).get("is_dimension_mainline")
            and not s.meta.get("original_geom", {}).get("is_dimension_extension"))
    ]
    if len(wall_segs) < 3:
        return []

    # --- Build graph with snapped nodes ---
    # Adaptive snap tolerance: 3 % of the median wall length, clamped to 0.08–0.20 m.
    # This handles both small apartment plans and large commercial floor plans.
    if wall_segs:
        median_wall_len = sorted(s.length() for s in wall_segs)[len(wall_segs) // 2]
        snap_tol = max(0.08, min(0.20, median_wall_len * 0.03))
    else:
        snap_tol = 0.15
    snap2 = snap_tol * snap_tol
    nodes: List[Tuple[float, float]] = []
    node_map: Dict[int, int] = {}  # seg_endpoint_hash → node_idx

    def _snap_or_add(pt: Tuple[float, float]) -> int:
        for idx, n in enumerate(nodes):
            if (pt[0] - n[0]) ** 2 + (pt[1] - n[1]) ** 2 < snap2:
                return idx
        nodes.append(pt)
        return len(nodes) - 1

    edges: List[Tuple[int, int]] = []
    seg_id_by_edge: Dict[Tuple[int, int], str] = {}
    for seg in wall_segs:
        ni = _snap_or_add(seg.a)
        nj = _snap_or_add(seg.b)
        if ni == nj:
            continue
        edges.append((ni, nj))
        edges.append((nj, ni))
        seg_id_by_edge[(ni, nj)] = seg.id
        seg_id_by_edge[(nj, ni)] = seg.id

    if len(edges) < 6:
        return []

    # Adjacency list sorted by angle (for left-turn following)
    from collections import defaultdict
    adj: Dict[int, List[int]] = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)

    def _angle(u: int, v: int) -> float:
        return math.atan2(nodes[v][1] - nodes[u][1], nodes[v][0] - nodes[u][0])

    # Sort adjacency by angle
    for u in adj:
        adj[u].sort(key=lambda v: _angle(u, v))

    # --- Find minimal faces via left-turn following ---
    used_edges: set = set()
    faces: List[List[int]] = []

    for start_u, start_v in edges:
        if (start_u, start_v) in used_edges:
            continue

        face = [start_u]
        prev, cur = start_u, start_v
        max_steps = len(nodes) + 2

        for _ in range(max_steps):
            face.append(cur)
            used_edges.add((prev, cur))

            neighbors = adj.get(cur, [])
            if not neighbors:
                break

            # Find the edge we came from, then take the next CCW edge
            incoming_angle = _angle(cur, prev)
            best_next = None
            best_diff = float("inf")
            for nb in neighbors:
                if nb == prev and len(neighbors) > 1:
                    continue
                out_angle = _angle(cur, nb)
                # Angle difference: smallest positive turn from incoming direction
                diff = (out_angle - incoming_angle) % (2 * math.pi)
                if diff < best_diff:
                    best_diff = diff
                    best_next = nb

            if best_next is None:
                break

            prev = cur
            cur = best_next

            if cur == start_u:
                used_edges.add((prev, cur))
                break
        else:
            continue

        if len(face) >= 3 and face[-1] == face[0]:
            face = face[:-1]  # remove duplicate start
            if 3 <= len(face) <= 25:  # relaxed from 12: L/U-shaped rooms can have more edges
                faces.append(face)

    # --- Convert faces to rooms ---
    rooms: List[Room] = []
    room_id = 1

    # Compute total bounding area for background rejection
    all_xs = [n[0] for n in nodes]
    all_ys = [n[1] for n in nodes]
    total_bbox_area = 0.0
    if all_xs and all_ys:
        total_bbox_area = (max(all_xs) - min(all_xs)) * (max(all_ys) - min(all_ys))

    for face in faces:
        polygon = [nodes[ni] for ni in face]
        area = _polygon_area(polygon)

        # Filter: plausible room size (0.8-400 m2). Lower bound lowered to 0.8 m²
        # to include small WC/storage rooms (Abstellraum, Tresor, Abst.).
        if area < 0.8 or area > 400.0:
            continue

        # Reject background: a face covering >60% of the bounding area is
        # likely the exterior / white background, not a real room.
        if total_bbox_area > 0 and area / total_bbox_area > 0.60:
            continue

        centroid = _polygon_centroid(polygon)
        label = _find_label_for_polygon(polygon, texts)

        # Collect bounding segment IDs
        bounding_seg_ids = []
        for i in range(len(face)):
            j = (i + 1) % len(face)
            edge_key = (face[i], face[j])
            sid = seg_id_by_edge.get(edge_key)
            if sid:
                bounding_seg_ids.append(sid)

        rooms.append(Room(
            id=f"room_{room_id}",
            label=label,
            area_m2=round(area, 1),
            polygon=polygon,
            center=centroid,
            bounding_segments=bounding_seg_ids,
            meta={
                "source": "wall_geometry",
                "room_type": _classify_room_type(label),
            },
        ))
        room_id += 1

    return rooms


def _extract_rooms(
    segments: List[AnnotatedSegment],
    elements: List[AnnotatedElement],
    texts: List[Dict[str, Any]],
) -> List[Room]:
    """Extract rooms from YOLO room_label elements, enriched with wall-geometry detection."""
    rooms: List[Room] = []
    room_id = 1

    # 1) YOLO-based rooms from room_label elements
    room_elements = [e for e in elements if e.class_name == "room_label"]

    for elem in room_elements:
        label = None
        for text in texts:
            text_bbox = text.get("bbox", (0, 0, 0, 0))
            if isinstance(text_bbox, (list, tuple)) and len(text_bbox) == 4:
                text_center = (
                    (text_bbox[0] + text_bbox[2]) / 2,
                    (text_bbox[1] + text_bbox[3]) / 2
                )
                # Margin: 5 % of the room bbox size — unit-safe for both
                # pixel and meter coordinate systems (fixed 20 px was too large).
                _rw = elem.bbox[2] - elem.bbox[0]
                _rh = elem.bbox[3] - elem.bbox[1]
                _room_margin = max(_rw, _rh) * 0.05
                if _point_in_bbox(text_center, elem.bbox, margin=_room_margin):
                    label = text.get("text", "")
                    break

        center = (
            (elem.bbox[0] + elem.bbox[2]) / 2,
            (elem.bbox[1] + elem.bbox[3]) / 2
        )

        polygon = [
            (elem.bbox[0], elem.bbox[1]),
            (elem.bbox[2], elem.bbox[1]),
            (elem.bbox[2], elem.bbox[3]),
            (elem.bbox[0], elem.bbox[3]),
        ]

        area_m2 = _polygon_area(polygon)
        if area_m2 < 0.1:
            area_m2 = None
        else:
            area_m2 = round(area_m2, 1)

        rooms.append(Room(
            id=f"room_{room_id}",
            label=label,
            area_m2=area_m2,
            polygon=polygon,
            center=center,
            bounding_segments=elem.segment_ids,
            meta={
                "source": "yolo",
                "source_element_id": elem.id,
                "room_type": _classify_room_type(label),
            },
        ))
        room_id += 1

    # 2) Geometry-based room detection (fallback or enrichment)
    geo_rooms = _detect_rooms_from_walls(segments, texts)

    if not rooms:
        # No YOLO rooms — use geometry-based rooms directly
        for gr in geo_rooms:
            gr.id = f"room_{room_id}"
            rooms.append(gr)
            room_id += 1
    else:
        # Merge: add geometry rooms that don't overlap existing YOLO rooms
        for gr in geo_rooms:
            gc = gr.center
            overlaps = False
            for existing in rooms:
                # Simple overlap check: centroid of one inside polygon of other
                if _point_in_polygon(gc[0], gc[1], existing.polygon):
                    # Enrich existing YOLO room with geometry-based area if missing
                    if existing.area_m2 is None and gr.area_m2 is not None:
                        existing.area_m2 = gr.area_m2
                    if not existing.bounding_segments and gr.bounding_segments:
                        existing.bounding_segments = gr.bounding_segments
                    if existing.meta.get("source") == "yolo":
                        existing.polygon = gr.polygon  # prefer wall-bounded polygon
                    overlaps = True
                    break
            if not overlaps:
                gr.id = f"room_{room_id}"
                rooms.append(gr)
                room_id += 1

    return rooms


def merge_manual_annotations(
    fusion_result: SemanticFusionResult,
    annotations: List[Dict[str, Any]],
) -> SemanticFusionResult:
    """
    Merge manual user annotations into the fusion result.
    
    Annotations format:
    [
        {
            "segment_id": "g123",
            "class": "door_single",
            "confidence": 1.0,
            "user_id": "user123"
        },
        ...
    ]
    """
    # Create lookup for annotations
    annotation_map = {a["segment_id"]: a for a in annotations if "segment_id" in a}
    
    # Update segments with manual annotations
    updated_segments = []
    for seg in fusion_result.segments:
        if seg.id in annotation_map:
            ann = annotation_map[seg.id]
            updated_segments.append(AnnotatedSegment(
                id=seg.id,
                a=seg.a,
                b=seg.b,
                class_name=ann.get("class", seg.class_name),
                confidence=ann.get("confidence", 1.0),
                source="manual",
                meta={
                    **seg.meta,
                    "previous_class": seg.class_name,
                    "previous_source": seg.source,
                    "annotator": ann.get("user_id"),
                }
            ))
        else:
            updated_segments.append(seg)
    
    return SemanticFusionResult(
        segments=updated_segments,
        elements=fusion_result.elements,
        rooms=fusion_result.rooms,
        texts=fusion_result.texts,
        meta={
            **fusion_result.meta,
            "manual_annotations_count": len(annotations),
        }
    )


def export_for_training(
    fusion_result: SemanticFusionResult,
    image_size: Tuple[int, int],
    output_dir: str,
    image_name: str,
) -> Dict[str, str]:
    """
    Export fusion result as YOLO training annotations.
    
    Returns paths to created files.
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export as YOLO format
    from planproof_geometry.ai.plan_yolo_detect import export_annotations_for_training
    
    # Convert elements to detection format
    detections = []
    for elem in fusion_result.elements:
        detections.append({
            "class": elem.class_name,
            "bbox": elem.bbox,
        })
    
    # Also export segments classified by YOLO or manual
    for seg in fusion_result.segments:
        if seg.source in ("yolo", "manual") and seg.class_name not in ("furniture", "room_label"):
            seg_bbox = _segment_bbox(seg.a, seg.b)
            detections.append({
                "class": seg.class_name,
                "bbox": seg_bbox,
            })
    
    annotation_path = str(output_path / f"{image_name}.txt")
    export_annotations_for_training(
        detections=detections,
        image_size=image_size,
        output_path=annotation_path,
        format="yolo",
    )
    
    return {
        "annotation": annotation_path,
    }
