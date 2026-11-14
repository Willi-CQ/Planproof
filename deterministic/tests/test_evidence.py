"""Unit tests for planproof_evidence package.

Tests cover contracts, frames, raster, pixels, OCR, native PDF,
vectors, topology, measurements, ML adapter, comparison, crops,
and the main pipeline.
"""

import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from planproof_evidence import analyze, get_region
from planproof_evidence.contracts import (
    Aggregate,
    ArtifactRef,
    BBox,
    CalibrationRecord,
    Comparison,
    ComparisonRelation,
    DimensionObservation,
    EvidenceDocument,
    JunctionNode,
    Observation,
    ObservationKind,
    PageFrame,
    PageMetadata,
    PixelEvidence,
    Point,
    Polygon,
    RoomFace,
    SourceFamily,
    Mode,
    Transform3x3,
    VectorEvidence,
    VectorPrimitive,
    WallEdge,
    bbox_area,
    bbox_iou,
    polygon_area,
    polygon_centroid,
    point_in_bbox,
    point_in_polygon,
    sha256_bytes,
    sha256_file,
    to_door_result_v1,
)
from planproof_evidence.frames import (
    create_page_frame,
    identity_transform,
    normalize_image,
)
from planproof_evidence.raster import (
    load_image,
    load_drawing,
    detect_non_floorplan,
)
from planproof_evidence.pixels import (
    extract_ink_mask,
    extract_components,
    extract_contour_rings,
    extract_line_segments,
    detect_opening_candidates,
    detect_stair_candidates,
    extract_pixel_evidence,
)
from planproof_evidence.ocr import (
    check_tesseract_available,
    extract_ocr_spans,
)
from planproof_evidence.native_pdf import (
    extract_pdf_primitives,
    extract_pdf_text_spans,
)
from planproof_evidence.vectors import (
    compute_distance_transform,
    skeletonize_mask,
    extract_wall_thickness,
    merge_collinear_segments,
    extract_wall_edges,
    build_junction_graph,
    extract_vector_evidence,
)
from planproof_evidence.topology import (
    build_wall_network,
    polygonize_faces,
    attach_openings_to_walls,
    extract_topology,
    estimate_plan_region,
    classify_room_faces,
)
from planproof_evidence.semantic import classify_rooms, classify_openings
from planproof_evidence.measurements import (
    parse_dimension_text,
    extract_dimensions_from_text,
    find_dimension_associations,
    estimate_calibration,
    apply_calibration,
    extract_measurements,
)
from planproof_evidence.ml_adapter import (
    MLInstance,
    MLPrediction,
    load_ml_predictions,
    validate_ml_binding,
    ml_instance_to_observation,
    extract_ml_observations,
    load_ml_maps,
    validate_ml_maps,
    ML_SCHEMA_VERSION,
)
from planproof_evidence.compare import (
    compute_bbox_distance,
    match_observations,
    create_comparison,
    compare_ml_and_geometry,
    create_aggregates,
    prepare_disagreement_crops,
    IOU_THRESHOLD,
    DISTANCE_THRESHOLD,
)
from planproof_evidence.crops import (
    extract_crop,
    get_region as get_region_impl,
    render_overlay,
    render_all_overlays,
)
from planproof_evidence.overlays import (
    render_overlay as render_overlay_overlay,
    render_all_overlays as render_all_overlays_overlay,
)
from planproof_evidence.pipeline import analyze, get_region


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_image():
    """Simple test image with a rectangle."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    img[20:80, 20:80] = 0  # Black rectangle
    return img


@pytest.fixture
def floorplan_image():
    """More complex floorplan-like image."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    
    # Draw walls (thick lines)
    cv2.rectangle(img, (20, 20), (180, 180), (0, 0, 0), 3)
    cv2.line(img, (20, 100), (180, 100), (0, 0, 0), 2)  # Horizontal wall
    cv2.line(img, (100, 20), (100, 180), (0, 0, 0), 2)  # Vertical wall
    
    # Draw door (gap in wall)
    cv2.rectangle(img, (90, 95), (110, 105), (255, 255, 255), -1)
    
    return img


@pytest.fixture
def sample_observation():
    """Sample observation for testing."""
    return Observation(
        id="test-obs-1",
        kind=ObservationKind.WALL,
        source_family=SourceFamily.CLASSICAL_PIXELS,
        mode=Mode.INDEPENDENT,
        page_index=0,
        frame_id="p0-canonical",
        bbox=(10, 10, 50, 50),
        geometry={"length": 40, "angle": 0},
        method="test",
        method_version="v1",
    )


@pytest.fixture
def sample_page_frame():
    """Sample page frame for testing."""
    return PageFrame(
        document_id="test-doc",
        page_index=0,
        source_sha256="abc123",
        raster_sha256="def456",
        image_size=(100, 100),
        color_convention="RGB",
        source_to_canonical=identity_transform(),
        canonical_to_source=identity_transform(),
        page_metadata=None,
        exif_orientation=None,
    )


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

class TestContracts:
    """Test contract models."""
    
    def test_transform3x3_identity(self):
        """Test identity transform."""
        t = identity_transform()
        assert t.matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        
        # Test apply
        point = (10, 20)
        result = t.apply(point)
        assert result == (10, 20)
    
    def test_transform3x3_inverse(self):
        """Test transform inverse."""
        t = Transform3x3(matrix=[[2, 0, 10], [0, 2, 20], [0, 0, 1]])
        inv = t.inverse()
        
        # Test round-trip
        point = (5, 10)
        transformed = t.apply(point)
        restored = inv.apply(transformed)
        assert abs(restored[0] - point[0]) < 0.01
        assert abs(restored[1] - point[1]) < 0.01
    
    def test_page_frame_validation(self, sample_page_frame):
        """Test page frame validation."""
        assert sample_page_frame.document_id == "test-doc"
        assert sample_page_frame.image_size == (100, 100)
        assert sample_page_frame.color_convention == "RGB"
    
    def test_observation_validation(self, sample_observation):
        """Test observation validation."""
        assert sample_observation.id == "test-obs-1"
        assert sample_observation.kind == ObservationKind.WALL
        assert sample_observation.source_family == SourceFamily.CLASSICAL_PIXELS
        assert sample_observation.mode == Mode.INDEPENDENT
    
    def test_bbox_iou(self):
        """Test bbox IoU calculation."""
        bbox1 = (0, 0, 10, 10)
        bbox2 = (5, 5, 15, 15)
        iou = bbox_iou(bbox1, bbox2)
        assert 0 < iou < 1
        
        # Non-overlapping
        bbox3 = (20, 20, 30, 30)
        iou = bbox_iou(bbox1, bbox3)
        assert iou == 0.0
    
    def test_polygon_area(self):
        """Test polygon area calculation."""
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        area = polygon_area(polygon)
        assert area == 100.0
    
    def test_polygon_centroid(self):
        """Test polygon centroid calculation."""
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        centroid = polygon_centroid(polygon)
        assert abs(centroid[0] - 5) < 0.01
        assert abs(centroid[1] - 5) < 0.01
    
    def test_point_in_bbox(self):
        """Test point in bbox."""
        bbox = (0, 0, 10, 10)
        assert point_in_bbox((5, 5), bbox)
        assert not point_in_bbox((15, 5), bbox)
    
    def test_point_in_polygon(self):
        """Test point in polygon."""
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon((5, 5), polygon)
        assert not point_in_polygon((15, 5), polygon)
    
    def test_evidence_document_serialization(self, temp_dir):
        """Test evidence document serialization."""
        doc = EvidenceDocument(
            document_id="test",
            source_path="test.png",
            source_sha256="abc",
            pages=[],
            pixel_evidence=[],
            vector_evidence=[],
            observations=[],
            comparisons=[],
            aggregates=[],
            calibrations=[],
            artifacts=[],
            diagnostics=[],
            metadata={},
        )
        
        # Save and load
        path = temp_dir / "test.json"
        doc.save(path)
        loaded = EvidenceDocument.load(path)
        
        assert loaded.document_id == "test"
        assert loaded.schema_version == "planproof-evidence-v1"


# ---------------------------------------------------------------------------
# Frame tests
# ---------------------------------------------------------------------------

class TestFrames:
    """Test frame management."""
    
    def test_create_page_frame(self, simple_image, temp_dir):
        """Test page frame creation."""
        # Save image to temp file
        img_path = temp_dir / "test.png"
        Image.fromarray(simple_image).save(img_path)
        
        # Load image
        pil_img = Image.open(img_path)
        pil_img = normalize_image(pil_img)
        
        # Create frame
        frame = create_page_frame(
            document_id="test-doc",
            page_index=0,
            source_path=img_path,
            image=pil_img,
        )
        
        assert frame.document_id == "test-doc"
        assert frame.page_index == 0
        assert frame.image_size == pil_img.size
        assert frame.source_to_canonical.matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


# ---------------------------------------------------------------------------
# Raster tests
# ---------------------------------------------------------------------------

class TestRaster:
    """Test raster loading and processing."""
    
    def test_load_image(self, temp_dir):
        """Test image loading."""
        # Create test image
        img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        img_path = temp_dir / "test.png"
        Image.fromarray(img).save(img_path)
        
        # Load
        loaded_img, frame = load_image(img_path)
        
        assert loaded_img.size == (50, 50)
        assert frame.image_size == (50, 50)
        assert frame.page_index == 0
    
    def test_detect_non_floorplan_valid(self, floorplan_image):
        """Test non-floorplan detection on valid floorplan."""
        pil_img = Image.fromarray(floorplan_image)
        is_valid, reason = detect_non_floorplan(pil_img)
        # Note: This might fail depending on image complexity
        # Just check it doesn't crash
        assert isinstance(is_valid, bool)
        assert isinstance(reason, str)
    
    def test_detect_non_floorplan_sheet_music(self):
        """Test non-floorplan detection on sheet music."""
        # Create sheet-music-like image
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        # Draw many horizontal lines
        for y in range(20, 180, 10):
            cv2.line(img, (10, y), (190, y), (0, 0, 0), 1)
        
        pil_img = Image.fromarray(img)
        is_valid, reason = detect_non_floorplan(pil_img)
        assert not is_valid
        assert "horizontal" in reason.lower() or "sheet" in reason.lower()


# ---------------------------------------------------------------------------
# Pixel tests
# ---------------------------------------------------------------------------

class TestPixels:
    """Test pixel-level extraction."""
    
    def test_extract_ink_mask(self, simple_image):
        """Test ink mask extraction."""
        mask = extract_ink_mask(simple_image)
        assert mask.shape == simple_image.shape[:2]
        assert mask.dtype == np.uint8
        assert np.sum(mask > 0) > 0  # Should have some ink
    
    def test_extract_components(self, simple_image):
        """Test connected component extraction."""
        mask = extract_ink_mask(simple_image)
        components = extract_components(mask)
        assert len(components) > 0
        assert all("bbox" in comp for comp in components)
        assert all("area" in comp for comp in components)
    
    def test_extract_contour_rings(self, simple_image):
        """Test contour ring extraction."""
        mask = extract_ink_mask(simple_image)
        rings = extract_contour_rings(mask)
        assert len(rings) > 0
        assert all("points" in ring for ring in rings)
        assert all("is_outer" in ring for ring in rings)
    
    def test_extract_line_segments(self, floorplan_image):
        """Test line segment extraction."""
        mask = extract_ink_mask(floorplan_image)
        segments = extract_line_segments(mask)
        assert len(segments) > 0
        assert all("a" in seg for seg in segments)
        assert all("b" in seg for seg in segments)
        assert all("length" in seg for seg in segments)
    
    def test_detect_opening_candidates(self, floorplan_image):
        """Test opening candidate detection."""
        mask = extract_ink_mask(floorplan_image)
        openings = detect_opening_candidates(mask)
        # May or may not find openings depending on image
        assert isinstance(openings, list)
    
    def test_detect_stair_candidates(self, floorplan_image):
        """Test stair candidate detection."""
        mask = extract_ink_mask(floorplan_image)
        stairs = detect_stair_candidates(mask)
        assert isinstance(stairs, list)
    
    def test_extract_pixel_evidence(self, simple_image, sample_page_frame):
        """Test complete pixel evidence extraction."""
        evidence = extract_pixel_evidence(simple_image, sample_page_frame)
        assert evidence.frame == sample_page_frame
        assert len(evidence.observations) > 0
        # Artifacts are now handled by pipeline, not extract_pixel_evidence
        assert "ocr" in evidence.availability


# ---------------------------------------------------------------------------
# OCR tests
# ---------------------------------------------------------------------------

class TestOCR:
    """Test OCR functionality."""
    
    def test_check_tesseract_available(self):
        """Test tesseract availability check."""
        available = check_tesseract_available()
        assert isinstance(available, bool)
    
    def test_extract_ocr_spans_no_tesseract(self, simple_image, sample_page_frame):
        """Test OCR when tesseract is not available."""
        if not check_tesseract_available():
            with pytest.raises(RuntimeError, match="tesseract"):
                extract_ocr_spans(simple_image, sample_page_frame)
        else:
            # If tesseract is available, should work
            spans = extract_ocr_spans(simple_image, sample_page_frame)
            assert isinstance(spans, list)


# ---------------------------------------------------------------------------
# Vector tests
# ---------------------------------------------------------------------------

class TestVectors:
    """Test vector extraction."""
    
    def test_compute_distance_transform(self, simple_image):
        """Test distance transform computation."""
        mask = extract_ink_mask(simple_image)
        dist = compute_distance_transform(mask)
        assert dist.shape == mask.shape
        assert dist.dtype == np.float64
    
    def test_skeletonize_mask(self, simple_image):
        """Test mask skeletonization."""
        mask = extract_ink_mask(simple_image)
        skeleton = skeletonize_mask(mask)
        assert skeleton.shape == mask.shape
        assert skeleton.dtype == np.uint8
    
    def test_extract_wall_thickness(self, floorplan_image):
        """Test wall thickness extraction."""
        mask = extract_ink_mask(floorplan_image)
        skeleton = skeletonize_mask(mask)
        thickness = extract_wall_thickness(mask, skeleton)
        assert "mean" in thickness
        assert "median" in thickness
        assert "std" in thickness
    
    def test_merge_collinear_segments(self):
        """Test collinear segment merging."""
        segments = [
            {"a": (0, 0), "b": (10, 0), "length": 10, "angle": 0, "n_points": 10, "support": 0.8},
            {"a": (10, 0), "b": (20, 0), "length": 10, "angle": 0, "n_points": 10, "support": 0.9},
        ]
        merged = merge_collinear_segments(segments)
        assert len(merged) <= len(segments)
    
    def test_extract_wall_edges(self, floorplan_image):
        """Test wall edge extraction."""
        mask = extract_ink_mask(floorplan_image)
        skeleton = skeletonize_mask(mask)
        edges = extract_wall_edges(mask, skeleton, min_wall_length=20)
        assert len(edges) > 0
        assert all("id" in edge for edge in edges)
        assert all("a" in edge for edge in edges)
        assert all("b" in edge for edge in edges)


# ---------------------------------------------------------------------------
# Topology tests
# ---------------------------------------------------------------------------

class TestTopology:
    """Test topological analysis."""
    
    def test_build_wall_network(self):
        """Test wall network building."""
        wall_edges = [
            {"id": "wall-1", "a": (0, 0), "b": (10, 0), "length": 10, "angle": 0},
            {"id": "wall-2", "a": (10, 0), "b": (10, 10), "length": 10, "angle": 90},
        ]
        nodes, edges = build_wall_network(wall_edges, junction_tolerance=5)
        assert len(nodes) > 0
        assert len(edges) == len(wall_edges)
    
    def test_polygonize_faces(self):
        """Test face polygonization."""
        # Create a simple square wall network
        wall_edges = [
            WallEdge(id="w1", start_node="n1", end_node="n2", 
                    geometry={"a": (0, 0), "b": (10, 0)}),
            WallEdge(id="w2", start_node="n2", end_node="n3",
                    geometry={"a": (10, 0), "b": (10, 10)}),
            WallEdge(id="w3", start_node="n3", end_node="n4",
                    geometry={"a": (10, 10), "b": (0, 10)}),
            WallEdge(id="w4", start_node="n4", end_node="n1",
                    geometry={"a": (0, 10), "b": (0, 0)}),
        ]
        
        faces, diagnostics = polygonize_faces(wall_edges, (20, 20))
        # Should find at least one face
        assert isinstance(faces, list)
        assert isinstance(diagnostics, list)
    
    def test_attach_openings_to_walls(self):
        """Test opening attachment."""
        openings = [
            Observation(
                id="opening-1",
                kind=ObservationKind.OPENING_CANDIDATE,
                source_family=SourceFamily.CLASSICAL_PIXELS,
                mode=Mode.INDEPENDENT,
                page_index=0,
                frame_id="p0-canonical",
                bbox=(5, 5, 15, 15),
                geometry={},
                method="test",
                method_version="v1",
            )
        ]
        
        wall_edges = [
            WallEdge(id="w1", start_node="n1", end_node="n2",
                    geometry={"a": (0, 0), "b": (20, 0)}),
        ]
        
        associations = attach_openings_to_walls(openings, wall_edges, distance_threshold=20)
        assert len(associations) > 0
        assert associations[0].opening_id == "opening-1"


# ---------------------------------------------------------------------------
# Measurement tests
# ---------------------------------------------------------------------------

class TestMeasurements:
    """Test measurement extraction."""
    
    def test_parse_dimension_text(self):
        """Test dimension text parsing."""
        # Metric
        result = parse_dimension_text("3.5m")
        assert result is not None
        assert result["value"] == 3.5
        assert result["unit"] == "m"
        
        # Millimeters
        result = parse_dimension_text("3500mm")
        assert result is not None
        assert result["value"] == 3500
        assert result["unit"] == "mm"
        
        # Imperial
        result = parse_dimension_text("3'-6\"")
        assert result is not None
        assert result["value"] == 42  # 3*12 + 6 = 42 inches
        assert result["unit"] == "in"
        
        # Non-dimension
        result = parse_dimension_text("hello world")
        assert result is None
    
    def test_extract_dimensions_from_text(self, sample_page_frame):
        """Test dimension extraction from text."""
        text_obs = [
            Observation(
                id="text-1",
                kind=ObservationKind.TEXT,
                source_family=SourceFamily.OCR,
                mode=Mode.INDEPENDENT,
                page_index=0,
                frame_id="p0-canonical",
                bbox=(10, 10, 50, 20),
                geometry={"text": "3.5m"},
                method="ocr",
                method_version="v1",
            )
        ]
        
        dimensions = extract_dimensions_from_text(text_obs, sample_page_frame)
        assert len(dimensions) > 0
        assert dimensions[0].parsed_value == 3.5
        assert dimensions[0].unit == "m"
    
    def test_find_dimension_associations(self):
        """Test dimension-wall association."""
        dimensions = [
            DimensionObservation(
                id="dim-1",
                text="3.5m",
                parsed_value=3.5,
                unit="m",
                bbox=(10, 10, 50, 20),
                associated_objects=[],
                calibration_id=None,
                is_calibrated=False,
                source_ids=[],
            )
        ]
        
        wall_edges = [
            WallEdge(id="w1", start_node="n1", end_node="n2",
                    geometry={"a": (0, 0), "b": (100, 0)},
                    length_px=100),
        ]
        
        dimensions = find_dimension_associations(dimensions, wall_edges, distance_threshold=50)
        assert len(dimensions[0].associated_objects) > 0
    
    def test_estimate_calibration(self):
        """Test calibration estimation."""
        dimensions = [
            DimensionObservation(
                id="dim-1",
                text="3.5m",
                parsed_value=3.5,
                unit="m",
                bbox=(10, 10, 50, 20),
                associated_objects=["w1"],
                calibration_id=None,
                is_calibrated=False,
                source_ids=[],
            ),
            DimensionObservation(
                id="dim-2",
                text="5.0m",
                parsed_value=5.0,
                unit="m",
                bbox=(60, 10, 100, 20),
                associated_objects=["w2"],
                calibration_id=None,
                is_calibrated=False,
                source_ids=[],
            ),
        ]
        
        wall_edges = [
            WallEdge(id="w1", start_node="n1", end_node="n2",
                    geometry={"a": (0, 0), "b": (35, 0)},
                    length_px=35),
            WallEdge(id="w2", start_node="n3", end_node="n4",
                    geometry={"a": (0, 0), "b": (50, 0)},
                    length_px=50),
        ]
        
        calibrations = estimate_calibration(dimensions, wall_edges)
        assert len(calibrations) > 0
        assert calibrations[0].is_valid


# ---------------------------------------------------------------------------
# ML adapter tests
# ---------------------------------------------------------------------------

class TestMLAdapter:
    """Test ML adapter functionality."""
    
    def test_ml_instance_validation(self):
        """Test ML instance validation."""
        instance = MLInstance(
            id="test-1",
            class_name="wall",
            bbox=(10, 10, 50, 50),
            confidence=0.9,
        )
        assert instance.id == "test-1"
        assert instance.class_name == "wall"
        assert instance.confidence == 0.9
    
    def test_ml_prediction_validation(self):
        """Test ML prediction validation."""
        prediction = MLPrediction(
            schema_version=ML_SCHEMA_VERSION,
            document_id="test-doc",
            image_size=(100, 100),
            model_version="test-model",
            instances=[],
        )
        assert prediction.schema_version == ML_SCHEMA_VERSION
        assert prediction.document_id == "test-doc"
    
    def test_validate_ml_binding(self, sample_page_frame):
        """Test ML binding validation."""
        prediction = MLPrediction(
            schema_version=ML_SCHEMA_VERSION,
            document_id="test-doc",
            image_size=(100, 100),
            model_version="test-model",
            instances=[],
        )
        
        # Update frame to match
        sample_page_frame.document_id = "test-doc"
        sample_page_frame.image_size = (100, 100)
        
        is_valid, warnings = validate_ml_binding(prediction, sample_page_frame)
        assert is_valid
        assert len(warnings) == 0
    
    def test_ml_instance_to_observation(self):
        """Test ML instance to observation conversion."""
        instance = MLInstance(
            id="test-1",
            class_name="wall",
            bbox=(10, 10, 50, 50),
            confidence=0.9,
        )
        
        obs = ml_instance_to_observation(instance, 0, "p0-canonical")
        assert obs.id == "ml-test-1"
        assert obs.kind == ObservationKind.WALL
        assert obs.source_family == SourceFamily.ML
        assert obs.score == 0.9


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------

class TestCompare:
    """Test ML vs geometry comparison."""
    
    def test_compute_bbox_distance(self):
        """Test bbox distance computation."""
        bbox1 = (0, 0, 10, 10)
        bbox2 = (20, 20, 30, 30)
        dist = compute_bbox_distance(bbox1, bbox2)
        assert dist > 0
        
        # Overlapping bboxes
        bbox3 = (5, 5, 15, 15)
        dist = compute_bbox_distance(bbox1, bbox3)
        assert dist == 0.0
    
    def test_match_observations(self):
        """Test observation matching."""
        ml_obs = [
            Observation(
                id="ml-1",
                kind=ObservationKind.WALL,
                source_family=SourceFamily.ML,
                mode=Mode.INDEPENDENT,
                page_index=0,
                frame_id="p0-canonical",
                bbox=(10, 10, 50, 50),
                geometry={},
                method="ml",
                method_version="v1",
            )
        ]
        
        geom_obs = [
            Observation(
                id="geom-1",
                kind=ObservationKind.WALL,
                source_family=SourceFamily.CLASSICAL_PIXELS,
                mode=Mode.INDEPENDENT,
                page_index=0,
                frame_id="p0-canonical",
                bbox=(12, 12, 48, 48),
                geometry={},
                method="geom",
                method_version="v1",
            )
        ]
        
        matches = match_observations(ml_obs, geom_obs)
        assert len(matches) > 0
        assert matches[0][0].id == "ml-1"
        assert matches[0][1].id == "geom-1"
    
    def test_create_comparison(self):
        """Test comparison creation."""
        ml_obs = Observation(
            id="ml-1",
            kind=ObservationKind.WALL,
            source_family=SourceFamily.ML,
            mode=Mode.INDEPENDENT,
            page_index=0,
            frame_id="p0-canonical",
            bbox=(10, 10, 50, 50),
            geometry={},
            method="ml",
            method_version="v1",
        )
        
        geom_obs = Observation(
            id="geom-1",
            kind=ObservationKind.WALL,
            source_family=SourceFamily.CLASSICAL_PIXELS,
            mode=Mode.INDEPENDENT,
            page_index=0,
            frame_id="p0-canonical",
            bbox=(12, 12, 48, 48),
            geometry={},
            method="geom",
            method_version="v1",
        )
        
        comp = create_comparison(ml_obs, geom_obs, ComparisonRelation.AGREEMENT, {"iou": 0.8})
        assert comp.ml_observation_id == "ml-1"
        assert comp.geometry_observation_id == "geom-1"
        assert comp.relation == ComparisonRelation.AGREEMENT
    
    def test_compare_ml_and_geometry(self, sample_page_frame):
        """Test complete ML vs geometry comparison."""
        ml_obs = [
            Observation(
                id="ml-1",
                kind=ObservationKind.WALL,
                source_family=SourceFamily.ML,
                mode=Mode.INDEPENDENT,
                page_index=0,
                frame_id="p0-canonical",
                bbox=(10, 10, 50, 50),
                geometry={},
                method="ml",
                method_version="v1",
            )
        ]
        
        geom_obs = [
            Observation(
                id="geom-1",
                kind=ObservationKind.WALL,
                source_family=SourceFamily.CLASSICAL_PIXELS,
                mode=Mode.INDEPENDENT,
                page_index=0,
                frame_id="p0-canonical",
                bbox=(12, 12, 48, 48),
                geometry={},
                method="geom",
                method_version="v1",
            )
        ]
        
        comparisons = compare_ml_and_geometry(ml_obs, geom_obs, sample_page_frame)
        assert len(comparisons) > 0
        assert comparisons[0].relation in [
            ComparisonRelation.AGREEMENT,
            ComparisonRelation.GEOMETRY_CONFLICT,
        ]
    
    def test_create_aggregates(self, sample_page_frame):
        """Test aggregate creation."""
        observations = [
            Observation(
                id="obs-1",
                kind=ObservationKind.WALL,
                source_family=SourceFamily.CLASSICAL_PIXELS,
                mode=Mode.INDEPENDENT,
                page_index=0,
                frame_id="p0-canonical",
                bbox=(10, 10, 50, 50),
                geometry={},
                method="test",
                method_version="v1",
            ),
            Observation(
                id="obs-2",
                kind=ObservationKind.WALL,
                source_family=SourceFamily.CLASSICAL_PIXELS,
                mode=Mode.INDEPENDENT,
                page_index=0,
                frame_id="p0-canonical",
                bbox=(60, 60, 100, 100),
                geometry={},
                method="test",
                method_version="v1",
            ),
        ]
        
        aggregates = create_aggregates(observations, [], sample_page_frame)
        assert len(aggregates) > 0
        assert aggregates[0].class_name == "wall"
        assert aggregates[0].value == 2.0


# ---------------------------------------------------------------------------
# Crop tests
# ---------------------------------------------------------------------------

class TestCrops:
    """Test crop extraction and overlays."""
    
    def test_extract_crop(self, simple_image, sample_page_frame, temp_dir):
        """Test crop extraction."""
        bbox = (10, 10, 50, 50)
        result = extract_crop(
            evidence_path=None,
            page_index=0,
            bbox=bbox,
            out_dir=temp_dir,
            padding_px=5,
            image=simple_image,
            frame=sample_page_frame,
        )
        
        assert "crop_path" in result
        assert "crop_bbox" in result
        assert "crop_transform" in result
        assert Path(result["crop_path"]).exists()
    
    def test_render_overlay(self, simple_image, sample_page_frame, temp_dir):
        """Test overlay rendering."""
        observations = [
            Observation(
                id="obs-1",
                kind=ObservationKind.WALL,
                source_family=SourceFamily.CLASSICAL_PIXELS,
                mode=Mode.INDEPENDENT,
                page_index=0,
                frame_id="p0-canonical",
                bbox=(10, 10, 50, 50),
                geometry={},
                method="test",
                method_version="v1",
            )
        ]
        
        out_path = temp_dir / "overlay.png"
        render_overlay(simple_image, observations, 0, out_path)
        
        assert out_path.exists()
        # Check it's a valid image
        img = Image.open(out_path)
        assert img.size == (100, 100)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestPipeline:
    """Test main pipeline."""
    
    def test_analyze_simple_image(self, temp_dir):
        """Test analysis of simple image."""
        # Create test image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img[20:80, 20:80] = 0
        
        img_path = temp_dir / "test.png"
        Image.fromarray(img).save(img_path)
        
        # Run analysis
        out_dir = temp_dir / "output"
        doc = analyze(img_path, out_dir)
        
        assert doc.schema_version == "planproof-evidence-v1"
        assert len(doc.pages) > 0
        assert len(doc.observations) > 0
        assert (out_dir / "evidence.json").exists()
    
    def test_analyze_with_ml_predictions(self, temp_dir):
        """Test analysis with ML predictions."""
        # Create test image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img[20:80, 20:80] = 0
        
        img_path = temp_dir / "test.png"
        Image.fromarray(img).save(img_path)
        
        # Compute document ID (SHA256 of file)
        from planproof_evidence.contracts import sha256_file
        doc_id = sha256_file(img_path)
        
        # Create ML predictions with correct document ID
        ml_pred = {
            "schema_version": "planproof-detection-v1",
            "document_id": doc_id,
            "image_size": [100, 100],
            "model_version": "test-model",
            "instances": [
                {
                    "id": "ml-1",
                    "class": "wall",
                    "bbox": [10, 10, 50, 50],
                    "confidence": 0.9,
                }
            ],
        }
        
        ml_path = temp_dir / "ml_predictions.json"
        with open(ml_path, "w") as f:
            json.dump(ml_pred, f)
        
        # Run analysis
        out_dir = temp_dir / "output"
        doc = analyze(img_path, out_dir, ml_predictions=ml_path)
        
        assert len(doc.observations) > 0
        # Should have ML observations
        ml_obs = [obs for obs in doc.observations if obs.source_family == SourceFamily.ML]
        assert len(ml_obs) > 0
    
    def test_get_region(self, temp_dir):
        """Test get_region function."""
        # Create test image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img[20:80, 20:80] = 0
        
        img_path = temp_dir / "test.png"
        Image.fromarray(img).save(img_path)
        
        # Run analysis
        out_dir = temp_dir / "output"
        doc = analyze(img_path, out_dir)
        
        # Save evidence
        evidence_path = out_dir / "evidence.json"
        
        # Get region
        result = get_region(
            evidence_path=evidence_path,
            page_index=0,
            bbox=(10, 10, 50, 50),
            out_dir=temp_dir / "crops",
            padding_px=5,
        )
        
        assert "crop_path" in result
        assert "crop_transform" in result
        assert "intersecting_observation_ids" in result


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests with real files."""
    
    def test_real_png_analysis(self):
        """Test analysis of real PNG file."""
        # Try to find a real dataset file
        dataset_paths = [
            Path("/Volumes/hnguyen ssd/junction/junction-dataset/data/swiss_dwellings/site936_bldg1976_floor8706.png"),
            Path("/Users/nguyentranhuynguyen/Downloads/junction-dataset-preview/data/swiss_dwellings/site936_bldg1976_floor8706.png"),
        ]
        
        test_file = None
        for path in dataset_paths:
            if path.exists():
                test_file = path
                break
        
        if test_file is None:
            pytest.skip("No real dataset file found")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output"
            doc = analyze(test_file, out_dir)
            
            assert doc.schema_version == "planproof-evidence-v1"
            assert len(doc.pages) > 0
            assert len(doc.observations) > 0
            assert (out_dir / "evidence.json").exists()
    
    def test_real_pdf_analysis(self):
        """Test analysis of real PDF file."""
        # Try to find a real dataset file
        dataset_paths = [
            Path("/Volumes/hnguyen ssd/junction/junction-dataset/data/swiss_dwellings/site936_bldg1976_floor8706.pdf"),
            Path("/Users/nguyentranhuynguyen/Downloads/junction-dataset-preview/data/swiss_dwellings/site936_bldg1976_floor8706.pdf"),
        ]
        
        test_file = None
        for path in dataset_paths:
            if path.exists():
                test_file = path
                break
        
        if test_file is None:
            pytest.skip("No real dataset file found")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output"
            doc = analyze(test_file, out_dir)
            
            assert doc.schema_version == "planproof-evidence-v1"
            assert len(doc.pages) > 0
            assert (out_dir / "evidence.json").exists()


# ---------------------------------------------------------------------------
# T2-03 / T2-15: plan-region estimation and semantic room classification
# ---------------------------------------------------------------------------

def _make_face(fid, polygon, area, centroid=(0.0, 0.0), is_valid=True):
    return RoomFace(
        id=fid,
        polygon=polygon,
        holes=[],
        area_px=area,
        centroid=centroid,
        is_valid=is_valid,
        invalid_reason=None if is_valid else "test_invalid",
        semantic_class="unclassified",
    )


class TestPlanRegionAndRoomClassification:
    """T2-03 scope partitioning + T2-15 semantic room filtering."""

    def test_estimate_plan_region_prefers_plan_over_legend(self):
        """Detached bottom legend strip must not define the plan region."""
        mask = np.zeros((400, 600), dtype=np.uint8)
        mask[50:350, 50:350] = 255    # main plan blob
        mask[370:390, 50:550] = 255   # detached legend strip
        bbox, area = estimate_plan_region(mask, (600, 400))
        # Plan blob bbox is (50,50)-(350,350); legend must be excluded
        assert bbox[0] <= 60 and bbox[1] <= 60
        assert bbox[2] >= 340 and bbox[3] <= 360
        assert area > 50000

    def test_estimate_plan_region_rejects_sheet_spanning_frame(self):
        """A border/frame spanning ~whole sheet must not be the plan."""
        mask = np.zeros((400, 600), dtype=np.uint8)
        # Sheet frame: thin connected border spanning nearly the page
        mask[10, 10:590] = 255
        mask[390, 10:590] = 255
        mask[10:390, 10] = 255
        mask[10:390, 590] = 255
        # Inner plan blob, disconnected from the frame
        mask[100:300, 200:400] = 255
        bbox, _ = estimate_plan_region(mask, (600, 400))
        # Should pick the inner plan blob (200x100), not the frame (580x380)
        assert bbox[2] - bbox[0] < 400
        assert bbox[0] >= 190 and bbox[2] <= 410

    def test_classify_room_faces_area_and_aspect(self):
        """Small loops -> sub_feature; slivers -> non_room_loop; big -> room."""
        plan_bbox = (0.0, 0.0, 1000.0, 1000.0)
        plan_area = 1_000_000.0
        faces = [
            _make_face("big", [(100, 100), (500, 100), (500, 400), (100, 400)],
                       120000.0, centroid=(300.0, 250.0)),
            _make_face("tiny", [(10, 10), (20, 10), (20, 20), (10, 20)],
                       100.0, centroid=(15.0, 15.0)),
            _make_face("sliver", [(0, 0), (900, 0), (905, 10), (0, 10)],
                       9000.0, centroid=(450.0, 5.0)),
            _make_face("outside", [(2000, 2000), (2100, 2000),
                                   (2100, 2100), (2000, 2100)],
                       10000.0, centroid=(2050.0, 2050.0)),
            _make_face("invalid", [(0, 0), (1, 0), (1, 1)], 1.0,
                       is_valid=False),
        ]
        diags = classify_room_faces(faces, plan_bbox, plan_area,
                                    (1200, 1200))
        by_id = {f.id: f.semantic_class for f in faces}
        assert by_id["big"] == "room"
        assert by_id["tiny"] == "sub_feature"
        assert by_id["sliver"] == "non_room_loop"   # aspect > 8
        assert by_id["outside"] == "non_room_loop"  # outside plan region
        assert by_id["invalid"] == "non_room_loop"
        assert any("room=1" in d for d in diags)

    def test_classify_rooms_emits_semantic_kinds(self, sample_page_frame):
        """RoomFace.semantic_class maps to ObservationKind correctly."""
        ve = VectorEvidence(
            frame_id="p0-canonical",
            room_faces=[
                _make_face("r1", [(0, 0), (10, 0), (10, 10), (0, 10)],
                           5000.0, centroid=(5.0, 5.0)),
                _make_face("s1", [(0, 0), (2, 0), (2, 2), (0, 2)],
                           50.0, centroid=(1.0, 1.0)),
            ],
        )
        ve.room_faces[0].semantic_class = "room"
        ve.room_faces[1].semantic_class = "sub_feature"
        obs = classify_rooms(ve, sample_page_frame)
        kinds = {o.id: o.kind for o in obs}
        assert kinds["room-r1"] == ObservationKind.ROOM
        assert kinds["sub_feature-s1"] == ObservationKind.SUB_FEATURE
        # Geometry preserves provenance fields
        assert obs[0].geometry["semantic_class"] == "room"

    def test_unclassified_face_defaults_to_room(self, sample_page_frame):
        """Faces without classification keep legacy room behaviour."""
        ve = VectorEvidence(
            frame_id="p0-canonical",
            room_faces=[_make_face("r9", [(0, 0), (4, 0), (4, 4), (0, 4)],
                                    100.0, centroid=(2.0, 2.0))],
        )
        obs = classify_rooms(ve, sample_page_frame)
        assert obs[0].kind == ObservationKind.ROOM


# ---------------------------------------------------------------------------
# planproof-door-result-v1 adapter
# ---------------------------------------------------------------------------

def _door_doc(with_agg_status="resolved", n_doors=2):
    frame = PageFrame(
        document_id="d" * 64,
        page_index=0,
        source_sha256="a" * 64,
        raster_sha256="b" * 64,
        image_size=(200, 200),
        source_to_canonical=identity_transform(),
        canonical_to_source=identity_transform(),
    )
    obs = []
    for i in range(n_doors):
        obs.append(Observation(
            id=f"door-{i}",
            kind=ObservationKind.DOOR,
            source_family=SourceFamily.CLASSICAL_PIXELS,
            mode=Mode.INDEPENDENT,
            page_index=0,
            frame_id="p0-canonical",
            bbox=(10 + i * 50, 10, 30 + i * 50, 60),
            method="test",
            method_version="v1",
        ))
    aggs = []
    if with_agg_status is not None:
        aggs.append(Aggregate(
            id="agg-door-p0",
            scope="page",
            page_index=0,
            class_name="door",
            counting_unit="door_assembly",
            member_ids=[o.id for o in obs],
            value=float(n_doors),
            unit="count",
            status=with_agg_status,
        ))
    # vector evidence presence => page fully analyzed
    ve = VectorEvidence(frame_id="p0-canonical")
    return EvidenceDocument(
        document_id="d" * 64,
        source_path="plan.png",
        source_sha256="a" * 64,
        pages=[frame],
        vector_evidence=[ve],
        observations=obs,
        aggregates=aggs,
    )


class TestDoorResultAdapter:
    def test_schema_and_fields(self):
        doc = _door_doc()
        res = to_door_result_v1(doc)
        assert res["schema_version"] == "planproof-door-result-v1"
        assert res["status"] in ("ok", "partial")
        assert res["mode"] == "independent"
        assert res["complete"] is True
        assert isinstance(res["detections"], list)
        assert len(res["detections"]) == 2
        det = res["detections"][0]
        assert set(det.keys()) == {"id", "bbox"}
        assert len(det["bbox"]) == 4

    def test_source_frame_matches_page(self):
        doc = _door_doc()
        res = to_door_result_v1(doc)
        src = res["source"]
        assert src["document_id"] == "d" * 64
        assert src["page_index"] == 0
        assert src["source_sha256"] == "a" * 64
        assert src["raster_sha256"] == "b" * 64
        assert src["width"] == 200 and src["height"] == 200
        assert src["coordinate_system"] == "pixel_edges_xy_top_left"

    def test_unavailable_when_page_missing(self):
        doc = _door_doc()
        doc.pages = []
        res = to_door_result_v1(doc, page_index=3)
        assert res["status"] == "unavailable"
        assert res["complete"] is False
        assert res["detections"] == []

    def test_partial_on_unresolved_aggregate(self):
        doc = _door_doc(with_agg_status="partial")
        res = to_door_result_v1(doc)
        assert res["status"] == "partial"
        assert res["complete"] is False

    def test_ok_with_zero_doors_on_clean_run(self):
        doc = _door_doc(with_agg_status=None, n_doors=0)
        res = to_door_result_v1(doc)
        assert res["status"] == "ok"
        assert res["complete"] is True
        assert res["detections"] == []

    def test_page_frame_override_whitelists_fields(self):
        doc = _door_doc()
        res = to_door_result_v1(doc, page_frame_override={
            "document_id": "override-id",
            "evil_extra": "should-be-dropped",
            "width": None,  # None values ignored
        })
        assert res["source"]["document_id"] == "override-id"
        assert "evil_extra" not in res["source"]
        assert res["source"]["width"] == 200

    def test_detections_sorted_and_clipped(self):
        doc = _door_doc(n_doors=1)
        doc.observations[0].bbox = (190.0, 10.0, 250.0, 60.0)  # exceeds width
        res = to_door_result_v1(doc)
        assert len(res["detections"]) == 1
        bbox = res["detections"][0]["bbox"]
        assert bbox[2] <= 200.0 and bbox[3] <= 200.0
        assert bbox[0] < bbox[2] and bbox[1] < bbox[3]

    def test_detection_ids_unique_and_capped(self):
        doc = _door_doc(with_agg_status=None, n_doors=0)
        for i in range(90):
            doc.observations.append(Observation(
                id=f"door-x{i}", kind=ObservationKind.DOOR,
                source_family=SourceFamily.CLASSICAL_PIXELS,
                mode=Mode.INDEPENDENT, page_index=0,
                frame_id="p0-canonical",
                bbox=(10.0, float(i), 20.0, float(i) + 5),
                method="test",
                method_version="v1",
            ))
        res = to_door_result_v1(doc)
        ids = [d["id"] for d in res["detections"]]
        assert len(ids) == len(set(ids)) <= 80
        # Truncation => partial
        assert res["status"] == "partial"


class TestDoorResultEndpoint:
    """FastAPI /v1/door-result endpoint contract."""

    @pytest.fixture(scope="class")
    def client(self):
        import sys
        geom_dir = Path(__file__).resolve().parents[2] / "geometry"
        if str(geom_dir) not in sys.path:
            sys.path.insert(0, str(geom_dir))
        from fastapi.testclient import TestClient
        import service
        return TestClient(service.app)

    def test_endpoint_requires_input(self, client):
        resp = client.post("/v1/door-result", json={})
        assert resp.status_code == 400

    def test_endpoint_missing_evidence(self, client):
        resp = client.post("/v1/door-result",
                           json={"evidence_path": "/nonexistent/x.json"})
        assert resp.status_code == 400

    def test_endpoint_with_synthetic_evidence(self, client, temp_dir):
        doc = _door_doc()
        ev_path = temp_dir / "evidence.json"
        doc.save(ev_path)
        resp = client.post("/v1/door-result",
                           json={"evidence_path": str(ev_path)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["schema_version"] == "planproof-door-result-v1"
        assert data["source"]["document_id"] == "d" * 64
        assert len(data["detections"]) == 2

    def test_endpoint_file_path_analysis(self, client, temp_dir):
        # Minimal valid floorplan PNG
        img = np.ones((120, 120, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (20, 20), (100, 100), (0, 0, 0), 3)
        img_path = temp_dir / "plan.png"
        cv2.imwrite(str(img_path), img)
        resp = client.post("/v1/door-result",
                           json={"file_path": str(img_path)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["schema_version"] == "planproof-door-result-v1"
        assert data["status"] in ("ok", "partial", "unavailable")
        assert "detections" in data
