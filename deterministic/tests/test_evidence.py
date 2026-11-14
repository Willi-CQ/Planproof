from __future__ import annotations

import copy
import json
from dataclasses import replace

import cv2
import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon, box

from planproof_evidence import analyze, get_region
from planproof_evidence.compare import assign, duplicate_groups, aggregates
from planproof_evidence.contracts import (
    Config,
    rgb_hash,
    sha256,
    transform_points,
    validate_document,
    write_json,
)
from planproof_evidence.geometry import (
    mask_polygons,
    merge_collinear,
    net_regions,
    node_graph,
    pair_boundaries,
    snap_polygon,
)
from planproof_evidence.inputs import load_pages
from planproof_evidence.measurements import (
    calibrate,
    classify_room_text,
    parse_dimension,
    quantities,
)
from planproof_evidence.pixels import (
    detect_arcs,
    extract_pixels,
    opening_strip_from_arc,
    detect_window_bands,
)


@pytest.fixture
def plan(tmp_path):
    image = np.full((200, 260, 3), 255, np.uint8)
    cv2.rectangle(image, (25, 25), (235, 175), (0, 0, 0), 8)
    cv2.line(image, (130, 25), (130, 175), (0, 0, 0), 8)
    path = tmp_path / "plan.png"
    Image.fromarray(image).save(path)
    return path, image


def test_mutual_pair_translation_invariant():
    segments = [[[5, 10], [105, 10]], [[5, 16], [105, 16]]]
    first, _ = pair_boundaries(segments, 2)
    translated = (np.asarray(segments) + [2000, -713]).tolist()
    second, _ = pair_boundaries(translated, 2)
    assert len(first) == len(second) == 1
    assert first[0]["thickness_px"] == pytest.approx(second[0]["thickness_px"])
    assert first[0]["polygon"].area == pytest.approx(second[0]["polygon"].area)


def test_no_merge_across_unobserved_gap():
    segments = [[[10, 20], [50, 20]], [[53, 20], [90, 20]]]
    ink = np.zeros((100, 100), np.uint8)
    # A large unsupported area cannot be declared a continuous observed wall.
    assert len(merge_collinear(segments, ink, 2)) == 2
    ink[20, 10:91] = 1
    assert len(merge_collinear(segments, ink, 2)) == 1


def test_noding_crossing_and_dangle():
    graph = node_graph([[[0, 5], [10, 5]], [[5, 0], [5, 10]]])
    assert len(graph["nodes"]) == 5
    assert len(graph["edges"]) == 4
    assert len(graph["regions"]) == 0
    assert graph["diagnostics"]["dangles"] == 4


def test_holes_and_disconnected_components():
    mask = np.zeros((100, 100), np.uint8)
    mask[5:60, 5:60] = 1
    mask[20:40, 20:40] = 0
    mask[70:90, 70:90] = 1
    polygons = mask_polygons(mask, epsilon=0.1)
    assert len(polygons) == 2
    assert len(polygons[0].interiors) == 1
    result = net_regions([box(0, 0, 100, 100)], [box(20, 20, 30, 30)])
    assert len(result[0].interiors) == 1
    assert result[0].area == 9900


def test_snap_guard_preserves_holes_and_bounds():
    polygon = Polygon(
        [(5, 5), (80, 5), (80, 80), (5, 80)], [[(20, 20), (30, 20), (30, 30), (20, 30)]]
    )
    ink = np.zeros((100, 100), np.uint8)
    cv2.rectangle(ink, (6, 6), (81, 81), 1, 1)
    snapped, _ = snap_polygon(polygon, ink, 2)
    assert snapped.is_valid
    assert len(snapped.interiors) == 1
    assert snapped.hausdorff_distance(polygon) <= 2.71


@pytest.mark.parametrize("text", ["2400", "2.40", "4", "2,40"])
def test_unitless_does_not_become_metres(text):
    assert parse_dimension(text)["status"] == "ambiguous_unit"


def test_explicit_units_decimal_comma_and_squared_conversion():
    assert parse_dimension("2400 mm")["si_value"] == 2.4
    assert parse_dimension("2,40 m")["si_value"] == 2.4
    assert parse_dimension("100 cm²")["si_value"] == pytest.approx(0.01)
    assert parse_dimension("2,400 mm")["status"] == "ambiguous_number"
    assert quantities(box(0, 0, 10, 20))["area_m2"] is None
    assert quantities(box(0, 0, 10, 20), {"status": "supported", "m_per_px": 0.1})[
        "area_m2"
    ] == pytest.approx(2)


def test_calibration_distinct_verified_consistent_anchors_only():
    a = {"id": "a", "text": "2 m", "endpoints": [[0, 0], [200, 0]], "verified_endpoints": True}
    b = {"id": "b", "text": "1 m", "endpoints": [[0, 20], [100, 20]], "verified_endpoints": True}
    assert calibrate([a])["status"] == "unresolved"
    assert calibrate([a, {**a, "id": "a-copy"}])["status"] == "unresolved"
    assert calibrate([a, b])["m_per_px"] == 0.01
    assert calibrate([a, {**b, "text": "2 m"}])["status"] == "conflict"
    assert calibrate([a, {**b, "scope": "detail"}])["status"] == "unresolved"
    assert calibrate([{**a, "verified_endpoints": False}, b])["status"] == "unresolved"


def test_room_category_requires_contained_text():
    room = box(0, 0, 100, 100)
    assert classify_room_text(room, [])["category"] == "unknown"
    assert (
        classify_room_text(room, [{"id": "x", "text": "Kitchen", "bbox": [5, 5, 50, 20]}])[
            "category"
        ]
        == "kitchen"
    )
    assert (
        classify_room_text(room, [{"id": "x", "text": "Kitchen", "bbox": [105, 5, 150, 20]}])[
            "category"
        ]
        == "unknown"
    )
    assert (
        classify_room_text(room, [{"id": "x", "text": "kitchen bedroom", "bbox": [5, 5, 50, 20]}])[
            "status"
        ]
        == "conflicting_text"
    )


def test_gated_matching_and_no_transitive_duplicate_chain():
    assert len(assign([[0.51, 0.49], [0.99, 0.51]], 0.5)) == 2
    assert assign(np.empty((0, 3))) == []
    items = [{"id": str(i), "class": "door", "bbox": [i * 2, 0, 10 + i * 2, 10]} for i in range(3)]
    groups = duplicate_groups(items, threshold=0.6)
    assert groups == [["0", "1"]]


def test_full_circle_is_not_a_door():
    gray = np.full((250, 250), 255, np.uint8)
    cv2.circle(gray, (125, 125), 25, 0, 2)
    assert (
        detect_arcs(
            gray, (gray < 128).astype(np.uint8), 2, replace(Config(), max_arc_radius_fraction=0.2)
        )
        == []
    )


def test_door_swing_is_projected_to_opening_band():
    ink = np.zeros((200, 200), np.uint8)
    ink[46, 10:50] = 1
    ink[54, 10:50] = 1
    polygon, features = opening_strip_from_arc(ink, [50, 50], 60, 0, 2)
    assert polygon.bounds == (50.0, 46.0, 110.0, 54.0)
    assert features["band_status"] == "flanking_ink_band"
    assert features["clear_width_px"] is None


def test_object_only_model_does_not_report_zero_rooms():
    result = aggregates([], 0, "cad", available_classes=["door", "window", "stairs"])
    room = next(r for r in result if r["class"] == "room")
    assert room["status"] == "unavailable" and room["value"] is None


def test_window_band_handles_unequal_pane_spacing():
    ink = np.zeros((200, 240), np.uint8)
    segments = [[[30, y], [130, y]] for y in (40, 44, 50)]
    for a, b in segments:
        cv2.line(ink, a, b, 1, 1)
    cv2.line(ink, (30, 40), (30, 50), 1, 1)
    cv2.line(ink, (130, 40), (130, 50), 1, 1)
    result = detect_window_bands(segments, ink, 2, Config())
    assert len(result) == 1
    assert result[0]["geometry"].bounds == (30.0, 40.0, 130.0, 50.0)


def test_uniform_many_treads_are_not_window_panes():
    ink = np.zeros((200, 240), np.uint8)
    segments = [[[30, y], [130, y]] for y in range(40, 66, 5)]
    for a, b in segments:
        cv2.line(ink, a, b, 1, 1)
    cv2.line(ink, (30, 40), (30, 65), 1, 1)
    cv2.line(ink, (130, 40), (130, 65), 1, 1)
    assert detect_window_bands(segments, ink, 2, Config()) == []


def test_blank_page_abstains_and_preserves_original_frame():
    data = extract_pixels(np.full((160, 220, 3), 255, np.uint8), Config())
    assert data["candidates"] == []
    assert data["availability"]["ocr"] == "unavailable"


def test_png_analysis_crops_cache_and_no_metric_invention(plan, tmp_path):
    path, image = plan
    out = tmp_path / "result"
    document = analyze(path, out)
    assert len(document["pages"]) == 1
    assert document["pages"][0]["frame"]["image_size"] == [260, 200]
    assert document["calibrations"][0]["status"] == "unresolved"
    assert any(o.get("class") == "wall" for o in document["observations"])
    assert any(o.get("class") == "room" for o in document["observations"])
    crop = get_region(out / "evidence.json", 0, [-5, -5, 100, 80], tmp_path / "crop", padding_px=4)
    assert crop["bbox"] == [0, 0, 104, 84]
    with Image.open(tmp_path / "crop/clean.png") as clean:
        assert np.array_equal(np.asarray(clean), image[:84, :104])
    context = json.loads((tmp_path / "crop/agent-context.json").read_text())
    assert context["scope"]["bbox"] == crop["bbox"]
    assert all(
        "polygons" not in o and "original_prediction" not in o for o in context["observations"]
    )
    second = analyze(path, out)
    assert second["pages"][0]["timing"]["pixel_cache_hit"]
    assert [o["id"] for o in second["observations"]] == [o["id"] for o in document["observations"]]
    different = analyze(path, out, config={"angle_deg": 4})
    assert not different["pages"][0]["timing"]["pixel_cache_hit"]


def make_binding(tmp_path, image):
    prediction = {
        "schema_version": "planproof-detection-v1",
        "document_id": "upstream-id",
        "page_index": 0,
        "image_size": [260, 200],
        "coordinate_system": "pixel_edges_xy_top_left",
        "model_version": "fixture",
        "instances": [{"id": "m1", "class": "door", "bbox": [40, 40, 70, 70], "confidence": 0.7}],
    }
    write_json(tmp_path / "prediction.json", prediction)
    source = {
        "name": "fixture",
        "prediction_path": "prediction.json",
        "prediction_sha256": sha256(tmp_path / "prediction.json"),
        "model_sha256": "a" * 64,
        "profile_sha256": "b" * 64,
        "raster_sha256": rgb_hash(image),
        "image_size": [260, 200],
        "page_index": 0,
        "document_id": "upstream-id",
    }
    binding = {"sources": [source]}
    write_json(tmp_path / "binding.json", binding)
    return binding


def test_ml_binding_rejects_same_size_wrong_image_and_modified_prediction(plan, tmp_path):
    path, image = plan
    binding = make_binding(tmp_path, image)
    binding["sources"][0]["raster_sha256"] = "0" * 64
    write_json(tmp_path / "binding.json", binding)
    with pytest.raises(ValueError, match="raster identity"):
        analyze(path, tmp_path / "bad", ml_predictions=tmp_path / "binding.json")
    binding["sources"][0]["raster_sha256"] = rgb_hash(image)
    write_json(tmp_path / "binding.json", binding)
    (tmp_path / "prediction.json").write_text("{}")
    with pytest.raises(ValueError, match="prediction hash"):
        analyze(path, tmp_path / "bad", ml_predictions=tmp_path / "binding.json")


def test_ml_hybrid_lineage_and_cache_separation(plan, tmp_path):
    path, image = plan
    make_binding(tmp_path, image)
    out = tmp_path / "bound"
    independent = analyze(path, out)
    document = analyze(path, out, ml_predictions=tmp_path / "binding.json")
    assert document["pages"][0]["timing"]["pixel_cache_hit"]
    assert document["comparisons"]
    hybrid = [o for o in document["observations"] if o["source_family"] == "hybrid"]
    assert len(hybrid) == 1 and hybrid[0]["mode"] == "hybrid"
    assert {o["id"] for o in independent["observations"]} <= {
        o["id"] for o in document["observations"]
    }
    invalid = copy.deepcopy(document)
    next(o for o in invalid["observations"] if o["source_family"] == "hybrid")["mode"] = (
        "independent"
    )
    with pytest.raises(ValueError, match="depends on"):
        validate_document(invalid)


def test_import_repairs_self_touching_ml_contour_and_retains_raw(plan, tmp_path):
    path, image = plan
    binding = make_binding(tmp_path, image)
    prediction_path = tmp_path / "prediction.json"
    prediction = json.loads(prediction_path.read_text())
    prediction["instances"][0]["polygons"] = [
        {"outer": [[40, 40], [70, 70], [40, 70], [70, 40], [40, 40]], "holes": []}
    ]
    write_json(prediction_path, prediction)
    binding["sources"][0]["prediction_sha256"] = sha256(prediction_path)
    write_json(tmp_path / "binding.json", binding)
    document = analyze(path, tmp_path / "repair", ml_predictions=tmp_path / "binding.json")
    imported = next(o for o in document["observations"] if o["source_family"] == "ml")
    assert imported["normalization"]["repairs"]
    assert imported["original_prediction"] == prediction["instances"][0]
    assert sha256(prediction_path) == binding["sources"][0]["prediction_sha256"]


def test_dependency_cycle_is_rejected(plan, tmp_path):
    path, _ = plan
    document = analyze(path, tmp_path / "cycle")
    document["observations"][0]["dependency_ids"] = [document["observations"][0]["id"]]
    with pytest.raises(ValueError, match="Cyclic"):
        validate_document(document)


def test_track1_can_supply_only_raw_masks_segments_and_ocr(plan, tmp_path):
    path, _ = plan

    def raw_adapter(image, config):
        h, w = image.shape[:2]
        ink = (image[:, :, 0] < 128).astype(np.uint8)
        return {
            "working_size": [w, h],
            "working_to_canonical": np.eye(3).tolist(),
            "ink": ink,
            "wall_mask": ink.copy(),
            "raw_segments": [],
            "availability": {"ocr": "provided_by_test_adapter"},
            "text_spans": [
                {
                    "id": "ocr-1",
                    "text": "Kitchen",
                    "bbox": [40, 60, 100, 75],
                    "source_family": "ocr",
                    "engine_score": 0.8,
                }
            ],
        }

    document = analyze(path, tmp_path / "track1", pixel_adapter=raw_adapter)
    assert document["pages"][0]["availability"]["ocr"] == "provided_by_test_adapter"
    room = next(
        o for o in document["observations"] if o.get("room_type", {}).get("category") == "kitchen"
    )
    text = next(o for o in document["observations"] if o["source_family"] == "ocr")
    assert text["id"] in room["dependency_ids"]
    assert text["raw_text_span"]["engine_score"] == 0.8


def write_pdf(path):
    """Tiny vector fixture: two differently sized pages, one rotated/cropped."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 240 160] /CropBox [20 10 220 150] /Resources << /Font << /F1 7 0 R >> >> /Contents 4 0 R >>",
        None,
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /CropBox [20 10 280 190] /Rotate 90 /Resources << /Font << /F1 7 0 R >> >> /Contents 6 0 R >>",
        None,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for index in (3, 5):
        stream = b"0 0 0 RG 2 w 40 30 120 80 re S BT /F1 12 Tf 60 70 Td (Kitchen) Tj ET"
        objects[index] = (
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
    contents = b"%PDF-1.4\n"
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(contents))
        contents += str(i).encode() + b" 0 obj\n" + obj + b"\nendobj\n"
    xref = len(contents)
    contents += b"xref\n0 8\n0000000000 65535 f \n"
    contents += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    contents += f"trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(contents)


def test_pdf_crop_rotation_native_paths_and_roundtrip(tmp_path):
    path = tmp_path / "drawing.pdf"
    write_pdf(path)
    pages = list(load_pages(path, Config(pdf_dpi=144)))
    assert len(pages) == 2
    assert pages[0][1].image_size == [400, 280]
    assert pages[1][1].image_size == [360, 520]
    for image, frame, lines, text, coverage in pages:
        assert len(lines) == 4
        assert any(span["text"] == "Kitchen" for span in text)
        points = [[40, 30], [160, 110], [80.125, 60.25]]
        projected = transform_points(points, frame.source_to_canonical)
        restored = transform_points(projected, frame.canonical_to_source)
        assert np.max(np.abs(np.asarray(restored) - points)) < 1e-8
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        near = cv2.dilate((gray < 128).astype(np.uint8), np.ones((3, 3), np.uint8))
        for line in lines:
            p = np.mean(line, axis=0).astype(int)
            assert near[p[1], p[0]] == 1
    document = analyze(path, tmp_path / "pdf-result", config={"pdf_dpi": 144})
    assert len(document["pages"]) == 2


def test_crop_rejects_path_escape_and_modified_raster(plan, tmp_path):
    path, _ = plan
    out = tmp_path / "safe"
    document = analyze(path, out)
    document["pages"][0]["source_raster"] = "../plan.png"
    write_json(out / "evidence.json", document)
    with pytest.raises(ValueError, match="escapes"):
        get_region(out / "evidence.json", 0, [0, 0, 40, 40], tmp_path / "crop")
