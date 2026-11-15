"""Vector boundary provenance and strict measurement fallback contracts."""

from copy import deepcopy
import pymupdf
import pytest

from native_text import extract_lines
from specialist_routing import DelegationRequired
from test_measurements import polygon_request
from test_question_agent import agent, final
from vector_regions import extract_regions, locate_regions, measure_regions


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_source_polygon_follows_rotation_and_keeps_label_association(rotation):
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        page.draw_rect(pymupdf.Rect(50, 50, 200, 150), fill=(.9, .9, .9))
        page.insert_text((80, 100), "Room A")
        page.set_rotation(rotation)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        lines = extract_lines(page, pix, 2)
        regions = extract_regions(page, pix, 2, lines)
    result = locate_regions(regions, {"query": "Room A"})
    assert len(result["regions"]) == 1
    assert result["regions"][0]["area_px2"] == pytest.approx(60_000)
    assert result["status"] == "candidates_only"
    assert "value" not in result["regions"][0]


def test_curves_and_compound_holes_are_omitted_not_silently_filled():
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        shape = page.new_shape()
        shape.draw_rect(pymupdf.Rect(50, 50, 200, 200))
        shape.draw_rect(pymupdf.Rect(80, 80, 100, 100))
        shape.finish(fill=(.5, .5, .5), even_odd=True)
        shape.commit()
        page.draw_circle((280, 120), 30, fill=(.5, .5, .5))
        pix = page.get_pixmap()
        assert extract_regions(page, pix, 1, ()) == ()


def fixture():
    value = agent()
    original = polygon_request()
    region = {"id": "V1", "polygon": original["polygon"], "bbox": [100, 200, 500, 800], "labels": ["Exercise"]}
    request = {k: original[k] for k in ("target", "output_unit", "boundary_basis", "view_bbox", "anchors")}
    request["regions"] = [{"region_id": "V1", "image_id": "overview"}]
    return value, region, request


def test_calibration_is_required_and_vector_area_uses_source_points():
    value, region, request = fixture()
    result = measure_regions(request, [region], {"V1"}, list(value.images.values()), (200, 100))
    assert result["value"] == 48
    assert result["approximate"]
    assert result["components"][0]["canonical_polygon"][0] == [20, 20]
    assert result["citations"][0]["bbox"] == [20, 20, 100, 80]
    with pytest.raises(ValueError, match="supplied"):
        measure_regions(request, [region], set(), list(value.images.values()), (200, 100))
    bad = deepcopy(request)
    bad["anchors"][1]["dimension"]["text"] = "12 m"
    with pytest.raises(ValueError, match="disagree"):
        measure_regions(bad, [region], {"V1"}, list(value.images.values()), (200, 100))


def test_duplicate_or_overlapping_regions_cannot_double_count_area():
    value, region, request = fixture()
    other = {**region, "id": "V2"}
    request["regions"].append({"region_id": "V2", "image_id": "overview"})
    with pytest.raises(ValueError, match="overlap"):
        measure_regions(request, [region, other], {"V1", "V2"}, list(value.images.values()), (200, 100))
    request["regions"][1]["region_id"] = "V1"
    with pytest.raises(ValueError, match="distinct"):
        measure_regions(request, [region], {"V1"}, list(value.images.values()), (200, 100))


def test_partial_crop_and_uninspected_measurement_cannot_establish_answer():
    value, region, request = fixture()
    value.crop([20, 20, 60, 60])
    request["regions"][0]["image_id"] = "crop-1"
    with pytest.raises(ValueError, match="entire"):
        measure_regions(request, [region], {"V1"}, list(value.images.values()), (200, 100))
    request["regions"][0]["image_id"] = "overview"
    result = measure_regions(request, [region], {"V1"}, list(value.images.values()), (200, 100))
    value.measurements["M1"] = result
    value.require_specialist = True
    with pytest.raises(DelegationRequired):
        value.finish({**final(), "answer": 48, "unit": "m2", "measurement_id": "M1"})
