"""Contract tests on generated arrays, not real-data accuracy tests."""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "review_reference10.py"
SPEC = importlib.util.spec_from_file_location("reference_review", SOURCE)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def targets():
    return {
        "masks": np.zeros((5, 32, 32), np.uint8),
        "instances": np.zeros((4, 32, 32), np.uint16),
        "valid": np.zeros((5, 32, 32), np.uint8),
        "pixel_valid": np.ones((32, 32), np.uint8),
        "objects": np.zeros((0, 5), np.float32),
        "object_complete": np.ones(3, np.uint8),
    }


def prediction(items=()):
    return {
        "document_id": "drawing-1",
        "page_index": 0,
        "image_size": [32, 32],
        "coordinate_system": "pixel_edges_xy_top_left",
        "instances": list(items),
    }


def item(name, box, identity="p1"):
    return {"class": name, "bbox": box, "id": identity}


def score(predicted, truth):
    return review.score_case(predicted, truth, "drawing-1", [32, 32])


def test_unannotated_classes_are_not_zero():
    truth = targets()
    truth["object_complete"][:] = 0
    assert all(
        v["status"] == "not_annotated" for v in score(prediction(), truth).values()
    )


def test_same_count_wrong_location_still_fails():
    truth = targets()
    truth["objects"] = np.array([[0, 2, 2, 8, 8]], np.float32)
    result = score(prediction([item("door", [20, 20, 26, 26])]), truth)["door"]
    assert result["count_error"] == 0
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)
    assert result["f1_iou50"] == 0


def test_known_box_overrides_pixel_ignore_but_unknown_predictions_do_not():
    truth = targets()
    truth["pixel_valid"][:] = 0
    truth["objects"] = np.array([[2, 4, 4, 12, 12]], np.float32)
    result = score(
        prediction(
            [
                item("stairs", [4, 4, 12, 12]),
                item("stairs", [20, 20, 30, 30], "ignored"),
            ]
        ),
        truth,
    )["stairs"]
    assert result["tp"] == 1 and result["fp"] == 0
    assert result["ignored_prediction_ids"] == ["ignored"]


def test_global_assignment_keeps_both_valid_matches():
    assert sorted(review.matching(np.array([[0.7, 0.6], [0.69, 0.0]]))) == [
        (0, 1),
        (1, 0),
    ]


def test_empty_assignment():
    assert review.matching(np.zeros((0, 3))) == []
    assert review.matching(np.zeros((4, 0))) == []


def test_room_area_preserves_hole():
    truth = targets()
    truth["valid"][1] = 1
    outer = [[3, 3], [28, 3], [28, 28], [3, 28]]
    hole = [[10, 10], [20, 10], [20, 20], [10, 20]]
    mask = np.zeros((32, 32), np.uint8)
    cv2.fillPoly(mask, [np.array(outer)], 1)
    cv2.fillPoly(mask, [np.array(hole)], 0)
    truth["instances"][0] = mask
    truth["masks"][1] = mask
    room = {
        **item("room", [3, 3, 29, 29]),
        "polygons": [{"outer": outer, "holes": [hole]}],
    }
    result = score(prediction([room]), truth)["room"]
    assert result["f1_iou50"] == 1
    assert result["matched_area_relative_error_mean"] == 0


@pytest.mark.parametrize(
    "field,value",
    [("document_id", "other"), ("page_index", 1), ("image_size", [64, 64])],
)
def test_binding_mismatch_rejected(field, value):
    supplied = prediction()
    supplied[field] = value
    with pytest.raises(ValueError):
        score(supplied, targets())


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        score(
            prediction([item("door", [2, 2, 8, 8]), item("window", [10, 10, 15, 15])]),
            targets(),
        )


def test_missing_prediction_is_not_run(tmp_path, monkeypatch):
    case = {
        "case_id": "generated-fixture",
        "source": "generated",
        "directory": "unused",
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"cases": [case]}))
    (tmp_path / "SHA256SUMS.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["review", "--pack", str(tmp_path)])
    review.main()
    report = json.loads((tmp_path / "failure-report.json").read_text())
    assert report["results"][0]["status"] == "not_run"
    assert report["summary"]["deterministic"]["cases_scored"] == 0
    assert report["summary"]["deterministic"]["classes"]["room"]["truth"] is None
    html = (tmp_path / "review.html").read_text(encoding="utf-8")
    assert "\u2014" in html
    assert "\u00e2\u20ac" not in html
