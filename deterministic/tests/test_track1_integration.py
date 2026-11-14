from __future__ import annotations

import json

import cv2
import numpy as np
from PIL import Image

from planproof_evidence import analyze
from planproof_evidence.contracts import Config
from planproof_evidence import track1_adapter, track1_pixels


def drawing():
    image = np.full((320, 480, 3), 255, np.uint8)
    cv2.rectangle(image, (40, 40), (440, 280), (0, 0, 0), 10)
    cv2.line(image, (240, 40), (240, 280), (0, 0, 0), 10)
    return image


def test_track1_masks_holes_frame_and_untyped_cues():
    data = track1_adapter.extract_primitives(drawing(), Config(max_side=256, enable_ocr=False))
    assert data["working_size"] == [256, 171]
    assert data["ink"].dtype == np.uint8 and set(np.unique(data["ink"])) == {0, 1}
    assert data["working_to_canonical"][0][0] == 480 / 256
    assert data["working_to_canonical"][1][1] == 320 / 171
    details = data["pixel_details"]
    assert details["upstream_commit"] == track1_adapter.UPSTREAM_COMMIT
    assert details["counts"]["raw_segments"] > 0
    assert any(not r["is_outer"] for r in details["contour_rings"])
    assert all(c["class"] == "unknown" for c in details["cues"])
    assert "candidates" not in data  # generic size/shape cues never become semantic truth


def test_upstream_hough_fix_accepts_opencv_nested_shape(monkeypatch):
    class NoLSD:
        def detect(self, mask):
            return (None, None, None, None)

    monkeypatch.setattr(track1_pixels.cv2, "createLineSegmentDetector", lambda *_: NoLSD())
    monkeypatch.setattr(
        track1_pixels.cv2,
        "HoughLinesP",
        lambda *a, **k: np.array([[[5, 6, 90, 6]]], dtype=np.int32),
    )
    segments = track1_pixels.extract_line_segments(np.zeros((100, 120), np.uint8))
    assert len(segments) == 1 and segments[0]["method"] == "hough"
    assert list(segments[0]["a"]) == [5, 6]


def test_missing_ocr_is_unavailable_not_empty_success(monkeypatch):
    monkeypatch.setattr(track1_adapter.shutil, "which", lambda _: None)
    spans, availability = track1_adapter.ocr_spans(drawing(), "eng")
    assert spans == [] and availability["status"] == "unavailable"


def test_default_pipeline_really_runs_upstream_track1(tmp_path):
    source = tmp_path / "plan.png"
    Image.fromarray(drawing()).save(source)
    doc = analyze(source, tmp_path / "job", config={"enable_ocr": False})
    raw = json.loads((tmp_path / "job/page-000/pixel-evidence.json").read_text())
    assert raw["pixel_details"]["backend"] == "github-track1"
    assert raw["pixel_details"]["upstream_commit"] == track1_adapter.UPSTREAM_COMMIT
    assert doc["config"]["pixel_backend"] == "track1"
    assert all(o["mode"] == "independent" for o in doc["observations"])
    assert {o.get("class") for o in doc["observations"]} >= {"room", "wall"}
    assert all(a["exact_physical_count"] is None for a in doc["aggregates"])
    assert (tmp_path / "job/agent-context.json").exists()
