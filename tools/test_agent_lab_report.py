import io
import json
import copy
from hashlib import sha256

from PIL import Image
import pytest

from build_agent_lab_report import build


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def trial(tmp_path):
    pack, run = tmp_path / "pack", tmp_path / "run"
    pack.mkdir()
    image = Image.new("RGB", (200, 100), "white")
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    (pack / "source.png").write_bytes(payload.getvalue())
    digest = sha256(payload.getvalue()).hexdigest()
    save(pack / "manifest.json", {"cases": [{"id": "one", "input_path": "source.png", "input_sha256": digest}]})
    answer = {"status": "answered", "answer": 2, "unit": "count", "explanation": "<script>alert(1)</script>"}
    save(run / "one/record.json", {"id": "one", "run": {"status": "completed", "result": {"answer": answer}}})
    row = {"id": "one", "trace": "one/record.json", "source_sha256": digest,
           "raster_sha256": sha256(image.tobytes()).hexdigest(), "question": "How many?",
           "reference": {"type": "count", "answer": 2, "unit": "count"}, "answer": answer,
           "source": "synthetic test", "category": "openings", "language": "en",
           "score": {"correct": True}, "seconds_http": 1.5}
    report = {"binding": {"case_ids": ["one"]}, "cases": [row]}
    save(run / "report.json", report)
    return pack, run, report


def test_visual_report_escapes_source_text_and_requires_original_pixels(tmp_path):
    pack, run, report = trial(tmp_path)
    output = tmp_path / "review.html"
    result = build([("Control", run)], [pack], output)
    assert result["drawings"] == 1
    html = output.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "\\u003cscript>" in html and "data:image/jpeg;base64," in html
    report["cases"][0]["raster_sha256"] = "wrong"
    save(run / "report.json", report)
    with pytest.raises(ValueError, match="rendering"):
        build([("Control", run)], [pack], output)


def test_incomplete_trials_and_escaped_traces_cannot_be_reported(tmp_path):
    pack, run, report = trial(tmp_path)
    report["binding"]["case_ids"].append("missing")
    save(run / "report.json", report)
    with pytest.raises(ValueError, match="Incomplete"):
        build([("Control", run)], [pack], tmp_path / "review.html")
    report["binding"]["case_ids"] = ["one"]
    report["cases"][0]["trace"] = "../outside.json"
    save(run / "report.json", report)
    with pytest.raises(ValueError, match="escapes"):
        build([("Control", run)], [pack], tmp_path / "review.html")


def test_comparison_rejects_changed_questions_and_holdout_assignment(tmp_path):
    pack, run, report = trial(tmp_path)
    other = tmp_path / "other"
    save(other / "one/record.json", json.loads((run / "one/record.json").read_text()))
    changed = copy.deepcopy(report)
    changed["cases"][0]["question"] = "A different task?"
    save(other / "report.json", changed)
    with pytest.raises(ValueError, match="question or evaluation cohort"):
        build([("Control", run), ("Changed", other)], [pack], tmp_path / "review.html")
    changed = copy.deepcopy(report)
    changed["binding"]["split"] = "holdout"
    save(other / "report.json", changed)
    with pytest.raises(ValueError, match="question or evaluation cohort"):
        build([("Control", run), ("Changed", other)], [pack], tmp_path / "review.html")
