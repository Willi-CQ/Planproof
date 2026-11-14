"""Offline checks for the production comparison, never provider requests."""

from copy import deepcopy
import json

import pytest

from audit_production_cohort import digest, evaluate, require_complete


def test_evaluator_stays_closed_for_missing_duplicate_or_running_cases(tmp_path):
    binding = {"case_ids": ["one", "two"]}
    completed = {"id": "one", "run": {"status": "completed"}}
    for records in (
        [completed],
        [completed, completed],
        [completed, {"id": "two", "run": {"status": "running"}}],
    ):
        with pytest.raises(ValueError):
            evaluate(binding, records, tmp_path, tmp_path)


def test_failed_attempts_are_complete_and_stay_in_denominator():
    require_complete(
        {"case_ids": ["one"]}, [{"id": "one", "run": {"status": "failed"}}]
    )


def test_primary_score_preserves_unit_errors_and_counts_failures(tmp_path):
    pack, baseline = tmp_path / "pack", tmp_path / "baseline"
    (pack / "evaluator").mkdir(parents=True)
    baseline.mkdir()
    labels = {
        cid: {"type": "count", "answer": 4, "unit": "count"} for cid in ("one", "two")
    }
    (pack / "evaluator/answers.json").write_text(json.dumps(labels))
    (pack / "manifest.json").write_text(
        json.dumps({"answers_sha256": digest(pack / "evaluator/answers.json")})
    )
    old = [
        {
            "id": cid,
            "mode": "auto",
            "question": "How many doors?",
            "answer": {"status": "answered", "answer": 4, "unit": "count"},
            "score": {"correct": True},
            "seconds": 10,
            "status": "ok",
            "raster_sha256": "same",
        }
        for cid in labels
    ]
    (baseline / "report.json").write_text(json.dumps({"cases": old}))
    binding = {
        "case_ids": list(labels),
        "baseline_report_sha256": digest(baseline / "report.json"),
    }
    records = [
        {
            "id": cid,
            "case_number": i + 1,
            "seconds_http": 20,
            "raster_sha256": "same",
            "run": {
                "status": "completed",
                "result": {
                    "status": "ok",
                    "answer": {
                        "status": "answered",
                        "answer": 4,
                        "unit": "door openings",
                    },
                },
            },
        }
        for i, cid in enumerate(labels)
    ]
    records[1]["run"] = {"status": "failed", "result": None}
    before = deepcopy(records)
    report = evaluate(binding, records, pack, baseline)
    assert report["summary"]["cases"] == 2
    assert report["summary"]["current_correct"] == 0
    assert report["summary"]["current_count_value_correct"] == 1
    assert report["summary"]["returned_answers"] == 1
    assert report["summary"]["regressions"] == [1, 2]
    assert records == before
