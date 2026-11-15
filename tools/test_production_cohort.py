"""Offline checks for the production comparison, never provider requests."""

from copy import deepcopy
import json

import pytest

from audit_production_cohort import digest, evaluate, require_complete, run_case


def test_stop_gate_admits_no_new_request(tmp_path):
    (tmp_path / "STOP").write_text("Pause after a regression")
    assert run_case({}, 1, {}, None, tmp_path) is None


def test_expired_admission_deadline_blocks_even_upload(tmp_path, monkeypatch):
    import audit_production_cohort as audit
    from unittest.mock import Mock
    network = Mock(side_effect=AssertionError("No network after admission cutoff"))
    monkeypatch.setattr(audit.httpx, "Client", network)
    assert run_case({}, 1, {"admission_deadline_unix": 1}, None, tmp_path) is None
    network.assert_not_called()
    assert (tmp_path / "STOP").read_text() == "Admission deadline reached"


@pytest.mark.parametrize("with_metadata", [True, False])
def test_runner_uses_production_routes_and_never_repeats_completed_request(
    tmp_path, monkeypatch, with_metadata
):
    from fastapi.testclient import TestClient
    from PIL import Image
    from unittest.mock import Mock
    import app
    import audit_production_cohort as audit
    from run_tracking import RunStore

    Image.new("RGB", (40, 30), "white").save(tmp_path / "page.png")
    store = RunStore(tmp_path / "runs")
    monkeypatch.setattr(app, "_run_store", store)
    execute = Mock(
        return_value={
            "status": "ok",
            "model": "fixture",
            "answer": {"status": "answered", "answer": 2},
            "canonical_citations": [],
        }
    )
    monkeypatch.setattr(app, "ask", execute)
    client = TestClient(app.app)
    monkeypatch.setattr(audit.httpx, "Client", lambda **kwargs: client)
    case = {
        "id": "one",
        "input_path": "page.png",
        "input_sha256": digest(tmp_path / "page.png"),
        "question": "How many rooms?",
        "unit": "count",
        "category": "rooms",
        "language": "en",
        "source": "fixture",
        "type": "count",
    }
    binding = {
        "server": "http://testserver",
        "model": "fixture",
        "request": {"mode": "auto", "allow_gemini": True, "allow_nemotron": True},
    }
    if not with_metadata:
        case.pop("category")
        case.pop("language")
    first = run_case(case, 1, binding, tmp_path, tmp_path)
    second = run_case(case, 1, binding, tmp_path, tmp_path)
    assert first == second
    assert first["run"]["result"]["answer"]["answer"] == 2
    assert first["category"] == ("rooms" if with_metadata else "unspecified")
    assert first["language"] == ("en" if with_metadata else "unknown")
    assert json.loads((tmp_path / "one/record.json").read_text(encoding="utf-8")) == first
    execute.assert_called_once()
    assert execute.call_args.kwargs["observer"] is not None
    store.pool.shutdown()


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
