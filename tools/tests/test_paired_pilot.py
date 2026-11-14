"""Comparison semantics and failure journaling, using generated records only."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_paired_pilot import (
    RecordingBackend,
    summarize,
    unrequested_presentation_failure,
    save,
)
from build_paired_report import build


def side(correct, status="ok", seconds=1):
    return {
        "score": {"correct": correct},
        "status": status,
        "seconds": seconds,
        "events": [{"status": "complete"}],
    }


def test_failed_and_abstained_conditions_remain_in_the_denominator():
    rows = [
        {"direct": side(True), "guided": side(False, "error", 3)},
        {"direct": side(False), "guided": side(True)},
        {"direct": side(False, "invalid_output"), "guided": side(False)},
    ]
    summary = summarize(rows)
    assert summary["direct"]["correct"] == summary["guided"]["correct"] == 1
    assert summary["direct"]["total"] == summary["guided"]["total"] == 3
    assert summary["guided"]["failures"] == 1
    assert summary["guided"]["seconds"] == 5
    assert summary["direct_wins"] == summary["guided_wins"] == 1


def test_transport_failure_is_saved_before_it_propagates():
    class Fails:
        def complete(self, messages):
            raise RuntimeError("Generated transport failure")

    events, states = [], []
    backend = RecordingBackend(
        Fails(), "routing", events, lambda: states.append(events[-1]["status"])
    )
    with pytest.raises(RuntimeError, match="Generated transport failure"):
        backend.complete([])
    assert states == ["running", "error"]
    assert events[0]["error_type"] == "RuntimeError"
    assert events[0]["seconds"] >= 0


def test_completed_routing_output_survives_later_contract_failure():
    class Completes:
        def complete(self, messages):
            return {"content": '{"action":"invalid"}', "usage": {"total_tokens": 12}}

    events = []
    backend = RecordingBackend(Completes(), "routing", events, lambda: None)
    backend.complete([])
    assert events[0]["completion"]["content"] == '{"action":"invalid"}'
    assert events[0]["status"] == "complete"


def test_browser_rename_lock_falls_back_to_writing_the_same_page(tmp_path, monkeypatch):
    import json
    import build_paired_report

    run, view = tmp_path / "run", tmp_path / "view"
    run.mkdir()
    view.mkdir()
    (run / "report.json").write_text(json.dumps({"cases": [], "status": "complete"}))
    (view / "index.html").write_text("Old page")

    def locked_replace(self, destination):
        raise PermissionError("Generated browser file handle denies rename")

    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr(build_paired_report.time, "sleep", lambda _: None)
    build(run, view)
    assert 'id="paired-data"' in (view / "index.html").read_text(encoding="utf-8")


def test_only_unrequested_local_failures_can_be_retried(tmp_path):
    record = {"status": "error", "error": "Cannot replace paired-index.tmp"}
    assert unrequested_presentation_failure(record, tmp_path)
    call = tmp_path / "calls" / "one"
    call.mkdir(parents=True)
    (call / "request.json").write_text("{}")
    assert not unrequested_presentation_failure(record, tmp_path)
    assert not unrequested_presentation_failure(
        {"status": "error", "error": "Invalid routing action"}, tmp_path
    )


def test_checkpoint_survives_a_windows_rename_lock(tmp_path, monkeypatch):
    import json
    import run_paired_pilot

    def locked(self, target):
        raise PermissionError("Generated checkpoint reader denies rename")

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(run_paired_pilot.time, "sleep", lambda _: None)
    save(tmp_path / "report.json", {"status": "saved"})
    assert json.loads((tmp_path / "report.json").read_text()) == {"status": "saved"}
