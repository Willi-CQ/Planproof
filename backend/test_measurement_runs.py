"""Measurement tools survive live observation and the selected PDF page contract."""

from uuid import uuid4

from fastapi.testclient import TestClient

import app as module
from run_tracking import RunStore
from test_measurements import rectangle_request
from test_question_agent import agent, final, response
from test_run_tracking import finished


def test_measurement_flows_through_resumable_api_with_citations(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "runs")
    monkeypatch.setattr(module, "_run_store", store)
    monkeypatch.setenv("PLANPROOF_AGENT_TRACE_ROOT", str(tmp_path / "traces"))
    scripted = agent(
        response("calculate_area", rectangle_request()),
        response(
            "submit_answer",
            {**final(), "answer": 251.12, "unit": "ft2", "measurement_id": "M1"},
        ),
    )
    monkeypatch.setattr("question_agent.settings", lambda: ("fixture", "fixture"))
    monkeypatch.setattr("question_agent.GeminiTransport", lambda *_: scripted.transport)
    # A stored page number other than one exercises the API's selected-PDF-page mapping.
    scripted.page.source_sha256 = "b" * 64
    document = module.store_page(scripted.page, 4)
    run_id = str(uuid4())
    with TestClient(module.app) as client:
        result = client.post(
            f"/api/v1/documents/{document.document_id}/runs",
            json={
                "request_id": run_id,
                "request": {
                    "question": "Room area in square feet?",
                    "allow_gemini": True,
                },
            },
        )
        assert result.status_code == 202
        finished(store, run_id)
        saved = client.get(f"/api/v1/runs/{run_id}").json()
        assert saved["status"] == "completed"
        assert saved["result"]["answer"]["answer"] == 251.12
        assert saved["result"]["measurement"]["page_index"] == 3
        assert all(
            c["page_index"] == 3
            for c in saved["result"]["measurement"]["canonical_citations"]
        )
        event = next(
            e
            for e in saved["events"]
            if e["stage"] == "calculate_area" and e["state"] == "completed"
        )
        assert "251.12 ft2" in event["summary"]
        assert event["citations"] and all(
            c["page_index"] == 3 for c in event["citations"]
        )
        calls = scripted.transport.generate.call_count
        client.get(f"/api/v1/runs/{run_id}")
        assert scripted.transport.generate.call_count == calls == 2
    store.pool.shutdown()


def test_legacy_ask_maps_measurement_citations_to_selected_page(monkeypatch):
    scripted = agent()
    scripted.page.source_sha256 = "b" * 64
    document = module.store_page(scripted.page, 4)
    measurement = scripted.execute("calculate_area", rectangle_request())[0]
    monkeypatch.setattr(
        module,
        "ask",
        lambda *_: {
            "status": "ok",
            "measurement": measurement,
            "canonical_citations": measurement["canonical_citations"],
        },
    )
    with TestClient(module.app) as client:
        saved = client.post(
            f"/api/v1/documents/{document.document_id}/ask", json={"question": "Area?"}
        ).json()
    assert saved["measurement"]["page_index"] == 3
    assert all(
        c["page_index"] == 3 for c in saved["measurement"]["canonical_citations"]
    )
