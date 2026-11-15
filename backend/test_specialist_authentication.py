"""Rejected credentials stop paid orchestration without weakening evidence rules."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from planproof_feedback.client import ProviderError
from specialist_routing import parallel_inspections
from test_question_agent import response
from test_specialist_routing import orchestrator, target


@pytest.mark.parametrize("status", [401, 403])
def test_parallel_auth_failure_prevents_queued_requests_and_redacts_body(monkeypatch, status):
    import specialist_routing as routing
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(routing, "_POOL", pool)
        calls = []

        def fail(item, seconds):
            calls.append(item)
            raise ProviderError("SECRET should never reach evidence", status_code=status)

        rows, attempted = parallel_inspections(list(range(6)), fail,
            deadline=time.monotonic() + 2, seconds=1, on_event=lambda *args: None)
    assert calls == [0] and attempted == 1
    assert rows[0]["http_status"] == status
    assert rows[0]["error_code"] == "SpecialistAuthenticationError"
    assert all(row["status"] == "not_sent" for row in rows[1:])
    assert "SECRET" not in repr(rows)


@pytest.mark.parametrize("status", [401, 403])
def test_auth_rejection_finishes_without_another_gemini_turn(status):
    agent = orchestrator(response("inspect_regions", {"regions": [target()]}))
    agent.specialist_factory.return_value.answer.side_effect = ProviderError("SECRET", status_code=status)
    result = agent.run()
    assert agent.transport.generate.call_count == 1
    assert result["stop_reason"] == "specialist_authentication_failed"
    assert result["answer"]["answer"] is None
    assert f"HTTP {status}" in result["answer"]["explanation"]
    assert result["reliability"]["specialist_http_status"] == status
    assert not result["reliability"]["recovery_used"]
    assert "SECRET" not in repr(result)


def test_transient_provider_error_does_not_disable_other_tiles():
    def inspect(item, seconds):
        if item == 0:
            raise ProviderError("untrusted body", status_code=503)
        return {"status": "ok", "result": {"answer": "visible"}}
    rows, attempted = parallel_inspections([0, 1], inspect,
        deadline=time.monotonic() + 2, seconds=1, on_event=lambda *args: None)
    assert attempted == 2
    assert rows[0]["http_status"] == 503
    assert rows[1]["status"] == "ok"
