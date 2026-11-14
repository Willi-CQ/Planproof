"""Injected failures test recovery guarantees; none of these are accuracy claims."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from google.genai import errors

from gemini_transport import (
    ProviderFailure,
    ProviderGate,
    ResilientGeminiTransport,
    RetryPolicy,
    failure_info,
    response_candidate,
    retry_after,
)
from test_question_agent import agent, final, response


@pytest.fixture
def transport():
    t = ResilientGeminiTransport(
        "never-log-this-key", "test-model", instruction="test", gate=ProviderGate()
    )
    t._generate = Mock(return_value=response("submit_answer", final()))
    with (
        patch("gemini_transport.time.sleep"),
        patch("gemini_transport.random.uniform", return_value=0),
    ):
        yield t


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("private host"),
        httpx.RemoteProtocolError("private body"),
        httpx.ReadTimeout("private request"),
        TimeoutError("deadline"),
        errors.ServerError(503, {"secret": "body"}),
    ],
)
def test_transient_failure_recovers_on_fresh_attempt_and_logs_safely(transport, exc):
    transport._generate.side_effect = [exc, response("submit_answer", final())]
    assert response_candidate(transport.generate([], [])).finish_reason.value == "STOP"
    assert transport.request_count == 2
    assert transport.events[0]["retryable"] and transport.events[1]["status"] == "ok"
    assert "private" not in json.dumps(transport.events) and "secret" not in json.dumps(
        transport.events
    )


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
def test_terminal_http_errors_are_not_retried(transport, status):
    transport._generate.side_effect = errors.ClientError(
        status, {"message": "credential-private"}
    )
    with pytest.raises(ProviderFailure) as err:
        transport.generate([], [])
    assert not err.value.info["retryable"] and err.value.info["http_status"] == status
    assert transport.request_count == 1 and "credential-private" not in str(err.value)


def test_exhaustion_is_bounded_and_records_every_failed_attempt(transport):
    snapshots = []
    transport.on_attempt = lambda: snapshots.append(len(transport.events))
    transport._generate.side_effect = httpx.ConnectError("secret")
    with pytest.raises(ProviderFailure):
        transport.generate([], [])
    assert transport.request_count == 3 and len(transport.events) == 3
    assert [e.get("retry_delay_seconds") for e in transport.events] == [1, 2, None]
    assert snapshots == [1, 2, 3]


def test_long_retry_after_does_not_sleep_past_budget_or_retry_early(transport):
    transport._generate.side_effect = errors.ClientError(
        429, {}, httpx.Response(429, headers={"Retry-After": "300"})
    )
    with (
        patch("gemini_transport.time.sleep") as sleep,
        pytest.raises(ProviderFailure) as err,
    ):
        transport.generate([], [], deadline=time.monotonic() + 5)
    assert err.value.info["retry_after_seconds"] == 300
    sleep.assert_not_called()
    assert transport.request_count == 1
    with pytest.raises(ProviderFailure) as cooldown:
        transport.generate([], [])
    assert (
        cooldown.value.info["category"] == "cooldown" and transport.request_count == 1
    )


def test_retry_after_http_date_and_invalid_values():
    from email.utils import formatdate

    exc = errors.ServerError(
        503,
        {},
        httpx.Response(
            503, headers={"Retry-After": formatdate(time.time() + 30, usegmt=True)}
        ),
    )
    assert 28 <= retry_after(exc) <= 31
    for raw in ("not-a-date", "inf", "nan"):
        assert (
            retry_after(
                errors.ClientError(
                    429, {}, httpx.Response(429, headers={"Retry-After": raw})
                )
            )
            is None
        )


def test_cooldown_after_repeated_outage_and_concurrency_limit():
    gate = ProviderGate()
    assert gate.slots.acquire(blocking=False) and gate.slots.acquire(blocking=False)
    assert not gate.slots.acquire(blocking=False)
    gate.slots.release()
    gate.slots.release()
    for _ in range(4):
        gate.record({"category": "connection"})
    with pytest.raises(ProviderFailure):
        gate.check()
    with patch("gemini_transport.time.monotonic", return_value=gate.until + 1):
        gate.check()


def test_expired_deadline_and_total_request_cap_start_no_call(transport):
    with pytest.raises(ProviderFailure):
        transport.generate([], [], deadline=time.monotonic() - 1)
    transport._generate.assert_not_called()
    transport.request_count = transport.policy.max_requests
    with pytest.raises(ProviderFailure):
        transport.generate([], [])
    transport._generate.assert_not_called()


def test_waiting_for_slot_is_bounded(transport):
    transport.gate.slots.acquire()
    transport.gate.slots.acquire()
    started = time.monotonic()
    with pytest.raises(ProviderFailure) as err:
        transport.generate([], [], deadline=started + 0.07)
    assert err.value.info["error_code"] == "ProviderQueueTimeout"
    assert time.monotonic() - started < 0.5
    transport._generate.assert_not_called()
    transport.gate.slots.release()
    transport.gate.slots.release()


def test_remaining_deadline_caps_single_attempt(transport):
    transport.generate([], [], deadline=time.monotonic() + 2)
    assert 0 < transport._generate.call_args.args[2] <= 2


@pytest.mark.parametrize(
    "reason,retryable",
    [
        ("MAX_TOKENS", True),
        ("MALFORMED_FUNCTION_CALL", True),
        ("SAFETY", False),
        ("RECITATION", False),
    ],
)
def test_incomplete_generation_keeps_finish_reason_and_never_executes_partial_tools(
    transport, reason, retryable
):
    incomplete = response("submit_answer", final())
    incomplete.candidates[0].finish_reason = reason
    transport._generate.return_value = incomplete
    with pytest.raises(ProviderFailure) as err:
        transport.generate([], [])
    assert err.value.info["finish_reason"] == reason
    assert err.value.info["retryable"] is retryable and transport.request_count == 1


def test_async_timeout_cancels_operation_and_closes_clients():
    cancelled = []

    async def hang(**kwargs):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)

    client = Mock()
    client.aio.models.generate_content = hang
    client.aio.aclose = AsyncMock()
    t = ResilientGeminiTransport(
        "key",
        "model",
        instruction="test",
        policy=RetryPolicy(attempts=1, attempt_seconds=0.05),
        gate=ProviderGate(),
    )
    started = time.monotonic()
    with (
        patch("gemini_transport.genai.Client", return_value=client),
        pytest.raises(ProviderFailure) as err,
    ):
        t.generate([], [])
    assert err.value.info["category"] == "timeout" and cancelled == [True]
    assert time.monotonic() - started < 2
    client.close.assert_called_once()
    client.aio.aclose.assert_awaited_once()


def test_recovery_reuses_evidence_without_replaying_algorithms_or_specialist(tmp_path):
    a = agent(
        response("get_algorithms", {"classes": ["room"]}),
        response("crop_room", {"room_id": "R1"}),
        httpx.ReadTimeout("secret"),
        response("submit_answer", final("crop-1")),
    )
    a.out = tmp_path / "result.json"
    result = a.run()
    assert result["status"] == "ok" and result["reliability"]["recovery_succeeded"]
    a.algorithm_factory.assert_called_once()
    a.specialist_factory.assert_not_called()
    history, tools = a.transport.generate.call_args.args
    assert [t.name for t in tools] == ["submit_answer"] and len(history) == 3
    assert "observations" in history[0].parts[0].text and "crop-1" in str(history[0])
    assert a.transport.generate.call_args.kwargs["phase"] == "recovery"
    assert json.loads(a.out.read_text())["answer"]["answer"] == 2
    assert "secret" not in a.out.read_text()


def test_no_draft_or_partial_answer_is_promoted_during_outage():
    incomplete = response("submit_answer", final())
    incomplete.candidates[0].finish_reason = "MAX_TOKENS"
    a = agent(incomplete, httpx.ReadTimeout("private"))
    r = a.run()
    assert r["answer"]["answer"] is None and r["status"] == "unavailable"
    assert (
        r["reliability"]["recovery_used"] and not r["reliability"]["recovery_succeeded"]
    )
    assert len(r["gemini_calls"]) == 2 and not r["tool_trace"]


def test_recovery_invalid_citation_abstains_and_does_not_loop():
    a = agent(httpx.ReadTimeout("private"), response("submit_answer", final("made-up")))
    r = a.run()
    assert r["status"] == "unavailable" and r["answer"]["answer"] is None
    assert (
        r["stop_reason"] == "invalid_recovery_answer"
        and a.transport.generate.call_count == 2
    )


def test_recovery_cannot_run_new_tools():
    a = agent(
        httpx.ReadTimeout("private"), response("get_algorithms", {"classes": ["room"]})
    )
    r = a.run()
    assert r["status"] == "unavailable"
    a.algorithm_factory.assert_not_called()


def test_authentication_and_blocked_responses_do_not_trigger_recovery():
    for exc in [
        errors.ClientError(403, {}),
        ProviderFailure(
            {
                "error_code": "IncompleteResponse",
                "category": "blocked_response",
                "retryable": False,
                "finish_reason": "SAFETY",
            }
        ),
    ]:
        a = agent(exc)
        r = a.run()
        assert (
            not r["reliability"]["recovery_used"]
            and a.transport.generate.call_count == 1
        )


def test_redaction_of_unknown_exception():
    info = failure_info(ValueError("private key / private file path"))
    assert info == {
        "error_code": "ValueError",
        "retryable": False,
        "category": "client_error",
    }


def test_budget_boundary_preserves_finalization_reserve():
    now = [0.0]
    a = agent()

    def generate(*args, **kwargs):
        if kwargs["phase"] == "normal":
            now[0] = 481.0
            raise ProviderFailure(
                {
                    "error_code": "ProviderBudgetExhausted",
                    "category": "budget",
                    "retryable": False,
                }
            )
        return response("submit_answer", final())

    a.transport.generate.side_effect = generate
    with patch("question_agent.time.monotonic", side_effect=lambda: now[0]):
        r = a.run()
    assert (
        r["reliability"]["recovery_succeeded"] and a.transport.generate.call_count == 2
    )
    assert a.transport.generate.call_args.kwargs["deadline"] == 600


def test_client_protocol_errors_are_not_transient(transport):
    transport._generate.side_effect = httpx.LocalProtocolError("invalid header")
    with pytest.raises(ProviderFailure) as err:
        transport.generate([], [])
    assert not err.value.info["retryable"] and transport.request_count == 1


def test_empty_blocked_response_does_not_start_recovery():
    from google.genai import types

    blocked = types.GenerateContentResponse(prompt_feedback={"block_reason": "SAFETY"})
    a = agent(blocked)
    result = a.run()
    assert result["answer"]["answer"] is None
    assert result["reliability"]["last_error"]["finish_reason"] == "SAFETY"
    assert not result["reliability"]["recovery_used"]


def test_actual_sdk_has_no_hidden_retries_and_keeps_http_status():
    original_client = httpx.AsyncClient
    calls = []

    async def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(
                503,
                json={
                    "error": {
                        "code": 503,
                        "message": "private-provider-body",
                        "status": "UNAVAILABLE",
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "submit_answer",
                                        "args": final(),
                                    }
                                }
                            ],
                        },
                    }
                ]
            },
        )

    def factory(**kwargs):
        return original_client(**kwargs, transport=httpx.MockTransport(handler))

    t = ResilientGeminiTransport(
        "test-key", "test-model", instruction="test", gate=ProviderGate()
    )
    from question_agent import declarations
    from google.genai import types

    history = [
        types.Content(
            role="user", parts=[types.Part.from_text(text="Count visible objects")]
        )
    ]
    with (
        patch("gemini_transport.httpx.AsyncClient", side_effect=factory),
        patch("gemini_transport.time.sleep"),
    ):
        result = t.generate(history, declarations("direct", False))
    assert len(calls) == 2 and len(t.events) == 2
    assert t.events[0]["http_status"] == 503 and t.events[1]["status"] == "ok"
    assert (
        response_candidate(result).content.parts[0].function_call.name
        == "submit_answer"
    )
    assert "temperature" not in calls[0].get("generationConfig", {})
    assert "private-provider-body" not in json.dumps(t.events)
