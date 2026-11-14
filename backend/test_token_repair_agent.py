"""Offline coordinator regressions: exhaustion recovery is bounded and evidential."""

import json
from unittest.mock import Mock, patch

import httpx
import pytest
from google.genai import types

from gemini_transport import ResilientGeminiTransport
from test_question_agent import agent, final, response


def exhausted(name="submit_answer", args=None):
    value = response(name, final() if args is None else args)
    value.candidates[0].finish_reason = "MAX_TOKENS"
    return value


def add_crops(value):
    # Disjoint existing source-pixel crops; repair must not invent new regions.
    for box in ([0, 0, 60, 80], [70, 0, 130, 80], [140, 0, 200, 80]):
        value.crop(box)


def phases(value):
    return [call.kwargs["phase"] for call in value.transport.generate.call_args_list]


def test_max_tokens_gets_distinct_final_only_repair_and_telemetry():
    value = agent(exhausted(), response("submit_answer", final()))
    result = value.run()
    assert phases(value) == ["normal", "token_repair"]
    assert result["status"] == "ok"
    assert result["answer"]["answer"] == 2
    reliability = result["reliability"]
    assert reliability["token_repair_used"] is True
    assert reliability["token_repair_succeeded"] is True
    assert reliability["delegated_recovery_calls"] == 0
    assert reliability["last_error"]["finish_reason"] == "MAX_TOKENS"
    assert reliability["retryable"] is False
    repair = value.transport.generate.call_args_list[-1]
    assert [tool.name for tool in repair.args[1]] == ["submit_answer"]
    context = json.loads(repair.args[0][0].parts[0].text)
    assert context["token_repair"] is True
    assert context["observations"]["coverage"] == "unknown"
    assert context["question"] == value.request.question
    assert [entry["phase"] for entry in result["gemini_calls"]] == phases(value)


def test_outage_recovery_can_itself_exhaust_then_use_distinct_token_repair():
    value = agent(
        httpx.ReadTimeout("private provider message"),
        exhausted(),
        response("submit_answer", final()),
    )
    result = value.run()
    assert phases(value) == ["normal", "recovery", "token_repair"]
    assert result["status"] == "ok"
    assert result["reliability"]["recovery_succeeded"] is True
    assert "private provider message" not in json.dumps(result)


def test_repeated_exhaustion_stops_without_promoting_partial_answer():
    value = agent(exhausted(), exhausted(), response("submit_answer", final()))
    result = value.run()
    assert phases(value) == ["normal", "token_repair"]
    assert result["status"] == "unavailable"
    assert result["answer"]["answer"] is None
    assert result["answer"]["citations"] == []
    assert result["reliability"]["token_repair_succeeded"] is False


@pytest.mark.parametrize(
    "name,args",
    [
        ("get_algorithms", {"classes": ["room"]}),
        ("crop_region", {"bbox": [0, 0, 500, 500]}),
        ("inspect_nemotron", {"image_ids": ["crop-1"], "question": "Count."}),
    ],
)
def test_incomplete_turn_tool_payloads_are_never_executed(name, args):
    value = agent(exhausted(name, args), response("submit_answer", final()))
    with patch.object(value, "execute", wraps=value.execute) as execute:
        result = value.run()
    execute.assert_not_called()
    value.algorithm_factory.assert_not_called()
    value.specialist_factory.assert_not_called()
    assert result["tool_trace"] == []
    assert list(value.images) == ["overview"]


@pytest.mark.parametrize("mode,allow", [("direct", True), ("auto", False)])
def test_repair_respects_routing_and_specialist_permission(mode, allow):
    value = agent(
        exhausted(), response("submit_answer", final()), mode=mode, allow=allow
    )
    add_crops(value)
    result = value.run()
    value.specialist_factory.assert_not_called()
    assert result["reliability"]["delegated_recovery_calls"] == 0
    assert (
        "specialist_not_authorized_for_this_mode"
        in result["reliability"]["token_repair_notes"]
    )
    assert result["status"] == "ok"


def test_delegation_is_two_crop_local_requests_with_short_timeouts_and_provenance():
    value = agent(exhausted(), response("submit_answer", final()))
    add_crops(value)
    registered = dict(value.images)
    result = value.run()
    calls = value.specialist_factory.return_value.answer.call_args_list
    assert len(calls) == 2
    ids = []
    for call in calls:
        question, images = call.args
        assert len(images) == 1
        image = images[0]
        assert image is registered[image.image_id]
        assert image.image_id != "overview"
        assert call.kwargs == {"timeout_seconds": 20}
        assert value.request.question in question
        assert "Do not answer for the whole page" in question
        ids.append(image.image_id)
    assert len(set(ids)) == 2
    assert value.images == registered
    value.algorithm_factory.assert_not_called()
    assert result["nemotron_calls"] == 2
    assert result["reliability"]["delegated_recovery_calls"] == 2
    assert all(
        entry["role"] == "token_recovery_subagent" for entry in result["tool_trace"]
    )
    assert all(entry["scope"] == "crop_local_only" for entry in result["tool_trace"])


@pytest.mark.parametrize(
    "specialist_calls,tool_calls,expected",
    [
        (3, 0, 0),
        (2, 0, 1),
        (0, 12, 0),
        (0, 11, 1),
    ],
)
def test_delegation_obeys_existing_specialist_and_tool_caps(
    specialist_calls, tool_calls, expected
):
    value = agent()
    add_crops(value)
    value.nemotron_calls = specialist_calls
    value.tool_log = [{"name": "reflect", "result": {}} for _ in range(tool_calls)]
    with patch("question_agent.time.monotonic", return_value=0):
        value.delegate_token_work(600)
    assert value.specialist_factory.return_value.answer.call_count == expected
    assert value.delegated_recovery_calls == expected
    assert value.nemotron_calls <= 3
    assert len(value.tool_log) <= 12


def test_slow_specialist_preserves_time_for_final_synthesis():
    value = agent()
    add_crops(value)
    now = [0.0]

    def answer(*args, **kwargs):
        now[0] += 20
        return {"status": "ok", "result": {"answer": 1}}

    value.specialist_factory.return_value.answer.side_effect = answer
    with patch("question_agent.time.monotonic", side_effect=lambda: now[0]):
        value.delegate_token_work(120)
    assert value.specialist_factory.return_value.answer.call_count == 1
    assert now[0] == 20
    assert "delegation_budget_reserved_for_finalization" in value.token_repair_notes


def test_deadline_exhaustion_does_not_schedule_repair():
    value = agent()
    now = [0.0]

    def generate(*args, **kwargs):
        now[0] = 600.0
        return exhausted()

    value.transport.generate.side_effect = generate
    with patch("question_agent.time.monotonic", side_effect=lambda: now[0]):
        result = value.run()
    assert phases(value) == ["normal"]
    assert result["status"] == "unavailable"
    assert result["reliability"]["token_repair_used"] is False


@pytest.mark.parametrize("blocked_by", ["request_budget", "cooldown"])
def test_no_delegation_without_an_available_finalization_request(blocked_by):
    value = agent()
    add_crops(value)
    transport = ResilientGeminiTransport(
        "fixture-only", "scripted-fixture", instruction="fixture", gate=Mock()
    )
    value.transport = transport
    if blocked_by == "request_budget":
        transport.request_count = transport.policy.max_requests
    else:
        transport.gate.check.side_effect = RuntimeError("blocked")
    with patch("question_agent.time.monotonic", return_value=0):
        value.delegate_token_work(600)
    value.specialist_factory.assert_not_called()
    assert value.delegated_recovery_calls == 0


def test_specialist_failure_remains_missing_evidence_without_blocking_finalization():
    value = agent(exhausted(), response("submit_answer", final()))
    add_crops(value)
    value.specialist_factory.return_value.answer.side_effect = httpx.ReadTimeout(
        "secret error"
    )
    result = value.run()
    assert result["status"] == "ok"
    assert result["reliability"]["delegated_recovery_calls"] == 2
    assert all(
        entry["result"]["status"] == "unavailable" for entry in result["tool_trace"]
    )
    assert "secret error" not in json.dumps(result)
    assert all(
        "never infer zero" in entry["result"]["instruction"]
        for entry in result["tool_trace"]
    )


def test_exhausted_attempt_usage_is_retained_alongside_successful_repair_usage():
    value = agent()
    transport = ResilientGeminiTransport(
        "fixture-only", "scripted-fixture", instruction="fixture", gate=Mock()
    )
    transport.events = [
        {
            "phase": "normal",
            "finish_reason": "MAX_TOKENS",
            "usage": {"thoughts_token_count": 8189, "candidates_token_count": 3},
        }
    ]
    completed = response("submit_answer", final())
    completed.usage_metadata = types.GenerateContentResponseUsageMetadata(
        candidates_token_count=50, thoughts_token_count=100, total_token_count=500
    )
    transport.generate = Mock(side_effect=[exhausted(), completed])
    value.transport = transport
    result = value.run()
    assert result["reliability"]["provider_attempts"] == transport.events
    assert result["gemini_calls"][-1]["phase"] == "token_repair"
    assert result["gemini_calls"][-1]["usage"]["candidates_token_count"] == 50
    assert result["gemini_calls"][-1]["usage"]["thoughts_token_count"] == 100


def test_repair_abstention_does_not_claim_successful_answer_recovery():
    abstention = final()
    abstention.update(
        status="abstain", answer=None, citations=[], uncertainties=["Missing evidence."]
    )
    value = agent(exhausted(), response("submit_answer", abstention))
    result = value.run()
    assert result["answer"]["status"] == "abstain"
    assert result["reliability"]["token_repair_used"] is True
    assert result["reliability"]["token_repair_succeeded"] is False
    assert result["reliability"]["recovery_succeeded"] is False
