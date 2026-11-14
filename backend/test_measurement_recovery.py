"""A provider outage must not disable arithmetic on already inspected evidence."""

from unittest.mock import Mock, patch

import httpx
import pytest

from test_measurements import polygon_request, rectangle_request
from test_question_agent import agent, final, response
from test_token_repair_agent import exhausted, phases
from gemini_transport import ResilientGeminiTransport


def measured(value=44, unit="m2"):
    return {**final(), "answer": value, "unit": unit, "measurement_id": "M1"}


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("private"), exhausted()])
def test_recovery_measures_then_finalizes_with_existing_source_citations(failure):
    value = agent(
        failure,
        response("measure_polygon", polygon_request()),
        response("submit_answer", measured()),
        allow=False,
    )
    result = value.run()
    assert result["answer"]["answer"] == 44
    assert result["measurement"]["approximate"]
    assert result["reliability"]["recovery_measurement_calls"] == 1
    assert result["reliability"]["recovery_succeeded"]
    assert len(value.calls) == 3
    assert phases(value)[1] == phases(value)[2]
    # Continuation retains the tool result in the same recovery conversation.
    history = value.transport.generate.call_args.args[0]
    assert any(
        p.function_response and p.function_response.name == "measure_polygon"
        for message in history
        for p in message.parts
    )
    value.algorithm_factory.assert_not_called()
    value.specialist_factory.assert_not_called()


def test_invalid_measurement_can_be_corrected_once_with_no_unverified_answer():
    invalid = polygon_request()
    invalid["anchors"][1]["dimension"]["text"] = "80 m"
    value = agent(
        httpx.ReadTimeout("private"),
        response("measure_polygon", invalid),
        response("measure_polygon", polygon_request()),
        response("submit_answer", measured()),
        allow=False,
    )
    result = value.run()
    assert result["reliability"]["recovery_measurement_calls"] == 2
    assert result["tool_trace"][0]["result"]["status"] == "unavailable"
    assert result["answer"]["answer"] == 44
    assert [t.name for t in value.transport.generate.call_args.args[1]] == [
        "submit_answer"
    ]


def test_repeated_calls_are_bounded_including_cache_hits():
    call = response("calculate_area", rectangle_request())
    value = agent(
        httpx.ReadTimeout("private"),
        call,
        call,
        call,
        response("submit_answer", measured(251.12, "ft2")),
        allow=False,
    )
    result = value.run()
    assert value.transport.generate.call_count == 4
    assert result["reliability"]["recovery_measurement_calls"] == 2
    assert result["answer"]["answer"] is None


@pytest.mark.parametrize("mode,budget", [("direct", 0), ("auto", 12)])
def test_direct_mode_and_shared_tool_cap_do_not_gain_measurement_permission(
    mode, budget
):
    value = agent(
        httpx.ReadTimeout("private"),
        response("measure_polygon", polygon_request()),
        mode=mode,
        allow=False,
    )
    value.tool_log = [{"name": "reflect", "result": {}} for _ in range(budget)]
    result = value.run()
    assert not value.measurements
    assert result["reliability"]["recovery_measurement_calls"] == 0
    assert [t.name for t in value.transport.generate.call_args.args[1]] == [
        "submit_answer"
    ]


@pytest.mark.parametrize(
    "name,args",
    [
        ("crop_region", {"bbox": [0, 0, 500, 500]}),
        ("get_algorithms", {"classes": ["room"]}),
        ("inspect_nemotron", {"image_ids": ["overview"], "question": "Inspect"}),
    ],
)
def test_measurement_recovery_does_not_reopen_observation_tools(name, args):
    value = agent(httpx.ReadTimeout("private"), response(name, args))
    with patch.object(value, "execute", wraps=value.execute) as execute:
        result = value.run()
    execute.assert_not_called()
    assert result["answer"]["answer"] is None


def test_little_time_remaining_keeps_final_only_recovery():
    value = agent()
    now = [0.0]

    def generate(*args, **kwargs):
        if now[0] == 0:
            now[0] = 575
            raise httpx.ReadTimeout("private")
        return response("submit_answer", final())

    value.transport.generate.side_effect = generate
    with patch("question_agent.time.monotonic", side_effect=lambda: now[0]):
        value.run()
    assert [t.name for t in value.transport.generate.call_args.args[1]] == [
        "submit_answer"
    ]


def test_abstention_reports_interruption_without_claiming_source_has_no_information():
    abstain = {**final(), "status": "abstain", "answer": None}
    value = agent(httpx.ReadTimeout("private"), response("submit_answer", abstain))
    result = value.run()
    assert "incomplete processing" in result["answer"]["uncertainties"][0]
    assert not result["reliability"]["recovery_succeeded"]


@pytest.mark.parametrize("remaining", [1, 2])
def test_provider_request_budget_reserves_a_final_answer(remaining):
    value = agent(allow=False)
    transport = ResilientGeminiTransport(
        "fixture", "fixture", instruction="fixture", gate=Mock()
    )
    transport.request_count = 12 - remaining
    recovery_calls = []

    def generate(history, tools, **kwargs):
        if kwargs["phase"] == "normal":
            raise httpx.ReadTimeout("fixture")
        recovery_calls.append(([t.name for t in tools], kwargs["max_attempts"]))
        transport.request_count += 1
        if remaining == 2 and len(recovery_calls) == 1:
            return response("measure_polygon", polygon_request())
        return response("submit_answer", measured() if remaining == 2 else final())

    transport.generate = Mock(side_effect=generate)
    value.transport = transport
    result = value.run()
    assert transport.request_count == 12
    assert result["status"] == "ok"
    assert recovery_calls[-1][0] == ["submit_answer"]
    if remaining == 2:
        assert recovery_calls[0][1] == 1


def test_incomplete_calculation_payload_never_executes():
    value = agent(
        exhausted("measure_polygon", polygon_request()),
        response("submit_answer", final()),
        allow=False,
    )
    result = value.run()
    assert not value.measurements
    assert result["reliability"]["recovery_measurement_calls"] == 0
