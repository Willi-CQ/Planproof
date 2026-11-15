"""Scope propagation contracts; these do not claim model interpretation accuracy."""

import pytest



import json

from test_specialist_routing import orchestrator, target


@pytest.fixture(autouse=True)
def candidate_profile(monkeypatch):
    monkeypatch.setenv("PLANPROOF_AGENT_PROFILE", "scope-count-candidate")


def selected_scope():
    return {
        "extent": "selected_unit",
        "target": "Apartment A",
        "basis": "The user selected the highlighted apartment A",
        "exclusions": ["Partial apartment B at the left edge"],
        "counting_unit": "toilet bowls",
    }


def test_scope_shared_between_workers_and_retained_for_followup():
    agent = orchestrator()
    original = agent.request.question
    result, _ = agent.execute("inspect_regions", {
        "task_scope": selected_scope(), "regions": [target()],
    }, timeout_seconds=2)
    first = agent.specialist_factory.return_value.answer.call_args.args[0]
    assert "Apartment A" in first and "Partial apartment B" in first
    assert "Do not infer unseen membership" in first
    assert result["task_scope"] == selected_scope()
    agent.execute("inspect_regions", {
        "regions": [target([500, 0, 1000, 500])],
    }, timeout_seconds=2)
    assert "Apartment A" in agent.specialist_factory.return_value.answer.call_args.args[0]
    assert agent.request.question == original
    assert agent.task_scope == selected_scope()
    recovery = json.loads(agent.recovery_history(compact=True)[0].parts[0].text)
    assert recovery["task_scope"] == selected_scope()


def test_scope_correction_does_not_reuse_a_different_context_inspection():
    agent = orchestrator()
    region = target()
    agent.execute("inspect_regions", {
        "task_scope": selected_scope(), "regions": [region],
    }, timeout_seconds=2)
    agent.execute("inspect_regions", {
        "task_scope": {**selected_scope(), "extent": "whole_sheet",
                       "target": "All visible units", "exclusions": [],
                       "basis": "User explicitly requested all visible fixtures"},
        "regions": [region],
    }, timeout_seconds=2)
    assert agent.specialist_factory.return_value.answer.call_count == 2
    assert agent.task_scope["extent"] == "whole_sheet"


def test_invalid_batch_cannot_change_scope_or_start_workers():
    agent = orchestrator()
    with pytest.raises(ValueError):
        agent.execute("inspect_regions", {
            "task_scope": selected_scope(),
            "regions": [target([0, 0, 1200, 300])],
        })
    assert agent.task_scope is None
    assert agent.specialist_factory.call_count == 0


def test_scope_does_not_delete_outside_observations_or_modify_answers():
    agent = orchestrator()
    result, _ = agent.execute("inspect_regions", {
        "task_scope": selected_scope(), "regions": [target()],
    }, timeout_seconds=2)
    assert result["inspections"][0]["advisory"]["answer"] == 1
    assert result["inspections"][0]["advisory"]["citations"]
