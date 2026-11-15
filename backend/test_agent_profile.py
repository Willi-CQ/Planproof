"""Release guards compare the default model contract with frozen qualified V6."""

import hashlib
import json
from pathlib import Path

import pytest

from agent_profile import CANDIDATE, QUALIFIED, active_profile
from question_agent import agent_instruction, declarations
from test_count_evidence import counted_answer, counting_agent, target
from test_task_scope import selected_scope


FROZEN = json.loads((Path(__file__).parent / "fixtures/qualified_agent_contracts.json").read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.fixture(autouse=True)
def qualified_default(monkeypatch):
    monkeypatch.delenv("PLANPROOF_AGENT_PROFILE", raising=False)


@pytest.mark.parametrize("case", FROZEN["schema_checks"])
def test_default_tool_schema_matches_frozen_v6(case):
    tools = declarations(case["mode"], case["allowed"], **case["options"])
    assert digest([t.model_dump(mode="json", exclude_none=True) for t in tools]) == case["sha256"]


@pytest.mark.parametrize("orchestrated", [False, True])
def test_default_instruction_matches_frozen_v6(orchestrated):
    assert digest(agent_instruction(orchestrated)) == FROZEN["instructions"][str(orchestrated)]


def test_candidate_requires_explicit_opt_in(monkeypatch):
    assert active_profile() == QUALIFIED
    assert "TASK SCOPE:" not in agent_instruction(True)
    monkeypatch.setenv("PLANPROOF_AGENT_PROFILE", CANDIDATE)
    assert active_profile() == CANDIDATE
    assert "TASK SCOPE:" in agent_instruction(True)
    tools = {t.name: t.parameters_json_schema for t in declarations("auto", True, orchestrated=True)}
    assert "task_scope" in tools["inspect_regions"]["properties"]
    assert "exclusions" in tools["submit_answer"]["$defs"]["CountBasis"]["properties"]


def test_unknown_profile_does_not_silently_enable_candidate(monkeypatch):
    monkeypatch.setenv("PLANPROOF_AGENT_PROFILE", "typo")
    with pytest.raises(ValueError, match="Unknown PLANPROOF_AGENT_PROFILE"):
        active_profile()


def test_status_reports_profile_without_claiming_provider_availability(monkeypatch):
    import app

    monkeypatch.setattr(app, "question_agent_settings", lambda: ("test-only", "configured-model"))
    status = app.question_agent_status()
    assert status["agent_profile"] == QUALIFIED
    assert status["configured"] and status["connection_verified"] is False


def test_default_context_call_does_not_request_or_promote_count_objects():
    agent = counting_agent()
    result, _ = agent.execute("inspect_regions", {"regions": [{**target(), "purpose": "inspect"}]}, timeout_seconds=2)
    assert "inventory_unit" not in agent.specialist_factory.return_value.answer.call_args.kwargs
    assert result["count_evidence"]["supported"] == 0
    assert "task_scope" not in result
    assert "task_scope" not in json.loads(agent.recovery_history(compact=True)[0].parts[0].text)
    assert not agent.count_observations


def test_default_rejects_candidate_scope_before_provider_calls():
    agent = counting_agent()
    with pytest.raises(ValueError, match="opt-in"):
        agent.execute("inspect_regions", {"task_scope": selected_scope(), "regions": [target()]})
    assert agent.specialist_factory.call_count == 0


def test_default_count_answer_retains_qualified_contract():
    agent = counting_agent()
    agent.execute("inspect_regions", {"regions": [target()]}, timeout_seconds=2)
    answer = agent.finish(counted_answer())
    assert answer["status"] == "answered" and answer["answer"] == 2
    assert "exclusions" not in answer["count_basis"]
    assert "objects_needing_count_coverage" not in agent.count_tool_inventory()


def test_default_cannot_apply_scope_exclusion_even_if_model_supplies_it():
    agent = counting_agent()
    with pytest.raises(ValueError, match="opt-in"):
        agent.apply_count_scope([{"object_id": "invented"}])
