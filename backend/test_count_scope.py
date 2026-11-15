"""Scoped totals retain cited exclusions instead of silently discarding objects."""

import pytest




from test_count_evidence import counting_agent, counted_answer, localized
from test_task_scope import selected_scope


@pytest.fixture(autouse=True)
def candidate_profile(monkeypatch):
    monkeypatch.setenv("PLANPROOF_AGENT_PROFILE", "scope-count-candidate")


def prepared():
    agent = counting_agent()
    _, images = agent.crop([0, 0, 100, 100])
    agent.record_inspection(localized("", images))
    agent.task_scope = {**selected_scope(), "counting_unit": "door openings"}
    return agent


def exclusion():
    return {"object_id": "crop-1-D2", "reason": "Door belongs to the neighboring unit outside the selected boundary",
            "citations": [{"image_id": "crop-1", "bbox": [25, 1, 50, 30]}]}


def scoped_answer(excluded=None):
    return {**counted_answer(1, ["crop-1-D1"]), "count_basis": {
        "unit": "door_opening", "scope": "complete", "object_ids": ["crop-1-D1"],
        "exclusions": excluded if excluded is not None else [exclusion()]}}


def test_selected_unit_total_preserves_excluded_object_and_cited_reason():
    agent = prepared()
    result = agent.finish(scoped_answer())
    assert result["status"] == "answered" and result["answer"] == 1
    evidence = agent.count_inventory()
    assert evidence["supported"] == 1
    assert evidence["excluded_objects"][0]["id"] == "crop-1-D2"
    assert evidence["excluded_objects"][0]["exclusion"] == exclusion()
    assert len(agent.count_records["crop-1"]["objects"]) == 2
    assert exclusion()["citations"][0] in result["citations"]
    assert "Apartment A" in result["coverage"]


@pytest.mark.parametrize("scope", [None, {**selected_scope(), "extent": "whole_sheet"},
                                   {**selected_scope(), "extent": "unresolved"}])
def test_whole_sheet_or_unspecified_scope_cannot_drop_objects(scope):
    agent = prepared()
    agent.task_scope = scope
    with pytest.raises(ValueError, match="explicit selected"):
        agent.finish(scoped_answer())
    assert agent.count_inventory()["supported"] == 2


@pytest.mark.parametrize("exclusions", [
    [{**exclusion(), "object_id": "made-up"}],
    [exclusion(), exclusion()],
    [{**exclusion(), "citations": [{"image_id": "crop-1", "bbox": [1, 1, 5, 5]}]}],
    [{**exclusion(), "citations": [{"image_id": "unknown", "bbox": [25, 1, 50, 30]}]}],
])
def test_invalid_or_unrelated_exclusions_do_not_mutate_inventory(exclusions):
    agent = prepared()
    with pytest.raises(ValueError):
        agent.finish(scoped_answer(exclusions))
    assert agent.count_inventory()["supported"] == 2
    assert not agent.count_exclusions


def test_proposed_subset_without_exclusion_evidence_still_abstains():
    agent = prepared()
    assert agent.finish(scoped_answer([]))["status"] == "abstain"
    assert agent.count_inventory()["supported"] == 2


def test_exclusion_cannot_turn_incomplete_coverage_into_complete_answer():
    agent = prepared()
    agent.count_records["crop-1"]["scope_complete"] = False
    result = agent.finish(scoped_answer())
    assert result["status"] == "abstain"
    assert not agent.count_exclusions
    assert agent.count_inventory()["supported"] == 2


def test_zero_with_all_observations_outside_scope_keeps_exclusion_citations():
    agent = prepared()
    first = {**exclusion(), "object_id": "crop-1-D1",
             "citations": [{"image_id": "crop-1", "bbox": [5, 1, 25, 30]}]}
    draft = scoped_answer([first, exclusion()])
    draft["answer"] = 0
    draft["count_basis"]["object_ids"] = []
    result = agent.finish(draft)
    assert result["status"] == "answered" and result["answer"] == 0
    assert len(agent.count_inventory()["excluded_objects"]) == 2
    assert all(c in result["citations"] for c in first["citations"] + exclusion()["citations"])


def test_later_failed_draft_clears_prior_scope_exclusions():
    agent = prepared()
    assert agent.finish(scoped_answer())["status"] == "answered"
    assert agent.finish(scoped_answer([]))["status"] == "abstain"
    assert not agent.count_exclusions
    assert agent.count_inventory()["supported"] == 2
