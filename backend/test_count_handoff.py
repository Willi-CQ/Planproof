"""Localized observations survive context routing without inventing coverage."""

import json

import pytest

from test_count_evidence import counting_agent, counted_answer, localized, target
from test_visual_specialist import observation, setup


def obj(image_id="crop-1", box=None, unit="door_opening"):
    return {"image_id": image_id, "bbox": box or [100, 200, 300, 400],
            "unit": unit, "status": "supported", "reason": "Visible leaf and arc"}


def test_context_question_keeps_its_answer_and_returns_located_objects():
    raw = {**observation(), "answer": "The door is beside the corridor label",
           "objects": [obj()], "scope_complete": True}
    specialist, backend, image = setup(raw)
    backend.settings = {}
    result = specialist.answer("Read the adjacent room label", [image],
                               inventory_unit="door_opening", timeout_seconds=2)
    assert result["status"] == "ok"
    assert result["result"]["answer"] == raw["answer"]
    assert result["result"]["objects"][0]["bbox"] == [20, 20, 60, 40]
    assert result["result"]["scope_complete"] is False
    sent = json.loads(backend.complete.call_args.args[0][1]["content"][0]["text"])
    assert sent["question"] == "Read the adjacent room label"
    assert sent["inventory_unit"] == "door_opening"


@pytest.mark.parametrize("change", [
    {"unit": "door_leaf"}, {"image_id": "unregistered"},
    {"bbox": [0, 0, 1001, 50]}, {"bbox": [30, 40, 20, 10]},
])
def test_context_objects_have_same_unit_and_coordinate_guards_as_counts(change):
    specialist, _, image = setup({**observation(), "objects": [{**obj(), **change}]})
    result = specialist.answer("Identify the doorway", [image], inventory_unit="door_opening")
    assert result["status"] == "invalid_output"
    assert result["result"] is None


def test_prose_count_is_not_promoted_into_objects_or_coverage():
    specialist, _, image = setup({**observation(), "answer": "Seven doors", "scope_complete": True})
    result = specialist.answer("Check the doorway symbols", [image], inventory_unit="door_opening")
    assert result["result"]["objects"] == []
    assert result["result"]["scope_complete"] is False


def test_context_discovery_survives_without_overwriting_full_count_record():
    agent = counting_agent()
    _, images = agent.crop([0, 0, 100, 100])
    original = localized("", images)
    agent.record_inspection(original)
    context = {"status": "ok", "result": {
        **original["result"], "answer": "Sliding door in the kitchen partition",
        "scope_complete": False,
        "objects": [obj(images[0].image_id, [55, 10, 65, 20])],
    }}
    agent.record_inspection(context, counting=False)
    agent.record_inspection(context, counting=False)
    assert agent.count_records[images[0].image_id] == original["result"]
    assert agent.count_inventory()["supported"] == 3
    assert not agent.uncovered_count_objects(agent.count_inventory())
    result = agent.finish({**counted_answer(3), "count_basis": {
        "unit": "door_opening", "scope": "complete"}})
    assert result["status"] == "answered" and result["answer"] == 3
    assert len(result["citations"]) == 3


def test_context_alone_never_establishes_complete_count():
    agent = counting_agent()
    _, images = agent.crop([0, 0, 100, 100])
    agent.record_inspection(localized("", images), counting=False)
    assert agent.count_inventory()["supported"] == 2
    assert len(agent.count_tool_inventory()["objects_needing_count_coverage"]) == 2
    assert agent.finish(counted_answer())["status"] == "abstain"


def test_new_object_outside_counted_region_blocks_total_until_region_is_counted():
    agent = counting_agent()
    _, first = agent.crop([0, 0, 100, 100])
    agent.record_inspection(localized("", first))
    _, second = agent.crop([150, 0, 200, 100])
    agent.record_inspection(localized("", second), counting=False)
    evidence = agent.count_tool_inventory()
    assert evidence["supported"] == 4
    assert len(evidence["objects_needing_count_coverage"]) == 2
    result = agent.finish({**counted_answer(4), "count_basis": {
        "unit": "door_opening", "scope": "complete"}})
    assert result["status"] == "abstain"


def test_actual_batch_requests_auxiliary_objects_without_rewriting_context_question():
    agent = counting_agent()
    region = {**target(), "purpose": "inspect", "question": "Locate the sliding leaf beside the kitchen"}
    result, _ = agent.execute("inspect_regions", {"regions": [region]}, timeout_seconds=2)
    call = agent.specialist_factory.return_value.answer.call_args
    assert "Locate the sliding leaf" in call.args[0]
    assert "COUNT_UNIT=" not in call.args[0]
    assert call.kwargs["inventory_unit"] == "door_opening"
    assert result["count_evidence"]["supported"] == 2
    assert not agent.count_records
