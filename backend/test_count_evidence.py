"""Scripted counting regressions; no external inference or accuracy claims."""

from dataclasses import replace

import pytest

from count_evidence import door_count_unit
from app import set_answer_page
from specialist_routing import InspectionPolicy, tile_question
from test_question_agent import final, response
from test_specialist_routing import orchestrator, target as base_target
from test_visual_specialist import observation, setup


@pytest.mark.parametrize(
    "question,expected",
    [
        (
            "How many doors are there? Exclude openings without a door leaf.",
            "door_opening",
        ),
        ("Count doorways that have leaves; ignore open passages.", "door_opening"),
        (
            "How many doors are shown? Do not count door leaves individually.",
            "door_opening",
        ),
        ("What is the number of individual door leaves?", "door_leaf"),
        ("Count each door leaf separately.", "door_leaf"),
        ("Count the leaves of the doors.", "door_leaf"),
        ("Where is the door leaf?", None),
        ("How many windows are visible?", None),
    ],
)
def test_counting_convention_comes_from_requested_noun(question, expected):
    assert door_count_unit(question) == expected


def target(*args, **kwargs):
    return {**base_target(*args, **kwargs), "purpose": "count"}


def counting_agent(*responses):
    value = orchestrator(*responses)
    value.request.question = "How many doors do we have in this drawing?"
    value.count_unit = door_count_unit(value.request.question)
    value.specialist_factory.return_value.answer.side_effect = localized
    return value


def localized(question, images, **kwargs):
    image = images[0]
    objects = [
        {
            "image_id": image.image_id,
            "bbox": [10, 10, 20, 20],
            "unit": "door_opening",
            "status": "supported",
            "reason": "Leaf and arc",
        },
        {
            "image_id": image.image_id,
            "bbox": [30, 10, 40, 20],
            "unit": "door_opening",
            "status": "supported",
            "reason": "Double-leaf pair, one opening",
        },
    ]
    return {
        "status": "ok",
        "result": {
            "status": "observed",
            "answer": None,
            "scope_complete": True,
            "objects": objects,
            "verdict": "uncertain",
            "findings": [],
            "citations": [{"image_id": image.image_id, "bbox": [0, 0, 50, 100]}],
        },
    }


def counted_answer(count=2, ids=None):
    return {
        **final("crop-1"),
        "answer": count,
        "count_basis": {
            "object_ids": ids or ["crop-1-D1", "crop-1-D2"],
            "scope": "complete",
            "unit": "door_opening",
        },
    }


def test_count_workers_get_original_question_not_leading_layout_or_coordinate_cutoff():
    task = tile_question(
        "Confirm 16 rooms x4=65; use x305",
        "How many doors?",
        [200, 100, 400, 200],
        [200, 0, 400, 200],
        1000,
        500,
        "door_opening",
    )
    assert "16 rooms" not in task and "x305" not in task
    assert "[200.0, 200.0, 400.0, 400.0]" in task
    assert "LOCAL" in task and "COUNT_UNIT=door_opening" in task
    assert "entire selected region: False" in task
    assert "How many doors?" not in task
    assert "locate ALL visible door symbols" in task


def test_context_question_is_preserved_and_never_becomes_a_door_count():
    value = counting_agent()
    seen = []

    def inspect(question, images, **kwargs):
        seen.append(question)
        if "COUNT_UNIT" in question:
            return localized(question, images)
        return {
            "status": "ok",
            "result": {
                "status": "observed",
                "answer": "East Wing",
                "findings": ["Label reads East Wing"],
                "verdict": "uncertain",
                "citations": [{"image_id": images[0].image_id, "bbox": [1, 1, 20, 20]}],
            },
        }

    value.specialist_factory.return_value.answer.side_effect = inspect
    result, _ = value.execute(
        "inspect_regions",
        {
            "regions": [
                {
                    "bbox": [500, 0, 1000, 500],
                    "question": "Read the wing label",
                    "purpose": "inspect",
                },
                target(),
            ]
        },
        timeout_seconds=2,
    )
    assert "Read the wing label" in seen[0] and "COUNT_UNIT" not in seen[0]
    assert "COUNT_UNIT" in seen[1] and "east appartement" not in seen[1]
    assert result["inspections"][0]["advisory"]["citations"]
    assert set(value.count_records) == {"crop-2"}
    assert not value.unresolved_inspections
    assert result["count_evidence"]["supported"] == 2


@pytest.mark.parametrize("candidate", [False, True])
def test_broad_context_region_does_not_starve_count_region_and_has_separate_cache_key(monkeypatch, candidate):
    monkeypatch.setenv("PLANPROOF_AGENT_PROFILE", "scope-count-candidate" if candidate else "qualified-v6")
    value = counting_agent()
    value.inspection_policy = replace(
        InspectionPolicy(), tile_edge=40, overlap=8, max_calls=2
    )
    result, _ = value.execute(
        "inspect_regions",
        {
            "regions": [
                {
                    "bbox": [0, 0, 500, 1000],
                    "question": "Read the title",
                    "purpose": "inspect",
                },
                target([500, 0, 700, 400]),
            ]
        },
        timeout_seconds=2,
    )
    assert [row["purpose"] for row in result["inspections"][:2]] == ["inspect", "count"]
    assert value.nemotron_calls == 2
    assert result["count_evidence"]["supported"] == (4 if candidate else 2)
    if candidate:
        assert len(result["count_evidence"]["objects_needing_count_coverage"]) == 2
    else:
        assert "objects_needing_count_coverage" not in result["count_evidence"]
    assert all(
        "COUNT_UNIT" not in call.args[0]
        for call in value.specialist_factory.return_value.answer.call_args_list[:1]
    )


def test_legacy_label_inspection_keeps_question_and_does_not_overwrite_inventory():
    value = counting_agent()
    value.crop([0, 0, 100, 100])
    value.execute(
        "inspect_nemotron",
        {
            "image_ids": ["crop-1"],
            "question": "Read the printed wing label",
            "purpose": "inspect",
        },
    )
    assert (
        value.specialist_factory.return_value.answer.call_args.args[0]
        == "Read the printed wing label"
    )
    assert not value.count_records


def test_ambiguous_scope_can_ask_a_question_without_erasing_local_candidates():
    proposed = {
        **counted_answer(),
        "status": "abstain",
        "answer": None,
        "clarification_question": "Do you mean the whole east wing or one room?",
    }
    value = counting_agent(
        response("inspect_regions", {"regions": [target()]}),
        response("submit_answer", proposed),
    )
    result = value.run()
    assert result["completion_reason"] == "clarification_needed"
    assert result["count_evidence"]["supported"] == 2
    assert (
        result["answer"]["briefing"][-1]["text"] == proposed["clarification_question"]
    )


def test_door_evidence_includes_room_navigation_without_a_second_algorithm_call():
    value = counting_agent()
    catalog, _ = value.execute("get_algorithms", {"classes": ["door"]})
    assert catalog["rooms"][0]["id"] == "R1"
    assert "R1" in value.rooms and value.algorithm_factory.call_count == 1


@pytest.mark.parametrize("verdict", ["uncertain", "observed"])
def test_real_specialist_normalizes_each_object_and_reserves_count_output_tokens(
    verdict,
):
    obs = {
        **observation(),
        "verdict": verdict,
        "scope_complete": True,
        "answer": 65,
        "objects": [
            {
                "image_id": "crop-1",
                "bbox": [100, 200, 500, 800],
                "unit": "door_opening",
                "status": "supported",
                "reason": "Leaf and arc",
            }
        ],
    }
    specialist, backend, image = setup(obs)
    backend.settings = {}
    result = specialist.answer(
        "COUNT_UNIT=door_opening. How many?", [image], timeout_seconds=10
    )
    assert result["status"] == "ok"
    assert result["result"]["answer"] is None
    assert result["result"]["objects"][0]["bbox"] == [20, 20, 100, 80]
    assert result["result"]["scope_complete"] is True


def test_uncertain_objects_and_legacy_aggregate_cannot_claim_complete_inventory():
    for objects in (
        None,
        [
            {
                "image_id": "crop-1",
                "bbox": [100, 200, 500, 800],
                "unit": "door_opening",
                "status": "uncertain",
                "reason": "Cut off",
            }
        ],
    ):
        obs = {**observation(), "answer": 65, "scope_complete": True}
        if objects is not None:
            obs["objects"] = objects
        specialist, _, image = setup(obs)
        result = specialist.answer("COUNT_UNIT=door_opening", [image])
        assert result["status"] == "ok"
        assert result["result"]["scope_complete"] is False
        assert result["result"]["answer"] is None


@pytest.mark.parametrize("box", [[-1, 0, 50, 50], [0, 0, 1001, 1000]])
def test_malformed_object_coordinates_never_reach_inventory(box):
    obj = {
        "image_id": "crop-1",
        "bbox": box,
        "unit": "door_opening",
        "status": "supported",
        "reason": "Arc",
    }
    specialist, _, image = setup({**observation(), "objects": [obj]})
    assert (
        specialist.answer("COUNT_UNIT=door_opening", [image])["status"]
        == "invalid_output"
    )


def test_supported_total_requires_exact_localized_ids_and_replaces_aggregate_prose():
    value = counting_agent(
        response("inspect_regions", {"regions": [target()]}),
        response("submit_answer", counted_answer()),
    )
    result = value.run()
    assert result["answer"]["answer"] == 2
    assert result["count_evidence"]["complete"]
    assert len(result["canonical_citations"]) == 2
    assert result["answer"]["unit"] == "count"
    assert result["answer"]["count_basis"]["unit"] == "door_opening"
    assert "door openings" in result["answer"]["explanation"]
    assert result["answer"]["briefing"][0]["citations"] == [1, 2]


@pytest.mark.parametrize(
    "count,ids", [(65, None), (2, ["made-up"]), (2, ["crop-1-D1", "crop-1-D1"])]
)
def test_guessed_or_duplicate_count_rejected_but_candidates_survive(count, ids):
    value = counting_agent(
        response("inspect_regions", {"regions": [target()]}),
        response("submit_answer", counted_answer(count, ids)),
    )
    result = value.run()
    assert result["answer"]["status"] == "abstain"
    assert result["answer"]["answer"] is None
    assert "65" not in result["answer"]["explanation"]
    assert result["count_evidence"]["supported"] == 2
    assert result["completion_reason"] == "inspection_incomplete"
    assert len(result["count_evidence"]["objects"]) == 2


def test_uncertain_tile_retains_candidates_and_selectively_retries_with_same_identity():
    value = counting_agent()

    def uncertain(question, images, **kwargs):
        result = localized(question, images)
        result["result"]["objects"][0]["status"] = "uncertain"
        result["result"]["scope_complete"] = False
        return result

    value.specialist_factory.return_value.answer.side_effect = uncertain
    args = {"regions": [target()]}
    first, _ = value.execute("inspect_regions", args, timeout_seconds=2)
    assert first["status"] == "partial" and first["count_evidence"]["uncertain"] == 1
    value.specialist_factory.return_value.answer.side_effect = localized
    second, _ = value.execute("inspect_regions", args, timeout_seconds=2)
    assert second["status"] == "ok" and second["count_evidence"]["uncertain"] == 0
    assert value.nemotron_calls == 2 and value.crop_count == 1
    assert value.finish(counted_answer())["answer"] == 2


def test_overlap_deduplicates_same_door_and_conflicts_remain_uncertain():
    value = counting_agent()
    _, images = value.crop([0, 0, 100, 100])
    first = localized("", images)
    value.record_inspection(first)
    _, images = value.crop([5, 0, 105, 100])
    second = localized("", images)
    # Same original locations, different crop offsets.
    for obj in second["result"]["objects"]:
        obj["bbox"][0] -= 5
        obj["bbox"][2] -= 5
    second["result"]["objects"][0]["status"] = "uncertain"
    value.record_inspection(second)
    evidence = value.count_inventory()
    assert len(evidence["objects"]) == 2
    assert evidence["supported"] == 1 and evidence["uncertain"] == 1


def test_overlapping_distinct_symbols_in_same_crop_are_not_silently_merged():
    value = counting_agent()
    _, images = value.crop([0, 0, 100, 100])
    record = localized("", images)
    record["result"]["objects"][1]["bbox"] = [12, 10, 22, 20]
    value.record_inspection(record)
    evidence = value.count_inventory()
    assert len(evidence["objects"]) == 2
    assert evidence["uncertain"] == 2


def test_six_remaining_calls_are_used_even_when_selection_expands_to_nine():
    value = counting_agent()
    value.inspection_policy = replace(InspectionPolicy(), tile_edge=40, overlap=8)
    value.nemotron_calls = 12
    result, _ = value.execute(
        "inspect_regions", {"regions": [target()]}, timeout_seconds=2
    )
    assert value.nemotron_calls == 18
    assert result["coverage"]["returned_tiles"] == 6
    assert result["coverage"]["deferred_tiles"] == 3
    assert len(value.unresolved_inspections) == 3


def test_argument_repairs_do_not_consume_productive_tool_budget_but_are_bounded():
    value = counting_agent()
    value.tool_log = [{"result": {"error_code": "ValueError"}}] * 12
    assert value.tool_budget_used == 0
    value.tool_log *= 2
    assert value.tool_budget_used == 12


def test_pdf_page_projection_keeps_inventory_with_the_correct_page():
    value = counting_agent()
    value.execute("inspect_regions", {"regions": [target()]}, timeout_seconds=2)
    result = {"count_evidence": value.count_inventory(), "canonical_citations": []}
    set_answer_page(result, 3)
    assert {o["page_index"] for o in result["count_evidence"]["objects"]} == {3}


def test_smaller_complete_views_replace_an_old_uncertain_inventory_and_recovery_keeps_ids():
    value = counting_agent()

    def uncertain(question, images, **kwargs):
        result = localized(question, images)
        result["result"].update(scope_complete=False)
        result["result"]["objects"][0]["status"] = "uncertain"
        return result

    value.specialist_factory.return_value.answer.side_effect = uncertain
    value.execute("inspect_regions", {"regions": [target()]}, timeout_seconds=2)
    value.specialist_factory.return_value.answer.side_effect = localized
    value.execute(
        "inspect_regions",
        {"regions": [target([0, 0, 500, 500]), target([0, 500, 500, 1000])]},
        timeout_seconds=2,
    )
    assert not value.unresolved_inspections
    assert "crop-1" not in value.count_records
    assert value.count_inventory()["supported"] == 4
    history = value.recovery_history(compact=True)
    assert "crop-2-D1" in history[0].parts[0].text
    proposed = {
        **final("crop-2"),
        "answer": 4,
        "count_basis": {"scope": "complete", "unit": "door_opening"},
    }
    assert value.finish(proposed)["answer"] == 4
