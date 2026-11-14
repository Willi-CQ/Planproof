"""Scripted providers verify orchestration and resolution, not model accuracy."""

import io
import json
import threading
import time
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import app
from question_agent import AgentBudget, GeminiTransport, declarations
from specialist_routing import (
    DelegationRequired,
    InspectionPolicy,
    parallel_inspections,
    region_boxes,
    require_inspected_citations,
    tiles,
)
from test_interactive_latency import batch
from test_question_agent import agent, final, response
from token_recovery import compact_observations, select_detail_images


def target(box=None, question="Identify each apartment entry and its corridor context"):
    return {"question": question, "bbox": box or [0, 0, 500, 1000]}


def observation(question, images, **kwargs):
    return {
        "status": "ok",
        "result": {
            "status": "observed",
            "answer": 1,
            "findings": ["One visible entry; role requires adjacent room context"],
            "verdict": "uncertain",
            "citations": [
                {"image_id": i.image_id, "bbox": [1, 1, 20, 20]} for i in images
            ],
        },
        "canonical_citations": [],
    }


def orchestrator(*responses):
    value = agent(*responses)
    value.require_specialist = True
    value.allowed = {d.name for d in declarations("auto", True, orchestrated=True)}
    value.budget = AgentBudget.interactive()
    value.specialist_factory.return_value.answer.side_effect = observation
    return value


def test_answer_requires_actual_specialist_findings_not_attempt_or_review_claim():
    value = orchestrator(
        response("submit_answer", final()),
        response("inspect_regions", {"regions": [target()]}),
        response("submit_answer", final("crop-1")),
    )
    result = value.run()
    assert result["answer"]["answer"] == 2
    assert result["nemotron_calls"] == 1 and value.transport.generate.call_count == 3
    assert result["delegation"]["required"]
    assert result["delegation"]["inspected_image_ids"] == ["crop-1"]
    value.algorithm_factory.assert_not_called()


def test_uncooperative_coordinator_cannot_bypass_gate_with_json_or_tool_call():
    value = orchestrator(
        response("submit_answer", final()), response("submit_answer", final())
    )
    result = value.run()
    assert result["stop_reason"] == "specialist_required"
    assert result["answer"]["answer"] is None
    assert value.transport.generate.call_count == 2
    value.specialist_factory.assert_not_called()
    with pytest.raises(DelegationRequired):
        value.finish(final())


def test_batched_workers_are_concurrent_and_return_metadata_without_gemini_pixels():
    value = orchestrator(
        response(
            "inspect_regions", {"regions": [target(), target([500, 0, 1000, 1000])]}
        ),
        response("submit_answer", final("crop-1")),
    )
    rendezvous = threading.Barrier(2)

    def inspect(*args, **kwargs):
        rendezvous.wait(timeout=3)
        return observation(*args, **kwargs)

    value.specialist_factory.return_value.answer.side_effect = inspect
    events = []
    value.observer = lambda stage, state, **data: events.append((stage, state, data))
    result = value.run()
    assert result["nemotron_calls"] == 2
    assert result["answer"]["answer"] == 2 and value.transport.generate.call_count == 2
    assert (
        len({e[2]["inspection_id"] for e in events if e[0] == "inspect_nemotron"}) == 2
    )
    history = value.transport.generate.call_args.args[0]
    assert sum(bool(p.inline_data) for content in history for p in content.parts) == 1
    inspections = result["tool_trace"][0]["result"]["inspections"]
    assert all(row["resolution"]["downscale"] == [1, 1] for row in inspections)


def test_new_evidence_cannot_be_used_by_final_in_same_turn():
    value = orchestrator(
        batch(
            response("inspect_regions", {"regions": [target()]}),
            response("submit_answer", final("crop-1")),
        ),
        response("submit_answer", final("crop-1")),
    )
    assert value.run()["status"] == "ok"
    assert value.transport.generate.call_count == 2


def test_failed_neighbor_is_preserved_and_only_failed_tile_retries():
    value = orchestrator()
    calls = []

    def inspect(question, images, **kwargs):
        image_id = images[0].image_id
        calls.append(image_id)
        if image_id == "crop-2" and calls.count(image_id) == 1:
            raise TimeoutError("private provider details")
        return observation(question, images, **kwargs)

    value.specialist_factory.return_value.answer.side_effect = inspect
    args = {"regions": [target(), target([500, 0, 1000, 1000])]}
    first, _ = value.execute("inspect_regions", args, timeout_seconds=2)
    assert first["status"] == "partial" and first["coverage"]["unresolved_tiles"] == 1
    assert "private provider" not in json.dumps(first)
    assert len(value.finish(final("crop-1"))["uncertainties"]) == 1
    second, _ = value.execute("inspect_regions", args, timeout_seconds=2)
    assert second["status"] == "ok" and not value.unresolved_inspections
    assert calls.count("crop-1") == 1 and calls.count("crop-2") == 2
    assert value.nemotron_calls == 3


def test_failed_or_uncited_inspection_never_unlocks_answer_and_retry_is_bounded():
    value = orchestrator()
    value.specialist_factory.return_value.answer.side_effect = TimeoutError()
    args = {"regions": [target()]}
    for _ in range(3):
        value.execute("inspect_regions", args, timeout_seconds=1)
    assert value.nemotron_calls == 2
    with pytest.raises(DelegationRequired):
        value.finish(final("crop-1"))
    value.record_inspection({"status": "ok", "result": {"citations": []}})
    with pytest.raises(DelegationRequired):
        value.finish(final("crop-1"))


def test_native_tiles_cover_selected_region_without_resizing_or_gaps():
    policy = InspectionPolicy()
    box = [300, 200, 4300, 2800]
    parts = tiles(box, policy)
    assert len(parts) == 6
    assert all(p[2] - p[0] <= 1536 and p[3] - p[1] <= 1536 for p in parts)
    require_inspected_citations([{"canonical_bbox": box}], parts)
    with pytest.raises(DelegationRequired):
        require_inspected_citations([{"canonical_bbox": box}], parts[:-1])
    with pytest.raises(DelegationRequired):
        require_inspected_citations([{"canonical_bbox": [0, 0, 10, 10]}], parts)
    narrow = tiles([0, 0, 1537, 400], policy)
    assert len(narrow) == 2
    assert max(b[2] - b[0] for b in narrow) < 900
    assert narrow[0][2] - narrow[1][0] == policy.overlap


def test_large_native_region_keeps_detail_and_citation_transform():
    value = orchestrator()
    value.original = Image.new("RGB", (3000, 2000), "white")
    stream = io.BytesIO()
    value.original.save(stream, format="PNG")
    value.page.png, value.page.width, value.page.height = stream.getvalue(), 3000, 2000
    value.add_image(value.original, "overview", [0, 0, 3000, 2000])
    output, images = value.execute(
        "inspect_regions", {"regions": [target([0, 0, 700, 1000])]}, timeout_seconds=3
    )
    assert not images and output["coverage"]["returned_tiles"] == 4
    assert all(
        row["resolution"]["downscale"] == [1, 1] for row in output["inspections"]
    )
    assert all(
        max(row["resolution"]["sent_pixels"]) <= 1536 for row in output["inspections"]
    )
    value.finish(final("overview", [0, 0, 1400, 1000]))


def test_orchestrator_policy_is_in_system_instruction_not_just_user_context():
    value = GeminiTransport("fixture", "fixture", orchestrated=True)
    assert "MANDATORY SPECIALIST ROUTING" in value.instruction
    value.close()


def test_room_context_and_invalid_batches_are_validated_before_provider_access():
    regions = region_boxes(
        {"regions": [{"room_id": "R1", "question": "Entrance?"}]},
        200,
        100,
        {"R1": {"bbox": [40, 30, 80, 60]}},
    )
    assert regions[0][1] == [24, 14, 96, 76]
    value = orchestrator()
    for invalid in (
        [0, 0, 1000, 1000],
        [500, 0, 10, 500],
        [True, 0, 100, 100],
        [0, 0, float("nan"), 100],
    ):
        with pytest.raises(ValueError):
            value.execute("inspect_regions", {"regions": [target(), target(invalid)]})
    with pytest.raises(ValueError):
        value.execute(
            "inspect_regions",
            {"regions": [{"room_id": "missing", "question": "Door?"}]},
        )
    assert value.crop_count == 0
    value.specialist_factory.assert_not_called()


def test_expanded_tile_cost_dispatches_available_slots_and_reports_every_deferred_tile():
    value = orchestrator()
    value.inspection_policy = replace(
        InspectionPolicy(), tile_edge=40, overlap=8, max_calls=2
    )
    result, _ = value.execute("inspect_regions", {"regions": [target()]})
    assert value.nemotron_calls == 2 and value.crop_count == 2
    assert result["status"] == "partial"
    assert result["coverage"]["requested_tiles"] == 9
    assert result["coverage"]["returned_tiles"] == 2
    assert result["coverage"]["deferred_tiles"] == 7
    assert result["coverage"]["remaining_calls"] == 0
    assert len(value.unresolved_inspections) == 7
    assert all(row.get("canonical_bbox") for row in result["inspections"])


def test_deadline_does_not_wait_for_late_results_or_start_expired_work():
    release = threading.Event()
    started = threading.Event()
    inspect = Mock(
        side_effect=lambda *args: (started.set(), release.wait(2), {"status": "ok"})[-1]
    )
    events = []
    before = time.monotonic()
    try:
        result, count = parallel_inspections(
            [1],
            inspect,
            deadline=before + 0.15,
            seconds=30,
            on_event=lambda *args: events.append(args),
        )
        assert started.is_set() and time.monotonic() - before < 1
        assert result[0]["status"] == "unavailable" and count == 1
        event_count = len(events)
        release.set()
        time.sleep(0.02)
        assert len(events) == event_count
        result, count = parallel_inspections(
            [1], inspect, deadline=time.monotonic() - 1, seconds=30, on_event=Mock()
        )
        assert count == 0 and result[0]["status"] == "not_sent"
        assert inspect.call_count == 1
    finally:
        release.set()


def test_recovery_retains_individual_tile_evidence_and_no_extra_image_payloads():
    value = orchestrator()
    args = {"regions": [target(), target([500, 0, 1000, 1000])]}
    result, _ = value.execute("inspect_regions", args, timeout_seconds=2)
    value.tool_log = [{"name": "inspect_regions", "args": args, "result": result}]
    compact = compact_observations(value.tool_log)
    assert len(compact["observations"]) == 2
    assert all(
        row["result"]["advisory"]["citations"] for row in compact["observations"]
    )
    assert select_detail_images(value.images, value.tool_log) == []
    history = value.recovery_history(compact=True)
    assert sum(bool(p.inline_data) for content in history for p in content.parts) == 1
    assert "MANDATORY SPECIALIST" in history[0].parts[0].text


def test_api_enforces_permission_and_enables_mandatory_routing():
    value = agent()
    with TestClient(app) as client:
        doc = client.post(
            "/api/documents", files={"file": ("test.png", value.page.png, "image/png")}
        ).json()
        path = f"/api/v1/documents/{doc['document_id']}/ask"
        with (
            patch("question_agent.settings", return_value=("fixture", "fixture")),
            patch("question_agent.GeminiTransport") as transport,
            patch("question_agent.DrawingAgent") as factory,
        ):
            denied = client.post(
                path, json={"question": "Doors?", "allow_gemini": True}
            ).json()
            assert denied["error_code"] == "nemotron_inspection_permission_required"
            transport.assert_not_called()
            factory.return_value.run.return_value = {
                "answer": None,
                "canonical_citations": [],
            }
            client.post(
                path,
                json={
                    "question": "Doors?",
                    "allow_gemini": True,
                    "allow_nemotron": True,
                },
            )
            assert factory.call_args.kwargs["require_specialist"]


def test_explicit_direct_benchmark_still_excludes_specialist_tools():
    assert [d.name for d in declarations("direct", True, orchestrated=True)] == [
        "submit_answer"
    ]
