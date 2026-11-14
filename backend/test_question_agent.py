"""Scripted provider tests exercise host behavior; these are not accuracy scores."""

import io
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from google.genai import types
from PIL import Image

from app import app
from question_agent import AgentRequest, DrawingAgent, GeminiTransport, declarations
import httpx


def response(name, args, *, signature=None):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                finish_reason="STOP",
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name=name, args=args, id="call-1"
                            ),
                            thought_signature=signature,
                        )
                    ],
                ),
            )
        ]
    )


def final(image="overview", box=None):
    return {
        "status": "answered",
        "answer": 2,
        "unit": "count",
        "explanation": "Two visible symbols.",
        "citations": [{"image_id": image, "bbox": box or [1, 1, 20, 20]}],
        "coverage": "Entire supplied page inspected.",
        "uncertainties": [],
    }


def agent(*responses, mode="auto", allow=True):
    stream = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(stream, format="PNG")
    page = SimpleNamespace(
        png=stream.getvalue(), width=200, height=100, raster_sha256="a" * 64, labels={}
    )
    transport = Mock(model="scripted-fixture")
    transport.generate.side_effect = list(responses)
    algorithm = Mock()
    algorithm.return_value.catalog = {
        "rooms": [
            {"id": "long-registered-room", "bbox": [10, 10, 80, 80], "neighbor_ids": []}
        ],
        "observations": [{"id": "observation-1", "class": "window"}],
        "openings": [],
        "wall_roles": {},
        "warnings": [],
    }
    specialist = Mock()
    specialist.return_value.answer.return_value = {
        "status": "ok",
        "result": {"answer": 1},
        "canonical_citations": [],
    }
    specialist.return_value.review.return_value = {
        "status": "ok",
        "result": {"verdict": "supported"},
        "canonical_citations": [],
    }
    return DrawingAgent(
        page,
        AgentRequest(
            question="How many bathtubs?",
            mode=mode,
            allow_gemini=True,
            allow_nemotron=allow,
        ),
        transport,
        algorithm,
        specialist,
    )


def test_direct_answer_never_initializes_algorithms_or_specialist():
    a = agent(response("submit_answer", final()))
    result = a.run()
    assert result["status"] == "ok" and not result["algorithm_used"]
    a.algorithm_factory.assert_not_called()
    a.specialist_factory.assert_not_called()


def test_transport_retries_transient_failure_once_only():
    transport = GeminiTransport.__new__(GeminiTransport)
    transport._generate = Mock(
        side_effect=[httpx.ConnectError("private network detail"), "success"]
    )
    with patch("question_agent.time.sleep"):
        assert transport.generate([], []) == "success"
    assert transport.last_attempts == 2
    transport._generate = Mock(side_effect=httpx.ConnectError("private network detail"))
    with patch("question_agent.time.sleep"), pytest.raises(httpx.ConnectError):
        transport.generate([], [])
    assert transport._generate.call_count == 2


def test_boolean_cannot_be_promoted_to_numeric_answer():
    a = agent()
    with pytest.raises(ValueError):
        a.finish({**final(), "answer": True})


def test_uncooperative_model_stops_at_turn_budget():
    a = agent(*[response("crop_region", {"bbox": [0, 0, 500, 500]}) for _ in range(8)])
    result = a.run()
    assert result["answer"]["status"] == "abstain"
    assert a.transport.generate.call_count == 8 and a.crop_count == 1


def test_specialist_call_budget_is_enforced():
    a = agent()
    for i in range(3):
        a.execute(
            "inspect_nemotron",
            {"image_ids": ["overview"], "question": f"Inspect area {i}"},
        )
    with pytest.raises(ValueError):
        a.execute(
            "inspect_nemotron", {"image_ids": ["overview"], "question": "Another"}
        )
    assert a.specialist_factory.return_value.answer.call_count == 3


def test_visual_feedback_independent_of_algorithms_and_signatures_preserved():
    first = response(
        "crop_region", {"bbox": [500, 0, 1000, 1000]}, signature=b"opaque-signature"
    )
    a = agent(
        first,
        response(
            "inspect_nemotron",
            {"image_ids": ["crop-1"], "question": "Count tubs in this crop only"},
        ),
        response(
            "review_answer",
            {
                "image_ids": ["overview", "crop-1"],
                "candidate": "Two tubs; check duplicates",
            },
        ),
        response("submit_answer", final("crop-1")),
        mode="visual_only",
    )
    result = a.run()
    assert result["status"] == "ok" and result["nemotron_calls"] == 2
    assert result["canonical_citations"][0]["canonical_bbox"] == [101, 1, 120, 20]
    a.algorithm_factory.assert_not_called()
    history = a.transport.generate.call_args.args[0]
    assert history[1] is first.candidates[0].content
    assert history[1].parts[0].thought_signature == b"opaque-signature"
    assert history[2].parts[0].function_response.id == "call-1"


def test_algorithms_are_search_aids_and_filtered_not_bathtub_detector():
    a = agent(
        response("get_algorithms", {"classes": ["room"]}),
        response("crop_room", {"room_id": "R1"}),
        response("submit_answer", final("crop-1")),
    )
    a.algorithm_factory.return_value.catalog["observations"].append(
        {"id": "unclassified-pixel-cue", "bbox": [1, 2, 3, 4]}
    )
    result = a.run()
    catalog = result["tool_trace"][0]["result"]
    assert result["algorithm_used"]
    assert catalog["rooms"][0]["id"] == "R1" and not catalog["observations"]
    assert "bathtub" in catalog["capabilities"]["not_detected"]
    assert a.algorithm_factory.call_count == 1


@pytest.mark.parametrize(
    "name,args",
    [
        ("get_algorithms", {"classes": ["bathtub"]}),
        ("crop_room", {"room_id": "../../secret"}),
        ("crop_region", {"bbox": [-1, 0, 30, 30]}),
        ("crop_region", {"bbox": [0, 0, float("nan"), 30]}),
        ("inspect_nemotron", {"image_ids": ["unknown"], "question": "Tubs?"}),
        ("execute_shell", {"command": "whoami"}),
    ],
)
def test_invalid_tools_fail_closed_without_factories(name, args):
    a = agent()
    with pytest.raises((ValueError, TypeError)):
        a.execute(name, args)
    a.algorithm_factory.assert_not_called()
    a.specialist_factory.assert_not_called()


def test_permission_and_mode_are_host_enforced():
    a = agent(mode="visual_only", allow=False)
    for name, args in [
        ("get_algorithms", {"classes": ["room"]}),
        ("inspect_nemotron", {"image_ids": ["overview"], "question": "tubs"}),
    ]:
        with pytest.raises(ValueError):
            a.execute(name, args)
    assert [t.name for t in declarations("direct", True)] == ["submit_answer"]


def test_crop_cache_and_budget():
    a = agent()
    args = {"bbox": [0, 0, 500, 500]}
    first = a.execute("crop_region", args)[0]
    second = a.execute("crop_region", args)[0]
    assert first["image"] == second["image"] and a.crop_count == 1
    for i in range(1, 6):
        a.execute("crop_region", {"bbox": [i * 10, 0, 600, 600]})
    with pytest.raises(ValueError):
        a.execute("crop_region", {"bbox": [100, 0, 600, 600]})


def test_algorithm_evidence_is_incremental_and_provenance_stays_host_side():
    a = agent()
    first, _ = a.execute("get_algorithms", {"classes": ["room", "window"]})
    second, _ = a.execute("get_algorithms", {"classes": ["room", "window", "door"]})
    assert first["rooms"] and first["observations"][0]["id"] == "O1"
    assert not second["rooms"] and not second["observations"]
    assert second["already_supplied_classes"] == ["room", "window"]
    assert a.algorithm_registry["R1"] == "long-registered-room"
    assert a.algorithm_registry["O1"] == "observation-1"


def test_invalid_citation_can_be_repaired_without_publishing_invalid_answer():
    a = agent(
        response("submit_answer", final("made-up")), response("submit_answer", final())
    )
    result = a.run()
    assert result["status"] == "ok"
    assert result["tool_trace"][0]["result"]["status"] == "unavailable"
    assert result["canonical_citations"][0]["image_id"] == "overview"


def test_algorithm_failure_allows_visual_fallback():
    a = agent(
        response("get_algorithms", {"classes": ["room"]}),
        response("crop_region", {"bbox": [0, 0, 500, 500]}),
        response("submit_answer", final("crop-1")),
    )
    a.algorithm_factory.side_effect = RuntimeError("private path or key")
    result = a.run()
    assert result["status"] == "ok" and not result["algorithm_used"]
    assert "private path" not in str(result)


def test_provider_failure_is_sanitized_and_abstains():
    a = agent()
    a.transport.generate.side_effect = RuntimeError("secret token")
    result = a.run()
    assert (
        result["answer"]["answer"] is None and result["stop_reason"] == "provider_error"
    )
    assert "secret token" not in str(result)


def test_duplicate_specialist_request_cached_and_review_bounded():
    call = response(
        "inspect_nemotron", {"image_ids": ["overview"], "question": "How many tubs?"}
    )
    a = agent(call, call, response("submit_answer", final()))
    assert a.run()["nemotron_calls"] == 1
    a.execute("review_answer", {"image_ids": ["overview"], "candidate": "2"})
    with pytest.raises(ValueError):
        a.execute("review_answer", {"image_ids": ["overview"], "candidate": "3"})


def test_api_missing_configuration_and_permission_do_not_run_models():
    a = agent()
    with TestClient(app) as client:
        doc = client.post(
            "/api/documents", files={"file": ("test.png", a.page.png, "image/png")}
        ).json()
        path = f"/api/v1/documents/{doc['document_id']}/ask"
        with (
            patch("question_agent.settings", side_effect=ValueError("private config")),
            patch("agent_pipeline.get_job") as factory,
        ):
            assert (
                client.post(path, json={"question": "Floors?"}).json()["status"]
                == "not_authorized"
            )
            result = client.post(
                path, json={"question": "Floors?", "allow_gemini": True}
            )
            assert result.json()["status"] == "not_configured"
            assert "private config" not in result.text
            factory.assert_not_called()
        assert client.post(path, json={"question": "  "}).status_code == 422
