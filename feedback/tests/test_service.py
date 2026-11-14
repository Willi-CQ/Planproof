import json

import pytest
from PIL import Image

from planproof_feedback.benchmark import write_json
from planproof_feedback.images import load_images, sha256
from planproof_feedback.regions import load_region
from planproof_feedback.service import FeedbackService, parse_object
from planproof_feedback.tools import FeedbackTools, TOOL_DEFINITIONS


IMAGE_CONFIG = {"max_edge": 100, "max_pages": 2, "pdf_dpi": 72}


@pytest.fixture
def images(tmp_path):
    path = tmp_path / "drawing.png"
    Image.new("RGB", (200, 100), "white").save(path)
    return load_images([path], IMAGE_CONFIG)


def answer(value=3):
    return {
        "status": "answered",
        "answer": value,
        "unit": "count",
        "explanation": "visible items",
        "citations": [{"image_id": "input-0-p0", "bbox": [1, 2, 30, 40]}],
    }


def review(verdict="supported"):
    return {
        "verdict": verdict,
        "issues": [],
        "suggested_answer": None,
        "unit": "count",
        "explanation": "visible support",
        "citations": answer()["citations"],
    }


class Backend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        return {
            "content": json.dumps(next(self.outputs)),
            "usage": {"total_tokens": 3},
            "model": "simulated-nemotron",
            "latency_seconds": 0.01,
        }


def test_feedback_and_agent_critique_are_separate_routes(images):
    feedback, critic = Backend([review()]), Backend([review("revise")])
    service = FeedbackService(feedback, critic)
    assert (
        service.review("How many doors?", images, candidate=3)["result"]["verdict"] == "supported"
    )
    assert (
        service.critique_draft("How many doors?", images, draft=3)["result"]["verdict"] == "revise"
    )
    assert len(feedback.messages) == len(critic.messages) == 1
    assert "correlated feedback" in critic.messages[0][0]["content"]


def test_single_revision_is_bounded_and_keeps_all_stages(images):
    primary = Backend([answer(3), answer(4)])
    critic = Backend([review("revise"), review("revise")])
    result = FeedbackService(primary, critic).answer_with_self_feedback("Doors?", images)
    assert len(result["calls"]) == 4 and result["final"]["result"]["answer"] == 4
    assert result["stop_reason"] == "revision_limit"
    revised_user = json.loads(primary.messages[1][1]["content"][0]["text"])
    assert revised_user["previous_draft"]["answer"] == 3
    assert revised_user["advisory_feedback"]["verdict"] == "revise"


def test_supported_draft_stops_without_revision(images):
    primary, critic = Backend([answer()]), Backend([review()])
    result = FeedbackService(primary, critic).answer_with_self_feedback("Doors?", images)
    assert len(result["calls"]) == 2 and result["stop_reason"] == "supported"


@pytest.mark.parametrize(
    "edit",
    [
        {"citations": []},
        {"citations": [{"image_id": "invented", "bbox": [1, 2, 3, 4]}]},
        {"citations": [{"image_id": "input-0-p0", "bbox": [0, 0, 101, 50]}]},
        {"answer": True},
        {"status": "abstain"},
    ],
)
def test_invalid_answers_do_not_become_success(images, edit):
    backend = Backend([{**answer(), **edit}])
    result = FeedbackService(backend, backend).answer("Doors?", images)
    assert result["status"] == "invalid_output" and result["result"] is None


def test_abstention_and_strict_finite_json(images):
    backend = Backend([{**answer(), "status": "abstain", "answer": None, "citations": []}])
    assert (
        FeedbackService(backend, backend).answer("Hidden floor volume?", images)["status"] == "ok"
    )
    assert parse_object('```json\n{"a": 2}\n```') == {"a": 2}
    for text in ('{"a": NaN}', 'prefix {"a": 1}', "[1, 2]", '{"a":Infinity}'):
        with pytest.raises(ValueError):
            parse_object(text)


def test_region_hash_and_citation_round_trip(tmp_path):
    path = tmp_path / "clean.png"
    Image.new("RGB", (200, 100), "white").save(path)
    region = {
        "clean_crop": "clean.png",
        "sha256": sha256(path),
        "page_index": 2,
        "bbox": [100, 200, 300, 300],
        "crop_to_canonical": [[1, 0, 100], [0, 1, 200], [0, 0, 1]],
        "observation_ids": ["door-1"],
    }
    write_json(tmp_path / "region.json", region)
    images, _ = load_region(tmp_path / "region.json", IMAGE_CONFIG)
    backend = Backend([review()])
    result = FeedbackService(backend, backend).review("Doors?", images, candidate=3)
    assert result["canonical_citations"][0]["canonical_bbox"] == [102, 204, 160, 280]
    tools = FeedbackTools(
        FeedbackService(Backend([review()]), Backend([review()])),
        IMAGE_CONFIG,
        {"door-detail": tmp_path / "region.json"},
    )
    arguments = {"question": "Doors?", "candidate": "three", "region_id": "door-detail"}
    assert len(TOOL_DEFINITIONS) == 2
    assert tools.call("nemotron_visual_feedback", arguments)["role"] == "feedback"
    assert (
        tools.call(
            "nemotron_self_feedback",
            {"question": "Doors?", "draft": "three", "region_id": "door-detail"},
        )["role"]
        == "self_feedback"
    )
    with pytest.raises(ValueError, match="not registered"):
        tools.call("nemotron_visual_feedback", {**arguments, "region_id": "../../secret"})
    with pytest.raises(ValueError, match="schema"):
        tools.call("nemotron_visual_feedback", {**arguments, "path": "../../secret"})
    Image.new("RGB", (200, 100), "black").save(path)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_region(tmp_path / "region.json", IMAGE_CONFIG)


def test_real_pdf_render_and_page_limit(tmp_path):
    path = tmp_path / "two-pages.pdf"
    image = Image.new("RGB", (200, 300), "white")
    image.save(path, save_all=True, append_images=[image])
    result = load_images([path], IMAGE_CONFIG)
    assert len(result) == 2 and result[1].metadata["page_index"] == 1
    with pytest.raises(ValueError, match="never silently dropped"):
        load_images([path], {**IMAGE_CONFIG, "max_pages": 1})
