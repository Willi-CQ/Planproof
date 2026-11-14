import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from planproof_feedback.artifacts import write_json
from planproof_feedback.guided import answer_guided, validate_decision
from planproof_feedback.images import sha256
from planproof_feedback.pipeline import worker_env
from planproof_feedback.regions import load_region
from planproof_feedback.service import FeedbackService


CONFIG = {"max_edge": 100, "max_pages": 12, "pdf_dpi": 144}
CATALOG = {"rooms": [{"id": "region:one"}], "observations": [{"id": "ml:door"}], "openings": []}


def decision(**changes):
    return {
        "action": "inspect_rooms",
        "region_ids": ["region:one"],
        "evidence_ids": ["ml:door"],
        "explanation": "Inspect the attached opening.",
        **changes,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"region_ids": ["../../.env"]},
        {"region_ids": ["region:one"] * 4},
        {"evidence_ids": ["unknown"]},
        {"action": "execute_shell"},
        {"action": "answer_overview"},
        {"region_ids": []},
        {"explanation": ""},
    ],
)
def test_router_cannot_invent_tools_ids_or_exceed_budget(changes):
    with pytest.raises(ValueError):
        validate_decision(decision(**changes), CATALOG)


def fake_job(tmp_path):
    image = tmp_path / "page.png"
    Image.new("RGB", (40, 30), "white").save(image)
    crop = tmp_path / "clean.png"
    Image.new("RGB", (10, 8), "white").save(crop)
    write_json(
        tmp_path / "agent-context.json",
        {"schema_version": "planproof-agent-context-v1", "observations": []},
    )
    write_json(
        tmp_path / "region.json",
        {
            "clean_crop": "clean.png",
            "sha256": sha256(crop),
            "crop_to_canonical": [[1, 0, 5], [0, 1, 7], [0, 0, 1]],
            "page_index": 0,
            "bbox": [5, 7, 15, 15],
            "observation_ids": ["ml:door"],
            "context_path": "agent-context.json",
        },
    )
    return SimpleNamespace(
        image=image,
        catalog=CATALOG,
        directory=tmp_path,
        crop=Mock(return_value=tmp_path / "region.json"),
    )


class Backend:
    def __init__(self, replies):
        self.replies, self.messages = list(replies), []

    def complete(self, messages):
        self.messages.append(messages)
        return {"content": json.dumps(self.replies.pop(0)), "usage": {"total_tokens": 10}}


def test_selected_registered_crop_and_citations_round_trip(tmp_path):
    backend = Backend(
        [
            decision(),
            {
                "status": "answered",
                "answer": 1,
                "unit": "count",
                "explanation": "A door is visible in this crop.",
                "citations": [{"image_id": "room-region-one", "bbox": [1, 2, 4, 6]}],
            },
        ]
    )
    job = fake_job(tmp_path)
    result = answer_guided(FeedbackService(backend, backend), job, "How many doors?", CONFIG)
    assert len(backend.messages) == 2
    job.crop.assert_called_once_with("region:one")
    assert result["final"]["canonical_citations"][0]["canonical_bbox"] == [6, 9, 9, 13]
    assert result["usage"]["total_tokens"] == 20
    assert "algorithm_evidence" in backend.messages[1][1]["content"][0]["text"]


def test_invalid_route_cannot_trigger_crop_or_answer(tmp_path):
    backend = Backend([decision(region_ids=["not-registered"])])
    job = fake_job(tmp_path)
    with pytest.raises(ValueError, match="registered"):
        answer_guided(FeedbackService(backend, backend), job, "Question?", CONFIG)
    assert len(backend.messages) == 1
    job.crop.assert_not_called()


def test_router_abstention_has_one_call_and_no_invented_answer(tmp_path):
    backend = Backend([decision(action="abstain", region_ids=[])])
    result = answer_guided(
        FeedbackService(backend, backend), fake_job(tmp_path), "Question?", CONFIG
    )
    assert len(backend.messages) == 1
    assert result["final"]["result"]["answer"] is None


def test_page_wide_selection_exceeding_budget_uses_overview_not_partial_count(tmp_path):
    job = fake_job(tmp_path)
    job.catalog = {"rooms": [{"id": str(i)} for i in range(5)], "observations": [], "openings": []}
    backend = Backend(
        [
            decision(region_ids=[str(i) for i in range(5)], evidence_ids=[]),
            {
                "status": "answered",
                "answer": 1,
                "unit": "count",
                "explanation": "Visible on the overview.",
                "citations": [{"image_id": "input-0-p0", "bbox": [1, 1, 4, 4]}],
            },
        ]
    )
    result = answer_guided(
        FeedbackService(backend, backend), job, "How many rooms on the whole page?", CONFIG
    )
    job.crop.assert_not_called()
    assert result["decision"]["host_fallback"] == "crop_budget_exceeded"
    assert result["decision"]["region_ids"] == []
    assert result["calls"][0]["raw_result"]["region_ids"] == [str(i) for i in range(5)]
    assert len(backend.messages) == 2


def test_invalid_answer_citation_does_not_become_success(tmp_path):
    backend = Backend(
        [
            decision(action="answer_overview", region_ids=[]),
            {
                "status": "answered",
                "answer": 4,
                "unit": "count",
                "explanation": "Guess",
                "citations": [{"image_id": "unsupplied-image", "bbox": [0, 0, 1, 1]}],
            },
        ]
    )
    result = answer_guided(
        FeedbackService(backend, backend), fake_job(tmp_path), "Question?", CONFIG
    )
    assert result["final"]["status"] == "invalid_output"
    assert result["final"]["result"] is None


def test_room_context_rejects_changed_sidecar(tmp_path):
    job = fake_job(tmp_path)
    region = json.loads(job.crop.return_value.read_text())
    region["context_sha256"] = "0" * 64
    write_json(tmp_path / "region.json", region)
    with pytest.raises(ValueError, match="context hash"):
        load_region(tmp_path / "region.json", CONFIG)


def test_algorithm_workers_do_not_receive_model_credentials(monkeypatch):
    monkeypatch.setenv("NVIDIA_NEMOTRON_KEY", "private")
    monkeypatch.setenv("GEMINI_API_KEY", "private")
    monkeypatch.setenv("OTHER_SECRET", "private")
    env = worker_env("ml")
    assert not any(
        name in env for name in ("NVIDIA_NEMOTRON_KEY", "GEMINI_API_KEY", "OTHER_SECRET")
    )
    assert env["PYTHONPATH"].endswith(str(Path("ml/src")))
