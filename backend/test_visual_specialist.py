import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from planproof_feedback.images import ImageInput
from visual_specialist import VisualSpecialist


def setup(result):
    backend = Mock()
    backend.complete.return_value = {
        "content": json.dumps(result),
        "usage": {"total_tokens": 20},
    }
    image = ImageInput(
        "crop-1",
        "data:image/jpeg;base64,AAAA",
        {
            "image_id": "crop-1",
            "sent_size": [200, 100],
            "page_index": 0,
            "sent_to_source_render": [2, 2],
            "crop_to_canonical": [[1, 0, 300], [0, 1, 400], [0, 0, 1]],
        },
    )
    specialist = VisualSpecialist(SimpleNamespace(backends={"feedback": backend}))
    return specialist, backend, image


def observation():
    return {
        "status": "observed",
        "answer": "One rectangular opening; class ambiguous",
        "findings": ["Rectangle in a wall"],
        "verdict": "uncertain",
        "citations": [{"image_id": "crop-1", "bbox": [100, 200, 500, 800]}],
    }


def test_local_normalized_coordinates_map_through_resize_and_page_offset():
    specialist, backend, image = setup(observation())
    result = specialist.answer("What is visible?", [image])
    assert result["status"] == "ok"
    assert result["result"]["citations"][0]["bbox"] == [20, 20, 100, 80]
    assert result["canonical_citations"][0]["canonical_bbox"] == [340, 440, 500, 560]
    text = backend.complete.call_args.args[0][1]["content"][0]["text"]
    assert "crop_to_canonical" not in text and "400" not in text


@pytest.mark.parametrize(
    "change",
    [
        {"citations": [{"image_id": "unknown", "bbox": [0, 0, 100, 100]}]},
        {"citations": [{"image_id": "crop-1", "bbox": [0, 0, 1001, 100]}]},
        {"citations": []},
        {"answer": [1, 2]},
        {"findings": "not a list"},
    ],
)
def test_invalid_specialist_evidence_cannot_reach_agent_as_valid(change):
    specialist, _, image = setup({**observation(), **change})
    result = specialist.answer("Inspect", [image])
    assert result["status"] == "invalid_output" and result["result"] is None


def test_review_sends_draft_but_keeps_local_frame():
    specialist, backend, image = setup(observation())
    specialist.review("Windows?", [image], candidate="2 windows")
    text = json.loads(backend.complete.call_args.args[0][1]["content"][0]["text"])
    assert text["draft_to_review"] == "2 windows"
