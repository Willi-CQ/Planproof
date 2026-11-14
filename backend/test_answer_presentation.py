"""Presentation contracts use scripted evidence, not accuracy benchmarks."""

from answer_presentation import attach_presentation
from test_question_agent import agent, final


def test_briefing_remaps_references_after_host_adds_measurement_sources():
    a = agent()
    source = {"image_id": "overview", "bbox": [30.0, 10.0, 60.0, 30.0]}
    a.measurements["M1"] = {
        "value": 12.0,
        "unit": "m2",
        "approximate": True,
        "citations": [source],
    }
    result = a.finish(
        {
            **final(),
            "answer": 12.0,
            "unit": "m2",
            "measurement_id": "M1",
            "briefing": [
                {
                    "section": "answer",
                    "text": "The selected room is approximately 12 square metres.",
                    "citations": [1],
                }
            ],
            "citation_details": [
                {"citation": 1, "kind": "detection", "label": "Room boundary"}
            ],
        }
    )
    assert result["answer"] == 12
    assert result["citations"][0] == source
    assert result["briefing"][0]["citations"] == [2]
    assert result["citation_details"][0]["citation"] == 2
    assert "Approximate" in result["uncertainties"][0]


def test_bad_optional_presentation_never_discards_a_valid_answer():
    a = agent()
    result = a.finish({**final(), "briefing": "malformed", "citation_details": [None]})
    assert result["answer"] == 2
    assert result["briefing"] == result["citation_details"] == []


def test_unknown_or_boolean_references_and_uncited_claims_are_omitted():
    citation = {"image_id": "overview", "bbox": [1, 1, 20, 20]}
    result = attach_presentation(
        {"citations": [citation]},
        [citation],
        {
            "briefing": [
                {"section": "answer", "text": "Unknown source", "citations": [2]},
                {
                    "section": "answer",
                    "text": "Invalid boolean index",
                    "citations": [True],
                },
                {"section": "basis", "text": "Uncited claim", "citations": []},
                {
                    "section": "limitation",
                    "text": "Other pages are not supplied.",
                    "citations": [],
                },
            ],
            "citation_details": [{"citation": 9, "kind": "text", "label": "Missing"}],
        },
    )
    assert len(result["briefing"]) == 1
    assert result["briefing"][0]["section"] == "limitation"
    assert result["citation_details"] == []


def test_legacy_final_answer_is_still_accepted_with_identical_citations():
    result = agent().finish(final())
    assert result["answer"] == 2
    assert result["citations"] == final()["citations"]
    assert result["briefing"] == []


def test_running_crop_event_precedes_execution_and_uses_native_frame():
    a = agent()
    events = []
    a.observer = lambda stage, state, **data: events.append((stage, state, data))
    a.execute("crop_region", {"bbox": [100, 200, 500, 600]})
    stage, state, payload = events[0]
    assert (stage, state) == ("crop_region", "running")
    assert payload["images"][0]["canonical_bbox"] == [20, 20, 100, 60]
    assert payload["images"][0]["image_id"] == "requested-region"
    assert events[-1][1] == "completed"
