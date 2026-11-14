import copy
import json
from types import SimpleNamespace

import pytest

from token_recovery import compact_observations, select_detail_images


def crop(image_id, bbox, size=(100, 100), **metadata):
    return SimpleNamespace(
        image_id=image_id,
        metadata={
            "canonical_bbox": bbox,
            "source_render_size": size,
            **metadata,
        },
    )


def specialist(image_id="left", status="ok", name="inspect_nemotron"):
    return {
        "name": name,
        "args": {
            "image_ids": [image_id],
            "question": "What is inside this crop?",
            "candidate": "DRAFT MUST NOT SURVIVE",
        },
        "result": {
            "status": status,
            "image_ids": [image_id],
            "scope": "Selected crop only; whole-page coverage unknown.",
            "advisory": {
                "answer": "One candidate",
                "status": "uncertain",
                "findings": ["A visible candidate has uncertain class."],
                "verdict": "uncertain",
                "citations": [{"image_id": image_id, "bbox": [1, 2, 3, 4]}],
                "citation_coordinates": "sent-image pixels",
            },
            "canonical_citations": [
                {"image_id": image_id, "canonical_bbox": [10, 20, 30, 40]}
            ],
        },
    }


@pytest.mark.parametrize("tool_name", ["inspect_nemotron", "review_answer"])
def test_compaction_keeps_scope_citations_and_uncertainty_but_never_drafts(tool_name):
    log = [
        {"name": "reflect", "result": {"draft": "UNVERIFIED REFLECTION"}},
        specialist(name=tool_name),
        {"name": "submit_answer", "result": {"answer": "UNVERIFIED FINAL"}},
    ]
    original = copy.deepcopy(log)
    result = compact_observations(log)
    serialized = json.dumps(result)
    assert result["coverage"] == "unknown"
    assert result["omitted_tool_records"] == 2
    retained = result["observations"][0]
    assert retained["result"] == original[1]["result"]
    assert retained["request_scope"]["question"] == log[1]["args"]["question"]
    assert "DRAFT" not in serialized
    assert "UNVERIFIED" not in serialized
    assert log == original


def test_production_specialist_result_is_retained_with_canonical_citations_atomically():
    entry = specialist()
    nested = entry["result"].pop("advisory")
    nested["findings"] = ["A visible candidate has uncertain class."]
    nested["verdict"] = "uncertain"
    entry["result"]["result"] = nested
    entry["result"]["completion"] = {"usage": {"total_tokens": 123}}
    result = compact_observations([entry])
    retained = result["observations"][0]["result"]
    assert retained["result"] == nested
    assert retained["canonical_citations"] == entry["result"]["canonical_citations"]
    assert "completion" not in retained
    entry["result"]["result"]["citations"] *= 500
    assert compact_observations([entry], max_chars=1400)["observations"] == []


def test_compaction_bounds_unicode_and_marks_algorithm_truncation():
    log = [
        specialist(),
        {
            "name": "get_algorithms",
            "result": {
                "mode": "ml_assisted; not independent confirmation",
                "observations": [
                    {
                        "id": str(index),
                        "class": "door",
                        "bbox": [1, 2, 3, 4],
                        "source_family": "ml",
                        "note": "ä" * 20,
                    }
                    for index in range(80)
                ],
                "omitted_observations": 4,
                "capabilities": {"complete_coverage": False},
            },
        },
    ]
    original = copy.deepcopy(log)
    result = compact_observations(log, max_chars=2700)
    assert len(json.dumps(result, ensure_ascii=True)) <= 2700
    assert result["truncated_tool_records"] == 1
    algorithm = next(
        item for item in result["observations"] if item["tool"] == "get_algorithms"
    )
    assert algorithm["result"]["omitted_observations"] == 4
    assert (
        algorithm["additional_omissions"]["observations"]
        + len(algorithm["result"]["observations"])
        == 80
    )
    assert algorithm["result"]["capabilities"]["complete_coverage"] is False
    assert log == original


def test_oversized_specialist_record_is_omitted_atomically():
    record = specialist()
    record["result"]["advisory"]["citations"] *= 500
    result = compact_observations([record], max_chars=1400)
    assert result["observations"] == []
    assert result["omitted_tool_records"] == 1
    assert "One candidate" not in json.dumps(result)


def test_crop_metadata_preserves_registered_coordinate_frame():
    metadata = {
        "image_id": "crop-1",
        "canonical_bbox": [20, 30, 50, 60],
        "sent_size": [300, 300],
        "source_render_size": [1000, 1000],
        "coordinate_system": "original page pixels",
    }
    result = compact_observations(
        [
            {"name": "crop_region", "result": {"image": metadata, "cached": True}},
        ]
    )
    assert result["observations"][0]["result"]["image"] == metadata
    assert result["observations"][0]["result"]["cached"] is True


def test_compaction_invalid_data_and_budget():
    invalid = specialist()
    invalid["result"]["advisory"]["answer"] = float("nan")
    result = compact_observations([invalid, None])
    assert result["omitted_tool_records"] == 2
    json.dumps(result, allow_nan=False)
    for budget in (0, 10, -1, True, 1.5):
        with pytest.raises(ValueError):
            compact_observations([], max_chars=budget)


def test_detail_selection_prefers_uninspected_and_keeps_original_ids():
    images = {
        "overview": crop("overview", [0, 0, 100, 100]),
        "first": crop("left", [0, 0, 40, 100]),
        "second": crop("right", [60, 0, 100, 100]),
        "third": crop("center", [30, 0, 70, 100]),
    }
    assert select_detail_images(images, [specialist("left")]) == ["right"]
    assert select_detail_images(images, [specialist("left", "unavailable")]) == [
        "left",
        "right",
    ]
    assert select_detail_images(images, [specialist("left", "invalid_output")]) == [
        "left",
        "right",
    ]
    assert (
        select_detail_images(
            [images["overview"], images["first"]], [specialist("left")]
        )
        == []
    )


@pytest.mark.parametrize("offset,expected", [(40, ["a", "b"]), (39, ["a"])])
def test_overlap_threshold_is_twenty_percent_of_smaller_area(offset, expected):
    images = [
        crop("overview", [0, 0, 100, 100]),
        crop("a", [0, 0, 50, 100]),
        crop("b", [offset, 0, offset + 50, 100]),
    ]
    assert select_detail_images(images, []) == expected


def test_overview_large_invalid_and_mismatched_frames_are_excluded():
    images = [
        crop("overview", [0, 0, 100, 100], page_index=0),
        crop("too-large", [0, 0, 76, 100]),
        crop("negative", [-1, 0, 10, 10]),
        crop("nan", [0, 0, float("nan"), 20]),
        crop("outside", [90, 0, 101, 20]),
        crop("bool", [False, 0, 20, 20]),
        crop("valid", [0, 0, 50, 100], page_index=0),
        crop("other-page", [50, 0, 100, 100], page_index=1),
    ]
    assert select_detail_images(images, []) == ["valid"]
    assert select_detail_images(
        [crop("overview", [0, 0, 100, 100]), crop("limit", [0, 0, 75, 100])], []
    ) == ["limit"]


def test_detail_selection_limit_and_duplicates():
    images = [
        crop("overview", [0, 0, 100, 100]),
        crop("a", [0, 0, 50, 100]),
        crop("a", [50, 0, 100, 100]),
    ]
    assert select_detail_images(images, [], max_count=0) == []
    assert select_detail_images(images, []) == ["a"]
    with pytest.raises(ValueError):
        select_detail_images(images, [], max_count=-1)


def test_production_offset_crop_uses_overview_page_frame():
    from test_question_agent import agent

    coordinator = agent()
    coordinator.crop([100, 10, 180, 90])
    registered = coordinator.images["crop-1"]
    assert registered.metadata["source_render_size"] == [80, 80]
    assert registered.metadata["canonical_bbox"] == [100, 10, 180, 90]
    original = copy.deepcopy(registered.metadata)
    assert select_detail_images(coordinator.images, []) == ["crop-1"]
    assert registered.metadata == original
    assert select_detail_images([registered], []) == []
