"""UI score semantics on generated report records; no real data in source."""

import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "build_evidence_lab.py"
SPEC = importlib.util.spec_from_file_location("evidence_lab_builder", SOURCE)
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def objects(tp, fp, fn):
    return {
        "status": "scored",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "truth": tp + fn,
        "predicted": tp + fp,
    }


def test_missing_coverage_and_model_support_are_not_zero():
    report = {
        "summary": {"cad": {}, "main": {}, "deterministic": {}},
        "results": [
            {
                "case_id": "one",
                "method": "cad",
                "status": "scored",
                "metrics": {
                    "wall": {"status": "scored", "mask_iou": 0},
                    "stairs": {"status": "not_annotated"},
                },
            }
        ],
    }
    scores = lab.case_scores(
        "one", [{"name": "cad", "available_classes": ["stairs"]}], report
    )
    assert scores["cad"]["wall"]["status"] == "model_unsupported"
    assert scores["cad"]["stairs"]["status"] == "not_annotated"
    assert scores["main"]["door"]["status"] == "not_run"
    assert scores["cad"]["wall"]["value"] is None


def test_cad_aggregate_uses_only_shared_scored_cases():
    cases = []
    for identity, supported in (("one", True), ("two", False)):
        cases.append(
            {
                "id": identity,
                "scores": {
                    "cad": {
                        "door": lab.score_entry(
                            objects(1, 1, 0),
                            "door",
                            [identity],
                            "scored" if supported else "not_run",
                        )
                    },
                    "deterministic": {
                        "door": lab.score_entry(objects(1, 0, 0), "door", [identity])
                    },
                },
            }
        )
    result = lab.score_comparisons(cases, ["cad", "deterministic"])["cad"]
    assert result["door"]["case_ids"] == ["one"]
    assert result["door"]["deterministic"]["tp"] == 1
    assert result["door"]["ml"]["value"] == pytest.approx(2 / 3)
    assert result["wall"]["count"] == 0
    assert result["wall"]["ml"]["value"] is None


def test_pooled_f1_is_not_average_of_page_f1():
    entries = [
        lab.score_entry(objects(1, 0, 0), "door", ["one"]),
        lab.score_entry(objects(0, 9, 9), "door", ["two"]),
    ]
    result = lab.aggregate_scores(entries, "door", ["one", "two"])
    assert result["value"] == pytest.approx(0.1)
    assert result["precision"] == pytest.approx(0.1)
    assert result["recall"] == pytest.approx(0.1)
    assert result["evaluated_case_ids"] == ["one", "two"]


def test_empty_instances_and_empty_predictions_do_not_invent_precision():
    empty = lab.score_entry(objects(0, 0, 0), "stairs", ["one"])
    assert empty["status"] == "scored"
    assert empty["value"] is None
    assert empty["precision"] is None
    missing = lab.score_entry(objects(0, 0, 3), "stairs", ["one"])
    assert missing["value"] == 0
    assert missing["recall"] == 0
    assert missing["precision"] is None


def test_wall_iou_is_mean_per_plan_not_object_f1():
    entries = [
        lab.score_entry({"status": "scored", "mask_iou": value}, "wall", [name])
        for name, value in (("one", 0.2), ("two", 0.8))
    ]
    result = lab.aggregate_scores(entries, "wall", ["one", "two"])
    assert result["metric"] == "mask_iou"
    assert result["value"] == pytest.approx(0.5)
    assert result["precision"] is None
