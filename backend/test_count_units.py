import pytest

from count_evidence import canonical_count_unit
from planproof_feedback.scoring import score_answer


@pytest.mark.parametrize("noun", ["stairs", "stair flights", "flights of stairs", "columns", "pillars", "beams", "balconies", "washbasins", "urinals"])
def test_explicit_count_contract_normalizes_supported_entity_labels(noun):
    answer = {"status": "answered", "answer": 1.0, "unit": canonical_count_unit(noun, "count")}
    assert score_answer({"type": "count", "answer": 1, "unit": "count"}, answer)["correct"]


@pytest.mark.parametrize("unit", ["m", "m2", "m3", "ft", "ft2", "mm", "unknown", None])
def test_count_contract_does_not_relabel_physical_or_unknown_units(unit):
    assert canonical_count_unit(unit, "count") == unit


def test_ordinary_chat_and_measurement_requests_keep_descriptive_units():
    assert canonical_count_unit("stair flights", None) == "stair flights"
    assert canonical_count_unit("columns", "m2") == "columns"
