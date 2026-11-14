import pytest

from planproof_feedback.scoring import score_answer


def scored(kind, expected, actual, unit="count", pred_unit=None):
    return score_answer(
        {"type": kind, "answer": expected, "unit": unit},
        {"status": "answered", "answer": actual, "unit": unit if pred_unit is None else pred_unit},
    )


@pytest.mark.parametrize(
    "actual,correct",
    [(3, True), ("3.0", True), (2, False), (3.1, False), (True, False), ("3 doors", False)],
)
def test_exact_count(actual, correct):
    assert scored("count", 3, actual)["correct"] is correct


@pytest.mark.parametrize(
    "actual,correct", [(98, True), (102, True), (102.001, False), (97.99, False)]
)
def test_two_percent_boundary(actual, correct):
    assert scored("area", 100, actual, "m2")["correct"] is correct


def test_units_conversion_and_zero_reference():
    assert scored("area", 0.09290304, 1, "m2", "ft²")["correct"]
    assert scored("length", 1, 1000, "m", "mm")["correct"]
    assert not scored("area", 100, 100, "m2", "m")["correct"]
    assert not scored("area", 100, 100, "m2", "")["correct"]
    assert scored("area", 0, 0, "m2")["correct"]
    assert not scored("area", 0, 0.0001, "m2")["correct"]


def test_text_normalization_and_abstentions():
    assert scored("text", "  Room   A  ", "room a", None)["correct"]
    assert not score_answer({"type": "count", "answer": 3}, {"status": "abstain"})["correct"]
    assert score_answer({"type": "other", "answer": 3}, None)["scorable"] is False
