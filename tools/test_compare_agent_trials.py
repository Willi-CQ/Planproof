from copy import deepcopy
import pytest

from compare_agent_trials import compare


def trial(scores):
    return {"cases": [{"id": str(i), "case_number": i + 1, "source_sha256": "source", "raster_sha256": "raster",
                       "question": "How many?", "reference": {"answer": 1}, "category": "fixtures",
                       "score": {"correct": score}, "answer": {"status": "answered" if score else "abstain"},
                       "seconds_http": 10} for i, score in enumerate(scores)]}


def test_comparison_keeps_abstentions_and_case_level_regressions():
    result = compare(trial([True, False, False]), trial([False, True, True]))
    assert result["control"]["cases"] == result["candidate"]["cases"] == 3
    assert result["candidate"]["abstained_or_failed"] == 1
    assert result["gained_cases"] == [2, 3]
    assert result["regressed_cases"] == [1]
    assert result["paired_exact_discordance_p"] == 1


@pytest.mark.parametrize("field", ["source_sha256", "raster_sha256", "question", "reference"])
def test_cannot_compare_different_pixels_questions_or_labels(field):
    control = trial([True])
    candidate = deepcopy(control)
    candidate["cases"][0][field] = "changed"
    with pytest.raises(ValueError, match=field):
        compare(control, candidate)
