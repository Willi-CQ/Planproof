import json
import pytest

from prepare_agent_policy_suite import build
from score_agent_policy_suite import policy_score


def test_negative_policy_cases_cannot_pass_with_an_unsupported_number():
    ref = {"expected_status": "abstain", "type": "area", "answer": None, "unit": "m2"}
    assert policy_score(ref, {"status": "abstain", "answer": None})["correct"]
    assert not policy_score(ref, {"status": "abstain", "answer": 99})["correct"]
    assert not policy_score(ref, {"status": "answered", "answer": 99})["correct"]
    assert not policy_score(ref, {})["correct"]
    assert not policy_score(ref, {"status": "abstain", "answer": None}, completed=False)["correct"]


def test_generated_suite_is_explicitly_synthetic_with_separate_labels(tmp_path):
    pack = tmp_path / "policy"
    assert build(pack)["cases"] == 13
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["cases"]) == 13
    assert all("answer" not in c and "expected_status" not in c for c in manifest["cases"])
    assert all((pack / c["input_path"]).is_file() for c in manifest["cases"])
    assert "not real drawing accuracy" in manifest["limitation"]
    with pytest.raises(ValueError, match="immutable"):
        build(pack)
