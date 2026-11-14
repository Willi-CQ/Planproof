"""Selection and evaluator-boundary regressions; no live provider calls."""

import json
from collections import Counter

import pytest

from run_agent_cohort import score_completed, sha256
from select_agent_cohort import QUOTAS, category, language, select


def test_retry_usage_counts_reported_attempts_without_double_counting_turns():
    from benchmark_agent import telemetry

    result = {
        "tool_trace": [],
        "gemini_calls": [{"usage": {"total_token_count": 10}}],
        "images": [{"image_id": "overview", "source_render_size": [10, 10]}],
        "stop_reason": "answered",
    }
    assert telemetry(result)["gemini_reported_tokens"] == 10
    result["reliability"] = {
        "provider_attempts": [
            {"usage": {"total_token_count": 7}},
            {"usage": {"total_token_count": 10}},
        ]
    }
    assert telemetry(result)["gemini_reported_tokens"] == 17


def test_balanced_selection_is_order_independent_and_excludes_prior_drawings():
    questions = {
        "area": "What is the net floor area?",
        "apartments": "How many apartments are shown?",
        "floors": "How many floors (storeys) are shown?",
        "openings": "How many windows are shown?",
        "fixtures": "How many showers are shown?",
        "rooms": "How many rooms labelled bedroom?",
    }
    cases = []
    for (group, source), count in QUOTAS.items():
        for i in range(count + 3):
            cid = f"{group}-{source}-{i}"
            cases.append(
                {
                    "id": cid,
                    "input_sha256": cid,
                    "source": source,
                    "type": "area" if group == "area" else "count",
                    "question": questions[group],
                    "answer": "MUST NOT BE USED",
                }
            )
    prior = cases[0]
    cases.append({**prior, "id": "different-question-same-prior-drawing"})
    a = select(cases, {prior["id"]}, "test-seed")
    b = select(list(reversed(cases)), {prior["id"]}, "test-seed")
    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert len(a) == len({c["input_sha256"] for c in a}) == 50
    assert prior["input_sha256"] not in {c["input_sha256"] for c in a}
    assert Counter((category(c), c["source"]) for c in a) == QUOTAS
    for case in cases:
        case["answer"] = "CHANGED WITHOUT EFFECT"
    assert [c["id"] for c in select(cases, {prior["id"]}, "test-seed")] == [
        c["id"] for c in a
    ]


def test_room_wc_and_fixture_wc_remain_distinct():
    assert (
        category({"type": "count", "question": "Kuinka monta erillistä WC-tilaa on?"})
        == "rooms"
    )
    assert (
        category({"type": "count", "question": "Kuinka monta WC-istuinta on?"})
        == "fixtures"
    )
    assert language({"question": "Qual è la superficie?"}) == "it"
    with pytest.raises(ValueError, match="Insufficient"):
        select([], set(), "seed")


def test_evaluator_is_not_opened_for_missing_or_duplicate_attempts(tmp_path):
    # There is deliberately no evaluator file. Completeness must fail first.
    binding = {"case_ids": ["one"], "modes": ["auto", "direct"]}
    for rows in ([], [{"id": "one", "mode": "auto"}] * 2):
        with pytest.raises(ValueError, match="evaluator remains unopened"):
            score_completed(tmp_path, {}, binding, rows)


def test_scoring_includes_failed_attempts_and_area_tolerance(tmp_path):
    directory = tmp_path / "evaluator"
    directory.mkdir()
    labels = directory / "answers.json"
    labels.write_text(
        json.dumps(
            {
                "c": {"type": "count", "unit": "count", "answer": 2},
                "a": {"type": "area", "unit": "m2", "answer": 100},
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for cid in ("c", "a"):
        for mode in ("auto", "direct"):
            failed = cid == "c" and mode == "direct"
            answer = {
                "status": "abstain" if failed else "answered",
                "answer": None if failed else 2 if cid == "c" else 101,
                "unit": "count" if cid == "c" else "m2",
            }
            rows.append(
                {
                    "id": cid,
                    "mode": mode,
                    "answer": answer,
                    "status": "unavailable" if failed else "ok",
                    "source": "fixture",
                    "type": "count" if cid == "c" else "area",
                    "category": "test",
                    "language": "en",
                    "seconds": 1,
                    "algorithm_used": False,
                    "nemotron_calls": 0,
                    "reflection_calls": 0,
                    "tool_failures": 0,
                    "gemini_reported_tokens": 0,
                    "nemotron_reported_tokens": 0,
                    "tool_names": [],
                }
            )
    report = score_completed(
        tmp_path,
        {"answers_sha256": sha256(labels)},
        {"case_ids": ["c", "a"], "modes": ["auto", "direct"]},
        rows,
    )
    assert report["summary"]["auto"]["correct"] == 2
    assert report["summary"]["direct"]["attempted"] == 2
    assert report["summary"]["direct"]["failures"] == 1
    assert report["paired"] == {"auto_only": 1, "both_correct": 1}
    assert "score" not in rows[0]  # Inference-only records stay unscored.
