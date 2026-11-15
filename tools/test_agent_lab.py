"""Evaluation separation and immutable trial identity; no paid model calls."""

from argparse import Namespace
from hashlib import sha256
import json

import pytest

from run_agent_lab import prepare, evaluate
from prepare_agent_lab import choose, group, QUOTAS


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_preparation_does_not_need_or_read_answers(tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "drawing.png").write_bytes(b"synthetic fixture")
    case = {"id": "case-1", "split": "development", "question": "How many windows?",
            "input_path": "drawing.png", "input_sha256": sha256(b"synthetic fixture").hexdigest()}
    save(pack / "manifest.json", {"cases": [case], "answers_sha256": "missing evaluator"})
    receipt = tmp_path / "runtime.json"
    save(receipt, {"image": "immutable-test-image"})
    args = Namespace(pack=pack, split="development", out=tmp_path / "run", runtime_receipt=receipt,
                     model="scripted", url="http://127.0.0.1:9999", workers=2, revision="abc")
    binding, cases = prepare(args)
    assert not binding["labels_used_for_inference"]
    assert not (pack / "evaluator").exists()
    assert prepare(args) == (binding, cases)
    import run_agent_lab as lab
    original_digest = lab.digest
    monkeypatch.setattr(lab, "digest", lambda p: "changed-helper" if p.name == "audit_production_cohort.py" else original_digest(p))
    with pytest.raises(ValueError, match="Trial changed"):
        prepare(args)
    monkeypatch.setattr(lab, "digest", original_digest)
    args.revision = "changed"
    with pytest.raises(ValueError, match="Trial changed"):
        prepare(args)
    with pytest.raises(ValueError, match="Incomplete"):
        evaluate(binding, [], pack, cases)


def test_selection_stratifies_disjoint_drawings_without_answer_fields():
    questions = {"area": "What is the area?", "openings": "How many windows?",
                 "fixtures": "How many bathtubs?", "rooms": "How many rooms?",
                 "floors": "How many floors (storeys)?", "apartments": "How many apartments?"}
    cases = []
    for kind, count in QUOTAS.items():
        for i in range(count * 2 + 2):
            cases.append({"id": f"{kind}-{i}", "path": f"source/{kind}-{i}.pdf", "source": "fixture",
                          "question": questions[kind], "type": "area" if kind == "area" else "count"})
    excluded = {group(cases[0])}
    selected = choose(cases, excluded, "fixed-seed")
    assert len(selected) == 60
    assert len({group(c) for c in selected}) == 60
    assert not {group(c) for c in selected} & excluded
    assert selected == choose(list(reversed(cases)), excluded, "fixed-seed")
    assert all("answer" not in c for c in selected)
