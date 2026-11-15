"""Separate synthetic behavior scoring; real benchmark scoring remains unchanged."""

import argparse
import json
from pathlib import Path
import statistics

from audit_production_cohort import digest, read, require_complete, write
from planproof_feedback.scoring import score_answer


def policy_score(reference, answer, *, completed=True):
    if not completed:
        return {"scorable": True, "correct": False, "reason": "execution_did_not_complete"}
    if reference["expected_status"] == "abstain":
        return {"scorable": True, "correct": answer.get("status") == "abstain" and answer.get("answer") is None,
                "reason": "required_abstention_without_unsupported_value"}
    return score_answer(reference, answer)


def evaluate(pack, directory):
    manifest, binding = read(pack / "manifest.json"), read(directory / "binding.json")
    if manifest["schema"] != "planproof-synthetic-policy-v1" or digest(pack / "manifest.json") != binding["manifest_sha256"]:
        raise ValueError("Synthetic policy suite binding changed")
    records = [read(directory / cid / "record.json") for cid in binding["case_ids"]]
    require_complete(binding, records)
    if digest(pack / "evaluator/answers.json") != manifest["answers_sha256"]:
        raise ValueError("Synthetic reference changed")
    references = read(pack / "evaluator/answers.json")
    cases = {c["id"]: c for c in manifest["cases"]}
    rows = []
    for r in records:
        result = r["run"].get("result") or {}
        answer = result.get("answer") or {}
        reference = references[r["id"]]
        rows.append({**{k: v for k, v in r.items() if k != "run"}, "question": cases[r["id"]]["question"],
                     "reference": reference, "answer": answer, "score": policy_score(reference, answer,
                         completed=r["run"]["status"] == "completed" and result.get("status") == "ok"),
                     "nemotron_calls": result.get("nemotron_calls", 0), "algorithm_used": result.get("algorithm_used", False),
                     "trace": f"{r['id']}/record.json"})
    report = {"binding": binding, "cases": rows, "summary": {
        "synthetic": True, "correct": sum(r["score"]["correct"] for r in rows), "cases": len(rows),
        "median_seconds": statistics.median(r["seconds_http"] for r in rows),
        "limitation": manifest["limitation"]}}
    write(directory / "report.json", report)
    return report["summary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.pack, args.run)))
