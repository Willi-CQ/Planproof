"""Immutable reference-cohort trials through real production HTTP routes.

Does not inspect evaluator labels until all selected attempts are terminal.
The server revision/image receipt is supplied independently of the local runner.
"""

import argparse
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import statistics

from audit_production_cohort import digest, read, write, run_case, require_complete


def prepare(args):
    manifest = read(args.pack / "manifest.json")
    cases = [c for c in manifest["cases"] if c.get("split") == args.split]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Empty or duplicate cohort")
    for case in cases:
        source = (args.pack / case["input_path"]).resolve()
        if not source.is_relative_to(args.pack.resolve()) or digest(source) != case["input_sha256"]:
            raise ValueError("Drawing integrity failure")
    binding = {
        "schema": "planproof-agent-lab-v1", "case_ids": [c["id"] for c in cases],
        "manifest_sha256": digest(args.pack / "manifest.json"),
        "runner_sha256": digest(Path(__file__)), "runtime": read(args.runtime_receipt),
        "runner_dependencies_sha256": {
            "audit_production_cohort.py": digest(Path(__file__).with_name("audit_production_cohort.py")),
        },
        "model": args.model, "server": args.url, "workers": args.workers,
        "split": args.split, "revision": args.revision,
        "request": {"mode": "auto", "allow_gemini": True, "allow_nemotron": True},
        "labels_used_for_inference": False,
        "unit_policy": "Publisher unit hint supplied identically in matched conditions.",
    }
    if getattr(args, "admission_deadline_utc", None):
        deadline = datetime.fromisoformat(args.admission_deadline_utc.replace("Z", "+00:00"))
        if deadline.tzinfo is None:
            raise ValueError("Admission deadline must include a timezone")
        binding["admission_deadline_unix"] = deadline.timestamp()
    if args.out.exists():
        if read(args.out / "binding.json") != binding:
            raise ValueError("Trial changed; choose a new output directory")
    else:
        args.out.mkdir(parents=True)
        write(args.out / "binding.json", binding)
        write(args.out / "inputs.json", cases)
    return binding, cases


def evaluate(binding, records, pack, cases):
    require_complete(binding, records)
    from planproof_feedback.scoring import score_answer
    manifest = read(pack / "manifest.json")
    if digest(pack / "evaluator/answers.json") != manifest["answers_sha256"]:
        raise ValueError("Evaluator integrity failure")
    references = read(pack / "evaluator/answers.json")
    inputs = {c["id"]: c for c in cases}
    rows = []
    for record in records:
        result = record["run"].get("result") or {}
        answer = result.get("answer") or {}
        attempts = result.get("reliability", {}).get("provider_attempts", [])
        rows.append({**{k: v for k, v in record.items() if k != "run"},
                     "question": inputs[record["id"]]["question"],
                     "reference": references[record["id"]], "answer": answer,
                     "score": score_answer(references[record["id"]], answer),
                     "algorithm_used": result.get("algorithm_used", False),
                     "nemotron_calls": result.get("nemotron_calls", 0),
                     "gemini_tokens": sum((a.get("usage") or {}).get("total_token_count") or 0 for a in attempts),
                     "stop_reason": result.get("stop_reason"),
                     "trace": f"{record['id']}/record.json"})
    return {"binding": binding, "cases": rows, "summary": {
        "cases": len(rows), "correct": sum(r["score"]["correct"] for r in rows),
        "abstentions": sum(r["answer"].get("status") != "answered" for r in rows),
        "median_seconds": statistics.median(r["seconds_http"] for r in rows),
        "mean_seconds": statistics.mean(r["seconds_http"] for r in rows),
        "gemini_reported_tokens": sum(r["gemini_tokens"] for r in rows),
        "nemotron_calls": sum(r["nemotron_calls"] for r in rows),
        "algorithm_delivered": sum(r["algorithm_used"] for r in rows),
        "categories": {k: {"cases": n, "correct": sum(r["score"]["correct"] for r in rows if r["category"] == k)}
                       for k, n in Counter(r["category"] for r in rows).items()},
    }}


def main(args):
    binding, cases = prepare(args)
    print(json.dumps({"cases": len(cases), "split": args.split, "revision": args.revision}), flush=True)
    if args.execute:
        def run(pair):
            try:
                return run_case(pair[1], pair[0] + 1, binding, args.pack, args.out)
            except Exception as exc:
                (args.out / "STOP").write_text(type(exc).__name__, encoding="utf-8")
                raise
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run, enumerate(cases)))
    paths = [args.out / c["id"] / "record.json" for c in cases]
    if not all(p.is_file() for p in paths):
        print(json.dumps({"completed": sum(p.is_file() for p in paths), "total": len(paths)}), flush=True)
        return
    if args.score:
        report = evaluate(binding, [read(p) for p in paths], args.pack, cases)
        write(args.out / "report.json", report)
        print(json.dumps(report["summary"], indent=2), flush=True)
    else:
        print('All cases terminal; evaluator remains unopened.', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--model", default="gemini-4-carbon")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--admission-deadline-utc", help="Absolute cutoff for new paid requests; timezone required")
    parser.add_argument("--score", action="store_true")
    main(parser.parse_args())
