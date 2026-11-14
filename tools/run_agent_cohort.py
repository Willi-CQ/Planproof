"""Run a frozen paired cohort with bounded concurrency and durable attempt records.

Labels are read only after every scheduled attempt has a completed record. Resume
skips completed successes AND failures, never selectively rerunning bad answers.
"""

import argparse
import json
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmark_agent import (
    ROOT,
    AgentRequest,
    DrawingAgent,
    GeminiTransport,
    get_job,
    get_page,
    register_image,
    register_page,
    safe_path,
    settings,
    sha256,
    specialist_service,
    telemetry,
    write_json,
)
from select_agent_cohort import category, language


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(rows):
    return {
        "attempted": len(rows),
        "correct": sum(r["score"]["correct"] is True for r in rows),
        "failures": sum(r["status"] != "ok" for r in rows),
        "seconds": round(sum(r["seconds"] for r in rows), 3),
        "median_seconds": statistics.median(r["seconds"] for r in rows),
        "max_seconds": max(r["seconds"] for r in rows),
        "algorithms_used": sum(r["algorithm_used"] for r in rows),
        "nemotron_calls": sum(r["nemotron_calls"] for r in rows),
        "abstentions": sum(r["answer"]["status"] == "abstain" for r in rows),
        "reflection_calls": sum(r["reflection_calls"] for r in rows),
        "tool_failures": sum(r["tool_failures"] for r in rows),
        "gemini_reported_tokens": sum(r["gemini_reported_tokens"] for r in rows),
        "nemotron_reported_tokens": sum(r["nemotron_reported_tokens"] for r in rows),
        "tool_usage": dict(Counter(t for r in rows for t in r["tool_names"])),
    }


def score_completed(pack, manifest, binding, records):
    # Verify completeness before opening evaluator answers, including resume runs.
    expected = Counter((i, m) for i in binding["case_ids"] for m in binding["modes"])
    if Counter((r["id"], r["mode"]) for r in records) != expected:
        raise ValueError("Incomplete/duplicate run; evaluator remains unopened")
    from planproof_feedback.scoring import score_answer

    path = pack / "evaluator/answers.json"
    if sha256(path) != manifest["answers_sha256"]:
        raise ValueError("Changed evaluator answers")
    answers = read(path)
    scored = [
        {**r, "score": score_answer(answers[r["id"]], r["answer"])} for r in records
    ]
    summary = {
        m: summarize([r for r in scored if r["mode"] == m]) for m in binding["modes"]
    }
    breakdown = {
        field: {
            value: {
                m: summarize(
                    [r for r in scored if r[field] == value and r["mode"] == m]
                )
                for m in binding["modes"]
            }
            for value in sorted({r[field] for r in scored})
        }
        for field in ("category", "source", "type", "language")
    }
    paired = Counter()
    for cid in binding["case_ids"]:
        a, d = [
            next(r for r in scored if r["id"] == cid and r["mode"] == m)["score"][
                "correct"
            ]
            for m in ("auto", "direct")
        ]
        paired[
            "both_correct"
            if a and d
            else "auto_only"
            if a
            else "direct_only"
            if d
            else "both_incorrect"
        ] += 1
    return {
        "binding": binding,
        "summary": summary,
        "breakdown": breakdown,
        "paired": dict(paired),
        "cases": scored,
    }


def run_case(case, order, pack, out, metadata, record_done):
    cid = case["id"]
    directory = out / cid
    directory.mkdir(exist_ok=True)
    pending = [m for m in order if not (directory / (m + "-record.json")).exists()]
    if not pending:
        return
    source = safe_path(pack, case["input_path"])
    if sha256(source) != case["input_sha256"]:
        raise ValueError("Source drawing changed")
    payload = source.read_bytes()
    doc = (
        register_page(payload, 1)
        if source.suffix.lower() == ".pdf"
        else register_image(payload, 1)
    )
    _, page = get_page(doc.document_id)
    (directory / "page.png").write_bytes(page.png)
    key, model = settings()
    for mode in pending:
        # Recover a trace whose process ended before its lightweight record write.
        trace_path = directory / (mode + ".json")
        if trace_path.exists():
            result = read(trace_path)
        else:
            transport = GeminiTransport(key, model)
            try:
                request = AgentRequest(
                    question=case["question"],
                    unit=case.get("unit"),
                    mode=mode,
                    allow_gemini=True,
                    allow_nemotron=True,
                )
                result = DrawingAgent(
                    page,
                    request,
                    transport,
                    lambda: get_job(doc.document_id, page),
                    specialist_service,
                    out=trace_path,
                ).run()
            finally:
                transport.close()
        record = {
            "id": cid,
            "mode": mode,
            "question": case["question"],
            "source": case["source"],
            "type": case["type"],
            "category": metadata["category"],
            "language": metadata["language"],
            "source_sha256": case["input_sha256"],
            "raster_sha256": page.raster_sha256,
            "status": result["status"],
            "answer": result["answer"],
            "algorithm_used": result["algorithm_used"],
            "nemotron_calls": result["nemotron_calls"],
            "tool_names": [t["name"] for t in result["tool_trace"]],
            "seconds": result["seconds"],
            "gemini_calls": len(result["gemini_calls"]),
            "trace": str(Path(cid) / (mode + ".json")),
            **telemetry(result),
        }
        write_json(directory / (mode + "-record.json"), record)
        record_done(record)


def main(args):
    started = time.monotonic()
    _, model = settings()
    pack, out = args.pack.resolve(), args.out.resolve()
    manifest = read(pack / "manifest.json")
    selection = read(args.selection)
    if selection["manifest_sha256"] != sha256(pack / "manifest.json"):
        raise ValueError("Selection belongs to another manifest")
    ids = selection["case_ids"]
    if (
        not 1 <= len(ids) <= 100
        or len(set(ids)) != len(ids)
        or not 1 <= args.workers <= 3
    ):
        raise ValueError("Invalid cohort or worker count")
    by_id = {c["id"]: c for c in manifest["cases"]}
    cases = [by_id[i] for i in ids]
    if len({c["input_sha256"] for c in cases}) != len(cases):
        raise ValueError("Repeated drawing in cohort")
    # No provider call until all input files, page counts and metadata are checked.
    import pymupdf

    for case in cases:
        path = safe_path(pack, case["input_path"])
        if sha256(path) != case["input_sha256"]:
            raise ValueError("Changed source")
        if path.suffix.lower() == ".pdf":
            with pymupdf.open(path) as pdf:
                if len(pdf) != 1 or pdf.needs_pass:
                    raise ValueError("Cohort requires single-page unencrypted drawings")
        meta = selection["metadata"][case["id"]]
        if meta["category"] != category(case) or meta["language"] != language(case):
            raise ValueError("Metadata mismatch")
    sources = sorted(
        {
            p.relative_to(ROOT).as_posix()
            for folder in ("backend", "feedback/src", "deterministic/src", "ml/src")
            for p in (ROOT / folder).rglob("*.py")
        }
        | {
            "tools/run_agent_cohort.py",
            "tools/select_agent_cohort.py",
            "tools/benchmark_agent.py",
            "models/baselines/v1/corrected-3500-global/SHA256SUMS",
        }
    )
    binding = {
        "schema": "planproof-agent-cohort-run-v1",
        "case_ids": ids,
        "manifest_sha256": sha256(pack / "manifest.json"),
        "selection_sha256": sha256(args.selection),
        "selection": selection,
        "model": model,
        "modes": ["auto", "direct"],
        "workers": args.workers,
        "mode_orders": {
            cid: (["auto", "direct"] if i % 2 == 0 else ["direct", "auto"])
            for i, cid in enumerate(ids)
        },
        "implementation_sha256": {p: sha256(ROOT / p) for p in sources},
        "condition": "Identical pages/questions; unchanged agent policy and budgets. Counterbalanced within-pair order. Up to three cases concurrent; host serializes algorithm preparation. No inference labels.",
        "limitation": selection["limitation"],
    }
    if args.resume:
        if read(out / "binding.json") != binding:
            raise ValueError("Cannot resume after source, cohort or settings changed")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "binding.json", binding)
        snapshots = out / "implementation"
        snapshots.mkdir()
        for p in sources:
            (snapshots / p.replace("/", "__")).write_bytes((ROOT / p).read_bytes())
    records = [read(p) for cid in ids for p in (out / cid).glob("*-record.json")]
    expected = {(i, m) for i in ids for m in binding["modes"]}
    if len({(r["id"], r["mode"]) for r in records}) != len(records) or any(
        (r["id"], r["mode"]) not in expected for r in records
    ):
        raise ValueError("Invalid saved attempt records")
    lock = threading.Lock()

    def save(record):
        with lock:
            records.append(record)
            write_json(out / "predictions.json", records)
            print(
                json.dumps(
                    {
                        "completed": len(records),
                        "total": len(expected),
                        "case": ids.index(record["id"]) + 1,
                        "mode": record["mode"],
                        "category": record["category"],
                        "status": record["status"],
                        "seconds": record["seconds"],
                        "tools": record["tool_names"],
                    }
                ),
                flush=True,
            )

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_case,
                c,
                binding["mode_orders"][c["id"]],
                pack,
                out,
                selection["metadata"][c["id"]],
                save,
            ): c["id"]
            for c in cases
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                # Do not serialize provider exception text, credentials or bodies.
                errors.append(
                    {"case_id": futures[future], "error_code": type(exc).__name__}
                )
                write_json(out / "orchestration-errors.json", errors)
    if errors:
        raise RuntimeError(
            "Orchestration failure; completed attempts preserved, evaluation not run"
        )
    records.sort(key=lambda r: (ids.index(r["id"]), binding["modes"].index(r["mode"])))
    write_json(out / "predictions.json", records)
    report = score_completed(pack, manifest, binding, records)
    report["wall_seconds_this_invocation"] = round(time.monotonic() - started, 3)
    write_json(out / "report.json", report)
    print(
        json.dumps(
            {
                "summary": report["summary"],
                "paired": report["paired"],
                "wall_seconds": report["wall_seconds_this_invocation"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
