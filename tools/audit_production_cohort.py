"""Replay a fixed cohort through the production upload/run/poll API.

No evaluator answers are loaded until every requested case is terminal. Existing
attempts, including failures, are never silently replaced. Artifacts stay local.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys
import time
from uuid import UUID, uuid5

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "feedback/src"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def implementation():
    files = [
        p
        for folder in ("backend", "feedback/src", "deterministic/src", "ml/src")
        for p in (ROOT / folder).rglob("*.py")
    ]
    files += [
        ROOT / "feedback/config.toml",
        Path(__file__),
        ROOT / "models/baselines/v1/corrected-3500-global/SHA256SUMS",
    ]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}


def prepare(baseline, out, url, workers):
    inventory = read(baseline / "cohort-inputs.json")
    old = read(baseline / "binding.json")
    cases = inventory["cases"]
    ids = [c["id"] for c in cases]
    if ids != old["case_ids"] or len(set(ids)) != len(ids):
        raise ValueError("Cohort identity/order differs from frozen benchmark")
    pack = Path(inventory["source_pack"]).resolve()
    if digest(pack / "manifest.json") != old["manifest_sha256"]:
        raise ValueError("Reference manifest changed")
    for case in cases:
        source = (pack / case["input_path"]).resolve()
        if not source.is_relative_to(pack) or digest(source) != case["input_sha256"]:
            raise ValueError("Source drawing changed or escaped pack")
    binding = {
        "schema": "planproof-production-cohort-v1",
        "case_ids": ids,
        "baseline_binding_sha256": digest(baseline / "binding.json"),
        "baseline_report_sha256": digest(baseline / "report.json"),
        "manifest_sha256": old["manifest_sha256"],
        "model": old["model"],
        "implementation_sha256": implementation(),
        "workers": workers,
        "server": url,
        "request": {"mode": "auto", "allow_gemini": True, "allow_nemotron": True},
        "unit_policy": "Original benchmark unit hint retained for matched comparison; ordinary chat omits this optional hint.",
        "condition": "Current production upload/run/poll, observer enabled, interactive budget and mandatory specialist. Historical baseline is not a simultaneous control.",
        "labels_used_for_inference": False,
        "limit": "Development cohort previously inspected. Citation bounds are structural checks, not semantic validation. Failed attempts stay in denominator.",
    }
    if out.exists():
        if read(out / "binding.json") != binding:
            raise ValueError("Frozen run changed; use a separate output directory")
    else:
        out.mkdir(parents=True)
        write(out / "binding.json", binding)
        write(out / "inputs.json", inventory)
        snapshots = out / "implementation"
        snapshots.mkdir()
        for relative in binding["implementation_sha256"]:
            (snapshots / relative.replace("/", "__")).write_bytes(
                (ROOT / relative).read_bytes()
            )
    return binding, cases, pack


def run_case(case, index, binding, pack, out):
    def admission_closed():
        deadline = binding.get("admission_deadline_unix")
        if deadline is not None and time.time() >= deadline:
            (out / "STOP").write_text("Admission deadline reached", encoding="utf-8")
        return (out / "STOP").exists()

    if admission_closed():
        return None  # Finish in-flight requests, but admit no more paid work.
    directory = out / case["id"]
    directory.mkdir(exist_ok=True)
    destination = directory / "record.json"
    if destination.exists():
        return read(destination)
    run_id = str(
        uuid5(
            UUID("62d90551-51fb-4398-b6ec-0d9f6b0343bb"),
            str(out.resolve()) + case["id"],
        )
    )
    admission = directory / "admission.json"
    with httpx.Client(
        base_url=binding["server"], timeout=60, trust_env=False
    ) as client:
        started = time.monotonic()
        if admission.exists():
            saved = read(admission)
            response = client.get(f"/api/v1/runs/{run_id}")
            # An ambiguous admission must be investigated, not automatically billed again.
            response.raise_for_status()
            snapshot = response.json()
        else:
            source = (pack / case["input_path"]).resolve()
            with source.open("rb") as stream:
                upload = client.post(
                    "/api/documents",
                    files={"file": (source.name, stream)},
                    data={"page": "1"},
                )
            upload.raise_for_status()
            document = upload.json()
            frame = client.get(f"/api/v1/documents/{document['document_id']}/frame")
            frame.raise_for_status()
            request = {
                **binding["request"],
                "question": case["question"],
                "unit": case.get("unit"),
            }
            if admission_closed():
                return None
            saved = {
                "run_id": run_id,
                "request": request,
                "document": document,
                "source": frame.json(),
                "started": time.time(),
            }
            write(admission, saved)
            response = client.post(
                f"/api/v1/documents/{document['document_id']}/runs",
                json={"request_id": run_id, "request": request},
            )
            response.raise_for_status()
            snapshot = response.json()
        print(json.dumps({"case": index, "state": "started"}), flush=True)
        while snapshot["status"] not in {"completed", "failed", "interrupted"}:
            if time.monotonic() - started > 900:
                raise TimeoutError(
                    "Run still active; preserve admission for later resume"
                )
            time.sleep(1)
            response = client.get(f"/api/v1/runs/{run_id}")
            response.raise_for_status()
            snapshot = response.json()
        result = snapshot.get("result") or {}
        if result and result.get("model") != binding["model"]:
            raise ValueError("Unexpected model in production result")
        record = {
            "id": case["id"],
            "case_number": index,
            # Older reference packs do not carry these reporting annotations.
            # Missing metadata must never discard an already completed paid run.
            "category": case.get("category") or "unspecified",
            "language": case.get("language") or "unknown",
            "source": case["source"],
            "type": case["type"],
            "run_id": run_id,
            "source_sha256": case["input_sha256"],
            "raster_sha256": saved["source"]["raster_sha256"],
            "seconds_http": round(time.time() - saved["started"], 3),
            "run": snapshot,
        }
        write(destination, record)
        print(
            json.dumps(
                {
                    "case": index,
                    "state": "completed",
                    "status": snapshot["status"],
                    "answer_status": (result.get("answer") or {}).get("status"),
                    "seconds": record["seconds_http"],
                    "nemotron_calls": result.get("nemotron_calls", 0),
                }
            ),
            flush=True,
        )
        return record


def require_complete(binding, records):
    if Counter(r["id"] for r in records) != Counter(binding["case_ids"]):
        raise ValueError("Incomplete/duplicate cohort; evaluator remains unopened")
    if any(
        r["run"]["status"] not in {"completed", "failed", "interrupted"}
        for r in records
    ):
        raise ValueError("Nonterminal attempt; evaluator remains unopened")


def evaluate(binding, records, pack, baseline):
    require_complete(binding, records)
    from planproof_feedback.scoring import score_answer

    manifest = read(pack / "manifest.json")
    labels = pack / "evaluator/answers.json"
    if digest(labels) != manifest["answers_sha256"]:
        raise ValueError("Evaluator labels changed")
    if digest(baseline / "report.json") != binding["baseline_report_sha256"]:
        raise ValueError("Historical report changed")
    references = read(labels)
    previous = {
        r["id"]: r
        for r in read(baseline / "report.json")["cases"]
        if r["mode"] == "auto"
    }
    rows = []
    for record in records:
        result = record["run"].get("result") or {}
        answer = result.get("answer") or {}
        old = previous[record["id"]]
        reference = references[record["id"]]
        score = score_answer(reference, answer)
        # Additional count-value diagnostic only; primary scoring is unchanged.
        numeric = (
            score_answer(reference, {**answer, "unit": "count"})
            if reference["type"] == "count"
            else score
        )
        attempts = result.get("reliability", {}).get("provider_attempts", [])
        rows.append(
            {
                **{k: v for k, v in record.items() if k != "run"},
                "question": old["question"],
                "reference": reference,
                "before": {
                    "answer": old["answer"],
                    "score": old["score"],
                    "seconds": old["seconds"],
                    "status": old["status"],
                },
                "answer": answer,
                "score": score,
                "count_value_score": numeric,
                "raster_matches": old["raster_sha256"] == record["raster_sha256"],
                "status": result.get("status", record["run"]["status"]),
                "stop_reason": result.get("stop_reason"),
                "completion_reason": result.get("completion_reason"),
                "nemotron_calls": result.get("nemotron_calls", 0),
                "algorithm_used": result.get("algorithm_used", False),
                "gemini_tokens": sum(
                    (a.get("usage") or {}).get("total_token_count") or 0
                    for a in attempts
                ),
                "provider_failures": dict(
                    Counter(a["category"] for a in attempts if a.get("category"))
                ),
                "citation_count": len(result.get("canonical_citations", [])),
                "trace": f"{record['id']}/record.json",
            }
        )
    summary = {
        "cases": len(rows),
        "baseline_correct": sum(r["before"]["score"]["correct"] for r in rows),
        "current_correct": sum(r["score"]["correct"] for r in rows),
        "current_count_value_correct": sum(
            r["count_value_score"]["correct"] for r in rows
        ),
        "returned_answers": sum(r["answer"].get("status") == "answered" for r in rows),
        "regressions": [
            r["case_number"]
            for r in rows
            if r["before"]["score"]["correct"] and not r["score"]["correct"]
        ],
        "gains": [
            r["case_number"]
            for r in rows
            if not r["before"]["score"]["correct"] and r["score"]["correct"]
        ],
        "median_seconds": statistics.median(r["seconds_http"] for r in rows),
        "gemini_reported_tokens": sum(r["gemini_tokens"] for r in rows),
        "nemotron_calls": sum(r["nemotron_calls"] for r in rows),
        "rasters_match": all(r["raster_matches"] for r in rows),
        "cost": "No verified endpoint tariff; reported tokens retained, no invented price.",
    }
    return {"binding": binding, "summary": summary, "cases": rows}


def main(args):
    binding, cases, pack = prepare(args.baseline, args.out, args.url, args.workers)
    print(
        json.dumps(
            {
                "prepared_cases": len(cases),
                "model": binding["model"],
                "execute": args.execute,
            }
        ),
        flush=True,
    )
    if not args.execute and not args.score:
        return
    if args.execute:
        with httpx.Client(base_url=args.url, trust_env=False) as client:
            health = client.get("/api/v1/agent").json()
        if not health.get("configured") or health.get("model") != binding["model"]:
            raise ValueError("Production model configuration differs")

        def run_bound(pair):
            try:
                return run_case(pair[1], pair[0] + 1, binding, pack, args.out)
            except Exception as exc:
                (args.out / "STOP").write_text(type(exc).__name__, encoding="utf-8")
                raise

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_bound, enumerate(cases)))
    if implementation() != binding["implementation_sha256"]:
        raise ValueError("Implementation changed during the run")
    paths = [args.out / cid / "record.json" for cid in binding["case_ids"]]
    if (args.out / "STOP").exists() and not all(p.exists() for p in paths):
        print(
            json.dumps(
                {
                    "paused_completed": sum(p.exists() for p in paths),
                    "total": len(paths),
                }
            ),
            flush=True,
        )
        return
    records = [read(path) for path in paths]
    report = evaluate(binding, records, pack, args.baseline)
    write(args.out / "report.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8005")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--score", action="store_true")
    main(parser.parse_args())
