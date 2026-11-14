"""Compare direct Gemini, visual tools and question-aware algorithms on fixed cases.

Inference finishes before reference answers are read. Source drawings and original
Nemotron benchmark remain unchanged. These previously inspected cases are a
development comparison, not a fresh held-out accuracy estimate.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from agent_pipeline import get_job, specialist_service  # noqa: E402
from app import get_page, register_image, register_page  # noqa: E402
from question_agent import AgentRequest, DrawingAgent, GeminiTransport, settings  # noqa: E402
from planproof_feedback.artifacts import safe_path, write_json  # noqa: E402
from planproof_feedback.images import sha256  # noqa: E402


def run(args):
    key, model = settings()
    pack = args.pack.resolve()
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection["manifest_sha256"] != sha256(pack / "manifest.json"):
        raise ValueError("Selection belongs to another question pack")
    by_id = {c["id"]: c for c in manifest["cases"]}
    ids = selection["case_ids"][: args.count]
    if not ids or len(ids) != args.count or not 1 <= args.count <= 100:
        raise ValueError("Invalid case count")
    modes = args.modes.split(",")
    if (
        not modes
        or len(set(modes)) != len(modes)
        or not set(modes) <= {"direct", "visual_only", "auto"}
    ):
        raise ValueError("Invalid modes")
    args.out.mkdir(parents=True, exist_ok=False)
    binding = {
        "case_ids": ids,
        "manifest_sha256": sha256(pack / "manifest.json"),
        "model": model,
        "modes": modes,
        "source_branch": "iamd-johan-dev@7803f6c reviewed; integration rebuilt on main",
        "implementation_sha256": {
            p: sha256(ROOT / p)
            for p in [
                "backend/question_agent.py",
                "backend/visual_specialist.py",
                "backend/agent_pipeline.py",
                "backend/app.py",
                "feedback/src/planproof_feedback/pipeline.py",
                "tools/benchmark_agent.py",
            ]
        },
        "condition": "Same canonical drawing, question, Gemini settings and budgets; tools differ by mode. No inference labels.",
        "limitation": "Previously inspected development cohort. Model randomness and small sample limit conclusions.",
    }
    write_json(args.out / "binding.json", binding)
    snapshots = args.out / "implementation"
    snapshots.mkdir()
    for source_path in binding["implementation_sha256"]:
        (snapshots / source_path.replace("/", "__")).write_bytes(
            (ROOT / source_path).read_bytes()
        )
    records = []
    for cid in ids:
        case = by_id[cid]
        source = safe_path(pack, case["input_path"])
        if sha256(source) != case["input_sha256"]:
            raise ValueError("Changed source drawing")
        doc = (
            register_page(source.read_bytes(), 1)
            if source.suffix.lower() == ".pdf"
            else register_image(source.read_bytes(), 1)
        )
        _, page = get_page(doc.document_id)
        case_dir = args.out / cid
        case_dir.mkdir()
        (case_dir / "page.png").write_bytes(page.png)
        for mode in modes:
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
                    out=case_dir / (mode + ".json"),
                ).run()
            finally:
                transport.close()
            records.append(
                {
                    "id": cid,
                    "mode": mode,
                    "question": case["question"],
                    "source": case["source"],
                    "type": case["type"],
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
                }
            )
            write_json(args.out / "predictions.json", records)
            print(
                json.dumps(
                    {
                        "completed": len(records),
                        "total": len(ids) * len(modes),
                        "mode": mode,
                        "status": result["status"],
                        "algorithms": result["algorithm_used"],
                        "tools": records[-1]["tool_names"],
                    }
                ),
                flush=True,
            )
    # Strict evaluator boundary. The model has already finished every attempt.
    from planproof_feedback.scoring import score_answer

    labels = pack / "evaluator/answers.json"
    if sha256(labels) != manifest["answers_sha256"]:
        raise ValueError("Changed reference answers")
    answers = json.loads(labels.read_text(encoding="utf-8"))
    for record in records:
        record["score"] = score_answer(answers[record["id"]], record["answer"])
    summary = {}
    for mode in modes:
        rows = [r for r in records if r["mode"] == mode]
        summary[mode] = {
            "attempted": len(rows),
            "correct": sum(r["score"]["correct"] is True for r in rows),
            "failures": sum(r["status"] != "ok" for r in rows),
            "seconds": sum(r["seconds"] for r in rows),
            "algorithms_used": sum(r["algorithm_used"] for r in rows),
            "nemotron_calls": sum(r["nemotron_calls"] for r in rows),
            "tool_usage": dict(Counter(t for r in rows for t in r["tool_names"])),
        }
    write_json(
        args.out / "report.json",
        {"binding": binding, "summary": summary, "cases": records},
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--modes", default="direct,visual_only,auto")
    run(parser.parse_args())
