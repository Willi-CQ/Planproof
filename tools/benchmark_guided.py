"""Matched pilot: existing fixed Stoa questions, algorithm-guided Nemotron vs direct.

No reference answers are opened until all selected inference attempts finish.
The original benchmark, its predictions and source drawings remain unchanged.
"""

import argparse
import hashlib
import json
import random
import subprocess
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

from PIL import Image, ImageOps

from planproof_feedback.artifacts import safe_path, write_json
from planproof_feedback.config import load_config
from planproof_feedback.guided import ROUTE, answer_guided
from planproof_feedback.images import load_images, sha256
from planproof_feedback.pipeline import EvidencePipeline, EvidenceJob
from planproof_feedback.service import create_service


def select(cases, count, include_pdf=False):
    if not 1 <= count <= 100:
        raise ValueError("Pilot count must be between 1 and 100")
    # Balance by source, then sample within each source independently of answers.
    groups = {}
    for case in sorted(cases, key=lambda c: c["id"]):
        if Path(case["input_path"]).suffix.lower() in (
            (".png", ".jpg", ".jpeg", ".pdf")
            if include_pdf
            else (".png", ".jpg", ".jpeg")
        ):
            groups.setdefault(case["source"], []).append(case)
    rng = random.Random("planproof-guided-pilot-v1")
    for values in groups.values():
        rng.shuffle(values)
    chosen = []
    while len(chosen) < count:
        added = False
        for source in sorted(groups):
            if groups[source] and len(chosen) < count:
                chosen.append(groups[source].pop())
                added = True
        if not added:
            raise ValueError("Not enough raster questions in the frozen pack")
    return chosen


def run(args):
    pack, out = args.pack.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    eligible = manifest["cases"]
    if args.include_pdf:
        import pypdfium2 as pdfium

        eligible = []
        for case in manifest["cases"]:
            source = safe_path(pack, case["input_path"])
            if source.suffix.lower() == ".pdf":
                with closing(pdfium.PdfDocument(source)) as pdf:
                    if len(pdf) != 1:
                        continue
            eligible.append(case)
    selected = select(eligible, args.count, args.include_pdf)
    config = load_config()
    config["env_file"] = str(args.env_file.resolve())
    config["audit_dir"] = str(out / "calls")
    if args.disable_thinking:
        for settings in config["providers"].values():
            settings["enable_thinking"] = False
    # Keep the same provider/model/image configuration as the original baseline.
    service = create_service(config)
    pipeline = EvidencePipeline()
    binding = {
        "manifest_sha256": sha256(pack / "manifest.json"),
        "case_ids": [c["id"] for c in selected],
        "model": config["providers"]["nemotron"]["model"],
        "routing_prompt_sha256": hashlib.sha256(ROUTE.encode()).hexdigest(),
        "condition": "algorithms + routing + evidence-conditioned answer; two calls maximum",
        "count": args.count,
        "by_source": dict(Counter(c["source"] for c in selected)),
        "by_type": dict(Counter(c["type"] for c in selected)),
        "limits": "Selected raster-only pilot, not full benchmark; source-balanced, not population representative.",
    }
    source_root = Path(__file__).resolve().parents[1]
    files = [
        "feedback/src/planproof_feedback/" + name + ".py"
        for name in ("guided", "pipeline", "regions", "service")
    ]
    files.append("backend/pipeline_worker.py")
    binding["implementation_sha256"] = {
        name: sha256(source_root / name) for name in files
    }
    binding["generation_settings"] = config["providers"]
    binding["paired_direct_control"] = args.paired_direct
    binding["single_page_pdf_included"] = args.include_pdf
    if (out / "binding.json").exists() and json.loads(
        (out / "binding.json").read_text(encoding="utf-8")
    ) != binding:
        raise ValueError("Existing pilot uses another binding")
    write_json(out / "binding.json", binding)
    predictions_path = out / "predictions.json"
    records = (
        json.loads(predictions_path.read_text(encoding="utf-8"))
        if predictions_path.exists()
        else []
    )
    done = {r["id"] for r in records}
    for case in selected:
        if case["id"] in done:
            continue
        start = time.perf_counter()
        directory = out / case["id"]
        directory.mkdir(exist_ok=True)
        record = {
            "id": case["id"],
            "question": case["question"],
            "source": case["source"],
            "type": case["type"],
        }
        try:
            source = safe_path(pack, case["input_path"])
            if sha256(source) != case["input_sha256"]:
                raise ValueError("Changed source drawing")
            if args.paired_direct:
                control_file = directory / "direct-control.json"
                if control_file.exists():
                    control = json.loads(control_file.read_text(encoding="utf-8"))
                else:
                    try:
                        control = service.answer(
                            case["question"],
                            load_images([source], config["images"]),
                            unit=case["unit"],
                            question_type=case["type"],
                        )
                    except (ValueError, RuntimeError, OSError) as exc:
                        control = {
                            "status": "error",
                            "result": {},
                            "error_type": type(exc).__name__,
                        }
                    write_json(control_file, control)
                record["control_answer"] = control["result"] or {}
                record["control_status"] = control["status"]
            canonical = directory / "canonical.png"
            if source.suffix.lower() == ".pdf":
                import pypdfium2 as pdfium

                with closing(pdfium.PdfDocument(source)) as pdf:
                    page = pdf[0]
                    width, height = page.get_size()
                    bitmap = page.render(
                        scale=min(
                            config["images"]["pdf_dpi"] / 72,
                            config["images"]["max_edge"] / max(width, height),
                        )
                    )
                    rendered = bitmap.to_pil().copy()
                    bitmap.close()
                    page.close()
            else:
                with Image.open(source) as original:
                    rendered = ImageOps.exif_transpose(original).copy()
            with rendered as image:
                image = ImageOps.exif_transpose(image)
                background = Image.new("RGBA", image.size, "white")
                background.alpha_composite(image.convert("RGBA"))
                rgb = background.convert("RGB")
                rgb.thumbnail(
                    (config["images"]["max_edge"],) * 2, Image.Resampling.LANCZOS
                )
                rgb.save(canonical)
            # Require identical overview JPEG pixels to the original direct condition.
            original_sent = load_images([source], config["images"])[0].metadata[
                "sent_sha256"
            ]
            if (
                load_images([canonical], config["images"])[0].metadata["sent_sha256"]
                != original_sent
            ):
                raise ValueError(
                    "Guided overview differs from original direct overview"
                )
            job_file = directory / "job.json"
            if job_file.exists():
                job = EvidenceJob(
                    json.loads(job_file.read_text(encoding="utf-8"))["directory"],
                    pipeline.settings,
                )
            else:
                job = pipeline.prepare(canonical)
                write_json(job_file, {"directory": str(job.directory)})
            trace = answer_guided(
                service,
                job,
                case["question"],
                config["images"],
                unit=case["unit"],
                question_type=case["type"],
                out=directory / "guided-trace.json",
            )
            record.update(
                status=trace["final"]["status"],
                answer=trace["final"]["result"] or {},
                decision=trace["decision"],
                usage=trace["usage"],
                job_id=job.directory.name,
                matched_overview_sha256=original_sent,
                original_input_sha256=case["input_sha256"],
            )
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            # The provider transport sanitizes errors; workers receive no secrets.
            record.update(
                status="error", answer={}, error_type=type(exc).__name__, error=str(exc)
            )
        record["seconds"] = time.perf_counter() - start
        records.append(record)
        write_json(predictions_path, records)
        print(
            json.dumps(
                {
                    "completed": len(records),
                    "total": len(selected),
                    "id": case["id"][:12],
                    "status": record["status"],
                    "seconds": record["seconds"],
                }
            ),
            flush=True,
        )
    # Evaluation boundary: model attempts are complete before reading any answer labels.
    from planproof_feedback.scoring import score_answer

    labels = pack / "evaluator/answers.json"
    if sha256(labels) != manifest["answers_sha256"]:
        raise ValueError("Changed reference answers")
    answers = json.loads(labels.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if isinstance(baseline, dict):
        baseline = baseline["cases"]
    direct = {r["id"]: r for r in baseline if r.get("mode") == "direct"}
    rows = []
    for record in records:
        before = direct.get(record["id"])
        rows.append(
            {
                **record,
                "guided_score": score_answer(answers[record["id"]], record["answer"]),
                "control_score": score_answer(
                    answers[record["id"]], record.get("control_answer", {})
                )
                if args.paired_direct
                else None,
                "direct_answer": before.get("answer") if before else None,
                "direct_score": score_answer(
                    answers[record["id"]], before.get("answer", {})
                )
                if before
                else None,
                "baseline_available": before is not None,
            }
        )
    matched = [r for r in rows if r["baseline_available"]]
    result = {
        "binding": binding,
        "cases": rows,
        "summary": {
            "attempted": len(rows),
            "matched": len(matched),
            "direct_correct": sum(r["direct_score"]["correct"] for r in matched),
            "guided_correct": sum(r["guided_score"]["correct"] for r in matched),
            "guided_errors": sum(r["status"] != "ok" for r in rows),
            "control_correct": sum(
                bool(r.get("control_score", {}).get("correct")) for r in rows
            )
            if args.paired_direct
            else None,
            "control_failures": sum(r.get("control_status") != "ok" for r in rows)
            if args.paired_direct
            else None,
            "improved": sum(
                not r["direct_score"]["correct"] and r["guided_score"]["correct"]
                for r in matched
            ),
            "regressed": sum(
                r["direct_score"]["correct"] and not r["guided_score"]["correct"]
                for r in matched
            ),
        },
    }
    write_json(out / "report.json", result)
    print(json.dumps(result["summary"], indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Apply to BOTH new direct and guided calls, preserving old baseline",
    )
    parser.add_argument(
        "--paired-direct",
        action="store_true",
        help="Run a fresh direct control with matching generation settings",
    )
    parser.add_argument(
        "--include-pdf",
        action="store_true",
        help="Include single-page PDFs with identical overview pixels",
    )
    run(parser.parse_args())
