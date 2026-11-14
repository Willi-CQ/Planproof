"""Run two fresh, isolated Nemotron conditions on an explicitly bound pilot.

Algorithm runtime is snapshotted before importing it, so another working chat
cannot change an experiment in flight. Labels never enter model inputs; scoring
follows all attempts. Explicit resume preserves completed provider attempts.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from build_paired_report import build


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary.write_text(text, encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))
    path.write_text(text, encoding="utf-8")
    temporary.unlink(missing_ok=True)


def snapshot(source, target):
    paths = [source / "feedback/config.toml", source / "backend/pipeline_worker.py"]
    for package in ("feedback", "deterministic", "ml"):
        paths.extend(
            p
            for p in (source / package / "src").rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    hashes = {}
    for path in paths:
        raw = path.read_bytes()
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        hashes[relative.as_posix()] = hashlib.sha256(raw).hexdigest()
    if any(digest(source / path) != sha for path, sha in hashes.items()):
        raise RuntimeError(
            "Runtime changed while snapshotting; create a fresh experiment"
        )
    save(target / "source-hashes.json", hashes)
    return hashes


class RecordingBackend:
    def __init__(self, backend, role, events, publish):
        self.backend, self.role, self.events, self.publish = (
            backend,
            role,
            events,
            publish,
        )

    def complete(self, messages):
        event = {"role": self.role, "status": "running", "started_at": now()}
        self.events.append(event)
        self.publish()
        start = time.perf_counter()
        try:
            completion = self.backend.complete(messages)
            event.update(status="complete", completion=completion)
            return completion
        except Exception as exc:
            event.update(status="error", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            event["seconds"] = time.perf_counter() - start
            self.publish()


def now():
    return datetime.now(timezone.utc).isoformat()


def unrequested_presentation_failure(record, directory):
    return (
        record["status"] == "error"
        and any(
            name in record.get("error", "")
            for name in ("paired-index.tmp", "report.json.tmp")
        )
        and not any((directory / "calls").glob("*/request.json"))
    )


def run(args):
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    out, pack, view = args.out.resolve(), args.pack.resolve(), args.view.resolve()
    resuming = args.resume and out.exists()
    if out.exists() and not resuming:
        raise ValueError(
            "Use a new output directory; previous evidence cannot be overwritten"
        )
    selection = json.loads(args.binding.read_text(encoding="utf-8"))
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    if digest(pack / "manifest.json") != selection["manifest_sha256"]:
        raise ValueError("Selection is not bound to this source pack")
    ids = selection["case_ids"]
    by_id = {case["id"]: case for case in manifest["cases"]}
    if not ids or len(ids) != selection["count"] or len(set(ids)) != len(ids):
        raise ValueError("Invalid pilot selection")
    selected = [by_id[i] for i in ids]
    if any(Path(case["input_path"]).suffix.lower() != ".png" for case in selected):
        raise ValueError("This controlled pilot requires original PNG sources")
    out.mkdir(parents=True, exist_ok=resuming)
    runtime = out / "runtime"
    if resuming:
        source_hashes = json.loads((runtime / "source-hashes.json").read_text())
        if any(digest(runtime / path) != sha for path, sha in source_hashes.items()):
            raise ValueError("Saved runtime snapshot changed")
    else:
        source_hashes = snapshot(args.runtime.resolve(), runtime)
    orchestration = (
        out / "orchestrator" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    orchestration.mkdir(parents=True)
    for name in (Path(__file__).name, "build_paired_report.py", "paired_pilot.html"):
        shutil.copy2(Path(__file__).with_name(name), orchestration / name)
    # Worker storage guards inspect both cwd and module roots. Launch from this
    # ignored experiment directory, not the parent workspace repository.
    os.chdir(out)
    sys.path.insert(0, str(runtime / "feedback/src"))
    from PIL import Image, ImageOps
    from planproof_feedback.config import load_config
    from planproof_feedback.images import load_images
    from planproof_feedback.pipeline import (
        EvidenceJob,
        EvidencePipeline,
        PipelineSettings,
    )
    from planproof_feedback.guided import ROUTE, answer_guided
    from planproof_feedback.service import create_service, PROMPT_VERSION

    config = load_config(runtime / "feedback/config.toml")
    config["env_file"] = str(args.env_file.resolve())
    config["providers"] = {
        "nemotron": {**config["providers"]["nemotron"], "enable_thinking": False},
        "nemotron-routing": {
            **config["providers"]["nemotron"],
            "enable_thinking": False,
            "max_tokens": 1024,
        },
    }
    config["routes"] = {
        "feedback": "nemotron",
        "self_feedback": "nemotron",
        "routing": "nemotron-routing",
    }
    config["providers"]["nemotron"]["max_tokens"] = 4096
    config["providers"]["nemotron"]["temperature"] = 0.0
    checkpoint = (
        args.runtime.resolve() / "models/baselines/v1/corrected-3500-global/detector.pt"
    )
    calibration = checkpoint.with_name("calibration.json")
    settings = PipelineSettings(
        str(args.geometry_python.resolve()),
        str(args.ml_python.resolve()),
        checkpoint,
        calibration,
        out / "jobs",
        "cpu",
    )
    if checkpoint.stat().st_size < 1000:
        raise ValueError("Actual frozen model weights are required")
    inputs = json.loads((view / "questions-for-model.json").read_text(encoding="utf-8"))
    input_map = {c["id"]: c for c in inputs["cases"]}
    state = {
        "schema": "planproof-paired-pilot-v1",
        "status": "running",
        "started_at": now(),
        "stage": "Preparing",
        "binding": {
            "case_ids": ids,
            "manifest_sha256": selection["manifest_sha256"],
            "selection_binding": str(args.binding.resolve()),
            "model": "nemotron-3.5-super-vl",
            "generation_settings": config["providers"],
            "images": config["images"],
            "answer_prompt_sha256": PROMPT_VERSION,
            "routing_prompt_sha256": hashlib.sha256(ROUTE.encode()).hexdigest(),
            "runtime_source_sha256": source_hashes,
            "checkpoint_sha256": digest(checkpoint),
            "calibration_sha256": digest(calibration),
            "order": "alternate direct-first and guided-first across questions",
            "labels_used_for_inference": False,
            "limits": "Eight selected Stoa reference cases, all CubiCasa counts; not a held-out or official Stoa score. Guided includes extra routing call and algorithm time.",
        },
        "cases": [],
    }
    for case in selected:
        entry = input_map[case["id"]]
        state["cases"].append(
            {
                "id": case["id"],
                "question": case["question"],
                "source": case["source"],
                "type": case["type"],
                "unit": case["unit"],
                "original": entry["original_path"],
                "preview": entry["images"][0]["asset_path"],
                "input_sha256": case["input_sha256"],
                "direct": {"status": "pending", "events": []},
                "guided": {"status": "pending", "events": []},
            }
        )
    if not resuming:
        save(out / "binding.json", state["binding"])
    reused = None
    if args.direct_from:
        prior = json.loads(
            (args.direct_from / "report.json").read_text(encoding="utf-8")
        )
        expected = state["binding"]
        for key in ("case_ids", "manifest_sha256", "answer_prompt_sha256", "images"):
            if prior["binding"][key] != expected[key]:
                raise ValueError("Previous direct condition has a different " + key)
        if (
            prior["binding"]["generation_settings"]["nemotron"]
            != expected["generation_settings"]["nemotron"]
        ):
            raise ValueError("Previous direct generation settings differ")
        reused = {r["id"]: r for r in prior["cases"]}
        state["binding"]["direct_condition_reused_from"] = str(args.direct_from)
        state["binding"]["direct_report_sha256"] = digest(
            args.direct_from / "report.json"
        )
        state["binding"]["order"] = (
            "Direct recorded first; guided after local worker launch repair. Temporal ordering is a possible confound."
        )
        if not resuming:
            save(out / "binding.json", state["binding"])
    if resuming:
        prior = json.loads((out / "report.json").read_text(encoding="utf-8"))
        if prior["binding"] != state["binding"]:
            raise ValueError("Resume settings differ from the saved experiment")
        state = prior
        save(orchestration / "checkpoint-before-resume.json", prior)
        state["status"] = "running"
        state.pop("summary", None)
        for row in state["cases"]:
            row.pop("reference", None)
            for condition in ("direct", "guided"):
                row[condition].pop("score", None)
        state.setdefault("execution_history", []).append(
            {"resumed_at": now(), "runner_sha256": digest(Path(__file__))}
        )
    # Preserve both earlier full results and the eight-question-only presentation.
    if (view / "index.html").exists() and not (view / "questions-only.html").exists():
        shutil.copy2(view / "index.html", view / "questions-only.html")

    def publish():
        state["updated_at"] = now()
        save(out / "report.json", state)
        try:
            build(out, view)
        except OSError as exc:
            # A browser/IDE file lock must never cancel an inference call.
            state["view_warning"] = {"error": str(exc), "observed_at": now()}
            save(out / "view-warning.json", state["view_warning"])

    publish()
    pipeline = EvidencePipeline(settings)
    for number, (case, row) in enumerate(zip(selected, state["cases"]), 1):
        source = pack / case["input_path"]
        if digest(source) != case["input_sha256"]:
            raise ValueError("Original source bytes changed")
        folder = out / case["id"]
        folder.mkdir(exist_ok=resuming)
        canonical = folder / "canonical.png"
        with Image.open(source) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGBA")
            background = Image.new("RGBA", rgb.size, "white")
            background.alpha_composite(rgb)
            rgb = background.convert("RGB")
            rgb.thumbnail((config["images"]["max_edge"],) * 2, Image.Resampling.LANCZOS)
            rgb.save(canonical)
        original_images = load_images([source], config["images"])
        if (
            load_images([canonical], config["images"])[0].metadata["sent_sha256"]
            != original_images[0].metadata["sent_sha256"]
        ):
            raise ValueError("Direct and guided overview pixels differ")
        row["overview_sent_sha256"] = original_images[0].metadata["sent_sha256"]
        order = ("direct", "guided") if number % 2 else ("guided", "direct")
        row["order"] = list(order)
        if reused is not None:
            row["order"] = ["saved_current_direct", "guided"]
        for condition in order:
            existing = row[condition]
            recovering_local = False
            if resuming and existing["status"] not in ("pending", "running"):
                # Preserve actual model attempts, including unsuccessful ones.
                # Only retry a local presentation failure that made no request.
                if not unrequested_presentation_failure(existing, folder / condition):
                    continue
                recovering_local = True
                save(folder / condition / "local-presentation-failure.json", existing)
                row[condition] = {
                    "status": "pending",
                    "events": [],
                    "job_directory": existing.get("job_directory"),
                    "algorithm_seconds": existing.get("algorithm_seconds", 0),
                    "local_failure_recovered": True,
                }
            if condition == "direct" and reused is not None and not recovering_local:
                previous = reused[case["id"]]
                if (
                    previous["input_sha256"] != row["input_sha256"]
                    or previous["overview_sent_sha256"] != row["overview_sent_sha256"]
                ):
                    raise ValueError("Previous direct source pixels differ")
                row["direct"] = previous["direct"]
                row["direct"].pop("score", None)
                row["direct"]["reused_from"] = str(
                    args.direct_from / case["id"] / "direct"
                )
                shutil.copytree(
                    args.direct_from / case["id"] / "direct", folder / "direct"
                )
                publish()
                continue
            record = row[condition]
            state["stage"] = f"Question {number}/{len(selected)} · {condition}"
            record["status"] = "running"
            record["started_at"] = now()
            condition_dir = folder / condition
            condition_dir.mkdir(exist_ok=resuming)
            config["audit_dir"] = str(condition_dir / "calls")
            service = create_service(config)
            service.backends = {
                role: RecordingBackend(
                    backend,
                    "routing" if role == "routing" else "answer",
                    record["events"],
                    publish,
                )
                for role, backend in service.backends.items()
            }
            start = time.perf_counter()
            publish()
            try:
                if condition == "direct":
                    final = service.answer(
                        case["question"],
                        original_images,
                        unit=case["unit"],
                        question_type=case["type"],
                    )
                else:
                    algo_start = time.perf_counter()
                    if record.get("job_directory"):
                        job = EvidenceJob(record["job_directory"], settings)
                        receipt = json.loads(
                            (job.directory / "pipeline-receipt.json").read_text()
                        )
                        if (
                            receipt["input_sha256"] != digest(canonical)
                            or receipt["model_sha256"]
                            != state["binding"]["checkpoint_sha256"]
                        ):
                            raise ValueError(
                                "Cached algorithm evidence has another source or model"
                            )
                        record["algorithm_cache_reused"] = True
                    else:
                        job = pipeline.prepare(canonical)
                        record["algorithm_seconds"] = time.perf_counter() - algo_start
                    record["job_directory"] = str(job.directory)
                    record["catalog"] = job.catalog
                    record["catalog_sha256"] = hashlib.sha256(
                        json.dumps(job.catalog, sort_keys=True).encode()
                    ).hexdigest()
                    publish()
                    trace = answer_guided(
                        service,
                        job,
                        case["question"],
                        config["images"],
                        unit=case["unit"],
                        question_type=case["type"],
                        out=condition_dir / "guided-trace.json",
                    )
                    record["decision"] = trace["decision"]
                    final = trace["final"]
                record.update(status=final["status"], final=final)
            except Exception as exc:
                record.update(
                    status="error", error_type=type(exc).__name__, error=str(exc)
                )
            record["seconds"] = time.perf_counter() - start
            if record.get("algorithm_cache_reused"):
                record["seconds"] += record["algorithm_seconds"]
            record["finished_at"] = now()
            save(condition_dir / "result.json", record)
            publish()
            print(
                json.dumps(
                    {
                        "question": number,
                        "condition": condition,
                        "status": record["status"],
                        "seconds": record["seconds"],
                    }
                ),
                flush=True,
            )
    # Only now open reference labels. No labels are passed to either condition.
    from planproof_feedback.scoring import score_answer

    labels_path = pack / "evaluator/answers.json"
    if digest(labels_path) != manifest["answers_sha256"]:
        raise ValueError("Reference labels changed")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    for row in state["cases"]:
        row["reference"] = labels[row["id"]]
        for condition in ("direct", "guided"):
            side = row[condition]
            side["score"] = score_answer(
                labels[row["id"]], side.get("final", {}).get("result") or {}
            )
    state["summary"] = summarize(state["cases"])
    state.update(status="complete", stage="Complete", finished_at=now())
    publish()
    print(json.dumps(state["summary"], indent=2), flush=True)


def summarize(rows):
    result = {}
    for condition in ("direct", "guided"):
        sides = [row[condition] for row in rows]
        result[condition] = {
            "correct": sum(bool(s["score"]["correct"]) for s in sides),
            "total": len(rows),
            "failures": sum(s["status"] != "ok" for s in sides),
            "seconds": sum(s.get("seconds", 0) for s in sides),
            "calls": sum(len(s["events"]) for s in sides),
            "answered": sum(
                (s.get("final", {}).get("result") or {}).get("status") == "answered"
                for s in sides
            ),
            "abstained": sum(
                (s.get("final", {}).get("result") or {}).get("status") == "abstain"
                for s in sides
            ),
            "algorithm_seconds": sum(s.get("algorithm_seconds", 0) for s in sides),
            "model_seconds": sum(
                e.get("seconds", 0) for s in sides for e in s["events"]
            ),
        }
    result["guided_wins"] = sum(
        r["guided"]["score"]["correct"] and not r["direct"]["score"]["correct"]
        for r in rows
    )
    result["direct_wins"] = sum(
        r["direct"]["score"]["correct"] and not r["guided"]["score"]["correct"]
        for r in rows
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "runtime",
        "pack",
        "binding",
        "out",
        "view",
        "env-file",
        "ml-python",
        "geometry-python",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--direct-from",
        type=Path,
        help="Reuse an exact matching direct condition after a local worker failure",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the same immutable experiment without repeating model attempts",
    )
    run(parser.parse_args())
