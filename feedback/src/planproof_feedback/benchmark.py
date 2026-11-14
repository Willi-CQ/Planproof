from __future__ import annotations

import csv
import json
import random
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .archive import RangeArchive
from .artifacts import fingerprint, safe_path, write_json
from .client import ProviderError
from .images import load_images, sha256
from .service import PROMPT_VERSION


SCHEMA = "planproof-stoa-random-v1"


def archive_member(relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError("Unsafe archive member")
    return "junction-dataset/" + relative


def select_cases(question_file: Path, inventory_file: Path, *, count: int, seed: str):
    if count < 1:
        raise ValueError("Sample size must be positive")
    if "final" in question_file.name.lower():
        raise ValueError("Final Stoa partition is sealed; use evaluation/reference only")
    with inventory_file.open(encoding="utf-8-sig", newline="") as stream:
        inventory = {row["Path"] for row in csv.DictReader(stream)}
    alternatives = {}
    rank = {".png": 0, ".jpg": 1, ".jpeg": 2, ".pdf": 3}
    for member in sorted(inventory):
        path = PurePosixPath(member)
        if path.suffix.lower() in rank:
            alternatives.setdefault(str(path.with_suffix("")), []).append(member)
    cases, answers, skipped, seen = [], {}, Counter(), set()
    total = 0
    with question_file.open(encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            member = archive_member(row["path"])
            if PurePosixPath(member).suffix.lower() in rank and member in inventory:
                selected = member
            elif PurePosixPath(member).suffix.lower() not in rank:
                # No inferred IFC-level/model joins. Only exact-basename alternate formats.
                options = alternatives.get(str(PurePosixPath(member).with_suffix("")), [])
                if not options:
                    skipped["no_exact_basename_2d_source"] += 1
                    continue
                selected = min(options, key=lambda p: (rank[PurePosixPath(p).suffix.lower()], p))
            else:
                skipped["source_not_in_inventory"] += 1
                continue
            if row.get("type") not in ("count", "area", "length", "volume", "text"):
                skipped["unsupported_question_type"] += 1
                continue
            identity = {k: row.get(k) for k in ("path", "question", "unit", "type")}
            case_id = fingerprint(identity)
            if case_id in seen:
                skipped["duplicate_question"] += 1
                continue
            seen.add(case_id)
            source = PurePosixPath(row["path"]).parts[1]
            relative = selected.removeprefix("junction-dataset/")
            case = {
                "id": case_id,
                **identity,
                "source": source,
                "archive_member": selected,
                "input_path": "inputs/" + relative,
                "reference_line": index + 1,
            }
            cases.append(case)
            answers[case_id] = {k: row.get(k) for k in ("answer", "unit", "type")}
    if count > len(cases):
        raise ValueError(f"Requested {count} cases but only {len(cases)} are eligible")
    # Fixed seed and sorted identities: independent of answer values and model predictions.
    sample = random.Random(seed).sample(sorted(cases, key=lambda x: x["id"]), count)
    return (
        sample,
        {x["id"]: answers[x["id"]] for x in sample},
        {
            "reference_rows": total,
            "eligible_unique_questions": len(cases),
            "excluded": dict(skipped),
            "sample_size": count,
            "seed": seed,
            "sampling": "uniform random unique questions from reference-only 2D-resolvable population",
            "by_source": dict(Counter(x["source"] for x in sample)),
            "by_type": dict(Counter(x["type"] for x in sample)),
        },
    )


def prepare(config: dict, *, archive_factory=RangeArchive, progress=print):
    settings = config["benchmark"]
    root = Path(settings["pack_dir"])
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = verify_pack(root)
        if (
            manifest["population"]["seed"] != settings["seed"]
            or len(manifest["cases"]) != settings["sample_size"]
        ):
            raise ValueError("Existing fixed pack uses another seed/size; choose a new pack_dir")
        return {
            "status": "already_prepared",
            "manifest": str(manifest_path),
            "cases": len(manifest["cases"]),
        }
    questions, inventory = Path(settings["question_file"]), Path(settings["inventory_file"])
    cases, answers, population = select_cases(
        questions, inventory, count=settings["sample_size"], seed=settings["seed"]
    )
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive_url": settings["archive_url"],
        "question_file_sha256": sha256(questions),
        "inventory_sha256": sha256(inventory),
        "partition": "evaluation/reference; not final",
        "population": population,
        "cases": cases,
        "artifacts": {},
        "limitations": [
            "Zero-shot diagnostic, not the organizer's official evaluator.",
            "Reference-only sample; not an unseen final benchmark or a tuning set.",
            "PNG/JPEG/PDF only; unsupported 3D/BIM questions may require abstention.",
            "Publisher overlap and pretrained-model exposure are unknown.",
            "Text uses normalized exact matching; numeric uses README-reported rules.",
        ],
    }
    members = {case["archive_member"]: case["input_path"] for case in cases}
    members["junction-dataset/README.md"] = "notices/README.md"
    for source in population["by_source"]:
        members[f"junction-dataset/data/{source}/LICENSE.md"] = f"notices/{source}/LICENSE.md"
    root.mkdir(parents=True, exist_ok=True)
    with archive_factory(settings["archive_url"], budget=settings["max_download_bytes"]) as remote:
        with zipfile.ZipFile(remote) as archive:
            for index, (member, relative) in enumerate(sorted(members.items()), 1):
                info = archive.getinfo(member)
                if info.file_size > 64 * 1024 * 1024:
                    raise ValueError("Selected source exceeds the 64 MiB/member limit")
                destination = safe_path(root, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                # zipfile verifies the archive's CRC32 before returning bytes.
                content = archive.read(info)
                destination.write_bytes(content)
                manifest["artifacts"][relative] = {
                    "sha256": sha256(destination),
                    "bytes": len(content),
                    "archive_member": member,
                    "crc32": f"{info.CRC:08x}",
                }
                progress(f"Prepared {index}/{len(members)} source/notice files")
        manifest["archive_bytes"] = remote.size
        manifest["archive_etag"] = remote.etag
        manifest["range_bytes_downloaded"] = remote.downloaded
    # Keep reference answers in their own evaluator-only file, never in the cases manifest.
    answer_path = root / "evaluator/answers.json"
    write_json(answer_path, answers)
    manifest["answers_sha256"] = sha256(answer_path)
    for case in cases:
        case["input_sha256"] = manifest["artifacts"][case["input_path"]]["sha256"]
    write_json(manifest_path, manifest)
    verify_pack(root)
    return {
        "status": "prepared",
        "manifest": str(manifest_path),
        "cases": len(cases),
        "population": population,
    }


def verify_pack(root: Path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("partition") != "evaluation/reference; not final"
    ):
        raise ValueError("Invalid or unsealed benchmark pack")
    ids = [case["id"] for case in manifest["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate benchmark case IDs")
    for relative, artifact in manifest["artifacts"].items():
        if sha256(safe_path(root, relative)) != artifact["sha256"]:
            raise ValueError("Benchmark input/notice hash mismatch")
    for case in manifest["cases"]:
        if "answer" in case or sha256(safe_path(root, case["input_path"])) != case["input_sha256"]:
            raise ValueError("Benchmark case contains an answer or a changed input")
    return manifest


def run(config: dict, service, *, modes=("direct",), run_dir: Path, resume=False, progress=print):
    root = Path(config["benchmark"]["pack_dir"])
    manifest = verify_pack(root)
    if (
        not modes
        or len(set(modes)) != len(modes)
        or any(x not in ("direct", "self-feedback") for x in modes)
    ):
        raise ValueError("Modes must be unique direct and/or self-feedback")
    binding = {
        "manifest_sha256": sha256(root / "manifest.json"),
        "prompt_version": PROMPT_VERSION,
        "images": config["images"],
        "routes": config["routes"],
        "modes": list(modes),
        "providers": {
            name: {k: v for k, v in settings.items() if k != "api_key_env"}
            for name, settings in config["providers"].items()
        },
        "runtime_sha256": fingerprint(
            {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        ),
    }
    binding["execution_sha256"] = fingerprint(binding)
    run_dir.mkdir(parents=True, exist_ok=True)
    binding_file, predictions_file = run_dir / "run.json", run_dir / "predictions.json"
    predictions = []
    if binding_file.exists():
        if not resume:
            raise ValueError("Run already exists; use --resume or choose a new run directory")
        if json.loads(binding_file.read_text()) != binding:
            raise ValueError("Resume refused: input/model/prompt/render/settings changed")
        if predictions_file.exists():
            predictions = json.loads(predictions_file.read_text(encoding="utf-8"))
    else:
        write_json(binding_file, binding)
    completed = {(x["id"], x["mode"]) for x in predictions}
    if len(completed) != len(predictions) or not completed.issubset(
        {(c["id"], m) for c in manifest["cases"] for m in modes}
    ):
        raise ValueError("Resume predictions contain duplicate or unknown cases")
    for case in manifest["cases"]:
        for mode in modes:
            if (case["id"], mode) in completed:
                continue
            start = time.perf_counter()
            record = {
                "id": case["id"],
                "source": case["source"],
                "type": case["type"],
                "mode": mode,
            }
            try:
                images = load_images([safe_path(root, case["input_path"])], config["images"])
                if mode == "direct":
                    final = service.answer(
                        case["question"], images, unit=case["unit"], question_type=case["type"]
                    )
                    trace = {"final": final, "calls": [final], "stop_reason": "single_pass"}
                else:
                    trace = service.answer_with_self_feedback(
                        case["question"], images, unit=case["unit"], question_type=case["type"]
                    )
                    final = trace["final"]
                record.update(
                    {"status": final["status"], "answer": final["result"] or {}, "trace": trace}
                )
                usages = [call["completion"]["usage"] for call in trace["calls"]]
                record["usage"] = (
                    {
                        k: sum(u[k] for u in usages)
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                        if all(k in u for u in usages)
                    }
                    if all(usages)
                    else {}
                )
            except (ValueError, RuntimeError, OSError) as exc:
                if isinstance(exc, ProviderError) and exc.status_code in (401, 403, 404):
                    write_json(run_dir / "last-error.json", {**record, "error": str(exc)})
                    raise RuntimeError(
                        "Benchmark stopped: authentication/model/endpoint rejected. "
                        "Fix the key and use --resume, or use a new run if model/endpoint changes. "
                        "This case was not scored."
                    ) from None
                # ProviderError only exposes sanitized transport messages; no credentials persisted.
                record.update({"status": "error", "answer": {}, "error": str(exc), "usage": {}})
            record["latency_seconds"] = time.perf_counter() - start
            predictions.append(record)
            write_json(predictions_file, predictions)
            progress(
                f"Completed {len(predictions)}/{len(manifest['cases']) * len(modes)}: {record['status']}"
            )
    # The model stage has completed. Only now open answers and import the evaluator.
    from .scoring import score_answer, summarize

    answer_path = root / "evaluator/answers.json"
    if sha256(answer_path) != manifest["answers_sha256"]:
        raise ValueError("Evaluator answer hash mismatch")
    answers = json.loads(answer_path.read_text(encoding="utf-8"))
    scored = [
        {**record, "score": score_answer(answers[record["id"]], record["answer"])}
        for record in predictions
    ]
    report = {
        "status": "complete",
        "binding": binding,
        "population": manifest["population"],
        "limitations": manifest["limitations"],
        "metrics": summarize(scored),
        "cases": scored,
    }
    write_json(run_dir / "report.json", report)
    with (run_dir / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "id",
            "source",
            "type",
            "mode",
            "status",
            "correct",
            "scorable",
            "reason",
            "latency_seconds",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in scored:
            row = {**record, **record["score"]}
            writer.writerow({k: row[k] for k in fields})
    return {"report": str(run_dir / "report.json"), "metrics": report["metrics"]}
