"""Offline comparison artifacts. This is evaluator-side code, never model context."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import fingerprint, safe_path, write_json
from .benchmark import verify_pack
from .images import load_images, sha256


def dataset_identity(manifest):
    return fingerprint(
        [
            {key: case[key] for key in ("id", "question", "type", "unit", "input_sha256")}
            for case in sorted(manifest["cases"], key=lambda c: c["id"])
        ]
    )


def references(root, manifest):
    path = root / "evaluator/answers.json"
    if sha256(path) != manifest["answers_sha256"]:
        raise ValueError("Evaluator answer hash mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def _assets(config, manifest, root, destination):
    index_file = destination / "image-index.json"
    binding = {"dataset_sha256": dataset_identity(manifest), "images": config["images"]}
    if index_file.exists():
        saved = json.loads(index_file.read_text(encoding="utf-8"))
        if saved["binding"] == binding and all(
            sha256(safe_path(destination, image["asset_path"])) == image["sent_sha256"]
            for images in saved["cases"].values()
            for image in images
        ):
            return saved["cases"]
    result = {}
    for case in manifest["cases"]:
        result[case["id"]] = []
        for image in load_images([safe_path(root, case["input_path"])], config["images"]):
            relative = f"assets/{case['id']}-{image.metadata['sent_sha256'][:16]}.jpg"
            path = safe_path(destination, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(image.data_url.split(",", 1)[1]))
            result[case["id"]].append({**image.metadata, "asset_path": relative})
    write_json(index_file, {"binding": binding, "cases": result})
    return result


def _run(config, manifest, root, directory, answer_key):
    from .scoring import score_answer, summarize

    report_file, predictions_file = directory / "report.json", directory / "predictions.json"
    if report_file.exists():
        report = json.loads(report_file.read_text(encoding="utf-8"))
        records, binding = report["cases"], report["binding"]
        label = report.get("label")
    elif predictions_file.exists():
        records = json.loads(predictions_file.read_text(encoding="utf-8"))
        binding = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        label = None
    else:
        return []
    if binding.get("manifest_sha256") != sha256(root / "manifest.json"):
        # Keep runs on disk, but never quietly pool results from another cohort.
        return []
    known = {case["id"]: case for case in manifest["cases"]}
    groups = {}
    for record in records:
        if record["id"] not in known:
            raise ValueError("Saved run contains an unknown case ID")
        mode = record["mode"]
        group = groups.setdefault(mode, [])
        case = known[record["id"]]
        group.append(
            {
                **record,
                "input_sha256": case["input_sha256"],
                "score": score_answer(
                    answer_key[record["id"]],
                    record.get("answer") if record.get("status") == "ok" else None,
                ),
            }
        )
    result = []
    for mode, rows in groups.items():
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Saved run has duplicate case IDs within a mode")
        provider_id = binding.get("routes", {}).get("feedback")
        provider = binding.get("providers", {}).get(provider_id, {})
        model = provider.get("model", "unknown")
        result.append(
            {
                "id": directory.name + ":" + mode,
                "label": label or ("Nemotron" if "nemotron" in model.lower() else model),
                "model": model,
                "mode": mode,
                "status": "complete" if len(rows) == len(known) else "partial",
                "records": rows,
                "metrics": summarize(rows),
                "binding": binding,
                "run_path": str(directory),
                "image_policy_matches": binding.get("images") == config["images"],
            }
        )
    return result


def build_report(config: dict, *, destination: Path | None = None):
    root = Path(config["benchmark"]["pack_dir"])
    manifest = verify_pack(root)
    answer_key = references(root, manifest)
    destination = destination or Path(config["report_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    assets = _assets(config, manifest, root, destination)
    credits = {}
    notices = {}
    for source in manifest["population"]["by_source"]:
        notice = root / "notices" / source / "LICENSE.md"
        notices[source] = notice.read_text(encoding="utf-8")
        credits[source] = (
            notices[source].split("Credit:", 1)[-1].split("\n\n", 1)[0].replace("\n", " ").strip()
        )
    runs_root = root.parent / "runs"
    runs = []
    if runs_root.exists():
        for directory in sorted(runs_root.iterdir()):
            if directory.is_dir():
                runs.extend(_run(config, manifest, root, directory, answer_key))
    data = {
        "schema": "planproof-comparison-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "sha256": dataset_identity(manifest),
            "manifest_sha256": sha256(root / "manifest.json"),
            "population": manifest["population"],
            "image_policy": config["images"],
            "limitations": manifest["limitations"],
        },
        "cases": [
            {
                **{
                    k: case[k] for k in ("id", "question", "unit", "type", "source", "input_sha256")
                },
                "reference": answer_key[case["id"]]["answer"],
                "images": assets[case["id"]],
                "original_path": Path(
                    os.path.relpath(root / case["input_path"], destination)
                ).as_posix(),
                "credit": next(
                    (
                        line.split("|")[4].strip() + " · " + line.split("|")[3].strip()
                        for line in notices[case["source"]].splitlines()
                        if line.startswith("|")
                        and Path(case["input_path"]).name in line
                        and len(line.split("|")) > 5
                    ),
                    credits[case["source"]],
                ),
                "license_notice": Path(
                    os.path.relpath(root / "notices" / case["source"] / "LICENSE.md", destination)
                ).as_posix(),
            }
            for case in manifest["cases"]
        ],
        "runs": runs,
        "storage": {"pack": str(root), "runs": str(runs_root), "audit": config.get("audit_dir")},
    }
    write_json(destination / "comparison.json", data)
    # Reusable Gemini/other-runner input contract. Reference answers are deliberately absent.
    write_json(
        destination / "questions-for-model.json",
        {
            "schema": "planproof-questions-v1",
            "dataset_sha256": data["dataset"]["sha256"],
            "image_policy": config["images"],
            "cases": [
                {
                    k: case[k]
                    for k in (
                        "id",
                        "question",
                        "unit",
                        "type",
                        "source",
                        "input_sha256",
                        "images",
                        "original_path",
                    )
                }
                for case in data["cases"]
            ],
        },
    )
    web = Path(__file__).parent / "web"
    for name in ("report.css", "report.js"):
        shutil.copyfile(web / name, destination / name)
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = (web / "index.html").read_text(encoding="utf-8")
    (destination / "index.html").write_text(
        template.replace("__REPORT_DATA__", payload), encoding="utf-8"
    )
    return {
        "report": str(destination / "index.html"),
        "data": str(destination / "comparison.json"),
        "dataset_sha256": data["dataset"]["sha256"],
        "runs": len(runs),
        "cases": len(data["cases"]),
    }


def import_run(config: dict, source: Path, *, label: str, destination: Path):
    """Persist a future Gemini/other run and score it with the same frozen evaluator."""
    from .scoring import score_answer, summarize

    root = Path(config["benchmark"]["pack_dir"])
    manifest = verify_pack(root)
    raw = source.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema") != "planproof-predictions-v1":
        raise ValueError("Expected planproof-predictions-v1; see feedback/COMPARISON.md")
    if value.get("dataset_sha256") != dataset_identity(manifest):
        raise ValueError("Comparison refused: different frozen question/input set")
    cases = {case["id"]: case for case in manifest["cases"]}
    if not isinstance(value.get("records"), list) or not isinstance(
        value.get("mode", "direct"), str
    ):
        raise ValueError("Imported records must be a list and mode must be a string")
    records, seen = [], set()
    for row in value["records"]:
        if not isinstance(row, dict):
            raise ValueError("Imported case records must be objects")
        case_id = row.get("id")
        if not isinstance(case_id, str):
            raise ValueError("Imported question IDs must be strings")
        if case_id not in cases or case_id in seen:
            raise ValueError("Imported results contain unknown or duplicate case IDs")
        seen.add(case_id)
        case = cases[case_id]
        if row.get("input_sha256") != case["input_sha256"]:
            raise ValueError("Comparison refused: source drawing hash mismatch")
        answer = row.get("answer", {})
        status = row.get("status", "ok")
        usage = row.get("usage") or {}
        if not isinstance(usage, dict) or any(
            key in usage
            and (not isinstance(usage[key], int) or isinstance(usage[key], bool) or usage[key] < 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            raise ValueError("Imported token usage must contain nonnegative integers")
        latency = row.get("latency_seconds")
        if latency is not None and (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise ValueError("Imported latency must be a finite nonnegative number, or omitted")
        if status == "ok":
            from .service import scalar

            if not isinstance(answer, dict) or answer.get("status") not in ("answered", "abstain"):
                raise ValueError(
                    "Successful imported rows require a structured answer or abstention"
                )
            if not scalar(answer.get("answer")) or (
                answer["status"] == "answered" and answer.get("answer") is None
            ):
                raise ValueError(
                    "Imported answers require a finite scalar value or explicit abstention"
                )
            if answer["status"] == "abstain" and answer.get("answer") is not None:
                raise ValueError("Imported abstentions must have answer=null")
        else:
            answer = {}  # Original bytes retain any stale answer; a failed request cannot score as correct.
        records.append(
            {
                **{key: value for key, value in row.items() if key != "latency_seconds"},
                "answer": answer,
                "source": case["source"],
                "type": case["type"],
                "mode": value.get("mode", "direct"),
                "status": status,
                "usage": usage,
                **({"latency_seconds": latency} if latency is not None else {}),
            }
        )
    if not records:
        raise ValueError("Imported run has no case records")
    answer_key = references(root, manifest)
    scored = [
        {**row, "score": score_answer(answer_key[row["id"]], row.get("answer"))} for row in records
    ]
    binding = {
        "manifest_sha256": sha256(root / "manifest.json"),
        "images": value.get("image_policy"),
        "routes": {"feedback": "imported"},
        "providers": {"imported": {"model": value.get("model", label)}},
        "imported_source_sha256": hashlib.sha256(raw).hexdigest(),
    }
    if destination.exists():
        raise ValueError(
            "Import destination already exists; previous results are never overwritten"
        )
    destination.mkdir(parents=True)
    (destination / "original-import.json").write_bytes(raw)
    write_json(
        destination / "report.json",
        {
            "status": "complete" if len(scored) == len(cases) else "partial",
            "label": label,
            "binding": binding,
            "cases": scored,
            "metrics": summarize(scored),
        },
    )
    write_json(
        destination / "import-receipt.json",
        {
            "source_sha256": binding["imported_source_sha256"],
            "dataset_sha256": dataset_identity(manifest),
            "records": len(records),
            "policy": "Original source bytes retained; scores recomputed using frozen reference answers",
        },
    )
    return {"imported": str(destination), "records": len(records), "label": label}
