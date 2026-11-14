from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import write_json
from .benchmark import prepare, run
from .client import NemotronClient
from .config import DEFAULT_CONFIG, get_key, load_config
from .images import load_images
from .regions import load_region
from .service import create_service


def preserve_comparison(config):
    from .preservation import backup
    from .reporting import build_report

    try:
        report = build_report(config)
    except Exception as exc:
        receipt = backup(config)
        raise RuntimeError(
            f"Visual report failed ({exc}); all data backed up and verified at {receipt['archive']}"
        ) from None
    receipt = backup(config)
    return {"visual_report": report, "verified_backup": receipt}


def input_value(value):
    if value is None:
        return None
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    try:
        return json.loads(value)
    except ValueError:
        return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Planproof: Nemotron visual feedback and Stoa baseline"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Authenticate and verify the configured model ID")
    for name in ("feedback", "self-feedback", "answer"):
        command = commands.add_parser(name)
        command.add_argument("--question", required=True)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--image", type=Path, action="append")
        source.add_argument("--region", type=Path, help="get_region's region.json")
        command.add_argument(
            "--candidate", required=name != "answer", help="Text, JSON or @draft.json"
        )
        command.add_argument(
            "--evidence", type=Path, help="Inference observations only; never labels"
        )
        command.add_argument("--out", type=Path)
        if name == "answer":
            command.add_argument("--self-review", action="store_true")
            command.add_argument("--unit")
            command.add_argument("--type", dest="question_type")
    preparation = commands.add_parser(
        "benchmark-prepare", help="Fetch the fixed reference-only sample; no model calls"
    )
    preparation.add_argument("--sample-size", type=int)
    preparation.add_argument("--seed")
    preparation.add_argument("--pack-dir", type=Path)
    preparation.add_argument("--questions", type=Path)
    preparation.add_argument("--inventory", type=Path)
    bench = commands.add_parser(
        "benchmark", help="Run zero-shot direct and/or self-feedback baseline"
    )
    bench.add_argument("--mode", choices=("direct", "self-feedback", "both"), default="direct")
    bench.add_argument("--pack-dir", type=Path)
    bench.add_argument("--out", type=Path)
    bench.add_argument("--resume", action="store_true")
    report = commands.add_parser(
        "report", help="Build the offline visual comparison; no inference calls"
    )
    report.add_argument("--out", type=Path)
    commands.add_parser(
        "backup", help="Preserve all local benchmark data in a verified immutable ZIP"
    )
    verify = commands.add_parser("verify-backup", help="Verify every payload in a saved local ZIP")
    verify.add_argument("archive", type=Path)
    imported = commands.add_parser(
        "benchmark-import", help="Preserve and score a future Gemini/other run"
    )
    imported.add_argument("--input", type=Path, required=True)
    imported.add_argument("--label", default="Gemini Carbon")
    imported.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "check":
            provider = config["providers"][config["routes"]["feedback"]]
            result = NemotronClient(
                provider, get_key(config, provider), audit_dir=config.get("audit_dir")
            ).check()
        elif args.command == "report":
            from .reporting import build_report

            result = build_report(config, destination=args.out)
        elif args.command in ("backup", "verify-backup"):
            from .preservation import backup, verify_backup

            result = backup(config) if args.command == "backup" else verify_backup(args.archive)
        elif args.command == "benchmark-import":
            from .reporting import import_run

            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            destination = args.out or Path(config["benchmark"]["pack_dir"]).parent / "runs" / (
                "import-" + stamp
            )
            result = import_run(config, args.input, label=args.label, destination=destination)
            result.update(preserve_comparison(config))
        elif args.command == "benchmark-prepare":
            for argument, field in (
                ("sample_size", "sample_size"),
                ("seed", "seed"),
                ("pack_dir", "pack_dir"),
                ("questions", "question_file"),
                ("inventory", "inventory_file"),
            ):
                value = getattr(args, argument)
                if value is not None:
                    config["benchmark"][field] = (
                        str(value.resolve()) if isinstance(value, Path) else value
                    )
            result = prepare(config)
            result.update(preserve_comparison(config))
        elif args.command == "benchmark":
            if args.pack_dir:
                config["benchmark"]["pack_dir"] = str(args.pack_dir.resolve())
            service = create_service(config)
            modes = ("direct", "self-feedback") if args.mode == "both" else (args.mode,)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            destination = args.out or Path(config["benchmark"]["pack_dir"]).parent / "runs" / stamp
            try:
                result = run(config, service, modes=modes, run_dir=destination, resume=args.resume)
            except (Exception, KeyboardInterrupt):
                # Retain partial predictions and raw calls as well as completed runs.
                try:
                    preserved = preserve_comparison(config)
                    print(json.dumps(preserved, indent=2), file=sys.stderr)
                except Exception as preservation_error:
                    print(
                        f"Data remains on disk; report/backup could not finish: {preservation_error}",
                        file=sys.stderr,
                    )
                raise
            result.update(preserve_comparison(config))
        else:
            if args.region:
                images, evidence = load_region(args.region, config["images"])
            else:
                images, evidence = load_images(args.image, config["images"]), None
            if args.evidence:
                evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
            service = create_service(config)
            if args.command == "feedback":
                result = service.review(
                    args.question, images, candidate=input_value(args.candidate), evidence=evidence
                )
            elif args.command == "self-feedback":
                result = service.critique_draft(
                    args.question, images, draft=input_value(args.candidate), evidence=evidence
                )
            elif args.self_review:
                result = service.answer_with_self_feedback(
                    args.question, images, unit=args.unit, question_type=args.question_type
                )
            else:
                result = service.answer(
                    args.question, images, unit=args.unit, question_type=args.question_type
                )
            if args.out:
                write_json(args.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if args.command in ("feedback", "self-feedback", "answer"):
            final = result.get("final", result)
            return 0 if final["status"] == "ok" else 2
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, zipfile.BadZipFile) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
