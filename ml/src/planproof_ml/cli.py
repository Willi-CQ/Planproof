from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from .schema import read_json, write_json, sha256
from .storage import ensure_artifact_directory


def parser():
    p = argparse.ArgumentParser(
        prog="planproof-ml", description="Floor-plan detector preparation and GPU training"
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("sources", help="List original dataset sources without downloading")
    q = sub.add_parser(
        "status", help="Read run health, validation trend, throughput and session cost estimate"
    )
    q.add_argument("--run", type=Path, required=True)
    q.add_argument("--hourly-rate", type=float)
    q = sub.add_parser("compare", help="Select experiments using matching validation cohorts")
    q.add_argument("--baseline", type=Path, required=True)
    q.add_argument("--candidate", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("doctor", help="Inspect environment without downloading data/weights")
    q.add_argument("--require-gpu", action="store_true")
    q.add_argument("--smoke", action="store_true")
    q = sub.add_parser("fetch", help="Download publisher archives explicitly on the GPU host")
    q.add_argument("--source", required=True)
    q.add_argument("--data-root", type=Path, required=True)
    q.add_argument("--url", action="append", help="Explicit publisher URL after an access request")
    q.add_argument("--extract", action="store_true")
    q = sub.add_parser("extract")
    q.add_argument("archive", type=Path)
    q.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--source", required=True)
    q.add_argument("--raw", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--class-map", type=Path)
    q.add_argument("--trust-pickle", action="store_true")
    q.add_argument("--limit", type=int, help="Explicit diagnostic import only; not a full training corpus")
    q.add_argument("--workers", type=int, default=1, help="Bounded CPU import workers (1-16)")
    q = sub.add_parser("audit")
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--deep", action="store_true")
    q.add_argument("--near", action="store_true")
    q.add_argument("--out", type=Path)
    q = sub.add_parser("inspect", help="Render prepared ground-truth overlays for source-adapter review")
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--per-source", type=int, default=8)
    q = sub.add_parser("preflight")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--manifest", type=Path, required=True)
    q = sub.add_parser("regroup", help="Apply confirmed duplicate groups before starting training")
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--mapping", type=Path, required=True)
    q = sub.add_parser("train")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--run", type=Path, required=True)
    q.add_argument("--device", default="cuda")
    q.add_argument("--resume", type=Path)
    q.add_argument(
        "--init-weights",
        type=Path,
        help="Initialize a new dataset/run from compatible model weights; reset optimizer and schedule",
    )
    q.add_argument(
        "--max-minutes",
        type=float,
        help="Soft session time cap; finish update, validate and save before exiting",
    )
    q.add_argument(
        "--max-steps",
        type=int,
        help="Stop after this absolute optimizer step; retain full schedule for resume",
    )
    for name in ("evaluate", "calibrate"):
        q = sub.add_parser(name)
        q.add_argument("--checkpoint", type=Path, required=True)
        q.add_argument("--manifest", type=Path, required=True)
        q.add_argument("--out", type=Path, required=True)
        q.add_argument("--device", default="cuda")
        if name == "evaluate":
            q.add_argument("--split", choices=("val", "test"), default="val")
            q.add_argument("--calibration", type=Path)
            q.add_argument("--overlays", type=int, default=12)
        else:
            q.add_argument(
                "--limit-per-source",
                type=int,
                default=None,
                help="Optional diagnostic subset; default full validation",
            )
    q = sub.add_parser("infer")
    q.add_argument("--checkpoint", type=Path, required=True)
    q.add_argument("--image", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--calibration", type=Path)
    q.add_argument("--document-id")
    q.add_argument("--page-index", type=int, default=0)
    q.add_argument("--save-probabilities", action="store_true")
    q.add_argument("--device", default="cuda")
    q = sub.add_parser("export")
    q.add_argument("--checkpoint", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--calibration", type=Path)
    q = sub.add_parser("smoke", help="Train/resume/evaluate/export on generated diagnostic fixtures")
    q.add_argument(
        "--out", type=Path, help="Optional ignored artifact directory; otherwise temporary and cleaned up"
    )
    q.add_argument("--device", default="cpu")
    q.add_argument("--skip-export", action="store_true")
    return p


def execute(args):
    if getattr(args, "out", None):
        target = args.out if args.command not in {"audit", "calibrate", "compare"} else args.out.parent
        ensure_artifact_directory(target)
    command = args.command
    if command == "status":
        from .supervision import status

        return status(args.run, args.hourly_rate)
    if command == "compare":
        from .compare import compare

        result = compare(args.baseline, args.candidate)
        write_json(args.out, result)
        return result
    if command == "sources":
        from .acquire import sources

        return sources()
    if command == "doctor":
        from .doctor import doctor

        return doctor(args.require_gpu, args.smoke)
    if command == "fetch":
        from .acquire import fetch

        return fetch(args.source, args.data_root, args.url, args.extract)
    if command == "extract":
        from .acquire import extract_archive

        extract_archive(args.archive, args.out)
        return {"extracted": str(args.out)}
    if command == "prepare":
        from .prepare import prepare

        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        return prepare(
            args.source, args.raw, args.out, args.class_map, args.trust_pickle, args.limit, args.workers
        )
    if command == "audit":
        from .audit import audit_manifest

        result = audit_manifest(args.manifest, args.deep, args.near)
        if args.out:
            write_json(args.out, result)
        return result
    if command == "inspect":
        from .inspect_data import inspect_data

        return inspect_data(args.manifest, args.out, args.per_source)
    if command == "preflight":
        from .config import load_config
        from .manifest import load_manifest
        from .train import preflight

        return preflight(load_config(args.config), load_manifest(args.manifest, verify_files=True))
    if command == "regroup":
        from .manifest import regroup

        return regroup(args.manifest, args.mapping)
    if command == "train":
        from .train import train

        return train(
            args.config,
            args.manifest,
            args.run,
            args.device,
            args.resume,
            args.max_steps,
            args.max_minutes,
            args.init_weights,
        )
    if command == "smoke":
        from .checks import smoke

        return smoke(args.out, args.device, not args.skip_export)
    if command in {"evaluate", "calibrate", "infer", "export"}:
        from .model import load_detector

        device = getattr(args, "device", "cpu")
        model, data = load_detector(args.checkpoint, device)
        size = data["config"].get("image_size", 1024)
        calibration = read_json(args.calibration) if getattr(args, "calibration", None) else None
        if calibration and (
            calibration.get("manifest_fingerprint") != data["manifest_fingerprint"]
            or calibration.get("checkpoint_sha256") != sha256(args.checkpoint)
            or calibration.get("split") != "val"
        ):
            raise ValueError("Calibration must come from validation and match this checkpoint/dataset")
        thresholds = calibration["thresholds"] if calibration else None
        if command == "evaluate":
            from .evaluate import evaluate

            result = evaluate(
                model,
                args.manifest,
                args.split,
                device,
                size,
                thresholds,
                output=args.out,
                overlays=args.overlays,
            )
            result.update(
                {
                    "checkpoint_sha256": sha256(args.checkpoint),
                    "training_step": data["step"],
                    "training_manifest_fingerprint": data["manifest_fingerprint"],
                }
            )
            write_json(args.out / (args.split + "-metrics.json"), result)
            return result
        if command == "calibrate":
            from .evaluate import calibrate

            result = calibrate(model, args.manifest, device, size, args.limit_per_source)
            result["manifest_fingerprint"] = data["manifest_fingerprint"]
            result["checkpoint_sha256"] = sha256(args.checkpoint)
            write_json(args.out, result)
            return result
        if command == "infer":
            from .adapters import load_image
            from .inference import detect, overlay
            from PIL import Image

            image = load_image(args.image)
            result, maps = detect(
                model,
                image,
                device,
                size,
                thresholds,
                document_id=args.document_id,
                page_index=args.page_index,
                model_version=f"step-{data['step']}",
                return_maps=True,
            )
            args.out.mkdir(parents=True, exist_ok=True)
            write_json(args.out / "detections.json", result)
            Image.fromarray(overlay(image, result)).save(args.out / "overlay.png")
            if args.save_probabilities:
                import numpy as np

                np.savez_compressed(args.out / "probabilities.npz", **maps)
            return {"detections": str(args.out / "detections.json"), "instances": len(result["instances"])}
        from .export import export

        return export(args.checkpoint, args.out, size, thresholds)
    raise ValueError("Unknown command")


def main():
    args = parser().parse_args()
    try:
        result = execute(args)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"planproof-ml: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
