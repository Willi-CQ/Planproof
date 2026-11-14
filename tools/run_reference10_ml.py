"""Run frozen ML inference on an existing reference pack; no training or tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

from planproof_ml.adapters import load_image
from planproof_ml.inference import detect, overlay
from planproof_ml.model import load_detector


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--releases", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.pack / "manifest.json").read_text())
    if len(manifest["cases"]) != 10:
        raise ValueError("Expected frozen ten-plan pack")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Explicit CUDA inference requested but unavailable")
    bindings = {
        "purpose": "frozen ML comparison only; deterministic results not run",
        "models": {},
    }
    for name, bundle in (
        ("main", "corrected-3500-global"),
        ("cad", "cad-objects-3000-global"),
    ):
        directory = args.releases / bundle
        checkpoint = directory / "detector.pt"
        sha = digest(checkpoint)
        calibration = json.loads((directory / "calibration.json").read_text())
        if calibration["checkpoint_sha256"] != sha or calibration["tiled"]:
            raise ValueError("Checkpoint/profile binding or inference mode mismatch")
        out = args.pack / "baselines" / name
        out.mkdir(parents=True, exist_ok=True)
        model, data = load_detector(checkpoint, "cuda")
        intended = data.get("intended_classes")
        rows = []
        for case in manifest["cases"]:
            if name == "cad" and case["source"] != "floorplancad":
                continue
            image = load_image(args.pack / case["directory"] / "plan.png")
            torch.cuda.synchronize()
            start = time.monotonic()
            prediction = detect(
                model,
                image,
                "cuda",
                size=1024,
                thresholds=calibration["thresholds"],
                tiled=False,
                document_id=case["source_id"],
                page_index=0,
                model_version=f"{bundle}-{sha[:12]}",
            )
            if intended:
                prediction["instances"] = [
                    x for x in prediction["instances"] if x["class"] in intended
                ]
                prediction["intended_classes"] = intended
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            (out / (case["case_id"] + ".json")).write_text(
                json.dumps(prediction, indent=2) + "\n"
            )
            Image.fromarray(overlay(image, prediction)).save(
                out / (case["case_id"] + ".png")
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "image_sha256": case["image_sha256"],
                    "image_hash_mode": case["image_hash_mode"],
                    "seconds": elapsed,
                }
            )
            print(json.dumps({"model": name, **rows[-1]}), flush=True)
        if digest(checkpoint) != sha:
            raise ValueError("Frozen checkpoint changed")
        bindings["models"][name] = {
            "checkpoint_sha256": sha,
            "bundle": bundle,
            "calibration_sha256": digest(directory / "calibration.json"),
            "inference_mode": "global1024",
            "cases": rows,
            "checkpoint_unchanged": True,
        }
        del model
        torch.cuda.empty_cache()
    (args.pack / "ML_BINDINGS.json").write_text(json.dumps(bindings, indent=2) + "\n")


if __name__ == "__main__":
    main()
