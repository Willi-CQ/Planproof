from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

from .adapters import load_image
from .inference import detect, overlay
from .manifest import fingerprint, load_manifest
from .metrics import Evaluator
from .schema import write_json
from .storage import ensure_artifact_directory


def evaluate(
    model,
    manifest: Path,
    split="val",
    device="cpu",
    size=1024,
    thresholds=None,
    limit_per_source=None,
    tiled=True,
    output: Path | None = None,
    overlays=0,
):
    if output:
        ensure_artifact_directory(output)
    records = load_manifest(manifest, verify_files=True)
    selected = [r for r in records if r["split"] == split]
    if not selected:
        raise ValueError(f"No {split} samples in manifest")
    evaluator = Evaluator()
    counts = {}
    cohorts = {}
    done = 0
    for record in sorted(selected, key=lambda r: r["id"]):
        source = record["source"]
        if limit_per_source and counts.get(source, 0) >= limit_per_source:
            continue
        image = load_image(manifest.parent / record["image"])
        prediction, maps = detect(
            model,
            image,
            device,
            size,
            thresholds,
            tiled=tiled,
            document_id=record["id"],
            model_version="evaluation",
            return_maps=True,
        )
        maps["wall_threshold"] = (thresholds or {}).get("masks", [0.5] * 5)[0]
        with np.load(manifest.parent / record["targets"], allow_pickle=False) as z:
            targets = {k: z[k] for k in z.files}
        evaluator.add(source, prediction, targets, maps)
        cohorts.setdefault(source, []).append(
            [record["id"], record["image_sha256"], record.get("targets_sha256")]
        )
        if output and done < overlays:
            directory = output / "overlays"
            directory.mkdir(parents=True, exist_ok=True)
            Image.fromarray(overlay(image, prediction)).save(
                directory / (record["id"].replace(":", "_") + ".png")
            )
            write_json(directory / (record["id"].replace(":", "_") + ".json"), prediction)
        counts[source] = counts.get(source, 0) + 1
        done += 1
    report = evaluator.report()
    report.update(
        {
            "split": split,
            "manifest_fingerprint": fingerprint(records),
            "image_size": size,
            "tiled": tiled,
            "limit_per_source": limit_per_source,
            "thresholds": thresholds,
            "protocol": "project-v1-original-pixel-coordinates",
            "cohort_fingerprints": {
                source: hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
                for source, rows in cohorts.items()
            },
        }
    )
    if output:
        write_json(output / (split + "-metrics.json"), report)
    return report


def calibrate(model, manifest, device="cpu", size=1024, limit_per_source=None):
    """Threshold selection uses validation only; all tested candidates retained."""
    candidates = []
    for mask in (0.35, 0.5, 0.65):
        for center in (0.3, 0.5):
            thresholds = {"masks": [mask] * 5, "centers": [center] * 4}
            report = evaluate(model, manifest, "val", device, size, thresholds, limit_per_source, tiled=True)
            candidates.append({"thresholds": thresholds, "score": report["selection_score"]})
    usable = [c for c in candidates if c["score"] is not None]
    if not usable:
        raise ValueError("Validation must cover wall, room and object metrics for calibration")
    chosen = max(usable, key=lambda c: c["score"])
    return {
        "thresholds": chosen["thresholds"],
        "validation_score": chosen["score"],
        "candidates": candidates,
        "split": "val",
        "tiled": True,
        "limit_per_source": limit_per_source,
        "confidence_semantics": "threshold-calibrated model score, not a calibrated correctness probability",
    }
