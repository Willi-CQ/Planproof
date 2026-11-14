"""Validate existing global-view component-area settings without retraining.

Cache one model pass per drawing, rebuild only wall/room components, then
confirm the selected setting through the ordinary production evaluator.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from planproof_ml.adapters import load_image
from planproof_ml.evaluate import evaluate
from planproof_ml.inference import component_instances, detect, split_rooms
from planproof_ml.manifest import fingerprint, load_manifest
from planproof_ml.metrics import Evaluator
from planproof_ml.model import load_detector
from planproof_ml.schema import read_json, sha256, write_json
from planproof_ml.storage import ensure_artifact_directory


def calibrate_area(checkpoint, manifest, calibration_path, areas, limit=32, device="cuda"):
    if not areas or any(not isinstance(area, int) or area < 1 for area in areas):
        raise ValueError("Component areas must be positive integers")
    if limit is not None and limit < 1:
        raise ValueError("Validation limit must be positive")
    started = time.monotonic()
    calibration = read_json(calibration_path)
    records = load_manifest(manifest, verify_files=True)
    model, data = load_detector(checkpoint, device)
    checkpoint_hash = sha256(checkpoint)
    size = data["config"]["image_size"]
    if (
        calibration["checkpoint_sha256"] != checkpoint_hash
        or calibration["manifest_fingerprint"] != fingerprint(records)
        or data["manifest_fingerprint"] != fingerprint(records)
        or calibration["split"] != "val"
        or calibration["tiled"] is not False
        or calibration["image_size"] != size
    ):
        raise ValueError("Use matching global-view validation calibration, checkpoint and manifest")
    evaluators = {area: Evaluator() for area in dict.fromkeys(areas)}
    counts = {}
    thresholds = calibration["thresholds"]
    for record in sorted((r for r in records if r["split"] == "val"), key=lambda r: r["id"]):
        source = record["source"]
        if limit and counts.get(source, 0) >= limit:
            continue
        image = load_image(manifest.parent / record["image"])
        prediction, maps = detect(model, image, device, size, thresholds, tiled=False, return_maps=True)
        with np.load(manifest.parent / record["targets"], allow_pickle=False) as archive:
            targets = {key: archive[key] for key in archive.files}
        masks, types = maps["mask_probabilities"], maps["room_type_probabilities"]
        maps["wall_threshold"], maps["room_threshold"] = thresholds["masks"][:2]
        sx, sy = maps["original_size"] / maps["work_size"]
        labels = {
            "wall": ndimage.label(masks[0] >= thresholds["masks"][0])[0],
            "room": split_rooms(
                masks[1], maps["boundary_probability"], maps["room_center_probability"], thresholds
            ),
        }
        for area, evaluator in evaluators.items():
            instances = [x for x in prediction["instances"] if x["class"] not in ("wall", "room")]
            for name, channel in (("wall", 0), ("room", 1)):
                items = component_instances(name, labels[name], channel, masks, types, area)
                for item in items:
                    item["bbox"] = (np.asarray(item["bbox"]) * [sx, sy, sx, sy]).tolist()
                    for polygon in item.get("polygons", []):
                        polygon["outer"] = (np.asarray(polygon["outer"]) * [sx, sy]).tolist()
                        polygon["holes"] = [(np.asarray(h) * [sx, sy]).tolist() for h in polygon["holes"]]
                instances.extend(items)
            evaluator.add(source, {"instances": instances}, targets, maps)
        counts[source] = counts.get(source, 0) + 1
    reports = {area: evaluator.report() for area, evaluator in evaluators.items()}
    if any(report["selection_score"] is None for report in reports.values()):
        raise ValueError("Validation must cover walls, rooms and objects")
    best = max(reports, key=lambda area: reports[area]["selection_score"])
    selected = {**thresholds, "min_pixels": best}
    confirmation = evaluate(model, manifest, "val", device, size, selected, limit, tiled=False)
    if not np.isclose(confirmation["selection_score"], reports[best]["selection_score"], atol=1e-9, rtol=0):
        raise ValueError("Cached area calibration differs from production evaluation")
    return {
        **calibration,
        "thresholds": selected,
        "validation_report": confirmation,
        "validation_macro": confirmation["macro"],
        "validation_score": confirmation["selection_score"],
        "cohort_fingerprints": confirmation["cohort_fingerprints"],
        "limit_per_source": limit,
        "component_area_candidates": {str(area): report for area, report in reports.items()},
        "component_area_parent_calibration_sha256": sha256(calibration_path),
        "component_area_forward_passes": sum(counts.values()),
        "component_area_confirmation_passes": sum(counts.values()),
        "component_area_seconds": time.monotonic() - started,
        "checkpoint_step": data["step"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--areas", type=int, nargs="+", default=[4, 32, 128, 512, 2048])
    parser.add_argument("--limit-per-source", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    ensure_artifact_directory(args.out.parent)
    torch.set_num_threads(2)
    result = calibrate_area(
        args.checkpoint, args.manifest, args.calibration, args.areas, args.limit_per_source, args.device
    )
    write_json(args.out, result)
    write_json(
        args.out.with_suffix(".metrics.json"),
        {
            **result["validation_report"],
            "checkpoint_step": result["checkpoint_step"],
            "checkpoint_sha256": result["checkpoint_sha256"],
        },
    )
    print(json.dumps({"thresholds": result["thresholds"], "macro": result["validation_macro"]}))


if __name__ == "__main__":
    main()
