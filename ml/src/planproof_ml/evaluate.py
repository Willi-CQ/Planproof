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
        maps["room_threshold"] = (thresholds or {}).get("masks", [0.5] * 5)[1]
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


def calibrate(model, manifest, device="cpu", size=1024, limit_per_source=None, tiled=True):
    """Calibrate independent tasks from one model pass per validation drawing.

    Dense postprocessing is replayed on cached per-drawing maps. Object NMS at
    the lowest tested confidence retains every higher-confidence selected box.
    A final ordinary evaluation confirms the combined operating point.
    """
    from scipy import ndimage
    from .inference import DEFAULT_THRESHOLDS, component_instances, split_rooms

    records = load_manifest(manifest, verify_files=True)
    selected = [r for r in records if r["split"] == "val"]
    if not selected:
        raise ValueError("No validation samples for calibration")
    wall_candidates = {threshold: Evaluator() for threshold in (0.35, 0.5, 0.65)}
    room_candidates = {
        (mask, center): Evaluator() for mask in (0.5, 0.65) for center in (0.05, 0.1, 0.2, 0.3)
    }
    object_candidates = {threshold: Evaluator() for threshold in (0.1, 0.3, 0.5)}
    counts = {}
    passes = 0
    for record in sorted(selected, key=lambda r: r["id"]):
        source = record["source"]
        if limit_per_source and counts.get(source, 0) >= limit_per_source:
            continue
        image = load_image(manifest.parent / record["image"])
        prediction, maps = detect(
            model, image, device, size, {"centers": [0.3, 0.1, 0.1, 0.1]}, tiled=tiled, return_maps=True
        )
        passes += 1
        with np.load(manifest.parent / record["targets"], allow_pickle=False) as z:
            targets = {k: z[k] for k in z.files}
        masks, types = maps["mask_probabilities"], maps["room_type_probabilities"]
        sx, sy = maps["original_size"] / maps["work_size"]

        def instances(name, labels, channel):
            result = component_instances(
                name, labels, channel, masks, types, DEFAULT_THRESHOLDS["min_pixels"]
            )
            for item in result:
                item["bbox"] = (np.asarray(item["bbox"]) * [sx, sy, sx, sy]).tolist()
                for polygon in item.get("polygons", []):
                    polygon["outer"] = (np.asarray(polygon["outer"]) * [sx, sy]).tolist()
                    polygon["holes"] = [(np.asarray(r) * [sx, sy]).tolist() for r in polygon["holes"]]
            return {"instances": result}

        def only(component):
            valid = np.zeros_like(targets["valid"])
            if component in (0, 1):
                valid[component] = targets["valid"][component]
            complete = (
                targets["object_complete"] if component == 2 else np.zeros_like(targets["object_complete"])
            )
            return {**targets, "valid": valid, "object_complete": complete}

        wall_targets, room_targets, object_targets = (only(i) for i in (0, 1, 2))
        for threshold, evaluator in wall_candidates.items():
            labels = ndimage.label(masks[0] >= threshold)[0]
            evaluator.add(source, instances("wall", labels, 0), wall_targets)
        for (mask, center), evaluator in room_candidates.items():
            thresholds = {**DEFAULT_THRESHOLDS, "masks": [0.5, mask, 0.5, 0.5, 0.5], "centers": [center] * 4}
            labels = split_rooms(
                masks[1], maps["boundary_probability"], maps["room_center_probability"], thresholds
            )
            evaluator.add(source, instances("room", labels, 1), room_targets)
        objects = [item for item in prediction["instances"] if item["class"] in {"door", "window", "stairs"}]
        for threshold, evaluator in object_candidates.items():
            evaluator.add(
                source, {"instances": [p for p in objects if p["confidence"] >= threshold]}, object_targets
            )
        counts[source] = counts.get(source, 0) + 1

    candidates = []

    def choose(component, options, metric):
        scored = []
        for setting, evaluator in options.items():
            score = evaluator.report()["macro"][metric]
            candidates.append({"component": component, "setting": setting, "metric": metric, "score": score})
            if score is not None:
                scored.append((score, setting))
        if not scored:
            raise ValueError("Validation must cover wall, room and object metrics for calibration")
        return max(scored, key=lambda value: value[0])[1]

    wall = choose("wall", wall_candidates, "wall_iou")
    room, room_center = choose("room", room_candidates, "room_instance_f1_iou50")
    object_center = choose("objects", object_candidates, "object_map50")
    thresholds = {
        "masks": [wall, room, 0.5, 0.5, 0.5],
        "centers": [room_center, object_center, object_center, object_center],
    }
    confirmation = evaluate(model, manifest, "val", device, size, thresholds, limit_per_source, tiled=tiled)
    return {
        "thresholds": thresholds,
        "validation_score": confirmation["selection_score"],
        "validation_macro": confirmation["macro"],
        "candidates": candidates,
        "split": "val",
        "tiled": tiled,
        "limit_per_source": limit_per_source,
        "manifest_fingerprint": fingerprint(records),
        "cohort_fingerprints": confirmation["cohort_fingerprints"],
        "cached_inference_passes": passes,
        "confirmation_inference_passes": sum(counts.values()),
        "confidence_semantics": "validation-selected operating thresholds, not calibrated correctness probabilities",
    }
