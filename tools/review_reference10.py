"""Verify a local pack, score bound predictions and build an offline review page.

--method NAME evaluates baselines/NAME/CASE.json. Missing predictions are NOT RUN,
not empty detections. IoU50 counts are diagnostics, not the ML training AP metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

OBJECTS = ("door", "window", "stairs")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matching(iou, threshold=0.5):
    if not iou.size:
        return []
    forbidden = float(sum(iou.shape) + 1)
    rows, cols = linear_sum_assignment(np.where(iou >= threshold, 1 - iou, forbidden))
    return [(int(i), int(j)) for i, j in zip(rows, cols) if iou[i, j] >= threshold]


def box_iou(a, b):
    low, high = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    intersection = float(np.prod(np.maximum(0, high - low)))
    union = float(
        np.prod(np.asarray(a[2:]) - a[:2])
        + np.prod(np.asarray(b[2:]) - b[:2])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def rasterize(item, shape):
    out = np.zeros(shape, np.uint8)
    for polygon in item.get("polygons", []):
        region = np.zeros(shape, np.uint8)
        outer = np.asarray(polygon["outer"], dtype=float)
        if len(outer) < 3 or not np.isfinite(outer).all():
            raise ValueError("Invalid prediction polygon")
        cv2.fillPoly(region, [np.rint(outer).astype(np.int32)], 1)
        for hole in polygon.get("holes", []):
            cv2.fillPoly(region, [np.rint(hole).astype(np.int32)], 0)
        out |= region
    return out.astype(bool)


def counts(matches, predictions, truths):
    tp = len(matches)
    denominator = predictions + truths
    return {
        "truth": truths,
        "predicted": predictions,
        "tp": tp,
        "fp": predictions - tp,
        "fn": truths - tp,
        "count_error": predictions - truths,
        "f1_iou50": 2 * tp / denominator if denominator else 1.0,
    }


def score_case(prediction, targets, expected_id, size):
    if (
        prediction.get("document_id") != expected_id
        or prediction.get("page_index") != 0
    ):
        raise ValueError("Wrong document/page binding")
    if (
        prediction.get("image_size") != size
        or prediction.get("coordinate_system") != "pixel_edges_xy_top_left"
    ):
        raise ValueError("Wrong prediction frame")
    items = prediction["instances"]
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prediction IDs")
    for item in items:
        box = item["bbox"]
        if (
            item["class"] not in ("wall", "room", *OBJECTS)
            or not np.isfinite(box).all()
        ):
            raise ValueError("Invalid prediction class/geometry")
        if not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]):
            raise ValueError("Prediction outside page")
    result = {}
    h, w = targets["pixel_valid"].shape
    for i, name in enumerate(OBJECTS):
        if not targets["object_complete"][i]:
            result[name] = {"status": "not_annotated"}
            continue
        truth = targets["objects"][targets["objects"][:, 0] == i, 1:]
        predictions, ignored = [], []
        for item in items:
            if item["class"] != name:
                continue
            a, b, c, d = item["bbox"]
            cx, cy = (a + c) / 2, (b + d) / 2
            if targets["pixel_valid"][min(h - 1, int(cy)), min(w - 1, int(cx))] or any(
                g[0] <= cx < g[2] and g[1] <= cy < g[3] for g in truth
            ):
                predictions.append(item)
            else:
                ignored.append(item["id"])
        matrix = np.array(
            [[box_iou(p["bbox"], g) for g in truth] for p in predictions]
        ).reshape(len(predictions), len(truth))
        pairs = matching(matrix)
        missed = sorted(set(range(len(truth))) - {j for _, j in pairs})
        false = sorted(set(range(len(predictions))) - {j for j, _ in pairs})
        result[name] = {
            "status": "scored",
            **counts(pairs, len(predictions), len(truth)),
            "missed_truth_boxes": [truth[j].tolist() for j in missed],
            "false_prediction_ids": [predictions[j]["id"] for j in false],
            "ignored_prediction_ids": ignored,
        }
    for name, channel in (("wall", 0), ("room", 1)):
        valid = targets["valid"][channel].astype(bool)
        if not valid.any():
            result[name] = {"status": "not_annotated"}
            continue
        selected = [item for item in items if item["class"] == name]
        masks = [rasterize(item, (h, w)) & valid for item in selected]
        if name == "wall":
            combined = np.zeros((h, w), bool)
            for mask in masks:
                combined |= mask
            truth = targets["masks"][channel].astype(bool) & valid
            union = int((combined | truth).sum())
            result[name] = {
                "status": "scored",
                "mask_iou": int((combined & truth).sum()) / union if union else 1.0,
                "physical_wall_count": None,
            }
            continue
        truth = [
            (targets["instances"][0] == i) & valid
            for i in np.unique(targets["instances"][0])
            if i > 0
        ]
        matrix = np.zeros((len(masks), len(truth)))
        for i, predicted in enumerate(masks):
            for j, actual in enumerate(truth):
                union = int((predicted | actual).sum())
                matrix[i, j] = int((predicted & actual).sum()) / union if union else 0
        pairs = matching(matrix)
        errors = [
            abs(int(masks[i].sum()) - int(truth[j].sum())) / max(1, int(truth[j].sum()))
            for i, j in pairs
        ]
        result[name] = {
            "status": "scored",
            **counts(pairs, len(masks), len(truth)),
            "matched_area_relative_error_mean": float(np.mean(errors))
            if errors
            else None,
            "area_error_matched_rooms_only": True,
            "missed_truth_boxes": [
                [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]
                for j, mask in enumerate(truth)
                if j not in {gi for _, gi in pairs}
                for yy, xx in [np.where(mask)]
                if len(xx)
            ],
            "false_prediction_ids": [
                item["id"]
                for j, item in enumerate(selected)
                if j not in {pi for pi, _ in pairs}
            ],
        }
    return result


def error_overlay(image_path, prediction, metrics, out_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(2, image.width // 650)
    by_id = {item["id"]: item for item in prediction["instances"]}
    for name in ("room", *OBJECTS):
        record = metrics[name]
        for box in record.get("missed_truth_boxes", []):
            draw.rectangle(box, outline=(220, 25, 45), width=width)
            draw.text(
                (box[0], max(0, box[1] - 12)), "missed " + name, fill=(220, 25, 45)
            )
        for identity in record.get("false_prediction_ids", []):
            box = by_id[identity]["bbox"]
            draw.rectangle(box, outline=(230, 130, 0), width=width)
            draw.text((box[0], box[1]), "false " + name, fill=(190, 100, 0))
    image.save(out_path)


def render_review(root, manifest, report):
    payload = json.dumps(
        {"cases": manifest["cases"], "report": report}, ensure_ascii=True
    ).replace("<", chr(92) + "u003c")
    template = (
        Path(__file__).with_name("reference10_viewer.html").read_text(encoding="utf-8")
    )
    (root / "review.html").write_text(
        template.replace("__REFERENCE_DATA__", payload), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--method", action="append", default=[])
    args = parser.parse_args()
    root = args.pack
    manifest = json.loads((root / "manifest.json").read_text())
    checks = json.loads((root / "SHA256SUMS.json").read_text())
    for relative, expected in checks.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError(f"Pack checksum/path failure: {relative}")
    methods = list(dict.fromkeys([*args.method, "deterministic"]))
    rows, summary = [], {}
    for method in methods:
        if not method or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in method
        ):
            raise ValueError("Method must be a simple directory name")
        summary[method] = {"status": "not_run", "cases_scored": 0, "classes": {}}
        for case in manifest["cases"]:
            record = {
                "case_id": case["case_id"],
                "source": case["source"],
                "method": method,
            }
            path = root / "baselines" / method / (case["case_id"] + ".json")
            if not path.is_file():
                rows.append({**record, "status": "not_run"})
                continue
            with np.load(
                root / case["directory"] / "targets.npz", allow_pickle=False
            ) as data:
                targets = {key: data[key] for key in data.files}
            predicted = json.loads(path.read_text())
            metrics = score_case(
                predicted,
                targets,
                case["source_id"],
                case["image_size"],
            )
            rows.append({**record, "status": "scored", "metrics": metrics})
            error_overlay(
                root / case["directory"] / "plan.png",
                predicted,
                metrics,
                path.with_name(path.stem + "-errors.png"),
            )
            summary[method]["status"] = "scored"
            summary[method]["cases_scored"] += 1
        for name in ("wall", "room", *OBJECTS):
            supported = [
                row["metrics"][name]
                for row in rows
                if row["method"] == method
                and row["status"] == "scored"
                and row["metrics"][name]["status"] == "scored"
            ]
            entry = {
                "annotated_cases_scored": len(supported),
                "status": "scored"
                if supported
                else (
                    "not_run"
                    if summary[method]["cases_scored"] == 0
                    else "not_annotated"
                ),
            }
            if name == "wall":
                entry["mean_per_plan_mask_iou"] = (
                    float(np.mean([v["mask_iou"] for v in supported]))
                    if supported
                    else None
                )
            else:
                for field in ("truth", "predicted", "tp", "fp", "fn"):
                    entry[field] = (
                        sum(v[field] for v in supported) if supported else None
                    )
                entry["cases_with_misses"] = (
                    sum(v["fn"] > 0 for v in supported) if supported else None
                )
                entry["cases_with_false_positives"] = (
                    sum(v["fp"] > 0 for v in supported) if supported else None
                )
                entry["mean_absolute_count_error"] = (
                    float(np.mean([abs(v["count_error"]) for v in supported]))
                    if supported
                    else None
                )
            summary[method]["classes"][name] = entry
    report = {
        "purpose": "ten-plan development diagnostics, not final QA accuracy",
        "results": rows,
        "summary": summary,
        "threshold_iou": 0.5,
        "protocol": "mask/box IoU50 assignment; original pixels; explicit coverage; known-object override in ignored pixels",
    }
    (root / "failure-report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (root / "failure-table.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case",
                "method",
                "class",
                "status",
                "truth",
                "predicted",
                "tp",
                "fp",
                "fn",
                "count_error",
                "wall_iou",
            ]
        )
        for row in rows:
            for name in ("wall", "room", *OBJECTS):
                metrics = row.get("metrics", {}).get(name, {"status": row["status"]})
                writer.writerow(
                    [
                        row["case_id"],
                        row["method"],
                        name,
                        metrics["status"],
                        *[
                            metrics.get(key)
                            for key in (
                                "truth",
                                "predicted",
                                "tp",
                                "fp",
                                "fn",
                                "count_error",
                                "mask_iou",
                            )
                        ],
                    ]
                )
    render_review(root, manifest, report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
