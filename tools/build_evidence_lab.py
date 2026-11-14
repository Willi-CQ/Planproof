"""Build an offline visual inspection of real pixel -> geometry <-> ML evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from planproof_evidence.contracts import sha256, validate_document, write_json


CLASSES = ("wall", "room", "door", "window", "stairs")


def score_entry(metrics, kind, case_ids, status=None):
    """Compact evaluated scores; missing coverage is never a zero score."""
    status = status or metrics.get("status", "not_run")
    entry = {
        "status": status,
        "metric": "mask_iou" if kind == "wall" else "f1_iou50",
        "value": None,
        "precision": None,
        "recall": None,
        "evaluated_case_ids": case_ids if status == "scored" else [],
        "reason": {
            "not_run": "No evaluated prediction file for this method and plan.",
            "not_annotated": "Class annotation coverage is incomplete or absent.",
            "model_unsupported": "This model does not predict this class.",
            "no_comparable_cases": "No plans have scored coverage for both methods.",
        }.get(status),
    }
    if status != "scored":
        return entry
    if kind == "wall":
        entry["value"] = metrics["mask_iou"]
        return entry
    for field in ("truth", "predicted", "tp", "fp", "fn"):
        entry[field] = metrics[field]
    tp, fp, fn = (entry[field] for field in ("tp", "fp", "fn"))
    entry.update(
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        value=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        ignored_predictions=metrics.get(
            "ignored_predictions", len(metrics.get("ignored_prediction_ids", []))
        ),
    )
    if not tp + fp + fn:
        entry["reason"] = "No labeled or predicted instances; F1 is not informative."
    return entry


def case_scores(case_id, models, report):
    rows = {
        row["method"]: row for row in report["results"] if row["case_id"] == case_id
    }
    coverage = {m["name"]: m["available_classes"] for m in models}
    result = {}
    for method in report["summary"]:
        row = rows.get(method, {"status": "not_run"})
        result[method] = {}
        for kind in CLASSES:
            metrics = row.get("metrics", {}).get(kind, {})
            status = metrics.get("status", row["status"])
            if method in coverage and kind not in coverage[method]:
                status = "model_unsupported"
            result[method][kind] = score_entry(metrics, kind, [case_id], status)
    return result


def aggregate_scores(entries, kind, case_ids):
    if not entries:
        return score_entry({}, kind, [], "no_comparable_cases")
    if kind == "wall":
        metrics = {"mask_iou": sum(e["value"] for e in entries) / len(entries)}
    else:
        metrics = {
            field: sum(e[field] for e in entries)
            for field in ("truth", "predicted", "tp", "fp", "fn", "ignored_predictions")
        }
    return score_entry(metrics, kind, case_ids, "scored")


def score_comparisons(cases, methods):
    """Use the exact same class-specific cohort on both sides, including CAD."""
    result = {}
    for method in methods:
        if method == "deterministic":
            continue
        result[method] = {}
        for kind in CLASSES:
            comparable = [
                case
                for case in cases
                if all(
                    case["scores"].get(m, {}).get(kind, {}).get("status") == "scored"
                    for m in (method, "deterministic")
                )
            ]
            ids = [c["id"] for c in comparable]
            result[method][kind] = {
                "case_ids": ids,
                "count": len(ids),
                "ml": aggregate_scores(
                    [c["scores"][method][kind] for c in comparable], kind, ids
                ),
                "deterministic": aggregate_scores(
                    [c["scores"]["deterministic"][kind] for c in comparable], kind, ids
                ),
            }
    return result


def payload(pack):
    manifest = json.loads((pack / "manifest.json").read_text())
    report = json.loads((pack / "failure-report.json").read_text())
    cases = []
    for case in manifest["cases"]:
        job = pack / "geometry" / case["case_id"]
        document = json.loads((job / "evidence.json").read_text())
        validate_document(document)
        page = document["pages"][0]
        if sha256(pack / case["directory"] / "plan.png") != document["source_sha256"]:
            raise ValueError("Dashboard source drawing is not bound to the evidence")
        artifact_hashes = {a["path"]: a["sha256"] for a in document["artifacts"]}
        for filename in (
            "page-000/ink.png",
            "page-000/pixel-evidence.json",
            "page-000/vector-evidence.json",
        ):
            if sha256(job / filename) != artifact_hashes[filename]:
                raise ValueError("Dashboard input artifact changed: " + filename)
        raw = json.loads((job / "page-000/pixel-evidence.json").read_text())
        vectors = json.loads((job / "page-000/vector-evidence.json").read_text())
        ink = np.asarray(Image.open(job / "page-000/ink.png")) > 0
        pixels = np.zeros((*ink.shape, 4), np.uint8)
        pixels[:, :, :3] = [147, 51, 234]
        pixels[:, :, 3] = ink * 170
        Image.fromarray(pixels).save(job / "page-000/pixel-layer.png")
        observations = [
            {
                key: o[key]
                for key in (
                    "id",
                    "class",
                    "bbox",
                    "polygons",
                    "source_family",
                    "mode",
                    "method",
                    "score",
                    "warnings",
                    "dependency_ids",
                    "features",
                    "feature_frame",
                    "quantities",
                    "room_type",
                    "normalization",
                    "overlay_label",
                )
                if key in o
            }
            | {"model_source": o.get("model_binding", {}).get("name")}
            for o in document["observations"]
            if o.get("class")
        ]
        sx, sy = raw["working_to_canonical"][0][0], raw["working_to_canonical"][1][1]
        lines = [
            [[x * sx, y * sy] for x, y in line["segment"]]
            for line in vectors["centerlines"]
        ]
        gaps = [
            [[x * sx, y * sy] for x, y in gap["segment"]]
            for gap in vectors["gap_hypotheses"]
        ]
        raw_cues = [
            {
                **cue,
                "bbox": [
                    v * (sx if i % 2 == 0 else sy) for i, v in enumerate(cue["bbox"])
                ],
            }
            for cue in raw.get("pixel_details", {}).get("cues", [])
        ]
        cases.append(
            {
                "id": case["case_id"],
                "source": case["source"],
                "size": case["image_size"],
                "image": case["directory"] + "/plan.png",
                "truth_image": case["directory"] + "/labels-overlay.png",
                "pixel_layer": f"geometry/{case['case_id']}/page-000/pixel-layer.png",
                "evidence": f"geometry/{case['case_id']}/evidence.json",
                "truth_counts": case["counts"],
                "frame": page["frame"],
                "observations": observations,
                "comparisons": document["comparisons"],
                "aggregates": document["aggregates"],
                "availability": page["availability"],
                "models": [
                    {
                        "name": m["name"],
                        "available_classes": m["available_classes"],
                        "model_sha256": m["binding"]["model_sha256"],
                        "profile_sha256": m["binding"]["profile_sha256"],
                        "prediction_sha256": m["binding"]["prediction_sha256"],
                    }
                    for m in page["ml_sources"]
                ],
                "warnings": page["warnings"],
                "pixel_details": {
                    key: value
                    for key, value in raw.get("pixel_details", {}).items()
                    if key not in ("components", "contour_rings", "cues")
                },
                "raw_cues": raw_cues,
                "centerlines": lines,
                "gaps": gaps,
                "topology": vectors["graph"]["diagnostics"],
                "timing": document["timing"],
                "metrics": {
                    r["method"]: r.get("metrics", {})
                    for r in report["results"]
                    if r["case_id"] == case["case_id"]
                },
                "scores": case_scores(case["case_id"], page["ml_sources"], report),
            }
        )
    return {
        "schema_version": "planproof-evidence-lab-v1",
        "cases": cases,
        "summary": report["summary"],
        "score_comparisons": score_comparisons(cases, report["summary"]),
        "score_notes": {
            "threshold_iou": report["threshold_iou"],
            "wall": "Mean per-plan wall mask IoU, not physical wall count.",
            "room": "Room instance F1 at mask IoU >= 0.5.",
            "objects": "Door/window/stair instance F1 at box IoU >= 0.5; not AP.",
            "aggregation": "F1/precision/recall use pooled TP/FP/FN on the same class-specific plans for both methods.",
            "coverage": "Unannotated and unsupported classes are excluded, not scored as zero. Predictions in ignored areas follow the frozen evaluator policy.",
            "agreement": "Spatial/class agreement between methods is separate from correctness against labels. Conflicts and unmatched records are not disjoint object counts.",
            "scope": "Ten selected development plans; not an unbiased test set or Stoa answer accuracy.",
        },
        "purpose": "Real ten-plan development evidence, not final Stoa QA accuracy",
        "upstream_commit": cases[0]["pixel_details"].get("upstream_commit"),
        "labels_used_for_inference": False,
    }


def build(pack):
    data = payload(pack)
    review_path = Path(__file__).with_name("evidence_lab_review.json")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    data["review_findings"] = (
        review["findings"]
        if review["input_report_sha256"] == sha256(pack / "failure-report.json")
        else []
    )
    template = Path(__file__).with_name("evidence_lab.html").read_text(encoding="utf-8")
    encoded = json.dumps(
        data, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).replace("<", "\\u003c")
    target = pack / "evidence-lab.html"
    target.write_text(template.replace("__EVIDENCE_DATA__", encoded), encoding="utf-8")
    write_json(
        pack / "EVIDENCE_LAB.json",
        {
            "html_sha256": sha256(target),
            "cases": len(data["cases"]),
            "input_report_sha256": sha256(pack / "failure-report.json"),
            "upstream_commit": data["upstream_commit"],
            "labels_used_for_inference": False,
            "review_source_sha256": sha256(review_path),
            "source_template_sha256": sha256(
                Path(__file__).with_name("evidence_lab.html")
            ),
        },
    )
    print(
        json.dumps(
            {
                "dashboard": str(target),
                "cases": len(data["cases"]),
                "bytes": target.stat().st_size,
            }
        )
    )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    build(parser.parse_args().pack)


if __name__ == "__main__":
    main()
