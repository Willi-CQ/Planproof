"""Generate room artifacts first, then evaluate against the existing local labels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from planproof_evidence import build_room_boundaries, get_room_context
from planproof_evidence.contracts import sha256, write_json
from review_reference10 import score_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    args = parser.parse_args()
    pack = args.pack.resolve()
    manifest = json.loads((pack / "manifest.json").read_text())
    prior = json.loads((pack / "failure-report.json").read_text())
    checks = json.loads((pack / "SHA256SUMS.json").read_text())
    rows = []
    started = time.perf_counter()
    for case in manifest["cases"]:
        case_start = time.perf_counter()
        out = pack / "room-boundaries" / case["case_id"]
        evidence = pack / "geometry" / case["case_id"] / "evidence.json"
        evidence_sha = sha256(evidence)
        bundle = build_room_boundaries(evidence, out)
        page = bundle["pages"][0]
        prediction = {
            "document_id": case["source_id"],
            "page_index": 0,
            "image_size": case["image_size"],
            "coordinate_system": "pixel_edges_xy_top_left",
            "instances": [
                {
                    "id": r["id"],
                    "class": "room",
                    "bbox": r["bbox"],
                    "polygons": r["polygons"],
                    "confidence": 1.0,
                    "confidence_semantics": "not a confidence; evaluation adapter only",
                }
                for r in page["rooms"]
            ],
        }
        write_json(out / "room-predictions.json", prediction)
        opening_prediction = {
            **prediction,
            "instances": [
                {
                    "id": o["id"],
                    "class": o["type"],
                    "bbox": o["bbox"],
                    "confidence": 1.0,
                }
                for o in page["openings"]
                if o["type"] in ("door", "window")
            ],
        }
        # Labels are first opened here, after the room artifact has been generated.
        target_path = pack / case["directory"] / "targets.npz"
        key = target_path.relative_to(pack).as_posix()
        if sha256(target_path) != checks[key]:
            raise ValueError("Label hash changed")
        with np.load(target_path, allow_pickle=False) as data:
            targets = {k: data[k] for k in data.files}
            metrics = score_case(
                prediction,
                targets,
                case["source_id"],
                case["image_size"],
            )["room"]
            opening_metrics = score_case(
                opening_prediction, targets, case["source_id"], case["image_size"]
            )
        previous = {
            r["method"]: r["metrics"]["room"]
            for r in prior["results"]
            if r["case_id"] == case["case_id"] and r["status"] == "scored"
        }
        example = next(
            (r for r in page["rooms"] if r["status"] == "bounded_seed_region"),
            next(iter(page["rooms"]), None),
        )
        if example:
            get_room_context(
                out / "room-boundaries.json", example["id"], out / "example-room"
            )
        if sha256(evidence) != evidence_sha:
            raise ValueError("Source evidence was mutated")
        row = {
            "case_id": case["case_id"],
            "source": case["source"],
            "seconds": time.perf_counter() - case_start,
            "artifact": str(
                (out / "room-boundaries.json").relative_to(pack).as_posix()
            ),
            "artifact_sha256": sha256(out / "room-boundaries.json"),
            "source_evidence_sha256": evidence_sha,
            "room_metrics": metrics,
            "opening_metrics": {
                key: opening_metrics[key] for key in ("door", "window")
            },
            "baseline_room_metrics": previous,
            "region_statuses": {
                s: sum(r["status"] == s for r in page["rooms"])
                for s in sorted({r["status"] for r in page["rooms"]})
            },
            "opening_hypotheses": {
                s: sum(o["type"] == s for o in page["openings"])
                for s in ("door", "window", "walkthrough", "unknown")
            },
            "wall_roles": {
                s: sum(w["role"] == s for w in page["walls"])
                for s in ("inner", "outer", "unknown")
            },
            "unattached_ml_openings": len(
                page["diagnostics"].get("unattached_ml_openings", [])
            ),
            "unresolved_boundary_sections": sum(
                len(r["unresolved_sections"]) for r in page["rooms"]
            ),
            "accepted_refinements": sum(
                d["accepted"]
                for d in page["diagnostics"].get("refinement_decisions", [])
            ),
            "ml_opening_class_conflicts": sum(
                o.get("classification_conflict", False) for o in page["openings"]
            ),
            "semantic_wall_role_walkthrough_accuracy": "N/A: no suitable reference labels",
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "case_id",
                        "seconds",
                        "region_statuses",
                        "opening_hypotheses",
                        "room_metrics",
                    )
                }
            ),
            flush=True,
        )
    valid = [r["room_metrics"] for r in rows if r["room_metrics"]["status"] == "scored"]
    totals = {k: sum(r[k] for r in valid) for k in ("tp", "fp", "fn")}
    denominator = 2 * totals["tp"] + totals["fp"] + totals["fn"]
    totals["f1_iou50"] = 2 * totals["tp"] / denominator if denominator else None
    write_json(
        pack / "ROOM_BOUNDARY_REPORT.json",
        {
            "purpose": "Development diagnostics, not semantic room truth or QA accuracy",
            "cases": rows,
            "room_totals": totals,
            "seconds": time.perf_counter() - started,
            "labels_used_for_inference": False,
            "baseline_report_sha256": sha256(pack / "failure-report.json"),
            "manifest_sha256": sha256(pack / "manifest.json"),
            "note": "Unresolved ML fallbacks and shared/open regions remain in evaluation; no cherry-picked subset.",
        },
    )


if __name__ == "__main__":
    main()
