"""Preserve and compare one room-rule iteration without changing inference or labels."""

import argparse
import json
import shutil
from pathlib import Path

from planproof_evidence.contracts import item_geometry, sha256, write_json


def review(pack, out):
    pack, out = Path(pack).resolve(), Path(out).resolve()
    report_path = pack / "ROOM_BOUNDARY_REPORT.json"
    report = json.loads(report_path.read_text())
    if report["baseline_report_sha256"] != sha256(pack / "failure-report.json"):
        raise ValueError("Changed baseline")
    out.mkdir(parents=True, exist_ok=False)
    comparisons = []
    primary_geometry_checks = []
    for row in report["cases"]:
        current, baseline = (
            row["room_metrics"],
            row["baseline_room_metrics"].get("main", {}),
        )
        artifact = (pack / row["artifact"]).resolve()
        if (
            not artifact.is_relative_to(pack)
            or sha256(artifact) != row["artifact_sha256"]
        ):
            raise ValueError("Changed room artifact")
        bundle = json.loads(artifact.read_text())
        evidence_path = pack / "geometry" / row["case_id"] / "evidence.json"
        if sha256(evidence_path) != bundle["source_evidence_sha256"]:
            raise ValueError("Changed source evidence")
        observations = {
            o["id"]: o for o in json.loads(evidence_path.read_text())["observations"]
        }
        for page in bundle["pages"]:
            for room in page["rooms"]:
                if room.get("primary_geometry") == "retained_ml_seed":
                    original = item_geometry(observations[room["ml_seed_ids"][0]])
                    if original.symmetric_difference(item_geometry(room)).area >= 1e-6:
                        raise ValueError("Retained primary geometry was changed")
                    primary_geometry_checks.append(room["id"])
        if current["status"] != "scored" or baseline.get("status") != "scored":
            comparisons.append({"case_id": row["case_id"], "status": "N/A"})
            continue
        delta = current["f1_iou50"] - baseline["f1_iou50"]
        old_area = baseline.get("matched_area_relative_error_mean")
        new_area = current.get("matched_area_relative_error_mean")
        comparisons.append(
            {
                "case_id": row["case_id"],
                "status": "scored",
                "baseline_f1": baseline["f1_iou50"],
                "refined_f1": current["f1_iou50"],
                "f1_delta": delta,
                "f1_decision": "worse"
                if delta < -1e-9
                else "better"
                if delta > 1e-9
                else "unchanged",
                "matched_area_error_before": old_area,
                "matched_area_error_after": new_area,
                "matched_area_error_delta": new_area - old_area
                if old_area is not None and new_area is not None
                else None,
                "accepted_refinements": row.get("accepted_refinements"),
                "opening_metrics": row.get("opening_metrics"),
                "ml_opening_class_conflicts": row.get("ml_opening_class_conflicts"),
            }
        )
    for name in ("P05-resplan", "P01-cubicasa5k"):
        shutil.copytree(pack / "room-boundaries" / name, out / name)
        for prefix in ("room-boundaries-", "room-crop-", "room-baseline-"):
            image = pack / (prefix + name + ".png")
            if image.exists():
                shutil.copy2(image, out / image.name)
    source = Path(__file__).parents[1] / "deterministic/src/planproof_evidence/rooms.py"
    shutil.copy2(source, out / "rooms.py")
    shutil.copy2(report_path, out / "report.json")
    receipt = {
        "simple_case": "P05-resplan",
        "complex_case": "P01-cubicasa5k",
        "comparisons": comparisons,
        "pooled_room_metrics": report["room_totals"],
        "unchanged_imported_ml_primary_geometries": len(primary_geometry_checks),
        "f1_regressions": [
            r["case_id"] for r in comparisons if r.get("f1_decision") == "worse"
        ],
        "area_error_regressions": [
            r["case_id"]
            for r in comparisons
            if (r.get("matched_area_error_delta") or 0) > 1e-9
        ],
        "report_sha256": sha256(report_path),
        "implementation_sha256": sha256(source),
        "labels_used_for_inference": False,
        "limits": [
            "Development loop; no unseen QA claim.",
            "Unchanged F1 does not imply improved boundaries.",
            "Matched-room area error covers matched rooms only.",
            "Opening strip box IoU is not swing extent or physical access accuracy.",
            "Wall roles and walkthrough correctness remain unannotated.",
            "No LLM review was called by this tool.",
        ],
    }
    write_json(out / "feedback.json", receipt)
    write_json(
        out / "SHA256SUMS.json",
        {
            p.relative_to(out).as_posix(): sha256(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    )
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "pooled_room_metrics",
                    "f1_regressions",
                    "area_error_regressions",
                    "unchanged_imported_ml_primary_geometries",
                )
            },
            indent=2,
        )
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="New immutable iteration directory"
    )
    args = parser.parse_args()
    review(args.pack, args.out)
