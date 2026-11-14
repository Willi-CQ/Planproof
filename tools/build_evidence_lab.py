"""Build an offline visual inspection of real pixel -> geometry <-> ML evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from planproof_evidence.contracts import sha256, validate_document, write_json


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
            }
        )
    return {
        "schema_version": "planproof-evidence-lab-v1",
        "cases": cases,
        "summary": report["summary"],
        "purpose": "Real ten-plan development evidence, not final Stoa QA accuracy",
        "upstream_commit": cases[0]["pixel_details"].get("upstream_commit"),
        "labels_used_for_inference": False,
    }


def build(pack):
    data = payload(pack)
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
