"""Run independent geometry plus bound ML cross-checks; labels are evaluator-only.

Creates a separate derived local pack, retaining the frozen initial pack unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from planproof_evidence import analyze
from planproof_evidence.contracts import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.pack.resolve() == args.out.resolve() or args.out.resolve().is_relative_to(
        args.pack.resolve()
    ):
        raise ValueError("Use a separate derived output pack")
    if not args.out.exists():
        shutil.copytree(args.pack, args.out)
    elif sha256(args.out / "manifest.json") != sha256(args.pack / "manifest.json"):
        raise ValueError("Existing derived pack has a different manifest")
    manifest = json.loads((args.pack / "manifest.json").read_text())
    bindings = json.loads((args.pack / "ML_BINDINGS.json").read_text())
    records = []
    start = time.perf_counter()
    for case in manifest["cases"]:
        name = case["case_id"]
        out = args.out / "geometry" / name
        out.mkdir(parents=True, exist_ok=True)
        sources = []
        for model, binding in bindings["models"].items():
            if name not in {r["case_id"] for r in binding["cases"]}:
                continue
            prediction = (args.pack / "baselines" / model / (name + ".json")).resolve()
            sources.append(
                {
                    "name": model,
                    "prediction_path": str(prediction),
                    "prediction_sha256": sha256(prediction),
                    "model_sha256": binding["checkpoint_sha256"],
                    "profile_sha256": binding["calibration_sha256"],
                    "raster_sha256": case["image_sha256"],
                    "image_size": case["image_size"],
                    "document_id": case["source_id"],
                    "page_index": 0,
                }
            )
        write_json(out / "binding.json", {"sources": sources})
        document = analyze(
            args.pack / case["directory"] / "plan.png",
            out,
            ml_predictions=out / "binding.json",
        )
        # The reporting adapter changes only the upstream identifier, never geometry.
        # SHA-bound evidence retains its canonical source-file identity separately.
        prediction = json.loads((out / "page-000/detections.json").read_text())
        prediction["canonical_document_id"] = prediction["document_id"]
        prediction["document_id"] = case["source_id"]
        dest = args.out / "baselines/deterministic"
        dest.mkdir(exist_ok=True)
        write_json(dest / (name + ".json"), prediction)
        shutil.copy2(out / "page-000/classical-overlay.png", dest / (name + ".png"))
        record = {
            "case_id": name,
            "source_sha256": document["source_sha256"],
            "evidence_sha256": sha256(out / "evidence.json"),
            "timing": document["timing"],
            "comparisons": {
                relation: sum(
                    c["relation"] == relation for c in document["comparisons"]
                )
                for relation in sorted({c["relation"] for c in document["comparisons"]})
            },
            "candidates": {
                c["class"]: c["candidate_count"]
                for c in document["aggregates"]
                if c["source"] == "classical"
            },
            "warnings": document["pages"][0]["warnings"],
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    write_json(
        args.out / "GEOMETRY_RUN.json",
        {
            "purpose": "ten-plan iterative development, not final QA accuracy",
            "input_manifest_sha256": sha256(args.pack / "manifest.json"),
            "seconds": time.perf_counter() - start,
            "cases": records,
            "labels_used_for_inference": False,
            "note": "Run review_reference10.py separately for supervised evaluation",
        },
    )


if __name__ == "__main__":
    main()
