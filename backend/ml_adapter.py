"""Bind the frozen Planproof ML detection JSON to the page analyzed by the backend."""

import argparse
import json
from pathlib import Path
from typing import Any

from verification import AnalyzerResult, PageFrame, validate_source


def adapt_ml(raw: dict[str, Any], frame: PageFrame) -> AnalyzerResult:
    if (
        raw.get("schema_version") != "planproof-detection-v1"
        or raw.get("coordinate_system") != frame.coordinate_system
        or raw.get("page_index") != frame.page_index
        or raw.get("image_size") != [frame.width, frame.height]
    ):
        raise ValueError("Expected planproof-detection-v1 for this page and coordinate frame.")
    # Omitting --document-id makes Willi's launcher record the RGB input SHA256 here.
    if raw.get("document_id") != frame.raster_sha256:
        raise ValueError("Run inference on the canonical PNG and omit --document-id to record the raster hash.")
    if "intended_classes" in raw and "door" not in raw["intended_classes"]:
        raise ValueError("This model profile does not cover doors.")
    result = AnalyzerResult.model_validate({
        "schema_version": "planproof-door-result-v1",
        "source": frame,
        "version": raw["model_version"],
        "status": "ok",
        "mode": "independent",
        "complete": False,
        "detections": [
            {"id": item["id"], "bbox": item["bbox"]}
            for item in raw["instances"] if item["class"] == "door"
        ],
    })
    validate_source(result, frame)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detections", type=Path)
    parser.add_argument("frame", type=Path)
    args = parser.parse_args()
    try:
        frame = PageFrame.model_validate_json(args.frame.read_text(encoding="utf-8-sig"))
        raw = json.loads(args.detections.read_text(encoding="utf-8-sig"))
        result = adapt_ml(raw, frame)
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
