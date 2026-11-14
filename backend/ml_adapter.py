"""Bind the frozen Planproof ML detection JSON to the page analyzed by the backend."""

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from verification import AnalyzerResult, PageFrame, validate_source


class RawInstance(BaseModel):
    model_config = ConfigDict(strict=True)
    id: str
    kind: str = Field(alias="class")
    bbox: list[float]


class RawDetections(BaseModel):
    model_config = ConfigDict(strict=True)
    schema_version: str
    document_id: str
    coordinate_system: str
    page_index: int
    image_size: list[int]
    model_version: str
    intended_classes: list[str] | None = None
    instances: list[RawInstance]


def adapt_ml(raw: dict[str, Any], frame: PageFrame) -> AnalyzerResult:
    predictions = RawDetections.model_validate(raw)
    if (
        predictions.schema_version != "planproof-detection-v1"
        or predictions.coordinate_system != frame.coordinate_system
        or predictions.page_index != frame.page_index
        or predictions.image_size != [frame.width, frame.height]
    ):
        raise ValueError("Expected planproof-detection-v1 for this page and coordinate frame.")
    # Omitting --document-id makes Willi's launcher record the RGB input SHA256 here.
    if predictions.document_id != frame.raster_sha256:
        raise ValueError("Run inference on the canonical PNG and omit --document-id to record the raster hash.")
    if predictions.intended_classes is not None and "door" not in predictions.intended_classes:
        raise ValueError("This model profile does not cover doors.")
    result = AnalyzerResult.model_validate({
        "schema_version": "planproof-door-result-v1",
        "source": frame,
        "version": predictions.model_version,
        "status": "ok",
        "mode": "independent",
        "complete": False,
        "detections": [
            {"id": item.id, "bbox": item.bbox}
            for item in predictions.instances if item.kind == "door"
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
