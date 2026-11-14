"""Consume the evidence engine's get_region artifacts without importing its runtime."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

from .artifacts import safe_path
from .images import load_images, sha256


def load_region(region_file: Path, image_config: dict):
    region = json.loads(region_file.read_text(encoding="utf-8"))
    crop = safe_path(region_file.parent, region["clean_crop"])
    if sha256(crop) != region["sha256"]:
        raise ValueError("Clean region crop hash mismatch")
    matrix = region["crop_to_canonical"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
        or not all(isinstance(v, (float, int)) and math.isfinite(v) for row in matrix for v in row)
        or matrix[2] != [0, 0, 1]
    ):
        raise ValueError("Region requires a finite affine crop-to-canonical transform")
    if abs(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) < 1e-12:
        raise ValueError("Singular crop transform")
    images = load_images([crop], image_config)
    image = images[0]
    metadata = {
        **image.metadata,
        "page_index": region["page_index"],
        "crop_to_canonical": matrix,
        "canonical_bbox": region["bbox"],
        "observation_ids": region["observation_ids"],
    }
    context = None
    if region.get("context_path"):
        raw = json.loads(
            safe_path(region_file.parent, region["context_path"]).read_text(encoding="utf-8")
        )
        if raw.get("schema_version") != "planproof-agent-context-v1":
            raise ValueError("Unexpected region evidence context schema")
        # Only the inference contract, never arbitrary sidecar keys or evaluator labels.
        context = {
            k: raw[k]
            for k in (
                "schema_version",
                "document_id",
                "scope",
                "policy",
                "pages",
                "observations",
                "comparisons",
                "page_aggregates",
                "calibrations",
            )
            if k in raw
        }
    return [replace(image, metadata=metadata)], context


def canonical_citations(citations: list[dict], images):
    by_id = {image.image_id: image.metadata for image in images}
    results = []
    for citation in citations:
        meta = by_id[citation["image_id"]]
        if "crop_to_canonical" not in meta:
            continue
        matrix = meta["crop_to_canonical"]
        sx, sy = meta["sent_to_source_render"]
        x0, y0, x1, y1 = citation["bbox"]
        points = []
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            x, y = x * sx, y * sy
            points.append(
                [
                    matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
                    matrix[1][0] * x + matrix[1][1] * y + matrix[1][2],
                ]
            )
        results.append(
            {
                "image_id": citation["image_id"],
                "page_index": meta["page_index"],
                "canonical_polygon": points,
                "canonical_bbox": [
                    min(p[0] for p in points),
                    min(p[1] for p in points),
                    max(p[0] for p in points),
                    max(p[1] for p in points),
                ],
            }
        )
    return results
