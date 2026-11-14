"""Execute Norlan's independent PNG extraction using its own Python environment."""

import json
import sys
from pathlib import Path

from planproof_evidence.pipeline import analyze


if __name__ == "__main__":
    image, result = map(Path, sys.argv[1:])
    from hashlib import sha256
    from PIL import Image

    document = analyze(image, result.parent / "evidence")
    with Image.open(image) as source:
        rgb = source.convert("RGB")
        width, height = rgb.size
        raster_sha = sha256(rgb.tobytes()).hexdigest()
    source_sha = sha256(image.read_bytes()).hexdigest()
    candidates = [
        o
        for o in document["observations"]
        if o.get("class") == "door" and o.get("source_family") == "classical_pixels"
    ]
    output = {
        "schema_version": "planproof-door-result-v1",
        "source": {
            "document_id": source_sha,
            "page_index": 0,
            "source_sha256": source_sha,
            "raster_sha256": raster_sha,
            "width": width,
            "height": height,
            "coordinate_system": "pixel_edges_xy_top_left",
        },
        "version": "integrated-evidence-v1",
        "status": "partial",
        "mode": "independent",
        "complete": False,
        "detections": [{"id": o["id"], "bbox": o["bbox"]} for o in candidates[:80]],
    }
    result.write_text(json.dumps(output), encoding="utf-8")
