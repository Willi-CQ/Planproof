"""Geometry worker, isolated from the web server and model credentials."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deterministic/src"))


def main():
    from planproof_evidence import analyze, build_room_boundaries, get_room_context
    from planproof_evidence.contracts import rgb_hash, sha256, write_json
    import numpy as np
    from PIL import Image

    operation, request_file = sys.argv[1:]
    request = json.loads(Path(request_file).read_text(encoding="utf-8"))
    if operation == "crop":
        get_room_context(request["boundaries"], request["room_id"], request["out"])
        return
    image, out = Path(request["image"]), Path(request["out"])
    binding = None
    if request.get("prediction"):
        prediction = Path(request["prediction"])
        raw = json.loads(prediction.read_text(encoding="utf-8"))
        with Image.open(image) as source:
            rgb = np.asarray(source.convert("RGB"))
        binding = out.parent / "ml-binding.json"
        write_json(
            binding,
            {
                "sources": [
                    {
                        "name": "main",
                        "page_index": 0,
                        "document_id": raw["document_id"],
                        "image_size": list(rgb.shape[1::-1]),
                        "raster_sha256": rgb_hash(rgb),
                        "prediction_path": str(prediction.resolve()),
                        "prediction_sha256": sha256(prediction),
                        "model_sha256": request["model_sha256"],
                        "profile_sha256": request["profile_sha256"],
                    }
                ]
            },
        )
    analyze(image, out, ml_predictions=binding)
    build_room_boundaries(out / "evidence.json", out.parent / "rooms")


if __name__ == "__main__":
    main()
