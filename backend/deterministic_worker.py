"""Execute Norlan's independent PNG extraction using its own Python environment."""

import json
import sys
from pathlib import Path

from planproof_evidence.contracts import to_door_result_v1
from planproof_evidence.pipeline import analyze


if __name__ == "__main__":
    image, result = map(Path, sys.argv[1:])
    document = analyze(image, result.parent / "evidence", config={"include_ocr": False})
    result.write_text(json.dumps(to_door_result_v1(document)), encoding="utf-8")
