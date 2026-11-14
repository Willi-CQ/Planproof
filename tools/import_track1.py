"""Import reviewed pure pixel routines from the pinned teammate commit, without its app."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

COMMIT = "24e37023c9441c657050b8595f1cf738230e6b17"
SOURCE = "deterministic/src/planproof_evidence/pixels.py"
FUNCTIONS = (
    "extract_ink_mask",
    "extract_components",
    "extract_contour_rings",
    "extract_line_segments",
    "detect_opening_candidates",
    "detect_stair_candidates",
)


def main():
    root = Path(__file__).resolve().parents[1]
    content = subprocess.check_output(
        [
            "git",
            "-c",
            "safe.directory=" + root.as_posix(),
            "show",
            COMMIT + ":" + SOURCE,
        ],
        cwd=root,
    )
    source = content.decode("utf-8")
    tree = ast.parse(source)
    definitions = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    header = (
        '"""Reviewed Track 1 pixel routines from the teammate contribution.\n\n'
        "Source pin and adaptation record: deterministic/TRACK1_SOURCE.json.\n"
        "Typed app contracts and its duplicate orchestrator are intentionally not imported.\n"
        'Upstream heuristic scores are not calibrated correctness probabilities.\n"""\n\n'
        "from __future__ import annotations\n\nimport math\nfrom typing import Dict, List\n\n"
        "import cv2\nimport numpy as np\n\n"
    )
    target = root / "deterministic/src/planproof_evidence/track1_pixels.py"
    target.write_text(
        header + "\n\n\n".join(definitions[name] for name in FUNCTIONS) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "repository": "https://github.com/Willi-CQ/Planproof",
        "branch": "iamd-johan-deterministic-fix",
        "upstream_commit": COMMIT,
        "original_implementation_commit": "d4b8c003c250bfdccb9e47b8b4d73c367597b21d",
        "source_path": SOURCE,
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "functions": list(FUNCTIONS),
        "changes": [
            "Pure routines retained; app-specific Pydantic imports omitted.",
            "Formatting/lint normalization only in imported routines; integration logic lives in track1_adapter.py.",
            "Latest Hough coordinate-shape correction included.",
            "Upstream opening/stair proposals remain unclassified cues, not accepted semantic objects.",
        ],
    }
    (root / "deterministic/TRACK1_SOURCE.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
