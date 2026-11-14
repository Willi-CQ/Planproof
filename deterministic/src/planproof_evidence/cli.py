from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import analyze, get_region


def main():
    parser = argparse.ArgumentParser(description="Local PNG/PDF geometry evidence (CPU only)")
    commands = parser.add_subparsers(dest="command", required=True)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("input", type=Path)
    analysis.add_argument("--out", required=True, type=Path)
    analysis.add_argument("--ml-predictions", type=Path)
    analysis.add_argument("--config", type=Path)
    crop = commands.add_parser("crop")
    crop.add_argument("evidence", type=Path)
    crop.add_argument("--out", required=True, type=Path)
    crop.add_argument("--page", type=int, required=True)
    crop.add_argument("--bbox", type=float, nargs=4, required=True)
    crop.add_argument("--padding-px", type=float, default=0)
    args = parser.parse_args()
    if args.command == "analyze":
        config = json.loads(args.config.read_text()) if args.config else None
        document = analyze(args.input, args.out, ml_predictions=args.ml_predictions, config=config)
        print(
            json.dumps(
                {
                    "evidence": str(args.out / "evidence.json"),
                    "pages": len(document["pages"]),
                    "observations": len(document["observations"]),
                    "timing": document["timing"],
                }
            )
        )
    else:
        print(
            json.dumps(
                get_region(
                    args.evidence, args.page, args.bbox, args.out, padding_px=args.padding_px
                )
            )
        )


if __name__ == "__main__":
    main()
