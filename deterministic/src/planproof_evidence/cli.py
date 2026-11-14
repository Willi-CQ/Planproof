"""Command-line interface for planproof-evidence.

Provides 'analyze' and 'crop' commands for the evidence pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .pipeline import analyze, get_region
from .contracts import EvidenceDocument


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze a drawing file.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 = success, 1 = error)
    """
    input_path = Path(args.input)
    out_dir = Path(args.out or args.output)
    if args.out is None and args.output is None:
        print("Error: Output directory required (positional or -o/--out)", file=sys.stderr)
        return 1
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1
    
    # Load optional ML predictions
    ml_predictions = None
    if args.ml_predictions:
        ml_predictions = Path(args.ml_predictions)
        if not ml_predictions.exists():
            print(f"Error: ML predictions file not found: {ml_predictions}", file=sys.stderr)
            return 1
    
    # Load optional ML maps
    ml_maps = None
    if args.ml_maps:
        ml_maps = Path(args.ml_maps)
        if not ml_maps.exists():
            print(f"Error: ML maps file not found: {ml_maps}", file=sys.stderr)
            return 1
    
    # Load config if provided
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
    
    try:
        doc = analyze(
            input_path=input_path,
            out_dir=out_dir,
            ml_predictions=ml_predictions,
            ml_maps=ml_maps,
            config=config,
        )
        
        print(f"Analysis complete. Evidence saved to: {out_dir / 'evidence.json'}")
        print(f"Pages processed: {len(doc.pages)}")
        print(f"Observations: {len(doc.observations)}")
        print(f"Comparisons: {len(doc.comparisons)}")
        print(f"Aggregates: {len(doc.aggregates)}")
        
        if doc.diagnostics:
            print("\nDiagnostics:")
            for diag in doc.diagnostics:
                print(f"  - {diag}")
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_crop(args: argparse.Namespace) -> int:
    """Extract a crop from evidence.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 = success, 1 = error)
    """
    evidence_path = Path(args.evidence)
    out_dir = Path(args.output)
    page_index = args.page
    
    # Parse bbox
    try:
        bbox = tuple(float(x) for x in args.bbox.split(","))
        if len(bbox) != 4:
            raise ValueError("bbox must have 4 values")
    except Exception as e:
        print(f"Error: Invalid bbox format: {e}", file=sys.stderr)
        print("Expected format: x0,y0,x1,y1", file=sys.stderr)
        return 1
    
    padding_px = args.padding
    
    if not evidence_path.exists():
        print(f"Error: Evidence file not found: {evidence_path}", file=sys.stderr)
        return 1
    
    try:
        result = get_region(
            evidence_path=evidence_path,
            page_index=page_index,
            bbox=bbox,
            out_dir=out_dir,
            padding_px=padding_px,
        )
        
        print(f"Crop extracted: {result['crop_path']}")
        print(f"Size: {result['width']}x{result['height']}")
        print(f"Intersecting observations: {len(result['intersecting_observation_ids'])}")
        
        return 0
        
    except Exception as e:
        print(f"Error during crop extraction: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="planproof-evidence",
        description="Deterministic 2D evidence extraction for architectural drawings"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a drawing file")
    analyze_parser.add_argument("input", help="Input file path (PNG, JPG, PDF)")
    analyze_parser.add_argument("output", nargs="?", default=None,
                                help="Output directory for artifacts")
    analyze_parser.add_argument("-o", "--out", dest="out", default=None,
                                help="Alias for output directory")
    analyze_parser.add_argument("--ml-predictions", help="Path to ML predictions JSON")
    analyze_parser.add_argument("--ml-maps", help="Path to ML maps JSON")
    analyze_parser.add_argument("--config", help="Path to config JSON")
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Crop command
    crop_parser = subparsers.add_parser("crop", help="Extract a crop from evidence")
    crop_parser.add_argument("evidence", help="Path to evidence.json")
    crop_parser.add_argument("output", help="Output directory for crop")
    crop_parser.add_argument("page", type=int, help="Page index")
    crop_parser.add_argument("bbox", help="Bounding box: x0,y0,x1,y1")
    crop_parser.add_argument("--padding", type=int, default=0, help="Padding in pixels")
    crop_parser.set_defaults(func=cmd_crop)
    
    args = parser.parse_args()
    
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
