#!/usr/bin/env python3
"""Backward-compatible shim for the deterministic evidence engine.

This module delegates to planproof_evidence.analyze() for PNG/PDF files
and maintains the legacy extract_deterministic() interface for IFC/DXF
files (which are out of scope for the new evidence package).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add planproof_evidence to path
sys.path.insert(0, str(Path(__file__).parent.parent / "deterministic" / "src"))

try:
    from planproof_evidence import analyze, get_region
    from planproof_evidence.contracts import EvidenceDocument
except ImportError:
    # Fallback if package not installed
    analyze = None
    get_region = None
    EvidenceDocument = None

# Repository root for relative path resolution
REPO_ROOT = Path(__file__).parent.parent


def load_env_file(env_path: Path) -> None:
    """Load environment variables from .env file if present."""
    if env_path.exists():
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
        except Exception:
            pass


def resolve_dataset_root() -> Optional[Path]:
    """Resolve the active dataset root directory.
    
    Priority order:
    1. STOA_DATASET_DIR environment variable
    2. /Volumes/hnguyen ssd/junction/junction-dataset
    3. REPO_ROOT / "data" / "evaluation"
    4. ~/Downloads/junction-dataset-preview
    """
    # Load .env file
    load_env_file(REPO_ROOT / ".env")
    
    # 1. Environment variable
    env_dir = os.environ.get("STOA_DATASET_DIR")
    if env_dir:
        path = Path(env_dir)
        if path.exists():
            return path
    
    # 2. Official SSD dataset
    ssd_path = Path("/Volumes/hnguyen ssd/junction/junction-dataset")
    if ssd_path.exists():
        return ssd_path
    
    # 3. Local evaluation data
    eval_path = REPO_ROOT / "data" / "evaluation"
    if eval_path.exists():
        return eval_path
    
    # 4. Fallback preview dataset
    preview_path = Path.home() / "Downloads" / "junction-dataset-preview"
    if preview_path.exists():
        return preview_path
    
    return None


def resolve_file_path(file_path: str) -> Path:
    """Resolve a file path against dataset roots.
    
    Handles relative paths from questions.jsonl by checking:
    - <root>/<file_path>
    - <root>/data/<file_path>
    """
    path = Path(file_path)
    
    # If absolute path exists, use it
    if path.is_absolute() and path.exists():
        return path
    
    # Resolve against dataset root
    dataset_root = resolve_dataset_root()
    if dataset_root:
        # Try direct path
        candidate = dataset_root / file_path
        if candidate.exists():
            return candidate
        
        # Try with data/ prefix
        candidate = dataset_root / "data" / file_path
        if candidate.exists():
            return candidate
    
    # Try relative to current directory
    if path.exists():
        return path
    
    # Return original path (will fail later with clear error)
    return path


def extract_deterministic(
    file_path: str,
    floor_name: Optional[str] = None,
    scale_override: Optional[float] = None,
    extract_rooms: bool = True,
    extract_walls: bool = True,
    extract_openings: bool = True,
) -> Dict[str, Any]:
    """Extract deterministic geometry from a drawing file.
    
    This is a backward-compatible shim that delegates to planproof_evidence
    for PNG/PDF files and returns the legacy JSON format.
    
    Args:
        file_path: Path to drawing file (PNG, JPG, PDF, IFC, DXF)
        floor_name: Optional floor/storey name (ignored for PNG/PDF)
        scale_override: Optional scale override (meters per pixel)
        extract_rooms: Whether to extract rooms
        extract_walls: Whether to extract walls
        extract_openings: Whether to extract openings
    
    Returns:
        Dict with legacy JSON format
    """
    # Resolve file path
    resolved_path = resolve_file_path(file_path)
    
    # Check file exists
    if not resolved_path.exists():
        return {
            "ok": False,
            "is_valid_floorplan": False,
            "file": Path(file_path).name,
            "resolved_source_file": str(resolved_path),
            "engine_used": "unknown",
            "coordinate_frame": "normalized_0_1_top_left",
            "source_license": None,
            "walls": {"amount": 0, "net_area_m2": None, "items": []},
            "rooms": {"amount": 0, "total_area_m2": None, "items": []},
            "openings": {"doors": {"amount": 0, "items": []}, "windows": {"amount": 0, "items": []}},
            "raw_text_labels": [],
            "warnings": [f"File not found: {resolved_path}"],
            "scale_confidence": "none"
        }
    
    # Check file extension
    suffix = resolved_path.suffix.lower()
    
    # For PNG/JPG/PDF, use new evidence pipeline
    if suffix in [".png", ".jpg", ".jpeg", ".pdf"]:
        if analyze is None:
            return {
                "ok": False,
                "is_valid_floorplan": False,
                "file": Path(file_path).name,
                "resolved_source_file": str(resolved_path),
                "engine_used": "unknown",
                "coordinate_frame": "normalized_0_1_top_left",
                "source_license": None,
                "walls": {"amount": 0, "net_area_m2": None, "items": []},
                "rooms": {"amount": 0, "total_area_m2": None, "items": []},
                "openings": {"doors": {"amount": 0, "items": []}, "windows": {"amount": 0, "items": []}},
                "raw_text_labels": [],
                "warnings": ["planproof_evidence package not available"],
                "scale_confidence": "none"
            }
        
        try:
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as tmpdir:
                out_dir = Path(tmpdir)
                
                # Run analysis
                doc = analyze(
                    input_path=resolved_path,
                    out_dir=out_dir,
                    ml_predictions=None,
                    ml_maps=None,
                    config={
                        "include_ocr": True,
                        "max_pages": 1,
                        "render_scale": 1.0,
                    }
                )
                
                # Convert EvidenceDocument to legacy format
                return evidence_to_legacy_format(doc, resolved_path)
                
        except Exception as e:
            return {
                "ok": False,
                "is_valid_floorplan": False,
                "file": Path(file_path).name,
                "resolved_source_file": str(resolved_path),
                "engine_used": "unknown",
                "coordinate_frame": "normalized_0_1_top_left",
                "source_license": None,
                "walls": {"amount": 0, "net_area_m2": None, "items": []},
                "rooms": {"amount": 0, "total_area_m2": None, "items": []},
                "openings": {"doors": {"amount": 0, "items": []}, "windows": {"amount": 0, "items": []}},
                "raw_text_labels": [],
                "warnings": [f"Analysis failed: {str(e)}"],
                "scale_confidence": "none"
            }
    
    # For IFC/DXF, return unsupported message
    return {
        "ok": False,
        "is_valid_floorplan": False,
        "file": Path(file_path).name,
        "resolved_source_file": str(resolved_path),
        "engine_used": "unsupported",
        "coordinate_frame": "normalized_0_1_top_left",
        "source_license": None,
        "walls": {"amount": 0, "net_area_m2": None, "items": []},
        "rooms": {"amount": 0, "total_area_m2": None, "items": []},
        "openings": {"doors": {"amount": 0, "items": []}, "windows": {"amount": 0, "items": []}},
        "raw_text_labels": [],
        "warnings": [f"IFC/DXF files are not supported by planproof_evidence. Use PNG/PDF instead."],
        "scale_confidence": "none"
    }


def evidence_to_legacy_format(doc: EvidenceDocument, source_path: Path) -> Dict[str, Any]:
    """Convert EvidenceDocument to legacy JSON format.
    
    Args:
        doc: EvidenceDocument from planproof_evidence
        source_path: Path to source file
    
    Returns:
        Dict in legacy format
    """
    # Initialize result
    result = {
        "ok": True,
        "is_valid_floorplan": len(doc.pages) > 0,
        "file": source_path.name,
        "resolved_source_file": str(source_path),
        "engine_used": "planproof_evidence",
        "coordinate_frame": "normalized_0_1_top_left",
        "source_license": None,
        "walls": {"amount": 0, "net_area_m2": None, "items": []},
        "rooms": {"amount": 0, "total_area_m2": None, "items": []},
        "openings": {"doors": {"amount": 0, "items": []}, "windows": {"amount": 0, "items": []}},
        "raw_text_labels": [],
        "warnings": doc.diagnostics.copy(),
        "scale_confidence": "none"
    }
    
    # Process observations
    walls = []
    rooms = []
    doors = []
    windows = []
    text_labels = []
    
    for obs in doc.observations:
        if obs.page_index != 0:  # Only first page for legacy format
            continue
        
        # Normalize coordinates to [0, 1]
        if obs.bbox and doc.pages:
            frame = doc.pages[0]
            width, height = frame.image_size
            
            x0, y0, x1, y1 = obs.bbox
            norm_bbox = [
                x0 / width,
                y0 / height,
                x1 / width,
                y1 / height,
            ]
        else:
            norm_bbox = [0, 0, 0, 0]
        
        # Categorize by kind
        if obs.kind.value == "wall":
            walls.append({
                "coordinates": norm_bbox,
                "thickness_px": obs.geometry.get("thickness") if obs.geometry else None,
            })
        elif obs.kind.value == "room":
            rooms.append({
                "coordinates": norm_bbox,
                "area_px": obs.geometry.get("area") if obs.geometry else None,
                "area_m2": None,  # Will be filled if calibrated
            })
        elif obs.kind.value == "door":
            doors.append({
                "coordinates": norm_bbox,
            })
        elif obs.kind.value == "window":
            windows.append({
                "coordinates": norm_bbox,
            })
        elif obs.kind.value == "text":
            text = obs.geometry.get("text", "") if obs.geometry else ""
            if text:
                text_labels.append(text)
    
    # Update counts
    result["walls"]["amount"] = len(walls)
    result["walls"]["items"] = walls[:50]  # Limit to 50 items
    
    result["rooms"]["amount"] = len(rooms)
    result["rooms"]["items"] = rooms[:20]
    
    result["openings"]["doors"]["amount"] = len(doors)
    result["openings"]["doors"]["items"] = doors[:50]
    
    result["openings"]["windows"]["amount"] = len(windows)
    result["openings"]["windows"]["items"] = windows[:50]
    
    result["raw_text_labels"] = text_labels[:100]
    
    # Check for calibration
    if doc.calibrations:
        result["scale_confidence"] = "high"
        # TODO: Apply calibration to convert px to m
    else:
        result["scale_confidence"] = "none"
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Planproof Deterministic Engine (Legacy Shim)")
    parser.add_argument("file_path", help="Path to drawing file")
    parser.add_argument("--floor-name", help="Floor/storey name (for IFC)")
    parser.add_argument("--scale-override", type=float, help="Scale override (meters per pixel)")
    parser.add_argument("--no-rooms", action="store_true", help="Skip room extraction")
    parser.add_argument("--no-walls", action="store_true", help="Skip wall extraction")
    parser.add_argument("--no-openings", action="store_true", help="Skip opening extraction")
    parser.add_argument("--output", help="Output JSON to file instead of stdout")
    
    args = parser.parse_args()
    
    result = extract_deterministic(
        args.file_path,
        floor_name=args.floor_name,
        scale_override=args.scale_override,
        extract_rooms=not args.no_rooms,
        extract_walls=not args.no_walls,
        extract_openings=not args.no_openings,
    )
    
    json_output = json.dumps(result, indent=2)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_output)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
