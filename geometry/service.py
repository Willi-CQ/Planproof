#!/usr/bin/env python3
"""FastAPI service for Planproof Evidence Extraction.

Exposes REST endpoints for Imad & Johan's agent harness to call
the deterministic evidence extraction pipeline.

Usage:
    python geometry/service.py [--host 127.0.0.1] [--port 8080]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the evidence pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "deterministic" / "src"))
from planproof_evidence import analyze, get_region
from planproof_evidence.contracts import EvidenceDocument, to_door_result_v1

# Legacy import for backward compatibility
try:
    from deterministic_engine import extract_deterministic, resolve_dataset_root
except ImportError:
    # If deterministic_engine doesn't exist, provide stub
    def extract_deterministic(*args, **kwargs):
        return {"ok": False, "is_valid_floorplan": False, "warnings": ["Legacy engine not available"]}
    
    def resolve_dataset_root():
        return None

app = FastAPI(
    title="Planproof Evidence Service",
    description="Deterministic 2D evidence extraction for architectural drawings",
    version="0.1.0"
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    file_path: str
    ml_predictions: Optional[str] = None
    ml_maps: Optional[str] = None
    config: Optional[dict] = None


class CropRequest(BaseModel):
    evidence_path: str
    page_index: int
    bbox: list[float]  # [x0, y0, x1, y1]
    padding_px: int = 0


class ExtractRequest(BaseModel):
    file_path: str
    floor_name: Optional[str] = None
    scale_override: Optional[float] = None
    extract_rooms: bool = True
    extract_walls: bool = True
    extract_openings: bool = True


class DoorResultRequest(BaseModel):
    """Request for planproof-door-result-v1 adapter.

    Provide either file_path (runs analyze) or evidence_path (reuses an
    existing evidence.json). page_frame may carry the caller's canonical
    frame identity to override/augment the reported source block.
    """
    file_path: Optional[str] = None
    evidence_path: Optional[str] = None
    page_index: int = 0
    page_frame: Optional[dict] = None
    ml_predictions: Optional[str] = None
    ml_maps: Optional[str] = None
    config: Optional[dict] = None


def resolve_file_path(file_path: str) -> Path:
    """Resolve file path against dataset roots.
    
    Checks:
    1. Absolute path
    2. STOA_DATASET_DIR environment variable
    3. /Volumes/hnguyen ssd/junction/junction-dataset
    4. Relative to current directory
    """
    path = Path(file_path)
    
    # If absolute path exists, use it
    if path.is_absolute() and path.exists():
        return path
    
    # Check environment variable
    dataset_dir = os.environ.get("STOA_DATASET_DIR")
    if dataset_dir:
        candidate = Path(dataset_dir) / file_path
        if candidate.exists():
            return candidate
        
        # Try with data/ prefix
        candidate = Path(dataset_dir) / "data" / file_path
        if candidate.exists():
            return candidate
    
    # Check default SSD path
    ssd_path = Path("/Volumes/hnguyen ssd/junction/junction-dataset")
    if ssd_path.exists():
        candidate = ssd_path / file_path
        if candidate.exists():
            return candidate
        
        candidate = ssd_path / "data" / file_path
        if candidate.exists():
            return candidate
    
    # Try relative to current directory
    if path.exists():
        return path
    
    # Return original path (will fail later with clear error)
    return path


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/config")
async def config():
    """Return configuration info including active dataset root."""
    dataset_root = resolve_dataset_root()
    questions_jsonl_found = False
    if dataset_root:
        questions_path = dataset_root / "questions.jsonl"
        questions_jsonl_found = questions_path.exists()
    
    return {
        "active_dataset_root": str(dataset_root) if dataset_root else None,
        "questions_jsonl_found": questions_jsonl_found
    }


@app.post("/analyze")
async def analyze_endpoint(request: AnalyzeRequest):
    """Analyze a drawing file and produce evidence document.
    
    Accepts PNG, JPG, PDF files and returns EvidenceDocument JSON.
    """
    try:
        # Resolve file path
        input_path = resolve_file_path(request.file_path)
        
        if not input_path.exists():
            raise HTTPException(status_code=400, detail=f"File not found: {request.file_path}")
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            
            # Resolve optional ML paths
            ml_predictions = None
            if request.ml_predictions:
                ml_predictions = resolve_file_path(request.ml_predictions)
                if not ml_predictions.exists():
                    raise HTTPException(status_code=400, detail=f"ML predictions not found: {request.ml_predictions}")
            
            ml_maps = None
            if request.ml_maps:
                ml_maps = resolve_file_path(request.ml_maps)
                if not ml_maps.exists():
                    raise HTTPException(status_code=400, detail=f"ML maps not found: {request.ml_maps}")
            
            # Run analysis
            doc = analyze(
                input_path=input_path,
                out_dir=out_dir,
                ml_predictions=ml_predictions,
                ml_maps=ml_maps,
                config=request.config,
            )
            
            # Return evidence document as JSON
            return json.loads(doc.to_json())
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "schema_version": "planproof-evidence-v1",
        })


@app.post("/crop")
async def crop_endpoint(request: CropRequest):
    """Extract a crop from evidence document.
    
    Returns crop path, intersecting observation IDs, and transform.
    """
    try:
        evidence_path = Path(request.evidence_path)
        
        if not evidence_path.exists():
            raise HTTPException(status_code=400, detail=f"Evidence file not found: {request.evidence_path}")
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            
            # Extract crop
            result = get_region(
                evidence_path=evidence_path,
                page_index=request.page_index,
                bbox=tuple(request.bbox),
                out_dir=out_dir,
                padding_px=request.padding_px,
            )
            
            # Read crop file and return as base64 or path
            # For now, return the result dict
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": str(e),
        })


@app.post("/v1/door-result")
async def door_result_endpoint(request: DoorResultRequest):
    """Return planproof-door-result-v1 for Imad & Johan's verification backend.

    Accepts either a drawing file_path (runs the evidence pipeline) or a
    previously produced evidence_path. The response matches their
    AnalyzerResult contract: schema_version, source PageFrame, version,
    status, mode, complete, detections[{id, bbox}].
    """
    if not request.file_path and not request.evidence_path:
        raise HTTPException(status_code=400, detail="Provide file_path or evidence_path")

    try:
        # Path 1: reuse an existing evidence document
        if request.evidence_path:
            ev_path = Path(request.evidence_path)
            if not ev_path.exists():
                raise HTTPException(status_code=400, detail=f"Evidence file not found: {request.evidence_path}")
            doc = EvidenceDocument.load(ev_path)
            return to_door_result_v1(
                doc,
                page_frame_override=request.page_frame,
                page_index=request.page_index,
            )

        # Path 2: run analyze on the drawing
        input_path = resolve_file_path(request.file_path)
        if not input_path.exists():
            raise HTTPException(status_code=400, detail=f"File not found: {request.file_path}")

        ml_predictions = None
        if request.ml_predictions:
            ml_predictions = resolve_file_path(request.ml_predictions)
            if not ml_predictions.exists():
                raise HTTPException(status_code=400, detail=f"ML predictions not found: {request.ml_predictions}")

        ml_maps = None
        if request.ml_maps:
            ml_maps = resolve_file_path(request.ml_maps)
            if not ml_maps.exists():
                raise HTTPException(status_code=400, detail=f"ML maps not found: {request.ml_maps}")

        with tempfile.TemporaryDirectory() as tmpdir:
            doc = analyze(
                input_path=input_path,
                out_dir=Path(tmpdir),
                ml_predictions=ml_predictions,
                ml_maps=ml_maps,
                config=request.config,
            )
            return to_door_result_v1(
                doc,
                page_frame_override=request.page_frame,
                page_index=request.page_index,
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "schema_version": "planproof-door-result-v1",
            "error": str(e),
        })


@app.post("/extract/deterministic")
async def extract_deterministic_endpoint(request: ExtractRequest):
    """Legacy endpoint for backward compatibility.
    
    Extracts deterministic geometry from IFC, DXF, PDF, PNG, JPG files.
    """
    try:
        result = extract_deterministic(
            request.file_path,
            floor_name=request.floor_name,
            scale_override=request.scale_override,
            extract_rooms=request.extract_rooms,
            extract_walls=request.extract_walls,
            extract_openings=request.extract_openings
        )
        
        # Return 400 for corrupt/unreadable files, 422 for non-floorplans
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result)
        
        if not result.get("is_valid_floorplan"):
            raise HTTPException(status_code=422, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "ok": False,
            "is_valid_floorplan": False,
            "file": Path(request.file_path).name,
            "resolved_source_file": Path(request.file_path).name,
            "engine_used": "unknown",
            "coordinate_frame": "normalized_0_1_top_left",
            "source_license": None,
            "walls": {"amount": 0, "net_area_m2": None, "items": []},
            "rooms": {"amount": 0, "total_area_m2": None, "items": []},
            "openings": {"doors": {"amount": 0, "items": []}, "windows": {"amount": 0, "items": []}},
            "raw_text_labels": [],
            "warnings": [f"Internal server error: {str(e)}"],
            "scale_confidence": "none"
        })


def main():
    parser = argparse.ArgumentParser(description="Planproof Evidence Service")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    
    args = parser.parse_args()
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
