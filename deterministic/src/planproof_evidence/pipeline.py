"""Main pipeline orchestration.

Coordinates all extraction stages: raster loading, pixel extraction,
native PDF primitives, vector extraction, topology, measurements,
ML comparison, and artifact generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .contracts import (
    Aggregate,
    ArtifactRef,
    CalibrationRecord,
    Comparison,
    EvidenceDocument,
    Observation,
    PageFrame,
    PixelEvidence,
    SourceFamily,
    Mode,
    VectorEvidence,
    sha256_file,
)
from .frames import create_page_frame, identity_transform
from .raster import load_drawing, detect_non_floorplan
from .pixels import extract_pixel_evidence, extract_ink_mask
from .ocr import extract_ocr_spans, check_tesseract_available
from .native_pdf import extract_pdf_primitives, extract_pdf_text_spans
from .vectors import extract_vector_evidence, skeletonize_mask, extract_wall_edges
from .topology import extract_topology
from .measurements import extract_measurements
from .ml_adapter import load_ml_predictions, extract_ml_observations, load_ml_maps, validate_ml_maps
from .compare import compare_ml_and_geometry, create_aggregates
from .crops import extract_crop, get_region as get_region_impl
from .overlays import render_all_overlays
from .semantic import classify_all_semantic


def analyze(
    input_path: Path,
    out_dir: Path,
    ml_predictions: Optional[Path] = None,
    ml_maps: Optional[Path] = None,
    config: Optional[Dict] = None,
) -> EvidenceDocument:
    """Analyze a drawing file and produce evidence document.
    
    Args:
        input_path: Path to PNG/JPG/PDF file
        out_dir: Output directory for artifacts
        ml_predictions: Optional path to ML predictions JSON
        ml_maps: Optional path to ML maps JSON
        config: Optional configuration dict
    
    Returns:
        EvidenceDocument with all evidence
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    if config is None:
        config = {}
    
    include_ocr = config.get("include_ocr", True)
    max_pages = config.get("max_pages", 1)
    render_scale = config.get("render_scale", 1.0)
    
    # Load drawing
    pages_data = load_drawing(input_path)
    
    # Initialize document
    doc = EvidenceDocument(
        document_id="",  # Will be set
        source_path=str(input_path),
        source_sha256="",  # Will be set
        pages=[],
        pixel_evidence=[],
        vector_evidence=[],
        observations=[],
        comparisons=[],
        aggregates=[],
        calibrations=[],
        artifacts=[],
        diagnostics=[],
        metadata={},
    )
    
    # Process each page
    all_observations = []
    all_artifacts = []
    
    for page_idx, (image, frame, metadata) in enumerate(pages_data[:max_pages]):
        # Check non-floorplan
        is_valid, reason = detect_non_floorplan(image)
        if not is_valid:
            doc.diagnostics.append(f"Page {page_idx}: {reason}")
            continue
        
        # Add page frame
        doc.pages.append(frame)
        
        # Convert PIL to numpy
        img_array = np.array(image)
        
        # Extract pixel evidence
        pixel_ev = extract_pixel_evidence(img_array, frame, include_ocr=include_ocr)
        doc.pixel_evidence.append(pixel_ev)
        all_observations.extend(pixel_ev.observations)
        
        # Save mask artifact to output directory
        mask = extract_ink_mask(img_array)
        mask_path = out_dir / f"mask_p{page_idx}.png"
        cv2.imwrite(str(mask_path), mask)
        all_artifacts.append(ArtifactRef(
            path=str(mask_path.relative_to(out_dir)),
            sha256=sha256_file(mask_path),
            kind="mask",
            dimensions=mask.shape[::-1],
            frame_id=f"p{page_idx}-canonical",
        ))
        
        # Extract native PDF primitives if PDF
        if input_path.suffix.lower() == ".pdf":
            pdf_obs = extract_pdf_primitives(input_path, page_idx, frame)
            all_observations.extend(pdf_obs)
            
            # Extract text spans
            text_obs = extract_pdf_text_spans(input_path, page_idx, frame)
            all_observations.extend(text_obs)
        
        # Extract vector evidence
        mask = extract_ink_mask(img_array)
        skeleton = skeletonize_mask(mask)
        wall_edges = extract_wall_edges(mask, skeleton)
        
        # Get opening candidates from pixel evidence
        openings = [obs for obs in pixel_ev.observations 
                   if obs.kind.value in ["opening_candidate", "door", "window"]]
        
        # Extract topology
        vector_ev = extract_topology(mask, wall_edges, openings, frame)
        doc.vector_evidence.append(vector_ev)
        
        # Extract measurements
        text_obs = [obs for obs in all_observations if obs.kind.value == "text"]
        dimensions, calibrations = extract_measurements(text_obs, vector_ev.wall_edges, frame)
        
        # Add dimensions to vector evidence
        vector_ev.dimensions = dimensions
        vector_ev.calibrations = calibrations
        doc.calibrations.extend(calibrations)
        
        # Classify semantic objects (walls, rooms, doors, windows, stairs)
        semantic_obs = classify_all_semantic(vector_ev, pixel_ev, frame)
        all_observations.extend(semantic_obs)
        
        # Render overlays
        overlay_paths = render_all_overlays(img_array, pixel_ev.observations, out_dir)
        for path in overlay_paths:
            all_artifacts.append(ArtifactRef(
                path=str(path.relative_to(out_dir)),
                sha256=sha256_file(path),
                kind="overlay",
                dimensions=None,
                frame_id=f"p{page_idx}-canonical",
            ))
    
    # Process ML predictions if provided
    ml_obs = []
    if ml_predictions is not None:
        try:
            prediction = load_ml_predictions(ml_predictions)
            # Use first page frame for validation
            if doc.pages:
                frame = doc.pages[0]
                ml_obs, warnings = extract_ml_observations(prediction, frame)
                doc.diagnostics.extend(warnings)
                all_observations.extend(ml_obs)
        except Exception as e:
            doc.diagnostics.append(f"ML prediction loading failed: {e}")
    
    # Process ML maps if provided
    if ml_maps is not None:
        try:
            maps = load_ml_maps(ml_maps)
            if doc.pages:
                frame = doc.pages[0]
                is_valid, warnings = validate_ml_maps(maps, frame)
                doc.diagnostics.extend(warnings)
        except Exception as e:
            doc.diagnostics.append(f"ML maps loading failed: {e}")
    
    # Compare ML and geometry
    if ml_obs:
        # Get geometry observations (walls, rooms, doors, windows)
        geom_obs = [obs for obs in all_observations 
                   if obs.source_family != SourceFamily.ML and
                   obs.kind.value in ["wall", "room", "door", "window", "stairs"]]
        
        comparisons = compare_ml_and_geometry(ml_obs, geom_obs, doc.pages[0] if doc.pages else None)
        doc.comparisons = comparisons
    
    # Create aggregates
    doc.observations = all_observations
    doc.aggregates = create_aggregates(all_observations, doc.comparisons, doc.pages[0] if doc.pages else None)
    
    # Set document IDs
    doc.document_id = sha256_file(input_path)
    doc.source_sha256 = doc.document_id
    
    # Add artifacts
    doc.artifacts = all_artifacts
    
    # Save evidence document
    evidence_path = out_dir / "evidence.json"
    doc.save(evidence_path)
    
    return doc


def get_region(
    evidence_path: Path,
    page_index: int,
    bbox: Tuple[float, float, float, float],
    out_dir: Path,
    padding_px: int = 0,
) -> Dict:
    """Public API for extracting a region from evidence.
    
    Args:
        evidence_path: Path to evidence.json
        page_index: Page index
        bbox: Bounding box [x0, y0, x1, y1]
        out_dir: Output directory
        padding_px: Padding around bbox
    
    Returns:
        Dict with crop information
    """
    return get_region_impl(evidence_path, page_index, bbox, out_dir, padding_px)
