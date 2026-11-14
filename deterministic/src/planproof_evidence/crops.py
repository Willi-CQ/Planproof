"""Crop extraction and overlay rendering.

Extracts clean, unannotated crops from canonical pixel frames
and renders numbered observation overlays.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .contracts import (
    ArtifactRef,
    BBox,
    EvidenceDocument,
    Observation,
    PageFrame,
    Point,
    bbox_intersection,
)


def extract_crop(
    evidence_path: Optional[Path],
    page_index: int,
    bbox: BBox,
    out_dir: Path,
    padding_px: int = 0,
    image: Optional[np.ndarray] = None,
    frame: Optional[PageFrame] = None,
) -> Dict:
    """Extract a clean crop from canonical pixel frame.
    
    Args:
        evidence_path: Path to evidence.json (for standalone use)
        page_index: Page index
        bbox: Bounding box [x0, y0, x1, y1] in canonical pixels
        out_dir: Output directory for crop
        padding_px: Padding around bbox in pixels
        image: Optional image array (if already loaded)
        frame: Optional page frame (if already loaded)
    
    Returns:
        Dict with crop_path, intersecting_observation_ids, crop_transform
    """
    # If image/frame not provided, load from evidence
    if image is None or frame is None:
        if evidence_path is None:
            raise ValueError("Must provide either image+frame or evidence_path")
        
        # Load evidence document
        doc = EvidenceDocument.load(evidence_path)
        
        # Find page frame
        page_frame = None
        for pf in doc.pages:
            if pf.page_index == page_index:
                page_frame = pf
                break
        
        if page_frame is None:
            raise ValueError(f"Page {page_index} not found in evidence")
        
        frame = page_frame
        
        # Load image from artifacts (would need to implement artifact loading)
        # For now, assume image is provided
        raise NotImplementedError("Artifact loading not yet implemented")
    
    # Apply padding
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding_px)
    y0 = max(0, y0 - padding_px)
    x1 = min(frame.image_size[0], x1 + padding_px)
    y1 = min(frame.image_size[1], y1 + padding_px)
    
    # Convert to integer pixels (exclusive right/bottom)
    x0_int = int(math.floor(x0))
    y0_int = int(math.floor(y0))
    x1_int = int(math.ceil(x1))
    y1_int = int(math.ceil(y1))
    
    # Clip to image bounds
    x0_int = max(0, x0_int)
    y0_int = max(0, y0_int)
    x1_int = min(frame.image_size[0], x1_int)
    y1_int = min(frame.image_size[1], y1_int)
    
    # Extract crop
    crop = image[y0_int:y1_int, x0_int:x1_int]
    
    # Save crop
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_filename = f"crop_p{page_index}_{x0_int}_{y0_int}_{x1_int}_{y1_int}.png"
    crop_path = out_dir / crop_filename
    
    # Convert to PIL and save
    if len(crop.shape) == 3:
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    else:
        pil_crop = Image.fromarray(crop)
    pil_crop.save(crop_path)
    
    # Compute crop-to-canonical transform
    # Crop pixel (0,0) corresponds to canonical (x0_int, y0_int)
    crop_transform = {
        "type": "translation",
        "dx": x0_int,
        "dy": y0_int,
        "scale": 1.0,
    }
    
    # Find intersecting observations
    intersecting_ids = []
    crop_bbox = (x0_int, y0_int, x1_int, y1_int)
    
    # This would need access to observations - for now return empty
    # In practice, this would be called from pipeline with full context
    
    return {
        "crop_path": str(crop_path),
        "crop_bbox": crop_bbox,
        "crop_transform": crop_transform,
        "intersecting_observation_ids": intersecting_ids,
        "width": x1_int - x0_int,
        "height": y1_int - y0_int,
    }


def get_region(
    evidence_path: Path,
    page_index: int,
    bbox: BBox,
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
    # Load evidence document
    doc = EvidenceDocument.load(evidence_path)
    
    # Find page frame
    page_frame = None
    for pf in doc.pages:
        if pf.page_index == page_index:
            page_frame = pf
            break
    
    if page_frame is None:
        raise ValueError(f"Page {page_index} not found in evidence")
    
    # Find source image path from metadata
    source_path = doc.source_path
    
    # Load image
    if source_path.endswith(".pdf"):
        # Would need to render PDF page
        raise NotImplementedError("PDF page loading not yet implemented")
    else:
        # Load raster image
        image = np.array(Image.open(source_path).convert("RGB"))
    
    # Find intersecting observations
    intersecting_ids = []
    for obs in doc.observations:
        if obs.page_index != page_index:
            continue
        if obs.bbox is None:
            continue
        
        if bbox_intersection(bbox, obs.bbox) is not None:
            intersecting_ids.append(obs.id)
    
    # Extract crop
    result = extract_crop(
        evidence_path=None,
        page_index=page_index,
        bbox=bbox,
        out_dir=out_dir,
        padding_px=padding_px,
        image=image,
        frame=page_frame,
    )
    
    result["intersecting_observation_ids"] = intersecting_ids
    
    return result


def render_overlay(
    image: np.ndarray,
    observations: List[Observation],
    page_index: int,
    out_path: Path,
) -> Path:
    """Render numbered observation overlay.
    
    Args:
        image: RGB image array
        observations: List of observations to overlay
        page_index: Page index
        out_path: Output path for overlay image
    
    Returns:
        Path to saved overlay image
    """
    # Convert to PIL
    if len(image.shape) == 3:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        pil_image = Image.fromarray(image)
    
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("Arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Draw observations
    for i, obs in enumerate(observations):
        if obs.page_index != page_index or obs.bbox is None:
            continue
        
        x0, y0, x1, y1 = obs.bbox
        
        # Draw bbox
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        
        # Draw label
        label = f"{i+1}:{obs.kind.value[:3]}"
        draw.text((x0, y0 - 15), label, fill="red", font=font)
    
    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(out_path)
    
    return out_path


def render_all_overlays(
    image: np.ndarray,
    observations: List[Observation],
    out_dir: Path,
) -> List[Path]:
    """Render overlays for all pages.
    
    Args:
        image: RGB image array
        observations: List of observations
        out_dir: Output directory
    
    Returns:
        List of overlay file paths
    """
    # Group by page
    pages = {}
    for obs in observations:
        if obs.page_index not in pages:
            pages[obs.page_index] = []
        pages[obs.page_index].append(obs)
    
    overlay_paths = []
    for page_index, page_obs in pages.items():
        overlay_path = out_dir / f"overlay_p{page_index}.png"
        render_overlay(image, page_obs, page_index, overlay_path)
        overlay_paths.append(overlay_path)
    
    return overlay_paths
