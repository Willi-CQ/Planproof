"""Overlay rendering for observations.

Renders numbered observation overlays that match observation IDs
in evidence.json. Separate from clean source crops.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .contracts import Observation, PageFrame


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
        
        # Draw bbox with color based on source family
        color = {
            "native_pdf": "blue",
            "classical_pixels": "green",
            "ocr": "orange",
            "ml": "red",
            "hybrid": "purple",
        }.get(obs.source_family.value, "yellow")
        
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        
        # Draw label with observation ID
        label = f"{i+1}:{obs.id[:8]}"
        draw.text((x0, y0 - 15), label, fill=color, font=font)
    
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
