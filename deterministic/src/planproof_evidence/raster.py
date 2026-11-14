"""Raster image loading and PDF page rendering.

Handles PNG/JPG loading and PDF page rasterization with pypdfium2,
preserving source hashes and transforms.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from .contracts import PageFrame, PageMetadata, Transform3x3
from .frames import create_page_frame, identity_transform, normalize_image


def load_image(path: Path) -> Tuple[Image.Image, PageFrame]:
    """Load a raster image (PNG/JPG) and create its page frame.
    
    Args:
        path: Path to image file
    
    Returns:
        (PIL Image, PageFrame)
    """
    image = Image.open(path)
    image = normalize_image(image)
    
    document_id = create_page_frame.__module__  # Will be set properly
    # Compute document ID from file hash
    from .contracts import sha256_file
    document_id = sha256_file(path)
    
    frame = create_page_frame(
        document_id=document_id,
        page_index=0,
        source_path=path,
        image=image,
        source_to_canonical=identity_transform(),
        page_metadata=None,
    )
    
    return image, frame


def render_pdf_page(
    pdf_path: Path,
    page_index: int = 0,
    scale: float = 1.0,
    max_pixels: int = 4096,
) -> Tuple[Image.Image, PageFrame, PageMetadata]:
    """Render a PDF page to image.
    
    Args:
        pdf_path: Path to PDF file
        page_index: Zero-based page index
        scale: Render scale (1.0 = 72 DPI)
        max_pixels: Maximum dimension in pixels
    
    Returns:
        (PIL Image, PageFrame, PageMetadata)
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_index]
    
    # Get page dimensions in points
    width_pt, height_pt = page.get_size()
    
    # Calculate render scale with bounds
    effective_scale = scale
    if max(width_pt, height_pt) * effective_scale > max_pixels:
        effective_scale = max_pixels / max(width_pt, height_pt)
    
    # Render page
    bitmap = page.render(scale=effective_scale)
    pil_image = bitmap.to_pil()
    
    # Normalize image
    pil_image = normalize_image(pil_image)
    
    # Get page metadata
    rotation = page.get_rotation()
    crop_box = page.get_cropbox()
    media_box = page.get_mediabox()
    
    page_metadata = PageMetadata(
        width=width_pt,
        height=height_pt,
        rotation=rotation,
        crop_box=crop_box,
        media_box=media_box,
    )
    
    # Compute transform: PDF points -> pixels
    # PDF origin is bottom-left, image origin is top-left
    # Y needs flipping
    
    # Create transform matrix
    # For now, assume scale=1 and no rotation (most common case)
    # This will be refined in native_pdf.py for complex cases
    
    transform = Transform3x3(matrix=[
        [effective_scale, 0, 0],
        [0, -effective_scale, height_pt * effective_scale],  # Flip Y
        [0, 0, 1]
    ])
    
    # Create frame
    from .contracts import sha256_file, sha256_bytes
    document_id = sha256_file(pdf_path)
    
    # Raster hash
    rgb_image = pil_image.convert("RGB")
    raster_sha256 = sha256_bytes(rgb_image.tobytes())
    
    frame = PageFrame(
        document_id=document_id,
        page_index=page_index,
        source_sha256=document_id,
        raster_sha256=raster_sha256,
        image_size=pil_image.size,
        color_convention="RGB",
        source_to_canonical=transform,
        canonical_to_source=transform.inverse(),
        page_metadata=page_metadata,
    )
    
    return pil_image, frame, page_metadata


def load_pdf(
    pdf_path: Path,
    max_pages: int = 1,
    scale: float = 1.0,
) -> List[Tuple[Image.Image, PageFrame, PageMetadata]]:
    """Load all pages of a PDF.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to load
        scale: Render scale
    
    Returns:
        List of (Image, PageFrame, PageMetadata) tuples
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    page_count = min(len(pdf), max_pages)
    
    pages = []
    for i in range(page_count):
        image, frame, metadata = render_pdf_page(pdf_path, i, scale)
        pages.append((image, frame, metadata))
    
    return pages


def detect_non_floorplan(image: Image.Image) -> Tuple[bool, str]:
    """Detect if image is not a floorplan.
    
    Returns:
        (is_valid_floorplan, reason)
    """
    import cv2
    
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    
    # Check wall density
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    wall_pixels = np.sum(binary > 0)
    wall_density = wall_pixels / (width * height)
    
    if wall_density < 0.005:
        return False, "insufficient wall structure"
    
    # Check for dense horizontal lines (sheet music)
    edges = cv2.Canny(enhanced, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=width//4, maxLineGap=10)
    
    if lines is not None:
        horizontal_lines = 0
        vertical_lines = 0
        for line in lines:
            # HoughLinesP returns shape (N, 4) where each line is [x1, y1, x2, y2]
            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
            angle = abs(np.arctan2(y2-y1, x2-x1)) * 180 / np.pi
            if angle < 10 or angle > 170:
                horizontal_lines += 1
            elif 80 < angle < 100:
                vertical_lines += 1
        
        if horizontal_lines > 30 and vertical_lines < 10:
            return False, "dense parallel horizontal lines (possible sheet music)"
        elif horizontal_lines > 50:
            return False, "excessive horizontal line density"
    
    # Check for photo-like texture
    patch_size = 16
    variances = []
    for y in range(0, height - patch_size, patch_size * 2):
        for x in range(0, width - patch_size, patch_size * 2):
            patch = gray[y:y+patch_size, x:x+patch_size]
            variances.append(np.var(patch))
    
    if variances and np.mean(variances) > 5000:
        return False, "photo-like texture (high variance)"
    
    return True, ""


def load_drawing(path: Path) -> List[Tuple[Image.Image, PageFrame, PageMetadata]]:
    """Load a drawing file (PNG, JPG, PDF) and return page data.
    
    Args:
        path: Path to drawing file
    
    Returns:
        List of (Image, PageFrame, PageMetadata) tuples
    """
    suffix = path.suffix.lower()
    
    if suffix in [".png", ".jpg", ".jpeg"]:
        image, frame = load_image(path)
        return [(image, frame, None)]
    
    elif suffix == ".pdf":
        return load_pdf(path)
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
