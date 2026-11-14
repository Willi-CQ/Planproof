"""Page frame management and coordinate transforms.

Handles canonical pixel frame creation, transform computation, and
coordinate conversions between source and canonical spaces.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from .contracts import (
    PageFrame,
    PageMetadata,
    Transform3x3,
    sha256_bytes,
    sha256_file,
)


def identity_transform() -> Transform3x3:
    """Return identity 3x3 transform."""
    return Transform3x3(matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def create_page_frame(
    document_id: str,
    page_index: int,
    source_path: Path,
    image: Image.Image,
    source_to_canonical: Optional[Transform3x3] = None,
    page_metadata: Optional[PageMetadata] = None,
) -> PageFrame:
    """Create a PageFrame for an image.
    
    Args:
        document_id: SHA256 of source file
        page_index: Zero-based page index
        source_path: Path to source file
        image: PIL Image (already decoded)
        source_to_canonical: Transform from source to canonical coords
        page_metadata: Optional PDF page metadata
    
    Returns:
        PageFrame with all identity and transform info
    """
    # Get image dimensions
    width, height = image.size
    
    # Compute raster hash (RGB pixels)
    rgb_image = image.convert("RGB")
    pixel_bytes = rgb_image.tobytes()
    raster_sha256 = sha256_bytes(pixel_bytes)
    
    # Source hash
    source_sha256 = sha256_file(source_path)
    
    # Default to identity transform
    if source_to_canonical is None:
        source_to_canonical = identity_transform()
    
    canonical_to_source = source_to_canonical.inverse()
    
    # EXIF orientation
    exif_orientation = None
    try:
        exif = image.getexif()
        if exif:
            exif_orientation = exif.get(274)  # Orientation tag
    except Exception:
        pass
    
    return PageFrame(
        document_id=document_id,
        page_index=page_index,
        source_sha256=source_sha256,
        raster_sha256=raster_sha256,
        image_size=(width, height),
        color_convention="RGB",
        source_to_canonical=source_to_canonical,
        canonical_to_source=canonical_to_source,
        page_metadata=page_metadata,
        exif_orientation=exif_orientation,
    )


def apply_exif_orientation(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation to image.
    
    Returns a new image with orientation applied.
    """
    try:
        exif = image.getexif()
        if not exif:
            return image
        
        orientation = exif.get(274)
        if orientation is None:
            return image
        
        # Map EXIF orientation to PIL transpose
        transpose_map = {
            2: Image.Transpose.FLIP_LEFT_RIGHT,
            3: Image.Transpose.ROTATE_180,
            4: Image.Transpose.FLIP_TOP_BOTTOM,
            5: Image.Transpose.TRANSPOSE,
            6: Image.Transpose.ROTATE_270,
            7: Image.Transpose.TRANSVERSE,
            8: Image.Transpose.ROTATE_90,
        }
        
        if orientation in transpose_map:
            return image.transpose(transpose_map[orientation])
        
        return image
    except Exception:
        return image


def normalize_image(image: Image.Image) -> Image.Image:
    """Normalize image for processing.
    
    - Applies EXIF orientation
    - Converts to RGB if needed
    - Preserves original dimensions
    
    Returns:
        Normalized PIL Image
    """
    # Apply EXIF orientation
    image = apply_exif_orientation(image)
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image


def compute_document_id(source_path: Path) -> str:
    """Compute document ID from source file hash."""
    return sha256_file(source_path)


def validate_frame(frame: PageFrame, image: Image.Image) -> bool:
    """Validate that frame matches image.
    
    Returns True if frame is consistent with image dimensions.
    """
    width, height = image.size
    expected_size = (width, height)
    return frame.image_size == expected_size


def frame_to_dict(frame: PageFrame) -> dict:
    """Serialize frame to dict."""
    return frame.model_dump(mode="json")


def frame_from_dict(data: dict) -> PageFrame:
    """Deserialize frame from dict."""
    return PageFrame.model_validate(data)
