"""OCR text extraction using pytesseract.

Extracts text spans with bounding boxes, orientation, and confidence.
Degrades gracefully if tesseract binary is not available.
"""

from __future__ import annotations

import shutil
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

from .contracts import (
    Observation,
    ObservationKind,
    PageFrame,
    SourceFamily,
    Mode,
)


def check_tesseract_available() -> bool:
    """Check if tesseract binary is available."""
    return shutil.which("tesseract") is not None


def extract_ocr_spans(
    image: np.ndarray,
    frame: PageFrame,
    languages: str = "eng",
) -> List[Observation]:
    """Extract OCR text spans from image.
    
    Args:
        image: RGB image array
        frame: PageFrame for the image
        languages: Tesseract language codes
    
    Returns:
        List of Observation objects for text spans
    
    Raises:
        RuntimeError: If tesseract is not available
    """
    if not check_tesseract_available():
        raise RuntimeError("tesseract binary not found")
    
    # Convert numpy array to PIL Image
    if len(image.shape) == 3:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        pil_image = Image.fromarray(image)
    
    # Get OCR data with bounding boxes
    data = pytesseract.image_to_data(
        pil_image,
        lang=languages,
        output_type=pytesseract.Output.DICT,
        config="--psm 6",  # Assume uniform block of text
    )
    
    observations = []
    n_boxes = len(data["text"])
    
    for i in range(n_boxes):
        text = data["text"][i].strip()
        if not text:
            continue
        
        # Get bounding box
        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        
        # Get confidence
        conf = data["conf"][i]
        if conf == -1:
            conf = 0.0
        else:
            conf = conf / 100.0  # Normalize to [0, 1]
        
        # Get block/par/line/word info
        block_num = data["block_num"][i]
        par_num = data["par_num"][i]
        line_num = data["line_num"][i]
        word_num = data["word_num"][i]
        
        observations.append(Observation(
            id=f"ocr-p{frame.page_index}-b{block_num}-p{par_num}-l{line_num}-w{word_num}",
            kind=ObservationKind.TEXT,
            source_family=SourceFamily.OCR,
            mode=Mode.INDEPENDENT,
            page_index=frame.page_index,
            frame_id=f"p{frame.page_index}-canonical",
            bbox=(x, y, x + w, y + h),
            geometry={
                "text": text,
                "confidence": conf,
                "block_num": block_num,
                "par_num": par_num,
                "line_num": line_num,
                "word_num": word_num,
            },
            method="tesseract_ocr",
            method_version="v1",
            score=conf,
            score_semantics="ocr_confidence",
        ))
    
    return observations


def extract_ocr_lines(
    image: np.ndarray,
    frame: PageFrame,
    languages: str = "eng",
) -> List[dict]:
    """Extract OCR text lines (grouped words).
    
    Args:
        image: RGB image array
        frame: PageFrame for the image
        languages: Tesseract language codes
    
    Returns:
        List of line dicts with text and bbox
    """
    if not check_tesseract_available():
        return []
    
    # Convert to PIL Image
    if len(image.shape) == 3:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        pil_image = Image.fromarray(image)
    
    # Get OCR data
    data = pytesseract.image_to_data(
        pil_image,
        lang=languages,
        output_type=pytesseract.Output.DICT,
        config="--psm 6",
    )
    
    # Group by block/par/line
    lines = {}
    n_boxes = len(data["text"])
    
    for i in range(n_boxes):
        text = data["text"][i].strip()
        if not text:
            continue
        
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        
        if key not in lines:
            lines[key] = {
                "text": [],
                "bbox": [data["left"][i], data["top"][i], 
                        data["left"][i] + data["width"][i], 
                        data["top"][i] + data["height"][i]],
                "confidences": [],
            }
        
        lines[key]["text"].append(text)
        lines[key]["confidences"].append(data["conf"][i])
        
        # Update bbox
        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        
        lines[key]["bbox"][0] = min(lines[key]["bbox"][0], x)
        lines[key]["bbox"][1] = min(lines[key]["bbox"][1], y)
        lines[key]["bbox"][2] = max(lines[key]["bbox"][2], x + w)
        lines[key]["bbox"][3] = max(lines[key]["bbox"][3], y + h)
    
    # Convert to list
    result = []
    for key, line in lines.items():
        result.append({
            "id": f"ocr-line-p{frame.page_index}-b{key[0]}-p{key[1]}-l{key[2]}",
            "text": " ".join(line["text"]),
            "bbox": tuple(line["bbox"]),
            "confidence": np.mean(line["confidences"]) / 100.0 if line["confidences"] else 0.0,
        })
    
    return result
