"""Native PDF primitive extraction using pdfplumber.

Extracts characters, text spans, lines, curves, and fills from PDF
and maps them into the canonical pixel frame.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber
import numpy as np

from .contracts import (
    Observation,
    ObservationKind,
    PageFrame,
    PageMetadata,
    SourceFamily,
    Mode,
    Transform3x3,
)


def extract_pdf_primitives(
    pdf_path: Path,
    page_index: int,
    frame: PageFrame,
) -> List[Observation]:
    """Extract native PDF primitives for a page.
    
    Args:
        pdf_path: Path to PDF file
        page_index: Zero-based page index
        frame: PageFrame with transform info
    
    Returns:
        List of Observation objects for PDF primitives
    """
    observations = []
    
    with pdfplumber.open(str(pdf_path)) as pdf:
        if page_index >= len(pdf.pages):
            return observations
        
        page = pdf.pages[page_index]
        
        # Extract characters (text)
        chars = page.chars
        for i, char in enumerate(chars):
            x0, y0, x1, y1 = char["x0"], char["y0"], char["x1"], char["y1"]
            text = char["text"]
            
            # Convert PDF coords (bottom-left) to canonical (top-left)
            # PDF y increases upward, image y increases downward
            # Need to flip Y and apply transform
            
            # For now, assume simple scale=1 transform
            # This will be refined based on actual transform
            canonical_x0 = x0 * frame.source_to_canonical.matrix[0][0]
            canonical_y0 = frame.image_size[1] - (y1 * frame.source_to_canonical.matrix[1][1])
            canonical_x1 = x1 * frame.source_to_canonical.matrix[0][0]
            canonical_y1 = frame.image_size[1] - (y0 * frame.source_to_canonical.matrix[1][1])
            
            observations.append(Observation(
                id=f"pdf-char-p{page_index}-{i}",
                kind=ObservationKind.TEXT,
                source_family=SourceFamily.NATIVE_PDF,
                mode=Mode.INDEPENDENT,
                page_index=page_index,
                frame_id=f"p{page_index}-canonical",
                bbox=(canonical_x0, canonical_y0, canonical_x1, canonical_y1),
                geometry={
                    "text": text,
                    "fontname": char.get("fontname", ""),
                    "size": char.get("size", 0),
                    "upright": char.get("upright", True),
                },
                method="pdfplumber_chars",
                method_version="v1",
            ))
        
        # Extract lines
        lines = page.lines
        for i, line in enumerate(lines):
            x0, y0, x1, y1 = line["x0"], line["y0"], line["x1"], line["y1"]
            
            # Convert to canonical coords
            canonical_x0 = x0 * frame.source_to_canonical.matrix[0][0]
            canonical_y0 = frame.image_size[1] - (y1 * frame.source_to_canonical.matrix[1][1])
            canonical_x1 = x1 * frame.source_to_canonical.matrix[0][0]
            canonical_y1 = frame.image_size[1] - (y0 * frame.source_to_canonical.matrix[1][1])
            
            observations.append(Observation(
                id=f"pdf-line-p{page_index}-{i}",
                kind=ObservationKind.RAW_SEGMENT,
                source_family=SourceFamily.NATIVE_PDF,
                mode=Mode.INDEPENDENT,
                page_index=page_index,
                frame_id=f"p{page_index}-canonical",
                bbox=(min(canonical_x0, canonical_x1), min(canonical_y0, canonical_y1),
                      max(canonical_x0, canonical_x1), max(canonical_y0, canonical_y1)),
                geometry={
                    "a": (canonical_x0, canonical_y0),
                    "b": (canonical_x1, canonical_y1),
                    "width": line.get("width", 1),
                    "stroke": line.get("stroke", True),
                    "fill": line.get("fill", False),
                },
                method="pdfplumber_lines",
                method_version="v1",
            ))
        
        # Extract curves
        curves = page.curves
        for i, curve in enumerate(curves):
            # Curves are more complex - store as path
            points = curve.get("pts", [])
            if len(points) < 2:
                continue
            
            # Convert all points
            canonical_points = []
            for pt in points:
                x, y = pt
                canonical_x = x * frame.source_to_canonical.matrix[0][0]
                canonical_y = frame.image_size[1] - (y * frame.source_to_canonical.matrix[1][1])
                canonical_points.append((canonical_x, canonical_y))
            
            # Compute bbox
            xs = [p[0] for p in canonical_points]
            ys = [p[1] for p in canonical_points]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            
            observations.append(Observation(
                id=f"pdf-curve-p{page_index}-{i}",
                kind=ObservationKind.RAW_SEGMENT,
                source_family=SourceFamily.NATIVE_PDF,
                mode=Mode.INDEPENDENT,
                page_index=page_index,
                frame_id=f"p{page_index}-canonical",
                bbox=bbox,
                geometry={
                    "points": canonical_points,
                    "stroke": curve.get("stroke", True),
                    "fill": curve.get("fill", False),
                    "width": curve.get("width", 1),
                },
                method="pdfplumber_curves",
                method_version="v1",
            ))
        
        # Extract rects (fills)
        rects = page.rects
        for i, rect in enumerate(rects):
            x0, y0, x1, y1 = rect["x0"], rect["y0"], rect["x1"], rect["y1"]
            
            # Convert to canonical coords
            canonical_x0 = x0 * frame.source_to_canonical.matrix[0][0]
            canonical_y0 = frame.image_size[1] - (y1 * frame.source_to_canonical.matrix[1][1])
            canonical_x1 = x1 * frame.source_to_canonical.matrix[0][0]
            canonical_y1 = frame.image_size[1] - (y0 * frame.source_to_canonical.matrix[1][1])
            
            observations.append(Observation(
                id=f"pdf-rect-p{page_index}-{i}",
                kind=ObservationKind.MASK_REGION,
                source_family=SourceFamily.NATIVE_PDF,
                mode=Mode.INDEPENDENT,
                page_index=page_index,
                frame_id=f"p{page_index}-canonical",
                bbox=(canonical_x0, canonical_y0, canonical_x1, canonical_y1),
                geometry={
                    "fill": rect.get("fill", False),
                    "stroke": rect.get("stroke", True),
                    "fill_color": rect.get("fill_color", None),
                    "stroke_color": rect.get("stroke_color", None),
                },
                method="pdfplumber_rects",
                method_version="v1",
            ))
    
    return observations


def extract_pdf_text_spans(
    pdf_path: Path,
    page_index: int,
    frame: PageFrame,
) -> List[Observation]:
    """Extract text spans from PDF (grouped characters).
    
    Args:
        pdf_path: Path to PDF file
        page_index: Zero-based page index
        frame: PageFrame with transform info
    
    Returns:
        List of Observation objects for text spans
    """
    observations = []
    
    with pdfplumber.open(str(pdf_path)) as pdf:
        if page_index >= len(pdf.pages):
            return observations
        
        page = pdf.pages[page_index]
        
        # Extract words (grouped characters)
        words = page.extract_words()
        
        for i, word in enumerate(words):
            # pdfplumber extract_words returns x0, x1, top, bottom (not y0, y1)
            x0 = word.get("x0", 0)
            x1 = word.get("x1", 0)
            top = word.get("top", 0)
            bottom = word.get("bottom", 0)
            text = word.get("text", "")
            
            # Convert to canonical coords (PDF y increases upward, image y increases downward)
            canonical_x0 = x0 * frame.source_to_canonical.matrix[0][0]
            canonical_y0 = frame.image_size[1] - (bottom * frame.source_to_canonical.matrix[1][1])
            canonical_x1 = x1 * frame.source_to_canonical.matrix[0][0]
            canonical_y1 = frame.image_size[1] - (top * frame.source_to_canonical.matrix[1][1])
            
            observations.append(Observation(
                id=f"pdf-word-p{page_index}-{i}",
                kind=ObservationKind.TEXT,
                source_family=SourceFamily.NATIVE_PDF,
                mode=Mode.INDEPENDENT,
                page_index=page_index,
                frame_id=f"p{page_index}-canonical",
                bbox=(canonical_x0, canonical_y0, canonical_x1, canonical_y1),
                geometry={
                    "text": text,
                    "confidence": word.get("confidence", 1.0),
                    "fontname": word.get("fontname", ""),
                    "size": word.get("size", 0),
                },
                method="pdfplumber_words",
                method_version="v1",
            ))
    
    return observations
