"""Connect the teammate's pinned raw pixel extraction to Track 2, never to labels/ML."""

from __future__ import annotations

import shutil
import time
import importlib.metadata

import cv2
import numpy as np

from . import track1_pixels as upstream
from .pixels import wall_mask_from_ink

UPSTREAM_COMMIT = "24e37023c9441c657050b8595f1cf738230e6b17"
COMPONENT_BUDGET = 2000


def runtime_identity(config):
    binary = shutil.which("tesseract") if config.enable_ocr else None
    try:
        wrapper = importlib.metadata.version("pytesseract")
    except importlib.metadata.PackageNotFoundError:
        wrapper = None
    from pathlib import Path

    stat = Path(binary).stat() if binary else None
    return {
        "ocr_enabled": config.enable_ocr,
        "binary_available": bool(binary),
        "wrapper_version": wrapper,
        "binary_size": stat.st_size if stat else None,
        "binary_mtime_ns": stat.st_mtime_ns if stat else None,
        "languages": config.ocr_languages,
    }


def ocr_spans(image, languages):
    """Sparse drawing OCR; RGB retained, engine score never becomes correctness."""
    if not shutil.which("tesseract"):
        return [], {"status": "unavailable", "reason": "Tesseract executable not installed/on PATH"}
    try:
        import pytesseract
        from PIL import Image

        data = pytesseract.image_to_data(
            Image.fromarray(image),
            lang=languages,
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
            timeout=20,
        )
        result = []
        h, w = image.shape[:2]
        for i, raw in enumerate(data["text"]):
            text = str(raw).strip()
            if not text:
                continue
            score = float(data["conf"][i])
            x, y = int(data["left"][i]), int(data["top"][i])
            right, bottom = x + int(data["width"][i]), y + int(data["height"][i])
            bbox = [max(0, x), max(0, y), min(w, right), min(h, bottom)]
            if not np.isfinite(score) or score < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            result.append(
                {
                    "id": f"track1-ocr-{i:05d}",
                    "text": text,
                    "bbox": bbox,
                    "source_family": "ocr",
                    "engine_score": min(100, score) / 100,
                    "score_semantics": "Tesseract confidence, not calibrated correctness",
                    "engine": "tesseract",
                    "languages": languages,
                    "psm": 11,
                }
            )
        return result, {
            "status": "available",
            "word_count": len(result),
            "languages": languages,
            "configuration": "--psm 11; original RGB",
        }
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return [], {"status": "unavailable", "reason": str(exc)}


def extract_primitives(image, config):
    start = time.perf_counter()
    height, width = image.shape[:2]
    factor = min(1.0, config.max_side / max(height, width))
    w, h = max(1, round(width * factor)), max(1, round(height * factor))
    working = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA) if factor < 1 else image
    mask = upstream.extract_ink_mask(working)
    # Track 1 uses 0/255; Track 2 expects exactly uint8 0/1.
    ink = (mask > 0).astype(np.uint8)
    wall, stroke = wall_mask_from_ink(ink)
    segments = upstream.extract_line_segments(mask)
    raw_segments = [[list(map(float, s["a"])), list(map(float, s["b"]))] for s in segments]
    components = upstream.extract_components(mask)
    rings = upstream.extract_contour_rings(mask)
    openings = upstream.detect_opening_candidates(mask)
    stairs = upstream.detect_stair_candidates(mask)
    cues = []
    for cue in openings + stairs:
        cues.append(
            {
                "id": "track1:" + cue["id"],
                "bbox": list(map(float, cue["bbox"])),
                "class": "unknown",
                "upstream_kind": cue["kind"],
                "reason": "raw shape/size cue; not sufficient for semantic classification",
                "features": {k: v for k, v in cue.items() if k not in ("id", "bbox", "kind")},
            }
        )
    warnings = []
    if len(components) > COMPONENT_BUDGET or len(rings) > COMPONENT_BUDGET * 2:
        warnings.append(
            "Track 1 component/contour detail budget truncated; original ink mask retained"
        )
    text, ocr = (
        ocr_spans(working, config.ocr_languages)
        if config.enable_ocr
        else ([], {"status": "disabled"})
    )
    details = {
        "backend": "github-track1",
        "upstream_commit": UPSTREAM_COMMIT,
        "independent": True,
        "mask_normalization": "upstream uint8 0/255 -> uint8 0/1; no image orientation change",
        "working_to_canonical": [[width / w, 0, 0], [0, height / h, 0], [0, 0, 1]],
        "counts": {
            "components": len(components),
            "contour_rings": len(rings),
            "raw_segments": len(segments),
            "opening_cues": len(openings),
            "stair_cues": len(stairs),
        },
        "components": [
            {
                "id": c["id"],
                "bbox": list(map(int, c["bbox"])),
                "area_px2": float(c["area"]),
                "contours": [r[:, 0, :].tolist() for r in c["contours"]],
            }
            for c in components[:COMPONENT_BUDGET]
        ],
        "contour_rings": [
            {**r, "is_outer": bool(r["is_outer"]), "parent_idx": int(r["parent_idx"])}
            for r in rings[: COMPONENT_BUDGET * 2]
        ],
        "cues": cues,
        "raw_segment_methods": {
            name: sum(s["method"] == name for s in segments) for name in ("lsd", "hough")
        },
        "ocr": ocr,
        "source_scores": "fixed upstream line scores omitted; not correctness",
        "seconds": time.perf_counter() - start,
    }
    return {
        "working_size": [w, h],
        "working_to_canonical": details["working_to_canonical"],
        "ink": ink,
        "wall_mask": wall,
        "stroke_px": stroke,
        "raw_segments": raw_segments,
        "mode": "independent",
        "text_spans": text,
        "unclassified_patterns": cues,
        "pixel_details": details,
        "warnings": warnings,
        "availability": {
            "walls": "track1_ink_plus_track2_body_filter",
            "ocr": ocr["status"],
            "components": "available",
            "contours": "available",
            "raw_symbol_cues": "unclassified",
        },
    }
