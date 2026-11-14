from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ParsedScale:
    found: bool
    ratio: Optional[str] = None  # "1:50"
    confidence: float = 0.0
    source_text_id: Optional[str] = None


@dataclass
class ParsedMeasurement:
    id: str
    raw_text_id: str
    kind: str              # "length" | "area" | "height"
    value: float
    unit: str              # "mm" | "cm" | "m" | "m2"
    confidence: float      # 0..1


# Explicit scale (prefixed with Maßstab/Scale/M) — authoritative when found.
_SCALE_EXPLICIT_RE = re.compile(r"(?i)\b(?:m(?:aßstab)?|scale)\s*[: ]*\s*(\d+)\s*[:/]\s*(\d+)\b")
# Bare "N:M" pattern — only accepted when nothing else in the text looks like
# area/sqft/dimension. Restricted to typical architectural ratios in the
# filter step below.
_SCALE_BARE_RE = re.compile(r"\b(\d+)\s*[:/]\s*(\d+)\b")
# Tokens that indicate the text is about area/sqft/etc., not scale.
_NOT_SCALE_CONTEXT_RE = re.compile(
    r"(?i)(sq\.?\s*ft|sqft|ft[²2]?|\bft\b|sq\.?\s*m|m[²2]|\bqm\b|\bm2\b|fläche|flaeche|\barea\b|floor\s*\d|\d+\s*,\s*\d{3})"
)
# Typical architectural scales are 1:1 exact, 1:10 … 1:2000. Outside this
# band we reject the match even if the regex hit (e.g. "1:1,214" -> 1:1
# from a sqft label).
_SCALE_MIN_RATIO = 10       # 1:10 — detailed door/window detail
_SCALE_MAX_RATIO = 2000     # 1:2000 — rare overview plan
_SCALE_ALLOWED_NUMERATORS = {1}  # only "1:X" is a valid architectural scale
_LEN_RE = re.compile(r"(?i)(?<!\d)(\d+(?:[.,]\d+)?)\s*(mm|cm|m)\b")
_AREA_RE = re.compile(r"(?i)\b(\d+(?:[.,]\d+)?)\s*(m²|m2|qm|sq\.?\s*m)\b")


def _is_height_prefix(prefix: str) -> bool:
    """Detect height annotations like 'h', 'h ', 'ho', 'Hoehe', 'Height', OCR variants."""
    p = prefix.strip().lower()
    if not p:
        return False
    # Explicit height words
    if p in ("h", "hoehe", "höhe", "height", "rh", "raumhoehe", "raumhöhe"):
        return True
    # Short prefix starting with 'h' + up to 2 OCR-garbage chars (not a real word)
    if len(p) <= 4 and p.startswith("h"):
        _NON_HEIGHT = {"haus", "hall", "holz", "hof", "have", "half"}
        return p not in _NON_HEIGHT
    return False
_BARE_NUM_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*$")


def _to_float(num: str) -> float:
    return float(num.replace(",", ".").strip())


def _ocr_conf01(t: Dict) -> Optional[float]:
    # raster texts may carry "conf" 0..100; vector has -1
    try:
        c = float(t.get("conf", -1.0))
    except Exception:
        return None
    if c < 0:
        return None
    return max(0.0, min(1.0, c / 100.0))


def _is_plausible_arch_scale(a: int, b: int) -> bool:
    """Architectural plans use 1:N ratios in a narrow band. Reject obvious
    false positives like "1:1" (lifesize, matched from 'FLOOR 1: 1,214 sqft')
    or weirdly large ratios (e.g. '5:7' from a room number).
    """
    if a not in _SCALE_ALLOWED_NUMERATORS:
        return False
    if b < _SCALE_MIN_RATIO or b > _SCALE_MAX_RATIO:
        return False
    return True


def parse_scale(texts: List[Dict]) -> ParsedScale:
    # First pass: look for explicit "Maßstab/Scale N:M" — authoritative.
    for t in texts:
        s = (t.get("text") or "").strip()
        m = _SCALE_EXPLICIT_RE.search(s)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if not _is_plausible_arch_scale(a, b):
            continue
        c01 = _ocr_conf01(t)
        conf = 0.95 if c01 is None else min(0.95, max(0.6, c01))
        return ParsedScale(found=True, ratio=f"{a}:{b}", confidence=conf, source_text_id=t.get("id"))

    # Second pass: bare "N:M" — only when the surrounding text isn't obviously
    # about area/sqft/floor numbers, and the ratio is architecturally plausible.
    for t in texts:
        s = (t.get("text") or "").strip()
        if not s:
            continue
        if _NOT_SCALE_CONTEXT_RE.search(s):
            # "FLOOR 1: 1,214 sq. ft" — sqft/floor/1,214 → not a scale.
            continue
        m = _SCALE_BARE_RE.search(s)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if not _is_plausible_arch_scale(a, b):
            continue
        c01 = _ocr_conf01(t)
        conf = 0.75 if c01 is None else min(0.75, max(0.5, c01))
        return ParsedScale(found=True, ratio=f"{a}:{b}", confidence=conf, source_text_id=t.get("id"))

    return ParsedScale(found=False)


def parse_measurements(texts: List[Dict]) -> List[ParsedMeasurement]:
    out: List[ParsedMeasurement] = []
    mid = 1

    for t in texts:
        text_id = str(t.get("id"))
        s = (t.get("text") or "").strip()
        if not s:
            continue

        ocr01 = _ocr_conf01(t)

        # Area
        am = _AREA_RE.search(s)
        if am:
            val = _to_float(am.group(1))
            base = 0.85
            conf = base if ocr01 is None else min(base, ocr01)
            out.append(ParsedMeasurement(
                id=f"m{mid}", raw_text_id=text_id, kind="area",
                value=val, unit="m2", confidence=conf,
            ))
            mid += 1
            continue

        # Length with explicit unit
        lm = _LEN_RE.search(s)
        if lm:
            val = _to_float(lm.group(1))
            unit = lm.group(2).lower()
            # Detect height annotations: text before the number contains "h", "Hoehe", etc.
            prefix = s[:lm.start()].strip()
            is_height = _is_height_prefix(prefix)
            kind = "height" if is_height else "length"
            base = 0.80
            conf = base if ocr01 is None else min(base, ocr01)
            out.append(ParsedMeasurement(
                id=f"m{mid}", raw_text_id=text_id, kind=kind,
                value=val, unit=unit, confidence=conf,
            ))
            mid += 1
            continue

        # Bare numbers (common on dimension texts) -> interpret as meters cautiously
        bm = _BARE_NUM_RE.match(s)
        if bm:
            val = _to_float(bm.group(1))

            # Exclude common non-measurement patterns:
            # - Room numbers: "1.01", "2.03" (two decimal places)
            # - Floor labels that snuck through: single digits < 0.3
            if _is_room_number(s):
                continue

            if 0.30 <= val <= 25.0:  # plausible architectural dimension range
                # Check if this text is near a dimension line (higher confidence)
                is_dim = t.get("is_dimension_mainline") or t.get("is_dimension_line")
                base = 0.55 if is_dim else 0.20
                conf = base if ocr01 is None else min(base, ocr01)
                out.append(ParsedMeasurement(
                    id=f"m{mid}", raw_text_id=text_id, kind="length",
                    value=val, unit="m", confidence=conf,
                ))
                mid += 1
                continue

    return out


# Floor/room number patterns to exclude from bare-number measurements
_ROOM_NUM_RE = re.compile(r"^\s*\d{1,2}\.\d{2,3}\s*$")  # "1.01", "2.03", "12.001"
_FLOOR_LABEL_RE = re.compile(r"(?i)^\s*(?:EG|OG|UG|DG|KG|\d+\.?\s*OG|\d+\.?\s*UG)\s*$")


def _is_room_number(text: str) -> bool:
    """Check if text looks like a room number or floor label, not a dimension."""
    s = text.strip()
    if _ROOM_NUM_RE.match(s):
        return True
    if _FLOOR_LABEL_RE.match(s):
        return True
    return False
