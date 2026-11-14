"""Unit-safe dimensions, explicit calibration and text-supported room categories."""

from __future__ import annotations

import re

import numpy as np
from shapely.geometry import LineString, Point

ROOM_ALIASES = {
    "kitchen": ("kitchen", "küche", "keittiö"),
    "living": ("living room", "wohnzimmer", "olohuone"),
    "dining": ("dining", "esszimmer", "ruokailu"),
    "bedroom": ("bedroom", "schlafzimmer", "makuuhuone"),
    "bathroom": ("bathroom", "badezimmer", "wc", "kylpyhuone"),
    "hallway": ("hallway", "flur", "eteinen"),
    "storage": ("storage", "abstellraum", "varasto"),
    "garage": ("garage", "autotalli"),
    "balcony": ("balcony", "balkon", "parveke"),
    "other": (),
}


def parse_dimension(text):
    """No implicit metre assignment, no typical-door calibration, no lost alternatives."""
    original = text
    text = text.strip().lower().replace("²", "2")
    match = re.fullmatch(r"([+]?[0-9]+(?:[.,][0-9]+)?)\s*(mm|cm|m|ft|in)(2)?", text)
    if not match:
        if re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?", text):
            return {
                "text": original,
                "status": "ambiguous_unit",
                "numeric_value": float(text.replace(",", ".")),
                "alternatives": ["unitless label", "dimension with unspecified unit"],
            }
        return {"text": original, "status": "not_explicit_dimension"}
    number, unit, squared = match.groups()
    if re.fullmatch(r"[0-9]{1,3}[.,][0-9]{3}", number):
        return {
            "text": original,
            "status": "ambiguous_number",
            "unit": unit,
            "alternatives": ["decimal separator", "thousands separator"],
            "reason": "locale/format evidence required",
        }
    value = float(number.replace(",", "."))
    factor = {"mm": 0.001, "cm": 0.01, "m": 1, "ft": 0.3048, "in": 0.0254}[unit]
    return {
        "text": original,
        "status": "explicit",
        "value": value,
        "unit": unit + ("2" if squared else ""),
        "quantity": "area" if squared else "length",
        "si_value": value * factor ** (2 if squared else 1),
    }


def associate_dimension(span, segments, max_distance_px=30):
    """Proximity/orientation gives alternatives, never automatically a trusted anchor.

    Endpoint/extension-line verification remains mandatory before an association
    can calibrate a page. Native text is still potentially a title or stale note.
    """
    parsed = parse_dimension(span["text"])
    if parsed["status"] != "explicit" or parsed["quantity"] != "length":
        return {"parsed": parsed, "status": "unresolved", "alternatives": []}
    x0, y0, x1, y1 = span["bbox"]
    center = Point((x0 + x1) / 2, (y0 + y1) / 2)
    alternatives = []
    for index, segment in enumerate(segments):
        line = LineString(segment)
        distance = line.distance(center)
        if distance <= max_distance_px and line.length > 0:
            alternatives.append(
                {
                    "segment_index": index,
                    "distance_px": float(distance),
                    "pixel_length": float(line.length),
                    "status": "needs_extension_endpoint_verification",
                }
            )
    return {
        "parsed": parsed,
        "status": "candidate_associations",
        "alternatives": sorted(alternatives, key=lambda a: a["distance_px"])[:3],
    }


def calibrate(anchors, *, scope="page", tolerance=0.03):
    """Require two distinct verified anchors; reject mixed views and inconsistent scales."""
    usable = [a for a in anchors if a.get("verified_endpoints") and a.get("scope", "page") == scope]
    unique = {a["id"]: a for a in usable}
    if len(unique) < 2:
        return {
            "status": "unresolved",
            "scope": scope,
            "reason": "two distinct verified anchors required",
        }
    factors, ids, weights = [], [], []
    seen_segments = set()
    for a in unique.values():
        parsed = parse_dimension(a["text"])
        if parsed["status"] != "explicit" or parsed["quantity"] != "length":
            continue
        endpoints = np.asarray(a["endpoints"], dtype=float)
        if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all():
            raise ValueError("Calibration endpoints invalid")
        key = tuple(sorted(tuple(p) for p in endpoints))
        length = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        if key in seen_segments or length <= 0 or parsed["si_value"] <= 0:
            continue
        seen_segments.add(key)
        factors.append(parsed["si_value"] / length)
        ids.append(a["id"])
        weights.append(length)
    if len(factors) < 2:
        return {
            "status": "unresolved",
            "scope": scope,
            "reason": "insufficient distinct explicit length anchors",
        }
    order = np.argsort(factors)
    cumulative = np.cumsum(np.asarray(weights)[order])
    median = float(np.asarray(factors)[order][np.searchsorted(cumulative, cumulative[-1] / 2)])
    residual = np.abs(np.asarray(factors) / median - 1)
    if np.max(residual) > tolerance:
        return {
            "status": "conflict",
            "scope": scope,
            "anchor_ids": ids,
            "factors_m_per_px": factors,
            "relative_residuals": residual.tolist(),
            "reason": "inconsistent scale; no automatic view mixing",
        }
    return {
        "status": "supported",
        "scope": scope,
        "m_per_px": median,
        "anchor_ids": ids,
        "relative_residuals": residual.tolist(),
        "tolerance": tolerance,
    }


def quantities(polygon, calibration=None):
    result = {
        "definition": "polygon_outer_minus_holes",
        "area_px2": float(polygon.area),
        "perimeter_px": float(polygon.length),
        "area_m2": None,
        "perimeter_m": None,
    }
    if calibration and calibration.get("status") == "supported":
        factor = calibration["m_per_px"]
        result.update(area_m2=polygon.area * factor**2, perimeter_m=polygon.length * factor)
    return result


def classify_room_text(polygon, spans):
    results = []
    for span in spans:
        x0, y0, x1, y1 = span["bbox"]
        if not polygon.covers(Point((x0 + x1) / 2, (y0 + y1) / 2)):
            continue
        text = span["text"].casefold()
        for category, aliases in ROOM_ALIASES.items():
            if any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text) for alias in aliases):
                results.append({"category": category, "text_id": span["id"], "text": span["text"]})
    categories = sorted({r["category"] for r in results})
    return {
        "category": categories[0] if len(categories) == 1 else "unknown",
        "status": "supported_text"
        if len(categories) == 1
        else ("conflicting_text" if categories else "no_semantic_text"),
        "evidence": results,
    }
