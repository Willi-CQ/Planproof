"""LLM pre-classification step: vision-based zone detection for plan-to-BIM.

Sends the rendered plan image to a vision LLM early in the pipeline to identify:
- Building outline (bounding box of actual structure)
- Dimension line zones (border measurement areas)
- Estimated drawing scale
- Building type

Results are stored in state and used by downstream steps (SemanticFusion,
FilterGeometry) to exclude non-structural geometry.

soft_fail = True — pipeline works without LLM, results are just less clean.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# System prompt for the vision LLM
PRECLASSIFY_SYSTEM_PROMPT = """\
Du bist ein Experte fuer Architektur-Grundrisse und BIM-Modellierung.

Analysiere das Grundriss-Bild und identifiziere folgende Bereiche:

1. **building_outline**: Das Begrenzungsrechteck des eigentlichen Gebaeudes (ohne
   Bemassung, Legende, Titelblock). Angabe als normalisierte Koordinaten (0.0 bis 1.0
   relativ zur Bildgroesse).

2. **dimension_zones**: Liste von Rechtecken, die Bemassungslinien enthalten (typisch
   an den Raendern des Plans mit Massangaben wie "3,50" oder "5.20m"). Gleiche
   normalisierte Koordinaten.

3. **estimated_scale**: Der geschaetzte Zeichnungsmassstab als String (z.B. "1:100",
   "1:50", "1:200"). Beruecksichtige Papiergroesse und Gebaeudeabmessungen.

4. **building_type**: "residential" (Wohngebaeude), "commercial" (Gewerbe),
   oder "industrial" (Industrie/Lager).

5. **estimated_rooms**: Geschaetzte Anzahl der Raeume.

6. **elements**: Liste erkannter Bauelemente. Fokus auf Tueren, Fenster, Treppen und
   Stuetzen — NICHT einzelne Wandsegmente. Fuer jedes Element:
   - "class": eines von "door", "window", "stairs", "column", "furniture"
   - "bbox": {"x_min": 0.0-1.0, "y_min": 0.0-1.0, "x_max": 0.0-1.0, "y_max": 0.0-1.0}
     (normalisierte Koordinaten relativ zur Bildgroesse)
   - "confidence": 0.0-1.0 (wie sicher du bist)
   - "hint": optionaler String fuer Details, z.B. "swing_left", "swing_right",
     "double_door", "sliding_door", "french_window", "spiral_stairs"
   Tuerrichtung: Erkenne den Kreisbogen-Symbol im Plan — "swing_left" wenn der
   Bogen nach links oeffnet, "swing_right" nach rechts.

Antworte NUR als JSON. Keine Erklaerungen ausserhalb des JSON.

Beispiel-Antwort:
```json
{
  "building_outline": {"x_min": 0.12, "y_min": 0.08, "x_max": 0.88, "y_max": 0.85},
  "dimension_zones": [
    {"x_min": 0.0, "y_min": 0.0, "x_max": 0.10, "y_max": 1.0},
    {"x_min": 0.0, "y_min": 0.88, "x_max": 1.0, "y_max": 1.0}
  ],
  "estimated_scale": "1:100",
  "building_type": "residential",
  "estimated_rooms": 8,
  "elements": [
    {"class": "door", "bbox": {"x_min": 0.32, "y_min": 0.41, "x_max": 0.36, "y_max": 0.47}, "confidence": 0.9, "hint": "swing_right"},
    {"class": "door", "bbox": {"x_min": 0.55, "y_min": 0.28, "x_max": 0.59, "y_max": 0.34}, "confidence": 0.85, "hint": "double_door"},
    {"class": "window", "bbox": {"x_min": 0.12, "y_min": 0.50, "x_max": 0.12, "y_max": 0.62}, "confidence": 0.8},
    {"class": "stairs", "bbox": {"x_min": 0.70, "y_min": 0.15, "x_max": 0.82, "y_max": 0.30}, "confidence": 0.9}
  ],
  "confidence": 0.85
}
```\
"""


def _encode_plan_image(image, max_size: int = 1568) -> str:
    """Encode numpy image to base64 PNG, downsized for API efficiency."""
    import base64

    try:
        import cv2
    except ImportError:
        return ""

    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("ascii")


def _extract_json_from_text(text: str) -> str:
    """Extract JSON object from LLM response text, handling various formats."""
    import re

    text = text.strip()

    # Strategy 1: markdown code blocks (```json ... ``` or ``` ... ```)
    md_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if md_match:
        return md_match.group(1).strip()

    # Strategy 2: find the outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]

    # Strategy 3: return as-is (let json.loads try)
    return text


# Mapping of common LLM building type responses to canonical values
_BUILDING_TYPE_MAP: Dict[str, str] = {}
for _canonical, _aliases in {
    "residential": [
        "residential", "wohngebaeude", "wohngebäude", "wohnbau", "wohnen",
        "wohn", "apartment", "house", "housing", "mehrfamilienhaus",
        "einfamilienhaus", "dwelling",
    ],
    "commercial": [
        "commercial", "gewerbe", "buero", "büro", "office", "retail",
        "geschaeft", "geschäft", "handel",
    ],
    "industrial": [
        "industrial", "industrie", "lager", "warehouse", "factory",
        "fabrik", "halle", "produktion",
    ],
}.items():
    for _alias in _aliases:
        _BUILDING_TYPE_MAP[_alias] = _canonical


def parse_preclassify_response(text: str) -> Dict[str, Any]:
    """Parse the LLM JSON response for pre-classification.

    Handles various LLM response formats robustly:
    - Markdown code blocks (```json)
    - Preamble text before JSON
    - German/English field values
    - Various scale notations
    """
    import re

    json_text = _extract_json_from_text(text)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse preclassify LLM response as JSON. First 300 chars: %s",
            text[:300],
        )
        return {}

    if not isinstance(data, dict):
        logger.warning("Preclassify response is not a JSON object, got %s", type(data).__name__)
        return {}

    logger.info(
        "Preclassify JSON parsed OK, top-level keys: %s",
        list(data.keys()),
    )

    result: Dict[str, Any] = {}

    # ── building_outline ───────────────────────────────────────────
    outline = data.get("building_outline") or data.get("outline") or {}
    if isinstance(outline, dict):
        bbox_keys = ("x_min", "y_min", "x_max", "y_max")
        if all(k in outline for k in bbox_keys):
            try:
                result["building_outline"] = {
                    k: max(0.0, min(1.0, float(outline[k]))) for k in bbox_keys
                }
            except (TypeError, ValueError):
                pass

    # ── dimension_zones ────────────────────────────────────────────
    zones = data.get("dimension_zones") or data.get("dimensionZones") or []
    valid_zones = []
    if isinstance(zones, list):
        for z in zones:
            if isinstance(z, dict) and all(k in z for k in ("x_min", "y_min", "x_max", "y_max")):
                try:
                    valid_zones.append({
                        k: max(0.0, min(1.0, float(z[k])))
                        for k in ("x_min", "y_min", "x_max", "y_max")
                    })
                except (TypeError, ValueError):
                    continue
    result["dimension_zones"] = valid_zones

    # ── estimated_scale ────────────────────────────────────────────
    scale_raw = data.get("estimated_scale") or data.get("scale") or ""
    scale_str = str(scale_raw).strip()
    # Accept "1:100", "1 : 100", "M 1:50", "1/200", etc.
    scale_match = re.search(r"1\s*[:/]\s*(\d+)", scale_str)
    if scale_match:
        result["estimated_scale"] = f"1:{scale_match.group(1)}"

    # ── building_type ──────────────────────────────────────────────
    btype_raw = str(data.get("building_type") or data.get("type") or "").lower().strip()
    # Direct match first
    if btype_raw in _BUILDING_TYPE_MAP:
        result["building_type"] = _BUILDING_TYPE_MAP[btype_raw]
    else:
        # Substring match for compound values like "residential / mixed"
        for alias, canonical in _BUILDING_TYPE_MAP.items():
            if alias in btype_raw:
                result["building_type"] = canonical
                break

    # ── estimated_rooms ────────────────────────────────────────────
    rooms = data.get("estimated_rooms") or data.get("rooms") or data.get("room_count")
    if isinstance(rooms, (int, float)) and rooms > 0:
        result["estimated_rooms"] = int(rooms)

    # ── confidence ─────────────────────────────────────────────────
    conf = data.get("confidence")
    if isinstance(conf, (int, float)):
        result["confidence"] = max(0.0, min(1.0, float(conf)))

    # Elements (element-level detections)
    _VALID_CLASSES = {"door", "window", "stairs", "column", "furniture", "room_label"}
    elements_raw = data.get("elements", [])
    valid_elements = []
    if isinstance(elements_raw, list):
        for el in elements_raw:
            if not isinstance(el, dict):
                continue
            cls = el.get("class", "")
            if cls not in _VALID_CLASSES:
                continue
            bbox = el.get("bbox", {})
            if not (isinstance(bbox, dict)
                    and all(k in bbox for k in ("x_min", "y_min", "x_max", "y_max"))):
                continue
            try:
                validated = {
                    "class": cls,
                    "bbox": {
                        "x_min": max(0.0, min(1.0, float(bbox["x_min"]))),
                        "y_min": max(0.0, min(1.0, float(bbox["y_min"]))),
                        "x_max": max(0.0, min(1.0, float(bbox["x_max"]))),
                        "y_max": max(0.0, min(1.0, float(bbox["y_max"]))),
                    },
                    "confidence": max(0.0, min(1.0, float(el.get("confidence", 0.5)))),
                }
                hint = el.get("hint")
                if isinstance(hint, str) and hint:
                    validated["hint"] = hint
                valid_elements.append(validated)
            except (TypeError, ValueError):
                continue
    result["elements"] = valid_elements

    return result


def _call_preclassify_api(
    api_url: str,
    api_key: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Call the /internal/pipeline/preclassify endpoint."""
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required for LLM preclassification")

    url = f"{api_url.rstrip('/')}/internal/pipeline/preclassify"
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=180.0) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


class LLMPreClassifyStep:
    """Send plan image to vision LLM for zone detection and scale estimation.

    Runs after RenderImageStep. Results stored in state.config for downstream use.
    """
    name = "llm_preclassify"
    soft_fail = True

    def run(self, ctx, state):
        if not state.config.get("llm_preclassify_enabled", True):
            ctx.log("LLM pre-classification disabled, skipping")
            return state

        if state.input_image is None:
            ctx.log("No plan image available for pre-classification, skipping")
            return state

        api_url = state.config.get("api_url") or os.environ.get("PLANPROOF_API_URL")
        api_key = state.config.get("api_key") or os.environ.get("API_KEY")

        if not api_url or not api_key:
            ctx.log("LLM pre-classify: API not configured, skipping")
            state.quality_warnings.append("llm_preclassify_skipped_no_api")
            return state

        image_b64 = _encode_plan_image(state.input_image)
        if not image_b64:
            ctx.log("LLM pre-classify: failed to encode image, skipping")
            return state

        payload = {
            "job_id": state.config.get("job_id", "unknown"),
            "preview_image_base64": image_b64,
        }

        # Add page dimensions if available
        if state.extraction_dict:
            page_size = state.extraction_dict.get("page_size", {})
            if page_size:
                payload["page_dimensions_pt"] = {
                    "w": page_size.get("w"),
                    "h": page_size.get("h"),
                }

        ctx.log("Sending plan image for LLM pre-classification...")

        try:
            result = _call_preclassify_api(api_url, api_key, payload)
        except Exception as e:
            error_detail = str(e)[:300]
            error_type = "unknown"
            try:
                import httpx as _httpx
                if isinstance(e, _httpx.TimeoutException):
                    error_type = "timeout"
                elif isinstance(e, _httpx.HTTPStatusError):
                    error_type = f"http_{e.response.status_code}"
                elif isinstance(e, (_httpx.ConnectError, ConnectionError)):
                    error_type = "connection"
            except ImportError:
                pass
            ctx.log(f"LLM pre-classify FAILED [{error_type}]: {error_detail}")
            state.metrics["llm_preclassify_error_type"] = error_type
            state.quality_warnings.append(f"llm_preclassify_failed: {error_type}: {error_detail}")
            raise  # Let pipeline soft_fail handle this transparently

        # Log what the API returned for debugging (use `is not None` to not hide 0-valued fields)
        result_keys = [k for k in result if result[k] is not None and result[k] != [] and result[k] != ""]
        ctx.log(f"LLM pre-classify API returned keys: {result_keys}")

        # Store results in config for downstream steps to use
        if result.get("building_outline"):
            state.config["_llm_building_outline"] = result["building_outline"]
            ctx.log(f"Building outline: {result['building_outline']}")

        if result.get("dimension_zones"):
            state.config["_llm_dimension_zones"] = result["dimension_zones"]
            ctx.log(f"Dimension zones: {len(result['dimension_zones'])} areas detected")

        if result.get("estimated_scale"):
            state.config["_llm_estimated_scale"] = result["estimated_scale"]
            ctx.log(f"Estimated scale: {result['estimated_scale']}")

        if result.get("building_type"):
            state.config["_llm_building_type"] = result["building_type"]

        # Elements (element-level detections from LLM)
        elements = result.get("elements", [])
        if elements:
            state.config["_llm_elements"] = elements
            by_cls: Dict[str, int] = {}
            for el in elements:
                c = el.get("class", "?")
                by_cls[c] = by_cls.get(c, 0) + 1
            ctx.log(f"LLM elements: {sum(by_cls.values())} total -- {by_cls}")

        # Consensus score (from ensemble, if available)
        consensus = result.get("consensus_score")
        if consensus is not None:
            state.config["_llm_preclassify_consensus"] = float(consensus)

        # Metrics (guard against None values from API response)
        confidence = result.get("confidence") or 0.0
        state.metrics["llm_preclassify_confidence"] = confidence
        state.metrics["llm_preclassify_scale"] = result.get("estimated_scale") or ""
        state.metrics["llm_preclassify_building_type"] = result.get("building_type") or ""
        state.metrics["llm_preclassify_zones"] = len(result.get("dimension_zones") or [])
        state.metrics["llm_preclassify_elements"] = len(elements)

        ctx.log(
            f"LLM pre-classify: scale={result.get('estimated_scale')}, "
            f"type={result.get('building_type')}, "
            f"outline={'yes' if result.get('building_outline') else 'no'}, "
            f"dim_zones={len(result.get('dimension_zones') or [])}, "
            f"elements={len(elements)}, "
            f"confidence={confidence:.2f}"
        )
        return state


_LLM_CLASS_MAP = {"door", "window", "stairs", "column", "furniture"}


def convert_llm_elements_to_detections(
    elements: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    confidence_cap: float = 0.55,
) -> List[Dict[str, Any]]:
    """Convert normalized LLM element boxes to YOLO-format detection dicts.

    Returns list of dicts compatible with ``state.detection_result["detections"]``.
    """
    if not elements or image_width <= 0 or image_height <= 0:
        return []

    detections: List[Dict[str, Any]] = []
    for i, el in enumerate(elements):
        cls = el.get("class", "")
        if cls not in _LLM_CLASS_MAP:
            continue
        bbox = el.get("bbox", {})
        if not all(k in bbox for k in ("x_min", "y_min", "x_max", "y_max")):
            continue

        x1 = float(bbox["x_min"]) * image_width
        y1 = float(bbox["y_min"]) * image_height
        x2 = float(bbox["x_max"]) * image_width
        y2 = float(bbox["y_max"]) * image_height

        # Skip degenerate boxes
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            continue

        conf = min(confidence_cap, float(el.get("confidence", 0.5)))

        meta: Dict[str, Any] = {"source": "llm_ensemble"}
        hint = el.get("hint")
        if hint:
            meta["hint"] = hint

        detections.append({
            "id": f"llm_{i}",
            "class": cls,
            "confidence": round(conf, 3),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "polygon": None,
            "meta": meta,
        })

    return detections


def point_in_normalized_rect(
    x_norm: float, y_norm: float, rect: Dict[str, float],
) -> bool:
    """Check if a normalized (0-1) point is inside a normalized rect."""
    return (rect["x_min"] <= x_norm <= rect["x_max"]
            and rect["y_min"] <= y_norm <= rect["y_max"])


def segment_in_dimension_zone(
    a: Tuple[float, float],
    b: Tuple[float, float],
    page_w: float,
    page_h: float,
    dimension_zones: List[Dict[str, float]],
) -> bool:
    """Check if a segment's midpoint falls inside any LLM-detected dimension zone."""
    if not dimension_zones or page_w <= 0 or page_h <= 0:
        return False
    mid_x = (a[0] + b[0]) * 0.5 / page_w
    mid_y = (a[1] + b[1]) * 0.5 / page_h
    return any(point_in_normalized_rect(mid_x, mid_y, z) for z in dimension_zones)


def segment_outside_building_outline(
    a: Tuple[float, float],
    b: Tuple[float, float],
    page_w: float,
    page_h: float,
    outline: Dict[str, float],
    margin: float = 0.02,
) -> bool:
    """Check if a segment's midpoint falls outside the LLM-detected building outline.

    A small margin is applied to avoid clipping segments at the boundary.
    """
    if not outline or page_w <= 0 or page_h <= 0:
        return False
    mid_x = (a[0] + b[0]) * 0.5 / page_w
    mid_y = (a[1] + b[1]) * 0.5 / page_h
    expanded = {
        "x_min": outline["x_min"] - margin,
        "y_min": outline["y_min"] - margin,
        "x_max": outline["x_max"] + margin,
        "y_max": outline["y_max"] + margin,
    }
    return not point_in_normalized_rect(mid_x, mid_y, expanded)
