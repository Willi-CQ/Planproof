from __future__ import annotations

from typing import Dict, Tuple, Optional

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def compute_points_to_m(scale_ratio: Optional[str]) -> float:
    """
    PDF unit: point = 1/72 inch.
    Convert to meters on paper, then apply scale (e.g. 1:50 -> *50).
    """
    m_per_point_on_paper = 0.0254 / 72.0
    if not scale_ratio:
        return m_per_point_on_paper

    try:
        a_str, b_str = scale_ratio.split(":")
        a = float(a_str)
        b = float(b_str)
        if a <= 0 or b <= 0:
            return m_per_point_on_paper
        return m_per_point_on_paper * (b / a)
    except Exception:
        return m_per_point_on_paper


def pt_to_m_xy(x_pt: float, y_pt: float, page_h_pt: float, points_to_m: float) -> Point:
    """
    PDF coords: origin top-left, y downward.
    Convert to meters with y-up.
    """
    x_m = x_pt * points_to_m
    y_m = (page_h_pt - y_pt) * points_to_m
    return (x_m, y_m)


def bbox_pt_to_m(bbox: BBox, page_h_pt: float, points_to_m: float) -> BBox:
    x0, y0, x1, y1 = bbox
    p0 = pt_to_m_xy(x0, y0, page_h_pt, points_to_m)
    p1 = pt_to_m_xy(x1, y1, page_h_pt, points_to_m)
    return (min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))


def transform_extraction_to_m(extraction: Dict) -> Dict:
    """
    Overwrites texts[].bbox and geometry[].a/.b to meters (y-up).
    Adds doc.transform meta.

    Priority for points_to_m:
      1) scale.ratio (explicit "1:50")
      2) doc.transform.points_to_m if present (e.g. inferred from measurements)
      3) paper-default
    """
    doc = extraction.get("doc", {})
    page_h_pt = float(doc.get("page_size", {}).get("h", 0.0))
    scale = (doc.get("scale") or {})
    ratio = scale.get("ratio")

    prev_tf = doc.get("transform") or {}

    if ratio:
        points_to_m = compute_points_to_m(ratio)
        source = "scale_ratio"
    else:
        # accept inferred/manual override if present
        ptm_override = prev_tf.get("points_to_m")
        try:
            ptm_override = float(ptm_override) if ptm_override is not None else None
        except Exception:
            ptm_override = None

        if ptm_override and ptm_override > 0:
            points_to_m = ptm_override
            source = str(prev_tf.get("source") or "override")
        else:
            # Fallback: typischer Architekturmaßstab 1:100, wenn OCR keine Maße lieferte
            fallback_ratio = prev_tf.get("fallback_scale_ratio") or "1:100"
            points_to_m = compute_points_to_m(fallback_ratio)
            source = f"fallback_{fallback_ratio}"

    doc["transform"] = {
        "from": "pdf_points_y_down",
        "to": "meters_y_up",
        "points_to_m": float(points_to_m),
        "page_h_pt": float(page_h_pt),
        "source": source,
    }
    extraction["doc"] = doc

    for t in extraction.get("texts", []):
        bbox = t.get("bbox")
        if bbox and len(bbox) == 4:
            t["bbox"] = list(bbox_pt_to_m(tuple(bbox), page_h_pt, points_to_m))

    for g in extraction.get("geometry", []):
        if g.get("type") != "line":
            continue
        a = g.get("a")
        b = g.get("b")
        if a and b and len(a) == 2 and len(b) == 2:
            g["a"] = list(pt_to_m_xy(a[0], a[1], page_h_pt, points_to_m))
            g["b"] = list(pt_to_m_xy(b[0], b[1], page_h_pt, points_to_m))

    return extraction
