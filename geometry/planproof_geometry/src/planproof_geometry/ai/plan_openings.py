"""
BIM-Öffnungen: Wände an Türen/Fenstern aufteilen.

Verwendet YOLO-Elemente (door, window) mit BBox, um Wandsegmente zu splitten.
Ergebnis: Wände mit Lücken statt durchgehender Extrusion, Türen/Fenster als Öffnungen.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2

OPENING_CLASSES = frozenset({"door", "window"})
WALL_CLASSES = frozenset({"wall", "raw_line"})


def _to_xyxy(bbox: Any) -> Tuple[float, float, float, float]:
    """Normalize bbox from list or dict to (x1, y1, x2, y2)."""
    if isinstance(bbox, dict):
        return (
            float(bbox.get("x_min", bbox.get("x1", bbox.get(0, 0)))),
            float(bbox.get("y_min", bbox.get("y1", bbox.get(1, 0)))),
            float(bbox.get("x_max", bbox.get("x2", bbox.get(2, 0)))),
            float(bbox.get("y_max", bbox.get("y2", bbox.get(3, 0)))),
        )
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def _line_rect_intersection_t(
    ax: float, ay: float, bx: float, by: float,
    x1: float, y1: float, x2: float, y2: float,
) -> List[float]:
    """
    Finde t-Werte (0..1) wo Segment (a,b) die Rechteckkanten schneidet.
    P(t) = a + t*(b-a). Gibt sortierte t-Liste zurück.
    """
    dx = bx - ax
    dy = by - ay
    ts: List[float] = []

    def add_t(denom: float, num: float) -> None:
        if abs(denom) > 1e-12:
            t = num / denom
            if 0 <= t <= 1:
                ts.append(t)

    add_t(dx, x1 - ax)
    add_t(dx, x2 - ax)
    add_t(dy, y1 - ay)
    add_t(dy, y2 - ay)

    return sorted(set(ts))


def _point_on_segment(ax: float, ay: float, bx: float, by: float, t: float) -> Point:
    return (ax + t * (bx - ax), ay + t * (by - ay))


def _point_in_rect(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    return x1 <= px <= x2 and y1 <= py <= y2


def _segment_length(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(bx - ax, by - ay)


def split_wall_at_opening(
    seg: Dict[str, Any],
    bbox: BBox,
    min_segment_m: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Teilt ein Wandsegment an einer Tür-/Fenster-BBox.

    Wenn das Segment die BBox schneidet, wird es in bis zu 2 Teile zerlegt
    (vor und nach der Öffnung). Der Teil innerhalb der BBox wird entfernt.

    Args:
        seg: Segment-Dict mit a, b, id, class, ...
        bbox: (x1, y1, x2, y2) der Öffnung
        min_segment_m: Minimale Länge für behaltene Teile

    Returns:
        Liste von 0, 1 oder 2 Segmenten (1 = unverändert, 0 = komplett in Öffnung)
    """
    a = seg.get("a")
    b = seg.get("b")
    if not a or not b:
        return [seg]

    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    x1, y1, x2, y2 = bbox
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    ts = _line_rect_intersection_t(ax, ay, bx, by, x1, y1, x2, y2)
    if len(ts) < 2:
        a_in = _point_in_rect(ax, ay, x1, y1, x2, y2)
        b_in = _point_in_rect(bx, by, x1, y1, x2, y2)
        if a_in and b_in:
            return []
        return [seg]

    t_enter, t_exit = ts[0], ts[-1]
    p_enter = _point_on_segment(ax, ay, bx, by, t_enter)
    p_exit = _point_on_segment(ax, ay, bx, by, t_exit)

    result: List[Dict[str, Any]] = []
    base_id = str(seg.get("id", "g"))
    kind = seg.get("class", seg.get("kind", "raw_line"))

    if t_enter > 1e-6:
        p0 = (ax, ay)
        p1 = p_enter
        L = _segment_length(p0[0], p0[1], p1[0], p1[1])
        if L >= min_segment_m:
            result.append({
                **seg,
                "id": f"{base_id}_a",
                "a": list(p0),
                "b": list(p1),
                "class": kind,
                "kind": kind,
            })

    if t_exit < 1 - 1e-6:
        p0 = p_exit
        p1 = (bx, by)
        L = _segment_length(p0[0], p0[1], p1[0], p1[1])
        if L >= min_segment_m:
            result.append({
                **seg,
                "id": f"{base_id}_b",
                "a": list(p0),
                "b": list(p1),
                "class": kind,
                "kind": kind,
            })

    return result if result else [seg]


def _bbox_center_segment(
    elem: Dict[str, Any],
    seg_id_prefix: str,
) -> Optional[Dict[str, Any]]:
    """Create a line segment through the opening bbox.

    Uses the wall direction stashed by SnapOpeningsToWallStep
    (meta.wall_direction + meta.opening_length_m) if available — the
    opening then lies EXACTLY along the wall's axis regardless of bbox
    aspect ratio. This is the key to vertical-window correctness
    (window in a vertical wall has a bbox that is tall+narrow, but the
    opening midline must still run PARALLEL to the wall, not horizontal
    through the bbox).

    Falls back to bbox-long-axis midline when no wall_direction is set
    (YOLO detections without snap).
    """
    raw_bbox = elem.get("bbox")
    if not raw_bbox:
        return None
    x1, y1, x2, y2 = _to_xyxy(raw_bbox)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    cls = elem.get("class", "door").lower()
    meta = elem.get("meta") or {}
    wall_dir = meta.get("wall_direction")
    opening_len = meta.get("opening_length_m")

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    if wall_dir and isinstance(wall_dir, (list, tuple)) and len(wall_dir) == 2 and opening_len:
        ux, uy = float(wall_dir[0]), float(wall_dir[1])
        half = float(opening_len) * 0.5
        ax, ay = cx - ux * half, cy - uy * half
        bx, by = cx + ux * half, cy + uy * half
    else:
        # Fallback: midline along the long axis
        bw = x2 - x1
        bh = y2 - y1
        if bw >= bh:
            ax, ay = x1, cy
            bx, by = x2, cy
        else:
            ax, ay = cx, y1
            bx, by = cx, y2

    seg = {
        "a": [ax, ay],
        "b": [bx, by],
        "id": f"{seg_id_prefix}_{cls}",
        "class": cls,
        "kind": cls,
        "confidence": float(elem.get("confidence", 0.7)),
        "source": elem.get("source") or "yolo_opening",
    }
    # Propagate subtype (door_double, door_sliding, window_large) so the
    # BIM exporter / Phase-E can render them differently.
    subtype = meta.get("manual_class_raw") or meta.get("subtype")
    if subtype:
        seg["meta"] = {"subtype": subtype, **({k: v for k, v in meta.items() if k != "subtype"})}
    return seg


def compute_openings(
    segments: List[Dict[str, Any]],
    elements: List[Dict[str, Any]],
    min_segment_m: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Wendet Tuer-/Fenster-Oeffnungen auf Wandsegmente an.

    - Wandsegmente, die eine Oeffnungs-BBox schneiden, werden aufgeteilt
    - Door/Window-Segmente bleiben erhalten (werden als Oeffnungen extrudiert)
    - Fuer YOLO-Oeffnungs-Elemente ohne passendes Segment wird ein neues erzeugt

    Args:
        segments: Annotierte Segmente aus SemanticFusion
        elements: YOLO-Elemente mit bbox
        min_segment_m: Minimale Laenge fuer Wandreste

    Returns:
        Neue Segmentliste mit aufgeteilten Waenden und Oeffnungs-Segmenten
    """
    opening_elements = [
        e for e in elements
        if e.get("class", "").lower() in OPENING_CLASSES
        and e.get("bbox")
    ]
    if not opening_elements:
        return segments

    wall_segments = [
        s for s in segments
        if s.get("class", s.get("kind", "raw_line")) in WALL_CLASSES
    ]
    opening_segments = [
        s for s in segments
        if s.get("class", s.get("kind", "raw_line")) in OPENING_CLASSES
    ]
    non_wall_segments = [
        s for s in segments
        if s.get("class", s.get("kind", "raw_line")) not in WALL_CLASSES
        and s.get("class", s.get("kind", "raw_line")) not in OPENING_CLASSES
    ]

    # Split walls at opening bboxes
    split_walls: List[Dict[str, Any]] = []
    for seg in wall_segments:
        current = [seg]

        for elem in opening_elements:
            bbox = _to_xyxy(elem["bbox"])
            next_current: List[Dict[str, Any]] = []
            for s in current:
                parts = split_wall_at_opening(s, bbox, min_segment_m)
                next_current.extend(parts)
            current = next_current

        split_walls.extend(current)

    # Generate opening segments from YOLO elements that have no matching segment
    generated: List[Dict[str, Any]] = []
    for idx, elem in enumerate(opening_elements):
        raw_bbox = elem.get("bbox")
        if not raw_bbox:
            continue
        x1, y1, x2, y2 = _to_xyxy(raw_bbox)
        cx_el, cy_el = (x1 + x2) * 0.5, (y1 + y2) * 0.5

        has_match = False
        for os_seg in opening_segments:
            a, b = os_seg.get("a"), os_seg.get("b")
            if not a or not b:
                continue
            mx = (float(a[0]) + float(b[0])) * 0.5
            my = (float(a[1]) + float(b[1])) * 0.5
            d = ((mx - cx_el) ** 2 + (my - cy_el) ** 2) ** 0.5
            diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if d < max(diag * 0.5, 0.3):
                has_match = True
                break

        if not has_match:
            new_seg = _bbox_center_segment(elem, f"opening_{idx}")
            if new_seg:
                generated.append(new_seg)

    result = non_wall_segments + split_walls + opening_segments + generated
    return result
