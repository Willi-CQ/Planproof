"""Raum-Kontext fuer die Moebel-Erkennung.

Der Plan-to-BIM-Pipeline fehlen echte Raum-Polygone — Raeume existieren nur als
``room_label``-Text-Bboxes (siehe label_schema.py). Das reicht aber, um jeder
Moebel-Bbox einen Raum zuzuordnen: Wir nehmen die Raum-Label-Bbox als grobes
Raumgebiet und ordnen Moebel per Containment / Naehe zu.

Warum das wichtig ist (User-Wunsch): Der Raum wird dem LLM **immer
mitgegeben**. "Dieser Crop liegt im Raum 'Bad'" macht aus einer mehrdeutigen
weissen Box mit Bogen ein WC statt eines Sessels. Raum-Prior schlaegt
Crop-Mehrdeutigkeit.

Koordinaten: alle Bboxes hier in Metern ``(x0, y0, x1, y1)`` im selben
Welt-Frame wie Waende/Tueren (plan-px * scale_xy * px_to_m).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..label_schema import to_plan_element_class

BBoxM = Tuple[float, float, float, float]

# Felder, in denen der Raumname stecken koennte (Annotator/Editor-Varianten).
_LABEL_TEXT_FIELDS = ("label", "text", "name", "room_type", "value", "title")

# Heuristische Raumtyp-Erkennung aus dem Label-Text (nur ein *Hinweis* an das
# LLM, keine harte Entscheidung). Reihenfolge = Prioritaet.
_ROOM_TYPE_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("Bad", ("bad", "bath", "wc", "dusche", "shower", "sanitaer", "sanitar", "toilette", "gaeste-wc", "gaeste wc", "ensuite", "powder")),
    ("Kueche", ("kueche", "kuche", "kitchen", "kochen", "kk", "pantry", "speis")),
    ("Schlafzimmer", ("schlaf", "bedroom", "kinderzimmer", "kind", "elternzimmer", "primary bedroom", "master")),
    ("Wohnzimmer", ("wohn", "living", "wohnzimmer", "wohnen", "sitting", "lounge", "family")),
    ("Esszimmer", ("ess", "dining", "essbereich", "essen")),
    ("Buero", ("buero", "buro", "office", "arbeitszimmer", "arbeit", "studio", "den", "gym", "fitness")),
    ("Flur", ("flur", "diele", "korridor", "hallway", "gang", "eingang", "entree", "entry", "landing", "foyer", "hall")),
    ("Abstellraum", ("abstell", "abst", "storage", "hwr", "technik", "keller", "vorrat", "closet", "utility", "laundry", "garage", "wardrobe room")),
    ("Balkon", ("balkon", "terrasse", "loggia", "balcony", "deck", "veranda", "patio", "porch")),
]


@dataclass
class RoomRegion:
    """Ein erkannter Raum aus einem ``room_label``-Annotation."""

    room_id: str
    label_text: str          # roher Annotator-Text (kann leer sein)
    room_type: str           # heuristischer Typ ("Bad", ...) oder "" wenn unklar
    bbox_m: BBoxM            # (x0, y0, x1, y1) in Metern

    @property
    def center_m(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bbox_m
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    def contains(self, point_m: Tuple[float, float]) -> bool:
        x0, y0, x1, y1 = self.bbox_m
        px, py = point_m
        return x0 <= px <= x1 and y0 <= py <= y1

    @property
    def area_m2(self) -> float:
        x0, y0, x1, y1 = self.bbox_m
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def describe(self) -> str:
        """Knapper Raum-Kontext-String fuer den LLM-Prompt."""
        if self.room_type and self.label_text:
            return f"{self.room_type} (Label: \"{self.label_text}\")"
        if self.room_type:
            return self.room_type
        if self.label_text:
            return f"Label: \"{self.label_text}\""
        return "unbekannter Raum"


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (s or "").strip().lower())


def classify_room_type(label_text: str) -> str:
    """Heuristischer Raumtyp aus dem Label-Text. "" wenn nichts passt."""
    n = _norm_text(label_text)
    if not n:
        return ""
    for room_type, keywords in _ROOM_TYPE_KEYWORDS:
        for kw in keywords:
            if kw in n:
                return room_type
    return ""


def _label_text_of(ann: Dict[str, Any]) -> str:
    for f in _LABEL_TEXT_FIELDS:
        v = ann.get(f)
        if isinstance(v, str) and v.strip() and v.strip().lower() != "room_label":
            return v.strip()
    meta = ann.get("meta")
    if isinstance(meta, dict):
        for f in _LABEL_TEXT_FIELDS:
            v = meta.get(f)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def _annotation_bbox_m(
    ann: Dict[str, Any],
    scale_xy: Tuple[float, float],
    px_to_m: float,
) -> Optional[BBoxM]:
    """Bbox eines Annotations in Metern (bbox oder polygon-Huelle).

    Identische Konvention wie window_label_blocks._annotation_bbox_m, damit
    Moebel/Raeume im selben Welt-Frame wie Waende/Tueren/Fenster liegen.
    """
    sx, sy = scale_xy
    if ann.get("type") == "polygon":
        pts = ann.get("points") or []
        if not pts:
            return None
        xs = [float(p[0]) * sx * px_to_m for p in pts]
        ys = [float(p[1]) * sy * px_to_m for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))
    bbox = ann.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = bbox
    xs = sorted([float(x0) * sx * px_to_m, float(x1) * sx * px_to_m])
    ys = sorted([float(y0) * sy * px_to_m, float(y1) * sy * px_to_m])
    return (xs[0], ys[0], xs[1], ys[1])


def _ann_bbox_render_px(ann: Dict[str, Any], scale_xy: Tuple[float, float]) -> Optional[BBoxM]:
    """Bbox eines Annotations in Render-Pixeln (plan-px * scale_xy) — fuer OCR-Crop."""
    sx, sy = scale_xy
    if ann.get("type") == "polygon":
        pts = ann.get("points") or []
        if not pts:
            return None
        xs = [float(p[0]) * sx for p in pts]
        ys = [float(p[1]) * sy for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))
    bbox = ann.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = bbox
    xs = sorted([float(x0) * sx, float(x1) * sx])
    ys = sorted([float(y0) * sy, float(y1) * sy])
    return (xs[0], ys[0], xs[1], ys[1])


def _ocr_room_name(image, bbox_render_px: BBoxM) -> str:
    """LLM-UNABHAENGIG: liest den auf dem Plan gedruckten Raumnamen aus der
    room_label-Region via PaddleOCR. Soft-fail -> "" wenn OCR/Bild fehlt.

    Viele Annotatoren markieren nur die Raum-REGION ohne Text zu tippen — der
    Raumname steht aber als Druck im Plan. Diese Funktion holt ihn als
    deterministische Zusatzinfo (kein LLM), die dann als Kontext an den
    Klassifikations-LLM geht.
    """
    if image is None:
        return ""
    try:
        import cv2  # noqa: PLC0415
        from ..ocr_paddle import ocr_paddle  # noqa: PLC0415
    except ImportError:
        return ""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in bbox_render_px)
    m = 8
    x0 = max(0, x0 - m)
    y0 = max(0, y0 - m)
    x1 = min(w, x1 + m)
    y1 = min(h, y1 + m)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return ""
    crop = image[y0:y1, x0:x1]
    # Kleine Crops fuer OCR hochskalieren (duenne Plan-Schrift).
    long_edge = max(crop.shape[0], crop.shape[1])
    if 0 < long_edge < 220:
        s = 220.0 / long_edge
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
    try:
        res = ocr_paddle(crop, lang="en", min_confidence=0.3)
    except Exception:  # noqa: BLE001 — OCR darf den Schritt nie brechen
        return ""
    # Laengste/erste alphabetische Textzeile als Raumname (Dims/Zahlen ignorieren).
    cands = [ln.text.strip() for ln in getattr(res, "lines", []) if any(c.isalpha() for c in ln.text)]
    cands.sort(key=lambda t: sum(c.isalpha() for c in t), reverse=True)
    return cands[0] if cands else ""


def extract_rooms(
    annotations: Sequence[Dict[str, Any]],
    *,
    scale_xy: Tuple[float, float] = (1.0, 1.0),
    px_to_m: float = 1.0,
    image=None,
) -> List[RoomRegion]:
    """Alle ``room_label``-Annotationen zu :class:`RoomRegion` extrahieren.

    Wenn ein Label keinen Text hat und ``image`` (gerenderter Plan) vorliegt,
    wird der Raumname LLM-UNABHAENGIG per OCR aus der Region gelesen (Zusatzinfo).
    """
    rooms: List[RoomRegion] = []
    for idx, ann in enumerate(annotations):
        cls = ann.get("class") or ""
        if to_plan_element_class(cls) != "room_label":
            continue
        bbox_m = _annotation_bbox_m(ann, scale_xy, px_to_m)
        if bbox_m is None:
            continue
        text = _label_text_of(ann)
        if not text and image is not None:
            px = _ann_bbox_render_px(ann, scale_xy)
            if px is not None:
                text = _ocr_room_name(image, px)
        rooms.append(RoomRegion(
            room_id=str(ann.get("id") or f"room_{idx}"),
            label_text=text,
            room_type=classify_room_type(text),
            bbox_m=bbox_m,
        ))
    return rooms


def _point_to_bbox_distance_m(point: Tuple[float, float], bbox: BBoxM) -> float:
    px, py = point
    x0, y0, x1, y1 = bbox
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return (dx * dx + dy * dy) ** 0.5


def assign_room(
    furniture_center_m: Tuple[float, float],
    rooms: Sequence[RoomRegion],
    *,
    max_snap_m: float = 1.5,
) -> Optional[RoomRegion]:
    """Den Raum bestimmen, in dem eine Moebel-Bbox liegt.

    1. Containment: Moebel-Zentrum in einer Raum-Bbox -> kleinster solcher Raum
       (kleinste Flaeche gewinnt, robust gegen verschachtelte/grosse Labels).
    2. Sonst: naechster Raum innerhalb ``max_snap_m`` (Raum-Label sitzt oft
       neben statt mittig im Raum).
    3. Sonst: None (kein Raum-Kontext).
    """
    if not rooms:
        return None

    containing = [r for r in rooms if r.contains(furniture_center_m)]
    if containing:
        return min(containing, key=lambda r: r.area_m2)

    nearest: Optional[RoomRegion] = None
    nearest_d = float("inf")
    for r in rooms:
        d = _point_to_bbox_distance_m(furniture_center_m, r.bbox_m)
        if d < nearest_d:
            nearest_d = d
            nearest = r
    if nearest is not None and nearest_d <= max_snap_m:
        return nearest
    return None
