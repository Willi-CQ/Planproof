"""Deterministische Moebel-Platzierung — die "Geometrie-Erkennung".

Das LLM hat benannt WAS in der Bbox steht und in welcher Reihenfolge
(classify.py). Hier entsteht daraus die GEOMETRIE — vollstaendig
deterministisch, ohne LLM:

  - **Run-Layout** (mehrere Items): die laengere Bbox-Kante ist die Zeilen-
    Achse. Items werden mit ihren Katalog-Nennbreiten der Reihe nach gepackt
    (breitengewichtet, NICHT gleichverteilt — eine 60-cm-Spuele bleibt 60 cm).
    Passen sie nicht, werden alle proportional verkleinert. Das ist das
    Fenster-Equipartition-Pattern, verallgemeinert auf ungleiche Breiten.
  - **Einzelobjekt**: fuellt die Bbox, geklemmt auf Katalog-Nennmass +/- Faktor
    (eine schlampige Bbox erzeugt kein 3-m-WC).
  - **Orientierung**: nicht dem LLM die Winkel glauben. Die "Rueckseite" zeigt
    zur naechsten Wand, die Front in den Raum — bestimmt aus der Wand-Naehe,
    nicht aus dem LLM-Hint (der ist nur Tiebreaker ohne Wand).

3D-Konvention identisch zu Tueren/Fenstern: Plan (x, y) in Metern -> 3D
(x, hoehe, y), Y ist oben. Footprints sind 4-Punkt-Polygone (CCW) in (x, y).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .catalog import PLACEMENT_FREE, resolve_item
from .classify import FurnitureClassification

BBoxM = Tuple[float, float, float, float]
PointM = Tuple[float, float]

# Wie weit ein Einzelobjekt-Footprint vom Katalog-Nennmass abweichen darf,
# bevor wir klemmen. Moebel variieren, darum grosszuegig (halb bis doppelt).
CLAMP_MIN_FACTOR = 0.5
CLAMP_MAX_FACTOR = 2.0


@dataclass
class PlacedItem:
    """Ein einzelnes platziertes Katalog-Item innerhalb einer Bbox."""

    catalog_id: str
    order: int
    footprint_m: List[PointM]   # 4 Eckpunkte CCW in (x, y) Metern
    height_m: float
    facing_deg: float           # Rotation um Y; 0=+x, 90=+y, 180=-x, 270=-y (Front)
    confidence: float

    @property
    def centroid_m(self) -> PointM:
        return (
            sum(p[0] for p in self.footprint_m) / len(self.footprint_m),
            sum(p[1] for p in self.footprint_m) / len(self.footprint_m),
        )


@dataclass
class PlacedFurniture:
    """Alle platzierten Items einer Moebel-Bbox."""

    bbox_id: str
    zone_type: str
    items: List[PlacedItem]
    room_id: Optional[str] = None
    room_type: str = ""
    source: str = "llm"
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geometrie-Helfer
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _point_to_bbox_distance_m(point: PointM, bbox: BBoxM) -> float:
    px, py = point
    x0, y0, x1, y1 = bbox
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return (dx * dx + dy * dy) ** 0.5


def _facing_deg(front_axis: str, front_sign: int) -> float:
    if front_axis == "x":
        return 0.0 if front_sign > 0 else 180.0
    return 90.0 if front_sign > 0 else 270.0


def _rect(run_lo: float, run_hi: float, dep_lo: float, dep_hi: float,
          run_axis: str) -> List[PointM]:
    """Rechteck aus Run-/Tiefen-Intervall -> 4 CCW (x, y)-Punkte.

    ``run_axis`` sagt, welche Welt-Achse die Run-Achse ist.
    """
    if run_axis == "x":
        # run = x, depth = y
        return [
            (run_lo, dep_lo), (run_hi, dep_lo),
            (run_hi, dep_hi), (run_lo, dep_hi),
        ]
    # run = y, depth = x
    return [
        (dep_lo, run_lo), (dep_lo, run_hi),
        (dep_hi, run_hi), (dep_hi, run_lo),
    ]


def _resolve_back_side(
    bbox_m: BBoxM,
    run_axis: str,
    walls: Sequence[BBoxM],
    facing_hint: str,
) -> Tuple[float, float, int]:
    """Bestimme die Tiefen-Achse: (dep_lo, dep_hi, back_sign).

    ``back_sign`` = -1 wenn die Rueckseite an der unteren Tiefen-Kante liegt,
    +1 wenn an der oberen. Die Rueckseite zeigt zur naechsten Wand.
    """
    x0, y0, x1, y1 = bbox_m
    if run_axis == "x":
        dep_lo, dep_hi = y0, y1
        run_mid = 0.5 * (x0 + x1)
        face_low = (run_mid, dep_lo)
        face_high = (run_mid, dep_hi)
    else:
        dep_lo, dep_hi = x0, x1
        run_mid = 0.5 * (y0 + y1)
        face_low = (dep_lo, run_mid)
        face_high = (dep_hi, run_mid)

    back_sign = -1  # default: Rueckseite an der unteren Kante
    if walls:
        d_low = min(_point_to_bbox_distance_m(face_low, w) for w in walls)
        d_high = min(_point_to_bbox_distance_m(face_high, w) for w in walls)
        back_sign = -1 if d_low <= d_high else +1
    elif facing_hint in ("toward_room", "front"):
        back_sign = -1
    return dep_lo, dep_hi, back_sign


def _layout_widths(nominal_widths: Sequence[float], run_len: float) -> List[float]:
    """Items breitengewichtet auf die Run-Laenge legen.

    - Passen sie (Summe <= run_len): Nennbreiten beibehalten (echte Proportionen).
    - Passen sie nicht: alle proportional verkleinern, sodass sie exakt fuellen.
    """
    total = sum(nominal_widths)
    if total <= 0:
        n = max(1, len(nominal_widths))
        return [run_len / n] * n
    if total <= run_len:
        return list(nominal_widths)
    scale = run_len / total
    return [w * scale for w in nominal_widths]


# ---------------------------------------------------------------------------
# Platzierung
# ---------------------------------------------------------------------------

def place_furniture(
    bbox_m: BBoxM,
    classification: FurnitureClassification,
    *,
    walls: Sequence[BBoxM] = (),
    room_id: Optional[str] = None,
) -> PlacedFurniture:
    """Eine Moebel-Bbox + LLM-Klassifikation zu platzierter Geometrie machen."""
    x0, y0, x1, y1 = bbox_m
    bw = abs(x1 - x0)
    bh = abs(y1 - y0)
    run_axis = "x" if bw >= bh else "y"
    run_len = bw if run_axis == "x" else bh

    dep_lo, dep_hi, back_sign = _resolve_back_side(
        bbox_m, run_axis, walls, classification.facing_hint,
    )
    depth_avail = abs(dep_hi - dep_lo)

    contents = sorted(classification.contents, key=lambda c: c.order)
    items_cat = [resolve_item(c.catalog_id) for c in contents]

    notes: List[str] = []
    placed: List[PlacedItem] = []

    if len(contents) <= 1:
        # --- Einzelobjekt: Bbox fuellen, auf Nennmass geklemmt ---
        cat = items_cat[0]
        content = contents[0]
        run_w = _clamp(run_len,
                       cat.nominal_w_m * CLAMP_MIN_FACTOR,
                       cat.nominal_w_m * CLAMP_MAX_FACTOR)
        run_w = min(run_w, run_len)  # nie groesser als die Bbox
        dep_w = _clamp(depth_avail,
                       cat.nominal_d_m * CLAMP_MIN_FACTOR,
                       cat.nominal_d_m * CLAMP_MAX_FACTOR)
        dep_w = min(dep_w, depth_avail)

        # Run-Zentrierung
        run_min = (x0 if run_axis == "x" else y0)
        run_start = run_min + 0.5 * (run_len - run_w)
        # Tiefen-Anker: free -> zentriert, sonst Rueckseite an die Wand
        if cat.placement == PLACEMENT_FREE:
            d_lo = dep_lo + 0.5 * (depth_avail - dep_w)
            d_hi = d_lo + dep_w
            front_sign = +1
        elif back_sign < 0:
            d_lo, d_hi = dep_lo, dep_lo + dep_w
            front_sign = +1
        else:
            d_hi, d_lo = dep_hi, dep_hi - dep_w
            front_sign = -1

        footprint = _rect(run_start, run_start + run_w, d_lo, d_hi, run_axis)
        placed.append(PlacedItem(
            catalog_id=cat.item_id,
            order=0,
            footprint_m=footprint,
            height_m=cat.nominal_h_m,
            facing_deg=_facing_deg(run_axis_to_depth(run_axis), front_sign),
            confidence=content.confidence,
        ))
    else:
        # --- Run: breitengewichtet entlang der laengeren Kante packen ---
        nominal_widths = [cat.nominal_w_m for cat in items_cat]
        widths = _layout_widths(nominal_widths, run_len)
        if sum(nominal_widths) > run_len:
            notes.append("run scaled down to fit bbox")

        run_min = (x0 if run_axis == "x" else y0)
        cursor = run_min
        front_sign = +1 if back_sign < 0 else -1
        for i, (cat, content, w) in enumerate(zip(items_cat, contents, widths)):
            dep_w = min(depth_avail,
                        _clamp(depth_avail,
                               cat.nominal_d_m * CLAMP_MIN_FACTOR,
                               cat.nominal_d_m * CLAMP_MAX_FACTOR))
            if back_sign < 0:
                d_lo, d_hi = dep_lo, dep_lo + dep_w
            else:
                d_hi, d_lo = dep_hi, dep_hi - dep_w
            footprint = _rect(cursor, cursor + w, d_lo, d_hi, run_axis)
            placed.append(PlacedItem(
                catalog_id=cat.item_id,
                order=i,
                footprint_m=footprint,
                height_m=cat.nominal_h_m,
                facing_deg=_facing_deg(run_axis_to_depth(run_axis), front_sign),
                confidence=content.confidence,
            ))
            cursor += w

    return PlacedFurniture(
        bbox_id=classification.bbox_id,
        zone_type=classification.zone_type,
        items=placed,
        room_id=room_id,
        room_type=classification.room_type,
        source=classification.source,
        notes=notes,
    )


def run_axis_to_depth(run_axis: str) -> str:
    """Die Tiefen-Achse ist senkrecht zur Run-Achse."""
    return "y" if run_axis == "x" else "x"
