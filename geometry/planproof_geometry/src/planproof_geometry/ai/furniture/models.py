"""Prozeduraler Standardmodell-Generator + Modelldatenbank.

Jede Katalog-ID bekommt ein erkennbares, zusammengesetztes Standardmodell
(nicht nur eine Box): WC = Becken + Spuelkasten, Sofa = Sitz + Lehne + Armlehnen,
Tisch = Platte + 4 Beine, Herd = Korpus + Kochfelder, usw.

**Lizenz:** 100% prozedural generiert -> kein Drittmaterial, kein Lizenzrisiko.
Wer lieber echte CC0-Modelle will (z.B. Kenney Furniture Kit, CC0/Public Domain),
setzt in catalog.py ``glb_path`` pro Item — mesh.py bevorzugt dann das GLB und
faellt sonst auf dieses Modul zurueck.

Geometrie wird in einem **normierten Frame** beschrieben und ueber :class:`_Frame`
in den platzierten Footprint abgebildet — so muss kein Builder selbst mit
Welt-Achsen, Orientierung oder Spiegelung rechnen:

  u in [0,1]  = Breite (entlang der laengeren Bbox-/Run-Achse)
  v in [0,1]  = Tiefe; v=0 ist die RUECKSEITE (zur Wand), v=1 die FRONT
  w in [0,1]  = Hoehe; w=0 Boden, w=1 Oberkante (Katalog-Nennhoehe)

Der Frame uebersetzt (u, v, w) achsen-treu in Welt (x, y=hoehe, z) unter
Beruecksichtigung von ``facing_deg`` (welche Welt-Achse die Tiefe ist, und wo
vorne liegt). Damit landet z.B. die Sofa-Lehne immer an der Wand.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Eine normierte Box: (u0, u1, v0, v1, w0, w1). Werte leicht > 1 / < 0 sind
# erlaubt (z.B. Griffe, die vor der Front stehen).
NBox = Tuple[float, float, float, float, float, float]


@dataclass
class _Frame:
    """Bildet normierte (u, v, w) achsen-treu in den Welt-Footprint ab."""

    xmin: float
    xmax: float
    zmin: float
    zmax: float
    h: float
    width_axis: str    # 'x' oder 'z' — welche Welt-Achse die Breite (u) ist
    front_sign: int    # +1: +Tiefenachse ist vorne; -1: gespiegelt

    def map(self, u: float, v: float, w: float) -> Tuple[float, float, float]:
        vv = v if self.front_sign > 0 else (1.0 - v)
        if self.width_axis == "x":
            x = self.xmin + u * (self.xmax - self.xmin)
            z = self.zmin + vv * (self.zmax - self.zmin)
        else:
            z = self.zmin + u * (self.zmax - self.zmin)
            x = self.xmin + vv * (self.xmax - self.xmin)
        return (x, w * self.h, z)


def _frame_from(xmin, xmax, zmin, zmax, h, facing_deg) -> _Frame:
    fd = int(round(facing_deg)) % 360
    if fd in (90, 270):           # Tiefe entlang Z
        width_axis = "x"
        front_sign = 1 if fd == 90 else -1
    else:                          # fd 0/180 -> Tiefe entlang X
        width_axis = "z"
        front_sign = 1 if fd == 0 else -1
    return _Frame(xmin, xmax, zmin, zmax, h, width_axis, front_sign)


def _box_norm(frame: _Frame, b: NBox):
    """Eine normierte Box in eine welt-achsenparallele trimesh.Box wandeln."""
    import trimesh

    u0, u1, v0, v1, w0, w1 = b
    corners = [
        frame.map(u, v, w)
        for u in (u0, u1) for v in (v0, v1) for w in (w0, w1)
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    zs = [c[2] for c in corners]
    ex = max(max(xs) - min(xs), 1e-3)
    ey = max(max(ys) - min(ys), 1e-3)
    ez = max(max(zs) - min(zs), 1e-3)
    box = trimesh.creation.box(extents=(ex, ey, ez))
    box.apply_translation((
        0.5 * (max(xs) + min(xs)),
        0.5 * (max(ys) + min(ys)),
        0.5 * (max(zs) + min(zs)),
    ))
    return box


# ============================================================================
# Archetyp-Definitionen (Listen normierter Boxen). v=0 = Rueckseite/Wand.
# ============================================================================

_GENERIC: List[NBox] = [(0, 1, 0, 1, 0, 1)]

_BED: List[NBox] = [
    (0.00, 1.00, 0.00, 1.00, 0.00, 0.32),   # Bettgestell
    (0.00, 1.00, 0.00, 0.06, 0.00, 0.90),   # Kopfteil (an der Wand)
    (0.04, 0.96, 0.08, 0.98, 0.32, 0.58),   # Matratze
    (0.08, 0.48, 0.08, 0.30, 0.58, 0.70),   # Kissen links
    (0.52, 0.92, 0.08, 0.30, 0.58, 0.70),   # Kissen rechts
]

_SOFA: List[NBox] = [
    (0.00, 1.00, 0.00, 0.18, 0.00, 0.95),   # Rueckenlehne (Wand)
    (0.00, 1.00, 0.18, 1.00, 0.00, 0.42),   # Sitzbasis
    (0.00, 0.12, 0.18, 1.00, 0.00, 0.62),   # Armlehne links
    (0.88, 1.00, 0.18, 1.00, 0.00, 0.62),   # Armlehne rechts
    (0.13, 0.87, 0.28, 0.96, 0.42, 0.55),   # Sitzkissen
]

_WC: List[NBox] = [
    (0.20, 0.80, 0.00, 0.28, 0.00, 0.85),   # Spuelkasten (Wand)
    (0.26, 0.74, 0.28, 0.92, 0.00, 0.46),   # Becken
    (0.23, 0.77, 0.30, 0.95, 0.42, 0.50),   # Sitz/Deckel
]

_SMALL_FIXTURE: List[NBox] = [   # Bidet / Urinal
    (0.28, 0.72, 0.10, 0.85, 0.00, 0.60),
]

_WASHBASIN: List[NBox] = [
    (0.38, 0.62, 0.20, 0.80, 0.00, 0.72),   # Saeule
    (0.06, 0.94, 0.05, 0.95, 0.72, 0.86),   # Waschbecken-Platte
]

_BATHTUB: List[NBox] = [
    (0.00, 1.00, 0.85, 1.00, 0.00, 0.55),   # Rand vorne
    (0.00, 1.00, 0.00, 0.15, 0.00, 0.55),   # Rand hinten
    (0.00, 0.08, 0.00, 1.00, 0.00, 0.55),   # Rand links
    (0.92, 1.00, 0.00, 1.00, 0.00, 0.55),   # Rand rechts
    (0.06, 0.94, 0.12, 0.88, 0.00, 0.14),   # Wannenboden
]

_SHOWER: List[NBox] = [
    (0.00, 1.00, 0.00, 1.00, 0.00, 0.06),   # Duschtasse
    (0.00, 1.00, 0.00, 0.05, 0.00, 1.00),   # Rueckwand
    (0.00, 0.05, 0.00, 1.00, 0.00, 1.00),   # Seitenwand
]

_STOVE: List[NBox] = [
    (0.00, 1.00, 0.00, 1.00, 0.00, 0.88),   # Korpus
    (0.10, 0.45, 0.10, 0.45, 0.88, 0.93),   # Kochfeld 1
    (0.55, 0.90, 0.10, 0.45, 0.88, 0.93),   # Kochfeld 2
    (0.10, 0.45, 0.55, 0.90, 0.88, 0.93),   # Kochfeld 3
    (0.55, 0.90, 0.55, 0.90, 0.88, 0.93),   # Kochfeld 4
]

_FRONT_APPLIANCE: List[NBox] = [   # Backofen / Geschirrspueler / Waschmaschine
    (0.00, 1.00, 0.00, 0.92, 0.00, 1.00),   # Korpus
    (0.06, 0.94, 0.92, 1.00, 0.08, 0.92),   # Tuer (Front)
    (0.30, 0.70, 1.00, 1.04, 0.40, 0.78),   # Bullauge/Griff
]

_FRIDGE: List[NBox] = [
    (0.00, 1.00, 0.00, 1.00, 0.00, 1.00),   # Korpus
    (0.02, 0.98, 0.96, 1.02, 0.02, 0.55),   # Tuer unten
    (0.02, 0.98, 0.96, 1.02, 0.58, 0.98),   # Tuer oben
    (0.86, 0.92, 1.02, 1.05, 0.30, 0.70),   # Griff
]

_CABINET: List[NBox] = [
    (0.00, 1.00, 0.00, 1.00, 0.00, 1.00),   # Korpus
    (0.04, 0.96, 0.96, 1.02, 0.04, 0.96),   # Tuer
    (0.85, 0.92, 1.02, 1.05, 0.40, 0.60),   # Griff
]

_WARDROBE: List[NBox] = [
    (0.00, 1.00, 0.00, 1.00, 0.00, 1.00),   # Korpus
    (0.02, 0.49, 0.96, 1.02, 0.04, 0.96),   # Tuer links
    (0.51, 0.98, 0.96, 1.02, 0.04, 0.96),   # Tuer rechts
    (0.45, 0.49, 1.02, 1.05, 0.40, 0.70),   # Griff links
    (0.51, 0.55, 1.02, 1.05, 0.40, 0.70),   # Griff rechts
]

_NIGHTSTAND: List[NBox] = [
    (0.00, 1.00, 0.00, 1.00, 0.00, 1.00),   # Korpus
    (0.08, 0.92, 0.96, 1.02, 0.55, 0.90),   # Schublade
    (0.42, 0.58, 1.02, 1.04, 0.70, 0.78),   # Griff
]

_TABLE: List[NBox] = [
    (0.00, 1.00, 0.00, 1.00, 0.85, 1.00),   # Tischplatte
    (0.00, 0.09, 0.00, 0.09, 0.00, 0.85),   # Bein 1
    (0.91, 1.00, 0.00, 0.09, 0.00, 0.85),   # Bein 2
    (0.00, 0.09, 0.91, 1.00, 0.00, 0.85),   # Bein 3
    (0.91, 1.00, 0.91, 1.00, 0.00, 0.85),   # Bein 4
]

_CHAIR: List[NBox] = [
    (0.10, 0.90, 0.10, 0.90, 0.45, 0.55),   # Sitzflaeche
    (0.10, 0.90, 0.00, 0.12, 0.55, 1.00),   # Rueckenlehne
    (0.10, 0.20, 0.10, 0.20, 0.00, 0.45),   # Bein 1
    (0.80, 0.90, 0.10, 0.20, 0.00, 0.45),   # Bein 2
    (0.10, 0.20, 0.80, 0.90, 0.00, 0.45),   # Bein 3
    (0.80, 0.90, 0.80, 0.90, 0.00, 0.45),   # Bein 4
]

_OFFICE_CHAIR: List[NBox] = [
    (0.12, 0.88, 0.12, 0.88, 0.45, 0.55),   # Sitz
    (0.18, 0.82, 0.00, 0.12, 0.55, 1.00),   # Lehne
    (0.42, 0.58, 0.42, 0.58, 0.05, 0.45),   # Saeule
    (0.20, 0.80, 0.20, 0.80, 0.00, 0.06),   # Fusskreuz
]

_LOWBOARD: List[NBox] = [
    (0.00, 1.00, 0.00, 1.00, 0.00, 1.00),   # Korpus
    (0.02, 0.32, 0.96, 1.01, 0.10, 0.90),   # Tuer 1
    (0.35, 0.65, 0.96, 1.01, 0.10, 0.90),   # Tuer 2
    (0.68, 0.98, 0.96, 1.01, 0.10, 0.90),   # Tuer 3
]

_SHELF: List[NBox] = [
    (0.00, 1.00, 0.00, 0.10, 0.00, 1.00),   # Rueckwand
    (0.00, 0.05, 0.00, 1.00, 0.00, 1.00),   # Seite links
    (0.95, 1.00, 0.00, 1.00, 0.00, 1.00),   # Seite rechts
    (0.05, 0.95, 0.10, 1.00, 0.03, 0.06),   # Boden
    (0.05, 0.95, 0.10, 1.00, 0.36, 0.39),   # Brett
    (0.05, 0.95, 0.10, 1.00, 0.68, 0.71),   # Brett
    (0.05, 0.95, 0.10, 1.00, 0.96, 0.99),   # Deckel
]


# ID -> Archetyp. Jede Katalog-ID bekommt ein Standardmodell.
_ID_TO_BOXES: Dict[str, List[NBox]] = {
    "wc": _WC, "wc_wall_hung": _WC,
    "bidet": _SMALL_FIXTURE, "urinal": _SMALL_FIXTURE,
    "washbasin": _WASHBASIN, "double_washbasin": _WASHBASIN,
    "shower": _SHOWER, "bathtub": _BATHTUB,
    "kitchen_sink": _WASHBASIN,
    "stove": _STOVE,
    "oven": _FRONT_APPLIANCE, "dishwasher": _FRONT_APPLIANCE,
    "washing_machine": _FRONT_APPLIANCE,
    "refrigerator": _FRIDGE,
    "base_cabinet": _CABINET, "tall_cabinet": _CABINET,
    "bed_single": _BED, "bed_double": _BED,
    "nightstand": _NIGHTSTAND,
    "wardrobe": _WARDROBE,
    "sofa": _SOFA, "armchair": _SOFA,
    "coffee_table": _TABLE, "dining_table": _TABLE, "desk": _TABLE,
    "chair": _CHAIR, "office_chair": _OFFICE_CHAIR,
    "tv_unit": _LOWBOARD, "shelf": _SHELF,
    "furniture_generic": _GENERIC,
}


def has_standard_model(catalog_id: str) -> bool:
    return catalog_id in _ID_TO_BOXES


def build_standard_model(
    catalog_id: str,
    *,
    xmin: float,
    xmax: float,
    zmin: float,
    zmax: float,
    h: float,
    facing_deg: float,
):
    """Zusammengesetztes Standardmodell im Welt-Footprint bauen.

    Gibt eine trimesh.Trimesh zurueck (mehrere Boxen verschmolzen) oder None,
    wenn trimesh fehlt oder die ID kein Archetyp hat (Aufrufer -> Box-Fallback).
    """
    boxes = _ID_TO_BOXES.get(catalog_id)
    if boxes is None:
        return None
    try:
        import trimesh
    except ImportError:
        return None

    frame = _frame_from(xmin, xmax, zmin, zmax, h, facing_deg)
    parts = [_box_norm(frame, b) for b in boxes]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return trimesh.util.concatenate(parts)
