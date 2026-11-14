"""Synthetischer DXF-Generator -- Testfixture **und** Demo-Input.

Erzeugt einen vollstaendigen, realistischen Grundriss als ``.dxf``:

  * Aussenwaende als **Doppellinien** auf ``A_Wand_a`` (Dicke 0.30 m)
  * Innenwaende als Doppellinien auf ``A_Wand_i`` (Dicke 0.115 m), mit
    T-Stoessen (uebt das Junction-Splitting)
  * Tueren/Fenster als benannte **Bloecke** (INSERT) mit Rotation
    (= Schwingrichtung) auf ``A_Tuer`` / ``A_Fenster``
  * Raumlabels als TEXT auf ``A_Raum`` ("Wohnzimmer", "Küche", "Bad")
  * Einheit Millimeter ($INSUNITS=4)

So laesst sich die komplette DWG->BIM-Pipeline ohne externe Datei und ohne
ODA/LibreDWG end-to-end testen.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

Point2 = Tuple[float, float]

_MM = 1000.0  # Meter -> Millimeter


def _add_double_wall(msp, a: Point2, b: Point2, thickness_m: float, layer: str) -> None:
    """Zeichnet eine Wand als zwei parallele Linien (Achse +/- t/2), in mm."""
    ax, ay = a[0] * _MM, a[1] * _MM
    bx, by = b[0] * _MM, b[1] * _MM
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return
    nx, ny = -dy / L, dx / L           # Normale
    h = thickness_m * _MM / 2.0
    for s in (+1.0, -1.0):
        msp.add_line(
            (ax + nx * h * s, ay + ny * h * s),
            (bx + nx * h * s, by + ny * h * s),
            dxfattribs={"layer": layer},
        )


def _define_door_block(doc, name: str, width_m: float) -> None:
    if name in doc.blocks:
        return
    blk = doc.blocks.new(name=name)
    w = width_m * _MM
    # Tuerblatt (Linie) + Schwingbogen -> Bbox ~ w x w
    blk.add_line((0, 0), (w, 0), dxfattribs={"layer": "0"})
    blk.add_arc(center=(0, 0), radius=w, start_angle=0, end_angle=90,
                dxfattribs={"layer": "0"})


def _define_window_block(doc, name: str, width_m: float) -> None:
    if name in doc.blocks:
        return
    blk = doc.blocks.new(name=name)
    w = width_m * _MM
    d = 0.20 * _MM
    # Rahmen-Rechteck w x d
    blk.add_lwpolyline(
        [(0, 0), (w, 0), (w, d), (0, d)], close=True, dxfattribs={"layer": "0"},
    )
    blk.add_line((0, d / 2), (w, d / 2), dxfattribs={"layer": "0"})


def build_demo_plan(out_path: str | Path) -> Path:
    """Schreibt einen 3-Raum-Demo-Grundriss als DXF und liefert den Pfad.

    Grundriss (Achsen, Meter):
        Aussenrechteck 10 x 8
        Vertikale Innenwand x=4 (y 0..8)  -> teilt in links / rechts
        Horizontale Innenwand y=4 (x 4..10) -> teilt rechts in unten/oben
    => 3 Raeume: Wohnzimmer (links), Küche (rechts-unten), Bad (rechts-oben)
    """
    import ezdxf

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.MM   # $INSUNITS = 4
    msp = doc.modelspace()

    for lyr, color in (
        ("A_Wand_a", 1), ("A_Wand_i", 2), ("A_Tuer", 3),
        ("A_Fenster", 4), ("A_Raum", 7),
    ):
        if lyr not in doc.layers:
            doc.layers.add(lyr, color=color)

    # --- Aussenwaende (Achsen-Rechteck 10x8), Dicke 0.30 ---
    ext = 0.30
    corners = [(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)]
    for i in range(4):
        _add_double_wall(msp, corners[i], corners[(i + 1) % 4], ext, "A_Wand_a")

    # --- Innenwaende (Dicke 0.115), T-Stoesse ---
    inn = 0.115
    _add_double_wall(msp, (4.0, 0.0), (4.0, 8.0), inn, "A_Wand_i")   # vertikal
    _add_double_wall(msp, (4.0, 4.0), (10.0, 4.0), inn, "A_Wand_i")  # horizontal

    # --- Tuer-/Fenster-Bloecke definieren ---
    _define_door_block(doc, "Tuer_1010", 1.01)
    _define_door_block(doc, "Tuer_0885", 0.885)
    _define_window_block(doc, "Fenster_1260", 1.26)
    _define_window_block(doc, "Fenster_1010", 1.01)

    # --- Tueren (Position auf Wand-Achse, Rotation = Schwingrichtung) ---
    # Tuer in vertikaler Innenwand (verbindet links<->rechts), entlang y
    msp.add_blockref("Tuer_1010", (4.0 * _MM, 2.0 * _MM),
                     dxfattribs={"layer": "A_Tuer", "rotation": 90})
    # Tuer in horizontaler Innenwand (rechts unten<->oben), entlang x
    msp.add_blockref("Tuer_0885", (7.0 * _MM, 4.0 * _MM),
                     dxfattribs={"layer": "A_Tuer", "rotation": 0})
    # Aussentuer in unterer Aussenwand (Wohnzimmer)
    msp.add_blockref("Tuer_1010", (2.0 * _MM, 0.0),
                     dxfattribs={"layer": "A_Tuer", "rotation": 0})

    # --- Fenster (in Aussenwaenden) ---
    msp.add_blockref("Fenster_1260", (1.5 * _MM, 8.0 * _MM),
                     dxfattribs={"layer": "A_Fenster", "rotation": 0})    # oben, Wohnzimmer
    msp.add_blockref("Fenster_1010", (10.0 * _MM, 2.0 * _MM),
                     dxfattribs={"layer": "A_Fenster", "rotation": 90})   # rechts, Küche
    msp.add_blockref("Fenster_1260", (10.0 * _MM, 6.0 * _MM),
                     dxfattribs={"layer": "A_Fenster", "rotation": 90})   # rechts, Bad

    # --- Raumlabels ---
    for txt, (x, y) in (
        ("Wohnzimmer", (2.0, 4.0)),
        ("Küche", (7.0, 2.0)),
        ("Bad", (7.0, 6.0)),
    ):
        t = msp.add_text(txt, height=250, dxfattribs={"layer": "A_Raum"})
        t.dxf.insert = (x * _MM, y * _MM, 0)

    doc.saveas(str(out))
    return out


__all__ = ["build_demo_plan"]
