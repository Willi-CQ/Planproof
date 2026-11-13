"""Hilfen fuer die Defekt-Tests: kleine synthetische DXF-Dateien aus Meterkoordinaten.

Alle Koordinaten werden in Metern angegeben und als Millimeter ($INSUNITS = 4) geschrieben.
Eine optionale Drehung (Grad um den Ursprung) dreht die ganze Zeichnung, auch Bloecke.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import ezdxf

MM = 1000.0
Pt = Tuple[float, float]


class Bau:
    def __init__(
        self, pfad: Path, *, drehung_grad: float = 0.0, insunits: Optional[int] = 4
    ) -> None:
        self.pfad = Path(pfad)
        self.doc = ezdxf.new("R2018", setup=False)
        if insunits is not None:
            self.doc.header["$INSUNITS"] = insunits
        self.msp = self.doc.modelspace()
        self.rot = math.radians(drehung_grad)
        self.drehung_grad = drehung_grad

    # -- Geometrie ---------------------------------------------------------
    def p(self, x: float, y: float) -> Pt:
        c, s = math.cos(self.rot), math.sin(self.rot)
        return ((x * c - y * s) * MM, (x * s + y * c) * MM)

    def linie(self, a: Pt, b: Pt, layer: str) -> None:
        self.msp.add_line(self.p(*a), self.p(*b), dxfattribs={"layer": layer})

    def polylinie(self, pts: Sequence[Pt], layer: str, *, geschlossen: bool = True) -> None:
        self.msp.add_lwpolyline(
            [self.p(*q) for q in pts], close=geschlossen, dxfattribs={"layer": layer}
        )

    def text(self, s: str, x: float, y: float, layer: str, hoehe: float = 0.25) -> None:
        t = self.msp.add_text(s, height=hoehe * MM, dxfattribs={"layer": layer})
        t.dxf.insert = (*self.p(x, y), 0.0)

    # -- Waende ------------------------------------------------------------
    def rechteck_doppel(
        self, x0: float, y0: float, x1: float, y1: float, t: float, layer: str
    ) -> None:
        """Geschlossene Doppelkontur um ein Rechteck: Achsen laufen bei x0/y0/x1/y1.

        Aussenkante voll, Innenkante an den Ecken verkuerzt (wie in jeder Doppellinien-Zeichnung);
        die Enden der Achsenpaare liegen deshalb um t/2 vor der Ecke.
        """
        h = t / 2
        for a, b in self._rechteck_kanten(x0 - h, y0 - h, x1 + h, y1 + h):
            self.linie(a, b, layer)
        for a, b in self._rechteck_kanten(x0 + h, y0 + h, x1 - h, y1 - h):
            self.linie(a, b, layer)

    @staticmethod
    def _rechteck_kanten(x0: float, y0: float, x1: float, y1: float) -> List[Tuple[Pt, Pt]]:
        return [
            ((x0, y0), (x1, y0)),
            ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)),
            ((x0, y1), (x0, y0)),
        ]

    def wand_doppel(self, a: Pt, b: Pt, t: float, layer: str) -> None:
        """Eine Wand als zwei Linien ueber die ganze Laenge a-b (Achse a-b)."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy)
        nx, ny = -dy / n * t / 2, dx / n * t / 2
        self.linie((a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny), layer)
        self.linie((a[0] - nx, a[1] - ny), (b[0] - nx, b[1] - ny), layer)

    def wand_rechteck(self, a: Pt, b: Pt, t: float, layer: str) -> None:
        """Eine Wand als geschlossene Rechteck-Polylinie (Kontur) um die Achse a-b."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy)
        nx, ny = -dy / n * t / 2, dx / n * t / 2
        self.polylinie(
            [
                (a[0] + nx, a[1] + ny),
                (b[0] + nx, b[1] + ny),
                (b[0] - nx, b[1] - ny),
                (a[0] - nx, a[1] - ny),
            ],
            layer,
        )

    # -- Oeffnungen --------------------------------------------------------
    def tuer_block(self, name: str, breite: float) -> None:
        """Tuerblock wie in CAD ueblich: Ursprung am Drehpunkt, Blatt entlang +x, Bogen im
        Viertelkreis."""
        if name in self.doc.blocks:
            return
        blk = self.doc.blocks.new(name=name)
        w = breite * MM
        blk.add_line((0, 0), (w, 0), dxfattribs={"layer": "0"})
        blk.add_arc((0, 0), w, 0, 90, dxfattribs={"layer": "0"})

    def fenster_block(self, name: str, breite: float, tiefe: float = 0.24) -> None:
        if name in self.doc.blocks:
            return
        blk = self.doc.blocks.new(name=name)
        w, d = breite * MM, tiefe * MM
        blk.add_lwpolyline(
            [(0, -d / 2), (w, -d / 2), (w, d / 2), (0, d / 2)],
            close=True,
            dxfattribs={"layer": "0"},
        )
        blk.add_line((0, 0), (w, 0), dxfattribs={"layer": "0"})

    def einfuegen(
        self, name: str, x: float, y: float, layer: str, rotation_grad: float = 0.0, **kw
    ) -> None:
        attribs = {"layer": layer, "rotation": rotation_grad + self.drehung_grad}
        attribs.update(kw)
        self.msp.add_blockref(name, self.p(x, y), dxfattribs=attribs)

    def speichern(self) -> Path:
        self.pfad.parent.mkdir(parents=True, exist_ok=True)
        self.doc.saveas(str(self.pfad))
        return self.pfad


def haus(
    pfad: Path,
    *,
    w: float = 8.0,
    h: float = 6.0,
    t: float = 0.30,
    layer: str = "A-WALL",
    innen_x: Optional[float] = None,
    ti: float = 0.115,
    drehung_grad: float = 0.0,
) -> Bau:
    """Rechteckhaus mit Doppelkontur und optional einer durchgehenden Innenwand bei x = innen_x."""
    bau = Bau(pfad, drehung_grad=drehung_grad)
    bau.rechteck_doppel(0, 0, w, h, t, layer)
    if innen_x is not None:
        # Innenwand zwischen den Innenkanten der Aussenwaende (Doppellinie, T-Stoesse)
        bau.linie((innen_x - ti / 2, t / 2), (innen_x - ti / 2, h - t / 2), layer)
        bau.linie((innen_x + ti / 2, t / 2), (innen_x + ti / 2, h - t / 2), layer)
    return bau


def mittelpunkt(punkte: Iterable[Pt]) -> Pt:
    pts = list(punkte)
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
