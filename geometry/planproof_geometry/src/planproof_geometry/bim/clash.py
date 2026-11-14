"""Geometrische Kollisionspruefung (Clash Detection, K5) — AABB, nativ.

Prueft achsenausgerichtete Bounding-Boxen (AABB) von Modell-Komponenten paarweise
auf Durchdringung. Broadphase via Sweep-and-Prune auf der X-Achse (O(n log n + k)),
Narrowphase = 3-Achsen-Overlap mit Toleranz. Rein typisiert, ohne DB-/Netz-
Abhaengigkeit — Mirror der uebrigen Engines. Downstream: der API-Clash-Endpoint
laedt ProjectModelComponent (center_* + length/width/height) und ruft das hier.

Bewusst AABB (nicht mesh-genau) und innerhalb EINES Modells: der Rechenkern
schliesst den 'keine Clash-Engine'-Gap; Multi-Modell-Foederation (ARCH/TWP/TGA)
+ mesh-genaue Kollision sind die openBIM-Folge-Welle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Toleranz-Default: unter diesem Overlap (Meter) gilt es als Beruehrung, nicht als
# Durchdringung (angrenzende Bauteile teilen sich Kanten).
DEFAULT_TOLERANCE_M = 0.005


@dataclass(frozen=True)
class ClashBox:
    """Ein Bauteil als AABB (Zentrum + volle Ausdehnung je Achse, in Metern)."""

    id: str
    class_code: Optional[str]
    cx: float
    cy: float
    cz: float
    dx: float
    dy: float
    dz: float

    def bounds(self) -> Tuple[float, float, float, float, float, float]:
        hx, hy, hz = self.dx / 2.0, self.dy / 2.0, self.dz / 2.0
        return (self.cx - hx, self.cx + hx, self.cy - hy, self.cy + hy, self.cz - hz, self.cz + hz)


@dataclass(frozen=True)
class Clash:
    """Eine erkannte Durchdringung zweier Bauteile."""

    a_id: str
    b_id: str
    a_class: Optional[str]
    b_class: Optional[str]
    penetration_m: float  # kleinste Overlap-Ausdehnung (Eindringtiefe)
    overlap_volume_m3: float
    center: Tuple[float, float, float]  # Mittelpunkt des Overlap-Quaders

    def to_dict(self) -> dict:
        return {
            "a_id": self.a_id,
            "b_id": self.b_id,
            "a_class": self.a_class,
            "b_class": self.b_class,
            "penetration_m": round(self.penetration_m, 4),
            "overlap_volume_m3": round(self.overlap_volume_m3, 6),
            "center": [round(c, 4) for c in self.center],
        }


def _valid(box: ClashBox) -> bool:
    """Nur Boxen mit vollstaendiger, endlicher Geometrie sind pruefbar."""
    vals = (box.cx, box.cy, box.cz, box.dx, box.dy, box.dz)
    return all(v is not None for v in vals) and all(v == v for v in vals)  # NaN-Check


# Audit M10: hartes Obergrenze der Paar-Vergleiche (Narrowphase). Der Sweep-Abbruch
# greift nur, wenn Boxen entlang X getrennt sind; teilen sich ALLE Boxen einen
# X-Bereich (crafted/degeneriertes Modell), laeuft die innere Schleife voll ->
# O(n^2). ``max_clashes`` deckelt nur das ERGEBNIS, nicht die Vergleiche. Dieses
# Budget deckelt die CPU-Arbeit fail-safe (~50 Mio Vergleiche ~ wenige Sekunden).
DEFAULT_MAX_COMPARISONS = 50_000_000


def detect_clashes(
    boxes: Sequence[ClashBox],
    *,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    ignore_same_class: bool = False,
    max_clashes: int = 2000,
    max_comparisons: int = DEFAULT_MAX_COMPARISONS,
) -> List[Clash]:
    """Findet AABB-Durchdringungen (Overlap in ALLEN drei Achsen > Toleranz).

    Args:
        boxes: Bauteil-AABBs.
        tolerance_m: Mindest-Overlap je Achse, ab dem eine Durchdringung zaehlt
            (darunter = Beruehrung angrenzender Bauteile).
        ignore_same_class: True blendet Paare gleicher Klasse aus (z.B. zwei
            aneinanderstossende Waende), um Anschluss-Fehlalarme zu reduzieren.
        max_clashes: Obergrenze der zurueckgegebenen Clashes (Explosionsschutz;
            wird die Grenze erreicht, bricht die Suche kontrolliert ab).
        max_comparisons: Obergrenze der Paar-Vergleiche (DoS-Schutz gegen O(n^2)
            bei degenerierter X-Verteilung; darueber bricht die Suche kontrolliert ab).

    Returns:
        Liste von Clashes, nach Eindringtiefe absteigend sortiert.
    """
    usable = [b for b in boxes if _valid(b) and (b.dx > 0 or b.dy > 0 or b.dz > 0)]
    # Sweep-and-Prune: nach X-Min sortieren.
    entries = []
    for b in usable:
        x0, x1, y0, y1, z0, z1 = b.bounds()
        entries.append((x0, x1, y0, y1, z0, z1, b))
    entries.sort(key=lambda e: e[0])

    clashes: List[Clash] = []
    n = len(entries)
    hit_cap = False
    comparisons = 0
    for i in range(n):
        if hit_cap:
            break
        ax0, ax1, ay0, ay1, az0, az1, a = entries[i]
        for j in range(i + 1, n):
            # Audit M10: Vergleichs-Budget erschoepft -> kontrolliert abbrechen.
            comparisons += 1
            if comparisons > max_comparisons:
                hit_cap = True
                break
            bx0, bx1, by0, by1, bz0, bz1, b = entries[j]
            # Sweep-Abbruch: sortiert nach X-Min -> sobald B links von A endet.
            if bx0 > ax1 - tolerance_m:
                break
            if ignore_same_class and a.class_code is not None and a.class_code == b.class_code:
                continue
            ox = min(ax1, bx1) - max(ax0, bx0)
            oy = min(ay1, by1) - max(ay0, by0)
            oz = min(az1, bz1) - max(az0, bz0)
            if ox > tolerance_m and oy > tolerance_m and oz > tolerance_m:
                cx = (max(ax0, bx0) + min(ax1, bx1)) / 2.0
                cy = (max(ay0, by0) + min(ay1, by1)) / 2.0
                cz = (max(az0, bz0) + min(az1, bz1)) / 2.0
                clashes.append(
                    Clash(
                        a_id=a.id,
                        b_id=b.id,
                        a_class=a.class_code,
                        b_class=b.class_code,
                        penetration_m=min(ox, oy, oz),
                        overlap_volume_m3=ox * oy * oz,
                        center=(cx, cy, cz),
                    )
                )
                if len(clashes) >= max_clashes:
                    hit_cap = True
                    break

    clashes.sort(key=lambda c: c.penetration_m, reverse=True)
    return clashes
