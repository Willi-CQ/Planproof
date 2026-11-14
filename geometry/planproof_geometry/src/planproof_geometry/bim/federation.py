"""Modell-Foederation + Cross-Model-Clash (K5-Rest, openBIM-Koordination).

Fuehrt mehrere Fachmodelle (ARCH/TWP/TGA ...) in einem gemeinsamen Koordinatenraum
zusammen und prueft NUR modelluebergreifende Durchdringungen — die eigentliche
openBIM-Koordinationsfrage ("kollidiert ein TGA-Strang mit einem Traeger?"). Die
Intra-Modell-Clashes deckt bereits ``clash.detect_clashes`` ab.

Setzt gemeinsame Georeferenzierung voraus (Fachmodelle sind auf denselben Ursprung
koordiniert) — Standard-Annahme der Modell-Foederation. Broadphase = Sweep-and-Prune
auf X (O(n log n + k)), Narrowphase = 3-Achsen-AABB-Overlap mit Toleranz. Mesh-genaue
Kollision bleibt der Integrations-/Folgeschritt (Solibri/Navisworks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

from .clash import DEFAULT_TOLERANCE_M, ClashBox


@dataclass(frozen=True)
class FederatedClash:
    """Eine modelluebergreifende Durchdringung zweier Bauteile."""

    a_model: str
    b_model: str
    a_id: str
    b_id: str
    a_class: Optional[str]
    b_class: Optional[str]
    penetration_m: float
    overlap_volume_m3: float
    center: Tuple[float, float, float]

    def to_dict(self) -> dict:
        return {
            "a_model": self.a_model,
            "b_model": self.b_model,
            "a_id": self.a_id,
            "b_id": self.b_id,
            "a_class": self.a_class,
            "b_class": self.b_class,
            "penetration_m": round(self.penetration_m, 4),
            "overlap_volume_m3": round(self.overlap_volume_m3, 6),
            "center": [round(c, 4) for c in self.center],
        }


@dataclass(frozen=True)
class FederationResult:
    """Ergebnis der Foederations-Pruefung: Clashes + Statistik je Modell(paar)."""

    clashes: List[FederatedClash]
    model_box_counts: dict
    pairs: List[dict]  # [{a_model, b_model, count}] modelluebergreifende Paar-Zaehlung
    clash_count: int
    truncated: bool

    def to_dict(self) -> dict:
        return {
            "clash_count": self.clash_count,
            "truncated": self.truncated,
            "model_box_counts": self.model_box_counts,
            "pairs": self.pairs,
            "clashes": [c.to_dict() for c in self.clashes],
        }


def _valid(box: ClashBox) -> bool:
    vals = (box.cx, box.cy, box.cz, box.dx, box.dy, box.dz)
    return all(v is not None for v in vals) and all(v == v for v in vals)  # NaN-Check


def detect_federation_clashes(
    models: Mapping[str, Sequence[ClashBox]],
    *,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    max_clashes: int = 2000,
) -> FederationResult:
    """Findet modelluebergreifende AABB-Durchdringungen zwischen Fachmodellen.

    Args:
        models: {Modellname: Bauteil-AABBs}. Es werden nur Paare aus VERSCHIEDENEN
            Modellen geprueft (Intra-Modell -> ``clash.detect_clashes``).
        tolerance_m: Mindest-Overlap je Achse (darunter = Beruehrung).
        max_clashes: Obergrenze der zurueckgegebenen Clashes (Explosionsschutz).

    Returns:
        FederationResult mit Clashes (nach Eindringtiefe absteigend), Box-Zahl je
        Modell und modelluebergreifender Paar-Zaehlung.
    """
    entries: list[tuple] = []
    model_box_counts: dict[str, int] = {}
    for model_name, boxes in models.items():
        count = 0
        for b in boxes:
            if not _valid(b) or not (b.dx > 0 or b.dy > 0 or b.dz > 0):
                continue
            x0, x1, y0, y1, z0, z1 = b.bounds()
            entries.append((x0, x1, y0, y1, z0, z1, model_name, b))
            count += 1
        model_box_counts[model_name] = count

    entries.sort(key=lambda e: e[0])

    clashes: List[FederatedClash] = []
    pair_counts: dict[tuple, int] = {}
    truncated = False
    n = len(entries)
    for i in range(n):
        if truncated:
            break
        ax0, ax1, ay0, ay1, az0, az1, am, a = entries[i]
        for j in range(i + 1, n):
            bx0, bx1, by0, by1, bz0, bz1, bm, b = entries[j]
            if bx0 > ax1 - tolerance_m:
                break  # Sweep-Abbruch (nach X-Min sortiert)
            if am == bm:
                continue  # nur modelluebergreifend
            ox = min(ax1, bx1) - max(ax0, bx0)
            oy = min(ay1, by1) - max(ay0, by0)
            oz = min(az1, bz1) - max(az0, bz0)
            if ox > tolerance_m and oy > tolerance_m and oz > tolerance_m:
                cx = (max(ax0, bx0) + min(ax1, bx1)) / 2.0
                cy = (max(ay0, by0) + min(ay1, by1)) / 2.0
                cz = (max(az0, bz0) + min(az1, bz1)) / 2.0
                clashes.append(
                    FederatedClash(
                        a_model=am,
                        b_model=bm,
                        a_id=a.id,
                        b_id=b.id,
                        a_class=a.class_code,
                        b_class=b.class_code,
                        penetration_m=min(ox, oy, oz),
                        overlap_volume_m3=ox * oy * oz,
                        center=(cx, cy, cz),
                    )
                )
                key = tuple(sorted((am, bm)))
                pair_counts[key] = pair_counts.get(key, 0) + 1
                if len(clashes) >= max_clashes:
                    truncated = True
                    break

    clashes.sort(key=lambda c: c.penetration_m, reverse=True)
    pairs = [
        {"a_model": k[0], "b_model": k[1], "count": v}
        for k, v in sorted(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return FederationResult(
        clashes=clashes,
        model_box_counts=model_box_counts,
        pairs=pairs,
        clash_count=len(clashes),
        truncated=truncated,
    )
