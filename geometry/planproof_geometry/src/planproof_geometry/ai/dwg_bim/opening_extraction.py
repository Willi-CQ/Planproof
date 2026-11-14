"""Tuer-/Fenster-Bloecke -> parametrische Oeffnungen am WallGraph.

Jeder als Tuer/Fenster klassifizierte ``INSERT`` ist ein bereits gelabeltes
Objekt: Typ (Block-Name), Position, **Rotation** (= Schwingrichtung -- genau
der Punkt, der im Raster-Pfad teuer/unsicher ist) und Breite (aus Block-Bbox
oder ATTRIB). Die Oeffnung wird an die naechstliegende Wandkante gehaengt; die
Wand-Achse liefert Offset und Ausrichtung deterministisch -- kein ML noetig.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from planproof_geometry.domain.wall_graph import ParametricOpening, WallGraph

from .dxf_loader import RawInsert
from . import layer_classifier as lc

logger = logging.getLogger(__name__)


def _subtype_from_name(name: str, kind: str) -> Optional[str]:
    f = lc.fold(name)
    if kind == "door":
        if any(k in f for k in ("schiebe", "sliding", "slide")):
            return "door_sliding"
        if any(k in f for k in ("doppel", "double", "zweifluegel", "2fluegel")):
            return "door_double"
        return "door_single"
    if kind == "window":
        if any(k in f for k in ("band", "fensterband", "large", "panorama")):
            return "window_large"
    return None


def _width_from_insert(ins: RawInsert, default_m: float) -> float:
    # 1) ATTRIB WIDTH / BREITE (in mm oder m -> heuristisch)
    for tag in ("WIDTH", "BREITE", "B", "W"):
        if tag in ins.attribs:
            try:
                raw = float(str(ins.attribs[tag]).replace(",", "."))
                # Werte > 5 sind vermutlich mm/cm -> auf Meter bringen
                if raw > 5:
                    raw = raw / 1000.0 if raw > 50 else raw / 100.0
                if 0.3 <= raw <= 5.0:
                    return raw
            except (TypeError, ValueError):
                pass
    # 2) Block-Bbox-Breite (laengere horizontale Ausdehnung)
    if ins.width_m and ins.height_m:
        w = max(ins.width_m, ins.height_m)
        if 0.3 <= w <= 5.0:
            return w
    if ins.width_m and 0.3 <= ins.width_m <= 5.0:
        return ins.width_m
    return default_m


def extract_openings(
    graph: WallGraph,
    inserts: List[RawInsert],
    config: Optional[dict] = None,
) -> dict:
    """Haengt Tuer-/Fenster-Bloecke als ParametricOpening an den Graph.

    Returns Statistik-Dict.
    """
    config = config or {}
    max_perp = float(config.get("opening_max_perp_m", 1.0))
    door_default_w = float(config.get("door_default_width_m", 0.90))
    win_default_w = float(config.get("window_default_width_m", 1.20))
    door_head = float(config.get("door_head_m", 2.10))
    win_sill = float(config.get("window_sill_m", 0.90))
    win_head = float(config.get("window_head_m", 2.10))

    n_attached = 0
    n_unmatched = 0
    n_doors = 0
    n_windows = 0

    for ins in inserts:
        cls = lc.classify_block(ins.name, ins.layer).cls
        if cls not in (lc.DOOR, lc.WINDOW):
            continue
        kind = "door" if cls == lc.DOOR else "window"
        default_w = door_default_w if kind == "door" else win_default_w
        width = _width_from_insert(ins, default_w)

        # naechste Wandkante zur Block-Position
        hit = graph.nearest_edge(ins.pos, max_perp_m=max_perp)
        if hit is None:
            n_unmatched += 1
            continue
        edge, offset_m, perp = hit

        # Offset in den gueltigen Bereich [halbe Breite .. L - halbe Breite] ziehen
        L = graph.edge_length(edge.id)
        half = width * 0.5
        offset_m = max(half, min(L - half, offset_m))

        subtype = _subtype_from_name(ins.name, kind)
        if kind == "door":
            sill, head = 0.0, door_head
        else:
            sill, head = win_sill, win_head

        opening = ParametricOpening(
            id=f"op_{kind}_{n_attached}",
            kind=kind,
            offset_m=float(offset_m),
            width_m=float(width),
            sill_m=float(sill),
            head_m=float(head),
            subtype=subtype,
            confidence=0.9,
            source="dwg_block",
            attrs={
                "block_name": ins.name,
                "rotation_deg": round(ins.rotation_deg, 1),
                "perp_dist_m": round(perp, 3),
            },
        )
        graph.attach_opening(edge.id, opening)
        n_attached += 1
        if kind == "door":
            n_doors += 1
        else:
            n_windows += 1

    stats = {
        "inserts_total": len(inserts),
        "openings_attached": n_attached,
        "doors": n_doors,
        "windows": n_windows,
        "unmatched": n_unmatched,
    }
    logger.info("Oeffnungen: %s", stats)
    return stats


__all__ = ["extract_openings"]
