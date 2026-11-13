"""Tuer-/Fenster-Bloecke -> parametrische Oeffnungen am WallGraph.

Jeder als Tuer/Fenster klassifizierte ``INSERT`` ist ein bereits gelabeltes
Objekt: Typ (Block-Name), Position, **Rotation** (= Schwingrichtung) und Breite.
Die Oeffnung wird an die Wand gehaengt, deren Achse die orientierte Bounding-Box des Blocks
trifft (nicht der Einfuegepunkt: Tuerbloecke haben ihren Ursprung am Drehpunkt, nicht in der Mitte).

Position und Breite entlang der Wand kommen aus der Ausdehnung der orientierten Block-Box in
Richtung der Wand. Ein gedrehter Block behaelt damit seine echte Breite; die aufgeblaehte
achsparallele Box der Drehung geht nicht ein. Hoehen (Tuersturz, Bruestung, Fenstersturz) stehen
nicht in der Zeichnung und kommen aus der Konfiguration; die Pipeline weist sie als Annahme aus.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from planproof_cad.model.wall_graph import ParametricOpening, WallGraph

from . import layer_classifier as lc
from .dxf_loader import RawInsert

logger = logging.getLogger(__name__)

Point2 = Tuple[float, float]

MIN_OPENING_W = 0.3
MAX_OPENING_W = 5.0


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


def _attrib_width(ins: RawInsert) -> Optional[float]:
    """ATTRIB WIDTH / BREITE (in mm, cm oder m -> heuristisch), sonst None."""
    for tag in ("WIDTH", "BREITE", "B", "W"):
        if tag in ins.attribs:
            try:
                raw = float(str(ins.attribs[tag]).replace(",", "."))
                # Werte > 5 sind vermutlich mm/cm -> auf Meter bringen
                if raw > 5:
                    raw = raw / 1000.0 if raw > 50 else raw / 100.0
                if MIN_OPENING_W <= raw <= MAX_OPENING_W:
                    return raw
            except (TypeError, ValueError):
                pass
    return None


def _width_from_insert(ins: RawInsert, default_m: float) -> float:
    w = _attrib_width(ins)
    if w is not None:
        return w
    # Block-Ausdehnung im Blocksystem (laengere Seite)
    if ins.width_m and ins.height_m:
        w = max(ins.width_m, ins.height_m)
        if MIN_OPENING_W <= w <= MAX_OPENING_W:
            return w
    if ins.width_m and MIN_OPENING_W <= ins.width_m <= MAX_OPENING_W:
        return ins.width_m
    return default_m


def _box(ins: RawInsert) -> np.ndarray:
    """Vier Ecken der orientierten Box; ohne Boxdaten ein Punkt (der Einfuegepunkt)."""
    if ins.corners and len(ins.corners) >= 3:
        return np.array(ins.corners, dtype=float)
    return np.array([ins.pos], dtype=float)


def _find_host(graph: WallGraph, box: np.ndarray, max_perp: float):
    """Wand, deren Achse die Box trifft (Abstand 0) oder ihr am naechsten liegt.

    Rueckgabe: (edge, along_min, along_max, perp) mit Lage entlang der Kante ab node_a, oder None.
    """
    best = None
    for edge in graph.edges.values():
        a, b = graph.edge_endpoints(edge.id)
        d = np.array(b) - np.array(a)
        L = float(np.hypot(*d))
        if L < 1e-9:
            continue
        u = d / L
        n = np.array([-u[1], u[0]])
        rel = box - np.array(a)
        along = rel @ u
        side = rel @ n
        lo, hi = float(along.min()), float(along.max())
        if hi < 0.0 or lo > L:
            continue  # die Box liegt neben der Kante, nicht an ihr
        straddles = side.min() <= 0.0 <= side.max()
        perp = 0.0 if straddles else float(np.abs(side).min())
        if perp > max_perp:
            continue
        overlap = min(hi, L) - max(lo, 0.0)
        key = (round(perp, 3), -overlap)
        if best is None or key < best[0]:
            best = (key, edge, lo, hi, perp)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


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

        box = _box(ins)
        hit = _find_host(graph, box, max_perp)
        if hit is None:
            n_unmatched += 1
            continue
        edge, lo, hi, perp = hit

        # Breite: ATTRIB, sonst Ausdehnung der Box entlang der Wand, sonst Block-Ausdehnung/Standard
        attrib_w = _attrib_width(ins)
        extent = hi - lo
        L = graph.edge_length(edge.id)
        if attrib_w is not None:
            width, width_source = attrib_w, "attrib"
        elif len(box) >= 3 and MIN_OPENING_W <= extent <= MAX_OPENING_W:
            width, width_source = extent, "block_bbox"
        else:
            width, width_source = _width_from_insert(ins, default_w), "block_or_default"

        # Mitte entlang der Wand: Mitte der Box (bei fehlender Box der Einfuegepunkt)
        center_along = 0.5 * (lo + hi)
        half = width * 0.5
        offset_m = max(half, min(L - half, center_along)) if L > width else L * 0.5

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
                "width_source": width_source,
                "position_source": "block_bbox_center" if len(box) >= 3 else "insert_point",
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


def opening_boxes(inserts: List[RawInsert]) -> List[List[Point2]]:
    """Orientierte Boxen (vier Ecken) der Tuer-/Fensterbloecke, fuer das Schliessen von
    Wandluecken."""
    out: List[List[Point2]] = []
    for ins in inserts:
        if lc.classify_block(ins.name, ins.layer).cls in (lc.DOOR, lc.WINDOW) and ins.corners:
            out.append(list(ins.corners))
    return out


__all__ = ["extract_openings", "opening_boxes"]
