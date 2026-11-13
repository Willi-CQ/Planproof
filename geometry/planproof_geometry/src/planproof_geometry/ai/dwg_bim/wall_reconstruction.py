"""Wand-Rekonstruktion aus Wand-Layer-Linien.

Zwei Zeichenstile werden unterstuetzt und automatisch erkannt:

  * **Doppellinien-Waende** (der Regelfall): jede Wand ist ein Paar paralleler
    Linien. Aus dem Paar werden Achse (Mittellinie) und Dicke (Abstand)
    gewonnen -- die klassische ISARC-2022-Methode.
  * **Einzellinien-Waende** (schematisch / Wandobjekt-Export): Linien sind
    bereits Achsen; es wird eine Default-Dicke angesetzt.

Ergebnis ist eine Liste von ``Centerline``-Achsen (Meter, mit Dicke + Wandtyp),
die ``build_wall_graph`` zu einem topologisch sauberen ``WallGraph`` verdichtet
(Knoten-Snapping + T-Stoss-Splitting).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from planproof_geometry.domain.wall_graph import WallGraph

from .dxf_loader import RawSegment
from . import layer_classifier as lc

Point2 = Tuple[float, float]


@dataclass
class Centerline:
    a: Point2
    b: Point2
    thickness_m: float
    wall_type: str           # "exterior" | "interior"
    source: str = "dwg_double_line"

    def length(self) -> float:
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])


# ── Vektor-Helfer ────────────────────────────────────────────────────────

def _np(p: Point2) -> np.ndarray:
    return np.array([p[0], p[1]], dtype=float)


def _seg_dir(a: Point2, b: Point2) -> Tuple[np.ndarray, float]:
    v = _np(b) - _np(a)
    L = float(np.hypot(v[0], v[1]))
    if L < 1e-9:
        return np.array([1.0, 0.0]), 0.0
    return v / L, L


def _angle_mod_pi(u: np.ndarray) -> float:
    """Richtungswinkel in [0, pi) (Linie hat keine Orientierung)."""
    a = math.atan2(u[1], u[0])
    if a < 0:
        a += math.pi
    if a >= math.pi:
        a -= math.pi
    return a


def _project_scalar(p: np.ndarray, origin: np.ndarray, u: np.ndarray) -> float:
    return float(np.dot(p - origin, u))


def _point_line_perp(p: np.ndarray, a: np.ndarray, u: np.ndarray) -> float:
    """Vorzeichenbehafteter Normalabstand von p zur Geraden (a, u)."""
    n = np.array([-u[1], u[0]])
    return float(np.dot(p - a, n))


# ── Doppellinien-Paarung ─────────────────────────────────────────────────

def _pair_double_lines(
    segs: List[Tuple[RawSegment, np.ndarray, float, float]],
    *,
    min_thickness_m: float,
    max_thickness_m: float,
    angle_tol_deg: float,
    overlap_frac: float,
) -> Tuple[List[Centerline], set]:
    """Findet parallele Linienpaare und bildet Achsen.

    ``segs`` ist eine vorbereitete Liste (RawSegment, dir_unit, angle, length).
    Liefert (centerlines, consumed_indices).
    """
    angle_tol = math.radians(angle_tol_deg)
    n = len(segs)
    consumed: set = set()
    centerlines: List[Centerline] = []

    # Sortiere nach Winkel fuer guenstigeres Nachbar-Fenster
    order = sorted(range(n), key=lambda i: segs[i][2])

    for oi in range(n):
        i = order[oi]
        if i in consumed:
            continue
        seg_i, ui, ang_i, len_i = segs[i]
        ai, bi = _np(seg_i.a), _np(seg_i.b)

        best_j = -1
        best_perp = 0.0
        best_overlap = 0.0

        # nur Kandidaten mit aehnlichem Winkel betrachten
        for oj in range(oi + 1, n):
            j = order[oj]
            if j in consumed:
                continue
            seg_j, uj, ang_j, len_j = segs[j]
            dang = abs(ang_i - ang_j)
            dang = min(dang, math.pi - dang)
            if dang > angle_tol:
                # Winkel sortiert -> sobald zu weit, abbrechen (mit Toleranz fuer Wrap)
                if ang_j - ang_i > angle_tol and ang_j < math.pi - angle_tol:
                    break
                continue

            aj = _np(seg_j.a)
            # Normalabstand von j zur Geraden i
            perp = abs(_point_line_perp(aj, ai, ui))
            perp_b = abs(_point_line_perp(_np(seg_j.b), ai, ui))
            # beide Endpunkte muessen aehnlich weit weg sein (echt parallel)
            if abs(perp - perp_b) > max(0.04, 0.25 * max(perp, perp_b)):
                continue
            thick = 0.5 * (perp + perp_b)
            if not (min_thickness_m <= thick <= max_thickness_m):
                continue

            # Projektions-Overlap entlang Richtung i
            ti0 = _project_scalar(ai, ai, ui)
            ti1 = _project_scalar(bi, ai, ui)
            tj0 = _project_scalar(aj, ai, ui)
            tj1 = _project_scalar(_np(seg_j.b), ai, ui)
            lo = max(min(ti0, ti1), min(tj0, tj1))
            hi = min(max(ti0, ti1), max(tj0, tj1))
            overlap = hi - lo
            if overlap <= 0:
                continue
            if overlap < overlap_frac * min(len_i, len_j):
                continue

            # bestes Paar = laengste Ueberlappung
            if overlap > best_overlap:
                best_overlap = overlap
                best_j = j
                best_perp = thick

        if best_j < 0:
            continue

        seg_j, uj, ang_j, len_j = segs[best_j]
        aj, bj = _np(seg_j.a), _np(seg_j.b)

        # Achse = Mittellinie ueber dem Overlap-Intervall entlang ui
        ti0 = _project_scalar(ai, ai, ui)
        ti1 = _project_scalar(bi, ai, ui)
        tj0 = _project_scalar(aj, ai, ui)
        tj1 = _project_scalar(bj, ai, ui)
        lo = max(min(ti0, ti1), min(tj0, tj1))
        hi = min(max(ti0, ti1), max(tj0, tj1))
        # Seitlicher Versatz: halber vorzeichenbehafteter Abstand
        signed = 0.5 * (_point_line_perp(aj, ai, ui) + _point_line_perp(bj, ai, ui)) * 0.5
        nvec = np.array([-ui[1], ui[0]])
        c_lo = ai + ui * lo + nvec * signed
        c_hi = ai + ui * hi + nvec * signed

        wall_type = _wall_type_of(seg_i, seg_j)
        centerlines.append(Centerline(
            a=(float(c_lo[0]), float(c_lo[1])),
            b=(float(c_hi[0]), float(c_hi[1])),
            thickness_m=round(float(best_perp), 3),
            wall_type=wall_type,
        ))
        consumed.add(i)
        consumed.add(best_j)

    return centerlines, consumed


def _wall_type_of(seg_i: RawSegment, seg_j: RawSegment) -> str:
    for seg in (seg_i, seg_j):
        c = lc.classify_layer(seg.layer).cls
        if c == lc.WALL_EXTERIOR:
            return "exterior"
        if c == lc.WALL_INTERIOR:
            return "interior"
    return "interior"


# ── Hauptfunktion: Linien -> Achsen ──────────────────────────────────────

def reconstruct_walls(
    wall_segments: List[RawSegment],
    config: Optional[dict] = None,
) -> Tuple[List[Centerline], dict]:
    """Rekonstruiert Wand-Achsen aus klassifizierten Wand-Linien.

    Returns (centerlines, stats).
    """
    config = config or {}
    min_t = float(config.get("wall_min_thickness_m", 0.05))
    max_t = float(config.get("wall_max_thickness_m", 0.60))
    angle_tol = float(config.get("wall_angle_tol_deg", 6.0))
    overlap_frac = float(config.get("wall_overlap_frac", 0.30))
    default_t_int = float(config.get("wall_thickness_interior_m", 0.115))
    default_t_ext = float(config.get("wall_thickness_exterior_m", 0.30))
    min_len = float(config.get("wall_min_length_m", 0.10))

    segs_prepared = []
    for s in wall_segments:
        if s.length() < min_len:
            continue
        u, L = _seg_dir(s.a, s.b)
        segs_prepared.append((s, u, _angle_mod_pi(u), L))

    total_len = sum(p[3] for p in segs_prepared)
    centerlines, consumed = _pair_double_lines(
        segs_prepared,
        min_thickness_m=min_t,
        max_thickness_m=max_t,
        angle_tol_deg=angle_tol,
        overlap_frac=overlap_frac,
    )

    matched_len = sum(c.length() for c in centerlines) * 2.0  # Paar = 2 Linien
    matched_ratio = (matched_len / total_len) if total_len > 0 else 0.0

    mode = "double_line"
    # Wenig gepaart -> Zeichnung ist vermutlich einlinig: alle Linien = Achsen
    if matched_ratio < 0.25:
        mode = "single_line"
        centerlines = []
        for s, u, ang, L in segs_prepared:
            ct = lc.classify_layer(s.layer).cls
            wt = "exterior" if ct == lc.WALL_EXTERIOR else "interior"
            thick = default_t_ext if wt == "exterior" else default_t_int
            centerlines.append(Centerline(
                a=s.a, b=s.b, thickness_m=thick, wall_type=wt,
                source="dwg_single_line",
            ))

    stats = {
        "mode": mode,
        "input_wall_segments": len(wall_segments),
        "centerlines": len(centerlines),
        "matched_ratio": round(matched_ratio, 3),
        "total_wall_length_m": round(sum(c.length() for c in centerlines), 2),
    }
    return centerlines, stats


# ── Achsen -> WallGraph (Snapping + T-Stoss-Splitting) ───────────────────

def build_wall_graph(
    centerlines: List[Centerline],
    config: Optional[dict] = None,
    wall_height_m: float = 2.80,
) -> WallGraph:
    """Verdichtet Achsen zu einem topologisch sauberen WallGraph.

    1. Knoten-Snapping ueber ``snap_radius_m`` (Junctions zusammenfuehren)
    2. T-Stoss-Splitting: liegt ein Knoten auf dem Inneren einer Kante,
       wird die Kante dort geteilt (sonst keine geschlossenen Raum-Faces).
    """
    config = config or {}
    snap_r = float(config.get("wall_snap_radius_m", 0.25))
    split_tol = float(config.get("wall_split_tol_m", 0.12))

    graph = WallGraph()
    for cl in centerlines:
        if cl.length() < 1e-6:
            continue
        try:
            graph.add_edge_from_segment(
                cl.a, cl.b,
                thickness_m=cl.thickness_m,
                height_m=wall_height_m,
                wall_type=cl.wall_type,
                source=cl.source,
                snap_radius_m=snap_r,
            )
        except ValueError:
            # Endpunkte auf denselben Knoten gesnappt -> Null-Wand, skip
            continue

    _split_edges_at_nodes(graph, split_tol)
    graph.refresh_node_kinds()
    graph.prune_orphan_nodes()
    return graph


def _split_edges_at_nodes(graph: WallGraph, tol_m: float) -> None:
    """Teilt Kanten, durch deren Inneres ein bestehender Knoten verlaeuft."""
    changed = True
    guard = 0
    while changed and guard < 1000:
        changed = False
        guard += 1
        for edge in list(graph.edges.values()):
            a_xy, b_xy = graph.edge_endpoints(edge.id)
            L = graph.edge_length(edge.id)
            if L < 2 * tol_m:
                continue
            ua, _ = _seg_dir(a_xy, b_xy)
            best_node = None
            best_off = None
            for node in graph.nodes.values():
                if node.id in (edge.node_a, edge.node_b):
                    continue
                off, perp = graph.project_to_edge(edge.id, node.xy)
                # echter Innenpunkt mit kleinem Normalabstand
                if perp <= tol_m and (tol_m < off < L - tol_m):
                    if best_off is None or off < best_off:
                        best_off = off
                        best_node = node
            if best_node is not None:
                # Kante an best_node teilen: zwei neue Kanten, alte entfernen
                graph.remove_edge(edge.id)
                for (na, nb) in ((edge.node_a, best_node.id), (best_node.id, edge.node_b)):
                    if na == nb:
                        continue
                    graph.add_edge(
                        na, nb,
                        thickness_m=edge.thickness_m,
                        height_m=edge.height_m,
                        wall_type=edge.wall_type,
                        source=edge.source,
                        confidence=edge.confidence,
                    )
                changed = True
                break


__all__ = [
    "Centerline",
    "reconstruct_walls",
    "build_wall_graph",
]
