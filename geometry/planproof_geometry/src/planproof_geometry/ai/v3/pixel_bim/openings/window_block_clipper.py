"""Hartes Kriterium #1 — Phase 2: Bloecke auf Wand x Fenster-Breite zuschneiden.

Eingabe:
  - ``WindowBlock``s aus Phase 1
  - User-Wand-Polygone (Liste von Punkt-Listen in Metern)

Ausgabe: ``ClippedWindowBlock`` pro Eingabe-Block mit:
  - ``polygon_m``: 4-Punkt-Rechteck (CCW) im Block-Frame
  - Achsen-Bookkeeping (tangent, perp, s_lo/hi, perp_lo/hi)

Strategie:
  Die Block-Bbox-Long-Axis ist der Tangenten-Anker (parallel zur
  Wand-Linie, kommt aus den User-gesetzten Fenster-Labels).
  Die Wand-Dicke kommt aus den Wand-Polygon-Punkten, die LOKAL im
  Block-Bereich liegen — nicht aus ``minAreaRect`` der ganzen Wand.
  Damit funktioniert es auch fuer L-shaped, U-shaped oder konkave
  Wand-Annotationen (Matterport-Aussenwand = ein Polygon ueber die
  ganze Aussenkante).

Walls-immutable: Wand-Polygone werden ausschliesslich gelesen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .window_label_blocks import WindowBlock


# Senkrechter Suchradius beim Sammeln von Wand-Punkten um den Block.
# 1.5 m deckt jede plausible Wand-Dicke + Annotator-Drift ab.
PERP_SEARCH_RADIUS_M = 1.5

# Kleiner tangentialer Padding-Bereich um den Block, damit Wand-Punkte
# direkt an den Block-Enden noch mitgenommen werden.
TANGENT_PADDING_M = 0.10

# Fallback-Wand-Dicke wenn kein Wand-Polygon-Punkt im Block-Bereich
# gefunden wird (DIN-Standard Innenwand mit 11.5 cm Mauerstein +
# Putz auf jeder Seite ~ 17.5 cm — wir nehmen 0.30 m als sichere
# Default die der Algorithmus signalisieren kann).
FALLBACK_WALL_THICKNESS_M = 0.30


@dataclass
class ClippedWindowBlock:
    """Block + auf die lokale Wand zugeschnittenes Rechteck."""
    block: WindowBlock
    wall_idx: Optional[int]
    polygon_m: List[Tuple[float, float]]            # 4 Punkte, CCW
    axis_origin_m: Tuple[float, float]
    axis_tangent: Tuple[float, float]
    axis_perp: Tuple[float, float]
    axis_s_lo_m: float
    axis_s_hi_m: float
    axis_perp_lo_m: float
    axis_perp_hi_m: float
    wall_thickness_m: float
    notes: List[str] = field(default_factory=list)

    @property
    def length_m(self) -> float:
        return self.axis_s_hi_m - self.axis_s_lo_m

    @property
    def count(self) -> int:
        return self.block.count


def _project(point: Tuple[float, float],
             origin: Tuple[float, float],
             axis: Tuple[float, float]) -> float:
    return (point[0] - origin[0]) * axis[0] + (point[1] - origin[1]) * axis[1]


def _bbox_corners(bbox: Tuple[float, float, float, float]
                  ) -> List[Tuple[float, float]]:
    x0, y0, x1, y1 = bbox
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _block_axis(block: WindowBlock
                ) -> Tuple[Tuple[float, float], Tuple[float, float],
                           Tuple[float, float]]:
    """Block-Long-Axis aus der Bbox.

    Returns (origin, tangent, perp) im Welt-Frame.
    Tangent zeigt entlang der laengeren Bbox-Seite.
    """
    x0, y0, x1, y1 = block.bbox_m
    width = x1 - x0
    height = y1 - y0
    origin = block.centroid_m
    if width >= height:
        tangent = (1.0, 0.0)
        perp = (0.0, 1.0)
    else:
        tangent = (0.0, 1.0)
        perp = (-1.0, 0.0)
    return origin, tangent, perp


def _sample_wall_points(
    polygon_m: Sequence[Tuple[float, float]],
    *,
    samples_per_edge: int = 8,
) -> List[Tuple[float, float]]:
    """Erzeuge dichte Punkte auf den Polygon-Edges.

    Die User-Annotation hat oft nur Eckpunkte (4–8 pro Polygon). Damit
    wir die Wand-Dicke an einer beliebigen Block-Position bestimmen
    koennen, sample wir die Edges. ``samples_per_edge`` Punkte zwischen
    je zwei Eckpunkten plus die Eckpunkte selbst.
    """
    n = len(polygon_m)
    if n < 2:
        return list(polygon_m)
    samples: List[Tuple[float, float]] = []
    for i in range(n):
        a = polygon_m[i]
        b = polygon_m[(i + 1) % n]
        for k in range(samples_per_edge):
            t = k / samples_per_edge
            samples.append((a[0] + t * (b[0] - a[0]),
                            a[1] + t * (b[1] - a[1])))
    return samples


def _snap_short_side_to_wall_edge(
    s_block_end: float,
    side: str,                   # "left" or "right"
    block_perp_lo: float,
    block_perp_hi: float,
    origin: Tuple[float, float],
    tangent: Tuple[float, float],
    perp: Tuple[float, float],
    wall_polygons_m: Sequence[Sequence[Tuple[float, float]]],
    *,
    max_search_m: float = 0.30,
    perp_overlap_tol_m: float = 0.05,
) -> Tuple[float, bool]:
    """Snap die kurze Block-Seite an die naechste Wand-Polygon-Edge.

    Sucht entlang ``-tangent`` (side="left") bzw. ``+tangent`` (side="right")
    nach der naechsten Wand-Polygon-Edge die ein **kurzes Wand-Ende** ist
    (also senkrecht zur Block-Tangente steht) und im perp-Bereich des
    Blocks liegt. Snappt auf deren Tangent-Projektion.

    Returns ``(s_new, snapped)`` wobei ``snapped=True`` wenn eine geeignete
    Edge gefunden wurde.
    """
    perp_band_lo = block_perp_lo - perp_overlap_tol_m
    perp_band_hi = block_perp_hi + perp_overlap_tol_m

    best_s = s_block_end
    best_dist = float("inf")
    for wp in wall_polygons_m:
        n = len(wp)
        if n < 2:
            continue
        for i in range(n):
            a = wp[i]
            b = wp[(i + 1) % n]
            ax_len = b[0] - a[0]
            ay_len = b[1] - a[1]
            edge_len = (ax_len * ax_len + ay_len * ay_len) ** 0.5
            if edge_len < 1e-3:
                continue
            edge_dir_x = ax_len / edge_len
            edge_dir_y = ay_len / edge_len
            # Dot-Produkt mit Block-Tangente: ~0 = senkrecht zur Tangente
            # (d.h. eine "kurze" Wand-Edge), ~1 = parallel
            along_t = abs(edge_dir_x * tangent[0] + edge_dir_y * tangent[1])
            if along_t > 0.30:  # nicht senkrecht genug
                continue
            # Perp-Bereich der Edge muss sich mit dem Block ueberlappen
            a_perp = _project(a, origin, perp)
            b_perp = _project(b, origin, perp)
            edge_perp_lo = min(a_perp, b_perp)
            edge_perp_hi = max(a_perp, b_perp)
            if edge_perp_hi < perp_band_lo or edge_perp_lo > perp_band_hi:
                continue
            # Edge-Mittelpunkt-Tangent-Projektion
            em_s = 0.5 * (_project(a, origin, tangent)
                          + _project(b, origin, tangent))
            # Side-Filter
            if side == "left" and em_s > s_block_end + 1e-6:
                continue
            if side == "right" and em_s < s_block_end - 1e-6:
                continue
            dist = abs(em_s - s_block_end)
            if dist > max_search_m:
                continue
            if dist < best_dist:
                best_dist = dist
                best_s = em_s
    return (best_s, best_dist != float("inf"))


def clip_blocks_to_walls(
    blocks: Sequence[WindowBlock],
    wall_polygons_m: Sequence[Sequence[Tuple[float, float]]],
    *,
    perp_search_radius_m: float = PERP_SEARCH_RADIUS_M,
    tangent_padding_m: float = TANGENT_PADDING_M,
    fallback_wall_thickness_m: float = FALLBACK_WALL_THICKNESS_M,
    snap_short_sides_to_walls: bool = True,
    snap_max_distance_m: float = 0.30,
) -> List[ClippedWindowBlock]:
    """Schneide jeden Block auf die lokale Wand zu + snappe an Wand-Edges.

    Ablauf pro Block:
      1. Block-Achse = Long-Axis der Block-Bbox.
      2. Achsen-Span (s_lo, s_hi) initial aus den Member-Bbox-Eckpunkten.
      3. **Snap kurze Seiten** an die naechste Wand-Polygon-Edge in
         tangentialer Richtung (geschlossener Wand-Anschluss ohne Gap).
         Wand-Polygone sollten die FERTIGEN BIM-Waende
         (``extrude_result.polygons``) sein; siehe Memory
         ``feedback_use_extruded_walls_not_labels.md``.
      4. Sammle Wand-Polygon-Punkte (gesampelt entlang der Edges) im
         lokalen Bereich -> Wand-Dicke perp_lo/perp_hi.
      5. Polygon = 4 Eckpunkte im Block-Frame -> Welt-Koord.
      6. wall_idx = Wand mit den meisten Beitraegen zum Punkt-Set.
    """
    out: List[ClippedWindowBlock] = []
    for block in blocks:
        origin, tangent, perp = _block_axis(block)

        bboxes = block.member_bboxes_m or [block.bbox_m]
        s_values: List[float] = []
        for bb in bboxes:
            for c in _bbox_corners(bb):
                s_values.append(_project(c, origin, tangent))
        if not s_values:
            continue
        s_lo = min(s_values)
        s_hi = max(s_values)

        # Wand-Punkte einsammeln, die LOKAL im Block-Bereich liegen.
        contributors_per_wall: List[int] = [0] * len(wall_polygons_m)
        local_perps: List[float] = []
        s_lo_pad = s_lo - tangent_padding_m
        s_hi_pad = s_hi + tangent_padding_m
        for w_idx, wp in enumerate(wall_polygons_m):
            if len(wp) < 2:
                continue
            samples = _sample_wall_points(wp, samples_per_edge=8)
            for pt in samples:
                s = _project(pt, origin, tangent)
                if s < s_lo_pad or s > s_hi_pad:
                    continue
                p = _project(pt, origin, perp)
                if abs(p) > perp_search_radius_m:
                    continue
                local_perps.append(p)
                contributors_per_wall[w_idx] += 1

        notes: List[str] = []
        if local_perps:
            perp_lo = min(local_perps)
            perp_hi = max(local_perps)
            wall_idx: Optional[int] = max(
                range(len(contributors_per_wall)),
                key=lambda i: contributors_per_wall[i],
            )
            if contributors_per_wall[wall_idx] == 0:
                wall_idx = None
        else:
            half = 0.5 * fallback_wall_thickness_m
            perp_lo = -half
            perp_hi = +half
            wall_idx = None
            notes.append("fallback_no_wall_in_local_strip")

        thickness = perp_hi - perp_lo
        if thickness < 1e-3:
            half = 0.5 * fallback_wall_thickness_m
            perp_lo, perp_hi = -half, +half
            thickness = fallback_wall_thickness_m
            notes.append("fallback_degenerate_thickness")

        # Schritt 3 — Snap kurze Seiten an naechste Wand-Edge.
        if snap_short_sides_to_walls:
            new_s_lo, snapped_lo = _snap_short_side_to_wall_edge(
                s_lo, "left", perp_lo, perp_hi,
                origin, tangent, perp, wall_polygons_m,
                max_search_m=snap_max_distance_m,
            )
            new_s_hi, snapped_hi = _snap_short_side_to_wall_edge(
                s_hi, "right", perp_lo, perp_hi,
                origin, tangent, perp, wall_polygons_m,
                max_search_m=snap_max_distance_m,
            )
            if snapped_lo:
                notes.append(f"snapped_left_{(s_lo - new_s_lo)*1000:+.1f}mm")
                s_lo = new_s_lo
            if snapped_hi:
                notes.append(f"snapped_right_{(new_s_hi - s_hi)*1000:+.1f}mm")
                s_hi = new_s_hi

        ox, oy = origin
        tx, ty = tangent
        px, py = perp

        def _world(s: float, t: float) -> Tuple[float, float]:
            return (ox + s * tx + t * px, oy + s * ty + t * py)

        polygon = [
            _world(s_lo, perp_lo),
            _world(s_hi, perp_lo),
            _world(s_hi, perp_hi),
            _world(s_lo, perp_hi),
        ]

        out.append(ClippedWindowBlock(
            block=block,
            wall_idx=wall_idx,
            polygon_m=polygon,
            axis_origin_m=origin,
            axis_tangent=tangent,
            axis_perp=perp,
            axis_s_lo_m=float(s_lo),
            axis_s_hi_m=float(s_hi),
            axis_perp_lo_m=float(perp_lo),
            axis_perp_hi_m=float(perp_hi),
            wall_thickness_m=float(thickness),
            notes=notes,
        ))
    return out
