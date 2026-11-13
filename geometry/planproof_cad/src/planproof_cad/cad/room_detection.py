"""Raum-Erkennung aus dem Wand-Achsennetz.

Raeume sind die geschlossenen Flaechen (Faces) des planaren Wand-Graphen. Wir
nutzen ``shapely.ops.polygonize`` ueber den Achsen -- die robuste Standard-
methode, um aus einem Liniennetz Polygone zu gewinnen.

Flaeche: Ein Face aus den Achsen ist groesser als der Raum (die halbe Wanddicke jeder Wand liegt
darin). ``area_axis_m2`` ist die Achsflaeche; ``area_net_m2`` die lichte Flaeche, die entsteht, wenn
die tatsaechlichen Wandkoerper (Achse +/- halbe Dicke der jeweiligen Wand) abgezogen werden.
``area_m2`` ist die lichte Flaeche, ``polygon`` deren Aussenring (``polygon_axis`` der Achsring).

Label: das TEXT/MTEXT, das im Face liegt. Massangaben, rein numerische Texte und Texte auf
Beschriftungs-, Bemassungs-, Tuer- und Fensterlayern sind keine Raumnamen; Texte auf Raumlayern
haben Vorrang vor allen anderen.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from . import layer_classifier as lc
from .dxf_loader import RawText

logger = logging.getLogger(__name__)

Point2 = Tuple[float, float]

_NUMERIC_ONLY = re.compile(r"^[\s\d.,:;/\\+\-*x×°²³'\"()=%]*$", re.IGNORECASE)
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

# Label-Keyword -> room_type (fuer Boden-Farbgebung + IfcSpace-Klassifikation)
_ROOM_TYPE_KEYWORDS = (
    ("wohn", "living"),
    ("schlaf", "bedroom"),
    ("kinder", "bedroom"),
    ("bad", "bathroom"),
    ("wc", "bathroom"),
    ("dusche", "bathroom"),
    ("kueche", "kitchen"),
    ("küche", "kitchen"),
    ("kuche", "kitchen"),
    ("koch", "kitchen"),
    ("flur", "corridor"),
    ("diele", "corridor"),
    ("gang", "corridor"),
    ("buero", "office"),
    ("büro", "office"),
    ("office", "office"),
    ("lager", "storage"),
    ("abstell", "storage"),
    ("keller", "basement"),
    ("technik", "technical"),
    ("hwr", "technical"),
)


def _room_type(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    low = label.lower()
    for kw, rt in _ROOM_TYPE_KEYWORDS:
        if kw in low:
            return rt
    return "generic"


def is_room_label_candidate(text: str, layer: str) -> bool:
    """Kann dieser Text ein Raumname sein? Nein bei Massen, Nummern und
    Beschriftungs-/Bauteillayern."""
    t = (text or "").strip()
    if not t or _NUMERIC_ONLY.match(t) or not _HAS_LETTER.search(t):
        return False
    if lc.is_room_text_layer(layer):
        return True
    cls = lc.classify_layer(layer).cls
    if cls == lc.ANNOTATION:
        return lc.is_generic_text_layer(layer)  # Bemassung, Schraffur, Tuernummern: nein
    return cls not in (lc.DOOR, lc.WINDOW)


def _pick_label(poly, texts) -> Optional[str]:
    """Bester Text im Polygon: Raumlayer vor sonstigen, dann der naechste zum inneren Punkt."""
    ref = poly.representative_point()
    best: Optional[Tuple[int, float, str]] = None
    for pt, txt, layer in texts:
        if not poly.contains(pt):
            continue
        rank = 0 if lc.is_room_text_layer(layer) else 1
        key = (rank, ref.distance(pt), txt.strip())
        if best is None or key < best:
            best = key
    return best[2] if best else None


def _wall_bodies(graph):
    """Vereinigung aller Wandkoerper (Achse +/- halbe Dicke, gerade Enden)."""
    from shapely.geometry import LineString  # noqa: PLC0415
    from shapely.ops import unary_union  # noqa: PLC0415

    bodies = []
    for edge in graph.edges.values():
        a, b = graph.edge_endpoints(edge.id)
        if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 < 1e-12:
            continue
        bodies.append(
            LineString([a, b]).buffer(max(edge.thickness_m, 0.01) / 2.0, cap_style="flat")
        )
    return unary_union(bodies) if bodies else None


def _largest_polygon(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    return max(parts, key=lambda g: g.area) if parts else None


def detect_rooms(
    graph,
    texts: List[RawText],
    config: Optional[dict] = None,
) -> List[dict]:
    """Erkennt Raumpolygone (Faces) aus dem WallGraph und ordnet Labels zu.

    Quelle ist bewusst der **gesnappte** WallGraph (geteilte Junctions =
    bit-identische Endpunkt-Koordinaten), nicht die rohen Mittellinien --
    sonst verhindern Gleitkomma-Mikroluecken das Noding einzelner Faces.
    Zusaetzlich wird auf ein mm-Raster gesnappt.

    Returns rooms_list (kompatibel mit build_scene_from_wall_graph):
    [{id, label, area_m2, area_axis_m2, area_net_m2, center:[x,y], polygon:[[x,y],...],
      polygon_axis, meta:{room_type}}]
    """
    config = config or {}
    min_area = float(config.get("room_min_area_m2", 1.0))
    max_area = float(config.get("room_max_area_m2", 100000.0))
    grid_m = float(config.get("room_snap_grid_m", 0.005))
    area_mode = str(config.get("room_area_mode", "net"))

    try:
        from shapely import set_precision
        from shapely.geometry import LineString, Point
        from shapely.ops import polygonize, unary_union
    except Exception as exc:  # noqa: BLE001
        logger.warning("shapely nicht verfuegbar, Raum-Erkennung uebersprungen: %s", exc)
        return []

    lines = []
    for edge in graph.edges.values():
        a, b = graph.edge_endpoints(edge.id)
        if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 < 1e-12:
            continue
        lines.append(LineString([a, b]))
    if not lines:
        return []

    # mm-Raster-Snap + verschmelzen (nodet Kreuzungen robust) -> Faces
    try:
        merged = unary_union(lines)
        if grid_m > 0:
            try:
                merged = set_precision(merged, grid_m)
            except Exception:  # noqa: BLE001
                pass
        polygons = list(polygonize(merged))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Polygonize fehlgeschlagen: %s", exc)
        return []

    bodies = _wall_bodies(graph)

    # TEXT-Punkte vorbereiten
    text_pts = [
        (Point(t.pos[0], t.pos[1]), t.text, t.layer)
        for t in texts
        if is_room_label_candidate(t.text, t.layer)
    ]

    rooms: List[dict] = []
    idx = 0
    for poly in polygons:
        area_axis = float(poly.area)
        if area_axis > max_area:
            continue
        net = _largest_polygon(poly.difference(bodies)) if bodies is not None else poly
        if net is None:
            continue
        area_net = float(net.area)
        if area_net < min_area:
            continue
        label = _pick_label(poly, text_pts)
        c = net.representative_point()
        ring = [[float(x), float(y)] for x, y in net.exterior.coords[:-1]]
        ring_axis = [[float(x), float(y)] for x, y in poly.exterior.coords[:-1]]
        if area_mode == "axis":  # nur fuer Vergleichsmessungen: wie vor der lichten Flaeche
            area_net, ring = area_axis, ring_axis
        rt = _room_type(label)
        rooms.append(
            {
                "id": f"room_{idx}",
                "label": label,
                "area_m2": round(area_net, 2),
                "area_net_m2": round(area_net, 2),
                "area_axis_m2": round(area_axis, 2),
                "center": [float(c.x), float(c.y)],
                "polygon": ring,
                "polygon_axis": ring_axis,
                "meta": {"room_type": rt} if rt else {},
            }
        )
        idx += 1

    logger.info("Raeume erkannt: %d (von %d Faces)", len(rooms), len(polygons))
    return rooms


__all__ = ["detect_rooms", "is_room_label_candidate"]
