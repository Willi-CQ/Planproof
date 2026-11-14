"""Raum-Erkennung aus dem Wand-Achsennetz.

Raeume sind die geschlossenen Flaechen (Faces) des planaren Wand-Graphen. Wir
nutzen ``shapely.ops.polygonize`` ueber den Achsen -- die robuste Standard-
methode, um aus einem Liniennetz Polygone zu gewinnen. Jedem Polygon wird das
TEXT/MTEXT-Label zugeordnet, dessen Position darin liegt (z.B. "Wohnzimmer",
"Bad"), plus Flaeche und Schwerpunkt. Das Resultat ist mit
``build_scene_from_wall_graph`` (rooms_list) und dem glTF-Floor-Renderer
kompatibel.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from .dxf_loader import RawText

logger = logging.getLogger(__name__)

Point2 = Tuple[float, float]


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
    [{id, label, area_m2, center:[x,y], polygon:[[x,y],...], meta:{room_type}}]
    """
    config = config or {}
    min_area = float(config.get("room_min_area_m2", 1.0))
    max_area = float(config.get("room_max_area_m2", 100000.0))
    grid_m = float(config.get("room_snap_grid_m", 0.005))

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

    # TEXT-Punkte vorbereiten
    text_pts = [(Point(t.pos[0], t.pos[1]), t.text) for t in texts if t.text and t.text.strip()]

    rooms: List[dict] = []
    idx = 0
    for poly in polygons:
        area = float(poly.area)
        if area < min_area or area > max_area:
            continue
        # Label: erstes TEXT, das im Polygon liegt
        label = None
        for pt, txt in text_pts:
            if poly.contains(pt):
                label = txt.strip()
                break
        c = poly.representative_point()
        # Aussenring als Polygon-Liste
        ring = [[float(x), float(y)] for x, y in poly.exterior.coords[:-1]]
        rt = _room_type(label)
        rooms.append({
            "id": f"room_{idx}",
            "label": label,
            "area_m2": round(area, 2),
            "center": [float(c.x), float(c.y)],
            "polygon": ring,
            "meta": {"room_type": rt} if rt else {},
        })
        idx += 1

    logger.info("Raeume erkannt: %d (von %d Faces)", len(rooms), len(polygons))
    return rooms


__all__ = ["detect_rooms"]
