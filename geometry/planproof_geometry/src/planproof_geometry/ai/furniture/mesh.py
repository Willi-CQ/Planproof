"""Mesh-Builder fuer platzierte Moebel.

Phase 1: prozedurale Boxen, auf den platzierten Footprint + Katalog-Hoehe
skaliert, eingefaerbt nach Kategorie. Wenn ein Katalog-Item ein ``glb_path``
hat und die Datei existiert, wird stattdessen das echte Modell geladen und in
die Footprint-Box skaliert (Haken fuer den spaeteren echten Modell-Katalog).

3D-Konvention wie Tueren/Fenster: Plan (x, y) [m] -> 3D (x, hoehe, y), Y oben.
Jedes Item wird ein eigener, benannter glTF-Knoten ``furniture/<id>/<bbox>/<order>``
— damit bleibt die Klassen-Einfaerbung/Gruppierung im BIM-Editor erhalten und
Moebel sind einzeln selektierbar.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from .catalog import get_item
from .placement import PlacedFurniture, PlacedItem


def _footprint_extents(item: PlacedItem) -> Tuple[float, float, float, float, float, float]:
    xs = [p[0] for p in item.footprint_m]
    ys = [p[1] for p in item.footprint_m]
    return min(xs), max(xs), min(ys), max(ys), 0.0, item.height_m


def build_item_mesh(item: PlacedItem):
    """trimesh.Trimesh fuer ein platziertes Item (Box oder GLB). None bei Fehler."""
    try:
        import trimesh
    except ImportError:
        return None

    xmin, xmax, ymin, ymax, zmin, zmax = _footprint_extents(item)
    dx = max(1e-3, xmax - xmin)
    dy = max(1e-3, ymax - ymin)
    h = max(1e-3, zmax - zmin)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    cat = get_item(item.catalog_id)
    color = list(cat.color_rgba) if cat else [190, 190, 190, 255]

    mesh = None
    # 1. Echtes CC0-Modell (falls hinterlegt) hat Vorrang.
    if cat and cat.glb_path:
        mesh = _load_and_fit_glb(cat.glb_path, dx, h, dy, cx, cy)

    # 2. Prozedurales Standardmodell (Modelldatenbank) — erkennbare Form.
    #    ymin/ymax sind Plan-y = Welt-Z; zmax ist die Hoehe.
    if mesh is None:
        from .models import build_standard_model
        mesh = build_standard_model(
            item.catalog_id,
            xmin=xmin, xmax=xmax, zmin=ymin, zmax=ymax,
            h=h, facing_deg=item.facing_deg,
        )

    # 3. Letzter Fallback: einfache Box.
    if mesh is None:
        box = trimesh.creation.box(extents=(dx, h, dy))
        box.apply_translation((cx, 0.5 * h, cy))
        mesh = box

    try:
        mesh.visual.face_colors = color
    except Exception:  # noqa: BLE001 — Farbe ist kosmetisch
        pass
    return mesh


def _load_and_fit_glb(glb_path: str, dx: float, h: float, dy: float,
                      cx: float, cy: float):
    """Echtes Modell laden und in die Footprint-Box skalieren. None bei Fehler."""
    p = Path(glb_path)
    if not p.exists():
        return None
    try:
        import trimesh
        loaded = trimesh.load(str(p), force="mesh")
        if loaded is None or not hasattr(loaded, "bounds") or loaded.bounds is None:
            return None
        (bx0, by0, bz0), (bx1, by1, bz1) = loaded.bounds
        sx = dx / max(1e-6, bx1 - bx0)
        sy = h / max(1e-6, by1 - by0)
        sz = dy / max(1e-6, bz1 - bz0)
        loaded.apply_scale((sx, sy, sz))
        (nx0, ny0, nz0), _ = loaded.bounds
        loaded.apply_translation((cx - (nx0 + 0.5 * dx), -ny0, cy - (nz0 + 0.5 * dy)))
        return loaded
    except Exception:  # noqa: BLE001
        return None


def furniture_nodes(
    placed: Sequence[PlacedFurniture],
) -> List[Tuple[str, object]]:
    """(node_name, mesh) fuer alle platzierten Moebel-Items."""
    nodes: List[Tuple[str, object]] = []
    for pf in placed:
        for item in pf.items:
            mesh = build_item_mesh(item)
            if mesh is None or not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
                continue
            node_name = f"furniture/{item.catalog_id}/{pf.bbox_id}/{item.order}"
            nodes.append((node_name, mesh))
    return nodes


def append_furniture_to_scene(scene, placed: Sequence[PlacedFurniture]) -> int:
    """Moebel-Knoten an eine bestehende trimesh.Scene anhaengen. Gibt die
    Anzahl hinzugefuegter Knoten zurueck. Bestehende Geometrie bleibt unberuehrt
    (walls-immutable-Geist: wir lesen Waende nur, wir fuegen nur hinzu)."""
    added = 0
    for node_name, mesh in furniture_nodes(placed):
        try:
            scene.add_geometry(mesh, node_name=node_name)
            added += 1
        except Exception:  # noqa: BLE001
            continue
    return added
