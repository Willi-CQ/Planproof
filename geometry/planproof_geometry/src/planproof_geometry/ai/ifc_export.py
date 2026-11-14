"""GLB → IFC4 Export (Sprint B).

Liest die nach Bauteilen organisierte GLB der `perfect_bim`-Pipeline und
schreibt ein IFC4-File mit dem Mindest-Spatial-Tree:

    IfcProject
      └─ IfcSite
           └─ IfcBuilding
                └─ IfcBuildingStorey
                     ├─ IfcWall(s)        (klassifiziert)
                     ├─ IfcDoor(s)
                     ├─ IfcWindow(s)
                     └─ IfcBuildingElementProxy(s)  (Fallback)

Geometrie ist bewusst pragmatisch — bbox-extrudierte Quader pro Bauteil.
Die GLB ist Y-up (glTF/three.js), IFC ist Z-up: die bbox wird per
(x, y, z) -> (x, -z, y) gedreht, damit Wände im IFC-Viewer stehen statt zu
liegen. Jede erzeugte IfcShapeRepresentation wird via
``geometry.assign_representation`` an ihr Produkt gebunden — sonst steht das
Element ohne sichtbare Geometrie im Baum. Das genügt um in Revit/AutoCAD/
ArchiCAD einen geometrischen Aufschlag zu sehen + die Klassifikation zu nutzen.
Mengen (Länge/Breite/Höhe/Fläche/Volumen aus der bbox) sind als
IfcElementQuantity + Pset hinterlegt, damit die Massenermittlung in der
BIM-Software stimmt.

Failure-Modi:
  - GLB nicht parsebar → IfcExportError mit klarer Meldung
  - ifcopenshell nicht installiert → IfcExportError("ifcopenshell missing")
  - Keine klassifizierbaren Nodes → leeres aber valides IFC mit nur dem
    Spatial-Tree (so weiß der Architekt: hier kam kein Bauteil rüber)
"""
from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class IfcExportError(Exception):
    pass


# GLB-Layout-Konstanten (gleiche wie model_indexing.py — duplicate weil
# wir hier im Core-Package leben und nicht auf das API-Service zugreifen.)
_GLB_MAGIC = 0x46546C67
_GLB_CHUNK_JSON = 0x4E4F534A


def _parse_glb_json(blob: bytes) -> dict[str, Any]:
    if len(blob) < 20:
        raise IfcExportError("GLB zu kurz für Header")
    magic, version, total_length = struct.unpack_from("<III", blob, 0)
    if magic != _GLB_MAGIC:
        raise IfcExportError(f"Kein GLB-Magic (0x{magic:08x})")
    if version != 2:
        raise IfcExportError(f"GLB-Version {version} nicht unterstützt (nur 2)")
    chunk_len, chunk_type = struct.unpack_from("<II", blob, 12)
    if chunk_type != _GLB_CHUNK_JSON:
        raise IfcExportError("Erster GLB-Chunk ist kein JSON-Chunk")
    json_bytes = blob[20:20 + chunk_len]
    try:
        return json.loads(json_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IfcExportError(f"GLB-JSON-Parse-Fehler: {exc}") from exc


def _accessor_min_max(gltf: dict[str, Any], idx: int) -> tuple[list[float] | None, list[float] | None]:
    accs = gltf.get("accessors") or []
    if idx < 0 or idx >= len(accs):
        return None, None
    a = accs[idx]
    return a.get("min"), a.get("max")


def _mesh_bbox(gltf: dict[str, Any], mesh_idx: int) -> tuple[list[float], list[float]] | None:
    """Bounding-Box in Local-Coords (vor Node-Transform)."""
    meshes = gltf.get("meshes") or []
    if mesh_idx < 0 or mesh_idx >= len(meshes):
        return None
    bb_min: list[float] | None = None
    bb_max: list[float] | None = None
    for prim in meshes[mesh_idx].get("primitives") or []:
        attrs = prim.get("attributes") or {}
        pos = attrs.get("POSITION")
        if pos is None:
            continue
        mn, mx = _accessor_min_max(gltf, pos)
        if not mn or not mx or len(mn) < 3 or len(mx) < 3:
            continue
        if bb_min is None:
            bb_min = list(mn[:3])
            bb_max = list(mx[:3])
        else:
            assert bb_max is not None
            for i in range(3):
                bb_min[i] = min(bb_min[i], mn[i])
                bb_max[i] = max(bb_max[i], mx[i])
    if bb_min is None or bb_max is None:
        return None
    return bb_min, bb_max


_CLASS_KEYWORDS = (
    # Reihenfolge: spezifischer vor allgemeiner — "railing" muss vor "stair"
    # geprueft werden, weil Geländer-Knoten als "stair/.../railing/N" heissen.
    ("railing", "railing"),
    ("wall_exterior", "wall_exterior"),
    ("wall_interior", "wall_interior"),
    ("wand_aussen", "wall_exterior"),
    ("wand_innen", "wall_interior"),
    ("wall", "wall_interior"),
    ("wand", "wall_interior"),
    ("door", "door"),
    ("tuer", "door"),
    ("tur", "door"),
    ("window", "window"),
    ("fenster", "window"),
    ("column", "column"),
    ("stuetze", "column"),
    ("beam", "beam"),
    ("traeger", "beam"),
    ("stair", "stair"),
    ("treppe", "stair"),
)


def _heuristic_class(name: str) -> str:
    nm = (name or "").lower()
    for kw, cls in _CLASS_KEYWORDS:
        if kw in nm:
            return cls
    return "proxy"


def _walk_nodes(gltf: dict[str, Any]) -> list[dict[str, Any]]:
    """Walks the scene-graph and emits a flat list of bbox-positioned nodes.

    Each entry:
        {name, class, center: [x,y,z], dims: [dx,dy,dz]}
    """
    nodes = gltf.get("nodes") or []
    scenes = gltf.get("scenes") or []
    if not nodes or not scenes:
        return []
    scene_idx = gltf.get("scene", 0)
    if scene_idx >= len(scenes):
        scene_idx = 0
    roots = scenes[scene_idx].get("nodes") or []
    out: list[dict[str, Any]] = []

    def visit(idx: int, parent_translation: tuple[float, float, float]) -> None:
        if idx < 0 or idx >= len(nodes):
            return
        node = nodes[idx]
        name = node.get("name") or f"node_{idx}"
        t = node.get("translation") or [0.0, 0.0, 0.0]
        world_t = (
            parent_translation[0] + float(t[0]),
            parent_translation[1] + float(t[1]),
            parent_translation[2] + float(t[2]),
        )
        mesh_idx = node.get("mesh")
        if mesh_idx is not None:
            bb = _mesh_bbox(gltf, mesh_idx)
            if bb is not None:
                mn, mx = bb
                dx = max(1e-3, mx[0] - mn[0])
                dy = max(1e-3, mx[1] - mn[1])
                dz = max(1e-3, mx[2] - mn[2])
                cx = world_t[0] + (mn[0] + mx[0]) * 0.5
                cy = world_t[1] + (mn[1] + mx[1]) * 0.5
                cz = world_t[2] + (mn[2] + mx[2]) * 0.5
                out.append({
                    "name": name,
                    "class": _heuristic_class(name),
                    "center": [cx, cy, cz],
                    "dims": [dx, dy, dz],
                })
        for child in node.get("children") or []:
            visit(int(child), world_t)

    for r in roots:
        visit(int(r), (0.0, 0.0, 0.0))
    return out


# Minimale Box-Kantenlänge (m), damit das Extrusions-Profil nie degeneriert
# (Null-Fläche -> add_wall_representation würde scheitern). 2 cm ist dünn
# genug, um die echte Geometrie kaum zu verfälschen, aber sichtbar.
_MIN_EXTENT_M = 0.02


# IFC4 class mapping per detected class
_IFC_CLASS_MAP = {
    "wall_exterior": "IfcWall",
    "wall_interior": "IfcWall",
    "door": "IfcDoor",
    "window": "IfcWindow",
    "column": "IfcColumn",
    "beam": "IfcBeam",
    "stair": "IfcStair",
    "railing": "IfcRailing",
    "proxy": "IfcBuildingElementProxy",
}


def export_glb_as_ifc(
    glb_path: Path,
    out_ifc_path: Path,
    *,
    project_name: str = "Planproof Plan→BIM",
    storey_name: str = "EG",
) -> dict[str, Any]:
    """Write an IFC4 file next to the given GLB.

    Returns a small summary dict ``{n_nodes, n_classified, n_walls, n_doors,
    n_windows, n_proxy}`` so callers can log / surface stats.

    Raises ``IfcExportError`` on missing ifcopenshell or parse failures.
    """
    try:
        import ifcopenshell  # noqa: PLC0415
        import ifcopenshell.api  # noqa: PLC0415
    except ImportError as exc:
        raise IfcExportError(
            "ifcopenshell ist nicht installiert — IFC-Export nicht verfügbar"
        ) from exc
    import numpy as np  # noqa: PLC0415

    glb_bytes = glb_path.read_bytes()
    gltf = _parse_glb_json(glb_bytes)
    elements = _walk_nodes(gltf)

    # ifcopenshell 0.8 renamed kwarg: schema= -> version=
    model = ifcopenshell.api.run("project.create_file", version="IFC4")
    project = ifcopenshell.api.run(
        "root.create_entity", model, ifc_class="IfcProject", name=project_name
    )
    # SI-Einheiten OHNE Prefix = METRE / SQUARE_METRE / CUBIC_METRE. Der
    # Default von unit.assign_unit wäre MILLIMETRE — dann liefen Placement,
    # Geometrie und Mengen in unterschiedlichen Maßstäben auseinander. Die GLB
    # liegt in Metern vor; wir halten das ganze IFC konsistent in Metern, damit
    # edit_object_placement (m -> Datei-Einheit), add_wall_representation und die
    # Qto-Mengen ohne Skalierungs-Salat übereinstimmen.
    length_unit = ifcopenshell.api.run(
        "unit.add_si_unit", model, unit_type="LENGTHUNIT", prefix=None
    )
    area_unit = ifcopenshell.api.run(
        "unit.add_si_unit", model, unit_type="AREAUNIT", prefix=None
    )
    volume_unit = ifcopenshell.api.run(
        "unit.add_si_unit", model, unit_type="VOLUMEUNIT", prefix=None
    )
    ifcopenshell.api.run(
        "unit.assign_unit", model, units=[length_unit, area_unit, volume_unit]
    )

    geom_context = ifcopenshell.api.run("context.add_context", model, context_type="Model")
    body_context = ifcopenshell.api.run(
        "context.add_context", model,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=geom_context,
    )

    site = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.run(
        "root.create_entity", model, ifc_class="IfcBuilding", name="Gebäude"
    )
    storey = ifcopenshell.api.run(
        "root.create_entity", model, ifc_class="IfcBuildingStorey", name=storey_name
    )
    ifcopenshell.api.run("aggregate.assign_object", model, products=[site], relating_object=project)
    ifcopenshell.api.run("aggregate.assign_object", model, products=[building], relating_object=site)
    ifcopenshell.api.run("aggregate.assign_object", model, products=[storey], relating_object=building)

    counts = {"wall": 0, "door": 0, "window": 0, "column": 0, "beam": 0,
              "stair": 0, "railing": 0, "proxy": 0}
    n_classified = 0
    n_representations = 0
    n_with_psets = 0

    for elem in elements:
        cls_key = elem["class"]
        ifc_cls = _IFC_CLASS_MAP.get(cls_key, "IfcBuildingElementProxy")
        bucket = (
            "wall" if "Wall" in ifc_cls
            else "door" if "Door" in ifc_cls
            else "window" if "Window" in ifc_cls
            else "column" if "Column" in ifc_cls
            else "beam" if "Beam" in ifc_cls
            else "stair" if "Stair" in ifc_cls
            else "railing" if "Railing" in ifc_cls
            else "proxy"
        )
        counts[bucket] += 1
        if cls_key != "proxy":
            n_classified += 1

        product = ifcopenshell.api.run(
            "root.create_entity", model, ifc_class=ifc_cls, name=elem["name"]
        )

        cx, cy, cz = elem["center"]
        dx, dy, dz = elem["dims"]
        # Y-up (glTF) -> Z-up (IFC): (x, y, z) -> (x, -z, y). Die Box von
        # add_wall_representation ist eck-verankert (lokal [0..length] x
        # [0..thickness] x [0..height], Extrusion entlang +Z), NICHT zentriert.
        # Wir setzen das Placement deshalb auf die bbox-Min-Ecke in
        # IFC-Koordinaten, dann deckt die Box exakt die transformierte glTF-Bbox
        # ab (X = Länge, Y = Tiefe/Dicke, Z = Höhe).
        lx = max(_MIN_EXTENT_M, dx)   # IFC X  <- glTF dx (Länge)
        ly = max(_MIN_EXTENT_M, dz)   # IFC Y  <- glTF dz (Tiefe/Dicke)
        lz = max(_MIN_EXTENT_M, dy)   # IFC Z  <- glTF dy (Höhe, up)
        ifc_center = (cx, -cz, cy)
        corner = (
            ifc_center[0] - lx / 2.0,
            ifc_center[1] - ly / 2.0,
            ifc_center[2] - lz / 2.0,
        )
        try:
            matrix = np.eye(4)
            matrix[:3, 3] = corner
            ifcopenshell.api.run(
                "geometry.edit_object_placement", model, product=product, matrix=matrix
            )
        except Exception:
            pass  # Placement-Failure ist nicht-tödlich, Element existiert trotzdem

        # Body-Repräsentation: bbox-Box via add_wall_representation (baut den
        # generischen Extrusions-Solid für JEDE IfcClass). Der Rückgabewert MUSS
        # via assign_representation an das Produkt gebunden werden — sonst steht
        # das Element ohne Geometrie im Baum (genau der Bug, den dieser Export
        # vorher hatte: Representation erzeugt, aber nie zugewiesen).
        try:
            rep = ifcopenshell.api.run(
                "geometry.add_wall_representation",
                model,
                context=body_context,
                length=lx,
                height=lz,
                thickness=ly,
            )
            ifcopenshell.api.run(
                "geometry.assign_representation",
                model, product=product, representation=rep,
            )
            n_representations += 1
        except Exception as exc:
            logger.debug("ifc_export: representation failed for %s: %s", elem["name"], exc)

        # Mengen (Massenermittlung) als IfcElementQuantity + beschreibendes Pset.
        # bbox-Näherung: Länge/Breite = horizontale Extents, Höhe = vertikaler
        # Extent. Werte in Metern (Datei-Einheit ist METRE).
        try:
            horiz = sorted((dx, dz), reverse=True)
            qto = ifcopenshell.api.run(
                "pset.add_qto", model, product=product,
                name="Qto_PlanproofBaseQuantities",
            )
            ifcopenshell.api.run(
                "pset.edit_qto", model, qto=qto, properties={
                    "Length": round(float(horiz[0]), 4),
                    "Width": round(float(horiz[1]), 4),
                    "Height": round(float(dy), 4),
                    "GrossFootprintArea": round(float(dx * dz), 4),
                    "GrossVolume": round(float(dx * dy * dz), 4),
                },
            )
            pset = ifcopenshell.api.run(
                "pset.add_pset", model, product=product, name="Pset_Planproof",
            )
            ifcopenshell.api.run(
                "pset.edit_pset", model, pset=pset, properties={
                    "Reference": cls_key,
                    "IfcClass": ifc_cls,
                    "GeometrySource": "perfect_bim GLB bounding-box",
                },
            )
            n_with_psets += 1
        except Exception as exc:
            logger.debug("ifc_export: pset failed for %s: %s", elem["name"], exc)

        ifcopenshell.api.run(
            "spatial.assign_container", model, products=[product], relating_structure=storey
        )

    out_ifc_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(out_ifc_path))

    summary = {
        "n_nodes": len(elements),
        "n_classified": n_classified,
        "n_representations": n_representations,
        "n_with_psets": n_with_psets,
        "n_walls": counts["wall"],
        "n_doors": counts["door"],
        "n_windows": counts["window"],
        "n_columns": counts["column"],
        "n_beams": counts["beam"],
        "n_stairs": counts["stair"],
        "n_railings": counts["railing"],
        "n_proxy": counts["proxy"],
        "ifc_path": str(out_ifc_path),
    }
    logger.info("ifc_export: %s", summary)
    return summary
