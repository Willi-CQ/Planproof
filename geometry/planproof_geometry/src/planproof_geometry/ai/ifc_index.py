"""Leichtgewichtiger IFC -> Komponenten-Parser (K5, ohne Geometrie-Engine).

Liest IfcProduct-Elemente (Waende/Tueren/Fenster/Traeger/Stuetzen/Treppen/
Decken/Daecher/Raeume) mit Typ -> class_code und — soweit im Modell hinterlegt —
Basis-Mengen (Laenge/Breite/Hoehe/Flaeche/Volumen) aus den IfcElementQuantity-Sets,
normalisiert ueber den IFC-Laengeneinheiten-Scale auf Meter.

Bewusst KEINE ``ifcopenshell.geom``-Nutzung (kein OpenCascade-Overhead) — nur
Metadaten + Quantities, damit der API-Indexer IFC ohne schwere Geometrie einlesen
kann. Damit sieht der Komponenten-Tree (und DIN-Audit/Inventar) IFC-Modelle, statt
sie wie bisher als "0 Komponenten" zu verwerfen.

Downstream: ``planproof_api.services.model_indexing`` (Branch ``fmt == 'ifc'``).
``ifcopenshell`` wird bewusst erst IN der Funktion importiert, damit der Modul-Import
ohne die (schwere) Dependency gelingt.
"""

from __future__ import annotations

from typing import Any

# IfcType -> unser class_code (deckungsgleich mit model_indexing._heuristic_class).
# by_type() liefert Subtypen mit (z.B. IfcWallStandardCase unter IfcWall) — die
# Deduplizierung ueber die Element-ID verhindert Doppel-Eintraege.
_IFC_TYPE_TO_CLASS: tuple[tuple[str, str], ...] = (
    ("IfcWall", "wall"),
    ("IfcDoor", "door"),
    ("IfcWindow", "window"),
    ("IfcBeam", "beam"),
    ("IfcColumn", "column"),
    ("IfcStair", "stairs"),
    ("IfcSlab", "floor"),
    ("IfcRoof", "roof"),
    ("IfcCovering", "ceiling"),
    ("IfcSpace", "room"),
)


def _extract_quantities(ifc_element: Any, el: Any, scale: float) -> dict[str, Any]:
    """Basis-Mengen aus IfcElementQuantity-Sets (in Meter skaliert)."""
    dims: dict[str, Any] = {
        "length_m": None,
        "width_m": None,
        "height_m": None,
        "area_m2": None,
        "volume_m3": None,
    }
    try:
        qsets = ifc_element.get_psets(el, qtos_only=True)
    except Exception:
        return dims
    flat: dict[str, float] = {}
    for qs in (qsets or {}).values():
        if not isinstance(qs, dict):
            continue
        for k, v in qs.items():
            if k != "id" and isinstance(v, (int, float)) and not isinstance(v, bool):
                flat[str(k).lower()] = float(v)

    def pick(*names: str) -> float | None:
        for n in names:
            if n in flat:
                return flat[n]
        return None

    length = pick("length", "netlength", "grosslength")
    width = pick("width", "netwidth", "grosswidth", "thickness")
    height = pick("height", "netheight", "grossheight", "depth")
    area = pick(
        "netarea", "grossarea", "netsidearea", "grosssidearea",
        "netfootprintarea", "grossfootprintarea", "area",
    )
    volume = pick("netvolume", "grossvolume", "volume")

    if length is not None:
        dims["length_m"] = round(length * scale, 4)
    if width is not None:
        dims["width_m"] = round(width * scale, 4)
    if height is not None:
        dims["height_m"] = round(height * scale, 4)
    if area is not None:
        dims["area_m2"] = round(area * scale * scale, 4)
    if volume is not None:
        dims["volume_m3"] = round(volume * scale**3, 6)
    return dims


def _first_material(ifc_element: Any, el: Any) -> str | None:
    try:
        mat = ifc_element.get_material(el)
    except Exception:
        return None
    if mat is None:
        return None
    name = getattr(mat, "Name", None)
    return str(name)[:255] if name else None


def _material_names(ifc_element: Any, el: Any) -> list[str]:
    """Flacht das Material (auch Layer-/Constituent-Sets) auf Namen ab."""
    try:
        mat = ifc_element.get_material(el)
    except Exception:
        return []
    if mat is None:
        return []
    names: list[str] = []

    def _collect(obj: Any) -> None:
        if obj is None:
            return
        # Direktes IfcMaterial
        name = getattr(obj, "Name", None)
        if name and getattr(obj, "is_a", lambda: "")() == "IfcMaterial":
            names.append(str(name))
            return
        # Layer-Set / Constituent-Set / MaterialList
        for attr in ("MaterialLayers", "MaterialConstituents", "Materials", "ForLayerSet"):
            sub = getattr(obj, attr, None)
            if sub is None:
                continue
            items = sub if isinstance(sub, (list, tuple)) else [sub]
            for it in items:
                inner = getattr(it, "Material", None)
                _collect(inner if inner is not None else it)
        # LayerSetUsage -> ForLayerSet
        fls = getattr(obj, "ForLayerSet", None)
        if fls is not None:
            _collect(fls)

    _collect(mat)
    # Dedupe unter Erhalt der Reihenfolge.
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _classifications(el: Any) -> list[dict[str, str | None]]:
    """Klassifikations-Referenzen (System + Kennung) eines Elements."""
    try:
        from ifcopenshell.util import classification as ifc_classification
    except Exception:
        return []
    out: list[dict[str, str | None]] = []
    try:
        refs = ifc_classification.get_references(el)
    except Exception:
        return []
    for ref in refs or []:
        value = getattr(ref, "Identification", None) or getattr(ref, "ItemReference", None)
        system: str | None = None
        try:
            src = ifc_classification.get_classification(ref)
            system = getattr(src, "Name", None)
        except Exception:
            system = getattr(ref, "ReferencedSource", None)
            system = getattr(system, "Name", None) if system is not None else None
        out.append({"system": str(system) if system else None, "value": str(value) if value else None})
    return out


def _all_psets(ifc_element: Any, el: Any) -> dict[str, dict[str, Any]]:
    """Alle Property-Sets (ohne Quantities) als {pset: {prop: value}}."""
    try:
        psets = ifc_element.get_psets(el, psets_only=True)
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for pset_name, entries in (psets or {}).items():
        if not isinstance(entries, dict):
            continue
        clean = {k: v for k, v in entries.items() if k != "id"}
        if clean:
            out[str(pset_name)] = clean
    return out


def parse_ifc_for_ids(blob_bytes: bytes, *, max_elements: int = 10000) -> list[dict[str, Any]]:
    """Parst eine IFC-Datei in reiche Element-Dicts fuer die IDS-Pruefung.

    Schema je Element (siehe ``planproof_geometry.standards.ids.spec``):
    ``ifc_class`` (UPPER), ``predefined_type``, ``global_id``, ``name``,
    ``attributes`` (Name/Tag/Description/ObjectType), ``properties``
    ({pset: {prop: value}}), ``materials`` ([Name]), ``classifications``
    ([{system, value}]). Braucht ``ifcopenshell`` (Worker-Env) — der Aufrufer
    faengt ImportError ab und degradiert.
    """
    import os
    import tempfile

    import ifcopenshell
    from ifcopenshell.util import element as ifc_element

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as tf:
            tf.write(blob_bytes)
            tmp_path = tf.name
        model = ifcopenshell.open(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    # Physische Bauteile + raeumliche Elemente (Raeume) — beide sind IDS-relevant.
    for base_type in ("IfcElement", "IfcSpatialElement", "IfcSpatialStructureElement"):
        try:
            elements = model.by_type(base_type)
        except Exception:
            continue
        for el in elements:
            try:
                eid = el.id()
            except Exception:
                continue
            if eid in seen:
                continue
            seen.add(eid)
            try:
                ifc_class = el.is_a().upper()
            except Exception:
                continue
            try:
                predefined = ifc_element.get_predefined_type(el)
            except Exception:
                predefined = getattr(el, "PredefinedType", None)
            attrs = {
                "Name": getattr(el, "Name", None),
                "Description": getattr(el, "Description", None),
                "Tag": getattr(el, "Tag", None),
                "ObjectType": getattr(el, "ObjectType", None),
            }
            out.append(
                {
                    "global_id": getattr(el, "GlobalId", None),
                    "ifc_class": ifc_class,
                    "predefined_type": str(predefined) if predefined else None,
                    "name": getattr(el, "Name", None),
                    "attributes": {k: v for k, v in attrs.items() if v is not None},
                    "properties": _all_psets(ifc_element, el),
                    "materials": _material_names(ifc_element, el),
                    "classifications": _classifications(el),
                }
            )
            if len(out) >= max_elements:
                return out
    return out


def parse_ifc_components(blob_bytes: bytes, *, max_components: int = 5000) -> list[dict[str, Any]]:
    """Parst eine IFC-Datei (STEP/IFC-SPF) in Komponenten-Dicts.

    Jedes Dict traegt dieselben Keys, die ``model_indexing`` fuer GLB/GLTF erzeugt
    (node_path/name/class_code/material_name/length_m/width_m/height_m/area_m2/
    volume_m3/center_*), damit der Indexer sie 1:1 als ProjectModelComponent
    persistieren kann. ``center_*`` bleiben None (keine Geometrie-Engine).

    Raises bei fehlendem ifcopenshell oder Parse-Fehler — der Aufrufer faengt das
    ab und degradiert auf einen leeren Index.
    """
    import os
    import tempfile

    import ifcopenshell
    from ifcopenshell.util import element as ifc_element

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as tf:
            tf.write(blob_bytes)
            tmp_path = tf.name
        model = ifcopenshell.open(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    scale = 1.0
    try:
        from ifcopenshell.util import unit as ifc_unit

        scale = float(ifc_unit.calculate_unit_scale(model)) or 1.0
    except Exception:
        scale = 1.0

    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for ifc_type, cls in _IFC_TYPE_TO_CLASS:
        try:
            elements = model.by_type(ifc_type)
        except Exception:
            continue
        for el in elements:
            try:
                eid = el.id()
            except Exception:
                continue
            if eid in seen:
                continue
            seen.add(eid)
            name = getattr(el, "Name", None) or f"{ifc_type}_{eid}"
            dims = _extract_quantities(ifc_element, el, scale)
            out.append(
                {
                    "node_path": f"{ifc_type}/{name}"[:1024],
                    "name": str(name)[:255],
                    "mesh_index": None,
                    "material_name": _first_material(ifc_element, el),
                    "class_code": cls,
                    "center_x": None,
                    "center_y": None,
                    "center_z": None,
                    **dims,
                }
            )
            if len(out) >= max_components:
                return out
    return out
