"""IFC4-Export aus SceneModel + Template-Assignments.

Baut aus der Pipeline-Output-Struktur (``SceneModel.components`` +
``template_assignments`` + ``templates_by_key``) eine vollständige IFC4-Datei
mit korrekter Spatial-Structure (Project → Site → Building → Storey),
Material-Layer-Sets aus den Template-Schichten und IfcPropertySets aus den
vorberechneten ``TemplateSpec.ifc_psets``.

Design-Entscheidungen:
- **Schema-Version**: default ``IFC4``, ``IFC2x3`` optional (String via Argument)
- **Koordinaten-Transformation**: Scene Y-up → IFC Z-up (Swap y↔z in Positionen)
- **Multi-Storey**: _detect_storeys() clustert Boden-Y der Komponenten mit
  Gap-Schwellwert 1.5m - bei Einfamilienhäusern 1 Storey, MFH 2-3 Storeys
- **Missing Templates**: Element wird trotzdem als generische IFC-Entity
  exportiert (ohne Material-Layer-Set)
- **Psets**: 1:1 aus ``TemplateSpec.ifc_psets`` auf ``IfcPropertySet`` via
  ``IfcPropertySingleValue``-Einträge

Abhängigkeit: ``ifcopenshell>=0.8``. Wenn nicht verfügbar, raised der Konstruktor
``ImportError`` - der Pipeline-Step fängt das ab und skippt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from planproof_cad.catalog.schemas import LayerSpec, TemplateSpec

logger = logging.getLogger(__name__)

IfcSchemaVersion = Literal["IFC4", "IFC2X3"]


# ───────────────────────── Data classes ─────────────────────────


@dataclass
class ExportedElement:
    """Ergebnis pro Component: welche IFC-Entity wurde erzeugt."""

    component_id: str
    ifc_type: str
    global_id: str
    material_set_attached: bool


@dataclass
class IfcExportResult:
    path: str
    schema: str
    element_count: int
    exported: list[ExportedElement]
    warnings: list[str]


# ───────────────────────── Helpers ─────────────────────────


def _scene_to_ifc_coords(point: tuple[float, float, float]) -> tuple[float, float, float]:
    """Scene (X, Y=up, Z) → IFC (X, -Z, Y). Klassisches Y-up→Z-up Swap."""
    x, y, z = point
    return (float(x), float(-z), float(y))


def _value_to_ifc_property(ifc_file, name: str, value: Any):
    """Erzeugt IfcPropertySingleValue für ein Pset-Feld.

    Typ-Inferenz nach Python-Type:
    - bool → IfcBoolean
    - int → IfcInteger
    - float → IfcReal
    - str → IfcLabel (kurze) oder IfcText (lange)
    """
    if isinstance(value, bool):
        nom = ifc_file.create_entity("IfcBoolean", wrappedValue=bool(value))
    elif isinstance(value, int):
        nom = ifc_file.create_entity("IfcInteger", wrappedValue=int(value))
    elif isinstance(value, float):
        nom = ifc_file.create_entity("IfcReal", wrappedValue=float(value))
    elif value is None:
        return None
    else:
        s = str(value)
        if len(s) > 255:
            nom = ifc_file.create_entity("IfcText", wrappedValue=s)
        else:
            nom = ifc_file.create_entity("IfcLabel", wrappedValue=s)
    return ifc_file.create_entity(
        "IfcPropertySingleValue",
        Name=name,
        NominalValue=nom,
    )


def _detect_storeys(
    components: list,
    min_gap_m: float = 1.5,
) -> list[tuple[float, int]]:
    """Cluster components by floor Y into building storeys.

    Returns a sorted list of (scene_y_elevation, storey_index) tuples,
    one entry per detected storey level (bottom-to-top).

    Scene is Y-up; IFC-Z = scene-Y, so the returned elevations are scene-Y
    values that the caller must transform via _scene_to_ifc_coords when
    setting IfcBuildingStorey.Elevation.
    """
    if not components:
        return [(0.0, 0)]

    bottoms: list[float] = []
    for comp in components:
        cy = float(comp.center[1]) if hasattr(comp, "center") else 0.0
        dims = getattr(comp, "dims", None)
        hy = (float(dims[1]) / 2.0) if (dims and len(dims) > 1 and dims[1] is not None) else 0.0
        bottoms.append(cy - hy)

    unique_sorted = sorted(set(round(b, 2) for b in bottoms))

    clusters: list[float] = [unique_sorted[0]]
    for b in unique_sorted[1:]:
        if b - clusters[-1] > min_gap_m:
            clusters.append(b)

    return [(elev, idx) for idx, elev in enumerate(clusters)]


_STOREY_LABELS = [
    "Erdgeschoss",
    "1. Obergeschoss",
    "2. Obergeschoss",
    "3. Obergeschoss",
    "4. Obergeschoss",
    "Dachgeschoss",
]


# ───────────────────────── Main Exporter ─────────────────────────


class IfcExporter:
    """Scene + Assignments → IFC-Datei.

    Nutzung::

        exporter = IfcExporter(schema="IFC4", project_name="Foo")
        result = exporter.export(
            scene=state.scene,
            template_assignments=state.template_assignments,
            templates_by_key=state._catalog_templates_cache,
            output_path="/out/plan.ifc",
        )
    """

    def __init__(
        self,
        *,
        schema: IfcSchemaVersion = "IFC4",
        project_name: str = "Planproof Plan-to-BIM",
        application_name: str = "Planproof",
        application_version: str = "0.1",
        rotate_walls_to_axis: bool = False,
    ) -> None:
        try:
            import ifcopenshell  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "IfcExporter benötigt ifcopenshell - installiere via `pip install ifcopenshell`."
            ) from exc
        self._schema = schema
        self._project_name = project_name
        self._app_name = application_name
        self._app_version = application_version
        # Opt-in: orientiert Wand-/Tuer-/Fenster-Boxen entlang ihrer Plan-Achse
        # (RefDirection aus segment_a/segment_b). Default aus -> bestehende
        # achsparallele Geometrie unveraendert (Raster-Pfad). Der DWG-Pfad
        # liefert exakte Achsen und schaltet das ein -> diagonale Waende stehen
        # in Revit/ArchiCAD korrekt statt achsparallel.
        self._rotate_walls = bool(rotate_walls_to_axis)
        self._warnings: list[str] = []

    # ─── Helpers für Setup ───

    def _new_guid(self) -> str:
        import ifcopenshell

        return ifcopenshell.guid.new()

    def _make_owner_history(self, ifc_file):
        """Minimale OwnerHistory - IFC verlangt's, aber wir haben keinen echten User."""
        person = ifc_file.create_entity("IfcPerson", FamilyName="Pipeline", GivenName="Planproof")
        organization = ifc_file.create_entity("IfcOrganization", Name="Planproof")
        person_org = ifc_file.create_entity(
            "IfcPersonAndOrganization",
            ThePerson=person,
            TheOrganization=organization,
        )
        application = ifc_file.create_entity(
            "IfcApplication",
            ApplicationDeveloper=organization,
            Version=self._app_version,
            ApplicationFullName=self._app_name,
            ApplicationIdentifier=self._app_name,
        )
        import time

        return ifc_file.create_entity(
            "IfcOwnerHistory",
            OwningUser=person_org,
            OwningApplication=application,
            ChangeAction="NOCHANGE",
            CreationDate=int(time.time()),
        )

    def _make_axis_placement_origin(self, ifc_file):
        """Standard-Ursprung (0,0,0) in IFC-Koordinaten."""
        p = ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
        return ifc_file.create_entity("IfcAxis2Placement3D", Location=p)

    def _make_context(self, ifc_file, origin_placement):
        """Erzeugt 3D-Model-Context (IfcGeometricRepresentationContext)."""
        ctx = ifc_file.create_entity(
            "IfcGeometricRepresentationContext",
            ContextType="Model",
            CoordinateSpaceDimension=3,
            Precision=1e-5,
            WorldCoordinateSystem=origin_placement,
        )
        return ctx

    def _make_units(self, ifc_file):
        """SI-Einheiten: METRE, SECOND, RADIAN."""
        length = ifc_file.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE")
        area = ifc_file.create_entity("IfcSIUnit", UnitType="AREAUNIT", Name="SQUARE_METRE")
        volume = ifc_file.create_entity("IfcSIUnit", UnitType="VOLUMEUNIT", Name="CUBIC_METRE")
        angle = ifc_file.create_entity("IfcSIUnit", UnitType="PLANEANGLEUNIT", Name="RADIAN")
        time_unit = ifc_file.create_entity("IfcSIUnit", UnitType="TIMEUNIT", Name="SECOND")
        return ifc_file.create_entity(
            "IfcUnitAssignment",
            Units=[length, area, volume, angle, time_unit],
        )

    def _make_spatial_structure(self, ifc_file, owner_history, context, units):
        """Baut IfcProject → IfcSite → IfcBuilding. Returns (building_placement, building)."""
        origin = self._make_axis_placement_origin(ifc_file)
        project = ifc_file.create_entity(
            "IfcProject",
            GlobalId=self._new_guid(),
            OwnerHistory=owner_history,
            Name=self._project_name,
            RepresentationContexts=[context],
            UnitsInContext=units,
        )
        site_placement = ifc_file.create_entity("IfcLocalPlacement", RelativePlacement=origin)
        site = ifc_file.create_entity(
            "IfcSite",
            GlobalId=self._new_guid(),
            OwnerHistory=owner_history,
            Name="Default Site",
            ObjectPlacement=site_placement,
            CompositionType="ELEMENT",
        )
        building_placement = ifc_file.create_entity(
            "IfcLocalPlacement",
            PlacementRelTo=site_placement,
            RelativePlacement=origin,
        )
        building = ifc_file.create_entity(
            "IfcBuilding",
            GlobalId=self._new_guid(),
            OwnerHistory=owner_history,
            Name="Default Building",
            ObjectPlacement=building_placement,
            CompositionType="ELEMENT",
        )
        ifc_file.create_entity(
            "IfcRelAggregates",
            GlobalId=self._new_guid(),
            OwnerHistory=owner_history,
            RelatingObject=project,
            RelatedObjects=[site],
        )
        ifc_file.create_entity(
            "IfcRelAggregates",
            GlobalId=self._new_guid(),
            OwnerHistory=owner_history,
            RelatingObject=site,
            RelatedObjects=[building],
        )
        return building_placement, building

    def _make_storeys(
        self,
        ifc_file,
        owner_history,
        building_placement,
        building,
        storey_specs: list[tuple[float, int]],
    ) -> list[tuple[float, Any, Any]]:
        """Create IfcBuildingStoreys and aggregate under building.

        Returns list of (scene_y_elevation, storey_placement, storey).
        """
        origin = self._make_axis_placement_origin(ifc_file)
        storey_list = []
        for elev, idx in storey_specs:
            label = _STOREY_LABELS[idx] if idx < len(_STOREY_LABELS) else f"Geschoss {idx}"
            ifc_z = float(elev)  # scene-Y = IFC-Z; elevation is relative to building
            storey_placement = ifc_file.create_entity(
                "IfcLocalPlacement",
                PlacementRelTo=building_placement,
                RelativePlacement=origin,
            )
            storey = ifc_file.create_entity(
                "IfcBuildingStorey",
                GlobalId=self._new_guid(),
                OwnerHistory=owner_history,
                Name=label,
                ObjectPlacement=storey_placement,
                CompositionType="ELEMENT",
                Elevation=ifc_z,
            )
            storey_list.append((elev, storey_placement, storey))

        ifc_file.create_entity(
            "IfcRelAggregates",
            GlobalId=self._new_guid(),
            OwnerHistory=owner_history,
            RelatingObject=building,
            RelatedObjects=[s for _, _, s in storey_list],
        )
        return storey_list

    # ─── Material-Handling ───

    def _build_material_layer_set(
        self,
        ifc_file,
        layers: list[LayerSpec],
    ):
        """Baut IfcMaterialLayerSet aus Template-Layers."""
        material_cache: dict[str, Any] = {}
        ifc_layers = []
        for layer in layers:
            if layer.material_key not in material_cache:
                material_cache[layer.material_key] = ifc_file.create_entity(
                    "IfcMaterial",
                    Name=layer.material_key,
                )
            material = material_cache[layer.material_key]
            thickness_m = (layer.thickness_mm or 1.0) / 1000.0
            ifc_layer = ifc_file.create_entity(
                "IfcMaterialLayer",
                Material=material,
                LayerThickness=thickness_m,
                IsVentilated=False,
                Name=layer.function,
            )
            ifc_layers.append(ifc_layer)
        if not ifc_layers:
            return None
        return ifc_file.create_entity(
            "IfcMaterialLayerSet",
            MaterialLayers=ifc_layers,
        )

    def _attach_material_to_element(
        self,
        ifc_file,
        element,
        template: TemplateSpec,
        owner_history,
        element_kind: str,
    ) -> bool:
        """Verknüpft Material-Layer-Set mit einer Wand/Decke/Dach.

        Für Türen/Fenster/Treppen/Räume: einfaches IfcMaterial (Primär).
        """
        if not template.layers:
            return False
        if element_kind in ("WALL", "SLAB", "FLOOR", "CEILING", "ROOF"):
            layer_set = self._build_material_layer_set(ifc_file, template.layers)
            if layer_set is None:
                return False
            total_thk = sum((layer.thickness_mm or 0) for layer in template.layers) / 1000.0
            layer_set_usage = ifc_file.create_entity(
                "IfcMaterialLayerSetUsage",
                ForLayerSet=layer_set,
                LayerSetDirection="AXIS2",
                DirectionSense="POSITIVE",
                OffsetFromReferenceLine=-total_thk / 2.0,
            )
            ifc_file.create_entity(
                "IfcRelAssociatesMaterial",
                GlobalId=self._new_guid(),
                OwnerHistory=owner_history,
                RelatedObjects=[element],
                RelatingMaterial=layer_set_usage,
            )
            return True
        # Einfache Material-Zuordnung für Tür/Fenster/Treppe
        primary_name = template.layers[0].material_key if template.layers else template.template_key
        mat = ifc_file.create_entity("IfcMaterial", Name=primary_name)
        ifc_file.create_entity(
            "IfcRelAssociatesMaterial",
            GlobalId=self._new_guid(),
            OwnerHistory=owner_history,
            RelatedObjects=[element],
            RelatingMaterial=mat,
        )
        return True

    def _attach_psets(
        self,
        ifc_file,
        element,
        template: TemplateSpec,
        owner_history,
    ) -> None:
        """Baut pro Pset-Name in ``template.ifc_psets`` ein IfcPropertySet."""
        for pset_name, props in (template.ifc_psets or {}).items():
            if not isinstance(props, dict) or not props:
                continue
            prop_entities = []
            for key, value in props.items():
                p = _value_to_ifc_property(ifc_file, key, value)
                if p is not None:
                    prop_entities.append(p)
            if not prop_entities:
                continue
            pset = ifc_file.create_entity(
                "IfcPropertySet",
                GlobalId=self._new_guid(),
                OwnerHistory=owner_history,
                Name=pset_name,
                HasProperties=prop_entities,
            )
            ifc_file.create_entity(
                "IfcRelDefinesByProperties",
                GlobalId=self._new_guid(),
                OwnerHistory=owner_history,
                RelatedObjects=[element],
                RelatingPropertyDefinition=pset,
            )

    # ─── Element-Mapping ───

    def _make_placement(self, ifc_file, parent_placement, center_scene, ref_direction=None):
        """Erzeugt IfcLocalPlacement aus Scene-Center-Koordinaten.

        ``ref_direction`` (optional, IFC-Koordinaten) setzt die lokale X-Achse
        (Axis = +Z) - damit rotiert das Element in der Grundriss-Ebene entlang
        seiner Wand-Achse.
        """
        if center_scene is None:
            origin = (0.0, 0.0, 0.0)
        else:
            origin = _scene_to_ifc_coords(center_scene)
        point = ifc_file.create_entity("IfcCartesianPoint", Coordinates=origin)
        if ref_direction is not None:
            axis = ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
            refd = ifc_file.create_entity(
                "IfcDirection",
                DirectionRatios=tuple(float(x) for x in ref_direction),
            )
            placement_3d = ifc_file.create_entity(
                "IfcAxis2Placement3D", Location=point, Axis=axis, RefDirection=refd
            )
        else:
            placement_3d = ifc_file.create_entity("IfcAxis2Placement3D", Location=point)
        return ifc_file.create_entity(
            "IfcLocalPlacement",
            PlacementRelTo=parent_placement,
            RelativePlacement=placement_3d,
        )

    def _ref_direction_for(self, component):
        """IFC-RefDirection aus den Plan-Achsenpunkten segment_a/segment_b.

        Plan (x, y) -> Scene (x, up, z=y) -> IFC (x, -y, 0). Damit zeigt die
        lokale X-Achse (Box-Laenge) entlang der Wand. None, wenn keine
        Achsen-Info vorliegt (dann achsparallele Default-Geometrie).
        """
        attrs = getattr(component, "attrs", {}) or {}
        a = attrs.get("segment_a")
        b = attrs.get("segment_b")
        if not a or not b:
            return None
        try:
            dx = float(b[0]) - float(a[0])
            dy = float(b[1]) - float(a[1])
        except (TypeError, ValueError, IndexError):
            return None
        import math

        L = math.hypot(dx, dy)
        if L < 1e-9:
            return None
        return (dx / L, -dy / L, 0.0)

    def _make_box_representation(
        self,
        ifc_file,
        context,
        dims: tuple[float, float, float] | None,
    ):
        """Einfache Bounding-Box-Darstellung via IfcExtrudedAreaSolid.

        Für MVP: alle Elemente als Box repräsentieren. Präzisere Geometrie
        (Wand-Trimming an T-Stössen, Tür-/Fenster-Öffnungen) kommen später.
        """
        if dims is None or any(d is None for d in dims):
            return None
        try:
            length, height, thickness = (float(x) for x in dims)
        except (TypeError, ValueError):
            return None
        if length <= 0 or height <= 0:
            return None
        thickness = max(thickness or 0.1, 0.01)
        # 2D-Profil: Rechteck length × thickness in XY-Ebene
        profile_placement = ifc_file.create_entity(
            "IfcAxis2Placement2D",
            Location=ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0)),
        )
        profile = ifc_file.create_entity(
            "IfcRectangleProfileDef",
            ProfileType="AREA",
            Position=profile_placement,
            XDim=length,
            YDim=thickness,
        )
        direction = ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
        solid_placement = ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)),
        )
        solid = ifc_file.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=profile,
            Position=solid_placement,
            ExtrudedDirection=direction,
            Depth=height,
        )
        shape_rep = ifc_file.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=context,
            RepresentationIdentifier="Body",
            RepresentationType="SweptSolid",
            Items=[solid],
        )
        return ifc_file.create_entity(
            "IfcProductDefinitionShape",
            Representations=[shape_rep],
        )

    def _component_kind_to_ifc_type(self, kind: str) -> str:
        mapping = {
            "WALL": "IfcWallStandardCase",
            "DOOR": "IfcDoorStandardCase",
            "WINDOW": "IfcWindowStandardCase",
            "SLAB": "IfcSlab",
            "FLOOR": "IfcSlab",
            "CEILING": "IfcSlab",
            "ROOF": "IfcRoof",
            "STAIR": "IfcStair",
            "ROOM": "IfcSpace",
            "SPACE": "IfcSpace",
        }
        return mapping.get(kind, "IfcBuildingElementProxy")

    def _create_element(
        self,
        ifc_file,
        component,
        storey_placement,
        context,
        owner_history,
    ):
        ifc_type = self._component_kind_to_ifc_type(component.kind)
        ref_dir = None
        if self._rotate_walls and component.kind in ("WALL", "DOOR", "WINDOW"):
            ref_dir = self._ref_direction_for(component)
        placement = self._make_placement(
            ifc_file, storey_placement, component.center, ref_direction=ref_dir
        )
        representation = self._make_box_representation(ifc_file, context, component.dims)
        # IfcSpace hat Sonder-Attribute (PredefinedType, CompositionType)
        common = {
            "GlobalId": self._new_guid(),
            "OwnerHistory": owner_history,
            "Name": component.name or component.id,
            "ObjectPlacement": placement,
        }
        if representation is not None:
            common["Representation"] = representation
        if ifc_type == "IfcSpace":
            common["CompositionType"] = "ELEMENT"
            try:
                return ifc_file.create_entity(ifc_type, **common)
            except Exception:
                return ifc_file.create_entity("IfcSpace", **common)
        return ifc_file.create_entity(ifc_type, **common)

    def _wire_openings(
        self,
        *,
        ifc_file,
        scene,
        ifc_entity_by_comp_id: dict,
        owner_history,
        context,
    ) -> None:
        """Create IfcOpeningElement + IfcRelVoidsElement + IfcRelFillsElement
        for every HAS_OPENING relationship in the scene.

        Each opening borrows geometry from its filling (door/window) component:
        box-shaped void aligned with the wall, sized to the opening's
        dims, placed at the opening's center. Revit/ArchiCAD then render
        the wall with a real hole instead of a flat surface with a
        door sitting on top.
        """
        relationships = getattr(scene, "relationships", [])
        if not relationships:
            return

        wired = 0
        for rel in relationships:
            if getattr(rel, "rel_type", None) != "HAS_OPENING":
                continue
            wall_comp = scene.components.get(rel.src_id)
            fill_comp = scene.components.get(rel.dst_id)
            if wall_comp is None or fill_comp is None:
                continue
            wall_ifc = ifc_entity_by_comp_id.get(rel.src_id)
            fill_ifc = ifc_entity_by_comp_id.get(rel.dst_id)
            if wall_ifc is None or fill_ifc is None:
                continue

            # Opening dims: match the filling component's dims (width x height
            # x wall_thickness). Inflate thickness slightly so the boolean
            # subtract produces a clean cut through both faces.
            dims = fill_comp.dims or (0.9, 2.1, 0.15)
            try:
                op_w = float(dims[0])
                op_h = float(dims[1])
                op_t = float(dims[2]) * 1.2 + 0.04  # wall thickness + margin
            except (TypeError, ValueError, IndexError):
                continue

            try:
                op_ref_dir = self._ref_direction_for(fill_comp) if self._rotate_walls else None
                placement = self._make_placement(
                    ifc_file,
                    None,
                    fill_comp.center,
                    ref_direction=op_ref_dir,
                )
                representation = self._make_box_representation(
                    ifc_file,
                    context,
                    (op_w, op_h, op_t),
                )
                if representation is None:
                    continue
                opening = ifc_file.create_entity(
                    "IfcOpeningElement",
                    GlobalId=self._new_guid(),
                    OwnerHistory=owner_history,
                    Name=f"Opening for {fill_comp.name or fill_comp.id}",
                    ObjectPlacement=placement,
                    Representation=representation,
                    PredefinedType="OPENING",
                )
                ifc_file.create_entity(
                    "IfcRelVoidsElement",
                    GlobalId=self._new_guid(),
                    OwnerHistory=owner_history,
                    Name="WallVoid",
                    RelatingBuildingElement=wall_ifc,
                    RelatedOpeningElement=opening,
                )
                ifc_file.create_entity(
                    "IfcRelFillsElement",
                    GlobalId=self._new_guid(),
                    OwnerHistory=owner_history,
                    Name="OpeningFill",
                    RelatingOpeningElement=opening,
                    RelatedBuildingElement=fill_ifc,
                )
                wired += 1
            except Exception as exc:
                self._warnings.append(
                    f"Opening wiring failed for {rel.src_id}->{rel.dst_id}: {exc}"
                )

        if wired:
            logger.info("IFC: wired %d opening(s) via IfcRelVoidsElement/IfcRelFillsElement", wired)

    # ─── Hauptmethode ───

    def export(
        self,
        *,
        scene,
        template_assignments: Iterable[dict] | None,
        templates_by_key: dict[str, TemplateSpec] | None,
        output_path: str | Path,
    ) -> IfcExportResult:
        import ifcopenshell

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ifc_file = ifcopenshell.file(schema=self._schema)
        owner_history = self._make_owner_history(ifc_file)
        origin = self._make_axis_placement_origin(ifc_file)
        context = self._make_context(ifc_file, origin)
        units = self._make_units(ifc_file)
        building_placement, building = self._make_spatial_structure(
            ifc_file, owner_history, context, units
        )

        assignment_by_id: dict[str, dict] = {
            a["element_id"]: a for a in (template_assignments or [])
        }
        templates_by_key = templates_by_key or {}

        exported: list[ExportedElement] = []
        components = list(getattr(scene, "components", {}).values())

        # Component id -> IFC entity handle. Populated during the main
        # element loop below, consumed in the post-pass that wires
        # IfcOpeningElement / IfcRelVoidsElement / IfcRelFillsElement.
        ifc_entity_by_comp_id: dict[str, Any] = {}

        # Detect storey levels and create IfcBuildingStoreys
        storey_specs = _detect_storeys(components)
        storey_data = self._make_storeys(
            ifc_file, owner_history, building_placement, building, storey_specs
        )

        def _storey_for_comp(comp) -> tuple:
            """Return (storey_placement, storey) for the nearest floor level."""
            cy = float(comp.center[1]) if hasattr(comp, "center") else 0.0
            dims = getattr(comp, "dims", None)
            hy = (float(dims[1]) / 2.0) if (dims and len(dims) > 1 and dims[1] is not None) else 0.0
            bottom = cy - hy
            best = min(storey_data, key=lambda t: abs(t[0] - bottom))
            return best[1], best[2]  # storey_placement, storey

        # Collect (element, storey) pairs for spatial containment
        storey_elements: dict[int, list] = {i: [] for i in range(len(storey_data))}
        storey_index_map = {id(sd[2]): i for i, sd in enumerate(storey_data)}

        for comp in components:
            s_placement, s_storey = _storey_for_comp(comp)
            try:
                ifc_element = self._create_element(
                    ifc_file,
                    comp,
                    s_placement,
                    context,
                    owner_history,
                )
            except Exception as exc:
                self._warnings.append(f"{comp.id}: IFC-Entity-Erzeugung fehlgeschlagen - {exc}")
                continue

            # Material + Psets wenn Assignment vorhanden
            material_attached = False
            assignment = assignment_by_id.get(comp.id)
            if assignment and assignment.get("template_key"):
                tmpl = templates_by_key.get(assignment["template_key"])
                if tmpl:
                    try:
                        material_attached = self._attach_material_to_element(
                            ifc_file,
                            ifc_element,
                            tmpl,
                            owner_history,
                            comp.kind,
                        )
                        self._attach_psets(ifc_file, ifc_element, tmpl, owner_history)
                    except Exception as exc:
                        self._warnings.append(
                            f"{comp.id}: Material/Pset-Anhang fehlgeschlagen - {exc}"
                        )

            s_idx = storey_index_map[id(s_storey)]
            storey_elements[s_idx].append(ifc_element)
            ifc_entity_by_comp_id[comp.id] = ifc_element
            exported.append(
                ExportedElement(
                    component_id=comp.id,
                    ifc_type=ifc_element.is_a(),
                    global_id=ifc_element.GlobalId,
                    material_set_attached=material_attached,
                )
            )

        # ─── Canonical wall ↔ opening ↔ filling topology ──────────────
        # Per IFC 4.3 spec (IfcWallStandardCase + IfcRelVoidsElement +
        # IfcRelFillsElement), a door is not placed "next to" a wall; the
        # wall HAS an IfcOpeningElement that VOIDS a slice of it, and the
        # opening is FILLED by the door or window. Without these
        # relationships Revit/ArchiCAD open the file with floating doors
        # and solid walls.
        self._wire_openings(
            ifc_file=ifc_file,
            scene=scene,
            ifc_entity_by_comp_id=ifc_entity_by_comp_id,
            owner_history=owner_history,
            context=context,
        )

        # One IfcRelContainedInSpatialStructure per storey
        for i, (_, _, storey) in enumerate(storey_data):
            members = storey_elements.get(i, [])
            if members:
                ifc_file.create_entity(
                    "IfcRelContainedInSpatialStructure",
                    GlobalId=self._new_guid(),
                    OwnerHistory=owner_history,
                    Name="BuildingStoreyContainer",
                    RelatingStructure=storey,
                    RelatedElements=members,
                )

        ifc_file.write(str(output_path))
        logger.info(
            "IFC-Export ok: %s (%d Elemente, Schema %s)",
            output_path,
            len(exported),
            self._schema,
        )
        return IfcExportResult(
            path=str(output_path),
            schema=self._schema,
            element_count=len(exported),
            exported=exported,
            warnings=list(self._warnings),
        )


__all__ = [
    "IfcExporter",
    "IfcExportResult",
    "_detect_storeys",
    "ExportedElement",
    "IfcSchemaVersion",
]
