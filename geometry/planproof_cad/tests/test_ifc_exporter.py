"""Sprint 8: IFC4-Export Tests.

Deckt:
- IfcExporter end-to-end auf Fixture-Scene
- Roundtrip: write -> re-read -> Struktur passt
- Material-Layer-Set wird korrekt angelegt
- IfcPropertySets aus Template uebernommen
- Missing-Template: Element wird trotzdem exportiert (ohne Material-Set)
- Unbekannter Kind: faellt auf IfcBuildingElementProxy zurueck
- IFC2x3-Fallback ohne neue IFC4-Attribute
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Ohne ifcopenshell skippen wir das ganze Modul
ifcopenshell = pytest.importorskip("ifcopenshell")

from planproof_cad.catalog.schemas import LayerSpec, TemplateSpec  # noqa: E402
from planproof_cad.export.ifc import IfcExporter, _detect_storeys  # noqa: E402
from planproof_cad.model.scene_model import Component, Evidence, SceneModel  # noqa: E402

# ───────────── Fixtures ─────────────


def _make_wall_template(u=0.22):
    return TemplateSpec(
        template_key="aw_test",
        name="AW Test",
        element_type="wall",
        ifc_entity_type="IfcWallStandardCase",
        category="exterior_wall_loadbearing",
        layers=[
            LayerSpec(material_key="gipsputz_15", thickness_mm=15, function="interior_finish"),
            LayerSpec(material_key="ks_1_8_240", thickness_mm=240, function="structural"),
            LayerSpec(
                material_key="wdvs_eps_035_140", thickness_mm=140, function="thermal_insulation"
            ),
        ],
        computed_properties={"u_value": u, "total_thickness_mm": 395},
        ifc_psets={
            "Pset_WallCommon": {
                "IsExternal": True,
                "LoadBearing": True,
                "FireRating": "REI 90",
                "ThermalTransmittance": u,
            }
        },
        din_references=["DIN 4108-2"],
    )


def _make_door_template():
    return TemplateSpec(
        template_key="t_innen_86_2010",
        name="Innentür 86 × 198.5",
        element_type="door",
        ifc_entity_type="IfcDoorStandardCase",
        category="interior_door",
        layers=[LayerSpec(material_key="tuerblatt_holz_innen", thickness_mm=40, function="leaf")],
        geometry_params={
            "default_width_mm": 860,
            "default_height_mm": 1985,
            "leaf_type": "solid_wood",
        },
        ifc_psets={"Pset_DoorCommon": {"IsExternal": False, "FireRating": "None"}},
        din_references=["DIN 18101"],
    )


def _make_room_template():
    return TemplateSpec(
        template_key="r_wohnen",
        name="Wohnzimmer",
        element_type="space",
        ifc_entity_type="IfcSpace",
        category="living_room",
        layers=[],
        geometry_params={"space_type": "living", "nuf_class": "NUF1.1"},
        ifc_psets={"Pset_SpaceCommon": {"Category": "living"}},
        din_references=["DIN 277"],
    )


def _make_fixture_scene() -> SceneModel:
    scene = SceneModel()
    scene.add_component(
        Component(
            id="w1",
            name="Wand 1",
            kind="WALL",
            center=(0.0, 1.375, 0.0),
            dims=(5.0, 2.75, 0.395),
            attrs={"thickness_m": 0.395, "wall_type": "exterior", "is_load_bearing": True},
            evidence=[Evidence(source="test", method="test")],
        )
    )
    scene.add_component(
        Component(
            id="w2",
            name="Wand 2",
            kind="WALL",
            center=(2.5, 1.375, 2.5),
            dims=(5.0, 2.75, 0.395),
            attrs={"thickness_m": 0.395},
            evidence=[Evidence(source="test", method="test")],
        )
    )
    scene.add_component(
        Component(
            id="d1",
            name="Tür",
            kind="DOOR",
            center=(1.0, 1.0, 0.2),
            dims=(0.86, 2.01, 0.1),
            attrs={"width_m": 0.86, "height_m": 1.985},
            evidence=[Evidence(source="test", method="test")],
        )
    )
    scene.add_component(
        Component(
            id="r1",
            name="Wohnzimmer",
            kind="ROOM",
            center=(2.0, 1.375, 2.0),
            dims=(4.0, 2.75, 4.0),
            attrs={"area_m2": 16.0, "label": "Wohnzimmer"},
            evidence=[Evidence(source="test", method="test")],
        )
    )
    return scene


# ───────────── Exporter Tests ─────────────


def test_exporter_writes_file_and_returns_result(tmp_path):
    scene = _make_fixture_scene()
    assignments = [
        {
            "element_id": "w1",
            "element_kind": "WALL",
            "template_key": "aw_test",
            "template_version": 1,
            "confidence": 0.9,
            "rationale": "t",
        },
        {
            "element_id": "w2",
            "element_kind": "WALL",
            "template_key": "aw_test",
            "template_version": 1,
            "confidence": 0.9,
            "rationale": "t",
        },
        {
            "element_id": "d1",
            "element_kind": "DOOR",
            "template_key": "t_innen_86_2010",
            "template_version": 1,
            "confidence": 0.9,
            "rationale": "t",
        },
        {
            "element_id": "r1",
            "element_kind": "ROOM",
            "template_key": "r_wohnen",
            "template_version": 1,
            "confidence": 0.9,
            "rationale": "t",
        },
    ]
    templates = {
        "aw_test": _make_wall_template(),
        "t_innen_86_2010": _make_door_template(),
        "r_wohnen": _make_room_template(),
    }
    out = tmp_path / "plan.ifc"
    result = IfcExporter().export(
        scene=scene,
        template_assignments=assignments,
        templates_by_key=templates,
        output_path=out,
    )
    assert out.exists()
    assert result.element_count == 4
    assert result.schema == "IFC4"
    assert result.warnings == []


def test_exported_ifc_parses_and_has_project_hierarchy(tmp_path):
    scene = _make_fixture_scene()
    out = tmp_path / "plan.ifc"
    IfcExporter().export(
        scene=scene,
        template_assignments=[],
        templates_by_key={},
        output_path=out,
    )
    f = ifcopenshell.open(str(out))
    assert f.schema == "IFC4"
    assert len(f.by_type("IfcProject")) == 1
    assert len(f.by_type("IfcSite")) == 1
    assert len(f.by_type("IfcBuilding")) == 1
    assert len(f.by_type("IfcBuildingStorey")) == 1


def test_walls_are_ifcwallstandardcase(tmp_path):
    scene = _make_fixture_scene()
    out = tmp_path / "plan.ifc"
    IfcExporter().export(
        scene=scene,
        template_assignments=[],
        templates_by_key={},
        output_path=out,
    )
    f = ifcopenshell.open(str(out))
    walls = f.by_type("IfcWallStandardCase")
    assert len(walls) == 2
    # Walls must be contained in storey
    rels = f.by_type("IfcRelContainedInSpatialStructure")
    assert len(rels) >= 1
    contained = [e.id() for r in rels for e in r.RelatedElements]
    for w in walls:
        assert w.id() in contained


def test_doors_are_ifcdoorstandardcase(tmp_path):
    scene = _make_fixture_scene()
    out = tmp_path / "plan.ifc"
    IfcExporter().export(
        scene=scene,
        template_assignments=[],
        templates_by_key={},
        output_path=out,
    )
    f = ifcopenshell.open(str(out))
    assert len(f.by_type("IfcDoorStandardCase")) == 1


def test_rooms_are_ifcspace(tmp_path):
    scene = _make_fixture_scene()
    out = tmp_path / "plan.ifc"
    IfcExporter().export(
        scene=scene,
        template_assignments=[],
        templates_by_key={},
        output_path=out,
    )
    f = ifcopenshell.open(str(out))
    spaces = f.by_type("IfcSpace")
    assert len(spaces) == 1
    assert spaces[0].Name == "Wohnzimmer"


def test_material_layer_set_attached_to_wall(tmp_path):
    scene = _make_fixture_scene()
    assignments = [
        {
            "element_id": "w1",
            "element_kind": "WALL",
            "template_key": "aw_test",
            "template_version": 1,
            "confidence": 0.9,
            "rationale": "t",
        }
    ]
    templates = {"aw_test": _make_wall_template()}
    out = tmp_path / "plan.ifc"
    IfcExporter().export(
        scene=scene,
        template_assignments=assignments,
        templates_by_key=templates,
        output_path=out,
    )
    f = ifcopenshell.open(str(out))
    layer_sets = f.by_type("IfcMaterialLayerSet")
    assert len(layer_sets) == 1
    assert len(layer_sets[0].MaterialLayers) == 3
    # Thickness summiert auf 395mm = 0.395m
    total = sum(layer.LayerThickness for layer in layer_sets[0].MaterialLayers)
    assert abs(total - 0.395) < 1e-6


def test_material_layer_set_usage_created(tmp_path):
    scene = _make_fixture_scene()
    assignments = [
        {
            "element_id": "w1",
            "element_kind": "WALL",
            "template_key": "aw_test",
            "template_version": 1,
            "confidence": 0.9,
            "rationale": "t",
        }
    ]
    out = tmp_path / "plan.ifc"
    IfcExporter().export(
        scene=scene,
        template_assignments=assignments,
        templates_by_key={"aw_test": _make_wall_template()},
        output_path=out,
    )
    f = ifcopenshell.open(str(out))
    usages = f.by_type("IfcMaterialLayerSetUsage")
    assert len(usages) == 1
    assert usages[0].LayerSetDirection == "AXIS2"
    assert usages[0].DirectionSense == "POSITIVE"


def test_pset_wall_common_populated(tmp_path):
    scene = _make_fixture_scene()
    assignments = [
        {
            "element_id": "w1",
            "element_kind": "WALL",
            "template_key": "aw_test",
            "template_version": 1,
            "confidence": 0.9,
            "rationale": "t",
        }
    ]
    out = tmp_path / "plan.ifc"
    IfcExporter().export(
        scene=scene,
        template_assignments=assignments,
        templates_by_key={"aw_test": _make_wall_template()},
        output_path=out,
    )
    f = ifcopenshell.open(str(out))
    psets = [p for p in f.by_type("IfcPropertySet") if p.Name == "Pset_WallCommon"]
    assert len(psets) == 1
    props = {p.Name: p.NominalValue.wrappedValue for p in psets[0].HasProperties}
    assert props["IsExternal"] is True
    assert props["LoadBearing"] is True
    assert props["FireRating"] == "REI 90"
    assert abs(float(props["ThermalTransmittance"]) - 0.22) < 1e-6


def test_missing_template_still_exports_element(tmp_path):
    scene = _make_fixture_scene()
    # Nur 1 Wand bekommt ein Assignment, der Rest nicht
    assignments = [
        {
            "element_id": "w1",
            "element_kind": "WALL",
            "template_key": "aw_test",
            "template_version": 1,
            "confidence": 0.9,
            "rationale": "t",
        }
    ]
    out = tmp_path / "plan.ifc"
    result = IfcExporter().export(
        scene=scene,
        template_assignments=assignments,
        templates_by_key={"aw_test": _make_wall_template()},
        output_path=out,
    )
    assert result.element_count == 4
    f = ifcopenshell.open(str(out))
    # Beide Waende exportiert, aber nur 1 Material-Rel
    assert len(f.by_type("IfcWallStandardCase")) == 2
    assert len(f.by_type("IfcRelAssociatesMaterial")) == 1


def test_unknown_kind_falls_back_to_proxy(tmp_path):
    scene = SceneModel()
    scene.add_component(
        Component(
            id="x1",
            name="Mystery",
            kind="WEIRD_THING",
            center=(0.0, 1.0, 0.0),
            dims=(1.0, 2.0, 0.1),
            attrs={},
            evidence=[Evidence(source="t", method="t")],
        )
    )
    out = tmp_path / "plan.ifc"
    IfcExporter().export(
        scene=scene,
        template_assignments=[],
        templates_by_key={},
        output_path=out,
    )
    f = ifcopenshell.open(str(out))
    assert len(f.by_type("IfcBuildingElementProxy")) == 1


def test_empty_scene_still_produces_valid_project(tmp_path):
    scene = SceneModel()
    out = tmp_path / "plan.ifc"
    result = IfcExporter().export(
        scene=scene,
        template_assignments=[],
        templates_by_key={},
        output_path=out,
    )
    assert result.element_count == 0
    f = ifcopenshell.open(str(out))
    assert len(f.by_type("IfcProject")) == 1


def test_ifc2x3_schema_export(tmp_path):
    scene = _make_fixture_scene()
    out = tmp_path / "plan.ifc"
    exporter = IfcExporter(schema="IFC2X3")
    result = exporter.export(
        scene=scene,
        template_assignments=[],
        templates_by_key={},
        output_path=out,
    )
    assert result.schema == "IFC2X3"
    f = ifcopenshell.open(str(out))
    assert f.schema == "IFC2X3"


def test_coordinate_transformation_y_up_to_z_up(tmp_path):
    scene = SceneModel()
    scene.add_component(
        Component(
            id="w1",
            name="High Wall",
            kind="WALL",
            center=(3.0, 2.0, 5.0),  # scene Y=up: y=2.0
            dims=(4.0, 2.75, 0.24),
            attrs={},
            evidence=[Evidence(source="t", method="t")],
        )
    )
    out = tmp_path / "plan.ifc"
    IfcExporter().export(
        scene=scene,
        template_assignments=[],
        templates_by_key={},
        output_path=out,
    )
    f = ifcopenshell.open(str(out))
    wall = f.by_type("IfcWallStandardCase")[0]
    placement = wall.ObjectPlacement.RelativePlacement.Location.Coordinates
    # scene (3, 2, 5) → ifc (3, -5, 2)
    assert placement[0] == pytest.approx(3.0)
    assert placement[1] == pytest.approx(-5.0)
    assert placement[2] == pytest.approx(2.0)


def test_warnings_captured_for_failed_element(tmp_path, monkeypatch):
    scene = _make_fixture_scene()
    exporter = IfcExporter()
    # Patch _create_element to raise for a specific component
    original = exporter._create_element

    def failing(ifc_file, comp, *args, **kwargs):
        if comp.id == "w1":
            raise RuntimeError("boom")
        return original(ifc_file, comp, *args, **kwargs)

    monkeypatch.setattr(exporter, "_create_element", failing)
    out = tmp_path / "plan.ifc"
    result = exporter.export(
        scene=scene,
        template_assignments=[],
        templates_by_key={},
        output_path=out,
    )
    assert any("w1" in w for w in result.warnings)
    assert result.element_count == 3  # w1 skipped


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Storey
# ═══════════════════════════════════════════════════════════════════════════


def _fake_comp(id_: str, center_y: float, height: float = 3.0):
    return SimpleNamespace(
        id=id_,
        kind="WALL",
        name=id_,
        center=(0.0, center_y, 0.0),
        dims=(5.0, height, 0.25),
        attrs={},
    )


def test_detect_storeys_single_level():
    comps = [_fake_comp("w1", 1.5), _fake_comp("w2", 1.5), _fake_comp("w3", 1.6)]
    result = _detect_storeys(comps)
    assert len(result) == 1
    assert result[0][1] == 0  # storey_index 0


def test_detect_storeys_two_levels():
    # EG: bottom Y ≈ 0, OG: bottom Y ≈ 3.0 (gap > 1.5m)
    comps = [
        _fake_comp("eg1", 1.5),  # bottom = 1.5 - 1.5 = 0.0
        _fake_comp("og1", 4.5),  # bottom = 4.5 - 1.5 = 3.0
    ]
    result = _detect_storeys(comps)
    assert len(result) == 2
    assert result[0][1] == 0
    assert result[1][1] == 1
    assert result[1][0] - result[0][0] > 1.5


def test_ifc_export_multi_storey_assigns_elements(tmp_path):
    """Two floor levels → two IfcBuildingStorey, elements distributed correctly."""
    wall_tmpl = _make_wall_template()
    eg_comp = _fake_comp("eg_wall", center_y=1.5, height=3.0)  # bottom=0.0
    og_comp = _fake_comp("og_wall", center_y=4.5, height=3.0)  # bottom=3.0
    scene = SimpleNamespace(components={"eg_wall": eg_comp, "og_wall": og_comp})

    assignments = [
        {
            "element_id": "eg_wall",
            "element_kind": "WALL",
            "template_key": "wall_tmpl",
            "template_version": 1,
            "confidence": 1.0,
            "rationale": "",
        },
        {
            "element_id": "og_wall",
            "element_kind": "WALL",
            "template_key": "wall_tmpl",
            "template_version": 1,
            "confidence": 1.0,
            "rationale": "",
        },
    ]
    templates_by_key = {"wall_tmpl": wall_tmpl}

    out_path = tmp_path / "multi.ifc"
    exporter = IfcExporter(schema="IFC4")
    result = exporter.export(
        scene=scene,
        template_assignments=assignments,
        templates_by_key=templates_by_key,
        output_path=out_path,
    )
    assert result.element_count == 2

    ifc = ifcopenshell.open(str(out_path))
    storeys = ifc.by_type("IfcBuildingStorey")
    assert len(storeys) == 2, f"Expected 2 storeys, got {len(storeys)}"

    # Each storey should contain exactly one wall
    for storey in storeys:
        rels = [
            r
            for r in ifc.by_type("IfcRelContainedInSpatialStructure")
            if r.RelatingStructure == storey
        ]
        total_elements = sum(len(r.RelatedElements) for r in rels)
        assert total_elements == 1, (
            f"Storey '{storey.Name}' should contain 1 element, got {total_elements}"
        )
