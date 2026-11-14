"""Sprint 7: Template-Assignment-Rules + DIN-Validator + Pipeline-Steps."""

from __future__ import annotations

from types import SimpleNamespace

from planproof_cad.catalog.assignment_rules import (
    Assignment,
    DetectedElement,
    apply_forced_overrides,
    assign_door,
    assign_element,
    assign_many,
    assign_room,
    assign_slab,
    assign_stair,
    assign_wall,
    assign_window,
    validate_templates_exist,
)
from planproof_cad.catalog.din_validator import (
    DINViolation,
    summarize_violations,
    validate_assignment,
    validate_assignments,
)
from planproof_cad.catalog.schemas import LayerSpec, TemplateSpec

# ═══════════════════════════════════════════════════════════════════════════
# Wand-Rules
# ═══════════════════════════════════════════════════════════════════════════


def test_wall_exterior_loadbearing_36cm_maps_to_ks24_wdvs140():
    elem = DetectedElement(
        id="w1",
        kind="WALL",
        thickness_m=0.365,
        is_external=True,
        is_load_bearing=True,
    )
    a = assign_wall(elem)
    assert a.template_key == "aw_ks24_wdvs140_eps"
    assert a.confidence > 0.5


def test_wall_interior_loadbearing_24cm_maps_to_ks240():
    elem = DetectedElement(
        id="w2",
        kind="WALL",
        thickness_m=0.24,
        is_external=False,
        is_load_bearing=True,
    )
    a = assign_wall(elem)
    assert a.template_key == "iw_ks240_tragend"


def test_wall_interior_partition_thin_maps_to_gk():
    # 8cm - zu duenn fuer KS (min 9cm), muss GK-Staender sein
    elem = DetectedElement(
        id="w3",
        kind="WALL",
        thickness_m=0.08,
        is_external=False,
        is_load_bearing=False,
    )
    a = assign_wall(elem)
    assert a.template_key == "iw_zt_gk_staender100_rw45"


def test_wall_unknown_external_falls_back_to_generic_thickness_rule():
    elem = DetectedElement(id="w4", kind="WALL", thickness_m=0.36)
    a = assign_wall(elem)
    # None/None rule matches 0.30..0.46 → aw_ks24_wdvs140_eps
    assert a.template_key == "aw_ks24_wdvs140_eps"


def test_wall_without_thickness_gets_no_template():
    elem = DetectedElement(id="w5", kind="WALL")
    a = assign_wall(elem)
    assert a.template_key is None
    assert a.confidence == 0.0


def test_wall_confidence_peak_at_range_center():
    # KS 24 + WDVS 140 → Range 0.33-0.40, mid = 0.365
    mid = DetectedElement(
        id="w", kind="WALL", thickness_m=0.365, is_external=True, is_load_bearing=True
    )
    edge = DetectedElement(
        id="w", kind="WALL", thickness_m=0.40, is_external=True, is_load_bearing=True
    )
    assert assign_wall(mid).confidence > assign_wall(edge).confidence


# ═══════════════════════════════════════════════════════════════════════════
# Tür-Rules
# ═══════════════════════════════════════════════════════════════════════════


def test_door_standard_86_matches_din_18101():
    elem = DetectedElement(id="t1", kind="DOOR", width_m=0.86, height_m=1.985)
    a = assign_door(elem)
    assert a.template_key == "t_innen_86_2010"
    assert a.confidence >= 0.95


def test_door_near_standard_885_matches_86():
    elem = DetectedElement(id="t1", kind="DOOR", width_m=0.885, height_m=2.01)
    a = assign_door(elem)
    assert a.template_key == "t_innen_86_2010"
    # Leicht daneben → confidence < 1.0
    assert 0.5 <= a.confidence < 1.0


def test_door_way_off_gets_no_assignment():
    elem = DetectedElement(id="t1", kind="DOOR", width_m=1.8, height_m=2.5)
    a = assign_door(elem)
    assert a.template_key is None


def test_door_without_dims_no_assignment():
    elem = DetectedElement(id="t1", kind="DOOR")
    a = assign_door(elem)
    assert a.template_key is None


# ═══════════════════════════════════════════════════════════════════════════
# Fenster-Rules
# ═══════════════════════════════════════════════════════════════════════════


def test_window_120x140_matches_standard():
    elem = DetectedElement(id="f1", kind="WINDOW", width_m=1.2, height_m=1.4)
    a = assign_window(elem)
    assert a.template_key == "f_pvc_3fach_120_140"


def test_window_off_size_falls_back_to_pvc_3fach():
    elem = DetectedElement(id="f1", kind="WINDOW", width_m=5.0, height_m=5.0)
    a = assign_window(elem)
    assert a.template_key == "f_pvc_3fach_120_140"
    assert a.confidence == 0.3


# ═══════════════════════════════════════════════════════════════════════════
# Slab-Rules
# ═══════════════════════════════════════════════════════════════════════════


def test_slab_geschossdecke_18cm():
    elem = DetectedElement(id="s1", kind="SLAB", thickness_m=0.18, is_ground_slab=False)
    a = assign_slab(elem)
    assert a.template_key == "decke_beton180_estrich60_eps"


def test_slab_bodenplatte_25cm():
    elem = DetectedElement(id="s1", kind="SLAB", thickness_m=0.25, is_ground_slab=True)
    a = assign_slab(elem)
    assert a.template_key == "bp_beton250_perimeter120"


# ═══════════════════════════════════════════════════════════════════════════
# Stair-Rules
# ═══════════════════════════════════════════════════════════════════════════


def test_stair_efh_by_width():
    elem = DetectedElement(id="st1", kind="STAIR", width_m=0.90)
    a = assign_stair(elem)
    assert a.template_key == "s_beton_efh_efh_18_28_100"


def test_stair_mfh_by_width():
    elem = DetectedElement(id="st1", kind="STAIR", width_m=1.10)
    a = assign_stair(elem)
    assert a.template_key == "s_beton_mfh_17_29_125"


def test_stair_buero_by_width():
    elem = DetectedElement(id="st1", kind="STAIR", width_m=1.50)
    a = assign_stair(elem)
    assert a.template_key == "s_beton_buero_17_30_150"


# ═══════════════════════════════════════════════════════════════════════════
# Room-Rules
# ═══════════════════════════════════════════════════════════════════════════


def test_room_by_label_wohnzimmer():
    elem = DetectedElement(id="r1", kind="ROOM", label="Wohnzimmer", area_m2=22.0)
    a = assign_room(elem)
    assert a.template_key == "r_wohnen"
    assert a.confidence >= 0.9


def test_room_by_label_kueche():
    elem = DetectedElement(id="r1", kind="ROOM", label="Küche")
    a = assign_room(elem)
    assert a.template_key == "r_kueche"


def test_room_by_area_fallback():
    elem = DetectedElement(id="r1", kind="ROOM", area_m2=12.0)  # no label
    a = assign_room(elem)
    assert a.template_key == "r_schlafen"


# ═══════════════════════════════════════════════════════════════════════════
# Dispatch + Validator
# ═══════════════════════════════════════════════════════════════════════════


def test_assign_many_dispatches_by_kind():
    elems = [
        DetectedElement(
            id="w", kind="WALL", thickness_m=0.24, is_external=False, is_load_bearing=True
        ),
        DetectedElement(id="t", kind="DOOR", width_m=0.86, height_m=1.985),
    ]
    results = assign_many(elems)
    assert len(results) == 2
    assert results[0].template_key == "iw_ks240_tragend"
    assert results[1].template_key == "t_innen_86_2010"


def test_assign_unknown_kind():
    elem = DetectedElement(id="x", kind="UNKNOWN_KIND")
    a = assign_element(elem)
    assert a.template_key is None


def test_validate_templates_exist_filters_missing():
    assignments = [
        Assignment(
            element_id="w1",
            element_kind="WALL",
            template_key="iw_ks240_tragend",
            confidence=0.9,
            rationale="test",
        ),
        Assignment(
            element_id="w2",
            element_kind="WALL",
            template_key="ghost_template",
            confidence=0.9,
            rationale="test",
        ),
    ]
    available = {"iw_ks240_tragend"}
    result = validate_templates_exist(assignments, available)
    assert result[0].template_key == "iw_ks240_tragend"
    assert result[1].template_key is None
    assert "nicht im Katalog" in result[1].rationale


# ═══════════════════════════════════════════════════════════════════════════
# DIN-Validator
# ═══════════════════════════════════════════════════════════════════════════


def _make_wall_template(
    key: str = "aw_test",
    u_value: float = 0.22,
    rw: int | None = 53,
    fire: str | None = "REI 90",
    category: str = "exterior_wall_loadbearing",
    layers: list | None = None,
    psets: dict | None = None,
) -> TemplateSpec:
    return TemplateSpec(
        template_key=key,
        name=key,
        element_type="wall",
        ifc_entity_type="IfcWallStandardCase",
        category=category,
        layers=layers or [LayerSpec(material_key="ks_24", thickness_mm=240, function="structural")],
        geometry_params={},
        computed_properties={
            **({"u_value": u_value} if u_value is not None else {}),
            **({"sound_reduction_rw": rw} if rw is not None else {}),
            **({"fire_resistance": fire} if fire else {}),
        },
        ifc_psets=psets or {"Pset_WallCommon": {"LoadBearing": True}},
        din_references=["DIN 4108-2"],
    )


def test_geg_flags_excessive_u_value_exterior_wall():
    tmpl = _make_wall_template(u_value=0.35)
    a = Assignment("w1", "WALL", "aw_test", confidence=0.9, rationale="")
    violations = validate_assignment(a, tmpl)
    assert any(v.severity == "error" and "GEG" in v.din_reference for v in violations)


def test_geg_info_level_when_near_threshold():
    # 0.27 ist 96% vom Grenzwert 0.28
    tmpl = _make_wall_template(u_value=0.27)
    a = Assignment("w1", "WALL", "aw_test", confidence=0.9, rationale="")
    violations = validate_assignment(a, tmpl)
    assert any(v.severity == "info" for v in violations)


def test_geg_clean_u_no_violations():
    tmpl = _make_wall_template(u_value=0.18)
    a = Assignment("w1", "WALL", "aw_test", confidence=0.9, rationale="")
    violations = validate_assignment(a, tmpl)
    assert all(v.severity != "error" for v in violations)


def test_rw_violation_wohnungstrennwand():
    tmpl = _make_wall_template(
        u_value=None,
        rw=45,
        category="interior_wall_apartment_separation",
        layers=[LayerSpec(material_key="ks_24", thickness_mm=175, function="structural")],
    )
    a = Assignment("w1", "WALL", "aw_test", confidence=0.9, rationale="")
    violations = validate_assignment(a, tmpl)
    assert any(v.din_reference.startswith("DIN 4109") and v.severity == "error" for v in violations)


def test_firewall_requires_m_classification():
    tmpl = _make_wall_template(
        u_value=None,
        rw=None,
        fire="REI 90",
        category="interior_wall_firewall",
    )
    a = Assignment("w1", "WALL", "aw_test", confidence=0.9, rationale="")
    violations = validate_assignment(a, tmpl)
    assert any("M" in v.message and v.severity == "error" for v in violations)


def test_firewall_with_m_passes():
    tmpl = _make_wall_template(
        u_value=None,
        rw=None,
        fire="REI 90-M",
        category="interior_wall_firewall",
    )
    a = Assignment("w1", "WALL", "aw_test", confidence=0.9, rationale="")
    violations = validate_assignment(a, tmpl)
    assert all(v.din_reference != "DIN 4102-3" or v.severity != "error" for v in violations)


def test_din_1053_min_thickness_enforced():
    tmpl = _make_wall_template(
        u_value=None,
        rw=None,
        layers=[LayerSpec(material_key="ks_thin", thickness_mm=80, function="structural")],
    )
    a = Assignment("w1", "WALL", "aw_test", confidence=0.9, rationale="")
    violations = validate_assignment(a, tmpl)
    assert any(v.din_reference == "DIN 1053-1" and v.severity == "error" for v in violations)


def test_validate_assignments_skips_none_template():
    a = Assignment("w1", "WALL", None, confidence=0.0, rationale="none")
    result = validate_assignments([a], {})
    assert result == []


def test_summarize_violations_counts_by_severity():
    violations = [
        DINViolation("w1", "WALL", "GEG", "error", "x"),
        DINViolation("w2", "WALL", "GEG", "error", "y"),
        DINViolation("w3", "WALL", "GEG", "warning", "z"),
    ]
    summary = summarize_violations(violations)
    assert summary == {"info": 0, "warning": 1, "error": 2}


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline-Step-Integration
# ═══════════════════════════════════════════════════════════════════════════


def _fake_ctx():
    logs = []

    class Ctx:
        def log(self, msg):
            logs.append(msg)

    c = Ctx()
    c.logs = logs
    return c


def test_template_assignment_step_skips_without_scene():
    from planproof_cad.catalog.template_assignment import TemplateAssignmentStep

    state = SimpleNamespace(scene=None, config={}, metrics={})
    step = TemplateAssignmentStep()
    ctx = _fake_ctx()
    out = step.run(ctx, state)
    assert out is state
    assert any("keine Scene-Components" in m for m in ctx.logs)


def test_template_assignment_step_skips_when_disabled():
    from planproof_cad.catalog.template_assignment import TemplateAssignmentStep

    state = SimpleNamespace(
        scene=SimpleNamespace(components={"x": "y"}),
        config={"template_assignment_enabled": False},
        metrics={},
    )
    step = TemplateAssignmentStep()
    ctx = _fake_ctx()
    step.run(ctx, state)
    assert any("deaktiviert" in m for m in ctx.logs)


def test_template_assignment_runs_end_to_end_with_packaged_seed(tmp_path):
    """Smoke-Test: mitgelieferter Seed wird gefunden, Scene wird zugeordnet."""
    from planproof_cad.catalog.template_assignment import TemplateAssignmentStep
    from planproof_cad.model.scene_model import Component, Evidence, SceneModel

    scene = SceneModel()
    scene.add_component(
        Component(
            id="wall-1",
            name="Wall 1",
            kind="WALL",
            dims=(5.0, 2.75, 0.365),
            attrs={"thickness_m": 0.365, "wall_type": "exterior", "is_load_bearing": True},
            evidence=[Evidence(source="test", method="test")],
        )
    )
    scene.add_component(
        Component(
            id="door-1",
            name="Door 1",
            kind="DOOR",
            dims=(0.86, 2.01, 0.1),
            attrs={"width_m": 0.86, "height_m": 1.985},
            evidence=[Evidence(source="test", method="test")],
        )
    )
    state = SimpleNamespace(scene=scene, config={}, metrics={})
    step = TemplateAssignmentStep()
    ctx = _fake_ctx()
    step.run(ctx, state)

    assert step._loaded, state.template_assignment
    assert state.template_assignment["ok"] is True
    assert state.template_assignment["templates_loaded"] > 0

    assert state.template_assignments is not None
    assert len(state.template_assignments) == 2
    keys = {a["template_key"] for a in state.template_assignments}
    assert "aw_ks24_wdvs140_eps" in keys
    assert "t_innen_86_2010" in keys
    assert "template_assignments" in state.metrics


def test_din_validator_step_skips_without_assignments():
    from planproof_cad.catalog.template_assignment import DINValidatorStep

    state = SimpleNamespace(config={}, metrics={}, template_assignments=None)
    step = DINValidatorStep()
    ctx = _fake_ctx()
    step.run(ctx, state)
    assert any("keine Assignments" in m for m in ctx.logs)


def test_din_validator_step_reports_violations():
    from planproof_cad.catalog.template_assignment import DINValidatorStep

    bad_tmpl = _make_wall_template(u_value=0.40)
    state = SimpleNamespace(
        config={},
        metrics={},
        template_assignments=[
            {
                "element_id": "w1",
                "element_kind": "WALL",
                "template_key": "aw_test",
                "template_version": 1,
                "confidence": 0.9,
                "rationale": "test",
            }
        ],
    )
    state._catalog_templates_cache = {"aw_test": bad_tmpl}
    step = DINValidatorStep()
    ctx = _fake_ctx()
    step.run(ctx, state)
    assert state.din_violations
    assert state.metrics["din_violations"]["error"] >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Override-Bridge
# ═══════════════════════════════════════════════════════════════════════════


def test_apply_forced_overrides_replaces_auto():
    """User override wins: template_key ersetzt, confidence=1.0, rationale nennt 'User-Override'."""
    assignments = [
        Assignment(
            element_id="w1",
            element_kind="WALL",
            template_key="aw_ks24_wdvs140_eps",
            confidence=0.8,
            rationale="auto",
        ),
        Assignment(
            element_id="w2",
            element_kind="WALL",
            template_key="aw_ks24_wdvs140_eps",
            confidence=0.9,
            rationale="auto",
        ),
    ]
    forced = {"w1": "aw_ks30_wdvs160_mw"}
    result = apply_forced_overrides(assignments, forced)

    assert len(result) == 2
    overridden = next(a for a in result if a.element_id == "w1")
    unchanged = next(a for a in result if a.element_id == "w2")

    assert overridden.template_key == "aw_ks30_wdvs160_mw"
    assert overridden.confidence == 1.0
    assert "User-Override" in overridden.rationale
    assert "aw_ks24_wdvs140_eps" in overridden.rationale  # previous key preserved in rationale

    assert unchanged.template_key == "aw_ks24_wdvs140_eps"
    assert unchanged.confidence == 0.9  # unchanged


def test_apply_forced_overrides_unknown_id_ignored():
    """Override for an element_id not in assignments is silently dropped."""
    assignments = [
        Assignment(
            element_id="w1",
            element_kind="WALL",
            template_key="aw_ks24_wdvs140_eps",
            confidence=0.8,
            rationale="auto",
        ),
    ]
    forced = {"NONEXISTENT": "aw_ks30_wdvs160_mw"}
    result = apply_forced_overrides(assignments, forced)

    assert len(result) == 1
    assert result[0].template_key == "aw_ks24_wdvs140_eps"


def test_template_assignment_step_reads_overrides_file(tmp_path):
    """TemplateAssignmentStep wendet manual_template_overrides.json neben dem Input an."""
    import json

    from planproof_cad.catalog.template_assignment import TemplateAssignmentStep

    # Write a fake input file and overrides file (sibling convention)
    input_file = tmp_path / "input_0.pdf"
    input_file.write_bytes(b"%PDF fake")
    overrides_file = tmp_path / "input_0.manual_template_overrides.json"
    overrides_file.write_text(
        json.dumps(
            {
                "version": 1,
                "overrides": {
                    "wall_001": {"template_key": "aw_ks30_wdvs160_mw", "template_version": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    wall = SimpleNamespace(
        id="wall_001",
        kind="WALL",
        center=(0, 1.5, 0),
        dims=(5.0, 3.0, 0.365),
        attrs={"wall_type": "exterior", "is_load_bearing": True},
    )
    scene = SimpleNamespace(components={"wall_001": wall})
    state = SimpleNamespace(
        scene=scene,
        config={},
        metrics={},
        input_path=str(input_file),
    )

    step = TemplateAssignmentStep()
    ctx = _fake_ctx()
    step.run(ctx, state)

    assert step._loaded, state.template_assignment
    assert state.template_assignment["ok"] is True

    assert state.template_assignments is not None
    a = next((x for x in state.template_assignments if x["element_id"] == "wall_001"), None)
    assert a is not None
    assert a["template_key"] == "aw_ks30_wdvs160_mw"
    assert a["confidence"] == 1.0
    assert "User-Override" in a["rationale"]
    assert any("Override" in m for m in ctx.logs)
