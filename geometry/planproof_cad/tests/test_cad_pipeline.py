"""Tests fuer die DWG/DXF -> BIM-Pipeline.

Nutzt den synthetischen DXF-Generator als Fixture -- kein externes ODA/LibreDWG
und keine echte Datei noetig. Deckt jede Stufe einzeln ab plus einen
End-to-End-Lauf, der die IFC-Inhalte (Spatial-Tree, parametrische Oeffnungen,
Raeume) und das GLB validiert.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from planproof_cad.cad import layer_classifier as lc
from planproof_cad.cad import run_dwg_to_bim_pipeline
from planproof_cad.cad.dxf_loader import load_and_extract
from planproof_cad.cad.opening_extraction import extract_openings
from planproof_cad.cad.room_detection import detect_rooms
from planproof_cad.cad.synth import build_demo_plan
from planproof_cad.cad.wall_reconstruction import (
    build_wall_graph,
    reconstruct_walls,
)


@pytest.fixture()
def demo_dxf(tmp_path) -> Path:
    return build_demo_plan(tmp_path / "demo.dxf")


# ── layer_classifier ─────────────────────────────────────────────────────


class TestLayerClassifier:
    def test_aia_layers(self):
        assert lc.classify_layer("A-WALL").cls == lc.WALL
        assert lc.classify_layer("A-DOOR").cls == lc.DOOR
        assert lc.classify_layer("A-GLAZ").cls == lc.WINDOW  # AIA: Verglasung = Fenster
        assert lc.classify_layer("A-COLS").cls == lc.COLUMN

    def test_dach_layers(self):
        assert lc.classify_layer("A_Wand_a").cls == lc.WALL_EXTERIOR
        assert lc.classify_layer("A_Wand_i").cls == lc.WALL_INTERIOR
        assert lc.classify_layer("A_Tuer_i").cls == lc.DOOR
        assert lc.classify_layer("A_Fenster").cls == lc.WINDOW

    def test_umlaut_both_forms(self):
        # Echte Umlaute UND ASCII-gefaltete Form muessen treffen
        assert lc.classify_layer("Türen").cls == lc.DOOR
        assert lc.classify_layer("Tueren").cls == lc.DOOR
        assert lc.classify_layer("Stütze").cls == lc.COLUMN
        assert lc.classify_layer("Stuetze").cls == lc.COLUMN
        assert lc.classify_layer("Möbel").cls == lc.FURNITURE

    def test_annotation_filtered(self):
        assert lc.classify_layer("A-DIMS").cls == lc.ANNOTATION
        assert lc.classify_layer("Bemassung").cls == lc.ANNOTATION

    def test_block_overrides_layer(self):
        # ArchiCAD-Fall: Tuer-Block auf Wand-Layer -> Block gewinnt
        c = lc.classify_block("Tuer_1010", "A_Wand_i")
        assert c.cls == lc.DOOR

    def test_unknown(self):
        assert lc.classify_layer("RandomLayer42").cls == lc.UNKNOWN

    def test_template_key_match(self):
        keys = ["iw_ks240_tragend", "aw_beton300_wdvs160_mw"]
        assert lc.match_template_key("A_Wand_KS240_tragend", keys) == "iw_ks240_tragend"


# ── dxf_loader ───────────────────────────────────────────────────────────


class TestDxfLoader:
    def test_units_and_counts(self, demo_dxf):
        ex = load_and_extract(demo_dxf, {})
        assert ex.unit_label == "mm"
        assert ex.unit_scale == pytest.approx(0.001)
        # 6 Doppellinien-Waende = 12 Linien
        assert len(ex.segments) == 12
        # 3 Tueren + 3 Fenster
        assert len(ex.inserts) == 6
        assert len(ex.texts) == 3
        # Bbox ~ 10.3 x 8.3 m (inkl. Wanddicke-Ueberstand)
        w = ex.bbox[1][0] - ex.bbox[0][0]
        h = ex.bbox[1][1] - ex.bbox[0][1]
        assert 10.0 <= w <= 11.0
        assert 8.0 <= h <= 9.0

    def test_forced_unit_override(self, demo_dxf):
        ex = load_and_extract(demo_dxf, {"dxf_unit": "cm"})
        assert ex.unit_scale == pytest.approx(0.01)

    def test_missing_file(self, tmp_path):
        from planproof_cad.cad.dxf_loader import DxfLoadError

        with pytest.raises(DxfLoadError):
            load_and_extract(tmp_path / "nope.dxf", {})


# ── wall_reconstruction ──────────────────────────────────────────────────


class TestWallReconstruction:
    def test_double_line_pairing(self, demo_dxf):
        ex = load_and_extract(demo_dxf, {})
        walls = [s for s in ex.segments if lc.classify_layer(s.layer).cls in lc.WALL_CLASSES]
        cls, stats = reconstruct_walls(walls, {})
        assert stats["mode"] == "double_line"
        assert stats["centerlines"] == 6
        assert stats["matched_ratio"] == pytest.approx(1.0, abs=0.01)
        # Dicken: exterior 0.30, interior 0.115
        thicks = sorted(round(c.thickness_m, 3) for c in cls)
        assert thicks.count(0.3) == 4
        assert thicks.count(0.115) == 2

    def test_graph_junction_splitting(self, demo_dxf):
        ex = load_and_extract(demo_dxf, {})
        walls = [s for s in ex.segments if lc.classify_layer(s.layer).cls in lc.WALL_CLASSES]
        cls, _ = reconstruct_walls(walls, {})
        graph = build_wall_graph(cls, {})
        # 6 Achsen werden an T-Stoessen geteilt -> mehr Kanten
        assert len(graph.edges) >= 8
        # alle Knoten sind mit mind. einer Kante verbunden
        assert graph.prune_orphan_nodes() == 0

    def test_single_line_fallback(self):
        from planproof_cad.cad.dxf_loader import RawSegment

        # Einlinige Waende (keine Paare) -> single_line-Modus
        segs = [
            RawSegment((0, 0), (5, 0), "A-WALL"),
            RawSegment((5, 0), (5, 4), "A-WALL"),
            RawSegment((5, 4), (0, 4), "A-WALL"),
            RawSegment((0, 4), (0, 0), "A-WALL"),
        ]
        cls, stats = reconstruct_walls(segs, {})
        assert stats["mode"] == "single_line"
        assert stats["centerlines"] == 4


# ── opening_extraction ───────────────────────────────────────────────────


class TestOpeningExtraction:
    def test_openings_attached(self, demo_dxf):
        ex = load_and_extract(demo_dxf, {})
        walls = [s for s in ex.segments if lc.classify_layer(s.layer).cls in lc.WALL_CLASSES]
        cls, _ = reconstruct_walls(walls, {})
        graph = build_wall_graph(cls, {})
        inserts = [
            i for i in ex.inserts if lc.classify_block(i.name, i.layer).cls in ("door", "window")
        ]
        stats = extract_openings(graph, inserts, {})
        assert stats["openings_attached"] == 6
        assert stats["doors"] == 3
        assert stats["windows"] == 3
        assert stats["unmatched"] == 0
        # Oeffnungen tragen Rotation (Schwingrichtung) als Attribut
        all_ops = [o for e in graph.edges.values() for o in e.openings]
        assert len(all_ops) == 6
        assert all("rotation_deg" in o.attrs for o in all_ops)


# ── room_detection ───────────────────────────────────────────────────────


class TestRoomDetection:
    def test_three_labelled_rooms(self, demo_dxf):
        ex = load_and_extract(demo_dxf, {})
        walls = [s for s in ex.segments if lc.classify_layer(s.layer).cls in lc.WALL_CLASSES]
        cls, _ = reconstruct_walls(walls, {})
        graph = build_wall_graph(cls, {})
        rooms = detect_rooms(graph, ex.texts, {})
        assert len(rooms) == 3
        labels = {r["label"] for r in rooms}
        assert "Wohnzimmer" in labels
        assert "Küche" in labels
        assert "Bad" in labels
        # jeder Raum hat Polygon + plausible Flaeche
        for r in rooms:
            assert len(r["polygon"]) >= 4
            assert r["area_m2"] > 1.0


# ── End-to-End ───────────────────────────────────────────────────────────


class TestPipelineEndToEnd:
    def test_full_pipeline_ifc_and_glb(self, demo_dxf, tmp_path):
        out = tmp_path / "out"
        res = run_dwg_to_bim_pipeline(str(demo_dxf), str(out), str(tmp_path / "work"), {})
        assert res.ok, res.error
        assert res.ifc_path and Path(res.ifc_path).exists()
        assert res.model_path and Path(res.model_path).exists()

        # Artefakte vollstaendig
        names = {Path(a).name for a in res.artifacts}
        assert {
            "plan.ifc",
            "model.glb",
            "scene.json",
            "quantity_takeoff.csv",
            "meta.json",
        }.issubset(names)

        # --- IFC inhaltlich pruefen ---
        ifcopenshell = pytest.importorskip("ifcopenshell")
        f = ifcopenshell.open(res.ifc_path)
        assert f.schema == "IFC4"
        assert len(f.by_type("IfcProject")) == 1
        assert len(f.by_type("IfcBuildingStorey")) == 1
        assert len(f.by_type("IfcWallStandardCase")) >= 8
        assert len(f.by_type("IfcDoorStandardCase")) == 3
        assert len(f.by_type("IfcWindowStandardCase")) == 3
        assert len(f.by_type("IfcSpace")) == 3
        # parametrische Oeffnungs-Topologie
        assert len(f.by_type("IfcOpeningElement")) == 6
        assert len(f.by_type("IfcRelVoidsElement")) == 6
        assert len(f.by_type("IfcRelFillsElement")) == 6
        # DIN-Katalog-Material-Layer-Sets
        assert len(f.by_type("IfcMaterialLayerSet")) >= 1
        # Raumlabels (Umlaute erhalten)
        space_names = {s.Name for s in f.by_type("IfcSpace")}
        assert "Wohnzimmer" in space_names
        assert "Küche" in space_names
        # Waende achsen-rotiert (RefDirection gesetzt)
        rotated = sum(
            1
            for w in f.by_type("IfcWallStandardCase")
            if w.ObjectPlacement.RelativePlacement.RefDirection is not None
        )
        assert rotated == len(f.by_type("IfcWallStandardCase"))

        # --- GLB pruefen ---
        blob = Path(res.model_path).read_bytes()
        magic, ver, length = struct.unpack_from("<III", blob, 0)
        assert magic == 0x46546C67
        assert ver == 2
        assert length == len(blob)
        clen, ctype = struct.unpack_from("<II", blob, 12)
        assert ctype == 0x4E4F534A  # JSON-Chunk
        gltf = json.loads(blob[20 : 20 + clen])
        assert len(gltf.get("nodes", [])) >= 16

    def test_quantity_takeoff_has_rows(self, demo_dxf, tmp_path):
        out = tmp_path / "out"
        res = run_dwg_to_bim_pipeline(str(demo_dxf), str(out), str(tmp_path / "work"), {})
        assert res.ok
        csv_text = (out / "quantity_takeoff.csv").read_text(encoding="utf-8")
        lines = [ln for ln in csv_text.splitlines() if ln.strip()]
        assert len(lines) >= 17  # Header + >=16 Elemente
        assert "template_key" in lines[0]
