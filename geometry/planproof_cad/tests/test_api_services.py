"""Dienste hinter CLI, REST und MCP: takeoff, assign_kg, list_layers, run_dxf, Pfadschutz."""

from __future__ import annotations

import json

import pytest
from mengen_support import DEMO_SCENE, HAUS_EG, lade

from planproof_cad import api
from planproof_cad.cad.synth import build_demo_plan
from planproof_cad.services import ServiceError, geschoss_dateisuffix, resolve_inside


@pytest.fixture(scope="module")
def demo_dxf(tmp_path_factory):
    return build_demo_plan(tmp_path_factory.mktemp("dxf") / "demo.dxf")


@pytest.fixture(scope="module")
def demo_lauf(tmp_path_factory, demo_dxf):
    out = tmp_path_factory.mktemp("lauf")
    return out, api.run_dxf(demo_dxf, out, {"wall_height_m": 2.8}, geschoss="EG")


# --- run_dxf ----------------------------------------------------------------------------------


def test_run_dxf_schreibt_alle_dateien(demo_lauf):
    out, res = demo_lauf
    assert res["ok"] is True and res["error"] is None
    for name in (
        "plan.ifc",
        "model.glb",
        "scene.json",
        "quantity_takeoff.csv",  # englische Pipeline-Datei bleibt bestehen
        "meta.json",
        "result.json",
        "mengen.json",
        "mengen_de.csv",
        "kg_vorschlaege.json",
    ):
        assert (out / name).is_file(), name
    # Die englische CSV ist unveraendert englisch.
    kopf = (out / "quantity_takeoff.csv").read_text(encoding="utf-8").splitlines()[0]
    assert kopf.startswith("id,kind,wall_type,length_m")
    assert res["mengen"]["geschoss"] == "EG"
    assert res["kg_vorschlaege"]["proposals"]
    assert json.loads((out / "mengen.json").read_text(encoding="utf-8")) == res["mengen"]


def test_run_dxf_mengen_aus_lauf_gleich_takeoff_der_scene(demo_lauf):
    out, res = demo_lauf
    nochmal = api.takeoff(out / "scene.json", geschoss="EG")
    assert nochmal["waende"] == res["mengen"]["waende"]
    assert nochmal["oeffnungen"] == res["mengen"]["oeffnungen"]


def test_run_dxf_ohne_mengen(tmp_path, demo_dxf):
    res = api.run_dxf(demo_dxf, tmp_path / "o", mengen=False)
    assert res["ok"] and res["mengen"] is None
    assert not (tmp_path / "o" / "mengen.json").exists()


def test_run_dxf_fehler(tmp_path):
    with pytest.raises(ServiceError) as e:
        api.run_dxf(tmp_path / "gibt_es_nicht.dxf", tmp_path / "o")
    assert e.value.code == "input_not_found"
    txt = tmp_path / "x.txt"
    txt.write_text("x", encoding="utf-8")
    with pytest.raises(ServiceError) as e:
        api.run_dxf(txt, tmp_path / "o")
    assert e.value.code == "unsupported_input"
    with pytest.raises(ServiceError) as e:
        api.run_dxf(txt.with_suffix(".dxf"), tmp_path / "o")  # existiert nicht
    assert e.value.code == "input_not_found"


def test_run_dxf_ungueltige_hoehe(tmp_path, demo_dxf):
    with pytest.raises(ServiceError) as e:
        api.run_dxf(demo_dxf, tmp_path / "o", {"wall_height_m": 99})
    assert e.value.code == "invalid_wall_height"


# --- takeoff ------------------------------------------------------------------------------------


def test_takeoff_aus_dict_pfad_und_modell():
    s = lade(HAUS_EG)
    a = api.takeoff(s, geschoss="EG")
    b = api.takeoff(HAUS_EG, geschoss="EG")
    assert a == b
    from planproof_cad.model.scene_model import SceneModel

    c = api.takeoff(SceneModel.from_dict(s), geschoss="EG")
    assert c["waende"]["summe"] == a["waende"]["summe"]


def test_takeoff_schreibt_dateien_mit_out_dir(tmp_path):
    api.takeoff(HAUS_EG, geschoss="EG", out_dir=tmp_path)
    assert (tmp_path / "mengen.json").is_file()
    assert (tmp_path / "mengen_de.csv").is_file()


def test_takeoff_wandhoehe_parameter():
    m = api.takeoff(HAUS_EG, wall_height_m=3.0)
    a = next(x for x in m["annahmen"] if x["id"] == "wandhoehe")
    assert a["wert"] == 3.0 and a["art"] == "parameter"
    with pytest.raises(ServiceError) as e:
        api.takeoff(HAUS_EG, wall_height_m=0.1)
    assert e.value.code == "invalid_wall_height"


def test_takeoff_aus_dxf_braucht_ausgabeordner(demo_dxf, tmp_path):
    with pytest.raises(ServiceError) as e:
        api.takeoff(demo_dxf)
    assert e.value.code == "out_dir_required"
    m = api.takeoff(demo_dxf, geschoss="EG", out_dir=tmp_path / "o")
    assert (tmp_path / "o" / "scene.json").is_file()
    assert (tmp_path / "o" / "mengen.json").is_file()
    assert m["waende"]["summe"]["anzahl_kanten"] == 10


def test_takeoff_fehlerfaelle(tmp_path):
    with pytest.raises(ServiceError) as e:
        api.takeoff(tmp_path / "fehlt.json")
    assert e.value.code == "scene_unreadable"
    kaputt = tmp_path / "kaputt.json"
    kaputt.write_text("{nein", encoding="utf-8")
    with pytest.raises(ServiceError) as e:
        api.takeoff(kaputt)
    assert e.value.code == "scene_unreadable"
    keine_szene = tmp_path / "x.json"
    keine_szene.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ServiceError) as e:
        api.takeoff(keine_szene)
    assert e.value.code == "scene_unreadable"
    with pytest.raises(ServiceError) as e:
        api.takeoff({"a": 1})
    assert e.value.code == "not_a_scene"
    with pytest.raises(ServiceError) as e:
        api.takeoff(42)
    assert e.value.code == "unsupported_input"


def test_takeoff_liest_meta_neben_der_scene(tmp_path):
    import shutil

    shutil.copy(DEMO_SCENE, tmp_path / "scene.json")
    meta = json.loads((DEMO_SCENE.with_name("meta.json")).read_text(encoding="utf-8"))
    meta["metrics"]["openings"]["unmatched"] = 3
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    m = api.takeoff(tmp_path / "scene.json")
    assert any(h["code"] == "PIPELINE_OEFFNUNG_NICHT_ANGEHAENGT" for h in m["hinweise"])


# --- assign_kg ------------------------------------------------------------------------------------


def test_assign_kg_akzeptiert_mengen_scene_pfad_und_dxf(tmp_path, demo_dxf):
    mengen = api.takeoff(HAUS_EG, geschoss="EG")
    a = api.assign_kg(mengen)
    b = api.assign_kg(lade(HAUS_EG), geschoss="EG")
    c = api.assign_kg(HAUS_EG, geschoss="EG")
    assert a == b == c
    mpfad = tmp_path / "mengen.json"
    mpfad.write_text(json.dumps(mengen), encoding="utf-8")
    assert api.assign_kg(mpfad) == a
    d = api.assign_kg(demo_dxf, geschoss="EG", out_dir=tmp_path / "dxf")
    assert (tmp_path / "dxf" / "kg_vorschlaege_EG.json").is_file()
    assert {p["kg"] for p in d["proposals"]} == {"330", "334", "340", "344"}


def test_assign_kg_liste_mit_summen():
    eg = api.takeoff(HAUS_EG, geschoss="EG")
    og = api.takeoff(HAUS_EG, geschoss="OG")
    doc = api.assign_kg([eg, og])
    assert doc["geschosse"] == ["EG", "OG"]
    s = {(x["kg"], x["element_class"]): x for x in doc["summen"]}
    assert s[("334", "Aussentuer")]["quantity"] == 2
    assert api.assign_kg([eg])["geschoss"] == "EG"
    with pytest.raises(ServiceError) as e:
        api.assign_kg([])
    assert e.value.code == "empty_input"


def test_assign_kg_schreibt_datei(tmp_path):
    api.assign_kg(HAUS_EG, geschoss="EG", out_dir=tmp_path)
    doc = json.loads((tmp_path / "kg_vorschlaege_EG.json").read_text(encoding="utf-8"))
    assert doc["schema"] == "planproof_cad.kg_vorschlaege"
    assert all(p["status"] == "vorschlag" for p in doc["proposals"])


def test_assign_kg_fehlerfaelle(tmp_path):
    kaputt = tmp_path / "m.json"
    kaputt.write_text("{", encoding="utf-8")
    with pytest.raises(ServiceError) as e:
        api.assign_kg(kaputt)
    assert e.value.code == "input_unreadable"
    with pytest.raises(ServiceError) as e:
        api.assign_kg({"schema": "planproof_cad.mengen", "waende": 1})
    assert e.value.code == "not_mengen"


# --- list_layers ------------------------------------------------------------------------


def test_list_layers_demo(demo_dxf):
    res = api.list_layers(demo_dxf)
    layer = {la["name"]: la for la in res["layers"]}
    assert layer["A_Wand_a"]["class"] == "wall_exterior"
    assert layer["A_Wand_a"]["read_as"] == "wand"
    assert layer["A_Wand_a"]["entities"]["lines"] == 8
    assert layer["A_Wand_i"]["class"] == "wall_interior"
    assert layer["A_Wand_i"]["entities"]["lines"] == 4
    assert layer["A_Tuer"]["read_as"] == "oeffnung_block"
    assert layer["A_Tuer"]["entities"]["inserts"] == 3
    assert layer["A_Fenster"]["entities"]["inserts"] == 3
    assert layer["A_Raum"]["entities"]["texts"] == 3
    assert layer["0"]["class"] == "unknown"
    s = res["summary"]
    assert s["wall_layers"] == ["A_Wand_a", "A_Wand_i"]
    assert s["wall_line_entities"] == 12 == s["all_line_entities"]
    assert s["fallback_all_lines_as_walls"] is False
    assert s["opening_blocks"] == 6
    assert res["hinweise"] == []
    bloecke = {b["name"]: b for b in res["blocks"]}
    assert bloecke["Tuer_1010"]["count"] == 2 and bloecke["Tuer_1010"]["class"] == "door"


def test_list_layers_rueckfall_ohne_wand_layer(tmp_path):
    import ezdxf

    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    for a, b in [((0, 0), (5000, 0)), ((5000, 0), (5000, 4000))]:
        msp.add_line(a, b, dxfattribs={"layer": "Linien_1"})
    pfad = tmp_path / "frei.dxf"
    doc.saveas(pfad)
    res = api.list_layers(pfad)
    assert res["summary"]["fallback_all_lines_as_walls"] is True
    assert res["summary"]["wall_line_entities"] == 0
    assert any("Rückfall" in h or "alle Linien" in h for h in res["hinweise"])
    assert any("Tür oder Fenster" in h for h in res["hinweise"])


def test_list_layers_fehler(tmp_path):
    with pytest.raises(ServiceError) as e:
        api.list_layers(tmp_path / "fehlt.dxf")
    assert e.value.code == "input_not_found"
    kaputt = tmp_path / "k.dxf"
    kaputt.write_text("das ist keine dxf", encoding="utf-8")
    with pytest.raises(ServiceError) as e:
        api.list_layers(kaputt)
    assert e.value.code == "dxf_unreadable"


# --- Pfadschutz -------------------------------------------------------------------------


def test_resolve_inside(tmp_path):
    wurzel = tmp_path / "work"
    (wurzel / "sub").mkdir(parents=True)
    ok = resolve_inside("sub/a.dxf", [wurzel])
    assert ok == (wurzel / "sub" / "a.dxf").resolve()
    assert resolve_inside(str(wurzel / "sub"), [wurzel]) == (wurzel / "sub").resolve()
    for boese in ("../x.dxf", "sub/../../x.dxf", str(tmp_path / "anders.dxf"), "..\\x.dxf"):
        with pytest.raises(ServiceError) as e:
            resolve_inside(boese, [wurzel])
        assert e.value.code == "path_outside_root", boese
    with pytest.raises(ServiceError) as e:
        resolve_inside("", [wurzel])
    assert e.value.code == "path_invalid"
    with pytest.raises(ServiceError):
        resolve_inside("a", [])


def test_resolve_inside_nachbarordner_mit_gleichem_praefix(tmp_path):
    """/work und /work2: ein String-Praefix-Vergleich wuerde /work2 durchlassen."""
    (tmp_path / "work").mkdir()
    (tmp_path / "work2").mkdir()
    with pytest.raises(ServiceError):
        resolve_inside(str(tmp_path / "work2" / "a.dxf"), [tmp_path / "work"])


def _verzeichnis_link(link, ziel) -> None:
    """Symlink, unter Windows ersatzweise eine Junction (braucht keine Adminrechte)."""
    import subprocess
    import sys

    try:
        link.symlink_to(ziel, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if sys.platform == "win32":
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(ziel)], capture_output=True
        )
        if proc.returncode == 0:
            return
    pytest.skip("weder Symlink noch Junction anlegbar")


def test_resolve_inside_symlink_nach_draussen(tmp_path):
    wurzel = tmp_path / "work"
    aussen = tmp_path / "aussen"
    wurzel.mkdir()
    aussen.mkdir()
    (aussen / "geheim.dxf").write_text("x", encoding="utf-8")
    _verzeichnis_link(wurzel / "link", aussen)
    with pytest.raises(ServiceError) as e:
        resolve_inside("link/geheim.dxf", [wurzel])
    assert e.value.code == "path_outside_root"


def test_geschoss_dateisuffix():
    assert geschoss_dateisuffix("EG") == "_EG"
    assert geschoss_dateisuffix("1. OG/Süd") == "_1._OG_S_d"
    assert geschoss_dateisuffix(None) == ""
    assert geschoss_dateisuffix("///") == ""
