"""Kommandozeile: takeoff, kg, layers, convert --json, demo --json, Fehlerwege."""

from __future__ import annotations

import json

import pytest
from mengen_support import HAUS_EG, HAUS_OG

from planproof_cad.cad.synth import build_demo_plan
from planproof_cad.cli import main


@pytest.fixture(scope="module")
def demo_dxf(tmp_path_factory):
    return build_demo_plan(tmp_path_factory.mktemp("dxf") / "demo.dxf")


def _json(capsys):
    out = capsys.readouterr().out
    return json.loads(out)


# --- takeoff --------------------------------------------------------------------------------


def test_takeoff_lesbar(capsys):
    assert main(["takeoff", str(HAUS_EG), "--geschoss", "EG"]) == 0
    out = capsys.readouterr().out
    assert "Mengen, Geschoss EG" in out
    assert "Außenwand tragend" in out and "43,50 m2" in out
    assert "Räume (Bezugsfläche, keine Kostengruppe)" in out
    assert "Nicht abgedeckt: Stützen, Treppen, Decken, Dächer" in out
    assert "Wandhöhe 2,8 m" in out


def test_takeoff_json_ist_reines_json(capsys):
    assert main(["takeoff", str(HAUS_EG), "--geschoss", "EG", "--json"]) == 0
    doc = _json(capsys)
    assert doc["schema"] == "planproof_cad.mengen" and doc["geschoss"] == "EG"
    assert doc["waende"]["summe"]["anzahl_kanten"] == 8


def test_takeoff_hoehe_und_ausgabeordner(tmp_path, capsys):
    assert main(["takeoff", str(HAUS_EG), "--height", "3.0", "-o", str(tmp_path)]) == 0
    assert (tmp_path / "mengen.json").is_file() and (tmp_path / "mengen_de.csv").is_file()
    m = json.loads((tmp_path / "mengen.json").read_text(encoding="utf-8"))
    a = next(x for x in m["annahmen"] if x["id"] == "wandhoehe")
    assert a["art"] == "parameter" and a["wert"] == 3.0


def test_takeoff_zwei_geschosse(tmp_path, capsys):
    argv = ["takeoff", str(HAUS_EG), str(HAUS_OG), "--geschoss", "EG", "--geschoss", "OG"]
    assert main([*argv, "-o", str(tmp_path), "--json"]) == 0
    doc = _json(capsys)
    assert [m["geschoss"] for m in doc["geschosse"]] == ["EG", "OG"]
    for name in ("mengen_EG.json", "mengen_OG.json", "mengen_de_EG.csv", "mengen_de_OG.csv"):
        assert (tmp_path / name).is_file(), name


def test_takeoff_geschoss_anzahl_stimmt_nicht(capsys):
    assert main(["takeoff", str(HAUS_EG), str(HAUS_OG), "--geschoss", "EG"]) == 2
    assert "genau eines je Eingabe" in capsys.readouterr().err
    assert main(["takeoff", str(HAUS_EG), str(HAUS_OG)]) == 2
    assert (
        main(["takeoff", str(HAUS_EG), str(HAUS_OG), "--geschoss", "EG", "--geschoss", "EG"]) == 2
    )


def test_takeoff_fehlende_datei(tmp_path, capsys):
    assert main(["takeoff", str(tmp_path / "gibt_es_nicht.json")]) == 2
    assert "FEHLER" in capsys.readouterr().err
    assert main(["takeoff", str(tmp_path / "gibt_es_nicht.json"), "--json"]) == 2
    doc = _json(capsys)
    assert doc["ok"] is False and doc["error"]["code"] == "scene_unreadable"


def test_takeoff_aus_dxf_braucht_ausgabeordner(demo_dxf, tmp_path, capsys):
    assert main(["takeoff", str(demo_dxf)]) == 2
    assert main(["takeoff", str(demo_dxf), "-o", str(tmp_path / "o"), "--geschoss", "EG"]) == 0
    assert (tmp_path / "o" / "mengen.json").is_file()
    assert (tmp_path / "o" / "scene.json").is_file()


# --- kg ---------------------------------------------------------------------------------------


def test_kg_lesbar_enthaelt_beispielzeilen(capsys):
    assert main(["kg", str(HAUS_EG), "--geschoss", "EG"]) == 0
    out = capsys.readouterr().out
    assert "Vorschläge zur Freigabe" in out
    assert "Außenwände EG 14 m2 netto, KG 330, Vorschlag zur Freigabe" in out
    assert "Außenfenster EG 3 Stk, KG 334, Vorschlag zur Freigabe" in out
    assert "[Konfidenz" in out
    assert "Bezugsfläche Räume" in out and "keine Kostengruppe" in out
    assert "Ohne Vorschlag" in out


def test_kg_json_und_datei(tmp_path, capsys):
    assert main(["kg", str(HAUS_EG), "--geschoss", "EG", "--json", "-o", str(tmp_path)]) == 0
    doc = _json(capsys)
    assert doc["schema"] == "planproof_cad.kg_vorschlaege"
    assert all(p["status"] == "vorschlag" for p in doc["proposals"])
    assert json.loads((tmp_path / "kg_vorschlaege.json").read_text(encoding="utf-8")) == doc


def test_kg_aus_mengen_json(tmp_path, capsys):
    assert main(["takeoff", str(HAUS_EG), "--geschoss", "EG", "-o", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["kg", str(tmp_path / "mengen.json"), "--json"]) == 0
    assert _json(capsys)["geschoss"] == "EG"


def test_kg_zwei_geschosse_mit_summen(capsys):
    argv = ["kg", str(HAUS_EG), str(HAUS_OG), "--geschoss", "EG", "--geschoss", "OG"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "Summen über alle Geschosse" in out
    assert main([*argv, "--json"]) == 0
    assert _json(capsys)["geschosse"] == ["EG", "OG"]


# --- layers -------------------------------------------------------------------------------------


def test_layers_lesbar_und_json(demo_dxf, capsys):
    assert main(["layers", str(demo_dxf)]) == 0
    out = capsys.readouterr().out
    assert "A_Wand_a" in out and "wall_exterior" in out
    assert "Wand-Layer: A_Wand_a, A_Wand_i; Wandlinien: 12 von 12; Tür-/Fensterblöcke: 6" in out
    assert main(["layers", str(demo_dxf), "--json"]) == 0
    assert _json(capsys)["summary"]["opening_blocks"] == 6


def test_layers_fehler(tmp_path, capsys):
    assert main(["layers", str(tmp_path / "fehlt.dxf")]) == 2
    assert "nicht gefunden" in capsys.readouterr().err


# --- convert / demo mit --json --------------------------------------------------------------------


def test_convert_json_stdout_ist_reines_json(demo_dxf, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["convert", str(demo_dxf), str(out), "--json", "--geschoss", "EG"]) == 0
    captured = capsys.readouterr()
    doc = json.loads(captured.out)
    assert doc["ok"] is True and doc["geschoss"] == "EG"
    assert doc["mengen"]["geschoss"] == "EG"
    assert doc["kg_vorschlaege"]["proposals"]
    assert "[log]" in captured.err  # Log geht bei --json nach stderr
    for name in ("mengen.json", "mengen_de.csv", "kg_vorschlaege.json", "quantity_takeoff.csv"):
        assert (out / name).is_file(), name


def test_convert_lesbar_zeigt_vorschlaege(demo_dxf, tmp_path, capsys):
    assert main(["convert", str(demo_dxf), str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "templates" in out
    assert "Kostengruppen-Vorschläge" in out and "Vorschlag zur Freigabe" in out


def test_demo_json(tmp_path, capsys):
    assert main(["demo", str(tmp_path / "d"), "--json"]) == 0
    doc = _json(capsys)
    assert doc["ok"] is True and doc["input"] == "demo.dxf"


# --- Verdrahtung ------------------------------------------------------------------------


def test_alle_unterbefehle_haben_hilfe(capsys):
    for befehl in ("demo", "convert", "takeoff", "kg", "layers", "serve-rest", "serve-mcp"):
        with pytest.raises(SystemExit) as e:
            main([befehl, "--help"])
        assert e.value.code == 0, befehl
        capsys.readouterr()


def test_serve_mcp_braucht_out_dir():
    with pytest.raises(SystemExit) as e:
        main(["serve-mcp"])
    assert e.value.code == 2


def test_serve_rest_ohne_extra_meldet_sich(monkeypatch, capsys):
    import builtins

    echt = builtins.__import__

    def sperre(name, *a, **k):
        if name == "uvicorn":
            raise ImportError("gesperrt")
        return echt(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", sperre)
    assert main(["serve-rest"]) == 2
    assert "Extra 'rest'" in capsys.readouterr().err
