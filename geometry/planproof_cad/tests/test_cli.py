"""Tests der Kommandozeile ``planproof-cad``."""

from __future__ import annotations

import json

import pytest

from planproof_cad import __version__
from planproof_cad.cad.synth import build_demo_plan
from planproof_cad.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_demo_schreibt_alle_artefakte(tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["demo", str(out)]) == 0
    for name in (
        "plan.ifc",
        "model.glb",
        "scene.json",
        "quantity_takeoff.csv",
        "meta.json",
        "result.json",
    ):
        assert (out / name).exists(), name
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["template_assignment"]["ok"] is True
    assert "templates" in capsys.readouterr().out


def test_convert_auf_eigener_dxf(tmp_path):
    dxf = build_demo_plan(tmp_path / "eigen.dxf")
    out = tmp_path / "out"
    assert main(["convert", str(dxf), str(out), "--height", "3.0"]) == 0
    scene = json.loads((out / "scene.json").read_text(encoding="utf-8"))
    hoehen = {c["attrs"]["height_m"] for c in scene["components"].values() if c["kind"] == "WALL"}
    assert hoehen == {3.0}


def test_convert_fehlende_eingabe(tmp_path, capsys):
    assert main(["convert", str(tmp_path / "gibt_es_nicht.dxf"), str(tmp_path / "out")]) == 2
    assert "nicht gefunden" in capsys.readouterr().err


def test_unterbefehl_ist_pflicht():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
