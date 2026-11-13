"""Das gebaute Wheel traegt Mengen, Kostengruppen-Regeln und Schnittstellen.

``built_wheel`` und ``wheel_venv`` kommen aus conftest.py.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

from conftest import venv_python, venv_script
from mengen_support import HAUS_EG

ERWARTET = {
    "planproof_cad/kg/kg_rules.json",
    "planproof_cad/kg/zuordnung.py",
    "planproof_cad/kg/__init__.py",
    "planproof_cad/quantities/takeoff.py",
    "planproof_cad/quantities/writers.py",
    "planproof_cad/services.py",
    "planproof_cad/api.py",
    "planproof_cad/rest.py",
    "planproof_cad/mcp_stdio.py",
    "planproof_cad/render.py",
}


def _planproof-cad(wheel_venv: Path, tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(venv_script(wheel_venv, "planproof-cad")), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(tmp_path),
    )


def test_wheel_enthaelt_kg_rules_und_module(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as zf:
        namen = set(zf.namelist())
    fehlt = ERWARTET - namen
    assert not fehlt, fehlt
    assert not [n for n in namen if "/tests/" in n or n.startswith("tests/")]
    assert not [n for n in namen if n.endswith((".mjs", "scene_haus_eg.json"))]


def test_wheel_rules_gleich_eingebettete_konstante(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as zf:
        aus_wheel = json.loads(zf.read("planproof_cad/kg/kg_rules.json").decode("utf-8"))
    from planproof_cad.kg.zuordnung import KG_RULES

    assert aus_wheel == KG_RULES


def test_wheel_im_frischen_venv_mengen_kg_und_regeln(wheel_venv: Path, tmp_path: Path) -> None:
    py = venv_python(wheel_venv)
    code = (
        "from planproof_cad.kg import regeln_aus_datei, KG_RULES;"
        "assert regeln_aus_datei() == KG_RULES;"
        "print(len(KG_RULES['rules']))"
    )
    proc = subprocess.run(
        [str(py), "-I", "-c", code], capture_output=True, text=True, cwd=str(tmp_path), check=True
    )
    assert int(proc.stdout.strip()) == 10

    ausgabe = tmp_path / "out"
    proc = _planproof-cad(
        wheel_venv, tmp_path, "kg", str(HAUS_EG), "--geschoss", "EG", "--json", "-o", str(ausgabe)
    )
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(proc.stdout)
    assert {p["kg"] for p in doc["proposals"]} >= {"330", "331", "332", "340", "341", "342"}
    assert (ausgabe / "kg_vorschlaege.json").is_file()

    proc = _planproof-cad(wheel_venv, tmp_path, "takeoff", str(HAUS_EG), "-o", str(ausgabe))
    assert proc.returncode == 0, proc.stderr
    assert (ausgabe / "mengen_de.csv").read_bytes().startswith(b"\xef\xbb\xbfKategorie;")


def test_wheel_ohne_rest_extra_meldet_den_fehlenden_extra(wheel_venv: Path, tmp_path: Path) -> None:
    proc = _planproof-cad(wheel_venv, tmp_path, "serve-rest")
    assert proc.returncode == 2
    assert "rest" in proc.stderr
