"""Importverbot fuer die Befehle ``takeoff``, ``kg``, ``layers``, die MCP-Nachrichten und die
REST-Anwendung: kein torch, cv2, scipy usw. in ``sys.modules`` (Liste wie in test_importverbot).

Dazu: ``takeoff`` und ``kg`` auf einer scene.json laden weder ezdxf noch ifcopenshell noch
shapely noch numpy. Wer nur Mengen braucht, bezahlt die Pipeline nicht.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from mengen_support import HAUS_EG
from test_importverbot import FORBIDDEN

SCHWER = ["ezdxf", "ifcopenshell", "shapely", "numpy"]

RUNNER = r"""
import importlib.abc
import json
import sys

FORBIDDEN = {forbidden!r}


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in FORBIDDEN:
            raise ImportError("gesperrt durch test_importverbot_mengen: " + fullname)
        return None


sys.meta_path.insert(0, _Blocker())

import io
from contextlib import redirect_stdout

{body}

wurzeln = lambda ns: sorted({{n.split(".", 1)[0] for n in sys.modules}} & set(ns))
report = {{
    "rc": rc,
    "verboten": wurzeln(FORBIDDEN),
    "schwer": wurzeln({schwer!r}),
    "fastapi": "fastapi" in sys.modules,
}}
with open({report!r}, "w", encoding="utf-8") as fh:
    json.dump(report, fh)
"""


def _lauf(tmp_path: Path, body: str, forbidden: list[str] | None = None) -> dict:
    report = tmp_path / "report.json"
    script = tmp_path / "runner.py"
    script.write_text(
        RUNNER.format(
            forbidden=set(forbidden if forbidden is not None else FORBIDDEN),
            body=body,
            schwer=SCHWER,
            report=str(report),
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-I", str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return json.loads(report.read_text(encoding="utf-8"))


def test_takeoff_und_kg_laden_nur_die_stdlib(tmp_path):
    body = (
        "from planproof_cad.cli import main\n"
        "with redirect_stdout(io.StringIO()):\n"
        f"    rc = main(['takeoff', {str(HAUS_EG)!r}, '--geschoss', 'EG', '--json'])\n"
        f"    rc2 = main(['kg', {str(HAUS_EG)!r}, '--geschoss', 'EG', '--json', "
        f"'-o', {str(tmp_path / 'o')!r}])\n"
        "rc = rc or rc2\n"
    )
    data = _lauf(tmp_path, body)
    assert data["rc"] == 0
    assert data["verboten"] == []
    assert data["schwer"] == [], "takeoff/kg haben die Pipeline geladen"
    assert (tmp_path / "o" / "kg_vorschlaege.json").is_file()


def test_layers_und_convert_ohne_verbotene_module(tmp_path):
    from planproof_cad.cad.synth import build_demo_plan

    dxf = build_demo_plan(tmp_path / "demo.dxf")
    body = (
        "from planproof_cad.cli import main\n"
        "with redirect_stdout(io.StringIO()):\n"
        f"    rc = main(['layers', {str(dxf)!r}, '--json'])\n"
        f"    rc2 = main(['convert', {str(dxf)!r}, {str(tmp_path / 'o')!r}, '--json'])\n"
        "rc = rc or rc2\n"
    )
    data = _lauf(tmp_path, body)
    assert data["rc"] == 0
    assert data["verboten"] == []
    assert (tmp_path / "o" / "mengen.json").is_file()


def test_mcp_nachrichten_ohne_verbotene_module(tmp_path):
    (tmp_path / "in").mkdir()
    szene = tmp_path / "in" / "scene.json"
    szene.write_text(HAUS_EG.read_text(encoding="utf-8"), encoding="utf-8")
    aufruf = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "quantity_takeoff", "arguments": {"path": str(szene)}},
    }
    body = (
        "from planproof_cad.mcp_stdio import McpServer\n"
        f"s = McpServer({str(tmp_path / 'out')!r}, [{str(tmp_path / 'in')!r}])\n"
        "s.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {}})\n"
        f"r = s.handle({aufruf!r})\n"
        "rc = 0 if r['result']['isError'] is False else 1\n"
    )
    data = _lauf(tmp_path, body)
    assert data["rc"] == 0
    assert data["verboten"] == [] and data["schwer"] == []


def test_rest_anwendung_ohne_verbotene_module(tmp_path):
    pytest.importorskip("fastapi")
    # httpx steht in FORBIDDEN; create_app braucht es nicht.
    body = (
        "from planproof_cad.rest import create_app\n"
        f"app = create_app(workdir={str(tmp_path / 'werk')!r})\n"
        "rc = 0 if any(getattr(r, 'path', '') == '/healthz' for r in app.routes) else 1\n"
    )
    data = _lauf(tmp_path, body)
    assert data["rc"] == 0
    assert data["fastapi"] is True
    assert data["verboten"] == []


def test_blocker_wirkt_wirklich(tmp_path):
    """Gegenprobe: sperrt man ein benoetigtes Modul, scheitert der Lauf."""
    report = tmp_path / "report.json"
    script = tmp_path / "runner.py"
    script.write_text(
        RUNNER.format(
            forbidden={"argparse"},
            body="from planproof_cad.cli import main\nrc = 0\n",
            schwer=SCHWER,
            report=str(report),
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-I", str(script)], capture_output=True, text=True, cwd=str(tmp_path)
    )
    assert proc.returncode != 0
