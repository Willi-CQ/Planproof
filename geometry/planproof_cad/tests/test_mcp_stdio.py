"""MCP-Server ueber stdio: echter Subprozess (Handshake, tools/list, tools/call) und
Protokollfaelle direkt am ``McpServer``."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from mengen_support import HAUS_EG

from planproof_cad.cad.synth import build_demo_plan
from planproof_cad.mcp_stdio import PROTOCOL_VERSION, TOOLS, McpServer

TOOL_NAMEN = {"dwg_convert", "quantity_takeoff", "assign_kg_proposal", "list_layers"}


class Client:
    """Startet ``python -m planproof_cad.mcp_stdio`` und spricht zeilenweise JSON-RPC."""

    def __init__(self, out_dir: Path, cwd: Path, extra: list[str] | None = None) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-I", "-m", "planproof_cad.mcp_stdio", "--out-dir", str(out_dir)]
            + (extra or []),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
        )
        self._zeilen: queue.Queue[bytes] = queue.Queue()
        self._thread = threading.Thread(target=self._lesen, daemon=True)
        self._thread.start()
        self._id = 0

    def _lesen(self) -> None:
        assert self.proc.stdout is not None
        for zeile in self.proc.stdout:
            self._zeilen.put(zeile)
        self._zeilen.put(b"")

    def senden_roh(self, zeile: bytes) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(zeile + b"\n")
        self.proc.stdin.flush()

    def senden(self, obj: dict) -> None:
        self.senden_roh(json.dumps(obj).encode("utf-8"))

    def antwort(self, timeout: float = 120.0) -> dict:
        zeile = self._zeilen.get(timeout=timeout)
        assert zeile, "Server hat stdout geschlossen"
        assert zeile.endswith(b"\n") and b"\n" not in zeile[:-1]
        return json.loads(zeile.decode("utf-8"))

    def anfrage(self, methode: str, params: dict | None = None) -> dict:
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": methode}
        if params is not None:
            msg["params"] = params
        self.senden(msg)
        a = self.antwort()
        assert a["id"] == self._id
        return a

    def werkzeug(self, name: str, **argumente) -> dict:
        return self.anfrage("tools/call", {"name": name, "arguments": argumente})

    def handshake(self) -> dict:
        a = self.anfrage(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        )
        self.senden({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return a

    def schliessen(self) -> tuple[int, str]:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        try:
            rc = self.proc.wait(timeout=30)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
        err = self.proc.stderr.read().decode("utf-8", errors="replace") if self.proc.stderr else ""
        return rc, err


def _baum(wurzel: Path) -> set[str]:
    return {p.relative_to(wurzel).as_posix() for p in wurzel.rglob("*")}


@pytest.fixture()
def raum(tmp_path):
    """tmp_path/proj: Arbeitsverzeichnis des Servers (enthaelt demo.dxf), tmp_path/out: Ausgabe."""
    proj = tmp_path / "proj"
    proj.mkdir()
    build_demo_plan(proj / "demo.dxf")
    (proj / "scene_haus.json").write_text(HAUS_EG.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "geheim.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_handshake_liste_und_aufruf_im_subprozess(raum):
    proj, out = raum / "proj", raum / "out"
    c = Client(out, proj)
    try:
        init = c.handshake()
        res = init["result"]
        assert init["jsonrpc"] == "2.0"
        assert res["protocolVersion"] == "2025-06-18"
        assert res["capabilities"] == {"tools": {"listChanged": False}}
        assert res["serverInfo"]["name"] == "planproof-cad"

        assert c.anfrage("ping")["result"] == {}

        liste = c.anfrage("tools/list")["result"]["tools"]
        assert {t["name"] for t in liste} == TOOL_NAMEN
        for t in liste:
            assert t["inputSchema"]["type"] == "object"
            assert t["inputSchema"]["additionalProperties"] is False
            assert t["description"]

        # dwg_convert auf die Demo
        a = c.werkzeug("dwg_convert", path="demo.dxf", geschoss="EG")
        ergebnis = a["result"]
        assert ergebnis["isError"] is False, ergebnis
        daten = ergebnis["structuredContent"]
        assert daten["ok"] is True
        assert {"plan.ifc", "model.glb", "scene.json", "mengen.json", "kg_vorschlaege.json"} <= set(
            daten["artifacts"]
        )
        assert Path(daten["output_dir"]) == (out / "demo").resolve()
        assert (out / "demo" / "plan.ifc").is_file()
        proposals = {p["element_class"]: p for p in daten["kg_vorschlaege"]["proposals"]}
        assert (
            proposals["Aussenwand"]["kg"] == "330"
            and proposals["Aussenwand"]["status"] == "vorschlag"
        )
        assert "elemente" not in daten["mengen"]
        # text-Inhalt ist dasselbe JSON
        text = ergebnis["content"][0]
        assert text["type"] == "text" and json.loads(text["text"]) == daten

        # takeoff auf die erzeugte scene.json (liegt im Ausgabeverzeichnis, ist Lese-Root)
        t = c.werkzeug("quantity_takeoff", path=str(out / "demo" / "scene.json"), geschoss="EG")
        assert t["result"]["isError"] is False
        assert t["result"]["structuredContent"]["waende"]["summe"]["anzahl_kanten"] == 10

        # Kostengruppen-Vorschlag aus der handgebauten Szene
        k = c.werkzeug("assign_kg_proposal", path="scene_haus.json", geschoss="EG")
        kd = k["result"]["structuredContent"]
        assert {p["kg"] for p in kd["proposals"]} >= {"331", "332", "341", "342", "334", "344"}
        assert (out / "scene_haus" / "kg_vorschlaege_EG.json").is_file()

        # Layer
        la = c.werkzeug("list_layers", path="demo.dxf")["result"]["structuredContent"]
        assert la["summary"]["wall_layers"] == ["A_Wand_a", "A_Wand_i"]
    finally:
        rc, err = c.schliessen()
    assert rc == 0, err


def test_schreibzugriff_nur_im_ausgabeverzeichnis(raum):
    proj, out = raum / "proj", raum / "out"
    vorher_proj = _baum(proj)
    vorher_root = {p.name for p in raum.iterdir()}
    c = Client(out, proj)
    try:
        c.handshake()
        assert c.werkzeug("dwg_convert", path="demo.dxf")["result"]["isError"] is False
        assert c.werkzeug("quantity_takeoff", path="scene_haus.json")["result"]["isError"] is False
        assert (
            c.werkzeug("assign_kg_proposal", path="scene_haus.json")["result"]["isError"] is False
        )
        assert c.werkzeug("list_layers", path="demo.dxf")["result"]["isError"] is False
    finally:
        rc, err = c.schliessen()
    assert rc == 0, err
    # Weder im Eingabeordner noch daneben ist etwas hinzugekommen.
    assert _baum(proj) == vorher_proj
    assert {p.name for p in raum.iterdir()} == vorher_root | {"out"}
    assert any(out.rglob("mengen.json"))


def test_lesen_ausserhalb_der_wurzeln_ist_fehlerergebnis(raum):
    proj, out = raum / "proj", raum / "out"
    c = Client(out, proj)
    try:
        c.handshake()
        for pfad in (str(raum / "geheim.json"), "../geheim.json", "C:/Windows/win.ini"):
            r = c.werkzeug("quantity_takeoff", path=pfad)["result"]
            assert r["isError"] is True, pfad
            assert r["structuredContent"]["error"]["code"] == "path_outside_root"
        # Mit --read-root ist der Ordner freigegeben.
    finally:
        c.schliessen()
    c2 = Client(out, proj, ["--read-root", str(raum)])
    try:
        c2.handshake()
        r = c2.werkzeug("quantity_takeoff", path=str(raum / "geheim.json"))["result"]
        # Jetzt wird gelesen; die Datei ist aber keine Szene.
        assert r["isError"] is True
        assert r["structuredContent"]["error"]["code"] == "scene_unreadable"
    finally:
        c2.schliessen()


def test_protokollfehler_im_subprozess(raum):
    proj, out = raum / "proj", raum / "out"
    c = Client(out, proj)
    try:
        # Vor initialize
        a = c.anfrage("tools/list")
        assert a["error"]["code"] == -32002
        c.handshake()
        # Kaputtes JSON
        c.senden_roh(b"{das ist kein json")
        a = c.antwort()
        assert a["id"] is None and a["error"]["code"] == -32700
        # Leerzeile wird uebergangen, Batch abgelehnt
        c.senden_roh(b"")
        c.senden_roh(b'[{"jsonrpc":"2.0","id":1,"method":"ping"}]')
        assert c.antwort()["error"]["code"] == -32600
        # Unbekannte Methode und unbekanntes Werkzeug
        assert c.anfrage("resources/list")["error"]["code"] == -32601
        a = c.werkzeug("gibt_es_nicht")
        assert a["error"]["code"] == -32602
        # Werkzeugfehler als isError
        r = c.werkzeug("list_layers", path="fehlt.dxf")["result"]
        assert r["isError"] is True and "input_not_found" in r["content"][0]["text"]
        # Falsche Argumente
        r = c.werkzeug("quantity_takeoff")["result"]
        assert (
            r["isError"] is True and r["structuredContent"]["error"]["code"] == "invalid_arguments"
        )
        r = c.werkzeug("quantity_takeoff", path="scene_haus.json", wall_height_m="hoch")["result"]
        assert r["isError"] is True
        # Server lebt weiter
        assert c.anfrage("ping")["result"] == {}
    finally:
        rc, _err = c.schliessen()
    assert rc == 0


def test_stdout_enthaelt_nur_protokoll(raum):
    """Logs gehen nach stderr; jede Zeile auf stdout ist eine JSON-RPC-Nachricht."""
    proj, out = raum / "proj", raum / "out"
    c = Client(out, proj)
    try:
        c.handshake()
        c.werkzeug("dwg_convert", path="demo.dxf")
    finally:
        _rc, err = c.schliessen()
    assert "planproof-cad MCP" in err
    rest = []
    while True:
        try:
            z = c._zeilen.get_nowait()
        except queue.Empty:
            break
        if z:
            rest.append(z)
    for z in rest:
        assert json.loads(z)["jsonrpc"] == "2.0"


# --- direkt am Server (schnell) -------------------------------------------------------------------


@pytest.fixture()
def server(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "scene.json").write_text(HAUS_EG.read_text(encoding="utf-8"), "utf-8")
    s = McpServer(tmp_path / "out", [tmp_path / "in"])
    s.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
    return s


def _call(s: McpServer, name: str, **args):
    return s.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }
    )


def test_tool_schemas_sind_gueltig():
    assert {t["name"] for t in TOOLS} == TOOL_NAMEN
    for t in TOOLS:
        s = t["inputSchema"]
        assert s["type"] == "object" and s["additionalProperties"] is False
        assert set(s.get("required", [])) <= set(s["properties"])
        for p in s["properties"].values():
            assert "type" in p and p.get("description")
        assert t["annotations"]["destructiveHint"] is False


def test_notification_ohne_antwort_und_ohne_id(server):
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/unbekannt"}) is None
    assert server.handle({"jsonrpc": "2.0", "id": 7, "result": {}}) is None  # Antwort des Clients


def test_ungueltige_nachrichten(server):
    assert server.handle("text")["error"]["code"] == -32600
    assert server.handle({"id": 1, "method": "ping"})["error"]["code"] == -32600
    assert server.handle({"jsonrpc": "2.0", "id": 1})["error"]["code"] == -32600
    assert (
        server.handle({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": 5})["error"]["code"]
        == -32602
    )


def test_ids_bleiben_erhalten(server):
    for id_ in (0, 5, "abc"):
        a = server.handle({"jsonrpc": "2.0", "id": id_, "method": "ping"})
        assert a["id"] == id_


def test_include_elements(server):
    knapp = _call(server, "quantity_takeoff", path="scene.json")["result"]["structuredContent"]
    voll = _call(server, "quantity_takeoff", path="scene.json", include_elements=True)["result"][
        "structuredContent"
    ]
    assert "elemente" not in knapp and "elemente" in voll


def test_assign_kg_mit_inline_mengen(server):
    m = _call(server, "quantity_takeoff", path="scene.json", geschoss="EG")["result"][
        "structuredContent"
    ]
    r = _call(server, "assign_kg_proposal", mengen=m)["result"]
    assert r["isError"] is False
    assert {p["kg"] for p in r["structuredContent"]["proposals"]} >= {"331", "344"}
    zu_viel = _call(server, "assign_kg_proposal", mengen=m, path="scene.json")["result"]
    assert zu_viel["isError"] is True
    keins = _call(server, "assign_kg_proposal")["result"]
    assert keins["isError"] is True


def test_ausgabeordnername_ist_bereinigt(server, tmp_path):
    (tmp_path / "in" / "wei rd;name&.json").write_text(HAUS_EG.read_text("utf-8"), "utf-8")
    r = _call(server, "quantity_takeoff", path="wei rd;name&.json")["result"]
    assert r["isError"] is False
    ordner = Path(r["structuredContent"]["geschrieben_nach"])
    assert ordner.parent == (tmp_path / "out").resolve()
    assert ordner.name == "wei_rd_name"


def test_interner_fehler_wird_zum_fehlerergebnis(server, monkeypatch):
    def kaputt(*_a, **_k):
        raise RuntimeError("kaputt")

    monkeypatch.setattr("planproof_cad.mcp_stdio.list_layers", kaputt)
    pfad = server.out_dir / "x.dxf"
    pfad.write_text("x", encoding="utf-8")
    r = _call(server, "list_layers", path=str(pfad))["result"]
    assert r["isError"] is True
    assert r["structuredContent"]["error"]["code"] == "internal_error"
