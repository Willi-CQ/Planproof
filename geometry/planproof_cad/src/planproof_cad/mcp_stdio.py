"""MCP-Server ueber stdio, ohne SDK.

Schlanker JSON-RPC-2.0-Server fuer das Model Context Protocol, Protokollversion 2025-06-18,
Transport stdio (eine JSON-Nachricht je Zeile, UTF-8, Log nur nach stderr). Unterstuetzt:
``initialize``, ``notifications/initialized``, ``ping``, ``tools/list``, ``tools/call``.

Werkzeuge: ``dwg_convert``, ``quantity_takeoff``, ``assign_kg_proposal``, ``list_layers``.
Alle vier rufen die Dienste in ``planproof_cad.services`` auf.

Dateizugriff:
* Geschrieben wird ausschliesslich unter dem Ausgabeverzeichnis (``--out-dir``). Der Server
  waehlt die Unterordner selbst, ein Aufrufer kann keinen Zielpfad vorgeben.
* Gelesen werden Eingaben nur unter dem Ausgabeverzeichnis, dem aktuellen Verzeichnis beim
  Start und den mit ``--read-root`` freigegebenen Verzeichnissen.

Fehler bei der Ausfuehrung eines Werkzeugs kommen als Ergebnis mit ``isError: true`` zurueck,
damit das Modell sie lesen und korrigieren kann. Unbekannte Werkzeugnamen und kaputte
Nachrichten sind JSON-RPC-Fehler.
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from planproof_cad import __version__
from planproof_cad.services import (
    ServiceError,
    assign_kg,
    geschoss_dateisuffix,
    list_layers,
    resolve_inside,
    run_dxf,
    schreibe_ausgaben,
    takeoff,
)

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "planproof-cad"

# JSON-RPC-Fehlercodes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
NOT_INITIALIZED = -32002

INSTRUCTIONS = (
    "Wandelt DXF-Zeichnungen in Mengen und Kostengruppen-Vorschläge um. Alle Kostengruppen-"
    "Einträge sind Vorschläge zur Freigabe durch einen Menschen. Wandhöhe und Öffnungshöhen "
    "sind Annahmen und nicht gemessen; die Ergebnisse nennen sie unter 'annahmen'."
)

_GESCHOSS = {
    "type": "string",
    "description": "Geschoss als freier Text, z. B. EG. Die DXF kennt keine Geschosse.",
    "maxLength": 40,
}
_HOEHE = {
    "type": "number",
    "minimum": 0.5,
    "maximum": 20,
    "description": "Wandhöhe in m. Überschreibt die Höhe im scene.json. Bleibt eine Annahme.",
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "dwg_convert",
        "title": "DXF in BIM umrechnen",
        "description": (
            "Rechnet eine DXF-Zeichnung (DWG nur mit ODA- oder LibreDWG-Brücke) in ein BIM-Modell "
            "um und schreibt plan.ifc, model.glb, scene.json, quantity_takeoff.csv, mengen.json, "
            "mengen_de.csv und kg_vorschlaege.json in einen Unterordner des Ausgabeverzeichnisses. "
            "Liefert Kennzahlen, Mengen und Kostengruppen-Vorschläge."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Pfad zur .dxf (oder .dwg)."},
                "geschoss": _GESCHOSS,
                "unit": {
                    "type": "string",
                    "enum": ["mm", "cm", "m", "in", "ft"],
                    "description": "Zeicheneinheit erzwingen (sonst $INSUNITS oder Heuristik).",
                },
                "wall_height_m": _HOEHE,
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "quantity_takeoff",
        "title": "Mengen je Bauteilart",
        "description": (
            "Mengen aus einer scene.json (oder einer DXF, die zuerst umgerechnet wird): Wände "
            "außen und innen mit Länge, Brutto- und Nettofläche, Öffnungen je Art, Räume als "
            "Bezugsfläche. Öffnungen werden von der Elternwand abgezogen. Annahmen stehen im "
            "Ergebnis. Schreibt mengen.json und mengen_de.csv."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Pfad zur scene.json oder .dxf."},
                "geschoss": _GESCHOSS,
                "wall_height_m": _HOEHE,
                "include_elements": {
                    "type": "boolean",
                    "default": False,
                    "description": "Zeilen je Wandkante, Öffnung und Raum mitliefern.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "assign_kg_proposal",
        "title": "Kostengruppen-Vorschlag",
        "description": (
            "Macht aus Mengen Kostengruppen-Vorschläge (330/331/332, 340/341/342, 334, 344) mit "
            "Begründung, Konfidenz und Annahmen. Räume bekommen keine Kostengruppe. Jeder Eintrag "
            "hat status 'vorschlag': ein Mensch gibt frei. Eingabe ist eine mengen.json, eine "
            "scene.json, eine DXF oder ein Mengen-Objekt."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Pfad zur mengen.json, scene.json oder .dxf.",
                },
                "mengen": {
                    "type": "object",
                    "description": "Alternativ zu path: ein Mengen-Objekt aus quantity_takeoff.",
                },
                "geschoss": _GESCHOSS,
                "wall_height_m": _HOEHE,
            },
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "list_layers",
        "title": "Layer einer DXF",
        "description": (
            "Listet die Layer einer DXF mit Klassifikation (Wand, Tür, Fenster ...), Zahl der "
            "Entities und dem Hinweis, wie die Datei gelesen wird. Schreibt nichts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Pfad zur .dxf (oder .dwg)."}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
]


class _ArgumentFehler(ServiceError):
    def __init__(self, message: str) -> None:
        super().__init__("invalid_arguments", message)


def _lauf_name(pfad: Path) -> str:
    stamm = pfad.parent.name if pfad.stem.lower() in ("scene", "mengen") else pfad.stem
    sauber = re.sub(r"[^A-Za-z0-9._-]+", "_", stamm).strip("._")
    return sauber or "lauf"


def _text_arg(args: dict[str, Any], name: str, *, pflicht: bool = False) -> str | None:
    wert = args.get(name)
    if wert is None:
        if pflicht:
            raise _ArgumentFehler(f"Pflichtargument fehlt: {name}")
        return None
    if not isinstance(wert, str) or not wert.strip():
        raise _ArgumentFehler(f"{name} muss ein nicht leerer Text sein.")
    return wert


def _zahl_arg(args: dict[str, Any], name: str) -> float | None:
    wert = args.get(name)
    if wert is None:
        return None
    if isinstance(wert, bool) or not isinstance(wert, (int, float)):
        raise _ArgumentFehler(f"{name} muss eine Zahl sein.")
    return float(wert)


def _ohne_elemente(mengen: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in mengen.items() if k != "elemente"}


class McpServer:
    """Zustand und Nachrichtenverarbeitung. Kein I/O, damit sich alles direkt testen laesst."""

    def __init__(self, out_dir: str | Path, read_roots: list[str | Path] | None = None) -> None:
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.read_roots = [Path.cwd().resolve(), self.out_dir] + [
            Path(r).resolve() for r in (read_roots or [])
        ]
        self.initialized = False
        self._werkzeuge: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "dwg_convert": self._dwg_convert,
            "quantity_takeoff": self._quantity_takeoff,
            "assign_kg_proposal": self._assign_kg,
            "list_layers": self._list_layers,
        }

    # -- Werkzeuge ---------------------------------------------------------

    def _lesen(self, pfad: str) -> Path:
        return resolve_inside(pfad, self.read_roots)

    def _lauf_ordner(self, quelle: Path) -> Path:
        ordner = self.out_dir / _lauf_name(quelle)
        ordner.mkdir(parents=True, exist_ok=True)
        return ordner

    def _dwg_convert(self, args: dict[str, Any]) -> dict[str, Any]:
        quelle = self._lesen(_text_arg(args, "path", pflicht=True) or "")
        geschoss = _text_arg(args, "geschoss")
        einheit = _text_arg(args, "unit")
        hoehe = _zahl_arg(args, "wall_height_m")
        if einheit is not None and einheit not in ("mm", "cm", "m", "in", "ft"):
            raise _ArgumentFehler("unit muss mm, cm, m, in oder ft sein.")
        config: dict[str, Any] = {}
        if einheit:
            config["dxf_unit"] = einheit
        if hoehe is not None:
            config["wall_height_m"] = hoehe
        ordner = self._lauf_ordner(quelle)
        res = run_dxf(quelle, ordner, config, geschoss=geschoss)
        return {
            "ok": res["ok"],
            "error": res["error"],
            "output_dir": str(ordner),
            "artifacts": [Path(a).name for a in res["artifacts"]],
            "warnings": res["warnings"],
            "template_assignment": res["template_assignment"],
            "metrics": res["metrics"],
            "mengen": _ohne_elemente(res["mengen"]) if res["mengen"] else None,
            "kg_vorschlaege": res["kg_vorschlaege"],
        }

    def _quantity_takeoff(self, args: dict[str, Any]) -> dict[str, Any]:
        quelle = self._lesen(_text_arg(args, "path", pflicht=True) or "")
        geschoss = _text_arg(args, "geschoss")
        hoehe = _zahl_arg(args, "wall_height_m")
        mit_elementen = args.get("include_elements") is True
        ordner = self._lauf_ordner(quelle)
        mengen = takeoff(quelle, geschoss=geschoss, wall_height_m=hoehe, work_dir=ordner)
        schreibe_ausgaben(mengen, None, ordner, geschoss_dateisuffix(geschoss))
        ausgabe = mengen if mit_elementen else _ohne_elemente(mengen)
        ausgabe = dict(ausgabe)
        ausgabe["geschrieben_nach"] = str(ordner)
        return ausgabe

    def _assign_kg(self, args: dict[str, Any]) -> dict[str, Any]:
        pfad = _text_arg(args, "path")
        inline = args.get("mengen")
        if (pfad is None) == (inline is None):
            raise _ArgumentFehler("Genau eines von path oder mengen angeben.")
        geschoss = _text_arg(args, "geschoss")
        hoehe = _zahl_arg(args, "wall_height_m")
        if inline is not None:
            if not isinstance(inline, dict):
                raise _ArgumentFehler("mengen muss ein Objekt sein.")
            kg = assign_kg(inline, geschoss=geschoss, wall_height_m=hoehe)
            ordner = self.out_dir / "kg"
        else:
            quelle = self._lesen(pfad or "")
            ordner = self._lauf_ordner(quelle)
            kg = assign_kg(quelle, geschoss=geschoss, wall_height_m=hoehe, work_dir=ordner)
        ordner.mkdir(parents=True, exist_ok=True)
        ziel = ordner / f"kg_vorschlaege{geschoss_dateisuffix(kg.get('geschoss'))}.json"
        ziel.write_text(json.dumps(kg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        kg = dict(kg)
        kg["geschrieben_nach"] = str(ziel)
        return kg

    def _list_layers(self, args: dict[str, Any]) -> dict[str, Any]:
        quelle = self._lesen(_text_arg(args, "path", pflicht=True) or "")
        return list_layers(quelle, work_dir=self.out_dir / "_work_layers")

    # -- Protokoll -----------------------------------------------------------

    @staticmethod
    def _fehler(id_: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}

    @staticmethod
    def _ergebnis(id_: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": id_, "result": result}

    def _tool_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str) or name not in self._werkzeuge:
            raise KeyError(f"Unbekanntes Werkzeug: {name!r}")
        args = params.get("arguments", {})
        if args is None:
            args = {}
        try:
            if not isinstance(args, dict):
                raise _ArgumentFehler("arguments muss ein Objekt sein.")
            daten = self._werkzeuge[name](args)
        except ServiceError as exc:
            return {
                "content": [{"type": "text", "text": f"{exc.code}: {exc.message}"}],
                "structuredContent": {"error": exc.as_dict()},
                "isError": True,
            }
        except Exception as exc:  # noqa: BLE001 - Werkzeugfehler gehen als Ergebnis zurueck
            traceback.print_exc(file=sys.stderr)
            meldung = f"internal_error: {type(exc).__name__}: {exc}"
            return {
                "content": [{"type": "text", "text": meldung}],
                "structuredContent": {
                    "error": {"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"}
                },
                "isError": True,
            }
        return {
            "content": [{"type": "text", "text": json.dumps(daten, ensure_ascii=False, indent=2)}],
            "structuredContent": daten,
            "isError": False,
        }

    def handle(self, nachricht: Any) -> dict[str, Any] | None:
        """Verarbeitet eine geparste Nachricht. ``None`` heisst: keine Antwort (Notification)."""
        if isinstance(nachricht, list):
            return self._fehler(
                None, INVALID_REQUEST, "Batch-Nachrichten werden nicht unterstützt."
            )
        if not isinstance(nachricht, dict) or nachricht.get("jsonrpc") != "2.0":
            return self._fehler(None, INVALID_REQUEST, "Keine JSON-RPC-2.0-Nachricht.")
        methode = nachricht.get("method")
        hat_id = "id" in nachricht
        id_ = nachricht.get("id")
        if not isinstance(methode, str):
            # Antworten des Clients auf Server-Anfragen: der Server stellt keine.
            return (
                None
                if "result" in nachricht or "error" in nachricht
                else self._fehler(id_, INVALID_REQUEST, "method fehlt.")
            )
        params = nachricht.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return (
                self._fehler(id_, INVALID_PARAMS, "params muss ein Objekt sein.")
                if hat_id
                else None
            )

        if not hat_id:  # Notification
            if methode == "notifications/initialized":
                self.initialized = True
            return None

        if methode == "ping":
            return self._ergebnis(id_, {})
        if methode == "initialize":
            self.initialized = True
            return self._ergebnis(
                id_,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "title": "Planproof Plan",
                        "version": __version__,
                    },
                    "instructions": INSTRUCTIONS,
                },
            )
        if methode in ("tools/list", "tools/call"):
            if not self.initialized:
                return self._fehler(id_, NOT_INITIALIZED, "Server nicht initialisiert.")
            if methode == "tools/list":
                return self._ergebnis(id_, {"tools": TOOLS})
            try:
                return self._ergebnis(id_, self._tool_call(params))
            except KeyError as exc:
                return self._fehler(id_, INVALID_PARAMS, str(exc.args[0]))
        return self._fehler(id_, METHOD_NOT_FOUND, f"Methode nicht gefunden: {methode}")


def serve(
    out_dir: str | Path,
    read_roots: list[str | Path] | None = None,
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | None = None,
) -> int:
    """Liest Nachrichten zeilenweise von stdin und antwortet auf stdout, bis stdin endet."""
    server = McpServer(out_dir, read_roots)
    eingabe = stdin if stdin is not None else sys.stdin.buffer
    ausgabe = stdout if stdout is not None else sys.stdout.buffer
    print(f"planproof-cad MCP (stdio), Ausgabe: {server.out_dir}", file=sys.stderr, flush=True)

    def senden(antwort: dict[str, Any]) -> None:
        zeile = json.dumps(antwort, ensure_ascii=False, separators=(",", ":"))
        ausgabe.write(zeile.encode("utf-8") + b"\n")
        ausgabe.flush()

    for roh in eingabe:
        zeile = roh.decode("utf-8", errors="replace").strip()
        if not zeile:
            continue
        try:
            nachricht = json.loads(zeile)
        except ValueError:
            senden(server._fehler(None, PARSE_ERROR, "Ungültiges JSON."))
            continue
        try:
            antwort = server.handle(nachricht)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            id_ = nachricht.get("id") if isinstance(nachricht, dict) else None
            antwort = server._fehler(id_, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        if antwort is not None:
            senden(antwort)
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(prog="python -m planproof_cad.mcp_stdio")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--read-root", action="append")
    a = p.parse_args()
    raise SystemExit(serve(a.out_dir, a.read_root or []))
