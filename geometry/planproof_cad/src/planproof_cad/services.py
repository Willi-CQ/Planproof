"""Dienste hinter CLI, REST und MCP: eine Quelle, keine Transportdetails.

Alle Funktionen nehmen Pfade oder Dicts und liefern JSON-faehige Dicts. Fehler, die der
Aufrufer beheben kann (Datei fehlt, falsches Format), kommen als ``ServiceError`` mit
stabilem ``code``; die Transporte machen daraus Exit-Code, HTTP-Status oder isError.

Das Modul importiert die Pipeline (ezdxf, ifcopenshell) erst in der jeweiligen Funktion.
``takeoff`` und ``assign_kg`` auf einer scene.json laden daher weder ezdxf noch
ifcopenshell.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from planproof_cad.kg.zuordnung import (
    MENGEN_SCHEMA,
    vorschlaege_aus_mengen,
    vorschlaege_zusammenfassen,
)
from planproof_cad.quantities.takeoff import (
    SceneLadeFehler,
    berechne_mengen,
    lade_pipeline_meta,
    lade_scene,
)
from planproof_cad.quantities.writers import schreibe_mengen_csv_de, schreibe_mengen_json

DXF_ENDUNGEN = (".dxf", ".dwg")
LoggerFn = Callable[[str], None]


class ServiceError(Exception):
    """Fehler mit stabilem Code, den ein Aufrufer beheben kann."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


# ---------------------------------------------------------------------------
# Pfade und Hilfen
# ---------------------------------------------------------------------------


def resolve_inside(path: str | Path, roots: Iterable[str | Path]) -> Path:
    """Loest ``path`` auf und verlangt, dass das Ergebnis in einem der ``roots`` liegt.

    Relative Pfade werden der Reihe nach unter jedem Root gesucht: der erste Treffer, der
    existiert, gewinnt; existiert keiner, gilt der erste Kandidat (der Aufrufer meldet dann
    "nicht gefunden"). ``..``-Abschnitte und Symlinks werden vor dem Vergleich aufgeloest; liegt
    ein Kandidat danach nicht in seinem Root, wird er verworfen. Bleibt keiner uebrig, gibt es
    ``path_outside_root``.
    """
    wurzeln = [Path(r).resolve() for r in roots]
    if not wurzeln:
        raise ServiceError("path_outside_root", "Kein erlaubtes Verzeichnis konfiguriert.")
    roh = str(path)
    if not roh.strip() or "\x00" in roh:
        raise ServiceError("path_invalid", "Ungültiger Pfad.")
    p = Path(roh)

    def _innerhalb(ziel: Path, ws: list[Path]) -> bool:
        return any(ziel == w or w in ziel.parents for w in ws)

    if p.is_absolute():
        ziel = p.resolve()
        if _innerhalb(ziel, wurzeln):
            return ziel
    else:
        gueltig = []
        for w in wurzeln:
            kandidat = (w / p).resolve()
            if _innerhalb(kandidat, [w]):
                gueltig.append(kandidat)
        for kandidat in gueltig:
            if kandidat.exists():
                return kandidat
        if gueltig:
            return gueltig[0]
    raise ServiceError(
        "path_outside_root",
        "Der Pfad liegt außerhalb der erlaubten Verzeichnisse.",
    )


def geschoss_dateisuffix(geschoss: str | None) -> str:
    """Dateinamen-Zusatz fuer ein Geschoss: ``EG`` -> ``_EG``, leer -> leer."""
    if not geschoss:
        return ""
    sauber = re.sub(r"[^A-Za-z0-9._-]+", "_", geschoss).strip("_")
    return f"_{sauber}" if sauber else ""


def _json_sicher(wert: Any) -> Any:
    return json.loads(json.dumps(wert, default=str))


def _pruefe_hoehe(wall_height_m: float | None) -> None:
    if wall_height_m is not None and not (0.5 <= float(wall_height_m) <= 20.0):
        raise ServiceError(
            "invalid_wall_height", "wall_height_m muss zwischen 0,5 und 20 m liegen."
        )


def _dxf_pfad(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_file():
        raise ServiceError("input_not_found", f"Eingabe nicht gefunden: {p.name}")
    if p.suffix.lower() not in DXF_ENDUNGEN:
        raise ServiceError("unsupported_input", f"Erwartet .dxf oder .dwg, erhalten: {p.suffix}")
    return p


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


def run_dxf(
    path: str | Path,
    out_dir: str | Path,
    config: dict[str, Any] | None = None,
    *,
    geschoss: str | None = None,
    mengen: bool = True,
    logger_fn: LoggerFn | None = None,
) -> dict[str, Any]:
    """DXF (oder DWG ueber eine ODA-/LibreDWG-Bruecke) in ein BIM-Modell umrechnen.

    Schreibt in ``out_dir``: plan.ifc, model.glb, scene.json, quantity_takeoff.csv, meta.json,
    result.json und, wenn ``mengen`` gesetzt ist und ein scene-Modell entstand, ``mengen.json``,
    ``mengen_de.csv`` und ``kg_vorschlaege.json``.
    """
    from planproof_cad.cad import run_dwg_to_bim_pipeline

    eingabe = _dxf_pfad(path)
    cfg = dict(config or {})
    if "wall_height_m" in cfg:
        _pruefe_hoehe(cfg["wall_height_m"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    res = run_dwg_to_bim_pipeline(
        str(eingabe), str(out), str(out / "_work"), cfg, logger_fn=logger_fn
    )
    antwort: dict[str, Any] = {
        "ok": bool(res.ok),
        "error": res.error,
        "input": eingabe.name,
        "out_dir": str(out),
        "geschoss": geschoss,
        "metrics": _json_sicher(res.metrics),
        "warnings": list(res.warnings),
        "template_assignment": _json_sicher(res.template_assignment),
        "artifacts": [str(a) for a in res.artifacts],
        "mengen": None,
        "kg_vorschlaege": None,
    }
    if mengen and res.scene is not None:
        try:
            m = berechne_mengen(
                res.scene.to_dict(),
                geschoss=geschoss,
                pipeline_meta={"metrics": _json_sicher(res.metrics)},
            )
            antwort["mengen"] = m
            antwort["kg_vorschlaege"] = vorschlaege_aus_mengen(m)
            antwort["artifacts"] += [
                str(p) for p in schreibe_ausgaben(m, antwort["kg_vorschlaege"], out)
            ]
        except Exception as exc:  # noqa: BLE001 - die Umrechnung selbst war ok
            antwort["warnings"].append(f"Mengen und Kostengruppen-Vorschlag fehlgeschlagen: {exc}")
    (out / "result.json").write_text(
        json.dumps(
            {
                "ok": antwort["ok"],
                "metrics": antwort["metrics"],
                "warnings": antwort["warnings"],
                "error": antwort["error"],
                "template_assignment": antwort["template_assignment"],
                "artifacts": [Path(a).name for a in antwort["artifacts"]],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return antwort


def schreibe_ausgaben(
    mengen: dict[str, Any],
    kg: dict[str, Any] | None,
    out_dir: str | Path,
    suffix: str = "",
) -> list[Path]:
    """Schreibt mengen.json, mengen_de.csv und (falls vorhanden) kg_vorschlaege.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pfade = [
        schreibe_mengen_json(mengen, out / f"mengen{suffix}.json"),
        schreibe_mengen_csv_de(mengen, out / f"mengen_de{suffix}.csv"),
    ]
    if kg is not None:
        ziel = out / f"kg_vorschlaege{suffix}.json"
        ziel.write_text(json.dumps(kg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        pfade.append(ziel)
    return pfade


# ---------------------------------------------------------------------------
# takeoff
# ---------------------------------------------------------------------------


def _scene_aus_quelle(
    quelle: Any,
    *,
    work_dir: Path | None,
    config: dict[str, Any] | None,
    logger_fn: LoggerFn | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Gibt (scene-Dict, Pipeline-Meta) zurueck. Quelle: Dict, Modell, scene.json oder DXF."""
    if isinstance(quelle, dict):
        if "components" not in quelle:
            raise ServiceError("not_a_scene", "Das Dict ist kein scene-Dict (components fehlt).")
        return quelle, None
    if hasattr(quelle, "to_dict"):
        return quelle.to_dict(), None
    if isinstance(quelle, (str, Path)):
        p = Path(quelle)
        endung = p.suffix.lower()
        if endung == ".json":
            try:
                return lade_scene(p), lade_pipeline_meta(p)
            except SceneLadeFehler as exc:
                raise ServiceError("scene_unreadable", str(exc)) from exc
        if endung in DXF_ENDUNGEN:
            if work_dir is None:
                raise ServiceError(
                    "out_dir_required",
                    "Für eine DXF/DWG als Quelle wird ein Ausgabeverzeichnis gebraucht.",
                )
            res = run_dxf(p, work_dir, config, mengen=False, logger_fn=logger_fn)
            if not res["ok"]:
                raise ServiceError("convert_failed", res["error"] or "Umrechnung fehlgeschlagen")
            return lade_scene(Path(work_dir) / "scene.json"), {"metrics": res["metrics"]}
        raise ServiceError("unsupported_input", f"Erwartet .json, .dxf oder .dwg: {p.name}")
    raise ServiceError("unsupported_input", "Quelle muss ein scene-Dict oder ein Pfad sein.")


def takeoff(
    quelle: Any,
    geschoss: str | None = None,
    wall_height_m: float | None = None,
    *,
    out_dir: str | Path | None = None,
    work_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
    logger_fn: LoggerFn | None = None,
) -> dict[str, Any]:
    """Mengen je Bauteilart (``mengen.json``-Inhalt).

    ``quelle``: scene-Dict, ``SceneModel``, Pfad zu einer scene.json oder Pfad zu einer DXF/DWG
    (dann wird zuerst umgerechnet; das Ergebnis der Umrechnung landet in ``work_dir``, ersatzweise
    in ``out_dir``, eines von beiden ist dann Pflicht). Mit ``out_dir`` werden ``mengen.json`` und
    ``mengen_de.csv`` geschrieben. ``wall_height_m`` ueberschreibt die Wandhoehe; sie bleibt
    eine Annahme.
    """
    _pruefe_hoehe(wall_height_m)
    arbeit = work_dir if work_dir is not None else out_dir
    scene, meta = _scene_aus_quelle(
        quelle, work_dir=Path(arbeit) if arbeit else None, config=config, logger_fn=logger_fn
    )
    try:
        mengen = berechne_mengen(
            scene, geschoss=geschoss, wall_height_m=wall_height_m, pipeline_meta=meta
        )
    except ValueError as exc:
        raise ServiceError("not_a_scene", str(exc)) from exc
    if out_dir is not None:
        schreibe_ausgaben(mengen, None, out_dir)
    return mengen


# ---------------------------------------------------------------------------
# kg
# ---------------------------------------------------------------------------


def _ist_mengen(wert: Any) -> bool:
    return isinstance(wert, dict) and wert.get("schema") == MENGEN_SCHEMA


def assign_kg(
    mengen: Any,
    *,
    geschoss: str | None = None,
    wall_height_m: float | None = None,
    out_dir: str | Path | None = None,
    work_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
    logger_fn: LoggerFn | None = None,
) -> dict[str, Any]:
    """Kostengruppen-Vorschlaege (``kg_vorschlaege.json``-Inhalt).

    ``mengen``: ein Mengen-Dict, eine Liste davon (mehrere Geschosse, dann mit Summen), ein Pfad
    zu einer ``mengen.json`` oder, bequem, alles, was ``takeoff`` annimmt (scene-Dict, scene.json,
    DXF). Jeder Eintrag hat ``status = "vorschlag"``: ein Mensch gibt frei.
    """
    if isinstance(mengen, (list, tuple)):
        if not mengen:
            raise ServiceError("empty_input", "Keine Mengen übergeben.")
        dokumente = [
            assign_kg(
                m,
                geschoss=geschoss,
                wall_height_m=wall_height_m,
                work_dir=work_dir,
                config=config,
                logger_fn=logger_fn,
            )
            for m in mengen
        ]
        kombi = dokumente[0] if len(dokumente) == 1 else vorschlaege_zusammenfassen(dokumente)
        if out_dir is not None:
            _schreibe_kg(kombi, out_dir)
        return kombi

    m = mengen
    if isinstance(m, (str, Path)) and Path(m).suffix.lower() == ".json":
        p = Path(m)
        try:
            roh = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ServiceError(
                "input_unreadable", f"Datei nicht lesbar: {p.name} ({type(exc).__name__})"
            ) from exc
        m = roh if _ist_mengen(roh) else p
    if not _ist_mengen(m):
        m = takeoff(
            m,
            geschoss=geschoss,
            wall_height_m=wall_height_m,
            work_dir=work_dir if work_dir is not None else out_dir,
            config=config,
            logger_fn=logger_fn,
        )
    try:
        kg = vorschlaege_aus_mengen(m)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ServiceError(
            "not_mengen", f"Mengen-Dokument unvollständig oder fehlerhaft: {exc}"
        ) from exc
    if out_dir is not None:
        _schreibe_kg(kg, out_dir)
    return kg


def _schreibe_kg(kg: dict[str, Any], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = geschoss_dateisuffix(kg.get("geschoss"))
    ziel = out / f"kg_vorschlaege{suffix}.json"
    ziel.write_text(json.dumps(kg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ziel


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------

_LINIEN_TYPEN = ("LINE", "LWPOLYLINE", "POLYLINE")
_TEXT_TYPEN = ("TEXT", "MTEXT")


def list_layers(path: str | Path, work_dir: str | Path | None = None) -> dict[str, Any]:
    """Layer einer DXF mit Klassifikation und Entity-Zahlen.

    Zeigt, wie die Datei gelesen wird: welche Layer als Wandgeometrie gelten, welche Bloecke als
    Tuer oder Fenster, und ob mangels Wand-Layer alle Linien als Wandkandidaten zaehlen.
    DWG-Dateien werden nur gelesen, wenn eine ODA-/LibreDWG-Bruecke vorhanden ist.
    """
    import ezdxf

    from planproof_cad.cad.layer_classifier import (
        DOOR,
        WALL_CLASSES,
        WALL_EXTERIOR,
        WINDOW,
        classify_block,
        classify_layer,
    )

    eingabe = _dxf_pfad(path)
    dxf_pfad = eingabe
    if eingabe.suffix.lower() == ".dwg":
        import tempfile

        from planproof_cad.cad.dxf_loader import DwgConversionError, _convert_dwg_to_dxf

        try:
            wd = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="planproof-cad_layers_"))
            dxf_pfad = _convert_dwg_to_dxf(eingabe, wd)
        except DwgConversionError as exc:
            raise ServiceError("dwg_bridge_missing", str(exc)) from exc
    try:
        doc = ezdxf.readfile(str(dxf_pfad))
    except Exception:  # noqa: BLE001
        try:
            from ezdxf import recover

            doc, _ = recover.readfile(str(dxf_pfad))
        except Exception as exc:  # noqa: BLE001
            raise ServiceError("dxf_unreadable", f"DXF nicht lesbar: {exc}") from exc

    msp = doc.modelspace()
    zaehler: dict[str, dict[str, int]] = {}
    bloecke: dict[tuple[str, str], int] = {}
    for e in msp:
        typ = e.dxftype()
        layer = str(getattr(e.dxf, "layer", "0"))
        z = zaehler.setdefault(layer, {})
        z[typ] = z.get(typ, 0) + 1
        if typ == "INSERT":
            name = str(getattr(e.dxf, "name", ""))
            bloecke[(name, layer)] = bloecke.get((name, layer), 0) + 1

    namen = {str(la.dxf.name) for la in doc.layers} | set(zaehler)
    wand_linien_gesamt = 0
    layer_liste: list[dict[str, Any]] = []
    for name in sorted(namen, key=str.casefold):
        klasse = classify_layer(name)
        z = zaehler.get(name, {})
        linien = sum(z.get(t, 0) for t in _LINIEN_TYPEN)
        inserts = z.get("INSERT", 0)
        texte = sum(z.get(t, 0) for t in _TEXT_TYPEN)
        if klasse.cls in WALL_CLASSES:
            wand_linien_gesamt += linien
            gelesen = "wand"
            beschr = (
                "Linien gelten als Außenwand"
                if klasse.cls == WALL_EXTERIOR
                else "Linien gelten als Wandgeometrie"
            )
        elif klasse.cls in (DOOR, WINDOW):
            gelesen = "oeffnung_block"
            beschr = "Blöcke (INSERT) auf diesem Layer gelten als Tür oder Fenster"
        else:
            gelesen = "nicht_gelesen"
            beschr = (
                "Linien zählen nur als Wand, wenn die Datei gar keinen Wand-Layer hat"
                if linien
                else "keine Wandlinien"
            )
        if texte:
            beschr += "; Texte dienen als Raumnamen"
        layer_liste.append(
            {
                "name": name,
                "class": klasse.cls,
                "confidence": klasse.confidence,
                "signal": klasse.signal if klasse.cls != "unknown" else None,
                # Tragverhalten aus dem Namen, sobald der Klassifikator es liefert.
                "bearing": getattr(klasse, "bearing", None),
                "read_as": gelesen,
                "description": beschr,
                "entities": {
                    "total": sum(z.values()),
                    "lines": linien,
                    "inserts": inserts,
                    "texts": texte,
                    "by_type": dict(sorted(z.items())),
                },
            }
        )

    block_liste: list[dict[str, Any]] = []
    for (name, layer), anzahl in sorted(
        bloecke.items(), key=lambda kv: (kv[0][0].casefold(), kv[0][1])
    ):
        k = classify_block(name, layer)
        block_liste.append(
            {
                "name": name,
                "layer": layer,
                "count": anzahl,
                "class": k.cls,
                "confidence": k.confidence,
                "signal": k.signal if k.cls != "unknown" else None,
                "read_as": "oeffnung_block" if k.cls in (DOOR, WINDOW) else "nicht_gelesen",
            }
        )

    linien_gesamt = sum(la["entities"]["lines"] for la in layer_liste)
    rueckfall = wand_linien_gesamt == 0 and linien_gesamt > 0
    hinweise: list[str] = []
    if rueckfall:
        hinweise.append(
            "Kein Layer mit Wandlinien erkannt: die Pipeline nimmt dann alle Linien als "
            "Wandkandidaten. Das liefert meist zu viele und falsche Wände. Layer nach dem "
            "Schema A_Wand_a, A_Wand_i oder A-WALL benennen."
        )
    if not any(b["read_as"] == "oeffnung_block" for b in block_liste):
        hinweise.append(
            "Kein Block als Tür oder Fenster erkannt. Öffnungen entstehen nur aus Blöcken (INSERT) "
            "mit Tür-/Fenster-Namen oder auf Tür-/Fenster-Layern."
        )
    insunits = doc.header.get("$INSUNITS", 0)
    return {
        "input": eingabe.name,
        "insunits": insunits,
        "layers": layer_liste,
        "blocks": block_liste,
        "summary": {
            "layer_count": len(layer_liste),
            "wall_layers": [la["name"] for la in layer_liste if la["read_as"] == "wand"],
            "wall_line_entities": wand_linien_gesamt,
            "all_line_entities": linien_gesamt,
            "fallback_all_lines_as_walls": rueckfall,
            "opening_blocks": sum(
                b["count"] for b in block_liste if b["read_as"] == "oeffnung_block"
            ),
        },
        "hinweise": hinweise,
    }


__all__ = [
    "DXF_ENDUNGEN",
    "ServiceError",
    "assign_kg",
    "geschoss_dateisuffix",
    "list_layers",
    "resolve_inside",
    "run_dxf",
    "schreibe_ausgaben",
    "takeoff",
]
