"""Ausgabedateien der Mengen: ``mengen.json`` und ``mengen_de.csv``.

``mengen_de.csv`` hat den Aufbau, den die Desktop-Anwendung liest (``quantityCsv.ts``):
UTF-8 mit BOM, Semikolon, Punkt als Dezimalzeichen, die Kopfzeile unten, Datenzeilen bis zur
ersten Leerzeile. Danach folgen Zusammenfassung und Annahmen; der Parser ignoriert sie, ein
Mensch in Excel sieht sie.

Die Kategorien bleiben wie in der Desktop-Pipeline ASCII (``Wand``, ``Tuer``, ``Fenster``,
``Raum``): sie sind Schluesselwerte, keine Anzeigetexte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CSV_HEADER = "Kategorie;ID;Typ;Laenge_m;Hoehe_m;Dicke_m;Flaeche_m2;Volumen_m3;Material;Eigenschaft"


# Typ-Spalte wie in der Desktop-Pipeline: "exterior" und "interior", unbekannt bleibt leer.
_WAND_TYP = {"aussen": "exterior", "innen": "interior"}


def _zahl(wert: Any) -> str:
    if wert is None:
        return ""
    return f"{float(wert):.3f}"


def _text(wert: Any) -> str:
    """Freitext fuer eine Zelle: kein Semikolon, kein Zeilenumbruch."""
    return " ".join(str(wert).replace(";", ",").split())


def _eigenschaft(teile: list[str]) -> str:
    return _text(", ".join(t for t in teile if t))


def csv_zeilen(mengen: dict[str, Any]) -> list[list[str]]:
    """Datenzeilen der CSV, eine je Wandkante, Tuer, Fenster und Raum."""
    geschoss = mengen.get("geschoss")
    kopf = [f"Geschoss={geschoss}"] if geschoss else []
    elemente = mengen.get("elemente", {})
    zeilen: list[list[str]] = []

    for w in elemente.get("waende", []):
        tragend = w.get("tragend")
        trag = "unbekannt" if tragend is None else ("true" if tragend else "false")
        volumen = w["flaeche_netto_m2"] * w["dicke_m"] if w.get("dicke_m") is not None else None
        zeilen.append(
            [
                "Wand",
                _text(w["id"]),
                _WAND_TYP.get(w["lage"], ""),
                _zahl(w["laenge_m"]),
                _zahl(w["hoehe_m"]),
                _zahl(w.get("dicke_m")),
                _zahl(w["flaeche_netto_m2"]),
                _zahl(volumen),
                "",
                _eigenschaft(
                    kopf
                    + [
                        f"tragend={trag}",
                        f"netto = brutto {w['flaeche_brutto_m2']:.3f} m2 - Abzug "
                        f"{w['oeffnungsabzug_m2']:.3f} m2",
                        "Höhe angenommen",
                        "Wandkante",
                    ]
                ),
            ]
        )

    for o in elemente.get("oeffnungen", []):
        kat = "Tuer" if o["art"] == "tuer" else "Fenster"
        unterart = (o.get("unterart") or "").removeprefix("door_")
        teile = kopf + [f"Lage={o['lage']}"]
        if o.get("eltern_id"):
            teile.append(f"Wand={o['eltern_id']}")
        if o["art"] == "fenster":
            teile.append(f"Brüstung {o['bruestung_m']:.3f} m")
        if o["abgezogen"]:
            teile.append(f"Abzug {o['abzug_m2']:.3f} m2")
        else:
            teile.append("ohne Elternwand, nicht abgezogen")
        teile.append("Höhe angenommen")
        zeilen.append(
            [
                kat,
                _text(o["id"]),
                _text(unterart),
                _zahl(o["breite_m"]),
                _zahl(o["hoehe_m"]),
                "",
                _zahl(o["flaeche_m2"]),
                "",
                "",
                _eigenschaft(teile),
            ]
        )

    for r in elemente.get("raeume", []):
        teile = kopf + ["Bezugsfläche ohne Kostengruppe"]
        if r["flaeche_art"] != "netto":
            teile.append("area_m2, keine Nettofläche")
        if r.get("bezeichnung"):
            teile.append(f"Bezeichnung={r['bezeichnung']}")
        zeilen.append(
            [
                "Raum",
                _text(r["id"]),
                _text(r.get("raumart") or ""),
                "",
                "",
                "",
                _zahl(r["flaeche_m2"]),
                "",
                "",
                _eigenschaft(teile),
            ]
        )
    return zeilen


def csv_text(mengen: dict[str, Any]) -> str:
    zeilen = csv_zeilen(mengen)
    out = [CSV_HEADER]
    out += [";".join(z) for z in zeilen]
    out.append("")
    out.append(f"ZUSAMMENFASSUNG;{len(zeilen)} Positionen")
    out.append(
        "ANNAHME;"
        + _text(
            "Flaeche_m2 der Wandzeilen ist die Nettofläche nach Öffnungsabzug, Volumen_m3 ist "
            "Nettofläche mal Dicke. Bei Türen und Fenstern ist Flaeche_m2 die Öffnungsfläche, "
            "bei Räumen die Bezugsfläche."
        )
    )
    for a in mengen.get("annahmen", []):
        out.append("ANNAHME;" + _text(a["text"]))
    for h in mengen.get("hinweise", []):
        out.append("HINWEIS;" + _text(h["text"]))
    return "\n".join(out) + "\n"


def schreibe_mengen_json(mengen: dict[str, Any], pfad: str | Path) -> Path:
    p = Path(pfad)
    p.write_text(json.dumps(mengen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def schreibe_mengen_csv_de(mengen: dict[str, Any], pfad: str | Path) -> Path:
    p = Path(pfad)
    p.write_bytes(("﻿" + csv_text(mengen)).encode("utf-8"))
    return p


__all__ = [
    "CSV_HEADER",
    "csv_text",
    "csv_zeilen",
    "schreibe_mengen_csv_de",
    "schreibe_mengen_json",
]
