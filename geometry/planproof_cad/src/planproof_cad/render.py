"""Lesbare Textausgabe fuer Mengen, Vorschlaege und Layer (CLI).

Reine Darstellung, keine Rechenlogik. Alle Texte kommen aus den Ergebnis-Dicts.
"""

from __future__ import annotations

from typing import Any

_LAGE = {"aussen": "Außen", "innen": "Innen", "unbekannt": "Lage unbekannt"}


def zahl(wert: float, stellen: int = 2) -> str:
    """Deutsches Dezimalkomma, feste Nachkommastellen."""
    return f"{wert:.{stellen}f}".replace(".", ",")


def _wand_name(g: dict[str, Any]) -> str:
    name = {"aussen": "Außenwand", "innen": "Innenwand", "unbekannt": "Wand (Lage unbekannt)"}[
        g["lage"]
    ]
    if g["tragend"] is True:
        name += " tragend"
    elif g["tragend"] is False:
        name += " nichttragend"
    return name


def format_mengen(m: dict[str, Any]) -> str:
    z: list[str] = []
    kopf = "Mengen" + (f", Geschoss {m['geschoss']}" if m.get("geschoss") else "")
    z.append(kopf)
    w = m["waende"]
    z.append("Wände (Kanten, keine physischen Wände)")
    if not w["gruppen"]:
        z.append("  keine")
    for g in w["gruppen"]:
        z.append(
            f"  {_wand_name(g):<26} {g['anzahl_kanten']:>3} Kanten  "
            f"{zahl(g['laenge_m']):>8} m  brutto {zahl(g['flaeche_brutto_m2']):>8} m2  "
            f"Abzug {zahl(g['oeffnungsabzug_m2']):>7} m2  "
            f"netto {zahl(g['flaeche_netto_m2']):>8} m2"
        )
    z.append("Öffnungen")
    if not m["oeffnungen"]["gruppen"]:
        z.append("  keine")
    for g in m["oeffnungen"]["gruppen"]:
        art = "Tür" if g["art"] == "tuer" else "Fenster"
        z.append(
            f"  {art + ' ' + _LAGE[g['lage']].lower():<26} {g['anzahl']:>3} Stk    "
            f"Breite gesamt {zahl(g['breite_summe_m']):>7} m  Fläche {zahl(g['flaeche_m2']):>7} m2"
        )
    r = m["raeume"]
    z.append("Räume (Bezugsfläche, keine Kostengruppe)")
    if r["anzahl"]:
        z.append(f"  {r['anzahl']} Räume, {zahl(r['flaeche_m2'])} m2 ({r['flaeche_quelle']})")
    else:
        z.append("  keine")
    if m["hinweise"]:
        z.append("Hinweise")
        z += [f"  - {h['text']}" for h in m["hinweise"]]
    z.append("Annahmen")
    z += [f"  - {a['text']}" for a in m["annahmen"]]
    z.append("Nicht abgedeckt: " + ", ".join(e["label"] for e in m["nicht_abgedeckt"]))
    return "\n".join(z)


def format_kg(d: dict[str, Any]) -> str:
    z: list[str] = []
    z.append(d["status_hinweis"])
    for p in d["proposals"]:
        z.append(f"  {p['summary']}  [Konfidenz {p['confidence']}]")
    if not d["proposals"]:
        z.append("  (keine Vorschläge)")
    if d.get("summen") and len(d.get("geschosse", [])) > 1:
        z.append("Summen über alle Geschosse")
        for s in d["summen"]:
            menge = f"{s['quantity']} Stk" if s["unit"] == "Stk" else f"{zahl(s['quantity'])} m2"
            z.append(f"  KG {s['kg']}  {s['label']}  {menge}  [Konfidenz {s['confidence']}]")
    for ref in d["reference_quantities"]:
        z.append(f"  Bezugsfläche {ref['label']}: {zahl(ref['quantity'])} m2, keine Kostengruppe")
    if d["unassigned"]:
        z.append("Ohne Vorschlag")
        for u in d["unassigned"]:
            z.append(f"  - {u['component']}: {u['quantity']} {u['unit']}. {u['reason']}")
    z.append("Nicht abgedeckt: " + ", ".join(e["label"] for e in d["not_covered"]))
    return "\n".join(z)


def format_layers(d: dict[str, Any]) -> str:
    z = [f"Layer in {d['input']}"]
    z.append(
        f"  {'Layer':<28} {'Klasse':<14} {'Linien':>7} {'Blöcke':>7} {'Texte':>6}  Gelesen als"
    )
    for la in d["layers"]:
        e = la["entities"]
        z.append(
            f"  {la['name']:<28} {la['class']:<14} {e['lines']:>7} {e['inserts']:>7} "
            f"{e['texts']:>6}  {la['description']}"
        )
    if d["blocks"]:
        z.append("Blöcke")
        for b in d["blocks"]:
            z.append(f"  {b['name']:<28} Layer {b['layer']:<20} {b['count']:>3}x  {b['class']}")
    s = d["summary"]
    z.append(
        f"Wand-Layer: {', '.join(s['wall_layers']) or 'keine'}; Wandlinien: "
        f"{s['wall_line_entities']} von {s['all_line_entities']}; Tür-/Fensterblöcke: "
        f"{s['opening_blocks']}"
    )
    z += [f"Hinweis: {h}" for h in d["hinweise"]]
    return "\n".join(z)


__all__ = ["format_kg", "format_layers", "format_mengen", "zahl"]
