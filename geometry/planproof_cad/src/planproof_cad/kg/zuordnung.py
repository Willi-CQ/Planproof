"""Mengenkern und Kostengruppen-Vorschlag auf dem scene-Dict.

Diese Datei ist in sich geschlossen: reine Standardbibliothek (nur ``typing``), keine
Importe aus ``planproof_cad``. Sie kann deshalb byte-identisch in ein anderes Projekt
kopiert werden, das ein scene-Dict (``scene_v1``) hat, aber dieses Paket nicht.

Zwei Teile:

A. Mengenkern (``mengen_aus_scene``): aus dem scene-Dict die Mengen je Bauteilart.
   Wandkanten, Oeffnungen (Tueren, Fenster) und Raeume. Jede Annahme wird ausgewiesen.

B. Vorschlag (``vorschlaege_aus_mengen``, ``vorschlaege_aus_scene``): Mengen werden mit der
   Regeltabelle ``KG_RULES`` zu Kostengruppen-Vorschlaegen. Die Tabelle enthaelt nur
   Nummern und eigene, neutrale Bauteilbezeichnungen, keine Texte einer Norm. Jeder
   Eintrag hat ``status = "vorschlag"``: ein Mensch gibt frei, das Modul legt nie etwas an.

Raeume bekommen keine Kostengruppe. Raumflaechen sind Bezugsgroessen und werden getrennt
ausgewiesen (``reference_quantities``).

Was die Zeichnung nicht liefert, ist als Annahme markiert und nie als Messung ausgegeben:
Wandhoehe, Oeffnungshoehen, Brutto statt Netto an Anschluessen.

``KG_RULES`` muss inhaltsgleich zu ``kg_rules.json`` im Paket bleiben (ein Test prueft das).
"""

from __future__ import annotations

import json
from typing import Any

MENGEN_SCHEMA = "planproof_cad.mengen"
MENGEN_SCHEMA_VERSION = "mengen_v1"
VORSCHLAG_SCHEMA = "planproof_cad.kg_vorschlaege"
VORSCHLAG_SCHEMA_VERSION = "kg_vorschlaege_v1"

# Standard-Wandhoehe, wenn weder ein Parameter noch das scene-Dict eine Hoehe nennt.
# Das ist eine Konfigurationsannahme, keine Messung.
STANDARD_WANDHOEHE_M = 2.80

# ---------------------------------------------------------------------------
# Regeltabelle (eingebettet, damit die Datei allein kopierbar ist)
# ---------------------------------------------------------------------------

KG_RULES: dict[str, Any] = {
    "version": 1,
    "hinweis": (
        "Vorschlagsregeln mit Nummern und eigenen Bauteilbezeichnungen. Enthaelt keine "
        "Normtexte, keine Abgrenzungen und keine Tabellen einer Norm."
    ),
    "rules": [
        {
            "id": "wand_aussen",
            "component": "wand",
            "location": "aussen",
            "bearing": None,
            "kg": "330",
            "element_class": "Aussenwand",
            "label": "Außenwand",
            "label_plural": "Außenwände",
            "basis": "Nettoflaeche",
            "unit": "m2",
            "rationale": (
                "Außenwand, Tragverhalten nicht bekannt: die allgemeine Nummer ohne "
                "Unterscheidung tragend oder nichttragend."
            ),
        },
        {
            "id": "wand_aussen_tragend",
            "component": "wand",
            "location": "aussen",
            "bearing": True,
            "kg": "331",
            "element_class": "Aussenwand tragend",
            "label": "tragende Außenwand",
            "label_plural": "tragende Außenwände",
            "basis": "Nettoflaeche",
            "unit": "m2",
            "rationale": "Außenwand, als tragend gekennzeichnet.",
        },
        {
            "id": "wand_aussen_nichttragend",
            "component": "wand",
            "location": "aussen",
            "bearing": False,
            "kg": "332",
            "element_class": "Aussenwand nichttragend",
            "label": "nichttragende Außenwand",
            "label_plural": "nichttragende Außenwände",
            "basis": "Nettoflaeche",
            "unit": "m2",
            "rationale": "Außenwand, als nichttragend gekennzeichnet.",
        },
        {
            "id": "wand_innen",
            "component": "wand",
            "location": "innen",
            "bearing": None,
            "kg": "340",
            "element_class": "Innenwand",
            "label": "Innenwand",
            "label_plural": "Innenwände",
            "basis": "Nettoflaeche",
            "unit": "m2",
            "rationale": (
                "Innenwand, Tragverhalten nicht bekannt: die allgemeine Nummer ohne "
                "Unterscheidung tragend oder nichttragend."
            ),
        },
        {
            "id": "wand_innen_tragend",
            "component": "wand",
            "location": "innen",
            "bearing": True,
            "kg": "341",
            "element_class": "Innenwand tragend",
            "label": "tragende Innenwand",
            "label_plural": "tragende Innenwände",
            "basis": "Nettoflaeche",
            "unit": "m2",
            "rationale": "Innenwand, als tragend gekennzeichnet.",
        },
        {
            "id": "wand_innen_nichttragend",
            "component": "wand",
            "location": "innen",
            "bearing": False,
            "kg": "342",
            "element_class": "Innenwand nichttragend",
            "label": "nichttragende Innenwand",
            "label_plural": "nichttragende Innenwände",
            "basis": "Nettoflaeche",
            "unit": "m2",
            "rationale": "Innenwand, als nichttragend gekennzeichnet.",
        },
        {
            "id": "tuer_aussen",
            "component": "tuer",
            "location": "aussen",
            "bearing": None,
            "kg": "334",
            "element_class": "Aussentuer",
            "label": "Außentür",
            "label_plural": "Außentüren",
            "basis": "Stueck",
            "unit": "Stk",
            "rationale": "Tür in einer Außenwand.",
        },
        {
            "id": "fenster_aussen",
            "component": "fenster",
            "location": "aussen",
            "bearing": None,
            "kg": "334",
            "element_class": "Aussenfenster",
            "label": "Außenfenster",
            "label_plural": "Außenfenster",
            "basis": "Stueck",
            "unit": "Stk",
            "rationale": "Fenster in einer Außenwand.",
        },
        {
            "id": "tuer_innen",
            "component": "tuer",
            "location": "innen",
            "bearing": None,
            "kg": "344",
            "element_class": "Innentuer",
            "label": "Innentür",
            "label_plural": "Innentüren",
            "basis": "Stueck",
            "unit": "Stk",
            "rationale": "Tür in einer Innenwand.",
        },
        {
            "id": "fenster_innen",
            "component": "fenster",
            "location": "innen",
            "bearing": None,
            "kg": "344",
            "element_class": "Innenfenster",
            "label": "Innenfenster",
            "label_plural": "Innenfenster",
            "basis": "Stueck",
            "unit": "Stk",
            "rationale": "Fenster in einer Innenwand.",
        },
    ],
    "not_covered": [
        {
            "element_class": "Stuetze",
            "label": "Stützen",
            "reason": "Die Pipeline verarbeitet keine Stützen. Keine Mengen, kein Vorschlag.",
        },
        {
            "element_class": "Treppe",
            "label": "Treppen",
            "reason": "Die Pipeline verarbeitet keine Treppen. Keine Mengen, kein Vorschlag.",
        },
        {
            "element_class": "Decke",
            "label": "Decken",
            "reason": "Die Pipeline verarbeitet keine Decken. Keine Mengen, kein Vorschlag.",
        },
        {
            "element_class": "Dach",
            "label": "Dächer",
            "reason": "Die Pipeline verarbeitet keine Dächer. Keine Mengen, kein Vorschlag.",
        },
    ],
    "no_cost_group": [
        {
            "element_class": "Raum",
            "label": "Räume",
            "reason": (
                "Raumflächen sind Bezugsgrößen und bekommen keine Kostengruppe. "
                "Sie gehören fachlich zu DIN 277."
            ),
        }
    ],
}

# ---------------------------------------------------------------------------
# Hilfen
# ---------------------------------------------------------------------------

_KONFIDENZ_RANG = {"niedrig": 0, "mittel": 1, "hoch": 2}

# Beleg einer Zuordnung -> Konfidenzstufe. "hoch" gibt es nur mit Layer- bzw. Blockname.
_BELEG_STUFE = {
    "layer": "hoch",
    "block": "hoch",
    "geometrie": "mittel",
    "unbekannt": "mittel",
    "standard": "niedrig",
}

# Werte der optionalen Markierung ``wall_type_source`` / ``bearing_source`` -> Beleg.
_QUELLE_ZU_BELEG = {
    "layer": "layer",
    "layername": "layer",
    "geometry": "geometrie",
    "geometrie": "geometrie",
    "heuristic": "geometrie",
    "heuristik": "geometrie",
    "label": "geometrie",
    "default": "standard",
    "standard": "standard",
    "fallback": "standard",
}

_LAGE_AUSSEN = {"exterior", "aussen", "außen", "external", "outer"}
_LAGE_INNEN = {"interior", "innen", "internal", "inner"}
_TRAGEND_JA = {"true", "tragend", "bearing", "load_bearing", "load-bearing", "ja", "yes", "1"}
_TRAGEND_NEIN = {
    "false",
    "nichttragend",
    "nicht tragend",
    "non_bearing",
    "non-bearing",
    "nonbearing",
    "nein",
    "no",
    "0",
}


def _kopie(wert: Any) -> Any:
    """Tiefe Kopie fuer JSON-Daten (ohne ``copy``, nur json)."""
    return json.loads(json.dumps(wert))


def _zahl(wert: Any) -> float | None:
    """Endliche Zahl oder None. Wahrheitswerte und NaN zaehlen nicht."""
    if wert is None or isinstance(wert, bool):
        return None
    try:
        z = float(wert)
    except (TypeError, ValueError):
        return None
    if z != z or z in (float("inf"), float("-inf")):
        return None
    return z


def _r(wert: float, stellen: int = 3) -> float:
    r = round(wert, stellen)
    return 0.0 if r == 0 else r


def _bool_oder_none(wert: Any) -> bool | None:
    if isinstance(wert, bool):
        return wert
    if isinstance(wert, (int, float)) and wert in (0, 1):
        return bool(wert)
    if isinstance(wert, str):
        s = wert.strip().lower()
        if s in _TRAGEND_JA:
            return True
        if s in _TRAGEND_NEIN:
            return False
    return None


def _lage_von(wall_type: Any) -> str:
    s = str(wall_type or "").strip().lower()
    if s in _LAGE_AUSSEN:
        return "aussen"
    if s in _LAGE_INNEN:
        return "innen"
    return "unbekannt"


def _suche(attrs: dict[str, Any], comp: dict[str, Any], schluessel: tuple[str, ...]) -> Any:
    """Liest einen optionalen, additiv ergaenzten Wert aus attrs, attrs.meta oder comp.meta."""
    quellen = [attrs]
    for kandidat in (attrs.get("meta"), comp.get("meta")):
        if isinstance(kandidat, dict):
            quellen.append(kandidat)
    for q in quellen:
        for k in schluessel:
            if k in q and q[k] is not None:
                return q[k]
    return None


def _beleg_aus_quelle(quelle: Any, ohne_markierung: str) -> str:
    if quelle is None:
        return ohne_markierung
    return _QUELLE_ZU_BELEG.get(str(quelle).strip().lower(), "unbekannt")


def _komponenten(scene: Any) -> list[tuple[str, dict[str, Any]]]:
    if hasattr(scene, "to_dict") and not isinstance(scene, dict):
        scene = scene.to_dict()
    if not isinstance(scene, dict):
        raise ValueError("scene muss ein Dict im Format scene_v1 sein")
    comps = scene.get("components")
    out: list[tuple[str, dict[str, Any]]] = []
    if isinstance(comps, dict):
        for key, c in comps.items():
            if isinstance(c, dict):
                out.append((str(c.get("id", key)), c))
    elif isinstance(comps, list):
        for i, c in enumerate(comps):
            if isinstance(c, dict):
                out.append((str(c.get("id", f"c{i}")), c))
    else:
        raise ValueError("scene enthält keine components")
    return out


def _dims(comp: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    d = comp.get("dims")
    if isinstance(d, (list, tuple)) and len(d) >= 3:
        return _zahl(d[0]), _zahl(d[1]), _zahl(d[2])
    return None, None, None


def _fmt_de(wert: float, stellen: int = 2) -> str:
    """Zahl fuer Anzeigetexte: deutsches Komma, ohne ueberfluessige Nullen."""
    s = f"{wert:.{stellen}f}".rstrip("0").rstrip(".")
    return (s or "0").replace(".", ",")


def _distinct(werte: list[float]) -> list[float]:
    return sorted({_r(w, 3) for w in werte})


# ---------------------------------------------------------------------------
# A. Mengenkern
# ---------------------------------------------------------------------------


def mengen_aus_scene(
    scene: Any,
    *,
    geschoss: str | None = None,
    wall_height_m: float | None = None,
    pipeline_meta: dict[str, Any] | None = None,
    erzeuger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mengen je Bauteilart aus einem scene-Dict (``scene_v1``).

    Wandzeilen im scene-Dict sind Kanten zwischen Knoten, keine physischen Waende. Die
    Mengen summieren Kanten. Oeffnungen haengen ueber ``parent_edge_id`` an einer Wandkante;
    ihre Flaeche (Breite mal wirksame Hoehe, Bruestung bis Sturz, auf die Wandhoehe begrenzt)
    wird von der Elternwand abgezogen. Oeffnungen ohne Elternwand werden nicht abgezogen
    und als Hinweis ausgewiesen.

    Parameter
    ---------
    geschoss        freier Text (z. B. ``"EG"``); die DXF kennt keine Geschosse.
    wall_height_m   ueberschreibt die Wandhoehe aller Waende. Ohne Angabe gilt die Hoehe im
                    scene-Dict, sonst ``STANDARD_WANDHOEHE_M``. In allen Faellen ist es eine
                    Annahme, keine Messung.
    pipeline_meta   optional, Inhalt der ``meta.json`` der Pipeline (fuer Hinweise auf
                    Oeffnungen, die keiner Wand zugeordnet werden konnten).
    """
    komponenten = _komponenten(scene)
    ueberschreibung = _zahl(wall_height_m)
    if ueberschreibung is not None and ueberschreibung <= 0:
        raise ValueError("wall_height_m muss größer als 0 sein")

    hinweise: list[dict[str, Any]] = []
    waende: dict[str, dict[str, Any]] = {}
    oeffnungs_komp: list[tuple[str, dict[str, Any]]] = []
    raum_komp: list[tuple[str, dict[str, Any]]] = []
    andere: dict[str, int] = {}
    hoehen_quellen: set[str] = set()

    # --- Waende ---------------------------------------------------------
    for cid, comp in komponenten:
        art = str(comp.get("kind", "")).upper()
        if art == "WALL":
            attrs = comp.get("attrs") if isinstance(comp.get("attrs"), dict) else {}
            dl, dh, dd = _dims(comp)
            laenge = _zahl(attrs.get("length_m")) or dl
            if laenge is None or laenge <= 0:
                hinweise.append(
                    {
                        "code": "WAND_OHNE_LAENGE",
                        "text": "Wandkante ohne Länge, nicht in den Mengen enthalten.",
                        "ids": [cid],
                    }
                )
                continue
            if ueberschreibung is not None:
                hoehe, hq = ueberschreibung, "parameter"
            else:
                h = _zahl(attrs.get("height_m")) or dh
                if h is not None and h > 0:
                    hoehe, hq = h, "szene"
                else:
                    hoehe, hq = STANDARD_WANDHOEHE_M, "standard"
            hoehen_quellen.add(hq)
            dicke = _zahl(attrs.get("thickness_m")) or dd
            lage = _lage_von(attrs.get("wall_type"))
            lage_quelle = _suche(attrs, comp, ("wall_type_source", "lage_quelle"))
            # Vertrag der Pipeline: "exterior" entsteht nur aus einem Layernamen, "interior" ist
            # auch der Standardwert. Ohne Markierung ist "aussen" daher belegt, "innen" nicht.
            if lage == "unbekannt":
                lage_beleg = "unbekannt"
            else:
                lage_beleg = _beleg_aus_quelle(
                    lage_quelle, "layer" if lage == "aussen" else "unbekannt"
                )
            tragend = _bool_oder_none(
                _suche(attrs, comp, ("bearing", "load_bearing", "is_load_bearing", "tragend"))
            )
            tragend_beleg = "unbekannt"
            if tragend is not None:
                # Vertrag der Pipeline: ``bearing`` ist nur gesetzt, wenn der Layername es sagt, und
                # ``layer`` nennt dann den Layer. Ohne Layerangabe und ohne bearing_source bleibt
                # das Tragverhalten unbelegt.
                hat_layer = bool(str(attrs.get("layer") or "").strip())
                tragend_beleg = _beleg_aus_quelle(
                    _suche(attrs, comp, ("bearing_source", "tragend_quelle")),
                    "layer" if hat_layer else "unbekannt",
                )
            brutto = laenge * hoehe
            waende[cid] = {
                "id": cid,
                "lage": lage,
                "tragend": tragend,
                "lage_beleg": lage_beleg,
                "tragend_beleg": tragend_beleg,
                "laenge_m": laenge,
                "hoehe_m": hoehe,
                "dicke_m": dicke,
                "flaeche_brutto_m2": brutto,
                "_abzug_roh": 0.0,
                "_breiten": 0.0,
                "oeffnungen": [],
            }
        elif art in ("DOOR", "WINDOW"):
            oeffnungs_komp.append((cid, comp))
        elif art == "ROOM":
            raum_komp.append((cid, comp))
        else:
            andere[art or "UNBEKANNT"] = andere.get(art or "UNBEKANNT", 0) + 1

    # --- Oeffnungen -----------------------------------------------------
    oeffnungen: list[dict[str, Any]] = []
    waisen: list[str] = []
    for cid, comp in oeffnungs_komp:
        attrs = comp.get("attrs") if isinstance(comp.get("attrs"), dict) else {}
        dl, dh, _dd = _dims(comp)
        breite = _zahl(attrs.get("width_m")) or dl
        bruestung = _zahl(attrs.get("sill_height_m"))
        sturz = _zahl(attrs.get("head_height_m"))
        hoehe_attr = _zahl(attrs.get("height_m")) or dh
        if bruestung is None:
            bruestung = 0.0
        if sturz is None and hoehe_attr is not None:
            sturz = bruestung + hoehe_attr
        if breite is None or breite <= 0 or sturz is None or sturz - bruestung <= 0:
            hinweise.append(
                {
                    "code": "OEFFNUNG_OHNE_MASS",
                    "text": "Öffnung ohne Breite oder Höhe, nicht in den Mengen enthalten.",
                    "ids": [cid],
                }
            )
            continue
        art = "tuer" if str(comp.get("kind", "")).upper() == "DOOR" else "fenster"
        eltern_id = attrs.get("parent_edge_id")
        eltern = waende.get(str(eltern_id)) if eltern_id is not None else None
        nominal_h = sturz - bruestung
        rec: dict[str, Any] = {
            "id": cid,
            "art": art,
            "eltern_id": str(eltern_id) if eltern_id is not None else None,
            "lage": eltern["lage"] if eltern else "unbekannt",
            "lage_beleg": eltern["lage_beleg"] if eltern else "unbekannt",
            "beleg": "block" if str(attrs.get("source", "")) == "dwg_block" else "unbekannt",
            "unterart": attrs.get("subtype") if isinstance(attrs.get("subtype"), str) else None,
            "breite_m": breite,
            "hoehe_m": nominal_h,
            "bruestung_m": bruestung,
            "sturz_m": sturz,
            "flaeche_m2": breite * nominal_h,
            "hoehe_wirksam_m": None,
            "abzug_m2": 0.0,
            "abgezogen": False,
        }
        if eltern is None:
            waisen.append(cid)
        else:
            wh = eltern["hoehe_m"]
            wirksam = max(0.0, min(sturz, wh) - max(bruestung, 0.0))
            rec["hoehe_wirksam_m"] = wirksam
            rec["abzug_m2"] = breite * wirksam
            rec["abgezogen"] = True
            eltern["_abzug_roh"] += breite * wirksam
            eltern["_breiten"] += breite
            eltern["oeffnungen"].append(cid)
        oeffnungen.append(rec)

    if waisen:
        hinweise.append(
            {
                "code": "OEFFNUNG_OHNE_WAND",
                "text": (
                    f"{len(waisen)} Öffnung(en) ohne zugeordnete Wandkante: nicht von einer "
                    "Wandfläche abgezogen, Lage außen oder innen unbekannt."
                ),
                "ids": waisen,
            }
        )
    unmatched = None
    if isinstance(pipeline_meta, dict):
        m = pipeline_meta.get("metrics") if isinstance(pipeline_meta.get("metrics"), dict) else {}
        op = m.get("openings") if isinstance(m.get("openings"), dict) else {}
        unmatched = _zahl(op.get("unmatched"))
    if unmatched:
        hinweise.append(
            {
                "code": "PIPELINE_OEFFNUNG_NICHT_ANGEHAENGT",
                "text": (
                    f"Die Pipeline hat {int(unmatched)} Tür-/Fensterblock/-blöcke keiner "
                    "Wandkante zuordnen können. Sie stehen weder im scene-Dict noch in den Mengen."
                ),
                "ids": [],
            }
        )

    # --- Wandsummen je Wand (mit gedeckeltem Abzug) ----------------------
    for w in waende.values():
        abzug = w.pop("_abzug_roh")
        breiten = w.pop("_breiten")
        brutto = w["flaeche_brutto_m2"]
        if breiten > w["laenge_m"] + 1e-9:
            hinweise.append(
                {
                    "code": "OEFFNUNGSBREITE_GROESSER_WAND",
                    "text": (
                        f"Die Öffnungen an Wandkante {w['id']} sind zusammen {_fmt_de(breiten)} m "
                        f"breit, die Kante ist {_fmt_de(w['laenge_m'])} m lang."
                    ),
                    "ids": [w["id"]] + list(w["oeffnungen"]),
                }
            )
        if abzug > brutto + 1e-9:
            hinweise.append(
                {
                    "code": "ABZUG_GROESSER_BRUTTO",
                    "text": (
                        f"Der Öffnungsabzug an Wandkante {w['id']} übersteigt die Bruttofläche; "
                        "der Abzug wurde auf die Bruttofläche begrenzt."
                    ),
                    "ids": [w["id"]],
                }
            )
            abzug = brutto
        w["oeffnungsabzug_m2"] = abzug
        w["flaeche_netto_m2"] = brutto - abzug

    # --- Wandgruppen ------------------------------------------------------
    def _wand_summe(liste: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "anzahl_kanten": len(liste),
            "laenge_m": _r(sum(w["laenge_m"] for w in liste)),
            "flaeche_brutto_m2": _r(sum(w["flaeche_brutto_m2"] for w in liste)),
            "oeffnungsabzug_m2": _r(sum(w["oeffnungsabzug_m2"] for w in liste)),
            "flaeche_netto_m2": _r(sum(w["flaeche_netto_m2"] for w in liste)),
            "hoehen_m": _distinct([w["hoehe_m"] for w in liste]),
        }

    gruppen_schluessel: dict[tuple[str, bool | None], list[dict[str, Any]]] = {}
    for w in waende.values():
        gruppen_schluessel.setdefault((w["lage"], w["tragend"]), []).append(w)
    lagen_reihenfolge = {"aussen": 0, "innen": 1, "unbekannt": 2}
    trag_reihenfolge = {True: 0, False: 1, None: 2}
    wand_gruppen: list[dict[str, Any]] = []
    for (lage, tragend), liste in sorted(
        gruppen_schluessel.items(),
        key=lambda kv: (lagen_reihenfolge[kv[0][0]], trag_reihenfolge[kv[0][1]]),
    ):
        teil = "" if tragend is None else ("_tragend" if tragend else "_nichttragend")
        belege_lage: dict[str, int] = {}
        belege_trag: dict[str, int] = {}
        for w in liste:
            belege_lage[w["lage_beleg"]] = belege_lage.get(w["lage_beleg"], 0) + 1
            if tragend is not None:
                belege_trag[w["tragend_beleg"]] = belege_trag.get(w["tragend_beleg"], 0) + 1
        g = {"id": f"wand_{lage}{teil}", "lage": lage, "tragend": tragend}
        g.update(_wand_summe(liste))
        g["belege"] = {"lage": belege_lage, "tragend": belege_trag}
        g["source_ids"] = [w["id"] for w in liste]
        wand_gruppen.append(g)
    je_lage: dict[str, Any] = {}
    for lage in ("aussen", "innen", "unbekannt"):
        liste = [w for w in waende.values() if w["lage"] == lage]
        if liste:
            je_lage[lage] = _wand_summe(liste)

    # --- Oeffnungsgruppen -------------------------------------------------
    og_schluessel: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for o in oeffnungen:
        og_schluessel.setdefault((o["art"], o["lage"]), []).append(o)
    art_reihenfolge = {"tuer": 0, "fenster": 1}
    oeffnungs_gruppen: list[dict[str, Any]] = []
    for (art, lage), liste in sorted(
        og_schluessel.items(),
        key=lambda kv: (art_reihenfolge[kv[0][0]], lagen_reihenfolge[kv[0][1]]),
    ):
        belege_lage = {}
        belege_oeff: dict[str, int] = {}
        for o in liste:
            belege_lage[o["lage_beleg"]] = belege_lage.get(o["lage_beleg"], 0) + 1
            belege_oeff[o["beleg"]] = belege_oeff.get(o["beleg"], 0) + 1
        oeffnungs_gruppen.append(
            {
                "id": f"{art}_{lage}",
                "art": art,
                "lage": lage,
                "anzahl": len(liste),
                "breite_summe_m": _r(sum(o["breite_m"] for o in liste)),
                "hoehen_m": _distinct([o["hoehe_m"] for o in liste]),
                "flaeche_m2": _r(sum(o["flaeche_m2"] for o in liste)),
                "abgezogen_m2": _r(sum(o["abzug_m2"] for o in liste)),
                "belege": {"lage": belege_lage, "oeffnung": belege_oeff},
                "source_ids": [o["id"] for o in liste],
            }
        )
    je_art: dict[str, Any] = {}
    for art in ("tuer", "fenster"):
        liste = [o for o in oeffnungen if o["art"] == art]
        if liste:
            je_art[art] = {
                "anzahl": len(liste),
                "breite_summe_m": _r(sum(o["breite_m"] for o in liste)),
                "flaeche_m2": _r(sum(o["flaeche_m2"] for o in liste)),
            }

    # --- Raeume -----------------------------------------------------------
    raeume: list[dict[str, Any]] = []
    for cid, comp in raum_komp:
        attrs = comp.get("attrs") if isinstance(comp.get("attrs"), dict) else {}
        netto = _zahl(attrs.get("area_net_m2"))
        roh = _zahl(attrs.get("area_m2"))
        if netto is not None and netto > 0:
            flaeche, art_f = netto, "netto"
        elif roh is not None and roh > 0:
            flaeche, art_f = roh, "area_m2"
        else:
            hinweise.append(
                {
                    "code": "RAUM_OHNE_FLAECHE",
                    "text": "Raum ohne Fläche, nicht in den Mengen enthalten.",
                    "ids": [cid],
                }
            )
            continue
        raeume.append(
            {
                "id": cid,
                "bezeichnung": attrs.get("label") if isinstance(attrs.get("label"), str) else None,
                "raumart": attrs.get("room_type")
                if isinstance(attrs.get("room_type"), str)
                else None,
                "flaeche_m2": flaeche,
                "flaeche_art": art_f,
            }
        )
    arten = {r["flaeche_art"] for r in raeume}
    if not raeume:
        flaeche_quelle = None
    elif arten == {"netto"}:
        flaeche_quelle = "area_net_m2"
    elif arten == {"area_m2"}:
        flaeche_quelle = "area_m2"
    else:
        flaeche_quelle = "gemischt"
    raum_hinweis = None
    if flaeche_quelle in ("area_m2", "gemischt"):
        raum_hinweis = (
            "Für mindestens einen Raum liegt nur area_m2 vor (Polygon der Raumkontur), keine "
            "Nettofläche. Die Summe ist deshalb keine Nettoraumfläche."
        )
        hinweise.append(
            {
                "code": "RAUM_FLAECHE_NICHT_NETTO",
                "text": raum_hinweis,
                "ids": [r["id"] for r in raeume if r["flaeche_art"] == "area_m2"],
            }
        )

    # --- Annahmen -----------------------------------------------------------
    annahmen: list[dict[str, Any]] = []
    hoehen_alle = _distinct([w["hoehe_m"] for w in waende.values()])
    if waende:
        if hoehen_quellen == {"parameter"}:
            art_h = "parameter"
            quelle_text = "als Parameter vorgegeben"
        elif hoehen_quellen == {"standard"}:
            art_h = "standardwert"
            quelle_text = f"Standardwert des Moduls ({_fmt_de(STANDARD_WANDHOEHE_M)} m)"
        else:
            art_h = "konfiguration"
            quelle_text = "aus dem Szenenmodell (Konfiguration der Pipeline)"
        werte_text = " / ".join(_fmt_de(h) for h in hoehen_alle)
        annahmen.append(
            {
                "id": "wandhoehe",
                "art": art_h,
                "wert": hoehen_alle[0] if len(hoehen_alle) == 1 else hoehen_alle,
                "einheit": "m",
                "text": (
                    f"Wandhöhe {werte_text} m, {quelle_text}. Das ist eine Annahme und nicht "
                    "gemessen: die DXF enthält keine Geschosshöhe."
                ),
            }
        )
    if oeffnungen:
        h_tuer = _distinct([o["hoehe_m"] for o in oeffnungen if o["art"] == "tuer"])
        h_fen = _distinct([o["hoehe_m"] for o in oeffnungen if o["art"] == "fenster"])
        teile = []
        if h_tuer:
            teile.append("Türhöhe " + " / ".join(_fmt_de(h) for h in h_tuer) + " m")
        if h_fen:
            b_fen = _distinct([o["bruestung_m"] for o in oeffnungen if o["art"] == "fenster"])
            teile.append(
                "Fensterhöhe "
                + " / ".join(_fmt_de(h) for h in h_fen)
                + " m (Brüstung "
                + " / ".join(_fmt_de(b) for b in b_fen)
                + " m)"
            )
        annahmen.append(
            {
                "id": "oeffnungshoehen",
                "art": "konfiguration",
                "wert": {"tuer_m": h_tuer, "fenster_m": h_fen},
                "einheit": "m",
                "text": (
                    "; ".join(teile)
                    + ". Höhen und Brüstungen stammen aus dem Szenenmodell (Standardwerte der "
                    "Pipeline-Konfiguration, solange der Block keine Höhe trägt), nicht gemessen. "
                    "Die Breite kommt aus dem Block oder ist ein Standardwert."
                ),
            }
        )
    annahmen.append(
        {
            "id": "netto_methode",
            "art": "methode",
            "wert": None,
            "einheit": None,
            "text": (
                "Wandfläche = Kantenlänge (Wandachse) mal Höhe. Netto = Brutto minus die "
                "verknüpften Öffnungen (Breite mal wirksame Höhe von Brüstung bis Sturz, auf die "
                "Wandhöhe begrenzt). Überlappungen an Ecken und Anschlüssen, Laibungen und "
                "Übermessungsregeln sind nicht berücksichtigt."
            ),
        }
    )
    annahmen.append(
        {
            "id": "wandkanten",
            "art": "methode",
            "wert": None,
            "einheit": None,
            "text": (
                "Wandzeilen sind Kanten zwischen Knoten des Wandgraphen. Eine physische Wand "
                "kann aus mehreren Kanten bestehen; die Summen addieren Kanten."
            ),
        }
    )
    annahmen.append(
        {
            "id": "lage_aussen_innen",
            "art": "methode",
            "wert": None,
            "einheit": None,
            "text": (
                "Außen oder innen übernimmt das Modul aus der Wandkomponente (wall_type). "
                "Die Pipeline setzt 'außen' nur aus dem Layernamen; 'innen' ist zugleich der "
                "Standardwert und deshalb ohne Markierung (wall_type_source) nicht belegt. "
                "Tragend oder nichttragend wird nur ausgewiesen, wenn die Wandkomponente es angibt."
            ),
        }
    )
    annahmen.append(
        {
            "id": "geschoss",
            "art": "parameter" if geschoss else "offen",
            "wert": geschoss,
            "einheit": None,
            "text": (
                f"Geschoss '{geschoss}' ist ein Parameter; die DXF kennt keine Geschosse."
                if geschoss
                else "Kein Geschoss angegeben; die DXF kennt keine Geschosse."
            ),
        }
    )

    # --- nicht verarbeitet ------------------------------------------------
    nicht_abgedeckt = [
        {"element_class": e["element_class"], "label": e["label"], "reason": e["reason"]}
        for e in KG_RULES["not_covered"]
    ]
    nicht_verarbeitet = {k: andere[k] for k in sorted(andere)}
    if nicht_verarbeitet:
        hinweise.append(
            {
                "code": "BAUTEILE_NICHT_VERARBEITET",
                "text": (
                    "Das scene-Dict enthält Komponenten, für die dieses Modul keine Mengen bildet: "
                    + ", ".join(f"{k} ({n})" for k, n in nicht_verarbeitet.items())
                    + "."
                ),
                "ids": [],
            }
        )

    szene = scene.to_dict() if hasattr(scene, "to_dict") and not isinstance(scene, dict) else scene
    return {
        "schema": MENGEN_SCHEMA,
        "schema_version": MENGEN_SCHEMA_VERSION,
        "erzeugt_von": dict(erzeuger) if erzeuger else {"name": "kg.zuordnung"},
        "geschoss": geschoss,
        "einheiten": {"laenge": "m", "flaeche": "m2"},
        "quelle": {
            "scene_version": szene.get("version") if isinstance(szene, dict) else None,
            "scene_einheit": szene.get("units") if isinstance(szene, dict) else None,
            "komponenten": len(komponenten),
        },
        "annahmen": annahmen,
        "waende": {
            "gruppen": wand_gruppen,
            "je_lage": je_lage,
            "summe": _wand_summe(list(waende.values())),
        },
        "oeffnungen": {
            "gruppen": oeffnungs_gruppen,
            "je_art": je_art,
            "ohne_wand": len(waisen),
        },
        "raeume": {
            "anzahl": len(raeume),
            "flaeche_m2": _r(sum(r["flaeche_m2"] for r in raeume)),
            "flaeche_quelle": flaeche_quelle,
            "hinweis": raum_hinweis,
            "kostengruppe": None,
            "bezug": ("Bezugsfläche ohne Kostengruppe. Raumflächen gehören fachlich zu DIN 277."),
        },
        "hinweise": hinweise,
        "nicht_abgedeckt": nicht_abgedeckt,
        "nicht_verarbeitet": nicht_verarbeitet,
        "elemente": {
            "waende": [_wand_ausgabe(w) for w in waende.values()],
            "oeffnungen": [_oeffnung_ausgabe(o) for o in oeffnungen],
            "raeume": [_raum_ausgabe(r) for r in raeume],
        },
    }


def _wand_ausgabe(w: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": w["id"],
        "lage": w["lage"],
        "tragend": w["tragend"],
        "lage_beleg": w["lage_beleg"],
        "tragend_beleg": w["tragend_beleg"],
        "laenge_m": _r(w["laenge_m"]),
        "hoehe_m": _r(w["hoehe_m"]),
        "dicke_m": _r(w["dicke_m"]) if w["dicke_m"] is not None else None,
        "flaeche_brutto_m2": _r(w["flaeche_brutto_m2"]),
        "oeffnungsabzug_m2": _r(w["oeffnungsabzug_m2"]),
        "flaeche_netto_m2": _r(w["flaeche_netto_m2"]),
        "oeffnungen": list(w["oeffnungen"]),
    }


def _oeffnung_ausgabe(o: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": o["id"],
        "art": o["art"],
        "unterart": o["unterart"],
        "lage": o["lage"],
        "eltern_id": o["eltern_id"],
        "abgezogen": o["abgezogen"],
        "breite_m": _r(o["breite_m"]),
        "hoehe_m": _r(o["hoehe_m"]),
        "bruestung_m": _r(o["bruestung_m"]),
        "sturz_m": _r(o["sturz_m"]),
        "hoehe_wirksam_m": _r(o["hoehe_wirksam_m"]) if o["hoehe_wirksam_m"] is not None else None,
        "flaeche_m2": _r(o["flaeche_m2"]),
        "abzug_m2": _r(o["abzug_m2"]),
    }


def _raum_ausgabe(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": r["id"],
        "bezeichnung": r["bezeichnung"],
        "raumart": r["raumart"],
        "flaeche_m2": _r(r["flaeche_m2"]),
        "flaeche_art": r["flaeche_art"],
    }


# ---------------------------------------------------------------------------
# B. Kostengruppen-Vorschlag
# ---------------------------------------------------------------------------


def _regel_index() -> dict[tuple[str, str, bool | None], dict[str, Any]]:
    return {(r["component"], r["location"], r["bearing"]): r for r in KG_RULES["rules"]}


def _schwaechste(stufen: list[str]) -> str:
    if not stufen:
        return "niedrig"
    return min(stufen, key=lambda s: _KONFIDENZ_RANG[s])


def _stufe(belege: dict[str, int]) -> str:
    return _schwaechste([_BELEG_STUFE.get(k, "mittel") for k, n in belege.items() if n])


def _annahme_texte(mengen: dict[str, Any], ids: tuple[str, ...]) -> list[str]:
    return [a["text"] for a in mengen.get("annahmen", []) if a.get("id") in ids]


def _beleg_satz_wand(konfidenz_lage: str, tragend: bool | None, konfidenz_trag: str | None) -> str:
    if konfidenz_lage == "hoch":
        satz = "Außen oder innen steht im Layernamen."
    elif konfidenz_lage == "mittel":
        satz = "Außen oder innen ist nicht durch einen Layernamen belegt."
    else:
        satz = "Außen oder innen ist ein Standardwert ohne Beleg."
    if tragend is not None:
        if konfidenz_trag == "hoch":
            satz += " Das Tragverhalten steht im Layernamen."
        else:
            satz += " Das Tragverhalten ist angegeben, aber nicht durch einen Layernamen belegt."
    return satz


def _beleg_satz_oeffnung(konfidenz_lage: str, konfidenz_oeff: str) -> str:
    if konfidenz_lage == "hoch":
        satz = "Außen oder innen folgt der Elternwand (Layername)."
    else:
        satz = "Außen oder innen folgt der Elternwand, dort ohne Beleg durch einen Layernamen."
    if konfidenz_oeff == "hoch":
        satz += " Tür oder Fenster steht im Blocknamen."
    else:
        satz += " Tür oder Fenster ist nicht durch einen Blocknamen belegt."
    return satz


def _zusammenfassung(regel: dict[str, Any], geschoss: str | None, menge: float) -> str:
    teile = [regel["label_plural"]]
    if geschoss:
        teile.append(geschoss)
    if regel["unit"] == "Stk":
        teile.append(f"{_fmt_de(menge, 0)} Stk")
    else:
        teile.append(f"{_fmt_de(menge)} m2 netto")
    return " ".join(teile) + f", KG {regel['kg']}, Vorschlag zur Freigabe"


def vorschlaege_aus_mengen(mengen: dict[str, Any]) -> dict[str, Any]:
    """Kostengruppen-Vorschlaege fuer ein Mengen-Dict (ein Geschoss).

    Liefert ``proposals`` (Vorschlaege mit ``status = "vorschlag"``), ``reference_quantities``
    (Raeume, ohne Kostengruppe), ``unassigned`` (Mengen, fuer die keine Nummer vorgeschlagen
    wird, z. B. unbekannte Lage), ``not_covered`` und ``hinweise``.
    """
    if not isinstance(mengen, dict) or mengen.get("schema") != MENGEN_SCHEMA:
        raise ValueError("kein Mengen-Dict (schema 'planproof_cad.mengen' erwartet)")
    geschoss = mengen.get("geschoss")
    index = _regel_index()
    proposals: list[dict[str, Any]] = []
    unassigned: list[dict[str, Any]] = []
    ohne_wand = int(mengen.get("oeffnungen", {}).get("ohne_wand", 0) or 0)
    waisen_satz = (
        f"{ohne_wand} Öffnung(en) ohne zugeordnete Wandkante sind in keiner Wandfläche abgezogen."
        if ohne_wand
        else None
    )
    geschoss_satz = _annahme_texte(mengen, ("geschoss",))

    for g in mengen.get("waende", {}).get("gruppen", []):
        regel = index.get(("wand", g["lage"], g["tragend"]))
        netto = float(g["flaeche_netto_m2"])
        if regel is None:
            unassigned.append(
                {
                    "component": "wand",
                    "quantity": netto,
                    "unit": "m2",
                    "source_ids": list(g["source_ids"]),
                    "reason": "Außen oder innen ist nicht bekannt: keine Nummer vorgeschlagen.",
                }
            )
            continue
        k_lage = _stufe(g["belege"]["lage"])
        k_trag = _stufe(g["belege"]["tragend"]) if g["tragend"] is not None else None
        konfidenz = _schwaechste([k for k in (k_lage, k_trag) if k])
        assumptions = _annahme_texte(mengen, ("wandhoehe", "netto_methode", "wandkanten"))
        assumptions += geschoss_satz
        if waisen_satz:
            assumptions.append(waisen_satz)
        proposals.append(
            _vorschlag(
                regel,
                geschoss,
                netto,
                konfidenz,
                regel["rationale"] + " " + _beleg_satz_wand(k_lage, g["tragend"], k_trag),
                assumptions,
                g["source_ids"],
                {
                    "kanten": g["anzahl_kanten"],
                    "laenge_m": g["laenge_m"],
                    "flaeche_brutto_m2": g["flaeche_brutto_m2"],
                    "oeffnungsabzug_m2": g["oeffnungsabzug_m2"],
                    "flaeche_netto_m2": g["flaeche_netto_m2"],
                },
            )
        )

    for g in mengen.get("oeffnungen", {}).get("gruppen", []):
        regel = index.get((g["art"], g["lage"], None))
        if regel is None:
            unassigned.append(
                {
                    "component": g["art"],
                    "quantity": g["anzahl"],
                    "unit": "Stk",
                    "source_ids": list(g["source_ids"]),
                    "reason": (
                        "Die Öffnung hängt an keiner Wandkante, außen oder innen ist nicht "
                        "bekannt: es wird keine Nummer vorgeschlagen."
                    ),
                }
            )
            continue
        k_lage = _stufe(g["belege"]["lage"])
        k_oeff = _stufe(g["belege"]["oeffnung"])
        konfidenz = _schwaechste([k_lage, k_oeff])
        assumptions = _annahme_texte(mengen, ("oeffnungshoehen", "netto_methode"))
        assumptions += geschoss_satz
        proposals.append(
            _vorschlag(
                regel,
                geschoss,
                float(g["anzahl"]),
                konfidenz,
                regel["rationale"] + " " + _beleg_satz_oeffnung(k_lage, k_oeff),
                assumptions,
                g["source_ids"],
                {
                    "breite_summe_m": g["breite_summe_m"],
                    "flaeche_m2": g["flaeche_m2"],
                    "abgezogen_m2": g["abgezogen_m2"],
                },
            )
        )

    proposals.sort(key=lambda p: (p["kg"], p["element_class"]))

    reference: list[dict[str, Any]] = []
    raeume = mengen.get("raeume", {})
    if raeume.get("anzahl"):
        info = KG_RULES["no_cost_group"][0]
        reference.append(
            {
                "element_class": info["element_class"],
                "label": info["label"],
                "kg": None,
                "geschoss": geschoss,
                "count": raeume["anzahl"],
                "quantity": raeume["flaeche_m2"],
                "unit": "m2",
                "area_basis": raeume.get("flaeche_quelle"),
                "note": info["reason"],
                "assumptions": [raeume["hinweis"]] if raeume.get("hinweis") else [],
                "source_ids": [r["id"] for r in mengen.get("elemente", {}).get("raeume", [])],
            }
        )

    return {
        "schema": VORSCHLAG_SCHEMA,
        "schema_version": VORSCHLAG_SCHEMA_VERSION,
        "rules_version": KG_RULES["version"],
        "geschoss": geschoss,
        "status_hinweis": (
            "Alle Einträge sind Vorschläge zur Freigabe. Ein Mensch prüft und gibt frei; "
            "dieses Modul legt nichts an."
        ),
        "proposals": proposals,
        "reference_quantities": reference,
        "unassigned": unassigned,
        "not_covered": _kopie(mengen.get("nicht_abgedeckt") or KG_RULES["not_covered"]),
        "hinweise": _kopie(mengen.get("hinweise", [])),
    }


def _vorschlag(
    regel: dict[str, Any],
    geschoss: str | None,
    menge: float,
    konfidenz: str,
    rationale: str,
    assumptions: list[str],
    source_ids: list[str],
    detail: dict[str, Any],
) -> dict[str, Any]:
    menge_r = _r(menge, 2) if regel["unit"] == "m2" else int(round(menge))
    return {
        "id": "kgv_" + (f"{geschoss}_" if geschoss else "") + regel["id"],
        "kg": regel["kg"],
        "element_class": regel["element_class"],
        "label": regel["label"],
        "geschoss": geschoss,
        "basis": regel["basis"],
        "quantity": menge_r,
        "unit": regel["unit"],
        "confidence": konfidenz,
        "rationale": rationale,
        "assumptions": assumptions,
        "source_ids": list(source_ids),
        "quantity_detail": detail,
        "summary": _zusammenfassung(regel, geschoss, menge_r),
        "status": "vorschlag",
    }


def vorschlaege_aus_scene(
    scene: Any,
    *,
    geschoss: str | None = None,
    wall_height_m: float | None = None,
    pipeline_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mengen aus dem scene-Dict und daraus die Vorschlaege (ein Aufruf)."""
    mengen = mengen_aus_scene(
        scene, geschoss=geschoss, wall_height_m=wall_height_m, pipeline_meta=pipeline_meta
    )
    return vorschlaege_aus_mengen(mengen)


def vorschlaege_zusammenfassen(dokumente: list[dict[str, Any]]) -> dict[str, Any]:
    """Fasst die Vorschlaege mehrerer Geschosse zusammen (Summe je Nummer und Bauteilklasse).

    Die Vorschlaege der einzelnen Geschosse bleiben erhalten. ``summen`` addiert gleiche
    Nummer und Bauteilklasse ueber alle Geschosse; die Konfidenz ist die schwaechste.
    """
    if not dokumente:
        raise ValueError("keine Vorschlags-Dokumente übergeben")
    proposals: list[dict[str, Any]] = []
    referenz: list[dict[str, Any]] = []
    unassigned: list[dict[str, Any]] = []
    hinweise: list[dict[str, Any]] = []
    for d in dokumente:
        proposals += d["proposals"]
        referenz += d["reference_quantities"]
        for u in d["unassigned"]:
            unassigned.append({**u, "geschoss": d.get("geschoss")})
        for h in d["hinweise"]:
            hinweise.append({**h, "geschoss": d.get("geschoss")})
    summen: dict[tuple[str, str], dict[str, Any]] = {}
    for p in proposals:
        k = (p["kg"], p["element_class"])
        s = summen.setdefault(
            k,
            {
                "kg": p["kg"],
                "element_class": p["element_class"],
                "label": p["label"],
                "basis": p["basis"],
                "quantity": 0.0,
                "unit": p["unit"],
                "confidence": p["confidence"],
                "geschosse": [],
                "source_ids": [],
                "status": "vorschlag",
            },
        )
        s["quantity"] += p["quantity"]
        s["confidence"] = _schwaechste([s["confidence"], p["confidence"]])
        s["geschosse"].append(p["geschoss"])
        s["source_ids"] += p["source_ids"]
    summen_liste = []
    for k in sorted(summen):
        s = summen[k]
        s["quantity"] = int(round(s["quantity"])) if s["unit"] == "Stk" else _r(s["quantity"], 2)
        summen_liste.append(s)
    erstes = dokumente[0]
    return {
        "schema": VORSCHLAG_SCHEMA,
        "schema_version": VORSCHLAG_SCHEMA_VERSION,
        "rules_version": erstes["rules_version"],
        "geschoss": None,
        "geschosse": [d.get("geschoss") for d in dokumente],
        "status_hinweis": erstes["status_hinweis"],
        "proposals": proposals,
        "summen": summen_liste,
        "reference_quantities": referenz,
        "unassigned": unassigned,
        "not_covered": _kopie(erstes["not_covered"]),
        "hinweise": hinweise,
    }


__all__ = [
    "KG_RULES",
    "MENGEN_SCHEMA",
    "MENGEN_SCHEMA_VERSION",
    "STANDARD_WANDHOEHE_M",
    "VORSCHLAG_SCHEMA",
    "VORSCHLAG_SCHEMA_VERSION",
    "mengen_aus_scene",
    "vorschlaege_aus_mengen",
    "vorschlaege_aus_scene",
    "vorschlaege_zusammenfassen",
]
