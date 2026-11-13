"""Mengenkern: Rechenregeln gegen von Hand gerechnete Werte.

Die Soll-Werte stehen im Test als Rechnung, nicht aus einem Lauf des Codes abgeschrieben.
Fixture ``scene_haus_eg.json`` (Wandhoehe 2,80 m), Spalten: Laenge, brutto, Abzug, netto.

  Aussen tragend      w_a1 10,0 + w_a2 8,0 = 18 m   50,4   3,9 + 3,0   43,5
  Aussen nichttragend w_a3 4,0 m                    11,2   0           11,2
  Aussen ohne Angabe  w_a4 6,0 m                    16,8   2,8 (gekappt) 14,0
  Innen tragend       w_i1 5,0 m                    14,0   0,8         13,2
  Innen nichttragend  w_i2 3,0 m                     8,4   1,89         6,51
  Innen ohne Angabe   w_i3 2,5 m                     7,0   2,1          4,9
  Lage unbekannt      w_x  1,0 m                     2,8   1,89         0,91
"""

from __future__ import annotations

import pytest
from mengen_support import DEMO_SCENE, HAUS_EG, HAUS_OG, lade, oeffnung, szene, wand

from planproof_cad.kg.zuordnung import MENGEN_SCHEMA_VERSION, mengen_aus_scene
from planproof_cad.quantities import berechne_mengen, lade_pipeline_meta, lade_scene


def _gruppen(m: dict) -> dict:
    return {g["id"]: g for g in m["waende"]["gruppen"]}


def _o_gruppen(m: dict) -> dict:
    return {g["id"]: g for g in m["oeffnungen"]["gruppen"]}


@pytest.fixture(scope="module")
def eg() -> dict:
    return berechne_mengen(lade(HAUS_EG), geschoss="EG")


def test_schema_und_geschoss(eg):
    assert eg["schema"] == "planproof_cad.mengen"
    assert eg["schema_version"] == MENGEN_SCHEMA_VERSION == "mengen_v1"
    assert eg["geschoss"] == "EG"
    assert eg["erzeugt_von"]["name"] == "planproof_cad"
    assert eg["einheiten"] == {"laenge": "m", "flaeche": "m2"}


def test_wandgruppen_nach_lage_und_tragverhalten(eg):
    g = _gruppen(eg)
    assert set(g) == {
        "wand_aussen_tragend",
        "wand_aussen_nichttragend",
        "wand_aussen",
        "wand_innen_tragend",
        "wand_innen_nichttragend",
        "wand_innen",
        "wand_unbekannt",
    }
    soll = {
        # id: (Kanten, Laenge, brutto, Abzug, netto)
        "wand_aussen_tragend": (2, 18.0, 18 * 2.8, 2.1 + 1.5 * 1.2 + 2.0 * 1.5, None),
        "wand_aussen_nichttragend": (1, 4.0, 4 * 2.8, 0.0, 11.2),
        "wand_aussen": (1, 6.0, 6 * 2.8, 1.0 * 2.8, 14.0),
        "wand_innen_tragend": (1, 5.0, 5 * 2.8, 0.8 * 1.0, 13.2),
        "wand_innen_nichttragend": (1, 3.0, 3 * 2.8, 0.9 * 2.1, 6.51),
        "wand_innen": (1, 2.5, 2.5 * 2.8, 1.0 * 2.1, 4.9),
        "wand_unbekannt": (1, 1.0, 1.0 * 2.8, 0.9 * 2.1, 0.91),
    }
    for gid, (kanten, laenge, brutto, abzug, netto) in soll.items():
        x = g[gid]
        assert x["anzahl_kanten"] == kanten, gid
        assert x["laenge_m"] == pytest.approx(laenge), gid
        assert x["flaeche_brutto_m2"] == pytest.approx(brutto, abs=1e-3), gid
        assert x["oeffnungsabzug_m2"] == pytest.approx(abzug, abs=1e-3), gid
        assert x["flaeche_netto_m2"] == pytest.approx(brutto - abzug, abs=1e-3), gid
        if netto is not None:
            assert x["flaeche_netto_m2"] == pytest.approx(netto, abs=1e-3), gid
    assert g["wand_aussen_tragend"]["flaeche_netto_m2"] == pytest.approx(43.5, abs=1e-3)
    assert g["wand_aussen_tragend"]["source_ids"] == ["w_a1", "w_a2"]


def test_summen_je_lage(eg):
    je = eg["waende"]["je_lage"]
    assert je["aussen"]["flaeche_netto_m2"] == pytest.approx(43.5 + 11.2 + 14.0, abs=1e-3)
    assert je["innen"]["flaeche_netto_m2"] == pytest.approx(13.2 + 6.51 + 4.9, abs=1e-3)
    assert je["aussen"]["anzahl_kanten"] == 4
    assert eg["waende"]["summe"]["anzahl_kanten"] == 8
    assert eg["waende"]["summe"]["laenge_m"] == pytest.approx(10 + 8 + 4 + 6 + 5 + 3 + 2.5 + 1)


def test_oeffnung_wird_auf_wandhoehe_gekappt(eg):
    # f_a3: Bruestung 0, Sturz 3,5 m an einer 2,8 m hohen Wand: Nennflaeche 3,5, Abzug 2,8.
    zeile = next(o for o in eg["elemente"]["oeffnungen"] if o["id"] == "f_a3")
    assert zeile["flaeche_m2"] == pytest.approx(3.5)
    assert zeile["hoehe_wirksam_m"] == pytest.approx(2.8)
    assert zeile["abzug_m2"] == pytest.approx(2.8)


def test_oeffnungen_je_art_und_lage(eg):
    o = _o_gruppen(eg)
    assert set(o) == {
        "tuer_aussen",
        "tuer_innen",
        "tuer_unbekannt",
        "fenster_aussen",
        "fenster_innen",
    }
    assert o["tuer_aussen"]["anzahl"] == 1
    assert o["tuer_aussen"]["flaeche_m2"] == pytest.approx(2.1)
    assert o["fenster_aussen"]["anzahl"] == 3
    assert o["fenster_aussen"]["breite_summe_m"] == pytest.approx(1.5 + 2.0 + 1.0)
    assert o["fenster_aussen"]["flaeche_m2"] == pytest.approx(1.5 * 1.2 + 2.0 * 1.5 + 1.0 * 3.5)
    assert o["fenster_aussen"]["abgezogen_m2"] == pytest.approx(1.5 * 1.2 + 2.0 * 1.5 + 2.8)
    assert o["tuer_innen"]["anzahl"] == 2
    assert o["tuer_innen"]["flaeche_m2"] == pytest.approx(0.9 * 2.1 + 1.0 * 2.1)
    assert o["fenster_innen"]["anzahl"] == 1
    assert o["fenster_innen"]["hoehen_m"] == [1.0]
    assert eg["oeffnungen"]["je_art"]["tuer"]["anzahl"] == 1 + 2 + 3
    assert eg["oeffnungen"]["je_art"]["fenster"]["anzahl"] == 4


def test_oeffnung_ohne_elternwand_wird_nicht_abgezogen_und_ausgewiesen(eg):
    # Zwei Waisen: Eltern-ID verweist ins Leere (d_x) bzw. fehlt (d_y).
    assert eg["oeffnungen"]["ohne_wand"] == 2
    hinweis = next(h for h in eg["hinweise"] if h["code"] == "OEFFNUNG_OHNE_WAND")
    assert sorted(hinweis["ids"]) == ["d_x", "d_y"]
    # Die Waisen stehen in der unbekannten Tuergruppe, ohne Abzug; d_w (an w_x) wird abgezogen.
    g = _o_gruppen(eg)["tuer_unbekannt"]
    assert g["anzahl"] == 3
    assert g["abgezogen_m2"] == pytest.approx(0.9 * 2.1)
    for o in eg["elemente"]["oeffnungen"]:
        if o["id"] in ("d_x", "d_y"):
            assert o["abgezogen"] is False and o["abzug_m2"] == 0.0 and o["eltern_id"] != "w_a1"
    # Gesamter Abzug der Waende = Summe der Abzuege der verknuepften Oeffnungen.
    verknuepft = sum(o["abzug_m2"] for o in eg["elemente"]["oeffnungen"] if o["abgezogen"])
    assert eg["waende"]["summe"]["oeffnungsabzug_m2"] == pytest.approx(verknuepft, abs=1e-3)


def test_raeume_flaeche_netto_wenn_vorhanden_sonst_area_m2_mit_hinweis(eg):
    r = eg["raeume"]
    assert r["anzahl"] == 3
    assert r["flaeche_m2"] == pytest.approx(20.0 + 14.8 + 12.0)
    assert r["flaeche_quelle"] == "gemischt"
    assert r["kostengruppe"] is None
    assert "DIN 277" in r["bezug"]
    arten = {x["id"]: x["flaeche_art"] for x in eg["elemente"]["raeume"]}
    assert arten == {"r1": "area_m2", "r2": "netto", "r3": "area_m2"}
    hinweis = next(h for h in eg["hinweise"] if h["code"] == "RAUM_FLAECHE_NICHT_NETTO")
    assert sorted(hinweis["ids"]) == ["r1", "r3"]


def test_raeume_nur_netto_ohne_hinweis():
    m = berechne_mengen(lade(HAUS_OG), geschoss="OG")
    assert m["raeume"]["flaeche_quelle"] == "area_net_m2"
    assert m["raeume"]["flaeche_m2"] == pytest.approx(43.0)
    assert m["raeume"]["hinweis"] is None
    assert not [h for h in m["hinweise"] if h["code"] == "RAUM_FLAECHE_NICHT_NETTO"]


def test_zweites_geschoss_eigene_hoehe():
    m = berechne_mengen(lade(HAUS_OG), geschoss="OG")
    g = _gruppen(m)
    assert g["wand_aussen_tragend"]["flaeche_brutto_m2"] == pytest.approx(12 * 3.0)
    assert g["wand_aussen_tragend"]["flaeche_netto_m2"] == pytest.approx(36.0 - 1.2 * 1.2)
    assert g["wand_innen"]["flaeche_netto_m2"] == pytest.approx(18.0 - 0.9 * 2.1)
    assert g["wand_aussen_tragend"]["hoehen_m"] == [3.0]


def test_wandhoehe_parameter_ueberschreibt_und_ist_als_parameter_ausgewiesen():
    m = berechne_mengen(lade(HAUS_EG), geschoss="EG", wall_height_m=2.5)
    g = _gruppen(m)
    # w_a4: 6 m * 2,5 m = 15; f_a3 hat Sturz 3,5, wird auf 2,5 gekappt: Abzug 1,0 * 2,5.
    assert g["wand_aussen"]["flaeche_brutto_m2"] == pytest.approx(15.0)
    assert g["wand_aussen"]["flaeche_netto_m2"] == pytest.approx(12.5)
    a = next(x for x in m["annahmen"] if x["id"] == "wandhoehe")
    assert a["art"] == "parameter" and a["wert"] == 2.5


def test_wandhoehe_aus_szene_ist_konfiguration_nicht_messung(eg):
    a = next(x for x in eg["annahmen"] if x["id"] == "wandhoehe")
    assert a["art"] == "konfiguration"
    assert a["wert"] == 2.8
    assert "nicht gemessen" in a["text"]


def test_wandhoehe_standardwert_wenn_szene_keine_hoehe_hat():
    w = {"id": "w", "kind": "WALL", "attrs": {"length_m": 5.0, "wall_type": "interior"}}
    m = mengen_aus_scene(szene(w))
    a = next(x for x in m["annahmen"] if x["id"] == "wandhoehe")
    assert a["art"] == "standardwert" and a["wert"] == 2.8
    assert m["waende"]["summe"]["flaeche_brutto_m2"] == pytest.approx(14.0)


def test_alle_annahmen_ausgewiesen(eg):
    ids = {a["id"] for a in eg["annahmen"]}
    assert {
        "wandhoehe",
        "oeffnungshoehen",
        "netto_methode",
        "wandkanten",
        "lage_aussen_innen",
        "geschoss",
    } <= ids
    kanten = next(a for a in eg["annahmen"] if a["id"] == "wandkanten")
    assert "mehreren Kanten" in kanten["text"]
    h = next(a for a in eg["annahmen"] if a["id"] == "oeffnungshoehen")
    assert "nicht gemessen" in h["text"]


def test_nicht_abgedeckt_und_nicht_verarbeitet(eg):
    labels = [e["label"] for e in eg["nicht_abgedeckt"]]
    assert labels == ["Stützen", "Treppen", "Decken", "Dächer"]
    assert eg["nicht_verarbeitet"] == {"COLUMN": 1}
    assert any(h["code"] == "BAUTEILE_NICHT_VERARBEITET" for h in eg["hinweise"])


def test_demo_szene_eingefroren():
    m = berechne_mengen(lade(DEMO_SCENE), geschoss="EG")
    g = _gruppen(m)
    # Hand: aussen 36 m * 2,8 = 100,8; Tuer 1,01*2,1; Fenster 1,26*1,2 + 1,01*1,2 + 1,26*1,2
    abzug_aussen = 1.01 * 2.1 + 1.26 * 1.2 + 1.01 * 1.2 + 1.26 * 1.2
    assert g["wand_aussen"]["flaeche_brutto_m2"] == pytest.approx(100.8)
    assert g["wand_aussen"]["oeffnungsabzug_m2"] == pytest.approx(abzug_aussen, abs=1e-3)
    abzug_innen = 0.885 * 2.1 + 1.01 * 2.1
    assert g["wand_innen"]["flaeche_netto_m2"] == pytest.approx(14 * 2.8 - abzug_innen, abs=1e-3)
    assert m["raeume"]["anzahl"] == 3
    assert m["raeume"]["flaeche_m2"] == pytest.approx(80.0)
    assert m["oeffnungen"]["ohne_wand"] == 0
    # Ohne Layer-Markierung in der Szene: aussen gilt als belegt, innen nicht.
    assert g["wand_aussen"]["belege"]["lage"] == {"layer": 7}
    assert g["wand_innen"]["belege"]["lage"] == {"unbekannt": 3}


def test_pipeline_meta_unmatched_wird_ausgewiesen():
    meta = lade_pipeline_meta(DEMO_SCENE)
    assert meta is not None and meta["metrics"]["openings"]["unmatched"] == 0
    m = berechne_mengen(lade_scene(DEMO_SCENE), pipeline_meta=meta)
    assert not [h for h in m["hinweise"] if h["code"] == "PIPELINE_OEFFNUNG_NICHT_ANGEHAENGT"]
    meta["metrics"]["openings"]["unmatched"] = 2
    m = berechne_mengen(lade_scene(DEMO_SCENE), pipeline_meta=meta)
    h = next(h for h in m["hinweise"] if h["code"] == "PIPELINE_OEFFNUNG_NICHT_ANGEHAENGT")
    assert "2" in h["text"]


def test_lade_pipeline_meta_ohne_datei(tmp_path):
    (tmp_path / "scene.json").write_text("{}", encoding="utf-8")
    assert lade_pipeline_meta(tmp_path / "scene.json") is None


# --- Grenzfaelle -------------------------------------------------------------


def test_oeffnungen_groesser_als_wand_werden_gekappt_und_gemeldet():
    s = szene(
        wand("w", 1.0, "interior"),
        oeffnung("d1", "DOOR", "w", 0.9, 0.0, 2.1),
        oeffnung("d2", "DOOR", "w", 0.9, 0.0, 2.1),
        oeffnung("f1", "WINDOW", "w", 2.0, 0.0, 2.8),
    )
    m = mengen_aus_scene(s)
    codes = {h["code"] for h in m["hinweise"]}
    assert {"OEFFNUNGSBREITE_GROESSER_WAND", "ABZUG_GROESSER_BRUTTO"} <= codes
    w = m["elemente"]["waende"][0]
    assert w["flaeche_brutto_m2"] == pytest.approx(2.8)
    assert w["oeffnungsabzug_m2"] == pytest.approx(2.8)
    assert w["flaeche_netto_m2"] == 0.0


def test_wand_ohne_laenge_und_oeffnung_ohne_mass_werden_gemeldet():
    s = szene(
        {"id": "w0", "kind": "WALL", "attrs": {"wall_type": "interior"}},
        wand("w1", 4.0, "interior"),
        {"id": "o0", "kind": "DOOR", "attrs": {"parent_edge_id": "w1"}},
    )
    m = mengen_aus_scene(s)
    codes = {h["code"] for h in m["hinweise"]}
    assert {"WAND_OHNE_LAENGE", "OEFFNUNG_OHNE_MASS"} <= codes
    assert m["waende"]["summe"]["anzahl_kanten"] == 1
    assert m["oeffnungen"]["gruppen"] == []


def test_leere_szene_liefert_leere_mengen():
    m = mengen_aus_scene({"version": "scene_v1", "components": {}})
    assert m["waende"]["gruppen"] == []
    assert m["waende"]["summe"]["flaeche_netto_m2"] == 0.0
    assert m["raeume"]["anzahl"] == 0
    assert m["raeume"]["flaeche_quelle"] is None


def test_fehlerhafte_eingaben():
    with pytest.raises(ValueError):
        mengen_aus_scene({"version": "scene_v1"})
    with pytest.raises(ValueError):
        mengen_aus_scene("keine szene")
    with pytest.raises(ValueError):
        mengen_aus_scene(lade(HAUS_EG), wall_height_m=-1)


def test_komponenten_als_liste_werden_gelesen():
    s = {
        "components": [
            wand("a", 5.0, "exterior"),
            oeffnung("b", "WINDOW", "a", 1.0, 0.9, 2.1),
        ]
    }
    m = mengen_aus_scene(s)
    assert m["waende"]["summe"]["flaeche_netto_m2"] == pytest.approx(14.0 - 1.2)


# --- Additive Felder der Pipeline ----------------------------------------------


@pytest.mark.parametrize(
    ("wert", "erwartet"),
    [
        (True, True),
        ("tragend", True),
        ("load_bearing", True),
        (False, False),
        ("nichttragend", False),
        ("non_bearing", False),
        ("unbekannt", None),
        (None, None),
        (7, None),
    ],
)
def test_bearing_wird_in_attrs_gelesen(wert, erwartet):
    m = mengen_aus_scene(szene(wand("w", 4.0, "interior", bearing=wert)))
    assert m["elemente"]["waende"][0]["tragend"] is erwartet


def test_bearing_in_verschachteltem_meta_wird_gelesen():
    w = wand("w", 4.0, "interior")
    w["attrs"]["meta"] = {"bearing": True, "bearing_source": "layer"}
    m = mengen_aus_scene(szene(w))
    z = m["elemente"]["waende"][0]
    assert z["tragend"] is True and z["tragend_beleg"] == "layer"
    w2 = wand("w2", 4.0, "interior")
    w2["meta"] = {"bearing": False}
    z2 = mengen_aus_scene(szene(w2))["elemente"]["waende"][0]
    assert z2["tragend"] is False


def test_area_net_m2_hat_vorrang():
    r = {"id": "r", "kind": "ROOM", "attrs": {"area_m2": 20.0, "area_net_m2": 17.5}}
    m = mengen_aus_scene(szene(r))
    assert m["raeume"]["flaeche_m2"] == pytest.approx(17.5)
    assert m["raeume"]["flaeche_quelle"] == "area_net_m2"


def test_raum_ohne_flaeche_wird_gemeldet():
    r = {"id": "r", "kind": "ROOM", "attrs": {"label": "x"}}
    m = mengen_aus_scene(szene(r))
    assert m["raeume"]["anzahl"] == 0
    assert any(h["code"] == "RAUM_OHNE_FLAECHE" for h in m["hinweise"])


def test_ergebnis_ist_json_faehig_und_eingabe_unveraendert():
    import copy
    import json

    s = lade(HAUS_EG)
    vorher = copy.deepcopy(s)
    m = berechne_mengen(s, geschoss="EG")
    assert s == vorher
    assert json.loads(json.dumps(m)) == m


def test_bearing_mit_layerangabe_ist_layer_belegt_sonst_nicht():
    """Vertrag der Pipeline: bearing steht nur im scene.json, wenn der Layername es sagt."""
    mit = wand("a", 4.0, "interior", bearing=True, layer="A_Wand_i_tragend")
    ohne = wand("b", 4.0, "interior", bearing=True)
    quelle = wand("c", 4.0, "interior", bearing=True, layer="x", bearing_source="geometry")
    z = {w["id"]: w for w in mengen_aus_scene(szene(mit, ohne, quelle))["elemente"]["waende"]}
    assert z["a"]["tragend_beleg"] == "layer"
    assert z["b"]["tragend_beleg"] == "unbekannt"
    assert z["c"]["tragend_beleg"] == "geometrie"  # ausdrueckliche Quelle schlaegt die Layerangabe
