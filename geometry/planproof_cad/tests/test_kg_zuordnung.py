"""Kostengruppen-Vorschlag: Regeltabelle, Zuordnung, Konfidenz, Eigenstaendigkeit der Datei."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
from mengen_support import HAUS_EG, HAUS_OG, lade, oeffnung, szene, wand

from planproof_cad.kg import regeln_aus_datei
from planproof_cad.kg import zuordnung as zu
from planproof_cad.kg.zuordnung import (
    KG_RULES,
    mengen_aus_scene,
    vorschlaege_aus_mengen,
    vorschlaege_aus_scene,
    vorschlaege_zusammenfassen,
)

ZUORDNUNG_DATEI = Path(zu.__file__)
ERLAUBTE_NUMMERN = {"330", "331", "332", "334", "340", "341", "342", "344"}


@pytest.fixture(scope="module")
def eg() -> dict:
    return vorschlaege_aus_scene(lade(HAUS_EG), geschoss="EG")


def _p(doc: dict) -> dict:
    return {p["element_class"]: p for p in doc["proposals"]}


# --- Regeltabelle ------------------------------------------------------------------


def test_regeltabelle_eingebettet_gleich_datei():
    datei = ZUORDNUNG_DATEI.with_name("kg_rules.json")
    assert json.loads(datei.read_text(encoding="utf-8")) == KG_RULES
    assert regeln_aus_datei() == KG_RULES


def test_regeltabelle_nur_erlaubte_nummern_und_neutrale_bezeichnungen():
    nummern = {r["kg"] for r in KG_RULES["rules"]}
    assert nummern == ERLAUBTE_NUMMERN
    klassen = [r["element_class"] for r in KG_RULES["rules"]]
    assert len(set(klassen)) == len(klassen)
    assert set(klassen) == {
        "Aussenwand",
        "Aussenwand tragend",
        "Aussenwand nichttragend",
        "Innenwand",
        "Innenwand tragend",
        "Innenwand nichttragend",
        "Aussentuer",
        "Aussenfenster",
        "Innentuer",
        "Innenfenster",
    }
    zuordnung = {r["element_class"]: r["kg"] for r in KG_RULES["rules"]}
    assert zuordnung["Aussenwand"] == "330"
    assert zuordnung["Aussenwand tragend"] == "331"
    assert zuordnung["Aussenwand nichttragend"] == "332"
    assert zuordnung["Innenwand"] == "340"
    assert zuordnung["Innenwand tragend"] == "341"
    assert zuordnung["Innenwand nichttragend"] == "342"
    assert zuordnung["Aussentuer"] == zuordnung["Aussenfenster"] == "334"
    assert zuordnung["Innentuer"] == zuordnung["Innenfenster"] == "344"


def test_regeltabelle_enthaelt_keinen_normtext():
    """Nur Nummern und eigene Bauteilbezeichnungen: keine Normtitel, keine Abgrenzungen."""
    text = json.dumps({k: v for k, v in KG_RULES.items() if k != "hinweis"}, ensure_ascii=False)
    for verboten in (
        "Außentüren und -fenster",
        "Innentüren und -fenster",
        "Tragende Außenwände",
        "Nichttragende Außenwände",
        "Abgrenzung",
        "einschließlich",
        "Kostengruppe 3",
        "Bauwerk - Baukonstruktionen",
    ):
        assert verboten not in text, verboten
    for r in KG_RULES["rules"]:
        assert len(r["rationale"]) < 160, r["id"]


def test_raeume_haben_keine_kostengruppe(eg):
    assert all(p["element_class"] != "Raum" for p in eg["proposals"])
    ref = eg["reference_quantities"]
    assert len(ref) == 1
    assert ref[0]["kg"] is None and ref[0]["element_class"] == "Raum"
    assert ref[0]["quantity"] == pytest.approx(20.0 + 14.8 + 12.0)
    assert "DIN 277" in ref[0]["note"]


def test_nicht_abgedeckte_bauteile_sind_aufgelistet_nicht_ausgegeben(eg):
    assert [e["element_class"] for e in eg["not_covered"]] == ["Stuetze", "Treppe", "Decke", "Dach"]
    klassen = {p["element_class"] for p in eg["proposals"]}
    assert not klassen & {"Stuetze", "Treppe", "Decke", "Dach"}


# --- Zuordnung ------------------------------------------------------------------------


def test_nummern_und_mengen_im_haus(eg):
    p = _p(eg)
    soll = {
        "Aussenwand tragend": ("331", 43.5, "m2", "Nettoflaeche"),
        "Aussenwand nichttragend": ("332", 11.2, "m2", "Nettoflaeche"),
        "Aussenwand": ("330", 14.0, "m2", "Nettoflaeche"),
        "Innenwand tragend": ("341", 13.2, "m2", "Nettoflaeche"),
        "Innenwand nichttragend": ("342", 6.51, "m2", "Nettoflaeche"),
        "Innenwand": ("340", 4.9, "m2", "Nettoflaeche"),
        "Aussentuer": ("334", 1, "Stk", "Stueck"),
        "Aussenfenster": ("334", 3, "Stk", "Stueck"),
        "Innentuer": ("344", 2, "Stk", "Stueck"),
        "Innenfenster": ("344", 1, "Stk", "Stueck"),
    }
    assert set(p) == set(soll)
    for klasse, (kg, menge, einheit, basis) in soll.items():
        x = p[klasse]
        assert x["kg"] == kg, klasse
        assert x["quantity"] == pytest.approx(menge, abs=0.01), klasse
        assert x["unit"] == einheit and x["basis"] == basis, klasse
        assert x["status"] == "vorschlag", klasse


def test_jeder_vorschlag_hat_alle_felder(eg):
    for x in eg["proposals"]:
        assert set(x) >= {
            "kg",
            "element_class",
            "basis",
            "quantity",
            "unit",
            "confidence",
            "rationale",
            "assumptions",
            "source_ids",
            "status",
        }
        assert x["confidence"] in {"hoch", "mittel", "niedrig"}
        assert x["rationale"] and x["source_ids"] and x["assumptions"]
        assert isinstance(x["kg"], str) and x["kg"] in ERLAUBTE_NUMMERN
        assert x["geschoss"] == "EG"


def test_lage_unbekannt_bekommt_keine_nummer(eg):
    kanten = {u["component"]: u for u in eg["unassigned"]}
    assert set(kanten) == {"wand", "tuer"}
    assert kanten["wand"]["source_ids"] == ["w_x"]
    assert kanten["wand"]["quantity"] == pytest.approx(0.91, abs=1e-3)
    assert sorted(kanten["tuer"]["source_ids"]) == ["d_w", "d_x", "d_y"]
    assert kanten["tuer"]["quantity"] == 3
    # Keine Waisenoeffnung taucht in einem Vorschlag auf.
    quellen = {i for p in eg["proposals"] for i in p["source_ids"]}
    assert not quellen & {"d_x", "d_y", "d_w", "w_x"}


def test_alle_quell_ids_sind_komponenten_der_szene(eg):
    ids = set(lade(HAUS_EG)["components"])
    for p in eg["proposals"]:
        assert set(p["source_ids"]) <= ids


def test_beispiel_innenwaende_eg_312_m2_netto():
    """Zwoelf Innentueren 1,0 x 2,0 m (24 m2) an zehn Innenwaenden 12 m x 2,8 m (336 m2)."""
    waende = [wand(f"w{i}", 12.0, "interior") for i in range(10)]
    tueren = [oeffnung(f"d{i}", "DOOR", f"w{i % 10}", 1.0, 0.0, 2.0) for i in range(12)]
    doc = vorschlaege_aus_scene(szene(*waende, *tueren), geschoss="EG")
    innen = _p(doc)["Innenwand"]
    assert innen["quantity"] == pytest.approx(312.0)
    assert innen["kg"] == "340"
    assert innen["summary"] == "Innenwände EG 312 m2 netto, KG 340, Vorschlag zur Freigabe"
    assert _p(doc)["Innentuer"]["summary"] == "Innentüren EG 12 Stk, KG 344, Vorschlag zur Freigabe"


def test_ohne_geschoss_keine_geschossangabe_im_text():
    doc = vorschlaege_aus_scene(szene(wand("w", 10.0, "exterior")))
    p = doc["proposals"][0]
    assert p["geschoss"] is None
    assert p["summary"] == "Außenwände 28 m2 netto, KG 330, Vorschlag zur Freigabe"
    assert p["id"] == "kgv_wand_aussen"


# --- Konfidenz --------------------------------------------------------------------------


def test_konfidenz_hoch_nur_mit_layer_evidenz(eg):
    p = _p(eg)
    # Aussen ohne Markierung: "aussen" entsteht in der Pipeline nur aus dem Layernamen.
    assert p["Aussenwand"]["confidence"] == "hoch"
    # Tragend ohne bearing_source bei w_a2: Tragverhalten nicht durch den Layer belegt.
    assert p["Aussenwand tragend"]["confidence"] == "mittel"
    # w_a3 mit bearing_source "layer": beides belegt.
    assert p["Aussenwand nichttragend"]["confidence"] == "hoch"
    # Innen ohne Markierung ist der Standardwert der Pipeline: nicht belegt.
    assert p["Innenwand tragend"]["confidence"] == "mittel"
    # Innen mit wall_type_source "layer".
    assert p["Innenwand"]["confidence"] == "hoch"
    # wall_type_source "default": ausdruecklich ohne Beleg.
    assert p["Innenwand nichttragend"]["confidence"] == "niedrig"


def test_konfidenz_oeffnungen_folgen_wand_und_block(eg):
    p = _p(eg)
    assert p["Aussentuer"]["confidence"] == "hoch"
    # f_a3 hat source "manual": die Oeffnungsart steht nicht in einem Blocknamen.
    assert p["Aussenfenster"]["confidence"] == "mittel"
    # Innenwaende ohne Layer-Beleg: mittel bzw. niedrig (w_i2 default).
    assert p["Innentuer"]["confidence"] == "niedrig"
    assert p["Innenfenster"]["confidence"] == "mittel"


def test_begruendung_nennt_beleg():
    doc = vorschlaege_aus_scene(
        szene(wand("a", 5.0, "exterior"), wand("b", 5.0, "interior", wall_type_source="fallback"))
    )
    p = _p(doc)
    assert "Layernamen" in p["Aussenwand"]["rationale"]
    assert p["Aussenwand"]["confidence"] == "hoch"
    assert "Standardwert ohne Beleg" in p["Innenwand"]["rationale"]
    assert p["Innenwand"]["confidence"] == "niedrig"


def test_annahmen_je_vorschlag(eg):
    p = _p(eg)
    wand_text = " ".join(p["Aussenwand"]["assumptions"])
    assert "Wandhöhe 2,8 m" in wand_text and "Annahme" in wand_text
    assert "2 Öffnung(en) ohne zugeordnete Wandkante" in wand_text
    tuer_text = " ".join(p["Aussentuer"]["assumptions"])
    assert "nicht gemessen" in tuer_text
    assert "Geschoss 'EG'" in wand_text


# --- mehrere Geschosse ----------------------------------------------------------------------


def test_zwei_geschosse_summen():
    eg = vorschlaege_aus_scene(lade(HAUS_EG), geschoss="EG")
    og = vorschlaege_aus_scene(lade(HAUS_OG), geschoss="OG")
    doc = vorschlaege_zusammenfassen([eg, og])
    assert doc["geschosse"] == ["EG", "OG"]
    s = {(x["kg"], x["element_class"]): x for x in doc["summen"]}
    # Aussenwand tragend: EG 43,5 + OG 12 * 3,0 - 1,2 * 1,2
    assert s[("331", "Aussenwand tragend")]["quantity"] == pytest.approx(43.5 + 36.0 - 1.44)
    assert s[("331", "Aussenwand tragend")]["geschosse"] == ["EG", "OG"]
    # Innenwand ohne Angabe: EG 4,9 + OG 18 - 0,9 * 2,1
    assert s[("340", "Innenwand")]["quantity"] == pytest.approx(4.9 + 18.0 - 1.89)
    assert s[("340", "Innenwand")]["confidence"] == "hoch"
    assert s[("334", "Aussenfenster")]["quantity"] == 3 + 1
    assert s[("334", "Aussenfenster")]["unit"] == "Stk"
    # Je Geschoss bleiben die Einzelvorschlaege erhalten.
    assert {p["geschoss"] for p in doc["proposals"]} == {"EG", "OG"}
    assert len(doc["reference_quantities"]) == 2
    assert all(x["status"] == "vorschlag" for x in doc["summen"])


def test_zusammenfassen_ohne_dokumente():
    with pytest.raises(ValueError):
        vorschlaege_zusammenfassen([])


def test_vorschlaege_aus_mengen_prueft_schema():
    with pytest.raises(ValueError):
        vorschlaege_aus_mengen({"waende": {}})


def test_vorschlaege_aus_mengen_gleich_aus_scene():
    m = mengen_aus_scene(lade(HAUS_EG), geschoss="EG")
    assert vorschlaege_aus_mengen(m) == vorschlaege_aus_scene(lade(HAUS_EG), geschoss="EG")


def test_vorschlaege_aendern_die_mengen_nicht():
    m = mengen_aus_scene(lade(HAUS_EG), geschoss="EG")
    vorher = json.dumps(m, sort_keys=True)
    d = vorschlaege_aus_mengen(m)
    d["proposals"][0]["assumptions"].append("x")
    d["not_covered"].clear()
    assert json.dumps(m, sort_keys=True) == vorher


# --- Eigenstaendigkeit der Datei --------------------------------------------------------------


def test_zuordnung_importiert_nur_stdlib_erlaubnisliste():
    baum = ast.parse(ZUORDNUNG_DATEI.read_text(encoding="utf-8"))
    module: set[str] = set()
    for knoten in ast.walk(baum):
        if isinstance(knoten, ast.Import):
            module |= {a.name.split(".")[0] for a in knoten.names}
        elif isinstance(knoten, ast.ImportFrom):
            assert knoten.level == 0, "relativer Import in zuordnung.py"
            module.add((knoten.module or "").split(".")[0])
    assert module <= {"__future__", "json", "typing", "dataclasses"}, module


def test_kopierte_datei_laeuft_ohne_das_paket(tmp_path):
    """Die Datei wird byte-identisch kopiert und in einem Interpreter ohne planproof_cad
    ausgefuehrt: gleiche Vorschlaege wie im Paket, planproof_cad nie importiert."""
    kopie = tmp_path / "kg_zuordnung_kopie.py"
    kopie.write_bytes(ZUORDNUNG_DATEI.read_bytes())
    runner = tmp_path / "run.py"
    runner.write_text(
        "import importlib.util, json, sys\n"
        "spec = importlib.util.spec_from_file_location('kopie', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['kopie'] = m\n"
        "spec.loader.exec_module(m)\n"
        "scene = json.load(open(sys.argv[2], encoding='utf-8'))\n"
        "doc = m.vorschlaege_aus_scene(scene, geschoss='EG')\n"
        "assert 'planproof_cad' not in sys.modules\n"
        "print(json.dumps(doc, sort_keys=True))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-I", str(runner), str(kopie), str(HAUS_EG)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(tmp_path),
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == json.loads(
        json.dumps(vorschlaege_aus_scene(lade(HAUS_EG), geschoss="EG"), sort_keys=True)
    )


def test_tragend_mit_layerangabe_ergibt_hohe_konfidenz():
    s = szene(
        wand("a", 5.0, "exterior", bearing=True, layer="A_Wand_a_tragend"),
        wand(
            "b",
            5.0,
            "interior",
            bearing=False,
            layer="A_Wand_i_nichttragend",
            wall_type_source="layer",
        ),
    )
    p = _p(vorschlaege_aus_scene(s))
    assert (
        p["Aussenwand tragend"]["kg"] == "331" and p["Aussenwand tragend"]["confidence"] == "hoch"
    )
    assert p["Innenwand nichttragend"]["kg"] == "342"
    assert p["Innenwand nichttragend"]["confidence"] == "hoch"
