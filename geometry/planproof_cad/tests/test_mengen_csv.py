"""mengen_de.csv: Aufbau und Lesbarkeit durch den Parser der Desktop-Anwendung.

``parse_quantity_csv`` unten bildet ``quantityCsv.ts`` (``parseQuantityCsv``) Zeile fuer Zeile
nach. ``test_echter_desktop_parser`` fuehrt zusaetzlich den echten TypeScript-Parser mit Node aus,
wenn Node und das Desktop-Verzeichnis vorhanden sind.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from mengen_support import DEMO_SCENE, HAUS_EG, lade

from planproof_cad.quantities import berechne_mengen, csv_text, schreibe_mengen_csv_de
from planproof_cad.quantities.writers import CSV_HEADER

REPO = Path(__file__).resolve().parents[3]
QUANTITY_CSV_TS = (
    REPO / "clients" / "desktop-tauri" / "src" / "components" / "plan-bim" / "quantityCsv.ts"
)
PROBE_MJS = Path(__file__).with_name("quantity_csv_probe.mjs")

KATEGORIEN = {"wand": "wand", "tuer": "tuer", "tür": "tuer", "fenster": "fenster", "raum": "raum"}


def parse_quantity_csv(text: str) -> list[dict]:
    """Python-Nachbau von parseQuantityCsv (quantityCsv.ts)."""
    clean = text[1:] if text[:1] == "﻿" else text
    zeilen = re.split(r"\r?\n", clean)
    rows = []
    for zeile in zeilen[1:]:  # Kopfzeile ueberspringen
        if not zeile.strip():
            break  # erste Leerzeile beendet die Daten
        if re.match(r"^ZUSAMMENFASSUNG", zeile.strip(), re.IGNORECASE):
            break
        cols = zeile.split(";")
        if len(cols) < 2:
            continue

        def num(i: int, cols: list[str] = cols):
            s = cols[i].strip() if i < len(cols) else ""
            if not s:
                return None
            try:
                return float(s.replace(",", "."))
            except ValueError:
                return None

        rows.append(
            {
                "category": KATEGORIEN.get(cols[0].strip().lower(), "unknown"),
                "id": cols[1].strip(),
                "type": cols[2].strip() if len(cols) > 2 else "",
                "length_m": num(3),
                "height_m": num(4),
                "thickness_m": num(5),
                "area_m2": num(6),
                "volume_m3": num(7),
                "material": cols[8].strip() if len(cols) > 8 else "",
                "property": cols[9].strip() if len(cols) > 9 else "",
            }
        )
    return rows


@pytest.fixture(scope="module")
def eg() -> dict:
    return berechne_mengen(lade(HAUS_EG), geschoss="EG")


def test_datei_aufbau(eg, tmp_path):
    pfad = schreibe_mengen_csv_de(eg, tmp_path / "mengen_de.csv")
    roh = pfad.read_bytes()
    assert roh.startswith(b"\xef\xbb\xbf"), "UTF-8-BOM fehlt"
    text = roh.decode("utf-8-sig")
    assert text.splitlines()[0] == CSV_HEADER
    assert CSV_HEADER == (
        "Kategorie;ID;Typ;Laenge_m;Hoehe_m;Dicke_m;Flaeche_m2;Volumen_m3;Material;Eigenschaft"
    )
    assert "\r" not in text
    # Dezimalpunkt, kein Dezimalkomma in den Zahlenspalten
    for zeile in text.split("\n\n", 1)[0].splitlines()[1:]:
        cols = zeile.split(";")
        assert len(cols) == 10, zeile
        for c in cols[3:8]:
            assert c == "" or re.fullmatch(r"-?\d+\.\d{3}", c), (c, zeile)


def test_kein_semikolon_im_freitext(eg):
    for zeile in csv_text(eg).split("\n\n", 1)[0].splitlines()[1:]:
        assert zeile.count(";") == 9, zeile


def test_parser_liest_jede_datenzeile(eg):
    text = "﻿" + csv_text(eg)
    rows = parse_quantity_csv(text)
    # 8 Wandkanten + 10 Oeffnungen (6 Tueren, 4 Fenster) + 3 Raeume, ohne die Zusammenfassung.
    assert len(rows) == 8 + 10 + 3
    assert all(r["category"] != "unknown" for r in rows)
    kategorien = [r["category"] for r in rows]
    assert kategorien.count("wand") == 8
    assert kategorien.count("tuer") == 6
    assert kategorien.count("fenster") == 4
    assert kategorien.count("raum") == 3


def test_parser_zahlen_und_typ(eg):
    rows = {r["id"]: r for r in parse_quantity_csv(csv_text(eg))}
    a1 = rows["w_a1"]
    assert (a1["category"], a1["type"]) == ("wand", "exterior")
    assert a1["length_m"] == 10.0 and a1["height_m"] == 2.8 and a1["thickness_m"] == 0.365
    # Flaeche_m2 der Wand ist netto: 10 * 2,8 - (1,0 * 2,1 + 1,5 * 1,2) = 24,1
    assert a1["area_m2"] == pytest.approx(24.1)
    assert a1["volume_m3"] == pytest.approx(24.1 * 0.365, abs=1e-3)
    assert "tragend=true" in a1["property"]
    assert rows["w_a4"]["property"].count("tragend=unbekannt") == 1
    assert rows["w_i2"]["type"] == "interior"
    assert rows["w_x"]["type"] == ""  # Lage unbekannt bleibt leer
    t = rows["d_a1"]
    assert t["category"] == "tuer" and t["type"] == "single"
    assert t["length_m"] == 1.0 and t["height_m"] == 2.1 and t["area_m2"] == pytest.approx(2.1)
    assert rows["d_i2"]["type"] == "double"
    f = rows["f_a1"]
    assert f["category"] == "fenster" and f["area_m2"] == pytest.approx(1.8)
    assert "Brüstung 0.900 m" in f["property"]
    r = rows["r2"]
    assert r["category"] == "raum" and r["area_m2"] == pytest.approx(14.8)
    assert r["length_m"] is None and r["volume_m3"] is None
    assert "Bezugsfläche ohne Kostengruppe" in r["property"]
    assert "area_m2" in rows["r1"]["property"]


def test_orphan_oeffnung_ist_in_der_zeile_markiert(eg):
    rows = {r["id"]: r for r in parse_quantity_csv(csv_text(eg))}
    assert "nicht abgezogen" in rows["d_x"]["property"]
    assert "Wand=" not in rows["d_y"]["property"]


def test_summen_der_zeilen_entsprechen_den_mengen(eg):
    rows = parse_quantity_csv(csv_text(eg))
    wand = sum(r["area_m2"] for r in rows if r["category"] == "wand")
    assert wand == pytest.approx(eg["waende"]["summe"]["flaeche_netto_m2"], abs=0.01)
    raum = sum(r["area_m2"] for r in rows if r["category"] == "raum")
    assert raum == pytest.approx(eg["raeume"]["flaeche_m2"], abs=0.01)


def test_nach_der_leerzeile_stehen_zusammenfassung_und_annahmen(eg):
    text = csv_text(eg)
    daten, rest = text.split("\n\n", 1)
    assert rest.startswith("ZUSAMMENFASSUNG;21 Positionen")
    assert "\nANNAHME;Wandhöhe 2,8 m" in rest
    assert "\nHINWEIS;" in rest
    # Der Parser bricht an der Leerzeile ab: nichts aus dem Rest wird eine Zeile.
    assert len(parse_quantity_csv(text)) == len(daten.splitlines()) - 1


def test_demo_szene_csv_lesbar():
    m = berechne_mengen(lade(DEMO_SCENE), geschoss="EG")
    rows = parse_quantity_csv(csv_text(m))
    assert len(rows) == 10 + 6 + 3
    assert {r["category"] for r in rows} == {"wand", "tuer", "fenster", "raum"}


def test_leere_mengen_nur_kopfzeile():
    m = berechne_mengen({"version": "scene_v1", "components": {}})
    text = csv_text(m)
    assert text.splitlines()[0] == CSV_HEADER
    assert parse_quantity_csv(text) == []


@pytest.mark.skipif(
    shutil.which("node") is None or not QUANTITY_CSV_TS.is_file(),
    reason="Node oder clients/desktop-tauri nicht vorhanden",
)
def test_echter_desktop_parser(eg, tmp_path):
    """Der echte parseQuantityCsv aus der Desktop-Anwendung liest die Datei."""
    csv_pfad = schreibe_mengen_csv_de(eg, tmp_path / "mengen_de.csv")
    proc = subprocess.run(
        ["node", "--no-warnings", str(PROBE_MJS), str(QUANTITY_CSV_TS), str(csv_pfad)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    if proc.returncode != 0 and "ERR_UNKNOWN_FILE_EXTENSION" in proc.stderr:
        pytest.skip("Node ohne TypeScript-Typentfernung (ab Node 22.6 bzw. 23.6)")
    assert proc.returncode == 0, proc.stderr
    echt = json.loads(proc.stdout.strip().splitlines()[-1])
    nachbau = parse_quantity_csv(csv_pfad.read_text(encoding="utf-8-sig"))
    assert echt["rows"] == nachbau
    assert len(echt["rows"]) == 21
    summen = {s["category"]: s for s in echt["summary"]}
    assert summen["wand"]["count"] == 8
    assert summen["wand"]["total_area_m2"] == pytest.approx(
        eg["waende"]["summe"]["flaeche_netto_m2"], abs=0.01
    )
    assert summen["tuer"]["count"] == 6
    assert summen["fenster"]["count"] == 4
    assert summen["raum"]["count"] == 3
    assert summen["raum"]["total_area_m2"] == pytest.approx(eg["raeume"]["flaeche_m2"], abs=0.01)
