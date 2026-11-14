"""Tests zu den in der Bestandsaufnahme gefundenen Defekten der DXF-Pipeline.

Jeder Test baut eine kleine synthetische DXF (siehe ``dxfbau``) und lief VOR dem jeweiligen Fix rot.
Die Nummern (D1 ...) entsprechen der Defektliste der Aufgabe.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from dxfbau import Bau, haus

from planproof_cad.cad import run_dwg_to_bim_pipeline


def lauf(dxf: Path, tmp_path: Path, config: dict | None = None):
    out = tmp_path / "out"
    res = run_dwg_to_bim_pipeline(dxf, out, config=config)
    assert res.ok, res.error
    return res


def waende(res):
    return [c for c in res.scene.components.values() if c.kind == "WALL"]


def raeume(res):
    return [c for c in res.scene.components.values() if c.kind == "ROOM"]


def wandlaenge(res) -> float:
    return sum(c.attrs["length_m"] for c in waende(res))


def oeffnungen(res, kind: str):
    return [c for c in res.scene.components.values() if c.kind == kind]


def mitte(comp) -> tuple[float, float]:
    a, b = comp.attrs["segment_a"], comp.attrs["segment_b"]
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


# ------------------------------------------------------------------ D1 dicke Waende
@pytest.mark.parametrize("t", [0.30, 0.365, 0.50])
def test_d1_dicke_waende_geschlossene_doppelkontur_gibt_einen_raum(tmp_path, t):
    dxf = haus(tmp_path / "h.dxf", t=t).speichern()
    res = lauf(dxf, tmp_path)
    assert len(raeume(res)) == 1
    assert wandlaenge(res) == pytest.approx(28.0, abs=0.2)
    assert raeume(res)[0].attrs["area_axis_m2"] == pytest.approx(48.0, abs=0.3)


@pytest.mark.parametrize("t", [0.365, 0.50])
def test_d1_t_stoss_mit_dicker_innenwand_gibt_zwei_raeume(tmp_path, t):
    dxf = haus(tmp_path / "h.dxf", t=t, innen_x=4.0, ti=t).speichern()
    res = lauf(dxf, tmp_path)
    assert len(raeume(res)) == 2
    assert wandlaenge(res) == pytest.approx(28.0 + 6.0, abs=0.3)


def test_d1_achsen_enden_liegen_im_schnittpunkt(tmp_path):
    from planproof_cad.cad.dxf_loader import load_and_extract
    from planproof_cad.cad.wall_reconstruction import reconstruct_walls

    dxf = haus(tmp_path / "h.dxf", t=0.5).speichern()
    ex = load_and_extract(dxf, {})
    cls, stats = reconstruct_walls(ex.segments, {})
    assert stats["centerlines"] == 4
    ecken = {(round(c.a[0], 2), round(c.a[1], 2)) for c in cls} | {
        (round(c.b[0], 2), round(c.b[1], 2)) for c in cls
    }
    # alle vier Achsen enden in denselben vier Eckpunkten (Achsenschnittpunkte)
    assert len(ecken) == 4


# ------------------------------------------------------------------ D2 Konturen
def test_d2_rechteck_polylinie_gibt_achse_und_dicke(tmp_path):
    from planproof_cad.cad.dxf_loader import load_and_extract
    from planproof_cad.cad.wall_reconstruction import reconstruct_walls

    bau = Bau(tmp_path / "k.dxf")
    bau.wand_rechteck((0, 0), (5, 0), 0.30, "A-WALL")
    ex = load_and_extract(bau.speichern(), {})
    cls, stats = reconstruct_walls(ex.segments, {})
    assert len(cls) == 1
    c = cls[0]
    assert c.thickness_m == pytest.approx(0.30, abs=0.01)
    assert c.length() == pytest.approx(5.0, abs=0.05)
    # waagerechte Achse mittig im Rechteck (Ursprung liegt an der Zeichnungsecke y = -0.15)
    assert abs(c.a[1] - c.b[1]) < 0.01
    assert c.a[1] == pytest.approx(0.15, abs=0.01)


def test_d2_vier_wandrechtecke_schliessen_einen_raum(tmp_path):
    bau = Bau(tmp_path / "k.dxf")
    t = 0.30
    h = t / 2
    # vier Wandrechtecke, die an den Ecken ueberlappen (Konturen je Wand)
    bau.wand_rechteck((-h, 0), (8 + h, 0), t, "A-WALL")
    bau.wand_rechteck((-h, 6), (8 + h, 6), t, "A-WALL")
    bau.wand_rechteck((0, h), (0, 6 - h), t, "A-WALL")
    bau.wand_rechteck((8, h), (8, 6 - h), t, "A-WALL")
    res = lauf(bau.speichern(), tmp_path)
    assert len(raeume(res)) == 1
    assert wandlaenge(res) == pytest.approx(28.0, abs=0.4)


# ------------------------------------------------------------------ D3 Luecken, Paarung
def _haus_mit_luecke(tmp_path, luecke: float, *, mit_tuer: bool, y_mitte: float = 3.0):
    t, ti = 0.30, 0.115
    bau = Bau(tmp_path / "l.dxf")
    bau.rechteck_doppel(0, 0, 8, 6, t, "A-WALL")
    y0, y1 = y_mitte - luecke / 2, y_mitte + luecke / 2
    for dx in (-ti / 2, ti / 2):
        bau.linie((4 + dx, t / 2), (4 + dx, y0), "A-WALL")
        bau.linie((4 + dx, y1), (4 + dx, 6 - t / 2), "A-WALL")
    if mit_tuer:
        bau.tuer_block("Tuer_1000", luecke)
        # Drehpunkt am unteren Ende der Oeffnung, Blatt entlang der Wand (Drehung 90)
        bau.einfuegen("Tuer_1000", 4.0, y0, "A-DOOR", rotation_grad=90)
    return bau.speichern()


def test_d3_luecke_mit_tuerblock_wird_geschlossen(tmp_path):
    res = lauf(_haus_mit_luecke(tmp_path, 1.0, mit_tuer=True), tmp_path)
    assert len(raeume(res)) == 2
    assert wandlaenge(res) == pytest.approx(28.0 + 6.0, abs=0.3)


def test_d3_kleine_kollinearluecke_ohne_block_wird_geschlossen(tmp_path):
    res = lauf(_haus_mit_luecke(tmp_path, 0.15, mit_tuer=False), tmp_path)
    assert len(raeume(res)) == 2


def test_d3_grosse_luecke_ohne_oeffnung_bleibt_offen(tmp_path):
    """Konservativ: ohne Oeffnungsblock wird nur bis zur wanddickenabhaengigen Toleranz
    gebrueckt."""
    res = lauf(_haus_mit_luecke(tmp_path, 1.5, mit_tuer=False), tmp_path)
    assert len(raeume(res)) == 1


def test_d3_mehrere_kurze_gegenlinien_an_einer_langen_linie(tmp_path):
    from planproof_cad.cad.dxf_loader import load_and_extract
    from planproof_cad.cad.wall_reconstruction import reconstruct_walls

    t = 0.24
    bau = Bau(tmp_path / "g.dxf")
    bau.linie((0, t / 2), (6, t / 2), "A-WALL")  # eine lange Seite
    bau.linie((0, -t / 2), (2.0, -t / 2), "A-WALL")  # drei kurze Gegenlinien
    bau.linie((2.5, -t / 2), (3.6, -t / 2), "A-WALL")
    bau.linie((4.0, -t / 2), (6, -t / 2), "A-WALL")
    ex = load_and_extract(bau.speichern(), {})
    cls, stats = reconstruct_walls(ex.segments, {})
    # alle drei Stuecke sind gepaart und zu einer Achse verbunden (die lange Linie traegt die
    # Luecken)
    assert len(cls) == 1
    assert cls[0].length() == pytest.approx(6.0, abs=0.05)
    # nur die beiden Luecken der Gegenseite (0,5 m und 0,4 m) sind ungepaart
    assert stats["matched_ratio"] > 0.9


# ------------------------------------------------------------------ D4 Schraffur
@pytest.mark.parametrize("hatch_layer", ["A-WALL-PATT", "A_Schraffur", "300_WAND_SCHRAFFUR"])
def test_d4_schraffurlinien_sind_keine_waende(tmp_path, hatch_layer):
    ohne = lauf(haus(tmp_path / "a.dxf").speichern(), tmp_path / "a")
    bau = haus(tmp_path / "b.dxf")
    for k in range(33):  # 45-Grad-Schraffur in der unteren Wand
        x = -0.15 + 0.25 * k
        bau.linie((x, -0.15), (x + 0.3, 0.15), hatch_layer)
    bau.msp.add_hatch(color=8, dxfattribs={"layer": hatch_layer}).paths.add_polyline_path(
        [(0, 0), (1000, 0), (1000, 300), (0, 300)], is_closed=True
    )
    mit = lauf(bau.speichern(), tmp_path / "b")
    assert len(waende(mit)) == len(waende(ohne)) == 4
    assert wandlaenge(mit) == pytest.approx(wandlaenge(ohne), abs=0.01)
    assert len(raeume(mit)) == 1


# ------------------------------------------------------------------ D5 Waende in Bloecken
def _haus_im_block(tmp_path, *, drehung: float, verschachtelt: bool):
    bau = Bau(tmp_path / "bl.dxf")
    h = 0.15  # halbe Wanddicke (Dicke 0,30 m)
    blk = bau.doc.blocks.new("HAUS")

    # Block-Inhalt auf Layer 0: erbt den Layer des INSERT
    def lin(a, b):
        blk.add_line(
            (a[0] * 1000, a[1] * 1000), (b[0] * 1000, b[1] * 1000), dxfattribs={"layer": "0"}
        )

    for x0, y0, x1, y1 in ((-h, -h, 8 + h, 6 + h), (h, h, 8 - h, 6 - h)):
        for a, b in (
            ((x0, y0), (x1, y0)),
            ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)),
            ((x0, y1), (x0, y0)),
        ):
            lin(a, b)
    if verschachtelt:
        aussen = bau.doc.blocks.new("AUSSEN")
        aussen.add_blockref("HAUS", (0, 0), dxfattribs={"layer": "0"})
        name = "AUSSEN"
    else:
        name = "HAUS"
    bau.msp.add_blockref(name, (5000, 5000), dxfattribs={"layer": "A-WALL", "rotation": drehung})
    return bau.speichern()


@pytest.mark.parametrize("verschachtelt", [False, True])
@pytest.mark.parametrize("drehung", [0.0, 90.0, 30.0])
def test_d5_waende_in_bloecken_werden_gelesen(tmp_path, drehung, verschachtelt):
    res = lauf(_haus_im_block(tmp_path, drehung=drehung, verschachtelt=verschachtelt), tmp_path)
    assert len(waende(res)) == 4
    assert wandlaenge(res) == pytest.approx(28.0, abs=0.2)
    assert len(raeume(res)) == 1


def test_d5_blocktiefe_ist_begrenzt_und_meldet_es(tmp_path):
    res = lauf(
        _haus_im_block(tmp_path, drehung=0.0, verschachtelt=True), tmp_path, {"max_block_depth": 1}
    )
    assert any("Blocktiefe" in w for w in res.warnings)


def test_d5_tuer_und_fensterbloecke_bleiben_oeffnungen(tmp_path):
    bau = haus(tmp_path / "o.dxf")
    bau.tuer_block("Tuer_0900", 0.9)
    bau.fenster_block("Fenster_1200", 1.2)
    bau.einfuegen("Tuer_0900", 2.0, 0.0, "A-DOOR")
    bau.einfuegen("Fenster_1200", 3.0, 6.0, "A-GLAZ")
    res = lauf(bau.speichern(), tmp_path)
    assert len(oeffnungen(res, "DOOR")) == 1
    assert len(oeffnungen(res, "WINDOW")) == 1
    assert len(waende(res)) == 4  # die Linien der Bloecke sind keine Waende
    assert wandlaenge(res) == pytest.approx(28.0, abs=0.2)


# ------------------------------------------------------------------ D6 Einheit
def _haus_mm_als_meter(tmp_path):
    bau = Bau(tmp_path / "u.dxf", insunits=6)  # deklariert Meter, gezeichnet in mm
    bau.rechteck_doppel(0, 0, 8, 6, 0.3, "A-WALL")
    return bau.speichern()


def test_d6_falsch_deklarierte_einheit_warnt_in_result_und_meta(tmp_path):
    res = lauf(_haus_mm_als_meter(tmp_path), tmp_path)
    einheit = [w for w in res.warnings if "Einheit" in w]
    assert einheit, res.warnings
    assert "mm" in einheit[0]  # nennt die vermutete Einheit
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    assert any("Einheit" in w for w in meta["warnings"])
    assert meta["metrics"]["unit_check"]["plausible"] is False


def test_d6_dxf_unit_hat_vorrang_und_beseitigt_die_warnung(tmp_path):
    res = lauf(_haus_mm_als_meter(tmp_path), tmp_path, {"dxf_unit": "mm"})
    assert not [w for w in res.warnings if "unplausibel" in w]
    assert wandlaenge(res) == pytest.approx(28.0, abs=0.2)
    assert res.metrics["unit_check"]["plausible"] is True


def test_d6_korrekt_deklarierte_einheit_ohne_warnung(tmp_path):
    res = lauf(haus(tmp_path / "ok.dxf").speichern(), tmp_path)
    assert not [w for w in res.warnings if "Einheit" in w]
    assert res.metrics["unit_check"]["plausible"] is True


# ------------------------------------------------------------------ D7 Oeffnungsposition
def test_d7_offnungsmitte_ist_bbox_mitte_nicht_einfuegepunkt(tmp_path):
    bau = haus(tmp_path / "p.dxf")
    bau.tuer_block("Tuer_0900", 0.9)
    bau.einfuegen("Tuer_0900", 2.0, 0.0, "A-DOOR")  # Drehpunkt (Blockursprung) bei x = 2.0
    res = lauf(bau.speichern(), tmp_path)
    (tuer,) = oeffnungen(res, "DOOR")
    # Block laeuft von x = 2.0 bis 2.9; die Mitte liegt bei 2.45. Die Pipeline verschiebt den
    # Ursprung
    # auf die Zeichnungsecke (-0.15, -0.15), also steht die Mitte bei 2.60 und auf der Wandachse y =
    # 0.15.
    assert tuer.attrs["width_m"] == pytest.approx(0.9, abs=0.02)
    mx, my = mitte(tuer)
    assert mx == pytest.approx(2.60, abs=0.03)
    assert my == pytest.approx(0.15, abs=0.03)


@pytest.mark.parametrize("winkel", [30.0, 90.0, 200.0])
def test_d7_gedrehte_bloecke_behalten_ihre_breite(tmp_path, winkel):
    bau = haus(tmp_path / "p.dxf", drehung_grad=winkel)
    bau.fenster_block("Fenster_1200", 1.2)
    bau.einfuegen("Fenster_1200", 3.0, 6.0, "A-GLAZ")
    res = lauf(bau.speichern(), tmp_path)
    (fenster,) = oeffnungen(res, "WINDOW")
    assert fenster.attrs["width_m"] == pytest.approx(1.2, abs=0.03)  # nicht die aufgeblaehte AABB


def test_d7_hoehen_sind_als_annahme_ausgewiesen(tmp_path):
    bau = haus(tmp_path / "p.dxf")
    bau.tuer_block("Tuer_0900", 0.9)
    bau.einfuegen("Tuer_0900", 2.0, 0.0, "A-DOOR")
    res = lauf(bau.speichern(), tmp_path)
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    ann = meta["assumptions"]
    for key in ("door_head_m", "window_sill_m", "window_head_m", "wall_height_m"):
        assert ann[key]["source"] == "config_default"
    assert ann["door_head_m"]["value"] == 2.10
    assert res.scene.meta["assumptions"]["door_head_m"]["value"] == 2.10


# ------------------------------------------------------------------ D8 lichte Flaeche
def test_d8_raum_hat_achs_und_lichte_flaeche(tmp_path):
    res = lauf(haus(tmp_path / "f.dxf", t=0.30).speichern(), tmp_path)
    (raum,) = raeume(res)
    a = raum.attrs
    assert a["area_axis_m2"] == pytest.approx(48.0, abs=0.2)
    assert a["area_net_m2"] == pytest.approx((8 - 0.30) * (6 - 0.30), abs=0.1)
    assert a["area_m2"] == a["area_net_m2"]
    assert a["area_net_m2"] < a["area_axis_m2"]


def test_d8_lichte_flaeche_nutzt_die_dicke_der_angrenzenden_waende(tmp_path):
    # Aussenwand 0.50 links, 0.20 rechts, 0.30 oben/unten: lichte Breite = 8 - 0.25 - 0.10
    t0 = 0.30
    bau = Bau(tmp_path / "d.dxf")
    for a, b, t in (
        ((0, 0), (8, 0), t0),
        ((0, 6), (8, 6), t0),
        ((0, 0), (0, 6), 0.50),
        ((8, 0), (8, 6), 0.20),
    ):
        bau.wand_doppel(a, b, t, "A-WALL")
    res = lauf(bau.speichern(), tmp_path)
    (raum,) = raeume(res)
    assert raum.attrs["area_net_m2"] == pytest.approx((8 - 0.25 - 0.10) * (6 - 0.30), abs=0.15)


# ------------------------------------------------------------------ D9 Raumlabel
def test_d9_massangaben_und_tuernummern_sind_keine_raumnamen(tmp_path):
    bau = haus(tmp_path / "r.dxf")
    bau.text("3600", 2.0, 3.0, "A-ANNO-DIMS")  # Bemassung, steht als erstes im Raum
    bau.text("T01", 3.0, 3.0, "Tx_Tuer")
    bau.text("12", 3.5, 3.0, "A-TEXT")  # rein numerisch
    bau.text("Wohnzimmer", 5.0, 3.0, "A_Raum")
    res = lauf(bau.speichern(), tmp_path)
    assert raeume(res)[0].attrs["label"] == "Wohnzimmer"


def test_d9_raumlayer_hat_vorrang_vor_anderen_texten(tmp_path):
    bau = haus(tmp_path / "r.dxf")
    bau.text("Zeichnung Nord", 2.0, 3.0, "A-TEXT")
    bau.text("Küche", 5.0, 3.0, "A-AREA")
    res = lauf(bau.speichern(), tmp_path)
    assert raeume(res)[0].attrs["label"] == "Küche"


def test_d9_ohne_raumtext_bleibt_ein_sonstiger_text_moeglich(tmp_path):
    bau = haus(tmp_path / "r.dxf")
    bau.text("Atelier", 2.0, 3.0, "A-TEXT")
    res = lauf(bau.speichern(), tmp_path)
    assert raeume(res)[0].attrs["label"] == "Atelier"


# ------------------------------------------------------------------ D10 gemischte Modi
def _doppel_plus_einlinig(tmp_path, *, rauschen: bool):
    bau = haus(tmp_path / "m.dxf")
    # ein Anbau weit weg, in einlinigen Achsen gezeichnet
    for a, b in (((20, 0), (26, 0)), ((26, 0), (26, 4)), ((26, 4), (20, 4)), ((20, 4), (20, 0))):
        bau.linie(a, b, "A-WALL")
    if rauschen:
        bau.linie((2.0, 3.0), (3.5, 3.0), "A-WALL")  # einzelne Linie im Haus, ohne Gegenlinie
    return bau.speichern()


def test_d10_gemischte_zeichnung_weist_modi_je_zusammenhang_aus(tmp_path):
    res = lauf(_doppel_plus_einlinig(tmp_path, rauschen=False), tmp_path)
    stats = res.metrics["wall_reconstruction"]
    assert stats["mode"] == "mixed"
    modi = sorted(c["mode"] for c in stats["components"])
    assert modi == ["double_line", "single_line"]
    assert len(waende(res)) == 4 + 4
    assert wandlaenge(res) == pytest.approx(28.0 + 20.0, abs=0.4)


def test_d10_einzelne_linie_im_doppellinienplan_wird_nicht_zur_wand(tmp_path):
    res = lauf(_doppel_plus_einlinig(tmp_path, rauschen=True), tmp_path)
    assert wandlaenge(res) == pytest.approx(28.0 + 20.0, abs=0.4)
    assert res.metrics["wall_reconstruction"]["unpaired_segments"] >= 1


# ------------------------------------------------------------------ D11 laute Fehler
def test_d11_nur_annotationslayer_ist_eine_warnung_mit_ursache(tmp_path):
    bau = Bau(tmp_path / "a.dxf")
    bau.rechteck_doppel(0, 0, 8, 6, 0.3, "A-ANNO-DIMS")
    res = lauf(bau.speichern(), tmp_path)
    assert not waende(res) and not raeume(res)
    ursache = [w for w in res.warnings if "Keine Wände" in w]
    assert ursache and "Wandlayer" in ursache[0]


def test_d11_zu_kurze_linien_werden_als_ursache_genannt(tmp_path):
    bau = Bau(tmp_path / "k.dxf")
    for k in range(5):
        bau.linie((k, 0), (k + 0.05, 0), "A-WALL")
    bau.linie((0, 5), (3, 5), "A-ANNO-DIMS")  # damit die Zeichnung eine Ausdehnung hat
    res = lauf(bau.speichern(), tmp_path)
    ursache = [w for w in res.warnings if "Keine Wände" in w]
    assert ursache and "kürzer" in ursache[0]


def test_d11_unplausible_einheit_ist_ursache_bei_leerem_ergebnis(tmp_path):
    bau = Bau(tmp_path / "e.dxf", insunits=4)
    bau.linie((0, 0), (0.004, 0), "A-WALL")  # 4 mm: Diagonale weit unter 3 m
    bau.linie((0, 0), (0, 0.003), "A-WALL")
    res = lauf(bau.speichern(), tmp_path)
    assert any("Einheit" in w for w in res.warnings)


def test_d11_waende_ohne_geschlossenen_raum_warnen(tmp_path):
    bau = Bau(tmp_path / "o.dxf")
    bau.wand_doppel((0, 0), (8, 0), 0.3, "A-WALL")
    bau.wand_doppel((0, 0), (0, 6), 0.3, "A-WALL")
    res = lauf(bau.speichern(), tmp_path)
    assert waende(res) and not raeume(res)
    assert any("kein geschlossener Raum" in w for w in res.warnings)


# ------------------------------------------------------------------ D12 bearing bis in die Szene
@pytest.mark.parametrize(
    "layer,soll",
    [("300_WAND_TRAGEND", True), ("300_WAND_NICHTTRAGEND", False), ("300_WAND", None)],
)
def test_d12_bearing_wird_in_die_wandkomponente_gereicht(tmp_path, layer, soll):
    res = lauf(haus(tmp_path / "b.dxf", layer=layer).speichern(), tmp_path)
    ws = waende(res)
    assert ws and all(w.attrs.get("bearing") is soll for w in ws)
    szene = json.loads((tmp_path / "out" / "scene.json").read_text(encoding="utf-8"))
    walls = [c for c in szene["components"].values() if c["kind"] == "WALL"]
    assert all(c["attrs"]["bearing"] is soll for c in walls)


def test_d12_mischung_tragend_und_nichttragend_je_wand(tmp_path):
    bau = Bau(tmp_path / "x.dxf")
    bau.rechteck_doppel(0, 0, 8, 6, 0.30, "300_WAND_TRAGEND")
    for dx in (-0.0575, 0.0575):
        bau.linie((4 + dx, 0.15), (4 + dx, 5.85), "300_WAND_NICHTTRAGEND")
    res = lauf(bau.speichern(), tmp_path)
    tragend = [w for w in waende(res) if w.attrs["bearing"] is True]
    nicht = [w for w in waende(res) if w.attrs["bearing"] is False]
    assert len(nicht) >= 1 and len(tragend) >= 4
    assert sum(w.attrs["length_m"] for w in nicht) == pytest.approx(6.0, abs=0.3)


# ------------------------------------------------------------------ Geometrie-Layer ohne Namen
def test_pen_layer_fallback_bestimmt_wandlayer_aus_der_geometrie(tmp_path):
    """ArchiCAD-Stiftlayer sind 'unknown'; die Wandlayer ergeben sich aus der Doppellinien-
    Geometrie."""
    bau = Bau(tmp_path / "pen.dxf")
    bau.rechteck_doppel(0, 0, 8, 6, 0.30, "STIFT_18")
    # Moebel und Bemassung auf anderen Stiftlayern duerfen keine Waende werden
    bau.polylinie([(1, 1), (1.6, 1), (1.6, 2.0), (1, 2.0)], "STIFT_3")
    bau.linie((0, -1), (8, -1), "STIFT_1")
    bau.linie((0, -1.1), (0, -0.9), "STIFT_1")
    res = lauf(bau.speichern(), tmp_path)
    assert wandlaenge(res) == pytest.approx(28.0, abs=0.3)
    assert len(raeume(res)) == 1
    assert any("STIFT_18" in w and "Geometrie" in w for w in res.warnings)


def test_wand_layer_konfiguration_hat_vorrang(tmp_path):
    bau = Bau(tmp_path / "cfg.dxf")
    bau.rechteck_doppel(0, 0, 8, 6, 0.30, "XYZ")
    res = lauf(bau.speichern(), tmp_path, {"wall_layers": ["XYZ"]})
    assert wandlaenge(res) == pytest.approx(28.0, abs=0.3)
    assert not any("aus der Geometrie" in w for w in res.warnings)


def test_gesamtsumme_aendert_sich_nicht_durch_drehung(tmp_path):
    a = lauf(haus(tmp_path / "a.dxf", innen_x=4.0).speichern(), tmp_path / "a")
    b = lauf(haus(tmp_path / "b.dxf", innen_x=4.0, drehung_grad=37.0).speichern(), tmp_path / "b")
    assert wandlaenge(a) == pytest.approx(wandlaenge(b), abs=0.05)
    assert len(raeume(a)) == len(raeume(b)) == 2
    assert math.isclose(
        sum(r.attrs["area_net_m2"] for r in raeume(a)),
        sum(r.attrs["area_net_m2"] for r in raeume(b)),
        abs_tol=0.1,
    )
