"""Einzelheiten der Wand-Rekonstruktion, die in Messungen auf dem Goldset aufgefallen sind.

Jeder Test hat eine kleine synthetische Zeichnung und beschreibt ein Verhalten, das nach der ersten
Fassung falsch war (steht jeweils im Docstring).
"""

from __future__ import annotations

import math

import pytest
from dxfbau import Bau

from planproof_cad.cad.dxf_loader import load_and_extract
from planproof_cad.cad.opening_extraction import opening_boxes
from planproof_cad.cad.wall_reconstruction import reconstruct_walls, select_wall_segments


def _lade(bau: Bau):
    return load_and_extract(bau.speichern(), {})


def test_naechster_partner_gestapelte_linien_ergeben_keine_phantomwand(tmp_path):
    """Drei parallele Linien im Abstand 0,10 m: Paare (1,2) und (2,3), nie (1,3) mit 0,20 m."""
    bau = Bau(tmp_path / "s.dxf")
    for y in (0.0, 0.10, 0.20):
        bau.linie((0, y), (3, y), "A-WALL")
    ex = _lade(bau)
    cls, _ = reconstruct_walls(ex.segments, {})
    dicken = sorted(round(c.thickness_m, 2) for c in cls)
    assert 0.20 not in dicken
    assert dicken == [0.10, 0.10]


def test_naechster_partner_darf_abgeschaltet_werden(tmp_path):
    bau = Bau(tmp_path / "s.dxf")
    for y in (0.0, 0.10, 0.20):
        bau.linie((0, y), (3, y), "A-WALL")
    ex = _lade(bau)
    cls, _ = reconstruct_walls(ex.segments, {"wall_nearest_partner": False})
    assert 0.2 in {round(c.thickness_m, 2) for c in cls}


def test_gegenueberliegende_flur_waende_werden_nicht_zur_wand(tmp_path):
    """Zwei Waende (je 0,12 m) mit 0,9 m Flur dazwischen: der Flur ist keine Wand von 0,9 m
    Dicke."""
    bau = Bau(tmp_path / "f.dxf")
    bau.wand_doppel((0, 0), (6, 0), 0.12, "A-WALL")
    bau.wand_doppel((0, 1.02), (6, 1.02), 0.12, "A-WALL")
    ex = _lade(bau)
    cls, _ = reconstruct_walls(ex.segments, {})
    assert len(cls) == 2
    assert all(c.thickness_m == pytest.approx(0.12, abs=0.005) for c in cls)


def test_sehr_dicke_wand_bis_ein_meter_wird_erkannt(tmp_path):
    bau = Bau(tmp_path / "d.dxf")
    bau.wand_doppel((0, 0), (6, 0), 0.90, "A-WALL")
    ex = _lade(bau)
    cls, _ = reconstruct_walls(ex.segments, {})
    assert len(cls) == 1 and cls[0].thickness_m == pytest.approx(0.90, abs=0.01)
    # mit der alten Obergrenze von 0,6 m ist sie keine Wand
    cls2, _ = reconstruct_walls(ex.segments, {"wall_max_thickness_m": 0.6})
    assert not cls2 or all(c.thickness_m < 0.6 for c in cls2)


def test_leicht_schiefe_wand_stuecke_werden_eine_achse(tmp_path):
    """Zwei Stuecke einer um 0,4 Grad geneigten Wand mit Fensterluecke: eine Achse statt zwei.

    Die erste Fassung verglich die Summe der Betraege der Achsenabstaende statt des
    Normalabstands und trennte lange, leicht geneigte Stuecke in verschiedene Achsen.
    """
    bau = Bau(tmp_path / "n.dxf")
    t = 0.24
    ang = math.radians(0.4)
    c, s = math.cos(ang), math.sin(ang)

    def punkt(x, off):
        return (x * c - off * s, x * s + off * c)

    for off in (t / 2, -t / 2):
        bau.linie(punkt(0, off), punkt(6, off), "A-WALL")
        bau.linie(punkt(7.2, off), punkt(14, off), "A-WALL")
    bau.fenster_block("Fenster_1200", 1.2)
    p = punkt(6.0, 0)
    bau.einfuegen("Fenster_1200", p[0], p[1], "A-GLAZ", rotation_grad=0.4)
    ex = _lade(bau)
    ops = [i for i in ex.inserts]
    cls, stats = reconstruct_walls(ex.segments, {}, opening_boxes=opening_boxes(ops))
    assert len(cls) == 1
    assert cls[0].length() == pytest.approx(14.0, abs=0.1)


def test_luecke_am_tuerblock_schliesst_auch_eine_freistehende_wand(tmp_path):
    """Eine freistehende Wand mit 1,2 m Tuerluecke: beide Stuecke gehoeren zur selben Linienmenge.

    Die Linienmengen entstehen ueber den Abstand; eine Luecke von 1,2 m trennte sie vor der
    Verbindung ueber den Oeffnungsblock, und kein Stueck sah das andere.
    """
    bau = Bau(tmp_path / "fw.dxf")
    t = 0.12
    for off in (t / 2, -t / 2):
        bau.linie((0, off), (4, off), "A-WALL")
        bau.linie((5.2, off), (9.2, off), "A-WALL")
    bau.tuer_block("Tuer_1200", 1.2)
    bau.einfuegen("Tuer_1200", 4.0, 0.0, "A-DOOR")
    ex = _lade(bau)
    cls, stats = reconstruct_walls(ex.segments, {}, opening_boxes=opening_boxes(ex.inserts))
    assert len(cls) == 1 and cls[0].length() == pytest.approx(9.2, abs=0.05)
    assert stats["bridged_gaps"] == 1


def test_luecke_ohne_tuerblock_bleibt_getrennt(tmp_path):
    bau = Bau(tmp_path / "fw.dxf")
    t = 0.12
    for off in (t / 2, -t / 2):
        bau.linie((0, off), (4, off), "A-WALL")
        bau.linie((5.2, off), (9.2, off), "A-WALL")
    ex = _lade(bau)
    cls, _ = reconstruct_walls(ex.segments, {})
    assert len(cls) == 2


def test_wandluecken_koennen_abgeschaltet_werden(tmp_path):
    bau = Bau(tmp_path / "fw.dxf")
    t = 0.12
    for off in (t / 2, -t / 2):
        bau.linie((0, off), (4, off), "A-WALL")
        bau.linie((4.1, off), (9.2, off), "A-WALL")
    ex = _lade(bau)
    ein, _ = reconstruct_walls(ex.segments, {})
    aus, _ = reconstruct_walls(ex.segments, {"wall_bridge_gaps": False})
    assert len(ein) == 1 and len(aus) == 2


def test_verlaengerung_laesst_sich_abschalten(tmp_path):
    bau = Bau(tmp_path / "e.dxf")
    bau.rechteck_doppel(0, 0, 8, 6, 0.5, "A-WALL")
    ex = _lade(bau)
    mit, st1 = reconstruct_walls(ex.segments, {})
    ohne, st2 = reconstruct_walls(ex.segments, {"wall_extend_junctions": False})
    assert st1["extended_ends"] == 8 and st2["extended_ends"] == 0
    assert sum(c.length() for c in mit) == pytest.approx(28.0, abs=0.1)
    assert sum(c.length() for c in ohne) < 28.0 - 1.0  # je Ecke fehlen t = 0,5 m


def test_wandlayer_wahl_aus_geometrie_schliesst_moebel_aus(tmp_path):
    bau = Bau(tmp_path / "v.dxf")
    bau.rechteck_doppel(0, 0, 8, 6, 0.3, "STIFT_18")
    for k in range(4):  # Moebel: kleine geschlossene Rechtecke
        x = 1.0 + 1.5 * k
        bau.polylinie([(x, 1), (x + 0.5, 1), (x + 0.5, 1.9), (x, 1.9)], "STIFT_3")
    ex = _lade(bau)
    segs, info = select_wall_segments(ex.segments, {})
    assert info["source"] == "geometry"
    assert info["layers"] == ["STIFT_18"]
    assert len(segs) == 8


def test_wandlayer_wahl_kann_abgeschaltet_werden(tmp_path):
    bau = Bau(tmp_path / "v.dxf")
    bau.rechteck_doppel(0, 0, 8, 6, 0.3, "STIFT_18")
    ex = _lade(bau)
    _segs, info = select_wall_segments(ex.segments, {"wall_layer_vote": False})
    assert info["source"] == "all_unknown"
