"""sd_plan: Ground-Truth-Ableitung an handgemachten Geschossen mit bekannter Wahrheit."""
from __future__ import annotations

import math

import pytest
from shapely import affinity
from shapely.geometry import Polygon, box

from goldset import sd_plan as S


def _row(et, es, geom: Polygon):
    return {"entity_type": et, "entity_subtype": es, "geometry": geom.wkt, "unit_usage": "RESIDENTIAL"}


def _room_floor(extra=()):
    """Ein Raum 4,0 x 3,0 m lichte Weite, Wanddicke 0,2 m, Achsen 4,2 x 3,2 m."""
    rows = [
        _row("separator", "WALL", box(-0.1, -0.1, 4.3, 0.1)),    # unten, volle Laenge bis zur Aussenkante
        _row("separator", "WALL", box(-0.1, 3.1, 4.3, 3.3)),     # oben
        _row("separator", "WALL", box(-0.1, 0.1, 0.1, 3.1)),     # links, stoesst an die Flaechen
        _row("separator", "WALL", box(4.1, 0.1, 4.3, 3.1)),      # rechts
        _row("area", "ROOM", box(0.1, 0.1, 4.1, 3.1)),
        _row("opening", "DOOR", box(1.0, -0.1, 1.9, 0.1)),
        _row("opening", "WINDOW", box(2.0, 3.1, 3.4, 3.3)),
    ]
    return rows + list(extra)


def test_einfacher_raum_exakte_wahrheit():
    a = S.analysiere_geschoss(_room_floor(), "1", "1")
    assert a.ok, a.gruende
    gt = a.plan["gt"]
    assert gt["wandlaenge_achse_m"] == pytest.approx(2 * 4.2 + 2 * 3.2, abs=0.01)  # Achse zu Achse
    assert gt["tueren"] == 1 and gt["fenster"] == 1 and gt["raeume"] == 1
    assert gt["raumflaeche_summe_m2"] == pytest.approx(12.0, abs=0.001)
    assert a.plan["raeume"][0]["label"] == "Zimmer"
    assert a.plan["oeffnungen"][0]["breite"] == pytest.approx(0.9, abs=0.01)


def test_wahrheit_ist_drehinvariant():
    base = S.analysiere_geschoss(_room_floor(), "1", "1").plan["gt"]
    rows = []
    for r in _room_floor():
        from shapely import wkt

        g = affinity.rotate(wkt.loads(r["geometry"]), 33.0, origin=(0, 0))
        rows.append({**r, "geometry": g.wkt})
    a = S.analysiere_geschoss(rows, "1", "1")
    assert a.ok, a.gruende
    gt = a.plan["gt"]
    for k in ("wandlaenge_achse_m", "raumflaeche_summe_m2"):
        assert gt[k] == pytest.approx(base[k], abs=0.02)
    assert (gt["tueren"], gt["fenster"], gt["raeume"]) == (1, 1, 1)
    # nach der Drehung auf die Hauptrichtung liegen die Waende wieder achsparallel
    assert a.plan["drehung_grad"] != 0.0


def test_exakte_duplikate_zaehlen_einmal():
    dup = [_row("opening", "DOOR", box(1.0, -0.1, 1.9, 0.1)), _row("separator", "WALL", box(-0.1, -0.1, 4.3, 0.1))]
    a = S.analysiere_geschoss(_room_floor(dup), "1", "1")
    assert a.ok and a.plan["gt"]["tueren"] == 1 and a.plan["gt"]["n_waende"] == 4


def test_haustuer_zaehlt_als_tuer():
    rows = _room_floor()
    rows[5] = _row("opening", "ENTRANCE_DOOR", box(1.0, -0.1, 1.9, 0.1))
    a = S.analysiere_geschoss(rows, "1", "1")
    assert a.plan["gt"]["tueren"] == 1 and a.plan["gt"]["tueren_davon_haustuer"] == 1


def test_oeffnung_ohne_wand_wird_ausgeschlossen():
    a = S.analysiere_geschoss(_room_floor([_row("opening", "WINDOW", box(10, 10, 11, 10.2))]), "1", "1")
    assert "oeffnung_ohne_wand" in a.gruende


def test_offener_raum_wird_ausgeschlossen():
    rows = _room_floor()
    del rows[2]  # linke Wand fehlt
    rows.append(_row("separator", "WALL", box(20, 20, 24, 20.2)))  # weit weg, damit >= 4 Waende bleiben
    a = S.analysiere_geschoss(rows, "1", "1")
    assert "raum_nicht_wandbegrenzt" in a.gruende


def test_zu_wenige_waende():
    assert S.analysiere_geschoss(_room_floor()[:2], "1", "1").gruende == ["zu_wenige_waende"]


def test_offene_nachbarraeume_werden_ein_raum():
    # Wohnen und Kueche ohne Wand dazwischen (1 cm Luecke wie in den Daten)
    rows = _room_floor()
    rows[4] = _row("area", "LIVING_ROOM", box(0.1, 0.1, 2.0, 3.1))
    rows.insert(5, _row("area", "KITCHEN", box(2.01, 0.1, 4.1, 3.1)))
    a = S.analysiere_geschoss(rows, "1", "1")
    assert a.ok, a.gruende
    assert a.plan["gt"]["raeume"] == 1 and a.plan["raeume"][0]["teile"] == 2
    assert a.plan["raeume"][0]["flaeche_m2"] == pytest.approx(1.9 * 3.0 + 2.09 * 3.0, abs=0.001)
    assert a.plan["raeume"][0]["label"] in ("Wohnzimmer", "Küche")


def test_aussenbereich_und_schacht_sind_keine_raeume():
    rows = _room_floor([
        _row("area", "BALCONY", box(5.0, 0.0, 7.0, 2.0)),
        _row("area", "SHAFT", box(0.2, 0.2, 0.6, 0.6)),
    ])
    a = S.analysiere_geschoss(rows, "1", "1")
    assert a.plan["gt"]["raeume"] == 1
    assert a.plan["gt"]["nichtraeume_eingeschlossen"] == 1
    assert len(a.plan["aussenbereiche"]) == 1


def test_umlaut_label():
    rows = _room_floor()
    rows[4] = _row("area", "KITCHEN", box(0.1, 0.1, 4.1, 3.1))
    a = S.analysiere_geschoss(rows, "1", "1")
    assert a.plan["raeume"][0]["label"] == "Küche"


def test_merge_walls_t_stoss_und_kollineare_teile():
    # zwei kollineare Teilstuecke derselben Wand + eine Querwand, die an der Flanke endet
    rects = [S.rect_fit(box(0, -0.1, 3, 0.1)), S.rect_fit(box(3.0, -0.1, 6, 0.1)), S.rect_fit(box(2.9, 0.1, 3.1, 4))]
    walls, diag = S.merge_walls(rects)
    assert len(walls) == 2
    lengths = sorted(w["laenge_achse"] for w in walls)
    assert lengths[1] == pytest.approx(6.0, abs=0.01)
    assert lengths[0] == pytest.approx(4.0, abs=0.01)  # bis zur Achse der Querwand gezogen
    assert diag["mehrdeutige_parallelpaare"] == 0


def test_merge_walls_mehrdeutige_parallele():
    rects = [S.rect_fit(box(0, -0.15, 5, 0.15)), S.rect_fit(box(1, 0.05, 4, 0.25))]  # seitlich teilweise ueberlappend
    _walls, diag = S.merge_walls(rects)
    assert diag["mehrdeutige_parallelpaare"] == 1 and diag["mehrdeutige_laenge_m"] == pytest.approx(3.0, abs=0.01)


def test_rect_fit_orientierung_und_masse():
    r = S.rect_fit(affinity.rotate(box(0, 0, 5, 0.3), 40, origin=(0, 0)))
    assert r.L == pytest.approx(5.0, abs=1e-6) and r.T == pytest.approx(0.3, abs=1e-6)
    assert r.ratio == pytest.approx(1.0, abs=1e-9)
    assert math.degrees(math.atan2(r.u[1], r.u[0])) == pytest.approx(40.0, abs=1e-6)


def test_kleinstes_wandstueck_bricht_nichts():
    """Ein 4 cm langes Wandstueck wird verworfen (Regression: veraltete Gruppennummer)."""
    a = S.analysiere_geschoss(_room_floor([_row("separator", "WALL", box(8.0, 8.0, 8.03, 8.04))]), "1", "1")
    assert a.ok, a.gruende
    assert a.plan["gt"]["n_waende"] == 4
    assert any(r["wand"] == -1 for r in a.plan["rects"])
