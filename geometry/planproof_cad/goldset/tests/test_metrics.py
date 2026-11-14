"""metrics.py gegen kuenstliche Pipeline-Ergebnisse (ohne Pipeline-Import)."""
from __future__ import annotations

import json

import pytest

from goldset import metrics


def _gt(data_dir, f):
    return json.loads((data_dir / f["gt_json"]).read_text(encoding="utf-8"))


def _scene_from_gt(gt, shift=(0.0, 0.0), area_scale=1.0, drop_doors=0, extra_windows=0, drop_rooms=0, wall_scale=1.0):
    """Baut eine scene.json im Format von dwg_bim/pipeline.py aus der Ground Truth, mit Verschiebung."""
    g = gt["gt"]
    dx, dy = shift
    comps = {}
    for i, w in enumerate(g["waende"]):
        a = [w["p0"][0] + dx, w["p0"][1] + dy]
        b = [w["p1"][0] + dx, w["p1"][1] + dy]
        length = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 * wall_scale
        comps[f"w{i}"] = {"id": f"w{i}", "kind": "WALL", "attrs": {"segment_a": a, "segment_b": b, "length_m": length}}
    doors_seen = 0
    for o in g["oeffnungen"]:
        if o["art"] == "tuer":
            doors_seen += 1
            if doors_seen <= drop_doors:
                continue
        half = o["breite"] / 2
        ux, uy = o["richtung"]
        a = [o["mitte"][0] + dx - ux * half, o["mitte"][1] + dy - uy * half]
        b = [o["mitte"][0] + dx + ux * half, o["mitte"][1] + dy + uy * half]
        comps[o["id"]] = {"id": o["id"], "kind": "DOOR" if o["art"] == "tuer" else "WINDOW",
                          "attrs": {"segment_a": a, "segment_b": b}}
    for k in range(extra_windows):
        comps[f"extra{k}"] = {"id": f"extra{k}", "kind": "WINDOW",
                              "attrs": {"segment_a": [900.0 + dx, 900.0 + dy], "segment_b": [901.0 + dx, 900.0 + dy]}}
    for j, r in enumerate(g["raeume_liste"]):
        if j < drop_rooms:
            continue
        poly = [[x + dx, y + dy] for x, y in r["polygon"]]
        comps[f"r{j}"] = {"id": f"r{j}", "kind": "ROOM",
                          "attrs": {"polygon": poly, "area_m2": r["flaeche_m2"] * area_scale, "label": r["label"]}}
    return {"version": "scene_v1", "units": "m", "components": comps, "relationships": [], "meta": {}}


@pytest.fixture(scope="module")
def gts(all_files, data_dir):
    # drei Plaene verschiedener Konvention und Groesse
    picks = []
    seen = set()
    for _, f in sorted(all_files, key=lambda pf: pf[1]["dxf_bytes"]):
        if f["konvention"] not in seen:
            seen.add(f["konvention"])
            picks.append(_gt(data_dir, f))
    return picks


def test_perfekte_pipeline(gts):
    for gt in gts:
        m = metrics.evaluate_plan(gt, metrics.extract_result(_scene_from_gt(gt)))
        assert m["waende"]["abw_pct"] == pytest.approx(0.0, abs=0.3)  # GT-Koordinaten sind auf 1 cm gerundet
        for k in ("tueren", "fenster"):
            if m[k]["gt"]:
                assert m[k]["recall"] == 1.0 and m[k]["precision"] == 1.0
                assert m[k]["abw_anzahl"] == 0
        r = m["raeume"]
        assert r["recall"] == 1.0 and r["precision"] == 1.0 and r["abw_anzahl"] == 0
        assert r["median_abs_rel_abw"] == pytest.approx(0.0, abs=1e-9)
        assert r["p90_abs_rel_abw"] == pytest.approx(0.0, abs=1e-9)


def test_verschiebung_wird_ausgerichtet(gts):
    """Die Pipeline schiebt den Ursprung: ohne Hinweis findet die Bounding-Box-Ausrichtung ihn."""
    for gt in gts:
        m = metrics.evaluate_plan(gt, metrics.extract_result(_scene_from_gt(gt, shift=(-12.34, -5.67))))
        assert m["ausrichtung"]["modus"] == "wand_bbox"
        assert m["raeume"]["recall"] == 1.0
        if m["tueren"]["gt"]:
            assert m["tueren"]["recall"] == 1.0
    gt = gts[0]
    m = metrics.evaluate_plan(
        gt, metrics.extract_result(_scene_from_gt(gt, shift=(-10.0, -10.0))), origin_hint=(10.0, 10.0)
    )
    assert m["ausrichtung"]["modus"] == "dxf_ursprung" and m["raeume"]["recall"] == 1.0


def test_fehlende_tueren_extra_fenster_fehlende_raeume(gts):
    gt = max(gts, key=lambda g: g["gt"]["tueren"])
    n_doors, n_win, n_rooms = gt["gt"]["tueren"], gt["gt"]["fenster"], gt["gt"]["raeume"]
    assert n_doors >= 2 and n_rooms >= 2
    m = metrics.evaluate_plan(gt, metrics.extract_result(_scene_from_gt(gt, drop_doors=2, extra_windows=3, drop_rooms=1)))
    assert m["tueren"]["treffer"] == n_doors - 2
    assert m["tueren"]["recall"] == pytest.approx((n_doors - 2) / n_doors, abs=1e-4)
    assert m["tueren"]["precision"] == 1.0
    assert m["tueren"]["abw_anzahl"] == -2
    assert m["fenster"]["ergebnis"] == n_win + 3
    assert m["fenster"]["precision"] == pytest.approx(n_win / (n_win + 3), abs=1e-4)
    assert m["fenster"]["recall"] == (1.0 if n_win else None)
    assert m["raeume"]["treffer"] == n_rooms - 1 and m["raeume"]["abw_anzahl"] == -1


def test_flaechen_und_wandabweichung(gts):
    gt = gts[0]
    m = metrics.evaluate_plan(gt, metrics.extract_result(_scene_from_gt(gt, area_scale=1.1, wall_scale=0.9)))
    assert m["raeume"]["median_abs_rel_abw"] == pytest.approx(0.1, abs=1e-3)
    assert m["raeume"]["p90_abs_rel_abw"] == pytest.approx(0.1, abs=1e-3)
    assert m["raeume"]["median_rel_abw_mit_vorzeichen"] == pytest.approx(0.1, abs=1e-3)
    assert m["raeume"]["flaeche_summe_abw_pct_treffer"] == pytest.approx(10.0, abs=0.05)
    assert m["waende"]["abw_pct"] == pytest.approx(-10.0, abs=0.3)


def test_explizite_raumliste_ueberschreibt_scene(gts):
    gt = gts[0]
    scene = _scene_from_gt(gt, drop_rooms=len(gt["gt"]["raeume_liste"]))
    rooms = [{"polygon": r["polygon"], "area_m2": r["flaeche_m2"], "label": r["label"]} for r in gt["gt"]["raeume_liste"]]
    m = metrics.evaluate_plan(gt, metrics.extract_result(scene, rooms=rooms))
    assert m["raeume"]["recall"] == 1.0


def test_leeres_ergebnis(gts):
    gt = gts[0]
    m = metrics.evaluate_plan(gt, metrics.extract_result({"components": {}}))
    assert m["waende"]["ergebnis_m"] == 0 and m["waende"]["abw_pct"] == pytest.approx(-100.0)
    assert m["tueren"]["recall"] == 0.0 and m["tueren"]["precision"] is None
    assert m["raeume"]["recall"] == 0.0 and m["raeume"]["precision"] is None


def test_perzentil_und_statistik():
    assert metrics._percentile([1, 2, 3, 4, 5], 90) == pytest.approx(4.6)
    assert metrics._percentile([5.0], 90) == 5.0
    s = metrics._stats([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["median"] == 3.0 and s["n"] == 5 and s["p90"] == pytest.approx(61.6)


def test_zusammenfassung_je_konvention_und_markdown(gts):
    items = []
    for i, gt in enumerate(gts):
        conv = gt["konvention"]["id"]
        m = metrics.evaluate_plan(gt, metrics.extract_result(_scene_from_gt(gt, shift=(-3.0, -4.0))))
        items.append({"datei": f"x{i}.dxf", "konvention": conv, "variante": "v", "plan_key": f"p{i}", "fehler": None,
                      "metriken": m, "gt_zahlen": {k: gt["gt"][k] for k in ("tueren", "fenster", "raeume")}})
    # ein fehlgeschlagener Lauf zaehlt gegen den Recall
    bad = gts[0]
    items.append({"datei": "bad.dxf", "konvention": bad["konvention"]["id"], "variante": "v", "plan_key": "bad",
                  "fehler": "Boom", "metriken": None,
                  "gt_zahlen": {k: bad["gt"][k] for k in ("tueren", "fenster", "raeume")}})
    s = metrics.summarize(items)
    assert s["gesamt"]["dateien"] == len(items) and s["gesamt"]["fehlgeschlagen"] == 1
    conv0 = bad["konvention"]["id"]
    assert s["je_konvention"][conv0]["fehlgeschlagen"] == 1
    assert s["je_konvention"][conv0]["raeume"]["recall"] < 1.0
    assert s["gesamt"]["wandlaenge_abw_pct_abs"]["median"] == pytest.approx(0.0, abs=0.3)
    md = metrics.summary_markdown(s, title="Test")
    assert md.startswith("## Test") and "| **gesamt** |" in md and conv0 in md
    json.dumps(s)  # serialisierbar


def test_extract_result_liest_core_format():
    """Format wie SceneModel.to_json es schreibt (WALL/DOOR/WINDOW/ROOM, attrs.segment_a/b, attrs.polygon)."""
    scene = {
        "components": {
            "e1": {"kind": "WALL", "attrs": {"segment_a": [0, 0], "segment_b": [3, 4], "length_m": 5.0}},
            "op": {"kind": "DOOR", "attrs": {"segment_a": [0, 0], "segment_b": [1, 0]}},
            "w": {"kind": "WINDOW", "center": [2.0, 1.0, 3.0], "attrs": {}},
            "r": {"kind": "ROOM", "attrs": {"polygon": [[0, 0], [1, 0], [1, 1]], "area_m2": 0.5, "label": "Bad"}},
        }
    }
    r = metrics.extract_result(scene)
    assert r["walls"][0]["length"] == 5.0 and r["doors"] == [(0.5, 0.0)] and r["windows"] == [(2.0, 3.0)]
    assert r["rooms"][0]["label"] == "Bad"


def test_dxf_loader_origin_stimmt_mit_dem_loader(all_files, data_dir):
    """Hinweis-Ursprung = Minimum aller Linien/Einfuegepunkte in Metern (mm-Zeichnung)."""
    _, f = all_files[0]
    o = metrics.dxf_loader_origin(data_dir / f["dxf"])
    gt = _gt(data_dir, f)
    x0, y0, _x1, _y1 = gt["gt"]["bbox_m"]
    # Bemassungslinien liegen ausserhalb der Waende, also ist der Ursprung kleiner als die Wand-Bbox
    assert o is not None and o[0] < x0 and o[1] < y0 and x0 - o[0] < 5 and y0 - o[1] < 5
