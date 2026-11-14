"""Ground-Truth-Konsistenz: die JSON-Wahrheit passt zum DXF, das daneben liegt.

Zaehlt mit ezdxf INSERT-Bloecke (Tueren, Fenster), Raum-Labels und prueft Summen und
Plausibilitaeten. Die Tests laufen ohne Rohdaten und ohne Pipeline.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

import ezdxf
import pytest
from shapely.geometry import LineString, Point, Polygon

SIZE_LIMIT_BYTES = 6_000_000


def _gt(data_dir, f):
    return json.loads((data_dir / f["gt_json"]).read_text(encoding="utf-8"))


def _prefix(pattern: str) -> str:
    return pattern.split("{", 1)[0]


def test_manifest_summen_und_limits(manifest, data_dir, all_files):
    assert 40 <= manifest["plaene"] <= 60
    assert manifest["dateien_dxf"] == len(all_files)
    total = sum(f["dxf_bytes"] + f["json_bytes"] for _, f in all_files)
    assert total == manifest["gesamtgroesse_bytes"]
    on_disk = sum(p.stat().st_size for p in data_dir.rglob("*") if p.is_file())
    assert on_disk < SIZE_LIMIT_BYTES, f"Datenordner {on_disk} Bytes"
    # jede Konvention kommt vor, jeder Plan hat mindestens eine Datei
    assert {f["konvention"] for _, f in all_files} == {"aia", "dach", "sib", "archicad"}
    assert all(p["dateien"] for p in manifest["plan_liste"])
    # Synthetik-Hinweis steht im Manifest
    assert "synthetisch" in manifest["hinweis_synthetisch"].lower()
    assert "Buero-Exporten" in manifest["hinweis_synthetisch"]


def test_attribution_vorhanden(data_dir):
    text = (data_dir / "ATTRIBUTION.md").read_text(encoding="utf-8")
    for must in ("Archilyse AG", "10.5281/zenodo.7788422", "CC BY 4.0", "Aenderungen", "synthetisch"):
        assert must in text


def test_dateien_integritaet(all_files, data_dir):
    for _, f in all_files:
        data = (data_dir / f["dxf"]).read_bytes()
        assert len(data) == f["dxf_bytes"]
        assert hashlib.sha256(data).hexdigest() == f["dxf_sha256"], f["dxf"]
        assert (data_dir / f["gt_json"]).stat().st_size == f["json_bytes"]
        assert _gt(data_dir, f)["dxf_sha256"] == f["dxf_sha256"]
        assert b"\r" not in data  # Zeilenende LF, plattformunabhaengiger Hash


def test_gt_json_enthaelt_synthetik_hinweis(all_files, data_dir):
    for _, f in all_files:
        gt = _gt(data_dir, f)
        assert gt["konvention"]["synthetisch"] is True
        assert "nicht auf echten Buero-Exporten" in gt["konvention"]["hinweis"]
        assert gt["einheiten"]["dxf"] == "mm" and gt["einheiten"]["insunits"] == 4


@pytest.fixture(scope="module")
def docs(all_files, data_dir):
    """DXF einmal laden: {dxf_pfad: (doc, gt)}."""
    out = {}
    for _, f in all_files:
        out[f["dxf"]] = (ezdxf.readfile(str(data_dir / f["dxf"])), _gt(data_dir, f))
    return out


def test_dxf_einheit_und_audit(docs):
    for name, (doc, _gt_) in docs.items():  # noqa: B007
        assert doc.header["$INSUNITS"] == 4, name
        assert doc.audit().errors == [], name


def test_zaehlung_tueren_fenster_gegen_dxf(docs):
    """INSERT-Bloecke im DXF = Anzahl in der Ground Truth, je Konvention nach Blockname."""
    for name, (doc, gt) in docs.items():
        kv = gt["konvention"]
        door_p, win_p = _prefix(kv["blockname_tuer"]), _prefix(kv["blockname_fenster"])
        names = Counter()
        for e in doc.modelspace().query("INSERT"):
            n = e.dxf.name
            names["tuer" if n.startswith(door_p) else "fenster" if n.startswith(win_p) else "sonst"] += 1
        assert names["sonst"] == 0, name
        assert names["tuer"] == gt["gt"]["tueren"], name
        assert names["fenster"] == gt["gt"]["fenster"], name


def test_inserts_liegen_an_den_gt_oeffnungen(docs):
    """Jeder INSERT liegt nahe an einer GT-Oeffnung derselben Art (Zeichnung und Wahrheit passen zusammen)."""
    for name, (doc, gt) in docs.items():
        kv = gt["konvention"]
        door_p = _prefix(kv["blockname_tuer"])
        g_open = gt["gt"]["oeffnungen"]
        for e in doc.modelspace().query("INSERT"):
            art = "tuer" if e.dxf.name.startswith(door_p) else "fenster"
            x, y = e.dxf.insert.x / 1000.0, e.dxf.insert.y / 1000.0
            best = min(
                (math.hypot(x - o["mitte"][0], y - o["mitte"][1]) - o["breite"] / 2 for o in g_open if o["art"] == art),
                default=None,
            )
            # Einfuegepunkt = Oeffnungsende; Abstand zur Mitte hoechstens halbe (gerundete) Breite
            assert best is not None and best <= 0.04, (name, e.dxf.name, best)


def test_raumlabels_gegen_dxf(docs):
    for name, (doc, gt) in docs.items():
        layer = gt["konvention"]["layer"]["raum"]
        labels = [e for e in doc.modelspace().query("TEXT") if e.dxf.layer == layer]
        assert len(labels) == gt["gt"]["raeume"], name
        # jedes Label liegt in einem GT-Raum, und jeder GT-Raum hat genau eines
        polys = {}
        for r in gt["gt"]["raeume_liste"]:
            polys[r["id"]] = Polygon(r["polygon"], r.get("loecher", [])).buffer(0.03)
        hits = Counter()
        for t in labels:
            p = Point(t.dxf.insert.x / 1000.0, t.dxf.insert.y / 1000.0)
            inside = [rid for rid, poly in polys.items() if poly.contains(p)]
            assert len(inside) >= 1, (name, t.dxf.text)
            hits[inside[0]] += 1
        assert all(v == 1 for v in hits.values()) and len(hits) == gt["gt"]["raeume"], name
        # SIB: zusaetzlich ein Raumpolygon je Raum
        if gt["konvention"]["id"] == "sib":
            polylines = [e for e in doc.modelspace().query("LWPOLYLINE") if e.dxf.layer == layer]
            assert len(polylines) >= gt["gt"]["raeume"], name


def test_umlaute_im_dxf_korrekt_kodiert(docs, all_files, data_dir):
    """Raumnamen mit Umlauten stehen als cp1252 im DXF ($DWGCODEPAGE) und lesen sich zurueck."""
    seen_umlaut = False
    for name, (doc, gt) in docs.items():
        layer = gt["konvention"]["layer"]["raum"]
        got = sorted(e.dxf.text for e in doc.modelspace().query("TEXT") if e.dxf.layer == layer)
        assert got == sorted(r["label"] for r in gt["gt"]["raeume_liste"]), name
        seen_umlaut |= any(not t.isascii() for t in got)
    assert seen_umlaut
    for _, f in all_files:
        raw = (data_dir / f["dxf"]).read_bytes()
        assert b"ANSI_1252" in raw
        assert b"\xc3\xbc" not in raw and b"\xc3\xa4" not in raw  # kein UTF-8 im cp1252-Kopf


def test_raumflaechen_und_summen(all_files, data_dir):
    for _, f in all_files:
        gt = _gt(data_dir, f)["gt"]
        rooms = gt["raeume_liste"]
        assert gt["raeume"] == len(rooms) >= 1
        assert abs(sum(r["flaeche_m2"] for r in rooms) - gt["raumflaeche_summe_m2"]) < 0.01
        x0, y0, x1, y1 = gt["bbox_m"]
        bbox_area = (x1 - x0) * (y1 - y0)
        # Raumflaeche liegt plausibel unter der Bounding-Box und ueber 5 % davon
        assert 0.05 * bbox_area < gt["raumflaeche_summe_m2"] < 1.0 * bbox_area
        for r in rooms:
            assert r["flaeche_m2"] >= 1.0
            poly = Polygon(r["polygon"], r.get("loecher", []))
            tol = 0.03 if r["teile"] > 1 else 0.02
            assert abs(poly.area - r["flaeche_m2"]) <= tol * r["flaeche_m2"] + 0.1, (r["id"], poly.area, r["flaeche_m2"])
            assert Polygon(r["polygon"]).buffer(0.02).contains(Point(r["punkt"]))


def test_wandlaenge_summe_und_plausibilitaet(all_files, data_dir):
    for _, f in all_files:
        gt = _gt(data_dir, f)["gt"]
        total = sum(math.dist(w["p0"], w["p1"]) for w in gt["waende"])
        # Koordinaten sind auf 1 cm gerundet: je Wand hoechstens 1,5 cm Fehler
        assert abs(total - gt["wandlaenge_achse_m"]) <= 0.015 * len(gt["waende"]) + 0.05, f["dxf"]
        assert len(gt["waende"]) == gt["n_waende"]
        # Achslaenge und Laenge Flaechenkante unterscheiden sich um wenige Prozent
        assert abs(gt["wandlaenge_achse_m"] - gt["wandlaenge_flaechenkante_m"]) / gt["wandlaenge_achse_m"] < 0.08
        # die Summe der Rechteck-Laengen zaehlt Ueberlappungen doppelt und ist nie kleiner als die Achslaenge minus Knotenzuschlag
        assert gt["wandlaenge_summe_rechtecke_m"] >= 0.9 * gt["wandlaenge_achse_m"]


def test_oeffnungen_sitzen_in_waenden(all_files, data_dir):
    for _, f in all_files:
        gt = _gt(data_dir, f)["gt"]
        walls = gt["waende"]
        for o in gt["oeffnungen"]:
            assert o["wand"] is not None and 0 <= o["wand"] < len(walls)
            w = walls[o["wand"]]
            d = LineString([w["p0"], w["p1"]]).distance(Point(o["mitte"]))
            assert d <= w["T"] / 2 + 0.2, (f["dxf"], o["id"], d)
            assert 0.2 <= o["breite"] <= 6.0
        assert sum(1 for o in gt["oeffnungen"] if o["art"] == "tuer") == gt["tueren"]
        assert sum(1 for o in gt["oeffnungen"] if o["art"] == "fenster") == gt["fenster"]


def test_layer_im_dxf_stammen_aus_der_konvention(docs):
    for name, (doc, gt) in docs.items():
        allowed = {v for v in gt["konvention"]["layer"].values() if v} | {"0"}
        used = {e.dxf.layer for e in doc.modelspace()}
        assert used <= allowed, (name, used - allowed)


def test_stoerelemente_vorhanden(docs):
    for name, (doc, gt) in docs.items():
        kv = gt["konvention"]
        ms = doc.modelspace()
        lay = kv["layer"]
        assert any(e.dxf.layer == lay["bemassung"] for e in ms.query("LINE")), name
        assert any(e.dxf.layer == lay["text"] for e in ms.query("TEXT")), name
        if kv["schraffur"] == "flaeche":
            assert any(e.dxf.layer == lay["schraffur"] for e in ms.query("HATCH")), name
        else:
            assert any(e.dxf.layer == lay["schraffur"] for e in ms.query("LINE")), name


def test_ausfuehrungen_unterscheiden_sich(docs):
    """Doppellinie hat LINE-Waende, Kontur LWPOLYLINE-Waende."""
    for name, (doc, gt) in docs.items():
        kv = gt["konvention"]
        wall_layers = {kv["layer"]["wand_aussen"], kv["layer"]["wand_innen"]}
        lines = [e for e in doc.modelspace().query("LINE") if e.dxf.layer in wall_layers]
        polys = [e for e in doc.modelspace().query("LWPOLYLINE") if e.dxf.layer in wall_layers]
        if kv["wand_ausfuehrung"] == "doppel":
            assert len(lines) > 0, name
        else:
            assert len(polys) > 0 and all(p.closed for p in polys), name
