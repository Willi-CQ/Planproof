"""Metriken: Pipeline-Ergebnis gegen die Ground Truth eines Goldset-Plans.

Dieses Modul importiert keine Pipeline. Es liest das, was die Pipeline schreibt:
``scene.json`` (SceneModel, Format ``scene_v1`` wie ``dwg_bim/pipeline.py`` es schreibt) mit

  WALL   attrs.segment_a, attrs.segment_b (Meter), attrs.length_m
  DOOR / WINDOW  attrs.segment_a, attrs.segment_b (Anfang und Ende der Oeffnung entlang der Wand)
  ROOM   attrs.polygon [[x, y], ...], attrs.area_m2, attrs.label

oder eine explizite Raumliste ``[{"polygon": [[x, y], ...], "area_m2": 12.3, "label": "Bad"}]``.

Koordinaten: Die Pipeline verschiebt die Zeichnung auf einen eigenen Ursprung (``dwg_bim`` zieht das
Minimum aller Linien und Bloecke ab). Die Ground Truth steht im Koordinatensystem der DXF-Datei. Der
Abgleich bestimmt deshalb eine Verschiebung aus drei Kandidaten (Ursprung der DXF laut Ladeverfahren,
keine Verschiebung, Bounding-Box der Wandachsen) und nimmt die mit den meisten Treffern. Welche
Verschiebung gewaehlt wurde, steht im Ergebnis (``ausrichtung``).

Definitionen (siehe GROUND_TRUTH.md):
  Wandlaenge       Summe ``length_m`` der WALL-Komponenten gegen ``wandlaenge_achse_m``
  Oeffnungen       Zuordnung 1:1 nach Art (Tuer/Fenster) und Abstand der Mittelpunkte
                   (Toleranz: halbe GT-Breite plus ``open_tol_m``); Recall = TP / GT, Precision = TP / Ergebnis
  Raeume           Zuordnung ueber einen Punkt der GT-Raumes (Schwerpunkt, sonst Innenpunkt) im Ergebnis-Polygon;
                   Abweichung je Raum = (Flaeche Ergebnis - Flaeche GT) / Flaeche GT
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Point2 = Tuple[float, float]

OPEN_TOL_M = 0.5


# ---------------------------------------------------------------- Laden
def load_gt(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_scene(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dxf_loader_origin(dxf_path: str | Path) -> Optional[Point2]:
    """Ursprung, den ``dwg_bim.dxf_loader`` abzieht: Minimum aller LINE/POLYLINE-Punkte und
    Einfuegepunkte (Meter, aus $INSUNITS). Nur ein Hinweis fuer die Ausrichtung."""
    try:
        import ezdxf  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    doc = ezdxf.readfile(str(dxf_path))
    scale = {4: 0.001, 5: 0.01, 6: 1.0}.get(int(doc.header.get("$INSUNITS", 4) or 4), 0.001)
    xs: List[float] = []
    ys: List[float] = []
    for e in doc.modelspace():
        t = e.dxftype()
        if t == "LINE":
            xs += [e.dxf.start.x, e.dxf.end.x]
            ys += [e.dxf.start.y, e.dxf.end.y]
        elif t == "LWPOLYLINE":
            for x, y, *_ in e.get_points("xyseb"):
                xs.append(x)
                ys.append(y)
        elif t == "POLYLINE":
            for v in e.vertices:
                xs.append(v.dxf.location.x)
                ys.append(v.dxf.location.y)
        elif t == "INSERT":
            xs.append(e.dxf.insert.x)
            ys.append(e.dxf.insert.y)
    if not xs:
        return None
    return (min(xs) * scale, min(ys) * scale)


# ---------------------------------------------------------------- Ergebnis aus scene.json
def _seg(attrs: Dict[str, Any]) -> Optional[Tuple[Point2, Point2]]:
    a, b = attrs.get("segment_a"), attrs.get("segment_b")
    if a and b:
        return (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
    return None


def extract_result(scene: Dict[str, Any], rooms: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Zieht Waende, Oeffnungen und Raeume aus einer scene.json (als Dict)."""
    walls: List[Dict[str, Any]] = []
    doors: List[Point2] = []
    windows: List[Point2] = []
    room_list: List[Dict[str, Any]] = []
    for comp in (scene.get("components") or {}).values():
        kind = str(comp.get("kind", "")).upper()
        attrs = comp.get("attrs") or {}
        if kind == "WALL":
            seg = _seg(attrs)
            length = attrs.get("length_m")
            if length is None and seg:
                length = math.dist(seg[0], seg[1])
            walls.append({"seg": seg, "length": float(length or 0.0)})
        elif kind in ("DOOR", "WINDOW"):
            seg = _seg(attrs)
            if seg:
                mid = ((seg[0][0] + seg[1][0]) / 2, (seg[0][1] + seg[1][1]) / 2)
            else:
                c = comp.get("center")
                if not c:
                    continue
                mid = (float(c[0]), float(c[2]))
            (doors if kind == "DOOR" else windows).append(mid)
        elif kind == "ROOM" and rooms is None:
            poly = attrs.get("polygon")
            if poly:
                room_list.append({"polygon": poly, "area_m2": attrs.get("area_m2"), "label": attrs.get("label")})
    if rooms is not None:
        room_list = [dict(r) for r in rooms]
    return {"walls": walls, "doors": doors, "windows": windows, "rooms": room_list}


# ---------------------------------------------------------------- Geometrie-Hilfen
def _poly(ring: Sequence[Sequence[float]], holes: Optional[Sequence[Sequence[Sequence[float]]]] = None):
    from shapely.geometry import Polygon  # noqa: PLC0415

    p = Polygon([(float(x), float(y)) for x, y in ring], [[(float(x), float(y)) for x, y in h] for h in (holes or [])])
    if not p.is_valid:
        p = p.buffer(0)
    return p


def _shift_pts(pts: Iterable[Point2], dx: float, dy: float) -> List[Point2]:
    return [(x + dx, y + dy) for x, y in pts]


def _match_openings(gt_open: List[Dict[str, Any]], pred: List[Point2], tol_m: float) -> int:
    """Greedy nach Abstand, 1:1. Rueckgabe: Zahl der Treffer."""
    cand: List[Tuple[float, int, int]] = []
    for i, g in enumerate(gt_open):
        gx, gy = g["mitte"]
        lim = 0.5 * float(g["breite"]) + tol_m
        for j, (px, py) in enumerate(pred):
            d = math.hypot(px - gx, py - gy)
            if d <= lim:
                cand.append((d, i, j))
    cand.sort()
    used_i: set = set()
    used_j: set = set()
    tp = 0
    for _d, i, j in cand:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        tp += 1
    return tp


def _match_rooms(gt_rooms: List[Dict[str, Any]], pred_rooms: List[Dict[str, Any]], dx: float, dy: float):
    """GT-Raum -> Ergebnis-Raum ueber einen Punkt des GT-Raums im Ergebnis-Polygon.

    Rueckgabe: (Liste (gt_index, pred_index, rel_abw), Menge benutzter pred_index)
    """
    from shapely.geometry import Point  # noqa: PLC0415

    preds = []
    for r in pred_rooms:
        try:
            p = _poly([(x + dx, y + dy) for x, y in r["polygon"]])
        except Exception:  # noqa: BLE001
            p = None
        area = r.get("area_m2")
        preds.append((p, float(area) if area is not None else (p.area if p is not None else 0.0)))
    used: set = set()
    out: List[Tuple[int, int, float]] = []
    order = sorted(range(len(gt_rooms)), key=lambda i: -gt_rooms[i]["flaeche_m2"])
    for gi in order:
        g = gt_rooms[gi]
        gp = _poly(g["polygon"], g.get("loecher"))
        c = gp.centroid
        pt = c if gp.contains(c) else Point(g["punkt"])
        best = None
        for pj, (pp, _pa) in enumerate(preds):
            if pj in used or pp is None or pp.is_empty:
                continue
            if pp.contains(pt):
                if best is None or pp.area < preds[best][0].area:
                    best = pj
        if best is not None:
            used.add(best)
            pa = preds[best][1]
            out.append((gi, best, (pa - g["flaeche_m2"]) / g["flaeche_m2"]))
    return out, used, preds


def _alignment_candidates(result: Dict[str, Any], gt: Dict[str, Any], origin_hint: Optional[Point2]):
    cands: List[Tuple[str, float, float]] = []
    if origin_hint is not None:
        cands.append(("dxf_ursprung", float(origin_hint[0]), float(origin_hint[1])))
    cands.append(("keine", 0.0, 0.0))
    pts = [p for w in result["walls"] if w["seg"] for p in w["seg"]]
    gpts = [tuple(p) for w in gt["waende"] for p in (w["p0"], w["p1"])]
    if pts and gpts:
        dx = min(p[0] for p in gpts) - min(p[0] for p in pts)
        dy = min(p[1] for p in gpts) - min(p[1] for p in pts)
        cands.append(("wand_bbox", dx, dy))
    return cands


# ---------------------------------------------------------------- Metriken je Plan
def evaluate_plan(
    gt: Dict[str, Any],
    result: Dict[str, Any],
    origin_hint: Optional[Point2] = None,
    open_tol_m: float = OPEN_TOL_M,
) -> Dict[str, Any]:
    """Vergleicht ein extrahiertes Ergebnis (``extract_result``) mit der Ground Truth (``load_gt``)."""
    g = gt["gt"]
    doors_gt = [o for o in g["oeffnungen"] if o["art"] == "tuer"]
    wins_gt = [o for o in g["oeffnungen"] if o["art"] == "fenster"]

    def score(dx: float, dy: float) -> Tuple[int, int]:
        tp = _match_openings(doors_gt, _shift_pts(result["doors"], dx, dy), open_tol_m)
        tp += _match_openings(wins_gt, _shift_pts(result["windows"], dx, dy), open_tol_m)
        rooms_tp = len(_match_rooms(g["raeume_liste"], result["rooms"], dx, dy)[0]) if result["rooms"] else 0
        return tp, rooms_tp

    best = None
    for name, dx, dy in _alignment_candidates(result, g, origin_hint):
        s = score(dx, dy)
        if best is None or (s[0] + s[1]) > best[0]:
            best = (s[0] + s[1], name, dx, dy)
    _, align_name, dx, dy = best  # type: ignore[misc]

    # Waende
    pred_len = sum(w["length"] for w in result["walls"])
    gt_len = float(g["wandlaenge_achse_m"])
    wall = {
        "gt_m": gt_len,
        "ergebnis_m": round(pred_len, 3),
        "abw_pct": round(100.0 * (pred_len - gt_len) / gt_len, 3) if gt_len else None,
        "abw_pct_flaechenkante": round(100.0 * (pred_len - g["wandlaenge_flaechenkante_m"]) / g["wandlaenge_flaechenkante_m"], 3)
        if g.get("wandlaenge_flaechenkante_m") else None,
        "anzahl_ergebnis": len(result["walls"]),
        "anzahl_gt": g["n_waende"],
    }

    def opening(gt_list: List[Dict[str, Any]], pred: List[Point2]) -> Dict[str, Any]:
        tp = _match_openings(gt_list, _shift_pts(pred, dx, dy), open_tol_m)
        n_gt, n_pred = len(gt_list), len(pred)
        return {
            "gt": n_gt, "ergebnis": n_pred, "treffer": tp, "abw_anzahl": n_pred - n_gt,
            "recall": round(tp / n_gt, 4) if n_gt else None,
            "precision": round(tp / n_pred, 4) if n_pred else None,
        }

    # Raeume
    rooms_gt = g["raeume_liste"]
    matches, used, preds = _match_rooms(rooms_gt, result["rooms"], dx, dy)
    # Ergebnis-Raeume, die einen eingeschlossenen Nicht-Raum (Schacht, Aufzug) treffen
    nonroom_hits = 0
    if g.get("nichtraeume"):
        from shapely.geometry import Point  # noqa: PLC0415

        for pj, (pp, _pa) in enumerate(preds):
            if pj in used or pp is None or pp.is_empty:
                continue
            for nr in g["nichtraeume"]:
                c = _poly(nr["polygon"]).representative_point()
                if pp.contains(Point(c.x, c.y)):
                    nonroom_hits += 1
                    break
    rel = [m[2] for m in matches]
    abs_rel = sorted(abs(x) for x in rel)
    gt_area = sum(r["flaeche_m2"] for r in rooms_gt)
    pred_area_matched = sum(preds[m[1]][1] for m in matches)
    gt_area_matched = sum(rooms_gt[m[0]]["flaeche_m2"] for m in matches)
    rooms = {
        "gt": len(rooms_gt), "ergebnis": len(result["rooms"]), "treffer": len(matches),
        "abw_anzahl": len(result["rooms"]) - len(rooms_gt),
        "ergebnis_nichtraum_treffer": nonroom_hits,
        "recall": round(len(matches) / len(rooms_gt), 4) if rooms_gt else None,
        "precision": round(len(matches) / max(1, len(result["rooms"]) - nonroom_hits), 4) if result["rooms"] else None,
        "rel_abw_je_raum": [round(x, 4) for x in rel],
        "median_abs_rel_abw": round(statistics.median(abs_rel), 4) if abs_rel else None,
        "p90_abs_rel_abw": round(_percentile(abs_rel, 90), 4) if abs_rel else None,
        "median_rel_abw_mit_vorzeichen": round(statistics.median(rel), 4) if rel else None,
        "flaeche_gt_summe_m2": round(gt_area, 3),
        "flaeche_summe_abw_pct_treffer": round(100.0 * (pred_area_matched - gt_area_matched) / gt_area_matched, 3)
        if gt_area_matched else None,
    }
    return {
        "ausrichtung": {"modus": align_name, "dx": round(dx, 4), "dy": round(dy, 4)},
        "waende": wall,
        "tueren": opening(doors_gt, result["doors"]),
        "fenster": opening(wins_gt, result["windows"]),
        "raeume": rooms,
    }


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    """Perzentil mit linearer Interpolation (wie numpy), ``sorted_vals`` aufsteigend."""
    if not sorted_vals:
        raise ValueError("leer")
    k = (len(sorted_vals) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


# ---------------------------------------------------------------- Zusammenfassung
def _stats(vals: List[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"n": 0, "median": None, "mittel": None, "p90": None}
    s = sorted(vals)
    return {"n": len(s), "median": round(statistics.median(s), 4), "mittel": round(statistics.fmean(s), 4),
            "p90": round(_percentile(s, 90), 4)}


def _micro(items: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    gt = sum(i[key]["gt"] for i in items)
    pred = sum(i[key]["ergebnis"] for i in items)
    tp = sum(i[key]["treffer"] for i in items)
    return {
        "gt": gt, "ergebnis": pred, "treffer": tp,
        "recall": round(tp / gt, 4) if gt else None,
        "precision": round(tp / pred, 4) if pred else None,
        "abw_anzahl_median": _stats([float(i[key]["abw_anzahl"]) for i in items])["median"],
    }


def summarize(per_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Faltet die Einzelergebnisse zusammen: gesamt und je Konvention.

    ``per_plan``: Eintraege ``{"datei", "konvention", "variante", "plan_key", "fehler": str|None,
    "metriken": evaluate_plan(...) | None}``. Ein Lauf mit Fehler zaehlt bei Recall mit null Treffern
    (GT-Anzahl geht ein, Ergebnis-Anzahl ist null) und bei Wandlaengen und Raumflaechen gar nicht.
    """

    def block(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        ok = [i for i in items if i.get("metriken")]
        failed = [i for i in items if not i.get("metriken")]
        mets = [i["metriken"] for i in ok]
        wall_abs = [abs(i["metriken"]["waende"]["abw_pct"]) for i in ok if i["metriken"]["waende"]["abw_pct"] is not None]
        wall_signed = [i["metriken"]["waende"]["abw_pct"] for i in ok if i["metriken"]["waende"]["abw_pct"] is not None]
        room_abs: List[float] = []
        room_signed: List[float] = []
        for i in ok:
            for x in i["metriken"]["raeume"]["rel_abw_je_raum"]:
                room_abs.append(abs(x))
                room_signed.append(x)
        # fehlgeschlagene Laeufe: GT-Zahlen gehen in Recall ein
        fail_gt = {k: sum(i["gt_zahlen"][k] for i in failed if i.get("gt_zahlen")) for k in ("tueren", "fenster", "raeume")}

        def with_fail(m: Dict[str, Any], k: str) -> Dict[str, Any]:
            gt = m["gt"] + fail_gt[k]
            return {**m, "gt": gt, "recall": round(m["treffer"] / gt, 4) if gt else None}

        return {
            "dateien": len(items),
            "fehlgeschlagen": len(failed),
            "wandlaenge_abw_pct_abs": _stats(wall_abs),
            "wandlaenge_abw_pct_mit_vorzeichen": _stats(wall_signed),
            "tueren": with_fail(_micro(mets, "tueren"), "tueren") if ok else None,
            "fenster": with_fail(_micro(mets, "fenster"), "fenster") if ok else None,
            "raeume": {
                **(with_fail(_micro(mets, "raeume"), "raeume") if ok else {}),
                # Precision der Raeume ohne Ergebnis-Raeume, die einen Schacht/Aufzug treffen
                "precision": (
                    round(
                        sum(m["raeume"]["treffer"] for m in mets)
                        / max(1, sum(m["raeume"]["ergebnis"] - m["raeume"]["ergebnis_nichtraum_treffer"] for m in mets)),
                        4,
                    )
                    if ok
                    else None
                ),
                "rel_abw_abs": _stats(room_abs),
                "rel_abw_mit_vorzeichen": _stats(room_signed),
                "abw_anzahl_abs": _stats([abs(float(i["metriken"]["raeume"]["abw_anzahl"])) for i in ok]),
            },
        }

    by_conv: Dict[str, List[Dict[str, Any]]] = {}
    for it in per_plan:
        by_conv.setdefault(it["konvention"], []).append(it)
    return {"gesamt": block(per_plan), "je_konvention": {k: block(v) for k, v in sorted(by_conv.items())}}


def _fmt(v: Optional[float], digits: int = 1, pct: bool = False) -> str:
    if v is None:
        return "-"
    if pct:
        return f"{100 * v:.{digits}f} %"
    return f"{v:.{digits}f}"


def summary_markdown(summary: Dict[str, Any], title: str = "Goldset-Ergebnis") -> str:
    """Eine Markdown-Tabelle: eine Zeile je Konvention und gesamt."""
    head = (
        "| Konvention | Dateien | Fehler | Wandlänge Abw. Median / P90 (%) | Tür Recall / Precision | "
        "Fenster Recall / Precision | Raum Recall / Precision | Raumzahl Abw. Median | "
        "Raumfläche Abw. Median / P90 (%) (n Räume) |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows: List[str] = []

    def row(name: str, b: Dict[str, Any]) -> str:
        w = b["wandlaenge_abw_pct_abs"]
        d, f, r = b["tueren"] or {}, b["fenster"] or {}, b["raeume"]
        return (
            f"| {name} | {b['dateien']} | {b['fehlgeschlagen']} | {_fmt(w['median'])} / {_fmt(w['p90'])} | "
            f"{_fmt(d.get('recall'), pct=True)} / {_fmt(d.get('precision'), pct=True)} | "
            f"{_fmt(f.get('recall'), pct=True)} / {_fmt(f.get('precision'), pct=True)} | "
            f"{_fmt(r.get('recall'), pct=True)} / {_fmt(r.get('precision'), pct=True)} | "
            f"{_fmt(r['abw_anzahl_abs']['median'])} | "
            f"{_fmt(None if r['rel_abw_abs']['median'] is None else 100 * r['rel_abw_abs']['median'])} / "
            f"{_fmt(None if r['rel_abw_abs']['p90'] is None else 100 * r['rel_abw_abs']['p90'])} "
            f"({r['rel_abw_abs']['n']}) |"
        )

    for k, b in summary["je_konvention"].items():
        rows.append(row(k, b))
    rows.append(row("**gesamt**", summary["gesamt"]))
    return f"## {title}\n\n" + head + "\n".join(rows) + "\n"
