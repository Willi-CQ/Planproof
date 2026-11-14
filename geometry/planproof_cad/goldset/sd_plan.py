"""Ein Swiss-Dwellings-Geschoss -> bereinigte Geometrie und Ground Truth.

Eingabe sind die Zeilen eines Geschosses aus geometries.csv. Ausgabe ist ein Plan-Dict
in Metern, gedreht auf die Hauptrichtung der Waende und verschoben, mit

  - waende:    Achssegmente (zusammengefasst, Knoten auf die Achsschnittpunkte gezogen)
  - rects:     die Wandrechtecke, so wie sie gezeichnet werden (ohne Duplikate)
  - oeffnungen: Tueren und Fenster (ohne Duplikate) mit Wirtswand
  - raeume:    Raeume im Sinn von GROUND_TRUTH.md (wandbegrenzte Raeume, offene Nachbarn
               zusammengefasst) mit lichter Flaeche
  - Stoerobjekte: Gelaender, Stuetzen, Ausstattung (nur zum Zeichnen)

Wenn die Wahrheit nicht eindeutig ist, liefert analyse_geschoss() eine Ausschlussliste
(`gruende`) und der Plan wird nicht ins Goldset aufgenommen. Die Schwellen stehen als
Konstanten oben und sind in GROUND_TRUTH.md beschrieben.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely import affinity, wkt
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

# ---------------------------------------------------------------- Schwellen
TOL_COORD_DEDUP = 0.005          # m, Rundung fuer exakte Duplikate
PARALLEL_SIN = math.sin(math.radians(1.5))   # Winkelgrenze "parallel"
SAME_WALL_LATERAL = 0.35         # Achsversatz / Wanddicke, darunter = dieselbe Wand
SAME_WALL_GAP = 0.02             # m, hoechste Luecke zwischen Teilstuecken derselben Wand
AMBIG_OVERLAP_ALONG = 0.10       # m, ab dieser Ueberlappung zaehlt ein Versatz als mehrdeutig
NON_RECT_RATIO = 0.93            # Flaeche / Flaeche des umschliessenden Drehrechtecks
NON_RECT_SHARE_MAX = 0.02        # Flaechenanteil nicht rechteckiger Waende, darueber Ausschluss
AMBIG_SHARE_MAX = 0.02           # Anteil der mehrdeutig doppelten Wandlaenge, darueber Ausschluss
MIN_WALL_LEN = 0.05              # m
SNAP_MIN_ANGLE_DEG = 15.0        # Wandanschluesse nur unter diesem Mindestwinkel
SNAP_EXTRA = 0.05                # m Toleranz beim Ziehen auf den Achsschnittpunkt
OPEN_HOST_MIN_FRACTION = 0.60    # Anteil der Oeffnungsflaeche, der in einer Wand liegen muss
OPEN_DEDUP_IOU = 0.5
OPEN_OVERLAP_FRACTION = 0.10     # Ueberlappung zweier Oeffnungen (Anteil der kleineren)
MIN_OPENING_WIDTH = 0.30         # m, schmalere Oeffnungen werden nur gezaehlt (Diagnose)
ROOM_MIN_AREA = 1.0              # m2, wie room_min_area_m2 der Pipeline
ROOM_TOUCH_DIST = 0.02           # m, Raeume, die sich beruehren, sind nicht durch Wand getrennt
ROOM_TOUCH_LEN = 0.20            # m, gemeinsame Kantenlaenge fuer "beruehren"
ROOM_CLOSING = 0.01              # m, Schliessen kleiner Luecken zwischen offenen Nachbarraeumen
ROOM_BOUNDARY_BUFFER = 0.05      # m
ROOM_BOUNDARY_COVER = 0.90       # Anteil der Raumkontur, der an Wand/Oeffnung/Stuetze liegen muss
AREA_OVERLAP_FRACTION = 0.05     # Ueberlappung zweier Flaechen (Anteil der kleineren)

OUTDOOR_TYPES = {
    "BALCONY", "LOGGIA", "TERRACE", "GARDEN", "PATIO", "OUTDOOR_VOID", "LIGHTWELL", "CARPARK",
    "RAMP",
}
NONROOM_ENCLOSED_TYPES = {"SHAFT", "ELEVATOR", "VOID"}

DOOR_TYPES = {"DOOR", "ENTRANCE_DOOR"}
WINDOW_TYPES = {"WINDOW"}

ROOM_LABELS_DE = {
    "ROOM": "Zimmer",
    "BEDROOM": "Schlafzimmer",
    "LIVING_ROOM": "Wohnzimmer",
    "LIVING_DINING": "Wohnen/Essen",
    "DINING": "Essen",
    "KITCHEN": "Küche",
    "KITCHEN_DINING": "Wohnküche",
    "COMMON_KITCHEN": "Gemeinschaftsküche",
    "BATHROOM": "Bad",
    "SANITARY_ROOMS": "Sanitär",
    "CORRIDOR": "Flur",
    "CORRIDORS_AND_HALLS": "Flur",
    "LOBBY": "Eingang",
    "STOREROOM": "Abstellraum",
    "STAIRCASE": "Treppenhaus",
    "OFFICE": "Büro",
    "OFFICE_SPACE": "Büro",
    "OFFICE_TECH_ROOM": "Technik",
    "TECHNICAL_AREA": "Technik",
    "HEATING": "Heizung",
    "BASEMENT": "Keller",
    "BASEMENT_COMPARTMENT": "Kellerabteil",
    "WASH_AND_DRY_ROOM": "Waschküche",
    "BIKE_STORAGE": "Fahrradraum",
    "PRAM_AND_BIKE_STORAGE_ROOM": "Fahrradraum",
    "COMMUNITY_ROOM": "Gemeinschaftsraum",
    "MEETING_ROOM": "Besprechung",
    "SALESROOM": "Verkauf",
    "SHELTER": "Schutzraum",
    "SPORTS_ROOMS": "Sport",
    "OPERATIONS_FACILITIES": "Betrieb",
    "GARAGE": "Garage",
    "WINTERGARTEN": "Wintergarten",
    "NOT_DEFINED": "Raum",
}


# ---------------------------------------------------------------- Hilfen
def _r3(v: float) -> float:
    return round(float(v), 3)


def _poly_key(poly: Polygon) -> Tuple:
    p = poly.normalize()
    return tuple((round(x / TOL_COORD_DEDUP), round(y / TOL_COORD_DEDUP)) for x, y in p.exterior.coords)


def _ring(poly: Polygon) -> List[List[float]]:
    return [[_r3(x), _r3(y)] for x, y in poly.exterior.coords[:-1]]


def _as_polygon(g) -> Optional[Polygon]:
    """Gibt ein einzelnes gueltiges Polygon zurueck oder None."""
    if g is None or g.is_empty:
        return None
    if not g.is_valid:
        g = g.buffer(0)
    if isinstance(g, MultiPolygon):
        parts = [p for p in g.geoms if p.area > 1e-6]
        if len(parts) != 1:
            return None
        g = parts[0]
    if not isinstance(g, Polygon) or g.is_empty or g.area < 1e-6:
        return None
    return g


@dataclass
class Rect:
    poly: Polygon
    c: np.ndarray
    u: np.ndarray       # Einheitsvektor der Laengsachse
    L: float
    T: float
    ratio: float
    group: int = -1

    @property
    def n(self) -> np.ndarray:
        return np.array([-self.u[1], self.u[0]])


def rect_fit(poly: Polygon) -> Optional[Rect]:
    mrr = poly.minimum_rotated_rectangle
    if not isinstance(mrr, Polygon) or mrr.area < 1e-9:
        return None
    c = np.array(mrr.exterior.coords[:4], dtype=float)
    e1 = c[1] - c[0]
    e2 = c[2] - c[1]
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    if l1 >= l2:
        u, L, T = e1 / l1, l1, l2
    else:
        u, L, T = e2 / l2, l2, l1
    if u[0] < -1e-12 or (abs(u[0]) <= 1e-12 and u[1] < 0):
        u = -u
    return Rect(poly=poly, c=c.mean(axis=0), u=u, L=L, T=T, ratio=float(poly.area / mrr.area))


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


# ---------------------------------------------------------------- Drehung
def dominant_angle(rects: Sequence[Rect]) -> float:
    """Hauptrichtung der Waende in Radiant (modulo 90 Grad), laengengewichtet."""
    if not rects:
        return 0.0
    s = c = 0.0
    for r in rects:
        a = math.atan2(r.u[1], r.u[0])
        s += r.L * math.sin(4 * a)
        c += r.L * math.cos(4 * a)
    return math.atan2(s, c) / 4.0


# ---------------------------------------------------------------- Waende
def merge_walls(rects: List[Rect]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fasst Wandrechtecke zu Achssegmenten zusammen und zieht die Enden auf Achsschnittpunkte.

    Rueckgabe: (waende, diag). Jede Wand: p0, p1 (Achsenden), T, n_teile, laenge_roh
    (Summe der Rechteck-Laengen der Teile), laenge_zusammen (vor dem Ziehen der Enden).
    """
    n = len(rects)
    uf = _UF(n)
    ambiguous = 0
    ambiguous_len = 0.0
    if n:
        tree = STRtree([r.poly for r in rects])
        for i, ri in enumerate(rects):
            cand = tree.query(ri.poly.buffer(SAME_WALL_GAP + 0.01))
            for j in cand:
                j = int(j)
                if j <= i:
                    continue
                rj = rects[j]
                cross = abs(ri.u[0] * rj.u[1] - ri.u[1] * rj.u[0])
                if cross > PARALLEL_SIN:
                    continue
                d = abs(float(np.dot(rj.c - ri.c, ri.n)))
                s = float(np.dot(rj.c - ri.c, ri.u))
                overlap = min(ri.L / 2, s + rj.L / 2) - max(-ri.L / 2, s - rj.L / 2)
                tmax = max(ri.T, rj.T)
                if d <= SAME_WALL_LATERAL * tmax and overlap >= -SAME_WALL_GAP:
                    uf.union(i, j)
                elif overlap >= AMBIG_OVERLAP_ALONG and d < 0.9 * (ri.T + rj.T) / 2:
                    # Rechtecke ueberlappen seitlich teilweise: weder dieselbe Wand noch
                    # zwei aneinanderliegende Waende. Unsicher ist die Laenge der Ueberlappung
                    # (doppelt gezaehlt oder nicht), und zwar hoechstens die der kuerzeren Wand.
                    ambiguous += 1
                    ambiguous_len += min(overlap, ri.L, rj.L)
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    walls: List[Dict[str, Any]] = []
    for members in groups.values():
        longest = max(members, key=lambda k: rects[k].L)
        u = rects[longest].u.copy()
        nrm = np.array([-u[1], u[0]])
        wsum = sum(rects[k].L for k in members)
        lat = sum(float(np.dot(rects[k].c, nrm)) * rects[k].L for k in members) / wsum
        lo = min(float(np.dot(rects[k].c, u)) - rects[k].L / 2 for k in members)
        hi = max(float(np.dot(rects[k].c, u)) + rects[k].L / 2 for k in members)
        T = max(rects[k].T for k in members)
        for k in members:
            rects[k].group = len(walls)
        walls.append({
            "u": u, "n": nrm, "lat": lat, "lo": lo, "hi": hi, "T": T,
            "members": members, "roh": wsum,
        })
    walls = [w for w in walls if w["hi"] - w["lo"] >= MIN_WALL_LEN]
    # Gruppennummern nach dem Filtern neu vergeben (verworfene Kleinstteile bekommen -1)
    for r in rects:
        r.group = -1
    for wi, w in enumerate(walls):
        for k in w["members"]:
            rects[k].group = wi

    def pt(w, s):
        return w["u"] * s + w["n"] * w["lat"]

    merged_len = sum(w["hi"] - w["lo"] for w in walls)
    # Enden auf Achsschnittpunkte ziehen
    min_cross = math.sin(math.radians(SNAP_MIN_ANGLE_DEG))
    box_polys = [
        Polygon([
            pt(w, w["lo"]) + w["n"] * w["T"] / 2, pt(w, w["hi"]) + w["n"] * w["T"] / 2,
            pt(w, w["hi"]) - w["n"] * w["T"] / 2, pt(w, w["lo"]) - w["n"] * w["T"] / 2,
        ])
        for w in walls
    ]
    tree = STRtree(box_polys) if box_polys else None
    snapped = 0
    for ai, a in enumerate(walls):
        a["lo2"], a["hi2"] = a["lo"], a["hi"]
        for end in ("lo", "hi"):
            s_end = a[end]
            P = pt(a, s_end)
            best = None
            for bj in tree.query(Point(P).buffer(a["T"] / 2 + 0.06)):
                bj = int(bj)
                if bj == ai:
                    continue
                b = walls[bj]
                cr = abs(a["u"][0] * b["u"][1] - a["u"][1] * b["u"][0])
                if cr < min_cross:
                    continue
                if box_polys[bj].distance(Point(P)) > a["T"] / 2 + 0.04:
                    continue
                # Schnitt der Achsgeraden: P + t*u_a liegt auf der Achse von b
                # (x - Ob)*n_b = lat_b  mit x = u_a*s + n_a*lat_a
                denom = float(np.dot(a["u"], b["n"]))
                if abs(denom) < 1e-9:
                    continue
                s_q = (b["lat"] - float(np.dot(a["n"] * a["lat"], b["n"]))) / denom
                if abs(s_q - s_end) > b["T"] / 2 + SNAP_EXTRA:
                    continue
                Q = pt(a, s_q)
                sb = float(np.dot(Q, b["u"]))
                if sb < b["lo"] - a["T"] / 2 - SNAP_EXTRA or sb > b["hi"] + a["T"] / 2 + SNAP_EXTRA:
                    continue
                cost = abs(s_q - s_end)
                if best is None or cost < best[0]:
                    best = (cost, s_q)
            if best is not None:
                a[end + "2"] = best[1]
                if abs(best[1] - s_end) > 1e-6:
                    snapped += 1
    out = []
    for w in walls:
        p0 = pt(w, w["lo2"])
        p1 = pt(w, w["hi2"])
        out.append({
            "p0": [_r3(p0[0]), _r3(p0[1])], "p1": [_r3(p1[0]), _r3(p1[1])],
            "T": _r3(w["T"]), "n_teile": len(w["members"]),
            "laenge_roh": _r3(w["roh"]), "laenge_zusammen": _r3(w["hi"] - w["lo"]),
            "laenge_achse": _r3(abs(w["hi2"] - w["lo2"])),
            "members": w["members"],
        })
    diag = {
        "waende_zusammengefasst": len(out),
        "mehrdeutige_parallelpaare": ambiguous,
        "mehrdeutige_laenge_m": round(ambiguous_len, 3),
        "enden_gezogen": snapped,
        "summe_zusammen": merged_len,
    }
    return out, diag


# ---------------------------------------------------------------- Analyse
@dataclass
class Analyse:
    plan_id: str
    floor_id: str
    gruende: List[str] = field(default_factory=list)
    diag: Dict[str, Any] = field(default_factory=dict)
    plan: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return not self.gruende


def _parse(rows: List[Dict[str, str]], diag: Dict[str, Any]):
    """Parst die WKT-Geometrien und trennt nach Typ. Exakte Duplikate fallen weg."""
    walls: List[Polygon] = []
    rails: List[Polygon] = []
    cols: List[Polygon] = []
    opens: List[Tuple[str, Polygon]] = []
    areas: List[Tuple[str, Polygon]] = []
    feats: List[Tuple[str, Polygon]] = []
    seen = defaultdict(set)
    invalid = 0
    raw = Counter()
    for r in rows:
        et, es = r["entity_type"], r["entity_subtype"]
        raw[f"{et}/{es}"] += 1
        g = wkt.loads(r["geometry"])
        if not g.is_valid:
            invalid += 1
        p = _as_polygon(g)
        if p is None:
            diag["ungueltig_verworfen"] = diag.get("ungueltig_verworfen", 0) + 1
            continue
        ecken = diag.setdefault("eckpunkte", {})
        grp = f"{et}/{es}" if es in ("WALL", "DOOR", "WINDOW", "ENTRANCE_DOOR") else et
        ecken.setdefault(grp, Counter())[len(p.exterior.coords) - 1] += 1
        if p.interiors:
            diag["flaechen_mit_loch"] = diag.get("flaechen_mit_loch", 0) + 1
        key = (et, es, _poly_key(p))
        if key in seen[et]:
            diag.setdefault("exakte_duplikate", Counter())[f"{et}/{es}"] += 1
            continue
        seen[et].add(key)
        if et == "separator":
            if es == "WALL":
                walls.append(p)
            elif es == "RAILING":
                rails.append(p)
            elif es == "COLUMN":
                cols.append(p)
        elif et == "opening":
            opens.append((es, p))
        elif et == "area":
            areas.append((es, p))
        elif et == "feature":
            feats.append((es, p))
    if "eckpunkte" in diag:
        diag["eckpunkte"] = {k: dict(v) for k, v in diag["eckpunkte"].items()}
    diag["zeilen_roh"] = dict(raw)
    diag["ungueltige_polygone_repariert"] = invalid
    if "exakte_duplikate" in diag:
        diag["exakte_duplikate"] = dict(diag["exakte_duplikate"])
    return walls, rails, cols, opens, areas, feats


def _dedupe_openings(opens: List[Tuple[str, Polygon]], diag: Dict[str, Any]):
    """Entfernt fast gleiche Oeffnungen (Tueren und Fenster getrennt) und zaehlt Ueberlappungen."""
    kept: List[Tuple[str, Polygon]] = []
    near_dup = 0
    overlaps = 0
    cls = lambda s: "door" if s in DOOR_TYPES else "window"  # noqa: E731
    if not opens:
        return kept
    tree = STRtree([p for _, p in opens])
    drop = set()
    for i, (si, pi) in enumerate(opens):
        if i in drop:
            continue
        for j in tree.query(pi):
            j = int(j)
            if j <= i or j in drop:
                continue
            sj, pj = opens[j]
            inter = pi.intersection(pj).area
            if inter <= 0:
                continue
            small = min(pi.area, pj.area)
            if cls(si) == cls(sj) and inter / (pi.union(pj).area) >= OPEN_DEDUP_IOU:
                drop.add(j)
                near_dup += 1
            elif inter / small >= OPEN_OVERLAP_FRACTION:
                overlaps += 1
    for i, o in enumerate(opens):
        if i not in drop:
            kept.append(o)
    diag["oeffnungen_fast_duplikate"] = near_dup
    diag["oeffnungen_ueberlappen"] = overlaps
    return kept


def analysiere_geschoss(rows: List[Dict[str, str]], plan_id: str, floor_id: str) -> Analyse:
    """Baut die Ground Truth fuer ein Geschoss oder nennt die Ausschlussgruende."""
    res = Analyse(plan_id=plan_id, floor_id=floor_id)
    diag = res.diag
    walls_p, rails, cols, opens, areas, feats = _parse(rows, diag)
    gr = res.gruende

    if len(walls_p) < 4:
        gr.append("zu_wenige_waende")
        return res
    rects: List[Rect] = []
    for p in walls_p:
        rf = rect_fit(p)
        if rf is None:
            diag["wand_entartet"] = diag.get("wand_entartet", 0) + 1
            continue
        rects.append(rf)
    ratios = [r.ratio for r in rects]
    nonrect = [r for r in rects if r.ratio < NON_RECT_RATIO]
    diag["wand_nicht_rechteckig"] = len(nonrect)
    diag["wand_ratio_min"] = _r3(min(ratios)) if ratios else None
    diag["wand_nicht_rechteckig_flaechenanteil"] = (
        round(sum(r.poly.area for r in nonrect) / sum(r.poly.area for r in rects), 4) if rects else 0.0
    )
    if diag["wand_nicht_rechteckig_flaechenanteil"] > NON_RECT_SHARE_MAX:
        gr.append("wand_nicht_rechteckig")

    # --- Drehung auf die Hauptrichtung, Verschiebung auf positive Koordinaten
    phi = dominant_angle(rects)
    diag["drehung_grad"] = _r3(math.degrees(phi))
    rot = lambda g: affinity.rotate(g, -phi, origin=(0, 0), use_radians=True)  # noqa: E731
    walls_p = [rot(p) for p in walls_p]
    rails = [rot(p) for p in rails]
    cols = [rot(p) for p in cols]
    opens = [(s, rot(p)) for s, p in opens]
    areas = [(s, rot(p)) for s, p in areas]
    feats = [(s, rot(p)) for s, p in feats]
    allg = unary_union(walls_p + [p for _, p in areas] + rails + cols)
    minx, miny, maxx, maxy = allg.bounds
    diag["bbox_nach_drehung_m"] = [_r3(maxx - minx), _r3(maxy - miny)]
    dx, dy = 10.0 - minx, 10.0 - miny
    mv = lambda g: affinity.translate(g, dx, dy)  # noqa: E731
    walls_p = [mv(p) for p in walls_p]
    rails = [mv(p) for p in rails]
    cols = [mv(p) for p in cols]
    opens = [(s, mv(p)) for s, p in opens]
    areas = [(s, mv(p)) for s, p in areas]
    feats = [(s, mv(p)) for s, p in feats]

    rects = [r for r in (rect_fit(p) for p in walls_p) if r is not None]
    diag["wand_rechtecke"] = len(rects)
    ortho = sum(r.L for r in rects if min(abs(r.u[0]), abs(r.u[1])) < math.sin(math.radians(2.0)))
    diag["anteil_rechtwinklig_2grad"] = _r3(ortho / sum(r.L for r in rects))

    # redundante Rechtecke (>= 95 % in einem anderen enthalten) fuer die Zeichnung entfernen
    tree = STRtree([r.poly for r in rects])
    redundant = set()
    for i, ri in enumerate(rects):
        for j in tree.query(ri.poly):
            j = int(j)
            if j == i or j in redundant:
                continue
            rj = rects[j]
            inter = ri.poly.intersection(rj.poly).area
            if inter >= 0.95 * ri.poly.area and (ri.poly.area < rj.poly.area or (ri.poly.area == rj.poly.area and i > j)):
                redundant.add(i)
                break
    draw_idx = [i for i in range(len(rects)) if i not in redundant]
    diag["wand_rechtecke_redundant"] = len(redundant)

    walls, wdiag = merge_walls(rects)
    diag.update(wdiag)
    L_axis = sum(w["laenge_achse"] for w in walls)
    diag["mehrdeutige_laenge_anteil"] = round(wdiag["mehrdeutige_laenge_m"] / L_axis, 4) if L_axis else 0.0
    if diag["mehrdeutige_laenge_anteil"] > AMBIG_SHARE_MAX:
        gr.append("wand_parallel_mehrdeutig")
    L_merged = sum(w["laenge_zusammen"] for w in walls)
    L_roh_all = sum(r.L for r in rects)
    diag["laenge_achse"] = _r3(L_axis)
    diag["laenge_zusammen"] = _r3(L_merged)
    diag["laenge_summe_rechtecke_alle"] = _r3(L_roh_all)
    diag["laengen_spanne_pct"] = _r3(100.0 * abs(L_axis - L_merged) / L_axis) if L_axis else None

    # --- aussen/innen (nur zur Layerwahl der Zeichnung)
    union_w = unary_union([r.poly for r in rects])
    polys = list(union_w.geoms) if hasattr(union_w, "geoms") else [union_w]
    filled = unary_union([Polygon(p.exterior) for p in polys])
    for w in walls:
        p0, p1 = np.array(w["p0"]), np.array(w["p1"])
        d = p1 - p0
        L = float(np.hypot(*d))
        if L < 1e-6:
            w["aussen"] = False
            continue
        u = d / L
        nrm = np.array([-u[1], u[0]])
        votes = 0
        for t in (0.25, 0.5, 0.75):
            q = p0 + d * t
            off = w["T"] / 2 + 0.05
            if not filled.contains(Point(q + nrm * off)) or not filled.contains(Point(q - nrm * off)):
                votes += 1
        w["aussen"] = votes >= 2
    rect_ext = {i: walls[rects[i].group]["aussen"] if rects[i].group >= 0 else False for i in range(len(rects))}

    # --- Oeffnungen
    opens = _dedupe_openings(opens, diag) or []
    if diag.get("oeffnungen_ueberlappen"):
        gr.append("oeffnung_ueberlappt")
    wall_boxes = [r.poly for r in rects]
    wtree = STRtree(wall_boxes)
    wall_union_for_host = union_w
    out_opens = []
    no_host = tiny = 0
    for sub, p in opens:
        rf = rect_fit(p)
        if rf is None:
            continue
        best = None
        for j in wtree.query(p):
            j = int(j)
            a = p.intersection(rects[j].poly).area
            if best is None or a > best[0]:
                best = (a, j)
        frac = p.intersection(wall_union_for_host).area / p.area
        host = None
        if best is not None and frac >= OPEN_HOST_MIN_FRACTION and rects[best[1]].group >= 0:
            host = rects[best[1]].group
            # Richtung und Breite kommen von der Wirtswand: die Breite ist die Ausdehnung des
            # Oeffnungspolygons ENTLANG der Wand, die Dicke die quer dazu. Das ist auch bei
            # Oeffnungen eindeutig, die schmaler als die Wand dick sind.
            u = rects[best[1]].u
            nrm = np.array([-u[1], u[0]])
            pts = np.array(p.exterior.coords[:-1])
            along = pts @ u
            across = pts @ nrm
            width = float(along.max() - along.min())
            depth = float(across.max() - across.min())
            centre = pts.mean(axis=0)
            direction = u
        else:
            no_host += 1
            width, depth, centre, direction = rf.L, rf.T, rf.c, rf.u
        if width < MIN_OPENING_WIDTH:
            tiny += 1
        out_opens.append({
            "id": f"o{len(out_opens)}",
            "art": "tuer" if sub in DOOR_TYPES else "fenster",
            "sd_typ": sub,
            "mitte": [_r3(centre[0]), _r3(centre[1])],
            "richtung": [_r3(direction[0]), _r3(direction[1])],
            "breite": _r3(width),
            "dicke": _r3(depth),
            "polygon": _ring(p),
            "wand": host,
        })
    diag["oeffnung_in_wand_anteil"] = [
        round(p.intersection(wall_union_for_host).area / p.area, 4) for _s, p in opens
    ]
    diag["oeffnung_breite_tuer_fenster"] = [(o["art"], o["breite"], o["dicke"]) for o in out_opens]
    diag["wand_dicken"] = [w["T"] for w in walls]
    diag["oeffnung_ohne_wand"] = no_host
    diag["oeffnung_schmal"] = tiny
    if no_host:
        gr.append("oeffnung_ohne_wand")

    # --- Raeume
    col_union = unary_union(cols) if cols else None
    open_union = unary_union([p for _, p in opens]) if opens else None
    wallish_parts = [union_w]
    if open_union is not None:
        wallish_parts.append(open_union)
    if col_union is not None:
        wallish_parts.append(col_union)
    wallish = unary_union(wallish_parts).buffer(ROOM_BOUNDARY_BUFFER)

    area_polys = [(s, p) for s, p in areas]
    # Flaechen, die sich ueberlappen -> mehrdeutig
    ov = 0
    if area_polys:
        atree = STRtree([p for _, p in area_polys])
        for i, (_si, pi) in enumerate(area_polys):
            for j in atree.query(pi):
                j = int(j)
                if j <= i:
                    continue
                inter = pi.intersection(area_polys[j][1]).area
                if inter / min(pi.area, area_polys[j][1].area) >= AREA_OVERLAP_FRACTION:
                    ov += 1
    diag["flaechen_ueberlappen"] = ov
    if ov:
        gr.append("flaechen_ueberlappen")

    room_items = [(s, p) for s, p in area_polys
                  if s not in OUTDOOR_TYPES and s not in NONROOM_ENCLOSED_TYPES and p.area >= ROOM_MIN_AREA]
    other_encl = [(s, p) for s, p in area_polys if s in NONROOM_ENCLOSED_TYPES or (
        s not in OUTDOOR_TYPES and p.area < ROOM_MIN_AREA)]
    diag["raum_typ_flaeche"] = [(s, round(p.area, 3)) for s, p in room_items]
    diag["flaechen_aussenbereich"] = sum(1 for s, _ in area_polys if s in OUTDOOR_TYPES)
    diag["flaechen_schacht_aufzug_void_klein"] = len(other_encl)
    diag["flaechen_raum_roh"] = len(room_items)

    uf = _UF(len(room_items))
    if room_items:
        rtree = STRtree([p for _, p in room_items])
        for i, (_si, pi) in enumerate(room_items):
            for j in rtree.query(pi.buffer(ROOM_TOUCH_DIST)):
                j = int(j)
                if j <= i:
                    continue
                pj = room_items[j][1]
                shared = pi.boundary.intersection(pj.buffer(ROOM_TOUCH_DIST)).length
                if shared >= ROOM_TOUCH_LEN:
                    uf.union(i, j)
    comp: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(room_items)):
        comp[uf.find(i)].append(i)
    rooms = []
    open_rooms = 0
    merged_cnt = 0
    multipoly = 0
    for _k, members in sorted(comp.items()):
        parts = [room_items[m][1] for m in members]
        u = unary_union(parts)
        sum_area = sum(p.area for p in parts)
        if len(members) > 1:
            merged_cnt += 1
            # Benachbarte Flaechen liegen etwa 1 cm auseinander (Digitalisierung); ein
            # Schliessen um 1 cm macht daraus ein Polygon. Die Flaeche bleibt die Summe der Teile.
            u = u.buffer(ROOM_CLOSING).buffer(-ROOM_CLOSING)
        poly = _as_polygon(u)
        if poly is None:
            multipoly += 1
            continue
        bd = poly.boundary
        cover = bd.intersection(wallish).length / bd.length if bd.length else 0.0
        if cover < ROOM_BOUNDARY_COVER:
            open_rooms += 1
        biggest = max(members, key=lambda m: room_items[m][1].area)
        sub = room_items[biggest][0]
        rp = poly.representative_point()
        rooms.append({
            "id": f"r{len(rooms)}",
            "sd_typen": sorted({room_items[m][0] for m in members}),
            "label": ROOM_LABELS_DE.get(sub, "Raum"),
            "flaeche_m2": round(sum_area, 4),
            "teile": len(members),
            "punkt": [_r3(rp.x), _r3(rp.y)],
            "schwerpunkt": [_r3(poly.centroid.x), _r3(poly.centroid.y)],
            "wandanteil": round(cover, 3),
            "polygon": _ring(poly),
            "loecher": [[[_r3(x), _r3(y)] for x, y in h.coords[:-1]] for h in poly.interiors],
        })
    diag["raeume"] = len(rooms)
    diag["raeume_aus_offenen_nachbarn"] = merged_cnt
    diag["raeume_nicht_wandbegrenzt"] = open_rooms
    diag["raeume_mehrteilig_verworfen"] = multipoly
    if not rooms:
        gr.append("keine_raeume")
    if open_rooms:
        gr.append("raum_nicht_wandbegrenzt")
    if multipoly:
        gr.append("raum_mehrteilig")

    nonroom = []
    for s, p in other_encl:
        pp = _as_polygon(p)
        if pp is not None:
            nonroom.append({"sd_typ": s, "flaeche_m2": round(pp.area, 4), "polygon": _ring(pp)})

    # --- Plan-Dict
    xs = [c for r in rects for c in r.poly.bounds[0::2]]
    ys = [c for r in rects for c in r.poly.bounds[1::2]]
    plan = {
        "plan_id": plan_id,
        "floor_id": floor_id,
        "bbox": [_r3(min(xs)), _r3(min(ys)), _r3(max(xs)), _r3(max(ys))],
        "drehung_grad": diag["drehung_grad"],
        "waende": [{k: v for k, v in w.items() if k != "members"} for w in walls],
        "rects": [
            {"poly": _ring(rects[i].poly), "T": _r3(rects[i].T), "aussen": bool(rect_ext[i]),
             "wand": rects[i].group}
            for i in draw_idx
        ],
        "oeffnungen": out_opens,
        "raeume": rooms,
        "nichtraeume": nonroom,
        "gelaender": [_ring(p) for p in rails],
        "stuetzen": [_ring(p) for p in cols],
        "ausstattung": [{"typ": s, "polygon": _ring(p)} for s, p in feats],
        "aussenbereiche": [{"sd_typ": s, "polygon": _ring(p)} for s, p in area_polys if s in OUTDOOR_TYPES],
    }
    n_doors = sum(1 for o in out_opens if o["art"] == "tuer")
    n_win = sum(1 for o in out_opens if o["art"] == "fenster")
    plan["gt"] = {
        "wandlaenge_achse_m": _r3(L_axis),
        "wandlaenge_flaechenkante_m": _r3(L_merged),
        "wandlaenge_summe_rechtecke_m": _r3(L_roh_all),
        "n_waende": len(walls),
        "tueren": n_doors,
        "tueren_davon_haustuer": sum(1 for o in out_opens if o["sd_typ"] == "ENTRANCE_DOOR"),
        "fenster": n_win,
        "raeume": len(rooms),
        "raumflaeche_summe_m2": round(sum(r["flaeche_m2"] for r in rooms), 3),
        "nichtraeume_eingeschlossen": len(nonroom),
    }
    sig = hashlib.sha1(
        repr((sorted((round(w["laenge_achse"], 2), round(w["T"], 2)) for w in plan["waende"]),
              n_doors, n_win, len(rooms))).encode()
    ).hexdigest()[:16]
    plan["signatur"] = sig
    res.plan = plan
    return res
