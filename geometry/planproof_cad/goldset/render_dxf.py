"""Rendert ein bereinigtes Swiss-Dwellings-Geschoss (sd_plan.analysiere_geschoss) als DXF.

Die Zeichenkonventionen sind SYNTHETISCH GEWAEHLT. Das Goldset beweist die Pipeline auf
Swiss-Dwellings-Geometrie mit selbst gewaehlten Layern, NICHT auf echten Buero-Exporten.

Konventionen (Layernamen siehe KONVENTIONEN):
  aia      AIA/NCS:           A-WALL, A-DOOR, A-GLAZ, A-AREA
  dach     DACH-Bueropraxis:  A_Wand_a/A_Wand_i, A_Tuer, A_Fenster, A_Raum
  sib      Sachsen-SIB-KG:    300_WAND, 300_TUEROEFFNUNGEN, 300_FENSTEROEFFNUNGEN, 0_RAUMPOLYGON
  archicad ArchiCAD-artig:    Stiftlayer (STIFT_23 ...), Tueren und Fenster als Bloecke auf Wandlayern

Wandausfuehrung: "doppel" (zwei Linien je Wandrechteck) oder "kontur" (geschlossene Polylinien
der Wandflaeche); mit oder ohne Unterbrechung der Wandlinien an Oeffnungen. Oeffnungen sind
INSERT-Bloecke, Raeume TEXT-Labels. Zusaetzlich Stoerelemente: Bemassung, Schraffur,
Beschriftung, Ausstattung, Gelaender, Stuetzen.

Das Ergebnis ist deterministisch: feste Metadaten, feste Zufallsquelle je (Plan, Variante),
Koordinaten auf ganze Millimeter gerundet, Zeilenende LF.
"""
from __future__ import annotations

import io
import math
import random
from typing import Any, Dict, List, Tuple

import ezdxf
import numpy as np
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import unary_union

ezdxf.options.write_fixed_meta_data_for_testing = True

MM = 1000.0

KONVENTIONEN: Dict[str, Dict[str, Any]] = {
    "aia": {
        "name": "AIA/NCS (US National CAD Standard)",
        "wand_aussen": "A-WALL", "wand_innen": "A-WALL",
        "tuer": "A-DOOR", "fenster": "A-GLAZ", "raum": "A-AREA", "raumpolygon": None,
        "bemassung": "A-ANNO-DIMS", "text": "A-ANNO-TEXT", "schraffur": "A-WALL-PATT",
        "ausstattung": "A-FURN", "stuetze": "A-COLS", "gelaender": "A-FLOR-HRAL",
        "treppe": "A-FLOR-STRS",
        "block_tuer": "DR_{w:04d}", "block_fenster": "WN_{w:04d}",
        "oeffnung_auf_wandlayer": False,
    },
    "dach": {
        "name": "DACH-Bueropraxis (oeffentliche Layerlisten)",
        "wand_aussen": "A_Wand_a", "wand_innen": "A_Wand_i",
        "tuer": "A_Tuer", "fenster": "A_Fenster", "raum": "A_Raum", "raumpolygon": None,
        "bemassung": "A_Bemassung", "text": "A_Text", "schraffur": "A_Wand_Schraffur",
        "ausstattung": "A_Moebel", "stuetze": "A_Stuetze", "gelaender": "A_Gelaender",
        "treppe": "A_Treppe",
        "block_tuer": "Tuer_{w:04d}", "block_fenster": "Fenster_{w:04d}",
        "oeffnung_auf_wandlayer": False,
    },
    "sib": {
        "name": "Sachsen-SIB Kostengruppen-Layer",
        "wand_aussen": "300_WAND", "wand_innen": "300_WAND",
        "tuer": "300_TUEROEFFNUNGEN", "fenster": "300_FENSTEROEFFNUNGEN",
        "raum": "0_RAUMPOLYGON", "raumpolygon": "0_RAUMPOLYGON",
        "bemassung": "0_BEMASSUNG", "text": "0_TEXT", "schraffur": "300_WAND_SCHRAFFUR",
        "ausstattung": "600_AUSSTATTUNG", "stuetze": "300_STUETZE", "gelaender": "300_GELAENDER",
        "treppe": "300_TREPPE",
        "block_tuer": "TUER_{w:04d}", "block_fenster": "FENSTER_{w:04d}",
        "oeffnung_auf_wandlayer": False,
    },
    "archicad": {
        "name": "ArchiCAD-artig (Stiftlayer, Oeffnungen als Bloecke auf Wandlayern)",
        "wand_aussen": "STIFT_18", "wand_innen": "STIFT_23",
        "tuer": None, "fenster": None, "raum": "STIFT_7", "raumpolygon": None,
        "bemassung": "STIFT_1", "text": "STIFT_2", "schraffur": "STIFT_5",
        "ausstattung": "STIFT_3", "stuetze": "STIFT_18", "gelaender": "STIFT_9",
        "treppe": "STIFT_9",
        "block_tuer": "AC_Tuer_{w:04d}", "block_fenster": "AC_Fenster_{w:04d}",
        "oeffnung_auf_wandlayer": True,
    },
}

# Acht Varianten, zwei je Konvention. Jeder Plan bekommt zwei Varianten verschiedener Konvention.
VARIANTEN: List[Dict[str, Any]] = [
    {"id": "aia_doppel", "konvention": "aia", "wand": "doppel", "unterbrechung": False, "schraffur": "flaeche"},
    {"id": "aia_kontur_unterbr", "konvention": "aia", "wand": "kontur", "unterbrechung": True, "schraffur": "linien"},
    {"id": "dach_doppel_unterbr", "konvention": "dach", "wand": "doppel", "unterbrechung": True, "schraffur": "flaeche"},
    {"id": "dach_kontur", "konvention": "dach", "wand": "kontur", "unterbrechung": False, "schraffur": "flaeche"},
    {"id": "sib_doppel", "konvention": "sib", "wand": "doppel", "unterbrechung": False, "schraffur": "flaeche"},
    {"id": "sib_kontur_unterbr", "konvention": "sib", "wand": "kontur", "unterbrechung": True, "schraffur": "flaeche"},
    {"id": "archicad_doppel_unterbr", "konvention": "archicad", "wand": "doppel", "unterbrechung": True, "schraffur": "flaeche"},
    {"id": "archicad_kontur", "konvention": "archicad", "wand": "kontur", "unterbrechung": False, "schraffur": "flaeche"},
]
VARIANTEN_BY_ID = {v["id"]: v for v in VARIANTEN}

HATCH_LINE_SPACING_M = 0.5
OPENING_CUT_EXTRA_M = 0.06


def varianten_fuer_index(i: int) -> List[Dict[str, Any]]:
    """Zwei Varianten verschiedener Konvention fuer den i-ten Plan (deterministisch)."""
    return [VARIANTEN[i % 8], VARIANTEN[(i + 3) % 8]]


def _mm(v: float) -> float:
    return float(round(v * MM))


def _pts(coords) -> List[Tuple[float, float]]:
    return [(_mm(x), _mm(y)) for x, y in coords]


def _iter_polys(g) -> List[Polygon]:
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


def _iter_lines(g) -> List[LineString]:
    if g is None or g.is_empty:
        return []
    if isinstance(g, LineString):
        return [g]
    if isinstance(g, MultiLineString):
        return list(g.geoms)
    return [x for x in getattr(g, "geoms", []) if isinstance(x, LineString)]


def _rect_corners(poly: Polygon) -> np.ndarray:
    mrr = poly.minimum_rotated_rectangle
    return np.array(mrr.exterior.coords[:4], dtype=float)


def _long_sides(poly: Polygon) -> List[Tuple[np.ndarray, np.ndarray]]:
    c = _rect_corners(poly)
    e1 = c[1] - c[0]
    e2 = c[2] - c[1]
    if np.hypot(*e1) >= np.hypot(*e2):
        return [(c[0], c[1]), (c[3], c[2])]
    return [(c[1], c[2]), (c[0], c[3])]


class _Counter:
    def __init__(self) -> None:
        self.c: Dict[str, int] = {}

    def add(self, k: str, n: int = 1) -> None:
        self.c[k] = self.c.get(k, 0) + n


def _define_door_block(doc, name: str, w_mm: float) -> None:
    if name in doc.blocks:
        return
    blk = doc.blocks.new(name=name)
    # Tuerblatt (Linie) und Schwingbogen: Ursprung = Bandseite, Breite entlang +x
    blk.add_line((0, 0), (w_mm, 0), dxfattribs={"layer": "0"})
    blk.add_arc(center=(0, 0), radius=w_mm, start_angle=0, end_angle=90, dxfattribs={"layer": "0"})


def _define_window_block(doc, name: str, w_mm: float, d_mm: float = 240.0) -> None:
    if name in doc.blocks:
        return
    blk = doc.blocks.new(name=name)
    h = d_mm / 2
    blk.add_lwpolyline([(0, -h), (w_mm, -h), (w_mm, h), (0, h)], close=True, dxfattribs={"layer": "0"})
    blk.add_line((0, 0), (w_mm, 0), dxfattribs={"layer": "0"})


def _add_layers(doc, names: List[str]) -> None:
    for i, n in enumerate(sorted(set(n for n in names if n))):
        if n not in doc.layers:
            doc.layers.add(n, color=1 + (i % 7))


def render_dxf(plan: Dict[str, Any], variante: Dict[str, Any], plan_key: str, seed: int = 20201020) -> Tuple[bytes, Dict[str, Any]]:
    """Gibt (DXF-Bytes, Zeichnungsinfo) zurueck."""
    kv = KONVENTIONEN[variante["konvention"]]
    rng = random.Random(f"{seed}:{plan_key}:{variante['id']}")
    cnt = _Counter()

    doc = ezdxf.new("R2000", setup=False)
    doc.units = ezdxf.units.MM
    doc.header["$MEASUREMENT"] = 1
    msp = doc.modelspace()

    L_w_ext, L_w_int = kv["wand_aussen"], kv["wand_innen"]
    L_door, L_win = kv["tuer"], kv["fenster"]
    layers = [L_w_ext, L_w_int, L_door, L_win, kv["raum"], kv["raumpolygon"], kv["bemassung"], kv["text"],
              kv["schraffur"], kv["ausstattung"], kv["stuetze"], kv["gelaender"], kv["treppe"]]
    _add_layers(doc, layers)

    # ------------------------------------------------------------ Waende
    rect_polys = [(Polygon(r["poly"]), bool(r["aussen"])) for r in plan["rects"]]
    # Schnittflaechen an den Oeffnungen (nur bei Unterbrechung)
    cuts: List[Polygon] = []
    host_walls = {i: w for i, w in enumerate(plan["waende"])}
    for o in plan["oeffnungen"]:
        u = np.array(o["richtung"])
        nrm = np.array([-u[1], u[0]])
        c = np.array(o["mitte"])
        w = o["breite"]
        host_t = host_walls[o["wand"]]["T"] if o["wand"] is not None else o["dicke"]
        t = max(o["dicke"], host_t) + OPENING_CUT_EXTRA_M
        cuts.append(Polygon([c - u * w / 2 - nrm * t / 2, c + u * w / 2 - nrm * t / 2,
                             c + u * w / 2 + nrm * t / 2, c - u * w / 2 + nrm * t / 2]))
    cut_union = unary_union(cuts) if (variante["unterbrechung"] and cuts) else None

    def wall_layer(aussen: bool) -> str:
        return L_w_ext if aussen else L_w_int

    union_all = unary_union([p for p, _ in rect_polys])
    hatch_geom = union_all
    if variante["wand"] == "doppel":
        for poly, aussen in rect_polys:
            for a, b in _long_sides(poly):
                seg = LineString([a, b])
                pieces = _iter_lines(seg.difference(cut_union)) if cut_union is not None else [seg]
                for pc in pieces:
                    if pc.length < 0.002:
                        continue
                    (x0, y0), (x1, y1) = pc.coords[0], pc.coords[-1]
                    msp.add_line((_mm(x0), _mm(y0)), (_mm(x1), _mm(y1)), dxfattribs={"layer": wall_layer(aussen)})
                    cnt.add("wand_linien")
        if cut_union is not None:
            # Laibungslinien quer zur Wand an beiden Oeffnungsenden
            for o in plan["oeffnungen"]:
                if o["wand"] is None:
                    continue
                u = np.array(o["richtung"])
                nrm = np.array([-u[1], u[0]])
                c = np.array(o["mitte"])
                w = o["breite"]
                t = host_walls[o["wand"]]["T"]
                aussen = host_walls[o["wand"]].get("aussen", False)
                for s in (-1.0, 1.0):
                    p0 = c + u * s * w / 2 - nrm * t / 2
                    p1 = c + u * s * w / 2 + nrm * t / 2
                    msp.add_line((_mm(p0[0]), _mm(p0[1])), (_mm(p1[0]), _mm(p1[1])),
                                 dxfattribs={"layer": wall_layer(aussen)})
                    cnt.add("wand_laibung")
            hatch_geom = union_all.difference(cut_union)
    else:
        ext = unary_union([p for p, a in rect_polys if a]) if any(a for _, a in rect_polys) else None
        inn = unary_union([p for p, a in rect_polys if not a]) if any(not a for _, a in rect_polys) else None
        for geom, layer in ((ext, wall_layer(True)), (inn, wall_layer(False))):
            if geom is None:
                continue
            if cut_union is not None:
                geom = geom.difference(cut_union)
            for poly in _iter_polys(geom):
                for ring in [poly.exterior] + list(poly.interiors):
                    ring = ring.simplify(0.0005)
                    pts = _pts(ring.coords[:-1])
                    if len(pts) < 3:
                        continue
                    msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer})
                    cnt.add("wand_konturen")
        if cut_union is not None:
            hatch_geom = union_all.difference(cut_union)

    # ------------------------------------------------------------ Schraffur
    if variante["schraffur"] == "flaeche":
        for poly in _iter_polys(hatch_geom):
            if poly.area < 0.01:
                continue
            h = msp.add_hatch(color=8, dxfattribs={"layer": kv["schraffur"]})
            h.set_pattern_fill("ANSI31", scale=25.0)
            h.paths.add_polyline_path(_pts(poly.exterior.coords[:-1]), is_closed=True, flags=1)
            for hole in poly.interiors:
                h.paths.add_polyline_path(_pts(hole.coords[:-1]), is_closed=True, flags=0)
            cnt.add("schraffur_hatch")
    elif variante["schraffur"] == "linien":
        # aufgeloeste Schraffur (wie nach einem Export): 45-Grad-Linien nur in den Aussenwaenden
        ext_geom = unary_union([p for p, a in rect_polys if a]) if any(a for _, a in rect_polys) else None
        if ext_geom is not None:
            if cut_union is not None:
                ext_geom = ext_geom.difference(cut_union)
            minx, miny, maxx, maxy = ext_geom.bounds
            span = (maxx - minx) + (maxy - miny)
            k = 0.0
            while k <= span:
                ln = LineString([(minx + k, miny), (minx + k - (maxy - miny), maxy)])
                for pc in _iter_lines(ln.intersection(ext_geom)):
                    if pc.length < 0.01:
                        continue
                    (x0, y0), (x1, y1) = pc.coords[0], pc.coords[-1]
                    msp.add_line((_mm(x0), _mm(y0)), (_mm(x1), _mm(y1)), dxfattribs={"layer": kv["schraffur"]})
                    cnt.add("schraffur_linien")
                k += HATCH_LINE_SPACING_M

    # ------------------------------------------------------------ Oeffnungen als Bloecke
    for o in plan["oeffnungen"]:
        u = np.array(o["richtung"], dtype=float)
        ang = math.degrees(math.atan2(u[1], u[0]))
        c = np.array(o["mitte"], dtype=float)
        w_mm = max(300.0, round(o["breite"] * MM / 25.0) * 25.0)
        host = host_walls[o["wand"]] if o["wand"] is not None else None
        aussen = bool(host.get("aussen")) if host else False
        if o["art"] == "tuer":
            name = kv["block_tuer"].format(w=int(w_mm))
            _define_door_block(doc, name, w_mm)
            layer = wall_layer(aussen) if kv["oeffnung_auf_wandlayer"] else L_door
            hinge_end = rng.random() < 0.5
            swing = 1.0 if rng.random() < 0.5 else -1.0
            if hinge_end:
                p = c + u * (w_mm / MM) / 2
                rot = ang + 180.0
                # bei 180 Grad Drehung spiegelt die Schwingrichtung mit; yscale gleicht das aus
                ys = -swing
            else:
                p = c - u * (w_mm / MM) / 2
                rot = ang
                ys = swing
            msp.add_blockref(name, (_mm(p[0]), _mm(p[1])),
                             dxfattribs={"layer": layer, "rotation": round(rot, 3), "yscale": ys})
            cnt.add("insert_tuer")
        else:
            name = kv["block_fenster"].format(w=int(w_mm))
            _define_window_block(doc, name, w_mm)
            layer = wall_layer(aussen) if kv["oeffnung_auf_wandlayer"] else L_win
            p = c - u * (w_mm / MM) / 2
            msp.add_blockref(name, (_mm(p[0]), _mm(p[1])),
                             dxfattribs={"layer": layer, "rotation": round(ang, 3)})
            cnt.add("insert_fenster")

    # ------------------------------------------------------------ Raeume
    for r in plan["raeume"]:
        t = msp.add_text(r["label"], height=250.0, dxfattribs={"layer": kv["raum"]})
        t.set_placement((_mm(r["punkt"][0]), _mm(r["punkt"][1])), align=TextEntityAlignment.MIDDLE_CENTER)
        cnt.add("raum_text")
        if kv["raumpolygon"]:
            msp.add_lwpolyline(_pts(r["polygon"]), close=True, dxfattribs={"layer": kv["raumpolygon"]})
            for hole in r.get("loecher", []):
                msp.add_lwpolyline(_pts(hole), close=True, dxfattribs={"layer": kv["raumpolygon"]})
            cnt.add("raum_polygon")

    # ------------------------------------------------------------ Stoerobjekte
    for g in plan["gelaender"]:
        msp.add_lwpolyline(_pts(g), close=True, dxfattribs={"layer": kv["gelaender"]})
        cnt.add("gelaender")
    for g in plan["stuetzen"]:
        msp.add_lwpolyline(_pts(g), close=True, dxfattribs={"layer": kv["stuetze"]})
        cnt.add("stuetzen")
    for f in plan["ausstattung"]:
        layer = kv["treppe"] if f["typ"] in ("STAIRS", "ELEVATOR", "RAMP") else kv["ausstattung"]
        msp.add_lwpolyline(_pts(f["polygon"]), close=True, dxfattribs={"layer": layer})
        cnt.add("ausstattung")

    # ------------------------------------------------------------ Bemassung (ausgeloest) + Beschriftung
    x0, y0, x1, y1 = plan["bbox"]
    off = 1.2
    dims = [
        ((x0, y1 + off), (x1, y1 + off), (x0, y1), (x1, y1), "h"),
        ((x0, y0 - off), (x1, y0 - off), (x0, y0), (x1, y0), "h"),
        ((x0 - off, y0), (x0 - off, y1), (x0, y0), (x0, y1), "v"),
        ((x1 + off, y0), (x1 + off, y1), (x1, y0), (x1, y1), "v"),
    ]
    for a, b, ea, eb, orient in dims:
        msp.add_line((_mm(a[0]), _mm(a[1])), (_mm(b[0]), _mm(b[1])), dxfattribs={"layer": kv["bemassung"]})
        for e_from, e_to in ((ea, a), (eb, b)):
            d = np.array(e_to) - np.array(e_from)
            n = np.hypot(*d)
            d2 = d / n if n else d
            e_end = np.array(e_to) + d2 * 0.1
            msp.add_line((_mm(e_from[0]), _mm(e_from[1])), (_mm(e_end[0]), _mm(e_end[1])),
                         dxfattribs={"layer": kv["bemassung"]})
            tick = np.array([0.1, 0.1]) if orient == "h" else np.array([0.1, 0.1])
            msp.add_line((_mm(e_to[0] - tick[0]), _mm(e_to[1] - tick[1])),
                         (_mm(e_to[0] + tick[0]), _mm(e_to[1] + tick[1])), dxfattribs={"layer": kv["bemassung"]})
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        tx = msp.add_text(str(int(round(length * MM))), height=200.0, dxfattribs={"layer": kv["bemassung"]})
        tx.dxf.rotation = 0.0 if orient == "h" else 90.0
        tx.set_placement((_mm(mid[0]), _mm(mid[1]) + (60.0 if orient == "h" else 0.0)),
                         align=TextEntityAlignment.BOTTOM_CENTER)
        cnt.add("bemassung")
    title = [
        "Grundriss M 1:100",
        f"Plan {plan_key} - abgeleitet aus Swiss Dwellings v3.0.0 (CC BY 4.0)",
        "Masse in mm, Zeichenkonvention synthetisch",
    ]
    for i, line in enumerate(title):
        tx = msp.add_text(line, height=180.0, dxfattribs={"layer": kv["text"]})
        tx.set_placement((_mm(x0), _mm(y0 - off - 1.0 - 0.35 * i)), align=TextEntityAlignment.LEFT)
        cnt.add("beschriftung")

    doc.header["$EXTMIN"] = (_mm(x0 - off - 1), _mm(y0 - off - 2.5), 0.0)
    doc.header["$EXTMAX"] = (_mm(x1 + off + 1), _mm(y1 + off + 1), 0.0)

    buf = io.StringIO()
    doc.write(buf)
    # DXF R2000 ist cp1252 ($DWGCODEPAGE ANSI_1252); Zeichen ausserhalb werden als \U+XXXX geschrieben
    data = buf.getvalue().replace("\r\n", "\n").encode(doc.output_encoding, errors="dxfreplace")
    info = {
        "entitaeten": dict(sorted(cnt.c.items())),
        "layer": sorted(set(n for n in layers if n)),
        "bloecke": sorted(b.name for b in doc.blocks if not b.name.startswith("*")),
    }
    return data, info
