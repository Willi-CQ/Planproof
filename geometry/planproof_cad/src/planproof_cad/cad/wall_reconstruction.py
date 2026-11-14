"""Wand-Rekonstruktion aus Wand-Layer-Linien.

Zwei Zeichenstile werden unterstuetzt, **je zusammenhaengender Linienmenge** erkannt:

  * **Doppellinien-Waende** (der Regelfall): jede Wand ist ein Paar paralleler Linien, auch als
    Kante einer Kontur (Rechteck-Polylinie, Vereinigung von Waenden). Aus dem Paar werden Achse
    (Mittellinie) und Dicke (Abstand) gewonnen -- die klassische ISARC-2022-Methode.
  * **Einzellinien-Waende** (schematisch / Wandobjekt-Export): Linien sind bereits Achsen; es wird
    eine Default-Dicke angesetzt.

Ablauf (Doppellinien):

  1. Paarsuche ohne Verbrauch: jede Linie kann mit mehreren Gegenlinien Streifen bilden (lange Linie
     gegenueber mehreren kurzen Linien). Ein Streifen ist ein Achsenstueck ueber der Ueberdeckung.
  2. Streifen mit gleicher Achse, Richtung und Dicke werden zu einer Achse vereinigt.
  3. Luecken in einer Achse werden geschlossen: wo eine durchgehende Wandkante die Luecke
     ueberspannt (T-Stoss), wo ein Tuer-/Fensterblock sie ueberdeckt, sonst bis zu einer
     wanddickenabhaengigen Toleranz.
  4. Achsenenden werden an Ecken und T-Stoessen auf den Schnittpunkt der Achsen gezogen (oder dort
     gekuerzt). Ohne diesen Schritt enden die Achsen um die halbe Wanddicke vor der Ecke.

Ergebnis ist eine Liste von ``Centerline``-Achsen (Meter, mit Dicke, Wandtyp und ``bearing``),
die ``build_wall_graph`` zu einem topologisch sauberen ``WallGraph`` verdichtet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from planproof_cad.model.wall_graph import WallGraph

from . import layer_classifier as lc
from .dxf_loader import RawSegment

Point2 = Tuple[float, float]

# Standardwerte (alle ueber ``config`` ueberschreibbar)
MIN_THICKNESS_M = 0.05
MAX_THICKNESS_M = 1.00
ANGLE_TOL_DEG = 6.0
MIN_OVERLAP_ABS_M = 0.20
OVERLAP_FRAC = 0.5
MIN_SEGMENT_M = 0.10
GAP_FACTOR = (
    2.0  # ohne Oeffnungsblock werden Luecken bis GAP_FACTOR * Dicke + GAP_SLACK_M geschlossen
)
GAP_SLACK_M = 0.05
OPENING_GAP_MAX_M = 8.0
JUNCTION_SLACK_M = 0.10
CROSS_SIDE_MIN_D = 0.40  # Abschattung ueber die Gegenseite nur fuer Kandidaten ab dieser Dicke ...
CROSS_SIDE_RATIO = 0.5  # ... wenn der Partner der anderen Seite hoechstens halb so dick ist
COMPONENT_COVERAGE_MIN = 0.25  # darunter gilt eine Linienmenge als einlinig gezeichnet
SINGLE_NOISE_MAX_M = (
    2.0  # lose Einzellinien (weniger als 3 Linien) bis zu dieser Laenge sind Rauschen
)


@dataclass
class Centerline:
    a: Point2
    b: Point2
    thickness_m: float
    wall_type: str  # "exterior" | "interior"
    source: str = "dwg_double_line"
    bearing: Optional[bool] = None  # tragend / nichttragend laut Layername, sonst None
    layer: str = ""
    ref_segment: int = -1  # Index einer Kantenlinie (fuer die Zuordnung zur Linienmenge), intern

    def length(self) -> float:
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])


# ── Vektor-Helfer ────────────────────────────────────────────────────────


def _np(p: Point2) -> np.ndarray:
    return np.array([p[0], p[1]], dtype=float)


def _seg_dir(a: Point2, b: Point2) -> Tuple[np.ndarray, float]:
    v = _np(b) - _np(a)
    L = float(np.hypot(v[0], v[1]))
    if L < 1e-9:
        return np.array([1.0, 0.0]), 0.0
    return v / L, L


def _union_length(intervals: Sequence[Tuple[float, float]], lo: float, hi: float) -> float:
    """Laenge der Vereinigung von Intervallen, auf [lo, hi] beschnitten."""
    iv = sorted((max(lo, a), min(hi, b)) for a, b in intervals if min(hi, b) > max(lo, a))
    total = 0.0
    cur_a = cur_b = None
    for a, b in iv:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total


@dataclass
class _Prep:
    """Vorbereitete Linien als Arrays (nur Linien ab Mindestlaenge)."""

    segs: List[RawSegment]
    A: np.ndarray
    B: np.ndarray
    U: np.ndarray
    L: np.ndarray
    ANG: np.ndarray  # Richtungswinkel in [0, pi)
    dropped_short: int = 0

    def __len__(self) -> int:
        return len(self.segs)


def _prepare(wall_segments: Sequence[RawSegment], min_len: float) -> _Prep:
    keep: List[RawSegment] = []
    dropped = 0
    for s in wall_segments:
        if s.length() < min_len:
            dropped += 1
            continue
        keep.append(s)
    if not keep:
        z2 = np.zeros((0, 2))
        return _Prep([], z2, z2, z2, np.zeros(0), np.zeros(0), dropped)
    A = np.array([s.a for s in keep], dtype=float)
    B = np.array([s.b for s in keep], dtype=float)
    D = B - A
    L = np.hypot(D[:, 0], D[:, 1])
    U = D / L[:, None]
    ang = np.arctan2(U[:, 1], U[:, 0]) % math.pi
    return _Prep(keep, A, B, U, L, ang, dropped)


# ── Doppellinien-Streifen ────────────────────────────────────────────────


@dataclass
class _Strip:
    a: np.ndarray  # Achsenanfang (Meter)
    b: np.ndarray
    u: np.ndarray  # Einheitsrichtung
    ang: float
    thick: float
    i: int  # Index der Bezugslinie
    j: int  # Index der Gegenlinie


@dataclass
class _Cand:
    """Kandidat fuer einen Streifen zwischen den Linien i und j (vor dem Abschatten)."""

    i: int
    j: int
    side_i: int  # auf welcher Seite von i liegt j (Vorzeichen)
    side_j: int  # auf welcher Seite von j liegt i
    d: float  # Dicke (Abstand)
    off: float  # halber Abstand mit Vorzeichen, in i-Normalenrichtung
    lo: float  # Ueberdeckung in Parametern von i
    hi: float
    s0: float  # dieselben Endpunkte in Parametern von j
    s1: float


def _shadowed(cands: List[_Cand], shadow_eps: float) -> Dict[int, List[Tuple[float, float]]]:
    """Je Kandidat die Teile seiner Ueberdeckung (i-Parameter), die ein naeherer Partner besetzt.

    Eine Wandkante hat je Seite einen Partner: die naechste Gegenkante. Weiter entfernte Kanten
    derselben Seite sind andere Waende oder gestapelte Linien (Pfeiler, Schaechte), keine
    Gegenseite. Auf der anderen Seite der Kante gilt das nur fuer deutlich dickere Kandidaten
    (Flur zwischen zwei duennen Waenden: 0,9 m Abstand gegen 0,12 m Wanddicke); ein Kandidat
    gleicher Groessenordnung auf der Gegenseite ist meist eine zweite Wand, die an derselben Linie
    anliegt.
    """
    by_face: Dict[int, List[int]] = {}
    for n, c in enumerate(cands):
        by_face.setdefault(c.i, []).append(n)
        by_face.setdefault(c.j, []).append(n)

    def in_i_frame(c: _Cand, face: int, a: float, b: float) -> Tuple[float, float]:
        if face == c.i:
            return a, b
        span = c.s1 - c.s0
        if abs(span) < 1e-9:
            return c.lo, c.hi
        t0 = c.lo + (a - c.s0) * (c.hi - c.lo) / span
        t1 = c.lo + (b - c.s0) * (c.hi - c.lo) / span
        return (min(t0, t1), max(t0, t1))

    shadows: Dict[int, List[Tuple[float, float]]] = {n: [] for n in range(len(cands))}
    for face, members in by_face.items():
        if len(members) < 2:
            continue
        for n in members:
            c = cands[n]
            side_n = c.side_i if face == c.i else c.side_j
            for m in members:
                o = cands[m]
                if m == n:
                    continue
                side_m = o.side_i if face == o.i else o.side_j
                if side_m == side_n:
                    if o.d >= c.d - shadow_eps:
                        continue
                elif not (c.d > CROSS_SIDE_MIN_D and o.d <= CROSS_SIDE_RATIO * c.d):
                    continue
                # Intervall des naeheren Partners auf dieser gemeinsamen Linie
                if face == o.i:
                    oa, ob = o.lo, o.hi
                else:
                    oa, ob = min(o.s0, o.s1), max(o.s0, o.s1)
                # in die Parameter der Linie 'face' innerhalb von c bringen, dann in den i-Rahmen
                # von c
                if face == c.i:
                    shadows[n].append((oa, ob))
                else:
                    shadows[n].append(in_i_frame(c, face, oa, ob))
    return shadows


def _free_pieces(
    lo: float, hi: float, shadow: List[Tuple[float, float]], eps: float
) -> List[Tuple[float, float]]:
    """[lo, hi] ohne die (um eps verkleinerten) Schatten-Intervalle."""
    cuts = sorted(
        (max(lo, a + eps), min(hi, b - eps))
        for a, b in shadow
        if min(hi, b - eps) > max(lo, a + eps)
    )
    out: List[Tuple[float, float]] = []
    pos = lo
    for a, b in cuts:
        if a > pos:
            out.append((pos, a))
        pos = max(pos, b)
    if hi > pos:
        out.append((pos, hi))
    return out


def _find_strips(
    prep: _Prep,
    *,
    tmin: float,
    tmax: float,
    ang_tol: float,
    min_overlap: float,
    overlap_frac: float,
    nearest: bool = True,
) -> Tuple[List[_Strip], np.ndarray]:
    """Alle Streifen zwischen parallelen Linien, ohne eine Linie zu verbrauchen.

    Eine lange Linie bildet mit mehreren kurzen Gegenlinien je einen Streifen. Je Seite zaehlt
    nur der naechste Partner (siehe ``_shadowed``). Rueckgabe: (Streifen, Anteil jeder Linie, der
    von mindestens einem Streifen ueberdeckt ist).
    """
    n = len(prep)
    strips: List[_Strip] = []
    cover: List[List[Tuple[float, float]]] = [[] for _ in range(n)]
    if n < 2:
        return strips, np.zeros(n)
    A, B, U, L, ANG = prep.A, prep.B, prep.U, prep.L, prep.ANG
    cands: List[_Cand] = []
    for i in range(n - 1):
        idx = np.arange(i + 1, n)
        dang = np.abs(ANG[idx] - ANG[i])
        dang = np.minimum(dang, math.pi - dang)
        idx = idx[dang <= ang_tol]
        if idx.size == 0:
            continue
        ui = U[i]
        ni = np.array([-ui[1], ui[0]])
        ai = A[i]
        da = A[idx] - ai
        db = B[idx] - ai
        pa = da @ ni
        pb = db @ ni
        same_side = pa * pb > 0
        aa, ab = np.abs(pa), np.abs(pb)
        thick = 0.5 * (aa + ab)
        ok = same_side & (np.abs(aa - ab) <= np.maximum(0.04, 0.25 * np.maximum(aa, ab)))
        ok &= (thick >= tmin) & (thick <= tmax)
        if not ok.any():
            continue
        ta = da @ ui
        tb = db @ ui
        jlo = np.minimum(ta, tb)
        jhi = np.maximum(ta, tb)
        lo = np.maximum(0.0, jlo)
        hi = np.minimum(L[i], jhi)
        ov = hi - lo
        need = np.maximum(min_overlap, 0.9 * thick)
        ok &= (ov >= need) & (ov >= overlap_frac * np.minimum(L[i], L[idx]))
        for k in np.nonzero(ok)[0]:
            j = int(idx[k])
            off = 0.25 * (pa[k] + pb[k])  # halber Abstand, mit Vorzeichen
            p0 = ai + ni * off
            sa = p0 + ui * lo[k]
            sb = p0 + ui * hi[k]
            uj = U[j]
            nj = np.array([-uj[1], uj[0]])
            cands.append(
                _Cand(
                    i=i,
                    j=j,
                    side_i=1 if pa[k] > 0 else -1,
                    side_j=1 if float((ai - A[j]) @ nj) > 0 else -1,
                    d=float(thick[k]),
                    off=float(off),
                    lo=float(lo[k]),
                    hi=float(hi[k]),
                    s0=float((sa - A[j]) @ uj),
                    s1=float((sb - A[j]) @ uj),
                )
            )

    shadows = _shadowed(cands, shadow_eps=0.015) if nearest else {n: [] for n in range(len(cands))}
    for m, c in enumerate(cands):
        ui = U[c.i]
        ni = np.array([-ui[1], ui[0]])
        p0 = A[c.i] + ni * c.off
        span = c.hi - c.lo
        for a, b in _free_pieces(c.lo, c.hi, shadows[m], eps=0.02):
            need = max(min_overlap, 0.9 * c.d)
            if b - a < need or (b - a) < overlap_frac * min(span, float(min(L[c.i], L[c.j]))):
                continue
            sa, sb = p0 + ui * a, p0 + ui * b
            strips.append(_Strip(sa, sb, ui, float(ANG[c.i]), c.d, c.i, c.j))
            cover[c.i].append((a, b))
            uj = U[c.j]
            s0 = float((sa - A[c.j]) @ uj)
            s1 = float((sb - A[c.j]) @ uj)
            cover[c.j].append((min(s0, s1), max(s0, s1)))
    covered = np.array(
        [_union_length(cover[k], 0.0, float(L[k])) / float(L[k]) for k in range(n)], dtype=float
    )
    return strips, covered


# ── Achsen aus Streifen: Clustern, Vereinigen, Luecken schliessen ────────


@dataclass
class _Cluster:
    p0: np.ndarray
    u: np.ndarray
    n: np.ndarray
    ang: float
    thick: float
    strips: List[_Strip] = field(default_factory=list)


def _cluster_strips(strips: List[_Strip], *, ang_tol: float, off_abs: float) -> List[_Cluster]:
    """Gruppiert Streifen mit gleicher Richtung, Achse und Dicke (laengste zuerst)."""
    order = sorted(strips, key=lambda s: -float(np.hypot(*(s.b - s.a))))
    clusters: List[_Cluster] = []
    cp = np.zeros((0, 2))
    cn = np.zeros((0, 2))
    cang = np.zeros(0)
    cth = np.zeros(0)
    for s in order:
        best = -1
        if clusters:
            dang = np.abs(cang - s.ang)
            dang = np.minimum(dang, math.pi - dang)
            pa = np.abs(((s.a - cp) * cn).sum(axis=1))
            pb = np.abs(((s.b - cp) * cn).sum(axis=1))
            tol_off = np.maximum(off_abs, 0.25 * cth)
            tol_th = np.maximum(0.03, 0.12 * cth)
            ok = (
                (dang <= ang_tol)
                & (pa <= tol_off)
                & (pb <= tol_off)
                & (np.abs(cth - s.thick) <= tol_th)
            )
            if ok.any():
                score = np.where(ok, pa + pb + np.abs(cth - s.thick), np.inf)
                best = int(np.argmin(score))
        if best < 0:
            n = np.array([-s.u[1], s.u[0]])
            clusters.append(_Cluster(s.a.copy(), s.u.copy(), n, s.ang, s.thick, [s]))
            cp = np.vstack([cp, s.a])
            cn = np.vstack([cn, n])
            cang = np.append(cang, s.ang)
            cth = np.append(cth, s.thick)
        else:
            clusters[best].strips.append(s)
    return clusters


def _boxes_union(opening_boxes: Optional[Sequence[Sequence[Point2]]]):
    if not opening_boxes:
        return None
    try:
        from shapely.geometry import Polygon  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    polys = []
    for box in opening_boxes:
        if box and len(box) >= 3:
            p = Polygon(box)
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty:
                polys.append(p)
    return polys or None


def _gap_covered_by_opening(boxes, p_from: np.ndarray, p_to: np.ndarray, thick: float) -> bool:
    if not boxes:
        return False
    from shapely.geometry import LineString  # noqa: PLC0415

    line = LineString([tuple(p_from), tuple(p_to)])
    gap = line.length
    if gap <= 1e-9:
        return True
    reach = thick / 2.0 + 0.05
    covered = 0.0
    for poly in boxes:
        if poly.distance(line) > reach:
            continue
        covered += line.intersection(poly.buffer(reach)).length
    return covered >= 0.7 * gap


def _cluster_to_centerlines(
    cl: _Cluster,
    prep: _Prep,
    boxes,
    *,
    gap_factor: float,
    stats: Dict[str, Any],
    source: str,
    layer_info: Dict[str, "lc.Classification"],
    default_thickness: Optional[float] = None,
    allow_bridge: bool = True,
) -> List[Centerline]:
    u, p0 = cl.u, cl.p0
    # Intervalle entlang der Cluster-Achse
    items: List[Tuple[float, float, _Strip]] = []
    for s in cl.strips:
        t0 = float((s.a - p0) @ u)
        t1 = float((s.b - p0) @ u)
        items.append((min(t0, t1), max(t0, t1), s))
    items.sort(key=lambda x: x[0])
    merged: List[List[Any]] = []  # [lo, hi, [strips]]
    for lo, hi, s in items:
        if merged and lo <= merged[-1][1] + 0.02:
            merged[-1][1] = max(merged[-1][1], hi)
            merged[-1][2].append(s)
        else:
            merged.append([lo, hi, [s]])

    # Gesamtdicke: laengengewichteter Mittelwert
    wsum = sum(float(np.hypot(*(s.b - s.a))) for s in cl.strips) or 1.0
    thick = sum(s.thick * float(np.hypot(*(s.b - s.a))) for s in cl.strips) / wsum
    if default_thickness is not None:
        thick = default_thickness

    # Flaechen-Linien (Index -> Projektionsintervall) fuer die Kontinuitaetspruefung
    face_ids = {s.i for s in cl.strips} | {s.j for s in cl.strips}
    face_iv: List[Tuple[float, float]] = []
    for k in face_ids:
        ta = float((prep.A[k] - p0) @ u)
        tb = float((prep.B[k] - p0) @ u)
        face_iv.append((min(ta, tb), max(ta, tb)))
    gap_tol = gap_factor * thick + GAP_SLACK_M

    joined: List[List[Any]] = [merged[0]] if merged else []
    for cur in merged[1:]:
        last = joined[-1]
        gap = cur[0] - last[1]
        bridge = False
        if not allow_bridge:
            pass
        elif gap <= gap_tol:
            bridge = True
        elif any(f0 <= last[1] + 0.03 and f1 >= cur[0] - 0.03 for f0, f1 in face_iv):
            bridge = True  # eine Wandkante laeuft durch: Wand ist nicht unterbrochen
        elif gap <= OPENING_GAP_MAX_M and _gap_covered_by_opening(
            boxes, p0 + u * last[1], p0 + u * cur[0], thick
        ):
            bridge = True
        if bridge:
            last[1] = max(last[1], cur[1])
            last[2].extend(cur[2])
            if gap > 0.02:
                stats["bridged_gaps"] += 1
        else:
            joined.append(cur)

    out: List[Centerline] = []
    for lo, hi, members in joined:
        wall_type, bearing, layer = _wall_attributes(members, prep, layer_info)
        out.append(
            Centerline(
                a=(float(p0[0] + u[0] * lo), float(p0[1] + u[1] * lo)),
                b=(float(p0[0] + u[0] * hi), float(p0[1] + u[1] * hi)),
                thickness_m=round(float(thick), 3),
                wall_type=wall_type,
                source=source,
                bearing=bearing,
                layer=layer,
                ref_segment=int(members[0].i),
            )
        )
    return out


def _classify_cached(layer: str, cache: Dict[str, "lc.Classification"]) -> "lc.Classification":
    c = cache.get(layer)
    if c is None:
        c = lc.classify_layer(layer)
        cache[layer] = c
    return c


def _wall_attributes(
    members: Sequence[_Strip], prep: _Prep, cache: Dict[str, "lc.Classification"]
) -> Tuple[str, Optional[bool], str]:
    """Wandtyp, bearing und Hauptlayer aus den Layern der Kantenlinien."""
    layers: Dict[str, float] = {}
    for s in members:
        for k in (s.i, s.j):
            layers[prep.segs[k].layer] = layers.get(prep.segs[k].layer, 0.0) + float(prep.L[k])
    return _attributes_of_layers(layers, cache)


def _attributes_of_layers(
    layers: Dict[str, float], cache: Dict[str, "lc.Classification"]
) -> Tuple[str, Optional[bool], str]:
    exterior = interior = False
    bearings = set()
    for name in layers:
        c = _classify_cached(name, cache)
        if c.cls == lc.WALL_EXTERIOR:
            exterior = True
        elif c.cls == lc.WALL_INTERIOR:
            interior = True
        if c.bearing is not None:
            bearings.add(c.bearing)
    wall_type = "exterior" if exterior and not interior else "interior"
    if exterior and interior:
        wall_type = "exterior"
    bearing = next(iter(bearings)) if len(bearings) == 1 else None
    main = max(layers, key=lambda k: layers[k]) if layers else ""
    return wall_type, bearing, main


# ── Zusammenhaengende Linienmengen ───────────────────────────────────────


def _components(prep: _Prep, prox: float, boxes=None) -> np.ndarray:
    """Nummer der Linienmenge je Linie.

    Linien, die naeher als ``prox`` beieinander liegen, gehoeren zusammen; ebenso Linien, die am
    selben Tuer-/Fensterblock enden (sonst zerfaellt eine Wand an jeder Oeffnung in eigene Mengen).
    """
    n = len(prep)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    if n == 0:
        return np.zeros(0, dtype=int)
    try:
        from shapely import STRtree  # noqa: PLC0415
        from shapely.geometry import LineString  # noqa: PLC0415

        lines = [LineString([tuple(a), tuple(b)]) for a, b in zip(prep.A, prep.B, strict=True)]
        tree = STRtree(lines)
        for i, ln in enumerate(lines):
            for j in tree.query(ln.buffer(prox)):
                j = int(j)
                if j > i and lines[j].distance(ln) <= prox:
                    union(i, j)
        for poly in boxes or []:
            near = [
                int(j)
                for j in tree.query(poly.buffer(prox))
                if lines[int(j)].distance(poly) <= prox
            ]
            for j in near[1:]:
                union(near[0], j)
    except Exception:  # noqa: BLE001
        return np.zeros(n, dtype=int)
    roots: Dict[int, int] = {}
    comp = np.zeros(n, dtype=int)
    for i in range(n):
        comp[i] = roots.setdefault(find(i), len(roots))
    return comp


def _sub_prep(prep: _Prep, idx: np.ndarray) -> _Prep:
    return _Prep(
        [prep.segs[k] for k in idx],
        prep.A[idx],
        prep.B[idx],
        prep.U[idx],
        prep.L[idx],
        prep.ANG[idx],
    )


# ── Auswahl der Wand-Linien ──────────────────────────────────────────────


def select_wall_segments(
    segments: Sequence[RawSegment], config: Optional[dict] = None
) -> Tuple[List[RawSegment], Dict[str, Any]]:
    """Waehlt die Wand-Linien aus allen Linien.

    Reihenfolge: (1) ``config['wall_layers']``, (2) Layer, die der Klassifikator als Wand
    erkennt, (3) Layer unbekannter Klasse (Stiftlayer wie STIFT_23) mit Doppellinien-Geometrie,
    (4) alle Linien unbekannter Layer. Beschriftungs-, Tuer-, Fenster- und Raumlayer sind nie
    Wandlinien.
    """
    config = config or {}
    info: Dict[str, Any] = {"source": "none", "layers": [], "notes": []}
    by_layer: Dict[str, List[RawSegment]] = {}
    for s in segments:
        by_layer.setdefault(s.layer, []).append(s)
    cache: Dict[str, lc.Classification] = {}

    forced = config.get("wall_layers")
    if forced:
        want = {str(x).casefold() for x in forced}
        chosen = [n for n in by_layer if n.casefold() in want]
        info.update(source="config", layers=sorted(chosen))
        return [s for n in chosen for s in by_layer[n]], info

    named = [n for n in by_layer if _classify_cached(n, cache).cls in lc.WALL_CLASSES]
    if named:
        info.update(source="layer_name", layers=sorted(named))
        return [s for n in named for s in by_layer[n]], info

    unknown = [n for n in by_layer if _classify_cached(n, cache).cls == lc.UNKNOWN]
    if not unknown:
        info["notes"].append(
            "alle Linien liegen auf Beschriftungs-, Bemaßungs-, Tür-, Fenster- oder Raumlayern"
        )
        return [], info

    # Geometrie-Abstimmung: welche unbekannten Layer enthalten lange Doppellinien-Achsen?
    tmin = float(config.get("wall_min_thickness_m", MIN_THICKNESS_M))
    tmax = float(config.get("wall_max_thickness_m", MAX_THICKNESS_M))
    scores: Dict[str, float] = {}
    for name in unknown if config.get("wall_layer_vote", True) else []:
        segs = by_layer[name]
        if len(segs) < 4:
            continue
        prep = _prepare(segs, float(config.get("wall_min_length_m", MIN_SEGMENT_M)))
        strips, _cov = _find_strips(
            prep,
            tmin=tmin,
            tmax=tmax,
            ang_tol=math.radians(float(config.get("wall_angle_tol_deg", ANGLE_TOL_DEG))),
            min_overlap=MIN_OVERLAP_ABS_M,
            overlap_frac=float(config.get("wall_overlap_frac", OVERLAP_FRAC)),
        )
        if not strips:
            continue
        tmp_stats = {"bridged_gaps": 0}
        total = 0.0
        for cl in _cluster_strips(strips, ang_tol=math.radians(3.0), off_abs=0.03):
            for c in _cluster_to_centerlines(
                cl,
                prep,
                None,
                gap_factor=GAP_FACTOR,
                stats=tmp_stats,
                source="vote",
                layer_info=cache,
            ):
                if c.length() >= 1.0:  # nur lange Achsen zaehlen: Moebel und Symbole bleiben kurz
                    total += c.length()
        scores[name] = total
    if scores:
        best = max(scores.values())
        min_score = float(config.get("wall_layer_vote_min_m", 4.0))
        chosen = [n for n, v in scores.items() if v >= max(min_score, 0.25 * best)]
        if chosen:
            info.update(
                source="geometry",
                layers=sorted(chosen),
                scores={k: round(v, 2) for k, v in scores.items()},
            )
            return [s for n in chosen for s in by_layer[n]], info

    info.update(source="all_unknown", layers=sorted(unknown))
    return [s for n in unknown for s in by_layer[n]], info


# ── Hauptfunktion: Linien -> Achsen ──────────────────────────────────────


def reconstruct_walls(
    wall_segments: List[RawSegment],
    config: Optional[dict] = None,
    opening_boxes: Optional[Sequence[Sequence[Point2]]] = None,
) -> Tuple[List[Centerline], dict]:
    """Rekonstruiert Wand-Achsen aus klassifizierten Wand-Linien.

    ``opening_boxes``: orientierte Bounding-Boxen (vier Ecken, Meter) der Tuer-/Fensterbloecke; wo
    eine davon eine Luecke in einer Wand ueberdeckt, wird die Luecke geschlossen.

    Returns (centerlines, stats).
    """
    config = config or {}
    min_t = float(config.get("wall_min_thickness_m", MIN_THICKNESS_M))
    max_t = float(config.get("wall_max_thickness_m", MAX_THICKNESS_M))
    ang_tol = math.radians(float(config.get("wall_angle_tol_deg", ANGLE_TOL_DEG)))
    overlap_frac = float(config.get("wall_overlap_frac", OVERLAP_FRAC))
    default_t_int = float(config.get("wall_thickness_interior_m", 0.115))
    default_t_ext = float(config.get("wall_thickness_exterior_m", 0.30))
    min_len = float(config.get("wall_min_length_m", MIN_SEGMENT_M))
    gap_factor = float(config.get("wall_gap_factor", GAP_FACTOR))
    cov_min = float(config.get("wall_component_coverage_min", COMPONENT_COVERAGE_MIN))
    allow_bridge = bool(config.get("wall_bridge_gaps", True))
    nearest = bool(config.get("wall_nearest_partner", True))

    prep = _prepare(wall_segments, min_len)
    boxes = _boxes_union(opening_boxes)
    cache: Dict[str, lc.Classification] = {}
    stats: Dict[str, Any] = {
        "bridged_gaps": 0,
        "extended_ends": 0,
        "dropped_short_segments": prep.dropped_short,
    }

    centerlines: List[Centerline] = []
    comp_info: List[Dict[str, Any]] = []
    unpaired = 0
    unpaired_len = 0.0
    n = len(prep)
    total_len = float(prep.L.sum()) if n else 0.0
    matched_len_total = 0.0

    if n:
        comp = _components(prep, max_t + 0.02, boxes)
        nc = int(comp.max()) + 1
        # Paarsuche ueber alle Linien; Streifen entstehen ohnehin nur innerhalb einer Linienmenge
        strips, covered = _find_strips(
            prep,
            tmin=min_t,
            tmax=max_t,
            ang_tol=ang_tol,
            min_overlap=MIN_OVERLAP_ABS_M,
            overlap_frac=overlap_frac,
            nearest=nearest,
        )
        comp_len = np.bincount(comp, weights=prep.L, minlength=nc)
        comp_cov = np.bincount(comp, weights=covered * prep.L, minlength=nc)
        comp_n = np.bincount(comp, minlength=nc)
        has_strips = np.zeros(nc, dtype=bool)
        for s in strips:
            has_strips[comp[s.i]] = True
        ratio_c = np.divide(comp_cov, comp_len, out=np.zeros(nc), where=comp_len > 0)
        is_double = has_strips & (ratio_c >= cov_min)
        noise_len = float(config.get("wall_single_noise_max_m", SINGLE_NOISE_MAX_M))
        modes_c: List[str] = []
        for c in range(nc):
            if is_double[c]:
                modes_c.append("double_line")
            elif is_double.any() and comp_n[c] < 3 and comp_len[c] < noise_len:
                modes_c.append(
                    "noise"
                )  # ein loser Strich neben einem Doppellinienplan ist Rauschen
            else:
                modes_c.append("single_line")
        mode_arr = np.array(modes_c)

        # Doppellinien: Streifen aller Doppellinien-Mengen gemeinsam clustern, damit Luecken
        # zwischen
        # Mengen (Tuer, Fenster) in einer Achse geschlossen werden koennen
        d_strips = [s for s in strips if mode_arr[comp[s.i]] == "double_line"]
        out_by_comp: Dict[int, List[Centerline]] = {c: [] for c in range(nc)}
        for cl in _cluster_strips(d_strips, ang_tol=math.radians(3.0), off_abs=0.03):
            for cline in _cluster_to_centerlines(
                cl,
                prep,
                boxes,
                gap_factor=gap_factor,
                stats=stats,
                source="dwg_double_line",
                layer_info=cache,
                allow_bridge=allow_bridge,
            ):
                out_by_comp[int(comp[cline.ref_segment])].append(cline)
        weak = (covered < 0.5) & (mode_arr[comp] == "double_line")
        unpaired += int(weak.sum())
        unpaired_len += float(prep.L[weak].sum())
        unpaired += int(comp_n[mode_arr == "noise"].sum())
        unpaired_len += float(comp_len[mode_arr == "noise"].sum())
        matched_len_total += float(comp_cov[mode_arr == "double_line"].sum())

        # Einzellinien: jede Linie ist eine Achse; deckungsgleiche Linien zaehlen einmal
        s_idx = np.nonzero(mode_arr[comp] == "single_line")[0]
        if s_idx.size:
            sub = _sub_prep(prep, s_idx)
            for cline in _single_line_centerlines(
                sub, boxes, default_t_int, default_t_ext, gap_factor, stats, cache
            ):
                out_by_comp[int(comp[s_idx[cline.ref_segment]])].append(cline)

        for c in range(nc):
            out = out_by_comp[c]
            centerlines.extend(out)
            comp_info.append(
                {
                    "mode": modes_c[c],
                    "segments": int(comp_n[c]),
                    "matched_ratio": round(float(ratio_c[c]), 3),
                    "centerlines": len(out),
                    "length_m": round(sum(x.length() for x in out), 2),
                }
            )

    centerlines = [c for c in centerlines if c.length() >= min_len]
    stats["extended_ends"] = (
        _extend_to_junctions(centerlines, config)
        if config.get("wall_extend_junctions", True)
        else 0
    )

    modes = {c["mode"] for c in comp_info if c["mode"] != "noise"}
    mode = "none" if not modes else ("mixed" if len(modes) > 1 else next(iter(modes)))
    stats.update(
        {
            "mode": mode,
            "input_wall_segments": len(wall_segments),
            "centerlines": len(centerlines),
            "matched_ratio": round(matched_len_total / total_len, 3) if total_len > 0 else 0.0,
            "total_wall_length_m": round(sum(c.length() for c in centerlines), 2),
            "components": comp_info,
            "unpaired_segments": unpaired,
            "unpaired_length_m": round(unpaired_len, 2),
        }
    )
    return centerlines, stats


def _single_line_centerlines(
    sub: _Prep,
    boxes,
    default_int: float,
    default_ext: float,
    gap_factor: float,
    stats: Dict[str, Any],
    cache: Dict[str, "lc.Classification"],
) -> List[Centerline]:
    """Einlinig gezeichnete Waende: jede Linie ist eine Achse; deckungsgleiche Linien zaehlen
    einmal."""
    strips: List[_Strip] = []
    for k in range(len(sub)):
        strips.append(
            _Strip(sub.A[k].copy(), sub.B[k].copy(), sub.U[k].copy(), float(sub.ANG[k]), 0.0, k, k)
        )
    out: List[Centerline] = []
    for cl in _cluster_strips(strips, ang_tol=math.radians(1.5), off_abs=0.03):
        part = _cluster_to_centerlines(
            cl,
            sub,
            boxes,
            gap_factor=gap_factor,
            stats=stats,
            source="dwg_single_line",
            layer_info=cache,
            default_thickness=default_int,
        )
        for c in part:
            if c.wall_type == "exterior":
                c.thickness_m = default_ext
        out.extend(part)
    return out


# ── Achsenenden auf Schnittpunkte ziehen ─────────────────────────────────


def _extend_to_junctions(cls: List[Centerline], config: dict, passes: int = 2) -> int:
    """Mehrere Durchlaeufe: ein Ende, das an einer erst im ersten Lauf verlaengerten Wand haengt,
    findet sie im zweiten."""
    total = 0
    for _ in range(passes):
        moved = _extend_once(cls, config)
        total += moved
        if moved == 0:
            break
    return total


def _extend_once(cls: List[Centerline], config: dict) -> int:
    """Zieht Achsenenden an Ecken und T-Stoessen auf den Schnittpunkt der Achsen.

    Das Ende wird bewegt, wenn es hoechstens eine halbe Dicke der Gegenwand (plus Reserve) vom
    Schnittpunkt entfernt ist, in beide Richtungen: eine zu kurze Achse wird verlaengert, eine
    ueberstehende gekuerzt. Rueckgabe: Zahl der bewegten Enden.
    """
    n = len(cls)
    if n < 2:
        return 0
    slack = float(config.get("wall_junction_slack_m", JUNCTION_SLACK_M))
    min_sin = math.sin(math.radians(20.0))
    A = np.array([c.a for c in cls], dtype=float)
    B = np.array([c.b for c in cls], dtype=float)
    D = B - A
    L = np.hypot(D[:, 0], D[:, 1])
    U = D / L[:, None]
    N = np.stack([-U[:, 1], U[:, 0]], axis=1)
    T = np.array([c.thickness_m for c in cls], dtype=float)
    newA, newB = A.copy(), B.copy()
    moved = 0
    for i in range(n):
        for end in (0, 1):
            pt = A[i] if end == 0 else B[i]
            d = -U[i] if end == 0 else U[i]  # nach aussen
            others = np.arange(n) != i
            sin = np.abs(U[i, 0] * U[:, 1] - U[i, 1] * U[:, 0])
            nd = N @ d  # Nenner je Gegenwand
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.einsum("ij,ij->i", N, A - pt) / nd
                X = pt + s[:, None] * d
                r = np.einsum("ij,ij->i", X - A, U)  # Lage auf der Gegenwand
            end_slack = 0.5 * T[i] + slack
            ok = (
                others
                & (sin >= min_sin)
                & np.isfinite(s)
                & (np.abs(s) <= 0.5 * T + slack)
                & (r >= -end_slack)
                & (r <= L + end_slack)
            )
            if not ok.any():
                continue
            j = int(np.argmin(np.where(ok, np.abs(s), np.inf)))
            if abs(s[j]) < 1e-6:
                continue
            if end == 0:
                newA[i] = X[j]
            else:
                newB[i] = X[j]
            moved += 1
    for i, c in enumerate(cls):
        a2 = (float(newA[i][0]), float(newA[i][1]))
        b2 = (float(newB[i][0]), float(newB[i][1]))
        if math.hypot(b2[0] - a2[0], b2[1] - a2[1]) >= 0.05:
            c.a, c.b = a2, b2
    return moved


# ── Achsen -> WallGraph (Snapping + T-Stoss-Splitting) ───────────────────


def build_wall_graph(
    centerlines: List[Centerline],
    config: Optional[dict] = None,
    wall_height_m: float = 2.80,
) -> WallGraph:
    """Verdichtet Achsen zu einem topologisch sauberen WallGraph.

    1. Knoten-Snapping ueber einen Radius relativ zur Wanddicke (Junctions zusammenfuehren);
       ``config['wall_snap_radius_m']`` setzt einen festen Radius.
    2. T-Stoss-Splitting: liegt ein Knoten auf dem Inneren einer Kante,
       wird die Kante dort geteilt (sonst keine geschlossenen Raum-Faces).
    """
    config = config or {}
    fixed_snap = config.get("wall_snap_radius_m")
    snap_factor = float(config.get("wall_snap_factor", 0.6))
    snap_add = float(config.get("wall_snap_add_m", 0.0))
    split_tol = config.get("wall_split_tol_m")

    graph = WallGraph()
    for cl in centerlines:
        if cl.length() < 1e-6:
            continue
        snap_r = (
            float(fixed_snap)
            if fixed_snap is not None
            else min(0.50, max(0.10, snap_factor * cl.thickness_m + snap_add))
        )
        attrs: Dict[str, object] = {"bearing": cl.bearing}
        if cl.layer:
            attrs["layer"] = cl.layer
        # Knoten ueber den groesseren der beiden Radien zusammenfuehren (dicke Wand + duenne Wand)
        na = _snap_node(graph, cl.a, snap_r)
        nb = _snap_node(graph, cl.b, snap_r)
        if na.id == nb.id:
            continue  # Null-Wand nach dem Snappen
        graph.add_edge(
            na.id,
            nb.id,
            thickness_m=cl.thickness_m,
            height_m=wall_height_m,
            wall_type=cl.wall_type,
            source=cl.source,
            attrs=attrs,
        )

    _split_edges_at_nodes(graph, None if split_tol is None else float(split_tol))
    graph.refresh_node_kinds()
    graph.prune_orphan_nodes()
    return graph


def _snap_node(graph: WallGraph, xy: Point2, radius: float):
    """Vorhandener Knoten, dessen Fangradius oder der des Neuen den Abstand deckt, sonst ein
    neuer."""
    best = None
    for node in graph.nodes.values():
        reach = max(radius, float(node.attrs.get("snap_r", 0.0) or 0.0))
        d = math.hypot(node.xy[0] - xy[0], node.xy[1] - xy[1])
        if d <= reach and (best is None or d < best[0]):
            best = (d, node)
    if best is not None:
        node = best[1]
        node.attrs["snap_r"] = max(radius, float(node.attrs.get("snap_r", 0.0) or 0.0))
        return node
    node = graph.add_node(xy)
    node.attrs["snap_r"] = radius
    return node


def _split_edges_at_nodes(graph: WallGraph, tol_m: Optional[float]) -> None:
    """Teilt Kanten, durch deren Inneres ein bestehender Knoten verlaeuft.

    Toleranz: ``tol_m`` oder, wenn None, relativ zur Dicke der Kante (mindestens 5 cm).
    """
    changed = True
    guard = 0
    while changed and guard < 1000:
        changed = False
        guard += 1
        for edge in list(graph.edges.values()):
            tol = tol_m if tol_m is not None else max(0.05, min(0.20, 0.35 * edge.thickness_m))
            a_xy, b_xy = graph.edge_endpoints(edge.id)
            L = graph.edge_length(edge.id)
            if L < 2 * tol:
                continue
            best_node = None
            best_off = None
            for node in graph.nodes.values():
                if node.id in (edge.node_a, edge.node_b):
                    continue
                off, perp = graph.project_to_edge(edge.id, node.xy)
                # echter Innenpunkt mit kleinem Normalabstand
                if perp <= tol and (tol < off < L - tol):
                    if best_off is None or off < best_off:
                        best_off = off
                        best_node = node
            if best_node is not None:
                # Kante an best_node teilen: zwei neue Kanten, alte entfernen
                graph.remove_edge(edge.id)
                for na, nb in ((edge.node_a, best_node.id), (best_node.id, edge.node_b)):
                    if na == nb:
                        continue
                    graph.add_edge(
                        na,
                        nb,
                        thickness_m=edge.thickness_m,
                        height_m=edge.height_m,
                        wall_type=edge.wall_type,
                        source=edge.source,
                        confidence=edge.confidence,
                        attrs=dict(edge.attrs),
                    )
                changed = True
                break


__all__ = [
    "Centerline",
    "reconstruct_walls",
    "select_wall_segments",
    "build_wall_graph",
]
