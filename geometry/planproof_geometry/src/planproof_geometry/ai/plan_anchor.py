from __future__ import annotations

import math
from typing import Dict, List, Tuple

Point = Tuple[float, float]


def _bbox_center(bbox) -> Point:
    x0, y0, x1, y1 = map(float, bbox)
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def _dist_point_to_segment(p: Point, a: Point, b: Point) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    vx, vy = (bx - ax, by - ay)
    wx, wy = (px - ax, py - ay)
    vv = vx * vx + vy * vy
    if vv <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = (wx * vx + wy * vy) / vv
    t = max(0.0, min(1.0, t))
    cx, cy = (ax + t * vx, ay + t * vy)
    return math.hypot(px - cx, py - cy)


def _dist_point_to_bbox(p: Point, bbox) -> float:
    """Euklidischer Abstand Punkt -> Axis-aligned BBox."""
    px, py = p
    x0, y0, x1, y1 = map(float, bbox)

    dx = 0.0
    if px < x0:
        dx = x0 - px
    elif px > x1:
        dx = px - x1

    dy = 0.0
    if py < y0:
        dy = y0 - py
    elif py > y1:
        dy = py - y1

    return math.hypot(dx, dy)


def _seg_angle_deg(a: Point, b: Point) -> float:
    ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    if ang < 0:
        ang += 180.0
    return ang


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _point_to_infinite_line_dist(p: Point, a: Point, b: Point) -> float:
    """Abstand Punkt -> unendliche Gerade durch a,b."""
    ax, ay = a
    bx, by = b
    px, py = p
    vx, vy = (bx - ax, by - ay)
    L = math.hypot(vx, vy)
    if L <= 1e-12:
        return math.hypot(px - ax, py - ay)
    # |(b-a) x (p-a)| / |b-a|
    return abs(vx * (py - ay) - vy * (px - ax)) / L


def _mark_interrupted_dimension_segments(
    geom_lines: List[Dict],
    *,
    base_line: Dict,
    text_bbox,
    used_geom_ids: set,
    angle_eps_deg: float = 6.0,
    collinear_tol_m: float = 0.08,
    end_dist_to_text_bbox_m: float = 0.40,
) -> None:
    """
    Deine Idee umgesetzt:
    Wenn eine Maßzahl in der Mitte steht und die Maßlinie dadurch "unterbrochen" ist,
    werden die Segmente links/rechts (kollinear, nahe an der Text-BBox) ebenfalls als Maßlinie markiert.
    """
    a = base_line.get("a")
    b = base_line.get("b")
    if not a or not b:
        return

    A = (float(a[0]), float(a[1]))
    B = (float(b[0]), float(b[1]))
    base_ang = _seg_angle_deg(A, B)

    # Wir suchen weitere Segmente, die:
    # - nahezu gleiche Richtung haben
    # - auf derselben Geraden liegen (collinear)
    # - mit einem Endpunkt nahe an der Text-BBox "andocken"
    for g in geom_lines:
        gid = g.get("id")
        if not gid or gid in used_geom_ids:
            continue
        ga = g.get("a")
        gb = g.get("b")
        if not ga or not gb:
            continue

        C = (float(ga[0]), float(ga[1]))
        D = (float(gb[0]), float(gb[1]))
        ang = _seg_angle_deg(C, D)

        if _angle_diff_deg(base_ang, ang) > angle_eps_deg:
            continue

        # Collinear: beide Endpunkte nahe an der Basis-Geraden
        if (
            _point_to_infinite_line_dist(C, A, B) > collinear_tol_m
            and _point_to_infinite_line_dist(D, A, B) > collinear_tol_m
        ):
            continue

        # "Andocken" an die Text-BBox: mind. ein Endpunkt ist nah an der BBox
        if min(_dist_point_to_bbox(C, text_bbox), _dist_point_to_bbox(D, text_bbox)) > end_dist_to_text_bbox_m:
            continue

        used_geom_ids.add(gid)


def attach_measurement_anchors(extraction: Dict, max_dist_m: float = 1.20) -> Dict:
    """
    Für jedes Längenmaß:
    - finde nächstes Liniensegment
    - schreibe m["anchor"] = {...}
    - markiere die zugehörige Linie als is_dimension_line=True
    - PLUS: markiere "unterbrochene" Maßliniensegmente neben der Maßzahl
    """
    texts = {t["id"]: t for t in extraction.get("texts", []) if "id" in t}
    geom_lines = [g for g in extraction.get("geometry", []) if g.get("type") == "line"]

    used_geom_ids = set()

    for m in extraction.get("measurements", []):
        if (m.get("kind") or "") != "length":
            continue

        t = texts.get(m.get("raw_text_id"))
        if not t:
            continue

        bbox = t.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        p = _bbox_center(bbox)

        best = None  # (dist, geom_obj)
        for g in geom_lines:
            a = g.get("a")
            b = g.get("b")
            if not a or not b:
                continue

            d = _dist_point_to_segment(
                p,
                (float(a[0]), float(a[1])),
                (float(b[0]), float(b[1])),
            )
            if best is None or d < best[0]:
                best = (d, g)

        if best is None:
            continue

        d, best_g = best
        if d > max_dist_m:
            continue

        gid = best_g.get("id")
        if not gid:
            continue

        # 1) Measurement bekommt Anchor-Info (Debug + Nachvollziehbarkeit)
        m["anchor"] = {
            "type": "nearest_line",
            "geometry_id": gid,
            "distance_m": float(d),
        }

        # 2) Basis-Segment als Maßlinie
        used_geom_ids.add(gid)

        _grow_dimension_chain(
            geom_lines,
            seed_line=best_g,
            used_geom_ids=used_geom_ids,
            angle_eps_deg=6.0,
            collinear_tol_m=0.05,
            join_tol_m=0.25,
        )

        # 3) Deine "unterbrochen durch Zahl"-Idee: zusätzliche Segmente links/rechts markieren
        _mark_interrupted_dimension_segments(
            geom_lines,
            base_line=best_g,
            text_bbox=bbox,
            used_geom_ids=used_geom_ids,
        )

    # Markieren
    for g in geom_lines:
        if g.get("id") in used_geom_ids:
            g["is_dimension_line"] = True

    return extraction

def _grow_dimension_chain(
    geom_lines: List[Dict],
    *,
    seed_line: Dict,
    used_geom_ids: set,
    angle_eps_deg: float = 6.0,
    collinear_tol_m: float = 0.05,
    join_tol_m: float = 0.25,
) -> None:
    a = seed_line.get("a"); b = seed_line.get("b")
    if not a or not b:
        return

    A = (float(a[0]), float(a[1]))
    B = (float(b[0]), float(b[1]))
    base_ang = _seg_angle_deg(A, B)

    # Endpunkte, an denen wir weiterwachsen
    frontier = [A, B]

    changed = True
    while changed:
        changed = False
        new_frontier = []

        for g in geom_lines:
            gid = g.get("id")
            if not gid or gid in used_geom_ids:
                continue
            ga = g.get("a"); gb = g.get("b")
            if not ga or not gb:
                continue

            C = (float(ga[0]), float(ga[1]))
            D = (float(gb[0]), float(gb[1]))

            # Richtung ähnlich?
            ang = _seg_angle_deg(C, D)
            if _angle_diff_deg(base_ang, ang) > angle_eps_deg:
                continue

            # Wirklich collinear (nicht nur parallel versetzt!)
            if _point_to_infinite_line_dist(C, A, B) > collinear_tol_m:
                continue
            if _point_to_infinite_line_dist(D, A, B) > collinear_tol_m:
                continue

            # Andocken an bisherige Frontier
            def close(p,q): 
                return math.hypot(p[0]-q[0], p[1]-q[1]) <= join_tol_m

            if any(close(C, f) or close(D, f) for f in frontier):
                used_geom_ids.add(gid)
                new_frontier.extend([C, D])
                changed = True

        frontier.extend(new_frontier)
