from __future__ import annotations

import math
from typing import Dict, List, Tuple

Point = Tuple[float, float]


def _len(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _angle(a: Point, b: Point) -> float:
    # normalize to [0, pi)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    if ang < 0:
        ang += math.pi
    return ang


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, math.pi - d)


def _point_line_distance(p: Point, a: Point, b: Point) -> float:
    # distance from point p to infinite line through a-b
    ax, ay = a; bx, by = b; px, py = p
    vx, vy = (bx - ax, by - ay)
    vv = vx * vx + vy * vy
    if vv <= 1e-12:
        return math.hypot(px - ax, py - ay)
    # perpendicular distance via cross product magnitude / |v|
    cross = abs((px - ax) * vy - (py - ay) * vx)
    return cross / math.sqrt(vv)


def _project_t(p: Point, a: Point, dirv: Point) -> float:
    # scalar projection onto dir vector (assumed unit-ish)
    return (p[0] - a[0]) * dirv[0] + (p[1] - a[1]) * dirv[1]


def merge_collinear_segments(
    segments: List[Dict],
    angle_eps_deg: float = 3.0,
    collinear_tol_m: float = 0.02,
    join_tol_m: float = 0.06,
    min_len_m: float = 0.50,
) -> List[Dict]:
    """
    Input: geometry line dicts with keys: id, a=[x,y], b=[x,y]
    Output: merged segments (same shape), ids become "wallN"
    Heuristics:
      - exclude very short segments
      - merge if angle similar + collinear + endpoints close/overlap
    """
    angle_eps = math.radians(angle_eps_deg)

    segs = []
    for g in segments:
        a = g.get("a"); b = g.get("b")
        if not a or not b:
            continue
        A = (float(a[0]), float(a[1]))
        B = (float(b[0]), float(b[1]))
        if _len(A, B) < min_len_m:
            continue
        segs.append({"id": g.get("id", ""), "a": A, "b": B})

    out = []
    used = [False] * len(segs)

    wall_id = 1

    for i in range(len(segs)):
        if used[i]:
            continue

        base = segs[i]
        used[i] = True

        # work segment that grows
        A = base["a"]
        B = base["b"]

        changed = True
        while changed:
            changed = False
            ang0 = _angle(A, B)
            # unit direction
            L = _len(A, B)
            dirv = ((B[0] - A[0]) / L, (B[1] - A[1]) / L)

            for j in range(len(segs)):
                if used[j]:
                    continue
                s = segs[j]
                a2, b2 = s["a"], s["b"]
                ang2 = _angle(a2, b2)

                if _angle_diff(ang0, ang2) > angle_eps:
                    continue

                # collinear check: both endpoints near line A-B
                if _point_line_distance(a2, A, B) > collinear_tol_m:
                    continue
                if _point_line_distance(b2, A, B) > collinear_tol_m:
                    continue

                # check if joinable/overlapping in projection
                tA = 0.0
                tB = _len(A, B)
                ta2 = _project_t(a2, A, dirv)
                tb2 = _project_t(b2, A, dirv)
                lo2 = min(ta2, tb2)
                hi2 = max(ta2, tb2)

                # Overlap or endpoints close
                if hi2 < tA - join_tol_m or lo2 > tB + join_tol_m:
                    continue

                # merge by extending projection interval
                new_lo = min(tA, lo2)
                new_hi = max(tB, hi2)
                A = (A[0] + dirv[0] * (new_lo - tA), A[1] + dirv[1] * (new_lo - tA))
                B = (A[0] + dirv[0] * (new_hi - new_lo), A[1] + dirv[1] * (new_hi - new_lo))

                used[j] = True
                changed = True

        out.append({
            "id": f"wall{wall_id}",
            "type": "wall_segment",
            "a": [A[0], A[1]],
            "b": [B[0], B[1]],
        })
        wall_id += 1

    return out
