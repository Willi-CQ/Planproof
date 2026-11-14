from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

Point = Tuple[float, float]

# Standard paper sizes (width, height) in meters for scale plausibility scoring
_PAPER_SIZES_M: List[Tuple[float, float]] = [
    (0.297, 0.210),  # A4
    (0.420, 0.297),  # A3
    (0.594, 0.420),  # A2
    (0.841, 0.594),  # A1
    (1.189, 0.841),  # A0
]


# --- kleine Geometrie-Helper -------------------------------------------------


def _center_bbox(bbox: List[float]) -> Point:
    x0, y0, x1, y1 = map(float, bbox)
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def _seg_len(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _dist_point_to_seg(p: Point, a: Point, b: Point) -> float:
    """Abstand Punkt -> Segment."""
    px, py = p
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay

    seg_len2 = vx * vx + vy * vy
    if seg_len2 <= 1e-12:
        return math.hypot(px - ax, py - ay)

    t = (vx * wx + vy * wy) / seg_len2
    if t <= 0.0:
        projx, projy = ax, ay
    elif t >= 1.0:
        projx, projy = bx, by
    else:
        projx, projy = ax + t * vx, ay + t * vy
    return math.hypot(px - projx, py - projy)


def _seg_angle_deg(a: Point, b: Point) -> float:
    """0° ~ horizontal, 90° ~ vertikal (auf 0..180° normalisiert)."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    ang = math.degrees(math.atan2(dy, dx))
    if ang < 0:
        ang += 180.0
    return ang


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b)
    while d > 180.0:
        d -= 360.0
    if d > 180.0:
        d = 360.0 - d
    return d


def _is_horiz(ang: float, eps: float = 15.0) -> bool:
    return min(_angle_diff_deg(ang, 0.0), _angle_diff_deg(ang, 180.0)) <= eps


def _is_vert(ang: float, eps: float = 15.0) -> bool:
    return _angle_diff_deg(ang, 90.0) <= eps


# --- Weighted Median ---------------------------------------------------------


def _weighted_median(values: List[float], weights: List[float]) -> Optional[float]:
    assert len(values) == len(weights)
    pairs = [(v, w) for v, w in zip(values, weights) if w > 0]
    if not pairs:
        return None
    pairs.sort(key=lambda vw: vw[0])
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return None
    cutoff = total_w * 0.5
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= cutoff:
            return v
    return pairs[-1][0]


# --- Dimension-Linien erkennen (Hauptlinie mit 2 orthogonalen Ticks) ---------


def _detect_dimension_mainlines(geom_lines: List[Dict]) -> Dict[str, bool]:
    """
    Heuristik:
    - eine längere Linie
    - an beiden Enden je mind. ein kleineres Segment, das ~90° dazu steht
    - kleine Segmente liegen nahe an den Endpunkten
    -> wird als Dimension-Hauptlinie markiert.
    """
    if not geom_lines:
        return {}

    # BBox zur Skalierung von Abständen
    xs: List[float] = []
    ys: List[float] = []
    for g in geom_lines:
        a = g.get("a")
        b = g.get("b")
        if not a or not b:
            continue
        xs.extend([float(a[0]), float(b[0])])
        ys.extend([float(a[1]), float(b[1])])

    if not xs or not ys:
        return {}

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span = max(maxx - minx, maxy - miny, 1e-6)

    # Parameter relativ zur Seitengröße
    min_main_len = 0.05 * span     # minimale Hauptlänge
    tick_len_min = 0.003 * span
    tick_len_max_ratio = 0.25      # Ticks max 25% der Hauptlänge
    tick_dist_max = 0.02 * span    # max. Abstand Endpunkt <-> Tick-Mitte

    n = len(geom_lines)
    is_main: Dict[str, bool] = {}

    # Vorberechnungen
    seg_cache = []
    for g in geom_lines:
        a = g.get("a")
        b = g.get("b")
        if not a or not b:
            seg_cache.append(None)
            continue
        A = (float(a[0]), float(a[1]))
        B = (float(b[0]), float(b[1]))
        L = _seg_len(A, B)
        ang = _seg_angle_deg(A, B)
        seg_cache.append((A, B, L, ang))

    for i, g in enumerate(geom_lines):
        if not seg_cache[i]:
            continue
        A, B, L, ang_main = seg_cache[i]
        if L < min_main_len:
            continue

        # Suche Ticks an A und B — collect tick indices for later flagging
        has_tick_A = False
        has_tick_B = False
        tick_indices: List[int] = []

        for j, h in enumerate(geom_lines):
            if i == j or not seg_cache[j]:
                continue
            C, D, L2, ang2 = seg_cache[j]

            # Klein & orthogonal?
            if L2 < tick_len_min or L2 > L * tick_len_max_ratio:
                continue
            if _angle_diff_deg(ang_main, ang2) < 60.0:  # nicht orthogonal genug
                continue

            mid = ((C[0] + D[0]) * 0.5, (C[1] + D[1]) * 0.5)
            dA = _seg_len(A, mid)
            dB = _seg_len(B, mid)

            if dA <= tick_dist_max:
                has_tick_A = True
                tick_indices.append(j)
            if dB <= tick_dist_max:
                has_tick_B = True
                tick_indices.append(j)

            if has_tick_A and has_tick_B:
                break

        if has_tick_A and has_tick_B:
            gid = str(g.get("id") or f"g{i}")
            is_main[gid] = True
            # Flag auch direkt am Geometry-Objekt ablegen (kann später genutzt werden)
            g["is_dimension_mainline"] = True
            # Flag tick segments so they can be excluded from BIM output
            for ti in set(tick_indices):
                geom_lines[ti]["is_dimension_tick"] = True

    return is_main


# --- Measurement -> Meter ----------------------------------------------------


def _measurement_to_m(m: Dict) -> Optional[float]:
    """Wert in Meter umrechnen (m, cm, mm)."""
    try:
        val = float(m.get("value"))
    except Exception:
        return None
    unit = (m.get("unit") or "m").lower()
    if unit in ("m", "meter", "metre"):
        factor = 1.0
    elif unit in ("cm", "centimeter", "centimetre"):
        factor = 0.01
    elif unit in ("mm", "millimeter", "millimetre"):
        factor = 0.001
    else:
        # unbekannte Einheit; lieber ignorieren
        return None
    value_m = val * factor
    if not (0.1 <= value_m <= 500.0):
        # alles was viel zu klein/groß ist, ist meist kein sinnvolles Längenmaß
        return None
    return value_m


def _best_measure_segment(
    geometry_lines: List[Dict],
    p: Point,
    *,
    prefer: str,
    max_dist: float,
) -> Optional[Dict]:
    """
    Two-pass matching: prefer dimension lines over arbitrary geometry.

    Pass 1: Search ONLY among segments flagged as dimension mainlines/lines.
    Pass 2 (fallback): Search all segments, scored by distance + orientation only
            (no length bonus that would favour long wall lines).
    """

    def _score_segment(g: Dict, *, dimension_only: bool) -> Optional[float]:
        if g.get("type") != "line":
            return None
        if g.get("is_dimension_tick"):
            return None  # Tick segments are short orthogonal marks, never valid measure targets
        a = g.get("a")
        b = g.get("b")
        if not a or not b:
            return None
        A = (float(a[0]), float(a[1]))
        B = (float(b[0]), float(b[1]))
        L = _seg_len(A, B)
        if L < 1e-6:
            return None

        is_dim = g.get("is_dimension_mainline") or g.get("is_dimension_line")
        if dimension_only and not is_dim:
            return None

        d = _dist_point_to_seg(p, A, B)
        if d > max_dist:
            return None

        ang = _seg_angle_deg(A, B)
        orient_penalty = 0.0
        if prefer == "h" and not _is_horiz(ang):
            orient_penalty = 2.0
        elif prefer == "v" and not _is_vert(ang):
            orient_penalty = 2.0

        return d * (1.0 + orient_penalty)

    # Pass 1: dimension lines only
    best = None
    best_score = 1e18
    for g in geometry_lines:
        s = _score_segment(g, dimension_only=True)
        if s is not None and s < best_score:
            best_score = s
            best = g
    if best is not None:
        return best

    # Pass 2: all segments (fallback — no length bonus)
    best_score = 1e18
    for g in geometry_lines:
        s = _score_segment(g, dimension_only=False)
        if s is not None and s < best_score:
            best_score = s
            best = g

    return best


# --- YOLO dimension_line hints -------------------------------------------------


def _apply_yolo_dimension_hints(
    geom: List[Dict],
    detections: List[Dict],
    px_to_coord_scale: float,
    class_filter: str = "dimension_line",
) -> None:
    """
    Mark geometry lines that overlap with YOLO dimension_line bboxes.
    Modifies geom in place: sets is_dimension_mainline=True for overlapping segments.
    """
    if not detections or not geom or px_to_coord_scale <= 0:
        return

    for g in geom:
        if g.get("type") != "line":
            continue
        a = g.get("a")
        b = g.get("b")
        if not a or not b:
            continue
        A = (float(a[0]), float(a[1]))
        B = (float(b[0]), float(b[1]))
        seg_mid = ((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5)

        for det in detections:
            if det.get("class") != class_filter:
                continue
            bbox = det.get("bbox", [0, 0, 0, 0])
            if len(bbox) != 4:
                continue
            # Scale from pixel to extraction coords
            x1 = bbox[0] * px_to_coord_scale
            y1 = bbox[1] * px_to_coord_scale
            x2 = bbox[2] * px_to_coord_scale
            y2 = bbox[3] * px_to_coord_scale
            # Check if segment midpoint is inside bbox (with margin)
            margin = 0.02 * max(abs(x2 - x1), abs(y2 - y1), 1.0)
            if (
                min(x1, x2) - margin <= seg_mid[0] <= max(x1, x2) + margin
                and min(y1, y2) - margin <= seg_mid[1] <= max(y1, y2) + margin
            ):
                g["is_dimension_mainline"] = True
                g["is_dimension_line"] = True
                break


# --- Hauptfunktion -----------------------------------------------------------


def infer_points_to_m_from_dimension_lines_only(
    extraction: Dict,
    detection_result: Optional[Dict] = None,
    px_to_coord_scale: float = 1.0,
) -> Tuple[Dict, bool]:
    """
    Infer points_to_m ausschließlich aus Maßlinien+Maßzahl-Paaren.
    Returns (extraction, success).
    """
    return _infer_impl(
        extraction,
        detection_result=detection_result,
        px_to_coord_scale=px_to_coord_scale,
        dimension_lines_only=True,
    )


def _infer_impl(
    extraction: Dict,
    detection_result: Optional[Dict] = None,
    px_to_coord_scale: float = 1.0,
    dimension_lines_only: bool = False,
) -> Tuple[Dict, bool]:
    """Intern: dimension_lines_only=True → nur Maßlinien-Paare verwenden. Returns (extraction, success)."""
    doc = extraction.get("doc", {})
    scale = doc.get("scale") or {}
    ratio = scale.get("ratio")

    # Wenn eine explizite Ratio existiert ("1:50" etc.), nichts überschreiben
    if ratio:
        return extraction, False

    texts = extraction.get("texts") or []
    geom = [g for g in (extraction.get("geometry") or []) if g.get("type") == "line"]
    meas = extraction.get("measurements") or []

    if not texts or not geom or not meas:
        return extraction, False

    # Text nach ID mappen
    text_by_id: Dict[str, Dict] = {}
    for t in texts:
        tid = str(t.get("id") or "")
        if not tid:
            continue
        text_by_id[tid] = t

    # YOLO dimension_line hints: mark geometry lines overlapping with detections
    if detection_result and px_to_coord_scale > 0:
        dets = detection_result.get("detections", [])
        _apply_yolo_dimension_hints(geom, dets, px_to_coord_scale, "dimension_line")

    # Dimension-Hauptlinien erkennen (für Gewichtung) – Heuristik + YOLO-Markierungen
    dim_main = _detect_dimension_mainlines(geom)
    for g in geom:
        if g.get("is_dimension_mainline") and g.get("id"):
            dim_main[str(g.get("id"))] = True

    # Bei dimension_lines_only: nur Maßlinien für Segment-Suche verwenden
    if dimension_lines_only:
        geom_for_pairing = [g for g in geom if g.get("is_dimension_mainline") or g.get("is_dimension_line")]
        if not geom_for_pairing:
            return extraction, False
        geom = geom_for_pairing

    # BBox im Planraum für max_dist-Schätzung
    xs: List[float] = []
    ys: List[float] = []
    for g in geom:
        a = g.get("a")
        b = g.get("b")
        if not a or not b:
            continue
        xs.extend([float(a[0]), float(b[0])])
        ys.extend([float(a[1]), float(b[1])])
    if not xs or not ys:
        return extraction, False
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span = max(maxx - minx, maxy - miny, 1e-6)
    max_dist = 0.06 * span

    candidates: List[Dict] = []

    for m in meas:
        if (m.get("kind") or "") != "length":
            continue

        value_m = _measurement_to_m(m)
        if value_m is None:
            continue

        # Vertrauen in das Measurement (0..1)
        try:
            mconf = float(m.get("confidence", 0.0))
        except Exception:
            mconf = 0.0

        tid = str(m.get("raw_text_id") or "")
        t = text_by_id.get(tid)
        if not t:
            continue
        bbox = t.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        # OCR-Confidence (0..100)
        ocr_conf = float(t.get("conf", -1.0))
        if ocr_conf < 0:
            ocr01: Optional[float] = None
        else:
            ocr01 = max(0.0, min(1.0, ocr_conf / 100.0))

        # Text-BBox -> Orientierung, um horizontale/vertikale Linien zu bevorzugen
        x0, y0, x1, y1 = map(float, bbox)
        bw = max(1e-6, x1 - x0)
        bh = max(1e-6, y1 - y0)
        if bw >= 1.2 * bh:
            prefer = "h"
        elif bh >= 1.2 * bw:
            prefer = "v"
        else:
            prefer = "any"

        p = _center_bbox(bbox)
        seg = _best_measure_segment(geom, p, prefer=prefer, max_dist=max_dist)
        if not seg:
            continue

        a = seg.get("a")
        b = seg.get("b")
        if not a or not b:
            continue
        A = (float(a[0]), float(a[1]))
        B = (float(b[0]), float(b[1]))
        L_pt = _seg_len(A, B)
        if L_pt < 0.02 * span:
            # Short lines (< 2% of page span) are rarely valid measurement targets
            continue

        pt_to_m = value_m / L_pt

        # grobe Plausibilitäts-Prüfung:
        span_pt = span
        span_m = span_pt * pt_to_m
        if not (2.0 <= span_m <= 500.0):
            # führt zu völlig unplausiblem Gesamtmaß -> nur als weak debuggen
            base_weight = 0.0
        else:
            # Score aus Measurement-Conf + OCR-Conf
            if ocr01 is None:
                base = mconf
            else:
                base = 0.6 * mconf + 0.4 * ocr01
            base = max(0.0, min(1.0, base))

            # Dimension-Hauptlinie? -> höheres Gewicht
            gid = str(seg.get("id") or "")
            if gid and (gid in dim_main or seg.get("is_dimension_line")):
                base *= 1.5

            base_weight = max(0.0, min(1.0, base))

        # Klassifikation nach deinem Schema:
        # strong (maßgebend) / medium / weak (Gewicht 0)
        if base_weight >= 0.7:
            weight = base_weight          # strong
            level = "strong"
        elif base_weight >= 0.4:
            weight = 0.5 * base_weight    # medium
            level = "medium"
        else:
            weight = 0.0                  # weak -> nur Debug
            level = "weak"

        candidates.append(
            {
                "measurement_id": m.get("id"),
                "text_id": tid,
                "value_m": value_m,
                "pt_to_m": pt_to_m,
                "span_m": span_m,
                "weight": float(weight),
                "level": level,
                "measurement_conf": mconf,
                "ocr_conf": ocr_conf if ocr_conf >= 0 else None,
                "is_dimension_mainline": bool(str(seg.get("id") or "") in dim_main),
                "geometry_id": seg.get("id"),
            }
        )

    # Nur Kandidaten mit Gewicht > 0 für die eigentliche Skala verwenden
    used = [c for c in candidates if c["weight"] > 0.0]
    if not used:
        # Debug-Infos trotzdem zurückgeben
        doc["scale_debug"] = {
            "candidates": candidates,
            "used": [],
            "reason": "no_weighted_candidates",
        }
        extraction["doc"] = doc
        return extraction, False

    scales = [c["pt_to_m"] for c in used]
    weights = [c["weight"] for c in used]

    pt_to_m_real = _weighted_median(scales, weights)
    if pt_to_m_real is None:
        doc["scale_debug"] = {
            "candidates": candidates,
            "used": [],
            "reason": "weighted_median_failed",
        }
        extraction["doc"] = doc
        return extraction, False

    # Confidence aus Streuung + Gesamtgewicht abschätzen
    used_scales = sorted(scales)
    n = len(used_scales)
    med = used_scales[n // 2]

    if n >= 4 and med > 1e-12:
        q1 = used_scales[n // 4]
        q3 = used_scales[(3 * n) // 4]
        rel_spread = abs(q3 - med) / med + abs(med - q1) / med
        rel_spread *= 0.5
    else:
        rel_spread = 0.0

    total_w = sum(weights)
    avg_w = total_w / max(1, len(weights))

    conf = 0.5
    conf += min(0.3, 0.06 * len(used))      # mehr Samples -> höheres Vertrauen
    conf += min(0.15, 0.3 * avg_w)          # höheres Gewicht -> etwas höheres Vertrauen
    conf -= min(0.35, 1.2 * rel_spread)     # hohe Streuung -> deutlich weniger Vertrauen

    # Boost: single measurement from dimension line with high OCR confidence
    if len(used) == 1 and used[0].get("is_dimension_mainline"):
        ocr_c = used[0].get("ocr_conf")
        if ocr_c is not None and ocr_c > 0.7:
            conf = max(conf, 0.75)

    conf = max(0.05, min(0.98, conf))

    # --- Plausibility check: reject wildly implausible scales ---
    # If the inferred points_to_m yields an implausible page span, try
    # standard architectural scales and pick the most plausible one.
    _page_size = doc.get("page_size", {})
    page_w_pt = float(_page_size.get("w") or 595)
    page_h_pt = float(_page_size.get("h") or 842)
    page_span_pt = max(page_w_pt, page_h_pt)
    page_span_m = page_span_pt * pt_to_m_real

    _STANDARD_SCALES = [
        (1, 50),   # pt_to_m = 1/(50*72/25.4) ≈ 0.00706
        (1, 75),
        (1, 100),  # typical residential ≈ 0.01411 * (100/72) ≈ 0.0353
        (1, 200),
        (1, 500),
    ]

    if page_span_m > 80.0 and len(used) <= 2:
        # The inferred scale gives an implausible page span.
        # Try standard scales: for each used measurement, compute what pt_to_m
        # would be at standard scales and check how well the measurement
        # text value matches the candidate geometry line length.
        best_std_pt_to_m = None
        best_std_score = 1e18
        best_std_info = None

        for cand in used:
            value_m = cand["value_m"]
            geom_id = cand.get("geometry_id")
            # Find the geometry line length in points
            geom_line = None
            for g in geom:
                if g.get("id") == geom_id:
                    geom_line = g
                    break
            if geom_line is None:
                continue
            ga = geom_line.get("a", [0, 0])
            gb = geom_line.get("b", [0, 0])
            geom_len_pt = _seg_len(
                (float(ga[0]), float(ga[1])),
                (float(gb[0]), float(gb[1])),
            )
            if geom_len_pt < 1e-6:
                continue

            # Physical paper dimensions for plausibility scoring
            physical_paper_w = page_w_pt / 72.0 * 0.0254
            physical_paper_h = page_h_pt / 72.0 * 0.0254

            for (sn, sd) in _STANDARD_SCALES:
                std_pt_to_m = sd * 0.0254 / 72.0  # e.g. 1:100 → 100*0.0254/72 ≈ 0.03528
                expected_line_pt = value_m / std_pt_to_m
                ratio = geom_len_pt / expected_line_pt if expected_line_pt > 1e-6 else 100
                geom_score = abs(math.log(max(ratio, 0.01)))

                # Paper-size plausibility: does the plan extent make sense?
                plan_span_m_std = max(page_w_pt, page_h_pt) * std_pt_to_m
                paper_bonus = 0.0
                for pw_m, ph_m in _PAPER_SIZES_M:
                    for ppw, pph in [(pw_m, ph_m), (ph_m, pw_m)]:
                        if (0.7 <= physical_paper_w / ppw <= 1.3
                                and 0.7 <= physical_paper_h / pph <= 1.3):
                            if 8.0 <= plan_span_m_std <= 55.0:
                                paper_bonus = 0.3
                            elif 5.0 <= plan_span_m_std <= 70.0:
                                paper_bonus = 0.15
                            break
                    if paper_bonus > 0:
                        break

                # Scale prior: 1:100 is the most common residential plan scale
                scale_prior = 0.1 if sd == 100 else 0.0

                score = geom_score - paper_bonus - scale_prior
                if score < best_std_score:
                    best_std_score = score
                    best_std_pt_to_m = std_pt_to_m
                    best_std_info = f"1:{sd}"

        # Also try: for each measurement, search ALL geometry lines for a better match at standard scales
        all_geom = [g for g in (extraction.get("geometry") or []) if g.get("type") == "line"]
        physical_paper_w = page_w_pt / 72.0 * 0.0254
        physical_paper_h = page_h_pt / 72.0 * 0.0254

        for cand in used:
            value_m = cand["value_m"]
            for (sn, sd) in _STANDARD_SCALES:
                std_pt_to_m = sd * 0.0254 / 72.0
                expected_line_pt = value_m / std_pt_to_m

                # Paper-size plausibility (same logic as first pass)
                plan_span_m_std = max(page_w_pt, page_h_pt) * std_pt_to_m
                paper_bonus = 0.0
                for pw_m, ph_m in _PAPER_SIZES_M:
                    for ppw, pph in [(pw_m, ph_m), (ph_m, pw_m)]:
                        if (0.7 <= physical_paper_w / ppw <= 1.3
                                and 0.7 <= physical_paper_h / pph <= 1.3):
                            if 8.0 <= plan_span_m_std <= 55.0:
                                paper_bonus = 0.3
                            elif 5.0 <= plan_span_m_std <= 70.0:
                                paper_bonus = 0.15
                            break
                    if paper_bonus > 0:
                        break
                scale_prior = 0.1 if sd == 100 else 0.0

                # Find closest geometry line to this expected length
                for g in all_geom:
                    if g.get("type") != "line":
                        continue
                    ga = g.get("a")
                    gb = g.get("b")
                    if not ga or not gb:
                        continue
                    gl = _seg_len(
                        (float(ga[0]), float(ga[1])),
                        (float(gb[0]), float(gb[1])),
                    )
                    if gl < 5.0:  # skip very short lines
                        continue
                    ratio = gl / expected_line_pt if expected_line_pt > 1e-6 else 100
                    if 0.8 <= ratio <= 1.2:  # within 20% of expected
                        geom_score = abs(math.log(max(ratio, 0.01))) * 0.5
                        score = geom_score - paper_bonus - scale_prior
                        if score < best_std_score:
                            best_std_score = score
                            best_std_pt_to_m = std_pt_to_m
                            best_std_info = f"1:{sd}"

        if best_std_pt_to_m is not None and best_std_score < 1.0:
            # Standard scale is a better fit
            logger.info(
                "Scale plausibility: inferred pt_to_m=%.4f gives page_span=%.1fm (implausible). "
                "Switching to standard scale %s (pt_to_m=%.5f, score=%.3f)",
                pt_to_m_real, page_span_m, best_std_info, best_std_pt_to_m, best_std_score,
            )
            pt_to_m_real = best_std_pt_to_m
            page_span_m = page_span_pt * pt_to_m_real
            conf *= 0.7  # reduce confidence since we overrode the measurement
            doc.setdefault("scale_debug", {})["plausibility_override"] = {
                "original_pt_to_m": float(candidates[0]["pt_to_m"]) if candidates else None,
                "original_page_span_m": float(page_span_pt * (candidates[0]["pt_to_m"] if candidates else 0)),
                "selected_scale": best_std_info,
                "selected_pt_to_m": float(best_std_pt_to_m),
                "corrected_page_span_m": float(page_span_m),
                "match_score": float(best_std_score),
            }

    tf = doc.get("transform") or {}
    tf["points_to_m"] = float(pt_to_m_real)
    tf["source"] = "measurement_infer_weighted"
    tf["from"] = "pdf_points_y_down"
    doc["transform"] = tf

    doc["scale"] = {
        "found": True,
        "ratio": None,
        "confidence": float(conf),
        "source_text_id": None,
        "method": "infer_from_measurements_weighted_median",
        "samples": int(len(used)),
        "samples_raw": int(len(candidates)),
        "rel_spread": float(rel_spread),
        "total_weight": float(total_w),
    }

    # Vollständige Debug-Infos zurückgeben
    doc["scale_debug"] = {
        "candidates": candidates,
        "used_ids": [c["measurement_id"] for c in used],
        "pt_to_m": float(pt_to_m_real),
    }

    if dimension_lines_only:
        tf["measure_lines_first"] = True

    extraction["doc"] = doc
    return extraction, True


def infer_points_to_m_from_measurements(
    extraction: Dict,
    detection_result: Optional[Dict] = None,
    px_to_coord_scale: float = 1.0,
) -> Dict:
    """
    Versucht, aus den erkannten Maßtexten + Linien einen Meter-pro-Punkt-Faktor
    zu bestimmen. Nutzt gewichtete Kandidaten:
    - starke Kandidaten (hoher Measurement- & OCR-Confidence, Dimension-Linie)
    - mittlere Kandidaten
    - schwache Kandidaten (Gewicht 0, d.h. nur Debug)
    """
    ext, _ = _infer_impl(
        extraction,
        detection_result=detection_result,
        px_to_coord_scale=px_to_coord_scale,
        dimension_lines_only=False,
    )
    return ext
