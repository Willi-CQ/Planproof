from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def _bbox_center(b: BBox) -> Point:
    x0, y0, x1, y1 = b
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def _collect_bounds(extraction: Dict) -> Optional[Tuple[float, float, float, float]]:
    xs: List[float] = []
    ys: List[float] = []

    for g in extraction.get("geometry", []):
        if g.get("type") != "line":
            continue
        a = g.get("a"); b = g.get("b")
        if not a or not b:
            continue
        xs += [float(a[0]), float(b[0])]
        ys += [float(a[1]), float(b[1])]

    for w in extraction.get("walls", []):
        a = w.get("a"); b = w.get("b")
        if not a or not b:
            continue
        xs += [float(a[0]), float(b[0])]
        ys += [float(a[1]), float(b[1])]

    for t in extraction.get("texts", []):
        bb = t.get("bbox")
        if not bb or len(bb) != 4:
            continue
        x0, y0, x1, y1 = map(float, bb)
        xs += [x0, x1]
        ys += [y0, y1]

    if not xs or not ys:
        return None

    return (min(xs), min(ys), max(xs), max(ys))


def write_debug_svg(
    extraction: Dict,
    out_svg_path: str,
    canvas_w: int = 1200,
    padding_px: int = 24,
    show_text_labels: bool = False,
    show_anchor_lines: bool = True,
) -> str:
    """
    Creates an SVG overlay showing:
      - all extracted geometry lines (gray)
      - dimension lines (red) via is_dimension_line
      - merged wall segments (green)
      - text bounding boxes (blue)
      - anchor links (orange) from measurement text -> anchored geometry midpoint
    Assumes extraction is already in meters y-up (after transform_extraction_to_m()).
    """
    out = Path(out_svg_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    bounds = _collect_bounds(extraction)
    if bounds is None:
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="120"><text x="10" y="60">No geometry/text to render.</text></svg>'
        out.write_text(svg, encoding="utf-8")
        return str(out)

    minx, miny, maxx, maxy = bounds
    # avoid zero area
    w = max(1e-6, maxx - minx)
    h = max(1e-6, maxy - miny)

    # keep aspect
    canvas_h = int(round(canvas_w * (h / w)))
    scale = (canvas_w - 2 * padding_px) / w
    # SVG y is down; our y is up -> invert using maxy
    def tx(x: float) -> float:
        return padding_px + (x - minx) * scale

    def ty(y: float) -> float:
        return padding_px + (maxy - y) * scale

    # Styling
    def line(x1, y1, x2, y2, stroke, sw, opacity=1.0, dash: Optional[str] = None):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"{dd} />'

    def rect(x, y, ww, hh, stroke, sw, opacity=1.0):
        return f'<rect x="{x:.2f}" y="{y:.2f}" width="{ww:.2f}" height="{hh:.2f}" fill="none" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}" />'

    def text(x, y, s, size=10, opacity=1.0):
        esc = (
            str(s).replace("&", "&amp;")
                  .replace("<", "&lt;")
                  .replace(">", "&gt;")
        )
        return f'<text x="{x:.2f}" y="{y:.2f}" font-family="monospace" font-size="{size}" fill="#111" opacity="{opacity}">{esc}</text>'

    # Build lookup for geometry by id
    geom_by_id = {g.get("id"): g for g in extraction.get("geometry", []) if g.get("type") == "line"}

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h + 90}">')
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white" />')

    # Legend
    parts.append(text(12, 18, "Plan Debug Overlay (meters, y-up)  |  gray=all lines, red=dimension lines, green=merged walls, blue=text bbox, orange=anchors", 12))
    # show scale info
    sc = (extraction.get("doc", {}).get("scale") or {})
    tr = (extraction.get("doc", {}).get("transform") or {})
    parts.append(text(12, 36, f"scale: {sc.get('ratio')} (conf={sc.get('confidence')}) | points_to_m={tr.get('points_to_m')}", 11, 0.9))

    # 1) Raw geometry lines
    for g in extraction.get("geometry", []):
        if g.get("type") != "line":
            continue
        a = g.get("a"); b = g.get("b")
        if not a or not b:
            continue
        is_dim = bool(g.get("is_dimension_line"))
        col = "#d11" if is_dim else "#999"
        op = 0.85 if is_dim else 0.35
        sw = 2.0 if is_dim else 1.0
        parts.append(line(tx(a[0]), ty(a[1]), tx(b[0]), ty(b[1]), col, sw, op))

    # 2) Merged wall segments
    for wseg in extraction.get("walls", []):
        a = wseg.get("a"); b = wseg.get("b")
        if not a or not b:
            continue
        parts.append(line(tx(a[0]), ty(a[1]), tx(b[0]), ty(b[1]), "#0a0", 2.5, 0.95))

    # 3) Text bounding boxes
    for tspan in extraction.get("texts", []):
        bb = tspan.get("bbox")
        if not bb or len(bb) != 4:
            continue
        x0, y0, x1, y1 = map(float, bb)
        rx = tx(x0)
        ry = ty(y1)  # because y inverted; top is y1 in our coords
        rw = (x1 - x0) * scale
        rh = (y1 - y0) * scale
        parts.append(rect(rx, ry, rw, rh, "#06c", 1.0, 0.6))

        if show_text_labels:
            parts.append(text(rx + 2, ry + 12, tspan.get("text", ""), 9, 0.7))

    # 4) Anchor lines: measurement text -> geometry midpoint
    if show_anchor_lines:
        texts_by_id = {t.get("id"): t for t in extraction.get("texts", []) if "id" in t}
        for m in extraction.get("measurements", []):
            anch = m.get("anchor") or {}
            if anch.get("type") != "near_geometry":
                continue
            tid = m.get("raw_text_id")
            gid = anch.get("geometry_id")
            tspan = texts_by_id.get(tid)
            g = geom_by_id.get(gid)
            if not tspan or not g:
                continue
            bb = tspan.get("bbox")
            a = g.get("a"); b = g.get("b")
            if not bb or not a or not b:
                continue

            p = _bbox_center(tuple(map(float, bb)))
            mid = ((float(a[0]) + float(b[0])) * 0.5, (float(a[1]) + float(b[1])) * 0.5)
            parts.append(line(tx(p[0]), ty(p[1]), tx(mid[0]), ty(mid[1]), "#f80", 1.2, 0.8, dash="4,3"))

    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")
    return str(out)
