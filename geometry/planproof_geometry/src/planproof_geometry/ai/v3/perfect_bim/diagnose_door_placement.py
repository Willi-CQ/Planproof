#!/usr/bin/env python3
"""V3 Doors Sprint — 2D placement diagnose mit Willi-Algorithmus.

Vier Schritte aus Willi 2026-04-26:
  1. BBox-Algorithmus -> flanking walls per Bbox-Distance (nicht Centroid).
  2. Pixel-Extraktion am bbox-Standort -> dominantes Linien-Cluster im
     gerenderten PDF.
  3. Pixel -> Tuermodell: Rechteck aus Linien-Cluster, Bogen verworfen.
  4. Verschiebung + Skalierung: Tuer skaliert auf gap_length, zentriert
     auf gap_mid, ausgerichtet an Wand-Achse. Walls bleiben unangetastet.

Plus zwei Robustheits-Fixes:
  - wall_depth lokal: nur Wand-Vertices innerhalb door-extent um die bbox
    werden in PCA-Minor einbezogen, damit L/T-Polygone nicht 3 m tiefe
    Tueren produzieren.
  - gap_length Fallback: wenn gap_geometry gap_length=0 liefert (z.B.
    Klasse-A-Doors deren bbox-Achse-Projektion das Wand-Gap nicht
    schneidet), nehmen wir die bbox-Extent entlang der Wand-Achse als
    Tuerbreite.

Output PNG zeigt:
  - Wand-Polygone grau
  - finaler Tuer-Rechteck orange (Primary: scaled-to-gap)
  - pixel-extrahiertes Tuer-Rechteck cyan transparent (Vergleich)
  - bbox-Outline duenn blau
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Reconfigure stdout/stderr to UTF-8 so Unicode in docstrings (>=,
# arrow chars, etc.) doesn't crash the script when output is piped to
# a file on Windows (default cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2
import numpy as np
from shapely.geometry import Point as ShapelyPoint, Polygon, box

from planproof_geometry.ai.cv_compat import hough_lines_p
from planproof_geometry.ai.v3.pixel_bim.extrude import (
    WallPolygon,
    extrude_wall_mask,
)
from planproof_geometry.ai.v3.pixel_bim.openings.door_models import (
    PlacedDoor,
    build_beside_wall_sliding_polygon,
    build_door_meshes_grouped,
    build_door_rectangle,
    build_door_rectangle_from_polygon,
    build_sliding_door_polygon,
    filter_for_class,
)
from planproof_geometry.ai.v3.pixel_bim.openings.wall_scan import (  # noqa: F401
    scan_wall_4dir,
)
from planproof_geometry.ai.v3.pixel_bim.openings.door_voids import (  # noqa: F401
    DoorVoid,
    apply_door_voids,
)
from planproof_geometry.ai.v3.pixel_bim.openings.wall_axis import (  # noqa: F401
    measure_wall_axis_at_door,
)
from planproof_geometry.ai.v3.pixel_bim.openings.height_bands import (  # noqa: F401
    resolve_heights,
)
from planproof_geometry.ai.v3.pixel_bim.openings.door_models import (  # noqa: F401
    compute_bridge_polygons_m,
)


PDF_PATH = Path(
    "C:/Users/deepn/OneDrive/Desktop/Planproof_Projects/Test"
    "/Zeichnungen/Matterport Sample_BW.pdf"
)
ANN_PATH = Path(
    "C:/Users/deepn/OneDrive/Desktop/Planproof_Projects/Test"
    "/plan_bim_data/Matterport Sample_BW/annotations.json"
)
OUT = Path("C:/tmp/debug_pipeline_out")

WALL_HEIGHT_M = 2.80
RENDER_MATRIX = 4.0

WALL_CLASSES = {"wall", "wall_interior", "wall_exterior"}
DOOR_CLASSES = {"door", "door_single", "door_double", "door_sliding"}

FLANK_W1_MAX_M = 0.50
FLANK_W2_MAX_M = 1.50
# Wider search radius for the corridor detector (Tag-3): two walls
# perpendicular to the panel pixel-line that flank a small room (bath / WC)
# may sit just past the W2 pair-cap when the room is up to ~2.5 m wide.
FLANK_CORRIDOR_MAX_M = 2.50

# Tag-6.10 (Willi 2026-04-28): split residential-DIN expectations from
# the algorithmic plausibility band. The algorithm runs against the
# wider band so industrial/special-purpose buildings still get a real
# measurement; the DIN band only drives a flag on the placement.
ALGO_THICKNESS_MAX_M = 0.50    # wall thicker than 50 cm is suspicious
DIN_THICKNESS_MAX_M = 0.25     # residential DIN guidance, used as flag
DOOR_DEPTH_FLOOR_M = 0.05
DOOR_DEPTH_DEFAULT_M = 0.12   # ~ Matterport interior wall stud thickness

# Tag-6.10: door symbol radius bands.
#   ALGO band — the RANSAC fit is rejected outside this (no real door
#   symbol is < 30 cm wide; > 3 m means RANSAC is fitting near-collinear
#   strokes as a quasi-circle, not an arc).
#   DIN band — the per-class DIN 18101 single / double / sliding range
#   used purely for flagging: a fit inside the algorithmic band but
#   outside its class-DIN range still passes, but gets a non_din_radius
#   flag on the placement so Willi sees it.
ALGO_DOOR_RADIUS_M = (0.30, 3.00)
_DIN_DOOR_RADIUS_RANGE_M: Dict[str, Tuple[float, float]] = {
    "door":         (0.60, 1.20),
    "door_single":  (0.65, 1.10),
    "door_double":  (1.20, 2.00),
    "door_sliding": (0.40, 1.60),
}
# Tag-6.10: gap detection bounds. ALGO_GAP_MIN is true sanity (under
# 30 cm a "gap" is not even a door opening); ALGO_GAP_MAX widened from
# DIN's 1.80 to 3.0 m so industrial / large openings detect cleanly,
# with a flag instead of a filter on out-of-DIN widths.
ALGO_GAP_MIN_M = 0.30
ALGO_GAP_MAX_M = 3.00
DIN_GAP_MAX_M = 1.80


@dataclass
class DoorPlacement:
    label_id: str
    label_class: str
    bbox_m: Tuple[float, float, float, float]
    w1_idx: int
    w2_idx: Optional[int]
    w1_distance_m: float
    w2_distance_m: Optional[float]
    door_width_m: float = 0.0
    door_depth_m: float = 0.0
    door_polygon_xy: List[Tuple[float, float]] = field(default_factory=list)
    pixel_polygon_xy: List[Tuple[float, float]] = field(default_factory=list)
    pixel_width_m: float = 0.0
    width_source: str = ""
    note: str = ""
    # Validation against label bbox + flanking walls (Willi 2026-04-27).
    inside_bbox: bool = False
    on_wall: bool = False
    inside_bbox_pct: float = 0.0
    wall_overlap_pct: float = 0.0
    pass_validation: bool = False
    validation_note: str = ""
    # Tag-6.10: DIN deviation flags. Algorithm decisions are unaffected
    # by these — they record where the placement breaks DIN expectations
    # (radius outside class range, thick wall, oversize gap, etc.) so
    # Willi can see DIN deviations without the algorithm filtering them.
    din_flags: List[str] = field(default_factory=list)
    # Tag-7.7: hinge-end voting (Idea 5). The four optional anchors below
    # are populated by place_door at every available point in the cascade
    # and read by ``assign_hinge_voting`` to derive the per-door
    # ``hinge_long_sign``. Storing the raw anchors (not just the votes)
    # keeps the voting logic in one place and makes it auditable.
    arc_hinge_xy_m: Optional[Tuple[float, float]] = None
    panel_hinge_xy_m: Optional[Tuple[float, float]] = None
    hinge_long_sign: Optional[int] = None
    hinge_confidence: float = 0.0
    hinge_votes: Dict[str, int] = field(default_factory=dict)
    # Tag-7.8: hard criterion — door must sit in an actual wall gap,
    # not in solid wall material. ``gap_overlap_pct`` is the fraction of
    # the door's wall-tangent extent where the wall pixel profile is
    # below ``GAP_PROFILE_THRESHOLD_FACTOR × baseline`` (i.e. the wall
    # outline is ABSENT — a real opening). 0.0 = entirely in solid
    # wall, 1.0 = entirely in gap. Below ``GAP_OVERLAP_HARD_MIN`` the
    # placement is rejected (``pass_validation`` flipped to False).
    gap_overlap_pct: float = 1.0
    # Tag-7.9: pixel-symbol anchor data. Captured at extract_pixel_door
    # time so ``attempt_pixel_recovery`` can rebuild the door rectangle
    # centred on the panel symbol (rather than on the bbox extent) when
    # the bbox-fallback path lands on solid wall.
    pixel_panel_center_xy_m: Optional[Tuple[float, float]] = None
    pixel_panel_axis_xy: Optional[Tuple[float, float]] = None
    # Tag-7.10: number of long-axis end-flanks with wall material
    # within snap distance. 2 = correctly framed by walls on both
    # sides; 1 = end-of-wall (e.g. closet sliders); 0 = floating in
    # open space (placement is suspect even if gap_overlap clears).
    jamb_anchors: int = 0
    # Tag-7.11: per-side anchor flags so the renderer can paint each
    # end-pin individually and Willi can see WHICH side floats. Order
    # matches ``_polygon_basis`` long-axis sign: minus=-W/2 end,
    # plus=+W/2 end.
    jamb_anchor_minus: bool = False
    jamb_anchor_plus: bool = False
    # Sprint Door-Wall-Bridge Tag-2: mesh-derived wall axis attrs.
    # ``wall_depth_mesh_m`` is the actual wall thickness measured from
    # the post-refine BIM polygon (ray-cast perpendicular to the door
    # axis). ``short_offset_m`` is the signed offset of the wall
    # centerline from the door polygon centre along +short_axis. Both
    # are 0.0 until ``assign_door_wall_axis_attrs`` runs.
    wall_depth_mesh_m: float = 0.0
    short_offset_m: float = 0.0
    # Sprint Sliding-Door-Rewrite Tag-3: 4-dir scan results captured at
    # placement time. When ``sliding_pocket_sign`` is not None the
    # downstream BIM viewer uses these directly instead of re-running
    # classify_sliding_mount (which would otherwise consume the raw
    # user-wall annotations and miss the new wall geometry).
    sliding_pocket_sign: Optional[int] = None
    sliding_mount_type: Optional[str] = None
    sliding_extent_long_pos_m: float = 0.0
    sliding_extent_long_neg_m: float = 0.0
    # Sprint Sliding-Door-In-Wall-Classifier Tag-3: mount_classifier
    # routing decision. ``"in_wall"`` = polygon sits on wall midplane
    # (build_sliding_door_polygon path, void carved through wall),
    # ``"beside_wall"`` = polygon at bbox centre beside the wall
    # (build_beside_wall_sliding_polygon path, NO wall void carved —
    # panel is a standalone barn-door element). ``None`` = legacy path
    # (wall_ctx.confidence-based routing, e.g. when the classifier
    # abstained and the cascade fallback ran).
    sliding_mount_position: Optional[str] = None


# ── Helpers ──────────────────────────────────────────────────────────────


def _render_pdf_grayscale(pdf_path: Path, matrix: float) -> np.ndarray:
    # pdfium statt PyMuPDF (AGPL), 14.09.2026 - gleiche Groesse, siehe planproof_geometry.pdf.
    from planproof_geometry.pdf import PdfDatei

    with PdfDatei(pdf_path) as pdf:
        # Issue #47: die Seitengeometrie kommt AUS DER DATEI. Mit festem Faktor
        # 4,0 ergab eine praeparierte MediaBox von 10800 x 10800 pt ein Bild von
        # 43200 x 43200 Pixeln — und der Render-Cache haelt es fest, waehrend
        # jeder der drei Aufrufer eine .copy() bekommt.
        from ...pdf_render_budget import fit_zoom

        matrix = fit_zoom(*pdf.seitengroesse(0), matrix)
        return pdf.rendern_grau(0, matrix)


def _calibrate_px_to_m_from_doors(annotations, scale_xy):
    sx, sy = scale_xy
    widths_px: List[float] = []
    for a in annotations:
        if (a.get("class") or "") != "door_single":
            continue
        b = a.get("bbox")
        if not b or len(b) != 4:
            continue
        x0, y0, x1, y1 = b
        widths_px.append(min(abs((x1 - x0) * sx), abs((y1 - y0) * sy)))
    if len(widths_px) < 3:
        return None
    widths_px.sort()
    median = widths_px[len(widths_px) // 2]
    if median <= 5:
        return None
    return 0.85 / median


def _poly_from_annotation(ann: Dict, scale: Tuple[float, float]
                          ) -> List[Tuple[float, float]]:
    sx, sy = scale
    if ann.get("type") == "polygon":
        pts = ann.get("points") or []
        return [(float(p[0]) * sx, float(p[1]) * sy) for p in pts]
    bbox = ann.get("bbox")
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = bbox
        return [
            (x0 * sx, y0 * sy), (x1 * sx, y0 * sy),
            (x1 * sx, y1 * sy), (x0 * sx, y1 * sy),
        ]
    return []


def _bbox_from_poly_m(poly_m: Sequence[Tuple[float, float]]
                      ) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly_m]
    ys = [p[1] for p in poly_m]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_in_pixels(label_bbox_m: Tuple[float, float, float, float],
                    px_to_m: float
                    ) -> Tuple[float, float, float, float]:
    return (
        label_bbox_m[0] / px_to_m, label_bbox_m[1] / px_to_m,
        label_bbox_m[2] / px_to_m, label_bbox_m[3] / px_to_m,
    )


def _poly_to_m(poly_px: List[Tuple[float, float]],
               px_to_m: float) -> List[Tuple[float, float]]:
    return [(x * px_to_m, y * px_to_m) for x, y in poly_px]


# ── Step 1 — flanking walls by bbox-distance ────────────────────────────


def find_flanking_walls_by_bbox(
    bbox_m: Tuple[float, float, float, float],
    wall_polygons: Sequence[WallPolygon],
) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
    bbox_poly = box(*bbox_m)
    distances: List[Tuple[int, float]] = []
    for i, wp in enumerate(wall_polygons):
        if len(wp.shell_xy) < 3:
            continue
        try:
            d = Polygon(wp.shell_xy).distance(bbox_poly)
        except Exception:
            continue
        distances.append((i, d))
    if not distances:
        return None, None, None, None
    distances.sort(key=lambda t: t[1])
    w1_idx, w1_d = distances[0]
    if w1_d > FLANK_W1_MAX_M:
        return None, None, None, None
    if len(distances) < 2 or distances[1][1] > FLANK_W2_MAX_M:
        return w1_idx, None, w1_d, None
    return w1_idx, distances[1][0], w1_d, distances[1][1]


def find_flanking_walls_ranked(
    bbox_m: Tuple[float, float, float, float],
    wall_polygons: Sequence[WallPolygon],
    *,
    max_distance_m: float = 1.50,
) -> List[Tuple[int, float]]:
    """Return ALL walls within ``max_distance_m`` of bbox, sorted ascending.

    Used for orientation-aware wall selection where the closest wall is
    not necessarily the wall the door belongs to (e.g. doors at an
    L-corner with two perpendicular walls equidistant — only one is the
    panel-hinge wall).
    """
    bbox_poly = box(*bbox_m)
    out: List[Tuple[int, float]] = []
    for i, wp in enumerate(wall_polygons):
        if len(wp.shell_xy) < 3:
            continue
        try:
            d = Polygon(wp.shell_xy).distance(bbox_poly)
        except Exception:
            continue
        if d <= max_distance_m:
            out.append((i, d))
    out.sort(key=lambda t: t[1])
    return out


def select_start_wall_by_panel_endpoint(
    wall_polygons: Sequence[WallPolygon],
    candidate_indices: Sequence[Tuple[int, float]],
    panel_e1: np.ndarray,
    panel_e2: np.ndarray,
    *,
    perp_threshold: float = 0.85,
    max_endpoint_dist_m: float = 0.25,
) -> Optional[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Willi's algorithm (2026-04-27) — identify the START wall (hinge wall).

    A 90°-open door symbol's panel pixel-line goes off perpendicularly
    FROM one wall (the hinge wall = "Anfangswand"). Its near endpoint
    touches that wall's surface; the other endpoint is in the room (or
    on a second "Endwand"). So the START wall is the wall whose
    nearest-long-edge tangent is perpendicular to the panel direction
    AND on whose surface one panel endpoint lies (within
    ``max_endpoint_dist_m``).

    Returns ``(start_wall_idx, hinge_endpoint, free_endpoint, origin,
    tangent)`` or ``None``.
      - ``hinge_endpoint`` — the panel endpoint sitting on the wall.
      - ``free_endpoint``  — the other endpoint (toward the room / end wall).
      - ``origin`` — perpendicular foot of ``hinge_endpoint`` on the wall
        edge (= door's anchor point on the wall surface).
      - ``tangent`` — wall-surface direction at the hinge.
    """
    panel_dir = panel_e2 - panel_e1
    panel_len = float(np.linalg.norm(panel_dir))
    if panel_len < 1e-6:
        return None
    panel_unit = panel_dir / panel_len
    panel_mid = 0.5 * (panel_e1 + panel_e2)

    best_idx = -1
    best_d = max_endpoint_dist_m
    best_hinge_ep: Optional[np.ndarray] = None
    best_free_ep: Optional[np.ndarray] = None
    best_origin: Optional[np.ndarray] = None
    best_tang: Optional[np.ndarray] = None

    for idx, _dist in candidate_indices:
        wp = wall_polygons[idx]
        if len(wp.shell_xy) < 3:
            continue
        origin, tang, _ = nearest_edge_axis(wp, panel_mid)
        cross = abs(
            float(tang[0] * panel_unit[1] - tang[1] * panel_unit[0])
        )
        if cross < perp_threshold:
            continue
        try:
            wall_poly = Polygon(wp.shell_xy).buffer(0)
        except Exception:
            continue
        if not wall_poly.is_valid or wall_poly.is_empty:
            continue
        d1 = float(wall_poly.distance(
            __import__("shapely.geometry", fromlist=["Point"]).Point(
                float(panel_e1[0]), float(panel_e1[1])
            )
        ))
        d2 = float(wall_poly.distance(
            __import__("shapely.geometry", fromlist=["Point"]).Point(
                float(panel_e2[0]), float(panel_e2[1])
            )
        ))
        d_min = min(d1, d2)
        if d_min < best_d:
            best_d = d_min
            best_idx = idx
            if d1 <= d2:
                best_hinge_ep = panel_e1
                best_free_ep = panel_e2
            else:
                best_hinge_ep = panel_e2
                best_free_ep = panel_e1
            # Re-anchor origin at the perpendicular foot of the HINGE
            # endpoint on this wall's nearest long edge.
            re_origin, re_tang, _ = nearest_edge_axis(wp, best_hinge_ep)
            best_origin = re_origin
            best_tang = re_tang

    if best_idx < 0 or best_hinge_ep is None or best_origin is None or best_tang is None:
        return None
    return best_idx, best_hinge_ep, best_free_ep, best_origin, best_tang


def detect_corridor_between_walls(
    wall_polygons: Sequence[WallPolygon],
    candidates: Sequence[Tuple[int, float]],
    bbox_center: np.ndarray,
    pixel_axis: Optional[Tuple[float, float]],
    *,
    perp_threshold: float = 0.85,
    parallel_threshold: float = 0.95,
    min_gap_m: float = 0.30,
) -> Optional[Tuple[int, int, float, float, np.ndarray]]:
    """Detect the **corridor** geometry: door fits in the gap *between* two
    walls that both run perpendicular to ``pixel_axis``.

    Tag-3 (Willi 2026-04-27): for some doors the panel pixel-line is
    parallel to the door axis (the symbol shows a closed door between two
    parallel walls), not perpendicular to it. In that case the door does
    not cut through any single wall — it sits in the gap between the two
    walls, with axis = ``pixel_axis``.

    Returns ``(wall_pos_idx, wall_neg_idx, gap_length_m, gap_mid_offset,
    pixel_axis_unit)`` where:
      * Both walls' nearest-long-edge tangents are perpendicular to
        ``pixel_axis`` (cross >= ``perp_threshold``).
      * Their perpendicular-foot points on the bbox centre straddle the
        bbox along ``pixel_axis``: one is in ``+pixel_axis`` direction,
        the other in ``-pixel_axis`` direction.
      * The two walls are roughly parallel to each other
        (cos angle between tangents >= ``parallel_threshold``).
      * Gap length (signed distance along pixel_axis between the two
        feet) >= ``min_gap_m``.

    ``gap_mid_offset`` is the signed distance from ``bbox_center`` to the
    gap midpoint along ``pixel_axis_unit``.
    """
    from shapely.ops import nearest_points
    from shapely.geometry import Point

    if pixel_axis is None or len(candidates) < 2:
        return None
    pix = np.asarray(pixel_axis, dtype=np.float64)
    n_pix = float(np.linalg.norm(pix))
    if n_pix < 1e-6:
        return None
    pix_unit = pix / n_pix

    bbox_pt = Point(float(bbox_center[0]), float(bbox_center[1]))

    perp_walls: List[Tuple[int, np.ndarray, np.ndarray, float]] = []
    for idx, _dist in candidates:
        wp = wall_polygons[idx]
        if len(wp.shell_xy) < 3:
            continue
        _, tang, _ = nearest_edge_axis(wp, bbox_center)
        cross = abs(float(tang[0] * pix_unit[1] - tang[1] * pix_unit[0]))
        if cross < perp_threshold:
            continue
        # Signed offset along pixel_axis from bbox_center to the closest
        # point on the wall polygon (much more robust than the foot on the
        # nearest-long-edge for L/T-shaped Phase-2 polygons where the long
        # edge is far from the part of the wall touching the bbox).
        try:
            wall_shp = Polygon(wp.shell_xy).buffer(0)
            if not wall_shp.is_valid or wall_shp.is_empty:
                continue
            np_pt, _ = nearest_points(wall_shp, bbox_pt)
            nearest_xy = np.array([np_pt.x, np_pt.y], dtype=np.float64)
        except Exception:
            continue
        signed = float((nearest_xy - bbox_center) @ pix_unit)
        perp_walls.append((idx, nearest_xy, tang, signed))

    if len(perp_walls) < 2:
        return None

    pos_walls = [pw for pw in perp_walls if pw[3] > 0.0]
    neg_walls = [pw for pw in perp_walls if pw[3] < 0.0]
    if not pos_walls or not neg_walls:
        return None

    pos_walls.sort(key=lambda t: t[3])           # smallest +offset first
    neg_walls.sort(key=lambda t: -t[3])          # largest (closest-to-0) first
    pos = pos_walls[0]
    neg = neg_walls[0]

    cos_par = abs(float(pos[2] @ neg[2]))
    if cos_par < parallel_threshold:
        return None

    gap_length = float(pos[3] - neg[3])
    if gap_length < min_gap_m:
        return None

    gap_mid_offset = 0.5 * (pos[3] + neg[3])
    return pos[0], neg[0], gap_length, gap_mid_offset, pix_unit


def select_hinge_wall_by_panel(
    wall_candidates: Sequence[Tuple[int, float]],
    wall_polygons: Sequence[WallPolygon],
    bbox_center: np.ndarray,
    pixel_axis: Optional[Tuple[float, float]],
    *,
    perp_threshold: float = 0.85,
    distance_floor_m: float = 0.05,
) -> Optional[int]:
    """Pick the wall the door is hinged on, using the door panel direction.

    A 90°-open door symbol shows the panel perpendicular to the hinge
    wall. ``pixel_axis`` (from HoughLinesP on the longest non-wall line in
    the bbox) is therefore perpendicular to the hinge wall's tangent.

    Strategy:
      1. Compute |sin(angle)| between each candidate wall's nearest-long-
         edge tangent and ``pixel_axis``. Score 1.0 = perpendicular.
      2. If the BEST score is ≥ ``perp_threshold`` AND it beats the
         second-best by ≥ 0.10, pick it. Otherwise return None (no clear
         winner — fall back to the closest wall).

    The closest wall is preferred when scores are similar to avoid
    flipping in ambiguous cases.
    """
    if pixel_axis is None or not wall_candidates:
        return None
    pix = np.array(pixel_axis, dtype=np.float64)
    n = float(np.linalg.norm(pix))
    if n < 1e-6:
        return None
    pix /= n

    scored: List[Tuple[int, float, float]] = []
    for idx, dist in wall_candidates:
        _, tang, _ = nearest_edge_axis(
            wall_polygons[idx], bbox_center
        )
        cross = abs(float(tang[0] * pix[1] - tang[1] * pix[0]))
        scored.append((idx, dist, cross))
    scored.sort(key=lambda t: -t[2])
    best = scored[0]
    second_best_score = scored[1][2] if len(scored) >= 2 else 0.0
    if best[2] >= perp_threshold and (best[2] - second_best_score) >= 0.10:
        return best[0]
    return None


# ── Wall axis / local thickness ─────────────────────────────────────────


def _pca_axes(points: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    _eig, vecs = np.linalg.eigh(cov)
    return points.mean(axis=0), vecs[:, -1], vecs[:, 0]


def wall_axis(wall_polygon: WallPolygon
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(wall_polygon.shell_xy, dtype=np.float64)
    return _pca_axes(pts)


def local_wall_axis(
    wall_polygon: WallPolygon,
    bbox_center_m: np.ndarray,
    *,
    window_m: float = 1.50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA on the wall vertices within ``window_m`` of the bbox centre.

    Phase-2 polygons are often L/T-shaped because morphological closing
    merges perpendicular wall pieces. The whole-polygon PCA major axis is
    the principal direction of the L, not the local wall direction. Doors
    placed against this skewed axis end up tilted. Restricting PCA to the
    window around the bbox returns the local wall axis.
    """
    pts = np.asarray(wall_polygon.shell_xy, dtype=np.float64)
    if pts.shape[0] < 2:
        return _pca_axes(pts) if pts.shape[0] >= 2 else (
            np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0])
        )
    distances = np.linalg.norm(pts - bbox_center_m, axis=1)
    near_mask = distances <= window_m
    near = pts[near_mask] if near_mask.sum() >= 2 else pts
    return _pca_axes(near)


_LONG_EDGE_MIN_M = 0.30


def nearest_edge_axis(
    wall_polygon: WallPolygon,
    bbox_center_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wall axis from the LONG polygon edge closest to ``bbox_center_m``.

    Phase-2 wall polygons often look like an "L" because morphological
    closing merged two perpendicular wall pieces. PCA — even windowed —
    can pick the diagonal of the L as major.

    Polygon edges fall into two groups:
      * LONG edges = the wall surface lines (room-side and outside-side,
        each running the length of the wall). These give the wall axis.
      * SHORT edges = end caps (wall-thickness segments at the ends of
        the wall, perpendicular to the wall direction).
    For doors near the end of a wall (between-walls case) the SHORT end
    cap is geometrically closer than either long edge, so an unfiltered
    nearest-edge search would return a tangent perpendicular to the
    wall. We restrict to edges with length ≥ ``_LONG_EDGE_MIN_M``.

    Returns ``(origin, major_unit, minor_unit)`` where ``origin`` is the
    foot of the perpendicular from ``bbox_center_m`` on the chosen edge.
    """
    pts = np.asarray(wall_polygon.shell_xy, dtype=np.float64)
    n = pts.shape[0]
    if n < 2:
        return (np.array(bbox_center_m, dtype=np.float64),
                np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    best_d2 = float("inf")
    best_foot = bbox_center_m.astype(np.float64)
    best_tangent = np.array([1.0, 0.0])
    fallback_d2 = float("inf")
    fallback_foot = best_foot.copy()
    fallback_tangent = best_tangent.copy()
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        ab = b - a
        seg_len2 = float(ab @ ab)
        if seg_len2 < 1e-9:
            continue
        seg_len = math.sqrt(seg_len2)
        t = float((bbox_center_m - a) @ ab / seg_len2)
        t = max(0.0, min(1.0, t))
        foot = a + t * ab
        d2 = float((foot - bbox_center_m) @ (foot - bbox_center_m))
        if seg_len >= _LONG_EDGE_MIN_M and d2 < best_d2:
            best_d2 = d2
            best_foot = foot
            best_tangent = ab / seg_len
        if d2 < fallback_d2:
            fallback_d2 = d2
            fallback_foot = foot
            fallback_tangent = ab / seg_len

    if best_d2 == float("inf"):
        # No long edge — degenerate polygon, use the global closest edge.
        best_foot = fallback_foot
        best_tangent = fallback_tangent

    major = best_tangent
    minor = np.array([-major[1], major[0]])
    return best_foot, major, minor


def local_wall_depth(
    wall_polygon: WallPolygon,
    bbox_center_m: np.ndarray,
    major: np.ndarray,
    minor: np.ndarray,
    origin: np.ndarray,
    *,
    door_extent_m: float,
    wall_mask: Optional[np.ndarray] = None,
    px_to_m: float = 1.0,
) -> float:
    """Wall thickness at the door position, in metres.

    Strategy:
      1. If ``wall_mask`` is provided, probe the rasterised mask directly:
         cv2.distanceTransform → max distance inside the wall around
         ``bbox_center_m`` × 2 = wall thickness. This is independent of the
         L/T-shape of the Phase-2 polygon and gives the true pixel-derived
         wall thickness at the door location.
      2. Otherwise fall back to a windowed PCA-minor extent on the polygon.
    Result is clamped to [DOOR_DEPTH_FLOOR_M, ALGO_THICKNESS_MAX_M] —
    the upper bound is algorithmic plausibility (50 cm), not the DIN
    residential cap (25 cm), so industrial walls measure correctly.
    The DIN deviation is recorded as a flag in place_door, not here.
    """
    if wall_mask is not None and px_to_m > 0:
        H, W = wall_mask.shape
        cx_px = int(round(bbox_center_m[0] / px_to_m))
        cy_px = int(round(bbox_center_m[1] / px_to_m))
        # Search window: 0.6 m around the bbox centre, clipped to image.
        win_px = max(8, int(round(0.30 / px_to_m)))
        x0 = max(0, cx_px - win_px)
        y0 = max(0, cy_px - win_px)
        x1 = min(W, cx_px + win_px)
        y1 = min(H, cy_px + win_px)
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            sub = wall_mask[y0:y1, x0:x1]
            if sub.any():
                dt = cv2.distanceTransform(
                    (sub > 0).astype(np.uint8) * 255,
                    cv2.DIST_L2, 5,
                )
                # Half-thickness ≈ max distance from inside-wall to the nearest
                # background pixel. Full thickness ≈ 2 × half. Cap.
                half_px = float(dt.max())
                if half_px > 0:
                    extent = 2.0 * half_px * px_to_m
                    extent = max(extent, DOOR_DEPTH_FLOOR_M)
                    extent = min(extent, ALGO_THICKNESS_MAX_M)
                    return extent

    pts = np.asarray(wall_polygon.shell_xy, dtype=np.float64)
    if pts.shape[0] < 2:
        return DOOR_DEPTH_DEFAULT_M
    t_proj = (pts - origin) @ major
    t_center = float((bbox_center_m - origin) @ major)
    half_window = max(0.60, door_extent_m) * 0.5 + 0.10
    near_mask = np.abs(t_proj - t_center) <= half_window
    near = pts[near_mask] if near_mask.any() else pts
    s_proj = (near - origin) @ minor
    extent = float(s_proj.max() - s_proj.min())
    extent = max(extent, DOOR_DEPTH_FLOOR_M)
    extent = min(extent, ALGO_THICKNESS_MAX_M)
    return extent


# ── Step 2/3 — pixel extraction ─────────────────────────────────────────


@dataclass
class DoorArc:
    """Door symbol = arc + straight panel line. The arc gives us:
      - centre = hinge (on the wall surface)
      - radius = door width
      - the arc-endpoint nearest a wall pixel = the closed-position tip
        (i.e. the door's tip when shut, lying along the wall axis)
      - the other endpoint = open-position tip (in the room)
    Door axis (wall-parallel) = direction from hinge to closed-endpoint.
    """
    hinge_xy_m: Tuple[float, float]
    radius_m: float
    closed_xy_m: Tuple[float, float]
    open_xy_m: Tuple[float, float]
    n_inliers: int
    arc_coverage: float  # inliers / (2*pi*r) — how much of a full circle
    din_flags: List[str] = field(default_factory=list)


def detect_door_arc(
    image_gray: np.ndarray,
    bbox_px: Tuple[float, float, float, float],
    wall_mask: np.ndarray,
    *,
    px_to_m: float,
    label_class: str = "door",
    pixel_door: Optional["PixelDoor"] = None,
    neighbour_bboxes_px: Optional[Sequence[Tuple[float, float, float, float]]] = None,
) -> Optional[DoorArc]:
    """RANSAC circle fit STRICTLY inside the label bbox.

    Validation:
      - centre (hinge) inside the crop
      - coverage >= 0.10 (>= 36 deg of arc)
      - radius inside the class-specific DIN range
      - closed-endpoint < 5 px (~ wall thickness) from a wall pixel
      - if ``pixel_door`` is supplied, cross-validate:
          * arc radius within 30 % of pixel_door panel length
          * arc hinge within 0.10 m of either pixel_door panel endpoint
    """
    from skimage.measure import CircleModel, ransac

    H, W = image_gray.shape
    x0 = int(max(0, math.floor(bbox_px[0])))
    y0 = int(max(0, math.floor(bbox_px[1])))
    x1 = int(min(W, math.ceil(bbox_px[2])))
    y1 = int(min(H, math.ceil(bbox_px[3])))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None

    crop = image_gray[y0:y1, x0:x1]
    crop_wall = wall_mask[y0:y1, x0:x1]

    _, binary = cv2.threshold(crop, 200, 255, cv2.THRESH_BINARY_INV)

    # Remove wall pixels (dilate slightly so wall edges are excluded too).
    kernel = np.ones((3, 3), np.uint8)
    dilated_wall = cv2.dilate(crop_wall, kernel, iterations=2)
    door_pixels = cv2.bitwise_and(binary, cv2.bitwise_not(dilated_wall))

    # Tag-6.8: subtract pixels inside any NEIGHBOUR-door bbox PLUS a
    # 15 cm padding. In dense clusters (Bath/WC) where 3 door bboxes
    # sit within a few cm of each other, the neighbours' arc symbols
    # bulge OUTSIDE their bbox into our crop — bbox-only subtraction
    # leaves those bulge pixels behind and RANSAC fits them as the
    # wrong door's arc. The padding catches arc strokes that extend
    # up to ~half a typical door radius outside the neighbour bbox,
    # while staying small enough not to eat into the current door's
    # own symbol when bboxes are well-separated.
    if neighbour_bboxes_px:
        H_crop, W_crop = door_pixels.shape
        nb_pad_px = max(2, int(round(0.15 / px_to_m)))
        nb_mask = np.zeros((H_crop, W_crop), dtype=np.uint8)
        for nbx in neighbour_bboxes_px:
            nx0 = int(max(0, math.floor(nbx[0]) - x0 - nb_pad_px))
            ny0 = int(max(0, math.floor(nbx[1]) - y0 - nb_pad_px))
            nx1 = int(min(W_crop, math.ceil(nbx[2]) - x0 + nb_pad_px))
            ny1 = int(min(H_crop, math.ceil(nbx[3]) - y0 + nb_pad_px))
            if nx1 > nx0 and ny1 > ny0:
                nb_mask[ny0:ny1, nx0:nx1] = 255
        if nb_mask.any():
            door_pixels = cv2.bitwise_and(
                door_pixels, cv2.bitwise_not(nb_mask)
            )

    ys, xs = np.where(door_pixels > 0)
    _DBG_HEAD = __import__("os").environ.get("DOOR_ARC_DEBUG") == "1"
    if _DBG_HEAD:
        print(f"  [arc-dbg] {label_class} crop {x1-x0}x{y1-y0} px, "
              f"door_pixels={len(xs)}")
    if len(xs) < 25:
        if _DBG_HEAD:
            print(f"  [arc-dbg] only {len(xs)} door_pixels (<25)")
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64)

    # Tag-6.10 algorithmic radius gate. The fit is rejected if the
    # circle radius falls outside the algorithmic plausibility band
    # (0.30 - 3.00 m). Below 0.30 m we are not looking at a door
    # symbol; above 3.00 m RANSAC is fitting near-collinear strokes
    # to a quasi-circle. The class-specific DIN band only generates
    # a flag for the placement record (per Willi's directive: DIN is
    # advisory, the algorithm has authority).
    algo_r_min_m, algo_r_max_m = ALGO_DOOR_RADIUS_M
    min_r_px = max(8.0, algo_r_min_m / px_to_m)
    max_r_px = algo_r_max_m / px_to_m
    din_r_min_m, din_r_max_m = _DIN_DOOR_RADIUS_RANGE_M.get(
        label_class, _DIN_DOOR_RADIUS_RANGE_M["door"]
    )

    _DBG = __import__("os").environ.get("DOOR_ARC_DEBUG") == "1"
    # Tag-6.0: multi-seed RANSAC. With min_samples=3 and ~4-10k mixed
    # arc/panel/frame pixels, a single seed can land on a degenerate
    # near-collinear sample and fit a huge fake circle (r > 1.5m on
    # what should be a 0.7m door). We retry across a small fixed seed
    # set and accept the first fit whose radius lands inside the
    # class-specific DIN range. Seeds are fixed for reproducibility.
    _SEEDS = (42, 7, 99, 13, 21, 314, 1234)
    model = None
    inliers_mask = None
    cx = cy = r = 0.0
    for seed in _SEEDS:
        try:
            cand_model, cand_mask = ransac(
                pts, CircleModel, min_samples=3,
                residual_threshold=2.5, max_trials=300,
                rng=seed,
            )
        except Exception as e:
            if _DBG:
                print(f"  [arc-dbg] seed={seed} RANSAC raised: {e}")
            continue
        if cand_model is None or cand_mask is None:
            continue
        ccx, ccy, cr = cand_model.params
        if min_r_px < cr < max_r_px:
            model, inliers_mask = cand_model, cand_mask
            cx, cy, r = ccx, ccy, cr
            if _DBG:
                print(f"  [arc-dbg] seed={seed} accepted r={cr*px_to_m:.3f}m")
            break
        elif _DBG:
            print(f"  [arc-dbg] seed={seed} r={cr*px_to_m:.3f}m out of algo [{algo_r_min_m},{algo_r_max_m}]m")
    if model is None or inliers_mask is None:
        if _DBG:
            print(f"  [arc-dbg] no seed produced r in algo [{algo_r_min_m},{algo_r_max_m}]m")
        return None

    inlier_pts = pts[inliers_mask]
    if inlier_pts.shape[0] < 50:
        if _DBG:
            print(f"  [arc-dbg] only {inlier_pts.shape[0]} inliers (<50)")
        return None
    full_circle_px = 2.0 * math.pi * r
    coverage = inlier_pts.shape[0] / full_circle_px
    if coverage < 0.10:        # need at least ~36 deg of arc
        if _DBG:
            print(f"  [arc-dbg] coverage {coverage:.3f} < 0.10")
        return None

    # Validation: the hinge (circle centre) MUST lie inside the bbox.
    # If the arc is fit to features extending past the bbox (e.g. a wall
    # corner), the centre will land outside. Tag-6.0: ±4 px margin
    # because wall-subtraction (dilate iters=2) can shift the RANSAC
    # centre by ~3 px at MM4 render scale, putting a valid hinge just
    # outside the strict crop.
    crop_w = x1 - x0
    crop_h = y1 - y0
    margin_hinge_px = 4
    if not (-margin_hinge_px <= cx <= crop_w + margin_hinge_px
            and -margin_hinge_px <= cy <= crop_h + margin_hinge_px):
        if _DBG:
            print(f"  [arc-dbg] hinge ({cx:.1f},{cy:.1f}) out of crop ({crop_w}x{crop_h}) +/- {margin_hinge_px}")
        return None

    # Validation: full arc must lie inside the bbox (no inlier pixel
    # outside crop bounds). Since pts were taken from the crop, all
    # inliers are in-crop -- but the arc ENDPOINTS might extrapolate
    # outside, indicating a wrong fit. Reject if either endpoint is
    # significantly outside the crop.
    margin_px = 4

    # Arc endpoints from the angular extent of inliers.
    angles = np.arctan2(inlier_pts[:, 1] - cy, inlier_pts[:, 0] - cx)
    angles_sorted = np.sort(angles)
    diffs = np.diff(angles_sorted)
    wrap_diff = (angles_sorted[0] + 2.0 * math.pi) - angles_sorted[-1]
    diffs_full = np.append(diffs, wrap_diff)
    gap_idx = int(np.argmax(diffs_full))
    if gap_idx == len(angles_sorted) - 1:
        arc_a = float(angles_sorted[0])
        arc_b = float(angles_sorted[-1])
    else:
        arc_a = float(angles_sorted[gap_idx + 1])
        arc_b = float(angles_sorted[gap_idx])
    p1 = (cx + r * math.cos(arc_a), cy + r * math.sin(arc_a))
    p2 = (cx + r * math.cos(arc_b), cy + r * math.sin(arc_b))

    # Reject if either endpoint is far outside the crop bounds
    for ep in (p1, p2):
        if (ep[0] < -margin_px or ep[0] > crop_w + margin_px
                or ep[1] < -margin_px or ep[1] > crop_h + margin_px):
            if _DBG:
                print(f"  [arc-dbg] endpoint ({ep[0]:.1f},{ep[1]:.1f}) outside crop ({crop_w}x{crop_h}) +/- {margin_px}")
            return None

    # Distance to nearest wall pixel. Tag-6.0: pad the crop by ~20 cm
    # in every direction before running distanceTransform — for doors
    # whose bbox sits entirely in the room (Class B Matterport
    # convention, wall is OUTSIDE the bbox), the strict-bbox crop_wall
    # contains zero wall pixels and DT degenerates to float32_max for
    # every endpoint. The padded crop captures the actual nearby wall.
    pad_dt_px = max(20, int(round(0.20 / px_to_m)))
    xw0 = max(0, x0 - pad_dt_px)
    yw0 = max(0, y0 - pad_dt_px)
    xw1 = min(W, x1 + pad_dt_px)
    yw1 = min(H, y1 + pad_dt_px)
    crop_wall_padded = wall_mask[yw0:yw1, xw0:xw1]
    if crop_wall_padded.sum() == 0:
        # No wall anywhere within 20 cm of the bbox — the door cannot
        # be wall-anchored from this crop. Skip the wall-touch check
        # by treating both endpoints as on-wall (downstream Tag-6.3
        # wall_anchor_validation will catch this case properly).
        d1 = d2 = 0.0
    else:
        inv_wall = ((crop_wall_padded == 0).astype(np.uint8)) * 255
        dt = cv2.distanceTransform(inv_wall, cv2.DIST_L2, 5)
        # Shift offset between original-crop coords (where p1, p2 live)
        # and padded-crop coords (where dt is indexed).
        ox = x0 - xw0
        oy = y0 - yw0

        def dist_to_wall(px_xy):
            ix = int(round(px_xy[0] + ox))
            iy = int(round(px_xy[1] + oy))
            ix = max(0, min(dt.shape[1] - 1, ix))
            iy = max(0, min(dt.shape[0] - 1, iy))
            return float(dt[iy, ix])

        d1 = dist_to_wall(p1)
        d2 = dist_to_wall(p2)
    if d1 <= d2:
        closed_p, open_p = p1, p2
        closed_d, open_d = d1, d2
    else:
        closed_p, open_p = p2, p1
        closed_d, open_d = d2, d1

    # Tag-6.0: scale-relative wall-touch tolerance. The hardcoded 5 px
    # equals 1.3 cm at MM4 — too tight given the wall-subtraction kernel
    # (dilate iters=2) erodes ~4 px of arc-near-wall pixels and shifts
    # the fit. 8 cm budget = DIN wall half-width (~6 cm) + render
    # anti-aliasing (~1 cm) + dilate-erosion (~1 cm). Doors with a
    # closed-endpoint farther than 8 cm from any wall are genuinely
    # off-wall (incidental circles in furniture, etc.).
    closed_tol_px = max(8.0, 0.08 / px_to_m)
    if closed_d > closed_tol_px:
        if _DBG:
            print(f"  [arc-dbg] closed_d={closed_d:.1f}px > tol={closed_tol_px:.1f}px (r={r*px_to_m:.3f}m)")
        return None
    # Open endpoint must extend into the room (NOT on a wall), with the
    # same tolerance for symmetry.
    if open_d < closed_tol_px:
        if _DBG:
            print(f"  [arc-dbg] open_d={open_d:.1f}px < tol={closed_tol_px:.1f}px (open is on a wall)")
        return None

    hinge_m = (
        float((x0 + cx) * px_to_m), float((y0 + cy) * px_to_m)
    )
    closed_m = (
        float((x0 + closed_p[0]) * px_to_m),
        float((y0 + closed_p[1]) * px_to_m),
    )
    open_m = (
        float((x0 + open_p[0]) * px_to_m),
        float((y0 + open_p[1]) * px_to_m),
    )
    radius_m = float(r * px_to_m)

    # Tag-6.0: pixel_door cross-validation removed.
    # On Matterport-style outlines the longest line in the bbox is often
    # a wall edge that survived morphological subtraction, not the door
    # panel. d_hinge then measures arc-hinge-to-wall-line distance and
    # rejects valid arcs (diagnose: 4/4 single-doors had d_hinge = 0.18
    # to 0.83 m vs the 0.10 m threshold). The remaining checks above
    # already enforce arc validity:
    #   - class-specific radius range
    #   - coverage >= 0.10 (arc subtends >= 36 deg)
    #   - hinge inside crop (with margin)
    #   - both endpoints inside crop
    #   - closed-endpoint near a wall pixel (scale-relative tolerance)
    #   - open-endpoint not on a wall
    # That set is sufficient to discriminate real door arcs from
    # incidental circles (wall corners, furniture); pixel_door stays as
    # a hint for downstream axis selection but does not gate detection.

    # Tag-6.10 DIN flag: radius accepted by the algorithm but outside
    # the class-specific DIN band → record it without affecting the fit.
    arc_din_flags: List[str] = []
    if not (din_r_min_m <= radius_m <= din_r_max_m):
        arc_din_flags.append(
            f"non_din_radius_{radius_m:.2f}m_class_{label_class}"
        )

    return DoorArc(
        hinge_xy_m=hinge_m,
        radius_m=radius_m,
        closed_xy_m=closed_m,
        open_xy_m=open_m,
        n_inliers=int(inlier_pts.shape[0]),
        arc_coverage=float(coverage),
        din_flags=arc_din_flags,
    )


def detect_twin_arcs_in_bbox(
    image_gray: np.ndarray,
    bbox_px: Tuple[float, float, float, float],
    wall_mask: np.ndarray,
    *,
    px_to_m: float,
    neighbour_bboxes_px: Optional[Sequence[Tuple[float, float, float, float]]] = None,
) -> Optional[Tuple[Optional["DoorArc"], Optional["DoorArc"]]]:
    """Tag-6.2: split a door_double bbox along its longer axis and
    run single-arc RANSAC on each half separately. Each panel's
    arc gets fitted in isolation, so two small arcs that would
    otherwise merge into a fake huge circle on the whole bbox are
    recovered as two valid hinges.

    Returns ``(arc_left, arc_right)`` where either may be None when
    that half fails the single-door radius/coverage gate. Returns
    None if the bbox is too small to split sensibly.
    """
    x0, y0, x1, y1 = bbox_px
    w = x1 - x0
    h = y1 - y0
    if w < 30 or h < 30:
        return None
    if w >= h:
        mid = 0.5 * (x0 + x1)
        left_box = (x0, y0, mid, y1)
        right_box = (mid, y0, x1, y1)
    else:
        mid = 0.5 * (y0 + y1)
        left_box = (x0, y0, x1, mid)
        right_box = (x0, mid, x1, y1)
    arc_l = detect_door_arc(
        image_gray, left_box, wall_mask,
        px_to_m=px_to_m, label_class="door_single",
        pixel_door=None,
        neighbour_bboxes_px=neighbour_bboxes_px,
    )
    arc_r = detect_door_arc(
        image_gray, right_box, wall_mask,
        px_to_m=px_to_m, label_class="door_single",
        pixel_door=None,
        neighbour_bboxes_px=neighbour_bboxes_px,
    )
    return (arc_l, arc_r)


@dataclass
class PixelDoor:
    center_xy_m: Tuple[float, float]
    axis_xy: Tuple[float, float]   # unit vector along the longest line
    width_m: float
    n_lines: int
    longest_line_px: float


def extract_pixel_door(
    image_gray: np.ndarray,
    bbox_px: Tuple[float, float, float, float],
    wall_mask: np.ndarray,
    *,
    px_to_m: float,
) -> Optional[PixelDoor]:
    """Find the door-panel rectangle STRICTLY inside the bbox.

    No padding, walls subtracted -- analysis is confined to the label
    bbox per Willi 2026-04-26, and wall pixels are masked out so the
    longest line is the door panel, not a wall edge.
    """
    H, W = image_gray.shape
    x0 = int(max(0, math.floor(bbox_px[0])))
    y0 = int(max(0, math.floor(bbox_px[1])))
    x1 = int(min(W, math.ceil(bbox_px[2])))
    y1 = int(min(H, math.ceil(bbox_px[3])))
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    crop = image_gray[y0:y1, x0:x1]
    crop_wall = wall_mask[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    _, binary = cv2.threshold(crop, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    dilated_wall = cv2.dilate(crop_wall, kernel, iterations=3)
    door_only = cv2.bitwise_and(binary, cv2.bitwise_not(dilated_wall))

    # Stronger reject mask used to drop Hough lines whose endpoints/midpoint
    # sit on (or right next to) the wall mask. Without this the longest line
    # is often a wall edge that survived the morphological subtraction.
    line_reject_mask = cv2.dilate(crop_wall, kernel, iterations=5)

    min_line_px = max(20, int(0.30 / px_to_m))
    # (N, 4) in OpenCV 4 wie 5 - siehe planproof_geometry.ai.cv_compat.
    lines = hough_lines_p(
        door_only, rho=1, theta=np.pi / 180.0, threshold=30,
        minLineLength=min_line_px, maxLineGap=5,
    )
    if len(lines) == 0:
        return None

    rh, rw = line_reject_mask.shape

    def _is_wall_pixel(px: float, py: float) -> bool:
        ix = int(round(px))
        iy = int(round(py))
        if 0 <= ix < rw and 0 <= iy < rh:
            return bool(line_reject_mask[iy, ix])
        return False

    longest_len = 0.0
    longest = None
    for lx0, ly0, lx1, ly1 in lines:
        # Reject lines whose midpoint or both endpoints are on the wall mask.
        mx = 0.5 * (lx0 + lx1)
        my = 0.5 * (ly0 + ly1)
        if _is_wall_pixel(mx, my):
            continue
        if _is_wall_pixel(lx0, ly0) and _is_wall_pixel(lx1, ly1):
            continue
        length = math.hypot(lx1 - lx0, ly1 - ly0)
        if length > longest_len:
            longest_len = length
            longest = (lx0, ly0, lx1, ly1)
    if longest is None:
        return None

    lx0, ly0, lx1, ly1 = longest
    cx_px = 0.5 * (lx0 + lx1) + x0
    cy_px = 0.5 * (ly0 + ly1) + y0
    dx = lx1 - lx0
    dy = ly1 - ly0
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    axis = (dx / norm, dy / norm)
    width_m = float(longest_len * px_to_m)
    return PixelDoor(
        center_xy_m=(cx_px * px_to_m, cy_px * px_to_m),
        axis_xy=axis,
        width_m=width_m,
        n_lines=len(lines),
        longest_line_px=float(longest_len),
    )


# ── Tag-4: Wall-Outline-Gap Detection (primary path) ────────────────────


def classify_wall_solidness(
    image_gray: np.ndarray,
    wall_polygon: WallPolygon,
    wall_mask: np.ndarray,
    *,
    px_to_m: float,
    solid_intensity_threshold: int = 160,
) -> bool:
    """Classify a wall polygon as solid-filled (dark) or outlined.

    Sample raster intensity at pixels that lie inside the polygon AND
    inside the original wall_mask. If mean intensity is below
    ``solid_intensity_threshold`` (typical 160 / 255), the wall is
    drawn as a solid dark stripe (e.g. plumbing walls in Matterport
    bathrooms) — the door gap is then a BRIGHT valley in the profile
    instead of a dark valley. Otherwise the wall is outlined (mostly
    white interior with thin dark strokes) and the profile polarity
    is unchanged.

    Reference: Ahmed et al. 2013, "Notation-invariant patch-based wall
    detector" (Springer GREC) — different wall notation styles
    (filled, hatched, outlined) require different per-style detectors.
    """
    H, W = image_gray.shape
    pts = np.asarray(wall_polygon.shell_xy, dtype=np.float64) / px_to_m
    poly_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [pts.astype(np.int32)], 255)
    inside = (poly_mask > 0) & (wall_mask > 0)
    if inside.sum() < 50:
        return False
    return float(image_gray[inside].mean()) < solid_intensity_threshold


def profile_wall_outline(
    image_gray: np.ndarray,
    origin_m: np.ndarray,
    major: np.ndarray,
    minor_into: np.ndarray,
    s_lo_m: float,
    s_hi_m: float,
    wall_thickness_m: float,
    px_to_m: float,
    *,
    step_px: float = 1.0,
    dark_threshold: int = 180,
    bright_threshold: int = 200,
    wall_is_solid: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample dark-pixel density across the wall column.

    For each step ``s`` along ``major``, count the fraction of pixels
    along the perpendicular column ``origin + s*major + k*minor_into``
    (for ``k`` in ``[2 px, thickness + 5 mm]``) according to a
    polarity-routed metric:

    * **Outlined wall** (``wall_is_solid=False``): density of pixels
      below ``dark_threshold`` — high inside intact wall (outlines
      contribute), low at door gap (mostly white).
    * **Solid-filled wall** (``wall_is_solid=True``, Ahmed et al. 2013):
      density of pixels below ``dark_threshold``, then the profile is
      INVERTED so the gap (white pixels in dark wall) becomes a low-
      profile region matching the same downstream gap detector. The
      INTACT part of a solid wall reads dark-density ≈ 0.7-0.9, so
      ``1 - dark_density`` ≈ 0.1-0.3; the GAP reads dark-density ≈ 0,
      so ``1 - dark_density`` ≈ 1.0 — wait that flips the polarity, so
      we need the absolute distance from the wall body density, not a
      naive ``1 - x``. We compute baseline-relative residual:
      ``profile_solid = max(0, baseline - dark_density)`` where the
      door gap (low dark) yields high residual. Easier: invert the
      sense in find_door_gap by passing the polarity-flipped profile.

      Concrete implementation here: compute dark_density profile as
      usual, then for solid walls set ``profile = 1 - profile`` so
      the same find_door_gap (which looks for LOW profile valleys)
      detects the gap correctly. Inside an intact solid wall,
      ``1 - dark_density`` ≈ 0.1-0.3; at the gap it stays in the
      same range OR jumps depending on background. To get a real
      contrast we use ``profile = bright_density`` for solid walls
      (white-pixel density) which is high at gap, low inside wall;
      then we flip to ``profile = 1 - bright_density`` so the gap
      reads LOW.
    """
    H, W = image_gray.shape
    if px_to_m <= 0 or s_hi_m <= s_lo_m:
        return np.zeros(0), np.zeros(0)

    step_m = step_px * px_to_m
    n_steps = max(2, int(round((s_hi_m - s_lo_m) / step_m)) + 1)
    s_values = np.linspace(s_lo_m, s_hi_m, n_steps)

    inset_px = 2
    pad_px = max(2, int(round(0.005 / px_to_m)))   # 5 mm extra past far edge
    depth_px = max(6, int(round(wall_thickness_m / px_to_m)) + pad_px)
    ks = np.arange(inset_px, depth_px + 1, dtype=np.float64)

    if wall_is_solid:
        # Solid-fill wall: door gap = bright (white) pixels in dark wall
        # body. We build profile = 1 - bright_density so an intact wall
        # body (low bright count) reads HIGH and a gap (high bright
        # count, white opening) reads LOW — matching the find_door_gap
        # contract that gaps are LOW-profile valleys.
        signal = 1.0 - (image_gray > bright_threshold).astype(np.float32)
    else:
        # Outlined wall: use dark-pixel density. Intact wall has outline
        # strokes contributing dark, gap has none → low profile at gap.
        signal = (image_gray < dark_threshold).astype(np.float32)

    profile = np.zeros(n_steps, dtype=np.float64)
    for i, s in enumerate(s_values):
        cx_px = (origin_m[0] + s * major[0]) / px_to_m
        cy_px = (origin_m[1] + s * major[1]) / px_to_m
        sx = cx_px + ks * minor_into[0]
        sy = cy_px + ks * minor_into[1]
        valid = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
        if not valid.any():
            profile[i] = 0.0
            continue
        ix = np.clip(np.round(sx[valid]).astype(np.int32), 0, W - 1)
        iy = np.clip(np.round(sy[valid]).astype(np.int32), 0, H - 1)
        profile[i] = float(np.mean(signal[iy, ix]))

    smooth_px = max(3, int(round(0.040 / px_to_m)))
    if profile.size >= smooth_px:
        kernel = np.ones(smooth_px, dtype=np.float64) / smooth_px
        profile = np.convolve(profile, kernel, mode="same")

    return s_values, profile


def find_door_gap(
    s_values: np.ndarray,
    profile: np.ndarray,
    bbox_s_lo: float,
    bbox_s_hi: float,
    *,
    gap_threshold_factor: float = 0.30,
    min_door_m: float = ALGO_GAP_MIN_M,
    max_door_m: float = ALGO_GAP_MAX_M,
    roi_pad_m: float = 0.30,
    min_baseline: float = 0.04,
) -> Optional[Tuple[float, float, float]]:
    """Detect the longest contiguous low-blackness run inside the bbox±pad.

    Baseline = ``median(profile)`` outside the ROI band → "full wall"
    reading. A door gap is a run where ``profile <
    gap_threshold_factor * baseline``. Run edges are sub-pixel-refined by
    linear interpolation across the threshold crossing so the door starts
    exactly where the wall outline restarts.

    Returns ``(gap_start_m, gap_end_m, baseline)`` or ``None`` when no
    plausible gap is found (gap too short/long, or wall too faint to
    establish a baseline).
    """
    if s_values.size < 5:
        return None

    roi_lo = bbox_s_lo - roi_pad_m
    roi_hi = bbox_s_hi + roi_pad_m
    in_roi = (s_values >= roi_lo) & (s_values <= roi_hi)
    out_of_roi = ~in_roi

    if out_of_roi.any():
        baseline = float(np.median(profile[out_of_roi]))
    else:
        baseline = float(np.median(profile))

    if baseline < min_baseline:
        return None

    threshold_value = gap_threshold_factor * baseline
    gap_mask = (profile < threshold_value) & in_roi
    if not gap_mask.any():
        return None

    best_start = best_end = -1
    best_len = 0
    n = len(gap_mask)
    i = 0
    while i < n:
        if gap_mask[i]:
            j = i
            while j < n and gap_mask[j]:
                j += 1
            if (j - i) > best_len:
                best_len = j - i
                best_start = i
                best_end = j - 1
            i = j
        else:
            i += 1
    if best_start < 0:
        return None

    def _interp(idx_a: int, idx_b: int) -> float:
        p_a = float(profile[idx_a])
        p_b = float(profile[idx_b])
        s_a = float(s_values[idx_a])
        s_b = float(s_values[idx_b])
        if p_a == p_b:
            return 0.5 * (s_a + s_b)
        frac = (threshold_value - p_a) / (p_b - p_a)
        frac = max(0.0, min(1.0, frac))
        return s_a + frac * (s_b - s_a)

    if best_start > 0 and profile[best_start - 1] > threshold_value:
        s_start = _interp(best_start - 1, best_start)
    else:
        s_start = float(s_values[best_start])

    if best_end < n - 1 and profile[best_end + 1] > threshold_value:
        s_end = _interp(best_end, best_end + 1)
    else:
        s_end = float(s_values[best_end])

    gap_length = s_end - s_start
    if gap_length < min_door_m or gap_length > max_door_m:
        return None
    return s_start, s_end, baseline


# ── Step 4 — gap-fit + scaling ──────────────────────────────────────────


def _project(pts: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return (pts - origin) @ axis


def walk_to_wall_along_axis(
    start_m: np.ndarray,
    axis_unit: np.ndarray,
    wall_polygons: Sequence[WallPolygon],
    *,
    sign: int = 1,
    max_m: float = 3.0,
    step_m: float = 0.02,
) -> float:
    """Walk from ``start_m`` along ``sign * axis_unit`` and return the
    distance (in metres) to the first wall hit.

    Returns ``max_m`` if no wall is encountered within the search range.
    """
    from shapely.geometry import Point
    walls = [
        Polygon(wp.shell_xy) for wp in wall_polygons
        if len(wp.shell_xy) >= 3
    ]
    if not walls:
        return float(max_m)
    n_steps = int(max_m / step_m) + 1
    for k in range(n_steps):
        d = k * step_m
        if d > max_m:
            break
        x = float(start_m[0] + sign * d * axis_unit[0])
        y = float(start_m[1] + sign * d * axis_unit[1])
        pt = Point(x, y)
        for w in walls:
            try:
                if w.contains(pt) or w.touches(pt):
                    return float(d)
            except Exception:
                continue
    return float(max_m)


def gap_extent_along_axis(
    w1: WallPolygon,
    w2: Optional[WallPolygon],
    bbox_m: Tuple[float, float, float, float],
    major: np.ndarray,
    origin: np.ndarray,
) -> Tuple[float, float, str]:
    """Return (gap_start_t, gap_end_t, source) along major axis.

    Source is a label describing which path was taken:
      "between"  : two flanking walls have a clean gap straddling bbox.
      "single"   : only one wall, or the second wall is on the same side.
      "bbox"     : fallback to bbox extent along major.
    """
    bbox_pts = np.array([
        [bbox_m[0], bbox_m[1]], [bbox_m[2], bbox_m[1]],
        [bbox_m[2], bbox_m[3]], [bbox_m[0], bbox_m[3]],
    ])
    bbox_t = _project(bbox_pts, origin, major)
    bbox_lo, bbox_hi = float(bbox_t.min()), float(bbox_t.max())
    bbox_mid = 0.5 * (bbox_lo + bbox_hi)

    w1_pts = np.asarray(w1.shell_xy, dtype=np.float64)
    w1_t = _project(w1_pts, origin, major)
    w1_lo, w1_hi = float(w1_t.min()), float(w1_t.max())

    if w2 is not None:
        w2_pts = np.asarray(w2.shell_xy, dtype=np.float64)
        w2_t = _project(w2_pts, origin, major)
        w2_lo, w2_hi = float(w2_t.min()), float(w2_t.max())
        # Try both orderings (W1 on the lower side or higher side).
        if bbox_mid >= w1_hi and bbox_mid <= w2_lo:
            return w1_hi, w2_lo, "between"
        if bbox_mid >= w2_hi and bbox_mid <= w1_lo:
            return w2_hi, w1_lo, "between"
        # Walls overlap or bbox sits to one side; fall through.

    return bbox_lo, bbox_hi, "bbox"


def build_rectangle_xy(
    center: np.ndarray,
    major: np.ndarray,
    minor: np.ndarray,
    width: float,
    depth: float,
) -> List[Tuple[float, float]]:
    half_w = width * 0.5
    half_d = depth * 0.5
    corners = [(-half_w, -half_d), (half_w, -half_d),
               (half_w, half_d), (-half_w, half_d)]
    return [
        (float(center[0] + t * major[0] + s * minor[0]),
         float(center[1] + t * major[1] + s * minor[1]))
        for t, s in corners
    ]


def _wall_interior_sign(
    wall_polygon: WallPolygon,
    origin: np.ndarray,
    minor: np.ndarray,
) -> float:
    """Return +1 if ``+minor`` points INTO the wall polygon, -1 otherwise.

    For a CCW shapely polygon ``minor = (-major_y, major_x)`` (90° CCW from
    the edge tangent) points to the interior, but Phase-2 polygons may be
    re-wound by ``buffer(0)`` or ``unary_union``, so we sample explicitly.
    """
    test_pt = origin + 0.05 * minor
    try:
        inside = Polygon(wall_polygon.shell_xy).buffer(0).contains(
            __import__("shapely.geometry", fromlist=["Point"]).Point(
                float(test_pt[0]), float(test_pt[1])
            )
        )
    except Exception:
        return 1.0
    return 1.0 if inside else -1.0


def _project_minor_onto_wall(
    bbox_center: np.ndarray,
    origin: np.ndarray,
    minor: np.ndarray,
    w1: WallPolygon,
    w2: Optional[WallPolygon],
) -> float:
    """Find the minor-axis offset that places the door on the wall surface.

    The label bbox surrounds the door symbol (panel + arc), so its centroid is
    offset from the wall surface. The wall surface midline at this location is
    the line ``origin + t * major + s * minor`` for whichever ``s`` lies on
    (or closest to) the wall polygon. We return that ``s``.
    """
    from shapely.geometry import Point

    walls: List[Polygon] = []
    for wp in (w1, w2):
        if wp is None or len(wp.shell_xy) < 3:
            continue
        poly = Polygon(wp.shell_xy).buffer(0)
        if poly.is_valid and not poly.is_empty:
            walls.append(poly)
    if not walls:
        return float((bbox_center - origin) @ minor)

    bbox_pt = Point(float(bbox_center[0]), float(bbox_center[1]))
    # Pick the wall closest to the bbox; project bbox_center onto its surface.
    walls.sort(key=lambda w: w.distance(bbox_pt))
    wall = walls[0]
    if wall.contains(bbox_pt):
        # bbox_center already inside wall (Class A) -- door minor = bbox minor.
        snapped_pt = bbox_pt
    else:
        # Project bbox_center onto wall boundary along the minor axis: take
        # the point on wall closest to bbox_center. For an axis-aligned wall
        # this lands on the wall's near edge (door surface).
        try:
            snapped = wall.exterior.interpolate(
                wall.exterior.project(bbox_pt)
            )
            snapped_pt = snapped
        except Exception:
            snapped_pt = bbox_pt
    snapped = np.array([snapped_pt.x, snapped_pt.y], dtype=np.float64)
    return float((snapped - origin) @ minor)


def place_door(
    bbox_m: Tuple[float, float, float, float],
    label_polygon_m: List[Tuple[float, float]],
    wall_polygons: Sequence[WallPolygon],
    image_gray: np.ndarray,
    wall_mask: np.ndarray,
    px_to_m: float,
    *,
    label_class: str = "door",
    other_door_bboxes_m: Optional[Sequence[Tuple[float, float, float, float]]] = None,
) -> DoorPlacement:
    label_id_default = ""
    label_class_default = ""
    placement = DoorPlacement(
        label_id=label_id_default, label_class=label_class_default,
        bbox_m=bbox_m, w1_idx=-1, w2_idx=None,
        w1_distance_m=float("inf"), w2_distance_m=None,
    )

    bbox_center = np.array([
        0.5 * (bbox_m[0] + bbox_m[2]),
        0.5 * (bbox_m[1] + bbox_m[3]),
    ])

    # Pixel extraction first — we need pixel_axis to choose the hinge wall
    # for doors at L-corners where the closest wall by distance is NOT the
    # wall the door is hinged on.
    bbox_px = _bbox_in_pixels(bbox_m, px_to_m)
    # Tag-6.8: project neighbour-door bboxes to pixel space for arc
    # detection's neighbour-mask. Only those that overlap the current
    # bbox-with-padding can affect the RANSAC pool.
    neighbour_bboxes_px = None
    if other_door_bboxes_m:
        neighbour_bboxes_px = [
            _bbox_in_pixels(b, px_to_m) for b in other_door_bboxes_m
        ]
    pixel_door = extract_pixel_door(
        image_gray, bbox_px, wall_mask, px_to_m=px_to_m,
    )
    # Tag-7.9 — stash the panel-symbol anchor on the placement record so
    # ``attempt_pixel_recovery`` (post-placement) can rebuild the door
    # rectangle centred on the symbol when the bbox-extent fallback
    # lands on solid wall.
    if pixel_door is not None and pixel_door.width_m > 0.30:
        try:
            placement.pixel_panel_center_xy_m = (
                float(pixel_door.center_xy_m[0]),
                float(pixel_door.center_xy_m[1]),
            )
            placement.pixel_panel_axis_xy = (
                float(pixel_door.axis_xy[0]),
                float(pixel_door.axis_xy[1]),
            )
            placement.pixel_width_m = float(pixel_door.width_m)
        except Exception:
            pass

    # Tag-6.1 class-routing: only swing-doors have a 90 deg arc symbol.
    #   - door / door_single: standard single-arc door, arc-anchored.
    #   - door_double: two arcs OR a single wide opening (Tag-6.2 will
    #     add twin-arc detection; for now falls through to the
    #     bbox_extent_on_hinge fallback when single-arc fails).
    #   - door_sliding: panel parallel to the wall, no arc — skip the
    #     RANSAC step entirely so we never waste seven retries on a
    #     class that geometrically cannot match. The longest line in
    #     the bbox (extracted as pixel_door) IS the door panel.
    # Tag-4.1 (research-backed, 2026-04-27): the door ARC is the most
    # reliable hinge anchor for L-corner cases. Following Liu et al.
    # 2017 + CubiCasa5K + DoorDet 2025: detect the door symbol as a
    # primitive (arc + radius) FIRST, then snap to a wall — instead of
    # detecting walls first and fishing for gaps. The arc centre =
    # hinge point on the wall surface; arc radius = exact door width;
    # closed-endpoint direction = swing direction along the wall.
    twin_arc_left = twin_arc_right = None
    if label_class == "door_sliding":
        door_arc = None
    elif label_class == "door_double":
        # Tag-6.2 twin-arc-via-bbox-halves: a door_double symbol on
        # raster plans is rarely one big 90 deg arc — it is two
        # smaller arcs (or a panel pair without arcs). RANSAC fitted
        # on the whole bbox lands on a r=4-15 m quasi-line; the
        # door_double class range [1.20, 2.00] m discards it. By
        # splitting the bbox along its longer axis and running the
        # single-door RANSAC on each half separately, each panel's
        # arc gets fitted in isolation. Two valid arcs anchored on a
        # wall confirm the twin-arc geometry; the door rectangle then
        # spans hinge_1..hinge_2 along the shared wall tangent.
        twin_arcs = detect_twin_arcs_in_bbox(
            image_gray, bbox_px, wall_mask, px_to_m=px_to_m,
            neighbour_bboxes_px=neighbour_bboxes_px,
        )
        if twin_arcs is not None and twin_arcs[0] is not None and twin_arcs[1] is not None:
            door_arc = None  # twin path takes over below
            twin_arc_left, twin_arc_right = twin_arcs
        else:
            twin_arc_left = twin_arc_right = None
            door_arc = detect_door_arc(
                image_gray, bbox_px, wall_mask,
                px_to_m=px_to_m, label_class=label_class,
                pixel_door=pixel_door,
                neighbour_bboxes_px=neighbour_bboxes_px,
            )
    else:
        twin_arc_left = twin_arc_right = None
        door_arc = detect_door_arc(
            image_gray, bbox_px, wall_mask,
            px_to_m=px_to_m, label_class=label_class,
            pixel_door=pixel_door,
            neighbour_bboxes_px=neighbour_bboxes_px,
        )

    # Rank ALL walls within W2 distance and pick the hinge wall by
    # panel orientation (Willi 2026-04-27 — "an der falschen Wand"):
    # for a 90°-open door symbol the panel pixel-line is perpendicular to
    # the hinge wall's surface. If the closest wall fails that test but
    # another nearby wall passes, swap them.
    # Search the wider corridor radius so detect_corridor_between_walls can
    # see a perpendicular pair past the W2 cap; the W1/W2 pair selection
    # below still uses FLANK_W2_MAX_M.
    ranked = find_flanking_walls_ranked(
        bbox_m, wall_polygons, max_distance_m=FLANK_CORRIDOR_MAX_M,
    )
    if not ranked or ranked[0][1] > FLANK_W1_MAX_M:
        placement.note = f"no wall within {FLANK_W1_MAX_M} m"
        return placement

    pixel_axis = pixel_door.axis_xy if pixel_door is not None else None

    # Debug: per-wall perpendicularity scores + signed pixel_axis offset.
    # The signed offset tells whether two perp-walls straddle the bbox
    # along the panel direction (corridor-pair criterion). The signed
    # value uses the closest point on the wall polygon, mirroring
    # detect_corridor_between_walls so the debug output matches the
    # detector's classification.
    debug_perp = ""
    if pixel_axis is not None:
        from shapely.ops import nearest_points
        from shapely.geometry import Point as _ShPoint

        pix = np.array(pixel_axis, dtype=np.float64)
        n = float(np.linalg.norm(pix))
        if n > 1e-6:
            pix /= n
            scored = []
            bbox_pt_dbg = _ShPoint(
                float(bbox_center[0]), float(bbox_center[1])
            )
            for idx, dist in ranked[:5]:
                wp_dbg = wall_polygons[idx]
                _, tang, _ = nearest_edge_axis(wp_dbg, bbox_center)
                cross = abs(float(tang[0] * pix[1] - tang[1] * pix[0]))
                try:
                    wpoly = Polygon(wp_dbg.shell_xy).buffer(0)
                    np_pt_dbg, _ = nearest_points(wpoly, bbox_pt_dbg)
                    nearest_dbg = np.array(
                        [np_pt_dbg.x, np_pt_dbg.y], dtype=np.float64
                    )
                    signed = float((nearest_dbg - bbox_center) @ pix)
                except Exception:
                    signed = 0.0
                scored.append((idx, dist, cross, signed))
            debug_perp = "  perp[" + " ".join(
                f"W{i}:d={d:.2f},perp={c:.2f},s={s:+.2f}"
                for i, d, c, s in scored
            ) + "]"

    # Tag-3 (Willi 2026-04-27): corridor detection runs BEFORE the start-wall
    # selector. If two walls perpendicular to the panel direction straddle
    # the bbox along that direction AND the bbox actually sits between
    # them (its pixel_axis projection fits inside the gap), the door is a
    # corridor door (sits in the gap, axis = pixel_axis), not a wall-cut.
    corridor_pick = None
    if pixel_axis is not None:
        cand = detect_corridor_between_walls(
            wall_polygons, ranked, bbox_center, pixel_axis,
        )
        if cand is not None:
            cand_pos, cand_neg, cand_gap, cand_mid, cand_axis = cand
            # Reject the candidate if the bbox is not actually inside the
            # gap along pixel_axis -- the two perp walls may exist but be
            # positioned such that the bbox sits at a corner / outside.
            bbox_pa_corners = (
                np.array([
                    [bbox_m[0], bbox_m[1]], [bbox_m[2], bbox_m[1]],
                    [bbox_m[2], bbox_m[3]], [bbox_m[0], bbox_m[3]],
                ]) - bbox_center
            ) @ cand_axis
            bbox_pa_lo = float(bbox_pa_corners.min())
            bbox_pa_hi = float(bbox_pa_corners.max())
            gap_lo = cand_mid - 0.5 * cand_gap
            gap_hi = cand_mid + 0.5 * cand_gap
            tol = 0.05  # 5 cm slack
            if (bbox_pa_lo >= gap_lo - tol
                    and bbox_pa_hi <= gap_hi + tol):
                # Tag-9.2 guard: a true corridor door FILLS the gap between
                # the two perpendicular walls. If the pixel-extracted door
                # symbol is much smaller than the corridor gap, the bbox
                # only HAPPENS to sit between two perp walls — the door
                # itself is wall-mounted at its pixel width, not stretched
                # across the corridor. (sliding-door bbox 0.92m sitting
                # between perp walls 3.76m apart was being mis-stretched.)
                pixel_w_for_corridor = (
                    pixel_door.width_m
                    if (pixel_door is not None and pixel_door.width_m > 0.30)
                    else 0.0
                )
                if (
                    pixel_w_for_corridor > 0.0
                    and pixel_w_for_corridor < 0.70 * cand_gap
                ):
                    corridor_pick = None
                else:
                    corridor_pick = cand

    debug_corridor = ""
    if corridor_pick is not None:
        debug_corridor = (
            f"  corridor[W{corridor_pick[0]},W{corridor_pick[1]}"
            f" gap={corridor_pick[2]:.3f}m]"
        )

    # Compute start_pick early so Tag-4 below can use the panel-endpoint
    # hinge wall as its sample wall (preferred over the geometrically
    # closest wall, which may be a short corner segment).
    start_pick_pre = None
    if corridor_pick is None and pixel_door is not None and pixel_door.width_m > 0.30:
        pa_pre = np.array(pixel_door.axis_xy, dtype=np.float64)
        n_pa_pre = float(np.linalg.norm(pa_pre))
        if n_pa_pre > 1e-6:
            pa_pre = pa_pre / n_pa_pre
            pc_pre = np.array(pixel_door.center_xy_m, dtype=np.float64)
            half_pw_pre = 0.5 * pixel_door.width_m
            panel_e1_pre = pc_pre - half_pw_pre * pa_pre
            panel_e2_pre = pc_pre + half_pw_pre * pa_pre
            start_pick_pre = select_start_wall_by_panel_endpoint(
                wall_polygons, ranked, panel_e1_pre, panel_e2_pre,
            )
            # Tag-7.7 — feed the panel-pixel hinge endpoint into the
            # cascade-agnostic voting that runs after place_door.
            # ``start_pick_pre[1]`` is the panel endpoint that sits ON
            # the wall surface (= hinge corner), already validated by
            # ``select_start_wall_by_panel_endpoint``.
            if start_pick_pre is not None:
                hep = start_pick_pre[1]
                placement.panel_hinge_xy_m = (float(hep[0]), float(hep[1]))

    # Tag-4 (Willi 2026-04-27): primary path — the real door geometry is the
    # gap in the wall outline as drawn in the plan raster. Labels are only
    # ROI hints; we sample wall-mask coverage perpendicular to the hinge
    # wall's tangent and look for the contiguous low-coverage run that
    # marks the door opening. When this finds a plausible gap, we use it
    # directly and skip the Tag-3.2 corridor / start_pick / wall_edge
    # cascade. Otherwise we fall through unchanged.
    # Tag-6.2 twin_arc_pick path — fires for door_double when both
    # bbox halves yielded a valid single-arc fit AND both arcs have a
    # wall anchor. The door rectangle then spans
    # closed_xy_left .. closed_xy_right along the wall tangent,
    # capturing the full opening across both panels.
    twin_pick = None
    debug_twin = ""
    if (label_class == "door_double" and corridor_pick is None
            and twin_arc_left is not None and twin_arc_right is not None):
        cl_l = np.array(twin_arc_left.closed_xy_m, dtype=np.float64)
        cl_r = np.array(twin_arc_right.closed_xy_m, dtype=np.float64)
        n_anch_twin, anch_walls_twin = validate_wall_anchors(
            [tuple(cl_l), tuple(cl_r)], wall_polygons,
        )
        if n_anch_twin >= 2 and anch_walls_twin[0] == anch_walls_twin[1]:
            twin_w_idx = anch_walls_twin[0]
            twin_wall = wall_polygons[twin_w_idx]
            # Origin = midpoint of the two closed-tips, projected onto
            # the wall edge nearest the midpoint. Major axis = wall
            # tangent at that point. The door rectangle spans both
            # closed-tips plus each arc radius (= each panel's swing).
            mid_xy = 0.5 * (cl_l + cl_r)
            tw_origin, tw_major, tw_minor = nearest_edge_axis(twin_wall, mid_xy)
            t_l = float((cl_l - tw_origin) @ tw_major)
            t_r = float((cl_r - tw_origin) @ tw_major)
            t_lo = min(t_l, t_r) - twin_arc_left.radius_m
            t_hi = max(t_l, t_r) + twin_arc_right.radius_m
            twin_t_center = 0.5 * (t_lo + t_hi)
            twin_width = t_hi - t_lo
            twin_thick = local_wall_depth(
                twin_wall, mid_xy, tw_major, tw_minor, tw_origin,
                door_extent_m=twin_width,
                wall_mask=wall_mask, px_to_m=px_to_m,
            )
            twin_pick = (
                twin_w_idx, tw_origin, tw_major, tw_minor, twin_thick,
                twin_t_center, twin_width, cl_l, cl_r,
            )
            debug_twin = (
                f"  twin[W{twin_w_idx} r1={twin_arc_left.radius_m:.3f} "
                f"r2={twin_arc_right.radius_m:.3f} W={twin_width:.3f}m]"
            )

    if twin_pick is not None:
        (tw_idx, tw_origin, tw_major, tw_minor, tw_thick,
         tw_t_center, tw_width, cl_l, cl_r) = twin_pick
        placement.w1_idx = tw_idx
        placement.w2_idx = None
        placement.w1_distance_m = ranked[0][1]
        placement.w2_distance_m = None

        bbox_pts_tw = np.array([
            [bbox_m[0], bbox_m[1]], [bbox_m[2], bbox_m[1]],
            [bbox_m[2], bbox_m[3]], [bbox_m[0], bbox_m[3]],
        ])
        bbox_s_tw = (bbox_pts_tw - tw_origin) @ tw_minor
        s_center_tw = _shift_s_into_bbox(
            tw_origin, tw_major, tw_minor, tw_t_center, tw_width, tw_thick,
            bbox_m, float(bbox_s_tw.min()), float(bbox_s_tw.max()),
        )
        door_center = tw_origin + tw_t_center * tw_major + s_center_tw * tw_minor
        door_polygon = build_rectangle_xy(
            door_center, tw_major, tw_minor, tw_width, tw_thick
        )

        inside_pct, between_score, _wo, val_note = validate_placement(
            door_polygon, bbox_m, wall_polygons[tw_idx], None,
            major=tw_major, origin=tw_origin,
        )
        placement.door_width_m = tw_width
        placement.door_depth_m = tw_thick
        placement.door_polygon_xy = door_polygon
        placement.pixel_polygon_xy = []
        placement.pixel_width_m = (
            twin_arc_left.radius_m + twin_arc_right.radius_m
        )
        placement.width_source = "twin_arc_centered"
        placement.inside_bbox_pct = inside_pct
        placement.wall_overlap_pct = between_score
        placement.inside_bbox = inside_pct >= 0.50
        placement.on_wall = between_score >= 0.5
        # Two anchors are the SUFFICIENT condition (Willi's contract);
        # twin-arc placements always pass the bbox-2D rule by Symbol-
        # First override.
        placement.pass_validation = (
            placement.inside_bbox or placement.on_wall
        )
        placement.validation_note = f"{val_note}; twin-arc anchor=W{tw_idx}"
        placement.note = (
            f"axis=twin_arc_centered "
            f"width={tw_width:.3f}m depth={tw_thick:.3f}m"
            f"{debug_perp}{debug_twin}"
        )
        return placement

    # Tag-4.1 arc_pick path — use door arc geometry to anchor the door
    # placement when the RANSAC arc fit succeeded. This is the most
    # reliable signal for L-corner cases because the arc centre (hinge)
    # is determined by the door symbol itself, not by which wall polygon
    # happens to be 1-2 cm closer to the bbox centroid.
    arc_pick = None
    debug_arc = ""
    if door_arc is not None and corridor_pick is None:
        from shapely.geometry import Point as _ShPoint
        hinge_xy = np.array(door_arc.hinge_xy_m, dtype=np.float64)
        closed_xy = np.array(door_arc.closed_xy_m, dtype=np.float64)
        # Tag-6.0: anchor on closed_xy (the arc endpoint that touches
        # the wall surface in the door symbol), NOT on hinge_xy. The
        # hinge is the arc CENTRE — for a 90 deg-open door symbol the
        # centre sits a wall half-thickness inside the wall polygon,
        # which can read 5–15 cm from the polygon boundary. closed_xy
        # is the swing-tip resting on the wall surface and is the
        # geometrically correct anchor for wall-snapping.
        closed_pt = _ShPoint(float(closed_xy[0]), float(closed_xy[1]))
        best_w_idx = -1
        best_w_d = float("inf")
        for idx, _ in ranked[:5]:
            wp = wall_polygons[idx]
            try:
                wpoly = Polygon(wp.shell_xy).buffer(0)
                d = float(wpoly.distance(closed_pt))
            except Exception:
                continue
            if d < best_w_d:
                best_w_d = d
                best_w_idx = idx
        # Tolerance matches detect_door_arc's closed_tol (8 cm + small
        # safety budget for polygon vs raster discretisation).
        if best_w_idx >= 0 and best_w_d <= 0.12:
            w1_arc = wall_polygons[best_w_idx]
            # Origin = projection of arc hinge onto the wall edge
            a_origin, a_major, a_minor = nearest_edge_axis(w1_arc, hinge_xy)
            # Disambiguate L-shapes: pick the long edge whose tangent is
            # closest to the hinge→closed direction (= the actual wall
            # the door swings on, even if a perpendicular edge is
            # geometrically closer to the hinge).
            swing_vec = closed_xy - hinge_xy
            swing_norm = float(np.linalg.norm(swing_vec))
            if swing_norm > 1e-6:
                swing_unit = swing_vec / swing_norm
                if abs(float(swing_unit @ a_major)) < 0.85:
                    # Major doesn't align with swing → search other long
                    # edges of this polygon for one that does.
                    pts = np.asarray(w1_arc.shell_xy, dtype=np.float64)
                    n = pts.shape[0]
                    best_align = abs(float(swing_unit @ a_major))
                    for i in range(n):
                        a = pts[i]
                        b = pts[(i + 1) % n]
                        ab = b - a
                        seg_len = float(np.linalg.norm(ab))
                        if seg_len < 0.30:
                            continue
                        tang = ab / seg_len
                        align = abs(float(swing_unit @ tang))
                        if align > best_align:
                            t = float((hinge_xy - a) @ ab) / (seg_len ** 2)
                            t = max(0.0, min(1.0, t))
                            foot = a + t * ab
                            a_origin = foot
                            a_major = tang
                            a_minor = np.array(
                                [-tang[1], tang[0]], dtype=np.float64
                            )
                            best_align = align
            a_thick = local_wall_depth(
                w1_arc, hinge_xy, a_major, a_minor, a_origin,
                door_extent_m=door_arc.radius_m,
                wall_mask=wall_mask, px_to_m=px_to_m,
            )
            arc_pick = (
                best_w_idx, a_origin, a_major, a_minor, a_thick,
                hinge_xy, closed_xy, door_arc.radius_m,
            )
            debug_arc = (
                f"  arc[W{best_w_idx} r={door_arc.radius_m:.3f}m "
                f"hinge_d={best_w_d*1000:.1f}mm]"
            )

    if arc_pick is not None:
        # Tag-4.1 primary path — door anchored on arc geometry.
        (a_w_idx, a_origin, a_major, a_minor, a_thick,
         hinge_xy, closed_xy, a_radius) = arc_pick

        placement.w1_idx = a_w_idx
        placement.w2_idx = None
        placement.w1_distance_m = ranked[0][1]
        placement.w2_distance_m = None

        w1_arc = wall_polygons[a_w_idx]
        origin = a_origin
        major = a_major
        minor = a_minor

        hinge_along = float((hinge_xy - origin) @ major)
        closed_along = float((closed_xy - origin) @ major)
        # Door rectangle runs from hinge to (hinge + radius) along the
        # closed-endpoint side. The arc gives us the swing direction.
        if closed_along >= hinge_along:
            t_center = hinge_along + 0.5 * a_radius
        else:
            t_center = hinge_along - 0.5 * a_radius
        door_width = a_radius
        door_depth = a_thick
        bbox_pts_arc = np.array([
            [bbox_m[0], bbox_m[1]], [bbox_m[2], bbox_m[1]],
            [bbox_m[2], bbox_m[3]], [bbox_m[0], bbox_m[3]],
        ])
        bbox_t_arc = (bbox_pts_arc - origin) @ major
        bbox_t_lo_arc = float(bbox_t_arc.min())
        bbox_t_hi_arc = float(bbox_t_arc.max())

        # Tag-6.9 (Willi 2026-04-28 review): the arc gives us the DIN
        # door width, but the actual wall opening can be wider — door
        # rectangles drawn smaller than the wall gap leave a sliver of
        # wall standing between door-end and wall-end. Sample the wall
        # outline along the hinge wall's tangent and, if a real pixel
        # gap inside the bbox extent is wider than the arc radius,
        # expand the door to fill that gap. This is the symbol-first
        # version of "snap to wall start/end": the arc defines the
        # door's class and orientation, but the pixel outline owns the
        # actual span.
        wall_is_solid_arc = classify_wall_solidness(
            image_gray, w1_arc, wall_mask, px_to_m=px_to_m,
        )
        g_sign_arc = _wall_interior_sign(w1_arc, origin, minor)
        minor_into_arc = g_sign_arc * minor
        s_vals_arc, profile_arc = profile_wall_outline(
            image_gray, origin, major, minor_into_arc,
            bbox_t_lo_arc - 0.30, bbox_t_hi_arc + 0.30,
            a_thick, px_to_m,
            wall_is_solid=wall_is_solid_arc,
        )
        gap_arc = find_door_gap(
            s_vals_arc, profile_arc, bbox_t_lo_arc, bbox_t_hi_arc,
        )
        bbox_extent_major = bbox_t_hi_arc - bbox_t_lo_arc
        if gap_arc is not None:
            g_lo_arc, g_hi_arc, _g_base = gap_arc
            gap_len_arc = g_hi_arc - g_lo_arc
            gap_mid_arc = 0.5 * (g_lo_arc + g_hi_arc)
            # Use the pixel gap only when its centre lies inside (or
            # within 5 cm of) the bbox extent along the wall — that
            # guarantees the gap belongs to THIS door's opening, not a
            # neighbouring door's gap that bled into the profile.
            gap_mid_in_bbox = (
                bbox_t_lo_arc - 0.05 <= gap_mid_arc <= bbox_t_hi_arc + 0.05
            )
            if gap_mid_in_bbox and gap_len_arc > door_width:
                door_width = gap_len_arc
                t_center = gap_mid_arc
                debug_arc += f" gap-expanded->{gap_len_arc:.3f}m"
        elif bbox_extent_major > door_width:
            # Profile-based gap detection failed (typical on Matterport
            # outlines too thin to register against anti-aliasing).
            # When the annotator's bbox-extent along the wall tangent
            # exceeds the arc radius, the bbox encodes the actual wall
            # opening directly: arc symbol ~ door class, bbox span ~
            # wall-end-to-wall-end. Trust the bbox extent for width
            # and centre. Door keeps its arc-derived axis and depth;
            # only width and centre shift along major.
            door_width = bbox_extent_major
            t_center = 0.5 * (bbox_t_lo_arc + bbox_t_hi_arc)
            debug_arc += f" bbox-expanded->{bbox_extent_major:.3f}m"

        bbox_s_arc = (bbox_pts_arc - origin) @ minor
        s_center = _shift_s_into_bbox(
            origin, major, minor, t_center, door_width, door_depth,
            bbox_m, float(bbox_s_arc.min()), float(bbox_s_arc.max()),
        )

        door_center = origin + t_center * major + s_center * minor
        door_polygon = build_rectangle_xy(
            door_center, major, minor, door_width, door_depth
        )

        # Tag-6.8 root-cause sanity check: the arc-derived rectangle must
        # overlap its annotator bbox at least 5 % in 2D. When dense
        # Bath/WC clusters cause RANSAC to fit pixels of a neighbouring
        # door's symbol (cross-talk through bbox proximity), the resulting
        # rectangle anchors on a wall but lands entirely outside this
        # door's bbox. Detecting that here lets the cascade fall through
        # to gap_pick / bbox_extent_on_hinge so the door still gets a
        # valid placement based on its OWN bbox extent.
        arc_bbox_2d_pct_check = _bbox_2d_overlap_pct(door_polygon, bbox_m)
        if arc_bbox_2d_pct_check < 0.05:
            arc_pick = None
            debug_arc += f" REJECTED bbox-2D={arc_bbox_2d_pct_check*100:.1f}%"

    if arc_pick is not None:
        # (re-bind locals after the fall-through guard above so the
        # block reads as one unit; the variables door_polygon,
        # door_center, etc. are still in scope from the build above.)

        pixel_polygon: List[Tuple[float, float]] = []
        if pixel_door is not None and pixel_door.width_m > 0.30:
            pix_axis = np.array(pixel_door.axis_xy, dtype=np.float64)
            pix_axis = pix_axis / max(np.linalg.norm(pix_axis), 1e-6)
            pix_normal = np.array([-pix_axis[1], pix_axis[0]])
            pix_center = np.array(pixel_door.center_xy_m, dtype=np.float64)
            pixel_polygon = build_rectangle_xy(
                pix_center, pix_axis, pix_normal,
                pixel_door.width_m, max(door_depth, DOOR_DEPTH_FLOOR_M),
            )

        inside_pct, between_score, _wall_overlap, val_note = validate_placement(
            door_polygon, bbox_m, w1_arc, None, major=major, origin=origin,
        )

        # Tag-6.3 wall-anchor validation. closed_xy is the arc tip on
        # the wall surface; if it sits within WALL_ANCHOR_TOL_M of any
        # wall polygon, the symbol is genuinely wall-anchored and the
        # placement is trustworthy regardless of the annotator bbox.
        n_anchored, anchor_walls = validate_wall_anchors(
            [tuple(closed_xy)], wall_polygons,
        )

        # Tag-6.10 DIN flags — propagate radius flag from the arc and
        # add a wall-thickness flag when the local thickness exceeds
        # DIN residential guidance. Algorithm decisions above are
        # already complete; flags are advisory output only.
        if door_arc is not None:
            placement.din_flags.extend(door_arc.din_flags)
        if door_depth > DIN_THICKNESS_MAX_M:
            placement.din_flags.append(
                f"thick_wall_{door_depth:.2f}m"
            )
        if door_width > DIN_GAP_MAX_M:
            placement.din_flags.append(
                f"oversize_opening_{door_width:.2f}m"
            )

        placement.door_width_m = door_width
        placement.door_depth_m = door_depth
        placement.door_polygon_xy = door_polygon
        placement.pixel_polygon_xy = pixel_polygon
        placement.width_source = "arc_centered_hinge"
        placement.inside_bbox_pct = inside_pct
        placement.wall_overlap_pct = between_score
        placement.inside_bbox = inside_pct >= 0.50
        placement.on_wall = between_score >= 0.5
        bbox_2d_pct = _bbox_2d_overlap_pct(door_polygon, bbox_m)
        in_bbox_2d = bbox_2d_pct >= 0.10
        # Tag-6.3 Symbol-First override: arc detection + wall anchor
        # together ARE the ground truth, not the annotator bbox. When
        # the closed-tip is anchored on a wall (n_anchored >= 1), the
        # bbox-2D-overlap rule (Tag-5.1) drops from a hard-rule to a
        # soft signal — the bbox can sit on the room side of the wall
        # while the door void runs through the wall midline (Class-B
        # Matterport convention).
        # Tag-6.7 (Willi 2026-04-28 review): Symbol-First applies only
        # when the placed door at least minimally touches its bbox in
        # 2D (>= 5%). If the rectangle sits ENTIRELY outside the bbox
        # — as for mo6zzgqv_z where RANSAC picked up a neighbouring
        # door's arc through dense Bath-cluster bbox proximity —
        # n_anchored is misleading and the override would silently
        # confirm a wrong placement. Below the minimum, fall back to
        # the strict Tag-5.1 rule so the issue is visible as FAIL.
        SYMBOL_FIRST_MIN_BBOX_OVERLAP = 0.05
        if n_anchored >= 1 and bbox_2d_pct >= SYMBOL_FIRST_MIN_BBOX_OVERLAP:
            placement.pass_validation = (
                placement.inside_bbox or placement.on_wall
            )
        else:
            placement.pass_validation = (
                (placement.inside_bbox or placement.on_wall) and in_bbox_2d
            )
        anchor_tag = (
            f"anchor=W{anchor_walls[0]}" if n_anchored >= 1
            else "anchor=NONE"
        )
        placement.validation_note = f"{val_note}; {anchor_tag}"
        # Tag-7.7 — feed the arc-detected hinge centre into the
        # cascade-agnostic voting that runs after place_door.
        placement.arc_hinge_xy_m = (float(hinge_xy[0]), float(hinge_xy[1]))
        placement.note = (
            f"axis=arc_centered_hinge "
            f"hinge=({hinge_xy[0]:.2f},{hinge_xy[1]:.2f}) "
            f"radius={a_radius:.3f}m depth={door_depth:.3f}m"
            f"{debug_perp}{debug_arc} {anchor_tag}"
        )
        return placement

    gap_pick = None
    debug_gap = ""
    if ranked and ranked[0][1] <= FLANK_W1_MAX_M and corridor_pick is None:
        # Prefer the panel-endpoint hinge wall over the closest-by-distance
        # wall: at L-corners the closest wall may be a short segment that
        # ends inside the bbox, while the actual hinge wall continues
        # through the door opening on both sides.
        if start_pick_pre is not None:
            w1_idx_g = start_pick_pre[0]
            w1_g = wall_polygons[w1_idx_g]
            sw_origin = start_pick_pre[3]
            sw_tang = start_pick_pre[4]
            g_origin = sw_origin
            g_major = sw_tang
            g_minor = np.array([-g_major[1], g_major[0]], dtype=np.float64)
        else:
            w1_idx_g = ranked[0][0]
            w1_g = wall_polygons[w1_idx_g]
            g_origin, g_major, g_minor = nearest_edge_axis(w1_g, bbox_center)
        # Sign minor so it points INTO the wall body, not into the room
        # where door symbols (arc + panel) live.
        g_sign = _wall_interior_sign(w1_g, g_origin, g_minor)
        g_minor_into = g_sign * g_minor
        g_thick = local_wall_depth(
            w1_g, bbox_center, g_major, g_minor, g_origin,
            door_extent_m=0.90,
            wall_mask=wall_mask, px_to_m=px_to_m,
        )
        bbox_pts_g = np.array([
            [bbox_m[0], bbox_m[1]], [bbox_m[2], bbox_m[1]],
            [bbox_m[2], bbox_m[3]], [bbox_m[0], bbox_m[3]],
        ])
        bbox_t_g = (bbox_pts_g - g_origin) @ g_major
        s_lo_bbox = float(bbox_t_g.min())
        s_hi_bbox = float(bbox_t_g.max())
        # Tag-4.3 Lead 2 (Ahmed et al. 2013): classify the hinge wall as
        # solid-filled (dark interior) vs outlined (light interior) and
        # route profile_wall_outline to the correct polarity.
        wall_is_solid_g = classify_wall_solidness(
            image_gray, wall_polygons[w1_idx_g], wall_mask,
            px_to_m=px_to_m,
        )
        s_vals_g, profile_g = profile_wall_outline(
            image_gray, g_origin, g_major, g_minor_into,
            s_lo_bbox - 0.30, s_hi_bbox + 0.30,
            g_thick, px_to_m,
            wall_is_solid=wall_is_solid_g,
        )
        gap_g = find_door_gap(s_vals_g, profile_g, s_lo_bbox, s_hi_bbox)
        bbox_extent = s_hi_bbox - s_lo_bbox
        # Reject a profile-detected gap that overlaps the bbox by less than
        # 50 % of its own length — that is typically a sliding-door pocket
        # adjacent to the panel (the open-position channel), not the door
        # opening itself. The bbox is the annotator's confidence region;
        # the gap must lie mostly within it to be the door we are placing.
        if gap_g is not None:
            g_lo, g_hi, _ = gap_g
            g_len = max(1e-6, g_hi - g_lo)
            ov_lo = max(g_lo, s_lo_bbox)
            ov_hi = min(g_hi, s_hi_bbox)
            ov = max(0.0, ov_hi - ov_lo)
            if ov / g_len < 0.50:
                gap_g = None

        if gap_g is not None:
            # Profile-based detection succeeded — use exact gap geometry.
            g_start_m, g_end_m, g_baseline = gap_g
            gap_pick = (
                w1_idx_g, g_origin, g_major, g_minor, g_thick,
                g_start_m, g_end_m, g_baseline,
            )
            polarity_tag = "solid" if wall_is_solid_g else "outlined"
            debug_gap = (
                f"  gap[W{w1_idx_g} "
                f"len={g_end_m-g_start_m:.3f}m base={g_baseline:.2f} "
                f"{polarity_tag}]"
            )
        elif (ranked[0][1] <= 0.10
              and 0.30 <= bbox_extent <= 1.80):
            # Tag-4 bbox-implicit fallback: when the profile detector
            # cannot resolve the gap (e.g., outlines too thin to register
            # against anti-aliasing noise) but the door bbox sits flush
            # against the hinge wall (≤10 cm), treat the bbox extent
            # along the wall tangent as the implied door opening.
            # The annotator's bbox brackets the door symbol; its
            # wall-parallel extent is the door width by construction.
            gap_pick = (
                w1_idx_g, g_origin, g_major, g_minor, g_thick,
                s_lo_bbox, s_hi_bbox, -1.0,
            )
            in_roi_d = (
                (s_vals_g >= s_lo_bbox - 0.30) &
                (s_vals_g <= s_hi_bbox + 0.30)
            )
            out_d = ~in_roi_d
            if profile_g.size and out_d.any():
                base_d = float(np.median(profile_g[out_d]))
            elif profile_g.size:
                base_d = float(np.median(profile_g))
            else:
                base_d = 0.0
            min_d = float(profile_g.min()) if profile_g.size else 0.0
            longest_run_m = 0.0
            if profile_g.size and base_d > 0:
                thr_d = 0.30 * base_d
                gap_d = (profile_g < thr_d) & in_roi_d
                cur = 0
                for v in gap_d:
                    if v:
                        cur += 1
                        if cur > longest_run_m:
                            longest_run_m = cur
                    else:
                        cur = 0
                step_m_d = (
                    (s_vals_g[-1] - s_vals_g[0])
                    / max(1, len(s_vals_g) - 1)
                )
                longest_run_m = longest_run_m * step_m_d
            polarity_tag = "solid" if wall_is_solid_g else "outlined"
            debug_gap = (
                f"  gap-bbox[W{w1_idx_g} len={bbox_extent:.3f}m "
                f"profile_failed: base={base_d:.3f} min={min_d:.3f} "
                f"run={longest_run_m:.3f}m {polarity_tag}]"
            )
        else:
            in_roi_d = (
                (s_vals_g >= s_lo_bbox - 0.30) &
                (s_vals_g <= s_hi_bbox + 0.30)
            )
            out_d = ~in_roi_d
            if profile_g.size and out_d.any():
                base_d = float(np.median(profile_g[out_d]))
            elif profile_g.size:
                base_d = float(np.median(profile_g))
            else:
                base_d = 0.0
            min_d = float(profile_g.min()) if profile_g.size else 0.0
            # Find longest sub-threshold run for diagnosis
            longest_run_m = 0.0
            if profile_g.size and base_d > 0:
                thr_d = 0.40 * base_d
                in_roi_mask = in_roi_d
                gap_d = (profile_g < thr_d) & in_roi_mask
                cur = 0
                for v in gap_d:
                    if v:
                        cur += 1
                        if cur > longest_run_m:
                            longest_run_m = cur
                    else:
                        cur = 0
                step_m_d = (
                    (s_vals_g[-1] - s_vals_g[0]) / max(1, len(s_vals_g) - 1)
                )
                longest_run_m = longest_run_m * step_m_d
            debug_gap = (
                f"  gap-miss[W{w1_idx_g} thick={g_thick:.3f}m "
                f"base={base_d:.3f} min={min_d:.3f} run={longest_run_m:.3f}m]"
            )

    if gap_pick is not None:
        # Tag-4 primary path — door = wall-outline gap on W1.
        (w1_idx_g, g_origin, g_major, g_minor, g_thick,
         g_start_m, g_end_m, g_baseline) = gap_pick

        placement.w1_idx = w1_idx_g
        placement.w2_idx = None
        placement.w1_distance_m = ranked[0][1]
        placement.w2_distance_m = None

        w1_g = wall_polygons[w1_idx_g]
        origin = g_origin
        major = g_major
        minor = g_minor

        door_width = g_end_m - g_start_m
        t_center = 0.5 * (g_start_m + g_end_m)
        door_depth = g_thick
        # Tag-5.1 (Willi-review): shift s_center so the door rectangle
        # actually overlaps its label bbox in 2D. The default s_center=0
        # puts the door on the wall midline, which can leave it
        # *completely* outside the bbox when the bbox sits a few cm off
        # the wall surface (Class-B Matterport convention). We test a
        # ±door_depth range and pick the s_center that maximises bbox-
        # 2D overlap, with a small bias toward s_center=0 so on-wall
        # placements aren't unnecessarily shifted.
        bbox_pts_2d = np.array([
            [bbox_m[0], bbox_m[1]], [bbox_m[2], bbox_m[1]],
            [bbox_m[2], bbox_m[3]], [bbox_m[0], bbox_m[3]],
        ])
        bbox_s = (bbox_pts_2d - origin) @ minor
        bbox_s_lo = float(bbox_s.min())
        bbox_s_hi = float(bbox_s.max())
        s_center = _shift_s_into_bbox(
            origin, major, minor, t_center, door_width, door_depth,
            bbox_m, bbox_s_lo, bbox_s_hi,
        )

        door_center = origin + t_center * major + s_center * minor
        door_polygon = build_rectangle_xy(
            door_center, major, minor, door_width, door_depth
        )

        pixel_polygon: List[Tuple[float, float]] = []
        if pixel_door is not None and pixel_door.width_m > 0.30:
            pix_axis = np.array(pixel_door.axis_xy, dtype=np.float64)
            pix_axis = pix_axis / max(np.linalg.norm(pix_axis), 1e-6)
            pix_normal = np.array([-pix_axis[1], pix_axis[0]])
            pix_center = np.array(pixel_door.center_xy_m, dtype=np.float64)
            pixel_polygon = build_rectangle_xy(
                pix_center, pix_axis, pix_normal,
                pixel_door.width_m, max(door_depth, DOOR_DEPTH_FLOOR_M),
            )
            placement.pixel_width_m = pixel_door.width_m

        inside_pct, between_score, _wall_overlap, val_note = validate_placement(
            door_polygon, bbox_m, w1_g, None, major=major, origin=origin,
        )

        placement.door_width_m = door_width
        placement.door_depth_m = door_depth
        placement.door_polygon_xy = door_polygon
        placement.pixel_polygon_xy = pixel_polygon
        placement.width_source = (
            "wall_outline_gap" if g_baseline >= 0 else "bbox_extent_on_hinge"
        )
        placement.inside_bbox_pct = inside_pct
        placement.wall_overlap_pct = between_score
        placement.inside_bbox = inside_pct >= 0.50
        placement.on_wall = between_score >= 0.5
        # Gap-path success criterion: the door sits on the wall (the gap IS
        # the wall opening, so on_wall is the strong signal) AND the bbox
        # actually overlaps the door (inside_bbox_pct >= 0.50). The bbox is
        # an ROI hint, not ground truth — strict bbox-containment would
        # reject perfectly-placed doors when the symbol envelope is offset.
        # Tag-5.1: hard-rule no element entirely outside its label bbox in 2D.
        bbox_2d_pct = _bbox_2d_overlap_pct(door_polygon, bbox_m)
        in_bbox_2d = bbox_2d_pct >= 0.10
        placement.pass_validation = (
            placement.on_wall and placement.inside_bbox and in_bbox_2d
        )
        placement.validation_note = val_note
        path_label = (
            "wall_outline_gap" if g_baseline >= 0 else "bbox_extent_on_hinge"
        )
        placement.note = (
            f"axis={path_label} "
            f"gap=[{g_start_m:.3f},{g_end_m:.3f}]m "
            f"width={door_width:.3f}m depth={door_depth:.3f}m "
            f"baseline={g_baseline:.2f}{debug_perp}{debug_gap}"
        )
        return placement

    # Willi 2026-04-27: the start-wall (Anfangswand) was already computed
    # above (start_pick_pre) so Tag-4 could use it as the gap-detection
    # wall. Tag-3.2 cascade reuses it here under the original name.
    start_pick = start_pick_pre

    if corridor_pick is not None:
        cor_pos_idx, cor_neg_idx, cor_gap_len, cor_mid_off, cor_axis = corridor_pick
        w1_idx = cor_pos_idx
        w2_idx = cor_neg_idx
        w1_d = next((d for i, d in ranked if i == cor_pos_idx), 0.0)
        w2_d = next((d for i, d in ranked if i == cor_neg_idx), 0.0)
    elif start_pick is not None:
        sw_idx, hinge_ep, free_ep, sw_origin, sw_tang = start_pick
        w1_idx = sw_idx
        w1_d = next(
            (d for i, d in ranked if i == sw_idx), 0.0
        )
        # W2 = next-closest wall excluding the start wall.
        w2_pool = [t for t in ranked if t[0] != sw_idx]
        if w2_pool and w2_pool[0][1] <= FLANK_W2_MAX_M:
            w2_idx, w2_d = w2_pool[0]
        else:
            w2_idx, w2_d = None, None
    else:
        w1_idx, w1_d = ranked[0]
        if len(ranked) < 2 or ranked[1][1] > FLANK_W2_MAX_M:
            w2_idx, w2_d = None, None
        else:
            w2_idx, w2_d = ranked[1]

    placement.w1_idx = w1_idx
    placement.w2_idx = w2_idx
    placement.w1_distance_m = w1_d if w1_d is not None else 0.0
    placement.w2_distance_m = w2_d

    w1 = wall_polygons[w1_idx]
    w2 = wall_polygons[w2_idx] if w2_idx is not None else None

    if door_arc is not None:
        placement.pixel_width_m = door_arc.radius_m
    elif pixel_door is not None:
        placement.pixel_width_m = pixel_door.width_m

    if corridor_pick is not None:
        # Tag-3: corridor geometry — door axis = pixel_axis_unit, runs in
        # the gap between two perpendicular walls. Origin is bbox_center so
        # validation projections compare door extent to bbox extent on the
        # same anchor (cor_mid_off is the offset to the gap centre along
        # major).
        origin = bbox_center.copy()
        major = cor_axis.copy()
        minor = np.array([-major[1], major[0]])
        axis_source = "corridor"
    elif start_pick is not None:
        # Anchor: door axis = start-wall tangent at the hinge endpoint.
        origin = sw_origin
        major = sw_tang
        minor = np.array([-major[1], major[0]])
        axis_source = "panel_endpoint"
    else:
        origin, major, minor = nearest_edge_axis(w1, bbox_center)
        axis_source = "wall_edge"

    if door_arc is not None and corridor_pick is None:
        # Arc-derived along-wall direction = hinge → closed-endpoint. Use it
        # only when it is roughly parallel (≤ 25°) to the nearest wall edge,
        # otherwise the arc fit is unreliable and we trust the wall edge.
        # Skipped for corridor case where the axis comes from the gap geometry.
        hx, hy = door_arc.hinge_xy_m
        cx, cy = door_arc.closed_xy_m
        ax = np.array([cx - hx, cy - hy], dtype=np.float64)
        n = float(np.linalg.norm(ax))
        if n > 1e-6:
            ax_unit = ax / n
            cos_a = abs(float(ax_unit @ major))
            if cos_a >= math.cos(math.radians(25.0)):
                major = ax_unit
                minor = np.array([-major[1], major[0]])
                axis_source = "arc"

    # Width selection: pixel extraction is the door's natural geometry;
    # gap-between-walls only takes over for clear between-wall cases.
    # Walls bleiben unangetastet -- Tuer wird auf gap_length skaliert
    # (per Willi) wenn klar between-walls; sonst pixel-Breite.
    bbox_pts = np.array([
        [bbox_m[0], bbox_m[1]], [bbox_m[2], bbox_m[1]],
        [bbox_m[2], bbox_m[3]], [bbox_m[0], bbox_m[3]],
    ])
    bbox_t = _project(bbox_pts, origin, major)
    bbox_axial_extent = float(bbox_t.max() - bbox_t.min())

    if corridor_pick is not None:
        # Corridor placement: gap_start/gap_end straddle bbox_center along
        # the panel direction; the wall-pair gap_extent_along_axis routine
        # is bypassed (it assumes major = wall tangent).
        gap_length = cor_gap_len
        gap_mid = cor_mid_off
        gap_start = cor_mid_off - 0.5 * cor_gap_len
        gap_end = cor_mid_off + 0.5 * cor_gap_len
        gap_source = "corridor"
    else:
        gap_start, gap_end, gap_source = gap_extent_along_axis(
            w1, w2, bbox_m, major, origin
        )
        gap_length = max(0.0, gap_end - gap_start)
        gap_mid = 0.5 * (gap_start + gap_end)

    if corridor_pick is not None:
        # Door fills the gap between the two corridor walls. Width = pixel
        # panel length when it spans the gap to within 15 %; otherwise we
        # expand to the gap so the door reaches both walls.
        if (pixel_door is not None and pixel_door.width_m > 0.30
                and pixel_door.width_m >= 0.85 * cor_gap_len):
            door_width = pixel_door.width_m
            width_source = "corridor_pixel_width"
        else:
            door_width = cor_gap_len
            width_source = "corridor_gap_filled"
        t_center = cor_mid_off
    elif start_pick is not None:
        # Willi-anchored placement: hinge sits ON the start wall surface.
        # The door void runs ALONG the start wall from the hinge outward,
        # in the direction of the bbox centroid (heuristic for swing
        # direction in the absence of a reliable arc).
        door_width = pixel_door.width_m
        hinge_along = float((hinge_ep - origin) @ major)
        bbox_along = float((bbox_center - origin) @ major)
        if bbox_along >= hinge_along:
            t_center = hinge_along + 0.5 * door_width
        else:
            t_center = hinge_along - 0.5 * door_width
        width_source = "panel_anchored_at_hinge"
    elif door_arc is not None:
        # Door rectangle goes from hinge to closed-endpoint along the
        # arc-derived axis. Width = radius. Centre = midpoint of that
        # segment (= hinge + 0.5*radius along axis).
        door_width = door_arc.radius_m
        hinge = np.array(door_arc.hinge_xy_m)
        midpoint_world = hinge + 0.5 * door_width * major
        t_center = float((midpoint_world - origin) @ major)
        width_source = "arc"
    elif pixel_door is not None and pixel_door.width_m > 0.30:
        door_width = pixel_door.width_m
        if gap_source == "between":
            # Door belongs in the gap between W1 and W2 along the wall.
            # Center on gap_mid; if pixel-extracted door is much narrower
            # than the gap (door_double labels often capture only ONE
            # panel with HoughLinesP), expand to fill the gap so the
            # void matches the real opening.
            if door_width < 0.85 * gap_length:
                door_width = gap_length
                width_source = "gap_between_walls(expanded)"
            else:
                width_source = "pixel_line_at_gap_center"
            t_center = gap_mid
        else:
            t_center = float((bbox_center - origin) @ major)
            width_source = "pixel_line_at_bbox_center"
    elif gap_source == "between" and gap_length > 0.05:
        door_width = gap_length
        t_center = gap_mid
        width_source = "gap_between_walls"
    else:
        door_width = bbox_axial_extent
        t_center = float((bbox_center - origin) @ major)
        width_source = "bbox_axis_fallback"

    # Bbox-Clamp (Willi 2026-04-27): the door must never extend outside the
    # label bbox along the wall, EXCEPT when the placement is anchored by
    # ground-truth geometry — the gap_mid for between-walls cases or the
    # panel hinge endpoint for Willi's start-wall algorithm. In those
    # cases the bbox is only an approximate symbol envelope.
    bbox_lo_t = float(bbox_t.min())
    bbox_hi_t = float(bbox_t.max())
    skip_bbox_clamp = (
        gap_source in ("between", "corridor")
        or start_pick is not None
        or corridor_pick is not None
    )
    if door_width > bbox_axial_extent and not skip_bbox_clamp:
        door_width = bbox_axial_extent
        width_source = f"{width_source}+capped_to_bbox"
    half_w = 0.5 * door_width
    if not skip_bbox_clamp:
        t_lower = bbox_lo_t + half_w
        t_upper = bbox_hi_t - half_w
        if t_upper >= t_lower:
            clamped_t = max(t_lower, min(t_upper, t_center))
            if abs(clamped_t - t_center) > 1e-3:
                width_source = f"{width_source}+clamped"
            t_center = clamped_t

    # Door centre along minor (Willi 2026-04-27 review):
    # The label bbox surrounds the door SYMBOL (panel + arc), which sits in
    # the room next to the wall surface — not centred on the wall midline.
    # The default s_center = 0 places the rectangle straddling the wall
    # surface (its near edge in the room, its far edge inside the wall),
    # so the 3D Boolean still cuts a real void. The Tag-3 minor-axis
    # clamp below shifts s_center toward the bbox/room when the rectangle
    # would otherwise sit entirely inside the wall polygon (common for
    # L/T-shaped Phase-2 walls whose nearest long edge is buried).
    s_center = 0.0

    # Door depth: probe wall thickness around bbox_center (for the wall
    # case) or use the bbox-perp extent (for the corridor case where the
    # door does not cut a single wall).
    if corridor_pick is not None:
        bbox_s_corridor = (bbox_pts - origin) @ minor
        bbox_perp_extent = float(bbox_s_corridor.max() - bbox_s_corridor.min())
        door_depth = max(
            min(bbox_perp_extent, ALGO_THICKNESS_MAX_M),
            DOOR_DEPTH_FLOOR_M,
        )
    else:
        # Probe at bbox_center first so door_depth is independent of the
        # minor-axis clamp below; fall back to door_center if needed.
        door_depth = local_wall_depth(
            w1, bbox_center, major, minor, origin,
            door_extent_m=door_width,
            wall_mask=wall_mask, px_to_m=px_to_m,
        )

    # Tag-5.1 (Willi-review): pick s_center to maximise bbox-2D overlap.
    # The previous half-depth clamp left some doors entirely outside the
    # label bbox along minor when ``origin`` sat on a wall whose surface
    # is several cm offset from the bbox in the perpendicular direction.
    # The new helper tests the wall midline, the bbox midline, and a
    # few interior offsets, returning the s_center that maximises 2D
    # overlap (with a small bias to s=0 so on-wall placements aren't
    # unnecessarily shifted).
    if corridor_pick is None:
        bbox_s = (bbox_pts - origin) @ minor
        s_center = _shift_s_into_bbox(
            origin, major, minor, t_center, door_width, door_depth,
            bbox_m, float(bbox_s.min()), float(bbox_s.max()),
        )

    door_center = origin + t_center * major + s_center * minor

    door_polygon = build_rectangle_xy(
        door_center, major, minor, door_width, door_depth
    )

    pixel_polygon: List[Tuple[float, float]] = []
    if pixel_door and pixel_door.width_m > 0.30:
        pix_axis = np.array(pixel_door.axis_xy)
        pix_axis = pix_axis / max(np.linalg.norm(pix_axis), 1e-6)
        pix_normal = np.array([-pix_axis[1], pix_axis[0]])
        pix_center = np.array(pixel_door.center_xy_m)
        pixel_polygon = build_rectangle_xy(
            pix_center, pix_axis, pix_normal,
            pixel_door.width_m, max(door_depth, DOOR_DEPTH_FLOOR_M),
        )

    # Validation against the label bbox + flanking walls (Willi 2026-04-27):
    # door is correct only if its major-axis extent lies inside the bbox AND
    # the door touches a wall (or sits in a gap between two walls).
    inside_pct, between_score, wall_overlap_pct, val_note = validate_placement(
        door_polygon, bbox_m, w1, w2,
        major=major, origin=origin,
    )

    # Tag-3.2 (Willi 2026-04-27 root-cause fix): the panel_aligned branch
    # introduced in Tag-3 was based on a misreading of "orthogonal zum
    # pixel extracted" -- it rotated the rectangle PARALLEL to the cyan
    # pixel-extracted line, while Willi meant ORTHOGONAL. For 90°-open
    # door symbols the panel sticks perpendicular to the wall, so the
    # door void runs ALONG the wall tangent (= orthogonal to pixel_axis,
    # = orthogonal to the cyan rectangle). The wall-aligned / panel_endpoint
    # placement (axis = wall tangent) already produces this orientation
    # for every door. The minor-axis bbox clamp above shifts the
    # orthogonal rectangle into the bbox without changing its
    # orientation. The panel_aligned branch was therefore both
    # geometrically wrong and unnecessary -- removed.

    placement.door_width_m = door_width
    placement.door_depth_m = door_depth
    placement.door_polygon_xy = door_polygon
    placement.pixel_polygon_xy = pixel_polygon
    placement.width_source = width_source

    placement.inside_bbox_pct = inside_pct
    placement.wall_overlap_pct = between_score
    # Inside-bbox threshold relaxed to 0.85 because the bbox is the SYMBOL
    # envelope (panel + arc) while the door void sits on the wall midline;
    # for between-walls cases the gap is the ground-truth position and a
    # ~10 % overhang past the bbox is acceptable when the door is perfectly
    # centred in the gap.
    #
    # Tag-3.2: pass_validation drops the on_wall requirement. When the
    # bbox label convention places the symbol envelope on the room side
    # only (Matterport: d_polygon(bbox, wall) > NEAR_TOL_M), the
    # orthogonal-to-pixel-axis rectangle inside the bbox is geometrically
    # forced off the wall surface in 2D -- there is no rectangle that is
    # simultaneously orthogonal-to-pixel-axis, inside the bbox, and
    # touching the wall. The pre-condition check at the top of place_door
    # already rejects doors with no W1 within FLANK_W1_MAX_M = 0.50 m, so
    # reaching validation guarantees a wall is nearby; on_wall stays as
    # informational only. The 3D Boolean cut step (Tag-4) computes the
    # void from W1 + panel hinge endpoint directly, independent of the
    # 2D rectangle position.
    placement.inside_bbox = inside_pct >= 0.85
    placement.on_wall = between_score >= 0.5
    # Tag-5.1: hard-rule no element entirely outside its label bbox in 2D.
    bbox_2d_pct = _bbox_2d_overlap_pct(door_polygon, bbox_m)
    in_bbox_2d = bbox_2d_pct >= 0.10
    placement.pass_validation = placement.inside_bbox and in_bbox_2d
    placement.validation_note = val_note

    placement.note = (
        f"axis={axis_source} gap_source={gap_source} "
        f"gap_len={gap_length:.3f}m  "
        f"pixel_w={placement.pixel_width_m:.3f}m  "
        f"depth={door_depth:.3f}m  width_src={width_source}"
        f"{debug_perp}{debug_corridor}{debug_gap}"
    )
    return placement


def validate_placement(
    door_polygon: List[Tuple[float, float]],
    bbox_m: Tuple[float, float, float, float],
    w1: WallPolygon,
    w2: Optional[WallPolygon],
    *,
    major: Optional[np.ndarray] = None,
    origin: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, str]:
    """Cross-check the placed door against the label bbox and flanking walls.

    The door is a void IN the wall, so half its area sits in the wall while
    the bbox surrounds the door SYMBOL (panel + arc) on the room side. A
    pure 2D-area inside-bbox check therefore tops out at ~50% even for a
    perfectly placed Class-A door. We instead validate ALONG the wall axis
    (major): the door's major-axis extent must be inside the bbox's
    major-axis extent. Perpendicular-direction sanity is covered by
    ``between_walls_score`` (door touches/in a wall).

    Returns ``(inside_bbox_pct, between_walls_score, wall_overlap_pct, note)``:
      * ``inside_bbox_pct`` — fraction of the door's MAJOR-axis projection
        that lies inside the bbox's MAJOR-axis projection (≥ 0.95 = ok).
      * ``between_walls_score`` — 1.0 when the door touches or cuts a wall;
        decays to 0 at 0.10 m clearance.
      * ``wall_overlap_pct`` — fraction of the door's 2D area that sits
        inside W1/W2 polygons (Tag-3: > 0.30 means the door is "deep in
        wall" and the panel-aligned recompute should kick in).
      * ``note`` — short diagnostic string for the console.
    """
    if len(door_polygon) < 3:
        return 0.0, 0.0, 0.0, "no door polygon"
    door_poly = Polygon(door_polygon).buffer(0)
    if not door_poly.is_valid or door_poly.is_empty or door_poly.area <= 0:
        return 0.0, 0.0, 0.0, "invalid door polygon"

    door_pts = np.asarray(door_polygon, dtype=np.float64)
    bbox_pts = np.array([
        [bbox_m[0], bbox_m[1]], [bbox_m[2], bbox_m[1]],
        [bbox_m[2], bbox_m[3]], [bbox_m[0], bbox_m[3]],
    ])
    if major is None or origin is None:
        # Fall back to door's own principal axis.
        _, major_eff, _ = _pca_axes(door_pts)
        origin_eff = door_pts.mean(axis=0)
    else:
        major_eff = major
        origin_eff = origin

    door_t = (door_pts - origin_eff) @ major_eff
    bbox_t = (bbox_pts - origin_eff) @ major_eff
    door_lo, door_hi = float(door_t.min()), float(door_t.max())
    bbox_lo, bbox_hi = float(bbox_t.min()), float(bbox_t.max())
    door_extent = max(1e-6, door_hi - door_lo)
    overlap_lo = max(door_lo, bbox_lo)
    overlap_hi = min(door_hi, bbox_hi)
    overlap = max(0.0, overlap_hi - overlap_lo)
    inside_pct = overlap / door_extent

    walls: List[Polygon] = []
    for wp in (w1, w2):
        if wp is None or len(wp.shell_xy) < 3:
            continue
        poly = Polygon(wp.shell_xy).buffer(0)
        if poly.is_valid and not poly.is_empty:
            walls.append(poly)
    if not walls:
        return inside_pct, 0.0, 0.0, "no walls"

    overlap_area = sum(
        float(door_poly.intersection(w).area) for w in walls
    )
    overlap_pct = overlap_area / door_poly.area
    nearest_dist = min(float(w.distance(door_poly)) for w in walls)
    NEAR_TOL_M = 0.10
    if overlap_pct > 0.02 or nearest_dist == 0.0:
        between_score = 1.0
    elif nearest_dist <= NEAR_TOL_M:
        between_score = max(0.0, 1.0 - nearest_dist / NEAR_TOL_M)
    else:
        between_score = 0.0

    parts: List[str] = []
    if inside_pct >= 0.99:
        parts.append("fully in bbox (major)")
    elif inside_pct >= 0.90:
        parts.append(f"mostly in bbox ({inside_pct*100:.0f}% along wall)")
    else:
        parts.append(f"OUT along wall ({inside_pct*100:.0f}% in)")
    if overlap_pct > 0.02:
        parts.append(f"in wall {overlap_pct*100:.0f}%")
    elif nearest_dist <= NEAR_TOL_M:
        parts.append(f"in gap (d={nearest_dist*1000:.0f}mm)")
    else:
        parts.append(f"OFF wall (d={nearest_dist*1000:.0f}mm)")

    # Tag-5.1 (Willi-review): 2D bbox overlap. Doors must intersect the
    # label bbox at least partially in 2D — never sit completely outside
    # (e.g. when the door snaps onto a wall whose surface is several cm
    # away from the bbox in the perpendicular direction). The major-axis
    # inside_pct check passes such cases because along the wall they are
    # fully inside, but they violate the hard constraint that the
    # placed element must stay overlapping its annotator-drawn region.
    from shapely.geometry import box as _box
    bbox_poly = _box(bbox_m[0], bbox_m[1], bbox_m[2], bbox_m[3])
    try:
        bbox_2d_overlap = float(door_poly.intersection(bbox_poly).area)
    except Exception:
        bbox_2d_overlap = 0.0
    bbox_2d_pct = bbox_2d_overlap / max(1e-9, door_poly.area)
    if bbox_2d_pct >= 0.20:
        parts.append(f"bbox-2D {bbox_2d_pct*100:.0f}%")
    elif bbox_2d_pct >= 0.10:
        parts.append(f"bbox-2D edge {bbox_2d_pct*100:.0f}%")
    else:
        parts.append(f"OUTSIDE bbox-2D ({bbox_2d_pct*100:.1f}%)")
    return inside_pct, between_score, overlap_pct, "; ".join(parts)


def _bbox_2d_overlap_pct(
    door_polygon: List[Tuple[float, float]],
    bbox_m: Tuple[float, float, float, float],
) -> float:
    """Fraction of the door's 2D area that lies inside the label bbox."""
    if len(door_polygon) < 3:
        return 0.0
    from shapely.geometry import box as _box
    try:
        d = Polygon(door_polygon).buffer(0)
        if d.is_empty or d.area <= 0:
            return 0.0
        b = _box(bbox_m[0], bbox_m[1], bbox_m[2], bbox_m[3])
        return float(d.intersection(b).area) / float(d.area)
    except Exception:
        return 0.0


# Tag-6.3 (Willi 2026-04-28): wall-anchor validation as a hard rule.
# DIN-consistent tolerance — a real door's hinge endpoint sits on the
# wall surface; raster discretisation, polygon simplification, and
# render anti-aliasing add up to ~8 cm worst-case offset. Anything
# beyond that is not snapped to a wall.
WALL_ANCHOR_TOL_M = 0.08


def _placement_confidence(p: "DoorPlacement") -> float:
    """Tag-6.4: composite confidence used to pick winners in
    pairwise conflicts. Symbol-derived placements outrank fallback
    geometry; bbox- and wall-overlap percentages add finer ordering.
    """
    by_source = {
        "arc_centered_hinge": 1.0,
        "wall_outline_gap": 0.7,
        "bbox_extent_on_hinge": 0.4,
    }
    score = by_source.get(p.width_source, 0.2)
    score += p.inside_bbox_pct * 0.2
    score += p.wall_overlap_pct * 0.1
    return score


def assign_door_wall_axis_attrs(
    placements: List["DoorPlacement"],
    wall_polygons: Sequence[WallPolygon],
) -> int:
    """Sprint Door-Wall-Bridge Tag-2: precompute (wall_depth_mesh_m,
    short_offset_m) for every door BEFORE the door-void cut.

    Mirrors the window-pipeline call ``assign_window_short_offsets``.
    The measured wall thickness comes from the post-refine BIM polygons
    (ray-cast perpendicular to the door axis); the centerline offset is
    used downstream by ``build_door_rectangle`` to place frame, void,
    and bridge on the wall midplane.

    Returns the count of placements where a non-zero offset was found
    (informational — call site decides whether to log).
    """
    n_offset = 0
    for p in placements:
        if not p.door_polygon_xy:
            continue
        d_actual, short_offset = measure_wall_axis_at_door(
            p.door_polygon_xy, wall_polygons,
        )
        if d_actual > 0.04:
            p.wall_depth_mesh_m = float(d_actual)
            p.short_offset_m = float(short_offset)
            if abs(short_offset) > 0.001:
                n_offset += 1
        else:
            p.wall_depth_mesh_m = 0.0
            p.short_offset_m = 0.0
    return n_offset


def resolve_door_conflicts(
    placements: List["DoorPlacement"],
    wall_polygons: Sequence[WallPolygon],
    *,
    min_distance_m: float = 0.10,
    max_iterations: int = 8,
) -> int:
    """Tag-6.4 greedy conflict resolver. When two placed doors are
    closer than ``min_distance_m``, shift the lower-confidence door
    along its own wall tangent until the gap is restored. The shift
    direction is chosen to push the movable door AWAY from the fixed
    door (signed projection onto the wall tangent). Up to 50 cm of
    travel in 1 cm steps; if no offset within that range clears the
    conflict, the placement is left untouched and Tag-6.5 will mark
    it FAIL so the issue stays visible.

    This is the simpler variant of the architectural plan's full
    bipartite-Hungarian reassignment to alternative wall edges - it
    handles the dominant case (multiple doors stacked on the same
    wall edge by an over-aggressive nearest-wall pick) without the
    coupling of a full re-placement pipeline. Wall-edge reassignment
    is deferred to a future sprint when symbol detection itself
    starts mis-snapping; today's Bath cluster is already structurally
    separated by Tag-6.0/6.3 so this resolver is dormant.

    Returns the number of placements that were shifted.
    """
    n_shifted = 0
    for _it in range(max_iterations):
        any_conflict = False
        for i, pa in enumerate(placements):
            if not pa.door_polygon_xy:
                continue
            try:
                poly_a = Polygon(pa.door_polygon_xy).buffer(0)
            except Exception:
                continue
            if poly_a.is_empty or poly_a.area <= 0:
                continue
            for j in range(i + 1, len(placements)):
                pb = placements[j]
                if not pb.door_polygon_xy:
                    continue
                try:
                    poly_b = Polygon(pb.door_polygon_xy).buffer(0)
                except Exception:
                    continue
                if poly_b.is_empty or poly_b.area <= 0:
                    continue
                d = float(poly_a.distance(poly_b))
                if d >= min_distance_m:
                    continue
                any_conflict = True
                conf_a = _placement_confidence(pa)
                conf_b = _placement_confidence(pb)
                movable = pb if conf_a >= conf_b else pa
                fixed = pa if conf_a >= conf_b else pb
                if _shift_door_apart_from(
                    movable, fixed,
                    min_distance_m=min_distance_m,
                    wall_polygons=wall_polygons,
                ):
                    n_shifted += 1
        if not any_conflict:
            break
    return n_shifted


def _shift_door_apart_from(
    movable: "DoorPlacement",
    fixed: "DoorPlacement",
    *,
    min_distance_m: float,
    wall_polygons: Sequence[WallPolygon],
    max_shift_m: float = 0.50,
    step_m: float = 0.01,
) -> bool:
    """Translate ``movable``'s door rectangle along its W1 wall tangent
    until its distance from ``fixed`` reaches ``min_distance_m``. Only
    shifts along the wall - never perpendicular - so the door stays
    snapped to its detected wall edge. Returns True if a clearing
    offset was found within ``max_shift_m``.
    """
    if movable.w1_idx < 0 or movable.w1_idx >= len(wall_polygons):
        return False
    try:
        poly_f = Polygon(fixed.door_polygon_xy).buffer(0)
        poly_m = Polygon(movable.door_polygon_xy).buffer(0)
    except Exception:
        return False
    if poly_f.is_empty or poly_m.is_empty:
        return False

    centroid_m = np.array([poly_m.centroid.x, poly_m.centroid.y])
    centroid_f = np.array([poly_f.centroid.x, poly_f.centroid.y])
    _, major, _ = nearest_edge_axis(
        wall_polygons[movable.w1_idx], centroid_m
    )
    direction = centroid_m - centroid_f
    sign = 1.0 if float(direction @ major) >= 0 else -1.0

    n_steps = int(max_shift_m / step_m)
    for k in range(1, n_steps + 1):
        delta = sign * k * step_m
        dx = delta * float(major[0])
        dy = delta * float(major[1])
        shifted = [
            (float(x + dx), float(y + dy))
            for x, y in movable.door_polygon_xy
        ]
        try:
            new_poly = Polygon(shifted).buffer(0)
        except Exception:
            continue
        if new_poly.is_empty:
            continue
        if float(new_poly.distance(poly_f)) >= min_distance_m:
            movable.door_polygon_xy = shifted
            movable.validation_note = (
                (movable.validation_note or "")
                + f"; resolved conflict by shifting {k*step_m*1000:.0f}mm"
            )
            return True
    return False


def validate_wall_anchors(
    anchor_points: Sequence[Tuple[float, float]],
    wall_polygons: Sequence[WallPolygon],
    *,
    tol_m: float = WALL_ANCHOR_TOL_M,
) -> Tuple[int, List[int]]:
    """Test door symbol anchor points against wall polygons.

    Willi's contract:
      * 1 anchor = NECESSARY condition. Single-arc swing doors are
        anchored at the closed-position arc tip on the hinge wall;
        one wall anchor is enough.
      * 2 anchors = SUFFICIENT condition. Double-arc doors (twin-panel
        symbols, French doors) anchor at both arc tips, on either the
        same wall or two opposite walls. Both anchors confirm the
        door spans a real opening.
      * 0 anchors = FAIL. The detected symbol is not on a wall;
        either the symbol detection went off (incidental circle in
        furniture) or the wall extrusion missed the wall the door
        sits on.

    Returns ``(n_anchored, wall_indices_per_anchor)``. Wall index = -1
    when the anchor exceeds ``tol_m`` from every wall polygon.
    """
    from shapely.geometry import Point as _Point
    n_anchored = 0
    wall_indices: List[int] = []
    for ax, ay in anchor_points:
        pt = _Point(float(ax), float(ay))
        best_d = float("inf")
        best_idx = -1
        for i, wp in enumerate(wall_polygons):
            if len(wp.shell_xy) < 3:
                continue
            try:
                d = float(Polygon(wp.shell_xy).buffer(0).distance(pt))
            except Exception:
                continue
            if d < best_d:
                best_d = d
                best_idx = i
        if best_d <= tol_m:
            n_anchored += 1
            wall_indices.append(best_idx)
        else:
            wall_indices.append(-1)
    return n_anchored, wall_indices


def _shift_s_into_bbox(
    origin: np.ndarray,
    major: np.ndarray,
    minor: np.ndarray,
    t_center: float,
    door_width: float,
    door_depth: float,
    bbox_m: Tuple[float, float, float, float],
    bbox_s_lo: float,
    bbox_s_hi: float,
    *,
    target_overlap: float = 0.25,
    max_search_m: float = 0.40,
) -> float:
    """Pick the SMALLEST |s_center| that meets the bbox-2D overlap target.

    Tag-5.1 (Willi-review): the placed door rectangle must overlap its
    label bbox in 2D — at least lightly. The default ``s_center = 0``
    puts the door on the wall midline; if that already gives the
    target overlap (≥ ``target_overlap``), we stay there and the door
    sits cleanly on the wall. Otherwise we walk |s| outward in 1 cm
    steps and pick the first offset that reaches the target. Walking
    in both directions (positive and negative) lets the door shift
    toward whichever side of the wall the bbox is on. If no offset
    within ``max_search_m`` reaches the target, we return the offset
    with the highest overlap so the validation reports the actual
    geometric truth (door still partially overlapping bbox).
    """
    def _ov_for(s: float) -> float:
        center = origin + t_center * major + s * minor
        rect = build_rectangle_xy(
            center, major, minor, door_width, door_depth
        )
        return _bbox_2d_overlap_pct(rect, bbox_m)

    # 1. If wall midline (s=0) already meets the target, use it.
    if _ov_for(0.0) >= target_overlap:
        return 0.0

    # 2. Walk outward in 1 cm steps in both directions; pick the
    # smallest |s| that reaches the target.
    step = 0.01
    n_steps = int(max_search_m / step) + 1
    best_s = 0.0
    best_ov = _ov_for(0.0)
    for k in range(1, n_steps + 1):
        d = k * step
        for s in (d, -d):
            ov = _ov_for(s)
            if ov >= target_overlap:
                return s
            if ov > best_ov:
                best_ov = ov
                best_s = s
    return best_s


# ── PNG render ──────────────────────────────────────────────────────────


# DIN 18101 standard door widths (metres) for visual scale verification.
_DIN_DOOR_WIDTHS_M = (0.610, 0.735, 0.860, 0.985, 1.235)
_DIN_TOL_RATIO = 0.05  # ±5 % to count as DIN-matching


def _din_match(width_m: float) -> bool:
    """True if ``width_m`` is within ±5 % of a DIN 18101 standard width."""
    if width_m <= 0:
        return False
    for w in _DIN_DOOR_WIDTHS_M:
        if abs(width_m - w) / w <= _DIN_TOL_RATIO:
            return True
    return False


def _polygon_longest_edge(
    shell_xy: List[Tuple[float, float]]
) -> Tuple[float, Tuple[float, float], Tuple[float, float]]:
    """Return (length_m, p_a, p_b) of the longest edge of ``shell_xy``."""
    best_len = 0.0
    best_a = best_b = (0.0, 0.0)
    n = len(shell_xy)
    for i in range(n):
        a = shell_xy[i]
        b = shell_xy[(i + 1) % n]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L > best_len:
            best_len = L
            best_a, best_b = a, b
    return best_len, best_a, best_b



def render_placement_png(
    wall_polygons: List[WallPolygon],
    placements: List[DoorPlacement],
    out_path: Path,
) -> None:
    if not wall_polygons:
        return
    all_xy = []
    for wp in wall_polygons:
        all_xy.extend(wp.shell_xy)
    for p in placements:
        all_xy.extend(p.door_polygon_xy)
        all_xy.extend(p.pixel_polygon_xy)
        all_xy.extend([
            (p.bbox_m[0], p.bbox_m[1]), (p.bbox_m[2], p.bbox_m[1]),
            (p.bbox_m[2], p.bbox_m[3]), (p.bbox_m[0], p.bbox_m[3]),
        ])
    xs = [p[0] for p in all_xy]
    ys = [p[1] for p in all_xy]
    pad = 0.6
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad

    px_per_m = 90.0
    W = max(int(round((xmax - xmin) * px_per_m)), 320)
    H = max(int(round((ymax - ymin) * px_per_m)), 320)

    img = np.full((H, W, 3), 245, dtype=np.uint8)

    def to_px(x_m: float, y_m: float) -> Tuple[int, int]:
        u = int(round((x_m - xmin) * px_per_m))
        v = int(round((y_m - ymin) * px_per_m))
        v = H - 1 - v
        return (u, v)

    for w_idx, wp in enumerate(wall_polygons):
        pts = np.array([to_px(x, y) for x, y in wp.shell_xy],
                       dtype=np.int32)
        cv2.fillPoly(img, [pts], (210, 210, 210))
        cv2.polylines(img, [pts], True, (60, 60, 60), 2)

        # Tag-5: annotate the wall's longest edge with its length so Willi
        # can visually verify the scale calibration.
        L, a, b = _polygon_longest_edge(wp.shell_xy)
        if L >= 0.30:
            mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
            mid_px = to_px(mid[0], mid[1])
            cv2.putText(
                img, f"{L:.2f}m", (mid_px[0] + 4, mid_px[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (110, 110, 110), 1, cv2.LINE_AA,
            )

        # Tag-7.12 — wall index label at the polygon centroid so the
        # reviewer can spot missing wall polygons by index when a
        # door's recovery search reports e.g. "all viable candidates
        # on W6". If the surrounding visual shows walls that should
        # exist but no W-label sits there, the reconstruction step
        # dropped them.
        try:
            shell_arr = np.array(wp.shell_xy, dtype=np.float64)
            centroid_xy = shell_arr.mean(axis=0)
            cpx = to_px(float(centroid_xy[0]), float(centroid_xy[1]))
            cv2.putText(
                img, f"W{w_idx}", (cpx[0] - 12, cpx[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (30, 90, 160), 1, cv2.LINE_AA,
            )
        except Exception:
            pass

    for p in placements:
        bx0, by0, bx1, by1 = p.bbox_m
        bbox_pts = np.array([
            to_px(bx0, by0), to_px(bx1, by0),
            to_px(bx1, by1), to_px(bx0, by1),
        ], dtype=np.int32)
        cv2.polylines(img, [bbox_pts], True, (220, 180, 120), 1)

        # Pixel-extracted rect (cyan, semi-transparent)
        if p.pixel_polygon_xy:
            d_pts = np.array(
                [to_px(x, y) for x, y in p.pixel_polygon_xy],
                dtype=np.int32,
            )
            overlay = img.copy()
            cv2.fillPoly(overlay, [d_pts], (220, 220, 0))  # cyan
            img[:] = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
            cv2.polylines(img, [d_pts], True, (160, 160, 0), 1)

        # Final placed door — fill orange always; border GREEN if pass, RED
        # if fail (Willi 2026-04-27: visual pass/fail vs label bbox + walls).
        if p.door_polygon_xy:
            d_pts = np.array(
                [to_px(x, y) for x, y in p.door_polygon_xy],
                dtype=np.int32,
            )
            cv2.fillPoly(img, [d_pts], (0, 140, 255))
            border = (40, 180, 40) if p.pass_validation else (40, 40, 220)
            tag4 = p.width_source in (
                "wall_outline_gap", "bbox_extent_on_hinge",
                "arc_centered_hinge", "twin_arc_centered",
            )
            border_thick = 3 if tag4 else 2
            cv2.polylines(img, [d_pts], True, border, border_thick)
            cx = int(d_pts[:, 0].mean())
            cy = int(d_pts[:, 1].mean())
            short_id = p.label_id.replace("ann_", "")[:10]
            tick = "OK" if p.pass_validation else "FAIL"
            if p.sliding_mount_position == "beside_wall":
                tag = "[BESIDE] "
            elif p.width_source == "arc_centered_hinge":
                tag = "[ARC] "
            elif p.width_source == "twin_arc_centered":
                tag = "[TWIN] "
            elif p.width_source == "wall_outline_gap":
                tag = "[GAP] "
            elif p.width_source == "bbox_extent_on_hinge":
                tag = "[BBOX] "
            else:
                tag = ""
            cv2.putText(
                img, f"{tick} {tag}{short_id}", (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (20, 20, 20), 1, cv2.LINE_AA,
            )
            # Tag-5: width annotation, colour-coded by DIN match
            din_ok = _din_match(p.door_width_m)
            width_color = (
                (40, 140, 40) if din_ok else (40, 100, 200)
            )
            cv2.putText(
                img, f"{p.door_width_m*1000:.0f}mm",
                (cx + 6, cy + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                width_color, 1, cv2.LINE_AA,
            )

            # Tag-7.12 — jamb-probe overlay. Show, for each long-axis
            # end of the placed door, the rectangular region the
            # algorithm tests for wall material (4 cm probe + 5 cm
            # snap buffer ≈ 9 cm reach). Green outline = the probe
            # found a wall on that side; red outline = no wall →
            # door is floating there. Lets Willi spot upstream
            # wall-reconstruction gaps: if visual walls exist around
            # a red probe, the wall_polygons list dropped them.
            try:
                pts2 = np.asarray(p.door_polygon_xy, dtype=np.float64)
                e01 = pts2[1] - pts2[0]
                e12 = pts2[2] - pts2[1]
                if np.linalg.norm(e01) >= np.linalg.norm(e12):
                    long_v_d = e01
                    long_l_d = float(np.linalg.norm(e01))
                    short_l_d = float(np.linalg.norm(e12))
                else:
                    long_v_d = e12
                    long_l_d = float(np.linalg.norm(e12))
                    short_l_d = float(np.linalg.norm(e01))
                if long_l_d > 1e-6:
                    long_u_d = long_v_d / long_l_d
                    short_u_d = np.array([-long_u_d[1], long_u_d[0]])
                    door_c2 = pts2.mean(axis=0)
                    half_W_d = long_l_d * 0.5
                    half_S_d = max(short_l_d, 0.10) * 0.5
                    probe_long_d = max(0.04, 0.04)
                    snap_d = 0.05
                    flag_minus_plus = (
                        bool(p.jamb_anchor_minus),
                        bool(p.jamb_anchor_plus),
                    )
                    for sign_idx, sign_d in enumerate((-1.0, +1.0)):
                        ec = door_c2 + sign_d * (
                            half_W_d + probe_long_d * 0.5
                        ) * long_u_d
                        # Inflate probe by snap_tolerance for the
                        # visualisation so the buffer that is part of
                        # the actual shapely test shows up.
                        ec_long_extent = probe_long_d + 2 * snap_d
                        ec_short_extent = 2 * (half_S_d + snap_d)
                        probe_corners_buf = [
                            ec + sign_d * (ec_long_extent * 0.5) * long_u_d
                              + (ec_short_extent * 0.5) * short_u_d,
                            ec - sign_d * (ec_long_extent * 0.5) * long_u_d
                              + (ec_short_extent * 0.5) * short_u_d,
                            ec - sign_d * (ec_long_extent * 0.5) * long_u_d
                              - (ec_short_extent * 0.5) * short_u_d,
                            ec + sign_d * (ec_long_extent * 0.5) * long_u_d
                              - (ec_short_extent * 0.5) * short_u_d,
                        ]
                        probe_pts_px = np.array(
                            [to_px(c[0], c[1]) for c in probe_corners_buf],
                            dtype=np.int32,
                        )
                        anchored = flag_minus_plus[sign_idx]
                        outline_col = (40, 180, 40) if anchored else (40, 40, 220)
                        cv2.polylines(
                            img, [probe_pts_px], True, outline_col, 1,
                            lineType=cv2.LINE_AA,
                        )
            except Exception:
                pass
        elif not p.door_polygon_xy:
            cv2.polylines(img, [bbox_pts], True, (60, 60, 220), 2)
            cx = int(bbox_pts[:, 0].mean())
            cy = int(bbox_pts[:, 1].mean())
            short_id = p.label_id.replace("ann_", "")[:10]
            cv2.putText(
                img, f"SKIP {short_id}", (cx - 30, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (60, 60, 220), 1, cv2.LINE_AA,
            )

    cv2.rectangle(img, (10, 10), (28, 22), (210, 210, 210), -1)
    cv2.putText(img, "wall", (34, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.rectangle(img, (10, 30), (28, 42), (0, 140, 255), -1)
    cv2.putText(img, "door (final)", (34, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 180), 1, cv2.LINE_AA)
    cv2.rectangle(img, (10, 50), (28, 62), (220, 220, 0), -1)
    cv2.putText(img, "pixel-extracted", (34, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img, (10, 70), (28, 82), (245, 245, 245), -1)
    cv2.rectangle(img, (10, 70), (28, 82), (220, 180, 120), 2)
    cv2.putText(img, "label bbox", (34, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.rectangle(img, (10, 90), (28, 102), (245, 245, 245), -1)
    cv2.rectangle(img, (10, 90), (28, 102), (40, 180, 40), 2)
    cv2.putText(img, "pass (in bbox + on wall)", (34, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 140, 40), 1, cv2.LINE_AA)
    cv2.rectangle(img, (10, 110), (28, 122), (245, 245, 245), -1)
    cv2.rectangle(img, (10, 110), (28, 122), (40, 40, 220), 2)
    cv2.putText(img, "fail", (34, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 180), 1, cv2.LINE_AA)
    cv2.putText(img, "width green = DIN 18101 match",
                (10, 142),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 140, 40), 1,
                cv2.LINE_AA)
    cv2.putText(img, "width blue  = non-standard",
                (10, 158),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 100, 200), 1,
                cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)


# ── Tag-7.8: hard gap-overlap criterion ─────────────────────────────────


# Fraction of profile-baseline below which a sample is "gap" (no wall
# outline pixels). Matches ``find_door_gap.gap_threshold_factor``.
GAP_PROFILE_THRESHOLD_FACTOR = 0.30
# Minimum fraction of the door's long-axis extent that must lie in real
# gap pixels for the placement to satisfy the hard "in-gap" criterion.
# Below this, ``pass_validation`` is forced to False — the door is
# silently sitting in solid wall material (e.g. mo6zzgqv_z bbox-fallback
# regression: profile_failed but bbox_extent_on_hinge stretched the door
# across an intact wall section).
GAP_OVERLAP_HARD_MIN = 0.50


def measure_door_gap_overlap(
    door_polygon: Sequence[Tuple[float, float]],
    wall_polygon: WallPolygon,
    image_gray: np.ndarray,
    wall_mask: np.ndarray,
    px_to_m: float,
) -> float:
    """Fraction of the door's long-axis extent that sits in a real wall
    gap (low-coverage profile region) rather than on solid wall.

    Sampling reuses the existing pipeline:
      1. Project the door polygon onto the wall edge tangent at the door
         centre to recover ``(s_lo, s_hi)`` extent and the local axis.
      2. Run ``profile_wall_outline`` along that range — same polarity
         routing (``classify_wall_solidness``) and same depth
         (``local_wall_depth``) as the placement-time gap detector.
      3. Estimate the baseline from samples OUTSIDE the door extent
         (with a 30 cm pad on each side, falling back to the median if
         that region is empty).
      4. Return the share of in-door samples whose profile is below
         ``GAP_PROFILE_THRESHOLD_FACTOR × baseline`` — i.e. samples
         where the wall outline is absent.

    Returns a value in [0, 1]. Returns 1.0 when measurement is impossible
    (degenerate inputs, empty profile) so the absence of evidence does
    not punish a placement.
    """
    pts = np.asarray(door_polygon, dtype=np.float64)
    if pts.shape[0] < 4 or wall_polygon is None or len(wall_polygon.shell_xy) < 3:
        return 1.0
    door_center = pts.mean(axis=0)
    try:
        origin, tang, _ = nearest_edge_axis(wall_polygon, door_center)
    except Exception:
        return 1.0
    minor = np.array([-tang[1], tang[0]], dtype=np.float64)
    sign = _wall_interior_sign(wall_polygon, origin, minor)
    minor_into = sign * minor

    door_t = (pts - origin) @ tang
    s_lo = float(door_t.min())
    s_hi = float(door_t.max())
    door_extent_m = max(0.05, s_hi - s_lo)
    if door_extent_m < 0.05:
        return 1.0

    try:
        g_thick = local_wall_depth(
            wall_polygon, door_center, tang, minor, origin,
            door_extent_m=door_extent_m,
            wall_mask=wall_mask, px_to_m=px_to_m,
        )
    except Exception:
        g_thick = 0.20
    g_thick = max(0.04, float(g_thick))

    try:
        wall_is_solid = classify_wall_solidness(
            image_gray, wall_polygon, wall_mask, px_to_m=px_to_m,
        )
    except Exception:
        wall_is_solid = False

    pad_m = 0.30
    sample_lo = s_lo - pad_m
    sample_hi = s_hi + pad_m
    s_values, profile = profile_wall_outline(
        image_gray, origin, tang, minor_into,
        sample_lo, sample_hi, g_thick, px_to_m,
        wall_is_solid=wall_is_solid,
    )
    if s_values.size < 5 or profile.size != s_values.size:
        return 1.0

    in_door = (s_values >= s_lo) & (s_values <= s_hi)
    out_door = ~in_door
    if out_door.any():
        baseline = float(np.median(profile[out_door]))
    else:
        baseline = float(np.median(profile))
    if baseline <= 0.04:
        # Wall outline too faint to measure — abstain (don't punish).
        return 1.0

    threshold = GAP_PROFILE_THRESHOLD_FACTOR * baseline
    in_door_profile = profile[in_door]
    if in_door_profile.size == 0:
        return 1.0
    gap_share = float(np.mean(in_door_profile < threshold))
    return max(0.0, min(1.0, gap_share))


def _pixel_mask_at_jamb_end(
    door_polygon: Sequence[Tuple[float, float]],
    end_sign: float,
    wall_mask: np.ndarray,
    px_to_m: float,
    *,
    snap_tolerance_m: float,
    probe_along_m: float,
    density_threshold: float = 0.10,
) -> bool:
    """Tag-7.14 — pixel-mask fallback for the jamb anchor probe.

    Rasterises the probe rectangle (long-axis end + tolerance) into
    the ``wall_mask`` raster and returns True iff the fraction of
    wall_mask=255 pixels inside the probe rectangle exceeds
    ``density_threshold``. Catches walls that exist in the raster
    pixel data but were dropped from ``wall_polygons`` by Phase-2
    morphological closing — the pixel anchor still snaps to them
    even when the polygon list does not.
    """
    pts = np.asarray(door_polygon, dtype=np.float64)
    if pts.shape[0] < 4 or px_to_m <= 0:
        return False
    e01 = pts[1] - pts[0]
    e12 = pts[2] - pts[1]
    n01 = float(np.linalg.norm(e01))
    n12 = float(np.linalg.norm(e12))
    if n01 >= n12:
        long_vec = e01
        long_len = n01
        short_len = n12
    else:
        long_vec = e12
        long_len = n12
        short_len = n01
    if long_len < 1e-6:
        return False
    long_u = long_vec / long_len
    short_u = np.array([-long_u[1], long_u[0]], dtype=np.float64)
    center = pts.mean(axis=0)

    half_W = long_len * 0.5
    half_S = max(short_len, 0.10) * 0.5
    probe_long = max(probe_along_m, 0.03)

    end_center = center + end_sign * (half_W + probe_long * 0.5) * long_u
    half_long = probe_long * 0.5 + snap_tolerance_m
    half_short = half_S + snap_tolerance_m
    corners_m = np.array([
        end_center + half_long * long_u + half_short * short_u,
        end_center + half_long * long_u - half_short * short_u,
        end_center - half_long * long_u - half_short * short_u,
        end_center - half_long * long_u + half_short * short_u,
    ], dtype=np.float64)

    H, W = wall_mask.shape
    corners_px = np.empty_like(corners_m, dtype=np.int32)
    corners_px[:, 0] = np.round(corners_m[:, 0] / px_to_m).astype(np.int32)
    corners_px[:, 1] = np.round(corners_m[:, 1] / px_to_m).astype(np.int32)
    if (corners_px[:, 0].max() < 0 or corners_px[:, 1].max() < 0
            or corners_px[:, 0].min() >= W or corners_px[:, 1].min() >= H):
        return False

    rect_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(rect_mask, [corners_px.reshape(-1, 1, 2)], 255)
    n_total = int((rect_mask > 0).sum())
    if n_total == 0:
        return False
    n_wall = int(((rect_mask > 0) & (wall_mask > 0)).sum())
    return (n_wall / n_total) >= density_threshold


def _dark_pixel_at_jamb_end(
    door_polygon: Sequence[Tuple[float, float]],
    end_sign: float,
    image_gray: np.ndarray,
    px_to_m: float,
    *,
    snap_tolerance_m: float,
    probe_along_m: float,
    dark_threshold: int = 130,
    density_threshold: float = 0.10,
) -> bool:
    """Tag-7.16 — last-tier jamb fallback against the RAW PDF render.

    The polygon and ``wall_mask`` checks both miss walls that Phase-1
    annotation skipped (or that the refinement stage shrank below the
    door's probe area). The raw PDF still has them as dark wall
    lines. This helper rasterises the same probe rectangle as
    ``_pixel_mask_at_jamb_end`` into ``image_gray`` and counts the
    fraction of pixels darker than ``dark_threshold``. ≥
    ``density_threshold`` → anchored.

    Trade-off: text, dimensions, and fixture symbols also produce
    dark pixels — false positives are possible. The probe is small
    (9 cm × door_depth + 10 cm) so accidental ink under a label is
    unlikely to dominate. ``density_threshold=0.10`` (10 %) keeps
    false positives down while still catching un-annotated walls.
    """
    pts = np.asarray(door_polygon, dtype=np.float64)
    if pts.shape[0] < 4 or px_to_m <= 0:
        return False
    e01 = pts[1] - pts[0]
    e12 = pts[2] - pts[1]
    n01 = float(np.linalg.norm(e01))
    n12 = float(np.linalg.norm(e12))
    if n01 >= n12:
        long_vec, long_len, short_len = e01, n01, n12
    else:
        long_vec, long_len, short_len = e12, n12, n01
    if long_len < 1e-6:
        return False
    long_u = long_vec / long_len
    short_u = np.array([-long_u[1], long_u[0]], dtype=np.float64)
    center = pts.mean(axis=0)
    half_W = long_len * 0.5
    half_S = max(short_len, 0.10) * 0.5
    probe_long = max(probe_along_m, 0.03)
    end_center = center + end_sign * (half_W + probe_long * 0.5) * long_u
    half_long = probe_long * 0.5 + snap_tolerance_m
    half_short = half_S + snap_tolerance_m
    corners_m = np.array([
        end_center + half_long * long_u + half_short * short_u,
        end_center + half_long * long_u - half_short * short_u,
        end_center - half_long * long_u - half_short * short_u,
        end_center - half_long * long_u + half_short * short_u,
    ], dtype=np.float64)
    H, W = image_gray.shape
    corners_px = np.empty_like(corners_m, dtype=np.int32)
    corners_px[:, 0] = np.round(corners_m[:, 0] / px_to_m).astype(np.int32)
    corners_px[:, 1] = np.round(corners_m[:, 1] / px_to_m).astype(np.int32)
    if (corners_px[:, 0].max() < 0 or corners_px[:, 1].max() < 0
            or corners_px[:, 0].min() >= W or corners_px[:, 1].min() >= H):
        return False
    rect_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(rect_mask, [corners_px.reshape(-1, 1, 2)], 255)
    n_total = int((rect_mask > 0).sum())
    if n_total == 0:
        return False
    n_dark = int(((rect_mask > 0) & (image_gray < dark_threshold)).sum())
    return (n_dark / n_total) >= density_threshold


def _dark_pixel_density_at_jamb_end(
    door_polygon: Sequence[Tuple[float, float]],
    end_sign: float,
    image_gray: np.ndarray,
    px_to_m: float,
    *,
    snap_tolerance_m: float = 0.05,
    probe_along_m: float = 0.04,
    dark_threshold: int = 130,
) -> float:
    """Diagnostic-only — image_gray dark-pixel fraction at a jamb end.
    Returns the raw fraction so the audit can show whether a side has
    visible PDF wall material that the wall_mask missed.
    """
    pts = np.asarray(door_polygon, dtype=np.float64)
    if pts.shape[0] < 4 or px_to_m <= 0:
        return 0.0
    e01 = pts[1] - pts[0]
    e12 = pts[2] - pts[1]
    n01 = float(np.linalg.norm(e01))
    n12 = float(np.linalg.norm(e12))
    if n01 >= n12:
        long_vec, long_len, short_len = e01, n01, n12
    else:
        long_vec, long_len, short_len = e12, n12, n01
    if long_len < 1e-6:
        return 0.0
    long_u = long_vec / long_len
    short_u = np.array([-long_u[1], long_u[0]], dtype=np.float64)
    center = pts.mean(axis=0)
    half_W = long_len * 0.5
    half_S = max(short_len, 0.10) * 0.5
    probe_long = max(probe_along_m, 0.03)
    end_center = center + end_sign * (half_W + probe_long * 0.5) * long_u
    half_long = probe_long * 0.5 + snap_tolerance_m
    half_short = half_S + snap_tolerance_m
    corners_m = np.array([
        end_center + half_long * long_u + half_short * short_u,
        end_center + half_long * long_u - half_short * short_u,
        end_center - half_long * long_u - half_short * short_u,
        end_center - half_long * long_u + half_short * short_u,
    ], dtype=np.float64)
    H, W = image_gray.shape
    corners_px = np.empty_like(corners_m, dtype=np.int32)
    corners_px[:, 0] = np.round(corners_m[:, 0] / px_to_m).astype(np.int32)
    corners_px[:, 1] = np.round(corners_m[:, 1] / px_to_m).astype(np.int32)
    rect_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(rect_mask, [corners_px.reshape(-1, 1, 2)], 255)
    n_total = int((rect_mask > 0).sum())
    if n_total == 0:
        return 0.0
    n_dark = int(((rect_mask > 0) & (image_gray < dark_threshold)).sum())
    return n_dark / n_total


def _pixel_mask_density_at_jamb_end(
    door_polygon: Sequence[Tuple[float, float]],
    end_sign: float,
    wall_mask: np.ndarray,
    px_to_m: float,
    *,
    snap_tolerance_m: float = 0.05,
    probe_along_m: float = 0.04,
) -> float:
    """Diagnostic-only — same probe geometry as
    ``_pixel_mask_at_jamb_end`` but returns the raw wall-pixel fraction
    instead of a bool. Used to print why a side did or did not anchor.
    """
    pts = np.asarray(door_polygon, dtype=np.float64)
    if pts.shape[0] < 4 or px_to_m <= 0:
        return 0.0
    e01 = pts[1] - pts[0]
    e12 = pts[2] - pts[1]
    n01 = float(np.linalg.norm(e01))
    n12 = float(np.linalg.norm(e12))
    if n01 >= n12:
        long_vec, long_len, short_len = e01, n01, n12
    else:
        long_vec, long_len, short_len = e12, n12, n01
    if long_len < 1e-6:
        return 0.0
    long_u = long_vec / long_len
    short_u = np.array([-long_u[1], long_u[0]], dtype=np.float64)
    center = pts.mean(axis=0)
    half_W = long_len * 0.5
    half_S = max(short_len, 0.10) * 0.5
    probe_long = max(probe_along_m, 0.03)
    end_center = center + end_sign * (half_W + probe_long * 0.5) * long_u
    half_long = probe_long * 0.5 + snap_tolerance_m
    half_short = half_S + snap_tolerance_m
    corners_m = np.array([
        end_center + half_long * long_u + half_short * short_u,
        end_center + half_long * long_u - half_short * short_u,
        end_center - half_long * long_u - half_short * short_u,
        end_center - half_long * long_u + half_short * short_u,
    ], dtype=np.float64)
    H, W = wall_mask.shape
    corners_px = np.empty_like(corners_m, dtype=np.int32)
    corners_px[:, 0] = np.round(corners_m[:, 0] / px_to_m).astype(np.int32)
    corners_px[:, 1] = np.round(corners_m[:, 1] / px_to_m).astype(np.int32)
    rect_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(rect_mask, [corners_px.reshape(-1, 1, 2)], 255)
    n_total = int((rect_mask > 0).sum())
    if n_total == 0:
        return 0.0
    n_wall = int(((rect_mask > 0) & (wall_mask > 0)).sum())
    return n_wall / n_total


def jamb_anchor_status(
    door_polygon: Sequence[Tuple[float, float]],
    wall_polygons: Sequence[WallPolygon],
    *,
    snap_tolerance_m: float = 0.10,
    probe_along_m: float = 0.06,
    wall_mask: Optional[np.ndarray] = None,
    px_to_m: Optional[float] = None,
    image_gray: Optional[np.ndarray] = None,
) -> Tuple[bool, bool]:
    """Per-side jamb anchor status. Returns ``(anchored_minus,
    anchored_plus)`` — which long-axis end touches wall material
    within ``snap_tolerance_m``. ``count_jamb_anchors`` is a thin
    wrapper that just sums these.

    The split lets the renderer paint each end's pin individually so
    Willi can see WHICH side floats, not just the count — important
    for end-of-wall placements where one jamb is correct and the
    other is the suspect side.

    Tag-7.14 — when ``wall_mask`` and ``px_to_m`` are provided, each
    side that does NOT anchor on the polygon list is re-checked
    against the raw pixel mask. The pixel fallback catches walls
    present in the raster but missing from Phase-2's polygon
    reconstruction.

    Tag-7.16 — when ``image_gray`` is also provided, sides still
    floating after the wall_mask check fall through to a third tier:
    dark-pixel density in the raw PDF render. Catches walls visible
    in the PDF that Phase-1 annotation missed entirely (so they're
    absent from both wall_polygons AND wall_mask).
    """
    pts = np.asarray(door_polygon, dtype=np.float64)
    if pts.shape[0] < 4:
        return False, False
    e01 = pts[1] - pts[0]
    e12 = pts[2] - pts[1]
    n01 = float(np.linalg.norm(e01))
    n12 = float(np.linalg.norm(e12))
    if n01 >= n12:
        long_vec = e01
        long_len = n01
        short_len = n12
    else:
        long_vec = e12
        long_len = n12
        short_len = n01
    if long_len < 1e-6:
        return False, False
    long_u = long_vec / long_len
    short_u = np.array([-long_u[1], long_u[0]])
    center = pts.mean(axis=0)

    walls_shp = []
    for wp in wall_polygons:
        if not wp.shell_xy or len(wp.shell_xy) < 3:
            continue
        try:
            poly = Polygon(wp.shell_xy).buffer(0)
        except Exception:
            continue
        if poly.is_valid and not poly.is_empty:
            walls_shp.append(poly)
    if not walls_shp:
        return False, False

    half_W = long_len * 0.5
    half_S = max(short_len, 0.10) * 0.5
    probe_long = max(probe_along_m, 0.03)

    flags: List[bool] = []
    for sign in (-1.0, +1.0):
        end_center = center + sign * (half_W + probe_long * 0.5) * long_u
        corners = [
            end_center + sign * (probe_long * 0.5) * long_u + half_S * short_u,
            end_center - sign * (probe_long * 0.5) * long_u + half_S * short_u,
            end_center - sign * (probe_long * 0.5) * long_u - half_S * short_u,
            end_center + sign * (probe_long * 0.5) * long_u - half_S * short_u,
        ]
        try:
            probe_poly = Polygon(
                [(float(c[0]), float(c[1])) for c in corners]
            ).buffer(snap_tolerance_m)
        except Exception:
            flags.append(False)
            continue
        if probe_poly.is_empty or not probe_poly.is_valid:
            flags.append(False)
            continue
        flags.append(any(probe_poly.intersects(wp) for wp in walls_shp))

    # Tag-7.14 — pixel-mask fallback. For every side that the polygon
    # check did not anchor, retry against the raw raster wall_mask.
    if wall_mask is not None and px_to_m is not None and px_to_m > 0:
        for sign_idx, sign in enumerate((-1.0, +1.0)):
            if flags[sign_idx]:
                continue
            try:
                if _pixel_mask_at_jamb_end(
                    door_polygon, sign, wall_mask, px_to_m,
                    snap_tolerance_m=snap_tolerance_m,
                    probe_along_m=probe_along_m,
                ):
                    flags[sign_idx] = True
            except Exception:
                pass

    # Tag-7.16 — image_gray dark-pixel fallback. Catches Phase-1
    # annotation gaps: walls visible in the raw PDF but not labeled.
    if image_gray is not None and px_to_m is not None and px_to_m > 0:
        for sign_idx, sign in enumerate((-1.0, +1.0)):
            if flags[sign_idx]:
                continue
            try:
                if _dark_pixel_at_jamb_end(
                    door_polygon, sign, image_gray, px_to_m,
                    snap_tolerance_m=snap_tolerance_m,
                    probe_along_m=probe_along_m,
                ):
                    flags[sign_idx] = True
            except Exception:
                pass
    return bool(flags[0]), bool(flags[1])


def count_jamb_anchors(
    door_polygon: Sequence[Tuple[float, float]],
    wall_polygons: Sequence[WallPolygon],
    *,
    snap_tolerance_m: float = 0.10,
    probe_along_m: float = 0.06,
    wall_mask: Optional[np.ndarray] = None,
    px_to_m: Optional[float] = None,
    image_gray: Optional[np.ndarray] = None,
) -> int:
    """Total number of long-axis end-flanks anchored on a wall — sum
    of ``jamb_anchor_status``. See that helper for probe geometry.

    Tag-7.14 — forwards ``wall_mask`` and ``px_to_m`` so the underlying
    helper can OR the polygon check with a pixel-mask check.
    Tag-7.16 — forwards ``image_gray`` so the helper can also fall
    through to a raw-PDF dark-pixel check.
    """
    minus, plus = jamb_anchor_status(
        door_polygon, wall_polygons,
        snap_tolerance_m=snap_tolerance_m,
        probe_along_m=probe_along_m,
        wall_mask=wall_mask, px_to_m=px_to_m,
        image_gray=image_gray,
    )
    return int(minus) + int(plus)


def _estimate_avg_wall_thickness_from_mask(
    wall_polygons: Sequence[WallPolygon],
    wall_mask: np.ndarray,
    px_to_m: float,
    *,
    sample_walls: int = 6,
    samples_per_wall: int = 3,
    fallback_m: float = 0.20,
) -> float:
    """Average wall thickness measured directly against ``wall_mask``.

    Picks the ``sample_walls`` longest walls; for each, casts a
    perpendicular ray through ``wall_mask`` at ``samples_per_wall``
    evenly-spaced points along the wall's longest edge; counts
    consecutive ``>0`` pixels in both directions; converts the median
    pixel-thickness to metres via ``px_to_m``. Capped at
    ``ALGO_THICKNESS_MAX_M`` and floored at 5 cm. Returns
    ``fallback_m`` when no measurements succeed.

    Efficient by design (only ``sample_walls × samples_per_wall``
    casts; default 18 rays). Robust to outliers via median.
    """
    if wall_mask is None or px_to_m <= 0 or not wall_polygons:
        return fallback_m
    H, W = wall_mask.shape[:2]

    walls_with_edge: List[Tuple[WallPolygon, np.ndarray, np.ndarray, float]] = []
    for wp in wall_polygons:
        if wp is None or len(wp.shell_xy) < 3:
            continue
        pts = np.asarray(wp.shell_xy, dtype=np.float64)
        n = pts.shape[0]
        best_len = 0.0
        best_a = pts[0]
        best_tang = np.array([1.0, 0.0], dtype=np.float64)
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            ab = b - a
            seg_len = float(np.linalg.norm(ab))
            if seg_len > best_len:
                best_len = seg_len
                best_a = a
                best_tang = ab / seg_len
        if best_len >= _LONG_EDGE_MIN_M:
            walls_with_edge.append((wp, best_a, best_tang, best_len))
    if not walls_with_edge:
        return fallback_m
    walls_with_edge.sort(key=lambda t: -t[3])

    thicknesses_m: List[float] = []
    max_reach_px = max(1, int(round(1.0 / px_to_m)))
    for wp, edge_origin, edge_tang, edge_len in walls_with_edge[:sample_walls]:
        normal = np.array([-edge_tang[1], edge_tang[0]], dtype=np.float64)
        try:
            sign_into = _wall_interior_sign(wp, edge_origin, normal)
        except Exception:
            sign_into = 1.0
        normal_into = sign_into * normal
        for k in range(samples_per_wall):
            t_along = (k + 1.0) / (samples_per_wall + 1) * edge_len
            sample_m = (edge_origin + t_along * edge_tang
                        + 0.03 * normal_into)
            sx = int(round(sample_m[0] / px_to_m))
            sy = int(round(sample_m[1] / px_to_m))
            if not (0 <= sx < W and 0 <= sy < H):
                continue
            if wall_mask[sy, sx] == 0:
                continue
            run_total = 1
            for direction in (+1.0, -1.0):
                for step in range(1, max_reach_px + 1):
                    px = int(round(sx + direction * step * normal[0]))
                    py = int(round(sy + direction * step * normal[1]))
                    if not (0 <= px < W and 0 <= py < H):
                        break
                    if wall_mask[py, px] > 0:
                        run_total += 1
                    else:
                        break
            thicknesses_m.append(run_total * px_to_m)

    if not thicknesses_m:
        return fallback_m
    thicknesses_m.sort()
    median_m = thicknesses_m[len(thicknesses_m) // 2]
    return float(max(0.05, min(median_m, ALGO_THICKNESS_MAX_M)))


def attempt_pixel_recovery(
    placement: "DoorPlacement",
    wall_polygons: Sequence[WallPolygon],
    image_gray: np.ndarray,
    wall_mask: np.ndarray,
    px_to_m: float,
    bbox_m: Tuple[float, float, float, float],
) -> Optional[float]:
    """Door-Guesser (Willi 2026-04-29): place 4 simulated doors, one
    per bbox edge, each with door long-axis ALONG the bbox edge.

    Step 1 — measure standard wall thickness from ``wall_mask``.
    Step 2 — pick door width: pixel-symbol >= ALGO_GAP_MIN_M, else 0.84 m DIN.
    Step 3 — for each of the 4 bbox edges, build a rectangle CENTERED
              on the edge midpoint with the long edge running ALONG the
              edge tangent (parallel-to-bbox-edge invariant by construction;
              never perpendicular).
    Step 4 — at each short side, probe 3 points (midpoint + 2 corners): if
              within ``SNAP_TOL_M`` of any wall polygon, count it as
              snapped; otherwise record the raw shapely distance to the
              nearest wall edge.
    Step 5 — fallback: if any of the 6 probes missed, sweep the door
              along the bbox edge in 5 cm steps over +/- 60 cm and pick
              the t that maximises (snaps, -total_distance).
    Step 6 — pick winner across the 4 edges by
              (jambs, same_polygon_bonus, -total_distance, gap_pct).
              Inner walls (Willi-rule) need both jambs on the SAME
              polygon — extra tiebreaker bonus when this holds.
    Step 7 — accept iff jambs >= 2 OR gap_pct >= GAP_OVERLAP_HARD_MIN.
              The 2/2-jambs override stays from Tag-7.17.

    Replaces the original Tag-7.10/7.13/7.17 brute-force phases (~2100
    candidates) with at most 4 + 4*24 = 100 candidates while honouring
    the parallel-to-bbox-edge invariant by construction.
    """
    if placement.pixel_panel_center_xy_m is None:
        print(f"      recover[{placement.label_id}]: skipped — no pixel_panel_center")
        return None

    # Step 1 — wall thickness measured directly from the raster.
    wall_thickness_m = _estimate_avg_wall_thickness_from_mask(
        wall_polygons, wall_mask, px_to_m,
    )

    # Step 2 — door width: pixel-symbol > 0.84 m DIN fallback.
    pix_w = float(placement.pixel_width_m)
    door_width_m = pix_w if pix_w >= ALGO_GAP_MIN_M else 0.84

    # Pre-build wall shapely shapes for distance queries.
    walls_shp: List[Tuple[int, Polygon]] = []
    for w_idx, wp in enumerate(wall_polygons):
        if wp is None or len(wp.shell_xy) < 3:
            continue
        try:
            shp = Polygon(wp.shell_xy).buffer(0)
            if shp.is_valid and not shp.is_empty:
                walls_shp.append((w_idx, shp))
        except Exception:
            continue
    if not walls_shp:
        print(f"      recover[{placement.label_id}]: skipped — no usable walls")
        return None

    # Step 3 — 4 bbox edges. Each entry: (label, p1, p2).
    bbox_edges = [
        ("top",
         np.array([bbox_m[0], bbox_m[3]], dtype=np.float64),
         np.array([bbox_m[2], bbox_m[3]], dtype=np.float64)),
        ("bottom",
         np.array([bbox_m[0], bbox_m[1]], dtype=np.float64),
         np.array([bbox_m[2], bbox_m[1]], dtype=np.float64)),
        ("left",
         np.array([bbox_m[0], bbox_m[1]], dtype=np.float64),
         np.array([bbox_m[0], bbox_m[3]], dtype=np.float64)),
        ("right",
         np.array([bbox_m[2], bbox_m[1]], dtype=np.float64),
         np.array([bbox_m[2], bbox_m[3]], dtype=np.float64)),
    ]

    SNAP_TOL_M = 0.10
    SCAN_HALF_M = 0.60
    SCAN_STEP_M = 0.05

    def _evaluate_at(
        center_xy: np.ndarray,
        edge_tang: np.ndarray,
        edge_normal: np.ndarray,
    ) -> Dict:
        """Build door rectangle at ``center_xy``; return per-side best
        snap distance (across 3 probes per side: 2 short-edge corners +
        midpoint) and the snapped wall index."""
        polygon = build_rectangle_xy(
            center_xy, edge_tang, edge_normal,
            door_width_m, wall_thickness_m,
        )
        half_thick = 0.5 * wall_thickness_m
        side_best: Dict[float, Tuple[float, int]] = {
            -1.0: (1e9, -1), +1.0: (1e9, -1),
        }
        for sign in (-1.0, +1.0):
            short_mid = center_xy + sign * 0.5 * door_width_m * edge_tang
            for off in (+half_thick, 0.0, -half_thick):
                pt = short_mid + off * edge_normal
                pt_shp = ShapelyPoint(float(pt[0]), float(pt[1]))
                for w_idx, shp in walls_shp:
                    try:
                        d = float(shp.distance(pt_shp))
                    except Exception:
                        continue
                    if d < side_best[sign][0]:
                        side_best[sign] = (d, w_idx)
        d_minus, w_minus = side_best[-1.0]
        d_plus, w_plus = side_best[+1.0]
        return {
            "polygon": polygon,
            "center": center_xy,
            "d_minus": d_minus, "d_plus": d_plus,
            "w_minus": w_minus, "w_plus": w_plus,
            "snap_minus": d_minus <= SNAP_TOL_M,
            "snap_plus": d_plus <= SNAP_TOL_M,
        }

    candidates: List[Dict] = []
    for edge_label, ep1, ep2 in bbox_edges:
        edge_vec = ep2 - ep1
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len < 1e-6:
            continue
        edge_tang = edge_vec / edge_len
        edge_normal = np.array([-edge_tang[1], edge_tang[0]], dtype=np.float64)
        edge_mid = 0.5 * (ep1 + ep2)

        # Step 4 — initial position centred on the bbox edge midpoint.
        res = _evaluate_at(edge_mid, edge_tang, edge_normal)
        chosen_t = 0.0

        # Step 5 — fallback sweep along the edge if any probe missed.
        if not (res["snap_minus"] and res["snap_plus"]):
            best_res = res
            best_t = 0.0
            best_score = (
                int(res["snap_minus"]) + int(res["snap_plus"]),
                -(res["d_minus"] + res["d_plus"]),
            )
            n_steps = int(round(SCAN_HALF_M / SCAN_STEP_M))
            for k in range(-n_steps, n_steps + 1):
                if k == 0:
                    continue
                t_off = k * SCAN_STEP_M
                center_t = edge_mid + t_off * edge_tang
                res_t = _evaluate_at(center_t, edge_tang, edge_normal)
                score_t = (
                    int(res_t["snap_minus"]) + int(res_t["snap_plus"]),
                    -(res_t["d_minus"] + res_t["d_plus"]),
                )
                if score_t > best_score:
                    best_score = score_t
                    best_res = res_t
                    best_t = t_off
            res = best_res
            chosen_t = best_t

        # Step 6a — full jamb status (polygon -> mask -> PDF tiers).
        try:
            jm, jp = jamb_anchor_status(
                res["polygon"], wall_polygons,
                wall_mask=wall_mask, px_to_m=px_to_m,
                image_gray=image_gray,
            )
            jambs = int(jm) + int(jp)
        except Exception:
            jambs = 0

        # Step 6b — gap_overlap on the closest wall behind this candidate.
        try:
            mid_pt = ShapelyPoint(
                float(res["center"][0]), float(res["center"][1]),
            )
            best_widx = -1
            best_d = 1e9
            for w_idx, shp in walls_shp:
                d = float(shp.distance(mid_pt))
                if d < best_d:
                    best_d = d
                    best_widx = w_idx
        except Exception:
            best_widx = -1
        if best_widx >= 0:
            try:
                gap_pct = measure_door_gap_overlap(
                    res["polygon"], wall_polygons[best_widx],
                    image_gray, wall_mask, px_to_m,
                )
            except Exception:
                gap_pct = 0.0
        else:
            gap_pct = 0.0

        # Step 6c — inner-wall bonus (Willi-rule: both jambs on SAME polygon).
        same_poly = int(
            res["snap_minus"] and res["snap_plus"]
            and res["w_minus"] == res["w_plus"]
            and res["w_minus"] >= 0
        )

        candidates.append({
            "edge": edge_label,
            "polygon": res["polygon"],
            "t": chosen_t,
            "jambs": jambs,
            "total_dist": res["d_minus"] + res["d_plus"],
            "gap_pct": gap_pct,
            "wall_idx": best_widx,
            "same_poly": same_poly,
            "snap_minus": res["snap_minus"],
            "snap_plus": res["snap_plus"],
            "d_minus": res["d_minus"],
            "d_plus": res["d_plus"],
            "w_minus": res["w_minus"],
            "w_plus": res["w_plus"],
        })

    if not candidates:
        return None

    # Audit print.
    print(
        f"      recover[{placement.label_id}]: door_guesser "
        f"W={door_width_m:.3f}m thick={wall_thickness_m:.3f}m"
    )
    for c in candidates:
        flag = "*" if c["same_poly"] else " "
        print(
            f"        - {c['edge']:6s} t={c['t']:+.2f}m "
            f"d-/d+={c['d_minus']*100:5.1f}/{c['d_plus']*100:5.1f}cm "
            f"snap={int(c['snap_minus'])}{int(c['snap_plus'])} "
            f"jambs={c['jambs']}/2 gap={c['gap_pct']*100:3.0f}% "
            f"W{c['wall_idx']}{flag}"
        )

    # Step 6 winner: max (jambs, same_poly, -total_dist, gap_pct).
    candidates.sort(
        key=lambda c: (
            c["jambs"], c["same_poly"],
            -c["total_dist"], c["gap_pct"],
        ),
        reverse=True,
    )
    winner = candidates[0]

    # Step 7 — acceptance gate (Tag-7.17 2/2-jambs override preserved).
    if winner["jambs"] < 2 and winner["gap_pct"] < GAP_OVERLAP_HARD_MIN:
        print(
            f"      recover[{placement.label_id}]: best edge={winner['edge']} "
            f"jambs={winner['jambs']}/2 gap={winner['gap_pct']*100:.0f}% — "
            f"neither 2/2 jambs nor gap >= "
            f"{GAP_OVERLAP_HARD_MIN*100:.0f}%, FAIL"
        )
        return None

    placement.door_polygon_xy = winner["polygon"]
    placement.door_width_m = door_width_m
    placement.door_depth_m = wall_thickness_m
    placement.gap_overlap_pct = float(winner["gap_pct"])
    if winner["wall_idx"] >= 0:
        placement.w1_idx = int(winner["wall_idx"])
    placement.width_source = f"door_guesser({winner['edge']})"
    placement.note = (
        (placement.note or "")
        + f"  recovered[edge={winner['edge']} t={winner['t']:+.2f}m "
          f"W={door_width_m:.3f}m d={wall_thickness_m:.3f}m "
          f"jambs={winner['jambs']}/2 gap={winner['gap_pct']*100:.0f}%]"
    )
    print(
        f"      recover[{placement.label_id}]: WINNER edge={winner['edge']} "
        f"t={winner['t']:+.2f}m jambs={winner['jambs']}/2 "
        f"gap={winner['gap_pct']*100:.0f}% W{winner['wall_idx']}"
    )
    return float(winner["gap_pct"])


def enforce_gap_criterion(
    placements: Sequence["DoorPlacement"],
    wall_polygons: Sequence[WallPolygon],
    image_gray: np.ndarray,
    wall_mask: np.ndarray,
    px_to_m: float,
) -> None:
    """Hard rule (Willi 2026-04-29): every placed door must sit in a
    real wall gap (≥ ``GAP_OVERLAP_HARD_MIN`` of its length on absent
    wall outline). Doors that pass the bbox / on-wall heuristics but
    fall into solid wall — typically the bbox_extent_on_hinge fallback
    when ``find_door_gap`` returned no plausible run — first get a
    Tag-7.9 ``attempt_pixel_recovery`` pass that rebuilds the rectangle
    around the panel symbol; only when recovery fails does the
    placement get ``pass_validation`` flipped to False with a
    ``solid_wall_no_gap`` DIN flag.

    Skips placements without a door polygon (no body to check) and
    sliding doors of which only the panel is visible (``door_sliding``)
    — those are scored by their own track-on-wall validation in the
    sliding cascade and the profile criterion would punish a perfectly
    legitimate slider whose pocket sits inside a solid pocket wall.
    """
    log_lines: List[str] = []
    flipped = 0
    recovered = 0
    for p in placements:
        if not p.door_polygon_xy or p.w1_idx < 0 or p.w1_idx >= len(wall_polygons):
            continue
        if (p.label_class or "").lower() == "door_sliding":
            continue
        try:
            pct = measure_door_gap_overlap(
                p.door_polygon_xy,
                wall_polygons[p.w1_idx],
                image_gray, wall_mask, px_to_m,
            )
        except Exception as exc:
            log_lines.append(
                f"    {p.label_id}: gap-overlap measurement raised "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        p.gap_overlap_pct = float(pct)
        try:
            p.jamb_anchor_minus, p.jamb_anchor_plus = jamb_anchor_status(
                p.door_polygon_xy, wall_polygons,
                wall_mask=wall_mask, px_to_m=px_to_m,
                image_gray=image_gray,
            )
            p.jamb_anchors = (
                int(p.jamb_anchor_minus) + int(p.jamb_anchor_plus)
            )
        except Exception:
            p.jamb_anchors = 0
            p.jamb_anchor_minus = False
            p.jamb_anchor_plus = False
        if pct < GAP_OVERLAP_HARD_MIN:
            recovered_pct = attempt_pixel_recovery(
                p, wall_polygons,
                image_gray, wall_mask, px_to_m, p.bbox_m,
            )
            if recovered_pct is not None:
                recovered += 1
                # Recovery moved the polygon — re-measure jamb anchors
                # against the new geometry so downstream consumers (PNG,
                # voting) see the winner's actual flank status, not the
                # stale pre-recovery count.
                try:
                    p.jamb_anchor_minus, p.jamb_anchor_plus = (
                        jamb_anchor_status(
                            p.door_polygon_xy, wall_polygons,
                            wall_mask=wall_mask, px_to_m=px_to_m,
                            image_gray=image_gray,
                        )
                    )
                    p.jamb_anchors = (
                        int(p.jamb_anchor_minus) + int(p.jamb_anchor_plus)
                    )
                    # Tag-7.14 — print pixel-mask density at each end so
                    # Willi can tell if the still-floating side is in
                    # raster open space (genuine end-of-wall) or sits on
                    # raster wall material that the fallback rejected
                    # (threshold too tight).
                    try:
                        d_minus = _pixel_mask_density_at_jamb_end(
                            p.door_polygon_xy, -1.0, wall_mask, px_to_m,
                        )
                        d_plus = _pixel_mask_density_at_jamb_end(
                            p.door_polygon_xy, +1.0, wall_mask, px_to_m,
                        )
                        # Tag-7.16 — also compute raw-PDF dark density
                        # so audit shows both wall_mask-based and
                        # PDF-based jamb evidence side by side.
                        g_minus = _dark_pixel_density_at_jamb_end(
                            p.door_polygon_xy, -1.0, image_gray, px_to_m,
                        )
                        g_plus = _dark_pixel_density_at_jamb_end(
                            p.door_polygon_xy, +1.0, image_gray, px_to_m,
                        )
                        log_lines.append(
                            f"      pixel-jamb density "
                            f"[{p.label_id}]: "
                            f"minus=mask{d_minus*100:.0f}%/"
                            f"pdf{g_minus*100:.0f}% "
                            f"({'anchor' if p.jamb_anchor_minus else 'float'}) "
                            f"plus=mask{d_plus*100:.0f}%/"
                            f"pdf{g_plus*100:.0f}% "
                            f"({'anchor' if p.jamb_anchor_plus else 'float'})"
                        )
                    except Exception:
                        pass
                except Exception:
                    pass
                log_lines.append(
                    f"    {p.label_id}: gap_overlap={pct*100:.0f}% "
                    f"-> recovered via pixel anchor "
                    f"(new gap={recovered_pct*100:.0f}%, "
                    f"jambs={p.jamb_anchors}/2)"
                )
                continue
            was_passing = p.pass_validation
            p.pass_validation = False
            note_extra = (
                f"; in-solid-wall (gap_overlap={pct*100:.0f}%, "
                f"min={GAP_OVERLAP_HARD_MIN*100:.0f}%)"
            )
            p.validation_note = (p.validation_note or "") + note_extra
            if "solid_wall_no_gap" not in p.din_flags:
                p.din_flags.append("solid_wall_no_gap")
            if was_passing:
                flipped += 1
            log_lines.append(
                f"    {p.label_id}: gap_overlap={pct*100:.0f}% "
                f"< {GAP_OVERLAP_HARD_MIN*100:.0f}% — pass_validation -> False"
            )
        else:
            log_lines.append(
                f"    {p.label_id}: gap_overlap={pct*100:.0f}% "
                f"jambs={p.jamb_anchors}/2 (ok)"
            )

    if log_lines:
        print(
            f"\n  Tag-7.8/7.9 gap-criterion (hard min "
            f"{GAP_OVERLAP_HARD_MIN*100:.0f}%, "
            f"{recovered} recovered via pixel anchor, {flipped} -> FAIL):"
        )
        for line in log_lines:
            print(line)



# ── Tag-7.7: hinge-sign voting ──────────────────────────────────────────


def _collect_wall_corners_m(
    wall_polygons: Sequence[WallPolygon],
) -> List[Tuple[float, float]]:
    """All distinct shell vertices across every wall polygon.

    These act as candidate hinge anchors for ``compute_hinge_long_sign_
    from_corners``: T-junctions, L-corners, and wall ends are by
    definition shell vertices of one or more polygons. Buffered
    union-vertices would lose individual corners — we want every
    annotation endpoint, even if two walls share a near-coincident
    point.
    """
    corners: List[Tuple[float, float]] = []
    for wp in wall_polygons:
        if not wp.shell_xy:
            continue
        for x, y in wp.shell_xy:
            corners.append((float(x), float(y)))
    return corners


def assign_hinge_voting(
    placements: Sequence["DoorPlacement"],
    wall_polygons: Sequence[WallPolygon],
    image_gray: np.ndarray,
    px_to_m: float,
) -> None:
    """Tag-7.7 — populate ``hinge_long_sign``/``hinge_confidence``/
    ``hinge_votes`` for every single-leaf swing door in ``placements``.

    Up to four independent signals contribute, each one returning ``+1``,
    ``-1`` or ``None`` (abstain):

    * **arc**      — ``placement.arc_hinge_xy_m`` (arc-detected hinge
      centre). Direct geometric measurement; strongest signal.
    * **panel**    — ``placement.panel_hinge_xy_m`` (panel pixel-line
      endpoint that sits on a wall surface, from
      ``select_start_wall_by_panel_endpoint``). Pixel-symbol-derived,
      independent of arc detection.
    * **corners**  — wall-corner asymmetry. The long-axis end of the
      door rectangle that is closer to a wall corner is the hinge end.
    * **centroid** — dark-pixel centroid inside the bbox is biased
      toward the latch end of the symbol; hinge end is the opposite.

    ``vote_hinge_long_sign`` aggregates the votes into a single sign +
    confidence + DIN flag list. Double-leaf and sliding doors are
    skipped because ``hinge_long_sign`` is irrelevant there.

    Prints a short per-door audit line for every door, including the
    chosen sign, confidence, and per-signal vote breakdown.
    """
    from planproof_geometry.ai.v3.pixel_bim.openings.door_models import (
        compute_hinge_long_sign,
        compute_hinge_long_sign_from_corners,
        compute_hinge_long_sign_from_pixel_centroid,
        vote_hinge_long_sign,
    )

    wall_corners = _collect_wall_corners_m(wall_polygons)
    log_lines: List[str] = []

    for p in placements:
        cls = (p.label_class or "").lower()
        if cls in ("door_double", "door_sliding"):
            continue
        if not p.door_polygon_xy or len(p.door_polygon_xy) < 4:
            continue

        votes: Dict[str, int] = {}
        try:
            if p.arc_hinge_xy_m is not None:
                votes["arc"] = compute_hinge_long_sign(
                    p.door_polygon_xy, p.arc_hinge_xy_m
                )
        except Exception:
            pass
        try:
            if p.panel_hinge_xy_m is not None:
                votes["panel"] = compute_hinge_long_sign(
                    p.door_polygon_xy, p.panel_hinge_xy_m
                )
        except Exception:
            pass
        try:
            sign_corners = compute_hinge_long_sign_from_corners(
                p.door_polygon_xy, wall_corners
            )
            if sign_corners is not None:
                votes["corners"] = sign_corners
        except Exception:
            pass
        try:
            sign_centroid = compute_hinge_long_sign_from_pixel_centroid(
                p.door_polygon_xy, p.bbox_m, image_gray, px_to_m
            )
            if sign_centroid is not None:
                votes["centroid"] = sign_centroid
        except Exception:
            pass

        sign, conf, flags = vote_hinge_long_sign(votes)
        p.hinge_long_sign = int(sign)
        p.hinge_confidence = float(conf)
        p.hinge_votes = dict(votes)
        if flags:
            p.din_flags.extend(flags)

        votes_str = " ".join(f"{k}={v:+d}" for k, v in votes.items()) or "-"
        log_lines.append(
            f"    {p.label_id}: hinge_long={sign:+d} "
            f"conf={conf:.2f} votes=[{votes_str}]"
        )

    if log_lines:
        print("\n  Tag-7.7 hinge-sign voting:")
        for line in log_lines:
            print(line)


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 72)
    print("  V3 Doors — Willi-algorithm placement (gap-fit + pixel + fallback)")
    print("=" * 72)

    OUT.mkdir(parents=True, exist_ok=True)

    image_gray = _render_pdf_grayscale(PDF_PATH, RENDER_MATRIX)
    H, W = image_gray.shape
    raw_ann = json.loads(ANN_PATH.read_text(encoding="utf-8"))
    img_size = raw_ann.get("image_size") or {}
    ann_w = int(img_size.get("width") or 0)
    ann_h = int(img_size.get("height") or 0)
    sx = W / ann_w if ann_w else 1.0
    sy = H / ann_h if ann_h else 1.0
    annotations = raw_ann.get("annotations") or []

    px_to_m = _calibrate_px_to_m_from_doors(annotations, (sx, sy))
    if px_to_m is None:
        raise SystemExit("Scale calibration failed.")
    print(f"  px_to_m={px_to_m:.5f}  ({1.0/px_to_m:.1f} px/m)")

    from planproof_geometry.ai.v3.pixel_bim.refine.wall_dispatcher import (
        refine_walls,
    )
    wall_polys_px: List[List[Tuple[float, float]]] = []
    for ann in annotations:
        cls = ann.get("class") or ""
        if cls not in WALL_CLASSES:
            continue
        poly = _poly_from_annotation(ann, (sx, sy))
        if len(poly) < 3:
            continue
        wall_polys_px.append(poly)
    # Sprint Sliding-Door-Rewrite Tag-3: sliding doors are IN-wall
    # apertures, not separate elements. We do NOT subtract them from the
    # wall mask anymore — the wall extrusion produces continuous wall
    # material through the sliding-door region; the head_height void
    # carves the opening later. Swing doors continue to skip subtraction
    # (their ink is jamb/arc/sill, not wall material), so the
    # ``door_polygons_px=None`` path is now the universal call.
    wall_mask, dispatch_info = refine_walls(
        image_gray, wall_polys_px, px_to_m=px_to_m,
        door_polygons_px=None,
    )
    corrected = dispatch_info.get("wall_polygons_corrected_px")
    if corrected is not None:
        wall_polys_px = corrected
    mode = dispatch_info.get("mode", "unknown")
    if mode == "filled":
        print(
            f"  refine_walls mode=filled "
            f"paired_ratio={dispatch_info.get('paired_ratio', 0):.2f} "
            f"model={dispatch_info.get('model_loaded')} "
            f"tiled={dispatch_info.get('used_tiled')} "
            f"tolerance_px={dispatch_info.get('tolerance_px')} "
            f"fallbacks={dispatch_info.get('n_fallback')}/{dispatch_info.get('n_polygons')} "
            f"in_field={dispatch_info.get('ml_pixels_in_field')}px "
            f"dropped_outside={dispatch_info.get('ml_pixels_outside_field')}px"
        )
    else:
        print(
            f"  refine_walls mode={mode} "
            f"paired_ratio={dispatch_info.get('paired_ratio', 0):.2f} "
            f"polys={dispatch_info.get('n_polygons')} "
            f"fallbacks={dispatch_info.get('fallbacks', 0)} "
            f"phantom_filtered={dispatch_info.get('phantom_filtered', 0)}"
        )

    # Tag-4.2 Lead 3 (Liu et al. 2017 / CubiCasa pattern): detect openings
    # BEFORE morphological closing. We subtract each door annotation's
    # bbox region from the wall_mask used for Phase-2 extrusion — this
    # prevents Phase-2's closing kernel from merging adjacent
    # perpendicular walls at L-corners into combined polygons. Each
    # door creates a clean notch that breaks the would-be L-junction
    # back into two separate rectangles, so nearest_edge_axis on the
    # resulting polygon picks the correct hinge wall unambiguously.
    door_cut_mask = np.zeros((H, W), dtype=np.uint8)
    for ann in annotations:
        cls = ann.get("class") or ""
        if cls not in DOOR_CLASSES:
            continue
        # Sprint Sliding-Door-Rewrite Tag-3: sliding doors stay AS WALL
        # material in the Phase-2 mask. Without this skip, the door_cut
        # would punch a hole in the wall right where the sliding panel
        # sits — defeating the "wall with aperture" model.
        if cls == "door_sliding":
            continue
        poly_px = _poly_from_annotation(ann, (sx, sy))
        if len(poly_px) < 3:
            continue
        pts = np.array(poly_px, dtype=np.int32)
        cv2.fillPoly(door_cut_mask, [pts], 255)
    # Dilate the cut by ~30 mm so it goes fully through any narrow wall,
    # even when the bbox is slightly inside the wall's room-side edge.
    cut_dilate_px = max(2, int(round(0.030 / px_to_m)))
    if cut_dilate_px > 0:
        kernel_cut = np.ones(
            (cut_dilate_px * 2 + 1, cut_dilate_px * 2 + 1), np.uint8
        )
        door_cut_mask = cv2.dilate(door_cut_mask, kernel_cut, iterations=1)
    wall_mask_for_phase2 = cv2.bitwise_and(
        wall_mask, cv2.bitwise_not(door_cut_mask)
    )
    n_walls_before = (wall_mask > 0).sum()
    n_walls_after = (wall_mask_for_phase2 > 0).sum()
    print(
        f"  door-cut: removed {n_walls_before - n_walls_after} px "
        f"({100*(n_walls_before-n_walls_after)/max(1,n_walls_before):.1f}%) "
        f"from wall_mask before Phase-2"
    )

    user_wall_union_mask = np.zeros_like(wall_mask)
    for poly in wall_polys_px:
        if len(poly) < 3:
            continue
        pts = np.array(
            [[int(round(x)), int(round(y))] for x, y in poly], dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.fillPoly(user_wall_union_mask, [pts], 255)
    extrude_result = extrude_wall_mask(
        wall_mask_for_phase2, px_to_m=px_to_m, wall_height_m=WALL_HEIGHT_M,
        simplify_m=0.005, min_polygon_area_m2=0.005,
        image_gray=image_gray,
        user_polygon_union_mask=user_wall_union_mask,
    )
    wall_polygons = extrude_result.polygons
    print(f"  walls: {len(wall_polygons)} phase-2 polygons")

    # Tag-6.8: pre-collect every door bbox so place_door can mask
    # neighbour-door pixels out of its RANSAC crop. Without this, dense
    # clusters (Bath/WC) cause RANSAC to fit a neighbour's arc, snapping
    # the door to the wrong wall.
    all_door_bboxes_m: Dict[str, Tuple[float, float, float, float]] = {}
    for ann in annotations:
        cls = ann.get("class") or ""
        if cls not in DOOR_CLASSES:
            continue
        label_px = _poly_from_annotation(ann, (sx, sy))
        if len(label_px) < 3:
            continue
        label_m = _poly_to_m(label_px, px_to_m)
        all_door_bboxes_m[ann["id"]] = _bbox_from_poly_m(label_m)

    placements: List[DoorPlacement] = []
    for ann in annotations:
        cls = ann.get("class") or ""
        if cls not in DOOR_CLASSES:
            continue
        label_px = _poly_from_annotation(ann, (sx, sy))
        if len(label_px) < 3:
            continue
        label_m = _poly_to_m(label_px, px_to_m)
        bbox_m = _bbox_from_poly_m(label_m)
        other_bboxes_m = [
            b for k, b in all_door_bboxes_m.items() if k != ann["id"]
        ]

        # Sprint Sliding-Door-Rewrite Tag-3: sliding doors take a
        # dedicated 4-dir scan path, NOT the place_door cascade. The
        # gap+jamb+RANSAC machinery is sized for swing doors with a
        # separate hinge/leaf and does not produce useful signals for
        # in-wall sliding panels.
        if cls == "door_sliding":
            bbox_cx = (bbox_m[0] + bbox_m[2]) * 0.5
            bbox_cy = (bbox_m[1] + bbox_m[3]) * 0.5
            wall_ctx = scan_wall_4dir(
                (bbox_cx, bbox_cy), wall_polygons,
                bbox_m=bbox_m,
                image_gray=image_gray,
                px_to_m=px_to_m,
            )
            mc = wall_ctx.mount_classification
            if mc is not None:
                from planproof_geometry.ai.v3.pixel_bim.openings.mount_classifier import (
                    _bbox_wall_overlap_fraction,
                )
                frac = _bbox_wall_overlap_fraction(bbox_m, wall_polygons)
                frac_s = "n/a" if frac is None else f"{frac*100:.1f}%"
                print(
                    f"  sliding {ann['id']:<14} {mc.mount_class:<11} "
                    f"conf={mc.confidence:.2f} overlap={frac_s} votes={mc.votes}"
                )
            bbox_w = bbox_m[2] - bbox_m[0]
            bbox_h = bbox_m[3] - bbox_m[1]
            panel_width_m = max(bbox_w, bbox_h)

            # Sprint Sliding-Door-In-Wall-Classifier Tag-3: route by
            # mount_classification first; fall back to confidence-based
            # gate when the classifier abstains.
            in_wall_decided = (
                mc is not None
                and mc.mount_class == "in_wall"
                and mc.confidence >= 0.5
                and wall_ctx.found
                and wall_ctx.confidence >= 0.5
            )
            beside_wall_decided = (
                mc is not None
                and mc.mount_class == "beside_wall"
                and mc.confidence >= 0.5
                and wall_ctx.found
            )

            if in_wall_decided:
                polygon_xy, short_offset, pocket_sign = (
                    build_sliding_door_polygon(
                        wall_ctx, bbox_m, panel_width_m,
                    )
                )
                panel_slide_m = panel_width_m * 0.7
                max_extent = max(
                    wall_ctx.extent_long_pos_m,
                    wall_ctx.extent_long_neg_m,
                )
                mount_type = "pocket" if max_extent >= panel_slide_m else "surface"
                p = DoorPlacement(
                    label_id=ann["id"], label_class=cls, bbox_m=bbox_m,
                    w1_idx=wall_ctx.wall_polygon_idx, w2_idx=None,
                    w1_distance_m=0.0, w2_distance_m=None,
                    door_width_m=panel_width_m,
                    door_depth_m=wall_ctx.wall_depth_m,
                    door_polygon_xy=polygon_xy,
                    pass_validation=True,
                    inside_bbox=True,
                    on_wall=True,
                    inside_bbox_pct=1.0,
                    wall_overlap_pct=1.0,
                    width_source="sliding_4dir_scan",
                    note=(
                        f"sliding 4dir-scan conf={wall_ctx.confidence:.2f} "
                        f"mount_class={mc.mount_class}({mc.confidence:.2f}) "
                        f"extents=[+{wall_ctx.extent_long_pos_m:.2f}m / "
                        f"-{wall_ctx.extent_long_neg_m:.2f}m] "
                        f"mount={mount_type} pocket={pocket_sign:+d}"
                    ),
                    validation_note="sliding via 4dir-scan in_wall",
                    wall_depth_mesh_m=wall_ctx.wall_depth_m,
                    short_offset_m=short_offset,
                    sliding_pocket_sign=pocket_sign,
                    sliding_mount_type=mount_type,
                    sliding_extent_long_pos_m=wall_ctx.extent_long_pos_m,
                    sliding_extent_long_neg_m=wall_ctx.extent_long_neg_m,
                    sliding_mount_position="in_wall",
                )
                placements.append(p)
                continue

            if beside_wall_decided:
                # Barn-door beside the wall: polygon at bbox centre,
                # axis along the nearest wall, panel-thickness depth.
                # NO wall-void carve — apply_door_voids skips placements
                # whose ``sliding_mount_position == "beside_wall"``.
                polygon_xy, short_offset, pocket_sign = (
                    build_beside_wall_sliding_polygon(
                        wall_ctx, bbox_m, panel_width_m,
                    )
                )
                panel_thickness_m = 0.04
                p = DoorPlacement(
                    label_id=ann["id"], label_class=cls, bbox_m=bbox_m,
                    w1_idx=wall_ctx.wall_polygon_idx, w2_idx=None,
                    w1_distance_m=0.0, w2_distance_m=None,
                    door_width_m=panel_width_m,
                    door_depth_m=panel_thickness_m,
                    door_polygon_xy=polygon_xy,
                    pass_validation=True,
                    inside_bbox=True,
                    on_wall=False,
                    inside_bbox_pct=1.0,
                    wall_overlap_pct=0.0,
                    width_source="sliding_4dir_scan_beside",
                    note=(
                        f"sliding 4dir-scan beside-wall "
                        f"mount_class={mc.mount_class}({mc.confidence:.2f}) "
                        f"votes={mc.votes} pocket={pocket_sign:+d}"
                    ),
                    validation_note="sliding via 4dir-scan beside_wall",
                    wall_depth_mesh_m=panel_thickness_m,
                    short_offset_m=short_offset,
                    sliding_pocket_sign=pocket_sign,
                    sliding_mount_type="beside",
                    sliding_extent_long_pos_m=wall_ctx.extent_long_pos_m,
                    sliding_extent_long_neg_m=wall_ctx.extent_long_neg_m,
                    sliding_mount_position="beside_wall",
                )
                placements.append(p)
                continue

            # Fallback: classifier abstained or confidence too low.
            # Use the legacy cascade so we don't drop the placement.
            mc_dbg = (
                f"mount_class={mc.mount_class}({mc.confidence:.2f})"
                if mc is not None else "no mc"
            )
            print(
                f"  sliding-fallback: {ann['id']} 4dir-scan "
                f"conf={wall_ctx.confidence:.2f} {mc_dbg} -> place_door cascade"
            )

        p = place_door(
            bbox_m, label_m, wall_polygons,
            image_gray, wall_mask, px_to_m,
            label_class=cls,
            other_door_bboxes_m=other_bboxes_m,
        )
        p.label_id = ann["id"]
        p.label_class = cls
        placements.append(p)

    # Tag-6.4 conflict resolution: if any door placements overlap or
    # come closer than 10 cm, shift the lower-confidence one along its
    # wall tangent until separation is restored. Today's Matterport
    # plan has zero conflicts because Tag-6.0/6.3 already separated
    # the Bath cluster onto distinct walls, so this is dormant; it
    # earns its keep on plans with denser door clusters where symbol
    # detection lands two anchors on the same wall segment.
    n_shifted = resolve_door_conflicts(placements, wall_polygons)
    if n_shifted > 0:
        print(f"  Tag-6.4 conflict resolver shifted {n_shifted} placements")

    # Tag-6.5 (Willi 2026-04-28): inter-door minimum distance hard
    # rule. Two placed doors must keep at least MIN_INTER_DOOR_DISTANCE_M
    # of wall material between their rectangles — that is the minimum
    # frame-to-frame separation a real opening allows. When the
    # distance falls below the threshold (i.e. Tag-6.4's shift could
    # not clear it within max_shift_m), both placements get
    # pass_validation=False with a "conflict" note so the cluster
    # stays visually marked.
    MIN_INTER_DOOR_DISTANCE_M = 0.10
    for i, pa in enumerate(placements):
        if not pa.door_polygon_xy:
            continue
        try:
            poly_a = Polygon(pa.door_polygon_xy).buffer(0)
        except Exception:
            continue
        if poly_a.is_empty or poly_a.area <= 0:
            continue
        for j in range(i + 1, len(placements)):
            pb = placements[j]
            if not pb.door_polygon_xy:
                continue
            try:
                poly_b = Polygon(pb.door_polygon_xy).buffer(0)
            except Exception:
                continue
            if poly_b.is_empty or poly_b.area <= 0:
                continue
            d = float(poly_a.distance(poly_b))
            if d < MIN_INTER_DOOR_DISTANCE_M:
                pa.pass_validation = False
                pb.pass_validation = False
                conflict_a = f"; conflict with {pb.label_id} (d={d*1000:.0f}mm)"
                conflict_b = f"; conflict with {pa.label_id} (d={d*1000:.0f}mm)"
                pa.validation_note = (pa.validation_note or "") + conflict_a
                pb.validation_note = (pb.validation_note or "") + conflict_b

    # Tag-7.8: hard "in-gap" criterion. Every door must sit on actual
    # wall-outline absence, not in solid wall material. Doors that
    # passed bbox + on-wall heuristics but fail the profile-based gap
    # overlap check (e.g. mo6zzgqv_z bbox_extent_on_hinge fallback that
    # stretched the rectangle across an intact section of W11) get
    # the Tag-7.9 ``attempt_pixel_recovery`` pass before
    # ``pass_validation`` is flipped — recovery may move the door to a
    # different wall (mo6zzgqv_z migrated from W11 to W6 with 92% gap
    # overlap). MUST run BEFORE ``assign_hinge_voting`` because the
    # recovered polygon has a different long axis basis and the hinge
    # vote needs to project signals against the final geometry.
    enforce_gap_criterion(
        placements, wall_polygons, image_gray, wall_mask, px_to_m,
    )

    # Tag-7.7: hinge-end voting from up to four independent signals
    # (arc, panel, corners, centroid). Single-leaf swing doors that fall
    # back to the bbox path used to silently default to ``hinge_long_sign
    # = -1`` and rendered the leaf at the wrong corner. The voting picks
    # the actual hinge end and records a confidence so the verifier can
    # flag low-confidence placements for review.
    assign_hinge_voting(placements, wall_polygons, image_gray, px_to_m)

    placed = sum(1 for p in placements if p.door_polygon_xy)
    passed = sum(1 for p in placements if p.pass_validation)
    arc_path = sum(
        1 for p in placements if p.width_source == "arc_centered_hinge"
    )
    gap_real = sum(
        1 for p in placements if p.width_source == "wall_outline_gap"
    )
    gap_bbox = sum(
        1 for p in placements if p.width_source == "bbox_extent_on_hinge"
    )
    print(
        f"  doors: {len(placements)} total, placed={placed}, "
        f"validated={passed}/{placed}, "
        f"arc={arc_path}/{placed}, gap={gap_real}/{placed}, "
        f"bbox={gap_bbox}/{placed}"
    )
    for p in placements:
        if not p.door_polygon_xy:
            flag = "SKIP"
        elif p.pass_validation:
            flag = "PASS"
        else:
            flag = "FAIL"
        flank = (f"W1={p.w1_idx} d={p.w1_distance_m:.2f}m"
                 if p.w1_idx >= 0 else "no W1")
        if p.w2_idx is not None:
            flank += f" W2={p.w2_idx} d={p.w2_distance_m:.2f}m"
        else:
            flank += " (single)"
        print(f"   {flag}  {p.label_id:<24} {p.label_class:<14} "
              f"w={p.door_width_m:.3f}m d={p.door_depth_m:.3f}m  {flank}")
        print(f"        {p.note}")
        print(f"        validation: {p.validation_note}")


    out_png = OUT / "door_placement.png"
    render_placement_png(
        wall_polygons, placements, out_png,
    )
    print(f"\n  PNG: {out_png}")


    # Sprint Door-Wall-Bridge Tag-2: precompute the actual wall thickness
    # (measured perpendicular to each door against the post-refine BIM
    # polygons) and the wall-centerline short_offset. These attrs land
    # on the DoorPlacement and feed into the void carve below + the
    # frame/bridge geometry in _emit_bim_viewer. Mirrors the window
    # pipeline's ``assign_window_short_offsets``.
    n_off = assign_door_wall_axis_attrs(placements, wall_polygons)
    if n_off > 0:
        offsets_log = ", ".join(
            f"{p.label_id}={p.short_offset_m * 1000:+.0f}mm"
            for p in placements
            if p.door_polygon_xy and abs(p.short_offset_m) > 0.001
        )
        print(f"  Tag-2 wall-axis offsets: {offsets_log}")

    # Sprint Door-Wall-Bridge Tag-3: build the door-void rectangle from
    # the SSOT helper (build_door_rectangle_from_polygon) so the carve
    # uses the SAME rectangle as the frame and bridge. The rectangle is:
    #   * centered on (polygon_centre + short_offset_m * short_u)
    #     — wall midplane
    #   * long extent = polygon long edge
    #   * short extent = wall_depth_mesh_m (measured) when available,
    #     else the placement's polygon-derived door_depth_m
    # Without this Tag-3 wiring, the carve sat on the raw polygon while
    # the frame was shifted by short_offset_m → visible "door next to
    # wall" misalignment for any door with non-zero offset.
    wall_mesh = extrude_result.mesh
    door_voids = []
    for p in placements:
        if not p.door_polygon_xy:
            continue
        # Sprint Sliding-Door-In-Wall-Classifier Tag-3: beside-wall
        # sliders are barn-doors hung beside the wall — no aperture
        # exists in the wall, so no void to carve. The panel mesh is
        # added separately by build_door_meshes via mount=beside.
        if p.sliding_mount_position == "beside_wall":
            continue
        if p.wall_depth_mesh_m > 0.04:
            wd_for_void = p.wall_depth_mesh_m
            so_for_void = p.short_offset_m
        else:
            wd_for_void = p.door_depth_m or 0.12
            so_for_void = 0.0
        rect_xy = build_door_rectangle_from_polygon(
            p.door_polygon_xy,
            wall_depth_m=wd_for_void,
            short_offset_m=so_for_void,
        )
        door_voids.append(DoorVoid(
            polygon_xy=rect_xy,
            label_id=p.label_id,
        ))
    print(f"\n  3D mesh: input wall_mesh has "
          f"{len(wall_mesh.vertices)} vertices, "
          f"{len(wall_mesh.faces)} faces; "
          f"applying {len(door_voids)} door voids")
    wall_mesh_with_voids = apply_door_voids(
        wall_mesh, door_voids,
        wall_height_m=WALL_HEIGHT_M,
    )
    out_glb = OUT / "door_placement.glb"
    try:
        wall_mesh_with_voids.export(str(out_glb))
        print(f"  GLB: {out_glb} "
              f"({len(wall_mesh_with_voids.vertices)} verts, "
              f"{len(wall_mesh_with_voids.faces)} faces)")
    except Exception as exc:
        print(f"  GLB export failed: {exc}")

    # ── BIM viewer: two GLBs (swing + sliding) + browser HTML ──────────
    # The Phase-2 wall mesh is built from ``wall_mask_for_phase2``,
    # which has each door already cut out so the contour finder can
    # separate L-junctions cleanly. Side-effect: every door breaks the
    # surrounding wall into its own polygon, producing 16+ disconnected
    # extruded boxes with visible seams at junctions.
    #
    # For the BIM viewer we want CONNECTED walls. Re-extrude from the
    # *pre-door-cut* ``wall_mask`` so wall regions that meet at
    # corners/T-junctions extrude as one mesh; then cut the corrected
    # door voids inside ``_emit_bim_viewer``.
    bim_extrude_result = extrude_wall_mask(
        wall_mask,
        px_to_m=px_to_m,
        wall_height_m=WALL_HEIGHT_M,
        simplify_m=0.005,
        min_polygon_area_m2=0.005,
        image_gray=image_gray,
        user_polygon_union_mask=user_wall_union_mask,
    )
    bim_wall_mesh_clean = bim_extrude_result.mesh
    print(
        f"  BIM viewer wall mesh (pre-door-cut): "
        f"{len(bim_wall_mesh_clean.vertices)} verts, "
        f"{len(bim_wall_mesh_clean.faces)} faces, "
        f"{len(bim_extrude_result.polygons)} polygons"
    )

    # Build user-drawn wall polygons in metres (raw annotations, no refine)
    # for the wall-sufficiency sliding-door classifier + pocket-bridge cap.
    user_wall_polys_m_for_bim: List[List[Tuple[float, float]]] = []
    for ann in annotations:
        if (ann.get("class") or "") not in WALL_CLASSES:
            continue
        p_px = _poly_from_annotation(ann, (sx, sy))
        if len(p_px) >= 3:
            user_wall_polys_m_for_bim.append(
                [(float(x) * px_to_m, float(y) * px_to_m) for x, y in p_px]
            )

    _emit_bim_viewer(
        wall_mesh_with_voids, placements, wall_polygons,
        wall_mesh_clean=bim_wall_mesh_clean,
        user_wall_polys_m=user_wall_polys_m_for_bim,
    )


def _emit_bim_viewer(wall_mesh_with_voids, placements, wall_polygons,
                     *, wall_mesh_clean=None,
                     user_wall_polys_m: "Optional[List[List[Tuple[float, float]]]]" = None):
    """Build two GLBs (swing-only model + sliding-only model) plus a
    self-contained HTML viewer (base64-embedded) and open it in the
    default browser.

    * ``bim_swing.glb``    — wall mesh + voids + swing-classed doors
      rendered as frame (jambs + lintel) and a leaf swung 75° open.
    * ``bim_sliding.glb``  — wall mesh + voids + sliding-classed doors
      rendered as a track rail and a panel slid 70 % toward the pocket.
    """
    import base64

    import trimesh

    from planproof_geometry.ai.v3.pixel_bim.openings.door_models import (
        compute_mount_type,
        compute_pocket_sign,
        compute_swing_dir_sign,
    )
    # ``classify_sliding_mount`` is the legacy bbox-extension cascade
    # used as a fallback when the 4-dir scan path did not produce a
    # ``sliding_pocket_sign`` (i.e. very old placements). Matterport
    # plans never hit it. The module ships separately and may be
    # missing on lean checkouts; tolerate the absence so the rest of
    # the pipeline runs.
    try:
        from planproof_geometry.ai.v3.pixel_bim.openings.pocket_surface_classifier import (  # noqa: E501
            classify_sliding_mount,
        )
    except Exception:  # pragma: no cover - module is optional
        classify_sliding_mount = None

    # Tag-9.5: caller passes user-drawn wall polygons in metres via
    # ``user_wall_polys_m`` parameter. These are the raw annotations
    # (before any refine/carving) and represent the annotator's intent
    # for where wall material exists; the wall-sufficiency classifier
    # and the bridge wing extender both consult them.
    if user_wall_polys_m is None:
        user_wall_polys_m = []

    # Sprint Door-Wall-Bridge Tag-2: wall-axis attrs are now precomputed
    # by ``assign_door_wall_axis_attrs`` BEFORE the door-void cut, so the
    # placement carries (wall_depth_mesh_m, short_offset_m) into both the
    # carve and the frame. Here we just read them.
    placed_doors: List[PlacedDoor] = []
    mount_log: List[str] = []
    depth_log: List[str] = []
    for p in placements:
        if not p.door_polygon_xy:
            continue
        d_actual = p.wall_depth_mesh_m
        short_offset = p.short_offset_m
        d_placement = p.door_depth_m or 0.12
        # Fall back to the placement value if the ray missed all walls.
        if d_actual > 0.04:
            wall_d = d_actual
        else:
            wall_d = d_placement
            short_offset = 0.0
        if d_actual > 0.0 and (
            abs(d_actual - d_placement) > 0.02
            or abs(short_offset) > 0.01
        ):
            depth_log.append(
                f"    {p.label_id}: placement={d_placement*1000:.0f}mm "
                f"-> mesh={d_actual*1000:.0f}mm "
                f"(used {wall_d*1000:.0f}mm) "
                f"short_offset={short_offset*1000:+.0f}mm"
            )
        try:
            swing_sign = compute_swing_dir_sign(p.door_polygon_xy, p.bbox_m)
        except Exception:
            swing_sign = +1
        # Tag-9.5: for sliding doors use the wall-sufficiency classifier
        # which jointly determines mount_type + pocket_sign by measuring
        # how much continuous user-drawn wall material exists on each
        # long-axis side of the door. Falls back to legacy bbox-extension
        # heuristic for swing/double doors (mount_type irrelevant there).
        is_sliding_label = (p.label_class or "").lower() == "door_sliding"
        if is_sliding_label:
            # Sprint Sliding-Door-Rewrite Tag-3: prefer 4-dir-scan
            # results when the placement came from the new path. This
            # avoids re-running classify_sliding_mount with raw user
            # walls that no longer reflect the post-refine BIM
            # geometry (which now includes the sliding-door region).
            if p.sliding_pocket_sign is not None:
                pocket_sign = int(p.sliding_pocket_sign)
                mount_type = p.sliding_mount_type or "pocket"
                _slide_cls_debug = (
                    f"4dir_scan extents=[+{p.sliding_extent_long_pos_m:.3f}m"
                    f" / -{p.sliding_extent_long_neg_m:.3f}m]"
                )
            elif classify_sliding_mount is not None:
                try:
                    placed_w = float(p.door_width_m or 0.0) or None
                    cls_res = classify_sliding_mount(
                        p.door_polygon_xy,
                        p.bbox_m,
                        wall_d,
                        wall_polygons_xy=user_wall_polys_m,
                        panel_width_m=placed_w,
                    )
                    mount_type = cls_res.mount_type
                    pocket_sign = cls_res.pocket_sign
                    _slide_cls_debug = (
                        f"wall_len[+1]={cls_res.wall_len_pos:.3f}m "
                        f"wall_len[-1]={cls_res.wall_len_neg:.3f}m "
                        f"panel_slide={cls_res.panel_slide_m:.3f}m "
                        f"conf={cls_res.confidence:.2f}"
                    )
                except Exception as _exc:
                    mount_type = "pocket"
                    try:
                        pocket_sign = compute_pocket_sign(
                            p.door_polygon_xy, p.bbox_m
                        )
                    except Exception:
                        pocket_sign = +1
                    _slide_cls_debug = f"classifier_failed: {_exc}"
            else:
                # Fallback when pocket_surface_classifier is missing.
                mount_type = "pocket"
                try:
                    pocket_sign = compute_pocket_sign(
                        p.door_polygon_xy, p.bbox_m
                    )
                except Exception:
                    pocket_sign = +1
                _slide_cls_debug = "classify_sliding_mount unavailable"
        else:
            try:
                pocket_sign = compute_pocket_sign(p.door_polygon_xy, p.bbox_m)
            except Exception:
                pocket_sign = +1
            try:
                mount_type = compute_mount_type(
                    p.door_polygon_xy, p.bbox_m, wall_d
                )
            except Exception:
                mount_type = "pocket"
            _slide_cls_debug = ""
        # Tag-7.7 — hinge end derived from the multi-signal vote in
        # ``assign_hinge_voting``. ``p.hinge_long_sign`` is populated for
        # every single-leaf swing door (and is None for double / sliding,
        # where the value is irrelevant). Default falls back to -1 for
        # backwards compatibility with the legacy single-arc path.
        hinge_long_sign = (
            int(p.hinge_long_sign) if p.hinge_long_sign is not None else -1
        )
        hinge_confidence = float(p.hinge_confidence)
        placed_doors.append(PlacedDoor(
            polygon_xy=p.door_polygon_xy,
            label_class=p.label_class,
            label_id=p.label_id,
            head_height_m=2.10,
            wall_depth_m=wall_d,
            swing_dir_sign=swing_sign,
            pocket_sign=pocket_sign,
            mount_type=mount_type,
            hinge_long_sign=hinge_long_sign,
            hinge_confidence=hinge_confidence,
            hinge_votes=dict(p.hinge_votes),
            gap_overlap_pct=float(p.gap_overlap_pct),
            pass_validation=bool(p.pass_validation),
            pixel_polygon_xy=list(p.pixel_polygon_xy),
            width_source=p.width_source,
            jamb_anchors=int(p.jamb_anchors),
            jamb_anchor_minus=bool(p.jamb_anchor_minus),
            jamb_anchor_plus=bool(p.jamb_anchor_plus),
            short_offset_m=short_offset,
            sliding_mount_position=p.sliding_mount_position,
        ))
        if (p.label_class or "").lower() == "door_sliding":
            mount_log.append(
                f"    {p.label_id} {p.label_class}: "
                f"mount={mount_type} pocket_sign={pocket_sign:+d} "
                f"wall_depth={wall_d:.3f}m"
                + (f"  [{_slide_cls_debug}]" if _slide_cls_debug else "")
            )
        else:
            votes_str = (
                " ".join(f"{k}={v:+d}" for k, v in p.hinge_votes.items())
                or "no_signal"
            )
            mount_log.append(
                f"    {p.label_id} {p.label_class}: "
                f"swing={swing_sign:+d} hinge_long={hinge_long_sign:+d} "
                f"conf={hinge_confidence:.2f} votes=[{votes_str}]"
            )

    if depth_log:
        print("\n  Wall-thickness override (placement -> actual mesh):")
        for line in depth_log:
            print(line)

    if mount_log:
        print("\n  Sliding-door mount classification:")
        for line in mount_log:
            print(line)

    swing_doors = filter_for_class(
        placed_doors, ("door", "door_single", "door_double")
    )
    sliding_doors = filter_for_class(placed_doors, ("door_sliding",))

    print(
        f"\n  BIM viewer: {len(swing_doors)} swing-class doors, "
        f"{len(sliding_doors)} sliding-class doors"
    )

    # Per-door meshes (one mesh per door) so each door becomes its own glTF
    # node in the exported scene — selectable/editable individually in the BIM
    # editor instead of being merged into the wall buffer.
    # Sliding doors render with their classified mount (in_wall / beside).
    # No 3-way override — the mount_classifier already decided per door.
    swing_door_groups = build_door_meshes_grouped(swing_doors, mode="swing")
    sliding_door_groups = build_door_meshes_grouped(sliding_doors, mode="sliding")

    # Door void rectangles (canonical wall-midplane rectangles). Beside-
    # wall sliders carry no aperture so they are skipped.
    door_voids_xy: List[Tuple[List[Tuple[float, float]], float]] = []
    for placed in placed_doors:
        if placed.sliding_mount_position == "beside_wall":
            continue
        try:
            rect_xy = build_door_rectangle(placed)
        except Exception as exc:
            print(
                f"  build_door_rectangle failed for "
                f"{placed.label_id}: {exc}"
            )
            continue
        door_voids_xy.append((rect_xy, placed.head_height_m))

    bim_wall_mesh = wall_mesh_with_voids

    # Tint the wall mesh light grey for a clearer read against the
    # warm-wood door geometry.
    wall_visual = bim_wall_mesh.copy()
    try:
        wall_visual.visual.face_colors = [220, 220, 220, 255]
    except Exception:
        pass


    # Frame jambs + sill + head are dimensioned with
    # JAMB_WALL_OVERLAP_M / SILL_HEAD_WALL_OVERLAP_M so they
    # geometrically OVERLAP the wall material (10 mm into the wall on
    # the long axis, 10 mm into parapet/lintel). The viewer reads
    # this as a flush, gap-free junction even when the meshes stay
    # topologically separate. A boolean.union of frames into walls is
    # too brittle here (manifold3d rejects the wall mesh when the
    # void cuts produce non-watertight components), so we accept
    # geometric overlap as the connection.

    # BIM scene with SEPARATE named nodes — each wall run and each door is its
    # own glTF node, so they are individually selectable/editable in the BIM
    # editor (instead of one merged buffer). Windows + lintel bridges are added
    # later by merge_doors_and_windows_bim.py, also as separate nodes.
    # Node names carry the class prefix (wall/…, door_single/…, …) so the
    # editor's class-colouring/grouping keeps working. Sliding doors render with
    # their classifier-picked mount (in_wall / beside) — no 3-way override.
    bim_scene = trimesh.Scene()
    node_total = 0

    # Walls: split into connected components → one node per wall run.
    wall_pieces = []
    try:
        wall_pieces = list(wall_visual.split(only_watertight=False))
    except Exception as exc:
        print(f"  wall split failed ({exc}); keeping walls as one node")
    if not wall_pieces:
        wall_pieces = [wall_visual]
    for i, wp in enumerate(wall_pieces):
        if hasattr(wp, "vertices") and len(wp.vertices):
            bim_scene.add_geometry(wp, node_name=f"wall/{i}")
            node_total += 1

    # Doors: one node per door, named "<class>/<index>".
    door_groups = list(swing_door_groups) + list(sliding_door_groups)
    for idx, (placed, mesh) in enumerate(door_groups):
        if not (hasattr(mesh, "vertices") and len(mesh.vertices)):
            continue
        cls = (placed.label_class or "door").lower()
        bim_scene.add_geometry(mesh, node_name=f"{cls}/{idx}")
        node_total += 1

    bim_path = OUT / "bim.glb"
    try:
        bim_scene.export(str(bim_path))
        print(
            f"  GLB: {bim_path} ({node_total} nodes: "
            f"{len(wall_pieces)} wall + {len(door_groups)} doors) — "
            f"{len(swing_doors)} swing + {len(sliding_doors)} sliding, "
            f"each as a separate selectable node"
        )
    except Exception as exc:
        print(f"  BIM GLB export failed: {exc}")
        return

    bim_b64 = base64.b64encode(bim_path.read_bytes()).decode("ascii")

    swing_count = len(swing_doors)
    sliding_count = len(sliding_doors)

    html = _BIM_HTML_TEMPLATE.format(
        bim_b64=bim_b64,
        swing_count=swing_count,
        sliding_count=sliding_count,
    )
    html_path = OUT / "bim_viewer.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"  HTML: {html_path}")

    # ── Self-review pass ────────────────────────────────────────────
    # Seit dem Library-Move (ADR-016) liegt verify_bim_view als Geschwister-
    # Modul im perfect_bim-Package — kein sys.path-Hack mehr noetig.
    from .verify_bim_view import (
        render_elevation_png,
        render_topdown_png,
        render_walls_and_bridges_png,
        verify_door,
    )

    try:
        render_walls_and_bridges_png(
            placed_doors, wall_polygons,
            OUT / "bim_walls_and_bridges.png",
            wall_height_m=WALL_HEIGHT_M,
            user_wall_polys_m=user_wall_polys_m,
        )
        print(f"  Walls+Bridges PNG: {OUT / 'bim_walls_and_bridges.png'}")
    except Exception as exc:
        print(f"  Walls+Bridges render failed: {exc}")


    label_bboxes_m = [p.bbox_m for p in placements if p.door_polygon_xy]
    print("\n  Self-review: invariant checks")
    total_issues = 0
    for placed in placed_doors:
        rep = verify_door(placed)
        if rep.issues:
            total_issues += len(rep.issues)
            print(f"    [FAIL] {rep.label_id} ({rep.label_class}):")
            for iss in rep.issues:
                print(f"           - {iss}")
        else:
            cls_lc = (placed.label_class or "").lower()
            if cls_lc == "door_sliding":
                tag = f"mount={placed.mount_type}"
            else:
                tag = (
                    f"swing={rep.swing_dir_actual:+d} "
                    f"(target {rep.swing_dir_sign:+d}) "
                    f"hinge={placed.hinge_long_sign:+d}"
                    f"@conf={rep.hinge_confidence:.2f}"
                )
            print(f"    [PASS] {rep.label_id} ({rep.label_class})  {tag}  "
                  f"max_short_extent={rep.max_short_extent_m*1000:.1f}mm "
                  f"wall={rep.wall_depth_m*1000:.0f}mm")

    print(f"\n  Self-review summary: {total_issues} invariant issue(s)")

    try:
        render_topdown_png(
            placed_doors, wall_polygons,
            OUT / "bim_verify_topdown.png",
            label_bboxes_m=label_bboxes_m,
        )
        print(f"  Top-down PNG: {OUT / 'bim_verify_topdown.png'}")
    except Exception as exc:
        print(f"  Top-down render failed: {exc}")

    try:
        render_elevation_png(
            placed_doors, OUT / "bim_verify_elevations.png",
        )
        print(f"  Elevation PNG: {OUT / 'bim_verify_elevations.png'}")
    except Exception as exc:
        print(f"  Elevation render failed: {exc}")

    # BIM viewer auto-open suppressed (Willi 2026-05-02): inspection
    # happens via the PNG diagnostics (door_placement.png,
    # windows_on_pdf.png, bim_walls_and_bridges.png, bim_verify_*).
    print(f"  BIM viewer HTML written but NOT opened: {html_path}")


_BIM_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>Planproof BIM Viewer</title>
<script type=\"module\"
        src=\"https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js\"></script>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; background: #1b1d22;
                font-family: 'Segoe UI', system-ui, sans-serif; color: #ddd; }}
  header {{ padding: 0.7rem 1rem; background: #14161a;
            border-bottom: 1px solid #2a2d33; }}
  header h1 {{ margin: 0; font-size: 1.05rem; font-weight: 600; }}
  header span.sub {{ color: #888; font-size: 0.85rem; margin-left: 0.5rem; }}
  main {{ display: flex; padding: 0.5rem;
         height: calc(100vh - 56px); box-sizing: border-box; }}
  .pane {{ flex: 1 1 auto; display: flex; flex-direction: column;
           background: #22252b; border: 1px solid #2a2d33;
           border-radius: 6px; overflow: hidden; }}
  .pane h2 {{ margin: 0; padding: 0.5rem 0.8rem; font-size: 0.95rem;
              background: #1b1e23; color: #f0f0f0;
              border-bottom: 1px solid #2a2d33; }}
  .pane h2 .count {{ float: right; color: #6cb1ff; font-weight: 500;
                     font-size: 0.85rem; }}
  model-viewer {{ flex: 1 1 auto; width: 100%; background: #d8dbe0; }}
</style>
</head>
<body>
<header>
  <h1>Planproof BIM Viewer
      <span class=\"sub\">{swing_count} swing · {sliding_count} sliding · drag to orbit · scroll to zoom · right-drag to pan</span>
  </h1>
</header>
<main>
  <section class=\"pane\">
    <h2>BIM-Modell
        <span class=\"count\">Walls + Lintels + Doors (1 Modell)</span></h2>
    <model-viewer
        src=\"data:model/gltf-binary;base64,{bim_b64}\"
        camera-controls touch-action=\"pan-y\"
        shadow-intensity=\"1\" exposure=\"1.05\"
        environment-image=\"neutral\"
        camera-orbit=\"35deg 65deg auto\"
        min-camera-orbit=\"auto auto 0.5m\"
        max-camera-orbit=\"auto auto auto\"
        interaction-prompt=\"none\">
    </model-viewer>
  </section>
</main>
</body>
</html>
"""


if __name__ == "__main__":
    main()
