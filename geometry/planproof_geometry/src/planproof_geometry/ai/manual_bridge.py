"""Bridge: manual annotations.json -> pipeline semantics.

This module is the entry point of the Human-First Plan2BIM architecture.
The user labels the plan in the Tauri AnnotationEditor, producing an
``annotations.json`` file. This bridge converts that file into the same
``semantics_dict`` shape the rest of the pipeline already consumes.

Key invariants:

* annotations with ``confidence == 1.0`` are **locked** — downstream ML
  steps must not modify their geometry (enforced by ``filter_against_locked_boxes``
  in v2/detection_merge/merge.py).
* every produced item carries ``source="manual"`` and ``meta.weight`` so
  the gravity-well scoring step can distinguish them from ML detections.
* coordinates are converted from image-pixels to metres exactly once, here.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# annotation class -> internal element class
# Keep wall sub-types as meta.wall_type so the BIM exporter can pick the
# right thickness (exterior vs interior).
_WALL_CLASSES = {
    "wall": ("wall", "interior"),
    "wall_exterior": ("wall", "exterior"),
    "wall_interior": ("wall", "interior"),
}

_DOOR_CLASSES = {
    "door", "door_single", "door_double", "door_sliding",
}

_WINDOW_CLASSES = {
    "window", "window_large",
}

_STAIRS_CLASSES = {
    "stairs", "stairs_spiral", "elevator",
}

_COLUMN_CLASSES = {
    "column", "beam",
}

_ROOM_CLASSES = {
    "room_label",
}

# Classes that are fully ignored by the BIM chain
_IGNORED_CLASSES = {
    "furniture", "sanitary", "kitchen",
    "dimension_line", "dimension_text",
    "unknown",
}

# Default ML-confidence cap for non-locked manual annotations
LOCKED_CONFIDENCE_THRESHOLD = 0.999


def _bbox_corners(bbox: List[float]) -> Tuple[float, float, float, float]:
    """Normalize bbox to (x_min, y_min, x_max, y_max)."""
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _bbox_to_wall_segment(bbox: List[float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Convert a wall bbox into a centerline segment along its long axis.

    A wall bbox is essentially a thick rectangle. The centerline runs through
    the midpoints of the two short edges.
    """
    x1, y1, x2, y2 = _bbox_corners(bbox)
    w = x2 - x1
    h = y2 - y1
    if w >= h:
        # horizontal wall
        cy = (y1 + y2) * 0.5
        return (x1, cy), (x2, cy)
    # vertical wall
    cx = (x1 + x2) * 0.5
    return (cx, y1), (cx, y2)


def _dominant_edge_angle_deg(points: List[List[float]]) -> float:
    """Return the edge-length-weighted dominant angle (0-180°, undirected)
    of a polygon's perimeter. Used to decide whether to rectify a
    user-drawn polygon to the horizontal/vertical axis.
    """
    import math
    if len(points) < 2:
        return 0.0
    buckets: Dict[int, float] = {}
    n = len(points)
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0  # undirected
        bucket = int(angle // 2) * 2
        buckets[bucket] = buckets.get(bucket, 0.0) + L
    if not buckets:
        return 0.0
    return float(max(buckets.items(), key=lambda x: x[1])[0])


def _axis_align_polygon(
    points: List[List[float]],
    angle_tol_deg: float = 15.0,
) -> List[List[float]]:
    """Rotate a user-drawn polygon so its dominant edge hits exactly 0°
    or 90° — corrects the human tremor that leaves walls 1-10° off-axis.
    Diagonal walls beyond the tolerance (e.g. real 30° roof-line walls)
    are left untouched.

    Rotation is around the polygon centroid, preserving the wall's
    spatial position. Must be called BEFORE _extract_walls_from_polygon
    so the produced centerlines inherit the corrected orientation.
    """
    import math
    dom = _dominant_edge_angle_deg(points)
    # Closest axis: 0° (horizontal) or 90° (vertical)
    dist_to_0 = min(dom, 180.0 - dom)
    dist_to_90 = abs(dom - 90.0)
    if dist_to_0 <= angle_tol_deg and dist_to_0 <= dist_to_90:
        # Snap horizontal. Target = whichever side of 90° we're on.
        target = 0.0 if dom <= 90.0 else 180.0
    elif dist_to_90 <= angle_tol_deg:
        target = 90.0
    else:
        return [list(p) for p in points]  # real diagonal — leave alone
    delta_deg = target - dom
    if abs(delta_deg) < 0.05:
        return [list(p) for p in points]  # already aligned
    cx = sum(float(p[0]) for p in points) / len(points)
    cy = sum(float(p[1]) for p in points) / len(points)
    rad = math.radians(delta_deg)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    return [
        [
            cx + (float(p[0]) - cx) * cos_r - (float(p[1]) - cy) * sin_r,
            cy + (float(p[0]) - cx) * sin_r + (float(p[1]) - cy) * cos_r,
        ]
        for p in points
    ]


def _snap_segment_to_axis(
    a: Tuple[float, float],
    b: Tuple[float, float],
    angle_tol_deg: float = 15.0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Snap a single centerline segment to horizontal/vertical within
    tolerance. Rotates around the midpoint so the wall's centre stays put.
    Used as a final safety net AFTER edge-pair decomposition, since the
    midline of two near-parallel edges can still carry a fractional tilt.
    """
    import math
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return a, b
    angle = math.degrees(math.atan2(dy, dx))  # -180..180
    abs_angle = abs(angle)
    # Distance to the 4 axis directions (0, 90, -90, 180, -180)
    if abs_angle <= angle_tol_deg or abs_angle >= 180 - angle_tol_deg:
        target = 0.0 if angle >= 0 else 0.0  # horizontal preserving rough direction
        if angle > 90 or angle < -90:
            target = 180.0 if angle >= 0 else -180.0
    elif abs(angle - 90) <= angle_tol_deg or abs(angle + 90) <= angle_tol_deg:
        target = 90.0 if angle >= 0 else -90.0
    else:
        return a, b
    mx = (a[0] + b[0]) / 2
    my = (a[1] + b[1]) / 2
    target_rad = math.radians(target)
    half = L / 2
    return (
        (mx - half * math.cos(target_rad), my - half * math.sin(target_rad)),
        (mx + half * math.cos(target_rad), my + half * math.sin(target_rad)),
    )


def _extract_walls_from_polygon(
    points: List[List[float]],
    min_edge_ratio: float = 0.20,
    parallel_angle_tol_deg: float = 18.0,
    axis_align_tol_deg: float = 0.0,
) -> List[Dict[str, Any]]:
    """Decompose a wall polygon into one or more centerline segments.

    Architectural wall polygons come in three flavours:

    1. **Straight rectangular wall** (4-8 pts, aspect > 3): bbox-centerline works.
    2. **L-shaped corner** (6-8 pts, aspect ≈ 1): two wall arms meeting at a
       corner — bbox-centerline produces a *diagonal* that misses both arms.
    3. **Rotated rectangular wall**: bbox-centerline picks the wrong axis
       because bbox is axis-aligned.

    Strategy: pair each polygon edge with its most-parallel *opposite* edge and
    treat that pair as one wall (centerline = midline between the pair, wall
    thickness = perpendicular distance). Unpaired short edges are the wall
    end-caps; unpaired long edges become single-sided wall lines.

    Returns a list of dicts: ``{"a": (x,y), "b": (x,y), "thickness": float}``.
    At minimum one segment is returned, even for degenerate polygons (falls
    back to bbox-centerline).
    """
    import math

    if len(points) < 3:
        return []

    # Manual-Lock: axis-alignment is opt-in. Default 0.0 preserves the
    # exact angle the user drew — fixes 2026-04-21 review group E
    # ("wall rotated, opening displaced") where 1-15° tilted user walls
    # were forced to orthogonal, displacing every opening attached to
    # them. Caller can re-enable with axis_align_tol_deg > 0.
    if axis_align_tol_deg > 0:
        points = _axis_align_polygon(points, angle_tol_deg=axis_align_tol_deg)

    # Compute edges with lengths
    n = len(points)
    edges: List[Dict[str, Any]] = []
    for i in range(n):
        p1 = (float(points[i][0]), float(points[i][1]))
        p2 = (float(points[(i + 1) % n][0]), float(points[(i + 1) % n][1]))
        L = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if L <= 0:
            continue
        edges.append({"p1": p1, "p2": p2, "len": L, "paired": False})

    if not edges:
        return []

    longest = max(e["len"] for e in edges)
    min_len = longest * min_edge_ratio
    long_edges = [e for e in edges if e["len"] >= min_len]
    if len(long_edges) < 2:
        # Degenerate: fall back to bbox-centerline.
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        a, b = _bbox_to_wall_segment([min(xs), min(ys), max(xs), max(ys)])
        thickness = min(max(xs) - min(xs), max(ys) - min(ys))
        return [{"a": a, "b": b, "thickness": float(thickness)}]

    # Greedy pair long edges by "most parallel + closest midpoint"
    def angle_of(e):
        return math.atan2(e["p2"][1] - e["p1"][1], e["p2"][0] - e["p1"][0])

    def mid(e):
        return ((e["p1"][0] + e["p2"][0]) * 0.5,
                (e["p1"][1] + e["p2"][1]) * 0.5)

    def perp_distance(point, edge):
        """Perpendicular distance from point to infinite line through edge."""
        x0, y0 = point
        x1, y1 = edge["p1"]; x2, y2 = edge["p2"]
        denom = edge["len"]
        if denom <= 0:
            return math.hypot(x0 - x1, y0 - y1)
        return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / denom

    tol_rad = math.radians(parallel_angle_tol_deg)
    walls: List[Dict[str, Any]] = []
    sorted_edges = sorted(long_edges, key=lambda e: -e["len"])

    for i, e1 in enumerate(sorted_edges):
        if e1["paired"]:
            continue
        a1 = angle_of(e1)
        m1 = mid(e1)
        best_j = None; best_score = -1e9
        for j, e2 in enumerate(sorted_edges):
            if i == j or e2["paired"]:
                continue
            a2 = angle_of(e2)
            diff = abs(a1 - a2) % math.pi
            if diff > math.pi / 2:
                diff = math.pi - diff
            if diff > tol_rad:
                continue
            # Score: prefer longer partners that are closer & more similar length
            m2 = mid(e2)
            d_mid = math.hypot(m1[0] - m2[0], m1[1] - m2[1])
            len_ratio = min(e1["len"], e2["len"]) / max(e1["len"], e2["len"])
            score = len_ratio - d_mid / max(longest, 1.0)
            if score > best_score:
                best_score = score; best_j = j
        if best_j is None:
            continue
        e2 = sorted_edges[best_j]
        e1["paired"] = True; e2["paired"] = True
        # Pair endpoints by closest match so the centerline runs the RIGHT way
        d11 = math.hypot(e1["p1"][0] - e2["p1"][0], e1["p1"][1] - e2["p1"][1])
        d12 = math.hypot(e1["p1"][0] - e2["p2"][0], e1["p1"][1] - e2["p2"][1])
        if d11 <= d12:
            c1 = ((e1["p1"][0] + e2["p1"][0]) * 0.5, (e1["p1"][1] + e2["p1"][1]) * 0.5)
            c2 = ((e1["p2"][0] + e2["p2"][0]) * 0.5, (e1["p2"][1] + e2["p2"][1]) * 0.5)
        else:
            c1 = ((e1["p1"][0] + e2["p2"][0]) * 0.5, (e1["p1"][1] + e2["p2"][1]) * 0.5)
            c2 = ((e1["p2"][0] + e2["p1"][0]) * 0.5, (e1["p2"][1] + e2["p1"][1]) * 0.5)
        thickness = perp_distance(e1["p1"], e2)
        walls.append({"a": c1, "b": c2, "thickness": float(thickness)})

    if not walls:
        # Couldn't pair anything — fallback
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        a, b = _bbox_to_wall_segment([min(xs), min(ys), max(xs), max(ys)])
        thickness = min(max(xs) - min(xs), max(ys) - min(ys))
        walls.append({"a": a, "b": b, "thickness": float(thickness)})

    # Manual-Lock: centerline axis-snap is opt-in (same flag as
    # polygon rotation). Default 0.0 → return centerlines exactly as
    # derived from the user's polygon edge pairs.
    if axis_align_tol_deg <= 0:
        return walls
    aligned: List[Dict[str, Any]] = []
    for piece in walls:
        a_snap, b_snap = _snap_segment_to_axis(
            piece["a"], piece["b"], angle_tol_deg=axis_align_tol_deg
        )
        aligned.append({"a": a_snap, "b": b_snap, "thickness": piece["thickness"]})
    return aligned


def _polygon_to_wall_segment(points: List[List[float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Convert a polygon into a centerline by taking its bbox + long axis.

    Legacy single-segment version — kept for callers that want a simple line.
    Use _extract_walls_from_polygon() for the new edge-pair decomposition
    which yields multiple segments for L-shapes / complex walls.
    """
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return _bbox_to_wall_segment([min(xs), min(ys), max(xs), max(ys)])


def _bbox_thickness(bbox: List[float]) -> float:
    """Return the short dimension of a bbox in the same units as bbox."""
    x1, y1, x2, y2 = _bbox_corners(bbox)
    return min(x2 - x1, y2 - y1)


def load_annotations_file(path: str | Path) -> Optional[Dict[str, Any]]:
    """Read and validate an annotations.json. Returns None on parse failure."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("manual_bridge: failed to read %s: %s", path, exc)
        return None

    if not isinstance(data, dict) or "annotations" not in data or "image_size" not in data:
        logger.warning("manual_bridge: %s missing 'annotations' or 'image_size'", path)
        return None
    return data


def _annotations_for_page(
    annotations_file: Dict[str, Any],
    page_number: int,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Pick the right annotation list + image size for a given page.

    Multi-page PDFs store per-page data under ``pages[1..n]``; if absent,
    the top-level ``annotations`` + ``image_size`` apply to page 1.
    """
    pages = annotations_file.get("pages")
    if isinstance(pages, dict):
        # JSON keys are strings even when they represent integers
        page_data = pages.get(str(page_number)) or pages.get(page_number)
        if page_data:
            sz = page_data.get("image_size", {})
            return list(page_data.get("annotations", [])), (
                int(sz.get("width", 0)),
                int(sz.get("height", 0)),
            )

    sz = annotations_file.get("image_size", {})
    return list(annotations_file.get("annotations", [])), (
        int(sz.get("width", 0)),
        int(sz.get("height", 0)),
    )


def manual_annotations_to_semantics(
    annotations_file: Dict[str, Any],
    *,
    px_to_m: float,
    page_number: int = 1,
    default_wall_thickness_m: float = 0.12,
) -> Dict[str, Any]:
    """Convert an annotations file into a partial semantics_dict.

    Args:
        annotations_file: Parsed annotations.json.
        px_to_m: Pixel -> metre conversion factor (from scale inference).
        page_number: 1-based page to extract (multi-page PDFs).
        default_wall_thickness_m: Fallback when bbox short side is implausible.

    Returns:
        Dict with ``segments``, ``detections``, ``rooms`` keys, all marked
        ``source="manual"``. Locked items have ``meta.locked=True``.
    """
    raw_annotations, (img_w, img_h) = _annotations_for_page(annotations_file, page_number)
    if not raw_annotations:
        return {
            "segments": [], "detections": [], "rooms": [],
            "image_size_px": {"width": img_w, "height": img_h},
            "locked_count": 0, "manual_count": 0,
        }

    segments: List[Dict[str, Any]] = []
    detections: List[Dict[str, Any]] = []
    rooms: List[Dict[str, Any]] = []
    locked_count = 0

    for idx, ann in enumerate(raw_annotations):
        cls = str(ann.get("class", ""))
        if cls in _IGNORED_CLASSES:
            continue

        ann_id = str(ann.get("id") or f"m{idx}")
        confidence = float(ann.get("confidence", 1.0))
        is_locked = confidence >= LOCKED_CONFIDENCE_THRESHOLD
        if is_locked:
            locked_count += 1

        # Collect bbox in pixels regardless of annotation type
        if ann.get("type") == "polygon":
            points = ann.get("points") or []
            if len(points) < 3:
                continue
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            bbox_px = [min(xs), min(ys), max(xs), max(ys)]
        else:
            bbox_raw = ann.get("bbox")
            if not bbox_raw or len(bbox_raw) < 4:
                continue
            bbox_px = list(bbox_raw)

        meta_common = {
            "manual_annotation_id": ann_id,
            "manual_class_raw": cls,
            "weight": confidence,
            "locked": is_locked,
            "bbox_px": bbox_px,
        }

        # ---- Walls ----
        if cls in _WALL_CLASSES:
            _, wall_type = _WALL_CLASSES[cls]
            # For polygons: use edge-pair decomposition → multiple segments
            # for L-shapes / corners; single centerline for simple rectangles.
            # For bboxes: single centerline along long axis.
            wall_pieces: List[Dict[str, Any]] = []
            if ann.get("type") == "polygon":
                pieces = _extract_walls_from_polygon(ann["points"])
                for piece in pieces:
                    t_px = float(piece.get("thickness") or 0)
                    t_m = t_px * px_to_m if px_to_m > 0 else default_wall_thickness_m
                    if not (0.04 <= t_m <= 0.60):
                        t_m = default_wall_thickness_m
                    wall_pieces.append({
                        "a_px": piece["a"],
                        "b_px": piece["b"],
                        "thickness_m": t_m,
                    })
            else:
                a_px, b_px = _bbox_to_wall_segment(bbox_px)
                thickness_px = _bbox_thickness(bbox_px)
                t_m = thickness_px * px_to_m if px_to_m > 0 else default_wall_thickness_m
                if not (0.04 <= t_m <= 0.60):
                    t_m = default_wall_thickness_m
                wall_pieces.append({
                    "a_px": a_px,
                    "b_px": b_px,
                    "thickness_m": t_m,
                })

            for k, piece in enumerate(wall_pieces):
                piece_id = ann_id if len(wall_pieces) == 1 else f"{ann_id}_s{k}"
                segments.append({
                    "id": piece_id,
                    "a": [piece["a_px"][0] * px_to_m, piece["a_px"][1] * px_to_m],
                    "b": [piece["b_px"][0] * px_to_m, piece["b_px"][1] * px_to_m],
                    "class": "wall",
                    "confidence": confidence,
                    "source": "manual",
                    "meta": {
                        **meta_common,
                        "wall_type": wall_type,
                        "measured_thickness_m": piece["thickness_m"],
                        "is_reconstructed_wall": True,
                    },
                })
            continue

        # ---- Openings (door/window) and other detection-style elements ----
        bbox_m = [c * px_to_m for c in bbox_px]
        if cls in _DOOR_CLASSES:
            element_class = "door"
        elif cls in _WINDOW_CLASSES:
            element_class = "window"
        elif cls in _STAIRS_CLASSES:
            element_class = "stairs"
        elif cls in _COLUMN_CLASSES:
            element_class = "column"
        elif cls in _ROOM_CLASSES:
            element_class = "room_label"
        else:
            # Unknown but not explicitly ignored -> log and skip
            logger.info("manual_bridge: skipping unknown class %r (id=%s)", cls, ann_id)
            continue

        if element_class == "room_label":
            # Room labels become rooms with a centroid; downstream room
            # detection still runs and may attach polygons.
            cx = (bbox_m[0] + bbox_m[2]) * 0.5
            cy = (bbox_m[1] + bbox_m[3]) * 0.5
            rooms.append({
                "id": ann_id,
                "label": ann.get("label"),
                "center": [cx, cy],
                "polygon": [],
                "bounding_segments": [],
                "source": "manual",
                "meta": meta_common,
            })
            continue

        detections.append({
            "id": ann_id,
            "class": element_class,
            "bbox": bbox_m,
            "confidence": confidence,
            "source": "manual",
            "meta": meta_common,
        })

    return {
        "segments": segments,
        "detections": detections,
        "rooms": rooms,
        "image_size_px": {"width": img_w, "height": img_h},
        "locked_count": locked_count,
        "manual_count": len(segments) + len(detections) + len(rooms),
    }


def find_annotations_file(
    input_path: str | Path,
    annotations_dir: Optional[str | Path] = None,
) -> Optional[Path]:
    """Locate the annotations.json that belongs to a given input plan.

    Search order:
        1. Explicit ``annotations_dir`` if provided
        2. ``<input_dir>/plan_bim_data/<input_stem>/annotations.json``
           (matches Tauri AnnotationEditor convention)
        3. ``<input_dir>/<input_stem>.annotations.json`` (sibling file)
    """
    if annotations_dir:
        p = Path(annotations_dir) / "annotations.json"
        if p.exists():
            return p

    inp = Path(input_path)
    stem = inp.stem
    parent = inp.parent

    convention_path = parent / "plan_bim_data" / stem / "annotations.json"
    if convention_path.exists():
        return convention_path

    sibling_path = parent / f"{stem}.annotations.json"
    if sibling_path.exists():
        return sibling_path

    return None
