from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

Point2 = Tuple[float, float]


@dataclass
class RoomShell:
    """Minimal geometric representation of a room boundary."""
    wall_segments: List[Dict]
    width_m: float
    depth_m: float
    height_m: float
    origin_xy: Point2 = (0.0, 0.0)
    # Echter Footprint-Ring (Liste [x, y] szenenlokal) aus ``detect_rooms`` —
    # traegt die reale Raumkontur bis in Scene/GLB/Vergleich, statt sie auf die
    # achsenparallele Bounding-Box zu reduzieren. ``None`` = nur Rechteck bekannt
    # (Fallback-Shell); ``build_scene`` synthetisiert dann einen Rechteck-Ring.
    footprint: Optional[List[Point2]] = None
    # Stabiler Raum-Schluessel/-Id aus ``detect_rooms`` (z.B. ``room_1``) +
    # Shoelace-Flaeche, damit ``build_scene`` room_key + echte Flaeche tragen kann.
    room_id: Optional[str] = None
    area_m2: Optional[float] = None


def build_rect_room_shell(
    width_m: float,
    depth_m: float,
    height_m: float,
    origin_xy: Point2 = (0.0, 0.0),
) -> RoomShell:
    """Deterministic rectangular room shell (quality fallback)."""
    ox, oy = origin_xy
    p0 = (ox - width_m / 2.0, oy - depth_m / 2.0)
    p1 = (ox + width_m / 2.0, oy - depth_m / 2.0)
    p2 = (ox + width_m / 2.0, oy + depth_m / 2.0)
    p3 = (ox - width_m / 2.0, oy + depth_m / 2.0)

    segs = [
        {"id": "w1", "a": p0, "b": p1, "kind": "wall"},
        {"id": "w2", "a": p1, "b": p2, "kind": "wall"},
        {"id": "w3", "a": p2, "b": p3, "kind": "wall"},
        {"id": "w4", "a": p3, "b": p0, "kind": "wall"},
    ]
    return RoomShell(wall_segments=segs, width_m=width_m, depth_m=depth_m, height_m=height_m, origin_xy=origin_xy)
