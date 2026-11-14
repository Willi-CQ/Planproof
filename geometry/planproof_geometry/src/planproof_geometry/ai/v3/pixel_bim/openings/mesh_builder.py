"""Build the parapet + lintel meshes per door opening.

Each mesh is a box sitting in the gap between W1 and W2, matching the
wall thickness and aligned with the common wall axis. The fill (door
swing area) is left as void so the wall mesh shows the opening clearly.

Coordinate convention (matches extrude/mask_to_mesh.py):
  - X = plan X
  - Z = -plan Y (flipped; PDF Y goes down, scene Y goes up)
  - Y = vertical (up)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .gap_geometry import GapGeometry
from .records import OpeningRecord


@dataclass
class OpeningMeshes:
    """All meshes produced for one opening. Any of them may be None."""
    parapet: Optional["trimesh.Trimesh"] = None  # type: ignore[name-defined]  # noqa: F821
    lintel: Optional["trimesh.Trimesh"] = None   # type: ignore[name-defined]  # noqa: F821
    fill: Optional["trimesh.Trimesh"] = None     # type: ignore[name-defined]  # noqa: F821
    frame: Optional["trimesh.Trimesh"] = None    # type: ignore[name-defined]  # noqa: F821
    # Bridge: flat slab on top of the door frame extending into adjacent
    # wall stubs. Only built for swing-door classes (door, door_single,
    # door_double); sliding doors leave this None.
    bridge: Optional["trimesh.Trimesh"] = None   # type: ignore[name-defined]  # noqa: F821
    materials: List[str] = field(default_factory=list)


def _rectangle_corners_m(
    origin: Tuple[float, float],
    major: Tuple[float, float],
    minor: Tuple[float, float],
    t_lo: float, t_hi: float,
    s_lo: float, s_hi: float,
) -> List[Tuple[float, float]]:
    """Build 4 corners of an axis-aligned rectangle in the major/minor frame.

    Returns the corners in CCW order for shapely.
    """
    corners = [(t_lo, s_lo), (t_hi, s_lo), (t_hi, s_hi), (t_lo, s_hi)]
    return [
        (origin[0] + t * major[0] + s * minor[0],
         origin[1] + t * major[1] + s * minor[1])
        for (t, s) in corners
    ]


def _extrude_box_in_gap(
    origin_xy_m: Tuple[float, float],
    major: Tuple[float, float],
    minor: Tuple[float, float],
    gap_start: float, gap_end: float,
    minor_lo: float, minor_hi: float,
    *,
    base_y_m: float,
    height_y_m: float,
    flip_y_for_pdf: bool = True,
):
    """Build one watertight box mesh sitting on the floor at base_y_m."""
    import trimesh
    from shapely.geometry import Polygon

    if height_y_m <= 1e-6 or gap_end - gap_start <= 1e-6             or minor_hi - minor_lo <= 1e-6:
        return None

    corners = _rectangle_corners_m(
        origin_xy_m, major, minor,
        gap_start, gap_end, minor_lo, minor_hi,
    )
    if flip_y_for_pdf:
        corners = [(x, -y) for x, y in corners]
    poly = Polygon(corners).buffer(0)
    if poly.is_empty or not poly.is_valid:
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)

    mesh = trimesh.creation.extrude_polygon(poly, height_y_m)
    R = trimesh.transformations.rotation_matrix(-math.pi / 2.0, [1, 0, 0])
    mesh.apply_transform(R)
    mesh.apply_translation([0.0, base_y_m, 0.0])
    return mesh


def build_opening_meshes(
    record: OpeningRecord,
    gap: GapGeometry,
    *,
    wall_height_m: float = 2.80,
    flip_y_for_pdf: bool = True,
) -> OpeningMeshes:
    """Build parapet/lintel for one door opening.

    The function writes width_m / depth_m / axis_deg / centre into the
    ``record`` in place so the caller does not have to.
    """
    s_half = gap.wall_depth_m * 0.5
    s_lo, s_hi = -s_half, s_half

    record.width_m = gap.gap_length_m
    record.depth_m = gap.wall_depth_m
    record.axis_deg = math.degrees(
        math.atan2(gap.major_axis[1], gap.major_axis[0])
    ) % 180.0
    t_mid = 0.5 * (gap.gap_start + gap.gap_end)
    cx = gap.origin[0] + t_mid * gap.major_axis[0]
    cy = gap.origin[1] + t_mid * gap.major_axis[1]
    record.center_xy_m = (cx, cy)

    meshes = OpeningMeshes()

    sill = record.sill_m
    head = record.head_m

    # 1. Parapet (floor -> sill). Skip if sill == 0 (standard doors).
    if sill > 1e-3:
        parapet = _extrude_box_in_gap(
            gap.origin, gap.major_axis, gap.minor_axis,
            gap.gap_start, gap.gap_end, s_lo, s_hi,
            base_y_m=0.0, height_y_m=sill,
            flip_y_for_pdf=flip_y_for_pdf,
        )
        meshes.parapet = parapet

    # 2. Lintel (head -> wall_height).
    lintel_height = max(0.0, wall_height_m - head)
    if lintel_height > 1e-3:
        lintel = _extrude_box_in_gap(
            gap.origin, gap.major_axis, gap.minor_axis,
            gap.gap_start, gap.gap_end, s_lo, s_hi,
            base_y_m=head, height_y_m=lintel_height,
            flip_y_for_pdf=flip_y_for_pdf,
        )
        meshes.lintel = lintel

    # 3. Fill (sill -> head): door swing remains as void; no mesh built.

    # 4. Bridge (frame_top -> wall_height, +overlap into wall stubs).
    # Only swing-door classes; sliding doors leave bridge=None.
    from .door_bridges import maybe_build_bridge  # noqa: PLC0415
    meshes.bridge = maybe_build_bridge(
        record.label_class, gap,
        wall_height_m=wall_height_m,
        flip_y_for_pdf=flip_y_for_pdf,
    )

    return meshes
