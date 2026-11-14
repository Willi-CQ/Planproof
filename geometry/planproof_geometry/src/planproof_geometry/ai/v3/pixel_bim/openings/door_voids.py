"""Boolean-subtract door voids from a wall mesh.

Tag-5 (Plan2BIM V3 Doors Sprint): the 2D placement step yields a
rectangle per door (4 corners in plan-metres + a head height). We
extrude each rectangle from the floor (y=0) to the head height, then
boolean-subtract the resulting box from the wall mesh produced by
``extrude_wall_mask``. The result is a watertight wall mesh with
real openings where the doors sit, and the lintel above each door
remains closed.

We use ``trimesh.boolean.difference`` with ``engine="manifold"`` as
the primary backend (manifold3d is fast and robust) and fall back to
the unmodified wall mesh if the boolean op fails.

Coordinate convention (matches mesh_builder.py + mask_to_mesh.py):
  * X = plan X (metres)
  * Z = -plan Y when ``flip_y_for_pdf=True`` (PDF Y goes down, BIM Y
    goes up)
  * Y = vertical (up)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


# DIN 18101 typical door head height; the lintel runs from here to the
# wall top.
DEFAULT_HEAD_HEIGHT_M = 2.10


@dataclass
class DoorVoid:
    """Minimal record describing a single door void.

    polygon_xy is the 4 corners of the door rectangle in 2D plan-metre
    coordinates (CCW). Floor is implicit at y=0; head_height_m is the
    top of the void. The void is extruded perpendicular to the
    XY-plan, i.e. along the vertical axis of the scene.
    """
    polygon_xy: List[Tuple[float, float]]
    head_height_m: float = DEFAULT_HEAD_HEIGHT_M
    label_id: str = ""


def _door_void_to_mesh(
    void: DoorVoid,
    *,
    flip_y_for_pdf: bool = True,
    inflate_xy_m: float = 0.005,
    inflate_y_m: float = 0.005,
):
    """Build a watertight box mesh for one door void.

    The box is inflated by ``inflate_xy_m`` in plan and ``inflate_y_m``
    in height to ensure boolean subtraction cuts cleanly through the
    wall surface (anti-aliasing on the cut plane is the usual
    boolean-failure mode).
    """
    import trimesh
    from shapely.geometry import Polygon

    if len(void.polygon_xy) < 3:
        return None

    # Inflate the polygon outward in 2D so the cut goes fully through
    # the wall thickness on both sides. Buffer in metres directly.
    poly2d = Polygon(void.polygon_xy)
    if not poly2d.is_valid or poly2d.is_empty:
        poly2d = poly2d.buffer(0)
    if poly2d.is_empty:
        return None
    if inflate_xy_m > 0:
        poly2d = poly2d.buffer(inflate_xy_m, join_style=2)
    if poly2d.is_empty:
        return None

    if poly2d.geom_type == "MultiPolygon":
        poly2d = max(poly2d.geoms, key=lambda g: g.area)

    if flip_y_for_pdf:
        coords = [(x, -y) for x, y in poly2d.exterior.coords]
        poly2d = Polygon(coords)

    height = void.head_height_m + 2.0 * inflate_y_m
    if height <= 1e-6:
        return None

    try:
        box = trimesh.creation.extrude_polygon(poly2d, height)
    except Exception:
        return None

    # extrude_polygon builds along +Z; rotate so height runs along +Y.
    R = trimesh.transformations.rotation_matrix(-math.pi / 2.0, [1, 0, 0])
    box.apply_transform(R)
    # Shift down by inflate_y_m so the void starts slightly below
    # the floor (cuts cleanly through the floor plane of the wall mesh).
    box.apply_translation([0.0, -inflate_y_m, 0.0])
    return box


def apply_door_voids(
    wall_mesh,
    door_voids: Sequence[DoorVoid],
    *,
    wall_height_m: float = 2.80,
    flip_y_for_pdf: bool = True,
    boolean_engine: str = "manifold",
):
    """Subtract each door void from the wall mesh.

    Returns the modified mesh. On any boolean failure, returns the
    original wall mesh untouched (with a printed warning) so the
    pipeline never breaks downstream rendering.
    """
    import trimesh

    if not door_voids:
        return wall_mesh
    if wall_mesh is None or wall_mesh.is_empty or len(wall_mesh.vertices) < 4:
        return wall_mesh

    void_meshes = []
    for v in door_voids:
        head = min(v.head_height_m, wall_height_m - 0.02)
        if head <= 0.05:
            continue
        v_clamped = DoorVoid(
            polygon_xy=v.polygon_xy,
            head_height_m=head,
            label_id=v.label_id,
        )
        m = _door_void_to_mesh(v_clamped, flip_y_for_pdf=flip_y_for_pdf)
        if m is not None and not m.is_empty:
            void_meshes.append(m)

    if not void_meshes:
        return wall_mesh

    # Combine all voids into one mesh for a single boolean call (faster
    # than N successive subtractions on most engines).
    voids_combined = trimesh.util.concatenate(void_meshes)

    try:
        result = trimesh.boolean.difference(
            [wall_mesh, voids_combined],
            engine=boolean_engine,
        )
    except Exception as exc:
        print(
            f"  [door-voids] boolean.difference engine={boolean_engine} "
            f"failed: {exc} — wall mesh returned without voids"
        )
        return wall_mesh

    if result is None or (
        hasattr(result, "is_empty") and result.is_empty
    ) or (
        hasattr(result, "vertices") and len(result.vertices) < 4
    ):
        print(
            "  [door-voids] boolean.difference returned empty/degenerate "
            "result — wall mesh returned without voids"
        )
        return wall_mesh

    return result


def door_voids_from_polygons(
    polygons: Sequence[Sequence[Tuple[float, float]]],
    *,
    head_height_m: float = DEFAULT_HEAD_HEIGHT_M,
    label_ids: Optional[Sequence[str]] = None,
) -> List[DoorVoid]:
    """Convenience: turn a list of 2D rectangles into ``DoorVoid``s."""
    out: List[DoorVoid] = []
    if label_ids is None:
        label_ids = [f"door_{i}" for i in range(len(polygons))]
    for poly, lid in zip(polygons, label_ids):
        if len(poly) < 3:
            continue
        out.append(DoorVoid(
            polygon_xy=[(float(x), float(y)) for x, y in poly],
            head_height_m=head_height_m,
            label_id=lid,
        ))
    return out
