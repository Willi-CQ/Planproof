"""Build a straight-run stair mesh from a StairRecord.

The stair is modelled as a monolithic concrete staircase: N stacked
boxes, one per step. Each box i extends along the long axis from
``i * going`` to ``(i + 1) * going`` and from the floor up to
``(i + 1) * rise``. Side view = classic staircase silhouette.

Two construction modes:

    * ``build_stair_mesh(record)`` — classic rectangular-footprint
      mode. Each step is a uniform rectangular box of size
      ``going_m × width_m × (i+1)·rise_m``.
    * ``build_stair_mesh_from_polygon(record, stair_polygon_xy_m)`` —
      Tag 9 polygon mode. Each step is sliced from the trimmed stair
      footprint polygon (after wall-subtraction) — so the per-step
      shape follows the real stair outline, including notches where
      walls intrude into the user bbox.

Coordinate convention follows ``openings.mesh_builder``:
    * shapely polygons are built in (plan_x, -plan_y) — Y is flipped
      because PDF Y points down while scene Y points up.
    * extrude_polygon extrudes along local Z.
    * a -pi/2 rotation around X maps that local Z to scene Y, leaving
      plan_X -> scene_X and plan_Y -> scene_Z (double negation cancels).

Walls remain immutable: this module produces an independent mesh that
sits on the floor in front of (or against) walls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .records import StairRecord

# Slightly lighter than the wall colour (220, 220, 220) so stair volumes
# read as concrete in the BIM viewer without clashing with the wall mesh.
_STAIR_RGBA = (200, 200, 205, 255)


@dataclass
class StairMesh:
    """All meshes produced for one stair. ``flight`` may be None."""
    flight: Optional["trimesh.Trimesh"] = None  # type: ignore[name-defined]  # noqa: F821
    materials: List[str] = field(default_factory=list)


def _step_box(
    *,
    centroid_xy_m: Tuple[float, float],
    direction_up_xy: Tuple[float, float],
    short_axis_xy: Tuple[float, float],
    length_m: float,
    width_m: float,
    going_m: float,
    rise_m: float,
    step_index: int,
):
    """Build one watertight tread box (mesh) for step ``step_index``.

    Returns a trimesh.Trimesh in scene coordinates (Y up), or None on
    degenerate input.
    """
    import trimesh  # noqa: PLC0415
    from shapely.geometry import Polygon  # noqa: PLC0415

    if going_m <= 1e-6 or width_m <= 1e-6 or rise_m <= 1e-6:
        return None

    half_width = width_m * 0.5
    low_end_xy = (
        centroid_xy_m[0] - 0.5 * length_m * direction_up_xy[0],
        centroid_xy_m[1] - 0.5 * length_m * direction_up_xy[1],
    )
    t_lo = step_index * going_m
    t_hi = t_lo + going_m

    # Four plan-XY corners (CCW in tread frame), then build polygon.
    def _xy(t: float, s: float) -> Tuple[float, float]:
        return (
            low_end_xy[0] + t * direction_up_xy[0] + s * short_axis_xy[0],
            low_end_xy[1] + t * direction_up_xy[1] + s * short_axis_xy[1],
        )

    corners_xy = [
        _xy(t_lo, -half_width),
        _xy(t_hi, -half_width),
        _xy(t_hi, +half_width),
        _xy(t_lo, +half_width),
    ]
    # Flip Y for PDF -> scene-friendly Y direction (matches openings mesh_builder).
    corners_flipped = [(x, -y) for x, y in corners_xy]
    poly = Polygon(corners_flipped).buffer(0)
    if poly.is_empty or not poly.is_valid:
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)

    height_y_m = (step_index + 1) * rise_m
    mesh = trimesh.creation.extrude_polygon(poly, height_y_m)
    R = trimesh.transformations.rotation_matrix(-math.pi / 2.0, [1, 0, 0])
    mesh.apply_transform(R)
    return mesh


def build_stair_mesh(record: StairRecord) -> StairMesh:
    """Concatenate N stacked tread boxes into one monolithic stair mesh.

    The function writes ``length_m`` / ``width_m`` / ``long_axis_deg``
    onto the record only if missing — typical use is that these are
    already populated by the footprint step.
    """
    import trimesh  # noqa: PLC0415

    out = StairMesh()
    n_steps = record.n_steps
    rise_m = record.rise_m
    going_m = record.going_m
    width_m = record.width_m
    length_m = record.length_m
    if not n_steps or not rise_m or not going_m or width_m <= 0 or length_m <= 0:
        return out

    direction_up_xy = record.direction_up_xy
    norm = math.hypot(direction_up_xy[0], direction_up_xy[1])
    if norm <= 1e-9:
        return out
    direction_up_xy = (direction_up_xy[0] / norm, direction_up_xy[1] / norm)
    short_axis_xy = (-direction_up_xy[1], direction_up_xy[0])

    boxes: List[trimesh.Trimesh] = []
    for i in range(n_steps):
        box = _step_box(
            centroid_xy_m=record.centroid_xy_m,
            direction_up_xy=direction_up_xy,
            short_axis_xy=short_axis_xy,
            length_m=length_m,
            width_m=width_m,
            going_m=going_m,
            rise_m=rise_m,
            step_index=i,
        )
        if box is not None:
            boxes.append(box)

    if not boxes:
        return out

    flight = trimesh.util.concatenate(boxes) if len(boxes) > 1 else boxes[0]
    flight.visual.face_colors = np.array(_STAIR_RGBA, dtype=np.uint8)
    out.flight = flight
    return out


# ---------------------------------------------------------------------------
# Tag 9 — Polygon-based mesh builder (Walls-as-Boundary)  [V1, kept for tests]
# ---------------------------------------------------------------------------


def _step_slice_box(
    *,
    stair_polygon_xy,  # shapely Polygon in (plan_x, -plan_y) frame
    direction_up_xy: Tuple[float, float],
    short_axis_xy: Tuple[float, float],
    centroid_xy_m: Tuple[float, float],
    length_m: float,
    width_m: float,
    going_m: float,
    rise_m: float,
    step_index: int,
):
    """Slice ``stair_polygon_xy`` at step ``step_index`` and extrude.

    The slice is the intersection of the stair polygon with a strip
    perpendicular to ``direction_up_xy`` between long-axis positions
    ``[step_index·going_m, (step_index+1)·going_m]`` (measured from
    the low-end of the stair).

    Returns a trimesh.Trimesh in scene coordinates, or None if the
    slice is empty / invalid.
    """
    import trimesh  # noqa: PLC0415
    from shapely.geometry import MultiPolygon, Polygon  # noqa: PLC0415

    if going_m <= 1e-6 or rise_m <= 1e-6:
        return None

    half_width = max(width_m, 1.0) * 0.5
    low_end_xy = (
        centroid_xy_m[0] - 0.5 * length_m * direction_up_xy[0],
        centroid_xy_m[1] - 0.5 * length_m * direction_up_xy[1],
    )
    t_lo = step_index * going_m
    t_hi = t_lo + going_m

    def _xy(t: float, s: float) -> Tuple[float, float]:
        return (
            low_end_xy[0] + t * direction_up_xy[0] + s * short_axis_xy[0],
            low_end_xy[1] + t * direction_up_xy[1] + s * short_axis_xy[1],
        )

    # Strip corners in plan-XY (CCW).
    strip_xy = [
        _xy(t_lo, -half_width),
        _xy(t_hi, -half_width),
        _xy(t_hi, +half_width),
        _xy(t_lo, +half_width),
    ]
    # Flip Y to match the (plan_x, -plan_y) frame of stair_polygon_xy.
    strip_flipped = [(x, -y) for x, y in strip_xy]
    strip = Polygon(strip_flipped).buffer(0)
    if strip.is_empty or not strip.is_valid:
        return None

    slice_geom = stair_polygon_xy.intersection(strip)
    if slice_geom.is_empty:
        return None

    # Reduce MultiPolygon to its individual parts; each part becomes a
    # separate extruded box that we concat outside this function.
    if isinstance(slice_geom, MultiPolygon):
        parts = [g for g in slice_geom.geoms if not g.is_empty and g.is_valid]
    elif isinstance(slice_geom, Polygon):
        parts = [slice_geom]
    else:
        # GeometryCollection (mixed types) — pick polygons only.
        parts = [
            g for g in getattr(slice_geom, "geoms", [])
            if isinstance(g, Polygon) and not g.is_empty
        ]
    if not parts:
        return None

    height_y_m = (step_index + 1) * rise_m
    meshes: List["trimesh.Trimesh"] = []
    for part in parts:
        try:
            sub = part.buffer(0)
            if sub.is_empty or not sub.is_valid:
                continue
            if isinstance(sub, MultiPolygon):
                sub = max(sub.geoms, key=lambda g: g.area)
            mesh = trimesh.creation.extrude_polygon(sub, height_y_m)
        except Exception:
            continue
        R = trimesh.transformations.rotation_matrix(-math.pi / 2.0, [1, 0, 0])
        mesh.apply_transform(R)
        meshes.append(mesh)

    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def build_stair_mesh_from_polygon(
    record: StairRecord,
    stair_polygon_xy_m,  # shapely Polygon in (plan_x, -plan_y) frame
) -> StairMesh:
    """Tag 9 polygon-aware mesh builder.

    Args:
        record: StairRecord with populated n_steps / rise_m / going_m
            / direction_up_xy / centroid_xy_m / length_m / width_m.
        stair_polygon_xy_m: Shapely polygon of the stair footprint in
            the *flipped* plan frame (plan_x, -plan_y). Caller is
            responsible for the Y-flip — it matches the convention of
            ``_step_box``.

    Returns:
        ``StairMesh`` with ``flight`` populated, or empty mesh on
        degenerate inputs.
    """
    import trimesh  # noqa: PLC0415

    out = StairMesh()
    n_steps = record.n_steps
    rise_m = record.rise_m
    going_m = record.going_m
    width_m = record.width_m
    length_m = record.length_m
    if not n_steps or not rise_m or not going_m or width_m <= 0 or length_m <= 0:
        return out

    direction_up_xy = record.direction_up_xy
    norm = math.hypot(direction_up_xy[0], direction_up_xy[1])
    if norm <= 1e-9:
        return out
    direction_up_xy = (direction_up_xy[0] / norm, direction_up_xy[1] / norm)
    short_axis_xy = (-direction_up_xy[1], direction_up_xy[0])

    boxes: List[trimesh.Trimesh] = []
    for i in range(n_steps):
        box = _step_slice_box(
            stair_polygon_xy=stair_polygon_xy_m,
            direction_up_xy=direction_up_xy,
            short_axis_xy=short_axis_xy,
            centroid_xy_m=record.centroid_xy_m,
            length_m=length_m,
            width_m=width_m,
            going_m=going_m,
            rise_m=rise_m,
            step_index=i,
        )
        if box is not None:
            boxes.append(box)

    if not boxes:
        return out

    flight = trimesh.util.concatenate(boxes) if len(boxes) > 1 else boxes[0]
    flight.visual.face_colors = np.array(_STAIR_RGBA, dtype=np.uint8)
    out.flight = flight
    return out


# =========================================================================
# == V2 DESIGN (editable) — slabs + nosing + wood tone                    ==
# == Visual design is iterated freely; the V2 algorithm itself is locked. ==
# == See stair_v2_cells.py / stair_v2_arrow.py / stair_v2_target_rules.py ==
# =========================================================================
_V2_NOSING_M = 0.025                                # DIN 18065 tread overhang
_V2_TREAD_TOP_RGBA = (188, 149, 92, 255)            # warm oak top
_V2_TREAD_SIDE_RGBA = (142, 105, 60, 255)           # darker side / riser
_V2_BOTTOM_DARKEN = 0.78                            # bottom-face shading


def build_v2_flight_mesh(
    cells_xy_m_ordered: List[List[Tuple[float, float]]],
    *,
    rise_m: float,
    direction_xy: Tuple[float, float],
    junction_step_xy_m: Optional[List[Tuple[float, float]]] = None,
    junction_top_m: Optional[float] = None,
    nosing_m: float = _V2_NOSING_M,
    tread_top_rgba: Tuple[int, int, int, int] = _V2_TREAD_TOP_RGBA,
    tread_side_rgba: Tuple[int, int, int, int] = _V2_TREAD_SIDE_RGBA,
    bottom_darken: float = _V2_BOTTOM_DARKEN,
):  # -> trimesh.Trimesh | None
    """Build the production-design V2 flight mesh from cell polygons.

    Each cell becomes one rise-thick slab at its own z-band
    (``k*rise`` to ``(k+1)*rise``) so the staircase reads as discrete
    stepped slabs (no floor-to-top wedge). Each slab is extended
    ``nosing_m`` in the climb direction so the tread visually overhangs
    the riser below per DIN 18065. Top faces use the lighter wood tone,
    side faces a darker tone so the riser reads as a recessed shadow.

    Args:
        cells_xy_m_ordered: tread polygons in plan-XY metres, ordered
            START -> END along the climb direction. PDF Y down.
        rise_m: per-step rise (height of one slab).
        direction_xy: unit climb direction in plan-XY metres (PDF frame,
            Y down). Drives the nosing offset.
        junction_step_xy_m: optional bridge polygon for the unifying
            step at the top (paired U-stairs).
        junction_top_m: top z of the junction step.
        nosing_m: tread overhang per DIN 18065 (default 2.5 cm).
        tread_top_rgba / tread_side_rgba: face colours.
        bottom_darken: multiplier applied to side RGB for the bottom
            face (a subtle shadow keeps the slab from reading flat).

    Returns:
        A ``trimesh.Trimesh`` ready to add to a scene, or ``None`` when
        no slabs could be built (degenerate cells, etc.).
    """
    import trimesh  # noqa: PLC0415
    from shapely.affinity import translate as _shp_translate  # noqa: PLC0415
    from shapely.geometry import Polygon as _ShPoly  # noqa: PLC0415

    if not cells_xy_m_ordered or rise_m <= 0.0:
        return None

    # The nosing is applied in the FLIPPED-Y frame (the same one we
    # extrude in below), so flip the climb direction once here.
    d_nose = (float(direction_xy[0]), -float(direction_xy[1]))

    def _nosed(sp):
        try:
            shifted = _shp_translate(
                sp, xoff=d_nose[0] * nosing_m, yoff=d_nose[1] * nosing_m
            )
            merged = sp.union(shifted).convex_hull
            return merged if not merged.is_empty else sp
        except Exception:
            return sp

    def _color_faces(mesh):
        n = mesh.face_normals
        is_top = n[:, 2] > 0.7
        is_bot = n[:, 2] < -0.7
        is_side = ~(is_top | is_bot)
        faces = np.empty((len(n), 4), np.uint8)
        faces[is_top] = np.array(tread_top_rgba, np.uint8)
        faces[is_side] = np.array(tread_side_rgba, np.uint8)
        bot = np.array(tread_side_rgba, np.float32)
        bot[:3] *= bottom_darken
        faces[is_bot] = bot.astype(np.uint8)
        mesh.visual.face_colors = faces

    R = trimesh.transformations.rotation_matrix(-math.pi / 2.0, [1, 0, 0])

    def _slab(cell_xy_m, z_lo, z_hi):
        if z_hi - z_lo <= 1e-6:
            return None
        ring = [(float(x), -float(y)) for (x, y) in cell_xy_m]  # flip Y
        try:
            sp = _ShPoly(ring).buffer(0)
        except Exception:
            return None
        if sp.is_empty or sp.area <= 0.0:
            return None
        if sp.geom_type == "MultiPolygon":
            sp = max(sp.geoms, key=lambda g: g.area)
        sp = _nosed(sp)
        try:
            m = trimesh.creation.extrude_polygon(sp, z_hi - z_lo)
        except Exception:
            return None
        m.apply_translation([0.0, 0.0, z_lo])
        _color_faces(m)
        m.apply_transform(R)
        return m

    slabs = []
    for rank, cell in enumerate(cells_xy_m_ordered):
        s = _slab(cell, rank * rise_m, (rank + 1) * rise_m)
        if s is not None:
            slabs.append(s)

    if junction_step_xy_m and junction_top_m is not None:
        h = float(junction_top_m)
        s = _slab(junction_step_xy_m, h - rise_m, h)
        if s is not None:
            slabs.append(s)

    if not slabs:
        return None
    return trimesh.util.concatenate(slabs) if len(slabs) > 1 else slabs[0]
