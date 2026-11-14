"""Parametric BIM mesh generators for plan-to-BIM pipeline.

Each generator produces MeshPart(s) with baked-in geometry (vertices, normals,
indices) — no unit-cube scaling tricks.  Materials reference names from
BIM_MATERIALS for PBR rendering in glTF.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

Vec3 = Tuple[float, float, float]


# ── Data structures ────────────────────────────────────────────────


@dataclass
class MeshPart:
    """A single mesh primitive (vertices + normals + indices + material)."""
    name: str
    vertices: List[Vec3]
    normals: List[Vec3]
    indices: List[int]
    material_name: str


# ── PBR Materials ──────────────────────────────────────────────────


BIM_MATERIALS: Dict[str, Dict] = {
    "wall_exterior": {
        "baseColorFactor": [0.75, 0.72, 0.68, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.9,
    },
    "wall_interior": {
        "baseColorFactor": [0.90, 0.88, 0.85, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.8,
    },
    "door_frame": {
        "baseColorFactor": [0.45, 0.30, 0.18, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.6,
    },
    "door_panel": {
        "baseColorFactor": [0.55, 0.38, 0.22, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.5,
    },
    "window_frame": {
        "baseColorFactor": [0.80, 0.80, 0.82, 1.0],
        "metallicFactor": 0.1,
        "roughnessFactor": 0.3,
    },
    "window_glass": {
        "baseColorFactor": [0.7, 0.85, 0.95, 0.3],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.1,
        "alphaMode": "BLEND",
    },
    "column": {
        "baseColorFactor": [0.50, 0.50, 0.55, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.8,
    },
    "stairs": {
        "baseColorFactor": [0.60, 0.58, 0.55, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.9,
    },
    "slab": {
        "baseColorFactor": [0.50, 0.48, 0.45, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.95,
    },
    # Room floor materials — semi-transparent so slab and wall outlines show through
    "room_floor_living": {
        "baseColorFactor": [0.82, 0.68, 0.48, 0.75],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.8,
        "alphaMode": "BLEND",
    },
    "room_floor_bath": {
        "baseColorFactor": [0.65, 0.75, 0.80, 0.75],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.6,
        "alphaMode": "BLEND",
    },
    "room_floor_kitchen": {
        "baseColorFactor": [0.85, 0.80, 0.65, 0.75],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.7,
        "alphaMode": "BLEND",
    },
    "room_floor_generic": {
        "baseColorFactor": [0.88, 0.83, 0.75, 0.75],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.8,
        "alphaMode": "BLEND",
    },
}


# ── Box mesh helper ────────────────────────────────────────────────


def _box_mesh(
    sx: float, sy: float, sz: float,
    name: str = "box",
    material: str = "wall_exterior",
) -> MeshPart:
    """Generate an axis-aligned box centered at origin with correct normals.

    Parameters: sx, sy, sz = full extents along X, Y, Z.
    """
    hx, hy, hz = sx / 2, sy / 2, sz / 2

    # 6 faces × 4 vertices = 24 vertices (for flat-shaded normals)
    #   Face order: +X, -X, +Y, -Y, +Z, -Z
    vertices: List[Vec3] = []
    normals: List[Vec3] = []
    indices: List[int] = []

    face_defs = [
        # (normal, 4 corner offsets)
        ((1, 0, 0), [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)]),
        ((-1, 0, 0), [(-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz), (-hx, -hy, -hz)]),
        ((0, 1, 0), [(-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz)]),
        ((0, -1, 0), [(-hx, -hy, hz), (-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz)]),
        ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        ((0, 0, -1), [(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)]),
    ]

    for normal, corners in face_defs:
        base = len(vertices)
        for c in corners:
            vertices.append(c)
            normals.append(normal)
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    return MeshPart(name=name, vertices=vertices, normals=normals, indices=indices, material_name=material)


# ── Wall ───────────────────────────────────────────────────────────


def generate_wall_mesh(
    length: float,
    height: float,
    thickness: float,
    wall_type: str = "exterior",
) -> List[MeshPart]:
    """Wall: single box, length × height × thickness, centered at origin.

    The wall extends along X (length), Y (height), Z (thickness).
    """
    mat = "wall_exterior" if wall_type == "exterior" else "wall_interior"
    return [_box_mesh(length, height, thickness, name="wall", material=mat)]


@dataclass
class WallOpening:
    """An opening (door/window) positioned along a wall's local X axis."""
    offset: float     # distance from wall center-left to opening center along X
    width: float      # opening width
    bottom: float = 0.0   # Y of opening bottom (0 for doors)
    top: float = 2.10     # Y of opening top


def generate_wall_with_openings_mesh(
    length: float,
    height: float,
    thickness: float,
    openings: List[WallOpening],
    wall_type: str = "exterior",
) -> List[MeshPart]:
    """Wall with rectangular cutouts for doors/windows.

    Splits the wall into sub-boxes around each opening:
    - Horizontal strips left/between/right of openings
    - Lintel (above each opening)
    - Sill (below each window opening, where bottom > 0)

    All parts are centered at origin like generate_wall_mesh, with the wall
    extending along X (length), Y (height), Z (thickness).  Openings are
    positioned by *offset* — distance from wall's left edge (X = -length/2)
    to the opening center.
    """
    if not openings:
        return generate_wall_mesh(length, height, thickness, wall_type)

    mat = "wall_exterior" if wall_type == "exterior" else "wall_interior"
    parts: List[MeshPart] = []
    hl = length / 2  # half-length

    # Sort openings left-to-right by their left edge
    sorted_openings = sorted(openings, key=lambda o: o.offset - o.width / 2)

    # Clamp openings to wall bounds and merge overlapping ones
    clamped: List[WallOpening] = []
    for op in sorted_openings:
        left = max(-hl, op.offset - op.width / 2)
        right = min(hl, op.offset + op.width / 2)
        if right - left < 0.02:
            continue
        clamped.append(WallOpening(
            offset=(left + right) / 2,
            width=right - left,
            bottom=max(0.0, op.bottom),
            top=min(height, op.top),
        ))

    if not clamped:
        return generate_wall_mesh(length, height, thickness, wall_type)

    # Build horizontal wall strips (full-height) and lintels/sills
    cursor_x = -hl  # current X position (left edge of wall)

    for i, op in enumerate(clamped):
        op_left = op.offset - op.width / 2
        op_right = op.offset + op.width / 2

        # Strip to the left of this opening (full height)
        strip_w = op_left - cursor_x
        if strip_w > 0.005:
            strip = _box_mesh(strip_w, height, thickness, name=f"wall_strip_{i}", material=mat)
            strip_cx = cursor_x + strip_w / 2
            _offset_vertices(strip, dx=strip_cx, dy=0, dz=0)
            parts.append(strip)

        # Lintel above opening (from op.top to wall height).
        # _box_mesh centers the box at Y=0; the exporter shifts all parts up by
        # height/2 to move Y=0 to floor level, so we subtract height/2 here to
        # keep the lintel in the correct floor-relative position after that shift.
        lintel_h = height - op.top
        if lintel_h > 0.005:
            lintel = _box_mesh(op.width, lintel_h, thickness, name=f"wall_lintel_{i}", material=mat)
            lintel_cy = op.top + lintel_h / 2 - height / 2
            _offset_vertices(lintel, dx=op.offset, dy=lintel_cy, dz=0)
            parts.append(lintel)

        # Sill below opening (for windows with bottom > 0).
        # Same origin-centering adjustment as the lintel.
        if op.bottom > 0.005:
            sill = _box_mesh(op.width, op.bottom, thickness, name=f"wall_sill_{i}", material=mat)
            sill_cy = op.bottom / 2 - height / 2
            _offset_vertices(sill, dx=op.offset, dy=sill_cy, dz=0)
            parts.append(sill)

        cursor_x = op_right

    # Final strip to the right of last opening
    strip_w = hl - cursor_x
    if strip_w > 0.005:
        strip = _box_mesh(strip_w, height, thickness, name="wall_strip_end", material=mat)
        strip_cx = cursor_x + strip_w / 2
        _offset_vertices(strip, dx=strip_cx, dy=0, dz=0)
        parts.append(strip)

    if not parts:
        return generate_wall_mesh(length, height, thickness, wall_type)

    return parts


# ── Door ───────────────────────────────────────────────────────────


def generate_door_mesh(
    width: float = 0.90,
    height: float = 2.10,
    frame_depth: float = 0.08,
    frame_width: float = 0.05,
    panel_thickness: float = 0.04,
    swing_direction: Optional[str] = None,
    leaves: int = 1,
    sliding: bool = False,
) -> List[MeshPart]:
    """Door: frame (2 verticals + 1 horizontal) + panel(s) + optional swing arc.

    Centered at the door's midpoint, Y=0 at floor.
    Frame pieces are offset so the door opening is clear.

    Phase E support:
        leaves=2  → double-leaf door: two panels meeting in the middle
        sliding=True → sliding door: no swing arc, panels rendered as
                       thin slid-open slabs parallel to the opening
    """
    parts: List[MeshPart] = []
    hw = width / 2

    # Left vertical frame post
    left_post = _box_mesh(frame_width, height, frame_depth, name="door_frame_left", material="door_frame")
    _offset_vertices(left_post, dx=-(hw - frame_width / 2), dy=height / 2, dz=0)
    parts.append(left_post)

    # Right vertical frame post
    right_post = _box_mesh(frame_width, height, frame_depth, name="door_frame_right", material="door_frame")
    _offset_vertices(right_post, dx=(hw - frame_width / 2), dy=height / 2, dz=0)
    parts.append(right_post)

    # Top horizontal beam
    top_beam = _box_mesh(width, frame_width, frame_depth, name="door_frame_top", material="door_frame")
    _offset_vertices(top_beam, dx=0, dy=height - frame_width / 2, dz=0)
    parts.append(top_beam)

    # Door panel(s)
    panel_h = height - frame_width
    inside_w = width - 2 * frame_width
    if sliding:
        # Sliding door: two thin panels, one open (offset to the side)
        # Convention: main panel spans the opening, secondary panel slides
        # behind the wall (offset by inside_w/2 to +x, minimal z-offset to
        # suggest the rail).
        slide_panel_w = inside_w
        slide_panel = _box_mesh(
            slide_panel_w, panel_h, panel_thickness * 0.7,
            name="door_panel_sliding", material="door_panel",
        )
        _offset_vertices(slide_panel, dx=0, dy=panel_h / 2, dz=frame_depth * 0.5)
        parts.append(slide_panel)
        # Thin rail on top
        rail = _box_mesh(width, 0.02, 0.04, name="door_rail", material="door_frame")
        _offset_vertices(rail, dx=0, dy=height - 0.02, dz=frame_depth * 0.5)
        parts.append(rail)
    elif leaves == 2:
        # Double-leaf door: two half-width panels meeting in the middle.
        leaf_w = inside_w / 2 - 0.005  # 5mm gap between leaves
        for i, sign in enumerate((-1, 1)):
            leaf = _box_mesh(
                leaf_w, panel_h, panel_thickness,
                name=f"door_panel_leaf_{i}", material="door_panel",
            )
            _offset_vertices(leaf, dx=sign * (leaf_w / 2 + 0.005), dy=panel_h / 2, dz=0)
            parts.append(leaf)
    else:
        # Single-leaf door
        panel = _box_mesh(
            inside_w, panel_h, panel_thickness,
            name="door_panel", material="door_panel",
        )
        _offset_vertices(panel, dx=0, dy=panel_h / 2, dz=0)
        parts.append(panel)

    # Swing arc indicator on the floor (only for hinged doors)
    if not sliding and swing_direction in ("left", "right"):
        arc = _generate_swing_arc(inside_w, swing_direction)
        parts.append(arc)

    return parts


def _generate_swing_arc(radius: float, direction: str) -> MeshPart:
    """Generate a thin quarter-circle arc on the floor plane (Y≈0)."""
    segments = 16
    y_low = 0.001
    y_high = 0.005
    vertices: List[Vec3] = []
    normals: List[Vec3] = []
    indices: List[int] = []

    sign = -1.0 if direction == "left" else 1.0
    pivot_x = sign * radius / 2

    for i in range(segments + 1):
        angle = (math.pi / 2) * i / segments
        if direction == "left":
            x = pivot_x + radius * math.cos(angle)
            z = radius * math.sin(angle)
        else:
            x = pivot_x - radius * math.cos(angle)
            z = radius * math.sin(angle)

        vertices.append((x, y_low, z))
        vertices.append((x, y_high, z))
        normals.append((0, 1, 0))
        normals.append((0, 1, 0))

    for i in range(segments):
        base = i * 2
        indices.extend([base, base + 2, base + 3, base, base + 3, base + 1])

    return MeshPart(name="door_swing_arc", vertices=vertices, normals=normals, indices=indices, material_name="door_frame")


# ── Window ─────────────────────────────────────────────────────────


def generate_window_mesh(
    width: float = 1.20,
    height: float = 1.20,
    sill_height: float = 0.90,
    frame_depth: float = 0.08,
    frame_width: float = 0.05,
    glass_thickness: float = 0.006,
) -> List[MeshPart]:
    """Window: 4-piece frame + glass pane.

    Centered at midpoint, Y=0 at floor. Window sits at sill_height.
    """
    parts: List[MeshPart] = []
    hw = width / 2
    window_center_y = sill_height + height / 2

    # Four frame pieces
    # Left
    left = _box_mesh(frame_width, height, frame_depth, name="window_frame_left", material="window_frame")
    _offset_vertices(left, dx=-(hw - frame_width / 2), dy=window_center_y, dz=0)
    parts.append(left)

    # Right
    right = _box_mesh(frame_width, height, frame_depth, name="window_frame_right", material="window_frame")
    _offset_vertices(right, dx=(hw - frame_width / 2), dy=window_center_y, dz=0)
    parts.append(right)

    # Top
    top = _box_mesh(width, frame_width, frame_depth, name="window_frame_top", material="window_frame")
    _offset_vertices(top, dx=0, dy=sill_height + height - frame_width / 2, dz=0)
    parts.append(top)

    # Bottom (sill)
    bottom = _box_mesh(width, frame_width, frame_depth, name="window_frame_sill", material="window_frame")
    _offset_vertices(bottom, dx=0, dy=sill_height + frame_width / 2, dz=0)
    parts.append(bottom)

    # Glass pane
    glass_w = width - 2 * frame_width
    glass_h = height - 2 * frame_width
    glass = _box_mesh(glass_w, glass_h, glass_thickness, name="window_glass", material="window_glass")
    _offset_vertices(glass, dx=0, dy=window_center_y, dz=0)
    parts.append(glass)

    return parts


# ── Column ─────────────────────────────────────────────────────────


def generate_column_mesh(
    width: float = 0.30,
    depth: float = 0.30,
    height: float = 2.80,
) -> List[MeshPart]:
    """Column: simple box, centered at origin, Y=0 at floor."""
    col = _box_mesh(width, height, depth, name="column", material="column")
    _offset_vertices(col, dx=0, dy=height / 2, dz=0)
    return [col]


# ── Stairs ─────────────────────────────────────────────────────────


def generate_stairs_mesh(
    width: float = 1.00,
    height: float = 2.80,
    depth: float = 3.00,
    num_steps: int = 16,
) -> List[MeshPart]:
    """Stairs: stacked step boxes along Z axis.

    Each step rises by (height / num_steps) and runs (depth / num_steps).
    """
    parts: List[MeshPart] = []
    rise = height / num_steps
    run = depth / num_steps

    for i in range(num_steps):
        step = _box_mesh(width, rise, run, name=f"step_{i}", material="stairs")
        step_y = rise / 2 + i * rise
        step_z = -depth / 2 + run / 2 + i * run
        _offset_vertices(step, dx=0, dy=step_y, dz=step_z)
        parts.append(step)

    return parts


# ── Slab ───────────────────────────────────────────────────────────


def generate_slab_mesh(
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
    thickness: float = 0.20,
) -> List[MeshPart]:
    """Slab: flat box from bounding extents, sitting below Y=0."""
    sx = max_x - min_x
    sz = max_z - min_z
    cx = (min_x + max_x) / 2
    cz = (min_z + max_z) / 2

    slab = _box_mesh(sx, thickness, sz, name="slab", material="slab")
    _offset_vertices(slab, dx=cx, dy=-thickness / 2, dz=cz)
    return [slab]


# ── Vertex offset helper ──────────────────────────────────────────


def _offset_vertices(part: MeshPart, dx: float = 0, dy: float = 0, dz: float = 0) -> None:
    """Translate all vertices of a MeshPart in place."""
    part.vertices = [(v[0] + dx, v[1] + dy, v[2] + dz) for v in part.vertices]


# ── Ear-clipping triangulation ─────────────────────────────────────

def _ear_clip_triangulate(
    polygon: List[Tuple[float, float]],
) -> List[Tuple[int, int, int]]:
    """Triangulate a simple 2D polygon using ear-clipping.

    Handles both convex and concave (L/U-shaped) polygons correctly.
    Returns a list of (i, j, k) index triples into *polygon*.
    Centroid-fan fails for concave polygons; this algorithm does not.
    """
    n = len(polygon)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    def _cross2(o: Tuple[float, float],
                a: Tuple[float, float],
                b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _point_in_triangle(
        p: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float],
        c: Tuple[float, float],
    ) -> bool:
        d1 = _cross2(a, b, p)
        d2 = _cross2(b, c, p)
        d3 = _cross2(c, a, p)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    # Determine signed area to detect winding direction.
    area2 = sum(
        _cross2((0.0, 0.0), polygon[i], polygon[(i + 1) % n])
        for i in range(n)
    )
    cw = area2 < 0  # clockwise winding

    remaining = list(range(n))
    triangles: List[Tuple[int, int, int]] = []
    max_iters = n * n  # guard against degenerate inputs

    for _ in range(max_iters):
        m = len(remaining)
        if m < 3:
            break
        found_ear = False
        for i in range(m):
            pi = remaining[(i - 1) % m]
            ci = remaining[i]
            ni = remaining[(i + 1) % m]
            p, c, nxt = polygon[pi], polygon[ci], polygon[ni]

            # Convex vertex check (sign matches winding)
            cross = _cross2(p, c, nxt)
            if cw and cross > 0:
                continue
            if not cw and cross < 0:
                continue

            # Ear check: no other remaining vertex inside this triangle
            ear = True
            for j in remaining:
                if j in (pi, ci, ni):
                    continue
                if _point_in_triangle(polygon[j], p, c, nxt):
                    ear = False
                    break

            if ear:
                triangles.append((pi, ci, ni))
                remaining.pop(i)
                found_ear = True
                break

        if not found_ear:
            break  # degenerate polygon — stop

    if len(remaining) == 3:
        triangles.append(tuple(remaining))  # type: ignore[arg-type]

    return triangles


# ── Room floor ─────────────────────────────────────────────────────


_ROOM_TYPE_TO_MATERIAL: Dict[str, str] = {
    "living_room": "room_floor_living",
    "bedroom": "room_floor_living",
    "salon": "room_floor_living",
    "zimmer": "room_floor_living",
    "bathroom": "room_floor_bath",
    "wc": "room_floor_bath",
    "bad": "room_floor_bath",
    "toilet": "room_floor_bath",
    "kitchen": "room_floor_kitchen",
    "küche": "room_floor_kitchen",
}


def generate_room_floor_mesh(
    polygon: List[Tuple[float, float]],
    room_type: str = "generic",
    y_offset: float = 0.005,
) -> List[MeshPart]:
    """Fan-triangulate a 2D room polygon into a flat floor slab.

    Uses centroid-fan triangulation (works well for convex and near-convex rooms).
    The polygon must be in plan space (X, Y) — the glTF exporter maps Y→Z.
    The floor is placed at y_offset (slightly above the structural slab).
    """
    if len(polygon) < 3:
        return []

    # Look up material from room_type string (normalise and do prefix match)
    rt_lower = (room_type or "generic").lower()
    mat = "room_floor_generic"
    for key, mat_name in _ROOM_TYPE_TO_MATERIAL.items():
        if key in rt_lower:
            mat = mat_name
            break

    # Ear-clipping triangulation — safe for concave (L/U-shaped) rooms.
    # Centroid-fan triangulation produces triangles outside concave polygons,
    # overpainting adjacent spaces. Ear-clipping always stays inside the polygon.
    tri_indices = _ear_clip_triangulate(polygon)
    if not tri_indices:
        return []

    vertices: List[Vec3] = []
    normals: List[Vec3] = []
    indices: List[int] = []

    for (i, j, k) in tri_indices:
        base = len(vertices)
        for pt in (polygon[i], polygon[j], polygon[k]):
            # Plan X,Y → glTF X,Z (Y is up, floor at y_offset)
            vertices.append((pt[0], y_offset, pt[1]))
            normals.append((0.0, 1.0, 0.0))
        indices.extend([base, base + 1, base + 2])

    return [MeshPart(name="room_floor", vertices=vertices, normals=normals,
                     indices=indices, material_name=mat)]
