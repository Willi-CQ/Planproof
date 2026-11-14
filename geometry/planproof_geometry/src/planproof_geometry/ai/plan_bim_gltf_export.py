"""Export a SceneModel to glTF with parametric BIM meshes.

Replaces the old unit-cube approach (plan_gltf_walls.py) with proper per-element
mesh generation including normals, multi-part elements (door frame + panel),
and BIM-specific materials.
"""
from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from planproof_geometry.ai.plan_bim_meshes import (
    BIM_MATERIALS,
    MeshPart,
    WallOpening,
    generate_column_mesh,
    generate_door_mesh,
    generate_room_floor_mesh,
    generate_slab_mesh,
    generate_stairs_mesh,
    generate_wall_mesh,
    generate_wall_with_openings_mesh,
    generate_window_mesh,
)

try:
    from domain.scene_model import SceneModel
except ImportError:
    from planproof_geometry.domain.scene_model import SceneModel

Point = Tuple[float, float]


def _seg_yaw(a: Point, b: Point) -> float:
    """Yaw around glTF's +Y axis so local +X aligns with (b - a) in the
    mapped scene (x, z) plane.

    The mapper ``_map_xy`` flips plan-Y so scene-Z = -plan_Y. The wall-mesh
    renders local +X extended along its length; after Y-axis rotation by
    yaw, local +X lands at ``(cos yaw, 0, -sin yaw)`` (glTF RH convention).
    For local +X to point from mapped-a to mapped-b we need
        cos yaw = (bx-ax)/L,   -sin yaw = (by-ay)/L
    → yaw = atan2(-(by-ay), bx-ax).

    The old formula ``atan2(by-ay, bx-ax)`` was 180° off in the x-z plane,
    which for symmetric box meshes was invisible on axis-aligned walls but
    visibly mirrored any diagonal wall around its own midpoint and swapped
    the side openings land on for vertical walls.
    """
    return math.atan2(-(b[1] - a[1]), b[0] - a[0])


def _quat_from_yaw(yaw: float) -> List[float]:
    """Quaternion for rotation around Y axis."""
    s = math.sin(yaw * 0.5)
    c = math.cos(yaw * 0.5)
    return [0.0, s, 0.0, c]


# ── glTF binary buffer helpers ─────────────────────────────────────


def _pack_mesh_parts(parts_list: List[List[MeshPart]]) -> Tuple[bytes, List[Dict]]:
    """Pack all mesh parts into a single binary buffer.

    Returns (blob, mesh_infos) where mesh_infos contains byte offsets/counts
    for each mesh part.
    """
    blob = bytearray()
    mesh_infos: List[Dict] = []

    for parts in parts_list:
        part_infos = []
        for part in parts:
            vb_offset = len(blob)
            # Interleaved: POSITION (3f) + NORMAL (3f) = 24 bytes per vertex
            for v, n in zip(part.vertices, part.normals):
                blob.extend(struct.pack("<3f", *v))
                blob.extend(struct.pack("<3f", *n))
            vb_length = len(blob) - vb_offset

            ib_offset = len(blob)
            for idx in part.indices:
                blob.extend(struct.pack("<H", idx))
            ib_length = len(blob) - ib_offset

            # Pad to 4-byte alignment
            while len(blob) % 4 != 0:
                blob.extend(b"\x00")

            # Compute bounds
            xs = [v[0] for v in part.vertices]
            ys = [v[1] for v in part.vertices]
            zs = [v[2] for v in part.vertices]

            part_infos.append({
                "vb_offset": vb_offset,
                "vb_length": vb_length,
                "ib_offset": ib_offset,
                "ib_length": ib_length,
                "vertex_count": len(part.vertices),
                "index_count": len(part.indices),
                "material_name": part.material_name,
                "name": part.name,
                "min": [min(xs), min(ys), min(zs)],
                "max": [max(xs), max(ys), max(zs)],
            })
        mesh_infos.append(part_infos)

    return bytes(blob), mesh_infos


def export_scene_to_gltf(
    scene: SceneModel,
    out_path: str,
    wall_height_m: float = 2.80,
    default_wall_thickness_m: float = 0.12,
    plan_origin_xy: Optional[Tuple[float, float]] = None,
    flip_plan_y_for_pdf: bool = True,
    add_slab: bool = True,
) -> None:
    """Export SceneModel to a single glTF file with parametric BIM meshes.

    Each Component becomes a glTF node (or parent node with children for
    multi-part elements like doors).  Meshes are generated per-element with
    proper geometry and normals.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ox, oy = plan_origin_xy if plan_origin_xy is not None else (0.0, 0.0)

    def _map_xy(x: float, y: float) -> Tuple[float, float]:
        x2 = x - ox
        y2 = y - oy
        if flip_plan_y_for_pdf:
            y2 = -y2
        return x2, y2

    # ── Collect material list ──────────────────────────────────────
    material_names: List[str] = []
    material_index: Dict[str, int] = {}

    def _get_mat_idx(name: str) -> int:
        if name not in material_index:
            material_index[name] = len(material_names)
            material_names.append(name)
        return material_index[name]

    # ── Build wall→openings index from HAS_OPENING relationships ──
    wall_openings: Dict[str, List[str]] = {}  # wall_id → [opening_comp_id, ...]
    for rel in scene.relationships:
        if rel.rel_type == "HAS_OPENING":
            wall_openings.setdefault(rel.src_id, []).append(rel.dst_id)

    # ── Generate meshes per component ──────────────────────────────
    # Each entry: (component, parts, transform, geom_attrs)
    # geom_attrs carries pre-computed length/height/thickness for node extras.
    component_meshes: List[Tuple[Any, List[MeshPart], Dict, Dict]] = []

    # Track bounds for slab
    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    for comp in scene.components.values():
        if comp.kind == "ROOM":
            # Render rooms as coloured floor fills if polygon data is available
            polygon = comp.attrs.get("polygon")
            if polygon and len(polygon) >= 3:
                room_type = comp.attrs.get("room_type", "generic")
                # Apply the same coordinate transform used for walls/openings
                # (origin shift + optional Y-flip for PDF coordinate systems).
                polygon_mapped = [_map_xy(float(p[0]), float(p[1])) for p in polygon]
                parts = generate_room_floor_mesh(
                    polygon_mapped,
                    room_type=room_type,
                )
                if parts:
                    transform = {"translation": [0.0, 0.0, 0.0]}
                    geom_attrs = {"area_m2": comp.attrs.get("area_m2")}
                    geom_attrs = {k: v for k, v in geom_attrs.items() if v is not None}
                    component_meshes.append((comp, parts, transform, geom_attrs))
            continue

        attrs = comp.attrs
        a = attrs.get("segment_a")
        b = attrs.get("segment_b")
        if not a or not b:
            continue

        ax, ay = _map_xy(float(a[0]), float(a[1]))
        bx, by = _map_xy(float(b[0]), float(b[1]))

        minx = min(minx, ax, bx)
        miny = min(miny, ay, by)
        maxx = max(maxx, ax, bx)
        maxy = max(maxy, ay, by)

        length = math.hypot(bx - ax, by - ay)
        if length <= 1e-6:
            continue

        yaw = _seg_yaw((ax, ay), (bx, by))
        cx = (ax + bx) * 0.5
        cy = (ay + by) * 0.5

        thickness = float(attrs.get("thickness_m", default_wall_thickness_m))
        elem_height = float(attrs.get("height_m", wall_height_m))

        transform = {
            "translation": [cx, 0.0, cy],  # Y=0 at floor, meshes handle their own Y offset
            "rotation": _quat_from_yaw(yaw),
        }

        parts: List[MeshPart] = []

        if comp.kind == "WALL":
            wall_type = attrs.get("wall_type", "exterior")

            # Phase 4 — miter extensions so 90° corners render without
            # "W-zigzag" stubs at the junction. MiterWallJoinsStep has
            # written per-endpoint extensions into edge.attrs which
            # BuildSceneModel has propagated to comp.attrs. The wall
            # renders LONGER by (extend_a + extend_b) with midpoint
            # shifted by (extend_b - extend_a) / 2 along wall direction.
            # Opening offsets, computed below relative to the ORIGINAL
            # midpoint, need the same shift subtracted to stay fixed in
            # world space.
            miter_a = float(attrs.get("miter_extend_a_m", 0.0) or 0.0)
            miter_b = float(attrs.get("miter_extend_b_m", 0.0) or 0.0)
            rendered_length = length + miter_a + miter_b
            midpoint_shift = (miter_b - miter_a) * 0.5
            if abs(miter_a) > 1e-9 or abs(miter_b) > 1e-9:
                ux_world = (bx - ax) / length
                uy_world = (by - ay) / length
                cx_rendered = cx + ux_world * midpoint_shift
                cy_rendered = cy + uy_world * midpoint_shift
                transform["translation"] = [cx_rendered, 0.0, cy_rendered]
            else:
                cx_rendered = cx
                cy_rendered = cy

            # Collect openings that belong to this wall
            opening_ids = wall_openings.get(comp.id, [])
            wall_ops: List[WallOpening] = []
            for oid in opening_ids:
                ocomp = scene.components.get(oid)
                if not ocomp:
                    continue
                # Surface-mounted openings (sliding doors, etc.) run IN
                # FRONT of the wall on a rail — they must not subtract
                # an opening. Their own DOOR/WINDOW mesh is still drawn.
                if ocomp.attrs.get("cuts_wall") is False:
                    continue
                if ocomp.attrs.get("surface_mount") is True:
                    continue
                oa = ocomp.attrs.get("segment_a")
                ob = ocomp.attrs.get("segment_b")
                if not oa or not ob:
                    continue
                # Project opening midpoint onto wall axis to get offset
                oax_m, oay_m = _map_xy(float(oa[0]), float(oa[1]))
                obx_m, oby_m = _map_xy(float(ob[0]), float(ob[1]))
                omx = (oax_m + obx_m) / 2
                omy = (oay_m + oby_m) / 2
                # Vector along wall
                dx, dy = bx - ax, by - ay
                # Project onto wall axis: t = dot(OM - A, AB) / |AB|^2
                t = ((omx - ax) * dx + (omy - ay) * dy) / (length * length)
                offset_along_wall = (t - 0.5) * length  # relative to wall center
                o_width = math.hypot(obx_m - oax_m, oby_m - oay_m)
                if o_width < 0.05:
                    o_width = float(ocomp.attrs.get("width_m", 0.90))
                o_bottom = 0.0
                o_top = float(ocomp.attrs.get("height_m", elem_height * 0.75))
                if ocomp.kind == "WINDOW":
                    o_bottom = float(ocomp.attrs.get("sill_height_m", 0.90))
                    o_top = o_bottom + float(ocomp.attrs.get("height_m", 1.20))
                elif ocomp.kind == "DOOR":
                    o_top = float(ocomp.attrs.get("height_m", 2.10))
                # Phase 4 compensation: opening offset was computed
                # relative to the ORIGINAL wall midpoint; after miter
                # the midpoint shifts along the wall by midpoint_shift.
                # Subtract to keep the opening fixed in world space.
                wall_ops.append(WallOpening(
                    offset=offset_along_wall - midpoint_shift,
                    width=o_width,
                    bottom=o_bottom,
                    top=min(o_top, elem_height),
                ))

            if wall_ops:
                parts = generate_wall_with_openings_mesh(
                    rendered_length, elem_height, thickness, wall_ops, wall_type=wall_type,
                )
            else:
                parts = generate_wall_mesh(rendered_length, elem_height, thickness, wall_type=wall_type)
            # Wall meshes are centered at origin — offset Y to center vertically
            for p in parts:
                p.vertices = [(v[0], v[1] + elem_height / 2, v[2]) for v in p.vertices]

        elif comp.kind == "DOOR":
            door_w = float(attrs.get("width_m", length))
            door_h = float(attrs.get("height_m", elem_height))
            swing = attrs.get("swing_direction")
            # Phase E: multi-leaf / sliding variants from manual labels
            leaves = int(attrs.get("leaves", 1))
            sliding = bool(attrs.get("sliding", False))
            parts = generate_door_mesh(
                width=door_w,
                height=door_h,
                swing_direction=swing,
                leaves=leaves,
                sliding=sliding,
            )
            # Sliding (surface-mount) doors: push the panel outward by
            # wall_thickness/2 + 2 cm so it sits flush on the wall face
            # instead of centred inside it.
            if sliding or attrs.get("cuts_wall") is False:
                wall_thick = float(attrs.get("thickness_m", 0.12))
                push = wall_thick * 0.5 + 0.02
                L_sub = math.hypot(bx - ax, by - ay)
                if L_sub > 1e-6:
                    perp_x = -(by - ay) / L_sub
                    perp_z = (bx - ax) / L_sub
                    transform["translation"] = [
                        cx + perp_x * push, 0.0, cy + perp_z * push,
                    ]
            # Door meshes already have Y=0 at floor

        elif comp.kind == "WINDOW":
            win_w = float(attrs.get("width_m", length))
            win_h = float(attrs.get("height_m", elem_height))
            sill_h = float(attrs.get("sill_height_m", 0.90))
            parts = generate_window_mesh(
                width=win_w,
                height=win_h,
                sill_height=sill_h,
            )
            # Window meshes already positioned at sill_height

        elif comp.kind == "COLUMN":
            col_w = float(attrs.get("width_m", thickness))
            col_d = float(attrs.get("depth_m", thickness))
            parts = generate_column_mesh(width=col_w, depth=col_d, height=elem_height)
            # Column meshes already have Y=0 at floor

        elif comp.kind == "STAIRS":
            stair_w = float(attrs.get("width_m", length))
            stair_d = float(attrs.get("depth_m", length))
            num_steps = int(attrs.get("num_steps", 16))
            parts = generate_stairs_mesh(
                width=stair_w,
                height=elem_height,
                depth=stair_d,
                num_steps=num_steps,
            )

        if parts:
            geom_attrs = {
                "length_m": round(length, 3),
                "height_m": round(elem_height, 2),
                "thickness_m": round(thickness, 3),
                "area_m2": round(length * elem_height, 2),
                "volume_m3": round(length * elem_height * thickness, 3),
            }
            component_meshes.append((comp, parts, transform, geom_attrs))

    # ── Slab ───────────────────────────────────────────────────────
    slab_parts: Optional[List[MeshPart]] = None
    if add_slab and minx < maxx and miny < maxy:
        margin = default_wall_thickness_m
        slab_parts = generate_slab_mesh(
            minx - margin, maxx + margin,
            miny - margin, maxy + margin,
            thickness=0.20,
        )

    # ── Build glTF ─────────────────────────────────────────────────
    # Collect all mesh part lists for binary packing
    all_parts_lists: List[List[MeshPart]] = []
    for _, parts, _, _ in component_meshes:
        all_parts_lists.append(parts)
    if slab_parts:
        all_parts_lists.append(slab_parts)

    if not all_parts_lists:
        # Empty scene — write minimal valid glTF
        out.write_text(json.dumps({
            "asset": {"version": "2.0", "generator": "Planproof Plan-to-BIM"},
            "scenes": [{"nodes": []}],
            "scene": 0,
            "nodes": [],
        }), encoding="utf-8")
        return

    blob, mesh_infos_list = _pack_mesh_parts(all_parts_lists)
    uri = "data:application/octet-stream;base64," + base64.b64encode(blob).decode("ascii")

    buffers = [{"uri": uri, "byteLength": len(blob)}]
    buffer_views: List[Dict] = []
    accessors: List[Dict] = []
    meshes: List[Dict] = []
    nodes: List[Dict] = []
    scene_nodes: List[int] = []

    # Ensure all materials are registered
    for parts_infos in mesh_infos_list:
        for pi in parts_infos:
            _get_mat_idx(pi["material_name"])

    # Build buffer views, accessors, and meshes
    mesh_idx = 0
    for parts_infos in mesh_infos_list:
        primitives = []
        for pi in parts_infos:
            # Buffer view for interleaved vertex data (POSITION + NORMAL)
            vb_view_idx = len(buffer_views)
            buffer_views.append({
                "buffer": 0,
                "byteOffset": pi["vb_offset"],
                "byteLength": pi["vb_length"],
                "byteStride": 24,  # 3f position + 3f normal
                "target": 34962,   # ARRAY_BUFFER
            })

            # Buffer view for indices
            ib_view_idx = len(buffer_views)
            buffer_views.append({
                "buffer": 0,
                "byteOffset": pi["ib_offset"],
                "byteLength": pi["ib_length"],
                "target": 34963,   # ELEMENT_ARRAY_BUFFER
            })

            # Accessor for POSITION (offset 0 in stride)
            pos_acc_idx = len(accessors)
            accessors.append({
                "bufferView": vb_view_idx,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": pi["vertex_count"],
                "type": "VEC3",
                "min": pi["min"],
                "max": pi["max"],
            })

            # Accessor for NORMAL (offset 12 in stride)
            norm_acc_idx = len(accessors)
            accessors.append({
                "bufferView": vb_view_idx,
                "byteOffset": 12,
                "componentType": 5126,  # FLOAT
                "count": pi["vertex_count"],
                "type": "VEC3",
            })

            # Accessor for indices
            idx_acc_idx = len(accessors)
            accessors.append({
                "bufferView": ib_view_idx,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": pi["index_count"],
                "type": "SCALAR",
            })

            mat_idx = _get_mat_idx(pi["material_name"])
            primitives.append({
                "attributes": {"POSITION": pos_acc_idx, "NORMAL": norm_acc_idx},
                "indices": idx_acc_idx,
                "material": mat_idx,
            })

        # Mesh name set in the node-building loop below (we need comp.id)
        meshes.append({"primitives": primitives})
        mesh_idx += 1

    _KIND_TO_IFC = {
        "WALL": "IfcWall", "DOOR": "IfcDoor", "WINDOW": "IfcWindow",
        "COLUMN": "IfcColumn", "STAIRS": "IfcStair", "SLAB": "IfcSlab",
        "ROOM": "IfcSpace",
    }

    # Build nodes for components
    for i, (comp, parts, transform, geom_attrs) in enumerate(component_meshes):
        # Also set the mesh name now that we have comp available
        meshes[i]["name"] = f"{comp.kind.lower()}/{comp.id}"

        attrs = comp.attrs
        extras: Dict[str, Any] = {
            "seg_id": comp.id,
            "ifc_type": _KIND_TO_IFC.get(comp.kind, "IfcBuildingElement"),
            "kind": comp.kind.lower(),
            "class": comp.kind.lower(),
            "layer": "elements",
            # Geometry (pre-computed in mesh generation loop)
            **geom_attrs,
            # Classification provenance
            "confidence": round(comp.evidence[0].confidence, 2) if comp.evidence else None,
            "source": attrs.get("source"),
        }
        # Remove None values to keep extras clean
        extras = {k: v for k, v in extras.items() if v is not None}
        # Copy additional BIM properties for viewer access
        for key in ("wall_type", "is_loadbearing", "material_hint", "swing_direction",
                     "sill_height_m", "llm_reasoning",
                     # Scan-/Raum-Durchstich: Raumzugehoerigkeit + echte Kontur
                     # bis in den Indexer (Room-Upsert + raumweiser Vergleich).
                     "room_key", "polygon"):
            if key in attrs:
                extras[key] = attrs[key]

        node: Dict[str, Any] = {
            "name": f"{comp.kind.lower()}/{comp.id}",
            "mesh": i,
            "extras": extras,
        }
        if transform.get("translation"):
            node["translation"] = transform["translation"]
        if transform.get("rotation"):
            node["rotation"] = transform["rotation"]

        nodes.append(node)
        scene_nodes.append(len(nodes) - 1)

    # Slab node
    if slab_parts:
        slab_mesh_idx = len(component_meshes)
        nodes.append({
            "name": "slab/floor",
            "mesh": slab_mesh_idx,
            "extras": {"kind": "slab", "layer": "structure"},
        })
        scene_nodes.append(len(nodes) - 1)

    # Build materials
    gltf_materials: List[Dict] = []
    for mat_name in material_names:
        mat_def = BIM_MATERIALS.get(mat_name, BIM_MATERIALS["wall_exterior"])
        mat: Dict[str, Any] = {
            "name": mat_name,
            "pbrMetallicRoughness": {
                "baseColorFactor": mat_def["baseColorFactor"],
                "metallicFactor": mat_def.get("metallicFactor", 0.0),
                "roughnessFactor": mat_def.get("roughnessFactor", 0.9),
            },
        }
        if mat_def.get("alphaMode") == "BLEND":
            mat["alphaMode"] = "BLEND"
            mat["doubleSided"] = True  # transparent surfaces visible from both sides
        gltf_materials.append(mat)

    # Assemble glTF
    gltf: Dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "Planproof Plan-to-BIM v2"},
        "buffers": buffers,
        "bufferViews": buffer_views,
        "accessors": accessors,
        "materials": gltf_materials,
        "meshes": meshes,
        "nodes": nodes,
        "scenes": [{"nodes": scene_nodes}],
        "scene": 0,
    }

    out.write_text(json.dumps(gltf), encoding="utf-8")
