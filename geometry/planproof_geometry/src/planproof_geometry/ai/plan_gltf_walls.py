from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path
from typing import Dict, List, Tuple

Point = Tuple[float, float]


def _quat_from_yaw(yaw: float) -> List[float]:
    # rotation around Y (glTF default: Y-up)
    s = math.sin(yaw * 0.5)
    c = math.cos(yaw * 0.5)
    return [0.0, s, 0.0, c]


def _seg_len(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _seg_yaw(a: Point, b: Point) -> float:
    """Yaw around glTF's +Y axis so local +X aligns with (b-a) in the
    mapped scene plane. See plan_bim_gltf_export._seg_yaw for the
    derivation; the previous ``atan2(dy, dx)`` formula was 180° off for
    diagonal walls, which showed up as the wall being mirrored around
    its own midpoint.
    """
    return math.atan2(-(b[1] - a[1]), b[0] - a[0])


# Segment class -> (thickness_m, height_ratio, material_index)
# material_index: 0=walls, 1=doors, 2=windows, 3=stairs, 4=columns
# 7-class schema; wall_type metadata distinguishes exterior (thicker) from interior
_CLASS_CONFIG: Dict[str, tuple] = {
    "wall":   (0.10, 1.0, 0),
    "door":   (0.03, 0.9, 1),
    "window": (0.02, 0.6, 2),
    "stairs": (0.02, 0.5, 3),
    "column": (0.15, 1.0, 4),
}


def _get_segment_config(kind: str, wall_thickness_m: float, meta: Dict | None = None) -> tuple:
    """Return (thickness, height_ratio, material_index) for segment kind.

    For walls, uses meta["wall_type"] to distinguish exterior (12 cm) from interior (8 cm).
    """
    k = (kind or "raw_line").lower().strip()
    if k in _CLASS_CONFIG:
        thick, hr, mat = _CLASS_CONFIG[k]
        if k == "wall" and meta:
            wt = (meta or {}).get("wall_type", "")
            if wt == "exterior":
                thick = 0.12
            elif wt == "interior":
                thick = 0.08
        return (thick, hr, mat)
    return (wall_thickness_m, 1.0, 0)


def write_walls_gltf(
    out_path: str,
    wall_segments: List[Dict],
    wall_height_m: float = 3.00,
    wall_thickness_m: float = 0.12,
    add_slab: bool = False,
    slab_thickness_m: float = 0.05,

    # keep these for backward compatibility (not used by your current target workflow)
    add_plan_plane: bool = False,
    plan_plane_size_m: tuple[float, float] | None = None,
    plan_texture_uri: str | None = None,

    # NEW: 2D overlay for scan alignment
    overlay_segments_2d: List[Dict] | None = None,
    overlay_thickness_m: float = 0.01,
    overlay_height_m: float = 0.002,

    plan_origin_xy: tuple[float, float] | None =None,
    flip_plan_y_for_pdf: bool = False,
) -> None:
    """
    Writes a glTF with one unit cube mesh and many nodes scaled/rotated/translated.
    - wall_segments become extruded 3D elements (walls, doors, windows, stairs, columns)
    - Each segment's "kind" or "class" determines thickness, height, and material
    - overlay_segments_2d become thin extrusions on the floor for 2D alignment preview
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Plan -> glTF coordinate mapping
    # -----------------------------
    # Planraum: (x, y). glTF: (x, z) auf dem Boden, y ist up.
    # plan_origin_xy: zentriert den Plan (Kamera/Viewer)
    # flip_plan_y_for_pdf: spiegelt Plan-y, damit es zur PDF-Ansicht passt (kein Spiegelverkehrt mehr)
    ox, oy = (plan_origin_xy if plan_origin_xy is not None else (0.0, 0.0))

    def _map_xy(x: float, y: float) -> tuple[float, float]:
        x2 = x - ox
        y2 = y - oy
        if flip_plan_y_for_pdf:
            y2 = -y2
        return x2, y2

    # unit cube geometry (centered at origin)
    verts = [
        (-0.5, -0.5, -0.5),
        ( 0.5, -0.5, -0.5),
        ( 0.5,  0.5, -0.5),
        (-0.5,  0.5, -0.5),
        (-0.5, -0.5,  0.5),
        ( 0.5, -0.5,  0.5),
        ( 0.5,  0.5,  0.5),
        (-0.5,  0.5,  0.5),
    ]
    idx = [
        0, 1, 2,  0, 2, 3,
        4, 5, 6,  4, 6, 7,
        0, 4, 7,  0, 7, 3,
        1, 5, 6,  1, 6, 2,
        3, 2, 6,  3, 6, 7,
        0, 1, 5,  0, 5, 4,
    ]

    vb = b"".join(struct.pack("<3f", *v) for v in verts)
    ib = b"".join(struct.pack("<H", i) for i in idx)
    blob = vb + ib
    uri0 = "data:application/octet-stream;base64," + base64.b64encode(blob).decode("ascii")

    nodes: List[Dict] = []
    scene_nodes: List[int] = []

    # bounds for optional slab (in plan coords)
    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    # -------------------------
    # 3D walls (extruded)
    # Plan coords: (x,y) -> glTF: (x,z), height along Y
    # -------------------------
    for s in wall_segments or []:
        a = s.get("a")
        b = s.get("b")
        if not a or not b:
            continue

        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])

        ax, ay = _map_xy(ax, ay)
        bx, by = _map_xy(bx, by)

        minx = min(minx, ax, bx)
        miny = min(miny, ay, by)
        maxx = max(maxx, ax, bx)
        maxy = max(maxy, ay, by)

        L = _seg_len((ax, ay), (bx, by))
        if L <= 1e-6:
            continue

        kind = s.get("kind", s.get("class", s.get("type", "raw_line")))
        seg_meta = s.get("meta") or s.get("extras") or {}
        thick, height_ratio, mat_idx = _get_segment_config(kind, wall_thickness_m, seg_meta)
        elem_height = wall_height_m * height_ratio

        yaw = _seg_yaw((ax, ay), (bx, by))
        q = _quat_from_yaw(yaw)
        cx, cy = (ax + bx) * 0.5, (ay + by) * 0.5

        seg_id = str(s.get("id", f"g{len(nodes)}"))
        node = {
            "name": f"{kind}/{seg_id}",
            "mesh": mat_idx,
            "translation": [cx, elem_height * 0.5, cy],
            "rotation": q,
            "scale": [L, elem_height, thick],
            "extras": {
                "seg_id": seg_id,
                "source_geometry_id": seg_id,
                "kind": kind,
                "class": kind,
                "layer": "elements",
                "page": int(s.get("page", 0)),
                "is_dimension_line": bool(s.get("is_dimension_line", False)),
                "is_dimension_mainline": bool(s.get("is_dimension_mainline", False)),
                "is_dimension_extension": bool(s.get("is_dimension_extension", False)),
            },
        }

        nodes.append(node)

        scene_nodes.append(len(nodes) - 1)

    # -------------------------
    # 2D overlay (thin extrusions on floor)
    # -------------------------
    if overlay_segments_2d:
        for s in overlay_segments_2d:
            a = s.get("a")
            b = s.get("b")
            if not a or not b:
                continue

            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])

            ax, ay = _map_xy(ax, ay)
            bx, by = _map_xy(bx, by)

            L2 = _seg_len((ax, ay), (bx, by))
            if L2 <= 1e-6:
                continue

            yaw2 = _seg_yaw((ax, ay), (bx, by))
            q2 = _quat_from_yaw(yaw2)
            cx2, cy2 = (ax + bx) * 0.5, (ay + by) * 0.5

            seg_id = s.get("id") or s.get("source_geometry_id") or None

            node2 = {
                "mesh": 0,
                "name": f"overlay2d/{seg_id}" if seg_id else "overlay2d/line",
                "extras": {
                    "seg_id": seg_id,
                    "source_geometry_id": seg_id,
                    "kind": s.get("kind", "raw_line"),   # später: "dimension", "door", ...
                    "layer": "overlay_2d",
                    "page": int(s.get("page", 0)),
                    # optional: bereits bekannte Heuristik-Flags weitergeben
                    "is_dimension_line": bool(s.get("is_dimension_line", False)),
                    "is_dimension_mainline": bool(s.get("is_dimension_mainline", False)),
                    "is_dimension_extension": bool(s.get("is_dimension_extension", False)),
                },
                "translation": [cx2, overlay_height_m * 0.5, cy2],
                "rotation": q2,
                "scale": [L2, overlay_height_m, overlay_thickness_m],
            }

            nodes.append(node2)

            scene_nodes.append(len(nodes) - 1)

    # -------------------------
    # optional slab (floor) from bounds
    # -------------------------
    if add_slab and minx < maxx and miny < maxy:
        cx = (minx + maxx) * 0.5
        cy = (miny + maxy) * 0.5
        sx = (maxx - minx) + 2.0 * wall_thickness_m
        sy = (maxy - miny) + 2.0 * wall_thickness_m

        slab_node = {
            "mesh": 0,
            "translation": [cx, slab_thickness_m * 0.5, cy],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [sx, slab_thickness_m, sy],  # XZ floor size, Y thickness
        }
        nodes.append(slab_node)
        scene_nodes.append(len(nodes) - 1)

    # minimal glTF (single buffer: cube geometry; nodes place scaled cubes)
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": uri0, "byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(vb), "target": 34962},
            {"buffer": 0, "byteOffset": len(vb), "byteLength": len(ib), "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(verts),
                "type": "VEC3",
                "min": [-0.5, -0.5, -0.5],
                "max": [0.5, 0.5, 0.5],
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": len(idx),
                "type": "SCALAR",
            },
        ],
        "materials": [
            {"name": "wall", "pbrMetallicRoughness": {"baseColorFactor": [0.7, 0.7, 0.7, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}},
            {"name": "door", "pbrMetallicRoughness": {"baseColorFactor": [0.55, 0.35, 0.2, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}},
            {"name": "window", "pbrMetallicRoughness": {"baseColorFactor": [0.6, 0.8, 0.95, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.3}},
            {"name": "stairs", "pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.5, 0.5, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}},
            {"name": "column", "pbrMetallicRoughness": {"baseColorFactor": [0.4, 0.4, 0.45, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}},
        ],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]},
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 1}]},
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 2}]},
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 3}]},
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 4}]},
        ],
        "nodes": nodes,
        "scenes": [{"nodes": scene_nodes}],
        "scene": 0,
    }

    out.write_text(json.dumps(gltf), encoding="utf-8")
