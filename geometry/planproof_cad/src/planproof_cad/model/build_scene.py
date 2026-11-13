"""Build a SceneModel from plan-to-BIM semantics_dict.

Converts the flat segment list produced by SemanticFusion + ComputeOpenings +
ClassifyWallType + FilterGeometry into a structured SceneModel with typed
Components, Relationships, and Evidence chains.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from planproof_cad.model.scene_model import Component, Evidence, Relationship, SceneModel

# Segment class -> Component kind mapping
_KIND_MAP = {
    "wall": "WALL",
    "door": "DOOR",
    "window": "WINDOW",
    "stairs": "STAIRS",
    "column": "COLUMN",
    "raw_line": "WALL",
}

# Default thickness per class (meters)
_DEFAULT_THICKNESS = {
    "wall": 0.10,
    "door": 0.03,
    "window": 0.02,
    "stairs": 0.02,
    "column": 0.15,
}

# Height ratio per class (relative to wall_height_m)
_HEIGHT_RATIO = {
    "wall": 1.0,
    "door": 0.9,
    "window": 0.6,
    "stairs": 0.5,
    "column": 1.0,
}


def _seg_length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def build_scene_from_semantics(
    semantics_dict: Dict[str, Any],
    wall_height_m: float = 2.80,
    wall_thickness_m: float = 0.12,
    wall_thickness_exterior_m: Optional[float] = None,
    wall_thickness_interior_m: Optional[float] = None,
) -> SceneModel:
    """Convert a semantics_dict into a SceneModel.

    Parameters
    ----------
    semantics_dict : dict
        Output of SemanticFusionResult.to_dict() after all post-processing.
    wall_height_m : float
        Default wall height for extrusion.
    wall_thickness_m : float
        Fallback wall thickness.

    Returns
    -------
    SceneModel with components for every segment and relationships.
    """
    scene = SceneModel(version="scene_v1", units="m", up_axis="Y")
    segments = semantics_dict.get("segments", [])
    elements = semantics_dict.get("elements", [])
    rooms = semantics_dict.get("rooms", [])

    # Build element lookup: segment_id -> element info
    seg_to_element: Dict[str, Dict] = {}
    for elem in elements:
        for sid in elem.get("segment_ids", []):
            seg_to_element[sid] = elem

    # ---------- Components from segments ----------
    for seg in segments:
        seg_id = seg.get("id", "")
        a = seg.get("a", [0, 0])
        b = seg.get("b", [0, 0])
        cls = seg.get("class", "raw_line")
        confidence = float(seg.get("confidence", 0.0))
        source = seg.get("source", "heuristic")
        meta = seg.get("meta", {})

        kind = _KIND_MAP.get(cls, "WALL")
        length = _seg_length(tuple(a), tuple(b))

        # Determine thickness - prefer measured thickness from wall reconstruction,
        # fall back to exterior/interior defaults.
        measured_t = meta.get("measured_thickness_m")
        if measured_t is not None and 0.04 <= float(measured_t) <= 0.60:
            thickness = float(measured_t)
        else:
            thickness = _DEFAULT_THICKNESS.get(cls, wall_thickness_m)
            if cls == "wall":
                wt = meta.get("wall_type", "")
                if wt == "exterior":
                    thickness = (
                        wall_thickness_exterior_m if wall_thickness_exterior_m is not None else 0.12
                    )
                elif wt == "interior":
                    thickness = (
                        wall_thickness_interior_m if wall_thickness_interior_m is not None else 0.08
                    )

        # Determine height
        height_ratio = _HEIGHT_RATIO.get(cls, 1.0)
        elem_height = wall_height_m * height_ratio

        # Center in glTF coordinates: plan (x, y) -> glTF (x, z), Y = up
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        cx = (ax + bx) * 0.5
        cz = (ay + by) * 0.5

        # Attributes
        attrs: Dict[str, Any] = {
            "segment_a": [ax, ay],
            "segment_b": [bx, by],
            "length_m": round(length, 3),
            "thickness_m": thickness,
            "height_m": elem_height,
            "source": source,
        }
        if meta.get("wall_type"):
            attrs["wall_type"] = meta["wall_type"]
        if meta.get("detection_id"):
            attrs["detection_id"] = meta["detection_id"]

        # Phase E: opening subtype → leaf count / sliding flag
        # Drives the glTF exporter's per-opening geometry (door_double
        # becomes two leaves meeting mid-opening, door_sliding becomes
        # a thin slot with no visible leaf).
        subtype = meta.get("subtype") or meta.get("manual_class_raw")
        if subtype and kind in ("DOOR", "WINDOW"):
            attrs["subtype"] = subtype
            if subtype == "door_double":
                attrs["leaves"] = 2
            elif subtype == "door_sliding":
                attrs["leaves"] = 0  # slides - no visible leaves
                attrs["sliding"] = True
            elif subtype == "window_large":
                attrs["window_variant"] = "band"
            else:
                attrs["leaves"] = 1

        scene.add_component(
            Component(
                id=seg_id,
                name=f"{kind.title()} {seg_id}",
                kind=kind,
                geometry_ref=f"{cls}/{seg_id}",
                center=(cx, elem_height / 2, cz),
                dims=(length, elem_height, thickness),
                attrs=attrs,
                evidence=[
                    Evidence(
                        source="plan_to_bim",
                        method=f"{source}_classification",
                        confidence=confidence,
                    )
                ],
            )
        )

    # ---------- Relationships ----------
    # HAS_OPENING: link door/window segments to their parent wall segment.
    # Strategy: for each opening component, find the nearest wall component
    # whose endpoints are within snap distance of the opening endpoints.
    opening_kinds = {"DOOR", "WINDOW"}
    wall_comps = [c for c in scene.components.values() if c.kind == "WALL"]
    snap_m = 0.15

    for comp in scene.components.values():
        if comp.kind not in opening_kinds:
            continue
        oa = comp.attrs.get("segment_a")
        ob = comp.attrs.get("segment_b")
        if not oa or not ob:
            continue
        oax, oay = float(oa[0]), float(oa[1])
        obx, oby = float(ob[0]), float(ob[1])
        omx, omy = (oax + obx) / 2, (oay + oby) / 2
        snap2 = snap_m * snap_m

        best_wall = None
        best_dist = float("inf")

        for wc in wall_comps:
            wa = wc.attrs.get("segment_a")
            wb = wc.attrs.get("segment_b")
            if not wa or not wb:
                continue
            wax, way = float(wa[0]), float(wa[1])
            wbx, wby = float(wb[0]), float(wb[1])

            # Check if any opening endpoint is near any wall endpoint
            endpoint_near = False
            for px, py in [(oax, oay), (obx, oby)]:
                for qx, qy in [(wax, way), (wbx, wby)]:
                    if (px - qx) ** 2 + (py - qy) ** 2 < snap2:
                        endpoint_near = True
                        break
                if endpoint_near:
                    break

            if not endpoint_near:
                # Also try: opening midpoint projected onto wall is close
                wmx, wmy = (wax + wbx) / 2, (way + wby) / 2
                wlen = math.hypot(wbx - wax, wby - way)
                if wlen > 0.01:
                    dx, dy = wbx - wax, wby - way
                    t = ((omx - wax) * dx + (omy - way) * dy) / (wlen * wlen)
                    if -0.2 < t < 1.2:
                        proj_x = wax + t * dx
                        proj_y = way + t * dy
                        dist_to_line = math.hypot(omx - proj_x, omy - proj_y)
                        if dist_to_line < snap_m:
                            endpoint_near = True

            if endpoint_near:
                wmx, wmy = (wax + wbx) / 2, (way + wby) / 2
                d = math.hypot(omx - wmx, omy - wmy)
                if d < best_dist:
                    best_dist = d
                    best_wall = wc

        if best_wall is not None:
            scene.add_relationship(
                Relationship(
                    src_id=best_wall.id,
                    rel_type="HAS_OPENING",
                    dst_id=comp.id,
                    confidence=float(comp.evidence[0].confidence if comp.evidence else 0.5),
                )
            )

    # ---------- Room components ----------
    for room in rooms:
        room_id = room.get("id", "")
        label = room.get("label")
        area = room.get("area_m2")
        center = room.get("center", [0, 0])
        polygon = room.get("polygon", [])

        room_attrs: Dict[str, Any] = {}
        if label:
            room_attrs["label"] = label
        if area is not None:
            room_attrs["area_m2"] = area
        if polygon:
            room_attrs["polygon"] = polygon
        room_meta = room.get("meta", {})
        if room_meta.get("room_type"):
            room_attrs["room_type"] = room_meta["room_type"]

        scene.add_component(
            Component(
                id=room_id,
                name=label or f"Room {room_id}",
                kind="ROOM",
                center=(float(center[0]), wall_height_m / 2, float(center[1])),
                dims=None,
                attrs=room_attrs,
                evidence=[
                    Evidence(
                        source="plan_to_bim",
                        method="room_detection",
                        confidence=0.6,
                    )
                ],
            )
        )

        # Link bounding segments to room
        for seg_id in room.get("bounding_segments", []):
            if seg_id in scene.components:
                scene.add_relationship(
                    Relationship(
                        src_id=seg_id,
                        rel_type="BOUNDS",
                        dst_id=room_id,
                    )
                )

    # ---------- Scene meta ----------
    stats = semantics_dict.get("stats", {})
    scene.meta["segments_by_class"] = stats.get("segments_by_class", {})
    scene.meta["total_segments"] = stats.get("total_segments", len(segments))
    scene.meta["total_elements"] = stats.get("total_elements", len(elements))
    scene.meta["total_rooms"] = stats.get("total_rooms", len(rooms))
    scene.meta["wall_height_m"] = wall_height_m
    scene.meta["wall_thickness_m"] = wall_thickness_m

    return scene


def build_scene_from_wall_graph(
    graph: Any,
    rooms_list: Optional[list] = None,
    wall_height_m: float = 2.80,
) -> SceneModel:
    """Graph-Refactor Day 5 - produce a SceneModel directly from a WallGraph.

    The graph already holds the correct topology (edges + parametric
    openings). This function is a deterministic 1:1 projection:

      * WallEdge                  -> Component(kind="WALL")
      * ParametricOpening         -> Component(kind="DOOR"|"WINDOW")
      * edge.openings membership  -> Relationship("HAS_OPENING")

    The glTF exporter downstream reads per-component ``segment_a`` /
    ``segment_b`` / ``length_m`` / ``thickness_m`` / ``height_m`` /
    ``sill_height_m`` attrs, so we populate those from the graph's
    parametric form (center + edge_direction * half_width). No
    geometric search, no nearest-wall lookup - the parent-child link
    is explicit.
    """
    scene = SceneModel(version="scene_v1", units="m", up_axis="Y")

    # Walls
    for edge in graph.edges.values():
        a, b = graph.edge_endpoints(edge.id)
        length = graph.edge_length(edge.id)
        cx = (a[0] + b[0]) * 0.5
        cz = (a[1] + b[1]) * 0.5
        height = edge.height_m or wall_height_m
        attrs: Dict[str, Any] = {
            "segment_a": [a[0], a[1]],
            "segment_b": [b[0], b[1]],
            "length_m": round(length, 3),
            "thickness_m": edge.thickness_m,
            "height_m": height,
            "source": edge.source,
            "wall_type": edge.wall_type,
            "graph_edge_id": edge.id,
            # tragend / nichttragend laut Layername, sonst None (nur dann darf die
            # Kostengruppen-Zuordnung 331/332 bzw. 341/342 verfeinern)
            "bearing": edge.attrs.get("bearing"),
        }
        if edge.attrs.get("layer"):
            attrs["layer"] = edge.attrs["layer"]
        # Phase 4: miter extensions for clean 90° corner rendering.
        # Read straight off edge.attrs so MiterWallJoinsStep's output
        # flows transparently into glTF rendering without a second
        # pipeline hop.
        miter_a = float(edge.attrs.get("miter_extend_a_m", 0.0) or 0.0)
        miter_b = float(edge.attrs.get("miter_extend_b_m", 0.0) or 0.0)
        if miter_a > 0.0:
            attrs["miter_extend_a_m"] = miter_a
        if miter_b > 0.0:
            attrs["miter_extend_b_m"] = miter_b
        scene.add_component(
            Component(
                id=edge.id,
                name=f"Wall {edge.id}",
                kind="WALL",
                geometry_ref=f"wall/{edge.id}",
                center=(cx, height / 2, cz),
                dims=(length, height, edge.thickness_m),
                attrs=attrs,
                evidence=[
                    Evidence(
                        source=edge.source,
                        method="wall_graph_edge",
                        confidence=edge.confidence,
                    )
                ],
            )
        )

    # Openings (per edge, preserving attach order)
    for edge in graph.edges.values():
        a, b = graph.edge_endpoints(edge.id)
        length = graph.edge_length(edge.id)
        if length < 1e-6:
            continue
        ux = (b[0] - a[0]) / length
        uy = (b[1] - a[1]) / length
        for op in edge.openings:
            half = op.width_m * 0.5
            # DO NOT clamp off_lo/off_hi to the edge range - the door/window
            # mesh reads width_m directly, but the WALL hole is sized by the
            # seg_a→seg_b distance. Clamping shrinks the hole without
            # shrinking the mesh, so the leaf sticks out of a too-small
            # opening. generate_wall_with_openings_mesh already clips the
            # cut internally; keeping seg_a/b at the true opening extent
            # guarantees visual parity between door and hole.
            off_lo = op.offset_m - half
            off_hi = op.offset_m + half
            seg_a = (a[0] + ux * off_lo, a[1] + uy * off_lo)
            seg_b = (a[0] + ux * off_hi, a[1] + uy * off_hi)
            cx = (seg_a[0] + seg_b[0]) * 0.5
            cz = (seg_a[1] + seg_b[1]) * 0.5
            op_height = op.head_m - op.sill_m
            kind = "DOOR" if op.kind == "door" else "WINDOW"
            attrs = {
                "segment_a": [seg_a[0], seg_a[1]],
                "segment_b": [seg_b[0], seg_b[1]],
                "length_m": round(op_height, 3),  # vertical extent for exporter
                "width_m": round(op.width_m, 3),
                "height_m": round(op_height, 3),
                "sill_height_m": op.sill_m,
                "head_height_m": op.head_m,
                "thickness_m": edge.thickness_m,
                "parent_edge_id": edge.id,
                "source": op.source,
                # Mesh-exporter hint: False = surface-mounted panel, no
                # subtract in the parent wall mesh (e.g. sliding doors).
                "cuts_wall": bool(getattr(op, "cuts_wall", True)),
            }
            if op.subtype:
                attrs["subtype"] = op.subtype
                if op.subtype == "door_double":
                    attrs["leaves"] = 2
                elif op.subtype == "door_sliding":
                    attrs["leaves"] = 0
                    attrs["sliding"] = True
                    attrs["surface_mount"] = True
                elif op.subtype == "window_large":
                    attrs["window_variant"] = "band"
                else:
                    attrs["leaves"] = 1
            else:
                if kind == "DOOR":
                    attrs["leaves"] = 1

            scene.add_component(
                Component(
                    id=op.id,
                    name=f"{kind.title()} {op.id}",
                    kind=kind,
                    geometry_ref=f"{op.kind}/{op.id}",
                    center=(cx, op.sill_m + op_height / 2, cz),
                    dims=(op.width_m, op_height, edge.thickness_m),
                    attrs=attrs,
                    evidence=[
                        Evidence(
                            source=op.source,
                            method="parametric_graph_attach",
                            confidence=op.confidence,
                        )
                    ],
                )
            )

            scene.add_relationship(
                Relationship(
                    src_id=edge.id,
                    rel_type="HAS_OPENING",
                    dst_id=op.id,
                    confidence=op.confidence,
                )
            )

    # Rooms - the graph doesn't yet carry room polygons; keep the legacy
    # room reconstruction by accepting an optional rooms_list parameter
    # (DetectRoomPolygonsStep output).
    if rooms_list:
        for room in rooms_list:
            room_id = room.get("id", "")
            label = room.get("label")
            area = room.get("area_m2")
            center = room.get("center", [0, 0])
            polygon = room.get("polygon", [])
            room_attrs: Dict[str, Any] = {}
            if label:
                room_attrs["label"] = label
            if area is not None:
                room_attrs["area_m2"] = area
            for extra in ("area_axis_m2", "area_net_m2", "polygon_axis"):
                if room.get(extra) is not None:
                    room_attrs[extra] = room[extra]
            if polygon:
                room_attrs["polygon"] = polygon
            meta = room.get("meta", {})
            if meta.get("room_type"):
                room_attrs["room_type"] = meta["room_type"]
            scene.add_component(
                Component(
                    id=room_id,
                    name=label or f"Room {room_id}",
                    kind="ROOM",
                    center=(float(center[0]), wall_height_m / 2, float(center[1])),
                    dims=None,
                    attrs=room_attrs,
                    evidence=[
                        Evidence(
                            source="plan_to_bim",
                            method="room_detection",
                            confidence=0.6,
                        )
                    ],
                )
            )
            for seg_id in room.get("bounding_segments", []):
                if seg_id in scene.components:
                    scene.add_relationship(
                        Relationship(
                            src_id=seg_id,
                            rel_type="BOUNDS",
                            dst_id=room_id,
                        )
                    )

    # Scene meta
    stats = graph.stats()
    scene.meta["wall_graph_built"] = True
    scene.meta["total_segments"] = stats["edges"]
    scene.meta["total_rooms"] = len(rooms_list or [])
    scene.meta["wall_height_m"] = wall_height_m
    scene.meta["segments_by_class"] = {
        "wall": stats["edges"],
        "door": stats["doors"],
        "window": stats["windows"],
    }
    return scene


class BuildSceneModelStep:
    """Pipeline step: convert semantics_dict or wall_graph to SceneModel.

    When ``wall_graph_enabled`` is true AND a WallGraph exists on the
    state, this step builds the scene directly from the graph - giving
    the glTF exporter a parent-child wall-opening topology with zero
    geometric guessing. Otherwise it falls back to the legacy
    ``build_scene_from_semantics`` polygon/CSG path.
    """

    name = "build_scene_model"

    def run(self, ctx, state):
        wall_height = float(state.config.get("wall_height_m", 2.80))
        wall_thickness = float(state.config.get("wall_thickness_m", 0.12))
        measured = state.metrics.get("wall_thickness_measured_m")
        exterior_t = float(state.config.get("wall_thickness_exterior_m", measured or 0.12))
        interior_t = float(
            state.config.get("wall_thickness_interior_m", measured * 0.65 if measured else 0.08)
        )

        use_graph = (
            bool(state.config.get("wall_graph_enabled", True)) and state.wall_graph is not None
        )

        if use_graph:
            rooms = (state.semantics_dict or {}).get("rooms", [])
            state.scene = build_scene_from_wall_graph(
                state.wall_graph,
                rooms_list=rooms,
                wall_height_m=wall_height,
            )
            ctx.log(
                f"SceneModel: built from WallGraph "
                f"({len(state.wall_graph.edges)} edges, "
                f"{sum(len(e.openings) for e in state.wall_graph.edges.values())} openings)"
            )
        else:
            if not state.semantics_dict:
                ctx.log("No semantics data, skipping scene model build")
                return state
            state.scene = build_scene_from_semantics(
                state.semantics_dict,
                wall_height_m=wall_height,
                wall_thickness_m=wall_thickness,
                wall_thickness_exterior_m=exterior_t,
                wall_thickness_interior_m=interior_t,
            )

        n_comp = len(state.scene.components)
        n_rel = len(state.scene.relationships)
        state.metrics["scene_components"] = n_comp
        state.metrics["scene_relationships"] = n_rel

        kind_counts: Dict[str, int] = {}
        for c in state.scene.components.values():
            kind_counts[c.kind] = kind_counts.get(c.kind, 0) + 1
        state.metrics["scene_kinds"] = kind_counts

        ctx.log(
            f"SceneModel: {n_comp} components, {n_rel} relationships, "
            f"kinds={kind_counts}"
            f" {'(graph-driven)' if use_graph else '(legacy path)'}"
        )
        return state
