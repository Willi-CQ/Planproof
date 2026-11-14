"""Step 13: Build SceneModel from room shells + refined classifications."""

from __future__ import annotations

from planproof_geometry.domain.scene_model import SceneModel, Component, Relationship, Evidence


def _rect_footprint(origin_xy, width_m, depth_m):
    """Rechteck-Ring (4 Punkte [x, y]) um ``origin_xy`` — Fallback, wenn kein
    echter Footprint aus ``detect_rooms`` vorliegt, damit JEDER Raum eine
    Kontur + Flaeche in die GLB/den Vergleich traegt (statt zu verschwinden)."""
    ox, oy = origin_xy
    hw, hd = width_m / 2.0, depth_m / 2.0
    return [
        [ox - hw, oy - hd],
        [ox + hw, oy - hd],
        [ox + hw, oy + hd],
        [ox - hw, oy + hd],
    ]


def _unit_dir(a, b):
    """Normierte 2D-Richtung a->b (Fallback (1,0) bei Entartung)."""
    dx, dy = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-9:
        return 1.0, 0.0
    return dx / n, dy / n


class BuildSceneStep:
    name = "Build scene"
    soft_fail = False

    def run(self, ctx, state):
        shells = state.shells or []
        if not shells:
            raise RuntimeError("No room shells to build scene from")

        room_height = state.metrics.get("room_height_m", 2.80)
        wall_thickness = state.config.get("wall_thickness_m", 0.12)

        scene = SceneModel(version="scene_v1", units="m", up_axis="Y")

        label_map = getattr(state, "heuristic_labels_merged", {}) or {}
        openings = state.openings or []

        wall_counter = 0
        opening_counter = 0

        for room_idx, shell in enumerate(shells, start=1):
            room_id = f"room_{room_idx}"
            # Szenenlokaler, deterministischer Raum-Schluessel (stabil ueber
            # Re-Laeufe desselben Scans) — traegt bis in Indexer/Vergleich als
            # extras.room_key. detect_rooms-Id bevorzugt, sonst Positionsindex.
            room_key = shell.room_id or room_id

            # Echte Raumkontur aus detect_rooms; Fallback = Rechteck aus width/
            # depth, damit JEDER Raum eine Kontur + Flaeche traegt.
            footprint = shell.footprint or _rect_footprint(shell.origin_xy, shell.width_m, shell.depth_m)
            room_area = shell.area_m2 if shell.area_m2 is not None else round(shell.width_m * shell.depth_m, 2)

            room_attrs = {
                "polygon": footprint,
                "area_m2": room_area,
                "room_key": room_key,
                "room_type": "generic",
            }
            # Add LLM room label if available
            if hasattr(state, "llm_room_labels"):
                for rl in (state.llm_room_labels or []):
                    if rl.get("room_id") == room_id:
                        room_attrs["label"] = rl.get("label", "")
                        room_attrs["label_confidence"] = rl.get("confidence", 0)

            scene.add_component(Component(
                id=room_id,
                name=f"Room {room_idx}",
                kind="ROOM",
                center=(shell.origin_xy[0], room_height / 2, shell.origin_xy[1]),
                dims=(shell.width_m, room_height, shell.depth_m),
                attrs=room_attrs,
                evidence=[Evidence(source="scan_to_bim", method="room_detection", confidence=0.8)],
            ))

            # Floor and ceiling
            floor_id = f"floor_{room_idx}"
            ceiling_id = f"ceiling_{room_idx}"

            scene.add_component(Component(
                id=floor_id,
                name=f"Floor {room_idx}",
                kind="FLOOR",
                center=(shell.origin_xy[0], 0.025, shell.origin_xy[1]),
                dims=(shell.width_m, 0.05, shell.depth_m),
                geometry_ref="slab",
                attrs={"room_key": room_key},
                evidence=[Evidence(source="scan_to_bim", method="plane_detection", confidence=0.9)],
            ))
            scene.add_component(Component(
                id=ceiling_id,
                name=f"Ceiling {room_idx}",
                kind="CEILING",
                center=(shell.origin_xy[0], room_height - 0.025, shell.origin_xy[1]),
                dims=(shell.width_m, 0.05, shell.depth_m),
                geometry_ref="slab",
                attrs={"room_key": room_key},
                evidence=[Evidence(source="scan_to_bim", method="plane_detection", confidence=0.9)],
            ))

            scene.add_relationship(Relationship(src_id=floor_id, rel_type="PART_OF", dst_id=room_id))
            scene.add_relationship(Relationship(src_id=ceiling_id, rel_type="PART_OF", dst_id=room_id))

            # Walls — seg_id -> Wand-Komponenten-Id, damit die Oeffnungen unten
            # eine HAS_OPENING-Kante zur richtigen Wand ziehen koennen (der GLB-
            # Exporter schneidet die Oeffnung nur bei vorhandener HAS_OPENING).
            seg_id_to_wid: dict = {}
            for seg in shell.wall_segments:
                wall_counter += 1
                wid = f"wall_{wall_counter}"
                a = seg.get("a")
                b = seg.get("b")
                if not a or not b:
                    continue

                ax, ay = float(a[0]), float(a[1])
                bx, by = float(b[0]), float(b[1])
                cx = (ax + bx) / 2
                cz = (ay + by) / 2
                length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5

                # Get refined classification
                seg_id = seg.get("id", wid)
                seg_id_to_wid[seg_id] = wid
                label_info = label_map.get(seg_id, {})
                wall_kind = "WALL"
                # segment_a/b + thickness/height sind Pflicht, damit der echte
                # GLB-Exporter (export_scene_to_gltf) die Wand rendert; room_key
                # traegt die Raumzugehoerigkeit bis in den Vergleich.
                wall_attrs = {
                    "segment_a": [ax, ay],
                    "segment_b": [bx, by],
                    "thickness_m": wall_thickness,
                    "height_m": room_height,
                    "room_key": room_key,
                    "source": "scan_to_bim",
                }
                confidence = 0.5

                refined_class = label_info.get("refined_class") or label_info.get("primary_class", "wall")
                if refined_class in ("wall_exterior", "wall_interior"):
                    # Exporter erwartet wall_type ``exterior``/``interior``.
                    wall_attrs["wall_type"] = "exterior" if refined_class == "wall_exterior" else "interior"
                elif refined_class == "column":
                    wall_kind = "COLUMN"

                confidence = label_info.get("refined_confidence") or label_info.get("primary_confidence", 0.5)
                if label_info.get("reasoning"):
                    wall_attrs["llm_reasoning"] = label_info["reasoning"]
                if label_info.get("llm_attributes"):
                    wall_attrs.update(label_info["llm_attributes"])

                method = "llm_refinement" if label_info.get("refined_class") else "heuristic"

                scene.add_component(Component(
                    id=wid,
                    name=f"Wall {wall_counter}",
                    kind=wall_kind,
                    center=(cx, room_height / 2, cz),
                    dims=(length, room_height, wall_thickness),
                    geometry_ref=f"wall/{seg_id}",
                    attrs=wall_attrs,
                    evidence=[Evidence(
                        source="scan_to_bim",
                        method=method,
                        confidence=float(confidence),
                    )],
                ))

                scene.add_relationship(Relationship(src_id=wid, rel_type="PART_OF", dst_id=room_id))
                scene.add_relationship(Relationship(src_id=wid, rel_type="BOUNDS", dst_id=floor_id))

            # Openings for this room's walls
            room_wall_ids = {seg.get("id") for seg in shell.wall_segments}
            for opening in openings:
                parent_wall_idx = opening.get("parent_wall_idx")
                # Match opening to wall segment by index
                matched = False
                parent_seg_id = None
                parent_seg = None
                if parent_wall_idx is not None and state.wall_segments_2d:
                    if parent_wall_idx < len(state.wall_segments_2d):
                        parent_seg = state.wall_segments_2d[parent_wall_idx]
                        parent_seg_id = parent_seg.get("id")
                        if parent_seg_id in room_wall_ids:
                            matched = True

                if not matched:
                    continue

                opening_counter += 1
                oid = f"opening_{opening_counter}"

                label_info = label_map.get(opening["id"], {})
                opening_class = label_info.get("refined_class") or label_info.get("primary_class") or opening["type"]
                opening_conf = label_info.get("refined_confidence") or label_info.get("primary_confidence", 0.5)

                kind = "DOOR" if opening_class == "door" else "WINDOW"
                center_xy = opening.get("center", (0, 0))
                center_z = opening.get("center_z", room_height / 2)
                width_m = float(opening["width_m"])

                # segment_a/b entlang der Elternwand-Richtung (Zentrum +- w/2),
                # damit der echte GLB-Exporter die Oeffnung an der Wand platziert
                # und (via HAS_OPENING) schneidet.
                if parent_seg is not None and parent_seg.get("a") and parent_seg.get("b"):
                    ux, uy = _unit_dir(parent_seg["a"], parent_seg["b"])
                else:
                    ux, uy = 1.0, 0.0
                ocx, ocy = float(center_xy[0]), float(center_xy[1])
                seg_a = [ocx - ux * width_m / 2.0, ocy - uy * width_m / 2.0]
                seg_b = [ocx + ux * width_m / 2.0, ocy + uy * width_m / 2.0]

                opening_attrs = {
                    "width_m": width_m,
                    "height_m": opening["height_m"],
                    "base_height_m": opening["base_height_m"],
                    "segment_a": seg_a,
                    "segment_b": seg_b,
                    "room_key": room_key,
                    "source": "scan_to_bim",
                }
                if kind == "WINDOW":
                    # Exporter liest sill_height_m fuer die Fenster-Bruestung.
                    opening_attrs["sill_height_m"] = opening["base_height_m"]
                if label_info.get("llm_attributes"):
                    opening_attrs.update(label_info["llm_attributes"])

                method = "llm_refinement" if label_info.get("refined_class") else "heuristic"

                scene.add_component(Component(
                    id=oid,
                    name=f"{kind.title()} {opening_counter}",
                    kind=kind,
                    center=(center_xy[0], center_z, center_xy[1]),
                    dims=(opening["width_m"], opening["height_m"], wall_thickness),
                    attrs=opening_attrs,
                    evidence=[Evidence(
                        source="scan_to_bim",
                        method=method,
                        confidence=float(opening_conf),
                    )],
                ))

                scene.add_relationship(Relationship(src_id=oid, rel_type="PART_OF", dst_id=room_id))
                # HAS_OPENING(Wand -> Oeffnung): der Exporter schneidet die
                # Oeffnung nur, wenn diese Kante zur Elternwand existiert.
                parent_wid = seg_id_to_wid.get(parent_seg_id)
                if parent_wid:
                    scene.add_relationship(Relationship(src_id=parent_wid, rel_type="HAS_OPENING", dst_id=oid))

        # Add global observations from LLM as scene meta
        if state.llm_refinement and hasattr(state, "llm_global_observations"):
            scene.meta["llm_observations"] = state.llm_global_observations or {}
        scene.meta["quality_score"] = state.metrics.get("quality_score", 0)
        scene.meta["quality_grade"] = state.metrics.get("quality_grade", "D")

        state.scene = scene
        state.metrics["scene_components"] = len(scene.components)
        state.metrics["scene_relationships"] = len(scene.relationships)

        ctx.log(f"Scene: {len(scene.components)} components, {len(scene.relationships)} relationships")
        return state
