"""Step 11: Build RoomShell instances from refined pipeline geometry."""

from __future__ import annotations

from typing import List

from ..room_shell import RoomShell, build_rect_room_shell


class BuildShellsStep:
    name = "Build room shells"
    soft_fail = False

    def run(self, ctx, state):
        rooms = state.rooms or []
        wall_segments = state.wall_segments_2d or []
        room_height = state.metrics.get("room_height_m", 2.80)

        # Merge LLM refinement into labels if available
        label_map = {}
        for lbl in (state.heuristic_labels or []):
            label_map[lbl["element_id"]] = lbl
        if state.llm_refinement:
            for ref in state.llm_refinement:
                eid = ref.get("element_id") or ref.get("id")
                if eid and eid in label_map:
                    label_map[eid]["refined_class"] = ref.get("final_class")
                    label_map[eid]["refined_confidence"] = ref.get("confidence", 0)
                    label_map[eid]["reasoning"] = ref.get("reasoning", "")
                    label_map[eid]["llm_attributes"] = ref.get("attributes", {})

        shells: List[RoomShell] = []

        if rooms and wall_segments:
            # Build shell per room from detected geometry
            seg_map = {s["id"]: s for s in wall_segments}
            for room in rooms:
                room_segs = []
                for wid in room.get("wall_ids", []):
                    seg = seg_map.get(wid)
                    if seg:
                        room_segs.append(seg)

                if not room_segs:
                    continue

                # Compute room bounding box from segments
                all_x = []
                all_y = []
                for s in room_segs:
                    all_x.extend([s["a"][0], s["b"][0]])
                    all_y.extend([s["a"][1], s["b"][1]])

                width = max(all_x) - min(all_x)
                depth = max(all_y) - min(all_y)
                cx = (max(all_x) + min(all_x)) / 2
                cy = (max(all_y) + min(all_y)) / 2

                shell = RoomShell(
                    wall_segments=room_segs,
                    width_m=round(width, 2),
                    depth_m=round(depth, 2),
                    height_m=room_height,
                    origin_xy=(cx, cy),
                    # Echte Raumkontur + Id/Flaeche aus detect_rooms durchreichen
                    # (statt sie auf die bbox oben zu reduzieren).
                    footprint=room.get("polygon"),
                    room_id=room.get("id"),
                    area_m2=room.get("area_m2"),
                )
                shells.append(shell)
        elif wall_segments:
            # No rooms detected, single shell from all walls
            all_x = []
            all_y = []
            for s in wall_segments:
                all_x.extend([s["a"][0], s["b"][0]])
                all_y.extend([s["a"][1], s["b"][1]])

            width = max(all_x) - min(all_x)
            depth = max(all_y) - min(all_y)
            cx = (max(all_x) + min(all_x)) / 2
            cy = (max(all_y) + min(all_y)) / 2

            shell = RoomShell(
                wall_segments=wall_segments,
                width_m=round(width, 2),
                depth_m=round(depth, 2),
                height_m=room_height,
                origin_xy=(cx, cy),
            )
            shells.append(shell)
        else:
            # Fallback: manual dimensions or default
            w = state.config.get("manual_width_m") or state.metrics.get("bbox_x_m", 5.0)
            d = state.config.get("manual_depth_m") or state.metrics.get("bbox_y_m", 4.0)
            h = state.config.get("manual_height_m") or room_height
            shells.append(build_rect_room_shell(float(w), float(d), float(h)))
            state.quality_warnings.append("build_shells_fallback_rect")

        state.shells = shells
        state.heuristic_labels_merged = label_map
        state.metrics["rooms_built"] = len(shells)

        ctx.log(f"Built {len(shells)} room shell(s)")
        return state
