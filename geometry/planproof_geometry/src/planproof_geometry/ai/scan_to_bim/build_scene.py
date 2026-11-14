from __future__ import annotations

from planproof_geometry.domain.scene_model import SceneModel, Component, Relationship, Evidence
from .room_shell import RoomShell


def build_scene_from_shell(shell: RoomShell, wall_thickness_m: float = 0.12) -> SceneModel:
    """Build a clean SceneModel (BIM-lite) from a RoomShell."""
    scene = SceneModel(version="scene_v1", units="m", up_axis="Y")

    room_id = "room_1"
    scene.add_component(
        Component(
            id=room_id,
            name="Room 1",
            kind="ROOM",
            center=(0.0, shell.height_m / 2.0, 0.0),
            dims=(shell.width_m, shell.height_m, shell.depth_m),
            attrs={"origin_xy": shell.origin_xy},
            evidence=[Evidence(source="scan_to_bim", method="room_shell", confidence=1.0)],
        )
    )

    floor_id = "floor_1"
    ceiling_id = "ceiling_1"

    scene.add_component(
        Component(
            id=floor_id,
            name="Floor",
            kind="FLOOR",
            center=(0.0, 0.025, 0.0),
            dims=(shell.width_m, 0.05, shell.depth_m),
            geometry_ref="slab",
            evidence=[Evidence(source="scan_to_bim", method="room_shell", confidence=1.0)],
        )
    )
    scene.add_component(
        Component(
            id=ceiling_id,
            name="Ceiling",
            kind="CEILING",
            center=(0.0, shell.height_m - 0.025, 0.0),
            dims=(shell.width_m, 0.05, shell.depth_m),
            geometry_ref="slab",
            evidence=[Evidence(source="scan_to_bim", method="room_shell", confidence=1.0)],
        )
    )

    scene.add_relationship(Relationship(src_id=floor_id, rel_type="PART_OF", dst_id=room_id))
    scene.add_relationship(Relationship(src_id=ceiling_id, rel_type="PART_OF", dst_id=room_id))

    for i, seg in enumerate(shell.wall_segments, start=1):
        wid = f"wall_{i}"
        a = seg.get("a")
        b = seg.get("b")
        if not a or not b:
            continue

        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        cx = (ax + bx) / 2.0
        cz = (ay + by) / 2.0  # plan y -> scene z
        length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5

        scene.add_component(
            Component(
                id=wid,
                name=f"Wall {i}",
                kind="WALL",
                center=(cx, shell.height_m / 2.0, cz),
                dims=(length, shell.height_m, float(wall_thickness_m)),
                geometry_ref=f"wall/{seg.get('id', wid)}",
                attrs={"segment_id": seg.get("id"), "a": a, "b": b},
                evidence=[Evidence(source="scan_to_bim", method="room_shell", confidence=1.0)],
            )
        )

        scene.add_relationship(Relationship(src_id=wid, rel_type="PART_OF", dst_id=room_id))
        scene.add_relationship(Relationship(src_id=wid, rel_type="BOUNDS", dst_id=floor_id))
        scene.add_relationship(Relationship(src_id=wid, rel_type="BOUNDS", dst_id=ceiling_id))

    return scene
