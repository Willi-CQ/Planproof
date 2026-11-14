from __future__ import annotations

from pathlib import Path

from planproof_geometry.ai.plan_gltf_walls import write_walls_gltf
from .room_shell import RoomShell


def export_shell_to_gltf(
    shell: RoomShell,
    out_path: str,
    wall_thickness_m: float,
    add_slab: bool = True,
) -> str:
    """Exports a room shell to glTF compatible with the existing viewer."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    write_walls_gltf(
        out_path=str(out),
        wall_segments=shell.wall_segments,
        wall_height_m=shell.height_m,
        wall_thickness_m=wall_thickness_m,
        add_slab=add_slab,
        slab_thickness_m=0.05,
        plan_origin_xy=shell.origin_xy,
        flip_plan_y_for_pdf=False,
    )

    return str(out)
