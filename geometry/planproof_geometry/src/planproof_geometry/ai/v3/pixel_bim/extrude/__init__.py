"""Phase 2 — Pixel footprint extrusion.

mask_to_mesh: full-resolution wall mask -> list of 2D polygons -> one
watertight 3D mesh extruded to wall height. Replaces the V2 wall-graph
with a 5-step pure-pixel chain:

    wall_mask (union of Phase-1 refined masks)
        |
        +-- optional morph-close if paired-outline mode detected
        |
        +-- cv2.findContours(RETR_CCOMP) -> outer shells + holes
        |
        +-- shapely.Polygon(shell, holes).buffer(0).simplify(2px)
        |
        +-- shapely.ops.unary_union  (merges T-junctions, dedupes overlap)
        |
        +-- trimesh.creation.extrude_polygon(poly, height=wall_height_m)
"""

from .mask_to_mesh import (  # noqa: F401
    MaskToMeshResult,
    WallPolygon,
    extrude_wall_mask,
    mask_to_wall_polygons,
    wall_polygons_to_mesh,
)
from .paired_outline import detect_paired_outline_mode  # noqa: F401
