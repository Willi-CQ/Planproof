"""Phase 3 — Openings live in gaps between two walls (doors only).

Window detection was removed in the 2026-05-06 reset; only door-class
labels are routed through this package.

Algorithm per opening-label O:
  1. Find the two nearest wall_polygons (W1, W2) flanking O
  2. Establish the common wall axis via PCA over W1 + W2 vertices
  3. Project W1, W2, O onto the axis -> gap interval [t_gap_lo, t_gap_hi]
  4. Resolve DIN sill/head bands per label class
  5. Build three boxes in the gap:
       Parapet (0 -> sill)
       Fill    (sill -> head)   void for doors
       Lintel  (head -> wall_height)

Walls-immutable: W1 and W2 are never touched. The opening prefabs are
concatenated to the wall mesh as separate components.
"""
from .door_bridges import (  # noqa: F401
    SWING_DOOR_CLASSES_FOR_BRIDGE,
    build_door_bridge,
    is_bridge_class,
    maybe_build_bridge,
)
from .gap_geometry import (  # noqa: F401
    GapGeometry,
    compute_gap_geometry,
)
from .height_bands import (  # noqa: F401
    OpeningHeights,
    resolve_heights,
)
from .mesh_builder import (  # noqa: F401
    OpeningMeshes,
    build_opening_meshes,
)
from .nearest_walls import (  # noqa: F401
    FlankingWalls,
    find_flanking_walls,
)
from .records import (  # noqa: F401
    OpeningRecord,
    PlacementStatus,
)
from .window_block_clipper import (  # noqa: F401
    ClippedWindowBlock,
    clip_blocks_to_walls,
)
from .window_label_blocks import (  # noqa: F401
    WindowBlock,
    find_window_blocks,
)
from .window_mesh_builder import (  # noqa: F401
    build_window_meshes,
)
from .window_placement import (  # noqa: F401
    CriterionResult,
    PlacedWindow,
    place_windows_in_blocks,
    verify_placement_criterion_1,
)
