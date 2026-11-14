"""Door-bridge builder — flat lintel slab tying wall stubs over a swing door.

The bridge is a horizontal volume sitting on top of the door frame
(``frame_top_y_m``, default 2.11 m) up to the wall top
(``wall_height_m``, default 2.80 m). It spans the door rectangle and
extends ``overlap_m`` (default 10 cm) into each adjacent wall stub
along the major axis. The visual effect: the two wall sides left and
right of the door appear continuously connected over the door lintel.

Sliding doors are skipped on purpose — the panel/track sits on the
wall plane, not in a gap, so the bridge concept does not apply.

This module replaces the post-process ``_add_door_bridges`` in
``scripts/merge_doors_and_windows_bim.py``: that script read frame-top
vertices back from the GLB, but here we already have the full
``GapGeometry`` in memory, so we can build the bridge directly with no
GLB-readback round trip.
"""
from __future__ import annotations

from typing import Optional

from .gap_geometry import GapGeometry
from .mesh_builder import _extrude_box_in_gap

# Door classes that get a bridge. Sliding doors are excluded.
SWING_DOOR_CLASSES_FOR_BRIDGE = frozenset({"door", "door_single", "door_double"})

# Top of the door-frame profile (DUK + Stockprofil-Hoehe per DIN 18101).
# The bridge starts above this so it does not overlap the door frame
# vertically.
DEFAULT_FRAME_TOP_Y_M = 2.110

# Tangential overlap into each adjacent wall stub. 10 cm renders as a
# clear visual connection while staying small enough that very short
# wall stubs do not get duplicated as phantom geometry.
DEFAULT_BRIDGE_WALL_OVERLAP_M = 0.10

# Same render path as the wall mesh: ``face_colors`` (no PBR material),
# RGB exactly matching the wall vertex colour produced by
# ``extrude_wall_mask``.
_BRIDGE_RGBA = (220, 220, 220, 255)


def is_bridge_class(label_class: str) -> bool:
    """True for swing-door classes that should receive a bridge mesh."""
    return (label_class or "") in SWING_DOOR_CLASSES_FOR_BRIDGE


def build_door_bridge(
    gap: GapGeometry,
    *,
    wall_height_m: float = 2.80,
    frame_top_y_m: float = DEFAULT_FRAME_TOP_Y_M,
    overlap_m: float = DEFAULT_BRIDGE_WALL_OVERLAP_M,
    flip_y_for_pdf: bool = True,
):
    """Build the flat bridge mesh tying both wall stubs over a swing door.

    Args:
        gap: The opening's ``GapGeometry`` (origin + axes + gap interval +
            measured wall thickness).
        wall_height_m: Top of the bridge slab (matches wall top).
        frame_top_y_m: Bottom of the bridge slab (top of door frame).
        overlap_m: Tangential extension into each adjacent wall stub.
        flip_y_for_pdf: Mirror the openings mesh-builder convention so
            scene Z = +plan Y. Should always be True in the V3 pipeline.

    Returns:
        A trimesh.Trimesh with ``face_colors = (220, 220, 220, 255)``
        applied, or ``None`` when bridge geometry is degenerate
        (e.g. wall_height_m <= frame_top_y_m).
    """
    height_y_m = float(wall_height_m) - float(frame_top_y_m)
    if height_y_m <= 1e-3:
        return None
    if gap.gap_length_m <= 1e-3 or gap.wall_depth_m <= 1e-3:
        return None

    s_lo = float(gap.gap_start) - float(overlap_m)
    s_hi = float(gap.gap_end) + float(overlap_m)
    n_half = 0.5 * float(gap.wall_depth_m)
    mesh = _extrude_box_in_gap(
        gap.origin, gap.major_axis, gap.minor_axis,
        s_lo, s_hi, -n_half, n_half,
        base_y_m=float(frame_top_y_m),
        height_y_m=height_y_m,
        flip_y_for_pdf=flip_y_for_pdf,
    )
    if mesh is None or len(getattr(mesh, "vertices", [])) == 0:
        return None
    try:
        mesh.visual.face_colors = list(_BRIDGE_RGBA)
    except Exception:
        # Visual subclass might reject direct face_colors assignment;
        # the wall mesh is rendered the same way and works, so this
        # fallback is just defensive.
        pass
    return mesh


def maybe_build_bridge(
    label_class: str,
    gap: Optional[GapGeometry],
    *,
    wall_height_m: float = 2.80,
    frame_top_y_m: float = DEFAULT_FRAME_TOP_Y_M,
    overlap_m: float = DEFAULT_BRIDGE_WALL_OVERLAP_M,
    flip_y_for_pdf: bool = True,
):
    """Class-gated wrapper: build a bridge only for swing-door classes.

    Returns ``None`` for sliding doors and any non-door class so the
    caller can wire this in unconditionally without per-class branching.
    """
    if not is_bridge_class(label_class):
        return None
    if gap is None:
        return None
    return build_door_bridge(
        gap,
        wall_height_m=wall_height_m,
        frame_top_y_m=frame_top_y_m,
        overlap_m=overlap_m,
        flip_y_for_pdf=flip_y_for_pdf,
    )
