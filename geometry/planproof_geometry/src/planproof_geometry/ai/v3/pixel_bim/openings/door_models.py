"""3D door geometry for the BIM viewer.

Builds the visible door components (jambs, head trim, swing leaf or
sliding panel + track) for a placed door rectangle. The wall mesh has
already been cut by ``apply_door_voids``; these meshes sit *inside*
that void.

Design rules (Willi 2026-04-28 review):
  1. **Frame and leaf depth ≤ wall depth.** Every component is recessed
     2 mm inside the wall thickness so nothing visibly protrudes from
     either wall surface (and we avoid Z-fight with the wall mesh).
  2. **Swing direction is data-driven.** The caller passes
     ``swing_dir_sign``; the leaf rotates toward that half-space.
  3. **No standalone lintel beam.** The wall above the void
     (head_height → wall_height) already serves as the lintel; we only
     add a thin head trim (2 cm tall) recessed inside the wall so the
     wall above visually flows into the frame, not on top of a
     protruding beam.
  4. **Sliding doors:** ``mount_type='pocket'`` keeps the panel inside
     the wall thickness (slid sideways into a pocket, partially visible
     in the opening); ``mount_type='surface'`` rides the panel just
     outside the wall surface (typical barn-door style).

Coordinate convention (matches mask_to_mesh / door_voids):
  * plan (px, py)    → 3D (px, 0, py)        with ``flip_y_for_pdf=True``
  * height           → 3D +Y
  * a 2D unit vector (ux, uy) in plan → 3D (ux, 0, uy)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Visible-component dimensions (metres). Tuned for screenshot clarity,
# not millimetre BIM accuracy.
JAMB_WIDTH_M = 0.06           # how far the jamb sticks into the opening
HEAD_TRIM_HEIGHT_M = 0.04     # head trim is a thin band at the top of the void
LEAF_BOTTOM_GAP_M = 0.02
LEAF_TOP_GAP_M = 0.04
LEAF_DEFAULT_THICKNESS_M = 0.04
SWING_ANGLE_DEG = 75.0
DEFAULT_HEAD_HEIGHT_M = 2.10

# Sliding-door dimensions.
SLIDE_PANEL_THICKNESS_M = 0.035
SLIDE_OPEN_FRACTION = 0.7        # how far the panel slides toward the pocket
SLIDE_TRACK_HEIGHT_M = 0.04
SLIDE_TRACK_OVERHANG_M = 0.08    # rail extends past the void on the pocket side

# Surface-mounted sliding (barn-door style): panel sits just outside the
# wall surface. The total protrusion is panel_thickness + this gap.
SLIDE_SURFACE_GAP_M = 0.015
# Pocket sliding: panel is centred on the wall axis and recessed.
SLIDE_PANEL_INSET_M = 0.005      # 5 mm inset on each side of the wall

# Recess every in-wall component this much inside the wall depth so it
# does not Z-fight with the wall surface.
WALL_INSET_M = 0.002

# door_voids.py extrudes the void with these inflations, so the wall mesh
# has been carved 5 mm wider than the door rectangle in plan and 5 mm
# above ``head_height``. Frame components are extended by these margins
# so they fully cover the void edges (no visible seam between jamb/trim
# and the wall above / lateral wall flank).
VOID_INFLATE_XY_M = 0.005
VOID_INFLATE_Y_M = 0.005

# Lintel box (head -> wall_top) extends past the door rectangle on each
# side by this much. Wall annotations are typically drawn as separate
# rectangles that don't touch each other (spatial gaps of 1-2 cm at
# door endpoints are common, sometimes more), so a 5 mm match-the-void
# extension was too tight to overlap and fuse with the flanking walls.
# 50 mm reaches well into the wall material on both sides; the lintel
# is full wall-depth so the extension stays *inside* the wall flank,
# never visibly enlarging the wall outline.
LINTEL_LONG_OVERLAP_M = 0.050


# Material colours (RGBA 0-255). model-viewer / GLB understands these
# via trimesh.visual.face_colors.
COLOR_FRAME = (170, 120, 65, 255)     # warm walnut (jambs + head trim)
COLOR_LEAF = (215, 175, 110, 255)     # lighter oak (swing leaf)
COLOR_TRACK = (95, 100, 105, 255)     # anodised aluminium (sliding rail)
COLOR_PANEL_GLASS = (210, 225, 235, 200)  # frosted glass (sliding panel)


@dataclass
class PlacedDoor:
    """Minimal placement record consumed by the geometry builder."""
    polygon_xy: List[Tuple[float, float]]      # 4 corners (CCW), plan-metres
    label_class: str = "door_single"
    label_id: str = ""
    head_height_m: float = DEFAULT_HEAD_HEIGHT_M
    wall_depth_m: float = 0.12                  # wall thickness around door
    # Direction the leaf swings into. +1 = toward +short_axis (90° CCW
    # of long_axis in plan), -1 = toward -short_axis. Default +1 is the
    # legacy guess; callers should set this from a room-side heuristic.
    swing_dir_sign: int = +1
    # Sliding-door mount: 'pocket' (panel inside wall) or 'surface'
    # (panel rides just outside one wall face). 'auto' picks 'pocket'.
    mount_type: str = "pocket"
    # Which long-axis end is the pocket (sliding) or the hinge (swing).
    # +1 = hinge/pocket on the +long-axis end, -1 = on the -long-axis end.
    pocket_sign: int = +1
    # Hinge end of a single-leaf swing door: -1 = hinge at -W/2 end, +1 = at
    # +W/2 end. Used to flip the leaf so it pivots from the correct corner
    # of the opening (only relevant for ``label_class != door_double``).
    # Default -1 = legacy behaviour; callers with arc data should set it.
    hinge_long_sign: int = -1
    # Confidence of ``hinge_long_sign`` on a 0..1 scale, produced by
    # ``vote_hinge_long_sign`` (Tag-7.7). 1.0 = three or more independent
    # signals agreed, < 0.5 = single signal or split vote → reviewer
    # should treat the leaf orientation as tentative.
    hinge_confidence: float = 1.0
    # Per-signal hinge votes (signal_name -> +1/-1). Empty for double /
    # sliding doors. Carried through to the top-down render so the PNG
    # can show why a door's hinge sign is what it is.
    hinge_votes: Dict[str, int] = field(default_factory=dict)
    # Tag-7.8 review carry-throughs. ``gap_overlap_pct`` is the share of
    # the door's wall-tangent extent that lies on absent wall outline
    # (real opening); ``pass_validation`` mirrors the placement
    # ``DoorPlacement.pass_validation`` so the renderer can border the
    # door rectangle in green / red. Both default to "ok" so callers
    # that leave them blank don't get punished.
    gap_overlap_pct: float = 1.0
    pass_validation: bool = True
    # Tag-7.9 — diagnostic overlay: the cyan pixel-symbol rectangle
    # extracted by ``extract_pixel_door``. Lets the top-down PNG show
    # where the panel symbol actually sits vs. where the algorithm
    # placed the door, so a reviewer can see at a glance whether the
    # bbox or the symbol is the better anchor.
    pixel_polygon_xy: List[Tuple[float, float]] = field(default_factory=list)
    width_source: str = ""
    # Tag-7.10 — count of long-axis end-flanks within snap distance of
    # wall material. 2 = doubly framed, 1 = end-of-wall, 0 = floating.
    jamb_anchors: int = 2
    # Tag-7.11 — per-side anchor flags. minus=-W/2 long-axis end,
    # plus=+W/2 end. Default both True so doors that don't propagate
    # the per-side info (legacy callers) render as 2/2.
    jamb_anchor_minus: bool = True
    jamb_anchor_plus: bool = True
    # Signed offset of the wall centerline from the door polygon centre,
    # projected onto +short_axis. The geometry is translated by this
    # along short_axis so the frame sits centrally between the wall faces
    # (i.e. on the wall midplane), even when ``placement.door_polygon_xy``
    # is offset toward one wall surface. 0.0 = use door polygon centre.
    short_offset_m: float = 0.0
    # Sprint Sliding-Door-In-Wall-Classifier Tag-3: ``"in_wall"`` or
    # ``"beside_wall"`` for sliding doors. ``None`` for swing doors and
    # legacy-cascade sliding placements. Beside-wall sliders skip the
    # apply_door_voids cut — their panel is a standalone barn-door
    # element, not an aperture in the wall.
    sliding_mount_position: Optional[str] = None


# ── Geometry helpers ────────────────────────────────────────────────────


def _polygon_basis(polygon_xy: Sequence[Tuple[float, float]]
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """From a 4-corner CCW rectangle, return (center, long_unit, short_unit,
    long_len, short_len) in plan coordinates.

    long_unit points along the wall (matches the longer rectangle edge);
    short_unit is its 90° CCW rotation in plan, perpendicular to the wall.
    """
    pts = np.asarray(polygon_xy, dtype=np.float64)
    if pts.shape[0] < 4:
        raise ValueError("door polygon needs 4 corners")
    e01 = pts[1] - pts[0]
    e12 = pts[2] - pts[1]
    n01 = float(np.linalg.norm(e01))
    n12 = float(np.linalg.norm(e12))
    if n01 >= n12:
        long_vec, long_len = e01, n01
        short_len = n12
    else:
        long_vec, long_len = e12, n12
        short_len = n01
    if long_len < 1e-6:
        raise ValueError("degenerate door polygon")
    long_unit = long_vec / long_len
    short_unit = np.array([-long_unit[1], long_unit[0]])
    center = pts.mean(axis=0)
    return center, long_unit, short_unit, long_len, short_len


def build_door_rectangle_from_polygon(
    polygon_xy: Sequence[Tuple[float, float]],
    *,
    wall_depth_m: float,
    short_offset_m: float = 0.0,
) -> List[Tuple[float, float]]:
    """Low-level rectangle generator. Use this when you do not yet have
    a ``PlacedDoor`` (e.g. inside the placement pipeline, where only
    ``DoorPlacement`` exists). Same contract as ``build_door_rectangle``
    but takes the three primitive inputs directly.
    """
    center, long_u, short_u, W, _ = _polygon_basis(polygon_xy)
    new_center = center + short_offset_m * short_u
    half_w = W * 0.5
    half_d = max(0.05, wall_depth_m) * 0.5
    corners = [
        new_center + half_w * long_u + half_d * short_u,
        new_center - half_w * long_u + half_d * short_u,
        new_center - half_w * long_u - half_d * short_u,
        new_center + half_w * long_u - half_d * short_u,
    ]
    return [(float(c[0]), float(c[1])) for c in corners]


def build_sliding_door_polygon(
    wall_ctx,
    bbox_m: Tuple[float, float, float, float],
    panel_width_m: float,
) -> Tuple[List[Tuple[float, float]], float, int]:
    """Sprint Sliding-Door-Rewrite Tag-2.

    Construct a sliding-door rectangle from a ``WallContext`` (4-dir scan
    result) and the annotation bbox. Replaces the gap+jamb cascade for
    sliding doors with a deterministic 3-step construction:

      1. Centre = ``wall_ctx.wall_centerline_xy`` (already on the wall
         midplane — bbox centre projected perpendicularly to the wall).
      2. Long extent = ``panel_width_m`` along ``wall_ctx.wall_axis_long_u``.
      3. Short extent = ``wall_ctx.wall_depth_m`` along ``wall_axis_short_u``.

    Returns ``(polygon_xy, short_offset_m, pocket_sign)``:
      * ``polygon_xy`` — 4 CCW corners on the wall midplane.
      * ``short_offset_m`` — 0.0 (polygon already sits on the midplane,
        so no further offset is needed by build_door_rectangle).
      * ``pocket_sign`` — +1 if the wall extends further along
        +long_u than -long_u, else -1. The panel slides toward the
        longer-wall side. Falls out of the scan directly — no
        separate ``classify_sliding_mount`` pass needed.
    """
    long_u = np.asarray(wall_ctx.wall_axis_long_u, dtype=np.float64)
    short_u = np.asarray(wall_ctx.wall_axis_short_u, dtype=np.float64)
    centre = np.asarray(wall_ctx.wall_centerline_xy, dtype=np.float64)
    half_w = max(panel_width_m, 0.10) * 0.5
    half_d = max(wall_ctx.wall_depth_m, 0.05) * 0.5
    corners = [
        centre + half_w * long_u + half_d * short_u,
        centre - half_w * long_u + half_d * short_u,
        centre - half_w * long_u - half_d * short_u,
        centre + half_w * long_u - half_d * short_u,
    ]
    polygon_xy = [(float(c[0]), float(c[1])) for c in corners]
    pocket_sign = (
        +1 if wall_ctx.extent_long_pos_m >= wall_ctx.extent_long_neg_m else -1
    )
    return polygon_xy, 0.0, pocket_sign


def build_beside_wall_sliding_polygon(
    wall_ctx,
    bbox_m: Tuple[float, float, float, float],
    panel_width_m: float,
    *,
    panel_thickness_m: float = SLIDE_PANEL_THICKNESS_M,
) -> Tuple[List[Tuple[float, float]], float, int]:
    """Sprint Sliding-Door-In-Wall-Classifier Tag-3.

    Construct a beside-wall (barn-door) sliding rectangle. Unlike
    ``build_sliding_door_polygon``, this places the panel at the bbox
    centre — the symbol was drawn BESIDE the wall, not inside it — and
    the rectangle's short extent is the panel thickness, not the wall
    thickness. The renderer (mount=``beside``) then puts the panel
    mesh at the polygon centre with zero short-shift.

    Returns ``(polygon_xy, short_offset_m, pocket_sign)``:
      * ``polygon_xy`` — 4 CCW corners centred on the bbox, axis along
        the nearest wall's long axis (so the panel runs parallel to
        the wall rather than at an arbitrary angle).
      * ``short_offset_m`` — 0.0 (polygon already at the bbox centre,
        no further shift required).
      * ``pocket_sign`` — same convention as ``build_sliding_door_polygon``:
        +1 if the wall extends further along +long_u than -long_u.
    """
    long_u = np.asarray(wall_ctx.wall_axis_long_u, dtype=np.float64)
    short_u = np.asarray(wall_ctx.wall_axis_short_u, dtype=np.float64)
    bbox_center = np.array([
        (bbox_m[0] + bbox_m[2]) * 0.5,
        (bbox_m[1] + bbox_m[3]) * 0.5,
    ], dtype=np.float64)
    half_w = max(panel_width_m, 0.10) * 0.5
    half_t = max(panel_thickness_m, 0.015) * 0.5
    corners = [
        bbox_center + half_w * long_u + half_t * short_u,
        bbox_center - half_w * long_u + half_t * short_u,
        bbox_center - half_w * long_u - half_t * short_u,
        bbox_center + half_w * long_u - half_t * short_u,
    ]
    polygon_xy = [(float(c[0]), float(c[1])) for c in corners]
    pocket_sign = (
        +1 if wall_ctx.extent_long_pos_m >= wall_ctx.extent_long_neg_m else -1
    )
    return polygon_xy, 0.0, pocket_sign


def build_door_rectangle(
    placed: "PlacedDoor",
) -> List[Tuple[float, float]]:
    """Canonical wall-midplane rectangle for a placed door.

    Single source of truth shared by frame, void, and bridge so all three
    components reference identical 4 corners. The rectangle is:

      * **centered** on ``polygon_center + short_offset_m * short_u`` —
        the wall midplane, not the (possibly off-centre) polygon centre.
      * **long extent** = ``W`` from ``_polygon_basis`` (polygon's long
        edge — the door's actual placed width).
      * **short extent** = ``max(0.05, placed.wall_depth_m)`` — the
        measured wall depth, *not* the polygon's short edge. After
        authority-reclip in ``mask_to_mesh``, the polygon's short edge
        can drift from the actual wall thickness; ``wall_depth_m`` is
        the canonical measurement.

    Returns 4 CCW corners (plan-metres). Order matches ``_polygon_basis``:
    +long+short, -long+short, -long-short, +long-short.
    """
    return build_door_rectangle_from_polygon(
        placed.polygon_xy,
        wall_depth_m=placed.wall_depth_m,
        short_offset_m=placed.short_offset_m,
    )


def _basis_to_3d(long_unit_2d: np.ndarray,
                 short_unit_2d: np.ndarray) -> np.ndarray:
    """4×4 rotation that maps box-local (X→long, Y→up, Z→short) to 3D."""
    long_3d = np.array([long_unit_2d[0], 0.0, long_unit_2d[1]])
    short_3d = np.array([short_unit_2d[0], 0.0, short_unit_2d[1]])
    up_3d = np.array([0.0, 1.0, 0.0])
    R = np.eye(4)
    R[:3, 0] = long_3d
    R[:3, 1] = up_3d
    R[:3, 2] = short_3d
    return R


def _T(translation_xyz) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3] = translation_xyz
    return M


def _Ry(angle_rad: float) -> np.ndarray:
    """Rotation around the Y (up) axis."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    M = np.eye(4)
    M[0, 0] = c
    M[0, 2] = s
    M[2, 0] = -s
    M[2, 2] = c
    return M


def _box(extents_xyz, transform: np.ndarray, color):
    import trimesh
    m = trimesh.creation.box(extents=list(extents_xyz))
    m.apply_transform(transform)
    if color is not None:
        m.visual.face_colors = list(color)
    return m


def _wall_inset_depth(wall_depth_m: float, min_depth_m: float = 0.04) -> float:
    """Depth of an in-wall component, slightly recessed inside the wall."""
    return max(min_depth_m, wall_depth_m - 2 * WALL_INSET_M)


# ── Heuristic helpers (caller-side) ─────────────────────────────────────


def compute_swing_dir_sign(
    polygon_xy: Sequence[Tuple[float, float]],
    bbox_m: Tuple[float, float, float, float],
) -> int:
    """Pick +1/-1 swing direction from the label bbox centre.

    The label's bounding box typically extends from the wall surface
    *into the room* — that is where the swing arc symbol sits in the
    PDF. So the room half-space is the side the bbox centre lies on
    relative to the wall.
    """
    center, long_u, short_u, *_ = _polygon_basis(polygon_xy)
    bbox_center = np.array(
        [(bbox_m[0] + bbox_m[2]) * 0.5, (bbox_m[1] + bbox_m[3]) * 0.5]
    )
    delta = bbox_center - center
    proj = float(delta @ short_u)
    return +1 if proj >= 0.0 else -1


def compute_hinge_long_sign(
    polygon_xy: Sequence[Tuple[float, float]],
    hinge_xy_m: Tuple[float, float],
) -> int:
    """Pick which long-axis end of the door rectangle is the hinge.

    ``hinge_xy_m`` is the arc-detected hinge centre (from
    ``DoorPlacement.note``). We project hinge → door-centre onto the
    long axis; the sign tells which end of the opening the hinge sits
    near. Returns -1 if the hinge is closer to the -long-axis corner
    (default), +1 for the +long-axis corner.
    """
    center, long_u, _, _, _ = _polygon_basis(polygon_xy)
    delta = np.asarray(hinge_xy_m, dtype=np.float64) - center
    proj = float(delta @ long_u)
    return +1 if proj > 0.0 else -1


def compute_hinge_long_sign_from_corners(
    polygon_xy: Sequence[Tuple[float, float]],
    wall_corners_xy: Sequence[Tuple[float, float]],
    *,
    max_dist_m: float = 1.5,
    min_diff_m: float = 0.10,
):
    """Pick the hinge long-axis end by wall-corner proximity.

    Doors hinge near a wall end / T-junction / L-corner — the latch
    swings into open wall surface. Measure the distance from each
    long-axis end of the door rectangle to the nearest wall corner;
    the closer end is the hinge end.

    Returns ``+1`` / ``-1`` if the asymmetry is large enough
    (``min_diff_m``) and at least one corner is within ``max_dist_m``,
    otherwise ``None`` — caller should treat that as 'this signal
    abstains'.
    """
    if not wall_corners_xy:
        return None
    center, long_u, _, W, _ = _polygon_basis(polygon_xy)
    half_W = W * 0.5
    end_pos = center + half_W * long_u
    end_neg = center - half_W * long_u
    corners = np.asarray(wall_corners_xy, dtype=np.float64)
    if corners.ndim != 2 or corners.shape[1] < 2:
        return None
    d_pos = float(np.linalg.norm(corners - end_pos, axis=1).min())
    d_neg = float(np.linalg.norm(corners - end_neg, axis=1).min())
    if min(d_pos, d_neg) > max_dist_m:
        return None
    if abs(d_pos - d_neg) < min_diff_m:
        return None
    return +1 if d_pos < d_neg else -1


def compute_hinge_long_sign_from_pixel_centroid(
    polygon_xy: Sequence[Tuple[float, float]],
    bbox_m: Tuple[float, float, float, float],
    image_gray,
    px_to_m: float,
    *,
    dark_threshold: int = 200,
    min_pixels: int = 20,
    min_offset_frac: float = 0.05,
):
    """Pick the hinge long-axis end via dark-pixel centroid asymmetry.

    A door symbol fans out from the hinge corner: the arc's open mouth,
    the panel and the lever sit on the LATCH side. So the dark-pixel
    centroid inside the label bbox is biased toward the latch end —
    the hinge end is the OPPOSITE side.

    Returns ``+1`` / ``-1`` when the long-axis projection of the
    centroid offset from the door rectangle centre is at least
    ``min_offset_frac`` of the door width; ``None`` otherwise (signal
    abstains for symmetric symbols or empty crops).
    """
    if image_gray is None:
        return None
    H, W_px = image_gray.shape[:2]
    x0_px = max(0, int(round(bbox_m[0] / px_to_m)))
    y0_px = max(0, int(round(bbox_m[1] / px_to_m)))
    x1_px = min(W_px, int(round(bbox_m[2] / px_to_m)) + 1)
    y1_px = min(H, int(round(bbox_m[3] / px_to_m)) + 1)
    if x1_px <= x0_px or y1_px <= y0_px:
        return None
    crop = image_gray[y0_px:y1_px, x0_px:x1_px]
    mask = crop < int(dark_threshold)
    n = int(mask.sum())
    if n < int(min_pixels):
        return None
    ys, xs = np.where(mask)
    cx_m = (float(xs.mean()) + x0_px) * px_to_m
    cy_m = (float(ys.mean()) + y0_px) * px_to_m

    center, long_u, _, W, _ = _polygon_basis(polygon_xy)
    delta = np.array([cx_m - float(center[0]), cy_m - float(center[1])])
    proj = float(delta @ long_u)
    if abs(proj) < min_offset_frac * W:
        return None
    return -1 if proj > 0.0 else +1


def vote_hinge_long_sign(votes):
    """Aggregate per-signal hinge-end votes into a single sign + confidence.

    ``votes`` is a dict mapping signal names to ``+1``, ``-1`` or
    ``None``. Recognised signals (rough confidence ranking):

    * ``"arc"``      — arc-detected hinge_xy (geometric ground truth)
    * ``"panel"``    — door panel pixel-line endpoint nearest a wall
    * ``"corners"``  — wall-corner asymmetry
    * ``"centroid"`` — dark-pixel-centroid asymmetry inside the bbox

    Returns ``(chosen_sign, confidence, din_flags)``.

    * ``chosen_sign`` — ``+1`` / ``-1`` (defaults to ``-1`` if no signal
      voted, matching the legacy default).
    * ``confidence`` — ``ratio * min(1, n/3)`` where ``ratio`` = majority
      share. So 3+ unanimous signals yield 1.0; 2 unanimous → ~0.67;
      1 lone signal → ~0.33; a 1-vs-1 split → 0.33; 2-vs-1 → 0.67.
    * ``din_flags`` — ``["hinge_no_signal"]`` if nothing voted,
      ``["hinge_split_<p>p_<n>n"]`` if the vote split, otherwise empty.

    Ties are broken by signal priority (arc > panel > corners > centroid).
    """
    valid = {k: int(v) for k, v in (votes or {}).items() if v in (-1, 1)}
    n = len(valid)
    if n == 0:
        return -1, 0.0, ["hinge_no_signal"]

    pos = sum(1 for v in valid.values() if v > 0)
    neg = n - pos
    flags = []

    if pos == 0 or neg == 0:
        chosen = +1 if pos else -1
        ratio = 1.0
    else:
        if pos > neg:
            chosen = +1
            ratio = pos / n
        elif neg > pos:
            chosen = -1
            ratio = neg / n
        else:
            for sig in ("arc", "panel", "corners", "centroid"):
                if sig in valid:
                    chosen = valid[sig]
                    break
            else:
                chosen = -1
            ratio = 0.5
        flags.append(f"hinge_split_{pos}p_{neg}n")

    confidence = ratio * min(1.0, n / 3.0)
    return chosen, float(confidence), flags


def compute_pocket_sign(
    polygon_xy: Sequence[Tuple[float, float]],
    bbox_m: Tuple[float, float, float, float],
) -> int:
    """For a sliding door, pick which long-axis side is the pocket.

    The label bbox usually extends past the door rectangle on the side
    where the panel slides into — so we pick the long-axis sign of that
    extension.
    """
    center, long_u, _, _, _ = _polygon_basis(polygon_xy)
    bbox_center = np.array(
        [(bbox_m[0] + bbox_m[2]) * 0.5, (bbox_m[1] + bbox_m[3]) * 0.5]
    )
    delta = bbox_center - center
    proj = float(delta @ long_u)
    if abs(proj) < 1e-6:
        return +1
    return +1 if proj >= 0.0 else -1


def compute_mount_type(
    polygon_xy: Sequence[Tuple[float, float]],
    bbox_m: Tuple[float, float, float, float],
    wall_depth_m: float,
    *,
    surface_factor: float = 1.8,
    surface_pad_m: float = 0.08,
) -> str:
    """Heuristic: 'pocket' vs 'surface' for a sliding door.

    A pocket-sliding symbol is drawn *inside* the wall thickness — the
    label bbox extends roughly one wall-depth perpendicular to the
    wall. A surface-mounted (barn-door) symbol is drawn alongside the
    wall — the bbox extent perpendicular to the wall is roughly
    wall_depth + panel_clearance, i.e. clearly larger.

    We compare the bbox short-axis extent to ``surface_factor`` ×
    wall_depth + ``surface_pad_m`` and call it 'surface' above that
    threshold, otherwise 'pocket'.
    """
    _, long_u, short_u, _, _ = _polygon_basis(polygon_xy)
    corners = np.array([
        [bbox_m[0], bbox_m[1]],
        [bbox_m[2], bbox_m[1]],
        [bbox_m[2], bbox_m[3]],
        [bbox_m[0], bbox_m[3]],
    ])
    proj_short = corners @ short_u
    extent_short = float(proj_short.max() - proj_short.min())
    threshold = surface_factor * wall_depth_m + surface_pad_m
    return "surface" if extent_short > threshold else "pocket"


# ── Swing door (single + double) ────────────────────────────────────────


def _swing_components(placed: PlacedDoor, *, double: bool = False):
    """Return list of meshes for a swing door (jambs + head trim + leaf(s))."""
    center, long_u, short_u, W, _ = _polygon_basis(placed.polygon_xy)
    R_basis = _basis_to_3d(long_u, short_u)
    head = max(0.5, min(placed.head_height_m, 2.79))
    wall_depth = max(0.06, placed.wall_depth_m)
    frame_depth = _wall_inset_depth(wall_depth)
    leaf_thickness = min(LEAF_DEFAULT_THICKNESS_M, frame_depth - 0.005)
    leaf_thickness = max(0.015, leaf_thickness)

    # Shift the world centre by ``short_offset_m`` along the short axis so
    # the frame sits on the wall midplane (rather than on the door polygon
    # centre, which is often offset toward one wall surface).
    shifted_center = center + placed.short_offset_m * short_u
    cx, cz = float(shifted_center[0]), float(shifted_center[1])
    out = []

    # Frame extension to fully cover the boolean-cut void edges.
    # The void was extruded with VOID_INFLATE_XY_M lateral and
    # VOID_INFLATE_Y_M vertical inflation; we mirror that here so the
    # jamb's outer edge meets the wall flank seamlessly and its top
    # tucks just under the wall above (no visible 5 mm slot).
    jamb_long_extent = JAMB_WIDTH_M + VOID_INFLATE_XY_M
    jamb_long_center_abs = W * 0.5 + VOID_INFLATE_XY_M - jamb_long_extent * 0.5
    jamb_height = head + VOID_INFLATE_Y_M

    # ---- Two jambs (vertical pillars at the short ends of the void) ----
    for sign in (-1, +1):
        local_c = np.array([sign * jamb_long_center_abs,
                            jamb_height * 0.5, 0.0])
        T_world = _T([cx, 0.0, cz]) @ R_basis @ _T(local_c)
        out.append(_box(
            extents_xyz=(jamb_long_extent, jamb_height, frame_depth),
            transform=T_world,
            color=COLOR_FRAME,
        ))

    # ---- Head trim (thin band INSIDE the void at the top) ----
    # The wall above ``head`` is already the lintel — the trim is a
    # decorative band running between the jamb inner edges, with its
    # top tucked into the wall above so the wall and frame visually
    # merge (no 5 mm horizontal slot at y=head).
    trim_long_extent = W - 2 * JAMB_WIDTH_M
    trim_top = head + VOID_INFLATE_Y_M
    trim_bottom = head - HEAD_TRIM_HEIGHT_M
    trim_height_actual = trim_top - trim_bottom
    trim_y = (trim_top + trim_bottom) * 0.5
    T_trim = _T([cx, 0.0, cz]) @ R_basis @ _T([0.0, trim_y, 0.0])
    out.append(_box(
        extents_xyz=(trim_long_extent, trim_height_actual, frame_depth),
        transform=T_trim,
        color=COLOR_FRAME,
    ))

    # ---- Leaves (rotated open at SWING_ANGLE_DEG around the hinge) ----
    leaf_h = head - LEAF_BOTTOM_GAP_M - LEAF_TOP_GAP_M - HEAD_TRIM_HEIGHT_M
    if leaf_h <= 0.10:
        return out

    swing_dir = +1 if placed.swing_dir_sign >= 0 else -1
    # General rotation rule (derived from _Ry's right-handed +Y rotation
    # combined with the box-local→world basis): the leaf vector pointing
    # away from the hinge along the long-axis must rotate toward the
    # ``swing_dir`` half-space. Working it out:
    #   * hinge at -W/2 end (sign=-1): leaf points +X, target +short → R_y(-75°)
    #   * hinge at +W/2 end (sign=+1): leaf points -X, target +short → R_y(+75°)
    # which collapses to ``swing_rad = hinge_sign * swing_dir * 75°``.

    if double:
        # Two leaves, hinged at opposite ends. Each leaf swings into the
        # same ``swing_dir`` half-space.
        leaf_w = (W - 2 * JAMB_WIDTH_M) * 0.5 - 0.005
        for hinge_sign in (-1, +1):
            opening_sign = -hinge_sign
            hinge_local = np.array(
                [hinge_sign * (W * 0.5 - JAMB_WIDTH_M), 0.0, 0.0]
            )
            T_hinge_world = _T([cx, 0.0, cz]) @ R_basis @ _T(hinge_local)
            T_local = _T([
                opening_sign * leaf_w * 0.5,
                LEAF_BOTTOM_GAP_M + leaf_h * 0.5,
                0.0,
            ])
            R_swing = _Ry(math.radians(SWING_ANGLE_DEG)
                          * hinge_sign * swing_dir)
            T_world = T_hinge_world @ R_swing @ T_local
            out.append(_box(
                extents_xyz=(leaf_w, leaf_h, leaf_thickness),
                transform=T_world,
                color=COLOR_LEAF,
            ))
    else:
        hinge_sign = +1 if placed.hinge_long_sign >= 0 else -1
        opening_sign = -hinge_sign
        leaf_w = max(0.40, W - 2 * JAMB_WIDTH_M - 0.005)
        hinge_local = np.array(
            [hinge_sign * (W * 0.5 - JAMB_WIDTH_M), 0.0, 0.0]
        )
        T_hinge_world = _T([cx, 0.0, cz]) @ R_basis @ _T(hinge_local)
        T_local = _T([opening_sign * leaf_w * 0.5,
                      LEAF_BOTTOM_GAP_M + leaf_h * 0.5,
                      0.0])
        R_swing = _Ry(math.radians(SWING_ANGLE_DEG)
                      * hinge_sign * swing_dir)
        T_world = T_hinge_world @ R_swing @ T_local
        out.append(_box(
            extents_xyz=(leaf_w, leaf_h, leaf_thickness),
            transform=T_world,
            color=COLOR_LEAF,
        ))
    return out


# ── Sliding door ────────────────────────────────────────────────────────


def _sliding_components(
    placed: PlacedDoor,
    *,
    position_override: Optional[str] = None,
):
    """Return list of meshes for a sliding door (track + panel).

    ``position_override`` lets the caller force a panel position
    independent of ``placed.mount_type`` — used by the BIM viewer's
    3-variant output ("standard / beside left / beside right") so the
    user can pick the rendering that matches their intent without
    re-running the classifier:

      * ``"in_wall"``        → mount=pocket (panel inside wall thickness)
      * ``"beside_left"``    → mount=surface, surface_sign=-1
      * ``"beside_right"``   → mount=surface, surface_sign=+1
      * ``None`` (default)   → respect ``placed.mount_type`` /
                               ``placed.swing_dir_sign``
    """
    center, long_u, short_u, W, _ = _polygon_basis(placed.polygon_xy)
    R_basis = _basis_to_3d(long_u, short_u)
    head = max(0.5, min(placed.head_height_m, 2.79))
    wall_depth = max(0.06, placed.wall_depth_m)

    pocket_sign = +1 if placed.pocket_sign >= 0 else -1
    surface_sign = +1 if placed.swing_dir_sign >= 0 else -1
    mount = (placed.mount_type or "pocket").lower()
    if mount not in ("pocket", "surface", "beside"):
        mount = "pocket"

    if position_override == "in_wall":
        mount = "pocket"
    elif position_override == "beside_left":
        mount = "surface"
        surface_sign = -1
    elif position_override == "beside_right":
        mount = "surface"
        surface_sign = +1

    # Shift onto the wall midplane (matches _swing_components).
    shifted_center = center + placed.short_offset_m * short_u
    cx, cz = float(shifted_center[0]), float(shifted_center[1])
    out = []

    # ---- Track rail just below the head (inside the door void) ----
    # The wall above ``head`` is now closed by the lintel filler
    # (door_models.build_door_lintels). If we placed the track at
    # head+halfH it would sit *inside* the lintel and become invisible.
    # So we drop the track into the upper edge of the door opening so
    # it stays visible through the void from both sides while the
    # lintel still bridges the wall above the door fully.
    # Tag-5 viewer-variants: when a position_override forces beside_left
    # / beside_right, the WHOLE slider (rail + panel) must sit OUTSIDE
    # the wall on the chosen side so the user can see at a glance
    # which mode applies. The legacy "surface" mount kept the rail
    # mostly inside the wall thickness — fine for true barn-door
    # geometry but unreadable for the 3-way comparison view.
    full_outside_offset = position_override in ("beside_left", "beside_right")
    if mount == "pocket":
        # Rail recessed inside the wall depth — no protrusion.
        rail_depth = _wall_inset_depth(wall_depth, min_depth_m=0.04)
        rail_short_off = 0.0
    elif mount == "beside":
        # Beside-wall (barn-door drawn next to the wall, not in it):
        # rail sits at the polygon centre — the polygon is already at
        # the bbox centre beside the wall, no extra short-offset needed.
        rail_depth = max(SLIDE_PANEL_THICKNESS_M * 1.5, 0.06)
        rail_short_off = 0.0
    elif full_outside_offset:
        # Tag-5 viewer-variant: rail fully outside the wall.
        rail_depth = max(SLIDE_PANEL_THICKNESS_M * 1.5, 0.06)
        rail_short_off = surface_sign * (
            wall_depth * 0.5 + SLIDE_SURFACE_GAP_M + rail_depth * 0.5
        )
    else:
        # Surface-mounted: rail rides just outside the chosen wall face.
        # Total rail depth equals wall depth so it visually anchors the
        # panel without exceeding the wall thickness on the *other* side.
        rail_depth = _wall_inset_depth(wall_depth, min_depth_m=0.04)
        # Centre the rail on the panel's offset side.
        rail_short_off = surface_sign * (
            wall_depth * 0.5 - rail_depth * 0.5 + SLIDE_SURFACE_GAP_M
        )

    track_y = head - SLIDE_TRACK_HEIGHT_M * 0.5 - 0.005
    track_len = W + SLIDE_TRACK_OVERHANG_M
    track_long_off = pocket_sign * SLIDE_TRACK_OVERHANG_M * 0.5
    T_track = (
        _T([cx, 0.0, cz])
        @ R_basis
        @ _T([track_long_off, track_y, rail_short_off])
    )
    out.append(_box(
        extents_xyz=(track_len, SLIDE_TRACK_HEIGHT_M, rail_depth),
        transform=T_track,
        color=COLOR_TRACK,
    ))

    # ---- Sliding panel ----
    # Tag-5+: panel sits at the polygon centre along the long axis,
    # i.e. at the bbox-centre projected onto the wall — same place the
    # user drew the label. Closed position. The 3-mode toggle moves
    # the panel along the SHORT axis only (in / out of wall), the
    # long-axis position stays anchored to the label.
    panel_h = head - LEAF_BOTTOM_GAP_M - 0.04
    panel_w = max(0.40, W - 0.01)
    long_shift = 0.0

    if mount == "pocket":
        # Panel inside the wall thickness (centred on the wall axis).
        panel_thickness = min(SLIDE_PANEL_THICKNESS_M,
                              wall_depth - 2 * SLIDE_PANEL_INSET_M)
        panel_thickness = max(0.015, panel_thickness)
        short_shift = 0.0
        panel_color = COLOR_PANEL_GLASS
    elif mount == "beside":
        # Beside-wall: panel sits at the polygon centre. The polygon was
        # built at the bbox centre beside the wall, so no shift along
        # short_u is required. Distance to the wall comes from where
        # the user drew the bbox.
        panel_thickness = SLIDE_PANEL_THICKNESS_M
        short_shift = 0.0
        panel_color = COLOR_PANEL_GLASS
    else:
        # Surface-mounted (barn-door): panel rides just outside the wall.
        panel_thickness = SLIDE_PANEL_THICKNESS_M
        short_shift = surface_sign * (
            wall_depth * 0.5 + SLIDE_SURFACE_GAP_M + panel_thickness * 0.5
        )
        panel_color = COLOR_PANEL_GLASS

    T_panel = (
        _T([cx, 0.0, cz])
        @ R_basis
        @ _T([long_shift,
              LEAF_BOTTOM_GAP_M + panel_h * 0.5,
              short_shift])
    )
    out.append(_box(
        extents_xyz=(panel_w, panel_h, panel_thickness),
        transform=T_panel,
        color=panel_color,
    ))

    return out


# ── Public API ──────────────────────────────────────────────────────────


def build_door_meshes(
    placements: Sequence[PlacedDoor],
    *,
    mode: str = "auto",
    sliding_variant: Optional[str] = None,
):
    """Build a single concatenated trimesh containing every door's geometry.

    ``mode``:
      * ``"auto"``   — render each door according to its label_class
        (door_sliding → sliding, others → swing).
      * ``"swing"``  — render every door as a swing door regardless of class.
      * ``"sliding"``— render every door as a sliding door regardless of class.

    ``sliding_variant`` overrides the panel position for any sliding-
    rendered door. Forwards directly to ``_sliding_components`` —
    ``"in_wall"`` / ``"beside_left"`` / ``"beside_right"``. ``None``
    respects each placement's individual mount.
    """
    import trimesh

    parts = []
    for p in placements:
        if len(p.polygon_xy) < 4:
            continue
        cls = (p.label_class or "").lower()
        effective = mode
        if mode == "auto":
            effective = "sliding" if cls == "door_sliding" else "swing"
        try:
            if effective == "sliding":
                parts.extend(_sliding_components(
                    p, position_override=sliding_variant,
                ))
            else:
                double = (cls == "door_double")
                parts.extend(_swing_components(p, double=double))
        except Exception as exc:
            print(
                f"  [door-models] {p.label_id} ({cls}) geometry failed: {exc}"
            )

    if not parts:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(parts)


def build_door_meshes_grouped(
    placements: Sequence[PlacedDoor],
    *,
    mode: str = "auto",
    sliding_variant: Optional[str] = None,
) -> List[Tuple[PlacedDoor, "object"]]:
    """Like :func:`build_door_meshes`, but returns ONE concatenated mesh PER door.

    Used to export each door as its own named glTF node (so walls, doors and
    frames end up as individually selectable/editable parts in the BIM editor)
    instead of merging every door into a single buffer. Mirrors the per-door
    branching of :func:`build_door_meshes` exactly — same ``mode`` /
    ``sliding_variant`` semantics — but keeps the doors apart.
    """
    import trimesh

    out: List[Tuple[PlacedDoor, object]] = []
    for p in placements:
        if len(p.polygon_xy) < 4:
            continue
        cls = (p.label_class or "").lower()
        effective = mode
        if mode == "auto":
            effective = "sliding" if cls == "door_sliding" else "swing"
        try:
            if effective == "sliding":
                parts = _sliding_components(p, position_override=sliding_variant)
            else:
                double = cls == "door_double"
                parts = _swing_components(p, double=double)
        except Exception as exc:
            print(f"  [door-models] {p.label_id} ({cls}) geometry failed: {exc}")
            continue
        parts = [m for m in parts if hasattr(m, "vertices") and len(m.vertices)]
        if not parts:
            continue
        mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
        out.append((p, mesh))
    return out


def filter_for_class(
    placements: Sequence[PlacedDoor],
    keep_classes: Sequence[str],
) -> List[PlacedDoor]:
    """Subset placements by label_class."""
    keep = {c.lower() for c in keep_classes}
    return [p for p in placements if (p.label_class or "").lower() in keep]


# ── Lintel filler (Tag-7.4) ─────────────────────────────────────────────


def _user_wall_continuous_extent(
    edge_xy,
    sign: int,
    long_u,
    short_u,
    user_wall_shapely,
    *,
    wall_depth_m: float,
    max_extent_m: float = 2.0,
    sample_step_m: float = 0.020,
    gap_tolerance_m: float = 0.030,
) -> float:
    """Length of continuous user-drawn wall material starting at ``edge_xy``
    extending in the ``sign * long_u`` direction.

    Used to cap the pocket-side bridge wing for sliding doors so we
    never bridge past where the annotator drew a wall. Samples a thin
    box (``sample_step × wall_depth``) every ``sample_step_m`` along
    the ray and counts steps that overlap any user-wall polygon by
    ≥ 10 % of the box area. Tolerates gaps up to ``gap_tolerance_m``.
    """
    from shapely.geometry import Polygon  # noqa: PLC0415

    if not user_wall_shapely:
        return 0.0
    half_d = max(0.05, wall_depth_m) * 0.5
    edge_arr = np.asarray(edge_xy, dtype=np.float64)
    long_arr = np.asarray(long_u, dtype=np.float64)
    short_arr = np.asarray(short_u, dtype=np.float64)

    total = 0.0
    gap_running = 0.0
    n_samples = max(1, int(round(max_extent_m / sample_step_m)))
    for i in range(n_samples):
        d_center = (i + 0.5) * sample_step_m
        sample_center = edge_arr + sign * d_center * long_arr
        half_step = sample_step_m * 0.5
        corners = [
            sample_center + half_step * long_arr + half_d * short_arr,
            sample_center + half_step * long_arr - half_d * short_arr,
            sample_center - half_step * long_arr - half_d * short_arr,
            sample_center - half_step * long_arr + half_d * short_arr,
        ]
        coords = [(float(c[0]), float(c[1])) for c in corners]
        try:
            box = Polygon(coords).buffer(0)
        except Exception:
            continue
        if box.is_empty or box.area <= 0:
            continue
        has_wall = False
        for wp in user_wall_shapely:
            try:
                if not box.intersects(wp):
                    continue
                inter = box.intersection(wp)
                if inter.is_empty:
                    continue
                if inter.area > 0.10 * box.area:
                    has_wall = True
                    break
            except Exception:
                continue
        if has_wall:
            if gap_running > gap_tolerance_m:
                break
            total += sample_step_m + gap_running
            gap_running = 0.0
        else:
            gap_running += sample_step_m
            if gap_running > gap_tolerance_m:
                break
    return min(total, max_extent_m)


def build_door_wall_bridges(
    placements: Sequence[PlacedDoor],
    *,
    wall_height_m: float = 2.80,
    flip_y_for_pdf: bool = True,
    full_height: bool = True,
    wall_polygons_xy: "Sequence[Sequence[Tuple[float, float]]]" = None,
    user_wall_polygons_xy: "Sequence[Sequence[Tuple[float, float]]]" = None,
    wing_probe_buffer_m: float = 0.05,
):
    """Build a wall-coloured box per door that fills the wall material
    around every door opening.

    The wall mask used to extrude the wall mesh has a *full-height* gap
    at every door (doors appear as breaks in the floor-plan wall
    outline), and the wall annotations themselves are usually drawn as
    separate rectangles per wall segment that don't quite reach the
    door endpoints. As a result, after extrusion the door zone is empty
    from floor to wall_top, *and* the surrounding wall flanks may not
    cover the door's full extent.

    Two modes:
      * ``full_height=True`` (default, Tag-7.6): bridge spans floor to
        wall_top. Concat into the wall mesh BEFORE cutting door voids
        — the void cut then carves out the door opening (floor to
        head_height) and leaves the lintel zone (head to wall_top) as
        wall material. The bottom flanks are also filled, providing a
        solid wall flank for the door frame to sit in even when the
        original wall annotation falls short.
      * ``full_height=False`` (legacy Tag-7.4 lintel-only): box runs
        from ``head + VOID_INFLATE_Y_M`` to wall_top. Concat AFTER the
        void cut. Used by callers that want only the lintel, not the
        full door bridge.

    Geometry per box:
      * Long axis: ``W + 2 * LINTEL_LONG_OVERLAP_M`` so the bridge
        reaches well into the wall flanks on each side.
      * Short axis: ``placed.wall_depth_m`` (actual wall thickness
        measured perpendicular to the door at this point), centred on
        the wall midplane via ``placed.short_offset_m``.

    Returns a single concatenated trimesh ready to be merged with the
    wall mesh and tinted alongside it. The boxes themselves are
    untinted; the caller applies the wall colour after concatenation.
    """
    import trimesh
    from shapely.geometry import Polygon

    # Pre-build shapely wall polygons (in plan metres, NO flip — same
    # coordinate system as door polygon_xy). Used to test whether each
    # long-axis wing has actual wall material to anchor it; without
    # this the +50 mm wing on the unsupported end of an end-of-wall
    # door (e.g. closet sliders) extends as a floating stub past the
    # original wall outline.
    wall_shapely = []
    if wall_polygons_xy:
        for shell in wall_polygons_xy:
            if len(shell) < 3:
                continue
            try:
                wp = Polygon(shell).buffer(0)
            except Exception:
                continue
            if wp.is_empty or not wp.is_valid:
                continue
            wall_shapely.append(wp)

    # Tag-9.5: user-wall polygons (raw annotations) used to extend pocket-
    # side bridge wings to the panel-slide distance. Pocket sliding doors
    # need wall material covering the full slide distance to hide the
    # panel; the post-refine BIM walls may have carve-out gaps from door-
    # subtraction or authority-reclip that are smaller than the user
    # actually drew. Capping wing extension by user-wall extent prevents
    # phantom walls past where the user said the wall ends.
    user_wall_shapely = []
    if user_wall_polygons_xy:
        for shell in user_wall_polygons_xy:
            if len(shell) < 3:
                continue
            try:
                wp = Polygon(shell).buffer(0)
            except Exception:
                continue
            if wp.is_empty or not wp.is_valid:
                continue
            user_wall_shapely.append(wp)

    parts = []
    for p in placements:
        if len(p.polygon_xy) < 4:
            continue
        try:
            center, long_u, short_u, W, _ = _polygon_basis(p.polygon_xy)
        except Exception:
            continue

        if full_height:
            base_y = 0.0
            top_y = wall_height_m
        else:
            head = max(0.5, min(p.head_height_m, wall_height_m - 0.02))
            base_y = head + VOID_INFLATE_Y_M
            top_y = wall_height_m
        height = top_y - base_y
        if height <= 1e-3:
            continue

        new_center = center + p.short_offset_m * short_u
        half_d = max(0.05, p.wall_depth_m) * 0.5

        # Tag-9.3: extend each wing all the way to the next wall_polygon
        # along ``long_u``, so the bridge fuses seamlessly with adjacent
        # wall segments instead of stopping at a fixed 50 mm overlap. The
        # previous logic used a 100 mm probe and set wings to 0 if no wall
        # was inside that probe, leaving lintels floating mid-air whenever
        # the next wall annotation was further away (sliding-door pockets,
        # T-junctions where the perpendicular wall is the thing it should
        # reach to). The new probe is a full WING_MAX_M-long rectangle of
        # wall-depth thickness; we measure the actual distance to the
        # nearest wall intersection and extend the wing to that distance
        # plus a small overlap so trimesh.boolean.union fuses cleanly.
        WING_MAX_M = 2.0
        WING_FUSE_OVERLAP_M = 0.030
        wings = {}  # sign -> overlap in metres
        for sign in (-1, +1):
            edge_center = new_center + sign * (W * 0.5) * long_u
            if not wall_shapely:
                # No wall polygons supplied — fall back to symmetric
                # default wing so callers without wall context still
                # get a stitchable lintel.
                wings[sign] = LINTEL_LONG_OVERLAP_M
                continue
            far_end = edge_center + sign * WING_MAX_M * long_u
            corners_probe = [
                edge_center + half_d * short_u,
                far_end + half_d * short_u,
                far_end - half_d * short_u,
                edge_center - half_d * short_u,
            ]
            probe_coords = [(float(c[0]), float(c[1])) for c in corners_probe]
            try:
                probe = Polygon(probe_coords).buffer(0)
            except Exception:
                wings[sign] = LINTEL_LONG_OVERLAP_M
                continue
            if probe.is_empty or not probe.is_valid:
                wings[sign] = 0.0
                continue
            # Find the nearest wall intersection along sign*long_u.
            nearest_dist_m = WING_MAX_M
            edge_xy = np.array([edge_center[0], edge_center[1]])
            for wp in wall_shapely:
                inter = probe.intersection(wp)
                if inter.is_empty:
                    continue
                # Iterate vertices of the intersection geometry; for each
                # find the signed distance from edge_center along long_u
                # and keep the smallest positive value.
                geoms = (
                    list(inter.geoms) if hasattr(inter, "geoms") else [inter]
                )
                for g in geoms:
                    if g.is_empty:
                        continue
                    coords_iter = []
                    if hasattr(g, "exterior"):
                        coords_iter = list(g.exterior.coords)
                    elif hasattr(g, "coords"):
                        coords_iter = list(g.coords)
                    for c in coords_iter:
                        rel = np.array([c[0], c[1]]) - edge_xy
                        d_long = sign * float(rel @ long_u)
                        if 0.0 <= d_long < nearest_dist_m:
                            nearest_dist_m = d_long
            if nearest_dist_m >= WING_MAX_M:
                # No wall within 2 m on this side — leave end open.
                wings[sign] = 0.0
            else:
                # Extend wing to the wall + small fusion overlap. Always
                # at least LINTEL_LONG_OVERLAP_M so the merge is robust
                # even when the next wall is essentially touching.
                wings[sign] = max(
                    LINTEL_LONG_OVERLAP_M,
                    nearest_dist_m + WING_FUSE_OVERLAP_M,
                )

        # Tag-9.5: pocket-sliding-door wing extension on the pocket side.
        # The panel slides ``W * SLIDE_OPEN_FRACTION`` toward ``pocket_sign``
        # — for the panel to be hidden inside the wall when open, the
        # bridge must cover that distance. The post-refine BIM walls
        # often have carve-out gaps right past the door (door subtraction
        # + authority-reclip) that are smaller than what the user
        # actually drew; the wing-to-nearest-BIM-wall logic above stops
        # at the first gap and leaves the panel exposed. We override the
        # pocket-side wing to ``min(panel_slide, user_wall_extent)``,
        # capped by the user-drawn wall extent so we never bridge past
        # where the user said the wall ends.
        is_sliding = (
            getattr(p, "label_class", "") or ""
        ).lower() == "door_sliding"
        is_pocket = getattr(p, "mount_type", "pocket") == "pocket"
        pocket_sign = int(getattr(p, "pocket_sign", +1))
        if is_sliding and is_pocket and user_wall_shapely:
            panel_slide_m = W * SLIDE_OPEN_FRACTION
            edge_center_pocket = (
                new_center + pocket_sign * (W * 0.5) * long_u
            )
            user_wall_extent = _user_wall_continuous_extent(
                edge_center_pocket,
                pocket_sign,
                long_u,
                short_u,
                user_wall_shapely,
                wall_depth_m=p.wall_depth_m,
                max_extent_m=WING_MAX_M,
            )
            target_wing = min(panel_slide_m, user_wall_extent)
            # Take the larger of the existing wing (BIM-wall fusion) and
            # the pocket target, so we never shrink below fusion needs.
            current_wing = wings.get(pocket_sign, 0.0)
            if target_wing > current_wing:
                wings[pocket_sign] = target_wing

        # Phase E.1 phantom-bridge guard: if neither end is anchored to
        # a wall AND there is no wall material under the door body
        # itself, this door sits in free space (e.g. sliding-door
        # annotation drawn over a non-existent wall, or a labelling
        # mistake). A full-height bridge here would emit a phantom
        # lintel "wall" that is not in the drawing — exactly the
        # artefact Willi reports as "wall over the door connecting two
        # walls that aren't actually connected". Skip the bridge.
        if (
            wall_shapely
            and wings[-1] == 0.0
            and wings[+1] == 0.0
        ):
            half_d_test = max(0.05, p.wall_depth_m) * 0.5
            body_corners = [
                new_center + (W * 0.5) * long_u + half_d_test * short_u,
                new_center - (W * 0.5) * long_u + half_d_test * short_u,
                new_center - (W * 0.5) * long_u - half_d_test * short_u,
                new_center + (W * 0.5) * long_u - half_d_test * short_u,
            ]
            body_coords = [(float(c[0]), float(c[1])) for c in body_corners]
            try:
                body_poly = Polygon(body_coords).buffer(0)
            except Exception:
                body_poly = None
            body_anchored = (
                body_poly is not None
                and not body_poly.is_empty
                and any(body_poly.intersects(wp) for wp in wall_shapely)
            )
            if not body_anchored:
                continue

        half_w_minus = W * 0.5 + wings[-1]
        half_w_plus = W * 0.5 + wings[+1]

        # Rectangular bridge (Willi 2026-05-01 — reverted from
        # trapezoidal endcap snap): both long-axis endcaps use the
        # same ``half_d = wall_depth_m * 0.5``. The trapezoidal endcap
        # snap was producing visibly wrong shapes when one wing
        # extended into a perpendicular wall at a T-junction (the
        # endcap then snapped to the *perpendicular* wall's extents,
        # not the door's wall). Constant half_d keeps the bridge a
        # clean rectangle aligned with the door's measured wall depth;
        # the boolean.union step downstream still fuses cleanly with
        # any neighbouring walls of different thickness.
        edge_minus = new_center - half_w_minus * long_u
        edge_plus = new_center + half_w_plus * long_u

        corners = [
            edge_plus + half_d * short_u,
            edge_minus + half_d * short_u,
            edge_minus - half_d * short_u,
            edge_plus - half_d * short_u,
        ]
        coords = [(float(c[0]), float(c[1])) for c in corners]
        if flip_y_for_pdf:
            coords = [(x, -y) for x, y in coords]

        poly = Polygon(coords).buffer(0)
        if poly.is_empty or not poly.is_valid:
            continue
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)

        try:
            box = trimesh.creation.extrude_polygon(poly, height)
        except Exception:
            continue
        # extrude_polygon builds along +Z; rotate so height runs +Y.
        R = trimesh.transformations.rotation_matrix(
            -math.pi / 2.0, [1, 0, 0]
        )
        box.apply_transform(R)
        box.apply_translation([0.0, base_y, 0.0])
        parts.append(box)

    if not parts:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(parts)


def compute_bridge_polygons_m(
    placements: Sequence[PlacedDoor],
    *,
    wall_height_m: float = 2.80,
    full_height: bool = False,
    wall_polygons_xy: "Sequence[Sequence[Tuple[float, float]]]" = None,
    user_wall_polygons_xy: "Sequence[Sequence[Tuple[float, float]]]" = None,
) -> "List[Tuple[str, str, List[Tuple[float, float]]]]":
    """Return the 2D plan-metre footprint of the bridge for every door.

    Mirrors ``build_door_wall_bridges`` but yields the plan polygons
    (NOT y-flipped) instead of an extruded mesh — useful for 2D
    diagnostic renders that show the wall/lintel layout without
    rebuilding the geometry by hand.

    Each entry is ``(label_id, label_class, polygon_xy_m)``. Doors that
    the bridge builder skipped (phantom-bridge guard, degenerate
    polygon) do not appear in the output.
    """
    from shapely.geometry import MultiPoint  # noqa: PLC0415

    out: List[Tuple[str, str, List[Tuple[float, float]]]] = []
    for p in placements:
        if len(p.polygon_xy) < 4:
            continue
        try:
            mesh = build_door_wall_bridges(
                [p],
                wall_height_m=wall_height_m,
                full_height=full_height,
                wall_polygons_xy=wall_polygons_xy,
                user_wall_polygons_xy=user_wall_polygons_xy,
                flip_y_for_pdf=False,
            )
        except Exception:
            continue
        if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            continue
        v = np.asarray(mesh.vertices, dtype=np.float64)
        floor_y = float(v[:, 1].min())
        floor_pts = v[np.abs(v[:, 1] - floor_y) < 1e-6]
        if len(floor_pts) < 3:
            continue
        # extrude_polygon + rotate(-π/2, X): scene_X=plan_X, scene_Z=-plan_Y.
        xy = [(float(pt[0]), float(-pt[2])) for pt in floor_pts]
        hull = MultiPoint(xy).convex_hull
        if hull.is_empty or not hasattr(hull, "exterior"):
            continue
        poly_xy = [
            (float(c[0]), float(c[1]))
            for c in list(hull.exterior.coords)[:-1]
        ]
        out.append((p.label_id or "", p.label_class or "", poly_xy))
    return out


# Backwards-compatible alias for callers that still ask for the
# lintel-only form (Tag-7.4 API).
def build_door_lintels(
    placements: Sequence[PlacedDoor],
    *,
    wall_height_m: float = 2.80,
    flip_y_for_pdf: bool = True,
):
    return build_door_wall_bridges(
        placements,
        wall_height_m=wall_height_m,
        flip_y_for_pdf=flip_y_for_pdf,
        full_height=False,
    )


