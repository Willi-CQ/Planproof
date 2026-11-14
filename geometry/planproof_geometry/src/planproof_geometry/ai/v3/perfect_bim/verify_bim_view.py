"""Verification harness for the BIM viewer.

Runs invariant checks on the door geometry built by
``door_models.build_door_meshes`` and renders 2D top-down + side-view
PNGs so the result can be inspected without a 3D viewer.

Invariants (all in metres, with a 5 mm tolerance):
  1. Frame depth ≤ wall depth — every jamb / head trim / pocket-track /
     pocket-panel must stay within ±wall_depth/2 perpendicular to the
     wall.
  2. Swing leaf side — leaf centroid must lie on the +short_axis side
     when ``swing_dir_sign>=0`` and on the -short_axis side otherwise.
  3. Sliding panel position — pocket panels stay within wall depth;
     surface panels sit on ``swing_dir_sign`` side, just outside the
     wall (intentional protrusion).
  4. Frame fully covers void — jamb extents must reach to at least
     ±(W/2 + 5 mm) along the wall (matching the boolean void) and to
     y = head + 5 mm vertically.

Y-Slice helper (Tag 1 of Window-BIM-Fix sprint):
  ``render_bim_y_slice_png`` cuts the BIM mesh at a given world height
  (Y axis) and rasterises the resulting 2D polygons. Used to verify
  that wall material remains in the parapet (y≈0.5 m) and lintel
  (y≈2.4 m) zones above and below window glazing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from planproof_geometry.ai.v3.pixel_bim.openings.door_models import (
    PlacedDoor,
    VOID_INFLATE_XY_M,
    VOID_INFLATE_Y_M,
    _polygon_basis,
    _sliding_components,
    _swing_components,
)
from planproof_geometry.ai.v3.pixel_bim.openings.wall_axis import (  # noqa: F401
    measure_actual_wall_thickness,
    measure_wall_axis_at_door,
)


TOLERANCE_M = 0.005


HINGE_CONFIDENCE_LOW = 0.50  # Tag-7.7 — flag hinge votes below this


@dataclass
class DoorReport:
    label_id: str
    label_class: str
    issues: List[str]
    swing_dir_sign: int
    swing_dir_actual: int
    wall_depth_m: float
    max_short_extent_m: float       # max |proj on short| across in-wall comps
    leaf_centroid_short_m: float    # signed short-projection of leaf centroid
    hinge_confidence: float = 1.0   # Tag-7.7 — confidence of hinge_long_sign


def _component_labels(label_class: str, n: int) -> List[str]:
    cls = (label_class or "").lower()
    if cls == "door_sliding":
        # _sliding_components order: [track, panel]
        return ["track", "panel"][:n]
    if cls == "door_double":
        # [jamb-, jamb+, trim, leaf-1, leaf-2]
        names = ["jamb-", "jamb+", "trim", "leaf-1", "leaf-2"]
        return (names + [f"extra-{i}" for i in range(n - len(names))])[:n]
    # Single swing: [jamb-, jamb+, trim, leaf]
    names = ["jamb-", "jamb+", "trim", "leaf"]
    return (names + [f"extra-{i}" for i in range(n - len(names))])[:n]


def verify_door(placed: PlacedDoor) -> DoorReport:
    cls = (placed.label_class or "").lower()
    issues: List[str] = []
    if cls == "door_sliding":
        comps = _sliding_components(placed)
    else:
        comps = _swing_components(placed, double=(cls == "door_double"))

    center, long_u, short_u, W, _ = _polygon_basis(placed.polygon_xy)
    # The geometry is built around the wall midplane (=
    # door_polygon centre + short_offset_m * short_axis), so we project
    # against THAT centre — otherwise frame components legitimately on
    # the wall midplane look 'too far' from the door polygon centre.
    shifted_center = center + placed.short_offset_m * short_u
    door_center_3d = np.array([shifted_center[0], 0.0, shifted_center[1]])
    short_3d = np.array([short_u[0], 0.0, short_u[1]])
    long_3d = np.array([long_u[0], 0.0, long_u[1]])

    wall_d = max(0.06, placed.wall_depth_m)
    half_wall = wall_d * 0.5
    expected_swing_sign = +1 if placed.swing_dir_sign >= 0 else -1

    labels = _component_labels(placed.label_class, len(comps))
    max_short_extent = 0.0
    leaf_centroid_short = 0.0
    swing_dir_actual = 0

    mount = (placed.mount_type or "pocket").lower()

    for comp, name in zip(comps, labels):
        verts = np.asarray(comp.vertices)
        rel = verts - door_center_3d
        proj_short = rel @ short_3d
        proj_long = rel @ long_3d
        proj_y = verts[:, 1]

        s_min, s_max = float(proj_short.min()), float(proj_short.max())
        l_min, l_max = float(proj_long.min()), float(proj_long.max())
        y_max = float(proj_y.max())
        s_abs = max(abs(s_min), abs(s_max))

        in_wall = name in ("jamb-", "jamb+", "trim", "track") or (
            name == "panel" and mount == "pocket"
        )

        if in_wall and s_abs > half_wall + TOLERANCE_M:
            issues.append(
                f"{name}: short-extent {s_abs*1000:.1f} mm > "
                f"wall/2 {half_wall*1000:.1f} mm (+tol {TOLERANCE_M*1000:.0f} mm)"
            )
            max_short_extent = max(max_short_extent, s_abs)
        elif in_wall:
            max_short_extent = max(max_short_extent, s_abs)

        # Frame coverage: jambs reach ±(W/2 + inflate); trim top reaches
        # head + inflate.
        if name == "jamb-":
            expected_outer = -(W * 0.5 + VOID_INFLATE_XY_M)
            if l_min > expected_outer + TOLERANCE_M:
                issues.append(
                    f"jamb- outer edge at long={l_min:+.3f} m short of "
                    f"expected {expected_outer:+.3f} m (5 mm void seam)"
                )
            if y_max < placed.head_height_m + VOID_INFLATE_Y_M - TOLERANCE_M:
                issues.append(
                    f"jamb- top at y={y_max:.3f} m short of "
                    f"expected {placed.head_height_m + VOID_INFLATE_Y_M:.3f} m"
                )
        if name == "jamb+":
            expected_outer = +(W * 0.5 + VOID_INFLATE_XY_M)
            if l_max < expected_outer - TOLERANCE_M:
                issues.append(
                    f"jamb+ outer edge at long={l_max:+.3f} m short of "
                    f"expected {expected_outer:+.3f} m"
                )
        if name == "trim":
            if y_max < placed.head_height_m + VOID_INFLATE_Y_M - TOLERANCE_M:
                issues.append(
                    f"trim top at y={y_max:.3f} m short of expected "
                    f"{placed.head_height_m + VOID_INFLATE_Y_M:.3f} m"
                )

        # Swing leaf must hinge from the configured corner of the
        # opening (single-leaf only — double doors hinge at both ends).
        if name == "leaf":
            l_center_sign = +1 if (l_min + l_max) > 0.0 else -1
            if l_center_sign != placed.hinge_long_sign:
                issues.append(
                    f"leaf hinges at wrong end: leaf long-centre "
                    f"sign={l_center_sign:+d} but hinge_long_sign="
                    f"{placed.hinge_long_sign:+d}"
                )

        # Swing leaf must end up on the room side per swing_dir_sign.
        if name in ("leaf", "leaf-1", "leaf-2"):
            centroid = verts.mean(axis=0)
            rel_centroid = centroid - door_center_3d
            proj_short_centroid = float(rel_centroid @ short_3d)
            leaf_centroid_short = proj_short_centroid
            if abs(proj_short_centroid) < 0.05:
                issues.append(
                    f"{name}: centroid short={proj_short_centroid:.3f} m "
                    f"is near zero — leaf appears not swung"
                )
                continue
            actual_sign = +1 if proj_short_centroid > 0.0 else -1
            swing_dir_actual = actual_sign
            if actual_sign != expected_swing_sign:
                issues.append(
                    f"{name}: swung wrong side — expected sign "
                    f"{expected_swing_sign:+d}, got {actual_sign:+d} "
                    f"(centroid short={proj_short_centroid:+.3f} m)"
                )

    # Tag-7.7 — flag low-confidence hinge selections so the reviewer
    # knows that the leaf orientation is tentative even when the
    # geometry-vs-hinge_long_sign self-consistency check passes. Only
    # relevant for single-leaf swings (double doors hinge at both ends,
    # sliding doors don't hinge at all).
    if (
        cls not in ("door_double", "door_sliding")
        and placed.hinge_confidence < HINGE_CONFIDENCE_LOW
    ):
        issues.append(
            f"hinge_long_sign={placed.hinge_long_sign:+d} has low "
            f"confidence {placed.hinge_confidence:.2f} "
            f"(< {HINGE_CONFIDENCE_LOW:.2f}) — leaf side may be wrong"
        )

    return DoorReport(
        label_id=placed.label_id,
        label_class=placed.label_class,
        issues=issues,
        swing_dir_sign=expected_swing_sign,
        swing_dir_actual=swing_dir_actual,
        wall_depth_m=placed.wall_depth_m,
        max_short_extent_m=max_short_extent,
        leaf_centroid_short_m=leaf_centroid_short,
        hinge_confidence=placed.hinge_confidence,
    )


def render_topdown_png(
    placed_doors: Sequence[PlacedDoor],
    wall_polygons,
    out_path: Path,
    *,
    label_bboxes_m: Sequence[Tuple[float, float, float, float]] = (),
) -> None:
    """Render a top-down (XY) PNG showing wall polygons + every door's
    component footprints.

    The view is "look down +Y" so the X axis is horizontal and the
    plan-Y axis is vertical (image-down to match PDF coords). Each
    component is drawn as the convex hull of its mesh vertices
    projected to the XZ plane, then mapped back to plan-Y by negating
    Z (matching the flip_y_for_pdf convention).
    """
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon as ShPoly

    fig, ax = plt.subplots(figsize=(14, 14))

    # Walls (light grey).
    for wp in wall_polygons:
        if len(wp.shell_xy) < 3:
            continue
        xy = np.array(wp.shell_xy)
        ax.fill(xy[:, 0], xy[:, 1], facecolor="#cfd2d8",
                edgecolor="#8b8e94", linewidth=0.4)
        for h in wp.holes_xy:
            if len(h) >= 3:
                hxy = np.array(h)
                ax.fill(hxy[:, 0], hxy[:, 1], facecolor="white",
                        edgecolor="#8b8e94", linewidth=0.3)

    # Tag-8: punch a white "door void" through the wall fill for every
    # placed door. The wall_polygons are the raw Phase-2 BIM walls
    # (which cover sliding-door zones since the annotations are no
    # longer subtracted from the wall_mask), so without this overlay
    # every door looks "filled with wall" in the top-down view —
    # exactly Willi's complaint. Drawing the door rectangle in white
    # here matches the 3D mesh state (head-tall void carved through
    # the wall material for every in-wall placement).
    for placed in placed_doors:
        if (
            getattr(placed, "sliding_mount_position", None) == "beside_wall"
        ):
            continue
        if len(placed.polygon_xy) < 4:
            continue
        try:
            from planproof_geometry.ai.v3.pixel_bim.openings.door_models import (
                build_door_rectangle,
            )
            void_xy = np.array(build_door_rectangle(placed))
        except Exception:
            void_xy = np.array(placed.polygon_xy)
        ax.fill(void_xy[:, 0], void_xy[:, 1],
                facecolor="white", edgecolor="none", zorder=2)

    # Door label bboxes (thin blue outline, for context).
    for bb in label_bboxes_m:
        x0, y0, x1, y1 = bb
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                color="#3370c0", linewidth=0.5, linestyle=":")

    # Per-door components.
    color_map = {
        "jamb-": "#8a4a1a", "jamb+": "#8a4a1a", "trim": "#a06a30",
        "leaf": "#d59a55", "leaf-1": "#d59a55", "leaf-2": "#d59a55",
        "track": "#5a5e63", "panel": "#7eb6e0",
    }
    # Plan centre — used to push per-door annotations away from the
    # plan core so they don't overlap each other or the wall artwork.
    plan_xs, plan_ys = [], []
    for wp in wall_polygons:
        for x, y in wp.shell_xy:
            plan_xs.append(float(x))
            plan_ys.append(float(y))
    plan_cx = float(np.mean(plan_xs)) if plan_xs else 0.0
    plan_cy = float(np.mean(plan_ys)) if plan_ys else 0.0

    for placed in placed_doors:
        cls = (placed.label_class or "").lower()
        if cls == "door_sliding":
            comps = _sliding_components(placed)
        else:
            comps = _swing_components(placed, double=(cls == "door_double"))
        labels = _component_labels(placed.label_class, len(comps))

        poly = np.array(placed.polygon_xy)

        # Tag-7.9 diagnostic: the cyan pixel-symbol rectangle the door
        # algorithm extracted from the raster. When the placed door
        # rectangle sits far from this overlay, the bbox-fallback
        # stretched the rectangle past the actual symbol — useful when
        # diagnosing pass-vs-fail cases.
        if placed.pixel_polygon_xy and len(placed.pixel_polygon_xy) >= 3:
            ppx = np.array(placed.pixel_polygon_xy)
            ax.plot(np.append(ppx[:, 0], ppx[0, 0]),
                    np.append(ppx[:, 1], ppx[0, 1]),
                    color="#0ea5b7", linewidth=0.9, linestyle=(0, (3, 2)),
                    alpha=0.85, zorder=3)

        # Tag-7.8: the door rectangle border encodes pass / fail. Green
        # = passes hard criteria (in real gap, in bbox). Red = rejected
        # (e.g. door silently placed in solid wall).
        border_color = "#1d8a3e" if placed.pass_validation else "#c8302a"
        border_width = 1.4 if placed.pass_validation else 2.4
        ax.plot(np.append(poly[:, 0], poly[0, 0]),
                np.append(poly[:, 1], poly[0, 1]),
                color=border_color, linewidth=border_width, zorder=4)

        # Tag-7.10/7.11: jamb-anchor pins at each long-axis end of the
        # door rectangle. Per-side colour: green = anchored on a wall,
        # red = floating in open space on that side. Lets the reviewer
        # see WHICH side is loose for end-of-wall placements (jambs=1/2),
        # not just the total count.
        try:
            from planproof_geometry.ai.v3.pixel_bim.openings.door_models import (
                _polygon_basis,
            )
            _, long_u_jamb, _, W_jamb, _ = _polygon_basis(placed.polygon_xy)
            cx_jamb = float(np.array(placed.polygon_xy)[:, 0].mean())
            cy_jamb = float(np.array(placed.polygon_xy)[:, 1].mean())
            anchor_flags = {
                -1: bool(placed.jamb_anchor_minus),
                +1: bool(placed.jamb_anchor_plus),
            }
            for sign in (-1, +1):
                ex = cx_jamb + sign * (W_jamb * 0.5) * float(long_u_jamb[0])
                ey = cy_jamb + sign * (W_jamb * 0.5) * float(long_u_jamb[1])
                pin_color = "#1d8a3e" if anchor_flags[sign] else "#c8302a"
                ax.plot(ex, ey, marker="s", markersize=6,
                        markerfacecolor=pin_color,
                        markeredgecolor="#1a1a1a",
                        markeredgewidth=0.7, zorder=5)
        except Exception:
            pass

        for comp, name in zip(comps, labels):
            verts = np.asarray(comp.vertices)
            # Project to plan: world (X, Y, Z) → plan (X, Z); plan_y = world Z.
            # Wait: the wall mesh uses (x, y_height, plan_y). So plan_y = world Z.
            xz = verts[:, [0, 2]]
            try:
                hull = ShPoly([(p[0], p[1]) for p in xz]).convex_hull
                if hull.geom_type != "Polygon":
                    continue
                hx, hy = hull.exterior.xy
                ax.fill(list(hx), list(hy),
                        facecolor=color_map.get(name, "#888888"),
                        edgecolor="#222", linewidth=0.4, alpha=0.75)
            except Exception:
                continue

        cx = float(poly[:, 0].mean())
        cy = float(poly[:, 1].mean())

        # Tag-7.7: hinge marker (single-leaf swing doors only).
        hinge_sign = None
        marker_color = "#1d8a3e"
        if cls not in ("door_sliding", "door_double"):
            try:
                center, long_u, short_u, W, _ = _polygon_basis(
                    placed.polygon_xy
                )
            except Exception:
                center = np.array([cx, cy])
                long_u = np.array([1.0, 0.0])
                W = 0.0
            half_W = W * 0.5
            hinge_sign = +1 if placed.hinge_long_sign >= 0 else -1
            hinge_pt = center + hinge_sign * half_W * long_u
            latch_pt = center - hinge_sign * half_W * long_u
            conf = float(placed.hinge_confidence)
            if conf >= 0.70:
                marker_color = "#1d8a3e"
            elif conf >= 0.50:
                marker_color = "#d8861b"
            else:
                marker_color = "#c8302a"
            ax.plot(
                float(hinge_pt[0]), float(hinge_pt[1]),
                marker="o", markersize=10,
                markerfacecolor=marker_color, markeredgecolor="#1a1a1a",
                markeredgewidth=1.0, zorder=6,
            )
            ax.text(
                float(hinge_pt[0]), float(hinge_pt[1]), "H",
                fontsize=6, color="white", ha="center", va="center",
                weight="bold", zorder=7,
            )
            ax.plot(
                float(latch_pt[0]), float(latch_pt[1]),
                marker="x", markersize=7,
                markeredgecolor="#1a1a1a", markeredgewidth=1.3, zorder=6,
            )

        # Tag-7.8: annotation + leader line, anchored AWAY FROM the
        # plan centre so labels for clustered doors stay separated.
        # Direction = sign of (door_centre - plan_centre) projected onto
        # the door's swing-axis (single-leaf) or cardinal axis (other),
        # offset by 1.0 m past the door rectangle. The leader line keeps
        # the visual link clear even when the annotation lands in
        # whitespace several metres away.
        dx_plan = cx - plan_cx
        dy_plan = cy - plan_cy
        if cls not in ("door_sliding", "door_double"):
            try:
                _, _, short_u_anno, _, _ = _polygon_basis(placed.polygon_xy)
            except Exception:
                short_u_anno = np.array([0.0, 1.0])
            if (dx_plan * short_u_anno[0] + dy_plan * short_u_anno[1]) >= 0:
                anno_dir = short_u_anno
            else:
                anno_dir = -short_u_anno
        else:
            anno_dir = np.array(
                [np.sign(dx_plan) or 1.0, np.sign(dy_plan) or 1.0]
            )
            anno_dir /= max(np.linalg.norm(anno_dir), 1e-6)
        anno_offset_m = 1.10
        anno_xy = (cx + anno_dir[0] * anno_offset_m,
                   cy + anno_dir[1] * anno_offset_m)
        ax.plot(
            [cx, anno_xy[0]], [cy, anno_xy[1]],
            color="#777", linewidth=0.5, linestyle=":", zorder=3,
        )

        sigil = {"arc": "a", "panel": "p", "corners": "c",
                 "centroid": "t"}
        anno_lines = [placed.label_id.replace("ann_", "")]
        if hinge_sign is not None:
            vote_str = " ".join(
                f"{sigil.get(k, k[:1])}{'+' if v > 0 else '-'}"
                for k, v in placed.hinge_votes.items()
            ) or "-"
            anno_lines.append(
                f"hinge={hinge_sign:+d} conf={placed.hinge_confidence:.2f}"
            )
            anno_lines.append(f"[{vote_str}]")
        is_sliding_label = (placed.label_class or "").lower() == "door_sliding"
        status_tag = "PASS" if placed.pass_validation else "FAIL"
        if is_sliding_label:
            # Sprint Sliding-Door-Rewrite Tag-4: gap+jamb counters are
            # not meaningful for sliding doors. Show mount + pocket
            # direction instead.
            mount_label = (placed.mount_type or "pocket").lower()
            pocket_dir = "+" if int(placed.pocket_sign) >= 0 else "-"
            gap_line = (
                f"mount={mount_label} pocket={pocket_dir}long {status_tag}"
            )
        else:
            gap_pct_int = int(round(placed.gap_overlap_pct * 100))
            gap_line = (
                f"gap={gap_pct_int}% jambs={placed.jamb_anchors}/2 "
                f"{status_tag}"
            )
        if placed.width_source:
            gap_line += f" [{placed.width_source}]"
        anno_lines.append(gap_line)
        ax.annotate(
            "\n".join(anno_lines),
            xy=(cx, cy),
            xytext=anno_xy,
            fontsize=6.0, color="#1a1a1a",
            ha="center", va="center",
            bbox=dict(
                facecolor="white",
                edgecolor=marker_color if hinge_sign is not None else border_color,
                linewidth=0.7, alpha=0.95, pad=2,
            ),
            zorder=8,
        )

    # Tag-7.7 / 7.8 / 7.9 / 7.10 legend (top-left corner).
    legend_lines = [
        "Door border: green=pass, red=fail (Tag-7.8 hard gap-criterion)",
        "Cyan dashed: pixel-symbol extract (Tag-7.9 recovery anchor)",
        "Jamb pins (squares at door ends): green=anchored, red=floating",
        "Hinge marker: green=conf>=0.70, orange=0.50-0.70, red<0.50",
        "Vote sigils: a=arc, p=panel, c=corners, t=centroid",
    ]
    ax.text(
        0.01, 0.99, "\n".join(legend_lines),
        transform=ax.transAxes, fontsize=7, color="#222",
        ha="left", va="top",
        bbox=dict(facecolor="white", edgecolor="#888",
                  linewidth=0.5, alpha=0.92, pad=4),
        zorder=9,
    )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(
        "BIM viewer — top-down (gap criterion + hinge voting)",
        fontsize=11,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y plan (m)  [PDF Y axis, image-down]")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def render_walls_and_bridges_png(
    placed_doors: Sequence[PlacedDoor],
    wall_polygons,
    out_path: Path,
    *,
    wall_height_m: float = 2.80,
    user_wall_polys_m: "Sequence[Sequence[Tuple[float, float]]] | None" = None,
) -> None:
    """Clean 2D plan view of walls + lintel bridges only.

    Strips door leaves, panels, frames, labels — just the wall mass
    (grey) and the lintel-bridge footprint (orange) over each door.
    Useful to verify how the bridges fit between flanking walls without
    the door symbols and pixel diagnostics that crowd the regular
    top-down view.

    The bridge polygons match exactly what the 3D pipeline extrudes
    into the wall mesh (head -> wall_top), produced via the same
    ``compute_bridge_polygons_m`` helper that ``build_door_wall_bridges``
    uses. Beside-wall sliders are excluded (they have no aperture and
    therefore no lintel bridge).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    from planproof_geometry.ai.v3.pixel_bim.openings.door_models import (
        compute_bridge_polygons_m,
    )

    fig, ax = plt.subplots(figsize=(14, 14))

    # Walls: filled grey polygons.
    for wp in wall_polygons:
        if len(wp.shell_xy) < 3:
            continue
        xy = np.array(wp.shell_xy)
        ax.fill(xy[:, 0], xy[:, 1], facecolor="#cfd2d8",
                edgecolor="#6b6e74", linewidth=0.6, zorder=1)
        for h in wp.holes_xy:
            if len(h) >= 3:
                hxy = np.array(h)
                ax.fill(hxy[:, 0], hxy[:, 1], facecolor="white",
                        edgecolor="#6b6e74", linewidth=0.4, zorder=2)

    # Wall outlines (thick) for crisper boundaries.
    for wp in wall_polygons:
        if len(wp.shell_xy) < 3:
            continue
        xy = np.array(wp.shell_xy)
        closed = np.vstack([xy, xy[:1]])
        ax.plot(closed[:, 0], closed[:, 1],
                color="#1f2126", linewidth=0.7, zorder=3)

    # Bridge footprints: same logic as build_door_wall_bridges
    # (lintel-only, full_height=False), drawn as semi-transparent
    # orange polygons with a darker border.
    bridge_doors = [
        p for p in placed_doors
        if getattr(p, "sliding_mount_position", None) != "beside_wall"
    ]
    wall_shells = [list(wp.shell_xy) for wp in wall_polygons]
    bridge_polys = compute_bridge_polygons_m(
        bridge_doors,
        wall_height_m=wall_height_m,
        full_height=False,
        wall_polygons_xy=wall_shells,
        user_wall_polygons_xy=user_wall_polys_m,
    )

    n_bridges = 0
    for label_id, label_class, poly_xy in bridge_polys:
        if len(poly_xy) < 3:
            continue
        n_bridges += 1
        is_sliding = (label_class or "").lower() == "door_sliding"
        face = "#f4a26340" if is_sliding else "#f4a26370"
        edge = "#c25e1f"
        patch = MplPolygon(
            poly_xy, closed=True, facecolor=face, edgecolor=edge,
            linewidth=1.4, zorder=4,
        )
        ax.add_patch(patch)
        cx = float(np.mean([p[0] for p in poly_xy]))
        cy = float(np.mean([p[1] for p in poly_xy]))
        ax.annotate(
            label_id.replace("ann_", ""),
            xy=(cx, cy), ha="center", va="center",
            fontsize=6, color="#5a2a0c", zorder=5,
            bbox=dict(facecolor="#fff7ed", edgecolor="#c25e1f",
                      boxstyle="round,pad=0.15", linewidth=0.5, alpha=0.9),
        )

    # Door rectangles (thin black outline) for orientation — the bridges
    # should hug the door's long sides + extend toward the next wall.
    for placed in placed_doors:
        if len(placed.polygon_xy) < 4:
            continue
        if getattr(placed, "sliding_mount_position", None) == "beside_wall":
            continue
        poly = np.array(placed.polygon_xy)
        closed = np.vstack([poly, poly[:1]])
        ax.plot(closed[:, 0], closed[:, 1],
                color="#1a1a1a", linewidth=0.6, linestyle=(0, (3, 2)),
                zorder=6)

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m, plan)")
    ax.invert_yaxis()  # PDF Y goes down — match the topdown convention.
    ax.set_title(
        f"Walls + Lintel Bridges (plan view, {n_bridges} bridges) — "
        "grey = walls, orange = lintels (head -> wall_top), "
        "dashed = door rectangle"
    )
    ax.grid(True, alpha=0.25, linewidth=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def render_elevation_png(
    placed_doors: Sequence[PlacedDoor],
    out_path: Path,
    *,
    wall_height_m: float = 2.80,
) -> None:
    """Per-door elevation view (looking along +short_axis = into the
    swing direction). Stacks the doors vertically in a single PNG.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(placed_doors)
    if n == 0:
        return
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(5.0 * cols, 4.5 * rows), squeeze=False,
    )

    # Strong contrasts so each component is visually distinct.
    color_map = {
        "jamb-": "#5a2e0a", "jamb+": "#5a2e0a", "trim": "#8a4a1a",
        "leaf": "#e8a040", "leaf-1": "#e8a040", "leaf-2": "#cf7a30",
        "track": "#3a3d42", "panel": "#5fa6e0",
    }
    edge_color = "#1a1a1a"

    for i, placed in enumerate(placed_doors):
        ax = axes[i // cols][i % cols]
        cls = (placed.label_class or "").lower()
        if cls == "door_sliding":
            comps = _sliding_components(placed)
        else:
            comps = _swing_components(placed, double=(cls == "door_double"))
        labels = _component_labels(placed.label_class, len(comps))

        center, long_u, short_u, W, _ = _polygon_basis(placed.polygon_xy)
        door_center_3d = np.array([center[0], 0.0, center[1]])
        long_3d = np.array([long_u[0], 0.0, long_u[1]])

        wall_d = placed.wall_depth_m
        # Sketch wall section: just an outline of "wall above door" and the
        # void below.
        ax.add_patch(Rectangle(
            (-W * 0.5 - 0.15, placed.head_height_m),
            W + 0.30, wall_height_m - placed.head_height_m,
            facecolor="#dfe1e5", edgecolor="#777", linewidth=0.5,
        ))
        ax.add_patch(Rectangle(
            (-W * 0.5 - 0.15, 0.0),
            0.15, placed.head_height_m,
            facecolor="#dfe1e5", edgecolor="#777", linewidth=0.5,
        ))
        ax.add_patch(Rectangle(
            (W * 0.5, 0.0),
            0.15, placed.head_height_m,
            facecolor="#dfe1e5", edgecolor="#777", linewidth=0.5,
        ))

        # Draw closed-door outline for reference (dashed orange rectangle
        # spanning the actual door rectangle width × full height).
        ax.plot(
            [-W * 0.5, W * 0.5, W * 0.5, -W * 0.5, -W * 0.5],
            [0.0, 0.0, placed.head_height_m,
             placed.head_height_m, 0.0],
            color="#ff7700", linewidth=0.6, linestyle="--",
        )

        for comp, name in zip(comps, labels):
            verts = np.asarray(comp.vertices)
            rel = verts - door_center_3d
            proj_long = rel @ long_3d
            proj_y = verts[:, 1]
            try:
                from shapely.geometry import MultiPoint
                mp = MultiPoint(list(zip(proj_long, proj_y)))
                hull = mp.convex_hull
                if hull.geom_type != "Polygon":
                    continue
                hx, hy = hull.exterior.xy
                ax.fill(list(hx), list(hy),
                        facecolor=color_map.get(name, "#888"),
                        edgecolor=edge_color, linewidth=0.7, alpha=0.95)
                # Annotate component centre with name + footprint width.
                lo, hi = float(min(proj_long)), float(max(proj_long))
                yc_lo, yc_hi = float(min(proj_y)), float(max(proj_y))
                yc = (yc_lo + yc_hi) * 0.5
                # Skirt placement around adjacent components for legibility.
                if name == "trim":
                    text_y = yc_hi + 0.05
                elif name in ("leaf", "leaf-1", "leaf-2"):
                    text_y = 1.0
                elif name == "panel":
                    text_y = 1.0
                elif name == "track":
                    text_y = yc_hi + 0.08
                else:
                    text_y = yc
                ax.annotate(
                    f"{name}\n{(hi-lo)*1000:.0f}mm",
                    ((lo + hi) * 0.5, text_y),
                    color="white" if name in (
                        "leaf", "leaf-1", "leaf-2", "panel",
                    ) else "#222",
                    fontsize=7,
                    ha="center", va="center", weight="bold",
                    bbox=dict(facecolor=color_map.get(name, "#888"),
                              edgecolor="none", alpha=0.85, pad=2)
                    if name not in ("leaf", "leaf-1", "leaf-2", "panel")
                    else None,
                )
            except Exception:
                continue

        ax.set_aspect("equal")
        ax.set_xlim(-W * 0.5 - 0.30, W * 0.5 + 0.30)
        ax.set_ylim(-0.10, wall_height_m + 0.10)
        swing_tag = (f"sw={placed.swing_dir_sign:+d}"
                     if cls != "door_sliding"
                     else f"mt={placed.mount_type}")
        ax.set_title(
            f"{placed.label_id.replace('ann_', '')}\n"
            f"{placed.label_class}  W={W:.2f}m d={wall_d:.2f}m  {swing_tag}",
            fontsize=8,
        )
        ax.set_xlabel("along wall (m)")
        ax.set_ylabel("height (m)")

    # Hide unused axes.
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def render_bim_y_slice_png(
    mesh,
    y_m: float,
    out_path: Path,
    *,
    title: str = "",
    window_envelopes_xy: Sequence[Sequence[Tuple[float, float]]] = (),
) -> bool:
    """Render a horizontal cross-section of the BIM mesh at world Y.

    The cross-section reveals where wall material exists at a given
    height. For Window-BIM-Fix Tag 1 verification:
      * y_m=0.5 (parapet zone, between floor and sill_m=0.9): wall
        material must be present at every window position.
      * y_m=2.4 (lintel zone, between head_m=2.1 and ceiling=2.8):
        same.

    Optional ``window_envelopes_xy`` are overlaid as red dashed
    rectangles so the reader can visually confirm wall material
    fills each window's parapet/lintel strip.

    Returns True if a section was found and rasterised, False if the
    plane misses the mesh entirely.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    section = mesh.section(plane_origin=[0.0, y_m, 0.0],
                           plane_normal=[0.0, 1.0, 0.0])
    if section is None:
        return False
    # Newer trimesh prefers to_2D, older only has to_planar.
    if hasattr(section, "to_2D"):
        planar, _T = section.to_2D()
    else:
        planar, _T = section.to_planar()
    # ``polygons_full`` builds a hole hierarchy via rtree and isn't
    # available in stripped envs. ``polygons_closed`` returns each
    # closed loop as its own shapely polygon, which is enough for a
    # diagnostic visualization — outer rings come out filled, inner
    # rings (holes) get a slightly smaller area and overdraw white.
    raw_polys = getattr(planar, "polygons_closed", None)
    if raw_polys is None:
        return False
    polys = [p for p in list(raw_polys) if p is not None and not p.is_empty]
    if not polys:
        return False
    polys.sort(key=lambda p: -p.area)

    fig, ax = plt.subplots(figsize=(10, 7))
    for poly in polys:
        x, y = poly.exterior.xy
        # Largest loops = outer wall outlines (grey fill); smaller
        # nested loops = void holes that sit inside an outer loop.
        is_hole = any(
            outer.contains(poly) and outer.area > poly.area * 1.0001
            for outer in polys if outer is not poly
        )
        ax.fill(
            x, y,
            color=("white" if is_hole else "0.7"),
            edgecolor="0.3", linewidth=0.5,
        )

    # Window envelopes as red dashed boxes — section's to_planar maps
    # world (X, Z) -> planar (X, -Z) for our axis-aligned horizontal
    # cut, matching the BIM extrusion's flip_y_for_pdf=True convention
    # used in plan-coords elsewhere.
    for env_xy in window_envelopes_xy:
        if not env_xy or len(env_xy) < 3:
            continue
        ring = [(float(x), -float(y)) for x, y in env_xy]
        ax.add_patch(MplPolygon(
            ring, closed=True, fill=False,
            edgecolor="red", linewidth=1.2, linestyle="--",
        ))

    ax.set_aspect("equal")
    ax.set_title(title or f"BIM y-slice at y = {y_m:.2f} m")
    ax.set_xlabel("plan X (m)")
    ax.set_ylabel("plan Z (m, flipped)")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return True
