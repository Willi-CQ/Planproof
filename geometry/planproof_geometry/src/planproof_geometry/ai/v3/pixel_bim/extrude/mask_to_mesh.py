"""Pixel wall mask -> shapely polygons -> 3D extruded mesh.

Canonical 5-step pipeline:

    wall_mask_px                                                # uint8 full-res
        |
        | 1. (optional) paired-outline close + fill holes
        v
    closed_wall_mask
        |
        | 2. cv2.findContours(RETR_CCOMP)
        v
    outer_contours + hole_contours
        |
        | 3. shapely.Polygon(shell, holes).buffer(0).simplify(simplify_m)
        v
    list[shapely.Polygon]   (cleaned, self-healed)
        |
        | 4. shapely.ops.unary_union   (merges T-junctions, dedupes overlap)
        v
    shapely.MultiPolygon
        |
        | 5. trimesh.creation.extrude_polygon(p, height=wall_height_m)
        v
    trimesh.Trimesh       (watertight, ready for glTF/IFC export)

All polygons travel through shapely in **metres**, not pixels. The
conversion happens right after findContours via ``px_to_m``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .paired_outline import PairedOutlineReport, detect_paired_outline_mode


@dataclass
class WallPolygon:
    """One wall footprint in plan-metre space."""
    shell_xy: List[Tuple[float, float]]
    holes_xy: List[List[Tuple[float, float]]] = field(default_factory=list)
    area_m2: float = 0.0
    source: str = "classical"


@dataclass
class MaskToMeshResult:
    """End-to-end result of Phase 2."""
    polygons: List["WallPolygon"]
    mesh: "trimesh.Trimesh"  # type: ignore[name-defined]  # noqa: F821
    paired_outline: Optional[PairedOutlineReport]
    closed_wall_mask: Optional[np.ndarray] = None
    total_area_m2: float = 0.0
    notes: List[str] = field(default_factory=list)


# ── Private helpers ──────────────────────────────────────────────────────


def _contours_to_polygons(
    mask: np.ndarray,
    *,
    px_to_m: float,
    simplify_m: float,
    min_polygon_area_m2: float,
):
    """Turn a binary mask into shapely polygons in metre space.

    The mask is **not** rescaled — px_to_m converts each contour point
    directly. Shell/hole topology comes from ``cv2.RETR_CCOMP``.
    """
    from shapely.geometry import Polygon  # lazy import

    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return []

    # hier[0][i] = [next, prev, first_child, parent]
    # Outer contour: parent == -1. Its children are holes.
    polygons = []
    n = len(cnts)
    hier = hier[0] if hier is not None else [[-1, -1, -1, -1]] * n

    for i in range(n):
        if hier[i][3] != -1:
            continue  # skip holes here; we gather them with their parent
        shell_px = cnts[i].squeeze(axis=1)  # (N, 2)
        if shell_px.shape[0] < 3:
            continue
        shell_m = shell_px.astype(np.float64) * px_to_m
        # Collect child holes (direct children only).
        holes_m = []
        child = hier[i][2]
        while child != -1:
            hole_px = cnts[child].squeeze(axis=1)
            if hole_px.shape[0] >= 3:
                hole_m = (hole_px.astype(np.float64) * px_to_m).tolist()
                holes_m.append(hole_m)
            child = hier[child][0]  # next sibling
        try:
            poly = Polygon(shell_m.tolist(), holes=holes_m)
        except Exception:
            continue
        # Self-heal via buffer(0). Simplify for triangulation stability.
        poly = poly.buffer(0)
        if simplify_m > 0:
            poly = poly.simplify(simplify_m, preserve_topology=True)
        if poly.is_empty or not poly.is_valid:
            continue
        # buffer(0) may return a MultiPolygon; split so we handle one shell
        # at a time downstream.
        for piece in (poly.geoms if poly.geom_type == "MultiPolygon" else [poly]):
            if piece.area < min_polygon_area_m2:
                continue
            shell = list(piece.exterior.coords)
            holes = [list(r.coords) for r in piece.interiors]
            polygons.append(WallPolygon(
                shell_xy=[(float(x), float(y)) for x, y in shell],
                holes_xy=[[(float(x), float(y)) for x, y in h] for h in holes],
                area_m2=float(piece.area),
            ))
    return polygons


def _close_mask(mask: np.ndarray, kernel_px: int,
                fill_holes: bool = True) -> np.ndarray:
    """Morphological close + optional hole fill.

    Kernel-size safety: caller should cap kernel_px to
    ``min_din_wall_thickness / 2`` so we do not merge two adjacent
    parallel partition walls by accident.
    """
    if kernel_px <= 0:
        return mask.copy()
    k = max(3, kernel_px | 1)  # force odd
    # MORPH_RECT preserves sharp 90° corners (matches dispatcher choice);
    # ELLIPSE rounds them which is the dominant L/T-junction artefact.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if fill_holes:
        # Fill 4-connected background starting from the border, everything
        # else that is non-foreground becomes a hole we flip to foreground.
        border_flood = closed.copy()
        h, w = closed.shape
        # Seed from all four corners to be robust if (0, 0) happens to be wall
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
        for sx, sy in seeds:
            if border_flood[sy, sx] == 0:
                cv2.floodFill(border_flood, ff_mask, (sx, sy), 128)
        # holes = pixels still 0 after flood = interior voids
        holes = (border_flood == 0).astype(np.uint8) * 255
        closed = cv2.bitwise_or(closed, holes)
    return closed


def _reclip_polygons_to_authority(
    polygons: List["WallPolygon"],
    *,
    user_polygon_union_mask: np.ndarray,
    px_to_m: float,
    authority_dilate_m: float,
    min_polygon_area_m2: float,
    image_gray: Optional[np.ndarray] = None,
    ink_dilate_m: float = 0.030,
) -> List["WallPolygon"]:
    """Intersect each pixel-snapped polygon with the dilated user-polygon
    union AND a dilated Otsu ink region, so post-snap geometry never sits
    outside both the user's authority *and* the actual paper ink.

    Tag-7 Fix 2: the user-polygon authority alone allowed the wall to
    drift up to ``authority_dilate_m`` past the real ink stroke when the
    user labelled generously. Combining with a dilated ink mask
    (``ink_dilate_m`` ~30 mm) clamps the wall body to where there is
    actually paper material to anchor against.

    Implementation: dilate the user mask by ``authority_dilate_m``,
    optionally AND with a dilated Otsu ink mask, then vectorise the
    intersection, intersect each input polygon (in metre space).
    Polygons that fall apart into multi-pieces are split.
    """
    from shapely.geometry import Polygon  # lazy import
    from shapely.ops import unary_union

    inv_px = 1.0 / max(px_to_m, 1e-9)
    dilate_px = max(1, int(round(authority_dilate_m / max(px_to_m, 1e-9))))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px, dilate_px))
    auth_mask = cv2.dilate(user_polygon_union_mask, kernel, iterations=1)

    if image_gray is not None and ink_dilate_m > 0:
        # Otsu ink within the user-authority region (clip to auth_mask
        # so global furniture / hatching outside the wall annotations
        # cannot anchor wall geometry).
        _, otsu = cv2.threshold(
            image_gray, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )
        ink_in_auth = cv2.bitwise_and(otsu, auth_mask)
        # Distance transform tells us, for every pixel inside the
        # authority region, how far it is from the nearest ink stroke.
        # Pixels with ``dist <= ink_dilate_m`` are "near ink" and form
        # the wall_zone. For a 200 mm paired-outline wall the centre is
        # ~100 mm from each outline, so ink_dilate_m >= 0.10 m keeps
        # the wall body included; smaller values would exclude the
        # ink-free middle and split each polygon into two strips.
        inverted = (ink_in_auth == 0).astype(np.uint8)
        dist_to_ink = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
        thresh_px = ink_dilate_m / max(px_to_m, 1e-9)
        wall_zone = (dist_to_ink <= thresh_px).astype(np.uint8) * 255
        # Combine: wall geometry must sit inside BOTH the user authority
        # AND within ink_dilate_m of an ink stroke.
        auth_mask = cv2.bitwise_and(auth_mask, wall_zone)

    cnts, _ = cv2.findContours(auth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return polygons
    auth_polys = []
    for c in cnts:
        pts = c.squeeze(axis=1)
        if pts.shape[0] < 3:
            continue
        pts_m = pts.astype(np.float64) * px_to_m
        try:
            p = Polygon(pts_m.tolist()).buffer(0)
        except Exception:
            continue
        if not p.is_empty:
            auth_polys.append(p)
    if not auth_polys:
        return polygons
    authority = unary_union(auth_polys)

    out: List[WallPolygon] = []
    for wp in polygons:
        try:
            p = Polygon(wp.shell_xy, holes=wp.holes_xy).buffer(0)
        except Exception:
            out.append(wp)
            continue
        try:
            clipped = p.intersection(authority)
        except Exception:
            out.append(wp)
            continue
        if clipped.is_empty:
            continue
        pieces = (
            clipped.geoms if clipped.geom_type == "MultiPolygon" else [clipped]
        )
        for piece in pieces:
            if piece.is_empty or piece.area < min_polygon_area_m2:
                continue
            shell = list(piece.exterior.coords)
            holes = [list(r.coords) for r in piece.interiors]
            out.append(WallPolygon(
                shell_xy=[(float(x), float(y)) for x, y in shell],
                holes_xy=[[(float(x), float(y)) for x, y in h] for h in holes],
                area_m2=float(piece.area),
                source=wp.source,
            ))
    _ = inv_px  # silence unused: kept for symmetry with other helpers
    return out


# ── Public API ───────────────────────────────────────────────────────────


def mask_to_wall_polygons(
    wall_mask: np.ndarray,
    *,
    px_to_m: float,
    simplify_m: float = 0.02,
    min_polygon_area_m2: float = 0.005,
    min_din_wall_thickness_m: float = 0.10,
    auto_detect_paired_outline: bool = False,
    axis_snap_angle_deg: float = 5.0,
    axis_snap_cluster_m: float = 0.02,
    image_gray: Optional[np.ndarray] = None,
    user_polygon_union_mask: Optional[np.ndarray] = None,
    pixel_snap_cap_m: float = 0.012,
    pixel_snap_parallel_tol_deg: float = 4.0,
    authority_reclip_m: float = 0.015,
    right_angle_short_edge_m: float = 0.60,
) -> Tuple[List[WallPolygon], Optional[PairedOutlineReport], np.ndarray]:
    """Turn a wall mask into a list of 2D metre-space polygons.

    Parameters
    ----------
    wall_mask : np.ndarray
        uint8 mask (H, W). Non-zero = wall.
    px_to_m : float
        Metres per pixel.
    simplify_m : float
        Douglas-Peucker tolerance in metres (default 2 cm).
    min_polygon_area_m2 : float
        Drop polygons below this area (default 300 cm^2). Kills tiny
        fragments from morphological noise.
    min_din_wall_thickness_m : float
        Upper bound on the paired-outline close kernel / 2. Prevents the
        closer from bridging two physically separate partition walls.
    auto_detect_paired_outline : bool
        If True, run the paired-outline detector and close the mask only
        when paired mode is detected.
    """
    px_per_m = 1.0 / px_to_m if px_to_m > 0 else 1.0

    report: Optional[PairedOutlineReport] = None
    closed = wall_mask
    if auto_detect_paired_outline:
        report = detect_paired_outline_mode(wall_mask, px_per_m=px_per_m)
        if report.paired_mode and report.close_kernel_px > 0:
            # Cap kernel so we can never bridge two parallel partitions.
            max_safe_px = int(min_din_wall_thickness_m * px_per_m)
            k = min(report.close_kernel_px, max_safe_px)
            closed = _close_mask(wall_mask, k, fill_holes=True)

    polygons = _contours_to_polygons(
        closed,
        px_to_m=px_to_m,
        simplify_m=simplify_m,
        min_polygon_area_m2=min_polygon_area_m2,
    )
    if image_gray is not None and pixel_snap_cap_m > 0:
        from .pixel_snap import (  # noqa: PLC0415
            pixel_snap_polygon,
            precompute_ink_field,
        )
        roi_dilate_px = max(1, int(round(0.05 / max(px_to_m, 1e-9))))
        dist_to_ink, gx, gy = precompute_ink_field(
            image_gray,
            user_polygon_union_mask,
            roi_dilate_px=roi_dilate_px,
        )
        for wp in polygons:
            wp.shell_xy, _info = pixel_snap_polygon(
                wp.shell_xy,
                px_to_m=px_to_m,
                dist_to_ink_px=dist_to_ink,
                grad_x=gx,
                grad_y=gy,
                snap_cap_m=pixel_snap_cap_m,
                parallel_tol_deg=pixel_snap_parallel_tol_deg,
            )
            new_holes = []
            for h in wp.holes_xy:
                snapped, _ = pixel_snap_polygon(
                    h,
                    px_to_m=px_to_m,
                    dist_to_ink_px=dist_to_ink,
                    grad_x=gx,
                    grad_y=gy,
                    snap_cap_m=pixel_snap_cap_m,
                    parallel_tol_deg=pixel_snap_parallel_tol_deg,
                )
                new_holes.append(snapped)
            wp.holes_xy = new_holes
        # Phase B.2: re-clip every snapped polygon to the user authority
        # region dilated by ``authority_reclip_m``. Pixel-snap can push a
        # vertex up to ``snap_cap_m`` outwards onto an adjacent ink stroke
        # that sits outside the user polygon (e.g. an unrelated wall right
        # next to the labelled one). Without this clip, wall geometry can
        # drift past the user's authority by up to one snap cap.
        if (
            user_polygon_union_mask is not None
            and authority_reclip_m > 0
            and polygons
        ):
            polygons = _reclip_polygons_to_authority(
                polygons,
                user_polygon_union_mask=user_polygon_union_mask,
                px_to_m=px_to_m,
                authority_dilate_m=authority_reclip_m,
                min_polygon_area_m2=min_polygon_area_m2,
                image_gray=image_gray,
                ink_dilate_m=0.10,  # 100 mm = half of typical wall thickness
            )
    if axis_snap_angle_deg > 0 and axis_snap_cluster_m > 0:
        from .axis_snap import axis_snap_polygon  # noqa: PLC0415
        for wp in polygons:
            wp.shell_xy = axis_snap_polygon(
                wp.shell_xy,
                angle_deg=axis_snap_angle_deg,
                cluster_tol_m=axis_snap_cluster_m,
            )
            wp.holes_xy = [
                axis_snap_polygon(
                    h,
                    angle_deg=axis_snap_angle_deg,
                    cluster_tol_m=axis_snap_cluster_m,
                )
                for h in wp.holes_xy
            ]
    # Tag-7 Fix 1: right-angle enforcement on short connector edges
    # (wall-stirnen, Z-step stubs, end-caps). Runs AFTER axis_snap so
    # the long edges already sit on global X/Y rails — the stage only
    # touches the SHORT edge endpoints to make them perpendicular.
    if right_angle_short_edge_m > 0:
        from .right_angle_snap import enforce_right_angles_polygon  # noqa: PLC0415
        for wp in polygons:
            wp.shell_xy = enforce_right_angles_polygon(
                wp.shell_xy,
                short_edge_max_m=right_angle_short_edge_m,
                axis_tol_deg=axis_snap_angle_deg,
            )
            wp.holes_xy = [
                enforce_right_angles_polygon(
                    h,
                    short_edge_max_m=right_angle_short_edge_m,
                    axis_tol_deg=axis_snap_angle_deg,
                )
                for h in wp.holes_xy
            ]

    # Tag-9 Fix: axis_snap and right_angle_snap collapse vertices onto
    # shared X/Y rails, which can leave long runs of collinear vertices
    # that the original Douglas-Peucker simplify (run BEFORE the snap
    # stages) never had a chance to eliminate. A final shapely.simplify
    # at 1 mm tolerance drops collinear redundancy without disturbing the
    # rails (movements are <= 1 mm and only happen along the rail axis).
    if simplify_m > 0:
        from shapely.geometry import Polygon as _SPoly  # noqa: PLC0415
        for wp in polygons:
            try:
                shp = _SPoly(wp.shell_xy, holes=wp.holes_xy or None)
                if not shp.is_valid:
                    shp = shp.buffer(0)
                shp = shp.simplify(0.001, preserve_topology=True)
                if shp.is_empty or shp.geom_type != "Polygon":
                    continue
                wp.shell_xy = list(shp.exterior.coords)
                wp.holes_xy = [list(r.coords) for r in shp.interiors]
            except Exception:
                # Defensive: keep the un-resimplified polygon on any
                # shapely hiccup; geometry is still valid, just verbose.
                pass
    return polygons, report, closed


def wall_polygons_to_mesh(
    polygons: List[WallPolygon],
    *,
    wall_height_m: float = 2.80,
    y_axis_up: bool = True,
    flip_y_for_pdf: bool = True,
    merge_buffer_m: float = 0.030,
):
    """Extrude each wall polygon to ``wall_height_m`` and concatenate.

    Coordinate convention: we keep the plan's (X, Y) as (X, Z) in scene
    coordinates with Y as the vertical axis (matching our existing glTF
    exporter). ``flip_y_for_pdf`` mirrors Y for the usual "PDF Y goes
    down, BIM Y goes up" convention.

    Returns one ``trimesh.Trimesh`` (possibly containing multiple
    connected components) ready for direct export.
    """
    import trimesh
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    # Build a 2D shapely polygon per wall polygon (with holes preserved),
    # then UNION them together so touching wall segments merge into one
    # connected outline. Otherwise an L- or T-junction between two walls
    # produces two separate boxes that share a vertical edge but render
    # as visibly disconnected meshes (50+ components, seam artefacts,
    # redundant interior triangles facing into the junction).
    flat_polys: List["Polygon"] = []
    for wp in polygons:
        shell = [(x, -y) if flip_y_for_pdf else (x, y) for x, y in wp.shell_xy]
        holes = [
            [(x, -y) if flip_y_for_pdf else (x, y) for x, y in h]
            for h in wp.holes_xy
        ]
        poly = Polygon(shell, holes=holes).buffer(0)
        if poly.is_empty:
            continue
        if poly.geom_type == "MultiPolygon":
            flat_polys.extend(g for g in poly.geoms if not g.is_empty)
        else:
            flat_polys.append(poly)

    if not flat_polys:
        return trimesh.Trimesh()

    # Two-stage inflate-union-deflate: wall annotations are typically
    # labelled as one rectangle per wall segment, so adjacent walls at
    # L- and T-junctions arrive as separate polygons with gaps up to a
    # few cm. A single buffer pass with a large radius rounds short edge
    # tips; a single small-radius pass leaves visible seams at junctions
    # where the gap is wider than the buffer.
    #
    # Stage 1: small buffer (~merge_buffer_m / 3) — bridges sub-cm jitter
    # without rounding short walls. Most touching segments fuse here.
    # Stage 2: full buffer (merge_buffer_m, default 30 mm) on the
    # remaining MultiPolygon — closes wider L/T-junction gaps. 30 mm is
    # well under one wall-thickness (interior 100-200 mm) so we cannot
    # accidentally merge two distinct partition walls; mitre_limit=10
    # keeps 90° corners sharp.
    _STAGE1 = max(0.005, merge_buffer_m / 3.0)
    _STAGE2 = max(0.0, merge_buffer_m)
    try:
        inflated = [
            p.buffer(_STAGE1, join_style=2, mitre_limit=10.0)
            for p in flat_polys
        ]
        merged = unary_union(inflated)
        if merged is not None and not merged.is_empty:
            merged = merged.buffer(-_STAGE1, join_style=2, mitre_limit=10.0)
        if (
            _STAGE2 > _STAGE1
            and merged is not None
            and not merged.is_empty
            and merged.geom_type == "MultiPolygon"
        ):
            inflated2 = [
                g.buffer(_STAGE2, join_style=2, mitre_limit=10.0)
                for g in merged.geoms
            ]
            merged2 = unary_union(inflated2)
            if merged2 is not None and not merged2.is_empty:
                merged = merged2.buffer(
                    -_STAGE2, join_style=2, mitre_limit=10.0,
                )
    except Exception:
        # Fall back to pre-merge behaviour: concatenate per-polygon meshes.
        merged = None

    if merged is None or merged.is_empty:
        pieces: List["Polygon"] = flat_polys
    elif merged.geom_type == "MultiPolygon":
        pieces = [g for g in merged.geoms if not g.is_empty]
    else:
        pieces = [merged]

    meshes: List[trimesh.Trimesh] = []
    for piece in pieces:
        if piece.area <= 1e-6:
            continue
        try:
            m = trimesh.creation.extrude_polygon(piece, wall_height_m)
        except Exception as exc:
            # Triangulation failed — skip but keep going.
            print(f"  extrude_polygon failed on polygon area={piece.area:.2f}: {exc}")
            continue
        # The built-in extrude extrudes along +Z; we need walls to stand
        # on the XZ-plane with height along +Y. Rotate -90 deg around X.
        if y_axis_up:
            R = trimesh.transformations.rotation_matrix(
                -np.pi / 2, [1, 0, 0]
            )
            m.apply_transform(R)
        meshes.append(m)

    if not meshes:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(meshes)


def extrude_wall_mask(
    wall_mask: np.ndarray,
    *,
    px_to_m: float,
    wall_height_m: float = 2.80,
    simplify_m: float = 0.02,
    min_polygon_area_m2: float = 0.005,
    flip_y_for_pdf: bool = True,
    auto_detect_paired_outline: bool = False,
    image_gray: Optional[np.ndarray] = None,
    user_polygon_union_mask: Optional[np.ndarray] = None,
    pixel_snap_cap_m: float = 0.012,
    pixel_snap_parallel_tol_deg: float = 4.0,
    axis_snap_angle_deg: float = 5.0,
    axis_snap_cluster_m: float = 0.02,
    authority_reclip_m: float = 0.015,
    merge_buffer_m: float = 0.030,
    right_angle_short_edge_m: float = 0.60,
) -> MaskToMeshResult:
    """End-to-end convenience: wall mask -> MaskToMeshResult.

    Combines `mask_to_wall_polygons` and `wall_polygons_to_mesh`.
    """
    polygons, report, closed = mask_to_wall_polygons(
        wall_mask,
        px_to_m=px_to_m,
        simplify_m=simplify_m,
        min_polygon_area_m2=min_polygon_area_m2,
        auto_detect_paired_outline=auto_detect_paired_outline,
        image_gray=image_gray,
        user_polygon_union_mask=user_polygon_union_mask,
        pixel_snap_cap_m=pixel_snap_cap_m,
        pixel_snap_parallel_tol_deg=pixel_snap_parallel_tol_deg,
        axis_snap_angle_deg=axis_snap_angle_deg,
        axis_snap_cluster_m=axis_snap_cluster_m,
        authority_reclip_m=authority_reclip_m,
        right_angle_short_edge_m=right_angle_short_edge_m,
    )
    mesh = wall_polygons_to_mesh(
        polygons,
        wall_height_m=wall_height_m,
        flip_y_for_pdf=flip_y_for_pdf,
        merge_buffer_m=merge_buffer_m,
    )
    total_area = sum(wp.area_m2 for wp in polygons)
    notes = []
    if report and report.paired_mode:
        notes.append(
            f"paired_outline_close k={report.close_kernel_px}px "
            f"ratio={report.paired_ratio:.2f}"
        )
    return MaskToMeshResult(
        polygons=polygons,
        mesh=mesh,
        paired_outline=report,
        closed_wall_mask=closed,
        total_area_m2=total_area,
        notes=notes,
    )
