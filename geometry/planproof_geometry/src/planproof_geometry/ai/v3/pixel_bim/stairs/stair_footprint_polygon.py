"""Stair footprint = User-Bbox MINUS Wall-Union (Tag 9 — 2026-05-22).

Stairs in floor plans rarely fit cleanly into the rectangular bbox the
user draws around them. Walls intrude into the bbox at corners,
columns sit in the middle of a stair area, and L-shaped flights leave
big rectangular notches that the bbox covers but the stair does not.

The previous algorithm (PCA over all dark pixels in the bbox) ignored
this and produced rectangular footprints. The fix is **conceptually**
the same pattern the door / window pipelines already use: walls are
the source of truth, and other elements fit themselves to the walls.

Concretely::

    stair_polygon = user_polygon.difference(wall_union)

with three robustness affordances:

    * MultiPolygon results are reduced to the largest area piece —
      a stair label normally maps to one connected stair region; the
      smaller pieces are noise (wall slivers crossing the bbox).
    * Empty / degenerate results fall back to the original user
      polygon so the pipeline never explodes when wall extraction is
      missing or wrong.
    * A minimum-area filter rejects "wall-eaten-everything" cases.

The result is a shapely Polygon (potentially with holes) that the
pixel extractor and mesh builder consume downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# Minimum footprint area (m²) before we trust the difference result.
# Below this we assume the wall subtraction ate the stair and fall
# back to the raw user polygon — the geometry pipeline downstream
# will then reject the stair via the SKIPPED_TOO_SMALL gate.
MIN_TRIMMED_AREA_M2 = 0.50

# Minimum hole area (m²) before we admit a hole into the footprint.
# Sub-cm holes are noise from polygon snapping.
MIN_HOLE_AREA_M2 = 0.01


@dataclass
class StairFootprintPolygon:
    """Result of the wall-subtraction stair footprint computation."""

    polygon: "shapely.geometry.Polygon"  # type: ignore[name-defined]  # noqa: F821
    used_wall_subtraction: bool = False
    n_holes: int = 0
    n_dropped_pieces: int = 0
    fell_back_to_user_polygon: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def area_m2(self) -> float:
        try:
            return float(self.polygon.area)
        except (AttributeError, ValueError):
            return 0.0

    def exterior_xy(self) -> List[Tuple[float, float]]:
        """Exterior ring as plain list of (x, y) tuples."""
        try:
            return list(self.polygon.exterior.coords)
        except AttributeError:
            return []

    def hole_rings_xy(self) -> List[List[Tuple[float, float]]]:
        """Inner ring vertices per hole."""
        try:
            return [list(ring.coords) for ring in self.polygon.interiors]
        except AttributeError:
            return []


def compute_stair_polygon(
    user_polygon_m: Sequence[Tuple[float, float]],
    bim_walls_m: Optional[Sequence[Sequence[Tuple[float, float]]]] = None,
) -> StairFootprintPolygon:
    """Compute the stair footprint as (user_polygon - wall_union).

    Args:
        user_polygon_m: stair user polygon vertices in plan metres.
            Typically a rectangle (the bbox the user drew) but a
            polygon shape is also accepted.
        bim_walls_m: post-extruded wall polygons in plan metres.
            ``None`` or empty -> no wall subtraction, footprint is the
            unmodified user polygon.

    Returns:
        ``StairFootprintPolygon`` carrying the resulting Shapely
        polygon plus diagnostic flags. Always returns a valid polygon
        — caller never has to deal with None or MultiPolygon.
    """
    from shapely.geometry import (  # noqa: PLC0415
        MultiPolygon,
        Polygon,
    )
    from shapely.ops import unary_union  # noqa: PLC0415

    notes: List[str] = []
    if len(user_polygon_m) < 3:
        # Degenerate input — caller should have caught this earlier.
        # We still return a "polygon" sentinel so downstream code does
        # not crash; the area will be 0.
        return StairFootprintPolygon(
            polygon=Polygon(),
            fell_back_to_user_polygon=True,
            notes=["user polygon has < 3 vertices"],
        )

    user_poly = Polygon(user_polygon_m).buffer(0)
    if user_poly.is_empty or not user_poly.is_valid:
        return StairFootprintPolygon(
            polygon=Polygon(user_polygon_m),
            fell_back_to_user_polygon=True,
            notes=["user polygon is empty/invalid after buffer(0)"],
        )

    if not bim_walls_m:
        notes.append("no walls provided — footprint = user polygon")
        return StairFootprintPolygon(
            polygon=user_poly,
            used_wall_subtraction=False,
            fell_back_to_user_polygon=True,
            notes=notes,
        )

    # Build wall union. Skip walls that have < 3 vertices and any
    # individual polygons that fail .is_valid (buffer(0) heals most
    # cases, but truly broken ones we drop quietly).
    wall_polys: List[Polygon] = []
    for w in bim_walls_m:
        if len(w) < 3:
            continue
        try:
            wp = Polygon(w).buffer(0)
        except (TypeError, ValueError):
            continue
        if wp.is_empty or not wp.is_valid:
            continue
        wall_polys.append(wp)

    if not wall_polys:
        notes.append("no valid wall polygons after sanitisation")
        return StairFootprintPolygon(
            polygon=user_poly,
            used_wall_subtraction=False,
            fell_back_to_user_polygon=True,
            notes=notes,
        )

    wall_union = unary_union(wall_polys)
    if wall_union.is_empty:
        return StairFootprintPolygon(
            polygon=user_poly,
            used_wall_subtraction=False,
            fell_back_to_user_polygon=True,
            notes=["empty wall union — footprint = user polygon"],
        )

    trimmed = user_poly.difference(wall_union)
    if trimmed.is_empty:
        notes.append("difference is empty — falling back to user polygon")
        return StairFootprintPolygon(
            polygon=user_poly,
            used_wall_subtraction=False,
            fell_back_to_user_polygon=True,
            notes=notes,
        )

    n_holes = 0
    n_dropped = 0

    if isinstance(trimmed, MultiPolygon):
        pieces = sorted(trimmed.geoms, key=lambda g: -g.area)
        largest = pieces[0]
        n_dropped = sum(1 for _ in pieces[1:])
        notes.append(
            f"difference produced {len(pieces)} pieces — kept the largest "
            f"({largest.area:.2f} m²), dropped {n_dropped}"
        )
        trimmed = largest

    # At this point ``trimmed`` is a single Polygon. Sanitise its
    # holes: drop sub-noise rings.
    cleaned_interiors = []
    for ring in trimmed.interiors:
        try:
            ring_poly = Polygon(ring)
            if ring_poly.area >= MIN_HOLE_AREA_M2:
                cleaned_interiors.append(list(ring.coords))
        except (TypeError, ValueError):
            continue
    if len(cleaned_interiors) != len(list(trimmed.interiors)):
        trimmed = Polygon(
            list(trimmed.exterior.coords), holes=cleaned_interiors,
        ).buffer(0)
        if isinstance(trimmed, MultiPolygon):
            trimmed = max(trimmed.geoms, key=lambda g: g.area)
    n_holes = len(list(trimmed.interiors))

    if trimmed.area < MIN_TRIMMED_AREA_M2:
        notes.append(
            f"trimmed area {trimmed.area:.3f} m² < {MIN_TRIMMED_AREA_M2}"
            f" — falling back to user polygon"
        )
        return StairFootprintPolygon(
            polygon=user_poly,
            used_wall_subtraction=False,
            fell_back_to_user_polygon=True,
            notes=notes,
        )

    notes.append(
        f"wall-subtraction OK: {trimmed.area:.2f} m² "
        f"(of user {user_poly.area:.2f} m²), {n_holes} hole(s)"
    )
    return StairFootprintPolygon(
        polygon=trimmed,
        used_wall_subtraction=True,
        n_holes=n_holes,
        n_dropped_pieces=n_dropped,
        fell_back_to_user_polygon=False,
        notes=notes,
    )


def axis_anchor_from_user_polygon(
    user_polygon_m: Sequence[Tuple[float, float]],
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float, float, Tuple[float, float]]]:
    """Pick the user polygon's long axis as a cardinal-direction unit vector.

    The user's bbox encodes their intent: the longer of the bbox's two
    cardinal extents is the stair's *intended* long axis. PCA on the
    wall-trimmed polygon (which may be L-shaped) can pick a diagonal
    axis that contradicts that intent — this helper produces the
    snap-target for that case.

    Returns:
        ``(long_axis, short_axis, length_m, width_m, centroid)`` where
        ``long_axis`` is either ``(1, 0)`` or ``(0, 1)``,
        ``short_axis`` is the 90°-ccw rotation of ``long_axis``,
        and ``length_m`` / ``width_m`` are the cardinal-aligned extents
        of the polygon's axis-aligned bbox. ``None`` if the polygon is
        degenerate (< 3 vertices).
    """
    import numpy as np  # noqa: PLC0415

    if len(user_polygon_m) < 3:
        return None
    pts = np.asarray(user_polygon_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    extent = max_xy - min_xy
    centroid = (min_xy + max_xy) * 0.5
    if extent[0] >= extent[1]:
        long_axis = (1.0, 0.0)
        short_axis = (0.0, 1.0)
        length_m = float(extent[0])
        width_m = float(extent[1])
    else:
        long_axis = (0.0, 1.0)
        # Right-handed perpendicular to (0, 1) is (-1, 0).
        short_axis = (-1.0, 0.0)
        length_m = float(extent[1])
        width_m = float(extent[0])
    return (
        long_axis, short_axis,
        length_m, width_m,
        (float(centroid[0]), float(centroid[1])),
    )


def reproject_polygon_onto_axis(
    polygon: "shapely.geometry.Polygon",  # type: ignore[name-defined]  # noqa: F821
    long_axis: Tuple[float, float],
) -> Tuple[float, float, Tuple[float, float]]:
    """Project a polygon onto ``long_axis`` and return (length, width, centroid).

    Used to redefine length/width AFTER an axis snap so the values
    reflect the polygon's extent along the *snapped* axis, not along
    the original PCA-derived diagonal.
    """
    import numpy as np  # noqa: PLC0415

    try:
        ext = list(polygon.exterior.coords)
    except AttributeError:
        return 0.0, 0.0, (0.0, 0.0)
    if len(ext) >= 2 and ext[0] == ext[-1]:
        ext = ext[:-1]
    if len(ext) < 3:
        return 0.0, 0.0, (0.0, 0.0)
    pts = np.asarray(ext, dtype=np.float64)
    la = np.asarray(long_axis, dtype=np.float64)
    la /= max(np.linalg.norm(la), 1e-9)
    sa = np.array([-la[1], la[0]], dtype=np.float64)
    proj_long = pts @ la
    proj_short = pts @ sa
    length = float(proj_long.max() - proj_long.min())
    width = float(proj_short.max() - proj_short.min())
    centroid_long = float((proj_long.max() + proj_long.min()) * 0.5)
    centroid_short = float((proj_short.max() + proj_short.min()) * 0.5)
    centroid = (
        float(centroid_long * la[0] + centroid_short * sa[0]),
        float(centroid_long * la[1] + centroid_short * sa[1]),
    )
    return length, width, centroid


def pca_axes_from_polygon(
    polygon: "shapely.geometry.Polygon",  # type: ignore[name-defined]  # noqa: F821
) -> Tuple[Tuple[float, float], Tuple[float, float], float, float, Tuple[float, float]]:
    """Return PCA-derived (long_axis, short_axis, length, width, centroid) for a polygon.

    Wraps numpy PCA over the polygon's vertex set (exterior + interior
    rings sampled uniformly so a hole near the centroid does not skew
    the axis). Matches the contract of
    :func:`footprint.extract_stair_footprint` but works on the trimmed
    polygon rather than a raw vertex sequence.
    """
    import numpy as np  # noqa: PLC0415

    pts_xy: List[Tuple[float, float]] = []
    try:
        # ``.coords`` includes the closing duplicate vertex — drop it
        # so the PCA centroid is not biased toward the first corner.
        ext = list(polygon.exterior.coords)
        if len(ext) >= 2 and ext[0] == ext[-1]:
            ext = ext[:-1]
        pts_xy.extend(ext)
    except AttributeError:
        return ((1.0, 0.0), (0.0, 1.0), 0.0, 0.0, (0.0, 0.0))
    for ring in polygon.interiors:
        rc = list(ring.coords)
        if len(rc) >= 2 and rc[0] == rc[-1]:
            rc = rc[:-1]
        pts_xy.extend(rc)

    if len(pts_xy) < 3:
        return ((1.0, 0.0), (0.0, 1.0), 0.0, 0.0, (0.0, 0.0))

    pts = np.asarray(pts_xy, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    cov = np.cov(centred, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return ((1.0, 0.0), (0.0, 1.0), 0.0, 0.0, tuple(centroid))
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis = eigvecs[:, -1]
    if long_axis[0] < 0 or (long_axis[0] == 0 and long_axis[1] < 0):
        long_axis = -long_axis
    short_axis = np.array([-long_axis[1], long_axis[0]], dtype=np.float64)
    proj_long = centred @ long_axis
    proj_short = centred @ short_axis
    length = float(proj_long.max() - proj_long.min())
    width = float(proj_short.max() - proj_short.min())
    return (
        (float(long_axis[0]), float(long_axis[1])),
        (float(short_axis[0]), float(short_axis[1])),
        length,
        width,
        (float(centroid[0]), float(centroid[1])),
    )
