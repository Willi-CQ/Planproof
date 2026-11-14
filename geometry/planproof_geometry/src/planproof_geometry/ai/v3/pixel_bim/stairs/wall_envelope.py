"""Inside-walls filter for stair placement.

V1 scope: only stairs whose footprint centroid sits inside the building
envelope are built. Labels outside the envelope are most likely
labelling artefacts (sample symbols, off-floor markers) and are skipped
with ``SKIPPED_OUTSIDE_WALLS``.

The envelope is the **convex hull** of all wall-polygon shell vertices.
Convex hull is preferred over a union of wall polygons because:
    * walls are thin strips; their union is a network, not a region.
    * a stair touching a wall on one side legitimately overlaps the
      wall ink — strict-inside-union would reject it.
    * the convex hull captures the building footprint, which is what
      "inside the walls" actually means architecturally.

A small inset (1 cm by default) is applied so a stair sitting *exactly*
on the envelope edge still counts as inside.

Input polymorphism (Tag 4 update — for direct integration with
``perfect_bim`` merge pipeline):
    * iterable of objects exposing ``.shell_xy`` (e.g.
      ``extrude_result.polygons``) — original V1-prototype contract.
    * iterable of raw polygons, each ``list[Tuple[float, float]]`` —
      produced by ``merge_doors_and_windows_bim.py`` after the
      post-extrude flattening
      (``bim_walls_m = [list(wp.shell_xy) for wp in result.polygons]``).
Both formats accepted; the function inspects each entry on the fly.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple


def _iter_shell_points(wp) -> Iterable[Tuple[float, float]]:
    """Yield (x, y) tuples from one polygon, regardless of source shape.

    Accepts:
        * object with ``.shell_xy`` attribute holding an iterable of
          (x, y) tuples (original V1-prototype contract).
        * iterable of (x, y) tuples directly (raw polygon list, e.g.
          ``bim_walls_m`` in the merge pipeline).

    Non-numeric / malformed entries are silently skipped — the same
    behaviour the original prototype had via try/except.
    """
    shell = getattr(wp, "shell_xy", None)
    if shell is None:
        try:
            shell = list(wp)
        except TypeError:
            return
    for xy in shell:
        try:
            x = float(xy[0])
            y = float(xy[1])
        except (TypeError, ValueError, IndexError):
            continue
        yield (x, y)


def build_wall_envelope(wall_polygons) -> Optional["shapely.geometry.base.BaseGeometry"]:  # type: ignore[name-defined]  # noqa: F821
    """Compute the convex hull of all wall polygon shells.

    Args:
        wall_polygons: iterable of polygons; each entry is either an
            object with ``.shell_xy`` OR a raw list of ``(x, y)``
            tuples. See module docstring for the two accepted formats.

    Returns:
        A shapely Polygon for the convex hull, or ``None`` if fewer
        than 3 valid vertices were collected.
    """
    from shapely.geometry import MultiPoint  # noqa: PLC0415

    points: List[Tuple[float, float]] = []
    for wp in wall_polygons or []:
        for pt in _iter_shell_points(wp):
            points.append(pt)
    if len(points) < 3:
        return None
    hull = MultiPoint(points).convex_hull
    if hull.is_empty or hull.geom_type not in ("Polygon", "MultiPolygon"):
        return None
    return hull


def build_wall_envelope_from_polys(
    polygons_xy_m: Sequence[Sequence[Tuple[float, float]]],
) -> Optional["shapely.geometry.base.BaseGeometry"]:  # type: ignore[name-defined]  # noqa: F821
    """Convenience wrapper for the raw-polygon path used in the merge pipeline.

    Equivalent to ``build_wall_envelope(polygons_xy_m)`` but with a
    type-annotated signature that documents the
    ``merge_doors_and_windows_bim.py`` integration explicitly. Use this
    in the production hook so the call-site is self-explanatory.

    Args:
        polygons_xy_m: list of polygons, each polygon a list of
            ``(x, y)`` tuples in plan-metres. Typically built as
            ``[list(wp.shell_xy) for wp in extrude_result.polygons]``.

    Returns:
        A shapely Polygon for the convex hull, or ``None``.
    """
    return build_wall_envelope(polygons_xy_m)


def is_centroid_inside(
    centroid_xy_m: Tuple[float, float],
    envelope,
    *,
    tolerance_m: float = 0.01,
) -> bool:
    """Return True if a point is inside (or within ``tolerance_m`` of) the envelope.

    Args:
        centroid_xy_m: (x, y) in metres, plan frame.
        envelope: result of :func:`build_wall_envelope`. ``None`` -> False.
        tolerance_m: distance allowance for points sitting exactly on the
            envelope boundary. Default 1 cm absorbs floating-point noise.
    """
    if envelope is None:
        return False
    from shapely.geometry import Point  # noqa: PLC0415

    pt = Point(float(centroid_xy_m[0]), float(centroid_xy_m[1]))
    if envelope.contains(pt) or envelope.touches(pt):
        return True
    return envelope.distance(pt) <= max(0.0, tolerance_m)
