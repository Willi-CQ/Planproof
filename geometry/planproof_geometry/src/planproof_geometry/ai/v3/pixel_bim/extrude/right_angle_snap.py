"""Right-angle enforcement for axis-aligned wall polygons.

After ``axis_snap`` the long edges of a wall polygon collapse onto
shared X/Y rails, but short connector edges (wall-stirnen / Z-step
stubs / end-caps) are usually too short to make it into the axis-snap
classification — they keep the slanted angle of the original user
polygon. Visually the wall ends look "tipped".

This stage inspects every short edge whose two neighbours are already
axial. If both neighbours are horizontal, the short edge is forced to
be vertical (and vice versa). If the neighbours are co-axial (e.g.
end-cap of a wall pointing in the same direction), the short edge is
likewise forced to be perpendicular to that axis.

Snap mechanics:
  * Move only the two endpoints of the short edge.
  * For target='V' (vertical short edge): set both endpoints' X to
    their average — the long horizontal neighbours stay on their Y
    rails because we don't touch the endpoints' Y values.
  * For target='H': set both endpoints' Y to their average.

This is independent of pixel_snap (which handles long edges) and runs
AFTER axis_snap so the long edges already sit on the global rails.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def _classify_edge(
    dx: float, dy: float, axis_tol_deg: float,
) -> str:
    """Return 'H', 'V' or 'D' for an edge vector."""
    if dx == 0.0 and dy == 0.0:
        return "D"
    axis_tol_rad = math.radians(axis_tol_deg)
    angle = math.atan2(abs(dy), abs(dx))
    if angle <= axis_tol_rad:
        return "H"
    if angle >= math.pi / 2.0 - axis_tol_rad:
        return "V"
    return "D"


def enforce_right_angles_polygon(
    coords: Sequence[Tuple[float, float]],
    *,
    short_edge_max_m: float = 0.30,
    axis_tol_deg: float = 5.0,
    max_passes: int = 3,
) -> List[Tuple[float, float]]:
    """Force short connector edges onto right angles relative to their
    axial neighbours.

    Parameters
    ----------
    coords
        Closed polygon coordinates in metres. First vertex may equal
        last vertex (shapely-style); both are kept on output.
    short_edge_max_m
        Edges longer than this are left alone — they're usually
        legitimate diagonals or genuine long walls.
    axis_tol_deg
        Tolerance for classifying a neighbour edge as horizontal /
        vertical. Use the same value passed to axis_snap so we treat
        edges consistently.
    max_passes
        Run the snap up to N times; rounds out cascades where snapping
        one short edge unlocks the classification of an adjacent one.

    Returns
    -------
    list of (x, y) — same length as input, with right-angle-forced
    short edges. Polygon closure (first == last) is preserved.
    """
    n = len(coords)
    if n < 5:
        return [tuple(p) for p in coords]

    closed = (
        coords[0][0] == coords[-1][0] and coords[0][1] == coords[-1][1]
    )
    work: List[Tuple[float, float]] = list(
        coords[:-1] if closed else coords
    )
    m = len(work)
    if m < 4:
        return [tuple(p) for p in coords]

    for _ in range(max_passes):
        changed = False
        # Re-classify every iteration — earlier snaps may have changed
        # neighbours' classifications.
        edge_class: List[str] = []
        for i in range(m):
            j = (i + 1) % m
            dx = work[j][0] - work[i][0]
            dy = work[j][1] - work[i][1]
            edge_class.append(_classify_edge(dx, dy, axis_tol_deg))

        for i in range(m):
            if edge_class[i] not in ("D",):
                # Already axis-aligned (or zero-length); nothing to do.
                continue
            j = (i + 1) % m
            dx = work[j][0] - work[i][0]
            dy = work[j][1] - work[i][1]
            length = math.hypot(dx, dy)
            if length > short_edge_max_m:
                continue
            prev_i = (i - 1) % m
            prev_class = edge_class[prev_i]
            next_class = edge_class[j]

            if prev_class == "H" and next_class == "H":
                target = "V"
            elif prev_class == "V" and next_class == "V":
                target = "H"
            elif prev_class == "H" or next_class == "H":
                # short edge is an end-cap relative to one horizontal
                # long neighbour → force vertical so the cap is square.
                target = "V"
            elif prev_class == "V" or next_class == "V":
                target = "H"
            else:
                continue  # both neighbours diagonal — leave alone

            xi, yi = work[i]
            xj, yj = work[j]
            if target == "V":
                mid_x = 0.5 * (xi + xj)
                work[i] = (mid_x, yi)
                work[j] = (mid_x, yj)
            else:  # "H"
                mid_y = 0.5 * (yi + yj)
                work[i] = (xi, mid_y)
                work[j] = (xj, mid_y)
            changed = True

        if not changed:
            break

    out: List[Tuple[float, float]] = [
        (float(x), float(y)) for x, y in work
    ]
    if closed:
        out.append(out[0])
    return out


__all__ = ["enforce_right_angles_polygon"]
