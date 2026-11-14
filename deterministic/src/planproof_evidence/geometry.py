"""Pixel-space adaptations of ConstructQ line pairing, fitting and room topology.

See deterministic/NOTICE.md for exact source references and deliberate differences.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize_full, unary_union
from shapely.strtree import STRtree
from skimage.morphology import skeletonize

from .contracts import polygon_json, polygons_of


def fit_line(points):
    """Orthogonal least squares (PCA); retains oblique geometry without Manhattan snap."""
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    axis = vt[0]
    if axis[0] < 0 or (abs(axis[0]) < 1e-9 and axis[1] < 0):
        axis = -axis
    t = (points - center) @ axis
    residual = np.linalg.norm((points - center) - t[:, None] * axis, axis=1)
    return np.array([center + t.min() * axis, center + t.max() * axis]), float(
        np.sqrt(np.mean(residual**2))
    )


def line_frame(segment):
    points = np.asarray(segment, dtype=float)
    delta = points[1] - points[0]
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        raise ValueError("Degenerate line segment")
    u = delta / length
    if u[0] < -1e-9 or (abs(u[0]) < 1e-9 and u[1] < 0):
        u = -u
    return u, np.array([-u[1], u[0]]), length


def angle_difference(a, b):
    u, _, _ = line_frame(a)
    v, _, _ = line_frame(b)
    return math.degrees(math.acos(float(np.clip(abs(u @ v), 0, 1))))


def sample_support(mask, segment, radius=1):
    """Fraction of line samples near observed ink (not a correctness probability)."""
    p = np.asarray(segment, dtype=float)
    count = max(2, int(np.linalg.norm(p[1] - p[0])) + 1)
    xy = np.linspace(p[0], p[1], min(count, 8192))
    near = cv2.dilate(mask.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8))
    x = np.clip(np.floor(xy[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.floor(xy[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return float(np.mean(near[y, x] > 0))


def mask_polygons(mask, epsilon=0.75, min_area=1):
    """RETR_CCOMP keeps holes and disconnected islands; simplification must preserve validity."""
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return []
    result = []
    for i, ring in enumerate(contours):
        if hierarchy[0][i][3] != -1 or len(ring) < 3:
            continue
        holes = []
        child = hierarchy[0][i][2]
        while child != -1:
            if len(contours[child]) >= 3:
                holes.append((contours[child][:, 0].astype(float) + 0.5).tolist())
            child = hierarchy[0][child][0]
        polygon = Polygon(ring[:, 0].astype(float) + 0.5, holes)
        if not polygon.is_valid:
            # Repair is explicitly local to the extracted contour; original mask is saved.
            from shapely import make_valid

            polygon = make_valid(polygon)
        for part in polygons_of(polygon):
            simplified = part.simplify(epsilon, preserve_topology=True)
            if simplified.is_valid and simplified.area >= min_area:
                result.append(simplified)
    return sorted(result, key=lambda p: (-p.area, p.bounds))


def merge_collinear(segments, ink, stroke, angle_deg=5):
    """Merge overlaps or ink-supported gaps; keeps all parent indices and fit residuals."""
    if not segments:
        return []
    lines = [LineString(s) for s in segments]
    tree = STRtree(lines)
    assigned = set()
    result = []
    for i in sorted(range(len(lines)), key=lambda j: (-lines[j].length, j)):
        if i in assigned:
            continue
        members = [i]
        fitted = np.asarray(segments[i], dtype=float)
        u, n, _ = line_frame(fitted)
        candidates = tree.query(lines[i].buffer(max(2, 2 * stroke)))
        for j in sorted(int(k) for k in candidates):
            if j == i or j in assigned or angle_difference(fitted, segments[j]) > angle_deg:
                continue
            p = np.asarray(segments[j])
            # Both endpoints must stay close to the same fitted supporting line.
            if np.max(np.abs((p - fitted[0]) @ n)) > max(1, stroke * 0.6):
                continue
            t = (p - fitted[0]) @ u
            length = np.linalg.norm(fitted[1] - fitted[0])
            gap = max(0, float(t.min() - length), float(-t.max()))
            if gap > max(2, 2 * stroke):
                continue
            proposed, residual = fit_line(
                np.concatenate([np.asarray(segments[k]) for k in members + [j]])
            )
            if gap > 0 and sample_support(ink, proposed, max(1, int(stroke / 2))) < 0.9:
                continue
            if residual > max(1, stroke * 0.7):
                continue
            members.append(j)
            fitted = proposed
            u, n, _ = line_frame(fitted)
        assigned.update(members)
        fitted, residual = fit_line(np.concatenate([np.asarray(segments[k]) for k in members]))
        result.append({"segment": fitted.tolist(), "parents": members, "residual_px": residual})
    return result


def pair_boundaries(segments, stroke, angle_deg=5, max_pairs=1200):
    """Mutual nearest opposing lines with shared projected overlap.

    Uses relative normal offset, unlike the translation-sensitive absolute-offset
    sum in legacy strip logic. Output remains a wall *candidate* until ink checks.
    """
    if not segments:
        return [], False
    lines = [LineString(s) for s in segments]
    tree = STRtree(lines)
    candidates = []
    nearest = {}
    for i, a in enumerate(segments):
        u, n, length = line_frame(a)
        a = np.asarray(a)
        for j in tree.query(lines[i].buffer(max(6, stroke * 10))):
            j = int(j)
            if j <= i or angle_difference(a, segments[j]) > angle_deg:
                continue
            b = np.asarray(segments[j])
            offset = float(np.mean((b - a[0]) @ n))
            thickness = abs(offset)
            if not max(1.5, stroke * 0.65) <= thickness <= max(6, stroke * 10):
                continue
            ta = np.sort((a - a[0]) @ u)
            tb = np.sort((b - a[0]) @ u)
            lo, hi = max(ta[0], tb[0]), min(ta[1], tb[1])
            overlap = max(0.0, float(hi - lo))
            if overlap < max(10, 4 * stroke) or overlap / min(length, lines[j].length) < 0.5:
                continue
            side_i = 1 if offset > 0 else -1
            # Each side independently chooses its nearest eligible boundary.
            for key in ((i, side_i), (j, -side_i)):
                if key not in nearest or thickness < nearest[key][0]:
                    nearest[key] = (thickness, (i, j))
            start, end = a[0] + lo * u, a[0] + hi * u
            strip = Polygon([start, end, end + offset * n, start + offset * n])
            candidates.append(
                {
                    "indices": [i, j],
                    "side": side_i,
                    "thickness_px": thickness,
                    "overlap_fraction": overlap / min(length, lines[j].length),
                    "polygon": strip,
                    "centerline": [list(start + offset * n / 2), list(end + offset * n / 2)],
                }
            )
    pairs = [
        p
        for p in candidates
        if nearest[(p["indices"][0], p["side"])][1] == tuple(p["indices"])
        and nearest[(p["indices"][1], -p["side"])][1] == tuple(p["indices"])
    ]
    pairs.sort(key=lambda p: (-p["polygon"].area, p["indices"]))
    return pairs[:max_pairs], len(pairs) > max_pairs


def skeleton_lines(mask, min_length=10):
    binary = mask > 0
    skeleton = skeletonize(binary).astype(np.uint8)
    distances = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    lines = cv2.HoughLinesP(
        skeleton * 255,
        1,
        np.pi / 360,
        threshold=max(8, int(min_length)),
        minLineLength=min_length,
        maxLineGap=1,
    )
    result = []
    for entry in [] if lines is None else lines[:, 0]:
        segment = entry.reshape(2, 2).astype(float) + 0.5
        xy = np.linspace(
            segment[0], segment[1], max(2, int(np.linalg.norm(segment[1] - segment[0])))
        )
        x = np.clip(xy[:, 0].astype(int), 0, binary.shape[1] - 1)
        y = np.clip(xy[:, 1].astype(int), 0, binary.shape[0] - 1)
        result.append(
            {"segment": segment.tolist(), "thickness_px": float(2 * np.median(distances[y, x]))}
        )
    return result, skeleton


def node_graph(segments):
    """Exact intersections only; gaps stay dangles rather than silently becoming walls."""
    if not segments:
        return {
            "nodes": [],
            "edges": [],
            "regions": [],
            "diagnostics": {"dangles": 0, "cuts": 0, "invalid_rings": 0},
        }
    noded = unary_union([LineString(s) for s in segments])
    lines = list(noded.geoms) if noded.geom_type == "MultiLineString" else [noded]
    points = sorted({tuple(p) for line in lines for p in (line.coords[0], line.coords[-1])})
    indices = {p: i for i, p in enumerate(points)}
    edges = [
        {
            "from": indices[tuple(line.coords[0])],
            "to": indices[tuple(line.coords[-1])],
            "length_px": float(line.length),
        }
        for line in lines
    ]
    regions, cuts, dangles, invalid = polygonize_full(lines)
    return {
        "nodes": [list(p) for p in points],
        "edges": edges,
        "regions": [polygon_json(p) for p in polygons_of(regions)],
        "diagnostics": {
            "dangles": len(dangles.geoms),
            "cuts": len(cuts.geoms),
            "invalid_rings": len(invalid.geoms),
        },
    }


def gap_hypotheses(segments, ink, stroke):
    """Explicit unsupported short collinear endpoint gaps, never auto-inserted in graph."""
    ends = [(i, np.asarray(p)) for i, s in enumerate(segments) for p in s]
    if not ends:
        return []
    from scipy.spatial import cKDTree

    tree = cKDTree([p for _, p in ends])
    result = []
    for i, j in sorted(tree.query_pairs(max(2, stroke * 3))):
        a, p = ends[i]
        b, q = ends[j]
        if a == b or np.linalg.norm(p - q) < 0.75 or angle_difference(segments[a], segments[b]) > 5:
            continue
        result.append(
            {
                "segment": [p.tolist(), q.tolist()],
                "parents": [a, b],
                "ink_support": sample_support(ink, [p, q]),
                "status": "uninserted_gap_hypothesis",
            }
        )
    return result


def snap_polygon(polygon, ink, max_distance=2, ink_field=None):
    """Bounded vertex attraction to ink field with topology/hole/area guards."""
    from scipy.ndimage import distance_transform_edt

    if not np.any(ink):
        return polygon, {"accepted": False, "reason": "no ink"}
    distances, indices = (
        ink_field
        if ink_field is not None
        else distance_transform_edt(ink == 0, return_indices=True)
    )

    def ring(coords):
        result = []
        for x, y in coords:
            ix = int(np.clip(x, 0, ink.shape[1] - 1))
            iy = int(np.clip(y, 0, ink.shape[0] - 1))
            if distances[iy, ix] <= max_distance:
                result.append([float(indices[1, iy, ix]) + 0.5, float(indices[0, iy, ix]) + 0.5])
            else:
                result.append([x, y])
        return result

    outer = ring(polygon.exterior.coords)
    holes = [ring(r.coords) for r in polygon.interiors]
    candidate = Polygon(outer, holes)
    displacements = [
        np.linalg.norm(np.asarray(outer) - np.asarray(polygon.exterior.coords), axis=1)
    ]
    displacements.extend(
        np.linalg.norm(np.asarray(new) - np.asarray(old.coords), axis=1)
        for new, old in zip(holes, polygon.interiors)
    )
    max_displacement = max(float(d.max()) for d in displacements)
    accepted = (
        candidate.is_valid
        and not candidate.is_empty
        and len(candidate.interiors) == len(polygon.interiors)
        and abs(candidate.area - polygon.area) <= max(1, polygon.area * 0.1)
        and max_displacement <= max_distance + 0.71
    )
    return (candidate if accepted else polygon), {
        "accepted": bool(accepted),
        "max_displacement_px": max_distance,
    }


def opening_attachment(opening, wall_polygons, tolerance):
    if not wall_polygons:
        return {"status": "unresolved", "alternatives": []}
    tree = STRtree(wall_polygons)
    candidates = [int(i) for i in tree.query(opening.buffer(tolerance))]
    choices = sorted((float(opening.distance(wall_polygons[i])), i) for i in candidates)
    return {
        "status": "candidate" if choices else "unresolved",
        "alternatives": [{"wall_index": i, "distance_px": d} for d, i in choices[:3]],
    }


def project_opening(opening, centerline):
    """PCA/local-axis span from gap_geometry; symbol extent is not clear opening width."""
    u, n, _ = line_frame(centerline)
    origin = np.asarray(centerline[0])
    points = np.concatenate([np.asarray(p.exterior.coords) for p in polygons_of(opening)])
    along, across = (points - origin) @ u, (points - origin) @ n
    return {
        "axis": u.tolist(),
        "projected_symbol_span_px": float(np.ptp(along)),
        "projected_symbol_depth_px": float(np.ptp(across)),
        "clear_width_px": None,
        "definition": "projection of candidate geometry; not measured wall gap",
    }


def net_regions(centerline_regions, wall_polygons):
    """Do not confuse centerline/gross areas with clear interior regions."""
    walls = unary_union(wall_polygons)
    return [part for p in centerline_regions for part in polygons_of(p.difference(walls))]


def clip_polygon(polygon, size):
    return polygons_of(polygon.intersection(box(0, 0, *size)))
