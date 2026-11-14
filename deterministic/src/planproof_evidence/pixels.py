"""Replaceable reference pixel extractor, not a claim of Track 1 completion.

Adapted ideas: wall distance/skeleton/Hough, opening arcs/parallel lines,
stair PCA/repeated treads, conservative contour refinement. See NOTICE.md.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.strtree import STRtree
from skimage.morphology import skeletonize

from .geometry import (
    angle_difference,
    gap_hypotheses,
    line_frame,
    mask_polygons,
    merge_collinear,
    node_graph,
    pair_boundaries,
    skeleton_lines,
)


def candidate(kind, geometry, method, features, support, warnings=None):
    return {
        "class": kind,
        "geometry": geometry,
        "method": method,
        "features": features,
        "support": float(np.clip(support, 0, 1)),
        "warnings": warnings or [],
        "status": "supported_candidate",
    }


def suppress(candidates, threshold=0.35):
    """Complete suppression by overlapping representative, not transitive chaining."""
    result = []
    for c in sorted(candidates, key=lambda c: (-c["support"], c["geometry"].bounds)):
        g = c["geometry"]
        if any(
            g.intersection(r["geometry"]).area / max(g.union(r["geometry"]).area, 1e-9) >= threshold
            for r in result
            if r["class"] == c["class"]
        ):
            continue
        result.append(c)
    return result


def opening_strip_from_arc(ink, center, radius, angle, stroke):
    """Arc-to-opening projection; symbol extent and clear opening width stay separate."""
    center = np.asarray(center, dtype=float)
    u = np.array([math.cos(angle), math.sin(angle)])
    n = np.array([-u[1], u[0]])
    extent = max(4.0, min(radius * 0.3, stroke * 12))
    offsets = np.arange(-math.ceil(extent), math.ceil(extent) + 1, dtype=float)
    bands = []
    h, w = ink.shape
    for t in (-max(stroke * 3, radius * 0.08), radius + max(stroke * 3, radius * 0.08)):
        points = center + t * u + offsets[:, None] * n
        x, y = np.floor(points[:, 0]).astype(int), np.floor(points[:, 1]).astype(int)
        valid = (x >= 0) & (y >= 0) & (x < w) & (y < h)
        hits = offsets[valid & (ink[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)] > 0)]
        negative, positive = hits[hits < -stroke], hits[hits > stroke]
        if len(negative) and len(positive):
            bands.append((float(negative.max()), float(positive.min())))
    if bands:
        low, high = np.median(np.asarray(bands), axis=0)
        status = "flanking_ink_band"
    else:
        low, high = -stroke * 3, stroke * 3
        status = "unresolved_band_stroke_based_proposal"
    corners = [
        center + low * n,
        center + radius * u + low * n,
        center + radius * u + high * n,
        center + high * n,
    ]
    polygon = Polygon(corners).intersection(box(0, 0, w, h))
    return polygon, {
        "band_status": status,
        "band_offsets_px": [float(low), float(high)],
        "opening_axis": u.tolist(),
        "clear_width_px": None,
        "definition": "arc-implied opening span with flanking ink band; not verified clear width",
    }


def detect_arcs(gray, ink, stroke, config):
    """Quarter arc angular coverage plus leaf/hinge support, scale-free pixel bounds."""
    h, w = gray.shape
    lo = max(config.min_arc_radius_px, int(stroke * 3))
    hi = int(min(h, w) * config.max_arc_radius_fraction)
    if hi <= lo:
        return []
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (3, 3), 0.7),
        cv2.HOUGH_GRADIENT,
        dp=1.25,
        minDist=lo,
        param1=100,
        param2=18,
        minRadius=lo,
        maxRadius=hi,
    )
    if circles is None:
        return []
    near = cv2.dilate(ink, np.ones((3, 3), np.uint8)) > 0
    result = []
    theta = np.linspace(0, 2 * np.pi, 144, endpoint=False)

    def support(xy):
        x, y = np.floor(xy[:, 0]).astype(int), np.floor(xy[:, 1]).astype(int)
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        return float(np.mean(near[y[valid], x[valid]])) if valid.all() else 0.0

    for cx, cy, radius in circles[0][:400]:
        ring = np.column_stack((cx + radius * np.cos(theta), cy + radius * np.sin(theta)))
        quadrants = [support(p) for p in np.array_split(ring, 4)]
        q = int(np.argmax(quadrants))
        best = quadrants[q]
        others = sorted(quadrants)[-2]
        if best < config.min_arc_support or others > best * 0.8:
            continue
        a0, a1 = q * np.pi / 2, (q + 1) * np.pi / 2
        endpoints = [
            np.array([cx + radius * math.cos(a), cy + radius * math.sin(a)]) for a in (a0, a1)
        ]
        ray_support = [support(np.linspace([cx, cy], p, max(12, int(radius)))) for p in endpoints]
        leaf = max(ray_support)
        if leaf < 0.65:
            continue
        arc = np.concatenate((ring[q * 36 : (q + 1) * 36], [[cx, cy]], endpoints))
        bounds = [
            max(0, arc[:, 0].min() - stroke),
            max(0, arc[:, 1].min() - stroke),
            min(w, arc[:, 0].max() + stroke),
            min(h, arc[:, 1].max() + stroke),
        ]
        opening, band = opening_strip_from_arc(
            ink, [cx, cy], radius, (a0, a1)[int(np.argmin(ray_support))], stroke
        )
        if opening.is_empty or opening.area <= 0:
            continue
        result.append(
            candidate(
                "door",
                opening,
                "quarter-arc-leaf",
                {
                    "arc_support": best,
                    "other_quadrant_support": others,
                    "leaf_support": leaf,
                    "radius_px": float(radius),
                    "arc_center_px": [float(cx), float(cy)],
                    "symbol_bbox": [float(v) for v in bounds],
                    "ray_support": ray_support,
                    **band,
                },
                min(best, leaf),
                ["swing symbol candidate; assembly/leaf count unresolved"],
            )
        )
    return suppress(result)


def repeated_parallel_symbols(segments, wall_mask, stroke, config, abstentions=None):
    """Local PCA axes and regular line projections, with flank/rail evidence.

    Hatch and furniture ambiguity is retained; a repeated-line pattern alone is
    never sufficient for a stair classification.
    """
    if not segments:
        return []
    lines = [LineString(s) for s in segments]
    tree = STRtree(lines)
    result, seen = [], set()
    h, w = wall_mask.shape
    wall_near = cv2.dilate(wall_mask, np.ones((max(3, int(4 * stroke) | 1),) * 2, np.uint8)) > 0
    for i, s in enumerate(segments):
        u, n, length = line_frame(s)
        if length < max(10, 4 * stroke) or length > min(h, w) * 0.7:
            continue
        origin = np.asarray(s)[0]
        group = []
        for j in tree.query(lines[i].buffer(min(length * 1.2, 100))):
            j = int(j)
            if angle_difference(s, segments[j]) > config.angle_deg:
                continue
            p = np.asarray(segments[j])
            t = (p - origin) @ u
            overlap = max(0, min(length, t.max()) - max(0, t.min()))
            if overlap / length < 0.7 or lines[j].length < 0.65 * length:
                continue
            group.append((float(np.mean((p - origin) @ n)), j))
        group.sort()
        # Collapse the two edge traces of the same raster stroke.
        distinct = []
        for offset, j in group:
            if not distinct or offset - distinct[-1][0] > max(1.5, stroke * 0.7):
                distinct.append((offset, j))
        if len(distinct) < 3:
            continue
        # Break groups at unsupported spacing jumps; do not combine remote flights.
        runs, current = [], []
        for entry in distinct:
            if current and entry[0] - current[-1][0] > max(3 * stroke, length * 0.45):
                runs.append(current)
                current = []
            current.append(entry)
        runs.append(current)
        for run in runs:
            if len(run) < 3:
                continue
            key = tuple(j for _, j in run)
            if key in seen:
                continue
            seen.add(key)
            spacing = np.diff([offset for offset, _ in run])
            cv = float(np.std(spacing) / max(np.mean(spacing), 1e-9))
            if cv > 0.3:
                continue
            projections = [np.asarray(segments[j]) @ u - origin @ u for _, j in run]
            low = max(float(p.min()) for p in projections)
            high = min(float(p.max()) for p in projections)
            if high - low < max(10, length * 0.5):
                continue
            points = np.array(
                [origin + t * u + offset * n for offset, _ in run for t in (low, high)]
            )
            geometry = (
                Polygon(points)
                .convex_hull.buffer(max(1, stroke / 2), join_style=2)
                .intersection(box(0, 0, w, h))
            )
            if geometry.is_empty or geometry.area <= 0:
                continue
            # Boundary rails perpendicular to tread direction, near either run end.
            normal_span = run[-1][0] - run[0][0]
            rails = []
            for j in tree.query(geometry.buffer(max(3, stroke * 2))):
                j = int(j)
                if abs(angle_difference(s, segments[j]) - 90) < config.angle_deg:
                    if (
                        lines[j].intersection(geometry.buffer(stroke * 2)).length
                        >= normal_span * 0.65
                    ):
                        rails.append(j)
            mid = geometry.representative_point()
            context = bool(wall_near[min(h - 1, int(mid.y)), min(w - 1, int(mid.x))])
            regularity = max(0, 1 - cv)
            features = {
                "parallel_line_count": len(run),
                "spacing_cv": cv,
                "rail_count": len(rails),
                "wall_context": context,
            }
            similar_lengths = all(lines[j].length <= 1.5 * length for _, j in run)
            if len(run) >= config.min_stair_treads and len(rails) >= 2 and similar_lengths:
                result.append(
                    candidate(
                        "stairs",
                        geometry,
                        "repeated-treads-rails",
                        features,
                        regularity,
                        ["hatch/furniture ambiguity remains; counts refer to symbol groups"],
                    )
                )
            elif 3 <= len(run) <= 4 and context and similar_lengths and length <= min(h, w) * 0.25:
                result.append(
                    candidate(
                        "window",
                        geometry,
                        "parallel-lines-wall-context",
                        features,
                        regularity,
                        ["glazing symbol candidate; facade assembly count unresolved"],
                    )
                )
            elif abstentions is not None:
                abstentions.append(
                    {
                        "bbox": list(geometry.bounds),
                        "features": features,
                        "class": "unknown",
                        "reason": "parallel pattern lacks sufficient stair rails/window wall-context",
                    }
                )
    return suppress(result)


def polygon_support(polygon, ink, *, boundary=False):
    local = np.zeros_like(ink)
    rings = [
        np.floor(np.asarray(r.coords)).astype(np.int32)
        for r in [polygon.exterior, *polygon.interiors]
    ]
    if boundary:
        cv2.polylines(local, rings, True, 1, 1)
        observed = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    else:
        cv2.fillPoly(local, [rings[0]], 1)
        for hole in rings[1:]:
            cv2.fillPoly(local, [hole], 0)
        observed = ink
    return float(observed[local > 0].mean()) if np.any(local) else 0.0


def detect_window_bands(segments, ink, stroke, config):
    """Bounded parallel glazing bands with endcaps; spacing need not be uniform.

    Window panes are not stair treads: imposing equal spacing loses actual
    multi-line frames. The visible endcaps delimit the candidate assembly.
    """
    if not segments:
        return []
    lines = [LineString(s) for s in segments]
    tree = STRtree(lines)
    near = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    h, w = ink.shape
    results = []

    def support(a, b):
        points = np.linspace(a, b, max(3, round(np.linalg.norm(b - a))))
        x = np.clip(points[:, 0].astype(int), 0, w - 1)
        y = np.clip(points[:, 1].astype(int), 0, h - 1)
        return float(near[y, x].mean())

    for i, segment in enumerate(segments):
        u, n, length = line_frame(segment)
        if not max(20, stroke * 8) <= length <= min(h, w) * 0.6:
            continue
        origin = np.asarray(segment[0])
        offsets = []
        # Inspect wider context than the candidate band. Otherwise four cropped
        # lines from a longer tread/hatch sequence can masquerade as a window.
        for j in tree.query(lines[i].buffer(min(stroke * 16, length * 0.4))):
            j = int(j)
            if angle_difference(segment, segments[j]) > config.angle_deg:
                continue
            points = np.asarray(segments[j]) - origin
            t = points @ u
            overlap = max(0, min(length, t.max()) - max(0, t.min()))
            if overlap < length * 0.6:
                continue
            offsets.append(float(np.mean(points @ n)))
        distinct = []
        for offset in sorted(offsets):
            if not distinct or offset - distinct[-1] > max(1.5, stroke * 0.6):
                distinct.append(offset)
        if not 3 <= len(distinct) <= 8:
            continue
        low, high = distinct[0], distinct[-1]
        if not stroke * 2 <= high - low <= length * 0.22:
            continue
        spacing = np.diff(distinct)
        regularity_cv = float(np.std(spacing) / max(np.mean(spacing), 1e-9))
        if len(distinct) >= config.min_stair_treads and regularity_cv < 0.2:
            continue  # repeated hatch/tread ambiguity
        left = support(origin + low * n, origin + high * n)
        right = support(origin + length * u + low * n, origin + length * u + high * n)
        if min(left, right) < 0.55:
            continue
        polygon = Polygon(
            [
                origin + low * n,
                origin + length * u + low * n,
                origin + length * u + high * n,
                origin + high * n,
            ]
        ).intersection(box(0, 0, w, h))
        if polygon.is_empty:
            continue
        results.append(
            candidate(
                "window",
                polygon,
                "parallel-glazing-band-endcaps",
                {
                    "parallel_line_count": len(distinct),
                    "endcap_support": [left, right],
                    "spacing_cv": regularity_cv,
                    "band_thickness_px": high - low,
                },
                min(left, right),
                ["glazing-band candidate; furniture/frame ambiguity remains"],
            )
        )
    return suppress(results)


def wall_mask_from_ink(ink):
    """Shared scale-free wall-body preparation for both pixel front ends."""
    skeleton = skeletonize(ink > 0)
    distances = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    values = distances[skeleton]
    stroke = max(1.0, float(2 * np.percentile(values, 35))) if len(values) else 1.0
    core = (distances >= max(1.5, stroke * 0.4)).astype(np.uint8)
    k = max(3, int(stroke) | 1)
    expanded = cv2.dilate(core, np.ones((k, k), np.uint8)) & ink
    length = max(9, int(stroke * 5))
    horizontal = cv2.morphologyEx(expanded, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    vertical = cv2.morphologyEx(expanded, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    return cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8)) & ink, stroke


def extract_pixels(image, config):
    start = time.perf_counter()
    original_h, original_w = image.shape[:2]
    scale = min(1.0, config.max_side / max(original_h, original_w))
    w, h = max(1, round(original_w * scale)), max(1, round(original_h * scale))
    working = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA) if scale < 1 else image
    gray = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    wall, stroke = wall_mask_from_ink(ink)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = lsd.detect(gray)[0]
    raw = (
        []
        if detected is None
        else [
            s.reshape(2, 2).astype(float).tolist()
            for s in detected
            if np.linalg.norm(s.reshape(2, 2)[1] - s.reshape(2, 2)[0]) >= config.min_line_px
        ]
    )
    return vectorize_pixels(
        image,
        {
            "working_size": [w, h],
            "working_to_canonical": [[original_w / w, 0, 0], [0, original_h / h, 0], [0, 0, 1]],
            "ink": ink,
            "wall_mask": wall,
            "raw_segments": raw,
            "stroke_px": stroke,
            "gray": gray,
        },
        config,
        start=start,
    )


def vectorize_pixels(image, primitives, config, *, start=None):
    """Track 2 consumes Track 1 masks/segments; Track 1 need not fit any geometry."""
    start = time.perf_counter() if start is None else start
    original_h, original_w = image.shape[:2]
    w, h = primitives["working_size"]
    expected_transform = [[original_w / w, 0, 0], [0, original_h / h, 0], [0, 0, 1]]
    if not np.allclose(primitives["working_to_canonical"], expected_transform, atol=1e-9):
        raise ValueError("Track 1 working transform does not match the original raster")
    if primitives.get("mode", "independent") != "independent":
        raise ValueError("ML-driven pixels cannot enter independent vectorization")
    ink = np.asarray(primitives["ink"])
    wall = np.asarray(primitives["wall_mask"]).copy()
    for mask in (ink, wall):
        if mask.shape != (h, w) or mask.dtype != np.uint8 or not np.isin(mask, [0, 1]).all():
            raise ValueError("Track 1 masks must be uint8 binary arrays in the working frame")
    gray = primitives.get("gray")
    if gray is None:
        gray = cv2.cvtColor(
            cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY
        )
    distances = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    values = distances[skeletonize(ink > 0)]
    stroke = primitives.get(
        "stroke_px", max(1.0, float(2 * np.percentile(values, 35))) if len(values) else 1.0
    )
    if not np.isfinite(stroke) or stroke <= 0:
        raise ValueError("Invalid measured pixel stroke")
    raw = list(primitives["raw_segments"])
    for segment in raw:
        points = np.asarray(segment)
        if (
            points.shape != (2, 2)
            or not np.isfinite(points).all()
            or np.linalg.norm(points[1] - points[0]) < 1e-6
        ):
            raise ValueError("Invalid raw Track 1 segment")
    raw.sort(key=lambda s: (-LineString(s).length, s))
    truncated = len(raw) > config.max_segments
    raw = raw[: config.max_segments]
    merged = merge_collinear(raw, ink, stroke, config.angle_deg)
    segments = [m["segment"] for m in merged]
    pairs, pairs_truncated = pair_boundaries(segments, stroke, config.angle_deg, config.max_pairs)
    accepted_pairs = []
    for pair in pairs:
        p = pair["polygon"]
        local = np.zeros_like(ink)
        cv2.fillPoly(local, [np.round(np.asarray(p.exterior.coords)).astype(np.int32)], 1)
        support = float(ink[local > 0].mean()) if np.any(local) else 0
        pair["ink_fill_fraction"] = support
        # Filled paired outlines are useful wall evidence; hollow outlines remain
        # candidates in vector diagnostics, not authoritative filled walls.
        if support >= 0.55:
            wall |= local
            accepted_pairs.append(pair)
    candidates = list(primitives.get("candidates", []))
    for p in mask_polygons(wall, min_area=max(20, stroke**2 * 5)):
        candidates.append(
            candidate(
                "wall",
                p,
                "filled-core-and-boundary-pairs",
                {"area_px2": float(p.area), "ink_fill_fraction": polygon_support(p, ink)},
                polygon_support(p, ink),
                ["wall body candidate; physical wall count unavailable"],
            )
        )
    # Free-space components of observed wall bodies. No generic morphological gap
    # closing: open doorways produce merged regions and are reported as such.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1 - wall, connectivity=4)
    for label in range(1, count):
        x, y, rw, rh, area = stats[label]
        if x == 0 or y == 0 or x + rw == w or y + rh == h:
            continue
        if (
            not config.min_room_area_fraction * w * h
            <= area
            <= config.max_room_area_fraction * w * h
        ):
            continue
        for p in mask_polygons((labels == label).astype(np.uint8), min_area=area * 0.5):
            candidates.append(
                candidate(
                    "room",
                    p,
                    "observed-wall-free-space",
                    {
                        "area_px2": float(p.area),
                        "wall_boundary_support": polygon_support(p, wall, boundary=True),
                    },
                    polygon_support(p, wall, boundary=True),
                    [
                        "enclosed region; door gaps can merge semantic rooms",
                        "room category unknown without text",
                    ],
                )
            )
    symbols = detect_arcs(gray, ink, stroke, config)
    abstentions = list(primitives.get("unclassified_patterns", []))
    symbols += repeated_parallel_symbols(segments, wall, stroke, config, abstentions)
    symbols += detect_window_bands(segments, ink, stroke, config)
    candidates.extend(suppress(symbols))
    # Skeleton graph uses wall bodies, never all drawing strokes/dimensions.
    centers, skeleton_wall = skeleton_lines(wall, max(config.min_line_px, stroke * 4))
    graph = node_graph([c["segment"] for c in centers])
    gaps = gap_hypotheses([c["segment"] for c in centers], ink, stroke)
    warnings = list(primitives.get("warnings", []))
    if truncated:
        warnings.append("line budget truncated: dense drawing coverage partial")
    if pairs_truncated:
        warnings.append("wall pair budget truncated")
    return {
        "working_size": [w, h],
        "working_to_canonical": [[original_w / w, 0, 0], [0, original_h / h, 0], [0, 0, 1]],
        "ink": ink,
        "wall_mask": wall,
        "skeleton_mask": skeleton_wall,
        "stroke_px": stroke,
        "raw_segments": raw,
        "merged": merged,
        "pairs": pairs,
        "accepted_pair_count": len(accepted_pairs),
        "centerlines": centers,
        "graph": graph,
        "gaps": gaps,
        "candidates": candidates,
        "unclassified_patterns": abstentions,
        "text_spans": primitives.get("text_spans", []),
        "pixel_details": primitives.get(
            "pixel_details", {"backend": "reference", "independent": True}
        ),
        "warnings": warnings,
        "availability": {
            "walls": "reference_adapter",
            "rooms": "enclosed_regions_only",
            "doors": "swing_symbols_only",
            "windows": "parallel_symbols_only",
            "stairs": "treads_and_rails_only",
            "ocr": "unavailable",
            "room_categories": "requires_text",
            **primitives.get("availability", {}),
        },
        "seconds": time.perf_counter() - start,
    }
