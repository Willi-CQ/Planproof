"""ML-assisted room boundaries; no labels, inference, or physical-wall invention."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from shapely.affinity import scale as scale_geometry
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree
from skimage.morphology import skeletonize

from .contracts import (
    item_geometry,
    polygon_json,
    polygons_of,
    sha256,
    stable_id,
    validate_document,
    write_json,
)
from .geometry import mask_polygons, skeleton_lines, snap_polygon


@dataclass(frozen=True)
class RoomRules:
    version: str = "room-boundaries-1"
    angle_deg: float = 10
    lateral_thickness: float = 0.5
    min_gap_thickness: float = 0.5
    max_gap_thickness: float = 12
    max_endpoints: int = 600
    arc_support: float = 0.60
    leaf_support: float = 0.65
    clear_fraction: float = 0.08
    min_room_fraction: float = 0.001
    apply_refinements: bool = False

    def __post_init__(self):
        values = asdict(self)
        if any(
            not math.isfinite(v) or v <= 0
            for v in values.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ):
            raise ValueError("Room rule values must be finite and positive")
        if self.angle_deg >= 90 or self.min_gap_thickness >= self.max_gap_thickness:
            raise ValueError("Invalid room search limits")
        if (
            not isinstance(self.apply_refinements, bool)
            or not isinstance(self.max_endpoints, int)
            or any(
                not 0 < value <= 1
                for value in (
                    self.arc_support,
                    self.leaf_support,
                    self.clear_fraction,
                    self.min_room_fraction,
                )
            )
        ):
            raise ValueError("Invalid fraction or endpoint budget")


def rasterize(geometry, size):
    """Rasterize each island separately so its holes cannot erase another island."""
    mask = np.zeros((size[1], size[0]), np.uint8)
    for polygon in polygons_of(geometry):
        part = np.zeros_like(mask)
        cv2.fillPoly(part, [np.round(polygon.exterior.coords).astype(np.int32)], 1)
        for hole in polygon.interiors:
            cv2.fillPoly(part, [np.round(hole.coords).astype(np.int32)], 0)
        mask |= part
    return mask


def sample(mask, points):
    points = np.asarray(points)
    xy = np.round(points).astype(int)
    valid = (
        (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] < mask.shape[0])
    )
    result = np.zeros(len(xy), dtype=float)
    result[valid] = mask[xy[valid, 1], xy[valid, 0]]
    return result


def line_points(a, b, count=None):
    return np.linspace(a, b, count or max(12, int(np.linalg.norm(np.asarray(b) - a) * 2)))


def _wall_endpoints(wall, stroke, limit):
    """Walk actual skeleton terminals; arbitrary Hough segment ends are not endcaps."""
    skeleton = skeletonize(wall > 0).astype(np.uint8)
    degree = cv2.filter2D(skeleton, cv2.CV_16S, np.ones((3, 3), np.int16)) - skeleton
    distances = cv2.distanceTransform(wall, cv2.DIST_L2, 5)
    terminals = np.argwhere((skeleton > 0) & (degree == 1))
    endpoints = []
    for y, x in terminals[:limit]:
        path = [(int(y), int(x))]
        thickness = max(2 * stroke, 2 * float(distances[y, x]))
        for _ in range(max(8, int(thickness * 2))):
            cy, cx = path[-1]
            nearby = [
                (cy + dy, cx + dx)
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if (dy or dx)
                and 0 <= cy + dy < wall.shape[0]
                and 0 <= cx + dx < wall.shape[1]
                and skeleton[cy + dy, cx + dx]
                and (cy + dy, cx + dx) not in path
            ]
            if len(nearby) != 1:
                break
            path.append(nearby[0])
        if len(path) < 5:
            continue
        coordinates = np.asarray([[px, py] for py, px in path], float)
        # Thinning curls toward a corner of the endcap. Fit the shaft behind it,
        # rather than treating that terminal curl as the wall's orientation.
        shaft = coordinates[2:] if len(coordinates) >= 8 else coordinates
        centered = shaft - shaft.mean(axis=0)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        direction = axes[0]
        if direction @ (coordinates[0] - coordinates[-1]) < 0:
            direction = -direction
        origin = shaft.mean(axis=0) + direction * (
            (coordinates[0] - shaft.mean(axis=0)) @ direction
        )
        normal = np.array([-direction[1], direction[0]])
        thickness = max(2 * stroke, float(np.median([2 * distances[py, px] for py, px in path])))
        # The skeleton terminal is usually half a wall thickness inside the endcap.
        ray = origin + np.arange(0, 2 * thickness + 2)[:, None] * direction
        empty = np.flatnonzero(sample(wall, ray) == 0)
        if not len(empty):
            continue
        tip = ray[empty[0]]
        cap_points = (
            tip
            - direction * max(1, stroke)
            + np.linspace(-0.35, 0.35, 13)[:, None] * thickness * normal
        )
        cap_support = float(sample(wall, cap_points).mean())
        endpoints.append(
            {
                "point": tip,
                "direction": direction,
                "thickness": thickness,
                "cap_support": cap_support,
            }
        )
    return endpoints, len(terminals) > limit


def _gap_candidates(wall, stroke, rules):
    ends, truncated = _wall_endpoints(wall, stroke, rules.max_endpoints)
    gaps, rejected = [], []
    if not ends:
        return gaps, rejected, truncated
    tree = cKDTree([end["point"] for end in ends])
    cosine = math.cos(math.radians(rules.angle_deg))
    for i, a in enumerate(ends):
        for j in sorted(
            tree.query_ball_point(a["point"], rules.max_gap_thickness * a["thickness"] * 2)
        ):
            if j <= i:
                continue
            b = ends[j]
            vector = b["point"] - a["point"]
            length = float(np.linalg.norm(vector))
            thickness = (a["thickness"] + b["thickness"]) / 2
            if (
                length < rules.min_gap_thickness * thickness
                or length > rules.max_gap_thickness * thickness
            ):
                continue
            direction = vector / length
            alignment = [float(a["direction"] @ direction), float(-b["direction"] @ direction)]
            if min(alignment) < cosine:
                continue
            offset = abs(float(np.cross(a["direction"], vector)))
            ratio = a["thickness"] / b["thickness"]
            reason = None
            if offset > rules.lateral_thickness * thickness:
                reason = "wall_axes_not_aligned"
            elif not 0.5 <= ratio <= 2:
                reason = "wall_thickness_mismatch"
            elif min(a["cap_support"], b["cap_support"]) < 0.6:
                reason = "weak_opposing_endcaps"
            occupied = float(
                sample(
                    wall, line_points(a["point"] + 0.1 * vector, b["point"] - 0.1 * vector)
                ).mean()
            )
            if occupied > 0.20:
                reason = "wall_continues_through_gap"
            entry = {
                "segment": [a["point"].tolist(), b["point"].tolist()],
                "thickness_px": thickness,
                "length_px": length,
                "axis_alignment": alignment,
                "lateral_offset_px": offset,
                "endcap_support": [a["cap_support"], b["cap_support"]],
                "wall_occupancy": occupied,
            }
            if reason:
                rejected.append({**entry, "reason": reason})
            else:
                gaps.append(entry)
    # A terminal belongs to the nearest compatible partner, not every distant pair.
    chosen, occupied_ends = [], set()
    for gap in sorted(gaps, key=lambda g: (g["length_px"], g["segment"])):
        keys = [tuple(p) for p in gap["segment"]]
        if any(key in occupied_ends for key in keys):
            rejected.append({**gap, "reason": "nearer_competing_gap"})
            continue
        occupied_ends.update(keys)
        chosen.append(gap)
    return chosen, rejected, truncated


def classify_gap(gap, ink, stroke, rules=None):
    """Gap-first rotated templates: symbol and wall endpoints must agree spatially."""
    rules = rules or RoomRules()
    a, b = np.asarray(gap["segment"], float)
    u = (b - a) / np.linalg.norm(b - a)
    n = np.array([-u[1], u[0]])
    length, thickness = gap["length_px"], gap["thickness_px"]
    radius = max(1, int(round(stroke / 2)))
    near = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
    arcs = []
    for hinge, inward in ((a, u), (b, -u)):
        for side in (-1, 1):
            for fraction in (1.0, 0.5):
                r = length * fraction
                for offset in (0.0, -0.4, 0.4):
                    pivot = hinge + offset * thickness * n
                    for degrees in (60, 75, 90, 105):
                        theta = np.linspace(math.radians(12), math.radians(degrees - 8), 40)
                        arc = pivot + r * (
                            np.cos(theta)[:, None] * inward + side * np.sin(theta)[:, None] * n
                        )
                        angle = math.radians(degrees)
                        leaf_end = pivot + r * (
                            math.cos(angle) * inward + side * math.sin(angle) * n
                        )
                        leaf = line_points(pivot + (leaf_end - pivot) * 0.15, leaf_end)
                        support, leaf_support = (
                            float(sample(near, arc).mean()),
                            float(sample(near, leaf).mean()),
                        )
                        arcs.append(
                            {
                                "arc_support": support,
                                "leaf_support": leaf_support,
                                "hinge": pivot.tolist(),
                                "leaf_end": leaf_end.tolist(),
                                "radius_px": r,
                                "swing_side": side,
                                "angle_deg": degrees,
                                "half_span": fraction == 0.5,
                            }
                        )
    best = max(arcs, key=lambda x: min(x["arc_support"], x["leaf_support"]))
    door = best["arc_support"] >= rules.arc_support and best["leaf_support"] >= rules.leaf_support
    offsets = np.arange(-0.65 * thickness, 0.65 * thickness + 1, max(0.5, stroke / 2))
    line_support = [
        float(
            sample(
                ink, line_points(a + 0.12 * length * u + off * n, b - 0.12 * length * u + off * n)
            ).mean()
        )
        for off in offsets
    ]
    groups = []
    for off, support in zip(offsets, line_support):
        if support >= 0.70:
            if not groups or off - groups[-1][-1] > max(0.6, stroke * 0.75):
                groups.append([])
            groups[-1].append(float(off))
    glazing_count = len(groups)
    across = np.linspace(-0.45 * thickness, 0.45 * thickness, 20)[:, None] * n
    endcaps = [
        float(sample(near, endpoint + across).mean())
        for endpoint in (a - stroke * u, b + stroke * u)
    ]
    window = 2 <= glazing_count <= 4 and min(endcaps) >= 0.55
    xx, yy = np.meshgrid(
        np.linspace(0.15 * length, 0.85 * length, 40),
        np.linspace(-0.35 * thickness, 0.35 * thickness, 12),
    )
    passage = a + xx.reshape(-1, 1) * u + yy.reshape(-1, 1) * n
    ink_fraction = float(sample(ink, passage).mean())
    if door and window:
        kind, reason = "unknown", "conflicting_door_and_glazing_symbols"
    elif door:
        kind, reason = "door", "attached_leaf_and_swing_arc"
    elif window:
        kind, reason = "window", "bounded_parallel_glazing_at_wall_ends"
    elif ink_fraction <= rules.clear_fraction and best["arc_support"] < 0.4:
        kind, reason = "walkthrough", "clear_bounded_passage_no_supported_symbol"
    else:
        kind, reason = "unknown", "symbol_evidence_inconclusive"
    return {
        "type": kind,
        "reason": reason,
        "features": {
            "best_swing": best,
            "glazing_stroke_groups": glazing_count,
            "glazing_offsets_px": [float(np.mean(g)) for g in groups],
            "glazing_endcap_support": endcaps,
            "passage_ink_fraction": ink_fraction,
        },
        "alternatives": ["missing_or_damaged_door_symbol"] if kind == "walkthrough" else [],
        "status": "rule_supported_hypothesis" if kind != "unknown" else "unresolved_symbol",
    }


def _profile_gaps(wall, ink, stroke, rules):
    """Find hollow interruptions between solid endcaps, even when frames join the skeleton."""
    centers, _ = skeleton_lines(wall, max(12, stroke * 4))
    candidates = []
    for center in centers:
        a, b = np.asarray(center["segment"])
        length = float(np.linalg.norm(b - a))
        t = max(2 * stroke, center["thickness_px"])
        u = (b - a) / max(length, 1e-9)
        n = np.array([-u[1], u[0]])
        extension = rules.max_gap_thickness * t + t
        distances = np.arange(-extension, length + extension, 1.0)
        axis = a + distances[:, None] * u
        offsets = np.linspace(-0.35 * t, 0.35 * t, max(5, int(t)))
        samples = axis[:, None, :] + offsets[None, :, None] * n
        fill = sample(ink, samples.reshape(-1, 2)).reshape(len(axis), -1).mean(axis=1)
        # Filled wall flanks, not mere collinear linework, anchor this hypothesis.
        solid = fill >= 0.65
        change = np.diff(np.r_[False, ~solid, False].astype(int))
        for begin, end in zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)):
            span = end - begin
            flank = max(3, int(t))
            if begin < flank or end + flank >= len(axis):
                continue
            if not rules.min_gap_thickness * t <= span <= rules.max_gap_thickness * t:
                continue
            cap_support = [
                float(fill[begin - flank : begin].mean()),
                float(fill[end : end + flank].mean()),
            ]
            if min(cap_support) < 0.65 or float(fill[begin:end].mean()) > 0.35:
                continue
            p, q = axis[begin], axis[end - 1]
            # Both endcaps must also be attached to the ML wall proposal.
            near = [
                float(sample(wall, line_points(p - t * u, p - u)).mean()),
                float(sample(wall, line_points(q + u, q + t * u)).mean()),
            ]
            if min(near) < 0.65:
                continue
            candidates.append(
                {
                    "segment": [p.tolist(), q.tolist()],
                    "thickness_px": t,
                    "length_px": float(np.linalg.norm(q - p)),
                    "axis_alignment": [1.0, 1.0],
                    "lateral_offset_px": 0.0,
                    "endcap_support": cap_support,
                    "wall_occupancy": float(sample(wall, axis[begin:end]).mean()),
                    "pixel_gap_fill": float(fill[begin:end].mean()),
                    "method": "hollow_cross_section_between_solid_endcaps",
                }
            )
    return candidates


def _serialize_geometry(geometry, sx, sy):
    canonical = scale_geometry(geometry, sx, sy, origin=(0, 0))
    return {
        "bbox": list(canonical.bounds),
        "polygons": [polygon_json(p) for p in polygons_of(canonical)],
    }


def _partition(wall, separators):
    barrier = wall.copy()
    for separator in separators:
        if separator["active"]:
            a, b = np.round(separator["segment"]).astype(int)
            cv2.line(
                barrier,
                tuple(a),
                tuple(b),
                1,
                max(1, int(round(separator["thickness_px"]))),
                cv2.LINE_8,
            )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1 - barrier, connectivity=4)
    exterior = set(
        int(x)
        for x in np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    ) - {0}
    return barrier, labels, stats, exterior


def derive_rooms(frame, observations, ink, raw_vectors, stroke, model_source="main", rules=None):
    """Derive one page in working pixels, serialize all geometry in canonical pixels."""
    rules = rules or RoomRules()
    h, w = ink.shape
    sx, sy = frame["image_size"][0] / w, frame["image_size"][1] / h
    ml = [
        o
        for o in observations
        if o.get("source_family") == "ml"
        and o.get("model_binding", {}).get("name") == model_source
        and o["page_index"] == frame["page_index"]
    ]
    wall_sources = [o for o in ml if o["class"] == "wall"]
    seeds = [o for o in ml if o["class"] == "room"]
    result = {
        "schema_version": "planproof-room-boundaries-v1",
        "frame": frame,
        "model_source": model_source,
        "mode": "ml_assisted",
        "rules": asdict(rules),
        "working_size": [w, h],
        "working_to_canonical": [[sx, 0, 0], [0, sy, 0], [0, 0, 1]],
        "dependency_ids": [o["id"] for o in wall_sources + seeds],
        "walls": [],
        "openings": [],
        "separators": [],
        "rooms": [],
        "adjacency": [],
        "warnings": [
            "ML-assisted geometry is not independent confirmation or a physical room count."
        ],
        "diagnostics": {},
    }
    if not wall_sources:
        result["status"] = "unavailable_no_ml_walls"
        return result
    initial = np.zeros_like(ink)
    local_walls = []
    for o in wall_sources:
        g = scale_geometry(item_geometry(o), 1 / sx, 1 / sy, origin=(0, 0))
        initial |= rasterize(g, [w, h])
        local_walls.extend(polygons_of(g))
    distance = cv2.distanceTransform(initial, cv2.DIST_L2, 5)
    skeleton = skeletonize(initial > 0)
    thickness = float(np.median(2 * distance[skeleton])) if skeleton.any() else 2 * stroke
    thickness = max(2 * stroke, thickness)
    wall = np.zeros_like(initial)
    field = distance_transform_edt(1 - ink, return_indices=True)
    refinements = []
    for polygon in local_walls:
        refined, report = snap_polygon(
            polygon, ink, max_distance=min(2 * stroke, thickness * 0.25), ink_field=field
        )
        wall |= rasterize(refined, [w, h])
        refinements.append(report)
    # Support hollow paired outlines only near a learned wall body; never all page strokes.
    nearby_ml = cv2.dilate(initial, np.ones((max(3, int(2 * stroke) | 1),) * 2, np.uint8))
    outline_additions = 0
    for pair in raw_vectors.get("boundary_pairs", []):
        if "polygon" not in pair:
            continue
        value = pair["polygon"]
        polygon = (
            Polygon(value["outer"], value.get("holes", [])) if isinstance(value, dict) else value
        )
        if not polygon.is_valid:
            continue
        band = rasterize(polygon, [w, h])
        if band.any() and float(nearby_ml[band > 0].mean()) >= 0.9:
            wall |= band
            outline_additions += 1
    gaps, rejected, truncated = _gap_candidates(wall, stroke, rules)
    for proposed in _profile_gaps(wall, ink, stroke, rules):
        geometry = LineString(proposed["segment"])
        if any(
            geometry.distance(LineString(g["segment"]))
            < min(g["thickness_px"], proposed["thickness_px"])
            and geometry.intersection(LineString(g["segment"]).buffer(g["thickness_px"])).length
            > 0.4 * geometry.length
            for g in gaps
        ):
            continue
        gaps.append(proposed)
    separators = []
    for gap in gaps:
        oid = stable_id("opening", [frame["page_index"], model_source, gap["segment"]])
        classification = classify_gap(gap, ink, stroke, rules)
        segment = np.asarray(gap["segment"])
        geometry = LineString(segment).buffer(gap["thickness_px"] / 2, cap_style=2)
        opening = {
            "id": oid,
            **classification,
            **_serialize_geometry(geometry, sx, sy),
            "segment": (segment * [sx, sy]).tolist(),
            "wall_ids": [],
            "adjacent_region_ids": [],
            "gap_rule": gap,
            "dependency_ids": [o["id"] for o in wall_sources],
            "type_is_hypothesis": True,
        }
        best = opening["features"]["best_swing"]
        for name in ("hinge", "leaf_end"):
            best[name] = (np.asarray(best[name]) * [sx, sy]).tolist()
        opening["feature_frame"] = "working_pixels_except_explicit_canonical_hinge_leaf"
        result["openings"].append(opening)
        separators.append(
            {
                "id": oid + ":separator",
                "opening_id": oid,
                "active": True,
                "segment": gap["segment"],
                "thickness_px": gap["thickness_px"],
                "physical_wall": False,
                "reason": "opposing_endcaps_support_partition_only",
            }
        )
    barrier, labels, stats, exterior = _partition(wall, separators)
    cropped = bool(np.any(wall[0]) or np.any(wall[-1]) or np.any(wall[:, 0]) or np.any(wall[:, -1]))
    if cropped:
        result["warnings"].append("Wall reaches image edge; exterior classification is unresolved.")
    seed_masks, associations, fallbacks = {}, {}, []
    for seed in seeds:
        local = scale_geometry(item_geometry(seed), 1 / sx, 1 / sy, origin=(0, 0))
        mask = rasterize(local, [w, h])
        seed_masks[seed["id"]] = mask
        ids, counts = np.unique(labels[mask > 0], return_counts=True)
        candidates = sorted(
            [(int(c), int(i)) for i, c in zip(ids, counts) if i and i not in exterior], reverse=True
        )
        if candidates and candidates[0][0] >= 0.2 * max(1, mask.sum()):
            associations.setdefault(candidates[0][1], []).append(seed["id"])
        else:
            fallbacks.append((seed, local))
    local_regions = []
    for label in range(1, len(stats)):
        if label in exterior or stats[label, cv2.CC_STAT_AREA] < rules.min_room_fraction * w * h:
            continue
        parts = mask_polygons((labels == label).astype(np.uint8), min_area=4)
        for part_index, polygon in enumerate(parts):
            parent_ids = associations.get(label, [])
            rid = stable_id("region", [frame["page_index"], model_source, label, part_index])
            status = (
                "shared_open_region"
                if len(parent_ids) > 1
                else "bounded_seed_region"
                if parent_ids
                else "unseeded_enclosure"
            )
            local_regions.append((rid, label, polygon, parent_ids, status))
    interior_labels = {
        label for _, label, _, parents, _ in local_regions if label is not None and parents
    }
    # Components are useful search hypotheses, but must not replace several good ML
    # instances with one merged room, or split one multipart ML instance into many.
    result["enclosures"] = [
        {
            "id": rid,
            "component_label": label,
            "ml_seed_ids": parents,
            "status": status,
            **_serialize_geometry(polygon, sx, sy),
        }
        for rid, label, polygon, parents, status in local_regions
    ]
    components = local_regions
    local_regions = []
    refinement_decisions = []
    alternatives = {}
    for seed in seeds:
        original = scale_geometry(item_geometry(seed), 1 / sx, 1 / sy, origin=(0, 0))
        options = [entry for entry in components if seed["id"] in entry[3]]
        option = max(options, key=lambda x: original.intersection(x[2]).area, default=None)
        accepted, reason, label = False, "no_bounded_component", None
        geometry, status = original, "unresolved_ml_boundary"
        overlap = None
        if option:
            _, label, proposed, parents, _ = option
            overlap = original.intersection(proposed).area / max(
                original.union(proposed).area, 1e-9
            )
            if len(parents) > 1:
                status, reason, label = (
                    "shared_space_ml_boundary",
                    "multiple_seeds_share_unseparated_space",
                    None,
                )
            elif overlap >= 0.80 and 0.8 <= proposed.area / max(original.area, 1e-9) <= 1.2:
                accepted, reason = True, "bounded_component_agrees_with_seed"
                alternatives[seed["id"]] = _serialize_geometry(proposed, sx, sy)
                if rules.apply_refinements:
                    geometry, status = proposed, "bounded_seed_region"
                else:
                    status = "supported_boundary_alternative"
            else:
                reason, label = "component_would_change_seed_excessively", None
        rid = stable_id("region", [frame["page_index"], model_source, seed["id"]])
        local_regions.append((rid, label, geometry, [seed["id"]], status))
        refinement_decisions.append(
            {
                "room_id": rid,
                "seed_id": seed["id"],
                "accepted": accepted,
                "applied_to_primary": accepted and rules.apply_refinements,
                "reason": reason,
                "seed_component_iou": overlap,
            }
        )
    centers, _ = skeleton_lines(wall, max(6, stroke * 3))
    local_wall_records = []
    for center in centers:
        a, b = np.asarray(center["segment"])
        length = np.linalg.norm(b - a)
        if length == 0:
            continue
        u = (b - a) / length
        n = np.array([-u[1], u[0]])
        t = float(center.get("thickness_px", thickness))
        chunks = max(1, math.ceil(length / max(6 * t, 20)))
        for k in range(chunks):
            p, q = a + (b - a) * k / chunks, a + (b - a) * (k + 1) / chunks
            mid = line_points(p, q, 7)
            sides = [
                sample(labels, mid + sign * (t / 2 + 2 * stroke) * n).astype(int)
                for sign in (-1, 1)
            ]
            roles = []
            for left, right in zip(*sides):
                if cropped:
                    roles.append("unknown")
                elif (left in interior_labels and right in exterior) or (
                    right in interior_labels and left in exterior
                ):
                    roles.append("outer")
                elif left in interior_labels and right in interior_labels:
                    roles.append("inner")
                else:
                    roles.append("unknown")
            role = max(set(roles), key=lambda value: (roles.count(value), value))
            if roles.count(role) < 5:
                role = "unknown"
            wid = stable_id(
                "wall-boundary", [frame["page_index"], model_source, p.tolist(), q.tolist()]
            )
            geom = LineString([p, q]).buffer(t / 2, cap_style=2)
            entry = {
                "id": wid,
                "role": role,
                "segment": (np.asarray([p, q]) * [sx, sy]).tolist(),
                "thickness_working_px": t,
                "role_samples": roles,
                "dependency_ids": [o["id"] for o in wall_sources],
                **_serialize_geometry(geom, sx, sy),
            }
            result["walls"].append(entry)
            local_wall_records.append((wid, LineString([p, q]), t))
    near_wall = cv2.dilate(wall, np.ones((3, 3), np.uint8))
    separator_mask = (barrier > wall).astype(np.uint8)
    near_separator = cv2.dilate(separator_mask, np.ones((3, 3), np.uint8))
    wall_tree = STRtree([g for _, g, _ in local_wall_records])
    max_wall_thickness = max((t for _, _, t in local_wall_records), default=0)
    for rid, label, polygon, parents, status in local_regions:
        boundaries = []
        supported_length, perimeter = 0.0, 0.0
        for ring in [r for part in polygons_of(polygon) for r in [part.exterior, *part.interiors]]:
            coords = list(ring.coords)
            for a, b in zip(coords, coords[1:]):
                line = LineString([a, b])
                if line.length < 1e-6:
                    continue
                points = line_points(a, b)
                wall_support = float(sample(near_wall, points).mean())
                portal_support = float(sample(near_separator, points).mean())
                kind = (
                    "wall"
                    if wall_support >= 0.6
                    else "virtual_separator"
                    if portal_support >= 0.6
                    else "unresolved"
                )
                perimeter += line.length
                if kind != "unresolved":
                    supported_length += line.length
                wall_ids = [
                    local_wall_records[int(i)][0]
                    for i in sorted(wall_tree.query(line.buffer(max_wall_thickness)))
                    if line.distance(local_wall_records[int(i)][1]) <= local_wall_records[int(i)][2]
                ]
                opening_ids = [
                    op["id"]
                    for op, sep in zip(result["openings"], separators)
                    if line.distance(LineString(sep["segment"])) <= sep["thickness_px"]
                ]
                boundaries.append(
                    {
                        "id": stable_id("edge", [rid, a, b]),
                        "type": kind,
                        "segment": (np.asarray([a, b]) * [sx, sy]).tolist(),
                        "wall_ids": wall_ids,
                        "opening_ids": opening_ids,
                        "wall_support": wall_support,
                        "separator_support": portal_support,
                    }
                )
        result["rooms"].append(
            {
                "id": rid,
                "status": status,
                "ml_seed_ids": parents,
                "boundary_alternative": alternatives.get(parents[0]),
                "primary_geometry": "refined_component"
                if status == "bounded_seed_region"
                else "retained_ml_seed",
                "boundary_segments": boundaries,
                "boundary_support": supported_length / max(perimeter, 1e-9),
                "opening_ids": [],
                "neighbor_ids": [],
                "component_label": label,
                "unresolved_sections": [e["id"] for e in boundaries if e["type"] == "unresolved"],
                **_serialize_geometry(polygon, sx, sy),
            }
        )
    for opening, sep in zip(result["openings"], separators):
        a, b = np.asarray(sep["segment"])
        n = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
        sides = [
            int(
                sample(labels, [(a + b) / 2 + sign * (sep["thickness_px"] / 2 + 2 * stroke) * n])[0]
            )
            for sign in (-1, 1)
        ]
        # Associate retained ML regions locally too; leaked global components alone
        # cannot establish which side of this opening a seed belongs to.
        side_points = [
            (a + b) / 2 + sign * (sep["thickness_px"] / 2 + 2 * stroke) * n for sign in (-1, 1)
        ]
        side_rooms = []
        for point, side in zip(side_points, sides):
            matches = []
            for room, (_, label, geom, _, _) in zip(result["rooms"], local_regions):
                if geom.distance(Point(point)) <= 2 * stroke or (
                    label is not None and label == side
                ):
                    matches.append(room["id"])
            side_rooms.append(sorted(set(matches)))
        adjacent = [r for r in result["rooms"] if any(r["id"] in ids for ids in side_rooms)]
        opening["side_region_candidates"] = side_rooms
        opening["adjacency_ambiguous"] = any(len(ids) > 1 for ids in side_rooms)
        opening["adjacent_region_ids"] = [room["id"] for room in adjacent]
        opening["exterior_side"] = not cropped and any(side in exterior for side in sides)
        opening["wall_ids"] = [
            wid
            for wid, line, t in local_wall_records
            if min(line.distance(Point(a)), line.distance(Point(b))) <= 2 * t
        ]
        for room in adjacent:
            room["opening_ids"].append(opening["id"])
        if (
            len(adjacent) == 2
            and all(len(ids) == 1 for ids in side_rooms)
            and side_rooms[0] != side_rooms[1]
        ):
            for room in adjacent:
                room["neighbor_ids"].extend(r["id"] for r in adjacent if r["id"] != room["id"])
            result["adjacency"].append(
                {
                    "opening_id": opening["id"],
                    "region_ids": opening["adjacent_region_ids"],
                    "relation": "possible_passage"
                    if opening["type"] in ("door", "walkthrough")
                    else "window_boundary"
                    if opening["type"] == "window"
                    else "unresolved_opening",
                    "verified_access": False,
                }
            )
        result["separators"].append(
            {
                **sep,
                "segment": (np.asarray(sep["segment"]) * [sx, sy]).tolist(),
                "thickness_working_px": sep["thickness_px"],
                "coordinate_frame": "canonical_pixels",
            }
        )
    for room in result["rooms"]:
        room["neighbor_ids"] = sorted(set(room["neighbor_ids"]))
    # Retain ML openings that could not be attached; never cut the wall just to agree.
    hints = []
    for o in ml:
        if o["class"] not in ("door", "window"):
            continue
        hint = item_geometry(o)
        attached = [op for op in result["openings"] if hint.intersects(item_geometry(op))]
        for opening in attached:
            opening.setdefault("ml_class_alternatives", []).append(
                {"observation_id": o["id"], "class": o["class"]}
            )
        if not attached:
            hints.append(
                {
                    "observation_id": o["id"],
                    "class": o["class"],
                    "bbox": o["bbox"],
                    "reason": "no_supported_opposing_wall_gap; retained_for_inspection",
                }
            )
    for opening in result["openings"]:
        alternatives = opening.setdefault("ml_class_alternatives", [])
        opening["classification_conflict"] = any(
            hint["class"] != opening["type"] for hint in alternatives
        )
    result["status"] = (
        "partial"
        if cropped or truncated or any(not d["accepted"] for d in refinement_decisions)
        else "available_hypotheses"
    )
    result["diagnostics"] = {
        "measured_wall_thickness_working_px": thickness,
        "stroke_working_px": stroke,
        "wall_refinements": refinements,
        "supported_outline_bands": outline_additions,
        "rejected_gaps": rejected,
        "endpoint_budget_truncated": truncated,
        "wall_touches_image_edge": cropped,
        "unattached_ml_openings": hints,
        "exterior_labels": sorted(exterior),
        "seed_count": len(seeds),
        "refinement_decisions": refinement_decisions,
        "region_count_is_not_physical_room_count": True,
    }
    return result


def build_room_boundaries(evidence_path, out_dir, *, model_source="main", rules=None):
    """Postprocess an existing SHA-bound job without rewriting its evidence or predictions."""
    evidence_path = Path(evidence_path).resolve()
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    validate_document(document)
    artifacts = {entry["path"]: entry for entry in document["artifacts"]}
    consumed = {}

    def verified(relative):
        path = (evidence_path.parent / relative).resolve()
        if not path.is_relative_to(evidence_path.parent) or relative not in artifacts:
            raise ValueError("Unregistered/out-of-job room input")
        digest = sha256(path)
        if digest != artifacts[relative]["sha256"]:
            raise ValueError("Room input artifact changed: " + relative)
        consumed[relative] = digest
        return path

    pages = []
    for page in document["pages"]:
        prefix = f"page-{page['frame']['page_index']:03d}/"
        raw = json.loads(verified(prefix + "pixel-evidence.json").read_text())
        vectors = json.loads(verified(prefix + "vector-evidence.json").read_text())
        with Image.open(verified(prefix + "ink.png")) as image:
            ink = (np.asarray(image) > 0).astype(np.uint8)
        verified(page["source_raster"])
        if list(ink.shape[::-1]) != raw["working_size"]:
            raise ValueError("Room mask/working frame mismatch")
        sx, sy = (
            page["frame"]["image_size"][0] / ink.shape[1],
            page["frame"]["image_size"][1] / ink.shape[0],
        )
        expected = [[sx, 0, 0], [0, sy, 0], [0, 0, 1]]
        if not all(
            np.allclose(value["working_to_canonical"], expected) for value in (raw, vectors)
        ):
            raise ValueError("Room source transform mismatch")
        derived = derive_rooms(
            page["frame"],
            document["observations"],
            ink,
            vectors,
            raw["stroke_px"],
            model_source,
            rules,
        )
        derived["source_raster"] = page["source_raster"]
        pages.append(derived)
    result = {
        "schema_version": "planproof-room-boundary-bundle-v1",
        "document_id": document["document_id"],
        "source_sha256": document["source_sha256"],
        "source_evidence_path": str(evidence_path),
        "source_evidence_sha256": sha256(evidence_path),
        "implementation_sha256": sha256(Path(__file__)),
        "input_artifact_hashes": consumed,
        "labels_used_for_inference": False,
        "pages": pages,
    }
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "room-boundaries.json"
    if target == evidence_path:
        raise ValueError("Cannot overwrite source evidence")
    write_json(target, result)
    return result


def get_room_context(boundaries_path, room_id, out_dir):
    """Return an ordinary clean crop plus a separate room/opening overlay and typed context."""
    from .pipeline import get_region

    boundaries_path = Path(boundaries_path).resolve()
    bundle = json.loads(boundaries_path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != "planproof-room-boundary-bundle-v1":
        raise ValueError("Unsupported room-boundary schema")
    evidence = Path(bundle["source_evidence_path"])
    if sha256(evidence) != bundle["source_evidence_sha256"]:
        raise ValueError("Source evidence changed after room derivation")
    matches = [
        (page, room) for page in bundle["pages"] for room in page["rooms"] if room["id"] == room_id
    ]
    if len(matches) != 1:
        raise ValueError("Room ID must identify one registered region")
    page, room = matches[0]
    geometry = item_geometry(room)
    bbox = np.asarray(room["bbox"], float)
    if not np.isfinite(bbox).all() or geometry.is_empty or not geometry.is_valid:
        raise ValueError("Invalid room geometry")
    walls = [
        wall
        for wall in page["walls"]
        if wall["id"] in {wid for edge in room["boundary_segments"] for wid in edge["wall_ids"]}
    ]
    openings = [o for o in page["openings"] if o["id"] in room["opening_ids"]]
    sx, sy = page["working_to_canonical"][0][0], page["working_to_canonical"][1][1]
    thickness = page["diagnostics"]["measured_wall_thickness_working_px"] * max(sx, sy)
    padding = max(2 * thickness, 0.05 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    # Include attached opening symbols and their swings, without painting over source pixels.
    for opening in openings:
        boxes = [opening["bbox"]]
        if opening["type"] == "door":
            hinge = np.asarray(opening["features"]["best_swing"]["hinge"])
            radius = opening["features"]["best_swing"]["radius_px"] * max(sx, sy)
            boxes.append([*(hinge - radius), *(hinge + radius)])
        for other in boxes:
            bbox[:2] = np.minimum(bbox[:2], other[:2])
            bbox[2:] = np.maximum(bbox[2:], other[2:])
    out = Path(out_dir).resolve()
    if out == boundaries_path.parent or out == evidence.parent:
        raise ValueError("Use a separate room crop directory")
    region = get_region(
        evidence, page["frame"]["page_index"], bbox.tolist(), out, padding_px=padding
    )
    source_context = json.loads((out / "agent-context.json").read_text(encoding="utf-8"))
    context = {
        "schema_version": "planproof-room-context-v1",
        "room_id": room_id,
        "document_id": bundle["document_id"],
        "frame": page["frame"],
        "mode": "ml_assisted",
        "room": room,
        "source_evidence_context": source_context,
        "walls": walls,
        "openings": openings,
        "separators": [s for s in page["separators"] if s["opening_id"] in room["opening_ids"]],
        "neighbor_ids": room["neighbor_ids"],
        "boundaries_sha256": sha256(boundaries_path),
        "policy": [
            "Clean crop is unmodified; overlay is separate.",
            "Virtual separators are not physical walls.",
            "Walkthrough may be a door with a missing symbol; passage is not verified.",
            "Shared/open or unresolved ML regions are not verified semantic rooms.",
            "No metric scale or physical room count is established.",
        ],
    }
    write_json(out / "agent-context.json", context)
    with Image.open(out / "clean.png") as image:
        overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    offset = np.asarray(region["bbox"][:2])
    for edge in room["boundary_segments"]:
        color = {"wall": "#238169", "virtual_separator": "#9655d0", "unresolved": "#d64a50"}[
            edge["type"]
        ]
        draw.line([tuple(p) for p in np.asarray(edge["segment"]) - offset], fill=color, width=3)
    for index, opening in enumerate(openings, 1):
        a, b = np.asarray(opening["segment"]) - offset
        draw.line([tuple(a), tuple(b)], fill="#df8422", width=4)
        draw.text(
            tuple(a),
            f"O{index}: {opening['type']}",
            fill="#222222",
            stroke_width=1,
            stroke_fill="white",
        )
    overlay.save(out / "room-overlay.png")
    region.update(
        room_id=room_id,
        overlay="room-overlay.png",
        overlay_sha256=sha256(out / "room-overlay.png"),
        context_sha256=sha256(out / "agent-context.json"),
    )
    write_json(out / "region.json", region)
    return region
