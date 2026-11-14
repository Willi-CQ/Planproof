"""Gated assignment, complete-link duplicate groups and member-backed candidate counts."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import box
from shapely.strtree import STRtree

from .contracts import CLASSES, item_geometry


def iou(a, b):
    return float(a.intersection(b).area / max(a.union(b).area, 1e-12))


def assign(scores, threshold=0.5):
    """Maximum cardinality first, then IoU; forbidden edges never consume real matches."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("Expected score matrix")
    if not scores.size:
        return []
    bonus = sum(scores.shape) + 1
    cost = np.where(scores >= threshold, -bonus - scores, 0)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if scores[r, c] >= threshold]


def duplicate_groups(items, threshold=0.8):
    """Complete link: A~B and B~C do not imply A~C. Retains all raw observations."""
    groups = []
    geometries = {o["id"]: item_geometry(o) for o in items}
    for item in sorted(items, key=lambda o: o["id"]):
        placed = False
        for group in groups:
            if all(
                o["class"] == item["class"]
                and geometries[o["id"]].intersects(geometries[item["id"]])
                and iou(geometries[o["id"]], geometries[item["id"]]) >= threshold
                for o in group
            ):
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])
    return [[o["id"] for o in group] for group in groups if len(group) > 1]


def compare_sources(independent, ml, threshold=0.5):
    geometries = [item_geometry(o) for o in independent]
    ml_geometry = [item_geometry(o) for o in ml]
    scores = np.zeros((len(independent), len(ml)))
    if ml_geometry:
        tree = STRtree(ml_geometry)
        for i, g in enumerate(geometries):
            for j in tree.query(g):
                j = int(j)
                if independent[i]["class"] == ml[j]["class"]:
                    scores[i, j] = iou(g, ml_geometry[j])
    matches = assign(scores, threshold)
    used_i, used_j = {i for i, _ in matches}, {j for _, j in matches}
    result = []
    for i, j in matches:
        result.append(
            {
                "relation": "agreement",
                "source_ids": [independent[i]["id"], ml[j]["id"]],
                "iou": float(scores[i, j]),
                "bbox": list(geometries[i].union(ml_geometry[j]).bounds),
                "semantics": "spatial/class agreement, not proof of correctness",
            }
        )
    for i, a in enumerate(independent):
        if i in used_i:
            continue
        overlapping = [
            (iou(geometries[i], g), j)
            for j, g in enumerate(ml_geometry)
            if geometries[i].intersects(g)
        ]
        best = max(overlapping, default=(0, None))
        if best[0] >= threshold and a["class"] != ml[best[1]]["class"]:
            relation, parents = "class_conflict", [a["id"], ml[best[1]]["id"]]
        elif best[0] > 0.1 and a["class"] == ml[best[1]]["class"]:
            relation, parents = "geometry_conflict", [a["id"], ml[best[1]]["id"]]
        else:
            relation, parents = "unmatched_geometry", [a["id"]]
        result.append(
            {"relation": relation, "source_ids": parents, "iou": best[0], "bbox": a["bbox"]}
        )
    for j, b in enumerate(ml):
        if j not in used_j:
            result.append(
                {
                    "relation": "unmatched_ml",
                    "source_ids": [b["id"]],
                    "bbox": b["bbox"],
                    "semantics": "classical non-detection is not a negative finding",
                }
            )
    return result


def aggregates(observations, page_index, source, available_classes=None):
    units = {
        "wall": "wall_body_candidates",
        "room": "enclosed_region_candidates",
        "door": "arc_supported_opening_candidates",
        "window": "glazing_symbol_candidates",
        "stairs": "tread_group_candidates",
    }
    result = []
    for kind in CLASSES:
        if available_classes is not None and kind not in available_classes:
            result.append(
                {
                    "page_index": page_index,
                    "source": source,
                    "class": kind,
                    "counting_unit": "ml_instances",
                    "member_ids": [],
                    "duplicate_groups": [],
                    "value": None,
                    "candidate_count": None,
                    "exact_physical_count": None,
                    "status": "unavailable",
                    "coverage": "model does not predict this class",
                }
            )
            continue
        items = [o for o in observations if o.get("class") == kind]
        groups = duplicate_groups(items)
        suppressed = {oid for g in groups for oid in g[1:]}
        members = [o["id"] for o in items if o["id"] not in suppressed]
        result.append(
            {
                "page_index": page_index,
                "source": source,
                "class": kind,
                "counting_unit": units[kind] if source == "classical" else "ml_instances",
                "member_ids": members,
                "duplicate_groups": groups,
                "value": None if kind == "wall" else len(members),
                "candidate_count": len(members),
                "exact_physical_count": None,
                "status": "candidate_count_only",
                "coverage": "partial_recognition; no absence guarantee",
            }
        )
    return result


def room_relations(rooms, openings, distance_px=3):
    """Geometric contact is separate from confirmed passage/accessibility."""
    result = []
    if not rooms:
        return result
    geoms = [item_geometry(o) for o in rooms]
    tree = STRtree(geoms)
    for opening in openings:
        geometry = item_geometry(opening)
        nearby = [
            int(i)
            for i in tree.query(geometry.buffer(distance_px))
            if geoms[int(i)].distance(geometry) <= distance_px
        ]
        result.append(
            {
                "opening_id": opening["id"],
                "nearby_region_ids": [rooms[i]["id"] for i in nearby],
                "relation": "possible_passage"
                if opening["class"] == "door" and len(nearby) == 2
                else "geometric_contact",
                "verified_passage": False,
            }
        )
    return result


def contextual_box(bbox, size, padding):
    result = box(*bbox).buffer(padding).intersection(box(0, 0, *size))
    return list(result.bounds)
