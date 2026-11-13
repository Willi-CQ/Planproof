"""Step 9: Classify elements using geometric heuristics (+ optional ML fusion)."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from ..ml.semantic_segmentation import INDEX_TO_LABEL


class HeuristicClassifyStep:
    name = "Heuristic classify"
    soft_fail = True

    def run(self, ctx, state):
        wall_planes = state.wall_planes or []
        wall_segments = state.wall_segments_2d or []
        openings = state.openings or []
        points = state.points
        floor_plane = state.floor_plane

        room_height = state.metrics.get("room_height_m", 2.80)

        labels: List[Dict] = []

        # Classify wall elements
        hull_indices = _convex_hull_segments(wall_segments) if wall_segments else set()

        for i, (wp, seg) in enumerate(zip(wall_planes, wall_segments)):
            features = _extract_wall_features(
                wp, seg, points, floor_plane, room_height, i, hull_indices
            )
            classification = _classify_wall(features)
            labels.append({
                "element_id": seg["id"],
                "element_type": "wall",
                "plane_idx": i,
                **classification,
                "geometry_features": features,
                "source": "heuristic",
            })

        # Classify opening elements
        for opening in openings:
            features = _extract_opening_features(opening, room_height)
            classification = _classify_opening_element(features)
            labels.append({
                "element_id": opening["id"],
                "element_type": "opening",
                **classification,
                "geometry_features": features,
                "source": "heuristic",
            })

        # --- ML-Fusion: Pretrained semantische Labels (Step 3.5) ---
        # Wenn verfuegbar, per-Element Mehrheits-Voting gegen die Heuristik fahren.
        semantic_labels = getattr(state, "semantic_labels", None)
        semantic_confidence = getattr(state, "semantic_confidence", None)
        if semantic_labels is not None and points is not None:
            labels = _fuse_with_semantic(
                labels=labels,
                wall_planes=wall_planes,
                openings=openings,
                points=points,
                semantic_labels=semantic_labels,
                semantic_confidence=semantic_confidence,
                fusion_weight=float(state.config.get("semantic_fusion_weight", 0.6)),
                log=ctx.log,
            )
            state.metrics["semantic_fusion_applied"] = True
        else:
            state.metrics["semantic_fusion_applied"] = False

        state.heuristic_labels = labels
        state.metrics["elements_classified"] = len(labels)

        if labels:
            avg_conf = np.mean([l["primary_confidence"] for l in labels])
            state.metrics["avg_heuristic_confidence"] = round(float(avg_conf), 3)

        ctx.log(f"Classified {len(labels)} elements (avg confidence: {state.metrics.get('avg_heuristic_confidence', 0):.2f})")
        return state


def _extract_wall_features(
    wall_plane: Dict,
    segment: Dict,
    points: np.ndarray,
    floor_plane: Optional[Dict],
    room_height: float,
    wall_idx: int,
    hull_indices: set,
) -> Dict:
    """Extract geometric features for wall classification."""
    inlier_mask = wall_plane["inlier_mask"]
    wall_pts = points[inlier_mask]

    # Thickness estimation from point spread perpendicular to wall
    normal = np.array(wall_plane["normal"])
    centroid = np.array(wall_plane["centroid"])
    relative = wall_pts - centroid
    perp_dists = np.abs(relative @ normal)
    thickness = float(np.percentile(perp_dists, 95) * 2)  # 95th percentile spread

    # Height from Z extent
    z_range = wall_pts[:, 2].max() - wall_pts[:, 2].min()

    return {
        "length_m": segment["length_m"],
        "height_m": round(float(z_range), 2),
        "thickness_m": round(thickness, 3),
        "is_on_convex_hull": wall_idx in hull_indices,
        "inlier_count": wall_plane["inlier_count"],
        "fit_quality": wall_plane["fit_quality"],
        "room_height_ratio": round(float(z_range / room_height), 2) if room_height > 0 else 0,
    }


def _classify_wall(features: Dict) -> Dict:
    """Classify a wall segment based on geometric features."""
    thickness = features["thickness_m"]
    is_boundary = features["is_on_convex_hull"]
    length = features["length_m"]

    # Column detection: short, roughly square footprint
    if length < 0.5 and thickness > 0.1:
        return {
            "primary_class": "column",
            "primary_confidence": 0.55,
            "alt_classes": [
                {"class": "wall_interior", "confidence": 0.30},
                {"class": "wall_exterior", "confidence": 0.15},
            ],
        }

    # Exterior wall: thick + on boundary
    if is_boundary and thickness > 0.18:
        return {
            "primary_class": "wall_exterior",
            "primary_confidence": 0.70,
            "alt_classes": [
                {"class": "wall_interior", "confidence": 0.20},
                {"class": "column", "confidence": 0.10},
            ],
        }

    if is_boundary:
        return {
            "primary_class": "wall_exterior",
            "primary_confidence": 0.60,
            "alt_classes": [
                {"class": "wall_interior", "confidence": 0.30},
                {"class": "column", "confidence": 0.10},
            ],
        }

    # Interior wall: thin + not on boundary
    if thickness < 0.15:
        return {
            "primary_class": "wall_interior",
            "primary_confidence": 0.65,
            "alt_classes": [
                {"class": "wall_exterior", "confidence": 0.20},
                {"class": "column", "confidence": 0.15},
            ],
        }

    # Default: ambiguous
    return {
        "primary_class": "wall_interior",
        "primary_confidence": 0.50,
        "alt_classes": [
            {"class": "wall_exterior", "confidence": 0.35},
            {"class": "column", "confidence": 0.15},
        ],
    }


def _extract_opening_features(opening: Dict, room_height: float) -> Dict:
    """Extract features for opening classification."""
    return {
        "width_m": opening["width_m"],
        "height_m": opening["height_m"],
        "base_height_m": opening["base_height_m"],
        "aspect_ratio": round(opening["height_m"] / max(opening["width_m"], 0.01), 2),
        "height_ratio": round(opening["height_m"] / room_height, 2) if room_height > 0 else 0,
    }


def _classify_opening_element(features: Dict) -> Dict:
    """Classify an opening based on geometric features."""
    base = features["base_height_m"]
    height = features["height_m"]
    width = features["width_m"]

    # Door: base near floor, tall
    if base < 0.3 and height > 1.5:
        conf = 0.75 if (0.7 <= width <= 1.2 and height > 1.9) else 0.55
        return {
            "primary_class": "door",
            "primary_confidence": conf,
            "alt_classes": [
                {"class": "window", "confidence": round(1.0 - conf - 0.05, 2)},
                {"class": "archway", "confidence": 0.05},
            ],
        }

    # Window: elevated base
    if base > 0.5:
        conf = 0.70 if (0.5 <= width <= 2.0 and 0.5 <= height <= 1.5) else 0.50
        return {
            "primary_class": "window",
            "primary_confidence": conf,
            "alt_classes": [
                {"class": "door", "confidence": round(1.0 - conf - 0.05, 2)},
                {"class": "vent", "confidence": 0.05},
            ],
        }

    # Ambiguous
    return {
        "primary_class": opening_type_from_original(features),
        "primary_confidence": 0.45,
        "alt_classes": [
            {"class": "door", "confidence": 0.30},
            {"class": "window", "confidence": 0.25},
        ],
    }


def opening_type_from_original(features: Dict) -> str:
    """Fallback: guess from base height."""
    return "door" if features["base_height_m"] < 0.4 else "window"


def _convex_hull_segments(segments: List[Dict]) -> set:
    """Find wall segment indices that lie on the convex hull."""
    if len(segments) < 3:
        return set(range(len(segments)))

    all_points = []
    for seg in segments:
        all_points.append(seg["a"])
        all_points.append(seg["b"])

    pts = np.array(all_points)

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        hull_vertex_set = set(hull.vertices)
    except Exception:
        return set(range(len(segments)))

    hull_indices = set()
    for i, seg in enumerate(segments):
        a_idx = i * 2
        b_idx = i * 2 + 1
        if a_idx in hull_vertex_set or b_idx in hull_vertex_set:
            hull_indices.add(i)

    return hull_indices


def _fuse_with_semantic(
    *,
    labels: List[Dict],
    wall_planes: List[Dict],
    openings: List[Dict],
    points: np.ndarray,
    semantic_labels: np.ndarray,
    semantic_confidence: Optional[np.ndarray],
    fusion_weight: float,
    log: Callable[[str], None],
) -> List[Dict]:
    """Fuses pretrained per-point labels into the heuristic element labels.

    Per wall: Mehrheits-Voting ueber die RANSAC-inlier-Punkte der Wand. Die
    Semantik-Klassen werden auf Planproof-Wand-Klassen gemappt
    (wall → wall_interior, column → column). Passt Mehrheitsklasse zu
    Heuristik, wird die Confidence angehoben; widerspricht sie, wird die
    alt_classes-Liste entsprechend gewichtet.

    Per opening: semantische Labels innerhalb der Opening-Bbox werden
    aggregiert und `door`/`window` direkt voted.

    `fusion_weight` (0..1) bestimmt, wie stark die Semantik die Heuristik
    ueberschreibt — 0.0 = ignorieren, 1.0 = Semantik gewinnt immer.
    """
    if semantic_labels.shape[0] != points.shape[0]:
        log(
            f"Fusion: Semantic-Count-Mismatch "
            f"({semantic_labels.shape[0]} vs {points.shape[0]}), skipping"
        )
        return labels

    weight = max(0.0, min(1.0, fusion_weight))
    wall_class_map = {
        "wall": "wall_interior",
        "column": "column",
        "beam": "wall_interior",  # keine dedizierte beam-Klasse in der Heuristik
    }

    # Elements sind entweder "wall" (mit plane_idx) oder "opening" (mit element_id).
    # Fuer das Matching auf Openings brauchen wir die Original-Geometrie aus
    # `openings` (width/height/center), darum als dict per id vorhalten.
    openings_by_id = {op["id"]: op for op in openings}
    fused = 0

    for lbl in labels:
        if lbl["element_type"] == "wall":
            plane_idx = lbl.get("plane_idx")
            if plane_idx is None or plane_idx >= len(wall_planes):
                continue
            inlier_mask = wall_planes[plane_idx].get("inlier_mask")
            if inlier_mask is None:
                continue
            votes = semantic_labels[np.asarray(inlier_mask, dtype=bool)]
            if votes.size == 0:
                continue
            winner_idx, winner_frac = _majority(votes)
            winner_name = INDEX_TO_LABEL.get(int(winner_idx), "unknown")
            mapped = wall_class_map.get(winner_name)
            if mapped is None:
                continue
            _blend_into_label(lbl, mapped, winner_frac, weight)
            lbl["semantic_winner"] = winner_name
            lbl["semantic_winner_frac"] = round(float(winner_frac), 3)
            lbl["source"] = "heuristic+semantic"
            fused += 1
        else:  # opening
            op = openings_by_id.get(lbl["element_id"])
            if not op:
                continue
            votes = _votes_in_opening_bbox(points, semantic_labels, op)
            if votes is None or votes.size == 0:
                continue
            winner_idx, winner_frac = _majority(votes)
            winner_name = INDEX_TO_LABEL.get(int(winner_idx), "unknown")
            if winner_name in ("door", "window"):
                _blend_into_label(lbl, winner_name, winner_frac, weight)
                lbl["semantic_winner"] = winner_name
                lbl["semantic_winner_frac"] = round(float(winner_frac), 3)
                lbl["source"] = "heuristic+semantic"
                fused += 1

    if semantic_confidence is not None and semantic_confidence.size > 0:
        log(
            f"Fusion: {fused}/{len(labels)} Elemente per Semantik verfeinert "
            f"(avg_semantic_conf={float(semantic_confidence.mean()):.2f}, weight={weight:.2f})"
        )
    else:
        log(f"Fusion: {fused}/{len(labels)} Elemente per Semantik verfeinert (weight={weight:.2f})")
    return labels


def _majority(votes: np.ndarray) -> tuple[int, float]:
    """Returns (label_id, fraction) des haeufigsten Labels."""
    counts = np.bincount(votes.astype(np.int64))
    winner = int(np.argmax(counts))
    frac = float(counts[winner]) / float(votes.size)
    return winner, frac


def _votes_in_opening_bbox(
    points: np.ndarray,
    semantic_labels: np.ndarray,
    opening: Dict,
) -> Optional[np.ndarray]:
    """Gibt semantische Labels aller Punkte innerhalb der Opening-Bbox zurueck.

    Opening-Dicts variieren je nach Upstream-Step — wir versuchen mehrere
    Keys (center/corner/bbox). Wenn nichts matcht, None.
    """
    # Variante 1: explizite 3D-Bbox (min/max-XYZ)
    if "bbox_min" in opening and "bbox_max" in opening:
        lo = np.asarray(opening["bbox_min"], dtype=float)
        hi = np.asarray(opening["bbox_max"], dtype=float)
        mask = np.all((points >= lo) & (points <= hi), axis=1)
        return semantic_labels[mask] if mask.any() else None

    # Variante 2: center + width/height + Wand-Normal-Richtung
    center = opening.get("center") or opening.get("center_3d")
    width = opening.get("width_m")
    height = opening.get("height_m")
    base = opening.get("base_height_m")
    if center is None or width is None or height is None or base is None:
        return None
    cx, cy, _cz = (float(v) for v in center[:3])
    # konservative isotropische Bbox um das Opening-Zentrum (xy-Halbbreite = width/2,
    # z von base bis base+height, xy-Toleranz = 0.2m)
    half_xy = max(0.1, float(width) / 2.0 + 0.2)
    z_lo = float(base) - 0.1
    z_hi = float(base) + float(height) + 0.1
    mask = (
        (points[:, 0] >= cx - half_xy) & (points[:, 0] <= cx + half_xy)
        & (points[:, 1] >= cy - half_xy) & (points[:, 1] <= cy + half_xy)
        & (points[:, 2] >= z_lo) & (points[:, 2] <= z_hi)
    )
    return semantic_labels[mask] if mask.any() else None


def _blend_into_label(
    label: Dict,
    semantic_class: str,
    semantic_frac: float,
    weight: float,
) -> None:
    """Mischt die Semantik-Stimme in die Heuristik-Confidence.

    Regel:
    - Stimmt semantic_class mit primary_class ueberein: primary_confidence
      wird Richtung 1.0 gehoben (gewichtete Mischung mit semantic_frac).
    - Wenn semantic_class unter alt_classes liegt: schiebe sie nach oben.
    - Wenn semantic_class voellig neu ist und semantic_frac hoch + weight hoch:
      uebernimm semantic_class als primary, alte primary wandert als alt.
    """
    primary = label.get("primary_class")
    primary_conf = float(label.get("primary_confidence", 0.5))
    alts: List[Dict] = list(label.get("alt_classes") or [])

    if semantic_class == primary:
        label["primary_confidence"] = round(
            min(0.99, primary_conf + weight * semantic_frac * (1.0 - primary_conf)),
            3,
        )
        return

    # Heben einer bestehenden Alt-Klasse
    for alt in alts:
        if alt.get("class") == semantic_class:
            alt_conf = float(alt.get("confidence", 0.0))
            new_conf = round(min(0.99, alt_conf + weight * semantic_frac * (1.0 - alt_conf)), 3)
            alt["confidence"] = new_conf
            # Wenn Alt-Klasse jetzt die neue Mehrheit ist, swap
            if new_conf > primary_conf + 0.05:
                old_primary = {"class": primary, "confidence": round(primary_conf, 3)}
                label["primary_class"] = semantic_class
                label["primary_confidence"] = new_conf
                alts = [a for a in alts if a.get("class") != semantic_class]
                alts.append(old_primary)
                label["alt_classes"] = alts
            else:
                label["alt_classes"] = alts
            return

    # Voellig neue Klasse: nur uebernehmen wenn Semantik stark und Gewicht hoch
    if weight >= 0.6 and semantic_frac >= 0.6:
        old_primary = {"class": primary, "confidence": round(primary_conf, 3)}
        label["primary_class"] = semantic_class
        label["primary_confidence"] = round(min(0.95, weight * semantic_frac + 0.1), 3)
        label["alt_classes"] = [old_primary] + alts[:2]
    else:
        # Sonst als alt_class mit moderater Confidence eintragen
        alts.append({
            "class": semantic_class,
            "confidence": round(weight * semantic_frac, 3),
        })
        label["alt_classes"] = alts
