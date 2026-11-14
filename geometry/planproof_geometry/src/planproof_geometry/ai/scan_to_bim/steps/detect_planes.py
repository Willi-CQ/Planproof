"""Step 4: Detect planes via iterative RANSAC."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..ml.semantic_segmentation import LABEL_TO_INDEX
from ..ransac import PlaneResult, classify_plane, segment_planes


class DetectPlanesStep:
    name = "Detect planes"

    def run(self, ctx, state):
        config = state.config
        points = state.points

        # Optional: Semantic masking fuer RANSAC. Wenn Step 3.5 Labels gesetzt hat
        # und `semantic_ransac_masking` aktiv ist, laufen zwei getrennte RANSAC-Runs
        # (horizontal-Kandidaten: floor+ceiling Punkte; vertikal-Kandidaten: wall
        # Punkte). Punkte ohne verwertbares Label fliessen in den Allgemein-Lauf.
        semantic_labels = getattr(state, "semantic_labels", None)
        use_masking = bool(config.get("semantic_ransac_masking", True)) and (
            semantic_labels is not None
        )

        if use_masking:
            planes = _detect_planes_with_semantic(
                points=points,
                semantic_labels=semantic_labels,
                distance_threshold=config.get("ransac_distance_threshold", 0.03),
                max_iterations=config.get("ransac_iterations", 1000),
                min_plane_points=config.get("min_plane_points", 500),
                max_planes=config.get("max_planes", 50),
                ctx=ctx,
            )
        else:
            planes = _detect_planes(
                points,
                distance_threshold=config.get("ransac_distance_threshold", 0.03),
                max_iterations=config.get("ransac_iterations", 1000),
                min_plane_points=config.get("min_plane_points", 500),
                max_planes=config.get("max_planes", 50),
                ctx=ctx,
            )

        # Classify planes
        horizontals = []
        verticals = []
        for p in planes:
            kind = classify_plane(p.normal)
            entry = _plane_to_dict(p, kind)
            if kind == "horizontal":
                horizontals.append(entry)
            elif kind == "vertical":
                verticals.append(entry)

        # Pick floor (lowest centroid Z) and ceiling (highest)
        floor_plane = None
        ceiling_plane = None
        if horizontals:
            horizontals.sort(key=lambda h: h["centroid_z"])
            floor_plane = horizontals[0]
            if len(horizontals) > 1:
                ceiling_plane = horizontals[-1]

        state.planes = [_plane_to_dict(p, classify_plane(p.normal)) for p in planes]
        state.floor_plane = floor_plane
        state.ceiling_plane = ceiling_plane
        state.wall_planes = verticals

        room_height = None
        if floor_plane and ceiling_plane:
            room_height = ceiling_plane["centroid_z"] - floor_plane["centroid_z"]

        state.metrics["planes_detected"] = len(planes)
        state.metrics["horizontal_planes"] = len(horizontals)
        state.metrics["vertical_planes"] = len(verticals)
        state.metrics["semantic_ransac_masking_applied"] = bool(use_masking)
        if room_height is not None:
            state.metrics["detected_room_height_m"] = round(room_height, 2)

        ctx.log(
            f"Detected {len(planes)} planes: "
            f"{len(horizontals)} horizontal, {len(verticals)} vertical"
            f"{f', room height ~{room_height:.2f}m' if room_height else ''}"
        )
        return state


def _detect_planes(points, distance_threshold, max_iterations, min_plane_points, max_planes, ctx):
    """Try open3d first, fall back to numpy RANSAC."""
    try:
        return _detect_planes_o3d(
            points, distance_threshold, max_iterations, min_plane_points, max_planes, ctx
        )
    except Exception:
        ctx.log("open3d unavailable, using numpy RANSAC fallback")
        return segment_planes(
            points,
            distance_threshold=distance_threshold,
            max_iterations=max_iterations,
            min_plane_points=min_plane_points,
            max_planes=max_planes,
        )


def _detect_planes_o3d(points, distance_threshold, max_iterations, min_plane_points, max_planes, ctx):
    """Iterative RANSAC via open3d (faster C++ implementation)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    results = []
    remaining_idx = np.arange(points.shape[0])

    for _ in range(max_planes):
        if len(remaining_idx) < min_plane_points:
            break

        subset = pcd.select_by_index(remaining_idx.tolist())
        plane_model, inliers = subset.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=max_iterations,
        )

        if len(inliers) < min_plane_points:
            break

        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        norm_len = np.linalg.norm(normal)
        if norm_len > 0:
            normal = normal / norm_len
            d = d / norm_len

        # Map inliers back to original indices
        original_inliers = remaining_idx[inliers]
        full_mask = np.zeros(points.shape[0], dtype=bool)
        full_mask[original_inliers] = True

        results.append(PlaneResult(
            normal=normal,
            d=d,
            inlier_mask=full_mask,
            inlier_count=int(full_mask.sum()),
            centroid=points[full_mask].mean(axis=0),
            fit_quality=len(inliers) / len(remaining_idx),
        ))

        # Remove inliers
        remaining_idx = np.setdiff1d(remaining_idx, original_inliers)

    results.sort(key=lambda p: p.inlier_count, reverse=True)
    return results


def _plane_to_dict(plane: PlaneResult, kind: str) -> Dict:
    """Convert PlaneResult to serializable dict."""
    return {
        "normal": plane.normal.tolist(),
        "d": float(plane.d),
        "inlier_count": plane.inlier_count,
        "centroid": plane.centroid.tolist(),
        "centroid_z": float(plane.centroid[2]),
        "fit_quality": plane.fit_quality,
        "kind": kind,
        "inlier_mask": plane.inlier_mask,  # kept as numpy for downstream
    }


# --------------------------------------------------------------------------- #
# Semantisch-gemaskte RANSAC-Variante (nutzt Step 3.5 Labels)
# --------------------------------------------------------------------------- #

_HORIZONTAL_CLASSES = ("floor", "ceiling")
_VERTICAL_CLASSES = ("wall", "column", "beam", "door", "window")


def _detect_planes_with_semantic(
    *,
    points: np.ndarray,
    semantic_labels: np.ndarray,
    distance_threshold: float,
    max_iterations: int,
    min_plane_points: int,
    max_planes: int,
    ctx,
):
    """Zwei getrennte RANSAC-Laufe auf semantik-gefilterten Punktmengen.

    Idee: floor/ceiling-Punkte tendieren zu horizontalen Planes, wall/column/
    beam/door/window zu vertikalen. Die Klassen-Masken schraenken den Such-
    raum ein, reduzieren False-Positives und erhoehen die Robustheit auf
    rauschigen Scans. Punkte ohne verwertbares Label fliessen in einen dritten
    Rest-Lauf (falls genug uebrig bleiben), damit Scans ohne vollstaendige
    Klassifikation nichts verlieren.
    """
    if semantic_labels.shape[0] != points.shape[0]:
        ctx.log(
            f"SemRANSAC: Labels-Mismatch "
            f"({semantic_labels.shape[0]} vs {points.shape[0]} Punkte) — "
            "falle auf Standard-RANSAC zurueck"
        )
        return _detect_planes(
            points, distance_threshold, max_iterations, min_plane_points, max_planes, ctx
        )

    h_ids = {LABEL_TO_INDEX[n] for n in _HORIZONTAL_CLASSES if n in LABEL_TO_INDEX}
    v_ids = {LABEL_TO_INDEX[n] for n in _VERTICAL_CLASSES if n in LABEL_TO_INDEX}

    h_mask = np.isin(semantic_labels, list(h_ids))
    v_mask = np.isin(semantic_labels, list(v_ids))
    rest_mask = ~(h_mask | v_mask)

    all_planes = []

    # Verfuegbare Budget-Planes pro Lauf. Hoch angesetzt und durch
    # min_plane_points ohnehin limitiert — die RANSAC-Loop bricht frueh ab.
    budget_per_run = max(1, max_planes // 2)

    def _run(mask: np.ndarray, tag: str) -> list:
        count = int(mask.sum())
        if count < min_plane_points:
            return []
        # _detect_planes erwartet volles Point-Array; wir reduzieren auf Submask und
        # projizieren die inlier_masks zurueck auf die Original-Indizes.
        sub_points = points[mask]
        sub_idx = np.nonzero(mask)[0]
        sub_results = _detect_planes(
            sub_points, distance_threshold, max_iterations, min_plane_points,
            budget_per_run, ctx,
        )
        out: list[PlaneResult] = []
        for r in sub_results:
            full_mask = np.zeros(points.shape[0], dtype=bool)
            full_mask[sub_idx[r.inlier_mask]] = True
            out.append(PlaneResult(
                normal=r.normal,
                d=r.d,
                inlier_mask=full_mask,
                inlier_count=int(full_mask.sum()),
                centroid=points[full_mask].mean(axis=0) if full_mask.any() else r.centroid,
                fit_quality=r.fit_quality,
            ))
        ctx.log(f"SemRANSAC[{tag}]: {len(out)} Planes aus {count} Punkten")
        return out

    all_planes.extend(_run(h_mask, "horizontal"))
    all_planes.extend(_run(v_mask, "vertical"))
    all_planes.extend(_run(rest_mask, "rest"))

    all_planes.sort(key=lambda p: p.inlier_count, reverse=True)
    # Harte Kappe auf max_planes, um Downstream-Kosten begrenzt zu halten
    return all_planes[:max_planes]
