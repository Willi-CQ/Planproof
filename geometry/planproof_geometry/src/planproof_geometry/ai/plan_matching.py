from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math
import sqlite3

from .plan_prior import PlanPrior, PlannedObject, load_prior

from src.data.models_db import _connect
from ai.features import compute_scene_stats, build_feature_dict_for_row


@dataclass
class Match:
    planned_id: str
    component_id: int
    score: float
    coarse_expected: str
    fine_expected: Optional[str]
    coarse_pred: Optional[str]
    refined_pred: Optional[str]
    meta: Dict[str, Any]


@dataclass
class MatchSummary:
    matches: List[Match]
    missing_planned: List[str]
    extra_components: List[int]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _l2(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _dims_sim(d1: Tuple[float, float, float], d2: Tuple[float, float, float]) -> float:
    # 1 - relative L1 difference, clamped
    num = abs(d1[0]-d2[0]) + abs(d1[1]-d2[1]) + abs(d1[2]-d2[2])
    den = (abs(d2[0]) + abs(d2[1]) + abs(d2[2]) + 1e-6)
    return _clamp01(1.0 - (num / den))


def _mount_bonus(planned_mount: Optional[str], feat: Dict[str, float]) -> float:
    if not planned_mount:
        return 0.0
    wall = float(feat.get("is_wall_mounted", 0.0)) > 0.5
    ceiling = float(feat.get("is_ceiling_mounted", 0.0)) > 0.5
    floor = float(feat.get("is_floor_object", 0.0)) > 0.5
    if planned_mount == "wall" and wall:
        return 0.10
    if planned_mount == "ceiling" and ceiling:
        return 0.10
    if planned_mount == "floor" and floor:
        return 0.10
    return -0.05


def score_planned_to_component(po: PlannedObject, feat: Dict[str, float]) -> float:
    """
    Score in [0..1] (grob): Position + Größenähnlichkeit + Mount-Bonus.
    Diese Funktion ist absichtlich simpel (Grundlage).
    """
    c = (float(feat.get("cx_rel", 0.5)), float(feat.get("cy_rel", 0.5)), float(feat.get("cz_rel", 0.5)))
    d = (float(feat.get("d1_rel", feat.get("d1", 0.0))),
         float(feat.get("d2_rel", feat.get("d2", 0.0))),
         float(feat.get("d3_rel", feat.get("d3", 0.0))))

    dist = _l2(po.center_rel, c)
    # Radius: je nach Objektgröße etwas toleranter (min 0.12)
    rad = max(0.12, 0.35 * max(po.dims_rel))
    pos = _clamp01(1.0 - dist / rad)

    size = _dims_sim(d, po.dims_rel)

    bonus = _mount_bonus(po.mount, feat)

    s = 0.65 * pos + 0.35 * size + bonus
    return _clamp01(s)


def match_scan_to_plan(
    model_version_id: int,
    prior_path: str,
    min_score: float = 0.40,
) -> MatchSummary:
    """
    Lädt Komponenten aus DB für model_version_id, berechnet Features,
    und matched gegen PlanPrior.objects.
    """
    prior: PlanPrior = load_prior(prior_path)

    with _connect() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM components WHERE model_version_id=?",
            (int(model_version_id),),
        ).fetchall()
    if not rows:
        return MatchSummary(matches=[], missing_planned=[o.id for o in prior.objects], extra_components=[])

    stats = compute_scene_stats(rows)

    # Features je Component
    feats: Dict[int, Dict[str, float]] = {}
    for r in rows:
        d = build_feature_dict_for_row(r, stats)
        feats[int(r["id"])] = d

    # Greedy assignment (Grundlage; später Hungarian)
    used_components: set[int] = set()
    matches: List[Match] = []
    missing: List[str] = []

    # Für Extras danach:
    all_ids = [int(r["id"]) for r in rows]

    for po in prior.objects:
        best_id: Optional[int] = None
        best_score = -1.0

        for cid in all_ids:
            if cid in used_components:
                continue
            s = score_planned_to_component(po, feats[cid])
            if s > best_score:
                best_score = s
                best_id = cid

        if best_id is None or best_score < min_score:
            missing.append(po.id)
            continue

        used_components.add(best_id)

        # Predicted labels aus DB-Zeilen holen, falls du sie speicherst – sonst None
        # (Wir halten es kompatibel: funktioniert ohne zusätzliche Spalten.)
        coarse_pred = None
        refined_pred = None
        row = next((rr for rr in rows if int(rr["id"]) == best_id), None)
        if row is not None:
            try:
                coarse_pred = row.get("predicted_class")  # falls vorhanden
            except Exception:
                coarse_pred = None
            try:
                refined_pred = row.get("refined_label")   # falls vorhanden
            except Exception:
                refined_pred = None

        matches.append(
            Match(
                planned_id=po.id,
                component_id=best_id,
                score=float(best_score),
                coarse_expected=po.coarse,
                fine_expected=po.fine,
                coarse_pred=coarse_pred,
                refined_pred=refined_pred,
                meta={"mount": po.mount, "center_rel": po.center_rel, "dims_rel": po.dims_rel},
            )
        )

    extras = [cid for cid in all_ids if cid not in used_components]
    return MatchSummary(matches=matches, missing_planned=missing, extra_components=extras)
