import os
import sqlite3
from typing import Dict, Any

import numpy as np
import joblib

from src.ai.postprocess import (
    predict_batch_with_reject_and_mep_tiebreak,
    refine_subtype_for_coarse,
    validate_fine_label,
)

from src.ai.features import compute_features_from_row  # falls fallback genutzt wird
from src.ai.features import (
    MODEL_PATH,
    compute_scene_stats,
    compute_features_from_cluster,
    build_feature_dict_for_cluster,
    build_feature_dict_for_row,
)
from src.data.models_db import _connect


MODEL_PATH_FINE = MODEL_PATH.with_name(MODEL_PATH.stem + "_fine" + MODEL_PATH.suffix)
_payload_cache_by_path: Dict[str, Dict[str, Any]] = {}


def load_model(prefer_fine: bool | None = None):
    """
    Lädt Model-Payload.

    Step 3 Verhalten:
      - Wenn *_fine existiert: bevorzugen
      - Override: AI_FORCE_COARSE=1 erzwingt coarse
    """
    global _payload_cache_by_path

    force_coarse = os.getenv("AI_FORCE_COARSE", "").strip().lower() in {"1", "true", "yes"}

    if prefer_fine is None:
        prefer_fine = bool(MODEL_PATH_FINE.exists()) and not force_coarse

    path = MODEL_PATH_FINE if prefer_fine else MODEL_PATH

    # Fallbacks, falls eins der Files fehlt
    if not path.exists():
        if MODEL_PATH.exists():
            path = MODEL_PATH
        elif MODEL_PATH_FINE.exists():
            path = MODEL_PATH_FINE
        else:
            raise FileNotFoundError(f"Kein Modell gefunden unter {MODEL_PATH} oder {MODEL_PATH_FINE}")

    key = str(path)
    if key not in _payload_cache_by_path:
        # Audit L7: joblib.load ist pickle-basiert (RCE, WENN die Datei attacker-
        # kontrolliert waere). ``path`` ist hier ausschliesslich operator-kontrolliert
        # (MODEL_PATH-Konstanten / ENV), NIE aus einem Request oder der user-
        # setzbaren pipeline_options-Allowlist -> kein user-erreichbarer Sink.
        _payload_cache_by_path[key] = joblib.load(path)

    payload = _payload_cache_by_path[key]
    model = payload["model"]
    vec = payload.get("vectorizer", None)
    classes = np.array(payload["classes"], dtype=object)
    threshold = float(payload.get("threshold", 0.35))
    min_margin = float(payload.get("min_margin", 0.10))
    mode = payload.get("mode", "coarse")

    # IMPORTANT: payload geben wir mit zurück, damit der Rest keine global-cache hacks braucht.
    return model, vec, classes, threshold, min_margin, mode, payload

def _is_informative_name(name: str | None) -> bool:
    if not name:
        return False
    s = str(name).strip()
    if len(s) < 3:
        return False
    # viele Marketplace/Download-Namen sind oft nur codes → eher "nicht informativ"
    alnum = sum(ch.isalnum() for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    if alnum > 0 and (digits / alnum) > 0.55:
        return False
    # sehr generische Platzhalter
    low = s.lower()
    if low in {"mesh", "model", "component", "object", "export", "default"}:
        return False
    return True


def _select_variant_for_rows(payload: dict, rows: list[sqlite3.Row]):
    """
    Entscheidet pro Modellversion: text-variant wenn Namen brauchbar, sonst geo-variant.
    """
    geo_ok = payload.get("geo_model") is not None and payload.get("geo_vectorizer") is not None
    if not geo_ok:
        return "text"

    names = []
    for r in rows:
        if "name" in r.keys():
            names.append(r["name"])
        elif "node_path" in r.keys():
            names.append(r["node_path"])
        else:
            names.append(None)

    informative = sum(_is_informative_name(n) for n in names)
    ratio = informative / max(1, len(names))

    # wenn < 30% brauchbare Namen → Scan/BIM Mode
    return "geo" if ratio < 0.30 else "text"

def _bbox_tuple(row):
    return (
        float(row["bbox_min_x"] or 0), float(row["bbox_min_y"] or 0), float(row["bbox_min_z"] or 0),
        float(row["bbox_max_x"] or 0), float(row["bbox_max_y"] or 0), float(row["bbox_max_z"] or 0),
    )

def _overlap_1d(a0, a1, b0, b1):
    return min(a1, b1) - max(a0, b0)

def merge_components_by_bbox(rows, scene_diag: float, touch_ratio: float = 0.02, min_overlap_ratio: float = 0.02):
    """
    Strengeres Merging als vorher:
    - "touch" erlaubt Nähe
    - aber zusätzlich muss wenigstens in 1 Achse ein kleiner Overlap-Anteil existieren,
      sonst verklebt man zu viel.
    """
    touch = float(touch_ratio * scene_diag)
    n = len(rows)
    if n == 0:
        return []

    bboxes = [_bbox_tuple(r) for r in rows]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        ax0, ay0, az0, ax1, ay1, az1 = bboxes[i]
        adx, ady, adz = max(ax1-ax0, 1e-6), max(ay1-ay0, 1e-6), max(az1-az0, 1e-6)
        for j in range(i + 1, n):
            bx0, by0, bz0, bx1, by1, bz1 = bboxes[j]
            bdx, bdy, bdz = max(bx1-bx0, 1e-6), max(by1-by0, 1e-6), max(bz1-bz0, 1e-6)

            ox = _overlap_1d(ax0, ax1, bx0, bx1)
            oy = _overlap_1d(ay0, ay1, by0, by1)
            oz = _overlap_1d(az0, az1, bz0, bz1)

            close_x = ox > -touch
            close_y = oy > -touch
            close_z = oz > -touch

            # overlap ratio in mindestens 1 dimension
            overlap_ratio_x = max(ox, 0.0) / min(adx, bdx)
            overlap_ratio_y = max(oy, 0.0) / min(ady, bdy)
            overlap_ratio_z = max(oz, 0.0) / min(adz, bdz)
            ok_overlap = (overlap_ratio_x >= min_overlap_ratio) or (overlap_ratio_y >= min_overlap_ratio) or (overlap_ratio_z >= min_overlap_ratio)

            cond = (close_x and close_y and close_z and ok_overlap)
            if cond:
                union(i, j)

    clusters = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(rows[i])

    return list(clusters.values())

def detect_structural_surfaces(rows):
    eps = 1e-6
    minx = min(float(r["bbox_min_x"] or 0) for r in rows)
    miny = min(float(r["bbox_min_y"] or 0) for r in rows)
    minz = min(float(r["bbox_min_z"] or 0) for r in rows)
    maxx = max(float(r["bbox_max_x"] or 0) for r in rows)
    maxy = max(float(r["bbox_max_y"] or 0) for r in rows)
    maxz = max(float(r["bbox_max_z"] or 0) for r in rows)

    scene_h = maxz - minz
    scene_area_xy = max((maxx-minx) * (maxy-miny), eps)

    structural = {}
    for r in rows:
        cid = int(r["id"])
        bx0, by0, bz0 = float(r["bbox_min_x"] or 0), float(r["bbox_min_y"] or 0), float(r["bbox_min_z"] or 0)
        bx1, by1, bz1 = float(r["bbox_max_x"] or 0), float(r["bbox_max_y"] or 0), float(r["bbox_max_z"] or 0)
        dx, dy, dz = max(bx1-bx0,0), max(by1-by0,0), max(bz1-bz0,0)
        d1, d2, d3 = sorted([dx,dy,dz], reverse=True)

        area = float(r["area"] or 0)
        planarity = d3 / (d1 + d2 + d3 + eps)

        is_flat = planarity < 0.04 and d3 < 0.08 * max(scene_h, eps)
        is_large = area > 0.25 * scene_area_xy

        if is_flat and is_large:
            if bz0 < minz + 0.10 * scene_h:
                structural[cid] = "FLOOR"
            elif bz1 > maxz - 0.10 * scene_h:
                structural[cid] = "CEILING"

        is_wallish = (d1 > 0.60 * max(maxx-minx, maxy-miny) and d2 > 0.30*scene_h and d3 < 0.06*max(scene_h, eps))
        if is_wallish:
            structural[cid] = "WALL"

    return structural

def _format_prediction_output(
    pred_label: str,
    p1: float,
    margin: float,
    features: dict | None,
    representative_row: sqlite3.Row | dict | None,
    mode: str,
) -> tuple[str, str, float | None]:
    """
    Normalisiert Outputs für die UI:

    - coarse-mode: predicted_class = coarse, refined_label = coarse::subtype (falls Rule gefunden).
    - fine-mode: predicted_class = to_coarse(fine), refined_label = coarse::fine,
      plus Validator (überschreibt nur bei hoher Sicherheit).
    """
    label = str(pred_label)

    if label == "Unclassified":
        return "Unclassified", "Unclassified", None

    if label in {"FLOOR", "CEILING", "WALL"}:
        return label, label, None

    if mode == "fine":
        from src.ai.label_schema import to_coarse, is_coarse_label

        # Falls Fine-Modell doch coarse-label ausgibt → nicht "COARSE::COARSE" bauen
        if is_coarse_label(label):
            return label, label, None

        coarse = str(to_coarse(label))
        fine_out = label
        rule_conf: float | None = None

        if representative_row is not None:
            # Validator ist konservativ (ändert nur bei hoher Sicherheit, sonst ggf. kleine Downgrades)
            fine_out, rule_conf = validate_fine_label(fine_out, representative_row, features)

        return coarse, f"{coarse}::{fine_out}", rule_conf

    # coarse-mode (wie vorher)
    refined_label = label
    refined_rule_conf = None
    if representative_row is not None:
        sub, sub_conf = refine_subtype_for_coarse(representative_row, label, features)
        if sub:
            refined_label = f"{label}::{sub}"
            refined_rule_conf = float(sub_conf)

    return label, refined_label, refined_rule_conf


def analyze_model_version(
    model_version_id: int,
    top_k: int = 3,
    reject_threshold: float | None = None,
    min_margin: float | None = None,
    touch_ratio: float = 0.02,
):
    model, vec, classes, thr, mm, mode, payload = load_model()
    mm = mm if min_margin is None else float(min_margin)

    per_class_thr = payload.get("per_class_thresholds", None) or None
    per_class_mm = payload.get("per_class_min_margins", None) or None
    mep_cfg = payload.get("mep_tiebreak", {}) or {}
    mep_enabled = bool(mep_cfg.get("enabled", True))
    mep_margin_gate = float(mep_cfg.get("mep_margin_gate", 0.03))
    length_over_cross_thr = float(mep_cfg.get("length_over_cross_thr", 4.0))

    with _connect() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM components WHERE model_version_id=?",
            (model_version_id,),
        ).fetchall()

    if not rows:
        return []
    
    variant = _select_variant_for_rows(payload, rows)
    if variant == "geo":
        model = payload["geo_model"]
        vec = payload["geo_vectorizer"]
        classes = np.array(payload["geo_classes"], dtype=object)
        thr = float(payload.get("geo_threshold", thr))
        mm = float(payload.get("geo_min_margin", mm))
        per_class_thr = payload.get("geo_per_class_thresholds", None) or None
        per_class_mm = payload.get("geo_per_class_min_margins", None) or None
    else:
        thr = thr if reject_threshold is None else float(reject_threshold)
        mm = mm if min_margin is None else float(min_margin)
        per_class_thr = payload.get("per_class_thresholds", None) or None
        per_class_mm = payload.get("per_class_min_margins", None) or None


    stats = compute_scene_stats(rows)
    scene_diag = stats["diag"]

    structural = detect_structural_surfaces(rows)

    structural_results = []
    non_structural = []

    for r in rows:
        cid = int(r["id"])
        if cid in structural:
            structural_results.append({
                "component_ids": [cid],
                "node_paths": [r["node_path"]] if "node_path" in r.keys() else [],
                "name": r["name"],
                "predicted_class": structural[cid],
                "confidence": 1.0,
                "top_classes": [(structural[cid], 1.0)],
                "margin": 1.0,
                "is_cluster": False,
            })
        else:
            non_structural.append(r)

    clusters = merge_components_by_bbox(non_structural, scene_diag=scene_diag, touch_ratio=touch_ratio)

    cluster_results = []
    for cluster in clusters:
        feat = build_feature_dict_for_cluster(cluster, stats)

        # Proba-Matrix (1 x n_classes)
        if vec is None:
            # fallback: numeric-only
            x = compute_features_from_cluster(cluster, stats)[None, :]
            proba_mat = model.predict_proba(x)
        else:
            X = vec.transform([feat])
            proba_mat = model.predict_proba(X)

        # Einheitliches Reject + MEP-Tie-Breaker wie im Training + analyze_component
        pred_arr, p1_arr, margin_arr, _second_arr = predict_batch_with_reject_and_mep_tiebreak(
            proba=proba_mat,
            classes=classes,
            samples_dicts=[feat],
            thr=float(thr),
            min_margin=float(mm),
            per_class_thr=per_class_thr,
            per_class_mm=per_class_mm,
            mep_margin_gate=mep_margin_gate,
            mep_enabled=mep_enabled,
            length_over_cross_thr=length_over_cross_thr,
        )

        pred_label = str(pred_arr[0])
        p1 = float(p1_arr[0])
        margin = float(margin_arr[0])

        # Für die Top-k-Ansicht verwenden wir weiterhin die rohen Wahrscheinlichkeiten
        proba = proba_mat[0]
        top_idx = np.argsort(proba)[::-1][:top_k]
        top = [(str(classes[i]), float(proba[i])) for i in top_idx]

        cids = [int(r["id"]) for r in cluster]
        node_paths = [r["node_path"] for r in cluster if "node_path" in r.keys()]

        rep_name = None
        for r in cluster:
            nm = r["name"]
            if nm:
                rep_name = nm
                break
        
        display_pred, display_refined, refined_rule_conf = _format_prediction_output(
            pred_label, p1, margin, feat, cluster[0] if cluster else None, mode
        )

        cluster_results.append({
            "component_ids": cids,
            "node_paths": node_paths,
            "name": rep_name,
            "predicted_class": display_pred,
            "refined_label": display_refined,
            "refined_rule_conf": refined_rule_conf,
            "confidence": float(p1),
            "margin": float(margin),
            "top_classes": top,
            "is_cluster": True,
            "cluster_size": len(cluster),
        })

    return structural_results + cluster_results

def analyze_component(component_id: int):
    """
    Lightweight: Analyse genau einer Komponente → Top-1 (+ Reject) + Electrical refinement.
    Perfekt für den Inspector.
    """
    model, vec, classes, thr, mm, mode, payload = load_model()

    mep_cfg = payload.get("mep_tiebreak", {}) or {}
    mep_enabled = bool(mep_cfg.get("enabled", True))
    mep_margin_gate = float(mep_cfg.get("mep_margin_gate", 0.03))
    length_over_cross_thr = float(mep_cfg.get("length_over_cross_thr", 4.0))

    with _connect() as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM components WHERE id=?", (int(component_id),)).fetchone()
        if not row:
            return None
        version_id = int(row["model_version_id"])
        rows = con.execute("SELECT * FROM components WHERE model_version_id=?", (version_id,)).fetchall()

    if not rows:
        return None

    # Variant pro Component (hier reicht 1 Name)
    name = row["name"] if "name" in row.keys() else (row["node_path"] if "node_path" in row.keys() else None)
    use_geo = (
    payload.get("geo_model") is not None
        and payload.get("geo_vectorizer") is not None
        and not _is_informative_name(name)
    )

    if use_geo:
        model = payload["geo_model"]
        vec = payload["geo_vectorizer"]
        classes = np.array(payload["geo_classes"], dtype=object)
        thr = float(payload.get("geo_threshold", thr))
        mm = float(payload.get("geo_min_margin", mm))
        per_class_thr = payload.get("geo_per_class_thresholds", None) or None
        per_class_mm = payload.get("geo_per_class_min_margins", None) or None
    else:
        per_class_thr = payload.get("per_class_thresholds", None) or None
        per_class_mm = payload.get("per_class_min_margins", None) or None

    stats = compute_scene_stats(rows)
    sample = build_feature_dict_for_row(row, stats)

    if vec is None:
        # fallback numeric-only (sollte bei deinem Modell nicht passieren)
        x = compute_features_from_row(row, stats)[None, :]
        proba = model.predict_proba(x)[0]
        proba_mat = proba[None, :]
        samples_dicts = [sample]
    else:
        X = vec.transform([sample])
        proba_mat = model.predict_proba(X)
        samples_dicts = [sample]

    pred, p1, margin, second = predict_batch_with_reject_and_mep_tiebreak(
        proba=proba_mat,
        classes=classes,
        samples_dicts=samples_dicts,
        thr=float(thr),
        min_margin=float(mm),
        per_class_thr=per_class_thr,
        per_class_mm=per_class_mm,
        mep_margin_gate=mep_margin_gate,
        mep_enabled=mep_enabled,
        length_over_cross_thr=length_over_cross_thr,
    )

    label = str(pred[0])
    conf = float(p1[0])
    marg = float(margin[0])

    display_pred, display_refined, refined_rule_conf = _format_prediction_output(label, conf, marg, sample, row, mode)

    out = {
        "component_id": int(row["id"]),
        "node_path": row["node_path"],
        "name": row["name"],
        "predicted_class": display_pred,
        "confidence": conf,
        "margin": marg,
        "refined_label": display_refined,
        "refined_rule_conf": refined_rule_conf,
    }
    if mode == "fine":
        out["raw_predicted_label"] = label
    return out
