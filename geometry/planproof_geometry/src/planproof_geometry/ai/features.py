from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any

import numpy as np

from src.data.models_db import _connect
from ai.label_schema import is_valid_label, to_coarse, normalize_label

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "component_classifier.joblib"

# ---------
# V2 feature spec (numeric)
# ---------
FEATURE_NAMES: List[str] = [
    # absolute bbox
    "d1", "d2", "d3",
    "bbox_volume",
    "area", "volume",

    # ratios (+ extra geometry signals)
    "d1_d2_ratio", "d1_d3_ratio", "d2_d3_ratio",
    "planarity", "fill_ratio", "area_volume_ratio",

    # helps separating SURFACE_FINISH vs DUCT/PIPE/EQUIPMENT
    "thinness",            # d3 / d1
    "flatness",            # d3 / d2
    "anisotropy_12",       # (d1-d2)/d1
    "anisotropy_23",       # (d2-d3)/d2
    "cross_roundness",     # d3 / d2  (pipe ~ 1, duct < 1)
    "cross_area",          # d2*d3
    "length_over_cross",   # d1 / sqrt(d2*d3)
    
    # orientation / axis cues (helps MEP_DUCT vs MEP_PIPE)
    "axis_is_x", "axis_is_y", "axis_is_z",          # which axis is the longest (dx/dy/dz)
    "xy_elongation",                                # max(dx,dy)/(min(dx,dy)+eps)
    "z_elongation",                                 # dz/(max(dx,dy)+eps)
    "is_verticalish",                               # dz >= 0.9*max(dx,dy)
    "is_horizontalish",                             # max(dx,dy) >= 0.9*dz

    "bbox_surface_area",   # 2*(d1d2 + d1d3 + d2d3)
    "bbox_sa_to_vol",      # bbox_surface_area / bbox_volume

    # logs
    "log_d1", "log_d2", "log_d3",
    "log_bbox_volume", "log_area", "log_volume",

    # relative to scene/model_version bbox
    "cx_rel", "cy_rel", "cz_rel",
    "zmin_rel", "zmax_rel",
    # boundary distances (helps OPENING & surface vs interior objects)
    "dist_min_x_rel", "dist_max_x_rel",
    "dist_min_y_rel", "dist_max_y_rel",
    "dist_min_z_rel", "dist_max_z_rel",
    "touch_wall_any",
    "touch_xmin", "touch_xmax",
    "touch_ymin", "touch_ymax",
    "touch_floor", "touch_ceiling",

    "bbox_vol_rel",
    "footprint_rel",

    # missingness flags (super wichtig in echten Daten)
    "has_area", "has_volume",

    # cluster size (bei single parts = 1)
    "count_parts",
]

def _safe_float(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def _safe_get(row: sqlite3.Row, key: str, default=None):
    try:
        if key in row.keys():
            return row[key]
    except Exception:
        pass
    return default

def compute_scene_stats(rows: list[sqlite3.Row]) -> dict[str, float]:
    eps = 1e-6
    minx = min(_safe_float(_safe_get(r, "bbox_min_x")) for r in rows)
    miny = min(_safe_float(_safe_get(r, "bbox_min_y")) for r in rows)
    minz = min(_safe_float(_safe_get(r, "bbox_min_z")) for r in rows)
    maxx = max(_safe_float(_safe_get(r, "bbox_max_x")) for r in rows)
    maxy = max(_safe_float(_safe_get(r, "bbox_max_y")) for r in rows)
    maxz = max(_safe_float(_safe_get(r, "bbox_max_z")) for r in rows)

    sx = max(maxx - minx, 0.0)
    sy = max(maxy - miny, 0.0)
    sz = max(maxz - minz, 0.0)
    diag = float(np.linalg.norm([sx, sy, sz]) + eps)

    bbox_vol = max(sx * sy * sz, eps)
    area_xy = max(sx * sy, eps)

    return {
        "minx": minx, "miny": miny, "minz": minz,
        "maxx": maxx, "maxy": maxy, "maxz": maxz,
        "sx": sx, "sy": sy, "sz": sz,
        "diag": diag,
        "scene_bbox_vol": bbox_vol,
        "area_xy": area_xy,
        "eps": eps,
    }

def _numeric_features_from_bbox_area_volume(
    bbox_min_x: float, bbox_min_y: float, bbox_min_z: float,
    bbox_max_x: float, bbox_max_y: float, bbox_max_z: float,
    area: float, volume: float,
    stats: dict[str, float],
    count_parts: float,
) -> np.ndarray:
    eps = stats["eps"]

    dx = max(bbox_max_x - bbox_min_x, 0.0)
    dy = max(bbox_max_y - bbox_min_y, 0.0)
    dz = max(bbox_max_z - bbox_min_z, 0.0)

    # axis / orientation cues
    # longest axis identity (one-hot-ish)
    if dx >= dy and dx >= dz:
        axis_is_x, axis_is_y, axis_is_z = 1.0, 0.0, 0.0
    elif dy >= dx and dy >= dz:
        axis_is_x, axis_is_y, axis_is_z = 0.0, 1.0, 0.0
    else:
        axis_is_x, axis_is_y, axis_is_z = 0.0, 0.0, 1.0

    xy_max = max(dx, dy)
    xy_min = min(dx, dy)
    xy_elongation = xy_max / (xy_min + eps)
    z_elongation = dz / (xy_max + eps)

    is_verticalish = 1.0 if dz >= 0.9 * xy_max else 0.0
    is_horizontalish = 1.0 if xy_max >= 0.9 * dz else 0.0


    d1, d2, d3 = sorted([dx, dy, dz], reverse=True)
    bbox_vol = d1 * d2 * d3

    # ratios
    d1_d2 = d1 / (d2 + eps)
    d1_d3 = d1 / (d3 + eps)
    d2_d3 = d2 / (d3 + eps)
    planarity = d3 / (d1 + d2 + d3 + eps)
    fill_ratio = volume / (bbox_vol + eps)
    area_vol = area / (volume + eps)

    # extra geometry signals
    thinness = d3 / (d1 + eps)                    # very small for planar surfaces
    flatness = d3 / (d2 + eps)                    # distinguishes sheet-like vs rod-like
    anisotropy_12 = (d1 - d2) / (d1 + eps)        # elongated vs compact
    anisotropy_23 = (d2 - d3) / (d2 + eps)        # slab vs beam
    cross_roundness = d3 / (d2 + eps)
    cross_area = d2 * d3
    length_over_cross = d1 / (math.sqrt(cross_area) + eps)

    bbox_surface_area = 2.0 * (d1*d2 + d1*d3 + d2*d3)
    bbox_sa_to_vol = bbox_surface_area / (bbox_vol + eps)


    # logs
    log_d1 = math.log(d1 + eps)
    log_d2 = math.log(d2 + eps)
    log_d3 = math.log(d3 + eps)
    log_bbox_vol = math.log(bbox_vol + eps)
    log_area = math.log(area + eps)
    log_volume = math.log(volume + eps)

    # relative to scene
    cx = (bbox_min_x + bbox_max_x) * 0.5
    cy = (bbox_min_y + bbox_max_y) * 0.5
    cz = (bbox_min_z + bbox_max_z) * 0.5

    cx_rel = (cx - stats["minx"]) / (stats["sx"] + eps)
    cy_rel = (cy - stats["miny"]) / (stats["sy"] + eps)
    cz_rel = (cz - stats["minz"]) / (stats["sz"] + eps)

    zmin_rel = (bbox_min_z - stats["minz"]) / (stats["sz"] + eps)
    zmax_rel = (bbox_max_z - stats["minz"]) / (stats["sz"] + eps)

    dist_min_x_rel = (bbox_min_x - stats["minx"]) / (stats["sx"] + eps)
    dist_max_x_rel = (stats["maxx"] - bbox_max_x) / (stats["sx"] + eps)

    dist_min_y_rel = (bbox_min_y - stats["miny"]) / (stats["sy"] + eps)
    dist_max_y_rel = (stats["maxy"] - bbox_max_y) / (stats["sy"] + eps)

    dist_min_z_rel = (bbox_min_z - stats["minz"]) / (stats["sz"] + eps)
    dist_max_z_rel = (stats["maxz"] - bbox_max_z) / (stats["sz"] + eps)

    # "touch" if very close to any boundary (relative threshold)
    touch_eps = 0.02  # 2% of scene span

    touch_xmin = 1.0 if dist_min_x_rel < touch_eps else 0.0
    touch_xmax = 1.0 if dist_max_x_rel < touch_eps else 0.0
    touch_ymin = 1.0 if dist_min_y_rel < touch_eps else 0.0
    touch_ymax = 1.0 if dist_max_y_rel < touch_eps else 0.0
    touch_floor = 1.0 if dist_min_z_rel < touch_eps else 0.0
    touch_ceiling = 1.0 if dist_max_z_rel < touch_eps else 0.0

    touch_wall_any = 1.0 if (touch_xmin or touch_xmax or touch_ymin or touch_ymax) else 0.0


    bbox_vol_rel = bbox_vol / (stats["scene_bbox_vol"] + eps)
    footprint = max(dx * dy, 0.0)
    footprint_rel = footprint / (stats["area_xy"] + eps)

    has_area = 1.0 if area > 0 else 0.0
    has_volume = 1.0 if volume > 0 else 0.0

    feats = [
        d1, d2, d3,
        bbox_vol,
        area, volume,
        d1_d2, d1_d3, d2_d3,
        planarity, fill_ratio, area_vol,

        # extra geometry signals (order must match FEATURE_NAMES)
        thinness,
        flatness,
        anisotropy_12,
        anisotropy_23,
        cross_roundness,
        cross_area,
        length_over_cross,

        axis_is_x, axis_is_y, axis_is_z,
        xy_elongation,
        z_elongation,
        is_verticalish,
        is_horizontalish,

        bbox_surface_area,
        bbox_sa_to_vol,

        log_d1, log_d2, log_d3,
        log_bbox_vol, log_area, log_volume,
        cx_rel, cy_rel, cz_rel,
        zmin_rel, zmax_rel,

        dist_min_x_rel, dist_max_x_rel,
        dist_min_y_rel, dist_max_y_rel,
        dist_min_z_rel, dist_max_z_rel,
        touch_wall_any,
        touch_xmin, touch_xmax,
        touch_ymin, touch_ymax,
        touch_floor, touch_ceiling,

        bbox_vol_rel,
        footprint_rel,
        has_area, has_volume,
        count_parts,
    ]

    return np.array(feats, dtype="float32")

def compute_features_from_row(row: sqlite3.Row, stats: dict[str, float]) -> np.ndarray:
    bbox_min_x = _safe_float(_safe_get(row, "bbox_min_x"))
    bbox_min_y = _safe_float(_safe_get(row, "bbox_min_y"))
    bbox_min_z = _safe_float(_safe_get(row, "bbox_min_z"))
    bbox_max_x = _safe_float(_safe_get(row, "bbox_max_x"))
    bbox_max_y = _safe_float(_safe_get(row, "bbox_max_y"))
    bbox_max_z = _safe_float(_safe_get(row, "bbox_max_z"))

    area = _safe_float(_safe_get(row, "area"))
    volume = _safe_float(_safe_get(row, "volume"))

    return _numeric_features_from_bbox_area_volume(
        bbox_min_x, bbox_min_y, bbox_min_z,
        bbox_max_x, bbox_max_y, bbox_max_z,
        area, volume,
        stats=stats,
        count_parts=1.0,
    )

def compute_features_from_cluster(cluster_rows: list[sqlite3.Row], stats: dict[str, float]) -> np.ndarray:
    minx = min(_safe_float(_safe_get(r, "bbox_min_x")) for r in cluster_rows)
    miny = min(_safe_float(_safe_get(r, "bbox_min_y")) for r in cluster_rows)
    minz = min(_safe_float(_safe_get(r, "bbox_min_z")) for r in cluster_rows)
    maxx = max(_safe_float(_safe_get(r, "bbox_max_x")) for r in cluster_rows)
    maxy = max(_safe_float(_safe_get(r, "bbox_max_y")) for r in cluster_rows)
    maxz = max(_safe_float(_safe_get(r, "bbox_max_z")) for r in cluster_rows)

    area = float(sum(_safe_float(_safe_get(r, "area")) for r in cluster_rows))
    volume = float(sum(_safe_float(_safe_get(r, "volume")) for r in cluster_rows))

    return _numeric_features_from_bbox_area_volume(
        minx, miny, minz, maxx, maxy, maxz,
        area, volume,
        stats=stats,
        count_parts=float(len(cluster_rows)),
    )

# ----------
# Text token features (für name/node_path)
# ----------
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Tokens, die fast nie helfen (BIM/IFC/Generic/IDs)
_TOKEN_STOP = {
    "ifc", "bim", "guid", "uuid", "id", "ids",
    "element", "elements", "object", "objects",
    "type", "types", "generic", "default", "unknown",
    "model", "component", "components",
    "node", "path", "group", "part", "parts",
    "mesh", "geom", "geometry", "shape", "instance", "family",
    "proxy", "building", "project",
}

_HEX_GUID_RE = re.compile(r"^[0-9a-f]{12,}$")

def _is_guid_like(tok: str) -> bool:
    # Sehr lange Strings sind fast immer IDs/Hashes
    if len(tok) >= 24:
        return True
    # Hex-lastig = häufig GUID/Hash
    if _HEX_GUID_RE.match(tok) and len(tok) >= 12:
        return True
    # sehr ziffernlastig = ID
    digits = sum(ch.isdigit() for ch in tok)
    if digits / max(1, len(tok)) > 0.50:
        return True
    return False

def _norm_text(s: str) -> str:
    """
    Normalisiert Text so, dass Umlaute & Sonderzeichen robust werden:
    "Lüftung" -> "luftung", "ß" -> "ss"
    """
    if not s:
        return ""
    s = s.lower().replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # ü->u, ä->a, …
    return s

def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    t = _norm_text(text)
    raw = _TOKEN_RE.findall(t)

    out: list[str] = []
    for tok in raw:
        if tok in _TOKEN_STOP:
            continue
        if tok.isdigit():
            continue
        if _is_guid_like(tok):
            continue

        # Mini-Normalisierung: Plural "lights" -> "light"
        if tok.endswith("s") and len(tok) > 3:
            tok = tok[:-1]

        out.append(tok)

    return out

def _add_geo_heuristic_flags(d: Dict[str, float]) -> None:
    """Derive simple binary flags from bestehenden numerischen Geo-Features.

    Diese Flags sind bewusst heuristisch und konservativ gewählt.
    Sie werden nur als zusätzliche Features in den Vektor gehängt
    (DictVectorizer ignoriert sie bei alten Modellen automatisch).
    """
    # Robust gegen fehlende Keys: alles über get(..., 0.0)
    vol_rel = float(d.get("bbox_vol_rel", 0.0))
    cz_rel = float(d.get("cz_rel", 0.5))
    dist_min_z_rel = float(d.get("dist_min_z_rel", 0.5))
    dist_max_z_rel = float(d.get("dist_max_z_rel", 0.5))

    touch_floor = float(d.get("touch_floor", 0.0)) > 0.5
    touch_ceiling = float(d.get("touch_ceiling", 0.0)) > 0.5
    touch_wall_any = float(d.get("touch_wall_any", 0.0)) > 0.5

    d1 = float(d.get("d1", 0.0))
    d2 = float(d.get("d2", 0.0))
    d3 = float(d.get("d3", 0.0))
    eps = float(d.get("eps", 1e-6))

    flatness = float(d.get("flatness", 0.0))
    thinness = float(d.get("thinness", 0.0))

    # --- Größenkategorien (sehr grob, auf relativen Volumina) ---
    # Die Schwellen sind heuristisch und können später getuned werden.
    is_small = vol_rel > 0.0 and vol_rel < 1e-4
    is_large = vol_rel > 5e-3

    d["is_small_obj"] = 1.0 if is_small else 0.0
    d["is_large_obj"] = 1.0 if is_large else 0.0

    # --- Montageart / Lage ---
    # Bodenobjekt: berührt Boden, Schwerpunkt eher unten
    is_floor_object = touch_floor and cz_rel < 0.30
    d["is_floor_object"] = 1.0 if is_floor_object else 0.0

    # Wandmontage: berührt eine Wand, aber weder Boden noch Decke
    is_wall_mounted = touch_wall_any and not touch_floor and not touch_ceiling
    d["is_wall_mounted"] = 1.0 if is_wall_mounted else 0.0

    # Deckenmontage: berührt Decke, Volumen nicht extrem groß, Schwerpunkt eher oben
    is_ceiling_mounted = touch_ceiling and not touch_floor and vol_rel < 1e-2 and cz_rel > 0.40
    d["is_ceiling_mounted"] = 1.0 if is_ceiling_mounted else 0.0

    # --- Form-Flags ---
    # d1 >= d2 >= d3 (Sortierung kommt aus _numeric_features_from_bbox_area_volume)
    if d1 <= 0.0:
        panel_ratio = 1.0
    else:
        panel_ratio = d3 / (d1 + eps)

    # "Flaches Panel": eine Dimension deutlich kleiner als die anderen, plus flache Gestalt
    is_flat_panel = (panel_ratio < 0.08) and (flatness < 0.25) and (thinness < 0.25)
    d["is_flat_panel"] = 1.0 if is_flat_panel else 0.0

    # Kombiniertes Flag für Steckdosen-/Schalter-Kandidaten
    is_socket_switch_shape = is_small and is_wall_mounted and not is_flat_panel
    d["is_socket_switch_shape"] = 1.0 if is_socket_switch_shape else 0.0

def build_feature_dict_for_row(row: sqlite3.Row, stats: dict[str, float]) -> Dict[str, float]:
    v = compute_features_from_row(row, stats)
    d: Dict[str, float] = {k: float(x) for k, x in zip(FEATURE_NAMES, v)}

    # text signals
    name = str(_safe_get(row, "name", "") or "")
    node_path = str(_safe_get(row, "node_path", "") or "")

    # Optional: falls DB später mehr BIM-Metadaten hat (bricht NICHT, wenn Spalten fehlen)
    extra_fields = []
    for k in ("category", "family", "type_name", "ifc_class", "discipline", "system_type", "material"):
        v = _safe_get(row, k, "")
        if v:
            extra_fields.append(str(v))

    extra_blob = " ".join(extra_fields)

    toks = _tokenize(name) + _tokenize(node_path) + _tokenize(extra_blob)

    for tok in toks:
        d[f"tok={tok}"] = 1.0

    # wichtig: normalisieren (Umlaute etc.)
    text_blob = f"{name} {node_path} {extra_blob}".strip()

    # --- Keyword flags (Needles sind ASCII/normalized) ---
    d["kw_duct"] = _kw_flag(text_blob, ["duct", "kanal", "vent", "diffuser", "grille", "luft", "luftung", "lueft"])
    d["kw_pipe"] = _kw_flag(text_blob, ["pipe", "piping", "tube", "rohr", "valve", "fitting", "flange", "elbow", "tee"])
    # "air" entfernt (zu generisch / noisy). HVAC fokussierter:
    d["kw_hvac"] = _kw_flag(text_blob, ["hvac", "klima", "climate", "ahu", "chiller", "boiler", "heater", "cooler"])
    d["kw_sprinkler"] = _kw_flag(text_blob, ["sprink", "sprinkler"])
    d["kw_drain"] = _kw_flag(text_blob, ["drain", "waste", "abw", "abfluss", "sewer", "sanitary"])
    d["kw_pump_fan"] = _kw_flag(text_blob, ["pump", "pumpe", "fan", "lufter", "luefter", "blower"])

    # Neue, sehr hilfreiche Flags (brechen nichts – sind einfach zusätzliche Features)
    d["kw_electrical"] = _kw_flag(text_blob, ["electrical", "strom", "kabel", "cable", "conduit", "tray", "panel", "verteiler", "junction"])
    d["kw_lighting"] = _kw_flag(text_blob, ["light", "lighting", "luminaire", "leuchte", "lamp"])
    d["kw_opening"] = _kw_flag(text_blob, ["door", "window", "opening", "oeffnung", "offnung", "tor", "fenster", "tuer"])

    _add_geo_heuristic_flags(d)
    return d

def _kw_flag(text: str, needles: list[str]) -> float:
    if not text:
        return 0.0
    t = _norm_text(text)
    # needles bitte ebenfalls already-normalized (wir geben unten nur ascii-Needles rein)
    return 1.0 if any(n in t for n in needles) else 0.0

def build_feature_dict_for_cluster(cluster_rows: list[sqlite3.Row], stats: dict[str, float]) -> Dict[str, float]:
    v = compute_features_from_cluster(cluster_rows, stats)
    d: Dict[str, float] = {k: float(x) for k, x in zip(FEATURE_NAMES, v)}

    # representative text: concat names + node_paths (capped)
    names = []
    paths = []
    for r in cluster_rows[:50]:
        nm = str(_safe_get(r, "name", "") or "")
        npth = str(_safe_get(r, "node_path", "") or "")
        if nm:
            names.append(nm)
        if npth:
            paths.append(npth)

    toks = _tokenize(" ".join(names)) + _tokenize(" ".join(paths))
    for tok in toks:
        d[f"tok={tok}"] = 1.0
    
    text_blob = ((" ".join(names)) + " " + (" ".join(paths))).strip()

    d["kw_duct"] = _kw_flag(text_blob, ["duct", "kanal", "vent", "diffuser", "grille", "luft", "luftung", "lueft"])
    d["kw_pipe"] = _kw_flag(text_blob, ["pipe", "piping", "tube", "rohr", "valve", "fitting", "flange", "elbow", "tee"])
    d["kw_hvac"] = _kw_flag(text_blob, ["hvac", "klima", "climate", "ahu", "chiller", "boiler", "heater", "cooler"])
    d["kw_sprinkler"] = _kw_flag(text_blob, ["sprink", "sprinkler"])
    d["kw_drain"] = _kw_flag(text_blob, ["drain", "waste", "abw", "abfluss", "sewer", "sanitary"])
    d["kw_pump_fan"] = _kw_flag(text_blob, ["pump", "pumpe", "fan", "lufter", "luefter", "blower"])

    d["kw_electrical"] = _kw_flag(text_blob, ["electrical", "strom", "kabel", "cable", "conduit", "tray", "panel", "verteiler", "junction"])
    d["kw_lighting"] = _kw_flag(text_blob, ["light", "lighting", "luminaire", "leuchte", "lamp"])
    d["kw_opening"] = _kw_flag(text_blob, ["door", "window", "opening", "oeffnung", "offnung", "tor", "fenster", "tuer"])

    _add_geo_heuristic_flags(d)
    return d

# --------------------------
# Sampling / DB fetch (V2)
# --------------------------

_HEAVY_PER_MODEL_VERSION_CAP = {
    "SURFACE_FINISH": 250,
    "MEP_DUCT": 300,
    "MEP_PIPE": 300,
    "MEP_EQUIPMENT": 300,
    "LIGHTING": 300,
    "OPENING": 400,
    "FURNITURE": 300,
    "APPLIANCE": 250,
    "ELECTRICAL": 300,
}

def fetch_labeled_feature_dicts(
    mode: str = "coarse",
    min_per_class: int = 50,
    max_per_class_total: int | None = 8000,
    max_per_class_per_model_version: int | None = None,
    min_models_per_class: int = 10,
    include_other: bool = False,
    random_state: int = 42,
) -> Tuple[list[Dict[str, float]], list[str], np.ndarray]:
    """
    Liefert:
      samples: List[dict] -> numeric + tok=... features
      y: labels (coarse|fine abhängig von mode)
      groups: model_id (für GroupShuffleSplit)
    """
    rng = np.random.default_rng(random_state)

    with _connect() as con:
        con.row_factory = sqlite3.Row
        # WICHTIG: kein SELECT-alias in WHERE verwenden -> robust
        rows = con.execute(
            """
            SELECT
                c.*,
                mv.model_id AS model_id,
                mv.id AS model_version_id,
                COALESCE(c.class_code, m.class_code) AS resolved_label
            FROM components c
            JOIN model_versions mv ON mv.id = c.model_version_id
            JOIN models m ON m.id = mv.model_id
            WHERE COALESCE(c.class_code, m.class_code) IS NOT NULL
              AND TRIM(COALESCE(c.class_code, m.class_code)) <> ''
            """
        ).fetchall()

    if not rows:
        raise RuntimeError("Keine gelabelten Komponenten gefunden.")

# 0) label sanity
    rows = [r for r in rows if is_valid_label(str(r["resolved_label"]))]
    if not rows:
        raise RuntimeError("Nach Label-Filter keine Daten übrig (prüfe Labels).")

    # --- DEBUG: Feinverteilung der Labels pro coarse Klasse (vor Sampling) ---
    fine_by_coarse: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        fine = str(r["resolved_label"]).strip()
        coarse = to_coarse(fine)
        if coarse == "OTHER" and not include_other:
            continue
        fine_by_coarse[coarse][fine] += 1

    print("[DEBUG] Feinverteilung der Labels pro coarse Klasse:")
    for coarse in sorted(fine_by_coarse.keys()):
        items = sorted(fine_by_coarse[coarse].items(), key=lambda kv: (-kv[1], kv[0].lower()))
        print(f"[DEBUG]   {coarse}: {items}")
    # --- /DEBUG ---

    # 1) coarse grouping
    by_label: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        coarse = to_coarse(str(r["resolved_label"]))
        if coarse == "OTHER" and not include_other:
            continue
        by_label[coarse].append(r)

    # 2) per model_version cap für heavy-hitter
    stage1: list[sqlite3.Row] = []
    for label, items in by_label.items():
        by_mv: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for r in items:
            by_mv[int(r["model_version_id"])].append(r)

        cap = _HEAVY_PER_MODEL_VERSION_CAP.get(label, max_per_class_per_model_version)

        for mv_id, mv_items in by_mv.items():
            if cap is not None and len(mv_items) > cap:
                idx = rng.choice(len(mv_items), size=cap, replace=False)
                mv_items = [mv_items[i] for i in idx]
            stage1.extend(mv_items)

    # 3) total cap per class
    by_label2: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for r in stage1:
        coarse = to_coarse(str(r["resolved_label"]))
        if coarse == "OTHER" and not include_other:
            continue
        by_label2[coarse].append(r)

    final_rows: list[sqlite3.Row] = []
    for label, items in by_label2.items():
        if len(items) < min_per_class:
            continue
        if max_per_class_total is not None and len(items) > max_per_class_total:
            idx = rng.choice(len(items), size=max_per_class_total, replace=False)
            items = [items[i] for i in idx]
        final_rows.extend(items)

    if not final_rows:
        raise RuntimeError("Nach Sampling/Filtering keine Daten übrig.")

    # 4) min models per class (group split fairness)
    label_to_models: dict[str, set[int]] = defaultdict(set)
    for r in final_rows:
        coarse = to_coarse(str(r["resolved_label"]))
        label_to_models[coarse].add(int(r["model_id"]))

    kept = {lbl for lbl, ms in label_to_models.items() if len(ms) >= min_models_per_class}
    final_rows = [r for r in final_rows if to_coarse(str(r["resolved_label"])) in kept]

    if not final_rows:
        raise RuntimeError("Nach min_models_per_class Filter keine Daten übrig.")

    # 5) compute per-model_version scene stats for relative features
    mv_to_rows: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for r in final_rows:
        mv_to_rows[int(r["model_version_id"])].append(r)

    mv_stats: dict[int, dict[str, float]] = {mv: compute_scene_stats(rs) for mv, rs in mv_to_rows.items()}

    samples: list[Dict[str, float]] = []
    y: list[str] = []
    groups = []

    for r in final_rows:
        mv_id = int(r["model_version_id"])
        stats = mv_stats[mv_id]
        samples.append(build_feature_dict_for_row(r, stats))
        fine_lbl = normalize_label(str(r["resolved_label"]))
        coarse = to_coarse(fine_lbl)
        if coarse == "OTHER" and not include_other:
            continue
        if mode == "coarse":
            y.append(coarse)
        elif mode == "fine":
            y.append(fine_lbl)
        else:
            raise ValueError(f"Unknown mode: {mode}")        
        groups.append(int(r["model_id"]))

    return samples, y, np.array(groups, dtype=int)

def fetch_components_for_model_version(model_version_id: int):
    """
    Wird von der UI genutzt.

    Liefert pro Komponente genau zwei Spalten zurück, damit die UI die Rows
    direkt als (component_id, node_path) entpacken kann.

    Returns:
        list[sqlite3.Row]: Rows mit Keys: "id", "node_path"
    """
    with _connect() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
                id,
                COALESCE(node_path, '') AS node_path
            FROM components
            WHERE model_version_id=?
            ORDER BY id
            """,
            (model_version_id,),
        ).fetchall()
    if rows:
        r0 = rows[0]
        print("[DEBUG] fetch_components_for_model_version first row keys:", list(r0.keys()))
        print("[DEBUG] fetch_components_for_model_version first row len:", len(tuple(r0)))
        
    return rows
