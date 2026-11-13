# src/ai/postprocess.py
from __future__ import annotations

import re
import sqlite3
from typing import Any

import numpy as np


def _safe_row_get(row: Any, key: str, default: str = "") -> str:
    try:
        # sqlite3.Row supports "in row.keys()"
        if hasattr(row, "keys") and key in row.keys():
            v = row[key]
            return "" if v is None else str(v)
    except Exception:
        pass
    # dict-like fallback
    try:
        v = row.get(key, default)  # type: ignore[attr-defined]
        return "" if v is None else str(v)
    except Exception:
        return default


def _mep_tiebreaker_from_features(sample_dict: dict, top1: str, top2: str, length_over_cross_thr: float = 4.0) -> str | None:
    """
    Kopie der Trainingslogik: nur für ambigeous MEP_DUCT vs MEP_EQUIPMENT.
    """
    pair = {top1, top2}
    if pair != {"MEP_DUCT", "MEP_EQUIPMENT"}:
        return None

    loc = float(sample_dict.get("length_over_cross", 0.0))
    if loc > float(length_over_cross_thr):
        return "MEP_DUCT"
    return "MEP_EQUIPMENT"


def predict_batch_with_reject_and_mep_tiebreak(
    proba: np.ndarray,
    classes: np.ndarray,
    samples_dicts: list[dict],
    thr: float,
    min_margin: float,
    per_class_thr: dict[str, float] | None = None,
    per_class_mm: dict[str, float] | None = None,
    mep_margin_gate: float = 0.03,
    mep_enabled: bool = True,
    length_over_cross_thr: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      pred_labels (n,)
      p1 (n,)        top1 confidence
      margin (n,)    p1-p2
      second_labels (n,)
    """
    if proba.ndim != 2:
        raise ValueError(f"proba must be 2D, got shape={proba.shape}")

    n = proba.shape[0]
    pred_idx = np.argmax(proba, axis=1)
    p1 = proba[np.arange(n), pred_idx]

    # indices of 2 best (ascending last two)
    top2_idx = np.argsort(proba, axis=1)[:, -2:]
    second_idx = top2_idx[:, 0]
    p2 = proba[np.arange(n), second_idx]
    margin = p1 - p2

    pred = classes[pred_idx].astype(object)
    second = classes[second_idx].astype(object)

    # per-class thresholds
    if per_class_thr or per_class_mm:
        thr_vec = np.full_like(p1, float(thr), dtype="float64")
        mm_vec = np.full_like(p1, float(min_margin), dtype="float64")

        if per_class_thr:
            for lbl, t in per_class_thr.items():
                thr_vec[pred == lbl] = float(t)
        if per_class_mm:
            for lbl, m in per_class_mm.items():
                mm_vec[pred == lbl] = float(m)

        reject = (p1 < thr_vec) | (margin < mm_vec)
    else:
        reject = (p1 < float(thr)) | (margin < float(min_margin))

    # apply tie-breaker ONLY when ambiguous and not rejected
    if mep_enabled:
        for i in range(n):
            if reject[i]:
                continue
            if float(margin[i]) >= float(mep_margin_gate):
                continue
            forced = _mep_tiebreaker_from_features(samples_dicts[i], str(pred[i]), str(second[i]), length_over_cross_thr=length_over_cross_thr)
            if forced is not None:
                pred[i] = forced

    pred = pred.astype(object)
    pred[reject] = "Unclassified"
    return pred.astype(object), p1.astype("float64"), margin.astype("float64"), second.astype(object)


# ---------
# ELECTRICAL refinement (Rule-based)
# ---------

_ELECTRICAL_RULES: list[tuple[str, list[str]]] = [
    ("HeatDetector",      [r"\bheat\b", r"\bheat[\s_-]?detector\b", r"\btemp(erature)?\b", r"\bhd\b"]),
    ("SmokeDetector",     [r"\bsmoke\b", r"\bsmoke[\s_-]?detector\b", r"\bsd\b"]),

    # bleibt optional, falls du es später labelst:
    ("CO_DETECTOR",       [r"\bco\b", r"\bcarbon\b", r"\bmonoxide\b"]),

    ("MCP_MANUAL_CALLPOINT", [r"\bcall[\s_-]?point\b", r"\bmcp\b", r"\bmanual\b", r"\bhandmelder\b"]),
    ("SIREN_SOUND_BEACON",   [r"\bsiren\b", r"\bsounder\b", r"\bbeacon\b", r"\bstrobe\b"]),

    # wenn Text nur "sensor/detector" sagt, ist "JunctionBox" der konservative Fine-Default, der bei dir existiert:
    ("JunctionBox",      [r"\bsensor\b", r"\bdetector\b"]),

    ("Switch",           [r"\bswitch\b", r"\bschalter\b", r"\btaster\b"]),
    ("Socket",           [r"\bsocket\b", r"\boutlet\b", r"\bsteckdose\b"]),
    ("DistributionBox",  [r"\bpanel\b", r"\bswitchboard\b", r"\bdistribution\b", r"\bverteiler\b"]),
    ("CableTray",        [r"\bcable\s*tray\b", r"\bkabelrinne\b"]),
]

def refine_electrical_subtype(row: sqlite3.Row | dict) -> tuple[str, float]:
    """
    Returns: (subtype, rule_conf in [0..1])
    """
    text = " ".join([
        _safe_row_get(row, "name", ""),
        _safe_row_get(row, "node_path", ""),
        _safe_row_get(row, "class_code", ""),
        _safe_row_get(row, "material_name", ""),
    ]).lower()

    best_label = "ELECTRICAL_UNKNOWN"
    best_hits = 0

    for label, patterns in _ELECTRICAL_RULES:
        hits = 0
        for p in patterns:
            if re.search(p, text):
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_label = label

    # simple confidence mapping
    conf = min(1.0, best_hits / 3.0)  # 0 hits=0.0, 1 hit=0.33, 2 hits=0.66, 3+=1.0
    return best_label, float(conf)

# ---------
# Generic subtype refinement (for ALL coarse classes)
# ---------

import os

# Deine Library Top-Folder (Screenshot) -> Coarse-Klasse
# Passe das Mapping an deine Logik an.
_LIBRARY_TOP_TO_COARSE = {
    "appliances": "APPLIANCE",
    "kitchen": "APPLIANCE",

    "electrical": "ELECTRICAL",
    "sensors_iot": "ELECTRICAL",      # ggf. eigene Klasse später

    "lighting": "LIGHTING",

    "door": "OPENING",
    "openings_accessories": "OPENING",

    "furniture": "FURNITURE",
    "stairs_railings": "FURNITURE",   # ggf. eigene Klasse später
    "structure_simple": "OTHER",      # je nach Modell, sonst eigene Klasse

    # Finish_* -> SURFACE_FINISH
    "finish_ceiling": "SURFACE_FINISH",
    "finish_floor": "SURFACE_FINISH",
    "finish_wall": "SURFACE_FINISH",

    # MEP-ish (du kannst das feiner aufsplitten, wenn du willst)
    "hvac": "MEP_EQUIPMENT",
    "heating": "MEP_EQUIPMENT",
    "plumbing": "MEP_PIPE",
    "sanitary": "MEP_PIPE",
    "firesafety": "ELECTRICAL",       # Detektoren/Sirenen oft electrical; sonst anpassen
}

def _refine_electrical_with_geom(
    row: sqlite3.Row | dict,
    coarse_label: str,
    base_subtype: str | None,
    base_conf: float,
    features: dict | None = None,
) -> tuple[str | None, float]:
    """Kombiniert Text-Regeln + Geo-Heuristiken speziell für ELECTRICAL.

    base_subtype/base_conf kommen aus Ordner- oder COARSE_MAP-Heuristik.
    features enthält die neuen is_* Flags aus features.py (falls beim Inferenzpfad vorhanden).
    """
    # 1) Spezielle Text-Regeln (refine_electrical_subtype)
    text_sub, text_conf = refine_electrical_subtype(row)

    # "ELECTRICAL_UNKNOWN" behandeln wir wie kein Treffer
    if text_sub == "ELECTRICAL_UNKNOWN" and text_conf <= 0.0:
        text_sub = None

    # Bester Kandidat bisher: Ordner/Schema oder Text
    best_sub = base_subtype
    best_conf = float(base_conf)

    if text_sub and text_conf > best_conf:
        best_sub = text_sub
        best_conf = float(text_conf)

    # 2) Falls keine Features verfügbar sind, hier abbrechen
    if not isinstance(features, dict):
        return best_sub, best_conf

    # Geo-basierte Flags (kommen direkt aus dem Feature-Vektor)
    is_small = float(features.get("is_small_obj", 0.0)) > 0.5
    is_large = float(features.get("is_large_obj", 0.0)) > 0.5
    is_wall_mounted = float(features.get("is_wall_mounted", 0.0)) > 0.5
    is_ceiling_mounted = float(features.get("is_ceiling_mounted", 0.0)) > 0.5
    is_floor_object = float(features.get("is_floor_object", 0.0)) > 0.5
    is_flat_panel = float(features.get("is_flat_panel", 0.0)) > 0.5
    is_socket_switch_shape = float(features.get("is_socket_switch_shape", 0.0)) > 0.5

    cz_rel = float(features.get("cz_rel", 0.5))

    # 2a) Steckdosen / Schalter – kleine, wandmontierte Objekte im "Greifbereich"
    in_switch_band = (cz_rel >= 0.05) and (cz_rel <= 0.80)
    if is_socket_switch_shape and is_wall_mounted and in_switch_band:
        raw_text = " ".join([
            _safe_row_get(row, "name", ""),
            _safe_row_get(row, "node_path", ""),
            _safe_row_get(row, "class_code", ""),
            _safe_row_get(row, "material_name", ""),
        ]).lower()

        cand = None
        if re.search(r"\bswitch\b|\bschalter\b|\btaster\b", raw_text):
            cand = "SWITCH"
        elif re.search(r"\bsocket\b|\boutlet\b|\bsteckdose\b", raw_text):
            cand = "SOCKET"
        else:
            # Wenn Text nichts hergibt: Socket als konservativer Default
            cand = "SOCKET"

        # Wenn Text-Regeln schon SWITCH/SOCKET sagen, diese bevorzugt beibehalten
        if text_sub in {"SWITCH", "SOCKET"} and text_conf >= 0.33:
            cand = text_sub

        geo_conf = 0.60
        if cand and geo_conf > best_conf:
            best_sub = cand
            best_conf = geo_conf

    # 2b) Große, wandmontierte Objekte → eher Verteilerschrank/Panel
    if is_large and is_wall_mounted and not is_floor_object:
        geo_conf = 0.55
        cand = "PANEL"
        # Panel aus Ordner-Struktur (base) bleibt mit Konfidenz 1.0 unangetastet
        if cand and geo_conf > best_conf:
            best_sub = cand
            best_conf = geo_conf

    # 2c) Klein + deckenmontiert → Detektoren/Sensoren
    if is_small and is_ceiling_mounted and not is_flat_panel:
        # Wenn Text schon eindeutig Detektor/Sensor sagt, verstärken wir nur die Konfidenz
        if text_sub in {"HEAT_DETECTOR", "SMOKE_DETECTOR", "CO_DETECTOR", "SENSOR_GENERIC", "SIREN_SOUND_BEACON"}:
            geo_conf = max(float(text_conf), 0.70)
            if geo_conf > best_conf:
                best_sub = text_sub
                best_conf = geo_conf
        else:
            # Sonst neutraler Standard: generischer Sensor
            cand = "SENSOR_GENERIC"
            geo_conf = 0.50
            if geo_conf > best_conf:
                best_sub = cand
                best_conf = geo_conf

    return best_sub, best_conf

def _norm_folder(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    return s.strip("_")

def _extract_path_candidate(row: sqlite3.Row | dict) -> str:
    # Wir versuchen alle naheliegenden Key-Namen, ohne zu crashen.
    for k in (
        "source_path", "file_path", "filepath", "path", "uri", "source_uri",
        "asset_path", "import_path", "model_path", "model_file",
    ):
        v = _safe_row_get(row, k, "")
        if v:
            return v
    return ""

def _extract_library_subtype_from_path(path: str) -> tuple[str | None, str | None]:
    """
    Returns: (top_folder, subtype_folder) if parseable, else (None, None)
    """
    if not path:
        return None, None
    p = path.replace("/", "\\")
    parts = [x for x in p.split("\\") if x]
    if len(parts) < 2:
        return None, None

    # Heuristik: suche nach einem bekannten Top-Folder irgendwo im Pfad
    norm_parts = [_norm_folder(x) for x in parts]
    for i, n in enumerate(norm_parts):
        if n in _LIBRARY_TOP_TO_COARSE:
            top = parts[i]
            # subtype = nächster Ordner (falls vorhanden), sonst Dateiname ohne Ext
            if i + 1 < len(parts):
                subtype = parts[i + 1]
                # wenn subtype wie ein file aussieht, nimm stem
                if "." in subtype:
                    subtype = os.path.splitext(subtype)[0]
            else:
                subtype = os.path.splitext(parts[-1])[0]
            return top, subtype

    # fallback: nimm die letzten zwei Ordner, wenn nichts gefunden
    top = parts[-2]
    subtype = os.path.splitext(parts[-1])[0]
    return top, subtype

def _refine_electrical_with_geom(
    row: sqlite3.Row | dict,
    coarse_label: str,
    base_subtype: str | None,
    base_conf: float,
    features: dict | None = None,
) -> tuple[str | None, float]:
    """
    Kombiniert Text-Regeln + Geo-Heuristiken speziell für ELECTRICAL.
    base_subtype/base_conf kommt aus Ordner-/Keyword-Heuristik.
    features enthält is_* Flags aus features.py (falls vorhanden).
    """
    # 1) Text-Regeln
    text_sub, text_conf = refine_electrical_subtype(row)
    if text_sub == "ELECTRICAL_UNKNOWN" and float(text_conf) <= 0.0:
        text_sub = None

    best_sub = base_subtype
    best_conf = float(base_conf)

    if text_sub and float(text_conf) > best_conf:
        best_sub = str(text_sub)
        best_conf = float(text_conf)

    # 2) Wenn keine Features übergeben wurden, fertig
    if not isinstance(features, dict):
        return best_sub, best_conf

    # Geo Flags (robust: default 0.0)
    is_small = float(features.get("is_small_obj", 0.0)) > 0.5
    is_large = float(features.get("is_large_obj", 0.0)) > 0.5
    is_wall_mounted = float(features.get("is_wall_mounted", 0.0)) > 0.5
    is_ceiling_mounted = float(features.get("is_ceiling_mounted", 0.0)) > 0.5
    is_floor_object = float(features.get("is_floor_object", 0.0)) > 0.5
    is_flat_panel = float(features.get("is_flat_panel", 0.0)) > 0.5
    is_socket_switch_shape = float(features.get("is_socket_switch_shape", 0.0)) > 0.5

    cz_rel = float(features.get("cz_rel", 0.5))

    # 2a) Socket/Switch Kandidaten (klein + wandmontiert + "Greifband")
    in_switch_band = (cz_rel >= 0.05) and (cz_rel <= 0.80)
    if is_socket_switch_shape and is_wall_mounted and in_switch_band:
        raw_text = " ".join([
            _safe_row_get(row, "name", ""),
            _safe_row_get(row, "node_path", ""),
            _safe_row_get(row, "class_code", ""),
            _safe_row_get(row, "material_name", ""),
        ]).lower()

        cand = None
        if re.search(r"\bswitch\b|\bschalter\b|\btaster\b", raw_text):
            cand = "Switch"
        elif re.search(r"\bsocket\b|\boutlet\b|\bsteckdose\b", raw_text):
            cand = "Socket"
        else:
            # konservativer Default wenn Text nichts hergibt
            cand = "Socket"

        # Text-Regeln, wenn schon eindeutig, bevorzugen
        if text_sub in {"SWITCH", "SOCKET", "Switch", "Socket"} and float(text_conf) >= 0.33:
            cand = "Switch" if str(text_sub).upper() == "SWITCH" else "Socket"

        geo_conf = 0.60
        if geo_conf > best_conf:
            best_sub = cand
            best_conf = geo_conf

    # 2b) Große, wandmontierte Objekte → DistributionBox/Panel
    if is_large and is_wall_mounted and not is_floor_object:
        geo_conf = 0.55
        cand = "DistributionBox"
        if geo_conf > best_conf:
            best_sub = cand
            best_conf = geo_conf

    # 2c) Klein + deckenmontiert → Detector/Sensor
    if is_small and is_ceiling_mounted and not is_flat_panel:
        # Wenn Text schon Detektor sagt: verstärken
        if str(text_sub) in {"SMOKE_DETECTOR", "HEAT_DETECTOR", "SmokeDetector", "HeatDetector"}:
            # map auf deine Fine-Namen
            cand = "SmokeDetector" if "SMOKE" in str(text_sub).upper() else "HeatDetector"
            geo_conf = max(float(text_conf), 0.70)
            if geo_conf > best_conf:
                best_sub = cand
                best_conf = geo_conf
        else:
            # sonst neutral: JunctionBox ist häufig bei ceiling-mounted
            cand = "JunctionBox"
            geo_conf = 0.50
            if geo_conf > best_conf:
                best_sub = cand
                best_conf = geo_conf

    return best_sub, best_conf


def validate_fine_label(
    fine_label: str,
    row: sqlite3.Row | dict | None = None,
    features: dict | None = None,
) -> tuple[str, float | None]:
    """
    Validator für Fine-Model (Step 3.3):
    - Gibt (validated_label, rule_conf_if_changed) zurück.
    - Nutzt refine_subtype_for_coarse(...) als "sicheres Correction-Rule-System".
    - Überschreibt nur bei hoher Sicherheit (>=0.90).
    """
    from src.ai.label_schema import normalize_label, to_coarse

    fine_label = normalize_label(fine_label)
    coarse = to_coarse(fine_label)

    if row is None:
        return fine_label, None

    # Vorschlag aus Rules (Library/Text/Geo)
    sub, conf = refine_subtype_for_coarse(row, coarse, features)
    if sub and float(conf) >= 0.90:
        sub_norm = normalize_label(str(sub))
        # Nur dann ändern, wenn coarse gleich bleibt
        if to_coarse(sub_norm) == coarse and sub_norm != fine_label:
            return sub_norm, float(conf)

    # leichte Geo-Sanity (nur wenn features da sind)
    if isinstance(features, dict):
        is_wall = float(features.get("is_wall_mounted", 0.0)) > 0.5
        is_ceiling = float(features.get("is_ceiling_mounted", 0.0)) > 0.5
        is_floor = float(features.get("is_floor_object", 0.0)) > 0.5
        is_flat = float(features.get("is_flat_panel", 0.0)) > 0.5
        is_small = float(features.get("is_small_obj", 0.0)) > 0.5

        # Beispiele: sehr konservativ, nur downgrade (kein wildes um-labeln)
        if coarse == "ELECTRICAL" and fine_label in {"Socket", "Switch"}:
            if not (is_wall and is_small):
                # Downgrade: lieber "JunctionBox" statt Socket/Switch
                return "JunctionBox", 0.40

        if coarse == "LIGHTING" and fine_label == "PanelLight":
            if not (is_ceiling and is_flat):
                return "CeilingLight", 0.40

                # Wenn nicht Boden: lasse Fine, aber downgrade auf neutralere Finish-Klasse
        if coarse == "SURFACE_FINISH" and fine_label in {"Carpet", "Tile", "Wood", "Laminate", "Vinyl"}:
            if not is_floor:
                # Wenn nicht Boden: konservatives Downgrade auf naheliegende Finish-Klasse
                if is_ceiling:
                    return "Suspended", 0.35
                if is_wall:
                    return "AcousticPanel", 0.35
                return fine_label, None

    return fine_label, None

def refine_subtype_for_coarse(
    row: sqlite3.Row | dict,
    coarse_label: str,
    features: dict | None = None,
) -> tuple[str | None, float]:
    """
    Universal: erst Ordnerpfad/Library-Struktur, dann Keyword-Fallback.
    Für ELECTRICAL wird zusätzlich refine_electrical_subtype + Geo-Heuristik genutzt.
    Returns: (subtype or None, conf)
    """
    # 0) BEST: class_code aus DB (kommt aus Ordnerstruktur-Labeling)
    cc = _safe_row_get(row, "class_code", "").strip()
    if cc:
        try:
            from src.ai.label_schema import to_coarse
            if to_coarse(cc) == coarse_label:
                return cc, 1.0
        except Exception:
            pass

    coarse_label = str(coarse_label or "").strip()

    # 1) Ordnerstruktur → base_subtype, base_conf
    path = _extract_path_candidate(row)
    top, subtype = _extract_library_subtype_from_path(path)
    base_subtype: str | None = None
    base_conf: float = 0.0
    if subtype:
        mapped = _LIBRARY_TOP_TO_COARSE.get(_norm_folder(top or ""), None)
        if mapped is None or mapped == coarse_label:
            base_subtype = str(subtype)
            base_conf = 1.0

    # 2) Keyword-Fallback aus text (nutzt dein label_schema-Wissen indirekt)
    if base_subtype is None:
        try:
            from src.ai.label_schema import COARSE_MAP_RAW  # keys -> coarse
        except Exception:
            COARSE_MAP_RAW = {}

        text = " ".join([
            _safe_row_get(row, "name", ""),
            _safe_row_get(row, "node_path", ""),
            _safe_row_get(row, "class_code", ""),
            _safe_row_get(row, "material_name", ""),
        ]).lower()

        text_norm = re.sub(r"[\s_\-]+", "", text)

        best = None
        best_len = 0
        for k, coarse in COARSE_MAP_RAW.items():
            if str(coarse) != coarse_label:
                continue
            kn = re.sub(r"[\s_\-]+", "", str(k).lower())
            if kn and kn in text_norm:
                if len(kn) > best_len:
                    best = k
                    best_len = len(kn)

        if best:
            base_subtype = str(best)
            base_conf = 0.66

    # 3) ELECTRICAL: Text + Geo-Heuristik für Feintypen
    if coarse_label == "ELECTRICAL":
        return _refine_electrical_with_geom(row, coarse_label, base_subtype, base_conf, features)

    return base_subtype, base_conf
