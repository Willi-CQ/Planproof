"""Loader for the DIN JSON reference data.

Exposes a `DinStandards` dataclass with typed accessors. All dimensions are
converted from the raw mm values in the JSON files to metres, so downstream
callers (engineering_rules, scene_model) can work in pipeline-native units.

The three JSON files are read once via `importlib.resources` and cached at
module level. Call `load_din_standards()` anywhere — it's effectively free
after the first call.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_JSON_FILES = (
    "din_values_sprint1_critical.json",
    "din_values_sprint2_high.json",
    "din_values_sprint3_medium.json",
)


def _mm_to_m(value: Any) -> Optional[float]:
    """Convert a mm value to metres, tolerating None / non-numeric input."""
    if value is None:
        return None
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return None


def _mm_list_to_m(values: Iterable[Any]) -> List[float]:
    """Convert a list of mm values to metres, skipping None / invalid entries."""
    out: List[float] = []
    for v in values or ():
        m = _mm_to_m(v)
        if m is not None:
            out.append(m)
    return out


@dataclass(frozen=True)
class DinStandards:
    """Flattened, metre-native view of the DIN reference data.

    Only the subset actually consumed by the engineering-rules validator is
    exposed as typed fields. The full raw JSON per norm is kept in `raw` for
    rules that need deeper lookups.
    """

    # --- Walls (DIN 1053 / EC6, DIN 4103) --------------------------------
    wall_thicknesses_m: List[float] = field(default_factory=list)
    wall_thickness_snap_tolerance_m: float = 0.015
    wall_thickness_min_m: float = 0.050
    wall_thickness_max_m: float = 0.640
    wall_loadbearing_threshold_m: float = 0.175  # >= 17.5cm masonry likely load-bearing

    # --- Doors (DIN 18100 / DIN 18101) -----------------------------------
    door_opening_widths_m: List[float] = field(default_factory=list)   # Wandoeffnung Baurichtmaß
    door_opening_heights_m: List[float] = field(default_factory=list)
    door_leaf_widths_m: List[float] = field(default_factory=list)       # Türblatt
    door_leaf_heights_m: List[float] = field(default_factory=list)
    door_width_snap_tolerance_m: float = 0.025
    door_width_min_m: float = 0.585
    door_width_max_single_m: float = 1.260
    door_width_max_double_m: float = 2.500
    door_height_min_m: float = 1.875
    door_height_max_m: float = 2.250

    # --- Stairs (DIN 18065) ----------------------------------------------
    stair_riser_min_m: float = 0.140
    stair_riser_max_m: float = 0.200
    stair_tread_min_m: float = 0.230
    stair_tread_max_m: float = 0.370
    stair_width_min_m: float = 0.800
    stair_passage_height_min_m: float = 2.000

    # --- Windows (Baurichtmass raster from DIN 4172) ---------------------
    window_widths_m: List[float] = field(default_factory=list)
    window_heights_m: List[float] = field(default_factory=list)
    window_width_min_m: float = 0.400
    window_width_max_m: float = 4.000
    window_height_min_m: float = 0.400
    window_height_max_m: float = 3.000

    # --- Accessibility (DIN 18040-1/2) -----------------------------------
    accessible_door_width_min_m: float = 0.800
    accessible_door_width_r_min_m: float = 0.900  # "R" variant (wheelchair)
    accessible_hallway_width_min_m: float = 1.200

    # --- Room labels (DIN EN ISO 4157) -----------------------------------
    # German OCR abbreviation -> full room name
    room_label_map: Dict[str, str] = field(default_factory=dict)
    # Full room name -> DIN 277 usage category (NUF_1, NUF_7, VF, TF)
    room_category_map: Dict[str, str] = field(default_factory=dict)

    # --- Grid (DIN 4172) -------------------------------------------------
    grid_module_m: float = 0.125
    grid_snap_tolerance_m: float = 0.030

    # --- Raw JSON (norm-key -> full dict) --------------------------------
    raw: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------------
    def stair_limits(self, building_class: str = "wohngebaeude") -> Dict[str, Any]:
        """Return DIN 18065 stair limits (metres) for a building class.

        The typed ``stair_*`` fields hold the lenient *Wohngebaeude bis 2 WE*
        range (the P2B pipeline default). DIN 18065 also defines a STRICTER
        *Gebaeude allgemein* range (public/office/commercial): tread min 260 mm
        (vs 230), riser max 190 mm (vs 200), usable width min 1000 mm (vs 800).
        Answering a non-residential stair question with the residential range
        gives a wrong "konform" — so callers that know the building type must be
        able to request the matching range.

        building_class (case-insensitive, several aliases):
          - residential: 'wohngebaeude' | 'wohnen' | 'residential' (default)
          - general:     'allgemein' | 'gebaeude_allgemein' | 'general' |
                         'oeffentlich' | 'public' | 'buero' | 'office' | 'gewerbe'
        """
        bc = (building_class or "wohngebaeude").strip().lower()
        general_aliases = {
            "allgemein", "gebaeude_allgemein", "general", "oeffentlich",
            "public", "buero", "office", "gewerbe", "commercial",
        }
        if bc in general_aliases:
            block = (self.raw.get("din_18065", {}) or {}).get("gebaeude_allgemein", {})
            if isinstance(block, dict) and block:
                rise = block.get("steigung_s_mm", {}) or {}
                tread = block.get("auftritt_a_mm", {}) or {}
                width = block.get("nutzbare_laufbreite_mm", {}) or {}
                passage = block.get("lichte_durchgangshoehe_mm", {}) or {}
                return {
                    "riser_min_m": _mm_to_m(rise.get("min")) or 0.140,
                    "riser_max_m": _mm_to_m(rise.get("max")) or 0.190,
                    "tread_min_m": _mm_to_m(tread.get("min")) or 0.260,
                    "tread_max_m": _mm_to_m(tread.get("max")) or 0.370,
                    "width_min_m": _mm_to_m(width.get("min")) or 1.000,
                    "passage_min_m": _mm_to_m(passage.get("min")) or 2.000,
                    "norm_label": "DIN 18065 (Gebaeude allgemein)",
                    "building_class": "allgemein",
                }
        return {
            "riser_min_m": self.stair_riser_min_m,
            "riser_max_m": self.stair_riser_max_m,
            "tread_min_m": self.stair_tread_min_m,
            "tread_max_m": self.stair_tread_max_m,
            "width_min_m": self.stair_width_min_m,
            "passage_min_m": self.stair_passage_height_min_m,
            "norm_label": "DIN 18065 (Wohngebaeude bis 2 WE)",
            "building_class": "wohngebaeude",
        }

    def snap_wall_thickness(self, thickness_m: float) -> Optional[float]:
        """Snap a measured wall thickness to the nearest standard, within tolerance."""
        return _snap(thickness_m, self.wall_thicknesses_m, self.wall_thickness_snap_tolerance_m)

    def snap_door_opening_width(self, width_m: float) -> Optional[float]:
        """Snap a detected door-opening width to the nearest DIN 18100 Baurichtmaß."""
        return _snap(width_m, self.door_opening_widths_m, self.door_width_snap_tolerance_m)

    def snap_door_leaf_width(self, width_m: float) -> Optional[float]:
        """Snap a detected door-leaf width to the nearest DIN 18101 standard."""
        return _snap(width_m, self.door_leaf_widths_m, self.door_width_snap_tolerance_m)

    def snap_window_width(self, width_m: float) -> Optional[float]:
        """Snap a detected window width to the DIN 4172 raster."""
        return _snap(width_m, self.window_widths_m, self.grid_snap_tolerance_m)

    def classify_wall_loadbearing(self, thickness_m: float) -> bool:
        """Heuristic: >= 17.5cm masonry is probably load-bearing."""
        return thickness_m >= self.wall_loadbearing_threshold_m

    def category_for_room_label(self, label: str) -> Optional[str]:
        """Map an OCR room label ("WZ", "Bad", "Flur") to a DIN 277 category."""
        if not label:
            return None
        # Try abbreviation first (most common in floor plans)
        full = self.room_label_map.get(label.strip().upper())
        if full:
            return self.room_category_map.get(full)
        # Then try direct full-name match (case-insensitive)
        key = label.strip().lower()
        for name, cat in self.room_category_map.items():
            if name.lower() == key or name.lower().startswith(key):
                return cat
        return None


def _snap(value: float, standards: Iterable[float], tolerance: float) -> Optional[float]:
    best: Optional[float] = None
    best_dist = tolerance
    for std in standards:
        dist = abs(value - std)
        if dist < best_dist:
            best = std
            best_dist = dist
    return best


# ---------------------------------------------------------------------------
# JSON reading
# ---------------------------------------------------------------------------

def _read_json_resource(filename: str) -> Dict[str, Any]:
    """Read a JSON file that ships with the package via importlib.resources."""
    resource = files("planproof_geometry.standards.din").joinpath(filename)
    with resource.open("r", encoding="utf-8") as f:
        return json.load(f)


def _merge_raw() -> Dict[str, Any]:
    """Read all three JSON files and merge by top-level norm key.

    Norm keys don't overlap between files, so this is a flat merge with a
    collision warning if one ever does.
    """
    merged: Dict[str, Any] = {}
    for filename in _JSON_FILES:
        try:
            data = _read_json_resource(filename)
        except (FileNotFoundError, OSError) as exc:
            logger.warning("DIN data file missing: %s (%s)", filename, exc)
            continue
        for key, value in data.items():
            if key == "_meta":
                continue
            if key in merged:
                logger.warning("DIN key collision: %s present in multiple files", key)
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def _extract_room_label_map(iso_4157: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Flatten DIN EN ISO 4157 room abbreviations.

    The JSON groups abbreviations by semantic category (wohnen, sanitaer, ...).
    We flatten into {abbr -> full_name} and {full_name -> DIN 277 usage cat}.

    The NUF mapping in the JSON uses underscore-joined abbrs as keys
    ("WZ_SZ_KiZ_AZ_GZ_EZ" -> "NUF_1 ..."). We expand those to per-abbr entries.
    """
    labels: Dict[str, str] = {}
    rooms = iso_4157.get("raumbezeichnungen_deutsch", {}) if isinstance(iso_4157, dict) else {}
    for group_name, group in rooms.items():
        if not isinstance(group, dict):
            continue
        for abbr, full in group.items():
            if abbr.startswith("_") or not isinstance(full, str):
                continue
            # Later groups should not overwrite earlier ones (BA = Bad takes priority
            # over any later collision, which we've already resolved in the JSON).
            labels.setdefault(abbr.upper(), full)

    # Build full_name -> category via the nuf_mapping in p2b_anwendung
    categories: Dict[str, str] = {}
    p2b = iso_4157.get("p2b_anwendung", {}) if isinstance(iso_4157, dict) else {}
    nuf_map = p2b.get("nuf_mapping", {}) if isinstance(p2b, dict) else {}
    for joined_key, category_desc in nuf_map.items():
        # "NUF_1 (Wohnen und Aufenthalt)" -> "NUF_1"
        category = category_desc.split(" ", 1)[0] if isinstance(category_desc, str) else ""
        if not category:
            continue
        for abbr in joined_key.split("_"):
            abbr = abbr.strip().upper()
            full = labels.get(abbr)
            if full:
                categories.setdefault(full, category)
    return labels, categories


def _extract_wall_thicknesses(din_1053: Dict[str, Any]) -> List[float]:
    """Pull the primary snap table from DIN 1053."""
    if not isinstance(din_1053, dict):
        return []
    p2b = din_1053.get("p2b_anwendung", {})
    snap = p2b.get("wanddicken_snap_tabelle_mm") if isinstance(p2b, dict) else None
    if snap:
        return _mm_list_to_m(snap)
    # Fallback: all standard masonry thicknesses
    mw = din_1053.get("mauerwerk_standarddicken_mm", {})
    return _mm_list_to_m(mw.get("alle_dicken", [])) if isinstance(mw, dict) else []


def _extract_door_raster(
    din_18100: Dict[str, Any],
    din_18101: Dict[str, Any],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Pull opening-raster (18100) and door-leaf-raster (18101) in metres."""
    opening_w: List[float] = []
    opening_h: List[float] = []
    if isinstance(din_18100, dict):
        single = din_18100.get("wandoeffnungen_einflg", {})
        if isinstance(single, dict):
            opening_w = _mm_list_to_m(single.get("baurichtmass_breiten_mm", []))
            opening_h = _mm_list_to_m(single.get("baurichtmass_hoehen_mm", []))
        # Fold in double-door widths too
        double = din_18100.get("wandoeffnungen_zweiflg", {})
        if isinstance(double, dict):
            opening_w.extend(_mm_list_to_m(double.get("baurichtmass_breiten_mm", [])))

    leaf_w: List[float] = []
    leaf_h: List[float] = []
    if isinstance(din_18101, dict):
        gefaelzt = din_18101.get("tuerblaetter_gefaelzt_einflg", {})
        if isinstance(gefaelzt, dict):
            leaf_w = _mm_list_to_m(gefaelzt.get("breiten_mm", []))
            leaf_h = _mm_list_to_m(gefaelzt.get("hoehen_mm", []))
        zweiflg = din_18101.get("tuerblaetter_gefaelzt_zweiflg", {})
        if isinstance(zweiflg, dict):
            leaf_w.extend(_mm_list_to_m(zweiflg.get("breiten_mm", [])))

    return (
        sorted(set(opening_w)),
        sorted(set(opening_h)),
        sorted(set(leaf_w)),
        sorted(set(leaf_h)),
    )


def _extract_window_raster(fenster: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    if not isinstance(fenster, dict):
        return [], []
    raster = fenster.get("haeufige_fenstergroessen_baurichtmass_mm", {})
    if not isinstance(raster, dict):
        return [], []
    return _mm_list_to_m(raster.get("breiten", [])), _mm_list_to_m(raster.get("hoehen", []))


def _extract_stair_limits(din_18065: Dict[str, Any]) -> Dict[str, float]:
    """Pull residential-building stair limits (wider tolerances than allgemein)."""
    if not isinstance(din_18065, dict):
        return {}
    # Use the lenient "wohngebaeude_max_2we" range (covers most single-family BIM input)
    wohn = din_18065.get("wohngebaeude_max_2we", {})
    if not isinstance(wohn, dict):
        return {}
    result: Dict[str, float] = {}
    rise = wohn.get("steigung_s_mm", {})
    if isinstance(rise, dict):
        if "min" in rise:
            result["riser_min_m"] = _mm_to_m(rise["min"]) or 0.140
        if "max" in rise:
            result["riser_max_m"] = _mm_to_m(rise["max"]) or 0.200
    tread = wohn.get("auftritt_a_mm", {})
    if isinstance(tread, dict):
        if "min" in tread:
            result["tread_min_m"] = _mm_to_m(tread["min"]) or 0.230
        if "max" in tread:
            result["tread_max_m"] = _mm_to_m(tread["max"]) or 0.370
    width = wohn.get("nutzbare_laufbreite_mm", {})
    if isinstance(width, dict) and "min" in width:
        result["width_min_m"] = _mm_to_m(width["min"]) or 0.800
    passage = wohn.get("lichte_durchgangshoehe_mm", {})
    if isinstance(passage, dict) and "min" in passage:
        result["passage_min_m"] = _mm_to_m(passage["min"]) or 2.000
    return result


def _extract_accessibility(din_18040_2: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(din_18040_2, dict):
        return {}
    result: Dict[str, float] = {}
    tueren = din_18040_2.get("tueren_mm", {})
    if isinstance(tueren, dict):
        if "barrierefrei_lichte_breite_min" in tueren:
            result["door_width_min_m"] = _mm_to_m(tueren["barrierefrei_lichte_breite_min"]) or 0.800
        if "R_lichte_breite_min" in tueren:
            result["door_width_r_min_m"] = _mm_to_m(tueren["R_lichte_breite_min"]) or 0.900
    flure = din_18040_2.get("flure_mm", {})
    if isinstance(flure, dict) and "barrierefrei_breite_min" in flure:
        result["hallway_width_min_m"] = _mm_to_m(flure["barrierefrei_breite_min"]) or 1.200
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_din_standards() -> DinStandards:
    """Load the three DIN JSON files and return a typed, metre-native view.

    Cached: subsequent calls return the same instance. Safe to call from
    pipeline-step constructors.
    """
    raw = _merge_raw()

    walls = _extract_wall_thicknesses(raw.get("din_1053_ec6", {}))
    opening_w, opening_h, leaf_w, leaf_h = _extract_door_raster(
        raw.get("din_18100", {}), raw.get("din_18101", {}),
    )
    win_w, win_h = _extract_window_raster(raw.get("fenster_standardmasse", {}))
    stairs = _extract_stair_limits(raw.get("din_18065", {}))
    acc = _extract_accessibility(raw.get("din_18040_2", {}))
    labels, categories = _extract_room_label_map(raw.get("din_en_iso_4157", {}))

    # DIN 1053 snap tolerance comes from p2b_anwendung
    wall_tol_m = 0.015
    p2b_1053 = raw.get("din_1053_ec6", {}).get("p2b_anwendung", {})
    if isinstance(p2b_1053, dict):
        wall_tol_m = _mm_to_m(p2b_1053.get("tolerance_mm", 15)) or 0.015

    # DIN 18100 snap tolerance
    door_tol_m = 0.025
    p2b_18100 = raw.get("din_18100", {}).get("p2b_anwendung", {})
    if isinstance(p2b_18100, dict):
        door_tol_m = _mm_to_m(p2b_18100.get("tolerance_for_snap_mm", 25)) or 0.025

    # DIN 4172 grid
    grid_module_m = 0.125
    grid_tol_m = 0.030
    din_4172 = raw.get("din_4172", {})
    if isinstance(din_4172, dict):
        grid_module_m = _mm_to_m(din_4172.get("grundmodul_mm", 125)) or 0.125
        p2b_4172 = din_4172.get("p2b_anwendung", {})
        if isinstance(p2b_4172, dict):
            grid_tol_m = _mm_to_m(p2b_4172.get("tolerance_mm", 30)) or 0.030

    standards = DinStandards(
        wall_thicknesses_m=walls or [0.115, 0.175, 0.240, 0.300, 0.365, 0.490],
        wall_thickness_snap_tolerance_m=wall_tol_m,
        door_opening_widths_m=opening_w,
        door_opening_heights_m=opening_h,
        door_leaf_widths_m=leaf_w,
        door_leaf_heights_m=leaf_h,
        door_width_snap_tolerance_m=door_tol_m,
        window_widths_m=win_w,
        window_heights_m=win_h,
        stair_riser_min_m=stairs.get("riser_min_m", 0.140),
        stair_riser_max_m=stairs.get("riser_max_m", 0.200),
        stair_tread_min_m=stairs.get("tread_min_m", 0.230),
        stair_tread_max_m=stairs.get("tread_max_m", 0.370),
        stair_width_min_m=stairs.get("width_min_m", 0.800),
        stair_passage_height_min_m=stairs.get("passage_min_m", 2.000),
        accessible_door_width_min_m=acc.get("door_width_min_m", 0.800),
        accessible_door_width_r_min_m=acc.get("door_width_r_min_m", 0.900),
        accessible_hallway_width_min_m=acc.get("hallway_width_min_m", 1.200),
        room_label_map=labels,
        room_category_map=categories,
        grid_module_m=grid_module_m,
        grid_snap_tolerance_m=grid_tol_m,
        raw=raw,
    )

    logger.info(
        "DIN standards loaded: %d walls, %d door openings, %d door leaves, "
        "%d window sizes, %d room labels",
        len(standards.wall_thicknesses_m),
        len(standards.door_opening_widths_m),
        len(standards.door_leaf_widths_m),
        len(standards.window_widths_m),
        len(standards.room_label_map),
    )
    return standards
