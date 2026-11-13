# src/ai/label_schema.py
from __future__ import annotations

import re
from typing import List

# Case-insensitive mapping: wir normalisieren alles zu lower() und entfernen Trennzeichen.
def _norm(lbl: str) -> str:
    s = (lbl or "").strip()
    s = re.sub(r"[\s_\-]+", "", s)  # "surface_finish" -> "surfacefinish"
    return s.lower()


# ============================================================================
# Plan Element Classes – 8-class production schema
# ============================================================================

PLAN_ELEMENT_CLASSES: List[str] = [
    "wall",            # 0 – all walls (interior/exterior handled by pipeline ClassifyWallTypeStep)
    "door",            # 1 – all door types
    "window",          # 2 – all window types
    "stairs",          # 3 – stairs, spiral stairs, elevators
    "room_label",      # 4 – room name text regions
    "furniture",       # 5 – furniture, sanitary, kitchen (excluded from BIM)
    "column",          # 6 – columns, beams, structural supports
    "dimension_line",  # 7 – dimension lines, ticks, extension lines (excluded from BIM)
]

# Legacy class names that map into the 7-class schema
_LEGACY_ALIASES: dict[str, str] = {
    "wall_exterior": "wall", "wall_interior": "wall",
    "door_single": "door", "door_double": "door", "door_sliding": "door",
    "window_large": "window",
    "stairs_spiral": "stairs", "elevator": "stairs",
    "dimension_line": "dimension_line", "dimension_text": "dimension_line",
    "sanitary": "furniture", "kitchen": "furniture",
    "beam": "column",
    "unknown": "wall",
}

PLAN_ELEMENT_SET = set(PLAN_ELEMENT_CLASSES)


def is_plan_element_class(lbl: str) -> bool:
    """Check if label is a valid plan element class (or known alias)."""
    n = _norm(lbl)
    if n in {_norm(c) for c in PLAN_ELEMENT_CLASSES}:
        return True
    return n in {_norm(k) for k in _LEGACY_ALIASES}


def get_plan_element_classes() -> List[str]:
    """Get list of all plan element classes."""
    return PLAN_ELEMENT_CLASSES.copy()


# Synonym mapping: German/English labels -> unified 8-class name
PLAN_ELEMENT_MAP_RAW = {
    # Walls
    "wall": "wall",
    "wand": "wall", "aussenwand": "wall", "innenwand": "wall",
    "wall_exterior": "wall", "wall_interior": "wall",
    "exterior_wall": "wall", "interior_wall": "wall",

    # Doors
    "door": "door",
    "tür": "door", "tuer": "door",
    "door_single": "door", "door_double": "door", "door_sliding": "door",
    "einflügelig": "door", "zweiflügelig": "door",
    "schiebetür": "door", "sliding_door": "door",
    "single door": "door", "double door": "door", "sliding door": "door",

    # Windows
    "window": "window", "fenster": "window",
    "window_large": "window", "panoramafenster": "window",
    "large_window": "window", "bay window": "window", "blind window": "window",

    # Stairs
    "stairs": "stairs", "treppe": "stairs",
    "stairs_spiral": "stairs", "wendeltreppe": "stairs",
    "spiral_stairs": "stairs",
    "elevator": "stairs", "aufzug": "stairs",
    "fahrstuhl": "stairs", "lift": "stairs",

    # Room labels
    "room_label": "room_label",
    "raum": "room_label", "raumname": "room_label",
    "room_name": "room_label", "zimmer": "room_label",
    "Room": "room_label", "room": "room_label",

    # Furniture (includes sanitary + kitchen)
    "furniture": "furniture",
    "möbel": "furniture", "moebel": "furniture", "einrichtung": "furniture",
    "sanitary": "furniture", "kitchen": "furniture",
    "wc": "furniture", "toilette": "furniture",
    "waschbecken": "furniture", "badewanne": "furniture", "dusche": "furniture",
    "sanitär": "furniture", "sanitaer": "furniture", "bathroom": "furniture",
    "küche": "furniture", "kueche": "furniture",
    "herd": "furniture", "spüle": "furniture", "spuele": "furniture",
    "bed": "furniture", "sofa": "furniture", "chair": "furniture",
    "table": "furniture", "sink": "furniture", "toilet": "furniture",
    "bath": "furniture", "bath tub": "furniture",

    # Columns / beams
    "column": "column", "beam": "column",
    "stütze": "column", "stuetze": "column", "pfeiler": "column",
    "träger": "column", "traeger": "column", "balken": "column",

    # Dimension lines -> filtered from BIM output as non-structural
    "dimension_line": "dimension_line", "dimension_text": "dimension_line",
    "maßlinie": "dimension_line", "masslinie": "dimension_line",
    "bemaßung": "dimension_line", "maßzahl": "dimension_line", "masszahl": "dimension_line",
}

PLAN_ELEMENT_MAP = {_norm(k): v for k, v in PLAN_ELEMENT_MAP_RAW.items()}


def to_plan_element_class(lbl: str) -> str:
    """Map a label to a standard 8-class plan element class."""
    k = _norm(lbl)
    # Direct match against canonical classes
    for c in PLAN_ELEMENT_CLASSES:
        if k == _norm(c):
            return c
    # Synonym / legacy lookup
    mapped = PLAN_ELEMENT_MAP.get(k)
    if mapped:
        return mapped
    # Legacy alias (e.g. wall_interior -> wall)
    raw = (lbl or "").strip()
    if raw in _LEGACY_ALIASES:
        return _LEGACY_ALIASES[raw]
    return "wall"  # safe fallback for unknown structural elements

# Coarse labels
COARSE_CLASSES = {
    "mep_duct": "MEP_DUCT",
    "mep_pipe": "MEP_PIPE",
    "mep_equipment": "MEP_EQUIPMENT",
    "electrical": "ELECTRICAL",
    "lighting": "LIGHTING",
    "opening": "OPENING",
    "furniture": "FURNITURE",
    "appliance": "APPLIANCE",
    "surface_finish": "SURFACE_FINISH",
    "other": "OTHER",
}

COARSE_LABEL_SET = set(COARSE_CLASSES.values())

def is_coarse_label(lbl: str) -> bool: 
    return str(lbl or"").strip() in COARSE_LABEL_SET

# Fine + Synonyme + Top-Level Folder
COARSE_MAP_RAW = {
    # --- MEP DUCT ---
    "duct": "MEP_DUCT",
    "ductwork": "MEP_DUCT",
    "airterminal": "MEP_DUCT",  
    "diffuser": "MEP_DUCT",
    "vent": "MEP_DUCT",
    "grille": "MEP_DUCT",
    "grill": "MEP_DUCT",
    
    # --- MEP PIPE ---
    "pipe": "MEP_PIPE",
    "manifold": "MEP_PIPE",
    "valve": "MEP_PIPE",
    "trap": "MEP_PIPE",
    "piping": "MEP_PIPE",

    # --- MEP EQUIPMENT ---
    "ahu": "MEP_EQUIPMENT",
    "pump": "MEP_EQUIPMENT",
    "fancoil": "MEP_EQUIPMENT",
    "radiator": "MEP_EQUIPMENT",
    "towelradiator": "MEP_EQUIPMENT",
    "hvac": "MEP_EQUIPMENT",

    # --- ELECTRICAL ---
    "junctionbox": "ELECTRICAL",
    "distributionbox": "ELECTRICAL",
    "socket": "ELECTRICAL",
    "switch": "ELECTRICAL",
    "cabletray": "ELECTRICAL",
    "conduit": "ELECTRICAL",
    "heatdetector": "ELECTRICAL",
    "smokedetector": "ELECTRICAL",
    "electrical": "ELECTRICAL",
    
    # --- LIGHTING ---
    "panellight": "LIGHTING",
    "tracklight": "LIGHTING",
    "ceilinglight": "LIGHTING",
    "downlight": "LIGHTING",
    "emergencylight": "LIGHTING",
    "exitsign": "LIGHTING",
    "walllight": "LIGHTING",
    "lighting": "LIGHTING",
    "light": "LIGHTING",
    "luminaire": "LIGHTING",

    # --- OPENING ---
    "firedoor": "OPENING",
    "singleleaf": "OPENING",
    "doubleleaf": "OPENING",
    "sliding": "OPENING",
    "casement": "OPENING",
    "fixed": "OPENING",
    "glassdoor": "OPENING",
    "frame": "OPENING",
    "sill": "OPENING",
    "lintel": "OPENING",
    "door": "OPENING",
    "window": "OPENING",
    "opening": "OPENING",
    "shutter": "OPENING",
    "handle": "OPENING",
    "gasket": "OPENING",
    "threshold": "OPENING",

    # --- FURNITURE ---
    "basecabinet": "FURNITURE",
    "tallcabinet": "FURNITURE",
    "wallcabinet": "FURNITURE",
    "cabinet": "FURNITURE",
    "shelf": "FURNITURE",
    "table": "FURNITURE",
    "chair": "FURNITURE",
    "wardrobe": "FURNITURE",
    "bed": "FURNITURE",
    "worktop": "FURNITURE",
    "furniture": "FURNITURE",

    # --- APPLIANCE ---
    "refrigerator": "APPLIANCE",
    "dishwasher": "APPLIANCE",
    "washingmachine": "APPLIANCE",
    "dryer": "APPLIANCE",
    "oven": "APPLIANCE",
    "microwave": "APPLIANCE",
    "cookerhood": "APPLIANCE",
    "sink": "APPLIANCE",
    "appliance": "APPLIANCE",
    "fixture": "APPLIANCE",

    # --- SURFACE FINISH ---
    "carpet": "SURFACE_FINISH",
    "tile": "SURFACE_FINISH",
    "vinyl": "SURFACE_FINISH",
    "laminate": "SURFACE_FINISH",
    "wood": "SURFACE_FINISH",
    "gypsum": "SURFACE_FINISH",
    "plaster": "SURFACE_FINISH",
    "paneling": "SURFACE_FINISH",
    "skirting": "SURFACE_FINISH",
    "acousticpanel": "SURFACE_FINISH",
    "metalgrid": "SURFACE_FINISH",
    "suspended": "SURFACE_FINISH",
    "surfacefinish": "SURFACE_FINISH",
    "finish": "SURFACE_FINISH",
}

COARSE_MAP = { _norm(k): v for k, v in COARSE_MAP_RAW.items() }

INVALID_LABELS = {"test room", "testroom", "room", "scene", "test", "dummy"}

def normalize_label(lbl: str) -> str:
    return (lbl or "").strip()

def is_valid_label(lbl: str) -> bool:
    lbl = normalize_label(lbl)
    if not lbl:
        return False
    low = lbl.lower()
    if low in INVALID_LABELS:
        return False
    # Foldernamen sind meist ohne whitespace/slashes – wenn du später echte Labels zulässt, hier lockern.
    if " " in lbl or "/" in lbl or "\\" in lbl:
        return False
    return True

def to_coarse(lbl: str) -> str:
    k = _norm(lbl)
    return COARSE_MAP.get(k, "OTHER")
