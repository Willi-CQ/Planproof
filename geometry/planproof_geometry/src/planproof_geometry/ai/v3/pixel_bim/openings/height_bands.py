"""DIN-default opening heights + heuristic overrides.

Per label class, derive (sill_m, head_m) for parapet/lintel geometry.
Label metadata may override the default.

Reference:
  - DIN 18101:2014-08  (doors)    -> head (DUK) 2.01 m, sill 0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class OpeningHeights:
    sill_m: float
    head_m: float
    source: str          # "din_default" | "meta_override" | "fallback_unknown_class"
    din_reference: str   # e.g. "DIN 18101:2014"
    height_m: float = 0.0

    def __post_init__(self) -> None:
        self.height_m = max(0.0, self.head_m - self.sill_m)


_DOOR_DEFAULTS = {
    "sill_m": 0.00,
    "head_m": 2.01,
    "din_reference": "DIN 18101:2014",
}

_DOOR_CLASSES = {"door", "door_single", "door_double", "door_sliding"}


def resolve_heights(
    label_class: str,
    *,
    wall_height_m: float = 2.80,
    meta: Optional[Dict] = None,
    label_bbox_center_ratio: Optional[float] = None,
    label_bbox_height_ratio: Optional[float] = None,
) -> OpeningHeights:
    """Return the (sill_m, head_m) band for a door opening.

    Parameters
    ----------
    label_class
        Annotation class.
    wall_height_m
        Overall wall extrusion height from Phase 2. Head is capped at
        ``wall_height_m - 0.01`` so a lintel of at least 1 cm remains.
    meta
        Label metadata. If both ``sill_m`` and ``head_m`` are set, they
        override defaults.
    label_bbox_center_ratio, label_bbox_height_ratio
        Accepted for backward-compatible signature; ignored after window
        detection was removed (2026-05-06 reset).
    """
    del label_bbox_center_ratio, label_bbox_height_ratio
    meta = meta or {}

    # 1. Explicit override.
    if "sill_m" in meta and "head_m" in meta:
        sill = float(meta["sill_m"])
        head = min(float(meta["head_m"]), wall_height_m - 0.01)
        return OpeningHeights(
            sill, head,
            source="meta_override",
            din_reference=str(meta.get("din_reference", "user-supplied")),
        )

    # 2. Doors.
    if label_class in _DOOR_CLASSES:
        sill = _DOOR_DEFAULTS["sill_m"]
        head = min(_DOOR_DEFAULTS["head_m"], wall_height_m - 0.01)
        return OpeningHeights(
            sill, head,
            source="din_default",
            din_reference=_DOOR_DEFAULTS["din_reference"],
        )

    # Unknown class — treat as door-height.
    return OpeningHeights(
        0.00, min(2.01, wall_height_m - 0.01),
        source="fallback_unknown_class",
        din_reference="generic",
    )
