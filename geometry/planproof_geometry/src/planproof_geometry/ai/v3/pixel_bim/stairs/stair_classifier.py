"""Route a stair label class + subtype to its mesh algorithm.

V1 supports only straight runs. ``stairs_spiral``, ``elevator``, and any
other recognised-but-deferred class are routed to ``UNSUPPORTED`` so the
pipeline records a ``SKIPPED_UNSUPPORTED_CLASS_V1`` status without
crashing — a clear "V2 will handle this" signal in the diagnostics.

The annotation can carry an optional ``meta.subtype`` that refines the
class (e.g. ``stairs`` + ``meta.subtype="exterior"`` for an outdoor
flight). The subtype drives the IFC4.3 ``IfcStair.PredefinedType``
mapping; V1 still produces ``STRAIGHT_RUN_STAIR`` geometry for any
``v1_supported`` subtype.

Source of truth for the label-class strings + subtype IDs: the desktop
label editor ``annotationTypes.ts``.
"""
from __future__ import annotations

from typing import Optional

STAIR_ALGORITHM_STRAIGHT = "straight"
STAIR_ALGORITHM_UNSUPPORTED = "unsupported"

# IFC4.3 IfcStair PredefinedType (IfcStairTypeEnum) mapping. V1 covers
# the first row; the rest is documented here so future work has the
# routing table in one place.
IFC_PREDEFINED_TYPE_BY_ALGORITHM = {
    STAIR_ALGORITHM_STRAIGHT: "STRAIGHT_RUN_STAIR",
}

# Subtype routing table (matches STAIR_SUBTYPES in
# clients/desktop-tauri/src/components/plan-bim/annotationTypes.ts).
# Each entry maps the user-selected subtype ID to:
#   * algorithm: STRAIGHT or UNSUPPORTED — drives whether V1 builds a
#     mesh or emits SKIPPED_UNSUPPORTED_CLASS_V1.
#   * ifc_predefined_type: emitted into the StairRecord regardless of
#     algorithm so IFC export sees the correct enum value even when the
#     V1 mesh approximates a complex subtype with a straight run.
#   * v1_supported: True when the V1 mesh is an acceptable
#     approximation of the subtype geometry.
_STAIR_SUBTYPE_ROUTING = {
    "straight": {
        "algorithm": STAIR_ALGORITHM_STRAIGHT,
        "ifc_predefined_type": "STRAIGHT_RUN_STAIR",
        "v1_supported": True,
    },
    "exterior": {
        "algorithm": STAIR_ALGORITHM_STRAIGHT,
        "ifc_predefined_type": "STRAIGHT_RUN_STAIR",
        "v1_supported": True,
    },
    "quarter_turn": {
        "algorithm": STAIR_ALGORITHM_UNSUPPORTED,
        "ifc_predefined_type": "QUARTER_TURN_STAIR",
        "v1_supported": False,
    },
    "half_turn": {
        "algorithm": STAIR_ALGORITHM_UNSUPPORTED,
        "ifc_predefined_type": "HALF_TURN_STAIR",
        "v1_supported": False,
    },
    "two_straight_run": {
        "algorithm": STAIR_ALGORITHM_UNSUPPORTED,
        "ifc_predefined_type": "TWO_STRAIGHT_RUN_STAIR",
        "v1_supported": False,
    },
    "spiral": {
        "algorithm": STAIR_ALGORITHM_UNSUPPORTED,
        "ifc_predefined_type": "SPIRAL_STAIR",
        "v1_supported": False,
    },
    "curved_run": {
        "algorithm": STAIR_ALGORITHM_UNSUPPORTED,
        "ifc_predefined_type": "CURVED_RUN_STAIR",
        "v1_supported": False,
    },
}


def classify_stair(
    label_class: str,
    subtype: Optional[str] = None,
) -> str:
    """Return the algorithm key for a stair label class + optional subtype.

    Args:
        label_class: Annotation class string (matches label-editor IDs).
        subtype: optional ``meta.subtype`` value. When the subtype is
            in the routing table its decision wins over the class-level
            default. Unknown subtypes are ignored (class default used).

    Returns:
        ``STAIR_ALGORITHM_STRAIGHT`` for ``stairs`` + V1-supported
        subtype, ``STAIR_ALGORITHM_UNSUPPORTED`` otherwise.
    """
    cls = (label_class or "").strip()
    sub = (subtype or "").strip()

    if cls == "stairs_spiral" or cls == "elevator":
        return STAIR_ALGORITHM_UNSUPPORTED
    if cls != "stairs":
        return STAIR_ALGORITHM_UNSUPPORTED

    # Class is "stairs". Subtype decides if a routing override exists.
    if sub and sub in _STAIR_SUBTYPE_ROUTING:
        return _STAIR_SUBTYPE_ROUTING[sub]["algorithm"]

    return STAIR_ALGORITHM_STRAIGHT


def resolve_ifc_predefined_type(
    label_class: str,
    subtype: Optional[str] = None,
) -> str:
    """Return the IFC4.3 IfcStairTypeEnum value for a label + subtype.

    Falls back to ``STRAIGHT_RUN_STAIR`` when the class is ``stairs``
    without a known subtype, and to ``NOTDEFINED`` for everything else.
    """
    cls = (label_class or "").strip()
    sub = (subtype or "").strip()
    if cls == "stairs" and sub and sub in _STAIR_SUBTYPE_ROUTING:
        return _STAIR_SUBTYPE_ROUTING[sub]["ifc_predefined_type"]
    if cls == "stairs":
        return "STRAIGHT_RUN_STAIR"
    if cls == "stairs_spiral":
        return "SPIRAL_STAIR"
    return "NOTDEFINED"
