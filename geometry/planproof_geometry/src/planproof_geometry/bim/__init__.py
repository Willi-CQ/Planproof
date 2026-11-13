"""BIM-Analyse-Kerne (K5): geometrische Kollisionspruefung (Clash Detection).

AABB-Clash via Sweep-and-Prune, rein typisiert, ohne DB-/Netz-Abhaengigkeit.
Konsumiert vom API-Clash-Endpoint (ProjectModelComponent -> ClashBox).
"""
from planproof_geometry.bim.clash import (
    DEFAULT_TOLERANCE_M,
    Clash,
    ClashBox,
    detect_clashes,
)
from planproof_geometry.bim.federation import (
    FederatedClash,
    FederationResult,
    detect_federation_clashes,
)

__all__ = [
    "DEFAULT_TOLERANCE_M",
    "Clash",
    "ClashBox",
    "FederatedClash",
    "FederationResult",
    "detect_clashes",
    "detect_federation_clashes",
]
