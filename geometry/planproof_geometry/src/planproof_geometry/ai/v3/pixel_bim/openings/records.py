"""OpeningRecord — per-label result carrier.

Produced by the Phase-3 diagnostic; consumed by downstream exporters
(glTF / IFC) and by the semantics_v3.json writer for the viewer panel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class PlacementStatus(str, Enum):
    OK = "ok"
    # Only one flanking wall found (exterior door/window). Parapet/lintel
    # still built against the label-bbox edge on the open side.
    OK_SINGLE_SIDED = "ok_single_sided"
    # Label centre is not inside the gap between W1 and W2 — nearest-walls
    # picked a sibling pair that is not the actual flanks.
    SKIPPED_LABEL_NOT_IN_GAP = "skipped_label_not_in_gap"
    # No wall within 2 m of the label.
    SKIPPED_ISOLATED = "skipped_isolated"
    # W1 and W2 angle > 30 deg (can't form a common axis) and Overlap-based
    # corner resolution also failed.
    SKIPPED_CORNER_AMBIGUOUS = "skipped_corner_ambiguous"
    # Gap length degenerate (< 5 cm or > 10 m).
    SKIPPED_GAP_DEGENERATE = "skipped_gap_degenerate"
    # Building the three meshes failed at the trimesh level.
    FAILED_MESH_BUILD = "failed_mesh_build"


@dataclass
class OpeningRecord:
    """One opening placement result."""
    label_id: str
    label_class: str
    ifc_class: str
    ifc_predefined_type: str
    ifc_operation_type: Optional[str]

    # Host walls (Phase 2 polygon indices)
    wall1_index: int
    wall2_index: Optional[int]  # None for exterior/single-sided

    # Computed geometry
    width_m: float = 0.0
    depth_m: float = 0.0          # wall thickness
    axis_deg: float = 0.0         # 0..180 in plan frame
    sill_m: float = 0.0
    head_m: float = 0.0
    center_xy_m: Tuple[float, float] = (0.0, 0.0)

    # Provenance
    height_source: str = ""
    din_reference: str = ""

    status: PlacementStatus = PlacementStatus.OK
    notes: List[str] = field(default_factory=list)
    din_deviations: List[Dict[str, Any]] = field(default_factory=list)

    def to_semantics_dict(self) -> Dict[str, Any]:
        """Compact form for the viewer panel."""
        return {
            "id": self.label_id,
            "class": self.label_class,
            "ifc_class": self.ifc_class,
            "width_m": round(self.width_m, 3),
            "depth_m": round(self.depth_m, 3),
            "axis_deg": round(self.axis_deg, 1),
            "sill_m": round(self.sill_m, 3),
            "head_m": round(self.head_m, 3),
            "center_m": [round(v, 3) for v in self.center_xy_m],
            "wall1_index": self.wall1_index,
            "wall2_index": self.wall2_index,
            "height_source": self.height_source,
            "din_reference": self.din_reference,
            "status": self.status.value,
            "notes": list(self.notes),
            "din_deviations": list(self.din_deviations),
        }
