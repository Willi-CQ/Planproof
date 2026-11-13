"""StairRecord — per-label result carrier for stair placement.

Parallel to ``openings.records.OpeningRecord`` but tailored to stairs:
no flanking walls, no sill/head, but step layout (n_steps, rise, going)
and a long-axis direction.

V2 (2026-05-22): added IFC4.3-forward-compatible fields. ``walking_line_xy``
is the 2D walking-line directed upward (IfcStair Axis-representation
requirement). ``assembly_metadata`` describes the implicit IfcStair ->
IfcStairFlight assembly the mesh represents, so a later IFC-export sprint
can produce a clean IfcStair with IfcStairFlight body-children without
re-deriving the structure.

Consumed by:
    * ``perfect_bim`` merge pipeline -> embeds the dict in
      ``semantics_v3.json``
    * ``scripts/p2b/diagnose_stairs.py`` -> visualises footprint + axis
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class StairStatus(str, Enum):
    """Lifecycle status of one stair label."""

    OK = "ok"
    # Footprint polygon has < 3 points after scale conversion.
    SKIPPED_DEGENERATE_FOOTPRINT = "skipped_degenerate_footprint"
    # PCA returned NaN (collinear points or numerical instability).
    SKIPPED_PCA_FAILED = "skipped_pca_failed"
    # length_m or width_m below physically plausible threshold (< 30 cm).
    SKIPPED_TOO_SMALL = "skipped_too_small"
    # length_m above 50 m — likely a labelling mistake.
    SKIPPED_TOO_LARGE = "skipped_too_large"
    # Unsupported subclass (stairs_spiral / elevator) in V1 — deferred to V2.
    SKIPPED_UNSUPPORTED_CLASS_V1 = "skipped_unsupported_class_v1"
    # Footprint centroid lies outside the wall envelope.
    SKIPPED_OUTSIDE_WALLS = "skipped_outside_walls"
    # trimesh failed to build any tread box.
    FAILED_MESH_BUILD = "failed_mesh_build"


@dataclass
class StairRecord:
    """One stair placement result."""

    label_id: str
    label_class: str           # "stairs" in V1
    ifc_class: str             # "IfcStair"
    ifc_predefined_type: str   # "STRAIGHT_RUN_STAIR" for V1

    # Footprint geometry (plan XY, metres)
    centroid_xy_m: Tuple[float, float] = (0.0, 0.0)
    long_axis_deg: float = 0.0   # 0..180, ccw from +X
    length_m: float = 0.0        # extent along long axis
    width_m: float = 0.0         # extent along short axis

    # Step layout (None until layout step succeeds)
    n_steps: Optional[int] = None
    rise_m: Optional[float] = None
    going_m: Optional[float] = None
    total_height_m: float = 2.95   # 2.80 lichte Hoehe + 0.15 Deckenaufbau

    # Derived metrics (mirror n_steps/rise/going but always emitted,
    # also when status != OK; used by QualityPage-DIN-Audit later)
    derived_rise_m: Optional[float] = None
    derived_going_m: Optional[float] = None
    derived_schrittmass_m: Optional[float] = None   # 2*rise + going

    # Direction the user walks UP. Default = +long_axis; refined by
    # ``stair_direction.detect_direction_up`` when image data available.
    direction_up_xy: Tuple[float, float] = (1.0, 0.0)

    # Optional user-selected subtype from the label-editor dropdown
    # (e.g. ``straight``, ``exterior``, ``quarter_turn``). Empty when
    # the user did not pick one. Drives the IFC PredefinedType mapping
    # and routing of V2-deferred subtypes to SKIPPED_UNSUPPORTED_CLASS_V1.
    subtype: str = ""

    # IFC4.3 forward-compatibility: the 2D walking-line "directed into
    # the upward direction". Two-point line from low-end-centre to
    # high-end-centre, plan-metres.
    walking_line_xy: List[Tuple[float, float]] = field(default_factory=list)

    # IFC4.3 assembly metadata. For V1 straight runs: 1 flight, 0 landings.
    # Populated so a later IFC-export sprint can emit IfcStair as an
    # assembly with one IfcStairFlight body-child.
    assembly_metadata: Dict[str, Any] = field(
        default_factory=lambda: {"n_flights": 1, "n_landings": 0}
    )

    # Provenance
    height_source: str = "pipeline_default"
    # ^ one of: "annotation_meta" | "config" | "pipeline_default"
    din_reference: str = "DIN 18065:2020-08"

    status: StairStatus = StairStatus.OK
    notes: List[str] = field(default_factory=list)
    din_deviations: List[Dict[str, Any]] = field(default_factory=list)

    def to_semantics_dict(self) -> Dict[str, Any]:
        """Compact form for semantics_v3.json + viewer panel."""
        return {
            "id": self.label_id,
            "class": self.label_class,
            "ifc_class": self.ifc_class,
            "ifc_predefined_type": self.ifc_predefined_type,
            "subtype": self.subtype,
            "centroid_m": [round(v, 3) for v in self.centroid_xy_m],
            "long_axis_deg": round(self.long_axis_deg, 1),
            "length_m": round(self.length_m, 3),
            "width_m": round(self.width_m, 3),
            "n_steps": self.n_steps,
            "rise_m": (
                round(self.rise_m, 3) if self.rise_m is not None else None
            ),
            "going_m": (
                round(self.going_m, 3) if self.going_m is not None else None
            ),
            "total_height_m": round(self.total_height_m, 3),
            "derived_rise_m": (
                round(self.derived_rise_m, 3)
                if self.derived_rise_m is not None else None
            ),
            "derived_going_m": (
                round(self.derived_going_m, 3)
                if self.derived_going_m is not None else None
            ),
            "derived_schrittmass_m": (
                round(self.derived_schrittmass_m, 3)
                if self.derived_schrittmass_m is not None else None
            ),
            "direction_up_xy": [round(v, 3) for v in self.direction_up_xy],
            "walking_line_xy": [
                [round(p[0], 3), round(p[1], 3)] for p in self.walking_line_xy
            ],
            "assembly_metadata": dict(self.assembly_metadata),
            "height_source": self.height_source,
            "din_reference": self.din_reference,
            "status": self.status.value,
            "notes": list(self.notes),
            "din_deviations": list(self.din_deviations),
        }
