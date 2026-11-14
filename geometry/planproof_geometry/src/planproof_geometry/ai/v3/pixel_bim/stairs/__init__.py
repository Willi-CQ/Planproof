"""V3 Pixel-BIM stair placement — Foundation (Tag 1).

Stairs are independent volumes that sit on the floor in front of (or
against) walls. They do **not** mutate wall geometry — the walls-
immutable principle established for openings applies here too.

This package mirrors the structure of ``pixel_bim.openings`` but is
deliberately separate: stairs are not wall openings, they have no
flanking walls, no sill/head, and no gap geometry. Forcing them through
the openings module would muddy that abstraction.

V1 scope (Tag 1-7, sprint 2026-05-22):
    * ``stairs``         — straight run, treads stacked along PCA long
                           axis. IFC: STRAIGHT_RUN_STAIR.
    * ``stairs_spiral``  — deferred to V2 (SPIRAL_STAIR + Hough-Circle).
    * ``elevator``       — deferred to V2 (IfcTransportElement).

Tag 1 ships the foundation only (footprint + records + classifier +
wall_envelope + DIN). Tag 2 brings the pixel-extractor, Tag 3 the
direction-up voter, Tag 5 the mesh-builder, Tag 6 the integration into
``scripts/merge_doors_and_windows_bim.py``.
"""
from .diagnostic_renderer import render_stairs_diagnostic  # noqa: F401
from .footprint import StairFootprint, extract_stair_footprint  # noqa: F401
from .mesh_builder import (  # noqa: F401
    StairMesh,
    build_stair_mesh,
    build_stair_mesh_from_polygon,
)
from .stair_footprint_polygon import (  # noqa: F401
    MIN_HOLE_AREA_M2,
    MIN_TRIMMED_AREA_M2,
    StairFootprintPolygon,
    compute_stair_polygon,
    pca_axes_from_polygon,
)
from .pipeline import (  # noqa: F401
    DEFAULT_CEILING_BUILDUP_M,
    DEFAULT_PIPELINE_FLOOR_TO_FLOOR_M,
    STAIR_LABEL_CLASSES,
    StairPipelineResult,
    process_stairs,
    to_semantics_dict_list,
)
from .records import StairRecord, StairStatus  # noqa: F401
from .stair_arrow_detector import (  # noqa: F401
    ArrowDetection,
    detect_arrow_direction,
)
from .stair_direction import (  # noqa: F401
    DIRECTION_CONFIDENCE_THRESHOLD,
    DIRECTION_VOTE_ABSTAIN,
    DIRECTION_VOTE_MINUS,
    DIRECTION_VOTE_PLUS,
    DirectionVotingResult,
    build_walking_line_xy,
    detect_direction_up,
    signal_density_asymmetry,
    signal_peak_position,
)
from .stair_pixel_extractor import (  # noqa: F401
    BIMODALITY_CUTOFF,
    DARK_THRESHOLD_FLOOR,
    MIN_DARK_PIXELS,
    PERCENTILE_FALLBACK,
    VOTER_A_BASE_CONFIDENCE,
    VOTER_B_BASE_CONFIDENCE,
    VOTER_DISAGREEMENT_PCT,
    PixelExtractionUnavailable,
    StairPixelDetails,
    aggregate_n_steps,
    compute_adaptive_threshold,
    extract_stair_pixels,
)
from .stair_classifier import (  # noqa: F401
    IFC_PREDEFINED_TYPE_BY_ALGORITHM,
    STAIR_ALGORITHM_STRAIGHT,
    STAIR_ALGORITHM_UNSUPPORTED,
    classify_stair,
    resolve_ifc_predefined_type,
)
from .stair_din import (  # noqa: F401
    DEFAULT_TARGET_GOING_M,
    DEFAULT_TARGET_RISE_M,
    GOING_MAX_GENERAL_M,
    GOING_MAX_RESIDENTIAL_M,
    GOING_MIN_GENERAL_M,
    GOING_MIN_RESIDENTIAL_M,
    RISE_MAX_GENERAL_M,
    RISE_MAX_RESIDENTIAL_M,
    RISE_MIN_GENERAL_M,
    RISE_MIN_RESIDENTIAL_M,
    SCHRITTMASS_MAX_M,
    SCHRITTMASS_MIN_M,
    SCHRITTMASS_OPTIMUM_M,
    STEP_TO_STEP_TOLERANCE_M,
    WIDTH_MIN_GENERAL_M,
    WIDTH_MIN_RESIDENTIAL_M,
    derive_step_layout,
    resolve_total_height_m,
    validate_din,
)
from .wall_envelope import (  # noqa: F401
    build_wall_envelope,
    build_wall_envelope_from_polys,
    is_centroid_inside,
)
