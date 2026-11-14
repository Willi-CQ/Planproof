"""planproof-evidence: Deterministic 2D evidence extraction for architectural drawings.

This package provides a complete pipeline for extracting evidence from
PNG/JPG/PDF drawings, including pixel-level analysis, vector topology,
measurements, and ML comparison.

Public API:
    analyze(input_path, out_dir, ml_predictions=None, ml_maps=None, config=None) -> EvidenceDocument
    get_region(evidence_path, page_index, bbox, out_dir, padding_px=0) -> dict
"""

from .contracts import (
    EvidenceDocument,
    PageFrame,
    Observation,
    PixelEvidence,
    VectorEvidence,
    Comparison,
    Aggregate,
    CalibrationRecord,
    SCHEMA_VERSION,
)
from .pipeline import analyze, get_region

__version__ = "0.1.0"
__all__ = [
    "analyze",
    "get_region",
    "EvidenceDocument",
    "PageFrame",
    "Observation",
    "PixelEvidence",
    "VectorEvidence",
    "Comparison",
    "Aggregate",
    "CalibrationRecord",
    "SCHEMA_VERSION",
    "__version__",
]
