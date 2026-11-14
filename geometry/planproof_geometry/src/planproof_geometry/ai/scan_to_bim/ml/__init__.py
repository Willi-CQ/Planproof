"""ML-Adapter fuer Scan-to-BIM (pretrained semantische Segmentierung).

Diese Ebene kapselt alle externen Modelle (SPT, RandLA-Net, ...) hinter einem
einheitlichen `SemanticSegmenter`-Adapter. Die Pipeline-Steps haengen nur vom
Adapter ab und brauchen die Modell-Dependencies (`torch-geometric`, `open3d-ml`)
nicht zu kennen — Installation erfolgt ueber das Optional-Extra `[scan-ml]`.
"""
from .semantic_segmentation import (
    CANONICAL_LABELS,
    LABEL_TO_INDEX,
    INDEX_TO_LABEL,
    SemanticSegmenter,
    SemanticSegmentationResult,
    load_segmenter,
)

__all__ = [
    "CANONICAL_LABELS",
    "LABEL_TO_INDEX",
    "INDEX_TO_LABEL",
    "SemanticSegmenter",
    "SemanticSegmentationResult",
    "load_segmenter",
]
