"""Scan-vs-BIM: Punktwolke als Maske direkt gegen das Plan-BIM (ohne Scan->BIM).

Oeffentliche API:

* :func:`run_scan_vs_bim` — Job-Einstieg (Dateien rein, Artefakte raus)
* :class:`EngineConfig` — Tuning-Knoepfe der Vier-Klassen-Engine
* Typen: :class:`PlanElement`, :class:`ScanCloud`, :class:`Registration`,
  :class:`ElementVerdict`, :class:`ScanVsBimReport`
"""

from .engine import EngineConfig, run_engine
from .inputs import find_inputs, load_manifest, load_plan_elements
from .pipeline import (
    PREVIEW_FILENAME,
    REPORT_FILENAME,
    REPORT_SCHEMA_VERSION,
    ScanVsBimRunResult,
    run_scan_vs_bim,
)
from .registration import register
from .types import (
    ALL_VERDICTS,
    VERDICT_BUILT,
    VERDICT_DEVIATING,
    VERDICT_MISSING,
    VERDICT_NOT_VISIBLE,
    VERDICT_UNVERIFIABLE,
    VERIFIABLE_CLASSES,
    ElementVerdict,
    PlanElement,
    Registration,
    RoomSummary,
    ScanCloud,
    ScanVsBimReport,
)

__all__ = [
    "run_scan_vs_bim",
    "run_engine",
    "register",
    "EngineConfig",
    "ScanVsBimRunResult",
    "find_inputs",
    "load_manifest",
    "load_plan_elements",
    "PlanElement",
    "ScanCloud",
    "Registration",
    "ElementVerdict",
    "RoomSummary",
    "ScanVsBimReport",
    "REPORT_FILENAME",
    "PREVIEW_FILENAME",
    "REPORT_SCHEMA_VERSION",
    "VERDICT_BUILT",
    "VERDICT_MISSING",
    "VERDICT_DEVIATING",
    "VERDICT_NOT_VISIBLE",
    "VERDICT_UNVERIFIABLE",
    "VERIFIABLE_CLASSES",
    "ALL_VERDICTS",
]
