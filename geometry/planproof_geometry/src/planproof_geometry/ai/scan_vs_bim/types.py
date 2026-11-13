"""Datentypen der Scan-vs-BIM Masken-Engine.

Die Engine vergleicht eine Punktwolke (Ist) direkt gegen die indexierten
Plan-Komponenten (Soll, AABB+Klasse+room_key) — OHNE Scan->BIM-Rekonstruktion
und ohne ML. Alle Typen sind reine Daten; Rechenlogik lebt in
``registration.py`` und ``engine.py``.

Kanonischer Frame: ``(x, h, z)`` mit ``h`` = Hoehe (up). Plan-Versionen mit
``up_axis="Y"`` sind identisch; bei ``up_axis="Z"`` werden y/z getauscht.
Das Registrierungs-Ergebnis bildet ROHE Scan-Koordinaten auf den PLAN-Welt-
Frame ab (nicht auf den kanonischen), damit Preview-Artefakte deckungsgleich
mit dem Modell im Viewer liegen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# Element-Klassen, die die Engine geometrisch verifizieren kann. Alles andere
# (Moebel, TGA, ...) ist per Punktwolken-Maske nicht belastbar pruefbar und
# wird im Report ehrlich als "unverifiable" gefuehrt.
VERIFIABLE_CLASSES = ("wall", "door", "window", "floor", "ceiling", "column")

# Vier-Klassen-Verdikt pro Bauteil (Kern der Engine — die 4. Klasse
# "not_visible" verhindert Phantom-Maengel bei Verdeckung/Nicht-Erfassung).
VERDICT_BUILT = "built"
VERDICT_MISSING = "missing"
VERDICT_DEVIATING = "deviating"
VERDICT_NOT_VISIBLE = "not_visible"
VERDICT_UNVERIFIABLE = "unverifiable"

ALL_VERDICTS = (
    VERDICT_BUILT,
    VERDICT_MISSING,
    VERDICT_DEVIATING,
    VERDICT_NOT_VISIBLE,
    VERDICT_UNVERIFIABLE,
)


@dataclass
class PlanElement:
    """Soll-Bauteil aus ``plan_components.json``.

    Feld-Superset des dict-Adapters der bestehenden Compare-Engine
    (``scan_plan_compare.build_elements``) — zusaetzlich traegt jeder Eintrag
    ``aabb_min``/``aabb_max`` (achsentreue Welt-AABB aus dem GLB-Walk der
    Dispatch-Route), weil die DB-Spalten length/width nach Magnitude sortiert
    sind und die Horizontal-Orientierung dort verloren geht.
    ``aabb_min``/``aabb_max`` sind hier bereits in den kanonischen Frame
    (x, h, z) gebracht.
    """

    id: str
    node_path: str
    name: Optional[str]
    cls: Optional[str]
    aabb_min: Optional[np.ndarray]  # (3,) kanonisch (x, h, z)
    aabb_max: Optional[np.ndarray]  # (3,)
    room_key: Optional[str] = None
    polygon: Optional[List[List[float]]] = None  # [[x, z], ...] nur ROOM
    area_m2: Optional[float] = None

    @property
    def center(self) -> Optional[np.ndarray]:
        if self.aabb_min is None or self.aabb_max is None:
            return None
        return (self.aabb_min + self.aabb_max) / 2.0

    @property
    def size(self) -> Optional[np.ndarray]:
        if self.aabb_min is None or self.aabb_max is None:
            return None
        return np.abs(self.aabb_max - self.aabb_min)


@dataclass
class ScanCloud:
    """Geladene Punktwolke inkl. optionaler Sensor-Posen (E57 Multi-Scan).

    ``origins``/``scan_ids`` sind ``None``, wenn das Format keine Posen traegt
    (PLY/LAS/XYZ) — die Engine faellt dann auf posefreie Evidenz zurueck
    (Interior-Verletzung + Front-Probe statt Ray-Sichtbarkeit).
    """

    points: np.ndarray  # (N, 3) float64, Roh-Koordinaten der Datei
    colors: Optional[np.ndarray] = None  # (N, 3) float32 [0..1]
    origins: Optional[np.ndarray] = None  # (M, 3) Sensor-Positionen je Scan
    scan_ids: Optional[np.ndarray] = None  # (N,) int32 Index in origins

    @property
    def count(self) -> int:
        return int(self.points.shape[0])


@dataclass
class Registration:
    """Ergebnis der Ausrichtung Roh-Scan -> Plan-Welt-Frame.

    ``matrix`` ist die volle homogene 4x4 (inkl. Einheiten-Skalierung,
    Leveling, Yaw, Translation) fuer Punkt-Zeilenvektoren:
    ``p_plan = (M @ [p_raw, 1].T).T``.
    """

    matrix: np.ndarray  # (4, 4) float64
    unit_scale: float
    yaw_rad: float
    rms_m: float
    inlier_fraction: float
    method: str
    warnings: List[str] = field(default_factory=list)

    def apply(self, points: np.ndarray) -> np.ndarray:
        homo = np.hstack([points, np.ones((points.shape[0], 1))])
        return (homo @ self.matrix.T)[:, :3]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matrix": [[round(float(v), 8) for v in row] for row in self.matrix],
            "unit_scale": self.unit_scale,
            "yaw_deg": round(float(np.degrees(self.yaw_rad)), 3),
            "rms_m": round(self.rms_m, 4),
            "inlier_fraction": round(self.inlier_fraction, 4),
            "method": self.method,
            "warnings": list(self.warnings),
        }


@dataclass
class ElementVerdict:
    """Vier-Klassen-Verdikt + Messwerte fuer EIN Soll-Bauteil."""

    component_id: str
    node_path: str
    cls: str
    room_key: Optional[str]
    verdict: str
    # Zell-Statistik der erwarteten Oberflaeche (Anteile, Summe <= 1.0):
    hit_ratio: float = 0.0
    through_ratio: float = 0.0  # Evidenz GEGEN Existenz (interior/ray-through)
    blocked_ratio: float = 0.0  # verdeckt (Punkte vor der Flaeche)
    empty_ratio: float = 0.0  # keinerlei Evidenz
    opening_ratio: float = 0.0  # Zellen durch Soll-Oeffnungen maskiert
    # Messwerte (nur wenn belastbar):
    deviation_mm: Optional[float] = None  # medianer Flaechen-Offset
    flatness_mm: Optional[float] = None  # RMS-Residuum Ebenen-Fit (Wand)
    plumb_mm_per_m: Optional[float] = None  # Lot-Abweichung (Wand)
    severity: Optional[str] = None  # low|medium|high (advisory)
    rule_id: Optional[str] = None  # svb:{cls}:{missing|deviation|...}
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "node_path": self.node_path,
            "class": self.cls,
            "room_key": self.room_key,
            "verdict": self.verdict,
            "hit_ratio": round(self.hit_ratio, 4),
            "through_ratio": round(self.through_ratio, 4),
            "blocked_ratio": round(self.blocked_ratio, 4),
            "empty_ratio": round(self.empty_ratio, 4),
            "opening_ratio": round(self.opening_ratio, 4),
            "deviation_mm": (round(self.deviation_mm, 1) if self.deviation_mm is not None else None),
            "flatness_mm": (round(self.flatness_mm, 1) if self.flatness_mm is not None else None),
            "plumb_mm_per_m": (round(self.plumb_mm_per_m, 1) if self.plumb_mm_per_m is not None else None),
            "severity": self.severity,
            "rule_id": self.rule_id,
            "detail": self.detail,
        }


@dataclass
class RoomSummary:
    """Aggregat je Raum (room_key aus dem Indexer)."""

    room_key: str
    label: Optional[str]
    counts: Dict[str, int]
    verified_pct: float  # built+deviating an allen verifizierbaren Elementen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_key": self.room_key,
            "label": self.label,
            "counts": dict(self.counts),
            "verified_pct": round(self.verified_pct, 1),
        }


@dataclass
class ScanVsBimReport:
    """Gesamtreport — wird als ``scan_vs_bim_report.json`` ins Bundle geschrieben."""

    schema_version: int
    manifest: Dict[str, Any]
    registration: Registration
    elements: List[ElementVerdict]
    rooms: List[RoomSummary]
    summary: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest": self.manifest,
            "registration": self.registration.to_dict(),
            "elements": [e.to_dict() for e in self.elements],
            "rooms": [r.to_dict() for r in self.rooms],
            "summary": self.summary,
            "warnings": list(self.warnings),
        }
