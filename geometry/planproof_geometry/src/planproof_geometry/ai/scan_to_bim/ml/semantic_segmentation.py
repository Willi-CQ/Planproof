"""Pretrained semantic point-cloud segmentation adapter.

Entkoppelt die 14-Schritt-Pipeline von konkreten ML-Frameworks. Der Adapter
waehlt das Backend (Superpoint Transformer, RandLA-Net, ...) je nach Modell-
Datei oder Environment. Fehlt ein Modell oder die Framework-Dependencies, wird
`load_segmenter()` zu `None` — die Pipeline faellt dann auf reine Heuristik
zurueck (siehe Step 3.5 `SemanticSegmentationStep`, soft_fail).

Kanonisches Label-Set (ueber Backends hinweg vereinheitlicht):
    0 unknown | 1 wall | 2 floor | 3 ceiling | 4 door |
    5 window | 6 beam | 7 column | 8 clutter

Jedes Backend muss seine nativen Klassen auf diese Menge mappen, damit der
Rest der Pipeline (Voting in Step 9, RANSAC-Masking in Step 4) Backend-
agnostisch arbeiten kann.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Kanonisches Label-Set (backend-agnostisch)
# --------------------------------------------------------------------------- #

CANONICAL_LABELS: Tuple[str, ...] = (
    "unknown",   # 0 — nichts klassifiziert / Rauschen
    "wall",      # 1
    "floor",     # 2
    "ceiling",   # 3
    "door",      # 4
    "window",    # 5
    "beam",      # 6
    "column",    # 7
    "clutter",   # 8 — Moebel/Objekte, nicht BIM-strukturell
)

LABEL_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(CANONICAL_LABELS)}
INDEX_TO_LABEL: Dict[int, str] = {i: name for i, name in enumerate(CANONICAL_LABELS)}

# S3DIS-Klassen -> kanonische Indices. SPT/RandLA-pretrained auf S3DIS liefert
# 13 Klassen; wir mappen Strukturklassen auf `wall/floor/ceiling/door/window/
# beam/column`, den Rest auf `clutter`.
S3DIS_LABEL_MAP: Dict[str, str] = {
    "ceiling": "ceiling",
    "floor": "floor",
    "wall": "wall",
    "beam": "beam",
    "column": "column",
    "window": "window",
    "door": "door",
    "table": "clutter",
    "chair": "clutter",
    "sofa": "clutter",
    "bookcase": "clutter",
    "board": "clutter",
    "clutter": "clutter",
}


@dataclass
class SemanticSegmentationResult:
    """Per-Punkt Klassifikationsergebnis.

    labels:      (N,) int32 — Werte in [0..len(CANONICAL_LABELS)-1]
    confidence:  (N,) float32 — Softmax/Voting-Wahrscheinlichkeit [0..1]
    backend:     freier String (z.B. "spt_s3dis_v1", "randla_s3dis", "stub")
    avg_confidence: bequemer Aggregatwert
    """

    labels: np.ndarray
    confidence: np.ndarray
    backend: str

    @property
    def avg_confidence(self) -> float:
        if self.confidence.size == 0:
            return 0.0
        return float(self.confidence.mean())

    def label_names(self) -> np.ndarray:
        """(N,) array of canonical label strings — debug-freundlich."""
        return np.array([INDEX_TO_LABEL.get(int(v), "unknown") for v in self.labels])


# --------------------------------------------------------------------------- #
# Adapter-Interface
# --------------------------------------------------------------------------- #

class SemanticSegmenter:
    """Adapter, den die Pipeline aufruft. Backends werden via `_impl` gekapselt.

    Nutzung:
        seg = load_segmenter(model_path="...")      # None wenn unverfuegbar
        if seg is not None:
            res = seg.infer(points, colors=colors, logger=ctx.log)
    """

    def __init__(
        self,
        backend_name: str,
        impl: Callable[..., SemanticSegmentationResult],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.backend_name = backend_name
        self._impl = impl
        self.metadata: Dict[str, Any] = metadata or {}

    def infer(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        *,
        max_points: Optional[int] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> SemanticSegmentationResult:
        """Per-Punkt Inferenz. Gibt immer ein vollstaendiges Label-Array zurueck."""
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points muss (N,3) sein, got shape {points.shape}")
        if points.shape[0] == 0:
            return SemanticSegmentationResult(
                labels=np.zeros(0, dtype=np.int32),
                confidence=np.zeros(0, dtype=np.float32),
                backend=self.backend_name,
            )
        return self._impl(points=points, colors=colors, max_points=max_points, logger=logger)


# --------------------------------------------------------------------------- #
# Backend-Registry (SPT, RandLA, Stub)
# --------------------------------------------------------------------------- #

def load_segmenter(
    model_path: Optional[str] = None,
    *,
    backend: Optional[str] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> Optional[SemanticSegmenter]:
    """Laedt ein Pretrained-Modell. Gibt `None` zurueck, wenn nichts verfuegbar.

    Die Auflösungsreihenfolge ist:
        1. Explizites `backend` + `model_path`
        2. Aus `model_path`-Endung/Filename ableiten (`.ckpt` → SPT, `.pth`/`.pt` → RandLA)
        3. Kein `model_path` → `None`
    """

    def _log(msg: str) -> None:
        if logger:
            logger(msg)

    if not model_path:
        _log("SemSeg: kein model_path konfiguriert, Heuristik-Only-Modus")
        return None

    p = Path(model_path)
    if not p.exists():
        _log(f"SemSeg: model_path existiert nicht ({p}) — Heuristik-Only-Modus")
        return None

    resolved_backend = (backend or _infer_backend_from_path(p)).lower()

    if resolved_backend == "spt":
        impl = _try_load_spt(p, logger=logger)
    elif resolved_backend == "randla":
        impl = _try_load_randla(p, logger=logger)
    elif resolved_backend == "stub":
        impl = _build_stub(label="clutter")
    else:
        _log(f"SemSeg: unbekanntes Backend '{resolved_backend}' — Heuristik-Only-Modus")
        return None

    if impl is None:
        return None

    return SemanticSegmenter(
        backend_name=f"{resolved_backend}:{p.name}",
        impl=impl,
        metadata={"model_path": str(p), "backend": resolved_backend},
    )


def _infer_backend_from_path(p: Path) -> str:
    suffix = p.suffix.lower()
    name = p.stem.lower()
    if suffix == ".ckpt" or "spt" in name or "superpoint" in name:
        return "spt"
    if suffix in (".pth", ".pt") or "randla" in name:
        return "randla"
    return "spt"  # Default-Annahme: Lightning-Checkpoint


# --------------------------------------------------------------------------- #
# Backend: Superpoint Transformer (primary)
# --------------------------------------------------------------------------- #

def _try_load_spt(
    ckpt_path: Path,
    *,
    logger: Optional[Callable[[str], None]] = None,
) -> Optional[Callable[..., SemanticSegmentationResult]]:
    """Versucht SPT via torch + torch-geometric zu laden.

    Wenn die Abhaengigkeiten fehlen, wird das Modell im Inferenzmodus
    heruntergestuft und `None` zurueckgegeben. Die Pipeline laeuft dann rein
    heuristik-basiert weiter (soft-fail-Pfad).
    """

    def _log(msg: str) -> None:
        if logger:
            logger(msg)

    try:
        import torch  # noqa: F401
    except Exception as exc:  # torch nicht installiert → nichts zu tun
        _log(f"SemSeg/SPT: torch nicht verfuegbar ({exc}), skip")
        return None

    try:
        # Beide imports sind best-effort — fehlt eines, faellt SPT aus.
        import torch_geometric  # noqa: F401
    except Exception as exc:
        _log(f"SemSeg/SPT: torch-geometric nicht verfuegbar ({exc}), skip")
        return None

    # Lazy import der konkreten SPT-Klasse — aktuell als Platzhalter markiert,
    # weil der produktive Lader sehr SPT-spezifisch ist und zum Zeitpunkt von
    # Tag 4-5 ohne konkreten Checkpoint nicht gegen echte Gewichte getestet
    # werden kann. Sobald `models/scan_seg/spt_s3dis_v1.ckpt` verfuegbar ist,
    # wird diese Funktion um den eigentlichen `SuperCluster.load_from_checkpoint`-
    # Aufruf ergaenzt. Bis dahin: soft-fail.
    _log(
        "SemSeg/SPT: Checkpoint-Lader noch nicht implementiert "
        f"(ckpt={ckpt_path.name}); skip bis Modell-Integration abgeschlossen."
    )
    return None


# --------------------------------------------------------------------------- #
# Backend: RandLA-Net (fallback)
# --------------------------------------------------------------------------- #

def _try_load_randla(
    ckpt_path: Path,
    *,
    logger: Optional[Callable[[str], None]] = None,
) -> Optional[Callable[..., SemanticSegmentationResult]]:
    """RandLA-Net via open3d-ml laden. Soft-fail wenn nicht verfuegbar."""

    def _log(msg: str) -> None:
        if logger:
            logger(msg)

    try:
        import torch  # noqa: F401
    except Exception as exc:
        _log(f"SemSeg/RandLA: torch nicht verfuegbar ({exc}), skip")
        return None

    try:
        import open3d.ml.torch as ml3d  # type: ignore  # noqa: F401
    except Exception as exc:
        _log(f"SemSeg/RandLA: open3d-ml nicht verfuegbar ({exc}), skip")
        return None

    _log(
        "SemSeg/RandLA: Checkpoint-Lader noch nicht implementiert "
        f"(ckpt={ckpt_path.name}); skip bis Modell-Integration abgeschlossen."
    )
    return None


# --------------------------------------------------------------------------- #
# Backend: Stub (Test- / Fallback-Pfad)
# --------------------------------------------------------------------------- #

def _build_stub(label: str = "clutter") -> Callable[..., SemanticSegmentationResult]:
    """Stub-Backend fuer Smoke-Tests: gibt fuer alle Punkte die gleiche Klasse zurueck."""

    default_idx = LABEL_TO_INDEX.get(label, LABEL_TO_INDEX["clutter"])

    def _infer(
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        *,
        max_points: Optional[int] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> SemanticSegmentationResult:
        n = points.shape[0]
        return SemanticSegmentationResult(
            labels=np.full(n, default_idx, dtype=np.int32),
            confidence=np.full(n, 0.1, dtype=np.float32),
            backend="stub",
        )

    return _infer
