"""Run the deterministic evidence engine on the canonical page raster served by the backend."""

import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

from verification import AnalyzerResult, PageFrame, validate_source


ENGINE_SOURCE = Path(__file__).resolve().parent.parent / "deterministic" / "src"
EngineState = Literal["ok", "not_installed", "failed", "page_rejected", "raster_mismatch"]


def deterministic_available() -> bool:
    return (ENGINE_SOURCE.is_dir() or find_spec("planproof_evidence") is not None) and all(
        find_spec(name) is not None for name in ("numpy", "cv2", "shapely", "skimage")
    )


def unavailable(frame: PageFrame, version: str = "planproof-evidence") -> AnalyzerResult:
    return AnalyzerResult(
        schema_version="planproof-door-result-v1", source=frame, version=version,
        status="unavailable", mode="independent", complete=False,
    )


def run_deterministic(png: bytes, frame: PageFrame) -> tuple[AnalyzerResult, EngineState]:
    """Analyze the exact PNG the viewer displays. The engine never receives ML output here."""
    try:
        if ENGINE_SOURCE.is_dir() and str(ENGINE_SOURCE) not in sys.path:
            sys.path.insert(0, str(ENGINE_SOURCE))
        from planproof_evidence import analyze
        from planproof_evidence.contracts import to_door_result_v1
    except ImportError:
        return unavailable(frame), "not_installed"

    try:
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "page.png"
            image.write_bytes(png)
            document = analyze(image, Path(folder) / "out")
            # The engine sees a single image, so its page index is always 0.
            analyzed = next((page for page in document.pages if page.page_index == 0), None)
            if analyzed is None:
                return unavailable(frame), "page_rejected"
            if (analyzed.raster_sha256 != frame.raster_sha256
                    or list(analyzed.image_size) != [frame.width, frame.height]):
                return unavailable(frame), "raster_mismatch"
            raw = to_door_result_v1(document, page_frame_override=frame.model_dump(), page_index=0)
        result = AnalyzerResult.model_validate(raw, strict=False)
        validate_source(result, frame)
    except Exception:  # The engine is a separate component; its failure must not hide the page.
        return unavailable(frame), "failed"
    return result, "ok"
