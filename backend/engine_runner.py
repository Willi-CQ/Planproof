"""Run the team's two engines against the exact same canonical PNG."""

import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from ml_adapter import adapt_ml
from verification import AnalyzerResult, PageFrame, validate_source


logger = logging.getLogger(__name__)
run_lock = Lock()
MAX_RESULT = 4 * 1024 * 1024


def enabled_engines() -> dict[str, bool]:
    return {
        "deterministic": bool(os.getenv("PLANPROOF_DETERMINISTIC_PYTHON")),
        "ml": all(os.getenv(name) for name in (
            "PLANPROOF_ML_PYTHON", "PLANPROOF_ML_CHECKPOINT", "PLANPROOF_ML_CALIBRATION",
        )),
    }


def read_result(path: Path) -> dict:
    with path.open("rb") as file:
        payload = file.read(MAX_RESULT + 1)
    if len(payload) > MAX_RESULT:
        raise ValueError("Engine result is too large.")
    return json.loads(payload)


def run_engine(name: str, image: Path, frame: PageFrame, work: Path) -> AnalyzerResult | None:
    try:
        if name == "ml":
            output = work / "ml"
            command = [
                os.environ["PLANPROOF_ML_PYTHON"], "-B", "-m", "planproof_ml", "infer",
                "--checkpoint", os.environ["PLANPROOF_ML_CHECKPOINT"],
                "--calibration", os.environ["PLANPROOF_ML_CALIBRATION"], "--global-view",
                "--image", str(image), "--page-index", str(frame.page_index),
                "--out", str(output), "--device", os.getenv("PLANPROOF_ML_DEVICE", "cpu"),
            ]
            result_path = output / "detections.json"
        else:
            result_path = work / "deterministic.json"
            command = [
                os.environ["PLANPROOF_DETERMINISTIC_PYTHON"], "-B",
                str(Path(__file__).with_name("deterministic_worker.py").resolve()),
                str(image), str(result_path),
            ]
        env = os.environ.copy()
        for secret in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "NEMOTRON_API_KEY", "NVIDIA_NEMOTRON_KEY"):
            env.pop(secret, None)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")
        subprocess.run(command, env=env, check=True, timeout=120,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raw = read_result(result_path)
        if name == "ml":
            return adapt_ml(raw, frame)
        result = AnalyzerResult.model_validate(raw)
        input_hash = sha256(image.read_bytes()).hexdigest()
        expected_raster = frame.model_copy(update={
            "document_id": input_hash,
            "page_index": 0,
            "source_sha256": input_hash,
        })
        validate_source(result, expected_raster)
        if result.mode != "independent" or result.status == "unavailable":
            return None
        return result.model_copy(update={"source": frame})
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        logger.warning("%s worker unavailable (%s)", name, type(exc).__name__)
        return None


def run_engines(png: bytes, frame: PageFrame) -> dict[str, AnalyzerResult | None]:
    configured = enabled_engines()
    if not any(configured.values()):
        return {}
    with run_lock, TemporaryDirectory(prefix="planproof-") as directory, ThreadPoolExecutor(max_workers=2) as pool:
        work = Path(directory)
        image = work / "page.png"
        image.write_bytes(png)
        jobs = {name: pool.submit(run_engine, name, image, frame, work)
                for name, enabled in configured.items() if enabled}
        return {name: job.result() for name, job in jobs.items()}
