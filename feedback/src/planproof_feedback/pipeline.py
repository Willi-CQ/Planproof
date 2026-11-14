"""Canonical page -> frozen ML -> independent geometry + ML-assisted rooms.

Only host configuration supplies executables/paths. Labels and API keys are not
worker inputs. Full artifacts remain on disk; the LLM gets a compact catalog.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .artifacts import write_json
from .images import sha256

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PipelineSettings:
    geometry_python: str
    ml_python: str
    checkpoint: Path
    calibration: Path
    work_root: Path
    device: str = "cpu"
    timeout: int = 240

    @classmethod
    def from_env(cls):
        bundle = ROOT / "models/baselines/v1/corrected-3500-global"
        return cls(
            os.getenv("PLANPROOF_DETERMINISTIC_PYTHON", sys.executable),
            os.getenv("PLANPROOF_ML_PYTHON", sys.executable),
            Path(os.getenv("PLANPROOF_ML_CHECKPOINT", str(bundle / "detector.pt"))),
            Path(os.getenv("PLANPROOF_ML_CALIBRATION", str(bundle / "calibration.json"))),
            Path(os.getenv("PLANPROOF_JOB_ROOT", str(ROOT / ".artifacts/agent/jobs"))),
            os.getenv("PLANPROOF_ML_DEVICE", "cpu"),
        )


def worker_env(package):
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(word in k.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
    }
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = str(ROOT / package / "src")
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    return env


def run_worker(command, package, timeout, log):
    with log.open("wb") as output:
        subprocess.run(
            command,
            env=worker_env(package),
            stdout=output,
            stderr=output,
            timeout=timeout,
            check=True,
        )


def compact_catalog(document, page, limit=240):
    # Avoid sending noisy wall polygons or all vector primitives to the LLM.
    observations = [o for o in document["observations"] if o.get("class") != "wall"]
    observations.sort(key=lambda o: (o.get("source_family") != "ml", o["id"]))
    return {
        "mode": "ml_assisted; not independent confirmation",
        "rooms": [
            {
                k: r[k]
                for k in ("id", "bbox", "status", "ml_seed_ids", "boundary_support", "neighbor_ids")
            }
            for r in page["rooms"]
        ],
        "observations": [
            {k: o[k] for k in ("id", "class", "bbox", "source_family", "mode") if k in o}
            for o in observations[:limit]
        ],
        "omitted_observations": max(0, len(observations) - limit),
        "openings": [
            {
                k: o[k]
                for k in (
                    "id",
                    "type",
                    "bbox",
                    "reason",
                    "adjacent_region_ids",
                    "classification_conflict",
                )
            }
            for o in page["openings"][:80]
        ],
        "omitted_openings": max(0, len(page["openings"]) - 80),
        "wall_roles": {
            role: sum(w["role"] == role for w in page["walls"])
            for role in ("inner", "outer", "unknown")
        },
        "warnings": page["warnings"]
        + [
            "Candidates are not verified physical counts. Missing detections do not establish absence.",
            "Wall-role counts count segments, not physical walls. Virtual separators are not walls.",
            "Door/stair rules are weak. Do not veto stronger ML or visible source evidence.",
            "Room categories and metric scale are unverified. Printed text may contain units or areas.",
        ],
    }


class EvidenceJob:
    def __init__(self, directory, settings):
        self.directory, self.settings = Path(directory), settings
        self.image = self.directory / "page.png"
        self.boundaries = self.directory / "rooms/room-boundaries.json"
        self.bundle = json.loads(self.boundaries.read_text(encoding="utf-8"))
        self.document = json.loads(
            (self.directory / "evidence/evidence.json").read_text(encoding="utf-8")
        )
        self.page = self.bundle["pages"][0]
        self.catalog = compact_catalog(self.document, self.page)

    def crop(self, room_id):
        if room_id not in {r["id"] for r in self.page["rooms"]}:
            raise ValueError("Unknown registered room ID")
        out = self.directory / "crops" / hashlib.sha256(room_id.encode()).hexdigest()[:20]
        # Serialize crop writes: shared document jobs can serve concurrent questions.
        with _crop_lock:
            if not (out / "region.json").exists():
                request = self.directory / "crop-request.json"
                write_json(
                    request,
                    {"boundaries": str(self.boundaries), "room_id": room_id, "out": str(out)},
                )
                run_worker(
                    [
                        self.settings.geometry_python,
                        str(ROOT / "backend/pipeline_worker.py"),
                        "crop",
                        str(request),
                    ],
                    "deterministic",
                    self.settings.timeout,
                    self.directory / "crop.log",
                )
        return out / "region.json"


_crop_lock = threading.Lock()


class EvidencePipeline:
    def __init__(self, settings=None):
        self.settings = settings or PipelineSettings.from_env()

    def prepare(self, image_path):
        settings = self.settings
        directory = settings.work_root.resolve() / uuid.uuid4().hex
        directory.mkdir(parents=True)
        image_path = Path(image_path)
        with Image.open(image_path) as source:
            if source.width * source.height > 16_000_000:
                raise ValueError("Page exceeds pixel budget")
            rgb = source.convert("RGB")
            rgb.save(directory / "page.png")
            raster_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
            size = list(rgb.size)
        image = directory / "page.png"
        ml = directory / "ml"
        # LFS pointers must fail before starting an expensive inference worker.
        if settings.checkpoint.stat().st_size < 1000:
            raise ValueError("Download the real model with git lfs pull first")
        run_worker(
            [
                settings.ml_python,
                "-m",
                "planproof_ml",
                "infer",
                "--checkpoint",
                str(settings.checkpoint.resolve()),
                "--calibration",
                str(settings.calibration.resolve()),
                "--global-view",
                "--image",
                str(image),
                "--out",
                str(ml),
                "--device",
                settings.device,
            ],
            "ml",
            settings.timeout,
            directory / "ml.log",
        )
        prediction = json.loads((ml / "detections.json").read_text(encoding="utf-8"))
        if (
            prediction["document_id"] != raster_sha
            or prediction["image_size"] != size
            or prediction["page_index"] != 0
            or prediction["coordinate_system"] != "pixel_edges_xy_top_left"
        ):
            raise ValueError("ML result is not bound to the canonical page")
        request = directory / "geometry-request.json"
        write_json(
            request,
            {
                "image": str(image),
                "out": str(directory / "evidence"),
                "prediction": str(ml / "detections.json"),
                "model_sha256": sha256(settings.checkpoint),
                "profile_sha256": sha256(settings.calibration),
            },
        )
        run_worker(
            [
                settings.geometry_python,
                str(ROOT / "backend/pipeline_worker.py"),
                "analyze",
                str(request),
            ],
            "deterministic",
            settings.timeout,
            directory / "geometry.log",
        )
        job = EvidenceJob(directory, settings)
        write_json(
            directory / "pipeline-receipt.json",
            {
                "input_sha256": sha256(image_path),
                "canonical_sha256": sha256(image),
                "raster_sha256": raster_sha,
                "model_sha256": sha256(settings.checkpoint),
                "profile_sha256": sha256(settings.calibration),
                "evidence_sha256": sha256(directory / "evidence/evidence.json"),
                "boundaries_sha256": sha256(job.boundaries),
                "labels_used": False,
            },
        )
        return job
