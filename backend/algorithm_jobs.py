"""Bounded, source-addressed algorithm work; interactive callers need not wait."""

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache
from pathlib import Path

from planproof_feedback.artifacts import write_json
from planproof_feedback.images import sha256
from planproof_feedback.pipeline import (
    ROOT,
    EvidenceJob,
    EvidencePipeline,
    PipelineSettings,
)


class AlgorithmPending(ValueError):
    def __init__(self, status="pending"):
        self.status = status
        super().__init__(
            "Algorithm evidence is not ready; inspect the source meanwhile."
        )


class BackgroundJobs:
    """One CPU builder, one queued job, shared futures, no lock held during work."""

    def __init__(self, capacity=2, retained=8):
        self.pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="drawing-evidence"
        )
        self.capacity, self.retained = capacity, retained
        self.lock = threading.Lock()
        self.jobs = {}

    def get(self, key, build, timeout=None):
        with self.lock:
            entry = self.jobs.get(key)
            if (
                entry
                and entry[0].done()
                and entry[0].exception()
                and time.monotonic() - getattr(entry[0], "finished_at", entry[1]) > 60
            ):
                del self.jobs[key]
                entry = None
            if entry is None:
                if sum(not f.done() for f, _ in self.jobs.values()) >= self.capacity:
                    raise AlgorithmPending("busy")
                # Completed entries are bounded; active work is never evicted/restarted.
                for old in list(self.jobs):
                    if len(self.jobs) < self.retained:
                        break
                    if self.jobs[old][0].done():
                        del self.jobs[old]
                entry = (self.pool.submit(build), time.monotonic())
                entry[0].add_done_callback(
                    lambda future: setattr(future, "finished_at", time.monotonic())
                )
                self.jobs[key] = entry
        try:
            return entry[0].result(timeout=timeout)
        except TimeoutError:
            if entry[0].done():
                return entry[0].result()  # A worker failure is not a pending result.
            raise AlgorithmPending() from None


@lru_cache(maxsize=256)
def _digest(path, size, modified):
    return sha256(Path(path))


def digest(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _digest(str(path), stat.st_size, stat.st_mtime_ns)


def recipe(settings):
    paths = [
        ROOT / "backend/pipeline_worker.py",
        ROOT / "feedback/src/planproof_feedback/pipeline.py",
    ]
    for package in ("ml", "deterministic"):
        paths.extend(sorted((ROOT / package / "src").rglob("*.py")))
        paths.extend(sorted((ROOT / package / "configs").rglob("*.yaml")))
    return {
        "version": 1,
        "model": digest(settings.checkpoint),
        "profile": digest(settings.calibration),
        "implementation": [digest(p) for p in paths],
        "geometry_python": str(Path(settings.geometry_python).resolve()),
        "ml_python": str(Path(settings.ml_python).resolve()),
        "device": settings.device,
    }


def identity(page, config):
    return {
        "raster": page.raster_sha256,
        "size": [page.width, page.height],
        "recipe": config,
    }


def load_cached(root, key, binding, settings):
    """Only accept our versioned index and receipt-verified immutable artifacts."""
    try:
        index = json.loads((root / "cache" / f"{key}.json").read_text(encoding="utf-8"))
        if index["binding"] != binding:
            return None
        name = index["directory"]
        if len(name) != 32 or any(c not in "0123456789abcdef" for c in name):
            return None
        directory = root / name
        receipt = json.loads(
            (directory / "pipeline-receipt.json").read_text(encoding="utf-8")
        )
        if (
            receipt["raster_sha256"] != binding["raster"]
            or receipt["model_sha256"] != binding["recipe"]["model"]
            or receipt["profile_sha256"] != binding["recipe"]["profile"]
            or receipt.get("labels_used") is not False
        ):
            return None
        for file, field in (
            ("page.png", "canonical_sha256"),
            ("evidence/evidence.json", "evidence_sha256"),
            ("rooms/room-boundaries.json", "boundaries_sha256"),
        ):
            if sha256(directory / file) != receipt[field]:
                return None
        job = EvidenceJob(directory, settings)
        if job.page["frame"]["image_size"] != binding["size"]:
            return None
        return job
    except (OSError, ValueError, KeyError, TypeError):
        return None


_background = BackgroundJobs()


def get_algorithm_job(page, *, timeout=None):
    settings = PipelineSettings.from_env()
    binding = identity(page, recipe(settings))
    key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    root = settings.work_root.resolve()

    def build():
        cached = load_cached(root, key, binding, settings)
        if cached is not None:
            return cached
        staging = root / "uploads"
        staging.mkdir(parents=True, exist_ok=True)
        source = staging / f"{key}.png"
        source.write_bytes(page.png)
        job = EvidencePipeline(settings).prepare(source)
        try:
            write_json(
                root / "cache" / f"{key}.json",
                {"binding": binding, "directory": job.directory.name},
            )
        except OSError:
            pass  # A cache-write failure must not discard computed evidence.
        return job

    return _background.get((str(root), key), build, timeout=timeout)
