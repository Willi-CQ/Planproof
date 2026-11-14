"""Local, resumable observation of existing agent runs. Never dispatch from GET."""

import copy
import hashlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

TERMINAL = {"completed", "failed", "interrupted"}


def box_valid(box, source):
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(
            isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n)
            for n in box
        )
        and 0 <= box[0] < box[2] <= source["width"]
        and 0 <= box[1] < box[3] <= source["height"]
    )


def pick(value, keys):
    return {k: copy.deepcopy(value[k]) for k in keys if k in value}


def rule_cues(item):
    """Measured scalar cues only; no local paths or working-pixel coordinates."""
    features = item.get("features", {})
    cues = []
    for key in (
        "parallel_line_count",
        "spacing_cv",
        "rail_count",
        "wall_context",
        "glazing_stroke_groups",
        "passage_ink_fraction",
    ):
        value = features.get(key)
        if isinstance(value, (int, float, bool)) and math.isfinite(value):
            cues.append({"name": key, "value": value})
    for key in ("arc_support", "leaf_support"):
        value = features.get("best_swing", {}).get(key, features.get(key))
        if isinstance(value, (int, float)) and math.isfinite(value):
            cues.append({"name": key, "value": value})
    if isinstance(item.get("boundary_support"), (int, float)) and math.isfinite(
        item["boundary_support"]
    ):
        cues.append({"name": "boundary_support", "value": item["boundary_support"]})
    return cues


def evidence_bundle(job, source):
    """Allowlisted geometry only; no artifact paths or on-demand algorithms."""
    shapes = []
    for item in job.document.get("observations", []):
        if item.get("class") not in {"room", "wall", "door", "window", "stairs"}:
            continue
        family = item.get("source_family")
        layer = (
            "ml"
            if family == "ml"
            else "hybrid"
            if family == "hybrid" or item.get("mode") == "hybrid"
            else "deterministic"
        )
        value = pick(
            item,
            ("id", "class", "bbox", "polygons", "method", "mode", "status", "warnings"),
        )
        if box_valid(value.get("bbox"), source):
            shapes.append(
                {
                    **value,
                    "layer": layer,
                    "source_family": family,
                    "cues": rule_cues(item),
                }
            )
    for key, cls in (("rooms", "room"), ("walls", "wall"), ("openings", None)):
        for item in job.page.get(key, []):
            value = pick(
                item,
                (
                    "id",
                    "bbox",
                    "polygons",
                    "role",
                    "status",
                    "reason",
                    "boundary_support",
                    "classification_conflict",
                ),
            )
            if box_valid(value.get("bbox"), source):
                shapes.append(
                    {
                        **value,
                        "id": "boundary:" + str(item["id"]),
                        "layer": "hybrid",
                        "class": cls or item.get("type", "opening"),
                        "source_family": "ml_assisted",
                        "cues": rule_cues(item),
                    }
                )
    comparisons = [
        pick(c, ("relation", "source_ids", "iou", "bbox", "semantics"))
        for c in job.document.get("comparisons", [])
        if box_valid(c.get("bbox"), source)
    ]
    return {
        "version": 1,
        "status": "available",
        "source": source,
        "shapes": shapes,
        "comparisons": comparisons,
        "warnings": job.catalog.get("warnings", []),
        "scope": "Available algorithm output; only requested classes were supplied to the agent. Candidates are not verified facts.",
    }


def public_event(stage, state, payload, source):
    """Never publish raw provider messages, model thoughts, signatures or paths."""
    data = pick(
        payload,
        (
            "seconds",
            "classes",
            "checks",
            "unresolved",
            "next_action",
            "question",
            "phase",
            "cached",
            "delivered_ids",
            "category",
            "attempt",
            "inspection_id",
        ),
    )
    if "summary" in payload:
        data["summary"] = str(payload["summary"])[:2000]
    images = []
    for i in payload.get("images", []):
        if box_valid(i.get("canonical_bbox"), source):
            images.append(
                {
                    **pick(i, ("image_id", "canonical_bbox", "sent_size")),
                    "page_index": source["page_index"],
                }
            )
    data["images"] = images
    data["citations"] = [
        {**pick(c, ("image_id", "canonical_bbox")), "page_index": source["page_index"]}
        for c in payload.get("canonical_citations", [])
        if box_valid(c.get("canonical_bbox"), source)
    ]
    advisory = payload.get("advisory")
    if isinstance(advisory, dict):
        data["advisory"] = pick(advisory, ("status", "answer", "findings", "verdict"))
    return {"stage": stage, "state": state, **data}


class RunStore:
    def __init__(self, root, workers=2, waiting=4):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="planproof-run"
        )
        self.slots = threading.BoundedSemaphore(workers + waiting)
        self.runs = {}
        for file in self.root.glob("*.json"):
            try:
                record = json.loads(file.read_text(encoding="utf-8"))
                run_id = self.valid_id(record["request_id"])
                if record["status"] not in TERMINAL:
                    record["status"] = "interrupted"
                    record["message"] = (
                        "Server restarted. This question was not resubmitted."
                    )
                    self._persist(record)
                self.runs[run_id] = record
            except (ValueError, KeyError, OSError):
                continue

    @staticmethod
    def valid_id(value):
        return str(UUID(str(value)))

    def _persist(self, record):
        path = self.root / (self.valid_id(record["request_id"]) + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    def submit(self, run_id, source, request, execute):
        run_id = self.valid_id(run_id)
        fingerprint = hashlib.sha256(
            json.dumps({"source": source, "request": request}, sort_keys=True).encode()
        ).hexdigest()
        with self.lock:
            if run_id in self.runs:
                if self.runs[run_id]["fingerprint"] != fingerprint:
                    raise ValueError(
                        "Request ID belongs to a different question or drawing."
                    )
                return self.snapshot(run_id)
            if not self.slots.acquire(blocking=False):
                raise OverflowError("The agent is busy. Please try again shortly.")
            record = {
                "request_id": run_id,
                "fingerprint": fingerprint,
                "status": "queued",
                "source": source,
                "events": [],
                "result": None,
                "created": time.time(),
                "question": request.get("question", ""),
                "evidence": {
                    "version": 1,
                    "status": "not_run",
                    "source": source,
                    "shapes": [],
                    "comparisons": [],
                },
            }
            try:
                self._persist(record)  # Registration must be durable before acceptance.
                self.runs[run_id] = record
                self.pool.submit(self._run, run_id, execute)
            except Exception:
                self.slots.release()
                self.runs.pop(run_id, None)
                raise
            return self.snapshot(run_id)

    def _run(self, run_id, execute):
        def observe(stage, state, **payload):
            try:
                with self.lock:
                    record = self.runs[run_id]
                    if stage == "algorithm_bundle":
                        record["evidence"] = evidence_bundle(
                            payload["job"], record["source"]
                        )
                    else:
                        event = public_event(stage, state, payload, record["source"])
                        event.update(
                            sequence=len(record["events"]) + 1,
                            elapsed=round(time.time() - record["created"], 3),
                        )
                        record["events"].append(event)
                    try:
                        self._persist(record)
                    except (OSError, ValueError):
                        record["telemetry_warning"] = (
                            "Activity is available in this session but could not be saved on the server."
                        )
            except Exception:
                # The observer must never alter the agent's answer or execution.
                pass

        try:
            with self.lock:
                self.runs[run_id]["status"] = "running"
            observe("run", "running", summary="Inspecting the selected drawing page.")
            result = execute(observe)
            with self.lock:
                record = self.runs[run_id]
                record["result"] = result
                record["status"] = "completed"
                observe(
                    "answer",
                    "completed",
                    summary="Answer returned. Check the cited source and remaining uncertainty.",
                    canonical_citations=result.get("canonical_citations", []),
                )
        except Exception:
            with self.lock:
                self.runs[run_id]["status"] = "failed"
                self.runs[run_id]["message"] = (
                    "The run could not finish. No question was automatically resubmitted."
                )
                observe("run", "unavailable", summary="The run could not finish.")
        finally:
            self.slots.release()

    def snapshot(self, run_id, after=0):
        with self.lock:
            record = self.runs[self.valid_id(run_id)]
            value = {
                k: v
                for k, v in record.items()
                if k not in {"fingerprint", "evidence", "events"}
            }
            value["events"] = [e for e in record["events"] if e["sequence"] > after]
            value["last_sequence"] = len(record["events"])
            return copy.deepcopy(value)

    def evidence(self, run_id):
        with self.lock:
            return copy.deepcopy(self.runs[self.valid_id(run_id)]["evidence"])
