"""Read-only run supervision; estimates are not provider billing or accuracy claims."""

import math
from pathlib import Path
import time

from .schema import read_json


def status(run: Path, hourly_rate=None):
    if hourly_rate is not None and (not math.isfinite(hourly_rate) or hourly_rate < 0):
        raise ValueError("hourly_rate must be finite and nonnegative")
    report = read_json(run / "status.json")
    warnings = []
    age = max(0.0, time.time() - report["updated_unix"])
    if report["state"] == "training" and age > 300:
        warnings.append("No update for over five minutes: inspect process, data loading or validation")
    failure = run / "failure.json"
    if failure.exists():
        error = read_json(failure)
        if error["updated_unix"] >= report["updated_unix"]:
            warnings.append("Run failed: " + error["error"])
    if not math.isfinite(report["loss"]):
        warnings.append("Nonfinite loss; do not use this update")
    validation = []
    for path in sorted((run / "validation").glob("step-*.json")):
        item = read_json(path)
        if item.get("selection_score") is not None:
            validation.append({"step": item["step"], "score": item["selection_score"]})
    if len(validation) >= 4:
        recent = validation[-3:]
        prior_best = max(v["score"] for v in validation[:-3])
        if max(v["score"] for v in recent) <= prior_best + 0.002:
            warnings.append(
                "Three validation checks without >0.002 improvement; review before spending more time"
            )
    rate = report.get("steps_per_second", 0)
    return {
        **report,
        "status_age_seconds": age,
        "warnings": warnings,
        "recent_validation": validation[-5:],
        "remaining_step_time_estimate_seconds": max(0, report["planned_stop_step"] - report["step"]) / rate
        if rate > 0
        else None,
        "session_compute_cost_estimate_eur": report["elapsed_seconds"] / 3600 * hourly_rate
        if hourly_rate is not None
        else None,
        "best_checkpoint_exists": (run / "best.pt").is_file(),
        "last_checkpoint_exists": (run / "last.pt").is_file(),
        "notes": "ETA depends on measured throughput and validation overhead. Cost excludes idle instance time, prior sessions, storage and taxes. A paused training process does not stop cloud billing.",
    }
