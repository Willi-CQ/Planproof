from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError("Training config must be a mapping")
    size = cfg.get("image_size", 1024)
    if size < 64 or size % 32:
        raise ValueError("image_size must be >=64 and divisible by 32")
    if cfg.get("effective_batch", 16) < 1 or cfg.get("micro_batch", 2) < 1:
        raise ValueError("Batch sizes must be positive")
    phases = cfg.get("phases", [])
    if not phases or any(p.get("steps", 0) < 1 for p in phases):
        raise ValueError("Each training phase needs positive steps")
    for phase in phases:
        weights = phase.get("sources", {})
        if not weights or any(w <= 0 for w in weights.values()):
            raise ValueError("Each phase needs positive source sampling weights")
        if any("stoa" in k.lower() for k in weights):
            raise ValueError("Stoa is evaluation/reference only")
    if not 0 <= cfg.get("crop_probability", 0.5) <= 1:
        raise ValueError("Invalid crop probability")
    if not isinstance(cfg.get("data_workers", 0), int) or not 0 <= cfg.get("data_workers", 0) <= 16:
        raise ValueError("data_workers must be between 0 and 16")
    if cfg.get("box_loss", "smooth_l1") not in {"smooth_l1", "giou"}:
        raise ValueError("box_loss must be smooth_l1 or giou")
    box_weight = cfg.get("box_loss_weight", 2.0)
    if not isinstance(box_weight, (int, float)) or not math.isfinite(box_weight) or box_weight < 0:
        raise ValueError("box_loss_weight must be finite and nonnegative")
    cfg["image_size"] = size
    return cfg


def config_digest(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
