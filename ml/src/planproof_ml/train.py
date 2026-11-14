"""Resumable single-GPU/DDP training with split and dataset fingerprint checks."""

from __future__ import annotations

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import config_digest, load_config
from .dataset import sample, tensor_batch
from .evaluate import evaluate
from .losses import detection_loss
from .manifest import fingerprint, load_manifest
from .model import Detector
from .schema import schema_digest, write_json, sha256
from .storage import ensure_artifact_directory


def preflight(cfg, records):
    if any(r.get("simulated") for r in records) and not cfg.get("allow_simulated", False):
        raise ValueError("Generated diagnostic fixtures are not a production training dataset")
    if any(r.get("limited_import") for r in records) and not cfg.get("allow_limited", False):
        raise ValueError("Diagnostic --limit imports must be replaced with the full source before training")
    training = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "val"]
    if not training or not validation:
        raise ValueError("Both training and validation groups are required")
    available = {r["source"] for r in training}
    required = {source for phase in cfg["phases"] for source in phase["sources"]}
    missing = required - available
    if missing:
        raise ValueError(
            f"Recipe requires prepared training sources: {sorted(missing)}. No silent source substitution."
        )
    if not cfg.get("allow_simulated", False):
        positive = {c for record in training for c in record.get("positive_classes", [])}
        missing_targets = {"wall", "room", "door", "window", "stairs"} - positive
        if missing_targets:
            raise ValueError(f"Training corpus lacks positive targets for {sorted(missing_targets)}")
    return {
        "train": len(training),
        "val": len(validation),
        "sources": sorted(available),
        "required_sources": sorted(required),
        "fingerprint": fingerprint(records),
    }


def phase_at(phases, step):
    start = 0
    for phase in phases:
        if step < start + phase["steps"]:
            return phase, step - start
        start += phase["steps"]
    raise ValueError("Step outside training schedule")


def atomic_checkpoint(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def train(
    config: Path,
    manifest: Path,
    run: Path,
    device="cuda",
    resume: Path | None = None,
    max_steps=None,
    max_minutes=None,
    init_weights: Path | None = None,
):
    started = time.monotonic()
    if resume and init_weights:
        raise ValueError("Choose exact --resume or new-run --init-weights, not both")
    if max_minutes is not None and (not math.isfinite(max_minutes) or max_minutes <= 0):
        raise ValueError("max_minutes must be finite and positive")
    cfg = load_config(config)
    ensure_artifact_directory(run)
    records = load_manifest(manifest, verify_files=True)
    inventory = preflight(cfg, records)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA GPU unavailable. Use doctor --require-gpu on the GPU host; --device cpu is for checks."
            )
        device = f"cuda:{local_rank}" if world > 1 else ("cuda:0" if device == "cuda" else device)
        torch.cuda.set_device(torch.device(device))
    if world > 1:
        torch.distributed.init_process_group("nccl" if device.startswith("cuda") else "gloo")
    micro = cfg.get("micro_batch", 2)
    effective = cfg.get("effective_batch", 16)
    if effective % (micro * world):
        raise ValueError("effective_batch must be divisible by micro_batch * WORLD_SIZE")
    accumulation = effective // (micro * world)
    directory_error = None
    if rank == 0:
        if not resume and run.exists() and any(run.iterdir()):
            directory_error = "Run directory is nonempty; choose a new run or use --resume"
        else:
            run.mkdir(parents=True, exist_ok=True)
    if world > 1:
        shared = [directory_error]
        torch.distributed.broadcast_object_list(shared, src=0)
        directory_error = shared[0]
    if directory_error:
        if world > 1:
            torch.distributed.destroy_process_group()
        raise ValueError(directory_error)
    seed = cfg.get("seed", 42)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed + rank)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if not device.startswith("cuda"):
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    # Resumed runs never re-download encoder initialization weights.
    model = Detector(None if resume or init_weights else cfg.get("encoder_weights", "imagenet")).to(device)
    initialization = {"mode": "encoder", "encoder_weights": cfg.get("encoder_weights", "imagenet")}
    if init_weights:
        initial = torch.load(init_weights, map_location="cpu", weights_only=True)
        metadata = initial.get("model_metadata", {})
        if (
            metadata.get("schema_digest") != schema_digest()
            or metadata.get("architecture") != model.metadata()["architecture"]
        ):
            raise ValueError("Initialization checkpoint architecture or class schema is incompatible")
        model.load_state_dict(initial["model"], strict=True)
        initialization = {
            "mode": "checkpoint_transfer",
            "checkpoint": str(init_weights.resolve()),
            "checkpoint_sha256": sha256(init_weights),
            "source_step": initial["step"],
            "source_manifest_fingerprint": initial["manifest_fingerprint"],
            "source_config_digest": initial["config_digest"],
            "optimizer_reset": True,
            "schedule_reset": True,
        }
        del initial
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "base_lr": cfg.get("encoder_lr", 1e-4)},
            {
                "params": [p for name, p in model.named_parameters() if not name.startswith("encoder.")],
                "base_lr": cfg.get("head_lr", 3e-4),
            },
        ],
        lr=cfg.get("head_lr", 3e-4),
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    amp = device.startswith("cuda")
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
    start = 0
    best = -1.0
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=True)
        initialization = saved.get("initialization", {"mode": "legacy_resume"})
        if saved["manifest_fingerprint"] != inventory["fingerprint"]:
            raise ValueError("Dataset changed since checkpoint; resume refused")
        if (
            saved["config_digest"] != config_digest(cfg)
            or saved["model_metadata"]["schema_digest"] != schema_digest()
        ):
            raise ValueError("Config or class schema changed since checkpoint")
        if saved["world_size"] != world:
            raise ValueError("Resume with the same WORLD_SIZE for reproducibility")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start = saved["step"]
        best = saved["best_score"]
        torch.set_rng_state(saved["rng_cpu"].cpu())
        if amp and saved.get("rng_cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["rng_cuda"]])
    plain = model
    if world > 1:
        model = DDP(model, device_ids=[local_rank] if amp else None)
    if rank == 0:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = os.environ.get("PLANPROOF_SOURCE_COMMIT", "unknown")
        write_json(
            run / "run.json",
            {
                "config": cfg,
                "inventory": inventory,
                "schema_digest": schema_digest(),
                "torch": torch.__version__,
                "device": device,
                "world_size": world,
                "git_commit": commit,
                "manifest": str(manifest.resolve()),
                "config_digest": config_digest(cfg),
                "initialization": initialization,
                "determinism": "deterministic algorithms warn-only; per-step seeded source/crop sampling",
            },
        )
    training = {}
    for record in records:
        if record["split"] == "train":
            training.setdefault(record["source"], []).append(record)
    total = sum(p["steps"] for p in cfg["phases"])
    stop = min(total, max_steps) if max_steps is not None else total
    if stop <= start:
        raise ValueError("Requested stop step is not after resumed step")
    workers = cfg.get("data_workers", 0)
    prefetch = ThreadPoolExecutor(max_workers=workers) if workers else None
    pending = deque()
    tasks = iter((s, a) for s in range(start, stop) for a in range(accumulation))

    def submit_next():
        try:
            key = next(tasks)
        except StopIteration:
            return
        pending.append(
            (key, prefetch.submit(_sample_microbatch, key, training, manifest.parent, cfg, seed, rank))
        )

    def next_examples(step, acc):
        if prefetch is None:
            return _sample_microbatch((step, acc), training, manifest.parent, cfg, seed, rank)
        key, future = pending.popleft()
        if key != (step, acc):
            raise RuntimeError("Prefetch order differs from deterministic sampling schedule")
        examples = future.result()
        submit_next()
        return examples

    if prefetch is not None:
        for _ in range(workers * 2):
            submit_next()
    stop_reason = "step_limit"
    last_validation = None
    try:
        for step in range(start, stop):
            model.train()
            phase, within = phase_at(cfg["phases"], step)
            warmup = min(1.0, (within + 1) / max(1, min(500, phase["steps"] // 10)))
            cosine = 0.5 * (1 + math.cos(math.pi * within / max(1, phase["steps"])))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * phase.get("lr_scale", 1.0) * warmup * max(0.01, cosine)
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            data_wait = 0.0
            for acc in range(accumulation):
                wait_started = time.monotonic()
                examples = next_examples(step, acc)
                batch = tensor_batch(examples, device)
                data_wait += time.monotonic() - wait_started
                synchronization = model.no_sync() if world > 1 and acc < accumulation - 1 else nullcontext()
                with synchronization:
                    with torch.autocast(device_type="cuda", dtype=dtype, enabled=amp):
                        predictions = model(batch["image"])
                        loss, parts = detection_loss(
                            predictions,
                            batch,
                            cfg.get("box_loss", "smooth_l1"),
                            cfg.get("box_loss_weight", 2.0),
                        )
                    scaler.scale(loss / accumulation).backward()
                total_loss += float(loss.detach()) / accumulation
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            if world > 1:
                reduced = torch.tensor(total_loss, device=device)
                torch.distributed.all_reduce(reduced)
                total_loss = float(reduced) / world
            completed = step + 1
            time_up = max_minutes is not None and time.monotonic() - started >= max_minutes * 60
            stop_requested = (run / "STOP").exists() if rank == 0 else False
            if world > 1:
                stop_flags = torch.tensor([int(time_up), int(stop_requested)], device=device)
                torch.distributed.all_reduce(stop_flags, op=torch.distributed.ReduceOp.MAX)
                time_up, stop_requested = map(bool, stop_flags.tolist())
            stopping = completed == stop or time_up or stop_requested
            if time_up:
                stop_reason = "time_limit"
            elif stop_requested:
                stop_reason = "stop_file"
            if rank == 0:
                log = {
                    "step": completed,
                    "phase": phase["name"],
                    "loss": total_loss,
                    "data_wait_seconds": data_wait,
                    "data_workers": workers,
                    "lr": optimizer.param_groups[-1]["lr"],
                    "elapsed_seconds": time.monotonic() - started,
                    "loss_parts_rank0_last_microbatch": parts,
                    "steps_per_second": (completed - start) / max(time.monotonic() - started, 1e-6),
                    "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if amp else None,
                }
                with (run / "training.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log) + "\n")
                if completed % 10 == 0 or completed == stop:
                    print(json.dumps(log), flush=True)
            validation_due = completed % cfg.get("validation_every", 1000) == 0 or stopping
            improved = False
            checkpoint_due = validation_due or completed % cfg.get("checkpoint_every", 1000) == 0
            if rank == 0 and checkpoint_due:
                state = {
                    "step": completed,
                    "best_score": best,
                    "model": plain.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": cfg,
                    "config_digest": config_digest(cfg),
                    "manifest_fingerprint": inventory["fingerprint"],
                    "model_metadata": plain.metadata(),
                    "initialization": initialization,
                    "source_commit": commit,
                    "world_size": world,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if amp else [],
                }
                state["validation_pending"] = validation_due
                atomic_checkpoint(run / "last.pt", state)
            if rank == 0 and validation_due:
                plain.eval()
                report = evaluate(
                    plain,
                    manifest,
                    "val",
                    device,
                    cfg["image_size"],
                    limit_per_source=cfg.get("validation_limit_per_source", 32),
                    tiled=False,
                )
                report["step"] = completed
                report["purpose"] = (
                    "checkpoint selection; global-view validation preview, not final tiled benchmark"
                )
                write_json(run / "validation" / f"step-{completed:07d}.json", report)
                score = report["selection_score"]
                last_validation = {"step": completed, "score": score, "macro": report["macro"]}
                if score is not None and score > best:
                    best = score
                    improved = True
            if rank == 0 and checkpoint_due:
                state["best_score"] = best
                state["validation_pending"] = False
                atomic_checkpoint(run / "last.pt", state)
                if improved:
                    atomic_checkpoint(run / "best.pt", state)
            if rank == 0:
                log["elapsed_seconds"] = time.monotonic() - started
                write_json(
                    run / "status.json",
                    {
                        **log,
                        "state": "paused" if stopping else "training",
                        "stop_reason": stop_reason if stopping else None,
                        "session_start_step": start,
                        "planned_stop_step": stop,
                        "max_minutes": max_minutes,
                        "latest_validation": last_validation,
                        "checkpoint_step": completed
                        if validation_due or completed % cfg.get("checkpoint_every", 1000) == 0
                        else None,
                        "updated_unix": time.time(),
                    },
                )
            if world > 1 and validation_due:
                torch.distributed.barrier()
            if stopping:
                break
        return {
            "steps": completed,
            "best_validation_score": best,
            "run": str(run),
            "seconds": time.monotonic() - started,
            "stop_reason": stop_reason,
        }
    except Exception as exc:
        if rank == 0:
            write_json(
                run / "failure.json",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "updated_unix": time.time(),
                    "recovery": "Inspect failure and resume the last completed checkpoint in a fresh supervised session.",
                },
            )
        raise
    finally:
        if prefetch is not None:
            prefetch.shutdown(wait=True, cancel_futures=True)
        if world > 1:
            torch.distributed.destroy_process_group()


def _sample_microbatch(key, training, root, cfg, seed, rank):
    step, acc = key
    phase, _ = phase_at(cfg["phases"], step)
    micro = cfg.get("micro_batch", 2)
    rng = np.random.default_rng(seed + step * 100003 + rank * 1009 + acc * 37)
    names = list(phase["sources"])
    weights = np.array([phase["sources"][n] for n in names], float)
    examples = []
    for _ in range(micro):
        source = str(rng.choice(names, p=weights / weights.sum()))
        candidates = training[source]
        record = candidates[int(rng.integers(len(candidates)))]
        examples.append(
            sample(
                record,
                root,
                cfg["image_size"],
                rng,
                cfg.get("crop_probability", 0.5),
                augment=True,
            )
        )
    return examples
