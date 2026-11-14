"""Wall-segmentation inference wrapper.

Mirrors the loading pattern of [plan_yolo_detect.get_yolo_model]: thread-safe
LRU cache keyed by absolute model path, with mtime-based invalidation so a
freshly trained model swapped into ``models/`` is picked up without restart.

Two backends are supported:

* **ONNX** (preferred for the CPU worker) — uses :mod:`onnxruntime` with the
  ``CPUExecutionProvider``. Lighter dependency footprint, no PyTorch needed
  inside the worker container.
* **PyTorch** (.pt state-dict) — falls back to :mod:`segmentation_models_pytorch`
  to rebuild the UNet++/mit_b0 architecture. Used in dev when only the .pt
  file from Kaggle is available.

The Kaggle training notebook (``training/kaggle_wall_segmentation.ipynb``)
exports both ``wall_seg.pt`` and ``wall_seg.onnx``; production drops both
into :mod:`planproof_geometry.ai.models` and the wrapper picks ONNX first.

Public API:
    :func:`get_wall_seg_model` — load (and cache) the model.
    :func:`infer_wall_mask`   — run inference, return a binary mask + meta.
    :data:`DEFAULT_ONNX_PATH` / :data:`DEFAULT_PT_PATH` — bundled-model paths.
    :data:`MODEL_ENV_VAR` — name of the env var that overrides both.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --- Constants ---------------------------------------------------------------

MODEL_ENV_VAR = "PLANPROOF_WALLSEG_MODEL"

DEFAULT_MODEL_DIR = Path(__file__).parent / "models"
DEFAULT_ONNX_PATH = DEFAULT_MODEL_DIR / "wall_seg.onnx"
DEFAULT_PT_PATH = DEFAULT_MODEL_DIR / "wall_seg.pt"
DEFAULT_METADATA_PATH = DEFAULT_MODEL_DIR / "model_metadata.json"

DEFAULT_IMG_SIZE = 512
DEFAULT_THRESHOLD = 0.5
DEFAULT_POSTPROC_CLOSE_KERNEL = 3
DEFAULT_POSTPROC_MIN_OBJECT_PX = 50
# Auto-enable tiled inference once the long edge exceeds this pixel count.
# 2048 keeps standard plans on the fast path while catching scan-grade
# architectural drawings whose wall strokes are ~3-5px at 300dpi.
DEFAULT_TILE_AUTO_THRESHOLD_PX = 2048

# ImageNet normalisation — matches what the Kaggle training notebook uses
# (smp's get_preprocessing_fn for mit_b0 returns these). Stored in
# model_metadata.json under ``preprocessing`` if the notebook updates them.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class WallSegModel:
    """Wraps an ONNX session or a PyTorch module behind a uniform interface."""
    backend: str  # "onnx" or "torch"
    handle: Any   # InferenceSession or torch.nn.Module
    path: str
    img_size: int
    mean: np.ndarray
    std: np.ndarray
    metadata: Dict[str, Any]
    model_hash: str

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Run a forward pass on a normalised CHW batch and return logits.

        ``batch`` shape: (N, 3, H, W), float32, already normalised.
        Returns: (N, 1, H, W) raw logits as numpy.
        """
        if self.backend == "onnx":
            input_name = self.handle.get_inputs()[0].name
            outputs = self.handle.run(None, {input_name: batch.astype(np.float32)})
            return np.asarray(outputs[0])
        # torch backend
        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(batch).float()
            logits = self.handle(tensor)
            return logits.detach().cpu().numpy()


# --- Cache -------------------------------------------------------------------

_model_cache: Dict[str, WallSegModel] = {}
_model_mtime: Dict[str, float] = {}
_cache_lock = threading.Lock()


def _resolve_model_path(
    explicit: Optional[str] = None,
    *,
    prefer: str = "onnx",
) -> Optional[Path]:
    """Resolve the wall-seg model path with override precedence.

    Order:
      1. ``explicit`` argument (caller config)
      2. ``PLANPROOF_WALLSEG_MODEL`` env var
      3. Bundled ONNX (:data:`DEFAULT_ONNX_PATH`) if ``prefer == "onnx"``
      4. Bundled PT  (:data:`DEFAULT_PT_PATH`)
      5. The other-format bundled file (graceful fallback)

    Returns ``None`` if no resolvable file exists on disk.
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get(MODEL_ENV_VAR)
    if env:
        candidates.append(Path(env))
    if prefer == "onnx":
        candidates.extend([DEFAULT_ONNX_PATH, DEFAULT_PT_PATH])
    else:
        candidates.extend([DEFAULT_PT_PATH, DEFAULT_ONNX_PATH])

    for p in candidates:
        if p and p.exists() and p.is_file():
            return p
    return None


def _hash_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()[:16]


def _load_metadata(model_path: Path) -> Dict[str, Any]:
    """Load model_metadata.json from the model's directory if present."""
    candidate = model_path.parent / "model_metadata.json"
    if not candidate.exists():
        return {}
    try:
        with open(candidate, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not parse %s: %s", candidate, exc)
        return {}


def _build_onnx_session(path: Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required to load .onnx wall-seg models. "
            "Install with: pip install planproof-core[wallseg]"
        ) from exc
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), providers=providers)


def _build_torch_module(path: Path, metadata: Dict[str, Any]) -> Any:
    try:
        import torch
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "segmentation-models-pytorch + torch are required for .pt fallback. "
            "Install with: pip install planproof-core[wallseg-dev]"
        ) from exc

    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        # weights_only=False deserialisiert beliebige Python-Objekte → Pickle-RCE
        # bei manipuliertem Checkpoint. NICHT mehr still als Fallback; nur per
        # expliziter ENV (vertrauenswuerdiger Checkpoint) mit lauter Warnung.
        import os

        if os.getenv("PLANPROOF_ALLOW_UNSAFE_TORCH_LOAD", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            logger.warning(
                "Lade Checkpoint %s mit weights_only=False (UNSICHER, per "
                "PLANPROOF_ALLOW_UNSAFE_TORCH_LOAD erlaubt): %s",
                path,
                exc,
            )
            state = torch.load(str(path), map_location="cpu", weights_only=False)
        else:
            raise RuntimeError(
                f"Checkpoint {path} liess sich nicht sicher (weights_only=True) "
                f"laden: {exc}. Wenn der Checkpoint vertrauenswürdig ist, "
                "PLANPROOF_ALLOW_UNSAFE_TORCH_LOAD=true setzen."
            ) from exc
    ckpt_architecture: Optional[str] = None
    if isinstance(state, dict):
        ckpt_architecture = state.get("architecture")
        if "state_dict" in state:
            state = state["state_dict"]

    model_meta = (metadata.get("model") or {}) if metadata else {}
    # Dispatch: V5 multihead has its own wrapper; everything else is plain smp.
    architecture = model_meta.get("architecture") or ckpt_architecture or "binary"
    if architecture in ("multihead_v5", "multihead_v6"):
        return _build_multihead_v5_module(state, metadata)

    encoder_name = model_meta.get("encoder", "mit_b0")
    decoder_name = model_meta.get("decoder", "UnetPlusPlus")
    n_classes = int(model_meta.get("classes", 2))
    decoder_attention = model_meta.get("decoder_attention_type")

    decoder_factory = getattr(smp, decoder_name, smp.UnetPlusPlus)
    kwargs = dict(
        encoder_name=encoder_name,
        encoder_weights=None,  # weights come from state_dict
        in_channels=3,
        classes=n_classes,
    )
    # Only Unet/UnetPlusPlus accept decoder_attention_type; FPN/others would TypeError.
    if decoder_attention and decoder_name in ("Unet", "UnetPlusPlus"):
        kwargs["decoder_attention_type"] = decoder_attention
    model = decoder_factory(**kwargs)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _build_multihead_v5_module(state: Dict[str, Any], metadata: Dict[str, Any]) -> Any:
    """Load a V5/V6 multi-class + DT dual-head checkpoint into an inference wrapper.

    V5 (4 classes: bg/wall/door/window) and V6 (6 classes: + room/stairs)
    both produce per-pixel argmax.  Downstream binary wall-mask infra
    expects a single-channel pseudo-logit that survives a sigmoid.  We emit
    +5 where wall wins the argmax and -5 elsewhere, which sigmoid maps to
    ≈0.993 / 0.007 — so the default 0.5 threshold reproduces the exact
    argmax decision.  The dt_head weights are present in the checkpoint but
    unused at inference.

    Works identically for architecture tags ``multihead_v5`` and ``multihead_v6``.
    """
    import torch
    import torch.nn as nn
    import segmentation_models_pytorch as smp

    model_meta = (metadata.get("model") or {}) if metadata else {}
    labels_meta = (metadata.get("labels_meta") or metadata.get("labels") or {}) if metadata else {}
    encoder_name = model_meta.get("encoder", "mit_b4")
    n_classes = int(model_meta.get("n_classes") or model_meta.get("classes") or 4)
    decoder_attention = model_meta.get("decoder_attention_type", "scse")
    wall_class_idx = int(labels_meta.get("wall", 1))

    class WallSegMultiHeadInference(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=n_classes,
                decoder_attention_type=decoder_attention,
            )
            self.encoder = base.encoder
            self.decoder = base.decoder
            self.seg_head = base.segmentation_head
            self.wall_class_idx = wall_class_idx

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            feats = self.encoder(x)
            # smp ≥ 0.5 takes a feature list; smp ≤ 0.4 uses *feats.
            try:
                dec = self.decoder(feats)
            except TypeError:
                dec = self.decoder(*feats)
            seg_logits = self.seg_head(dec)  # (N, C, H, W)
            wall_is_argmax = (
                seg_logits.argmax(dim=1, keepdim=True) == self.wall_class_idx
            ).float()
            return wall_is_argmax * 10.0 - 5.0  # (N, 1, H, W)

    model = WallSegMultiHeadInference()
    # dt_head.* keys are expected-missing; strict=False keeps it a clean load.
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def get_wall_seg_model(
    model_path: Optional[str] = None,
    *,
    prefer: str = "onnx",
) -> Optional[WallSegModel]:
    """Load (or fetch from cache) a wall-segmentation model.

    Thread-safe with mtime-based cache invalidation.

    Returns ``None`` if no model is found at any candidate path; callers
    (the WallSegmentationStep) treat this as a soft skip rather than an
    error so the rest of the V2 pipeline still runs.
    """
    resolved = _resolve_model_path(model_path, prefer=prefer)
    if resolved is None:
        logger.info(
            "No wall-seg model found (looked at config, %s, %s, %s)",
            MODEL_ENV_VAR, DEFAULT_ONNX_PATH, DEFAULT_PT_PATH,
        )
        return None

    cache_key = str(resolved.resolve())
    with _cache_lock:
        cached_mtime = _model_mtime.get(cache_key)
        current_mtime = resolved.stat().st_mtime
        if cache_key in _model_cache and cached_mtime == current_mtime:
            return _model_cache[cache_key]
        if cache_key in _model_cache:
            logger.info("Wall-seg model file changed, reloading: %s", resolved)
            _model_cache.pop(cache_key, None)

        metadata = _load_metadata(resolved)
        preprocessing = metadata.get("preprocessing") or {}
        mean = np.asarray(preprocessing.get("mean", _IMAGENET_MEAN.tolist()), dtype=np.float32)
        std = np.asarray(preprocessing.get("std", _IMAGENET_STD.tolist()), dtype=np.float32)
        img_size = int((metadata.get("model") or {}).get("img_size", DEFAULT_IMG_SIZE))

        if resolved.suffix.lower() == ".onnx":
            handle = _build_onnx_session(resolved)
            backend = "onnx"
        else:
            handle = _build_torch_module(resolved, metadata)
            backend = "torch"

        model = WallSegModel(
            backend=backend,
            handle=handle,
            path=cache_key,
            img_size=img_size,
            mean=mean,
            std=std,
            metadata=metadata,
            model_hash=_hash_file(resolved),
        )
        _model_cache[cache_key] = model
        _model_mtime[cache_key] = current_mtime
        logger.info(
            "Loaded wall-seg model: backend=%s img_size=%d hash=%s path=%s",
            backend, img_size, model.model_hash, cache_key,
        )
        return model


# --- Inference ---------------------------------------------------------------

def _preprocess(
    image_bgr: np.ndarray,
    *,
    img_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Resize + normalise, return (CHW float32 batch, original (H, W))."""
    import cv2  # local import keeps module import lightweight

    if image_bgr.ndim == 2:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    elif image_bgr.shape[2] == 4:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = rgb.shape[:2]
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - mean) / std
    chw = np.transpose(normalized, (2, 0, 1))[None, ...]  # (1, 3, H, W)
    return chw, (orig_h, orig_w)


def _logits_to_wall_probs(logits: np.ndarray) -> np.ndarray:
    """Extract the wall channel from raw logits and return sigmoid probs (2D)."""
    # (N, 2, H, W) → channel 1 is wall; (N, 1, H, W) or (N, H, W) → channel 0.
    if logits.ndim == 4 and logits.shape[1] == 2:
        wall_logits = logits[0, 1]
    elif logits.ndim == 4:
        wall_logits = logits[0, 0]
    elif logits.ndim == 3:
        wall_logits = logits[0]
    else:
        wall_logits = logits
    return 1.0 / (1.0 + np.exp(-wall_logits))


def _predict_probs_fullres(
    image_bgr: np.ndarray,
    model: "WallSegModel",
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Single forward pass, returning sigmoid probs upsampled to the original size."""
    import cv2

    batch, original_size = _preprocess(
        image_bgr, img_size=model.img_size, mean=model.mean, std=model.std,
    )
    logits = model.predict(batch)
    probs = _logits_to_wall_probs(logits)
    orig_h, orig_w = original_size
    if probs.shape != original_size:
        probs = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return probs.astype(np.float32), original_size


def _tta_probs_fullres(
    image_bgr: np.ndarray,
    model: "WallSegModel",
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """4-way TTA: original + hflip + vflip + rot90, averaged at original resolution.

    Each transform is applied to the image, inference runs, then the *inverse*
    transform is applied to the prob map so every contribution is in the same
    orientation before averaging. Averaging sigmoid probs (not logits) is
    robust enough in practice and keeps the pipeline backend-agnostic.
    """
    original_size = image_bgr.shape[:2]

    # (forward, inverse) transform pairs.
    # Rot90: rotating image CCW by k=1, so invert with k=-1 on the probs.
    transforms = (
        (lambda im: im, lambda pr: pr),
        (lambda im: np.ascontiguousarray(im[:, ::-1]), lambda pr: pr[:, ::-1]),
        (lambda im: np.ascontiguousarray(im[::-1, :]), lambda pr: pr[::-1, :]),
        (lambda im: np.ascontiguousarray(np.rot90(im, k=1)),
         lambda pr: np.rot90(pr, k=-1)),
    )

    acc = None
    for fwd, inv in transforms:
        probs, _ = _predict_probs_fullres(fwd(image_bgr), model)
        probs_back = np.ascontiguousarray(inv(probs))
        if probs_back.shape != original_size:
            # Shouldn't happen (our inverses are shape-preserving), but guard
            # against arithmetic mismatches at odd odd sizes.
            import cv2
            probs_back = cv2.resize(
                probs_back, (original_size[1], original_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        acc = probs_back if acc is None else acc + probs_back

    return (acc / float(len(transforms))).astype(np.float32), original_size


def _resolve_threshold(
    threshold: Optional[float],
    metrics: Dict[str, Any],
) -> Tuple[float, str]:
    """Pick the threshold (explicit > metadata.best_threshold > default)."""
    if threshold is not None:
        return float(threshold), "explicit"
    meta_best = metrics.get("best_threshold")
    if isinstance(meta_best, (int, float)) and 0.0 < float(meta_best) < 1.0:
        return float(meta_best), "metadata"
    return DEFAULT_THRESHOLD, "default"


def _probs_to_mask(
    probs: np.ndarray,
    *,
    threshold: float,
    postproc: bool,
    postproc_close_kernel: int,
    postproc_min_object_px: int,
) -> np.ndarray:
    mask = (probs >= threshold).astype(np.uint8)
    if postproc:
        mask = _apply_postproc(
            mask,
            close_kernel=postproc_close_kernel,
            min_object_px=postproc_min_object_px,
        )
    return mask


def _apply_postproc(
    mask: np.ndarray,
    *,
    close_kernel: int = DEFAULT_POSTPROC_CLOSE_KERNEL,
    min_object_px: int = DEFAULT_POSTPROC_MIN_OBJECT_PX,
) -> np.ndarray:
    """Morphological close + small-object removal on a binary mask.

    Closes 1-2px holes inside wall runs (MORPH_CLOSE), then drops isolated
    specks below ``min_object_px`` pixels — these are typically text/noise
    the model half-triggered on.
    """
    import cv2

    if mask.size == 0 or not mask.any():
        return mask

    if close_kernel >= 2:
        k = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if min_object_px and min_object_px > 1:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num > 1:
            keep = np.zeros(num, dtype=bool)
            keep[0] = True  # background
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] >= min_object_px:
                    keep[i] = True
            mask = keep[labels].astype(np.uint8)

    return mask


def _build_meta(
    *,
    model: "WallSegModel",
    original_size: Tuple[int, int],
    threshold: float,
    threshold_source: str,
    postproc: bool,
    mask: np.ndarray,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics = (model.metadata.get("metrics") or {}) if model.metadata else {}
    meta: Dict[str, Any] = {
        "backend": model.backend,
        "model_path": model.path,
        "model_hash": model.model_hash,
        "img_size": model.img_size,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "postproc": postproc,
        "image_size": {"width": int(original_size[1]), "height": int(original_size[0])},
        "wall_pixel_ratio": float(mask.mean()) if mask.size else 0.0,
    }
    if extra:
        meta.update(extra)
    if metrics:
        # Carry forward the most useful training metrics so they appear in
        # the bundle's meta.json — helps verify which model produced the run.
        for key in ("val_iou", "val_dice", "val_precision", "val_recall"):
            if key in metrics:
                meta[f"train_{key}"] = float(metrics[key])
    return meta


def infer_wall_mask(
    image_bgr: np.ndarray,
    model: WallSegModel,
    *,
    threshold: Optional[float] = None,
    tta: bool = False,
    postproc: bool = True,
    postproc_close_kernel: int = DEFAULT_POSTPROC_CLOSE_KERNEL,
    postproc_min_object_px: int = DEFAULT_POSTPROC_MIN_OBJECT_PX,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run wall-segmentation on a single image.

    Args:
        image_bgr: HxW or HxWx3/4 numpy array (BGR per OpenCV convention).
        model: A loaded :class:`WallSegModel`.
        threshold: Probability cut-off for the binary mask. If ``None``,
            falls back to ``metrics.best_threshold`` in the model metadata
            (set by the v2 training notebook's threshold sweep), otherwise
            to :data:`DEFAULT_THRESHOLD` (0.5).
        tta: If True, run 4-way test-time augmentation (original + hflip +
            vflip + rot90) and average sigmoid probs before thresholding.
            ~4x slower but more stable on borderline walls.
        postproc: If True (default), run morphological close + small-object
            removal on the final mask before returning it.
        postproc_close_kernel: Kernel size for MORPH_CLOSE. 0 disables.
        postproc_min_object_px: Drop connected components smaller than this.

    Returns:
        ``(mask, meta)`` where ``mask`` is a uint8 array of shape (H, W) with
        values 0/1 at the *original* image resolution, and ``meta`` carries
        provenance info (model hash, threshold, training IoU, ...).
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        raise ValueError("infer_wall_mask: image_bgr must be a numpy array")
    if image_bgr.size == 0:
        raise ValueError("infer_wall_mask: image is empty")

    metrics = (model.metadata.get("metrics") or {}) if model.metadata else {}
    threshold_val, threshold_source = _resolve_threshold(threshold, metrics)

    if tta:
        probs, original_size = _tta_probs_fullres(image_bgr, model)
    else:
        probs, original_size = _predict_probs_fullres(image_bgr, model)

    mask = _probs_to_mask(
        probs,
        threshold=threshold_val,
        postproc=postproc,
        postproc_close_kernel=postproc_close_kernel,
        postproc_min_object_px=postproc_min_object_px,
    )

    return mask, _build_meta(
        model=model,
        original_size=original_size,
        threshold=threshold_val,
        threshold_source=threshold_source,
        postproc=postproc,
        mask=mask,
        extra={"tta": tta, "inference_mode": "tta" if tta else "single"},
    )


def infer_wall_mask_tiled(
    image_bgr: np.ndarray,
    model: WallSegModel,
    *,
    threshold: Optional[float] = None,
    overlap: float = 0.5,
    postproc: bool = True,
    postproc_close_kernel: int = DEFAULT_POSTPROC_CLOSE_KERNEL,
    postproc_min_object_px: int = DEFAULT_POSTPROC_MIN_OBJECT_PX,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Tile-based inference at native resolution for very large plans.

    The default :func:`infer_wall_mask` resizes the entire image to
    ``model.img_size`` (e.g. 512x512) before inference. For a 4000x3000
    plan that collapses wall strokes to <2px, below the model's effective
    receptive field. This function instead slides a ``(img_size, img_size)``
    window over the original image with ``overlap`` fractional overlap,
    averages sigmoid probabilities across overlapping tiles, then thresholds.

    Args mirror :func:`infer_wall_mask`. ``overlap`` is the fraction by which
    adjacent tiles overlap (0.5 → stride = img_size / 2).
    """
    import cv2

    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        raise ValueError("infer_wall_mask_tiled: image_bgr must be a numpy array")
    if image_bgr.size == 0:
        raise ValueError("infer_wall_mask_tiled: image is empty")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    metrics = (model.metadata.get("metrics") or {}) if model.metadata else {}
    threshold_val, threshold_source = _resolve_threshold(threshold, metrics)

    tile = int(model.img_size)
    stride = max(1, int(round(tile * (1.0 - overlap))))

    h, w = image_bgr.shape[:2]
    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)
    # Pad so every pixel is fully covered by at least one tile whose
    # bottom-right corner doesn't run off the image.
    if h + pad_h > tile:
        extra_h = (stride - ((h + pad_h - tile) % stride)) % stride
        pad_h += extra_h
    if w + pad_w > tile:
        extra_w = (stride - ((w + pad_w - tile) % stride)) % stride
        pad_w += extra_w
    padded = cv2.copyMakeBorder(
        image_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101,
    )
    ph, pw = padded.shape[:2]

    prob_acc = np.zeros((ph, pw), dtype=np.float32)
    count_acc = np.zeros((ph, pw), dtype=np.float32)

    y_positions = list(range(0, ph - tile + 1, stride))
    x_positions = list(range(0, pw - tile + 1, stride))
    # Ensure the last row/col always ends exactly at the padded edge.
    if y_positions[-1] != ph - tile:
        y_positions.append(ph - tile)
    if x_positions[-1] != pw - tile:
        x_positions.append(pw - tile)

    n_tiles = 0
    for y in y_positions:
        for x in x_positions:
            tile_img = padded[y:y + tile, x:x + tile]
            # Tile is already exactly (tile, tile) so preprocess is a no-op
            # on spatial dims (resize same → same).
            probs, _ = _predict_probs_fullres(tile_img, model)
            prob_acc[y:y + tile, x:x + tile] += probs
            count_acc[y:y + tile, x:x + tile] += 1.0
            n_tiles += 1

    avg_probs = prob_acc / np.maximum(count_acc, 1e-6)
    avg_probs = avg_probs[:h, :w]

    mask = _probs_to_mask(
        avg_probs,
        threshold=threshold_val,
        postproc=postproc,
        postproc_close_kernel=postproc_close_kernel,
        postproc_min_object_px=postproc_min_object_px,
    )

    return mask, _build_meta(
        model=model,
        original_size=(h, w),
        threshold=threshold_val,
        threshold_source=threshold_source,
        postproc=postproc,
        mask=mask,
        extra={
            "tta": False,
            "inference_mode": "tiled",
            "tile_size": tile,
            "tile_stride": stride,
            "tile_overlap": overlap,
            "tile_count": n_tiles,
        },
    )


__all__ = [
    "MODEL_ENV_VAR",
    "DEFAULT_ONNX_PATH",
    "DEFAULT_PT_PATH",
    "DEFAULT_METADATA_PATH",
    "DEFAULT_IMG_SIZE",
    "DEFAULT_THRESHOLD",
    "DEFAULT_TILE_AUTO_THRESHOLD_PX",
    "WallSegModel",
    "get_wall_seg_model",
    "infer_wall_mask",
    "infer_wall_mask_tiled",
]
