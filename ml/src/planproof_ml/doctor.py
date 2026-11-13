from __future__ import annotations
import importlib.metadata
import platform
import shutil
import sys


def doctor(require_gpu=False, smoke=False):
    packages = {}
    for name in (
        "planproof-ml",
        "torch",
        "torchvision",
        "numpy",
        "segmentation-models-pytorch",
        "opencv-python-headless",
        "onnx",
        "onnxruntime",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    import torch

    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "gpus": [],
        "nvidia_smi_available": bool(shutil.which("nvidia-smi")),
        "data_downloaded": False,
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(idx)
            result["gpus"].append(
                {
                    "index": idx,
                    "name": prop.name,
                    "vram_gib": round(prop.total_memory / 1024**3, 2),
                    "capability": list(torch.cuda.get_device_capability(idx)),
                }
            )
    if require_gpu and not result["cuda_available"]:
        raise ValueError(
            "CUDA is unavailable. Attach an NVIDIA GPU and use a CUDA-enabled PyTorch environment."
        )
    if smoke:
        from .model import Detector
        from .losses import detection_loss
        from .dataset import tensor_batch
        from .checks import check_example

        device = "cuda" if result["cuda_available"] else "cpu"
        torch.set_num_threads(2)
        model = Detector(None).to(device).train()
        batch = tensor_batch([check_example(64), check_example(64)], device)
        predictions = model(batch["image"])
        loss, _ = detection_loss(predictions, batch)
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError("Nonfinite gradients in detector probe")
        result["model_probe"] = {
            "device": device,
            "size": 64,
            "batch": 2,
            "loss": float(loss.detach()),
            "finite_gradients": True,
            "weights_downloaded": False,
        }
    return result
