from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from .model import ExportModel, OUTPUTS, load_detector
from .schema import sha256, write_json
from .storage import ensure_artifact_directory


class WallOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image)["masks"][:, 0:1]


def export(checkpoint: Path, output: Path, size=1024, thresholds=None, calibration=None):
    ensure_artifact_directory(output)
    import onnx
    import onnxruntime as ort

    output.mkdir(parents=True, exist_ok=True)
    model, data = load_detector(checkpoint, "cpu")
    checkpoint_hash = sha256(checkpoint)
    if calibration:
        if (
            calibration.get("checkpoint_sha256") != checkpoint_hash
            or calibration.get("manifest_fingerprint") != data["manifest_fingerprint"]
            or calibration.get("split") != "val"
            or calibration.get("image_size", data["config"]["image_size"]) != size
        ):
            raise ValueError("Export calibration must match this checkpoint, dataset and image size")
        if thresholds is not None and thresholds != calibration["thresholds"]:
            raise ValueError("Export thresholds differ from calibration")
        thresholds = calibration["thresholds"]
    example = torch.randn(1, 3, size, size, generator=torch.Generator().manual_seed(2026))
    wrapper = ExportModel(model).eval()
    target = output / "detector.onnx"
    dynamic = {"image": {0: "batch", 2: "height", 3: "width"}}
    dynamic.update({k: {0: "batch", 2: "height", 3: "width"} for k in OUTPUTS})
    torch.onnx.export(
        wrapper,
        example,
        str(target),
        input_names=["image"],
        output_names=list(OUTPUTS),
        dynamic_axes=dynamic,
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(target)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(target), sess_options=options, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        reference = wrapper(example)
    actual = session.run(None, {"image": example.numpy()})
    errors = {}
    for name, expected, predicted in zip(OUTPUTS, reference, actual):
        error = float(np.max(np.abs(expected.numpy() - predicted)))
        errors[name] = error
        if not np.allclose(expected.numpy(), predicted, rtol=1e-3, atol=1e-4):
            raise ValueError(f"ONNX parity failed for {name}: {error}")
    wall = output / "wall.onnx"
    torch.onnx.export(
        WallOnly(model).eval(),
        example,
        str(wall),
        input_names=["image"],
        output_names=["wall_logits"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "wall_logits": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
        dynamo=False,
    )
    wall_session = ort.InferenceSession(str(wall), sess_options=options, providers=["CPUExecutionProvider"])
    wall_result = wall_session.run(None, {"image": example.numpy()})[0]
    if not np.allclose(reference[0][:, 0:1].numpy(), wall_result, rtol=1e-3, atol=1e-4):
        raise ValueError("Wall-only ONNX parity failed")
    inference = output / "detector.pt"
    release = {
        key: value for key, value in data.items() if key not in {"optimizer", "scaler", "rng_cpu", "rng_cuda"}
    }
    release.update(artifact_type="inference", source_checkpoint_sha256=checkpoint_hash)
    temporary = inference.with_suffix(".tmp")
    torch.save(release, temporary)
    temporary.replace(inference)
    released = torch.load(inference, map_location="cpu", weights_only=True)
    if any(not torch.equal(value, released["model"][key]) for key, value in data["model"].items()):
        raise ValueError("Inference checkpoint weights differ from training checkpoint")
    inference_hash = sha256(inference)
    if calibration:
        write_json(
            output / "calibration.json",
            {
                **calibration,
                "source_checkpoint_sha256": checkpoint_hash,
                "checkpoint_sha256": inference_hash,
                "repackaged_without_optimizer": True,
            },
        )
    metadata = {
        **model.metadata(),
        "checkpoint_sha256": checkpoint_hash,
        "inference_checkpoint": "detector.pt",
        "inference_checkpoint_sha256": inference_hash,
        "training_source_commit": data.get("source_commit", "unknown"),
        "inference_tiled": calibration.get("tiled", True) if calibration else True,
        "calibration_file": "calibration.json" if calibration else None,
        "calibration_sha256": sha256(output / "calibration.json") if calibration else None,
        "model_sha256": sha256(target),
        "wall_sha256": sha256(wall),
        "manifest_fingerprint": data["manifest_fingerprint"],
        "training_step": data["step"],
        "canvas_size": size,
        "thresholds": thresholds,
        "onnx_parity_max_abs_error": errors,
        "input": {
            "name": "image",
            "dtype": "float32",
            "spatial_multiple": 32,
            "recommended_size": [size, size],
            "resize": "aspect-preserving letterbox; retain the Frame transform",
            "channels": "RGB",
            "range_before_normalization": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "output_geometry": "ONNX tensors use input-canvas coordinates; the postprocessor restores original pixel edges",
        "output_activations": {
            "masks": "sigmoid",
            "boundary": "sigmoid",
            "centers": "sigmoid",
            "types": "softmax",
            "boxes": "sigmoid width/height; tanh offsets times 0.5",
        },
        "wall_output": "one-channel logits; sigmoid only once",
    }
    write_json(output / "metadata.json", metadata)
    return metadata
