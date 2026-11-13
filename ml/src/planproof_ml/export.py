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


def export(checkpoint: Path, output: Path, size=1024, thresholds=None):
    ensure_artifact_directory(output)
    import onnx
    import onnxruntime as ort

    output.mkdir(parents=True, exist_ok=True)
    model, data = load_detector(checkpoint, "cpu")
    example = torch.zeros(1, 3, size, size)
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
    metadata = {
        **model.metadata(),
        "checkpoint_sha256": sha256(checkpoint),
        "model_sha256": sha256(target),
        "wall_sha256": sha256(wall),
        "manifest_fingerprint": data["manifest_fingerprint"],
        "training_step": data["step"],
        "canvas_size": size,
        "thresholds": thresholds,
        "onnx_parity_max_abs_error": errors,
        "input": {
            "name": "image",
            "channels": "RGB",
            "range_before_normalization": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "output_geometry": "original image pixel edges; use inference postprocessor",
        "wall_output": "one-channel logits; sigmoid only once",
    }
    write_json(output / "metadata.json", metadata)
    return metadata
