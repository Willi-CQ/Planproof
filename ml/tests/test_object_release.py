import json

import numpy as np
import onnxruntime as ort
import pytest
import torch
from PIL import Image

from planproof_ml.cli import execute, parser
from planproof_ml.export import export
from planproof_ml.inference import detect
from planproof_ml.model import Detector, load_detector
from planproof_ml.schema import sha256


def test_object_release_limits_onnx_and_cli_to_validated_outputs(tmp_path):
    model = Detector(None).eval()
    with torch.no_grad():
        model.mask_head.weight.zero_()
        model.mask_head.bias.fill_(10)
        model.center_head.weight.zero_()
        model.center_head.bias.zero_()
    image = np.full((64, 64, 3), 255, np.uint8)
    full = detect(model, image, size=64, tiled=False)
    assert {item["class"] for item in full["instances"]} == {"wall", "room", "door", "window", "stairs"}
    checkpoint = tmp_path / "training.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_metadata": model.metadata(),
            "config": {"image_size": 64},
            "manifest_fingerprint": "contract-fixture",
            "step": 3,
        },
        checkpoint,
    )
    calibration = dict(
        checkpoint_sha256=sha256(checkpoint),
        manifest_fingerprint="contract-fixture",
        split="val",
        tiled=False,
        image_size=64,
        thresholds={"masks": [0.5] * 5, "centers": [0.3] * 4},
    )
    out = tmp_path / "objects"
    metadata = export(checkpoint, out, 64, calibration=calibration, objects_only=True)
    assert metadata["outputs"] == ("centers", "boxes")
    assert metadata["wall_sha256"] is None and not (out / "wall.onnx").exists()
    assert metadata["intended_classes"] == ["door", "window", "stairs"]
    released_model, released = load_detector(out / "detector.pt")
    assert released["intended_classes"] == metadata["intended_classes"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(out / "detector.onnx"), sess_options=options)
    assert [output.name for output in session.get_outputs()] == ["centers", "boxes"]
    x = np.random.default_rng(42).normal(size=(1, 3, 96, 128)).astype(np.float32)
    with torch.inference_mode():
        reference = released_model(torch.from_numpy(x))
    for key, actual in zip(("centers", "boxes"), session.run(None, {"image": x})):
        np.testing.assert_allclose(reference[key].numpy(), actual, rtol=1e-3, atol=1e-4)
    input_path = tmp_path / "page.png"
    Image.fromarray(image).save(input_path)
    args = parser().parse_args(
        [
            "infer",
            "--checkpoint",
            str(out / "detector.pt"),
            "--calibration",
            str(out / "calibration.json"),
            "--global-view",
            "--device",
            "cpu",
            "--image",
            str(input_path),
            "--out",
            str(tmp_path / "inference"),
        ]
    )
    execute(args)
    result = json.loads((args.out / "detections.json").read_text())
    assert {item["class"] for item in result["instances"]} == {"door", "window", "stairs"}
    assert result["intended_classes"] == ["door", "window", "stairs"]
    args.save_probabilities = True
    with pytest.raises(ValueError, match="Dense probability maps"):
        execute(args)
    with pytest.raises(ValueError, match="empty output directory"):
        export(checkpoint, out, 64, calibration=calibration, objects_only=True)
    with pytest.raises(ValueError, match="require --objects-only"):
        export(out / "detector.pt", tmp_path / "invalid-full", 64)
