import json

import numpy as np
import pytest
import torch
import yaml

from planproof_ml.checks import check_corpus, check_example
from planproof_ml.dataset import tensor_batch
from planproof_ml.losses import detection_loss
from planproof_ml.model import Detector, load_detector
from planproof_ml.train import train, preflight
from planproof_ml.config import load_config
from planproof_ml.manifest import load_manifest
from planproof_ml.export import export
from planproof_ml.inference import detect
from planproof_ml.supervision import status


def config_file(tmp_path):
    cfg = {
        "name": "test",
        "seed": 7,
        "image_size": 64,
        "effective_batch": 2,
        "micro_batch": 2,
        "encoder_weights": None,
        "validation_every": 1,
        "checkpoint_every": 1,
        "validation_limit_per_source": 1,
        "crop_probability": 0.0,
        "allow_simulated": True,
        "phases": [{"name": "test", "steps": 3, "sources": {"fixture": 1.0}}],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_missing_class_coverage_zeroes_mask_gradient_and_loss_is_finite():
    example = check_example()
    example["valid"][:] = 0
    example["boundary_valid"][:] = 0
    example["center_valid"][:] = 0
    example["types"][:] = 255
    batch = tensor_batch([example, example], "cpu")
    pred = {
        name: torch.randn(*shape, requires_grad=True)
        for name, shape in {
            "masks": (2, 5, 64, 64),
            "boundary": (2, 1, 64, 64),
            "centers": (2, 4, 64, 64),
            "boxes": (2, 12, 64, 64),
            "types": (2, 10, 64, 64),
        }.items()
    }
    total, parts = detection_loss(pred, batch)
    total.backward()
    assert float(total) == 0.0
    assert pred["masks"].grad.abs().sum() == 0
    assert all(np.isfinite(v) for v in parts.values())


def test_weak_room_box_supervises_category_without_dense_room_masks():
    example = check_example()
    example["valid"][:] = 0
    example["boundary_valid"][:] = 0
    example["center_valid"][:] = 0
    example["types"][:] = 255
    example["weak_types"] = np.array([[0, 5, 5, 20, 20]], np.float32)
    batch = tensor_batch([example, example], "cpu")
    model = Detector(None)
    loss, _ = detection_loss(model(batch["image"]), batch)
    loss.backward()
    assert loss > 0
    assert model.type_head[-1].weight.grad.abs().sum() > 0
    assert model.mask_head.weight.grad.abs().sum() == 0


def test_preflight_refuses_missing_sources_and_fixture_production_mix(tmp_path):
    manifest = check_corpus(tmp_path / "data")
    cfg = config_file(tmp_path)
    config = load_config(cfg)
    records = load_manifest(manifest)
    config["allow_simulated"] = False
    with pytest.raises(ValueError, match="diagnostic"):
        preflight(config, records)
    config["allow_simulated"] = True
    config["phases"][0]["sources"] = {"floorplancad": 1.0}
    with pytest.raises(ValueError, match="No silent"):
        preflight(config, records)


def test_train_resume_matches_continuous_run_and_refuses_dataset_changes(tmp_path):
    manifest = check_corpus(tmp_path / "data")
    cfg = config_file(tmp_path)
    train(cfg, manifest, tmp_path / "continuous", "cpu", max_steps=2)
    train(cfg, manifest, tmp_path / "resumed", "cpu", max_steps=1)
    train(cfg, manifest, tmp_path / "resumed", "cpu", resume=tmp_path / "resumed/last.pt", max_steps=2)
    _, continuous = load_detector(tmp_path / "continuous/last.pt")
    model, resumed = load_detector(tmp_path / "resumed/last.pt")
    assert resumed["step"] == 2
    for name, tensor in continuous["model"].items():
        torch.testing.assert_close(tensor, resumed["model"][name], rtol=0, atol=0)
    lines = manifest.read_text().splitlines()
    row = json.loads(lines[0])
    row["source_key"] = "different-source-version"
    lines[0] = json.dumps(row)
    manifest.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="Dataset changed"):
        train(cfg, manifest, tmp_path / "resumed", "cpu", resume=tmp_path / "resumed/last.pt", max_steps=3)


def test_session_time_cap_checkpoints_and_resumes_without_changing_recipe(tmp_path):
    manifest = check_corpus(tmp_path / "data")
    cfg = config_file(tmp_path)
    run = tmp_path / "bounded"
    result = train(cfg, manifest, run, "cpu", max_minutes=1e-9)
    assert result["steps"] == 1 and result["stop_reason"] == "time_limit"
    health = status(run, hourly_rate=2.0)
    assert health["state"] == "paused" and health["last_checkpoint_exists"]
    assert health["session_compute_cost_estimate_eur"] > 0
    assert health["latest_validation"]["step"] == 1
    (run / "STOP").touch()
    result = train(cfg, manifest, run, "cpu", resume=run / "last.pt")
    assert result["steps"] == 2 and result["stop_reason"] == "stop_file"
    for minutes in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="positive"):
            train(cfg, manifest, tmp_path / "invalid", "cpu", max_minutes=minutes)


def test_transfer_initialization_allows_new_data_and_resets_optimizer(tmp_path):
    manifest = check_corpus(tmp_path / "data")
    cfg = config_file(tmp_path)
    train(cfg, manifest, tmp_path / "base", "cpu", max_steps=2)
    source = tmp_path / "base/last.pt"
    before = torch.load(source, weights_only=True)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["source_key"] = "new-dataset-version"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = yaml.safe_load(cfg.read_text())
    config.update(seed=71, name="transfer", head_lr=0.0, encoder_lr=0.0)
    cfg.write_text(yaml.safe_dump(config))
    result = train(cfg, manifest, tmp_path / "transfer", "cpu", max_steps=1, init_weights=source)
    after = torch.load(tmp_path / "transfer/last.pt", weights_only=True)
    assert result["steps"] == 1 and after["step"] == 1
    assert after["manifest_fingerprint"] != before["manifest_fingerprint"]
    assert after["initialization"]["source_step"] == 2
    assert after["initialization"]["optimizer_reset"]
    for name in ("encoder.conv1.weight", "mask_head.weight"):
        torch.testing.assert_close(after["model"][name], before["model"][name], rtol=0, atol=0)
    assert all(float(state["step"]) == 1 for state in after["optimizer"]["state"].values())
    with pytest.raises(ValueError, match="not both"):
        train(cfg, manifest, tmp_path / "invalid", "cpu", resume=source, init_weights=source)
    before["model_metadata"]["schema_digest"] = "incompatible"
    bad = tmp_path / "bad.pt"
    torch.save(before, bad)
    with pytest.raises(ValueError, match="incompatible"):
        train(cfg, manifest, tmp_path / "bad-run", "cpu", init_weights=bad)


def test_onnx_export_parity_and_original_coordinate_detection(tmp_path):
    manifest = check_corpus(tmp_path / "data")
    cfg = config_file(tmp_path)
    train(cfg, manifest, tmp_path / "run", "cpu", max_steps=1)
    report = export(tmp_path / "run/last.pt", tmp_path / "export", 64)
    assert set(report["onnx_parity_max_abs_error"]) == {"masks", "boundary", "centers", "boxes", "types"}
    assert (tmp_path / "export/wall.onnx").is_file()
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(tmp_path / "export/detector.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    outputs = session.run(None, {"image": np.zeros((1, 3, 96, 128), np.float32)})
    assert outputs[0].shape == (1, 5, 96, 128)
    model, _ = load_detector(tmp_path / "run/last.pt")
    result = detect(
        model,
        np.full((79, 131, 3), 255, np.uint8),
        size=64,
        tiled=True,
        document_id="original-page",
        page_index=2,
    )
    assert result["image_size"] == [131, 79]
    assert result["document_id"] == "original-page" and result["page_index"] == 2
    for item in result["instances"]:
        assert 0 <= item["bbox"][0] < item["bbox"][2] <= 131
        assert 0 <= item["bbox"][1] < item["bbox"][3] <= 79
