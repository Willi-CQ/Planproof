import pytest

from planproof_ml.checks import check_corpus
from planproof_ml.evaluate import calibrate, evaluate
from planproof_ml.model import Detector


def test_cached_task_calibration_confirms_same_production_metrics(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("planproof_ml.evaluate")
    manifest = check_corpus(tmp_path / "data")
    model = Detector(None).eval()
    original = module.detect
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "detect", counted)
    result = calibrate(model, manifest, "cpu", 64, limit_per_source=1, tiled=False)
    assert len(calls) == 2
    assert result["cached_inference_passes"] == 1
    assert result["confirmation_inference_passes"] == 1
    assert len(result["candidates"]) == 14
    confirmation = evaluate(model, manifest, "val", "cpu", 64, result["thresholds"], 1, tiled=False)
    assert result["validation_score"] == pytest.approx(confirmation["selection_score"])
    assert result["validation_macro"] == confirmation["macro"]
    assert result["validation_report"]["sources"] == confirmation["sources"]
    assert result["image_size"] == 64
    assert result["cohort_fingerprints"] == confirmation["cohort_fingerprints"]
    for component, metric in (
        ("wall", "wall_iou"),
        ("room", "room_instance_f1_iou50"),
        ("objects", "object_map50"),
    ):
        best = max(c["score"] for c in result["candidates"] if c["component"] == component)
        assert confirmation["macro"][metric] == pytest.approx(best)


def test_cli_refuses_calibration_against_another_dataset(tmp_path, monkeypatch):
    from argparse import Namespace
    from planproof_ml.cli import execute
    import planproof_ml.model as models

    manifest = check_corpus(tmp_path / "data")
    monkeypatch.setattr(
        models,
        "load_detector",
        lambda *args: (object(), {"config": {"image_size": 64}, "manifest_fingerprint": "different"}),
    )
    (tmp_path / "unused.pt").write_bytes(b"mock checkpoint")
    args = Namespace(
        command="calibrate",
        checkpoint=tmp_path / "unused.pt",
        manifest=manifest,
        out=tmp_path / "calibration.json",
        device="cpu",
        limit_per_source=1,
    )
    with pytest.raises(ValueError, match="manifest must match"):
        execute(args)
    assert not args.out.exists()


def test_cli_refuses_global_calibration_in_tiled_inference(tmp_path, monkeypatch):
    from argparse import Namespace
    from planproof_ml.cli import execute
    from planproof_ml.schema import write_json, sha256
    import planproof_ml.model as models

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"mock checkpoint")
    calibration = tmp_path / "calibration.json"
    write_json(
        calibration,
        dict(
            checkpoint_sha256=sha256(checkpoint),
            manifest_fingerprint="same",
            split="val",
            tiled=False,
            thresholds={},
        ),
    )
    monkeypatch.setattr(
        models,
        "load_detector",
        lambda *args: (object(), {"config": {"image_size": 64}, "manifest_fingerprint": "same"}),
    )
    args = Namespace(
        command="infer",
        checkpoint=checkpoint,
        calibration=calibration,
        out=tmp_path / "output",
        device="cpu",
        global_view=False,
    )
    with pytest.raises(ValueError, match="inference mode differs"):
        execute(args)


def test_component_area_sweep_matches_production_and_checks_binding(tmp_path):
    import importlib.util
    from pathlib import Path
    import torch
    from planproof_ml.schema import sha256, write_json

    path = Path(__file__).resolve().parents[1] / "scripts/calibrate_component_area.py"
    spec = importlib.util.spec_from_file_location("area_calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = check_corpus(tmp_path / "data")
    model = Detector(None).eval()
    calibration = calibrate(model, manifest, "cpu", 64, limit_per_source=1, tiled=False)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_metadata": model.metadata(),
            "config": {"image_size": 64},
            "manifest_fingerprint": calibration["manifest_fingerprint"],
            "step": 9,
        },
        checkpoint,
    )
    calibration["checkpoint_sha256"] = sha256(checkpoint)
    source = tmp_path / "calibration.json"
    write_json(source, calibration)
    result = module.calibrate_area(checkpoint, manifest, source, [4, 32], 1, "cpu")
    ordinary = evaluate(model, manifest, "val", "cpu", 64, result["thresholds"], 1, tiled=False)
    assert result["validation_report"] == ordinary
    assert result["validation_score"] == max(
        report["selection_score"] for report in result["component_area_candidates"].values()
    )
    assert result["component_area_forward_passes"] == 1
    assert result["component_area_confirmation_passes"] == 1
    assert result["checkpoint_step"] == 9
    with pytest.raises(ValueError, match="positive integers"):
        module.calibrate_area(checkpoint, manifest, source, [0], 1, "cpu")
    calibration["tiled"] = True
    write_json(source, calibration)
    with pytest.raises(ValueError, match="matching global-view"):
        module.calibrate_area(checkpoint, manifest, source, [4], 1, "cpu")
