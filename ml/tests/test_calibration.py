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
