import copy
import json
import zipfile

import numpy as np
import pytest

from planproof_ml import acquire
from planproof_ml.adapters import geometry_hash
from planproof_ml.checks import check_corpus
from planproof_ml.compare import compare
from planproof_ml.inspect_data import inspect_data
from planproof_ml.manifest import load_manifest, regroup
from planproof_ml.metrics import Evaluator, SourceMetrics
from planproof_ml.schema import write_json
from planproof_ml.storage import ensure_artifact_directory
from planproof_ml.targets import encode
from planproof_ml.audit import audit_manifest
from planproof_ml.supervision import status


def test_generated_storage_refuses_source_tree_but_accepts_ignored_directory():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="outside"):
        ensure_artifact_directory(root / "ml" / "training-data")
    assert ensure_artifact_directory(root / ".artifacts" / "training").is_absolute()


def test_deep_audit_detects_changed_pixels(tmp_path):
    from PIL import Image

    manifest = check_corpus(tmp_path / "data")
    assert audit_manifest(manifest, deep=True)["image_hashes_checked"] == 4
    assert audit_manifest(manifest, deep=True, near=True, workers=2) == audit_manifest(
        manifest, deep=True, near=True
    )
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(manifest.parent / "fixture-0.png")
    with pytest.raises(ValueError, match="Image hash mismatch"):
        audit_manifest(manifest, deep=True)


def test_supervision_flags_stale_failed_and_plateauing_runs(tmp_path):
    write_json(
        tmp_path / "status.json",
        {
            "updated_unix": 1,
            "state": "training",
            "loss": 1.0,
            "planned_stop_step": 2000,
            "step": 400,
            "steps_per_second": 2,
            "elapsed_seconds": 200,
        },
    )
    write_json(tmp_path / "failure.json", {"updated_unix": 2, "error": "diagnostic failure"})
    for step in (100, 200, 300, 400):
        write_json(
            tmp_path / "validation" / f"step-{step:07d}.json",
            {
                "step": step,
                "selection_score": 0.6,
            },
        )
    report = status(tmp_path, 3.6)
    assert len(report["warnings"]) == 3
    assert report["remaining_step_time_estimate_seconds"] == 800
    assert report["session_compute_cost_estimate_eur"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="nonnegative"):
        status(tmp_path, -1)


def test_geometry_dedup_is_invariant_to_d4_translation_scale_and_vertex_order():
    feature = {"class": "room", "room_type": "kitchen", "outer": [[0, 0], [9, 0], [9, 4], [0, 3]]}
    transformed = copy.deepcopy(feature)
    points = np.asarray(feature["outer"])
    transformed["outer"] = (points[:, ::-1] * [-3, 3] + [100, 70])[::-1].tolist()
    assert geometry_hash([feature]) == geometry_hash([transformed])


def test_inspection_renders_ground_truth_and_regroup_preserves_holdout(tmp_path):
    manifest = check_corpus(tmp_path / "data")
    report = inspect_data(manifest, tmp_path / "inspection", 2)
    assert report["rendered"] == 2
    assert len(list((tmp_path / "inspection").glob("*.png"))) == 2
    mapping = tmp_path / "groups.json"
    write_json(mapping, {"fixture:0": "duplicate", "fixture:3": "duplicate"})
    result = regroup(manifest, mapping)
    records = {r["id"]: r for r in load_manifest(manifest)}
    assert result["requires_new_training_run"]
    assert records["fixture:0"]["split"] == records["fixture:3"]["split"] == "test"


def test_cubicasa_fetch_uses_embedded_publisher_splits(tmp_path, monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "files": [
                    {
                        "key": "cubicasa5k.zip",
                        "links": {"self": "https://publisher/archive"},
                        "size": 1,
                        "checksum": "md5:example",
                    }
                ]
            }

    monkeypatch.setattr(acquire.requests, "get", lambda *a, **k: Response())
    calls = []

    def fake_download(url, destination, *args):
        destination.parent.mkdir(parents=True, exist_ok=True)
        calls.append(destination.name)
        assert destination.suffix == ".zip", "Publisher splits are inside the original ZIP"
        with zipfile.ZipFile(destination, "w") as archive:
            for split in ("train", "val", "test"):
                archive.writestr(f"cubicasa5k/{split}.txt", f"high_quality/{split}/\n")
        return "fixture-digest"

    monkeypatch.setattr(acquire, "download", fake_download)
    acquire.fetch("cubicasa5k", tmp_path, extract=True)
    assert calls == ["cubicasa5k.zip"]
    for split in ("train", "val", "test"):
        path = tmp_path / "extracted/cubicasa5k/cubicasa5k" / f"{split}.txt"
        assert path.read_text() == f"high_quality/{split}/\n"


def test_floorplancad_flat_archives_preserve_train_test_partitions(tmp_path, monkeypatch):
    monkeypatch.setattr(
        acquire,
        "sources",
        lambda: {
            "floorplancad": {
                "source_url": "https://publisher/",
                "access": "fixture",
                "artifacts": [
                    {"name": n, "url": "https://publisher/" + n} for n in ("train-00.zip", "test-00.zip")
                ],
            }
        },
    )

    def fake_download(url, destination, *args):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("same-name.svg", destination.name)
        return "fixture-digest"

    monkeypatch.setattr(acquire, "download", fake_download)
    acquire.fetch("floorplancad", tmp_path, extract=True)
    for split in ("train-00", "test-00"):
        assert (tmp_path / "extracted/floorplancad" / split / "same-name.svg").read_text() == split + ".zip"


def test_unknown_room_category_counts_as_false_negative_and_missing_objects_fail_gate():
    targets = encode(
        (32, 32),
        [
            {
                "class": "room",
                "room_type": "kitchen",
                "outer": [[2, 2], [29, 2], [29, 29], [2, 29]],
            }
        ],
        ["room"],
    )
    prediction = {
        "instances": [
            {
                "class": "room",
                "room_type": "unknown",
                "confidence": 0.9,
                "bbox": [2, 2, 30, 30],
                "polygons": [{"outer": [[2, 2], [29, 2], [29, 29], [2, 29]]}],
            }
        ]
    }
    metric = SourceMetrics()
    metric.add(prediction, targets)
    report = metric.report()
    assert report["room_type_macro_f1_matched"] == 0
    assert report["room_type_unknown_count"] == 1
    evaluator = Evaluator()
    evaluator.add("fixture", prediction, targets)
    assert not evaluator.report()["acceptance_gates"]["all_object_classes_supported"]


def test_comparison_uses_validation_cohorts_and_blocks_source_regression(tmp_path):
    def report(a, b):
        return {
            "split": "val",
            "protocol": "same",
            "image_size": 1024,
            "tiled": True,
            "cohort_fingerprints": {"a": "a", "b": "b"},
            "sources": {
                s: {m: v for m in ("wall_iou", "object_map50", "room_instance_f1_iou50")}
                for s, v in (("a", a), ("b", b))
            },
        }

    base, candidate = tmp_path / "baseline.json", tmp_path / "candidate.json"
    write_json(base, report(0.8, 0.8))
    write_json(candidate, report(0.82, 0.82))
    assert compare(base, candidate)["select_candidate"]
    write_json(candidate, report(0.77, 0.99))
    assert not compare(base, candidate)["select_candidate"]
    changed = report(0.9, 0.9)
    changed["cohort_fingerprints"]["a"] = "different"
    write_json(candidate, changed)
    with pytest.raises(ValueError, match="cohort"):
        compare(base, candidate)
    changed["split"] = "test"
    candidate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="never test"):
        compare(base, candidate)
