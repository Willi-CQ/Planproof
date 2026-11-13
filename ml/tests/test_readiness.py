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
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(manifest.parent / "fixture-0.png")
    with pytest.raises(ValueError, match="Image hash mismatch"):
        audit_manifest(manifest, deep=True)


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


def test_cubicasa_fetch_preserves_publisher_split_sidecars(tmp_path, monkeypatch):
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
        if destination.suffix == ".zip":
            with zipfile.ZipFile(destination, "w") as archive:
                archive.writestr("sample/example.txt", "generated test only")
        else:
            destination.write_text("high_quality/1/F1_scaled.png\n")
        return "fixture-digest"

    monkeypatch.setattr(acquire, "download", fake_download)
    acquire.fetch("cubicasa5k", tmp_path, extract=True)
    assert set(calls) == {"cubicasa5k.zip", "train.txt", "val.txt", "test.txt"}
    assert (tmp_path / "extracted/cubicasa5k/train.txt").is_file()


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
