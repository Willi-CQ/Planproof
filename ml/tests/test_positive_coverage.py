import json

import numpy as np
import pytest

from planproof_ml.checks import check_corpus
from planproof_ml.dataset import sample
from planproof_ml.targets import encode
from planproof_ml.metrics import SourceMetrics
from planproof_ml.config import load_config


def test_known_stair_center_survives_unknown_region_without_inventing_negatives(tmp_path):
    manifest = check_corpus(tmp_path / "data")
    record = json.loads(manifest.read_text().splitlines()[0])
    targets = encode(
        (64, 64),
        [{"class": "stairs", "outer": [[10, 10], [30, 10], [30, 30], [10, 30]]}],
        ["stairs"],
        [{"outer": [[0, 0], [63, 0], [63, 63], [0, 63]]}],
    )
    np.savez_compressed(manifest.parent / record["targets"], **targets)
    result = sample(record, manifest.parent, 64, augment=False)
    positive = result["centers"][3] == 1
    assert positive.sum() == 1
    assert result["center_valid"][3][positive].min() == 1
    assert result["center_valid"][3].sum() == 1
    assert not result["valid"].any()
    assert result["box_valid"][2].sum() == 1


def test_known_object_can_match_in_ignored_region_but_unknown_predictions_stay_ignored():
    targets = encode(
        (32, 32),
        [{"class": "door", "outer": [[10, 10], [20, 10], [20, 20], [10, 20]]}],
        ["door"],
        [{"outer": [[0, 0], [31, 0], [31, 31], [0, 31]]}],
    )
    truth = targets["objects"][0, 1:].tolist()
    instances = [
        {"class": "door", "bbox": truth, "confidence": 0.9},
        {"class": "door", "bbox": [23, 23, 28, 28], "confidence": 1.0},
    ]
    metrics = SourceMetrics()
    metrics.add({"instances": instances}, targets)
    assert metrics.report()["objects"]["door"]["ap50"] == 1
    targets["pixel_valid"][:4, :4] = 1
    metrics = SourceMetrics()
    metrics.add(
        {"instances": [*instances, {"class": "door", "bbox": [0, 0, 4, 4], "confidence": 1.0}]}, targets
    )
    assert metrics.report()["objects"]["door"]["ap50"] == pytest.approx(0.5)


def test_training_contract_versions_positive_target_semantics(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("image_size: 64\nphases: [{name: check, steps: 1, sources: {fixture: 1}}]\n")
    assert load_config(path)["target_transform_revision"] == 2
    path.write_text(path.read_text() + "target_transform_revision: 1\n")
    with pytest.raises(ValueError, match="initialize a new run"):
        load_config(path)
