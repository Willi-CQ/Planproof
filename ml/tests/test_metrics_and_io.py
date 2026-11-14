import zipfile
import hashlib
import numpy as np
import pytest
from planproof_ml.acquire import extract_archive, download, sources
from planproof_ml.metrics import average_precision, assignment, Evaluator
from planproof_ml.targets import encode
from planproof_ml.inference import nms
from planproof_ml.schema import VERSION


def test_101_point_ap_and_one_to_one_matching():
    assert average_precision([1], [0.9], 1) == 1.0
    assert average_precision([0, 1], [0.99, 0.8], 1) == 0.5
    assert average_precision([], [], 0) is None
    assert len(assignment(np.array([[0.9, 0.51], [0.8, 0.49]]), 0.5)) == 2


def test_perfect_predictions_and_absent_ground_truth_classes():
    feature = {"class": "room", "room_type": "kitchen", "outer": [[5, 5], [20, 5], [20, 20], [5, 20]]}
    targets = encode(
        (32, 32),
        [feature, {"class": "door", "outer": [[1, 10], [4, 10], [4, 15], [1, 15]]}],
        ["wall", "room", "door", "window", "stairs"],
    )
    result = {
        "schema_version": VERSION,
        "image_size": [32, 32],
        "instances": [
            {
                "id": "r",
                "class": "room",
                "bbox": [5, 5, 21, 21],
                "polygons": [{"outer": feature["outer"], "holes": []}],
                "confidence": 0.9,
                "room_type": "kitchen",
            },
            {"id": "d", "class": "door", "bbox": [1, 10, 5, 16], "confidence": 0.9},
        ],
    }
    ev = Evaluator()
    ev.add("fixture", result, targets)
    report = ev.report()
    assert report["selection_score"] == 1.0
    assert report["sources"]["fixture"]["objects"]["window"]["ap50"] is None
    assert report["sources"]["fixture"]["room_type_macro_f1_matched"] == 1.0


def test_partial_labels_do_not_turn_predictions_into_false_positives():
    targets = encode((16, 16), [], [])
    ev = Evaluator()
    ev.add("unlabeled", {"instances": [{"class": "door", "bbox": [1, 1, 5, 5], "confidence": 0.9}]}, targets)
    report = ev.report()
    assert report["selection_score"] is None
    assert report["sources"]["unlabeled"]["object_map50"] is None


def test_class_aware_nms():
    items = [
        {"class": "door", "bbox": [0, 0, 10, 10], "confidence": 0.9},
        {"class": "door", "bbox": [1, 1, 11, 11], "confidence": 0.8},
        {"class": "window", "bbox": [0, 0, 10, 10], "confidence": 0.7},
    ]
    assert len(nms(items, 0.5)) == 2


def test_archive_traversal_and_symlinks_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "outside.txt").exists()
    with zipfile.ZipFile(archive, "w") as z:
        info = zipfile.ZipInfo("link")
        info.external_attr = 0o120777 << 16
        z.writestr(info, "target")
    with pytest.raises(ValueError, match="symlink"):
        extract_archive(archive, tmp_path / "out")


class Response:
    status_code = 200
    headers = {"Content-Type": "application/zip"}

    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.data


def test_resuming_server_that_ignores_range_restarts_safely(tmp_path, monkeypatch):
    destination = tmp_path / "original.zip"
    destination.with_suffix(".zip.part").write_bytes(b"old-prefix")
    received = []

    def get(url, **kwargs):
        received.append(kwargs["headers"])
        return Response(b"complete")

    monkeypatch.setattr("planproof_ml.acquire.requests.get", get)
    checksum = "sha256:" + hashlib.sha256(b"complete").hexdigest()
    download("https://publisher.invalid/file", destination, 8, checksum)
    assert destination.read_bytes() == b"complete"
    assert received[0]["Range"] == "bytes=10-"


def test_checksum_failure_cannot_be_used_for_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("planproof_ml.acquire.requests.get", lambda *a, **k: Response(b"bad"))
    destination = tmp_path / "original.zip"
    with pytest.raises(ValueError, match="checksum"):
        download("https://publisher.invalid/file", destination, 3, "sha256:" + "0" * 64)
    assert not destination.exists()
    assert not destination.with_suffix(".zip.part").exists()


def test_source_registry_has_eleven_original_sources_and_no_stoa():
    registry = sources()
    assert len(registry) == 11
    assert {"cubicasa5k", "floorplancad", "resplan", "structured3d"} <= registry.keys()
    assert not any("stoa" in key for key in registry)
    assert len(registry["floorplancad"]["artifacts"]) == 3
