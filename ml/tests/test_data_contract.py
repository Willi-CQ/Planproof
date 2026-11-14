import json
import numpy as np
import pytest
from PIL import Image

from planproof_ml.schema import element_class, room_type, IGNORE, validate_prediction
from planproof_ml.targets import encode
from planproof_ml.spatial import Frame, polygon_mask, tile_origins
from planproof_ml.manifest import split_records, audit
from planproof_ml.prepare import prepare
from planproof_ml.svg import read_svg
from planproof_ml.dataset import sample, transform_boxes
from planproof_ml.checks import check_corpus
from planproof_ml.adapters import svg_samples


def test_unknown_and_elevators_never_become_walls_or_stairs():
    assert element_class("unknown") is None
    assert element_class("elevator") is None
    assert element_class("escalator") is None
    assert element_class("single_door") == "door"
    assert room_type("undefined") == IGNORE


def test_roundtrip_nonsquare_pixel_frames_and_tiles():
    frame = Frame.fit(731, 118, 128)
    xy = np.array([[0, 0], [731, 118], [123.45, 88.1]])
    np.testing.assert_allclose(frame.inverse(frame.forward(xy)), xy, atol=1e-8)
    starts = tile_origins(379, 128, 0.25)
    coverage = np.zeros(379)
    for start in starts:
        coverage[start : start + 128] += 1
    assert np.all(coverage > 0)
    assert starts[-1] == 251


def test_polygon_holes_are_preserved():
    m = polygon_mask(
        (32, 32), [[1, 1], [30, 1], [30, 30], [1, 30]], [[[10, 10], [20, 10], [20, 20], [10, 20]]]
    )
    assert m[5, 5] == 1 and m[15, 15] == 0


def test_box_only_cad_labels_do_not_invent_dense_masks():
    targets = encode(
        (64, 64),
        [{"class": "door", "box_only": True, "bbox": [10, 20, 30, 40]}],
        [],
        instance_complete=["door", "window", "stairs"],
    )
    assert not targets["masks"].any()
    assert not targets["valid"].any()
    assert targets["objects"].tolist() == [[0.0, 10.0, 20.0, 30.0, 40.0]]
    assert targets["object_complete"].tolist() == [1, 1, 1]


def test_houseexpo_weak_labels_do_not_create_room_polygons():
    t = encode(
        (64, 64), [{"class": "room", "weak": True, "room_type": "kitchen", "bbox": [2, 3, 20, 30]}], []
    )
    assert not t["masks"].any() and not t["valid"].any()
    assert t["types"].min() == IGNORE
    assert t["weak_types"].shape == (1, 5)


def test_svg_nested_transforms_and_unsupported_regions(tmp_path):
    svg = tmp_path / "model.svg"
    svg.write_text(
        '<svg viewBox="0 0 100 100"><g transform="translate(10,20)">'
        '<g id="Wall"><polygon points="0,0 20,0 20,4 0,4"/></g>'
        '<g class="Space Kitchen"><polygon points="0,10 20,10 20,30 0,30"/></g>'
        '<g id="Elevator"><polygon points="30,10 40,10 40,20 30,20"/></g>'
        "</g></svg>"
    )
    features, ignored, _, _ = read_svg(svg, (200, 200))
    assert features[0]["outer"][0] == [20.0, 40.0]
    assert features[1]["room_type"] == "Kitchen"
    assert len(ignored) == 1
    assert all(f["class"] != "stairs" for f in features)


def test_original_cad_numeric_ids_require_explicit_mapping(tmp_path):
    svg = tmp_path / "plan.svg"
    svg.write_text(
        '<svg viewBox="0 0 100 100"><g>'
        '<path semanticId="1" instanceId="7" d="M10 20 L30 20"/>'
        '<path semanticId="1" instanceId="7" d="M30 20 L30 40"/>'
        '<path semanticId="33" instanceId="-1" d="M0 0 L80 0"/>'
        "</g></svg>"
    )
    with pytest.raises(ValueError, match="class-map"):
        read_svg(svg, (100, 100))
    result, _, _, _ = read_svg(svg, (100, 100), {"mapping": {"1": "door"}, "complete": ["door"]})
    assert result == [{"class": "door", "bbox": [10.0, 20.0, 30.0, 40.0], "box_only": True}]


def row(idx, split, source="cubicasa5k", image=None, group=None):
    return {
        "id": str(idx),
        "source": source,
        "group_id": group or str(idx),
        "family": "real",
        "image": "x",
        "targets": "x",
        "image_sha256": image or str(idx),
        "publisher_split": split,
        "split": split,
    }


def test_exact_duplicates_and_derivative_groups_never_cross_splits():
    rows = [
        row(1, "train", image="same"),
        row(2, "test", image="same"),
        row(3, "train", source="resplan", group="resplan:5"),
        row(4, "val", source="modres", group="resplan:5"),
    ]
    split_records(rows)
    assert rows[0]["split"] == rows[1]["split"] == "test"
    assert rows[2]["split"] == rows[3]["split"] == "val"
    assert audit(rows)["groups"] == 2


def test_leakage_audit_rejects_unjoined_exact_images():
    with pytest.raises(ValueError, match="leakage"):
        audit([row(1, "train", image="same"), row(2, "test", image="same")])


def test_crop_box_transform_matches_rotation_and_reflection():
    frame = Frame.fit(100, 100, 100)
    out, clipped = transform_boxes(np.array([[0, 10, 20, 30, 40]], float), (0, 0, 100, 100), frame, 1, True)
    np.testing.assert_allclose(out, [[0, 60, 70, 80, 90]], atol=1e-5)
    assert not clipped
    out, clipped = transform_boxes(
        np.array([[0, 10, 20, 80, 90]], float), (20, 20, 90, 90), Frame.fit(70, 70, 128), 0, False
    )
    assert len(out) == 0 and len(clipped) == 1


def test_preparation_original_split_and_manifest_audit(tmp_path):
    raw = tmp_path / "upstream"
    prepared = tmp_path / "prepared"
    for i, split in enumerate(("train", "val", "test")):
        d = raw / "plans" / str(i)
        d.mkdir(parents=True)
        im = np.full((64, 64, 3), 255, np.uint8)
        im[0, 0] = i
        Image.fromarray(im).save(d / "F1_scaled.png")
        (d / "model.svg").write_text(
            '<svg viewBox="0 0 64 64"><g id="Wall">'
            f'<polygon points="2,2 {50 - i},2 {50 - i},7 2,7"/></g></svg>'
        )
        (raw / (split + ".txt")).write_text(f"/plans/{i}/\n")
    result = prepare("cubicasa5k", raw, prepared)
    assert result["added"] == 3
    report = json.loads((prepared / "manifest.audit.json").read_text())
    assert report["splits"] == {"train": 1, "val": 1, "test": 1}
    assert report["publisher_reassignments"] == 0
    prepare("cubicasa5k", raw, tmp_path / "parallel", workers=3)
    assert (prepared / "manifest.jsonl").read_text() == (tmp_path / "parallel/manifest.jsonl").read_text()
    with pytest.raises(ValueError, match="workers"):
        prepare("cubicasa5k", raw, tmp_path / "invalid", workers=0)


def test_stoa_archive_path_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Stoa"):
        prepare("cubicasa5k", tmp_path / "junction-dataset-preview", tmp_path / "out")


def test_cubicasa_uses_scaled_raster_pixels_when_svg_viewport_differs(tmp_path):
    raw = tmp_path / "upstream"
    drawing = raw / "colorful" / "1"
    drawing.mkdir(parents=True)
    Image.fromarray(np.full((100, 120, 3), 255, np.uint8)).save(drawing / "F1_scaled.png")
    (drawing / "model.svg").write_text(
        '<svg viewBox="0 0 80 60"><g id="Wall"><polygon points="10,20 40,20 40,25 10,25"/></g></svg>'
    )
    (raw / "train.txt").write_text("/colorful/1/\n")
    prepare("cubicasa5k", raw, tmp_path / "prepared")
    record = json.loads((tmp_path / "prepared/manifest.jsonl").read_text())
    with np.load(tmp_path / "prepared" / record["targets"]) as labels:
        yy, xx = np.where(labels["masks"][0])
    assert (xx.min(), xx.max(), yy.min(), yy.max()) == (10, 40, 20, 25)
    assert record["metadata"]["svg"]["coordinate_space"] == "pixels"
    capped = next(svg_samples("cubicasa5k", raw, raster_max_size=60))
    assert capped["image"].shape[:2] == (50, 60)
    assert capped["features"][0]["outer"][0] == [5.0, 10.0]
    assert capped["metadata"]["svg"]["original_raster_size"] == [120, 100]
    (drawing / "F1_scaled.png").rename(drawing / "F1_original.png")
    with pytest.raises(ValueError, match="Missing paired image"):
        prepare("cubicasa5k", raw, tmp_path / "other")


def test_decorative_malformed_paths_do_not_discard_valid_semantic_targets(tmp_path):
    path = tmp_path / "model.svg"
    path.write_text(
        '<svg viewBox="0 0 64 64"><g class="FixedFurniture Misc">'
        '<path d="M0,0 L46,90 LNaN,NaN"/></g><g id="Wall">'
        '<polygon points="2,2 20,2 20,8 2,8"/></g></svg>'
    )
    features, _, _, info = read_svg(path, (64, 64), collect_strokes=False)
    assert len(features) == 1 and features[0]["class"] == "wall"
    assert info["unlabeled_primitives_skipped"] == 1
    with pytest.raises(ValueError):
        read_svg(path, (64, 64))
    path.write_text('<svg viewBox="0 0 64 64"><g id="Wall"><path d="M0,0 LNaN,NaN"/></g></svg>')
    with pytest.raises(ValueError):
        read_svg(path, (64, 64), collect_strokes=False)


def test_padding_is_not_supervised_background(tmp_path):
    manifest = check_corpus(tmp_path / "fixture")
    record = json.loads(manifest.read_text().splitlines()[0])
    # Replace the image/targets with a narrow page to require letterbox padding.
    image = np.full((20, 64, 3), 255, np.uint8)
    Image.fromarray(image).save(manifest.parent / record["image"])
    target = encode((20, 64), [{"class": "wall", "outer": [[1, 1], [60, 1], [60, 5], [1, 5]]}], ["wall"])
    np.savez_compressed(manifest.parent / record["targets"], **target)
    batch = sample(record, manifest.parent, 64, np.random.default_rng(42), augment=False)
    assert not batch["valid"][:, :20].any()
    assert batch["valid"][0, 30, 10] == 1
    assert batch["types"][:20].min() == IGNORE


def test_prediction_bounds_and_schema():
    result = {
        "schema_version": "planproof-detection-v1",
        "image_size": [64, 64],
        "instances": [{"id": "x", "class": "door", "bbox": [10, 10, 20, 20], "confidence": 0.8}],
    }
    validate_prediction(result)
    result["instances"][0]["class"] = "unknown"
    with pytest.raises(ValueError):
        validate_prediction(result)
