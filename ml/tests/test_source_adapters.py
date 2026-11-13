import json
import pickle
import numpy as np
import pytest
from PIL import Image
from shapely.geometry import box, mapping
from planproof_ml.adapters import resplan_samples, modres_samples, rplan_samples, houseexpo_samples
from planproof_ml.extra_adapters import structured3d_samples, sesyd_samples
from planproof_ml.targets import encode


def test_resplan_requires_explicit_pickle_trust_and_preserves_canonical_split(tmp_path):
    plans = [
        {"wall": box(0, 0, 5, 0.2), "kitchen": box(0.2, 0.2, 4, 4), "door": box(1, 0, 2, 0.2)},
        {"wall": box(0, 0, 6, 0.2), "bedroom": box(0.2, 0.2, 5, 5), "door": None},
    ]
    with (tmp_path / "ResPlan.pkl").open("wb") as stream:
        pickle.dump(plans, stream)
    (tmp_path / "split.json").write_text(json.dumps({"train": [0], "test": [1], "augmented": []}))
    with pytest.raises(ValueError, match="trust-pickle"):
        list(resplan_samples(tmp_path))
    examples = list(resplan_samples(tmp_path, True))
    assert [x["publisher_split"] for x in examples] == ["train", "test"]
    assert examples[0]["image"].shape == (1024, 1024, 3)
    assert "stairs" not in examples[0]["complete"]
    assert "door" in examples[1]["complete"]  # explicitly present empty annotation field


def test_modres_geojson_masks_and_coverage_do_not_invent_negatives(tmp_path):
    folder = tmp_path / "Geometry/2D/Geometry"
    folder.mkdir(parents=True)
    (folder / "original-5.json").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": mapping(box(0, 0, 5, 0.2)),
                        "properties": {"class": "wall", "room_type": None},
                    },
                    {
                        "type": "Feature",
                        "geometry": mapping(box(0.2, 0.2, 4, 4)),
                        "properties": {"class": "room", "room_type": "kitchen"},
                    },
                ],
            }
        )
    )
    example = next(modres_samples(tmp_path))
    assert example["group"] == "resplan:original-5"
    assert example["complete"] == []
    assert any(f["class"] == "wall" for f in example["features"])
    arrays = encode((1024, 1024), example["features"], example["complete"])
    assert arrays["valid"][0].sum() == arrays["masks"][0].sum()
    assert arrays["valid"][0, 0, 0] == 0


def test_rplan_semantic_channels_are_not_training_image_inputs(tmp_path):
    image = np.zeros((64, 64, 4), np.uint8)
    image[:, :, 1] = 13
    image[10:50, 10:50, 1] = 2
    image[10:50, 10:50, 2] = 1
    image[8:52, 8:52, 3] = 255
    image[8:10, 8:52, 0] = 127
    Image.fromarray(image, "RGBA").save(tmp_path / "1.png")
    example = next(rplan_samples(tmp_path))
    assert example["image"].shape == (64, 64, 3)
    assert np.array_equal(example["image"][:, :, 0], example["image"][:, :, 1])
    assert any(f.get("room_type") == "kitchen" for f in example["features"])
    assert example["complete"] == ["wall", "room"]


def test_houseexpo_supplies_only_weak_category_boxes(tmp_path):
    (tmp_path / "sample.json").write_text(
        json.dumps(
            {
                "id": "own-fixture",
                "verts": [[0, 0], [5, 0], [5, 5], [0, 5]],
                "room_category": {"kitchen": [0, 0, 2, 2], "bedroom": [2, 0, 5, 5]},
            }
        )
    )
    example = next(houseexpo_samples(tmp_path))
    assert example["complete"] == []
    assert all(f["weak"] for f in example["features"])
    arrays = encode((1024, 1024), example["features"], [])
    assert arrays["masks"].sum() == 0 and arrays["weak_types"].shape == (2, 5)


def test_structured3d_derives_room_only_targets_and_original_scene_split(tmp_path):
    scene = tmp_path / "scene_03000"
    scene.mkdir()
    (scene / "annotation_3d.json").write_text(
        json.dumps(
            {
                "junctions": [
                    {"coordinate": [0, 0, 0]},
                    {"coordinate": [4, 0, 0]},
                    {"coordinate": [4, 3, 0]},
                    {"coordinate": [0, 3, 0]},
                ],
                "planes": [{"ID": 0, "type": "floor"}],
                "semantics": [{"type": "kitchen", "planeID": [0]}],
                "lineJunctionMatrix": [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]],
                "planeLineMatrix": [[1, 1, 1, 1]],
            }
        )
    )
    sample = next(structured3d_samples(tmp_path))
    assert sample["publisher_split"] == "val"
    assert sample["complete"] == ["room"]
    assert sample["features"][0]["class"] == "room"


def test_sesyd_template_group_and_symbol_boxes(tmp_path):
    folder = tmp_path / "layout03"
    folder.mkdir()
    Image.new("RGB", (64, 64), "white").save(folder / "example.png")
    (folder / "example.xml").write_text(
        "<annotation><object><name>door1</name><bndbox>"
        "<xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax>"
        "</bndbox></object></annotation>"
    )
    sample = next(sesyd_samples(tmp_path))
    assert sample["group"] == "template:3"
    assert sample["features"] == [{"class": "door", "bbox": [10.0, 20.0, 30.0, 40.0], "box_only": True}]
