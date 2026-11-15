"""Agent coordinates must match the crop tools without changing source evidence."""

from copy import deepcopy
from test_question_agent import agent


def test_all_catalogue_boxes_share_the_inspection_coordinate_frame():
    value = agent()
    source = value.algorithm_factory.return_value.catalog
    source["observations"][0]["bbox"] = [40, 20, 100, 40]
    source["openings"] = [{"id": "gap-1", "type": "window", "bbox": [10, 10, 20, 20],
                           "adjacent_region_ids": ["long-registered-room"]}]
    original = deepcopy(source)
    result, _ = value.execute("get_algorithms", {"classes": ["room", "window"]})
    assert result["coordinate_frame"] == "original_page_normalized_0_1000"
    assert result["rooms"][0]["bbox"] == [50, 100, 400, 800]
    assert result["observations"][0]["bbox"] == [200, 200, 500, 400]
    assert result["openings"][0]["bbox"] == [50, 100, 100, 200]
    assert source == original
    assert value.rooms["R1"]["bbox"] == [10, 10, 80, 80]
    # Passing a catalogue box directly to the agent crop tool recovers the
    # original physical region, not a double-scaled or displaced crop.
    _, images = value.execute("crop_region", {"bbox": result["observations"][0]["bbox"]})
    assert images[0].metadata["canonical_bbox"] == [40, 20, 100, 40]
    _, images = value.execute("crop_room", {"room_id": "R1"})
    assert images[0].metadata["canonical_bbox"] == [0, 0, 92, 92]


def test_coordinate_conversion_is_idempotent_across_class_delivery():
    value = agent()
    source = deepcopy(value.algorithm_factory.return_value.catalog)
    first, _ = value.execute("get_algorithms", {"classes": ["window"]})
    second, _ = value.execute("get_algorithms", {"classes": ["room"]})
    assert not first["rooms"]
    assert second["rooms"][0]["bbox"] == [50, 100, 400, 800]
    assert value.algorithm_factory.return_value.catalog == source
