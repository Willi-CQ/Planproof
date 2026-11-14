import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

from planproof_ml.inference import peaks, split_rooms, component_instances, DEFAULT_THRESHOLDS
from planproof_ml.metrics import item_mask, item_roi_mask
from planproof_ml.spatial import mask_polygons


def test_vectorized_nms_matches_stable_class_aware_reference():
    from planproof_ml.inference import nms, box_iou

    rng = np.random.default_rng(42)
    low = rng.integers(0, 80, (500, 2))
    high = low + rng.integers(0, 40, (500, 2))
    items = [
        dict(bbox=[*a, *b], confidence=float(i % 7), **{"class": str(i % 3)})
        for i, (a, b) in enumerate(zip(low, high))
    ]
    for threshold in (0.0, 0.3, 0.5, 1.0):
        expected = []
        for item in sorted(items, key=lambda x: x["confidence"], reverse=True):
            if all(
                item["class"] != other["class"] or box_iou(item["bbox"], other["bbox"]) < threshold
                for other in expected
            ):
                expected.append(item)
        assert nms(items, threshold) == expected


def reference_peaks(prob, threshold, distance=5):
    labels, count = ndimage.label(
        (prob >= threshold) & (prob == ndimage.maximum_filter(prob, size=2 * distance + 1))
    )
    result = []
    for index in range(1, count + 1):
        y, x = np.where(labels == index)
        best = int(np.argmax(prob[y, x]))
        result.append((int(y[best]), int(x[best])))
    return result


def reference_rooms(room, boundary, centers):
    foreground = room >= 0.5
    components, count = ndimage.label(foreground)
    markers = np.zeros_like(components)
    next_id = 0
    for y, x in reference_peaks(centers, 0.3):
        if foreground[y, x]:
            next_id += 1
            markers[y, x] = next_id
    for index in range(1, count + 1):
        component = components == index
        if not np.any(markers[component]):
            y, x = np.unravel_index(np.argmax(ndimage.distance_transform_edt(component)), component.shape)
            next_id += 1
            markers[y, x] = next_id
    return (
        watershed(boundary - room, markers=markers, mask=foreground).astype(np.int32) if next_id else markers
    )


def test_component_crops_preserve_peak_and_room_results():
    rng = np.random.default_rng(51)
    probability = rng.random((64, 80), dtype=np.float32)
    assert peaks(probability, 0.6) == reference_peaks(probability, 0.6)
    for room in [
        rng.random((64, 80), dtype=np.float32),
        np.ones((20, 30), np.float32),
        np.zeros((20, 30), np.float32),
    ]:
        boundary = rng.random(room.shape, dtype=np.float32)
        centers = rng.random(room.shape, dtype=np.float32) * 0.2
        np.testing.assert_array_equal(
            split_rooms(room, boundary, centers, DEFAULT_THRESHOLDS), reference_rooms(room, boundary, centers)
        )


def test_roi_rasterization_matches_full_image_including_rounding_and_holes():
    shape = (80, 100)
    for outer, holes in [
        ([[3.5, 2.5], [40.5, 2.5], [40.5, 35.5], [3.5, 35.5]], [[[10, 10], [20, 10], [20, 20], [10, 20]]]),
        ([[-8, -2], [25, -2], [25, 25], [-8, 25]], []),
        ([[120, 90], [130, 90], [130, 100], [120, 100]], []),
    ]:
        item = {"polygons": [{"outer": outer, "holes": holes}]}
        bounds, region = item_roi_mask(item, shape)
        rebuilt = np.zeros(shape, bool)
        rebuilt[bounds] = region
        np.testing.assert_array_equal(rebuilt, item_mask(item, shape))


def test_component_polygons_and_confidence_preserve_full_frame_results():
    labels = np.zeros((80, 100), np.int32)
    labels[0:15, 4:26] = 1
    labels[3:7, 10:15] = 0
    labels[35:54, 70:89] = 2
    rng = np.random.default_rng(9)
    masks = rng.random((5, *labels.shape), dtype=np.float32)
    types = np.zeros((10, *labels.shape), np.float32)
    types[2] = 0.9
    items = component_instances("room", labels, 1, masks, types, 4)
    assert len(items) == 2
    for label, item in enumerate(items, 1):
        reference = {"polygons": mask_polygons(labels == label)}
        np.testing.assert_array_equal(item_mask(item, labels.shape), item_mask(reference, labels.shape))
        assert item["confidence"] == float(masks[1][labels == label].mean())
        assert item["room_type"] == "dining"
