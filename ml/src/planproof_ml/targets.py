"""Dense targets with explicit class coverage; no missing-label negatives."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .schema import IGNORE, INSTANCE_CLASSES, MASK_CLASSES, room_type
from .spatial import polygon_mask


def encode(
    shape: tuple[int, int],
    features: list[dict],
    complete: list[str],
    ignore_regions=(),
    instance_complete=None,
) -> dict:
    h, w = shape
    masks = np.zeros((len(MASK_CLASSES), h, w), np.uint8)
    instances = np.zeros((len(INSTANCE_CLASSES), h, w), np.uint16)
    valid = np.zeros_like(masks)
    types = np.full((h, w), IGNORE, np.uint8)
    pixel_valid = np.ones((h, w), np.uint8)
    weak = []
    objects = []
    counters = {c: 0 for c in INSTANCE_CLASSES}
    for name in complete:
        if name not in MASK_CLASSES:
            raise ValueError(f"Unknown annotated target: {name}")
        valid[MASK_CLASSES.index(name)] = 1
    for feature in features:
        name = feature["class"]
        if name not in MASK_CLASSES:
            continue
        if feature.get("weak"):
            if name == "room" and room_type(feature.get("room_type")) != IGNORE:
                weak.append([*feature["bbox"], room_type(feature["room_type"])])
            continue
        if feature.get("box_only"):
            if name in ("door", "window", "stairs"):
                objects.append([("door", "window", "stairs").index(name), *feature["bbox"]])
            continue
        mask = polygon_mask(shape, feature["outer"], feature.get("holes", []))
        channel = MASK_CLASSES.index(name)
        masks[channel] |= mask
        if name not in complete:
            valid[channel] |= mask  # positive-only supervision; missing regions remain ignored
        if name in INSTANCE_CLASSES:
            counters[name] += 1
            if counters[name] >= 65535:
                raise ValueError("Too many instances for uint16 label storage")
            instances[INSTANCE_CLASSES.index(name)][mask.astype(bool)] = counters[name]
        if name in ("door", "window", "stairs") and mask.any():
            yy, xx = np.where(mask)
            objects.append(
                [
                    ("door", "window", "stairs").index(name),
                    float(xx.min()),
                    float(yy.min()),
                    float(xx.max() + 1),
                    float(yy.max() + 1),
                ]
            )
        if name == "room":
            types[mask.astype(bool)] = room_type(feature.get("room_type"))
    # Unsupported regions are excluded from dense losses. Explicit object boxes
    # retain positive evidence separately; unknown regions supply no negatives.
    for region in ignore_regions:
        ignored = polygon_mask(shape, region["outer"], region.get("holes", [])).astype(bool)
        valid[:, ignored] = 0
        pixel_valid[ignored] = 0
        types[ignored] = IGNORE
        instances[:, ignored] = 0
    types[~masks[1].astype(bool)] = IGNORE
    complete_instances = complete if instance_complete is None else instance_complete
    return {
        "masks": masks,
        "instances": instances,
        "valid": valid,
        "types": types,
        "weak_types": np.asarray(weak, np.float32).reshape(-1, 5),
        "objects": np.asarray(objects, np.float32).reshape(-1, 5),
        "pixel_valid": pixel_valid,
        "object_complete": np.array(
            [n in complete_instances for n in ("door", "window", "stairs")], np.uint8
        ),
    }


def derived(instances: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = instances.shape[1:]
    centers = np.zeros((4, h, w), np.float32)
    center_valid = np.stack([valid[MASK_CLASSES.index(c)] for c in INSTANCE_CLASSES]).astype(np.float32)
    boundary = np.zeros((1, h, w), np.float32)
    for channel, name in enumerate(INSTANCE_CLASSES):
        label_image = instances[channel]
        for label in np.unique(label_image):
            if label == 0:
                continue
            ys, xs = np.where(label_image == label)
            if not len(xs):
                continue
            # A clipped instance has no center-regression supervision. Mask loss remains valid.
            clipped = xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1
            if clipped:
                center_valid[channel, label_image == label] = 0
                continue
            # Choose an interior point, never a centroid outside a concave room.
            cy, cx = int(round(ys.mean())), int(round(xs.mean()))
            closest = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
            cy, cx = int(ys[closest]), int(xs[closest])
            sigma = max(1.0, min(5.0, np.sqrt(len(xs)) / 12))
            radius = int(np.ceil(sigma * 3))
            y0, y1, x0, x1 = (
                max(0, cy - radius),
                min(h, cy + radius + 1),
                max(0, cx - radius),
                min(w, cx + radius + 1),
            )
            yy, xx = np.mgrid[y0:y1, x0:x1]
            peak = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
            centers[channel, y0:y1, x0:x1] = np.maximum(centers[channel, y0:y1, x0:x1], peak)
    rooms = instances[0]
    if np.any(rooms):
        # Instance boundaries, not tile edges or mere room-type transitions.
        eroded = ndimage.minimum_filter(rooms, size=3)
        dilated = ndimage.maximum_filter(rooms, size=3)
        boundary[0] = ((eroded != dilated) & (rooms > 0)).astype(np.float32)
    return centers, center_valid, boundary


def normalize_image(image):
    x = image.astype(np.float32) / 255.0
    return np.transpose(
        (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32),
        (2, 0, 1),
    ).copy()


def validate_targets(targets):
    masks, labels, valid, types = (targets[k] for k in ("masks", "instances", "valid", "types"))
    if masks.ndim != 3 or masks.shape[0] != 5 or labels.shape != (4, *masks.shape[1:]):
        raise ValueError("Invalid target tensor shape")
    if valid.shape != masks.shape or types.shape != masks.shape[1:]:
        raise ValueError("Coverage or room-type dimensions differ from image")
    if not set(np.unique(masks)).issubset({0, 1}) or not set(np.unique(valid)).issubset({0, 1}):
        raise ValueError("Masks/coverage must be binary")
    from .schema import ROOM_TYPES

    if not set(np.unique(types)).issubset(set(range(len(ROOM_TYPES))) | {IGNORE}):
        raise ValueError("Invalid room-type class")
