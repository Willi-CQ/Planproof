"""Deterministic per-step full-plan/crop sampling and aligned target transforms."""

from __future__ import annotations
import cv2
import numpy as np
from .adapters import load_image
from .schema import IGNORE
from .spatial import Frame
from .targets import derived, normalize_image


def transform_boxes(boxes, crop, frame, k, flip):
    x0, y0, x1, y1 = crop
    transformed = []
    clipped = []
    for row in boxes:
        label, xmin, ymin, xmax, ymax = row
        if xmax <= x0 or ymax <= y0 or xmin >= x1 or ymin >= y1:
            continue
        partial = xmin < x0 or ymin < y0 or xmax > x1 or ymax > y1
        points = np.array(
            [[max(xmin, x0) - x0, max(ymin, y0) - y0], [min(xmax, x1) - x0, min(ymax, y1) - y0]]
        )
        points = frame.forward(points)
        xa, ya = points[0]
        xb, yb = points[1]
        corners = np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]])
        for _ in range(k):
            corners = np.column_stack((corners[:, 1], frame.size - corners[:, 0]))
        if flip:
            corners[:, 0] = frame.size - corners[:, 0]
        lo, hi = corners.min(0), corners.max(0)
        target = [label, *lo, *hi]
        (clipped if partial else transformed).append(target)
    return np.asarray(transformed, np.float32).reshape(-1, 5), clipped


def sample(record, root, size=1024, rng=None, crop_probability=0.5, augment=True):
    image = load_image(root / record["image"])
    with np.load(root / record["targets"], allow_pickle=False) as f:
        arrays = {k: f[k].copy() for k in f.files}
    rng = np.random.default_rng() if rng is None else rng
    h, w = image.shape[:2]
    x0 = y0 = 0
    x1, y1 = w, h
    if augment and rng.random() < crop_probability:
        side = min(max(64, int(rng.uniform(0.75, 1.25) * min(size, max(h, w)))), max(h, w))
        x0 = int(rng.integers(0, max(1, w - side + 1)))
        y0 = int(rng.integers(0, max(1, h - side + 1)))
        x1, y1 = min(w, x0 + side), min(h, y0 + side)
    frame = Frame.fit(x1 - x0, y1 - y0, size)
    image = frame.image(image[y0:y1, x0:x1])
    for key in ("masks", "instances", "valid"):
        arrays[key] = np.stack([frame.mask(a[y0:y1, x0:x1]) for a in arrays[key]])
    arrays["types"] = frame.mask(arrays["types"][y0:y1, x0:x1], IGNORE)
    pixel_valid = frame.mask(arrays.get("pixel_valid", np.ones((h, w), np.uint8))[y0:y1, x0:x1])
    active = frame.mask(np.ones((y1 - y0, x1 - x0), np.uint8))
    k = int(rng.integers(0, 4)) if augment else 0
    flip = bool(rng.integers(0, 2)) if augment else False

    def spatial(a):
        a = np.rot90(a, k, axes=(-2, -1))
        return np.flip(a, axis=-1).copy() if flip else a.copy()

    image = np.rot90(image, k, axes=(0, 1))
    if flip:
        image = np.flip(image, axis=1)
    image = image.copy()
    for key in ("masks", "instances", "valid", "types"):
        arrays[key] = spatial(arrays[key])
    active = spatial(active)
    active *= spatial(pixel_valid)
    objects, clipped = transform_boxes(arrays["objects"], (x0, y0, x1, y1), frame, k, flip)
    weak, _ = transform_boxes(
        np.column_stack((arrays["weak_types"][:, 4], arrays["weak_types"][:, :4]))
        if len(arrays["weak_types"])
        else np.empty((0, 5)),
        (x0, y0, x1, y1),
        frame,
        k,
        flip,
    )
    centers, center_valid, boundary = derived(arrays["instances"], arrays["valid"])
    # Object centers/size regression come from boxes, not arbitrary filled CAD rectangles.
    centers[1:] = 0
    center_valid[1:] = arrays["object_complete"][:, None, None] * active[None]
    box_target = np.zeros((12, size, size), np.float32)
    box_valid = np.zeros((3, size, size), np.float32)
    for label, xmin, ymin, xmax, ymax in objects:
        label = int(label)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        ix, iy = int(np.clip(round(cx), 0, size - 1)), int(np.clip(round(cy), 0, size - 1))
        sigma = max(1.0, min(5.0, np.sqrt((xmax - xmin) * (ymax - ymin)) / 12))
        radius = int(np.ceil(3 * sigma))
        yy0, yy1 = max(0, iy - radius), min(size, iy + radius + 1)
        xx0, xx1 = max(0, ix - radius), min(size, ix + radius + 1)
        yy, xx = np.mgrid[yy0:yy1, xx0:xx1]
        gaussian = np.exp(-((yy - iy) ** 2 + (xx - ix) ** 2) / (2 * sigma**2))
        centers[label + 1, yy0:yy1, xx0:xx1] = np.maximum(centers[label + 1, yy0:yy1, xx0:xx1], gaussian)
        box_target[label * 4 : (label + 1) * 4, iy, ix] = [
            (xmax - xmin) / size,
            (ymax - ymin) / size,
            cx - ix,
            cy - iy,
        ]
        box_valid[label, iy, ix] = 1
        # Partial-annotation positives can be supervised locally without global negatives.
        if not arrays["object_complete"][label]:
            center_valid[label + 1, yy0:yy1, xx0:xx1] = active[yy0:yy1, xx0:xx1]
    for label, xa, ya, xb, yb in clipped:
        center_valid[int(label) + 1, int(ya) : int(np.ceil(yb)), int(xa) : int(np.ceil(xb))] = 0
    if augment:
        if rng.random() < 0.2:
            image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        if rng.random() < 0.15:
            image = cv2.erode(image, np.ones((2, 2), np.uint8))  # thicken ink; never erase thin walls
        if rng.random() < 0.25:
            image = cv2.GaussianBlur(image, (3, 3), float(rng.uniform(0.1, 0.7)))
        if rng.random() < 0.35:
            image = np.clip(
                image.astype(float) * rng.uniform(0.8, 1.15) + rng.normal(0, 3, image.shape), 0, 255
            ).astype(np.uint8)
        if rng.random() < 0.15:
            ok, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(45, 90))],
            )
            if ok:
                image = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return {
        "image": normalize_image(image),
        "masks": arrays["masks"].astype(np.float32),
        "valid": arrays["valid"].astype(np.float32),
        "types": arrays["types"].astype(np.int64),
        "centers": centers,
        "center_valid": center_valid,
        "boundary": boundary,
        "boundary_valid": arrays["valid"][1:2].astype(np.float32),
        "boxes": box_target,
        "box_valid": box_valid,
        "weak_types": weak,
        "objects": objects,
        "instances": arrays["instances"],
        "frame": frame,
        "rgb": image,
    }


def tensor_batch(samples, device):
    import torch

    keys = (
        "image",
        "masks",
        "valid",
        "types",
        "centers",
        "center_valid",
        "boundary",
        "boundary_valid",
        "boxes",
        "box_valid",
    )
    batch = {k: torch.from_numpy(np.stack([s[k] for s in samples])).to(device) for k in keys}
    batch["weak_types"] = [s["weak_types"] for s in samples]
    return batch
