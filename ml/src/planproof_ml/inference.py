"""Full-plan and overlapping-tile detection in original image coordinates."""

from __future__ import annotations
import hashlib
from copy import deepcopy

import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

from .schema import VERSION, ROOM_TYPES, validate_prediction
from .spatial import Frame, tile_origins, mask_polygons
from .targets import normalize_image

DEFAULT_THRESHOLDS = {"masks": [0.5] * 5, "centers": [0.3] * 4, "nms_iou": 0.5, "min_pixels": 4}


def box_iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, xb - xa) * max(0, yb - ya)
    union = (
        max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - intersection
    )
    return intersection / union if union else 0.0


def nms(items, threshold=0.5):
    ordered = sorted(items, key=lambda x: x["confidence"], reverse=True)
    if not ordered:
        return []
    boxes = np.asarray([item["bbox"] for item in ordered], dtype=np.float64)
    classes = np.asarray([item["class"] for item in ordered])
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    remaining = np.arange(len(ordered))
    selected = []
    while len(remaining):
        index = remaining[0]
        selected.append(ordered[index])
        remaining = remaining[1:]
        low = np.maximum(boxes[index, :2], boxes[remaining, :2])
        high = np.minimum(boxes[index, 2:], boxes[remaining, 2:])
        intersection = np.maximum(0, high[:, 0] - low[:, 0]) * np.maximum(0, high[:, 1] - low[:, 1])
        union = areas[index] + areas[remaining] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
        remaining = remaining[(classes[remaining] != classes[index]) | (overlap < threshold)]
    return selected


def peaks(prob, threshold, min_distance=5):
    mask = (prob >= threshold) & (prob == ndimage.maximum_filter(prob, size=2 * min_distance + 1))
    labels, _ = ndimage.label(mask)
    found = []
    for idx, bounds in enumerate(ndimage.find_objects(labels), 1):
        if bounds is None:
            continue
        yy, xx = np.where(labels[bounds] == idx)
        best = int(np.argmax(prob[bounds][yy, xx]))
        found.append((int(yy[best] + bounds[0].start), int(xx[best] + bounds[1].start)))
    return found


def split_rooms(room_prob, boundary, center_prob, thresholds):
    foreground = room_prob >= thresholds["masks"][1]
    components, _ = ndimage.label(foreground)
    markers = np.zeros_like(components)
    next_id = 0
    for cy, cx in peaks(center_prob, thresholds["centers"][0]):
        if foreground[cy, cx]:
            next_id += 1
            markers[cy, cx] = next_id
    for idx, bounds in enumerate(ndimage.find_objects(components), 1):
        if bounds is None:
            continue
        # Include one in-image zero border, preserving whole-image EDT semantics
        # without scanning the entire page for every disconnected component.
        bounds = tuple(
            slice(max(0, b.start - 1), min(n, b.stop + 1)) for b, n in zip(bounds, components.shape)
        )
        component = components[bounds] == idx
        if not np.any(markers[bounds][component]):
            y, x = np.unravel_index(np.argmax(ndimage.distance_transform_edt(component)), component.shape)
            next_id += 1
            markers[y + bounds[0].start, x + bounds[1].start] = next_id
    if not next_id:
        return markers
    return watershed(boundary - room_prob, markers=markers, mask=foreground).astype(np.int32)


def run_view(model, image, device="cpu"):
    import torch

    with torch.inference_mode():
        x = torch.from_numpy(normalize_image(image))[None].to(device)
        pred = model(x)
        result = {}
        for key, value in pred.items():
            v = value[0].float()
            if key in {"masks", "boundary", "centers"}:
                v = v.sigmoid()
            elif key == "types":
                v = v.softmax(0)
            result[key] = v.cpu().numpy()
        return result


def view_boxes(pred, frame, offset, thresholds):
    boxes = []
    size = frame.size
    for cls, name in enumerate(("door", "window", "stairs")):
        for cy, cx in peaks(pred["centers"][cls + 1], thresholds["centers"][cls + 1]):
            if not (
                frame.left <= cx < frame.left + frame.scaled_width
                and frame.top <= cy < frame.top + frame.scaled_height
            ):
                continue
            vals = pred["boxes"][cls * 4 : (cls + 1) * 4, cy, cx]
            wh = 1 / (1 + np.exp(-np.clip(vals[:2], -30, 30))) * size
            dx, dy = np.tanh(vals[2:]) * 0.5
            points = np.array(
                [[cx + dx - wh[0] / 2, cy + dy - wh[1] / 2], [cx + dx + wh[0] / 2, cy + dy + wh[1] / 2]]
            )
            mapped = frame.inverse(points)
            mapped[:, 0] = np.clip(mapped[:, 0], 0, frame.width) + offset[0]
            mapped[:, 1] = np.clip(mapped[:, 1], 0, frame.height) + offset[1]
            x1, y1 = mapped[0]
            x2, y2 = mapped[1]
            if x2 > x1 and y2 > y1:
                boxes.append(
                    {
                        "class": name,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(pred["centers"][cls + 1, cy, cx]),
                    }
                )
    return boxes


def detect(
    model,
    image,
    device="cpu",
    size=1024,
    thresholds=None,
    tiled=True,
    max_side=4096,
    document_id=None,
    page_index=0,
    model_version="unversioned",
    return_maps=False,
):
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    original_h, original_w = image.shape[:2]
    scale = min(1.0, max_side / max(original_h, original_w))
    work = (
        cv2.resize(image, (max(1, round(original_w * scale)), max(1, round(original_h * scale))))
        if scale < 1
        else image
    )
    h, w = work.shape[:2]
    numerator = np.zeros((5 + 1 + len(ROOM_TYPES), h, w), np.float32)
    denominator = np.zeros((h, w), np.float32)
    room_centers = np.zeros((h, w), np.float32)
    candidates = []
    views = [(0, 0, w, h, 0.25 if tiled and max(h, w) > size else 1.0)]
    if tiled and max(h, w) > size:
        views.extend(
            (x, y, min(w, x + size), min(h, y + size), 1.0)
            for y in tile_origins(h, size)
            for x in tile_origins(w, size)
        )
    for xa, ya, xb, yb, weight in views:
        frame = Frame.fit(xb - xa, yb - ya, size)
        prediction = run_view(model, frame.image(work[ya:yb, xa:xb]), device)
        candidates.extend(view_boxes(prediction, frame, (xa, ya), thresholds))
        channels = np.concatenate([prediction["masks"], prediction["boundary"], prediction["types"]])
        restored = np.stack([frame.restore(c) for c in channels])
        hh, ww = yb - ya, xb - xa
        window = np.maximum(np.outer(np.hanning(hh), np.hanning(ww)), 0.05).astype(np.float32) * weight
        numerator[:, ya:yb, xa:xb] += restored * window
        denominator[ya:yb, xa:xb] += window
        room_centers[ya:yb, xa:xb] = np.maximum(
            room_centers[ya:yb, xa:xb], frame.restore(prediction["centers"][0])
        )
    maps = numerator / np.maximum(denominator, 1e-6)
    masks, boundary, types = maps[:5], maps[5], maps[6:]
    rooms = split_rooms(masks[1], boundary, room_centers, thresholds)
    instances = nms(candidates, thresholds["nms_iou"])
    for name, label_map, channel in (
        ("room", rooms, 1),
        ("wall", ndimage.label(masks[0] >= thresholds["masks"][0])[0], 0),
    ):
        instances.extend(
            component_instances(name, label_map, channel, masks, types, thresholds["min_pixels"])
        )
    sx, sy = original_w / w, original_h / h
    for idx, item in enumerate(instances):
        item["id"] = f"p{page_index}-{item['class']}-{idx:05d}"
        item["bbox"] = [
            item["bbox"][0] * sx,
            item["bbox"][1] * sy,
            item["bbox"][2] * sx,
            item["bbox"][3] * sy,
        ]
        for polygon in item.get("polygons", []):
            polygon["outer"] = (np.asarray(polygon["outer"]) * [sx, sy]).tolist()
            polygon["holes"] = [(np.asarray(r) * [sx, sy]).tolist() for r in polygon["holes"]]
    result = {
        "schema_version": VERSION,
        "document_id": document_id or hashlib.sha256(image.tobytes()).hexdigest(),
        "page_index": page_index,
        "image_size": [original_w, original_h],
        "model_version": model_version,
        "coordinate_system": "pixel_edges_xy_top_left",
        "inference_mode": "tiled" if tiled else "global",
        "canvas_size": size,
        "thresholds": deepcopy(thresholds),
        "original_to_work_scale_xy": [w / original_w, h / original_h],
        "instances": instances,
        "confidence_semantics": "model probability; calibrate on validation; not correctness probability",
    }
    validate_prediction(result)
    if return_maps:
        return result, {
            "mask_probabilities": masks,
            "boundary_probability": boundary,
            "room_center_probability": room_centers,
            "room_type_probabilities": types,
            "original_size": np.array([original_w, original_h]),
            "work_size": np.array([w, h]),
        }
    return result


def component_instances(name, label_map, channel, masks, types, min_pixels):
    instances = []
    counts = np.bincount(label_map.ravel())
    for label, bounds in enumerate(ndimage.find_objects(label_map), 1):
        if bounds is None or counts[label] < min_pixels:
            continue
        region = label_map[bounds] == label
        yy, xx = np.where(region)
        polygons = mask_polygons(region)
        if not polygons:
            continue
        offset = np.array([bounds[1].start, bounds[0].start])
        for polygon in polygons:
            polygon["outer"] = (np.asarray(polygon["outer"]) + offset).tolist()
            polygon["holes"] = [(np.asarray(ring) + offset).tolist() for ring in polygon["holes"]]
        item = {
            "class": name,
            "bbox": [
                float(xx.min() + offset[0]),
                float(yy.min() + offset[1]),
                float(xx.max() + 1 + offset[0]),
                float(yy.max() + 1 + offset[1]),
            ],
            "polygons": polygons,
            "confidence": float(masks[channel][bounds][region].mean()),
        }
        if name == "room":
            scores = types[(slice(None), *bounds)][:, region].mean(1)
            item["room_type"] = ROOM_TYPES[int(scores.argmax())] if scores.max() >= 0.5 else "unknown"
            item["room_type_confidence"] = float(scores.max())
        instances.append(item)
    return instances


def overlay(image, result):
    out = image.copy()
    colors = {
        "wall": (230, 70, 70),
        "room": (60, 180, 90),
        "door": (50, 100, 230),
        "window": (240, 180, 30),
        "stairs": (170, 60, 220),
    }
    for item in result["instances"]:
        color = colors[item["class"]]
        for polygon in item.get("polygons", []):
            cv2.polylines(out, [np.rint(polygon["outer"]).astype(np.int32)], True, color, 2)
        xa, ya, xb, yb = map(round, item["bbox"])
        cv2.rectangle(out, (xa, ya), (xb, yb), color, 1)
        cv2.putText(out, item["id"], (xa, max(12, ya)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return out
