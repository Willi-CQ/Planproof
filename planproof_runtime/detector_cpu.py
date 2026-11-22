"""Evaluate the released stride-four door CNN using its original FP32 parameters."""
from functools import lru_cache

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from detection_boxes import image_array


@lru_cache(maxsize=1)
def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        if archive["version"].item() != 1:
            raise ValueError("Unsupported detector weight format")
        model = {}
        for group, count in (("stem", 3), ("mid", 2), ("deep", 3), ("fuse", 2)):
            for index in range(count):
                layer = f"{group}.{index}"
                scale = archive[f"{layer}.1.weight"] / np.sqrt(archive[f"{layer}.1.running_var"] + 1e-5)
                bias = archive[f"{layer}.1.bias"] - scale * archive[f"{layer}.1.running_mean"]
                kernel = archive[f"{layer}.0.weight"] * scale[:, None, None, None]
                # Input windows use (channel, row, column), as do PyTorch's OIHW kernels.
                model[layer] = (np.ascontiguousarray(kernel.reshape(len(bias), -1).T), bias)
        for head in ("heat", "geom"):
            model[head] = (archive[f"{head}.weight"][:, :, 0, 0].T.copy(), archive[f"{head}.bias"])
        return model, float(archive["decision_threshold"])


def convolve(data, layer, stride=1):
    kernel, bias = layer
    padded = np.pad(data, ((1, 1), (1, 1), (0, 0)))
    windows = sliding_window_view(padded, (3, 3), axis=(0, 1))[::stride, ::stride]
    height, width = windows.shape[:2]
    matrix = np.ascontiguousarray(windows.reshape(height * width, -1))
    activation = (matrix @ kernel).reshape(height, width, -1) + bias
    return np.maximum(activation, 0, out=activation)


def forward(model, image):
    h, w = image.shape
    x = np.pad((255 - image).astype(np.float32) / 255, ((0, (-h) % 16), (0, (-w) % 16)))
    a = convolve(x[:, :, None], model["stem.0"], 2)
    a = convolve(convolve(a, model["stem.1"], 2), model["stem.2"])
    b = convolve(convolve(a, model["mid.0"], 2), model["mid.1"])
    c = convolve(convolve(b, model["deep.0"], 2), model["deep.1"])
    c = convolve(c, model["deep.2"])
    merged = np.concatenate((a, b.repeat(2, axis=0).repeat(2, axis=1),
                             c.repeat(4, axis=0).repeat(4, axis=1)), axis=2)
    features = convolve(convolve(merged, model["fuse.0"]), model["fuse.1"])
    return tuple(features @ model[k][0] + model[k][1] for k in ("heat", "geom"))


def view_predictions(model, image, threshold):
    """Decode the single-door head consumed by infer.candidates in source pixels."""
    pixels = image_array(image)
    logits, geometry = forward(model, pixels)
    # infer.candidates discards every class except index 3 (single_door).
    heat = 1 / (1 + np.exp(-np.clip(logits[:, :, 3], -80, 80)))
    maxima = cv2.dilate(heat, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero((heat >= threshold) & (heat == maxima))
    h, w = pixels.shape
    detections = []
    for y, x in zip(ys, xs):
        logw, logh, dx, dy = geometry[y, x]
        cx = (int(x) + float(np.clip(dx, 0, 1))) * 4
        cy = (int(y) + float(np.clip(dy, 0, 1))) * 4
        if cx >= w or cy >= h:
            continue
        bw, bh = np.exp(np.clip([logw, logh], 0, 5)) * 4
        box = [max(0, cx - float(bw) / 2), max(0, cy - float(bh) / 2),
               min(w, cx + float(bw) / 2), min(h, cy + float(bh) / 2)]
        if box[2] > box[0] and box[3] > box[1]:
            detections.append(dict(category="single_door", bbox=box, confidence=float(heat[y, x])))
    return detections
