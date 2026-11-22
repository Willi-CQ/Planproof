"""Image conversion, sliding windows and suppression shared by both CPU engines."""
import numpy as np
from PIL import Image


def image_array(image):
    paper = Image.new("RGBA", image.size, "white")
    paper.alpha_composite(image.convert("RGBA"))
    return np.array(paper.convert("L"), dtype=np.uint8)


def iou(box, other):
    width = max(0, min(box[2], other[2]) - max(box[0], other[0]))
    height = max(0, min(box[3], other[3]) - max(box[1], other[1]))
    intersection = width * height
    area = (box[2] - box[0]) * (box[3] - box[1])
    other_area = (other[2] - other[0]) * (other[3] - other[1])
    return intersection / max(1e-9, area + other_area - intersection)


def merge(rows, threshold=.40):
    kept = []
    for row in sorted(rows, key=lambda r: -r["confidence"]):
        if not any(row["category"] == k["category"] and iou(row["bbox"], k["bbox"]) > threshold for k in kept):
            kept.append(row)
    return kept


def positions(length, window, overlap):
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, round(window * (1 - overlap))))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts
