from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    """Original pixel edges -> padded canvas. Origin top-left, xy order."""

    width: int
    height: int
    size: int
    scaled_width: int
    scaled_height: int
    left: int
    top: int

    @classmethod
    def fit(cls, width: int, height: int, size: int) -> "Frame":
        if min(width, height, size) <= 0:
            raise ValueError("Nonpositive image dimensions")
        scale = size / max(width, height)
        sw, sh = max(1, round(width * scale)), max(1, round(height * scale))
        return cls(width, height, size, sw, sh, (size - sw) // 2, (size - sh) // 2)

    def forward(self, xy):
        p = np.asarray(xy, dtype=np.float64)
        return p * [self.scaled_width / self.width, self.scaled_height / self.height] + [self.left, self.top]

    def inverse(self, xy):
        return (np.asarray(xy, dtype=np.float64) - [self.left, self.top]) / [
            self.scaled_width / self.width,
            self.scaled_height / self.height,
        ]

    def image(self, image: np.ndarray, fill=255) -> np.ndarray:
        result = np.full((self.size, self.size, *image.shape[2:]), fill, dtype=image.dtype)
        result[self.top : self.top + self.scaled_height, self.left : self.left + self.scaled_width] = (
            cv2.resize(image, (self.scaled_width, self.scaled_height), interpolation=cv2.INTER_AREA)
        )
        return result

    def mask(self, array: np.ndarray, fill=0) -> np.ndarray:
        result = np.full((self.size, self.size), fill, dtype=array.dtype)
        result[self.top : self.top + self.scaled_height, self.left : self.left + self.scaled_width] = (
            cv2.resize(array, (self.scaled_width, self.scaled_height), interpolation=cv2.INTER_NEAREST)
        )
        return result

    def restore(self, array: np.ndarray) -> np.ndarray:
        crop = array[self.top : self.top + self.scaled_height, self.left : self.left + self.scaled_width]
        return cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

    def metadata(self):
        return {
            "original_size": [self.width, self.height],
            "canvas_size": self.size,
            "scale_xy": [self.scaled_width / self.width, self.scaled_height / self.height],
            "offset_xy": [self.left, self.top],
            "coordinate_system": "pixel_edges_xy_top_left",
        }


def tile_origins(length: int, tile: int, overlap: float = 0.25) -> list[int]:
    if tile < 1 or not 0 <= overlap < 1:
        raise ValueError("Invalid tiling parameters")
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, max(1, round(tile * (1 - overlap)))))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def polygon_mask(shape, outer, holes=()):
    mask = np.zeros(shape, np.uint8)
    points = np.rint(np.asarray(outer)).astype(np.int32)
    if len(points) >= 3:
        cv2.fillPoly(mask, [points], 1)
        for hole in holes:
            cv2.fillPoly(mask, [np.rint(hole).astype(np.int32)], 0)
    return mask


def mask_polygons(mask: np.ndarray) -> list[dict]:
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    polygons = []
    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] != -1 or len(contour) < 3:
            continue
        outer = cv2.approxPolyDP(contour, 0.5, True).reshape(-1, 2).tolist()
        if len(outer) < 3:
            continue
        holes = []
        child = hierarchy[0][idx][2]
        while child != -1:
            hole = contours[child].reshape(-1, 2).tolist()
            if len(hole) >= 3:
                holes.append(hole)
            child = hierarchy[0][child][0]
        polygons.append({"outer": outer, "holes": holes})
    return polygons
