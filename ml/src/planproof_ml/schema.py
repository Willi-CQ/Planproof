"""Versioned image-space training and inference contracts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MASK_CLASSES = ("wall", "room", "door", "window", "stairs")
INSTANCE_CLASSES = ("room", "door", "window", "stairs")
ROOM_TYPES = (
    "kitchen",
    "living",
    "dining",
    "bedroom",
    "bathroom",
    "hallway",
    "storage",
    "garage",
    "balcony",
    "other",
)
IGNORE = 255
VERSION = "planproof-detection-v1"

# Do not import the legacy geometry aliases: unknown != wall, elevator != stairs.
ALIASES = {
    "walls": "wall",
    "door": "door",
    "frontdoor": "door",
    "singledoor": "door",
    "doubledoor": "door",
    "slidingdoor": "door",
    "foldingdoor": "door",
    "revolvingdoor": "door",
    "rollingdoor": "door",
    "window": "window",
    "baywindow": "window",
    "blindwindow": "window",
    "stairs": "stairs",
    "stair": "stairs",
    "staircase": "stairs",
    "wall": "wall",
    "room": "room",
    "space": "room",
}
ROOM_ALIASES = {
    "kitchen": "kitchen",
    "livingroom": "living",
    "living": "living",
    "lounge": "living",
    "dining": "dining",
    "diningroom": "dining",
    "eatingarea": "dining",
    "bedroom": "bedroom",
    "masterroom": "bedroom",
    "masterbedroom": "bedroom",
    "childroom": "bedroom",
    "guestroom": "bedroom",
    "bath": "bathroom",
    "bathroom": "bathroom",
    "toilet": "bathroom",
    "sauna": "bathroom",
    "hallway": "hallway",
    "hall": "hallway",
    "entry": "hallway",
    "corridor": "hallway",
    "draughtlobby": "hallway",
    "storage": "storage",
    "closet": "storage",
    "dressingroom": "storage",
    "garage": "garage",
    "carport": "garage",
    "balcony": "balcony",
    "balacony": "balcony",
}
UNKNOWN_TYPES = {"", "undefined", "unknown", "untyped", "room", "space", "userdefined"}


def token(value: str) -> str:
    return "".join(c for c in str(value).lower() if c.isalnum())


def element_class(value: str) -> str | None:
    return ALIASES.get(token(value))


def room_type(value: str | None) -> int:
    key = token(value or "")
    return IGNORE if key in UNKNOWN_TYPES else ROOM_TYPES.index(ROOM_ALIASES.get(key, "other"))


def schema_digest() -> str:
    return hashlib.sha256(
        json.dumps([VERSION, MASK_CLASSES, INSTANCE_CLASSES, ROOM_TYPES]).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_record(record: dict, root: Path | None = None) -> None:
    required = {"id", "source", "group_id", "family", "split", "image", "targets", "image_sha256"}
    if required - record.keys():
        raise ValueError(f"Missing sample fields: {sorted(required - record.keys())}")
    if record["split"] not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split for {record['id']}")
    if "stoa" in str(record["source"]).lower() or "junction-dataset" in str(record["source"]).lower():
        raise ValueError("Stoa challenge samples cannot enter a training manifest")
    if root:
        for field in ("image", "targets"):
            p = (root / record[field]).resolve()
            if not p.is_relative_to(root.resolve()) or not p.is_file():
                raise ValueError(f"Invalid prepared {field} path: {p}")


def validate_prediction(result: dict) -> None:
    if result["schema_version"] != VERSION:
        raise ValueError("Incompatible detection schema")
    width, height = result["image_size"]
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size")
    ids = set()
    for item in result["instances"]:
        if item["id"] in ids or item["class"] not in MASK_CLASSES:
            raise ValueError("Invalid class or duplicate instance id")
        ids.add(item["id"])
        if not 0 <= item["confidence"] <= 1:
            raise ValueError("Invalid confidence")
        x1, y1, x2, y2 = item["bbox"]
        if not all(math.isfinite(x) for x in (x1, y1, x2, y2)):
            raise ValueError("Nonfinite box")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Box outside original image: {item['bbox']}")
