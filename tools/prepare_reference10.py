"""Export ten existing validation drawings for local, iterative failure analysis.

Run in the existing ML container, mounting its prepared data as --root. This
copies already downloaded inputs; it never acquires data, trains or tunes models.
The exported pack is development data, not a new unbiased test benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

CLASSES = ("wall", "room", "door", "window", "stairs")
OBJECTS = ("door", "window", "stairs")
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
COLORS = {
    "wall": (220, 60, 60),
    "room": (35, 160, 80),
    "door": (30, 95, 220),
    "window": (210, 145, 15),
    "stairs": (170, 45, 190),
}
SOURCES = {
    "cubicasa5k": "processed/cubicasa-clean-manifest.jsonl",
    "resplan": "resplan-canonical-processed/resplan-clean-manifest.jsonl",
    "floorplancad": "floorplancad-processed/manifest.jsonl",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    )


def rank(record):
    return hashlib.sha256(
        ("planproof-reference10-v1:" + record["id"]).encode()
    ).hexdigest()


def target_stats(record, base):
    with np.load(base / record["targets"], allow_pickle=False) as data:
        objects = data["objects"]
        rooms = int(np.count_nonzero(np.unique(data["instances"][0])))
        counts = {
            name: int((objects[:, 0] == i).sum()) for i, name in enumerate(OBJECTS)
        }
    return {"room": rooms, **counts}


def choose(records, base, source):
    candidates = sorted(records, key=rank)
    stair = [r for r in candidates if "stairs" in r.get("positive_classes", [])]
    without = [r for r in candidates if "stairs" not in r.get("positive_classes", [])]
    chosen = []
    groups = set()

    def take(rows, reason):
        for record in rows:
            if record["group_id"] not in groups:
                chosen.append((record, reason))
                groups.add(record["group_id"])
                return
        raise ValueError(
            f"Insufficient eligible distinct groups for {source}: {reason}"
        )

    if source != "floorplancad":
        take(stair, "stair-positive, hash-ranked selection")
        take(stair, "second distinct stair-positive drawing")
        pool = [(r, target_stats(r, base)) for r in without[:16]]
        pool.sort(key=lambda item: (sum(item[1].values()), rank(item[0])))
        take(
            [r for r, _ in pool], "lower annotation count within fixed hash-ranked pool"
        )
        take(
            [r for r, _ in reversed(pool)],
            "higher annotation count within same fixed pool",
        )
    else:
        take(stair, "CAD stair-positive drawing")
        pool = [(r, target_stats(r, base)) for r in candidates[:32]]
        pool.sort(key=lambda item: (sum(item[1].values()), rank(item[0])), reverse=True)
        take(
            [r for r, _ in pool],
            "object-dense CAD drawing within fixed hash-ranked pool",
        )
    return chosen


def polygons(mask):
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return []
    result = []
    for i, contour in enumerate(contours):
        if hierarchy[0, i, 3] != -1 or len(contour) < 3:
            continue
        holes = []
        child = int(hierarchy[0, i, 2])
        while child != -1:
            if len(contours[child]) >= 3:
                holes.append(contours[child][:, 0].tolist())
            child = int(hierarchy[0, child, 0])
        result.append({"outer": contour[:, 0].tolist(), "holes": holes})
    return result


def export_case(record, base, out, case_id, reason, manifest_sha):
    directory = out / "cases" / case_id
    directory.mkdir(parents=True)
    image_path, targets_path = base / record["image"], base / record["targets"]
    if digest(targets_path) != record["targets_sha256"]:
        raise ValueError("Source annotation checksum mismatch")
    image = np.asarray(Image.open(image_path).convert("RGB"))
    decoded = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
    if decoded != record["image_sha256"]:
        raise ValueError("Source raster checksum mismatch")
    shutil.copy2(image_path, directory / "plan.png")
    shutil.copy2(targets_path, directory / "targets.npz")
    with np.load(targets_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    masks, valid = arrays["masks"], arrays["valid"]
    if masks.shape != valid.shape or masks.shape[1:] != image.shape[:2]:
        raise ValueError("Targets/image shape mismatch")
    coverage = {
        name: {
            "mask_annotated": bool(valid[i].any()),
            "mask_valid_fraction": float(valid[i].mean()),
        }
        for i, name in enumerate(CLASSES)
    }
    counts = {"wall": None, "room": None}
    instances = []
    for i, name in enumerate(OBJECTS):
        coverage[name]["object_complete"] = bool(arrays["object_complete"][i])
        rows = arrays["objects"][arrays["objects"][:, 0] == i]
        counts[name] = len(rows) if arrays["object_complete"][i] else None
        for j, row in enumerate(rows):
            instances.append(
                {
                    "id": f"{case_id}-{name}-{j:03}",
                    "class": name,
                    "bbox": row[1:].tolist(),
                    "annotation_kind": "object_box",
                }
            )
    room_ids = np.unique(arrays["instances"][0])
    room_ids = room_ids[room_ids > 0]
    if valid[1].any():
        counts["room"] = len(room_ids)
    for value in room_ids:
        region = (arrays["instances"][0] == value) & (valid[1] > 0)
        yy, xx = np.where(region)
        if not len(xx):
            continue
        category = arrays["types"][region]
        category = category[category != 255]
        room_type = (
            ROOM_TYPES[int(np.bincount(category).argmax())]
            if len(category)
            else "unknown"
        )
        instances.append(
            {
                "id": f"{case_id}-room-{int(value):03}",
                "class": "room",
                "bbox": [
                    int(xx.min()),
                    int(yy.min()),
                    int(xx.max() + 1),
                    int(yy.max() + 1),
                ],
                "polygons": polygons(region),
                "area_px2": int(region.sum()),
                "room_type": room_type,
                "annotation_kind": "instance_mask",
            }
        )
    for name, channel in zip(CLASSES, masks):
        Image.fromarray(channel * 255).save(directory / f"mask-{name}.png")
    Image.fromarray(arrays["pixel_valid"] * 255).save(directory / "valid-pixels.png")
    overlay = image.copy()
    for name, i in (("wall", 0), ("room", 1)):
        region = (masks[i] > 0) & (valid[i] > 0)
        overlay[region] = (
            0.72 * overlay[region] + 0.28 * np.array(COLORS[name])
        ).astype(np.uint8)
    canvas = Image.fromarray(overlay)
    draw = ImageDraw.Draw(canvas)
    for item in instances:
        name = item["class"]
        if name == "room":
            for ring in item["polygons"]:
                draw.line(
                    [tuple(p) for p in ring["outer"]] + [tuple(ring["outer"][0])],
                    fill=COLORS[name],
                    width=max(1, image.shape[1] // 900),
                )
        else:
            draw.rectangle(
                item["bbox"], outline=COLORS[name], width=max(2, image.shape[1] // 700)
            )
        x, y = item["bbox"][:2]
        draw.text(
            (x, max(0, y - 12)),
            item["id"].rsplit(case_id + "-", 1)[-1],
            fill=COLORS[name],
        )
    canvas.save(directory / "labels-overlay.png")
    annotation = {
        "schema_version": "planproof-reference-v1",
        "case_id": case_id,
        "image_size": [image.shape[1], image.shape[0]],
        "coordinate_system": "pixel_edges_xy_top_left",
        "classes": list(CLASSES),
        "coverage": coverage,
        "counts": counts,
        "instances": instances,
        "dense_targets": "targets.npz",
        "mask_order": list(CLASSES),
        "instance_order": ["room", "door", "window", "stairs"],
        "object_box_order": list(OBJECTS),
        "metric_calibration": None,
        "notes": [
            "Contours are display approximations; NPZ masks are the exact prepared labels.",
            "Wall mask has no physical-wall instance count.",
            "Pixel areas are annotated areas, not square metres.",
            "Labels come from the existing audited publisher-data adapters; manual review is still needed.",
        ],
    }
    write_json(directory / "ground-truth.json", annotation)
    write_json(directory / "source-record.json", record)
    return {
        "case_id": case_id,
        "source": record["source"],
        "source_id": record["id"],
        "source_key": record["source_key"],
        "group_id": record["group_id"],
        "source_split": record["split"],
        "publisher_split": record.get("publisher_split"),
        "source_manifest_sha256": manifest_sha,
        "selection_reason": reason,
        "geometry_hash": record.get("geometry_hash"),
        "image_sha256": record["image_sha256"],
        "image_hash_mode": record.get("image_hash_mode"),
        "counts": counts,
        "coverage": coverage,
        "directory": f"cases/{case_id}",
        "image_size": annotation["image_size"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to replace an existing frozen reference pack")
    cv2.setNumThreads(1)
    manifests = {}
    blocked_groups, blocked_images, blocked_geometry = set(), set(), set()
    for source, relative in SOURCES.items():
        path = args.root / relative
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        manifests[source] = (path, rows)
        for r in rows:
            if r["split"] != "val":
                blocked_groups.add(r["group_id"])
                blocked_images.add(r["image_sha256"])
                if r.get("geometry_hash"):
                    blocked_geometry.add(r["geometry_hash"])
    selected = []
    for source, (path, rows) in manifests.items():
        eligible = [
            r
            for r in rows
            if r["split"] == "val"
            and r.get("publisher_split") != "test"
            and r["group_id"] not in blocked_groups
            and r["image_sha256"] not in blocked_images
            and (
                not r.get("geometry_hash") or r["geometry_hash"] not in blocked_geometry
            )
        ]
        selected.extend(
            (path, r, reason) for r, reason in choose(eligible, path.parent, source)
        )
    assert len(selected) == 10
    assert len({r["group_id"] for _, r, _ in selected}) == 10
    assert len({r["image_sha256"] for _, r, _ in selected}) == 10
    args.out.mkdir(parents=True)
    cases = []
    for index, (path, record, reason) in enumerate(selected, 1):
        case_id = f"P{index:02}-{record['source']}"
        cases.append(
            export_case(record, path.parent, args.out, case_id, reason, digest(path))
        )
        print(
            json.dumps({"exported": case_id, "counts": cases[-1]["counts"]}), flush=True
        )
    summary = {
        "schema_version": "planproof-reference10-v1",
        "purpose": "iterative development/failure analysis",
        "case_count": 10,
        "selection_uses_predictions": False,
        "cases": cases,
        "sources": dict(Counter(c["source"] for c in cases)),
        "separation_checks": {
            "all_project_validation": True,
            "publisher_test_excluded": True,
            "training_and_test_groups_excluded": True,
            "training_and_test_exact_image_hashes_excluded": True,
            "training_and_test_exact_geometry_hashes_excluded": True,
        },
        "warnings": [
            "Stair-enriched diagnostic selection, not representative or unseen final test data.",
            "ResPlan and CAD images are prepared renders of existing source drawings, not new hand-made fixtures.",
            "CAD has object boxes but no wall/room masks. Missing coverage is N/A, never zero.",
            "Do not tune on this pack then describe its scores as unbiased test accuracy.",
            "Deterministic algorithms are not implemented yet; their performance is not measured.",
        ],
    }
    write_json(args.out / "manifest.json", summary)
    notices = args.root / "releases/corrected-3500-global"
    for name in ("DATA_SOURCES.txt", "ResPlan-LICENSE.txt"):
        if (notices / name).is_file():
            shutil.copy2(notices / name, args.out / name)
    write_json(
        args.out / "SHA256SUMS.json",
        {
            p.relative_to(args.out).as_posix(): digest(p)
            for p in sorted(args.out.rglob("*"))
            if p.is_file()
        },
    )
    print(
        json.dumps({"complete": True, "cases": len(cases), "out": str(args.out)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
