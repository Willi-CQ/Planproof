"""Original-source readers. Unsupported schemas fail before training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import shape, Polygon

from .schema import MASK_CLASSES, element_class, token, read_json
from .svg import read_svg, render_strokes, numbers

ROOM_KEYS = {
    "living",
    "livingroom",
    "bedroom",
    "bathroom",
    "kitchen",
    "storage",
    "balcony",
    "balacony",
    "hallway",
    "corridor",
    "dining",
    "garage",
    "room",
    "utility",
    "other",
}


def load_image(path, max_pixels=64_000_000, max_size=None):
    with Image.open(path) as im:
        if im.width * im.height > max_pixels:
            raise ValueError(f"Image exceeds {max_pixels} pixels: {path}")
        if max_size is not None:
            im.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        # White compositing preserves transparent line drawings.
        rgba = im.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        return np.asarray(background.convert("RGB")).copy()


def polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        return [p for g in geom.geoms for p in polygons(g)]
    return []


def geometry_hash(features):
    """Canonical D4-normalized geometry fingerprint; coordinates, types, holes."""
    items = [f for f in features if "outer" in f and not f.get("weak")]
    if not items:
        return None
    allpts = np.vstack([f["outer"] for f in items])
    low, high = allpts.min(0), allpts.max(0)
    span = max(float((high - low).max()), 1e-6)
    midpoint = (low + high) / 2
    signatures = []
    for reflect in (1, -1):
        for k in range(4):
            angle = k * np.pi / 2
            matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ np.diag(
                [reflect, 1]
            )
            transformed = []
            for f in items:
                rings = [f["outer"], *f.get("holes", [])]
                coords = [((np.asarray(r) - midpoint) / span) @ matrix.T for r in rings]
                signature = []
                for c in coords:
                    rounded = np.round(c, 4)
                    rounded[np.abs(rounded) < 1e-6] = 0.0
                    signature.append(sorted(set(map(tuple, rounded.tolist()))))
                transformed.append((f["class"], str(f.get("room_type", "")), signature))
            signatures.append(json.dumps(sorted(transformed, key=str), sort_keys=True))
    return hashlib.sha256(min(signatures).encode()).hexdigest()


def room_features(plan):
    features = []
    for key, value in plan.items():
        label = element_class(key)
        if token(key) in ROOM_KEYS:
            label = "room"
        if label is None:
            continue
        geometries = value if isinstance(value, list) else [value]
        for item in geometries:
            if isinstance(item, dict) and "type" in item:
                geom = shape(item)
            elif hasattr(item, "geom_type"):
                geom = item
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                geom = Polygon(item)
            else:
                continue
            for poly in polygons(geom):
                if not poly.is_valid:
                    poly = poly.buffer(0)
                for valid in polygons(poly):
                    feature = {
                        "class": label,
                        "outer": list(valid.exterior.coords),
                        "holes": [list(r.coords) for r in valid.interiors],
                    }
                    if label == "room":
                        feature["room_type"] = key
                    features.append(feature)
    return features


def declared_coverage(plan):
    keys = {token(k) for k in plan}
    complete = [
        name
        for name in ("wall", "door", "window", "stairs")
        if name in keys or (name == "stairs" and "stair" in keys) or (name == "door" and "frontdoor" in keys)
    ]
    if keys & ROOM_KEYS:
        complete.append("room")
    return complete


def render_layout(features, size=1024):
    points = np.vstack([f["outer"] for f in features])
    low, high = points.min(0), points.max(0)
    scale = (size - 64) / max(float((high - low).max()), 1e-6)

    # Publisher layouts use Cartesian y-up; convert once and record it.
    def convert(ring):
        p = (np.asarray(ring) - low) * scale + 32
        p[:, 1] = size - p[:, 1]
        return p.tolist()

    transformed = []
    for f in features:
        transformed.append(
            {**f, "outer": convert(f["outer"]), "holes": [convert(r) for r in f.get("holes", [])]}
        )
    image = np.full((size, size, 3), 255, np.uint8)
    for f in transformed:
        pts = np.rint(f["outer"]).astype(np.int32)
        if f["class"] == "wall":
            cv2.fillPoly(image, [pts], (30, 30, 30))
            for hole in f.get("holes", []):
                cv2.fillPoly(image, [np.rint(hole).astype(np.int32)], (255, 255, 255))
        elif f["class"] == "room":
            cv2.polylines(image, [pts], True, (55, 55, 55), 1, cv2.LINE_AA)
        else:
            # Monochrome geometry, not semantic target colors.
            cv2.polylines(image, [pts], True, (0, 0, 0), 2, cv2.LINE_AA)
    return (
        image,
        transformed,
        {"layout_to_pixel": {"scale": scale, "input_min": low.tolist(), "y_reflection": True, "padding": 32}},
    )


def publisher_splits(source, raw):
    split_lookup = {}
    for name in ("train", "val", "test"):
        lists = list(raw.rglob(name + ".txt")) if source == "cubicasa5k" else []
        for split_file in lists:
            for line in split_file.read_text(encoding="utf-8-sig").splitlines():
                key = line.strip().replace("\\", "/").strip("/")
                if key:
                    if Path(key).suffix:
                        key = Path(key).parent.as_posix()
                    if key in split_lookup and split_lookup[key] != name:
                        raise ValueError("Publisher split files overlap")
                    split_lookup[key] = name
    return split_lookup


def svg_source_paths(source, raw):
    paths = sorted(raw.rglob("model.svg")) if source == "cubicasa5k" else sorted(raw.rglob("*.svg"))
    if source == "floorplancad":
        partitioned = [
            p for p in paths if any(part.startswith(("train-", "test-")) for part in p.relative_to(raw).parts)
        ]
        if partitioned:
            paths = partitioned
    return [p for p in paths if "coco_vis" not in p.parts]


def svg_samples(source, raw, class_map=None, svg_paths=None, split_lookup=None, raster_max_size=4096):
    if svg_paths is None:
        svg_paths = svg_source_paths(source, raw)
    if not svg_paths:
        raise ValueError(f"No annotated SVG files under {raw}")
    if split_lookup is None:
        split_lookup = publisher_splits(source, raw)
    for path in svg_paths:
        if "coco_vis" in path.parts:
            continue
        candidates = [
            path.parent / "F1_scaled.png",
            path.with_suffix(".png"),
            path.parent / "F1_original.png",
        ]
        candidates.extend(path.with_suffix(s) for s in (".jpg", ".jpeg", ".tif", ".tiff", ".PNG", ".JPG"))
        candidates.extend(
            path.parent.parent / d / path.with_suffix(".png").name for d in ("png", "pngs", "png_gt")
        )
        if source == "cubicasa5k":
            candidates = [path.parent / "F1_scaled.png"]
        image_path = next((p for p in candidates if p.is_file()), None)
        raster_size = None
        if image_path:
            with Image.open(image_path) as original:
                raster_size = original.size
            image = load_image(
                image_path,
                max_pixels=128_000_000 if source == "cubicasa5k" else 64_000_000,
                max_size=raster_max_size if source == "cubicasa5k" else None,
            )
            size = (image.shape[1], image.shape[0])
        elif source == "floorplancad":
            import xml.etree.ElementTree as ET

            root = ET.parse(path).getroot()
            view = numbers(root.attrib.get("viewBox", ""))
            if len(view) != 4:
                raise ValueError(f"SVG has no viewBox: {path}")
            ratio = min(1.0, 2048 / max(view[2:]))
            size = (max(1, round(view[2] * ratio)), max(1, round(view[3] * ratio)))
            image = None
        else:
            raise ValueError(f"Missing paired image for {path}")
        features, ignored, strokes, info = read_svg(
            path,
            size,
            class_map,
            coordinate_space="pixels" if source == "cubicasa5k" else "viewport",
            pixel_scale=(size[0] / raster_size[0], size[1] / raster_size[1])
            if source == "cubicasa5k"
            else (1, 1),
            collect_strokes=source != "cubicasa5k",
        )
        if raster_size is not None:
            info["original_raster_size"] = list(raster_size)
        if image is None:
            image = render_strokes(size, strokes)
        elif source == "floorplancad":
            # Ignore CAD layer/annotation colors; learn geometry from monochrome
            # linework rendered independently of semantic labels.
            image = render_strokes(size, strokes)
        if source == "floorplancad":
            info["input_rendering"] = "monochrome-svg-strokes"
        key = (
            path.parent.relative_to(raw).as_posix()
            if source == "cubicasa5k"
            else path.relative_to(raw).with_suffix("").as_posix()
        )
        matches = {v for k, v in split_lookup.items() if key == k or key.endswith("/" + k)}
        if len(matches) > 1:
            raise ValueError(f"Ambiguous publisher split: {path}")
        publisher_split = next(iter(matches), None)
        if source == "cubicasa5k" and not publisher_split:
            raise ValueError(f"Missing publisher split for {key}; use original split files")
        if source != "cubicasa5k":
            if any(part == "test" or part.startswith("test-") for part in path.parts):
                publisher_split = "test"
            elif "val" in path.parts:
                publisher_split = "val"
            elif any(part == "train" or part.startswith("train-") for part in path.parts):
                publisher_split = "train"
        complete = (
            list(MASK_CLASSES)
            if source == "cubicasa5k"
            else ([] if class_map else ["wall", "room", "door", "window"])
        )
        yield {
            "key": key,
            "group": "cad-family:" + path.stem.split("-")[0] if source == "floorplancad" else key,
            "image": image,
            "features": features,
            "ignore": ignored,
            "complete": complete,
            "instance_complete": class_map.get("complete", []) if class_map else complete,
            "publisher_split": publisher_split,
            "annotation": path,
            "metadata": {"svg": info},
        }


def resplan_samples(raw, trust_pickle=False):
    files = sorted(raw.rglob("*.pkl"))
    if len(files) != 1:
        raise ValueError("ResPlan requires one original ResPlan.pkl")
    if not trust_pickle:
        raise ValueError(
            "ResPlan uses Python pickle. Verify the publisher/hash, then explicitly pass --trust-pickle on the GPU machine"
        )
    with files[0].open("rb") as f:
        plans = pickle.load(f)
    split_files = list(raw.rglob("split.json"))
    lookup = {}
    if split_files:
        for split, ids in read_json(split_files[0]).items():
            if split in {"train", "val", "test"}:
                lookup.update({str(x): split for x in ids})
    seen_ids = set()
    for i, plan in enumerate(plans):
        plan_id = str(plan.get("id", i))
        if plan_id in seen_ids:
            raise ValueError(f"Duplicate ResPlan publisher ID: {plan_id}")
        seen_ids.add(plan_id)
        if lookup and plan_id not in lookup:  # publisher's augmentation list excluded by default
            continue
        features = room_features(plan)
        if not features:
            raise ValueError(f"No ResPlan geometry in plan {i}")
        image, features, meta = render_layout(features)
        yield {
            "key": plan_id,
            "group": "resplan:" + plan_id,
            "image": image,
            "features": features,
            "ignore": [],
            "complete": declared_coverage(plan),
            "instance_complete": declared_coverage(plan),
            "publisher_split": lookup.get(plan_id),
            "annotation": files[0],
            "metadata": {**meta, "publisher_plan_id": plan_id, "archive_index": i},
        }


def json_geometry(document):
    """GeoJSON features or TopologicPy JSON topology arrays."""
    features = []
    if isinstance(document, dict) and document.get("type") == "FeatureCollection":
        for item in document["features"]:
            props = item.get("properties") or {}
            key = props.get("class", props.get("type", props.get("name", "")))
            label = element_class(key)
            if token(key) in ROOM_KEYS or (label is None and props.get("room_type")):
                label = "room"
            if label:
                for poly in polygons(shape(item["geometry"])):
                    features.append(
                        {
                            "class": label,
                            "room_type": props.get("room_type", key),
                            "outer": list(poly.exterior.coords),
                            "holes": [list(r.coords) for r in poly.interiors],
                        }
                    )
        return features
    if isinstance(document, dict):
        features = room_features(document)
        if features:
            return features
    # TopologicPy exported JSON stores indexed vertices and faces with dictionaries.
    entries = document if isinstance(document, list) else [document]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        verts = entry.get("vertices", entry.get("Vertices", []))
        faces = entry.get("faces", entry.get("Faces", []))
        if not verts or not faces:
            continue
        coords = np.asarray(verts, dtype=float)[:, :2]
        dictionaries = entry.get("faceDictionaries", entry.get("dictionaries", []))
        for idx, face in enumerate(faces):
            indices = face.get("vertices", []) if isinstance(face, dict) else face
            props = (
                face.get("dictionary", {})
                if isinstance(face, dict)
                else (dictionaries[idx] if idx < len(dictionaries) else {})
            )
            key = props.get("type", props.get("class", props.get("name", "")))
            label = "room" if token(key) in ROOM_KEYS else element_class(key)
            if label and len(indices) >= 3:
                features.append({"class": label, "outer": coords[indices].tolist(), "room_type": key})
    return features


def modres_samples(raw):
    paths = [
        p
        for p in sorted(raw.rglob("*.json"))
        if "geometry" in str(p.parent).lower() and "2d" in str(p.parent).lower()
    ]
    if not paths:
        raise ValueError("No ModRes 2D geometry JSON files found under Geometry/2D/Geometry")
    for path in paths:
        features = json_geometry(read_json(path))
        if not features:
            raise ValueError(
                f"Unsupported ModRes geometry JSON in {path}; export annotated GeoJSON using publisher geometry tools"
            )
        image, features, meta = render_layout(features)
        key = path.stem
        # If a JSON export does not declare completeness, use its polygons only
        # as positives; do not invent negative labels for absent fields.
        doc = read_json(path)
        complete = doc.get("annotated_classes", []) if isinstance(doc, dict) else []
        yield {
            "key": key,
            "group": "resplan:" + key,
            "image": image,
            "features": features,
            "ignore": [],
            "complete": complete,
            "instance_complete": complete,
            "annotation": path,
            "metadata": meta,
        }


def houseexpo_samples(raw):
    files = sorted(raw.rglob("*.json"))
    for path in files:
        data = read_json(path)
        if not isinstance(data, dict) or "verts" not in data or "room_category" not in data:
            continue
        poly = Polygon(data["verts"])
        if not poly.is_valid:
            poly = poly.buffer(0)
        parts = polygons(poly)
        if not parts:
            raise ValueError(f"Invalid HouseExpo boundary: {path}")
        # Render the provided navigable boundary, not category boxes.
        features = [
            {
                "class": "room",
                "outer": list(part.exterior.coords),
                "holes": [list(r.coords) for r in part.interiors],
                "room_type": "unknown",
            }
            for part in parts
        ]
        image, _, meta = render_layout(features)
        low = np.array(meta["layout_to_pixel"]["input_min"])
        scale = meta["layout_to_pixel"]["scale"]
        weak = []
        for category, boxes in data["room_category"].items():
            b = np.asarray(boxes, dtype=float)
            if b.size == 4:
                b = b.reshape(1, 4)
            else:
                b = b.reshape(-1, 4)
            for x1, y1, x2, y2 in b:
                a = (np.array([x1, y1]) - low) * scale + 32
                z = (np.array([x2, y2]) - low) * scale + 32
                weak.append(
                    {
                        "class": "room",
                        "weak": True,
                        "room_type": category,
                        "bbox": [float(a[0]), float(1024 - z[1]), float(z[0]), float(1024 - a[1])],
                    }
                )
        if not weak:
            continue
        yield {
            "key": str(data.get("id", path.stem)),
            "group": str(data.get("id", path.stem)),
            "image": image,
            "features": weak,
            "ignore": [],
            "complete": [],
            "instance_complete": [],
            "annotation": path,
            "metadata": meta,
        }


RPLAN_TYPES = {
    0: "living",
    1: "bedroom",
    2: "kitchen",
    3: "bathroom",
    4: "dining",
    5: "bedroom",
    6: "bedroom",
    7: "other",
    8: "bedroom",
    9: "balcony",
    10: "hallway",
    11: "storage",
}


def rplan_samples(raw):
    for path in sorted(raw.rglob("*.png")):
        data = np.array(Image.open(path))
        if data.ndim != 3 or data.shape[2] != 4:
            raise ValueError(
                f"RPLAN requires original four-channel semantic PNG, not a rendered mirror: {path}"
            )
        boundary, categories, instances, inside = (data[:, :, i] for i in range(4))
        if np.max(categories) > 17 or not set(np.unique(inside)).issubset({0, 255}):
            raise ValueError(f"Unexpected RPLAN channel encoding: {path}; verify original channel schema")
        features = []
        # Original RPLAN RGBA: boundary, room type, room instance, interior indicator.
        for rid in np.unique(instances):
            if rid == 0:
                continue
            mask = (instances == rid) & (inside > 0) & (categories < 12)
            labels, counts = np.unique(categories[mask], return_counts=True)
            if not len(labels):
                continue
            rtype = RPLAN_TYPES.get(int(labels[counts.argmax()]), "other")
            from .spatial import mask_polygons

            for poly in mask_polygons(mask):
                features.append({"class": "room", "room_type": rtype, **poly})
        from .spatial import mask_polygons

        for poly in mask_polygons((boundary == 127) | (boundary == 255)):
            features.append({"class": "wall", **poly})
        if not features:
            raise ValueError(f"No original RPLAN room geometry in {path}")
        # Image is independently rendered from geometry: no semantic-channel input leakage.
        image = np.full((*data.shape[:2], 3), 255, np.uint8)
        image[(boundary == 127) | (boundary == 255)] = 20
        yield {
            "key": path.stem,
            "group": path.stem,
            "image": image,
            "features": features,
            "ignore": [],
            "complete": ["wall", "room"],
            "instance_complete": [],
            "annotation": path,
            "metadata": {
                "rplan_channels": "boundary,type,instance,interior; unsupported door codes excluded"
            },
        }


def normalized_samples(raw, source):
    """Explicit interchange for access-dependent annotation formats, not silent schema guessing.

    Each JSON contains image (relative), features, complete, instance_complete,
    group_id, and optional publisher_split. Source-specific publisher exporters
    can write this contract when native release formats differ.
    """
    count = 0
    for path in sorted(raw.rglob("*.annotation.json")):
        data = read_json(path)
        required = {"image", "features", "complete", "instance_complete", "group_id"}
        if required - data.keys():
            raise ValueError(f"Interchange fields missing in {path}: {required - data.keys()}")
        image_path = (path.parent / data["image"]).resolve()
        if not image_path.is_relative_to(raw.resolve()):
            raise ValueError("Interchange image path leaves raw source")
        if source == "sesyd" and not data["group_id"].startswith("template:"):
            raise ValueError("SESYD group_id must identify the base template, e.g. template:01")
        count += 1
        yield {
            "key": data.get("id", path.stem),
            "group": data["group_id"],
            "image": load_image(image_path),
            "features": data["features"],
            "ignore": data.get("ignore_regions", []),
            "complete": data["complete"],
            "instance_complete": data["instance_complete"],
            "publisher_split": data.get("publisher_split"),
            "annotation": path,
            "metadata": {"interchange_version": 1},
        }
    if not count:
        raise ValueError(
            f"{source} needs normalized *.annotation.json files; see the documented interchange schema"
        )


def iter_samples(source, raw, class_map=None, trust_pickle=False):
    raw = raw.resolve()
    if not raw.is_dir():
        raise ValueError(f"Raw source directory missing: {raw}")
    if source in {"cubicasa5k", "floorplancad", "cvcfp"}:
        yield from svg_samples(source, raw, class_map)
    elif source == "resplan":
        yield from resplan_samples(raw, trust_pickle)
    elif source == "modres":
        yield from modres_samples(raw)
    elif source == "houseexpo":
        yield from houseexpo_samples(raw)
    elif source == "rplan":
        yield from rplan_samples(raw)
    elif source == "structured3d":
        from .extra_adapters import structured3d_samples

        yield from structured3d_samples(raw)
    elif source == "sesyd":
        if list(raw.rglob("*.annotation.json")):
            yield from normalized_samples(raw, source)
        else:
            from .extra_adapters import sesyd_samples

            yield from sesyd_samples(raw, class_map)
    else:
        yield from normalized_samples(raw, source)
