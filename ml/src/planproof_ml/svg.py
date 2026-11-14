"""SVG geometry with inherited transforms; no scripts/resources are executed."""

from __future__ import annotations
from collections import defaultdict
import math
import re
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from svgpathtools import parse_path
from .schema import element_class, token

NUM = r"[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?"
PRIMITIVES = {"polygon", "polyline", "path", "rect", "line", "circle", "ellipse"}


def numbers(text):
    return [float(x) for x in re.findall(NUM, str(text))]


def transform(text):
    result = np.eye(3)
    for name, args in re.findall(r"([a-zA-Z]+)\s*\(([^)]*)\)", text or ""):
        v = numbers(args)
        m = np.eye(3)
        if name == "matrix" and len(v) == 6:
            a, b, c, d, e, f = v
            m = np.array([[a, c, e], [b, d, f], [0, 0, 1.0]])
        elif name == "translate" and len(v) in (1, 2):
            m[:2, 2] = [v[0], v[1] if len(v) > 1 else 0]
        elif name == "scale" and len(v) in (1, 2):
            m[0, 0], m[1, 1] = v[0], v[1] if len(v) > 1 else v[0]
        elif name == "rotate" and len(v) in (1, 3):
            angle = math.radians(v[0])
            m[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
            if len(v) == 3:
                center = np.array(v[1:])
                m[:2, 2] = center - m[:2, :2] @ center
        elif name in ("skewX", "skewY") and len(v) == 1:
            m[0 if name == "skewX" else 1, 1 if name == "skewX" else 0] = math.tan(math.radians(v[0]))
        else:
            raise ValueError(f"Unsupported SVG transform {name}({args})")
        result = result @ m
    return result


def primitive_points(node):
    tag = node.tag.split("}")[-1]
    a = node.attrib
    if tag in {"polygon", "polyline"}:
        return np.array(numbers(a.get("points", ""))).reshape(-1, 2), tag == "polygon"
    if tag == "rect":
        x, y, w, h = (float(a.get(k, 0)) for k in ("x", "y", "width", "height"))
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]), True
    if tag == "line":
        return np.array(
            [[float(a.get("x1", 0)), float(a.get("y1", 0))], [float(a.get("x2", 0)), float(a.get("y2", 0))]]
        ), False
    if tag in {"circle", "ellipse"}:
        cx, cy = float(a.get("cx", 0)), float(a.get("cy", 0))
        rx, ry = (
            (float(a.get("r", 0)),) * 2 if tag == "circle" else (float(a.get("rx", 0)), float(a.get("ry", 0)))
        )
        t = np.linspace(0, 2 * math.pi, 65)
        return np.column_stack((cx + rx * np.cos(t), cy + ry * np.sin(t))), True
    if tag == "path":
        path = parse_path(a.get("d", ""))
        if not len(path):
            return np.empty((0, 2)), False
        points = []
        for segment in path:
            count = max(2, min(128, int(segment.length() / 3) + 2))
            points.extend([[p.real, p.imag] for p in (segment.point(t) for t in np.linspace(0, 1, count))])
        return np.array(points), path.isclosed()
    return np.empty((0, 2)), False


def read_svg(
    path, image_size, class_map=None, coordinate_space="viewport", pixel_scale=(1, 1), collect_strokes=True
):
    if coordinate_space not in {"viewport", "pixels"}:
        raise ValueError("SVG coordinate space must be viewport or pixels")
    if len(pixel_scale) != 2 or not all(math.isfinite(x) and x > 0 for x in pixel_scale):
        raise ValueError("Pixel scale must contain two finite positive values")
    content = path.read_text(encoding="utf-8-sig")
    if "<!ENTITY" in content.upper():
        raise ValueError("SVG entity declarations are not supported")
    root = ET.fromstring(content)
    w, h = image_size
    view = numbers(root.attrib.get("viewBox", ""))
    if len(view) != 4:
        view = [
            0,
            0,
            numbers(root.attrib.get("width", str(w)))[0],
            numbers(root.attrib.get("height", str(h)))[0],
        ]
    vx, vy, vw, vh = view
    if min(vw, vh) <= 0:
        raise ValueError("Invalid SVG viewBox")
    sx, sy = w / vw, h / vh
    if root.attrib.get("preserveAspectRatio", "").strip() != "none":
        sx = sy = min(sx, sy)
    base = np.array([[sx, 0, (w - vw * sx) / 2 - vx * sx], [0, sy, (h - vh * sy) / 2 - vy * sy], [0, 0, 1]])
    if coordinate_space == "pixels":
        # CubiCasa polygons already use F1_scaled raster pixels. Its SVG viewport
        # can be smaller than the raster and must not rescale/center annotations.
        base = np.diag([*pixel_scale, 1.0])
    groups, ignored = defaultdict(list), defaultdict(list)
    strokes = []
    mapping = class_map["mapping"] if class_map else {}
    numeric = 0
    skipped = 0

    def walk(node, parent, owner=None, label=None, rtype=None, ignore=False):
        nonlocal numeric, skipped
        matrix = parent @ transform(node.attrib.get("transform", ""))
        tag = node.tag.split("}")[-1]
        if tag in {"text", "defs", "clipPath", "metadata", "title", "desc"}:
            return
        a = node.attrib
        classes = a.get("class", "").split()
        candidates = [
            a.get("data-label", ""),
            a.get("label", ""),
            a.get("category", ""),
            a.get("id", ""),
            *classes,
        ]
        new_label = next((element_class(x) for x in candidates if element_class(x)), None)
        new_type = None
        new_ignore = any(
            token(x) in {"elevator", "escalator", "railing", "outdoor", "opentobelow"} for x in candidates
        )
        if "Space" in classes and len(classes) > 1:
            if token(classes[1]) in {"outdoor", "opentobelow", "elevator"}:
                new_ignore, new_label = True, None
            else:
                new_label, new_type = "room", classes[1]
        semantic_id = a.get("semanticId", a.get("semantic-id"))
        instance_id = a.get("instanceId", a.get("instance-id", "-1"))
        if semantic_id is not None:
            numeric += 1
            if class_map is None:
                raise ValueError("Numeric semanticId requires a release-specific --class-map")
            new_label = mapping.get(semantic_id)
            owner = "cad:" + instance_id + ":" + semantic_id
            if instance_id == "-1":
                new_label = None
            label, rtype, ignore = new_label, None, False
        elif new_label or new_ignore:
            owner, label, rtype, ignore = str(id(node)), new_label, new_type, new_ignore
        if tag in PRIMITIVES:
            if not collect_strokes and not (owner and (label or ignore)):
                skipped += 1
                return
            points, closed = primitive_points(node)
            if len(points):
                points = (matrix @ np.column_stack((points, np.ones(len(points)))).T).T[:, :2]
                if collect_strokes:
                    strokes.append((points, closed))
                if owner and ignore:
                    ignored[owner].append((points, closed))
                elif owner and label:
                    groups[(owner, label, rtype)].append((points, closed))
        for child in node:
            walk(child, matrix, owner, label, rtype, ignore)

    walk(root, base)
    features = []
    for (_, label, rtype), shapes in groups.items():
        closed = [p for p, c in shapes if c and len(p) >= 3]
        if class_map is not None:
            pts = np.vstack([p for p, _ in shapes])
            low, high = pts.min(axis=0), pts.max(axis=0)
            if np.any(high <= low):
                continue
            features.append({"class": label, "bbox": [*low.tolist(), *high.tolist()], "box_only": True})
        else:
            if closed:
                poly = closed[0]
            else:
                pts = np.vstack([p for p, _ in shapes])
                if len(pts) < 3:
                    continue
                poly = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
            feature = {"class": label, "outer": poly.tolist()}
            if label == "room":
                feature["room_type"] = rtype or "unknown"
            features.append(feature)
    ignore_regions = []
    for shapes in ignored.values():
        closed = [p for p, c in shapes if c and len(p) >= 3]
        if closed:
            ignore_regions.append({"outer": closed[0].tolist()})
    if not features and not (class_map and numeric and class_map.get("complete")):
        raise ValueError(
            f"No supported annotations in {path}; inspect schema rather than making empty labels"
        )
    return (
        features,
        ignore_regions,
        strokes,
        {
            "recognized": len(features),
            "numeric": numeric,
            "coordinate_space": coordinate_space,
            "pixel_scale": list(pixel_scale),
            "unlabeled_primitives_skipped": skipped,
        },
    )


def render_strokes(size, strokes):
    w, h = size
    image = np.full((h, w, 3), 255, np.uint8)
    for points, closed in strokes:
        cv2.polylines(image, [np.rint(points).astype(np.int32)], closed, (0, 0, 0), 1, cv2.LINE_AA)
    return image
