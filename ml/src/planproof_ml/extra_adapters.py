"""Additional optional-source readers with explicit annotation coverage."""

from __future__ import annotations
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union

from .adapters import load_image, render_layout
from .schema import element_class, read_json, token
from .svg import read_svg


def template_group(path, root):
    for part in path.relative_to(root).parts:
        match = re.search(r"(?:template|layout|floorplan)[_-]?0*(\d+)", part, re.I)
        if match:
            return "template:" + match.group(1)
    parent = path.parent.name
    if parent.isdigit():
        return "template:" + str(int(parent))
    raise ValueError(
        f"Cannot identify SESYD's base layout for {path}; use normalized annotations with explicit template:NN group_id"
    )


def sesyd_samples(raw, class_map=None):
    xmls = sorted(raw.rglob("*.xml"))
    paths = xmls if xmls else sorted(raw.rglob("*.svg"))
    if not paths:
        raise ValueError("SESYD requires paired XML/SVG annotations and images")
    for path in paths:
        image_path = next(
            (path.with_suffix(s) for s in (".png", ".tif", ".tiff", ".jpg") if path.with_suffix(s).is_file()),
            None,
        )
        if image_path is None:
            raise ValueError(f"Missing paired SESYD image: {path}")
        image = load_image(image_path)
        features = []
        if path.suffix == ".svg":
            features, ignored, _, _ = read_svg(path, (image.shape[1], image.shape[0]))
            features = [f for f in features if f["class"] in {"door", "window", "stairs"}]
        else:
            content = path.read_text(encoding="utf-8-sig")
            if "<!ENTITY" in content.upper():
                raise ValueError("XML external entities are not supported")
            root = ET.fromstring(content)
            ignored = []
            for obj in root.iter():
                if obj.tag.split("}")[-1].lower() not in {"object", "symbol"}:
                    continue
                labels = [obj.get(k, "") for k in ("class", "label", "name", "model")]
                labels += [
                    c.text or ""
                    for c in obj
                    if c.tag.split("}")[-1].lower() in {"class", "label", "name", "model"}
                ]
                name = None
                for candidate in labels:
                    raw_label = Path(candidate).stem
                    mapped = (class_map or {}).get("mapping", {}).get(raw_label)
                    name = element_class(mapped or re.sub(r"\d+$", "", raw_label))
                    if name:
                        break
                if name not in {"door", "window", "stairs"}:
                    continue
                attrs = dict(obj.attrib)
                for child in obj.iter():
                    attrs.update(child.attrib)
                    local = child.tag.split("}")[-1].lower()
                    if child.text and local in {"xmin", "ymin", "xmax", "ymax", "x", "y", "width", "height"}:
                        attrs[local] = child.text
                if all(k in attrs for k in ("xmin", "ymin", "xmax", "ymax")):
                    bbox = [float(attrs[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
                elif all(k in attrs for k in ("x", "y", "width", "height")):
                    x, y, w, h = (float(attrs[k]) for k in ("x", "y", "width", "height"))
                    bbox = [x, y, x + w, y + h]
                else:
                    raise ValueError(f"Unsupported SESYD object box schema in {path}")
                features.append({"class": name, "bbox": bbox, "box_only": True})
        if not features:
            raise ValueError(
                f"No supported SESYD symbols in {path}; inspect publisher annotations before training"
            )
        yield {
            "key": path.relative_to(raw).with_suffix("").as_posix(),
            "group": template_group(path, raw),
            "image": image,
            "features": features,
            "ignore": ignored,
            "complete": [],
            "instance_complete": ["door", "window"],
            "annotation": path,
            "metadata": {"template_split": True, "supervision": "symbol boxes only"},
        }


def structured3d_samples(raw):
    paths = sorted(raw.rglob("annotation_3d.json"))
    if not paths:
        raise ValueError("Structured3D requires original scene_*/annotation_3d.json structure annotations")
    for path in paths:
        data = read_json(path)
        # Original releases use camelCase matrix keys; accept documented snake_case exports too.
        lj = data.get("lineJunctionMatrix", data.get("line_junction_matrix"))
        pl = data.get("planeLineMatrix", data.get("plane_line_matrix"))
        if lj is None or pl is None or not {"junctions", "planes", "semantics"} <= data.keys():
            raise ValueError(f"Unsupported Structured3D structure schema: {path}")
        vertices = np.array([j["coordinate"] for j in data["junctions"]], float)
        lj, pl = np.asarray(lj), np.asarray(pl)
        if lj.ndim != 2 or pl.ndim != 2 or lj.shape[1] != len(vertices) or pl.shape[1] != lj.shape[0]:
            raise ValueError("Inconsistent Structured3D incidence matrices")
        plane_types = {int(p.get("ID", i)): token(p.get("type", "")) for i, p in enumerate(data["planes"])}
        features = []
        for semantic in data["semantics"]:
            category = semantic["type"]
            if token(category) in {"door", "window", "outwall", "wall", "column"}:
                continue
            for plane in semantic.get("planeID", semantic.get("plane_ids", [])):
                if plane_types.get(int(plane)) != "floor":
                    continue
                lines = []
                for line in np.flatnonzero(pl[int(plane)]):
                    indices = np.flatnonzero(lj[line])
                    if len(indices) == 2:
                        lines.append(LineString(vertices[indices, :2]))
                for poly in polygonize(unary_union(lines)):
                    if poly.area > 0:
                        features.append(
                            {
                                "class": "room",
                                "room_type": category,
                                "outer": list(poly.exterior.coords),
                                "holes": [list(h.coords) for h in poly.interiors],
                            }
                        )
        if not features:
            raise ValueError(f"No room floor planes in {path}")
        image, features, meta = render_layout(features)
        match = re.search(r"scene_(\d+)", str(path))
        if not match:
            raise ValueError("Structured3D scene id is required for publisher split preservation")
        scene = int(match.group(1))
        split = "train" if scene < 3000 else "val" if scene < 3250 else "test"
        yield {
            "key": str(scene),
            "group": str(scene),
            "image": image,
            "features": features,
            "ignore": [],
            "complete": ["room"],
            "instance_complete": [],
            "publisher_split": split,
            "annotation": path,
            "metadata": {**meta, "supervision": "derived plan-view rooms; no invented wall/opening targets"},
        }
