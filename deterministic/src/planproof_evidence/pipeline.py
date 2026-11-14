"""Standalone analysis/crop API. Only the evaluator, outside this package, sees labels."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from shapely.affinity import scale as scale_geometry
from shapely.geometry import Polygon, box, shape, mapping

from .compare import aggregates, compare_sources, contextual_box, room_relations
from .context import agent_context
from .contracts import (
    CLASSES,
    COORDINATES,
    SCHEMA,
    Config,
    item_geometry,
    polygon_json,
    polygons_of,
    sha256,
    stable_id,
    validate_bbox,
    validate_document,
    validate_pixels,
    write_json,
)
from .geometry import (
    opening_attachment,
    snap_polygon,
    node_graph,
    pair_boundaries,
    skeleton_lines,
    project_opening,
    net_regions,
)
from .inputs import load_pages
from .measurements import associate_dimension, classify_room_text, quantities
from .pixels import extract_pixels, vectorize_pixels

COLORS = {
    "wall": "#e38b25",
    "room": "#3689dd",
    "door": "#df3b53",
    "window": "#00a59a",
    "stairs": "#9d54cb",
}


def render_overlay(image, instances):
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for index, instance in enumerate(instances, 1):
        color = COLORS.get(instance.get("class"), "#888888")
        for polygon in instance.get("polygons", []):
            for ring in [polygon["outer"], *polygon.get("holes", [])]:
                draw.line([tuple(p) for p in ring], fill=color, width=2)
        if not instance.get("polygons"):
            draw.rectangle(instance["bbox"], outline=color, width=2)
        x, y = instance["bbox"][:2]
        draw.text(
            (x, y),
            instance.get("overlay_label", str(index)) + ":" + instance.get("class", "?"),
            fill=color,
            stroke_width=1,
            stroke_fill="white",
        )
    return canvas


def _cached_extract(image, config, cache, adapter):
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = cache / "pixels.json"
    mask_names = ("ink", "wall_mask", "skeleton_mask")
    if adapter is None and meta_path.exists():
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        checks = saved.pop("mask_hashes")
        if all(sha256(cache / (key + ".png")) == checks[key] for key in mask_names):
            for key in mask_names:
                saved[key] = (np.asarray(Image.open(cache / (key + ".png"))) > 0).astype(np.uint8)
            for c in saved["candidates"]:
                c["geometry"] = shape(c["geometry"])
            for pair in saved["pairs"]:
                pair["polygon"] = shape(pair["polygon"])
            return saved, True
    data = (
        vectorize_pixels(image, adapter(image, config), config)
        if adapter
        else extract_pixels(image, config)
    )
    serializable = {k: v for k, v in data.items() if k not in mask_names}
    serializable["candidates"] = [
        {**c, "geometry": mapping(c["geometry"])} for c in data["candidates"]
    ]
    serializable["pairs"] = [{**p, "polygon": mapping(p["polygon"])} for p in data["pairs"]]
    for key in mask_names:
        Image.fromarray(data[key].astype(np.uint8) * 255).save(cache / (key + ".png"))
    serializable["mask_hashes"] = {key: sha256(cache / (key + ".png")) for key in mask_names}
    write_json(meta_path, serializable)
    return data, False


def _ml_sources(binding, frames):
    if binding is None:
        return []
    if isinstance(binding, (str, Path)):
        path = Path(binding)
        value = json.loads(path.read_text(encoding="utf-8"))
        base = path.resolve().parent
    else:
        raise ValueError("ML import requires a binding file, never an unbound instance list")
    sources = []
    names = set()
    for source in value["sources"]:
        name = source["name"]
        key = (name, source["page_index"])
        if key in names:
            raise ValueError("Duplicate model/page binding")
        names.add(key)
        frame = frames.get(source["page_index"])
        if frame is None or source["raster_sha256"] != frame.raster_sha256:
            raise ValueError("ML raster identity mismatch")
        if source.get("image_size") != frame.image_size:
            raise ValueError("ML binding dimensions mismatch")
        for field in ("prediction_sha256", "model_sha256", "profile_sha256"):
            if len(source.get(field, "")) != 64 or any(
                c not in "0123456789abcdef" for c in source[field]
            ):
                raise ValueError("Missing/invalid ML identity: " + field)
        path = (base / source["prediction_path"]).resolve()
        if sha256(path) != source["prediction_sha256"]:
            raise ValueError("ML prediction hash mismatch")
        prediction = json.loads(path.read_text(encoding="utf-8"))
        if (
            prediction["schema_version"] != "planproof-detection-v1"
            or prediction["page_index"] != frame.page_index
            or prediction["image_size"] != frame.image_size
            or prediction["coordinate_system"] != COORDINATES
            or prediction["document_id"] != source["document_id"]
        ):
            raise ValueError("ML prediction frame/schema mismatch")
        ids = set()
        normalized = {}
        for item in prediction["instances"]:
            if item["id"] in ids or item["class"] not in CLASSES:
                raise ValueError("Duplicate/invalid ML instance")
            ids.add(item["id"])
            validate_bbox(item["bbox"], frame.image_size)
            # Self-touching raster contours occur in the frozen ML exports.
            # Preserve the raw prediction and record an explicit normalization.
            from shapely import make_valid
            from shapely.validation import explain_validity

            polygons, repairs = [], []
            for polygon in item.get("polygons", []):
                for ring in [polygon["outer"], *polygon.get("holes", [])]:
                    values = np.asarray(ring, dtype=float)
                    if (
                        values.ndim != 2
                        or values.shape[1] != 2
                        or len(values) < 3
                        or not np.isfinite(values).all()
                    ):
                        raise ValueError("Malformed ML ring")
                    if (
                        values[:, 0].min() < 0
                        or values[:, 1].min() < 0
                        or values[:, 0].max() > frame.image_size[0]
                        or values[:, 1].max() > frame.image_size[1]
                    ):
                        raise ValueError("ML ring outside bound raster")
                g = Polygon(polygon["outer"], polygon.get("holes", []))
                if not g.is_valid:
                    repairs.append(explain_validity(g))
                    g = make_valid(g)
                polygons.extend(polygons_of(g))
            if item.get("polygons") and not polygons:
                raise ValueError("ML polygon has no usable area after explicit repair")
            normalized[item["id"]] = {
                "polygons": [polygon_json(p) for p in polygons],
                "repairs": repairs,
            }
            if not math.isfinite(item["confidence"]) or not 0 <= item["confidence"] <= 1:
                raise ValueError("Invalid ML score")
        sources.append({"binding": source, "prediction": prediction, "normalized": normalized})
    return sources


def _observation(
    frame,
    namespace,
    kind,
    geometry,
    method,
    *,
    parents=None,
    mode="independent",
    family="classical_pixels",
    **extra,
):
    bounds = list(geometry.bounds)
    validate_bbox(bounds, frame.image_size)
    polygons = [polygon_json(p) for p in polygons_of(geometry)]
    return {
        "id": stable_id(
            f"p{frame.page_index}:{namespace}",
            {
                "document_id": frame.document_id,
                "method": method,
                "bbox": bounds,
                "polygons": polygons,
                "parents": parents or [],
            },
        ),
        "kind": kind,
        "frame_id": frame.id,
        "page_index": frame.page_index,
        "bbox": bounds,
        "polygons": polygons,
        "source_family": family,
        "mode": mode,
        "method": method,
        "document_id": frame.document_id,
        "dependency_ids": parents or [],
        "warnings": [],
        **extra,
    }


def analyze(
    input_path, out_dir, *, ml_predictions=None, ml_maps=None, config=None, pixel_adapter=None
):
    """Write evidence, immutable raw extraction, comparisons, overlays and scoped crops.

    pixel_adapter(image_rgb, Config) may replace the reference adapter. Its result
    is in its recorded working frame; its masks must never come from evaluation labels.
    """
    if ml_maps is not None:
        raise ValueError(
            "Dense ML maps are not yet supported; use bound instance polygons, not unrecorded resampling"
        )
    config = config if isinstance(config, Config) else Config(**(config or {}))
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    loaded = list(load_pages(input_path, config))
    input_hash = sha256(input_path)
    if not loaded or any(frame.source_sha256 != input_hash for _, frame, _, _, _ in loaded):
        raise ValueError("Input changed while rendering or has no pages")
    frames = {f.page_index: f for _, f, _, _, _ in loaded}
    imported = _ml_sources(ml_predictions, frames)
    document = {
        "schema_version": SCHEMA,
        "algorithm_version": config.version,
        "config": asdict(config),
        "document_id": loaded[0][1].document_id,
        "source_sha256": sha256(input_path),
        "implementation_hashes": {
            name: sha256(Path(__file__).with_name(name + ".py"))
            for name in (
                "contracts",
                "geometry",
                "pixels",
                "pipeline",
                "inputs",
                "measurements",
                "compare",
                "context",
            )
        },
        "pages": [],
        "observations": [],
        "comparisons": [],
        "aggregates": [],
        "calibrations": [],
        "artifacts": [],
        "warnings": ["reference classical adapter; candidates are not verified physical facts"],
    }

    def artifact(path, kind, frame, **metadata):
        document["artifacts"].append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "kind": kind,
                "frame_id": frame.id,
                **metadata,
            }
        )

    for image, frame, native_lines, spans, native_coverage in loaded:
        pdir = root / f"page-{frame.page_index:03d}"
        pdir.mkdir(exist_ok=True)
        raster = pdir / "source.png"
        Image.fromarray(image).save(raster)
        artifact(raster, "clean_canonical_raster", frame, image_size=frame.image_size)
        extraction_code = {
            name: document["implementation_hashes"][name]
            for name in ("pixels", "geometry", "contracts")
        }
        cache_key = stable_id(
            "pixels", [frame.raster_sha256, asdict(config), extraction_code]
        ).split(":")[1]
        data, hit = _cached_extract(image, config, root / ".cache" / cache_key, pixel_adapter)
        validate_pixels(data, frame.image_size)
        ink_field = distance_transform_edt(data["ink"] == 0, return_indices=True)
        sx, sy = (
            frame.image_size[0] / data["working_size"][0],
            frame.image_size[1] / data["working_size"][1],
        )
        spans = list(spans)
        for span in data.get("text_spans", []):
            if span.get("source_family") != "ocr":
                raise ValueError("Pixel-adapter text must retain OCR provenance")
            bbox = [v * (sx if i % 2 == 0 else sy) for i, v in enumerate(span["bbox"])]
            validate_bbox(bbox, frame.image_size)
            spans.append({**span, "bbox": bbox})
        raw_path = pdir / "pixel-evidence.json"
        raw = {
            "schema_version": "planproof-pixels-v1",
            "frame_id": frame.id,
            "working_size": data["working_size"],
            "working_to_canonical": data["working_to_canonical"],
            "stroke_px": data["stroke_px"],
            "segments": data["raw_segments"],
            "merged_segments": data["merged"],
            "availability": data["availability"],
            "warnings": data["warnings"],
            "text_spans": spans,
            "native_segments": native_lines,
            "native_coverage": native_coverage,
            "unclassified_patterns": data.get("unclassified_patterns", []),
        }
        write_json(raw_path, raw)
        artifact(raw_path, "pixel_evidence", frame)
        pixel = _observation(
            frame,
            "pixels-" + cache_key,
            "pixel_extraction",
            box(0, 0, *frame.image_size),
            config.version,
            features={"working_size": data["working_size"], "availability": data["availability"]},
        )
        document["observations"].append(pixel)
        for name in ("ink", "wall_mask", "skeleton_mask"):
            path = pdir / (name + ".png")
            Image.fromarray(data[name] * 255).save(path)
            artifact(
                path,
                name,
                frame,
                image_size=data["working_size"],
                working_to_canonical=data["working_to_canonical"],
            )
        observations = []
        calibration = {
            "page_index": frame.page_index,
            "scope": "page",
            "status": "unresolved",
            "reason": "no two verified, explicitly unit-bearing dimension anchors",
        }
        document["calibrations"].append(calibration)
        for c in data["candidates"]:
            g = scale_geometry(c["geometry"], sx, sy, origin=(0, 0))
            o = _observation(
                frame,
                "classical",
                c["class"] + "_candidate",
                g,
                c["method"],
                parents=[pixel["id"]],
                **{"class": c["class"]},
                features=c["features"],
                feature_frame={
                    "working_size": data["working_size"],
                    "working_to_canonical": data["working_to_canonical"],
                },
                score={
                    "value": c["support"],
                    "semantics": "normalized rule support; not calibrated correctness",
                },
                status=c["status"],
                warnings=c["warnings"],
                quantities=quantities(g),
            )
            if c["class"] == "room":
                o["room_type"] = classify_room_text(g, spans)
            observations.append(o)
            o["overlay_label"] = "C" + str(len(observations))
        # Candidate boundary pairs and medial graph stay available even when not classified.
        vector_path = pdir / "vector-evidence.json"
        vector = {
            "schema_version": "planproof-vectors-v1",
            "frame_id": frame.id,
            "working_to_canonical": data["working_to_canonical"],
            "working_size": data["working_size"],
            "boundary_pairs": [{**p, "polygon": polygon_json(p["polygon"])} for p in data["pairs"]],
            "accepted_pair_count": data["accepted_pair_count"],
            "centerlines": data["centerlines"],
            "graph": data["graph"],
            "gap_hypotheses": data["gaps"],
            "gap_policy": "hypotheses retained, never silently inserted",
            "snap_policy": "ML-assisted only; raw geometry retained",
        }
        center_regions = [Polygon(p["outer"], p["holes"]) for p in data["graph"]["regions"]]
        wall_local = [c["geometry"] for c in data["candidates"] if c["class"] == "wall"]
        vector["net_graph_regions"] = [
            polygon_json(p) for p in net_regions(center_regions, wall_local)
        ]
        vector["region_definition"] = (
            "centerline polygonization minus observed wall bodies; free-space regions remain separate observations"
        )
        if native_lines:
            native_local = [
                [[x / sx, y / sy] for x, y in line] for line in native_lines[: config.max_segments]
            ]
            native_pairs, truncated = pair_boundaries(
                native_local, data["stroke_px"], config.angle_deg, config.max_pairs
            )
            vector["native"] = {
                "source_family": "native_pdf",
                "mode": "independent",
                "working_to_canonical": data["working_to_canonical"],
                "boundary_pairs": [
                    {**pair, "polygon": polygon_json(pair["polygon"])} for pair in native_pairs
                ],
                "graph": node_graph(native_local),
                "partial": truncated or len(native_lines) > config.max_segments,
                "semantics": "native line topology includes dimensions/furniture; not physical wall topology",
            }
        write_json(vector_path, vector)
        artifact(vector_path, "vector_evidence", frame)
        walls = [o for o in observations if o["class"] == "wall"]
        wall_geometries = [item_geometry(o) for o in walls]
        for o in observations:
            if o["class"] in ("door", "window"):
                association = opening_attachment(
                    item_geometry(o), wall_geometries, data["stroke_px"] * max(sx, sy) * 3
                )
                for a in association["alternatives"]:
                    a["wall_id"] = walls[a.pop("wall_index")]["id"]
                o["wall_attachment"] = association
                if data["pairs"]:
                    local = scale_geometry(item_geometry(o), 1 / sx, 1 / sy, origin=(0, 0))
                    nearest = min(data["pairs"], key=lambda p: p["polygon"].distance(local))
                    if nearest["polygon"].distance(local) <= 3 * data["stroke_px"]:
                        o["local_axis_projection"] = {
                            **project_opening(local, nearest["centerline"]),
                            "working_to_canonical": data["working_to_canonical"],
                        }
        document["observations"].extend(observations)
        native_obs = []
        for span in spans:
            text_observation = _observation(
                frame,
                "native-text",
                "text_span",
                box(*span["bbox"]),
                "pdfium-native-text"
                if span["source_family"] == "native_pdf"
                else "track1-ocr-text",
                family=span["source_family"],
                text=span["text"],
                source_text_id=span["id"],
                raw_text_span=span,
                dimension=associate_dimension(span, native_lines),
            )
            native_obs.append(text_observation)
        document["observations"].extend(native_obs)
        text_ids = {o["source_text_id"]: o["id"] for o in native_obs}
        for observation in observations:
            if observation["class"] == "room":
                observation["dependency_ids"].extend(
                    sorted({text_ids[e["text_id"]] for e in observation["room_type"]["evidence"]})
                )
        rooms = [o for o in observations if o["class"] == "room"]
        page = {
            "ml_sources": [],
            "frame": frame.to_dict(),
            "source_raster": raster.relative_to(root).as_posix(),
            "pixel_evidence": raw_path.relative_to(root).as_posix(),
            "vector_evidence": vector_path.relative_to(root).as_posix(),
            "availability": data["availability"],
            "warnings": data["warnings"],
            "room_relations": room_relations(
                rooms, [o for o in observations if o["class"] in ("door", "window")]
            ),
            "timing": {"pixel_cache_hit": hit, "original_extraction_seconds": data["seconds"]},
        }
        document["pages"].append(page)
        document["aggregates"].extend(aggregates(observations, frame.page_index, "classical"))
        detection = {
            "schema_version": "planproof-detection-v1",
            "document_id": frame.document_id,
            "page_index": frame.page_index,
            "image_size": frame.image_size,
            "coordinate_system": COORDINATES,
            "model_version": config.version,
            "inference_mode": "classical_full_page",
            "confidence_semantics": "normalized rule support, not calibrated correctness",
            "instances": [
                {
                    "id": o["id"],
                    "class": o["class"],
                    "bbox": o["bbox"],
                    "polygons": o["polygons"],
                    "confidence": o["score"]["value"],
                }
                for o in observations
            ],
        }
        path = pdir / "detections.json"
        write_json(path, detection)
        artifact(path, "detector_compatible_candidates", frame)
        overlay = pdir / "classical-overlay.png"
        render_overlay(image, observations).save(overlay)
        artifact(overlay, "classical_overlay", frame, image_size=frame.image_size)
        vector_canvas = Image.fromarray(image)
        draw = ImageDraw.Draw(vector_canvas)
        for center in data["centerlines"]:
            draw.line([(p[0] * sx, p[1] * sy) for p in center["segment"]], fill="#00a060", width=2)
        for gap in data["gaps"]:
            draw.line([(p[0] * sx, p[1] * sy) for p in gap["segment"]], fill="#ff3050", width=3)
        vector_overlay = pdir / "vector-overlay.png"
        vector_canvas.save(vector_overlay)
        artifact(
            vector_overlay,
            "vector_overlay",
            frame,
            image_size=frame.image_size,
            legend={"green": "observed wall medial lines", "red": "uninserted gap hypotheses"},
        )
        for source in imported:
            if source["binding"]["page_index"] != frame.page_index:
                continue
            name = source["binding"]["name"]
            intended = source["prediction"].get("intended_classes", list(CLASSES))
            page["ml_sources"].append(
                {"name": name, "available_classes": intended, "binding": source["binding"]}
            )
            ml_observations, hybrid = [], []
            for instance in source["prediction"]["instances"]:
                normalization = source["normalized"][instance["id"]]
                g = item_geometry({**instance, "polygons": normalization["polygons"]})
                ml = _observation(
                    frame,
                    "ml-" + name,
                    instance["class"] + "_proposal",
                    g,
                    "import-frozen-ml",
                    mode="hybrid",
                    family="ml",
                    **{"class": instance["class"]},
                    source_id=instance["id"],
                    score={
                        "value": instance["confidence"],
                        "semantics": source["prediction"].get(
                            "confidence_semantics",
                            "source detector score; not calibrated correctness",
                        ),
                    },
                    model_binding=source["binding"],
                    original_prediction=instance,
                )
                ml["normalization"] = {
                    "method": "shapely-make-valid",
                    "repairs": normalization["repairs"],
                    "original_preserved": True,
                }
                if normalization["repairs"]:
                    ml["warnings"].append(
                        "source ML rings repaired explicitly; raw prediction retained"
                    )
                ml_observations.append(ml)
                if instance.get("room_type") is not None:
                    ml["room_type"] = {
                        "category": instance["room_type"],
                        "source": "ml",
                        "score": instance.get("room_type_confidence"),
                        "status": "model_category_proposal",
                    }
                local = scale_geometry(g, 1 / sx, 1 / sy, origin=(0, 0))
                x0, y0, x1, y1 = local.bounds
                x0, y0 = max(0, int(x0)), max(0, int(y0))
                x1, y1 = (
                    min(data["ink"].shape[1], math.ceil(x1)),
                    min(data["ink"].shape[0], math.ceil(y1)),
                )
                support = float(data["ink"][y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0
                snapped, snap_report = (
                    snap_polygon(
                        local,
                        data["ink"],
                        max_distance=min(2, data["stroke_px"]),
                        ink_field=ink_field,
                    )
                    if local.geom_type == "Polygon" and instance["class"] in ("wall", "room")
                    else (local, {"accepted": False, "reason": "not a structural polygon"})
                )
                refined = scale_geometry(snapped, sx, sy, origin=(0, 0))
                h = _observation(
                    frame,
                    "hybrid-" + name,
                    "geometric_check",
                    refined,
                    "bounded-pixel-check",
                    parents=[ml["id"], pixel["id"]],
                    mode="hybrid",
                    family="hybrid",
                    **{"class": instance["class"]},
                    features={"bbox_ink_fraction": support, "snap": snap_report},
                    quantities=quantities(refined),
                    warnings=["ML-driven geometry check; cannot independently confirm its parent"],
                )
                hybrid.append(h)
            ml_wall_mask = np.zeros_like(data["ink"])
            for o in ml_observations:
                if o["class"] != "wall":
                    continue
                for polygon in o["polygons"]:
                    outer = np.round(np.asarray(polygon["outer"]) / [sx, sy]).astype(np.int32)
                    local_mask = np.zeros_like(ml_wall_mask)
                    import cv2

                    cv2.fillPoly(local_mask, [outer], 1)
                    for hole in polygon["holes"]:
                        cv2.fillPoly(
                            local_mask, [np.round(np.asarray(hole) / [sx, sy]).astype(np.int32)], 0
                        )
                    ml_wall_mask |= local_mask
            assisted_lines, _ = skeleton_lines(
                ml_wall_mask, max(config.min_line_px, data["stroke_px"] * 4)
            )
            ml_vector_path = pdir / (
                stable_id("ml", name).replace(":", "-") + "-wall-geometry.json"
            )
            write_json(
                ml_vector_path,
                {
                    "source_family": "hybrid",
                    "mode": "hybrid",
                    "model_source": name,
                    "dependency_ids": [o["id"] for o in ml_observations if o["class"] == "wall"],
                    "working_to_canonical": data["working_to_canonical"],
                    "centerlines": assisted_lines,
                    "graph": node_graph([line["segment"] for line in assisted_lines]),
                    "status": "available"
                    if np.any(ml_wall_mask)
                    else "unavailable_no_wall_mask_proposals",
                    "semantics": "ML-derived thickness/topology; not independent confirmation",
                },
            )
            artifact(ml_vector_path, "hybrid_wall_geometry", frame, model_source=name)
            comparisons = compare_sources(observations, ml_observations, config.match_iou)
            for comparison in comparisons:
                comparison.update(page_index=frame.page_index, model_source=name)
                comparison["crop_bbox"] = contextual_box(
                    comparison["bbox"], frame.image_size, 6 * data["stroke_px"] * max(sx, sy)
                )
            document["comparisons"].extend(comparisons)
            document["observations"].extend(ml_observations + hybrid)
            document["aggregates"].extend(
                aggregates(ml_observations, frame.page_index, name, available_classes=intended)
            )
            path = pdir / (stable_id("ml", name).replace(":", "-") + "-hybrid-overlay.png")
            render_overlay(image, hybrid).save(path)
            artifact(path, "hybrid_overlay", frame, image_size=frame.image_size, model_source=name)
        if not page["ml_sources"]:
            page["ml_comparison_status"] = "unavailable; no bound ML source"
    document["timing"] = {
        "total_seconds": time.perf_counter() - start,
        "device": "CPU",
        "training": False,
    }
    validate_document(document)
    context_path = root / "agent-context.json"
    write_json(context_path, agent_context(document))
    document["artifacts"].append(
        {
            "path": "agent-context.json",
            "sha256": sha256(context_path),
            "kind": "compact_agent_context",
            "frame_id": None,
        }
    )
    write_json(root / "evidence.json", document)
    return document


def get_region(evidence_path, page_index, bbox, out_dir, *, padding_px=0):
    evidence_path = Path(evidence_path).resolve()
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    validate_document(document)
    page = next((p for p in document["pages"] if p["frame"]["page_index"] == page_index), None)
    if page is None:
        raise ValueError("Page not present")
    if (
        len(bbox) != 4
        or not np.isfinite(bbox).all()
        or bbox[2] <= bbox[0]
        or bbox[3] <= bbox[1]
        or padding_px < 0
    ):
        raise ValueError("Invalid crop request")
    w, h = page["frame"]["image_size"]
    actual = [
        max(0, math.floor(bbox[0] - padding_px)),
        max(0, math.floor(bbox[1] - padding_px)),
        min(w, math.ceil(bbox[2] + padding_px)),
        min(h, math.ceil(bbox[3] + padding_px)),
    ]
    validate_bbox(actual, [w, h])
    raster = (evidence_path.parent / page["source_raster"]).resolve()
    if not raster.is_relative_to(evidence_path.parent):
        raise ValueError("Artifact path escapes job directory")
    entry = next(a for a in document["artifacts"] if a["path"] == page["source_raster"])
    if sha256(raster) != entry["sha256"]:
        raise ValueError("Source raster artifact changed")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with Image.open(raster) as image:
        crop = image.crop(actual)
        crop.save(out / "clean.png")
    region = box(*actual)
    ids = [
        o["id"]
        for o in document["observations"]
        if o["page_index"] == page_index and region.intersects(item_geometry(o))
    ]
    result = {
        "page_index": page_index,
        "bbox": actual,
        "requested_bbox": list(bbox),
        "crop_to_canonical": [[1, 0, actual[0]], [0, 1, actual[1]], [0, 0, 1]],
        "observation_ids": ids,
        "clean_crop": "clean.png",
        "sha256": sha256(out / "clean.png"),
    }
    write_json(out / "agent-context.json", agent_context(document, page_index, actual))
    result["context_path"] = "agent-context.json"
    write_json(out / "region.json", result)
    return result
