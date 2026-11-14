from __future__ import annotations
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import partial, lru_cache
from itertools import islice
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from .adapters import geometry_hash, iter_samples, svg_samples, publisher_splits, svg_source_paths
from .manifest import load_manifest, write_manifest
from .schema import sha256, write_json, read_json
from .targets import encode, validate_targets
from .acquire import sources
from .storage import ensure_artifact_directory


def prepare(
    source: str,
    raw: Path,
    output: Path,
    class_map: Path | None = None,
    trust_pickle=False,
    limit=None,
    workers=1,
    exclusions: Path | None = None,
):
    if not isinstance(workers, int) or not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    registry = sources()
    ensure_artifact_directory(output)
    if any(
        "junction-dataset" in part.lower() or part.lower().startswith("stoa-client") for part in raw.parts
    ):
        raise ValueError("Use separately downloaded publisher releases, not the Stoa challenge archive")
    if source not in registry or source in {"robin", "zind"}:
        raise ValueError("Select a detection training/test source; ROBIN/ZInD are future auxiliary tasks")
    mapping = read_json(class_map) if class_map else None
    if mapping and (
        not isinstance(mapping.get("mapping"), dict)
        or any(
            value not in {"wall", "room", "door", "window", "stairs"} for value in mapping["mapping"].values()
        )
        or any(
            value not in {"wall", "room", "door", "window", "stairs"} for value in mapping.get("complete", [])
        )
    ):
        raise ValueError(
            "Class map must use supported detection classes and declare compatible annotation coverage"
        )
    if source == "floorplancad" and mapping is None:
        raise ValueError("Specify the mapping for original SVG semanticId values with --class-map")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"
    existing = load_manifest(manifest) if manifest.exists() else []
    records = [r for r in existing if r["source"] != source]
    added = []
    annotation_hashes = {}
    excluded = read_json(exclusions) if exclusions else {}
    if not isinstance(excluded, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in excluded.items()
    ):
        raise ValueError("Exclusions must map source-relative SVG paths to explicit reasons")
    paths = None
    if source in {"cubicasa5k", "floorplancad"}:
        paths = svg_source_paths(source, raw)
        known = {p.relative_to(raw).as_posix() for p in paths}
        if set(excluded) - known:
            raise ValueError("Exclusions contain unknown source paths")
        paths = [p for p in paths if p.relative_to(raw).as_posix() not in excluded]
    elif excluded:
        raise ValueError("Explicit path exclusions currently support CubiCasa and FloorPlanCAD SVGs")
    executor_class = ThreadPoolExecutor
    if source in {"cubicasa5k", "floorplancad"} and workers > 1:
        iterator = ((path, raw, mapping, source, output, limit is not None) for path in paths)
        worker = _prepare_svg_path
        executor_class = ProcessPoolExecutor
    else:
        iterator = (
            svg_samples(source, raw, mapping, svg_paths=paths)
            if paths is not None
            else iter_samples(source, raw, mapping, trust_pickle)
        )
        worker = partial(
            _prepare_item,
            source=source,
            output=output,
            limited=limit is not None,
            annotation_hashes=annotation_hashes,
        )
    if limit is not None:
        iterator = islice(iterator, limit)
    started = time.monotonic()
    tasks = iter(enumerate(iterator))
    completed = {}
    with executor_class(max_workers=workers) as executor:
        pending = {}

        def submit_next():
            try:
                index, item = next(tasks)
            except StopIteration:
                return
            pending[executor.submit(worker, item)] = index

        for _ in range(workers):
            submit_next()
        reported = 0
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                completed[pending.pop(future)] = future.result()
                submit_next()
            if len(completed) - reported >= 100:
                print(
                    {"prepared": len(completed), "seconds": round(time.monotonic() - started, 1)},
                    file=sys.stderr,
                    flush=True,
                )
                reported = len(completed)
    added = [completed[i] for i in range(len(completed))]
    records.extend(added)
    if not added:
        raise ValueError(f"No samples prepared for {source}; existing manifest was not replaced")
    write_manifest(manifest, records)
    write_json(
        output / "provenance" / (source + ".json"),
        {
            "source": source,
            "publisher": registry[source]["source_url"],
            "access": registry[source]["access"],
            "raw_root": str(raw.resolve()),
            "records": len(added),
            "limited_import": limit is not None,
            "excluded_records": len(excluded),
            "exclusions": excluded,
            "adapter_validation": "automated contract tests; real publisher archive must pass inspect on GPU host",
        },
    )
    return {"manifest": str(manifest), "added": len(added), "total": len(records)}


def _prepare_item(item, source, output, limited, annotation_hashes):
    key = item["key"]
    digest = hashlib.sha256(f"{source}:{key}".encode()).hexdigest()[:24]
    image = item["image"]
    h, w = image.shape[:2]
    data = encode(
        (h, w), item["features"], item["complete"], item.get("ignore", []), item.get("instance_complete")
    )
    validate_targets(data)
    relative = f"samples/{source}/{digest}"
    image_path = output / (relative + ".png")
    labels_path = output / (relative + ".npz")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(image_path)
    np.savez_compressed(labels_path, **data)
    # Deduplicate decoded pixels, not JPEG/PNG container bytes.
    image_hash = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
    group = item["group"]
    if not group.startswith("resplan:"):
        group = source + ":" + group
    annotation_key = str(item["annotation"].resolve())
    if annotation_key not in annotation_hashes:
        annotation_hashes[annotation_key] = sha256(item["annotation"])
    record = {
        "id": source + ":" + digest,
        "source": source,
        "family": "synthetic"
        if source in {"resplan", "modres", "rplan", "sesyd", "structured3d", "houseexpo"}
        else "real",
        "group_id": group,
        "publisher_split": item.get("publisher_split"),
        "split": "test" if source == "cvcfp" else "train",
        "image": relative + ".png",
        "targets": relative + ".npz",
        "width": w,
        "height": h,
        "image_sha256": image_hash,
        "geometry_hash": geometry_hash(item["features"]),
        "targets_sha256": sha256(labels_path),
        "annotation_sha256": annotation_hashes[annotation_key],
        "positive_classes": sorted(
            set(
                [
                    name
                    for i, name in enumerate(("wall", "room", "door", "window", "stairs"))
                    if data["masks"][i].any()
                ]
                + [("door", "window", "stairs")[int(row[0])] for row in data["objects"]]
            )
        ),
        "limited_import": limited,
        "image_hash_mode": "decoded_rgb_shape",
        "source_key": key,
        "metadata": item.get("metadata", {}),
    }
    return record


@lru_cache(maxsize=16)
def _worker_splits(source, raw):
    return publisher_splits(source, raw)


def _prepare_svg_path(task):
    path, raw, mapping, source, output, limited = task
    item = next(svg_samples(source, raw, mapping, svg_paths=[path], split_lookup=_worker_splits(source, raw)))
    return _prepare_item(item, source, output, limited, {})
