from __future__ import annotations
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from .adapters import geometry_hash, iter_samples
from .manifest import load_manifest, write_manifest
from .schema import sha256, write_json, read_json
from .targets import encode, validate_targets
from .acquire import sources
from .storage import ensure_artifact_directory


def prepare(
    source: str, raw: Path, output: Path, class_map: Path | None = None, trust_pickle=False, limit=None
):
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
    for item in iter_samples(source, raw, mapping, trust_pickle):
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
            "limited_import": limit is not None,
            "image_hash_mode": "decoded_rgb_shape",
            "source_key": key,
            "metadata": item.get("metadata", {}),
        }
        records.append(record)
        added.append(record)
        if limit is not None and len(added) >= limit:
            break
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
            "adapter_validation": "automated contract tests; real publisher archive must pass inspect on GPU host",
        },
    )
    return {"manifest": str(manifest), "added": len(added), "total": len(records)}
