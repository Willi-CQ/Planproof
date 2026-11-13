from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .schema import sha256, validate_record, write_json


def load_manifest(path: Path, verify_files=False) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not records:
        raise ValueError("Empty dataset manifest")
    for r in records:
        validate_record(r, path.parent if verify_files else None)
    audit(records)
    return records


def fingerprint(records: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(records, key=lambda r: r["id"]), sort_keys=True).encode()
    ).hexdigest()


def audit(records: list[dict]):
    ids = set()
    groups = {}
    images = {}
    geometries = {}
    for r in records:
        if r["id"] in ids:
            raise ValueError(f"Duplicate record id: {r['id']}")
        ids.add(r["id"])
        for mapping, key in (
            (groups, r["group_id"]),
            (images, r["image_sha256"]),
            (geometries, r.get("geometry_hash")),
        ):
            if key:
                if key in mapping and mapping[key] != r["split"]:
                    raise ValueError(f"Train/validation/test leakage for {key}")
                mapping[key] = r["split"]
        if r["source"] == "cvcfp" and r["split"] != "test":
            raise ValueError("CVC-FP is external test only in v1")
    return {
        "samples": len(records),
        "groups": len(groups),
        "splits": {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")},
    }


def split_records(records: list[dict], seed=42) -> list[dict]:
    """Union original families, exact decoded images and geometry before splitting.

    Publisher held-out samples take precedence (test > val > train). Any changed
    publisher assignment is recorded; the clean project split is not claimed to
    reproduce the publisher benchmark.
    """
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    seen = {}
    for i, record in enumerate(records):
        for kind, key in (
            ("group", record["group_id"]),
            ("image", record["image_sha256"]),
            ("geometry", record.get("geometry_hash")),
        ):
            if key:
                k = (kind, key)
                if k in seen:
                    union(i, seen[k])
                seen[k] = i
    components = {}
    for i in range(len(records)):
        components.setdefault(find(i), []).append(records[i])
    priority = {"train": 0, "val": 1, "test": 2}
    sources_with_val = {r["source"] for r in records if r.get("publisher_split") == "val"}
    for component in components.values():
        group = min(r["group_id"] for r in component)
        prescribed = [r.get("publisher_split") for r in component if r.get("publisher_split")]
        if any(r["source"] == "cvcfp" for r in component):
            prescribed.append("test")
        if prescribed:
            split = max(prescribed, key=priority.__getitem__)
            if split == "train" and not any(r["source"] in sources_with_val for r in component):
                n = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:12], 16) / 16**12
                split = "val" if n < 0.1 else "train"
        else:
            n = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:12], 16) / 16**12
            split = "train" if n < 0.8 else "val" if n < 0.9 else "test"
        for r in component:
            r["group_id"] = group
            r["split"] = split
            r["split_adjusted"] = bool(r.get("publisher_split") and r["publisher_split"] != split)
    audit(records)
    return records


def write_manifest(path: Path, records: list[dict], seed=42):
    split_records(records, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in sorted(records, key=lambda r: r["id"])),
        encoding="utf-8",
    )
    write_json(
        path.with_suffix(".audit.json"),
        {
            **audit(records),
            "fingerprint": fingerprint(records),
            "manifest_sha256": sha256(path),
            "seed": seed,
            "protocol": "grouped-project-split; retain publisher_split for comparison",
            "publisher_reassignments": sum(r.get("split_adjusted", False) for r in records),
            "near_duplicate_review": "exact image/geometry/group duplicates are joined; perceptual candidates need audit review",
        },
    )


def regroup(path: Path, mapping_path: Path):
    """Apply reviewed duplicate aliases to entire existing groups."""
    from .schema import read_json

    records = load_manifest(path)
    aliases = read_json(mapping_path)
    if not isinstance(aliases, dict) or not all(isinstance(v, str) and v.strip() for v in aliases.values()):
        raise ValueError("Group map must be an object mapping record ids to nonempty reviewed group names")
    by_id = {r["id"]: r for r in records}
    unknown = set(aliases) - by_id.keys()
    if unknown:
        raise ValueError(f"Unknown group-map record ids: {sorted(unknown)}")
    by_group = {}
    for record_id, alias in aliases.items():
        group = by_id[record_id]["group_id"]
        if group in by_group and by_group[group] != alias:
            raise ValueError("One existing group cannot be split across multiple reviewed aliases")
        by_group[group] = alias
    for record in records:
        if record["group_id"] in by_group:
            record["group_id"] = "reviewed:" + by_group[record["group_id"]]
    write_manifest(path, records)
    return {
        "reviewed_records": len(aliases),
        "affected_original_groups": len(by_group),
        "new_fingerprint": fingerprint(records),
        "requires_new_training_run": True,
    }
