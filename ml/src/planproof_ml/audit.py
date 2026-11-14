from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import sys
import hashlib
import cv2
import numpy as np
from .adapters import load_image
from .manifest import audit, load_manifest, fingerprint
from .targets import validate_targets


def audit_manifest(path, deep=False, near=False, workers=1):
    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    records = load_manifest(path, verify_files=True)
    report = {**audit(records), "fingerprint": fingerprint(records)}
    if deep:
        items = [(record, path.parent) for record in records]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for count, _ in enumerate(pool.map(_deep_check, items, chunksize=4), 1):
                    if count % 1000 == 0:
                        print({"audit_checked": count}, file=sys.stderr, flush=True)
        else:
            for item in items:
                _deep_check(item)
        report["target_arrays_checked"] = len(records)
        report["image_hashes_checked"] = len(records)
    if near:
        buckets = defaultdict(list)
        hashes = []
        pairs = []
        truncated = False
        items = [(record, path.parent) for record in records]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                values = list(pool.map(_perceptual_hash, items, chunksize=16))
        else:
            values = list(map(_perceptual_hash, items))
        for i, record in enumerate(records):
            value = values[i]
            candidates = set()
            for band in range(4):
                key = (band, (value >> (16 * band)) & 65535)
                candidates.update(buckets[key])
                buckets[key].append(i)
            for j in candidates:
                other = records[j]
                if record["split"] != other["split"] and record["image_sha256"] != other["image_sha256"]:
                    distance = (value ^ hashes[j]).bit_count()
                    if distance <= 3:
                        pairs.append({"a": other["id"], "b": record["id"], "hamming": distance})
                        if len(pairs) >= 5000:
                            truncated = True
                            break
            hashes.append(value)
            if truncated:
                break
        report["near_duplicate_candidates"] = pairs
        report["near_review_truncated"] = truncated
        report["near_review_note"] = (
            "Candidates require manual geometry review; visual similarity alone does not prove duplicate source drawings."
        )
    return report


def _deep_check(item):
    from .schema import sha256

    record, root = item
    with np.load(root / record["targets"], allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    validate_targets(data)
    image = load_image(root / record["image"])
    digest = (
        hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        if record.get("image_hash_mode") == "decoded_rgb_shape"
        else sha256(root / record["image"])
    )
    if digest != record["image_sha256"]:
        raise ValueError(f"Image hash mismatch: {record['id']}")
    if image.shape[:2] != data["masks"].shape[1:]:
        raise ValueError(f"Image/target dimensions differ: {record['id']}")
    if record.get("targets_sha256") and sha256(root / record["targets"]) != record["targets_sha256"]:
        raise ValueError(f"Target hash mismatch: {record['id']}")
    return True


def _perceptual_hash(item):
    record, root = item
    image = load_image(root / record["image"])
    gray = cv2.cvtColor(cv2.resize(image, (32, 32)), cv2.COLOR_RGB2GRAY).astype(np.float32)
    low = cv2.dct(gray)[:8, :8].reshape(-1)
    bits = low > np.median(low[1:])
    value = sum(int(bit) << idx for idx, bit in enumerate(bits))
    return value
