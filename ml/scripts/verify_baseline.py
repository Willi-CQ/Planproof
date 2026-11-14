"""Verify a restored model baseline without loading models or training data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {}
    total = 0
    for entry in manifest["files"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
        path = (root / str(relative)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing baseline file: {relative}; run git lfs pull")
        with path.open("rb") as stream:
            if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"LFS pointer instead of model: {relative}; run git lfs pull")
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Checksum/size mismatch: {relative}")
        expected[str(relative)] = entry["sha256"]
        total += entry["bytes"]
    expected["manifest.json"] = digest(manifest_path)
    checksums = dict(
        (name, sha)
        for sha, name in (
            line.split("  ", 1) for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        )
    )
    if checksums != expected:
        raise ValueError("Top-level checksums do not match the baseline manifest")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(expected) | {"SHA256SUMS"}:
        raise ValueError("Unexpected or missing files in the immutable baseline")
    return {
        "verified": True,
        "baseline": manifest["version"],
        "payload_files": len(manifest["files"]),
        "payload_bytes": total,
        "source_commit": manifest["source_commit"],
        "manifest_sha256": expected["manifest.json"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("models/baselines/v1"))
    print(json.dumps(verify(parser.parse_args().root), indent=2))
