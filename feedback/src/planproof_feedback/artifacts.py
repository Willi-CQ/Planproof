from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def safe_path(root: Path, relative: str):
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or "\\" in relative or ":" in relative:
        raise ValueError("Unsafe artifact path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its pack")
    return path
