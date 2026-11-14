"""Append-only local backups, verified by archive CRC and per-payload SHA256."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import write_json
from .images import sha256


def verify_backup(path: Path):
    with zipfile.ZipFile(path) as archive:
        broken = archive.testzip()
        if broken:
            raise ValueError("Backup ZIP CRC mismatch")
        manifest = json.loads(archive.read("BACKUP-MANIFEST.json"))
        if set(archive.namelist()) != {"BACKUP-MANIFEST.json", *manifest["payloads"]}:
            raise ValueError("Backup has missing or extra payloads")
        for relative, expected in manifest["payloads"].items():
            if hashlib.sha256(archive.read(relative)).hexdigest() != expected["sha256"]:
                raise ValueError("Backup payload SHA256 mismatch")
    return {
        "status": "verified",
        "archive": str(path),
        "sha256": sha256(path),
        "payloads": len(manifest["payloads"]),
        "bytes": path.stat().st_size,
    }


def backup(config: dict):
    root = Path(config["benchmark"]["pack_dir"]).parent
    directory = root / "backups"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"nemotron-evidence-{stamp}.zip"
    payloads = {}
    files = {}
    for item in sorted(root.rglob("*")):
        if (
            item.is_file()
            and not item.is_relative_to(directory)
            and item.suffix != ".tmp"
            and item.name != ".env"
            and not item.name.startswith(".env.")
            and item.suffix not in (".key", ".pem")
        ):
            files["evidence/" + item.relative_to(root).as_posix()] = item
    for key in ("question_file", "inventory_file"):
        source = Path(config["benchmark"][key])
        if source.exists():
            files["registries/" + source.name] = source
    project = Path(config["config_path"]).parent.parent
    for name in ("nemotron.cmd", "nemotron.ps1", ".env.example"):
        source = project / name
        if source.exists():
            files["source/" + name] = source
    # Exact source/configuration snapshot, explicitly excluding the real .env and local environments.
    for item in sorted((project / "feedback").rglob("*")):
        if (
            item.is_file()
            and item.suffix in (".py", ".toml", ".md", ".html", ".css", ".js", ".mjs")
            and not any(
                part in ("__pycache__", ".artifacts", ".venv", ".ruff_cache")
                for part in item.relative_to(project / "feedback").parts
            )
        ):
            files["source/" + item.relative_to(project).as_posix()] = item
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, source in files.items():
            payloads[relative] = {"sha256": sha256(source), "bytes": source.stat().st_size}
            archive.write(source, relative)
        archive.writestr(
            "BACKUP-MANIFEST.json",
            json.dumps(
                {
                    "schema": "planproof-local-backup-v1",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "payloads": payloads,
                    "credential_policy": "Real .env and Authorization headers excluded",
                },
                indent=2,
            ),
        )
    receipt = verify_backup(path)
    write_json(path.with_suffix(".receipt.json"), receipt)
    return receipt
