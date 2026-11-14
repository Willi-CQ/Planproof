"""Explicit GPU-side acquisition; importing or doctor never downloads."""

from __future__ import annotations
import hashlib
import importlib.resources
from pathlib import Path
import stat
import shutil
import tarfile
import zipfile
import requests
from .schema import read_json, sha256, write_json
from .storage import ensure_artifact_directory


def sources():
    return read_json(Path(str(importlib.resources.files("planproof_ml").joinpath("assets/sources.json"))))


def safe_name(name):
    if not name or Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    return name


def extract_archive(archive: Path, output: Path):
    ensure_artifact_directory(output)
    output.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            if sum(i.file_size for i in z.infolist()) > shutil.disk_usage(output).free:
                raise ValueError("Insufficient free disk for extracted archive")
            for item in z.infolist():
                safe_name(item.filename)
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Archive symlinks are not permitted")
            z.extractall(output)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tar:
            if sum(i.size for i in tar.getmembers()) > shutil.disk_usage(output).free:
                raise ValueError("Insufficient free disk for extracted archive")
            for item in tar.getmembers():
                safe_name(item.name)
                if not (item.isfile() or item.isdir()):
                    raise ValueError("Archive links and special files are not permitted")
            tar.extractall(output, filter="data")
    else:
        raise ValueError(f"Not a supported archive: {archive}")


def download(url: str, destination: Path, expected_bytes=None, checksum=None):
    if not url.startswith(("https://", "http://")):
        raise ValueError("Use a publisher HTTP(S) download URL")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_bytes is not None and destination.stat().st_size != expected_bytes:
            raise ValueError(f"Existing file size differs: {destination}")
        if checksum:
            algorithm, expected = checksum.split(":", 1)
            h = hashlib.new(algorithm)
            with destination.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError("Existing publisher checksum mismatch")
        return sha256(destination)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if not checksum:
        offset = 0  # Unversioned URLs must not append bytes from a potentially changed release.
    headers = {"User-Agent": "Planproof-dataset-acquisition/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        if offset and response.status_code == 206:
            if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Incompatible resume range")
            mode = "ab"
        else:
            mode = "wb"
        if "text/html" in response.headers.get("Content-Type", ""):
            raise ValueError("Publisher returned HTML instead of a dataset file")
        with partial.open(mode) as f:
            for block in response.iter_content(1024 * 1024):
                f.write(block)
    if expected_bytes is not None and partial.stat().st_size != expected_bytes:
        raise ValueError("Incomplete download; .part retained")
    if checksum:
        algorithm, expected = checksum.split(":", 1)
        h = hashlib.new(algorithm)
        with partial.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != expected:
            partial.unlink()
            raise ValueError("Publisher checksum mismatch; corrupt partial removed")
    partial.replace(destination)
    return sha256(destination)


def fetch(source: str, data_root: Path, urls: list[str] | None = None, extract=False):
    registry = sources()
    if source not in registry or "stoa" in source:
        raise ValueError("Select an original training-source id from sources")
    spec = registry[source]
    artifacts = (
        [{"url": u, "name": u.split("?")[0].rsplit("/", 1)[-1]} for u in urls]
        if urls
        else list(spec["artifacts"])
    )
    if source == "cubicasa5k" and not urls:
        response = requests.get("https://zenodo.org/api/records/2613548", timeout=30)
        response.raise_for_status()
        item = next(f for f in response.json()["files"] if f["key"] == "cubicasa5k.zip")
        artifacts = [
            {
                "url": item["links"]["self"],
                "name": "cubicasa5k.zip",
                "bytes": item["size"],
                "checksum": item["checksum"],
            }
        ] + [a for a in spec["artifacts"] if a["name"] != "cubicasa5k.zip"]
    if not artifacts:
        raise ValueError(
            f"{source} requires publisher access/manual URL resolution: {spec['access']}; {spec['source_url']}"
        )
    directory = data_root.resolve() / "downloads" / source
    ensure_artifact_directory(directory)
    ensure_artifact_directory(data_root / "extracted" / source)
    receipts = []
    for artifact in artifacts:
        target = directory / safe_name(artifact["name"])
        if artifact.get("drive_id"):
            try:
                import gdown
            except ImportError as exc:
                raise ValueError("Install the download extra for publisher Drive files") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                partial = target.with_suffix(target.suffix + ".part")
                if (
                    gdown.download(id=artifact["drive_id"], output=str(partial), quiet=False, resume=True)
                    is None
                ):
                    raise ValueError("Drive access/quota failed; obtain original file from publisher")
                partial.replace(target)
            digest = sha256(target)
        else:
            digest = download(artifact["url"], target, artifact.get("bytes"), artifact.get("checksum"))
        receipt = {
            **artifact,
            "sha256": digest,
            "bytes_on_disk": target.stat().st_size,
            "source": source,
            "source_url": spec["source_url"],
            "access_notes": spec["access"],
        }
        write_json(target.with_suffix(target.suffix + ".source.json"), receipt)
        if extract and (zipfile.is_zipfile(target) or tarfile.is_tarfile(target)):
            extraction = data_root / "extracted" / source
            if source == "floorplancad":
                # Release archives contain flat paths. Keep train/test archives
                # separate so duplicate names cannot overwrite held-out records.
                extraction = extraction / target.name.split(".")[0]
            extract_archive(target, extraction)
        elif extract:
            sidecar = data_root / "extracted" / source / target.name
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, sidecar)
        receipts.append(receipt)
    return receipts
