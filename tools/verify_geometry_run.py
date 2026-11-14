"""Verify local geometry receipts and unchanged frozen inputs; never publishes data."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import planproof_evidence.pipeline as pipeline
from planproof_evidence.contracts import sha256, validate_document, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads((args.frozen / "PACK_SHA256SUMS.json").read_text())
    for filename, digest in frozen.items():
        if sha256(args.frozen / filename) != digest:
            raise ValueError("Frozen input changed: " + filename)
    manifest = json.loads((args.pack / "manifest.json").read_text())
    run = json.loads((args.pack / "GEOMETRY_RUN.json").read_text())
    evidence_files, artifacts, repairs, cache_hits = 0, 0, 0, 0
    implementation = Path(pipeline.__file__).parent
    for case in manifest["cases"]:
        root = args.pack / "geometry" / case["case_id"]
        document = json.loads((root / "evidence.json").read_text())
        validate_document(document)
        record = next(r for r in run["cases"] if r["case_id"] == case["case_id"])
        if sha256(root / "evidence.json") != record["evidence_sha256"]:
            raise ValueError("Run/evidence binding mismatch")
        for name, digest in document["implementation_hashes"].items():
            if sha256(implementation / (name + ".py")) != digest:
                raise ValueError("Run/source implementation mismatch: " + name)
        for artifact in document["artifacts"]:
            path = (root / artifact["path"]).resolve()
            if (
                not path.is_relative_to(root.resolve())
                or sha256(path) != artifact["sha256"]
            ):
                raise ValueError("Invalid/changed artifact: " + artifact["path"])
            artifacts += 1
        evidence_files += 1
        repairs += sum(
            bool(o.get("normalization", {}).get("repairs"))
            for o in document["observations"]
        )
        cache_hits += sum(p["timing"]["pixel_cache_hit"] for p in document["pages"])
    report = json.loads((args.pack / "failure-report.json").read_text())
    original = json.loads((args.frozen / "failure-report.json").read_text())
    if report["summary"]["deterministic"]["cases_scored"] != 10:
        raise ValueError("Incomplete ten-plan evaluation")
    for method in ("main", "cad"):
        if report["summary"][method] != original["summary"][method]:
            raise ValueError("Frozen ML evaluation drift: " + method)
    receipt = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "frozen_payloads_unchanged": len(frozen),
        "evidence_documents_verified": evidence_files,
        "artifact_hashes_verified": artifacts,
        "source_implementation_hashes_verified": True,
        "explicitly_repaired_ml_observations": repairs,
        "pixel_cache_hits": cache_hits,
        "frozen_ml_scores_unchanged": True,
        "training": False,
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "planproof-evidence",
                "numpy",
                "opencv-python-headless",
                "Pillow",
                "scipy",
                "shapely",
                "scikit-image",
                "pypdfium2",
            )
        },
        "report_sha256": sha256(args.pack / "failure-report.json"),
        "run_sha256": sha256(args.pack / "GEOMETRY_RUN.json"),
    }
    write_json(args.pack / "GEOMETRY_VERIFICATION.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
