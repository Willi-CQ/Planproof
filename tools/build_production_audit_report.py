"""Build a private, offline before/after review from a completed frozen audit."""

import argparse
import base64
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import statistics

from PIL import Image


def build(audit, baseline, translations, output):
    report = json.loads((audit / "report.json").read_text(encoding="utf-8"))
    if (
        sha256((baseline / "report.json").read_bytes()).hexdigest()
        != report["binding"]["baseline_report_sha256"]
    ):
        raise ValueError("Historical report changed")
    ids = report["binding"]["case_ids"]
    if [row["id"] for row in report["cases"]] != ids or len(set(ids)) != len(ids):
        raise ValueError("Incomplete audit")
    translated = json.loads(translations.read_text(encoding="utf-8"))
    cases = []
    for row in report["cases"]:
        cid = row["id"]
        if cid not in ids or cid in ("", ".", "..") or Path(cid).name != cid:
            raise ValueError("Unknown case ID")
        raw = (baseline / cid / "page.png").read_bytes()
        with Image.open(BytesIO(raw)) as image:
            raster = sha256(image.convert("RGB").tobytes()).hexdigest()
            width, height = image.size
        if raster != row["raster_sha256"] or not row["raster_matches"]:
            raise ValueError("Comparison uses different source pixels")
        before = json.loads((baseline / cid / "auto.json").read_text(encoding="utf-8"))
        current = json.loads((audit / cid / "record.json").read_text(encoding="utf-8"))[
            "run"
        ]
        result = current.get("result") or {}
        if (
            before["answer"] != row["before"]["answer"]
            or (result.get("answer") or {}) != row["answer"]
        ):
            raise ValueError("Trace/report answer mismatch")
        english = row["question"]
        if row["language"] != "en":
            entry = translated[cid]
            if entry["original"] != english:
                raise ValueError("Translation belongs to a different question")
            english = entry["english"]
        cases.append(
            {
                **row,
                "english": english,
                "width": width,
                "height": height,
                "image": "data:image/png;base64," + base64.b64encode(raw).decode(),
                "before_citations": before.get("canonical_citations", []),
                "before_algorithm_used": bool(before.get("algorithm_used")),
                "algorithm_requests": sum(
                    entry.get("name") == "get_algorithms"
                    for entry in result.get("tool_trace", [])
                ),
                "current_citations": result.get("canonical_citations", []),
                "activity": [
                    {
                        k: event[k]
                        for k in (
                            "stage",
                            "state",
                            "elapsed",
                            "category",
                            "question",
                            "advisory",
                            "unresolved",
                        )
                        if k in event
                    }
                    for event in current.get("events", [])
                ],
            }
        )
    payload = {
        "summary": {
            **report["summary"],
            "baseline_median_seconds": statistics.median(
                row["before"]["seconds"] for row in report["cases"]
            ),
            "baseline_algorithm_cases": sum(c["before_algorithm_used"] for c in cases),
            "current_algorithm_cases": sum(
                bool(c.get("algorithm_used")) for c in cases
            ),
            "current_algorithm_requested_cases": sum(
                c["algorithm_requests"] > 0 for c in cases
            ),
        },
        "cases": cases,
        "condition": report["binding"]["condition"],
    }
    template = (
        Path(__file__).with_name("production_audit.html").read_text(encoding="utf-8")
    )
    output.write_text(
        template.replace(
            "__AUDIT_DATA__",
            json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"),
        ),
        encoding="utf-8",
    )
    return {"cases": len(cases), "bytes": output.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.audit, args.baseline, args.translations, args.out)))
