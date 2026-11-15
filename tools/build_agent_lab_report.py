"""Offline visual comparison of completed trials; no model calls or label access."""

import argparse
import base64
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import statistics
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "feedback/src")]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build(runs, packs, output, *, overlap=None, translations=None):
    overlap_by_id = {r["id"]: r for r in read(overlap)["cases"]} if overlap else {}
    english = read(translations) if translations else {}
    sources = {}
    for pack in packs:
        for case in read(pack / "manifest.json")["cases"]:
            source = (pack / case["input_path"]).resolve()
            if not source.is_relative_to(pack.resolve()):
                raise ValueError("Source path escapes the reference pack")
            sources[case["id"]] = (source, case["input_sha256"])
    cases, summaries = {}, []
    for label, directory in runs:
        report = read(directory / "report.json")
        ids = report["binding"]["case_ids"]
        if [r["id"] for r in report["cases"]] != ids or len(ids) != len(set(ids)):
            raise ValueError("Incomplete or reordered trial report")
        for row in report["cases"]:
            trace = (directory / row["trace"]).resolve()
            if not trace.is_relative_to(directory.resolve()):
                raise ValueError("Trace path escapes the trial")
            record = read(trace)
            if record["id"] != row["id"] or record["run"]["status"] not in {"completed", "failed", "interrupted"}:
                raise ValueError("Mismatched or nonterminal trace")
            result = record["run"].get("result") or {}
            if row["answer"] != result.get("answer", {}):
                raise ValueError("Report answer differs from retained result")
            cid = row["id"]
            if cid not in cases:
                source, expected = sources[cid]
                payload = source.read_bytes()
                if sha256(payload).hexdigest() != expected or row["source_sha256"] != expected:
                    raise ValueError("Source drawing changed")
                from app import pages, register_image, register_page
                handle = (register_page if source.suffix.lower() == ".pdf" else register_image)(payload, 1)
                page = pages[handle.document_id][1]
                if page.raster_sha256 != row["raster_sha256"]:
                    raise ValueError("Source rendering differs from the actual trial")
                with Image.open(BytesIO(page.png)) as image:
                    image.thumbnail((1000, 1000))
                    buffer = BytesIO()
                    image.convert("RGB").save(buffer, format="JPEG", quality=88)
                cases[cid] = {"id": cid, "question": row["question"], "reference": row["reference"],
                    "question_english": english.get(cid),
                    "detector_source_overlap": overlap_by_id.get(cid),
                    "source": row["source"], "category": row["category"], "language": row["language"],
                    "width": page.width, "height": page.height, "raster_sha256": page.raster_sha256,
                    "image": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(),
                    "runs": []}
            case = cases[cid]
            if case["raster_sha256"] != row["raster_sha256"] or case["reference"] != row["reference"]:
                raise ValueError("Matched comparison changed pixels or reference answer")
            case["runs"].append({"label": label, "answer": row["answer"], "score": row["score"],
                "seconds": row["seconds_http"], "nemotron_calls": row.get("nemotron_calls", 0),
                "algorithm_used": row.get("algorithm_used", False),
                "stop_reason": result.get("stop_reason"),
                "execution_status": result.get("status"),
                "provider_interruption_count": sum(a.get("status") != "ok" for a in
                    result.get("reliability", {}).get("provider_attempts", [])),
                "citations": result.get("canonical_citations", []),
                "tools": [{"name": t["name"], "args": t["args"],
                           "status": t.get("result", {}).get("status", "returned"),
                           "seconds": t.get("seconds", 0),
                           "automatic_delivery": t.get("automatic_delivery", False)}
                          for t in result.get("tool_trace", [])]})
        rows = report["cases"]
        summaries.append({"label": label, "correct": sum(r["score"]["correct"] for r in rows),
            "total": len(rows), "median": statistics.median(r["seconds_http"] for r in rows),
            "revision": report["binding"].get("revision", "Frozen production control")})
    data = {"summaries": summaries, "cases": list(cases.values())}
    template = Path(__file__).with_name("agent_lab_report.html").read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__LAB_DATA__", json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")), encoding="utf-8")
    return {"drawings": len(cases), "trials": len(summaries), "bytes": output.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Label=completed trial directory")
    parser.add_argument("--pack", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overlap", type=Path, help="Private frozen detector source-ID overlap audit")
    parser.add_argument("--translations", type=Path, help="Optional ID-to-English-question mapping for display only")
    args = parser.parse_args()
    runs = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in args.run]
    print(json.dumps(build(runs, args.pack, args.out, overlap=args.overlap, translations=args.translations)))
