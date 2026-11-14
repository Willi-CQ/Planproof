"""Build a private, offline case review after a completed paired evaluation.

Reference answers and translations are display-only and never sent to inference.
The generated HTML embeds partner drawings/labels: keep it outside Git history.
"""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes run directory")
    return path


def outcome(row):
    if row["status"] != "ok":
        return "failed"
    if row["answer"]["status"] != "answered":
        return "abstained"
    return "correct" if row["score"]["correct"] else "wrong"


def build(run, answers_path, translations_path, output):
    report = read(run / "report.json")
    answers_bytes = answers_path.read_bytes()
    manifest = read(answers_path.parent.parent / "manifest.json")
    if hashlib.sha256(answers_bytes).hexdigest() != manifest["answers_sha256"]:
        raise ValueError("Reference answers changed")
    manifest_hash = hashlib.sha256(
        (answers_path.parent.parent / "manifest.json").read_bytes()
    ).hexdigest()
    if manifest_hash != report["binding"]["manifest_sha256"]:
        raise ValueError("Reference pack does not match frozen evaluation")
    answers = json.loads(answers_bytes)
    translations = read(translations_path)
    ids = report["binding"]["case_ids"]
    if len(report["cases"]) != len(ids) * 2:
        raise ValueError("Incomplete paired report")
    cases = []
    for number, cid in enumerate(ids, 1):
        rows = [r for r in report["cases"] if r["id"] == cid]
        if sorted(r["mode"] for r in rows) != ["auto", "direct"]:
            raise ValueError("Missing or duplicate mode")
        first = rows[0]
        english = first["question"]
        if first["language"] != "en":
            entry = translations[cid]
            if entry["original"] != first["question"] or not entry["english"].strip():
                raise ValueError("Translation is missing or belongs to another question")
            english = entry["english"]
        attempts = {}
        for row in rows:
            trace = read(contained(run, row["trace"]))
            if trace["answer"] != row["answer"]:
                raise ValueError("Trace/record answer mismatch")
            attempts[row["mode"]] = {
                **row,
                "outcome": outcome(row),
                "details": trace,
                "errors": [
                    c["error_code"] for c in trace["gemini_calls"] if "error_code" in c
                ],
            }
        image = contained(run, f"{cid}/page.png").read_bytes()
        with Image.open(io.BytesIO(image)) as source:
            raster_hash = hashlib.sha256(source.convert("RGB").tobytes()).hexdigest()
        if any(r["raster_sha256"] != raster_hash for r in rows):
            raise ValueError("Source raster changed")
        cases.append({
            "number": number, "id": cid, "question": first["question"],
            "english": english, "language": first["language"],
            "category": first["category"], "source": first["source"],
            "reference": answers[cid], "attempts": attempts,
            "image": "data:image/png;base64," + base64.b64encode(image).decode(),
        })
    payload = {"cases": cases, "summary": report["summary"]}
    template = Path(__file__).with_name("case_review.html").read_text(encoding="utf-8")
    page = template.replace("__REVIEW_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return {"cases": len(cases), "attempts": len(cases) * 2, "bytes": output.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run, args.answers, args.translations, args.out)))
