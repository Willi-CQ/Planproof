"""Build an offline question/drawing view without answers or model results."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Existing comparison directory")
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument("--ids", nargs="+", help="Explicit question IDs for a preset")
    preset.add_argument("--binding", type=Path, help="Saved pilot binding with exact case_ids")
    args = parser.parse_args()
    root = args.directory.resolve()
    source = json.loads((root / "questions-for-model.json").read_text(encoding="utf-8"))
    cases = source["cases"]
    if args.binding:
        binding = json.loads(args.binding.read_text(encoding="utf-8"))
        manifest = root.parent / "stoa-random-v1" / "manifest.json"
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != binding["manifest_sha256"]:
            parser.error("Pilot binding does not match the original question pack")
        args.ids = binding["case_ids"]
        if len(args.ids) != binding["count"]:
            parser.error("Pilot binding count does not match its question IDs")
    if args.ids:
        by_id = {case["id"]: case for case in cases}
        if len(set(args.ids)) != len(args.ids) or any(i not in by_id for i in args.ids):
            parser.error("Preset IDs must be unique and present in the question pack")
        cases = [by_id[i] for i in args.ids]
    public = []
    for case in cases:
        original = case["original_path"]
        images = [image["asset_path"] for image in case["images"]]
        for relative in [original, *images]:
            if not (root / relative).is_file():
                raise FileNotFoundError(relative)
        public.append({"id": case["id"], "question": case["question"],
                       "file": original, "name": Path(original).name, "images": images})
    destination = root / "index.html"
    preserved = root / "benchmark-results.html"
    if destination.exists() and 'id="question-browser"' not in destination.read_text(encoding="utf-8"):
        if preserved.exists() and preserved.read_bytes() != destination.read_bytes():
            raise FileExistsError("A different preserved benchmark page already exists")
        if not preserved.exists():
            shutil.copy2(destination, preserved)
    template = Path(__file__).with_name("question_browser.html").read_text(encoding="utf-8")
    payload = json.dumps(public, ensure_ascii=False).replace("<", "\\u003c")
    label = f"{len(public)}-question pilot" if args.ids else "Questions & drawings"
    page = template.replace("__QUESTION_DATA__", payload).replace("__PAGE_LABEL__", label)
    destination.write_text(page, encoding="utf-8")
    print(json.dumps({"questions": len(public), "page": str(destination),
                      "preserved_results": str(preserved)}, indent=2))


if __name__ == "__main__":
    main()
