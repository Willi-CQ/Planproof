"""Append a separate human-reviewed drawing cohort to the frozen offline viewer.

Only the generated private HTML contains drawings/results. No inferred reference
answers or retrospective changes to the original benchmark are introduced.
"""

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path


def build(base, inventory_path, results, output):
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    source_root = Path(inventory["source_pack"]).resolve()
    cases = []
    for item in inventory["cases"]:
        source = (source_root / item["input_path"]).resolve()
        if not source.is_relative_to(source_root):
            raise ValueError("Source outside inventory root")
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["input_sha256"]:
            raise ValueError("Source changed since inference binding")
        path = results / f"case-{item['number']}.json"
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if record and (
            record["id"] != item["id"] or record["question"] != item["question"]
        ):
            raise ValueError("Result belongs to another input/question")
        result = record["result"] if record else None
        if result and result.get("source", {}).get("source_sha256") != digest:
            raise ValueError("Result was not produced from the original upload")
        cases.append(
            {
                "number": item["number"],
                "id": item["id"],
                "question": item["question"],
                "filename": source.name,
                "size": item["original_size"],
                "result": result,
                "image": "data:image/png;base64," + base64.b64encode(raw).decode(),
                "review_key": digest
                + hashlib.sha256(item["question"].encode()).hexdigest(),
            }
        )
    payload = json.dumps(
        {"cases": cases, "limitation": inventory["limitation"]}, ensure_ascii=False
    ).replace("<", "\\u003c")
    extension = (
        Path(__file__)
        .with_name("hard_case_review.html")
        .read_text(encoding="utf-8")
        .replace("__HARD_DATA__", payload)
    )
    page = base.read_text(encoding="utf-8")
    page = re.sub(
        r"<!--HARD_REVIEW_START-->.*?<!--HARD_REVIEW_END-->\r?\n?", "", page, flags=re.S
    )
    if '<script id="review-data"' not in page or "</body>" not in page:
        raise ValueError("Expected the existing 50-case review artifact")
    page = page.replace("</body>", extension + "</body>")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(output)
    return {
        "cases": len(cases),
        "completed": sum(c["result"] is not None for c in cases),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.inventory, args.results, args.out)))
