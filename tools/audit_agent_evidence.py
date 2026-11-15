"""Audit retained evidence structure; this does not certify semantic correctness."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from compare_agent_trials import load, read


def audit_result(result):
    issues = []
    source = result.get("source", {})
    width, height = source.get("width", 0), source.get("height", 0)
    images = {image["image_id"]: image for image in result.get("images", [])}
    citations = result.get("canonical_citations", [])
    answer = result.get("answer", {})
    if answer.get("status") == "answered":
        if not citations:
            issues.append("answered_without_canonical_citations")
        if result.get("delegation", {}).get("required") and not result.get("nemotron_calls"):
            issues.append("answered_without_required_specialist_call")
        if (answer.get("count_basis") or {}).get("scope") == "partial":
            issues.append("answered_with_partial_count_basis")
    for image in images.values():
        if image.get("source_sha256") != source.get("raster_sha256"):
            issues.append("image_source_identity_mismatch")
    for citation in citations:
        box = citation.get("canonical_bbox")
        if citation.get("image_id") not in images:
            issues.append("citation_image_missing")
        if citation.get("page_index") != source.get("page_index"):
            issues.append("citation_page_mismatch")
        if not (isinstance(box, list) and len(box) == 4 and
                all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box)):
            issues.append("invalid_canonical_bbox")
            continue
        x0, y0, x1, y1 = box
        if not (-1e-6 <= x0 < x1 <= width + 1e-6 and -1e-6 <= y0 < y1 <= height + 1e-6):
            issues.append("citation_outside_source_or_empty")
    return sorted(set(issues))


def audit(directory):
    report = load(directory)
    rows = []
    provider_statuses = Counter()
    stops = Counter()
    tools = Counter()
    for row in report["cases"]:
        result = (read(directory / row["trace"])["run"].get("result") or {})
        stops[result.get("stop_reason", "no_result")] += 1
        provider_statuses.update(a.get("status") or a.get("category") or "unknown" for a in
                                 result.get("reliability", {}).get("provider_attempts", []))
        tools.update(t["name"] for t in result.get("tool_trace", []))
        rows.append({"id": row["id"], "case_number": row["case_number"],
                     "issues": audit_result(result), "answered": row["answer"].get("status") == "answered",
                     "correct": row["score"]["correct"]})
    return {"cases": rows, "cases_with_structural_issues": sum(bool(r["issues"]) for r in rows),
            "answered_correct": sum(r["answered"] and r["correct"] for r in rows),
            "answered_incorrect": sum(r["answered"] and not r["correct"] for r in rows),
            "stop_reasons": dict(stops), "provider_attempt_statuses": dict(provider_statuses),
            "tool_calls": dict(tools),
            "limitation": "Identity/bounds/contract audit only. Plausible boxes and specialist calls do not prove semantic support or full-page coverage."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
