"""Compare complete, matched trials without new inference or changing any score."""

import argparse
from collections import Counter
import json
from math import comb, sqrt
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load(directory):
    report = read(directory / "report.json")
    rows = report["cases"]
    ids = report["binding"]["case_ids"]
    if [r["id"] for r in rows] != ids or len(ids) != len(set(ids)):
        raise ValueError("Incomplete or reordered report")
    for row in rows:
        trace = (directory / row["trace"]).resolve()
        if not trace.is_relative_to(directory.resolve()):
            raise ValueError("Trace escapes trial")
        raw = read(trace)
        if raw["id"] != row["id"] or raw["run"]["status"] not in {"completed", "failed", "interrupted"}:
            raise ValueError("Nonterminal or mismatched trace")
        if row["answer"] != (raw["run"].get("result") or {}).get("answer", {}):
            raise ValueError("Reported answer differs from retained answer")
    return report


def wilson(correct, total):
    """Descriptive 95% binomial interval, not a guarantee of generalization."""
    z = 1.959963984540054
    p, zz = correct / total, z * z
    center = (p + zz / (2 * total)) / (1 + zz / total)
    half = z * sqrt(p * (1 - p) / total + zz / (4 * total * total)) / (1 + zz / total)
    return [round(max(0, center - half), 4), round(min(1, center + half), 4)]


def summary(rows):
    correct = sum(r["score"]["correct"] for r in rows)
    answered = sum(r["answer"].get("status") == "answered" for r in rows)
    return {"cases": len(rows), "correct": correct, "answered": answered,
            "abstained_or_failed": len(rows) - answered,
            "strict_accuracy": correct / len(rows), "descriptive_95_percent_interval": wilson(correct, len(rows)),
            "median_seconds": statistics.median(r["seconds_http"] for r in rows),
            "mean_seconds": statistics.mean(r["seconds_http"] for r in rows),
            "gemini_reported_tokens_including_cached_input": sum(r.get("gemini_tokens", 0) for r in rows),
            "nemotron_calls": sum(r.get("nemotron_calls", 0) for r in rows)}


def compare(control, candidate):
    left, right = control["cases"], candidate["cases"]
    if not left or [r["id"] for r in left] != [r["id"] for r in right]:
        raise ValueError("Matched trials must contain identical ordered cases")
    gains, losses = [], []
    for a, b in zip(left, right):
        for key in ("source_sha256", "raster_sha256", "question", "reference"):
            if a[key] != b[key]:
                raise ValueError(f"Matched case changed {key}")
        if not a["score"]["correct"] and b["score"]["correct"]:
            gains.append(b["case_number"])
        if a["score"]["correct"] and not b["score"]["correct"]:
            losses.append(b["case_number"])
    n = len(gains) + len(losses)
    p = min(1, 2 * sum(comb(n, k) for k in range(min(len(gains), len(losses)) + 1)) / 2 ** n) if n else 1
    return {"control": summary(left), "candidate": summary(right), "gained_cases": gains,
            "regressed_cases": losses, "paired_exact_discordance_p": p,
            "categories": {c: {"cases": n, "control_correct": sum(r["score"]["correct"] for r in left if r["category"] == c),
                              "candidate_correct": sum(r["score"]["correct"] for r in right if r["category"] == c)}
                           for c, n in Counter(r["category"] for r in left).items()},
            "limitations": ["One stochastic run per condition; shared GPU and provider variation affect latency.",
                            "Development comparisons are selection data, not held-out evidence.",
                            "Source-file independence does not establish building-family independence.",
                            "Reported input tokens can include cached tokens; these are not a price estimate.",
                            "Category rooms also includes other structural questions in this selection."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(load(args.control), load(args.candidate))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
