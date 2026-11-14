"""Select a deterministic, answer-blind mixed 50-case development cohort."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


# Approximately balance raster/CAD sources without duplicating sparse strata.
QUOTAS = {
    ("area", "swiss_dwellings"): 11,
    ("area", "ifc_models"): 1,
    ("apartments", "swiss_dwellings"): 2,
    ("floors", "cubicasa5k"): 4,
    ("openings", "cubicasa5k"): 4,
    ("openings", "swiss_dwellings"): 4,
    ("fixtures", "cubicasa5k"): 8,
    ("fixtures", "swiss_dwellings"): 2,
    ("rooms", "cubicasa5k"): 8,
    ("rooms", "swiss_dwellings"): 6,
}


def category(case):
    q = case["question"].casefold()
    if case["type"] == "area":
        return "area"
    if any(s in q for s in ("apartments", "appartements")):
        return "apartments"
    if "kerrosta" in q or "floors (storeys)" in q:
        return "floors"
    if any(s in q for s in ("windows", "fenster", "doors")):
        return "openings"
    if any(
        s in q
        for s in (
            "showers",
            "bathtubs",
            "toilet bowls",
            "toilets (wc)",
            "sinks",
            "lavabos",
            "wc-istuinta",
            "suihkua",
            "wc sont",
        )
    ):
        return "fixtures"
    if case["type"] == "count":
        return "rooms"
    raise ValueError("Unclassified question type")


def language(case):
    q = case["question"].casefold()
    for prefix, code in [
        ("kuinka", "fi"),
        ("wie ", "de"),
        ("combien", "fr"),
        ("quelle", "fr"),
        ("qual ", "it"),
        ("how ", "en"),
        ("what ", "en"),
    ]:
        if q.startswith(prefix):
            return code
    raise ValueError("Unclassified language")


def select(cases, excluded_ids, seed):
    excluded_sources = {c["input_sha256"] for c in cases if c["id"] in excluded_ids}
    available = [c for c in cases if c["input_sha256"] not in excluded_sources]
    selected, used = [], set()

    def rank(c):
        return hashlib.sha256((seed + c["id"]).encode()).hexdigest()

    for (group, source), quota in QUOTAS.items():
        candidates = [
            c for c in available if category(c) == group and c["source"] == source
        ]
        seen_language, seen_template = Counter(), Counter()
        for _ in range(quota):
            pool = [c for c in candidates if c["input_sha256"] not in used]
            if not pool:
                raise ValueError(f"Insufficient unique plans for {group}/{source}")
            chosen = min(
                pool,
                key=lambda c: (
                    seen_language[language(c)],
                    seen_template[c["question"]],
                    rank(c),
                ),
            )
            selected.append(chosen)
            used.add(chosen["input_sha256"])
            seen_language[language(chosen)] += 1
            seen_template[chosen["question"]] += 1
    return sorted(selected, key=rank)


def main(args):
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    excluded = json.loads(args.exclude.read_text(encoding="utf-8"))["case_ids"]
    chosen = select(manifest["cases"], set(excluded), args.seed)
    assert len(chosen) == 50
    result = {
        "schema": "planproof-mixed-agent50-v1",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "case_ids": [c["id"] for c in chosen],
        "seed": args.seed,
        "excluded_case_ids": excluded,
        "selection_policy": "Fixed category/source quotas; balance languages and exact question templates within strata; seeded SHA256 tie-break; distinct source hashes; exclude prior pilot plans. No answers or previous scores read.",
        "quotas": [
            {"category": g, "source": s, "count": n} for (g, s), n in QUOTAS.items()
        ],
        "metadata": {
            c["id"]: {
                "category": category(c),
                "language": language(c),
                "source": c["source"],
                "type": c["type"],
            }
            for c in chosen
        },
        "distribution": {
            field: dict(Counter(fn(c) for c in chosen))
            for field, fn in {
                "category": category,
                "language": language,
                "source": lambda c: c["source"],
                "type": lambda c: c["type"],
                "format": lambda c: Path(c["input_path"]).suffix,
            }.items()
        },
        "limitation": "Stratified development sample from an existing 100-case Nemotron benchmark, excluding eight prior Gemini pilot drawings. Not a population estimate or independent held-out detector evaluation. Single supplied 2D page only.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result["distribution"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", default="planproof-agent50-v1-20261114")
    main(parser.parse_args())
