"""Prepare answer-blind development/holdout strata from the Stoa reference split."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "feedback/src"))
from planproof_feedback.archive import RangeArchive  # noqa: E402
from planproof_feedback.artifacts import safe_path, write_json  # noqa: E402
from planproof_feedback.benchmark import select_cases, verify_pack  # noqa: E402
from planproof_feedback.images import sha256 as file_sha256  # noqa: E402
from select_agent_cohort import category, language  # noqa: E402

QUOTAS = {"area": 8, "openings": 6, "fixtures": 6, "rooms": 5,
          "floors": 2, "apartments": 3}


def group(case):
    # Exact-basename alternate formats belong to the same drawing group.
    return str(PurePosixPath(case["path"]).with_suffix(""))


def metadata(case):
    kind = case["type"]
    try:
        lang = language(case)
    except ValueError:
        lang = "other"
    return {"category": kind if kind in {"text", "length", "volume"} else category(case),
            "language": lang}


def choose(cases, excluded, seed):
    used = set(excluded)
    result = []
    pools = {kind: [] for kind in QUOTAS}
    for case in cases:
        item = {**case, **metadata(case)}
        if item["category"] in pools:
            pools[item["category"]].append(item)
    for split in ("development", "holdout"):
        source_counts, template_counts, language_counts = Counter(), Counter(), Counter()
        for kind, count in QUOTAS.items():
            pool = pools[kind]
            for _ in range(count):
                eligible = [c for c in pool if group(c) not in used]
                if not eligible:
                    raise ValueError(f"Insufficient drawing groups for {split}/{kind}")
                item = min(eligible, key=lambda c: (
                    source_counts[c["source"]], language_counts[c["language"]],
                    template_counts[c["question"]],
                    sha256((seed + split + c["id"]).encode()).hexdigest()))
                result.append({**item, "split": split})
                used.add(group(item))
                source_counts[item["source"]] += 1
                language_counts[item["language"]] += 1
                template_counts[item["question"]] += 1
    return result


def main(args):
    if args.out.exists():
        raise ValueError("Choose a new pack directory; existing data is immutable")
    previous = [json.loads(p.read_text(encoding="utf-8")) for p in args.exclude]
    excluded = {group(c) for manifest in previous for c in manifest["cases"]}
    old_hashes = {c["input_sha256"] for manifest in previous for c in manifest["cases"]}
    # This parser separates evaluator answers; selection uses only metadata above.
    _, _, population = select_cases(args.questions, args.inventory, count=1, seed=args.seed)
    cases, answers, _ = select_cases(args.questions, args.inventory,
                                    count=population["eligible_unique_questions"], seed=args.seed)
    selected = choose(cases, excluded, args.seed)
    selected.sort(key=lambda c: sha256((args.seed + c["id"]).encode()).hexdigest())
    args.out.mkdir(parents=True)
    write_json(args.out / "selection.json", {
        "seed": args.seed, "quotas_per_split": QUOTAS,
        "case_ids": [c["id"] for c in selected], "excluded_groups": sorted(excluded),
        "answer_blind": True, "holdout_policy": "No holdout output or scores inspected until candidate is frozen."})
    artifacts = {}
    members = {c["archive_member"]: c["input_path"] for c in selected}
    members["junction-dataset/README.md"] = "notices/README.md"
    for source in {c["source"] for c in selected}:
        members[f"junction-dataset/data/{source}/LICENSE.md"] = f"notices/{source}/LICENSE.md"
    with RangeArchive(args.archive, budget=256 * 1024 * 1024) as remote:
        with zipfile.ZipFile(remote) as archive:
            for i, (member, relative) in enumerate(sorted(members.items()), 1):
                info = archive.getinfo(member)
                if info.file_size > 64 * 1024 * 1024:
                    raise ValueError("Source member exceeds the existing 64 MiB limit")
                target = safe_path(args.out, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
                artifacts[relative] = {"sha256": file_sha256(target), "bytes": info.file_size,
                                       "archive_member": member, "crc32": f"{info.CRC:08x}"}
                print(json.dumps({"downloaded": i, "total": len(members)}), flush=True)
        downloaded, etag, archive_bytes = remote.downloaded, remote.etag, remote.size
    seen = set(old_hashes)
    for case in selected:
        case["input_sha256"] = artifacts[case["input_path"]]["sha256"]
        if case["input_sha256"] in seen:
            raise ValueError("Byte-identical source overlap detected; pack is not sealed")
        seen.add(case["input_sha256"])
    evaluator = args.out / "evaluator/answers.json"
    write_json(evaluator, {c["id"]: answers[c["id"]] for c in selected})
    manifest = {"schema": "planproof-stoa-random-v1", "partition": "evaluation/reference; not final",
                "cases": selected, "artifacts": artifacts, "answers_sha256": file_sha256(evaluator),
                "question_file_sha256": file_sha256(args.questions),
                "inventory_sha256": file_sha256(args.inventory), "archive_url": args.archive,
                "archive_bytes": archive_bytes, "archive_etag": etag,
                "range_bytes_downloaded": downloaded, "population": population,
                "limitations": ["Fresh reference drawings for this agent session, not organizer-final data.",
                                "Grouped by drawing basename, not independently audited by building family.",
                                "Publisher/pretrained-model exposure is unknown."]}
    write_json(args.out / "manifest.json", manifest)
    verify_pack(args.out)
    print(json.dumps({"sealed": len(selected), "downloaded_bytes": downloaded,
                      "categories": dict(Counter(c["category"] for c in selected)),
                      "sources": dict(Counter(c["source"] for c in selected))}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, nargs="+", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", default="planproof-agent-lab-20261115-v1")
    main(parser.parse_args())
