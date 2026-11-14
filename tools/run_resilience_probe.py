"""Exercise a local live upload/ask pipeline without opening evaluator answers.

Writes new development-run artifacts, never overwrites a benchmark or retries a
completed HTTP request. Model-call retries belong to the server transport.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import httpx


def run(args):
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    chosen = [int(n) for n in args.cases.split(",")]
    if len(set(chosen)) != len(chosen) or not all(
        1 <= n <= len(inventory["cases"]) for n in chosen
    ):
        raise ValueError("Invalid case selection")
    args.out.mkdir(parents=True, exist_ok=False)
    binding = {
        "server": args.url,
        "condition": args.condition,
        "case_numbers": chosen,
        "inventory_sha256": hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        "labels_used_for_inference": False,
        "limitation": inventory.get(
            "limitation",
            "Selected previously observed cases; sequential development smoke check, not a causal benchmark.",
        ),
    }
    (args.out / "binding.json").write_text(
        json.dumps(binding, indent=2), encoding="utf-8"
    )
    results = []
    with httpx.Client(timeout=900, trust_env=False) as client:
        for number in chosen:
            case = inventory["cases"][number - 1]
            pack = Path(inventory["source_pack"]).resolve()
            source = (pack / case["input_path"]).resolve()
            if not source.is_relative_to(pack):
                raise ValueError("Source path escapes pack")
            payload = source.read_bytes()
            if hashlib.sha256(payload).hexdigest() != case["input_sha256"]:
                raise ValueError("Source hash mismatch")
            started = time.monotonic()
            print(
                json.dumps({"starting_case": number, "condition": args.condition}),
                flush=True,
            )
            upload = client.post(
                args.url + "/api/documents",
                files={"file": (source.name, payload)},
                data={"page": "1"},
            )
            upload.raise_for_status()
            document = upload.json()
            response = client.post(
                args.url + f"/api/v1/documents/{document['document_id']}/ask",
                json={
                    "question": case["question"],
                    "unit": case.get("unit"),
                    "mode": "auto",
                    "allow_gemini": True,
                    "allow_nemotron": True,
                },
            )
            response.raise_for_status()
            result = response.json()
            record = {
                "case_number": number,
                "id": case["id"],
                "question": case["question"],
                "seconds_http": round(time.monotonic() - started, 3),
                "result": result,
            }
            results.append(record)
            (args.out / f"case-{number}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "completed_case": number,
                        "status": result["status"],
                        "seconds": record["seconds_http"],
                        "stop": result.get("stop_reason"),
                    }
                ),
                flush=True,
            )
    (args.out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8002")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cases", default="3,30,37,10")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
