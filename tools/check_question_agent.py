"""Run one real upload/question request against the local integrated app."""

import argparse
import json
from pathlib import Path

import httpx


def run(args):
    if args.out.exists():
        raise ValueError("Choose a new output file to preserve previous traces")
    with httpx.Client(
        base_url="http://127.0.0.1:8002", timeout=720, trust_env=False
    ) as client:
        with args.input.open("rb") as stream:
            upload = client.post(
                "/api/documents",
                files={"file": (args.input.name, stream)},
                data={"page": str(args.page)},
            )
        upload.raise_for_status()
        doc = upload.json()
        response = client.post(
            f"/api/v1/documents/{doc['document_id']}/ask",
            json={
                "question": args.question,
                "mode": args.mode,
                "allow_gemini": True,
                "allow_nemotron": args.nemotron,
            },
        )
        response.raise_for_status()
    result = response.json()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "algorithm_used": result.get("algorithm_used"),
                "nemotron_calls": result.get("nemotron_calls"),
                "tools": [t["name"] for t in result.get("tool_trace", [])],
                "citations": len(result.get("canonical_citations", [])),
                "seconds": result.get("seconds"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument(
        "--mode", choices=("auto", "visual_only", "direct"), default="auto"
    )
    parser.add_argument("--nemotron", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
