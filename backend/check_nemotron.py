"""Run PlanProof with Nemotron or test it on a synthetic disagreement."""

import argparse
import getpass
import os
import sys
import warnings
from time import perf_counter

import httpx
import uvicorn

from app import app, door_count_demo
from nemotron_review import discover_model, nemotron_client


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Model ID served by Verda. Defaults to detection via /v1/models.")
    parser.add_argument("--serve", action="store_true", help="Start the API in Nemotron mode.")
    parser.add_argument("--port", type=int, default=8001, help="Local API port (default: 8001).")
    args = parser.parse_args()
    key = os.getenv("NEMOTRON_API_KEY", "").strip()
    if not key:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                key = getpass.getpass("Verda inference API key (hidden): ").strip()
        except (EOFError, getpass.GetPassWarning):
            print("Run in an interactive terminal or set NEMOTRON_API_KEY in the process environment.")
            return 1
    if not key:
        print("A Verda inference API key is required.")
        return 1

    model = args.model or os.getenv("NEMOTRON_MODEL", "").strip()
    if not model:
        print("Contacting Verda to identify the served model.", flush=True)
        try:
            with nemotron_client(key) as client:
                model = discover_model(client)
        except httpx.HTTPStatusError as exc:
            print(f"Verda model discovery failed (HTTP {exc.response.status_code}).")
            return 1
        except httpx.RequestError:
            print("Could not reach Verda during model discovery.")
            return 1
        except (ValueError, LookupError, TypeError):
            print("Could not identify a single model. Use --model with the model ID served by Verda.")
            return 1

    os.environ["NEMOTRON_API_KEY"] = key
    os.environ["NEMOTRON_MODEL"] = model
    os.environ["PLANPROOF_REVIEW_PROVIDER"] = "nemotron"
    if args.serve:
        print(f"Open http://127.0.0.1:{args.port}/docs and set allow_nemotron_review=true in the door-count demo.")
        uvicorn.run(app, host="127.0.0.1", port=args.port)
        return 0

    print("Sending one synthetic crop to Nemotron.", flush=True)
    started = perf_counter()
    result = door_count_demo(scenario="disputed", allow_nemotron_review=True)
    review = result.nemotron
    if review.advisory is None:
        print(f"Nemotron review failed ({review.error_code or review.status}).")
        if review.detail:
            print(review.detail)
        return 1
    print(f"Nemotron responded: {review.advisory.resolution}. PlanProof status: {result.status}.")
    print("Cited evidence IDs:", ", ".join(review.advisory.evidence_ids) or "none")
    print(f"Review took {perf_counter() - started:.2f} seconds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
