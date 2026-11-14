"""Try the real Gemini review on a synthetic page and optionally start the API."""

import argparse
import getpass
import os
import sys
import warnings

import uvicorn

from app import app, door_count_demo


DEFAULT_MODEL = "gemini-3.8-flash"
ERROR_MESSAGES = {
    "request_rejected": "Google refused the request. Check the key and selected model.",
    "access_denied": "Check that the API key belongs to the Junction project and has Gemini access.",
    "model_unavailable": "Set GEMINI_MODEL to an image-capable model available in your project.",
    "quota": "Check the project quota in Google AI Studio.",
    "provider_error": "Google returned a service error.",
    "connection": "Could not reach the Gemini API.",
    "timeout": "Gemini did not respond within the time allowed.",
    "invalid_response": "Gemini returned an incomplete response or invalid evidence references.",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="Start the local API after a successful check.")
    parser.add_argument("--port", type=int, default=8001, help="Local port for --serve (default 8001).")
    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY", "").strip():
        if not sys.stdin.isatty():
            print("Open an interactive terminal or configure GEMINI_API_KEY in the process environment.")
            return 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                key = getpass.getpass("Gemini API key from Google AI Studio (hidden): ").strip()
        except (EOFError, getpass.GetPassWarning):
            print("Hidden input is unavailable. Run this command in PowerShell.")
            return 1
        if not key:
            print("A Gemini API key is required.")
            return 1
        os.environ["GEMINI_API_KEY"] = key
    model = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    os.environ["GEMINI_MODEL"] = model
    os.environ["GEMINI_ENABLED"] = "true"
    print(f"Sending one synthetic plan crop to {model}.", flush=True)
    result = door_count_demo(scenario="disputed", allow_gemini_review=True)
    review = result.gemini
    if review.status != "reviewed" or review.advisory is None:
        print(f"Gemini check failed ({review.error_code or review.status}).")
        if review.error_code:
            print(ERROR_MESSAGES[review.error_code])
        return 1
    print(f"Gemini responded: {review.advisory.resolution}. PlanProof status: {result.status}.")
    print("Cited evidence IDs:", ", ".join(review.advisory.evidence_ids) or "none")
    if args.serve:
        print(f"Open Planwise at http://127.0.0.1:{args.port}/")
        uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
