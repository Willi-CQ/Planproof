"""Start the integrated local app with host-selected worker environments."""

import argparse
import os
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-python", type=Path, required=True)
    parser.add_argument("--geometry-python", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--gemini-env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()
    os.environ["PLANPROOF_ML_PYTHON"] = str(args.ml_python.resolve())
    os.environ["PLANPROOF_DETERMINISTIC_PYTHON"] = str(args.geometry_python.resolve())
    os.environ["PLANPROOF_ENV_FILE"] = str(args.env_file.resolve())
    os.environ["PLANPROOF_GEMINI_ENV_FILE"] = str(args.gemini_env_file.resolve())
    os.environ["PLANPROOF_ML_DEVICE"] = args.device
    bundle = ROOT / "models/baselines/v1/corrected-3500-global"
    os.environ.setdefault("PLANPROOF_ML_CHECKPOINT", str(bundle / "detector.pt"))
    os.environ.setdefault("PLANPROOF_ML_CALIBRATION", str(bundle / "calibration.json"))
    os.environ["PLANPROOF_AGENT_THINKING"] = str(args.enable_thinking).lower()
    pid = ROOT / ".artifacts/server/server-pid.txt"
    pid.parent.mkdir(parents=True, exist_ok=True)
    pid.write_text(str(os.getpid()), encoding="utf-8")
    # Keep inference credentials in the literal env file. The feedback transport
    # reads only its configured key; algorithm subprocesses receive no secrets.
    sys.path.insert(0, str(ROOT / "backend"))
    # Use the checkout being launched, even when its interpreter comes from a
    # sibling worktree with an editable installation of an older feedback package.
    sys.path.insert(0, str(ROOT / "feedback" / "src"))
    uvicorn.run("app:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
