"""Portable local MVP bootstrap. Private environments stay under .artifacts/."""

import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVS = ROOT / ".artifacts" / "venvs"


def interpreter(name):
    return (
        ENVS
        / name
        / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    )


def install_commands():
    return {
        "host": [
            "-r",
            str(ROOT / "backend/requirements.txt"),
            "-e",
            str(ROOT / "feedback") + "[test]",
        ],
        "ml": ["-e", str(ROOT / "ml")],
        "geometry": ["-e", str(ROOT / "deterministic")],
    }


def launch_command(port):
    return [
        str(interpreter("host")),
        str(ROOT / "backend/run_local.py"),
        "--ml-python",
        str(interpreter("ml")),
        "--geometry-python",
        str(interpreter("geometry")),
        "--env-file",
        str(ROOT / ".env"),
        "--gemini-env-file",
        str(ROOT / ".env"),
        "--port",
        str(port),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "run"))
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    if args.action == "setup":
        if not (3, 11) <= sys.version_info[:2] < (3, 13):
            parser.error(
                "Run setup with Python 3.11 or 3.12 (the ML package requires it)"
            )
        for name, requirements in install_commands().items():
            python = interpreter(name)
            if not python.is_file():
                venv.EnvBuilder(with_pip=True).create(ENVS / name)
            subprocess.run(
                [str(python), "-m", "pip", "install", *requirements],
                cwd=ROOT,
                check=True,
            )
        if not (ROOT / ".env").exists():
            shutil.copyfile(ROOT / ".env.example", ROOT / ".env")
        print(
            "Setup finished. Fill the private .env, fetch the approved LFS weights, then run: python tools/mvp.py run"
        )
        return
    missing = [name for name in install_commands() if not interpreter(name).is_file()]
    if missing:
        parser.error(
            "Missing environments: " + ", ".join(missing) + ". Run setup first."
        )
    checkpoint = ROOT / "models/baselines/v1/corrected-3500-global/detector.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1024:
        parser.error(
            "Missing model payload (possibly an LFS pointer). Run git lfs pull first."
        )
    if not (ROOT / ".env").is_file():
        parser.error("Copy .env.example to .env and set server-side credentials.")
    subprocess.run(launch_command(args.port), cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
