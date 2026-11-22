"""Package the web app and its same-origin CPU reader from one committed revision."""

import argparse
import json
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
BUILD_FILES = {"index.html", "package.json", "package-lock.json", "vite.config.ts"}
PYTHON_FILES = {
    "api/reader.py", "backend/scene_geometry.py", "backend/scene_native.py",
    "models/sheet-doors-v1/eteks-door.npz", "models/sheet-doors-v1/open-symbol.npz",
    "models/sheet-doors-v1/sheet-config.json", "models/sheet-doors-v1/example-config.json",
    "models/sheet-doors-v1/LICENSE-CODE.txt", "models/sheet-doors-v1/NOTICE.md",
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def stage(output, revision="HEAD"):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty staging directory; existing contents are preserved")
    commit = git("rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip()
    template = json.loads(git("show", f"{commit}:deploy/vercel.template.json"))
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for name in git("ls-tree", "-r", "--name-only", commit, "frontend/").decode().splitlines():
        path = PurePosixPath(name).relative_to("frontend")
        if not (path.parts[0] in {"src", "public"} or str(path) in BUILD_FILES
                or (len(path.parts) == 1 and path.name.startswith("tsconfig")
                    and path.suffix == ".json")):
            continue
        if path.name.startswith(".env") or any(p.startswith(".") for p in path.parts):
            raise ValueError(f"Unexpected hidden source path: {path}")
        target = output.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git("show", f"{commit}:{name}"))
        files.append(str(path))
    if not BUILD_FILES.issubset(set(files)):
        raise ValueError("Missing committed frontend build files")
    paths = git("ls-tree", "-r", "--name-only", commit, "tools/free_agent_room_qa/",
                "models/sheet-doors-v1/", "planproof_runtime/").decode().splitlines()
    runtime = set(PYTHON_FILES)
    for name in paths:
        if (name.startswith("planproof_runtime/") or
                (name.endswith(".py") and not name.endswith("/train.py"))):
            runtime.add(name)
    for name in sorted(runtime):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git("show", f"{commit}:{name}"))
        files.append(name)
    (output / "requirements.txt").write_bytes((output / "planproof_runtime/requirements.txt").read_bytes())
    (output / ".python-version").write_text("3.12\n", encoding="utf-8")
    (output / "planproof_runtime/revision.txt").write_text(commit + "\n", encoding="ascii")
    (output / "vercel.json").write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    (output / ".vercelignore").write_text(
        "node_modules\ndist\n.env*\n*.log\n.git\n.brain\n.artifacts\n", encoding="utf-8"
    )
    return {"commit": commit, "frontend_tree": git("rev-parse", f"{commit}:frontend").decode().strip(),
            "files": len(files), "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    print(json.dumps(stage(args.output, args.revision), indent=2))
