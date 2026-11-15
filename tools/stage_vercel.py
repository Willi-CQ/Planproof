"""Stage only committed frontend source for Vercel, never the private workspace."""

import argparse
import json
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
BUILD_FILES = {"index.html", "package.json", "package-lock.json", "vite.config.ts"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def stage(output, origin, revision="HEAD"):
    parsed = urlsplit(origin)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
        raise ValueError("Supply an HTTPS origin without credentials, path or query")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty staging directory; existing contents are preserved")
    commit = git("rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip()
    template = json.loads(git("show", f"{commit}:deploy/vercel.template.json"))
    template["routes"][0]["dest"] = origin.rstrip("/") + "/api/$1"
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
    (output / "vercel.json").write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    (output / ".vercelignore").write_text(
        "node_modules\ndist\n.env*\n*.log\n.git\n.brain\n.artifacts\n", encoding="utf-8"
    )
    return {"commit": commit, "frontend_tree": git("rev-parse", f"{commit}:frontend").decode().strip(),
            "files": len(files), "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-origin", required=True)
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    print(json.dumps(stage(args.output, args.api_origin, args.revision), indent=2))
