"""Render a source-first, offline direct-versus-guided comparison."""

import argparse
import json
import os
import time
from pathlib import Path


def build(run, view):
    data = json.loads((run / "report.json").read_text(encoding="utf-8"))
    for row in data["cases"]:
        for condition in ("direct", "guided"):
            side = row[condition]
            directory = run / row["id"] / condition
            side["artifact"] = Path(
                os.path.relpath(directory / "result.json", view)
            ).as_posix()
            side["raw_calls"] = []
            side["raw_usage"] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            side["responses_with_usage"] = 0
            for call in sorted((directory / "calls").glob("*")):
                side["raw_calls"].append(
                    {
                        "id": call.name,
                        "files": [
                            {
                                "name": p.name,
                                "path": Path(os.path.relpath(p, view)).as_posix(),
                            }
                            for p in sorted(call.iterdir())
                            if p.is_file() and p.suffix != ".tmp"
                        ],
                    }
                )
                for response in call.glob("response-*.json"):
                    try:
                        usage = json.loads(response.read_text(encoding="utf-8")).get(
                            "usage", {}
                        )
                    except (ValueError, UnicodeError):
                        continue
                    if usage:
                        side["responses_with_usage"] += 1
                        for key in side["raw_usage"]:
                            if isinstance(usage.get(key), int):
                                side["raw_usage"][key] += usage[key]
            job_path = side.get("job_directory")
            if job_path:
                job = Path(job_path)
                for name, relative in (
                    ("ml_file", "ml/detections.json"),
                    ("geometry_file", "evidence/evidence.json"),
                    ("rooms_file", "rooms/room-boundaries.json"),
                ):
                    side[name] = Path(os.path.relpath(job / relative, view)).as_posix()
                side["crops"] = []
                for region in sorted((job / "crops").glob("*/region.json")):
                    raw = json.loads(region.read_text(encoding="utf-8"))
                    side["crops"].append(
                        {
                            "name": raw.get("room_id", region.parent.name),
                            "path": Path(
                                os.path.relpath(region.parent / raw["clean_crop"], view)
                            ).as_posix(),
                        }
                    )
    data["report_file"] = Path(os.path.relpath(run / "report.json", view)).as_posix()
    data["binding_file"] = Path(os.path.relpath(run / "binding.json", view)).as_posix()
    template = Path(__file__).with_name("paired_pilot.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    html = template.replace("__DATA__", payload)
    # No fetch/server is needed for file://. Reload preserves the selected ID hash.
    html = html.replace(
        "__REFRESH__",
        '<meta http-equiv="refresh" content="20">'
        if data["status"] == "running"
        else "",
    )
    temporary = view / "paired-index.tmp"
    temporary.write_text(html, encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(view / "index.html")
            return
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))
    # Windows readers may permit writing but hold a handle that denies rename.
    (view / "index.html").write_text(html, encoding="utf-8")
    temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("view", type=Path)
    args = parser.parse_args()
    build(args.run.resolve(), args.view.resolve())
