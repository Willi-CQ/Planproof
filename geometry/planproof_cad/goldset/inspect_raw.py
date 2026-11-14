"""Zaehlt die Rohdaten von Swiss Dwellings v3.0.0 (geometries.csv) und schreibt data/raw_stats.json.

Aufruf:
    python inspect_raw.py --csv C:/tmp/swiss-dwellings/raw/geometries.csv \
        --index-cache C:/tmp/swiss-dwellings/floor_index.json

Der Index je Geschoss wird zusaetzlich als JSON neben die CSV gelegt (nicht ins Repo),
damit build_goldset.py den ersten Durchlauf (ca. 1-2 Minuten) nicht wiederholen muss.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goldset import sd_data  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"


def save_index_cache(path: Path, scan: dict) -> None:
    floors = {
        f"{p}|{f}": {"n": dict(v["n"]), "hash": v["hash"]} for (p, f), v in scan["floors"].items()
    }
    path.write_text(json.dumps({"stats": scan["stats"], "floors": floors}), encoding="utf-8")


def load_index_cache(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    floors = {}
    for k, v in d["floors"].items():
        p, f = k.split("|")
        floors[(p, f)] = {"n": v["n"], "hash": v["hash"]}
    return {"stats": d["stats"], "floors": floors}


def get_scan(csv_path: Path, cache: Path | None, progress: bool = True) -> dict:
    if cache is not None and cache.exists() and cache.stat().st_mtime >= csv_path.stat().st_mtime:
        return load_index_cache(cache)
    scan = sd_data.scan_index(csv_path, progress=progress)
    if cache is not None:
        save_index_cache(cache, scan)
    return scan


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--index-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "raw_stats.json")
    args = ap.parse_args()

    t0 = time.time()
    scan = get_scan(args.csv, args.index_cache)
    stats = dict(scan["stats"])
    stats["repetition"] = sd_data.repetition_stats(scan["floors"])
    by_plan = sd_data.plan_floor_table(scan["floors"])
    walls_per_plan = sorted(fl[0][1]["n"].get("separator/WALL", 0) for fl in by_plan.values())
    n = len(walls_per_plan)
    stats["walls_per_first_floor_of_plan"] = {
        "min": walls_per_plan[0],
        "p10": walls_per_plan[n // 10],
        "median": walls_per_plan[n // 2],
        "p90": walls_per_plan[(9 * n) // 10],
        "max": walls_per_plan[-1],
    }
    stats["source"] = {
        "file": "swiss-dwellings-v3.0.0.zip / swiss-dwellings-v3.0.0/geometries.csv",
        "doi": "10.5281/zenodo.7788422",
        "scan_seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats["repetition"], indent=2))
    print(f"geschrieben: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
