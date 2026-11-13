"""Fuehrt alle Goldset-Plaene durch eine Pipeline und schreibt den Vergleichsbericht.

    python run_goldset.py --pipeline planproof_geometry --out C:/tmp/goldset-run/core
    python run_goldset.py --pipeline planproof_cad --out C:/tmp/goldset-run/plan
    python run_goldset.py --pipeline mein_modul.pfad:meine_funktion

Train/Held-out-Trennung: ``--split dev|heldout|all`` (Standard ``dev``). Held-out sind die Plaene mit
``int(sha256(plan_id).hexdigest(), 16) % 4 == 0`` (10 von 40 Plaenen, 20 von 74 Dateien); dev ist der Rest.
Entwickelt wird nur gegen dev. Jeder Lauf mit ``--split heldout`` oder ``all`` wird mit Zeitpunkt, Pipeline
und Kommentar (``--comment``) in ``heldout_runs.log`` angehaengt; das Log wird eingecheckt.

Die Pipeline-Funktion hat die Signatur ``f(input_path, output_dir, ...) -> Ergebnis`` und schreibt
``scene.json`` in ``output_dir`` (wie ``run_dwg_to_bim_pipeline``). Der Bericht entsteht aus dieser
Datei. Die Pipeline wird erst hier importiert (lazy); das Paket muss in der Python-Umgebung liegen,
mit der dieses Skript laeuft.

Es werden keine Cloud- oder Modell-Aufrufe gemacht; die Pipeline laeuft lokal.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goldset import metrics  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

PIPELINES: Dict[str, str] = {
    "planproof_geometry": "planproof_geometry.ai.dwg_bim.pipeline:run_dwg_to_bim_pipeline",
    "planproof_cad": "planproof_cad.cad.pipeline:run_dwg_to_bim_pipeline",
}


HELDOUT_LOG = HERE / "heldout_runs.log"
SPLITS = ("dev", "heldout", "all")


def is_heldout(plan_id: str) -> bool:
    """Held-out-Plan: ``int(sha256(plan_id).hexdigest(), 16) % 4 == 0``."""
    return int(hashlib.sha256(str(plan_id).encode("utf-8")).hexdigest(), 16) % 4 == 0


def in_split(plan_id: str, split: str) -> bool:
    if split == "all":
        return True
    if split == "heldout":
        return is_heldout(plan_id)
    if split == "dev":
        return not is_heldout(plan_id)
    raise ValueError(f"unbekannter Split {split!r}; erlaubt: {SPLITS}")


def log_heldout_run(pipeline: str, split: str, n_files: int, comment: str, log_path: Path = HELDOUT_LOG) -> None:
    """Haengt einen Lauf, der Held-out-Plaene beruehrt, an ``heldout_runs.log`` an."""
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"{stamp}\tsplit={split}\tpipeline={pipeline}\tdateien={n_files}\t{comment.strip() or '-'}\n"
    with open(log_path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)


def resolve_pipeline(name: str) -> Callable[..., Any]:
    spec = PIPELINES.get(name, name)
    if ":" not in spec:
        raise SystemExit(f"unbekannte Pipeline {name!r}; bekannt: {sorted(PIPELINES)} oder 'modul:funktion'")
    mod, func = spec.split(":", 1)
    return getattr(importlib.import_module(mod), func)


def run_one(fn: Callable[..., Any], dxf: Path, out_dir: Path, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = fn(str(dxf), str(out_dir), config=config) if config else fn(str(dxf), str(out_dir))
    ok = getattr(res, "ok", True)
    err = getattr(res, "error", None)
    scene_path = out_dir / "scene.json"
    return {
        "ok": bool(ok), "fehler": err, "dauer_s": round(time.time() - t0, 2),
        "scene": scene_path if scene_path.exists() else None,
        "warnungen": list(getattr(res, "warnings", []) or [])[:3],
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=None, help="Ausgabeordner (Standard: goldset/_runs/<pipeline>)")
    ap.add_argument("--only", default=None, help="nur Dateien, deren Name diesen Text enthaelt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--open-tol", type=float, default=metrics.OPEN_TOL_M)
    ap.add_argument("--split", choices=SPLITS, default="dev",
                    help="dev (Standard, 30 Plaene), heldout (10 Plaene, wird protokolliert) oder all")
    ap.add_argument("--config-json", default=None,
                    help="Pipeline-Konfiguration als JSON-Objekt, z. B. "
                         "'{\"wall_extend_junctions\": false}' (fuer Ablationen)")
    ap.add_argument("--comment", default="", help="Kommentar fuer heldout_runs.log (bei heldout und all)")
    ap.add_argument("--heldout-log", type=Path, default=HELDOUT_LOG, help=argparse.SUPPRESS)
    args = ap.parse_args()

    fn = resolve_pipeline(args.pipeline)
    pipeline_config = json.loads(args.config_json) if args.config_json else None
    out_root = args.out or (HERE / "_runs" / args.pipeline.replace(":", "_").replace(".", "_"))
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))

    per_plan: List[Dict[str, Any]] = []
    files = [(p, f) for p in manifest["plan_liste"] if in_split(p["plan_id"], args.split) for f in p["dateien"]]
    if args.split in ("heldout", "all"):
        log_heldout_run(args.pipeline, args.split, len(files), args.comment, args.heldout_log)
    if args.only:
        files = [(p, f) for p, f in files if args.only in f["dxf"]]
    if args.limit:
        files = files[: args.limit]
    t_all = time.time()
    for n, (p, f) in enumerate(files, 1):
        dxf = args.data / f["dxf"]
        gt = metrics.load_gt(args.data / f["gt_json"])
        item: Dict[str, Any] = {
            "datei": f["dxf"], "plan_key": p["plan_key"], "klasse": p["klasse"],
            "konvention": f["konvention"], "variante": f["variante"], "fehler": None, "metriken": None,
            "gt_zahlen": {"tueren": gt["gt"]["tueren"], "fenster": gt["gt"]["fenster"], "raeume": gt["gt"]["raeume"]},
        }
        try:
            r = run_one(fn, dxf, out_root / Path(f["dxf"]).stem, pipeline_config)
            item["dauer_s"] = r["dauer_s"]
            item["warnungen"] = r["warnungen"]
            if not r["ok"] or r["scene"] is None:
                item["fehler"] = r["fehler"] or "keine scene.json"
            else:
                result = metrics.extract_result(metrics.load_scene(r["scene"]))
                origin = metrics.dxf_loader_origin(dxf)
                item["metriken"] = metrics.evaluate_plan(gt, result, origin_hint=origin, open_tol_m=args.open_tol)
        except Exception as exc:  # noqa: BLE001
            item["fehler"] = f"{type(exc).__name__}: {exc}"
            item["traceback"] = traceback.format_exc(limit=3)
        per_plan.append(item)
        m = item["metriken"]
        print(f"[{n}/{len(files)}] {Path(f['dxf']).name}: " + (
            f"Wand {m['waende']['abw_pct']:+.1f} %, Tueren {m['tueren']['treffer']}/{m['tueren']['gt']}, "
            f"Fenster {m['fenster']['treffer']}/{m['fenster']['gt']}, Raeume {m['raeume']['treffer']}/{m['raeume']['gt']}"
            if m else f"FEHLER {item['fehler']}"), flush=True)

    summary = metrics.summarize(per_plan)
    report = {
        "pipeline": args.pipeline, "split": args.split, "config": pipeline_config, "dateien": len(per_plan), "dauer_s": round(time.time() - t_all, 1),
        "hinweis": "Die Zeichenkonventionen sind synthetisch gewaehlt; das Goldset misst die Pipeline auf "
                   "Swiss-Dwellings-Geometrie mit selbst gewaehlten Layern, nicht auf echten Buero-Exporten.",
        "zusammenfassung": summary, "je_datei": per_plan,
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    md = metrics.summary_markdown(summary, title=f"Goldset: {args.pipeline} (Split {args.split})")
    md += "\n" + report["hinweis"] + "\n"
    (out_root / "report.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Bericht: {out_root / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
