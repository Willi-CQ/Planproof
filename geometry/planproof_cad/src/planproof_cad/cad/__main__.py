"""CLI fuer die DWG/DXF -> BIM-Pipeline.

Beispiele::

    # Echte Zeichnung testen (DXF immer; DWG braucht ODA/LibreDWG-Bruecke)
    python -m planproof_cad.cad run plan.dxf ./out
    python -m planproof_cad.cad run plan.dwg ./out --unit mm

    # Demo-Grundriss erzeugen + komplett durchrechnen (kein Input noetig)
    python -m planproof_cad.cad demo ./out

Schreibt nach <out>: plan.ifc, model.glb, scene.json, quantity_takeoff.csv, meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _print_summary(res) -> None:
    print(f"\n  ok            : {res.ok}")
    if res.error:
        print(f"  error         : {res.error}")
    m = res.metrics
    print(f"  unit          : {m.get('unit')}")
    print(f"  wall segments : {m.get('wall_segments')}")
    wr = m.get("wall_reconstruction", {})
    print(f"  walls (axes)  : {wr.get('centerlines')} (mode={wr.get('mode')})")
    op = m.get("openings", {})
    print(
        f"  openings      : {op.get('openings_attached')} "
        f"(doors={op.get('doors')}, windows={op.get('windows')})"
    )
    print(f"  rooms         : {m.get('rooms')}")
    ifc = m.get("ifc", {})
    print(f"  IFC elements  : {ifc.get('element_count')} ({ifc.get('schema')})")
    print(f"  duration      : {m.get('duration_s')} s")
    if res.warnings:
        print("  warnings:")
        for w in res.warnings:
            print(f"    - {w}")
    print("\n  artifacts:")
    for a in res.artifacts:
        print(f"    {a}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="dwg_bim", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="DWG/DXF -> BIM")
    p_run.add_argument("input", help="Pfad zu .dwg oder .dxf")
    p_run.add_argument("output_dir", help="Ausgabeverzeichnis")
    p_run.add_argument(
        "--unit",
        choices=["mm", "cm", "m", "in", "ft"],
        help="Einheit erzwingen (sonst $INSUNITS/Heuristik)",
    )
    p_run.add_argument("--height", type=float, default=2.80, help="Wandhoehe in m")
    p_run.add_argument(
        "--converter",
        choices=["none", "oda", "libredwg"],
        help="DWG-Konverter (sonst Umgebungsvariable CQPLAN_DWG_CONVERTER, sonst keiner); "
        "siehe docs/ODA_AND_DWG.md",
    )

    p_demo = sub.add_parser("demo", help="Demo-Grundriss erzeugen + durchrechnen")
    p_demo.add_argument("output_dir", help="Ausgabeverzeichnis")

    args = parser.parse_args(argv)

    from planproof_cad.cad import run_dwg_to_bim_pipeline

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_work"

    if args.cmd == "demo":
        from planproof_cad.cad.synth import build_demo_plan

        dxf = build_demo_plan(out_dir / "demo.dxf")
        print(f"Demo-DXF erzeugt: {dxf}")
        input_path = str(dxf)
        config = {}
    else:
        input_path = args.input
        if not Path(input_path).exists():
            print(f"FEHLER: Eingabe nicht gefunden: {input_path}", file=sys.stderr)
            return 2
        config = {"wall_height_m": args.height}
        if args.unit:
            config["dxf_unit"] = args.unit
        if args.converter:
            config["converter"] = args.converter

    print(f"Pipeline laeuft: {input_path} -> {out_dir}")
    res = run_dwg_to_bim_pipeline(
        input_path,
        str(out_dir),
        str(work_dir),
        config,
        logger_fn=lambda msg: print(f"  [log] {msg}"),
    )
    _print_summary(res)

    (out_dir / "result.json").write_text(
        json.dumps(
            {"ok": res.ok, "metrics": res.metrics, "warnings": res.warnings, "error": res.error},
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
