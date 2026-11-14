"""Kommandozeile ``planproof-cad``.

Unterbefehle::

    planproof-cad demo ./out                      # synthetischen Demo-Grundriss durchrechnen
    planproof-cad convert plan.dxf ./out          # DXF in BIM umrechnen (+ Mengen, KG-Vorschlag)
    planproof-cad takeoff scene.json --geschoss EG [-o ./out]   # Mengen je Bauteilart
    planproof-cad kg mengen.json                  # Kostengruppen-Vorschlaege (zur Freigabe)
    planproof-cad layers plan.dxf                 # Layer, Klassifikation, Entity-Zahlen
    planproof-cad serve-rest                      # REST-Schnittstelle (Extra "rest")
    planproof-cad serve-mcp --out-dir ./out       # MCP-Server ueber stdio

Mit ``--json`` steht ausschliesslich ein JSON-Dokument auf stdout; Fortschritt und Log gehen
dann nach stderr. Die Schwergewichte (Pipeline, ezdxf, ifcopenshell, fastapi) werden erst im
jeweiligen Unterbefehl importiert, ``planproof-cad --help`` bleibt schnell. ``takeoff`` und ``kg`` auf
einer scene.json laden weder ezdxf noch ifcopenshell.

``convert`` schreibt nach <out>: plan.ifc, model.glb, scene.json, quantity_takeoff.csv, meta.json,
result.json und die deutschen Mengen-Dateien mengen.json, mengen_de.csv, kg_vorschlaege.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

CommandFn = Callable[[argparse.Namespace], int]

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------


def _meldung(args: argparse.Namespace, text: str) -> None:
    """Fortschritt: stdout, mit --json aber stderr (stdout bleibt reines JSON)."""
    print(text, file=sys.stderr if getattr(args, "json", False) else sys.stdout)


def _json_ausgabe(payload: Any) -> None:
    # ensure_ascii: stdout-Kodierung ist egal (Pipe, Konsole, Windows-Codepage).
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def _fehler(args: argparse.Namespace, code: str, text: str) -> int:
    if getattr(args, "json", False):
        _json_ausgabe({"ok": False, "error": {"code": code, "message": text}})
    else:
        print(f"FEHLER: {text}", file=sys.stderr)
    return EXIT_USAGE


def _print_summary(res: dict[str, Any]) -> None:
    print(f"\n  ok            : {res['ok']}")
    if res["error"]:
        print(f"  error         : {res['error']}")
    m = res["metrics"]
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
    ta = res["template_assignment"]
    print(
        f"  templates     : ok={ta.get('ok')} geladen={ta.get('templates_loaded')}"
        + (f" Fehler={ta.get('error')}" if ta.get("error") else "")
    )
    print(f"  duration      : {m.get('duration_s')} s")
    if res["warnings"]:
        print("  warnings:")
        for w in res["warnings"]:
            print(f"    - {w}")
    print("\n  artifacts:")
    for a in res["artifacts"]:
        print(f"    {a}")


def _run_pipeline(
    args: argparse.Namespace, input_path: str, out_dir: Path, config: dict[str, Any]
) -> int:
    from planproof_cad import services

    _meldung(args, f"Pipeline laeuft: {input_path} -> {out_dir}")
    try:
        res = services.run_dxf(
            input_path,
            out_dir,
            config,
            geschoss=getattr(args, "geschoss", None),
            logger_fn=lambda msg: _meldung(args, f"  [log] {msg}"),
        )
    except services.ServiceError as exc:
        return _fehler(args, exc.code, exc.message)
    if args.json:
        _json_ausgabe(res)
    else:
        _print_summary(res)
        if res["kg_vorschlaege"]:
            from planproof_cad.render import format_kg

            print("\n  Kostengruppen-Vorschläge:")
            print(format_kg(res["kg_vorschlaege"]))
    return EXIT_OK if res["ok"] else EXIT_FAILED


# ---------------------------------------------------------------------------
# Unterbefehle
# ---------------------------------------------------------------------------


def _cmd_demo(args: argparse.Namespace) -> int:
    from planproof_cad.cad.synth import build_demo_plan

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dxf = build_demo_plan(out_dir / "demo.dxf")
    _meldung(args, f"Demo-DXF erzeugt: {dxf}")
    return _run_pipeline(args, str(dxf), out_dir, {})


def _cmd_convert(args: argparse.Namespace) -> int:
    if not Path(args.input).exists():
        return _fehler(args, "input_not_found", f"Eingabe nicht gefunden: {args.input}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"wall_height_m": args.height}
    if args.unit:
        config["dxf_unit"] = args.unit
    return _run_pipeline(args, args.input, out_dir, config)


def _geschosse(args: argparse.Namespace) -> list[str | None] | None:
    """Ein Geschoss je Eingabe. None bei falscher Anzahl (Fehler ist schon gemeldet)."""
    n = len(args.inputs)
    g = list(args.geschoss or [])
    if not g:
        if n > 1:
            _fehler(args, "geschoss_missing", "Bei mehreren Eingaben je Eingabe ein --geschoss.")
            return None
        return [None]
    if len(g) != n:
        _fehler(
            args,
            "geschoss_count",
            f"{n} Eingabe(n), aber {len(g)} --geschoss. Es braucht genau eines je Eingabe.",
        )
        return None
    if len(set(g)) != len(g):
        _fehler(args, "geschoss_duplicate", "Jedes Geschoss darf nur einmal vorkommen.")
        return None
    return list(g)


def _cmd_takeoff(args: argparse.Namespace) -> int:
    from planproof_cad import services
    from planproof_cad.render import format_mengen

    geschosse = _geschosse(args)
    if geschosse is None:
        return EXIT_USAGE
    out_dir = Path(args.out_dir) if args.out_dir else None
    viele = len(args.inputs) > 1
    ergebnisse: list[dict[str, Any]] = []
    try:
        for quelle, g in zip(args.inputs, geschosse, strict=True):
            m = services.takeoff(
                quelle,
                geschoss=g,
                wall_height_m=args.height,
                work_dir=out_dir,
                config={"dxf_unit": args.unit} if args.unit else None,
                logger_fn=lambda msg: _meldung(args, f"  [log] {msg}"),
            )
            ergebnisse.append(m)
            if out_dir is not None:
                suffix = services.geschoss_dateisuffix(g) if viele else ""
                services.schreibe_ausgaben(m, None, out_dir, suffix)
    except services.ServiceError as exc:
        return _fehler(args, exc.code, exc.message)
    if args.json:
        _json_ausgabe(ergebnisse[0] if not viele else {"geschosse": ergebnisse})
    else:
        for m in ergebnisse:
            print(format_mengen(m))
            print()
        if out_dir is not None:
            print(f"Geschrieben nach {out_dir}: mengen*.json, mengen_de*.csv")
    return EXIT_OK


def _cmd_kg(args: argparse.Namespace) -> int:
    from planproof_cad import services
    from planproof_cad.kg.zuordnung import vorschlaege_zusammenfassen
    from planproof_cad.render import format_kg

    geschosse = _geschosse(args)
    if geschosse is None:
        return EXIT_USAGE
    out_dir = Path(args.out_dir) if args.out_dir else None
    dokumente: list[dict[str, Any]] = []
    try:
        for quelle, g in zip(args.inputs, geschosse, strict=True):
            dokumente.append(
                services.assign_kg(
                    quelle,
                    geschoss=g,
                    wall_height_m=args.height,
                    work_dir=out_dir,
                    config={"dxf_unit": args.unit} if args.unit else None,
                    logger_fn=lambda msg: _meldung(args, f"  [log] {msg}"),
                )
            )
    except services.ServiceError as exc:
        return _fehler(args, exc.code, exc.message)
    ergebnis = dokumente[0] if len(dokumente) == 1 else vorschlaege_zusammenfassen(dokumente)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ziel = out_dir / "kg_vorschlaege.json"
        ziel.write_text(json.dumps(ergebnis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.json:
        _json_ausgabe(ergebnis)
    else:
        print(format_kg(ergebnis))
        if out_dir is not None:
            print(f"\nGeschrieben: {out_dir / 'kg_vorschlaege.json'}")
    return EXIT_OK


def _cmd_layers(args: argparse.Namespace) -> int:
    from planproof_cad import services
    from planproof_cad.render import format_layers

    try:
        res = services.list_layers(args.input)
    except services.ServiceError as exc:
        return _fehler(args, exc.code, exc.message)
    if args.json:
        _json_ausgabe(res)
    else:
        print(format_layers(res))
    return EXIT_OK


def _cmd_serve_rest(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from planproof_cad.rest import create_app
    except ImportError as exc:
        print(
            f"FEHLER: Die REST-Schnittstelle braucht das Extra 'rest' "
            f'(pip install "planproof-cad[rest]"): {exc}',
            file=sys.stderr,
        )
        return EXIT_USAGE
    app = create_app(workdir=args.workdir, max_upload_mb=args.max_upload_mb)
    print(f"REST auf http://{args.host}:{args.port}  (Arbeitsverzeichnis {app.state.workdir})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return EXIT_OK


def _cmd_serve_mcp(args: argparse.Namespace) -> int:
    from planproof_cad.mcp_stdio import serve

    return serve(out_dir=args.out_dir, read_roots=args.read_root or [])


def _register_commands(sub: argparse._SubParsersAction) -> dict[str, CommandFn]:
    """Haengt die Unterbefehle ein."""
    handlers: dict[str, CommandFn] = {}

    def _json_flag(p: argparse.ArgumentParser) -> None:
        p.add_argument("--json", action="store_true", help="JSON auf stdout (Log nach stderr)")

    def _mengen_optionen(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--geschoss",
            action="append",
            metavar="NAME",
            help="Geschoss als freier Text (EG, 1.OG ...). Bei mehreren Eingaben eines je Eingabe",
        )
        p.add_argument(
            "--height",
            type=float,
            default=None,
            help="Wandhoehe in m (ueberschreibt die Hoehe im scene.json; bleibt eine Annahme)",
        )
        p.add_argument(
            "--unit",
            choices=["mm", "cm", "m", "in", "ft"],
            help="Nur fuer DXF-Eingaben: Einheit erzwingen",
        )
        p.add_argument("-o", "--out-dir", help="Ausgabeverzeichnis (Dateien schreiben)")

    p_demo = sub.add_parser("demo", help="Demo-Grundriss erzeugen und durchrechnen")
    p_demo.add_argument("output_dir", help="Ausgabeverzeichnis")
    _json_flag(p_demo)
    handlers["demo"] = _cmd_demo

    p_conv = sub.add_parser("convert", help="DXF (oder DWG mit ODA-Bruecke) in BIM umrechnen")
    p_conv.add_argument("input", help="Pfad zu .dxf (oder .dwg)")
    p_conv.add_argument("output_dir", help="Ausgabeverzeichnis")
    p_conv.add_argument(
        "--unit",
        choices=["mm", "cm", "m", "in", "ft"],
        help="Einheit erzwingen (sonst $INSUNITS/Heuristik)",
    )
    p_conv.add_argument("--height", type=float, default=2.80, help="Wandhoehe in m")
    p_conv.add_argument("--geschoss", default=None, metavar="NAME", help="Geschoss (freier Text)")
    _json_flag(p_conv)
    handlers["convert"] = _cmd_convert

    p_take = sub.add_parser("takeoff", help="Mengen je Bauteilart aus scene.json oder DXF")
    p_take.add_argument("inputs", nargs="+", help="scene.json (oder .dxf, dann mit -o)")
    _mengen_optionen(p_take)
    _json_flag(p_take)
    handlers["takeoff"] = _cmd_takeoff

    p_kg = sub.add_parser("kg", help="Kostengruppen-Vorschlaege (zur Freigabe) aus den Mengen")
    p_kg.add_argument("inputs", nargs="+", help="mengen.json, scene.json oder .dxf")
    _mengen_optionen(p_kg)
    _json_flag(p_kg)
    handlers["kg"] = _cmd_kg

    p_lay = sub.add_parser("layers", help="Layer einer DXF mit Klassifikation und Entity-Zahlen")
    p_lay.add_argument("input", help="Pfad zu .dxf (oder .dwg mit ODA-Bruecke)")
    _json_flag(p_lay)
    handlers["layers"] = _cmd_layers

    p_rest = sub.add_parser("serve-rest", help="REST-Schnittstelle starten (Extra 'rest')")
    p_rest.add_argument("--host", default="127.0.0.1", help="Standard 127.0.0.1")
    p_rest.add_argument("--port", type=int, default=8080)
    p_rest.add_argument(
        "--workdir",
        default=None,
        help="Arbeitsverzeichnis fuer Uploads, Ergebnisse und Pfad-Parameter "
        "(Standard $CQPLAN_WORKDIR oder ./planproof-cad-work)",
    )
    p_rest.add_argument("--max-upload-mb", type=float, default=50.0)
    handlers["serve-rest"] = _cmd_serve_rest

    p_mcp = sub.add_parser("serve-mcp", help="MCP-Server ueber stdio starten")
    p_mcp.add_argument(
        "--out-dir", required=True, help="Einziges Verzeichnis, in das geschrieben wird"
    )
    p_mcp.add_argument(
        "--read-root",
        action="append",
        metavar="DIR",
        help="Zusaetzliches Verzeichnis, aus dem Eingaben gelesen werden duerfen "
        "(Standard: out-dir und aktuelles Verzeichnis)",
    )
    handlers["serve-mcp"] = _cmd_serve_mcp

    return handlers


def _stdio_utf8() -> None:
    """Pipes unter Windows nutzen sonst die ANSI-Codepage; Umlaute sollen stimmen."""
    for strom in (sys.stdout, sys.stderr):
        try:
            if not strom.isatty():
                strom.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    from planproof_cad import __version__

    _stdio_utf8()
    parser = argparse.ArgumentParser(prog="planproof-cad", description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=f"planproof-cad {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)
    handlers = _register_commands(sub)

    args = parser.parse_args(argv)
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
