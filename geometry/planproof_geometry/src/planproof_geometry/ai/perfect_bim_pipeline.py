"""Lean Plan-to-BIM orchestrator that ships the perfect-BIM output directly.

The V3 production pipeline (``plan_to_bim_v3_pipeline.py``) was built
around its own door / window / bridge mesh-builders, then later bolted
on a Step-8.5 that overwrites its model.gltf with the perfect-BIM
output (``bim_complete_*.glb``). All of V3's mesh-building work is
discarded by Step 8.5 — it is wasted compute on every cloud worker
run.

This module is the cleanup. It runs only the work that ends up in the
bundle:

    1. Resolve the annotation sidecar.
    2. ``perfect_bim.run_door_placement(pdf, ann, solution_dir)`` —
       writes ``bim.glb`` + door diagnostics into ``solution_dir``.
    3. ``perfect_bim.run_full_bim(pdf, ann, solution_dir)`` —
       re-extrudes walls, runs the window pipeline, appends bridges,
       post-processes glass; writes ``bim_complete_<unix_ts>.glb`` and
       ``bim_complete_viewer.html``.
    4. Rename ``bim_complete_*.glb`` to ``model.glb`` and clean up
       intermediate artefacts (``bim.glb``, ``bim_*viewer.html``).
    5. Write the result artefacts via ``perfect_bim_artifacts``:
       ``semantics.json`` (Segmente aus den Annotations),
       ``quantity_takeoff.csv`` (Tuer-kalibrierter Massstab),
       ``materializations.json`` (Catalog-Rules + DIN-Validator) und
       ``meta.json`` (Metrik-Snapshot fuer die Desktop-UI).
    6. Write ``viewer.html`` referencing ``model.glb``.
    7. Bundle ``<basename>_solution/`` into
       ``<output_dir>/<basename>_solution.zip`` — the artefact the
       ScansPage downloads.

The function signature matches ``run_plan_to_bim_v3_pipeline`` so the
cloud worker (``cloud_entry._handle_plan_to_bim``) can swap one for
the other with no ABI change.
"""
from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from planproof_geometry.pipeline.core import PipelineResult, StepResult

from .plan_to_bim_v3_pipeline import (
    V3State,
    _bundle_outputs,
    _write_viewer_html,
)


def _resolve_annotation_path(input_p: Path, config: Dict[str, Any]) -> Optional[Path]:
    """Match the V3 pipeline's sidecar lookup so existing inputs work."""
    candidates = [
        input_p.with_suffix(".annotations.json"),
        input_p.parent / "input_0.annotations.json",
    ]
    extra = config.get("manual_annotations_path")
    if extra:
        candidates.append(Path(extra))
    for c in candidates:
        if c is not None and Path(c).exists():
            return Path(c)
    return None


def _resolve_floor_stack(input_p: Path) -> Optional[Dict[str, Any]]:
    """Read floor_stack.json sidecar (Sprint A — Multi-Storey).

    The ScansPage emits this sidecar next to the inputs when the source
    PDF has more than one page. Schema (frozen v1.0):

        {"version": "1.0",
         "default_wall_height_m": 2.8,
         "floors": [{"page_index": 0, "floor_label": "EG", "z_offset_m": 0},
                    {"page_index": 1, "floor_label": "OG", "z_offset_m": 2.8}]}

    Current pipeline only reads + logs the sidecar; actual GLB stacking
    is a follow-up sprint (door/wall extrusion needs to be parameterised
    by per-page input + per-page z_offset). The contract is frozen here
    so Frontend + Worker can ship out of sync.
    """
    sidecar = input_p.parent / "floor_stack.json"
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("version") != "1.0":
        return None
    floors = data.get("floors")
    if not isinstance(floors, list) or not floors:
        return None
    return data


def _exc_origin(exc: BaseException, *, frames: int = 3) -> str:
    """Fehlertext samt Herkunft: Typ, Meldung und die innersten Frames.

    ``str(exc)`` allein hat am 08.09.2026 einen Nachmittag gekostet:
    "cannot unpack non-iterable numpy.int32 object" nennt weder Datei noch
    Zeile, und der Traceback war im Worker schon verworfen, als der Text die
    Desktop-App erreichte. Deshalb wandern Typ und die innersten drei Frames
    (``Datei:Zeile Funktion``, innerster zuerst) mit in den Fehlertext, der
    ueber ``PipelineResult.error`` bis ins Protokoll der Desktop-App reist.
    """
    parts = [f"{type(exc).__name__}: {exc}"]
    tb = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
    inner = list(reversed(tb[-frames:]))
    if inner:
        where = " <- ".join(
            f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in inner
        )
        parts.append(f"[{where}]")
    return " ".join(parts)


def run_perfect_bim_pipeline(
    *,
    input_path: str,
    output_dir: str,
    work_dir: str,
    config: Dict[str, Any],
    logger_fn: Optional[Callable[[str], None]] = None,
) -> PipelineResult[V3State]:
    """Cloud-worker production entry point.

    Drop-in replacement for :func:`run_plan_to_bim_v3_pipeline`. Returns
    the same ``PipelineResult[V3State]`` so the worker treats both
    pipelines identically.
    """
    log = logger_fn if logger_fn is not None else (lambda msg: None)
    state = V3State()
    steps: list[StepResult] = []

    def _step(name: str, started: float, ok: bool, error: Optional[str] = None) -> None:
        steps.append(StepResult(
            name=name,
            ok=ok,
            started_at=started,
            finished_at=time.time(),
            error=error,
        ))

    input_p = Path(input_path)
    output_p = Path(output_dir)
    work_p = Path(work_dir)
    output_p.mkdir(parents=True, exist_ok=True)
    work_p.mkdir(parents=True, exist_ok=True)
    solution_dir = output_p / f"{input_p.stem}_solution"
    solution_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1 : Resolve annotations sidecar ─────────────────────────
    t0 = time.time()
    log("perfect_bim: resolve_annotations")
    ann_path = _resolve_annotation_path(input_p, config)
    if ann_path is None:
        msg = "perfect_bim pipeline requires an annotations.json sidecar"
        _step("perfect_bim_resolve_annotations", t0, False, error=msg)
        return PipelineResult(ok=False, state=state, steps=steps, error=msg)
    _step("perfect_bim_resolve_annotations", t0, True)

    # ── Step 1b : Floor-Stack sidecar (Sprint A) ─────────────────────
    # Multi-Storey-Plaene werden hier erkannt und im state.metrics
    # festgehalten. Actual stacking (per-page extrude + z_offset apply)
    # ist ein Folgeschritt — diese Pipeline fokussiert weiter auf Page 0.
    floor_stack = _resolve_floor_stack(input_p)
    if floor_stack is not None:
        n_floors = len(floor_stack.get("floors", []))
        labels = ", ".join(f.get("floor_label", "?") for f in floor_stack["floors"])
        log(f"perfect_bim: floor_stack_detected n={n_floors} labels={labels}")
        state.metrics["floor_stack"] = floor_stack
        state.metrics["floor_stack_n"] = n_floors

    render_matrix = float(config.get("v3_render_matrix", 4.0))
    wall_height_m = float(config.get("wall_height_m", 2.80))

    # Per-job PDF-render cache. Both wrapper functions monkey-patch the
    # underlying script's ``_render_pdf_grayscale`` with a memoized version
    # for the duration of their ``main()`` call — the door script calls
    # the renderer once and the merge script calls it twice, so the cache
    # saves two of three identical full-page rasterisations (4× matrix on
    # a typical plan ≈ 30–50 MP greyscale per render).
    # Lives on the stack only — the orchestrator drops the dict at function
    # exit, so cached arrays cannot leak between independent worker jobs.
    render_cache: dict = {}

    # ── Step 2 : Door placement (diagnose script via library wrapper) ─
    t0 = time.time()
    log("perfect_bim: door_placement")
    try:
        from planproof_geometry.ai.v3.perfect_bim import (  # noqa: PLC0415
            run_door_placement,
            run_full_bim,
        )
        run_door_placement(
            input_p, ann_path, solution_dir,
            render_matrix=render_matrix,
            wall_height_m=wall_height_m,
            render_cache=render_cache,
        )
        _step("perfect_bim_door_placement", t0, True)
    except Exception as exc:
        origin = _exc_origin(exc)
        log(f"perfect_bim: door_placement failed: {origin}")
        _step("perfect_bim_door_placement", t0, False, error=origin)
        return PipelineResult(
            ok=False, state=state, steps=steps,
            error=f"door_placement: {origin}",
        )

    # ── Step 3 : Full BIM assembly (merge script via library wrapper) ─
    t0 = time.time()
    log("perfect_bim: full_bim_assembly")
    try:
        full_result = run_full_bim(
            input_p, ann_path, solution_dir,
            render_matrix=render_matrix,
            render_cache=render_cache,
        )
        perfect_glb: Path = full_result["bim_complete_glb_path"]
        _step("perfect_bim_full_bim_assembly", t0, True)
    except Exception as exc:
        origin = _exc_origin(exc)
        log(f"perfect_bim: full_bim_assembly failed: {origin}")
        _step("perfect_bim_full_bim_assembly", t0, False, error=origin)
        return PipelineResult(
            ok=False, state=state, steps=steps,
            error=f"full_bim_assembly: {origin}",
        )

    # ── Step 3b : Add stairs (independent post-step, soft-fail) ──────
    # Stairs are layered on top of the perfect_bim output without
    # mutating the door/window/bridge pipeline. If this fails, the
    # original bim_complete_*.glb is still promoted as model.glb and
    # the user gets a BIM without stairs but with everything else.
    if bool(config.get("enable_stairs", True)):
        t0 = time.time()
        log("perfect_bim: add_stairs")
        try:
            from planproof_geometry.ai.v3.perfect_bim import run_add_stairs  # noqa: PLC0415
            stair_result = run_add_stairs(
                input_p, ann_path, solution_dir,
                render_matrix=render_matrix,
            )
            layered = stair_result.get("bim_with_stairs_glb_path")
            if layered is not None:
                # Promote the layered GLB so model.glb includes stairs.
                perfect_glb = layered
                state.metrics["stairs_added"] = True
                log("perfect_bim: stairs layered onto BIM")
            else:
                state.metrics["stairs_added"] = False
                log("perfect_bim: no stair annotations found, skipping")
            _step("perfect_bim_add_stairs", t0, True)
        except Exception as exc:
            # Soft-fail: BIM bundle remains valid even if stair pass crashes.
            origin = _exc_origin(exc)
            log(f"perfect_bim: add_stairs failed (soft): {origin}")
            state.metrics["stairs_added"] = False
            _step("perfect_bim_add_stairs", t0, False, error=origin)

    # ── Step 4 : Promote to model.glb + clean up intermediates ───────
    t0 = time.time()
    log("perfect_bim: promote_artefacts")
    target_glb = solution_dir / "model.glb"
    try:
        shutil.copy2(perfect_glb, target_glb)
    except Exception as exc:
        origin = _exc_origin(exc)
        log(f"perfect_bim: promote_artefacts failed: {origin}")
        _step("perfect_bim_promote_artefacts", t0, False, error=origin)
        return PipelineResult(
            ok=False, state=state, steps=steps,
            error=f"promote_artefacts: {origin}",
        )
    stale_paths = [
        solution_dir / "bim.glb",
        solution_dir / "bim_viewer.html",
        solution_dir / "bim_complete_viewer.html",
        solution_dir / "bim_with_stairs_viewer.html",
        perfect_glb,
    ]
    # Also remove the non-promoted intermediate (either bim_complete or
    # bim_with_stairs, depending on which one we promoted).
    for intermediate in solution_dir.glob("bim_complete_*.glb"):
        if intermediate != perfect_glb:
            stale_paths.append(intermediate)
    for intermediate in solution_dir.glob("bim_with_stairs_*.glb"):
        if intermediate != perfect_glb:
            stale_paths.append(intermediate)
    for stale in stale_paths:
        try:
            if stale.exists():
                stale.unlink()
        except OSError:
            pass
    state.model_path = str(target_glb)
    state.metrics["pipeline_mode"] = "perfect_bim"
    state.metrics["model_glb_size_kb"] = round(target_glb.stat().st_size / 1024, 1)
    _step("perfect_bim_promote_artefacts", t0, True)

    # ── Step 5 : Ergebnis-Artefakte + viewer ─────────────────────────
    # semantics.json (Segmente aus den Annotations), quantity_takeoff.csv,
    # materializations.json und meta.json — der Bundle-Vertrag, gegen den
    # die Desktop-Ergebnis-Features gebaut sind. Soft-fail im Modul; nur
    # ein Totalausfall des Steps wird als failed-Step protokolliert.
    t0 = time.time()
    log("perfect_bim: write_result_artifacts")
    try:
        from planproof_geometry.ai.perfect_bim_artifacts import write_result_artifacts  # noqa: PLC0415
        artifact_metrics = write_result_artifacts(
            solution_dir,
            ann_path=ann_path,
            model_glb_path=target_glb,
            input_path=input_p,
            wall_height_m=wall_height_m,
            log=log,
        )
        state.metrics.update(artifact_metrics)
        state.metrics["result_artifacts_ok"] = True
        _step("perfect_bim_write_result_artifacts", t0, True)
    except Exception as exc:
        origin = _exc_origin(exc)
        log(f"perfect_bim: write_result_artifacts failed (soft): {origin}")
        state.metrics["result_artifacts_ok"] = False
        state.metrics["result_artifacts_error"] = origin
        _step("perfect_bim_write_result_artifacts", t0, False, error=origin)
    # Ehrlichkeit statt still-gruen: der Bundle-Vertrag der Desktop-Ergebnis-
    # Features (Bauteil-Konsole, Mengenliste, Material-Tab) haengt an
    # semantics.json. Fehlt es nach dem Step, wird der Lauf NICHT hart
    # abgebrochen (GLB/IFC sind nutzbar), aber die Degradierung wandert als
    # fehlgeschlagener Step + Metrik nach oben (cloud_entry macht daraus eine
    # sichtbare Warnung) — statt einen leeren Ergebnis-Tab als gruenen Erfolg
    # auszugeben.
    if not (solution_dir / "semantics.json").exists():
        state.metrics["result_artifacts_ok"] = False
        _step(
            "perfect_bim_result_artifacts_contract",
            time.time(),
            False,
            error="semantics.json fehlt — Bauteil-/Mengen-/Material-Ansichten bleiben leer.",
        )
    _write_viewer_html(solution_dir, "model.glb")

    # ── Step 5b : IFC4 Export (Sprint B) ──────────────────────────────
    # Soft-Fail: das BIM-Bundle ist auch ohne IFC nutzbar. Wenn der Export
    # scheitert (ifcopenshell missing, weird GLB), bleibt die Pipeline grün
    # und der User sieht den Fehler nur im Log.
    if not bool(config.get("disable_ifc_export", False)):
        t0 = time.time()
        log("perfect_bim: ifc_export")
        try:
            from planproof_geometry.ai.ifc_export import export_glb_as_ifc  # noqa: PLC0415
            ifc_summary = export_glb_as_ifc(
                target_glb, solution_dir / "model.ifc",
                project_name=config.get("project_name", "Planproof Plan→BIM"),
            )
            state.metrics["ifc_export"] = ifc_summary
            log(
                f"perfect_bim: ifc_export done — "
                f"{ifc_summary['n_walls']}W {ifc_summary['n_doors']}D "
                f"{ifc_summary['n_windows']}F (+{ifc_summary['n_proxy']} proxy)"
            )
            _step("perfect_bim_ifc_export", t0, True)
        except Exception as exc:
            origin = _exc_origin(exc)
            log(f"perfect_bim: ifc_export failed (soft): {origin}")
            _step("perfect_bim_ifc_export", t0, False, error=origin)

    # ── Step 6 : Bundle to zip ───────────────────────────────────────
    t0 = time.time()
    log("perfect_bim: bundle_outputs")
    try:
        zip_path = _bundle_outputs(input_p, output_p, work_p)
        state.package_path = str(zip_path)
        _step("perfect_bim_bundle_outputs", t0, True)
    except Exception as exc:
        origin = _exc_origin(exc)
        log(f"perfect_bim: bundle_outputs failed: {origin}")
        _step("perfect_bim_bundle_outputs", t0, False, error=origin)
        return PipelineResult(
            ok=False, state=state, steps=steps,
            error=f"bundle_outputs: {origin}",
        )

    return PipelineResult(ok=True, state=state, steps=steps, error=None)
