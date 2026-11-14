"""End-to-End: DWG/DXF -> komplettes BIM-Modell (IFC4 + glTF + Artefakte).

Orchestriert die DWG-Frontend-Stufen und haengt sie an die bestehende,
produktionsreife BIM-Ausgabe:

    load_and_extract  -> RawSegments/Inserts/Texts (Meter)
    classify          -> Wand-Linien / Tuer-Fenster-Bloecke / Annotation
    reconstruct_walls -> Centerlines (Achse + Dicke)
    build_wall_graph  -> WallGraph (Junctions, T-Splits)
    extract_openings  -> ParametricOpenings am Graph
    detect_rooms      -> Raumpolygone + Labels
    build_scene_from_wall_graph -> SceneModel            [REUSE]
    TemplateAssignmentStep      -> DIN-Katalog-Mapping   [REUSE]
    export_scene_to_gltf        -> model.glb             [REUSE]
    IfcExporter.export          -> plan.ifc (parametrisch)[REUSE]

Output-Artefakte in ``output_dir``:
    plan.ifc, model.glb, scene.json, quantity_takeoff.csv, meta.json
"""
from __future__ import annotations

import base64
import csv
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

LoggerFn = Callable[[str], None]


@dataclass
class DwgBimResult:
    ok: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    ifc_path: Optional[str] = None
    model_path: Optional[str] = None
    scene: Any = None


def _noop(_msg: str) -> None:
    pass


# ── glTF-JSON -> GLB-Binary ──────────────────────────────────────────────

_GLB_MAGIC = 0x46546C67
_GLB_JSON = 0x4E4F534A
_GLB_BIN = 0x004E4942


def _gltf_to_glb(gltf_path: Path, glb_path: Path) -> None:
    """Packt eine self-contained glTF (Buffer als data-URI) in eine .glb.

    Der bestehende ``export_scene_to_gltf`` bettet den Binaerpuffer als
    base64-data-URI ein. GLB ist der Codebase-Standard (model.glb), daher
    extrahieren wir den Puffer in einen BIN-Chunk.
    """
    gltf = json.loads(gltf_path.read_text(encoding="utf-8"))
    buffers = gltf.get("buffers") or []
    bin_blob = b""
    if buffers:
        uri = buffers[0].get("uri", "")
        if isinstance(uri, str) and uri.startswith("data:"):
            b64 = uri.split("base64,", 1)[1] if "base64," in uri else ""
            bin_blob = base64.b64decode(b64) if b64 else b""
        buffers[0] = {"byteLength": len(bin_blob)}
        gltf["buffers"] = buffers

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b"\x20"
    bin_padded = bin_blob
    while len(bin_padded) % 4 != 0:
        bin_padded += b"\x00"

    total = 12 + 8 + len(json_bytes) + (8 + len(bin_padded) if bin_padded else 0)
    with open(glb_path, "wb") as f:
        f.write(struct.pack("<III", _GLB_MAGIC, 2, total))
        f.write(struct.pack("<II", len(json_bytes), _GLB_JSON))
        f.write(json_bytes)
        if bin_padded:
            f.write(struct.pack("<II", len(bin_padded), _GLB_BIN))
            f.write(bin_padded)


# ── Quantity-Takeoff ─────────────────────────────────────────────────────

def _write_quantity_takeoff(scene, assignments_by_id: Dict[str, dict], path: Path) -> int:
    rows = []
    for comp in scene.components.values():
        attrs = comp.attrs or {}
        dims = comp.dims or (None, None, None)
        length = attrs.get("length_m") or (dims[0] if dims else None)
        height = attrs.get("height_m") or (dims[1] if dims else None)
        thickness = attrs.get("thickness_m") or (dims[2] if dims else None)
        area = attrs.get("area_m2")
        if area is None and length and height and comp.kind == "WALL":
            area = round(float(length) * float(height), 3)
        volume = None
        if length and height and thickness and comp.kind == "WALL":
            volume = round(float(length) * float(height) * float(thickness), 3)
        asg = assignments_by_id.get(comp.id, {})
        rows.append({
            "id": comp.id,
            "kind": comp.kind,
            "wall_type": attrs.get("wall_type", ""),
            "length_m": round(float(length), 3) if length else "",
            "width_m": round(float(attrs.get("width_m")), 3) if attrs.get("width_m") else "",
            "thickness_m": round(float(thickness), 3) if thickness else "",
            "height_m": round(float(height), 3) if height else "",
            "area_m2": area if area is not None else "",
            "volume_m3": volume if volume is not None else "",
            "template_key": asg.get("template_key", "") or "",
            "label": attrs.get("label", "") or "",
        })
    fieldnames = ["id", "kind", "wall_type", "length_m", "width_m", "thickness_m",
                  "height_m", "area_m2", "volume_m3", "template_key", "label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ── Hauptpipeline ────────────────────────────────────────────────────────

def run_dwg_to_bim_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    work_dir: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
    logger_fn: Optional[LoggerFn] = None,
) -> DwgBimResult:
    """Baut aus einer DWG/DXF ein vollstaendiges BIM-Modell."""
    from planproof_geometry.ai.dwg_bim.dxf_loader import load_and_extract
    from planproof_geometry.ai.dwg_bim.layer_classifier import (
        WALL_CLASSES, classify_block, classify_layer,
    )
    from planproof_geometry.ai.dwg_bim.opening_extraction import extract_openings
    from planproof_geometry.ai.dwg_bim.room_detection import detect_rooms
    from planproof_geometry.ai.dwg_bim.wall_reconstruction import (
        build_wall_graph, reconstruct_walls,
    )
    from planproof_geometry.ai.plan_bim_gltf_export import export_scene_to_gltf
    from planproof_geometry.ai.plan_build_scene import build_scene_from_wall_graph

    log = logger_fn or _noop
    config = dict(config or {})
    t0 = time.time()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wall_height = float(config.get("wall_height_m", 2.80))

    metrics: Dict[str, Any] = {}
    warnings: List[str] = []
    artifacts: List[str] = []

    try:
        # 1) Laden + Extrahieren
        log(f"dwg_to_bim: lade {Path(input_path).name}")
        ex = load_and_extract(input_path, config, work_dir=work_dir)
        warnings.extend(ex.warnings)
        metrics["unit"] = ex.unit_label
        metrics["unit_scale"] = ex.unit_scale
        metrics["bbox_m"] = [ex.bbox[1][0] - ex.bbox[0][0], ex.bbox[1][1] - ex.bbox[0][1]]
        log(f"dwg_to_bim: {len(ex.segments)} Linien, {len(ex.inserts)} Bloecke, "
            f"{len(ex.texts)} Texte; Einheit={ex.unit_label}")

        # 2) Klassifizieren: Wand-Linien heraussuchen
        wall_segments = [
            s for s in ex.segments if classify_layer(s.layer).cls in WALL_CLASSES
        ]
        # Falls KEINE Wand-Layer erkannt (alles auf Layer 0): alle Linien als
        # Wand-Kandidaten nehmen (Tier-2-Fallback).
        if not wall_segments and ex.segments:
            warnings.append("Keine Wand-Layer erkannt -> alle Linien als Wand-Kandidaten.")
            wall_segments = list(ex.segments)
        opening_inserts = [
            i for i in ex.inserts
            if classify_block(i.name, i.layer).cls in ("door", "window")
        ]
        metrics["wall_segments"] = len(wall_segments)
        metrics["opening_inserts"] = len(opening_inserts)

        # 3) Wand-Rekonstruktion -> Achsen
        centerlines, wall_stats = reconstruct_walls(wall_segments, config)
        metrics["wall_reconstruction"] = wall_stats
        log(f"dwg_to_bim: Waende={wall_stats['centerlines']} "
            f"(Modus={wall_stats['mode']}, {wall_stats['total_wall_length_m']} m)")

        # 4) WallGraph (Junctions + Splits)
        graph = build_wall_graph(centerlines, config, wall_height_m=wall_height)

        # 5) Oeffnungen
        opening_stats = extract_openings(graph, opening_inserts, config)
        metrics["openings"] = opening_stats

        # 6) Raeume (aus dem gesnappten Graph -> robustes Noding)
        rooms = detect_rooms(graph, ex.texts, config)
        metrics["rooms"] = len(rooms)
        log(f"dwg_to_bim: Graph {len(graph.nodes)} Knoten / {len(graph.edges)} Kanten, "
            f"{opening_stats['openings_attached']} Oeffnungen, {len(rooms)} Raeume")

        # 7) SceneModel (REUSE)
        scene = build_scene_from_wall_graph(graph, rooms_list=rooms, wall_height_m=wall_height)
        metrics["scene_components"] = len(scene.components)

        # 8) DIN-Katalog-Zuweisung (REUSE der Pipeline-Steps via Shim)
        state = SimpleNamespace(
            config=config,
            scene=scene,
            input_path=str(input_path),
            output_dir=str(out_dir),
            metrics=metrics,
        )
        ctx = SimpleNamespace(log=log)
        assignments_by_id: Dict[str, dict] = {}
        try:
            from planproof_geometry.ai.template_assignment import (
                DINValidatorStep, TemplateAssignmentStep,
            )
            TemplateAssignmentStep().run(ctx, state)
            DINValidatorStep().run(ctx, state)
            assignments_by_id = {
                a["element_id"]: a for a in getattr(state, "template_assignments", []) or []
            }
            if getattr(state, "din_violations", None):
                metrics["din_violations"] = state.metrics.get("din_violations")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Template-Zuweisung uebersprungen: {exc}")
            logger.warning("Template-Assignment fehlgeschlagen: %s", exc)

        # 9) glTF -> GLB (REUSE; CAD-Y ist bereits oben -> kein PDF-Flip)
        gltf_tmp = out_dir / "model.gltf"
        glb_path = out_dir / "model.glb"
        try:
            export_scene_to_gltf(
                scene, str(gltf_tmp),
                wall_height_m=wall_height,
                flip_plan_y_for_pdf=False,
                add_slab=True,
            )
            _gltf_to_glb(gltf_tmp, glb_path)
            # Zwischen-glTF entfernen -> nur model.glb im Bundle (kein Duplikat)
            try:
                gltf_tmp.unlink()
            except OSError:
                pass
            artifacts.append(str(glb_path))
            metrics["model_path"] = str(glb_path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"glTF/GLB-Export fehlgeschlagen: {exc}")
            logger.exception("GLB-Export fehlgeschlagen")

        # 10) IFC4 (parametrisch, Wand-Achsen-rotiert)
        ifc_path = out_dir / "plan.ifc"
        try:
            from planproof_geometry.catalog.ifc_exporter import IfcExporter

            exporter = IfcExporter(
                schema=config.get("ifc_schema_version", "IFC4"),
                project_name=config.get("ifc_project_name") or Path(input_path).stem,
                rotate_walls_to_axis=True,
            )
            ifc_result = exporter.export(
                scene=scene,
                template_assignments=getattr(state, "template_assignments", None),
                templates_by_key=getattr(state, "_catalog_templates_cache", {}) or {},
                output_path=ifc_path,
            )
            artifacts.append(str(ifc_path))
            metrics["ifc"] = {
                "schema": ifc_result.schema,
                "element_count": ifc_result.element_count,
                "warnings": len(ifc_result.warnings),
            }
            log(f"dwg_to_bim: IFC {ifc_result.element_count} Elemente -> {ifc_path.name}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"IFC-Export fehlgeschlagen: {exc}")
            logger.exception("IFC-Export fehlgeschlagen")
            ifc_result = None

        # 11) Artefakte: scene.json, quantity_takeoff.csv, meta.json
        try:
            scene_path = out_dir / "scene.json"
            scene.to_json(str(scene_path))
            artifacts.append(str(scene_path))

            qto_path = out_dir / "quantity_takeoff.csv"
            n_rows = _write_quantity_takeoff(scene, assignments_by_id, qto_path)
            artifacts.append(str(qto_path))
            metrics["quantity_rows"] = n_rows
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Artefakt-Schreiben fehlgeschlagen: {exc}")

        metrics["duration_s"] = round(time.time() - t0, 3)
        metrics["warnings_count"] = len(warnings)

        meta_path = out_dir / "meta.json"
        meta = {
            "pipeline": "dwg_to_bim",
            "input": Path(input_path).name,
            "metrics": metrics,
            "warnings": warnings,
            "graph_stats": graph.stats(),
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts.append(str(meta_path))

        glb_ok = glb_path.exists()
        ifc_ok = ifc_path.exists()
        if not glb_ok and not ifc_ok:
            # Beide Exporte (GLB + IFC) sind nur als Warnung geschluckt worden -> es
            # gibt KEIN verwertbares BIM-Modell. NICHT als Erfolg melden, sonst
            # landet ein leeres Bundle als gruener Lauf in der Projekt-Historie.
            metrics["duration_s"] = round(time.time() - t0, 3)
            return DwgBimResult(
                ok=False,
                metrics=metrics,
                artifacts=artifacts,
                warnings=warnings,
                error="Weder GLB- noch IFC-Export erfolgreich — kein BIM-Modell erzeugt.",
            )

        return DwgBimResult(
            ok=True,
            metrics=metrics,
            artifacts=artifacts,
            warnings=warnings,
            ifc_path=str(ifc_path) if ifc_ok else None,
            model_path=str(glb_path) if glb_ok else None,
            scene=scene,
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception("dwg_to_bim-Pipeline fehlgeschlagen")
        metrics["duration_s"] = round(time.time() - t0, 3)
        return DwgBimResult(
            ok=False,
            metrics=metrics,
            artifacts=artifacts,
            warnings=warnings,
            error=str(exc),
        )


__all__ = ["run_dwg_to_bim_pipeline", "DwgBimResult"]
