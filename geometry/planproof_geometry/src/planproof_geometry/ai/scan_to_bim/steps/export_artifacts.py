"""Step 14: Export all artifacts to output directory."""

from __future__ import annotations

import json
from pathlib import Path


class ExportArtifactsStep:
    name = "Export artifacts"
    soft_fail = False

    def run(self, ctx, state):
        out_dir = Path(state.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        scene = state.scene

        if not scene:
            raise RuntimeError("No scene to export")

        # 1. Export glTF model — der VOLLE Scene-Graph (Raeume als Boden-Fill aus
        # echtem Footprint-Polygon, Waende/Tueren/Fenster als benannte Knoten +
        # glTF-``extras`` inkl. class/room_key/polygon), NICHT mehr nur Waende +
        # Platte des ersten Shells. Reuse des Plan-BIM-Exporters -> der Scan wird
        # ein echtes BIM (Viewer + raumweiser Ist/Soll-Vergleich).
        # ``flip_plan_y_for_pdf=False``: Scan-Koordinaten sind bereits metrisch
        # (kein PDF-Y-Flip wie beim Plan-Pfad).
        gltf_path = out_dir / "model.gltf"
        try:
            from planproof_geometry.ai.plan_bim_gltf_export import export_scene_to_gltf
            room_height = state.metrics.get("room_height_m", 2.80)
            wall_thickness = state.config.get("wall_thickness_m", 0.12)
            export_scene_to_gltf(
                scene,
                out_path=str(gltf_path),
                wall_height_m=room_height,
                default_wall_thickness_m=wall_thickness,
                flip_plan_y_for_pdf=False,
                add_slab=True,
            )
            state.model_path = str(gltf_path)
            ctx.log(f"Exported glTF: {gltf_path.name}")
        except Exception as e:
            ctx.log(f"glTF export failed: {e}")
            state.quality_warnings.append(f"gltf_export_failed: {e}")

        # 2. Export scene model JSON
        scene_path = out_dir / "scene_model.json"
        scene.to_json(str(scene_path))
        state.scene_json_path = str(scene_path)
        ctx.log(f"Exported scene: {scene_path.name}")

        # 3. Export heuristic labels
        if state.heuristic_labels:
            labels_path = out_dir / "heuristic_labels.json"
            labels_path.write_text(
                json.dumps(state.heuristic_labels, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # 4. Export LLM refinement
        if state.llm_refinement:
            llm_path = out_dir / "llm_refinement.json"
            llm_data = {
                "refinements": state.llm_refinement,
                "model_used": state.llm_model_used,
                "cost_usd": state.llm_cost,
            }
            llm_path.write_text(
                json.dumps(llm_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            ctx.log(f"Exported LLM refinement: {llm_path.name}")

        # 5. Export meta.json
        meta = {
            "version": "1.0",
            "pipeline": "scan_to_bim",
            "source": {
                "file": Path(state.input_path).name,
                "points_original": state.metrics.get("input_points_count", 0),
                "points_processed": state.metrics.get("points_after_preprocess", 0),
            },
            "quality": {
                "score": state.metrics.get("quality_score", 0),
                "grade": state.metrics.get("quality_grade", "D"),
                "dimensions": state.metrics.get("quality_dimensions", {}),
                "warnings": state.quality_warnings,
            },
            "geometry": {
                "planes_detected": state.metrics.get("planes_detected", 0),
                "wall_segments": state.metrics.get("wall_segments_count", 0),
                "rooms": state.metrics.get("rooms_detected", 0),
                "openings": state.metrics.get("openings_detected", 0),
                "floor_area_m2": state.metrics.get("floor_area_m2", 0),
                "room_height_m": state.metrics.get("room_height_m", 0),
            },
            "ml_heuristic": {
                "method": "geometric",
                "elements_classified": state.metrics.get("elements_classified", 0),
                "avg_confidence": state.metrics.get("avg_heuristic_confidence", 0),
            },
            "ml_semantic": {
                "applied": state.metrics.get("semantic_segmentation_applied", False),
                "backend": state.metrics.get("semantic_backend"),
                "avg_confidence": state.metrics.get("semantic_avg_confidence"),
                "class_counts": state.metrics.get("semantic_class_counts", {}),
                "fusion_applied": state.metrics.get("semantic_fusion_applied", False),
                "ransac_masking": state.metrics.get("semantic_ransac_masking_applied", False),
            },
            "llm_refinement": {
                "enabled": state.config.get("llm_refinement_enabled", True),
                "model_used": state.llm_model_used,
                "elements_refined": state.metrics.get("llm_refinement_count", 0),
                "avg_confidence_before": state.metrics.get("llm_avg_confidence_before"),
                "avg_confidence_after": state.metrics.get("llm_avg_confidence_after"),
                "confidence_delta": state.metrics.get("llm_confidence_delta"),
                "cost_usd": state.llm_cost,
            },
            "scene": {
                "components": state.metrics.get("scene_components", 0),
                "relationships": state.metrics.get("scene_relationships", 0),
            },
        }
        meta_path = out_dir / "meta.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        ctx.log(f"Exported meta: {meta_path.name}")

        return state
