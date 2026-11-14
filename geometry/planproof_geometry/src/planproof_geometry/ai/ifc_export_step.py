"""Pipeline-Step: exportiert state.scene als IFC4-Datei.

Laeuft nach DINValidatorStep. Liest ``state.scene``, ``state.template_assignments``
und ``state._catalog_templates_cache``, schreibt eine IFC-Datei neben der glTF
und setzt ``state.ifc_path``.

Soft-Fail:
- ``ifcopenshell`` nicht installiert: skip mit warning
- Keine Scene: skip
- Export-Fehler: log, aber Pipeline laeuft weiter (IFC ist nicht pipeline-kritisch)
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class IfcExportStep:
    """Schreibt ``<output_dir>/plan.ifc`` parallel zur glTF-Ausgabe."""

    name = "ifc_export"

    def run(self, ctx, state):
        cfg = getattr(state, "config", {}) or {}
        if cfg.get("ifc_export_enabled") is False:
            ctx.log("IfcExportStep: via config deaktiviert, skip")
            return state

        scene = getattr(state, "scene", None)
        if scene is None or not getattr(scene, "components", None):
            ctx.log("IfcExportStep: keine Scene, skip")
            return state

        try:
            from planproof_geometry.catalog.ifc_exporter import IfcExporter
        except ImportError:
            ctx.log("IfcExportStep: ifcopenshell nicht verfuegbar, skip")
            return state

        output_dir = Path(state.output_dir)
        output_path = output_dir / "plan.ifc"
        schema = cfg.get("ifc_schema_version", "IFC4")
        project_name = cfg.get("ifc_project_name") or Path(state.input_path).stem or "Planproof Plan"

        try:
            exporter = IfcExporter(
                schema=schema,
                project_name=project_name,
            )
        except ImportError:
            ctx.log("IfcExportStep: ifcopenshell Konstruktor-Fehler, skip")
            return state

        templates_by_key = getattr(state, "_catalog_templates_cache", {}) or {}
        assignments = getattr(state, "template_assignments", None) or []

        try:
            result = exporter.export(
                scene=scene,
                template_assignments=assignments,
                templates_by_key=templates_by_key,
                output_path=output_path,
            )
        except Exception as exc:
            ctx.log(f"IfcExportStep: Export fehlgeschlagen — {exc}")
            logger.exception("IFC-Export fehlgeschlagen")
            return state

        state.ifc_path = result.path
        state.metrics = getattr(state, "metrics", {}) or {}
        state.metrics["ifc_export"] = {
            "schema": result.schema,
            "element_count": result.element_count,
            "warnings": len(result.warnings),
        }
        ctx.log(
            f"IfcExport ok: {result.element_count} Elemente -> {output_path.name} "
            f"({result.schema})"
        )
        return state


__all__ = ["IfcExportStep"]
