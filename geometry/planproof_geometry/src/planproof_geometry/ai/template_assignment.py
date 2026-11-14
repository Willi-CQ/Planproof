"""Pipeline-Step: ordnet SceneModel-Components Katalog-Templates zu.

Laeuft nach ``BuildSceneModelStep``, liest ``state.scene.components``,
adaptiert sie zu ``DetectedElement``-Records und delegiert an die Rules-Engine
in ``planproof_geometry.catalog.assignment_rules``. Low-Confidence-Zuweisungen
werden in ``state.flagged_assignments`` gesammelt fuer das spaetere UI-Review.

Shared-Templates werden einmalig aus dem Repo-Seed geladen
(``packages/planproof_geometry/catalog/seed``). Org-private Templates werden im
Worker-Context noch nicht aufgeloest — das kommt in einem spaeteren Sprint,
wenn der Worker DB-Zugriff mit Org-Scope erhaelt.

Fehlertoleranz:
- Keine SceneModel vorhanden: skip
- Katalog nicht ladbar: skip mit log
- Component ohne ausreichende Geometrie: Assignment mit template_key=None
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _locate_template_overrides(state) -> Optional[Path]:
    """Locate manual_template_overrides.json.

    Search order (mirrors locate_annotations_file convention):
    1. <input_dir>/plan_bim_data/<stem>/manual_template_overrides.json  (Tauri local)
    2. <input_dir>/<stem>.manual_template_overrides.json                 (ZIP-sibling)
    """
    input_path = getattr(state, "input_path", None)
    if not input_path:
        return None
    inp = Path(input_path)
    conv = inp.parent / "plan_bim_data" / inp.stem / "manual_template_overrides.json"
    if conv.exists():
        return conv
    sibling = inp.parent / f"{inp.stem}.manual_template_overrides.json"
    if sibling.exists():
        return sibling
    return None


class TemplateAssignmentStep:
    """Match SceneModel-Components to catalog templates (Shared-only)."""

    name = "template_assignment"

    def __init__(self, seed_dir: Path | None = None) -> None:
        self._seed_dir = seed_dir
        self._templates_by_key: dict[str, object] = {}
        self._template_keys: set[str] = set()
        self._loaded = False

    def _load_catalog(self) -> bool:
        if self._loaded:
            return True
        from planproof_geometry.catalog.schemas import TemplateSpec  # noqa: F401
        from planproof_geometry.catalog.seed_loader import iter_template_seeds

        candidates: list[Path] = []
        if self._seed_dir:
            candidates.append(self._seed_dir)
        here = Path(__file__).resolve()
        # packages/planproof_geometry/src/planproof_geometry/ai/template_assignment.py
        # -> packages/planproof_geometry/catalog/seed
        for up in range(3, 6):
            try:
                cand = here.parents[up] / "catalog" / "seed"
                candidates.append(cand)
            except IndexError:
                continue
        for cand in candidates:
            if cand.is_dir():
                try:
                    templates = list(iter_template_seeds(cand))
                except Exception as exc:
                    logger.warning("Katalog-Load-Fehler aus %s: %s", cand, exc)
                    continue
                if templates:
                    self._templates_by_key = {t.template_key: t for t in templates}
                    self._template_keys = set(self._templates_by_key.keys())
                    self._loaded = True
                    logger.info(
                        "TemplateAssignmentStep: %d Templates aus %s geladen",
                        len(self._templates_by_key), cand,
                    )
                    return True
        logger.info("TemplateAssignmentStep: kein Seed-Verzeichnis gefunden, skip")
        return False

    def _component_to_detected(self, comp) -> object | None:
        from planproof_geometry.catalog.assignment_rules import DetectedElement

        attrs = getattr(comp, "attrs", {}) or {}
        dims = getattr(comp, "dims", None) or (None, None, None)
        try:
            length, height, thickness = (
                float(dims[0]) if dims[0] is not None else None,
                float(dims[1]) if dims[1] is not None else None,
                float(dims[2]) if dims[2] is not None else None,
            )
        except (TypeError, IndexError):
            length = height = thickness = None

        wall_type = attrs.get("wall_type")
        is_external = None
        if wall_type == "exterior":
            is_external = True
        elif wall_type == "interior":
            is_external = False
        is_load_bearing = attrs.get("is_load_bearing")
        is_ground_slab = attrs.get("is_ground_slab")

        return DetectedElement(
            id=comp.id,
            kind=comp.kind,
            length_m=attrs.get("length_m") or length,
            thickness_m=attrs.get("thickness_m") or thickness,
            height_m=attrs.get("height_m") or height,
            width_m=attrs.get("width_m"),
            area_m2=attrs.get("area_m2"),
            is_external=is_external,
            is_load_bearing=is_load_bearing,
            is_ground_slab=is_ground_slab,
            label=attrs.get("label") or attrs.get("room_type"),
            room_type_hint=attrs.get("room_type"),
        )

    def run(self, ctx, state):
        # Feature-Flag-Check (Config liegt evtl. im State)
        cfg = getattr(state, "config", {}) or {}
        if cfg.get("template_assignment_enabled") is False:
            ctx.log("TemplateAssignmentStep: via config deaktiviert, skip")
            return state

        scene = getattr(state, "scene", None)
        if scene is None or not getattr(scene, "components", None):
            ctx.log("TemplateAssignmentStep: keine Scene-Components vorhanden, skip")
            return state

        if not self._load_catalog():
            return state

        from planproof_geometry.catalog.assignment_rules import (
            apply_forced_overrides,
            assign_element,
            validate_templates_exist,
        )

        assignments_raw = []
        for comp in scene.components.values():
            detected = self._component_to_detected(comp)
            if detected is None:
                continue
            assignments_raw.append(assign_element(detected))

        assignments = validate_templates_exist(assignments_raw, self._template_keys)

        # Apply user overrides from manual_template_overrides.json (if present)
        overrides_path = _locate_template_overrides(state)
        if overrides_path:
            try:
                raw_overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
                forced: dict[str, str] = {
                    eid: v["template_key"]
                    for eid, v in raw_overrides.get("overrides", {}).items()
                    if isinstance(v, dict) and v.get("template_key")
                }
                assignments = apply_forced_overrides(assignments, forced)
                ctx.log(
                    f"TemplateAssignment: {len(forced)} User-Override(s) aus "
                    f"{overrides_path.name} angewendet"
                )
            except Exception as exc:
                logger.warning("manual_template_overrides.json lesen fehlgeschlagen: %s", exc)

        low_conf_threshold = float(cfg.get("assignment_low_confidence_threshold", 0.5))
        flagged = [
            a.element_id for a in assignments
            if a.template_key is None or a.confidence < low_conf_threshold
        ]

        # State schreiben
        state.template_assignments = [
            {
                "element_id": a.element_id,
                "element_kind": a.element_kind,
                "template_key": a.template_key,
                "template_version": a.template_version,
                "confidence": a.confidence,
                "rationale": a.rationale,
            }
            for a in assignments
        ]
        state.flagged_assignments = flagged
        state.metrics = getattr(state, "metrics", {}) or {}
        n_assigned = sum(1 for a in assignments if a.template_key)
        state.metrics["template_assignments"] = {
            "total": len(assignments),
            "assigned": n_assigned,
            "flagged_low_confidence": len(flagged),
        }
        # Templates fuer nachfolgende Steps (DINValidator) unter state.meta
        # cachen — vermeidet Doppel-Laden.
        if not hasattr(state, "_catalog_templates_cache"):
            pass
        state._catalog_templates_cache = self._templates_by_key

        ctx.log(
            f"TemplateAssignment: {n_assigned}/{len(assignments)} zugewiesen, "
            f"{len(flagged)} low-conf flagged"
        )
        return state


class DINValidatorStep:
    """Prueft Template-Assignments gegen GEG/DIN-Regelwerk (nicht-blockierend)."""

    name = "din_validator"

    def run(self, ctx, state):
        cfg = getattr(state, "config", {}) or {}
        if cfg.get("din_validator_enabled") is False:
            ctx.log("DINValidatorStep: via config deaktiviert, skip")
            return state

        assignments_data = getattr(state, "template_assignments", None)
        if not assignments_data:
            ctx.log("DINValidatorStep: keine Assignments vorhanden, skip")
            return state

        templates_cache = getattr(state, "_catalog_templates_cache", None)
        if not templates_cache:
            ctx.log("DINValidatorStep: kein Template-Cache, skip")
            return state

        from planproof_geometry.catalog.assignment_rules import Assignment
        from planproof_geometry.catalog.din_validator import (
            summarize_violations,
            validate_assignments,
        )

        # Rekonstruiere Assignment-Objekte aus state-Dict
        assignments = [
            Assignment(
                element_id=a["element_id"],
                element_kind=a["element_kind"],
                template_key=a["template_key"],
                template_version=a.get("template_version", 1),
                confidence=a.get("confidence", 0.0),
                rationale=a.get("rationale", ""),
            )
            for a in assignments_data
        ]
        violations = validate_assignments(assignments, templates_cache)

        state.din_violations = [
            {
                "element_id": v.element_id,
                "element_kind": v.element_kind,
                "din_reference": v.din_reference,
                "severity": v.severity,
                "message": v.message,
                "actual_value": v.actual_value,
                "required_value": v.required_value,
            }
            for v in violations
        ]
        summary = summarize_violations(violations)
        state.metrics = getattr(state, "metrics", {}) or {}
        state.metrics["din_violations"] = summary

        if summary["error"]:
            ctx.log(
                f"DINValidator: {summary['error']} Fehler, "
                f"{summary['warning']} Warnungen, {summary['info']} Hinweise"
            )
        else:
            ctx.log(
                f"DINValidator: ok — {summary['warning']} Warnungen, "
                f"{summary['info']} Hinweise"
            )
        return state


__all__ = ["TemplateAssignmentStep", "DINValidatorStep"]
