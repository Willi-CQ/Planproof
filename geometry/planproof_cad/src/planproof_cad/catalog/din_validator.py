"""DIN-Validator: Regelbasierte Pruefungen gegen GEG + DIN 4109 + DIN 4102.

Nimmt die Liste der Assignments + Template-Daten und emittiert Verstoesse.
Nicht-blockierend: die Pipeline laeuft durch, Violations werden gesammelt und
im UI/Report angezeigt.

Regeln:
- **GEG (2023)** Anlage 7 Referenzwerte fuer Hoechstwerte der Waermedurchgangs-
  koeffizienten im Neubau: AW U<=0.28, Dach U<=0.20, Decke gegen unbeheizt
  U<=0.35, Fenster U<=1.30, Aussentuer U<=1.80.
- **DIN 4109-1 (2018)**: Wohnungstrennwand R'w>=53, Wohnungstrenndecke
  R'w>=54, Haustrennwand R'w>=57 (nach DIN 4109-5 erhoehter Schallschutz).
- **DIN 4102-3**: Brandwand muss REI 90-M erreichen (Stosslast M).
- **DIN 1053-1**: Tragendes Mauerwerk Mindestdicke 115mm.

Alle Grenzwerte sind Anforderungen "Neubau" - Bestandsgebaeude haben eigene
Regelwerke und sind aus MVP-Scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .assignment_rules import Assignment
from .schemas import TemplateSpec

Severity = Literal["info", "warning", "error"]


@dataclass
class DINViolation:
    element_id: str
    element_kind: str
    din_reference: str
    severity: Severity
    message: str
    actual_value: float | str | None = None
    required_value: float | str | None = None


# ═══════════════════════════════════════════════════════════════════════════
# GEG Waermeschutz-Grenzwerte (Anlage 7, Neubau Referenz)
# ═══════════════════════════════════════════════════════════════════════════

# Element-Type → max U-Wert [W/(m²·K)]
_GEG_MAX_U: dict[str, float] = {
    "wall_exterior": 0.28,
    "wall_basement": 0.30,
    "roof": 0.20,
    "slab_ground": 0.35,
    "slab_ceiling_unheated": 0.35,  # Decke gegen unbeheizt
    "window": 1.30,
    "door_exterior": 1.80,
    "roof_window": 1.40,
}


def _u_value(template: TemplateSpec) -> float | None:
    v = template.computed_properties.get("u_value")
    return float(v) if v is not None else None


def _rw_value(template: TemplateSpec) -> int | None:
    v = template.computed_properties.get("sound_reduction_rw")
    return int(v) if v is not None else None


def _fire_resistance(template: TemplateSpec) -> str | None:
    v = template.computed_properties.get("fire_resistance")
    return str(v) if v else None


def _category_implies_external_wall(category: str) -> bool:
    return category.startswith("exterior_wall")


def _category_implies_basement_wall(category: str) -> bool:
    return category == "exterior_wall_basement"


def _category_implies_apartment_separation(category: str) -> bool:
    return category in (
        "interior_wall_apartment_separation",
        "interior_wall_house_separation",
    )


def _category_implies_firewall(category: str) -> bool:
    return category == "interior_wall_firewall"


def _category_implies_roof_window(category: str) -> bool:
    return category == "roof_window"


# ═══════════════════════════════════════════════════════════════════════════
# Einzel-Checks
# ═══════════════════════════════════════════════════════════════════════════


def _check_geg_u_value(
    assignment: Assignment,
    template: TemplateSpec,
) -> list[DINViolation]:
    u = _u_value(template)
    if u is None:
        return []
    kind = template.element_type
    cat = template.category
    max_u: float | None = None
    rule_name = ""
    if kind == "wall" and _category_implies_external_wall(cat):
        max_u = (
            _GEG_MAX_U["wall_basement"]
            if _category_implies_basement_wall(cat)
            else _GEG_MAX_U["wall_exterior"]
        )
        rule_name = "Aussenwand" if not _category_implies_basement_wall(cat) else "Kellerwand"
    elif kind == "roof":
        max_u = _GEG_MAX_U["roof"]
        rule_name = "Dach"
    elif kind == "slab" and template.geometry_params.get("is_ground_slab"):
        max_u = _GEG_MAX_U["slab_ground"]
        rule_name = "Bodenplatte"
    elif kind == "window":
        max_u = (
            _GEG_MAX_U["roof_window"]
            if _category_implies_roof_window(cat)
            else _GEG_MAX_U["window"]
        )
        rule_name = "Dachflaechenfenster" if _category_implies_roof_window(cat) else "Fenster"
    elif kind == "door" and cat == "exterior_door":
        max_u = _GEG_MAX_U["door_exterior"]
        rule_name = "Aussentuer"
    if max_u is None:
        return []
    if u > max_u:
        return [
            DINViolation(
                element_id=assignment.element_id,
                element_kind=assignment.element_kind,
                din_reference="GEG 2023 Anlage 7",
                severity="error",
                message=f"{rule_name}: U={u:.2f} > Höchstwert {max_u:.2f} W/(m²·K)",
                actual_value=u,
                required_value=max_u,
            )
        ]
    # Info-Level wenn U sehr nah am Grenzwert (±5%)
    if u > max_u * 0.95:
        return [
            DINViolation(
                element_id=assignment.element_id,
                element_kind=assignment.element_kind,
                din_reference="GEG 2023 Anlage 7",
                severity="info",
                message=f"{rule_name}: U={u:.2f} nah am Grenzwert {max_u:.2f}",
                actual_value=u,
                required_value=max_u,
            )
        ]
    return []


def _check_din_4109_rw(
    assignment: Assignment,
    template: TemplateSpec,
) -> list[DINViolation]:
    kind = template.element_type
    cat = template.category
    rw = _rw_value(template)
    if rw is None:
        return []
    violations: list[DINViolation] = []
    if kind == "wall" and _category_implies_apartment_separation(cat):
        if cat == "interior_wall_house_separation":
            required = 57
            rule = "Haustrennwand DIN 4109-5 erhoeht"
        else:
            required = 53
            rule = "Wohnungstrennwand DIN 4109-1"
        if rw < required:
            violations.append(
                DINViolation(
                    element_id=assignment.element_id,
                    element_kind=assignment.element_kind,
                    din_reference="DIN 4109-1 (2018)",
                    severity="error",
                    message=f"{rule}: R'w={rw} < erforderlich {required} dB",
                    actual_value=rw,
                    required_value=required,
                )
            )
    if kind == "slab" and cat == "ceiling_slab":
        required = 54
        if rw < required:
            violations.append(
                DINViolation(
                    element_id=assignment.element_id,
                    element_kind=assignment.element_kind,
                    din_reference="DIN 4109-1 (2018)",
                    severity="warning",
                    message=f"Wohnungstrenndecke: R'w={rw} < Anforderung {required} dB",
                    actual_value=rw,
                    required_value=required,
                )
            )
    return violations


def _check_din_4102_firewall(
    assignment: Assignment,
    template: TemplateSpec,
) -> list[DINViolation]:
    if not _category_implies_firewall(template.category):
        return []
    fr = _fire_resistance(template)
    if not fr:
        return [
            DINViolation(
                element_id=assignment.element_id,
                element_kind=assignment.element_kind,
                din_reference="DIN 4102-3",
                severity="error",
                message="Brandwand ohne Feuerwiderstands-Angabe",
            )
        ]
    # Erwartet: 'REI 90-M' oder 'REI 120-M'
    if "M" not in fr:
        return [
            DINViolation(
                element_id=assignment.element_id,
                element_kind=assignment.element_kind,
                din_reference="DIN 4102-3",
                severity="error",
                message=f"Brandwand muss Klassifizierung 'M' (Stoßlast) tragen, gefunden: {fr}",
                actual_value=fr,
                required_value="REI 90-M+",
            )
        ]
    return []


def _check_din_1053_min_thickness(
    assignment: Assignment,
    template: TemplateSpec,
) -> list[DINViolation]:
    # Nur fuer tragende Mauerwerkswaende
    if template.element_type != "wall":
        return []
    cat = template.category
    if "loadbearing" not in cat and "house_separation" not in cat and "firewall" not in cat:
        return []
    pset = template.ifc_psets.get("Pset_WallCommon", {})
    if not pset.get("LoadBearing"):
        return []
    # Gewahlter Primaer-Strukturlayer
    structural_thk = 0.0
    for layer in template.layers:
        if "structural" in layer.function and layer.thickness_mm:
            structural_thk = max(structural_thk, layer.thickness_mm)
    if structural_thk <= 0:
        return []
    # DIN 1053-1: Tragendes Mauerwerk Mindestdicke 115 mm
    if structural_thk < 115:
        return [
            DINViolation(
                element_id=assignment.element_id,
                element_kind=assignment.element_kind,
                din_reference="DIN 1053-1",
                severity="error",
                message=f"Tragende Wand: Struktur-Schicht {structural_thk:.0f}mm < 115mm",
                actual_value=structural_thk,
                required_value=115,
            )
        ]
    return []


# ═══════════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════════


def validate_assignment(
    assignment: Assignment,
    template: TemplateSpec,
) -> list[DINViolation]:
    """Fuehrt alle anwendbaren Checks fuer ein (Assignment, Template)-Paar aus."""
    if assignment.template_key is None:
        return []
    violations: list[DINViolation] = []
    violations.extend(_check_geg_u_value(assignment, template))
    violations.extend(_check_din_4109_rw(assignment, template))
    violations.extend(_check_din_4102_firewall(assignment, template))
    violations.extend(_check_din_1053_min_thickness(assignment, template))
    return violations


def validate_assignments(
    assignments: list[Assignment],
    templates_by_key: dict[str, TemplateSpec],
) -> list[DINViolation]:
    all_violations: list[DINViolation] = []
    for a in assignments:
        if a.template_key is None:
            continue
        tmpl = templates_by_key.get(a.template_key)
        if not tmpl:
            continue
        all_violations.extend(validate_assignment(a, tmpl))
    return all_violations


def summarize_violations(violations: list[DINViolation]) -> dict[str, int]:
    """Kurz-Statistik fuer das UI (Zaehler je Severity)."""
    summary = {"info": 0, "warning": 0, "error": 0}
    for v in violations:
        summary[v.severity] += 1
    return summary


__all__ = [
    "DINViolation",
    "Severity",
    "validate_assignment",
    "validate_assignments",
    "summarize_violations",
]
