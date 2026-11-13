"""U-Wert-Berechnung aus Schichtenaufbau (DIN EN ISO 6946).

Formel:
    1/U = Rsi + Σ(d_i / λ_i) + Rse

Wärmeübergangswiderstände nach DIN EN ISO 6946:
    - Wände (horizontaler Wärmestrom):     Rsi = 0.13, Rse = 0.04
    - Decken/Dächer (Wärmestrom aufwärts): Rsi = 0.10, Rse = 0.04
    - Böden gegen Erde (Wärmestrom abwärts): Rsi = 0.17, Rse = 0.00

Für die Berechnung wird ``thickness_mm`` aus der Layer-Definition (oder
fallback aus dem Material) verwendet. ``material.properties.thermal_conductivity``
liefert λ in W/(m·K).

Diese Funktion ist deterministisch und liefert physikalisch korrekte Werte —
die Schwankungsbreite kommt nur durch die Genauigkeit der Eingangswerte
(λ-Bemessungswerte aus DIN 4108-4 Tabelle).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .schemas import LayerSpec, MaterialSpec, TemplateSpec

HeatFlowDirection = Literal["horizontal", "upward", "downward"]

# DIN EN ISO 6946 Wärmeübergangswiderstände [m²·K/W]
_RSI_RSE: dict[HeatFlowDirection, tuple[float, float]] = {
    "horizontal": (0.13, 0.04),  # Wände
    "upward": (0.10, 0.04),       # Decken, Dächer
    "downward": (0.17, 0.00),     # Bodenplatte, Decke nach unten
}


@dataclass
class UValueResult:
    u_value: float                 # W/(m²·K)
    r_total: float                 # m²·K/W
    total_thickness_mm: float
    layer_resistances: list[float]  # je Schicht
    direction: HeatFlowDirection


def _direction_for_element(element_type: str) -> HeatFlowDirection:
    if element_type in ("wall",):
        return "horizontal"
    if element_type in ("roof", "slab_ceiling"):
        return "upward"
    if element_type in ("slab", "slab_ground"):
        # Geschossdecke neutral, Bodenplatte abwärts — Default abwärts für Bodenplatte/EG
        return "downward"
    return "horizontal"


def compute_u_value(
    layers: list[LayerSpec],
    materials_by_key: dict[str, MaterialSpec],
    direction: HeatFlowDirection = "horizontal",
) -> UValueResult:
    """Rechnerischer U-Wert aus Schichten + Material-Properties.

    ``materials_by_key`` muss alle in den Layern referenzierten ``material_key``
    enthalten, sonst raised KeyError. Schichten ohne ``thermal_conductivity``
    werden als Wärmebrücke (R=0) gewertet — nur sinnvoll für sehr dünne
    Anstriche/Folien.
    """
    rsi, rse = _RSI_RSE[direction]
    layer_r: list[float] = []
    total_thickness = 0.0
    for layer in layers:
        material = materials_by_key.get(layer.material_key)
        if material is None:
            raise KeyError(f"Material {layer.material_key!r} nicht im Katalog")
        d_mm = layer.thickness_mm
        if d_mm is None:
            d_mm = material.thickness_mm
        if d_mm is None:
            # Schicht ohne Dicke: Wärmebrücken-Annahme R=0
            layer_r.append(0.0)
            continue
        d_m = d_mm / 1000.0
        total_thickness += d_mm
        lam = material.properties.get("thermal_conductivity")
        if lam is None or lam <= 0:
            layer_r.append(0.0)
            continue
        layer_r.append(d_m / lam)
    r_total = rsi + sum(layer_r) + rse
    u = 1.0 / r_total if r_total > 0 else 0.0
    return UValueResult(
        u_value=round(u, 3),
        r_total=round(r_total, 3),
        total_thickness_mm=round(total_thickness, 1),
        layer_resistances=[round(r, 3) for r in layer_r],
        direction=direction,
    )


def compute_for_template(
    template: TemplateSpec,
    materials_by_key: dict[str, MaterialSpec],
) -> UValueResult:
    """Convenience wrapper für TemplateSpec."""
    direction = _direction_for_element(template.element_type)
    # Special case: Bodenplatte vs Geschossdecke — geometry_params.is_ground_slab
    if template.element_type == "slab" and template.geometry_params.get("is_ground_slab"):
        direction = "downward"
    elif template.element_type == "slab" and not template.geometry_params.get("is_ground_slab"):
        direction = "upward"
    return compute_u_value(template.layers, materials_by_key, direction=direction)


__all__ = [
    "HeatFlowDirection",
    "UValueResult",
    "compute_u_value",
    "compute_for_template",
]
