"""SVG-Preview-Generator für Bauteil-Aufbauten.

Rendert eine horizontale Schichten-Grafik (links nach rechts = innen nach außen
für Wände, oben nach unten für Decken/Dächer). Material-Kategorie bestimmt die
Farbe; Schichtbreite ist proportional zur Dicke.

Zweck: schnelle Visualisierung im SIMS-Switcher-UI ohne Asset-Pipeline.
Output ist eine SVG-Zeichenfolge die direkt in TemplateOut.preview_svg
gespeichert wird.
"""
from __future__ import annotations

from .schemas import LayerSpec, MaterialSpec

# Farb-Mapping nach Material-Kategorie (semantisch grobklotzig, schnell lesbar)
_CATEGORY_COLORS: dict[str, str] = {
    "masonry": "#c8b89a",       # KS/Ziegel beige
    "concrete": "#8a8a8a",       # Beton grau
    "insulation": "#f5d76e",     # Dämmung gelb
    "cladding": "#a07050",       # Verkleidung braun
    "finish": "#e0e0e0",         # Putz hellgrau
    "wood": "#9e6f3f",           # Holz braun
    "metal": "#7a8a99",          # Metall blaugrau
    "glass": "#a8d0e6",          # Glas hellblau
    "other": "#cccccc",          # Sonstiges grau
}


def render_layers_svg(
    layers: list[LayerSpec],
    materials_by_key: dict[str, MaterialSpec],
    width_px: int = 320,
    height_px: int = 96,
) -> str:
    """Generiert eine SVG-Schichten-Vorschau (innen → außen).

    Schichtbreite proportional zur Dicke. Bei Schichten ohne Dicke (z.B.
    Anstriche): minimal 4 px.
    """
    if not layers:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px}" />'

    # Dicken sammeln (Fallback Material-Default oder 1 mm)
    thicknesses: list[float] = []
    for layer in layers:
        m = materials_by_key.get(layer.material_key)
        d = layer.thickness_mm if layer.thickness_mm is not None else (m.thickness_mm if m else None) or 1.0
        thicknesses.append(max(d, 1.0))
    total = sum(thicknesses)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px}" '
        f'viewBox="0 0 {width_px} {height_px}">',
        f'<rect x="0" y="0" width="{width_px}" height="{height_px}" fill="#ffffff" stroke="#444" stroke-width="1"/>',
    ]

    x = 0.0
    for layer, d in zip(layers, thicknesses, strict=True):
        m = materials_by_key.get(layer.material_key)
        category = m.category if m else "other"
        color = _CATEGORY_COLORS.get(category, _CATEGORY_COLORS["other"])
        w = max(width_px * d / total, 2.0)
        parts.append(
            f'<rect x="{x:.1f}" y="0" width="{w:.1f}" height="{height_px}" '
            f'fill="{color}" stroke="#333" stroke-width="0.5"/>'
        )
        # Beschriftung: Dicke in mm
        if w > 18:
            label = f"{int(round(d))}"
            cx = x + w / 2
            parts.append(
                f'<text x="{cx:.1f}" y="{height_px - 6}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="10" fill="#222">{label}</text>'
            )
        x += w
    # Innen-/Außen-Marker
    parts.append(
        '<text x="4" y="14" font-family="sans-serif" font-size="9" fill="#444">innen</text>'
    )
    parts.append(
        f'<text x="{width_px - 4}" y="14" text-anchor="end" '
        f'font-family="sans-serif" font-size="9" fill="#444">außen</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


__all__ = ["render_layers_svg"]
