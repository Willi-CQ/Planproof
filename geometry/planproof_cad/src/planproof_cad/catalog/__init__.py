"""BIM-Komponenten-Katalog: Schemas, Regeln, DIN-Pruefung und die mitgelieferten Seeds.

Die Seeds liegen als Package-Data unter ``planproof_cad/catalog/data`` und werden
ueber ``importlib.resources`` geladen (siehe ``seed_loader.default_seed_root``).
"""

from __future__ import annotations

from .schemas import (
    CatalogScope,
    ElementType,
    LayerSpec,
    MaterialCategory,
    MaterialSpec,
    TemplateSpec,
)

__all__ = [
    "CatalogScope",
    "ElementType",
    "LayerSpec",
    "MaterialCategory",
    "MaterialSpec",
    "TemplateSpec",
]
