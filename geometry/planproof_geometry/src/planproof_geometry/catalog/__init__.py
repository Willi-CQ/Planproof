"""BIM-Komponenten-Katalog: Shared-Datenmodell zwischen API und Pipeline.

Die Schemas hier spiegeln die API-Pydantic-Modelle (``planproof_api.schemas.catalog``)
wider, damit die Pipeline ohne API-Imports arbeiten kann. Der SeedLoader
implementiert den Transfer-Mechanismus Repo-JSONs → Cloud-DB.
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
