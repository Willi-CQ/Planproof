"""Pydantic-Dataclasses für den Katalog - nutzbar in Pipeline und Seed-Loader.

Die Schemas sind 1:1-Spiegelbilder der API-Schemas in
``planproof_api.schemas.catalog``, aber ohne FastAPI-Abhängigkeit. So kann die
Pipeline im Worker-Context Templates/Materials deserialisieren, ohne die API
importieren zu müssen.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

MaterialCategory = Literal[
    "masonry",
    "concrete",
    "insulation",
    "cladding",
    "finish",
    "wood",
    "metal",
    "glass",
    "other",
]

ElementType = Literal[
    "wall",
    "door",
    "window",
    "slab",
    "roof",
    "stair",
    "space",
]

CatalogScope = Literal["all", "shared", "org"]


class LayerSpec(BaseModel):
    material_key: str
    thickness_mm: float | None = None
    function: str


class MaterialSpec(BaseModel):
    """Katalog-Eintrag für ein einzelnes Baustoff-Schicht-Material.

    Wird sowohl beim Seeding (Repo → Cloud) als auch bei der
    Template-Auflösung in der Pipeline verwendet.
    """

    material_key: str
    name: str
    category: MaterialCategory
    thickness_mm: float | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    source: str = "Planproof"


class TemplateSpec(BaseModel):
    """Katalog-Eintrag für einen kompletten Bauteil-Aufbau."""

    template_key: str
    name: str
    element_type: ElementType
    ifc_entity_type: str
    category: str
    layers: list[LayerSpec] = Field(default_factory=list)
    geometry_params: dict[str, Any] = Field(default_factory=dict)
    computed_properties: dict[str, Any] = Field(default_factory=dict)
    ifc_psets: dict[str, Any] = Field(default_factory=dict)
    din_references: list[str] = Field(default_factory=list)
    preview_svg: str | None = None
    source: str = "Planproof"
    reviewed_by: str | None = None
