"""Moebel-Standardkatalog — geschlossenes Vokabular fuer die LLM-Erkennung.

Architektur-Grundsatz (research-belegt, siehe .brain): Vision-LLMs sind stark
im **Benennen**, schwach in **Geometrie**. Darum bekommt das LLM hier keine
Freitext-Antwort, sondern muss aus einer **geschlossenen Liste** von Katalog-IDs
waehlen (``catalog_enum()``). Das ist der groesste Reliability-Hebel: kein
Halluzinieren von Phantasie-Moebeln, und jede ID hat deterministische
Nennmasse + Platzierungsregeln + IFC-Mapping.

Die eigentliche Geometrie (Position, Skalierung, Orientierung, Run-Layout)
entsteht NICHT im LLM, sondern deterministisch in ``placement.py`` aus den
Nennmassen dieses Katalogs — analog zu Tueren/Fenstern.

Masse sind Nenn-Footprints in Metern: ``(w, d, h)``
  - ``w`` = Breite (entlang Wand / Front)
  - ``d`` = Tiefe (Front-zu-Rueck)
  - ``h`` = Hoehe (Boden -> Oberkante, fuer die 3D-Extrusion)

Geometrie ist Phase-1 prozedural (skalierte Box). ``glb_path`` ist der Haken
fuer spaetere echte Modelle: ist er gesetzt und die Datei vorhanden, laedt der
Mesh-Builder das GLB und skaliert es auf den Footprint; sonst eine Box.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# IFC-Klassen (BIM-Export-Mapping). Sanitaerobjekte werden in IFC als
# IfcSanitaryTerminal gefuehrt, bewegliche Moebel als IfcFurniture, generische
# Platzhalter als IfcFurnishingElement (Oberklasse). Die einfachste gueltige
# IFC-Geometrie ist eine Bounding-Box — darum ist ein Platzhalter-Quader bereits
# IFC-konform.
IFC_SANITARY = "IfcSanitaryTerminal"
IFC_FURNITURE = "IfcFurniture"
IFC_FURNISHING = "IfcFurnishingElement"

# Platzierungsklassen — steuern die deterministische Orientierung/Anker-Logik.
PLACEMENT_WALL_BACK = "wall_back"   # Ruecken an die naechste Wand, Front in den Raum
PLACEMENT_FREE = "free"             # frei stehend, Orientierung aus Bbox-Langachse
PLACEMENT_CORNER = "corner"         # in eine Raumecke (Dusche)
PLACEMENT_RUN_FILLER = "run_filler" # Lueckenfueller in einer Moebelzeile (Unterschrank)

# Generische IDs, die das LLM als Fallback waehlen darf.
GENERIC_ID = "furniture_generic"
UNKNOWN_ID = "unknown"


@dataclass(frozen=True)
class CatalogItem:
    """Ein Standard-Moebelstueck im geschlossenen Katalog."""

    item_id: str                          # stabiler Enum-Wert, den das LLM waehlt
    label_de: str                         # menschenlesbares Label
    category: str                         # sanitary|kitchen|bedroom|living|office|storage|generic
    nominal_w_m: float                    # Breite (entlang Front/Wand)
    nominal_d_m: float                    # Tiefe (Front-zu-Rueck)
    nominal_h_m: float                    # Hoehe (Boden -> Oberkante)
    placement: str                        # PLACEMENT_*
    ifc_class: str                        # IFC_*
    color_rgba: Tuple[int, int, int, int] # Diagnose-/Viewer-Farbe
    glb_path: Optional[str] = None        # spaeteres echtes Modell; None -> Box
    movable: bool = True                  # False = Einbau (Kueche), beeinflusst IFC

    @property
    def nominal_footprint_m2(self) -> float:
        return self.nominal_w_m * self.nominal_d_m


# Diagnose-/Viewer-Farben pro Kategorie (RGBA). Bewusst gedaempft, damit sie sich
# von Waenden/Tueren/Fenstern abheben, aber nicht schreien.
_COL_SANITARY = (180, 205, 220, 255)   # hellblau-grau (Keramik)
_COL_KITCHEN = (200, 180, 150, 255)    # holz/creme
_COL_APPLIANCE = (150, 155, 160, 255)  # geraete-grau
_COL_BED = (175, 150, 180, 255)        # gedaempftes lila
_COL_LIVING = (160, 185, 160, 255)     # salbei
_COL_OFFICE = (170, 165, 150, 255)     # sand
_COL_STORAGE = (155, 140, 120, 255)    # braun
_COL_GENERIC = (190, 190, 190, 255)    # neutral grau


# ============================================================================
# Katalog — kuratiertes DACH-Set. Masse nach typischen Norm-/Handelsgroessen.
# ============================================================================

_ITEMS: List[CatalogItem] = [
    # --- Sanitaer (IfcSanitaryTerminal) ---------------------------------
    CatalogItem("wc", "WC", "sanitary", 0.37, 0.60, 0.40,
                PLACEMENT_WALL_BACK, IFC_SANITARY, _COL_SANITARY),
    CatalogItem("wc_wall_hung", "Wand-WC", "sanitary", 0.37, 0.55, 0.40,
                PLACEMENT_WALL_BACK, IFC_SANITARY, _COL_SANITARY),
    CatalogItem("bidet", "Bidet", "sanitary", 0.37, 0.55, 0.40,
                PLACEMENT_WALL_BACK, IFC_SANITARY, _COL_SANITARY),
    CatalogItem("washbasin", "Waschbecken", "sanitary", 0.60, 0.48, 0.85,
                PLACEMENT_WALL_BACK, IFC_SANITARY, _COL_SANITARY),
    CatalogItem("double_washbasin", "Doppelwaschtisch", "sanitary", 1.20, 0.50, 0.85,
                PLACEMENT_WALL_BACK, IFC_SANITARY, _COL_SANITARY),
    CatalogItem("shower", "Dusche", "sanitary", 0.90, 0.90, 2.00,
                PLACEMENT_CORNER, IFC_SANITARY, _COL_SANITARY),
    CatalogItem("bathtub", "Badewanne", "sanitary", 1.70, 0.75, 0.58,
                PLACEMENT_WALL_BACK, IFC_SANITARY, _COL_SANITARY),
    CatalogItem("urinal", "Urinal", "sanitary", 0.36, 0.35, 0.60,
                PLACEMENT_WALL_BACK, IFC_SANITARY, _COL_SANITARY),

    # --- Kueche: Geraete + Module (Einbau, run_filler) ------------------
    CatalogItem("kitchen_sink", "Spuele", "kitchen", 0.60, 0.60, 0.90,
                PLACEMENT_RUN_FILLER, IFC_SANITARY, _COL_APPLIANCE, movable=False),
    CatalogItem("stove", "Herd / Kochfeld", "kitchen", 0.60, 0.60, 0.90,
                PLACEMENT_RUN_FILLER, IFC_FURNITURE, _COL_APPLIANCE, movable=False),
    CatalogItem("oven", "Backofen", "kitchen", 0.60, 0.60, 0.90,
                PLACEMENT_RUN_FILLER, IFC_FURNITURE, _COL_APPLIANCE, movable=False),
    CatalogItem("dishwasher", "Geschirrspueler", "kitchen", 0.60, 0.60, 0.85,
                PLACEMENT_RUN_FILLER, IFC_FURNITURE, _COL_APPLIANCE, movable=False),
    CatalogItem("washing_machine", "Waschmaschine", "kitchen", 0.60, 0.60, 0.85,
                PLACEMENT_RUN_FILLER, IFC_FURNITURE, _COL_APPLIANCE, movable=False),
    CatalogItem("refrigerator", "Kuehlschrank", "kitchen", 0.60, 0.65, 1.80,
                PLACEMENT_RUN_FILLER, IFC_FURNITURE, _COL_APPLIANCE, movable=False),
    CatalogItem("base_cabinet", "Unterschrank", "kitchen", 0.60, 0.60, 0.90,
                PLACEMENT_RUN_FILLER, IFC_FURNITURE, _COL_KITCHEN, movable=False),
    CatalogItem("tall_cabinet", "Hochschrank", "kitchen", 0.60, 0.60, 2.00,
                PLACEMENT_RUN_FILLER, IFC_FURNITURE, _COL_KITCHEN, movable=False),

    # --- Schlafzimmer ---------------------------------------------------
    CatalogItem("bed_single", "Einzelbett (90)", "bedroom", 0.90, 2.00, 0.55,
                PLACEMENT_WALL_BACK, IFC_FURNITURE, _COL_BED),
    CatalogItem("bed_double", "Doppelbett (180)", "bedroom", 1.80, 2.00, 0.55,
                PLACEMENT_WALL_BACK, IFC_FURNITURE, _COL_BED),
    CatalogItem("wardrobe", "Kleiderschrank", "storage", 1.50, 0.60, 2.10,
                PLACEMENT_WALL_BACK, IFC_FURNITURE, _COL_STORAGE),
    CatalogItem("nightstand", "Nachttisch", "bedroom", 0.45, 0.40, 0.50,
                PLACEMENT_WALL_BACK, IFC_FURNITURE, _COL_BED),

    # --- Wohnen ---------------------------------------------------------
    CatalogItem("sofa", "Sofa", "living", 2.00, 0.90, 0.80,
                PLACEMENT_FREE, IFC_FURNITURE, _COL_LIVING),
    CatalogItem("armchair", "Sessel", "living", 0.80, 0.85, 0.80,
                PLACEMENT_FREE, IFC_FURNITURE, _COL_LIVING),
    CatalogItem("coffee_table", "Couchtisch", "living", 1.10, 0.60, 0.45,
                PLACEMENT_FREE, IFC_FURNITURE, _COL_LIVING),
    CatalogItem("dining_table", "Esstisch", "living", 1.60, 0.90, 0.75,
                PLACEMENT_FREE, IFC_FURNITURE, _COL_LIVING),
    CatalogItem("chair", "Stuhl", "living", 0.45, 0.50, 0.90,
                PLACEMENT_FREE, IFC_FURNITURE, _COL_LIVING),
    CatalogItem("tv_unit", "TV-Lowboard", "living", 1.60, 0.45, 0.50,
                PLACEMENT_WALL_BACK, IFC_FURNITURE, _COL_STORAGE),

    # --- Buero ----------------------------------------------------------
    CatalogItem("desk", "Schreibtisch", "office", 1.40, 0.70, 0.75,
                PLACEMENT_WALL_BACK, IFC_FURNITURE, _COL_OFFICE),
    CatalogItem("office_chair", "Buerostuhl", "office", 0.60, 0.60, 1.10,
                PLACEMENT_FREE, IFC_FURNITURE, _COL_OFFICE),
    CatalogItem("shelf", "Regal", "storage", 0.80, 0.35, 2.00,
                PLACEMENT_WALL_BACK, IFC_FURNITURE, _COL_STORAGE),

    # --- Generischer Fallback (Platzhalter, fuellt die Bbox) -----------
    CatalogItem(GENERIC_ID, "Moebel (generisch)", "generic", 0.60, 0.60, 0.80,
                PLACEMENT_FREE, IFC_FURNISHING, _COL_GENERIC),
]

FURNITURE_CATALOG: Dict[str, CatalogItem] = {it.item_id: it for it in _ITEMS}


# ============================================================================
# Public API
# ============================================================================

def get_item(item_id: str) -> Optional[CatalogItem]:
    """Katalog-Eintrag per ID, oder None bei unbekannter/``unknown`` ID."""
    if not item_id:
        return None
    return FURNITURE_CATALOG.get(item_id.strip().lower())


def resolve_item(item_id: Optional[str]) -> CatalogItem:
    """Wie :func:`get_item`, faellt aber bei unbekannter ID auf den
    generischen Platzhalter zurueck (nie None). Damit produziert die
    Pipeline auch bei LLM-Halluzination oder fehlendem LLM gueltige
    Geometrie."""
    item = get_item(item_id) if item_id else None
    return item or FURNITURE_CATALOG[GENERIC_ID]


def catalog_enum() -> List[str]:
    """Alle gueltigen Katalog-IDs + ``unknown`` — das geschlossene Vokabular,
    aus dem das LLM waehlen MUSS (JSON-Schema ``enum``)."""
    return list(FURNITURE_CATALOG.keys()) + [UNKNOWN_ID]


def is_appliance_run_item(item_id: str) -> bool:
    """True, wenn das Item typischerweise Teil einer linearen Moebelzeile ist
    (Kueche/Bad), also im Run-Layout breitengewichtet angeordnet wird."""
    item = get_item(item_id)
    return bool(item and item.placement == PLACEMENT_RUN_FILLER)


def catalog_prompt_table() -> str:
    """Kompakte Text-Tabelle des Katalogs fuer den LLM-Prompt.

    Bewusst knapp (ID | Label | Kategorie | Nennmass), damit der Token-Aufwand
    klein bleibt und das LLM die exakten IDs sieht, die es zurueckgeben darf.
    """
    by_cat: Dict[str, List[CatalogItem]] = {}
    for it in _ITEMS:
        by_cat.setdefault(it.category, []).append(it)

    lines: List[str] = []
    for cat in ("sanitary", "kitchen", "bedroom", "living", "office",
                "storage", "generic"):
        items = by_cat.get(cat)
        if not items:
            continue
        lines.append(f"# {cat}")
        for it in items:
            lines.append(
                f"  {it.item_id} | {it.label_de} | "
                f"{it.nominal_w_m:.2f}x{it.nominal_d_m:.2f}x{it.nominal_h_m:.2f}m"
            )
    lines.append(f"  {UNKNOWN_ID} | (nicht erkennbar) | -")
    return "\n".join(lines)
