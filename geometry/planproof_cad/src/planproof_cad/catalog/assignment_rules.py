"""Template-Assignment-Rules: detected Component → Template-Key + Confidence.

Deterministische Regeln ohne LLM. Match-Strategie:
- **Walls**: Dicke-Range × is_external × is_load_bearing → Template-Key
- **Doors**: Breite/Höhe → nearest DIN 18101 + Fire/RS-Anforderungen
- **Windows**: Breite/Höhe → nearest Standard-Fenster
- **Slabs**: Dicke × is_ground_slab → Geschossdecke/Bodenplatte
- **Roofs**: Typ (flat/pitched) × Dicke
- **Stairs**: Laufbreite (falls vorhanden) → MFH/EFH/Industrie
- **Spaces** (Rooms): Fläche + optional Name/Label → Raumtyp

Confidence-Score 0.0-1.0: wie zentral die Geometrie im Template-Match-Range
liegt. Low-Confidence (<0.5) ist ein Review-Flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schemas import TemplateSpec


@dataclass
class Assignment:
    element_id: str
    element_kind: str  # WALL|DOOR|WINDOW|SLAB|ROOF|STAIR|ROOM
    template_key: str | None
    template_version: int = 1
    confidence: float = 0.0
    rationale: str = ""


@dataclass
class DetectedElement:
    """Minimale Schnittstelle zwischen SceneModel und Assignment-Engine.

    Enthält nur die Felder die für die Template-Zuordnung gebraucht werden.
    SceneModel.Component → DetectedElement-Adapter in template_assignment.py.
    """

    id: str
    kind: str  # WALL|DOOR|WINDOW|SLAB|ROOF|STAIR|ROOM
    length_m: float | None = None
    thickness_m: float | None = None
    height_m: float | None = None
    width_m: float | None = None
    area_m2: float | None = None
    is_external: bool | None = None
    is_load_bearing: bool | None = None
    is_ground_slab: bool | None = None
    label: str | None = None  # z.B. "Wohnzimmer" aus OCR
    room_type_hint: str | None = None  # z.B. "living" aus ML-Klassifikation


# ═══════════════════════════════════════════════════════════════════════════
# WAND-ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

# Rules-Tabelle: (is_external, is_load_bearing, (thickness_min, thickness_max)) → template_key
# Thickness in Metern. Reihenfolge: spezifischer zuerst (Industrie), generisch am Ende.
_WALL_RULES: list[tuple[bool | None, bool | None, tuple[float, float], str]] = [
    # Außenwände tragend (EFH/MFH-Standard KS+WDVS)
    (True, True, (0.33, 0.40), "aw_ks24_wdvs140_eps"),  # ~36.5cm Standard
    (True, True, (0.40, 0.46), "aw_ks24_wdvs180_eps"),  # ~44cm KfW40
    (True, True, (0.30, 0.33), "aw_ks175_wdvs160_eps"),  # ~33cm KS 17.5
    (True, True, (0.46, 0.55), "aw_beton24_wdvs160_eps"),  # MFH-Standard
    (True, True, (0.55, 0.70), "aw_beton300_wdvs160_mw"),  # Hochhaus/Büro
    # Außenwände nicht-tragend (Sandwich Industrie)
    (True, False, (0.08, 0.20), "aw_sandwich_pir_120"),
    (True, False, (0.20, 0.40), "aw_sandwich_mw_150"),
    # Kellerwände (via is_external=True + is_load_bearing=True, aber Dicke nur 0.24-0.30 und
    # erdberührt - Detection kann das aber selten sicher erkennen)
    # → fällt in AW-Rules oben, Kellerwand-Nuance kommt via Label/Metadaten
    # Innenwände tragend
    (False, True, (0.22, 0.28), "iw_ks240_tragend"),  # 24cm KS
    (False, True, (0.16, 0.20), "iw_ks175_tragend"),  # 17.5cm KS
    (False, True, (0.10, 0.13), "iw_ks115_tragend"),  # 11.5cm KS tragend
    (False, True, (0.28, 0.35), "iw_beton200_tragend"),  # 20+ Beton (ggf. mit Putz)
    # Innenwände nicht-tragend (Zimmertrennwände + GK-Ständerwände)
    (False, False, (0.09, 0.13), "iw_zt_ks115_rw44"),  # 11.5cm Mauerwerk
    (False, False, (0.13, 0.17), "iw_zt_gk_staender125_rw52"),  # GK-Doppelständer
    (False, False, (0.07, 0.11), "iw_zt_gk_staender100_rw45"),  # Einfacher GK-Ständer
    # Fallback wenn is_external/is_load_bearing unbekannt (None)
    (None, None, (0.16, 0.28), "iw_ks175_tragend"),
    (None, None, (0.30, 0.46), "aw_ks24_wdvs140_eps"),
    (None, None, (0.07, 0.14), "iw_zt_gk_staender100_rw45"),
]


def _range_match(value: float, lo: float, hi: float) -> bool:
    return lo <= value <= hi


def _range_confidence(value: float, lo: float, hi: float) -> float:
    """1.0 wenn value mittig im Range, linear abfallend zu den Rändern."""
    if not _range_match(value, lo, hi):
        return 0.0
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    if half <= 0:
        return 1.0
    deviation = abs(value - mid) / half
    # 1.0 bei mid, 0.5 an den Rändern
    return round(1.0 - 0.5 * deviation, 3)


def assign_wall(element: DetectedElement) -> Assignment:
    t = element.thickness_m
    if t is None or t <= 0:
        return Assignment(
            element_id=element.id,
            element_kind="WALL",
            template_key=None,
            confidence=0.0,
            rationale="Keine Wanddicke ermittelt",
        )
    # Strikte Rules zuerst (is_external/is_load_bearing bekannt)
    for is_ext, is_lb, (lo, hi), key in _WALL_RULES:
        if is_ext is not None and element.is_external != is_ext:
            continue
        if is_lb is not None and element.is_load_bearing != is_lb:
            continue
        if _range_match(t, lo, hi):
            return Assignment(
                element_id=element.id,
                element_kind="WALL",
                template_key=key,
                confidence=_range_confidence(t, lo, hi),
                rationale=f"Dicke {t:.3f}m in [{lo:.2f},{hi:.2f}], ext={is_ext}, lb={is_lb}",
            )
    return Assignment(
        element_id=element.id,
        element_kind="WALL",
        template_key=None,
        confidence=0.0,
        rationale=f"Keine Regel für Dicke {t:.3f}m matched",
    )


# ═══════════════════════════════════════════════════════════════════════════
# TÜR-ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

# DIN 18101 Innentür-Standardmaße (in metern) → Template-Key
# Rangematching auf (width, height) mit Toleranz ±0.05m
_DOOR_STANDARD_SIZES: list[tuple[float, float, str]] = [
    (0.610, 1.985, "t_innen_61_2010"),
    (0.735, 1.985, "t_innen_735_2010"),
    (0.860, 1.985, "t_innen_86_2010"),
    (0.985, 1.985, "t_innen_985_2010"),
    (1.110, 1.985, "t_innen_111_2010"),
    (0.860, 2.125, "t_innen_86_2125"),
    (1.000, 2.100, "t_haus_100_210_alu"),  # Haustür Default
    (1.250, 2.200, "t_haus_125_220_alu_passiv"),  # Haustür Passiv
    (0.985, 2.010, "t_t30_98_201"),  # T30 Standard
    (1.110, 2.115, "t_t30_111_211"),  # T30 barrierefrei
    (1.250, 2.200, "t_t90_125_220"),  # T90 Industrie
]


def assign_door(element: DetectedElement) -> Assignment:
    w = element.width_m or element.length_m
    h = element.height_m
    if w is None or h is None:
        return Assignment(
            element_id=element.id,
            element_kind="DOOR",
            template_key=None,
            confidence=0.0,
            rationale="Fehlende Tür-Dimensionen",
        )
    best_key: str | None = None
    best_dist = float("inf")
    tolerance_w = 0.08
    tolerance_h = 0.12
    for std_w, std_h, key in _DOOR_STANDARD_SIZES:
        if abs(w - std_w) > tolerance_w or abs(h - std_h) > tolerance_h:
            continue
        d = ((w - std_w) ** 2 + (h - std_h) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_key = key
    if best_key is None:
        return Assignment(
            element_id=element.id,
            element_kind="DOOR",
            template_key=None,
            confidence=0.0,
            rationale=f"Kein DIN-Standardmaß für {w * 100:.0f}×{h * 100:.0f}cm",
        )
    # Confidence: 1.0 bei perfect match, linear auf 0.5 abfallend bis tolerance
    max_d = (tolerance_w**2 + tolerance_h**2) ** 0.5
    confidence = round(max(0.5, 1.0 - 0.5 * best_dist / max_d), 3)
    return Assignment(
        element_id=element.id,
        element_kind="DOOR",
        template_key=best_key,
        confidence=confidence,
        rationale=f"Nearest DIN 18101 match ({w:.2f}×{h:.2f}m)",
    )


# ═══════════════════════════════════════════════════════════════════════════
# FENSTER-ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

_WINDOW_STANDARD_SIZES: list[tuple[float, float, str]] = [
    (0.60, 0.60, "f_keller_60_60"),
    (0.90, 1.20, "f_pvc_3fach_90_120"),
    (1.20, 1.40, "f_pvc_3fach_120_140"),
    (1.50, 1.40, "f_pvc_3fach_150_140"),
    (1.20, 1.80, "f_buero_3fach_120_180"),
    (0.78, 1.40, "f_dach_78_140"),
    (1.14, 1.40, "f_dach_114_140_3fach"),
    (1.80, 2.20, "f_fest_iso_3fach_180_220"),
    (1.80, 2.80, "f_alu_pfosten_riegel_180_280"),
    (3.00, 1.20, "f_industrie_lichtband_300_120"),
]


def assign_window(element: DetectedElement) -> Assignment:
    w = element.width_m or element.length_m
    h = element.height_m
    if w is None or h is None:
        return Assignment(
            element_id=element.id,
            element_kind="WINDOW",
            template_key=None,
            confidence=0.0,
            rationale="Fehlende Fenster-Dimensionen",
        )
    best_key: str | None = None
    best_dist = float("inf")
    tolerance = 0.30  # Fenster-Größen sind variabler als Türen
    for std_w, std_h, key in _WINDOW_STANDARD_SIZES:
        dw = abs(w - std_w)
        dh = abs(h - std_h)
        if dw > tolerance or dh > tolerance:
            continue
        d = (dw**2 + dh**2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_key = key
    if best_key is None:
        return Assignment(
            element_id=element.id,
            element_kind="WINDOW",
            template_key="f_pvc_3fach_120_140",  # Fallback Standard
            confidence=0.3,
            rationale="Kein Standardmaß, Fallback auf 120×140",
        )
    max_d = tolerance * (2**0.5)
    confidence = round(max(0.5, 1.0 - 0.5 * best_dist / max_d), 3)
    return Assignment(
        element_id=element.id,
        element_kind="WINDOW",
        template_key=best_key,
        confidence=confidence,
        rationale=f"Nearest Window match ({w:.2f}×{h:.2f}m)",
    )


# ═══════════════════════════════════════════════════════════════════════════
# DECKEN / BODENPLATTE-ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

_SLAB_RULES: list[tuple[bool | None, tuple[float, float], str]] = [
    # is_ground_slab, (thickness_min, thickness_max), template_key
    (True, (0.20, 0.28), "bp_beton250_perimeter120"),
    (True, (0.28, 0.40), "bp_beton300_perimeter160"),
    (False, (0.16, 0.20), "decke_beton180_estrich60_eps"),
    (False, (0.20, 0.23), "decke_beton200_estrich60_mw"),
    (False, (0.23, 0.28), "decke_beton220_estrich60_erhoeht"),
    (False, (0.28, 0.40), "decke_beton250_buero"),
    # Fallback: thickness unbekannt → Annahme Geschossdecke
    (None, (0.16, 0.40), "decke_beton200_estrich60_mw"),
]


def assign_slab(element: DetectedElement) -> Assignment:
    t = element.thickness_m
    if t is None or t <= 0:
        return Assignment(
            element_id=element.id,
            element_kind="SLAB",
            template_key=None,
            confidence=0.0,
            rationale="Keine Decken-Dicke ermittelt",
        )
    for is_ground, (lo, hi), key in _SLAB_RULES:
        if is_ground is not None and element.is_ground_slab != is_ground:
            continue
        if _range_match(t, lo, hi):
            return Assignment(
                element_id=element.id,
                element_kind="SLAB",
                template_key=key,
                confidence=_range_confidence(t, lo, hi),
                rationale=f"Dicke {t:.3f}m in [{lo:.2f},{hi:.2f}], ground={is_ground}",
            )
    return Assignment(
        element_id=element.id,
        element_kind="SLAB",
        template_key=None,
        confidence=0.0,
        rationale=f"Keine Regel für Decken-Dicke {t:.3f}m",
    )


# ═══════════════════════════════════════════════════════════════════════════
# DACH-ASSIGNMENT (rudimentär - Plan-Sicht hat selten direkte Dach-Geometrie)
# ═══════════════════════════════════════════════════════════════════════════


def assign_roof(element: DetectedElement) -> Assignment:
    # Default: Satteldach mit guter Dämmung für EFH, Flachdach für MFH/Büro
    t = element.thickness_m
    label = (element.label or "").lower()
    if "flach" in label or "flat" in label:
        key = "fd_warm_beton_pir120"
        conf = 0.7
    elif "pult" in label:
        key = "pd_sparren240_zwd_blech"
        conf = 0.7
    else:
        # Default: Satteldach EFH
        key = "sd_sparren240_zwd_pfanne"
        conf = 0.5
    return Assignment(
        element_id=element.id,
        element_kind="ROOF",
        template_key=key,
        confidence=conf,
        rationale=f"Dach-Default ({label or 'unspezifiziert'})"
        f"{f', thickness={t:.2f}m' if t else ''}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# TREPPE-ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════


def assign_stair(element: DetectedElement) -> Assignment:
    # Laufbreite entscheidet: EFH (<=1.00) vs MFH notwendig (>=1.00) vs Büro (>=1.25)
    w = element.width_m or element.length_m
    if w is None:
        return Assignment(
            element_id=element.id,
            element_kind="STAIR",
            template_key="s_beton_efh_efh_18_28_100",
            confidence=0.4,
            rationale="Keine Laufbreite ermittelt, Fallback EFH-Default",
        )
    if w < 1.0:
        key = "s_beton_efh_efh_18_28_100"
        conf = 0.7
        reason = "Laufbreite <1.0m - EFH"
    elif w < 1.25:
        key = "s_beton_mfh_17_29_125"
        conf = 0.8
        reason = "Laufbreite 1.0-1.25m - MFH notwendig"
    else:
        key = "s_beton_buero_17_30_150"
        conf = 0.7
        reason = "Laufbreite >=1.25m - Büro/Sonderbau"
    return Assignment(
        element_id=element.id,
        element_kind="STAIR",
        template_key=key,
        confidence=conf,
        rationale=reason,
    )


# ═══════════════════════════════════════════════════════════════════════════
# RAUM-ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

# Wenn Label/OCR da ist, auf Raumtyp-Keyword mappen. Sonst: Flächen-Heuristik.
_ROOM_LABEL_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("wohn", "living"), "r_wohnen"),
    (("essen", "dining"), "r_essen"),
    (("küche", "kueche", "kitchen"), "r_kueche"),
    (("schlaf", "bedroom"), "r_schlafen"),
    (("kind", "child"), "r_kind"),
    (("bad", "dusche", "bathroom"), "r_bad_dusche_wc"),
    (("wc", "gäste-wc", "gaeste-wc"), "r_wc_gaeste"),
    (("flur", "diele", "corridor"), "r_flur"),
    (("treppenhaus", "treppen", "stairwell"), "r_treppenraum"),
    (("abstell", "abstellraum", "storage"), "r_abstellraum"),
    (("keller", "basement"), "r_keller_unbeheizt"),
    (("haustechnik", "technik", "har"), "r_haustechnik"),
    (("garage", "carport"), "r_garage"),
    (("büro", "buero", "office"), "r_buero_einzel"),
    (("besprechung", "meeting"), "r_besprechung"),
    (("empfang", "foyer", "lobby"), "r_empfang"),
    (("server", "it-raum", "datacenter"), "r_serverraum"),
    (("produktion", "werkhalle", "production"), "r_produktion"),
    (("lager", "warehouse"), "r_lager"),
    (("verladung", "loading"), "r_verladung"),
    (("werkstatt", "workshop"), "r_werkstatt"),
]


def assign_room(element: DetectedElement) -> Assignment:
    label = (element.label or "").lower()
    if label:
        for keywords, key in _ROOM_LABEL_KEYWORDS:
            if any(kw in label for kw in keywords):
                return Assignment(
                    element_id=element.id,
                    element_kind="ROOM",
                    template_key=key,
                    confidence=0.9,
                    rationale=f"Label '{label}' matched Keyword",
                )
    # Heuristik über Fläche
    area = element.area_m2
    if area is None:
        return Assignment(
            element_id=element.id,
            element_kind="ROOM",
            template_key="r_wohnen",
            confidence=0.3,
            rationale="Keine Fläche + kein Label, Fallback Wohnen",
        )
    if area < 3:
        key = "r_wc_gaeste"
    elif area < 6:
        key = "r_bad_dusche_wc"
    elif area < 10:
        key = "r_kind"
    elif area < 16:
        key = "r_schlafen"
    elif area < 30:
        key = "r_wohnen"
    elif area < 80:
        key = "r_buero_gross"
    else:
        key = "r_produktion"
    return Assignment(
        element_id=element.id,
        element_kind="ROOM",
        template_key=key,
        confidence=0.5,
        rationale=f"Fläche {area:.1f}m² - Heuristik",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════════

_KIND_DISPATCH = {
    "WALL": assign_wall,
    "DOOR": assign_door,
    "WINDOW": assign_window,
    "SLAB": assign_slab,
    "FLOOR": assign_slab,
    "CEILING": assign_slab,
    "ROOF": assign_roof,
    "STAIR": assign_stair,
    "ROOM": assign_room,
    "SPACE": assign_room,
}


def assign_element(element: DetectedElement) -> Assignment:
    fn = _KIND_DISPATCH.get(element.kind)
    if fn is None:
        return Assignment(
            element_id=element.id,
            element_kind=element.kind,
            template_key=None,
            confidence=0.0,
            rationale=f"Kein Handler für kind={element.kind!r}",
        )
    return fn(element)


def assign_many(elements: Iterable[DetectedElement]) -> list[Assignment]:
    return [assign_element(e) for e in elements]


def validate_templates_exist(
    assignments: list[Assignment],
    available_keys: set[str],
) -> list[Assignment]:
    """Markiert Assignments deren template_key nicht im verfügbaren Katalog ist.

    Setzt confidence=0, template_key=None mit rationale-Note. Wichtig für
    Dev-Setups wo Seed nur teilweise gepflegt ist.
    """
    result = []
    for a in assignments:
        if a.template_key and a.template_key not in available_keys:
            result.append(
                Assignment(
                    element_id=a.element_id,
                    element_kind=a.element_kind,
                    template_key=None,
                    confidence=0.0,
                    rationale=f"Template '{a.template_key}' nicht im Katalog ({a.rationale})",
                )
            )
        else:
            result.append(a)
    return result


def load_available_template_keys(templates: Iterable[TemplateSpec]) -> set[str]:
    return {t.template_key for t in templates}


def apply_forced_overrides(
    assignments: list[Assignment],
    forced_overrides: dict[str, str],
) -> list[Assignment]:
    """Overwrites auto-assigned template_key for elements that have a user override.

    Forced overrides get confidence=1.0 and a rationale that preserves the
    previous auto-assignment for audit purposes.
    """
    if not forced_overrides:
        return assignments
    result = []
    for a in assignments:
        override_key = forced_overrides.get(a.element_id)
        if override_key is not None:
            result.append(
                Assignment(
                    element_id=a.element_id,
                    element_kind=a.element_kind,
                    template_key=override_key,
                    template_version=a.template_version,
                    confidence=1.0,
                    rationale=f"User-Override (auto: {a.template_key})",
                )
            )
        else:
            result.append(a)
    return result


__all__ = [
    "Assignment",
    "DetectedElement",
    "assign_element",
    "assign_many",
    "assign_wall",
    "assign_door",
    "assign_window",
    "assign_slab",
    "assign_roof",
    "assign_stair",
    "assign_room",
    "validate_templates_exist",
    "load_available_template_keys",
    "apply_forced_overrides",
]
