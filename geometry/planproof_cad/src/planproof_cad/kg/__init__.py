"""Kostengruppen-Vorschlag aus den Mengen einer Zeichnung.

``zuordnung`` ist reine Standardbibliothek und in sich geschlossen (Mengenkern und Regeln,
auf dem scene-Dict). Die Regeltabelle liegt zusaetzlich als Datei ``kg_rules.json`` im Paket;
``regeln_aus_datei`` laedt sie, ein Test haelt Datei und eingebettete Konstante gleich.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from planproof_cad.kg.zuordnung import (
    KG_RULES,
    vorschlaege_aus_mengen,
    vorschlaege_aus_scene,
    vorschlaege_zusammenfassen,
)


def regeln_aus_datei() -> dict[str, Any]:
    """Liest ``kg_rules.json`` aus dem installierten Paket."""
    text = resources.files("planproof_cad.kg").joinpath("kg_rules.json").read_text("utf-8")
    return json.loads(text)


__all__ = [
    "KG_RULES",
    "regeln_aus_datei",
    "vorschlaege_aus_mengen",
    "vorschlaege_aus_scene",
    "vorschlaege_zusammenfassen",
]
