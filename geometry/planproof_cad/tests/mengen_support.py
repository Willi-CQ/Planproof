"""Hilfen fuer die Tests von Mengen, Kostengruppen-Vorschlag und Schnittstellen.

Die Fixtures unter ``tests/fixtures`` sind eingefrorene Kopien: ``demo/scene.json`` und
``demo/meta.json`` stammen aus ``planproof-cad demo``, ``scene_haus_eg.json`` und
``scene_haus_og.json`` sind von Hand gebaut (Aussen- und Innenwaende, tragend und
nichttragend, Oeffnungen mit und ohne Elternwand, zwei Geschosse). Die Tests haengen damit
nicht vom Pipeline-Code ab.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DEMO_SCENE = FIXTURES / "demo" / "scene.json"
DEMO_META = FIXTURES / "demo" / "meta.json"
HAUS_EG = FIXTURES / "scene_haus_eg.json"
HAUS_OG = FIXTURES / "scene_haus_og.json"


def lade(pfad: Path) -> dict[str, Any]:
    return json.loads(pfad.read_text(encoding="utf-8"))


def wand(cid: str, laenge: float, lage: str, hoehe: float = 2.8, **attrs: Any) -> dict[str, Any]:
    """Minimale Wandkomponente fuer synthetische Szenen."""
    a = {
        "length_m": laenge,
        "height_m": hoehe,
        "thickness_m": 0.2,
        "wall_type": lage,
        "source": "dwg_double_line",
    }
    a.update(attrs)
    return {"id": cid, "kind": "WALL", "dims": [laenge, hoehe, 0.2], "attrs": a}


def oeffnung(
    cid: str,
    art: str,
    eltern: str | None,
    breite: float,
    bruestung: float,
    sturz: float,
    **attrs: Any,
) -> dict[str, Any]:
    a: dict[str, Any] = {
        "width_m": breite,
        "height_m": sturz - bruestung,
        "sill_height_m": bruestung,
        "head_height_m": sturz,
        "source": "dwg_block",
    }
    if eltern is not None:
        a["parent_edge_id"] = eltern
    a.update(attrs)
    return {"id": cid, "kind": art, "attrs": a}


def szene(*komponenten: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": "scene_v1",
        "units": "m",
        "components": {k["id"]: k for k in komponenten},
        "relationships": [],
    }
