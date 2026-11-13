"""Oeffentliche Python-Schnittstelle von planproof_cad.

Vier transportneutrale Funktionen; CLI (``planproof-cad``), REST (``planproof_cad.rest``) und MCP
(``planproof_cad.mcp_stdio``) rufen genau diese auf::

    from planproof_cad import api

    res = api.run_dxf("plan.dxf", "out", {"wall_height_m": 2.8}, geschoss="EG")
    mengen = api.takeoff("out/scene.json", geschoss="EG")
    kg = api.assign_kg(mengen)
    layer = api.list_layers("plan.dxf")

Rueckgabewerte sind JSON-faehige Dicts. Behebbare Fehler (Datei fehlt, falsches Format) sind
``ServiceError`` mit stabilem ``code``. Kostengruppen-Eintraege haben immer
``status = "vorschlag"``: ein Mensch gibt frei, das Paket legt nie etwas an.

Das Modul haengt nur die Schnittstelle zusammen. Die Pipeline (ezdxf, ifcopenshell) wird erst
beim Aufruf von ``run_dxf`` und ``list_layers`` importiert.
"""

from __future__ import annotations

from planproof_cad.kg.zuordnung import vorschlaege_zusammenfassen as combine_kg
from planproof_cad.services import (
    ServiceError,
    assign_kg,
    list_layers,
    run_dxf,
    takeoff,
)

__all__ = [
    "ServiceError",
    "assign_kg",
    "combine_kg",
    "list_layers",
    "run_dxf",
    "takeoff",
]
