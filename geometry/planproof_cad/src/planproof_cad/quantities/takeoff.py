"""Mengen aus scene.json: Laden, Berechnen, Kontext der Pipeline.

Die Rechenregeln stehen im Mengenkern ``planproof_cad.kg.zuordnung.mengen_aus_scene``
(reine Standardbibliothek, damit dieselbe Datei in einem Projekt ohne dieses Paket laeuft).
Dieses Modul haengt das Paket-Wissen an: Versionsangabe, scene.json von der Platte lesen,
die ``meta.json`` der Pipeline daneben finden.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from planproof_cad import __version__
from planproof_cad.kg.zuordnung import mengen_aus_scene


class SceneLadeFehler(ValueError):
    """scene.json fehlt oder ist kein scene-Dict."""


def lade_scene(pfad: str | Path) -> dict[str, Any]:
    p = Path(pfad)
    if not p.is_file():
        raise SceneLadeFehler(f"scene.json nicht gefunden: {p.name}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SceneLadeFehler(f"scene.json nicht lesbar: {p.name} ({type(exc).__name__})") from exc
    if not isinstance(data, dict) or "components" not in data:
        raise SceneLadeFehler(f"{p.name} ist kein scene-Dict (Schlüssel 'components' fehlt)")
    return data


def lade_pipeline_meta(scene_pfad: str | Path) -> dict[str, Any] | None:
    """Die ``meta.json`` neben der scene.json, falls vorhanden und lesbar."""
    meta = Path(scene_pfad).with_name("meta.json")
    if not meta.is_file():
        return None
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def berechne_mengen(
    scene: Any,
    *,
    geschoss: str | None = None,
    wall_height_m: float | None = None,
    pipeline_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mengen je Bauteilart aus einem scene-Dict (siehe ``mengen_aus_scene``)."""
    return mengen_aus_scene(
        scene,
        geschoss=geschoss,
        wall_height_m=wall_height_m,
        pipeline_meta=pipeline_meta,
        erzeuger={"name": "planproof_cad", "version": __version__},
    )


__all__ = ["SceneLadeFehler", "berechne_mengen", "lade_pipeline_meta", "lade_scene"]
