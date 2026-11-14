"""Seed-Loader: Repo-JSONs → Katalog.

Wird in Sprint 6 genutzt um den Shared-Katalog initial zu befüllen. Liest
strukturierte JSON-Dateien aus einem Seed-Directory und gibt sie als typisierte
``MaterialSpec`` / ``TemplateSpec`` Objekte zurück. Der Aufrufer entscheidet, ob
der Upsert via API (Platform-Admin-Token) oder direkt in der DB (Migration)
erfolgt.

Erwartete Directory-Struktur::

    seed/
      materials/
        ks_24.json
        wdvs_eps_12.json
        ...
      templates/
        walls/
          aw_ks24_wdvs12_putz.json
          ...
        doors/
        windows/
        slabs/
        roofs/
        stairs/
        spaces/

Jede JSON-Datei enthält genau ein MaterialSpec- oder TemplateSpec-Dokument.
Der Loader validiert via Pydantic und raised bei Schema-Fehlern.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from .schemas import MaterialSpec, TemplateSpec

logger = logging.getLogger(__name__)


def iter_material_seeds(seed_dir: Path) -> Iterator[MaterialSpec]:
    """Yield alle MaterialSpec-Einträge aus ``seed_dir/materials/*.json``."""
    materials_dir = seed_dir / "materials"
    if not materials_dir.is_dir():
        logger.debug("Keine materials/ unter %s", seed_dir)
        return
    for path in sorted(materials_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            yield MaterialSpec.model_validate(payload)
        except Exception as exc:
            logger.warning("Seed %s uebersprungen: %s", path, exc)


def iter_template_seeds(seed_dir: Path) -> Iterator[TemplateSpec]:
    """Yield alle TemplateSpec-Einträge aus ``seed_dir/templates/**/*.json``."""
    templates_dir = seed_dir / "templates"
    if not templates_dir.is_dir():
        logger.debug("Keine templates/ unter %s", seed_dir)
        return
    for path in sorted(templates_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            yield TemplateSpec.model_validate(payload)
        except Exception as exc:
            logger.warning("Seed %s uebersprungen: %s", path, exc)


def load_all_seeds(seed_dir: Path) -> tuple[list[MaterialSpec], list[TemplateSpec]]:
    """Konvenienz: alle Seeds in zwei Listen laden."""
    materials = list(iter_material_seeds(seed_dir))
    templates = list(iter_template_seeds(seed_dir))
    logger.info("Seeds geladen aus %s: %d Materialien, %d Templates", seed_dir, len(materials), len(templates))
    return materials, templates


__all__ = [
    "iter_material_seeds",
    "iter_template_seeds",
    "load_all_seeds",
]
