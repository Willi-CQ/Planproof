"""Seed-Loader: mitgelieferte JSONs → Katalog.

Liest strukturierte JSON-Dateien aus einem Seed-Verzeichnis und gibt sie als
typisierte ``MaterialSpec`` / ``TemplateSpec`` Objekte zurück. Standardmäßig
kommt das Verzeichnis aus den Package-Data des installierten Pakets
(``default_seed_root``, geladen über ``importlib.resources``), nie relativ zu
einem Repo-Pfad.

Erwartete Directory-Struktur::

    data/
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
Der Loader validiert via Pydantic. Eine ungültige Datei wird übersprungen und
mit Warnung geloggt; wer die Fehler braucht, übergibt eine Liste als ``errors``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from .schemas import MaterialSpec, TemplateSpec

logger = logging.getLogger(__name__)

SeedRoot = Path | Traversable


def default_seed_root() -> Traversable:
    """Wurzel der mitgelieferten Seeds (``planproof_cad/catalog/data``)."""
    return resources.files("planproof_cad.catalog").joinpath("data")


def _iter_json(root: SeedRoot) -> Iterator[SeedRoot]:
    """Alle ``*.json`` unter ``root`` rekursiv, je Ebene nach Name sortiert."""
    if not root.is_dir():
        return
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.is_dir():
            yield from _iter_json(entry)
        elif entry.name.endswith(".json"):
            yield entry


def iter_material_seeds(
    seed_dir: SeedRoot, errors: list[str] | None = None
) -> Iterator[MaterialSpec]:
    """Yield alle MaterialSpec-Einträge aus ``seed_dir/materials/*.json``."""
    materials_dir = seed_dir.joinpath("materials")
    if not materials_dir.is_dir():
        logger.debug("Keine materials/ unter %s", seed_dir)
        return
    for path in _iter_json(materials_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            yield MaterialSpec.model_validate(payload)
        except Exception as exc:
            logger.warning("Seed %s uebersprungen: %s", path, exc)
            if errors is not None:
                errors.append(f"{path.name}: {exc}")


def iter_template_seeds(
    seed_dir: SeedRoot, errors: list[str] | None = None
) -> Iterator[TemplateSpec]:
    """Yield alle TemplateSpec-Einträge aus ``seed_dir/templates/**/*.json``."""
    templates_dir = seed_dir.joinpath("templates")
    if not templates_dir.is_dir():
        logger.debug("Keine templates/ unter %s", seed_dir)
        return
    for path in _iter_json(templates_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            yield TemplateSpec.model_validate(payload)
        except Exception as exc:
            logger.warning("Seed %s uebersprungen: %s", path, exc)
            if errors is not None:
                errors.append(f"{path.name}: {exc}")


def load_all_seeds(seed_dir: SeedRoot) -> tuple[list[MaterialSpec], list[TemplateSpec]]:
    """Konvenienz: alle Seeds in zwei Listen laden."""
    materials = list(iter_material_seeds(seed_dir))
    templates = list(iter_template_seeds(seed_dir))
    logger.info(
        "Seeds geladen aus %s: %d Materialien, %d Templates",
        seed_dir,
        len(materials),
        len(templates),
    )
    return materials, templates


__all__ = [
    "default_seed_root",
    "iter_material_seeds",
    "iter_template_seeds",
    "load_all_seeds",
]
