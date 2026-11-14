"""Lazy import of the perfect-BIM pipeline modules.

Historie: Bis ADR-016 lagen ``diagnose_door_placement`` /
``merge_doors_and_windows_bim`` / ``add_stairs_to_bim`` als Standalone-
Skripte in ``scripts/`` und wurden hier per ``sys.path``-Insert geladen.
Seit dem Library-Move sind sie regulaere Geschwister-Module dieses
Packages — die Loader-API bleibt erhalten, damit die Wrapper
(``door_pipeline`` / ``full_bim_pipeline`` / ``stair_pipeline``)
unveraendert funktionieren und die schweren Abhaengigkeiten (cv2,
shapely, trimesh) weiterhin erst beim ersten Aufruf importiert werden,
nicht schon beim Import des Packages.
"""
from __future__ import annotations

from types import ModuleType


def load_diagnose_module() -> ModuleType:
    """Return the ``diagnose_door_placement`` module (lazy, py-cached)."""
    from . import diagnose_door_placement as _diag

    return _diag


def load_merge_module() -> ModuleType:
    """Return the ``merge_doors_and_windows_bim`` module (lazy, py-cached)."""
    from . import merge_doors_and_windows_bim as _merge

    return _merge


def load_stairs_module() -> ModuleType:
    """Return the ``add_stairs_to_bim`` module (lazy, py-cached)."""
    from . import add_stairs_to_bim as _stairs

    return _stairs
