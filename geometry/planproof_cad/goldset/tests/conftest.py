"""Gemeinsame Fixtures. Die Tests brauchen nur ezdxf, shapely, numpy und pytest, keine Pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # packages/planproof_cad
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "goldset" / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def manifest() -> dict:
    return json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def all_files(manifest) -> list:
    """Liste (plan_eintrag, datei_eintrag) ueber alle DXF-Dateien des Manifests."""
    return [(p, f) for p in manifest["plan_liste"] for f in p["dateien"]]
