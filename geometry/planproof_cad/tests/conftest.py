"""Gemeinsame Fixtures: gebautes Wheel und frisches venv ausserhalb des Repos.

Das Wheel wird aus einer Kopie des Quellbaums gebaut (nur pyproject.toml, README.md,
src/), damit der Build weder das Repo mit build/ und egg-info verschmutzt noch
heimlich auf Dateien ausserhalb des Pakets zugreift.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PACKAGE_ROOT.parent / "planproof_geometry"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        **kwargs,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"Befehl fehlgeschlagen ({proc.returncode}): {' '.join(cmd)}\n"
            f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
        )
    return proc


def venv_python(venv_dir: Path) -> Path:
    sub = "Scripts" if sys.platform == "win32" else "bin"
    exe = "python.exe" if sys.platform == "win32" else "python"
    return venv_dir / sub / exe


def venv_script(venv_dir: Path, name: str) -> Path:
    sub = "Scripts" if sys.platform == "win32" else "bin"
    return venv_dir / sub / (name + (".exe" if sys.platform == "win32" else ""))


@pytest.fixture(scope="session")
def uv_exe() -> str:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv nicht im PATH: Wheel-Build und frisches venv sind ohne uv nicht pruefbar")
    return uv


@pytest.fixture(scope="session")
def built_wheel(uv_exe: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Baut das Wheel mit ``uv build --wheel`` aus einer Kopie des Quellbaums."""
    src_copy = tmp_path_factory.mktemp("wheel_src")
    shutil.copy2(PACKAGE_ROOT / "pyproject.toml", src_copy / "pyproject.toml")
    shutil.copy2(PACKAGE_ROOT / "README.md", src_copy / "README.md")
    # Lizenzdateien (falls vorhanden) mitnehmen, damit das Wheel sie wie im Release traegt.
    for pattern in ("LICENSE*", "NOTICE*", "THIRD_PARTY*"):
        for datei in PACKAGE_ROOT.glob(pattern):
            shutil.copy2(datei, src_copy / datei.name)
    shutil.copytree(
        PACKAGE_ROOT / "src",
        src_copy / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    dist = tmp_path_factory.mktemp("wheel_dist")
    _run([uv_exe, "build", "--wheel", "--out-dir", str(dist), str(src_copy)])
    wheels = list(dist.glob("planproof_cad-*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]


@pytest.fixture(scope="session")
def wheel_venv(uv_exe: str, built_wheel: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Frisches venv ausserhalb des Repos, nur das Wheel (mit Abhaengigkeiten) installiert."""
    venv_dir = tmp_path_factory.mktemp("fresh_venv") / "venv"
    _run(
        [
            uv_exe,
            "venv",
            "--python",
            f"{sys.version_info.major}.{sys.version_info.minor}",
            str(venv_dir),
        ]
    )
    _run([uv_exe, "pip", "install", "--python", str(venv_python(venv_dir)), str(built_wheel)])
    return venv_dir
