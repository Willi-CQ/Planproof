"""Das gebaute Wheel: Inhalt, Seeds als Package-Data und Lauf im frischen venv.

``built_wheel`` und ``wheel_venv`` kommen aus conftest.py (``uv build --wheel``, frisches
venv ausserhalb des Repos).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
from conftest import PACKAGE_ROOT, venv_python, venv_script

VERBOTENE_ENDUNGEN = (".pt", ".pth", ".onnx", ".ipynb")
VERBOTENE_VERZEICHNISSE = {"training", "models"}
VERBOTENE_ABHAENGIGKEITEN = (
    "torch",
    "torchvision",
    "ultralytics",
    "opencv",
    "scipy",
    "scikit-image",
    "onnxruntime",
    "pillow",
    "pypdfium2",
    "trimesh",
    "open3d",
    "scikit-learn",
    "pymupdf",
    "httpx",
    "planproof-core",
    "planproof_geometry",
)
SEED_QUELLE = PACKAGE_ROOT / "src" / "planproof_cad" / "catalog" / "data"


def _wheel_namen(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def test_wheel_ohne_modelle_und_notebooks(built_wheel: Path) -> None:
    namen = _wheel_namen(built_wheel)
    for name in namen:
        assert not name.lower().endswith(VERBOTENE_ENDUNGEN), name
        teile = set(name.lower().split("/"))
        assert not (teile & VERBOTENE_VERZEICHNISSE), name
        assert "planproof_geometry" not in name, name


def test_wheel_enthaelt_alle_seeds(built_wheel: Path) -> None:
    namen = _wheel_namen(built_wheel)
    im_wheel = sorted(
        n.split("planproof_cad/catalog/data/", 1)[1]
        for n in namen
        if n.startswith("planproof_cad/catalog/data/") and n.endswith(".json")
    )
    in_quelle = sorted(p.relative_to(SEED_QUELLE).as_posix() for p in SEED_QUELLE.rglob("*.json"))
    assert len(in_quelle) > 0
    assert im_wheel == in_quelle
    assert any(n.startswith("templates/walls/") for n in im_wheel)
    assert any(n.startswith("materials/") for n in im_wheel)


def test_seeds_sind_nicht_von_git_ignoriert() -> None:
    """Das Wurzel-.gitignore hat eine Regel ``data/``; ohne Gegenregel gingen die Seeds
    beim Commit still verloren und ein frischer Checkout baute ein Wheel ohne Katalog."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("git nicht im PATH")
    beispiel = next(SEED_QUELLE.rglob("*.json"))
    proc = subprocess.run(
        [git, "check-ignore", "-q", str(beispiel)],
        capture_output=True,
        cwd=str(PACKAGE_ROOT),
    )
    if proc.returncode == 128:
        pytest.skip("kein git-Repository")
    assert proc.returncode == 1, "Seeds werden von git ignoriert (check-ignore Treffer)"


def test_wheel_metadaten_ohne_verbotene_abhaengigkeiten(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as zf:
        meta_name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        meta = zf.read(meta_name).decode("utf-8")
    requires = [ln for ln in meta.splitlines() if ln.startswith("Requires-Dist:")]
    assert requires, "Wheel traegt keine Abhaengigkeiten"
    # Extras (rest, dev ...) werden nur auf Wunsch installiert: httpx steht nur im dev-Extra
    # (TestClient). Die Pflicht-Abhaengigkeiten sind die Zeilen ohne "extra ==".
    requires = [ln for ln in requires if "extra ==" not in ln]
    assert requires, "Wheel traegt keine Pflicht-Abhaengigkeiten"
    for ln in requires:
        paket = ln.split(":", 1)[1].strip().lower()
        for verboten in VERBOTENE_ABHAENGIGKEITEN:
            assert not paket.startswith(verboten), ln
    assert "Name: planproof-cad" in meta
    assert "Version: 0.1.0" in meta


def test_wheel_im_frischen_venv_laedt_seeds_und_laeuft(wheel_venv: Path, tmp_path: Path) -> None:
    py = venv_python(wheel_venv)

    # 1) Das installierte Paket liegt im venv, nicht im Repo.
    proc = subprocess.run(
        [str(py), "-I", "-c", "import planproof_cad, sys; print(planproof_cad.__file__)"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        check=True,
    )
    pfad = Path(proc.stdout.strip()).resolve()
    assert "site-packages" in pfad.parts, pfad
    assert PACKAGE_ROOT not in pfad.parents, pfad

    # 2) Templates > 0 nach Installation aus dem Wheel.
    proc = subprocess.run(
        [
            str(py),
            "-I",
            "-c",
            "from planproof_cad.catalog.seed_loader import default_seed_root, "
            "load_all_seeds; m, t = load_all_seeds(default_seed_root()); "
            "print(len(m), len(t))",
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        check=True,
    )
    materialien, templates = (int(x) for x in proc.stdout.split())
    assert templates > 0
    assert materialien > 0

    # 3) Demo-Lauf ueber das installierte Skript planproof-cad.
    out_dir = tmp_path / "demo_out"
    proc = subprocess.run(
        [str(venv_script(wheel_venv, "planproof-cad")), "demo", str(out_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    ta = meta["template_assignment"]
    assert ta["ok"] is True, ta
    assert ta["templates_loaded"] == templates
    assert ta["seed_files_skipped"] == 0
    assert (out_dir / "plan.ifc").stat().st_size > 0
    assert (out_dir / "model.glb").stat().st_size > 0
