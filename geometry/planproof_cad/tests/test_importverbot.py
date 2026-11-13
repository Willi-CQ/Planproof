"""Importverbot: der Demo-Lauf darf keines der schweren oder lizenzkritischen Module laden.

Der Lauf geschieht in einem Subprozess mit ``python -I`` (isolierter Modus). Davor
wird ein ``sys.meta_path``-Blocker installiert, der die verbotenen Module mit
``ImportError`` sperrt, auch wenn sie im venv liegen. Danach wird geprueft, dass
keines in ``sys.modules`` steht. Zusaetzlich wird die Liste aller geladenen
Fremdpakete (alles ausser Standardbibliothek und planproof_cad) ausgegeben, damit
neue Abhaengigkeiten im Review sichtbar werden.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FORBIDDEN = [
    "torch",
    "torchvision",
    "ultralytics",
    "cv2",
    "scipy",
    "skimage",
    "onnxruntime",
    "PIL",
    "pypdfium2",
    "trimesh",
    "open3d",
    "sklearn",
    "fitz",
    "pymupdf",
    "httpx",
    "planproof_geometry",
]

RUNNER = r"""
import importlib.abc
import json
import sys

FORBIDDEN = {forbidden!r}


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in FORBIDDEN:
            raise ImportError("gesperrt durch test_importverbot: " + fullname)
        return None


sys.meta_path.insert(0, _Blocker())

from planproof_cad.cli import main

rc = main(["demo", {out_dir!r}])

loaded_forbidden = sorted(
    name for name in sys.modules if name.split(".", 1)[0] in FORBIDDEN
)
stdlib = set(sys.stdlib_module_names)
third_party = sorted(
    {{
        name.split(".", 1)[0]
        for name in sys.modules
        if name.split(".", 1)[0] not in stdlib
        and not name.startswith("_")
    }}
)
with open({report!r}, "w", encoding="utf-8") as fh:
    json.dump(
        {{"rc": rc, "loaded_forbidden": loaded_forbidden, "third_party": third_party}}, fh
    )
"""


def _run_runner(
    tmp_path: Path, forbidden: list[str]
) -> tuple[subprocess.CompletedProcess, Path, Path]:
    out_dir = tmp_path / "out"
    report = tmp_path / "report.json"
    script = tmp_path / "runner.py"
    script.write_text(
        RUNNER.format(forbidden=set(forbidden), out_dir=str(out_dir), report=str(report)),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-I", str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        cwd=str(tmp_path),
    )
    return proc, out_dir, report


def test_demo_laeuft_ohne_verbotene_module(tmp_path: Path) -> None:
    proc, out_dir, report = _run_runner(tmp_path, FORBIDDEN)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["rc"] == 0
    assert data["loaded_forbidden"] == []
    # Der Lauf hat wirklich ein Modell erzeugt (kein stiller Soft-Fail).
    assert (out_dir / "plan.ifc").exists()
    assert (out_dir / "model.glb").exists()
    # Die deklarierten Pflicht-Abhaengigkeiten sind tatsaechlich im Einsatz.
    third_party = set(data["third_party"])
    assert {"numpy", "shapely", "ezdxf", "pydantic", "ifcopenshell"} <= third_party
    assert "planproof_cad" in third_party


def test_blocker_wirkt_wirklich(tmp_path: Path) -> None:
    """Gegenprobe: sperrt der Blocker ein benoetigtes Modul, schlaegt der Lauf fehl.

    Ohne diese Probe koennte der Blocker wirkungslos sein und der Haupttest
    trotzdem gruen bleiben.
    """
    proc, _out_dir, report = _run_runner(tmp_path, ["numpy"])
    assert proc.returncode != 0
    assert "gesperrt durch test_importverbot" in proc.stderr
    assert not report.exists()
