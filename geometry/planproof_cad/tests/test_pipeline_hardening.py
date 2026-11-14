"""Regression: Pipeline-Haertung (uebernommen aus planproof_geometry, Stand 2026-06-28).

* ``run_dwg_to_bim_pipeline`` meldete ``ok=True``, auch wenn GLB- UND IFC-Export
  beide (nur als Warnung geschluckt) fehlschlugen -> leeres Bundle als Erfolg.

Der Cloud-Dispatch- und der Scan-Datei-Test der Quelle gehoeren nicht zu diesem
Paket und sind nicht uebernommen.
"""

from __future__ import annotations

from planproof_cad.cad import run_dwg_to_bim_pipeline
from planproof_cad.cad.synth import build_demo_plan


def test_dwg_pipeline_ok_false_when_both_exports_fail(tmp_path, monkeypatch):
    demo = build_demo_plan(tmp_path / "demo.dxf")

    def _boom(*args, **kwargs):
        raise RuntimeError("export kaputt")

    # Beide Export-Pfade (funktions-lokal importiert) zum Scheitern bringen.
    monkeypatch.setattr("planproof_cad.export.gltf.export_scene_to_gltf", _boom)
    monkeypatch.setattr("planproof_cad.export.ifc.IfcExporter", _boom)

    res = run_dwg_to_bim_pipeline(str(demo), str(tmp_path / "out"), str(tmp_path / "work"), {})

    assert res.ok is False
    assert res.model_path is None
    assert res.ifc_path is None


def test_dwg_pipeline_ok_true_on_happy_path(tmp_path):
    """Gegenprobe: ohne sabotierte Exporte bleibt der Lauf erfolgreich."""
    demo = build_demo_plan(tmp_path / "demo.dxf")
    res = run_dwg_to_bim_pipeline(str(demo), str(tmp_path / "out"), str(tmp_path / "work"), {})
    assert res.ok is True
    assert res.model_path is not None or res.ifc_path is not None
