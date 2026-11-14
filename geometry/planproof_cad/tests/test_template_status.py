"""Ein Fehlschlag der Template-Zuweisung ist sichtbar (kein stiller Soft-Fail).

Die Quell-Pipeline schluckte einen nicht ladbaren Katalog mit einer Log-Zeile. Hier
muss der Status ``template_assignment = {ok, error, templates_loaded, ...}`` in
meta.json und im Ergebnisobjekt stehen.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from planproof_cad.cad import run_dwg_to_bim_pipeline
from planproof_cad.cad.synth import build_demo_plan
from planproof_cad.catalog.seed_loader import default_seed_root
from planproof_cad.catalog.template_assignment import TemplateAssignmentStep

SCHLUESSEL = {"ok", "error", "templates_loaded", "seed_files_skipped"}


def _lauf(tmp_path: Path):
    dxf = build_demo_plan(tmp_path / "demo.dxf")
    out = tmp_path / "out"
    res = run_dwg_to_bim_pipeline(str(dxf), str(out), str(tmp_path / "work"), {})
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    return res, meta


def test_erfolg_steht_in_ergebnis_und_meta(tmp_path):
    res, meta = _lauf(tmp_path)
    assert res.ok
    for status in (res.template_assignment, meta["template_assignment"]):
        assert set(status) >= SCHLUESSEL
        assert status["ok"] is True
        assert status["error"] is None
        assert status["templates_loaded"] > 0
        assert status["seed_files_skipped"] == 0
    assert not any("Template-Zuweisung" in w for w in res.warnings)


def test_leerer_katalog_ist_sichtbar(tmp_path, monkeypatch):
    leer = tmp_path / "leer_seed"
    (leer / "templates").mkdir(parents=True)
    monkeypatch.setattr("planproof_cad.catalog.seed_loader.default_seed_root", lambda: leer)

    res, meta = _lauf(tmp_path)

    # Das Modell entsteht weiter (IFC und GLB), aber der Ausfall ist nicht verborgen.
    assert res.ok
    for status in (res.template_assignment, meta["template_assignment"]):
        assert status["ok"] is False
        assert status["templates_loaded"] == 0
        assert "Keine Templates" in status["error"]
    assert any("Template-Zuweisung nicht erfolgreich" in w for w in res.warnings)
    assert any("Template-Zuweisung nicht erfolgreich" in w for w in meta["warnings"])


def test_fehlendes_seed_verzeichnis_ist_sichtbar(tmp_path, monkeypatch):
    fehlt = tmp_path / "gibt_es_nicht"
    monkeypatch.setattr("planproof_cad.catalog.seed_loader.default_seed_root", lambda: fehlt)

    res, meta = _lauf(tmp_path)

    assert res.ok
    assert res.template_assignment["ok"] is False
    assert "nicht gefunden" in res.template_assignment["error"]
    assert meta["template_assignment"] == res.template_assignment


def test_ausnahme_im_schritt_ist_sichtbar(tmp_path, monkeypatch):
    def kaputt(self, ctx, state):
        raise RuntimeError("Zuweisung explodiert")

    monkeypatch.setattr(TemplateAssignmentStep, "run", kaputt)

    res, meta = _lauf(tmp_path)

    assert res.ok
    assert res.template_assignment["ok"] is False
    assert "Zuweisung explodiert" in res.template_assignment["error"]
    assert meta["template_assignment"]["ok"] is False
    assert any("Zuweisung explodiert" in w for w in res.warnings)


def test_beschaedigte_seed_datei_wird_gezaehlt(tmp_path):
    seed = tmp_path / "seed"
    shutil.copytree(
        str(default_seed_root()), str(seed), ignore=shutil.ignore_patterns("__pycache__")
    )
    (seed / "templates" / "walls" / "kaputt.json").write_text("{kein json", encoding="utf-8")

    step = TemplateAssignmentStep(seed_dir=seed)
    state = SimpleNamespace(
        scene=SimpleNamespace(
            components={
                "w": SimpleNamespace(
                    id="w",
                    kind="WALL",
                    center=(0, 0, 0),
                    dims=(5.0, 2.8, 0.365),
                    attrs={"wall_type": "exterior"},
                )
            }
        ),
        config={},
        metrics={},
    )
    step.run(SimpleNamespace(log=lambda _m: None), state)

    assert state.template_assignment["ok"] is True
    assert state.template_assignment["seed_files_skipped"] == 1
    assert state.template_assignment["templates_loaded"] > 0


@pytest.mark.parametrize(
    ("config", "scene", "erwartet"),
    [
        (
            {"template_assignment_enabled": False},
            SimpleNamespace(components={"x": 1}),
            "deaktiviert",
        ),
        ({}, None, "keine Scene-Components"),
    ],
)
def test_uebersprungene_zuweisung_traegt_grund(config, scene, erwartet):
    state = SimpleNamespace(scene=scene, config=config, metrics={})
    TemplateAssignmentStep().run(SimpleNamespace(log=lambda _m: None), state)
    assert state.template_assignment["ok"] is False
    assert erwartet in state.template_assignment["error"]


def test_deaktivierung_per_config_erzeugt_keine_warnung(tmp_path):
    dxf = build_demo_plan(tmp_path / "demo.dxf")
    res = run_dwg_to_bim_pipeline(
        str(dxf),
        str(tmp_path / "out"),
        str(tmp_path / "work"),
        {"template_assignment_enabled": False},
    )
    assert res.ok
    assert res.template_assignment["ok"] is False
    assert "deaktiviert" in res.template_assignment["error"]
    assert not any("Template-Zuweisung nicht erfolgreich" in w for w in res.warnings)
