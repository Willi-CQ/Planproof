"""run_goldset.py Ende zu Ende mit einer erfundenen Pipeline (keine echte Pipeline, keine Kosten)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from goldset import metrics, run_goldset
from goldset.tests.test_metrics import _scene_from_gt


def perfekte_pipeline(input_path, output_dir, *args, **kwargs):
    """Liest die Ground Truth neben der DXF und schreibt sie als scene.json, mit verschobenem Ursprung
    (wie dwg_bim: Minimum aller Linien und Bloecke wird abgezogen)."""
    from types import SimpleNamespace

    gt = json.loads(Path(input_path).with_suffix(".json").read_text(encoding="utf-8"))
    ox, oy = metrics.dxf_loader_origin(input_path)
    scene = _scene_from_gt(gt, shift=(-ox, -oy))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    return SimpleNamespace(ok=True, error=None, warnings=[])


def kaputte_pipeline(input_path, output_dir, *args, **kwargs):
    raise RuntimeError("Boom")


def _run(monkeypatch, tmp_path, pipeline, limit=4):
    monkeypatch.setattr(
        sys, "argv",
        ["run_goldset.py", "--pipeline", pipeline, "--out", str(tmp_path), "--limit", str(limit)],
    )
    assert run_goldset.main() == 0
    return json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))


def test_lauf_mit_perfekter_pipeline(monkeypatch, tmp_path):
    rep = _run(monkeypatch, tmp_path, "goldset.tests.test_run_goldset:perfekte_pipeline")
    s = rep["zusammenfassung"]["gesamt"]
    assert s["dateien"] == 4 and s["fehlgeschlagen"] == 0
    assert s["wandlaenge_abw_pct_abs"]["median"] < 0.3
    assert s["raeume"]["recall"] == 1.0 and s["raeume"]["precision"] == 1.0
    assert s["tueren"]["recall"] == 1.0 and s["fenster"]["recall"] in (1.0, None)
    assert all(i["metriken"]["ausrichtung"]["modus"] == "dxf_ursprung" for i in rep["je_datei"])
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| **gesamt** |" in md and "synthetisch" in md


def test_lauf_mit_kaputter_pipeline_zaehlt_fehler(monkeypatch, tmp_path):
    rep = _run(monkeypatch, tmp_path, "goldset.tests.test_run_goldset:kaputte_pipeline", limit=2)
    s = rep["zusammenfassung"]["gesamt"]
    assert s["fehlgeschlagen"] == 2
    assert all("Boom" in i["fehler"] for i in rep["je_datei"])


def test_unbekannte_pipeline():
    with pytest.raises(SystemExit):
        run_goldset.resolve_pipeline("gibt_es_nicht")


# ---------------------------------------------------------------- Split dev / heldout
def test_split_regel_und_groessen(manifest):
    import hashlib

    plaene = manifest["plan_liste"]
    held = [p for p in plaene if run_goldset.is_heldout(p["plan_id"])]
    dev = [p for p in plaene if not run_goldset.is_heldout(p["plan_id"])]
    assert len(held) + len(dev) == len(plaene) == 40
    assert len(held) == 10 and sum(len(p["dateien"]) for p in held) == 20
    for p in plaene:
        regel = int(hashlib.sha256(p["plan_id"].encode()).hexdigest(), 16) % 4 == 0
        assert run_goldset.in_split(p["plan_id"], "heldout") is regel
        assert run_goldset.in_split(p["plan_id"], "dev") is (not regel)
        assert run_goldset.in_split(p["plan_id"], "all") is True
    with pytest.raises(ValueError):
        run_goldset.in_split("1", "foo")


def _run_split(monkeypatch, tmp_path, split, extra=()):
    log = tmp_path / "heldout.log"
    monkeypatch.setattr(
        sys, "argv",
        ["run_goldset.py", "--pipeline", "goldset.tests.test_run_goldset:perfekte_pipeline",
         "--out", str(tmp_path / "out"), "--split", split, "--heldout-log", str(log), *extra],
    )
    assert run_goldset.main() == 0
    return json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8")), log


def test_split_dev_ist_standard_und_schreibt_kein_log(monkeypatch, tmp_path, manifest):
    rep, log = _run_split(monkeypatch, tmp_path, "dev")
    assert rep["split"] == "dev" and rep["dateien"] == 54
    keys = {i["plan_key"] for i in rep["je_datei"]}
    ids = {p["plan_key"]: p["plan_id"] for p in manifest["plan_liste"]}
    assert not any(run_goldset.is_heldout(ids[k]) for k in keys)
    assert not log.exists()


def test_split_heldout_wird_protokolliert(monkeypatch, tmp_path):
    rep, log = _run_split(monkeypatch, tmp_path, "heldout", ("--comment", "Test ohne Git"))
    assert rep["split"] == "heldout" and rep["dateien"] == 20
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "split=heldout" in lines[0] and "dateien=20" in lines[0] and lines[0].endswith("Test ohne Git")


def test_split_all_protokolliert_und_zaehlt_alle(monkeypatch, tmp_path):
    rep, log = _run_split(monkeypatch, tmp_path, "all")
    assert rep["dateien"] == 74
    assert "split=all" in log.read_text(encoding="utf-8")


def konfig_pipeline(input_path, output_dir, *args, config=None, **kwargs):
    """Schreibt die erhaltene Konfiguration neben die scene.json und meldet einen leeren Lauf."""
    from types import SimpleNamespace

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_erhalten.json").write_text(json.dumps(config), encoding="utf-8")
    return SimpleNamespace(ok=False, error="nur Konfigurationstest", warnings=[])


def test_config_json_wird_an_die_pipeline_gereicht(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys, "argv",
        ["run_goldset.py", "--pipeline", "goldset.tests.test_run_goldset:konfig_pipeline",
         "--out", str(tmp_path), "--limit", "1", "--config-json", '{"wall_extend_junctions": false}'],
    )
    assert run_goldset.main() == 0
    rep = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert rep["config"] == {"wall_extend_junctions": False}
    erhalten = next(tmp_path.glob("*/config_erhalten.json"))
    assert json.loads(erhalten.read_text(encoding="utf-8")) == {"wall_extend_junctions": False}
