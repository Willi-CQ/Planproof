"""Reproduzierbarkeit: zweimal bauen ergibt identische SHA-256.

Der schnelle Test rendert einen handgemachten Plan in allen Varianten doppelt (laeuft immer). Der
volle Test baut das komplette Goldset zweimal aus den Rohdaten und vergleicht mit dem Manifest;
er laeuft nur, wenn die Umgebungsvariable SD_GEOMETRIES_CSV auf die geometries.csv zeigt (ca. 1 GB,
liegt ausserhalb des Repos), und braucht etwa vier Minuten.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from goldset import render_dxf, sd_plan


def _plan():
    from goldset.tests.test_sd_plan import _room_floor

    return sd_plan.analysiere_geschoss(_room_floor(), "7", "9").plan


def test_render_ist_deterministisch_in_allen_varianten():
    plan = _plan()
    for v in render_dxf.VARIANTEN:
        a, info_a = render_dxf.render_dxf(plan, v, "sd_p00007")
        b, info_b = render_dxf.render_dxf(plan, v, "sd_p00007")
        assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest(), v["id"]
        assert info_a == info_b
        assert b"\r" not in a
    # andere Seeds geben (bei Zufallsanteilen wie Anschlagseite) andere Bytes, aber ebenfalls stabile
    c1, _ = render_dxf.render_dxf(plan, render_dxf.VARIANTEN[0], "sd_p00007", seed=1)
    c2, _ = render_dxf.render_dxf(plan, render_dxf.VARIANTEN[0], "sd_p00007", seed=1)
    assert c1 == c2


def test_keine_zeitstempel_oder_guid_im_dxf():
    plan = _plan()
    data, _ = render_dxf.render_dxf(plan, render_dxf.VARIANTEN[0], "sd_p00007")
    text = data.decode("utf-8")
    # feste Metadaten: zweimal rendern mit Pause aendert nichts (siehe oben); hier: kein Rechnername o.ae.
    assert "$FINGERPRINTGUID" in text and "$VERSIONGUID" in text
    import time

    time.sleep(1.1)
    data2, _ = render_dxf.render_dxf(plan, render_dxf.VARIANTEN[0], "sd_p00007")
    assert data == data2


@pytest.mark.skipif(not os.environ.get("SD_GEOMETRIES_CSV"), reason="SD_GEOMETRIES_CSV nicht gesetzt (Rohdaten ausserhalb des Repos)")
def test_voller_neubau_zweimal_identisch(tmp_path):
    csv = os.environ["SD_GEOMETRIES_CSV"]
    cache = os.environ.get("SD_INDEX_CACHE")
    script = Path(__file__).resolve().parents[1] / "build_goldset.py"
    manifest = json.loads((Path(__file__).resolve().parents[1] / "data" / "manifest.json").read_text(encoding="utf-8"))
    hashes = []
    for run in ("a", "b"):
        out = tmp_path / run
        cmd = [sys.executable, str(script), "--csv", csv, "--out", str(out)]
        if cache:
            cmd += ["--index-cache", cache]
        subprocess.run(cmd, check=True, capture_output=True)
        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        hashes.append({f["dxf"]: f["dxf_sha256"] for p in m["plan_liste"] for f in p["dateien"]})
    assert hashes[0] == hashes[1]
    assert hashes[0] == {f["dxf"]: f["dxf_sha256"] for p in manifest["plan_liste"] for f in p["dateien"]}
