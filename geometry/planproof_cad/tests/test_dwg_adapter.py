"""DWG-Schnittstelle: Auswahl des Konverters, Adapter mit Fake-Konverter (ein Skript), Fehlertexte.

Es wird kein echter ODA- oder LibreDWG-Aufruf gemacht. Der Fake-Konverter ist ein Python-Skript,
das wie die echten Programme aufgerufen wird und eine vorbereitete DXF an den Zielort kopiert.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dxfbau import haus

from planproof_cad.cad import dwg_adapter as da
from planproof_cad.cad import run_dwg_to_bim_pipeline
from planproof_cad.cad.dxf_loader import DwgConversionError, load_and_extract

FAKE_ODA = """
import os, shutil, sys
from pathlib import Path
# ODAFileConverter <in_dir> <out_dir> <version> <type> <recurse> <audit> <name>
in_dir, out_dir, version, typ, rec, audit, name = sys.argv[1:8]
assert typ == "DXF" and version == "ACAD2018", sys.argv
if os.environ.get("FAKE_FAIL"):
    sys.stderr.write("ODA: Lizenz abgelaufen\\n")
    sys.exit(3)
shutil.copy(os.environ["FAKE_DXF"], Path(out_dir) / (Path(name).stem + ".dxf"))
"""

FAKE_LIBREDWG = """
import os, shutil, sys
# dwg2dxf -y -o <out.dxf> <in.dwg>
args = sys.argv[1:]
assert "-o" in args, args
out = args[args.index("-o") + 1]
if os.environ.get("FAKE_FAIL"):
    sys.stderr.write("dwg2dxf: unsupported version AC1032\\n")
    sys.exit(1)
shutil.copy(os.environ["FAKE_DXF"], out)
"""


@pytest.fixture()
def dxf_vorlage(tmp_path, monkeypatch) -> Path:
    pfad = haus(tmp_path / "vorlage.dxf").speichern()
    monkeypatch.setenv("FAKE_DXF", str(pfad))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    for k in (da.ENV_CONVERTER, da.ENV_ODA, da.ENV_LIBREDWG):
        monkeypatch.delenv(k, raising=False)
    return pfad


@pytest.fixture()
def fake_oda(tmp_path) -> Path:
    p = tmp_path / "bin" / "fake_oda.py"
    p.parent.mkdir()
    p.write_text(FAKE_ODA, encoding="utf-8")
    return p


@pytest.fixture()
def fake_libredwg(tmp_path) -> Path:
    p = tmp_path / "bin" / "fake_dwg2dxf.py"
    p.parent.mkdir(exist_ok=True)
    p.write_text(FAKE_LIBREDWG, encoding="utf-8")
    return p


@pytest.fixture()
def dwg(tmp_path) -> Path:
    p = tmp_path / "in" / "plan.dwg"
    p.parent.mkdir()
    p.write_bytes(b"AC1032 kein echtes DWG")
    return p


def test_protocol_wird_von_allen_adaptern_erfuellt():
    for conv in (da.NoConverter(), da.OdaFileConverter("x"), da.LibreDwgConverter("x")):
        assert isinstance(conv, da.DwgConverter)


def test_standard_ist_no_converter_mit_klarer_meldung(dxf_vorlage, dwg, tmp_path):
    assert isinstance(da.resolve_converter(None), da.NoConverter)
    with pytest.raises(da.DwgConversionError) as ei:
        da.convert_dwg(dwg, tmp_path / "o")
    msg = str(ei.value)
    assert "docs/ODA_AND_DWG.md" in msg
    assert "converter='oda'" in msg and "CQPLAN_DWG_CONVERTER" in msg
    assert "DXF" in msg and "plan.dwg" in msg


def test_oda_adapter_mit_fake_skript(dxf_vorlage, fake_oda, dwg, tmp_path):
    out = da.OdaFileConverter(fake_oda).convert(dwg, tmp_path / "o")
    assert out == tmp_path / "o" / "plan.dxf" and out.exists()


def test_oda_pfad_aus_umgebungsvariable(dxf_vorlage, fake_oda, dwg, tmp_path, monkeypatch):
    monkeypatch.setenv(da.ENV_ODA, str(fake_oda))
    assert da.OdaFileConverter().convert(dwg, tmp_path / "o").exists()


def test_oda_ohne_pfad_meldet_fehlenden_konverter(dxf_vorlage, dwg, tmp_path, monkeypatch):
    monkeypatch.setattr(da.shutil, "which", lambda _n: None)
    with pytest.raises(da.DwgConversionError, match="ODA File Converter nicht gefunden"):
        da.OdaFileConverter().convert(dwg, tmp_path / "o")


def test_oda_fehler_wird_mit_ausgabe_gemeldet(dxf_vorlage, fake_oda, dwg, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "1")
    with pytest.raises(da.DwgConversionError) as ei:
        da.OdaFileConverter(fake_oda).convert(dwg, tmp_path / "o")
    assert "Lizenz abgelaufen" in str(ei.value) and "Rückgabecode 3" in str(ei.value)


def test_libredwg_adapter_mit_fake_skript(dxf_vorlage, fake_libredwg, dwg, tmp_path):
    out = da.LibreDwgConverter(fake_libredwg).convert(dwg, tmp_path / "o")
    assert out.exists() and out.suffix == ".dxf"


def test_libredwg_fehler_wird_mit_ausgabe_gemeldet(
    dxf_vorlage, fake_libredwg, dwg, tmp_path, monkeypatch
):
    monkeypatch.setenv("FAKE_FAIL", "1")
    with pytest.raises(da.DwgConversionError, match="unsupported version AC1032"):
        da.LibreDwgConverter(fake_libredwg).convert(dwg, tmp_path / "o")


def test_programm_nicht_startbar_ist_eine_fehlermeldung(dwg, tmp_path):
    with pytest.raises(da.DwgConversionError, match="nicht"):
        da.LibreDwgConverter(tmp_path / "gibt_es_nicht.exe").convert(dwg, tmp_path / "o")


def test_auswahl_per_umgebungsvariable(dxf_vorlage, fake_oda, dwg, tmp_path, monkeypatch):
    monkeypatch.setenv(da.ENV_CONVERTER, "oda")
    monkeypatch.setenv(da.ENV_ODA, str(fake_oda))
    assert isinstance(da.resolve_converter(None), da.OdaFileConverter)
    assert da.convert_dwg(dwg, tmp_path / "o").exists()


def test_parameter_hat_vorrang_vor_umgebungsvariable(monkeypatch):
    monkeypatch.setenv(da.ENV_CONVERTER, "oda")
    assert isinstance(da.resolve_converter("none"), da.NoConverter)
    assert isinstance(da.resolve_converter("libredwg"), da.LibreDwgConverter)


def test_unbekannter_name_wird_abgelehnt():
    with pytest.raises(da.DwgConversionError, match="Unbekannter DWG-Konverter"):
        da.resolve_converter("autocad")


def test_instanz_als_parameter(dxf_vorlage, fake_oda, dwg, tmp_path):
    conv = da.OdaFileConverter(fake_oda)
    assert da.resolve_converter(conv) is conv


def test_loader_nutzt_den_konverter_aus_der_konfiguration(dxf_vorlage, fake_oda, dwg, tmp_path):
    ex = load_and_extract(
        dwg, {"converter": da.OdaFileConverter(fake_oda)}, work_dir=tmp_path / "w"
    )
    assert len(ex.segments) == 8 and ex.unit_label == "mm"


def test_loader_ohne_konverter_wirft_die_klare_meldung(dxf_vorlage, dwg, tmp_path):
    with pytest.raises(DwgConversionError, match="docs/ODA_AND_DWG.md"):
        load_and_extract(dwg, {}, work_dir=tmp_path / "w")


def test_pipeline_mit_dwg_und_fake_konverter(dxf_vorlage, fake_libredwg, dwg, tmp_path):
    res = run_dwg_to_bim_pipeline(
        dwg,
        tmp_path / "out",
        work_dir=tmp_path / "w",
        config={"converter": da.LibreDwgConverter(fake_libredwg)},
    )
    assert res.ok, res.error
    assert res.metrics["wall_reconstruction"]["centerlines"] == 4


def test_pipeline_meldet_konverterfehler_im_fehlertext(
    dxf_vorlage, fake_oda, dwg, tmp_path, monkeypatch
):
    monkeypatch.setenv("FAKE_FAIL", "1")
    res = run_dwg_to_bim_pipeline(
        dwg,
        tmp_path / "out",
        work_dir=tmp_path / "w",
        config={"converter": da.OdaFileConverter(fake_oda)},
    )
    assert not res.ok
    assert "Lizenz abgelaufen" in res.error  # nicht als logger.debug verschluckt


def test_pipeline_dwg_ohne_konverter_nennt_die_loesungswege(dxf_vorlage, dwg, tmp_path):
    res = run_dwg_to_bim_pipeline(dwg, tmp_path / "out", work_dir=tmp_path / "w")
    assert not res.ok and "docs/ODA_AND_DWG.md" in res.error
    assert "kein DWG-Konverter" in json.dumps(res.error, ensure_ascii=False)


def test_kompatibilitaets_shim_fuer_aeltere_aufrufer(
    dxf_vorlage, fake_oda, dwg, tmp_path, monkeypatch
):
    """services.list_layers ruft ``_convert_dwg_to_dxf(dwg, work_dir)`` auf; die Umgebung waehlt den
    Konverter."""
    from planproof_cad.cad.dxf_loader import _convert_dwg_to_dxf

    monkeypatch.setenv(da.ENV_CONVERTER, "oda")
    monkeypatch.setenv(da.ENV_ODA, str(fake_oda))
    assert _convert_dwg_to_dxf(dwg, tmp_path / "w").exists()
