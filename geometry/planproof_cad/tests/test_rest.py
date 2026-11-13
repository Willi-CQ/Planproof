"""REST-Schnittstelle: Endpunkte, Pfadschutz, Groessenlimit.

Braucht die Extras ``rest`` und ``dev`` (fastapi, python-multipart, httpx fuer den TestClient).
"""

from __future__ import annotations

import json

import pytest
from mengen_support import HAUS_EG, lade

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from planproof_cad.cad.synth import build_demo_plan  # noqa: E402
from planproof_cad.rest import create_app  # noqa: E402


@pytest.fixture()
def werk(tmp_path):
    return tmp_path / "werk"


@pytest.fixture()
def client(werk):
    return TestClient(create_app(workdir=werk, max_upload_mb=5))


@pytest.fixture(scope="module")
def demo_dxf(tmp_path_factory):
    return build_demo_plan(tmp_path_factory.mktemp("rest_dxf") / "demo.dxf")


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_app_legt_arbeitsverzeichnis_an(werk):
    app = create_app(workdir=werk)
    assert (werk / "uploads").is_dir() and (werk / "runs").is_dir()
    assert app.state.workdir == werk.resolve()


def test_convert_upload(client, demo_dxf):
    with open(demo_dxf, "rb") as f:
        r = client.post(
            "/v1/convert",
            files={"file": ("demo.dxf", f, "application/dxf")},
            data={"geschoss": "EG", "wall_height_m": "2.8"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["input"] == "demo.dxf"
    namen = {x["name"] for x in body["files"]}
    assert {"plan.ifc", "model.glb", "scene.json", "mengen.json", "mengen_de.csv"} <= namen
    assert "elemente" not in body["mengen"]
    assert body["mengen"]["geschoss"] == "EG"
    assert {p["kg"] for p in body["kg_vorschlaege"]["proposals"]} == {"330", "334", "340", "344"}
    assert all(p["status"] == "vorschlag" for p in body["kg_vorschlaege"]["proposals"])
    # Download einer Ergebnisdatei
    url = next(x["url"] for x in body["files"] if x["name"] == "mengen.json")
    d = client.get(url)
    assert d.status_code == 200
    assert json.loads(d.content)["geschoss"] == "EG"
    g = client.get(next(x["url"] for x in body["files"] if x["name"] == "mengen_de.csv"))
    assert g.content.startswith(b"\xef\xbb\xbfKategorie;ID;Typ")


def test_convert_pfad_im_arbeitsverzeichnis(client, werk, demo_dxf):
    ziel = werk / "eingang"
    ziel.mkdir(parents=True)
    (ziel / "plan.dxf").write_bytes(demo_dxf.read_bytes())
    r = client.post("/v1/convert", data={"path": "eingang/plan.dxf"})
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True


@pytest.mark.parametrize(
    "boese",
    [
        "../geheim.dxf",
        "eingang/../../geheim.dxf",
        "..\\geheim.dxf",
        "C:/Windows/win.ini",
        "/etc/passwd",
    ],
)
def test_pfad_ausserhalb_wird_abgewiesen(client, werk, boese):
    for ep in ("/v1/convert", "/v1/layers"):
        r = client.post(ep, data={"path": boese})
        assert r.status_code == 403, (ep, boese, r.text)
        assert r.json()["error"]["code"] == "path_outside_root"
    r = client.post("/v1/takeoff", json={"path": boese})
    assert r.status_code == 403
    r = client.post("/v1/kg-proposals", json={"path": boese})
    assert r.status_code == 403


def test_pfad_traversal_ueber_nachbarordner_mit_gleichem_praefix(tmp_path):
    werk = tmp_path / "werk"
    nachbar = tmp_path / "werk2"
    nachbar.mkdir()
    (nachbar / "x.json").write_text("{}", encoding="utf-8")
    c = TestClient(create_app(workdir=werk))
    r = c.post("/v1/takeoff", json={"path": str(nachbar / "x.json")})
    assert r.status_code == 403


def test_pfad_unter_link_nach_draussen(client, werk, tmp_path):
    import subprocess
    import sys

    aussen = tmp_path / "aussen"
    aussen.mkdir()
    (aussen / "scene.json").write_text(json.dumps(lade(HAUS_EG)), encoding="utf-8")
    link = werk / "link"
    try:
        link.symlink_to(aussen, target_is_directory=True)
    except (OSError, NotImplementedError):
        if sys.platform != "win32":
            pytest.skip("Symlink nicht anlegbar")
        p = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(aussen)], capture_output=True
        )
        if p.returncode != 0:
            pytest.skip("Junction nicht anlegbar")
    r = client.post("/v1/takeoff", json={"path": "link/scene.json"})
    assert r.status_code == 403


def test_pfad_nicht_gefunden_und_falsche_endung(client, werk):
    r = client.post("/v1/convert", data={"path": "fehlt.dxf"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "input_not_found"
    (werk / "a.txt").write_text("x", encoding="utf-8")
    r = client.post("/v1/convert", data={"path": "a.txt"})
    assert r.status_code == 415


def test_upload_falsche_endung(client):
    r = client.post("/v1/convert", files={"file": ("x.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 415


def test_file_und_path_zugleich_oder_keins(client, demo_dxf):
    with open(demo_dxf, "rb") as f:
        r = client.post("/v1/convert", files={"file": ("demo.dxf", f)}, data={"path": "x.dxf"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_arguments"
    r = client.post("/v1/convert", data={})
    assert r.status_code == 422


def test_upload_groessenlimit(werk):
    c = TestClient(create_app(workdir=werk, max_upload_mb=0.01))  # ca. 10 KB
    r = c.post("/v1/convert", files={"file": ("gross.dxf", b"0\nSECTION\n" * 5000)})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"
    # Ein Upload unterhalb des Limits kommt an (ezdxf liest die leere Datei ohne Layer).
    r = c.post("/v1/layers", files={"file": ("klein.dxf", b"0\nEOF\n")})
    assert r.status_code == 200 and r.json()["summary"]["wall_line_entities"] == 0


def test_ohne_content_length_411(client):
    r = client.post(
        "/v1/takeoff",
        content=iter([b'{"scene": {}}']),
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 411


def test_takeoff_scene_und_pfad(client, werk):
    scene = lade(HAUS_EG)
    r = client.post("/v1/takeoff", json={"scene": scene, "geschoss": "EG"})
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["waende"]["summe"]["anzahl_kanten"] == 8 and m["geschoss"] == "EG"
    assert "elemente" in m
    (werk / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    r2 = client.post(
        "/v1/takeoff", json={"path": "scene.json", "geschoss": "EG", "wall_height_m": 3}
    )
    assert r2.status_code == 200
    a = next(x for x in r2.json()["annahmen"] if x["id"] == "wandhoehe")
    assert a["art"] == "parameter" and a["wert"] == 3.0


def test_takeoff_fehler(client):
    assert client.post("/v1/takeoff", json={}).status_code == 422
    r = client.post("/v1/takeoff", json={"scene": {"a": 1}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "not_a_scene"
    r = client.post("/v1/takeoff", json={"scene": lade(HAUS_EG), "path": "x.json"})
    assert r.status_code == 422
    r = client.post("/v1/takeoff", json={"scene": lade(HAUS_EG), "wall_height_m": 99})
    assert r.status_code == 422  # pydantic-Grenze


def test_kg_proposals_aus_mengen_scene_und_pfad(client, werk):
    scene = lade(HAUS_EG)
    mengen = client.post("/v1/takeoff", json={"scene": scene, "geschoss": "EG"}).json()
    a = client.post("/v1/kg-proposals", json={"mengen": mengen})
    b = client.post("/v1/kg-proposals", json={"scene": scene, "geschoss": "EG"})
    (werk / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    c = client.post("/v1/kg-proposals", json={"path": "scene.json", "geschoss": "EG"})
    assert a.status_code == b.status_code == c.status_code == 200
    assert a.json() == b.json() == c.json()
    doc = a.json()
    p = {x["element_class"]: x for x in doc["proposals"]}
    assert p["Aussenwand tragend"]["kg"] == "331"
    assert p["Innentuer"]["quantity"] == 2
    assert all(x["status"] == "vorschlag" for x in doc["proposals"])
    assert doc["reference_quantities"][0]["kg"] is None


def test_kg_proposals_genau_eine_quelle(client):
    assert client.post("/v1/kg-proposals", json={}).status_code == 422
    r = client.post("/v1/kg-proposals", json={"mengen": {}, "scene": {}})
    assert r.status_code == 422


def test_kg_proposals_kaputte_mengen(client):
    r = client.post("/v1/kg-proposals", json={"mengen": {"schema": "planproof_cad.mengen"}})
    # fehlende Gruppen sind kein Absturz: leere Vorschlagsliste
    assert r.status_code == 200 and r.json()["proposals"] == []
    r = client.post("/v1/kg-proposals", json={"mengen": {"a": 1}})
    assert r.status_code == 422 and r.json()["error"]["code"] in {"not_a_scene", "not_mengen"}


def test_layers_upload_und_pfad(client, werk, demo_dxf):
    with open(demo_dxf, "rb") as f:
        r = client.post("/v1/layers", files={"file": ("demo.dxf", f)})
    assert r.status_code == 200, r.text
    assert r.json()["summary"]["wall_layers"] == ["A_Wand_a", "A_Wand_i"]
    (werk / "d.dxf").write_bytes(demo_dxf.read_bytes())
    r = client.post("/v1/layers", data={"path": "d.dxf"})
    assert r.status_code == 200 and r.json()["summary"]["opening_blocks"] == 6


def test_download_traversal_und_ungueltige_ids(client):
    assert client.get("/v1/runs/nicht-hex/files/plan.ifc").status_code == 400
    assert client.get(f"/v1/runs/{'0' * 32}/files/plan.ifc").status_code == 404
    for name in ("..%2F..%2Fgeheim", "%2e%2e", "a%5Cb"):
        r = client.get(f"/v1/runs/{'0' * 32}/files/{name}")
        assert r.status_code in (400, 404), (name, r.status_code)


def test_download_nur_aus_dem_eigenen_lauf(client, demo_dxf):
    with open(demo_dxf, "rb") as f:
        run_id = client.post("/v1/convert", files={"file": ("demo.dxf", f)}).json()["run_id"]
    assert client.get(f"/v1/runs/{run_id}/files/mengen.json").status_code == 200
    assert client.get(f"/v1/runs/{run_id}/files/_work").status_code in (400, 404)


def test_rest_import_laedt_fastapi_nicht():
    import subprocess
    import sys

    code = (
        "import sys, planproof_cad.rest as r;"
        "assert 'fastapi' not in sys.modules and 'starlette' not in sys.modules, 'fastapi geladen';"
        "print('ok')"
    )
    proc = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_antwort_enthaelt_keine_serverpfade(client, werk, demo_dxf):
    with open(demo_dxf, "rb") as f:
        r = client.post("/v1/convert", files={"file": ("demo.dxf", f)})
    assert r.status_code == 200
    assert str(werk.resolve()) not in r.text
    assert str(werk) not in r.text
