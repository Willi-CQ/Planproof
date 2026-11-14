"""REST-Schnittstelle (Extra ``rest``): ``create_app()`` liefert eine FastAPI-Anwendung.

Endpunkte::

    GET  /healthz
    POST /v1/convert         multipart: file ODER path, optional unit, wall_height_m, geschoss
    POST /v1/takeoff         JSON: scene ODER path, optional geschoss, wall_height_m
    POST /v1/kg-proposals    JSON: mengen ODER scene ODER path, optional geschoss, wall_height_m
    POST /v1/layers          multipart: file ODER path
    GET  /v1/runs/{run_id}/files/{name}   Ergebnisdatei eines /v1/convert-Laufs

Keine Authentifizierung, keine Datenbank, keine Cloud-Teile. Der Standard bindet an 127.0.0.1
(``planproof-cad serve-rest``); wer die Schnittstelle weiter oeffnet, setzt davor einen eigenen
Reverse-Proxy mit Anmeldung.

Pfad-Parameter (``path``) gelten nur innerhalb des Arbeitsverzeichnisses (``workdir``); ``..``
und Symlinks, die hinausfuehren, werden abgewiesen (HTTP 403). Uploads sind auf
``max_upload_mb`` begrenzt (HTTP 413, ohne Content-Length HTTP 411).

``fastapi`` wird erst in ``create_app`` importiert. Dieses Modul zu importieren laedt es nicht.
Dieses Modul hat bewusst kein ``from __future__ import annotations``: FastAPI liest die
Typangaben der Endpunkte zur Laufzeit.
"""

import os
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from planproof_cad import __version__
from planproof_cad.services import (
    DXF_ENDUNGEN,
    ServiceError,
    assign_kg,
    list_layers,
    resolve_inside,
    run_dxf,
    takeoff,
)

_STATUS = {
    "input_not_found": 404,
    "path_outside_root": 403,
    "path_invalid": 400,
    "unsupported_input": 415,
    "out_dir_required": 400,
    "empty_input": 400,
    "payload_too_large": 413,
    "length_required": 411,
}
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_DATEINAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CHUNK = 1024 * 1024


class TakeoffRequest(BaseModel):
    scene: Optional[dict[str, Any]] = Field(None, description="scene-Dict (scene_v1)")
    path: Optional[str] = Field(None, description="scene.json oder .dxf im Arbeitsverzeichnis")
    geschoss: Optional[str] = Field(None, max_length=40)
    wall_height_m: Optional[float] = Field(None, ge=0.5, le=20)


class KgRequest(BaseModel):
    mengen: Optional[dict[str, Any]] = Field(None, description="Mengen-Dict aus /v1/takeoff")
    scene: Optional[dict[str, Any]] = Field(None, description="scene-Dict statt Mengen")
    path: Optional[str] = Field(None, description="mengen.json, scene.json oder .dxf")
    geschoss: Optional[str] = Field(None, max_length=40)
    wall_height_m: Optional[float] = Field(None, ge=0.5, le=20)


def _ohne_serverpfade(metrics: dict[str, Any]) -> dict[str, Any]:
    """Absolute Pfade des Servers (z. B. model_path) gehen nicht an den Client."""
    sauber = dict(metrics)
    if isinstance(sauber.get("model_path"), str):
        sauber["model_path"] = Path(sauber["model_path"]).name
    return sauber


def _ohne_elemente(mengen: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    return None if mengen is None else {k: v for k, v in mengen.items() if k != "elemente"}


def create_app(workdir=None, max_upload_mb: float = 50.0):
    """Baut die FastAPI-Anwendung.

    ``workdir``: Arbeitsverzeichnis fuer Uploads (``uploads/``), Ergebnisse (``runs/``) und
    Pfad-Parameter. Standard ``$CQPLAN_WORKDIR``, sonst ``./planproof-cad-work``.
    """
    from fastapi import FastAPI, File, Form, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse

    wurzel = Path(workdir or os.environ.get("CQPLAN_WORKDIR") or "planproof-cad-work").resolve()
    (wurzel / "uploads").mkdir(parents=True, exist_ok=True)
    (wurzel / "runs").mkdir(parents=True, exist_ok=True)
    max_bytes = int(float(max_upload_mb) * 1024 * 1024)

    app = FastAPI(
        title="planproof-cad",
        version=__version__,
        description="DXF zu BIM, Mengen und Kostengruppen-Vorschlaegen. Lokal, ohne Anmeldung.",
    )
    app.state.workdir = wurzel
    app.state.max_upload_bytes = max_bytes
    sperre = (
        threading.Lock()
    )  # eine Umrechnung zugleich: ifcopenshell/ezdxf sind nicht darauf ausgelegt

    def _fehlerantwort(code: str, message: str, status: Optional[int] = None):
        return JSONResponse(
            status_code=status or _STATUS.get(code, 422),
            content={"error": {"code": code, "message": message}},
        )

    @app.exception_handler(ServiceError)
    async def _service_error(_request: Request, exc: ServiceError):
        return _fehlerantwort(exc.code, exc.message)

    @app.middleware("http")
    async def _groessenlimit(request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            laenge = request.headers.get("content-length")
            if laenge is None:
                return _fehlerantwort("length_required", "Content-Length fehlt.")
            try:
                n = int(laenge)
            except ValueError:
                return _fehlerantwort("length_required", "Content-Length ungültig.", 400)
            if n > max_bytes:
                return _fehlerantwort(
                    "payload_too_large",
                    f"Anfrage größer als {max_upload_mb:g} MB.",
                )
        return await call_next(request)

    def _neuer_lauf() -> tuple[str, Path]:
        run_id = uuid.uuid4().hex
        ordner = wurzel / "runs" / run_id
        ordner.mkdir(parents=True)
        return run_id, ordner

    def _upload_speichern(datei) -> Path:
        name = Path(datei.filename or "").name
        sauber = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:100] or "upload"
        if Path(sauber).suffix.lower() not in DXF_ENDUNGEN:
            raise ServiceError("unsupported_input", "Erwartet eine .dxf- oder .dwg-Datei.")
        ordner = wurzel / "uploads" / uuid.uuid4().hex
        ordner.mkdir(parents=True)
        ziel = ordner / sauber
        geschrieben = 0
        with open(ziel, "wb") as out:
            while True:
                block = datei.file.read(_CHUNK)
                if not block:
                    break
                geschrieben += len(block)
                if geschrieben > max_bytes:
                    out.close()
                    shutil.rmtree(ordner, ignore_errors=True)
                    raise ServiceError(
                        "payload_too_large", f"Datei größer als {max_upload_mb:g} MB."
                    )
                out.write(block)
        return ziel

    def _eingabe_datei(datei, pfad: Optional[str]) -> Path:
        if (datei is None) == (pfad is None):
            raise ServiceError("invalid_arguments", "Genau eines von file oder path angeben.")
        if datei is not None:
            return _upload_speichern(datei)
        return resolve_inside(pfad, [wurzel])

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "version": __version__}

    @app.post("/v1/convert")
    def convert(
        file: Optional[UploadFile] = File(None),  # noqa: B008
        path: Optional[str] = Form(None),
        unit: Optional[str] = Form(None),
        wall_height_m: Optional[float] = Form(None),
        geschoss: Optional[str] = Form(None),
    ):
        eingabe = _eingabe_datei(file, path)
        if unit is not None and unit not in ("mm", "cm", "m", "in", "ft"):
            raise ServiceError("invalid_arguments", "unit muss mm, cm, m, in oder ft sein.")
        config: dict[str, Any] = {}
        if unit:
            config["dxf_unit"] = unit
        if wall_height_m is not None:
            config["wall_height_m"] = wall_height_m
        run_id, ordner = _neuer_lauf()
        with sperre:
            res = run_dxf(eingabe, ordner, config, geschoss=geschoss)
        namen = sorted(p.name for p in ordner.iterdir() if p.is_file())
        antwort = {
            "ok": res["ok"],
            "error": res["error"],
            "run_id": run_id,
            "input": res["input"],
            "files": [{"name": n, "url": f"/v1/runs/{run_id}/files/{n}"} for n in namen],
            "warnings": res["warnings"],
            "template_assignment": res["template_assignment"],
            "metrics": _ohne_serverpfade(res["metrics"]),
            "mengen": _ohne_elemente(res["mengen"]),
            "kg_vorschlaege": res["kg_vorschlaege"],
        }
        return JSONResponse(status_code=200 if res["ok"] else 422, content=antwort)

    @app.get("/v1/runs/{run_id}/files/{name}")
    def lauf_datei(run_id: str, name: str):
        if not _RUN_ID.match(run_id) or not _DATEINAME.match(name):
            raise ServiceError("path_invalid", "Ungültige Lauf-ID oder Dateiname.")
        ziel = resolve_inside(name, [wurzel / "runs" / run_id])
        if not ziel.is_file():
            raise ServiceError("input_not_found", "Datei nicht gefunden.")
        return FileResponse(ziel, filename=name)

    @app.post("/v1/takeoff")
    def takeoff_endpunkt(req: TakeoffRequest):
        if (req.scene is None) == (req.path is None):
            raise ServiceError("invalid_arguments", "Genau eines von scene oder path angeben.")
        run_id = None
        if req.scene is not None:
            mengen = takeoff(req.scene, geschoss=req.geschoss, wall_height_m=req.wall_height_m)
        else:
            quelle = resolve_inside(req.path, [wurzel])
            arbeit = None
            if quelle.suffix.lower() in DXF_ENDUNGEN:
                run_id, arbeit = _neuer_lauf()
            with sperre:
                mengen = takeoff(
                    quelle,
                    geschoss=req.geschoss,
                    wall_height_m=req.wall_height_m,
                    work_dir=arbeit,
                )
        antwort = dict(mengen)
        if run_id:
            antwort["run_id"] = run_id
        return antwort

    @app.post("/v1/kg-proposals")
    def kg_endpunkt(req: KgRequest):
        gegeben = [x for x in (req.mengen, req.scene, req.path) if x is not None]
        if len(gegeben) != 1:
            raise ServiceError(
                "invalid_arguments", "Genau eines von mengen, scene oder path angeben."
            )
        if req.mengen is not None:
            quelle: Any = req.mengen
            arbeit = None
        elif req.scene is not None:
            quelle = req.scene
            arbeit = None
        else:
            quelle = resolve_inside(req.path, [wurzel])
            arbeit = None
            if quelle.suffix.lower() in DXF_ENDUNGEN:
                _run_id, arbeit = _neuer_lauf()
        with sperre:
            return assign_kg(
                quelle, geschoss=req.geschoss, wall_height_m=req.wall_height_m, work_dir=arbeit
            )

    @app.post("/v1/layers")
    def layers(
        file: Optional[UploadFile] = File(None),  # noqa: B008
        path: Optional[str] = Form(None),
    ):
        eingabe = _eingabe_datei(file, path)
        return list_layers(eingabe, work_dir=wurzel / "uploads" / "_dwg_work")

    return app


__all__ = ["KgRequest", "TakeoffRequest", "create_app"]
