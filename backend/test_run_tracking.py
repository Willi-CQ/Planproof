import json
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from run_tracking import RunStore, evidence_bundle, public_event

SOURCE = {"raster_sha256": "raster", "page_index": 3, "width": 100, "height": 80}


def finished(store, run_id):
    for _ in range(200):
        value = store.snapshot(run_id)
        if value["status"] in {"completed", "failed", "interrupted"}:
            return value
        time.sleep(0.01)
    raise AssertionError("Run did not finish")


def test_idempotency_delta_and_canonical_events(tmp_path):
    store = RunStore(tmp_path)
    gate = threading.Event()
    calls = []
    run_id = str(uuid4())

    def execute(observe):
        calls.append(1)
        observe(
            "crop_region",
            "completed",
            images=[
                {
                    "image_id": "crop-1",
                    "canonical_bbox": [2, 3, 20, 30],
                    "page_index": 0,
                    "path": "private",
                }
            ],
        )
        gate.wait(2)
        return {"status": "ok", "answer": {"answer": 2}}

    first = store.submit(run_id, SOURCE, {"question": "Doors?"}, execute)
    second = store.submit(run_id, SOURCE, {"question": "Doors?"}, execute)
    assert first["request_id"] == second["request_id"]
    with pytest.raises(ValueError):
        store.submit(run_id, SOURCE, {"question": "Windows?"}, execute)
    gate.set()
    value = finished(store, run_id)
    store.pool.shutdown(wait=True)
    assert len(calls) == 1
    value = store.snapshot(run_id)
    assert [e["sequence"] for e in value["events"]] == list(
        range(1, len(value["events"]) + 1)
    )
    crop = next(e for e in value["events"] if e["stage"] == "crop_region")
    assert crop["images"][0]["page_index"] == 3
    assert "private" not in json.dumps(value)
    assert all(e["sequence"] > 1 for e in store.snapshot(run_id, 1)["events"])
    restarted = RunStore(tmp_path)
    assert restarted.snapshot(run_id)["question"] == "Doors?"
    assert restarted.snapshot(run_id)["result"]["answer"]["answer"] == 2
    restarted.pool.shutdown()


def test_restart_marks_unfinished_and_does_not_execute(tmp_path):
    run_id = str(uuid4())
    (tmp_path / (run_id + ".json")).write_text(
        json.dumps(
            {
                "request_id": run_id,
                "status": "running",
                "events": [],
                "result": None,
                "source": SOURCE,
                "evidence": {},
                "fingerprint": "x",
            }
        )
    )
    store = RunStore(tmp_path)
    assert store.snapshot(run_id)["status"] == "interrupted"
    assert "not resubmitted" in store.snapshot(run_id)["message"]
    store.pool.shutdown()


def test_bounded_queue_failure_and_observer_disk_error(tmp_path, monkeypatch):
    store = RunStore(tmp_path, workers=1, waiting=0)
    gate = threading.Event()
    started = threading.Event()
    run_id = str(uuid4())

    def execute(observe):
        started.set()
        gate.wait(2)
        observe("reflect", "completed", checks=["Check bounds"], secret="never expose")
        return {"status": "ok"}

    store.submit(run_id, SOURCE, {}, execute)
    assert started.wait(2)
    with pytest.raises(OverflowError):
        store.submit(str(uuid4()), SOURCE, {}, execute)
    monkeypatch.setattr(
        store, "_persist", lambda value: (_ for _ in ()).throw(OSError("disk full"))
    )
    gate.set()
    assert finished(store, run_id)["result"]["status"] == "ok"
    store.pool.shutdown()
    assert "telemetry_warning" in store.snapshot(run_id)


def test_sanitized_bundle_separates_hybrid_and_is_read_only(tmp_path):
    job = SimpleNamespace(
        document={
            "observations": [
                {
                    "id": "m",
                    "class": "wall",
                    "bbox": [1, 1, 30, 10],
                    "source_family": "ml",
                    "private_path": "hidden",
                    "features": {
                        "parallel_line_count": 4,
                        "secret_path": "never expose",
                    },
                },
                {
                    "id": "d",
                    "class": "wall",
                    "bbox": [1, 1, 30, 10],
                    "source_family": "classical",
                },
            ],
            "comparisons": [
                {
                    "relation": "agreement",
                    "source_ids": ["d", "m"],
                    "bbox": [1, 1, 30, 10],
                    "iou": 1,
                }
            ],
        },
        page={"rooms": [{"id": "r", "bbox": [5, 5, 50, 60], "status": "ml_assisted"}]},
        catalog={"warnings": ["Candidates only"]},
    )
    bundle = evidence_bundle(job, SOURCE)
    assert [s["layer"] for s in bundle["shapes"]] == ["ml", "deterministic", "hybrid"]
    assert "hidden" not in json.dumps(bundle)
    assert "never expose" not in json.dumps(bundle)
    assert bundle["shapes"][0]["cues"] == [{"name": "parallel_line_count", "value": 4}]
    safe = public_event(
        "reflect",
        "completed",
        {
            "draft": "not public",
            "checks": ["bounds"],
            "images": [{"canonical_bbox": [-1, 0, 20, 20]}],
        },
        SOURCE,
    )
    assert "draft" not in safe and safe["images"] == []
    store = RunStore(tmp_path)
    run_id = str(uuid4())
    store.submit(run_id, SOURCE, {}, lambda observer: {"status": "ok"})
    finished(store, run_id)
    assert store.evidence(run_id)["status"] == "not_run"
    store.pool.shutdown()


def test_run_routes_use_same_agent_request_and_polling_never_executes(
    tmp_path, monkeypatch
):
    import io
    from fastapi.testclient import TestClient
    from PIL import Image
    import app as module

    store = RunStore(tmp_path)
    monkeypatch.setattr(module, "_run_store", store)
    calls = []

    def fake_ask(document_id, page, request, *, observer):
        calls.append(request.model_dump())
        observer("reflect", "completed", checks=["Read the label"])
        return {
            "status": "ok",
            "answer": {"status": "answered", "answer": 2},
            "canonical_citations": [],
        }

    monkeypatch.setattr(module, "ask", fake_ask)
    with TestClient(module.app) as client:
        stream = io.BytesIO()
        Image.new("RGB", (100, 80), "white").save(stream, format="PNG")
        drawing = client.post(
            "/api/documents",
            files={"file": ("page.png", stream.getvalue(), "image/png")},
        ).json()
        request_id = str(uuid4())
        body = {
            "request_id": request_id,
            "request": {
                "question": "Doors?",
                "mode": "auto",
                "allow_gemini": True,
                "allow_nemotron": True,
            },
        }
        endpoint = f"/api/v1/documents/{drawing['document_id']}/runs"
        assert client.post(endpoint, json=body).status_code == 202
        finished(store, request_id)
        assert client.post(endpoint, json=body).status_code == 202
        assert (
            client.get(f"/api/v1/runs/{request_id}").json()["result"]["source"]["width"]
            == 100
        )
        assert client.get(f"/api/v1/runs/{request_id}/evidence").status_code == 200
        assert len(calls) == 1
        assert client.get("/api/v1/runs/not-a-uuid").status_code == 422
    store.pool.shutdown()
