"""Synthetic saved runs exercise export without any paid inference."""

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from unittest.mock import Mock
from uuid import uuid4

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import app as module
from run_tracking import RunStore
from task_report import task_report


@pytest.fixture
def sample():
    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 450, 400), outline="black", width=5)
    draw.text((140, 160), "SYNTHETIC TEST / Room A", fill="black")
    stream = BytesIO()
    image.save(stream, format="PNG")
    png = stream.getvalue()
    source = {
        "raster_sha256": sha256(png).hexdigest(),
        "page_index": 3,
        "width": 1000,
        "height": 700,
    }
    region = {"canonical_bbox": [95, 95, 460, 410], "page_index": 3}
    record = {
        "request_id": str(uuid4()),
        "created": 1794614400,
        "status": "completed",
        "source": source,
        "question": "What is the room area?",
        "result": {
            "status": "ok",
            "source": source,
            "measurement": {"approximate": True},
            "answer": {
                "status": "answered",
                "answer": 12,
                "unit": "m²",
                "coverage": "Room A only.",
                "briefing": [
                    {
                        "text": "The room is approximately twelve square metres.",
                        "citations": [1],
                    }
                ],
                "citation_details": [
                    {"citation": 1, "kind": "measurement", "label": "Room boundary"}
                ],
                "uncertainties": ["The inner wall face is partly obscured."],
            },
            "canonical_citations": [region],
            "tool_trace": [{"private": "secret-sentinel"}],
        },
        "events": [
            {
                "sequence": 1,
                "stage": "inspect_nemotron",
                "state": "completed",
                "elapsed": 2.5,
                "question": "Is the boundary visible?",
                "images": [region],
                "advisory": {"findings": ["The south wall is partly obscured."]},
                "private": "secret-sentinel",
            }
        ],
    }
    return png, record


def pdf_text(data):
    with pymupdf.open(stream=data, filetype="pdf") as document:
        return "\n".join(page.get_text() for page in document)


def test_portable_report_contains_exact_question_answer_linked_crops_and_steps(
    sample, tmp_path
):
    png, record = sample
    data = task_report(record, png, "WRONG QUESTION", "Sample drawing.png")
    with pymupdf.open(stream=data, filetype="pdf") as document:
        assert len(document) >= 3
        contents = "\n".join(page.get_text() for page in document)
        for expected in (
            record["question"],
            "12 m²",
            "Page 4",
            "Room A only.",
            "Room boundary",
            "95, 95, 460, 410",
            "Is the boundary visible?",
            "The south wall is partly obscured.",
        ):
            assert expected in contents
        assert "WRONG QUESTION" not in contents
        assert "secret-sentinel" not in contents
        assert sum(len(page.get_images()) for page in document) >= 2
        links = [link for page in document for link in page.get_links()]
        assert links and all(link["kind"] == pymupdf.LINK_GOTO for link in links)
        assert all(0 <= link["page"] < len(document) for link in links)
        document[0].get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3)).save(
            tmp_path / "report-preview.png"
        )
    (tmp_path / "report.pdf").write_bytes(data)


def test_escape_content_preserve_zero_legacy_question_and_subpixel_citation(sample):
    png, record = sample
    record.pop("question")
    record["result"]["answer"]["answer"] = 0
    record["result"]["canonical_citations"][0]["canonical_bbox"] = [100.1, 100.2, 100.4, 100.8]
    question = '<script>unsafe()</script> <a href="https://example.com">Question</a>'
    data = task_report(record, png, question, "Legacy.png")
    assert "0 m²" in pdf_text(data)
    assert "<script>unsafe()</script>" in pdf_text(data)
    with pymupdf.open(stream=data, filetype="pdf") as document:
        assert not any(
            link.get("uri") for page in document for link in page.get_links()
        )


@pytest.mark.parametrize("status", ["failed", "interrupted", "abstain"])
def test_incomplete_reports_do_not_fabricate_an_answer(sample, status):
    png, record = sample
    if status == "abstain":
        record["result"]["answer"].update(status="abstain", answer=None)
    else:
        record.update(status=status, result=None)
    contents = pdf_text(task_report(record, png, "", "Sample.png"))
    assert "Evidence incomplete" in contents
    assert "0 m²" not in contents
    assert "Is the boundary visible?" in contents


def test_pagination_and_result_source_binding(sample):
    png, record = sample
    record["result"]["source"] = {**record["source"], "raster_sha256": "wrong"}
    record["events"] *= 45
    record["events"][-1] = {
        **record["events"][-1],
        "summary": "Last recorded event marker.",
    }
    data = task_report(record, png, "", "Sample.png")
    contents = pdf_text(data)
    assert "No source-bound visual citations" in contents
    assert "[1] Room boundary" not in contents
    assert "Last recorded event marker." in contents
    with pymupdf.open(stream=data, filetype="pdf") as document:
        assert len(document) > 5


def test_export_route_validates_run_and_frame_and_never_executes(
    sample, tmp_path, monkeypatch
):
    png, record = sample
    store = RunStore(tmp_path / "runs")
    store.runs[record["request_id"]] = {**deepcopy(record), "evidence": {}}
    monkeypatch.setattr(module, "_run_store", store)
    execute = Mock(side_effect=AssertionError("Report must not call the agent"))
    monkeypatch.setattr(module, "ask", execute)
    registered = module.store_page(
        module.Page(png, {}, 1000, 700, "a" * 64, record["source"]["raster_sha256"]),
        4,
    )
    body = {
        "document_id": registered.document_id,
        "question": "wrong",
        "filename": "Test.png",
    }
    endpoint = f"/api/v1/runs/{record['request_id']}/report"
    with TestClient(module.app) as client:
        for _ in range(2):
            response = client.post(endpoint, json=body)
            assert response.status_code == 200, response.text[:200]
            assert response.headers["content-type"] == "application/pdf"
            assert response.headers["cache-control"] == "no-store"
            assert "attachment" in response.headers["content-disposition"]
            assert record["question"] in pdf_text(response.content)
        assert (
            client.post(endpoint, json={**body, "document_id": "missing"}).status_code
            == 404
        )
        assert (
            client.post(f"/api/v1/runs/{uuid4()}/report", json=body).status_code == 404
        )
        store.runs[record["request_id"]]["status"] = "running"
        assert client.post(endpoint, json=body).status_code == 409
        store.runs[record["request_id"]]["status"] = "completed"
        store.runs[record["request_id"]]["source"] = {
            **record["source"],
            "page_index": 0,
        }
        assert client.post(endpoint, json=body).status_code == 409
        assert (
            client.post(endpoint, json={**body, "answer": "injected"}).status_code
            == 422
        )
    execute.assert_not_called()
    store.pool.shutdown()
