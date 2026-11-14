import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

import agent_pipeline
from app import app


class AgentPipelineTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        agent_pipeline._jobs.clear()
        stream = io.BytesIO()
        Image.new("RGB", (40, 30), "white").save(stream, format="PNG")
        self.doc = self.client.post(
            "/api/documents",
            files={"file": ("page.png", stream.getvalue(), "image/png")},
        ).json()

    def test_unknown_document_and_blank_question_are_rejected(self):
        self.assertEqual(
            self.client.post(
                "/api/v1/documents/missing/reason", json={"question": "How many doors?"}
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.post(
                f"/api/v1/documents/{self.doc['document_id']}/reason",
                json={"question": "   "},
            ).status_code,
            422,
        )

    def test_algorithm_failure_does_not_produce_an_answer(self):
        with patch("agent_pipeline.EvidencePipeline") as factory:
            factory.return_value.prepare.side_effect = ValueError("wrong raster")
            factory.return_value.settings.work_root = (
                Path(app.root_path or ".") / ".artifacts/test-agent"
            )
            response = self.client.post(
                f"/api/v1/documents/{self.doc['document_id']}/reason",
                json={"question": "Doors?"},
            )
        self.assertEqual(response.status_code, 503)

    def test_permission_gate_preserves_evidence_without_provider_call(self):
        agent_pipeline._jobs[self.doc["document_id"]] = SimpleNamespace(
            directory=Path("registered-job"), catalog={"rooms": []}
        )
        with patch("agent_pipeline.create_service") as factory:
            response = self.client.post(
                f"/api/v1/documents/{self.doc['document_id']}/reason",
                json={"question": "Doors?"},
            )
        self.assertEqual(response.json()["status"], "not_authorized")
        self.assertIsNone(response.json()["answer"])
        factory.assert_not_called()

    def test_guided_trace_is_returned_with_the_uploaded_page_frame(self):
        job = SimpleNamespace(
            directory=Path(".artifacts/test-agent/registered-job"),
            catalog={"rooms": []},
        )
        agent_pipeline._jobs[self.doc["document_id"]] = job
        trace = {
            "final": {
                "status": "ok",
                "result": {"status": "answered", "answer": 2},
                "canonical_citations": [
                    {"page_index": 0, "canonical_bbox": [2, 3, 8, 9]}
                ],
            },
            "decision": {"region_ids": [], "action": "answer_overview"},
            "usage": {"total_tokens": 20},
        }
        with (
            patch("agent_pipeline.create_service", return_value=Mock()),
            patch("agent_pipeline.answer_guided", return_value=trace) as run,
        ):
            response = self.client.post(
                f"/api/v1/documents/{self.doc['document_id']}/reason",
                json={"question": "Doors?", "allow_nemotron": True},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"]["answer"], 2)
        self.assertEqual(
            response.json()["source"]["document_id"], self.doc["document_id"]
        )
        self.assertEqual(
            response.json()["canonical_citations"][0]["canonical_bbox"], [2, 3, 8, 9]
        )
        self.assertEqual(run.call_args.args[1], job)
        self.assertIn("questions", str(run.call_args.kwargs["out"]))

    def test_missing_provider_credentials_preserve_algorithm_evidence(self):
        agent_pipeline._jobs[self.doc["document_id"]] = SimpleNamespace(
            directory=Path("registered-job"), catalog={"rooms": []}
        )
        with patch(
            "agent_pipeline.create_service",
            side_effect=ValueError("a private key must never be echoed"),
        ):
            response = self.client.post(
                f"/api/v1/documents/{self.doc['document_id']}/reason",
                json={"question": "Doors?", "allow_nemotron": True},
            )
        self.assertEqual(response.json()["status"], "unavailable")
        self.assertNotIn("private key", response.text)
        self.assertIsNone(response.json()["answer"])
