import json
import os
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app import app
from nemotron_review import VERDA_ENDPOINT


class NemotronReviewTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        environment = patch.dict(os.environ, {
            "NEMOTRON_API_KEY": "test-key",
            "NEMOTRON_MODEL": "nemotron-test",
            "PLANPROOF_REVIEW_PROVIDER": "nemotron",
            "GEMINI_ENABLED": "true",
            "GEMINI_API_KEY": "ignored-key",
            "GEMINI_MODEL": "gemini-test",
        })
        environment.start()
        self.addCleanup(environment.stop)
        gemini = patch("app.review_disagreement", side_effect=AssertionError("Unexpected call to Gemini"))
        gemini.start()
        self.addCleanup(gemini.stop)

    def demo(self, *, allow=True, scenario="disputed"):
        response = self.client.post("/api/v1/demo/door-count", params={
            "scenario": scenario, "allow_nemotron_review": allow, "allow_gemini_review": True,
        })
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_nemotron_reviews_the_crop_without_calling_gemini(self):
        def respond(request):
            self.assertEqual(str(request.url), f"{VERDA_ENDPOINT}/v1/chat/completions")
            body = json.loads(request.content)
            media, prompt = body["messages"][1]["content"]
            self.assertTrue(media["image_url"]["url"].startswith("data:image/png;base64,"))
            self.assertEqual(json.loads(prompt["text"])["candidates"][0]["id"], "deterministic:2")
            advisory = {
                "resolution": "SUPPORTS_DETERMINISTIC",
                "reason": "Visible door swing",
                "evidence_ids": ["deterministic:2"],
            }
            return httpx.Response(200, json={"choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(advisory)}},
            ]})

        adapter = httpx.Client(base_url=VERDA_ENDPOINT, transport=httpx.MockTransport(respond))
        with patch("nemotron_review.nemotron_client", return_value=adapter):
            report = self.demo()
        self.assertEqual(report["status"], "NEEDS_REVIEW")
        self.assertIsNone(report["answer"])
        self.assertEqual(report["nemotron"]["status"], "reviewed")
        self.assertEqual(report["nemotron"]["advisory"]["evidence_ids"], ["deterministic:2"])
        self.assertEqual(report["gemini"]["status"], "disabled")

    def test_nemotron_is_skipped_without_consent_or_disagreement(self):
        with patch("nemotron_review.nemotron_client") as client:
            self.assertEqual(self.demo(allow=False)["nemotron"]["status"], "not_authorized")
            self.assertEqual(self.demo(scenario="agreement")["nemotron"]["status"], "not_needed")
            client.assert_not_called()

    def test_a_timeout_preserves_the_analysis(self):
        with patch("nemotron_review.nemotron_client", side_effect=httpx.ReadTimeout("timeout")):
            report = self.demo()
        self.assertEqual(report["status"], "NEEDS_REVIEW")
        self.assertEqual(report["nemotron"]["error_code"], "timeout")
