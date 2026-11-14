import base64
import json
import os
import unittest
from hashlib import sha256
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient
from google.genai import Client as GeminiClient, errors, types
from httpx import Client, MockTransport, ReadTimeout, Response

from app import app
from check_gemini import DEFAULT_MODEL
from ml_adapter import adapt_ml
from verification import PageFrame


class DoorCountTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.environment = patch.dict(os.environ, {
            "GEMINI_ENABLED": "true", "GEMINI_API_KEY": "test-key", "GEMINI_MODEL": DEFAULT_MODEL,
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.sdk = patch("gemini_review.genai.Client")
        self.generate = self.sdk.start().return_value.__enter__.return_value.models.generate_content
        self.addCleanup(self.sdk.stop)

    def demo(self, scenario="disputed", allow=True):
        response = self.client.post("/api/v1/demo/door-count", params={
            "scenario": scenario, "allow_gemini_review": allow,
        })
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def result(self, source, boxes, *, complete=True, mode="independent"):
        return {
            "schema_version": "planproof-door-result-v1", "source": source,
            "version": "test", "status": "ok", "mode": mode, "complete": complete,
            "detections": [{"id": str(index), "bbox": box} for index, box in enumerate(boxes)],
        }

    def test_agreement_and_missing_engine_skip_gemini(self):
        agreed = self.demo("agreement")
        self.assertEqual((agreed["status"], agreed["answer"]), ("SUPPORTED", 2))
        self.assertEqual(agreed["mode"], "synthetic_demo")
        self.assertEqual(len(agreed["matched_evidence"]), 4)
        missing = self.demo("missing_ml")
        self.assertEqual(missing["status"], "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(missing["ml"]["detected_count"])
        self.generate.assert_not_called()

    def test_dispute_sends_actual_crop_and_keeps_advisory_separate(self):
        self.set_advisory(["deterministic:2"])
        report = self.demo()
        self.assertEqual(report["status"], "NEEDS_REVIEW")
        self.assertIsNone(report["answer"])
        self.assertEqual(report["gemini"]["status"], "reviewed")
        self.generate.assert_called_once()
        call = self.generate.call_args.kwargs
        prompt, part = call["contents"]
        self.assertNotIn("ml:1", prompt)
        left, top, right, bottom = report["gemini"]["crop_bbox_xyxy"]
        raw = self.client.get(report["image_url"]).content
        original = pymupdf.Pixmap(raw)
        self.assertEqual(sha256(original.samples).hexdigest(), report["source"]["raster_sha256"])
        crop = pymupdf.Pixmap(part.inline_data.data)
        self.assertLess(crop.width, original.width)
        self.assertEqual((crop.width, crop.height), (right - left, bottom - top))
        self.assertEqual(crop.pixel(0, 0), original.pixel(left, top))
        self.assertEqual(json.loads(prompt)["candidates"][0]["id"], "deterministic:2")

    def set_advisory(self, ids):
        self.generate.return_value = types.GenerateContentResponse(candidates=[
            types.Candidate(
                finish_reason="STOP",
                content=types.Content(parts=[types.Part.from_text(text=json.dumps({
                    "resolution": "SUPPORTS_DETERMINISTIC", "reason": "A door swing is visible.",
                    "evidence_ids": ids,
                }))]),
            )
        ])

    def test_unknown_citation_and_timeout_preserve_review(self):
        self.set_advisory(["made-up"])
        failures = (
            (None, "invalid_response"),
            (ReadTimeout("timed out"), "timeout"),
            (errors.ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED"}}), "quota"),
        )
        for failure, code in failures:
            with self.subTest(code=code):
                self.generate.side_effect = failure
                report = self.demo()
                self.assertEqual(report["status"], "NEEDS_REVIEW")
                self.assertEqual(report["gemini"]["status"], "unavailable")
                self.assertEqual(report["gemini"]["error_code"], code)
                self.assertIsNone(report["answer"])

    def test_actual_sdk_serializes_png_and_response_schema(self):
        requests = []

        def handle(request):
            requests.append(request)
            return Response(200, json={"candidates": [{
                "content": {"role": "model", "parts": [{"text": json.dumps({
                    "resolution": "UNRESOLVED", "reason": "The symbol is ambiguous.",
                    "evidence_ids": ["deterministic:2"],
                })}]},
                "finishReason": "STOP",
            }]})

        with Client(transport=MockTransport(handle)) as transport:
            def local_client(**kwargs):
                kwargs["http_options"].httpx_client = transport
                return GeminiClient(**kwargs)

            with patch("gemini_review.genai.Client", side_effect=local_client):
                report = self.demo()
        self.assertEqual(report["gemini"]["advisory"]["resolution"], "UNRESOLVED")
        self.assertEqual(len(requests), 1)
        self.assertTrue(requests[0].url.path.endswith(f"/models/{DEFAULT_MODEL}:generateContent"))
        data = json.loads(requests[0].content)
        inline = data["contents"][0]["parts"][1]["inlineData"]
        self.assertEqual(inline.get("mimeType", inline.get("mime_type")), "image/png")
        self.assertTrue(base64.urlsafe_b64decode(inline["data"]).startswith(b"\x89PNG"))
        self.assertIn("responseJsonSchema", data["generationConfig"])

    def test_opt_in_and_disabled_switch(self):
        self.assertEqual(self.demo(allow=False)["gemini"]["status"], "not_authorized")
        with patch.dict(os.environ, {"GEMINI_ENABLED": "false"}):
            self.assertEqual(self.demo()["gemini"]["status"], "disabled")
        self.generate.assert_not_called()

    def test_same_count_different_objects_and_incomplete_coverage(self):
        source = self.demo("agreement")["source"]
        self.assertEqual(self.client.get(f"/api/v1/documents/{source['document_id']}/frame").json(), source)
        body = {
            "document_id": source["document_id"],
            "deterministic": self.result(source, [[10, 10, 25, 25]]),
            "ml": self.result(source, [[70, 70, 90, 90]]),
        }
        dispute = self.client.post("/api/v1/door-count", json=body).json()
        self.assertEqual(dispute["status"], "NEEDS_REVIEW")
        self.assertEqual(len(dispute["disputed_evidence"]), 2)
        body["ml"] = body["deterministic"] = self.result(source, [], complete=False)
        report = self.client.post("/api/v1/door-count", json=body).json()
        self.assertEqual(report["status"], "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(report["answer"])

    def test_wrong_raster_and_out_of_bounds_get_422(self):
        source = self.demo("agreement")["source"]
        for result in (
            self.result({**source, "raster_sha256": "0" * 64}, []),
            self.result(source, [[0, 0, source["width"] + 1, 100]]),
        ):
            body = {"document_id": source["document_id"], "ml": result}
            self.assertEqual(self.client.post("/api/v1/door-count", json=body).status_code, 422)

    def test_possible_duplicates_cannot_confirm_a_count(self):
        source = self.demo("agreement")["source"]
        overlap = [[10, 10, 50, 50], [12, 12, 52, 52]]
        body = {
            "document_id": source["document_id"],
            "deterministic": self.result(source, overlap),
            "ml": self.result(source, overlap),
        }
        report = self.client.post("/api/v1/door-count", json=body).json()
        self.assertEqual(report["status"], "NEEDS_REVIEW")
        self.assertEqual(report["reasons"], ["ambiguous_matches"])
        self.assertIsNone(report["answer"])

    def test_hybrid_result_cannot_confirm_independence(self):
        source = self.demo("agreement")["source"]
        body = {
            "document_id": source["document_id"], "allow_gemini_review": True,
            "deterministic": self.result(source, [[10, 10, 25, 25]], mode="hybrid"),
            "ml": self.result(source, [[10, 10, 25, 25]]),
        }
        report = self.client.post("/api/v1/door-count", json=body).json()
        self.assertEqual(report["reasons"], ["non_independent"])
        self.generate.assert_not_called()

    def test_willi_output_binds_rgb_raster_and_preserves_unknown_coverage(self):
        frame = PageFrame.model_validate(self.demo("agreement")["source"])
        raw = {
            "schema_version": "planproof-detection-v1", "document_id": frame.raster_sha256,
            "page_index": frame.page_index, "image_size": [frame.width, frame.height],
            "coordinate_system": frame.coordinate_system, "model_version": "test-release",
            "instances": [
                {"id": "door-4", "class": "door", "bbox": [10, 20, 40, 50], "confidence": 0.8},
                {"id": "wall-1", "class": "wall", "bbox": [0, 0, 10, 80], "confidence": 0.7},
            ],
        }
        result = adapt_ml(raw, frame)
        self.assertFalse(result.complete)
        self.assertEqual([door.id for door in result.detections], ["door-4"])
        with self.assertRaisesRegex(ValueError, "raster hash"):
            adapt_ml({**raw, "document_id": "f" * 64}, frame)


if __name__ == "__main__":
    unittest.main()
