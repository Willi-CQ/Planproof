import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app
from gemini_review import Advisory, Review
from stoa_assistant import AssistantResponse
from verification import AnalyzerResult, PageFrame


class StoaPipelineTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        for name, value in {
            "assistant_config": ("gemini-test", "test-model"),
            "nemotron_config": ("nemotron-test", "test-model"),
        }.items():
            patcher = patch(f"app.{name}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_coordinator_runs_before_assistant_and_stays_private(self):
        events = []

        def coordinate(crop, bounds, page, comparison, *config):
            events.append("coordinate")
            self.assertTrue(crop.startswith(b"\x89PNG"))
            self.assertEqual(comparison.disputed_evidence[0].id, "deterministic:2")
            return Review(
                status="reviewed", model="private-model", crop_bbox_xyxy=bounds,
                advisory=Advisory(
                    resolution="SUPPORTS_DETERMINISTIC", reason="Arc visible.",
                    evidence_ids=["deterministic:2"],
                ),
            )

        def answer(comparison, review):
            events.append("answer")
            self.assertIsNone(comparison.answer)
            self.assertEqual(review.advisory.evidence_ids, ["deterministic:2"])
            return AssistantResponse(
                status="answered", message="La zone montre un arc d’ouverture.",
                evidence_ids=["deterministic:2"],
            )

        with patch("app.review_with_nemotron", side_effect=coordinate), patch(
            "app.synthesize_assistant", side_effect=answer,
        ), patch("app.review_disagreement") as legacy:
            result = self.client.post("/api/v1/demo/door-count", params={
                "scenario": "disputed", "allow_assistant_review": True,
                "allow_gemini_review": True, "allow_nemotron_review": True,
            })
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(events, ["coordinate", "answer"])
        legacy.assert_not_called()
        payload = result.json()
        self.assertEqual(payload["assistant"]["status"], "answered")
        self.assertEqual(payload["status"], "NEEDS_REVIEW")
        self.assertIsNone(payload["answer"])
        self.assertNotIn("nemotron", payload)
        self.assertNotIn("gemini", payload)
        self.assertNotIn("private-model", result.text)

    def test_consent_and_no_disagreement(self):
        with patch("app.review_with_nemotron") as coordinator, patch(
            "app.synthesize_assistant", return_value=AssistantResponse(status="answered", message="Deux portes."),
        ) as answer:
            refused = self.client.post("/api/v1/demo/door-count", params={
                "allow_assistant_review": False,
            }).json()
            self.assertEqual(refused["assistant"]["status"], "not_authorized")
            answer.assert_not_called()

            agreed = self.client.post("/api/v1/demo/door-count", params={
                "scenario": "agreement", "allow_assistant_review": True,
            }).json()
            self.assertEqual(agreed["answer"], 2)
            self.assertEqual(answer.call_args.args[1].status, "not_needed")
            coordinator.assert_not_called()

    def test_no_writer_configuration_skips_external_crop(self):
        with patch("app.assistant_config", return_value=None), patch(
            "app.review_with_nemotron",
        ) as coordinator, patch(
            "app.synthesize_assistant", return_value=AssistantResponse(status="unavailable"),
        ):
            payload = self.client.post("/api/v1/demo/door-count", params={
                "allow_assistant_review": True,
            }).json()
        self.assertEqual(payload["assistant"]["status"], "unavailable")
        self.assertEqual(payload["status"], "NEEDS_REVIEW")
        coordinator.assert_not_called()

    def test_many_disputes_review_a_local_subset(self):
        demo = self.client.post("/api/v1/demo/door-count").json()
        source = PageFrame.model_validate(demo["source"])
        fixture = {
            "schema_version": "planproof-door-result-v1", "source": source, "version": "test",
            "mode": "independent", "status": "ok", "complete": False,
        }
        deterministic = AnalyzerResult(**fixture)
        ml = AnalyzerResult(**fixture, detections=[
            {"id": str(index), "bbox": [index * 80.0, 390.0, index * 80.0 + 25, 420.0]}
            for index in range(16)
        ])

        def coordinate(crop, bounds, page, localized, *config):
            self.assertEqual(len(localized.disputed_evidence), 4)
            self.assertLess(bounds[2] - bounds[0], source.width / 2)
            self.assertEqual(localized.ml.detected_count, 16)
            return Review(status="reviewed", advisory=Advisory(
                resolution="UNRESOLVED", reason="Seule une ouverture est visible.", evidence_ids=[],
            ))

        with patch("app.run_engines", return_value={"deterministic": deterministic, "ml": ml}), patch(
            "app.review_with_nemotron", side_effect=coordinate,
        ) as coordinator, patch(
            "app.synthesize_assistant", return_value=AssistantResponse(status="answered", message="Échantillon."),
        ) as answer:
            result = self.client.post(
                f"/api/v1/documents/{source.document_id}/analyze-doors",
                params={"allow_assistant_review": True},
            )
        self.assertEqual(result.status_code, 200, result.text)
        coordinator.assert_called_once()
        full_comparison, review = answer.call_args.args
        self.assertEqual(len(full_comparison.disputed_evidence), 16)
        self.assertEqual(review.sampled_candidates, 4)
        payload = result.json()
        self.assertIsNone(payload["answer"])
        self.assertEqual(len(payload["disputed_evidence"]), 16)
