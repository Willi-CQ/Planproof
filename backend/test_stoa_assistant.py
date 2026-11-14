import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from httpx import ReadTimeout

from gemini_review import Advisory, Review
from stoa_assistant import synthesize_assistant
from verification import AnalyzerSummary, Comparison


class StoaAssistantTests(unittest.TestCase):
    def setUp(self):
        summary = AnalyzerSummary(status="ok", version=None, detected_count=1, complete=True)
        self.comparison = Comparison(
            status="NEEDS_REVIEW", reasons=["unmatched_detections"], deterministic=summary, ml=summary,
        )
        self.review = Review(
            status="reviewed",
            model="internal-provider-model",
            crop_bbox_xyxy=[0, 0, 20, 20],
            advisory=Advisory(
                resolution="SUPPORTS_DETERMINISTIC",
                reason="A door swing is visible.",
                evidence_ids=["deterministic:2"],
            ),
        )
        config = patch("stoa_assistant.assistant_config", return_value=("test-key", "test-model"))
        self.config = config.start()
        self.addCleanup(config.stop)
        sdk = patch("stoa_assistant.genai.Client")
        self.generate = sdk.start().return_value.__enter__.return_value.models.generate_content
        self.addCleanup(sdk.stop)

    def draft(self, message, evidence_ids, finish_reason="STOP"):
        self.generate.return_value = SimpleNamespace(
            candidates=[SimpleNamespace(finish_reason=finish_reason)],
            text=json.dumps({"message": message, "evidence_ids": evidence_ids}),
        )

    def test_uses_validated_review_and_keeps_the_count_open(self):
        self.draft("Un arc d’ouverture est visible dans la zone.", ["deterministic:2"])
        result = synthesize_assistant(self.comparison, self.review)
        self.assertEqual(result.status, "answered")
        self.assertTrue(result.message.startswith("Le total de portes reste à vérifier."))
        self.assertEqual(result.evidence_ids, ["deterministic:2"])
        context = json.loads(self.generate.call_args.kwargs["contents"])
        self.assertEqual(context, {
            "status": "NEEDS_REVIEW",
            "reasons": ["unmatched_detections"],
            "visual_review": {
                "resolution": "SUPPORTS_DETERMINISTIC",
                "observation": "A door swing is visible.",
                "evidence_ids": ["deterministic:2"],
            },
        })

    def test_rejects_invalid_or_missing_citations_and_provider_names(self):
        for message, ids in (
            ("L’arc est visible.", ["deterministic:7"]),
            ("L’arc est visible.", ["deterministic:2", "deterministic:2"]),
            ("L’arc est visible.", []),
            ("Nemotron a vu un arc.", ["deterministic:2"]),
        ):
            with self.subTest(message=message, ids=ids):
                self.draft(message, ids)
                result = synthesize_assistant(self.comparison, self.review)
                self.assertEqual(result.status, "unavailable")
                self.assertEqual(result.evidence_ids, [])

    def test_can_explain_disagreement_without_visual_review(self):
        self.draft("Les détections des deux moteurs divergent.", [])
        result = synthesize_assistant(
            self.comparison, Review(status="unavailable", detail="internal-provider-detail"),
        )
        self.assertEqual(result.status, "answered")
        context = json.loads(self.generate.call_args.kwargs["contents"])
        self.assertIsNone(context["visual_review"])
        self.assertNotIn("internal-provider-detail", json.dumps(context))

    def test_preserves_the_planproof_result_when_gemini_times_out(self):
        supported = self.comparison.model_copy(update={"status": "SUPPORTED", "answer": 1})
        self.generate.side_effect = ReadTimeout("internal-provider-detail")
        result = synthesize_assistant(supported, Review(status="not_needed"))
        self.assertEqual(result.status, "unavailable")
        self.assertTrue(result.message.startswith("1 porte confirmée sur cette page."))
        self.assertNotIn("internal-provider-detail", result.message)

    def test_requires_server_configuration(self):
        self.config.return_value = None
        result = synthesize_assistant(self.comparison, self.review)
        self.assertEqual(result.status, "unavailable")
        self.generate.assert_not_called()
