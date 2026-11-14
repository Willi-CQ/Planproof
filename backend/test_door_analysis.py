import importlib.util
import sys
import types
import unittest
from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import deterministic_adapter
from app import app
from deterministic_adapter import ENGINE_SOURCE, run_deterministic, unavailable
from verification import AnalyzerResult, PageFrame


if ENGINE_SOURCE.is_dir() and str(ENGINE_SOURCE) not in sys.path:
    sys.path.insert(0, str(ENGINE_SOURCE))
try:
    ENGINE_INSTALLED = importlib.util.find_spec("planproof_evidence") is not None and all(
        importlib.util.find_spec(name) for name in ("numpy", "cv2", "shapely", "skimage")
    )
except (ImportError, ValueError):
    ENGINE_INSTALLED = False


class DoorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        drawing = Image.new("RGB", (400, 300), "white")
        pen = ImageDraw.Draw(drawing)
        pen.rectangle((40, 40, 360, 260), outline="black", width=6)
        pen.line((200, 40, 200, 260), fill="black", width=6)
        payload = BytesIO()
        drawing.save(payload, format="PNG")
        self.png = payload.getvalue()
        upload = self.client.post("/api/documents", files={"file": ("plan.png", self.png, "image/png")})
        self.assertEqual(upload.status_code, 200, upload.text)
        self.document_id = upload.json()["document_id"]
        self.frame = PageFrame.model_validate(
            self.client.get(f"/api/v1/documents/{self.document_id}/frame").json(), strict=False
        )

    def analyze(self, **body):
        return self.client.post(f"/api/v1/documents/{self.document_id}/door-analysis", json=body)

    def engine(self, boxes, *, complete=True):
        result = AnalyzerResult.model_validate({
            "schema_version": "planproof-door-result-v1", "source": self.frame.model_dump(),
            "version": "fake-engine", "status": "ok", "mode": "independent", "complete": complete,
            "detections": [{"id": f"d{index}", "bbox": box} for index, box in enumerate(boxes)],
        }, strict=False)
        return patch("app.run_deterministic", return_value=(result, "ok"))

    def ml(self, boxes):
        return {
            "schema_version": "planproof-detection-v1", "document_id": self.frame.raster_sha256,
            "coordinate_system": "pixel_edges_xy_top_left", "page_index": 0,
            "image_size": [self.frame.width, self.frame.height], "model_version": "test-model",
            "instances": [{"id": f"m{index}", "class": "door", "bbox": box}
                          for index, box in enumerate(boxes)],
        }

    def test_missing_engine_keeps_page_inspectable(self):
        with patch("app.run_deterministic", return_value=(unavailable(self.frame), "not_installed")):
            response = self.analyze()
        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertEqual(report["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(report["reasons"], ["analyzer_unavailable"])
        self.assertEqual(report["engines"]["deterministic"], "not_installed")
        self.assertEqual(report["engines"]["ml"], "not_provided")
        self.assertEqual(report["mode"], "live_engines")
        self.assertEqual(self.client.get(report["image_url"]).status_code, 200)

    def test_engine_alone_stays_unverified(self):
        with self.engine([[50.0, 50.0, 90.0, 90.0]]):
            report = self.analyze().json()
        self.assertEqual(report["status"], "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(report["answer"])
        self.assertEqual([item["id"] for item in report["unverified_evidence"]], ["deterministic:d0"])

    def test_engine_and_ml_disagreement_needs_review(self):
        door = [50.0, 50.0, 90.0, 90.0]
        with self.engine([door, [250.0, 150.0, 290.0, 190.0]]):
            report = self.analyze(ml_detections=self.ml([door])).json()
        self.assertEqual(report["status"], "NEEDS_REVIEW")
        self.assertEqual(report["matched_pairs"], [["deterministic:d0", "ml:m0"]])
        self.assertEqual([item["id"] for item in report["disputed_evidence"]], ["deterministic:d1"])
        self.assertEqual(report["gemini"]["status"], "not_authorized")
        self.assertEqual(report["engines"]["ml"], "provided")

    def test_matching_engines_without_ml_coverage_give_no_count(self):
        door = [50.0, 50.0, 90.0, 90.0]
        with self.engine([door]):
            report = self.analyze(ml_detections=self.ml([door])).json()
        self.assertEqual((report["status"], report["answer"]), ("INSUFFICIENT_EVIDENCE", None))
        self.assertEqual(report["reasons"], ["coverage_unknown"])
        self.assertEqual(len(report["matched_evidence"]), 2)

    def test_ml_from_another_raster_is_rejected_before_the_engine_runs(self):
        foreign = self.ml([])
        foreign["document_id"] = "0" * 64
        with patch("app.run_deterministic") as engine:
            response = self.analyze(ml_detections=foreign)
        self.assertEqual(response.status_code, 422)
        engine.assert_not_called()

    def test_unknown_document_and_extra_fields(self):
        self.assertEqual(self.client.post("/api/v1/documents/missing/door-analysis", json={}).status_code, 404)
        self.assertEqual(self.analyze(deterministic={}).status_code, 422)

    def test_local_viewer_origin_is_allowed(self):
        response = self.client.get(
            f"/api/v1/documents/{self.document_id}/frame", headers={"Origin": "http://localhost:4173"}
        )
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:4173")
        foreign = self.client.get(
            f"/api/v1/documents/{self.document_id}/frame", headers={"Origin": "https://example.com"}
        )
        self.assertNotIn("access-control-allow-origin", foreign.headers)


class EngineAdapterTests(unittest.TestCase):
    frame = PageFrame(
        document_id="doc", page_index=2, source_sha256="a" * 64, raster_sha256="b" * 64,
        width=40, height=30,
    )

    def fake_engine(self, analyze, door_result=None):
        package = types.ModuleType("planproof_evidence")
        package.analyze = analyze
        contracts = types.ModuleType("planproof_evidence.contracts")
        contracts.to_door_result_v1 = door_result
        return patch.dict(sys.modules, {
            "planproof_evidence": package, "planproof_evidence.contracts": contracts,
        })

    def document(self, raster_sha256):
        page = types.SimpleNamespace(page_index=0, raster_sha256=raster_sha256, image_size=(40, 30))
        return types.SimpleNamespace(pages=[page])

    def test_engine_failure_and_rejected_page_are_unavailable(self):
        def broken(*_):
            raise RuntimeError("engine bug")
        for analyze, state in ((broken, "failed"), (lambda *_: types.SimpleNamespace(pages=[]), "page_rejected")):
            with self.fake_engine(analyze):
                result, reported = run_deterministic(b"png", self.frame)
            self.assertEqual((reported, result.status, result.detections), (state, "unavailable", []))

    def test_result_for_another_raster_is_discarded(self):
        with self.fake_engine(lambda *_: self.document("c" * 64)):
            result, state = run_deterministic(b"png", self.frame)
        self.assertEqual((state, result.status), ("raster_mismatch", "unavailable"))

    def test_engine_boxes_are_bound_to_the_backend_page(self):
        def door_result(document, page_frame_override, page_index):
            self.assertEqual(page_index, 0)
            return {
                "schema_version": "planproof-door-result-v1", "source": page_frame_override,
                "version": "0.1.0", "status": "ok", "mode": "independent", "complete": True,
                "detections": [{"id": "obs-1", "bbox": [1, 2, 11, 12]}],
            }
        with self.fake_engine(lambda *_: self.document("b" * 64), door_result):
            result, state = run_deterministic(b"png", self.frame)
        self.assertEqual(state, "ok")
        self.assertEqual(result.source, self.frame)
        self.assertEqual(result.detections[0].bbox, [1.0, 2.0, 11.0, 12.0])

    def test_missing_package_is_reported(self):
        with patch.dict(sys.modules, {"planproof_evidence": None}):
            result, state = run_deterministic(b"png", self.frame)
        self.assertEqual((state, result.status), ("not_installed", "unavailable"))


@unittest.skipUnless(ENGINE_INSTALLED, "deterministic engine dependencies are not installed")
class RealEngineTests(unittest.TestCase):
    def test_real_engine_analyzes_the_canonical_demo_raster(self):
        client = TestClient(app)
        self.addCleanup(client.close)
        demo = client.post("/api/v1/demo/door-count", params={"scenario": "missing_ml"}).json()
        report = client.post(f"/api/v1/documents/{demo['source']['document_id']}/door-analysis", json={})
        self.assertEqual(report.status_code, 200, report.text)
        body = report.json()
        self.assertEqual(body["engines"]["deterministic"], "ok")
        self.assertEqual(body["source"]["raster_sha256"], demo["source"]["raster_sha256"])
        self.assertEqual(body["deterministic"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
