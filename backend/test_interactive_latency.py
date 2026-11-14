"""Latency contracts use controlled tools/providers, never accuracy claims."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from google.genai import types

import algorithm_jobs as jobs
from algorithm_jobs import AlgorithmPending, BackgroundJobs
from gemini_transport import ProviderFailure, ProviderGate
from question_agent import AgentBudget, GeminiTransport
from test_question_agent import agent, final, response


def batch(*responses):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                finish_reason="STOP",
                content=types.Content(
                    role="model",
                    parts=[r.candidates[0].content.parts[0] for r in responses],
                ),
            )
        ]
    )


def reflection(action="answer"):
    return response(
        "reflect",
        {
            "draft": "Two separate stairs.",
            "checks": ["Checked duplicate locations."],
            "unresolved": [],
            "next_action": action,
        },
    )


def test_slow_algorithms_do_not_block_crops_or_claim_evidence():
    gate = threading.Event()
    manager = BackgroundJobs()
    build = Mock(side_effect=lambda: gate.wait(3) or None)
    value = agent(
        batch(
            response("get_algorithms", {"classes": ["stairs"]}),
            response("crop_region", {"bbox": [0, 0, 500, 500]}),
        ),
        response("submit_answer", final("crop-1")),
    )
    value.algorithm_factory = lambda: manager.get("same-source", build, timeout=0.01)
    events = []
    value.observer = lambda stage, state, **data: events.append((stage, state, data))
    started = time.monotonic()
    try:
        result = value.run()
        assert time.monotonic() - started < 1.5
        assert result["answer"]["answer"] == 2
        assert not result["algorithm_used"] and not result["algorithm_executed"]
        assert result["tool_trace"][0]["result"]["status"] == "pending"
        assert any(
            stage == "get_algorithms" and state == "pending"
            for stage, state, _ in events
        )
        assert not any(stage == "algorithm_bundle" for stage, _, _ in events)
        assert build.call_count == 1
    finally:
        gate.set()
        manager.pool.shutdown()


def test_single_flight_queue_limit_and_ready_result_reuse():
    manager = BackgroundJobs(capacity=2)
    gate = threading.Event()
    expected = object()

    def build():
        gate.wait(3)
        return expected

    builder = Mock(side_effect=build)
    try:
        for _ in range(3):
            with pytest.raises(AlgorithmPending):
                manager.get("a", builder, timeout=0.01)
        with pytest.raises(AlgorithmPending):
            manager.get("b", builder, timeout=0)
        with pytest.raises(AlgorithmPending) as error:
            manager.get("c", builder, timeout=0)
        assert error.value.status == "busy"
        gate.set()
        assert manager.get("a", builder) is expected
        assert manager.get("a", builder) is expected
        assert manager.get("b", builder) is expected
        assert builder.call_count == 2
    finally:
        gate.set()
        manager.pool.shutdown()


def test_failed_worker_is_not_pending_and_does_not_restart_per_call():
    manager = BackgroundJobs()
    builder = Mock(side_effect=TimeoutError("worker timed out"))
    try:
        for _ in range(2):
            with pytest.raises(TimeoutError):
                manager.get("a", builder)
        assert builder.call_count == 1
    finally:
        manager.pool.shutdown()


def test_reflection_and_final_can_share_one_turn_but_new_evidence_cannot():
    a = agent(batch(reflection(), response("submit_answer", final())))
    result = a.run()
    assert result["answer"]["answer"] == 2 and a.transport.generate.call_count == 1
    assert result["reflection_calls"] == 1
    b = agent(
        batch(
            response("crop_region", {"bbox": [0, 0, 500, 500]}),
            response("submit_answer", final("crop-1")),
        ),
        response("submit_answer", final("crop-1")),
    )
    assert b.run()["answer"]["answer"] == 2 and b.transport.generate.call_count == 2
    c = agent(
        batch(reflection("inspect"), response("submit_answer", final())),
        response("submit_answer", final()),
    )
    assert c.run()["answer"]["answer"] == 2 and c.transport.generate.call_count == 2


def test_fast_deadline_reserves_finalization_and_caps_specialists():
    now = [0.0]
    a = agent()
    a.budget = AgentBudget.interactive()

    def generate(*args, **kwargs):
        if kwargs["phase"] == "normal":
            assert kwargs["deadline"] == 85
            now[0] = 86
            raise ProviderFailure(
                {
                    "error_code": "ProviderBudgetExhausted",
                    "category": "budget",
                    "retryable": False,
                }
            )
        assert kwargs["deadline"] == 120
        return response("submit_answer", final())

    a.transport.generate.side_effect = generate
    with patch("question_agent.time.monotonic", side_effect=lambda: now[0]):
        result = a.run()
    assert result["reliability"]["recovery_succeeded"]
    assert result["latency_budget"]["seconds"] == 120
    b = agent(
        response("crop_region", {"bbox": [0, 0, 500, 500]}),
        response("review_answer", {"candidate": "Two stairs", "image_ids": ["crop-1"]}),
        response("submit_answer", final("crop-1")),
    )
    b.budget = AgentBudget.interactive()
    b.run()
    assert b.specialist.review.call_args.kwargs["timeout_seconds"] <= 15


def test_invalid_optional_self_check_does_not_add_another_model_turn():
    check = reflection()
    check.candidates[0].content.parts[0].function_call.args["checks"] = ["x" * 501]
    value = agent(batch(check, response("submit_answer", final())))
    result = value.run()
    assert result["status"] == "ok" and result["answer"]["answer"] == 2
    assert value.transport.generate.call_count == 1
    assert result["reflection_calls"] == 0
    assert result["tool_trace"][0]["result"]["status"] == "unavailable"


def test_interactive_transport_reduces_thinking_without_output_truncation():
    transport = GeminiTransport("fixture", "gemini-4-carbon")
    transport.gate = ProviderGate()
    assert transport.generation_profile() == {
        "max_output_tokens": 8192,
        "thinking_level": "LOW",
    }
    assert transport.policy.attempts == 2 and transport.policy.attempt_seconds == 35
    transport.phase = "token_repair"
    assert transport.generation_profile()["max_output_tokens"] == 16384
    unknown = GeminiTransport("fixture", "unknown-model")
    assert unknown.generation_profile()["thinking_level"] is None


def test_content_cache_reuses_after_reupload_and_restart_and_invalidates_recipe(
    tmp_path, monkeypatch
):
    from planproof_feedback.pipeline import PipelineSettings

    settings = PipelineSettings(
        "python", "python", tmp_path / "model.pt", tmp_path / "profile.json", tmp_path
    )
    settings.checkpoint.write_bytes(b"model")
    settings.calibration.write_bytes(b"profile")
    monkeypatch.setattr(jobs.PipelineSettings, "from_env", lambda: settings)
    monkeypatch.setattr(
        jobs,
        "recipe",
        lambda s: {
            "model": jobs.digest(s.checkpoint),
            "profile": jobs.digest(s.calibration),
            "version": 1,
        },
    )
    manager = BackgroundJobs()
    monkeypatch.setattr(jobs, "_background", manager)
    page = SimpleNamespace(
        raster_sha256="a" * 64, width=20, height=10, png=b"synthetic PNG fixture"
    )
    directory = tmp_path / ("b" * 32)

    def make_job(source):
        directory.mkdir(exist_ok=True)
        for name in (
            "page.png",
            "evidence/evidence.json",
            "rooms/room-boundaries.json",
        ):
            path = directory / name
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text("fixture", encoding="utf-8")
        record = {
            "raster_sha256": page.raster_sha256,
            "model_sha256": jobs.digest(settings.checkpoint),
            "profile_sha256": jobs.digest(settings.calibration),
            "labels_used": False,
        }
        for name, field in (
            ("page.png", "canonical_sha256"),
            ("evidence/evidence.json", "evidence_sha256"),
            ("rooms/room-boundaries.json", "boundaries_sha256"),
        ):
            record[field] = jobs.sha256(directory / name)
        jobs.write_json(directory / "pipeline-receipt.json", record)
        return SimpleNamespace(
            directory=directory, page={"frame": {"image_size": [20, 10]}}
        )

    prepare = Mock(side_effect=make_job)
    monkeypatch.setattr(
        jobs, "EvidencePipeline", lambda settings: SimpleNamespace(prepare=prepare)
    )
    monkeypatch.setattr(
        jobs,
        "EvidenceJob",
        lambda directory, settings: SimpleNamespace(
            directory=directory, page={"frame": {"image_size": [20, 10]}}
        ),
    )
    try:
        assert jobs.get_algorithm_job(page).directory == directory
        assert (
            jobs.get_algorithm_job(SimpleNamespace(**vars(page))).directory == directory
        )
        assert prepare.call_count == 1
        manager.pool.shutdown()
        manager = BackgroundJobs()
        monkeypatch.setattr(jobs, "_background", manager)
        jobs.get_algorithm_job(page)
        assert prepare.call_count == 1  # Restored receipt-verified artifacts.
        settings.checkpoint.write_bytes(b"new model")
        jobs.get_algorithm_job(page)
        assert prepare.call_count == 2
        # Corruption must invalidate a persisted cache entry after another restart.
        (directory / "evidence/evidence.json").write_text("corrupt", encoding="utf-8")
        manager.pool.shutdown()
        manager = BackgroundJobs()
        monkeypatch.setattr(jobs, "_background", manager)
        jobs.get_algorithm_job(page)
        assert prepare.call_count == 3
    finally:
        manager.pool.shutdown()


def test_cache_identity_separates_source_size_and_model():
    page = SimpleNamespace(raster_sha256="a", width=20, height=10)
    config = {"model": "a", "profile": "x"}
    first = jobs.identity(page, config)
    assert first != jobs.identity(page, {**config, "model": "b"})
    page.width = 21
    assert first != jobs.identity(page, config)
    page.width = 20
    page.raster_sha256 = "b"
    assert first != jobs.identity(page, config)
