"""Host integration: canonical upload -> algorithms -> bounded Nemotron reasoning."""

import os
import subprocess
from pathlib import Path
from threading import Lock
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from planproof_feedback.client import ProviderError
from planproof_feedback.config import DEFAULT_CONFIG, load_config
from planproof_feedback.guided import answer_guided
from planproof_feedback.pipeline import EvidencePipeline
from planproof_feedback.service import create_service

_lock = Lock()
_jobs = {}


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)
    unit: str | None = Field(default=None, max_length=40)
    question_type: str | None = Field(default=None, max_length=40)
    allow_nemotron: bool = False


def get_job(document_id, page):
    # Only the host supplies paths and owns document/room registration.
    with _lock:
        if document_id not in _jobs:
            pipeline = EvidencePipeline()
            staging = pipeline.settings.work_root / "uploads"
            staging.mkdir(parents=True, exist_ok=True)
            source = staging / (document_id + ".png")
            source.write_bytes(page.png)
            _jobs[document_id] = pipeline.prepare(source)
            if len(_jobs) > 8:
                _jobs.pop(next(iter(_jobs)))
        return _jobs[document_id]


def specialist_service():
    from visual_specialist import VisualSpecialist

    config = load_config(
        Path(os.getenv("PLANPROOF_FEEDBACK_CONFIG", str(DEFAULT_CONFIG)))
    )
    if os.getenv("PLANPROOF_ENV_FILE"):
        config["env_file"] = str(Path(os.environ["PLANPROOF_ENV_FILE"]).resolve())
    config["providers"]["nemotron"].update(
        enable_thinking=False, timeout_seconds=90, retries=0
    )
    return VisualSpecialist(create_service(config))


def ask(document_id, page, request):
    from question_agent import DrawingAgent, GeminiTransport, settings

    if not request.allow_gemini:
        return {"status": "not_authorized", "answer": None, "canonical_citations": []}
    try:
        key, model = settings()
    except (ValueError, OSError):
        return {
            "status": "not_configured",
            "answer": None,
            "canonical_citations": [],
            "error_code": "gemini_configuration_required",
        }
    transport = GeminiTransport(key, model)
    request_id = uuid4().hex
    out = Path(
        os.getenv(
            "PLANPROOF_AGENT_TRACE_ROOT",
            str(Path(__file__).resolve().parents[1] / ".artifacts/agent/questions"),
        )
    ) / (request_id + ".json")
    try:
        result = DrawingAgent(
            page,
            request,
            transport,
            lambda: get_job(document_id, page),
            specialist_service,
            out=out,
        ).run()
        return {**result, "request_id": request_id}
    finally:
        transport.close()


def reason(document_id, page, request):
    job = get_job(document_id, page)
    common = {
        "stage": "algorithms_then_nemotron",
        "job_id": job.directory.name,
        "algorithm_evidence": job.catalog,
        "final_agent": "Gemini (next integration stage)",
    }
    if not request.allow_nemotron:
        return {**common, "status": "not_authorized", "answer": None}
    config = load_config(
        Path(os.getenv("PLANPROOF_FEEDBACK_CONFIG", str(DEFAULT_CONFIG)))
    )
    if os.getenv("PLANPROOF_ENV_FILE"):
        config["env_file"] = str(Path(os.environ["PLANPROOF_ENV_FILE"]).resolve())
    # This interactive test condition is explicit and separate from the frozen
    # direct benchmark, whose provider settings remain untouched.
    config["providers"]["nemotron"]["enable_thinking"] = (
        os.getenv("PLANPROOF_AGENT_THINKING", "false").lower() == "true"
    )
    try:
        service = create_service(config)
        request_id = uuid4().hex
        trace = answer_guided(
            service,
            job,
            request.question,
            config["images"],
            unit=request.unit,
            question_type=request.question_type,
            out=job.directory / "questions" / (request_id + ".json"),
        )
    except (ProviderError, ValueError, OSError) as exc:
        # Provider errors are sanitized by the transport; keys and local paths
        # are never returned as API error details.
        return {
            **common,
            "status": "unavailable",
            "answer": None,
            "error_code": type(exc).__name__,
        }
    final = trace["final"]
    return {
        **common,
        "status": final["status"],
        "answer": final["result"],
        "decision": trace["decision"],
        "canonical_citations": final["canonical_citations"],
        "usage": trace["usage"],
        "room_ids": trace["decision"]["region_ids"],
        "request_id": request_id,
    }


PIPELINE_ERRORS = (OSError, ValueError, subprocess.SubprocessError)
