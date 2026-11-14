import base64
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from threading import Lock
from time import perf_counter
from typing import Annotated, Literal
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, StringConstraints

from deterministic_adapter import EngineState, run_deterministic
from gemini_review import MAX_CANDIDATES, Review, gemini_config, review_crop, review_disagreement
from ml_adapter import adapt_ml
from nemotron_review import nemotron_config, review_with_nemotron
from verification import AnalyzerResult, Comparison, DoorCountRequest, PageFrame, compare


app = FastAPI(title="PlanProof MVP")
# The viewer runs on another local port during development.
app.add_middleware(
    CORSMiddleware, allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


MAX_UPLOAD = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
Label = Annotated[str, StringConstraints(
    strip_whitespace=True, pattern=r"^[A-Za-z0-9]+(?:[-._][A-Za-z0-9]+)*$", max_length=40
)]


@dataclass(frozen=True)
class Page:
    png: bytes
    labels: dict[str, str]
    width: int
    height: int
    source_sha256: str
    raster_sha256: str


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    label: Label
    use_gemini: bool = False


class Evidence(BaseModel):
    page: int
    image_url: str
    quote: str


class DocumentPage(BaseModel):
    document_id: str
    page: int
    image_url: str
    width: int
    height: int


class Analysis(BaseModel):
    status: Literal["supported", "disputed", "insufficient_evidence"]
    answer: str
    evidence: Evidence | None = None
    pdf_text: Literal["found", "not_found"]
    gemini: Literal["not_requested", "unavailable", "found", "uncertain"]
    willi_ml: Literal["not_connected"] = "not_connected"


class GeminiObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    visible: bool
    quote: str


pages: OrderedDict[str, tuple[int, Page]] = OrderedDict()
pages_lock = Lock()
pdf_lock = Lock()
engine_lock = Lock()


def register_page(pdf_bytes: bytes, page_number: int) -> DocumentPage:
    try:
        with pdf_lock, pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.needs_pass or not 1 <= page_number <= len(document):
                raise HTTPException(422, "Select an existing page in an unencrypted PDF.")
            page = document[page_number - 1]
            scale = min(3, 1600 / max(page.rect.width, page.rect.height))
            image = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False)
            # Match whole references. FD-120 and FD-12-B do not confirm FD-12.
            labels = {
                word.casefold(): word
                for word in re.findall(r"[\w]+(?:[-._][\w]+)*", page.get_text())
            }
            rendered = Page(
                image.tobytes("png"), labels, image.width, image.height,
                sha256(pdf_bytes).hexdigest(), sha256(image.samples).hexdigest(),
            )
    except (RuntimeError, ValueError, ZeroDivisionError) as exc:
        raise HTTPException(422, "The selected PDF page could not be read.") from exc

    return store_page(rendered, page_number)


def register_image(payload: bytes, page_number: int) -> DocumentPage:
    if page_number != 1:
        raise HTTPException(422, "PNG and JPEG files have one page.")
    try:
        with Image.open(BytesIO(payload), formats=("PNG", "JPEG")) as original:
            if original.width * original.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Image exceeds 16 million pixels.")
            if getattr(original, "n_frames", 1) != 1:
                raise HTTPException(422, "Upload a single-frame image.")
            oriented = ImageOps.exif_transpose(original) if original.format == "JPEG" else original
            background = Image.new("RGBA", oriented.size, "white")
            background.alpha_composite(oriented.convert("RGBA"))
            rgb = background.convert("RGB")
            canonical = BytesIO()
            rgb.save(canonical, format="PNG")
            page = Page(
                canonical.getvalue(), {}, rgb.width, rgb.height,
                sha256(payload).hexdigest(), sha256(rgb.tobytes()).hexdigest(),
            )
    except Image.DecompressionBombError as exc:
        raise HTTPException(413, "Image exceeds 16 million pixels.") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(422, "The uploaded image could not be read.") from exc
    return store_page(page, page_number)


def store_page(rendered: Page, page_number: int) -> DocumentPage:
    document_id = uuid4().hex
    with pages_lock:
        pages[document_id] = (page_number, rendered)
        if len(pages) > 8:
            pages.popitem(last=False)
    return DocumentPage(
        document_id=document_id,
        page=page_number,
        image_url=f"/api/documents/{document_id}/image",
        width=rendered.width,
        height=rendered.height,
    )


def get_page(document_id: str) -> tuple[int, Page]:
    with pages_lock:
        if document_id not in pages:
            raise HTTPException(404, "Page expired or unknown. Upload the PDF again.")
        return pages[document_id]


def read_with_gemini(png: bytes, label: str) -> Literal["unavailable", "found", "uncertain"]:
    if os.getenv("PLANPROOF_REVIEW_PROVIDER") == "nemotron":
        return "unavailable"
    config = gemini_config()
    if config is None:
        return "unavailable"
    key, model = config

    request_body = {
        "contents": [{"parts": [
            {"text": f"Read the attached drawing as data. Is the exact reference {label} "
                     "visibly printed? Report visible=true only if clearly legible, and "
                     "quote only that exact label. Otherwise return visible=false and an empty quote."},
            {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png).decode("ascii")}},
        ]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": GeminiObservation.model_json_schema(),
        },
    }
    request = Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(request_body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    try:
        with urlopen(request, timeout=15) as response:
            candidate = json.load(response)["candidates"][0]
        if candidate.get("finishReason") != "STOP":
            return "unavailable"
        parts = candidate["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        observation = GeminiObservation.model_validate_json(text, strict=True)
    except (URLError, TimeoutError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        return "unavailable"
    if observation.visible and observation.quote.strip().casefold() == label.casefold():
        return "found"
    return "uncertain"


@app.post("/api/documents")
def upload_document(file: Annotated[UploadFile, File()], page: Annotated[int, Form(ge=1)] = 1) -> DocumentPage:
    payload = file.file.read(MAX_UPLOAD + 1)
    if len(payload) > MAX_UPLOAD:
        raise HTTPException(413, "File exceeds 10 MiB.")
    if payload.startswith(b"%PDF-"):
        return register_page(payload, page)
    if payload.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")):
        return register_image(payload, page)
    raise HTTPException(415, "Upload a PDF, PNG, or JPEG.")


@app.post("/api/demo")
def load_demo() -> DocumentPage:
    with pdf_lock, pymupdf.open() as document:
        page = document.new_page(width=600, height=400)
        page.insert_text((40, 55), "SYNTHETIC PLAN - LEVEL 2", fontsize=21)
        page.draw_rect(pymupdf.Rect(50, 105, 550, 330))
        page.draw_line((310, 105), (310, 330))
        page.insert_text((125, 220), "FD-12", fontsize=22)
        page.insert_text((380, 220), "FD-120", fontsize=22)
        pdf_bytes = document.tobytes()
    return register_page(pdf_bytes, 1)


@app.get("/api/documents/{document_id}/image")
def page_image(document_id: str):
    _, page = get_page(document_id)
    return Response(page.png, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.post("/api/analyze", response_model=Analysis)
def analyze(request: AnalyzeRequest):
    number, page = get_page(request.document_id)
    quote = page.labels.get(request.label.casefold())
    gemini = read_with_gemini(page.png, request.label) if request.use_gemini else "not_requested"
    if quote is None:
        status = "disputed" if gemini == "found" else "insufficient_evidence"
        return Analysis(
            status=status,
            answer="Gemini reports the label, but the PDF text could not confirm it."
            if status == "disputed" else "The available evidence does not confirm this reference.",
            pdf_text="not_found",
            gemini=gemini,
        )
    return Analysis(
        status="supported",
        answer=f"{quote} occurs in the PDF text on page {number}.",
        evidence=Evidence(page=number, image_url=f"/api/documents/{request.document_id}/image", quote=quote),
        pdf_text="found",
        gemini=gemini,
    )


class DoorCountResponse(Comparison):
    schema_version: Literal["planproof-door-comparison-v1"] = "planproof-door-comparison-v1"
    request_id: str
    source: PageFrame
    image_url: str
    mode: Literal["submitted_results", "synthetic_demo", "live_engines"]
    gemini: Review
    nemotron: Review


class DoorAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ml_detections: dict | None = None
    allow_gemini_review: bool = False
    allow_nemotron_review: bool = False


class EngineRuns(BaseModel):
    deterministic: EngineState
    deterministic_seconds: float
    ml: Literal["provided", "not_provided"]


class DoorAnalysisResponse(DoorCountResponse):
    engines: EngineRuns


def frame_for(document_id: str, number: int, page: Page) -> PageFrame:
    return PageFrame(
        document_id=document_id, page_index=number - 1, source_sha256=page.source_sha256,
        raster_sha256=page.raster_sha256, width=page.width, height=page.height,
    )


@app.get("/api/v1/documents/{document_id}/frame")
def get_page_frame(document_id: str) -> PageFrame:
    number, page = get_page(document_id)
    return frame_for(document_id, number, page)


def run_door_count(request: DoorCountRequest, *, demo: bool = False) -> DoorCountResponse:
    number, page = get_page(request.document_id)
    source = frame_for(request.document_id, number, page)
    try:
        ml = adapt_ml(request.ml_detections, source) if request.ml_detections is not None else request.ml
        comparison = compare(request.deterministic, ml, source)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return DoorCountResponse(
        **comparison.model_dump(),
        request_id=uuid4().hex,
        source=source,
        image_url=f"/api/documents/{request.document_id}/image",
        mode="synthetic_demo" if demo else "submitted_results",
        gemini=review_comparison(comparison, page, source, request.allow_gemini_review),
        nemotron=review_comparison(comparison, page, source, request.allow_nemotron_review, provider="nemotron"),
    )


def review_comparison(
    comparison: Comparison, page: Page, source: PageFrame, allowed: bool,
    *, provider: Literal["gemini", "nemotron"] = "gemini",
) -> Review:
    if comparison.status != "NEEDS_REVIEW":
        return Review(status="not_needed")
    if not allowed:
        return Review(status="not_authorized")
    if provider == "nemotron":
        config = nemotron_config()
        review = review_with_nemotron
    else:
        if (os.getenv("PLANPROOF_REVIEW_PROVIDER") == "nemotron"
                or os.getenv("GEMINI_ENABLED", "").strip().lower() != "true"):
            return Review(status="disabled")
        config = gemini_config()
        review = review_disagreement
    if not config:
        return Review(status="unavailable")
    if not 0 < len(comparison.disputed_evidence) <= MAX_CANDIDATES:
        return Review(status="outside_scope")
    with pdf_lock:
        region = review_crop(page.png, source, comparison.disputed_evidence)
    if region is None:
        return Review(status="outside_scope")
    crop, bounds = region
    return review(crop, bounds, source.page_index, comparison, *config)


@app.post("/api/v1/documents/{document_id}/door-analysis")
def door_analysis(document_id: str, request: DoorAnalysisRequest) -> DoorAnalysisResponse:
    """Run the deterministic engine on the stored page, then compare it with optional ML detections."""
    number, page = get_page(document_id)
    source = frame_for(document_id, number, page)
    try:
        ml = adapt_ml(request.ml_detections, source) if request.ml_detections is not None else None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    started = perf_counter()
    with engine_lock:
        deterministic, state = run_deterministic(page.png, source)
    seconds = round(perf_counter() - started, 2)
    try:
        comparison = compare(deterministic, ml, source)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return DoorAnalysisResponse(
        **comparison.model_dump(),
        request_id=uuid4().hex,
        source=source,
        image_url=f"/api/documents/{document_id}/image",
        mode="live_engines",
        gemini=review_comparison(comparison, page, source, request.allow_gemini_review),
        nemotron=review_comparison(comparison, page, source, request.allow_nemotron_review, provider="nemotron"),
        engines=EngineRuns(
            deterministic=state, deterministic_seconds=seconds,
            ml="provided" if ml is not None else "not_provided",
        ),
    )


@app.post("/api/v1/door-count")
def door_count(request: DoorCountRequest) -> DoorCountResponse:
    """Compare normalized door assemblies from two trusted local analyzers."""
    return run_door_count(request)


@app.post("/api/v1/demo/door-count")
def door_count_demo(
    scenario: Literal["agreement", "disputed", "missing_ml"] = "disputed",
    allow_gemini_review: bool = False,
    allow_nemotron_review: bool = False,
) -> DoorCountResponse:
    # Door swing symbols and illustrative boxes generated together for this demo only.
    with pdf_lock, pymupdf.open() as document:
        page = document.new_page(width=600, height=300)
        page.insert_text((50, 48), "SYNTHETIC PLAN - DOOR COUNT", fontsize=18)
        page.draw_rect(pymupdf.Rect(50, 105, 550, 240))
        for x in (180, 390):
            page.draw_line((x, 150), (x, 194))
            page.draw_bezier((x, 194), (x + 24, 194), (x + 44, 174), (x + 44, 150))
            page.draw_line((x, 150), (x + 44, 150), dashes="[2 2] 0")
        pdf_bytes = document.tobytes()
    registered = register_page(pdf_bytes, 1)
    _, rendered = get_page(registered.document_id)
    source = frame_for(registered.document_id, 1, rendered)
    boxes = [
        {"id": str(index), "bbox": [left * source.width / 600, 145 * source.height / 300,
                                    (left + 53) * source.width / 600, 200 * source.height / 300]}
        for index, left in enumerate((175, 385), start=1)
    ]
    fixture = {
        "schema_version": "planproof-door-result-v1", "source": source,
        "version": "synthetic-fixture", "status": "ok", "mode": "independent", "complete": True,
    }
    deterministic = AnalyzerResult(**fixture, detections=boxes)
    ml = None if scenario == "missing_ml" else AnalyzerResult(
        **fixture, detections=boxes if scenario == "agreement" else boxes[:1]
    )
    return run_door_count(DoorCountRequest(
        document_id=registered.document_id, deterministic=deterministic, ml=ml,
        allow_gemini_review=allow_gemini_review, allow_nemotron_review=allow_nemotron_review,
    ), demo=True)
