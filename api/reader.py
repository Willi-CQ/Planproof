"""Same-origin, stateless Vercel endpoint for the PlanProof drawing reader."""
from email import policy
from email.parser import BytesParser
from functools import lru_cache
from hashlib import sha256
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from contextlib import contextmanager
import json
import logging
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
import warnings

# Limit per-request native threads. Reader extraction uses PyMuPDF, which requires
# serialized access inside a single process (Fluid Compute can serve concurrently).
os.environ.update(PLANPROOF_COMPACT_CPU="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
READ_LOCK = Lock()
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "planproof_runtime"
MODEL = ROOT / "models" / "sheet-doors-v1"
MAX_FILE = 4_000_000
MAX_BODY = 4_350_000
MAX_PIXELS = 150_000_000
MAX_EDGE = 32_768
FULL_SCENE_PIXELS = 16_000_000
MAX_QUESTION = 2_000
WEIGHTS_SHA256 = "96a60af10bcde11b02f8422b7b8ef4b6a6022524ef478c0dc5ee6f3b3600a4f8"
TESSDATA_SHA256 = "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2"


class InputError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


@lru_cache(maxsize=1)
def capabilities():
    import cv2
    import pymupdf
    from scipy.spatial import cKDTree
    from PIL import ImageFont
    from tesserocr import PyTessBaseAPI
    from tools.free_agent_room_qa import answer, build_scene
    cv2.setNumThreads(1)
    if sha256((ASSETS / "eng.traineddata").read_bytes()).hexdigest() != TESSDATA_SHA256:
        raise RuntimeError("English recognition data mismatch")
    if sha256((MODEL / "eteks-door.npz").read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise RuntimeError("Door detector data mismatch")
    ImageFont.truetype(str(ASSETS / "DejaVuSans.ttf"), 12)
    with PyTessBaseAPI(path=str(ASSETS), lang="eng") as api:
        if "eng" not in api.GetLoadedLanguages():
            raise RuntimeError("English recognition unavailable")
    revision = ASSETS / "revision.txt"
    return dict(status="ready", ready=True, version="cbb026b", revision=revision.read_text().strip()
                if revision.is_file() else "development", reader="PlanProof",
                formats=["pdf", "png", "jpeg"], native_pdf=True, ocr="tesseract-eng",
                detector="eteks-door-numpy-fp32", weights_sha256=WEIGHTS_SHA256,
                max_file_bytes=MAX_FILE, max_image_pixels=MAX_PIXELS, max_image_edge=MAX_EDGE,
                max_geometry_pixels=FULL_SCENE_PIXELS, native_raster_text=True,
                max_question_chars=MAX_QUESTION)


def parse_upload(content_type, body):
    if len(body) > MAX_BODY:
        raise InputError(413, "The request exceeds the upload limit.")
    if not content_type.lower().startswith("multipart/form-data;"):
        raise InputError(415, "Send a multipart form containing file, question and page.")
    if "\n" in content_type or "\r" in content_type or len(content_type) > 512:
        raise InputError(400, "Invalid multipart content type.")
    try:
        header = content_type.encode("ascii")
    except UnicodeEncodeError:
        raise InputError(400, "Invalid multipart content type.") from None
    message = BytesParser(policy=policy.default).parsebytes(b"Content-Type: " + header + b"\r\n\r\n" + body)
    if not message.is_multipart() or message.defects:
        raise InputError(400, "Malformed multipart form.")
    fields = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name not in {"file", "page", "question"}:
            continue
        if name in fields or part.is_multipart():
            raise InputError(400, "Each form field must appear once.")
        fields[name] = part.get_payload(decode=True)
    data = fields.get("file")
    if not data:
        raise InputError(400, "Select a drawing.")
    if len(data) > MAX_FILE:
        raise InputError(413, "The selected drawing exceeds 4 MB.")
    if len(fields.get("page", b"1")) > 8 or len(fields.get("question", b"")) > MAX_QUESTION * 4:
        raise InputError(400, "The page or question is too long.")
    try:
        page = int(fields.get("page", b"1").decode("ascii"))
        question = fields.get("question", b"").decode("utf-8").strip()
    except ValueError:
        raise InputError(400, "Provide a valid page number and UTF-8 question.") from None
    if page < 1 or not 1 <= len(question) <= MAX_QUESTION:
        raise InputError(400, "Enter a question and a positive page number.")
    return data, page, question


def open_raster(data):
    from PIL import Image
    # Pillow's default warning threshold is lower than our 150 MP budget.
    # Keep its hard decompression limit and enforce our own dimensions below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        return Image.open(BytesIO(data), formats=["JPEG", "PNG"])


def check_file(data, page):
    import pymupdf
    from PIL import Image, UnidentifiedImageError
    if data.lstrip()[:5] == b"%PDF-":
        try:
            with pymupdf.open(stream=data, filetype="pdf") as doc:
                if doc.needs_pass:
                    raise InputError(422, "Unlock the PDF before uploading it.")
                if page > len(doc):
                    raise InputError(422, "The selected page is outside the PDF.")
                rect = doc[page - 1].rect
                if rect.is_empty or rect.is_infinite or not all(math.isfinite(v) for v in rect):
                    raise InputError(422, "The selected PDF page has invalid dimensions.")
                return ".pdf", len(doc), (rect.width, rect.height)
        except (RuntimeError, ValueError) as exc:
            raise InputError(422, "The PDF could not be opened.") from exc
    if not data.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")):
        raise InputError(415, "Choose a PDF, PNG or JPEG drawing.")
    try:
        with open_raster(data) as image:
            if image.width * image.height > MAX_PIXELS or max(image.size) > MAX_EDGE:
                raise InputError(413, "This image exceeds 150 megapixels or the 32,768-pixel edge limit.")
            if page != 1:
                raise InputError(422, "An image contains one page.")
            suffix = ".png" if image.format == "PNG" else ".jpg"
            size = image.size
            if image.width * image.height > FULL_SCENE_PIXELS:
                image.verify()
            else:
                image.load()
    except Image.DecompressionBombError as exc:
        raise InputError(413, "This image exceeds 150 megapixels.") from exc
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise InputError(422, "The drawing image could not be opened.") from exc
    return suffix, 1, size


@contextmanager
def reader_slot():
    if not READ_LOCK.acquire(timeout=5):
        raise InputError(503, "The reader is processing another drawing. Retry in a moment.")
    try:
        yield
    finally:
        READ_LOCK.release()


def read_raster_text(data):
    from planproof_runtime.native_raster import read_native
    with open_raster(data) as image:
        try:
            image.load()
        except (OSError, ValueError) as exc:
            raise InputError(422, "The drawing image could not be decoded.") from exc
        return read_native(image, sha256(data).hexdigest())


def read_drawing(data, page, question):
    from planproof_runtime.native_raster import empty_scene, read_native
    from tools.free_agent_room_qa import answer_program, build_scene, compile_question
    from tools.free_agent_room_qa.page_text import TEXT_OPS
    from tools.free_agent_room_qa.raster_tags import templates
    from tools.free_agent_room_qa.room_geometry import inspect_room
    from tools.free_agent_room_qa.reader import label_reads, native_reads, ocr_read
    with reader_slot():
        capabilities()
        suffix, count, size = check_file(data, page)
        program = compile_question(question)
        op = program["op"]
        if op in {"locate_room", "measure_room"}:
            digest = sha256(data).hexdigest()
            if suffix == ".pdf":
                with TemporaryDirectory(prefix="planproof-room-") as folder:
                    path = Path(folder) / "source.pdf"
                    path.write_bytes(data)
                    image, lines, _, width, height, _, _ = native_reads(path, page)
                scene = empty_scene(digest, page, size, pdf=True)
                sx, sy = image.width/width, image.height/height
                if not lines:
                    lines, _ = ocr_read(image, width, height)
                labels, _ = label_reads(lines)
                scene.update(lines=lines, rooms={k:{"label":v} for k,v in labels.items()})
                result = inspect_room(image, scene, program, frame_scale=(sx,sy))
            else:
                with open_raster(data) as image:
                    image.load()
                    scene = read_native(image, digest)
                    result = inspect_room(image, scene, program)
            return {**result, "program": program, "page": page, "page_count": count,
                    "width": size[0], "height": size[1], "coordinates": scene["coordinates"],
                    "source_sha256": digest, "reader": "Floor outline and graphic scale"}
        large = suffix != ".pdf" and size[0] * size[1] > FULL_SCENE_PIXELS
        large_geometry = large and op in {"adjacent_doors", "room_door_count"}
        if op == "unsupported" or large_geometry:
            scene = empty_scene(sha256(data).hexdigest(), page, size, pdf=suffix == ".pdf")
        elif suffix != ".pdf" and (large or op in TEXT_OPS):
            scene = read_raster_text(data)
        else:
            with TemporaryDirectory(prefix="planproof-") as folder:
                path = Path(folder) / ("drawing" + suffix)
                path.write_bytes(data)
                try:
                    scene = build_scene(path, page)
                finally:
                    # These glyph templates are keyed by text from the uploaded page.
                    templates.cache_clear()
        result = answer_program(scene, program)
        if op == "unsupported":
            result.setdefault("reason_code", "unsupported_question")
        if large_geometry:
            result.update(reason_code="geometry_on_large_raster",
                          reason="This large image supports native-resolution text reading. "
                                 "Confirming room connections requires physical opening geometry.")
        coverage = scene.get("raster_coverage")
        if coverage:
            result["raster_coverage"] = coverage
            evidence_missing = result["status"] == "unresolved" and result.get("reason_code") not in {
                "visual_count_required", "scope_unverified", "unsupported_question"}
            if evidence_missing and (coverage["deadline_reached"] or coverage["tiles_timed_out"]
                                     or coverage["line_limit_reached"]):
                result.update(reason_code="ocr_budget_exhausted",
                              reason="The reader could not verify the requested answer within the native OCR coverage budget.")
            elif evidence_missing and not scene["lines"]:
                result.update(reason_code="no_readable_text", reason="Native OCR found no legible text on this page.")
    return {**result, "page_count": count, "reader": scene["reader"]}


class handler(BaseHTTPRequestHandler):
    def send_json(self, status, value):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        try:
            self.send_json(200, capabilities())
        except Exception as exc:
            logging.error("Reader initialization failed (%s)", type(exc).__name__)
            self.send_json(503, {"error": "Drawing reader initialization failed.", "ready": False})

    def do_POST(self):
        try:
            try:
                size = int(self.headers.get("Content-Length", ""))
            except ValueError:
                raise InputError(411, "A content length is required.") from None
            if size > MAX_BODY:
                raise InputError(413, "The request exceeds the upload limit.")
            if size < 1:
                raise InputError(400, "The upload is empty.")
            data, page, question = parse_upload(self.headers.get("Content-Type", ""), self.rfile.read(size))
            response = read_drawing(data, page, question)
            self.send_json(200, response)
        except InputError as exc:
            self.send_json(exc.status, {"error": str(exc)})
        except Exception as exc:
            logging.error("Drawing reader failed (%s)", type(exc).__name__)
            self.send_json(500, {"error": "The drawing reader could not finish reading this page."})
