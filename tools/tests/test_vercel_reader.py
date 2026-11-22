"""Exercise the hosted HTTP contract, compact CPU backend, and embedded OCR."""
from hashlib import sha256
from io import BytesIO
from http.client import HTTPConnection
from http.server import HTTPServer
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pymupdf
import pytest

from api import reader
from tools.free_agent_room_qa import ocr


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), reader.handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address
    finally:
        httpd.shutdown()
        thread.join()
        httpd.server_close()


def post(server, data, question, page=1, room_point=None):
    boundary = b"PlanProofTestForm"
    form = []
    fields = [("page", str(page).encode()), ("question", question.encode()), ("file", data)]
    if room_point is not None:
        fields.append(("room_point",room_point.encode()))
    for field, payload in fields:
        disposition = f'Content-Disposition: form-data; name="{field}"'
        if field == "file":
            disposition += '; filename="sheet.pdf"'
        form.append(b"--" + boundary + b"\r\n" + disposition.encode() + b"\r\n\r\n" + payload + b"\r\n")
    form.append(b"--" + boundary + b"--\r\n")
    connection = HTTPConnection(*server, timeout=90)
    try:
        connection.request("POST", "/api/reader", b"".join(form),
                           {"Content-Type": "multipart/form-data; boundary=" + boundary.decode()})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_health_and_upload(server):
    connection = HTTPConnection(*server)
    try:
        connection.request("GET", "/api/reader")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["ocr"] == "tesseract-eng"
    finally:
        connection.close()
    with pymupdf.open() as doc:
        doc.new_page()
        page = doc.new_page(width=420, height=160)
        page.insert_text((24, 40), "D6 certified fire-resistance rating: 60 minutes")
        pdf = doc.tobytes()
    question = "What is the certified fire-resistance rating of door D6 in minutes?"
    status, result = post(server, pdf, question, 2)
    assert status == 200
    assert (result["status"], result["value"], result["unit"]) == ("answered", 60, "min")
    assert result["page"] == result["page_count"] == 2
    assert result["width"] == 420 and result["coordinates"] == "displayed_pdf_points"
    assert result["source_sha256"] == sha256(pdf).hexdigest()
    assert result["evidence"] and all(len(row["bbox"]) == 4 for row in result["evidence"])
    status, result = post(server, pdf, question.replace("D6", "D9"), 2)
    assert status == 200 and result["status"] == "unresolved" and result["value"] is None
    assert post(server, pdf, question, 3)[0] == 422
    assert post(server, pdf, " ")[0] == 400
    assert post(server, b"<html></html>", question)[0] == 415


def test_native_binding_word_boxes():
    image = Image.new("L", (600, 100), 255)
    font = ImageFont.truetype(str(ocr.ASSETS / "DejaVuSans.ttf"), 38)
    ImageDraw.Draw(image).text((22, 12), "Project North", fill=0, font=font)
    tsv = ocr.read_tsv(image, 7, 10)
    assert tsv.startswith("level\tpage_num")
    assert "\tNorth" in tsv


def test_hallway_natural_request_returns_outline_and_scale(server):
    sample = Path(__file__).resolve().parents[2] / "frontend/src/assets/plan-sample.png"
    data = sample.read_bytes()
    for question in (
        "mark me the hallway, how big is it?",
        "Markiere den Flur und sag mir, wie groß er ist.",
    ):
        status, result = post(server, data, question)
        assert status == 200 and result["status"] == "answered", result
        assert result["source_sha256"] == sha256(data).hexdigest()
        assert result["unit"] == "m²" and result["approximate"] is True
        assert result["value"] == pytest.approx(12.8, abs=.2)
        outline, scale = result["evidence"]
        assert outline["kind"] == "room_polygon" and len(outline["polygon"]) >= 4
        assert outline["bbox"] == pytest.approx([160,420,960,580], abs=3)
        assert scale["kind"] == "graphic_scale" and len(scale["line"]) == 2


def unlabeled_hallway():
    image = Image.new("L",(1120,740),255)
    pen = ImageDraw.Draw(image)
    pen.rectangle((400,70,540,490),fill=229,outline=30,width=8)
    pen.rectangle((440,55,500,70),fill=229)
    # Alternating blocks on two rows with a dark perimeter stroke.
    for i,(a,b) in enumerate(zip((160,260,360,560),(260,360,560))):
        mid=611
        box = (a,600,b-1,mid) if i%2 else (a,mid,b-1,622)
        pen.rectangle(box,fill=0)
    pen.rectangle((160,600,560,622),outline=0,width=2)
    font = ImageFont.truetype(str(ocr.ASSETS / "DejaVuSans.ttf"),27)
    for x,label in [(160,"0'"),(260,"10'"),(360,"20'"),(560,"40'")]:
        pen.text((x,565),label,font=font,fill=0,anchor="mt")
    data = BytesIO()
    image.save(data,format="PNG")
    return data.getvalue()


def test_point_selects_unlabelled_floor_and_reads_scale_above(server):
    source = unlabeled_hallway()
    question = "Markiere den Flur und sag mir, wie groß er ist."
    status, missing = post(server,source,question)
    assert status==200 and missing["reason_code"]=="room_target_not_found"
    status, answer = post(server,source,question,room_point="0.42,0.4")
    assert status==200 and answer["status"]=="answered", answer
    assert answer["value"] == pytest.approx(124*405*(40*.3048/400)**2, rel=.05)
    assert answer["unit"]=="m²"
    assert answer["evidence"][0]["kind"]=="room_polygon"
    assert answer["evidence"][1]["text"]=="40 ft"
    assert post(server,source,question,room_point="nan,0.2")[0]==400


def test_cpu_reference_prediction_parity(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(reader.MODEL))
    import detector
    from planproof_runtime import detector_cpu
    weights = reader.MODEL / "eteks-door.npz"
    compact, cutoff = detector_cpu.load_model(weights)
    reference, original_cutoff = detector.load_model(weights)
    assert cutoff == original_cutoff
    pixels = np.random.default_rng(37).integers(0, 256, (95, 137), dtype=np.uint8)
    image = Image.fromarray(pixels)
    torch.set_num_threads(1)
    with torch.inference_mode():
        tensor = (255 - torch.from_numpy(pixels)).float()[None, None] / 255
        tensor = torch.nn.functional.pad(tensor, (0, (-image.width) % 16, 0, (-image.height) % 16))
        reference_heads = reference(tensor)
    for a, b in zip(detector_cpu.forward(compact, pixels), reference_heads):
        np.testing.assert_allclose(a, b[0].permute(1, 2, 0).numpy(), rtol=1e-4, atol=2e-4)
    # A low cutoff also exercises center decoding and border clipping on this synthetic patch.
    expected = [r for r in detector.view_predictions(reference, image, 0) if r["category"] == "single_door"]
    actual = detector_cpu.view_predictions(compact, image, 0)
    assert expected and len(actual) == len(expected)
    np.testing.assert_allclose([r["bbox"] for r in actual], [r["bbox"] for r in expected], atol=.002)
    np.testing.assert_allclose([r["confidence"] for r in actual], [r["confidence"] for r in expected], atol=1e-5)
