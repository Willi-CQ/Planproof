"""Checks for original-pixel evidence without private evaluation drawings."""
from hashlib import sha256
from io import BytesIO
from struct import pack
import zlib

from PIL import Image, ImageDraw, ImageFont
import pytest

from api import reader
from planproof_runtime import native_raster


@pytest.fixture(scope="module")
def original_scan():
    with Image.new("L", (14_000, 9_902), 255) as image:
        font = ImageFont.truetype(str(reader.ASSETS / "DejaVuSans.ttf"), 44)
        ImageDraw.Draw(image).text((9_500, 7_700), "Door D6", fill=0, font=font)
        stream = BytesIO()
        image.save(stream, format="PNG", compress_level=1)
    return stream.getvalue()


def test_original_pixels_and_source_bound_boxes(original_scan):
    assert len(original_scan) < reader.MAX_FILE
    assert reader.capabilities()["max_image_pixels"] == 150_000_000
    result = reader.read_drawing(original_scan, 1, "Where is the door marked D6 on this sheet?")
    assert result["reader"] == "Native-resolution tiled OCR"
    assert (result["width"], result["height"]) == (14_000, 9_902)
    assert result["source_sha256"] == sha256(original_scan).hexdigest()
    assert (result["status"], result["value"], result["coordinates"]) == ("answered", "D6", "image_pixels")
    evidence = next(item for item in result["evidence"] if item["kind"] == "door_mark")
    left, top, right, bottom = evidence["bbox"]
    assert 9_500 < left < right < 9_850 and 7_699 < top < bottom < 7_800
    coverage = result["raster_coverage"]
    assert 0 < coverage["tiles_read"] < coverage["tiles_total"]
    assert coverage["tiles_read"] + coverage["tiles_blank"] == coverage["tiles_total"]
    assert not coverage["deadline_reached"]


def test_unsupported_skips_native_inference(original_scan, monkeypatch):
    def should_not_run(*_):
        pytest.fail("Unsupported intent should not launch OCR")
    monkeypatch.setattr(reader, "read_raster_text", should_not_run)
    output = reader.read_drawing(original_scan, 1, "Explain the history of penguins")
    assert output["status"] == "unresolved" and output["reason_code"] == "unsupported_question"
    assert output["source_sha256"] == sha256(original_scan).hexdigest()


@pytest.mark.parametrize("dimensions", [(15_000, 10_001), (32_769, 10)])
def test_pixel_limits_precede_decompression(dimensions):
    def chunk(kind, content):
        return pack(">I", len(content)) + kind + content + pack(">I", zlib.crc32(kind + content))
    header = pack(">IIBBBBB", *dimensions, 8, 0, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"\x00")) + chunk(b"IEND", b"")
    with pytest.raises(reader.InputError) as err:
        reader.check_file(data, 1)
    assert err.value.status == 413


def test_timeout_reports_unfinished_coverage(monkeypatch):
    clock = iter([0, native_raster.OCR_SECONDS + 1])
    monkeypatch.setattr(native_raster, "monotonic", lambda: next(clock))
    scene = native_raster.read_native(Image.new("L", (128, 128), 255), "test-fixture")
    assert scene["raster_coverage"]["deadline_reached"]
    assert scene["raster_coverage"]["tiles_read"] == 0
    assert scene["native_text_complete"] is False


def test_transparent_ink_is_not_evidence():
    source = Image.new("RGBA", (200, 80), (0, 0, 0, 0))
    ImageDraw.Draw(source).rectangle((24, 16, 100, 50), fill=(255, 50, 50, 0))
    assert native_raster.text_tile(source, (0, 0, 200, 80)) is None
