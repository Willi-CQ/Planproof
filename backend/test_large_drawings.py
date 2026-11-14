"""Original pixels survive upload and remain available for focused inspection."""

from hashlib import sha256
from io import BytesIO
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException
from PIL import Image

from app import get_page, register_image
from question_agent import AgentRequest, DrawingAgent
from planproof_feedback.images import MAX_SOURCE_PIXELS


def test_above_old_limit_keeps_dimensions_hash_and_native_detail_crop():
    # Just above the former 16MP limit; distinctive pixels at a distant corner
    # catch accidental downsampling and incorrect crop-to-page transforms.
    original = Image.new("RGB", (4097, 4096), "white")
    original.putpixel((4001, 4002), (15, 35, 55))
    stream = BytesIO()
    original.save(stream, format="PNG")
    payload = stream.getvalue()
    registered = register_image(payload, 1)
    _, page = get_page(registered.document_id)
    assert (page.width, page.height) == original.size
    assert page.source_sha256 == sha256(payload).hexdigest()
    assert page.raster_sha256 == sha256(original.tobytes()).hexdigest()
    a = DrawingAgent(
        page,
        AgentRequest(question="Inspect", allow_gemini=True),
        Mock(),
        Mock(),
        Mock(),
    )
    assert a.original.size == original.size
    assert a.original.getpixel((4001, 4002)) == (15, 35, 55)
    assert max(a.images["overview"].metadata["sent_size"]) == 2048
    _, images = a.crop([3990, 3990, 4030, 4030])
    assert images[0].metadata["source_render_size"] == [40, 40]
    assert images[0].metadata["sent_size"] == [40, 40]
    assert images[0].metadata["canonical_bbox"] == [3990, 3990, 4030, 4030]
    assert a.original.getpixel((4001, 4002)) == (15, 35, 55)


def test_new_limit_remains_finite_and_rejects_before_decode():
    assert MAX_SOURCE_PIXELS == 150_000_000
    fake = Mock(width=15001, height=10000)
    with patch("app.Image.open") as opened:
        opened.return_value.__enter__.return_value = fake
        with pytest.raises(HTTPException) as exc:
            register_image(b"header-only-test", 1)
    assert exc.value.status_code == 413
    fake.convert.assert_not_called()


def test_algorithm_input_accepts_native_large_page_before_worker_dispatch(tmp_path):
    from planproof_feedback.pipeline import EvidencePipeline, PipelineSettings

    source = tmp_path / "large.png"
    Image.new("RGB", (4001, 4000), "white").save(source)
    checkpoint = tmp_path / "pointer.pt"
    checkpoint.write_bytes(b"LFS pointer fixture")
    config = PipelineSettings(
        "unused", "unused", checkpoint, tmp_path / "cal.json", tmp_path / "jobs"
    )
    with patch("planproof_feedback.pipeline.run_worker") as worker:
        with pytest.raises(ValueError, match="git lfs pull"):
            EvidencePipeline(config).prepare(source)
    worker.assert_not_called()
    saved = next((tmp_path / "jobs").glob("*/page.png"))
    with Image.open(saved) as image:
        assert image.size == (4001, 4000)
