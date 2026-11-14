import hashlib
import json

import pytest

from extend_case_review import build


def fixture(tmp_path):
    from PIL import Image

    image = tmp_path / "plan.png"
    Image.new("RGB", (10, 8), "white").save(image)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    case = {
        "id": digest,
        "number": 1,
        "input_path": image.name,
        "input_sha256": digest,
        "question": "Count <doors>?",
        "original_size": [10, 8],
    }
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {"source_pack": str(tmp_path), "cases": [case], "limitation": "test only"}
        )
    )
    base = tmp_path / "review.html"
    base.write_text(
        '<html><body><main>Frozen</main><script id="review-data" type="application/json">{"unchanged":true}</script></body></html>'
    )
    results = tmp_path / "results"
    results.mkdir()
    return base, inventory, results, case


def test_extension_preserves_frozen_payload_pending_state_and_is_idempotent(tmp_path):
    base, inventory, results, _ = fixture(tmp_path)
    assert build(base, inventory, results, base) == {"cases": 1, "completed": 0}
    first = base.read_text(encoding="utf-8")
    assert '{"unchanged":true}' in first
    assert '"result": null' in first
    assert "Count \\u003cdoors>?" in first
    build(base, inventory, results, base)
    assert base.read_text(encoding="utf-8") == first


@pytest.mark.parametrize("bad", ["source", "question", "upload"])
def test_rejects_changed_sources_or_mismatched_inference(tmp_path, bad):
    base, inventory, results, case = fixture(tmp_path)
    record = {
        "id": case["id"],
        "question": case["question"],
        "result": {"source": {"source_sha256": case["id"]}},
    }
    if bad == "source":
        (tmp_path / "plan.png").write_bytes(b"changed")
    elif bad == "question":
        record["question"] = "Another question"
    else:
        record["result"]["source"]["source_sha256"] = "wrong-upload"
    (results / "case-1.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        build(base, inventory, results, base)
    assert "HARD_REVIEW_START" not in base.read_text()
