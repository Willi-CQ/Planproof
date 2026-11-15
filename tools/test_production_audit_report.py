"""The review must bind to real records, preserve pixels and stay offline."""

from hashlib import sha256
import json

from PIL import Image
import pytest

from build_production_audit_report import build


@pytest.fixture
def review(tmp_path):
    audit, baseline = tmp_path / "audit", tmp_path / "baseline"
    audit.mkdir()
    baseline.mkdir()
    historical = baseline / "report.json"
    historical.write_text("{}")
    translations = tmp_path / "translations.json"
    translations.write_text("{}")
    rows = []
    for cid in ("one", "two"):
        (audit / cid).mkdir()
        (baseline / cid).mkdir()
        img = Image.new("RGB", (40, 30), "white")
        img.save(baseline / cid / "page.png")
        answer = {"status": "answered", "answer": 2, "unit": "count"}
        (baseline / cid / "auto.json").write_text(json.dumps({"answer": answer}))
        (audit / cid / "record.json").write_text(
            json.dumps(
                {
                    "run": {
                        "result": {"answer": answer},
                        "events": [
                            {
                                "stage": "specialist",
                                "state": "completed",
                                "question": "</script><script>alert(1)</script>",
                                "private_reasoning": "do not export",
                                "api_key": "do not export",
                            }
                        ],
                    }
                }
            )
        )
        rows.append(
            {
                "id": cid,
                "before": {"answer": answer, "seconds": 10},
                "answer": answer,
                "raster_sha256": sha256(img.tobytes()).hexdigest(),
                "raster_matches": True,
                "language": "en",
                "question": "How many?",
            }
        )
    report = {
        "binding": {
            "case_ids": ["one", "two"],
            "condition": "synthetic test",
            "baseline_report_sha256": sha256(historical.read_bytes()).hexdigest(),
        },
        "summary": {"cases": 2},
        "cases": rows,
    }
    return audit, baseline, translations, tmp_path / "review.html", report


def render(review):
    audit, baseline, translations, output, report = review
    (audit / "report.json").write_text(json.dumps(report))
    return build(audit, baseline, translations, output)


def test_review_preserves_sources_and_only_exports_public_activity(review):
    assert render(review)["cases"] == 2
    html = review[3].read_text(encoding="utf-8")
    payload = json.loads(
        html.split('<script id="data" type="application/json">')[1].split("</script>")[
            0
        ]
    )
    assert len(payload["cases"]) == 2
    assert payload["cases"][0]["image"].startswith("data:image/png;base64,")
    assert payload["cases"][0]["activity"][0]["question"].startswith("</script>")
    assert "do not export" not in html
    assert "<script>alert(1)" not in html


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "raster", "trace", "history", "translation"]
)
def test_review_rejects_unmatched_artifacts(review, mutation):
    audit, baseline, translations, output, report = review
    if mutation == "duplicate":
        report["cases"][1] = report["cases"][0]
    elif mutation == "missing":
        report["cases"].pop()
    elif mutation == "raster":
        report["cases"][0]["raster_sha256"] = "changed"
    elif mutation == "trace":
        (audit / "one/record.json").write_text('{"run": {"result": {"answer": {}}}}')
    elif mutation == "history":
        (baseline / "report.json").write_text('{"changed": true}')
    else:
        report["cases"][0]["language"] = "fi"
        translations.write_text(
            json.dumps(
                {"one": {"original": "Different question", "english": "How many?"}}
            )
        )
    with pytest.raises(ValueError):
        render(review)
    assert not output.exists()
