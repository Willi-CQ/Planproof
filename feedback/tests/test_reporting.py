import json
import zipfile
from pathlib import Path

import pytest

from planproof_feedback.artifacts import write_json
from planproof_feedback.benchmark import verify_pack
from planproof_feedback.images import sha256
from planproof_feedback.preservation import backup, verify_backup
from planproof_feedback.reporting import build_report, dataset_identity, import_run
from test_benchmark import pack_config  # noqa: F401 -- shared generated ZIP fixture
from test_service import answer


@pytest.fixture
def reporting_config(pack_config, tmp_path):  # noqa: F811 -- pytest fixture injection
    pack_config["report_dir"] = str(
        Path(pack_config["benchmark"]["pack_dir"]).parent / "comparison"
    )
    pack_config["audit_dir"] = str(Path(pack_config["benchmark"]["pack_dir"]).parent / "calls")
    project = tmp_path / "source-project"
    (project / "feedback/src").mkdir(parents=True)
    (project / "feedback/config.toml").write_text("# generated fixture settings\n")
    (project / "feedback/src/generated_fixture.py").write_text("# generated fixture\n")
    (project / ".env").write_text("VERDA_INFERENCE_KEY=NEVER_ARCHIVE_THIS_KEY\n")
    pack_config["config_path"] = str(project / "feedback/config.toml")
    return pack_config


def portable(config, *, count=3):
    root = Path(config["benchmark"]["pack_dir"])
    manifest = verify_pack(root)
    return {
        "schema": "planproof-predictions-v1",
        "dataset_sha256": dataset_identity(manifest),
        "model": "simulated-gemini-carbon",
        "mode": "direct",
        "image_policy": config["images"],
        "records": [
            {
                "id": c["id"],
                "input_sha256": c["input_sha256"],
                "status": "ok",
                "answer": answer(7),
                "score": {"correct": False},
                "usage": {"total_tokens": 20},
                "latency_seconds": 1.5,
            }
            for c in manifest["cases"][:count]
        ],
    }


def test_pending_report_and_model_input_export_preserve_true_state(reporting_config):
    result = build_report(reporting_config)
    destination = Path(result["report"]).parent
    data = json.loads((destination / "comparison.json").read_text())
    assert data["runs"] == [] and len(data["cases"]) == 3
    assert data["cases"][0]["reference"] == 7
    inputs = json.loads((destination / "questions-for-model.json").read_text())
    assert inputs["dataset_sha256"] == data["dataset"]["sha256"]
    assert all("answer" not in c and "reference" not in c for c in inputs["cases"])
    for case in inputs["cases"]:
        for image in case["images"]:
            assert sha256(destination / image["asset_path"]) == image["sent_sha256"]
    original = (destination / "image-index.json").read_bytes()
    build_report(reporting_config)
    assert (destination / "image-index.json").read_bytes() == original


def test_gemini_import_preserves_original_bytes_and_uses_frozen_evaluator(
    reporting_config, tmp_path
):
    source = tmp_path / "gemini-carbon-original.json"
    value = portable(reporting_config)
    value["records"][1]["answer"] = answer(8)
    write_json(source, value)
    root = Path(reporting_config["benchmark"]["pack_dir"]).parent
    destination = root / "runs/gemini-fixture"
    result = import_run(
        reporting_config, source, label="Gemini Carbon (simulated fixture)", destination=destination
    )
    assert result["records"] == 3
    assert (destination / "original-import.json").read_bytes() == source.read_bytes()
    report = json.loads((destination / "report.json").read_text())
    assert report["metrics"]["overall"]["correct"] == 2  # supplied scores were deliberately false
    built = build_report(reporting_config)
    comparison = json.loads(Path(built["data"]).read_text())
    assert comparison["runs"][0]["status"] == "complete"
    assert comparison["runs"][0]["label"] == "Gemini Carbon (simulated fixture)"
    with pytest.raises(ValueError, match="never overwritten"):
        import_run(reporting_config, source, label="Other label", destination=destination)


@pytest.mark.parametrize("change", ["dataset", "drawing", "duplicate", "unknown"])
def test_comparison_rejects_mismatched_or_duplicate_cases(reporting_config, tmp_path, change):
    value = portable(reporting_config)
    if change == "dataset":
        value["dataset_sha256"] = "different-set"
    if change == "drawing":
        value["records"][0]["input_sha256"] = "different-image"
    if change == "duplicate":
        value["records"].append(value["records"][0])
    if change == "unknown":
        value["records"][0]["id"] = "unknown-case"
    source = tmp_path / "different.json"
    write_json(source, value)
    with pytest.raises(ValueError):
        import_run(reporting_config, source, label="Fixture", destination=tmp_path / "rejected")
    assert source.exists()


def test_partial_run_is_not_filled_with_zero_scores(reporting_config, tmp_path):
    source = tmp_path / "partial.json"
    write_json(source, portable(reporting_config, count=1))
    root = Path(reporting_config["benchmark"]["pack_dir"]).parent
    import_run(
        reporting_config, source, label="Partial simulated run", destination=root / "runs/partial"
    )
    result = build_report(reporting_config)
    data = json.loads(Path(result["data"]).read_text())
    assert data["runs"][0]["status"] == "partial"
    assert len(data["runs"][0]["records"]) == 1
    assert len(data["cases"]) == 3


def test_failed_foreign_request_cannot_score_a_stale_answer_as_correct(reporting_config, tmp_path):
    value = portable(reporting_config, count=1)
    value["records"][0]["status"] = "error"
    source = tmp_path / "failed.json"
    write_json(source, value)
    destination = Path(reporting_config["benchmark"]["pack_dir"]).parent / "runs/failed-import"
    import_run(reporting_config, source, label="Failed simulated import", destination=destination)
    report = json.loads((destination / "report.json").read_text())
    assert report["metrics"]["overall"]["correct"] == 0
    assert (destination / "original-import.json").read_bytes() == source.read_bytes()


def test_verified_backup_keeps_data_and_source_but_excludes_key(reporting_config):
    build_report(reporting_config)
    first = backup(reporting_config)
    second = backup(reporting_config)
    assert first["archive"] != second["archive"]
    assert verify_backup(Path(first["archive"]))["sha256"] == first["sha256"]
    with zipfile.ZipFile(first["archive"]) as archive:
        names = archive.namelist()
        assert any("evaluator/answers.json" in name for name in names)
        assert any("comparison/assets/" in name for name in names)
        assert any(name.startswith("registries/") for name in names)
        assert any(name.startswith("source/") for name in names)
        assert not any("backups/" in name or name.endswith("/.env") for name in names)
        assert not any(b"NEVER_ARCHIVE_THIS_KEY" in archive.read(name) for name in names)


def test_backup_sha_detects_payload_change_even_with_valid_zip_crc(reporting_config, tmp_path):
    receipt = backup(reporting_config)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(receipt["archive"]) as original, zipfile.ZipFile(tampered, "w") as target:
        for index, name in enumerate(original.namelist()):
            target.writestr(name, b"changed-payload" if index == 0 else original.read(name))
    with pytest.raises(ValueError, match="SHA256"):
        verify_backup(tampered)
