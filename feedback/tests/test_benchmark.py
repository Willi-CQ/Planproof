import csv
import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from planproof_feedback.archive import RangeArchive
from planproof_feedback.client import ProviderError
from planproof_feedback.benchmark import prepare, run, safe_path, select_cases, verify_pack
from planproof_feedback.service import FeedbackService
from test_service import Backend, IMAGE_CONFIG, answer


class RangeResponse(io.BytesIO):
    status = 206

    def __init__(self, data, start, end, total):
        super().__init__(data)
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}", "ETag": '"frozen"'}


class RangeOpener:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def open(self, request, timeout):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        self.calls.append((start, end))
        return RangeResponse(self.data[start : end + 1], start, end, len(self.data))


@pytest.fixture
def pack_config(tmp_path):
    image_bytes = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(image_bytes, format="PNG")
    content = io.BytesIO()
    names = [f"junction-dataset/data/demo/page-{index}.png" for index in range(5)]
    with zipfile.ZipFile(content, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("junction-dataset/README.md", "SIMULATED dataset notice")
        archive.writestr(
            "junction-dataset/data/demo/LICENSE.md", "SIMULATED test fixture; no partner data"
        )
        for name in names:
            archive.writestr(name, image_bytes.getvalue())
    inventory = tmp_path / "inventory.csv"
    with inventory.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Path"])
        writer.writeheader()
        writer.writerows({"Path": name} for name in names)
    questions = tmp_path / "eval-reference.jsonl"
    questions.write_text(
        "\n".join(
            json.dumps(
                {
                    "path": name.removeprefix("junction-dataset/"),
                    "question": f"How many test items on {index}?",
                    "answer": 7,
                    "type": "count",
                    "unit": "count",
                }
            )
            for index, name in enumerate(names)
        )
    )
    config = {
        "benchmark": {
            "pack_dir": str(tmp_path / "pack"),
            "question_file": str(questions),
            "inventory_file": str(inventory),
            "sample_size": 3,
            "seed": "fixed-fixture-seed",
            "archive_url": "https://example.test/archive.zip",
            "max_download_bytes": 1000000,
        },
        "images": IMAGE_CONFIG,
        "routes": {"feedback": "nemotron", "self_feedback": "nemotron"},
        "providers": {"nemotron": {"model": "simulated", "max_tokens": 100}},
    }
    archive_bytes = content.getvalue()

    def factory(url, budget):
        return RangeArchive(url, budget=budget, opener=RangeOpener(archive_bytes))

    prepare(config, archive_factory=factory, progress=lambda _: None)
    return config


def test_sampling_is_answer_independent_and_final_partition_refused(pack_config, tmp_path):
    settings = pack_config["benchmark"]
    path = Path(settings["question_file"])
    inventory = Path(settings["inventory_file"])
    first, _, stats = select_cases(path, inventory, count=3, seed="same")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["answer"] = "secret-reference-answer-changed"
    path.write_text("\n".join(json.dumps(row) for row in reversed(rows)))
    second, _, _ = select_cases(path, inventory, count=3, seed="same")
    assert [case["id"] for case in first] == [case["id"] for case in second]
    assert all("answer" not in case for case in second)
    assert stats["eligible_unique_questions"] == 5
    with pytest.raises(ValueError, match="sealed"):
        select_cases(tmp_path / "eval-final.jsonl", inventory, count=1, seed="same")


def test_pack_preparation_hashes_and_no_complete_archive(pack_config):
    settings = pack_config["benchmark"]
    root = Path(settings["pack_dir"])
    manifest = verify_pack(root)
    assert len(manifest["cases"]) == 3
    assert "answer" not in manifest["cases"][0]
    assert (root / "notices/demo/LICENSE.md").exists()
    assert manifest["range_bytes_downloaded"] < settings["max_download_bytes"]
    assert prepare(pack_config)["status"] == "already_prepared"
    path = root / manifest["cases"][0]["input_path"]
    path.write_bytes(b"tampered image")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_pack(root)


def test_model_stage_never_reads_reference_answers_and_resume_is_bound(
    pack_config, tmp_path, monkeypatch
):
    backend = Backend([answer(7), answer(7), answer(7)])
    service = FeedbackService(backend, backend)
    original_read = Path.read_text
    reads = []

    def read(path, *args, **kwargs):
        if path.name == "answers.json":
            reads.append(len(backend.messages))
            assert len(backend.messages) == 3
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    destination = tmp_path / "run"
    result = run(pack_config, service, run_dir=destination, progress=lambda _: None)
    assert result["metrics"]["overall"]["accuracy"] == 1
    assert result["metrics"]["overall"]["tokens"]["total_tokens"] == 9
    assert reads == [3]
    assert "reference-answer" not in json.dumps(backend.messages)
    assert "evaluator" not in json.dumps(backend.messages)
    assert all(
        "answer" not in json.loads(message[1]["content"][0]["text"]) for message in backend.messages
    )
    run(pack_config, service, run_dir=destination, resume=True, progress=lambda _: None)
    assert len(backend.messages) == 3
    pack_config["providers"]["nemotron"]["max_tokens"] = 999
    with pytest.raises(ValueError, match="Resume refused"):
        run(pack_config, service, run_dir=destination, resume=True)


def test_invalid_and_transport_failures_stay_in_denominator(pack_config, tmp_path):
    class Failures:
        count = 0

        def answer(self, *args, **kwargs):
            self.count += 1
            if self.count == 1:
                raise RuntimeError("simulated unavailable endpoint")
            return {"status": "invalid_output", "result": None, "completion": {"usage": {}}}

    result = run(pack_config, Failures(), run_dir=tmp_path / "errors", progress=lambda _: None)
    overall = result["metrics"]["overall"]
    assert overall["scorable"] == 3 and overall["accuracy"] == 0 and overall["failures"] == 3


def test_range_archive_refuses_full_response_and_enforces_budget():
    opener = RangeOpener(b"0123456789")
    reader = RangeArchive("https://example.test/a.zip", opener=opener, budget=5)
    with pytest.raises(ValueError, match="budget"):
        reader.read(10)

    class IgnoresRange:
        def open(self, request, timeout):
            response = RangeResponse(b"all", 0, 2, 3)
            response.status = 200
            return response

    with pytest.raises(ValueError, match="full download refused"):
        RangeArchive("https://example.test/a.zip", opener=IgnoresRange())


def test_auth_failure_stops_without_spending_the_whole_sample(pack_config, tmp_path):
    class Unauthorized:
        calls = 0

        def answer(self, *args, **kwargs):
            self.calls += 1
            raise ProviderError("Provider HTTP 401", status_code=401)

    service = Unauthorized()
    destination = tmp_path / "unauthorized"
    with pytest.raises(RuntimeError, match="Benchmark stopped"):
        run(pack_config, service, run_dir=destination)
    assert service.calls == 1
    assert (destination / "last-error.json").exists()
    assert not (destination / "predictions.json").exists()
    backend = Backend([answer(7), answer(7), answer(7)])
    result = run(
        pack_config,
        FeedbackService(backend, backend),
        run_dir=destination,
        resume=True,
        progress=lambda _: None,
    )
    assert result["metrics"]["overall"]["cases"] == 3


@pytest.mark.parametrize(
    "relative", ["../secret", "/absolute", "C:/secret", "nested/../../escape", "..\\secret"]
)
def test_artifact_paths_are_confined(tmp_path, relative):
    with pytest.raises(ValueError):
        safe_path(tmp_path, relative)
