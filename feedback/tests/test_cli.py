import json

import pytest
from PIL import Image

from planproof_feedback import cli
from planproof_feedback.service import FeedbackService
from test_service import Backend, IMAGE_CONFIG, answer, review


@pytest.mark.parametrize("command", ["answer", "feedback", "self-feedback"])
def test_user_commands_write_valid_output(tmp_path, monkeypatch, command):
    image = tmp_path / "drawing.png"
    Image.new("RGB", (200, 100), "white").save(image)
    config = {"images": IMAGE_CONFIG}
    backend = Backend([answer() if command == "answer" else review()])
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "create_service", lambda _: FeedbackService(backend, backend))
    destination = tmp_path / "result.json"
    arguments = [
        command,
        "--image",
        str(image),
        "--question",
        "How many doors?",
        "--out",
        str(destination),
    ]
    if command != "answer":
        arguments += ["--candidate", "Three doors"]
    assert cli.main(arguments) == 0
    result = json.loads(destination.read_text())
    assert result["status"] == "ok"
    assert (
        result["role"]
        == {"answer": "answer", "feedback": "feedback", "self-feedback": "self_feedback"}[command]
    )


def test_missing_key_points_to_one_env_file_without_network(tmp_path, monkeypatch, capsys):
    path = tmp_path / ".env"
    path.write_text("VERDA_INFERENCE_KEY=\n")
    monkeypatch.delenv("VERDA_INFERENCE_KEY", raising=False)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: {
            "env_file": str(path),
            "routes": {"feedback": "nemotron"},
            "providers": {"nemotron": {"api_key_env": "VERDA_INFERENCE_KEY"}},
        },
    )
    assert cli.main(["check"]) == 2
    error = capsys.readouterr().err
    assert str(path) in error and "VERDA_INFERENCE_KEY=" in error
