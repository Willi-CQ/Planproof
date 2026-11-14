"""Windows file contention must not discard an otherwise completed agent turn."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from planproof_feedback.artifacts import write_json


def test_transient_replace_lock_retries_without_exposing_partial_json(tmp_path):
    path = tmp_path / "progress.json"
    write_json(path, {"old": True})
    original = Path.replace
    attempts = []

    def replace(source, target):
        attempts.append(source)
        if len(attempts) < 3:
            assert json.loads(path.read_text()) == {"old": True}
            raise PermissionError("simulated Windows file lock")
        return original(source, target)

    with (
        patch.object(Path, "replace", replace),
        patch("planproof_feedback.artifacts.time.sleep") as sleep,
    ):
        write_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"new": True}
    assert len(attempts) == 3 and sleep.call_count == 2
    assert not path.with_suffix(".json.tmp").exists()


def test_persistent_lock_is_bounded_and_retains_both_versions(tmp_path):
    path = tmp_path / "progress.json"
    write_json(path, {"old": True})
    with (
        patch.object(Path, "replace", side_effect=PermissionError) as replace,
        patch("planproof_feedback.artifacts.time.sleep") as sleep,
    ):
        with pytest.raises(PermissionError):
            write_json(path, {"new": True})
    assert replace.call_count == 6 and sleep.call_count == 5
    assert json.loads(path.read_text()) == {"old": True}
    assert json.loads(path.with_suffix(".json.tmp").read_text()) == {"new": True}


def test_other_io_errors_are_not_retried(tmp_path):
    with (
        patch.object(Path, "replace", side_effect=OSError) as replace,
        patch("planproof_feedback.artifacts.time.sleep") as sleep,
    ):
        with pytest.raises(OSError):
            write_json(tmp_path / "progress.json", {"new": True})
    assert replace.call_count == 1
    sleep.assert_not_called()
