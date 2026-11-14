from pathlib import Path

import mvp


def test_launch_contract_and_separate_environments():
    command = mvp.launch_command(8123)
    assert command[-2:] == ["--port", "8123"]
    assert command[command.index("--env-file") + 1] == str(mvp.ROOT / ".env")
    assert command[command.index("--gemini-env-file") + 1] == str(mvp.ROOT / ".env")
    assert len({mvp.interpreter(n) for n in mvp.install_commands()}) == 3
    assert all(
        mvp.interpreter(n).is_relative_to(mvp.ROOT / ".artifacts")
        for n in mvp.install_commands()
    )
    assert Path(command[1]).is_file()


def test_setup_installs_the_host_feedback_and_both_real_workers():
    commands = mvp.install_commands()
    assert str(mvp.ROOT / "backend/requirements.txt") in commands["host"]
    assert str(mvp.ROOT / "feedback") + "[test]" in commands["host"]
    assert commands["ml"] == ["-e", str(mvp.ROOT / "ml")]
    assert commands["geometry"] == ["-e", str(mvp.ROOT / "deterministic")]
