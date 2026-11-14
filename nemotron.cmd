@echo off
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Install first: uv venv .venv then uv pip install --python .venv/Scripts/python.exe -e ./feedback[test]
  exit /b 2
)
"%~dp0.venv\Scripts\python.exe" -m planproof_feedback --config "%~dp0feedback\config.toml" %*
exit /b %errorlevel%
