param(
    [Parameter(Position = 0)]
    [string]$Action = "check",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Options
)
$ErrorActionPreference = "Stop"
$pythonPath = Join-Path $PSScriptRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Install first: uv venv .venv; uv pip install --python .venv/Scripts/python.exe -e './feedback[test]'"
}
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot "feedback/src"
    & $pythonPath -m planproof_feedback --config (Join-Path $PSScriptRoot "feedback/config.toml") $Action @Options
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
