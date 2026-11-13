#!/usr/bin/env bash
set -euo pipefail
PLANPROOF_REPO="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ $# -lt 1 ]]; then
  printf 'Usage: bash ml/scripts/gpu_prepare.sh /absolute/persistent/planproof-data\n' >&2
  exit 2
fi
PLANPROOF_DATA_ROOT="$1"
if [[ "$PLANPROOF_DATA_ROOT" != /* ]]; then
  printf 'Use an absolute persistent data path.\n' >&2
  exit 2
fi
cd "$PLANPROOF_REPO"
planproof-ml doctor --require-gpu --smoke
planproof-ml fetch --source cubicasa5k --data-root "$PLANPROOF_DATA_ROOT" --extract
planproof-ml prepare --source cubicasa5k --raw "$PLANPROOF_DATA_ROOT/extracted/cubicasa5k" --out "$PLANPROOF_DATA_ROOT/processed"
planproof-ml audit --manifest "$PLANPROOF_DATA_ROOT/processed/manifest.jsonl" --deep --near --out "$PLANPROOF_DATA_ROOT/processed/audit.json"
planproof-ml inspect --manifest "$PLANPROOF_DATA_ROOT/processed/manifest.jsonl" --out "$PLANPROOF_DATA_ROOT/inspection" --per-source 12
planproof-ml preflight --config ml/configs/baseline.yaml --manifest "$PLANPROOF_DATA_ROOT/processed/manifest.jsonl"
printf '\nPreparation completed. Review audit.json and inspection overlays before launching:\n'
printf 'planproof-ml train --config ml/configs/baseline.yaml --manifest %q --run %q\n' \
  "$PLANPROOF_DATA_ROOT/processed/manifest.jsonl" "$PLANPROOF_DATA_ROOT/runs/baseline"
