#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export LAIA_PRE_RUN_SLEEP_SECONDS="${LAIA_PRE_RUN_SLEEP_SECONDS:-120}"

echo "[nvidia-bfcl] $(date) starting in $(pwd)"
echo "[nvidia-bfcl] sleeping ${LAIA_PRE_RUN_SLEEP_SECONDS}s before run"
sleep "${LAIA_PRE_RUN_SLEEP_SECONDS}"

.venv/bin/laia ollama nemotron-3-nano:4b \
  --benchmark bfcl \
  --resume-samples \
  --no-progress \
  --no-auto-export \
  --no-auto-push

echo "[nvidia-bfcl] $(date) finished"
