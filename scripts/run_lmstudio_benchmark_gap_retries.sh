#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export LAIA_PRE_RUN_SLEEP_SECONDS="${LAIA_PRE_RUN_SLEEP_SECONDS:-120}"

run_step() {
  local label="$1"
  shift
  echo "[$label] $(date) sleeping ${LAIA_PRE_RUN_SLEEP_SECONDS}s before run"
  sleep "${LAIA_PRE_RUN_SLEEP_SECONDS}"
  echo "[$label] $(date) starting"
  "$@"
  echo "[$label] $(date) finished"
}

echo "[lmstudio-retries] $(date) starting in $(pwd)"

run_step "olmo" \
  .venv/bin/laia lmstudio olmo-3-7b-instruct@q4_k_m \
    --benchmark ifbench,bfcl,mbpp \
    --resume-samples \
    --no-progress \
    --no-auto-export \
    --no-auto-push

if [[ -f results/STOP_AFTER_OLMO ]]; then
  echo "[lmstudio-retries] $(date) STOP_AFTER_OLMO marker found; stopping before SmolLM3"
  exit 0
fi

run_step "smollm3" \
  .venv/bin/laia lmstudio smollm3-3b@q4_k_m \
    --benchmark global-mmlu-lite,rgb \
    --resume-samples \
    --no-progress \
    --no-auto-export \
    --no-auto-push

echo "[lmstudio-retries] $(date) finished"
