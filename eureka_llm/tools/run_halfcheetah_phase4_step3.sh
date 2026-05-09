#!/usr/bin/env bash
set -euo pipefail

# Phase-4 step3 validation pipeline for HalfCheetah-v4.
# Usage:
#   bash eureka_llm/tools/run_halfcheetah_phase4_step3.sh dry-run
#   DEEPSEEK_API_KEY=... bash eureka_llm/tools/run_halfcheetah_phase4_step3.sh train

MODE="${1:-dry-run}"  # dry-run | train

ROOT="eureka_llm"
ENV_DIR="$ROOT/envs/HalfCheetah-v4"
EXPLORE_JSON="$ROOT/explorations/HalfCheetah-v4.json"
CFG="$ROOT/configs/ppo_1M_halfcheetah.yaml"

echo "[1/3] Generate real exploration JSON"
python "$ROOT/tools/check_phase4_env.py" || true
python "$ROOT/tools/generate_halfcheetah_exploration.py" --output "$EXPLORE_JSON"

if [[ "$MODE" == "dry-run" ]]; then
  echo "[2/3] Round0+iterate dry-run (no LLM call, no training)"
  DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-dummy}" python "$ROOT/framework/pipeline.py" \
    --mode full \
    --env-dir "$ENV_DIR" \
    --exploration "$EXPLORE_JSON" \
    --config "$CFG" \
    --dry-run
else
  echo "[2/3] Full run (WILL call LLM and run training)"
  : "${DEEPSEEK_API_KEY:?Please export DEEPSEEK_API_KEY for train mode}"
  python "$ROOT/framework/pipeline.py" \
    --mode full \
    --env-dir "$ENV_DIR" \
    --exploration "$EXPLORE_JSON" \
    --config "$CFG"
fi

echo "[3/3] Done"
