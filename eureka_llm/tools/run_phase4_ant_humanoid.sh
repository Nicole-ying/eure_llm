#!/usr/bin/env bash
set -euo pipefail

# Phase-4 scaffold runner for Ant-v4 and Humanoid-v4.
# Usage:
#   bash eureka_llm/tools/run_phase4_ant_humanoid.sh dry-run
#   DEEPSEEK_API_KEY=... bash eureka_llm/tools/run_phase4_ant_humanoid.sh train

MODE="${1:-dry-run}"  # dry-run | train
ROOT="eureka_llm"

run_one() {
  local ENV_NAME="$1"         # Ant-v4 | Humanoid-v4
  local ENV_DIR="$2"          # eureka_llm/envs/<env>
  local EXPLORE_JSON="$3"     # eureka_llm/explorations/<env>.json
  local CFG="$4"              # eureka_llm/configs/<cfg>.yaml

  echo "==== Phase4 scaffold for ${ENV_NAME} ===="
  if [[ ! -f "${ENV_DIR}/env.py" || ! -f "${ENV_DIR}/step.py" ]]; then
    echo "[skip] Missing ${ENV_DIR}/env.py or step.py."
    echo "       Please add environment wrappers first (same layout as HalfCheetah-v4)."
    return 0
  fi

  if [[ ! -f "${EXPLORE_JSON}" ]]; then
    echo "[skip] Missing exploration file: ${EXPLORE_JSON}"
    echo "       Generate exploration JSON before running pipeline."
    return 0
  fi

  if [[ "${MODE}" == "dry-run" ]]; then
    DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-dummy}" python "${ROOT}/framework/pipeline.py" \
      --mode full \
      --env-dir "${ENV_DIR}" \
      --exploration "${EXPLORE_JSON}" \
      --config "${CFG}" \
      --dry-run
  else
    : "${DEEPSEEK_API_KEY:?Please export DEEPSEEK_API_KEY for train mode}"
    python "${ROOT}/framework/pipeline.py" \
      --mode full \
      --env-dir "${ENV_DIR}" \
      --exploration "${EXPLORE_JSON}" \
      --config "${CFG}"
  fi
}

run_one "Ant-v4" \
  "${ROOT}/envs/Ant-v4" \
  "${ROOT}/explorations/Ant-v4.json" \
  "${ROOT}/configs/ppo_1M_ant.yaml"

run_one "Humanoid-v4" \
  "${ROOT}/envs/Humanoid-v4" \
  "${ROOT}/explorations/Humanoid-v4.json" \
  "${ROOT}/configs/ppo_1M_humanoid.yaml"

echo "Phase-4 Ant/Humanoid scaffold run complete."
