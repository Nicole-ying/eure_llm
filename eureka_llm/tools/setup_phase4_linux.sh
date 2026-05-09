#!/usr/bin/env bash
set -euo pipefail

# One-shot setup for Linux + Phase4 train mode.
# Usage:
#   bash eureka_llm/tools/setup_phase4_linux.sh
#   DEEPSEEK_API_KEY=... bash eureka_llm/tools/run_halfcheetah_phase4_step3.sh train

python -m pip install --upgrade pip
python -m pip install numpy "gymnasium[mujoco]" openai

echo "=== Dependency check ==="
python eureka_llm/tools/check_phase4_env.py || true

echo "If NOT READY due to API key, export it and run:"
echo "  DEEPSEEK_API_KEY=... bash eureka_llm/tools/run_halfcheetah_phase4_step3.sh train"
