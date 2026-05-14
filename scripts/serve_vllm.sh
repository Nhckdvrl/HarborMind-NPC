#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-base}"
CONFIG="${VLLM_MODELS_CONFIG:-configs/vllm/models.yml}"

PYTHONPATH="${PYTHONPATH:-}:src" python3 -m game_npc_llm.serving.vllm_args --config "${CONFIG}" --profile "${PROFILE}" | bash
