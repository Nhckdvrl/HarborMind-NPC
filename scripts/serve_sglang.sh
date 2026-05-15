#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "${ROOT}/.venv-qwen35/bin/python" ]]; then
  PYTHON="${ROOT}/.venv-qwen35/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

PROFILE="${1:-base}"
CONFIG="${SGLANG_MODELS_CONFIG:-configs/sglang/models.yml}"

PYTHONPATH="${PYTHONPATH:-}:src" "${PYTHON}" -m game_npc_llm.serving.sglang_args \
  --config "${CONFIG}" \
  --profile "${PROFILE}" | bash
