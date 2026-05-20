#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMAFACTORY=(llamafactory-cli)
if [[ -x "${ROOT}/.venv-qwen3/bin/python" ]]; then
  LLAMAFACTORY=("${ROOT}/.venv-qwen3/bin/python" -m llamafactory.cli)
elif [[ -x "${ROOT}/.venv-qwen35/bin/python" ]]; then
  LLAMAFACTORY=("${ROOT}/.venv-qwen35/bin/python" -m llamafactory.cli)
fi

CONFIG="${1:-configs/llamafactory/qwen3_4b_npc_sft_export.yaml}"
"${LLAMAFACTORY[@]}" export "${CONFIG}"
