#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "${ROOT}/.venv-qwen35/bin/llamafactory-cli" ]]; then
  export PATH="${ROOT}/.venv-qwen35/bin:${PATH}"
fi

CONFIG="${1:-configs/llamafactory/qwen3_5_27b_npc_sft_export.yaml}"
llamafactory-cli export "${CONFIG}"
