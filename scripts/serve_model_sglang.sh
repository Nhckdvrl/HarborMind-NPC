#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${ROOT}/outputs/alignment/qwen3_4b_npc_orpo_merged}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-npc}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -x "${ROOT}/.venv-sglang/bin/python" ]]; then
  echo "Missing ${ROOT}/.venv-sglang/bin/python. Install or activate the SGLang environment first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES
export PATH="${ROOT}/.venv-sglang/bin:${PATH}"

exec "${ROOT}/.venv-sglang/bin/python" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --context-length "${CONTEXT_LENGTH:-4096}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.5}" \
  --skip-server-warmup \
  --disable-cuda-graph
