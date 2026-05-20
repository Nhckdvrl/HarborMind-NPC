#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
RELOAD="${RELOAD:-0}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export NPC_MODEL_BASE_URL="${NPC_MODEL_BASE_URL:-http://127.0.0.1:30000/v1}"
export NPC_MODEL_NAME="${NPC_MODEL_NAME:-qwen3-4b-npc}"
export NPC_MODEL_MAX_TOKENS="${NPC_MODEL_MAX_TOKENS:-128}"
export NPC_MODEL_TEMPERATURE="${NPC_MODEL_TEMPERATURE:-0.2}"
export NPC_MODEL_TIMEOUT="${NPC_MODEL_TIMEOUT:-120}"
export NPC_WORLD_PATH="${NPC_WORLD_PATH:-${ROOT}/data/worlds/kisaragi_harbor/world.json}"

args=(
  -m uvicorn game_npc_llm.product.server:app
  --host "${HOST}"
  --port "${PORT}"
)

if [[ "${RELOAD}" == "1" ]]; then
  args+=(--reload)
fi

exec "${ROOT}/.venv-qwen3/bin/python" "${args[@]}"
