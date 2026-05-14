#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8091}"

PYTHONPATH="${PYTHONPATH:-}:src" uvicorn game_npc_llm.rewards.server:app --host "${HOST}" --port "${PORT}"
