#!/usr/bin/env bash
# demo.sh — start / stop / status the full NPC demo stack
# Usage:
#   ./scripts/demo.sh start    # start SGLang + web server + tunnel
#   ./scripts/demo.sh stop     # stop everything
#   ./scripts/demo.sh status   # show what's running
#   ./scripts/demo.sh logs     # tail all logs
#   ./scripts/demo.sh restart  # stop then start

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── configurable ────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/home/xiang/models/qwen3-4b-npc}"
MODEL_NAME="${MODEL_NAME:-qwen3-4b-npc}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
WEB_PORT="${WEB_PORT:-8000}"
SGLANG_LOG="${SGLANG_LOG:-/tmp/sglang.log}"
WEB_LOG="${WEB_LOG:-/tmp/demo_server.log}"
TUNNEL_LOG="${TUNNEL_LOG:-/tmp/tunnel.log}"
VENV="${ROOT}/.venv-sglang"
# ────────────────────────────────────────────────────────────────────────────

_red()   { printf '\033[31m%s\033[0m\n' "$*"; }
_green() { printf '\033[32m%s\033[0m\n' "$*"; }
_cyan()  { printf '\033[36m%s\033[0m\n' "$*"; }
_bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

cmd_start() {
  _bold "=== Starting NPC demo stack ==="

  # 1. SGLang
  if pgrep -f "sglang.launch_server" > /dev/null; then
    _green "[SGLang]  already running"
  else
    _cyan "[SGLang]  launching on port ${SGLANG_PORT}..."
    PATH="${VENV}/bin:${PATH}" \
    "${VENV}/bin/python" -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --served-model-name "${MODEL_NAME}" \
      --host 0.0.0.0 \
      --port "${SGLANG_PORT}" \
      --trust-remote-code \
      --dtype bfloat16 \
      > "${SGLANG_LOG}" 2>&1 &

    _cyan "[SGLang]  waiting for ready (up to 60s)..."
    for i in $(seq 1 60); do
      if curl -sf "http://localhost:${SGLANG_PORT}/health" > /dev/null 2>&1; then
        _green "[SGLang]  ready"
        break
      fi
      sleep 1
      if [[ $i -eq 60 ]]; then
        _red "[SGLang]  timed out — check ${SGLANG_LOG}"
        exit 1
      fi
    done
  fi

  # 2. Web server
  if pgrep -f "uvicorn game_npc_llm" > /dev/null; then
    _green "[Web]     already running"
  else
    _cyan "[Web]     launching on port ${WEB_PORT}..."
    PYTHONPATH="${ROOT}/src" \
    NPC_MODEL_BASE_URL="http://localhost:${SGLANG_PORT}/v1" \
    NPC_MODEL_NAME="${MODEL_NAME}" \
    "${VENV}/bin/uvicorn" game_npc_llm.product.server:app \
      --host 0.0.0.0 \
      --port "${WEB_PORT}" \
      > "${WEB_LOG}" 2>&1 &

    sleep 3
    if curl -sf "http://localhost:${WEB_PORT}/health" > /dev/null 2>&1; then
      _green "[Web]     ready"
    else
      _red "[Web]     failed to start — check ${WEB_LOG}"
      exit 1
    fi
  fi

  # 3. localhost.run tunnel
  if pgrep -f "nokey@localhost.run" > /dev/null; then
    _green "[Tunnel]  already running"
    _cyan "[Tunnel]  URL: $(grep -o '[a-z0-9]*\.lhr\.life' "${TUNNEL_LOG}" | tail -1)"
  else
    _cyan "[Tunnel]  opening reverse tunnel..."
    : > "${TUNNEL_LOG}"  # clear old log
    ssh -o StrictHostKeyChecking=no \
        -o ServerAliveInterval=30 \
        -R "80:localhost:${WEB_PORT}" \
        nokey@localhost.run \
        > "${TUNNEL_LOG}" 2>&1 &

    _cyan "[Tunnel]  waiting for URL..."
    for i in $(seq 1 15); do
      url=$(grep -o '[a-z0-9]*\.lhr\.life' "${TUNNEL_LOG}" 2>/dev/null | tail -1)
      if [[ -n "${url}" ]]; then
        _green "[Tunnel]  ready"
        echo ""
        _bold "  Open on your Mac: https://${url}"
        echo ""
        break
      fi
      sleep 1
      if [[ $i -eq 15 ]]; then
        _red "[Tunnel]  timed out — check ${TUNNEL_LOG}"
      fi
    done
  fi

  _bold "=== Stack is up ==="
}

cmd_stop() {
  _bold "=== Stopping NPC demo stack ==="

  pkill -f "nokey@localhost.run" 2>/dev/null && _green "[Tunnel]  stopped" || _cyan "[Tunnel]  was not running"
  pkill -f "uvicorn game_npc_llm"  2>/dev/null && _green "[Web]     stopped" || _cyan "[Web]     was not running"
  pkill -f "sglang.launch_server"  2>/dev/null && _green "[SGLang]  stopped" || _cyan "[SGLang]  was not running"

  _bold "=== Done ==="
}

cmd_status() {
  _bold "=== NPC demo stack status ==="

  if pgrep -f "sglang.launch_server" > /dev/null; then
    _green "[SGLang]  running  (port ${SGLANG_PORT})"
  else
    _red   "[SGLang]  stopped"
  fi

  if pgrep -f "uvicorn game_npc_llm" > /dev/null; then
    _green "[Web]     running  (port ${WEB_PORT})"
  else
    _red   "[Web]     stopped"
  fi

  if pgrep -f "nokey@localhost.run" > /dev/null; then
    url=$(grep -o '[a-z0-9]*\.lhr\.life' "${TUNNEL_LOG}" 2>/dev/null | tail -1)
    _green "[Tunnel]  running  — https://${url:-<see ${TUNNEL_LOG}>}"
  else
    _red   "[Tunnel]  stopped"
  fi
}

cmd_logs() {
  _bold "=== Tailing all logs (Ctrl+C to exit) ==="
  tail -f "${SGLANG_LOG}" "${WEB_LOG}" "${TUNNEL_LOG}"
}

case "${1:-}" in
  start)   cmd_start   ;;
  stop)    cmd_stop    ;;
  status)  cmd_status  ;;
  logs)    cmd_logs    ;;
  restart) cmd_stop; echo; cmd_start ;;
  *)
    echo "Usage: $0 {start|stop|status|logs|restart}"
    exit 1
    ;;
esac
