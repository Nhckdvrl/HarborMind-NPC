#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/slime/qwen3_8b_npc_grpo_smoke.sh}"
bash "${CONFIG}"
