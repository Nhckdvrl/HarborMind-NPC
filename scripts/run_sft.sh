#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/axolotl/qwen3_8b_npc_qlora_smoke.yml}"
accelerate launch -m axolotl.cli.train "${CONFIG}"
