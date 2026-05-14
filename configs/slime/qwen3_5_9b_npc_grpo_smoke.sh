#!/usr/bin/env bash
set -euo pipefail

: "${POLICY_MODEL:=outputs/sft/qwen3_5_9b_npc_lora_smoke}"
: "${PROMPTS:=data/rl/grpo_prompts.jsonl}"
: "${OUTPUT_DIR:=outputs/grpo/qwen3_5_9b_npc_smoke}"
: "${REWARD_SERVER:=http://127.0.0.1:8091/reward}"

slime \
  --model "${POLICY_MODEL}" \
  --train-data "${PROMPTS}" \
  --output-dir "${OUTPUT_DIR}" \
  --advantage-estimator grpo \
  --num-generations 4 \
  --max-steps 10 \
  --rollout-temperature 0.8 \
  --max-prompt-length 1024 \
  --max-response-length 256 \
  --reward-url "${REWARD_SERVER}" \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-5 \
  --save-steps 10 \
  --logging-steps 1 \
  --bf16
