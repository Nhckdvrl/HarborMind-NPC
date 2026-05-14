#!/usr/bin/env bash
set -euo pipefail

: "${POLICY_MODEL:=outputs/sft/qwen3_32b_npc_qlora}"
: "${PROMPTS:=data/processed/grpo_prompts.jsonl}"
: "${OUTPUT_DIR:=outputs/grpo/qwen3_32b_npc}"
: "${REWARD_SERVER:=http://127.0.0.1:8091/reward}"
: "${NUM_GPUS:=4}"

slime \
  --model "${POLICY_MODEL}" \
  --train-data "${PROMPTS}" \
  --output-dir "${OUTPUT_DIR}" \
  --advantage-estimator grpo \
  --num-generations 8 \
  --rollout-temperature 0.8 \
  --rollout-top-p 0.95 \
  --max-prompt-length 2048 \
  --max-response-length 512 \
  --reward-url "${REWARD_SERVER}" \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-6 \
  --num-train-epochs 1 \
  --save-steps 100 \
  --logging-steps 5 \
  --bf16 \
  --tensor-parallel-size "${NUM_GPUS}"
