#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${HF_CHECKPOINT:=outputs/sft/qwen3_5_27b_npc_merged}"
: "${REF_LOAD:=outputs/slime/qwen3_5_27b_npc_ref_torch_dist}"
: "${LOAD_DIR:=outputs/grpo/qwen3_5_27b_npc_grpo_megatron}"
: "${SAVE_DIR:=outputs/grpo/qwen3_5_27b_npc_grpo_megatron}"
: "${PROMPTS:=${ROOT}/data/rl/grpo_prompts.jsonl}"
: "${ACTOR_GPUS:=4}"
: "${ROLLOUT_GPUS:=4}"
: "${ROLLOUT_GPUS_PER_ENGINE:=4}"
: "${TP_SIZE:=2}"
: "${CP_SIZE:=2}"
: "${SLIME_MODEL_ARGS_SCRIPT:=}"

if [[ -n "${SLIME_MODEL_ARGS_SCRIPT}" ]]; then
  # shellcheck source=/dev/null
  source "${SLIME_MODEL_ARGS_SCRIPT}"
else
  echo "Set SLIME_MODEL_ARGS_SCRIPT to slime's qwen3.5 27B Megatron model-args script." >&2
  exit 2
fi

COMMON_ARGS=(
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_LOAD}"
  --load "${LOAD_DIR}"
  --save "${SAVE_DIR}"
  --save-interval 100
  --prompt-data "${PROMPTS}"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --rollout-shuffle
  --custom-rm-path game_npc_llm.rewards.slime_rm.reward_func
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --eps-clip 0.2
  --eps-clip-high 0.28
  --num-rollout 300
  --rollout-batch-size 4
  --n-samples-per-prompt 8
  --num-steps-per-rollout 1
  --global-batch-size 32
  --rollout-max-response-len 512
  --rollout-temperature 0.8
  --rollout-top-p 0.95
  --balance-data
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${ACTOR_GPUS}"
  --rollout-num-gpus "${ROLLOUT_GPUS}"
  --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
  --tensor-model-parallel-size "${TP_SIZE}"
  --pipeline-model-parallel-size 1
  --context-parallel-size "${CP_SIZE}"
  --sequence-parallel
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 4096
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --sglang-context-length 4096
  --sglang-mem-fraction-static 0.75
  --bf16
)

PYTHONPATH="${PYTHONPATH:-}:${ROOT}/src" "${ROOT}/scripts/run_slime_train.sh" "${COMMON_ARGS[@]}"
