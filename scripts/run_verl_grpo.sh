#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$ROOT/outputs/sft/qwen3_5_27b_npc_merged}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/grpo/qwen3_5_27b_npc_grpo}"
REWARD_PATH="${REWARD_PATH:-$ROOT/src/game_npc_llm/rewards/verl_reward.py}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
ROLLOUT_MULTI_STAGE_WAKE_UP="${ROLLOUT_MULTI_STAGE_WAKE_UP:-false}"
SGLANG_MODEL_IMPL="${SGLANG_MODEL_IMPL:-transformers}"
N_GPUS="${N_GPUS:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-$N_GPUS}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:--1}"
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-False}"
RAY_ADDRESS_OVERRIDE="${RAY_ADDRESS_OVERRIDE:-}"
VERL_MAX_COLOCATE_COUNT="${VERL_MAX_COLOCATE_COUNT:-1}"

case "$MODE" in
  smoke)
    TRAIN_FILE="${TRAIN_FILE:-$ROOT/data/rl/verl/grpo_smoke_100.parquet}"
    ;;
  full)
    TRAIN_FILE="${TRAIN_FILE:-$ROOT/data/rl/verl/grpo_train.parquet}"
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Missing $TRAIN_FILE. Run: python scripts/prepare_verl_grpo_data.py" >&2
  exit 1
fi

if [[ -d "$ROOT/external/verl" ]]; then
  export PYTHONPATH="$ROOT/src:$ROOT/external/verl:${PYTHONPATH:-}"
else
  export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
fi
export VERL_MAX_COLOCATE_COUNT
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
if [[ "$ROLLOUT_NAME" == "sglang" ]]; then
  export ENABLE_SGLANG_VERL_PATCH=1
fi

HYDRA_OVERRIDES=()
if [[ -n "$RAY_ADDRESS_OVERRIDE" ]]; then
  HYDRA_OVERRIDES+=(+ray_kwargs.ray_init.address="$RAY_ADDRESS_OVERRIDE")
fi
if [[ -n "$TOTAL_TRAINING_STEPS" ]]; then
  HYDRA_OVERRIDES+=(trainer.total_training_steps="$TOTAL_TRAINING_STEPS")
fi

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$TRAIN_FILE" \
  data.prompt_key=prompt \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$TRAIN_BATCH_SIZE" \
  data.truncation=left \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.lora_rank="$LORA_RANK" \
  actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=3e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name="$ROLLOUT_NAME" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.multi_stage_wake_up="$ROLLOUT_MULTI_STAGE_WAKE_UP" \
  actor_rollout_ref.rollout.max_model_len="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
  actor_rollout_ref.rollout.max_num_seqs="$TRAIN_BATCH_SIZE" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.model_impl="$SGLANG_MODEL_IMPL" \
  reward_model.enable=False \
  custom_reward_function.path="$REWARD_PATH" \
  custom_reward_function.name=compute_score \
  trainer.project_name=game_npc_llm \
  trainer.experiment_name="qwen3_5_27b_npc_grpo_lora_${MODE}" \
  trainer.logger='["console"]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.default_hdfs_dir=null \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.critic_warmup=0 \
  +ray_kwargs.ray_init.include_dashboard="$RAY_INCLUDE_DASHBOARD" \
  "${HYDRA_OVERRIDES[@]}"
