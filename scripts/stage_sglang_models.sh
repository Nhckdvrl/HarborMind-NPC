#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_ROOT="${SGLANG_LOCAL_MODEL_DIR:-/tmp/${USER}/llm_models}"

if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "${ROOT}/.venv-sglang/bin/python" ]]; then
  PYTHON="${ROOT}/.venv-sglang/bin/python"
else
  PYTHON="python3"
fi

PROFILE="${1:-all}"
mkdir -p "${DEST_ROOT}"

stage_base() {
  "${PYTHON}" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-27B",
    local_dir="${DEST_ROOT}/qwen3_5_27b_base",
    local_dir_use_symlinks=False,
)
PY
}

stage_sft() {
  if [[ ! -d "${ROOT}/outputs/sft/qwen3_5_27b_npc_merged" ]]; then
    echo "Missing merged SFT model: ${ROOT}/outputs/sft/qwen3_5_27b_npc_merged" >&2
    exit 1
  fi
  mkdir -p "${DEST_ROOT}/qwen3_5_27b_npc_merged"
  rsync -a --info=progress2 \
    "${ROOT}/outputs/sft/qwen3_5_27b_npc_merged/" \
    "${DEST_ROOT}/qwen3_5_27b_npc_merged/"
}

stage_lora() {
  if [[ ! -d "${ROOT}/outputs/sft/qwen3_5_27b_npc_lora" ]]; then
    echo "Missing SFT LoRA adapter: ${ROOT}/outputs/sft/qwen3_5_27b_npc_lora" >&2
    exit 1
  fi
  mkdir -p "${DEST_ROOT}/qwen3_5_27b_npc_lora"
  rsync -a --info=progress2 \
    --exclude 'checkpoint-*' \
    "${ROOT}/outputs/sft/qwen3_5_27b_npc_lora/" \
    "${DEST_ROOT}/qwen3_5_27b_npc_lora/"
}

stage_lora_supported() {
  stage_lora
  "${PYTHON}" - <<PY
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

src = Path("${DEST_ROOT}/qwen3_5_27b_npc_lora")
dst = Path("${DEST_ROOT}/qwen3_5_27b_npc_lora_sglang_supported")
keep = {"q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "gate_proj", "up_proj", "out_proj"}

if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True)

for path in src.iterdir():
    if path.name == "adapter_model.safetensors":
        continue
    if path.is_file():
        shutil.copy2(path, dst / path.name)

state = {}
with safe_open(src / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
    for key in handle.keys():
        module = key.split(".")[-3]
        if module in keep:
            state[key] = handle.get_tensor(key)
save_file(state, dst / "adapter_model.safetensors")

config_path = dst / "adapter_config.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
config["target_modules"] = sorted(keep)
config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(f"Wrote SGLang-supported LoRA adapter: {dst}")
print(f"Kept tensors: {len(state)}")
PY
}

case "${PROFILE}" in
  base)
    stage_base
    ;;
  sft)
    stage_sft
    ;;
  lora)
    stage_lora
    ;;
  lora_supported)
    stage_lora_supported
    ;;
  all)
    stage_base
    stage_sft
    stage_lora_supported
    ;;
  *)
    echo "Usage: $0 [base|sft|lora|lora_supported|all]" >&2
    exit 1
    ;;
esac

echo "Staged models under: ${DEST_ROOT}"
