#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "${ROOT}/.venv-qwen35/bin/llamafactory-cli" ]]; then
  export PATH="${ROOT}/.venv-qwen35/bin:${PATH}"
fi
if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "${ROOT}/.venv-sglang/bin/python" ]]; then
  PYTHON="${ROOT}/.venv-sglang/bin/python"
elif [[ -x "${ROOT}/.venv-qwen35/bin/python" ]]; then
  PYTHON="${ROOT}/.venv-qwen35/bin/python"
else
  PYTHON="python3"
fi

CONFIG="${1:-configs/llamafactory/qwen3_5_27b_npc_sft_export.yaml}"
llamafactory-cli export "${CONFIG}"

if [[ "${REPAIR_QWEN35_MERGED:-1}" != "1" ]]; then
  exit 0
fi

EXPORT_DIR="$("${PYTHON}" - "${CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
print(config["export_dir"])
PY
)"
BASE_DIR="${QWEN35_BASE_DIR:-${SGLANG_LOCAL_MODEL_DIR:-/tmp/${USER}/llm_models}/qwen3_5_27b_base}"

if [[ ! -f "${EXPORT_DIR}/model.safetensors.index.json" ]]; then
  echo "No safetensors index found in ${EXPORT_DIR}; skip Qwen3.5 merged repair." >&2
  exit 0
fi

if [[ ! -f "${BASE_DIR}/model.safetensors.index.json" ]]; then
  cat >&2 <<EOF
Qwen3.5 merged repair needs the original base checkpoint index:
  ${BASE_DIR}/model.safetensors.index.json

Stage the base model first, or set QWEN35_BASE_DIR:
  bash scripts/stage_sglang_models.sh base
  QWEN35_BASE_DIR=/path/to/qwen3_5_27b_base bash scripts/export_sft.sh ${CONFIG}
EOF
  exit 1
fi

REPAIR_STATUS="$("${PYTHON}" - "${BASE_DIR}" "${EXPORT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

base_dir = Path(sys.argv[1])
export_dir = Path(sys.argv[2])
base_keys = set(json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"])
export_keys = set(json.loads((export_dir / "model.safetensors.index.json").read_text())["weight_map"])

has_bad_prefix = any(
    key.startswith("model.language_model.language_model.language_model.")
    or key.startswith("model.language_model.visual.")
    for key in export_keys
)
has_key_mismatch = base_keys != export_keys
print("repair" if has_bad_prefix or has_key_mismatch else "ok")
PY
)"

if [[ "${REPAIR_STATUS}" == "ok" ]]; then
  echo "Qwen3.5 merged checkpoint already matches base key structure."
  exit 0
fi

TMP_DIR="${EXPORT_DIR}.repaired_tmp"
BACKUP_DIR="${EXPORT_DIR}.bad_$(date +%Y%m%d_%H%M%S)"
echo "Repairing Qwen3.5 merged checkpoint for SGLang compatibility..."
"${PYTHON}" "${ROOT}/scripts/repair_qwen35_merged.py" \
  --base-dir "${BASE_DIR}" \
  --merged-dir "${EXPORT_DIR}" \
  --output-dir "${TMP_DIR}" \
  --overwrite

mv "${EXPORT_DIR}" "${BACKUP_DIR}"
mv "${TMP_DIR}" "${EXPORT_DIR}"
echo "Repaired merged model: ${EXPORT_DIR}"
echo "Original exported model backup: ${BACKUP_DIR}"
