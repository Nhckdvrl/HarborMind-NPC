#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMAFACTORY=(llamafactory-cli)
PYTHON_BIN=""
if [[ -x "${ROOT}/.venv-qwen3/bin/python" ]]; then
  PYTHON_BIN="${ROOT}/.venv-qwen3/bin/python"
elif [[ -x "${ROOT}/.venv-qwen35/bin/python" ]]; then
  PYTHON_BIN="${ROOT}/.venv-qwen35/bin/python"
fi

if [[ -n "${PYTHON_BIN}" ]]; then
  TORCHRUN_WRAPPER_DIR="$(mktemp -d)"
  trap 'rm -rf "${TORCHRUN_WRAPPER_DIR}"' EXIT
  cat > "${TORCHRUN_WRAPPER_DIR}/torchrun" <<EOF
#!/usr/bin/env bash
exec "${PYTHON_BIN}" -m torch.distributed.run "\$@"
EOF
  chmod +x "${TORCHRUN_WRAPPER_DIR}/torchrun"
  export PATH="${TORCHRUN_WRAPPER_DIR}:${PATH}"
  LLAMAFACTORY=("${PYTHON_BIN}" -m llamafactory.cli)
fi

CONFIG="${1:-configs/llamafactory/qwen3_4b_npc_sft_lora.yaml}"
"${LLAMAFACTORY[@]}" train "${CONFIG}"
