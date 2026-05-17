#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "${ROOT}/.venv-sglang/bin/python" ]]; then
  PYTHON="${ROOT}/.venv-sglang/bin/python"
else
  PYTHON="python3"
fi
PYTHON_BIN_DIR="$(cd "$(dirname "${PYTHON}")" && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

PROFILE="${1:-base}"
CONFIG="${SGLANG_MODELS_CONFIG:-configs/sglang/models.yml}"
LOG_DIR="${SGLANG_LOG_DIR:-${ROOT}/logs}"

if ! "${PYTHON}" -c "import importlib.util; raise SystemExit(importlib.util.find_spec('sglang') is None)" >/dev/null 2>&1; then
  cat >&2 <<EOF
SGLang is not installed in: ${PYTHON}

Recommended setup:
  conda create -p ${ROOT}/.venv-sglang python=3.11 -y
  conda activate ${ROOT}/.venv-sglang
  pip install -e ".[serve,eval]"

Or pass an existing serving environment explicitly:
  PYTHON=/path/to/serving-env/bin/python bash scripts/serve_sglang.sh ${PROFILE}
EOF
  exit 1
fi
if ! command -v ninja >/dev/null 2>&1; then
  cat >&2 <<EOF
The 'ninja' executable is not available on PATH.

Install or repair it in the serving environment:
  ${PYTHON} -m pip install --force-reinstall ninja
EOF
  exit 1
fi

COMMAND="$(PYTHONPATH="${PYTHONPATH:-}:src" "${PYTHON}" -m game_npc_llm.serving.sglang_args \
  --config "${CONFIG}" \
  --profile "${PROFILE}")"

PORT="$("${PYTHON}" - "${CONFIG}" "${PROFILE}" <<'PY'
import sys
import yaml

config_path, profile = sys.argv[1:3]
with open(config_path, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
print(config[profile].get("port", 30000))
PY
)"

MODEL_PATH="$("${PYTHON}" - "${CONFIG}" "${PROFILE}" <<'PY'
import sys
import yaml

config_path, profile = sys.argv[1:3]
with open(config_path, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
print(config[profile]["model"])
PY
)"

if [[ "${MODEL_PATH}" = /* ]]; then
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Local model path does not exist: ${MODEL_PATH}" >&2
    echo "If this is a staged model, run: bash scripts/stage_sglang_models.sh ${PROFILE%_local}" >&2
    exit 1
  fi
  if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Local model path is missing config.json: ${MODEL_PATH}" >&2
    echo "The staged model is incomplete. Re-run: bash scripts/stage_sglang_models.sh ${PROFILE%_local}" >&2
    exit 1
  fi
fi

if [[ "${SGLANG_DRY_RUN:-0}" == "1" ]]; then
  printf '%s\n' "${COMMAND}"
  exit 0
fi

if ss -ltn "( sport = :${PORT} )" | grep -q LISTEN; then
  echo "Port ${PORT} is already listening. Refusing to start another ${PROFILE} server." >&2
  exit 1
fi

if pgrep -af "sglang.launch_server .*--port ${PORT}( |$)" >/dev/null; then
  echo "An SGLang server process for port ${PORT} already exists but is not ready yet." >&2
  echo "Check it with: ps -ef | grep 'sglang.launch_server'" >&2
  echo "Stop it before retrying, or wait for it to finish loading." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
LOG_FILE="${SGLANG_LOG_FILE:-${LOG_DIR}/sglang_${PROFILE}_${PORT}.log}"
echo "Starting SGLang ${PROFILE} on port ${PORT}"
echo "Command: ${COMMAND}"
echo "Log: ${LOG_FILE}"

bash -c "${COMMAND}" 2>&1 | tee "${LOG_FILE}"
