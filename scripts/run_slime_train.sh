#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLIME_TRAIN_CMD:-}" ]]; then
  # shellcheck disable=SC2086
  exec ${SLIME_TRAIN_CMD} "$@"
fi

if [[ -n "${SLIME_DIR:-}" ]]; then
  exec "${PYTHON:-python3}" "${SLIME_DIR}/train.py" "$@"
fi

if [[ -f "train.py" && -d "slime" ]]; then
  exec "${PYTHON:-python3}" train.py "$@"
fi

if command -v slime-train >/dev/null 2>&1; then
  exec slime-train "$@"
fi

echo "Cannot find slime train.py. Set SLIME_DIR=/path/to/slime or SLIME_TRAIN_CMD='python /path/to/slime/train.py'." >&2
exit 2
