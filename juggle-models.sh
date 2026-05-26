#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
export LLAMA_SERVER_PANEL_DIR="$SCRIPT_DIR"
source "$SCRIPT_DIR/env.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/model_juggler.py" "$@"
