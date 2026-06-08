#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing Python interpreter. Set PYTHON_BIN or install python3/python." >&2
  exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/llama_role_command.py" exec vision --auto-tune "$@"
