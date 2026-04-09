#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
export LLAMA_SERVER_PANEL_DIR="$SCRIPT_DIR"
source "$SCRIPT_DIR/env.sh"

mkdir -p "$LOG_DIR"

if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
  echo "Missing llama-server binary at: $LLAMA_SERVER_BIN" >&2
  exit 1
fi

if [[ ! -f "$VISION_MODEL" ]]; then
  echo "Missing vision model at: $VISION_MODEL" >&2
  exit 1
fi

# Auto-tune on first run of this model
TUNE_FILE="$SCRIPT_DIR/bench-results/tuned/$(basename "${VISION_MODEL%.gguf}").vision.sh"
if [[ ! -f "$TUNE_FILE" ]]; then
  echo "First run for $(basename "$VISION_MODEL") — auto-tuning for fastest inference..."
  "$SCRIPT_DIR/auto-tune.sh" vision
fi
if [[ -f "$TUNE_FILE" ]]; then
  source "$TUNE_FILE"
fi

if command -v lsof >/dev/null 2>&1; then
  if lsof -nP -iTCP@"$LLAMA_HOST":"$VISION_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $LLAMA_HOST:$VISION_PORT is already in use:" >&2
    lsof -nP -iTCP@"$LLAMA_HOST":"$VISION_PORT" -sTCP:LISTEN >&2
    echo "Stop the existing process or change VISION_PORT in $SCRIPT_DIR/env.sh or env.local.sh." >&2
    exit 1
  fi
fi

exec "$LLAMA_SERVER_BIN" \
  --model "$VISION_MODEL" \
  --alias "$VISION_ALIAS" \
  --host "$LLAMA_HOST" \
  --port "$VISION_PORT" \
  --ctx-size "$VISION_CTX_SIZE" \
  --threads "$VISION_THREADS" \
  --no-mmap \
  --cache-type-k "$VISION_CACHE_TYPE_K" \
  --cache-type-v "$VISION_CACHE_TYPE_V" \
  --flash-attn on \
  --no-warmup \
  --jinja
