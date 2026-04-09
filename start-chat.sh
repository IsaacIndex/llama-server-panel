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

if [[ ! -f "$CHAT_MODEL" ]]; then
  echo "Missing chat model at: $CHAT_MODEL" >&2
  exit 1
fi

# Auto-tune on first run of this model
TUNE_FILE="$SCRIPT_DIR/bench-results/tuned/$(basename "${CHAT_MODEL%.gguf}").chat.sh"
if [[ ! -f "$TUNE_FILE" ]]; then
  echo "First run for $(basename "$CHAT_MODEL") — auto-tuning for fastest inference..."
  "$SCRIPT_DIR/auto-tune.sh" chat
fi
if [[ -f "$TUNE_FILE" ]]; then
  source "$TUNE_FILE"
fi

if command -v lsof >/dev/null 2>&1; then
  if lsof -nP -iTCP@"$LLAMA_HOST":"$CHAT_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $LLAMA_HOST:$CHAT_PORT is already in use:" >&2
    lsof -nP -iTCP@"$LLAMA_HOST":"$CHAT_PORT" -sTCP:LISTEN >&2
    echo "Stop the existing process or change CHAT_PORT in $SCRIPT_DIR/env.sh or env.local.sh." >&2
    exit 1
  fi
fi

exec "$LLAMA_SERVER_BIN" \
  --model "$CHAT_MODEL" \
  --alias "$CHAT_ALIAS" \
  --host "$LLAMA_HOST" \
  --port "$CHAT_PORT" \
  --ctx-size "$CHAT_CTX_SIZE" \
  --threads "$CHAT_THREADS" \
  --parallel "$CHAT_PARALLEL" \
  --n-cpu-moe "$CHAT_CPU_MOE_LAYERS" \
  --no-mmap \
  --cache-type-k "$CHAT_CACHE_TYPE_K" \
  --cache-type-v "$CHAT_CACHE_TYPE_V" \
  --flash-attn on \
  --no-warmup \
  --cache-ram 0 \
  --reasoning on \
  --reasoning-format deepseek \
  --temp "$CHAT_TEMPERATURE" \
  --top-k "$CHAT_TOP_K" \
  --top-p "$CHAT_TOP_P" \
  --min-p "$CHAT_MIN_P" \
  --presence-penalty "$CHAT_PRESENCE_PENALTY" \
  --jinja
