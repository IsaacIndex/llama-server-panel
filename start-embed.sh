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

if [[ ! -f "$EMBED_MODEL" ]]; then
  echo "Missing embedding model at: $EMBED_MODEL" >&2
  exit 1
fi

# Auto-tune on first run of this model
TUNE_FILE="$SCRIPT_DIR/bench-results/tuned/$(basename "${EMBED_MODEL%.gguf}").embed.sh"
if [[ ! -f "$TUNE_FILE" ]]; then
  echo "First run for $(basename "$EMBED_MODEL") — auto-tuning for fastest inference..."
  "$SCRIPT_DIR/auto-tune.sh" embed
fi
if [[ -f "$TUNE_FILE" ]]; then
  source "$TUNE_FILE"
fi

if command -v lsof >/dev/null 2>&1; then
  if lsof -nP -iTCP@"$LLAMA_HOST":"$EMBED_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $LLAMA_HOST:$EMBED_PORT is already in use:" >&2
    lsof -nP -iTCP@"$LLAMA_HOST":"$EMBED_PORT" -sTCP:LISTEN >&2
    echo "Stop the existing process or change EMBED_PORT in $SCRIPT_DIR/env.sh or env.local.sh." >&2
    exit 1
  fi
fi

exec "$LLAMA_SERVER_BIN" \
  --model "$EMBED_MODEL" \
  --host "$LLAMA_HOST" \
  --port "$EMBED_PORT" \
  --ctx-size "$EMBED_CTX_SIZE" \
  --threads "$EMBED_THREADS" \
  --batch-size "$EMBED_BATCH_SIZE" \
  --ubatch-size "$EMBED_UBATCH_SIZE" \
  --embedding \
  --pooling "$EMBED_POOLING"
