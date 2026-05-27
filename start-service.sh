#!/bin/zsh
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  ./start-service.sh [--port PORT] [--bind HOST] [--dry-run] [--check] [--no-auto-tune]

Starts one OpenAI-compatible gateway for chat, embeddings, and vision.
Defaults:
  --bind 0.0.0.0
  --port 8088, or SERVICE_GATEWAY_PORT when set
EOF
}

SCRIPT_DIR=${0:A:h}
export LLAMA_SERVER_PANEL_DIR="$SCRIPT_DIR"
source "$SCRIPT_DIR/env.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

GATEWAY_BIND="${SERVICE_GATEWAY_BIND:-0.0.0.0}"
GATEWAY_PORT="${SERVICE_GATEWAY_PORT:-8088}"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bind)
      [[ $# -ge 2 ]] || { echo "--bind requires a value" >&2; exit 2; }
      GATEWAY_BIND="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "--port requires a value" >&2; exit 2; }
      GATEWAY_PORT="$2"
      shift 2
      ;;
    --dry-run|--check|--no-auto-tune)
      PASSTHROUGH+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/model_juggler.py" \
  --gateway \
  --gateway-bind "$GATEWAY_BIND" \
  --gateway-port "$GATEWAY_PORT" \
  "${PASSTHROUGH[@]}"
