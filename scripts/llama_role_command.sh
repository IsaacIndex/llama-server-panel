#!/bin/zsh
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/llama_role_command.sh exec <chat|embed|vision> [--port PORT] [--host HOST] [--auto-tune]
  scripts/llama_role_command.sh argv0 <chat|embed|vision> [--port PORT] [--host HOST] [--auto-tune]
  scripts/llama_role_command.sh env0 <chat|embed|vision> [--port PORT] [--host HOST] [--auto-tune]
  scripts/llama_role_command.sh check <chat|embed|vision> [--port PORT] [--host HOST] [--auto-tune]

Modes:
  exec   Validate files and port, then exec llama-server.
  argv0  Print the role-specific llama-server argv separated by NUL bytes.
  env0   Print selected role environment as KEY=VALUE records separated by NUL bytes.
  check  Validate files only.
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

MODE="$1"
ROLE="$2"
shift 2

PORT_OVERRIDE=""
HOST_OVERRIDE=""
AUTO_TUNE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      [[ $# -ge 2 ]] || { echo "--port requires a value" >&2; exit 2; }
      PORT_OVERRIDE="$2"
      shift 2
      ;;
    --host)
      [[ $# -ge 2 ]] || { echo "--host requires a value" >&2; exit 2; }
      HOST_OVERRIDE="$2"
      shift 2
      ;;
    --auto-tune)
      AUTO_TUNE="1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$MODE" in
  exec|argv0|env0|check) ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage
    exit 2
    ;;
esac

case "$ROLE" in
  chat|embed|vision) ;;
  *)
    echo "Unknown role: $ROLE" >&2
    usage
    exit 2
    ;;
esac

SCRIPT_DIR=${0:A:h:h}
export LLAMA_SERVER_PANEL_DIR="$SCRIPT_DIR"
source "$SCRIPT_DIR/env.sh"

if [[ -n "$HOST_OVERRIDE" ]]; then
  export LLAMA_HOST="$HOST_OVERRIDE"
fi

mkdir -p "$LOG_DIR"

role_model() {
  case "$ROLE" in
    chat) printf '%s\n' "$CHAT_MODEL" ;;
    embed) printf '%s\n' "$EMBED_MODEL" ;;
    vision) printf '%s\n' "$VISION_MODEL" ;;
  esac
}

role_port() {
  case "$ROLE" in
    chat) printf '%s\n' "$CHAT_PORT" ;;
    embed) printf '%s\n' "$EMBED_PORT" ;;
    vision) printf '%s\n' "$VISION_PORT" ;;
  esac
}

role_tune_kind() {
  case "$ROLE" in
    chat) printf '%s\n' "chat" ;;
    embed) printf '%s\n' "embed" ;;
    vision) printf '%s\n' "vision" ;;
  esac
}

MODEL_PATH="$(role_model)"
TUNE_KIND="$(role_tune_kind)"
TUNE_FILE="$SCRIPT_DIR/bench-results/tuned/$(basename "${MODEL_PATH%.gguf}").${TUNE_KIND}.sh"

precheck_for_auto_tune() {
  if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
    echo "Missing llama-server binary at: $LLAMA_SERVER_BIN" >&2
    return 1
  fi
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Missing ${ROLE} model at: $MODEL_PATH" >&2
    return 1
  fi
  if [[ "$ROLE" = "vision" && ! -f "${VISION_MMPROJ:-}" ]]; then
    echo "Missing vision mmproj at: ${VISION_MMPROJ:-<unset>}" >&2
    return 1
  fi
}

if [[ "$AUTO_TUNE" = "1" && ! -f "$TUNE_FILE" ]]; then
  precheck_for_auto_tune
  echo "First run for $(basename "$MODEL_PATH") - auto-tuning for fastest inference..." >&2
  "$SCRIPT_DIR/auto-tune.sh" "$ROLE"
fi

if [[ -f "$TUNE_FILE" ]]; then
  source "$TUNE_FILE"
fi

if [[ -n "$PORT_OVERRIDE" ]]; then
  case "$ROLE" in
    chat) CHAT_PORT="$PORT_OVERRIDE" ;;
    embed) EMBED_PORT="$PORT_OVERRIDE" ;;
    vision) VISION_PORT="$PORT_OVERRIDE" ;;
  esac
fi

validate_files() {
  if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
    echo "Missing llama-server binary at: $LLAMA_SERVER_BIN" >&2
    return 1
  fi

  case "$ROLE" in
    chat)
      [[ -f "$CHAT_MODEL" ]] || { echo "Missing chat model at: $CHAT_MODEL" >&2; return 1; }
      ;;
    embed)
      [[ -f "$EMBED_MODEL" ]] || { echo "Missing embedding model at: $EMBED_MODEL" >&2; return 1; }
      ;;
    vision)
      [[ -f "$VISION_MODEL" ]] || { echo "Missing vision model at: $VISION_MODEL" >&2; return 1; }
      [[ -f "${VISION_MMPROJ:-}" ]] || { echo "Missing vision mmproj at: ${VISION_MMPROJ:-<unset>}" >&2; return 1; }
      ;;
  esac
}

check_port_free() {
  local port
  port="$(role_port)"

  if command -v lsof >/dev/null 2>&1; then
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "Port $LLAMA_HOST:$port is already in use:" >&2
      lsof -nP -iTCP:"$port" -sTCP:LISTEN >&2
      echo "Stop the existing process or change the role port in $SCRIPT_DIR/env.sh or env.local.sh." >&2
      return 1
    fi
  fi
}

build_argv() {
  case "$ROLE" in
    chat)
      ROLE_ARGV=(
        "$LLAMA_SERVER_BIN"
        --model "$CHAT_MODEL"
        --alias "$CHAT_ALIAS"
        --host "$LLAMA_HOST"
        --port "$CHAT_PORT"
        --ctx-size "$CHAT_CTX_SIZE"
        --threads "$CHAT_THREADS"
        --parallel "$CHAT_PARALLEL"
        --n-cpu-moe "$CHAT_CPU_MOE_LAYERS"
        --no-mmap
        --cache-type-k "$CHAT_CACHE_TYPE_K"
        --cache-type-v "$CHAT_CACHE_TYPE_V"
        --flash-attn on
        --no-warmup
        --cache-ram 0
        --reasoning on
        --reasoning-format deepseek
        --temp "$CHAT_TEMPERATURE"
        --top-k "$CHAT_TOP_K"
        --top-p "$CHAT_TOP_P"
        --min-p "$CHAT_MIN_P"
        --presence-penalty "$CHAT_PRESENCE_PENALTY"
        --jinja
      )
      ;;
    embed)
      ROLE_ARGV=(
        "$LLAMA_SERVER_BIN"
        --model "$EMBED_MODEL"
        --host "$LLAMA_HOST"
        --port "$EMBED_PORT"
        --ctx-size "$EMBED_CTX_SIZE"
        --batch-size "$EMBED_BATCH_SIZE"
        --ubatch-size "$EMBED_UBATCH_SIZE"
        --threads "$EMBED_THREADS"
        --batch-size "$EMBED_BATCH_SIZE"
        --ubatch-size "$EMBED_UBATCH_SIZE"
        --embedding
        --pooling "$EMBED_POOLING"
      )
      ;;
    vision)
      ROLE_ARGV=(
        "$LLAMA_SERVER_BIN"
        --model "$VISION_MODEL"
        --alias "$VISION_ALIAS"
        --host "$LLAMA_HOST"
        --port "$VISION_PORT"
        --ctx-size "$VISION_CTX_SIZE"
        --threads "$VISION_THREADS"
        --no-mmap
        --cache-type-k "$VISION_CACHE_TYPE_K"
        --cache-type-v "$VISION_CACHE_TYPE_V"
        --flash-attn on
        --no-warmup
        --mmproj "$VISION_MMPROJ"
        --jinja
      )
      ;;
  esac
}

emit_argv0() {
  printf '%s\0' "${ROLE_ARGV[@]}"
}

emit_env0() {
  case "$ROLE" in
    chat)
      records=(
        "ROLE=chat"
        "LLAMA_HOST=$LLAMA_HOST"
        "LOG_DIR=$LOG_DIR"
        "PORT=$CHAT_PORT"
        "MODEL=$CHAT_MODEL"
        "ALIAS=$CHAT_ALIAS"
        "CTX_SIZE=$CHAT_CTX_SIZE"
        "THREADS=$CHAT_THREADS"
        "TUNE_FILE=$TUNE_FILE"
      )
      ;;
    embed)
      records=(
        "ROLE=embed"
        "LLAMA_HOST=$LLAMA_HOST"
        "LOG_DIR=$LOG_DIR"
        "PORT=$EMBED_PORT"
        "MODEL=$EMBED_MODEL"
        "CTX_SIZE=$EMBED_CTX_SIZE"
        "THREADS=$EMBED_THREADS"
        "TUNE_FILE=$TUNE_FILE"
      )
      ;;
    vision)
      records=(
        "ROLE=vision"
        "LLAMA_HOST=$LLAMA_HOST"
        "LOG_DIR=$LOG_DIR"
        "PORT=$VISION_PORT"
        "MODEL=$VISION_MODEL"
        "ALIAS=$VISION_ALIAS"
        "CTX_SIZE=$VISION_CTX_SIZE"
        "THREADS=$VISION_THREADS"
        "MMPROJ=${VISION_MMPROJ:-}"
        "TUNE_FILE=$TUNE_FILE"
      )
      ;;
  esac
  printf '%s\0' "${records[@]}"
}

build_argv

case "$MODE" in
  check)
    validate_files
    ;;
  argv0)
    emit_argv0
    ;;
  env0)
    emit_env0
    ;;
  exec)
    validate_files
    check_port_free
    exec "${ROLE_ARGV[@]}"
    ;;
esac
