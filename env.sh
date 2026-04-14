#!/bin/zsh

if [[ -z "${LLAMA_SERVER_PANEL_DIR:-}" ]]; then
  LLAMA_SERVER_PANEL_DIR=${0:A:h}
fi

export LLAMA_SERVER_PANEL_DIR
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/opt/homebrew/bin/llama-server}"
export LLAMA_HOST="127.0.0.1"
export MODEL_DIR="${MODEL_DIR:-$HOME/models}"
export LOG_DIR="${LOG_DIR:-$LLAMA_SERVER_PANEL_DIR/logs}"

# export CHAT_MODEL="Qwen3-4B-BF16.gguf"
# export CHAT_MODEL="Qwen3-30B-A3B-Thinking-2507-UD-IQ3_XXS.gguf"
export CHAT_MODEL="gpt-oss-20b-mxfp4.gguf"
export CHAT_PORT="8080"
export CHAT_CTX_SIZE="4096"
export CHAT_THREADS="8"
export CHAT_PARALLEL="1"
export CHAT_ALIAS="qwen3-30b-a3b-thinking-2507"
export CHAT_CACHE_TYPE_K="q8_0"
export CHAT_CACHE_TYPE_V="q8_0"
export CHAT_CPU_MOE_LAYERS="40"
export CHAT_TEMPERATURE="0.6"
export CHAT_TOP_K="20"
export CHAT_TOP_P="0.95"
export CHAT_MIN_P="0"
export CHAT_PRESENCE_PENALTY="1.5"

# export EMBED_MODEL="nomic-embed-text-v1.5.Q8_0.gguf"
export EMBED_MODEL="Qwen3-Embedding-4B-Q4_K_M.gguf"
export EMBED_PORT="8081"
export EMBED_CTX_SIZE="2048"
export EMBED_THREADS="8"
export EMBED_POOLING="mean"


export VISION_MODEL="Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf"
export VISION_MMPROJ="/Users/isaac/models/mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf"
export VISION_ALIAS="qwen3vl-30b-a3b-instruct"
export VISION_PORT="8082"
export VISION_CTX_SIZE="4096"
export VISION_THREADS="8"
export VISION_ALIAS="qwen2.5-vl-3b-instruct"
export VISION_CACHE_TYPE_K="f16"
export VISION_CACHE_TYPE_V="f16"

if [[ -f "$LLAMA_SERVER_PANEL_DIR/env.local.sh" ]]; then
  source "$LLAMA_SERVER_PANEL_DIR/env.local.sh"
fi

resolve_model_path() {
  local value="${1:-}"
  if [[ -z "$value" || "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$MODEL_DIR/$value"
  fi
}

export CHAT_MODEL="$(resolve_model_path "$CHAT_MODEL")"
export EMBED_MODEL="$(resolve_model_path "$EMBED_MODEL")"
export VISION_MODEL="$(resolve_model_path "$VISION_MODEL")"
