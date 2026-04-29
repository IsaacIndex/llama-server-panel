#!/bin/zsh
set -euo pipefail

# ── auto-tune.sh ───────────────────────────────────────────────────
# Benchmarks a model with varying configs to find the fastest settings.
# On first run of a model, the start scripts call this automatically.
#
# Usage:
#   ./auto-tune.sh chat    # tune chat model
#   ./auto-tune.sh embed   # tune embed model
#   ./auto-tune.sh vision  # tune vision model
# ───────────────────────────────────────────────────────────────────

SCRIPT_DIR=${0:A:h}
export LLAMA_SERVER_PANEL_DIR="$SCRIPT_DIR"
source "$SCRIPT_DIR/env.sh"

MODE="${1:?Usage: $0 <chat|embed|vision>}"
TUNE_PORT="${TUNE_PORT:-9998}"
TUNE_HOST="$LLAMA_HOST"
TUNE_STARTUP_TIMEOUT="${TUNE_STARTUP_TIMEOUT:-120}"
TUNE_DIR="$SCRIPT_DIR/bench-results/tuned"

mkdir -p "$TUNE_DIR" "$LOG_DIR"

# ── Helpers ────────────────────────────────────────────────────────

log()  { printf "\033[1;36m[tune]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[tune]\033[0m %s\n" "$*" >&2; }

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local elapsed=0
  while (( elapsed < TUNE_STARTUP_TIMEOUT )); do
    if curl -sf "http://$TUNE_HOST:$TUNE_PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    (( elapsed++ ))
  done
  return 1
}

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
  sleep 1
}

now_ms() {
  local ts
  ts=$(date +%s%3N 2>/dev/null || true)
  if [[ "$ts" =~ '^[0-9]+$' ]]; then
    echo "$ts"
  else
    python3 -c 'import time; print(int(time.time() * 1000))'
  fi
}

# ── Prereqs ────────────────────────────────────────────────────────

if ! command -v jq >/dev/null 2>&1; then
  err "jq is required but not found. Install it with: brew install jq"
  exit 1
fi

if ! command -v bc >/dev/null 2>&1; then
  err "bc is required but not found."
  exit 1
fi

# ── Setup based on mode ───────────────────────────────────────────

case "$MODE" in
  chat)
    MODEL_PATH="$CHAT_MODEL"
    CTX_SIZE="$CHAT_CTX_SIZE"
    ;;
  embed)
    MODEL_PATH="$EMBED_MODEL"
    CTX_SIZE="$EMBED_CTX_SIZE"
    ;;
  vision)
    MODEL_PATH="$VISION_MODEL"
    CTX_SIZE="$VISION_CTX_SIZE"
    ;;
  *) echo "Usage: $0 <chat|embed|vision>" >&2; exit 1 ;;
esac

if [[ ! -f "$MODEL_PATH" ]]; then
  err "Model not found: $MODEL_PATH"
  exit 1
fi

MODEL_NAME=$(basename "$MODEL_PATH")
TUNE_FILE="$TUNE_DIR/${MODEL_NAME%.gguf}.${MODE}.sh"

# ── Port conflict check ───────────────────────────────────────────

if command -v lsof >/dev/null 2>&1; then
  if lsof -nP -iTCP@"$TUNE_HOST":"$TUNE_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    err "Port $TUNE_HOST:$TUNE_PORT is in use. Set TUNE_PORT to a different value."
    exit 1
  fi
fi

# ── Detect cores and build thread candidates ──────────────────────

TOTAL_CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 8)
PERF_CORES=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo "$TOTAL_CORES")

# Strategic thread values: quarter, half, perf cores, all cores
candidates=()
(( TOTAL_CORES / 4 >= 2 )) && candidates+=($(( TOTAL_CORES / 4 )))
(( TOTAL_CORES / 2 >= 2 )) && candidates+=($(( TOTAL_CORES / 2 )))
(( PERF_CORES >= 2 )) && candidates+=($PERF_CORES)
candidates+=($TOTAL_CORES)

# Deduplicate and sort
THREAD_VALUES=($(printf '%s\n' "${candidates[@]}" | sort -un))

log "Auto-tuning $MODE model: $MODEL_NAME"
log "CPU: ${TOTAL_CORES} cores (${PERF_CORES} performance)"
log "Thread sweep: ${THREAD_VALUES[*]}"

# ── Chat benchmark function ───────────────────────────────────────

BENCH_PROMPT="Explain the concept of recursion in programming with a clear example."

bench_chat() {
  local threads="$1" cache_k="$2" cache_v="$3"

  SERVER_PID=""
  "$LLAMA_SERVER_BIN" \
    --model "$MODEL_PATH" \
    --host "$TUNE_HOST" --port "$TUNE_PORT" \
    --ctx-size "$CTX_SIZE" \
    --threads "$threads" \
    --parallel "$CHAT_PARALLEL" \
    --n-cpu-moe "$CHAT_CPU_MOE_LAYERS" \
    --no-mmap \
    --cache-type-k "$cache_k" --cache-type-v "$cache_v" \
    --flash-attn on \
    --no-warmup --cache-ram 0 \
    2>"$TUNE_DIR/server-tune.log" &
  SERVER_PID=$!

  if ! wait_for_server; then
    err "  Server failed to start (check $TUNE_DIR/server-tune.log)"
    stop_server
    echo "0"
    return
  fi

  # Warmup
  curl -sf --max-time 60 "http://$TUNE_HOST:$TUNE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":5}' >/dev/null 2>&1 || true

  # Benchmark
  local response tg_tps
  response=$(curl -sf --max-time 120 "http://$TUNE_HOST:$TUNE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg p "$BENCH_PROMPT" \
      '{messages:[{role:"user",content:$p}],max_tokens:150,temperature:0.7}')" 2>/dev/null) || {
    stop_server
    echo "0"
    return
  }

  # Parse timings — response may contain unescaped control chars, so fallback to /slots
  tg_tps=$(printf '%s' "$response" | tr -d '\000-\037' | jq -r '.timings.predicted_per_second // 0' 2>/dev/null) || true

  # Fallback to /slots endpoint
  if [[ "$tg_tps" == "0" || "$tg_tps" == "null" || -z "$tg_tps" ]]; then
    local slots
    slots=$(curl -sf "http://$TUNE_HOST:$TUNE_PORT/slots" 2>/dev/null || echo "[]")
    tg_tps=$(printf '%s' "$slots" | jq -r '.[0].timings.predicted_per_second // 0' 2>/dev/null) || true
  fi

  stop_server
  echo "${tg_tps:-0}"
}

# ── Vision benchmark function ────────────────────────────────────

VISION_BENCH_PROMPT="Describe what you see in this image in detail."

bench_vision() {
  local threads="$1" cache_k="$2" cache_v="$3"

  SERVER_PID=""
  "$LLAMA_SERVER_BIN" \
    --model "$MODEL_PATH" \
    --host "$TUNE_HOST" --port "$TUNE_PORT" \
    --ctx-size "$CTX_SIZE" \
    --threads "$threads" \
    --no-mmap \
    --cache-type-k "$cache_k" --cache-type-v "$cache_v" \
    --flash-attn on \
    --no-warmup \
    --jinja \
    2>"$TUNE_DIR/server-tune.log" &
  SERVER_PID=$!

  if ! wait_for_server; then
    err "  Server failed to start (check $TUNE_DIR/server-tune.log)"
    stop_server
    echo "0"
    return
  fi

  # Warmup
  curl -sf --max-time 60 "http://$TUNE_HOST:$TUNE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":5}' >/dev/null 2>&1 || true

  # Benchmark (text-only prompt — vision model handles it fine)
  local response tg_tps
  response=$(curl -sf --max-time 120 "http://$TUNE_HOST:$TUNE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg p "$VISION_BENCH_PROMPT" \
      '{messages:[{role:"user",content:$p}],max_tokens:150,temperature:0.7}')" 2>/dev/null) || {
    stop_server
    echo "0"
    return
  }

  tg_tps=$(printf '%s' "$response" | tr -d '\000-\037' | jq -r '.timings.predicted_per_second // 0' 2>/dev/null) || true

  if [[ "$tg_tps" == "0" || "$tg_tps" == "null" || -z "$tg_tps" ]]; then
    local slots
    slots=$(curl -sf "http://$TUNE_HOST:$TUNE_PORT/slots" 2>/dev/null || echo "[]")
    tg_tps=$(printf '%s' "$slots" | jq -r '.[0].timings.predicted_per_second // 0' 2>/dev/null) || true
  fi

  stop_server
  echo "${tg_tps:-0}"
}

# ── Embed benchmark function ─────────────────────────────────────

EMBED_TEXT="Distributed systems are collections of independent computers that appear to users as a single coherent system. The key challenges include fault tolerance, consistency models, distributed consensus, and clock synchronization. Systems like Bigtable, Dynamo, and Cassandra use consistent hashing for data partitioning."

bench_embed() {
  local threads="$1"

  SERVER_PID=""
  "$LLAMA_SERVER_BIN" \
    --model "$MODEL_PATH" \
    --host "$TUNE_HOST" --port "$TUNE_PORT" \
    --ctx-size "$CTX_SIZE" \
    --batch-size "$EMBED_BATCH_SIZE" \
    --ubatch-size "$EMBED_UBATCH_SIZE" \
    --threads "$threads" \
    --batch-size "$EMBED_BATCH_SIZE" \
    --ubatch-size "$EMBED_UBATCH_SIZE" \
    --embedding \
    --pooling "$EMBED_POOLING" \
    2>"$TUNE_DIR/server-tune.log" &
  SERVER_PID=$!

  if ! wait_for_server; then
    err "  Server failed to start (check $TUNE_DIR/server-tune.log)"
    stop_server
    echo "0"
    return
  fi

  # Warmup
  curl -sf --max-time 30 "http://$TUNE_HOST:$TUNE_PORT/v1/embeddings" \
    -H 'Content-Type: application/json' \
    -d '{"input":"warmup","model":"embed"}' >/dev/null 2>&1 || true

  # Benchmark: time N requests
  local N=10
  local start_ms end_ms
  start_ms=$(now_ms)

  local req_body
  req_body=$(jq -n --arg t "$EMBED_TEXT" '{input:$t,model:"embed"}')
  for _ in $(seq 1 $N); do
    curl -sf --max-time 30 "http://$TUNE_HOST:$TUNE_PORT/v1/embeddings" \
      -H 'Content-Type: application/json' \
      -d "$req_body" >/dev/null 2>&1 || true
  done

  end_ms=$(now_ms)
  local elapsed_ms=$(( end_ms - start_ms ))
  local rps
  if (( elapsed_ms > 0 )); then
    rps=$(echo "scale=2; $N * 1000 / $elapsed_ms" | bc)
  else
    rps="0"
  fi

  stop_server
  echo "$rps"
}

# ── Main tuning loop ─────────────────────────────────────────────

best_score="0"
best_threads=""
best_cache_k=""
best_cache_v=""

if [[ "$MODE" == "chat" ]]; then
  CACHE_COMBOS=("f16:f16" "q8_0:q8_0" "q4_0:q4_0" "q8_0:q4_0")
  total=$(( ${#THREAD_VALUES[@]} * ${#CACHE_COMBOS[@]} ))
  current=0

  log "Testing $total configurations..."
  echo ""

  for threads in "${THREAD_VALUES[@]}"; do
    for combo in "${CACHE_COMBOS[@]}"; do
      cache_k="${combo%%:*}"
      cache_v="${combo##*:}"
      (( current++ )) || true
      log "[$current/$total] threads=$threads cache_k=$cache_k cache_v=$cache_v"
      score=$(bench_chat "$threads" "$cache_k" "$cache_v")
      log "  -> ${score} tok/s"

      if [[ $(echo "$score > $best_score" | bc -l) == "1" ]]; then
        best_score="$score"
        best_threads="$threads"
        best_cache_k="$cache_k"
        best_cache_v="$cache_v"
      fi
    done
  done

  echo ""
  log "Best: threads=$best_threads cache_k=$best_cache_k cache_v=$best_cache_v ($best_score tok/s)"

  cat > "$TUNE_FILE" <<EOF
# Auto-tuned config for $MODEL_NAME ($MODE)
# Generated: $(date -Iseconds)
# Host: $(hostname -s 2>/dev/null || echo unknown) (${TOTAL_CORES} cores, ${PERF_CORES} perf)
# Best generation speed: $best_score tok/s
# Tested: $total configurations
export CHAT_THREADS=$best_threads
export CHAT_CACHE_TYPE_K=$best_cache_k
export CHAT_CACHE_TYPE_V=$best_cache_v
EOF

elif [[ "$MODE" == "embed" ]]; then
  total=${#THREAD_VALUES[@]}
  current=0

  log "Testing $total configurations..."
  echo ""

  for threads in "${THREAD_VALUES[@]}"; do
    (( current++ )) || true
    log "[$current/$total] threads=$threads"
    score=$(bench_embed "$threads")
    log "  -> ${score} req/s"

    if (( $(echo "$score > $best_score" | bc -l) )); then
      best_score="$score"
      best_threads="$threads"
    fi
  done

  echo ""
  log "Best: threads=$best_threads ($best_score req/s)"

  cat > "$TUNE_FILE" <<EOF
# Auto-tuned config for $MODEL_NAME ($MODE)
# Generated: $(date -Iseconds)
# Host: $(hostname -s 2>/dev/null || echo unknown) (${TOTAL_CORES} cores, ${PERF_CORES} perf)
# Best throughput: $best_score req/s
# Tested: $total configurations
export EMBED_THREADS=$best_threads
EOF

elif [[ "$MODE" == "vision" ]]; then
  CACHE_COMBOS=("f16:f16" "q8_0:q8_0" "q4_0:q4_0" "q8_0:q4_0")
  total=$(( ${#THREAD_VALUES[@]} * ${#CACHE_COMBOS[@]} ))
  current=0

  log "Testing $total configurations..."
  echo ""

  for threads in "${THREAD_VALUES[@]}"; do
    for combo in "${CACHE_COMBOS[@]}"; do
      cache_k="${combo%%:*}"
      cache_v="${combo##*:}"
      (( current++ )) || true
      log "[$current/$total] threads=$threads cache_k=$cache_k cache_v=$cache_v"
      score=$(bench_vision "$threads" "$cache_k" "$cache_v")
      log "  -> ${score} tok/s"

      if [[ $(echo "$score > $best_score" | bc -l) == "1" ]]; then
        best_score="$score"
        best_threads="$threads"
        best_cache_k="$cache_k"
        best_cache_v="$cache_v"
      fi
    done
  done

  echo ""
  log "Best: threads=$best_threads cache_k=$best_cache_k cache_v=$best_cache_v ($best_score tok/s)"

  cat > "$TUNE_FILE" <<EOF
# Auto-tuned config for $MODEL_NAME ($MODE)
# Generated: $(date -Iseconds)
# Host: $(hostname -s 2>/dev/null || echo unknown) (${TOTAL_CORES} cores, ${PERF_CORES} perf)
# Best generation speed: $best_score tok/s
# Tested: $total configurations
export VISION_THREADS=$best_threads
export VISION_CACHE_TYPE_K=$best_cache_k
export VISION_CACHE_TYPE_V=$best_cache_v
EOF
fi

log "Saved to: $TUNE_FILE"
