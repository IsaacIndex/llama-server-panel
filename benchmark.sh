#!/bin/zsh
set -euo pipefail

# ── benchmark.sh ────────────────────────────────────────────────────
# Benchmarks GGUF models with llama-server on this machine.
# Starts each model, sends standardized prompts, records tok/s,
# and prints a ranked summary.
#
# Usage:
#   ./benchmark.sh                        # benchmark all .gguf in ~/models (except embed models)
#   ./benchmark.sh model1.gguf model2.gguf  # benchmark specific files
#   BENCH_CTX=8192 ./benchmark.sh         # override context size
# ────────────────────────────────────────────────────────────────────

SCRIPT_DIR=${0:A:h}
source "$SCRIPT_DIR/env.sh"

# ── Configuration (override with env vars) ──────────────────────────
BENCH_PORT="${BENCH_PORT:-9999}"
BENCH_HOST="${LLAMA_HOST:-127.0.0.1}"
BENCH_CTX="${BENCH_CTX:-4096}"
BENCH_THREADS="${BENCH_THREADS:-8}"
BENCH_GEN_TOKENS="${BENCH_GEN_TOKENS:-200}"   # tokens to generate per prompt
BENCH_WARMUP_TOKENS="${BENCH_WARMUP_TOKENS:-20}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-300}"          # max seconds to wait for generation
BENCH_STARTUP_TIMEOUT="${BENCH_STARTUP_TIMEOUT:-120}"
BENCH_RESULTS_DIR="${BENCH_RESULTS_DIR:-$SCRIPT_DIR/bench-results}"

mkdir -p "$BENCH_RESULTS_DIR"

# ── Prompts for benchmarking ────────────────────────────────────────
# Short prompt: measures generation speed (low pp, high tg)
PROMPT_SHORT="Explain the concept of recursion in programming with a clear example."

# Long prompt: measures prompt processing speed (high pp)
PROMPT_LONG="Below is a detailed technical document about distributed systems. Please read it carefully and provide a comprehensive summary.

Distributed systems are collections of independent computers that appear to users as a single coherent system. The key challenges in distributed systems include:

1. Fault Tolerance: Systems must continue operating correctly even when individual components fail. This is achieved through redundancy, replication, and consensus protocols like Raft and Paxos. Byzantine fault tolerance handles scenarios where components may behave maliciously.

2. Consistency Models: Different applications require different consistency guarantees. Strong consistency (linearizability) ensures all nodes see the same data at the same time but sacrifices availability. Eventual consistency allows temporary divergence for better performance. Causal consistency preserves cause-effect relationships while allowing concurrent operations to be seen in different orders.

3. Distributed Consensus: Reaching agreement among distributed nodes is fundamental. The FLP impossibility theorem proves that no deterministic consensus protocol can guarantee progress in an asynchronous system with even one faulty process. Practical protocols like Raft use leader election and log replication to achieve consensus in most real-world conditions.

4. CAP Theorem: Brewer's theorem states that a distributed system cannot simultaneously provide more than two of: Consistency, Availability, and Partition tolerance. Since network partitions are inevitable, systems must choose between consistency and availability during partitions.

5. Clock Synchronization: Without a global clock, ordering events across nodes is challenging. Lamport clocks provide logical ordering, while vector clocks capture causality. Hybrid logical clocks combine physical and logical timestamps for practical ordering with bounded drift.

6. Data Replication Strategies: Primary-backup replication is simple but has single points of failure. Multi-primary replication allows writes at any node but requires conflict resolution. Quorum-based systems require W+R>N for consistency, where W is write quorum, R is read quorum, and N is total replicas.

7. Distributed Storage: Systems like Google's Bigtable, Amazon's Dynamo, and Apache Cassandra use consistent hashing for data partitioning, gossip protocols for failure detection, and merkle trees for anti-entropy repair.

Please provide a detailed summary of the above covering all seven topics, including specific algorithms and theorems mentioned."

# ── Helpers ──────────────────────────────────────────────────────────

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

log() { printf "\033[1;34m[bench]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[bench]\033[0m %s\n" "$*" >&2; }
err() { printf "\033[1;31m[bench]\033[0m %s\n" "$*" >&2; }

wait_for_server() {
  local elapsed=0
  while (( elapsed < BENCH_STARTUP_TIMEOUT )); do
    if curl -sf "http://$BENCH_HOST:$BENCH_PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    (( elapsed++ ))
  done
  return 1
}

# Run a single prompt and return JSON with timing info
run_prompt() {
  local label="$1" prompt="$2" max_tokens="$3"

  local response
  response=$(curl -sf --max-time "$BENCH_TIMEOUT" \
    "http://$BENCH_HOST:$BENCH_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n \
      --arg prompt "$prompt" \
      --argjson max_tokens "$max_tokens" \
      '{
        messages: [{role: "user", content: $prompt}],
        max_tokens: $max_tokens,
        temperature: 0.7
      }')" 2>&1) || {
    err "  Request failed for $label"
    echo '{"error": true}'
    return 1
  }

  # Extract usage stats from the OpenAI-compatible response
  local prompt_tokens gen_tokens total_time_ms
  prompt_tokens=$(echo "$response" | jq -r '.usage.prompt_tokens // 0')
  gen_tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0')

  # llama-server includes timings in the response
  local pp_ms tg_ms pp_tps tg_tps
  pp_ms=$(echo "$response" | jq -r '.timings.prompt_ms // .usage.prompt_ms // 0' 2>/dev/null)
  tg_ms=$(echo "$response" | jq -r '.timings.predicted_ms // .usage.completion_ms // 0' 2>/dev/null)
  pp_tps=$(echo "$response" | jq -r '.timings.prompt_per_second // 0' 2>/dev/null)
  tg_tps=$(echo "$response" | jq -r '.timings.predicted_per_second // 0' 2>/dev/null)

  # If timings weren't in the response, try the /slots endpoint
  if [[ "$pp_tps" == "0" || "$pp_tps" == "null" ]]; then
    local slots_info
    slots_info=$(curl -sf "http://$BENCH_HOST:$BENCH_PORT/slots" 2>/dev/null || echo "[]")
    pp_tps=$(echo "$slots_info" | jq -r '.[0].timings.prompt_per_second // 0' 2>/dev/null)
    tg_tps=$(echo "$slots_info" | jq -r '.[0].timings.predicted_per_second // 0' 2>/dev/null)
    pp_ms=$(echo "$slots_info" | jq -r '.[0].timings.prompt_ms // 0' 2>/dev/null)
    tg_ms=$(echo "$slots_info" | jq -r '.[0].timings.predicted_ms // 0' 2>/dev/null)
  fi

  jq -n \
    --arg label "$label" \
    --argjson prompt_tokens "$prompt_tokens" \
    --argjson gen_tokens "$gen_tokens" \
    --argjson pp_tps "${pp_tps:-0}" \
    --argjson tg_tps "${tg_tps:-0}" \
    --argjson pp_ms "${pp_ms:-0}" \
    --argjson tg_ms "${tg_ms:-0}" \
    '{label: $label, prompt_tokens: $prompt_tokens, gen_tokens: $gen_tokens,
      pp_tps: $pp_tps, tg_tps: $tg_tps, pp_ms: $pp_ms, tg_ms: $tg_ms}'
}

# ── Collect models to benchmark ─────────────────────────────────────

models=()
if (( $# > 0 )); then
  for arg in "$@"; do
    if [[ -f "$arg" ]]; then
      models+=("$arg")
    elif [[ -f "$MODEL_DIR/$arg" ]]; then
      models+=("$MODEL_DIR/$arg")
    else
      err "Model not found: $arg"
      exit 1
    fi
  done
else
  # Auto-discover, skip small embedding models (<500MB)
  for f in "$MODEL_DIR"/*.gguf; do
    [[ -f "$f" ]] || continue
    local_size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)
    if (( local_size > 500000000 )); then
      models+=("$f")
    else
      warn "Skipping small model (likely embeddings): $(basename "$f")"
    fi
  done
fi

if (( ${#models[@]} == 0 )); then
  err "No models found to benchmark."
  echo ""
  echo "Place .gguf files in $MODEL_DIR or pass paths as arguments."
  echo ""
  echo "Suggested models to download for M3 Max 96GB:"
  echo "  huggingface-cli download bartowski/Qwen3-32B-GGUF Qwen3-32B-Q4_K_M.gguf --local-dir ~/models"
  echo "  huggingface-cli download bartowski/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --local-dir ~/models"
  echo "  huggingface-cli download bartowski/Llama-3.1-70B-Instruct-GGUF Llama-3.1-70B-Instruct-Q4_K_M.gguf --local-dir ~/models"
  echo "  huggingface-cli download bartowski/gemma-3-27b-it-GGUF gemma-3-27b-it-Q4_K_M.gguf --local-dir ~/models"
  echo "  huggingface-cli download bartowski/Mistral-Small-3.2-24B-Instruct-2506-GGUF Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf --local-dir ~/models"
  echo "  huggingface-cli download bartowski/DeepSeek-R1-0528-Qwen3-8B-GGUF DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf --local-dir ~/models"
  exit 1
fi

log "Found ${#models[@]} model(s) to benchmark"
log "Hardware: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
log "RAM: $(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f GB", $1/1024/1024/1024}')"
log "Config: ctx=$BENCH_CTX threads=$BENCH_THREADS gen_tokens=$BENCH_GEN_TOKENS"
echo ""

# ── Check for port conflicts ────────────────────────────────────────
if lsof -nP -iTCP@"$BENCH_HOST":"$BENCH_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  err "Port $BENCH_HOST:$BENCH_PORT is already in use. Set BENCH_PORT to a different value."
  exit 1
fi

# ── Timestamp for this run ──────────────────────────────────────────
RUN_ID=$(date +%Y%m%d-%H%M%S)
RUN_FILE="$BENCH_RESULTS_DIR/run-$RUN_ID.jsonl"

log "Results will be saved to: $RUN_FILE"
echo ""

# ── Main benchmark loop ─────────────────────────────────────────────

all_results=()

for model_path in "${models[@]}"; do
  model_name=$(basename "$model_path")
  model_size=$(stat -f%z "$model_path" 2>/dev/null || stat -c%s "$model_path" 2>/dev/null)
  model_size_gb=$(echo "scale=1; $model_size / 1024 / 1024 / 1024" | bc)

  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "Model: $model_name (${model_size_gb}GB)"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Start llama-server
  log "Starting llama-server..."
  SERVER_PID=""

  "$LLAMA_SERVER_BIN" \
    --model "$model_path" \
    --host "$BENCH_HOST" \
    --port "$BENCH_PORT" \
    --ctx-size "$BENCH_CTX" \
    --threads "$BENCH_THREADS" \
    --parallel 1 \
    --flash-attn on \
    --no-warmup \
    --cache-ram 0 \
    2>"$BENCH_RESULTS_DIR/server-$model_name.log" &
  SERVER_PID=$!

  if ! wait_for_server; then
    err "Server failed to start for $model_name (timeout ${BENCH_STARTUP_TIMEOUT}s)"
    err "Check log: $BENCH_RESULTS_DIR/server-$model_name.log"
    kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    continue
  fi
  log "Server ready (PID $SERVER_PID)"

  # Warmup run (discard results)
  log "Warmup run..."
  run_prompt "warmup" "Say hello." "$BENCH_WARMUP_TOKENS" >/dev/null 2>&1 || true

  # Benchmark: short prompt (generation-heavy)
  log "Running short prompt benchmark..."
  short_result=$(run_prompt "short" "$PROMPT_SHORT" "$BENCH_GEN_TOKENS")
  short_tg=$(echo "$short_result" | jq -r '.tg_tps')
  short_pp=$(echo "$short_result" | jq -r '.pp_tps')
  log "  Short: pp=${short_pp} tok/s, tg=${short_tg} tok/s"

  # Benchmark: long prompt (prompt-processing-heavy)
  log "Running long prompt benchmark..."
  long_result=$(run_prompt "long" "$PROMPT_LONG" "$BENCH_GEN_TOKENS")
  long_tg=$(echo "$long_result" | jq -r '.tg_tps')
  long_pp=$(echo "$long_result" | jq -r '.pp_tps')
  log "  Long:  pp=${long_pp} tok/s, tg=${long_tg} tok/s"

  # Stop server
  log "Stopping server..."
  kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  sleep 2  # let memory settle

  # Save results
  result_json=$(jq -n \
    --arg model "$model_name" \
    --arg model_path "$model_path" \
    --argjson model_size_bytes "$model_size" \
    --arg run_id "$RUN_ID" \
    --argjson ctx "$BENCH_CTX" \
    --argjson threads "$BENCH_THREADS" \
    --argjson short "$short_result" \
    --argjson long "$long_result" \
    '{
      model: $model, model_path: $model_path, model_size_bytes: $model_size_bytes,
      run_id: $run_id, ctx_size: $ctx, threads: $threads,
      short_prompt: $short, long_prompt: $long,
      avg_tg_tps: (($short.tg_tps + $long.tg_tps) / 2),
      avg_pp_tps: (($short.pp_tps + $long.pp_tps) / 2)
    }')

  echo "$result_json" >> "$RUN_FILE"
  all_results+=("$result_json")

  echo ""
done

# ── Summary ─────────────────────────────────────────────────────────

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "BENCHMARK RESULTS (ranked by avg generation tok/s)"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
printf "%-50s %8s %8s %8s %8s %8s\n" "MODEL" "SIZE" "PP(s)" "PP(l)" "TG(s)" "TG(l)"
printf "%-50s %8s %8s %8s %8s %8s\n" "-----" "----" "-----" "-----" "-----" "-----"

# Sort by avg tg tok/s descending
printf '%s\n' "${all_results[@]}" | jq -rs '
  sort_by(-.avg_tg_tps) | .[] |
  [.model,
   (.model_size_bytes / 1024 / 1024 / 1024 | . * 10 | round / 10 | tostring + "G"),
   (.short_prompt.pp_tps | . * 10 | round / 10 | tostring),
   (.long_prompt.pp_tps | . * 10 | round / 10 | tostring),
   (.short_prompt.tg_tps | . * 10 | round / 10 | tostring),
   (.long_prompt.tg_tps | . * 10 | round / 10 | tostring)
  ] | @tsv' | while IFS=$'\t' read -r name size pp_s pp_l tg_s tg_l; do
  printf "%-50s %8s %8s %8s %8s %8s\n" "$name" "$size" "$pp_s" "$pp_l" "$tg_s" "$tg_l"
done

echo ""
log "PP = prompt processing tok/s, TG = generation tok/s"
log "(s) = short prompt, (l) = long prompt"
log "Full results: $RUN_FILE"
echo ""

# ── Recommendation ──────────────────────────────────────────────────

best=$(printf '%s\n' "${all_results[@]}" | jq -rs 'sort_by(-.avg_tg_tps) | .[0].model')
log "Best model by generation speed: $best"
