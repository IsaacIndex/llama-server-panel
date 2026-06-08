#!/usr/bin/env python3
"""Cross-platform benchmark runner for local GGUF models."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from llama_runtime import (
    PanelError,
    ensure_llama_server_binary,
    load_config,
    popen_session_kwargs,
    port_in_use,
    repo_dir,
    terminate_process,
)


PROMPT_SHORT = "Explain the concept of recursion in programming with a clear example."
PROMPT_LONG = """Below is a detailed technical document about distributed systems. Please read it carefully and provide a comprehensive summary.

Distributed systems are collections of independent computers that appear to users as a single coherent system. The key challenges in distributed systems include:

1. Fault Tolerance: Systems must continue operating correctly even when individual components fail. This is achieved through redundancy, replication, and consensus protocols like Raft and Paxos. Byzantine fault tolerance handles scenarios where components may behave maliciously.

2. Consistency Models: Different applications require different consistency guarantees. Strong consistency (linearizability) ensures all nodes see the same data at the same time but sacrifices availability. Eventual consistency allows temporary divergence for better performance. Causal consistency preserves cause-effect relationships while allowing concurrent operations to be seen in different orders.

3. Distributed Consensus: Reaching agreement among distributed nodes is fundamental. The FLP impossibility theorem proves that no deterministic consensus protocol can guarantee progress in an asynchronous system with even one faulty process. Practical protocols like Raft use leader election and log replication to achieve consensus in most real-world conditions.

4. CAP Theorem: Brewer's theorem states that a distributed system cannot simultaneously provide more than two of: Consistency, Availability, and Partition tolerance. Since network partitions are inevitable, systems must choose between consistency and availability during partitions.

5. Clock Synchronization: Without a global clock, ordering events across nodes is challenging. Lamport clocks provide logical ordering, while vector clocks capture causality. Hybrid logical clocks combine physical and logical timestamps for practical ordering with bounded drift.

6. Data Replication Strategies: Primary-backup replication is simple but has single points of failure. Multi-primary replication allows writes at any node but requires conflict resolution. Quorum-based systems require W+R>N for consistency, where W is write quorum, R is read quorum, and N is total replicas.

7. Distributed Storage: Systems like Google's Bigtable, Amazon's Dynamo, and Apache Cassandra use consistent hashing for data partitioning, gossip protocols for failure detection, and merkle trees for anti-entropy repair.

Please provide a detailed summary of the above covering all seven topics, including specific algorithms and theorems mentioned."""


def log(message: str) -> None:
    print(f"[bench] {message}")


def warn(message: str) -> None:
    print(f"[bench] {message}", file=sys.stderr)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GGUF models with llama-server")
    parser.add_argument("models", nargs="*", help="model paths or names under MODEL_DIR")
    return parser.parse_args(argv)


def request_json(
    method: str,
    url: str,
    *,
    payload: Optional[dict[str, Any]] = None,
    timeout: float,
) -> dict[str, Any]:
    data = None
    headers = {"Connection": "close"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    text = raw.decode("utf-8", errors="ignore")
    decoded = json.loads(text)
    return decoded if isinstance(decoded, dict) else {}


def request_json_list(url: str, *, timeout: float) -> list[dict[str, Any]]:
    request = urllib.request.Request(url, method="GET", headers={"Connection": "close"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    decoded = json.loads(raw.decode("utf-8", errors="ignore"))
    if isinstance(decoded, list):
        return [item for item in decoded if isinstance(item, dict)]
    return []


def wait_for_server(host: str, port: int, timeout_seconds: int) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            request_json("GET", f"http://{host}:{port}/health", timeout=1.5)
            return True
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(1)
    return False


def run_prompt(host: str, port: int, prompt: str, max_tokens: int, timeout_seconds: int) -> dict[str, Any]:
    response = request_json(
        "POST",
        f"http://{host}:{port}/v1/chat/completions",
        payload={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
        timeout=timeout_seconds,
    )

    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    timings = response.get("timings") if isinstance(response.get("timings"), dict) else {}

    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    gen_tokens = int(usage.get("completion_tokens", 0) or 0)
    pp_ms = float(timings.get("prompt_ms", usage.get("prompt_ms", 0)) or 0)
    tg_ms = float(timings.get("predicted_ms", usage.get("completion_ms", 0)) or 0)
    pp_tps = float(timings.get("prompt_per_second", 0) or 0)
    tg_tps = float(timings.get("predicted_per_second", 0) or 0)

    if pp_tps <= 0 or tg_tps <= 0:
        try:
            slots = request_json_list(f"http://{host}:{port}/slots", timeout=2.0)
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            slots = []
        if slots:
            slot_timings = slots[0].get("timings")
            if isinstance(slot_timings, dict):
                pp_tps = float(slot_timings.get("prompt_per_second", pp_tps) or pp_tps)
                tg_tps = float(slot_timings.get("predicted_per_second", tg_tps) or tg_tps)
                pp_ms = float(slot_timings.get("prompt_ms", pp_ms) or pp_ms)
                tg_ms = float(slot_timings.get("predicted_ms", tg_ms) or tg_ms)

    return {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "pp_ms": pp_ms,
        "tg_ms": tg_ms,
    }


def discover_models(model_dir: Path, args: list[str]) -> list[Path]:
    if args:
        models = []
        for arg in args:
            candidate = Path(arg).expanduser()
            if candidate.is_file():
                models.append(candidate.resolve())
                continue
            joined = model_dir / arg
            if joined.is_file():
                models.append(joined.resolve())
                continue
            raise PanelError(f"Model not found: {arg}")
        return models

    models = []
    for candidate in sorted(model_dir.glob("*.gguf")):
        try:
            size_bytes = candidate.stat().st_size
        except OSError:
            continue
        if size_bytes > 500_000_000:
            models.append(candidate.resolve())
        else:
            warn(f"Skipping small model (likely embeddings): {candidate.name}")
    return models


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    panel_dir = repo_dir()
    config = load_config(panel_dir, apply_tune=False)

    bench_port = int(os.environ.get("BENCH_PORT", "9999"))
    bench_host = os.environ.get("BENCH_HOST", config["LLAMA_HOST"])
    bench_ctx = int(os.environ.get("BENCH_CTX", "4096"))
    bench_threads = int(os.environ.get("BENCH_THREADS", "8"))
    bench_gen_tokens = int(os.environ.get("BENCH_GEN_TOKENS", "200"))
    bench_warmup_tokens = int(os.environ.get("BENCH_WARMUP_TOKENS", "20"))
    bench_timeout = int(os.environ.get("BENCH_TIMEOUT", "300"))
    bench_startup_timeout = int(os.environ.get("BENCH_STARTUP_TIMEOUT", "120"))
    results_dir = Path(os.environ.get("BENCH_RESULTS_DIR", str(panel_dir / "bench-results"))).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if port_in_use(bench_host, bench_port):
        raise PanelError(f"Port {bench_host}:{bench_port} is already in use. Set BENCH_PORT to a different value.")

    models = discover_models(Path(config["MODEL_DIR"]), args.models)
    if not models:
        raise PanelError(f"No models found in {config['MODEL_DIR']}")

    binary = ensure_llama_server_binary(config)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_file = results_dir / f"run-{run_id}.jsonl"

    log(f"Found {len(models)} model(s) to benchmark")
    log(f"Config: ctx={bench_ctx} threads={bench_threads} gen_tokens={bench_gen_tokens}")
    log(f"Results will be saved to: {run_file}")

    all_results: list[dict[str, Any]] = []
    for model_path in models:
        model_name = model_path.name
        model_size = model_path.stat().st_size
        model_size_gb = model_size / 1024 / 1024 / 1024
        log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log(f"Model: {model_name} ({model_size_gb:.1f}GB)")
        log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log("Starting llama-server...")

        log_path = results_dir / f"server-{model_name}.log"
        log_fh = open(log_path, "wb")
        try:
            proc = subprocess.Popen(
                [
                    binary,
                    "--model",
                    str(model_path),
                    "--host",
                    bench_host,
                    "--port",
                    str(bench_port),
                    "--ctx-size",
                    str(bench_ctx),
                    "--threads",
                    str(bench_threads),
                    "--parallel",
                    "1",
                    "--flash-attn",
                    "on",
                    "--no-warmup",
                    "--cache-ram",
                    "0",
                ],
                cwd=str(panel_dir),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                **popen_session_kwargs(),
            )
        finally:
            log_fh.close()

        try:
            if not wait_for_server(bench_host, bench_port, bench_startup_timeout):
                warn(f"Server failed to start for {model_name}. Check log: {log_path}")
                terminate_process(proc)
                continue

            log("Warmup run...")
            try:
                run_prompt(bench_host, bench_port, "Say hello.", bench_warmup_tokens, 60)
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                pass

            log("Running short prompt benchmark...")
            short_result = run_prompt(bench_host, bench_port, PROMPT_SHORT, bench_gen_tokens, bench_timeout)
            log(f"  Short: pp={short_result['pp_tps']:.1f} tok/s, tg={short_result['tg_tps']:.1f} tok/s")

            log("Running long prompt benchmark...")
            long_result = run_prompt(bench_host, bench_port, PROMPT_LONG, bench_gen_tokens, bench_timeout)
            log(f"  Long:  pp={long_result['pp_tps']:.1f} tok/s, tg={long_result['tg_tps']:.1f} tok/s")
        finally:
            log("Stopping server...")
            terminate_process(proc)
            time.sleep(2)

        result = {
            "model": model_name,
            "model_path": str(model_path),
            "model_size_bytes": model_size,
            "run_id": run_id,
            "ctx_size": bench_ctx,
            "threads": bench_threads,
            "short_prompt": short_result,
            "long_prompt": long_result,
            "avg_tg_tps": (short_result["tg_tps"] + long_result["tg_tps"]) / 2,
            "avg_pp_tps": (short_result["pp_tps"] + long_result["pp_tps"]) / 2,
        }
        with run_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")) + "\n")
        all_results.append(result)
        print()

    if not all_results:
        raise PanelError("No benchmark results were collected.")

    all_results.sort(key=lambda item: item["avg_tg_tps"], reverse=True)
    log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log("BENCHMARK RESULTS (ranked by avg generation tok/s)")
    log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"{'MODEL':50} {'SIZE':>8} {'PP(s)':>8} {'PP(l)':>8} {'TG(s)':>8} {'TG(l)':>8}")
    print(f"{'-----':50} {'----':>8} {'-----':>8} {'-----':>8} {'-----':>8} {'-----':>8}")
    for item in all_results:
        print(
            f"{item['model'][:50]:50} "
            f"{item['model_size_bytes'] / 1024 / 1024 / 1024:>7.1f}G "
            f"{item['short_prompt']['pp_tps']:>8.1f} "
            f"{item['long_prompt']['pp_tps']:>8.1f} "
            f"{item['short_prompt']['tg_tps']:>8.1f} "
            f"{item['long_prompt']['tg_tps']:>8.1f}"
        )

    print()
    log("PP = prompt processing tok/s, TG = generation tok/s")
    log("Full results: {}".format(run_file))
    log(f"Best model by generation speed: {all_results[0]['model']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PanelError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
