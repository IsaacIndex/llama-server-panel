#!/usr/bin/env python3
"""Cross-platform auto-tuning for llama-server roles."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from llama_runtime import (
    PanelError,
    build_role_argv_for_config,
    cpu_counts,
    host_label,
    load_config,
    popen_session_kwargs,
    port_in_use,
    repo_dir,
    terminate_process,
    thread_candidates,
    tune_file_path,
    validate_role_files,
)


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
BENCH_PROMPT = "Explain the concept of recursion in programming with a clear example."
VISION_BENCH_PROMPT = "Describe what you see in this image in detail."
EMBED_TEXT = (
    "Distributed systems are collections of independent computers that appear to users as a single coherent "
    "system. The key challenges include fault tolerance, consistency models, distributed consensus, and clock "
    "synchronization. Systems like Bigtable, Dynamo, and Cassandra use consistent hashing for data partitioning."
)


def log(message: str) -> None:
    print(f"[tune] {message}")


def err(message: str) -> None:
    print(f"[tune] {message}", file=sys.stderr)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a role and write a tune file")
    parser.add_argument("role", choices=("chat", "embed", "vision"))
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
    text = CONTROL_CHARS_RE.sub("", raw.decode("utf-8", errors="ignore"))
    if not text.strip():
        return {}
    decoded = json.loads(text)
    return decoded if isinstance(decoded, dict) else {}


def request_json_list(url: str, *, timeout: float) -> list[dict[str, Any]]:
    request = urllib.request.Request(url, method="GET", headers={"Connection": "close"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    text = CONTROL_CHARS_RE.sub("", raw.decode("utf-8", errors="ignore"))
    decoded = json.loads(text)
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


def score_from_slots(host: str, port: int) -> float:
    try:
        slots = request_json_list(f"http://{host}:{port}/slots", timeout=2.0)
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return 0.0
    if not slots:
        return 0.0
    timings = slots[0].get("timings")
    if isinstance(timings, dict):
        try:
            return float(timings.get("predicted_per_second", 0) or 0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def start_server(
    role: str,
    config: dict[str, str],
    *,
    host: str,
    port: int,
    log_path: Path,
) -> subprocess.Popen[bytes]:
    argv = build_role_argv_for_config(config, role, host=host, port=str(port))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "wb")
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(repo_dir()),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            **popen_session_kwargs(),
        )
    finally:
        log_fh.close()
    return proc


def bench_chat_like(
    role: str,
    base_config: dict[str, str],
    *,
    host: str,
    port: int,
    threads: int,
    cache_k: str,
    cache_v: str,
    log_path: Path,
    prompt: str,
    startup_timeout: int,
) -> float:
    prefix = "CHAT" if role == "chat" else "VISION"
    config = dict(base_config)
    config[f"{prefix}_THREADS"] = str(threads)
    config[f"{prefix}_CACHE_TYPE_K"] = cache_k
    config[f"{prefix}_CACHE_TYPE_V"] = cache_v

    proc = start_server(role, config, host=host, port=port, log_path=log_path)
    try:
        if not wait_for_server(host, port, startup_timeout):
            err(f"  Server failed to start (check {log_path})")
            return 0.0

        try:
            request_json(
                "POST",
                f"http://{host}:{port}/v1/chat/completions",
                payload={"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5},
                timeout=60,
            )
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass

        try:
            response = request_json(
                "POST",
                f"http://{host}:{port}/v1/chat/completions",
                payload={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.7,
                },
                timeout=120,
            )
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            return 0.0

        timings = response.get("timings")
        if isinstance(timings, dict):
            try:
                score = float(timings.get("predicted_per_second", 0) or 0)
            except (TypeError, ValueError):
                score = 0.0
        else:
            score = 0.0

        if score <= 0:
            score = score_from_slots(host, port)
        return score
    finally:
        terminate_process(proc)
        time.sleep(1)


def bench_embed(
    base_config: dict[str, str],
    *,
    host: str,
    port: int,
    threads: int,
    log_path: Path,
    startup_timeout: int,
) -> float:
    config = dict(base_config)
    config["EMBED_THREADS"] = str(threads)

    proc = start_server("embed", config, host=host, port=port, log_path=log_path)
    try:
        if not wait_for_server(host, port, startup_timeout):
            err(f"  Server failed to start (check {log_path})")
            return 0.0

        try:
            request_json(
                "POST",
                f"http://{host}:{port}/v1/embeddings",
                payload={"input": "warmup", "model": "embed"},
                timeout=30,
            )
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass

        request_count = 10
        started = time.monotonic()
        for _ in range(request_count):
            try:
                request_json(
                    "POST",
                    f"http://{host}:{port}/v1/embeddings",
                    payload={"input": EMBED_TEXT, "model": "embed"},
                    timeout=30,
                )
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                return 0.0
        elapsed = time.monotonic() - started
        if elapsed <= 0:
            return 0.0
        return request_count / elapsed
    finally:
        terminate_process(proc)
        time.sleep(1)


def write_tune_file(role: str, config: dict[str, str], body_lines: list[str], *, best_score: float, tested: int, unit: str) -> None:
    total, perf = cpu_counts()
    tune_path = tune_file_path(config, role)
    tune_path.parent.mkdir(parents=True, exist_ok=True)
    model_name = Path(config[f"{'CHAT' if role == 'chat' else 'EMBED' if role == 'embed' else 'VISION'}_MODEL"]).name
    heading = "Best throughput" if role == "embed" else "Best generation speed"
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    lines = [
        f"# Auto-tuned config for {model_name} ({role})",
        f"# Generated: {timestamp}",
        f"# Host: {host_label()} ({total} cores, {perf} perf)",
        f"# {heading}: {best_score:.2f} {unit}",
        f"# Tested: {tested} configurations",
        *body_lines,
        "",
    ]
    tune_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"Saved to: {tune_path}")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    panel_dir = repo_dir()
    config = load_config(panel_dir, role=args.role, apply_tune=False)
    validate_role_files(args.role, config)
    prefix = "CHAT" if args.role == "chat" else "EMBED" if args.role == "embed" else "VISION"

    tune_port = int(os.environ.get("TUNE_PORT", "9998"))
    tune_host = config["LLAMA_HOST"]
    startup_timeout = int(os.environ.get("TUNE_STARTUP_TIMEOUT", "120"))
    tune_dir = panel_dir / "bench-results" / "tuned"
    log_path = tune_dir / "server-tune.log"

    if port_in_use(tune_host, tune_port):
        raise PanelError(f"Port {tune_host}:{tune_port} is in use. Set TUNE_PORT to a different value.")

    total, perf = cpu_counts()
    candidates = thread_candidates()
    log(f"Auto-tuning {args.role} model: {Path(config[f'{prefix}_MODEL']).name}")
    log(f"CPU: {total} cores ({perf} performance)")
    log(f"Thread sweep: {' '.join(str(value) for value in candidates)}")

    best_score = -1.0
    best_threads = ""
    best_cache_k = ""
    best_cache_v = ""

    if args.role in {"chat", "vision"}:
        cache_combos = [("f16", "f16"), ("q8_0", "q8_0"), ("q4_0", "q4_0"), ("q8_0", "q4_0")]
        total_runs = len(candidates) * len(cache_combos)
        current = 0
        for threads in candidates:
            for cache_k, cache_v in cache_combos:
                current += 1
                log(f"[{current}/{total_runs}] threads={threads} cache_k={cache_k} cache_v={cache_v}")
                score = bench_chat_like(
                    args.role,
                    config,
                    host=tune_host,
                    port=tune_port,
                    threads=threads,
                    cache_k=cache_k,
                    cache_v=cache_v,
                    log_path=log_path,
                    prompt=BENCH_PROMPT if args.role == "chat" else VISION_BENCH_PROMPT,
                    startup_timeout=startup_timeout,
                )
                log(f"  -> {score:.2f} tok/s")
                if score > best_score:
                    best_score = score
                    best_threads = str(threads)
                    best_cache_k = cache_k
                    best_cache_v = cache_v

        log(f"Best: threads={best_threads} cache_k={best_cache_k} cache_v={best_cache_v} ({best_score:.2f} tok/s)")
        prefix = "CHAT" if args.role == "chat" else "VISION"
        write_tune_file(
            args.role,
            config,
            [
                f"export {prefix}_THREADS={best_threads}",
                f"export {prefix}_CACHE_TYPE_K={best_cache_k}",
                f"export {prefix}_CACHE_TYPE_V={best_cache_v}",
            ],
            best_score=best_score,
            tested=total_runs,
            unit="tok/s",
        )
        return 0

    total_runs = len(candidates)
    for index, threads in enumerate(candidates, start=1):
        log(f"[{index}/{total_runs}] threads={threads}")
        score = bench_embed(
            config,
            host=tune_host,
            port=tune_port,
            threads=threads,
            log_path=log_path,
            startup_timeout=startup_timeout,
        )
        log(f"  -> {score:.2f} req/s")
        if score > best_score:
            best_score = score
            best_threads = str(threads)

    log(f"Best: threads={best_threads} ({best_score:.2f} req/s)")
    write_tune_file(
        args.role,
        config,
        [f"export EMBED_THREADS={best_threads}"],
        best_score=best_score,
        tested=total_runs,
        unit="req/s",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PanelError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
