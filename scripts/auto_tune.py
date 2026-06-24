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
    launch_diagnostics,
    load_config,
    popen_session_kwargs,
    prepare_llama_server_argv,
    port_in_use,
    repo_dir,
    terminate_process,
    thread_candidates,
    tune_file_path,
    validate_role_files,
    write_compat_filter_notice,
)


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
MEMORY_PROJECTION_RE = re.compile(
    r"projected to use\s+(\d+)\s+MiB of device memory vs\.\s+(\d+)\s+MiB of free device memory"
)
MEMORY_FIT_FAILURE_MARKERS = (
    "cannot meet free memory target",
    "failed to fit params to free device memory",
)
WINDOWS_ACCESS_VIOLATION_EXIT_CODES = {3221225477, -1073741819}
BENCH_PROMPT = "Explain the concept of recursion in programming with a clear example."
VISION_BENCH_PROMPT = "Describe what you see in this image in detail."
EMBED_TEXT = (
    "Distributed systems are collections of independent computers that appear to users as a single coherent "
    "system. The key challenges include fault tolerance, consistency models, distributed consensus, and clock "
    "synchronization. Systems like Bigtable, Dynamo, and Cassandra use consistent hashing for data partitioning."
)
TUNE_LOG_PATH: Optional[Path] = None
CHAT_LIKE_CTX_FALLBACKS = (8192, 4096, 3072, 2048, 1024, 512)
STARTUP_EXIT_CODES: list[int] = []


def set_tune_log_path(path: Optional[Path]) -> None:
    global TUNE_LOG_PATH
    TUNE_LOG_PATH = path


def reset_startup_failure_state() -> None:
    STARTUP_EXIT_CODES.clear()


def record_startup_exit_code(returncode: Optional[int]) -> None:
    if returncode is not None:
        STARTUP_EXIT_CODES.append(returncode)


def startup_exit_code_hint(returncode: Optional[int]) -> str:
    if returncode in WINDOWS_ACCESS_VIOLATION_EXIT_CODES:
        return (
            "Windows access violation 0xC0000005 "
            "(exit code 3221225477 / -1073741819); this usually points to a native "
            "llama.cpp crash rather than context-size memory pressure."
        )
    return ""


def startup_failure_hint() -> str:
    if not STARTUP_EXIT_CODES:
        return ""

    unique_codes = list(dict.fromkeys(STARTUP_EXIT_CODES))
    if any(startup_exit_code_hint(code) for code in unique_codes):
        return (
            "Candidate startup crashed with Windows access violation 0xC0000005 "
            "(exit code 3221225477 / -1073741819), which usually points to a native "
            "llama.cpp crash rather than context-size memory pressure. Check the candidate "
            "logs for the failing binary, model file, or startup flags."
        )

    joined_codes = ", ".join(str(code) for code in unique_codes)
    plural = "s" if len(unique_codes) != 1 else ""
    return f"Candidate startup exited before health checks completed (exit code{plural}: {joined_codes})."


def no_working_config_message(role: str, *, prefix: str, log_path: Path, tune_dir: Path) -> str:
    hint = startup_failure_hint()
    if hint:
        return (
            f"No working {role} tuning configuration started. "
            f"Check {log_path} and candidate logs under {tune_dir}. {hint}"
        )
    return (
        f"No working {role} tuning configuration started. "
        f"Check {log_path} and candidate logs under {tune_dir}; lowering {prefix}_CTX_SIZE may be required."
    )


def append_tune_log(line: str) -> None:
    if TUNE_LOG_PATH is None:
        return
    if os.environ.get("PANEL_AUTO_TUNE_STDOUT_LOG") == "1":
        return
    TUNE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TUNE_LOG_PATH.open("a", encoding="utf-8", errors="replace") as log_fh:
        log_fh.write(f"{line}\n")


def append_candidate_log(log_path: Optional[Path], line: str) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log_fh:
        log_fh.write(f"[panel] {line}\n")


def log(message: str) -> None:
    line = f"[tune] {message}"
    print(line)
    append_tune_log(line)


def err(message: str) -> None:
    line = f"[tune] {message}"
    print(line, file=sys.stderr)
    append_tune_log(line)


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


def wait_for_server(
    host: str,
    port: int,
    timeout_seconds: int,
    *,
    proc: Optional[subprocess.Popen[bytes]] = None,
    log_path: Optional[Path] = None,
    memory_headroom_mib: int = 512,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            record_startup_exit_code(proc.returncode)
            err(f"  Server exited during startup with code {proc.returncode}")
            append_candidate_log(log_path, f"server exited during startup with code {proc.returncode}")
            hint = startup_exit_code_hint(proc.returncode)
            if hint:
                err(f"  Startup exit detail: {hint}")
                append_candidate_log(log_path, f"startup exit detail: {hint}")
            return False
        if proc is not None and log_path is not None:
            pressure = startup_memory_pressure_message(log_path, memory_headroom_mib=memory_headroom_mib)
            if pressure:
                err(f"  {pressure}")
                terminate_process(proc)
                return False
        try:
            request_json("GET", f"http://{host}:{port}/health", timeout=1.5)
            return True
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(1)
    return False


def startup_memory_pressure_message(log_path: Path, *, memory_headroom_mib: int = 512) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""

    lowered = text.lower()
    for marker in MEMORY_FIT_FAILURE_MARKERS:
        if marker in lowered:
            return f"Terminating candidate early: llama.cpp reported memory fit failure in {log_path}"

    matches = list(MEMORY_PROJECTION_RE.finditer(text))
    if not matches:
        return ""
    match = matches[-1]
    projected_mib = int(match.group(1))
    free_mib = int(match.group(2))
    if projected_mib + memory_headroom_mib >= free_mib:
        return (
            "Terminating candidate early: projected device memory "
            f"{projected_mib} MiB is too close to free memory {free_mib} MiB "
            f"(headroom {memory_headroom_mib} MiB)"
        )
    return ""


def context_candidates(ctx_size: str) -> list[str]:
    raw = str(ctx_size).strip()
    try:
        configured = int(raw)
    except ValueError:
        return [raw]
    if configured <= 0:
        return [raw]

    values = [configured]
    values.extend(value for value in CHAT_LIKE_CTX_FALLBACKS if value < configured)
    return [str(value) for value in dict.fromkeys(values)]


def logs_indicate_memory_pressure(paths: list[Path]) -> bool:
    return any(startup_memory_pressure_message(path) for path in paths)


def path_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "default"


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
        log_fh.write(f"[panel] candidate log: {log_path}\n".encode("utf-8"))
        launch_argv, removed_flags = prepare_llama_server_argv(argv)
        write_compat_filter_notice(log_fh, removed_flags)
        log_fh.write(launch_diagnostics(f"{role} tune candidate", launch_argv, cwd=repo_dir()).encode("utf-8"))
        proc = subprocess.Popen(
            launch_argv,
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
        if not wait_for_server(host, port, startup_timeout, proc=proc, log_path=log_path):
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
            started = time.monotonic()
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
            elapsed = time.monotonic() - started
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            err(f"  Chat benchmark request failed (check {log_path})")
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
        if score <= 0:
            usage = response.get("usage")
            completion_tokens = 0
            if isinstance(usage, dict):
                try:
                    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                except (TypeError, ValueError):
                    completion_tokens = 0
            if completion_tokens > 0 and elapsed > 0:
                score = completion_tokens / elapsed
                log(
                    "  response did not include llama.cpp timings; "
                    f"estimated {score:.2f} tok/s from {completion_tokens} completion tokens over {elapsed:.2f}s"
                )
            else:
                err(f"  Chat benchmark response had no positive timings or token usage (check {log_path})")
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
        if not wait_for_server(host, port, startup_timeout, proc=proc, log_path=log_path):
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
    reset_startup_failure_state()
    panel_dir = repo_dir()
    config = load_config(panel_dir, role=args.role, apply_tune=False)
    validate_role_files(args.role, config)
    prefix = "CHAT" if args.role == "chat" else "EMBED" if args.role == "embed" else "VISION"

    tune_port = int(os.environ.get("TUNE_PORT", "9998"))
    tune_host = config["LLAMA_HOST"]
    startup_timeout = int(os.environ.get("TUNE_STARTUP_TIMEOUT", "120"))
    tune_dir = panel_dir / "bench-results" / "tuned"
    log_path = tune_dir / "server-tune.log"
    set_tune_log_path(log_path)
    log(f"Run log: {log_path}")

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
        ctx_values = context_candidates(config[f"{prefix}_CTX_SIZE"])
        total_runs = len(ctx_values) * len(candidates) * len(cache_combos)
        current = 0
        best_ctx_size = config[f"{prefix}_CTX_SIZE"]
        log(f"Context sweep: {' '.join(ctx_values)}")
        for ctx_index, ctx_size in enumerate(ctx_values):
            context_config = dict(config)
            context_config[f"{prefix}_CTX_SIZE"] = ctx_size
            context_best_score = -1.0
            context_logs: list[Path] = []
            log(f"Trying {prefix}_CTX_SIZE={ctx_size}")
            for threads in candidates:
                for cache_k, cache_v in cache_combos:
                    current += 1
                    candidate_log_path = tune_dir / (
                        f"{Path(config[f'{prefix}_MODEL']).stem}.{args.role}."
                        f"candidate-{current:02d}-ctx-{path_label(ctx_size)}-threads-{threads}-k-{cache_k}-v-{cache_v}.log"
                    )
                    context_logs.append(candidate_log_path)
                    log(f"[{current}/{total_runs}] ctx={ctx_size} threads={threads} cache_k={cache_k} cache_v={cache_v}")
                    log(f"  candidate log: {candidate_log_path}")
                    score = bench_chat_like(
                        args.role,
                        context_config,
                        host=tune_host,
                        port=tune_port,
                        threads=threads,
                        cache_k=cache_k,
                        cache_v=cache_v,
                        log_path=candidate_log_path,
                        prompt=BENCH_PROMPT if args.role == "chat" else VISION_BENCH_PROMPT,
                        startup_timeout=startup_timeout,
                    )
                    log(f"  -> {score:.2f} tok/s")
                    if score > context_best_score:
                        context_best_score = score
                    if score > best_score:
                        best_score = score
                        best_ctx_size = ctx_size
                        best_threads = str(threads)
                        best_cache_k = cache_k
                        best_cache_v = cache_v
            if context_best_score > 0:
                break
            if ctx_index >= len(ctx_values) - 1:
                break
            if logs_indicate_memory_pressure(context_logs):
                log(f"No working {args.role} candidate at {prefix}_CTX_SIZE={ctx_size}; retrying with lower context")
                continue
            log(
                f"No working {args.role} candidate at {prefix}_CTX_SIZE={ctx_size}, "
                "and candidate logs do not show memory pressure; not trying lower context sizes"
            )
            break

        log(
            f"Best: ctx={best_ctx_size} threads={best_threads} cache_k={best_cache_k} "
            f"cache_v={best_cache_v} ({best_score:.2f} tok/s)"
        )
        if best_score <= 0:
            message = no_working_config_message(args.role, prefix=prefix, log_path=log_path, tune_dir=tune_dir)
            err(message)
            raise PanelError(message)
        prefix = "CHAT" if args.role == "chat" else "VISION"
        write_tune_file(
            args.role,
            config,
            [
                f"export {prefix}_CTX_SIZE={best_ctx_size}",
                f"export {prefix}_THREADS={best_threads}",
                f"export {prefix}_CACHE_TYPE_K={best_cache_k}",
                f"export {prefix}_CACHE_TYPE_V={best_cache_v}",
            ],
            best_score=best_score,
            tested=current,
            unit="tok/s",
        )
        return 0

    total_runs = len(candidates)
    for index, threads in enumerate(candidates, start=1):
        candidate_log_path = tune_dir / f"{Path(config[f'{prefix}_MODEL']).stem}.{args.role}.candidate-{index:02d}-threads-{threads}.log"
        log(f"[{index}/{total_runs}] threads={threads}")
        log(f"  candidate log: {candidate_log_path}")
        score = bench_embed(
            config,
            host=tune_host,
            port=tune_port,
            threads=threads,
            log_path=candidate_log_path,
            startup_timeout=startup_timeout,
        )
        log(f"  -> {score:.2f} req/s")
        if score > best_score:
            best_score = score
            best_threads = str(threads)

    log(f"Best: threads={best_threads} ({best_score:.2f} req/s)")
    if best_score <= 0:
        message = no_working_config_message(args.role, prefix="EMBED", log_path=log_path, tune_dir=tune_dir)
        err(message)
        raise PanelError(message)
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
