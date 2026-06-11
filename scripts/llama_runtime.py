#!/usr/bin/env python3
"""Cross-platform helpers for launching the local llama-server panel."""

from __future__ import annotations

import errno
import json
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional


ROLE_PREFIX = {
    "chat": "CHAT",
    "embed": "EMBED",
    "vision": "VISION",
}
GUI_OVERRIDE_FILE = "env.local.gui.json"

PORT_IN_USE_ERRNOS = {
    errno.EADDRINUSE,
    getattr(errno, "WSAEADDRINUSE", 10048),
}

ENV_ASSIGNMENT_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
VAR_RE = re.compile(r"\$(\w+)|\${([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?}")


class PanelError(Exception):
    pass


def repo_dir() -> Path:
    return Path(os.environ.get("LLAMA_SERVER_PANEL_DIR", Path(__file__).resolve().parents[1])).resolve()


def default_llama_server_bin() -> str:
    return "llama-server"


def default_config(panel_dir: Optional[Path] = None) -> Dict[str, str]:
    panel_dir = (panel_dir or repo_dir()).resolve()
    model_dir = panel_dir / "models"
    return {
        "LLAMA_SERVER_PANEL_DIR": str(panel_dir),
        "LLAMA_SERVER_BIN": default_llama_server_bin(),
        "LLAMA_HOST": "127.0.0.1",
        "MODEL_DIR": str(model_dir),
        "LOG_DIR": str(panel_dir / "logs"),
        "CHAT_MODEL": "Mistral-Small-3.2-24B-Instruct-2506-BF16.gguf",
        "CHAT_PORT": "8080",
        "CHAT_CTX_SIZE": "128000",
        "CHAT_THREADS": "8",
        "CHAT_PARALLEL": "1",
        "CHAT_ALIAS": "qwen3-30b-a3b-thinking-2507",
        "CHAT_CACHE_TYPE_K": "q8_0",
        "CHAT_CACHE_TYPE_V": "q8_0",
        "CHAT_CPU_MOE_LAYERS": "40",
        "CHAT_TEMPERATURE": "0.6",
        "CHAT_TOP_K": "20",
        "CHAT_TOP_P": "0.95",
        "CHAT_MIN_P": "0",
        "CHAT_PRESENCE_PENALTY": "1.5",
        "EMBED_MODEL": "Qwen3-Embedding-4B-Q6_K.gguf",
        "EMBED_PORT": "8081",
        "EMBED_CTX_SIZE": "4096",
        "EMBED_THREADS": "8",
        "EMBED_BATCH_SIZE": "4096",
        "EMBED_UBATCH_SIZE": "4096",
        "EMBED_POOLING": "mean",
        "VISION_MODEL": "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf",
        "VISION_MMPROJ": "mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf",
        "VISION_ALIAS": "qwen2.5-vl-3b-instruct",
        "VISION_PORT": "8082",
        "VISION_CTX_SIZE": "256000",
        "VISION_THREADS": "8",
        "VISION_CACHE_TYPE_K": "f16",
        "VISION_CACHE_TYPE_V": "f16",
    }


def expand_shell_like(value: str, scope: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        bare = match.group(1)
        named = match.group(2)
        default = match.group(3)
        key = bare or named or ""
        if key in scope:
            return scope[key]
        if key in os.environ:
            return os.environ[key]
        return default or ""

    return VAR_RE.sub(replace, value)


def parse_assignment_value(raw: str, scope: Mapping[str, str]) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        inner = value[1:-1]
        if value[0] == "'":
            return inner
        return expand_shell_like(inner, scope)
    return expand_shell_like(value, scope)


def load_env_assignments(path: Path, config: MutableMapping[str, str]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ENV_ASSIGNMENT_RE.match(stripped)
        if not match:
            continue
        key = match.group(1)
        config[key] = parse_assignment_value(match.group(2), config)


def load_json_assignments(path: Path, config: MutableMapping[str, str]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except json.JSONDecodeError as exc:
        raise PanelError(f"Invalid JSON in {path.name}: {exc}") from exc

    if not isinstance(payload, dict):
        raise PanelError(f"{path.name} must contain a JSON object.")

    for key, value in payload.items():
        if value is None:
            continue
        config[str(key)] = str(value)


def apply_process_environment(config: MutableMapping[str, str]) -> None:
    for key in list(config.keys()):
        value = os.environ.get(key)
        if value is not None:
            config[key] = value


def normalize_path(path_text: str, *, panel_dir: Path) -> str:
    candidate = Path(os.path.expanduser(path_text))
    if not candidate.is_absolute():
        candidate = panel_dir / candidate
    return str(candidate.resolve())


def normalize_config_paths(config: MutableMapping[str, str]) -> None:
    panel_dir = Path(config["LLAMA_SERVER_PANEL_DIR"]).resolve()
    config["LLAMA_SERVER_PANEL_DIR"] = str(panel_dir)
    config["MODEL_DIR"] = normalize_path(config["MODEL_DIR"], panel_dir=panel_dir)
    config["LOG_DIR"] = normalize_path(config["LOG_DIR"], panel_dir=panel_dir)

    model_dir = Path(config["MODEL_DIR"])
    for key in ("CHAT_MODEL", "EMBED_MODEL", "VISION_MODEL", "VISION_MMPROJ"):
        value = config.get(key, "")
        if not value:
            continue
        candidate = Path(os.path.expanduser(value))
        if not candidate.is_absolute():
            candidate = model_dir / candidate
        config[key] = str(candidate.resolve())


def tune_file_path(config: Mapping[str, str], role: str) -> Path:
    prefix = ROLE_PREFIX[role]
    model_name = Path(config[f"{prefix}_MODEL"]).name
    stem = model_name[:-5] if model_name.endswith(".gguf") else Path(model_name).stem
    return Path(config["LLAMA_SERVER_PANEL_DIR"]) / "bench-results" / "tuned" / f"{stem}.{role}.sh"


def load_config(panel_dir: Optional[Path] = None, *, role: Optional[str] = None, apply_tune: bool = True) -> Dict[str, str]:
    panel_dir = (panel_dir or repo_dir()).resolve()
    config = default_config(panel_dir)
    apply_process_environment(config)

    load_env_assignments(panel_dir / "env.local.env", config)
    load_json_assignments(panel_dir / "env.local.json", config)
    load_env_assignments(panel_dir / "env.local.sh", config)
    load_json_assignments(panel_dir / GUI_OVERRIDE_FILE, config)
    normalize_config_paths(config)

    if role and apply_tune:
        tune_path = tune_file_path(config, role)
        load_env_assignments(tune_path, config)
        normalize_config_paths(config)

    return config


def resolve_executable(command: str) -> Optional[str]:
    expanded = os.path.expanduser(command)
    path_like = any(separator in command for separator in (os.sep, "/", "\\"))
    candidate = Path(expanded)
    if candidate.is_absolute() or path_like:
        if candidate.exists():
            return str(candidate)
        return None
    return shutil.which(command)


def ensure_llama_server_binary(config: Mapping[str, str]) -> str:
    binary = config["LLAMA_SERVER_BIN"]
    resolved = resolve_executable(binary)
    if resolved:
        return resolved
    raise PanelError(f"Missing llama-server binary at: {binary}")


def validate_role_files(role: str, config: Mapping[str, str]) -> None:
    ensure_llama_server_binary(config)

    prefix = ROLE_PREFIX[role]
    model_path = Path(config[f"{prefix}_MODEL"])
    if not model_path.is_file():
        label = "embedding" if role == "embed" else role
        raise PanelError(f"Missing {label} model at: {model_path}")

    if role == "vision":
        mmproj = Path(config.get("VISION_MMPROJ", ""))
        if not mmproj.is_file():
            raise PanelError(
                "Missing vision mmproj at: "
                f"{config.get('VISION_MMPROJ', '<unset>')}. "
                f"Configured VISION_MODEL is {model_path}. "
                "Install the matching mmproj file or set VISION_MMPROJ in env.local.sh/env.local.env."
            )


def port_in_use(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    bind_host = "::" if host == "::" else host

    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, port))
        return False
    except OSError as exc:
        if exc.errno in PORT_IN_USE_ERRNOS:
            return True

    connect_host = "127.0.0.1" if host in {"0.0.0.0", "localhost"} else host
    try:
        with socket.create_connection((connect_host, port), timeout=0.4):
            return True
    except OSError:
        return False


def tune_file_exists(role: str, panel_dir: Optional[Path] = None) -> bool:
    config = load_config(panel_dir, role=role, apply_tune=False)
    return tune_file_path(config, role).is_file()


def ensure_tune_file(role: str, panel_dir: Optional[Path] = None) -> None:
    panel_dir = (panel_dir or repo_dir()).resolve()
    config = load_config(panel_dir, role=role, apply_tune=False)
    tune_path = tune_file_path(config, role)
    if tune_path.is_file():
        return

    validate_role_files(role, config)
    auto_tune_script = panel_dir / "scripts" / "auto_tune.py"
    proc = subprocess.run(
        [sys.executable, str(auto_tune_script), role],
        cwd=str(panel_dir),
        check=False,
    )
    if proc.returncode != 0:
        raise PanelError(f"auto-tune failed for {role}")


def role_environment(
    role: str,
    *,
    panel_dir: Optional[Path] = None,
    port_override: Optional[int] = None,
    host_override: Optional[str] = None,
) -> Dict[str, str]:
    config = load_config(panel_dir, role=role)
    prefix = ROLE_PREFIX[role]

    result = {
        "ROLE": role,
        "LLAMA_HOST": host_override or config["LLAMA_HOST"],
        "LOG_DIR": config["LOG_DIR"],
        "PORT": str(port_override or config[f"{prefix}_PORT"]),
        "MODEL": config[f"{prefix}_MODEL"],
        "CTX_SIZE": config[f"{prefix}_CTX_SIZE"],
        "THREADS": config[f"{prefix}_THREADS"],
        "TUNE_FILE": str(tune_file_path(config, role)),
    }
    if role in {"chat", "vision"}:
        result["ALIAS"] = config[f"{prefix}_ALIAS"]
    if role == "vision":
        result["MMPROJ"] = config["VISION_MMPROJ"]
    return result


def build_role_argv(
    role: str,
    *,
    panel_dir: Optional[Path] = None,
    port_override: Optional[int] = None,
    host_override: Optional[str] = None,
    auto_tune: bool = False,
) -> list[str]:
    panel_dir = (panel_dir or repo_dir()).resolve()
    if auto_tune:
        ensure_tune_file(role, panel_dir)
    config = load_config(panel_dir, role=role)
    prefix = ROLE_PREFIX[role]
    return build_role_argv_for_config(
        config,
        role,
        host=host_override or config["LLAMA_HOST"],
        port=str(port_override or config[f"{prefix}_PORT"]),
    )


def build_role_argv_for_config(
    config: Mapping[str, str],
    role: str,
    *,
    host: str,
    port: str,
) -> list[str]:
    binary = ensure_llama_server_binary(config)
    if role == "chat":
        return [
            binary,
            "--model",
            config["CHAT_MODEL"],
            "--alias",
            config["CHAT_ALIAS"],
            "--host",
            host,
            "--port",
            port,
            "--ctx-size",
            config["CHAT_CTX_SIZE"],
            "--threads",
            config["CHAT_THREADS"],
            "--parallel",
            config["CHAT_PARALLEL"],
            "--n-cpu-moe",
            config["CHAT_CPU_MOE_LAYERS"],
            "--no-mmap",
            "--cache-type-k",
            config["CHAT_CACHE_TYPE_K"],
            "--cache-type-v",
            config["CHAT_CACHE_TYPE_V"],
            "--flash-attn",
            "on",
            "--no-warmup",
            "--cache-ram",
            "0",
            "--reasoning",
            "on",
            "--reasoning-format",
            "deepseek",
            "--temp",
            config["CHAT_TEMPERATURE"],
            "--top-k",
            config["CHAT_TOP_K"],
            "--top-p",
            config["CHAT_TOP_P"],
            "--min-p",
            config["CHAT_MIN_P"],
            "--presence-penalty",
            config["CHAT_PRESENCE_PENALTY"],
            "--jinja",
        ]

    if role == "embed":
        return [
            binary,
            "--model",
            config["EMBED_MODEL"],
            "--host",
            host,
            "--port",
            port,
            "--ctx-size",
            config["EMBED_CTX_SIZE"],
            "--batch-size",
            config["EMBED_BATCH_SIZE"],
            "--ubatch-size",
            config["EMBED_UBATCH_SIZE"],
            "--threads",
            config["EMBED_THREADS"],
            "--batch-size",
            config["EMBED_BATCH_SIZE"],
            "--ubatch-size",
            config["EMBED_UBATCH_SIZE"],
            "--embedding",
            "--pooling",
            config["EMBED_POOLING"],
        ]

    if role == "vision":
        return [
            binary,
            "--model",
            config["VISION_MODEL"],
            "--alias",
            config["VISION_ALIAS"],
            "--host",
            host,
            "--port",
            port,
            "--ctx-size",
            config["VISION_CTX_SIZE"],
            "--threads",
            config["VISION_THREADS"],
            "--no-mmap",
            "--cache-type-k",
            config["VISION_CACHE_TYPE_K"],
            "--cache-type-v",
            config["VISION_CACHE_TYPE_V"],
            "--flash-attn",
            "on",
            "--no-warmup",
            "--mmproj",
            config["VISION_MMPROJ"],
            "--jinja",
        ]

    raise PanelError(f"Unknown role: {role}")


def popen_session_kwargs() -> Dict[str, object]:
    if os.name == "nt":
        return {
            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        }
    return {"start_new_session": True}


def terminate_process(proc: subprocess.Popen[bytes], *, terminate_timeout: float = 20.0, kill_timeout: float = 10.0) -> None:
    if proc.poll() is not None:
        return

    if os.name == "nt":
        proc.terminate()
        try:
            proc.wait(timeout=terminate_timeout)
            return
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=kill_timeout)
            return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        proc.terminate()

    try:
        proc.wait(timeout=terminate_timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait(timeout=kill_timeout)


def write_env0_records(records: Mapping[str, str]) -> bytes:
    payload = bytearray()
    for key, value in records.items():
        payload.extend(f"{key}={value}".encode("utf-8"))
        payload.append(0)
    return bytes(payload)


def host_label() -> str:
    return socket.gethostname().split(".", 1)[0]


def cpu_counts() -> tuple[int, int]:
    total = os.cpu_count() or 8
    perf = total
    if sys.platform == "darwin":
        try:
            perf_output = subprocess.check_output(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            perf = int(perf_output)
        except (OSError, ValueError, subprocess.CalledProcessError):
            perf = total
    return total, max(1, perf)


def thread_candidates() -> list[int]:
    total, perf = cpu_counts()
    values = {total}
    if total // 4 >= 2:
        values.add(total // 4)
    if total // 2 >= 2:
        values.add(total // 2)
    if perf >= 2:
        values.add(perf)
    return sorted(values)
