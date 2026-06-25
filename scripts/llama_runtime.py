#!/usr/bin/env python3
"""Cross-platform helpers for launching the local llama-server panel."""

from __future__ import annotations

import contextlib
import errno
import json
import os
import re
import signal
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, Mapping, MutableMapping, Optional


ROLE_PREFIX = {
    "chat": "CHAT",
    "embed": "EMBED",
    "vision": "VISION",
}
GUI_OVERRIDE_FILE = "env.local.gui.json"
INLINE_LOGS_ENV = "PANEL_INLINE_LOGS"
COMPAT_FILTER_ENV = "LLAMA_SERVER_COMPAT_FILTER"
STARTUP_EXIT_GRACE_SECONDS = 1.0
STARTUP_LOG_TAIL_BYTES = 8192

OPTIONAL_LLAMA_ARG_VALUE_COUNTS = {
    "--n-cpu-moe": 1,
    "--cache-type-k": 1,
    "--cache-type-v": 1,
    "--flash-attn": 1,
    "--no-warmup": 0,
    "--cache-ram": 1,
    "--reasoning": 1,
    "--reasoning-format": 1,
    "--jinja": 0,
}

PORT_IN_USE_ERRNOS = {
    errno.EADDRINUSE,
    getattr(errno, "WSAEADDRINUSE", 10048),
}

ENV_ASSIGNMENT_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
VAR_RE = re.compile(r"\$(\w+)|\${([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?}")


class PanelError(Exception):
    pass


def repo_dir() -> Path:
    override = os.environ.get("LLAMA_SERVER_PANEL_DIR")
    if override:
        return Path(override).resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


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
        "JUGGLE_ROLE_PROXY_BIND_HOST": "127.0.0.1",
        "JUGGLE_CHAT_PROXY_BIND_HOST": "",
        "JUGGLE_EMBED_PROXY_BIND_HOST": "",
        "JUGGLE_VISION_PROXY_BIND_HOST": "",
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


def role_server_log_path(config: Mapping[str, str], role: str) -> Path:
    if role not in ROLE_PREFIX:
        raise PanelError(f"Unknown role: {role}")
    return Path(config["LOG_DIR"]) / f"{role}.log"


def inline_logs_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    env = env or os.environ
    return str(env.get(INLINE_LOGS_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


def compat_filter_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    env = env or os.environ
    return str(env.get(COMPAT_FILTER_ENV, "1")).strip().lower() not in {"0", "false", "no", "off"}


def _stream_binary_target(stream: object) -> Optional[BinaryIO]:
    if stream is None:
        return None
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        return buffer
    if hasattr(stream, "write") and hasattr(stream, "flush"):
        return stream  # type: ignore[return-value]
    return None


def write_output_chunk(log_fh: BinaryIO, chunk: bytes, *, stream: object = None) -> None:
    log_fh.write(chunk)
    log_fh.flush()
    target = _stream_binary_target(stream)
    if target is None:
        return
    target.write(chunk)
    target.flush()


def mirror_process_output(proc: subprocess.Popen[bytes], log_fh: BinaryIO, *, stream: object = None) -> None:
    stdout = proc.stdout
    if stdout is None:
        return
    with stdout:
        while True:
            chunk = stdout.read1(8192)
            if not chunk:
                break
            write_output_chunk(log_fh, chunk, stream=stream)


def looks_like_llama_server(command: str) -> bool:
    name = command.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name in {"llama-server", "llama-server.exe"}


def llama_server_help_text(binary: str, *, timeout: float = 8.0) -> Optional[str]:
    try:
        proc = subprocess.run(
            [binary, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            timeout=timeout,
            **popen_session_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout


def filter_unsupported_llama_args(argv: list[str], help_text: Optional[str]) -> tuple[list[str], list[str]]:
    if not argv or help_text is None:
        return list(argv), []

    filtered: list[str] = []
    removed: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        value_count = OPTIONAL_LLAMA_ARG_VALUE_COUNTS.get(item)
        if value_count is not None and item not in help_text:
            removed.append(item)
            index += 1 + value_count
            continue
        filtered.append(item)
        index += 1
    return filtered, removed


def prepare_llama_server_argv(argv: list[str], *, env: Optional[Mapping[str, str]] = None) -> tuple[list[str], list[str]]:
    if not argv or not compat_filter_enabled(env) or not looks_like_llama_server(argv[0]):
        return list(argv), []
    return filter_unsupported_llama_args(argv, llama_server_help_text(argv[0]))


def write_compat_filter_notice(log_fh: BinaryIO, removed_flags: list[str], *, stream: object = None) -> None:
    if not removed_flags:
        return
    flags = ", ".join(removed_flags)
    message = (
        "[panel] omitted unsupported optional llama-server arguments after inspecting "
        f"`llama-server --help`: {flags}\n"
    )
    write_output_chunk(log_fh, message.encode("utf-8"), stream=stream)


def tail_log_text(path: Path, *, max_bytes: int = STARTUP_LOG_TAIL_BYTES) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except FileNotFoundError:
        return ""
    return data.decode("utf-8", errors="replace")


def raise_if_process_exited(
    proc: subprocess.Popen[bytes],
    label: str,
    log_path: Path,
    *,
    grace_seconds: float = STARTUP_EXIT_GRACE_SECONDS,
) -> None:
    try:
        returncode = proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        return

    tail = tail_log_text(log_path).strip()
    message = f"{label} exited during startup with code {returncode}."
    if tail:
        message += f"\n\nLast log output from {log_path}:\n{tail}"
    else:
        message += f" Check {log_path} for details."
    raise PanelError(message)


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
        resolved_path = Path(resolved)
        if resolved_path.suffix.lower() in {".cmd", ".bat"} and resolved_path.stem.lower() in {
            "start-chat",
            "start-embed",
            "start-vision",
            "start-gui",
            "llama_role_command",
        }:
            raise PanelError(
                "LLAMA_SERVER_BIN must point to the llama-server executable, "
                f"not the panel launcher script: {resolved_path}"
            )
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
    tune_log_path = panel_dir / "bench-results" / "tuned" / "server-tune.log"
    tune_log_path.parent.mkdir(parents=True, exist_ok=True)
    auto_tune_script = panel_dir / "scripts" / "auto_tune.py"
    inline_stream = sys.stdout if inline_logs_enabled() else None
    if getattr(sys, "frozen", False):
        previous_stdout_log = os.environ.get("PANEL_AUTO_TUNE_STDOUT_LOG")
        os.environ["PANEL_AUTO_TUNE_STDOUT_LOG"] = "1"
        try:
            with tune_log_path.open("a", encoding="utf-8", errors="replace") as log_fh:
                preamble = launch_diagnostics(f"auto-tune {role}", ["<bundled>", "auto_tune", role], cwd=panel_dir)
                log_fh.write(preamble)
                log_fh.flush()
                if inline_stream is None:
                    stdout_target = log_fh
                    stderr_target = log_fh
                else:
                    class TeeTextIO:
                        def __init__(self, *targets: object) -> None:
                            self.targets = targets

                        def write(self, data: str) -> int:
                            for target in self.targets:
                                target.write(data)
                            return len(data)

                        def flush(self) -> None:
                            for target in self.targets:
                                target.flush()

                    stdout_target = TeeTextIO(log_fh, sys.stdout)
                    stderr_target = TeeTextIO(log_fh, sys.stderr)
                    sys.stdout.write(preamble)
                    sys.stdout.flush()
                with contextlib.redirect_stdout(stdout_target), contextlib.redirect_stderr(stderr_target):
                    import auto_tune

                    returncode = auto_tune.main([role])
        except Exception as exc:
            raise PanelError(
                f"auto-tune failed for {role} while creating {tune_path}. "
                f"Check {tune_log_path}. Original error: {exc}"
            ) from exc
        finally:
            if previous_stdout_log is None:
                os.environ.pop("PANEL_AUTO_TUNE_STDOUT_LOG", None)
            else:
                os.environ["PANEL_AUTO_TUNE_STDOUT_LOG"] = previous_stdout_log
    else:
        argv = [sys.executable, str(auto_tune_script), role]
        tune_env = os.environ.copy()
        tune_env["PANEL_AUTO_TUNE_STDOUT_LOG"] = "1"
        with tune_log_path.open("ab", buffering=0) as log_fh:
            preamble = launch_diagnostics(f"auto-tune {role}", argv, cwd=panel_dir).encode("utf-8")
            write_output_chunk(log_fh, preamble, stream=inline_stream)
            if inline_stream is None:
                proc = subprocess.run(
                    argv,
                    cwd=str(panel_dir),
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    check=False,
                    env=tune_env,
                )
                returncode = proc.returncode
            else:
                proc = subprocess.Popen(
                    argv,
                    cwd=str(panel_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=tune_env,
                    **popen_session_kwargs(),
                )
                try:
                    mirror_process_output(proc, log_fh, stream=inline_stream)
                    returncode = proc.wait()
                except KeyboardInterrupt:
                    terminate_process(proc)
                    raise
    if returncode != 0:
        raise PanelError(
            f"auto-tune failed for {role} while creating {tune_path}. "
            f"Check {tune_log_path}."
        )


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
    batch_size_override: Optional[int] = None,
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
        batch_size_override=batch_size_override,
    )


def build_role_argv_for_config(
    config: Mapping[str, str],
    role: str,
    *,
    host: str,
    port: str,
    batch_size_override: Optional[int] = None,
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
        batch_value = config["EMBED_BATCH_SIZE"]
        ubatch_value = config["EMBED_UBATCH_SIZE"]
        ctx_value = config["EMBED_CTX_SIZE"]
        if batch_size_override is not None:
            # Escalated batch size (try-and-error path for oversized chunks).
            # Bump both the logical and physical batch so a single large chunk
            # fits, and grow the context window to at least the batch size since
            # the chunk must also fit within n_ctx.
            override_text = str(batch_size_override)
            batch_value = override_text
            ubatch_value = override_text
            try:
                if int(ctx_value) < batch_size_override:
                    ctx_value = override_text
            except (TypeError, ValueError):
                ctx_value = override_text
        return [
            binary,
            "--model",
            config["EMBED_MODEL"],
            "--host",
            host,
            "--port",
            port,
            "--ctx-size",
            ctx_value,
            "--batch-size",
            batch_value,
            "--ubatch-size",
            ubatch_value,
            "--threads",
            config["EMBED_THREADS"],
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


def run_role_argv_with_log(role: str, argv: list[str], *, panel_dir: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    inline_stream = sys.stdout if inline_logs_enabled() else None
    with log_path.open("ab", buffering=0) as log_fh:
        launch_argv, removed_flags = prepare_llama_server_argv(argv)
        write_compat_filter_notice(log_fh, removed_flags, stream=inline_stream)
        write_output_chunk(
            log_fh,
            launch_diagnostics(f"{role} llama-server", launch_argv, cwd=panel_dir).encode("utf-8"),
            stream=inline_stream,
        )
        popen_kwargs = popen_session_kwargs()
        if inline_stream is None:
            proc = subprocess.Popen(
                launch_argv,
                cwd=str(panel_dir),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                **popen_kwargs,
            )
        else:
            proc = subprocess.Popen(
                launch_argv,
                cwd=str(panel_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                **popen_kwargs,
            )
        write_output_chunk(
            log_fh,
            launch_diagnostics(f"{role} llama-server", launch_argv, cwd=panel_dir, pid=proc.pid).encode("utf-8"),
            stream=inline_stream,
        )
        try:
            if inline_stream is not None:
                mirror_process_output(proc, log_fh, stream=inline_stream)
            return proc.wait()
        except KeyboardInterrupt:
            terminate_process(proc)
            raise


def popen_session_kwargs() -> Dict[str, object]:
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return {
            "creationflags": creationflags,
        }
    return {"start_new_session": True}


def format_command_for_log(argv: Iterable[str]) -> str:
    parts = [str(part) for part in argv]
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def launch_diagnostics(label: str, argv: Iterable[str], *, cwd: Path, pid: Optional[int] = None) -> str:
    action = "started" if pid is not None else "launching"
    lines = [
        "",
        f"[panel] {time.strftime('%Y-%m-%d %H:%M:%S')} {action} {label}",
        f"[panel] cwd: {cwd}",
        f"[panel] command: {format_command_for_log(argv)}",
    ]
    if pid is not None:
        lines.append(f"[panel] pid: {pid}")
    lines.append("")
    return "\n".join(lines)


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
