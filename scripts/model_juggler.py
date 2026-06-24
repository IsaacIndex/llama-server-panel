#!/usr/bin/env python3
"""Proxy/supervisor for juggling chat and vision llama-server processes."""

from __future__ import annotations

import argparse
import atexit
import http.client
import ipaddress
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlsplit

from llama_runtime import (
    PanelError,
    build_role_argv,
    inline_logs_enabled,
    launch_diagnostics,
    load_config,
    mirror_process_output,
    popen_session_kwargs,
    prepare_llama_server_argv,
    port_in_use as runtime_port_in_use,
    raise_if_process_exited,
    role_environment,
    tail_log_text,
    terminate_process as terminate_runtime_process,
    validate_role_files,
    write_compat_filter_notice,
    write_output_chunk,
)


HEAVY_ROLES = {"chat", "vision"}
GATEWAY_DEFAULT_BIND = "127.0.0.1"
GATEWAY_DEFAULT_PORT = 8088
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class ModelBusy(Exception):
    pass


class StartupError(Exception):
    pass


def repo_dir() -> Path:
    return Path(os.environ.get("LLAMA_SERVER_PANEL_DIR", Path(__file__).resolve().parents[1])).resolve()


REPO_DIR = repo_dir()


def helper_env(role: str, *, port: Optional[int] = None, host: Optional[str] = None) -> Dict[str, str]:
    try:
        return role_environment(role, panel_dir=REPO_DIR, port_override=port, host_override=host)
    except PanelError as exc:
        raise StartupError(str(exc)) from exc


def helper_argv(role: str, *, port: int, host: str, auto_tune: bool) -> List[str]:
    try:
        return build_role_argv(role, panel_dir=REPO_DIR, port_override=port, host_override=host, auto_tune=auto_tune)
    except PanelError as exc:
        raise StartupError(str(exc)) from exc


def helper_check(role: str) -> None:
    try:
        validate_role_files(role, role_environment_to_config(role))
    except PanelError as exc:
        raise StartupError(str(exc)) from exc


def role_environment_to_config(role: str) -> Dict[str, str]:
    return load_config(REPO_DIR, role=role)


def port_is_open(host: str, port: int) -> bool:
    return runtime_port_in_use(host, port)


def llama_ready(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", "/v1/models", headers={"Connection": "close"})
        response = conn.getresponse()
        response.read()
        return 200 <= response.status < 300
    except OSError:
        return False
    finally:
        try:
            conn.close()  # type: ignore[name-defined]
        except Exception:
            pass


def wait_ready(
    host: str,
    port: int,
    timeout: float,
    *,
    proc: Optional[subprocess.Popen[bytes]] = None,
    log_path: Optional[Path] = None,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            message = f"llama-server on {host}:{port} exited during startup with code {proc.returncode}"
            if log_path is not None:
                tail = tail_log_text(log_path).strip()
                if tail:
                    message += f"\n\nLast log output from {log_path}:\n{tail}"
            raise StartupError(message)
        if llama_ready(host, port):
            return
        time.sleep(1)
    raise StartupError(f"llama-server on {host}:{port} did not become ready within {int(timeout)}s")


@dataclass
class RoleRuntime:
    role: str
    public_port: int
    backend_port: int
    host: str
    bind_host: str
    log_path: Path
    external: bool = False
    process: Optional[subprocess.Popen[bytes]] = None
    model_path: str = ""
    alias: str = ""


@dataclass(frozen=True)
class LocalAddress:
    interface: str
    address: str
    status: str


def mirror_runtime_output(proc: subprocess.Popen[bytes], log_path: Path, *, stream: object = None) -> None:
    with log_path.open("ab", buffering=0) as log_fh:
        mirror_process_output(proc, log_fh, stream=stream)


class JugglerState:
    def __init__(
        self,
        roles: Dict[str, RoleRuntime],
        *,
        auto_tune: bool,
        switch_timeout: float,
        startup_timeout: float,
        request_timeout: float,
    ) -> None:
        self.roles = roles
        self.auto_tune = auto_tune
        self.switch_timeout = switch_timeout
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.switch_lock = threading.Lock()
        self.request_cv = threading.Condition()
        self.active_heavy: Optional[str] = None
        self.active_requests = {"chat": 0, "vision": 0}

    def validate_files(self) -> None:
        for role in ("chat", "embed", "vision"):
            helper_check(role)

    def start_embed_baseline(self) -> None:
        runtime = self.roles["embed"]
        if runtime.external:
            return
        with self.switch_lock:
            self.ensure_process("embed")

    def prepare_request(self, role: str) -> int:
        runtime = self.roles[role]
        if runtime.external:
            return runtime.public_port

        if role not in HEAVY_ROLES:
            with self.switch_lock:
                self.ensure_process(role)
            return runtime.backend_port

        deadline = time.monotonic() + self.switch_timeout
        acquired = self.switch_lock.acquire(timeout=self.switch_timeout)
        if not acquired:
            raise ModelBusy("model switch lock timed out")

        try:
            other = "vision" if role == "chat" else "chat"
            with self.request_cv:
                while self.active_requests[other] > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ModelBusy(f"{other} request still running")
                    self.request_cv.wait(timeout=min(1.0, remaining))

            if self.active_heavy and self.active_heavy != role:
                self.stop_process(self.active_heavy)
                self.active_heavy = None

            self.ensure_process(role)
            self.active_heavy = role

            with self.request_cv:
                self.active_requests[role] += 1
            return runtime.backend_port
        finally:
            self.switch_lock.release()

    def finish_request(self, role: str) -> None:
        if role not in HEAVY_ROLES:
            return
        with self.request_cv:
            self.active_requests[role] = max(0, self.active_requests[role] - 1)
            self.request_cv.notify_all()

    def ensure_process(self, role: str) -> None:
        runtime = self.roles[role]
        if runtime.external:
            return

        if runtime.process and runtime.process.poll() is None and llama_ready(runtime.host, runtime.backend_port):
            return

        if runtime.process and runtime.process.poll() is not None:
            runtime.process = None

        if port_is_open(runtime.host, runtime.backend_port):
            if llama_ready(runtime.host, runtime.backend_port):
                return
            raise StartupError(f"backend port {runtime.host}:{runtime.backend_port} is already in use")

        runtime.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(runtime.log_path, "ab", buffering=0)
        inline_stream = sys.stdout if inline_logs_enabled() else None
        try:
            write_output_chunk(
                log_fh,
                f"[panel] preparing {role} backend on {runtime.host}:{runtime.backend_port}\n".encode("utf-8"),
                stream=inline_stream,
            )
            argv = helper_argv(role, port=runtime.backend_port, host=runtime.host, auto_tune=self.auto_tune)
            launch_argv, removed_flags = prepare_llama_server_argv(argv)
            write_compat_filter_notice(log_fh, removed_flags, stream=inline_stream)
            write_output_chunk(
                log_fh,
                launch_diagnostics(f"{role} backend", launch_argv, cwd=REPO_DIR).encode("utf-8"),
                stream=inline_stream,
            )
            if inline_stream is None:
                runtime.process = subprocess.Popen(
                    launch_argv,
                    cwd=str(REPO_DIR),
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    **popen_session_kwargs(),
                )
            else:
                runtime.process = subprocess.Popen(
                    launch_argv,
                    cwd=str(REPO_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    **popen_session_kwargs(),
                )
            write_output_chunk(
                log_fh,
                launch_diagnostics(f"{role} backend", launch_argv, cwd=REPO_DIR, pid=runtime.process.pid).encode("utf-8"),
                stream=inline_stream,
            )
            if inline_stream is not None and runtime.process.stdout is not None:
                threading.Thread(
                    target=mirror_runtime_output,
                    args=(runtime.process, runtime.log_path),
                    kwargs={"stream": inline_stream},
                    daemon=True,
                ).start()
        except Exception as exc:
            write_output_chunk(
                log_fh,
                f"[panel] startup failed: {exc}\n".encode("utf-8", errors="replace"),
                stream=inline_stream,
            )
            raise
        finally:
            log_fh.close()

        if runtime.process is not None:
            try:
                raise_if_process_exited(runtime.process, f"{role} backend", runtime.log_path)
            except PanelError as exc:
                raise StartupError(str(exc)) from exc

        try:
            wait_ready(
                runtime.host,
                runtime.backend_port,
                self.startup_timeout,
                proc=runtime.process,
                log_path=runtime.log_path,
            )
        except Exception as exc:
            with runtime.log_path.open("ab", buffering=0) as failure_log_fh:
                failure_log_fh.write(f"[panel] startup failed: {exc}\n".encode("utf-8", errors="replace"))
            raise

    def stop_process(self, role: str) -> None:
        runtime = self.roles[role]
        proc = runtime.process
        if not proc or proc.poll() is not None:
            runtime.process = None
            return

        try:
            terminate_runtime_process(proc)
        finally:
            runtime.process = None

    def shutdown(self) -> None:
        for role in ("chat", "vision", "embed"):
            runtime = self.roles.get(role)
            if runtime and not runtime.external:
                self.stop_process(role)


def read_request_body(handler: BaseHTTPRequestHandler) -> bytes:
    length_text = handler.headers.get("Content-Length", "0") or "0"
    try:
        length = int(length_text)
    except ValueError:
        raise ValueError(f"invalid Content-Length: {length_text}")
    return handler.rfile.read(length) if length else b""


def proxy_request(
    handler: BaseHTTPRequestHandler,
    state: JugglerState,
    role: str,
    *,
    body: Optional[bytes] = None,
) -> None:
    runtime = state.roles[role]
    backend_port: Optional[int] = None
    prepared = False

    try:
        backend_port = state.prepare_request(role)
        prepared = True
        if body is None:
            body = read_request_body(handler)

        headers = {
            key: value
            for key, value in handler.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        headers["Host"] = f"{runtime.host}:{backend_port}"
        headers["Connection"] = "close"

        conn = http.client.HTTPConnection(runtime.host, backend_port, timeout=state.request_timeout)
        conn.request(handler.command, handler.path, body=body, headers=headers)
        response = conn.getresponse()

        handler.send_response(response.status, response.reason)
        sent_connection = False
        for key, value in response.getheaders():
            lower_key = key.lower()
            if lower_key in HOP_BY_HOP_HEADERS:
                continue
            if lower_key == "connection":
                sent_connection = True
                continue
            handler.send_header(key, value)
        if not sent_connection:
            handler.send_header("Connection", "close")
        handler.end_headers()

        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            handler.wfile.write(chunk)
            handler.wfile.flush()
        conn.close()
    except ModelBusy as exc:
        write_error(handler, 503, f"model switch busy: {exc}")
    except StartupError as exc:
        write_error(handler, 503, str(exc))
    except Exception as exc:
        write_error(handler, 502, f"proxy error: {exc}")
    finally:
        if prepared:
            state.finish_request(role)
        handler.close_connection = True


def write_json(handler: BaseHTTPRequestHandler, status: int, payload: object) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(data)


def write_error(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    write_json(handler, status, {"error": {"message": message, "type": "model_juggler"}})


def role_model_id(runtime: RoleRuntime) -> str:
    if runtime.alias:
        return runtime.alias
    if runtime.model_path:
        return Path(runtime.model_path).name
    return runtime.role


def role_model_ids(runtime: RoleRuntime) -> set[str]:
    ids = {role_model_id(runtime)}
    if runtime.alias:
        ids.add(runtime.alias)
    if runtime.model_path:
        ids.add(runtime.model_path)
        ids.add(Path(runtime.model_path).name)
    return {value for value in ids if value}


def combined_models_payload(roles: Dict[str, RoleRuntime]) -> Dict[str, object]:
    data = []
    seen: set[str] = set()
    for role in ("chat", "embed", "vision"):
        model_id = role_model_id(roles[role])
        if model_id in seen:
            continue
        seen.add(model_id)
        data.append(
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "llama.cpp",
            }
        )
    return {"object": "list", "data": data}


def request_payload(body: bytes) -> Dict[str, object]:
    if not body:
        return {}
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def contains_image_content(value: object) -> bool:
    if isinstance(value, dict):
        content_type = str(value.get("type", "")).lower()
        if content_type in {"image_url", "input_image"}:
            return True
        if "image_url" in value:
            return True
        return any(contains_image_content(child) for child in value.values())
    if isinstance(value, list):
        return any(contains_image_content(item) for item in value)
    return False


def gateway_role_for_request(path: str, body: bytes, roles: Dict[str, RoleRuntime]) -> Optional[str]:
    request_path = urlsplit(path).path
    if request_path == "/v1/embeddings":
        return "embed"
    if request_path != "/v1/chat/completions":
        return None

    payload = request_payload(body)
    requested_model = payload.get("model")
    if isinstance(requested_model, str) and requested_model in role_model_ids(roles["vision"]):
        return "vision"
    if contains_image_content(payload.get("messages")):
        return "vision"
    return "chat"


def make_gateway_handler(state: JugglerState):
    class GatewayHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            if urlsplit(self.path).path == "/v1/models":
                write_json(self, 200, combined_models_payload(state.roles))
                return
            write_error(self, 404, f"gateway path not found: {urlsplit(self.path).path}")

        def do_POST(self) -> None:
            try:
                body = read_request_body(self)
            except ValueError as exc:
                write_error(self, 400, str(exc))
                return

            role = gateway_role_for_request(self.path, body, state.roles)
            if role is None:
                write_error(self, 404, f"gateway path not found: {urlsplit(self.path).path}")
                return
            proxy_request(self, state, role, body=body)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Allow", "GET,POST,OPTIONS")
            self.send_header("Connection", "close")
            self.end_headers()

        def log_message(self, fmt: str, *args: object) -> None:
            sys.stderr.write(f"[gateway] {self.address_string()} - {fmt % args}\n")

    return GatewayHandler


def make_handler(role: str, state: JugglerState):
    class RoleProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            proxy_request(self, state, role)

        def do_POST(self) -> None:
            proxy_request(self, state, role)

        def do_OPTIONS(self) -> None:
            proxy_request(self, state, role)

        def log_message(self, fmt: str, *args: object) -> None:
            sys.stderr.write(f"[{role}] {self.address_string()} - {fmt % args}\n")

    return RoleProxyHandler


def parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


ROLE_PROXY_BIND_DEFAULT = "127.0.0.1"
ROLE_PROXY_BIND_KEY = "JUGGLE_ROLE_PROXY_BIND_HOST"
ROLE_PROXY_BIND_KEYS = {
    "chat": "JUGGLE_CHAT_PROXY_BIND_HOST",
    "embed": "JUGGLE_EMBED_PROXY_BIND_HOST",
    "vision": "JUGGLE_VISION_PROXY_BIND_HOST",
}


def resolve_role_proxy_bind_hosts(
    *,
    config: Optional[Dict[str, str]] = None,
    global_override: Optional[str] = None,
    role_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    config = config or load_config(REPO_DIR, apply_tune=False)
    global_bind = (global_override or config.get(ROLE_PROXY_BIND_KEY) or ROLE_PROXY_BIND_DEFAULT).strip()
    if not global_bind:
        global_bind = ROLE_PROXY_BIND_DEFAULT
    role_overrides = role_overrides or {}

    resolved: Dict[str, str] = {}
    for role in ("chat", "embed", "vision"):
        configured = role_overrides.get(role)
        if configured is None:
            configured = config.get(ROLE_PROXY_BIND_KEYS[role], "")
        configured = configured.strip()
        resolved[role] = configured or global_bind
    return resolved


def bind_exposes_network(bind_host: str) -> bool:
    return bind_host not in {"127.0.0.1", "::1", "localhost"}


def check_gateway_port(bind_host: str, port: int) -> None:
    if port_is_open(bind_host, port):
        raise StartupError(f"gateway port {bind_host}:{port} is already in use")


def address_sort_key(address: LocalAddress) -> tuple[int, str, str]:
    ip = ipaddress.ip_address(address.address)
    if ip.is_link_local:
        rank = 0
    elif address.interface.startswith("bridge"):
        rank = 1
    elif ip.is_private:
        rank = 2
    else:
        rank = 3
    inactive_penalty = 10 if address.status == "inactive" else 0
    return (rank + inactive_penalty, address.interface, address.address)


def local_ipv4_addresses() -> List[LocalAddress]:
    if os.name == "nt":
        return local_ipv4_addresses_windows()
    return local_ipv4_addresses_unix()


def local_ipv4_addresses_unix() -> List[LocalAddress]:
    try:
        proc = subprocess.run(
            ["ifconfig"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []

    blocks: List[List[str]] = []
    current: List[str] = []
    for line in proc.stdout.splitlines():
        if line and not line[0].isspace() and ":" in line.split()[0]:
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)

    addresses: List[LocalAddress] = []
    for block in blocks:
        interface = block[0].split(":", 1)[0]
        status = "unknown"
        for line in block:
            match = re.search(r"\bstatus:\s+(\S+)", line)
            if match:
                status = match.group(1)
                break
        for line in block:
            match = re.search(r"\binet\s+(\d+\.\d+\.\d+\.\d+)\b", line)
            if not match:
                continue
            address = match.group(1)
            try:
                ip = ipaddress.ip_address(address)
            except ValueError:
                continue
            if ip.is_loopback or ip.is_unspecified:
                continue
            addresses.append(LocalAddress(interface, address, status))
    return sorted(addresses, key=address_sort_key)


def local_ipv4_addresses_windows() -> List[LocalAddress]:
    try:
        proc = subprocess.run(
            ["ipconfig"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []

    addresses: List[LocalAddress] = []
    interface = "unknown"
    status = "active"
    for raw_line in proc.stdout.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if line == stripped and stripped.endswith(":"):
            interface = stripped[:-1]
            status = "active"
            continue
        if stripped.lower().startswith("media state") and "disconnected" in stripped.lower():
            status = "inactive"
            continue
        if "IPv4 Address" not in stripped and "Autoconfiguration IPv4 Address" not in stripped:
            continue

        _, _, value = stripped.partition(":")
        address = value.replace("(Preferred)", "").strip()
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if ip.is_loopback or ip.is_unspecified:
            continue
        addresses.append(LocalAddress(interface, address, status))

    return sorted(addresses, key=address_sort_key)


def address_label(address: LocalAddress) -> str:
    ip = ipaddress.ip_address(address.address)
    if ip.is_link_local:
        return f"direct-link/link-local candidate, {address.interface}"
    if address.interface.startswith("bridge"):
        return f"bridge candidate, {address.interface}"
    if ip.is_private:
        return f"private network/VPN, {address.interface}"
    return address.interface


def client_url_candidates(bind_host: str, gateway_port: int) -> List[tuple[str, str]]:
    if bind_host in {"127.0.0.1", "localhost"}:
        return [(f"http://127.0.0.1:{gateway_port}/v1", "local only")]
    if bind_host not in {"0.0.0.0", "::"}:
        return [(f"http://{bind_host}:{gateway_port}/v1", "configured bind address")]

    candidates = [
        (f"http://{address.address}:{gateway_port}/v1", address_label(address))
        for address in local_ipv4_addresses()
    ]
    return candidates or [(f"http://<this-mac-ip>:{gateway_port}/v1", "no non-loopback IPv4 detected")]


def has_direct_link_candidate(bind_host: str) -> bool:
    if bind_host not in {"0.0.0.0", "::"}:
        return False
    for address in local_ipv4_addresses():
        ip = ipaddress.ip_address(address.address)
        if ip.is_link_local or address.interface.startswith("bridge"):
            return True
    return False


def print_gateway_access(bind_host: str, gateway_port: int, roles: Dict[str, RoleRuntime]) -> None:
    print(f"gateway bind: {bind_host}:{gateway_port}")
    if bind_exposes_network(bind_host):
        print("WARNING: this gateway is exposed beyond localhost and does not enforce an API key.")
        print("Use it only on a trusted direct link or private network.")
    print("client base URL candidates:")
    for url, label in client_url_candidates(bind_host, gateway_port):
        print(f"  {url}  ({label})")
    if bind_host in {"0.0.0.0", "::"} and not has_direct_link_candidate(bind_host):
        print("WARNING: no link-local or bridge IPv4 address was detected; check the USB-C/Thunderbolt link if the other laptop cannot connect.")

    first_url = client_url_candidates(bind_host, gateway_port)[0][0]
    print("OpenAI-compatible environment:")
    print(f"  export OPENAI_BASE_URL={first_url}")
    print("  export OPENAI_API_KEY=local")
    print("model IDs:")
    print(f"  chat: {role_model_id(roles['chat'])}")
    print(f"  embeddings: {role_model_id(roles['embed'])}")
    print(f"  vision: {role_model_id(roles['vision'])}")


def choose_public_port(
    role: str,
    desired_port: int,
    fallback_port: Optional[int],
    host: str,
    *,
    dry_run: bool = False,
) -> tuple[int, bool]:
    if not port_is_open(host, desired_port):
        return desired_port, False
    if llama_ready(host, desired_port) or (dry_run and role != "chat"):
        return desired_port, True
    if role == "chat" and fallback_port is not None:
        if not port_is_open(host, fallback_port):
            return fallback_port, False
        if llama_ready(host, fallback_port) or dry_run:
            return fallback_port, True
    raise StartupError(f"public {role} port {host}:{desired_port} is already in use")


def build_runtimes(
    *,
    dry_run: bool = False,
    backend_host: Optional[str] = None,
    expose_public_ports: bool = True,
    role_proxy_bind_host: Optional[str] = None,
    role_proxy_bind_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, RoleRuntime]:
    chat_env = helper_env("chat", host=backend_host)
    embed_env = helper_env("embed", host=backend_host)
    vision_env = helper_env("vision", host=backend_host)
    host = chat_env["LLAMA_HOST"]
    log_dir = Path(chat_env["LOG_DIR"])
    bind_hosts = resolve_role_proxy_bind_hosts(
        global_override=role_proxy_bind_host,
        role_overrides=role_proxy_bind_overrides,
    )

    if expose_public_ports:
        chat_public, chat_external = choose_public_port(
            "chat",
            int(chat_env["PORT"]),
            parse_int_env("JUGGLE_CHAT_PUBLIC_FALLBACK_PORT", 18080),
            bind_hosts["chat"],
            dry_run=dry_run,
        )
        embed_public, embed_external = choose_public_port(
            "embed",
            int(embed_env["PORT"]),
            None,
            bind_hosts["embed"],
            dry_run=dry_run,
        )
        vision_public, vision_external = choose_public_port(
            "vision",
            int(vision_env["PORT"]),
            None,
            bind_hosts["vision"],
            dry_run=dry_run,
        )
    else:
        chat_public, chat_external = int(chat_env["PORT"]), False
        embed_public, embed_external = int(embed_env["PORT"]), False
        vision_public, vision_external = int(vision_env["PORT"]), False

    return {
        "chat": RoleRuntime(
            "chat",
            chat_public,
            parse_int_env("JUGGLE_CHAT_BACKEND_PORT", 18180),
            host,
            bind_hosts["chat"],
            log_dir / "chat-18180.log",
            chat_external,
            model_path=chat_env.get("MODEL", ""),
            alias=chat_env.get("ALIAS", ""),
        ),
        "embed": RoleRuntime(
            "embed",
            embed_public,
            parse_int_env("JUGGLE_EMBED_BACKEND_PORT", 18181),
            host,
            bind_hosts["embed"],
            log_dir / "embed-18181.log",
            embed_external,
            model_path=embed_env.get("MODEL", ""),
        ),
        "vision": RoleRuntime(
            "vision",
            vision_public,
            parse_int_env("JUGGLE_VISION_BACKEND_PORT", 18182),
            host,
            bind_hosts["vision"],
            log_dir / "vision-18182.log",
            vision_external,
            model_path=vision_env.get("MODEL", ""),
            alias=vision_env.get("ALIAS", ""),
        ),
    }


def print_dry_run(roles: Dict[str, RoleRuntime], *, auto_tune: bool) -> None:
    for role in ("chat", "embed", "vision"):
        runtime = roles[role]
        print(
            f"{role}: proxy_bind={runtime.bind_host}:{runtime.public_port} "
            f"backend={runtime.host}:{runtime.backend_port} external={runtime.external}"
        )
        if runtime.external:
            print("  using existing llama-server endpoint")
            continue
        argv = helper_argv(role, port=runtime.backend_port, host=runtime.host, auto_tune=False)
        print(f"  {shlex.join(argv)}")
    print(f"auto_tune_on_start={auto_tune}")


def print_gateway_dry_run(
    roles: Dict[str, RoleRuntime],
    *,
    bind_host: str,
    gateway_port: int,
    auto_tune: bool,
) -> None:
    print(f"gateway: public={bind_host}:{gateway_port}")
    print_gateway_access(bind_host, gateway_port, roles)
    for role in ("chat", "embed", "vision"):
        runtime = roles[role]
        print(f"{role}: backend={runtime.host}:{runtime.backend_port}")
        argv = helper_argv(role, port=runtime.backend_port, host=runtime.host, auto_tune=False)
        print(f"  {shlex.join(argv)}")
    print(f"auto_tune_on_start={auto_tune}")


def print_role_proxy_access(roles: Dict[str, RoleRuntime]) -> None:
    print("role proxy listeners:")
    for role in ("chat", "embed", "vision"):
        runtime = roles[role]
        if runtime.external:
            print(f"  {role}: existing endpoint http://{runtime.host}:{runtime.public_port}/v1")
            continue
        print(
            f"  {role}: bind {runtime.bind_host}:{runtime.public_port} "
            f"-> backend {runtime.host}:{runtime.backend_port}"
        )
        if bind_exposes_network(runtime.bind_host):
            print(f"  WARNING: {role} proxy is exposed beyond localhost and does not enforce an API key.")


def serve(roles: Dict[str, RoleRuntime], state: JugglerState) -> None:
    servers: List[ThreadingHTTPServer] = []
    for role in ("chat", "embed", "vision"):
        runtime = roles[role]
        if runtime.external:
            print(f"{role} already available at http://{runtime.host}:{runtime.public_port}/v1")
            continue
        server = ThreadingHTTPServer((runtime.bind_host, runtime.public_port), make_handler(role, state))
        servers.append(server)
        thread = threading.Thread(target=server.serve_forever, name=f"{role}-proxy", daemon=True)
        thread.start()
        print(
            f"{role} proxy ready at http://{runtime.bind_host}:{runtime.public_port}/v1 "
            f"-> backend {runtime.host}:{runtime.backend_port}"
        )

    print("model juggler is running; press Ctrl-C to stop")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        state.shutdown()


def serve_gateway(
    roles: Dict[str, RoleRuntime],
    state: JugglerState,
    *,
    bind_host: str,
    gateway_port: int,
) -> None:
    server = ThreadingHTTPServer((bind_host, gateway_port), make_gateway_handler(state))
    print(f"single gateway ready at http://{bind_host}:{gateway_port}/v1")
    print_gateway_access(bind_host, gateway_port, roles)
    print("gateway is running; press Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        state.shutdown()


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run llama-server model juggler")
    parser.add_argument("--dry-run", action="store_true", help="print proxy ports and rendered role commands")
    parser.add_argument("--check", action="store_true", help="validate configured files and ports, then exit")
    parser.add_argument("--no-auto-tune", action="store_true", help="do not run missing role auto-tune before first start")
    parser.add_argument("--gateway", action="store_true", help="serve one OpenAI-compatible gateway for all roles")
    parser.add_argument(
        "--role-proxy-bind",
        dest="role_proxy_bind",
        help="default listener bind address for role proxy mode; defaults to JUGGLE_ROLE_PROXY_BIND_HOST or 127.0.0.1",
    )
    parser.add_argument("--chat-proxy-bind", dest="chat_proxy_bind", help="listener bind address for the chat role proxy")
    parser.add_argument("--embed-proxy-bind", dest="embed_proxy_bind", help="listener bind address for the embedding role proxy")
    parser.add_argument("--vision-proxy-bind", dest="vision_proxy_bind", help="listener bind address for the vision role proxy")
    parser.add_argument(
        "--gateway-bind",
        "--bind",
        dest="gateway_bind",
        default=os.environ.get("SERVICE_GATEWAY_BIND", GATEWAY_DEFAULT_BIND),
        help="bind address for --gateway mode",
    )
    parser.add_argument(
        "--gateway-port",
        "--port",
        dest="gateway_port",
        type=int,
        default=parse_int_env("SERVICE_GATEWAY_PORT", GATEWAY_DEFAULT_PORT),
        help="port for --gateway mode",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    roles = build_runtimes(
        dry_run=args.dry_run or args.check,
        backend_host="127.0.0.1" if args.gateway else None,
        expose_public_ports=not args.gateway,
        role_proxy_bind_host=args.role_proxy_bind,
        role_proxy_bind_overrides={
            role: value
            for role, value in {
                "chat": args.chat_proxy_bind,
                "embed": args.embed_proxy_bind,
                "vision": args.vision_proxy_bind,
            }.items()
            if value is not None
        },
    )
    auto_tune = not args.no_auto_tune

    if args.dry_run:
        if args.gateway:
            print_gateway_dry_run(
                roles,
                bind_host=args.gateway_bind,
                gateway_port=args.gateway_port,
                auto_tune=auto_tune,
            )
        else:
            print_dry_run(roles, auto_tune=auto_tune)
        return 0

    state = JugglerState(
        roles,
        auto_tune=auto_tune,
        switch_timeout=parse_int_env("JUGGLE_SWITCH_TIMEOUT_SECONDS", 600),
        startup_timeout=parse_int_env("JUGGLE_STARTUP_TIMEOUT_SECONDS", 900),
        request_timeout=parse_int_env("JUGGLE_REQUEST_TIMEOUT_SECONDS", 3600),
    )
    atexit.register(state.shutdown)

    if args.check:
        if args.gateway:
            print_gateway_access(args.gateway_bind, args.gateway_port, roles)
        else:
            print_role_proxy_access(roles)
        state.validate_files()
        if args.gateway:
            check_gateway_port(args.gateway_bind, args.gateway_port)
            print("service gateway configuration check passed")
        else:
            print("juggler configuration check passed")
        return 0

    state.validate_files()
    if args.gateway:
        check_gateway_port(args.gateway_bind, args.gateway_port)
        state.start_embed_baseline()
        serve_gateway(roles, state, bind_host=args.gateway_bind, gateway_port=args.gateway_port)
        return 0

    state.start_embed_baseline()
    serve(roles, state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StartupError as exc:
        print(f"model juggler error: {exc}", file=sys.stderr)
        raise SystemExit(1)
