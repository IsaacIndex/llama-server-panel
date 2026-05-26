#!/usr/bin/env python3
"""Proxy/supervisor for juggling chat and vision llama-server processes."""

from __future__ import annotations

import argparse
import atexit
import http.client
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Optional


HEAVY_ROLES = {"chat", "vision"}
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
HELPER = REPO_DIR / "scripts" / "llama_role_command.sh"


def parse_env0(payload: bytes) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for record in payload.split(b"\0"):
        if not record:
            continue
        text = record.decode("utf-8")
        key, _, value = text.partition("=")
        result[key] = value
    return result


def run_helper(mode: str, role: str, *, port: Optional[int] = None, auto_tune: bool = False) -> bytes:
    cmd: List[str] = ["/bin/zsh", str(HELPER), mode, role]
    if port is not None:
        cmd.extend(["--port", str(port)])
    if auto_tune:
        cmd.append("--auto-tune")

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise StartupError(stderr or f"{mode} {role} failed with exit code {proc.returncode}")
    if proc.stderr:
        sys.stderr.write(proc.stderr.decode("utf-8", errors="replace"))
    return proc.stdout


def helper_env(role: str, *, port: Optional[int] = None) -> Dict[str, str]:
    return parse_env0(run_helper("env0", role, port=port))


def helper_argv(role: str, *, port: int, auto_tune: bool) -> List[str]:
    payload = run_helper("argv0", role, port=port, auto_tune=auto_tune)
    return [part.decode("utf-8") for part in payload.split(b"\0") if part]


def helper_check(role: str) -> None:
    run_helper("check", role)


def port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        if sock.connect_ex((host, port)) == 0:
            return True

    lsof = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return lsof.returncode == 0


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


def wait_ready(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
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
    log_path: Path
    external: bool = False
    process: Optional[subprocess.Popen[bytes]] = None


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
        argv = helper_argv(role, port=runtime.backend_port, auto_tune=self.auto_tune)
        log_fh = open(runtime.log_path, "ab", buffering=0)
        try:
            runtime.process = subprocess.Popen(
                argv,
                cwd=str(REPO_DIR),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_fh.close()

        wait_ready(runtime.host, runtime.backend_port, self.startup_timeout)

    def stop_process(self, role: str) -> None:
        runtime = self.roles[role]
        proc = runtime.process
        if not proc or proc.poll() is not None:
            runtime.process = None
            return

        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            runtime.process = None
            return
        except OSError:
            proc.terminate()

        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait(timeout=10)
        finally:
            runtime.process = None

    def shutdown(self) -> None:
        for role in ("chat", "vision", "embed"):
            runtime = self.roles.get(role)
            if runtime and not runtime.external:
                self.stop_process(role)


def proxy_request(handler: BaseHTTPRequestHandler, state: JugglerState, role: str) -> None:
    runtime = state.roles[role]
    backend_port: Optional[int] = None
    prepared = False

    try:
        backend_port = state.prepare_request(role)
        prepared = True
        length = int(handler.headers.get("Content-Length", "0") or "0")
        body = handler.rfile.read(length) if length else None

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


def write_error(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    payload = json.dumps(
        {"error": {"message": message, "type": "model_juggler"}},
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(payload)


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


def build_runtimes(*, dry_run: bool = False) -> Dict[str, RoleRuntime]:
    chat_env = helper_env("chat")
    embed_env = helper_env("embed")
    vision_env = helper_env("vision")
    host = chat_env["LLAMA_HOST"]
    log_dir = Path(chat_env["LOG_DIR"])

    chat_public, chat_external = choose_public_port(
        "chat",
        int(chat_env["PORT"]),
        parse_int_env("JUGGLE_CHAT_PUBLIC_FALLBACK_PORT", 18080),
        host,
        dry_run=dry_run,
    )
    embed_public, embed_external = choose_public_port("embed", int(embed_env["PORT"]), None, host, dry_run=dry_run)
    vision_public, vision_external = choose_public_port("vision", int(vision_env["PORT"]), None, host, dry_run=dry_run)

    return {
        "chat": RoleRuntime(
            "chat",
            chat_public,
            parse_int_env("JUGGLE_CHAT_BACKEND_PORT", 18180),
            host,
            log_dir / "chat-18180.log",
            chat_external,
        ),
        "embed": RoleRuntime(
            "embed",
            embed_public,
            parse_int_env("JUGGLE_EMBED_BACKEND_PORT", 18181),
            host,
            log_dir / "embed-18181.log",
            embed_external,
        ),
        "vision": RoleRuntime(
            "vision",
            vision_public,
            parse_int_env("JUGGLE_VISION_BACKEND_PORT", 18182),
            host,
            log_dir / "vision-18182.log",
            vision_external,
        ),
    }


def print_dry_run(roles: Dict[str, RoleRuntime], *, auto_tune: bool) -> None:
    for role in ("chat", "embed", "vision"):
        runtime = roles[role]
        print(f"{role}: public={runtime.host}:{runtime.public_port} backend={runtime.host}:{runtime.backend_port} external={runtime.external}")
        if runtime.external:
            print("  using existing llama-server endpoint")
            continue
        argv = helper_argv(role, port=runtime.backend_port, auto_tune=False)
        print(f"  {shlex.join(argv)}")
    print(f"auto_tune_on_start={auto_tune}")


def serve(roles: Dict[str, RoleRuntime], state: JugglerState) -> None:
    servers: List[ThreadingHTTPServer] = []
    for role in ("chat", "embed", "vision"):
        runtime = roles[role]
        if runtime.external:
            print(f"{role} already available at http://{runtime.host}:{runtime.public_port}/v1")
            continue
        server = ThreadingHTTPServer((runtime.host, runtime.public_port), make_handler(role, state))
        servers.append(server)
        thread = threading.Thread(target=server.serve_forever, name=f"{role}-proxy", daemon=True)
        thread.start()
        print(f"{role} proxy ready at http://{runtime.host}:{runtime.public_port}/v1 -> backend :{runtime.backend_port}")

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


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run llama-server model juggler")
    parser.add_argument("--dry-run", action="store_true", help="print proxy ports and rendered role commands")
    parser.add_argument("--check", action="store_true", help="validate configured files and ports, then exit")
    parser.add_argument("--no-auto-tune", action="store_true", help="do not run missing role auto-tune before first start")
    args = parser.parse_args(list(argv) if argv is not None else None)

    roles = build_runtimes(dry_run=args.dry_run or args.check)
    auto_tune = not args.no_auto_tune

    if args.dry_run:
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
        state.validate_files()
        print("juggler configuration check passed")
        return 0

    state.validate_files()
    state.start_embed_baseline()
    serve(roles, state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StartupError as exc:
        print(f"model juggler error: {exc}", file=sys.stderr)
        raise SystemExit(1)
