#!/usr/bin/env python3
"""Cross-platform replacement for scripts/llama_role_command.sh."""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from llama_runtime import (
    PanelError,
    build_role_argv,
    load_config,
    port_in_use,
    repo_dir,
    role_server_log_path,
    role_environment,
    run_role_argv_with_log,
    validate_role_files,
    write_env0_records,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render or run a role-specific llama-server command")
    parser.add_argument("mode", choices=("exec", "argv0", "env0", "check"))
    parser.add_argument("role", choices=("chat", "embed", "vision"))
    parser.add_argument("--port", type=int, help="override the role port")
    parser.add_argument("--host", help="override LLAMA_HOST")
    parser.add_argument("--auto-tune", action="store_true", help="create a tune file before launch when one is missing")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    panel_dir = repo_dir()
    role_config = load_config(panel_dir, role=args.role)

    if args.mode == "check":
        validate_role_files(args.role, role_config)
        return 0

    if args.mode == "env0":
        records = role_environment(
            args.role,
            panel_dir=panel_dir,
            port_override=args.port,
            host_override=args.host,
        )
        sys.stdout.buffer.write(write_env0_records(records))
        return 0

    if args.mode == "argv0":
        argv0 = build_role_argv(
            args.role,
            panel_dir=panel_dir,
            port_override=args.port,
            host_override=args.host,
            auto_tune=args.auto_tune,
        )
        sys.stdout.buffer.write(b"\0".join(part.encode("utf-8") for part in argv0))
        if argv0:
            sys.stdout.buffer.write(b"\0")
        return 0

    env_config = role_environment(
        args.role,
        panel_dir=panel_dir,
        port_override=args.port,
        host_override=args.host,
    )
    validate_role_files(args.role, role_config)
    host = env_config["LLAMA_HOST"]
    port = int(env_config["PORT"])
    if port_in_use(host, port):
        raise PanelError(
            f"Port {host}:{port} is already in use. Stop the existing process or change the role port."
        )

    role_argv = build_role_argv(
        args.role,
        panel_dir=panel_dir,
        port_override=args.port,
        host_override=args.host,
        auto_tune=args.auto_tune,
    )
    role_config = load_config(panel_dir, role=args.role)
    log_path = role_server_log_path(role_config, args.role)
    print(f"[panel] writing {args.role} llama-server log to {log_path}", file=sys.stderr)
    return run_role_argv_with_log(args.role, role_argv, panel_dir=panel_dir, log_path=log_path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PanelError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
