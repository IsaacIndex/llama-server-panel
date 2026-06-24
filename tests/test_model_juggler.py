from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import model_juggler
from model_juggler import RoleRuntime, StartupError, build_runtimes, print_dry_run, serve, wait_ready


class ModelJugglerStartupTest(unittest.TestCase):
    def test_wait_ready_fails_fast_when_process_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "chat.log"
            log_path.write_text("unknown argument: --reasoning\n", encoding="utf-8")
            proc = SimpleNamespace(returncode=9, poll=lambda: 9)

            with (
                patch("model_juggler.llama_ready", return_value=False),
                patch("model_juggler.time.sleep") as sleep,
                self.assertRaises(StartupError) as ctx,
            ):
                wait_ready("127.0.0.1", 18180, 60, proc=proc, log_path=log_path)

        sleep.assert_not_called()
        message = str(ctx.exception)
        self.assertIn("exited during startup with code 9", message)
        self.assertIn("unknown argument: --reasoning", message)

    def _role_env(self, tmp: str, role: str, *, host: Optional[str] = None) -> dict[str, str]:
        ports = {"chat": "8080", "embed": "8081", "vision": "8082"}
        result = {
            "LLAMA_HOST": host or "127.0.0.1",
            "PORT": ports[role],
            "LOG_DIR": tmp,
            "MODEL": f"{role}.gguf",
        }
        if role in {"chat", "vision"}:
            result["ALIAS"] = f"{role}-model"
        return result

    def test_build_runtimes_uses_default_local_role_proxy_bind_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "JUGGLE_ROLE_PROXY_BIND_HOST": "127.0.0.1",
                "JUGGLE_CHAT_PROXY_BIND_HOST": "",
                "JUGGLE_EMBED_PROXY_BIND_HOST": "",
                "JUGGLE_VISION_PROXY_BIND_HOST": "",
            }

            with (
                patch("model_juggler.helper_env", side_effect=lambda role, **_kwargs: self._role_env(tmp, role)),
                patch("model_juggler.load_config", return_value=config),
                patch("model_juggler.port_is_open", return_value=False),
                patch.dict(os.environ, {}, clear=True),
            ):
                roles = build_runtimes(dry_run=True)

        self.assertEqual(roles["chat"].bind_host, "127.0.0.1")
        self.assertEqual(roles["embed"].bind_host, "127.0.0.1")
        self.assertEqual(roles["vision"].bind_host, "127.0.0.1")
        self.assertEqual(roles["embed"].host, "127.0.0.1")

    def test_build_runtimes_uses_configured_per_role_proxy_bind_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "JUGGLE_ROLE_PROXY_BIND_HOST": "127.0.0.1",
                "JUGGLE_CHAT_PROXY_BIND_HOST": "",
                "JUGGLE_EMBED_PROXY_BIND_HOST": "0.0.0.0",
                "JUGGLE_VISION_PROXY_BIND_HOST": "",
            }

            with (
                patch("model_juggler.helper_env", side_effect=lambda role, **_kwargs: self._role_env(tmp, role)),
                patch("model_juggler.load_config", return_value=config),
                patch("model_juggler.port_is_open", return_value=False),
                patch.dict(os.environ, {}, clear=True),
            ):
                roles = build_runtimes(dry_run=True)

        self.assertEqual(roles["embed"].bind_host, "0.0.0.0")
        self.assertEqual(roles["embed"].host, "127.0.0.1")

    def test_serve_role_proxy_binds_configured_listener_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            roles = {
                "chat": RoleRuntime("chat", 8080, 18180, "127.0.0.1", "127.0.0.1", log_dir / "chat.log"),
                "embed": RoleRuntime("embed", 8081, 18181, "127.0.0.1", "0.0.0.0", log_dir / "embed.log"),
                "vision": RoleRuntime("vision", 8082, 18182, "127.0.0.1", "127.0.0.1", log_dir / "vision.log"),
            }
            state = SimpleNamespace(shutdown=Mock())
            server_instances: list[object] = []
            server_addresses: list[tuple[str, int]] = []

            class FakeServer:
                def __init__(self, address: tuple[str, int], _handler: object) -> None:
                    self.address = address
                    self.shutdown = Mock()
                    self.server_close = Mock()
                    server_addresses.append(address)
                    server_instances.append(self)

                def serve_forever(self) -> None:
                    return None

            with (
                patch("model_juggler.ThreadingHTTPServer", FakeServer),
                patch("model_juggler.threading.Thread.start", return_value=None),
                patch("model_juggler.time.sleep", side_effect=KeyboardInterrupt),
                redirect_stdout(io.StringIO()),
            ):
                serve(roles, state)  # type: ignore[arg-type]

        self.assertIn(("0.0.0.0", 8081), server_addresses)
        for server in server_instances:
            server.shutdown.assert_called_once()
            server.server_close.assert_called_once()
        state.shutdown.assert_called_once()

    def test_dry_run_prints_proxy_bind_and_backend_host_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            roles = {
                "chat": RoleRuntime("chat", 8080, 18180, "127.0.0.1", "127.0.0.1", log_dir / "chat.log"),
                "embed": RoleRuntime("embed", 8081, 18181, "127.0.0.1", "0.0.0.0", log_dir / "embed.log"),
                "vision": RoleRuntime("vision", 8082, 18182, "127.0.0.1", "127.0.0.1", log_dir / "vision.log"),
            }
            output = io.StringIO()

            with (
                patch("model_juggler.helper_argv", return_value=["llama-server", "--host", "127.0.0.1"]),
                redirect_stdout(output),
            ):
                print_dry_run(roles, auto_tune=False)

        text = output.getvalue()
        self.assertIn("embed: proxy_bind=0.0.0.0:8081 backend=127.0.0.1:18181", text)

    def test_main_dry_run_forwards_role_proxy_bind_overrides(self) -> None:
        roles: dict[str, RoleRuntime] = {}

        with (
            patch("model_juggler.build_runtimes", return_value=roles) as build,
            patch("model_juggler.print_dry_run"),
        ):
            result = model_juggler.main(["--dry-run", "--embed-proxy-bind", "0.0.0.0"])

        self.assertEqual(result, 0)
        self.assertEqual(build.call_args.kwargs["role_proxy_bind_overrides"], {"embed": "0.0.0.0"})


if __name__ == "__main__":
    unittest.main()
