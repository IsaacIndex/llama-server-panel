from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
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

    def test_build_runtimes_loads_role_proxy_bind_from_gui_override_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            (panel_dir / "env.local.gui.json").write_text('{"JUGGLE_EMBED_PROXY_BIND_HOST": "0.0.0.0"}\n', encoding="utf-8")

            with (
                patch("model_juggler.REPO_DIR", panel_dir),
                patch("model_juggler.helper_env", side_effect=lambda role, **_kwargs: self._role_env(tmp, role)),
                patch("model_juggler.port_is_open", return_value=False),
                patch.dict(os.environ, {}, clear=True),
            ):
                roles = build_runtimes(dry_run=True)

        self.assertEqual(roles["chat"].bind_host, "127.0.0.1")
        self.assertEqual(roles["embed"].bind_host, "0.0.0.0")

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


class EmbedBatchEscalationTest(unittest.TestCase):
    def test_detects_too_large_error(self) -> None:
        body = b'{"error":{"message":"input (4391 tokens) is too large to process. increase the physical batch"}}'
        self.assertTrue(model_juggler._is_embed_batch_error(500, body))
        self.assertTrue(model_juggler._is_embed_batch_error(400, body))

    def test_ignores_success_and_unrelated_errors(self) -> None:
        self.assertFalse(model_juggler._is_embed_batch_error(200, b'{"data":[]}'))
        self.assertFalse(model_juggler._is_embed_batch_error(500, b'{"error":"boom"}'))

    def test_parses_required_tokens(self) -> None:
        body = b"input (4391 tokens) is too large to process. increase the physical batch"
        self.assertEqual(model_juggler._required_tokens_from_error(body), 4391)
        self.assertIsNone(model_juggler._required_tokens_from_error(b"too large to process"))

    def _fake_conn(self, status: int, body: bytes):
        read_sizes: list[object] = []
        chunks = [body, b""]

        def read(size: object = None) -> bytes:
            read_sizes.append(size)
            return chunks.pop(0) if chunks else b""

        response = SimpleNamespace(
            status=status,
            reason="OK" if status < 400 else "Error",
            getheaders=lambda: [("Content-Type", "application/json")],
            read=read,
            read_sizes=read_sizes,
        )
        conn = Mock()
        conn.getresponse.return_value = response
        conn.response = response
        return conn

    def _fake_handler(self):
        handler = Mock()
        handler.headers = {"Content-Type": "application/json"}
        handler.command = "POST"
        handler.path = "/v1/embeddings"
        return handler

    def test_escalates_then_reverts_on_too_large(self) -> None:
        too_large = b'{"error":{"message":"input (4391 tokens) is too large to process. increase the physical batch"}}'
        ok = b'{"data":[{"embedding":[0.1,0.2]}]}'

        runtime = RoleRuntime("embed", 8081, 18181, "127.0.0.1", "127.0.0.1", Path("/tmp/embed.log"))
        state = Mock()
        state.roles = {"embed": runtime}
        state.request_timeout = 5
        state.embed_configured_batch_size.return_value = 4096
        restarts: list = []
        events: list[str] = []

        def restart(value):
            restarts.append(value)
            events.append(f"restart:{value}")
            return 18181

        state.begin_embed_backend_request.side_effect = lambda: events.append("begin")
        state.finish_embed_backend_request.side_effect = lambda: events.append("finish")
        state.restart_embed_with_batch.side_effect = restart

        conns = [self._fake_conn(500, too_large), self._fake_conn(200, ok)]
        target_batch = 4391 + model_juggler.EMBED_BATCH_ESCALATION_MARGIN

        with patch("model_juggler.http.client.HTTPConnection", side_effect=conns):
            handler = self._fake_handler()
            model_juggler._proxy_embed_with_escalation(
                handler, state, "embed", runtime, 18181, "POST", "/v1/embeddings", b"{}"
            )

        # First escalation grows past the offending token count, final revert is None.
        self.assertEqual(restarts[0], target_batch)
        self.assertIsNone(restarts[-1])
        self.assertEqual(events, ["begin", "finish", f"restart:{target_batch}", "begin", "finish", "restart:None"])
        # The successful (200) body must be relayed to the client.
        handler.send_response.assert_called_with(200, "OK")

    def test_passes_through_success_without_restart(self) -> None:
        ok = b'{"data":[{"embedding":[0.1]}]}'
        runtime = RoleRuntime("embed", 8081, 18181, "127.0.0.1", "127.0.0.1", Path("/tmp/embed.log"))
        state = Mock()
        state.roles = {"embed": runtime}
        state.request_timeout = 5
        conn = self._fake_conn(200, ok)

        with patch("model_juggler.http.client.HTTPConnection", side_effect=[conn]):
            handler = self._fake_handler()
            model_juggler._proxy_embed_with_escalation(
                handler, state, "embed", runtime, 18181, "POST", "/v1/embeddings", b"{}"
            )

        state.begin_embed_backend_request.assert_called_once()
        state.finish_embed_backend_request.assert_called_once()
        state.restart_embed_with_batch.assert_not_called()
        handler.send_response.assert_called_with(200, "OK")
        handler.wfile.write.assert_called_with(ok)
        self.assertEqual(conn.response.read_sizes, [65536, 65536])

    def test_restart_waits_for_active_embed_request(self) -> None:
        runtime = RoleRuntime("embed", 8081, 18181, "127.0.0.1", "127.0.0.1", Path("/tmp/embed.log"))
        state = model_juggler.JugglerState(
            {"embed": runtime},
            auto_tune=False,
            switch_timeout=1.0,
            startup_timeout=1.0,
            request_timeout=1.0,
        )
        state.stop_process = Mock()  # type: ignore[method-assign]
        state.ensure_process = Mock()  # type: ignore[method-assign]
        state.begin_embed_backend_request()

        started = threading.Event()
        finished = threading.Event()

        def restart() -> None:
            started.set()
            state.restart_embed_with_batch(8192)
            finished.set()

        thread = threading.Thread(target=restart)
        thread.start()
        try:
            self.assertTrue(started.wait(timeout=1.0))
            self.assertFalse(finished.wait(timeout=0.05))
            state.finish_embed_backend_request()
            self.assertTrue(finished.wait(timeout=1.0))
        finally:
            if state.active_embed_requests:
                state.finish_embed_backend_request()
            thread.join(timeout=1.0)

        state.stop_process.assert_called_once_with("embed")
        state.ensure_process.assert_called_once_with("embed")
        self.assertEqual(runtime.batch_size_override, 8192)


if __name__ == "__main__":
    unittest.main()
