from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import llama_runtime
from scripts.llama_runtime import (
    PanelError,
    default_config,
    ensure_llama_server_binary,
    filter_unsupported_llama_args,
    launch_diagnostics,
    load_config,
    popen_session_kwargs,
    prepare_llama_server_argv,
    raise_if_process_exited,
    role_server_log_path,
    run_role_argv_with_log,
)


class LlamaRuntimePublicDefaultsTest(unittest.TestCase):
    class _StdoutCapture:
        def __init__(self) -> None:
            self.buffer = io.BytesIO()

        def flush(self) -> None:
            self.buffer.flush()

    def test_defaults_are_repo_local_and_path_based(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)

            config = default_config(panel_dir)

            self.assertEqual(config["LLAMA_SERVER_BIN"], "llama-server")
            self.assertEqual(config["MODEL_DIR"], str((panel_dir / "models").resolve()))
            self.assertEqual(config["LOG_DIR"], str((panel_dir / "logs").resolve()))
            self.assertEqual(config["VISION_MMPROJ"], "mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf")
            self.assertEqual(config["JUGGLE_ROLE_PROXY_BIND_HOST"], "127.0.0.1")
            self.assertEqual(config["JUGGLE_EMBED_PROXY_BIND_HOST"], "")

    def test_relative_default_model_paths_resolve_under_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)

            config = load_config(panel_dir, apply_tune=False)

            self.assertEqual(config["MODEL_DIR"], str((panel_dir / "models").resolve()))
            self.assertEqual(
                config["VISION_MMPROJ"],
                str((panel_dir / "models" / "mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf").resolve()),
            )

    def test_load_config_prefers_gui_role_proxy_bind_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            (panel_dir / "env.local.env").write_text("JUGGLE_EMBED_PROXY_BIND_HOST=127.0.0.1\n", encoding="utf-8")
            (panel_dir / "env.local.gui.json").write_text('{"JUGGLE_EMBED_PROXY_BIND_HOST": "0.0.0.0"}\n', encoding="utf-8")

            config = load_config(panel_dir, apply_tune=False)

        self.assertEqual(config["JUGGLE_EMBED_PROXY_BIND_HOST"], "0.0.0.0")

    def test_repo_dir_uses_executable_parent_when_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "installed"
            meipass_dir = root / "Temp" / "_MEI12345"
            install_dir.mkdir()
            meipass_dir.mkdir(parents=True)

            with (
                patch.object(llama_runtime.sys, "frozen", True, create=True),
                patch.object(llama_runtime.sys, "executable", str(install_dir / "llama-server-panel.exe")),
                patch.object(llama_runtime.sys, "_MEIPASS", str(meipass_dir), create=True),
                patch.dict(os.environ, {}, clear=True),
            ):
                self.assertEqual(llama_runtime.repo_dir(), install_dir.resolve())

    def test_load_config_uses_frozen_executable_parent_not_meipass_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "installed"
            meipass_dir = root / "Temp" / "_MEI12345"
            install_dir.mkdir()
            meipass_dir.mkdir(parents=True)
            (install_dir / "env.local.gui.json").write_text('{"JUGGLE_EMBED_PROXY_BIND_HOST": "0.0.0.0"}\n', encoding="utf-8")
            (meipass_dir / "env.local.gui.json").write_text('{"JUGGLE_EMBED_PROXY_BIND_HOST": "127.0.0.1"}\n', encoding="utf-8")

            with (
                patch.object(llama_runtime.sys, "frozen", True, create=True),
                patch.object(llama_runtime.sys, "executable", str(install_dir / "llama-server-panel.exe")),
                patch.object(llama_runtime.sys, "_MEIPASS", str(meipass_dir), create=True),
                patch.dict(os.environ, {}, clear=True),
            ):
                config = load_config(apply_tune=False)

        self.assertEqual(config["LLAMA_SERVER_PANEL_DIR"], str(install_dir.resolve()))
        self.assertEqual(config["JUGGLE_EMBED_PROXY_BIND_HOST"], "0.0.0.0")

    def test_popen_session_kwargs_windows_hides_child_console(self) -> None:
        with (
            patch.object(llama_runtime.os, "name", "nt"),
            patch.object(llama_runtime.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, create=True),
            patch.object(llama_runtime.subprocess, "CREATE_NO_WINDOW", 0x8000000, create=True),
        ):
            kwargs = popen_session_kwargs()

        self.assertEqual(kwargs["creationflags"], 0x8000200)

    def test_popen_session_kwargs_posix_starts_new_session(self) -> None:
        with patch.object(llama_runtime.os, "name", "posix"):
            self.assertEqual(popen_session_kwargs(), {"start_new_session": True})

    def test_launch_diagnostics_includes_command_context(self) -> None:
        cwd = Path("/tmp/panel")
        message = launch_diagnostics("Chat", ["llama-server", "--model", "chat.gguf"], cwd=cwd, pid=123)

        self.assertIn("[panel]", message)
        self.assertIn("started Chat", message)
        self.assertIn(f"cwd: {cwd}", message)
        self.assertIn("command:", message)
        self.assertIn("llama-server", message)
        self.assertIn("pid: 123", message)

    def test_role_server_log_path_is_per_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = {"LOG_DIR": tmp}

            self.assertEqual(role_server_log_path(config, "chat"), Path(tmp) / "chat.log")
            self.assertEqual(role_server_log_path(config, "embed"), Path(tmp) / "embed.log")
            self.assertEqual(role_server_log_path(config, "vision"), Path(tmp) / "vision.log")
            with self.assertRaisesRegex(PanelError, "Unknown role"):
                role_server_log_path(config, "rerank")

    def test_run_role_argv_with_log_captures_output_and_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            log_path = panel_dir / "logs" / "chat.log"

            returncode = run_role_argv_with_log(
                "chat",
                [sys.executable, "-c", "print('server output')"],
                panel_dir=panel_dir,
                log_path=log_path,
            )

            self.assertEqual(returncode, 0)
            text = log_path.read_text(encoding="utf-8")
            self.assertIn("launching chat llama-server", text)
            self.assertIn("started chat llama-server", text)
            self.assertIn("server output", text)

    def test_run_role_argv_with_log_streams_inline_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            log_path = panel_dir / "logs" / "chat.log"
            stdout_capture = self._StdoutCapture()

            with (
                patch.dict(os.environ, {"PANEL_INLINE_LOGS": "1"}, clear=False),
                patch.object(llama_runtime.sys, "stdout", stdout_capture),
            ):
                returncode = run_role_argv_with_log(
                    "chat",
                    [sys.executable, "-c", "print('inline output')"],
                    panel_dir=panel_dir,
                    log_path=log_path,
                )

            self.assertEqual(returncode, 0)
            text = log_path.read_text(encoding="utf-8")
            self.assertIn("inline output", text)
            self.assertIn("inline output", stdout_capture.buffer.getvalue().decode("utf-8"))

    def test_filter_unsupported_llama_args_removes_optional_flags_missing_from_help(self) -> None:
        argv = [
            "llama-server",
            "--model",
            "chat.gguf",
            "--n-cpu-moe",
            "40",
            "--cache-type-k",
            "q8_0",
            "--reasoning",
            "on",
            "--temp",
            "0.6",
            "--jinja",
        ]

        filtered, removed = filter_unsupported_llama_args(argv, "--model\n--cache-type-k\n--temp\n")

        self.assertEqual(removed, ["--n-cpu-moe", "--reasoning", "--jinja"])
        self.assertEqual(filtered, ["llama-server", "--model", "chat.gguf", "--cache-type-k", "q8_0", "--temp", "0.6"])

    def test_prepare_llama_server_argv_handles_windows_exe_path(self) -> None:
        argv = [r"C:\Tools\llama-server.EXE", "--model", "chat.gguf", "--reasoning", "on"]

        with patch.object(llama_runtime, "llama_server_help_text", return_value="--model\n"):
            filtered, removed = prepare_llama_server_argv(argv)

        self.assertEqual(removed, ["--reasoning"])
        self.assertEqual(filtered, [r"C:\Tools\llama-server.EXE", "--model", "chat.gguf"])

    def test_prepare_llama_server_argv_can_be_disabled(self) -> None:
        argv = ["llama-server", "--model", "chat.gguf", "--reasoning", "on"]

        with patch.object(llama_runtime, "llama_server_help_text") as help_text:
            filtered, removed = prepare_llama_server_argv(argv, env={"LLAMA_SERVER_COMPAT_FILTER": "0"})

        help_text.assert_not_called()
        self.assertEqual(removed, [])
        self.assertEqual(filtered, argv)

    def test_raise_if_process_exited_reports_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "chat.log"
            with log_path.open("wb") as log_fh:
                proc = llama_runtime.subprocess.Popen(
                    [sys.executable, "-c", "import sys; print('unknown argument: --reasoning'); sys.exit(9)"],
                    stdout=log_fh,
                    stderr=llama_runtime.subprocess.STDOUT,
                )

            with self.assertRaises(PanelError) as ctx:
                raise_if_process_exited(proc, "Chat", log_path, grace_seconds=5)

        message = str(ctx.exception)
        self.assertIn("Chat exited during startup with code 9", message)
        self.assertIn("unknown argument: --reasoning", message)

    def test_ensure_llama_server_binary_rejects_panel_launcher_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            launcher = Path(tmp) / "start-chat.cmd"
            launcher.write_text("@echo off\n", encoding="utf-8")

            with self.assertRaisesRegex(PanelError, "must point to the llama-server executable"):
                ensure_llama_server_binary({"LLAMA_SERVER_BIN": str(launcher)})


if __name__ == "__main__":
    unittest.main()
