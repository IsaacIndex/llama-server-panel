from __future__ import annotations

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
    launch_diagnostics,
    load_config,
    popen_session_kwargs,
    role_server_log_path,
    run_role_argv_with_log,
)


class LlamaRuntimePublicDefaultsTest(unittest.TestCase):
    def test_defaults_are_repo_local_and_path_based(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)

            config = default_config(panel_dir)

            self.assertEqual(config["LLAMA_SERVER_BIN"], "llama-server")
            self.assertEqual(config["MODEL_DIR"], str((panel_dir / "models").resolve()))
            self.assertEqual(config["LOG_DIR"], str((panel_dir / "logs").resolve()))
            self.assertEqual(config["VISION_MMPROJ"], "mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf")

    def test_relative_default_model_paths_resolve_under_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)

            config = load_config(panel_dir, apply_tune=False)

            self.assertEqual(config["MODEL_DIR"], str((panel_dir / "models").resolve()))
            self.assertEqual(
                config["VISION_MMPROJ"],
                str((panel_dir / "models" / "mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf").resolve()),
            )

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

    def test_ensure_llama_server_binary_rejects_panel_launcher_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            launcher = Path(tmp) / "start-chat.cmd"
            launcher.write_text("@echo off\n", encoding="utf-8")

            with self.assertRaisesRegex(PanelError, "must point to the llama-server executable"):
                ensure_llama_server_binary({"LLAMA_SERVER_BIN": str(launcher)})


if __name__ == "__main__":
    unittest.main()
