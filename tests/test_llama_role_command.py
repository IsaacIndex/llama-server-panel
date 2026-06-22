from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scripts import llama_role_command


class LlamaRoleCommandTest(unittest.TestCase):
    def test_exec_mode_uses_role_log_path_and_returns_runner_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            log_path = panel_dir / "logs" / "chat.log"
            role_config = {"LOG_DIR": str(log_path.parent)}

            with (
                patch.object(llama_role_command, "repo_dir", return_value=panel_dir),
                patch.object(llama_role_command, "load_config", return_value=role_config),
                patch.object(llama_role_command, "role_environment", return_value={"LLAMA_HOST": "127.0.0.1", "PORT": "8080"}),
                patch.object(llama_role_command, "validate_role_files"),
                patch.object(llama_role_command, "port_in_use", return_value=False),
                patch.object(llama_role_command, "build_role_argv", return_value=["llama-server", "--model", "chat.gguf"]),
                patch.object(llama_role_command, "role_server_log_path", return_value=log_path),
                patch.object(llama_role_command, "run_role_argv_with_log", return_value=7) as run_role,
            ):
                result = llama_role_command.main(["exec", "chat", "--auto-tune"])

            self.assertEqual(result, 7)
            run_role.assert_called_once_with(
                "chat",
                ["llama-server", "--model", "chat.gguf"],
                panel_dir=panel_dir,
                log_path=log_path,
            )


if __name__ == "__main__":
    unittest.main()
