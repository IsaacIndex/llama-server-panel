from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.llama_runtime import default_config, load_config


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


if __name__ == "__main__":
    unittest.main()
