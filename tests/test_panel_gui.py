from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from llama_runtime import PanelError, load_config
from panel_gui import (
    build_gui_overrides,
    compact_path_value,
    default_assign_key_for_role,
    gui_override_path,
    import_model_file,
    model_dir_from_value,
    model_config_value,
    save_gui_overrides,
)


class PanelGuiHelpersTest(unittest.TestCase):
    def test_empty_model_dir_uses_repo_local_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp) / "panel"

            self.assertEqual(model_dir_from_value("", panel_dir=panel_dir), (panel_dir / "models").resolve())

    def test_model_path_inside_model_dir_is_stored_as_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models"
            model_dir.mkdir()
            model_path = model_dir / "chat.gguf"
            model_path.write_text("model", encoding="utf-8")

            self.assertEqual(model_config_value(str(model_path), model_dir), "chat.gguf")

    def test_log_dir_inside_panel_dir_is_stored_as_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp) / "panel"
            log_dir = panel_dir / "logs"
            log_dir.mkdir(parents=True)

            self.assertEqual(compact_path_value(str(log_dir), base_dir=panel_dir), "logs")

    def test_import_model_copies_gguf_into_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "downloads"
            model_dir = Path(tmp) / "models"
            source_dir.mkdir()
            source = source_dir / "embed.gguf"
            source.write_bytes(b"gguf")

            destination = import_model_file(source, model_dir)

            self.assertEqual(destination, model_dir.resolve() / "embed.gguf")
            self.assertEqual(destination.read_bytes(), b"gguf")
            with self.assertRaises(PanelError):
                import_model_file(source, model_dir)

    def test_import_model_rejects_non_gguf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.bin"
            source.write_bytes(b"raw")

            with self.assertRaises(PanelError):
                import_model_file(source, Path(tmp) / "models")

    def test_gui_override_file_wins_after_legacy_shell_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            model_dir = panel_dir / "models"
            model_dir.mkdir()
            (panel_dir / "env.local.sh").write_text('export CHAT_MODEL="legacy.gguf"\n', encoding="utf-8")
            save_gui_overrides(panel_dir, {"MODEL_DIR": str(model_dir), "CHAT_MODEL": "gui.gguf"})

            config = load_config(panel_dir, apply_tune=False)

            self.assertEqual(config["CHAT_MODEL"], str((model_dir / "gui.gguf").resolve()))
            self.assertTrue(gui_override_path(panel_dir).is_file())

    def test_build_gui_overrides_compacts_model_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp) / "panel"
            model_dir = Path(tmp) / "models"
            log_dir = panel_dir / "logs"
            model_dir.mkdir()
            log_dir.mkdir(parents=True)

            overrides = build_gui_overrides(
                {
                    "MODEL_DIR": str(model_dir),
                    "LOG_DIR": str(log_dir),
                    "CHAT_MODEL": str(model_dir / "chat.gguf"),
                    "CHAT_ALIAS": "chat-local",
                },
                panel_dir=panel_dir,
            )

            self.assertEqual(overrides["CHAT_MODEL"], "chat.gguf")
            self.assertEqual(overrides["LOG_DIR"], "logs")
            self.assertEqual(overrides["CHAT_ALIAS"], "chat-local")

    def test_default_assign_key_tracks_role_tabs(self) -> None:
        self.assertEqual(default_assign_key_for_role("chat"), "CHAT_MODEL")
        self.assertEqual(default_assign_key_for_role("embed"), "EMBED_MODEL")
        self.assertEqual(default_assign_key_for_role("vision"), "VISION_MODEL")
        with self.assertRaises(PanelError):
            default_assign_key_for_role("rerank")


if __name__ == "__main__":
    unittest.main()
