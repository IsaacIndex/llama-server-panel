from __future__ import annotations

import io
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from llama_runtime import PanelError, load_config
from panel_gui import (
    build_chat_payload,
    build_embedding_payload,
    build_gui_overrides,
    chat_model_id_for_role,
    compact_path_value,
    default_assign_key_for_role,
    embedding_model_id_for_config,
    extract_chat_text,
    gui_override_path,
    image_data_url,
    import_model_file,
    model_dir_from_value,
    model_config_value,
    post_json,
    role_log_display_text,
    role_log_path,
    save_gui_overrides,
    start_status_for_auto_tune,
    summarize_embedding_response,
    tail_file_text,
)


class ResponseStub:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "ResponseStub":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.body


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

    def test_start_status_shows_loading_when_auto_tune_will_run(self) -> None:
        with patch("panel_gui.tune_file_exists", side_effect=[True, False]) as tune_file_exists:
            status = start_status_for_auto_tune(True, ("chat", "embed"), ROOT)

        self.assertEqual(status, "Loading")
        self.assertEqual([call.args[0] for call in tune_file_exists.call_args_list], ["chat", "embed"])

    def test_start_status_uses_starting_when_tunes_exist_or_auto_tune_disabled(self) -> None:
        with patch("panel_gui.tune_file_exists", return_value=True):
            self.assertEqual(start_status_for_auto_tune(True, ("chat", "embed"), ROOT), "Starting")
        with patch("panel_gui.tune_file_exists") as tune_file_exists:
            self.assertEqual(start_status_for_auto_tune(False, ("chat",), ROOT), "Starting")
            tune_file_exists.assert_not_called()

    def test_build_chat_payload_uses_text_only_content_without_image(self) -> None:
        payload = build_chat_payload(" hello ", "chat-model")

        self.assertEqual(payload["model"], "chat-model")
        self.assertEqual(payload["messages"][0]["content"], "hello")

    def test_build_chat_payload_embeds_image_data_url_when_image_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "sample.png"
            image.write_bytes(b"\x89PNG\r\n")

            payload = build_chat_payload("describe", "vision-model", image_path=image)

        content = payload["messages"][0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": "describe"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_image_data_url_rejects_missing_file(self) -> None:
        with self.assertRaises(PanelError):
            image_data_url(Path("/missing/not-here.png"))

    def test_image_data_url_rejects_unsupported_mime_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "sample.txt"
            image.write_text("not an image", encoding="utf-8")

            with self.assertRaisesRegex(PanelError, "Unsupported image type"):
                image_data_url(image)

    def test_image_data_url_rejects_large_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "sample.png"
            image.write_bytes(b"12345")

            with self.assertRaisesRegex(PanelError, "too large"):
                image_data_url(image, max_bytes=4)

    def test_build_embedding_payload_trims_input(self) -> None:
        self.assertEqual(build_embedding_payload(" hello world ", "embed.gguf"), {"model": "embed.gguf", "input": "hello world"})

    def test_chat_model_id_for_role_reports_missing_alias(self) -> None:
        with self.assertRaisesRegex(PanelError, "Missing CHAT_ALIAS"):
            chat_model_id_for_role({}, "chat")

    def test_embedding_model_id_for_config_uses_model_basename(self) -> None:
        self.assertEqual(embedding_model_id_for_config({"EMBED_MODEL": "/models/embed.gguf"}), "embed.gguf")
        with self.assertRaisesRegex(PanelError, "Missing EMBED_MODEL"):
            embedding_model_id_for_config({})

    def test_extract_chat_text_handles_openai_chat_response(self) -> None:
        self.assertEqual(
            extract_chat_text({"choices": [{"message": {"content": [{"type": "text", "text": "ok"}]}}]}),
            "ok",
        )

    def test_summarize_embedding_response_reports_dimensions_and_preview(self) -> None:
        summary = summarize_embedding_response({"data": [{"embedding": [0.1, -0.2, 0.3]}]})

        self.assertIn("Embedding dimensions: 3", summary)
        self.assertIn("[0.1000, -0.2000, 0.3000]", summary)

    def test_post_json_returns_decoded_object(self) -> None:
        with patch("panel_gui.urllib.request.urlopen", return_value=ResponseStub(b'{"ok": true}')):
            self.assertEqual(post_json("http://local.test/v1", {"input": "hello"}), {"ok": True})

    def test_post_json_handles_http_error_with_body(self) -> None:
        error = urllib.error.HTTPError("http://local.test/v1", 500, "Server Error", {}, io.BytesIO(b"boom"))

        with patch("panel_gui.urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(PanelError, "HTTP 500.*boom"):
                post_json("http://local.test/v1", {})

    def test_post_json_invalid_json_raises_panel_error(self) -> None:
        with patch("panel_gui.urllib.request.urlopen", return_value=ResponseStub(b"not-json")):
            with self.assertRaisesRegex(PanelError, "Invalid JSON response"):
                post_json("http://local.test/v1", {})

    def test_post_json_network_error_raises_panel_error(self) -> None:
        error = urllib.error.URLError("connection refused")

        with patch("panel_gui.urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(PanelError, "Request failed"):
                post_json("http://local.test/v1", {})

    def test_role_log_path_is_per_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = {"LOG_DIR": tmp}

            self.assertEqual(role_log_path(config, "chat"), Path(tmp) / "chat-gui.log")
            self.assertEqual(role_log_path(config, "embed"), Path(tmp) / "embed-gui.log")
            self.assertEqual(role_log_path(config, "vision"), Path(tmp) / "vision-gui.log")
            with self.assertRaises(PanelError):
                role_log_path(config, "rerank")

    def test_tail_file_text_handles_missing_and_truncates_large_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "chat-gui.log"

            self.assertIn("No log yet", tail_file_text(log_path, max_bytes=8))

            log_path.write_bytes(b"first line\nsecond line\n")
            self.assertEqual(tail_file_text(log_path, max_bytes=128), "first line\nsecond line\n")

            self.assertEqual(
                tail_file_text(log_path, max_bytes=6),
                f"... showing last 6 bytes of {log_path}\n\n line\n",
            )

    def test_role_log_display_includes_role_and_auto_tune_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            log_dir = panel_dir / "logs"
            tune_dir = panel_dir / "bench-results" / "tuned"
            log_dir.mkdir()
            tune_dir.mkdir(parents=True)
            (log_dir / "chat-gui.log").write_text("server startup\n", encoding="utf-8")
            (tune_dir / "server-tune.log").write_text("auto tune candidate\n", encoding="utf-8")

            text = role_log_display_text({"LOG_DIR": str(log_dir)}, "chat", panel_dir=panel_dir)

            self.assertIn("Chat GUI/server log", text)
            self.assertIn("server startup", text)
            self.assertIn("Auto-tune log", text)
            self.assertIn("auto tune candidate", text)

    def test_role_log_display_omits_missing_auto_tune_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            log_dir = panel_dir / "logs"
            log_dir.mkdir()
            (log_dir / "chat-gui.log").write_text("server startup\n", encoding="utf-8")

            text = role_log_display_text({"LOG_DIR": str(log_dir)}, "chat", panel_dir=panel_dir)

            self.assertIn("server startup", text)
            self.assertNotIn("Auto-tune log", text)


if __name__ == "__main__":
    unittest.main()
