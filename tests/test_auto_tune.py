from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import auto_tune
import llama_runtime
from llama_runtime import PanelError


class AutoTuneFailureHandlingTest(unittest.TestCase):
    def setUp(self) -> None:
        auto_tune.set_tune_log_path(None)
        auto_tune.reset_startup_failure_state()

    def tearDown(self) -> None:
        auto_tune.set_tune_log_path(None)
        auto_tune.reset_startup_failure_state()

    def test_wait_for_server_stops_when_process_exits(self) -> None:
        proc = SimpleNamespace(returncode=42, poll=lambda: 42)

        with (
            patch("auto_tune.err"),
            patch("auto_tune.request_json") as request_json,
            patch("auto_tune.time.sleep") as sleep,
        ):
            self.assertFalse(auto_tune.wait_for_server("127.0.0.1", 9998, 120, proc=proc))

        request_json.assert_not_called()
        sleep.assert_not_called()

    def test_startup_memory_pressure_message_detects_close_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "server.log"
            log_path.write_text(
                "common_params_fit_impl: projected to use 12000 MiB of device memory "
                "vs. 12300 MiB of free device memory\n",
                encoding="utf-8",
            )

            message = auto_tune.startup_memory_pressure_message(log_path, memory_headroom_mib=512)

        self.assertIn("projected device memory 12000 MiB", message)

    def test_wait_for_server_terminates_candidate_on_memory_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "server.log"
            log_path.write_text(
                "common_params_fit_impl: cannot meet free memory target of 1024 MiB\n",
                encoding="utf-8",
            )
            proc = SimpleNamespace(returncode=None, poll=lambda: None)

            with (
                patch("auto_tune.terminate_process") as terminate_process,
                patch("auto_tune.request_json") as request_json,
                patch("auto_tune.time.sleep") as sleep,
                patch("auto_tune.err"),
            ):
                self.assertFalse(auto_tune.wait_for_server("127.0.0.1", 9998, 120, proc=proc, log_path=log_path))

        terminate_process.assert_called_once_with(proc)
        request_json.assert_not_called()
        sleep.assert_not_called()

    def test_wait_for_server_reports_windows_native_crash_exit_code(self) -> None:
        proc = SimpleNamespace(returncode=3221225477, poll=lambda: 3221225477)

        with (
            patch("auto_tune.err") as err,
            patch("auto_tune.request_json") as request_json,
            patch("auto_tune.time.sleep") as sleep,
        ):
            self.assertFalse(auto_tune.wait_for_server("127.0.0.1", 9998, 120, proc=proc))

        messages = "\n".join(str(call.args[0]) for call in err.call_args_list)
        self.assertIn("Server exited during startup with code 3221225477", messages)
        self.assertIn("0xC0000005", messages)
        self.assertIn("access violation", messages)
        request_json.assert_not_called()
        sleep.assert_not_called()

    def test_wait_for_server_writes_windows_native_crash_to_candidate_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "candidate.log"
            log_path.write_text("[panel] launching candidate\n", encoding="utf-8")
            proc = SimpleNamespace(returncode=3221225477, poll=lambda: 3221225477)

            with (
                patch("auto_tune.err"),
                patch("auto_tune.request_json") as request_json,
                patch("auto_tune.time.sleep") as sleep,
            ):
                self.assertFalse(auto_tune.wait_for_server("127.0.0.1", 9998, 120, proc=proc, log_path=log_path))

            text = log_path.read_text(encoding="utf-8")
            self.assertIn("server exited during startup with code 3221225477", text)
            self.assertIn("0xC0000005", text)
            self.assertIn("native llama.cpp crash", text)
            request_json.assert_not_called()
            sleep.assert_not_called()

    def test_chat_tune_rejects_all_zero_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            config = {
                "LLAMA_HOST": "127.0.0.1",
                "LLAMA_SERVER_PANEL_DIR": str(panel_dir),
                "CHAT_MODEL": str(panel_dir / "models" / "chat.gguf"),
                "CHAT_CTX_SIZE": "4096",
            }

            with (
                patch("auto_tune.repo_dir", return_value=panel_dir),
                patch("auto_tune.load_config", return_value=config),
                patch("auto_tune.validate_role_files"),
                patch("auto_tune.port_in_use", return_value=False),
                patch("auto_tune.cpu_counts", return_value=(10, 6)),
                patch("auto_tune.thread_candidates", return_value=[2]),
                patch("auto_tune.bench_chat_like", return_value=0.0) as bench_chat_like,
                patch("builtins.print"),
            ):
                with self.assertRaisesRegex(PanelError, "No working chat tuning configuration"):
                    auto_tune.main(["chat"])

            tune_log = panel_dir / "bench-results" / "tuned" / "server-tune.log"
            tune_log_text = tune_log.read_text(encoding="utf-8")
            self.assertIn("candidate log:", tune_log_text)
            self.assertIn("No working chat tuning configuration started", tune_log_text)
            candidate_paths = {call.kwargs["log_path"] for call in bench_chat_like.call_args_list}
            self.assertNotIn(tune_log, candidate_paths)
            self.assertTrue(all("candidate-" in path.name for path in candidate_paths))

    def test_context_candidates_preserve_configured_value_then_lower_sizes(self) -> None:
        self.assertEqual(auto_tune.context_candidates("4000"), ["4000", "3072", "2048", "1024", "512"])
        self.assertEqual(auto_tune.context_candidates("1024"), ["1024", "512"])
        self.assertEqual(auto_tune.context_candidates("custom"), ["custom"])
        self.assertEqual(auto_tune.context_candidates("0"), ["0"])
        self.assertEqual(auto_tune.context_candidates(""), [""])

    def test_chat_tune_retries_lower_context_after_memory_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            config = {
                "LLAMA_HOST": "127.0.0.1",
                "LLAMA_SERVER_PANEL_DIR": str(panel_dir),
                "CHAT_MODEL": str(panel_dir / "models" / "chat.gguf"),
                "CHAT_CTX_SIZE": "4000",
            }

            def bench_with_context(*_args, **kwargs) -> float:
                base_config = _args[1]
                log_path = kwargs["log_path"]
                if base_config["CHAT_CTX_SIZE"] == "4000":
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_text(
                        "common_params_fit_impl: cannot meet free memory target of 1024 MiB\n",
                        encoding="utf-8",
                    )
                    return 0.0
                return 12.0 if kwargs["cache_k"] == "q4_0" and kwargs["cache_v"] == "q4_0" else 4.0

            with (
                patch("auto_tune.repo_dir", return_value=panel_dir),
                patch("auto_tune.load_config", return_value=config),
                patch("auto_tune.validate_role_files"),
                patch("auto_tune.port_in_use", return_value=False),
                patch("auto_tune.cpu_counts", return_value=(10, 6)),
                patch("auto_tune.host_label", return_value="test-host"),
                patch("auto_tune.thread_candidates", return_value=[2]),
                patch("auto_tune.bench_chat_like", side_effect=bench_with_context) as bench_chat_like,
                patch("builtins.print"),
            ):
                self.assertEqual(auto_tune.main(["chat"]), 0)

            contexts = [call.args[1]["CHAT_CTX_SIZE"] for call in bench_chat_like.call_args_list]
            self.assertEqual(contexts, ["4000", "4000", "4000", "4000", "3072", "3072", "3072", "3072"])

            tune_path = panel_dir / "bench-results" / "tuned" / "chat.chat.sh"
            tune_text = tune_path.read_text(encoding="utf-8")
            self.assertIn("export CHAT_CTX_SIZE=3072", tune_text)
            self.assertIn("export CHAT_THREADS=2", tune_text)
            self.assertIn("export CHAT_CACHE_TYPE_K=q4_0", tune_text)
            self.assertIn("export CHAT_CACHE_TYPE_V=q4_0", tune_text)

            tune_log = panel_dir / "bench-results" / "tuned" / "server-tune.log"
            tune_log_text = tune_log.read_text(encoding="utf-8")
            self.assertIn("retrying with lower context", tune_log_text)
            self.assertIn("Best: ctx=3072", tune_log_text)

    def test_chat_tune_retries_lower_context_after_projected_memory_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            config = {
                "LLAMA_HOST": "127.0.0.1",
                "LLAMA_SERVER_PANEL_DIR": str(panel_dir),
                "CHAT_MODEL": str(panel_dir / "models" / "chat.gguf"),
                "CHAT_CTX_SIZE": "4000",
            }

            def bench_with_projection(*_args, **kwargs) -> float:
                base_config = _args[1]
                log_path = kwargs["log_path"]
                if base_config["CHAT_CTX_SIZE"] == "4000":
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_text(
                        "common_params_fit_impl: projected to use 12000 MiB of device memory "
                        "vs. 12300 MiB of free device memory\n",
                        encoding="utf-8",
                    )
                    return 0.0
                return 9.0

            with (
                patch("auto_tune.repo_dir", return_value=panel_dir),
                patch("auto_tune.load_config", return_value=config),
                patch("auto_tune.validate_role_files"),
                patch("auto_tune.port_in_use", return_value=False),
                patch("auto_tune.cpu_counts", return_value=(10, 6)),
                patch("auto_tune.host_label", return_value="test-host"),
                patch("auto_tune.thread_candidates", return_value=[2]),
                patch("auto_tune.bench_chat_like", side_effect=bench_with_projection) as bench_chat_like,
                patch("builtins.print"),
            ):
                self.assertEqual(auto_tune.main(["chat"]), 0)

            contexts = [call.args[1]["CHAT_CTX_SIZE"] for call in bench_chat_like.call_args_list]
            self.assertEqual(contexts, ["4000", "4000", "4000", "4000", "3072", "3072", "3072", "3072"])

    def test_chat_tune_does_not_retry_lower_context_without_memory_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            config = {
                "LLAMA_HOST": "127.0.0.1",
                "LLAMA_SERVER_PANEL_DIR": str(panel_dir),
                "CHAT_MODEL": str(panel_dir / "models" / "chat.gguf"),
                "CHAT_CTX_SIZE": "4000",
            }

            def bench_without_pressure(*_args, **kwargs) -> float:
                log_path = kwargs["log_path"]
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("server exited before health check\n", encoding="utf-8")
                return 0.0

            with (
                patch("auto_tune.repo_dir", return_value=panel_dir),
                patch("auto_tune.load_config", return_value=config),
                patch("auto_tune.validate_role_files"),
                patch("auto_tune.port_in_use", return_value=False),
                patch("auto_tune.cpu_counts", return_value=(10, 6)),
                patch("auto_tune.thread_candidates", return_value=[2]),
                patch("auto_tune.bench_chat_like", side_effect=bench_without_pressure) as bench_chat_like,
                patch("builtins.print"),
            ):
                with self.assertRaisesRegex(PanelError, "No working chat tuning configuration"):
                    auto_tune.main(["chat"])

            contexts = [call.args[1]["CHAT_CTX_SIZE"] for call in bench_chat_like.call_args_list]
            self.assertEqual(contexts, ["4000", "4000", "4000", "4000"])

            tune_log = panel_dir / "bench-results" / "tuned" / "server-tune.log"
            tune_log_text = tune_log.read_text(encoding="utf-8")
            self.assertIn("not trying lower context sizes", tune_log_text)

    def test_chat_tune_reports_windows_native_crash_as_cause_of_all_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            config = {
                "LLAMA_HOST": "127.0.0.1",
                "LLAMA_SERVER_PANEL_DIR": str(panel_dir),
                "CHAT_MODEL": str(panel_dir / "models" / "chat.gguf"),
                "CHAT_CTX_SIZE": "4000",
            }

            def bench_native_crash(*_args, **kwargs) -> float:
                log_path = kwargs["log_path"]
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("Server exited during startup with code 3221225477\n", encoding="utf-8")
                auto_tune.record_startup_exit_code(3221225477)
                return 0.0

            with (
                patch("auto_tune.repo_dir", return_value=panel_dir),
                patch("auto_tune.load_config", return_value=config),
                patch("auto_tune.validate_role_files"),
                patch("auto_tune.port_in_use", return_value=False),
                patch("auto_tune.cpu_counts", return_value=(10, 6)),
                patch("auto_tune.thread_candidates", return_value=[2]),
                patch("auto_tune.bench_chat_like", side_effect=bench_native_crash) as bench_chat_like,
                patch("builtins.print"),
            ):
                with self.assertRaises(PanelError) as raised:
                    auto_tune.main(["chat"])

            message = str(raised.exception)
            self.assertIn("No working chat tuning configuration started", message)
            self.assertIn("3221225477", message)
            self.assertIn("0xC0000005", message)
            self.assertIn("native llama.cpp crash", message)

            contexts = [call.args[1]["CHAT_CTX_SIZE"] for call in bench_chat_like.call_args_list]
            self.assertEqual(contexts, ["4000", "4000", "4000", "4000"])

            tune_log = panel_dir / "bench-results" / "tuned" / "server-tune.log"
            tune_log_text = tune_log.read_text(encoding="utf-8")
            self.assertIn("candidate log:", tune_log_text)
            self.assertIn("0xC0000005", tune_log_text)
            self.assertIn("native llama.cpp crash", tune_log_text)

    def test_bench_chat_like_estimates_score_from_usage_when_timings_missing(self) -> None:
        proc = SimpleNamespace(returncode=None, poll=lambda: None)

        with (
            patch("auto_tune.start_server", return_value=proc),
            patch("auto_tune.wait_for_server", return_value=True),
            patch("auto_tune.request_json", side_effect=[{}, {"usage": {"completion_tokens": 20}}]),
            patch("auto_tune.score_from_slots", return_value=0.0),
            patch("auto_tune.terminate_process"),
            patch("auto_tune.time.monotonic", side_effect=[10.0, 12.0]),
            patch("auto_tune.time.sleep"),
            patch("builtins.print"),
        ):
            score = auto_tune.bench_chat_like(
                "chat",
                {},
                host="127.0.0.1",
                port=9998,
                threads=2,
                cache_k="q8_0",
                cache_v="q8_0",
                log_path=Path("candidate.log"),
                prompt="hello",
                startup_timeout=1,
            )

        self.assertEqual(score, 10.0)

    def test_start_server_writes_candidate_launch_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            binary = panel_dir / "llama-server"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            log_path = panel_dir / "candidate.log"
            config = {
                "LLAMA_SERVER_BIN": str(binary),
                "CHAT_MODEL": str(panel_dir / "chat.gguf"),
                "CHAT_ALIAS": "chat",
                "CHAT_CTX_SIZE": "4096",
                "CHAT_THREADS": "2",
                "CHAT_PARALLEL": "1",
                "CHAT_CPU_MOE_LAYERS": "0",
                "CHAT_CACHE_TYPE_K": "q8_0",
                "CHAT_CACHE_TYPE_V": "q8_0",
                "CHAT_TEMPERATURE": "0.6",
                "CHAT_TOP_K": "20",
                "CHAT_TOP_P": "0.95",
                "CHAT_MIN_P": "0",
                "CHAT_PRESENCE_PENALTY": "1.5",
            }

            with (
                patch("auto_tune.prepare_llama_server_argv", side_effect=lambda argv: (argv, [])),
                patch("auto_tune.subprocess.Popen", return_value=SimpleNamespace(pid=123)) as popen,
            ):
                auto_tune.start_server("chat", config, host="127.0.0.1", port=9998, log_path=log_path)

            text = log_path.read_text(encoding="utf-8")
            self.assertIn("candidate log:", text)
            self.assertIn("launching chat tune candidate", text)
            self.assertIn("command:", text)
            popen.assert_called_once()

    def test_ensure_tune_file_reports_tune_path_and_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            model_dir = panel_dir / "models"
            model_dir.mkdir()
            binary = panel_dir / "llama-server"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            model = model_dir / "chat.gguf"
            model.write_text("model", encoding="utf-8")
            (panel_dir / "env.local.json").write_text(
                json.dumps(
                    {
                        "LLAMA_SERVER_BIN": str(binary),
                        "MODEL_DIR": str(model_dir),
                        "CHAT_MODEL": model.name,
                    }
                ),
                encoding="utf-8",
            )

            completed = SimpleNamespace(returncode=7)
            with patch("llama_runtime.subprocess.run", return_value=completed) as run:
                with self.assertRaises(PanelError) as ctx:
                    llama_runtime.ensure_tune_file("chat", panel_dir)

            message = str(ctx.exception)
            self.assertIn("auto-tune failed for chat", message)
            self.assertIn("chat.chat.sh", message)
            self.assertIn("server-tune.log", message)
            run.assert_called_once()
            self.assertIn("launching auto-tune chat", (panel_dir / "bench-results" / "tuned" / "server-tune.log").read_text(encoding="utf-8"))

    def test_ensure_tune_file_uses_bundled_auto_tune_when_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            model_dir = panel_dir / "models"
            model_dir.mkdir()
            binary = panel_dir / "llama-server"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            model = model_dir / "chat.gguf"
            model.write_text("model", encoding="utf-8")
            (panel_dir / "env.local.json").write_text(
                json.dumps(
                    {
                        "LLAMA_SERVER_BIN": str(binary),
                        "MODEL_DIR": str(model_dir),
                        "CHAT_MODEL": model.name,
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.object(llama_runtime.sys, "frozen", True, create=True),
                patch("llama_runtime.subprocess.run") as run,
                patch("auto_tune.main", return_value=0) as auto_tune_main,
            ):
                llama_runtime.ensure_tune_file("chat", panel_dir)

            run.assert_not_called()
            auto_tune_main.assert_called_once_with(["chat"])
            tune_log = (panel_dir / "bench-results" / "tuned" / "server-tune.log").read_text(encoding="utf-8")
            self.assertIn("launching auto-tune chat", tune_log)
            self.assertIn("<bundled>", tune_log)
            self.assertIn("auto_tune", tune_log)


if __name__ == "__main__":
    unittest.main()
