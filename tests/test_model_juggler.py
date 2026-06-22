from __future__ import annotations

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

from model_juggler import StartupError, wait_ready


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


if __name__ == "__main__":
    unittest.main()
