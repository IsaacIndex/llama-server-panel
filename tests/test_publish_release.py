from __future__ import annotations

import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from publish_release import ensure_clean_worktree, next_release_tag, resolve_release_tag, wait_for_ci


def completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class PublishReleaseTest(unittest.TestCase):
    def test_next_release_tag_defaults_to_next_minor(self) -> None:
        self.assertEqual(next_release_tag(["v0.6.0", "v0.5.0", "not-a-release"], "minor"), "v0.7.0")
        self.assertEqual(next_release_tag(["v0.6.0"], "patch"), "v0.6.1")
        self.assertEqual(next_release_tag(["v0.6.0"], "major"), "v1.0.0")

    def test_resolve_release_tag_uses_explicit_tag_when_set(self) -> None:
        def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
            self.fail(f"unexpected git call: {command}")

        self.assertEqual(resolve_release_tag("minor", runner, explicit_tag="v2.3.4"), "v2.3.4")

    def test_resolve_release_tag_rejects_invalid_explicit_tag(self) -> None:
        with self.assertRaisesRegex(ValueError, "vMAJOR.MINOR.PATCH"):
            resolve_release_tag("minor", explicit_tag="v2.3")

    def test_ensure_clean_worktree_fails_when_status_has_output(self) -> None:
        with redirect_stdout(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Working tree is not clean"):
                ensure_clean_worktree(lambda command: completed(" M scripts/panel_gui.py\n"))

    def test_wait_for_ci_returns_when_latest_run_succeeds(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            if command == ["git", "rev-parse", "HEAD"]:
                return completed("abc123def456\n")
            if command[:3] == ["gh", "run", "list"]:
                return completed('[{"status":"completed","conclusion":"success","url":"https://example.test/run"}]')
            self.fail(f"unexpected command: {command}")

        with redirect_stdout(StringIO()):
            wait_for_ci("CI", timeout_seconds=1, poll_interval_seconds=1, runner=runner, sleeper=lambda _: None)

        self.assertEqual(calls[0], ["git", "rev-parse", "HEAD"])
        self.assertIn("--commit", calls[1])
        self.assertIn("abc123def456", calls[1])

    def test_wait_for_ci_fails_when_latest_run_fails(self) -> None:
        def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
            if command == ["git", "rev-parse", "HEAD"]:
                return completed("abc123def456\n")
            if command[:3] == ["gh", "run", "list"]:
                return completed('[{"status":"completed","conclusion":"failure"}]')
            self.fail(f"unexpected command: {command}")

        with redirect_stdout(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "completed with conclusion failure"):
                wait_for_ci("CI", timeout_seconds=1, poll_interval_seconds=1, runner=runner, sleeper=lambda _: None)

    def test_wait_for_ci_times_out_before_publishing(self) -> None:
        def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
            if command == ["git", "rev-parse", "HEAD"]:
                return completed("abc123def456\n")
            if command[:3] == ["gh", "run", "list"]:
                return completed('[{"status":"in_progress","conclusion":""}]')
            self.fail(f"unexpected command: {command}")

        with redirect_stdout(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Timed out waiting for CI"):
                wait_for_ci("CI", timeout_seconds=0, poll_interval_seconds=1, runner=runner, sleeper=lambda _: None)


if __name__ == "__main__":
    unittest.main()
