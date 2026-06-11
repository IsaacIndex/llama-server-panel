from __future__ import annotations

import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from update_checker import (
    VERSION_ENV,
    LatestRelease,
    UpdateCheckError,
    VersionSource,
    build_update_result,
    check_for_updates,
    compare_versions,
    current_app_version,
    fetch_latest_release,
    parse_latest_release,
    resolve_update_repo,
)


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class UpdateCheckerTest(unittest.TestCase):
    def test_compare_versions_handles_v_prefix_and_numeric_order(self) -> None:
        self.assertEqual(compare_versions("v0.1.0", "v0.2.0"), -1)
        self.assertEqual(compare_versions("0.10.0", "0.2.0"), 1)
        self.assertEqual(compare_versions("v1.2", "1.2.0"), 0)
        self.assertIsNone(compare_versions("dev", "v1.0.0"))

    def test_current_app_version_prefers_env_then_version_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            (panel_dir / "VERSION").write_text("v0.1.0\n", encoding="utf-8")

            self.assertEqual(current_app_version(panel_dir, environ={VERSION_ENV: "v0.2.0"}).value, "v0.2.0")

            version = current_app_version(panel_dir, environ={})

            self.assertEqual(version.value, "v0.1.0")
            self.assertEqual(version.source, str((panel_dir / "VERSION").resolve()))

            with patch.dict("os.environ", {VERSION_ENV: "v9.9.9"}):
                self.assertEqual(current_app_version(panel_dir, environ={}).value, "v0.1.0")

    def test_parse_latest_release_requires_tag(self) -> None:
        release = parse_latest_release({"tag_name": "v0.2.0", "html_url": "https://example.test/release"})

        self.assertEqual(release.tag_name, "v0.2.0")
        self.assertEqual(release.html_url, "https://example.test/release")

        with self.assertRaises(UpdateCheckError):
            parse_latest_release({"html_url": "https://example.test/release"})

    def test_build_update_result_reports_available_current_and_unknown(self) -> None:
        latest = LatestRelease("v0.2.0", "https://example.test/release")

        available = build_update_result(VersionSource("v0.1.0", "VERSION"), latest)
        self.assertTrue(available.update_available)
        self.assertTrue(available.comparable)
        self.assertIn("Update available", available.message)

        current = build_update_result(VersionSource("v0.2.0", "VERSION"), latest)
        self.assertFalse(current.update_available)
        self.assertTrue(current.comparable)
        self.assertIn("matches latest release", current.message)

        unknown = build_update_result(VersionSource(None, "unknown"), latest)
        self.assertFalse(unknown.update_available)
        self.assertFalse(unknown.comparable)
        self.assertIn("current app version is unknown", unknown.message)

    def test_check_for_updates_uses_fake_github_response(self) -> None:
        def opener(request: object, timeout: int) -> FakeResponse:
            self.assertEqual(timeout, 8)
            self.assertIn("/repos/IsaacIndex/llama-server-panel/releases/latest", request.full_url)
            return FakeResponse(b'{"tag_name":"v0.2.0","html_url":"https://example.test/release"}')

        with tempfile.TemporaryDirectory() as tmp:
            result = check_for_updates(Path(tmp), environ={VERSION_ENV: "v0.1.0"}, opener=opener)

        self.assertTrue(result.update_available)
        self.assertEqual(result.latest.tag_name, "v0.2.0")

    def test_check_for_updates_reports_unknown_current_version(self) -> None:
        def opener(request: object, timeout: int) -> FakeResponse:
            return FakeResponse(b'{"tag_name":"v0.2.0","html_url":"https://example.test/release"}')

        with tempfile.TemporaryDirectory() as tmp:
            result = check_for_updates(Path(tmp), environ={}, opener=opener)

        self.assertFalse(result.update_available)
        self.assertFalse(result.comparable)
        self.assertIn("current app version is unknown", result.message)

    def test_fetch_latest_release_rejects_malformed_json(self) -> None:
        with self.assertRaises(UpdateCheckError):
            fetch_latest_release("IsaacIndex/llama-server-panel", opener=lambda request, timeout: FakeResponse(b"{"))

        with self.assertRaises(UpdateCheckError):
            fetch_latest_release("IsaacIndex/llama-server-panel", opener=lambda request, timeout: FakeResponse(b"[]"))

    def test_resolve_update_repo_rejects_invalid_override(self) -> None:
        with self.assertRaises(UpdateCheckError):
            resolve_update_repo({"LLAMA_SERVER_PANEL_UPDATE_REPO": "not-a-slug"})


if __name__ == "__main__":
    unittest.main()
