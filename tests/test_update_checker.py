from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from update_checker import (
    VERSION_ENV,
    LatestRelease,
    ReleaseAsset,
    UpdateCheckError,
    VersionSource,
    _installer_script_contents,
    build_update_result,
    check_for_updates,
    compare_versions,
    current_app_version,
    download_url,
    download_update_archive,
    executable_name,
    expected_archive_name,
    extract_update_archive,
    fetch_latest_release,
    format_byte_count,
    parse_latest_release,
    parse_checksum_file,
    resolve_update_repo,
    verify_archive_checksum,
)


class FakeResponse:
    def __init__(self, body: bytes, *, headers: dict[str, str] | None = None) -> None:
        self.body = body
        self.headers = headers or {}
        self.offset = 0

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if self.offset >= len(self.body):
            return b""
        if size is None or size < 0:
            size = len(self.body) - self.offset
        chunk = self.body[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


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

    def test_parse_latest_release_parses_download_assets(self) -> None:
        release = parse_latest_release(
            {
                "tag_name": "v0.2.0",
                "html_url": "https://example.test/release",
                "assets": [
                    {
                        "name": "llama-server-panel-macos-arm64.zip",
                        "browser_download_url": "https://example.test/app.zip",
                    },
                    {
                        "name": "llama-server-panel-macos-arm64.zip.sha256",
                        "browser_download_url": "https://example.test/app.zip.sha256",
                    },
                    {"name": "", "browser_download_url": "https://example.test/ignored.zip"},
                    {"name": "ignored.zip", "browser_download_url": ""},
                ],
            }
        )

        self.assertEqual(
            release.assets,
            (
                ReleaseAsset("llama-server-panel-macos-arm64.zip", "https://example.test/app.zip"),
                ReleaseAsset("llama-server-panel-macos-arm64.zip.sha256", "https://example.test/app.zip.sha256"),
            ),
        )

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

    def test_format_byte_count_reports_human_readable_values(self) -> None:
        self.assertEqual(format_byte_count(512), "512 bytes")
        self.assertEqual(format_byte_count(1536), "1.5 KB")
        self.assertEqual(format_byte_count(2 * 1024 * 1024), "2.0 MB")

    def test_download_url_streams_chunks_and_reports_progress(self) -> None:
        progress: list[str] = []

        def opener(request: object, timeout: int) -> FakeResponse:
            self.assertEqual(timeout, 8)
            return FakeResponse(b"abc123", headers={"Content-Length": "6"})

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "download.bin"
            download_url("https://example.test/download.bin", destination, opener=opener, progress=progress.append)

            self.assertEqual(destination.read_bytes(), b"abc123")
            self.assertIn("Downloaded 6 bytes of 6 bytes", progress)

    def test_download_update_archive_selects_platform_asset_and_verifies_checksum(self) -> None:
        archive_name = expected_archive_name()
        archive_bytes = b"archive bytes"
        digest = __import__("hashlib").sha256(archive_bytes).hexdigest()
        release = LatestRelease(
            "v0.2.0",
            "https://example.test/release",
            assets=(
                ReleaseAsset("other-platform.zip", "https://example.test/other.zip"),
                ReleaseAsset(archive_name, "https://example.test/app.zip"),
                ReleaseAsset(f"{archive_name}.sha256", "https://example.test/app.zip.sha256"),
            ),
        )

        def opener(request: object, timeout: int) -> FakeResponse:
            if request.full_url == "https://example.test/app.zip":
                return FakeResponse(archive_bytes)
            if request.full_url == "https://example.test/app.zip.sha256":
                return FakeResponse(f"{digest}  {archive_name}\n".encode("utf-8"))
            self.fail(f"unexpected URL {request.full_url}")

        with tempfile.TemporaryDirectory() as tmp:
            archive_path = download_update_archive(release, Path(tmp), opener=opener)

            self.assertEqual(archive_path.name, archive_name)
            self.assertEqual(archive_path.read_bytes(), archive_bytes)

    def test_download_update_archive_requires_platform_asset(self) -> None:
        release = LatestRelease(
            "v0.2.0",
            "https://example.test/release",
            assets=(ReleaseAsset("other-platform.zip", "https://example.test/other.zip"),),
        )

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(UpdateCheckError, expected_archive_name()):
                download_update_archive(release, Path(tmp), opener=lambda request, timeout: FakeResponse(b""))

    def test_verify_archive_checksum_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "app.zip"
            archive_path.write_bytes(b"bad")

            self.assertEqual(parse_checksum_file("0" * 64 + "  app.zip\n", "app.zip"), "0" * 64)
            with self.assertRaisesRegex(UpdateCheckError, "checksum did not match"):
                verify_archive_checksum(archive_path, "0" * 64 + "  app.zip\n", "app.zip")

    def test_extract_update_archive_rejects_unsafe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "update.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.txt", "bad")
                archive.writestr(executable_name(), "binary")

            with self.assertRaisesRegex(UpdateCheckError, "unsafe path"):
                extract_update_archive(archive_path, root / "extracted")

    def test_extract_update_archive_requires_executable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "update.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(VERSION_ENV, "v0.2.0")

            with self.assertRaisesRegex(UpdateCheckError, executable_name()):
                extract_update_archive(archive_path, root / "extracted")

    def test_installer_script_contents_windows_has_wait_copy_and_restart(self) -> None:
        with patch("update_checker.os.name", "nt"):
            script = _installer_script_contents()

        self.assertIn("@echo off", script)
        self.assertIn('set "PID=%~1"', script)
        self.assertIn('set "SOURCE_DIR=%~2"', script)
        self.assertIn('set "INSTALL_DIR=%~3"', script)
        self.assertIn('set "EXE_PATH=%~4"', script)
        self.assertIn('tasklist /FI "PID eq %PID%"', script)
        self.assertIn("timeout /t 1 /nobreak >nul", script)
        self.assertIn('if exist "%SOURCE_DIR%\\llama-server-panel.exe" copy /Y', script)
        self.assertIn('start "" "%EXE_PATH%"', script)

    def test_resolve_update_repo_rejects_invalid_override(self) -> None:
        with self.assertRaises(UpdateCheckError):
            resolve_update_repo({"LLAMA_SERVER_PANEL_UPDATE_REPO": "not-a-slug"})


if __name__ == "__main__":
    unittest.main()
