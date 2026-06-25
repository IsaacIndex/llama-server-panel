from __future__ import annotations

import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_release
from update_checker import VERSION_FILE_NAME, VersionSource


class BuildReleaseTest(unittest.TestCase):
    def test_pyinstaller_command_uses_windowed_mode_on_windows(self) -> None:
        dist_platform_dir = Path("dist") / "windows-x64"
        with patch.object(build_release.os, "name", "nt"):
            command = build_release.pyinstaller_command(dist_platform_dir)

        self.assertIn("--windowed", command)

    def test_pyinstaller_command_keeps_console_available_on_non_windows(self) -> None:
        dist_platform_dir = Path("dist") / "macos-arm64"
        with patch.object(build_release.os, "name", "posix"):
            command = build_release.pyinstaller_command(dist_platform_dir)

        self.assertNotIn("--windowed", command)

    def test_release_version_prefers_github_tag_ref(self) -> None:
        with patch.dict(os.environ, {"GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "v9.8.7"}):
            self.assertEqual(build_release.release_version(), "v9.8.7")

    def test_build_archive_includes_version_when_known(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            executable = tmp_dir / build_release.executable_name()
            executable.write_bytes(b"binary")

            with (
                patch.object(build_release, "RELEASE_DIR", tmp_dir / "release"),
                patch.object(build_release, "ROOT", tmp_dir / "root"),
                patch.object(build_release, "platform_slug", return_value="test-platform"),
                patch.object(build_release, "release_version", return_value="v1.2.3"),
            ):
                archive_path = build_release.build_archive(executable)

            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(archive.read(VERSION_FILE_NAME).decode("utf-8"), "v1.2.3\n")

    def test_build_archive_prefers_tag_version_when_current_version_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            executable = tmp_dir / build_release.executable_name()
            executable.write_bytes(b"binary")

            with (
                patch.dict(os.environ, {"GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "v2.3.4"}),
                patch.object(build_release, "RELEASE_DIR", tmp_dir / "release"),
                patch.object(build_release, "ROOT", tmp_dir / "root"),
                patch.object(build_release, "platform_slug", return_value="test-platform"),
                patch.object(build_release, "current_app_version", return_value=VersionSource(None, "unknown")),
            ):
                archive_path = build_release.build_archive(executable)

            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(archive.read(VERSION_FILE_NAME).decode("utf-8"), "v2.3.4\n")

    def test_build_archive_omits_version_when_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            executable = tmp_dir / build_release.executable_name()
            executable.write_bytes(b"binary")

            with (
                patch.dict(os.environ, {"GITHUB_REF_TYPE": "", "GITHUB_REF_NAME": ""}),
                patch.object(build_release, "RELEASE_DIR", tmp_dir / "release"),
                patch.object(build_release, "ROOT", tmp_dir / "root"),
                patch.object(build_release, "platform_slug", return_value="test-platform"),
                patch.object(build_release, "current_app_version", return_value=VersionSource(None, "unknown")),
            ):
                archive_path = build_release.build_archive(executable)

            with zipfile.ZipFile(archive_path) as archive:
                self.assertNotIn(VERSION_FILE_NAME, archive.namelist())


if __name__ == "__main__":
    unittest.main()
