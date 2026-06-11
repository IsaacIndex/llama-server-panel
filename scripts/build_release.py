#!/usr/bin/env python3
"""Build a distributable GUI executable archive for the current platform."""

from __future__ import annotations

import hashlib
import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "llama-server-panel"
BUILD_DIR = ROOT / "build" / "pyinstaller"
DIST_DIR = ROOT / "dist"
RELEASE_DIR = ROOT / "release"


def platform_slug() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower().replace("amd64", "x64").replace("x86_64", "x64")
    if system == "darwin":
        system = "macos"
    return f"{system}-{machine}"


def executable_name() -> str:
    return f"{APP_NAME}.exe" if os.name == "nt" else APP_NAME


def run_pyinstaller() -> Path:
    if importlib.util.find_spec("PyInstaller") is None:
        raise RuntimeError("PyInstaller is not installed. Run: python -m pip install -r requirements-build.txt")

    dist_platform_dir = DIST_DIR / platform_slug()
    shutil.rmtree(dist_platform_dir, ignore_errors=True)
    dist_platform_dir.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--onefile",
        "--hidden-import",
        "tkinter",
        "--hidden-import",
        "tkinter.ttk",
        "--hidden-import",
        "tkinter.filedialog",
        "--hidden-import",
        "tkinter.messagebox",
        "--name",
        APP_NAME,
        "--distpath",
        str(dist_platform_dir),
        "--workpath",
        str(BUILD_DIR / platform_slug()),
        "--specpath",
        str(BUILD_DIR),
        str(ROOT / "scripts" / "panel_gui.py"),
    ]
    subprocess.run(command, cwd=str(ROOT), check=True)

    executable = dist_platform_dir / executable_name()
    if not executable.is_file():
        raise FileNotFoundError(f"Expected executable was not built: {executable}")
    return executable


def write_checksum(path: Path) -> Path:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return checksum_path


def build_archive(executable: Path) -> Path:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RELEASE_DIR / f"{APP_NAME}-{platform_slug()}.zip"
    if archive_path.exists():
        archive_path.unlink()

    docs = [
        ROOT / "README.md",
        ROOT / ".env.example",
        ROOT / "LICENSE",
        ROOT / "SECURITY.md",
    ]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(executable, executable.name)
        for doc in docs:
            if doc.is_file():
                archive.write(doc, doc.name)
    write_checksum(archive_path)
    return archive_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Llama Server Panel executable release archive")
    parser.add_argument(
        "--print-platform",
        action="store_true",
        help="print the release platform slug and exit without building",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.print_platform:
        print(platform_slug())
        return 0

    try:
        executable = run_pyinstaller()
        archive = build_archive(executable)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
