#!/usr/bin/env python3
"""GitHub release update checks for Llama Server Panel."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional


APP_NAME = "llama-server-panel"
DEFAULT_UPDATE_REPO = "IsaacIndex/llama-server-panel"
UPDATE_REPO_ENV = "LLAMA_SERVER_PANEL_UPDATE_REPO"
VERSION_ENV = "LLAMA_SERVER_PANEL_VERSION"
VERSION_FILE_NAME = "VERSION"
GITHUB_RELEASES_URL = f"https://github.com/{DEFAULT_UPDATE_REPO}/releases"
REQUEST_TIMEOUT_SECONDS = 8
VERSION_RE = re.compile(r"(\d+(?:\.\d+)*)")


class UpdateCheckError(Exception):
    """Raised when a release update check cannot complete."""


@dataclass(frozen=True)
class VersionSource:
    value: Optional[str]
    source: str


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    browser_download_url: str


@dataclass(frozen=True)
class LatestRelease:
    tag_name: str
    html_url: str
    name: str = ""
    published_at: str = ""
    assets: tuple[ReleaseAsset, ...] = ()


@dataclass(frozen=True)
class UpdateCheckResult:
    current: VersionSource
    latest: LatestRelease
    update_available: bool
    comparable: bool
    message: str


@dataclass(frozen=True)
class UpdateInstallResult:
    version: str
    message: str
    restart_started: bool


def platform_slug() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower().replace("amd64", "x64").replace("x86_64", "x64")
    if system == "darwin":
        system = "macos"
    return f"{system}-{machine}"


def executable_name() -> str:
    return f"{APP_NAME}.exe" if os.name == "nt" else APP_NAME


def expected_archive_name() -> str:
    return f"{APP_NAME}-{platform_slug()}.zip"


def _version_key(version: str) -> Optional[tuple[int, ...]]:
    match = VERSION_RE.search(version.strip())
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def compare_versions(left: str, right: str) -> Optional[int]:
    left_key = _version_key(left)
    right_key = _version_key(right)
    if left_key is None or right_key is None:
        return None

    size = max(len(left_key), len(right_key))
    padded_left = left_key + (0,) * (size - len(left_key))
    padded_right = right_key + (0,) * (size - len(right_key))
    if padded_left < padded_right:
        return -1
    if padded_left > padded_right:
        return 1
    return 0


def version_file_candidates(panel_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().with_name(VERSION_FILE_NAME))
    candidates.append(panel_dir / VERSION_FILE_NAME)

    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def git_describe_version(panel_dir: Path) -> Optional[str]:
    if not (panel_dir / ".git").exists():
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(panel_dir), "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = proc.stdout.strip()
    return value if proc.returncode == 0 and value else None


def current_app_version(panel_dir: Path, *, environ: Optional[Mapping[str, str]] = None) -> VersionSource:
    env = os.environ if environ is None else environ
    env_value = env.get(VERSION_ENV, "").strip()
    if env_value:
        return VersionSource(env_value, VERSION_ENV)

    for path in version_file_candidates(panel_dir):
        try:
            value = path.read_text(encoding="utf-8").splitlines()[0].strip()
        except (FileNotFoundError, IndexError, OSError):
            continue
        if value:
            return VersionSource(value, str(path))

    git_value = git_describe_version(panel_dir)
    if git_value:
        return VersionSource(git_value, "git tag")
    return VersionSource(None, "unknown")


def resolve_update_repo(environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    repo = env.get(UPDATE_REPO_ENV, DEFAULT_UPDATE_REPO).strip().strip("/")
    if repo.count("/") != 1:
        raise UpdateCheckError(f"{UPDATE_REPO_ENV} must be in owner/repo form.")
    return repo


def latest_release_api_url(repo_slug: str) -> str:
    return f"https://api.github.com/repos/{repo_slug}/releases/latest"


def parse_latest_release(payload: Mapping[str, object]) -> LatestRelease:
    tag_name = str(payload.get("tag_name") or "").strip()
    if not tag_name:
        raise UpdateCheckError("GitHub latest release response did not include a tag name.")
    html_url = str(payload.get("html_url") or "").strip() or GITHUB_RELEASES_URL
    assets: list[ReleaseAsset] = []
    raw_assets = payload.get("assets") or []
    if isinstance(raw_assets, list):
        for raw_asset in raw_assets:
            if not isinstance(raw_asset, Mapping):
                continue
            name = str(raw_asset.get("name") or "").strip()
            download_url = str(raw_asset.get("browser_download_url") or "").strip()
            if name and download_url:
                assets.append(ReleaseAsset(name, download_url))
    return LatestRelease(
        tag_name=tag_name,
        html_url=html_url,
        name=str(payload.get("name") or "").strip(),
        published_at=str(payload.get("published_at") or "").strip(),
        assets=tuple(assets),
    )


def fetch_latest_release(
    repo_slug: str,
    *,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
    opener: Optional[Callable[..., object]] = None,
) -> LatestRelease:
    request = urllib.request.Request(
        latest_release_api_url(repo_slug),
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "llama-server-panel-update-check",
        },
    )
    urlopen = opener or urllib.request.urlopen
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise UpdateCheckError(f"GitHub release check failed with HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise UpdateCheckError(f"Could not reach GitHub releases: {reason}") from exc
    except OSError as exc:
        raise UpdateCheckError(f"Could not reach GitHub releases: {exc}") from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise UpdateCheckError(f"GitHub latest release response was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise UpdateCheckError("GitHub latest release response was not a JSON object.")
    return parse_latest_release(payload)


def find_release_asset(release: LatestRelease, asset_name: str) -> Optional[ReleaseAsset]:
    for asset in release.assets:
        if asset.name == asset_name:
            return asset
    return None


def download_url(
    url: str,
    destination: Path,
    *,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
    opener: Optional[Callable[..., object]] = None,
) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "llama-server-panel-update-check"},
    )
    urlopen = opener or urllib.request.urlopen
    try:
        with urlopen(request, timeout=timeout) as response:
            with destination.open("wb") as output:
                shutil.copyfileobj(response, output)
    except urllib.error.HTTPError as exc:
        raise UpdateCheckError(f"Download failed with HTTP {exc.code}: {url}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise UpdateCheckError(f"Could not download update: {reason}") from exc
    except OSError as exc:
        raise UpdateCheckError(f"Could not download update: {exc}") from exc


def parse_checksum_file(text: str, expected_name: str) -> Optional[str]:
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        digest = parts[0].lower()
        filename = parts[-1].lstrip("*")
        if filename == expected_name and re.fullmatch(r"[0-9a-f]{64}", digest):
            return digest
    return None


def verify_archive_checksum(archive_path: Path, checksum_text: str, expected_name: str) -> None:
    expected = parse_checksum_file(checksum_text, expected_name)
    if expected is None:
        raise UpdateCheckError(f"Checksum file did not include a SHA256 entry for {expected_name}.")
    actual = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if actual != expected:
        raise UpdateCheckError(f"Downloaded update checksum did not match {expected_name}.")


def extract_update_archive(archive_path: Path, destination: Path) -> Path:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            destination_root = destination.resolve()
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if target != destination_root and destination_root not in target.parents:
                    raise UpdateCheckError(f"Downloaded update archive contains an unsafe path: {member.filename}")
            archive.extractall(destination)
    except (OSError, zipfile.BadZipFile) as exc:
        raise UpdateCheckError(f"Downloaded update archive could not be extracted: {exc}") from exc

    executable = destination / executable_name()
    if not executable.is_file():
        raise UpdateCheckError(f"Downloaded update archive did not contain {executable_name()}.")
    return executable


def download_update_archive(
    release: LatestRelease,
    destination_dir: Path,
    *,
    opener: Optional[Callable[..., object]] = None,
) -> Path:
    archive_name = expected_archive_name()
    archive_asset = find_release_asset(release, archive_name)
    if archive_asset is None:
        raise UpdateCheckError(f"Latest release does not include {archive_name}. Download manually: {release.html_url}")

    archive_path = destination_dir / archive_name
    download_url(archive_asset.browser_download_url, archive_path, opener=opener)

    checksum_asset = find_release_asset(release, f"{archive_name}.sha256")
    if checksum_asset is not None:
        checksum_path = destination_dir / checksum_asset.name
        download_url(checksum_asset.browser_download_url, checksum_path, opener=opener)
        verify_archive_checksum(archive_path, checksum_path.read_text(encoding="utf-8"), archive_name)
    return archive_path


def _installer_script_contents() -> str:
    names = [executable_name(), VERSION_FILE_NAME, "README.md", ".env.example", "LICENSE", "SECURITY.md"]
    if os.name == "nt":
        copy_commands = "\n".join(
            f'if exist "%SOURCE_DIR%\\{name}" copy /Y "%SOURCE_DIR%\\{name}" "%INSTALL_DIR%\\{name}" >nul'
            for name in names
        )
        return f"""@echo off
setlocal
set "PID=%~1"
set "SOURCE_DIR=%~2"
set "INSTALL_DIR=%~3"
set "EXE_PATH=%~4"
:wait
tasklist /FI "PID eq %PID%" | find "%PID%" >nul
if not errorlevel 1 (
  timeout /t 1 /nobreak >nul
  goto wait
)
{copy_commands}
start "" "%EXE_PATH%"
"""

    copy_commands = "\n".join(
        f'if [ -f "$source_dir/{name}" ]; then cp "$source_dir/{name}" "$install_dir/{name}"; fi' for name in names
    )
    return f"""#!/bin/sh
set -eu
pid="$1"
source_dir="$2"
install_dir="$3"
exe_path="$4"
while kill -0 "$pid" 2>/dev/null; do
  sleep 1
done
{copy_commands}
chmod +x "$install_dir/{executable_name()}"
"$exe_path" >/dev/null 2>&1 &
"""


def start_installer_process(source_dir: Path, install_dir: Path, executable: Path) -> None:
    suffix = ".cmd" if os.name == "nt" else ".sh"
    script_path = Path(tempfile.mkdtemp(prefix="llama-panel-install-")) / f"install-update{suffix}"
    script_path.write_text(_installer_script_contents(), encoding="utf-8")
    if os.name != "nt":
        script_path.chmod(0o700)
        subprocess.Popen(
            [str(script_path), str(os.getpid()), str(source_dir), str(install_dir), str(executable)],
            close_fds=True,
            start_new_session=True,
        )
    else:
        subprocess.Popen(
            ["cmd", "/c", str(script_path), str(os.getpid()), str(source_dir), str(install_dir), str(executable)],
            close_fds=True,
        )


def apply_update(
    result: UpdateCheckResult,
    panel_dir: Path,
    *,
    opener: Optional[Callable[..., object]] = None,
) -> UpdateInstallResult:
    if not result.update_available:
        raise UpdateCheckError("No update is available to install.")
    if not getattr(sys, "frozen", False):
        raise UpdateCheckError(f"Automatic install is only available in the packaged app. Download: {result.latest.html_url}")

    executable = Path(sys.executable).resolve()
    install_dir = executable.parent
    work_dir = Path(tempfile.mkdtemp(prefix="llama-panel-update-"))
    archive_path = download_update_archive(result.latest, work_dir, opener=opener)
    extract_dir = work_dir / "extracted"
    extract_dir.mkdir()
    extract_update_archive(archive_path, extract_dir)
    start_installer_process(extract_dir, install_dir, executable)
    return UpdateInstallResult(
        version=result.latest.tag_name,
        message=f"Update {result.latest.tag_name} is ready. The app will close, install it, and reopen.",
        restart_started=True,
    )


def build_update_result(current: VersionSource, latest: LatestRelease) -> UpdateCheckResult:
    if current.value is None:
        message = f"Latest release is {latest.tag_name}, but the current app version is unknown. Download: {latest.html_url}"
        return UpdateCheckResult(current, latest, update_available=False, comparable=False, message=message)

    comparison = compare_versions(current.value, latest.tag_name)
    if comparison is None:
        message = (
            f"Latest release is {latest.tag_name}, but current version {current.value} could not be compared. "
            f"Download: {latest.html_url}"
        )
        return UpdateCheckResult(current, latest, update_available=False, comparable=False, message=message)

    current_label = f"{current.value} ({current.source})"
    if comparison < 0:
        message = f"Update available: {latest.tag_name} is newer than {current_label}. Download: {latest.html_url}"
        return UpdateCheckResult(current, latest, update_available=True, comparable=True, message=message)
    if comparison > 0:
        message = f"No update available: current version {current_label} is newer than latest release {latest.tag_name}."
        return UpdateCheckResult(current, latest, update_available=False, comparable=True, message=message)

    message = f"No update available: current version {current_label} matches latest release {latest.tag_name}."
    return UpdateCheckResult(current, latest, update_available=False, comparable=True, message=message)


def check_for_updates(
    panel_dir: Path,
    *,
    environ: Optional[Mapping[str, str]] = None,
    opener: Optional[Callable[..., object]] = None,
) -> UpdateCheckResult:
    env = os.environ if environ is None else environ
    repo_slug = resolve_update_repo(env)
    current = current_app_version(panel_dir, environ=env)
    latest = fetch_latest_release(repo_slug, opener=opener)
    return build_update_result(current, latest)
