#!/usr/bin/env python3
"""GitHub release update checks for Llama Server Panel."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional


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
class LatestRelease:
    tag_name: str
    html_url: str
    name: str = ""
    published_at: str = ""


@dataclass(frozen=True)
class UpdateCheckResult:
    current: VersionSource
    latest: LatestRelease
    update_available: bool
    comparable: bool
    message: str


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
    return LatestRelease(
        tag_name=tag_name,
        html_url=html_url,
        name=str(payload.get("name") or "").strip(),
        published_at=str(payload.get("published_at") or "").strip(),
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
