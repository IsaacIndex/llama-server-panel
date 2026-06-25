#!/usr/bin/env python3
"""Create and push the next release tag."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Sequence


TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
RunCommand = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
SleepFunction = Callable[[float], None]


@dataclass(frozen=True, order=True)
class SemVerTag:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str) -> "SemVerTag | None":
        match = TAG_RE.match(value.strip())
        if match is None:
            return None
        return cls(*(int(part) for part in match.groups()))

    def bump(self, part: str) -> "SemVerTag":
        if part == "major":
            return SemVerTag(self.major + 1, 0, 0)
        if part == "minor":
            return SemVerTag(self.major, self.minor + 1, 0)
        if part == "patch":
            return SemVerTag(self.major, self.minor, self.patch + 1)
        raise ValueError(f"Unsupported bump part: {part}")

    def format(self) -> str:
        return f"v{self.major}.{self.minor}.{self.patch}"


def run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def validate_tag(value: str) -> str:
    tag = value.strip()
    if TAG_RE.match(tag) is None:
        raise ValueError("Release tag must use vMAJOR.MINOR.PATCH, for example v0.7.0.")
    return tag


def latest_release_tag(tags: Sequence[str]) -> SemVerTag:
    parsed = [tag for tag in (SemVerTag.parse(value) for value in tags) if tag is not None]
    if not parsed:
        return SemVerTag(0, 0, 0)
    return max(parsed)


def next_release_tag(tags: Sequence[str], bump: str) -> str:
    return latest_release_tag(tags).bump(bump).format()


def git_output(command: Sequence[str], runner: RunCommand = run) -> str:
    proc = runner(command)
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or f"{' '.join(command)} failed").strip()
        raise RuntimeError(message)
    return proc.stdout


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def ensure_clean_worktree(runner: RunCommand = run) -> None:
    status = git_output(["git", "status", "--short"], runner)
    print(status, end="")
    if status.strip():
        raise RuntimeError("Working tree is not clean; commit or stash changes before publishing.")


def resolve_release_tag(bump: str, runner: RunCommand = run, explicit_tag: str | None = None) -> str:
    if explicit_tag:
        return validate_tag(explicit_tag)
    tags_output = git_output(["git", "tag", "--list", "v*"], runner)
    return next_release_tag(tags_output.splitlines(), bump)


def current_head_sha(runner: RunCommand = run) -> str:
    return git_output(["git", "rev-parse", "HEAD"], runner).strip()


def latest_ci_run(workflow: str, sha: str, runner: RunCommand = run) -> dict[str, object]:
    output = git_output(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            workflow,
            "--commit",
            sha,
            "--json",
            "databaseId,status,conclusion,url",
            "--limit",
            "10",
        ],
        runner,
    )
    try:
        runs = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse GitHub Actions response: {exc}") from exc
    if not isinstance(runs, list):
        raise RuntimeError("GitHub Actions response was not a run list.")
    if not runs:
        raise RuntimeError(f"No {workflow} workflow run found for {sha[:12]}; push the commit first or use --no-wait-for-ci.")
    run_info = runs[0]
    if not isinstance(run_info, dict):
        raise RuntimeError("GitHub Actions response contained an invalid run record.")
    return run_info


def wait_for_ci(
    workflow: str,
    *,
    timeout_seconds: int,
    poll_interval_seconds: int,
    runner: RunCommand = run,
    sleeper: SleepFunction = time.sleep,
) -> None:
    sha = current_head_sha(runner)
    deadline = time.monotonic() + timeout_seconds
    print(f"Waiting for {workflow} workflow on {sha[:12]} before publishing...")
    while True:
        run_info = latest_ci_run(workflow, sha, runner)
        status = str(run_info.get("status") or "unknown")
        conclusion = str(run_info.get("conclusion") or "")
        url = str(run_info.get("url") or "")
        suffix = f" ({url})" if url else ""
        print(f"{workflow} status: {status}{f' / {conclusion}' if conclusion else ''}{suffix}")
        if status == "completed":
            if conclusion == "success":
                return
            raise RuntimeError(f"{workflow} completed with conclusion {conclusion or 'unknown'}; release tag was not pushed.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"Timed out waiting for {workflow} to complete; release tag was not pushed.")
        sleeper(min(poll_interval_seconds, remaining))


def publish_release(tag: str, runner: RunCommand = run, *, dry_run: bool = False) -> None:
    print(f"Release tag: {tag}")
    if dry_run:
        print(f"Dry run: git tag {tag}")
        print(f"Dry run: git push origin {tag}")
        return
    git_output(["git", "tag", tag], runner)
    git_output(["git", "push", "origin", tag], runner)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and push the next Llama Server Panel release tag")
    parser.add_argument(
        "--bump",
        choices=("major", "minor", "patch"),
        default=os.environ.get("RELEASE_BUMP", "minor"),
        help="version part to bump when RELEASE_TAG is not set (default: minor)",
    )
    parser.add_argument(
        "--tag",
        default=os.environ.get("RELEASE_TAG"),
        help="explicit release tag, overriding dynamic tag calculation",
    )
    parser.add_argument(
        "--wait-for-ci",
        dest="wait_for_ci",
        action="store_true",
        help="wait for the current commit's GitHub Actions CI run before pushing the release tag",
    )
    parser.add_argument(
        "--no-wait-for-ci",
        dest="wait_for_ci",
        action="store_false",
        help="push the release tag without checking GitHub Actions CI",
    )
    parser.set_defaults(wait_for_ci=env_bool("RELEASE_WAIT_FOR_CI", True))
    parser.add_argument(
        "--ci-workflow",
        default=os.environ.get("RELEASE_CI_WORKFLOW", "CI"),
        help="GitHub Actions workflow name to wait for before publishing (default: CI)",
    )
    parser.add_argument(
        "--ci-timeout-seconds",
        type=int,
        default=int(os.environ.get("RELEASE_CI_TIMEOUT_SECONDS", "3600")),
        help="maximum seconds to wait for CI before aborting (default: 3600)",
    )
    parser.add_argument(
        "--ci-poll-interval-seconds",
        type=int,
        default=int(os.environ.get("RELEASE_CI_POLL_INTERVAL_SECONDS", "30")),
        help="seconds between CI status checks (default: 30)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the commands without creating or pushing a tag")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        ensure_clean_worktree()
        tag = resolve_release_tag(args.bump, explicit_tag=args.tag)
        if args.wait_for_ci and not args.dry_run:
            wait_for_ci(
                args.ci_workflow,
                timeout_seconds=args.ci_timeout_seconds,
                poll_interval_seconds=args.ci_poll_interval_seconds,
            )
        publish_release(tag, dry_run=args.dry_run)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
