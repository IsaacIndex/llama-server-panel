#!/usr/bin/env python3
"""Create and push the next release tag."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Sequence


TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
RunCommand = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


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
    parser.add_argument("--dry-run", action="store_true", help="print the commands without creating or pushing a tag")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        ensure_clean_worktree()
        tag = resolve_release_tag(args.bump, explicit_tag=args.tag)
        publish_release(tag, dry_run=args.dry_run)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
