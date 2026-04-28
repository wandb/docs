#!/usr/bin/env python3
"""Compare latest GitHub release tags to the newest <Update> block in docs release notes.

Exits with code 1 when the docs repo is missing a release that exists upstream, so CI
can flag drift for a human or bot to open a docs PR.

Usage:
  python3 scripts/check_sdk_release_notes_drift.py
  python3 scripts/check_sdk_release_notes_drift.py --json
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class DriftResult:
    repo: str
    docs_path: str
    latest_github: str | None
    latest_docs: str | None
    drift: bool

    def message(self) -> str:
        if self.latest_github is None:
            return f"{self.repo}: could not fetch latest release"
        if self.latest_docs is None:
            return f"{self.repo}: could not parse version from {self.docs_path}"
        if self.drift:
            return (
                f"{self.repo}: docs newest is {self.latest_docs} but GitHub newest is "
                f"{self.latest_github}. Update {self.docs_path} or adjust the parser."
            )
        return f"{self.repo}: OK (docs {self.latest_docs} matches or exceeds GitHub {self.latest_github})"


def _fetch_latest_tag(owner: str, repo: str) -> str | None:
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "wandb-docs-check-sdk-release-notes-drift",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except (OSError, ValueError):
        return None
    tag = data.get("tag_name")
    return str(tag) if tag else None


def _normalize_version(tag: str) -> tuple[int, ...]:
    """Turn 'v0.52.37' or '0.52.37' into (0, 52, 37)."""
    t = tag.strip()
    if t.startswith("v"):
        t = t[1:]
    parts: list[int] = []
    for piece in t.split("."):
        digits = re.sub(r"[^0-9].*", "", piece)
        if digits == "":
            continue
        parts.append(int(digits))
    return tuple(parts)


def _first_update_label(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    m = re.search(r'<Update\s+label="([^"]+)"', text)
    return m.group(1) if m else None


def _compare(docs_version: str, gh_tag: str) -> bool:
    """Return True if GitHub has a strictly newer semver than docs (drift)."""
    dv = _normalize_version(docs_version)
    gv = _normalize_version(gh_tag)
    if not gv:
        return False
    if not dv:
        return True
    return gv > dv


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of human text",
    )
    args = parser.parse_args()

    checks = [
        ("wandb", "wandb", "release-notes/sdk-releases.mdx"),
        ("wandb", "weave", "release-notes/weave-sdk-releases.mdx"),
    ]
    results: list[DriftResult] = []
    for owner, repo, rel_path in checks:
        gh = _fetch_latest_tag(owner, repo)
        doc = _first_update_label(rel_path)
        drift = bool(gh and doc and _compare(doc, gh))
        results.append(
            DriftResult(
                repo=f"{owner}/{repo}",
                docs_path=rel_path,
                latest_github=gh,
                latest_docs=doc,
                drift=drift,
            )
        )

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "repo": r.repo,
                        "docs_path": r.docs_path,
                        "latest_github": r.latest_github,
                        "latest_docs": r.latest_docs,
                        "drift": r.drift,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
    else:
        for r in results:
            print(r.message())

    if any(r.drift for r in results):
        return 1
    if any(r.latest_github is None or r.latest_docs is None for r in results):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
