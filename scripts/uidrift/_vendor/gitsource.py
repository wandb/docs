# Vendored from wandb/release-note-genie:scripts/rolling/gitsource.py at 5846dd1
# Do not edit here. Upstream changes must be re-vendored deliberately.
# See scripts/uidrift/ADAPTING.md for why this is a copy and not an import.

#!/usr/bin/env python3
"""Read commit deltas from a local wandb/core clone.

The rolling watcher analyzes the delta between the last SHA it saw and master. In CI the
runner checks out wandb/core; locally the clone lives at ~/core. Reading history from git
(rather than the GitHub compare API) means the watcher needs no token and is fast for the
incremental delta.

``iter_commits`` returns commit dicts shaped like the GitHub API objects the existing
scoring code expects (``commit['commit']['message']``, ``commit['files']``, ...), so
score_commit_impact / extract_pr_number can be reused unchanged.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

_REC_SEP = "\x1e"
_FIELD_SEP = "\x1f"


def _git(core: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(core), *args], capture_output=True, text=True)


def resolve_sha(core: Path, ref: str) -> Optional[str]:
    """Resolve a ref to a concrete SHA, trying a few common fallbacks."""
    for candidate in (ref, f"origin/{ref}"):
        r = _git(core, "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}")
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    return None


def ref_exists(core: Path, ref: str) -> bool:
    return resolve_sha(core, ref) is not None


def commit_diff(core: Path, sha: str, *pathspec: str) -> Optional[str]:
    """Return a commit's unified diff, optionally limited to a pathspec.

    Used for signals that need the diff's content rather than its file list (see
    scripts/diff_signals.py). Always pass a pathspec when only certain files matter: some
    wandb/core commits touch hundreds of files and the full patch is large enough to be worth
    not materializing. Returns None when the commit or repo is unavailable, so callers degrade
    to message/path-only classification instead of failing.
    """
    args = ["show", "--format=", "--no-color", "-M", sha]
    if pathspec:
        args += ["--", *pathspec]
    r = _git(core, *args)
    if r.returncode != 0:
        return None
    return r.stdout or None


def is_ancestor(core: Path, sha: str, ref: str) -> Optional[bool]:
    """True if sha is an ancestor of ref; None if ref cannot be resolved."""
    if not ref_exists(core, ref):
        return None
    target = resolve_sha(core, ref)
    r = _git(core, "merge-base", "--is-ancestor", sha, target or ref)
    if r.returncode == 0:
        return True
    if r.returncode == 1:
        return False
    return None


def iter_commits(
    core: Path,
    base: str,
    head: str,
    *,
    owner_repo: str = "wandb/core",
    limit: Optional[int] = None,
    include_merges: bool = False,
) -> list[dict]:
    """Return API-shaped commit dicts for ``base..head`` (commits in head not in base)."""
    fmt = _REC_SEP + "%H" + _FIELD_SEP + "%an" + _FIELD_SEP + "%aI" + _FIELD_SEP + "%B" + _FIELD_SEP
    args = ["log", f"--format={fmt}", "--numstat", "--date=iso-strict"]
    if not include_merges:
        args.append("--no-merges")
    args.append(f"{base}..{head}")
    r = _git(core, *args)
    if r.returncode != 0:
        raise RuntimeError(f"git log {base}..{head} failed: {r.stderr.strip()}")

    commits: list[dict] = []
    chunks = r.stdout.split(_REC_SEP)
    for chunk in chunks:
        if not chunk.strip():
            continue
        parts = chunk.split(_FIELD_SEP)
        if len(parts) < 5:
            continue
        sha, author, date_iso, body, numstat_block = parts[0], parts[1], parts[2], parts[3], parts[4]
        sha = sha.strip()
        if not sha:
            continue
        paths: list[str] = []
        for line in numstat_block.splitlines():
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) == 3 and cols[2]:
                paths.append(cols[2])
        commit = {
            "sha": sha,
            "commit": {
                "message": body.strip("\n"),
                "author": {"name": author, "date": date_iso},
            },
            "html_url": f"https://github.com/{owner_repo}/commit/{sha}",
            "files": [{"filename": p} for p in paths],
        }
        commits.append(commit)
        if limit and len(commits) >= limit:
            break
    return commits
