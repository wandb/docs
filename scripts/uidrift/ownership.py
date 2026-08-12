"""Who should look at this?

Two sources with different strengths. CODEOWNERS gives the accountable team but
is coarse -- `/frontends/app/ @wandb/frontend-reviewers` covers most of the app.
Git authorship names actual humans but says nothing about accountability. Report
both; neither is a substitute for the other.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from . import config

# Below this many distinct recent authors the window is too thin to rank, so it
# widens to full history. Measured: the flagship file has 5 authors with 1-2
# commits each over six months, but one clear owner (23 commits) over all time.
MIN_RECENT_AUTHORS = 3

_BOT = re.compile(r"\[bot\]$|\bbot\b|-agent$|^wandbot|devin-ai", re.I)


@dataclass(frozen=True)
class Ownership:
    reviewers: list[str]
    team: Optional[str]
    source: str  # "recent" | "all-time" | "none"


def _git_authors(core: Path, path: str, since: Optional[str]) -> list[str]:
    cmd = ["git", "-C", str(core), "log", config.SOURCE.default_head, "--format=%an"]
    if since:
        cmd.append(f"--since={since}")
    cmd += ["--", path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []

    counts: dict[str, int] = {}
    for name in out.stdout.splitlines():
        name = name.strip()
        # Cherry-pick and codegen bots are not reviewers.
        if not name or _BOT.search(name):
            continue
        counts[name] = counts.get(name, 0) + 1
    return [n for n, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def suggest_reviewers(
    path: str,
    *,
    core: Optional[Path] = None,
    since: str = "6 months ago",
    top: int = 3,
) -> tuple[list[str], str]:
    """Rank humans who have touched this file, most commits first."""
    root = core or config.SOURCE.path
    recent = _git_authors(root, path, since)
    if len(recent) >= MIN_RECENT_AUTHORS:
        return recent[:top], "recent"
    # Too few recent commits to rank meaningfully -- widen rather than report a
    # single drive-by contributor as the owner.
    all_time = _git_authors(root, path, None)
    if all_time:
        return all_time[:top], "all-time"
    return recent[:top], "none" if not recent else "recent"


def _codeowners_regex(pattern: str) -> re.Pattern[str]:
    """Translate a CODEOWNERS glob into a regex.

    Supports the subset that appears in wandb/core: a leading `/` anchor,
    `**` across segments, `*` within a segment, and a trailing `/` for
    directories.
    """
    anchored = pattern.startswith("/")
    p = pattern.lstrip("/")
    directory = p.endswith("/")
    p = p.rstrip("/")

    out: list[str] = []
    i = 0
    while i < len(p):
        if p.startswith("**", i):
            out.append(".*")
            i += 2
        elif p[i] == "*":
            out.append("[^/]*")
            i += 1
        else:
            out.append(re.escape(p[i]))
            i += 1

    body = "".join(out)
    prefix = "^" if anchored else "^(?:.*/)?"
    suffix = "(?:/.*)?$" if directory else "(?:/.*)?$"
    return re.compile(prefix + body + suffix)


def owning_team(path: str, *, core: Optional[Path] = None) -> Optional[str]:
    """The CODEOWNERS entry for a path.

    LAST match wins, not first -- that is the GitHub rule, and wandb/core relies
    on it: `/frontends/app/` is overridden by `/frontends/app/src/weave` and by
    `/frontends/app/**/*ramp**` further down the file.
    """
    root = core or config.SOURCE.path
    content = None
    for candidate in config.SOURCE.codeowners:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "show",
                 f"{config.SOURCE.default_head}:{candidate}"],
                capture_output=True, text=True, timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0:
            content = out.stdout
            break
    if content is None:
        return None

    winner = None
    for line in content.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        if _codeowners_regex(parts[0]).match(path):
            winner = " ".join(parts[1:])
    return winner


def resolve(
    path: str, *, core: Optional[Path] = None, commit_author: str = ""
) -> Ownership:
    """Reviewers and owning team for one path."""
    reviewers, source = suggest_reviewers(path, core=core)
    if commit_author and not _BOT.search(commit_author):
        # The person who made the change is the most relevant reviewer, and a
        # cherry-pick bot never is.
        reviewers = [commit_author] + [r for r in reviewers if r != commit_author]
    return Ownership(reviewers[:3], owning_team(path, core=core), source)
