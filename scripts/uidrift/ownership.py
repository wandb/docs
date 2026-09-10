"""Who should look at this?

Two sources with different strengths. CODEOWNERS gives the accountable team but
is coarse -- `/frontends/app/ @wandb/frontend-reviewers` covers most of the app.
Git authorship names actual humans but says nothing about accountability. Report
both; neither is a substitute for the other.

Both answers are the same for every finding in a run, so both are computed once
per run rather than once per finding. The naive version shelled out `git show`
for CODEOWNERS and one-to-two `git log` invocations per path, which is ~3
subprocesses per finding for data that never changes mid-run. See ADAPTING.md.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import config

# Below this many distinct recent authors the window is too thin to rank, so it
# widens to full history. Measured: the flagship file has 5 authors with 1-2
# commits each over six months, but one clear owner (23 commits) over all time.
MIN_RECENT_AUTHORS = 3

_BOT = re.compile(r"\[bot\]$|\bbot\b|-agent$|^wandbot|devin-ai", re.I)

# Author lines are prefixed so they cannot be confused with a path. \x01 cannot
# appear in a git author name or a filename.
_AUTHOR_MARK = "\x01"

# (root, head, since) -> {path: {author: commits}}. Populated once per run.
_AUTHOR_INDEX: dict[tuple[str, str, Optional[str]], dict[str, dict[str, int]]] = {}
# (root, head) -> parsed CODEOWNERS rules, most general first.
_CODEOWNERS: dict[tuple[str, str], list[tuple[re.Pattern[str], str]]] = {}
# The ref this run reads history and CODEOWNERS from. Set once per run, because
# it has to be the same ref `scan` selected its commit range from -- attributing
# a commit on one ref to the authors of another names the wrong people.
_HEAD: Optional[str] = None


def reset_caches(head: Optional[str] = None) -> None:
    """Drop the per-run caches and set the ref this run reads.

    Tests call this; a cron process is short-lived. `head` is the ref `scan`
    resolved (`--head`, defaulting to `config.SOURCE.default_head`); passing
    None restores the default.
    """
    global _HEAD
    _HEAD = head
    _AUTHOR_INDEX.clear()
    _CODEOWNERS.clear()


def _head() -> str:
    """The ref to read. `reset_caches(head=...)` wins over the configured default."""
    return _HEAD or config.SOURCE.default_head


@dataclass(frozen=True)
class Ownership:
    reviewers: list[str]
    team: Optional[str]
    source: str  # "recent" | "all-time" | "none"


def _build_author_index(core: Path, since: Optional[str]) -> dict[str, dict[str, int]]:
    """One `git log` over the UI roots, yielding every (path, author) pair.

    Scoped to `ui_roots` by pathspec because that is the only place a finding's
    path can come from, and unscoped history over wandb/core is far larger than
    anything this needs.
    """
    cmd = [
        "git", "-C", str(core), "log", _head(),
        f"--format={_AUTHOR_MARK}%an", "--name-only",
    ]
    if since:
        cmd.append(f"--since={since}")
    cmd += ["--", *config.SOURCE.ui_roots]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return {}
    if out.returncode != 0:
        return {}

    index: dict[str, dict[str, int]] = {}
    author = ""
    for line in out.stdout.splitlines():
        if line.startswith(_AUTHOR_MARK):
            author = line[len(_AUTHOR_MARK):].strip()
            continue
        path = line.strip()
        # Cherry-pick and codegen bots are not reviewers.
        if not path or not author or _BOT.search(author):
            continue
        counts = index.setdefault(path, {})
        counts[author] = counts.get(author, 0) + 1
    return index


def _author_index(core: Path, since: Optional[str]) -> dict[str, dict[str, int]]:
    key = (str(core), _head(), since)
    if key not in _AUTHOR_INDEX:
        _AUTHOR_INDEX[key] = _build_author_index(core, since)
    return _AUTHOR_INDEX[key]


def _git_authors(core: Path, path: str, since: Optional[str]) -> list[str]:
    counts = _author_index(core, since).get(path, {})
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

    A trailing `/` is stripped and then ignored, because CODEOWNERS globs are
    gitignore-style: a pattern that names a directory owns everything beneath it
    whether or not it ends in a slash. `/frontends/app/src/weave` and
    `/frontends/app/src/weave/` are the same rule, and wandb/core writes both
    forms.
    """
    anchored = pattern.startswith("/")
    p = pattern.lstrip("/").rstrip("/")

    out: list[str] = []
    i = 0
    while i < len(p):
        if p.startswith("**/", i):
            # `**/` spans ZERO or more directories, so it has to consume the
            # slash too. Emitting `.*` and then the literal `/` requires at
            # least one directory, which makes `/src/**/*ramp*` miss
            # `src/ramp.tsx` -- and a missed pattern is a missed last-match
            # override, which names the wrong team rather than no team.
            out.append("(?:.*/)?")
            i += 3
        elif p.startswith("**", i):
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
    # Deliberately not a conditional on the trailing slash. Both forms have to
    # match everything beneath the directory (see the docstring), so branching
    # here can only produce two spellings of the same regex -- which is why this
    # read as a dead conditional and got "simplified" to a bare `$` for the
    # slash-less form. That silently unmatched `/frontends/app/src/weave`
    # against files inside it, dropping the last-match override and naming the
    # broader team instead. The regression is invisible in the report.
    suffix = "(?:/.*)?$"
    return re.compile(prefix + body + suffix)


def _codeowners_rules(core: Path) -> list[tuple[re.Pattern[str], str]]:
    """Read and compile CODEOWNERS once per run.

    The file is identical for every finding in a scan, so this is read once and
    the globs are compiled once. Rules stay in file order because matching
    depends on it.
    """
    key = (str(core), _head())
    if key in _CODEOWNERS:
        return _CODEOWNERS[key]

    content = None
    for candidate in config.SOURCE.codeowners:
        try:
            out = subprocess.run(
                ["git", "-C", str(core), "show",
                 f"{_head()}:{candidate}"],
                capture_output=True, text=True, timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0:
            content = out.stdout
            break

    rules: list[tuple[re.Pattern[str], str]] = []
    for line in (content or "").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        rules.append((_codeowners_regex(parts[0]), " ".join(parts[1:])))
    _CODEOWNERS[key] = rules
    return rules


def owning_team(path: str, *, core: Optional[Path] = None) -> Optional[str]:
    """The CODEOWNERS entry for a path.

    LAST match wins, not first -- that is the GitHub rule, and wandb/core relies
    on it: `/frontends/app/` is overridden by `/frontends/app/src/weave` and by
    `/frontends/app/**/*ramp**` further down the file.
    """
    root = core or config.SOURCE.path
    winner = None
    for pattern, owners in _codeowners_rules(root):
        if pattern.match(path):
            winner = owners
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
