# Vendored from wandb/release-note-genie:scripts/commit_text.py at 5846dd1
# Do not edit here. Upstream changes must be re-vendored deliberately.
# See scripts/uidrift/ADAPTING.md for why this is a copy and not an import.

#!/usr/bin/env python3
"""Normalize a commit message down to the prose that actually describes the change.

Scoring and flag detection used to run keyword scans over the raw commit message. In
wandb/core that message is dominated by PR-template boilerplate, and the boilerplate was
moving scores more than the change itself:

  - Every PR has a ``## Testing`` section, so the low-signal keyword "test" fired on
    *every* commit — a universal -1 dressed up as signal.
  - Go identifiers inside code spans leaked into prose matching: ``internalKeyInfo``
    matched the low-signal keyword "internal" and cost a real perf fix a point.
  - Cherry-pick preambles name the target branch (``server-release-0.83.x``), which
    supplied the "release" half of a bogus GA boost, and their conflict-resolution
    narrative describes *the port* rather than the change.
  - Co-author trailers and Devin session URLs contribute nothing but match on substrings.

``clean_for_scoring`` strips all of that while always preserving the subject line, which is
the highest-signal text in the message.
"""

from __future__ import annotations

import re

# Fenced code blocks and inline code spans. Identifiers are implementation detail, not a
# description of user-visible behavior, and they are the main source of false keyword hits.
_FENCED_RE = re.compile(r"```.*?```", re.S)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")

# Markdown headings whose contents never describe user-visible behavior.
_DROP_SECTION_TITLES = (
    "testing",
    "test plan",
    "how to test",
    "checklist",
    "conflict resolution",
    "screenshots",
)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*#*\s*$")

# Commit trailers and generated links.
_TRAILER_RE = re.compile(
    r"^\s*(co-authored-by|signed-off-by|reviewed-by|acked-by|tested-by|cc|"
    r"link to devin session|devin session|generated with|reported-by|fixes|closes)\s*:",
    re.I,
)
_URL_RE = re.compile(r"https?://\S+")

# Markdown task-list rows ("- [x] Added unit tests").
_CHECKBOX_RE = re.compile(r"^\s*[-*]\s*\[[ xX]\]\s*")

# Cherry-pick bookkeeping: names the target release branch and describes the port.
_CHERRY_LINE_RE = re.compile(r"cherry[- ]pick", re.I)
_RELEASE_BRANCH_RE = re.compile(r"\b(?:server-)?release[-/][\w.\-]*\b", re.I)


def _is_dropped_heading(text: str) -> bool:
    lowered = text.strip().lower().rstrip(":").strip()
    return any(lowered.startswith(title) for title in _DROP_SECTION_TITLES)


def clean_for_scoring(message: str) -> str:
    """Return subject + description prose, with boilerplate and code identifiers removed.

    The result is intended for keyword/regex scanning only. It is lossy by design and must
    never be shown to a human or used as draft text.
    """
    if not message:
        return ""

    raw_lines = message.splitlines()
    subject = raw_lines[0] if raw_lines else ""
    body = "\n".join(raw_lines[1:])

    body = _FENCED_RE.sub(" ", body)
    body = _INLINE_CODE_RE.sub(" ", body)

    kept: list[str] = []
    skipping = False
    for line in body.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            # A new heading always ends any section we were skipping.
            skipping = _is_dropped_heading(heading.group(1))
            continue
        if skipping:
            continue
        if _TRAILER_RE.match(line) or _CHERRY_LINE_RE.search(line):
            continue
        line = _CHECKBOX_RE.sub("", line)
        line = _URL_RE.sub(" ", line)
        kept.append(line)

    cleaned = f"{subject}\n" + "\n".join(kept)
    # Release-branch names survive outside cherry-pick lines too (e.g. "onto release-0.83.x").
    cleaned = _RELEASE_BRANCH_RE.sub(" ", cleaned)
    # Collapse the blank lines left behind, but keep newlines: several regexes are
    # deliberately sentence-local and rely on line breaks as boundaries.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


def unwrap_paragraphs(text: str) -> str:
    """Join hard-wrapped lines within a paragraph into single logical lines.

    PR bodies wrap at ~72 columns, which splits phrases the sentence-local flag regexes
    need to see whole ("gated by the new Statsig gate\\n`some_gate_name`").
    """
    if not text:
        return ""
    paragraphs = re.split(r"\n\s*\n", text)
    joined = [re.sub(r"\s*\n\s*", " ", p).strip() for p in paragraphs]
    return "\n".join(p for p in joined if p)
