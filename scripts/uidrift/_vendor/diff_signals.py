# Vendored from wandb/release-note-genie:scripts/diff_signals.py at 5846dd1
# Do not edit here. Upstream changes must be re-vendored deliberately.
# See scripts/uidrift/ADAPTING.md for why this is a copy and not an import.

#!/usr/bin/env python3
"""Signals that can only be read from a commit's diff, not its message or file list.

Classification runs on the commit message plus the changed-path list. Several judgments have
turned out to need more than that, and the honest response to "the deciding evidence isn't
visible" has been to route to review. This module narrows that set where the diff *can* decide
deterministically — no model, no heuristics over prose.

Currently one signal, because it is the one where the diff genuinely resolves the ambiguity:

  graphql_contract_change
      Editing a published ``.graphql`` file covers two unrelated things, and the path cannot
      tell them apart:

        contract change    ``+  limit: Int`` / ``+  pattern: String`` on ``historyKeys`` —
                           new arguments clients can send. Note-worthy.
        server annotation  ``-  id: ID!`` / ``+  id: ID! @skipFieldTrace`` across 34 fields —
                           cuts ~129M Datadog spans/day. Clients observe nothing.

      The diff separates them exactly: strip the internal directives and a directive-only
      change has identical removed and added line sets, while a contract change does not.

Deliberately NOT here: whether a changed default is note-worthy. That looked like a diff
question and isn't. ``ConnMaxLifetime`` carries a ``schema:"conn_max_lifetime"`` tag, so the
diff says "operator-settable" — yet the parameter appears nowhere in W&B's public docs, so no
admin can act on it and reviewers rejected the note. The deciding fact lives in the docs, not
the diff, so changed defaults still go to a human. A docs cross-reference (does the parameter
name appear in the published docs?) would be the capability that resolves it.
"""

from __future__ import annotations

import re
from typing import Optional

# Directives that are server-side implementation detail: adding or removing one changes
# nothing a client can observe. Anything NOT listed here is treated as contract-relevant, so an
# unfamiliar directive fails toward "contract change" (a human looks) rather than being
# silently ignored. @deprecated is intentionally absent — deprecating a field IS user-facing.
INTERNAL_DIRECTIVES = frozenset({
    "skipFieldTrace",   # tracing suppression (wandb/core)
    "goField",          # gqlgen codegen binding
    "goModel",
    "goTag",
    "goExtraField",
    "goEnum",
})

_DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
_HUNK_RE = re.compile(r"^@@")
_DIRECTIVE_RE = re.compile(r"@(\w+)(\s*\([^)]*\))?")


def _iter_file_diffs(diff: str):
    """Yield ``(path, body)`` for each file section of a unified diff."""
    matches = list(_DIFF_FILE_RE.finditer(diff or ""))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(diff)
        # Prefer the b/ path (post-image); falls back to a/ for deletions.
        yield m.group(2) or m.group(1), diff[m.end():end]


def graphql_files_in_diff(diff: str) -> list[str]:
    return [p for p, _ in _iter_file_diffs(diff) if p.lower().endswith(".graphql")]


def _strip_internal_directives(line: str) -> str:
    """Remove only the directives listed in INTERNAL_DIRECTIVES."""
    def sub(m: re.Match) -> str:
        return "" if m.group(1) in INTERNAL_DIRECTIVES else m.group(0)
    return _DIRECTIVE_RE.sub(sub, line)


def _normalize(line: str) -> Optional[str]:
    """Normalize a schema line for comparison, or None if it carries no contract meaning."""
    body = line[1:]  # drop the +/- marker
    body = _strip_internal_directives(body)
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return None
    if body.startswith("#"):
        return None  # a comment-only edit is not a contract change
    return body


def graphql_contract_change(diff: str) -> Optional[bool]:
    """Did a diff change a published GraphQL contract?

    Returns True when at least one ``.graphql`` file gained or lost real schema content,
    False when every ``.graphql`` change is internal-directive or comment noise, and None
    when the diff contains no ``.graphql`` files at all (nothing to say).
    """
    if not diff:
        return None

    saw_graphql = False
    for path, body in _iter_file_diffs(diff):
        if not path.lower().endswith(".graphql"):
            continue
        saw_graphql = True

        removed: list[str] = []
        added: list[str] = []
        for raw in body.splitlines():
            if raw.startswith("---") or raw.startswith("+++") or _HUNK_RE.match(raw):
                continue
            if raw.startswith("-"):
                n = _normalize(raw)
                if n is not None:
                    removed.append(n)
            elif raw.startswith("+"):
                n = _normalize(raw)
                if n is not None:
                    added.append(n)

        # Identical multisets => every surviving difference was an internal directive or a
        # comment, so no client-visible schema content moved.
        if sorted(removed) != sorted(added):
            return True

    return False if saw_graphql else None
