"""Render findings as the markdown table.

This is the v1 deliverable the working group agreed on: a table, not Jira
tickets. Jira metadata rides along in a column so the day someone flips filing
on, nothing has to be recomputed.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable, Sequence

from . import config
from .finding import (
    KIND_MOVED,
    KIND_NEW_SETTING,
    TRIAGE_AGENT,
    TRIAGE_HUMAN,
    TRIAGE_PAIR,
    Finding,
)

_LANE_ORDER = (TRIAGE_AGENT, TRIAGE_PAIR, TRIAGE_HUMAN)
_LANE_TITLE = {
    TRIAGE_AGENT: "Agent can fix unattended",
    TRIAGE_PAIR: "Needs a writer's call first",
    TRIAGE_HUMAN: "Needs a human to write",
}
_REPO_URL = "https://github.com/wandb/core/commit/"


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _commit_links(f: Finding) -> str:
    return " ".join(
        f"[`{c.sha[:7]}`]({_REPO_URL}{c.sha})" for c in f.commits
    )


def _change_cell(f: Finding) -> str:
    if f.kind == KIND_NEW_SETTING:
        return f"new setting **{_escape(f.new_string)}**"
    if f.kind == KIND_MOVED:
        return f"`{_escape(f.old_string)}` moved"
    if not f.old_string:
        return f"new **{_escape(f.new_string)}**"
    if not f.new_string:
        return f"`{_escape(f.old_string)}` removed"
    return f"`{_escape(f.old_string)}` → `{_escape(f.new_string)}`"


def _docs_cell(f: Finding) -> str:
    pages = (f.docs or {}).get("pages") or []
    if not pages:
        return "*no page covers this surface*"
    shown = [f"`{p['page']}:{p['line']}`" + (" **(release notes)**" if p["immutable"] else "")
             for p in pages[:3]]
    if len(pages) > 3:
        shown.append(f"+{len(pages) - 3} more")
    tr = (f.docs or {}).get("translations_affected") or {}
    if tr:
        shown.append("mirrors: " + ", ".join(f"{n}×{loc}" for loc, n in sorted(tr.items())))
    return "<br>".join(shown)


def _notes_cell(f: Finding) -> str:
    bits = [_commit_links(f)]
    if f.gate:
        state = "not yet visible" if f.not_yet_visible else "gated (already ramped)"
        bits.append(f"gate `{f.gate['name']}` — {state}")
    if "case_only_change" in f.signals:
        bits.append("case-only change")
    if "testid_unchanged" in f.signals:
        bits.append("`data-test` unchanged")
    if "url_stable" in f.signals:
        bits.append("URL stable")
    if "url_changed" in f.signals:
        bits.append("**URL changed**")
    if not f.settled:
        bits.append(f"**unsettled** (<{config.SETTLED_DAYS}d)")
    return "<br>".join(bits)


def _row(f: Finding) -> str:
    reviewers = ", ".join(f.reviewers) if f.reviewers else "—"
    team = f.owning_team or "—"
    return "| " + " | ".join([
        f"`{f.id}`",
        _escape(f.surface + (f" (+{len(f.surfaces) - 1} more)" if len(f.surfaces) > 1 else "")),
        _change_cell(f),
        _escape(f.triage_reason),
        _docs_cell(f),
        f"{_escape(reviewers)}<br>{_escape(team)}",
        _notes_cell(f),
    ]) + " |"


def render(
    findings: Sequence[Finding],
    *,
    scanned_range: str,
    today: date,
    commits: int,
    ui_commits: int,
    candidates: int,
    docs_pages: int,
    gaps: int = 0,
) -> str:
    lines: list[str] = []
    a = lines.append

    a("# UI label drift — wandb/core → docs")
    a("")
    a(f"Scanned `{scanned_range}` on {today.isoformat()}.")
    a(f"{commits} commits → {ui_commits} touching UI → {candidates} stage-1 candidates "
      f"→ **{len(findings)} findings**, against {docs_pages} indexed doc pages.")
    a("")

    if not findings:
        a("No drift found in this window.")
        a("")
        a("An empty table is a real result, not a broken run: it means every UI label")
        a("change in the window is either already reflected in docs or touches no")
        a("documented surface.")
        return "\n".join(lines) + "\n"

    by_lane: dict[str, list[Finding]] = {lane: [] for lane in _LANE_ORDER}
    for f in findings:
        by_lane.setdefault(f.triage, []).append(f)

    a("| Lane | Findings | What it means |")
    a("|---|---|---|")
    a(f"| **agent** | {len(by_lane[TRIAGE_AGENT])} | mechanical; safe to apply unattended |")
    a(f"| **pair** | {len(by_lane[TRIAGE_PAIR])} | a writer scopes it, then an agent applies |")
    a(f"| **human** | {len(by_lane[TRIAGE_HUMAN])} | prose has to be written |")
    a("")

    for lane in _LANE_ORDER:
        rows = by_lane[lane]
        if not rows:
            continue
        a(f"## {_LANE_TITLE[lane]} ({len(rows)})")
        a("")
        a("| ID | Surface | Change | Why this lane | Docs | Reviewers / team | Evidence |")
        a("|---|---|---|---|---|---|---|")
        for f in sorted(rows, key=lambda x: (-len((x.docs or {}).get("pages") or []), x.surface)):
            a(_row(f))
        a("")

    gated = [f for f in findings if f.not_yet_visible]
    if gated:
        a("### Advance warning")
        a("")
        a(f"{len(gated)} finding(s) are behind a gate that this commit added or newly")
        a("wrapped, so users cannot see the change yet. Draft the docs while the change")
        a("is fresh and hold the PR — this is lead time, not noise.")
        a("")

    if gaps:
        a("### Undocumented surfaces")
        a("")
        a(f"{gaps} changed label(s) match no documentation at all, so nothing in the")
        a("docs became wrong and they are not listed above. Most are incidental copy")
        a("— empty states, spinner labels, status pills. The count is here because a")
        a("sustained rise in it is worth noticing, not because each one needs a row.")
        a("")

    return "\n".join(lines) + "\n"
