"""Render findings as the markdown table.

This is the v1 deliverable the working group agreed on: a table, not Jira
tickets. Jira metadata rides along in a column so the day someone flips filing
on, nothing has to be recomputed.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

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
# Derived, not hard-coded: config.SOURCE is the one place that names the watched
# repo, and a report that links to a repo the scan did not read is worse than no
# link at all.
_REPO_URL = f"https://github.com/{config.SOURCE.owner_repo}/commit/"


def _gaps_section(gaps: int) -> list[str]:
    """The undocumented-surface count.

    Shared by both paths on purpose. A run whose findings are all gaps used to
    print "No drift to act on" and then omit the one number that explains why,
    which reads as "nothing happened" when what happened is that every changed
    label was undocumented.
    """
    if not gaps:
        return []
    return [
        "### Undocumented surfaces",
        "",
        f"{gaps} changed label(s) match no documentation at all, so nothing in the",
        "docs became wrong and they are not listed above. Most are incidental copy",
        "— empty states, spinner labels, status pills. The count is here because a",
        "sustained rise in it is worth noticing, not because each one needs a row.",
        "",
    ]


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


def _decision_bits(f: Finding) -> list[str]:
    """Whatever a human already said about this finding."""
    bits: list[str] = []
    if any(s.startswith("reopened:") for s in f.signals):
        prior = next(s.split(":", 1)[1] for s in f.signals if s.startswith("reopened:"))
        bits.append(f"**reopened** — was `{prior}`, docs evidence has grown since")
    if "marked_fixed_still_detected" in f.signals:
        bits.append("**marked fixed but still detected**")
    elif f.status and not any(s.startswith("reopened:") for s in f.signals):
        bits.append(f"status `{f.status}`")
    where = []
    if f.jira_key:
        where.append(f.jira_key)
    if f.docs_pr:
        where.append(f"docs#{f.docs_pr}")
    if where:
        bits.append(" / ".join(where))
    if f.assignee:
        bits.append(f"assigned {_escape(f.assignee)}")
    return bits


def _notes_cell(f: Finding) -> str:
    bits = [_commit_links(f)]
    bits.extend(_decision_bits(f))
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
    if len(f.commits) > 1:
        bits.append(f"changed {len(f.commits)}× since {f.first_seen_date[:10]}")
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


def _ledger_sections(
    suppressed: Sequence[Finding],
    reopened: Sequence[Finding],
    unresolved: Sequence[Finding],
    orphans: Sequence[str],
    reverted: Sequence[Finding],
) -> list[str]:
    """What the stored decisions did to this run.

    Suppression is always accounted for in the report. A detector that quietly
    drops rows is one nobody can audit, and the count is the only way a reader
    can tell "no drift" from "all drift already dismissed".
    """
    out: list[str] = []
    a = out.append

    if reopened:
        a(f"### Reopened decisions ({len(reopened)})")
        a("")
        a("These were decided once, but the docs evidence has grown since — a page")
        a("now mentions the string, or a new editable occurrence appeared. The prior")
        a("decision is shown in the table above rather than applied.")
        a("")

    if unresolved:
        a(f"### Marked fixed, still detected ({len(unresolved)})")
        a("")
        a("A `fixed` finding normally disappears on its own: the next scan looks for")
        a("the old string in docs and does not find it. These are still detected, so")
        a("the fix did not land, did not cover every occurrence, or was reverted.")
        a("")

    if suppressed:
        a(f"### Held back by earlier decisions ({len(suppressed)})")
        a("")
        a("| ID | Change | Decision | Who | When |")
        a("|---|---|---|---|---|")
        for f in sorted(suppressed, key=lambda x: x.decided_at):
            a("| " + " | ".join([
                f"`{f.id}`",
                _change_cell(f),
                f"`{f.status}`" + (f" ({f.detection_agreement})" if f.detection_agreement else ""),
                _escape(f.decided_by or "—"),
                f.decided_at or "—",
            ]) + " |")
        a("")

    if reverted:
        a(f"### Renamed and renamed back ({len(reverted)})")
        a("")
        a("These labels changed and then changed back within the window, so the docs")
        a("were never wrong and there is nothing to edit. Counted rather than listed,")
        a("for the same reason as undocumented surfaces.")
        a("")

    if orphans:
        a(f"### Stored decisions that matched nothing ({len(orphans)})")
        a("")
        a("Either the drift is genuinely gone, or this scan's window does not reach")
        a("back to the commit that caused it. Ambiguous, so nothing was deleted:")
        a("")
        a("```")
        for did in orphans:
            a(did)
        a("```")
        a("")

    return out


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
    candidate_deltas: int = 0,
    suppressed: Sequence[Finding] = (),
    reopened: Sequence[Finding] = (),
    unresolved: Sequence[Finding] = (),
    orphans: Sequence[str] = (),
    reverted: Sequence[Finding] = (),
) -> str:
    lines: list[str] = []
    a = lines.append

    a(f"# UI label drift — {config.SOURCE.owner_repo} → docs")
    a("")
    a(f"Scanned `{scanned_range}` on {today.isoformat()}.")
    detail = f" ({candidate_deltas} changed strings)" if candidate_deltas else ""
    funnel = (
        f"{commits} commits → {ui_commits} touching UI → {candidates} stage-1 "
        f"candidates{detail} → **{len(findings)} findings**, against "
        f"{docs_pages} indexed doc pages."
    )
    if suppressed:
        funnel += f" {len(suppressed)} previously decided finding(s) held back."
    a(funnel)
    a("")

    if reopened:
        a(f"> **{len(reopened)} decided finding(s) reopened.** The docs evidence behind "
          f"the original call has grown, so the decision was surfaced instead of "
          f"honored. See *Reopened decisions* below.")
        a("")

    if not findings:
        a("No drift to act on in this window.")
        a("")
        a("An empty table is a real result, not a broken run: every UI label change in")
        a("the window is already reflected in docs, touches no documented surface, or")
        if suppressed:
            # Saying "nothing found" when findings were held back would be a
            # lie of omission, and the reader has no way to catch it.
            a("was already decided on. See the sections below for what was held back.")
        else:
            a("was renamed back before anyone had to act on it.")
        a("")
        lines.extend(_gaps_section(gaps))
        lines.extend(_ledger_sections(suppressed, reopened, unresolved, orphans, reverted))
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

    # Known lanes first, then anything unrecognized. A finding with an
    # unexpected lane must still appear: the header has already counted it, and
    # a row that is tallied but not shown is the one failure mode a reader
    # cannot detect.
    for lane in (*_LANE_ORDER, *(l for l in by_lane if l not in _LANE_ORDER)):
        rows = by_lane[lane]
        if not rows:
            continue
        title = _LANE_TITLE.get(lane) or f"Unclassified ({lane or 'no lane'})"
        a(f"## {title} ({len(rows)})")
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

    lines.extend(_gaps_section(gaps))

    lines.extend(_ledger_sections(suppressed, reopened, unresolved, orphans, reverted))
    return "\n".join(lines) + "\n"
