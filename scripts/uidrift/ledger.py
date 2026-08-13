"""State that survives a re-scan.

Two jobs live here and only one of them needs a file on disk.

`merge_findings` collapses the per-commit finding lists a scan produces into one
row per docs task. It looks like the ledger's job and is not: identity is already
content-addressed, and every input is in the commit history, so a re-scan
recomputes it identically. Storing it would only create something that can go
stale. This is what finally resolves the two-commit `ORG ROLE` cluster -- and it
also fixes a real bug in doing so. `settled` was computed from the date of
whichever commit happened to be in front of the loop; it has to come from the
LAST change, or a label renamed 60 days ago and renamed again yesterday reports
as safe to act on.

`apply_decisions` is the part that genuinely needs a file, because "a human
looked at this and it is fine" cannot be derived from anything. That is the only
thing the ledger stores. Dedupe, settledness, triage, ownership and docs
coverage are all recomputed every run.

The suppression here obeys the same one-directional rule as the rest of the
detector: a stored decision must never hide a finding that has since become
real. So a decision records the docs evidence it was made against, and if that
evidence later EXPANDS -- a page starts mentioning the string, a new editable
occurrence appears -- the decision is surfaced as stale rather than honored.
Evidence shrinking is not a reopen; that is the work getting done.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from . import config
from .finding import (
    COVERAGE_COVERED,
    COVERAGE_NONE,
    HUMAN_FIELDS,
    KIND_RENAME,
    Finding,
    action_for,
    triage,
)

LEDGER_VERSION = 1

# A human looked at this and decided it needs no docs change. Suppressed.
STATUS_DISMISSED = "dismissed"
# Real, and queued -- jira_key or assignee says where. Still reported.
STATUS_ACCEPTED = "accepted"
# Docs were updated. Normally self-extinguishing: the next scan probes the old
# string, does not find it in docs, and never emits the finding. So a `fixed`
# decision that still matches a finding means the fix did not land or was
# reverted, and that is worth saying out loud rather than suppressing.
STATUS_FIXED = "fixed"

STATUSES = (STATUS_DISMISSED, STATUS_ACCEPTED, STATUS_FIXED)


class LedgerError(Exception):
    """The ledger file is unreadable or malformed.

    Always raised, never swallowed. This file is hand-edited and holds the only
    unrecoverable state in the system; silently starting from an empty ledger
    would discard human decisions and re-report everything already settled.
    """


@dataclass
class Decision:
    """One human judgment about one finding."""

    status: str
    assignee: str = ""
    decided_by: str = ""
    decided_at: str = ""
    docs_pr: Optional[int] = None
    jira_key: Optional[str] = None
    detection_agreement: str = ""  # "" | detected | missed | false_positive
    note: str = ""
    # The docs evidence this decision was made against. See evidence_expanded.
    evidence: dict[str, Any] = field(default_factory=dict)
    # Decorative: a 12-character id tells a human nothing when they open the
    # file to add a note. Never read back, so it cannot drift into a lie.
    change: str = ""

    def human_fields(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "assignee": self.assignee,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at,
            "docs_pr": self.docs_pr,
            "jira_key": self.jira_key,
            "detection_agreement": self.detection_agreement,
        }


@dataclass
class Merged:
    """One run's findings, collapsed to one row per docs task."""

    findings: list[Finding] = field(default_factory=list)
    # A->B->A. Net zero change, so docs are already correct and there is nothing
    # to report -- but counted, never silently dropped.
    reverted: list[Finding] = field(default_factory=list)


@dataclass
class Applied:
    """What the ledger did to this run's findings."""

    findings: list[Finding] = field(default_factory=list)
    suppressed: list[Finding] = field(default_factory=list)
    reopened: list[Finding] = field(default_factory=list)
    unresolved: list[Finding] = field(default_factory=list)
    # Decision ids that matched nothing this run. Never auto-deleted -- see
    # prunable() for why that is a human's call.
    orphans: list[str] = field(default_factory=list)


# --- merge (recomputed, never stored) -------------------------------------


def _is_settled(when: str, today: date) -> bool:
    try:
        parsed = datetime.fromisoformat(when).date()
    except ValueError:
        return False
    return (today - parsed) >= timedelta(days=config.SETTLED_DAYS)


def _dedup(values: Iterable[str]) -> list[str]:
    """Order-preserving dedupe. Order is what makes the report diff readably."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _last_date(f: Finding) -> str:
    return max((c.date for c in f.commits), default="")


def _first_date(f: Finding) -> str:
    return min((c.date for c in f.commits), default="")


def _absorb(base: Finding, parts: Sequence[Finding], *, today: date) -> Finding:
    """Fold the commits, surfaces and evidence of `parts` into a copy of `base`."""
    out = copy.deepcopy(base)

    commits: dict[str, Any] = {}
    for inst in parts:
        for c in inst.commits:
            commits.setdefault(c.sha, c)
    out.commits = sorted(commits.values(), key=lambda c: c.date)

    dates = [c.date for c in out.commits if c.date]
    out.first_seen_date = min(dates) if dates else ""
    out.last_changed_date = max(dates) if dates else ""
    # The whole point of settledness is "has stopped moving", so it comes from
    # the most recent change, not whichever one the loop happened to be on.
    out.settled = _is_settled(out.last_changed_date, today) if dates else False

    out.surfaces = _dedup(s for inst in parts for s in (inst.surfaces or [inst.surface]))
    out.signals = _dedup(s for inst in parts for s in inst.signals)
    # Any instance with incomplete evidence routes the merged row down.
    out.degradations = _dedup(d for inst in parts for d in inst.degradations)
    return out


def _chain_links(renames: Sequence[Finding]) -> dict[int, Optional[int]]:
    """Map each rename to the rename that supersedes it, by index.

    A link exists only when it is unambiguous in both directions: exactly one
    later rename starts from this one's new string, and exactly one rename ends
    at that string. A fork or a join is left unlinked and flagged, because
    guessing which target is current is precisely the mistake that would put a
    stale string into published docs.
    """
    by_old: dict[str, list[int]] = {}
    by_new: dict[str, list[int]] = {}
    for i, f in enumerate(renames):
        by_old.setdefault(f.old_string, []).append(i)
        by_new.setdefault(f.new_string, []).append(i)

    links: dict[int, Optional[int]] = {}
    for i, f in enumerate(renames):
        nxt = [
            j for j in by_old.get(f.new_string, [])
            if j != i and _first_date(renames[j]) >= _last_date(f)
        ]
        prev = [j for j in by_new.get(f.new_string, []) if j != i]
        links[i] = nxt[0] if len(nxt) == 1 and len(prev) == 0 else None
        if len(nxt) > 1:
            f.signals.append("ambiguous_chain")
    return links


def _chain_renames(renames: Sequence[Finding], *, today: date) -> Merged:
    """Collapse A->B->C into A->C.

    The head's docs payload is kept, not the tail's: the probe that found docs
    evidence ran against the head's OLD string, which is what the pages actually
    say. Only the replacement target moves to the end of the chain.
    """
    links = _chain_links(renames)
    successors = {j for j in links.values() if j is not None}
    out = Merged()
    walked: set[int] = set()

    for i in range(len(renames)):
        if i in successors or i in walked:
            continue
        chain = [i]
        walked.add(i)
        nxt = links[i]
        while nxt is not None and nxt not in walked:
            chain.append(nxt)
            walked.add(nxt)
            nxt = links[nxt]

        parts = [renames[k] for k in chain]
        if len(parts) == 1:
            out.findings.append(_absorb(parts[0], parts, today=today))
            continue

        head, tail = parts[0], parts[-1]
        folded = _absorb(head, parts, today=today)
        folded.new_string = tail.new_string
        folded.signals = _dedup([*folded.signals, f"rename_chain:{len(parts)}"])
        if folded.old_string == folded.new_string:
            # Renamed and renamed back. Docs were right all along.
            out.reverted.append(folded)
        else:
            out.findings.append(folded)

    # Anything a cycle kept us from reaching still has to be reported.
    for i, f in enumerate(renames):
        if i not in walked:
            out.findings.append(_absorb(f, [f], today=today))
    return out


def merge_findings(findings: Sequence[Finding], *, today: date) -> Merged:
    """Collapse findings that describe the same docs task.

    Two passes, because two different things produce duplicate rows.
    `build_findings` dedupes within one commit; the first pass here dedupes the
    same change appearing in several commits, and the second follows renames
    that were themselves renamed later. Triage is recomputed at the end so it
    sees the merged picture rather than one commit's slice of it.
    """
    groups: dict[str, list[Finding]] = {}
    for f in findings:
        groups.setdefault(f.id, []).append(f)

    by_id: list[Finding] = []
    for instances in groups.values():
        ordered = sorted(instances, key=_last_date)
        by_id.append(_absorb(ordered[-1], ordered, today=today))

    renames = [f for f in by_id if f.kind == KIND_RENAME and f.old_string and f.new_string]
    chainable = {id(f) for f in renames}
    out = _chain_renames(renames, today=today)
    out.findings.extend(f for f in by_id if id(f) not in chainable)

    for f in (*out.findings, *out.reverted):
        f.triage, f.triage_reason = triage(f)
        f.action = action_for(f.triage, f.kind)
    return out


# --- decisions (the only persisted state) ---------------------------------


def evidence_of(f: Finding) -> dict[str, Any]:
    """The docs evidence a decision is made against.

    Pages, not line numbers: lines churn on every unrelated docs edit, and a
    reopen on that would be pure noise.
    """
    docs = f.docs or {}
    return {
        "coverage": docs.get("coverage") or COVERAGE_NONE,
        "pages": sorted({p["page"] for p in (docs.get("pages") or [])}),
        "targets": sorted({t["page"] for t in (docs.get("replace_targets") or [])}),
    }


def evidence_expanded(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    """Has the docs evidence grown since the decision was made?

    Only growth counts. A page that stopped mentioning the string means someone
    did the work; a page that started mentioning it means the decision was made
    on a smaller picture than the one we have now.
    """
    # A hand-written decision with no evidence block is taken at face value.
    # Requiring the fingerprint would mean a writer cannot dismiss something by
    # editing the file, which is the main way this file gets used.
    if not stored:
        return False
    if stored.get("coverage") == COVERAGE_NONE and current.get("coverage") == COVERAGE_COVERED:
        return True
    for key in ("pages", "targets"):
        if set(current.get(key) or []) - set(stored.get(key) or []):
            return True
    return False


def apply_decisions(
    findings: Sequence[Finding], decisions: dict[str, Decision]
) -> Applied:
    """Overlay stored human decisions onto this run's findings."""
    out = Applied()
    seen: set[str] = set()

    for f in findings:
        decision = decisions.get(f.id)
        if decision is None:
            out.findings.append(f)
            continue

        seen.add(f.id)
        for name, value in decision.human_fields().items():
            if name in HUMAN_FIELDS:
                setattr(f, name, value)

        if evidence_expanded(decision.evidence, evidence_of(f)):
            f.signals.append(f"reopened:{decision.status}")
            out.reopened.append(f)
            out.findings.append(f)
        elif decision.status == STATUS_DISMISSED:
            out.suppressed.append(f)
        elif decision.status == STATUS_FIXED:
            f.signals.append("marked_fixed_still_detected")
            out.unresolved.append(f)
            out.findings.append(f)
        else:
            out.findings.append(f)

    out.orphans = sorted(set(decisions) - seen)
    return out


def record(
    finding: Finding,
    status: str,
    *,
    today: date,
    decided_by: str = "",
    assignee: str = "",
    docs_pr: Optional[int] = None,
    jira_key: Optional[str] = None,
    detection_agreement: str = "",
    note: str = "",
) -> Decision:
    """Build a Decision, capturing the evidence it was made against.

    Going through here rather than constructing a Decision directly is what
    makes the reopen check work, so it is the only supported way to add one
    programmatically.
    """
    if status not in STATUSES:
        raise LedgerError(f"unknown status {status!r}; expected one of {', '.join(STATUSES)}")
    return Decision(
        status=status,
        assignee=assignee,
        decided_by=decided_by,
        decided_at=today.isoformat(),
        docs_pr=docs_pr,
        jira_key=jira_key,
        detection_agreement=detection_agreement,
        note=note,
        evidence=evidence_of(finding),
        change=f"{finding.old_string or '(new)'} -> {finding.new_string or '(removed)'}",
    )


def prunable(applied: Applied, decisions: dict[str, Decision]) -> list[str]:
    """Orphaned decisions that look finished.

    Reported, never acted on automatically. An orphan is ambiguous: the finding
    may be genuinely resolved, or the scan window may simply not reach back far
    enough to see its commit. Deleting a human's record on that guess is not a
    call this code gets to make.
    """
    return [
        did for did in applied.orphans
        if decisions[did].status in (STATUS_FIXED, STATUS_DISMISSED)
    ]


# --- persistence ----------------------------------------------------------

_DECISION_KEYS = set(Decision.__dataclass_fields__)


def _parse_decision(finding_id: str, raw: Any) -> Decision:
    if not isinstance(raw, dict):
        raise LedgerError(f"decision {finding_id!r} is {type(raw).__name__}, expected an object")

    unknown = sorted(set(raw) - _DECISION_KEYS)
    if unknown:
        # A typo'd key silently doing nothing is the worst outcome for a
        # hand-edited file -- the writer thinks they recorded a decision.
        raise LedgerError(
            f"decision {finding_id!r} has unknown field(s): {', '.join(unknown)}. "
            f"Valid fields: {', '.join(sorted(_DECISION_KEYS))}"
        )

    status = raw.get("status")
    if status not in STATUSES:
        raise LedgerError(
            f"decision {finding_id!r} has status {status!r}; "
            f"expected one of {', '.join(STATUSES)}"
        )
    return Decision(**raw)


def load(path: Optional[Path] = None) -> dict[str, Decision]:
    """Read the ledger. A missing file is a normal first run, not an error."""
    target = path or (config.DOCS.path / config.LEDGER_PATH)
    if not target.exists():
        return {}
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LedgerError(f"cannot read ledger at {target}: {exc}") from exc

    if not isinstance(raw, dict):
        raise LedgerError(f"ledger at {target} is not an object")
    version = raw.get("version")
    if version != LEDGER_VERSION:
        raise LedgerError(
            f"ledger at {target} is version {version!r}, expected {LEDGER_VERSION}"
        )
    decisions = raw.get("decisions") or {}
    if not isinstance(decisions, dict):
        raise LedgerError(f"ledger at {target}: 'decisions' is not an object")

    return {fid: _parse_decision(fid, d) for fid, d in decisions.items()}


def save(decisions: dict[str, Decision], path: Optional[Path] = None) -> Path:
    """Write the ledger.

    Sorted and indented because this file is read and edited by hand, and its
    diffs land in review.
    """
    target = path or (config.DOCS.path / config.LEDGER_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": LEDGER_VERSION,
        "decisions": {fid: asdict(decisions[fid]) for fid in sorted(decisions)},
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
