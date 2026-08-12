"""The Finding record.

Step 1 defines the schema and the identity rule only. The triage decision
procedure, docs coverage, and ownership arrive in later steps -- the fields are
declared here so the shape is reviewable before anything depends on it.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .config import MAX_DOCS_PAGES, SETTLED_DAYS

# Findings are keyed by CONTENT, not by commit SHA.
#
# Two commits routinely produce one documentation problem: `f4861ad` and
# `17d7bf7` title-cased the same family of table headers two weeks apart, and a
# SHA-keyed store would file two rows for one fix. The inverse is just as real:
# `c99e959` renamed a toggle and `ccd66e2` renamed it again seven days later, so
# a SHA-keyed store files two rows for a string that only ever needed one.
#
# SHA does not disappear -- it becomes an array, and remains the join key into
# release-note-genie's cycles/<version>/ledger.json.

KIND_RENAME = "rename"
KIND_ADDED = "added"
KIND_REMOVED = "removed"
KIND_MOVED = "moved"
KIND_NEW_SETTING = "new_setting"

TRIAGE_AGENT = "agent"
TRIAGE_PAIR = "pair"
TRIAGE_HUMAN = "human"

COVERAGE_COVERED = "covered"
COVERAGE_NONE = "none"


@dataclass
class CommitRef:
    """One commit that contributed to this finding. Append-only."""

    sha: str
    date: str
    subject: str
    author: str
    file: str
    line: int
    pr: Optional[int] = None


@dataclass
class Finding:
    kind: str
    surface: str
    old_string: str
    new_string: str
    literal_kind: str  # attr | obj | jsx
    literal_key: str

    # Every code surface that made this same change. One docs page says
    # "MODELS SEAT" once; renaming it in three different member tables is still
    # one edit, so the id must not include the surface or the report shows the
    # same fix three times.
    surfaces: list[str] = field(default_factory=list)

    commits: list[CommitRef] = field(default_factory=list)
    first_seen_date: str = ""
    last_changed_date: str = ""
    settled: bool = False

    # Reported, never suppressed. A gated change is advance warning: draft the
    # docs while the change is fresh and hold the PR.
    gate: Optional[dict[str, Any]] = None
    not_yet_visible: bool = False

    docs: dict[str, Any] = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)
    confidence: float = 0.0
    degradations: list[str] = field(default_factory=list)

    triage: str = ""
    triage_reason: str = ""
    action: str = ""
    reviewers: list[str] = field(default_factory=list)
    owning_team: str = ""
    jira: dict[str, Any] = field(default_factory=dict)

    # --- human fields; a re-scan must never clobber these -----------------
    status: str = ""
    assignee: str = ""
    decided_by: str = ""
    decided_at: str = ""
    docs_pr: Optional[int] = None
    jira_key: Optional[str] = None
    # Captured at decision time because it cannot be reconstructed later.
    detection_agreement: str = ""  # "" | detected | missed | false_positive

    first_seen: str = ""
    last_updated: str = ""

    @property
    def id(self) -> str:
        raw = f"{self.kind}|{self.old_string}|{self.new_string}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["id"] = self.id
        return d


HUMAN_FIELDS = (
    "status", "assignee", "decided_by", "decided_at",
    "docs_pr", "jira_key", "detection_agreement",
)

MACHINE_REFRESH = (
    "kind", "surface", "surfaces", "old_string", "new_string", "literal_kind", "literal_key",
    "commits", "last_changed_date", "settled", "gate", "not_yet_visible",
    "docs", "signals", "confidence", "degradations",
    "triage", "triage_reason", "action", "reviewers", "owning_team", "jira",
)


# --- triage ---------------------------------------------------------------
#
# Deliberately asymmetric: easy to fall out of the agent lane, hard to fall in.
# A wrong `agent` call puts a false statement into published docs with nobody
# watching, which ends the project's credibility in one PR. A wrong `pair` call
# costs a writer fifteen minutes. So every uncertainty routes down.


def triage(f: "Finding") -> tuple[str, str]:
    """Return (lane, reason). First match wins."""
    docs = f.docs or {}
    targets = docs.get("replace_targets") or []

    # --- human: prose has to be written, not substituted ------------------
    if f.kind in (KIND_ADDED, KIND_NEW_SETTING):
        return TRIAGE_HUMAN, "new copy on screen; there is no old string to swap"
    if f.kind == KIND_REMOVED and docs.get("coverage") == COVERAGE_COVERED:
        return TRIAGE_HUMAN, "docs describe a control that is gone; deprecation is a judgment"
    if docs.get("coverage") == COVERAGE_NONE:
        return TRIAGE_HUMAN, "no page covers this surface (coverage gap, not a dead end)"
    if f.degradations:
        return TRIAGE_HUMAN, f"incomplete evidence: {', '.join(f.degradations)}"

    # --- pair: a mechanical edit exists, but its blast radius is unclear ---
    if f.not_yet_visible:
        return TRIAGE_PAIR, "gated: draft the change now, hold the PR until it ships"
    if f.kind == KIND_MOVED:
        return TRIAGE_PAIR, "string relocated rather than changed; it may still render"
    if not f.settled:
        return TRIAGE_PAIR, f"changed within {SETTLED_DAYS}d or changed twice; still moving"
    if "ambiguous_pairing" in f.signals:
        return TRIAGE_PAIR, "several equally good replacements; cannot tell which is which"
    if "url_changed" in f.signals:
        return TRIAGE_PAIR, "slug changed, so links and anchors moved too, not just words"
    if not targets:
        return TRIAGE_PAIR, "only occurrences are in published release notes; nothing to edit"
    if docs.get("code_context_only"):
        return TRIAGE_PAIR, "only appears in code spans; may be an API value, not a label"
    if docs.get("corpus_frequency", 0) > MAX_DOCS_PAGES:
        return TRIAGE_PAIR, f"appears on {docs['corpus_frequency']} pages; too broad to be one control"

    # --- agent ------------------------------------------------------------
    n = len(targets)
    where = "page" if len({t["page"] for t in targets}) == 1 else "pages"
    return TRIAGE_AGENT, (
        f"settled 1:1 rename, {n} marked-up occurrence{'s' if n != 1 else ''} "
        f"across {len({t['page'] for t in targets})} {where}"
    )


def action_for(lane: str, kind: str) -> str:
    if lane == TRIAGE_AGENT:
        return "cut a docs PR (find-and-replace on marked-up occurrences)"
    if lane == TRIAGE_PAIR:
        return "writer confirms scope, then an agent applies it"
    if kind in (KIND_ADDED, KIND_NEW_SETTING):
        return "write new docs"
    return "review and decide"
