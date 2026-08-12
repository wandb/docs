"""The Finding record.

Step 1 defines the schema and the identity rule only. The triage decision
procedure, docs coverage, and ownership arrive in later steps -- the fields are
declared here so the shape is reviewable before anything depends on it.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

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
        raw = f"{self.kind}|{self.old_string}|{self.new_string}|{self.surface}"
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
    "kind", "surface", "old_string", "new_string", "literal_kind", "literal_key",
    "commits", "last_changed_date", "settled", "gate", "not_yet_visible",
    "docs", "signals", "confidence", "degradations",
    "triage", "triage_reason", "action", "reviewers", "owning_team", "jira",
)
