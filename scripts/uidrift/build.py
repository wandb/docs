"""Assemble Findings from one commit's analysis.

Everything here is deterministic. The model pass (step 7) refines `surface` and
sanity-checks the classification; it does not discover anything this file
missed, which is why the table is useful before it exists.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

from . import config, docsindex, ownership, structure
from .extract import LabelDelta
from .finding import (
    COVERAGE_COVERED,
    COVERAGE_NONE,
    KIND_ADDED,
    KIND_MOVED,
    KIND_NEW_SETTING,
    KIND_REMOVED,
    KIND_RENAME,
    CommitRef,
    Finding,
    action_for,
    triage,
)

_PR = re.compile(r"\(#(\d+)\)")
# Split camelCase without shredding acronyms: LLMAsAJudgeScorerForm becomes
# "LLM as a judge scorer form", not "L L M As A Judge Scorer Form".
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def surface_from_path(path: str) -> str:
    """A human-readable name for the screen a string lives on.

    Derived from the component filename, which is a decent proxy and costs
    nothing. The stage-2 model pass replaces this with something a reader would
    recognize ("Organization dashboard -> Members table"); until then this is
    honest about being mechanical.
    """
    stem = Path(path).stem
    for suffix in ("Content", "Component", "Container", "Tab", "Page"):
        if stem.endswith(suffix) and len(stem) > len(suffix):
            stem = stem[: -len(suffix)]
    words = _CAMEL.sub(" ", stem).replace("_", " ").strip()
    return words[:1].upper() + words[1:] if words else path


def _docs_payload(lookup: docsindex.DocsLookup) -> dict:
    targets = lookup.replace_targets
    return {
        "coverage": COVERAGE_COVERED if lookup.ui_occurrences else COVERAGE_NONE,
        "eligible": lookup.eligible,
        "ineligible_reason": lookup.reason,
        "pages": [
            {"page": o.page, "line": o.line, "context": o.context, "immutable": o.immutable}
            for o in lookup.ui_occurrences
        ],
        "replace_targets": [
            {"page": o.page, "line": o.line, "context": o.context} for o in targets
        ],
        "translations_affected": lookup.translations_affected,
        "corpus_frequency": lookup.corpus_frequency,
        "all_occurrences_emphasized": lookup.all_occurrences_emphasized,
        "match_confidence": lookup.match_confidence,
        "code_context_only": bool(targets) and all(
            t["context"] == docsindex.CTX_CODE
            for t in ({"context": o.context} for o in targets)
        ),
        "touches_immutable": lookup.touches_immutable,
    }


def _landed_date(commit: dict) -> str:
    """When the commit landed on the watched branch, not when it was written.

    Settledness asks "has this stopped moving on master", and the author date
    cannot answer that: rebase and cherry-pick both preserve it, so a commit
    authored in March and landed today arrives already older than SETTLED_DAYS
    and skips the churn protection entirely -- straight into the unattended
    agent lane.

    The committer date is the landing date. It is read from the GitHub API's
    own shape (`commit.committer.date`), which `scan` fills in from the local
    clone, so this keeps working unchanged if the vendored reader ever starts
    supplying it. Falls back to the author date when it is absent, because a
    slightly-too-settled finding is a better failure than a crash.
    """
    committer = (commit.get("commit") or {}).get("committer") or {}
    return committer.get("date") or commit["commit"]["author"]["date"]


def _commit_ref(commit: dict, delta: LabelDelta) -> CommitRef:
    message = commit["commit"]["message"]
    subject = message.splitlines()[0]
    m = _PR.search(subject)
    return CommitRef(
        sha=commit["sha"],
        date=_landed_date(commit),
        subject=subject,
        author=commit["commit"]["author"].get("name", ""),
        file=delta.path,
        line=delta.line_no,
        pr=int(m.group(1)) if m else None,
    )


def _is_settled(commit_date: str, today: date) -> bool:
    try:
        when = datetime.fromisoformat(commit_date).date()
    except ValueError:
        return False
    return (today - when) >= timedelta(days=config.SETTLED_DAYS)


def build_findings(
    commit: dict,
    added: Sequence[LabelDelta],
    removed: Sequence[LabelDelta],
    moved: Sequence[LabelDelta],
    diff: str,
    index: docsindex.DocsIndex,
    *,
    today: date,
    core: Optional[Path] = None,
    resolve_owners: bool = True,
) -> tuple[list[Finding], list[str]]:
    streams = structure.parse_streams(diff)
    lifecycle = structure.flag_lifecycle(diff)
    pairs, unpaired_add, unpaired_rem = structure.pair_renames(added, removed, streams)

    commit_date = _landed_date(commit)
    settled = _is_settled(commit_date, today)
    findings: dict[str, Finding] = {}

    def emit(kind: str, old: str, new: str, probe: str, delta: LabelDelta,
             extra_signals: Sequence[str] = ()) -> None:
        lookup = docsindex.find(index, probe)
        gate = structure.gate_scope(streams, delta)
        signals = list(extra_signals)

        if structure.testid_corroboration(streams, delta):
            signals.append("testid_unchanged")
        if gate:
            signals.append(f"gate:{gate.name}")
            if gate.conditional_added:
                signals.append("conditional_added")

        # Gated only when this commit says so: a newly-added conditional around
        # the string, or the gate itself entering the registry here. Static
        # presence is never enough.
        gate_added_here = bool(gate and gate.key and lifecycle.get(gate.key) == "added")
        not_yet_visible = bool(gate and (gate.conditional_added or gate_added_here))
        for name, event in lifecycle.items():
            signals.append(f"flag_{event}:{name}")

        f = Finding(
            kind=kind,
            surface=surface_from_path(delta.path),
            old_string=old,
            new_string=new,
            literal_kind=delta.kind,
            literal_key=delta.key,
            commits=[_commit_ref(commit, delta)],
            first_seen_date=commit_date,
            last_changed_date=commit_date,
            settled=settled,
            gate={"name": gate.name, "hook": gate.hook,
                  "conditional_added": gate.conditional_added} if gate else None,
            not_yet_visible=not_yet_visible,
            docs=_docs_payload(lookup),
            signals=signals,
        )
        f.surfaces = [f.surface]
        if delta.wrapped:
            f.degradations.append("literal is interpolated or a ternary branch")

        if f.id in findings:
            existing = findings[f.id]
            if f.surface not in existing.surfaces:
                existing.surfaces.append(f.surface)
            known = {c.sha for c in existing.commits}
            existing.commits.extend(c for c in f.commits if c.sha not in known)
            return
        lane, reason = triage(f)
        f.triage, f.triage_reason = lane, reason
        f.action = action_for(lane, kind)
        if resolve_owners:
            own = ownership.resolve(
                delta.path, core=core, commit_author=f.commits[0].author
            )
            f.reviewers, f.owning_team = own.reviewers, own.team or ""
        findings[f.id] = f

    # A change to an UNDOCUMENTED label is not drift. Nothing in the docs became
    # wrong, because nothing in the docs said it. Emitting a row per such string
    # buried the two real findings under twenty rows of `Loading members` and
    # `Invited` the first time this ran.
    #
    # The coverage gap is still real, so it is counted and reported in aggregate.
    # That is not the suppression the direction-discipline rule forbids: absence
    # of docs never hides a finding that exists, it just stops manufacturing
    # findings that do not.
    gaps: list[str] = []

    def emit_if_documented(kind, old, new, probe, delta, extra=()):
        if not docsindex.find(index, probe).ui_occurrences:
            gaps.append(probe)
            return
        emit(kind, old, new, probe, delta, extra)

    for p in pairs:
        extra = ["case_only_change"] if p.case_only else []
        if p.ambiguous:
            extra.append("ambiguous_pairing")
        stability = structure.slug_stability(streams, p)
        if stability != "n/a":
            extra.append(stability)
        # Probe the OLD string: the question is whether docs still say the thing
        # the product stopped saying.
        emit_if_documented(KIND_RENAME, p.old.norm, p.new.norm, p.old.norm, p.old, extra)

    for d in unpaired_rem:
        emit_if_documented(KIND_REMOVED, d.norm, "", d.norm, d)

    for d in moved:
        emit_if_documented(KIND_MOVED, d.norm, d.norm, d.norm, d)

    # New copy is aggregated to ONE row per surface, not one per string. A new
    # settings panel adds a heading, a description, two field labels and a
    # button; that is one docs task, not five. Incidental additions (an empty
    # state, a spinner's aria-label) are not reported at all -- nobody documents
    # "Loading members".
    for path in {d.path for d in unpaired_add}:
        in_path = [d for d in unpaired_add if d.path == path]
        new_setting = structure.is_new_setting(unpaired_add, path)
        gate_added = any(ev == "added" for ev in lifecycle.values())
        if not (new_setting or gate_added):
            gaps.extend(d.norm for d in in_path)
            continue
        # Lead with the most descriptive string on the surface -- a heading or
        # title beats a bare field label like "Path", which tells a reader
        # nothing about what was added.
        lead = max(
            in_path,
            key=lambda d: (
                d.key in ("title", "heading", "header", "Header"),
                d.kind == "obj",
                min(len(d.norm), 60),
            ),
        )
        others = [d.norm for d in in_path if d.norm != lead.norm]
        emit(
            KIND_NEW_SETTING if new_setting else KIND_ADDED,
            "", lead.norm, lead.norm, lead,
            [f"and {len(others)} more new string(s) on this surface"] if others else [],
        )

    return list(findings.values()), gaps
