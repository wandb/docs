"""The entrypoint. Wires the pipeline together and writes the report.

    python3 -m uidrift.scan --since "60 days ago"
    python3 -m uidrift.scan --incremental          # since the last report
    python3 -m uidrift.scan decide <id> --status dismissed --by matt

Incremental mode takes its base from the newest report already in
`uidrift/reports/`, whose filename carries the head SHA it scanned. There is
deliberately no "last scanned" state file: the reports ARE the record, so the
watermark cannot drift away from what was actually published. See lesson 18 in
ADAPTING.md.

The scan never writes the ledger. Only `decide` does. A scan that could modify
decisions is a scan that could lose them.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Sequence

from . import build, config, docsindex, extract, ledger, ownership, report
from ._vendor import gitsource

# uidrift/reports/2026-08-13-71fa9d10412a.md
_REPORT_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})-([0-9a-f]{7,40})\.md$")


class ScanError(Exception):
    """Something the operator has to fix, reported without a traceback."""


@dataclass
class ScanResult:
    markdown: str
    stats: dict
    applied: ledger.Applied
    merged: ledger.Merged

    def by_id(self, finding_id: str):
        """Any finding this scan saw, reported or not.

        Suppressed and reverted rows are searchable on purpose: changing your
        mind about a dismissal is the main reason to reach for `decide` twice.
        """
        for f in (*self.applied.findings, *self.applied.suppressed,
                  *self.applied.unresolved, *self.merged.reverted):
            if f.id == finding_id:
                return f
        return None


def _resolve_base(core: Path, *, since: Optional[str], base: Optional[str],
                  head: str) -> tuple[str, str]:
    """Return (base_ref, description) for the range to scan."""
    if base:
        resolved = gitsource.resolve_sha(core, base)
        if not resolved:
            raise ScanError(f"cannot resolve --base {base!r} in {core}")
        return resolved, f"{resolved[:12]}..{head}"

    # `git log --since` would drop commits whose author date predates the window
    # but which landed inside it. Pinning a base SHA by date and diffing forward
    # keeps the range a contiguous range of history.
    r = subprocess.run(
        ["git", "-C", str(core), "rev-list", "-1", f"--before={since}", head],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        raise ScanError(f"no commit in {core} before {since!r} on {head}")
    return r.stdout.strip(), f"{since} .. {head}"


def _last_report(report_dir: Path) -> Optional[tuple[date, str, Path]]:
    """The newest report on disk, as (date, head_sha, path)."""
    found: list[tuple[date, str, Path]] = []
    for path in report_dir.glob("*.md"):
        m = _REPORT_NAME.match(path.name)
        if not m:
            continue
        try:
            when = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue
        found.append((when, m.group(2), path))
    return max(found) if found else None


def scan(
    *,
    since: Optional[str] = "60 days ago",
    base: Optional[str] = None,
    head: Optional[str] = None,
    today: Optional[date] = None,
    limit: Optional[int] = None,
    resolve_owners: bool = True,
    core: Optional[Path] = None,
    report_dir: Optional[Path] = None,
    incremental: bool = False,
    progress=lambda msg: None,
) -> ScanResult:
    """Run the pipeline."""
    root = core or config.SOURCE.path
    if not (root / ".git").exists():
        raise ScanError(
            f"{root} is not a git checkout. Set {config.SOURCE.local_path_env} "
            f"or clone {config.SOURCE.owner_repo} there."
        )
    head = head or config.SOURCE.default_head
    today = today or date.today()
    reports = report_dir or (config.DOCS.path / config.REPORT_DIR)

    if incremental:
        previous = _last_report(reports)
        if not previous:
            raise ScanError(
                f"--incremental needs a previous report in {reports}; "
                f"run once with --since first"
            )
        base = base or previous[1]
        progress(f"incremental from {previous[2].name}")

    base_sha, scanned_range = _resolve_base(root, since=since, base=base, head=head)

    commits = gitsource.iter_commits(root, base_sha, head, limit=limit)
    progress(f"{len(commits)} commits in range")

    ui_commits = [
        c for c in commits
        if any(extract.path_is_ui(f["filename"]) for f in c.get("files", []))
    ]
    progress(f"{len(ui_commits)} touch UI paths")

    index = docsindex.build_index()
    progress(f"docs index: {len(index)} pages")

    # Resolved once for the whole run rather than per finding; the caches are
    # process-local, so a fresh run always re-reads them.
    ownership.reset_caches()

    raw: list = []
    gaps: list[str] = []
    # Two different counts, because the established funnel reports commits while
    # the useful calibration number is deltas.
    candidate_commits = 0
    candidate_deltas = 0
    for i, commit in enumerate(ui_commits, 1):
        diff = gitsource.commit_diff(root, commit["sha"], *config.SOURCE.ui_roots)
        if not diff:
            continue
        surviving = extract.surviving_deltas(extract.extract_deltas(diff))
        if not surviving:
            continue
        candidate_commits += 1
        candidate_deltas += len(surviving)
        added, removed, moved = extract.commit_net_change(surviving)
        found, commit_gaps = build.build_findings(
            commit, added, removed, moved, diff, index,
            today=today, core=root, resolve_owners=resolve_owners,
        )
        raw.extend(found)
        gaps.extend(commit_gaps)
        if i % 50 == 0:
            progress(f"  {i}/{len(ui_commits)} commits, {len(raw)} raw findings")

    progress(f"{candidate_commits} commits with candidate strings "
             f"({candidate_deltas} deltas) -> {len(raw)} raw findings")

    merged = ledger.merge_findings(raw, today=today)
    applied = ledger.apply_decisions(merged.findings, ledger.load())
    progress(
        f"{len(applied.findings)} findings "
        f"({len(applied.suppressed)} suppressed, {len(applied.reopened)} reopened, "
        f"{len(merged.reverted)} reverted)"
    )

    markdown = report.render(
        applied.findings,
        scanned_range=scanned_range,
        today=today,
        commits=len(commits),
        ui_commits=len(ui_commits),
        candidates=candidate_commits,
        candidate_deltas=candidate_deltas,
        docs_pages=len(index),
        gaps=len(gaps),
        suppressed=applied.suppressed,
        reopened=applied.reopened,
        unresolved=applied.unresolved,
        orphans=applied.orphans,
        reverted=merged.reverted,
    )
    stats = {
        "base": base_sha,
        "head": gitsource.resolve_sha(root, head) or head,
        "commits": len(commits),
        "ui_commits": len(ui_commits),
        "candidate_commits": candidate_commits,
        "candidate_deltas": candidate_deltas,
        "findings": len(applied.findings),
        "suppressed": len(applied.suppressed),
        "reopened": len(applied.reopened),
        "reverted": len(merged.reverted),
        "gaps": len(gaps),
        "orphans": applied.orphans,
    }
    return ScanResult(markdown=markdown, stats=stats, applied=applied, merged=merged)


def _cmd_scan(args: argparse.Namespace) -> int:
    def progress(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    result = scan(
        since=args.since,
        base=args.base,
        head=args.head,
        limit=args.limit,
        resolve_owners=not args.no_owners,
        incremental=args.incremental,
        progress=progress,
    )
    stats = result.stats

    if args.stdout:
        print(result.markdown)
        return 0

    reports = config.DOCS.path / config.REPORT_DIR
    reports.mkdir(parents=True, exist_ok=True)
    # The head SHA in the name is what makes --incremental work.
    out = reports / f"{date.today().isoformat()}-{stats['head'][:12]}.md"
    out.write_text(result.markdown, encoding="utf-8")
    progress(f"wrote {out}")

    if stats["reopened"]:
        # Worth a distinct exit code: a reopened decision means a human's
        # earlier call no longer matches the evidence, which is the one outcome
        # that should be able to fail a CI step.
        return 3
    return 0


def _cmd_decide(args: argparse.Namespace) -> int:
    """Record a human decision against a finding.

    The finding is re-derived by scanning rather than read out of a report,
    because the evidence fingerprint has to reflect the corpus as it is now. A
    decision stamped with stale evidence would never reopen.
    """
    def progress(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    result = scan(since=args.since, base=args.base, resolve_owners=False,
                  progress=progress)
    finding = result.by_id(args.finding_id)
    if finding is None:
        raise ScanError(
            f"no finding {args.finding_id!r} in this window. Widen --since, or "
            f"check the id against the newest report."
        )

    decisions = ledger.load()
    existing = decisions.get(args.finding_id)
    if existing and not args.force:
        raise ScanError(
            f"{args.finding_id} is already {existing.status!r} "
            f"(by {existing.decided_by or 'unknown'} on "
            f"{existing.decided_at or 'unknown date'}). Pass --force to replace it."
        )

    decisions[args.finding_id] = ledger.record(
        finding, args.status, today=date.today(),
        decided_by=args.by, assignee=args.assignee, docs_pr=args.docs_pr,
        jira_key=args.jira, detection_agreement=args.agreement, note=args.note,
    )
    path = ledger.save(decisions)
    print(f"{args.finding_id} -> {args.status} ({path})")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="uidrift.scan",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    s = sub.add_parser("scan", help="scan a commit range and write a report")
    s.add_argument("--since", default="60 days ago",
                   help="window start, as a git date (default: %(default)r)")
    s.add_argument("--base", help="explicit base SHA, overrides --since")
    s.add_argument("--head", help=f"default: {config.SOURCE.default_head}")
    s.add_argument("--incremental", action="store_true",
                   help="base on the head SHA of the newest existing report")
    s.add_argument("--limit", type=int, help="stop after N commits (for testing)")
    s.add_argument("--no-owners", action="store_true",
                   help="skip reviewer/team resolution")
    s.add_argument("--stdout", action="store_true",
                   help="print the report instead of writing it")
    s.add_argument("--quiet", action="store_true", help="no progress on stderr")
    s.set_defaults(func=_cmd_scan)

    d = sub.add_parser("decide", help="record a human decision")
    d.add_argument("finding_id")
    d.add_argument("--status", required=True, choices=ledger.STATUSES)
    d.add_argument("--by", default="", help="who decided")
    d.add_argument("--note", default="", help="why")
    d.add_argument("--assignee", default="")
    d.add_argument("--jira", help="JIRA key, e.g. DOCS-1234")
    d.add_argument("--docs-pr", type=int, help="docs PR number")
    d.add_argument("--agreement", default="",
                   choices=("", "detected", "missed", "false_positive"),
                   help="was the detector right? captured now, unreconstructable later")
    d.add_argument("--force", action="store_true",
                   help="replace an existing decision for this id")
    d.add_argument("--since", default="60 days ago")
    d.add_argument("--base", help="explicit base SHA, overrides --since")
    d.add_argument("--quiet", action="store_true")
    d.set_defaults(func=_cmd_decide)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except ScanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
