#!/usr/bin/env python3
"""UI label drift detector -- CLI entrypoint.

Step 1 ships `--dry-run` only: walk a commit range in the watched repo, run the
deterministic stage-1 funnel, and print what survives. No ledger, no docs index,
no model, no network, no token.

    python3 scripts/uidrift_watch.py --dry-run --since "60 days ago"
    python3 scripts/uidrift_watch.py --dry-run --range ccd66e2~1..ccd66e2 -v

The point of shipping this alone is that the funnel's reduction ratio and its
per-commit output are reviewable before anything depends on them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uidrift import config, extract  # noqa: E402
from uidrift._vendor import gitsource  # noqa: E402


def _resolve_range(core: Path, args, cfg: config.SourceRepo) -> tuple[str, str]:
    if args.range:
        if ".." not in args.range:
            raise SystemExit(f"--range must be BASE..HEAD, got {args.range!r}")
        base, head = args.range.split("..", 1)
        return base, head

    head = args.head or cfg.default_head
    since = args.since or "60 days ago"
    # NB: git applies --max-count BEFORE --reverse, so combining them returns the
    # NEWEST commit, not the oldest. Take the whole window and use its last line.
    out = subprocess.run(
        ["git", "-C", str(core), "log", head, f"--since={since}", "--format=%H"],
        capture_output=True, text=True,
    )
    shas = out.stdout.strip().splitlines()
    if not shas:
        raise SystemExit(f"no commits on {head} since {since!r}")
    return f"{shas[-1]}~1", head


def scan(args) -> int:
    cfg = config.SOURCE
    core = cfg.path
    if not (core / ".git").exists():
        raise SystemExit(
            f"no git repo at {core}. Set {cfg.local_path_env} or clone {cfg.owner_repo}."
        )

    base, head = _resolve_range(core, args, cfg)
    commits = gitsource.iter_commits(core, base, head, owner_repo=cfg.owner_repo)

    # Cheap path filter first: no diff is fetched for a commit that touches no
    # UI file. Deliberately NOT filtered on conventional-commit type -- half the
    # richest real finding in the corpus arrived under `refactor(app):`.
    ui_commits = [
        c for c in commits
        if any(extract.path_is_ui(f["filename"], cfg) for f in c.get("files", []))
    ]

    candidates = []
    for c in ui_commits:
        # Pathspec-limited: this is what keeps a 2,600-line commit affordable.
        diff = gitsource.commit_diff(core, c["sha"], *cfg.ui_roots)
        if not diff:
            continue
        surviving = extract.surviving_deltas(extract.extract_deltas(diff, cfg))
        if not surviving:
            continue
        added, removed, moved = extract.commit_net_change(surviving)
        if not (added or removed or moved):
            continue
        candidates.append((c, added, removed, moved))

    print(f"range      {base}..{head}")
    print(f"commits    {len(commits)}")
    print(f"UI-touching{len(ui_commits):>6}")
    print(f"candidates {len(candidates):>6}")
    if ui_commits:
        pct = 100 * (1 - len(candidates) / len(ui_commits))
        print(f"reduction  {pct:>5.1f}%  (stage 1, deterministic)")
    print()

    for c, added, removed, moved in candidates:
        subject = c["commit"]["message"].splitlines()[0]
        print(f"{c['sha'][:7]}  +{len(added):<3} -{len(removed):<3} ~{len(moved):<3}  {subject[:78]}")
        if args.verbose:
            for label, items in (("+", added), ("-", removed), ("~", moved)):
                for d in items:
                    flag = " (wrapped)" if d.wrapped else ""
                    print(f"           {label} {d.kind}/{d.key:<16} {d.norm[:58]!r}{flag}")
                    print(f"             {d.path}:{d.line_no}")
            print()

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", action="store_true",
                   help="print stage-1 candidates and exit (the only mode in step 1)")
    p.add_argument("--range", help="explicit BASE..HEAD")
    p.add_argument("--head", help=f"head ref (default {config.SOURCE.default_head})")
    p.add_argument("--since", help='window when --range is absent, e.g. "60 days ago"')
    p.add_argument("-v", "--verbose", action="store_true", help="print every label delta")
    args = p.parse_args()

    if not args.dry_run:
        p.error("step 1 ships --dry-run only")
    return scan(args)


if __name__ == "__main__":
    raise SystemExit(main())
