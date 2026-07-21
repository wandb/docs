#!/usr/bin/env python3
"""
Readability delta PR report
===========================

Turns the readability impact of a pull request into Markdown for humans. GitHub
Actions either posts that Markdown as a single (upserted) PR comment or writes it
to the workflow job summary.

What this script does (high level)
----------------------------------
1. Lists the ``.mdx`` files changed between the PR base and head commit
   (``git diff --name-status base head``), skipping localized content under
   ``ja/``, ``ko/``, and ``fr/`` (translation is handled separately).
2. For each changed file, reads the **before** version (``git show base:path``)
   and the **after** version (``git show head:path``), extracts narrative prose,
   and scores both with the readability analyzer from the ``coreweave/docs-skills``
   submodule (``.claude/scripts/_readability.py``).
3. Optionally runs the AI-agent-comprehension LLM judge (W&B Inference) on the
   before and after versions when ``--judge`` is set and ``WANDB_API_KEY`` is
   available. The deterministic readability delta always runs; the judge is
   supplementary and degrades gracefully.
4. Aggregates a word-weighted Flesch-Kincaid grade delta across the changed
   pages and builds a Markdown report, optionally prefixed with an HTML marker so
   the workflow can upsert a single comment.

The deterministic analyzer and the LLM judge both live in the submodule so the
logic is shared with ``coreweave/documentation``. This script is the thin
wandb/docs plumbing around them. The Markdown-building functions are pure and
take plain numbers, so they can be unit tested without ``textstat`` or network
access.

This check is informational. It never fails a PR.

See also
--------
- ``.claude/scripts/_readability.py`` for the analyzer.
- ``.claude/scripts/_docs_eval_lib.py`` for the comprehension judge rubric.
- ``.github/workflows/readability-delta.yml`` for when this runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GitHub issue comments are upserted by searching for this exact HTML comment.
REPORT_MARKER = "<!-- readability-delta-report -->"

# Visible heading in PR comments and job summaries (sentence case per docs style).
REPORT_TITLE = "## Readability impact"

# Shown when no scorable English .mdx prose changed in the PR.
REPORT_FALLBACK_PARAGRAPH = (
    "No English documentation prose changed in this PR, so there is no "
    "readability delta to report."
)

REPORT_FALLBACK_BODY = f"{REPORT_TITLE}\n\n{REPORT_FALLBACK_PARAGRAPH}"

# One-line explanation included under the headline so reviewers know how to read
# the numbers and that the check is advisory.
REPORT_LEGEND = (
    "Lower Flesch-Kincaid grade and higher reading ease both mean easier to "
    "read. This check is informational and never blocks a PR."
)

# Localized content prefixes to skip. Translation is handled separately, so
# scoring localized prose with English-tuned formulas is not meaningful.
_LOCALE_PREFIXES = ("ja/", "ko/", "fr/")

# Cap on how many pages get the (paid, slower) LLM comprehension judge per run.
# The deterministic delta still covers every changed page.
MAX_JUDGED_FILES = int(os.environ.get("READABILITY_MAX_JUDGED_FILES", "10"))


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _run_git(args: List[str], repo_root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def changed_mdx_name_status(base: str, head: str, repo_root: Path) -> str:
    """Return ``git diff --name-status base head`` output (raw)."""
    result = _run_git(["diff", "--name-status", base, head], repo_root)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode or 1)
    return result.stdout


def git_show(ref_path: str, repo_root: Path) -> str:
    """Return the content of ``ref:path`` (``git show``), or "" if absent."""
    result = _run_git(["show", ref_path], repo_root)
    if result.returncode != 0:
        return ""
    return result.stdout


# ---------------------------------------------------------------------------
# Parsing changed files
# ---------------------------------------------------------------------------


def parse_changed_files(name_status: str) -> List[Dict[str, str]]:
    """
    Parse ``git diff --name-status`` into changed English ``.mdx`` entries.

    Returns a list of dicts with ``status`` (``A``/``M``/``D``/``R``/``C``),
    ``path`` (the new path for renames/copies), and ``old_path`` (the base-side
    path, which differs from ``path`` only for renames/copies). Localized files
    and non-mdx files are skipped. Deletions are kept so the report can note
    removed pages, but they are not scored (no after version).
    """
    entries: List[Dict[str, str]] = []
    for raw in name_status.splitlines():
        line = raw.rstrip("\n\r")
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0]
        # Renames/copies (R###/C###) have three fields: status, old, new.
        # Score against the new path but read the base version from the old one.
        path = parts[-1]
        old_path = parts[1] if status[0] in ("R", "C") and len(parts) >= 3 else path
        if not path.endswith(".mdx"):
            continue
        if path.startswith(_LOCALE_PREFIXES):
            continue
        entries.append({"status": status[0], "path": path, "old_path": old_path})
    return entries


# ---------------------------------------------------------------------------
# Scoring (requires the submodule analyzer; isolated for testability)
# ---------------------------------------------------------------------------


def _import_readability(skills_scripts: Path):
    """Import the _readability module from the docs-skills submodule."""
    skills_scripts = skills_scripts.resolve()
    if str(skills_scripts) not in sys.path:
        sys.path.insert(0, str(skills_scripts))
    import _readability  # type: ignore

    return _readability


def score_entry(
    entry: Dict[str, str],
    base: str,
    head: str,
    repo_root: Path,
    readability,
) -> Dict[str, object]:
    """Score one changed file, returning a plain-number result row."""
    path = entry["path"]
    status = entry["status"]
    # For renames/copies the base version lives at the old path.
    old_path = entry.get("old_path", path)

    if status == "D":
        return {"path": path, "status": "deleted"}

    before_text = "" if status == "A" else git_show(f"{base}:{old_path}", repo_root)
    after_text = git_show(f"{head}:{path}", repo_root)

    before_prose = readability.extract_prose_mdx(before_text)
    after_prose = readability.extract_prose_mdx(after_text)
    report = readability.analyze_texts(before_prose, after_prose)

    before = report["before"]
    after = report["after"]
    delta = report["delta"]
    headline = report["headline"]

    # A new file (or one with no scorable before) has no delta.
    if after["insufficient_prose"]:
        return {"path": path, "status": "insufficient_prose",
                "after_word_count": after["word_count"]}

    return {
        "path": path,
        "old_path": old_path,
        "status": "new" if status == "A" or before["insufficient_prose"] else "scored",
        "fk_before": before["metrics"]["flesch_kincaid_grade"] if before["metrics"] else None,
        "fk_after": after["metrics"]["flesch_kincaid_grade"] if after["metrics"] else None,
        "fk_delta": delta["flesch_kincaid_grade"],
        "ease_delta": delta["flesch_reading_ease"],
        "direction": headline["direction"],
        "after_word_count": after["word_count"],
    }


def run_comprehension_judge(
    skills_scripts: Path,
    before_text: str,
    after_text: str,
) -> Optional[Dict[str, object]]:
    """
    Score AI agent comprehension on the before and after text via W&B Inference.

    Returns a dict with before/after ratings and the delta, or None if the judge
    is unavailable (no WANDB_API_KEY, missing deps, or an API error). Imported
    lazily so the deterministic path and the unit tests do not need weave/openai.
    """
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        skills_scripts = skills_scripts.resolve()
        if str(skills_scripts) not in sys.path:
            sys.path.insert(0, str(skills_scripts))
        from _docs_eval_lib import _RUBRIC_AI_COMPREHENSION, _call_judge  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on CI env
        print(f"[readability] comprehension judge unavailable: {exc}", file=sys.stderr)
        return None

    try:
        before_res = (
            _call_judge(_RUBRIC_AI_COMPREHENSION, before_text) if before_text.strip() else None
        )
        after_res = _call_judge(_RUBRIC_AI_COMPREHENSION, after_text)
    except Exception as exc:  # pragma: no cover - network/runtime
        print(f"[readability] comprehension judge error: {exc}", file=sys.stderr)
        return None

    before_rating = before_res.get("rating") if before_res else None
    after_rating = after_res.get("rating") if after_res else None
    rating_delta = (
        after_rating - before_rating
        if isinstance(before_rating, int) and isinstance(after_rating, int)
        else None
    )
    return {
        "before_rating": before_rating,
        "after_rating": after_rating,
        "rating_delta": rating_delta,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """
    Word-weighted Flesch-Kincaid grade delta across scored files.

    Weighting by the after-version word count keeps a one-line tweak on a tiny
    page from dominating a large rewrite. Returns None when nothing was scored.
    """
    weighted_sum = 0.0
    weight_total = 0
    scored = 0
    for r in results:
        if r.get("status") != "scored":
            continue
        fk_delta = r.get("fk_delta")
        words = r.get("after_word_count") or 0
        if fk_delta is None or words <= 0:
            continue
        weighted_sum += float(fk_delta) * int(words)
        weight_total += int(words)
        scored += 1

    if weight_total == 0:
        return None

    weighted_delta = round(weighted_sum / weight_total, 1)
    # round() can return -0.0 for tiny negative averages, which renders as
    # "-0.0" in the headline even though the direction is "unchanged". Normalize.
    if weighted_delta == 0:
        weighted_delta = 0.0
    if weighted_delta < 0:
        direction = "easier"
    elif weighted_delta > 0:
        direction = "harder"
    else:
        direction = "unchanged"
    return {
        "weighted_fk_delta": weighted_delta,
        "direction": direction,
        "pages_scored": scored,
    }


# ---------------------------------------------------------------------------
# Markdown building (pure; safe to unit test)
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:+.1f}" if isinstance(value, float) else str(value)


def _fmt_abs(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.1f}"


def build_report_markdown(
    results: List[Dict[str, object]],
    summary: Optional[Dict[str, object]],
    *,
    baselines: Optional[Dict[str, object]] = None,
    judge_by_path: Optional[Dict[str, Dict[str, object]]] = None,
    run_url: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    """Build the report Markdown body (without the HTML marker)."""
    scorable = [r for r in results if r.get("status") in ("scored", "new", "insufficient_prose", "deleted")]
    if not scorable:
        return REPORT_FALLBACK_BODY

    lines: List[str] = [REPORT_TITLE, ""]

    if summary:
        n = summary["pages_scored"]
        lines.append(
            f"Word-weighted Flesch-Kincaid grade change across "
            f"{n} changed page{'s' if n != 1 else ''}: "
            f"**{summary['weighted_fk_delta']:+.1f} ({summary['direction']})**."
        )
    else:
        lines.append(
            "No changed page had a scorable before-and-after version, so there "
            "is no readability delta to report. See the per-page details below."
        )
    lines.append("")
    lines.append(REPORT_LEGEND)
    lines.append("")

    # --- Human readability table ---
    lines.append("### Human readability")
    lines.append("")
    lines.append("| Page | FK grade before | FK grade after | FK Δ | Reading ease Δ | Direction |")
    lines.append("|------|-----------------|----------------|------|----------------|-----------|")
    for r in results:
        status = r.get("status")
        path = r.get("path")
        if status == "scored" or status == "new":
            lines.append(
                f"| `{path}` | {_fmt_abs(r.get('fk_before'))} | "
                f"{_fmt_abs(r.get('fk_after'))} | {_fmt(r.get('fk_delta'))} | "
                f"{_fmt(r.get('ease_delta'))} | {r.get('direction')} |"
            )
        elif status == "insufficient_prose":
            lines.append(
                f"| `{path}` | — | — | — | — | too little prose to score |"
            )
        elif status == "deleted":
            lines.append(f"| `{path}` | — | — | — | — | page removed |")

    # --- AI agent comprehension table ---
    if judge_by_path:
        lines.append("")
        lines.append("### AI agent comprehension")
        lines.append("")
        lines.append("Rated 0-3 (higher is easier for an agent to parse and act on).")
        lines.append("")
        lines.append("| Page | Before | After | Δ |")
        lines.append("|------|--------|-------|---|")
        for path, j in judge_by_path.items():
            br = j.get("before_rating")
            ar = j.get("after_rating")
            rd = j.get("rating_delta")
            rd_str = f"{rd:+d}" if isinstance(rd, int) else "—"
            lines.append(
                f"| `{path}` | {br if br is not None else '—'} | "
                f"{ar if ar is not None else '—'} | {rd_str} |"
            )

    # --- Corpus baseline context ---
    baseline_line = _baseline_context_line(baselines)
    if baseline_line:
        lines.append("")
        lines.append(baseline_line)

    # --- Footer ---
    if run_url:
        lines.append("")
        if run_id:
            lines.append(f"*From [workflow run {run_id}]({run_url})*")
        else:
            lines.append(f"*From [workflow run]({run_url})*")

    return "\n".join(lines)


def _baseline_context_line(baselines: Optional[Dict[str, object]]) -> str:
    """One-line median FK grade context per doc type, if baselines are present."""
    if not baselines:
        return ""
    by_type = baselines.get("by_doc_type") or {}
    if not isinstance(by_type, dict) or not by_type:
        overall = baselines.get("overall") or {}
        fk = (overall.get("flesch_kincaid_grade") or {}) if isinstance(overall, dict) else {}
        median = fk.get("median")
        if median is None:
            return ""
        return f"<sub>Corpus baseline: median FK grade {median} across curated docs.</sub>"
    parts = []
    for doc_type in sorted(by_type):
        fk = by_type[doc_type].get("flesch_kincaid_grade") or {}
        median = fk.get("median")
        if median is not None:
            parts.append(f"{doc_type} {median}")
    if not parts:
        return ""
    return "<sub>Curated-docs baseline median FK grade by type: " + ", ".join(parts) + ".</sub>"


def format_pr_body(inner_markdown: str, *, include_marker: bool = True) -> str:
    """Optionally prefix the report with REPORT_MARKER for GitHub upserts."""
    if not include_marker:
        return inner_markdown
    return f"{REPORT_MARKER}\n\n{inner_markdown}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_baselines(path: Optional[Path]) -> Optional[Dict[str, object]]:
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the readability delta PR comment or job-summary Markdown.",
    )
    parser.add_argument("--base", required=True, help="Base commit SHA or ref.")
    parser.add_argument("--head", required=True, help="Head commit SHA or ref.")
    parser.add_argument(
        "--repo-root", type=Path, default=Path("."),
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--skills-scripts", type=Path, default=Path(".claude/scripts"),
        help="Path to the docs-skills scripts dir (the submodule).",
    )
    parser.add_argument(
        "--baselines", type=Path, default=Path(".claude/scripts/readability_baselines.json"),
        help="Path to readability_baselines.json for corpus context.",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Also run the AI comprehension LLM judge (needs WANDB_API_KEY).",
    )
    parser.add_argument("--run-url", default=None, help="Link to the Actions run.")
    parser.add_argument("--run-id", default=None, help="Actions run id for link text.")
    parser.add_argument(
        "--include-marker", action="store_true",
        help="Prefix output with the HTML marker for PR comment upserts.",
    )
    parser.add_argument(
        "--name-status", default=None,
        help="For tests: use this git diff --name-status text instead of running git.",
    )
    args = parser.parse_args()

    readability = _import_readability(args.skills_scripts)

    if args.name_status is not None:
        name_status = args.name_status
    else:
        name_status = changed_mdx_name_status(args.base, args.head, args.repo_root)

    entries = parse_changed_files(name_status)
    results = [
        score_entry(e, args.base, args.head, args.repo_root, readability)
        for e in entries
    ]
    summary = aggregate(results)
    baselines = _load_baselines(args.baselines)

    judge_by_path: Optional[Dict[str, Dict[str, object]]] = None
    if args.judge:
        judge_by_path = {}
        judged = 0
        # Judge the most-changed pages first (by after word count), capped.
        scored_paths = sorted(
            [r for r in results if r.get("status") in ("scored", "new")],
            key=lambda r: int(r.get("after_word_count") or 0),
            reverse=True,
        )
        for r in scored_paths:
            if judged >= MAX_JUDGED_FILES:
                break
            path = str(r["path"])
            old_path = str(r.get("old_path", path))
            status = r.get("status")
            before_text = (
                "" if status == "new"
                else git_show(f"{args.base}:{old_path}", args.repo_root)
            )
            after_text = git_show(f"{args.head}:{path}", args.repo_root)
            j = run_comprehension_judge(args.skills_scripts, before_text, after_text)
            if j is not None:
                judge_by_path[path] = j
                judged += 1
        if not judge_by_path:
            judge_by_path = None

    inner = build_report_markdown(
        results,
        summary,
        baselines=baselines,
        judge_by_path=judge_by_path,
        run_url=args.run_url,
        run_id=args.run_id,
    )
    out = format_pr_body(inner, include_marker=args.include_marker)
    sys.stdout.write(out)
    if not out.endswith("\n"):
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
