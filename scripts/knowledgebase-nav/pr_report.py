#!/usr/bin/env python3
"""
Knowledgebase navigation PR report
===================================

This script turns **what changed after** ``generate_tags.py`` runs into Markdown
for humans. GitHub Actions either posts that Markdown on a pull request or
writes it to the workflow **job summary** when you run the workflow by hand.

Who should read this
----------------------
If you are new to Python, here is the map of this file:

- **Imports** at the top pull in the standard library (``argparse``, ``re``,
  ``subprocess``, and others). ``from __future__ import annotations`` lets us
  write type hints like ``list[str]`` on older Python 3.9+ runtimes in some
  setups; you can ignore it if you are learning the basics.
- **Type hints** (for example ``path: str``, ``-> Optional[...]``) describe what
  each function expects and returns. They help editors and ``mypy`` catch
  mistakes; they do not change how the code runs at runtime.
- **Functions** whose names start with a single underscore (``_is_article_path``)
  are **private helpers** used only inside this file. Public functions have no
  leading underscore.

What this script does (high level)
-----------------------------------
1. After the generator edits files, CI runs ``git diff --name-status HEAD``.
   That compares the **working tree** (files on disk) to **HEAD** (the latest
   commit you checked out, for example the tip of a PR branch). Each line looks
   like ``M\\tsupport/models/articles/foo.mdx`` where the first letter is the
   **change type** (Modified, Added, Deleted, Renamed, and so on).
2. We **parse** those lines and **count** how many paths fall into each bucket
   (articles, tag pages, and others).
3. We collect the page ids of any tag pages that were **added** or
   **removed** so the report can ask a human to update the matching
   ``Support: <display_name>`` tab in ``docs.json``. The generator never edits
   ``docs.json`` itself.
4. We read **stderr** that was saved to ``generator-warnings.log`` and count
   distinct **unknown keyword** warnings so tech writers see how many tags are
   not in ``config.yaml`` yet.
5. We **build Markdown** (plain text with ``#`` headings and bullet lists) and
   print it to **stdout**. The workflow captures that output and posts it.

This file is intentionally **standalone**: it does not import ``generate_tags.py``
so you can test the reporting logic without running the full generator. It
does optionally read ``config.yaml`` to map product slugs to display names so
the docs.json edit instructions can name the right ``Support:`` tab.

See also
--------
- ``generate_tags.py`` for what actually edits support content.
- ``.github/workflows/knowledgebase-nav.yml`` for when ``pr_report.py`` runs.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

# ---------------------------------------------------------------------------
# Constants (shared with tests via REPORT_FALLBACK_BODY and REPORT_MARKER)
# ---------------------------------------------------------------------------

# GitHub issue comments are upserted by searching for this exact HTML comment.
# The workflow finds an existing comment with this substring and PATCHes it,
# or creates a new comment if none exists.
REPORT_MARKER = "<!-- knowledgebase-nav-report -->"

# Visible heading in PR comments and job summaries (sentence case per docs style).
REPORT_TITLE = "## Knowledgebase navigation update"

# Paragraph when there is nothing to report: no file changes vs HEAD,
# no unknown-keyword warnings parsed from the log, and an empty warnings file.
REPORT_FALLBACK_PARAGRAPH = (
    "No updates to support articles, tag pages, or product indexes from this run."
)

# Default location of the generator config (relative to repo root).  Used by
# the docs.json edit section to look up the ``Support: <display_name>`` tab
# for each product slug.  When the config cannot be read the section falls
# back to ``Support: <slug>`` so the report still renders.
DEFAULT_CONFIG_PATH = Path("scripts/knowledgebase-nav/config.yaml")

# Heading and intro paragraph for the "docs.json update required" section.
# The generator never modifies docs.json; humans must edit it after the
# workflow posts this comment.
DOCS_JSON_SECTION_HEADING = "### docs.json update required"
DOCS_JSON_SECTION_INTRO = (
    "A human must update `docs.json` because tag pages were added or removed. "
    "Edit the matching `Support: <display_name>` tab under "
    "`navigation.languages[language=\"en\"].tabs[]`."
)

# Full Markdown for the empty case: title plus fallback paragraph.
REPORT_FALLBACK_BODY = f"{REPORT_TITLE}\n\n{REPORT_FALLBACK_PARAGRAPH}"

# ``re.compile`` builds a reusable pattern. Parentheses in the pattern form a
# **capture group**: ``findall`` returns only the text that matched inside the
# backticks. This mirrors the f-string used in ``generate_tags.build_tag_index``.
_UNKNOWN_KEYWORD_RE = re.compile(
    r"Unknown keyword `([^`]+)`",
)

# GitHub Actions checks out the repo under a path like
# ``/home/runner/work/<repo>/<repo>/``. Strip that prefix from messages so PR
# comments show repo-relative paths when paths appear inside a warning body.
_CI_WORKSPACE_PREFIX_RE = re.compile(r"/home/runner/work/[^/]+/[^/]+/")

# Standard library warning line (see ``warnings._formatwarnmsg`` / default
# ``formatwarning``): ``path:lineno: Category: message``.
_WARNING_LINE_RE = re.compile(
    r"^(.+?):(\d+):\s*([\w]+Warning):\s*(.+)$",
)


# ---------------------------------------------------------------------------
# Formatting captured stderr for pull request comments
# ---------------------------------------------------------------------------


def normalize_ci_paths(text: str) -> str:
    """Remove GitHub Actions workspace prefix from paths embedded in text."""
    return _CI_WORKSPACE_PREFIX_RE.sub("", text)


def format_warnings_for_display(warnings_text: str) -> str:
    """
    Turn raw stderr from the generator into short Markdown list items.

    Python prints each warning as a line ``file:lineno: UserWarning: ...`` and
    often a second indented line showing ``warnings.warn(``. Readers care about
    the message, not the runner-specific absolute path or the scaffolding line.

    Lines that match the standard warning pattern become ``- message`` bullets.
    Continuation lines that only show ``warnings.warn(`` are skipped. Any other
    non-empty lines are kept with CI paths stripped (rare).

    Parameters
    ----------
    warnings_text
        Full text read from ``generator-warnings.log``.

    Returns
    -------
    str
        Markdown bullet list, or empty string if ``warnings_text`` is blank.
    """
    if not warnings_text.strip():
        return ""

    out_lines: List[str] = []
    for line in warnings_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        m = _WARNING_LINE_RE.match(line.rstrip())
        if m:
            msg = normalize_ci_paths(m.group(4).strip())
            out_lines.append(f"- {msg}")
            continue

        # Skip indented continuation (for example ``  warnings.warn(``).
        if line.startswith((" ", "\t")):
            continue

        out_lines.append(f"- {normalize_ci_paths(line.rstrip())}")

    return "\n".join(out_lines).strip()


# ---------------------------------------------------------------------------
# Parsing ``git diff --name-status`` lines
# ---------------------------------------------------------------------------


def parse_name_status_line(line: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    Parse one line produced by ``git diff --name-status``.

    Git separates fields with **tab characters** (``\\t``), not spaces, so file
    names with spaces stay intact.

    Parameters
    ----------
    line
        A single line of diff output, for example ``M\\tpath/to/file.mdx`` or a
        three-field rename line.

    Returns
    -------
    tuple or None
        ``(status, first_path, second_path)``. For a rename or copy (status
        starts with ``R`` or ``C``), ``second_path`` is the new location. For all
        other statuses, ``second_path`` is ``None``. Returns ``None`` for blank
        or malformed lines so callers can skip them safely.
    """
    line = line.rstrip("\n\r")
    if not line.strip():
        return None
    parts = line.split("\t")
    if len(parts) < 2:
        return None
    status = parts[0]
    if status.startswith(("R", "C")):
        if len(parts) < 3:
            return None
        return status, parts[1], parts[2]
    return status, parts[1], None


def _is_article_path(path: str) -> bool:
    """True if ``path`` is ``support/<product>/articles/<name>.mdx``."""
    parts = path.split("/")
    return (
        len(parts) == 4
        and parts[0] == "support"
        and parts[2] == "articles"
        and parts[3].endswith(".mdx")
    )


def _is_tag_page_path(path: str) -> bool:
    """True if ``path`` is ``support/<product>/tags/<slug>.mdx``."""
    parts = path.split("/")
    return (
        len(parts) == 4
        and parts[0] == "support"
        and parts[2] == "tags"
        and parts[3].endswith(".mdx")
    )


def _is_product_index_path(path: str) -> bool:
    """
    True if ``path`` is a product landing page: ``support/<product>.mdx``.

    These live directly under ``support/``, not under ``articles/`` or
    ``tags/``. Example: ``support/models.mdx``.
    """
    parts = path.split("/")
    return len(parts) == 2 and parts[0] == "support" and parts[1].endswith(".mdx")


def _is_root_support_mdx(path: str) -> bool:
    """True if ``path`` is the repo-root ``support.mdx`` (the top-level hub)."""
    return path == "support.mdx"


def _is_deleted_pages_scope(path: str) -> bool:
    """
    True if a **deleted** path should count toward the "pages deleted" bucket.

    We include anything under ``support/``. We do not count deletions only
    under ``scripts/knowledgebase-nav/`` in that bucket so the report stays
    focused on reader-facing content.
    """
    return path.startswith("support/")


def _tag_page_id_from_path(path: str) -> Optional[Tuple[str, str]]:
    """
    Convert a ``support/<product>/tags/<slug>.mdx`` path to a docs.json id.

    Returns
    -------
    tuple of (product_slug, page_id) or None
        ``page_id`` is the docs.json page id form (no ``.mdx`` suffix), for
        example ``support/models/tags/experiments``.  Returns ``None`` if
        ``path`` is not a tag-page filesystem path.
    """
    if not _is_tag_page_path(path):
        return None
    parts = path.split("/")
    product_slug = parts[1]
    file_stem = parts[3][: -len(".mdx")]
    return product_slug, f"support/{product_slug}/tags/{file_stem}"


# ---------------------------------------------------------------------------
# Turning diff lines into numeric buckets
# ---------------------------------------------------------------------------


def categorize_name_status_lines(lines: List[str]) -> Dict[str, int]:
    """
    Count how many files fall into each **report category** from diff lines.

    Git status letters we care about:

    - **M** modified: file existed and changed.
    - **A** added: new file.
    - **D** deleted: file removed.
    - **R** / **C** rename or copy: Git emits the old and new paths. We treat
      tag renames as "one keyword removed, one added" and **do not** also mark
      them as "tag pages modified" (that label is for **M** on tag files only).

    Parameters
    ----------
    lines
        Complete lines from ``git diff --name-status HEAD`` (no shell splitting).

    Returns
    -------
    dict
        String keys (for example ``"articles_badges_updated"``) mapping to
        non-negative integer counts. Every key is always present so callers can
        loop or index without ``KeyError``.
    """
    buckets: Dict[str, int] = {
        "articles_badges_updated": 0,
        "tag_pages_modified": 0,
        "product_index_pages": 0,
        "support_mdx_root": 0,
        "deleted_pages": 0,
        "new_keywords_tag_pages": 0,
        "keywords_no_longer_in_use": 0,
    }

    for raw in lines:
        parsed = parse_name_status_line(raw)
        if parsed is None:
            continue
        status, first, second = parsed
        st = status[0]

        if st in ("R", "C"):
            old_p, new_p = first, second
            assert new_p is not None
            # Treat rename as remove old + add new for tag pages and deletions.
            if _is_deleted_pages_scope(old_p):
                buckets["deleted_pages"] += 1
            if _is_tag_page_path(old_p):
                buckets["keywords_no_longer_in_use"] += 1
            if _is_tag_page_path(new_p):
                buckets["new_keywords_tag_pages"] += 1
            if _is_article_path(new_p):
                buckets["articles_badges_updated"] += 1
            elif _is_product_index_path(new_p):
                buckets["product_index_pages"] += 1
            # Tag renames are counted as new + removed keywords, not as "modified".
            if _is_root_support_mdx(new_p):
                buckets["support_mdx_root"] += 1
            continue

        path = first
        if st == "D":
            if _is_deleted_pages_scope(path):
                buckets["deleted_pages"] += 1
            if _is_tag_page_path(path):
                buckets["keywords_no_longer_in_use"] += 1
        elif st == "A":
            if _is_tag_page_path(path):
                buckets["new_keywords_tag_pages"] += 1
            if _is_article_path(path):
                buckets["articles_badges_updated"] += 1
            elif _is_product_index_path(path):
                buckets["product_index_pages"] += 1
            if _is_root_support_mdx(path):
                buckets["support_mdx_root"] += 1
        elif st == "M":
            if _is_article_path(path):
                buckets["articles_badges_updated"] += 1
            elif _is_tag_page_path(path):
                buckets["tag_pages_modified"] += 1
            elif _is_product_index_path(path):
                buckets["product_index_pages"] += 1
            if _is_root_support_mdx(path):
                buckets["support_mdx_root"] += 1

    return buckets


def collect_tag_page_changes(
    lines: List[str],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Collect tag page additions and removals grouped by product slug.

    Walks ``git diff --name-status HEAD`` lines and returns two dicts mapping
    product slug to the docs.json **page ids** (no ``.mdx`` suffix) of the
    affected tag pages.  Renames produce one entry in each dict (the new path
    is added, the old path is removed).

    Parameters
    ----------
    lines
        Complete lines from ``git diff --name-status HEAD``.

    Returns
    -------
    tuple of (added, removed)
        ``added`` maps product slug to a sorted list of new tag page ids.
        ``removed`` maps product slug to a sorted list of removed tag page
        ids.  Each list is deduplicated and sorted alphabetically so the
        output of the report is deterministic.
    """
    added: Dict[str, set] = {}
    removed: Dict[str, set] = {}

    def _add(bucket: Dict[str, set], product: str, page_id: str) -> None:
        bucket.setdefault(product, set()).add(page_id)

    for raw in lines:
        parsed = parse_name_status_line(raw)
        if parsed is None:
            continue
        status, first, second = parsed
        st = status[0]

        if st in ("R", "C"):
            assert second is not None
            old_info = _tag_page_id_from_path(first)
            new_info = _tag_page_id_from_path(second)
            if old_info is not None:
                _add(removed, *old_info)
            if new_info is not None:
                _add(added, *new_info)
            continue

        if st == "A":
            info = _tag_page_id_from_path(first)
            if info is not None:
                _add(added, *info)
        elif st == "D":
            info = _tag_page_id_from_path(first)
            if info is not None:
                _add(removed, *info)

    return (
        {slug: sorted(ids) for slug, ids in added.items()},
        {slug: sorted(ids) for slug, ids in removed.items()},
    )


# ---------------------------------------------------------------------------
# Loading product display names from config.yaml
# ---------------------------------------------------------------------------


def load_product_display_names(config_path: Path) -> Dict[str, str]:
    """
    Read product slug -> display name mapping from ``config.yaml``.

    The mapping is used to build the ``Support: <display_name>`` heading in
    the docs.json edit section.  Returns an empty dict if the file is
    missing, unreadable, or malformed; the caller falls back to the slug.

    Parameters
    ----------
    config_path
        Path to ``scripts/knowledgebase-nav/config.yaml`` (or any file with
        the same shape).

    Returns
    -------
    dict
        Mapping of product slug to display name.  Products missing either
        ``slug`` or ``display_name`` are skipped silently.
    """
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}

    products = config.get("products") if isinstance(config, dict) else None
    if not isinstance(products, list):
        return {}

    mapping: Dict[str, str] = {}
    for product in products:
        if not isinstance(product, dict):
            continue
        slug = product.get("slug")
        display_name = product.get("display_name")
        if isinstance(slug, str) and isinstance(display_name, str):
            mapping[slug] = display_name
    return mapping


def distinct_unknown_keywords_from_warnings(warnings_text: str) -> Set[str]:
    """
    Collect distinct keyword strings from "Unknown keyword" warning lines.

    Expects ``generate_tags`` format: ``Unknown keyword `tag` used in ...``.

    The generator may emit the same unknown tag for several articles; we count
    **distinct** tags so the PR summary matches how many new ``config.yaml``
    entries are needed.

    Parameters
    ----------
    warnings_text
        Full text of ``generator-warnings.log`` (or any string containing the
        same warning format).

    Returns
    -------
    set of str
        Unique keyword strings, with no guaranteed ordering.
    """
    return set(_UNKNOWN_KEYWORD_RE.findall(warnings_text))


# ---------------------------------------------------------------------------
# Building Markdown for GitHub
# ---------------------------------------------------------------------------


def build_docs_json_section(
    added_tag_pages: Dict[str, List[str]],
    removed_tag_pages: Dict[str, List[str]],
    display_names: Dict[str, str],
) -> str:
    """
    Build the "docs.json update required" Markdown section.

    Returns an empty string when both ``added_tag_pages`` and
    ``removed_tag_pages`` are empty.  Otherwise produces a heading
    (``DOCS_JSON_SECTION_HEADING``), a one-paragraph intro
    (``DOCS_JSON_SECTION_INTRO``), and one subsection per affected product
    (sorted by slug) listing entries to add to and remove from the
    product's ``Support: <display_name>`` ``pages`` array.

    Each per-product subsection uses the form::

        #### Support: W&B Models

        Add to `pages`:
        - `support/models/tags/foo`

        Remove from `pages`:
        - `support/models/tags/old`

    Either "Add" or "Remove" subsections are omitted when their list is
    empty for that product.

    Parameters
    ----------
    added_tag_pages
        Mapping of product slug to a sorted list of newly added tag page
        ids (page ids, not filesystem paths).
    removed_tag_pages
        Mapping of product slug to a sorted list of removed tag page ids.
    display_names
        Mapping of product slug to display name (from ``config.yaml``).
        Slugs not present in this mapping fall back to ``Support: <slug>``
        so the section still renders.

    Returns
    -------
    str
        Markdown for the docs.json edit section, without a leading or
        trailing blank line.  Empty string when there are no changes.
    """
    if not added_tag_pages and not removed_tag_pages:
        return ""

    lines: List[str] = [DOCS_JSON_SECTION_HEADING, "", DOCS_JSON_SECTION_INTRO]

    product_slugs = sorted(set(added_tag_pages) | set(removed_tag_pages))
    for slug in product_slugs:
        display_name = display_names.get(slug, slug)
        lines.append("")
        lines.append(f"#### Support: {display_name}")

        added = added_tag_pages.get(slug, [])
        removed = removed_tag_pages.get(slug, [])

        if added:
            lines.append("")
            lines.append("Add to `pages`:")
            for page_id in added:
                lines.append(f"- `{page_id}`")
        if removed:
            lines.append("")
            lines.append("Remove from `pages`:")
            for page_id in removed:
                lines.append(f"- `{page_id}`")

    return "\n".join(lines)


def build_report_markdown(
    buckets: Dict[str, int],
    unknown_keywords: Set[str],
    warnings_text: str,
    run_url: Optional[str] = None,
    run_id: Optional[str] = None,
    added_tag_pages: Optional[Dict[str, List[str]]] = None,
    removed_tag_pages: Optional[Dict[str, List[str]]] = None,
    display_names: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build the Markdown body **without** the HTML comment marker.

    If every bucket is zero, there are no unknown keywords, no added or
    removed tag pages, and the warnings text is blank, returns
    ``REPORT_FALLBACK_BODY`` (``REPORT_TITLE`` plus
    ``REPORT_FALLBACK_PARAGRAPH``).

    Otherwise builds a short report starting with ``REPORT_TITLE``, then bullet
    lines (counts only), optional "no categorized file changes" when warnings
    exist but nothing matched our path rules, an optional "docs.json update
    required" section listing the exact tag page ids a human must add to or
    remove from each ``Support:`` tab, a fenced code block with the raw
    warnings log if present, and an optional footer link when ``run_url`` is
    set (with ``run_id`` in the link text when provided).

    Parameters
    ----------
    buckets
        Counts from :func:`categorize_name_status_lines`.
    unknown_keywords
        Distinct unknown tags from :func:`distinct_unknown_keywords_from_warnings`.
    warnings_text
        Raw stderr captured from the generator. The Generator warnings section
        uses :func:`format_warnings_for_display` when it can parse standard
        warning lines; otherwise it falls back to a fenced copy of the raw log.
    run_url
        If set, appended as a footer link to the Actions run.
    run_id
        GitHub Actions ``GITHUB_RUN_ID`` (same number as in ``/actions/runs/<id>``).
        If set with ``run_url``, the link text is ``workflow run <id>`` so the
        number is visible without hovering.
    added_tag_pages, removed_tag_pages
        Tag page ids grouped by product slug, from
        :func:`collect_tag_page_changes`.  When either is non-empty, the
        report includes a "docs.json update required" section.
    display_names
        Product slug to display-name mapping (see
        :func:`load_product_display_names`).  Used to label
        ``Support: <display_name>`` subsections.

    Returns
    -------
    str
        Markdown suitable for a PR comment body or job summary (without marker).
    """
    lines_out: List[str] = []

    added_tag_pages = added_tag_pages or {}
    removed_tag_pages = removed_tag_pages or {}
    display_names = display_names or {}

    uk_count = len(unknown_keywords)
    if (
        not any(buckets[k] > 0 for k in buckets)
        and uk_count == 0
        and not warnings_text.strip()
        and not added_tag_pages
        and not removed_tag_pages
    ):
        return REPORT_FALLBACK_BODY

    lines_out.append(REPORT_TITLE)
    lines_out.append("")

    added_bullet = False
    if buckets["articles_badges_updated"] > 0:
        n = buckets["articles_badges_updated"]
        lines_out.append(
            f"- Articles with Badges updated: {n} article{'s' if n != 1 else ''}."
        )
        added_bullet = True
    if buckets["tag_pages_modified"] > 0:
        n = buckets["tag_pages_modified"]
        lines_out.append(
            f"- Tag pages modified: {n} page{'s' if n != 1 else ''}."
        )
        added_bullet = True
    if buckets["product_index_pages"] > 0:
        n = buckets["product_index_pages"]
        lines_out.append(
            f"- Product index pages updated: {n} page{'s' if n != 1 else ''}."
        )
        added_bullet = True
    if buckets["support_mdx_root"] > 0:
        lines_out.append("- Root support.mdx updated.")
        added_bullet = True
    if buckets["deleted_pages"] > 0:
        n = buckets["deleted_pages"]
        lines_out.append(f"- Pages deleted: {n} page{'s' if n != 1 else ''}.")
        added_bullet = True
    if buckets["new_keywords_tag_pages"] > 0:
        n = buckets["new_keywords_tag_pages"]
        lines_out.append(
            f"- New keywords (new tag pages): {n} keyword{'s' if n != 1 else ''}."
        )
        added_bullet = True
    if buckets["keywords_no_longer_in_use"] > 0:
        n = buckets["keywords_no_longer_in_use"]
        lines_out.append(
            f"- Keywords no longer in use (tag pages removed): {n} keyword{'s' if n != 1 else ''}."
        )
        added_bullet = True
    if uk_count > 0:
        lines_out.append(
            f"- Keywords not on the allowed list: {uk_count} distinct keyword{'s' if uk_count != 1 else ''}."
        )
        added_bullet = True

    if not added_bullet and warnings_text.strip():
        lines_out.append("- (No categorized file changes.)")

    docs_json_section = build_docs_json_section(
        added_tag_pages, removed_tag_pages, display_names
    )
    if docs_json_section:
        lines_out.append("")
        lines_out.append(docs_json_section)

    if warnings_text.strip():
        lines_out.append("")
        lines_out.append("### Generator warnings")
        lines_out.append("")
        formatted_warnings = format_warnings_for_display(warnings_text)
        if formatted_warnings.strip():
            lines_out.append(formatted_warnings)
        else:
            lines_out.append("```")
            lines_out.append(warnings_text.rstrip("\n"))
            lines_out.append("```")

    if run_url:
        lines_out.append("")
        if run_id:
            lines_out.append(f"*From [workflow run {run_id}]({run_url})*")
        else:
            lines_out.append(f"*From [workflow run]({run_url})*")

    return "\n".join(lines_out)


def format_pr_body(
    inner_markdown: str,
    *,
    include_marker: bool = True,
) -> str:
    """
    Optionally prefix the report with ``REPORT_MARKER`` for GitHub upserts.

    The ``*`` before ``include_marker`` means **keyword-only**: callers must
    pass ``include_marker=...`` by name, which keeps the optional flag obvious
    at call sites and avoids mixing up argument order.

    Parameters
    ----------
    inner_markdown
        Output of :func:`build_report_markdown`.
    include_marker
        When ``True`` (default), prepend the HTML comment GitHub searches for.

    Returns
    -------
    str
        Text ready to POST or PATCH as an issue comment body.
    """
    if not include_marker:
        return inner_markdown
    return f"{REPORT_MARKER}\n\n{inner_markdown}"


def git_diff_name_status_head(repo_root: Path) -> str:
    """
    Run ``git diff --name-status HEAD`` in ``repo_root`` and return stdout.

    **HEAD** means the commit currently checked out. Uncommitted edits from the
    generator show up as differences versus that commit.

    Raises
    ------
    SystemExit
        If Git returns a non-zero exit code (for example not a Git repository).

    Returns
    -------
    str
        Raw diff text, possibly empty if the working tree matches HEAD.
    """
    result = subprocess.run(
        ["git", "diff", "--name-status", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode or 1)
    return result.stdout


def main() -> None:
    """
    Command-line entry point used by GitHub Actions and local debugging.

    Reads flags, loads the warnings file if present, builds Markdown, prints to
    **stdout** (the shell redirects that to a file in CI).
    """
    parser = argparse.ArgumentParser(
        description="Build Knowledgebase Nav PR comment or job summary Markdown.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Repository root (contains .git).",
    )
    parser.add_argument(
        "--warnings-file",
        type=Path,
        default=Path("generator-warnings.log"),
        help="Path to captured stderr from generate_tags.py.",
    )
    parser.add_argument(
        "--run-url",
        default=None,
        help="Link to the GitHub Actions workflow run (optional).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Workflow run id (GITHUB_RUN_ID) for the footer link text (optional)."
        ),
    )
    parser.add_argument(
        "--include-marker",
        action="store_true",
        help="Prefix output with the HTML marker for PR comment upserts.",
    )
    parser.add_argument(
        "--diff-text",
        type=str,
        default=None,
        help="For tests: use this text instead of running git diff.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to scripts/knowledgebase-nav/config.yaml. "
            "Used to map product slugs to display names in the docs.json edit "
            "section. Defaults to <repo-root>/scripts/knowledgebase-nav/config.yaml."
        ),
    )
    args = parser.parse_args()

    if args.diff_text is not None:
        diff_out = args.diff_text
    else:
        diff_out = git_diff_name_status_head(args.repo_root)

    diff_lines = [ln for ln in diff_out.splitlines() if ln.strip()]
    buckets = categorize_name_status_lines(diff_lines)
    added_tag_pages, removed_tag_pages = collect_tag_page_changes(diff_lines)

    warnings_text = ""
    if args.warnings_file.exists():
        warnings_text = args.warnings_file.read_text(encoding="utf-8")

    config_path = args.config or (args.repo_root / DEFAULT_CONFIG_PATH)
    display_names = load_product_display_names(config_path)

    unknown_set = distinct_unknown_keywords_from_warnings(warnings_text)
    inner = build_report_markdown(
        buckets,
        unknown_set,
        warnings_text,
        run_url=args.run_url,
        run_id=args.run_id,
        added_tag_pages=added_tag_pages,
        removed_tag_pages=removed_tag_pages,
        display_names=display_names,
    )
    out = format_pr_body(inner, include_marker=args.include_marker)
    sys.stdout.write(out)
    if not out.endswith("\n"):
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
