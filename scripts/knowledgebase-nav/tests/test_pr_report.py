"""
Unit tests for ``pr_report.py`` (Knowledgebase navigation PR report).

What these tests do
--------------------
They check **pure functions**: parsing fake ``git diff`` lines, bucketing paths
into counts, extracting unknown keywords from warning text, and building
Markdown. No real Git repository is required for most tests.

If you are new to **pytest**:

- A file named ``test_*.py`` is discovered automatically.
- A function named ``test_*`` is one test case.
- ``assert`` compares values; if the comparison fails, pytest prints a clear
  diff. There is no need to import a special ``assertEqual`` helper.

How ``pr_report`` is imported
-----------------------------
The script ``pr_report.py`` lives in ``scripts/knowledgebase-nav/``, not in an
installed Python package. The block below adds that directory to ``sys.path`` so
``import pr_report`` works when pytest runs from the repo root. The
``# noqa: E402`` comment tells the linter not to complain that imports are not
at the very top of the file (they must come after ``sys.path`` is updated).

Run with::

    pytest scripts/knowledgebase-nav/tests/test_pr_report.py -v
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import the module under test (see module docstring above).
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import pr_report  # noqa: E402


def test_parse_name_status_simple():
    """
    ``parse_name_status_line`` should split Modified and Deleted lines.

    Tab-separated fields match what Git prints; the third tuple element is
    ``None`` when there is only one path.
    """
    assert pr_report.parse_name_status_line("M\tsupport/models/articles/a.mdx") == (
        "M",
        "support/models/articles/a.mdx",
        None,
    )
    assert pr_report.parse_name_status_line("D\tsupport/models/tags/old.mdx") == (
        "D",
        "support/models/tags/old.mdx",
        None,
    )


def test_parse_rename():
    """
    Rename lines include **two** paths; status may include a similarity score.

    Example prefix ``R095`` means roughly 95 percent similar; we only need the
    leading ``R`` elsewhere in the code.
    """
    line = "R095\tsupport/models/tags/a.mdx\tsupport/models/tags/b.mdx"
    assert pr_report.parse_name_status_line(line) == (
        "R095",
        "support/models/tags/a.mdx",
        "support/models/tags/b.mdx",
    )


def test_categorize_article_modified():
    """
    A modified article under ``support/.../articles/`` counts toward badge sync.

    Only the ``articles_badges_updated`` bucket should increment for this line.
    """
    lines = ["M\tsupport/models/articles/foo.mdx"]
    b = pr_report.categorize_name_status_lines(lines)
    assert b["articles_badges_updated"] == 1
    assert b["tag_pages_modified"] == 0


def test_categorize_tag_added_deleted_modified():
    """
    Added tag pages, deleted tag pages, and modified tag pages use different buckets.

    - **A** on a tag file counts as a **new** keyword (new tag page).
    - **D** counts as a keyword **no longer in use**.
    - **M** counts as **tag pages modified** (content change, not add or remove).
    """
    b = pr_report.categorize_name_status_lines(
        [
            "A\tsupport/models/tags/new-tag.mdx",
            "D\tsupport/models/tags/gone.mdx",
            "M\tsupport/models/tags/existing.mdx",
        ]
    )
    assert b["new_keywords_tag_pages"] == 1
    assert b["keywords_no_longer_in_use"] == 1
    assert b["tag_pages_modified"] == 1


def test_categorize_product_index_and_support_root():
    """
    Product indexes, root ``support.mdx``, and ``docs.json`` each have their own flags.

    ``support/models.mdx`` is two path segments; the root ``support.mdx`` is a
    single file at the repository root, not inside ``support/``.
    """
    b = pr_report.categorize_name_status_lines(
        [
            "M\tsupport/models.mdx",
            "M\tsupport.mdx",
            "M\tdocs.json",
        ]
    )
    assert b["product_index_pages"] == 1
    assert b["support_mdx_root"] == 1
    assert b["docs_json"] == 1


def test_categorize_deleted_pages():
    """A deleted article increments the general **deleted_pages** counter."""
    b = pr_report.categorize_name_status_lines(
        ["D\tsupport/models/articles/deleted.mdx"]
    )
    assert b["deleted_pages"] == 1


def test_rename_tag_does_not_increment_tag_modified():
    """
    Renaming a tag file is **not** the same as modifying it in place.

    We expect one "new keyword" (new path) and one "no longer in use" (old
    path), but **zero** for ``tag_pages_modified`` (that bucket is for **M**
    only on tag files).
    """
    lines = [
        "R100\tsupport/models/tags/a.mdx\tsupport/models/tags/b.mdx",
    ]
    b = pr_report.categorize_name_status_lines(lines)
    assert b["tag_pages_modified"] == 0
    assert b["new_keywords_tag_pages"] == 1
    assert b["keywords_no_longer_in_use"] == 1


def test_unknown_keyword_distinct():
    """
    The same unknown keyword repeated across lines should dedupe to one entry.

    ``distinct_unknown_keywords_from_warnings`` returns a **set**, which is an
    unordered collection of unique items.
    """
    text = """
    /path/gen.py:1: UserWarning: Unknown keyword `Foo` used in `support/x/a.mdx`.
    /path/gen.py:2: UserWarning: Unknown keyword `Foo` used in `support/x/b.mdx`.
    /path/gen.py:3: UserWarning: Unknown keyword `Bar` used in `support/x/c.mdx`.
    """
    s = pr_report.distinct_unknown_keywords_from_warnings(text)
    assert s == {"Foo", "Bar"}


def test_build_report_fallback():
    """
    With empty buckets, no unknown keywords, and no warnings text, output is the fallback constant.

    We build an all-zero bucket dict by running ``categorize_name_status_lines``
    on an empty list so keys always match production.
    """
    empty = {k: 0 for k in pr_report.categorize_name_status_lines([])}
    out = pr_report.build_report_markdown(empty, set(), "")
    assert out == pr_report.REPORT_FALLBACK_BODY


def test_build_report_footer_includes_run_id():
    """
    The footer should embed ``GITHUB_RUN_ID`` in the Markdown link label.

    The URL still points at the run; the id matches the ``/actions/runs/<id>`` segment.
    """
    empty = {k: 0 for k in pr_report.categorize_name_status_lines([])}
    empty["docs_json"] = 1
    out = pr_report.build_report_markdown(
        empty,
        set(),
        "",
        run_url="https://github.com/org/repo/actions/runs/12345",
        run_id="12345",
    )
    assert "*From [workflow run 12345](https://github.com/org/repo/actions/runs/12345)*" in out


def test_build_report_with_counts():
    """
    Non-zero buckets should appear as bullet lines under ``REPORT_TITLE``.

    We mutate a fresh zero dict rather than hard-coding every key so new
    bucket keys in production are picked up automatically.
    """
    empty = {k: 0 for k in pr_report.categorize_name_status_lines([])}
    empty["articles_badges_updated"] = 2
    empty["docs_json"] = 1
    out = pr_report.build_report_markdown(empty, set(), "")
    assert pr_report.REPORT_TITLE in out
    assert "Articles with tab Badges updated: 2 articles" in out
    assert "docs.json updated" in out


def test_format_warnings_for_display_strips_path_and_scaffolding():
    """
    CI stderr includes a long path and a second ``warnings.warn(`` line.

    Output should be a single Markdown bullet with only the UserWarning message.
    """
    raw = (
        "/home/runner/work/docs/docs/scripts/knowledgebase-nav/generate_tags.py:"
        "969: UserWarning: Unknown keyword `foobar` used in "
        "`support/models/articles/adding-multiple-authors-to-a-report.mdx`. "
        "Add it to `scripts/knowledgebase-nav/config.yaml` to suppress this warning.\n"
        "  warnings.warn(\n"
    )
    out = pr_report.format_warnings_for_display(raw)
    assert "home/runner" not in out
    assert "warnings.warn" not in out
    assert "Unknown keyword `foobar`" in out
    assert out.startswith("- ")


def test_format_pr_body_includes_marker():
    """
    PR comments need the HTML marker so the workflow can find and update them.

    ``include_marker=True`` prepends ``REPORT_MARKER`` followed by a blank line.
    The inner body includes ``REPORT_TITLE`` and the fallback paragraph.
    """
    inner = pr_report.REPORT_FALLBACK_BODY
    body = pr_report.format_pr_body(inner, include_marker=True)
    assert pr_report.REPORT_MARKER in body
    assert pr_report.REPORT_FALLBACK_BODY in body
    assert pr_report.REPORT_TITLE in body
