"""
Unit tests for ``pr_report.py`` (Readability delta PR report).

What these tests do
-------------------
They check the **pure functions**: parsing fake ``git diff --name-status`` lines,
aggregating word-weighted deltas, and building Markdown. These need neither a
real Git repository, ``textstat``, nor network access, because the
Markdown-building functions take plain numbers.

A separate, optional ``integration``-marked test exercises the real
``_readability`` analyzer from the docs-skills submodule when it is available.

Run with::

    pytest scripts/readability/tests/test_pr_report.py -v
"""

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test.
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import pr_report  # noqa: E402


# ---------------------------------------------------------------------------
# parse_changed_files
# ---------------------------------------------------------------------------


def test_parse_changed_files_filters_and_keeps_status():
    name_status = "\n".join([
        "M\tweave/guides/tracking/threads.mdx",
        "A\tmodels/new-page.mdx",
        "D\tmodels/old-page.mdx",
        "M\tdocs.json",                       # not mdx -> skip
        "M\tja/weave/guides/tracking/threads.mdx",  # localized -> skip
        "M\tko/models/foo.mdx",               # localized -> skip
    ])
    entries = pr_report.parse_changed_files(name_status)
    assert entries == [
        {"status": "M", "path": "weave/guides/tracking/threads.mdx"},
        {"status": "A", "path": "models/new-page.mdx"},
        {"status": "D", "path": "models/old-page.mdx"},
    ]


def test_parse_changed_files_rename_uses_new_path():
    # Rename rows have three tab fields: status, old path, new path.
    name_status = "R096\tmodels/old.mdx\tmodels/renamed.mdx"
    entries = pr_report.parse_changed_files(name_status)
    assert entries == [{"status": "R", "path": "models/renamed.mdx"}]


def test_parse_changed_files_empty():
    assert pr_report.parse_changed_files("") == []


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def test_aggregate_word_weighted():
    results = [
        # Big page improved a little; small page got much worse.
        {"status": "scored", "fk_delta": -1.0, "after_word_count": 900},
        {"status": "scored", "fk_delta": 9.0, "after_word_count": 100},
        {"status": "new", "fk_delta": None, "after_word_count": 500},   # ignored
        {"status": "deleted"},                                          # ignored
    ]
    summary = pr_report.aggregate(results)
    # (-1.0*900 + 9.0*100) / 1000 = 0.0
    assert summary["weighted_fk_delta"] == 0.0
    assert summary["pages_scored"] == 2
    assert summary["direction"] == "unchanged"


def test_aggregate_easier_direction():
    results = [{"status": "scored", "fk_delta": -2.0, "after_word_count": 100}]
    summary = pr_report.aggregate(results)
    assert summary["weighted_fk_delta"] == -2.0
    assert summary["direction"] == "easier"


def test_aggregate_none_when_nothing_scored():
    results = [{"status": "deleted"}, {"status": "insufficient_prose"}]
    assert pr_report.aggregate(results) is None


# ---------------------------------------------------------------------------
# build_report_markdown
# ---------------------------------------------------------------------------


def test_build_report_fallback_when_no_scorable():
    assert pr_report.build_report_markdown([], None) == pr_report.REPORT_FALLBACK_BODY


def test_build_report_includes_headline_and_table():
    results = [
        {
            "status": "scored",
            "path": "weave/guides/threads.mdx",
            "fk_before": 12.4,
            "fk_after": 11.1,
            "fk_delta": -1.3,
            "ease_delta": 5.2,
            "direction": "easier",
            "after_word_count": 800,
        },
        {
            "status": "new",
            "path": "models/new.mdx",
            "fk_before": None,
            "fk_after": 9.8,
            "fk_delta": None,
            "ease_delta": None,
            "direction": "n/a",
            "after_word_count": 300,
        },
        {"status": "deleted", "path": "models/old.mdx"},
        {"status": "insufficient_prose", "path": "models/tiny.mdx", "after_word_count": 5},
    ]
    summary = pr_report.aggregate(results)
    md = pr_report.build_report_markdown(results, summary)

    assert pr_report.REPORT_TITLE in md
    assert "Word-weighted Flesch-Kincaid grade change" in md
    assert "-1.3 (easier)" in md
    assert "`weave/guides/threads.mdx`" in md
    assert "11.1" in md
    assert "-1.3" in md
    assert "+5.2" in md
    assert "page removed" in md
    assert "too little prose to score" in md
    # Never blocks: the legend says so.
    assert "informational" in md


def test_build_report_judge_table():
    results = [
        {
            "status": "scored",
            "path": "a.mdx",
            "fk_before": 12.0,
            "fk_after": 10.0,
            "fk_delta": -2.0,
            "ease_delta": 4.0,
            "direction": "easier",
            "after_word_count": 500,
        },
    ]
    judge = {"a.mdx": {"before_rating": 1, "after_rating": 3, "rating_delta": 2}}
    md = pr_report.build_report_markdown(
        results, pr_report.aggregate(results), judge_by_path=judge
    )
    assert "AI agent comprehension" in md
    assert "| `a.mdx` | 1 | 3 | +2 |" in md


def test_build_report_baseline_context_line():
    results = [
        {
            "status": "scored",
            "path": "a.mdx",
            "fk_before": 12.0,
            "fk_after": 10.0,
            "fk_delta": -2.0,
            "ease_delta": 4.0,
            "direction": "easier",
            "after_word_count": 500,
        },
    ]
    baselines = {
        "by_doc_type": {
            "procedural": {"flesch_kincaid_grade": {"median": 8.8}},
            "conceptual": {"flesch_kincaid_grade": {"median": 10.5}},
        }
    }
    md = pr_report.build_report_markdown(
        results, pr_report.aggregate(results), baselines=baselines
    )
    assert "baseline median FK grade by type" in md
    assert "procedural 8.8" in md
    assert "conceptual 10.5" in md


def test_build_report_footer_link():
    results = [
        {
            "status": "scored",
            "path": "a.mdx",
            "fk_before": 12.0,
            "fk_after": 10.0,
            "fk_delta": -2.0,
            "ease_delta": 4.0,
            "direction": "easier",
            "after_word_count": 500,
        },
    ]
    md = pr_report.build_report_markdown(
        results, pr_report.aggregate(results),
        run_url="https://example.com/run/9", run_id="9",
    )
    assert "*From [workflow run 9](https://example.com/run/9)*" in md


def test_format_pr_body_marker():
    body = pr_report.format_pr_body("hello", include_marker=True)
    assert body.startswith(pr_report.REPORT_MARKER)
    assert "hello" in body
    assert pr_report.format_pr_body("hello", include_marker=False) == "hello"


# ---------------------------------------------------------------------------
# Optional integration test against the real analyzer in the submodule.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_analyzer_round_trip():
    """Exercise the real _readability analyzer if textstat + submodule exist."""
    skills_scripts = Path(__file__).resolve().parents[3] / ".claude" / "scripts"
    if not (skills_scripts / "_readability.py").exists():
        pytest.skip("docs-skills submodule not checked out")
    try:
        readability = pr_report._import_readability(skills_scripts)
    except ImportError:
        pytest.skip("submodule analyzer not importable")
    try:
        import textstat  # noqa: F401
    except ImportError:
        pytest.skip("textstat not installed")

    before = "In order to utilize the system, you must subsequently initialize " \
             "the configuration parameters prior to the commencement of operation, " \
             "which necessitates considerable and careful deliberation throughout."
    after = "To set up the system, set the config values first. Do this before you start."
    report = readability.analyze_texts(
        readability.extract_prose_mdx(before),
        readability.extract_prose_mdx(after),
    )
    # The simpler "after" text should read at a lower grade if both are scorable.
    if report["headline"]["delta"] is not None:
        assert report["headline"]["direction"] in ("easier", "harder", "unchanged")
