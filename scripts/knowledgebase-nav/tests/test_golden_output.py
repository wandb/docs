"""
Golden-file integration test for the Knowledgebase Nav generator.

This is the most important test in the suite.  It runs the generator against
the real wandb-docs repository and verifies that the output is **byte-for-byte
identical** to the existing files on disk.

If the template, slug logic, body-preview truncation, sort order, or
navigation structure drifts even by one character, this test catches it
and reports the exact difference as a unified diff.

What it tests:
    - Every tag page: support/<product>/tags/<tag>.mdx
    - Product index pages: support/<product>.mdx (for example support/models.mdx)
    - Article footers: support/<product>/articles/*.mdx (tab-page Badge sync)
    - docs.json support navigation tabs (page lists, tab names, ordering)
    - Root support.mdx (product card count lines)

How to run:
    pytest scripts/knowledgebase-nav/tests/test_golden_output.py -v -m integration

The ``integration`` marker is registered in ``conftest.py`` in this directory.

Prerequisites:
    - The wandb-docs repo must be present at the expected location.
    - The test reads existing files as the "expected" golden output, then
      generates new output to a temporary directory and compares.
"""

import difflib
import json
import shutil
import textwrap
from pathlib import Path

import pytest

import sys

_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import generate_tags  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Repo root: parent of scripts/ when this file lives at
# scripts/knowledgebase-nav/tests/test_golden_output.py
WANDB_DOCS_ROOT = Path(__file__).resolve().parent.parent.parent.parent

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
_config = generate_tags.load_config(CONFIG_PATH)
PRODUCTS = [p["slug"] for p in _config["products"]]

# Mark all tests in this module as integration tests so they can be
# run separately from the fast unit tests.
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_file(path: Path) -> str:
    """Read a file and return its contents as a string."""
    return path.read_text(encoding="utf-8")


def _unified_diff(expected: str, actual: str, label: str) -> str:
    """
    Produce a unified diff between expected and actual content.

    Returns an empty string if the files are identical, or a multi-line
    diff string showing exactly what changed.

    Parameters
    ----------
    expected : str
        The original (golden) file content.
    actual : str
        The generated file content.
    label : str
        A label for the file (used in diff headers).

    Returns
    -------
    str
        The unified diff, or empty string if identical.
    """
    expected_lines = expected.splitlines(keepends=True)
    actual_lines = actual.splitlines(keepends=True)
    diff = difflib.unified_diff(
        expected_lines,
        actual_lines,
        fromfile=f"expected: {label}",
        tofile=f"generated: {label}",
    )
    return "".join(diff)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def golden_repo_root():
    """
    Return the path to the real wandb-docs repo.

    Skips if docs.json is missing at the resolved root (for example a
    sparse checkout or wrong working directory).
    """
    if not (WANDB_DOCS_ROOT / "docs.json").exists():
        pytest.skip(
            f"wandb-docs repo not found at {WANDB_DOCS_ROOT}. "
            "Skipping golden-file tests."
        )
    return WANDB_DOCS_ROOT


@pytest.fixture(scope="module")
def generated_output(golden_repo_root, tmp_path_factory):
    """
    Copy the repo to a temp directory, run the generator, and return
    a dict with the generated file contents and the temp root path.

    This fixture runs once per test module (scope="module") to avoid
    redundant generation.  All test functions share the same generated
    output.

    Returns
    -------
    dict with keys:
        - "root": Path to the temporary repo copy
        - "tag_pages": dict mapping relative path to file content
        - "product_indexes": dict mapping support/<slug>.mdx to file content
        - "article_files": dict mapping relative path to file content
        - "docs_json": the parsed docs.json dict
        - "support_mdx": full text of generated support.mdx
    """
    tmp_root = tmp_path_factory.mktemp("golden")

    # Copy the support/ directory, support.mdx, and docs.json to the temp
    # location.  We only need these for the generator to work.
    shutil.copytree(
        golden_repo_root / "support",
        tmp_root / "support",
    )
    shutil.copy2(
        golden_repo_root / "docs.json",
        tmp_root / "docs.json",
    )
    shutil.copy2(
        golden_repo_root / "support.mdx",
        tmp_root / "support.mdx",
    )

    # Run the generator against the temp copy
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    generate_tags.run_pipeline(tmp_root, config_path)

    # Collect generated tag page contents
    tag_pages = {}
    for product in PRODUCTS:
        tags_dir = tmp_root / "support" / product / "tags"
        if tags_dir.exists():
            for mdx_file in sorted(tags_dir.glob("*.mdx")):
                rel_path = f"support/{product}/tags/{mdx_file.name}"
                tag_pages[rel_path] = mdx_file.read_text(encoding="utf-8")

    # Collect generated article files (footer sync may have modified them)
    article_files = {}
    for product in PRODUCTS:
        articles_dir = tmp_root / "support" / product / "articles"
        if articles_dir.exists():
            for mdx_file in sorted(articles_dir.glob("*.mdx")):
                rel_path = f"support/{product}/articles/{mdx_file.name}"
                article_files[rel_path] = mdx_file.read_text(encoding="utf-8")

    # Read the generated docs.json
    docs_json = json.loads((tmp_root / "docs.json").read_text(encoding="utf-8"))

    # Read the generated support.mdx
    support_mdx = (tmp_root / "support.mdx").read_text(encoding="utf-8")

    # Product hub pages at support/<slug>.mdx
    product_indexes = {}
    for product in PRODUCTS:
        idx_path = tmp_root / "support" / f"{product}.mdx"
        if idx_path.exists():
            rel = f"support/{product}.mdx"
            product_indexes[rel] = idx_path.read_text(encoding="utf-8")

    return {
        "root": tmp_root,
        "tag_pages": tag_pages,
        "product_indexes": product_indexes,
        "article_files": article_files,
        "docs_json": docs_json,
        "support_mdx": support_mdx,
    }


# ===========================================================================
# Tests: Tag pages match existing
# ===========================================================================

class TestTagPagesMatchExisting:
    """
    Verify that every generated tag page is byte-for-byte identical to
    the existing file in the wandb-docs repository.

    Each tag page is tested individually so that failures pinpoint the
    exact file and show a unified diff of the mismatch.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, golden_repo_root, generated_output):
        """Store references for use by test methods."""
        self.repo_root = golden_repo_root
        self.generated = generated_output

    def _check_tag_page(self, rel_path: str):
        """
        Compare a single generated tag page against the existing file.

        Fails the test with a unified diff if there is any difference.
        """
        existing_path = self.repo_root / rel_path
        if not existing_path.exists():
            pytest.fail(f"Generated tag page {rel_path} has no existing counterpart.")

        expected = _read_file(existing_path)
        actual = self.generated["tag_pages"].get(rel_path)
        if actual is None:
            pytest.fail(f"Tag page {rel_path} was not generated.")

        diff = _unified_diff(expected, actual, rel_path)
        if diff:
            pytest.fail(
                f"Tag page {rel_path} does not match existing file.\n"
                f"Diff:\n{diff}"
            )

    def test_all_tag_pages_match(self):
        """
        Every generated tag page should be identical to the existing file.

        This test iterates over all generated tag pages and compares each
        one.  It collects all mismatches and reports them together so you
        can see the full picture in one test run.
        """
        mismatches = []

        for rel_path in sorted(self.generated["tag_pages"].keys()):
            existing_path = self.repo_root / rel_path
            if not existing_path.exists():
                mismatches.append(f"  NEW (no existing file): {rel_path}")
                continue

            expected = _read_file(existing_path)
            actual = self.generated["tag_pages"][rel_path]

            if expected != actual:
                diff = _unified_diff(expected, actual, rel_path)
                mismatches.append(f"  DIFFERS: {rel_path}\n{textwrap.indent(diff, '    ')}")

        # Also check that no existing tag pages are missing from generated output
        for product in PRODUCTS:
            existing_dir = self.repo_root / "support" / product / "tags"
            if existing_dir.exists():
                for existing_file in sorted(existing_dir.glob("*.mdx")):
                    rel = f"support/{product}/tags/{existing_file.name}"
                    if rel not in self.generated["tag_pages"]:
                        mismatches.append(f"  MISSING (not generated): {rel}")

        if mismatches:
            pytest.fail(
                f"Tag page mismatches found ({len(mismatches)}):\n"
                + "\n".join(mismatches)
            )

    def test_tag_page_count_matches(self):
        """
        The number of generated tag pages should match the number of
        existing tag pages across all products.
        """
        existing_count = 0
        for product in PRODUCTS:
            existing_dir = self.repo_root / "support" / product / "tags"
            if existing_dir.exists():
                existing_count += len(list(existing_dir.glob("*.mdx")))

        generated_count = len(self.generated["tag_pages"])
        assert generated_count == existing_count, (
            f"Generated {generated_count} tag pages but {existing_count} exist on disk."
        )


# ===========================================================================
# Tests: Product index pages match existing
# ===========================================================================


class TestProductIndexPagesMatchExisting:
    """
    Verify that each support/<product>.mdx file produced in the temp tree
    matches the committed file in the real repository byte-for-byte.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, golden_repo_root, generated_output):
        self.repo_root = golden_repo_root
        self.generated = generated_output

    def test_all_product_index_pages_match(self):
        """
        Every configured product must have a generated index page that
        matches the repo.  Report all mismatches in one failure message.
        """
        mismatches = []

        for rel_path in sorted(self.generated["product_indexes"].keys()):
            existing_path = self.repo_root / rel_path
            if not existing_path.exists():
                mismatches.append(f"  NEW (no existing file): {rel_path}")
                continue

            expected = _read_file(existing_path)
            actual = self.generated["product_indexes"][rel_path]

            if expected != actual:
                diff = _unified_diff(expected, actual, rel_path)
                mismatches.append(f"  DIFFERS: {rel_path}\n{textwrap.indent(diff, '    ')}")

        for product in PRODUCTS:
            rel = f"support/{product}.mdx"
            existing_path = self.repo_root / rel
            if existing_path.exists() and rel not in self.generated["product_indexes"]:
                mismatches.append(f"  MISSING (not generated): {rel}")

        if mismatches:
            pytest.fail(
                f"Product index mismatches found ({len(mismatches)}):\n"
                + "\n".join(mismatches)
            )

    def test_product_index_count_matches(self):
        """Each product in config should have exactly one index file on disk and in output."""
        for product in PRODUCTS:
            rel = f"support/{product}.mdx"
            assert rel in self.generated["product_indexes"], (
                f"Generator did not produce {rel}"
            )
            assert (self.repo_root / rel).exists(), (
                f"Expected golden file missing: {rel}"
            )

        assert len(self.generated["product_indexes"]) == len(PRODUCTS), (
            f"Expected {len(PRODUCTS)} product index files, got "
            f"{len(self.generated['product_indexes'])}"
        )


# ===========================================================================
# Tests: Article files match existing (footer sync)
# ===========================================================================


class TestArticleFilesMatchExisting:
    """
    Verify that every article file after footer sync is byte-for-byte
    identical to the existing file in the wandb-docs repository.

    The generator rewrites tab-page Badge links on each article to match
    ``keywords`` in front matter. Running the pipeline on already-correct
    files should produce no changes.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, golden_repo_root, generated_output):
        self.repo_root = golden_repo_root
        self.generated = generated_output

    def test_all_article_files_match(self):
        """
        Every article file should be identical after the pipeline runs.
        Collects all mismatches and reports them together.
        """
        mismatches = []

        for rel_path in sorted(self.generated["article_files"].keys()):
            existing_path = self.repo_root / rel_path
            if not existing_path.exists():
                mismatches.append(f"  NEW (no existing file): {rel_path}")
                continue

            expected = _read_file(existing_path)
            actual = self.generated["article_files"][rel_path]

            if expected != actual:
                diff = _unified_diff(expected, actual, rel_path)
                mismatches.append(f"  DIFFERS: {rel_path}\n{textwrap.indent(diff, '    ')}")

        if mismatches:
            pytest.fail(
                f"Article file mismatches found ({len(mismatches)}):\n"
                + "\n".join(mismatches)
            )

    def test_article_file_count_matches(self):
        """
        The set of article files should be the same before and after the
        pipeline (no files created or deleted).
        """
        existing_count = 0
        for product in PRODUCTS:
            articles_dir = self.repo_root / "support" / product / "articles"
            if articles_dir.exists():
                existing_count += len(list(articles_dir.glob("*.mdx")))

        generated_count = len(self.generated["article_files"])
        assert generated_count == existing_count, (
            f"Generated output has {generated_count} article files "
            f"but {existing_count} exist on disk."
        )


# ===========================================================================
# Tests: docs.json support tabs match existing
# ===========================================================================

class TestDocsJsonMatchExisting:
    """
    Verify that the support navigation tabs in the generated docs.json
    match the existing docs.json exactly: same tab names, same page
    lists, same ordering.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, golden_repo_root, generated_output):
        """Store references for use by test methods."""
        self.repo_root = golden_repo_root
        self.generated_docs = generated_output["docs_json"]

    def _get_support_tabs(self, docs_json: dict) -> dict:
        """
        Extract the support tabs from a docs.json config.

        Returns a dict mapping tab name to the tab dict.
        """
        en_root = None
        for lang in docs_json.get("navigation", {}).get("languages", []):
            if lang.get("language") == "en":
                en_root = lang
                break

        if en_root is None:
            return {}

        return {
            tab["tab"]: tab
            for tab in en_root.get("tabs", [])
            if isinstance(tab, dict) and tab.get("tab", "").startswith("Support:")
        }

    def test_support_tabs_match(self):
        """
        The hidden support tabs in the generated docs.json should have
        the same names, page lists, and ordering as the existing file.
        """
        existing_docs = json.loads(
            _read_file(self.repo_root / "docs.json")
        )

        existing_tabs = self._get_support_tabs(existing_docs)
        generated_tabs = self._get_support_tabs(self.generated_docs)

        # Same set of tab names
        assert set(existing_tabs.keys()) == set(generated_tabs.keys()), (
            f"Tab names differ.\n"
            f"  Existing: {sorted(existing_tabs.keys())}\n"
            f"  Generated: {sorted(generated_tabs.keys())}"
        )

        # Each tab has the same pages in the same order
        for tab_name in sorted(existing_tabs.keys()):
            existing_pages = existing_tabs[tab_name].get("pages", [])
            generated_pages = generated_tabs[tab_name].get("pages", [])

            assert existing_pages == generated_pages, (
                f"Pages differ for tab '{tab_name}'.\n"
                f"  Existing:  {existing_pages}\n"
                f"  Generated: {generated_pages}"
            )

    def test_support_tabs_are_hidden(self):
        """All support tabs should have hidden: true."""
        generated_tabs = self._get_support_tabs(self.generated_docs)
        for tab_name, tab in generated_tabs.items():
            assert tab.get("hidden") is True, (
                f"Tab '{tab_name}' is not marked as hidden."
            )

    def test_non_support_tabs_preserved(self):
        """
        Non-support tabs (like Platform, Models, Weave) should be
        completely unchanged by the generator.
        """
        existing_docs = json.loads(
            _read_file(self.repo_root / "docs.json")
        )

        existing_en = None
        generated_en = None
        for lang in existing_docs["navigation"]["languages"]:
            if lang.get("language") == "en":
                existing_en = lang
                break
        for lang in self.generated_docs["navigation"]["languages"]:
            if lang.get("language") == "en":
                generated_en = lang
                break

        existing_non_support = [
            t for t in existing_en.get("tabs", [])
            if isinstance(t, dict) and not t.get("tab", "").startswith("Support:")
        ]
        generated_non_support = [
            t for t in generated_en.get("tabs", [])
            if isinstance(t, dict) and not t.get("tab", "").startswith("Support:")
        ]

        assert existing_non_support == generated_non_support, (
            "Non-support tabs were modified by the generator."
        )


# ===========================================================================
# Tests: support.mdx product card counts match existing
# ===========================================================================

class TestSupportIndexMatchExisting:
    """
    Verify that the product card counts in the generated support.mdx
    match the existing file exactly.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, golden_repo_root, generated_output):
        """Store references for use by test methods."""
        self.repo_root = golden_repo_root
        self.generated_support_mdx = generated_output["support_mdx"]

    def test_support_index_matches(self):
        """
        The generated support.mdx should be byte-for-byte identical to
        the existing file.  The generator only modifies the article/tag
        count lines inside product Cards; everything else is preserved.
        """
        expected = _read_file(self.repo_root / "support.mdx")
        actual = self.generated_support_mdx

        diff = _unified_diff(expected, actual, "support.mdx")
        if diff:
            pytest.fail(
                f"support.mdx does not match existing file.\n"
                f"Diff:\n{diff}"
            )
