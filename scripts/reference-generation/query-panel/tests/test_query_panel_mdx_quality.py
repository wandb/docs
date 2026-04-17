"""Quality checks for generated query panel reference MDX."""

from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_converter():
    path = REPO_ROOT / "scripts/reference-generation/query-panel/convert_query_panel_md.py"
    spec = importlib.util.spec_from_file_location("convert_query_panel_md", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


QUERY_PANEL_DIR = REPO_ROOT / "models" / "ref" / "query-panel"


def _anchor_ids(content: str) -> list[str]:
    return re.findall(r'<a id="([^"]+)"', content)


class ConverterUnitTests(unittest.TestCase):
    def test_fix_indefinite_article_in_table_cell(self) -> None:
        m = _load_converter()
        row = "| `artifactType` | A [artifactType](/models/ref/query-panel/artifact-type) |"
        fixed = m.fix_indefinite_article_before_links(row)
        self.assertIn("An [artifactType]", fixed)
        row2 = "| `run` | A [run](/models/ref/query-panel/run) |"
        self.assertEqual(m.fix_indefinite_article_before_links(row2), row2)

    def test_page_display_title_prefers_overview_label(self) -> None:
        m = _load_converter()
        raw = "# artifact-type\n\n## Chainable Ops\n"
        labels = {"artifact-type": "artifactType"}
        self.assertEqual(
            m.page_display_title("artifact-type", raw, labels),
            "artifactType",
        )

    def test_strip_trailing_slashes_on_internal_links(self) -> None:
        m = _load_converter()
        s = "See [run](/models/ref/query-panel/run/) and [x](https://example.com/a/)"
        out = m.strip_query_panel_link_trailing_slashes(s)
        self.assertIn("](/models/ref/query-panel/run)", out)
        self.assertNotIn("/models/ref/query-panel/run/", out)
        self.assertIn("https://example.com/a/", out)


class QueryPanelMdxQualityTests(unittest.TestCase):
    def test_no_duplicate_anchor_ids_in_generated_pages(self) -> None:
        for path in sorted(QUERY_PANEL_DIR.glob("*.mdx")):
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                ids = _anchor_ids(text)
                dupes = sorted({i for i in ids if ids.count(i) > 1})
                self.assertEqual(dupes, [], f"{path.name} has duplicate anchor ids: {dupes}")

    def test_chainable_and_list_ops_sections_for_run(self) -> None:
        run = (QUERY_PANEL_DIR / "run.mdx").read_text(encoding="utf-8")
        self.assertIn("## Chainable Ops", run)
        self.assertIn("## List Ops", run)

    def test_internal_links_use_models_path(self) -> None:
        for slug in ("run", "table", "artifact"):
            with self.subTest(slug=slug):
                text = (QUERY_PANEL_DIR / f"{slug}.mdx").read_text(encoding="utf-8")
                self.assertNotIn("https://docs.wandb.ai/ref/weave/", text)

    def test_internal_query_panel_links_have_no_trailing_slash(self) -> None:
        for path in sorted(QUERY_PANEL_DIR.glob("*.mdx")):
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertNotRegex(
                    text,
                    r"/models/ref/query-panel/[a-zA-Z0-9-]+/",
                    msg=f"{path.name} contains /models/ref/query-panel/.../ (trailing slash)",
                )

    def test_landing_has_generator_markers(self) -> None:
        landing = (REPO_ROOT / "models" / "ref" / "query-panel.mdx").read_text(
            encoding="utf-8"
        )
        self.assertIn("{/* query-panel-generated-data-types:start */}", landing)
        self.assertIn("{/* query-panel-generated-data-types:end */}", landing)


if __name__ == "__main__":
    unittest.main()
