"""Quality checks for generated query panel reference MDX."""

from __future__ import annotations

import importlib.util
import json
import re
import tempfile
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

    def test_user_keeps_a_not_an(self) -> None:
        m = _load_converter()
        row = "| `user` | A [user](/models/ref/query-panel/user) |"
        self.assertEqual(m.fix_indefinite_article_before_links(row), row)

    def test_wrong_an_before_user_is_corrected(self) -> None:
        m = _load_converter()
        row = "| `user` | An [user](/models/ref/query-panel/user) |"
        self.assertEqual(
            m.fix_indefinite_article_before_links(row),
            "| `user` | A [user](/models/ref/query-panel/user) |",
        )

    def test_unicode_style_label_keeps_a(self) -> None:
        m = _load_converter()
        row = "| `unicode` | A [unicode](/models/ref/query-panel/unicode) |"
        self.assertEqual(m.fix_indefinite_article_before_links(row), row)

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

    def test_replace_generated_types_section_idempotent_no_extra_blank_lines(self) -> None:
        """Re-running landing replacement must not grow vertical whitespace after {end}."""
        m = _load_converter()
        bullets = "* [a](./query-panel/a)\n* [b](./query-panel/b)\n"
        end = "{/* query-panel-generated-data-types:end */}"
        base = (
            "intro\n\n## Data Types\n\n"
            "{/* query-panel-generated-data-types:start */}\n"
            "* old\n"
            f"{end}\n"
            "## Next section\n"
        )
        out1 = m.replace_generated_types_section(base, bullets)
        out2 = m.replace_generated_types_section(out1, bullets)
        self.assertEqual(out1, out2)
        tail = out1.split(end, 1)[1]
        self.assertTrue(
            tail.startswith("\n\n## Next"),
            msg=f"expected single blank line after end marker, got repr: {tail[:40]!r}",
        )
        self.assertNotRegex(tail, r"\n{4,}")

    def test_update_docs_json_nav_only_english_query_panel_pages(self) -> None:
        m = _load_converter()
        ja_before = ["ja/models/ref/query-panel", "ja/models/ref/query-panel/run"]
        data = {
            "navigation": {
                "languages": [
                    {
                        "language": "en",
                        "groups": [
                            {
                                "group": "Query Expression Language",
                                "pages": ["models/ref/query-panel", "models/ref/query-panel/old"],
                            }
                        ],
                    },
                    {
                        "language": "ja",
                        "groups": [
                            {
                                "group": "Query Expression Language",
                                "pages": list(ja_before),
                            }
                        ],
                    },
                ]
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            m.update_docs_json_nav(path, ["run", "table"])
            updated = json.loads(path.read_text(encoding="utf-8"))
            langs = updated["navigation"]["languages"]
            en_pages = next(
                g["pages"]
                for L in langs
                if L["language"] == "en"
                for g in L["groups"]
                if g["group"] == "Query Expression Language"
            )
            ja_pages = next(
                g["pages"]
                for L in langs
                if L["language"] == "ja"
                for g in L["groups"]
                if g["group"] == "Query Expression Language"
            )
            self.assertEqual(
                en_pages,
                [
                    "models/ref/query-panel",
                    "models/ref/query-panel/run",
                    "models/ref/query-panel/table",
                ],
            )
            self.assertEqual(ja_pages, ja_before)
        finally:
            path.unlink(missing_ok=True)


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
