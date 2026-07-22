"""Unit tests for the report-embed validator. No network."""

from __future__ import annotations

import datetime
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "check_embeds.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_embeds", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    # Register before exec so dataclass type introspection can resolve the
    # module (required on Python 3.9; harmless on newer versions).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()

TODAY = datetime.date.today().isoformat()


def _entry(**over):
    base = {
        "id": "Vmlldzo12345",
        "url": "https://wandb.ai/wandb/docs-embeds/reports/Foo--Vmlldzo12345",
        "owner": "@me",
        "purpose": "demo",
        "pages": ["models/sweeps/visualize-sweep-results.mdx"],
        "last_reviewed": TODAY,
        "height": 700,
    }
    base.update(over)
    return base


class RegistrySchemaTests(unittest.TestCase):
    def _load_yaml(self, doc: str):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "registry.yaml"
            p.write_text(doc, encoding="utf-8")
            return m.load_registry(p)

    def test_empty_registry_is_valid(self):
        _, findings = self._load_yaml("reports: []\n")
        self.assertEqual(findings, [])

    def test_missing_reports_key_is_shape_error(self):
        _, findings = self._load_yaml("something: else\n")
        self.assertTrue(any(f.code == "REGISTRY_SHAPE" for f in findings))

    def test_top_level_list_is_shape_error_not_crash(self):
        # A top-level list/string must yield REGISTRY_SHAPE, not AttributeError.
        _, findings = self._load_yaml("- a\n- b\n")
        self.assertTrue(any(f.code == "REGISTRY_SHAPE" for f in findings))
        _, findings = self._load_yaml("just a string\n")
        self.assertTrue(any(f.code == "REGISTRY_SHAPE" for f in findings))

    def test_bad_id_pattern(self):
        import yaml
        doc = yaml.safe_dump({"reports": [_entry(id="notanid", url="https://wandb.ai/w/p/reports/Foo--notanid")]})
        _, findings = self._load_yaml(doc)
        self.assertTrue(any(f.code == "BAD_ID" for f in findings))

    def test_url_id_mismatch(self):
        import yaml
        doc = yaml.safe_dump({"reports": [_entry(url="https://wandb.ai/w/p/reports/Foo--Vmlldzo99999")]})
        _, findings = self._load_yaml(doc)
        self.assertTrue(any(f.code == "URL_ID_MISMATCH" for f in findings))

    def test_missing_required_field(self):
        import yaml
        e = _entry()
        del e["owner"]
        doc = yaml.safe_dump({"reports": [e]})
        _, findings = self._load_yaml(doc)
        self.assertTrue(any(f.code == "MISSING_FIELD" for f in findings))

    def test_future_date(self):
        import yaml
        future = (datetime.date.today() + datetime.timedelta(days=5)).isoformat()
        doc = yaml.safe_dump({"reports": [_entry(last_reviewed=future)]})
        _, findings = self._load_yaml(doc)
        self.assertTrue(any(f.code == "FUTURE_DATE" for f in findings))

    def test_bad_host(self):
        import yaml
        doc = yaml.safe_dump({"reports": [_entry(url="https://evil.example/reports/Foo--Vmlldzo12345")]})
        _, findings = self._load_yaml(doc)
        self.assertTrue(any(f.code == "BAD_URL_HOST" for f in findings))


class ExtractionTests(unittest.TestCase):
    def test_self_closing(self):
        text = '<WandbReport src="https://wandb.ai/w/p/reports/Foo--Vmlldzo1" title="t" />'
        self.assertEqual(m.extract_embeds(text), [("https://wandb.ai/w/p/reports/Foo--Vmlldzo1", 1)])

    def test_multiline_and_brace_src(self):
        text = (
            "intro\n"
            "<WandbReport\n"
            '  src={"https://wandb.ai/w/p/reports/Foo---Vmlldzo2"}\n'
            "  height={700}\n"
            "/>\n"
        )
        embeds = m.extract_embeds(text)
        self.assertEqual(len(embeds), 1)
        self.assertEqual(embeds[0][0], "https://wandb.ai/w/p/reports/Foo---Vmlldzo2")
        self.assertEqual(embeds[0][1], 2)

    def test_strip_removes_component(self):
        text = 'before <WandbReport src="x--Vmlldzo3" /> after'
        self.assertNotIn("Vmlldzo3", m.strip_embeds(text))
        self.assertIn("before", m.strip_embeds(text))


class MdxConsistencyTests(unittest.TestCase):
    def _run(self, files: dict[str, str], entries: list[dict]):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for rel, content in files.items():
                fp = root / rel
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(content, encoding="utf-8")
            return m.check_mdx(root, entries)

    def test_valid_embed_with_prose_link_passes(self):
        page = (
            "See the [sweep report](https://wandb.ai/w/p/reports/Foo--Vmlldzo12345).\n\n"
            '<WandbReport src="https://wandb.ai/w/p/reports/Foo--Vmlldzo12345" title="t" height={700} />\n'
        )
        findings = self._run({"models/sweeps/visualize-sweep-results.mdx": page}, [_entry()])
        self.assertEqual([f for f in findings if f.level == "error"], [])

    def test_missing_prose_link(self):
        page = '<WandbReport src="https://wandb.ai/w/p/reports/Foo--Vmlldzo12345" title="t" />\n'
        findings = self._run({"models/sweeps/visualize-sweep-results.mdx": page}, [_entry()])
        self.assertTrue(any(f.code == "MISSING_PROSE_LINK" for f in findings))

    def test_unregistered_embed(self):
        page = (
            "[link](https://wandb.ai/w/p/reports/Foo--Vmlldzo99999)\n"
            '<WandbReport src="https://wandb.ai/w/p/reports/Foo--Vmlldzo99999" title="t" />\n'
        )
        findings = self._run({"models/sweeps/visualize-sweep-results.mdx": page}, [])
        self.assertTrue(any(f.code == "EMBED_NOT_IN_REGISTRY" for f in findings))

    def test_orphan_entry_warns(self):
        findings = self._run({"models/sweeps/visualize-sweep-results.mdx": "no embeds here\n"}, [_entry()])
        self.assertTrue(any(f.code == "ORPHAN_ENTRY" and f.level == "warning" for f in findings))

    def test_too_many_embeds_warns(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        link = f"[r]({src})\n"
        page = link + "".join(f'<WandbReport src="{src}" title="t{i}" />\n' for i in range(3))
        entry = _entry(pages=["models/x.mdx"])
        findings = self._run({"models/x.mdx": page}, [entry])
        self.assertTrue(any(f.code == "TOO_MANY_EMBEDS" and f.level == "warning" for f in findings))

    def test_locale_files_ignored(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        page = f'<WandbReport src="{src}" title="t" />\n'  # no prose link — would error if scanned
        # Same embed on the English page (valid) plus a locale copy (must be skipped).
        english = f"[r]({src})\n" + page
        findings = self._run(
            {"models/x.mdx": english, "ja/models/x.mdx": page},
            [_entry(pages=["models/x.mdx"])],
        )
        self.assertEqual([f for f in findings if f.level == "error"], [])

    def test_embed_in_code_fence_ignored(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        page = "Usage example:\n\n```mdx\n" + f'<WandbReport src="{src}" title="t" />\n' + "```\n"
        findings = self._run({"models/x.mdx": page}, [])
        self.assertEqual(findings, [])

    def test_embed_in_mdx_comment_ignored(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        page = "{/* staged, fill in real URL later:\n" + f'<WandbReport src="{src}" title="t" />\n' + "*/}\n"
        findings = self._run({"models/x.mdx": page}, [])
        self.assertEqual(findings, [])

    def test_embed_in_snippet_errors(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        findings = self._run(
            {"snippets/_includes/thing.mdx": f"[r]({src})\n<WandbReport src=\"{src}\" title=\"t\" />\n"},
            [_entry(pages=["snippets/_includes/thing.mdx"])],
        )
        self.assertTrue(any(f.code == "EMBED_IN_SNIPPET" for f in findings))


if __name__ == "__main__":
    unittest.main()
