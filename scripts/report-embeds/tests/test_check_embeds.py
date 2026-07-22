"""Unit tests for the report-embed checker. No network."""

from __future__ import annotations

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


class ScanTests(unittest.TestCase):
    def _run(self, files: dict[str, str]):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for rel, content in files.items():
                fp = root / rel
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(content, encoding="utf-8")
            return m.scan(root)

    def test_valid_embed_with_prose_link_passes(self):
        page = (
            "See the [sweep report](https://wandb.ai/w/p/reports/Foo--Vmlldzo12345).\n\n"
            '<WandbReport src="https://wandb.ai/w/p/reports/Foo--Vmlldzo12345" title="t" height={700} />\n'
        )
        findings, embeds = self._run({"models/sweeps/visualize-sweep-results.mdx": page})
        self.assertEqual([f for f in findings if f.level == "error"], [])
        self.assertEqual(len(embeds), 1)
        self.assertEqual(embeds[0].url, "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345")

    def test_missing_prose_link(self):
        page = '<WandbReport src="https://wandb.ai/w/p/reports/Foo--Vmlldzo12345" title="t" />\n'
        findings, _ = self._run({"models/sweeps/visualize-sweep-results.mdx": page})
        self.assertTrue(any(f.code == "MISSING_PROSE_LINK" for f in findings))

    def test_bad_embed_src(self):
        page = '<WandbReport src="https://wandb.ai/w/p/reports/not-a-report" title="t" />\n'
        findings, embeds = self._run({"models/x.mdx": page})
        self.assertTrue(any(f.code == "BAD_EMBED_SRC" for f in findings))
        self.assertEqual(embeds, [])

    def test_too_many_embeds_warns(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        link = f"[r]({src})\n"
        page = link + "".join(f'<WandbReport src="{src}" title="t{i}" />\n' for i in range(3))
        findings, _ = self._run({"models/x.mdx": page})
        self.assertTrue(any(f.code == "TOO_MANY_EMBEDS" and f.level == "warning" for f in findings))

    def test_duplicate_url_deduped_for_liveness(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        link = f"[r]({src})\n"
        # Same report embedded on two pages: two prose links, one unique URL.
        findings, embeds = self._run({
            "models/a.mdx": link + f'<WandbReport src="{src}" title="a" />\n',
            "models/b.mdx": link + f'<WandbReport src="{src}" title="b" />\n',
        })
        self.assertEqual([f for f in findings if f.level == "error"], [])
        self.assertEqual(len(embeds), 1)

    def test_locale_files_ignored(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        page = f'<WandbReport src="{src}" title="t" />\n'  # no prose link — would error if scanned
        english = f"[r]({src})\n" + page
        findings, _ = self._run({"models/x.mdx": english, "ja/models/x.mdx": page})
        self.assertEqual([f for f in findings if f.level == "error"], [])

    def test_embed_in_code_fence_ignored(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        page = "Usage example:\n\n```mdx\n" + f'<WandbReport src="{src}" title="t" />\n' + "```\n"
        findings, embeds = self._run({"models/x.mdx": page})
        self.assertEqual(findings, [])
        self.assertEqual(embeds, [])

    def test_embed_in_mdx_comment_ignored(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        page = "{/* staged, fill in real URL later:\n" + f'<WandbReport src="{src}" title="t" />\n' + "*/}\n"
        findings, embeds = self._run({"models/x.mdx": page})
        self.assertEqual(findings, [])
        self.assertEqual(embeds, [])

    def test_embed_in_snippet_errors(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        findings, _ = self._run(
            {"snippets/_includes/thing.mdx": f'[r]({src})\n<WandbReport src="{src}" title="t" />\n'},
        )
        self.assertTrue(any(f.code == "EMBED_IN_SNIPPET" for f in findings))


if __name__ == "__main__":
    unittest.main()
