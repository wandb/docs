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
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()


class ExtractionTests(unittest.TestCase):
    def test_self_closing(self):
        text = '<WandbReport src="https://wandb.ai/w/p/reports/Foo--Vmlldzo1" title="t" />'
        self.assertEqual(m.extract_embeds(text), [("https://wandb.ai/w/p/reports/Foo--Vmlldzo1", 1)])

    def test_multiline_and_brace_src(self):
        text = 'x\n<WandbReport\n  src={"https://wandb.ai/w/p/reports/Foo---Vmlldzo2"}\n/>\n'
        self.assertEqual(m.extract_embeds(text), [("https://wandb.ai/w/p/reports/Foo---Vmlldzo2", 2)])


class ScanTests(unittest.TestCase):
    def _run(self, files: dict[str, str]):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for rel, content in files.items():
                fp = root / rel
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(content, encoding="utf-8")
            return m.scan(root)

    def test_valid_embed_passes(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        page = f'<WandbReport src="{src}" title="t" />\n'
        findings, embeds = self._run({"models/x.mdx": page})
        self.assertEqual(findings, [])
        self.assertEqual(embeds, [(src, "models/x.mdx", 1)])

    def test_bad_embed_src(self):
        page = '<WandbReport src="https://wandb.ai/w/p/reports/not-a-report" title="t" />\n'
        findings, embeds = self._run({"models/x.mdx": page})
        self.assertTrue(any(f.code == "BAD_EMBED_SRC" for f in findings))
        self.assertEqual(embeds, [])

    def test_duplicate_url_deduped_for_liveness(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        page = f'<WandbReport src="{src}" title="t" />\n'
        findings, embeds = self._run({"models/a.mdx": page, "models/b.mdx": page})
        self.assertEqual(findings, [])
        self.assertEqual(len(embeds), 1)

    def test_masked_regions_ignored(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        embed = f'<WandbReport src="{src}" title="t" />\n'
        for page in (f"```mdx\n{embed}```\n", "{/* later:\n" + embed + "*/}\n"):
            self.assertEqual(self._run({"models/x.mdx": page}), ([], []))

    def test_embed_in_snippet_errors(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        findings, _ = self._run({"snippets/_includes/t.mdx": f'<WandbReport src="{src}" title="t" />\n'})
        self.assertTrue(any(f.code == "EMBED_IN_SNIPPET" for f in findings))


if __name__ == "__main__":
    unittest.main()
