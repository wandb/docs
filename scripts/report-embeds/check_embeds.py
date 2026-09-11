#!/usr/bin/env python3
"""Check the live W&B report embeds (<WandbReport>) in docs pages.

Embeds are found by scanning the English .mdx sources (no registry): each is
checked for placement and a recognizable report URL, then fetched anonymously
to confirm the report still renders.

COMPONENT_RE / SRC_ATTR_RE are a contract with snippets/WandbReport.jsx — keep
them in sync if the component's tag or `src` prop is renamed.
"""

from __future__ import annotations

import argparse
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]

# A report URL ends in `--<id>` or `---<id>` where <id> starts with "Vmlldz".
REPORT_ID_RE = re.compile(r"-{2,3}(Vmlldz[A-Za-z0-9]+)")
# A <WandbReport ...> opening tag (self-closing or not), across newlines.
COMPONENT_RE = re.compile(r"<WandbReport\b[^>]*?(?:/>|>)")
# The src value: src="...", src='...', or src={"..."} / {'...'} / {`...`}.
SRC_ATTR_RE = re.compile(r"""src\s*=\s*\{?\s*['"`]([^'"`]+)['"`]""")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
MDX_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)

# Locale content is managed by a separate translation pipeline (AGENTS.md);
# only English sources are governed here.
SKIP_PREFIXES = (
    "ja/", "ko/", "fr/",
    "snippets/ja/", "snippets/ko/", "snippets/fr/",
    "node_modules/", ".git/",
)
SNIPPETS_PREFIX = "snippets/"  # embeds belong on pages, not shared partials
LOGIN_MARKERS = ("/login", "/signin", "/site/login", "/authorize")


@dataclass
class Finding:
    code: str
    message: str
    path: str | None = None
    line: int | None = None


# --- MDX scanning ---
def iter_mdx(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.mdx")):
        rel = path.relative_to(root).as_posix()
        if not any(rel.startswith(pre) for pre in SKIP_PREFIXES):
            yield path


def extract_embeds(text: str) -> list[tuple[str, int]]:
    """Return (src, line) for every <WandbReport> whose src can be parsed."""
    out = []
    for m in COMPONENT_RE.finditer(text):
        src_m = SRC_ATTR_RE.search(m.group(0))
        if src_m:
            out.append((src_m.group(1), text.count("\n", 0, m.start()) + 1))
    return out


def mask_noncontent(text: str) -> str:
    # Blank fenced code + MDX comments with same-length whitespace (keeps line
    # numbers stable) so example markup isn't scanned as a live embed.
    def blank(m: re.Match) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))
    return MDX_COMMENT_RE.sub(blank, FENCE_RE.sub(blank, text))


def scan(root: Path) -> tuple[list[Finding], list[tuple[str, str, int]]]:
    """Return (findings, embeds); embeds is (url, page, line) deduped by URL."""
    findings: list[Finding] = []
    embeds: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for path in iter_mdx(root):
        rel = path.relative_to(root).as_posix()
        occurrences = extract_embeds(mask_noncontent(path.read_text(encoding="utf-8")))
        if not occurrences:
            continue
        if rel.startswith(SNIPPETS_PREFIX):
            findings.append(Finding("EMBED_IN_SNIPPET", "<WandbReport> must be used on a page, not a shared snippet", rel, occurrences[0][1]))
            continue
        for src, line in occurrences:
            if not REPORT_ID_RE.search(src):
                findings.append(Finding("BAD_EMBED_SRC", f"src `{src}` is not a recognizable report URL (needs --Vmlldz...)", rel, line))
            elif src not in seen:
                seen.add(src)
                embeds.append((src, rel, line))
    return findings, embeds


# --- Liveness ---
def check_url(url: str, *, timeout: int, attempts: int, delay: float) -> Finding | None:
    """Anonymous GET; require a final 200 that is not a login redirect."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; wandb-docs-embed-check/1.0)",
        "Accept": "text/html,application/xhtml+xml",
    })
    last_err = ""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.getcode() != 200:
                    return Finding("URL_NOT_200", f"{url}: final status {resp.getcode()}")
                if any(mk in resp.geturl() for mk in LOGIN_MARKERS):
                    return Finding("URL_LOGIN_REDIRECT", f"{url}: redirected to login ({resp.geturl()}) — link likely revoked")
                return None
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < attempts:
                last_err = f"HTTP {exc.code}"
                time.sleep(delay * (2 ** attempt))
                continue
            return Finding("URL_NOT_200", f"{url}: HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = str(getattr(exc, "reason", exc))
            if attempt < attempts:
                time.sleep(delay * (2 ** attempt))
                continue
    return Finding("URL_UNREACHABLE", f"{url}: unreachable after {attempts} attempts ({last_err})")


def check_liveness(embeds: list[tuple[str, str, int]], *, timeout: int = 30, attempts: int = 3, delay: float = 1.0) -> list[Finding]:
    findings: list[Finding] = []
    for url, rel, line in embeds:
        finding = check_url(url, timeout=timeout, attempts=attempts, delay=delay)
        if finding:
            finding.path, finding.line = rel, line
            findings.append(finding)
        time.sleep(delay)  # be polite; the embed set is small
    return findings


# --- Output + CLI ---
def _loc(f: Finding) -> str:
    return (f.path or "-") + (f":{f.line}" if f.path and f.line else "")


def render_markdown(findings: list[Finding], *, embeds: int) -> str:
    if not findings:
        return f"# Report embed check\n\n✅ All {embeds} embed(s) render anonymously.\n"
    lines = ["# Report embed check", "", f"Found {len(findings)} problem(s).", "", "| Code | Location | Detail |", "| --- | --- | --- |"]
    lines += [f"| {f.code} | {_loc(f)} | {f.message} |" for f in findings]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-md", type=Path, help="write a Markdown report to this file")
    args = parser.parse_args(argv)

    findings, embeds = scan(args.root)
    findings += check_liveness(embeds)

    for f in findings:
        print(f"ERROR [{f.code}] {_loc(f)}: {f.message}")
    if args.output_md:
        args.output_md.write_text(render_markdown(findings, embeds=len(embeds)), encoding="utf-8")
    if not findings:
        print(f"OK: {len(embeds)} report embed(s) checked.")
    return 1 if findings else 0


# --- Tests — python3 -m unittest discover -s scripts/report-embeds -p 'check_embeds.py' ---
import tempfile
import unittest


class Tests(unittest.TestCase):
    def _scan(self, files: dict[str, str]):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for rel, content in files.items():
                fp = root / rel
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(content, encoding="utf-8")
            return scan(root)

    def test_extract(self):
        self.assertEqual(extract_embeds('<WandbReport src="x--Vmlldzo1" />'), [("x--Vmlldzo1", 1)])
        self.assertEqual(extract_embeds('a\n<WandbReport\n  src={"y---Vmlldzo2"}\n/>\n'), [("y---Vmlldzo2", 2)])

    def test_valid_and_dedup(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        embed = f'<WandbReport src="{src}" title="t" />\n'
        findings, embeds = self._scan({"models/a.mdx": embed, "models/b.mdx": embed})
        self.assertEqual(findings, [])
        self.assertEqual(embeds, [(src, "models/a.mdx", 1)])  # deduped by URL

    def test_bad_src(self):
        findings, embeds = self._scan({"models/x.mdx": '<WandbReport src="https://wandb.ai/w/p/reports/nope" />\n'})
        self.assertEqual([f.code for f in findings], ["BAD_EMBED_SRC"])
        self.assertEqual(embeds, [])

    def test_snippet_errors_and_masked_ignored(self):
        src = "https://wandb.ai/w/p/reports/Foo--Vmlldzo12345"
        embed = f'<WandbReport src="{src}" title="t" />\n'
        snippet, _ = self._scan({"snippets/_includes/t.mdx": embed})
        self.assertEqual([f.code for f in snippet], ["EMBED_IN_SNIPPET"])
        for masked in (f"```mdx\n{embed}```\n", "{/* " + embed + "*/}\n"):
            self.assertEqual(self._scan({"models/x.mdx": masked}), ([], []))


if __name__ == "__main__":
    raise SystemExit(main())
