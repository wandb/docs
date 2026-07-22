#!/usr/bin/env python3
"""Check W&B report embeds used in docs pages.

Embeds are discovered by scanning the English `.mdx` sources for the
`<WandbReport>` component — there is no registry to keep in sync. Every run
checks placement and a recognizable report URL, then verifies over the network
that each report still renders anonymously.

No third-party dependencies (standard library only).

The tag name `WandbReport` and the `src` prop below are a contract with
snippets/WandbReport.jsx. If the component renames either, update COMPONENT_RE
and SRC_ATTR_RE here (see scripts/report-embeds/README.md).
"""

from __future__ import annotations

import argparse
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]

# A report URL ends in `--<id>` or `---<id>` where <id> starts with "Vmlldz".
REPORT_ID_RE = re.compile(r"-{2,3}(Vmlldz[A-Za-z0-9]+)")
# Match a <WandbReport ...> opening tag (self-closing or not), including
# multi-line attribute lists. `[^>]` also matches newlines.
COMPONENT_RE = re.compile(r"<WandbReport\b[^>]*?(?:/>|>)")
# Extract the src value: src="...", src='...', src={"..."}, src={'...'}, src={`...`}
SRC_ATTR_RE = re.compile(r"""src\s*=\s*\{?\s*['"`]([^'"`]+)['"`]""")
# Fenced code blocks and MDX comments hold example/illustrative markup, not live
# embeds — mask them before scanning so a documented <WandbReport> usage or a
# report link shown as sample code isn't treated as the real thing.
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
MDX_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)

# Locale content is managed by a separate translation pipeline (AGENTS.md);
# the checker only governs English sources.
SKIP_PREFIXES = (
    "ja/", "ko/", "fr/",
    "snippets/ja/", "snippets/ko/", "snippets/fr/",
    "node_modules/", ".git/",
)
# Snippets are shared partials, not pages. Embeds belong on pages only.
SNIPPETS_PREFIX = "snippets/"
LOGIN_MARKERS = ("/login", "/signin", "/site/login", "/authorize")


@dataclass
class Finding:
    code: str
    message: str
    path: str | None = None
    line: int | None = None


# --------------------------------------------------------------------------- #
# MDX scanning
# --------------------------------------------------------------------------- #
def iter_mdx(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.mdx")):
        rel = path.relative_to(root).as_posix()
        if not any(rel.startswith(pre) for pre in SKIP_PREFIXES):
            yield path


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def extract_embeds(text: str) -> list[tuple[str, int]]:
    """Return (src, line) for every <WandbReport> whose src can be parsed."""
    out: list[tuple[str, int]] = []
    for m in COMPONENT_RE.finditer(text):
        src_m = SRC_ATTR_RE.search(m.group(0))
        if src_m:
            out.append((src_m.group(1), _line_of(text, m.start())))
    return out


def _blank(match: re.Match) -> str:
    # Replace matched span with same-length whitespace so byte offsets — and
    # therefore reported line numbers — are preserved.
    return re.sub(r"[^\n]", " ", match.group(0))


def mask_noncontent(text: str) -> str:
    return MDX_COMMENT_RE.sub(_blank, FENCE_RE.sub(_blank, text))


def scan(root: Path) -> tuple[list[Finding], list[tuple[str, str, int]]]:
    """Scan English .mdx for <WandbReport> embeds.

    Returns (findings, embeds). `embeds` is (url, page, line) deduplicated by URL
    for the liveness pass; findings cover the static checks.
    """
    findings: list[Finding] = []
    embeds: list[tuple[str, str, int]] = []
    seen: set[str] = set()

    for path in iter_mdx(root):
        rel = path.relative_to(root).as_posix()
        text = mask_noncontent(path.read_text(encoding="utf-8"))
        occurrences = extract_embeds(text)
        if not occurrences:
            continue
        if rel.startswith(SNIPPETS_PREFIX):
            findings.append(Finding("EMBED_IN_SNIPPET", "<WandbReport> must be used on a page, not in a shared snippet", rel, occurrences[0][1]))
            continue

        for src, line in occurrences:
            if not REPORT_ID_RE.search(src):
                findings.append(Finding("BAD_EMBED_SRC", f"src `{src}` is not a recognizable report URL (needs --Vmlldz...)", rel, line))
                continue
            if src not in seen:
                seen.add(src)
                embeds.append((src, rel, line))

    return findings, embeds


# --------------------------------------------------------------------------- #
# Liveness
# --------------------------------------------------------------------------- #
def check_url(url: str, *, timeout: int, attempts: int, delay: float) -> Finding | None:
    """Anonymous GET; require a final 200 that is not a login redirect."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; wandb-docs-embed-check/1.0)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    last_err = ""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.getcode() != 200:
                    return Finding("URL_NOT_200", f"{url}: final status {resp.getcode()}")
                if any(marker in resp.geturl() for marker in LOGIN_MARKERS):
                    return Finding("URL_LOGIN_REDIRECT", f"{url}: redirected to login ({resp.geturl()}) — magic link likely revoked")
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


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _loc(f: Finding) -> str:
    return (f.path or "-") + (f":{f.line}" if f.path and f.line else "")


def emit_github_annotations(findings: Iterable[Finding]) -> None:
    for f in findings:
        loc = (f"file={f.path}" + (f",line={f.line}" if f.line else "")) if f.path else ""
        print(f"::error {loc}::[{f.code}] {f.message}")


def render_markdown(findings: list[Finding], *, embeds: int) -> str:
    lines = ["# Report embed check", ""]
    if not findings:
        lines.append(f"✅ All {embeds} embed(s) render anonymously.")
        return "\n".join(lines) + "\n"
    lines += [f"Found {len(findings)} problem(s).", "", "| Code | Location | Detail |", "| --- | --- | --- |"]
    lines += [f"| {f.code} | {_loc(f)} | {f.message} |" for f in findings]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--github", action="store_true", help="emit GitHub Actions annotations")
    parser.add_argument("--output-md", type=Path, help="write a Markdown report to this file")
    args = parser.parse_args(argv)

    findings, embeds = scan(args.root)
    findings += check_liveness(embeds)

    if args.github:
        emit_github_annotations(findings)
    else:
        for f in findings:
            print(f"ERROR [{f.code}] {_loc(f)}: {f.message}")

    if args.output_md:
        args.output_md.write_text(render_markdown(findings, embeds=len(embeds)), encoding="utf-8")

    if not findings:
        print(f"OK: {len(embeds)} report embed(s) checked.")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
