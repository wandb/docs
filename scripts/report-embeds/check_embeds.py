#!/usr/bin/env python3
"""Validate the W&B report-embed registry and its use in docs pages.

Three modes:
  static   - registry schema + <WandbReport> usage consistency (no network)
  liveness - anonymous HTTP check that each registered URL still renders
  all      - both

Dependencies: PyYAML only (HTTP uses the standard library).

The tag name `WandbReport` and the `src` prop below are a contract with
snippets/WandbReport.jsx. If the component renames either, update COMPONENT_RE
and SRC_ATTR_RE here (see scripts/report-embeds/README.md).
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = Path(__file__).resolve().parent / "registry.yaml"

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
# the validator only governs English sources.
SKIP_PREFIXES = (
    "ja/", "ko/", "fr/",
    "snippets/ja/", "snippets/ko/", "snippets/fr/",
    "node_modules/", ".git/",
)
# Snippets are shared partials, not pages; an embed there makes the registry's
# `pages` field ambiguous. Embeds belong on pages only.
SNIPPETS_PREFIX = "snippets/"
LOGIN_MARKERS = ("/login", "/signin", "/site/login", "/authorize")

MAX_EMBEDS_PER_PAGE = 2


@dataclass
class Finding:
    level: str          # "error" | "warning"
    code: str
    message: str
    path: str | None = None
    line: int | None = None


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def load_registry(path: Path) -> tuple[list[dict], list[Finding]]:
    """Parse and schema-validate the registry. Returns (entries, findings)."""
    findings: list[Finding] = []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [], [Finding("error", "REGISTRY_MISSING", f"Registry not found: {path}")]
    except yaml.YAMLError as exc:
        return [], [Finding("error", "REGISTRY_UNREADABLE", f"Invalid YAML in {path}: {exc}")]

    if not isinstance(raw, dict) or raw.get("reports") is None:
        return [], [Finding("error", "REGISTRY_SHAPE", "Registry must have a top-level `reports:` list")]
    entries = raw["reports"]
    if not isinstance(entries, list):
        return [], [Finding("error", "REGISTRY_SHAPE", "`reports` must be a list")]

    today = datetime.date.today()
    seen_ids: set[str] = set()
    valid: list[dict] = []

    for i, entry in enumerate(entries):
        loc = f"registry.yaml reports[{i}]"
        if not isinstance(entry, dict):
            findings.append(Finding("error", "ENTRY_SHAPE", f"{loc}: entry must be a mapping"))
            continue

        rid = entry.get("id")
        url = entry.get("url")
        pages = entry.get("pages")

        for field in ("id", "url", "owner", "purpose", "pages", "last_reviewed"):
            if entry.get(field) in (None, "", []):
                findings.append(Finding("error", "MISSING_FIELD", f"{loc}: required field `{field}` is missing or empty"))

        if isinstance(rid, str):
            if not re.fullmatch(r"Vmlldz[A-Za-z0-9]+", rid):
                findings.append(Finding("error", "BAD_ID", f"{loc}: id `{rid}` must match Vmlldz[A-Za-z0-9]+"))
            if rid in seen_ids:
                findings.append(Finding("error", "DUPLICATE_ID", f"{loc}: duplicate id `{rid}`"))
            seen_ids.add(rid)

        if isinstance(url, str):
            if not url.startswith("https://wandb.ai/"):
                findings.append(Finding("error", "BAD_URL_HOST", f"{loc}: url must start with https://wandb.ai/"))
            if "/reports/" not in url:
                findings.append(Finding("error", "BAD_URL_PATH", f"{loc}: url must contain /reports/"))
            url_ids = REPORT_ID_RE.findall(url)
            if isinstance(rid, str) and rid not in url_ids:
                findings.append(Finding("error", "URL_ID_MISMATCH", f"{loc}: url must contain --{rid}"))

        if isinstance(pages, list):
            for p in pages:
                if not isinstance(p, str) or not p.endswith(".mdx"):
                    findings.append(Finding("error", "BAD_PAGE", f"{loc}: page `{p}` must be a repo-relative .mdx path"))
                    continue
                if any(p.startswith(pre) for pre in SKIP_PREFIXES):
                    findings.append(Finding("error", "LOCALE_PAGE", f"{loc}: page `{p}` is a locale/excluded path; register English sources only"))
                elif not (REPO_ROOT / p).is_file():
                    findings.append(Finding("error", "PAGE_NOT_FOUND", f"{loc}: page `{p}` does not exist"))

        reviewed = entry.get("last_reviewed")
        if reviewed is not None:
            d = reviewed if isinstance(reviewed, datetime.date) else None
            if d is None:
                try:
                    d = datetime.date.fromisoformat(str(reviewed))
                except ValueError:
                    findings.append(Finding("error", "BAD_DATE", f"{loc}: last_reviewed `{reviewed}` is not an ISO date (YYYY-MM-DD)"))
            if d is not None and d > today:
                findings.append(Finding("error", "FUTURE_DATE", f"{loc}: last_reviewed `{d}` is in the future"))

        height = entry.get("height")
        if height is not None and not (isinstance(height, int) and height > 0):
            findings.append(Finding("error", "BAD_HEIGHT", f"{loc}: height must be a positive integer"))

        valid.append(entry)

    return valid, findings


# --------------------------------------------------------------------------- #
# MDX scanning
# --------------------------------------------------------------------------- #
def iter_mdx(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.mdx")):
        rel = path.relative_to(root).as_posix()
        if any(rel.startswith(pre) for pre in SKIP_PREFIXES):
            continue
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


def strip_embeds(text: str) -> str:
    return COMPONENT_RE.sub("", text)


def _blank(match: re.Match) -> str:
    # Replace matched span with same-length whitespace so byte offsets — and
    # therefore reported line numbers — are preserved.
    return re.sub(r"[^\n]", " ", match.group(0))


def mask_noncontent(text: str) -> str:
    return MDX_COMMENT_RE.sub(_blank, FENCE_RE.sub(_blank, text))


def check_mdx(root: Path, entries: list[dict]) -> list[Finding]:
    findings: list[Finding] = []
    by_id = {e["id"]: e for e in entries if isinstance(e.get("id"), str)}
    # id -> set of pages that actually embed it (English sources)
    embedded_on: dict[str, set[str]] = {rid: set() for rid in by_id}

    for path in iter_mdx(root):
        rel = path.relative_to(root).as_posix()
        text = mask_noncontent(path.read_text(encoding="utf-8"))
        embeds = extract_embeds(text)
        if not embeds:
            continue

        if rel.startswith(SNIPPETS_PREFIX):
            findings.append(Finding("error", "EMBED_IN_SNIPPET", "<WandbReport> must be used on a page, not in a shared snippet", rel, embeds[0][1]))
            continue

        if len(embeds) > MAX_EMBEDS_PER_PAGE:
            findings.append(Finding("warning", "TOO_MANY_EMBEDS", f"{len(embeds)} embeds on one page; each boots the full W&B app (cap is {MAX_EMBEDS_PER_PAGE})", rel, embeds[0][1]))

        prose = strip_embeds(text)
        for src, line in embeds:
            ids = REPORT_ID_RE.findall(src)
            rid = ids[0] if ids else None
            if rid is None:
                findings.append(Finding("error", "BAD_EMBED_SRC", f"src `{src}` is not a recognizable report URL (needs --Vmlldz...)", rel, line))
                continue
            if rid not in by_id:
                findings.append(Finding("error", "EMBED_NOT_IN_REGISTRY", f"report {rid} is embedded here but absent from registry.yaml", rel, line))
            else:
                embedded_on[rid].add(rel)
                if rel not in (by_id[rid].get("pages") or []):
                    findings.append(Finding("error", "PAGE_NOT_LISTED", f"registry entry for {rid} does not list this page in `pages`", rel, line))
            # Prose-link rule: the id must appear somewhere outside the component.
            if rid not in prose:
                findings.append(Finding("error", "MISSING_PROSE_LINK", f"no plain link to report {rid} in the page prose (agents/llms.txt read source, where the iframe is opaque)", rel, line))

    for rid, entry in by_id.items():
        listed = set(entry.get("pages") or [])
        actual = embedded_on[rid]
        if not actual:
            findings.append(Finding("warning", "ORPHAN_ENTRY", f"registry entry {rid} is not embedded by any English page"))
        for stale in sorted(listed - actual):
            findings.append(Finding("error", "STALE_PAGES_FIELD", f"registry entry {rid} lists `{stale}` but that page has no matching embed"))

    return findings


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
                final_url = resp.geturl()
                status = resp.getcode()
                if status != 200:
                    return Finding("error", "URL_NOT_200", f"{url}: final status {status}")
                if any(marker in final_url for marker in LOGIN_MARKERS):
                    return Finding("error", "URL_LOGIN_REDIRECT", f"{url}: redirected to login ({final_url}) — magic link likely revoked")
                return None
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < attempts:
                last_err = f"HTTP {exc.code}"
                time.sleep(delay * (2 ** attempt))
                continue
            return Finding("error", "URL_NOT_200", f"{url}: HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = str(getattr(exc, "reason", exc))
            if attempt < attempts:
                time.sleep(delay * (2 ** attempt))
                continue
    return Finding("error", "URL_UNREACHABLE", f"{url}: unreachable after {attempts} attempts ({last_err})")


def check_liveness(entries: list[dict], *, timeout: int = 30, attempts: int = 3, delay: float = 1.0) -> list[Finding]:
    findings: list[Finding] = []
    for entry in entries:
        url = entry.get("url")
        if not isinstance(url, str):
            continue
        finding = check_url(url, timeout=timeout, attempts=attempts, delay=delay)
        if finding:
            findings.append(finding)
        time.sleep(delay)  # be polite; registry is small
    return findings


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def emit_github_annotations(findings: Iterable[Finding]) -> None:
    for f in findings:
        kind = "error" if f.level == "error" else "warning"
        loc = ""
        if f.path:
            loc = f"file={f.path}"
            if f.line:
                loc += f",line={f.line}"
        print(f"::{kind} {loc}::[{f.code}] {f.message}")


def render_markdown(findings: list[Finding], *, checked_urls: int, embeds: int) -> str:
    errors = [f for f in findings if f.level == "error"]
    warnings = [f for f in findings if f.level == "warning"]
    lines = ["# Report embed check", ""]
    if not findings:
        lines.append(f"✅ All checks passed ({embeds} embed(s), {checked_urls} URL(s) checked).")
        return "\n".join(lines) + "\n"
    lines.append(f"Found {len(errors)} error(s) and {len(warnings)} warning(s).")
    lines.append("")
    lines.append("| Level | Code | Location | Detail |")
    lines.append("| --- | --- | --- | --- |")
    for f in errors + warnings:
        where = f.path or "registry.yaml"
        if f.line:
            where += f":{f.line}"
        lines.append(f"| {f.level} | {f.code} | {where} | {f.message} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["static", "liveness", "all"], default="static")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--github", action="store_true", help="emit GitHub Actions annotations")
    parser.add_argument("--output-md", type=Path, help="write a Markdown report to this file")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args(argv)

    entries, findings = load_registry(args.registry)
    if any(f.code in ("REGISTRY_MISSING", "REGISTRY_UNREADABLE", "REGISTRY_SHAPE") for f in findings):
        for f in findings:
            print(f"ERROR [{f.code}] {f.message}", file=sys.stderr)
        return 2

    embeds_seen = 0
    if args.mode in ("static", "all"):
        findings += check_mdx(args.root, entries)
    if args.mode in ("liveness", "all"):
        findings += check_liveness(entries)

    # Count embeds for the summary line (cheap re-scan of registry pages only).
    embeds_seen = sum(len(e.get("pages") or []) for e in entries)
    checked_urls = len(entries) if args.mode in ("liveness", "all") else 0

    errors = [f for f in findings if f.level == "error"]
    warnings = [f for f in findings if f.level == "warning"]

    if args.github:
        emit_github_annotations(findings)
    else:
        for f in errors + warnings:
            where = f" {f.path}:{f.line}" if f.path and f.line else (f" {f.path}" if f.path else "")
            print(f"{f.level.upper()} [{f.code}]{where}: {f.message}")

    if args.output_md:
        args.output_md.write_text(render_markdown(findings, checked_urls=checked_urls, embeds=embeds_seen), encoding="utf-8")

    if not findings:
        print(f"OK: {len(entries)} registry entrie(s), {embeds_seen} embed page reference(s).")

    if errors:
        return 1
    if warnings and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
