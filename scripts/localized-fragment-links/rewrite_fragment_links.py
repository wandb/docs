#!/usr/bin/env python3
"""
Rewrite fragment links in localized MDX (ja/, ko/) to use correct anchor slugs.

Mintlify generates heading IDs from the heading text. In translated content,
headings are in the locale language, so the fragment (e.g. #how-to-get-started)
does not match the actual id on the page (e.g. #開始方法). This script:

1. Builds a map per localized page: EN fragment -> localized fragment (by
   matching heading order with the EN counterpart).
2. Scans each localized MDX for links that point to localized pages with a
   fragment, and rewrites the fragment when we have a mapping.

Usage:
  python3 rewrite_fragment_links.py [--dry-run] [--locale ja,ko]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Repo root: parent of scripts/ (script lives in scripts/localized-fragment-links/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from headings import heading_texts_only
from mintlify_slug import mintlify_slugs_for_headings
from path_utils import get_en_counterpart, path_to_page_key, resolve_link_target


# Link pattern: [text](url). Captures link text and url.
_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def build_slug_map_for_page(
    en_path: Path,
    localized_path: Path,
) -> dict[str, str]:
    """
    Build mapping EN fragment -> localized fragment by heading position.

    Assumes the Nth heading on the EN page corresponds to the Nth heading
    on the localized page. Only includes entries where the slugs differ.
    Mixed content (e.g. "W&B" kept in English, rest translated) is handled
    correctly: the slug is computed from the full heading text, so each
    heading yields one slug regardless of language mix.
    If EN and localized have different heading counts or order, mapping
    may be wrong for that page; run with --dry-run and -v to inspect.
    """
    en_texts = heading_texts_only(en_path)
    loc_texts = heading_texts_only(localized_path)
    en_slugs = mintlify_slugs_for_headings(en_texts)
    loc_slugs = mintlify_slugs_for_headings(loc_texts)
    mapping: dict[str, str] = {}
    for i, (es, ls) in enumerate(zip(en_slugs, loc_slugs)):
        if es and ls and es != ls:
            mapping[es] = ls
    return mapping


def build_all_slug_maps(
    repo_root: Path,
    locales: list[str],
) -> dict[str, dict[str, str]]:
    """
    Build page_key -> (en_fragment -> localized_fragment) for all localized
    pages that have an EN counterpart and at least one differing slug.
    """
    maps: dict[str, dict[str, str]] = {}
    for locale in locales:
        locale_dir = repo_root / locale
        if not locale_dir.is_dir():
            continue
        for mdx_path in locale_dir.rglob("*.mdx"):
            en_path = get_en_counterpart(mdx_path, repo_root)
            if en_path is None:
                continue
            slug_map = build_slug_map_for_page(en_path, mdx_path)
            if not slug_map:
                continue
            page_key = path_to_page_key(mdx_path, repo_root)
            if page_key:
                maps[page_key] = slug_map
    return maps


def rewrite_content(
    content: str,
    from_path: Path,
    repo_root: Path,
    locale: str,
    slug_maps: dict[str, dict[str, str]],
) -> tuple[str, list[tuple[str, str, str]]]:
    """
    Rewrite fragment links in content. Returns (new_content, list of changes).

    Each change is (page_key, old_fragment, new_fragment). We collect all
    rewrites then apply from end to start so string indices stay valid.
    """
    # Collect (start, end, replacement, (page_key, old_frag, new_frag))
    replacements: list[tuple[int, int, str, tuple[str, str, str]]] = []
    for m in _LINK_PATTERN.finditer(content):
        link_text, url = m.group(1), m.group(2)
        if "#" not in url:
            continue
        resolved = resolve_link_target(url, from_path, repo_root, locale)
        if resolved is None:
            continue
        page_key, fragment = resolved
        new_fragment = slug_maps.get(page_key, {}).get(fragment)
        if new_fragment is None or new_fragment == fragment:
            continue
        path_part, _ = url.split("#", 1)
        new_url = f"{path_part}#{new_fragment}"
        replacement = f"[{link_text}]({new_url})"
        replacements.append((m.start(), m.end(), replacement, (page_key, fragment, new_fragment)))

    if not replacements:
        return content, []

    # Apply from end to start so indices remain valid
    replacements.sort(key=lambda r: r[0], reverse=True)
    result = content
    changes: list[tuple[str, str, str]] = []
    for start, end, replacement, (pk, old_f, new_f) in replacements:
        result = result[:start] + replacement + result[end:]
        changes.append((pk, old_f, new_f))

    return result, changes


def process_file(
    path: Path,
    repo_root: Path,
    locale: str,
    slug_maps: dict[str, dict[str, str]],
    dry_run: bool,
) -> list[tuple[str, str, str]]:
    """
    Rewrite fragment links in one file. Returns list of (page_key, old, new).
    """
    content = path.read_text(encoding="utf-8")
    new_content, changes = rewrite_content(
        content, path, repo_root, locale, slug_maps
    )
    if not changes:
        return []
    if not dry_run:
        path.write_text(new_content, encoding="utf-8")
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite localized MDX fragment links to use correct slugs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files",
    )
    parser.add_argument(
        "--locale",
        default="ja,ko",
        metavar="LIST",
        help="Comma-separated locales (default: ja,ko)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each file and change",
    )
    args = parser.parse_args()
    locales = [s.strip() for s in args.locale.split(",") if s.strip()]

    repo_root = _REPO_ROOT
    if not (repo_root / "ja").is_dir() and not (repo_root / "ko").is_dir():
        print("Run from docs repo root (or parent of ja/ and ko/).", file=sys.stderr)
        return 1

    slug_maps = build_all_slug_maps(repo_root, locales)
    if args.verbose:
        print(f"Built slug maps for {len(slug_maps)} pages.", file=sys.stderr)

    total_changes = 0
    files_changed = 0
    for locale in locales:
        locale_dir = repo_root / locale
        if not locale_dir.is_dir():
            continue
        for mdx_path in sorted(locale_dir.rglob("*.mdx")):
            changes = process_file(
                mdx_path, repo_root, locale, slug_maps, args.dry_run
            )
            if changes:
                files_changed += 1
                total_changes += len(changes)
                rel = mdx_path.relative_to(repo_root)
                if args.verbose or args.dry_run:
                    print(f"{rel}")
                    for page_key, old_f, new_f in changes:
                        print(f"  #{old_f} -> #{new_f}  (target: {page_key})")

    if args.dry_run:
        print(f"\nDry run: would rewrite {total_changes} link(s) in {files_changed} file(s).")
    else:
        print(f"Rewrote {total_changes} link(s) in {files_changed} file(s).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
