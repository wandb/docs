#!/usr/bin/env python3
"""
Extract headings from MDX and compute Mintlify-compatible slugs.

Usage:
  python3 extract_heading_slugs.py <path.mdx>              # list headings + slugs for one file
  python3 extract_heading_slugs.py --en EN.mdx --ja JA.mdx   # compare by position for link rewriting
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "localized-fragment-links"))
from headings import extract_headings
from mintlify_slug import mintlify_slug, mintlify_slugs_for_headings


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract MDX headings and compute slugs")
    parser.add_argument("path", nargs="?", help="Path to MDX file (e.g. ja/launch/launch-terminology.mdx)")
    parser.add_argument("--en", metavar="PATH", help="English MDX path (for --ja comparison)")
    parser.add_argument("--ja", metavar="PATH", help="Japanese MDX path (for --en comparison)")
    args = parser.parse_args()

    if args.en and args.ja:
        en_path = REPO_ROOT / args.en
        ja_path = REPO_ROOT / args.ja
        if not en_path.is_file():
            print(f"EN file not found: {en_path}", file=sys.stderr)
            return 1
        if not ja_path.is_file():
            print(f"JA file not found: {ja_path}", file=sys.stderr)
            return 1
        en_headings = [t for _, t in extract_headings(en_path)]
        ja_headings = [t for _, t in extract_headings(ja_path)]
        en_slugs = mintlify_slugs_for_headings(en_headings)
        ja_slugs = mintlify_slugs_for_headings(ja_headings)
        print("Position | EN heading (slug) -> JA heading (slug)\n")
        for i, (eh, es, jh, js) in enumerate(
            zip(en_headings, en_slugs, ja_headings, ja_slugs)
        ):
            match = "same slug" if es == js else f"rewrite #{es} -> #{js}"
            print(f"  {i}  | {eh[:50]!r} (#{es})")
            print(f"      -> {jh[:50]!r} (#{js})  [{match}]")
            print()
        if len(en_headings) != len(ja_headings):
            print(f"Warning: EN has {len(en_headings)} headings, JA has {len(ja_headings)}")
        return 0

    if not args.path:
        parser.print_help()
        return 0

    path = Path(args.path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    headings = extract_headings(path)
    texts = [t for _, t in headings]
    slugs = mintlify_slugs_for_headings(texts)
    print(f"Headings and slugs for {path.relative_to(REPO_ROOT)}\n")
    for (lev, text), slug in zip(headings, slugs):
        print(f"  {'#' * lev} {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"      -> #{slug}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
