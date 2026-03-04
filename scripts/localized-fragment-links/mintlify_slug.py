#!/usr/bin/env python3
"""
Mintlify-compatible slug generation for heading anchors.

Observed behavior (from local + live docs.wandb.ai):
- Spaces become hyphens.
- ASCII letters are lowercased; other characters (CJK, etc.) are kept.
- Duplicate slugs get -2, -3, ... appended.

Use this to compute the fragment (id) that Mintlify will assign to a heading
so we can rewrite links in ja/ and ko/ content to use the correct fragment.
"""

import re
from collections import Counter


def mintlify_slug(text: str) -> str:
    """
    Produce a single slug from heading text to match Mintlify's anchor ids.

    - Collapse and strip whitespace, then replace spaces with hyphens.
    - Lowercase ASCII letters only; leave other characters unchanged.
    - Strip characters that are not letters (any script), numbers, or hyphens.
    """
    if not text or not isinstance(text, str):
        return ""
    # Collapse whitespace and strip
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return ""
    # Replace spaces with hyphens
    slug = normalized.replace(" ", "-")
    # Lowercase ASCII letters only; keep letters (CJK etc.), digits, hyphens
    result = []
    for c in slug:
        if c.isascii() and c.isalpha():
            result.append(c.lower())
        elif c == "-" or c.isalnum():
            result.append(c)
        # else: drop punctuation etc.
    # Collapse multiple hyphens and strip leading/trailing hyphens
    s = "".join(result)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def mintlify_slugs_for_headings(headings: list[str]) -> list[str]:
    """
    Given an ordered list of heading texts (e.g. from an MDX file), return
    the list of slugs Mintlify would assign, with -2, -3 for duplicates.
    """
    seen: Counter[str] = Counter()
    out: list[str] = []
    for text in headings:
        base = mintlify_slug(text)
        if not base:
            out.append("")
            continue
        count = seen[base]
        seen[base] += 1
        if count == 0:
            out.append(base)
        else:
            out.append(f"{base}-{count + 1}")
    return out


# --- Sanity checks against observed ids -------------------------------------

_OBSERVED = [
    # (heading_text, expected_slug)
    ("Inference サービスへのアクセス", "inference-サービスへのアクセス"),
    ("Playground でモデルを試す", "playground-でモデルを試す"),
    ("複数のモデルを比較する", "複数のモデルを比較する"),
    ("What is Launch?", "what-is-launch"),
    ("開始方法", "開始方法"),
    ("Launch queue", "launch-queue"),
    ("Target resources", "target-resources"),
    ("Launch agent", "launch-agent"),
    ("ローンンチジョブ", "ローンンチジョブ"),
    ("argparse で設定を行う", "argparse-で設定を行う"),
    ("実験設定のセットアップ", "実験設定のセットアップ"),
    # Mixed English + translated (e.g. "W&B" kept, rest localized)
    ("W&B Brand Banner", "wb-brand-banner"),
    ("W&B ブランドバナー", "wb-ブランドバナー"),
]


def _main():
    print("Checking mintlify_slug() against observed ids:\n")
    all_ok = True
    for text, expected in _OBSERVED:
        got = mintlify_slug(text)
        ok = got == expected
        if not ok:
            all_ok = False
        print(f"  {text!r}")
        print(f"    expected {expected!r}  got {got!r}  {'ok' if ok else 'MISMATCH'}")
    print()
    if all_ok:
        print("All checks passed.")
    else:
        print("Some mismatches; adjust mintlify_slug() to match Mintlify.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
