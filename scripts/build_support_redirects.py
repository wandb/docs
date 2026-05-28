#!/usr/bin/env python3
"""
Build redirect mapping and delete list for models/support -> support/models migration.

Reads:
- snippets/kb_article_map.mdx (page + title per article)
- support/models/articles/*.mdx (frontmatter title)
- models/support/ and support/models/tags/ for tag page mapping

Outputs (to repo or /tmp):
- Redirects: source|destination (one per line) or JSON array
- List of models/support files to delete (full paths)

Do not edit docs or docs.json.
"""

import json
import re
import sys
from pathlib import Path


def normalize_title(s: str) -> str:
    """Lowercase, strip punctuation (keep letters/numbers), collapse spaces."""
    if not s:
        return ""
    # Normalize Unicode apostrophes/quotes to ASCII for matching
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u0026", "&")  # if present as literal
    s = s.lower().strip()
    # Normalize W&B / wandb to same form for matching
    s = re.sub(r"w\s*&\s*b", " wandb ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    # Remove backticks and strip punctuation (except spaces)
    s = re.sub(r"`", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def slugify_tag(tag: str) -> str:
    """Tag name to URL slug: lowercase, spaces -> hyphens."""
    return tag.lower().strip().replace(" ", "-")


def _decode_unicode_escapes(s: str) -> str:
    """Decode \\uXXXX and \\UXXXXXXXX in string (e.g. from YAML frontmatter)."""
    def repl(m):
        return chr(int(m.group(1), 16))
    s = re.sub(r"\\u([0-9a-fA-F]{4})", repl, s)
    s = re.sub(r"\\U([0-9a-fA-F]{8})", repl, s)
    return s


def extract_frontmatter_title(filepath: Path) -> str | None:
    """Read first few lines and parse title from YAML frontmatter."""
    try:
        text = filepath.read_text(encoding="utf-8")
    except Exception:
        return None
    if not text.startswith("---"):
        return None
    end = text.find("---", 3)
    if end == -1:
        return None
    block = text[3:end]
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("title:"):
            title = line[6:].strip()
            if title.startswith('"') and title.endswith('"'):
                title = title[1:-1]
            elif title.startswith("'") and title.endswith("'"):
                title = title[1:-1]
            title = _decode_unicode_escapes(title)
            return title
    return None


def parse_kb_article_map(mdx_path: Path) -> list[dict]:
    """Extract the JS array from kb_article_map.mdx."""
    text = mdx_path.read_text(encoding="utf-8")
    # export const kbArticleMap = [ ... ];
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end <= start:
        return []
    json_str = text[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    # Try with relaxed parsing (trailing commas)
    json_str = re.sub(r",\s*]", "]", json_str)
    json_str = re.sub(r",\s*}", "}", json_str)
    return json.loads(json_str)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    kb_path = repo / "snippets" / "kb_article_map.mdx"
    articles_dir = repo / "support" / "models" / "articles"
    old_support_dir = repo / "models" / "support"
    new_tags_dir = repo / "support" / "models" / "tags"

    if not kb_path.exists():
        print("kb_article_map.mdx not found", file=sys.stderr)
        sys.exit(1)
    if not articles_dir.is_dir():
        print("support/models/articles not found", file=sys.stderr)
        sys.exit(1)

    # 1) Load kb article map
    kb_entries = parse_kb_article_map(kb_path)
    print(f"Loaded {len(kb_entries)} entries from kb_article_map", file=sys.stderr)

    # 2) Build title -> new article slug from support/models/articles
    title_to_slug: dict[str, str] = {}
    for mdx in sorted(articles_dir.glob("*.mdx")):
        title = extract_frontmatter_title(mdx)
        if title:
            norm = normalize_title(title)
            slug = mdx.stem
            if norm and slug:
                title_to_slug[norm] = slug
    print(f"Indexed {len(title_to_slug)} article titles from support/models/articles", file=sys.stderr)

    # 3) Article redirects: /models/support/<old_slug> -> /support/models/articles/<new_slug>
    redirects: list[tuple[str, str]] = []
    seen_old_slugs: dict[str, list[str]] = {}
    for entry in kb_entries:
        page = entry.get("page") or ""
        title = entry.get("title") or ""
        if not page.startswith("/models/support/"):
            continue
        old_slug = page.rstrip("/").split("/")[-1]
        norm = normalize_title(title)
        new_slug = title_to_slug.get(norm)
        if new_slug:
            source = f"/models/support/{old_slug}"
            dest = f"/support/models/articles/{new_slug}"
            redirects.append((source, dest))
            seen_old_slugs.setdefault(old_slug, []).append(dest)
        else:
            print(f"No match for kb title: {title!r} (old_slug={old_slug})", file=sys.stderr)

    # Deduplicate redirects: same source should point to one destination; keep first by source
    by_source: dict[str, str] = {}
    for s, d in redirects:
        if s not in by_source:
            by_source[s] = d
        elif by_source[s] != d:
            print(f"Conflict: {s} -> {by_source[s]} vs {d}", file=sys.stderr)
    article_redirects = [(s, by_source[s]) for s in sorted(by_source)]

    # 4) Tag mapping: old tag slug -> new tag slug (when different)
    # Old has "crashing-and-hanging-runs", new has "run-crashes"
    tag_old_to_new: dict[str, str] = {
        "crashing-and-hanging-runs": "run-crashes",
    }
    new_tag_files = {f.stem for f in new_tags_dir.glob("*.mdx")}

    # Old tag pages = files in models/support that have a counterpart in support/models/tags
    # (same stem, or old stem in tag_old_to_new mapping)
    old_tag_pages: set[str] = set()
    for f in old_support_dir.glob("*.mdx"):
        slug = f.stem
        if slug in new_tag_files:
            old_tag_pages.add(slug)
        elif slug in tag_old_to_new:
            old_tag_pages.add(slug)

    tag_redirects: list[tuple[str, str]] = []
    for old_slug in sorted(old_tag_pages):
        new_slug = tag_old_to_new.get(old_slug, old_slug)
        if new_slug not in new_tag_files:
            continue
        source = f"/models/support/{old_slug}"
        dest = f"/support/models/tags/{new_slug}"
        tag_redirects.append((source, dest))

    # 5) All redirects (articles + tags), no trailing slashes in destinations
    all_redirects = article_redirects + tag_redirects
    # Remove any trailing slash from destination
    all_redirects = [(s, d.rstrip("/")) for s, d in all_redirects]

    # 6) Files to delete: all files under models/support/ (repo-relative paths)
    to_delete_rel = sorted(str(p.relative_to(repo)) for p in old_support_dir.glob("*.mdx"))
    to_delete_full = [str(p.resolve()) for p in old_support_dir.glob("*.mdx")]

    # Output directory: repo root (user said "in the repo or /tmp")
    out_dir = repo
    redirects_txt = out_dir / "support_redirects.txt"
    redirects_json = out_dir / "support_redirects.json"
    delete_list_txt = out_dir / "models_support_files_to_delete.txt"
    tag_mapping_txt = out_dir / "support_tag_mapping.txt"

    # Write redirects: one per line source|destination
    with open(redirects_txt, "w", encoding="utf-8") as f:
        for s, d in all_redirects:
            f.write(f"{s}|{d}\n")

    # Also JSON array
    with open(redirects_json, "w", encoding="utf-8") as f:
        json.dump([{"source": s, "destination": d} for s, d in all_redirects], f, indent=2)

    # Write delete list: repo-relative paths (portable); one line each
    with open(delete_list_txt, "w", encoding="utf-8") as f:
        for p in to_delete_rel:
            f.write(p + "\n")

    # Write tag mapping (old_slug -> new_slug)
    with open(tag_mapping_txt, "w", encoding="utf-8") as f:
        f.write("old_tag_slug|new_tag_slug\n")
        for old_slug in sorted(old_tag_pages):
            new_slug = tag_old_to_new.get(old_slug, old_slug)
            if new_slug in new_tag_files:
                f.write(f"{old_slug}|{new_slug}\n")

    print(f"Wrote {len(all_redirects)} redirects to {redirects_txt} and {redirects_json}", file=sys.stderr)
    print(f"Wrote {len(to_delete_rel)} paths to {delete_list_txt}", file=sys.stderr)
    print(f"Wrote tag mapping to {tag_mapping_txt}", file=sys.stderr)
    print(f"Article redirects: {len(article_redirects)}", file=sys.stderr)
    print(f"Tag redirects: {len(tag_redirects)}", file=sys.stderr)


if __name__ == "__main__":
    main()
