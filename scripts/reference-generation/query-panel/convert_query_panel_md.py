#!/usr/bin/env python3
"""
Convert weave-js docs_gen Markdown into Mintlify MDX for models/ref/query-panel/.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# "A [artifactType](..." in table cells (pipe, spaces, A, link with vowel-start text).
_VOWEL_LINK = re.compile(
    r"(\|\s+)(A|a)(\s+\[)([aeiouAEIOU][^\]]*)(\]\()",
)

# Undo mistaken "An" when upstream or a prior run used vowel-letter heuristics only.
_WRONG_AN_BEFORE_LINK = re.compile(
    r"(\|\s+)(An|an)(\s+\[)([aeiouAEIOU][^\]]*)(\]\()",
)


LEGACY_REF_RE = re.compile(
    r"https://docs\.wandb\.ai/ref/weave/([A-Za-z0-9-]+)(/?)(?=[)\s])"
)

# Mintlify prefers slash-less paths; strip trailing slash after datatype slug.
_INTERNAL_QUERY_PANEL_SLASH = re.compile(
    r"(/models/ref/query-panel/[A-Za-z0-9-]+)/",
)

H3_HTML_RE = re.compile(
    r'<h3 id="([^"]+)"><code>([^<]+)</code></h3>\s*\n',
    re.MULTILINE,
)


def rewrite_internal_links(md: str) -> str:
    def repl(m: re.Match[str]) -> str:
        slug = m.group(1)
        return f"/models/ref/query-panel/{slug}"

    return LEGACY_REF_RE.sub(repl, md)


def strip_query_panel_link_trailing_slashes(md: str) -> str:
    """Remove trailing slashes from /models/ref/query-panel/<slug>/ (Mintlify redirect)."""

    def repl(m: re.Match[str]) -> str:
        return m.group(1)

    return _INTERNAL_QUERY_PANEL_SLASH.sub(repl, md)


def fix_argument_table_header(md: str) -> str:
    return md.replace("| Argument |  |", "| Argument | Description |")


def _link_label_takes_an(link_text: str) -> bool:
    """
    Whether prose before [link_text] should use \"an\" (vowel sound).

    The generator only sees the bracket label, not pronunciation. Several
    English words and technical names start with a vowel letter but a
    consonant glide (/j/ or /w/), for example \"user\" and \"Unicode\".
    """
    label = link_text.strip()
    if not label:
        return True
    word = re.split(r"\s+", label, maxsplit=1)[0]
    wl = word.lower()

    # Written vowel at start but consonant sound (/j/ or /w/).
    if wl.startswith(
        (
            "uni",
            "use",
            "usu",
            "util",
            "uri",
            "url",
            "uuid",
            "euro",
        )
    ):
        return False
    if wl == "one" or wl.startswith("one"):
        return False

    return True


def fix_indefinite_article_before_links(md: str) -> str:
    """Fix 'A [artifactType](...' in table rows (upstream uses 'A' before type links)."""

    def repl_a_to_an(m: re.Match[str]) -> str:
        p1, article, p3, link_text, p5 = m.groups()
        if not _link_label_takes_an(link_text):
            return m.group(0)
        new_art = "An" if article == "A" else "an"
        return f"{p1}{new_art}{p3}{link_text}{p5}"

    def repl_an_to_a(m: re.Match[str]) -> str:
        p1, article, p3, link_text, p5 = m.groups()
        if _link_label_takes_an(link_text):
            return m.group(0)
        new_art = "A" if article == "An" else "a"
        return f"{p1}{new_art}{p3}{link_text}{p5}"

    md = _VOWEL_LINK.sub(repl_a_to_an, md)
    md = _WRONG_AN_BEFORE_LINK.sub(repl_an_to_a, md)
    return md


def extract_hash_title(raw_md: str) -> str | None:
    """First markdown h1 line from upstream, if present."""
    lines = raw_md.strip().splitlines()
    if lines and lines[0].startswith("# "):
        return lines[0][2:].strip()
    return None


def page_display_title(slug: str, raw_md: str, labels: dict[str, str]) -> str:
    """
    Mintlify page title: prefer the overview link label (matches historical docs and
    QEL spellings like artifactType), else upstream # heading, else URL slug.
    """
    if slug in labels:
        return labels[slug]
    return extract_hash_title(raw_md) or slug


def format_yaml_title(title: str) -> str:
    """Emit a single-line YAML title value (quote when needed)."""
    if re.fullmatch(r"[\w.-]+", title):
        return title
    return json.dumps(title, ensure_ascii=False)


def html_h3_to_markdown(md: str) -> str:
    def repl(m: re.Match[str]) -> str:
        op_id, code = m.group(1), m.group(2)
        return f'### <a id="{op_id}"></a>`{code}`\n\n'

    return H3_HTML_RE.sub(repl, md)


def uniquify_anchor_ids_in_section(section: str) -> str:
    """Ensure each <a id="..."> appears at most once within this section."""
    counts: dict[str, int] = {}

    def repl(m: re.Match[str]) -> str:
        op_id, code = m.group(1), m.group(2)
        n = counts.get(op_id, 0)
        counts[op_id] = n + 1
        new_id = op_id if n == 0 else f"{op_id}-{n + 1}"
        return f'### <a id="{new_id}"></a>`{code}`'

    return re.sub(
        r'### <a id="([^"]+)"></a>`([^`]+)`',
        repl,
        section,
    )


def uniquify_anchor_ids_by_ops_section(md: str) -> str:
    chain_tag = "## Chainable Ops\n"
    list_tag = "## List Ops\n"
    if chain_tag not in md or list_tag not in md:
        return uniquify_anchor_ids_in_section(md)
    pre, rest = md.split(chain_tag, 1)
    chain_body, list_rest = rest.split(list_tag, 1)
    chain_body = uniquify_anchor_ids_in_section(chain_body)
    list_body = uniquify_anchor_ids_in_section(list_rest)
    return pre + chain_tag + chain_body + list_tag + list_body


def dedupe_list_ops_anchors(md: str) -> str:
    """List Ops repeat the same op names as Chainable Ops; suffix anchor ids in List Ops."""
    marker = "\n## List Ops\n"
    if marker not in md:
        return md
    head, tail = md.split(marker, 1)

    def fix_list_ids(m: re.Match[str]) -> str:
        op_id, code = m.group(1), m.group(2)
        if op_id.endswith("-list"):
            return m.group(0)
        return f'### <a id="{op_id}-list"></a>`{code}`'

    tail = re.sub(
        r'### <a id="([^"]+)"></a>`([^`]+)`',
        fix_list_ids,
        tail,
    )
    return head + marker + tail


def strip_title_heading(md: str, slug: str) -> str:
    lines = md.splitlines()
    if lines and lines[0] == f"# {slug}":
        return "\n".join(lines[1:]).lstrip("\n")
    if lines and lines[0].startswith("# "):
        return "\n".join(lines[1:]).lstrip("\n")
    return md


def slug_from_filename(name: str) -> str:
    assert name.endswith(".md")
    return name[: -len(".md")]


def build_mdx_body(raw_md: str, slug: str) -> str:
    body = strip_title_heading(raw_md.strip(), slug)
    body = rewrite_internal_links(body)
    body = strip_query_panel_link_trailing_slashes(body)
    body = fix_argument_table_header(body)
    body = fix_indefinite_article_before_links(body)
    body = html_h3_to_markdown(body)
    body = uniquify_anchor_ids_by_ops_section(body)
    body = dedupe_list_ops_anchors(body)
    return body.rstrip() + "\n"


def build_mdx_file(title: str, body: str) -> str:
    title_yaml = format_yaml_title(title)
    return (
        "---\n"
        f"title: {title_yaml}\n"
        "---\n"
        f"{body}"
    )


def discover_slugs(docs_gen: Path) -> list[str]:
    slugs = []
    for p in docs_gen.glob("*.md"):
        if p.name.lower() == "readme.md":
            continue
        slugs.append(slug_from_filename(p.name))
    return sorted(slugs)


def parse_landing_link_labels(landing_text: str) -> dict[str, str]:
    """Map slug -> link label from existing landing bullets."""
    labels: dict[str, str] = {}
    for m in re.finditer(
        r"^\* \[([^\]]+)\]\(\./query-panel/([^)]+)\)\s*$", landing_text, re.MULTILINE
    ):
        labels[m.group(2)] = m.group(1)
    return labels


def replace_generated_types_section(landing_text: str, bullets: str) -> str:
    # MDX does not accept HTML comments; use JSX comments for Mintlify.
    start = "{/* query-panel-generated-data-types:start */}"
    end = "{/* query-panel-generated-data-types:end */}"
    if start in landing_text and end in landing_text:
        pre, rest = landing_text.split(start, 1)
        _, post = rest.split(end, 1)
        # Avoid accumulating blank lines: the replacement ends with a newline and
        # post often begins with one (or many from prior runs). Normalize to a
        # single blank line before the following section, or a single trailing
        # newline when nothing follows the end marker.
        post = post.lstrip("\n")
        block_core = f"{start}\n{bullets.rstrip()}\n{end}"
        if post:
            return pre + block_core + "\n\n" + post
        return pre + block_core + "\n"

    legacy = re.compile(
        r"(## Data Types\n\n)(?:\* \[[^\]]+\]\(\./query-panel/[^\)]+\)\n)+",
        re.MULTILINE,
    )
    if not legacy.search(landing_text):
        raise ValueError(
            "landing page must contain ## Data Types list or generator markers"
        )
    block = (
        "## Data Types\n\n"
        f"{start}\n{bullets.rstrip()}\n{end}\n"
    )
    return legacy.sub(block, landing_text, count=1)


def build_type_bullets(slugs: list[str], labels: dict[str, str]) -> str:
    lines = []
    for slug in slugs:
        label = labels.get(slug, slug)
        lines.append(f"* [{label}](./query-panel/{slug})")
    return "\n".join(lines) + "\n"


def update_docs_json_nav(docs_json_path: Path, slugs: list[str]) -> None:
    with docs_json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    def build_pages(prefix: str) -> list[str]:
        base = f"{prefix}models/ref/query-panel"
        return [base] + [f"{base}/{s}" for s in slugs]

    targets = {
        "en": build_pages(""),
        "ja": build_pages("ja/"),
        "ko": build_pages("ko/"),
    }

    def walk(node: object, lang: str | None) -> None:
        if isinstance(node, dict):
            if (
                node.get("group") == "Query Expression Language"
                and isinstance(node.get("pages"), list)
                and lang in targets
            ):
                node["pages"] = targets[lang]
            for v in node.values():
                walk(v, lang)
        elif isinstance(node, list):
            for item in node:
                walk(item, lang)

    for lang_entry in data.get("navigation", {}).get("languages", []):
        lang = lang_entry.get("language")
        if not isinstance(lang, str):
            continue
        walk(lang_entry, lang)

    with docs_json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-gen", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--landing", type=Path, required=True)
    ap.add_argument("--docs-json", type=Path, required=True)
    args = ap.parse_args()

    docs_gen: Path = args.docs_gen
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    slugs = discover_slugs(docs_gen)
    if not slugs:
        raise SystemExit(f"No *.md files under {docs_gen}")

    landing_text = args.landing.read_text(encoding="utf-8")
    labels = parse_landing_link_labels(landing_text)
    bullets = build_type_bullets(slugs, labels)
    new_landing = replace_generated_types_section(landing_text, bullets)
    args.landing.write_text(new_landing, encoding="utf-8")

    for slug in slugs:
        src = docs_gen / f"{slug}.md"
        raw = src.read_text(encoding="utf-8")
        display = page_display_title(slug, raw, labels)
        body = build_mdx_body(raw, slug)
        out = build_mdx_file(display, body)
        (out_dir / f"{slug}.mdx").write_text(out, encoding="utf-8")

    update_docs_json_nav(args.docs_json, slugs)
    print(f"Wrote {len(slugs)} pages to {out_dir}")
    print(f"Updated {args.landing} and {args.docs_json}")


if __name__ == "__main__":
    main()
