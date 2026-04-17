"""
Extract Markdown and HTML headings from MDX files.

Used by both the extract_heading_slugs CLI and the fragment-link rewriter
to get ordered heading text for slug computation. Mintlify generates
anchors from both ## / ### and <h2> / <h3> in document order, so we
extract both and merge by position.
"""

import re
from pathlib import Path


# Markdown: ## or ### at start of line
_MD_PATTERN = re.compile(r"^#{2,3}\s+(.+)$", re.MULTILINE)

# HTML/JSX: <h2>...</h2> or <h3>...</h3>, with optional attributes
_HTML_PATTERN = re.compile(
    r"<h([23])(?:\s[^>]*)?>(.*?)</h\1>",
    re.DOTALL,
)


def _text_from_html_content(raw: str) -> str:
    """Strip inner tags and normalize whitespace for slug-relevant text."""
    text = re.sub(r"<[^>]+>", "", raw)
    return " ".join(text.split()).strip()


def extract_headings(mdx_path: Path) -> list[tuple[int, str]]:
    """
    Return list of (level, text) for ## / ### and <h2> / <h3> in document order.

    Level is 2 or 3. Text is stripped. Markdown and HTML headings are both
    included and ordered by their position in the file so the list matches
    the order Mintlify uses for anchor generation.
    """
    text = mdx_path.read_text(encoding="utf-8")
    matches: list[tuple[int, int, str]] = []  # (start_pos, level, text)

    for m in _MD_PATTERN.finditer(text):
        raw = m.group(0)
        title = m.group(1).strip()
        level = len(raw) - len(title) - 1
        if level in (2, 3):
            matches.append((m.start(), level, title))

    for m in _HTML_PATTERN.finditer(text):
        level = int(m.group(1))
        title = _text_from_html_content(m.group(2))
        if title:
            matches.append((m.start(), level, title))

    matches.sort(key=lambda x: x[0])
    return [(lev, t) for _, lev, t in matches]


def heading_texts_only(mdx_path: Path) -> list[str]:
    """Return heading text in order (for slug generation)."""
    return [text for _, text in extract_headings(mdx_path)]
