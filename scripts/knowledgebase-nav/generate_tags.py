#!/usr/bin/env python3
"""
Knowledgebase Nav Generator
=========================

A standalone script that regenerates knowledgebase nav pages and updates the
docs.json navigation for the Weights & Biases documentation repository.

This script is designed to run as part of a GitHub Actions workflow, but can
also be run locally for testing and previewing changes.

What it does (six phases)
-------------------------

Phase 1. Crawl and parse:
    Walks each product's articles/ directory and parses YAML front matter and
    the main article body from every MDX file (see ``crawl_articles``). The
    tag-to-articles map is built in the next step inside the pipeline.

Phase 2. Generate tag pages and clean up stale ones:
    For each (product, tag) pair, renders a tag page MDX file containing
    Mintlify Card components that link to the tagged articles. After
    writing current pages, ``cleanup_stale_tag_pages`` deletes any
    ``.mdx`` files in the tags directory that no longer correspond to a
    keyword used by any article.

Phase 3. Generate product index pages (group-generator pattern):
    For each product, renders a product index MDX page with a "Featured
    articles" section (if any articles have featured: true) and a "Browse
    by category" section listing all tags with article counts.

Phase 4. Sync tab-page Badges:
    For each support article, rewrites only ``<Badge>`` links to
    ``/support/<product>/tags/...`` so they match ``keywords`` in front matter.
    Tech writers do not maintain those Badges by hand. Badges are wrapped in
    MDX comment markers (``_BADGE_START`` / ``_BADGE_END``) so the generator
    can find them without regex matching on subsequent runs. Articles that
    predate the markers are migrated automatically on the first run. Runs
    after tag pages are generated so articles are not modified if earlier
    phases fail.

Phase 5. Update docs.json navigation (meta-generator pattern):
    Reads the existing docs.json, finds or creates hidden support tabs for
    each product, updates the page lists to reflect current tags, and writes
    the file back while preserving all unrelated navigation entries.

Phase 6. Update support.mdx (meta-generator pattern):
    Refreshes article and tag counts on the root support landing page
    (count lines are wrapped in ``{/* auto-generated counts */}`` markers
    so writers can add other content in each Card body), and regenerates
    the featured-articles section from articles that have
    ``featured: true`` in front matter.

Inputs
------
- Article MDX files under support/<product>/articles/*.mdx
- Configuration file (config.yaml) listing products and allowed keywords
- Jinja2 templates in the templates/ directory
- The existing docs.json in the repo root

Outputs
-------
- Updated support article MDX files at support/<product>/articles/*.mdx
  (only ``<Badge>`` components whose Markdown link targets
  ``/support/<product>/tags/...`` are rewritten from ``keywords``; other
  Badges and body text are left alone. Managed Badges are wrapped in
  ``_BADGE_START`` / ``_BADGE_END`` marker comments. If no such Badges
  exist yet, a blank line, markers, and tab Badges are appended when
  ``keywords`` is non-empty)
- Tag page MDX files at support/<product>/tags/<tag-slug>.mdx
- Product index MDX files at support/<product>.mdx
- Updated docs.json with correct support navigation tabs
- Updated support.mdx product card counts (inside marker comments)
  and featured-articles section (inside marker comments)

Usage
-----
    python generate_tags.py --repo-root /path/to/wandb-docs

The --repo-root argument should point to the root of the wandb-docs repo
(the directory that contains docs.json and the support/ folder).
"""

import argparse
import html
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from jinja2 import Environment, FileSystemLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Plain-text preview length for tag and index Cards (after stripping MDX or Markdown).
BODY_PREVIEW_MAX_LENGTH = 120

# The suffix appended to truncated body previews.  The leading space keeps
# the ellipsis visually separated from the last word.
BODY_PREVIEW_SUFFIX = " ..."

# Curly and typographic quotation marks and apostrophes mapped to ASCII before
# the preview allowlist so text like "Python's" does not become "Python s".
_PLAIN_TEXT_TYPOGRAPHIC_TO_ASCII = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u2032": "'",
        "\u02bc": "'",
        "\u00b4": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2033": '"',
    }
)

# The language entry in docs.json that contains the English navigation.
# The W&B docs site uses a multi-language navigation structure under
# navigation.languages[]; we only modify the English ("en") entry.
DOCS_JSON_NAV_LANGUAGE = "en"


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and validate the generator configuration from a YAML file.

    The configuration file defines which products to process and which
    keywords (tags) are allowed for each product.  See config.yaml for
    the expected structure.

    Parameters
    ----------
    config_path : Path
        Absolute or relative path to the YAML configuration file.

    Returns
    -------
    dict
        The parsed configuration dictionary with a "products" key containing
        a list of product definitions.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the configuration is missing required fields or is malformed.

    Example
    -------
    >>> config = load_config(Path("config.yaml"))
    >>> config["products"][0]["slug"]
    'models'
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config or "products" not in config:
        raise ValueError(
            f"Configuration file {config_path} must contain a 'products' key "
            "with a list of product definitions."
        )

    for i, product in enumerate(config["products"]):
        if "slug" not in product:
            raise ValueError(
                f"Product at index {i} in {config_path} is missing 'slug'."
            )
        if "display_name" not in product:
            raise ValueError(
                f"Product '{product.get('slug', '?')}' in {config_path} "
                "is missing 'display_name'."
            )
        if "allowed_keywords" not in product:
            raise ValueError(
                f"Product '{product['slug']}' in {config_path} is missing "
                "'allowed_keywords'."
            )

    return config


# ---------------------------------------------------------------------------
# Front matter parsing
# ---------------------------------------------------------------------------

def parse_frontmatter(file_path: Path) -> Tuple[Dict[str, Any], str]:
    """
    Parse YAML front matter and body text from an MDX file.

    MDX files in the W&B docs repo follow this structure::

        ---
        title: "Article title"
        keywords: ["Tag1", "Tag2"]
        featured: true
        ---

        Body content here...

        ---

        <Badge ...>...</Badge>

    This function extracts the YAML front matter (between the first pair
    of ``---`` delimiters) and the body text (everything after the closing
    ``---`` of the front matter, up to but not including the trailing
    ``---`` separator that precedes the Badge footer).

    Parameters
    ----------
    file_path : Path
        Path to the MDX file to parse.

    Returns
    -------
    tuple of (dict, str)
        A two-element tuple:
        - The parsed front matter as a dictionary.  Missing keys default
          to sensible values (empty list for keywords, False for featured).
        - The body text as a string, with the Badge footer stripped.

    Raises
    ------
    ValueError
        If the file does not start with ``---`` (no valid front matter).

    Example
    -------
    >>> fm, body = parse_frontmatter(Path("support/models/articles/example.mdx"))
    >>> fm["title"]
    'Example article'
    >>> fm["keywords"]
    ['Experiments', 'Metrics']
    """
    text = file_path.read_text(encoding="utf-8")

    # Front matter is enclosed between the first two "---" lines.
    # We split on the "---" delimiter to extract it.
    if not text.startswith("---"):
        raise ValueError(f"File {file_path} does not start with '---' (no front matter).")

    # Split into at most 3 parts: before first ---, front matter, rest
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"File {file_path} has malformed front matter (missing closing '---').")

    frontmatter_str = parts[1]
    body_and_footer = parts[2]

    # Parse the YAML front matter
    frontmatter = yaml.safe_load(frontmatter_str) or {}

    # Body is the text after front matter, stopping before the keyword footer.
    # The footer starts at the last occurrence of the substring "\n---" in the
    # remainder (horizontal rule before Badges). If that never appears, the
    # whole remainder is treated as body.
    body = _extract_body(body_and_footer)

    return frontmatter, body


def _extract_body(body_and_footer: str) -> str:
    """
    Return the article body, excluding the auto-managed badge footer.

    Uses ``_BADGE_START`` as the boundary: everything before it is the
    body.  A trailing ``---`` line (the cosmetic horizontal rule writers
    sometimes place before badges) is stripped, but ``---`` has no
    special delimiter role; horizontal rules elsewhere in the body are
    preserved.

    Parameters
    ----------
    body_and_footer : str
        The text after the front matter closing ``---``.

    Returns
    -------
    str
        The body text with leading/trailing whitespace stripped.
    """
    marker_pos = body_and_footer.find(_BADGE_START)
    if marker_pos != -1:
        body = body_and_footer[:marker_pos]
    else:
        body = body_and_footer

    body = re.sub(r"\n---\s*$", "", body)
    return body.strip()


# ---------------------------------------------------------------------------
# Body preview generation
# ---------------------------------------------------------------------------

def plain_text(body: str) -> str:
    """
    Convert article body text to plain text safe for embedding in MDX Cards.

    Strips common Markdown, raw URLs, HTML, and MDX or JSX so previews do not
    show link syntax, tags, or emphasis markers. Processing is heuristic (not
    a full parser). Single-underscore ``_emphasis_`` is only removed when
    the underscores are not adjacent to letters or digits (so snake_case
    identifiers survive). A final allowlist removes other stray symbols.

    Steps (order matters): fenced code blocks, HTML comments, horizontal rules,
    reference links,
    inline links and images, autolinks, repeated angle-bracket tags, MDX
    ``{...}`` expressions, inline backticks, emphasis, headings and list
    markers, footnote refs, bare ``http(s)`` URLs, ``html.unescape`` (then
    replace U+00A0 with a normal space), map typographic quotes to ASCII,
    allowlist, and whitespace collapse.

    Allowed characters after cleanup: letters, digits, space, underscore,
    equals, and ``.,:;!?&'"()-/@+``.

    Parameters
    ----------
    body : str
        The raw article body text (Markdown/MDX content).

    Returns
    -------
    str
        A single-line plain text string.

    Example
    -------
    >>> plain_text("Use `wandb.init()` to **start** a run.")
    'Use wandb.init() to start a run.'
    >>> plain_text("See [docs](https://wandb.ai) for **more**.")
    'See docs for more.'
    """
    text = body

    # Fenced code blocks (drop content; not suitable for short previews).
    text = re.sub(r"```[\w.-]*\s*[\s\S]*?```", " ", text)

    # HTML comments.
    text = re.sub(r"<!--[\s\S]*?-->", " ", text)

    # Horizontal rules (---, ***, ___) on their own line.
    text = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", " ", text)

    # Markdown reference-style links: [label][ref] (including [label][]).
    text = re.sub(r"\[([^\]]+)\]\s*\[[^\]]*\]", r"\1", text)

    # Markdown inline links [text](url): keep visible label only.
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)

    # Markdown images ![alt](url): keep alt text if any.
    text = re.sub(
        r"!\[([^\]]*)\]\([^)]*\)",
        lambda m: (m.group(1) if m.group(1).strip() else " "),
        text,
    )

    # Autolinks <https://...> and <mailto:...>
    text = re.sub(r"<https?://[^>\s]+>", " ", text)
    text = re.sub(r"<mailto:[^>\s]+>", " ", text)

    # HTML / MDX / JSX tags: remove <...> segments until stable (handles simple nesting).
    for _ in range(64):
        new_t = re.sub(r"<[^>]+>", " ", text)
        if new_t == text:
            break
        text = new_t

    # MDX expression braces (single level, common in components).
    for _ in range(32):
        new_t = re.sub(r"\{[^{}]+\}", " ", text)
        if new_t == text:
            break
        text = new_t

    # Inline code: `code` (repeat for rare doubled segments).
    for _ in range(16):
        new_t = re.sub(r"`([^`]+)`", r"\1", text)
        if new_t == text:
            break
        text = new_t
    text = text.replace("`", " ")

    # Bold / strong, then italic (iterations handle nested markers loosely).
    for _ in range(16):
        new_t = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        if new_t == text:
            break
        text = new_t
    for _ in range(16):
        new_t = re.sub(r"__([^_]+)__", r"\1", text)
        if new_t == text:
            break
        text = new_t
    for _ in range(16):
        new_t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
        if new_t == text:
            break
        text = new_t
    # Single-underscore emphasis only when delimiters are not inside a word
    # (avoid turning snake_case like my_awesome_fn into myawesomefn).
    for _ in range(16):
        new_t = re.sub(
            r"(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])",
            r"\1",
            text,
        )
        if new_t == text:
            break
        text = new_t

    # Markdown headings and blockquote prefixes at line starts.
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    # Simple unordered list markers at line start.
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)

    # Footnote references like [^note] (after link extraction).
    text = re.sub(r"\[\^[^\]]+\]", " ", text)

    # Bare http(s) URLs (after markdown links removed their parentheses).
    text = re.sub(r"https?://[^\s<>\")\]]+", " ", text)

    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = text.translate(_PLAIN_TEXT_TYPOGRAPHIC_TO_ASCII)

    # Allowlist: drop anything that could still be markup or stray symbols.
    text = re.sub(r"[^a-zA-Z0-9 .,:;!?&'\"()\-/_=@+]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_body_preview(body: str, max_len: int = BODY_PREVIEW_MAX_LENGTH) -> str:
    """
    Create a short preview snippet from article body text.

    Converts the body to plain text, then truncates to ``max_len``
    characters and appends an ellipsis suffix if the text was longer.

    Parameters
    ----------
    body : str
        The raw article body text (Markdown/MDX content).
    max_len : int, optional
        Maximum number of characters before truncation.  Defaults to
        ``BODY_PREVIEW_MAX_LENGTH`` (120).

    Returns
    -------
    str
        The full plain text if its length is at most ``max_len``, otherwise
        the first ``max_len`` characters followed by ``BODY_PREVIEW_SUFFIX``
        (space plus three dots).

    Example
    -------
    >>> extract_body_preview("Short text.")
    'Short text.'
    >>> extract_body_preview("A" * 200) == "A" * 120 + " ..."
    True
    """
    text = plain_text(body)
    if len(text) > max_len:
        return text[:max_len] + BODY_PREVIEW_SUFFIX
    return text


# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------

def tag_slug(tag_name: str) -> str:
    """
    Convert a human-readable tag name to a URL-safe slug.

    The slug is used as the filename for tag pages and in URL paths.
    For example, "Environment Variables" becomes "environment-variables"
    and "Authentication & Access" becomes "authentication-access".

    This algorithm matches the slug used for existing support tag URLs in
    the repo so filenames and links stay stable.

    Parameters
    ----------
    tag_name : str
        The human-readable tag name (for example "Run Crashes").

    Returns
    -------
    str
        A lowercase, hyphen-separated slug with no leading or trailing
        hyphens (for example "run-crashes").

    Example
    -------
    >>> tag_slug("Environment Variables")
    'environment-variables'
    >>> tag_slug("Authentication & Access")
    'authentication-access'
    >>> tag_slug("AWS")
    'aws'
    """
    # Replace any run of non-alphanumeric characters with a single hyphen,
    # then strip leading/trailing hyphens.
    return re.sub(r"[^a-z0-9]+", "-", tag_name.lower()).strip("-")


# ---------------------------------------------------------------------------
# Tab-page Badges (support/<product>/tags/...) in article MDX
# ---------------------------------------------------------------------------

def _split_frontmatter_raw(text: str) -> Tuple[str, str]:
    """
    Split raw MDX into the front matter block (including delimiters) and
    the remainder (article body plus optional auto-managed footer).

    Raises
    ------
    ValueError
        If the file does not start with ``---`` or the closing front matter
        delimiter is missing.
    """
    if not text.startswith("---"):
        raise ValueError("File does not start with '---' (no valid front matter).")

    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("Malformed front matter (missing closing '---').")

    fm_block = "---" + parts[1] + "---"
    body_and_footer = parts[2]
    return fm_block, body_and_footer



def _normalize_keywords(raw: Any, article_path: Path) -> List[str]:
    """
    Normalize the ``keywords`` front matter value to a list of strings.

    YAML mistakes such as ``keywords: "Alpha"`` (a string) would otherwise
    make ``for kw in keywords`` iterate characters and corrupt tag indexing.
    A single non-empty string is coerced to a one-item list after a warning.
    Lists are accepted; each element is coerced with ``str()`` (``None``
    entries are skipped). Any other type emits a warning and yields an
    empty list.

    Parameters
    ----------
    raw : any
        Parsed ``keywords`` value from front matter, or ``None`` if missing.
    article_path : Path
        Article path (for warning messages only).

    Returns
    -------
    list of str
        Tag names for footers, crawling, and tag index building.
    """
    if raw is None:
        return []

    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            warnings.warn(
                f"Expected 'keywords' to be a YAML list in {article_path}; "
                f"got an empty string. Treating as no keywords."
            )
            return []
        warnings.warn(
            f"Expected 'keywords' to be a YAML list in {article_path}; "
            f"got a string {raw!r}. Treating it as a single tag."
        )
        return [stripped]

    if isinstance(raw, list):
        out: List[str] = []
        for k in raw:
            if k is None:
                continue
            s = str(k).strip()
            if not s:
                warnings.warn(
                    f"Ignoring empty keyword entry in 'keywords' list "
                    f"in {article_path}."
                )
                continue
            out.append(s)
        return out

    warnings.warn(
        f"Expected 'keywords' to be a YAML list in {article_path}; "
        f"got {type(raw).__name__}. Treating as no keywords."
    )
    return []


def _keywords_list_for_footer(
    frontmatter: Dict[str, Any],
    article_path: Path,
) -> List[str]:
    """
    Return keyword strings from front matter for footer generation.

    Uses ``_normalize_keywords`` so values match crawling and tag indexing
    (see ``crawl_articles``).
    """
    return _normalize_keywords(frontmatter.get("keywords", []), article_path)


_BADGE_START = "{/* ---- AUTO-GENERATED: tab badges ----\n  Managed by scripts/knowledgebase-nav/generate_tags.py from keywords in front matter.\n  Do not edit between these markers by hand.\n---- */}"
_BADGE_END = "{/* ---- END AUTO-GENERATED: tab badges ---- */}"


def _tab_badge_pattern(product_slug: str) -> re.Pattern[str]:
    """
    Match a Mintlify ``<Badge>`` whose inner Markdown link targets this
    product's tag pages: ``/support/<slug>/tags/<anything>``.
    """
    return re.compile(
        rf'<Badge\b[^>]*>\[[^\]]*\]\(/support/{re.escape(product_slug)}/tags/[^)]+\)</Badge>'
    )


def build_tab_badges_mdx(product_slug: str, keywords: List[str]) -> str:
    """
    Build concatenated ``<Badge>`` elements for tag navigation (no ``---``).

    One Badge per keyword, in list order, linking to
    ``/support/<product>/tags/<tag-slug>``.
    """
    if not keywords:
        return ""
    return "".join(
        '<Badge stroke shape="pill" color="orange" size="md">'
        f"[{kw}](/support/{product_slug}/tags/{tag_slug(kw)})"
        "</Badge>"
        for kw in keywords
    )


def build_keyword_footer_mdx(product_slug: str, keywords: List[str]) -> str:
    """
    Build text to append when an article has no tab Badges yet.

    Wraps the Badges in marker comments so future runs can find and
    replace them without regex matching.  Returns empty when
    ``keywords`` is empty.
    """
    if not keywords:
        return ""
    badges = build_tab_badges_mdx(product_slug, keywords)
    return f"\n\n{_BADGE_START}\n{badges}\n{_BADGE_END}"


def _replace_tab_badges_in_body(
    body_and_footer: str,
    product_slug: str,
    keywords: List[str],
) -> str:
    """
    Replace or remove only tab-page Badges; append a default footer if needed.

    Prefers marker comments (``_BADGE_START`` / ``_BADGE_END``) when present:
    everything between the markers (inclusive) is replaced with the new
    marked block, or removed when ``keywords`` is empty.

    Falls back to regex matching for articles that have not yet been
    migrated to markers: finds ``<Badge>`` elements whose link path is
    ``/support/<product_slug>/tags/...``, removes them, and inserts the
    new marked block at the position of the first removed match.

    If neither markers nor matching Badges exist and ``keywords`` is
    non-empty, appends ``build_keyword_footer_mdx`` (blank line plus
    marked Badges) to the trimmed body.
    """
    start_idx = body_and_footer.find(_BADGE_START)
    end_idx = body_and_footer.find(_BADGE_END)

    if start_idx != -1 and end_idx != -1:
        end_idx += len(_BADGE_END)
        if keywords:
            new_block = f"{_BADGE_START}\n{build_tab_badges_mdx(product_slug, keywords)}\n{_BADGE_END}"
        else:
            new_block = ""
        return body_and_footer[:start_idx] + new_block + body_and_footer[end_idx:]

    # Migration path: find bare tab Badges and wrap them in markers.
    pattern = _tab_badge_pattern(product_slug)
    matches = list(pattern.finditer(body_and_footer))

    if matches:
        out = body_and_footer
        for m in reversed(matches):
            out = out[: m.start()] + out[m.end() :]
        insert_pos = matches[0].start()
        if keywords:
            new_block = f"{_BADGE_START}\n{build_tab_badges_mdx(product_slug, keywords)}\n{_BADGE_END}"
        else:
            new_block = ""
        return out[:insert_pos] + new_block + out[insert_pos:]

    if keywords:
        return body_and_footer.rstrip() + build_keyword_footer_mdx(product_slug, keywords)
    return body_and_footer


def sync_support_article_footer(article_path: Path, product_slug: str) -> bool:
    """
    Align tab-page ``<Badge>`` links with ``keywords`` in front matter.

    Only Badges whose Markdown link path is ``/support/<product>/tags/...``
    are updated or removed. Other Badges, any ``---`` line you added, and any
    text after the tab Badges are not rewritten. If the article has no such
    Badges and ``keywords`` is non-empty, appends a blank line and tab Badges
    at the end of the body (no ``---`` is inserted).

    Returns
    -------
    bool
        True if the file was written, False if it was already up to date.

    Raises
    ------
    ValueError
        If the file is missing valid opening or closing front matter delimiters
        (from ``_split_frontmatter_raw``).
    """
    text = article_path.read_text(encoding="utf-8")
    fm_block, body_and_footer = _split_frontmatter_raw(text)
    # YAML between the first two "---" delimiters (same first split as parse_frontmatter).
    inner_yaml = text.split("---", 2)[1]
    frontmatter = yaml.safe_load(inner_yaml) or {}
    keywords = _keywords_list_for_footer(frontmatter, article_path)

    new_body = _replace_tab_badges_in_body(body_and_footer, product_slug, keywords)
    new_text = fm_block + new_body

    if new_text == text:
        return False

    article_path.write_text(new_text, encoding="utf-8")
    return True


def sync_all_support_article_footers(repo_root: Path, product_slug: str) -> int:
    """
    Run ``sync_support_article_footer`` on every ``*.mdx`` under
    ``support/<product_slug>/articles/``.

    Returns
    -------
    int
        Number of files that were modified.

    Malformed files are skipped with a warning rather than failing the run.
    """
    articles_dir = repo_root / "support" / product_slug / "articles"
    if not articles_dir.exists():
        return 0

    updated = 0
    for mdx_path in sorted(articles_dir.glob("*.mdx")):
        try:
            if sync_support_article_footer(mdx_path, product_slug):
                updated += 1
        except ValueError as exc:
            warnings.warn(f"Skipping footer sync for {mdx_path}: {exc}")

    return updated


# ---------------------------------------------------------------------------
# Article crawling
# ---------------------------------------------------------------------------

def crawl_articles(repo_root: Path, product_slug: str) -> List[Dict[str, Any]]:
    """
    Crawl all MDX article files for a product and return structured data.

    Reads every ``.mdx`` file in ``support/<product_slug>/articles/``,
    parses its front matter and body, and returns a list of article
    dictionaries ready for use in template rendering and tag indexing.

    Parameters
    ----------
    repo_root : Path
        Path to the root of the wandb-docs repository.
    product_slug : str
        The product identifier (for example "models", "weave", or "inference").

    Returns
    -------
    list of dict
        Each dictionary contains:
        - ``title`` (str): The article title from front matter.
        - ``title_attr`` (str): Title safe for use in HTML attributes
          (double quotes escaped as ``&quot;``).
        - ``keywords`` (list of str): Tag names after
          ``_normalize_keywords`` (see that function for string and type
          coercion).
        - ``featured`` (bool): Whether the article has ``featured: true``.
        - ``body_preview`` (str): Truncated plain-text preview of the body.
        - ``page_path`` (str): The URL path without leading slash
          (for example ``support/models/articles/my-article``).
        - ``mdx_path`` (str): Repo-relative path to the MDX file using forward
          slashes (for example ``support/models/articles/my-article.mdx``).
        - ``file_stem`` (str): The filename without extension
          (for example ``my-article``).
        - ``tag_links`` (list of dict): Badge link data for each keyword,
          with ``name`` and ``href`` keys.

    Example
    -------
    >>> articles = crawl_articles(Path("/repo"), "models")
    >>> articles[0]["title"]
    'Can I run wandb offline?'
    >>> articles[0]["keywords"]
    ['Experiments', 'Environment Variables']
    """
    articles_dir = repo_root / "support" / product_slug / "articles"
    if not articles_dir.exists():
        warnings.warn(f"Articles directory not found: {articles_dir}")
        return []

    articles = []
    for mdx_file in sorted(articles_dir.glob("*.mdx")):
        try:
            frontmatter, body = parse_frontmatter(mdx_file)
        except ValueError as exc:
            warnings.warn(f"Skipping {mdx_file}: {exc}")
            continue

        title = frontmatter.get("title", "")
        raw_keywords = frontmatter.get("keywords", [])
        if raw_keywords is None:
            raw_keywords = []
        keywords = _normalize_keywords(raw_keywords, mdx_file)
        featured = frontmatter.get("featured", False)
        body_preview = extract_body_preview(body)
        file_stem = mdx_file.stem

        # Build Badge link data for each keyword so templates can render
        # tag links on featured article cards.
        article_tag_links = [
            {
                "name": kw,
                "href": f"/support/{product_slug}/tags/{tag_slug(kw)}",
            }
            for kw in keywords
        ]

        mdx_rel = mdx_file.relative_to(repo_root).as_posix()

        articles.append({
            "title": title,
            "title_attr": title.replace('"', "&quot;"),
            "keywords": keywords,
            "featured": bool(featured),
            "body_preview": body_preview,
            "page_path": f"support/{product_slug}/articles/{file_stem}",
            "mdx_path": mdx_rel,
            "file_stem": file_stem,
            "tag_links": article_tag_links,
        })

    return articles


# ---------------------------------------------------------------------------
# Featured article filtering
# ---------------------------------------------------------------------------

def get_featured_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter and return only articles that have ``featured: true``.

    Results are sorted alphabetically by title to ensure consistent
    ordering in the product index page.

    Parameters
    ----------
    articles : list of dict
        The full list of article dictionaries for a product (as returned
        by ``crawl_articles``).

    Returns
    -------
    list of dict
        The subset of articles where ``featured`` is True, sorted by title.

    Example
    -------
    >>> articles = [
    ...     {"title": "B article", "featured": True},
    ...     {"title": "A article", "featured": True},
    ...     {"title": "C article", "featured": False},
    ... ]
    >>> [a["title"] for a in get_featured_articles(articles)]
    ['A article', 'B article']
    """
    return sorted(
        [a for a in articles if a.get("featured")],
        key=lambda a: a.get("title", ""),
    )


# ---------------------------------------------------------------------------
# Tag index building
# ---------------------------------------------------------------------------

def build_tag_index(
    articles: List[Dict[str, Any]],
    allowed_keywords: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a mapping from tag names to lists of articles that use that tag.

    Iterates over all articles and groups them by their ``keywords``.
    If an article uses a keyword not in the ``allowed_keywords`` list,
    a warning is emitted but the tag page is still generated (to avoid
    silently dropping content).

    Parameters
    ----------
    articles : list of dict
        The article dictionaries for a single product.
    allowed_keywords : list of str
        The keywords recognized for this product (from config.yaml).

    Returns
    -------
    dict of str to list of dict
        Keys are tag names, values are lists of article dicts that
        have that tag in their ``keywords``.  Articles within each
        list are sorted alphabetically by title.

    Example
    -------
    >>> articles = [
    ...     {"title": "A", "keywords": ["Security"]},
    ...     {"title": "B", "keywords": ["Security", "Billing"]},
    ... ]
    >>> index = build_tag_index(articles, ["Security", "Billing"])
    >>> [a["title"] for a in index["Security"]]
    ['A', 'B']
    """
    allowed_set = set(allowed_keywords)
    warned_keywords = set()

    tag_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for article in articles:
        for keyword in article.get("keywords", []):
            # Warn on unknown keywords, but only once per keyword
            if keyword not in allowed_set and keyword not in warned_keywords:
                source = article.get("mdx_path") or article.get("title", "?")
                warnings.warn(
                    f"Unknown keyword '{keyword}' used in '{source}'. "
                    f"Add it to config.yaml to suppress this warning."
                )
                warned_keywords.add(keyword)

            tag_index[keyword].append(article)

    # Sort articles within each tag by title for stable, reviewable diffs.
    for keyword in tag_index:
        tag_index[keyword].sort(key=lambda a: a.get("title", ""))

    return dict(tag_index)


# ---------------------------------------------------------------------------
# Template rendering setup
# ---------------------------------------------------------------------------

def tojson_unicode(value: Any) -> str:
    """
    Serialize a value to JSON with non-ASCII characters preserved.

    Registered in ``create_template_env`` as the Jinja2 filter
    ``tojson_unicode``.  Jinja's built-in ``tojson`` filter escapes ``&``,
    ``<``, and ``>`` as Unicode escape sequences (for example ``\\u0026``)
    for HTML safety; MDX front matter in this project uses this filter so
    those characters stay literal in the output.

    Parameters
    ----------
    value : any
        The value to serialize (typically a string or number).

    Returns
    -------
    str
        A JSON string with non-ASCII characters preserved, not escaped.

    Example
    -------
    >>> tojson_unicode("Authentication & Access")
    '"Authentication & Access"'
    """
    return json.dumps(value, ensure_ascii=False)


def create_template_env(templates_dir: Path) -> Environment:
    """
    Create a Jinja2 template environment configured for MDX generation.

    Enables ``trim_blocks`` and ``lstrip_blocks`` so Jinja block tags
    (``{% %}`` and ``{# #}``) do not add extra blank lines in the MDX output.

    Registers the ``tojson_unicode`` filter so YAML front matter serializes
    with ``ensure_ascii=False`` and characters like ``&`` stay literal.

    Parameters
    ----------
    templates_dir : Path
        Path to the directory containing ``.j2`` template files.

    Returns
    -------
    jinja2.Environment
        A configured Jinja2 environment ready to render templates.

    Example
    -------
    >>> env = create_template_env(Path("templates"))
    >>> template = env.get_template("support_tag.mdx.j2")
    """
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )

    env.filters["tojson_unicode"] = tojson_unicode

    return env


# ---------------------------------------------------------------------------
# Phase 3: Tag page generation (pipeline phase; see module docstring)
# ---------------------------------------------------------------------------

def render_tag_pages(
    repo_root: Path,
    product_slug: str,
    tag_index: Dict[str, List[Dict[str, Any]]],
    template_env: Environment,
) -> List[str]:
    """
    Generate tag page MDX files for a product.

    For each tag in the tag index, renders a tag page using the
    ``support_tag.mdx.j2`` template and writes it to
    ``support/<product>/tags/<tag-slug>.mdx``.

    Each tag page contains a list of Mintlify Card components linking
    to the articles tagged with that keyword.

    Parameters
    ----------
    repo_root : Path
        Path to the root of the wandb-docs repository.
    product_slug : str
        The product identifier (for example "models").
    tag_index : dict
        Mapping of tag names to article lists (from ``build_tag_index``).
    template_env : jinja2.Environment
        The configured Jinja2 environment.

    Returns
    -------
    list of str
        The docs.json page paths for the generated tag pages
        (for example ``["support/models/tags/experiments", ...]``), sorted
        alphabetically.

    Side effects
    ------------
    Writes MDX files to ``support/<product>/tags/``.
    """
    template = template_env.get_template("support_tag.mdx.j2")
    tags_dir = repo_root / "support" / product_slug / "tags"
    tags_dir.mkdir(parents=True, exist_ok=True)

    generated_page_paths = []

    for tag_name, articles in sorted(tag_index.items()):
        slug = tag_slug(tag_name)
        output_path = tags_dir / f"{slug}.mdx"

        # Build the template context.  article.title_attr has quotes
        # escaped for safe use inside HTML attribute values.
        articles_ctx = [
            {
                "title_attr": a["title_attr"],
                "body_preview": a["body_preview"],
                "page_path": a["page_path"],
            }
            for a in articles
        ]

        content = template.render(tag=tag_name, articles=articles_ctx)
        output_path.write_text(content, encoding="utf-8")

        page_path = f"support/{product_slug}/tags/{slug}"
        generated_page_paths.append(page_path)

    return sorted(generated_page_paths)


def cleanup_stale_tag_pages(
    repo_root: Path,
    product_slug: str,
    generated_page_paths: List[str],
) -> List[str]:
    """
    Delete tag page MDX files that no longer correspond to any article keyword.

    Compares the set of files on disk in ``support/<product>/tags/`` with
    the set just generated by ``render_tag_pages``.  Any ``.mdx`` file
    present on disk but absent from ``generated_page_paths`` is removed.

    Parameters
    ----------
    repo_root : Path
        Path to the root of the wandb-docs repository.
    product_slug : str
        The product identifier (for example "models").
    generated_page_paths : list of str
        Page paths returned by ``render_tag_pages`` (for example
        ``["support/models/tags/experiments", ...]``).

    Returns
    -------
    list of str
        Filenames (stems) of the deleted tag pages, sorted alphabetically.
    """
    tags_dir = repo_root / "support" / product_slug / "tags"
    if not tags_dir.exists():
        return []

    generated_stems = {p.rsplit("/", 1)[-1] for p in generated_page_paths}
    removed: List[str] = []

    for mdx_file in sorted(tags_dir.glob("*.mdx")):
        if mdx_file.stem not in generated_stems:
            mdx_file.unlink()
            removed.append(mdx_file.stem)

    return removed


# ---------------------------------------------------------------------------
# Phase 4: Product index page generation
# ---------------------------------------------------------------------------

def render_product_index(
    repo_root: Path,
    product_slug: str,
    product_display_name: str,
    tag_index: Dict[str, List[Dict[str, Any]]],
    featured_articles: List[Dict[str, Any]],
    template_env: Environment,
) -> None:
    """
    Generate the product index MDX page for a product.

    Renders ``support/<product>.mdx`` with two sections:

    1. **Featured articles.** Shown only if there are articles with
       ``featured: true``.  Each featured article is rendered as a
       horizontal Card with a body preview and Badge tag links.

    2. **Browse by category.** Lists all tags alphabetically with
       article counts, rendered as Cards linking to tag pages.

    Parameters
    ----------
    repo_root : Path
        Path to the root of the wandb-docs repository.
    product_slug : str
        The product identifier (for example "models").
    product_display_name : str
        The human-readable product name (for example "W&B Models").
    tag_index : dict
        Mapping of tag names to article lists.
    featured_articles : list of dict
        Articles with ``featured: true``, sorted by title.
    template_env : jinja2.Environment
        The configured Jinja2 environment.

    Side effects
    ------------
    Writes the product index MDX file to ``support/<product>.mdx``.
    """
    template = template_env.get_template("support_product_index.mdx.j2")

    # Build the tag categories list: one entry per tag, sorted
    # alphabetically by tag name.
    tag_categories = sorted(
        [
            {
                "name": tag_name,
                "href": f"/support/{product_slug}/tags/{tag_slug(tag_name)}",
                "count": len(articles),
            }
            for tag_name, articles in tag_index.items()
        ],
        key=lambda t: t["name"],
    )

    content = template.render(
        product_name=product_display_name,
        featured_articles=featured_articles,
        tag_categories=tag_categories,
    )

    output_path = repo_root / "support" / f"{product_slug}.mdx"
    output_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 5: docs.json navigation update
# ---------------------------------------------------------------------------

def update_docs_json(
    repo_root: Path,
    products: List[Dict[str, Any]],
    all_tag_page_paths: Dict[str, List[str]],
) -> None:
    """
    Update the docs.json navigation with current support tag pages.

    Reads the existing docs.json, finds the English ("en") language
    navigation root, and for each product either updates the existing
    hidden support tab or creates a new one.  Each tab's page list is
    set to the product overview page followed by sorted tag page paths.

    All non-support navigation entries are preserved untouched.

    Parameters
    ----------
    repo_root : Path
        Path to the root of the wandb-docs repository.
    products : list of dict
        Product definitions from config.yaml, each with ``slug`` and
        ``display_name`` keys.
    all_tag_page_paths : dict
        Mapping of product slug to sorted list of tag page paths
        (as returned by ``render_tag_pages``).

    Side effects
    ------------
    Overwrites docs.json with updated navigation.

    Raises
    ------
    FileNotFoundError
        If docs.json does not exist at the repo root.
    ValueError
        If the English language navigation entry is not found.

    Example
    -------
    The resulting tab structure for each product looks like::

        {
            "tab": "Support: W&B Models",
            "hidden": true,
            "pages": [
                "support/models",
                "support/models/tags/academic",
                "support/models/tags/administrator",
                ...
            ]
        }
    """
    docs_json_path = repo_root / "docs.json"
    if not docs_json_path.exists():
        raise FileNotFoundError(
            f"docs.json not found at {docs_json_path}. "
            "Ensure this script is run from the wandb-docs repo root."
        )

    with open(docs_json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Navigate to the English language navigation root.
    # The W&B docs site uses navigation.languages[] for multi-language
    # support; we only modify the English entry.
    nav = config.get("navigation", {})
    languages = nav.get("languages", [])
    en_root = None
    for lang_entry in languages:
        if isinstance(lang_entry, dict) and lang_entry.get("language") == DOCS_JSON_NAV_LANGUAGE:
            en_root = lang_entry
            break

    if en_root is None:
        raise ValueError(
            f"docs.json navigation.languages[] has no entry with "
            f"language='{DOCS_JSON_NAV_LANGUAGE}'. Cannot update navigation."
        )

    tabs = en_root.setdefault("tabs", [])

    for product in products:
        slug = product["slug"]
        display_name = product["display_name"]
        tab_name = f"Support: {display_name}"

        # The overview page is always first in the tab's page list
        overview_page = f"support/{slug}"
        tag_pages = all_tag_page_paths.get(slug, [])
        new_pages = [overview_page] + tag_pages

        # Find the existing tab or create a new one
        target_tab = None
        for tab in tabs:
            if isinstance(tab, dict) and tab.get("tab") == tab_name:
                target_tab = tab
                break

        if target_tab is None:
            # Create a new hidden tab and append it
            target_tab = {"tab": tab_name, "hidden": True, "pages": new_pages}
            tabs.append(target_tab)
        else:
            # Update existing tab's pages; preserve hidden flag
            target_tab["pages"] = new_pages

    # Write the updated docs.json back.
    # Use indent=2 and a trailing newline to match the existing format.
    with open(docs_json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Phase 6: Update support.mdx (counts and featured articles)
# ---------------------------------------------------------------------------

_FEATURED_START = "{/* ---- AUTO-GENERATED: featured articles ----"
_FEATURED_END = "{/* ---- END AUTO-GENERATED: featured articles ---- */}"


def _build_featured_section_mdx(
    all_featured: Dict[str, Tuple[str, List[Dict[str, Any]]]],
) -> str:
    """
    Build the auto-managed featured-articles block for support.mdx.

    Parameters
    ----------
    all_featured : dict
        Mapping of product slug to a ``(display_name, featured_articles)``
        tuple.  Products with no featured articles are skipped.

    Returns
    -------
    str
        The MDX content between (but not including) the start and end
        markers.  Includes the hint comment for tech writers.
    """
    hint = (
        "{/* ---- AUTO-GENERATED: featured articles ----\n"
        "  This section is managed by scripts/knowledgebase-nav/generate_tags.py.\n"
        "  To feature an article, add \"featured: true\" to its front matter.\n"
        "  To remove it, set \"featured: false\" or remove the field.\n"
        "  Do not edit the content between these markers by hand.\n"
        "---- */}"
    )
    parts = [hint]

    if all_featured:
        parts.append("\n## Featured articles")

    for slug in sorted(all_featured):
        display_name, articles = all_featured[slug]
        if not articles:
            continue
        parts.append(f"\n### {display_name}\n")
        for article in articles:
            badges = " ".join(
                f'<Badge stroke shape="pill" color="orange" size="md">'
                f'[{tl["name"]}]({tl["href"]})</Badge>'
                for tl in article["tag_links"]
            )
            parts.append(
                f'<Card title="{article["title_attr"]}" '
                f'href="/{article["page_path"]}" arrow="true" horizontal>\n'
                f"  {article['body_preview']}\n"
                f"\n"
                f"{badges} \n"
                f"</Card>"
            )

    parts.append("")
    parts.append(_FEATURED_END)
    return "\n".join(parts)


def update_support_featured(
    repo_root: Path,
    all_featured: Dict[str, Tuple[str, List[Dict[str, Any]]]],
) -> None:
    """
    Replace the featured-articles section of support.mdx between markers.

    Looks for ``_FEATURED_START`` and ``_FEATURED_END`` in support.mdx
    and replaces everything between them (inclusive) with a freshly
    generated block.  If the markers are missing, emits a warning and
    leaves the file unchanged.

    Parameters
    ----------
    repo_root : Path
        Path to the root of the wandb-docs repository.
    all_featured : dict
        Mapping of product slug to ``(display_name, featured_articles)``.
    """
    support_path = repo_root / "support.mdx"
    content = support_path.read_text(encoding="utf-8")

    start_idx = content.find(_FEATURED_START)
    end_idx = content.find(_FEATURED_END)

    if start_idx == -1 or end_idx == -1:
        warnings.warn(
            "Could not find featured-article markers in support.mdx. "
            "Skipping featured section update."
        )
        return

    end_idx += len(_FEATURED_END)
    new_block = _build_featured_section_mdx(all_featured)
    new_content = content[:start_idx] + new_block + content[end_idx:]

    if new_content != content:
        support_path.write_text(new_content, encoding="utf-8")


def update_support_index(
    repo_root: Path,
    product_stats: Dict[str, Dict[str, int]],
) -> None:
    """
    Update the article and tag counts in the top-level support.mdx page.

    The support.mdx page is human-authored and contains product Cards
    with article and tag counts in the format::

        <Card title="W&B Models" href="/support/models" ...>
          180 articles &middot; 33 tags
        </Card>

    This function reads support.mdx, finds each product Card by its
    ``href="/support/<slug>"`` attribute, and replaces the count line
    with the current numbers.  All other content is left untouched.

    Parameters
    ----------
    repo_root : Path
        Path to the root of the wandb-docs repository.
    product_stats : dict
        Mapping of product slug to a dict with ``article_count`` and
        ``tag_count`` keys.  For example::

            {
                "models": {"article_count": 180, "tag_count": 33},
                "weave":  {"article_count": 15,  "tag_count": 8},
            }

    Side effects
    ------------
    Overwrites support.mdx in place with updated counts.

    Raises
    ------
    FileNotFoundError
        If support.mdx does not exist at the repo root.

    Example
    -------
    Given a Card like::

        <Card title="W&B Models" href="/support/models" arrow="true" ...>
          180 articles &middot; 33 tags
        </Card>

    After calling with ``{"models": {"article_count": 185, "tag_count": 34}}``,
    the line becomes::

          185 articles &middot; 34 tags
    """
    support_index_path = repo_root / "support.mdx"
    if not support_index_path.exists():
        raise FileNotFoundError(
            f"support.mdx not found at {support_index_path}. "
            "Ensure this script is run from the wandb-docs repo root."
        )

    content = support_index_path.read_text(encoding="utf-8")

    for slug, stats in product_stats.items():
        article_count = stats["article_count"]
        tag_count = stats["tag_count"]

        # Pluralize: "1 article" vs "2 articles", "1 tag" vs "2 tags"
        article_word = "article" if article_count == 1 else "articles"
        tag_word = "tag" if tag_count == 1 else "tags"

        new_count_line = f"  {article_count} {article_word} &middot; {tag_count} {tag_word}"

        _COUNTS_OPEN = r"[ \t]+\{/\* auto-generated counts \*/\}\n"
        _COUNTS_CLOSE = r"\n[ \t]+\{/\* end auto-generated counts \*/\}"

        # Prefer the marker-wrapped format; fall back to bare count line
        # for migration.
        pattern = (
            r'(<Card[^>]*href="/support/' + re.escape(slug) + r'"[^>]*>\n)'
            + _COUNTS_OPEN
            + r'[ \t]+\d+ articles? &middot; \d+ tags?'
            + _COUNTS_CLOSE
        )
        replacement = (
            r'\g<1>'
            + "  {/* auto-generated counts */}\n"
            + new_count_line
            + "\n  {/* end auto-generated counts */}"
        )

        new_content, count = re.subn(pattern, replacement, content)
        if count == 0:
            # Migration: bare count line without markers.
            pattern = (
                r'(<Card[^>]*href="/support/' + re.escape(slug) + r'"[^>]*>\n)'
                r'[ \t]+\d+ articles? &middot; \d+ tags?'
            )
            replacement = (
                r'\g<1>'
                + "  {/* auto-generated counts */}\n"
                + new_count_line
                + "\n  {/* end auto-generated counts */}"
            )
            new_content, count = re.subn(pattern, replacement, content)

        content = new_content

        if count == 0:
            warnings.warn(
                f"Could not find product card for '{slug}' in support.mdx. "
                f"Expected a Card with href=\"/support/{slug}\" followed by "
                f"a line with article/tag counts."
            )

    support_index_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(repo_root: Path, config_path: Path) -> None:
    """
    Execute the full knowledgebase nav generation pipeline.

    Orchestrates all phases: crawl articles, generate tag pages, generate
    product index pages, sync tab-page Badges on articles, update docs.json
    navigation, and update support.mdx product card counts and featured
    articles.

    Parameters
    ----------
    repo_root : Path
        Path to the root of the wandb-docs repository.
    config_path : Path
        Path to the config.yaml file.

    Side effects
    ------------
    - May rewrite support/<product>/articles/*.mdx tab-page ``<Badge>`` links
      (see ``sync_support_article_footer``)
    - Writes tag page MDX files under support/<product>/tags/ and deletes
      stale ones that no longer correspond to any article keyword
    - Writes product index MDX files at support/<product>.mdx
    - Overwrites docs.json with updated navigation
    - Updates support.mdx with current article/tag counts and featured articles
    - Prints summary information to stdout
    - Emits warnings for unknown keywords, invalid ``keywords`` types, skipped
      articles, missing ``support.mdx`` cards, and footer sync failures
    """
    config = load_config(config_path)
    products = config["products"]

    # Set up Jinja2 templates from the templates/ directory next to this script
    script_dir = Path(__file__).resolve().parent
    templates_dir = script_dir / "templates"
    template_env = create_template_env(templates_dir)

    # Track tag page paths per product for the docs.json update step
    all_tag_page_paths: Dict[str, List[str]] = {}

    # Track article and tag counts per product for the support.mdx update step
    product_stats: Dict[str, Dict[str, int]] = {}

    # Track featured articles per product for the support.mdx featured section
    all_featured: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}

    print("=" * 60)
    print("Knowledgebase Nav Generator")
    print("=" * 60)

    for product in products:
        slug = product["slug"]
        display_name = product["display_name"]
        allowed_keywords = product.get("allowed_keywords", [])

        print(f"\n--- {display_name} ({slug}) ---")

        # Phase 1: Crawl and parse articles
        articles = crawl_articles(repo_root, slug)
        print(f"  Found {len(articles)} articles")

        # Phase 1b: Build the tag index
        tag_index = build_tag_index(articles, allowed_keywords)
        print(f"  Found {len(tag_index)} unique tags")

        # Phase 2: Generate tag pages and remove stale ones
        tag_page_paths = render_tag_pages(repo_root, slug, tag_index, template_env)
        all_tag_page_paths[slug] = tag_page_paths
        print(f"  Generated {len(tag_page_paths)} tag pages")

        removed = cleanup_stale_tag_pages(repo_root, slug, tag_page_paths)
        if removed:
            print(f"  Removed {len(removed)} stale tag page(s): {', '.join(removed)}")

        # Phase 3: Generate product index page
        featured = get_featured_articles(articles)
        render_product_index(
            repo_root, slug, display_name, tag_index, featured, template_env
        )
        print(f"  Generated product index (featured: {len(featured)} articles)")

        if featured:
            all_featured[slug] = (display_name, featured)

        # Phase 4: Sync tab-page Badges from front matter keywords
        # Runs after tag pages exist so articles are not updated if
        # earlier phases fail.
        footer_updates = sync_all_support_article_footers(repo_root, slug)
        if footer_updates:
            print(f"  Updated tab Badges on {footer_updates} article(s)")

        # Record stats for Phase 6
        product_stats[slug] = {
            "article_count": len(articles),
            "tag_count": len(tag_index),
        }

    # Phase 5: Update docs.json navigation
    update_docs_json(repo_root, products, all_tag_page_paths)
    print(f"\n--- docs.json ---")
    print(f"  Updated navigation for {len(products)} products")

    # Phase 6: Update support.mdx product card counts and featured articles
    update_support_index(repo_root, product_stats)
    update_support_featured(repo_root, all_featured)
    print(f"\n--- support.mdx ---")
    print(f"  Updated product card counts and featured articles")
    print(f"\n{'=' * 60}")
    print("Done.")


def main() -> None:
    """
    CLI entry point for the knowledgebase nav generator.

    Parses command-line arguments and runs the pipeline.  Intended to be
    called from a GitHub Actions workflow or from the command line for
    local testing.

    Usage::

        python generate_tags.py --repo-root /path/to/wandb-docs

    If ``--config`` is not specified, the script looks for ``config.yaml``
    in the same directory as this script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sync support article keyword footers, regenerate tag and index pages, "
            "and update docs.json and support.mdx."
        ),
        epilog="See README.md for detailed usage instructions.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Path to the root of the wandb-docs repository (contains docs.json).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml. Defaults to config.yaml in the script directory.",
    )

    args = parser.parse_args()

    # Default config path: same directory as this script
    if args.config is None:
        args.config = Path(__file__).resolve().parent / "config.yaml"

    try:
        run_pipeline(args.repo_root, args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
